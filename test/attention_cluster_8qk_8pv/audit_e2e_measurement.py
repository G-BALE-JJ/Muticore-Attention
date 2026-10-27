#!/usr/bin/env python3
"""Audit archived clean-guest timing without running or modifying the simulator.

PASS means the archived timing contract is consistent, not that every possible
guest interference has been eliminated. Pending isolation items stay explicit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def load(case, name):
    return json.loads((case / name).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(case):
    case = Path(case).resolve()
    provenance = load(case, 'provenance.json')
    env = provenance['environment']
    require(env['VANADIS_CPU_CLOCK'] in ('1GHz', '1.0GHz') and
            env['GOLEM_ARRAY_CLOCK'] in ('1GHz', '1.0GHz'),
            'legacy projection parser requires CPU and array clocks at 1 GHz')
    require(env.get('GOLEM_ATTENTION_NORMALIZATION_CLOCK', '1GHz') in ('1GHz', '1.0GHz'),
            'normalization clock differs from projection component cycles')
    require(env.get('GOLEM_ATTENTION_DIAGNOSTIC_MVM_DUMP', '0') == '0',
            'diagnostic MVM dump enabled')
    require(env.get('GOLEM_DIAGNOSTIC_ORACLE') == 'none', 'oracle active')
    require(provenance.get('compile_time_startup_diagnostics') == 0,
            'case is not the clean-startup measurement profile')
    for relative, expected in provenance['sha256'].items():
        archived = case / 'source_snapshot' / relative
        require(hashlib.sha256(archived.read_bytes()).hexdigest() == expected,
                f'archived source/binary differs: {relative}')
    projection = load(case, 'projection_e2e_result.json')
    experiment = load(case, 'experiment_result.json')
    resources = load(case, 'resource_analysis.json')
    numerical = load(case, 'fused_attention_result.json')
    require(all(d['status'] == 'PASS' for d in
                (projection, experiment, resources, numerical)), 'case verification failed')
    require(all(row['audit'] == 0 and row['norm_audit_bytes'] == 0 and
                row['raw_audit_bytes'] == 0
                for row in projection['fusion_by_worker'].values()),
            'modeled diagnostic output traffic exists')
    guest_text = '\n'.join(p.read_text(errors='replace')
                           for p in (case / 'stdout').rglob('*') if p.is_file())
    require(not re.search(r'\[(ATTENTION_GUEST_START|PROJECTION_START|PROJECTION_DISPATCH)\]',
                          guest_text), 'startup diagnostic ran')
    statuses = re.findall(r'FUSED_ATTENTION status=(\d+).*?manager=(\d+)', guest_text)
    require(sorted(statuses) == [('0', str(i)) for i in range(4)],
            'missing, duplicate, or failed guest final status')
    log = next((case / 'logs').glob('attention-sst*.log'))
    managers = {i: [] for i in range(4)}
    worker_end_ticks, root_end_ticks, timebases = [], [], set()
    simulation_end = None
    milestone = re.compile(r'\[ATTENTION_MILESTONE\] stage=(\w+) status=done '
                           r'sst_tick=(\d+) rocc_cycle=(\d+) core=(\d+)')
    with log.open(errors='replace') as stream:
        for line in stream:
            match = re.search(r'\[RESOURCE_TIMEBASE\].*ticks_per_cycle=(\d+)', line)
            if match:
                timebases.add(int(match[1]))
            match = milestone.search(line)
            if match:
                stage, tick, _, core = match.groups()
                tick, core = int(tick), int(core)
                if stage == 'worker_complete':
                    worker_end_ticks.append(tick)
                elif stage == 'root_tensor_complete':
                    root_end_ticks.append(tick)
                elif stage in ('manager_wait_observed', 'software_wait_observed'):
                    managers[core].append(tick)
            match = re.search(r'Simulation is complete, simulated time:\s*([\d.]+)\s*([munp]?s)', line)
            if match:
                simulation_end = float(match[1]) * {'s':1e9, 'ms':1e6,
                    'us':1e3, 'ns':1, 'ps':1e-3}[match[2]]
    require(timebases == {1000}, 'raw timebase disagrees with 1 GHz parser')
    require(worker_end_ticks and root_end_ticks and simulation_end is not None,
            'missing completion evidence')
    require(all(len(ticks) == projection['shape']['Hkv'] for ticks in managers.values()),
            'incomplete per-manager wait evidence')
    worker_end = max(worker_end_ticks) / 1000
    root_end = max(root_end_ticks) / 1000
    reported_end = projection['stages']['attention']['end_cycle']
    # Component timestamps discard fractional cycles; tolerate only that rounding.
    require(abs(worker_end - reported_end) < 1, 'reported endpoint differs from raw worker completion')
    require(root_end == resources['last_root_tensor_complete_cycle'],
            'raw root endpoint differs from resource analysis')
    start = projection['chain_start_cycle']
    require(reported_end - start == experiment['end_to_end_cycles'] ==
            projection['end_to_end_cycles'], 'E2E arithmetic mismatch')
    release = min(c['release_issue_cycle'] for c in resources['worker_control'].values())
    output = resources['last_attention_output_ack_cycle']
    require(release < start < output <= root_end <= simulation_end,
            'completion/frontier ordering invalid')
    return {
        'artifact_root': str(case), 'status': 'PASS_TIMING_CONTRACT',
        'sequence': projection['shape']['query_length'],
        'reported_job_to_worker_complete_cycles': projection['end_to_end_cycles'],
        'release_to_job_start_cycles': start - release,
        'release_to_last_output_ack_cycles': output - release,
        'release_to_root_complete_cycles': root_end - release,
        'last_output_ack_from_zero_cycles': output,
        'root_complete_from_zero_cycles': root_end,
        'root_to_simulation_exit_cycles': simulation_end - root_end,
        'manager_final_wait_from_zero_cycles': {str(i): max(t) / 1000 for i,t in managers.items()},
        'possible_final_print_overlap_cycles': {
            str(i): max(0, worker_end - max(t) / 1000) for i,t in managers.items()},
        'pending_isolation': ['non-root guest final printf and process exit may overlap remaining work',
                              'trace on/off modeled-timing equivalence has not been tested here'],
        'floor_classification': 'scoped work estimate; additive fused RMSNorm/projection is not a proven lower bound',
        'archived_sha256_verified': len(provenance['sha256']),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {'status': 'PASS_TIMING_CONTRACT_WITH_PENDING_ISOLATION',
              'cases': [audit(case) for case in args.cases]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for row in result['cases']:
        print(f"S{row['sequence']}: {row['status']}; release→root={row['release_to_root_complete_cycles']}")


if __name__ == '__main__':
    main()
