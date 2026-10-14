#!/usr/bin/env python3
"""Audit complete, matched baseline/read/group projection experiments."""
import argparse
import json
from pathlib import Path
import re

from run_projection_overlap import VARIANTS, digest


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    rows = []
    experiment_flags = {'GOLEM_PROJECTION_INPUT_PAIR_PREFETCH', 'GOLEM_PROJECTION_LOCAL_WEIGHT_PREFETCH',
                        'GOLEM_PROJECTION_ARRAY_GROUPS', 'GOLEM_PROJECTION_INPUT_REFERENCE',
                        'GOLEM_MPI_RANK_LOG_ROOT', 'GOLEM_MPI_ARGS'}
    for sequence in (1024, 2048):
        base = root / 'baseline' / f's{sequence}'
        before = read(base / 'experiment_result.json')
        base_provenance = read(base / 'baseline_provenance.json')
        base_report = read(base / 'projection_e2e_result.json')
        base_gm = base_report['stages']['projection']['local_gm_by_worker']
        shared = {k: v for k, v in base_provenance['environment'].items() if k not in experiment_flags}
        for variant in ('baseline', 'reads', 'groups'):
            case = root / variant / f's{sequence}'
            result = read(case / 'experiment_result.json')
            provenance = read(case / 'baseline_provenance.json')
            report = read(case / 'projection_e2e_result.json')
            if (result['status'] != 'PASS' or not result['byte_exact']['status'] == 'PASS' or
                    provenance['sha256'] != base_provenance['sha256'] or
                    provenance['input_sha256'] != base_provenance['input_sha256'] or
                    {k: v for k, v in provenance['environment'].items() if k not in experiment_flags} != shared):
                raise ValueError(f'unmatched or failed experiment: {case}')
            for key, expected in {'GOLEM_LOCAL_GM_BYTES_PER_CYCLE': '256',
                    'GOLEM_LOCAL_GM_BASE_LATENCY_CYCLES': '1', 'GOLEM_LOCAL_GM_READ_PORTS': '1',
                    'GOLEM_LOCAL_GM_WRITE_PORTS': '1', 'GOLEM_LOCAL_GM_MAX_REQUEST_BYTES': '4096',
                    'GOLEM_LOCAL_GM_QUEUE_DEPTH': '256'}.items():
                if provenance['environment'].get(key) != expected:
                    raise ValueError(f'unexpected GM resource: {key}')
            configs = list((case / 'stats').glob('**/run_config.env'))
            if len(configs) != 1:
                raise ValueError('missing resolved run configuration')
            resolved = dict(re.findall(r'^\s+(GOLEM_\w+)=(.*)$', configs[0].read_text(), re.MULTILINE))
            for key, expected in {'GOLEM_NUM_ARRAYS': '64', 'GOLEM_TOTAL_CORES': '20',
                    'GOLEM_GLOBAL_STRIDE_BYTES': '2097152', 'GOLEM_ARRAY_BUFFER_PORTS': '1',
                    'GOLEM_ARRAY_BUFFER_BYTES_PER_CYCLE': '64', 'GOLEM_ARRAY_BUFFER_BASE_LATENCY_CYCLES': '1',
                    'GOLEM_ARRAY_BUFFER_QUEUE_DEPTH': '64'}.items():
                if resolved.get(key) != expected:
                    raise ValueError(f'unexpected resolved hardware resource: {key}')
            pair, weight, groups = VARIANTS[variant]
            for worker, plan in result['dataflow_by_worker'].items():
                if (plan['input_pair_prefetch'], plan['local_weight_prefetch'], plan['array_groups']) != (pair, weight, groups):
                    raise ValueError('unexpected effective dataflow configuration')
                gm = report['stages']['projection']['local_gm_by_worker'][worker]
                if any(gm[k] != base_gm[worker][k] for k in ('read_bytes', 'write_bytes')):
                    raise ValueError('operand GM traffic changed')
            log_paths = list((case / 'logs').glob('attention-sst*.log'))
            if len(log_paths) != 1:
                raise ValueError('missing resource/service evidence')
            log_text = log_paths[0].read_text(errors='replace')
            jobs = [tuple(map(int, row)) for row in re.findall(
                r'\[PROJECTION_JOB\] core=(\d+) start=(\d+) end=(\d+) cycles=(\d+)', log_text)]
            if sorted(row[0] for row in jobs) != list(range(4, 20)):
                raise ValueError('incomplete projection timing evidence')
            critical = max(jobs, key=lambda row: row[2])
            buffer = {}
            for line in re.findall(r'GOLEM_ARRAY_BUFFER_STATS ([^\n]+)', log_text):
                fields = {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', line)}
                buffer[str(fields.pop('core'))] = fields
            if sorted(map(int, buffer)) != list(range(4, 20)):
                raise ValueError('incomplete array buffer evidence')
            workers = {}
            for worker in result['dataflow_by_worker']:
                stage = report['stages']['projection']
                b = buffer[worker]
                workers[worker] = {'dataflow': result['dataflow_by_worker'][worker],
                    'foreground_gm': stage['local_gm_by_worker'][worker],
                    'phase_cycles': stage['cycle_breakdown_by_worker'][worker],
                    'array_buffer_total_transfer_cycles': b['transfer_cycles'],
                    'matrix_broadcast_requests': b['matrix_broadcast_requests'],
                    'matrix_broadcast_ingress_bytes': b['matrix_broadcast_ingress_bytes'],
                    'matrix_broadcast_sink_bytes': b['matrix_broadcast_sink_bytes'],
                    'matrix_broadcast_transfer_cycles': b['matrix_broadcast_transfer_cycles'],
                    'input_scatter_transfer_cycles': b['input_scatter_transfer_cycles']}
            saved = before['end_to_end_cycles'] - result['end_to_end_cycles']
            floor = report['optimistic_resource_floor_cycles']
            rows.append({'sequence': sequence, 'variant': variant, 'status': 'PASS',
                'end_to_end_cycles': result['end_to_end_cycles'], 'projection_cycles': result['projection_cycles'],
                'attention_cycles': result['attention_cycles'], 'handoff_cycles': result['handoff_cycles'],
                'end_to_end_saved_cycles': saved, 'end_to_end_reduction_percent': 100 * saved / before['end_to_end_cycles'],
                'projection_reduction_percent': 100 * (before['projection_cycles'] - result['projection_cycles']) / before['projection_cycles'],
                'optimistic_floor_cycles': floor, 'actual_to_floor_ratio': result['end_to_end_cycles'] / floor,
                'critical_projection_worker': critical[0], 'critical_worker_job_cycles': critical[3],
                'critical_worker_start_skew_cycles': critical[1] - min(row[1] for row in jobs),
                'excess_over_compute_floor_removed_percent': 100 * saved / (before['end_to_end_cycles'] - floor),
                'byte_exact': result['byte_exact'], 'workers': workers, 'artifact_root': str(case)})
    summary = read(root / 'summary.json')
    if len(summary['cases']) != 6:
        raise ValueError('formal experiment is incomplete')
    # Recheck that the reported source and binaries still identify this build.
    source_root = Path(__file__).resolve().parents[2]
    for name, expected in base_provenance['sha256'].items():
        if digest(source_root / name) != expected:
            raise ValueError(f'source or binary changed: {name}')
    best = {str(sequence): min((row for row in rows if row['sequence'] == sequence),
            key=lambda row: row['end_to_end_cycles'])['variant'] for sequence in (1024, 2048)}
    args.output.write_text(json.dumps({'status': 'PASS', 'complete_cases': 6,
        'scope': 'RMSNorm -> QKV -> RoPE -> causal GQA -> O; excludes Wo/residual/MLP',
        'best_variant_by_sequence': best,
        'variant_flags': {name: dict(zip(('input_pair_prefetch', 'local_weight_prefetch', 'array_groups'), flags))
                          for name, flags in VARIANTS.items()},
        'same_source_binary_inputs_hardware': True, 'cases': rows}, indent=2) + '\n')
    for row in rows:
        print(f"S{row['sequence']} {row['variant']}: {row['end_to_end_cycles']} cycles, "
              f"E2E reduction {row['end_to_end_reduction_percent']:.2f}%, "
              f"projection reduction {row['projection_reduction_percent']:.2f}%")


if __name__ == '__main__':
    main()
