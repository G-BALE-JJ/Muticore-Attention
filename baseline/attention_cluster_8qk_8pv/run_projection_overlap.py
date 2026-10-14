#!/usr/bin/env python3
"""Controlled end-to-end experiments for GM and disjoint-array overlap."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from run_causal_balance import BEST
from run_llama_projection import GM_PROFILE

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VARIANTS = {
    'baseline': (0, 0, 1),
    'input': (1, 0, 1),
    'reads': (1, 1, 1),
    'groups': (1, 1, 2),
    'groups_only': (0, 0, 2),
}
SOURCES = [ROOT / 'src/sst/elements/golem/attention/projectionJob.h',
           ROOT / 'src/sst/elements/golem/tests/small/muticore_attention/report_projection_e2e.py',
           ROOT / 'src/sst/elements/golem/tests/small/muticore_attention/run_fused_attention_scale.sh',
           ROOT / 'install/lib/sst-elements-library/libgolem.so',
           ROOT / 'src/sst/elements/golem/tests/small/muticore_attention/riscv64/fused_attention']
SOURCES += [ROOT / path for path in (
    'src/sst/elements/golem/attention/resourceTrace.h',
    'src/sst/elements/golem/globalmemory/globalmemory.cc',
    'src/sst/elements/golem/globalmemory/globalmemory.h',
    'src/sst/elements/golem/array/computeArray.h',
    'src/sst/elements/golem/array/mvmComputeArray.h',
    'src/sst/elements/golem/workercmdproc/workercmdproc.h',
    'src/sst/elements/golem/rocc/roccAnalog.h',
    'src/sst/elements/golem/sfu/sfu.cc',
    'src/sst/elements/golem/sfu/sfu.h',
    'src/sst/elements/memHierarchy/membackend/ramulator2Backend.cc',
    'src/sst/elements/memHierarchy/membackend/ramulator2Backend.h',
    'install/lib/sst-elements-library/libmemHierarchy.so',
    'baseline/attention_cluster_8qk_8pv/merge_mpi_rank_logs.py',
    'baseline/attention_cluster_8qk_8pv/report_dma_e2e.py',
    'baseline/attention_cluster_8qk_8pv/analyze_dma_lower_bound.py',
    'baseline/attention_cluster_8qk_8pv/run_sst.sh',
)]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def exact_outputs(case, reference):
    for name in ('projection_x.bin', 'projection_gamma.bin', 'projection_weights.bin'):
        if digest(case / name) != digest(reference / name):
            raise ValueError(f'input mismatch: {name}')
    current = json.loads((case / 'attention_hbm_layout.json').read_text())
    previous = json.loads((reference / 'attention_hbm_layout.json').read_text())
    regions, total = 0, 0
    for node, old in zip(current['nodes'], previous['nodes'], strict=True):
        if node['node'] != old['node']:
            raise ValueError('HBM node mismatch')
        old_regions = {r['name']: r for r in old['regions']}
        filename = f"hbm_out_node{node['node']}.bin"
        with (case / 'hbm' / filename).open('rb') as actual, (reference / 'hbm' / filename).open('rb') as expected:
            for region in node['regions']:
                other = old_regions[region['name']]
                size = region['end'] - region['begin']
                if size != other['end'] - other['begin']:
                    raise ValueError('HBM region size mismatch')
                actual.seek(region['begin'])
                expected.seek(other['begin'])
                if actual.read(size) != expected.read(size):
                    raise ValueError(f"HBM mismatch: node{node['node']} {region['name']}")
                regions += 1
                total += size
    return {'status': 'PASS', 'regions': regions, 'bytes': total, 'reference': str(reference)}


def verify_group_trace(text, cores, heads, blocks, tiles):
    events = {}
    for c, block, head, group, tile, event, cycle in re.findall(
            r'\[PROJECTION_GROUP_TRACE\] core=(\d+) block=(\d+) head=(\d+) group=(\d+) tile=(\d+) event=(\w+) cycle=(\d+)', text):
        key = tuple(map(int, (c, block, head, group, tile)))
        if event in events.setdefault(key, {}):
            raise ValueError('duplicate group trace event')
        events[key][event] = int(cycle)
    if len(events) != cores * heads * blocks * tiles * 2:
        raise ValueError('incomplete group lifecycle trace')
    overlapped = 0
    for key, event in events.items():
        if set(event) != {'program', 'launch', 'done'} or not event['program'] <= event['launch'] < event['done']:
            raise ValueError('invalid group lifecycle timestamps')
    for key, event in events.items():
        c, block, head, group, tile = key
        previous = events.get((c, block, head, group, tile - 1))
        other = events.get((c, block, head, 1 - group, tile))
        if group not in (0, 1) or other is None or (tile and previous is None):
            raise ValueError('incomplete group lifecycle trace')
        if tile and event['program'] < previous['done']:
            raise ValueError('operand programming overlapped its own compute')
        overlapped += max(0, min(event['launch'], other['done']) - max(event['program'], other['launch']))
    if not overlapped:
        raise ValueError('no disjoint-group programming/compute overlap')
    return {'status': 'PASS', 'tile_group_lifecycles': len(events),
            'same_operand_conflicts': 0, 'same_tile_cross_group_overlap_cycles': overlapped}


def summarize(case, reference=None, trace=False):
    names = ('projection_e2e_result', 'fused_attention_result', 'attention_hbm_layout',
             'attention_mpi_partition', 'sst_qk_bridge_result')
    reports = {n: json.loads((case / f'{n}.json').read_text()) for n in names}
    if any(r['status'] != 'PASS' for r in reports.values()):
        raise ValueError('failed end-to-end verification')
    report = reports['projection_e2e_result']
    if not report.get('single_operand_context_valid') or not report.get('resource_topology_valid'):
        raise ValueError('resource contract violation')
    projection = report['stages']['projection']
    result = {'artifact_root': str(case), 'status': 'PASS',
              'shape': report['shape'], 'end_to_end_cycles': report['end_to_end_cycles'],
              'projection_cycles': projection['elapsed_cycles'],
              'attention_cycles': report['stages']['attention']['elapsed_cycles'],
              'handoff_cycles': report['stage_handoff_cycles']['projection_to_attention'],
              'weight_programs': projection['weight_programs'],
              'O_max_abs_error': reports['fused_attention_result']['max_abs_error'],
              'dataflow_by_worker': projection['dataflow_by_worker'],
              'input_flow_by_worker': projection['input_flow_by_worker'],
              'local_gm_by_worker': projection['local_gm_by_worker']}
    # Keep the historical field for phase reconstruction compatibility. New
    # performance consumers use the explicit canonical primary metric.
    result['legacy_job_to_worker_complete_cycles'] = result['end_to_end_cycles']
    canonical_path = case / 'dma_e2e_result.json'
    if canonical_path.exists():
        canonical = json.loads(canonical_path.read_text())
        if canonical['status'] != 'PASS' or canonical['shape'] != report['shape']:
            raise ValueError('canonical DMA timing verification failed')
        result['primary_end_to_end_cycles'] = canonical['end_to_end_cycles']
        result['primary_timing_contract'] = canonical['contract']['version']
        result['dma_to_root_complete_cycles'] = canonical['dma_to_root_complete_cycles']
    if reference:
        result['byte_exact'] = exact_outputs(case, reference)
        old = json.loads((reference / 'projection_e2e_result.json').read_text())
        if 'primary_end_to_end_cycles' in result:
            from report_dma_e2e import report as report_dma
            # Recompute archived references in memory; do not overwrite their
            # original reports or compare a DMA metric against a legacy metric.
            previous_dma = report_dma(reference)
            if previous_dma['contract']['version'] != result['primary_timing_contract']:
                raise ValueError('primary performance timing contracts differ')
            before, after = previous_dma['end_to_end_cycles'], result['primary_end_to_end_cycles']
            result['primary_end_to_end_comparison'] = {
                'before': before, 'after': after, 'saved_cycles': before-after,
                'reduction_percent': 100*(before-after)/before,
                'timing_contract': result['primary_timing_contract']}
        for stage, new, previous in (
                ('end_to_end', result['end_to_end_cycles'], old['end_to_end_cycles']),
                ('projection', result['projection_cycles'], old['stages']['projection']['elapsed_cycles'])):
            result[stage + '_comparison'] = {'before': previous, 'after': new,
                'saved_cycles': previous - new, 'reduction_percent': 100 * (previous - new) / previous}
    if trace:
        logs = list((case / 'logs').glob('attention-sst*.log'))
        result['group_trace'] = verify_group_trace(logs[0].read_text(errors='replace'), 16,
            report['shape']['Hq'] + 2 * report['shape']['Hkv'], report['shape']['query_length'] // 1024,
            report['shape']['Hq'] * report['shape']['head_dim'] // 64)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--stage', choices=('small', 'formal'), default='small')
    parser.add_argument('--variants', choices=tuple(VARIANTS), nargs='+', default=['baseline', 'input', 'reads', 'groups'])
    parser.add_argument('--sequences', type=int, nargs='+')
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--slow-gm', action='store_true')
    parser.add_argument('--unfused', action='store_true', help='Validate row-major input and the unfused RMSNorm path')
    parser.add_argument('--critical-trace', action='store_true', help='Enable phase, physical resource and worker control tracing')
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    sequences = args.sequences or ([1024, 2048] if args.stage == 'formal' else [1024])
    original_hashes = {str(p.relative_to(ROOT)): digest(p) for p in SOURCES}
    snapshots = root / 'source_version'
    snapshots.mkdir(exist_ok=False)
    for source in SOURCES:
        if source.suffix in ('.h', '.cc', '.py', '.sh'):
            shutil.copy2(source, snapshots / source.name)
    results = []
    for variant in args.variants:
        pair, weight, groups = VARIANTS[variant]
        for sequence in sequences:
            case = root / variant / f's{sequence}'
            case.mkdir(parents=True, exist_ok=False)
            baseline = root / 'baseline' / f's{sequence}'
            reference = (baseline if variant != 'baseline' and (baseline / 'projection_e2e_result.json').exists()
                         else args.reference_root.resolve() / f's{sequence}' if args.reference_root else None)
            rank_logs = case / 'mpi_rank_output'
            env = {**os.environ, **GM_PROFILE, **BEST,
                   'GOLEM_PROJECTION_ARRAYS': '64', 'GOLEM_PROJECTION_RETAIN_OUTPUT': '1',
                   'GOLEM_PROJECTION_PIPELINE': '1', 'GOLEM_PROJECTION_SHARED_INPUT': '1',
                   'GOLEM_PROJECTION_WEIGHT_PREFETCH': '1', 'GOLEM_PROJECTION_WEIGHT_PREFETCH_DEPTH': '4',
                   'GOLEM_ARRAY_OPERAND_CONTEXT_BANKS': '1', 'GOLEM_ATTENTION_CROSS_TILE_OPERAND_PIPELINE': '0',
                   'GOLEM_ATTENTION_CAUSAL_BALANCE': '1', 'GOLEM_PROJECTION_ROW_FIRST': '0',
                   'GOLEM_PROJECTION_INPUT_PAIR_PREFETCH': str(pair), 'GOLEM_PROJECTION_LOCAL_WEIGHT_PREFETCH': str(weight),
                   'GOLEM_PROJECTION_ARRAY_GROUPS': str(groups),
                   'GOLEM_PROJECTION_OVERLAP_TRACE': str(int(args.stage == 'small' and groups == 2)),
                   'GOLEM_CRITICAL_PATH_TRACE': str(int(args.critical_trace)),
                   'GOLEM_MPI_RANK_LOG_ROOT': str(rank_logs),
                   'GOLEM_MPI_ARGS': f'--bind-to core --cpu-set 4-7 --output-filename {rank_logs}'}
            env.pop('GOLEM_PROJECTION_WEIGHT_STATIONARY', None)
            if args.slow_gm:
                env.update({'GOLEM_LOCAL_GM_BYTES_PER_CYCLE': '64',
                            'GOLEM_LOCAL_GM_BASE_LATENCY_CYCLES': '3', 'GOLEM_LOCAL_GM_QUEUE_DEPTH': '64'})
            if args.unfused:
                env.update({'GOLEM_FUSION_NORM_PROJECTION': '0', 'GOLEM_PROJECTION_PACKED_NORM_INPUT': '0',
                            'GOLEM_FUSION_SKIP_AUDIT_OUTPUT': '0'})
            if reference:
                env['GOLEM_PROJECTION_INPUT_REFERENCE'] = str(reference)
            heads = (32, 8) if args.stage == 'formal' else (4, 2)
            command = [str(HERE / 'run_sst.sh'), '--projection', '--query-length', str(sequence),
                       '--kv-length', str(sequence), '--num-query-heads', str(heads[0]),
                       '--num-kv-heads', str(heads[1]), '--head-dim', '64', '--causal', '--rope',
                       '--artifact-root', str(case)]
            provenance = {'sha256': original_hashes, 'command': command,
                          'environment': {k: v for k, v in env.items() if k.startswith('GOLEM_')}}
            (case / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
            print(f'Running {variant} S{sequence}: {case}', flush=True)
            with (case / 'driver.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            if {str(p.relative_to(ROOT)): digest(p) for p in SOURCES} != original_hashes:
                raise ValueError('source/binary changed during the controlled experiment')
            result = summarize(case, reference, args.stage == 'small' and groups == 2)
            provenance['input_sha256'] = {p.name: digest(p) for p in sorted(case.glob('*.bin'))}
            (case / 'baseline_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
            result['variant'] = variant
            results.append(result)
            (case / 'experiment_result.json').write_text(json.dumps(result, indent=2) + '\n')
            complete = len(results) == len(args.variants) * len(sequences)
            (root / 'summary.json').write_text(json.dumps({'status': 'PASS' if complete else 'RUNNING',
                'expected_cases': len(args.variants) * len(sequences), 'cases': results}, indent=2) + '\n')
            print(f"{variant} S{sequence}: E2E={result['end_to_end_cycles']}, projection={result['projection_cycles']}, PASS", flush=True)


if __name__ == '__main__':
    main()
