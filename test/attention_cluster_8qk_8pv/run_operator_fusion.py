#!/usr/bin/env python3
"""Measure opt-in fusion against a fixed-resource control, with private artifacts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VARIANTS = [('control', 0, 0, 0), ('norm_audit', 1, 0, 0),
            ('norm_production', 1, 0, 1), ('full_audit', 1, 1, 0),
            ('full_production', 1, 1, 1)]


def run_variant(root, variant, reference, sequences, cpu_set=None):
    name, norm, rope, skip = variant
    env = dict(os.environ, GOLEM_PROJECTION_CBUFFER_PARTIAL='1',
               GOLEM_FUSION_NORM_PROJECTION=str(norm),
               GOLEM_FUSION_PROJECTION_ROPE=str(rope),
               GOLEM_FUSION_SKIP_AUDIT_OUTPUT=str(skip))
    if cpu_set:
        env['GOLEM_MPI_ARGS'] = f'--bind-to core --cpu-set {cpu_set}'
    command = [sys.executable, str(HERE / 'run_llama_projection.py'),
               '--artifact-root', str(root / name), '--sequences', *map(str, sequences)]
    if reference:
        command += ['--reference-root', str(reference)]
    print(f'START {name}', flush=True)
    with (root / f'{name}.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    print(f'PASS {name}', flush=True)


def analyze(root):
    summaries = {v[0]: json.loads((root / v[0] / 'summary.json').read_text()) for v in VARIANTS}
    control = {c['pipeline']['shape']['query_length']: c for c in summaries['control']['cases']}
    rows = []
    for name, _, _, _ in VARIANTS:
        summary = summaries[name]
        if summary['status'] != 'PASS':
            raise ValueError(f'{name}: failed summary')
        for case in summary['cases']:
            report = case['pipeline']
            sequence = report['shape']['query_length']
            before = control[sequence]['pipeline']['end_to_end_cycles']
            fusion = report['fusion_by_manager']
            rows.append(dict(variant=name, sequence=sequence, status=report['status'],
                e2e_cycles=report['end_to_end_cycles'],
                saved_cycles=before-report['end_to_end_cycles'],
                reduction_percent=100*(before-report['end_to_end_cycles'])/before,
                projection_span_cycles=report['stages']['projection']['elapsed_cycles'],
                producer_span_cycles=report['stages']['projection']['end_cycle']-report['chain_start_cycle'],
                producer_to_attention_cycles=report['stage_handoff_cycles']['projection_to_attention'],
                attention_span_cycles=report['stages']['attention']['elapsed_cycles'],
                rmsnorm_active_cycles_max=max(report['stages']['rmsnorm']['active_cycles_by_manager'].values()),
                norm_vector_cycles_max=report['floor_components_cycles']['rmsnorm_vector'],
                rope_vector_cycles_max=max(e['rope_vector_cycles'] for e in fusion.values()),
                fusion_local_read_bytes=sum(e['local_read_bytes'] for e in fusion.values()),
                fusion_local_write_bytes=sum(e['local_write_bytes'] for e in fusion.values()),
                norm_audit_bytes=sum(e['norm_audit_bytes'] for e in fusion.values()),
                raw_audit_bytes=sum(e['raw_audit_bytes'] for e in fusion.values()),
                panel_bytes=sum(e['panel_bytes'] for e in fusion.values()),
                O_max_abs_error=case['max_abs_errors']['O']))
    result = dict(status='PASS', scope='Synthetic FP16 RMSNorm/QKV/RoPE/causal GQA/O; excludes Wo/residual/MLP',
        hardware='Same 4 managers, 64 arrays/manager, single output context, 2 MiB GM, 1 MiB C-buffer, ports/bandwidth/credits/NoC',
        stage_note='Projection span includes fused RMSNorm/RoPE; RMSNorm span overlaps projection at multi-block lengths. Do not sum spans.',
        retained_boundaries='Projection-completion synchronization and Q/K/V panel HBM transport', rows=rows)
    (root / 'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    with (root / 'analysis.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--sequences', type=int, nargs='+', default=[1024, 2048])
    parser.add_argument('--input-reference-root', type=Path, help='verified earlier input/control artifact root')
    parser.add_argument('--reuse-control', action='store_true', help='use an already completed control under artifact root')
    parser.add_argument('--cpu-sets', nargs='+', help='disjoint physical CPU sets, each with >=4 cores; one serial queue per set')
    parser.add_argument('--analyze-only', action='store_true')
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        analyze(root)
        return
    if args.reuse_control:
        summary = json.loads((root / 'control/summary.json').read_text())
        if summary['status'] != 'PASS' or {c['pipeline']['shape']['query_length'] for c in summary['cases']} != set(args.sequences):
            raise ValueError('missing/invalid control cases')
    else:
        run_variant(root, VARIANTS[0], args.input_reference_root, args.sequences,
                    args.cpu_sets[0] if args.cpu_sets else None)
    slots = args.cpu_sets or [None]
    def run_queue(index):
        for variant in VARIANTS[1:][index::len(slots)]:
            run_variant(root, variant, root / 'control', args.sequences, slots[index])
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures = [pool.submit(run_queue, index) for index in range(len(slots))]
        for future in futures:
            future.result()
    analyze(root)
    print(root / 'analysis.json')


if __name__ == '__main__':
    main()
