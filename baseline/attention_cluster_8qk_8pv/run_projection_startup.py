#!/usr/bin/env python3
"""Paired startup experiment: identical IW hardware/dataflow, two real guests."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from run_batch_b import environment, SOURCE_FILES
from run_projection_overlap import ROOT, HERE, digest, summarize
from analyze_resource_timeline import analyze as analyze_resources
from analyze_critical_path import analyze as analyze_critical
from analyze_projection_startup import analyze, compare

GUEST_DIR = ROOT / 'src/sst/elements/golem/tests/small/muticore_attention'
GUESTS = {'baseline': GUEST_DIR / 'riscv64/fused_attention',
          'batch_release': GUEST_DIR / 'riscv64/batch_release/fused_attention'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--stage', choices=('small', 'formal'), default='small')
    parser.add_argument('--sequences', nargs='+', type=int, default=[1024])
    parser.add_argument('--variants', nargs='+', choices=tuple(GUESTS), default=list(GUESTS))
    parser.add_argument('--cpu-set', default='0-3')
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    sources = list(dict.fromkeys(SOURCE_FILES + list(GUESTS.values()) + [
        GUEST_DIR / 'golem_attention_runtime.cpp', GUEST_DIR / 'golem_attention_runtime.h',
        GUEST_DIR / 'golem_projection_runtime.h', GUEST_DIR / 'Makefile',
        HERE / 'run_projection_startup.py', HERE / 'analyze_projection_startup.py',
        HERE / 'run_llama_projection.py', HERE / 'run_causal_balance.py']))
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    (root / 'source_sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    for source in sources:
        target = root / 'source_snapshot' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    rows = []
    for sequence in args.sequences:
        reference = args.reference_root.resolve() / f's{sequence}'
        for variant in args.variants:
            case = root / variant / f's{sequence}'
            case.mkdir(parents=True)
            env = environment(case, (1, 1, 1), 'none', sequence, args.stage)
            env['GOLEM_ATTENTION_GUEST_EXECUTABLE'] = str(GUESTS[variant])
            env['GOLEM_PROJECTION_INPUT_REFERENCE'] = str(reference)
            env['GOLEM_MPI_ARGS'] = f'--bind-to core --cpu-set {args.cpu_set} --output-filename {case / "mpi_rank_output"}'
            if env.get('VANADIS_CPU_CLOCK') != '1.0GHz':
                raise ValueError('startup comparison requires the frozen 1 GHz platform')
            heads = (32, 8) if args.stage == 'formal' else (4, 2)
            command = [str(HERE / 'run_sst.sh'), '--projection', '--query-length', str(sequence),
                '--kv-length', str(sequence), '--num-query-heads', str(heads[0]),
                '--num-kv-heads', str(heads[1]), '--head-dim', '64', '--causal', '--rope',
                '--artifact-root', str(case)]
            provenance = {'sha256': hashes, 'command': command, 'variant': variant,
                'compile_time_batch_release': int(variant == 'batch_release'),
                'guest_sha256': digest(GUESTS[variant]),
                'environment': {k: v for k, v in env.items() if k.startswith(('GOLEM_', 'VANADIS_'))}}
            (case / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
            print(f'Running {variant} S{sequence}', flush=True)
            with (case / 'driver.log').open('w') as stream:
                subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                               stderr=subprocess.STDOUT, check=True)
            if hashes != {str(p.relative_to(ROOT)): digest(p) for p in sources}:
                raise ValueError('sources or binaries changed during startup experiment')
            row = summarize(case, reference)
            resources = analyze_resources(case, allow_filtered=True)
            (case / 'resource_analysis.json').write_text(json.dumps(resources, indent=2) + '\n')
            critical = analyze_critical(case)
            if critical['reconstructed_e2e_cycles'] != row['end_to_end_cycles']:
                raise ValueError('phase/control chain does not reproduce E2E')
            (case / 'critical_path_analysis.json').write_text(json.dumps(critical, indent=2) + '\n')
            inputs = {p.name: digest(p) for p in sorted(case.glob('*.bin'))}
            provenance['input_sha256'] = inputs
            (case / 'baseline_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
            row.update({'variant': variant, 'input_sha256': inputs, 'sha256': hashes,
                        'startup': analyze(case, resources)})
            if variant == 'baseline' and row['end_to_end_comparison']['saved_cycles'] != 0:
                raise ValueError('unchanged guest baseline failed to reproduce frozen IW E2E')
            if variant == 'batch_release' and any(v >= 1000 for v in row['startup']['manager_release_span_cycles'].values()):
                raise ValueError('batched release did not eliminate per-worker logging gaps')
            (case / 'experiment_result.json').write_text(json.dumps(row, indent=2) + '\n')
            rows.append(row)
            complete = len(rows) == len(args.sequences) * len(args.variants)
            comparisons = {}
            for seq in args.sequences:
                paired = {r['variant']: r for r in rows if r['shape']['query_length'] == seq}
                if set(paired) == set(GUESTS):
                    comparisons[str(seq)] = compare(paired['baseline'], paired['batch_release'])
            (root / 'summary.json').write_text(json.dumps({'status': 'PASS' if complete else 'RUNNING',
                'cases': rows, 'comparisons': comparisons}, indent=2) + '\n')
            print(f'{variant} S{sequence}: PASS E2E={row["end_to_end_cycles"]}, accept span={row["startup"]["worker_accept_span_cycles"]}', flush=True)


if __name__ == '__main__':
    main()
