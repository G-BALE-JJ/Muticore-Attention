#!/usr/bin/env python3
"""Collect completed migration gates without modifying their evidence."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PHASES = (
    ('01_partial_c_resume_20261027', 'Partial-C readiness'),
    ('02_hbm_stats', 'HBM statistics v2'),
    ('03_cu_tu_te', 'CU/TU/TE separation'),
    ('04_precision_active_k', 'Explicit packed precision'),
    ('05_placement', 'Explicit workload/HBM placement'),
    ('06_timing_fast_path', 'Timing numerical-data fast path'),
)
EXPECTED = dict(GEMM=3500, Attention=12852, GQA=37928, prefill=6084167.75)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', type=Path, default=ROOT/'test/reports')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    public = read(ROOT/'test/attention_cluster_8qk_8pv/SELECTED_PREFILL_2026-10-26.json')
    phases = []
    for name, mechanism in PHASES:
        directory = args.evidence_root/('alignment_'+name)
        gate, summary, selection = (read(directory/file) for file in
                                   ('gate.json', 'summary.json', 'candidate_selection.json'))
        if gate['status'] != 'PASS' or gate['stages'] != {s: 'PASS' for s in EXPECTED}:
            raise ValueError('Incomplete gate: '+name)
        cycles = {r['stage']: r['end_to_end_cycles'] for r in summary['stages']}
        if summary['status'] != 'PASS' or cycles != EXPECTED or gate['cycles'] != EXPECTED:
            raise ValueError('Functional/cycle contract changed: '+name)
        if any(r['status'] != 'PASS' for r in summary['stages']):
            raise ValueError('Failed stage: '+name)
        if gate['library_sha256'] != summary['library_sha256'] or \
           gate['library_sha256'] != selection['measured_library_sha256']:
            raise ValueError('Library provenance mismatch: '+name)
        for key, value in public.items():
            if key != 'measured_library_sha256' and selection.get(key) != value:
                raise ValueError('Public workload selection changed: '+name+' '+key)
        if summary['guest_sha256'] != public['measured_guest_sha256']:
            raise ValueError('Selected guest changed: '+name)
        provenance_path = Path(summary['invocations']['prefill']['artifact_root']) / \
            'reuse_window_probe/decoder_stack_provenance.json'
        provenance = read(provenance_path)
        memory = provenance['sst_memhierarchy_sha256']
        if provenance['sst_library_sha256'] != gate['library_sha256'] or \
           provenance['guest_sha256'] != public['measured_guest_sha256']:
            raise ValueError('Prefill binary provenance mismatch: '+name)
        if 'memhierarchy_sha256' in summary and summary['memhierarchy_sha256'] != memory:
            raise ValueError('Memory summary mismatch: '+name)
        if 'memhierarchy_sha256' in gate and gate['memhierarchy_sha256'] != memory:
            raise ValueError('Memory provenance mismatch: '+name)
        phases.append(dict(phase=name, mechanism=mechanism, status='PASS',
                           source_commit=gate['source_commit'],
                           library_sha256=gate['library_sha256'],
                           memhierarchy_sha256=memory,
                           prefill_provenance_sha256=digest(provenance_path),
                           cycles=cycles, evidence_directory=str(directory.resolve()),
                           evidence_sha256={file: digest(directory/file) for file in
                               ('gate.json', 'summary.json', 'candidate_selection.json')}))
    control = read(args.evidence_root/'alignment_05_timing_control/result.json')
    timing = read(args.evidence_root/'alignment_06_timing_fast_path/result.json')
    if any(r['status'] != 'PASS' or r['cycles'] != EXPECTED['GEMM'] or
           r['final_c_writeback_complete_ratio'] != '4/4' for r in (control, timing)) or \
       timing.get('delta_cycles') != 0 or timing.get('same_build_delta_cycles') != 0 or \
       timing.get('same_build_metrics_equal') is not True:
        raise ValueError('Paired timing contract changed')
    same_build = read(Path(timing['same_build_control']))
    for key in ('library_sha256', 'memhierarchy_sha256', 'source_commit', 'cycles', 'metrics_sha256'):
        if same_build[key] != timing[key]:
            raise ValueError('Same-build evidence mismatch: '+key)
    result = dict(status='PASS', latency_unit='cycles', formal_runs=24,
                  selected_guest_sha256=public['measured_guest_sha256'], phases=phases,
                  timing_control=control, timing_candidate=timing, timing_same_build_control=same_build,
                  claim='Default functional and cycle equivalence; no measured speedup claimed.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = [
        '# LLM mechanism migration', '',
        'All six sequential gates PASS: 24 full functional runs. Each gate passed',
        'before the following feature was applied. Accelerator latency is in cycles.', '',
        '| Mechanism | GEMM | Attention | GQA | prefill |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for phase in phases:
        lines.append('| '+phase['mechanism']+' | '+
                     ' | '.join(str(phase['cycles'][s]) for s in EXPECTED)+' |')
    lines += [
        '', 'Every delta from the public baseline is zero. Hardware resources,',
        'legacy arithmetic, the pinned guest and the selected schedule are retained.',
        'The four workloads have different shapes and boundaries; do not add their latencies.', '',
        'Prefill uses synthetic S1024/H2048/FFN8192, two layers, vocabulary 128256,',
        '16 workers and MPI8. Sparse-valued inputs execute all dense tiles.',
        'Its boundary is the first token-ID DMA send through the final valid-token',
        'full-vocabulary logits DMA ACK. Host preparation, validation and argmax are excluded.', '',
        'The timing-only GEMM before and after the numerical-data fast path is',
        '3500 cycles with 4/4 output writebacks complete; the paired delta is zero.',
        'A same-build unoptimized control additionally matches all recorded execution,',
        'terminal, memory, DMA, NoC and HBM command statistics byte for byte.',
        'Numerical correctness is established by the functional gates.', '',
        'Packed precision and workload placement are explicit native opt-ins.',
        'Installed-library TU/TE fixture tests cover the new interfaces, arithmetic,',
        'cross-K accumulation, B replication and functional/timing event equivalence.',
        'They do not establish NoC/HBM performance for a new opt-in mapping.',
        'No TileLang/TileBin components were introduced.', '',
        'Interface contracts and restrictions: ',
        '[NATIVE_INTERFACES.md](../../maintenance/mechanism_alignment/NATIVE_INTERFACES.md).',
        'The manifest records per-phase source commits, library hashes and evidence hashes.',
        'Raw artifacts remain in the experimental worktree paths recorded there.',
        'The original interrupted phase 01 is retained separately and is not counted.',
        'The recovered phase 01 report-only finalization fix is explained in',
        '[RECOVERY.md](../../maintenance/mechanism_alignment/RECOVERY.md).', '',
    ]
    (args.output_dir/'REPORT.md').write_text('\n'.join(lines))
    print('PASS: six gates, 24 formal functional runs, zero cycle deltas')


if __name__ == '__main__':
    main()
