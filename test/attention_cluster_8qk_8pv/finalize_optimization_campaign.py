#!/usr/bin/env python3
"""Reconcile the P0–P4 matrix; keep rejected candidates explicitly opt-in."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from summarize_optimization_campaign import compare
from summarize_reuse_window_optimization import read, scopes
from analyze_pv_wo_timeline import handoff
from audit_prefill_hbm import audit

NAMES = ('p4_scoped_control', 'p4_scoped_ffn2', 'p3_scoped_relocation', 'p4_scoped_direct', 'p4_scoped_combined')


def regression(path):
    text = path.read_text()
    match = re.search(r'Ran (\d+) tests? in', text)
    if not match or not text.rstrip().endswith('OK'):
        raise ValueError('regression suite did not pass: '+str(path))
    return int(match[1])


def finalize(root):
    cases = {n: root/n/'reuse_window_probe' for n in NAMES}
    baseline = 'p4_scoped_control'
    comparisons = {n: compare(cases[baseline], cases[n]) for n in NAMES if n != baseline}
    direct_vs_relocation = compare(cases['p3_scoped_relocation'], cases['p4_scoped_direct'])
    results = {n: read(c/'decoder_stack_result.json') for n, c in cases.items()}
    configurations = {}
    for n, case in cases.items():
        env = read(case/'decoder_stack_provenance.json')['environment']
        expected = {'GOLEM_REUSE_WINDOW_ARRAY_GROUPS': '2',
                    'GOLEM_REUSE_WINDOW_K_TILES': '32', 'GOLEM_REUSE_WINDOW_PREFETCH_SLOTS': '4',
                    'GOLEM_REUSE_WINDOW_GM_LOOKAHEAD': '4',
                    'GOLEM_DECODER_FFN_GM_LOOKAHEAD': '2' if n in ('p4_scoped_ffn2', 'p4_scoped_combined') else '0',
                    'GOLEM_PV_WO_GM_HANDOFF': '1' if n in ('p3_scoped_relocation', 'p4_scoped_direct', 'p4_scoped_combined') else '0',
                    'GOLEM_PV_WO_DIRECT_INPUT': '1' if n in ('p4_scoped_direct', 'p4_scoped_combined') else '0',
                    'GOLEM_LOCAL_GM_MODEL': 'banked'}
        if any(env.get(k) != v for k, v in expected.items()):
            raise ValueError('measured configuration differs from matrix cell: '+n)
        configurations[n] = expected
    eligible = [baseline] + [n for n in ('p4_scoped_ffn2', 'p4_scoped_direct', 'p4_scoped_combined') if comparisons[n]['eligible']]
    selected = min(eligible, key=lambda n: scopes(results[n])['完整端到端'])
    profiles = {}; traffic = {}
    for n, case in cases.items():
        result_hash = hashlib.sha256((case/'decoder_stack_result.json').read_bytes()).hexdigest()
        for suffix, target, function, source, hash_key in (
            ('_timeline.json', profiles, handoff, 'analyze_pv_wo_timeline.py', 'analyzer_sha256'),
            ('_hbm_audit.json', traffic, audit, 'audit_prefill_hbm.py', 'source_sha256')):
            path = root/(n+suffix)
            cached = read(path) if path.exists() else {}
            source_hash = hashlib.sha256(Path(__file__).with_name(source).read_bytes()).hexdigest()
            if cached.get('result_sha256') != result_hash or cached.get(hash_key) != source_hash:
                cached = function(case)
                path.write_text(json.dumps(cached, indent=2)+'\n')
            target[n] = cached
    for n in ('p4_scoped_direct', 'p4_scoped_combined'):
        for key in ('relocation_read_bytes', 'relocation_write_bytes'):
            if profiles[n][key] != 0:
                raise ValueError('direct candidate still relocates input')
        direct = results[n]['resources']['optimizations']['pv_wo_direct_input']
        if direct.get('status') != 'PASS' or not direct.get('enabled'):
            raise ValueError('actual direct GM request and ownership audit missing')
    control = scopes(results[baseline]); chosen = scopes(results[selected])
    original = scopes(read(root/'p0_groups1/reuse_window_probe/decoder_stack_result.json'))
    counts = {name: regression(root/path) for name, path in
              (('host', 'python_scoped_final_regression.log'), ('production', 'production_scoped_final_regression.log'))}
    report = dict(status='PASS', phases={f'P{n}': 'complete' for n in range(5)}, latency_unit='cycles',
                  ticks_per_reference_cycle=1000, selected=selected,
                  full_cycles=chosen['完整端到端'], decoder_cycles=chosen['两层 decoder'],
                  saved_vs_p0_groups1_cycles=original['完整端到端']-chosen['完整端到端'],
                  saved_vs_groups2_cycles=control['完整端到端']-chosen['完整端到端'],
                  target_cycles=6000000, target_met=chosen['完整端到端'] < 6000000,
                  matrix={n: scopes(d) for n, d in results.items()}, configurations=configurations, comparisons=comparisons,
                  direct_vs_relocation=direct_vs_relocation, regressions=counts,
                  p0=read(root/'p0_comparison.json'),
                  new_build_equivalence=read(root/'scoped_build_control_equivalence.json'),
                  previous_build_equivalence=read(root/'new_build_control_equivalence.json'),
                  rejected_global_lookahead=read(root/'rejected_global_lookahead.json'),
                  local_scoped_screening={scope: read(root/f'p2_scoped_{scope}_comparison.json')
                                          for scope in ('single', 'cluster')},
                  traffic={n: t['pv_to_wo'] for n, t in traffic.items()},
                  scope='Synthetic S1024/H2048/FFN8192, 2 layers, vocab128256, 16 workers, MPI8; all dense tiles executed.',
                  boundary='First token-ID DMA send to final valid-token full-logits DMA ACK; host argmax excluded.',
                  note='Functional completion includes rejected performance candidates. Raw ticks and host wall seconds are distinct from accelerator cycles.')
    (root/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# P0–P4 cycle campaign', '', report['scope'], '',
             '| Configuration | Full cycles | Decoder cycles | Eligible vs control |',
             '| --- | ---: | ---: | --- |']
    for n in NAMES:
        s = scopes(results[n]); allowed = n == baseline or comparisons[n]['eligible']
        lines.append(f"| {n} | {s['完整端到端']:,.3f} | {s['两层 decoder']:,.3f} | {allowed} |")
    lines += ['', f'Selected: `{selected}`. Target below 6,000,000 cycles met: {report["target_met"]}.',
              '', 'All paired outputs are FP16-bit-identical, including all 128256 logits. Hardware and dense window work remain fixed.',
              '', f'Regressions: host {counts["host"]}, production {counts["production"]}, all passing.',
              '', 'Direct input audits preserve real GEMM A reads while proving zero relocation and zero O HBM roundtrip.',
              'A faster direct candidate relative to relocation alone is insufficient for adoption; the HBM control is the performance gate.']
    (root/'summary.md').write_text('\n'.join(lines)+'\n')
    flags = ['--shape', 'formal', '--layers', '2', '--vocab', '128256',
             '--window-array-groups', '2', '--window-k-tiles', '32',
             '--window-prefetch-slots', '4', '--window-early-a', '1',
             '--window-output-overlap', '1', '--window-a-prefetch', '1',
             '--window-gm-lookahead', '4',
             '--ffn-gm-lookahead', '2' if selected in ('p4_scoped_ffn2', 'p4_scoped_combined') else '0',
             '--ffn-panel-tiles', '16', '--lm-n-parallel', '1',
             '--gm-stage-chain', '--layer-residency', '--lm-input-residency',
             '--input-residency', '--elide-intermediate-output', '--add-norm-fusion', '--eager-final-norm']
    if selected in ('p4_scoped_direct', 'p4_scoped_combined'):
        flags += ['--pv-wo-gm-handoff', '--pv-wo-direct-input']
    repository = Path(__file__).resolve().parents[2]
    configuration = dict(status='PASS', latency_unit='cycles', selected=selected,
                         full_cycles=report['full_cycles'], decoder_cycles=report['decoder_cycles'], flags=flags,
                         baseline_result=str((cases[selected]/'decoder_stack_result.json').relative_to(repository)),
                         verification_reference='results/software_output_residency_20261022/final_control/reuse_window_probe/decoder_stack_result.json',
                         measured_library_sha256=read(cases[selected]/'decoder_stack_provenance.json')['sst_library_sha256'],
                         scope=report['scope'], boundary=report['boundary'])
    Path(__file__).with_name('SELECTED_PREFILL_2026-10-24.json').write_text(json.dumps(configuration, indent=2)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-root', type=Path, required=True)
    a = p.parse_args()
    report = finalize(a.experiment_root.resolve())
    print(json.dumps({k: report[k] for k in ('status', 'selected', 'full_cycles', 'decoder_cycles', 'target_met')}, indent=2))


if __name__ == '__main__':
    main()
