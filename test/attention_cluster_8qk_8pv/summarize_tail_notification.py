#!/usr/bin/env python3
"""Audit restored A/A controls and independent software tail candidates."""
import argparse
import json
from pathlib import Path

from compare_final_norm_lm_stream import BINARY_KEYS, normalized_sources, cycle_comparison
from summarize_reuse_window_optimization import comparable_inputs, read, scopes
from summarize_software_stage_chain import output_bits
from summarize_software_dataflow_next import environment
from summarize_gm_prefetch import flow_stats

FLAGS = {'GOLEM_DECODER_FINAL_NORM_LM_STREAM', 'GOLEM_DECODER_FINAL_NORM_PRODUCER_FIRST'}
SOFTWARE = ('final_norm_lm_stream', 'final_norm_producer_first')


def fixed_hardware_across_builds(cases, provenance):
    """Cross-build comparisons diagnose total cost; they cannot isolate a flag."""
    manifests = [comparable_inputs(c) for c in cases]
    for m in manifests:
        for k in (*SOFTWARE, 'tail_schedule_experiment', 'lm_handoff_protocol'):
            m.pop(k, None)
    if manifests[0] != manifests[1]:
        raise ValueError('cross-build inputs/layout/configuration differ')
    for k in BINARY_KEYS[1:]:
        if not provenance[0][k] or provenance[0][k] != provenance[1][k]:
            raise ValueError('cross-build simulator library mismatch: '+k)
    hardware = []
    for p in provenance:
        hardware.append({k:v for k,v in p['source_sha256'].items()
                         if k.startswith('src/sst/elements/golem/') and
                         ('/tests/' not in k or '/tests/architecture/' in k)})
    if not hardware[0] or hardware[0] != hardware[1]:
        raise ValueError('cross-build hardware sources differ')
    env = [environment(p,c) for p,c in zip(provenance,cases)]
    delta = {k:dict(default=env[0].get(k),experimental=env[1].get(k))
             for k in set(env[0])|set(env[1]) if env[0].get(k)!=env[1].get(k)}
    if set(delta)-(FLAGS|{'GOLEM_ATTENTION_GUEST_EXECUTABLE'}):
        raise ValueError('cross-build hardware environment differences: '+repr(delta))
    return dict(status='PASS', hardware_source_files=len(hardware[0]),
                environment_changes=delta,
                simulator_binaries={k:provenance[0][k] for k in BINARY_KEYS[1:]},
                same_guest_binary=False)


def configuration_pair(cases, provenance):
    manifests = [comparable_inputs(p) for p in cases]
    for m in manifests:
        for k in SOFTWARE:
            m.pop(k, None)
    if manifests[0] != manifests[1]:
        raise ValueError('paired inputs/layout/configuration differ')
    for k in BINARY_KEYS:
        if not provenance[0][k] or provenance[0][k] != provenance[1][k]:
            raise ValueError('paired binary mismatch: '+k)
    if normalized_sources(provenance[0]['source_sha256']) != normalized_sources(provenance[1]['source_sha256']):
        raise ValueError('paired sources differ')
    env = [environment(p, c) for p, c in zip(provenance, cases)]
    delta = {k: dict(control=env[0].get(k), candidate=env[1].get(k))
             for k in set(env[0]) | set(env[1]) if env[0].get(k) != env[1].get(k)}
    if set(delta)-FLAGS:
        raise ValueError('unplanned environment differences: '+repr(delta))
    return delta


def pair(a, b):
    cases = [a, b]
    result = [read(c/'decoder_stack_result.json') for c in cases]
    provenance = [read(c/'decoder_stack_provenance.json') for c in cases]
    for c, r in zip(cases, result):
        if r['status'] != 'PASS' or read(c.parent/'all_gemm_reuse_window_report.json')['status'] != 'PASS':
            raise ValueError('full numerical/resource/GEMM audit failed')
        if r['gm_banks']['untimed_tensor_bytes']:
            raise ValueError('untimed tensor IO')
    changes = configuration_pair(cases, provenance)
    flows = [flow_stats(c, p['environment'])['traffic'] for c, p in zip(cases, provenance)]
    if flows[0] != flows[1]:
        raise ValueError('dense GEMM work changed')
    timing = cycle_comparison(*map(scopes, result))
    prefix_equal = (result[0]['embedding']['first_token_ids_dma_tick'] == result[1]['embedding']['first_token_ids_dma_tick']
                    and result[0]['embedding']['last_embedding_output_ack_tick'] == result[1]['embedding']['last_embedding_output_ack_tick']
                    and result[0]['lifecycle']['stage_spans']['1'] == result[1]['lifecycle']['stage_spans']['1'])
    tail = {}
    for label,r in zip(('control','candidate'),result):
        lm=r['lm_head'];norm=r['final_norm']
        tail[label]=dict(producer_core=lm['input_producer_core'],
                         producer_ready_cycle=norm['worker_output_ack_ticks'][str(lm['input_producer_core'])]/1000,
                         all_norm_ready_cycle=norm['last_output_ack_tick']/1000,
                         first_lm_input_cycle=lm['first_input_dma_tick']/1000,
                         lm_norm_overlap_cycles=max(0,(norm['last_output_ack_tick']-lm['first_input_dma_tick'])/1000),
                         norm_to_lm_handoff_cycles=lm['handoff_cycles'])
    return dict(status='PASS', timing=timing, bits=output_bits(*cases), environment_changes=changes,tail=tail,
                first_layer_prefix_identical=prefix_equal, binaries={k: provenance[0][k] for k in BINARY_KEYS},
                eligible=timing['完整端到端']['saved_cycles'] > 0)


def audit(root):
    cases = {n: (root/n/'reuse_window_probe').resolve() for n in
             ('p7_restored_a', 'p7_restored_b', 'p7_tail_control', 'p7_tail_notify', 'p7_tail_priority')}
    legacy = pair(cases['p7_restored_a'], cases['p7_restored_b'])
    historical = read(root/'p4_scoped_control/reuse_window_probe/decoder_stack_provenance.json')
    if legacy['binaries']['guest_sha256'] != historical['guest_sha256']:
        raise ValueError('default guest is not the historical measured binary')
    if any(t['saved_cycles'] for t in legacy['timing'].values()):
        raise ValueError('restored A/A timing is not repeatable')
    old = read(root/'p4_scoped_control/reuse_window_probe/decoder_stack_result.json')
    restored = read(cases['p7_restored_a']/'decoder_stack_result.json')
    control = read(cases['p7_tail_control']/'decoder_stack_result.json')
    fixed = fixed_hardware_across_builds(
        [cases['p7_restored_a'],cases['p7_tail_control']],
        [read(cases[n]/'decoder_stack_provenance.json') for n in ('p7_restored_a','p7_tail_control')])
    if scopes(old) != scopes(restored):
        raise ValueError('default cycles did not reproduce historical selection')
    baseline_bits = output_bits(root/'p4_scoped_control/reuse_window_probe', cases['p7_restored_a'])
    candidates = {}
    for name in ('p7_tail_notify', 'p7_tail_priority'):
        d = pair(cases['p7_tail_control'], cases[name])
        r = read(cases[name]/'decoder_stack_result.json')
        if not d['first_layer_prefix_identical']:
            raise ValueError('experimental switch perturbed the first layer prefix')
        if r['lm_head']['handoff_protocol'] != 'producer_remote_store_v1' or not r['lm_head']['worker_norm_flag_polling_forbidden']:
            raise ValueError('candidate did not verify zero worker HBM polling')
        full = r['embedding']['end_to_end_cycles']
        d['saved_vs_restored_default_cycles'] = restored['embedding']['end_to_end_cycles']-full
        d['adopt'] = d['eligible'] and d['saved_vs_restored_default_cycles'] > 0
        candidates[name] = d
    priority_effect = pair(cases['p7_tail_notify'], cases['p7_tail_priority'])
    return dict(status='PASS', unit='cycles', ticks_per_cycle=1000,
                boundary='first token-ID DMA send to final valid-token full-logits ACK; all Norm outputs complete; host argmax excluded',
                default_cycles=restored['embedding']['end_to_end_cycles'], restored_aa=legacy,
                fixed_hardware_across_builds=fixed,
                experimental_control_vs_default=dict(
                    timing=cycle_comparison(scopes(restored),scopes(control)),
                    bits=output_bits(cases['p7_restored_a'],cases['p7_tail_control']),
                    interpretation='Different guest builds. This measures the total experimental-path cost, not the causal effect of notification. Prefix timing already differs before tail options are read; total regression cannot be assigned entirely to final polling or getenv.'),
                historical_output_bits=baseline_bits, candidates=candidates, priority_effect=priority_effect,
                selected=min([('restored_default',restored['embedding']['end_to_end_cycles'])]+
                             [(n,read(cases[n]/'decoder_stack_result.json')['embedding']['end_to_end_cycles'])
                              for n,d in candidates.items() if d['adopt']],key=lambda x:x[1])[0],
                acceptance_rule='All correctness/provenance/resource checks pass; complete end-to-end cycles improve against both the same-build experimental control and the restored default. Decoder changes are reported separately because this experiment targets the tail.',
                note='Stage spans overlap. Producer-first prioritizes an existing 64-row Norm job; it is not row-level fusion and removes no tensor HBM IO.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.root.resolve())
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],selected=result['selected'],default_cycles=result['default_cycles']),indent=2))


if __name__ == '__main__':
    main()
