#!/usr/bin/env python3
"""Audit a same-build final-Norm/LM scheduling pair and compare cycles."""
import argparse
import hashlib
import json
import re
from pathlib import Path

from summarize_reuse_window_optimization import comparable_inputs, read, scopes
from summarize_software_stage_chain import output_bits
from summarize_software_dataflow_next import environment
from summarize_gm_prefetch import flow_stats

FLAG = 'GOLEM_DECODER_FINAL_NORM_LM_STREAM'
BINARY_KEYS = ('guest_sha256', 'sst_library_sha256', 'sst_memhierarchy_sha256')


def normalized_sources(mapping):
    result = {}
    for name, digest in mapping.items():
        name = re.sub(r'\.all_gemm_reuse_runner_[^/]+/', '<isolated-runner>/', name)
        if name in result:
            raise ValueError('duplicate normalized source name')
        result[name] = digest
    return result


def check_pair_configuration(cases, provenance):
    manifests = [comparable_inputs(case) for case in cases]
    if [m.pop('final_norm_lm_stream', False) for m in manifests] != [False, True]:
        raise ValueError('expected stream-off control and stream-on candidate')
    if manifests[0] != manifests[1]:
        raise ValueError('input tensors, numerical contract or layout changed')
    for key in BINARY_KEYS:
        if not provenance[0][key] or provenance[0][key] != provenance[1][key]:
            raise ValueError('paired binaries differ: '+key)
    if normalized_sources(provenance[0]['source_sha256']) != normalized_sources(provenance[1]['source_sha256']):
        raise ValueError('paired source hashes differ')
    envs = [environment(p, case) for p, case in zip(provenance, cases)]
    changes = {k: dict(control=envs[0].get(k), candidate=envs[1].get(k))
               for k in sorted(set(envs[0]) | set(envs[1])) if envs[0].get(k) != envs[1].get(k)}
    if changes != {FLAG: dict(control='0', candidate='1')}:
        raise ValueError('unplanned environment difference: '+repr(changes))
    return changes


def cycle_comparison(before, after):
    return {k: dict(control_cycles=before[k], candidate_cycles=v,
                    saved_cycles=before[k]-v, reduction_percent=100*(1-v/before[k]))
            for k, v in after.items()}


def compare(control, candidate):
    cases = [control.resolve(), candidate.resolve()]
    results = [read(p/'decoder_stack_result.json') for p in cases]
    provenance = [read(p/'decoder_stack_provenance.json') for p in cases]
    reports = [read(p.parent/'all_gemm_reuse_window_report.json') for p in cases]
    if any(d['status'] != 'PASS' for d in results + reports):
        raise ValueError('numerical/lifecycle/GEMM coverage audit failed')
    changes = check_pair_configuration(cases, provenance)
    for i, d in enumerate(results):
        if d['gm_banks']['status'] != 'PASS' or d['gm_banks']['untimed_tensor_bytes']:
            raise ValueError('untimed tensor traffic')
        if d['lm_head']['stream_mode'] != bool(i):
            raise ValueError('verified stream mode mismatch')
        if d['lm_head']['first_input_dma_tick'] <= d['lm_head']['input_ready_ack_tick']:
            raise ValueError('LM read before producer output ACK')
    flows = [flow_stats(case, p['environment']) for case, p in zip(cases, provenance)]
    if flows[0]['traffic'] != flows[1]['traffic']:
        raise ValueError('dense GEMM MAC/A/B/C traffic changed')
    bits = output_bits(*cases)
    measurements = []
    for d in results:
        s = scopes(d)
        s['最终 Norm 后至 logits ACK'] = d['lm_head']['tail_cycles']
        s['最终 decoder 后至 logits ACK'] = d['lm_head']['end_to_end_cycles']-d['dma_timing']['end_to_end_cycles']
        measurements.append(s)
    timing = cycle_comparison(*measurements)
    eligible = all(timing[k]['saved_cycles'] > 0 for k in ('完整端到端', '两层 decoder'))
    return dict(status='PASS', unit='cycles', ticks_per_cycle=1000,
                boundary='first token-ID DMA send to final valid-token full-logits DMA ACK; host argmax excluded',
                control=str(cases[0]), candidate=str(cases[1]), timing=timing,
                eligible=eligible, decision='ACCEPT' if eligible else 'REJECT',
                acceptance_rule='Both complete end-to-end and decoder cycles must decrease; all audits and bitwise outputs must pass.',
                environment_changes=changes, binaries={k: provenance[0][k] for k in BINARY_KEYS},
                bits=bits, dense_window_work=flows[0]['traffic'],
                postrun_verifier_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ('run_decoder_stack.py', 'verify_lm_head.py', 'verification_retained_input.py',
                                 'compare_final_norm_lm_stream.py')},
                verification_reference=[read(p.parent/'verification_reference.json') for p in cases],
                note='Stage spans overlap and must not be added. Host placement is normalized; host wall time is not accelerator latency.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = compare(a.control, a.candidate)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({k: result[k] for k in ('status', 'decision', 'timing')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
