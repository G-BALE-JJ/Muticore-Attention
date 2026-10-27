#!/usr/bin/env python3
"""Compare cycle measurements only after provenance and output checks pass."""
import argparse
import json
from pathlib import Path

from summarize_reuse_window_optimization import comparable_inputs, read, scopes
from summarize_software_stage_chain import output_bits
from summarize_software_dataflow_next import environment
from summarize_gm_prefetch import flow_stats

BINARY_KEYS = ('guest_sha256', 'sst_library_sha256', 'sst_memhierarchy_sha256')
ALLOWED = {'GOLEM_DECODER_FFN_GM_LOOKAHEAD', 'GOLEM_REUSE_WINDOW_ARRAY_GROUPS', 'GOLEM_REUSE_WINDOW_GM_LOOKAHEAD',
           'GOLEM_PV_WO_GM_HANDOFF', 'GOLEM_PV_WO_DIRECT_INPUT'}


def inputs(case):
    m = comparable_inputs(case)
    for key in ('pv_wo_gm_handoff', 'pv_wo_direct_input'):
        m.pop(key, None)
    return m


def compare(control, candidate):
    cases = [control.resolve(), candidate.resolve()]
    results = [read(p/'decoder_stack_result.json') for p in cases]
    provenance = [read(p/'decoder_stack_provenance.json') for p in cases]
    reports = [read(p.parent/'all_gemm_reuse_window_report.json') for p in cases]
    if any(d['status'] != 'PASS' for d in results + reports):
        raise ValueError('numerical/lifecycle/GEMM coverage audit failed')
    if inputs(cases[0]) != inputs(cases[1]):
        raise ValueError('input tensors or layout changed')
    for key in BINARY_KEYS:
        if provenance[0][key] != provenance[1][key]:
            raise ValueError('paired measured binaries differ: '+key)
    # The isolated Python launchers embed the chosen scheduling environment;
    # production C++ source hashes must still match throughout a paired run.
    source_maps = [read(p.parent/'experiment_provenance.json')['source_sha256'] for p in cases]
    if source_maps[0] != source_maps[1]:
        raise ValueError('paired production source hashes differ')
    envs = [environment(p, c) for p, c in zip(provenance, cases)]
    changes = {k: dict(control=envs[0].get(k), candidate=envs[1].get(k))
               for k in sorted(set(envs[0]) | set(envs[1])) if envs[0].get(k) != envs[1].get(k)}
    if set(changes) - ALLOWED:
        raise ValueError('unplanned environment difference: '+repr(changes))
    flows = [flow_stats(c, p['environment']) for c, p in zip(cases, provenance)]
    if flows[0]['traffic'] != flows[1]['traffic']:
        raise ValueError('dense GEMM MAC/A/B/C traffic changed')
    for d in results:
        if d['gm_banks']['status'] != 'PASS' or d['gm_banks']['untimed_tensor_bytes']:
            raise ValueError('untimed tensor traffic')
    before, after = map(scopes, results)
    timing = {k: dict(control_cycles=before[k], candidate_cycles=v,
                      saved_cycles=before[k]-v, reduction_percent=100*(1-v/before[k]))
              for k, v in after.items()}
    return dict(status='PASS', unit='cycles', control=str(cases[0]), candidate=str(cases[1]),
                boundary='first token-ID DMA send to final last-token full-logits DMA ACK',
                eligible=all(timing[k]['saved_cycles'] > 0 for k in ('完整端到端', '两层 decoder')),
                timing=timing, bits=output_bits(*cases), environment_changes=changes,
                wait_counter_scope='Flow waits are accumulated across workers/windows and may overlap useful work; they are not end-to-end latency.',
                binaries={k: provenance[0][k] for k in BINARY_KEYS},
                dense_window_work=flows[0]['traffic'], flow=dict(control=flows[0], candidate=flows[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = compare(a.control, a.candidate)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('status', 'eligible', 'timing')}, indent=2))


if __name__ == '__main__':
    main()
