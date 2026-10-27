#!/usr/bin/env python3
"""Matched full-stack audit for software-only FFN array subset interleaving."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from run_decoder_layer import read_bands
from verify_decoder_dataflow_resources import verify

FIELDS = re.compile(r'(\w+)=(-?\d+)')


def read(case):
    log = next((case / 'logs').glob('attention-sst*.log'))
    arrays, groups, sfu = {}, {}, {}
    for line in log.open():
        if not any(s in line for s in ('GOLEM_ARRAY_BUFFER_STATS', '[DECODER_GROUP_FLOW]', 'GOLEM_SFU_HW_PIPELINE')):
            continue
        fields = {k: int(v) for k, v in FIELDS.findall(line)}
        if 'GOLEM_ARRAY_BUFFER_STATS' in line:
            key = fields['core']
            assert key not in arrays
            arrays[key] = fields
        elif '[DECODER_GROUP_FLOW]' in line:
            key = (fields['core'], fields['epoch'])
            assert key not in groups
            groups[key] = fields
        else:
            key = (fields['core'], re.search(r'unit=(\w+)', line).group(1))
            assert key not in sfu
            sfu[key] = fields['accepted_tokens']
    return arrays, groups, sfu


def analyze(baseline, candidate):
    cases = [baseline, candidate]
    manifests = [json.loads((c / 'decoder_stack_inputs.json').read_text()) for c in cases]
    provenance = [json.loads((c / 'decoder_stack_provenance.json').read_text()) for c in cases]
    results = [json.loads((c / 'decoder_stack_result.json').read_text()) for c in cases]
    assert all(r['status'] == 'PASS' for r in results)
    assert results[0]['shape'] == results[1]['shape'] == dict(S=1024, hidden=2048, intermediate=8192, layers=2)
    for key in ('source_sha256', 'guest_sha256', 'sst_library_sha256'):
        assert provenance[0][key] == provenance[1][key], key
    ignored = {'GOLEM_DECODER_ARRAY_GROUPS', 'GOLEM_ARTIFACT_ROOT', 'GOLEM_DECODER_STACK_MANIFEST',
               'GOLEM_MPI_ARGS', 'GOLEM_MPI_RANK_LOG_ROOT'}
    environments = [p['environment'] for p in provenance]
    differences = {k: [e.get(k) for e in environments] for k in set(environments[0]) | set(environments[1])
                   if environments[0].get(k) != environments[1].get(k)}
    assert set(differences) <= ignored, differences
    assert [e['GOLEM_DECODER_ARRAY_GROUPS'] for e in environments] == ['1', '2']
    def normalized(m):
        return dict(m, preloads=[{k: v for k, v in p.items() if k != 'path'} for p in m['preloads']])
    assert normalized(manifests[0]) == normalized(manifests[1]), 'input/layout mismatch'
    layout = manifests[0]['layout']
    targets = {f'layer{i}': l['output'] for i, l in enumerate(layout['layers'], 1)}
    targets.update(final_norm=layout['final_norm']['output'], embedding=layout['embedding']['output'])
    targets.update({k: layout['shared'][k] for k in ('wo', 'h1', 'norm2')})
    equality = {}
    for name, offset in targets.items():
        tensors = [read_bands(c, offset, 2048) for c in cases]
        assert np.array_equal(tensors[0].view('<u2'), tensors[1].view('<u2')), name
        equality[name] = dict(status='PASS', elements=int(tensors[0].size),
                              sha256=hashlib.sha256(tensors[0].tobytes()).hexdigest())
    logits = [np.load(c / 'last_token_logits.npy') for c in cases]
    assert np.array_equal(logits[0].view('<u2'), logits[1].view('<u2'))
    equality['logits'] = dict(status='PASS', elements=int(logits[0].size),
                              sha256=hashlib.sha256(logits[0].tobytes()).hexdigest())
    tokens = [r['lm_head']['next_token']['token_id'] for r in results]
    assert tokens[0] == tokens[1]
    parsed = [read(c) for c in cases]
    assert parsed[0][2] == parsed[1][2], 'SFU work differs'
    assert set(parsed[0][1]) == set(parsed[1][1]) == {(core, epoch) for core in range(4, 20) for epoch in (1, 2)}
    for flow in parsed[0][1].values():
        assert flow['groups'] == 1 and flow['launches'] == 0
    for flow in parsed[1][1].values():
        assert flow['groups'] == 2 and flow['launches'] == 24576
        assert flow['compute_array_cycles'] == 12288 * 64 * 67
        assert flow['program_cycles'] == 24576 * 39
        assert flow['both_groups_compute_cycles'] > 0
    array_deltas = {}
    for core in range(4, 20):
        a, b = parsed[0][0][core], parsed[1][0][core]
        tiles = 2 * 12288
        assert b['matrix_broadcast_requests'] - a['matrix_broadcast_requests'] == tiles
        assert b['matrix_broadcast_ingress_bytes'] - a['matrix_broadcast_ingress_bytes'] == tiles * 8192
        assert b['matrix_broadcast_transfer_cycles'] - a['matrix_broadcast_transfer_cycles'] == tiles * 37
        assert b['input_scatter_requests'] - a['input_scatter_requests'] == tiles
        assert b['input_scatter_transfer_cycles'] - a['input_scatter_transfer_cycles'] == tiles
        for field in ('matrix_broadcast_sink_bytes', 'input_scatter_bytes', 'output_scatter_requests',
                      'output_gather_requests', 'output_scatter_gather_bytes', 'output_scatter_gather_transfer_cycles'):
            assert a[field] == b[field], (core, field)
        array_deltas[core] = dict(extra_buffer_service_cycles=b['transfer_cycles'] - a['transfer_cycles'],
                                  extra_matrix_ingress_bytes=b['matrix_broadcast_ingress_bytes'] - a['matrix_broadcast_ingress_bytes'])
    audits = [verify(c) for c in cases]
    assert all(a['status'] == 'PASS' and a['verified_decoder_jobs'] == 416 for a in audits)
    assert audits[0]['by_epoch_stage'] == audits[1]['by_epoch_stage'], 'DMA/GM traffic differs'
    cycles = [r['embedding']['end_to_end_cycles'] for r in results]
    return dict(status='PASS', baseline=str(baseline), candidate=str(candidate),
                baseline_cycles=cycles[0], candidate_cycles=cycles[1], saved_cycles=cycles[0] - cycles[1],
                saved_percent=100 * (1 - cycles[1] / cycles[0]), fp16_bitwise_equality=equality,
                greedy_token=tokens[0], environment_differences=differences,
                array_deltas=array_deltas, baseline_resources=audits[0], candidate_resources=audits[1],
                SFU_accepted_tokens_equal=True, group_flows=[list(p[1].values()) for p in parsed],
                timing_scope='first token-ID DMA send to final logits ACK; host argmax excluded')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.baseline, args.candidate)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'baseline_cycles', 'candidate_cycles', 'saved_cycles', 'saved_percent')}, indent=2))


if __name__ == '__main__':
    main()
