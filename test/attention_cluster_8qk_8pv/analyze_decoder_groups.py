#!/usr/bin/env python3
"""Audit paired local FFN subset scheduling using actual array port counters."""
import argparse
import json
from pathlib import Path
import re

FIELDS = re.compile(r'(\w+)=(-?\d+)')


def counters(case):
    arrays, flows = {}, {}
    for line in (case / 'logs/attention-sst.log').open():
        if 'GOLEM_ARRAY_BUFFER_STATS' in line:
            values = {k: int(v) for k, v in FIELDS.findall(line)}
            assert values['core'] not in arrays
            arrays[values['core']] = values
        elif '[DECODER_GROUP_FLOW]' in line:
            values = {k: int(v) for k, v in FIELDS.findall(line)}
            assert values['core'] not in flows
            flows[values['core']] = values
    return arrays, flows


def analyze(root):
    baseline, candidate = 'residual16q4', 'groups2residual16q4'
    cases = {v: json.loads((root / v / 'result.json').read_text()) for v in (baseline, candidate)}
    # compare() checks builds, input hashes, hardware environment and SFU work.
    from run_decoder_micro import compare
    compared = compare(cases, baseline)
    before, before_flows = counters(root / baseline)
    after, after_flows = counters(root / candidate)
    workers = cases[baseline]['shape']['workers']
    hidden = cases[baseline]['shape']['hidden']
    tiles = 3 * (hidden // 64) * (4 * hidden // 64)
    results = {}
    for core in range(4, 4 + workers):
        a, b = before[core], after[core]
        assert before_flows[core]['groups'] == 1
        flow = after_flows[core]
        assert flow['groups'] == 2 and flow['launches'] == 2 * tiles
        assert flow['compute_array_cycles'] == tiles * 64 * 67
        assert flow['both_groups_compute_cycles'] > 0
        assert b['matrix_broadcast_requests'] - a['matrix_broadcast_requests'] == tiles
        assert b['matrix_broadcast_ingress_bytes'] - a['matrix_broadcast_ingress_bytes'] == tiles * 8192
        assert b['matrix_broadcast_transfer_cycles'] - a['matrix_broadcast_transfer_cycles'] == tiles * 37
        assert b['input_scatter_requests'] - a['input_scatter_requests'] == tiles
        assert b['input_scatter_transfer_cycles'] - a['input_scatter_transfer_cycles'] == tiles
        for field in ('matrix_broadcast_sink_bytes', 'input_scatter_bytes',
                      'output_scatter_requests', 'output_gather_requests',
                      'output_scatter_gather_bytes', 'output_scatter_gather_transfer_cycles'):
            assert a[field] == b[field], (core, field)
        assert a['matrix_broadcast_max_fanout'] == 64
        assert b['matrix_broadcast_max_fanout'] == 32
        assert a['input_scatter_max_destinations'] == 64
        assert b['input_scatter_max_destinations'] == 32
        results[core] = dict(
            before_array_buffer_service_cycles=a['transfer_cycles'],
            after_array_buffer_service_cycles=b['transfer_cycles'],
            extra_array_buffer_service_cycles=b['transfer_cycles'] - a['transfer_cycles'],
            extra_matrix_ingress_bytes=b['matrix_broadcast_ingress_bytes'] - a['matrix_broadcast_ingress_bytes'],
            matrix_sink_bytes_unchanged=b['matrix_broadcast_sink_bytes'],
            input_scatter_bytes_unchanged=b['input_scatter_bytes'],
            callback_array_cycles_unchanged=flow['compute_array_cycles'],
            both_groups_callback_active_cycles=flow['both_groups_compute_cycles'])
    for field in ('hbm_read_bytes', 'hbm_write_bytes'):
        assert cases[baseline]['dma_timing'][field] == cases[candidate]['dma_timing'][field]
    return dict(status='PASS', comparison=compared, array_port_audit=results,
                scope='Actual SST counters; callback-active overlap includes WCP completion delivery. '
                      'Buffer busy + callback-active time must not be added as disjoint phases.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['comparison'], indent=2))


if __name__ == '__main__':
    main()
