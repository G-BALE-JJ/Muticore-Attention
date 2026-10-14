#!/usr/bin/env python3
"""Estimate Norm overlap from measured batches; does not simulate a new plan."""
import argparse
import json
import re
from pathlib import Path

NORM = re.compile(r'\[SFU_RMSNORM\] core=(\d+) tag=(\d+) rows=(\d+) cols=(\d+) issue_tick=(\d+) vector_ready_tick=(\d+) complete_tick=(\d+) elapsed_cycles=(\d+) vector_cycles=(\d+) status=0')


def analyze(root):
    result = {'kind': 'analysis_only', 'timebase': '1GHz; 1000 SST ticks per cycle',
              'assumptions': ['same 256-row weight reuse and per-row arithmetic order',
                              'one existing Norm batch and unchanged projection snapshots',
                              'spare GM regions require D64, hidden2048, fused Norm, packed input, no fused RoPE',
                              'estimates exclude extra shared-port/DMA contention and controller overhead'],
              'cases': []}
    for sequence in (1024, 2048):
        case = root / f's{sequence}'
        pipeline = json.loads((case / 'projection_e2e_result.json').read_text())
        logs = list((case / 'logs').glob('attention*.log'))
        if len(logs) != 1:
            raise ValueError('expected one SST log')
        events = [tuple(map(int, e)) for e in NORM.findall(logs[0].read_text())]
        shape = pipeline['shape']
        if (shape['Hq'], shape['Hkv'], shape['head_dim']) != (32, 8, 64):
            raise ValueError('analysis requires measured 32:8/D64 profile')
        manager_data = {}
        for manager in range(4):
            rows = sorted((e for e in events if e[0] == manager), key=lambda e: e[1])
            if len(rows) != sequence // 16 or any(e[2:4] != (4, 2048) for e in rows):
                raise ValueError('incomplete or unsupported Norm batches')
            blocks = []
            for first in range(0, sequence // 4, 256):
                batch = [e for e in rows if first <= e[1] < first + 256]
                if [e[1] for e in batch] != list(range(first, first + 256, 4)):
                    raise ValueError('Norm row coverage mismatch')
                start = batch[0][4]
                ready = {str(count): (batch[count // 4 - 1][6] - start) // 1000
                         for count in (64, 128, 192, 256)}
                blocks.append({'first_row': first, 'start_cycle': start // 1000,
                               'ready_prefix_cycles': ready,
                               'vector_cycles': sum(e[8] for e in batch)})
            partition = pipeline['stages']['projection']['cycle_breakdown_by_manager'][str(manager)]
            norm_span = sum(b['ready_prefix_cycles']['256'] for b in blocks)
            norm_vector = sum(b['vector_cycles'] for b in blocks)
            optimistic = {}
            for prefix in (64, 128):
                hidden = sum(b['ready_prefix_cycles'][str(prefix)] for b in blocks[1:])
                optimistic[str(prefix)] = {'hidden_norm_cycles_estimate': hidden,
                    'e2e_cycles_if_no_new_contention': pipeline['end_to_end_cycles'] - hidden,
                    'e2e_reduction_percent_estimate': 100 * hidden / pipeline['end_to_end_cycles']}
            pairs = (sequence // 4 // 64) * 48 * 16
            projection_partition = sum(partition[k] for k in
                ('load_input', 'load_weight', 'load_weight_dma', 'program_weight',
                 'program_weight_ready', 'program_input', 'launch', 'wait_input_prefetch',
                 'read_output', 'restore_output', 'restore_output_ready', 'prepare_output',
                 'stage_partial', 'write_output', 'write_drain', 'write_completion'))
            projection_partition += partition['local_weight_read'] + partition['local_input_read']
            setup = partition['local_weight_read'] + partition['program_weight'] + partition['program_weight_ready']
            manager_data[str(manager)] = {'blocks': blocks, 'norm_total_span_cycles': norm_span,
                'norm_vector_cycles': norm_vector, 'norm_nonvector_span_cycles': norm_span - norm_vector,
                'projection_controller_cycles_per_pair_average': projection_partition / pairs,
                'extra_weight_setup_if_128row_reuse_blocks_current_schedule': setup,
                'cross_block_prefix_estimates': optimistic}
        result['cases'].append({'sequence': sequence, 'measured_e2e_cycles': pipeline['end_to_end_cycles'],
                               'by_manager': manager_data})
    # Offsets relative to each manager GM, for the selected scratch=GM+0x8000 plan.
    intervals = [('metadata', 0, 0x8000), ('future_prefix_A', 0x8000, 0x48000),
                 ('norm_raw', 0x48000, 0x4c000), ('gamma', 0x58000, 0x59000),
                 ('D64_weight_cache', 0x68000, 0xa8000),
                 ('future_prefix_B', 0xa8000, 0xe8000),
                 ('reserved_partial_region', 0xe8000, 0xf0000),
                 ('current_norm_input', 0xf0000, 0x1f0000),
                 ('existing_snapshot_budget', 0x1f0000, 0x1f8000)]
    if any(a[2] > b[1] for a, b in zip(intervals, intervals[1:])) or intervals[-1][2] > 0x200000 - 64:
        raise ValueError('proposed scratch allocation overlaps or exceeds existing GM')
    result['proposed_GM_intervals'] = [{'name': name, 'begin': hex(begin), 'end_exclusive': hex(end),
                                      'bytes': end - begin} for name, begin, end in intervals]
    result['allocation_scope'] = 'static selected-path check, not an implemented dynamic lifetime verifier'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.artifact_root)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for case in result['cases']:
        m = case['by_manager']['0']
        print(f"S{case['sequence']}: Norm {m['norm_total_span_cycles']:,}, "
              f"vector {m['norm_vector_cycles']:,}; "
              f"128-row prefix estimate {m['cross_block_prefix_estimates']['128']}")
