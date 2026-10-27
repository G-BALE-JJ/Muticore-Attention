#!/usr/bin/env python3
"""Analyze archived full prefill without rerunning SST.

Operation windows overlap across workers. Only the boundary table is additive.
Bounds retain archived work, ownership and phase barriers; they are necessary
bounds, not a feasible optimal schedule or a prediction of optimization speedup.
"""
import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics

NUMBERS = re.compile(r'(\w+)=(-?\d+)')
INTERVAL = re.compile(r'resource=(\w+) core=(\d+).*?start_tick=(\d+) done_tick=(\d+).*?bytes=(\d+).*?address=(\d+) client=(-?\d+)')
NAMES = {1: 'Wo', 2: 'residual1', 3: 'RMSNorm2', 4: 'gate',
         5: 'up', 6: 'SiLU', 7: 'multiply', 8: 'down', 9: 'residual2'}


def linear_bounds(k, n_tiles, rows):
    """One operand bank, 64x64 MVM; no measured queue/transport waits."""
    launches = (k // 64) * n_tiles
    weight_read = 2 * 17  # 8192 B split into two <=4096 B GM requests
    input_read = (rows * 128 + 255) // 256 + (rows * 128 + 4095) // 4096
    matrix = 33 + (rows - 1).bit_length()
    scatter = 1 + (rows * 128 + 255) // 256
    gather = scatter
    output_write = input_read
    return {
        'launches': launches,
        'MAC': launches * 66,
        'array_buffer_port': launches * (matrix + scatter) + n_tiles * gather,
        # This stronger bound additionally retains the existing callback order.
        # Weight DMA and input preparation/output DMA are deliberately omitted.
        'serial_state_machine': launches * (weight_read + matrix + input_read + scatter + 66)
                                + n_tiles * (gather + output_write),
    }


def job_bounds(job, resources):
    stage = job['stage']
    if stage in (1, 4, 5, 8, 11):
        if stage == 11:
            tiles = (job['hbm_read'] - 4096) // 8192 // 32
            b = linear_bounds(2048, tiles, 1)
        else:
            k, n = {1: (2048, 2048), 4: (2048, 8192),
                    5: (2048, 8192), 8: (8192, 2048)}[stage]
            b = linear_bounds(k, n // 64, 64)
        capacity = max(b['MAC'], b['array_buffer_port'])
        serial = b['serial_state_machine']
    else:
        # Current requests execute serially, including fill/drain per request.
        vector = {2: 16 * 515, 3: 16 * 1747, 6: 64 * 8217,
                  7: 64 * 515, 9: 16 * 515, 10: 16 * 1747}[stage]
        capacity = serial = vector
        b = {'vector_request_service': vector}
    ports = {port: resources.get(port, {}).get('cycles', 0)
             for port in ('local_read', 'local_write')}
    return {'resource_capacity': max(capacity, *ports.values()),
            'current_serial_order': max(serial, *ports.values()),
            'components': dict(b, **ports)}


def analyze(case, summary, scan_resources):
    archived = json.loads(summary.read_text())
    c = archived['cases']['formal_rank4']
    actual = json.loads((case / 'decoder_stack_result.json').read_text())
    if actual['shape'] != c['result']['shape'] or actual['embedding']['first_token_ids_dma_tick'] != c['result']['embedding']['first_token_ids_dma_tick'] or actual['lm_head']['last_logits_ack_tick'] != c['result']['lm_head']['last_logits_ack_tick']:
        raise ValueError('case differs from the canonical archived formal timing/shape')
    if c['result']['shape'] != {'S': 1024, 'hidden': 2048, 'intermediate': 8192, 'layers': 2}:
        raise ValueError('formulas currently require the archived formal shape')
    if c['provenance']['environment']['GOLEM_DIAGNOSTIC_ORACLE'] != 'none':
        raise ValueError('oracle service is not a baseline')
    breakdown = c['cycle_breakdown']
    total = sum(breakdown.values())
    assert total == archived['comparisons']['formal']['rank4_cycles']
    assert total == archived['comparisons']['formal']['rank8_cycles']
    log = next((case / 'logs').glob('attention-sst*.log'))
    jobs, sfu = [], {}
    for line in log.open(errors='replace'):
        if '[DECODER_OP] event=done' in line:
            j = {k: int(v) for k, v in NUMBERS.findall(line)}
            j['cycles'] = (j['tick'] - j['start_tick']) / 1000
            jobs.append(j)
        if 'GOLEM_SFU_HW_PIPELINE' in line:
            unit = re.search(r'unit=(\w+)', line).group(1)
            fields = {k: int(v) for k, v in NUMBERS.findall(line)}
            sfu[(fields['core'], unit)] = fields
    by_core = defaultdict(list)
    for j in jobs:
        by_core[j['core']].append(j)
    for js in by_core.values():
        js.sort(key=lambda j: j['start_tick'])
        for a, b in zip(js, js[1:]):
            assert a['tick'] <= b['start_tick'], 'same-worker jobs overlap'
    starts = {core: [j['start_tick'] for j in js] for core, js in by_core.items()}
    resource_totals = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'bytes': 0, 'cycles': 0., 'tiny128_count': 0}))
    node_payload = defaultdict(lambda: defaultdict(int))
    first_dma = c['result']['embedding']['first_token_ids_dma_tick']
    last_dma = c['result']['lm_head']['last_logits_ack_tick']
    if scan_resources:
        with log.with_suffix('.resources.jsonl').open(errors='replace') as stream:
            for line in stream:
                m = INTERVAL.search(line)
                if not m:
                    continue
                resource, core, start, end, size, address, client = m.groups()
                core, start, end, size, address, client = map(int, (core, start, end, size, address, client))
                node = address // (2 * 1024**3)
                if first_dma <= start and end <= last_dma and node in (1, 2, 3, 4) and resource in ('dma_read_transport', 'dma_write_transport'):
                    node_payload[node][resource] += size
                if core not in starts:
                    continue
                index = bisect_right(starts[core], start) - 1
                if index < 0:
                    continue
                j = by_core[core][index]
                if end > j['tick']:
                    continue
                key = (j['epoch'], j['stage'], core)
                # client=0: DMA prefetch/landing; client=1: explicit RoCC access.
                for name in (resource, f'{resource}.client{client}'):
                    r = resource_totals[key][name]
                    r['count'] += 1
                    r['bytes'] += size
                    r['cycles'] += (end - start) / 1000
                    r['tiny128_count'] += size == 128
        for j in jobs:
            if j['stage'] == 12:
                continue
            r = resource_totals[(j['epoch'], j['stage'], j['core'])]
            assert r['dma_read_transport']['bytes'] == j['hbm_read'], (j, 'read mismatch')
            assert r['dma_write_transport']['bytes'] == j['hbm_write'], (j, 'write mismatch')
    stages, chains = {}, {}
    for epoch in (1, 2):
        layer = [j for j in jobs if j['epoch'] == epoch and 1 <= j['stage'] <= 9]
        assert len(layer) == 16 * 9
        critical = max((j for j in layer if j['stage'] == 9), key=lambda j: j['tick'])['core']
        chain = [j for j in layer if j['core'] == critical]
        chain.sort(key=lambda j: j['stage'])
        chains[str(epoch)] = {
            'last_finishing_core': critical,
            'own_compute_and_dataflow_job_cycles': sum(j['cycles'] for j in chain),
            'inter_job_dispatch_cycles': sum((b['start_tick'] - a['tick']) / 1000 for a, b in zip(chain, chain[1:])),
            'dispatch_before_Wo_and_boundary_adjustment_cycles': breakdown[f'layer{epoch}_post_attention']
                - (chain[-1]['tick'] - chain[0]['start_tick']) / 1000,
        }
        for stage, name in NAMES.items():
            js = [j for j in layer if j['stage'] == stage]
            cj = next(j for j in js if j['core'] == critical)
            bounds = [job_bounds(j, resource_totals[(epoch, stage, j['core'])]) for j in js]
            stages[f'layer{epoch}.{name}'] = {
                'global_activity_window_cycles': (max(j['tick'] for j in js) - min(j['start_tick'] for j in js)) / 1000,
                'worker_duration_min': min(j['cycles'] for j in js),
                'worker_duration_median': statistics.median(j['cycles'] for j in js),
                'worker_duration_max': max(j['cycles'] for j in js),
                'last_finishing_worker_duration': cj['cycles'],
                'per_worker_capacity_bound_max': max(b['resource_capacity'] for b in bounds),
                'per_worker_serial_order_bound_max': max(b['current_serial_order'] for b in bounds),
                'critical_worker_resource_intervals': resource_totals[(epoch, stage, critical)],
                'HBM_read_bytes': sum(j['hbm_read'] for j in js),
                'HBM_write_bytes': sum(j['hbm_write'] for j in js),
            }
    post_bounds = {}
    for epoch in (1, 2):
        post_bounds[str(epoch)] = {}
        for kind in ('resource_capacity', 'current_serial_order'):
            post_bounds[str(epoch)][kind] = max(sum(
                job_bounds(j, resource_totals[(j['epoch'], j['stage'], core)])[kind]
                for j in js if j['epoch'] == epoch and 1 <= j['stage'] <= 9)
                for core, js in by_core.items())
    head_bounds = {kind: max(sum(job_bounds(j, resource_totals[(j['epoch'], 11, core)])[kind]
                              for j in js if j['stage'] == 11)
                            for core, js in by_core.items())
                   for kind in ('resource_capacity', 'current_serial_order')}
    # Eight 128-bit channels at 2500 MT/s -> 320 GB/s per data node.
    # Total read+write payload/peak excludes command, refresh and packet overhead.
    post_HBM = {str(epoch): sum(j['hbm_read'] + j['hbm_write'] for j in jobs
                               if j['epoch'] == epoch and 1 <= j['stage'] <= 9) / 1280
                for epoch in (1, 2)}
    # End-to-end stops at logits ACK, before the last completion-flag ACK.
    # Charge tensor bytes only here; never extend the bound beyond its endpoint.
    head_HBM = (actual['lm_head']['input_and_weight_dma_bytes'] + actual['lm_head']['output_ack_bytes']) / 1280
    for epoch in (1, 2):
        for kind in post_bounds[str(epoch)]:
            post_bounds[str(epoch)][kind] = max(post_bounds[str(epoch)][kind], post_HBM[str(epoch)])
    for kind in head_bounds:
        head_bounds[kind] = max(head_bounds[kind], head_HBM)
    # SiLU issues exactly 2*64*8192 reciprocal and /16 EXP tokens per worker.
    # Remaining EXP tokens are causal-attention work; layers have equal shapes.
    attention_exp = max(v['accepted_tokens'] - 2 * 64 * 8192 // 16
                        for (core, unit), v in sfu.items() if unit == 'exp') / 2
    assert attention_exp == 139264
    projection_buffer = 1536 * (39 + 33) + 48 * 33
    projection_serial = 1536 * (39 + 33 + 66) + 48 * 33
    lower = {}
    for kind in ('resource_capacity', 'current_serial_order'):
        projection = projection_buffer if kind == 'resource_capacity' else projection_serial
        lower[kind] = 2 * (projection + attention_exp) + sum(b[kind] for b in post_bounds.values()) + head_bounds[kind] + 27952
    return {
        'status': 'PASS', 'scope': archived['scope'], 'shape': c['result']['shape'],
        'total_cycles': total, 'clock_GHz': 1, 'simulated_ms': total / 1e6,
        'additive_boundary_stages': {k: {'cycles': v, 'percent': 100 * v / total} for k, v in breakdown.items()},
        'operation_profiles': stages, 'last_finishing_worker_chains': chains,
        'lower_bounds_cycles': lower, 'post_attention_bounds_cycles': post_bounds,
        'LM_head_bounds_cycles': head_bounds, 'attention_EXP_issue_per_layer': attention_exp,
        'HBM_peak_GB_per_second_per_node': 320,
        'post_attention_HBM_payload_bound_cycles': post_HBM,
        'LM_head_HBM_payload_bound_cycles': head_HBM,
        'whole_chain_DMA_payload_bytes_by_data_node': node_payload,
        'whole_chain_HBM_payload_bound_cycles': max((sum(v.values()) / 320 for v in node_payload.values()), default=0),
        'projection_array_resource_bound': projection_buffer, 'projection_serial_array_bound': projection_serial,
        'actual_to_bound': {k: total / v for k, v in lower.items()},
        'resource_sidecar_scanned': scan_resources,
        'cautions': [
            'Activity windows and max-worker durations overlap across workers and must not be summed.',
            'Only the last-finishing-worker chain adds within a post-attention phase; boundary adjustment excludes its final flag ACK.',
            'DMA transport intervals are measured RTT, not theoretical HBM bandwidth or isolated resource service.',
            'DMA write transport includes local-read prefetch; DMA landing duplicates local-write service. Do not double count.',
            'Bounds retain current phase barriers, job ownership, dense tile work and serial SFU request sizes.',
            'Serial-order bound additionally retains per-tile GM read -> matrix -> input read -> scatter -> MAC -> drain.',
            'HBM payload bound uses configured peak bandwidth; bank conflicts, refresh, read/write turnarounds, packet overhead and NoC cuts remain omitted.',
            'Bounds omit embedding service, control handoffs and some prepare/drain work; they are optimistic necessary bounds, not attainable optima.',
            'Changing mapping, fusion, traffic reuse or resource sizes requires recomputing bounds.',
            'Timing covers token DMA through last logits ACK; host greedy argmax and SST wall time are outside it.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--summary', type=Path, default=Path(__file__).parent / 'progress_results/2026-10-14/embedding_token_summary.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scan-resources', action='store_true', help='stream the large GM sidecar and validate all DMA byte totals')
    args = parser.parse_args()
    result = analyze(args.case, args.summary, args.scan_resources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'total_cycles', 'lower_bounds_cycles', 'actual_to_bound')}))


if __name__ == '__main__':
    main()
