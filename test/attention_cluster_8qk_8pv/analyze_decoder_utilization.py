#!/usr/bin/env python3
"""Offline utilization for the archived two-layer, fused-down/residual prefill.

No simulation or hardware changes. Bounds are necessary resource bounds, not
predictions of achievable latency. Transport RTT is never counted as port busy.
"""
import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path
import re

NUMBERS = re.compile(r'(\w+)=(-?\d+)')
INTERVAL = re.compile(
    r'resource=(\w+) core=(\d+).*?start_tick=(\d+) done_tick=(\d+)'
    r'.*?bytes=(\d+).*?address=(\d+) client=(-?\d+)')


def analyze(case, archive):
    result = json.loads((case / 'decoder_stack_result.json').read_text())
    archived = json.loads((archive / 'stack_fused_decoder_stack_result.json').read_text())
    assert result == archived, 'only the matched archived run is supported'
    assert result['shape'] == dict(S=1024, hidden=2048, intermediate=8192, layers=2)
    contract = json.loads((archive / 'hardware_contract.json').read_text())
    assert contract['workers'] == 16 and contract['arrays_per_worker'] == 64
    assert contract['operand_contexts'] == 1 and not contract['hardware_changed']
    first = result['embedding']['first_token_ids_dma_tick']
    last = result['lm_head']['last_logits_ack_tick']
    total = (last - first) / 1000
    phases = json.loads((archive / 'phase_comparison.json').read_text())['phases']['stack_fused']
    assert sum(phases.values()) == total
    flows = json.loads((archive / 'stack_fused_weight_analysis.json').read_text())['per_worker_flows']
    log = next((case / 'logs').glob('attention-sst*.log'))
    jobs, arrays, sfu = [], {}, {}
    for line in log.open(errors='replace'):
        if '[DECODER_OP] event=done' in line:
            job = {k: int(v) for k, v in NUMBERS.findall(line)}
            job['duration_cycles'] = (job['tick'] - job['start_tick']) / 1000
            jobs.append(job)
        elif 'GOLEM_ARRAY_BUFFER_STATS' in line:
            fields = {k: int(v) for k, v in NUMBERS.findall(line)}
            arrays[fields['core']] = fields
        elif 'GOLEM_SFU_HW_PIPELINE' in line:
            fields = {k: int(v) for k, v in NUMBERS.findall(line)}
            unit = re.search(r'unit=(\w+)', line).group(1)
            sfu[(fields['core'], unit)] = fields
    assert len(jobs) == 416 and set(arrays) == set(range(4, 20))
    by_core = defaultdict(list)
    for job in jobs:
        by_core[job['core']].append(job)
    for core_jobs in by_core.values():
        core_jobs.sort(key=lambda j: j['start_tick'])
        assert all(a['tick'] <= b['start_tick'] for a, b in zip(core_jobs, core_jobs[1:]))
    starts = {c: [j['start_tick'] for j in js] for c, js in by_core.items()}
    services = defaultdict(lambda: defaultdict(lambda: dict(bytes=0, cycles=0., count=0)))
    ports = defaultdict(lambda: defaultdict(float))
    port_ends = {}
    payload = defaultdict(lambda: defaultdict(int))
    tiny_control = defaultdict(int)
    # Each local_read/local_write is service on its respective single GM port.
    # DMA transport is one payload record per request, not backend command bytes.
    with log.with_suffix('.resources.jsonl').open(errors='replace') as stream:
        for line in stream:
            match = INTERVAL.search(line)
            if not match:
                continue
            resource, core, start, end, size, address, client = match.groups()
            core, start, end, size, address, client = map(int, (core, start, end, size, address, client))
            clipped = max(0, min(end, last) - max(start, first))
            if resource in ('local_read', 'local_write') and clipped:
                key = (core, resource)
                assert start >= port_ends.get(key, 0), 'port intervals overlap or are unordered'
                port_ends[key] = end
                ports[core][resource] += clipped / 1000
            node = address // (2 * 1024**3)
            if resource in ('dma_read_transport', 'dma_write_transport') and first <= start and end <= last and node in (1, 2, 3, 4):
                if size < 128:
                    tiny_control[resource] += size
                else:
                    payload[node][resource] += size
            if core not in starts:
                continue
            index = bisect_right(starts[core], start) - 1
            if index < 0:
                continue
            job = by_core[core][index]
            if end > job['tick']:
                continue
            value = services[(job['epoch'], job['stage'], core)][resource]
            value['bytes'] += size
            value['cycles'] += (end - start) / 1000
            value['count'] += 1
    for job in jobs:
        resources = services[(job['epoch'], job['stage'], job['core'])]
        assert resources['dma_read_transport']['bytes'] == job['hbm_read']
        assert resources['dma_write_transport']['bytes'] == job['hbm_write']
    # Useful dense MACs, excluding masked causal pairs and non-matmul SFU work.
    S, H, I, layers, vocab = 1024, 2048, 8192, 2, 128256
    macs = dict(QKV=layers*S*H*(H+2*512), Wo=layers*S*H*H,
                FFN=layers*3*S*H*I, causal_attention=layers*2*32*64*S*(S+1)//2,
                LM_head=H*vocab)
    peak_mac = 16 * 64 * 64  # one MAC per output CU per cycle
    compute_bound = sum(macs.values()) / peak_mac
    payload_bytes = sum(sum(v.values()) for v in payload.values())
    node_bounds = {n: sum(v.values()) / 320 for n, v in payload.items()}
    hbm_bound = max(node_bounds.values())
    ffns, critical_chains = [], {}
    for epoch in (1, 2):
        layer_jobs = [j for j in jobs if j['epoch'] == epoch and 1 <= j['stage'] <= 4]
        critical = max((j for j in layer_jobs if j['stage'] == 4), key=lambda j: j['tick'])['core']
        chain = sorted((j for j in layer_jobs if j['core'] == critical), key=lambda j: j['stage'])
        critical_chains[epoch] = dict(core=critical, jobs=chain)
        for job in layer_jobs:
            if job['stage'] != 4:
                continue
            assert job['op'] == 9
            flow = next(f for f in flows if f['core'] == job['core'] and f['epoch'] == epoch)
            launches = flow['requests']
            assert launches == 12288 and flow['compute_sum'] == launches * 67
            assert flow['program_sum'] == launches * 40
            duration = job['duration_cycles']
            resources = services[(epoch, 4, job['core'])]
            # No gather/restore, partial spills, control, or GM waits charged.
            # Each tile retains matrix -> scatter -> MAC order, on the same
            # 64 arrays with a single operand context; these cannot overlap.
            ordered_floor = launches * (39 + 33 + 66)
            ffns.append(dict(
                core=job['core'], epoch=epoch, last_finishing_worker=job['core'] == critical,
                duration_cycles=duration, launches=launches,
                physical_MAC_cycles=launches*66,
                MAC_temporal_busy_percent=100*launches*66/duration,
                useful_MAC_peak_percent=100*(3*64*H*I)/(64*64*duration),
                matrix_program_callback_cycles=flow['program_sum'],
                input_scatter_service_cycles=launches*33,
                weight_foreground_wait_cycles=flow['wait_total'],
                current_order_service_floor_cycles=ordered_floor,
                gap_above_current_order_floor_cycles=duration-ordered_floor,
                gm_service=resources,
                reciprocal_issue_capacity_percent=100*524288/duration))
    # Retain per-worker sequential Wo -> Add1 -> Norm2 -> fused FFN order.
    # Omit upstream projection/attention/embedding altogether to keep this a
    # conservative necessary bound independent of their internal schedules.
    post_floor = 1024*(39+33+66) + 16*515 + 16*1747 + 12288*(39+33+66)
    head_hbm_floor = (result['lm_head']['input_and_weight_dma_bytes'] + result['lm_head']['output_ack_bytes']) / 1280
    retained_order_floor = 2*post_floor + 16*1747 + head_hbm_floor
    array_util = {
        c: dict(buffer_port_service_cycles=v['transfer_cycles'],
                buffer_port_busy_percent=100*v['transfer_cycles']/total,
                matrix_program_service_cycles=v['matrix_broadcast_transfer_cycles'],
                input_scatter_service_cycles=v['input_scatter_transfer_cycles'],
                gm_port_busy_percent={k: 100*t/total for k, t in ports[c].items()})
        for c, v in arrays.items()}
    return dict(
        status='PASS', shape=result['shape'], total_cycles=total, simulated_ms=total/1e6,
        additive_phases={k: dict(cycles=v, percent=100*v/total) for k, v in phases.items()},
        useful_MACs_by_operation=macs, peak_MACs_per_cycle=peak_mac,
        useful_MAC_peak_utilization_percent=100*compute_bound/total,
        lower_bounds_cycles=dict(useful_MAC_only=compute_bound,
                                 current_payload_HBM_peak=hbm_bound,
                                 retained_post_attention_order_omitting_upstream=retained_order_floor),
        actual_to_bound=dict(MAC_only=total/compute_bound, current_payload_HBM_peak=total/hbm_bound,
                             retained_post_attention_order=total/retained_order_floor),
        tensor_DMA_bytes_by_node=dict(payload), sub128_control_DMA_bytes=dict(tiny_control),
        HBM_payload_GB_per_second=payload_bytes/total,
        HBM_payload_peak_percent=100*payload_bytes/(total*1280),
        HBM_payload_peak_percent_by_node={n: 100*v/total for n, v in node_bounds.items()},
        per_worker_resource_utilization=array_util, FFN_worker_profiles=ffns,
        last_finishing_worker_chains=critical_chains,
        LM_head_active_arrays_per_worker=1, LM_head_mapping_spatial_utilization_percent=100/64,
        scope=[
            'Two synthetic dense-work decoder layers, embedding, final norm and last-token LM head; not a full checkpoint inference.',
            'Endpoint: first token-ID DMA send to final logits ACK; 1000 ticks/cycle, 1 GHz.',
            'MAC utilization is useful dense mathematical work / configured peak, not measured busy-time union; masked attention and padding excluded.',
            'HBM payload bound retains current DMA traffic; reuse/fusion changes it. 320 GB/s/node, four nodes. Omits sub128-byte controls and protocol amplification; controls of at least 128 bytes may be included.',
            'A request crossing the endpoint is omitted from payload; local port service is clipped. Transport RTT is not resource busy time.',
            'Array-buffer counters cover all modeled array work; initialization/host preload does no array work. GM ports are clipped to the endpoint.',
            'FFN MAC busy is 66 cycles per accepted launch; compute callback counter is 67 and matrix callback counter 40 versus physical 39.',
            'Reciprocal utilization is accepted-token issue demand / single-lane capacity, not a measure of queueing or pipeline occupancy.',
            'The retained-order floor keeps current sequential Wo/Add1/Norm2/FFN ownership and single-context matrix/scatter/MAC order. It omits all upstream work and many output/prepare costs.',
            'Bounds are optimistic necessities; ratios do not predict obtainable speedup. Further software scheduling may change the retained-order bound.',
            'The legacy run_summary.csv GEMM utilization and HBM utilization use different work/time/roofline scopes and are not whole-prefill utilization.',
        ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.case, args.archive)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'total_cycles', 'lower_bounds_cycles',
          'useful_MAC_peak_utilization_percent', 'HBM_payload_peak_percent')}, indent=2))


if __name__ == '__main__':
    main()
