#!/usr/bin/env python3
"""Canonical HBM-data-DMA send to final O write ACK timing, contract v1.

This reporter supports the square D64 worker-cluster projection workload.
Legacy job-to-worker reports are inputs for shape attribution, never timing
fallbacks. Missing trace evidence fails instead of silently changing scope.
"""
import argparse
import json
from pathlib import Path

from analyze_resource_timeline import FIELDS, merged

CONTRACT = {
    'version': 'hbmdma_to_O_ack_v1',
    'workload': 'HBM X/gamma/weights -> RMSNorm -> QKV -> RoPE -> causal Attention -> HBM O',
    'start': 'earliest successful NoC send of a workload data read DMA across all workers',
    'end': 'latest successful final O DMA_WRITE_COMPLETE ACK across all output nodes',
    'excluded': 'pre-first-data-DMA setup; post-output-ACK notification/printing/exit',
    'duration': '(end_tick-start_tick)/ticks_per_reference_cycle; no inclusive +1',
    'reference_clock_hz': 1000000000,
    'initial_state': 'X/gamma/weights in HBM; no preloaded operands in arrays',
}


def contained(address, size, ranges):
    return any(lower <= address and address + size <= upper for lower, upper in ranges)


def summarize_events(events, inputs, outputs, factor, root_tick):
    if factor <= 0:
        raise ValueError('invalid timebase')
    first = None
    last_ack = 0
    writes = []
    output_bytes = 0
    for e in events:
        if e['domain'] != 'gm':
            continue
        address, size = e.get('address', -1), e['bytes']
        if e['resource'] == 'dma_read_transport' and contained(address, size, inputs):
            if e['ok'] != 1 or not e['submit_tick'] <= e['start_tick'] <= e['done_tick']:
                raise ValueError('invalid data DMA event')
            if first is None or e['start_tick'] < first['start_tick']:
                first = e
        elif e['resource'] == 'dma_write_transport' and contained(address, size, outputs):
            if e['ok'] != 1 or not e['submit_tick'] <= e['start_tick'] <= e['done_tick']:
                raise ValueError('invalid output DMA event')
            last_ack = max(last_ack, e['done_tick'])
            writes.append((address, address + size))
            output_bytes += size
    if first is None:
        raise ValueError('missing workload data DMA')
    if merged(writes) != merged(outputs):
        raise ValueError('final O output ACK coverage incomplete')
    if not first['start_tick'] < last_ack <= root_tick:
        raise ValueError('DMA/output/root ordering invalid')
    return {
        'status': 'PASS', 'contract': CONTRACT,
        'ticks_per_reference_cycle': factor,
        'first_data_dma': first,
        'first_data_dma_send_tick': first['start_tick'],
        'last_O_ack_tick': last_ack,
        'root_complete_tick': root_tick,
        'end_to_end_cycles': (last_ack - first['start_tick']) / factor,
        'dma_to_root_complete_cycles': (root_tick - first['start_tick']) / factor,
        'output_ack_to_root_cycles': (root_tick - last_ack) / factor,
        'initial_dma_admission_delay_cycles': (first['start_tick'] - first['submit_tick']) / factor,
        'output_ack_bytes': output_bytes,
        'output_covered_bytes': sum(b-a for a,b in merged(writes)),
    }


def data_ranges(shape, layout):
    sequence, dim = shape['query_length'], shape['head_dim']
    hq, hkv = shape['Hq'], shape['Hkv']
    if dim != 64 or sequence != shape['kv_length'] or sequence % 1024 or not shape['causal']:
        raise ValueError('unsupported workload for DMA contract v1')
    hidden, rows = hq * dim, sequence // 4
    norm_bytes = rows * hidden * 2
    weight_offset = (0x01100000 + norm_bytes + 0xfffff) & ~0xfffff
    weights_bytes = (hq + 2*hkv) * dim * hidden * 2
    inputs, outputs = [], []
    if [n['node'] for n in layout['nodes']] != [1, 2, 3, 4]:
        raise ValueError('unsupported HBM node ownership')
    for node in layout['nodes']:
        base = node['node'] * node['capacity_bytes']
        if node['capacity_bytes'] != 0x08000000:
            raise ValueError('HBM address ABI changed')
        inputs += [(base, base + norm_bytes),
                   (base + 0x01000000, base + 0x01000000 + hidden*2),
                   (base + 0x01100000, base + 0x01100000 + norm_bytes),
                   (base + weight_offset, base + weight_offset + weights_bytes),
                   (base + 0x07000000, base + 0x07000000 + sequence*dim*2)]
        for region in node['regions']:
            span = (base + region['begin'], base + region['end'])
            if region['name'] == 'O':
                outputs.append(span)
            elif region['name'] in ('Q_panels', 'K_panels', 'V_panels', 'Q', 'K', 'V'):
                inputs.append(span)
    if sum(b-a for a,b in outputs) != sequence * hq * dim * 2:
        raise ValueError('O shape/coverage contract differs')
    return inputs, outputs


def report(case):
    case = Path(case)
    projection = json.loads((case/'projection_e2e_result.json').read_text())
    layout = json.loads((case/'attention_hbm_layout.json').read_text())
    numerical = json.loads((case/'fused_attention_result.json').read_text())
    if any(d['status'] != 'PASS' for d in (projection, layout, numerical)):
        raise ValueError('workload/numerical verification failed')
    configs = list((case/'stats').rglob('run_config.env'))
    if len(configs) != 1:
        raise ValueError('missing resolved run configuration')
    config = dict(line.strip().split('=', 1) for line in configs[0].read_text().splitlines()
                  if '=' in line)
    if any(config.get(k) not in ('1GHz', '1.0GHz') for k in ('VANADIS_CPU_CLOCK','GOLEM_ARRAY_CLOCK')):
        raise ValueError('contract v1 requires native CPU/array clocks at 1GHz')
    inputs, outputs = data_ranges(projection['shape'], layout)
    logs = list((case/'logs').glob('attention-sst*.log'))
    if len(logs) != 1:
        raise ValueError('expected one merged log')
    periods, root_ticks = set(), []
    with logs[0].open(errors='replace') as stream:
        for line in stream:
            if '[RESOURCE_TIMEBASE]' in line:
                periods.add(int(dict(FIELDS.findall(line))['ticks_per_cycle']))
            elif '[ATTENTION_MILESTONE]' in line and 'stage=root_tensor_complete ' in line and 'status=done ' in line:
                root_ticks.append(int(dict(FIELDS.findall(line))['sst_tick']))
    if periods != {1000} or not root_ticks:
        raise ValueError('contract v1 requires verified 1GHz/1ps timebase and root completion')
    def events():
        sidecar = logs[0].with_suffix('.resources.jsonl')
        with (sidecar if sidecar.exists() else logs[0]).open(errors='replace') as stream:
            for line in stream:
                if '[RESOURCE_INTERVAL]' in line:
                    raw = dict(FIELDS.findall(line))
                    yield {k: v if k in ('domain', 'resource') else int(v) for k,v in raw.items()}
    result = summarize_events(events(), inputs, outputs, 1000, max(root_ticks))
    result.update(shape=projection['shape'], input_address_ranges=inputs,
                  output_address_ranges=outputs,
                  legacy_job_to_worker_complete_cycles=projection['end_to_end_cycles'],
                  artifact_root=str(case.resolve()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = report(args.case)
    output = args.output or args.case/'dma_e2e_result.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(f"HBM data DMA -> O ACK: {result['end_to_end_cycles']:,.2f} cycles; "
          f"DMA -> root: {result['dma_to_root_complete_cycles']:,.2f} cycles")


if __name__ == '__main__':
    main()
