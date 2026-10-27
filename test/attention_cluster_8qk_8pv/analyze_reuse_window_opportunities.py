#!/usr/bin/env python3
"""Offline evidence for remaining opportunities; no simulated speedup claims."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics

FIELDS = re.compile(r'(\w+)=(-?\d+)')
INTERVAL = re.compile(r'resource=(\w+) core=(\d+).*?start_tick=(\d+) done_tick=(\d+)'
                      r'.*?bytes=(\d+).*?address=(\d+) client=(-?\d+)')


def analyze(case):
    result = json.loads((case/'decoder_stack_result.json').read_text())
    assert result['status'] == 'PASS'
    assert result['shape'] == dict(S=1024, padded_S=1024, hidden=2048, intermediate=8192, layers=2)
    first = result['embedding']['first_token_ids_dma_tick']
    last = result['lm_head']['last_logits_ack_tick']
    total = (last-first)/1000
    log = next((case/'logs').glob('attention-sst*.log'))
    jobs, arrays, flows, prefetch = [], {}, {}, []
    for line in log.open():
        if '[DECODER_OP] event=done' in line:
            job = {k:int(v) for k,v in FIELDS.findall(line)}
            job['cycles'] = (job['tick']-job['start_tick'])/1000
            jobs.append(job)
        elif 'GOLEM_ARRAY_BUFFER_STATS' in line:
            f = {k:int(v) for k,v in FIELDS.findall(line)};arrays[f['core']] = f
        elif '[DECODER_GROUP_FLOW]' in line:
            f = {k:int(v) for k,v in FIELDS.findall(line)};flows[f['epoch'],f['core']] = f
        elif '[DECODER_PREFETCH_FLOW]' in line:
            prefetch.append({k:int(v) for k,v in FIELDS.findall(line)})
    by_core = defaultdict(list)
    for job in jobs:by_core[job['core']].append(job)
    for js in by_core.values():
        js.sort(key=lambda j:j['start_tick'])
        assert all(a['tick']<=b['start_tick'] for a,b in zip(js,js[1:]))
    starts = {c:[j['start_tick'] for j in js] for c,js in by_core.items()}
    services = defaultdict(lambda:defaultdict(lambda:dict(bytes=0,count=0,service_sum_cycles=0)))
    payload = Counter();weight_reads = Counter();resource_counts = Counter()
    weight_cores = defaultdict(set)
    for line in log.with_suffix('.resources.jsonl').open():
        m = INTERVAL.search(line)
        if not m:continue
        resource,core,start,end,size,address,client = m.groups()
        core,start,end,size,address,client = map(int,(core,start,end,size,address,client))
        resource_counts[resource] += 1
        node = address//(2*1024**3)
        if (resource in ('dma_read_transport','dma_write_transport') and size>=128
                and first<=start and end<=last and node in (1,2,3,4)):
            payload[node] += size
        if core not in starts:continue
        index = bisect_right(starts[core],start)-1
        if index<0:continue
        job = by_core[core][index]
        if end>job['tick']:continue
        d = services[job['epoch'],job['stage'],core][resource]
        d['bytes']+=size;d['count']+=1;d['service_sum_cycles']+=(end-start)/1000
        if resource=='dma_read_transport' and job['stage']==4 and size==8192:
            weight_reads[job['epoch'],address] += 1
            weight_cores[job['epoch'],address].add(core)
    for job in jobs:
        if job['stage']==12:continue
        s = services[job['epoch'],job['stage'],job['core']]
        assert s['dma_read_transport']['bytes']==job['hbm_read'], (job,'read mismatch')
        assert s['dma_write_transport']['bytes']==job['hbm_write'], (job,'write mismatch')
    phases = {}
    def phase(name,start,end):
        assert end>=start
        phases[name] = dict(cycles=(end-start)/1000,percent=100*(end-start)/(last-first))
    e = result['embedding'];d = result['dma_timing'];n=result['final_norm'];lm=result['lm_head']
    phase('embedding',first,e['last_embedding_output_ack_tick'])
    phase('embedding_handoff',e['last_embedding_output_ack_tick'],d['layers']['1']['first_input_dma_tick'])
    for epoch in (1,2):
        l=d['layers'][str(epoch)]
        phase(f'layer{epoch}_projection_attention',l['first_input_dma_tick'],l['last_attention_o_ack_tick'])
        phase(f'layer{epoch}_post_attention',l['last_attention_o_ack_tick'],l['last_y_ack_tick'])
        if epoch==1:phase('layer_handoff',l['last_y_ack_tick'],d['layers']['2']['first_input_dma_tick'])
    phase('final_norm_handoff',d['layers']['2']['last_y_ack_tick'],n['first_input_dma_tick'])
    phase('final_norm',n['first_input_dma_tick'],n['last_output_ack_tick'])
    phase('lm_handoff',n['last_output_ack_tick'],lm['first_input_dma_tick'])
    phase('lm',lm['first_input_dma_tick'],last)
    assert sum(p['cycles'] for p in phases.values())==total
    stages = {};ffns = []
    for epoch in (1,2):
        for stage in (1,2,3,4):
            js=[j for j in jobs if j['epoch']==epoch and j['stage']==stage]
            assert len(js)==16
            late=max(js,key=lambda j:j['tick']);durations=[j['cycles'] for j in js]
            s=dict(last_core=late['core'],duration_min=min(durations),duration_median=statistics.median(durations),
                   duration_max=max(durations),last_core_duration=late['cycles'],
                   last_core_resources=services[epoch,stage,late['core']],
                   completion_spread_cycles=(max(j['tick'] for j in js)-min(j['tick'] for j in js))/1000)
            stages[f'{epoch}.{stage}']=s
            if stage==4:
                for j in js:
                    f=flows[epoch,j['core']]
                    assert f['launches']==12288 and f['compute_array_cycles']==12288*64*67
                    ffns.append(dict(epoch=epoch,core=j['core'],cycles=j['cycles'],
                                     physical_mac_busy_percent=100*12288*66/j['cycles'],
                                     useful_mac_peak_percent=100*12288*64/j['cycles'],
                                     matrix_callback_cycles=f['program_cycles'],
                                     serial_program_scatter_mac_floor_cycles=12288*(39+33+66),
                                     gap_above_serial_floor=j['cycles']-12288*(39+33+66)))
    histogram=Counter(weight_reads.values())
    assert histogram=={16:24576},histogram
    assert all(cores==set(range(4,20)) for cores in weight_cores.values())
    macs = dict(qkv=2*1024*2048*(2048+2*512),wo=2*1024*2048*2048,
                ffn=2*3*1024*2048*8192,attention=2*2*32*64*1024*1025//2,lm=2048*128256)
    return dict(status='PASS',scope='offline analysis of completed PASS run; no simulation or scheduling changes',
                total_cycles=total,additive_phases=phases,stage_profiles=stages,ffn_workers=ffns,
                useful_macs=macs,useful_mac_peak_efficiency_percent=100*sum(macs.values())/(16*64*64*total),
                hbm_payload_bytes_by_node=dict(payload),hbm_current_payload_floor_cycles=max(payload.values())/320,
                hbm_payload_peak_percent=100*sum(payload.values())/(1280*total),
                ffn_weight_reads=dict(requests=sum(weight_reads.values()),unique_epoch_addresses=len(weight_reads),
                                      repetition_histogram=dict(histogram),
                                      current_bytes=sum(weight_reads.values())*8192,
                                      unique_bytes=len(weight_reads)*8192,
                                      optimistic_hbm_byte_saving=(sum(weight_reads.values())-len(weight_reads))*8192),
                array_port_counters=arrays,legacy_prefetch_counters=prefetch,resource_record_counts=dict(resource_counts),
                caveats=['Useful MAC efficiency is useful work/configured peak, not measured array busy-time union.',
                         'GM service sums and DMA RTT sums are not port busy-time unions or removable stalls.',
                         'Prefetch hit means consumed successfully; new window wait counters are not instrumented.',
                         'FFN weight reuse assumes a shared fetch/multicast scheme whose delivery/synchronization costs are not modeled.',
                         'Serial floor retains full-64 single-context matrix/scatter/MAC order, omits other costs; not a global hardware bound.',
                         'Legacy groups=2 field is independent of actual window groups=1; launches and compute work establish actual behavior.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=analyze(a.case);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:r[k] for k in ('status','total_cycles','additive_phases',
                     'useful_mac_peak_efficiency_percent','hbm_payload_peak_percent',
                     'hbm_current_payload_floor_cycles','ffn_weight_reads')},indent=2))


if __name__=='__main__':main()
