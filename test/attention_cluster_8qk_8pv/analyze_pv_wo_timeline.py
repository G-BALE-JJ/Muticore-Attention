#!/usr/bin/env python3
"""Profile the timed Wo input handoff without summing overlapping workers."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

from analyze_reuse_window_critical_path import analyze

FIELDS = re.compile(r'(\w+)=([^\s]+)')
TICKS_PER_CYCLE = 1000


def span(events):
    if not events:
        return dict(bytes=0, requests=0, span_cycles=0, queue_request_sum_cycles=0,
                    service_request_sum_cycles=0)
    if any(e['ok'] != 1 or e['done_tick'] < e['start_tick'] or
           e['start_tick'] < e['submit_tick'] for e in events):
        raise ValueError('failed or reversed timed GM event')
    return dict(bytes=sum(e['bytes'] for e in events), requests=len(events),
                first_submit_tick=min(e['submit_tick'] for e in events),
                last_ack_tick=max(e['done_tick'] for e in events),
                span_cycles=(max(e['done_tick'] for e in events)-min(e['submit_tick'] for e in events))/TICKS_PER_CYCLE,
                queue_request_sum_cycles=sum(e['start_tick']-e['submit_tick'] for e in events)/TICKS_PER_CYCLE,
                service_request_sum_cycles=sum(e['done_tick']-e['start_tick'] for e in events)/TICKS_PER_CYCLE)


def handoff(case):
    manifest=json.loads((case/'decoder_stack_inputs.json').read_text())
    result=json.loads((case/'decoder_stack_result.json').read_text())
    if result['status']!='PASS':
        raise ValueError('completed PASS result required')
    jobs={};locals_by_job=defaultdict(list);flows=defaultdict(list)
    for line in next((case/'logs').glob('attention-sst*.log')).open():
        if '[DECODER_OP] event=done ' in line:
            f=dict(FIELDS.findall(line))
            if f['stage']=='1':
                key=(int(f['epoch']),int(f['core']))
                if key in jobs:raise ValueError('duplicate Wo job')
                jobs[key]={k:int(v) for k,v in f.items() if k!='event'}
        elif '[GEMM_WINDOW_FLOW]' in line:
            f=dict(FIELDS.findall(line))
            phases={k:int(v) for k,v in f.items() if k.startswith('phase_')}
            if sum(phases.values())!=int(f['active_cycles']):
                raise ValueError('exclusive window phase accounting mismatch')
            flows[int(f['epoch']),int(f['core']),f['operator']].append(f)
    jobs_by_core=defaultdict(list)
    for key,job in jobs.items():jobs_by_core[key[1]].append((key,job))
    weights={};input_reads=defaultdict(list);output_writes=defaultdict(list)
    for line in next((case/'logs').glob('*.resources.jsonl')).open():
        if not any(f'resource={r} ' in line for r in ('local_read','local_write','dma_read_transport')):
            continue
        f=dict(FIELDS.findall(line));core=int(f['core'])
        if core not in jobs_by_core:continue
        e={k:int(v) for k,v in f.items() if k not in ('domain','resource')}
        selected=[(key,j) for key,j in jobs_by_core[core]
                  if j['start_tick']<=e['submit_tick'] and e['done_tick']<=j['tick']]
        if not selected:continue
        if len(selected)!=1:raise ValueError('overlapping Wo jobs on one worker')
        key,job=selected[0]
        if e['ok']!=1:raise ValueError('failed Wo resource request')
        if f['resource']=='dma_read_transport':
            layer=manifest['layout']['layers'][key[0]-1]
            stride=manifest['layout']['node_stride'];offset=e['address']%stride
            weight_bytes=manifest['hidden']*manifest['hidden']*2//4
            if layer['wo_weight']<=offset and offset+e['bytes']<=layer['wo_weight']+weight_bytes:
                weights[key]=min(weights.get(key,e['submit_tick']),e['submit_tick'])
        else:
            address=e['address']-core*0x200000
            # Keep only the two relevant payload arenas. Later B-slot reads in
            # the tail are excluded below by the first weight DMA boundary.
            if (0x1a0000<=address and address+e['bytes']<=0x1e0000) or (
                    0x10000<=address and address+e['bytes']<=0x50000):
                locals_by_job[key].append((f['resource'],address,e))
            if f['resource']=='local_read' and 0x1a0000<=address and address+e['bytes']<=0x1e0000:
                input_reads[key].append(e)
            if f['resource']=='local_write' and 0x70000<=address and address+e['bytes']<=0xb0000:
                output_writes[key].append(e)
    workers={}
    for key,job in sorted(jobs.items()):
        if key not in weights:raise ValueError('Wo weight DMA boundary missing')
        first_weight=weights[key]
        before=[(r,a,e) for r,a,e in locals_by_job[key] if e['done_tick']<=first_weight]
        input_setup_writes=[e for r,a,e in before if r=='local_write' and a<0x50000]
        relocation_reads=[e for r,a,e in before if r=='local_read' and a>=0x1a0000] if manifest.get('pv_wo_gm_handoff') else []
        relocation_writes=input_setup_writes if manifest.get('pv_wo_gm_handoff') else []
        worker=dict(epoch=key[0],core=key[1],wo_start_tick=job['start_tick'],wo_end_tick=job['tick'],
                    wo_cycles=(job['tick']-job['start_tick'])/TICKS_PER_CYCLE,
                    first_weight_submit_tick=first_weight,
                    wo_start_to_first_weight_cycles=(first_weight-job['start_tick'])/TICKS_PER_CYCLE,
                    relocation_reads=span(relocation_reads),relocation_writes=span(relocation_writes),
                    preweight_input_setup_writes=span(input_setup_writes),
                    tail_local_reads_during_wo=span(input_reads[key]),
                    retained_wo_output=span(output_writes[key]))
        workers[f'{key[0]}.{key[1]}']=worker
    operators={}
    for epoch,core,operator in sorted(flows):
        fs=flows[epoch,core,operator]
        operators[f'{epoch}.{core}.{operator}']={
            'groups':sorted({int(f['groups']) for f in fs}),
            'windows':len(fs),
            **{k:sum(int(f.get(k,0)) for f in fs) for k in fs[0]
               if k.startswith(('phase_','wait_')) or k=='active_cycles'}}
    return dict(status='PASS',case=str(case.resolve()),latency_unit='cycles',ticks_per_cycle=TICKS_PER_CYCLE,
                result_sha256=hashlib.sha256((case/'decoder_stack_result.json').read_bytes()).hexdigest(),
                analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                full_cycles=result['embedding']['end_to_end_cycles'],
                decoder_cycles=result['dma_timing']['end_to_end_cycles'],
                pv_wo_gm_handoff=bool(manifest.get('pv_wo_gm_handoff')),
                pv_wo_direct_input=bool(manifest.get('pv_wo_direct_input')),
                critical_path=analyze(case),workers=workers,operators=operators,
                relocation_read_bytes=sum(w['relocation_reads']['bytes'] for w in workers.values()),
                relocation_write_bytes=sum(w['relocation_writes']['bytes'] for w in workers.values()),
                notes=['All latency fields are cycles; raw timestamp fields ending in _tick are evidence only.',
                       'Request service/queue sums can overlap; they are not stage latency.',
                       'Tail reads after weight DMA may include B scratch in the legacy handoff layout.',
                       'Existing traces do not isolate per-payload NoC queue and wire service; no causal attribution is made from missing timestamps.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=handoff(a.case)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:r[k] for k in ('status','full_cycles','decoder_cycles','relocation_read_bytes','relocation_write_bytes')}))


if __name__=='__main__':main()
