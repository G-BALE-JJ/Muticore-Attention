#!/usr/bin/env python3
"""Account HBM transport once per completion, classified by frozen tensor layout."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

from analyze_reuse_window_critical_path import analyze

FIELDS=re.compile(r'(\w+)=([^\s]+)')
TRANSPORTS={'dma_read_transport':'read','dma_write_transport':'write'}


def category(name):
    if 'weight' in name or 'gamma' in name:return 'weights_and_coefficients'
    if name.endswith('done') or name=='embedding_tokens':return 'control_and_token_ids'
    if name=='lm_output':return 'logits'
    return 'activations_and_panels'


def transport_ledger(lines,layout,first,last):
    regions=sorted(layout['regions'],key=lambda r:r['offset'])
    starts=[r['offset'] for r in regions]
    if any(a['offset']+a['bytes']>b['offset'] for a,b in zip(regions,regions[1:])):
        raise ValueError('overlapping HBM layout regions')
    buckets={};seen=set();excluded=0;boundary_polls=[]
    for line in lines:
        if 'resource=dma_read_transport ' not in line and 'resource=dma_write_transport ' not in line:continue
        f=dict(FIELDS.findall(line));resource=f['resource'];direction=TRANSPORTS[resource]
        start=int(f['start_tick']);end=int(f['done_tick'])
        if end<first or start>last:excluded+=1;continue
        if int(f['ok'])!=1 or int(f['visible_tick'])!=end:raise ValueError('failed or unpublished transport')
        key=(int(f['core']),resource,int(f['request']))
        if key in seen:raise ValueError('duplicate transport completion')
        seen.add(key)
        address=int(f['address']);size=int(f['bytes']);node,offset=divmod(address,layout['node_stride'])
        if not 1<=node<=4 or size<=0:raise ValueError('invalid data-node transport')
        index=bisect_right(starts,offset)-1
        if index<0:raise ValueError('unmapped HBM transport')
        region=regions[index]
        if offset+size>region['offset']+region['bytes']:raise ValueError('transport crosses HBM region')
        if not first<=start<=end<=last:
            # Root completion polling is independent of tensor execution. Its
            # last read can ACK after the last logits ACK; never clip payload
            # bytes or silently discard a tensor crossing the timing edge.
            if int(f['core'])!=0 or direction!='read' or not region['name'].endswith('done'):
                raise ValueError('tensor transport crosses measurement boundary')
            boundary_polls.append(dict(region=region['name'],bytes=size,start_tick=start,done_tick=end))
            continue
        key=f"{region['layer']}.{region['name']}"
        if key not in buckets:
            buckets[key]=dict(name=region['name'],layer=region['layer'],category=category(region['name']),
                              read_bytes=0,write_bytes=0,read_requests=0,write_requests=0,
                              first_start_tick=start,last_ack_tick=end,read_workers=set(),write_workers=set())
        b=buckets[key];b[direction+'_bytes']+=size;b[direction+'_requests']+=1
        b['first_start_tick']=min(b['first_start_tick'],start);b['last_ack_tick']=max(b['last_ack_tick'],end)
        b[direction+'_workers'].add(int(f['core']))
    totals=defaultdict(lambda:dict(read_bytes=0,write_bytes=0,read_requests=0,write_requests=0))
    for b in buckets.values():
        for direction in ('read','write'):b[direction+'_workers']=sorted(b[direction+'_workers'])
        for k in totals[b['category']]:totals[b['category']][k]+=b[k]
    return dict(status='PASS',regions=buckets,categories=dict(totals),
                totals={k:sum(b[k] for b in buckets.values()) for k in ('read_bytes','write_bytes','read_requests','write_requests')},
                excluded_outside_timing=excluded,
                excluded_boundary_control_polls=boundary_polls,
                scope='All data-node DMA transport completions inside token-ID send to final logits ACK. Landing/local events excluded; job/window counters are not added.')


def audit(case):
    inputs=json.loads((case/'decoder_stack_inputs.json').read_text())
    result=json.loads((case/'decoder_stack_result.json').read_text())
    critical=analyze(case)
    trace=next((case/'logs').glob('*.resources.jsonl'))
    with trace.open() as stream:
        ledger=transport_ledger(stream,inputs['layout'],result['embedding']['first_token_ids_dma_tick'],result['lm_head']['last_logits_ack_tick'])
    # O is row-major on each of four HBM query bands. The optional handoff
    # replaces its entire write/read roundtrip with timed NoC and local GM.
    o=ledger['regions'].get('0.o',dict(read_bytes=0,write_bytes=0,read_workers=[],write_workers=[]))
    tensor_bytes=inputs['sequence']*inputs['hidden']*2*int(inputs['layers'])
    handoff=bool(inputs.get('pv_wo_gm_handoff'))
    expected=0 if handoff else tensor_bytes
    if o['read_bytes']!=expected or o['write_bytes']!=expected:
        raise ValueError('PV->Wo O traffic does not match the selected HBM/GM contract')
    if handoff and sum(v['pv_wo_gm_landing_bytes'] for v in result['dma_timing']['layers'].values())!=tensor_bytes:
        raise ValueError('PV->Wo GM landing does not cover all layers')
    direct=bool(inputs.get('pv_wo_direct_input'))
    direct_audit=result['resources']['optimizations'].get('pv_wo_direct_input',{})
    if direct and (not direct_audit.get('enabled') or direct_audit.get('status')!='PASS'):
        raise ValueError('direct input requires actual GM request and lease audit')
    return dict(status='PASS',case=str(case.resolve()),critical_path=critical,hbm=ledger,
                pv_to_wo=dict(status='PASS',gm_handoff_enabled=handoff,direct_input_enabled=direct,
                              actual_direct_input_audit=direct_audit,
                              write_bytes=o['write_bytes'],reread_bytes=o['read_bytes'],
                              maximum_removable_hbm_bytes=o['write_bytes']+o['read_bytes'],
                              eliminated_hbm_bytes=2*tensor_bytes if handoff else 0,
                              noc_payload_bytes=tensor_bytes if handoff else 0,
                              producers=o['write_workers'],consumers=o['read_workers'],
                              requires_cross_worker_redistribution=True,
                              note=(('Zero O HBM IO and zero relocation verified against timed GM requests.' if direct else
                                     'Zero O HBM IO verified against complete timed GM landings; local relocation is charged.') if handoff else
                                    'Bytes are a traffic opportunity, not predicted latency savings. Bounded GM landing and ACK ownership are required.')),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                result_sha256=hashlib.sha256((case/'decoder_stack_result.json').read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.case);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(dict(status=r['status'],total_cycles=r['critical_path']['total_cycles'],
                          hbm=r['hbm']['totals'],pv_to_wo=r['pv_to_wo']),indent=2))


if __name__=='__main__':main()
