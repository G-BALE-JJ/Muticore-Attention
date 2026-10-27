"""Audit direct Wo input against actual GM requests and consumer leases."""
from collections import defaultdict
import json
from pathlib import Path
import re

FIELDS=re.compile(r'(\w+)=([^\s]+)')
GM_STRIDE=0x200000
LANDING=0x1a0000
TILE_BYTES=64*2048*2
WEIGHTS=0x10000
WEIGHT_BYTES=4*8192


def audit_requests(states, events):
    """States are already matched to complete production Wo job lifetimes."""
    totals=defaultdict(int)
    for state in states.values():
        state['a_reads']=defaultdict(int);state['last_a_ack']=0;state['b_landing_bytes']=0
    for resource,e in events:
        if resource not in ('local_read','local_write','dma_read_landing'):continue
        state=next((s for s in states.values() if s['core']==e['core'] and
                    s['start_tick']<=e['submit_tick']<=s['end_tick']),None)
        if state is None:continue
        if e['ok']!=1 or e['done_tick']>state['end_tick'] or e['done_tick']<e['start_tick']:
            raise ValueError('direct Wo resource failed or escaped its lifetime')
        offset=e['address']-state['core']*GM_STRIDE;size=e['bytes']
        overlaps_landing=offset<LANDING+TILE_BYTES and LANDING<offset+size
        if resource=='local_read' and overlaps_landing:
            if not LANDING<=offset or offset+size>LANDING+TILE_BYTES or size!=4096 or offset%4096:
                raise ValueError('unexpected direct Wo input request bounds')
            state['a_reads'][offset]+=1
            state['last_a_ack']=max(state['last_a_ack'],e['done_tick'])
            totals['direct_a_read_bytes']+=size
        elif overlaps_landing:
            raise ValueError('live direct Wo landing overwritten')
        if resource=='local_write' and e.get('client')==1 and WEIGHTS<=offset<WEIGHTS+TILE_BYTES:
            raise ValueError('direct Wo still relocates its input')
        if resource=='dma_read_landing':
            if not WEIGHTS<=offset or offset+size>WEIGHTS+WEIGHT_BYTES:
                raise ValueError('direct Wo weight DMA uses an unexpected scratch range')
            state['b_landing_bytes']+=size
            totals['weight_landing_bytes']+=size
    for state in states.values():
        expected={LANDING+i:state['n_tiles'] for i in range(0,TILE_BYTES,4096)}
        if dict(state['a_reads'])!=expected:
            raise ValueError('direct Wo input read coverage missing or duplicated')
        if state['last_a_ack']>state['release_tick']:
            raise ValueError('direct Wo input released before last read ACK')
        expected_b=state['n_tiles']*(2048//64)*8192
        if state['b_landing_bytes']!=expected_b:
            raise ValueError('direct Wo dense weight landing coverage mismatch')
    return dict(status='PASS',jobs=len(states),relocation_read_bytes=0,relocation_write_bytes=0,
                **totals,scope='Actual timed GM A reads, immutable landing, disjoint B DMA and lease-release boundaries.')


def verify(case,manifest,jobs):
    log=next((case/'logs').glob('attention-sst*.log'))
    begins={};releases={};handoffs={}
    for line in log.open():
        tag=next((t for t in ('[PV_WO_DIRECT_INPUT]','[PV_WO_HANDOFF]') if t in line),None)
        if tag is None:continue
        f=dict(FIELDS.findall(line));key=(int(f['epoch']),int(f['core']))
        target=handoffs if tag=='[PV_WO_HANDOFF]' else begins if f.get('event')=='begin' else releases
        if key in target:raise ValueError('duplicate PV-Wo direct lifecycle record')
        target[key]=f
    if not manifest.get('pv_wo_direct_input'):
        if begins or releases:raise ValueError('direct Wo executed without matching manifest')
        return dict(status='PASS',enabled=False)
    if not manifest.get('pv_wo_gm_handoff') or manifest['sequence']!=1024 or manifest['hidden']!=2048:
        raise ValueError('direct Wo requires formal GM handoff shape')
    selected={(j['epoch'],j['core']):j for j in jobs if j['stage']==1}
    expected={(epoch,core) for epoch in range(1,manifest['layers']+1) for core in range(4,20)}
    if set(selected)!=expected or set(begins)!=expected or set(releases)!=expected or set(handoffs)!=expected:
        raise ValueError('direct Wo lifecycle coverage incomplete')
    states={}
    for key,job in selected.items():
        b=begins[key];r=releases[key];h=handoffs[key];core=key[1]
        if (int(b['input_gm'])!=core*GM_STRIDE+LANDING or int(b['weight_gm'])!=core*GM_STRIDE+WEIGHTS or
                int(b['bytes'])!=TILE_BYTES or int(r['status'])!=0 or
                any(int(record[field])!=0 for record in (b,h)
                    for field in ('relocation_read_bytes','relocation_write_bytes')) or
                not job['start_tick']<=int(b['tick'])<=int(r['tick'])<=job['tick']):
            raise ValueError('invalid direct Wo addresses or lease boundaries')
        states[key]=dict(core=core,start_tick=job['start_tick'],end_tick=job['tick'],release_tick=int(r['tick']),
                         n_tiles=manifest['hidden']//64)
    def events():
        for line in log.with_suffix('.resources.jsonl').open():
            if not any(f'resource={r} ' in line for r in ('local_read','local_write','dma_read_landing')):continue
            f=dict(FIELDS.findall(line))
            yield f['resource'],{k:int(v) for k,v in f.items() if k not in ('domain','resource')}
    return dict(enabled=True,**audit_requests(states,events()))
