#!/usr/bin/env python3
"""Audit final per-core bank counters against completed logical traffic."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def verify(case):
    logs=list((case/'logs').glob('attention-sst*.log'))
    if len(logs)!=1: raise ValueError('need one merged simulator log')
    local={}; banks={}; grants={}; untimed=Counter();logical_intervals=0
    expected_beats=Counter()
    def evidence():
        for path in (logs[0],logs[0].with_suffix(".resources.jsonl")):
            if path.exists():
                with path.open() as stream:yield from stream
    for line in evidence():
        if 'GOLEM_LOCAL_GM_' in line:
            fields={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)}
            core=fields.get('core')
            if 'GOLEM_LOCAL_GM_BANK_STATS ' in line:
                if core in banks: raise ValueError('duplicate bank stats')
                banks[core]=fields
            elif 'GOLEM_LOCAL_GM_BANK_GRANTS ' in line:
                key=(core,fields['bank'])
                if key in grants: raise ValueError('duplicate bank grants')
                grants[key]=fields['beats']
            elif 'GOLEM_LOCAL_GM_STATS ' in line: local[core]=fields
        elif '[RESOURCE_INTERVAL]' in line and ('resource=local_read ' in line or 'resource=local_write ' in line):
            f=dict(re.findall(r'(\w+)=([^\s]+)',line));logical_intervals+=1
            # Formal baseline core time is 1 ps and the explicit GM clock is 1 GHz.
            assert int(f['start_tick'])%1000==0 and int(f['done_tick'])%1000==0
            assert int(f['submit_tick'])<=int(f['start_tick'])<int(f['done_tick'])
            assert f.get('ok')=='1', 'accepted workloads must contain only completed successful accesses'
            lane='dma_beats' if int(f['client'])==0 else 'compute_beats'
            expected_beats[(int(f['core']),lane)]+=(int(f['address'])%128+int(f['bytes'])+127)//128
        elif '[RESOURCE_INTERVAL]' in line and 'resource=untimed_' in line:
            f=dict(re.findall(r'(\w+)=([^\s]+)',line))
            offset=int(f['address'])-int(f['core'])*2**21
            untimed[(f['resource'],offset,int(f['bytes']))]+=1
    if logical_intervals==0:raise ValueError("missing logical GM interval evidence")
    if set(banks)!=set(range(20)) or set(local)!=set(banks):raise ValueError('missing core/model counters')
    untimed_data_bytes=sum(size*count for (_,offset,size),count in untimed.items() if offset<0 or offset+size>0x8000)
    if untimed_data_bytes: raise ValueError(f"untimed tensor traffic: {untimed_data_bytes} bytes")
    total=Counter()
    for core,b in banks.items():
        assert b['capacity']==2**21 and b['banks']==8 and b['line_bytes']==128
        assert b['pending']==b['queued']==0 and b['response_high_water']<=b['response_capacity']
        assert sum(grants[(core,i)] for i in range(8))==b['compute_beats']+b['dma_beats']
        assert b['payload_bytes']==local[core]['gmem_local_read_bytes']+local[core]['gmem_local_write_bytes']
        assert b['payload_bytes']<=128*(b['compute_beats']+b['dma_beats'])
        for lane in ('compute_beats','dma_beats'):
            assert b[lane]==expected_beats[(core,lane)], 'physical beats must match line-rounded logical requests'
        for key in ('compute_beats','dma_beats','payload_bytes','bank_conflict_cycles','response_blocked_lane_cycles'):total[key]+=b[key]
    return {'status':'PASS','cores':20,'gm_bytes_per_core':2**21,'configuration':{k:banks[4][k] for k in ('banks','line_bytes','response_cycles','response_capacity')},
        'logical_intervals_checked':logical_intervals,'line_rounded_lane_beats_reconciled':True,'ticks_per_gm_cycle':1000,'totals':dict(total),'untimed_tensor_bytes':untimed_data_bytes,'control_window_bytes':0x8000,'untimed_accesses':[{'resource':k[0],'gm_offset':k[1],'bytes':k[2],'count':v} for k,v in sorted(untimed.items())],
        'scope':'Counter conservation and drained bounded queues; cycle arbitration and data semantics are tested by production probe and RTL differential.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('case',type=Path);a=p.parse_args()
    report=verify(a.case);(a.case/'gm_bank_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
