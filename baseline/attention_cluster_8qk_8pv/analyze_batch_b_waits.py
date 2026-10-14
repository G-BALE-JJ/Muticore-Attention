#!/usr/bin/env python3
"""Projection-window queue/service and array-weighted MAC occupancy."""
from collections import defaultdict
import json
from pathlib import Path
import re

FIELDS=re.compile(r'(\w+)=([^\s]+)')


def overlap(start,end,lower,upper):
    return max(0,min(end,upper)-max(start,lower))


def analyze(case):
    case=Path(case)
    report=json.loads((case/'projection_e2e_result.json').read_text())
    resource=json.loads((case/'resource_analysis.json').read_text())
    factor=resource['ticks_per_cycle']
    windows={int(core):(v['start_cycle']*factor,v['end_cycle']*factor)
             for core,v in report.get('projection_worker_jobs',{}).items()}
    if not windows:
        log=next((case/'logs').glob('attention-sst*.log'))
        with log.open(errors='replace') as stream:
            for line in stream:
                if '[PROJECTION_JOB]' in line:
                    v=dict(FIELDS.findall(line));windows[int(v['core'])]=(int(v['start'])*factor,int(v['end'])*factor)
    if set(windows)!=set(range(4,20)):raise ValueError('projection windows incomplete')
    result={str(core):{'resources':{},'array_mac_cycles':0.0} for core in windows}
    log=next((case/'logs').glob('attention-sst*.log'))
    with log.with_suffix('.resources.jsonl').open() as stream:
        for line in stream:
            r=dict(FIELDS.findall(line));core=int(r['core'])
            if core not in windows:continue
            lower,upper=windows[core]
            submit,start,done,visible=(int(r[k]) for k in ('submit_tick','start_tick','done_tick','visible_tick'))
            if visible<lower or submit>=upper:continue
            name=r['domain']+'.'+r['resource']
            totals=result[str(core)]['resources'].setdefault(name,{'count':0,'queue_cycles':0.0,'service_cycles':0.0,'visibility_cycles':0.0})
            totals['count']+=1
            totals['queue_cycles']+=overlap(submit,start,lower,upper)/factor
            totals['service_cycles']+=overlap(start,done,lower,upper)/factor
            totals['visibility_cycles']+=overlap(done,visible,lower,upper)/factor
            if name=='array.compute':
                arrays=int(r['client'])
                if arrays not in (32,64):raise ValueError('unexpected Projection MAC group size')
                result[str(core)]['array_mac_cycles']+=arrays*overlap(start,done,lower,upper)/factor
    for core,values in result.items():
        lower,upper=windows[int(core)]
        values['elapsed_cycles']=(upper-lower)/factor
        values['mac_array_utilization']=values['array_mac_cycles']/(64*values['elapsed_cycles'])
    return {'status':'PASS','by_worker':result,
            'scope':'Clipped to each Projection job; summed queue/service may overlap. MAC utilization weights each interval by actual 32/64 array group size.'}
