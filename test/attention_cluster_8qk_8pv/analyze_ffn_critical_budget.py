#!/usr/bin/env python3
"""Audit the FFN critical workers and optimistic fixed-start wait-removal bounds."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

from analyze_reuse_window_critical_path import analyze

FIELDS=re.compile(r'(\w+)=([^\s]+)')


def fixed_start_budget(rows):
    if not rows or any(r['removable_cycles']<0 for r in rows):
        raise ValueError('empty or negative removable budget')
    old=max(rows,key=lambda r:r['end_cycle'])
    new=max(rows,key=lambda r:r['end_cycle']-r['removable_cycles'])
    earliest=new['end_cycle']-new['removable_cycles']
    return dict(current_last_core=old['core'],optimistic_last_core=new['core'],
                current_finish_cycle=old['end_cycle'],optimistic_finish_cycle=earliest,
                optimistic_saved_cycles=old['end_cycle']-earliest)


def audit(case):
    r=json.loads((case/'decoder_stack_result.json').read_text())
    p=json.loads((case/'decoder_stack_provenance.json').read_text())
    if r['status']!='PASS':raise ValueError('completed PASS case required')
    if p['environment'].get('GOLEM_DECODER_FFN_BLOCK_PIPELINE','0')!='0':
        raise ValueError('overlapping window engines require interval-level union')
    jobs={};flows=defaultdict(list)
    with next((case/'logs').glob('attention-sst*.log')).open() as stream:
        for line in stream:
            if '[DECODER_OP] event=done ' in line:
                f=dict(FIELDS.findall(line))
                if f['stage']=='4' and f['op']=='9':
                    key=int(f['epoch']),int(f['core'])
                    if key in jobs:raise ValueError('duplicate fused FFN job')
                    jobs[key]={k:int(v) for k,v in f.items() if k!='event'}
            elif '[GEMM_WINDOW_FLOW]' in line:
                f=dict(FIELDS.findall(line))
                if f['stage']=='4' and f['operator'] in ('gate','up','down'):
                    flows[int(f['epoch']),int(f['core'])].append(f)
    layers={}
    for epoch in sorted({k[0] for k in jobs}):
        rows=[]
        for key,j in jobs.items():
            if key[0]!=epoch:continue
            selected=flows[key]
            if {f['operator'] for f in selected}!={'gate','up','down'}:
                raise ValueError('missing FFN window operators')
            phase=defaultdict(int);wait=defaultdict(int);bypass=0
            for f in selected:
                if sum(int(v) for k,v in f.items() if k.startswith('phase_'))!=int(f['active_cycles']):
                    raise ValueError('phase counters do not partition window time')
                for k,v in f.items():
                    if k.startswith('phase_'):phase[k]+=int(v)
                    if k.startswith('wait_'):wait[k]+=int(v)
                bypass+=int(f.get('b_head_bypass_admissions','0'))
            duration=(j['tick']-j['start_tick'])/1000
            if sum(phase.values())>duration or wait['wait_b_gm']>phase['phase_acquire']:
                raise ValueError('overlap or inconsistent window accounting')
            rows.append(dict(core=j['core'],start_cycle=j['start_tick']/1000,end_cycle=j['tick']/1000,
                             job_cycles=duration,window_phase_cycles=dict(phase),
                             overlapping_wait_cycles=dict(wait),head_bypass_admissions=bypass,
                             removable_cycles=wait['wait_b_gm']))
        if len(rows)!=16:raise ValueError('formal budget requires all 16 workers')
        last=max(rows,key=lambda x:x['end_cycle'])
        layers[str(epoch)]=dict(last_worker=last,workers=rows,
                               erase_all_b_gm_wait_fixed_starts=fixed_start_budget(rows),
                               erase_all_acquire_fixed_starts=fixed_start_budget([
                                   dict(v,removable_cycles=v['window_phase_cycles']['phase_acquire']) for v in rows]))
    critical=analyze(case)
    return dict(status='PASS',unit='cycles',ticks_per_cycle=1000,case=str(case.resolve()),
                full_cycles=r['embedding']['end_to_end_cycles'],
                boundary='first token-ID DMA send to final valid-token full-logits ACK; host argmax excluded',
                additive_e2e_phases=critical['additive_phases'],ffn=layers,
                selected_hypothesis='Demand-ordered FFN GM weight reads, same four slots and all IO. Fixed slot IDs can submit future sequences ahead of a consuming sequence at wraparound.',
                limits=['Budgets erase all selected waits optimistically, hold job starts fixed and recompute the last finisher. They are not achievable savings predictions.',
                        'Per-worker and per-layer budgets must not be summed into end-to-end gain. No downstream replay or shared-resource feedback is modeled.',
                        'A demand-order change can affect only a subset of B GM waiting; its causal gain requires same-build SST pairs.',
                        'Wait counters overlap useful work and are not added to phase time. Matrix/scatter phases are required work plus waiting, not wholly removable idle time.'],
                experiment_gate=dict(micro_min_reduction_percent=0.3,
                                     formal='Run paired full chain only after formal 16-worker micro passes all audits and improves at least 0.3%; adoption requires full and decoder gains versus same-build control and no regression versus selected default.'),
                result_sha256=hashlib.sha256((case/'decoder_stack_result.json').read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();d=audit(a.case.resolve());a.output.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=d['status'],full_cycles=d['full_cycles'],budgets={
        k:v['erase_all_b_gm_wait_fixed_starts'] for k,v in d['ffn'].items()}),indent=2))


if __name__=='__main__':main()
