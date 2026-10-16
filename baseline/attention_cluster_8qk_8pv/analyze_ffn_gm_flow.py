#!/usr/bin/env python3
"""Reconcile FFN physical GM beats by client/arena, with layout/reuse work bounds."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path
import re

FIELDS=re.compile(r'(\w+)=([^\s]+)')
REGIONS=[(0,0x8000,'control'),(0x8000,0x10000,'projection'),(0x10000,0x110000,'input_and_residual'),
 (0x110000,0x112000,'weight_scratch'),(0x112000,0x152000,'gate_gated_panels'),
 (0x152000,0x162000,'producer_weight_slots'),(0x162000,0x1a2000,'up_panels'),
 (0x1a2000,0x1b2000,'down_weight_slots'),(0x1b2000,0x1f2000,'down_output_panels')]


def analyze(case,hidden):
 log=next((case/'logs').glob('attention-sst*.log'));jobs=defaultdict(list)
 for line in log.open():
  if '[DECODER_OP] event=done' in line:
   f=dict(FIELDS.findall(line))
   if int(f['stage'])==4 and int(f['op'])==9:jobs[int(f['core'])].append({k:int(f[k]) for k in ('core','epoch','start_tick','tick','gm_read','gm_write')})
 if not jobs:raise ValueError('no fused FFN jobs')
 for v in jobs.values():v.sort(key=lambda j:j['start_tick'])
 starts={c:[j['start_tick'] for j in js] for c,js in jobs.items()};counts=defaultdict(lambda:defaultdict(int));perjob=defaultdict(lambda:defaultdict(int))
 for line in log.with_suffix('.resources.jsonl').open():
  if 'resource=local_read ' not in line and 'resource=local_write ' not in line:continue
  f=dict(FIELDS.findall(line));core=int(f['core']);start=int(f['start_tick'])
  if core not in jobs:continue
  i=bisect_right(starts[core],start)-1
  if i<0 or start>=jobs[core][i]['tick']:continue
  j=jobs[core][i];assert int(f['done_tick'])<=j['tick'] and f['ok']=='1'
  size=int(f['bytes']);offset=int(f['address'])-core*2**21;assert offset>=0
  region=next((name for a,b,name in REGIONS if a<=offset and offset+size<=b),None)
  if region is None:raise ValueError(f'unclassified GM range {offset:x}+{size}')
  lane='dma' if f['client']=='0' else 'compute';direction='read' if f['resource']=='local_read' else 'write'
  key=f'{lane}.{direction}.{region}';v=counts[key]
  v['requests']+=1;v['bytes']+=size;v['beats']+=(offset%128+size+127)//128
  perjob[(core,j['epoch'])][lane+'.'+direction]+=size
 for c,js in jobs.items():
  for j in js:
   v=perjob[c,j['epoch']];assert v['compute.read']==j['gm_read'] and v['compute.write']==j['gm_write']
 weights=3*(4*hidden//64)*(hidden//64)*8192
 down_reuse_saved=(hidden//64-1)*(4*hidden//64)*8192
 return {'status':'PASS','jobs':sum(map(len,jobs.values())),'hidden':hidden,'by_lane_direction_arena':dict(counts),
  'per_job_compute_bytes':{f'{c}:{e}':dict(v) for (c,e),v in perjob.items()},
  'formal_64_row_reference_work':{'weight_read_bytes':weights,'repeated_array_input_read_bytes':weights,
   'input_pack_read_write_bytes':2*64*hidden*2,'gate_up_repack_read_write_bytes':4*64*4*hidden*2,
   'gated_to_down_repack_read_write_bytes':2*64*4*hidden*2,'down_output_repack_read_write_bytes':2*64*hidden*2,
   'layout1_compute_beats_avoidable':6*64*4*hidden*2//128,
   'layout2_compute_beats_avoidable':(6*64*4*hidden*2+2*64*hidden*2)//128,
   'down_K_outer_input_read_beats_avoidable':down_reuse_saved//128,
   'down_K_outer_additional_Cbuffer_read_write_bytes':2*((4*hidden//64)-(4*hidden//64+15)//16)*64*hidden*2},
  'critical_ffn_cycles':max((j['tick']-j['start_tick'])/1000 for js in jobs.values() for j in js),
  'scope':'Logical requests reconciled to line beats; arena totals separate DMA and compute. Whole logical spans are not physical port busy time. Work formulas use full 64-row fused down/residual and 16-tile panels.'}

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('case',type=Path);p.add_argument('--hidden',type=int,default=2048);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 r=analyze(a.case,a.hidden);a.output.write_text(json.dumps(r,indent=2)+'\n');print('PASS FFN traffic jobs',r['jobs'])
