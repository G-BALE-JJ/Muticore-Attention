#!/usr/bin/env python3
"""Matched full-stack comparison for the opt-in FFN GM layout schedule."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from compare_decoder_groups_stack import read
from run_decoder_layer import read_bands


def phases(r):
 d=r['dma_timing'];e=r['embedding'];n=r['final_norm'];lm=r['lm_head']
 p={'embedding':e['embedding_cycles'],'embedding_handoff':e['handoff_cycles']}
 for i in ('1','2'):
  p['layer'+i+'_front']=d['layers'][i]['attention_cycles'];p['layer'+i+'_post']=d['layers'][i]['post_attention_cycles']
  if i=='1':p['layer_handoff']=d['layer_handoff_cycles']
 p.update(final_norm_handoff=n['handoff_cycles'],final_norm=n['rmsnorm_cycles'],lm_handoff=lm['handoff_cycles'],lm_head=lm['lm_head_cycles'])
 assert sum(p.values())==e['end_to_end_cycles'];return p


def dma_control_comparison(cases,audits):
 counters=[];banks=[]
 for case in cases:
  log=next((case/'logs').glob('attention-sst*.log'));bank={};counts=Counter()
  for line in log.open():
   if 'GOLEM_LOCAL_GM_BANK_STATS ' in line:
    f={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)};bank[f['core']]=f
  for line in log.with_suffix('.resources.jsonl').open():
   if ('resource=local_read ' not in line and 'resource=local_write ' not in line) or not any(f'core={c} ' in line for c in range(4)):continue
   f=dict(re.findall(r'(\w+)=([^\s]+)',line));core=int(f['core']);offset=int(f['address'])-core*2**21;size=int(f['bytes'])
   assert 0<=offset and offset+size<=0x8000,'root/manager tensor access'
   if f['client']=='0':counts[(core,f['resource'],offset,size)]+=1
  for core in range(4):
   beats=sum(count*((offset%128+size+127)//128) for (c,_,offset,size),count in counts.items() if c==core)
   assert beats==bank[core]['dma_beats'],'root/manager DMA conservation'
  counters.append(counts);banks.append(bank)
 for core in range(4,20):assert banks[0][core]['dma_beats']==banks[1][core]['dma_beats'],('worker DMA work changed',core)
 delta=[{'core':k[0],'resource':k[1],'offset':k[2],'bytes':k[3],
  'requests_saved':counters[0][k]-counters[1][k]} for k in sorted(counters[0].keys()|counters[1].keys()) if counters[0][k]!=counters[1][k]]
 assert all(d['resource']=='local_write' and d['offset']==0x3800 and d['bytes']==8 and d['requests_saved']>0 for d in delta),delta
 saved=audits[0]['totals']['dma_beats']-audits[1]['totals']['dma_beats']
 assert saved==sum(d['requests_saved'] for d in delta),'unexplained DMA difference'
 return {'status':'PASS','beats_saved':saved,'payload_bytes_saved':8*saved,'differences':delta,
  'scope':'Only 8B root/manager control-poll landing writes at 0x3800 differ; every worker DMA beat count is unchanged.'}


def compare(base,candidate):
 cases=(base,candidate);man=[json.loads((c/'decoder_stack_inputs.json').read_text()) for c in cases]
 prov=[json.loads((c/'decoder_stack_provenance.json').read_text()) for c in cases]
 res=[json.loads((c/'decoder_stack_result.json').read_text()) for c in cases]
 assert all(r['status']=='PASS' for r in res) and res[0]['shape']==res[1]['shape']
 for k in ('source_sha256','guest_sha256','sst_library_sha256','sst_memhierarchy_sha256'):assert prov[0][k]==prov[1][k],k
 ignored={'GOLEM_DECODER_FFN_LAYOUT','GOLEM_ARTIFACT_ROOT','GOLEM_DECODER_STACK_MANIFEST','GOLEM_MPI_ARGS','GOLEM_MPI_RANK_LOG_ROOT'}
 env=[p['environment'] for p in prov]
 diffs={k:[e.get(k) for e in env] for k in env[0].keys()|env[1].keys() if env[0].get(k)!=env[1].get(k)}
 assert set(diffs)<=ignored,diffs
 assert env[0]['GOLEM_DECODER_FFN_LAYOUT']=='0' and env[1]['GOLEM_DECODER_FFN_LAYOUT'] in ('1','2')
 assert all(e['GOLEM_DECODER_DOWN_INPUT_REUSE']=='0' for e in env)
 def norm(m):return {k:([{k:v for k,v in p.items() if k!='path'} for p in v] if k=='preloads' else v) for k,v in m.items() if k!='ffn_layout'}
 assert norm(man[0])==norm(man[1]),'input/layout contract changed'
 def down_flows(r):return {(f['core'],f['epoch']):f for f in r['lifecycle']['ffn_down_flow']}
 assert down_flows(res[0])==down_flows(res[1]),'C-buffer partial work changed'
 tokens=[r['lm_head']['next_token']['token_id'] for r in res];assert tokens[0]==tokens[1]
 layout=man[0]['layout'];h=man[0]['hidden'];targets={f'layer{i}':l['output'] for i,l in enumerate(layout['layers'],1)}
 targets.update(final_norm=layout['final_norm']['output'],embedding=layout['embedding']['output'])
 targets.update({k:layout['shared'][k] for k in ('wo','h1','norm2')})
 equal={}
 for name,offset in targets.items():
  a,b=(read_bands(c,offset,h) for c in cases)
  assert np.array_equal(a.view('u2'),b.view('u2')),name
  equal[name]={'status':'PASS','elements':a.size,'sha256':hashlib.sha256(a.tobytes()).hexdigest()}
 a,b=(np.load(c/'last_token_logits.npy') for c in cases);assert np.array_equal(a.view('u2'),b.view('u2'))
 equal['logits']={'status':'PASS','elements':a.size,'sha256':hashlib.sha256(a.tobytes()).hexdigest()}
 parsed=[read(c) for c in cases]
 assert parsed[0][1].keys()==parsed[1][1].keys(),'array group jobs changed'
 group_fields=('core','stage','epoch','groups','launches','program_cycles','compute_array_cycles')
 for job,a in parsed[0][1].items():
  b=parsed[1][1][job]
  for key in group_fields:assert a[key]==b[key],(job,key)
 assert parsed[0][2]==parsed[1][2],'SFU work changed'
 # Contention counters may differ. Every physical array transfer and MAC
 # work counter must remain the same for a pure GM layout transformation.
 array_fields=('matrix_broadcast_requests','matrix_broadcast_ingress_bytes','matrix_broadcast_sink_bytes',
  'matrix_broadcast_transfer_cycles','input_scatter_requests','input_scatter_bytes','input_scatter_transfer_cycles',
  'output_scatter_requests','output_gather_requests','output_scatter_gather_bytes','output_scatter_gather_transfer_cycles','active_k_launches','active_k_columns')
 for core in range(4,20):
  for key in array_fields:assert parsed[0][0][core][key]==parsed[1][0][core][key],(core,key)
 audits=[json.loads((c/'gm_bank_verification.json').read_text()) for c in cases]
 resources=[json.loads((c/'decoder_resources.json').read_text()) for c in cases]
 assert all(x['status']=='PASS' and x['untimed_tensor_bytes']==0 for x in audits)
 assert all(x['status']=='PASS' and x['verified_decoder_jobs']==416 for x in resources)
 for stage,v in resources[0]['by_epoch_stage'].items():
  for resource in ('dma_read_transport','dma_write_transport'):
   assert v.get(resource)==resources[1]['by_epoch_stage'][stage].get(resource),(stage,resource)
 compute_saved=audits[0]['totals']['compute_beats']-audits[1]['totals']['compute_beats']
 expected_per_job=(6*64*4*h*2+(2*64*h*2 if env[1]['GOLEM_DECODER_FFN_LAYOUT']=='2' else 0))//128
 assert compute_saved==32*expected_per_job,(compute_saved,expected_per_job)
 controls=dma_control_comparison(cases,audits)
 totals=[r['embedding']['end_to_end_cycles'] for r in res]
 return {'status':'PASS','baseline_cycles':totals[0],'candidate_cycles':totals[1],'saved_cycles':totals[0]-totals[1],
  'saved_percent':100*(1-totals[1]/totals[0]),'phases':[phases(r) for r in res],
  'FP16_bitwise_equal':equal,'greedy_token':tokens,'Cbuffer_partial_work_equal':True,
  'compute_GM_beats_saved':compute_saved,'expected_per_FFN_job_compute_beats_saved':expected_per_job,
  'DMA_beats_equal':controls['beats_saved']==0,'tensor_DMA_work_equal':True,'control_poll_difference':controls,
  'array_physical_work_equal':True,'SFU_accepted_tokens_equal':True,
  'group_compute_overlap_cycles':[{f'{c}:{e}':v['both_groups_compute_cycles'] for (c,e),v in p[1].items()} for p in parsed],
  'environment_differences':diffs,'hardware_changed':False,
  'scope':'Same build, inputs and banked hardware model; only software FFN layout differs. Host argmax excluded.'}

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 r=compare(a.baseline,a.candidate);a.output.write_text(json.dumps(r,indent=2)+'\n');print({k:r[k] for k in ('status','baseline_cycles','candidate_cycles','saved_percent')})
