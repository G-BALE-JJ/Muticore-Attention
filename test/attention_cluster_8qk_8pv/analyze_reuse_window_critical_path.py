#!/usr/bin/env python3
"""Identify stage finishers and exclusive window phases from completed runs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics


def additive_timeline(result):
    """Partition elapsed time, attributing an eager Norm only its exposed tail."""
    e=result['embedding'];norm=result['final_norm'];lm=result['lm_head']
    first=e['first_token_ids_dma_tick'];last=lm['last_logits_ack_tick']
    edges=[('embedding',first,e['last_embedding_output_ack_tick'])]
    previous=e['last_embedding_output_ack_tick']
    for layer,l in sorted(result['dma_timing']['layers'].items(),key=lambda pair:int(pair[0])):
        edges.extend([(f'before_layer{layer}',previous,l['first_input_dma_tick']),
                      (f'layer{layer}_projection_attention',l['first_input_dma_tick'],l['last_attention_o_ack_tick']),
                      (f'layer{layer}_post_attention',l['last_attention_o_ack_tick'],l['last_y_ack_tick'])])
        previous=l['last_y_ack_tick']
    norm_start=norm['first_input_dma_tick'];norm_end=norm['last_output_ack_tick']
    if norm_end<norm_start:raise ValueError('final Norm has reversed boundaries')
    exposed_start=max(previous,norm_start);exposed_end=max(previous,norm_end)
    edges.extend([('before_final_norm',previous,exposed_start),
                  ('final_norm',exposed_start,exposed_end),
                  ('before_lm',exposed_end,lm['first_input_dma_tick']),
                  ('lm',lm['first_input_dma_tick'],last)])
    phases={name:dict(cycles=(end-start)/1000,percent=100*(end-start)/(last-first)) for name,start,end in edges}
    if any(v['cycles']<0 for v in phases.values()) or sum(v['cycles'] for v in phases.values())!=(last-first)/1000:
        raise ValueError('end-to-end additive partition is invalid')
    return phases,dict(final_norm_span_cycles=(norm_end-norm_start)/1000,
                       final_norm_overlap_with_decoder_cycles=max(0,min(previous,norm_end)-norm_start)/1000,
                       final_norm_exposed_tail_cycles=(exposed_end-exposed_start)/1000)


def analyze(case):
    result=json.loads((case/'decoder_stack_result.json').read_text())
    if result['status']!='PASS':raise ValueError('completed PASS run required')
    env=json.loads((case/'decoder_stack_provenance.json').read_text())['environment']
    jobs=[];flows=defaultdict(list)
    for line in next((case/'logs').glob('attention-sst*.log')).open():
        fields=dict(re.findall(r'(\w+)=([^\s]+)',line))
        if '[DECODER_OP] event=done' in line:
            jobs.append({k:int(v) for k,v in fields.items() if k!='event'})
        elif '[GEMM_WINDOW_FLOW]' in line:
            f={k:int(v) for k,v in fields.items() if k!='operator'};f['operator']=fields['operator']
            if sum(v for k,v in f.items() if k.startswith('phase_'))!=f['active_cycles']:
                raise ValueError('exclusive window phases do not sum to active cycles')
            flows[f['core'],f['epoch'],f['stage']].append(f)
    stages={}
    for epoch,stage in sorted({(j['epoch'],j['stage']) for j in jobs}):
        js=[j for j in jobs if (j['epoch'],j['stage'])==(epoch,stage)]
        final=max(js,key=lambda j:j['tick']);slow=max(js,key=lambda j:j['tick']-j['start_tick'])
        durations=[(j['tick']-j['start_tick'])/1000 for j in js]
        selected=flows[final['core'],epoch,stage];phases=defaultdict(int);waits=defaultdict(int)
        for f in selected:
            for k,v in f.items():
                if k.startswith('phase_'):phases[k]+=v
                if k.startswith('wait_'):waits[k]+=v
        stages[f'{epoch}.{stage}']=dict(
            workers=len(js),span_cycles=(final['tick']-min(j['start_tick'] for j in js))/1000,
            start_spread_cycles=(max(j['start_tick'] for j in js)-min(j['start_tick'] for j in js))/1000,
            completion_spread_cycles=(final['tick']-min(j['tick'] for j in js))/1000,
            duration_min=min(durations),duration_median=statistics.median(durations),duration_max=max(durations),
            last_core=final['core'],longest_duration_core=slow['core'],
            last_core_duration=(final['tick']-final['start_tick'])/1000,
            last_core_window_phases=dict(phases),last_core_overlapping_wait_counters=dict(waits),
            last_core_hbm_read=final['hbm_read'],last_core_hbm_write=final['hbm_write'])
    first=result['embedding']['first_token_ids_dma_tick'];last=result['lm_head']['last_logits_ack_tick']
    phases,overlap=additive_timeline(result)
    return dict(status='PASS',total_cycles=(last-first)/1000,additive_phases=phases,stages=stages,
                overlap=overlap,
                block_pipeline=env.get('GOLEM_DECODER_FFN_BLOCK_PIPELINE','0')=='1',
                notes=['Stage spans overlap across workers/stages; they are not additive.',
                       'Window phases are exclusive within each engine. Two engines may overlap; their sums are not job critical-path time.',
                       'Wait counters can overlap useful work and must not be added to window phases.',
                       'The last finishing worker can differ from the worker with the longest job duration.'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=analyze(a.case)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(dict(status=r['status'],total_cycles=r['total_cycles'],additive_phases=r['additive_phases']),indent=2))


if __name__=='__main__':main()
