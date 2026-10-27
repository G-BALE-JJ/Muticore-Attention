#!/usr/bin/env python3
"""Compare demand-order scheduling with strict paired-work and cycle checks."""
import argparse
import json
from pathlib import Path

from compare_final_norm_lm_stream import BINARY_KEYS, normalized_sources, cycle_comparison
from summarize_reuse_window_optimization import comparable_inputs, comparable_micro_inputs, read, scopes
from summarize_software_dataflow_next import environment
from summarize_software_stage_chain import output_bits
from summarize_gm_prefetch import flow_stats, micro_group

FLAG='GOLEM_DECODER_FFN_DEMAND_ORDER'


def check_pair(cases,provenance,micro=False):
    for key in BINARY_KEYS:
        if not provenance[0][key] or provenance[0][key]!=provenance[1][key]:
            raise ValueError('paired binary mismatch: '+key)
    if normalized_sources(provenance[0]['source_sha256'])!=normalized_sources(provenance[1]['source_sha256']):
        raise ValueError('paired source mismatch')
    manifests=[comparable_micro_inputs(c.parent) if micro else comparable_inputs(c) for c in cases]
    for m in manifests:m.pop('ffn_demand_order',None)
    if manifests[0]!=manifests[1]:raise ValueError('paired input/layout mismatch')
    env=[environment(p,c) for p,c in zip(provenance,cases)]
    changes={k:dict(control=env[0].get(k),candidate=env[1].get(k)) for k in set(env[0])|set(env[1])
             if env[0].get(k)!=env[1].get(k)}
    if changes!={FLAG:dict(control='0',candidate='1')}:
        raise ValueError('unexpected paired environment differences: '+repr(changes))
    return changes


def micro_audit(root):
    names=['micro_control','micro_demand'];cases=[root/n/'layout2' for n in names]
    results=[read(c/'result.json') for c in cases]
    delta=check_pair(cases,[r['provenance'] for r in results],micro=True)
    rows=micro_group(root,names)
    reduction=rows[names[1]]['reduction_percent']
    return dict(status='PASS',unit='cycles',cases=rows,environment_changes=delta,
                saved_cycles=rows[names[0]]['cycles']-rows[names[1]]['cycles'],
                min_reduction_percent=0.3,eligible_for_full_chain=reduction>=0.3)


def formal_audit(root):
    cases=[root/n/'reuse_window_probe' for n in ('formal_control','formal_demand')]
    results=[read(c/'decoder_stack_result.json') for c in cases]
    provenance=[read(c/'decoder_stack_provenance.json') for c in cases]
    changes=check_pair(cases,provenance)
    for c,r in zip(cases,results):
        if r['status']!='PASS' or read(c.parent/'all_gemm_reuse_window_report.json')['status']!='PASS':
            raise ValueError('full numerical/lifecycle/GEMM audit failed')
        if r['gm_banks']['status']!='PASS' or r['gm_banks']['untimed_tensor_bytes']:
            raise ValueError('resource or untimed tensor IO audit failed')
    flow=[flow_stats(c,p['environment']) for c,p in zip(cases,provenance)]
    if flow[0]['traffic']!=flow[1]['traffic']:raise ValueError('dense GM/HBM/C/MAC work changed')
    timing=cycle_comparison(*map(scopes,results))
    # The first intervention is FFN1. Verify every embedding/Attention/Wo/Add
    # timestamp preceding it, not merely the total prefix duration.
    prefix=True
    for section,keys in (('embedding',('first_token_ids_dma_tick','last_embedding_output_ack_tick')),):
        prefix &= all(results[0][section][k]==results[1][section][k] for k in keys)
    prefix &= all(results[0]['lifecycle']['stage_spans']['1'][k]==results[1]['lifecycle']['stage_spans']['1'][k]
                  for k in ('1','2'))
    prefix &= results[0]['dma_timing']['layers']['1']['last_attention_o_ack_tick']==results[1]['dma_timing']['layers']['1']['last_attention_o_ack_tick']
    if not prefix:raise ValueError('prefix before FFN1 changed')
    baseline=root.parent/'optimization_20261024/p7_restored_a/reuse_window_probe'
    old=read(baseline/'decoder_stack_result.json')
    control_reproduces=scopes(old)==scopes(results[0])
    previous_bits=output_bits(baseline,cases[0])
    eligible=(all(timing[k]['saved_cycles']>0 for k in ('完整端到端','两层 decoder')) and
              results[1]['embedding']['end_to_end_cycles']<old['embedding']['end_to_end_cycles'])
    return dict(status='PASS',unit='cycles',ticks_per_cycle=1000,timing=timing,
                environment_changes=changes,binaries={k:provenance[0][k] for k in BINARY_KEYS},
                first_ffn_prefix_identical=bool(prefix),control_reproduces_selected_cycles=control_reproduces,
                historical_output_bits=previous_bits,paired_output_bits=output_bits(*cases),flow=flow,
                saved_vs_selected_cycles=old['embedding']['end_to_end_cycles']-results[1]['embedding']['end_to_end_cycles'],
                adopt=bool(eligible and control_reproduces),
                boundary='first token-ID DMA send to final valid-token full-logits ACK; host argmax excluded',
                note='Same four B slots, GM capacity, ports, array contexts and dense work; only FFN GM request submission order differs. Window waits overlap useful work and must not be added to stage spans.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--micro-only',action='store_true');a=p.parse_args();root=a.root.resolve()
    result=dict(status='PASS',micro=micro_audit(root))
    if not a.micro_only:result['formal']=formal_audit(root)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],micro_eligible=result['micro']['eligible_for_full_chain'],
                          adopt=result.get('formal',{}).get('adopt')),indent=2))


if __name__=='__main__':main()
