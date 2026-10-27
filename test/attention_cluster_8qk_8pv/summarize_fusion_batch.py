#!/usr/bin/env python3
"""Reconcile the complete software fusion batch on an identical hardware pair."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
from statistics import mean
from summarize_reuse_window_optimization import ROOT, read, digest, comparable_inputs, scopes
from summarize_software_dataflow_next import environment
from summarize_software_stage_chain import output_bits
from summarize_gm_prefetch import flow_stats


def jobs(case):
    result=[]
    with next((case/'logs').glob('attention-sst*.log')).open() as stream:
        for line in stream:
            if '[DECODER_OP] event=done ' in line:
                result.append({k:int(v) for k,v in re.findall(r'(\w+)=(-?\d+)',line)})
    return result


def totals(items):
    result=Counter(jobs=len(items))
    for j in items:
        for field in ('hbm_read','hbm_write','gm_read','gm_write'):result[field]+=j[field]
    return dict(result)


def fixed_hardware():
    snapshot=ROOT/'results/software_dataflow_next_20261021'
    fixed=read(snapshot/'fixed_hardware_sources.json')
    layout='src/sst/elements/golem/attention/decoderMemoryLayout.h';old=fixed.pop(layout)
    text=re.sub(r'    // Stage-chain Wo output temporarily aliases.*?"Wo handoff overlaps retained residual input"\);\n',
                '',(ROOT/layout).read_text(),flags=re.S)
    if hashlib.sha256(text.encode()).hexdigest()!=old:raise ValueError('GM layout changed beyond the existing software alias')
    guest_env='src/sst/elements/golem/tests/architecture/ncores_selfcom_dma_ctrl.py'
    env_sha=fixed.pop(guest_env)
    original_env=(ROOT/guest_env).read_text()
    for flag in ('GOLEM_DECODER_ADD_NORM_FUSION','GOLEM_DECODER_EAGER_FINAL_NORM'):
        line=f'    "{flag}",\n'
        if original_env.count(line)!=1:raise ValueError('guest environment whitelist differs')
        original_env=original_env.replace(line,'')
    if hashlib.sha256(original_env.encode()).hexdigest()!=env_sha:
        raise ValueError('system configuration changed beyond two guest software environment variables')
    for name,sha in fixed.items():
        if digest(ROOT/name)!=sha:raise ValueError('fixed hardware source changed: '+name)
    state=read(snapshot/'before_state.json');original=ROOT / 'Muticore-Attention'
    if (subprocess.check_output(['git','diff','HEAD'],cwd=original,text=True)!=state['original_diff'] or
            digest(original/'install/lib/sst-elements-library/libgolem.so')!=state['original_library']):
        raise ValueError('original baseline changed')
    return dict(source_files_unchanged=len(fixed),layout_alias_unchanged=True,
                system_configuration_change='only forward two software switches into the guest environment',
                original_baseline_preserved=True)


def audit(root):
    names=('final_control','final_candidate');a,b=names
    cases={n:root/n/'reuse_window_probe' for n in names}
    results={n:read(c/'decoder_stack_result.json') for n,c in cases.items()}
    reports={n:read(root/n/'all_gemm_reuse_window_report.json') for n in names}
    provenance={n:read(c/'decoder_stack_provenance.json') for n,c in cases.items()}
    builds={n:read(root/n/'experiment_provenance.json') for n in names}
    for n in names:
        if results[n]['status']!='PASS' or reports[n]['status']!='PASS':raise ValueError('full numerical/resource/reuse-window audit failed: '+n)
        if read(cases[n]/'verification_provenance.json')['verifier_source_sha256']!=digest(ROOT/'baseline/attention_cluster_8qk_8pv/run_decoder_stack.py'):
            raise ValueError('numerical verifier changed after verification: '+n)
    binary_keys=('guest_sha256','sst_library_sha256','sst_memhierarchy_sha256')
    if any(provenance[a][k]!=provenance[b][k] for k in binary_keys):raise ValueError('paired binaries differ')
    if builds[a]['source_sha256']!=builds[b]['source_sha256']:raise ValueError('paired runtime sources differ')
    for name,sha in builds[b]['source_sha256'].items():
        if digest(ROOT/name)!=sha:raise ValueError('runtime source changed after measurement: '+name)
    for name,sha in provenance[b]['source_sha256'].items():
        if name.startswith('src/') and digest(ROOT/name)!=sha:raise ValueError('guest/hardware source changed after measurement: '+name)
    if digest(ROOT/'install/lib/sst-elements-library/libgolem.so')!=provenance[b]['sst_library_sha256']:
        raise ValueError('installed and measured libraries differ')
    manifests={n:comparable_inputs(c) for n,c in cases.items()}
    for flag in ('add_norm_fusion','eager_final_norm'):
        if {n:manifests[n].pop(flag,False) for n in names}!={a:False,b:True}:
            raise ValueError('paired fusion switches differ from experiment')
    if manifests[a]!=manifests[b]:raise ValueError('paired inputs, weights or layout differ')
    envs={n:environment(provenance[n],cases[n]) for n in names}
    changes={k:dict(control=envs[a].get(k),candidate=envs[b].get(k))
             for k in set(envs[a])|set(envs[b]) if envs[a].get(k)!=envs[b].get(k)}
    want={k:dict(control='0',candidate='1') for k in ('GOLEM_DECODER_ADD_NORM_FUSION','GOLEM_DECODER_EAGER_FINAL_NORM')}
    if changes!=want:raise ValueError('unexpected environment change: '+repr(changes))
    for n in names:
        proof=read(cases[n]/'verification_retained_input.json')
        if (proof!=reports[n]['projection_panels']['input_reference'] or
                proof!=results[n]['numerics']['last_layer_input_reference'] or
                digest(Path(proof['reference_result']))!=proof['reference_result_sha256']):
            raise ValueError('host input reference proofs differ: '+n)
    records={n:jobs(c) for n,c in cases.items()};counts={n:totals(j) for n,j in records.items()}
    m=manifests[a];tensor=m['sequence']*m['hidden']*2*m['layers']
    if counts[a]['jobs']-counts[b]['jobs']!=16*m['layers']:raise ValueError('expected one fewer descriptor per worker/layer')
    if (counts[a]['gm_read']-counts[b]['gm_read']!=tensor or counts[a]['gm_write']!=counts[b]['gm_write'] or
            any(counts[a][k]!=counts[b][k] for k in ('hbm_read','hbm_write'))):
        raise ValueError('traffic changed beyond the fused H1 GM reread')
    fusion=results[b]['resources']['optimizations']['add_norm_fusion']
    chain=results[b]['resources']['optimizations']['stage_chain']
    if (fusion['saved_gm_read_bytes']!=tensor or chain['incremental_saved_gm_read_bytes']!=tensor or
            fusion['verified_jobs']!=16*m['layers']):raise ValueError('fused payload byte accounting failed')
    flows={n:flow_stats(c,provenance[n]['environment']) for n,c in cases.items()}
    if flows[a]['traffic']!=flows[b]['traffic']:raise ValueError('dense GEMM/reuse-window work changed')
    timing={k:dict(control_cycles=v,candidate_cycles=scopes(results[b])[k],
                   reduction_percent=100*(1-scopes(results[b])[k]/v)) for k,v in scopes(results[a]).items()}
    previous=ROOT/'results/software_output_residency_20261022/final_candidate/reuse_window_probe'
    old=read(previous/'decoder_stack_result.json')
    previous_timing={k:dict(previous_cycles=v,control_cycles=scopes(results[a])[k],candidate_cycles=scopes(results[b])[k],
        candidate_reduction_percent=100*(1-scopes(results[b])[k]/v)) for k,v in scopes(old).items()}
    # A new control can also regress against the previously adopted binary.
    # Keep that previous result if the candidate does not improve both scopes.
    selected=b if all(timing[k]['reduction_percent']>0 and previous_timing[k]['candidate_reduction_percent']>0
                      for k in ('完整端到端','两层 decoder')) else 'previous_selected'
    fused_spans={}
    handoffs={}
    for epoch in range(1,m['layers']+1):
        start=min(j['start_tick'] for j in records[a] if j['epoch']==epoch and j['stage']==2)
        stop=max(j['tick'] for j in records[a] if j['epoch']==epoch and j['stage']==3)
        first=min(j['start_tick'] for j in records[b] if j['epoch']==epoch and j['stage']==2)
        last=max(j['tick'] for j in records[b] if j['epoch']==epoch and j['stage']==2)
        fused_spans[str(epoch)]=dict(control_add_norm_span_cycles=(stop-start)/1000,
                                     candidate_fused_span_cycles=(last-first)/1000)
        handoffs[str(epoch)]={}
        for name in names:
            by={(j['core'],j['stage']):j for j in records[name] if j['epoch']==epoch}
            cores=[core for core,stage in by if stage==1]
            def stats(values):return dict(min=min(values),mean=mean(values),max=max(values))
            last_stage=2 if name==b else 3
            handoffs[str(epoch)][name]=dict(
                wo_done_to_ffn_start_cycles=stats([(by[c,4]['start_tick']-by[c,1]['tick'])/1000 for c in cores]),
                norm_done_to_ffn_start_cycles=stats([(by[c,4]['start_tick']-by[c,last_stage]['tick'])/1000 for c in cores]))
    result=dict(status='PASS',selected=selected,timing=timing,previous_selected_timing=previous_timing,
        add_norm_spans=fused_spans,worker_handoffs=handoffs,decoder_job_totals=counts,incremental_saved_gm_read_bytes=tensor,
        incremental_saved_hbm_read_bytes=0,incremental_saved_hbm_write_bytes=0,
        norm_packing=dict(previous_requests=tensor//128,candidate_requests=fusion['norm_write_requests']),
        fusion_payload_audit=fusion,eager_final_norm={k:results[b]['final_norm'][k] for k in
            ('eager_worker_handoff','overlap_with_decoder_tail_cycles','handoff_cycles','tail_cycles')},
        paired_bits=output_bits(cases[a],cases[b]),previous_selected_bits=output_bits(previous,cases[b]),
        environment_changes=changes,hardware=fixed_hardware(),reuse_window_flow=flows,
        all_gemm_coverage={n:{k:reports[n]['coverage'][k] for k in
            ('status','completed_window_jobs','tile_group_launches','workers','qk_completed_jobs',
             'pv_completed_windows','c_traffic_status','window_flow_status','pipeline_execution_status','input_pipeline_status')} for n in names},
        next_token=results[b]['lm_head']['next_token']['token_id'],
        binaries={k:provenance[b][k] for k in binary_keys},runtime_source_sha256=builds[b]['source_sha256'],
        guest_source_sha256={name:sha for name,sha in provenance[b]['source_sha256'].items()
            if name.endswith(('golem_attention_runtime.cpp','golem_decoder_runtime.h'))},
        host_validation_source_sha256={name:digest(ROOT/'baseline/attention_cluster_8qk_8pv'/name) for name in
            ('run_decoder_stack.py','run_all_gemm_reuse_window_baseline.py','verification_retained_input.py',
             'verify_software_stage_chain.py','verify_add_norm_fusion.py','verify_layer_residency.py',
             'verify_decoder_dataflow_resources.py','verify_lm_head.py','summarize_fusion_batch.py')},
        notes=['Simulated cycles at 1 GHz; host runtime is excluded.',
            'Both runs use the same rebuilt guest and library. Added software branches may change control timing relative to the previous selected binary.',
            'Add and RMSNorm retain the existing SFU operations, FP16 boundary, bounded result and timed H1/Norm GM writes.',
            'Eager final Norm starts only after its own worker FFN/output/publication/completion ACKs; root resets flags before releasing the last decoder layer.',
            'Both runs snapshot immutable decoder environment settings before work and read the LM logical sequence once per worker rather than once per vocabulary descriptor.',
            'Two initial pairs were stopped and excluded: an old Add descriptor compatibility regression, then missing host-to-guest environment forwarding. They remain under attempt1_* and attempt2_*.',
            'Q/K/V panels, PV-to-Wo redistribution, final Y/Norm/logits and completion HBM writes remain.'])
    (root/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-root',type=Path,default=ROOT/'results/software_fusion_batch_20261022')
    a=p.parse_args();result=audit(a.experiment_root.resolve())
    print(json.dumps({k:result[k] for k in ('status','selected','timing','incremental_saved_gm_read_bytes','norm_packing')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
