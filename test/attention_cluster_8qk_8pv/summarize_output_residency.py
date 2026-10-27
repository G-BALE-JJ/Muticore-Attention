#!/usr/bin/env python3
"""Compare paired intermediate-output elision runs on the fixed hardware."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from summarize_reuse_window_optimization import ROOT, read, digest, comparable_inputs, scopes
from summarize_software_dataflow_next import environment
from summarize_software_stage_chain import output_bits, job_totals
from summarize_gm_prefetch import flow_stats


def audit(root):
    names=('final_control','final_candidate');a,b=names
    cases={n:root/n/'reuse_window_probe' for n in names}
    results={n:read(c/'decoder_stack_result.json') for n,c in cases.items()}
    reports={n:read(root/n/'all_gemm_reuse_window_report.json') for n in names}
    provenance={n:read(c/'decoder_stack_provenance.json') for n,c in cases.items()}
    builds={n:read(root/n/'experiment_provenance.json') for n in names}
    for n in names:
        if results[n]['status']!='PASS' or reports[n]['status']!='PASS':
            raise ValueError('full-stack numerical/resource/reuse-window audit failed: '+n)
        if read(cases[n]/'verification_provenance.json')['verifier_source_sha256']!=digest(ROOT/'test/attention_cluster_8qk_8pv/run_decoder_stack.py'):
            raise ValueError('numerical verifier changed after audit: '+n)
    proof=read(cases[b]/'verification_retained_input.json')
    if (not proof['candidate_hbm_input_intentionally_absent'] or
            proof!=reports[b]['projection_panels']['input_reference'] or
            proof!=results[b]['numerics']['last_layer_input_reference'] or
            digest(Path(proof['reference_result']))!=proof['reference_result_sha256']):
        raise ValueError('retained-input host oracle proof differs between checks')
    binary_keys=('guest_sha256','sst_library_sha256','sst_memhierarchy_sha256')
    if any(provenance[a][k]!=provenance[b][k] for k in binary_keys):raise ValueError('paired binaries differ')
    if builds[a]['source_sha256']!=builds[b]['source_sha256']:raise ValueError('paired sources differ')
    for name,sha in builds[b]['source_sha256'].items():
        if digest(ROOT/name)!=sha:raise ValueError('measured runtime source changed: '+name)
    if digest(ROOT/'install/lib/sst-elements-library/libgolem.so')!=provenance[b]['sst_library_sha256']:
        raise ValueError('installed and measured libraries differ')
    manifests={n:comparable_inputs(c) for n,c in cases.items()}
    flags={n:manifests[n].pop('elide_intermediate_output',False) for n in names}
    if flags!={a:False,b:True} or manifests[a]!=manifests[b]:raise ValueError('paired inputs/layout differ')
    envs={n:environment(provenance[n],cases[n]) for n in names}
    changes={k:dict(control=envs[a].get(k),candidate=envs[b].get(k))
             for k in set(envs[a])|set(envs[b]) if envs[a].get(k)!=envs[b].get(k)}
    if changes!={'GOLEM_DECODER_ELIDE_INTERMEDIATE_OUTPUT':dict(control='0',candidate='1')}:
        raise ValueError('unexpected configuration change: '+repr(changes))
    m=manifests[a];expected=m['sequence']*m['hidden']*2*(m['layers']-1)
    opt=results[b]['resources']['optimizations']['output_residency']
    if opt['saved_hbm_write_bytes']!=expected or opt['actual_gm_payload_bytes']!=expected:
        raise ValueError('actual output write saving mismatch')
    totals={n:job_totals(c) for n,c in cases.items()}
    if totals[a]['hbm_write']-totals[b]['hbm_write']!=expected or totals[a]['hbm_read']!=totals[b]['hbm_read']:
        raise ValueError('HBM traffic differs beyond intermediate Y mirror writes')
    flows={n:flow_stats(c,provenance[n]['environment']) for n,c in cases.items()}
    if flows[a]['traffic']!=flows[b]['traffic']:raise ValueError('dense GEMM/reuse-window work changed')
    timing={k:dict(control_cycles=v,candidate_cycles=scopes(results[b])[k],
                   reduction_percent=100*(1-scopes(results[b])[k]/v)) for k,v in scopes(results[a]).items()}
    fixed=read(ROOT/'results/software_dataflow_next_20261021/fixed_hardware_sources.json')
    layout='src/sst/elements/golem/attention/decoderMemoryLayout.h';old_layout_sha=fixed.pop(layout)
    original_layout=re.sub(r'    // Stage-chain Wo output temporarily aliases.*?"Wo handoff overlaps retained residual input"\);\n',
                           '',(ROOT/layout).read_text(),flags=re.S)
    if hashlib.sha256(original_layout.encode()).hexdigest()!=old_layout_sha:
        raise ValueError('allocator changed beyond existing software alias')
    for name,sha in fixed.items():
        if digest(ROOT/name)!=sha:raise ValueError('hardware source changed: '+name)
    old_state=read(ROOT/'results/software_dataflow_next_20261021/before_state.json')
    original=ROOT / 'test/history/legacy-primary'
    if (subprocess.check_output(['git','diff','HEAD'],cwd=original,text=True)!=old_state['original_diff'] or
            digest(original/'install/lib/sst-elements-library/libgolem.so')!=old_state['original_library']):
        raise ValueError('original baseline changed')
    previous=ROOT/'results/software_input_residency_next_20261021/fixed_candidate/reuse_window_probe'
    old=read(previous/'decoder_stack_result.json')
    for k in ('完整端到端','两层 decoder'):
        if scopes(old)[k]!=scopes(results[a])[k]:raise ValueError('control failed to reproduce prior selected timing')
    selected=b if all(timing[k]['reduction_percent']>0 for k in ('完整端到端','两层 decoder')) else a
    result=dict(status='PASS',selected=selected,timing=timing,
                incremental_saved_hbm_read_bytes=0,incremental_saved_hbm_write_bytes=expected,
                previous_selected_bits=output_bits(previous,cases[b]),paired_bits=output_bits(cases[a],cases[b]),
                environment_changes=changes,decoder_job_hbm_totals=totals,reuse_window_flow=flows,
                output_residency_audit=opt,hardware_sources_unchanged=len(fixed),original_preserved=True,
                all_gemm_coverage={n:{k:reports[n]['coverage'][k] for k in
                    ('status','completed_window_jobs','tile_group_launches','workers','qk_completed_jobs',
                     'pv_completed_windows','c_traffic_status','window_flow_status','pipeline_execution_status','input_pipeline_status')}
                    for n in names},
                projection_panels={n:reports[n]['projection_panels'] for n in names},
                host_validation_source_sha256={name:digest(ROOT/'test/attention_cluster_8qk_8pv'/name) for name in
                    ('run_decoder_stack.py','run_all_gemm_reuse_window_baseline.py','verification_retained_input.py',
                     'verify_output_residency.py','verify_decoder_dataflow_resources.py','summarize_output_residency.py')},
                next_token=results[b]['lm_head']['next_token']['token_id'],
                same_binaries_inputs_hardware=True,software_only=True,
                binaries={k:provenance[b][k] for k in binary_keys},runtime_source_sha256=builds[b]['source_sha256'],
                notes=['Simulated cycles at 1 GHz; host runtime excluded.',
                       'Only intermediate Y HBM mirrors are omitted. GM payload/publication and completion DMA remain timed.',
                       'Final Y, final Norm and logits retain their HBM outputs.',
                       'Host-only local checks use a verified materialized counterpart for the elided input; the independent composed reference still checks the complete chain through final Y.',
                       'Host validation was updated after simulation launch to handle absent intermediate HBM data; simulated runtime binaries and measured source hashes did not change.',
                       'The initial pair was stopped before completion to tighten the residual-input residency guard; it is excluded.'])
    (root/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-root',type=Path,default=ROOT/'results/software_output_residency_20261022')
    a=p.parse_args();result=audit(a.experiment_root.resolve())
    print(json.dumps({k:result[k] for k in ('status','selected','timing','incremental_saved_hbm_write_bytes')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
