#!/usr/bin/env python3
"""Compare paired full-stack input-residency experiments on fixed hardware."""
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
    names=('fixed_control','fixed_candidate')
    cases={n:root/n/'reuse_window_probe' for n in names}
    results={n:read(c/'decoder_stack_result.json') for n,c in cases.items()}
    provenance={n:read(c/'decoder_stack_provenance.json') for n,c in cases.items()}
    reports={n:read(c.parent/'all_gemm_reuse_window_report.json') for n,c in cases.items()}
    for n in names:
        if results[n]['status']!='PASS' or reports[n]['status']!='PASS':
            raise ValueError('full-stack numerical/resource/GEMM coverage failed: '+n)
    a,b=names
    for key in ('guest_sha256','sst_library_sha256','sst_memhierarchy_sha256'):
        if provenance[a][key]!=provenance[b][key]:raise ValueError('paired binaries differ: '+key)
    builds={n:read(root/n/'experiment_provenance.json') for n in names}
    if builds[a]['source_sha256']!=builds[b]['source_sha256']:
        raise ValueError('paired runtime sources differ')
    for name,sha in builds[b]['source_sha256'].items():
        if digest(ROOT/name)!=sha:raise ValueError('measured runtime source changed: '+name)
    if digest(ROOT/'install/lib/sst-elements-library/libgolem.so')!=provenance[b]['sst_library_sha256']:
        raise ValueError('installed library differs from measured library')
    manifests={n:comparable_inputs(c) for n,c in cases.items()}
    flags={n:manifests[n].pop('input_residency',False) for n in names}
    if flags!={a:False,b:True} or manifests[a]!=manifests[b]:raise ValueError('input tensors/layout differ')
    envs={n:environment(provenance[n],cases[n]) for n in names}
    changes={k:dict(control=envs[a].get(k),candidate=envs[b].get(k))
             for k in set(envs[a])|set(envs[b]) if envs[a].get(k)!=envs[b].get(k)}
    if changes!={'GOLEM_DECODER_INPUT_RESIDENCY':dict(control='0',candidate='1')}:
        raise ValueError('unexpected configuration change: '+repr(changes))
    matrix=manifests[a];expected=matrix['sequence']*matrix['hidden']*2*matrix['layers']
    opt=results[b]['resources']['optimizations']['input_residency']
    if opt['saved_hbm_read_bytes']!=expected or opt['actual_payload']['saved_hbm_read_bytes']!=expected:
        raise ValueError('actual retained input saving mismatch')
    totals={n:job_totals(c) for n,c in cases.items()}
    if totals[a]['hbm_read']-totals[b]['hbm_read']!=expected:
        raise ValueError('decoder job HBM read saving mismatch')
    if totals[a]['hbm_write']!=totals[b]['hbm_write']:raise ValueError('HBM writes changed')
    flows={n:flow_stats(c,provenance[n]['environment']) for n,c in cases.items()}
    if flows[a]['traffic']!=flows[b]['traffic']:raise ValueError('dense GEMM work changed')
    timing={k:dict(control_cycles=v,candidate_cycles=scopes(results[b])[k],
                   reduction_percent=100*(1-scopes(results[b])[k]/v)) for k,v in scopes(results[a]).items()}
    fixed=read(ROOT/'results/software_dataflow_next_20261021/fixed_hardware_sources.json')
    layout='src/sst/elements/golem/attention/decoderMemoryLayout.h'
    old_layout_sha=fixed.pop(layout)
    layout_source=(ROOT/layout).read_text()
    original_layout=re.sub(r'    // Stage-chain Wo output temporarily aliases.*?"Wo handoff overlaps retained residual input"\);\n',
                           '',layout_source,flags=re.S)
    if hashlib.sha256(original_layout.encode()).hexdigest()!=old_layout_sha:
        raise ValueError('software allocator changed beyond the Wo scratch alias')
    for name,sha in fixed.items():
        if digest(ROOT/name)!=sha:raise ValueError('fixed hardware source changed: '+name)
    old_state=read(ROOT/'results/software_dataflow_next_20261021/before_state.json')
    original=ROOT.parent/'Muticore-Attention'
    if (subprocess.check_output(['git','diff','HEAD'],cwd=original,text=True)!=old_state['original_diff'] or
            digest(original/'install/lib/sst-elements-library/libgolem.so')!=old_state['original_library']):
        raise ValueError('original baseline changed')
    previous=ROOT/'results/software_dataflow_next_20261021/formal_resident_lm/reuse_window_probe'
    old=read(previous/'decoder_stack_result.json')
    for k in ('完整端到端','两层 decoder'):
        if scopes(old)[k]!=scopes(results[a])[k]:raise ValueError('control did not reproduce previous selected timing')
    selected=b if all(timing[k]['reduction_percent']>0 for k in ('完整端到端','两层 decoder')) else a
    result=dict(status='PASS',selected=selected,timing=timing,
                incremental_saved_hbm_read_bytes=expected,saved_hbm_write_bytes=0,
                previous_selected_bits=output_bits(previous,cases[b]),
                paired_bits=output_bits(cases[a],cases[b]),
                environment_changes=changes,decoder_job_hbm_totals=totals,
                hardware_sources_unchanged=len(fixed),original_preserved=True,
                allocator_change='adds Wo scratch alias and bounds assertion; all previous allocations/capacities unchanged',
                same_binaries_inputs_hardware=True,software_only=True,
                binaries={k:provenance[b][k] for k in ('guest_sha256','sst_library_sha256','sst_memhierarchy_sha256')},
                input_residency_audit=opt,
                runtime_source_sha256=builds[b]['source_sha256'],
                notes=['Cycles are simulated at 1 GHz; host execution time is excluded.',
                       'Dense GEMM MAC and A/B/C window work are identical.',
                       'HBM layer outputs remain materialized; original residual input rereads are eliminated.',
                       'An earlier candidate with a missing asynchronous phase transition was stopped and excluded.'])
    (root/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-root',type=Path,default=ROOT/'results/software_input_residency_next_20261021')
    a=p.parse_args();result=audit(a.experiment_root.resolve())
    print(json.dumps({k:result[k] for k in ('status','selected','timing','incremental_saved_hbm_read_bytes')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
