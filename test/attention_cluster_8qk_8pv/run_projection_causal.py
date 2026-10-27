#!/usr/bin/env python3
"""Run fixed-hardware Projection causal traces and one-factor scheduling experiments."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

from run_batch_b import environment, SOURCE_FILES
from run_projection_overlap import ROOT, HERE, digest, summarize
from analyze_resource_timeline import analyze as analyze_resources
from analyze_critical_path import analyze as analyze_critical
from analyze_projection_startup import startup_metrics
from analyze_projection_holes import analyze as analyze_holes


def analyze_clean(case, resources):
    for p in (case / 'stdout').rglob('stderr-*'):
        if re.search(r'\[(?:ATTENTION_GUEST_START|PROJECTION_START|PROJECTION_DISPATCH)\]', p.read_text()):
            raise ValueError('startup diagnostics executed in clean guest')
    projection = json.loads((case / 'projection_e2e_result.json').read_text())
    log = next((case / 'logs').glob('attention-sst*.log'))
    simulation_end = None
    with log.open(errors='replace') as stream:
        for line in stream:
            m = re.search(r'Simulation is complete, simulated time:\s*([\d.]+)\s*([munp]?s)', line)
            if m:
                simulation_end = float(m[1]) * {'s':1e9, 'ms':1e6, 'us':1e3, 'ns':1, 'ps':1e-3}[m[2]]
    if simulation_end is None:
        raise ValueError('simulation completion time missing')
    return startup_metrics(projection, resources, simulation_end)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--stage', choices=('small','formal'), default='formal')
    parser.add_argument('--sequence', type=int, required=True)
    parser.add_argument('--cpu-set', default='0-3')
    parser.add_argument('--variant', choices=('baseline','budget1','depth8'), default='baseline')
    args = parser.parse_args()
    case, reference = args.artifact_root.resolve(), args.reference.resolve()
    case.mkdir(parents=True, exist_ok=False)
    guest_dir = ROOT / 'test/golem/small/muticore_attention'
    guest = guest_dir / 'riscv64/clean_startup/fused_attention'
    sources = list(dict.fromkeys(SOURCE_FILES + [guest,
        guest_dir/'golem_attention_runtime.cpp', guest_dir/'golem_attention_runtime.h',
        guest_dir/'golem_projection_runtime.h', guest_dir/'Makefile',
        HERE/'run_projection_causal.py', HERE/'analyze_projection_holes.py', HERE/'analyze_projection_startup.py',
        HERE/'run_llama_projection.py', HERE/'run_causal_balance.py']))
    hashes = {str(p.relative_to(ROOT)):digest(p) for p in sources}
    for source in sources:
        target=case/'source_snapshot'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    env=environment(case,(1,1,1),'none',args.sequence,args.stage)
    env['GOLEM_PROJECTION_CAUSAL_TRACE']='1'
    if args.variant == 'budget1': env['GOLEM_PROJECTION_INPUT_PREFETCH_BUDGET']='1'
    if args.variant == 'depth8': env['GOLEM_PROJECTION_WEIGHT_PREFETCH_DEPTH']='8'
    env['GOLEM_ATTENTION_GUEST_EXECUTABLE']=str(guest)
    env['GOLEM_PROJECTION_INPUT_REFERENCE']=str(reference)
    env['GOLEM_MPI_ARGS']=f'--bind-to core --cpu-set {args.cpu_set} --output-filename {case / "mpi_rank_output"}'
    if env.get('VANADIS_CPU_CLOCK')!='1.0GHz':raise ValueError('platform must remain 1 GHz')
    heads=(32,8) if args.stage=='formal' else (4,2)
    command=[str(HERE/'run_sst.sh'),'--projection','--query-length',str(args.sequence),
        '--kv-length',str(args.sequence),'--num-query-heads',str(heads[0]),
        '--num-kv-heads',str(heads[1]),'--head-dim','64','--causal','--rope',
        '--artifact-root',str(case)]
    provenance={'sha256':hashes,'command':command,'variant':args.variant,
        'guest_sha256':digest(guest),'compile_time_startup_diagnostics':0,
        'compile_time_batch_release':0,
        'environment':{k:v for k,v in env.items() if k.startswith(('GOLEM_','VANADIS_'))}}
    (case/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(f'Running causal {args.variant} S{args.sequence}',flush=True)
    with (case/'driver.log').open('w') as stream:
        subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    if hashes!={str(p.relative_to(ROOT)):digest(p) for p in sources}:
        raise ValueError('source or binary changed during clean startup experiment')
    row=summarize(case,reference)
    resources=analyze_resources(case,allow_filtered=True)
    (case/'resource_analysis.json').write_text(json.dumps(resources,indent=2)+'\n')
    critical=analyze_critical(case)
    if critical['reconstructed_e2e_cycles']!=row['end_to_end_cycles']:
        raise ValueError('E2E reconstruction mismatch')
    (case/'critical_path_analysis.json').write_text(json.dumps(critical,indent=2)+'\n')
    inputs={p.name:digest(p) for p in sorted(case.glob('*.bin'))}
    provenance['input_sha256']=inputs
    (case/'baseline_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    row.update({'variant':args.variant,'sha256':hashes,'input_sha256':inputs,
                'startup':analyze_clean(case,resources)})
    if any(v>=1000 for v in row['startup']['manager_release_span_cycles'].values()):
        raise ValueError('clean startup still has serialized diagnostic gaps')
    (case/'experiment_result.json').write_text(json.dumps(row,indent=2)+'\n')
    holes=analyze_holes(case)
    (case/'projection_holes.json').write_text(json.dumps(holes,indent=2)+'\n')
    print(f'PASS causal {args.variant} S{args.sequence}: E2E={row["end_to_end_cycles"]}, accept span={row["startup"]["worker_accept_span_cycles"]}',flush=True)


if __name__=='__main__':main()
