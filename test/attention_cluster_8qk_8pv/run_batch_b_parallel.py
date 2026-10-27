#!/usr/bin/env python3
"""Run independent SST cases on four disjoint host core sets (no agent workers)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

from run_batch_b import HERE, ROOT
from summarize_batch_b import combine, markdown

SHARDS=(('oracle_gm_read','oracle_worker_start','matrix_W','matrix_IW'),
        ('oracle_matrix_program','oracle_projection_barrier','matrix_WG','matrix_IWG'),
        ('oracle_input_scatter','oracle_attention_barrier','matrix_I'),
        ('oracle_weight_arrival','matrix_G','matrix_IG'))


def host_cores(text):
    cores=set()
    for piece in text.split(','):
        if '-' in piece:
            lo,hi=map(int,piece.split('-'))
            if lo>hi:raise ValueError('reversed host core range')
            cores.update(range(lo,hi+1))
        else:cores.add(int(piece))
    if len(cores)!=4:raise ValueError('each MPI host core set must contain four physical cores')
    return cores


def validate_host_sets(sets,physical_cores):
    occupied=set()
    if len(sets)!=4:raise ValueError('four host core sets required')
    for text in sets:
        cores=host_cores(text)
        if cores & occupied:raise ValueError('host MPI core sets overlap')
        if not cores <= physical_cores:raise ValueError('host CPU set exceeds physical-core IDs; SMT threads are not core IDs')
        occupied |= cores


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root',type=Path,required=True)
    p.add_argument('--reference-root',type=Path,required=True)
    p.add_argument('--cpu-sets',nargs=4,default=['0-3','4-7','8-11','12-15'])
    args=p.parse_args()
    topology=subprocess.check_output(['lscpu','-p=CORE,SOCKET'],text=True)
    identities={tuple(map(int,line.split(','))) for line in topology.splitlines() if line and not line.startswith('#')}
    # Open MPI's logical physical-core indices are contiguous in hwloc order.
    validate_host_sets(args.cpu_sets,set(range(len(identities))))
    root=args.artifact_root.resolve();root.mkdir(parents=True,exist_ok=False)
    def launch(label,options):
        with (root/f'{label}.log').open('w') as stream:
            subprocess.run([sys.executable,str(HERE/'run_batch_b.py'),*options],cwd=ROOT,
                           stdout=stream,stderr=subprocess.STDOUT,check=True)
        return root/label
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(launch,f'baseline_{s}',[
            '--stage','formal','--suite','baseline','--sequences',str(s),
            '--artifact-root',str(root/f'baseline_{s}'),'--reference-root',str(args.reference_root.resolve()),
            '--cpu-set',args.cpu_sets[index]]) for index,s in enumerate((1024,2048))]
        for future in futures:future.result()
    baseline=root/'reference_baseline';baseline.mkdir()
    for s in (1024,2048):
        (baseline/f's{s}').symlink_to(root/f'baseline_{s}'/'baseline'/f's{s}')
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(launch,f'shard_{index}',[
            '--stage','formal','--suite','all','--sequences','1024','2048',
            '--artifact-root',str(root/f'shard_{index}'),'--reference-root',str(baseline),
            '--case-names',*cases,'--cpu-set',args.cpu_sets[index]]) for index,cases in enumerate(SHARDS)]
        shards=[future.result() for future in futures]
    summary=combine(baseline,shards)
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (root/'summary.md').write_text(markdown(summary))
    print('PASS: 30 independently verified formal cases')


if __name__=='__main__':main()
