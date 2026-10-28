#!/usr/bin/env python3
"""Run functional/timing pairs across shapes, GQA groups, layers and MPI ranks."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
CASES=[
 dict(name='gemm_k64',stage='GEMM',flags=['--gemm-k','64']),
 dict(name='gemm_k192_mpi2',stage='GEMM',flags=['--gemm-m','256','--gemm-k','192','--mpi-ranks','2']),
 dict(name='attention_d64_mpi2',stage='Attention',flags=['--query-length','512','--kv-length','512','--head-dim','64','--mpi-ranks','2']),
 dict(name='attention_causal_rope',stage='Attention',flags=['--query-length','512','--kv-length','512','--head-dim','64','--causal','--rope','--mpi-ranks','4']),
 dict(name='gqa_group4',stage='GQA',flags=['--query-length','512','--kv-length','512','--head-dim','64','--num-query-heads','4','--num-kv-heads','1','--mpi-ranks','2']),
 dict(name='gqa_group2_d128',stage='GQA',flags=['--query-length','512','--kv-length','1024','--head-dim','128','--num-query-heads','4','--num-kv-heads','2','--mpi-ranks','4']),
 dict(name='prefill_token1',stage='prefill',flags=['--layers','1','--sequence','1','--mpi-ranks','2']),
 dict(name='prefill_l2_token65',stage='prefill',flags=['--layers','2','--sequence','65','--mpi-ranks','4']),
 dict(name='prefill_padded2048',stage='prefill',flags=['--layers','1','--sequence','1025','--mpi-ranks','4']),
]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def result(stage,root):
    read=lambda name:json.loads((root/name).read_text())
    invocation=read('stage_invocation.json')
    if invocation['status']!='PASS':raise ValueError('stage invocation did not pass')
    if stage=='GEMM':
        paths=list(root.glob('**/execution_summary.csv'))
        if len(paths)!=1:raise ValueError('expected one GEMM timing report')
        metrics={x['metric']:x['value'] for x in csv.DictReader(paths[0].open())}
        terminal={x['metric']:x['value'] for x in csv.DictReader(paths[0].with_name('terminal_summary.csv').open())}
        ratio=terminal['c_writeback_complete_ratio'];done,want=map(int,ratio.split('/'))
        if not want or done!=want:raise ValueError('incomplete GEMM final C writeback')
        return dict(cycles=float(metrics['gemm_system_latency_cycles']),completion=dict(final_c_writebacks=done),
                    scope='first descriptor start through last WCP completion, inclusive +1')
    if stage in ('Attention','GQA'):
        d=read('sst_qk_bridge_result.json')
        if d['status']!='PASS':raise ValueError('attention lifecycle failed')
        return dict(cycles=d['end_to_end_cycles'],scope=d['scope'],
                    completion=dict(jobs=d['completed_jobs'],stages={k:v['count'] for k,v in d['stage_spans'].items()}))
    d=read('case/decoder_stack_result.json')
    for key in ['lifecycle','resources','gm_banks','final_norm','lm_head','embedding']:
        if d[key]['status']!='PASS':raise ValueError('prefill '+key+' check failed')
    return dict(cycles=d['embedding']['end_to_end_cycles'],scope=d['embedding']['contract'],
        completion=dict(lifecycle=d['lifecycle'],embedding_bytes=d['embedding']['output_ack_bytes'],
                        logits_bytes=d['lm_head']['output_ack_bytes'],lm_jobs=d['lm_head']['worker_jobs'],shape=d['shape']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root',type=Path)
    p.add_argument('--case',action='append',choices=[c['name'] for c in CASES])
    p.add_argument('--list',action='store_true')
    p.add_argument('--cpu-set',default='0-3')
    a=p.parse_args();cases=[c for c in CASES if not a.case or c['name'] in a.case]
    if a.list:print(json.dumps(cases,indent=2));return
    if not a.artifact_root:p.error('--artifact-root is required')
    root=a.artifact_root.resolve();root.mkdir(parents=True,exist_ok=False)
    library=ROOT/'install/lib/sst-elements-library/libgolem.so';memory=library.with_name('libmemHierarchy.so')
    signature=(sha(library),sha(memory));report=dict(status='RUNNING',latency_unit='cycles',library_sha256=signature[0],memhierarchy_sha256=signature[1],cases=[])
    def save():(root/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        for case in cases:
            pairs={}
            for mode in ('full-functional','full-timing'):
                if signature!=(sha(library),sha(memory)):raise ValueError('installed build changed')
                out=root/case['name']/mode
                command=[sys.executable,str(ROOT/'test/run.py'),case['stage'],
                         '--smoke' if case['stage']=='prefill' else '--run','--artifact-root',str(out),'--sim-mode',mode,'--',*case['flags']]
                if case['stage']=='prefill':command+=['--cpu-set',a.cpu_set]
                env={k:v for k,v in os.environ.items() if not k.startswith('GOLEM_')}
                env.update(GOLEM_MPI_ARGS='--bind-to core --cpu-set '+a.cpu_set,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
                with (root/(case['name']+'_'+mode+'.log')).open('w') as stream:
                    subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
                pairs[mode]=dict(result(case['stage'],out),artifact_root=str(out))
            if signature!=(sha(library),sha(memory)):raise ValueError('installed build changed')
            if pairs['full-functional']['cycles']!=pairs['full-timing']['cycles'] or pairs['full-functional']['completion']!=pairs['full-timing']['completion']:
                raise ValueError('functional/timing cycles or completed workload differ for '+case['name'])
            report['cases'].append(dict(**case,status='PASS',delta_cycles=0,pairs=pairs));save()
            print(case['name']+' PASS '+str(pairs['full-timing']['cycles'])+' cycles',flush=True)
        report['status']='PASS';save()
    except BaseException as error:
        report.update(status='FAIL',error=str(error));save();raise
if __name__=='__main__':main()
