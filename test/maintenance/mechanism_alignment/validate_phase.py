#!/usr/bin/env python3
"""Gate one installed mechanism with four full functional stage runs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
STAGES = ('GEMM', 'Attention', 'GQA', 'prefill')
EXPECTED = dict(GEMM=3500, Attention=12852, GQA=37928, prefill=6084167.75)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase')
    args = parser.parse_args()
    if Path(args.phase).name != args.phase:
        parser.error('phase must be a single directory name')
    name = 'alignment_' + args.phase
    evidence = ROOT/'test/reports'/name
    evidence.mkdir(parents=True, exist_ok=True)
    result_path = evidence/'gate.json'
    if result_path.exists():
        parser.error('gate already exists; use a fresh phase name')
    result = dict(status='RUNNING', phase=args.phase,
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  library_sha256=digest(ROOT/'install/lib/sst-elements-library/libgolem.so'),
                  memhierarchy_sha256=digest(ROOT/'install/lib/sst-elements-library/libmemHierarchy.so'),
                  stages={})
    def save():
        result_path.write_text(json.dumps(result,indent=2)+'\n')
    save()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    def run(command, log):
        with (evidence/log).open('w') as stream:
            subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    try:
        for stage in STAGES:
            run([str(ROOT/'test'/stage/'run.sh'),'--check'],stage+'_host.log')
        selection = json.loads((ROOT/'test/attention_cluster_8qk_8pv/SELECTED_PREFILL_2026-10-26.json').read_text())
        selection['measured_library_sha256'] = result['library_sha256']
        selection['candidate_validation'] = dict(source_commit=result['source_commit'],phase=args.phase,
            public_selection_unchanged=True,reason='Rebuild of the candidate; original guest pin, hardware and schedule retained')
        config = evidence/'candidate_selection.json'
        config.write_text(json.dumps(selection,indent=2)+'\n')
        def stage_run(stage):
            command=[str(ROOT/'test'/stage/'run.sh'),'--run','--artifact-root',
                     str(ROOT/'test'/stage/'results'/name)]
            if stage=='prefill':
                command+=['--','--configuration',str(config),'--cpu-set','0-7']
            elif stage=='GEMM':
                command=['taskset','-c','20',*command,'--','--cpuset','20']
            else:
                command=['taskset','-c','16-19',*command]
            run(command,stage+'_formal.log')
            result['stages'][stage]='PASS'
            save()
            print(stage+' formal PASS',flush=True)
        # Attention and GQA share one guest build target: keep them sequential.
        # Prefill uses a separate guest target and separate CPU cores.
        with ThreadPoolExecutor(max_workers=2) as executor:
            prefill=executor.submit(stage_run,'prefill')
            for stage in STAGES[:-1]:
                stage_run(stage)
            prefill.result()
        run([sys.executable,str(ROOT/'test/summarize_stages.py'),
             '--run-name',name,'--output-dir',str(evidence),
             '--configuration',str(config)],'summary.log')
        summary=json.loads((evidence/'summary.json').read_text())
        if summary['memhierarchy_sha256'] != result['memhierarchy_sha256']:
            raise ValueError('Memory library changed while the gate was running')
        measured={row['stage']:row['end_to_end_cycles'] for row in summary['stages']}
        if summary['status']!='PASS' or measured!=EXPECTED:
            raise ValueError('Formal functional/cycle contract changed: '+str(measured))
        result.update(status='PASS',cycles=measured,delta_from_public_cycles={s:measured[s]-EXPECTED[s] for s in STAGES})
        save()
        print(json.dumps(result,indent=2),flush=True)
    except BaseException as error:
        result.update(status='FAIL',error=str(error))
        save()
        raise


if __name__=='__main__':
    main()
