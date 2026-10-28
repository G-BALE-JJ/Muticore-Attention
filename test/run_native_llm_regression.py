#!/usr/bin/env python3
"""Same-build native LLM operator functional/timing acceptance matrix."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CASES = [dict(name='qkv_' + mode, operator='qkv', precision=mode, flags=[])
         for mode in ('int4', 'int8', 'fp16', 'bf16', 'tf32', 'fp32', 'fp8e4m3', 'fp8e5m2')]
CASES += [dict(name=operator, operator=operator, precision='fp16', flags=['--reuse', '1'])
          for operator in ('wo', 'gate', 'up', 'down', 'lm_head')]
CASES += [
    dict(name='qkv_striped', operator='qkv', precision='fp16', flags=['--panel-mapping', 'reuse-striped']),
    dict(name='qkv_replicated', operator='qkv', precision='fp16', flags=['--replicate-b']),
    dict(name='qkv_independent2', operator='qkv', precision='fp16',
         flags=['--instances', '2', '--panel-mapping', 'reuse-striped', '--replicate-b']),
    dict(name='qkv_independent4_mpi4', operator='qkv', precision='int8',
         flags=['--instances', '4', '--mpi-ranks', '4']),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--cpu-set', default='4-7')
    parser.add_argument('--case', action='append', choices=[c['name'] for c in CASES])
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    library = ROOT/'install/lib/sst-elements-library/libgolem.so'
    memory = library.with_name('libmemHierarchy.so')
    signature = (digest(library), digest(memory))
    report = dict(status='RUNNING', library_sha256=signature[0], memhierarchy_sha256=signature[1],
                  latency_unit='cycles', cases=[])

    def save():
        (root/'result.json').write_text(json.dumps(report, indent=2)+'\n')

    save()
    try:
        for case in CASES:
            if args.case and case['name'] not in args.case:
                continue
            pair = {}
            for mode in ('full-functional', 'full-timing'):
                if signature != (digest(library), digest(memory)):
                    raise ValueError('installed build changed')
                out = root/case['name']/mode
                command = [sys.executable, str(ROOT/'test/run.py'), 'prefill', '--smoke',
                           '--native-operator', case['operator'], '--artifact-root', str(out),
                           '--sim-mode', mode, '--', '--precision', case['precision'],
                           '--cpu-set', args.cpu_set, *case['flags']]
                env = {k: v for k, v in os.environ.items() if not k.startswith('GOLEM_')}
                env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
                with (root/(case['name']+'_'+mode+'.log')).open('w') as stream:
                    subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, check=True)
                result = json.loads((out/'native_case/result.json').read_text())
                if result['status'] != 'PASS' or not result['real_noc_hbm']:
                    raise ValueError('native integration failed')
                expected = 'PASS' if mode == 'full-functional' else 'SKIPPED'
                if any(n['status'] != expected for n in result['numerics']):
                    raise ValueError('unexpected numerical validation status')
                pair[mode] = result
            keys = ('cycles', 'output_tiles', 'native_workers', 'logical_shape', 'padded_shape', 'instances')
            if any(pair['full-functional'][k] != pair['full-timing'][k] for k in keys):
                raise ValueError('functional/timing workload or cycle mismatch: '+case['name'])
            report['cases'].append(dict(**case, status='PASS', delta_cycles=0, pairs=pair))
            save()
            print(case['name']+' PASS '+str(pair['full-timing']['cycles'])+' cycles', flush=True)
        if signature != (digest(library), digest(memory)):
            raise ValueError('installed build changed')
        report['status'] = 'PASS'
        save()
    except BaseException as error:
        report.update(status='FAIL', error=str(error))
        save()
        raise


if __name__ == '__main__':
    main()
