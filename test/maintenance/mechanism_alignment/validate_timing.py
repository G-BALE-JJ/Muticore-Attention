#!/usr/bin/env python3
"""Record timing GEMM before/after migration and with a same-build control."""
import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
METRICS = ('execution_summary.csv', 'terminal_summary.csv', 'memory_summary.csv',
           'dma_summary.csv', 'noc_summary.csv', 'hbm_read_command_summary.csv')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(name, fast_path):
    out = ROOT/'test/reports'/name
    out.mkdir(exist_ok=False)
    config = json.loads((ROOT/'test/GEMM/baseline.json').read_text())
    arguments = [x for x in config['arguments'] if x != '--verify-c']
    env = dict(os.environ, **config['environment'],
               GOLEM_ARTIFACT_ROOT=str(ROOT/'test/GEMM/results'/name),
               GOLEM_TIMING_NUMERICAL_FAST_PATH=str(fast_path),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    command = ['taskset', '-c', '20', 'bash', str(ROOT/'test/golem/run_noc_dma_pipeline.sh'),
               *arguments, '--sim-mode', 'full-timing', '--cpuset', '20']
    compute = ROOT/'install/lib/sst-elements-library/libgolem.so'
    memory = ROOT/'install/lib/sst-elements-library/libmemHierarchy.so'
    before = (digest(compute), digest(memory))
    with (out/'run.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    if before != (digest(compute), digest(memory)):
        raise ValueError('Installed libraries changed during timing run')
    stats = list((ROOT/'test/GEMM/results'/name).glob('**/execution_summary.csv'))
    if len(stats) != 1:
        raise ValueError('Expected one timing summary')
    def load(path):
        with path.open() as stream:
            return {x['metric']: x['value'] for x in csv.DictReader(stream)}
    measured = load(stats[0])
    terminal = load(stats[0].with_name('terminal_summary.csv'))
    cycles = float(measured['gemm_system_latency_cycles'])
    if cycles != 3500 or terminal['c_writeback_complete_ratio'] != '4/4':
        raise ValueError('Timing or final-output completion changed')
    numerical_modes = []
    if 'timingNumericalFastPath' in (ROOT/'src/sst/elements/golem/array/mvmComputeArray.h').read_text():
        pattern = r'\[ARRAY_NUMERICAL_DATA\] core=-?\d+ functional=(\d+) timing_fast_path=(\d+) materialize=(\d+)'
        for logfile in (ROOT/'test/GEMM/results'/name/'logs').glob('*.log'):
            numerical_modes.extend(tuple(map(int, row)) for row in re.findall(pattern, logfile.read_text(errors='replace')))
        if not numerical_modes or any(row != (0, fast_path, int(not fast_path)) for row in numerical_modes):
            raise ValueError('Backend did not observe the requested numerical materialization mode: '+str(numerical_modes))
    result = dict(status='PASS', cycles=cycles, latency_unit='cycles',
                  scope='first descriptor start through last WCP completion, inclusive +1',
                  mode='full-timing', timing_numerical_fast_path=fast_path, command=command,
                  library_sha256=before[0], memhierarchy_sha256=before[1],
                  metrics_sha256={file: digest(stats[0].with_name(file)) for file in METRICS},
                  final_c_writeback_complete_ratio=terminal['c_writeback_complete_ratio'],
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    result['observed_backend_mode'] = dict(functional=0, timing_fast_path=fast_path, materialize=int(not fast_path), observed_arrays=len(numerical_modes)) if numerical_modes else None
    return out, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        parser.error('name must be a directory name')
    control = None
    if args.compare:
        if 'timingNumericalFastPath' not in (ROOT/'src/sst/elements/golem/array/mvmComputeArray.h').read_text():
            raise ValueError('Same-build control requires the phase 06 validation switch')
        control_out, control = measure(args.name+'_same_build_control', 0)
        (control_out/'result.json').write_text(json.dumps(control, indent=2)+'\n')
    out, result = measure(args.name, 1)
    if args.compare:
        old = json.loads(args.compare.read_text())
        if old['status'] != 'PASS' or old['cycles'] != result['cycles'] or \
           old['final_c_writeback_complete_ratio'] != result['final_c_writeback_complete_ratio']:
            raise ValueError('Before/after timing evidence differs')
        result.update(paired_control=str(args.compare), delta_cycles=result['cycles']-old['cycles'])
        for key in ('library_sha256', 'memhierarchy_sha256', 'source_commit', 'cycles',
                    'final_c_writeback_complete_ratio', 'metrics_sha256'):
            if control[key] != result[key]:
                raise ValueError('Same-build timing control differs: '+key)
        result.update(same_build_control=str(control_out/'result.json'),
                      same_build_metrics_equal=True, same_build_delta_cycles=0)
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
