#!/usr/bin/env python3
"""Run one archived baseline against the public architecture or a feature workspace."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('GEMM', 'Attention', 'GQA', 'prefill'))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='run host contract and numerical tests')
    mode.add_argument('--show-config', action='store_true', help='show command without executing it')
    mode.add_argument('--dry-run', action='store_true', help='validate configuration or generate prefill schedule')
    mode.add_argument('--run', action='store_true', help='run SST and existing numerical/resource validators')
    mode.add_argument('--smoke', action='store_true', help='run a small SST case; prefill uses one layer and one logical token')
    parser.add_argument('--workspace', type=Path, default=ROOT, help='architecture workspace to test')
    parser.add_argument('--artifact-root', type=Path)
    parser.add_argument('--sim-mode', choices=('full-functional', 'full-timing'))
    parser.add_argument('--native-operator',choices=('qkv','wo','gate','up','down','lm_head'),help='opt-in native packed LLM operator experiment (prefill only)')
    args, extra = parser.parse_known_args()
    if extra[:1] == ['--']:
        extra = extra[1:]
    mode_parser=argparse.ArgumentParser(add_help=False)
    mode_parser.add_argument('--sim-mode', choices=('full-functional','full-timing'))
    forwarded_mode,extra=mode_parser.parse_known_args(extra)
    sim_mode=forwarded_mode.sim_mode or args.sim_mode or os.environ.get('GOLEM_SIM_MODE','full-functional')
    if sim_mode not in ('full-functional','full-timing'):parser.error('invalid GOLEM_SIM_MODE')
    if not any((args.check, args.show_config, args.dry_run, args.run, args.smoke)):
        args.check = True
    workspace = args.workspace.resolve()
    if not (workspace / 'configure.ac').is_file():
        parser.error('--workspace must contain the SST source layout')
    config = json.loads((ROOT / 'test' / args.stage / 'baseline.json').read_text())
    golem = workspace / 'test/golem'
    baseline = workspace / 'test/attention_cluster_8qk_8pv'
    # An explicitly selected older experimental worktree has its own native
    # layout. The public workspace never depends on these compatibility paths.
    if workspace != ROOT:
        if not golem.is_dir():
            golem = workspace / 'src' / 'sst' / 'elements' / 'golem' / 'tests'
        if not baseline.is_dir():
            baseline = workspace / 'baseline' / 'attention_cluster_8qk_8pv'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    for key, value in config.get('environment', {}).items():
        env.setdefault(key, value)
    sfu_precision = env.setdefault('GOLEM_SFU_INTERNAL_PRECISION', 'fp16')
    if sfu_precision not in ('fp16', 'fp32'):
        parser.error('GOLEM_SFU_INTERNAL_PRECISION must be fp16 or fp32')
    sfu_backend = env.setdefault('GOLEM_SFU_NONLINEAR_BACKEND', 'lut')
    if sfu_backend not in ('lut', 'libm'):
        parser.error('GOLEM_SFU_NONLINEAR_BACKEND must be lut or libm')
    if sfu_backend == 'lut' and sfu_precision != 'fp16':
        parser.error('GOLEM_SFU_NONLINEAR_BACKEND=lut requires fp16')
    if args.native_operator and (args.stage!='prefill' or args.check):
        parser.error('--native-operator requires prefill --run/--smoke/--dry-run/--show-config')
    if args.check:
        if extra:
            parser.error('extra simulator arguments require --run, --smoke or --dry-run')
        for suite, patterns in config['host_checks'].items():
            directory = {'guest': golem / 'small/muticore_attention', 'golem': golem, 'baseline': baseline}[suite]
            for pattern in patterns:
                subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(directory), '-p', pattern],
                               cwd=workspace, env=env, check=True)
        return
    now = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    artifacts = (args.artifact_root or ROOT / 'test' / args.stage / 'results' / now).resolve()
    if args.native_operator:
        command=[sys.executable,str(golem/'run_native_llm_operator.py'),'--artifact-root',str(artifacts/'native_case'),
                 '--operator',args.native_operator,'--shape','small' if args.smoke else 'formal']
        if args.dry_run:command.append('--prepare-only')
    elif args.stage == 'GEMM':
        arguments=[x for x in config['arguments'] if not (sim_mode=='full-timing' and x=='--verify-c')]
        command = ['bash', str(golem / 'run_noc_dma_pipeline.sh'), *arguments,
                   '--tensor-dir', str(artifacts / 'inputs')]
        env.update(GOLEM_ARTIFACT_ROOT=str(artifacts))
        env.setdefault('GOLEM_TEST_BINARY',str(artifacts/'guest/test_noc_dma'))
        if args.dry_run:
            command.append('--dry-run')
    elif args.stage in ('Attention', 'GQA'):
        command = ['bash', str(workspace / 'scripts/test_flash_attention.sh'), *config['arguments'],
                   '--artifact-root', str(artifacts)]
        if args.smoke:
            # The shared reuse-window engine needs full 64-row worker blocks;
            # the placement validator accepts at least two MPI ranks.
            command += ['--query-length', '512', '--kv-length', '512', '--mpi-ranks', '2']
        if args.dry_run:
            command.append('--show-config')
    elif args.smoke:
        # This driver creates its case with exist_ok=False; metadata stays in
        # the enclosing invocation directory.
        command = [sys.executable, str(baseline / 'run_decoder_stack.py'), '--artifact-root', str(artifacts / 'case'),
                   '--shape', 'small', '--layers', '1', '--sequence', '1', '--embedding', '--lm-head',
                   '--mpi-ranks', '2']
    else:
        command = [sys.executable, str(baseline / 'run_selected_prefill.py'), '--artifact-root', str(artifacts)]
        if args.dry_run:
            command.append('--schedule-only')
    command += [*extra,'--sim-mode',sim_mode]
    env['GOLEM_SIM_MODE']=sim_mode
    details = dict(stage=args.stage, native_operator=args.native_operator, sim_mode=sim_mode, workspace=str(workspace), artifact_root=str(artifacts), command=command,
                   sfu_internal_precision=sfu_precision,
                   environment_overrides={k: env[k] for k in (*config.get('environment', {}), 'GOLEM_SFU_INTERNAL_PRECISION', 'GOLEM_SFU_NONLINEAR_BACKEND')},
                   latency_unit='cycles', mode='smoke' if args.smoke else 'dry-run' if args.dry_run else 'run')
    print(json.dumps(details, indent=2, ensure_ascii=False), flush=True)
    if args.show_config:
        return
    # Each invocation has its own output directory; never overwrite previous evidence.
    if artifacts.exists() and any(artifacts.iterdir()):
        parser.error(f'output directory is not empty: {artifacts}')
    artifacts.mkdir(parents=True, exist_ok=True)
    if args.stage=='GEMM':Path(env['GOLEM_TEST_BINARY']).parent.mkdir(parents=True,exist_ok=True)
    details['library_sha256'] = digest(workspace / 'install/lib/sst-elements-library/libgolem.so')
    details['git_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=workspace, text=True).strip()
    details['git_status'] = subprocess.check_output(['git', 'status', '--porcelain'], cwd=workspace, text=True)
    metadata = artifacts / 'stage_invocation.json'
    metadata.write_text(json.dumps(details, indent=2, ensure_ascii=False) + '\n')
    print(shlex.join(command), flush=True)
    # Inherited console output preserves existing simulator progress reporting.
    result = subprocess.run(command, cwd=workspace, env=env)
    if result.returncode==0 and args.stage=='GEMM' and sim_mode=='full-timing':
        sys.path.insert(0,str(baseline))
        from timing_mode import verify_timing_backend
        logs=list((artifacts/'logs').glob('*.log'))
        if len(logs)!=1:raise ValueError('expected one GEMM runtime log')
        (artifacts/'timing_backend.json').write_text(json.dumps(verify_timing_backend(logs[0]),indent=2)+'\n')
    details.update(exit_code=result.returncode, status='PASS' if result.returncode == 0 else 'FAIL')
    metadata.write_text(json.dumps(details, indent=2, ensure_ascii=False) + '\n')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
