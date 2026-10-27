#!/usr/bin/env python3
"""Remove tensors only from stale failed/incomplete historical experiments.

Logs, metadata, snapshots, shared inputs outside a case, and any case with a
successful validation report are retained. The manifest is written first.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

LLM = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPOS = [LLM / 'Muticore-Attention', LLM / 'Muticore-Attention-all-gemm']
REPORT = 'all_gemm_reuse_window_report.json'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def active_simulators():
    found = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            cmd = (proc / 'cmdline').read_bytes().split(b'\0')
            executable = Path(os.fsdecode(cmd[0])).name
            if executable in ('sst', 'sstsim.x', 'mpirun', 'mpiexec', 'orterun'):
                found.append(os.fsdecode(b' '.join(cmd)))
        except (OSError, IndexError):
            pass
    return found


def signature(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns]


def protected_state():
    state = {}
    for repo in REPOS:
        state[str(repo)] = {
            'git_status': subprocess.check_output(['git', 'status', '--porcelain', '-uall'], cwd=repo).decode(),
            'source_sha256': {str(p.relative_to(repo)): sha256(p)
                              for folder in ('src', 'scripts', 'baseline')
                              for p in (repo / folder).rglob('*')
                              if p.is_file() and not p.is_symlink()
                              and p.suffix in ('.h', '.cc', '.cpp', '.py', '.sh', '.md')},
            'library_sha256': sha256(repo / 'install/lib/sst-elements-library/libgolem.so'),
            'pass_reports': {str(p.relative_to(repo)): sha256(p)
                             for p in (repo / 'results').rglob('*.json')
                             if p.stat().st_size < 4 * 1024**2 and is_validated(p)},
        }
    return state


def is_validated(path):
    if any(word in path.name for word in ('schedule', 'placement', 'partition', 'artifact_cleanup')):
        return False
    try:
        data = json.loads(path.read_text())
        return isinstance(data, dict) and data.get('status') == 'PASS'
    except (ValueError, OSError):
        return False


def plan():
    cases, skipped = [], []
    # All runs here predate 2026-10-27 00:00 Asia/Shanghai.
    cutoff = 1793030400
    for repo in REPOS:
        for report in sorted((repo / 'results').rglob(REPORT)):
            data = json.loads(report.read_text())
            if data.get('status') not in ('FAIL', 'RUNNING'):
                continue
            case = report.parent
            passed = [str(p.relative_to(case)) for p in case.rglob('*.json')
                      if p.stat().st_size < 4 * 1024**2 and is_validated(p)]
            if passed:
                skipped.append({'case': str(case), 'reason': 'contains successful validation', 'reports': passed})
                continue
            files = []
            for p in sorted(case.rglob('*')):
                if p.is_symlink() or not p.is_file() or p.suffix not in ('.bin', '.npy'):
                    continue
                p.resolve().relative_to(case.resolve())
                s = p.stat()
                if s.st_mtime >= cutoff or s.st_nlink != 1:
                    raise RuntimeError(f'Unexpected recent or linked file: {p}')
                files.append({'path': str(p), 'logical_bytes': s.st_size,
                              'allocated_bytes': s.st_blocks * 512, 'signature': signature(p)})
            if files:
                cases.append({'case': str(case), 'status': data['status'],
                              'report_sha256': sha256(report), 'files': files})
    return {'cases': cases, 'skipped': skipped,
            'allocated_bytes': sum(f['allocated_bytes'] for c in cases for f in c['files'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if active_simulators():
        raise RuntimeError('Simulator processes are active; cleanup stopped')
    manifest = plan()
    manifest_path = HERE / 'manifest.json'
    if manifest_path.exists() and args.apply:
        raise RuntimeError('An audit already exists; do not overwrite it')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'cases': len(manifest['cases']), 'allocated_gib': manifest['allocated_bytes'] / 2**30,
                      'skipped': len(manifest['skipped'])}), flush=True)
    for c in manifest['cases']:
        print(c['status'], c['case'], round(sum(f['allocated_bytes'] for f in c['files']) / 2**30, 3), flush=True)
    if not args.apply:
        return
    before = protected_state()
    (HERE / 'protected_before.json').write_text(json.dumps(before, indent=2) + '\n')
    # Verify every candidate before the first deletion.
    for c in manifest['cases']:
        if sha256(Path(c['case']) / REPORT) != c['report_sha256']:
            raise RuntimeError('Report changed since inventory')
        for f in c['files']:
            if signature(Path(f['path'])) != f['signature']:
                raise RuntimeError('Artifact changed since inventory')
    if active_simulators():
        raise RuntimeError('Simulator started during inventory; cleanup stopped')
    deleted = []
    with (HERE / 'deleted.jsonl').open('x') as journal:
        for c in manifest['cases']:
            for f in c['files']:
                Path(f['path']).unlink()
                deleted.append(f)
                journal.write(json.dumps(f) + '\n')
                journal.flush()
            note = {'status': 'CLEANED_INVALID_TENSORS', 'original_run_status': c['status'],
                    'date': '2026-10-27', 'logs_and_source_snapshots_retained': True,
                    'manifest': str(manifest_path), 'numerical_revalidation_available': False}
            (Path(c['case']) / 'invalid_artifact_cleanup.json').write_text(json.dumps(note, indent=2) + '\n')
    after = protected_state()
    (HERE / 'protected_after.json').write_text(json.dumps(after, indent=2) + '\n')
    if before != after:
        raise RuntimeError('Protected state changed; inspect audit')
    result = {'status': 'PASS', 'deleted_files': len(deleted), 'cleaned_cases': len(manifest['cases']),
              'reclaimed_allocated_bytes': sum(f['allocated_bytes'] for f in deleted),
              'source_git_libraries_and_pass_reports_unchanged': True}
    (HERE / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
