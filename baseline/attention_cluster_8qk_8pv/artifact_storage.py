#!/usr/bin/env python3
"""Host artifact retention; never changes simulated capacity or timing."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def preflight(path, min_free_gib=None):
    """Check the destination filesystem before creating inputs or directories."""
    minimum = float(os.environ.get('GOLEM_ARTIFACT_MIN_FREE_GIB', '24')
                    if min_free_gib is None else min_free_gib)
    if not math.isfinite(minimum) or minimum < 0:
        raise ValueError('GOLEM_ARTIFACT_MIN_FREE_GIB must be finite and nonnegative')
    if os.environ.get('GOLEM_ARTIFACT_POLICY', 'compact') not in ('compact', 'full'):
        raise ValueError('GOLEM_ARTIFACT_POLICY must be compact or full')
    parent = Path(path).resolve()
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    if free < minimum * 2**30:
        raise OSError(f'artifact filesystem {parent}: free {free / 2**30:.2f} GiB '
                      f'< required reserve {minimum:g} GiB; remove archived artifacts first')
    return {'filesystem_path': str(parent), 'free_bytes': free, 'minimum_free_gib': minimum}


def signature(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns


def compact_logs(case, log=None, rank_root=None):
    log = Path(log) if log is not None else case / 'logs' / 'attention-sst.log'
    # Refuse to remove files outside this experiment, including symlink escapes.
    log.resolve().relative_to(case)
    rank_root = Path(rank_root) if rank_root else case / 'mpi_rank_output'
    rank_root.resolve().relative_to(case)
    streams = sorted(rank_root.rglob('stdout'))
    console = log.with_suffix('.mpi-console-raw')
    raw = log.with_suffix('.raw')
    if streams:
        ranks = [int(m[1]) for p in streams
                 if (m := re.fullmatch(r'rank\.(\d+)', p.parent.name))]
        if sorted(ranks) != list(range(len(streams))) or any(p.stat().st_size == 0 for p in streams):
            return {'status': 'SKIP', 'reason': 'missing, duplicate or empty MPI streams'}
        sources = [p for stream in streams for p in (stream, stream.with_name('stderr')) if p.exists()]
        targets = [log, log.with_suffix('.resources.jsonl'), log.with_suffix('.causal.jsonl')]
        deletion = sources + ([console] if console.exists() else [])
    elif raw.exists():
        sources, targets, deletion = [raw], [log, log.with_suffix('.resources.jsonl')], [raw]
    else:
        return {'status': 'PASS', 'deleted_files': 0, 'reclaimed_bytes': 0}
    if any(not p.is_file() or p.is_symlink() for p in sources + targets + deletion):
        return {'status': 'SKIP', 'reason': 'missing canonical log or unexpected symlink'}
    snapshots = {p: signature(p) for p in set(sources + targets + deletion)}
    hashes = [hashlib.sha256() for _ in targets]
    # Reconstruct exactly the normalized text emitted by merge_mpi_rank_logs.
    # The single-rank splitter only separates resource records.
    for source in sources:
        with source.open(errors='replace') as stream:
            for line in stream:
                index = 1 if '[RESOURCE_INTERVAL]' in line else \
                    2 if len(targets) == 3 and '[PROJECTION_CAUSAL]' in line else 0
                hashes[index].update((line.rstrip('\n') + '\n').encode())
    if any(h.hexdigest() != digest(p) for h, p in zip(hashes, targets)):
        return {'status': 'SKIP', 'reason': 'canonical logs differ; all raw evidence retained'}
    if any(signature(p) != s for p, s in snapshots.items()):
        return {'status': 'SKIP', 'reason': 'logs changed during verification'}
    released = sum(p.stat().st_blocks * 512 for p in deletion)
    for p in deletion:
        p.unlink()
    return {'status': 'PASS', 'deleted_files': len(deletion), 'reclaimed_bytes': released,
            'canonical_sha256': {str(p.relative_to(case)): h.hexdigest() for p, h in zip(targets, hashes)}}


def tensor_paths(case):
    # Deliberately do not recurse: shared inputs and source snapshots are untouched.
    return sorted(set((case / 'hbm').glob('*.bin')) | set(case.glob('*.bin')) | set(case.glob('*.npy')))


def write_report(path, report):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)


def finalize(case, policy=None, discard_tensors=False, validated=False, log=None, rank_root=None):
    """Call only after the simulation and all validators have returned successfully.

    compact retains canonical logs and tensors; full retains raw duplicates too.
    discard_tensors is an explicit archival operation, never an environment default.
    """
    case = Path(case).resolve()
    policy = policy or os.environ.get('GOLEM_ARTIFACT_POLICY', 'compact')
    if policy not in ('compact', 'full'):
        raise ValueError('policy must be compact or full')
    if policy == 'full' and discard_tensors:
        raise ValueError('full policy cannot discard tensors')
    results = [case / 'result.json', case / 'decoder_stack_result.json']
    if not validated and not any(p.is_file() and json.loads(p.read_text()).get('status') == 'PASS' for p in results):
        raise ValueError('artifact cleanup requires a PASS result after validation')
    report_path = case / 'artifact_cleanup.json'
    previous = json.loads(report_path.read_text()) if report_path.exists() else {}
    report = {'status': 'PASS', 'policy': policy, 'finished_unix_seconds': time.time(),
              'simulated_hardware_changed': False, 'discarded_tensors': previous.get('discarded_tensors', [])}
    if policy == 'full':
        report['logs'] = {'status': 'SKIP', 'reason': 'full retention requested'}
    else:
        report['logs'] = compact_logs(case, log, rank_root)
        if report['logs'].get('deleted_files') == 0 and previous.get('logs', {}).get('deleted_files', 0):
            report['logs'] = dict(previous['logs'], already_cleaned=True)
        report['sparse_files'] = []
        for path in tensor_paths(case):
            if path.is_symlink() or not path.is_file():
                continue
            before = path.stat().st_blocks * 512
            # Linux FALLOC_FL_PUNCH_HOLE on zero ranges preserves logical bytes.
            operation = subprocess.run(['fallocate', '--dig-holes', str(path)], capture_output=True, text=True)
            report['sparse_files'].append({'path': str(path.relative_to(case)),
                'status': 'PASS' if operation.returncode == 0 else 'SKIP',
                'reclaimed_bytes': max(0, before - path.stat().st_blocks * 512),
                'reason': operation.stderr.strip()})
    if discard_tensors:
        entries = []
        for path in tensor_paths(case):
            if path.suffix != '.bin' or path.is_symlink() or not path.is_file():
                continue
            entries.append({'path': str(path.relative_to(case)), 'bytes': path.stat().st_size,
                            'sha256': digest(path), 'state': 'pending_delete'})
        report['discarded_tensors'].extend(entries)
        # Persist hashes before deleting, including when interrupted midway.
        write_report(report_path, report)
        for entry in entries:
            (case / entry['path']).unlink()
            entry['state'] = 'deleted'
    write_report(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('preflight')
    check.add_argument('case', type=Path)
    check.add_argument('--min-free-gib', type=float)
    cleanup = commands.add_parser('finalize')
    cleanup.add_argument('case', type=Path)
    cleanup.add_argument('--policy', choices=('compact', 'full'))
    cleanup.add_argument('--discard-tensors', action='store_true', help='archive only: delete own .bin files, retaining hashes and canonical logs')
    cleanup.add_argument('--validated', action='store_true', help='runner has just completed ALL validators successfully')
    cleanup.add_argument('--log', type=Path, help='canonical log inside this case')
    cleanup.add_argument('--rank-root', type=Path, help='MPI rank streams inside this case')
    args = parser.parse_args()
    if args.command == 'preflight':
        print(json.dumps(preflight(args.case, args.min_free_gib)))
    else:
        report = finalize(args.case, args.policy, args.discard_tensors, args.validated, args.log, args.rank_root)
        print(json.dumps({'status': report['status'], 'policy': report['policy'], 'logs': report['logs'],
                          'report': str(args.case / 'artifact_cleanup.json')}))


if __name__ == '__main__':
    main()
