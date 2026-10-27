#!/usr/bin/env python3
"""Archive interrupted pre-validation trials and compress their logs losslessly."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from cleanup_invalid import LLM, HERE, active_simulators, is_validated, protected_state, signature


def main():
    if active_simulators():
        raise RuntimeError('Active simulator')
    destination = HERE / 'round2'
    destination.mkdir(exist_ok=False)
    before = protected_state()
    (destination / 'protected_before.json').write_text(json.dumps(before, indent=2) + '\n')
    latest = LLM / 'Muticore-Attention-all-gemm/results'
    old = LLM / 'Muticore-Attention/results'
    candidates = [latest / ('all_gemm_reuse_window_20261019_' + name)
                  for name in ('linear', 'linear2', 'linear3', 'linear4', 'linear5', 'internal_ports', 'trace', 'trace2')]
    candidates += [old / 'all_gemm_reuse_window_20261019',
                   old / 'lm_head_20261014/two_layer_formal',
                   old / 'sweeps_trace2/hq4_hkv2_q512_k512_d128',
                   old / 'fine_resource_timeline_20261012/formal_v2']
    # These are stale stopped trials with no completed numerical result, not
    # cases that merely failed a parent performance comparison after validation.
    cases = []
    for case in candidates:
        if not case.is_dir():
            continue
        passed = [p for p in case.rglob('*.json') if p.stat().st_size < 4 * 1024**2 and is_validated(p)]
        if passed:
            raise RuntimeError(f'Unexpected successful report in {case}: {passed}')
        files = [p for p in case.rglob('*') if p.is_file() and not p.is_symlink()]
        if any(p.stat().st_mtime >= 1793030400 for p in files):
            raise RuntimeError(f'Recent files in {case}')
        cases.append(case)
    previous = json.loads((HERE / 'manifest.json').read_text())
    logs_cases = cases + [Path(c['case']) for c in previous['cases']]
    deletions, compressions = [], []
    for case in cases:
        for p in sorted(case.rglob('*')):
            if p.is_symlink() or not p.is_file() or p.suffix not in ('.bin', '.npy'):
                continue
            p.resolve().relative_to(case.resolve())
            s = p.stat()
            if s.st_nlink != 1:
                raise RuntimeError(f'Linked tensor {p}')
            deletions.append(dict(path=str(p), allocated_bytes=s.st_blocks * 512, logical_bytes=s.st_size,
                                  signature=signature(p)))
    for case in logs_cases:
        for p in sorted(case.rglob('*')):
            if p.is_symlink() or not p.is_file() or p.stat().st_size < 16 * 1024**2:
                continue
            if p.suffix not in ('.log', '.jsonl', '.mpi-console-raw') and p.name not in ('stdout', 'stderr'):
                continue
            if p.with_name(p.name + '.gz').exists():
                raise RuntimeError(f'Archive already exists for {p}')
            compressions.append(dict(path=str(p), allocated_bytes=p.stat().st_blocks * 512,
                                     logical_bytes=p.stat().st_size, signature=signature(p)))
    manifest = dict(cases=[str(p) for p in cases], deletions=deletions, compressions=compressions)
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(tensor_files=len(deletions), tensor_gib=sum(f['allocated_bytes'] for f in deletions)/2**30,
                          logs=len(compressions), log_gib=sum(f['allocated_bytes'] for f in compressions)/2**30)), flush=True)
    # All candidates and retained evidence were inspected before changing files.
    if active_simulators():
        raise RuntimeError('Simulator started during inventory')
    reclaimed = 0
    with (destination / 'journal.jsonl').open('x') as journal:
        for entry in deletions:
            p = Path(entry['path'])
            if signature(p) != entry['signature']:
                raise RuntimeError(f'File changed: {p}')
            p.unlink()
            reclaimed += entry['allocated_bytes']
            journal.write(json.dumps(dict(action='delete_invalid_tensor', **entry)) + '\n')
            journal.flush()
        for entry in compressions:
            p = Path(entry['path'])
            archive = p.with_name(p.name + '.gz')
            temporary = p.with_name(p.name + '.gz.part')
            if signature(p) != entry['signature']:
                raise RuntimeError(f'File changed: {p}')
            h = hashlib.sha256()
            with p.open('rb') as source, temporary.open('xb') as target:
                with gzip.GzipFile(filename='', mode='wb', compresslevel=1, fileobj=target, mtime=0) as compressed:
                    for block in iter(lambda: source.read(8 * 1024**2), b''):
                        h.update(block)
                        compressed.write(block)
            # Verify decompression reproduces every byte before removing the log.
            check = hashlib.sha256()
            with gzip.open(temporary, 'rb') as source:
                for block in iter(lambda: source.read(8 * 1024**2), b''):
                    check.update(block)
            if h.digest() != check.digest() or signature(p) != entry['signature']:
                raise RuntimeError(f'Archive verification failed: {p}')
            temporary.rename(archive)
            p.unlink()
            saved = entry['allocated_bytes'] - archive.stat().st_blocks * 512
            reclaimed += saved
            journal.write(json.dumps(dict(action='lossless_gzip', archive=str(archive), sha256=h.hexdigest(),
                                          reclaimed_bytes=saved, **entry)) + '\n')
            journal.flush()
            print(f'Archived {p.name}: reclaimed {saved / 2**30:.3f} GiB', flush=True)
    after = protected_state()
    (destination / 'protected_after.json').write_text(json.dumps(after, indent=2) + '\n')
    if before != after:
        raise RuntimeError('Protected state changed')
    result = dict(status='PASS', cleaned_cases=len(cases), tensor_files=len(deletions), compressed_logs=len(compressions),
                  reclaimed_allocated_bytes=reclaimed, source_git_libraries_and_pass_reports_unchanged=True)
    (destination / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
