#!/usr/bin/env python3
"""Retire reproducible historical tensors, retaining accepted references and evidence."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
from cleanup_invalid import LLM, HERE, active_simulators, protected_state, signature


def protected_roots():
    latest = (LLM / 'Muticore-Attention-all-gemm/results').resolve()
    old = (LLM / 'test/history/legacy-primary/results').resolve()
    protected = [old / 'prefill_generalization_20261016/formal_l2_s1024']
    for selection in (LLM / 'test/attention_cluster_8qk_8pv').glob('SELECTED_PREFILL_*.json'):
        data = json.loads(selection.read_text())
        for key in ('baseline_result', 'verification_reference'):
            # Protect the full accepted case, including its parent experiment inputs.
            case = (LLM / data[key]).resolve().parent
            protected.append(case.parent if case.name == 'reuse_window_probe' else case)
    for root in (old, latest):
        for path in root.iterdir():
            if (any(word in path.name for word in ('checkpoint_import', 'gpu_prefill_import', 'gpu_prefill_handoff'))
                    or path.name.startswith('fp16_e2e_') or path.name == 'llama32_8_d64'):
                protected.append(path)
    return (old, latest), sorted(set(protected))


def inventory():
    roots, protected = protected_roots()
    tensors, logs, linked = [], [], []
    for root in roots:
        for base, directories, names in os.walk(root):
            directory = Path(base)
            if any(directory == p or p in directory.parents for p in protected):
                directories[:] = []
                continue
            if any('source' in part or part in ('snapshot', 'library') for part in directory.relative_to(root).parts):
                directories[:] = []
                continue
            for name in sorted(names):
                path = directory / name
                if path.is_symlink() or not path.is_file():
                    continue
                stat = path.stat()
                entry = dict(path=str(path), allocated_bytes=stat.st_blocks * 512,
                             logical_bytes=stat.st_size, signature=signature(path))
                if path.suffix in ('.bin', '.npy'):
                    if stat.st_nlink != 1:
                        linked.append(entry)
                        continue
                    tensors.append(entry)
                elif stat.st_size >= 16 * 1024**2 and (path.suffix in ('.log', '.jsonl', '.mpi-console-raw')
                                                        or path.name in ('stdout', 'stderr')):
                    if path.with_name(path.name + '.gz').exists() or path.with_name(path.name + '.zst').exists():
                        raise RuntimeError(f'Existing archive: {path}')
                    logs.append(entry)
    return dict(protected_roots=[str(p) for p in protected], tensors=tensors, logs=logs, retained_linked_tensors=linked)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compress-logs', action='store_true')
    args = parser.parse_args()
    if active_simulators():
        raise RuntimeError('Active simulator; stop archival')
    folder = HERE / 'retired_history'
    if not args.compress_logs:
        folder.mkdir(exist_ok=False)
        before = protected_state()
        (folder / 'protected_before.json').write_text(json.dumps(before, indent=2) + '\n')
        manifest = inventory()
        (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        # Validate the complete deletion set before changing the first file.
        for entry in manifest['tensors']:
            if signature(Path(entry['path'])) != entry['signature']:
                raise RuntimeError('File changed during inventory')
        if active_simulators():
            raise RuntimeError('Simulator started during inventory')
        reclaimed = 0
        with (folder / 'deleted.jsonl').open('x') as journal:
            for index, entry in enumerate(manifest['tensors'], 1):
                path = Path(entry['path'])
                if signature(path) != entry['signature']:
                    raise RuntimeError(f'File changed: {path}')
                path.unlink()
                reclaimed += entry['allocated_bytes']
                journal.write(json.dumps(entry) + '\n')
                if index % 500 == 0:
                    journal.flush()
                    print(f'Deleted {index} retired tensors; reclaimed {reclaimed / 2**30:.2f} GiB', flush=True)
        after = protected_state()
        (folder / 'protected_after.json').write_text(json.dumps(after, indent=2) + '\n')
        if before != after:
            raise RuntimeError('Protected source/metadata state changed')
        result = dict(status='PASS', tensor_files=len(manifest['tensors']), reclaimed_allocated_bytes=reclaimed,
                      protected_reference_roots=len(manifest['protected_roots']),
                      source_git_libraries_and_pass_reports_unchanged=True,
                      retired_tensor_revalidation_requires_regeneration=True)
        (folder / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        return
    manifest = json.loads((folder / 'manifest.json').read_text())
    before = protected_state()
    journal_path = folder / 'compressed_logs.jsonl'
    completed = {row['path'] for line in journal_path.read_text().splitlines()
                 if (row := json.loads(line))} if journal_path.exists() else set()
    reclaimed = 0
    with journal_path.open('a') as journal:
        for index, entry in enumerate(manifest['logs'], 1):
            if entry['path'] in completed:
                continue
            path = Path(entry['path'])
            if signature(path) != entry['signature']:
                raise RuntimeError(f'File changed: {path}')
            archive = path.with_name(path.name + '.zst')
            temporary = path.with_name(path.name + '.zst.part')
            if archive.exists() or temporary.exists():
                raise RuntimeError(f'Inspect incomplete archive before resuming: {path}')
            # Zstandard uses four CPU threads and retains a frame checksum.
            subprocess.run(['zstd', '-q', '-T4', '-1', '--check', str(path), '-o', str(temporary)], check=True)
            # Compare SHA256 of the complete logical byte stream before unlinking.
            h, check = hashlib.sha256(), hashlib.sha256()
            with path.open('rb') as source:
                for block in iter(lambda: source.read(8 * 1024**2), b''):
                    h.update(block)
            with subprocess.Popen(['zstd', '-q', '-d', '-c', str(temporary)], stdout=subprocess.PIPE) as decoder:
                for block in iter(lambda: decoder.stdout.read(8 * 1024**2), b''):
                    check.update(block)
                if decoder.wait() != 0:
                    raise RuntimeError(f'Decompression failed: {temporary}')
            if h.digest() != check.digest() or signature(path) != entry['signature']:
                raise RuntimeError(f'Compression mismatch: {path}')
            temporary.rename(archive)
            path.unlink()
            saved = entry['allocated_bytes'] - archive.stat().st_blocks * 512
            reclaimed += saved
            journal.write(json.dumps(dict(path=str(path), archive=str(archive), sha256=h.hexdigest(),
                                          reclaimed_bytes=saved)) + '\n')
            journal.flush()
            print(f'Archived {index}/{len(manifest["logs"])}; reclaimed this run {reclaimed / 2**30:.2f} GiB', flush=True)
    after = protected_state()
    if before != after:
        raise RuntimeError('Protected source/metadata state changed')
    rows = [json.loads(line) for line in journal_path.read_text().splitlines()]
    result = dict(status='PASS', compressed_logs=len(rows), reclaimed_allocated_bytes=sum(r['reclaimed_bytes'] for r in rows),
                  sha256_verified_before_deletion=True, source_git_libraries_and_pass_reports_unchanged=True)
    (folder / 'log_result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
