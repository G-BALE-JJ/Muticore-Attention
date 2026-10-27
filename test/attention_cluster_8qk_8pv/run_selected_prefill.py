#!/usr/bin/env python3
"""Reproduce the accepted cycle configuration from a completed campaign."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def verify_digest(path, expected, label):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError('Selected '+label+' changed: expected '+expected+', got '+actual+
                         '. Revalidate cycles with an explicit experimental configuration before replacing the selection.')


def verify_guest_digest(path, expected):
    verify_digest(path, expected, 'guest')


def verify_library_digest(path, expected):
    verify_digest(path, expected, 'library')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root', type=Path, required=True)
    p.add_argument('--configuration', type=Path, default=HERE/'SELECTED_PREFILL_2026-10-26.json')
    p.add_argument('--verification-reference', type=Path)
    p.add_argument('--baseline-result', type=Path)
    p.add_argument('--cpu-set', default='8-15')
    p.add_argument('--schedule-only', action='store_true')
    a = p.parse_args()
    config = json.loads(a.configuration.read_text())
    if config['status'] != 'PASS' or config['latency_unit'] != 'cycles':
        p.error('configuration must come from a completed cycle campaign')
    command = [sys.executable, str(HERE/'run_all_gemm_reuse_window_baseline.py'),
               '--artifact-root', str(a.artifact_root), '--cpu-set', a.cpu_set,
               '--baseline-result', str(a.baseline_result or ROOT/config['baseline_result']),
               '--verification-reference', str(a.verification_reference or ROOT/config['verification_reference']),
               *config['flags']]
    if not a.schedule_only:
        command.append('--run')
        if config.get('measured_library_sha256'):
            verify_library_digest(ROOT/'install/lib/sst-elements-library/libgolem.so',config['measured_library_sha256'])
        if config.get('measured_guest_sha256'):
            guest = ROOT/'test/golem/small/muticore_attention'
            target = config.get('measured_guest_target','decoder-embedding')
            subprocess.run(['make','-C',str(guest),target],cwd=ROOT,check=True)
            verify_guest_digest(guest/'riscv64'/target/'fused_attention',config['measured_guest_sha256'])
    subprocess.run(command, cwd=ROOT,
                   env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'), check=True)


if __name__ == '__main__':
    main()
