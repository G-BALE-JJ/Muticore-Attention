#!/usr/bin/env python3
"""Run both causal dataflow schedules with the existing best projection plan."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
BEST = {
    'GOLEM_PROJECTION_CBUFFER_PARTIAL': '1',
    'GOLEM_PROJECTION_WRAP_PREFETCH': '1',
    'GOLEM_PROJECTION_PARTIAL_RESTORE_AHEAD': '1',
    'GOLEM_FUSION_NORM_PROJECTION': '1',
    'GOLEM_FUSION_PROJECTION_ROPE': '0',
    'GOLEM_FUSION_SKIP_AUDIT_OUTPUT': '1',
    'GOLEM_PROJECTION_PACKED_NORM_INPUT': '1',
    'GOLEM_PROJECTION_INPUT_PREFETCH_BUDGET': '0',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, required=True)
    args = parser.parse_args()
    for name, enabled in [('control', '0'), ('balanced', '1')]:
        env = {**os.environ, **BEST, 'GOLEM_ATTENTION_CAUSAL_BALANCE': enabled}
        subprocess.run([sys.executable, str(HERE / 'run_llama_projection.py'),
                        '--artifact-root', str(args.artifact_root / name),
                        '--reference-root', str(args.reference_root)], env=env, check=True)
    subprocess.run([sys.executable, str(HERE / 'analyze_causal_balance.py'),
                    '--artifact-root', str(args.artifact_root),
                    '--output', str(args.artifact_root / 'causal_balance_results.json')], check=True)


if __name__ == '__main__':
    main()
