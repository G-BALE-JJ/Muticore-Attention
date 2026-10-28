#!/usr/bin/env python3
"""Check that timing runs actually disabled MACs on all sixteen workers."""
import argparse
import json
from pathlib import Path
import re


def verify_timing_backend(log):
    observed = {}
    pattern = re.compile(r'\[ARRAY_NUMERICAL_DATA\] core=(\d+) functional=(\d+) timing_fast_path=(\d+) materialize=(\d+)')
    with Path(log).open() as stream:
        for line in stream:
            match = pattern.search(line)
            if match:
                core, functional, fast, materialize = map(int, match.groups())
                mode = (functional, fast, materialize)
                if core in observed and observed[core] != mode:
                    raise ValueError('conflicting array numerical mode evidence')
                observed[core] = mode
    if set(observed) != set(range(4, 20)) or any(v[0] for v in observed.values()):
        raise ValueError('full-timing requires functional=0 evidence on all sixteen worker arrays')
    if any(materialize != int(not fast) for _, fast, materialize in observed.values()):
        raise ValueError('array materialization mode differs from timing fast path')
    return {'status': 'PASS', 'sim_mode': 'full-timing', 'observed_workers': len(observed),
            'array_numerical_macs': False,
            'numerics': {'status': 'SKIPPED', 'reason': 'full-timing skips array numerical MACs'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify_timing_backend(args.log)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
