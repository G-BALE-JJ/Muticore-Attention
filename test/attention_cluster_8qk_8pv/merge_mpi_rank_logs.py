#!/usr/bin/env python3
"""Use Open MPI's independent rank streams without console interleaving."""
import argparse
from pathlib import Path
import re


def merge(root, output, ranks):
    streams = sorted(Path(root).rglob('stdout'))
    rank_ids = [int(m[1]) for p in streams
                if (m := re.fullmatch(r'rank\.(\d+)', p.parent.name))]
    if sorted(rank_ids) != list(range(ranks)) or len(streams) != ranks:
        raise ValueError('missing or duplicate MPI rank stdout streams')
    if any(p.stat().st_size == 0 for p in streams):
        raise ValueError('empty MPI rank stdout stream')
    output = Path(output)
    output.rename(output.with_suffix('.mpi-console-raw'))
    # Fine resource records can be several GiB. Preserve the raw rank streams
    # and stream these records to a sidecar so existing report parsers do not
    # have to hold them in memory. Timebase/control events stay in the main log.
    with output.open('w') as main, output.with_suffix('.resources.jsonl').open('w') as resources, \
            output.with_suffix('.causal.jsonl').open('w') as causal:
        for path in streams:
            for source in (path, path.with_name('stderr')):
                if not source.exists():
                    continue
                with source.open(errors='replace') as stream:
                    for line in stream:
                        target = resources if '[RESOURCE_INTERVAL]' in line else \
                            causal if '[PROJECTION_CAUSAL]' in line else main
                        target.write(line.rstrip('\n') + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ranks', type=int, required=True)
    args = parser.parse_args()
    merge(args.root, args.output, args.ranks)


if __name__ == '__main__':
    main()
