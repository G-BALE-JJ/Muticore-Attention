#!/usr/bin/env python3
"""Add Projection-window resource waits and per-worker phases to startup results."""
import argparse
import json
from pathlib import Path
import re

from analyze_batch_b_waits import analyze
from run_projection_overlap import HERE, ROOT, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    for row in summary['cases']:
        case = Path(row['artifact_root'])
        cache = case / 'projection_resource_waits.json'
        if cache.exists() and not args.refresh:
            waits = json.loads(cache.read_text())
        else:
            waits = analyze(case)
            cache.write_text(json.dumps(waits, indent=2) + '\n')
        if waits['status'] != 'PASS' or set(waits['by_worker']) != {str(c) for c in range(4, 20)}:
            raise ValueError('incomplete Projection-window analysis')
        phases = row.get('projection_phase_cycles_by_worker')
        if phases is None or args.refresh:
            phases = {}
            log = next((case / 'logs').glob('attention-sst*.log'))
            with log.open(errors='replace') as stream:
                for line in stream:
                    if '[PROJECTION_CYCLE_BREAKDOWN]' in line:
                        fields = {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', line)}
                        core = fields.pop('core')
                        if str(core) in phases:
                            raise ValueError('duplicate Projection phase breakdown')
                        phases[str(core)] = fields
        if set(phases) != {str(c) for c in range(4, 20)}:
            raise ValueError('missing per-worker Projection phase breakdown')
        row['projection_phase_cycles_by_worker'] = phases
        row['projection_resource_waits'] = waits
    files = [Path(__file__).resolve(), HERE / 'analyze_batch_b_waits.py']
    summary['extra_resource_analysis_sha256'] = {str(p.relative_to(ROOT)): digest(p) for p in files}
    args.summary.write_text(json.dumps(summary, indent=2) + '\n')
    print('PASS all six cases: per-worker phases and Projection-window resource waits')


if __name__ == '__main__':main()
