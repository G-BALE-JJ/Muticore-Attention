#!/usr/bin/env python3
"""Stream GM evidence and reconcile each decoder job, including prefetch/drain."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path
import re

FIELDS = re.compile(r'(\w+)=(-?\d+)')
INTERVAL = re.compile(r'resource=(\w+) core=(\d+).*?start_tick=(\d+) done_tick=(\d+).*?bytes=(\d+)')


def verify(case):
    log = next((case / 'logs').glob('attention-sst*.log'))
    jobs = defaultdict(list)
    with log.open() as stream:
        for line in stream:
            if '[DECODER_OP] event=done' in line:
                j = {k: int(v) for k, v in FIELDS.findall(line)}
                if j['status']:
                    raise ValueError('failed decoder job')
                jobs[j['core']].append(j)
    for js in jobs.values():
        js.sort(key=lambda j: j['start_tick'])
    starts = {core: [j['start_tick'] for j in js] for core, js in jobs.items()}
    totals = defaultdict(lambda: defaultdict(lambda: {'bytes': 0, 'requests': 0, 'service_or_RTT_cycles': 0.}))
    with log.with_suffix('.resources.jsonl').open() as stream:
        for line in stream:
            m = INTERVAL.search(line)
            if not m:
                continue
            resource, core, start, end, size = m.groups()
            core, start, end, size = map(int, (core, start, end, size))
            if core not in starts or resource not in ('dma_read_transport', 'dma_write_transport', 'local_read', 'local_write'):
                continue
            index = bisect_right(starts[core], start) - 1
            if index < 0:
                continue
            j = jobs[core][index]
            if start >= j['tick']:
                continue  # Projection/Attention or guest work between decoder jobs.
            if end > j['tick']:
                raise ValueError(f'outstanding {resource} crosses job completion: {j}')
            key = (core, j['epoch'], j['stage'])
            v = totals[key][resource]
            v['bytes'] += size
            v['requests'] += 1
            v['service_or_RTT_cycles'] += (end - start) / 1000
    by_stage = defaultdict(lambda: defaultdict(lambda: {'bytes': 0, 'requests': 0}))
    count = 0
    for core, js in jobs.items():
        for j in js:
            count += 1
            rs = totals[(core, j['epoch'], j['stage'])]
            for resource, field in (('dma_read_transport', 'hbm_read'), ('dma_write_transport', 'hbm_write')):
                if rs[resource]['bytes'] != j[field]:
                    raise ValueError(f'{resource} byte mismatch: {j}, {rs[resource]}')
            for resource, v in rs.items():
                combined = by_stage[f"{j['epoch']}.{j['stage']}"][resource]
                combined['bytes'] += v['bytes']; combined['requests'] += v['requests']
    return {'status': 'PASS', 'verified_decoder_jobs': count, 'by_epoch_stage': by_stage,
            'scope': 'all decoder DMA bytes agree with job counters; no outstanding GM interval crosses job completion',
            'note': 'DMA transport duration is RTT; GM local duration is port service. They overlap and must not be summed.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = verify(args.case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['status'], result['verified_decoder_jobs'])


if __name__ == '__main__':
    main()
