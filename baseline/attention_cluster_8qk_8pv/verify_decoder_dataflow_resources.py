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


def residual_audit(case, expected_jobs, hidden, groups=1):
    flows=[]
    with next((case/'logs').glob('attention-sst*.log')).open() as stream:
        for line in stream:
            if '[DECODER_RESIDUAL_FLOW]' in line:
                flows.append({k:int(v) for k,v in FIELDS.findall(line)})
    keys={(f['core'],f['epoch'],f['stage']) for f in flows}
    if len(flows)!=len(expected_jobs) or keys!=expected_jobs:
        raise ValueError('missing/duplicate residual epilogue lifecycle')
    chunks=0
    for n in range(0,hidden//64,16):
        rows_per_chunk=8192//(min(16,hidden//64-n)*128)
        chunks+=(64+rows_per_chunk-1)//rows_per_chunk
    for f in flows:
        if f['residual_hbm_bytes']!=groups*64*hidden*2 or f['scratch_bytes']!=8192 or f['chunks']!=groups*chunks:
            raise ValueError('residual epilogue traffic or scratch mismatch')
    return {'status':'PASS','verified_jobs':len(flows),'scratch_bytes':8192,
            'gemm_overlap_cycles_sum':sum(f['gemm_overlap_cycles'] for f in flows)}


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
    manifest=json.loads((case/'decoder_stack_inputs.json').read_text()) if (case/'decoder_stack_inputs.json').exists() else {}
    if manifest.get('ffn_panel_fusion'):
        hidden=manifest['hidden'];width=4*hidden;groups=manifest.get('sequence',1024)//1024
        fused_down=bool(manifest.get('ffn_down_fusion'))
        expected_read=(3 if fused_down else 2)*(width//64)*(hidden//64)*8192+64*hidden*2
        residual=bool(manifest.get('down_residual_fusion'))
        if residual:expected_read+=64*hidden*2
        expected_read*=groups
        expected_write=groups*64*(hidden if fused_down else width)*2+(8 if residual else 0)
        paired=[j for js in jobs.values() for j in js if j['stage']==4]
        if len(paired)!=16*manifest.get('layers',2) or any(j['op']!=(9 if fused_down else 8) or j['hbm_read']!=expected_read or j['hbm_write']!=expected_write for j in paired):
            raise ValueError('fused FFN must read one Norm2 row group plus all dense weights and write only '+('final residual' if residual else 'down' if fused_down else 'gated'))
    residual_flow=None
    if manifest.get('down_residual_fusion'):
        if any(j['stage']==9 for js in jobs.values() for j in js):
            raise ValueError('standalone residual must be elided')
        residual_flow=residual_audit(case,{(j['core'],j['epoch'],j['stage']) for js in jobs.values() for j in js if j['stage']==4},manifest['hidden'],manifest.get('sequence',1024)//1024)
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
            'residual_flow':residual_flow,
            'ffn_panel_only_gated_DMA':bool(manifest.get('ffn_panel_fusion') and not manifest.get('ffn_down_fusion')),
            'ffn_panel_only_down_DMA':bool(manifest.get('ffn_down_fusion') and not manifest.get('down_residual_fusion')),
            'down_residual_only_final_DMA':bool(manifest.get('down_residual_fusion')),
            'scope': 'all decoder DMA bytes agree with job counters; no outstanding GM interval crosses job completion',
            'note': 'DMA transport duration is RTT; GM local duration is a logical request service span (banked spans can overlap); physical beats require bank stats. They overlap and must not be summed.'}


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
