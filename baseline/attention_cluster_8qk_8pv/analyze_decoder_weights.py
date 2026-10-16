#!/usr/bin/env python3
"""Audit measured weight lifetimes and replay bounded cross-worker reuse.

Replay estimates byte reuse opportunities only. It does not model a multicast
implementation, its NoC/GM cost, or accelerator cycle savings.
"""
import argparse
from collections import OrderedDict, defaultdict
import json
from pathlib import Path
import re

FIELDS=re.compile(r'(\w+)=(\d+)')


def percentile(values, p):
    values=sorted(values)
    return values[min(len(values)-1,int((len(values)-1)*p))] if values else None


def analyze(case):
    log=next((case/'logs').glob('attention-sst*.log'))
    flows=[];tiles=[]
    with log.open() as stream:
        for line in stream:
            if '[DECODER_PREFETCH_FLOW]' in line:flows.append({k:int(v) for k,v in FIELDS.findall(line)})
            if '[DECODER_WEIGHT_TILE]' in line:tiles.append({k:int(v) for k,v in FIELDS.findall(line)})
    if not flows or not tiles:raise ValueError('missing prefetch flows or tile trace')
    flow_keys={(f['core'],f['epoch'],f['stage']) for f in flows}
    tiles=[e for e in tiles if (e['core'],e['epoch'],e['stage']) in flow_keys]
    if not tiles:raise ValueError('missing FFN weight tile trace')
    by_job=defaultdict(list)
    for e in tiles:
        if not e['submit']<=e['dma_done']<=e['gm_done']<=e['consume']:
            raise ValueError('weight consumed before DMA/GM completion')
        by_job[e['core'],e['epoch'],e['stage']].append(e)
    if len(flows)!=len(by_job):raise ValueError('missing or duplicate weight job trace')
    for f in flows:
        es=by_job[f['core'],f['epoch'],f['stage']]
        if f['enabled']!=1 or f['slots'] not in (2,4,8) or f['requests']!=len(es) or f['hits']!=len(es):
            raise ValueError('prefetch request/hit/trace mismatch')
        if f['wait_total']!=f['wait_dma']+f['wait_gm']:
            raise ValueError('weight ready wait partition mismatch')
        if f['dma_sum']!=sum(e['dma_done']-e['submit'] for e in es):
            raise ValueError('DMA latency sum differs from tile trace')
        if f['gm_sum']!=sum(e['gm_done']-e['dma_done'] for e in es):
            raise ValueError('GM latency sum differs from tile trace')
        # Down k is relative to its gated panel; the HBM address includes
        # that panel's global K offset and identifies the actual weight tile.
        if len({(e['row'],e['path'],e['address']) for e in es})!=len(es):
            raise ValueError('duplicate weight tile consumption')
    groups=defaultdict(list)
    for e in tiles:
        # Four row owners of the same input HBM node: 4+m,8+m,12+m,16+m.
        groups[(e['epoch'],e['stage'],e['core']%4,e['path'],e['address'])].append(e)
    skews=[max(e['submit'] for e in es)-min(e['submit'] for e in es) for es in groups.values() if len(es)>1]
    replay={}
    for capacity in (2,4,8,16,32):
        caches=defaultdict(OrderedDict);misses=hits=inflight_hits=resident_hits=0
        for e in sorted(tiles,key=lambda e:(e['submit'],e['core'])):
            cache=caches[e['core']%4];key=(e['epoch'],e['stage'],e['path'],e['address'])
            if key in cache:
                hits+=1
                if e['submit']<cache[key]:inflight_hits+=1
                else:resident_hits+=1
                cache.move_to_end(key)
            else:
                misses+=1;cache[key]=e['dma_done']
                if len(cache)>capacity:cache.popitem(last=False)
        replay[str(capacity)]={'capacity_bytes_per_group':capacity*8192,'HBM_misses':misses,
            'reuse_hits':hits,'inflight_coalescing_hits':inflight_hits,'resident_cache_hits':resident_hits,
            'optimistic_read_saved_bytes':hits*8192,
            'optimistic_read_saved_percent':100*hits/len(tiles)}
    return {'status':'PASS','case':str(case),'jobs':len(flows),'tiles':len(tiles),
        'per_worker_flows':flows,
        'latency_cycles':{name:{'p50':percentile(v,.5),'p95':percentile(v,.95),'max':max(v)} for name,v in {
            'HBM_DMA':[e['dma_done']-e['submit'] for e in tiles],
            'GM_read':[e['gm_done']-e['dma_done'] for e in tiles],
            'ready_before_consume':[e['consume']-e['gm_done'] for e in tiles]}.items()},
        'four_worker_request_skew_cycles':{'p50':percentile(skews,.5),'p95':percentile(skews,.95),'max':max(skews) if skews else None},
        'bounded_reuse_replay':replay,
        'measurement_scope':'wait_dma and wait_gm partition foreground weight-ready stalls; latency sums overlap across requests and must not be added to job cycles; program_sum and compute_sum sum accepted command-to-callback durations, including WCP delivery; grouped launches overlap and change the launch count, so compute_sum is neither a wall-time union nor array-weighted MAC work; wait_array includes API rejection retries and software waits for an existing subset to finish',
        'replay_scope':'optimistic LRU admission at request time; in-flight requests may coalesce; ignores multicast delivery, GM writes, ACKs, synchronization and capacity occupied by other data; byte opportunity only, no simulated cycle speedup'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args();r=analyze(a.case)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n')
    print('PASS',r['jobs'],'jobs,',r['tiles'],'weight tiles')


if __name__=='__main__':main()
