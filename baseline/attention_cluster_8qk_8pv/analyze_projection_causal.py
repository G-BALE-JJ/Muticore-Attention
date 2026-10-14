#!/usr/bin/env python3
"""Join Projection operand requests and array commands to physical resource timestamps."""
import argparse
from collections import defaultdict
from bisect import bisect_left,bisect_right
import json
from pathlib import Path

from analyze_resource_timeline import FIELDS, merged, intersection

KEY_FIELDS = ('block', 'kind', 'head', 'dim', 'row', 'tile')


def key(record):
    return tuple(int(record[k]) for k in KEY_FIELDS)


def stats(values):
    values = sorted(values)
    return {'count': len(values), 'sum_cycles': sum(values), 'mean_cycles': sum(values)/len(values) if values else 0,
            'p50_cycles': values[len(values)//2] if values else 0,
            'p95_cycles': values[min(len(values)-1, int(len(values)*.95))] if values else 0,
            'max_cycles': max(values, default=0)}


def analyze(case):
    case = Path(case)
    log, = (case/'logs').glob('attention-sst*.log')
    factor = json.loads((case/'resource_analysis.json').read_text())['ticks_per_cycle']
    windows = {}
    with log.open() as stream:
        for line in stream:
            if '[PROJECTION_JOB]' in line:
                r = dict(FIELDS.findall(line))
                windows[int(r['core'])] = (int(r['start'])*factor, int(r['end'])*factor)
    causal = log.with_suffix('.causal.jsonl')
    if not causal.exists():
        causal = log
    events = defaultdict(list)
    with causal.open() as stream:
        for line in stream:
            if '[PROJECTION_CAUSAL]' in line:
                r = dict(FIELDS.findall(line))
                for field in ('core', 'tick', 'tag', 'address', 'bytes', *KEY_FIELDS):
                    r[field] = int(r[field])
                if r['event'].endswith('failed'):
                    raise ValueError(f'causal failure: {r}')
                events[r['core']].append(r)
    if set(events) != set(range(4,20)):
        raise ValueError('causal events must cover workers 4..19')
    resources = defaultdict(list)
    with log.with_suffix('.resources.jsonl').open() as stream:
        for line in stream:
            if '[RESOURCE_INTERVAL]' not in line: continue
            r = dict(FIELDS.findall(line))
            core = int(r['core'])
            if core not in windows: continue
            lo, hi = windows[core]
            if int(r['submit_tick']) >= hi or int(r['visible_tick']) < lo: continue
            for field in ('core','tag','address','bytes','submit_tick','start_tick','done_tick','visible_tick'):
                r[field] = int(r[field])
            resources[core].append(r)

    workers = {}
    tile_file = case/'projection_tile_timeline.jsonl'
    with tile_file.open('w') as tile_stream:
        for core in sorted(events):
            es = events[core]
            rs = resources[core]
            read_submit = {r['tag']: r for r in es if r['event']=='read_submit'}
            read_done = {r['tag']: r for r in es if r['event']=='read_done'}
            physical_reads = {r['tag']: r for r in rs if r['domain']=='gm' and r['resource']=='local_read'}
            compute = merged((r['start_tick'],r['done_tick']) for r in rs if r['domain']=='array' and r['resource']=='compute')
            compute_starts=[r[0] for r in compute]
            compute_ends=[r[1] for r in compute]
            roles = defaultdict(lambda: defaultdict(list))
            ready = defaultdict(list)
            read_count = 0
            for tag, event in read_submit.items():
                if tag not in physical_reads or tag not in read_done:
                    raise ValueError(f'missing read join core={core} tag={tag}')
                r = physical_reads[tag]
                done = read_done[tag]
                if (event['tick'],event['address'],event['bytes'],done['tick']) != (
                    r['submit_tick'],r['address'],r['bytes'],r['visible_tick']):
                    raise ValueError(f'read timestamp/payload mismatch core={core} tag={tag}')
                role = event['role']
                roles[role]['queue'].append((r['start_tick']-r['submit_tick'])/factor)
                roles[role]['service'].append((r['done_tick']-r['start_tick'])/factor)
                roles[role]['request'].append((r['visible_tick']-r['submit_tick'])/factor)
                roles[role]['service_overlap_mac'].append(sum(max(0,min(b,r['done_tick'])-max(a,r['start_tick'])) for a,b in compute[bisect_right(compute_ends,r['start_tick']):bisect_left(compute_starts,r['done_tick'])])/factor)
                ready[(key(event), role)].append(done['tick'])
                read_count += 1
            tiles = defaultdict(dict)
            for event in es:
                if event['event'] in ('matrix_submit','matrix_done','scatter_submit','scatter_done',
                                      'mac_submit','mac_done','decode_start','decode_done'):
                    tile = tiles[key(event)]
                    if event['event'] in tile:
                        raise ValueError(f'duplicate tile event core={core} {key(event)} {event["event"]}')
                    tile[event['event']] = event
            if not tiles: raise ValueError('no tile events')
            physical_array = {(r['resource'],r['tag']):r for r in rs if r['domain']=='array' and r['resource']!='compute'}
            cmd_stats = defaultdict(list)
            mac_records=sorted((r for r in rs if r['domain']=='array' and r['resource']=='compute'),key=lambda r:r['start_tick'])
            if len(mac_records)!=len(tiles): raise ValueError('tile/MAC resource count mismatch')
            mac_by_key={}
            ordered_tiles=sorted(tiles.items(),key=lambda item:item[1]['mac_submit']['tick'])
            for (tk,t),r in zip(ordered_tiles,mac_records):
                if not t['mac_submit']['tick'] <= r['start_tick'] <= r['done_tick'] <= t['mac_done']['tick']:
                    raise ValueError('MAC request does not enclose physical service')
                mac_by_key[tk]=r
            previous_mac={tk:(mac_records[i-1]['done_tick'] if i else None) for i,(tk,t) in enumerate(ordered_tiles)}
            joins = 0
            for tile_key, tile in tiles.items():
                required = ('matrix_submit','matrix_done','scatter_submit','scatter_done','mac_submit','mac_done','decode_start','decode_done')
                if set(tile) != set(required):
                    raise ValueError(f'incomplete tile core={core} key={tile_key}: {set(tile)}')
                out = {'core':core, **dict(zip(KEY_FIELDS,tile_key)),
                       **{name+'_cycle':tile[name]['tick']/factor for name in required}}
                for operation, resource in (('matrix','matrix_broadcast'),('scatter','input_scatter')):
                    event = tile[operation+'_submit']
                    r = physical_array.get((resource,event['tag']))
                    if r is None:
                        raise ValueError(f'array resource join missing core={core} tag={event["tag"]}')
                    if r['visible_tick'] != tile[operation+'_done']['tick']:
                        raise ValueError('array callback timestamp mismatch')
                    out[operation+'_queue_cycles'] = (r['start_tick']-r['submit_tick'])/factor
                    out[operation+'_service_cycles'] = (r['done_tick']-r['start_tick'])/factor
                    out[operation+'_command_before_resource_cycles'] = (r['submit_tick']-event['tick'])/factor
                    out[operation+'_visibility_cycles'] = (r['visible_tick']-r['done_tick'])/factor
                    for name in ('queue','service','command_before_resource','visibility'):
                        cmd_stats[operation+'_'+name].append(out[operation+'_'+name+'_cycles'])
                    joins += 1
                mac_resource=mac_by_key[tile_key]
                out['physical_mac_start_cycle']=mac_resource['start_tick']/factor
                out['physical_mac_done_cycle']=mac_resource['done_tick']/factor
                prev=previous_mac[tile_key]
                if prev is not None:
                    out['gap_after_previous_mac_cycles']=(mac_resource['start_tick']-prev)/factor
                    out['wait_before_matrix_submit_cycles']=(tile['matrix_submit']['tick']-prev)/factor
                    cmd_stats['wait_before_matrix_submit'].append(out['wait_before_matrix_submit_cycles'])
                weight_ready = max(ready.get((tile_key,'foreground_weight'), [])+ready.get((tile_key,'background_weight'), []), default=None)
                if weight_ready is not None:
                    out['weight_read_done_cycle'] = weight_ready/factor
                    out['weight_ready_to_matrix_cycles'] = (tile['matrix_submit']['tick']-weight_ready)/factor
                    if out['weight_ready_to_matrix_cycles'] < 0: raise ValueError('matrix submitted before operand read completes')
                    if prev is not None:
                        out['weight_read_late_after_previous_mac_cycles']=max(0,weight_ready-prev)/factor
                        cmd_stats['weight_read_late_after_previous_mac'].append(out['weight_read_late_after_previous_mac_cycles'])
                cmd_stats['decode'].append((tile['decode_done']['tick']-tile['decode_start']['tick'])/factor)
                tile_stream.write(json.dumps(out)+'\n')
            dma_done = defaultdict(list)
            for event in es:
                if event['event']=='dma_done': dma_done[event['address']].append(event)
            dma_submit = defaultdict(list)
            for event in es:
                if event['event']=='dma_submit': dma_submit[event['address']].append(event)
            dma_physical = defaultdict(list)
            for r in rs:
                if r['domain']=='gm' and r['resource']=='dma_read_transport' and r['address'] in dma_submit:
                    dma_physical[r['address']].append(r)
            dma_stats = defaultdict(list)
            for address, submits in dma_submit.items():
                actual = sorted(dma_physical[address],key=lambda r:r['submit_tick'])
                submits = sorted(submits,key=lambda r:r['tick'])
                done_events = sorted(dma_done[address],key=lambda r:r['tick'])
                if len(done_events)!=len(submits): raise ValueError('DMA callback count differs')
                if len(actual)!=len(submits):
                    raise ValueError(f'DMA address join counts differ core={core} address={address}: {len(actual)} vs {len(submits)}')
                for event,r,completion in zip(submits,actual,done_events):
                    if r['submit_tick'] < event['tick']: raise ValueError('DMA precedes caller')
                    dma_stats['pre_send'].append((r['start_tick']-event['tick'])/factor)
                    dma_stats['transport'].append((r['done_tick']-r['start_tick'])/factor)
                    dma_stats['arrival_to_callback'].append((completion['tick']-r['done_tick'])/factor)
                    dma_stats['submit_to_callback'].append((completion['tick']-event['tick'])/factor)
                    if completion['tick'] < r['done_tick']: raise ValueError('DMA callback precedes data arrival')
            workers[str(core)] = {'joined_local_read_requests':read_count,'joined_array_commands':joins,
                'tile_count':len(tiles),'local_read_by_role':{role:{k:stats(v) for k,v in metrics.items()} for role,metrics in roles.items()},
                'array_command':{k:stats(v) for k,v in cmd_stats.items()},'weight_dma':{k:stats(v) for k,v in dma_stats.items()}}
    return {'status':'PASS','artifact_root':str(case),'ticks_per_cycle':factor,'by_worker':workers,
            'tile_timeline':str(tile_file),'scope':'Exact local-read joins use original tag, submit/visible tick, address and bytes. Matrix/scatter joins use resource/tag and callback visibility. Weight DMA is paired chronologically per core/address; transport includes NoC, memory and response. Decode is host C++ and has zero modeled cycles.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('case',type=Path)
    args=p.parse_args()
    result=analyze(args.case)
    (args.case/'projection_causal_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'PASS causal joins: {args.case}, {sum(v["tile_count"] for v in result["by_worker"].values())} tiles')

if __name__=='__main__': main()
