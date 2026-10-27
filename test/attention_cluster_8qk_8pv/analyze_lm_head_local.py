#!/usr/bin/env python3
"""Audit isolated LM port deltas, dense weight lifecycles and resource bounds."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from run_lm_head_micro import compare

FIELDS=re.compile(r'(\w+)=(-?\d+)')

def analyze(root):
    m=json.loads((root/'inputs/inputs.json').read_text())
    results={str(n):json.loads((root/f'n{n}/result.json').read_text()) for n in (1,2,4,8)}
    summary=compare(results);workers=summary['shape']['workers'];h=m['hidden'];columns=m['columns'];chunks=(columns+8191)//8192
    baseline=results['1'];details={}
    for n in (1,2,4,8):
        r=results[str(n)];case=root/f'n{n}';tiles={};flows={};jobs={}
        for line in (case/'logs/attention-sst.log').open():
            if '[DECODER_WEIGHT_TILE]' in line:
                f={k:int(v) for k,v in FIELDS.findall(line)};key=(f['core'],f['epoch'],f['n'],f['k'])
                assert key not in tiles and f['row']==f['path']==0
                assert f['submit']<=f['dma_done']<=f['gm_done']<=f['consume']
                tile=(f['epoch']-4)*128+f['n'];stride=m['layout']['node_stride']
                assert f['address']==(1+tile%4)*stride+m['layout']['lm_head']['weight']+(tile//4*(h//64)+f['k'])*8192
                tiles[key]=f
            elif '[DECODER_DATAFLOW]' in line:
                f={k:int(v) for k,v in FIELDS.findall(line)};key=(f['core'],f['epoch']);assert key not in flows;flows[key]=f
            elif '[DECODER_OP] event=done' in line:
                f={k:int(v) for k,v in FIELDS.findall(line)};key=(f['core'],f['epoch']);assert key not in jobs;jobs[key]=f
        expected={(c,4+chunk,t,k) for c in range(4,4+workers) for chunk in range(chunks)
                  for t in range(c-4,min(128,columns//64-chunk*128),16) for k in range(h//64)}
        assert set(tiles)==expected and len(tiles)==r['measurement']['dense_weight_tiles']
        assert len(flows)==len(jobs)==workers*chunks
        for key,f in flows.items():
            count=sum(t[:2]==key for t in expected)
            assert f['prefetch']==1 and f['prefetch_requests']==f['prefetch_hits']==count
        per_core={};expected_stage=defaultdict(lambda:dict(read_requests_saved=0,write_requests_saved=0))
        for c in range(4,4+workers):
            total=groups=0;scatter=gather=0
            for chunk in range(chunks):
                count=len(range(c-4,min(128,columns//64-chunk*128),16));widths=[min(n,count-i) for i in range(0,count,n)]
                total+=count;groups+=len(widths)
                scatter+=(h//64)*(2*count-sum(1+(p+1)//2 for p in widths))
                gather+=2*count-sum(1+(p+1)//2 for p in widths)
                d=expected_stage[f'{4+chunk}.11'];d['read_requests_saved']+=(count-len(widths))*(h//64);d['write_requests_saved']+=count-len(widths)
            a=baseline['measurement']['array_stats'][str(c)];b=r['measurement']['array_stats'][str(c)]
            for k in ('matrix_broadcast_requests','matrix_broadcast_ingress_bytes','matrix_broadcast_sink_bytes','matrix_broadcast_transfer_cycles','input_scatter_bytes','output_scatter_requests','output_scatter_gather_bytes'):
                assert a[k]==b[k],(n,c,k)
            assert a['input_scatter_requests']-b['input_scatter_requests']==(total-groups)*(h//64)
            assert a['output_gather_requests']-b['output_gather_requests']==total-groups
            assert a['input_scatter_transfer_cycles']-b['input_scatter_transfer_cycles']==scatter
            assert a['output_scatter_gather_transfer_cycles']-b['output_scatter_gather_transfer_cycles']==gather
            assert a['transfer_cycles']-b['transfer_cycles']==scatter+gather
            ts=[t for key,t in tiles.items() if key[0]==c]
            per_core[c]=dict(owned_N_tiles=total,parallel_N_groups=groups,port_cycles_saved=scatter+gather,
                             prefetch_wait_cycles=sum(f['prefetch_wait_cycles'] for key,f in flows.items() if key[0]==c),
                             job_duration_cycles=sum((j['tick']-j['start_tick'])/1000 for key,j in jobs.items() if key[0]==c),
                             mean_weight_DMA_RTT_cycles=sum(t['dma_done']-t['submit'] for t in ts)/len(ts),
                             mean_weight_GM_wait_and_service_cycles=sum(t['gm_done']-t['dma_done'] for t in ts)/len(ts))
        for stage,d in expected_stage.items():
            a=baseline['resources']['by_epoch_stage'][stage];b=r['resources']['by_epoch_stage'][stage]
            for resource in ('dma_read_transport','dma_write_transport'):assert a[resource]==b[resource]
            assert a['local_read']['bytes']-b['local_read']['bytes']==d['read_requests_saved']*128
            assert a['local_read']['requests']-b['local_read']['requests']==d['read_requests_saved']
            assert a['local_write']['bytes']==b['local_write']['bytes']
            assert a['local_write']['requests']-b['local_write']['requests']==d['write_requests_saved']
        read=r['measurement']['hbm_read_bytes'];write=r['measurement']['hbm_write_bytes']-workers*8
        # Use per-node payload to avoid assuming a single worker uses all HBM channels.
        node_payload=defaultdict(int)
        for c in range(4,4+workers):
            count=len(range(c-4,columns//64,16));node_payload[1+(c-4)%4]+=count*h*128+count*128
        node_payload[4]+=workers*chunks*h*2
        bound=max(node_payload.values())/320
        macs=len(expected)*4096
        details[str(n)]=dict(dense_weight_tiles=len(expected),matrix_ingress_bytes=len(expected)*8192,
                             hbm_payload_bytes=read+write,HBM_payload_lower_bound_cycles=bound,
                             measured_over_HBM_bound=r['measurement']['cycles']/bound,
                             useful_MACs=macs,useful_MAC_peak_efficiency=macs/(workers*64*64*r['measurement']['cycles']),
                             physical_MAC_array_weighted_busy_fraction=len(expected)*66/(workers*64*r['measurement']['cycles']),
                             per_worker=per_core)
    return dict(status='PASS',comparison=summary,details=details,
                scope='payload-only per-node HBM bound; excludes DRAM commands/refresh/NoC overhead; RTT means overlap, never summed as wall time')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=analyze(a.root);a.output.write_text(json.dumps(r,indent=2)+'\n');print(r['status'],r['comparison']['shape'])

if __name__=='__main__':main()
