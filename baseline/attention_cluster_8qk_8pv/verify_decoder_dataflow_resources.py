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


def residual_audit(case, expected_jobs, hidden, groups=1, local_savings=None):
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
        if f['residual_hbm_bytes']!=groups*64*hidden*2-(local_savings or {}).get((f['core'],f['epoch']),0) or f['scratch_bytes']!=8192 or f['chunks']!=groups*chunks:
            raise ValueError('residual epilogue traffic or scratch mismatch')
    return {'status':'PASS','verified_jobs':len(flows),'scratch_bytes':8192,
            'gemm_overlap_cycles_sum':sum(f['gemm_overlap_cycles'] for f in flows)}


def block_pipeline_audit(case, jobs):
    provenance_path=case/'decoder_stack_provenance.json'
    if not provenance_path.exists():provenance_path=case/'provenance.json'
    env=json.loads(provenance_path.read_text()).get('environment',{}) if provenance_path.exists() else {}
    enabled=env.get('GOLEM_DECODER_FFN_BLOCK_PIPELINE','0')=='1'
    flows=[]
    with next((case/'logs').glob('attention-sst*.log')).open() as stream:
        for line in stream:
            if '[DECODER_BLOCK_PIPELINE]' in line:flows.append({k:int(v) for k,v in FIELDS.findall(line)})
    expected={(j['core'],j['epoch'],j['stage']):j for js in jobs.values() for j in js if j['op']==9} if enabled else {}
    if len(flows)!=len(expected) or {(f['core'],f['epoch'],f['stage']) for f in flows}!=set(expected):
        raise ValueError('missing/duplicate/unexpected block pipeline lifecycle')
    for f in flows:
        if (f['arrays']!=64 or f['contexts']!=1 or f['extra_gm_capacity']!=0 or
            f['owner_at_finish'] or f['pending_at_finish'] or f['panels']<=0):
            raise ValueError('pipeline ownership, scratch or completion invariant failed')
        if ('GOLEM_DECODER_FFN_PIPELINE_POLICY' in env and
            f.get('policy')!=int(env['GOLEM_DECODER_FFN_PIPELINE_POLICY'])):
            raise ValueError('pipeline execution policy differs from recorded configuration')
    return dict(status='PASS',enabled=enabled,panel_tiles=int(env.get('GOLEM_DECODER_FFN_PANEL_TILES','16')),flows=flows,
                both_active_cycles=sum(f['both_active'] for f in flows),
                producer_transfer_down_compute_cycles=sum(f['producer_transfer_down_compute'] for f in flows),
                down_transfer_producer_compute_cycles=sum(f['down_transfer_producer_compute'] for f in flows),
                note='Both engines may prepare concurrently; all 64 arrays have one exclusive C owner. Overlap counters are not additive critical-path time.')


def optimization_audit(case, jobs, manifest):
    path=case/'decoder_stack_provenance.json'
    if not path.exists():path=case/'provenance.json'
    env=json.loads(path.read_text()).get('environment',{}) if path.exists() else {}
    streams=[];handoffs=[]
    with next((case/'logs').glob('attention-sst*.log')).open() as stream:
        for line in stream:
            if '[DECODER_SFU_STREAM]' in line:streams.append({k:int(v) for k,v in FIELDS.findall(line)})
            if '[DECODER_NORM_HANDOFF]' in line:handoffs.append({k:int(v) for k,v in FIELDS.findall(line)})
    ffn={(j['core'],j['epoch'],j['stage']) for js in jobs.values() for j in js if j.get('op') in (8,9)}
    streaming=env.get('GOLEM_DECODER_FFN_SFU_STREAM','0')=='1'
    if streaming or streams:
        if len(streams)!=len(ffn) or {(f['core'],f['epoch'],f['stage']) for f in streams}!=ffn:
            raise ValueError('SFU stream lifecycle missing/duplicate')
        for f in streams:
            if (f['enabled']!=int(streaming) or f['contexts']!=2 or f['drained']!=1 or
                f['chunk_elements']!=int(env.get('GOLEM_DECODER_FFN_SFU_CHUNK','8192')) or
                not 0<=f['high_water']<=2 or (streaming and f['high_water']==0)):
                raise ValueError('SFU stream configuration/capacity/drain mismatch')
    enabled=env.get('GOLEM_DECODER_NORM_FFN_HANDOFF','0')=='1'
    eligible=enabled and manifest.get('sequence',1024)<=1024
    expected={(j['core'],j['epoch'],j['stage']) for js in jobs.values() for j in js
              if eligible and ((j.get('op')==3 and j['stage']==3) or j.get('op')==9)}
    if len(handoffs)!=len(expected) or {(f['core'],f['epoch'],f['stage']) for f in handoffs}!=expected:
        raise ValueError('Norm handoff lifecycle missing/duplicate/unexpected')
    savings={};row_bytes=64*manifest.get('hidden',2048)*2
    for f in handoffs:
        producer=f['stage']==3
        if (f['producer']!=int(producer) or f['consumer']!=int(not producer) or
            f['cache_bytes']!=(row_bytes if producer else 0) or
            f['saved_hbm_read']!=(0 if producer else row_bytes) or
            f['materialized']!=1 or f['capacity_added']!=0):
            raise ValueError('Norm handoff byte/configuration mismatch')
        if not producer:savings[f['core'],f['epoch']]=f['saved_hbm_read']
    return dict(status='PASS',sfu_stream_enabled=streaming,sfu_stream_flows=streams,
                norm_handoff_enabled=enabled,norm_handoff_flows=handoffs,
                saved_hbm_read_bytes=sum(savings.values())),savings


def verify(case):
    log = next((case / 'logs').glob('attention-sst*.log'))
    jobs = defaultdict(list)
    shared = {}
    with log.open() as stream:
        for line in stream:
            if '[DECODER_OP] event=done' in line:
                j = {k: int(v) for k, v in FIELDS.findall(line)}
                if j['status']:
                    raise ValueError('failed decoder job')
                jobs[j['core']].append(j)
            if '[GEMM_WEIGHT_SHARE]' in line:
                f={k:int(v) for k,v in FIELDS.findall(line)}
                key=f['core'],f['epoch']
                if key in shared:raise ValueError('duplicate weight-sharing lifecycle')
                shared[key]=f
    for js in jobs.values():
        js.sort(key=lambda j: j['start_tick'])
    manifest=json.loads((case/'decoder_stack_inputs.json').read_text()) if (case/'decoder_stack_inputs.json').exists() else {}
    optimizations,handoff_savings=optimization_audit(case,jobs,manifest)
    from verify_software_stage_chain import audit as audit_stage_chain
    chain,chain_savings,residual_savings=audit_stage_chain(case,jobs,manifest)
    optimizations['stage_chain']=chain
    from verify_add_norm_fusion import audit as audit_add_norm
    optimizations['add_norm_fusion']=audit_add_norm(case,jobs,manifest)
    from verify_layer_residency import audit as audit_layer_residency, audit_input_residency, verify_input_payload, verify_projection_payload
    optimizations['layer_residency']=audit_layer_residency(case,jobs,manifest)
    optimizations['layer_residency']['actual_projection_payload']=verify_projection_payload(case,manifest)
    optimizations['input_residency']=audit_input_residency(case,jobs,manifest)
    optimizations['input_residency']['actual_payload']=verify_input_payload(case,jobs,manifest)
    from verify_output_residency import audit as audit_output_residency
    optimizations['output_residency']=audit_output_residency(case,jobs,manifest)
    for key,value in chain_savings.items():handoff_savings[key]=handoff_savings.get(key,0)+value
    if manifest.get('ffn_panel_fusion'):
        hidden=manifest['hidden'];width=4*hidden;groups=manifest.get('sequence',1024)//1024
        fused_down=bool(manifest.get('ffn_down_fusion'))
        expected_read=(3 if fused_down else 2)*(width//64)*(hidden//64)*8192+64*hidden*2
        residual=bool(manifest.get('down_residual_fusion'))
        if residual:expected_read+=64*hidden*2
        expected_read*=groups
        expected_write=groups*64*(hidden if fused_down else width)*2+(8 if residual else 0)
        paired=[j for js in jobs.values() for j in js if j['stage']==4]
        if shared:
            if set(shared)!={(j['core'],j['epoch']) for j in paired}:raise ValueError('missing shared FFN worker lifecycle')
            weights=groups*3*(width//64)*(hidden//64)*8192
            for j in paired:
                f=shared[j['core'],j['epoch']]
                leader=4+(j['core']-4)%4+((j['core']-4)//4//f['cohort'])*f['cohort']*4
                if (f['cohort'] not in (2,4) or f['slots']!=4 or f['leader']!=leader or f['released']!=1
                    or f['requests']!=weights//8192 or f['gm_write']!=weights
                    or (j['core']!=leader and (f['hbm_bytes'] or f['server_requests']))
                    or (j['core']==leader and f['server_requests']!=f['requests']*f['cohort'])
                    or j['hbm_read']!=expected_read-weights+f['hbm_bytes']):
                    raise ValueError('shared FFN dense work, physical fetch or release mismatch')
        if len(paired)!=16*manifest.get('layers',2) or any(j['op']!=(9 if fused_down else 8) or
                (not shared and j['hbm_read']!=expected_read-handoff_savings.get((j['core'],j['epoch']),0)) or
                 j['hbm_write']!=(8 if manifest.get('elide_intermediate_output') and j['epoch']<manifest.get('layers',2) else expected_write)
                for j in paired):
            raise ValueError('fused FFN must read one Norm2 row group plus all dense weights and write only '+('final residual' if residual else 'down' if fused_down else 'gated'))
    residual_flow=None
    if manifest.get('down_residual_fusion'):
        if any(j['stage']==9 for js in jobs.values() for j in js):
            raise ValueError('standalone residual must be elided')
        residual_flow=residual_audit(case,{(j['core'],j['epoch'],j['stage']) for js in jobs.values() for j in js if j['stage']==4},manifest['hidden'],manifest.get('sequence',1024)//1024,residual_savings)
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
            'optimizations':optimizations,
            'weight_cohort':list(shared.values()),'block_pipeline':block_pipeline_audit(case,jobs),
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
