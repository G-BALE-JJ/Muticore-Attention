"""Reconcile software GM stage ownership and exact omitted HBM transfers."""
import json
from pathlib import Path
import re


def audit(case,jobs,manifest):
    p=case/'decoder_stack_provenance.json'
    env=json.loads(p.read_text()).get('environment',{}) if p.exists() else {}
    enabled=env.get('GOLEM_DECODER_GM_STAGE_CHAIN','0')=='1'
    if enabled!=bool(manifest.get('gm_stage_chain',False)):
        raise ValueError('stage-chain manifest and measured switch disagree')
    eligible=enabled and manifest.get('sequence',1024)<=1024
    expected={(j['core'],j['epoch'],j['stage']):j for js in jobs.values() for j in js
              if eligible and j['stage'] in (1,2,3,4)}
    flows=[]
    with next((case/'logs').glob('attention-sst*.log')).open() as stream:
        for line in stream:
            if '[DECODER_STAGE_CHAIN]' in line:flows.append({k:int(v) for k,v in re.findall(r'(\w+)=(-?\d+)',line)})
    if len(flows)!=len(expected) or {(f['core'],f['epoch'],f['stage']) for f in flows}!=set(expected):
        raise ValueError('stage-chain lifecycle missing/duplicate/unexpected')
    row_bytes=64*manifest.get('hidden',2048)*2
    ffn_savings={};residual_savings={}
    for f in flows:
        step=f['step'];j=expected[f['core'],f['epoch'],f['stage']]
        reads={1:0,2:row_bytes,3:row_bytes,4:2*row_bytes}[step]
        writes=row_bytes if step<4 else 0
        if (step!=f['stage'] or f['state']!=(step if step<4 else 0) or
            f['row_bytes']!=row_bytes or f['local_produced']!=(row_bytes if step<4 else 0) or
            f['saved_hbm_read']!=reads or f['saved_hbm_write']!=writes or f['capacity_added'] or f['status']):
            raise ValueError('stage-chain bytes/ownership/resource mismatch')
        if step<4 and j['hbm_write']!=0:raise ValueError('elided intermediate still materialized')
        residual_read=0 if manifest.get('input_residency') else row_bytes
        if step==2 and j['hbm_read']!=residual_read:
            raise ValueError('Add original X traffic differs from configured residency')
        if step==3 and j['hbm_read']!=manifest['hidden']*2:raise ValueError('Norm2 must read only gamma from HBM')
        if step==4:
            ffn_savings[f['core'],f['epoch']]=reads
            residual_savings[f['core'],f['epoch']]=row_bytes
    return dict(status='PASS',enabled=enabled,eligible=eligible,flows=flows,
                saved_hbm_read_bytes=sum(f['saved_hbm_read'] for f in flows),
                saved_hbm_write_bytes=sum(f['saved_hbm_write'] for f in flows)),ffn_savings,residual_savings
