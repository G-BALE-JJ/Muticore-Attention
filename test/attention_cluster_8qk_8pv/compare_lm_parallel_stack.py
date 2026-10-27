#!/usr/bin/env python3
"""Matched full-stack acceptance of LM N scheduling, including exact traffic deltas."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from run_decoder_layer import read_bands
from compare_decoder_groups_stack import read as read_common
from verify_decoder_dataflow_resources import verify

FIELDS=re.compile(r'(\w+)=(-?\d+)')

def phases(result):
    e=result['embedding'];d=result['dma_timing'];f=result['final_norm'];lm=result['lm_head']
    parts=dict(embedding=e['embedding_cycles'],embedding_handoff=e['handoff_cycles'],
               layer1_attention=d['layers']['1']['attention_cycles'],layer1_post_attention=d['layers']['1']['post_attention_cycles'],
               layer_handoff=d['layer_handoff_cycles'],layer2_attention=d['layers']['2']['attention_cycles'],
               layer2_post_attention=d['layers']['2']['post_attention_cycles'],final_norm_handoff=f['handoff_cycles'],
               final_norm=f['rmsnorm_cycles'],lm_head_handoff=lm['handoff_cycles'],lm_head=lm['lm_head_cycles'])
    assert sum(parts.values())==e['end_to_end_cycles']
    return parts


def analyze(baseline,candidate):
    cases=[baseline,candidate]
    manifests=[json.loads((c/'decoder_stack_inputs.json').read_text()) for c in cases]
    prov=[json.loads((c/'decoder_stack_provenance.json').read_text()) for c in cases]
    results=[json.loads((c/'decoder_stack_result.json').read_text()) for c in cases]
    assert all(r['status']=='PASS' for r in results)
    assert results[0]['shape']==results[1]['shape']==dict(S=1024,hidden=2048,intermediate=8192,layers=2)
    for k in ('source_sha256','guest_sha256','sst_library_sha256'):assert prov[0][k]==prov[1][k],k
    ignored={'GOLEM_DECODER_LM_N_PARALLEL','GOLEM_ARTIFACT_ROOT','GOLEM_DECODER_STACK_MANIFEST','GOLEM_MPI_ARGS','GOLEM_MPI_RANK_LOG_ROOT'}
    env=[p['environment'] for p in prov]
    diff={k:[e.get(k) for e in env] for k in env[0].keys()|env[1].keys() if env[0].get(k)!=env[1].get(k)}
    assert set(diff)<=ignored,diff
    assert env[0]['GOLEM_DECODER_LM_N_PARALLEL']=='1'
    parallel=int(env[1]['GOLEM_DECODER_LM_N_PARALLEL']);assert parallel in (2,4,8)
    def normalized(m):return dict(m,preloads=[{k:v for k,v in p.items() if k!='path'} for p in m['preloads']])
    assert normalized(manifests[0])==normalized(manifests[1])
    layout=manifests[0]['layout'];vocab=layout['lm_head']['vocab'];columns=(vocab+63)//64*64
    equality={};targets={f'layer{i}':l['output'] for i,l in enumerate(layout['layers'],1)}
    targets.update(final_norm=layout['final_norm']['output'],embedding=layout['embedding']['output'])
    targets.update({k:layout['shared'][k] for k in ('wo','h1','norm2')})
    for name,offset in targets.items():
        tensors=[read_bands(c,offset,2048) for c in cases]
        assert np.array_equal(tensors[0].view('<u2'),tensors[1].view('<u2')),name
        equality[name]=dict(status='PASS',elements=int(tensors[0].size),sha256=hashlib.sha256(tensors[0].tobytes()).hexdigest())
    logits=[np.load(c/'last_token_logits.npy') for c in cases]
    assert np.array_equal(logits[0].view('<u2'),logits[1].view('<u2'))
    equality['logits']=dict(status='PASS',elements=int(logits[0].size),sha256=hashlib.sha256(logits[0].tobytes()).hexdigest())
    tokens=[r['lm_head']['next_token']['token_id'] for r in results];assert tokens[0]==tokens[1]
    common=[read_common(c) for c in cases]
    assert common[0][1]==common[1][1], 'FFN group work changed'
    assert common[0][2]==common[1][2], 'SFU accepted-token work changed'
    flows=[];weight_audits=[]
    for c in cases:
        fs={};weights=set();k_order={}
        for line in next((c/'logs').glob('attention-sst*.log')).open():
            if '[DECODER_WEIGHT_TILE]' in line and ' stage=11 ' in line:
                f={k:int(v) for k,v in FIELDS.findall(line)}
                key=(f['core'],f['epoch'],f['n'],f['k']);assert key not in weights
                assert f['path']==f['row']==0 and f['submit']<=f['dma_done']<=f['gm_done']<=f['consume']
                order=(f['core'],f['epoch'],f['n']);assert f['k']==k_order.get(order,0);k_order[order]=f['k']+1
                tile=(f['epoch']-4)*128+f['n']
                assert f['address']==(1+tile%4)*layout['node_stride']+layout['lm_head']['weight']+(tile//4*32+f['k'])*8192
                weights.add(key)
            if '[DECODER_LM_FLOW]' in line:
                f={k:int(v) for k,v in FIELDS.findall(line)};key=(f['core'],f['epoch'])
                assert key not in fs;fs[key]=f
        expected={(core,4+chunk,n,k) for core in range(4,20) for chunk in range((columns+8191)//8192)
                  for n in range(core-4,min(128,columns//64-chunk*128),16) for k in range(32)}
        assert weights==expected and all(k==32 for k in k_order.values())
        weight_audits.append(dict(status='PASS',unique_dense_weight_tiles=len(weights),original_K_order_preserved=True))
        flows.append(fs)
    chunks=(columns+8191)//8192
    keys={(c,4+chunk) for c in range(4,20) for chunk in range(chunks)}
    assert all(set(f)==keys for f in flows)
    deltas={};expected_stage={}
    for core in range(4,20):
        tile_count=group_count=0;scatter_cycles=gather_cycles=0
        for chunk in range(chunks):
            width=min(128,columns//64-chunk*128)
            owned=list(range(core-4,width,16));tile_count+=len(owned)
            widths=[min(parallel,len(owned)-n) for n in range(0,len(owned),parallel)]
            group_count+=len(widths)
            scatter_cycles+=32*(2*len(owned)-sum(1+(p+1)//2 for p in widths))
            gather_cycles+=2*len(owned)-sum(1+(p+1)//2 for p in widths)
            a,b=[fs[core,4+chunk] for fs in flows]
            assert a['n_parallel']==1 and a['launches']==a['array_k_tiles']==0
            assert b['n_parallel']==parallel and b['launches']==len(widths)*32 and b['array_k_tiles']==len(owned)*32
            stage=expected_stage.setdefault(f'{4+chunk}.11',dict(read_requests_saved=0,write_requests_saved=0))
            stage['read_requests_saved']+=(len(owned)-len(widths))*32
            stage['write_requests_saved']+=len(owned)-len(widths)
        a,b=[p[0][core] for p in common]
        for k in ('matrix_broadcast_requests','matrix_broadcast_ingress_bytes','matrix_broadcast_sink_bytes','matrix_broadcast_transfer_cycles','input_scatter_bytes','output_scatter_requests','output_scatter_gather_bytes'):
            assert a[k]==b[k],(core,k)
        assert a['input_scatter_requests']-b['input_scatter_requests']==(tile_count-group_count)*32
        assert a['output_gather_requests']-b['output_gather_requests']==tile_count-group_count
        assert a['input_scatter_transfer_cycles']-b['input_scatter_transfer_cycles']==scatter_cycles
        assert a['output_scatter_gather_transfer_cycles']-b['output_scatter_gather_transfer_cycles']==gather_cycles
        assert a['transfer_cycles']-b['transfer_cycles']==scatter_cycles+gather_cycles
        deltas[core]=dict(owned_N_tiles=tile_count,parallel_groups=group_count,input_requests_saved=(tile_count-group_count)*32,
                          output_gathers_saved=tile_count-group_count,array_buffer_cycles_saved=scatter_cycles+gather_cycles)
    audits=[verify(c) for c in cases]
    assert all(a['status']=='PASS' and a['verified_decoder_jobs']==416 for a in audits)
    stages=[a['by_epoch_stage'] for a in audits];assert stages[0].keys()==stages[1].keys()
    for stage in stages[0]:
        a,b=[s[stage] for s in stages]
        if stage not in expected_stage:assert a==b,(stage,'upstream traffic changed');continue
        for resource in ('dma_read_transport','dma_write_transport'):assert a[resource]==b[resource],(stage,resource)
        d=expected_stage[stage]
        assert a['local_read']['bytes']-b['local_read']['bytes']==d['read_requests_saved']*128
        assert a['local_read']['requests']-b['local_read']['requests']==d['read_requests_saved']
        assert a['local_write']['bytes']==b['local_write']['bytes']
        assert a['local_write']['requests']-b['local_write']['requests']==d['write_requests_saved']
    cycles=[r['embedding']['end_to_end_cycles'] for r in results];lm=[r['lm_head']['lm_head_cycles'] for r in results]
    assert results[0]['lm_head']['first_input_dma_tick']==results[1]['lm_head']['first_input_dma_tick'],'upstream schedule changed'
    return dict(status='PASS',baseline=str(baseline),candidate=str(candidate),parallel=parallel,
                baseline_cycles=cycles[0],candidate_cycles=cycles[1],saved_cycles=cycles[0]-cycles[1],saved_percent=100*(1-cycles[1]/cycles[0]),
                lm_head_cycles=lm,lm_saved_percent=100*(1-lm[1]/lm[0]),phases={c.name:phases(r) for c,r in zip(cases,results)},
                useful_MACs=129115881472,useful_MAC_peak_efficiency=[1970152/c for c in cycles],
                fp16_bitwise_equality=equality,greedy_token=tokens[0],
                environment_differences=diff,array_deltas=deltas,baseline_resources=audits[0],candidate_resources=audits[1],
                SFU_accepted_tokens_equal=True,FFN_group_work_equal=True,LM_first_input_tick_equal=True,LM_weight_lifecycles=weight_audits,
                timing_scope='first token-ID DMA send to final logits ACK; host argmax excluded')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('baseline','candidate','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();r=analyze(a.baseline,a.candidate);a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:r[k] for k in ('status','parallel','baseline_cycles','candidate_cycles','saved_percent','lm_head_cycles','lm_saved_percent')},indent=2))

if __name__=='__main__':main()
