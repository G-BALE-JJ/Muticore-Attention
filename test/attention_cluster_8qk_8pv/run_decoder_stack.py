#!/usr/bin/env python3
"""Parameterized prefill stack; existing kernels, shared allocator and CPU oracles."""
import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np
from run_decoder_layer import GUEST_DIR, linear_reference, storage_half, read_bands
from run_batch_b import environment
from run_projection_overlap import ROOT, HERE, digest
from artifact_storage import preflight


@lru_cache(maxsize=None)
def memory_layout(hq, hkv, count=2, final_norm=False, vocab=0, node_bytes=128*2**20,embedding=False,sequence=1024,retain_outputs=False):
    # C++ is the single source of offsets. Python never duplicates its arithmetic.
    with tempfile.TemporaryDirectory() as directory:
        exe=Path(directory)/'layout'
        subprocess.run(['g++','-std=c++17','-O2',str(GUEST_DIR/'dump_decoder_layout.cpp'),'-o',str(exe)],check=True)
        result=subprocess.run([str(exe),str(sequence),str(hq),str(hkv),'64',str(count),str(int(final_norm)),str(vocab),str(node_bytes),str(int(embedding)),str(int(retain_outputs))],check=True,capture_output=True,text=True)
        return json.loads(result.stdout)


def sparse_weight(rng,n,k,scale):
    w=np.zeros((n,k),dtype='<f2')
    for row in range(n):
        cols=sorted(set(((row*17+3)%k,(row*31+79)%k,(row*7+137)%k,k-1)))
        w[row,cols]=rng.uniform(-scale,scale,len(cols)).astype('<f2')
    return w


def prepare(case,hq,hkv,final_norm=False,vocab=0,node_bytes=128*2**20,embedding=False,
            layers=2,sequence=1024,weight_kind='sparse',retain_outputs=False):
    if not 1<=layers<=32 or not 1<=sequence<=8192 or weight_kind not in ('sparse','dense'):
        raise ValueError('invalid prefill layer count, sequence or weight kind')
    logical=sequence;physical=((sequence+1023)//1024)*1024;rows=physical//4
    layout=memory_layout(hq,hkv,layers,final_norm=final_norm,vocab=vocab,node_bytes=node_bytes,
                         embedding=embedding,sequence=physical,retain_outputs=retain_outputs); hidden=hq*64
    def weight(rng,n,k,scale):
        if weight_kind=='sparse':return sparse_weight(rng,n,k,scale)
        # Match the variance of four nonzero uniform values per row, without
        # reducing the number of dense hardware tiles or changing precision.
        return rng.uniform(-scale*np.sqrt(4/k),scale*np.sqrt(4/k),(n,k)).astype('<f2')
    rng=np.random.default_rng(20261015)
    preloads=[]
    def preload(name,path,node,offset):
        preloads.append({'name':name,'path':str(path),'node':node,'offset':offset,
                         'bytes':path.stat().st_size,'sha256':digest(path)})
    x=rng.normal(0,.3,(physical,hidden)).astype('<f2'); x[logical:]=0; x.tofile(case/'projection_x.bin')
    for node in range(1,5):
        path=case/f'x_node{node}.bin';x[(node-1)*rows:node*rows].tofile(path)
        if not embedding:preload('X0',path,node,layout['activations'][0])
    for index,l in enumerate(layout['layers'],1):
        for name in ('gamma1','gamma2'):
            data=rng.uniform(.8,1.2,hidden).astype('<f2')
            path=case/f'layer{index}_{name}.bin';data.tofile(path)
            for node in range(1,5):preload(name,path,node,l[name])
        for name,n,k,scale in [('qkv',hidden+2*hkv*64,hidden,.06),('wo',hidden,hidden,.25),
                              ('gate',4*hidden,hidden,.25),('up',4*hidden,hidden,.25),
                              ('down',hidden,4*hidden,.25)]:
            w=weight(rng,n,k,scale)
            np.save(case/f'layer{index}_{name}.npy',w)
            panels=w.reshape(n//64,64,k//64,64).transpose(0,2,1,3).copy()
            for node in range(1,5):
                path=case/f'layer{index}_{name}_node{node}.bin'
                (panels if name=='qkv' else panels[node-1::4]).tofile(path)
                preload(name,path,node,l[name+'_weight'])
    sys.path.insert(0,str(GUEST_DIR))
    from attention_case import generate_rope_table
    path=case/'rope_64.bin';generate_rope_table(path,physical,64)
    preload('rope',path,1,layout['shared']['rope'])
    # Poisoned flags prove that the guest clears both sets before layer 1.
    path=case/'poison_flags.bin';np.full(4,0xfeedface,dtype='<u8').tofile(path)
    for node in range(1,5):
        for name in ('projection_done','decoder_done'):
            preload(name,path,node,layout['shared'][name])
    if final_norm:
        gamma=case/'final_gamma.bin';rng.uniform(.7,1.3,hidden).astype('<f2').tofile(gamma)
        output=case/'poison_final_output.bin';np.full((rows,hidden),np.nan,dtype='<f2').tofile(output)
        for node in range(1,5):
            preload('final_gamma',gamma,node,layout['final_norm']['gamma'])
            preload('final_done',path,node,layout['final_norm']['done'])
            preload('final_output',output,node,layout['final_norm']['output'])
    manifest={'sequence':physical,'logical_sequence':logical,'padding_rows':physical-logical,'retain_outputs':retain_outputs,'hidden':hidden,'num_query_heads':hq,'num_kv_heads':hkv,
              'layers':layers,'weight_kind':weight_kind+' synthetic matrices; all dense hardware tiles execute',
              'layout':layout,'preloads':preloads}
    if final_norm:manifest['final_norm']=True
    if vocab:
        columns=((vocab+63)//64)*64
        # Keep all Decoder inputs identical. A separate seed defines the head.
        w=weight(np.random.default_rng(20261016),columns,hidden,.15)
        if columns>vocab:w[vocab:]=0
        np.save(case/'lm_weight.npy',w)
        panels=w.reshape(columns//64,64,hidden//64,64).transpose(0,2,1,3).copy()
        for node in range(1,5):
            weight=case/f'lm_weight_node{node}.bin';panels[node-1::4].tofile(weight)
            preload('lm_weight',weight,node,layout['lm_head']['weight'])
            preload('lm_done',path,node,layout['lm_head']['done'])
        output=case/'poison_lm_output.bin';np.full(columns,np.nan,dtype='<f2').tofile(output)
        for node in range(1,5):preload('lm_output',output,node,layout['lm_head']['output'])
        manifest['lm_head']=True
    if embedding:
        rng_embedding=np.random.default_rng(20261018)
        table=rng_embedding.normal(0,.3,(vocab,hidden)).astype('<f2')
        np.save(case/'embedding_weight.npy',table)
        tokens=rng_embedding.integers(0,vocab,physical,dtype=np.uint32)
        tokens[:min(4,logical)]=np.arange(min(4,logical))%vocab
        if logical>=8:tokens[4:8]=tokens[:4]
        tokens[logical-1]=vocab-1;tokens[logical:]=0
        tokens.astype('<u4').tofile(case/'input_token_ids.bin')
        x=table[tokens];x.tofile(case/'projection_x.bin')
        for node in range(1,5):
            weight=case/f'embedding_weight_node{node}.bin';table[node-1::4].tofile(weight)
            ids=case/f'embedding_tokens_node{node}.bin';tokens[(node-1)*rows:node*rows].astype('<u4').tofile(ids)
            preload('embedding_weight',weight,node,layout['embedding']['weight'])
            preload('embedding_tokens',ids,node,layout['embedding']['tokens'])
            preload('embedding_done',case/'poison_flags.bin',node,layout['embedding']['done'])
            preload('embedding_output',case/'poison_final_output.bin',node,layout['embedding']['output'])
        manifest['embedding']=True
    manifest['oracle_sha256']={p.name:digest(p) for p in sorted(case.glob('*.npy'))}
    (case/'decoder_stack_inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def rms(x,gamma):
    f=x.astype(np.float32); square=np.zeros(len(x),dtype=np.float32)
    for col in range(x.shape[1]):square+=f[:,col]*f[:,col]
    return storage_half(f*(1/np.sqrt(square/np.float32(x.shape[1])+np.float32(1e-6)))[:,None]*gamma.astype(np.float32))


@lru_cache(maxsize=1)
def scalar_silu_reference_executable():
    # Independent scalar oracle: NumPy x/(1+exp(-x)) uses a different sequence
    # from FP32 EXP -> add -> reciprocal -> multiply. Previously actual `act`
    # supplied that rounding boundary; fused jobs intentionally elide its DMA.
    directory=tempfile.TemporaryDirectory()
    root=Path(directory.name);source=root/'silu.cc';exe=root/'silu'
    source.write_text('''#include <cmath>
#include <cstdint>
#include <iostream>
#include <sst/elements/golem/fp16.h>
using namespace SST::Golem;
int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    uint16_t bits;
    while (std::cin.read(reinterpret_cast<char*>(&bits),2)) {
        const float x=golem_fp16_to_float(bits);
        const float exponential=std::exp(-x);
        const float denominator=1.0f+exponential;
        const float reciprocal=1.0f/denominator;
        const float output=x*reciprocal;
        bits=golem_float_to_fp16(output);
        std::cout.write(reinterpret_cast<const char*>(&bits),2);
    }
    return std::cin.eof()?0:1;
}
''')
    subprocess.run(['g++','-std=c++17','-O2','-I',str(ROOT/'src'),str(source),'-o',str(exe)],check=True)
    return directory,exe


def scalar_silu_reference(x):
    _,exe=scalar_silu_reference_executable()
    data=np.asarray(x,dtype='<f2')
    output=subprocess.run([str(exe)],input=data.tobytes(),capture_output=True,check=True).stdout
    if len(output)!=data.nbytes:raise ValueError('SiLU scalar oracle size mismatch')
    return np.frombuffer(output,dtype='<f2').reshape(data.shape)


def attention_reference(x,qkv,hq,hkv,table):
    sys.path.insert(0,str(GUEST_DIR))
    from verify_fused_attention_scale_output import rotate_rope,compute_attention_blocked
    sequence=len(x)
    p=linear_reference(x,qkv).reshape(sequence,hq+2*hkv,64)
    output=np.empty((sequence,hq,64),dtype='<f2')
    for head in range(hq):
        kv=head//(hq//hkv)
        q=rotate_rope(p[:,head],sequence,64,table,simulator_storage=True)
        k=rotate_rope(p[:,hq+kv],sequence,64,table,simulator_storage=True)
        output[:,head]=storage_half(np.array(compute_attention_blocked(q,k,p[:,hq+hkv+kv],sequence,sequence,64,causal=True)).reshape(sequence,64))
    return output.reshape(sequence,-1)


def compare(got,want,tolerance,name):
    if got.shape!=want.shape or not got.size:raise ValueError(f'{name}: shape mismatch')
    error=np.abs(got.astype(np.float32)-want.astype(np.float32))
    maximum=float(error.max())
    if not np.isfinite(got).all() or not np.isfinite(want).all() or maximum>tolerance:
        raise ValueError(f'{name}: max_abs_error={maximum}, tolerance={tolerance}')
    return {'status':'PASS','max_abs_error':maximum,'tolerance':tolerance,'elements':int(got.size),'rms_error':float(np.sqrt(np.mean(error.astype(np.float64)**2)))}


def compare_normalized_chain(got,want,actual_input,reference_input,gamma):
    """Propagate the fixed layer-output budget through RMSNorm analytically.

    Attention's independent mathematical oracle is not bit-exact. A small
    bounded input error can grow when RMSNorm divides by a small RMS. Keep
    the input and local operator checks, and test the composed result against
    a Lipschitz bound rather than widening a shape-independent absolute limit.
    """
    delta=.004;local_budget=.002
    compare(actual_input,reference_input,delta,'final_norm_input_composed')
    local_expected=rms(actual_input,gamma)
    local=compare(got,local_expected,local_budget,'final_norm_local_contract')
    if got.shape!=want.shape or not np.isfinite(want).all():
        raise ValueError('final_norm_composed: nonfinite or inconsistent reference')
    x=actual_input.astype('f8');g=np.asarray(gamma,dtype='f8')
    if g.shape!=(x.shape[1],) or not np.isfinite(g).all():raise ValueError('invalid norm gamma')
    radius=np.sqrt(np.mean(x*x,axis=1,keepdims=True)+1e-6)
    minimum=np.maximum(radius-delta,1e-3)
    propagated=np.abs(g)*delta/minimum*(1+np.abs(x)/radius)
    # Two FP16 storage roundings, conservatively bounded by 2^-10 of the
    # stored magnitudes plus subnormal spacing; local FP32 effects are covered
    # by the independently checked local operator budget.
    rounding=2**-10*(np.abs(local_expected.astype('f8'))+np.abs(want.astype('f8')))+2**-23
    bound=propagated+rounding+local_budget
    error=np.abs(got.astype('f8')-want.astype('f8'))
    ratio=float(np.max(error/bound))
    if ratio>1:raise ValueError(f'final_norm_composed: propagated error budget exceeded: {ratio}')
    maximum=float(error.max())
    return {'status':'PASS','criterion':'bounded layer-input error propagated through RMSNorm plus storage/local budgets',
        'max_abs_error':maximum,'rms_error':float(np.sqrt(np.mean(error*error))),'elements':int(got.size),
        'input_absolute_budget':delta,'local_operator_budget':local_budget,'local_max_abs_error':local['max_abs_error'],
        'max_error_to_bound_ratio':ratio,'legacy_fixed_006_exceeded':maximum>.006}


def verify_numerics(case,m):
    hidden=m['hidden']; hq=m['num_query_heads'];hkv=m['num_kv_heads'];s=m['layout']['shared']
    sequence=m['sequence'];logical=m.get('logical_sequence',sequence)
    def read(offset,width):return read_bands(case,offset,width,rows=sequence//4)[:logical]
    x=np.fromfile(case/'projection_x.bin',dtype='<f2').reshape(sequence,hidden)[:logical]
    table=np.fromfile(case/'rope_64.bin',dtype='<f2')
    results={}; expected=x
    precision=x.astype(np.float32) if m['weight_kind'].startswith('dense') else None
    precision_layers={}
    # Compose all layers independently. Ping-pong keeps only the last two
    # outputs; retained-output mode checks each layer without extra copy DMA.
    for index,l in enumerate(m['layout']['layers'],1):
        weights={n:np.load(case/f'layer{index}_{n}.npy') for n in ('qkv','wo','gate','up','down')}
        gamma1=np.fromfile(case/f'layer{index}_gamma1.bin',dtype='<f2')
        gamma2=np.fromfile(case/f'layer{index}_gamma2.bin',dtype='<f2')
        o=attention_reference(rms(expected,gamma1),weights['qkv'],hq,hkv,table)
        h1=storage_half(expected.astype('f4')+linear_reference(o,weights['wo']).astype('f4'))
        norm=rms(h1,gamma2); gate=linear_reference(norm,weights['gate']);up=linear_reference(norm,weights['up'])
        g=gate.astype('f4');act=scalar_silu_reference(gate)
        gated=storage_half(act.astype('f4')*up.astype('f4'))
        expected=storage_half(linear_reference(gated,weights['down']).astype('f4')+h1.astype('f4'))
        if precision is not None:
            from prefill_precision_reference import layer as fp32_layer, error as precision_error
            precision=fp32_layer(precision,weights,gamma1,gamma2,hq,hkv,table)
        if m.get('elide_intermediate_output') and index<m['layers']:
            results[f'layer{index}_composed']={'status':'ELIDED','reason':'intermediate Y retained in timed GM; full independent chain composed through final Y'}
        elif m.get('retain_outputs') or index>=m['layers']-1:
            got=read(l['output'],hidden)
            results[f'layer{index}_composed']=compare(got,expected,4e-3,f'layer{index}_composed')
            if precision is not None:precision_layers[str(index)]=precision_error(got,precision)
        else:
            results[f'layer{index}_composed']={'status':'OVERWRITTEN','reason':'ping-pong output reused; full chain still composed independently'}
    if precision is not None:results['fp32_precision_by_layer']=precision_layers
    if m.get('final_norm'):
        final=m['layout']['final_norm'];gamma=np.fromfile(case/'final_gamma.bin',dtype='<f2')
        got=read(final['output'],hidden)
        final_expected=rms(expected,gamma)
        results['final_norm_composed']=compare_normalized_chain(got,final_expected,
            read(m['layout']['layers'][-1]['output'],hidden),expected,gamma)
        if precision is not None:
            from prefill_precision_reference import rms as fp32_rms
            precision_final=fp32_rms(precision,gamma)
            results['fp32_final_norm']=precision_error(got,precision_final)
            if m.get('lm_head'):np.save(case/'fp32_final_reference_last_token.npy',precision_final[-1:])
        if m.get('lm_head'):np.save(case/'final_reference_last_token.npy',final_expected[-1:])
        results['final_norm_local']=compare(got,rms(read(m['layout']['layers'][-1]['output'],hidden),gamma),2e-3,'final_norm_local')
    # The last layer's scratch survives. Check its nine stages against device
    # inputs so errors cannot cancel between operators or across layers.
    l=m['layout']['layers'][-1]; index=m['layers']
    from verification_retained_input import last_layer_input
    x,input_proof=last_layer_input(case,m);x=x[:logical]
    if m.get('pv_wo_gm_handoff'):
        from verification_retained_input import retained_attention
        o,proof=retained_attention(case,m);o=o[:logical]
        results['last_attention_handoff_reference']=proof
    else:o=read(s['o'],hidden)
    results['last_layer_input_reference']=input_proof
    attention_check=compare(o,attention_reference(rms(x,gamma1),weights['qkv'],hq,hkv,table),3e-4,'last_attention')
    results['last_attention']=(dict(status='ELIDED',reason='O consumed from timed GM; candidate final Y independently checked',
        materialized_reference_check=attention_check) if m.get('pv_wo_gm_handoff') else attention_check)
    actual={}
    def check(name,want,tolerance=0):
        if m.get('gm_stage_chain') and sequence<=1024 and name in ('wo','h1','norm2'):
            actual[name]=want
            results['last_'+name]={'status':'ELIDED','reason':'consumed from timed GM stage chain; reconstructed independently; complete device Y still checked'}
        else:
            actual[name]=read(s[name],want.shape[1])
            results['last_'+name]=compare(actual[name],want,tolerance,name)
    check('wo',linear_reference(o,weights['wo']))
    check('h1',storage_half(x.astype('f4')+actual['wo'].astype('f4')))
    check('norm2',rms(actual['h1'],gamma2),2e-3)
    if m.get('ffn_panel_fusion'):
        for name in ('gate','up'):
            actual[name]=linear_reference(actual['norm2'],weights[name])
            results['last_'+name]={'status':'ELIDED','reason':'producer panel consumed on chip; independently reconstructed for gated validation'}
    else:
        check('gate',linear_reference(actual['norm2'],weights['gate']))
        check('up',linear_reference(actual['norm2'],weights['up']))
    g=actual['gate'].astype('f4');act_expected=storage_half(g/(np.float32(1)+np.exp(-g)))
    if m.get('silu_multiply_fusion') or m.get('ffn_panel_fusion'):
        actual['act']=scalar_silu_reference(actual['gate'])
        compare(actual['act'],act_expected,5e-4,'fused_SiLU_scalar_vs_numpy')
        results['last_act']={'status':'ELIDED','reason':'FP16-rounded bounded SiLU result consumed directly by Multiply'}
    else:check('act',act_expected,5e-4)
    if m.get('ffn_down_fusion'):
        actual['gated']=storage_half(actual['act'].astype('f4')*actual['up'].astype('f4'))
        results['last_gated']={'status':'ELIDED','reason':'gated panel consumed directly by down; independently reconstructed for down validation'}
    else:check('gated',storage_half(actual['act'].astype('f4')*actual['up'].astype('f4')))
    if m.get('down_residual_fusion'):
        actual['down']=linear_reference(actual['gated'],weights['down'])
        results['last_down']={'status':'ELIDED','reason':'FP16 down output consumed directly by residual Add; independently reconstructed'}
    else:check('down',linear_reference(actual['gated'],weights['down']))
    results['last_residual2']=compare(read(l['output'],hidden),
        storage_half(actual['down'].astype('f4')+actual['h1'].astype('f4')),0,'last_residual2')
    for node in range(1,5):
        with (case/'hbm'/f'hbm_out_node{node}.bin').open('rb') as stream:
            for flag in ('projection_done','decoder_done'):
                stream.seek(s[flag]); values=np.frombuffer(stream.read(32),dtype='<u8')
                if values.size!=4 or not np.all(values==m['layers']):raise ValueError('stale final completion flags')
            if m.get('final_norm'):
                stream.seek(m['layout']['final_norm']['done']);values=np.frombuffer(stream.read(32),dtype='<u8')
                if values.size!=4 or not np.all(values==m['layers']+1):raise ValueError('stale final RMSNorm completion flags')
    return results


def fields(line):
    return {k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)}


def verify_epoch_resets(group,flags,epoch,epoch_flags):
    required=not epoch_flags or epoch==1
    if (len(group)!=(len(flags) if required else 0) or
        {e['address'] for e in group}!=(flags if required else set()) or
        any(e['value'] or e['status'] for e in group)):
        raise ValueError('flag reset not ACKed before projection release')


def verify_lifecycle(case,m):
    epochs=range(1,m['layers']+1);jobs=16*m['layers'];groups=m['sequence']//1024
    provenance=case/'decoder_stack_provenance.json'
    env=json.loads(provenance.read_text()).get('environment',{}) if provenance.exists() else {}
    epoch_flags=env.get('GOLEM_DECODER_EPOCH_FLAGS','0')=='1'
    log=next((case/'logs').glob('attention-sst*.log'))
    accepts=[]; ops=[];resets=[];scratch=[];ffn_flow=[];down_flow=[]
    with log.open() as stream:
        for line in stream:
            if '[CONTROL_EVENT] stage=projection_accept ' in line:accepts.append(fields(line))
            if '[DECODER_OP] event=done ' in line:ops.append(fields(line))
            if '[DECODER_SYNC] event=flag_write_ack ' in line:resets.append(fields(line))
            if '[DECODER_SYNC] event=scratch_retire ' in line:scratch.append(fields(line))
            if '[DECODER_FFN_FLOW]' in line:ffn_flow.append(fields(line))
            if '[DECODER_DOWN_FLOW]' in line:down_flow.append(fields(line))
    if m.get('final_norm'):
        final_flags={node*m['layout']['node_stride']+m['layout']['final_norm']['done']+lane*8 for node in range(1,5) for lane in range(4)}
        if m.get('lm_head'):
            final_flags.update(node*m['layout']['node_stride']+m['layout']['lm_head']['done']+lane*8 for node in range(1,5) for lane in range(4))
        if m.get('embedding'):
            final_flags.update(node*m['layout']['node_stride']+m['layout']['embedding']['done']+lane*8 for node in range(1,5) for lane in range(4))
        ops=[e for e in ops if e['stage']!=10 and not (m.get('lm_head') and e['stage']==11) and not (m.get('embedding') and e['stage']==12)]
        resets=[e for e in resets if e['address'] not in final_flags]
    elided={5,6,7} if m.get('ffn_panel_fusion') else ({6} if m.get('silu_multiply_fusion') else set())
    if m.get('ffn_down_fusion'):elided.add(8)
    if m.get('down_residual_fusion'):elided.add(9)
    if m.get('add_norm_fusion'):elided.add(3)
    expected_stages=[stage for stage in range(1,10) if stage not in elided]
    if len(accepts)!=jobs or len(ops)!=jobs*len(expected_stages) or len(resets)!=(32 if epoch_flags else 2*jobs) or len(scratch)!=2*jobs:raise ValueError('missing or duplicate layer jobs/reset ACKs')
    if m.get('ffn_panel_fusion'):
        panel_tiles=m['ffn_panel_tiles'];panels=(4*m['hidden']//64+panel_tiles-1)//panel_tiles
        expected={(core,epoch) for core in range(4,20) for epoch in epochs}
        if len(ffn_flow)!=jobs or {(e['core'],e['epoch']) for e in ffn_flow}!=expected:
            raise ValueError('missing/duplicate fused FFN worker generation')
        if any(e['panels_produced']!=panels*groups or e['panels_consumed']!=panels*groups or e['gm_required_bytes']>2*2**20 for e in ffn_flow):
            raise ValueError('fused FFN panel lifecycle or GM capacity mismatch')
        if any(e['op']!=(9 if m.get('ffn_down_fusion') else 8) for e in ops if e['stage']==4):raise ValueError('fused FFN descriptor did not execute')
        if m.get('ffn_down_fusion'):
            partial_bytes=64*m['hidden']*2;passes=4*m['hidden']//64 if m.get('down_input_reuse') else panels;movement=(passes-1)*partial_bytes*groups
            if len(down_flow)!=jobs or {(e['core'],e['epoch']) for e in down_flow}!=expected:
                raise ValueError('missing/duplicate fused down worker generation')
            if any(e['panels_consumed']!=panels*groups or e['cbuffer_bytes']!=partial_bytes or
                   e['cbuffer_read_bytes']!=movement or e['cbuffer_write_bytes']!=movement or
                   (e['restores']!=(passes-1)*(m['hidden']//64)*groups and
                    not (os.environ.get('GOLEM_ALL_GEMM_REUSE_WINDOW')=='1' and e['restores']==0))
                   for e in down_flow):
                raise ValueError('fused down partial storage/restore lifecycle mismatch')
    begin={};end={};stages={};s=m['layout']['shared'];stride=m['layout']['node_stride']
    flags={node*stride+s[name]+lane*8 for node in range(1,5) for name in ('projection_done','decoder_done') for lane in range(4)}
    for epoch in epochs:
        group=[e for e in accepts if e['epoch']==epoch]
        if {e['core'] for e in group}!=set(range(4,20)) or len(group)!=16:raise ValueError('projection worker generation mismatch')
        for e in group:
            node=1+(e['core']-4)%4;l=m['layout']['layers'][epoch-1]
            if e['input']!=node*stride+l['input'] or e['weights']!=node*stride+l['qkv_weight']:
                raise ValueError('layer input or weight address mismatch')
        begin[epoch]=min(e['tick'] for e in group)
        stage_summary={}
        for stage in expected_stages:
            group=[e for e in ops if e['epoch']==epoch and e['stage']==stage]
            if len(group)!=16 or {e['core'] for e in group}!=set(range(4,20)) or any(e['status'] for e in group):
                raise ValueError('incomplete/failed decoder stages')
            stage_summary[str(stage)]={'start_tick':min(e['start_tick'] for e in group),'end_tick':max(e['tick'] for e in group),
                'span_cycles':(max(e['tick'] for e in group)-min(e['start_tick'] for e in group))/1000,
                'workers':16,'hbm_read_bytes':sum(e['hbm_read'] for e in group),'hbm_write_bytes':sum(e['hbm_write'] for e in group)}
        stages[str(epoch)]=stage_summary;end[epoch]=stage_summary['4' if m.get('down_residual_fusion') else '9']['end_tick']
        group=[e for e in resets if (epoch==1 or e['tick']>=end[epoch-1]) and e['tick']<begin[epoch]]
        verify_epoch_resets(group,flags,epoch,epoch_flags)
    for core in range(4,20):
        retires=sorted((e for e in scratch if e['core']==core),key=lambda e:e['tick'])
        if len(retires)!=2*m['layers']:raise ValueError('missing worker scratch ownership transitions')
        for epoch in epochs:
            accepted=next(e for e in accepts if e['core']==core and e['epoch']==epoch)
            wo=next(e for e in ops if e['core']==core and e['epoch']==epoch and e['stage']==1)
            if not (retires[2*epoch-2]['tick']<=accepted['tick']<retires[2*epoch-1]['tick']<=wo['start_tick']):
                raise ValueError('operand cache invalidation order mismatch')
    if any(begin[n]<=end[n-1] for n in epochs if n>1):raise ValueError('next layer started before previous layer completion')
    stdout='\n'.join(p.read_text() for p in (case/'stdout').rglob('stdout-*'))
    completed=re.findall(r'DECODER_LAYER status=(\d+) stage=4 hidden=\d+ intermediate=\d+ layer=(\d+)',stdout)
    if sorted(completed,key=lambda e:int(e[1]))!=[('0',str(n)) for n in epochs]:raise ValueError('root completion missing/failed')
    attention=re.findall(r'FUSED_ATTENTION status=(\d+) job=\d+ manager=(\d+)',stdout)
    if len(attention)!=4*m['layers'] or any(status!='0' for status,manager in attention):raise ValueError('manager Attention completion failed')
    # Reuse established event patterns but verify each layer separately. SFU
    # totals also contain FFN SiLU, so global single-Attention counters cannot
    # serve as a multilayer acceptance criterion.
    from report_sst import QK,SFU,PV,E2E
    text=log.read_text();cluster={}
    expected_jobs=4*m['num_kv_heads']*2
    expected_pv=m['num_query_heads']*sum((block*64+255)//256 for block in range(1,m['sequence']//64+1))
    blocks=m['sequence']//64
    expected_tiles=m['num_query_heads']*blocks*(blocks+1)//2
    for epoch in epochs:
        low=begin[epoch]/1000;high=end[epoch]/1000
        qk=[e for e in QK.findall(text) if low<=int(e[2])<=high]
        sfu=[e for e in SFU.findall(text) if low<=int(e[1])<=high]
        pv=[e for e in PV.findall(text) if low<=int(e[5])<=high]
        done=[e for e in E2E.findall(text) if low<=int(e[1])<=high]
        if (len(qk)!=expected_jobs or len(sfu)!=expected_jobs or len(done)!=expected_jobs or
            len(pv)!=expected_pv or sum(int(e[4]) for e in qk)!=expected_tiles or
            any(e[4]!=e[5] for e in qk) or {int(e[0]) for e in qk}!=set(range(4,12)) or
            {int(e[0]) for e in pv}!=set(range(12,20))):
            raise ValueError(f'layer {epoch} Attention lifecycle incomplete: QK={len(qk)} SFU={len(sfu)} PV={len(pv)} E2E={len(done)}')
        roots=[fields(line) for line in text.splitlines() if '[ATTENTION_MILESTONE] stage=root_descriptor_accept ' in line and
               begin[epoch]<=fields(line)['sst_tick']<=end[epoch]]
        tags={e['tag'] for e in roots}
        if len(roots)!=m['num_kv_heads'] or len(tags)!=m['num_kv_heads']:raise ValueError('Attention root tags missing/duplicated')
        cluster[str(epoch)]={'status':'PASS','qk_jobs':len(qk),'sfu_jobs':len(sfu),'pv_windows':len(pv),
                            'completed_jobs':len(done),'qk_tiles':expected_tiles,'root_tags':sorted(tags),
                            'first_root_descriptor_tick':min(e['sst_tick'] for e in roots)}
    tags=[tag for entry in cluster.values() for tag in entry['root_tags']]
    if len(set(tags))!=len(tags):raise ValueError('layer Attention tag reused')
    (case/'decoder_stack_attention.json').write_text(json.dumps(cluster,indent=2)+'\n')
    return log,begin,end,{'status':'PASS','projection_jobs':jobs,'decoder_ops':len(ops),'reset_ack_count':len(resets),'epoch_flags':epoch_flags,
        'layers':m['layers'],'scratch_retire_count':2*jobs,'distinct_attention_tags':True,'all_attention_jobs_checked':sum(e['completed_jobs'] for e in cluster.values()),'stage_spans':stages,
        'ffn_panel_flow':ffn_flow,'ffn_down_flow':down_flow,
        'attention_start_ticks':{epoch:entry['first_root_descriptor_tick'] for epoch,entry in cluster.items()}}


def verify_dma(log,m,begin,end):
    from report_dma_e2e import contained
    from analyze_resource_timeline import FIELDS,merged
    epochs=range(1,m['layers']+1)
    stride=m['layout']['node_stride'];qbytes=(m['sequence']//4)*m['hidden']*2;s=m['layout']['shared']
    layers={epoch:{'first_input':None,'last_o':0,'last_y':0,'o_writes':[],'y_writes':[],'o_bytes':0,'y_bytes':0,'o_read_bytes':0,'landing_writes':[],'landing_bytes':0,'last_landing':0} for epoch in epochs}
    pv_wo=bool(m.get('pv_wo_gm_handoff',False))
    elide_intermediate=bool(m.get('elide_intermediate_output',False))
    ranges=lambda offset:[(node*stride+offset,node*stride+offset+qbytes) for node in range(1,5)]
    resident=m.get('layer_residency',False) and m.get('sequence',1024)<=1024
    with log.with_suffix('.resources.jsonl').open() as stream:
        for line in stream:
            if ('resource=dma_read_transport ' not in line and 'resource=dma_write_transport ' not in line
                and not (resident and 'resource=local_read ' in line)
                and not (pv_wo and 'resource=local_write ' in line)):continue
            e={k:int(v) for k,v in FIELDS.findall(line) if k not in ('domain','resource')}
            epoch=next((n for n in epochs if begin[n]<=e['start_tick']<=end[n]),None)
            if epoch is None:continue
            state=layers[epoch];l=m['layout']['layers'][epoch-1]
            local_input=(resident and epoch>1 and 'resource=local_read ' in line and 4<=e['core']<20 and
                         e['core']*0x200000+0xb0000<=e['address'] and
                         e['address']+e['bytes']<=e['core']*0x200000+0xb0000+64*m['hidden']*2)
            hbm_input=(not (resident and epoch>1) and 'resource=dma_read_transport ' in line and
                       contained(e['address'],e['bytes'],ranges(l['input'])))
            if local_input or hbm_input:
                if e['ok']!=1:raise ValueError('layer input read failed')
                state['first_input']=e['start_tick'] if state['first_input'] is None else min(state['first_input'],e['start_tick'])
            if pv_wo and 'resource=dma_read_transport ' in line and contained(e['address'],e['bytes'],ranges(s['o'])):
                state['o_read_bytes']+=e['bytes']
            if pv_wo and 'resource=local_write ' in line and 4<=e['core']<20:
                landing=e['core']*0x200000+0x1a0000
                if landing<=e['address'] and e['address']+e['bytes']<=landing+64*2048*2:
                    # Projection and FFN reuse this scratch in other phases.
                    # Only Attention owns PV landing between root acceptance and Wo.
                    wo_start=m['_lifecycle']['stage_spans'][str(epoch)]['1']['start_tick']
                    if m['_lifecycle']['attention_start_ticks'][str(epoch)]<=e['start_tick']<wo_start:
                        if e['ok']!=1:raise ValueError('PV-Wo GM landing failure')
                        state['landing_writes'].append((e['address'],e['address']+e['bytes']))
                        state['landing_bytes']+=e['bytes'];state['last_landing']=max(state['last_landing'],e['done_tick'])
            if 'resource=dma_write_transport ' in line:
                for key,offset in [('o',s['o']),('y',l['output'])]:
                    if contained(e['address'],e['bytes'],ranges(offset)):
                        if e['ok']!=1:raise ValueError('output DMA failure')
                        state[key+'_writes'].append((e['address'],e['address']+e['bytes']))
                        state[key+'_bytes']+=e['bytes'];state['last_'+key]=max(state['last_'+key],e['done_tick'])
    result={}
    for epoch,state in layers.items():
        if state['first_input'] is None:raise ValueError('no layer input DMA')
        for key,offset in [('o',s['o']),('y',m['layout']['layers'][epoch-1]['output'])]:
            if key=='o' and pv_wo:
                expected=[(core*0x200000+0x1a0000,core*0x200000+0x1e0000) for core in range(4,20)]
                if state['o_bytes'] or state['o_read_bytes'] or merged(state['landing_writes'])!=merged(expected) or state['landing_bytes']!=4*qbytes:
                    raise ValueError('PV-Wo must cover all timed GM landings with zero O HBM reads/writes')
                state['last_o']=state['last_landing']
                continue
            if key=='y' and elide_intermediate and epoch<m['layers']:
                if state[key+'_writes'] or state[key+'_bytes']:
                    raise ValueError('elided intermediate layer output still has HBM writes')
                state['last_y']=m['_lifecycle']['stage_spans'][str(epoch)]['4' if m.get('down_residual_fusion') else '9']['end_tick']
                continue
            if merged(state[key+'_writes'])!=merged(ranges(offset)) or state[key+'_bytes']!=4*qbytes:
                raise ValueError('incomplete or duplicate layer output ACK coverage')
        if state['last_o']>min(v['start_tick'] for v in m['_lifecycle']['stage_spans'][str(epoch)].values()):
            raise ValueError('Wo started before all Attention O ACKs')
        if state['first_input']>=state['last_o']:raise ValueError('layer input must precede attention output')
        result[str(epoch)]={'first_input_dma_tick':state['first_input'],'input_source':'GM' if resident and epoch>1 else 'HBM','last_attention_o_ack_tick':state['last_o'],
            'attention_o_ready_source':'GM landing ACK' if pv_wo else 'HBM output ACK',
            'attention_o_hbm_write_bytes':state['o_bytes'],'attention_o_hbm_read_bytes':state['o_read_bytes'],
            'pv_wo_gm_landing_bytes':state['landing_bytes'],
            'last_y_ack_tick':state['last_y'],'y_ready_source':'GM publication and completion ACK' if elide_intermediate and epoch<m['layers'] else 'HBM output ACK','attention_cycles':(state['last_o']-state['first_input'])/1000,
            'post_attention_cycles':(state['last_y']-state['last_o'])/1000,
            'layer_cycles':(state['last_y']-state['first_input'])/1000,'output_ack_bytes':state['y_bytes']}
    if any(result[str(n)]['first_input_dma_tick']<=result[str(n-1)]['last_y_ack_tick'] for n in epochs if n>1):raise ValueError('layer input read before previous Y ACK')
    return {'contract':'first layer X input DMA send to final layer Y final DMA ACK',
        'ticks_per_cycle':1000,'layers':result,
        'layer_handoff_cycles':sum((result[str(n)]['first_input_dma_tick']-result[str(n-1)]['last_y_ack_tick'])/1000 for n in epochs if n>1),
        'end_to_end_cycles':(result[str(m['layers'])]['last_y_ack_tick']-result['1']['first_input_dma_tick'])/1000,
        'note':'Stage spans overlap; cannot be summed. Includes handoff/reset between layers; excludes host preload/startup and final exit.'}


def verify_final_norm(case,m,log,end,timing):
    from analyze_resource_timeline import FIELDS,merged
    final=m['layout']['final_norm'];stride=m['layout']['node_stride'];hidden=m['hidden']
    groups=m.get('sequence',1024)//1024
    row_bytes=64*hidden*2;epoch=m['layers']+1
    def ranges(node,lane,offset):
        return [(node*stride+offset+(block*256+lane*64)*hidden*2,node*stride+offset+(block*256+lane*64+64)*hidden*2) for block in range(groups)]
    flags={node*stride+final['done']+lane*8 for node in range(1,5) for lane in range(4)}
    ops=[];resets=[];last_decoder=[]
    with log.open() as stream:
        for line in stream:
            if '[DECODER_OP] event=done ' in line:
                e=fields(line)
                if e['stage']==10:ops.append(e)
                elif e['epoch']==m['layers'] and e['stage'] in (1,4):last_decoder.append(e)
            if '[DECODER_SYNC] event=flag_write_ack ' in line:
                e=fields(line)
                if e['address'] in flags:resets.append(e)
    if (len(ops)!=16 or {e['core'] for e in ops}!=set(range(4,20)) or
        any(e['status'] or e['op']!=3 or e['epoch']!=epoch for e in ops)):
        raise ValueError('missing/duplicate/failed final RMSNorm worker generation')
    start=min(e['start_tick'] for e in ops)
    eager=bool(m.get('eager_final_norm',False))
    if eager:
        producers={e['core']:e for e in last_decoder if e['stage']==4}
        if (len(producers)!=16 or len(last_decoder)!=32 or
                any(e['status'] for e in last_decoder) or
                max(e['tick'] for e in producers.values())!=end[m['layers']] or
                any(e['start_tick']<=producers[e['core']]['tick'] for e in ops)):
            raise ValueError('eager final Norm must follow each worker complete FFN/output ACK')
        reset_begin=end.get(m['layers']-1,0)
        reset_end=min(e['start_tick'] for e in last_decoder if e['stage']==1)
    else:reset_begin=end[m['layers']];reset_end=start
    if (len(resets)!=16 or {e['address'] for e in resets}!=flags or
        any(e['status'] or e['value'] or not reset_begin<=e['tick']<reset_end for e in resets)):
        raise ValueError('final RMSNorm flag reset ACK/barrier failed')
    states={e['core']:{'op':e,'reads':[],'writes':[],'read_bytes':0,'write_bytes':0,
        'gamma_bytes':0,'gamma_reads':[],'last_output':0,'flag_acks':[]} for e in ops}
    provenance_path=case/'decoder_stack_provenance.json'
    recorded_env=json.loads(provenance_path.read_text())['environment'] if provenance_path.exists() else {}
    if eager!=(recorded_env.get('GOLEM_DECODER_EAGER_FINAL_NORM','0')=='1'):
        raise ValueError('eager final Norm manifest/configuration mismatch')
    gamma_reuse=recorded_env.get('GOLEM_DECODER_GAMMA_REUSE','0')=='1'
    resident=recorded_env.get('GOLEM_DECODER_LAYER_RESIDENCY','0')=='1' and m.get('sequence',1024)<=1024
    for state in states.values():state.update(local_reads=[],local_bytes=0)
    first_input=None
    with log.with_suffix('.resources.jsonl').open() as stream:
        for line in stream:
            if ('resource=dma_read_transport ' not in line and 'resource=dma_write_transport ' not in line
                and not (resident and 'resource=local_read ' in line)):continue
            e={k:int(v) for k,v in FIELDS.findall(line) if k not in ('domain','resource')}
            if e['core'] not in states:continue
            state=states[e['core']];op=state['op']
            if not op['start_tick']<=e['start_tick']<=op['tick']:continue
            node=1+(e['core']-4)%4;lane=(e['core']-4)//4
            input_ranges=ranges(node,lane,m['layout']['layers'][-1]['output'])
            output_ranges=ranges(node,lane,final['output'])
            address=e['address'];stop=address+e['bytes']
            if 'resource=local_read ' in line:
                base=e['core']*0x200000+0xb0000
                if base<=address<stop<=base+row_bytes:
                    ready=producers[e['core']]['tick'] if eager else timing['layers'][str(m['layers'])]['last_y_ack_tick']
                    if e['ok']!=1 or e['start_tick']<=ready:
                        raise ValueError('final RMSNorm GM read before final layer Y ACK')
                    state['local_reads'].append((address,stop));state['local_bytes']+=e['bytes']
                    first_input=e['start_tick'] if first_input is None else min(first_input,e['start_tick'])
            elif 'resource=dma_read_transport ' in line:
                if any(a<=address<stop<=b for a,b in input_ranges):
                    ready=producers[e['core']]['tick'] if eager else timing['layers'][str(m['layers'])]['last_y_ack_tick']
                    if e['ok']!=1 or e['start_tick']<=ready:
                        raise ValueError('final RMSNorm input read before final layer Y ACK')
                    state['reads'].append((address,stop));state['read_bytes']+=e['bytes']
                    first_input=e['start_tick'] if first_input is None else min(first_input,e['start_tick'])
                elif node*stride+final['gamma']<=address<stop<=node*stride+final['gamma']+hidden*2:
                    if e['ok']!=1:raise ValueError('final gamma DMA failure')
                    state['gamma_bytes']+=e['bytes']
                    state['gamma_reads'].append((address,stop))
                else:raise ValueError('unexpected final RMSNorm input DMA address')
            else:
                if any(a<=address<stop<=b for a,b in output_ranges):
                    if e['ok']!=1:raise ValueError('final RMSNorm output DMA failure')
                    state['writes'].append((address,stop));state['write_bytes']+=e['bytes']
                    state['last_output']=max(state['last_output'],e['done_tick'])
                elif address==node*stride+final['done']+lane*8 and e['bytes']==8:
                    if e['ok']!=1:raise ValueError('final RMSNorm completion DMA failure')
                    state['flag_acks'].append(e)
                else:raise ValueError('unexpected final RMSNorm output DMA address')
    for core,state in states.items():
        node=1+(core-4)%4;lane=(core-4)//4;op=state['op']
        for key,offset in [('reads',m['layout']['layers'][-1]['output']),('writes',final['output'])]:
            expected=[] if resident and key=='reads' else merged(ranges(node,lane,offset))
            if merged(state[key])!=expected:raise ValueError('final RMSNorm DMA coverage incomplete')
        if resident and (merged(state['local_reads'])!=[(core*0x200000+0xb0000,core*0x200000+0xb0000+row_bytes)] or state['local_bytes']!=row_bytes):
            raise ValueError('final RMSNorm retained GM coverage incomplete/duplicate')
        gamma_bytes=hidden*2 if gamma_reuse else ((64*hidden+8191)//8192)*hidden*2*groups
        gamma_start=node*stride+final['gamma']
        if merged(state['gamma_reads'])!=[(gamma_start,gamma_start+hidden*2)]:
            raise ValueError('final RMSNorm gamma DMA coverage incomplete')
        input_bytes=0 if resident else row_bytes*groups
        if (state['read_bytes']!=input_bytes or state['write_bytes']!=row_bytes*groups or state['gamma_bytes']!=gamma_bytes or
            op['hbm_read']!=input_bytes+gamma_bytes or op['hbm_write']!=row_bytes*groups+8):
            raise ValueError('final RMSNorm duplicate/missing data or gamma DMA')
        if (len(state['flag_acks'])!=1 or state['flag_acks'][0]['start_tick']<state['last_output'] or
            not state['last_output']<=state['flag_acks'][0]['done_tick']<=op['tick']):
            raise ValueError('final RMSNorm completed before output/flag DMA ACK')
    stdout='\n'.join(p.read_text() for p in (case/'stdout').rglob('stdout-*'))
    if re.findall(r'DECODER_FINAL_NORM status=(\d+) hidden=(\d+) epoch=(\d+)',stdout)!=[('0',str(hidden),str(epoch))]:
        raise ValueError('final RMSNorm root completion missing/failed')
    last_output=max(s['last_output'] for s in states.values())
    return {'status':'PASS','workers':16,'epoch':epoch,'reset_ack_count':16,'eager_worker_handoff':eager,
        'overlap_with_decoder_tail_cycles':max(0,(end[m['layers']]-start)/1000) if eager else 0,
        'first_input_dma_tick':first_input,'input_source':'GM' if resident else 'HBM','last_output_ack_tick':last_output,
        'last_completion_ack_tick':max(s['flag_acks'][0]['done_tick'] for s in states.values()),
        'worker_output_ack_ticks':{str(core):s['last_output'] for core,s in states.items()},
        'output_ack_bytes':16*row_bytes*groups,'rmsnorm_cycles':(last_output-first_input)/1000,
        'handoff_cycles':(first_input-timing['layers'][str(m['layers'])]['last_y_ack_tick'])/1000,
        'tail_cycles':(last_output-timing['layers'][str(m['layers'])]['last_y_ack_tick'])/1000,
        'end_to_end_cycles':(last_output-timing['layers']['1']['first_input_dma_tick'])/1000,
        'contract':'first layer X input DMA send to final RMSNorm output DMA ACK',
        'epsilon':1e-6,'io_dtype':'fp16','internal_dtype':'fp32'}


def validate_inputs(case):
    manifest=json.loads((case/'decoder_stack_inputs.json').read_text())
    expected=memory_layout(manifest['num_query_heads'],manifest['num_kv_heads'],manifest['layers'],sequence=manifest['sequence'],retain_outputs=manifest.get('retain_outputs',False),final_norm=manifest.get('final_norm',False),
        vocab=manifest['layout'].get('lm_head',{}).get('vocab',0),node_bytes=manifest['layout']['node_stride'],embedding=manifest.get('embedding',False))
    if manifest['layout']!=expected:raise ValueError('manifest differs from authoritative layout')
    logical=manifest.get('logical_sequence',manifest['sequence'])
    if not 1<=logical<=manifest['sequence'] or (logical+1023)//1024*1024!=manifest['sequence']:
        raise ValueError('invalid logical/padded sequence contract')
    for name,expected_digest in manifest.get('oracle_sha256',{}).items():
        if Path(name).name!=name or digest(case/name)!=expected_digest:raise ValueError('oracle provenance mismatch')
    for entry in manifest['preloads']:
        path=Path(entry['path'])
        if path.stat().st_size!=entry['bytes'] or digest(path)!=entry['sha256']:raise ValueError('input provenance mismatch')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root',type=Path,required=True)
    p.add_argument('--gm-response-cycles',type=int,choices=(1,2,4),default=1)
    p.add_argument('--gm-model',choices=('legacy','banked'),default='banked',help='local GM timing contract; banked is reference-aligned candidate')
    p.add_argument('--shape',choices=('small','formal'),default='small')
    p.add_argument('--layers',type=int,default=2,help='decoder layers, 1..32 (target model: 16)')
    p.add_argument('--sequence',type=int,default=1024,help='logical token count, 1..8192; existing kernels pad to 1024')
    p.add_argument('--weights',choices=('sparse','dense'),default='sparse',help='synthetic numerical stress; all dense hardware tiles always execute')
    p.add_argument('--retain-layer-outputs',action='store_true',help='validation layout: retain each layer activation in existing HBM')
    p.add_argument('--host-hbm-root',type=Path,help='host backing directory, e.g. tmpfs; simulated capacities/timing unchanged')
    p.add_argument('--cpu-set',help='default: one physical core per rank, starting at core 0')
    p.add_argument('--mpi-ranks',type=int,choices=(2,4,8),default=4)
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--validate-inputs',action='store_true')
    p.add_argument('--final-norm',action='store_true',help='append independent final RMSNorm after both layers')
    p.add_argument('--lm-head',action='store_true',help='append final norm and last-token LM head')
    p.add_argument('--vocab',type=int,help='default: 8201 for small, 128256 for formal')
    p.add_argument('--node-mib',type=int,help='HBM capacity per node; default 2048 for LM head, otherwise 128')
    p.add_argument('--embedding',action='store_true',help='on-architecture token lookup before the complete prefill path')
    p.add_argument('--output-panel-tiles',type=int,choices=range(17),default=16)
    p.add_argument('--output-write-window',type=int,choices=range(1,17),default=8)
    p.add_argument('--gemm-prefetch',action='store_true')
    p.add_argument('--prefetch-slots',type=int,choices=(2,4,8),default=2)
    p.add_argument('--weight-trace',action='store_true')
    p.add_argument('--silu-multiply-fusion',action='store_true')
    p.add_argument('--ffn-panel-fusion',action='store_true',help='gate/up producer panels feed shared SFU, then write gated for the unchanged down job')
    p.add_argument('--ffn-down-fusion',action='store_true',help='consume gated panels directly with down; preserve K-order partials in existing C-buffer')
    p.add_argument('--down-residual-fusion',action='store_true',help='stream final down panels through existing SFU Add; omit down HBM intermediate')
    p.add_argument('--ffn-panel-tiles',type=int,choices=range(1,17),default=16)
    p.add_argument('--ffn-layout',type=int,choices=(0,1,2),default=0,help='software intermediate layout; 1 retains tile-major FFN, 2 also scatters final down output directly')
    p.add_argument('--down-input-reuse',action='store_true',help='K-major down loop reuses existing array inputs; spills only to existing timed partial SRAM')
    p.add_argument('--ffn-array-groups',type=int,choices=(1,2),default=1,
                   help='software interleave two disjoint 32-array subsets for fused FFN; hardware unchanged')
    p.add_argument('--ffn-gm-lookahead',type=int,choices=(0,1,2,4),default=0,help='fused FFN GM read depth; 0 follows the shared GEMM setting')
    p.add_argument('--ffn-demand-order',action='store_true',help='submit FFN GM weight reads in consumption order using the existing slots')
    p.add_argument('--pv-wo-gm-handoff',action='store_true',help='S=1024 formal PV output transfers over NoC into timed worker GM')
    p.add_argument('--pv-wo-direct-input',action='store_true',help='consume leased PV landing directly using disjoint existing Wo B scratch')
    p.add_argument('--gm-stage-chain',action='store_true',help='software residency across Wo/Add/Norm2/FFN in existing GM')
    p.add_argument('--add-norm-fusion',action='store_true',help='forward rounded Add result directly to RMSNorm2')
    p.add_argument('--eager-final-norm',action='store_true',help='overlap per-worker final Norm with the last decoder tail')
    p.add_argument('--final-norm-lm-stream',action='store_true',help='last-token producer notifies LM workers through existing timed remote stores')
    p.add_argument('--tail-schedule-experiment',action='store_true',help='build opt-in tail guest, including its stream-off paired control')
    p.add_argument('--final-norm-producer-first',action='store_true',help='prioritize last-token producer Norm job before other Norm workers')
    p.add_argument('--layer-residency',action='store_true',help='reuse existing GM Y in next Norm1/final Norm')
    p.add_argument('--input-residency',action='store_true',help='retain layer input in existing GM for residual Add')
    p.add_argument('--lm-input-residency',action='store_true',help='reuse GM last-token input across adjacent LM descriptors')
    p.add_argument('--elide-intermediate-output',action='store_true',help='elide non-final layer output HBM mirror when GM stage chain owns the consumer')
    p.add_argument('--lm-n-parallel',type=int,choices=(1,2,4,8),default=1,
                   help='map distinct M=1 vocabulary tiles to existing arrays; hardware unchanged')
    a=p.parse_args();case=a.artifact_root.resolve()
    if not 1<=a.layers<=32:p.error('--layers must be in [1,32]')
    if not 1<=a.sequence<=8192:p.error('--sequence must be in [1,8192]')
    if a.gm_stage_chain and (not a.down_residual_fusion or a.ffn_layout!=2):p.error('--gm-stage-chain requires --down-residual-fusion --ffn-layout 2')
    if a.layer_residency and not a.gm_stage_chain:p.error('--layer-residency requires --gm-stage-chain')
    if a.add_norm_fusion and not (a.gm_stage_chain and a.input_residency):p.error('--add-norm-fusion requires --gm-stage-chain --input-residency')
    if a.add_norm_fusion and a.sequence>1024:p.error('--add-norm-fusion requires S<=1024')
    if a.eager_final_norm and not (a.layer_residency and (a.final_norm or a.lm_head or a.embedding)):p.error('--eager-final-norm requires --layer-residency and final Norm')
    if a.final_norm_lm_stream and not ((a.lm_head or a.embedding) and a.eager_final_norm and a.sequence<=1024):p.error('--final-norm-lm-stream requires LM head, --eager-final-norm and S<=1024')
    if a.final_norm_producer_first and not a.final_norm_lm_stream:p.error('--final-norm-producer-first requires --final-norm-lm-stream')
    if a.final_norm_lm_stream:a.tail_schedule_experiment=True
    if a.tail_schedule_experiment and not ((a.lm_head or a.embedding) and a.sequence<=1024 and a.eager_final_norm):p.error('--tail-schedule-experiment requires LM head, --eager-final-norm and S<=1024')
    if a.input_residency and not a.layer_residency:p.error('--input-residency requires --layer-residency')
    if a.elide_intermediate_output and not a.layer_residency:p.error('--elide-intermediate-output requires --layer-residency')
    if a.elide_intermediate_output and not a.input_residency:p.error('--elide-intermediate-output requires --input-residency for the next residual Add')
    if a.elide_intermediate_output and (a.sequence>1024 or a.retain_layer_outputs):
        p.error('--elide-intermediate-output requires S<=1024 and excludes --retain-layer-outputs')
    if a.down_residual_fusion:a.ffn_down_fusion=True
    if a.ffn_down_fusion:a.ffn_panel_fusion=True
    if a.ffn_array_groups==2 and not a.ffn_panel_fusion:
        p.error('--ffn-array-groups=2 requires fused FFN')
    if a.ffn_gm_lookahead and not a.ffn_panel_fusion:p.error('--ffn-gm-lookahead requires fused FFN')
    if a.ffn_demand_order and not a.ffn_panel_fusion:p.error('--ffn-demand-order requires fused FFN')
    if a.pv_wo_gm_handoff and (a.shape!='formal' or a.sequence!=1024):p.error('--pv-wo-gm-handoff requires --shape formal --sequence 1024')
    if a.pv_wo_direct_input and not a.pv_wo_gm_handoff:p.error('--pv-wo-direct-input requires --pv-wo-gm-handoff')
    if a.validate_inputs:validate_inputs(case);return
    if a.verify_only:
        m=validate_inputs(case)
        log,begin,end,lifecycle=verify_lifecycle(case,m)
        m['_lifecycle']=lifecycle
        result={'status':'PASS','shape':{'S':m.get('logical_sequence',m['sequence']),'padded_S':m['sequence'],'hidden':m['hidden'],'intermediate':4*m['hidden'],'layers':m['layers']},
            'numerics':verify_numerics(case,m),'lifecycle':lifecycle,'dma_timing':verify_dma(log,m,begin,end),
            'memory':{'per_node_used_bytes':m['layout']['used_bytes'],'per_node_capacity_bytes':m['layout']['node_stride'],
                      'activation_slots':m['layers']+1 if m.get('retain_outputs') else 2,
                      'shared_scratch':True,'weight_sets':m['layers']}}
        if m.get('final_norm'):
            result['final_norm']=verify_final_norm(case,m,log,end,result['dma_timing'])
        if m.get('lm_head'):
            from verify_lm_head import verify_lm_head
            result['lm_head']=verify_lm_head(case,m,log,result['final_norm'],result['dma_timing'])
        if m.get('embedding'):
            from verify_embedding import verify_embedding
            result['embedding']=verify_embedding(case,m,log,begin,result['lm_head'])
        from verify_decoder_dataflow_resources import verify as verify_resources
        from verify_gm_banked import verify as verify_banks
        result['resources']=verify_resources(case)
        result['gm_banks']=verify_banks(case)
        result['padding']={'logical_rows':m.get('logical_sequence',m['sequence']),
            'executed_rows':m['sequence'],'extra_rows':m['sequence']-m.get('logical_sequence',m['sequence']),
            'all_padded_work_timed':True,'causal_prefix_only':True}
        (case/'decoder_stack_result.json').write_text(json.dumps(result,indent=2)+'\n')
        (case/'verification_provenance.json').write_text(json.dumps({
            'verifier_source_sha256':digest(Path(__file__).resolve()),
            'scalar_SiLU_oracle':bool(m.get('silu_multiply_fusion') or m.get('ffn_panel_fusion')),
            'scope':'host validation only; simulator source/library and run environment remain in decoder_stack_provenance.json'},indent=2)+'\n')
        print(json.dumps(result,indent=2));return
    preflight(case)
    if a.host_hbm_root:preflight(a.host_hbm_root)
    case.mkdir(parents=True,exist_ok=False)
    if a.host_hbm_root:
        backing=a.host_hbm_root.resolve()/case.name
        backing.mkdir(parents=True,exist_ok=False)
        (case/'hbm').symlink_to(backing,target_is_directory=True)
        (case/'host_backing_storage.json').write_text(json.dumps({
            'host_backing_directory':str(backing),'simulated_hardware_changed':False,
            'note':'Only host file placement changes; wall time comparisons must record this condition.'},indent=2)+'\n')
    if a.embedding:a.lm_head=True
    vocab=(a.vocab if a.vocab is not None else (128256 if a.shape=='formal' else 8201)) if a.lm_head else 0
    if a.vocab is not None and not a.lm_head:p.error('--vocab requires --lm-head')
    if a.lm_head and not 1<=vocab<=1048576:p.error('vocab must be in [1,1048576]')
    node_mib=a.node_mib if a.node_mib is not None else (2048 if a.lm_head else 128)
    if node_mib<128 or node_mib & (node_mib-1):p.error('--node-mib must be a power of two >=128')
    node_bytes=node_mib*2**20
    m=prepare(case,*( (32,8) if a.shape=='formal' else (4,2)),final_norm=a.final_norm or a.lm_head,vocab=vocab,node_bytes=node_bytes,embedding=a.embedding,layers=a.layers,sequence=a.sequence,weight_kind=a.weights,retain_outputs=a.retain_layer_outputs)
    if a.silu_multiply_fusion or a.ffn_panel_fusion:
        m['silu_multiply_fusion']=bool(a.silu_multiply_fusion or a.ffn_panel_fusion)
        m['ffn_panel_fusion']=bool(a.ffn_panel_fusion)
        m['ffn_down_fusion']=bool(a.ffn_down_fusion)
        m['ffn_demand_order']=bool(a.ffn_demand_order)
        m['down_residual_fusion']=bool(a.down_residual_fusion)
        m['gm_stage_chain']=bool(a.gm_stage_chain)
        m['add_norm_fusion']=bool(a.add_norm_fusion)
        m['eager_final_norm']=bool(a.eager_final_norm)
        m['final_norm_lm_stream']=bool(a.final_norm_lm_stream)
        m['tail_schedule_experiment']=bool(a.tail_schedule_experiment)
        m['final_norm_producer_first']=bool(a.final_norm_producer_first)
        if a.tail_schedule_experiment:m['lm_handoff_protocol']='producer_remote_store_v1'
        m['layer_residency']=bool(a.layer_residency)
        m['input_residency']=bool(a.input_residency)
        m['lm_input_residency']=bool(a.lm_input_residency)
        m['elide_intermediate_output']=bool(a.elide_intermediate_output)
        m['ffn_layout']=a.ffn_layout
        m['down_input_reuse']=a.down_input_reuse
        if a.ffn_panel_fusion:m['ffn_panel_tiles']=a.ffn_panel_tiles
        (case/'decoder_stack_inputs.json').write_text(json.dumps(m,indent=2)+'\n')
    if a.tail_schedule_experiment:
        m.update(tail_schedule_experiment=True,final_norm_lm_stream=bool(a.final_norm_lm_stream),
                 final_norm_producer_first=bool(a.final_norm_producer_first),lm_handoff_protocol='producer_remote_store_v1')
        (case/'decoder_stack_inputs.json').write_text(json.dumps(m,indent=2)+'\n')
    if a.pv_wo_gm_handoff:
        m['pv_wo_gm_handoff']=True
        if a.pv_wo_direct_input:m['pv_wo_direct_input']=True
        (case/'decoder_stack_inputs.json').write_text(json.dumps(m,indent=2)+'\n')
    env=environment(case,(1,1,1),'none',m['sequence'],a.shape)
    env.update({'GOLEM_LOCAL_GM_MODEL':getattr(a, 'gm_model', 'legacy'),
        'GOLEM_LOCAL_GM_RESPONSE_CYCLES':str(getattr(a, 'gm_response_cycles', 1)),
        'GOLEM_LOCAL_GM_RESPONSE_CAPACITY':'32'})
    env.update({'GOLEM_DECODER_LAYERS':str(m['layers']),'GOLEM_DECODER_LOGICAL_SEQUENCE':str(m['logical_sequence']),'GOLEM_DECODER_RETAIN_OUTPUTS':str(int(m['retain_outputs'])),'GOLEM_DECODER_STACK_MANIFEST':str(case/'decoder_stack_inputs.json'),
        'GOLEM_ATTENTION_GUEST_EXECUTABLE':str(GUEST_DIR/'riscv64'/('decoder-embedding' if a.embedding else 'decoder-lm-head' if a.lm_head else 'decoder-final-norm' if a.final_norm else 'decoder-two-layer')/'fused_attention'),
        'GOLEM_RESOURCE_TRACE_DOMAINS':'gm','GOLEM_CRITICAL_PATH_TRACE':'1','OPENBLAS_NUM_THREADS':'1',
        'GOLEM_MPI_RANKS':str(a.mpi_ranks),
        'GOLEM_MPI_ARGS':f'--bind-to core --cpu-set {a.cpu_set or "0-"+str(a.mpi_ranks-1)} --output-filename {case/"mpi_rank_output"}'})
    # Make checks source/header timestamps; an existing executable alone does
    # not establish that a guest-side protocol change was compiled.
    guest_target=Path(env['GOLEM_ATTENTION_GUEST_EXECUTABLE']).parent.name
    if a.tail_schedule_experiment:
        guest_target+='-tail'
        env['GOLEM_ATTENTION_GUEST_EXECUTABLE']=str(GUEST_DIR/'riscv64'/guest_target/'fused_attention')
    subprocess.run(['make','-C',str(GUEST_DIR),guest_target],check=True)
    env['GOLEM_PV_WO_GM_HANDOFF']=str(int(a.pv_wo_gm_handoff))
    env['GOLEM_PV_WO_DIRECT_INPUT']=str(int(a.pv_wo_direct_input))
    env['GOLEM_DECODER_FFN_GM_LOOKAHEAD']=str(a.ffn_gm_lookahead)
    env['GOLEM_DECODER_FFN_DEMAND_ORDER']=str(int(a.ffn_demand_order))
    env['GOLEM_DECODER_GM_STAGE_CHAIN']=str(int(a.gm_stage_chain))
    env['GOLEM_DECODER_ADD_NORM_FUSION']=str(int(a.add_norm_fusion))
    env['GOLEM_DECODER_EAGER_FINAL_NORM']=str(int(a.eager_final_norm))
    # Leave the default guest environment identical to the measured legacy path.
    env.pop('GOLEM_DECODER_FINAL_NORM_LM_STREAM',None)
    env.pop('GOLEM_DECODER_FINAL_NORM_PRODUCER_FIRST',None)
    if a.tail_schedule_experiment:
        env['GOLEM_DECODER_FINAL_NORM_LM_STREAM']=str(int(a.final_norm_lm_stream))
        env['GOLEM_DECODER_FINAL_NORM_PRODUCER_FIRST']=str(int(a.final_norm_producer_first))
    env['GOLEM_DECODER_LAYER_RESIDENCY']=str(int(a.layer_residency))
    env['GOLEM_DECODER_INPUT_RESIDENCY']=str(int(a.input_residency))
    env['GOLEM_DECODER_LM_INPUT_RESIDENCY']=str(int(a.lm_input_residency))
    env['GOLEM_DECODER_ELIDE_INTERMEDIATE_OUTPUT']=str(int(a.elide_intermediate_output))
    env['GOLEM_DECODER_NODE_BYTES']=str(node_bytes)
    env['GOLEM_DECODER_OUTPUT_PANEL_TILES']=str(a.output_panel_tiles)
    env['GOLEM_DECODER_OUTPUT_WRITE_WINDOW']=str(a.output_write_window)
    env['GOLEM_DECODER_GEMM_PREFETCH']=str(int(a.gemm_prefetch))
    env['GOLEM_DECODER_PREFETCH_SLOTS']=str(a.prefetch_slots)
    env['GOLEM_DECODER_WEIGHT_TRACE']=str(int(a.weight_trace))
    env['GOLEM_DECODER_SILU_MULTIPLY_FUSION']=str(int(a.silu_multiply_fusion))
    env['GOLEM_DECODER_FFN_PANEL_FUSION']=str(int(a.ffn_panel_fusion))
    env['GOLEM_DECODER_FFN_DOWN_FUSION']=str(int(a.ffn_down_fusion))
    env['GOLEM_DECODER_DOWN_RESIDUAL_FUSION']=str(int(a.down_residual_fusion))
    env['GOLEM_DECODER_FFN_PANEL_TILES']=str(a.ffn_panel_tiles)
    env['GOLEM_DECODER_FFN_LAYOUT']=str(a.ffn_layout)
    env['GOLEM_DECODER_DOWN_INPUT_REUSE']=str(int(a.down_input_reuse))
    env['GOLEM_DECODER_ARRAY_GROUPS']=str(a.ffn_array_groups)
    env['GOLEM_DECODER_LM_N_PARALLEL']=str(a.lm_n_parallel)
    if a.lm_head:env['GOLEM_DECODER_VOCAB']=str(vocab)
    command=[str(HERE/'run_sst.sh'),'--projection','--query-length',str(m['sequence']),'--kv-length',str(m['sequence']),
        '--num-query-heads',str(m['num_query_heads']),'--num-kv-heads',str(m['num_kv_heads']),
        '--head-dim','64','--causal','--rope','--artifact-root',str(case)]
    sources=[ROOT/'src/sst/elements/golem'/name for name in (
        'sfu/sfu.cc','sfu/sfu.h','sfu/vector_math.h','sfu/hardware_pipeline.h','sfu/vector_pipeline.h',
        'tests/architecture/cpu_builder.py',
        'globalmemory/globalmemory.cc','globalmemory/globalmemory.h','globalmemory/localGmBankScheduler.h',
        'attention/decoderMemoryLayout.h','attention/projectionJobAbi.h','attention/projectionJob.h',
        'attention/pvWoHandoff.h','attention/pvWoInputInterface.h','attention/decoderOpAbi.h','attention/decoderOpJob.h','rocc/roccAnalog.h','workercmdproc/workercmdproc.h',
        'tests/small/muticore_attention/golem_projection_runtime.h',
        'tests/small/muticore_attention/golem_decoder_runtime.h',
        'tests/small/muticore_attention/golem_tail_schedule.h',
        'tests/small/muticore_attention/golem_attention_runtime.cpp',
        'tests/small/muticore_attention/dump_decoder_layout.cpp',
        'tests/small/muticore_attention/run_fused_attention_scale.sh','tests/tools/gen_hbm_init.py',
        'tests/architecture/ncores_selfcom_dma_ctrl.py')]
    sources.append(Path(__file__).resolve())
    sources.extend([HERE/'run_sst.sh',HERE/'merge_mpi_rank_logs.py',HERE/'artifact_storage.py',HERE/'run_decoder_layer.py',HERE/'dense_reference.py',HERE/'dense_linear_reference.cpp',HERE/'prefill_precision_reference.py',ROOT/'src/sst/elements/golem/tests/architecture/partition_planner.py'])
    if a.lm_head:sources.extend([HERE/'verify_lm_head.py',HERE/'select_next_token.py'])
    if a.embedding:sources.append(HERE/'verify_embedding.py')
    sources.extend([HERE/'verify_pv_wo_direct.py',HERE/'verify_decoder_dataflow_resources.py',HERE/'verification_retained_input.py'])
    for source in sources:
        target=case/'source_snapshot'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    (case/'decoder_stack_provenance.json').write_text(json.dumps({'command':command,
        'source_sha256':{str(s.relative_to(ROOT)):digest(s) for s in sources},
        'guest_sha256':digest(Path(env['GOLEM_ATTENTION_GUEST_EXECUTABLE'])),
        'sst_library_sha256':digest(ROOT/'install/lib/sst-elements-library/libgolem.so'),
        'sst_memhierarchy_sha256':digest(ROOT/'install/lib/sst-elements-library/libmemHierarchy.so'),
        'environment':{k:v for k,v in env.items() if k.startswith(('GOLEM_','VANADIS_'))}},indent=2)+'\n')
    print(f'Running {a.layers} layers, S={a.sequence} padded={m["sequence"]}: {a.shape}, {m["layout"]["used_bytes"]/2**20:.3f} MiB/data node',flush=True)
    driver_start=time.perf_counter()
    with (case/'driver.log').open('w') as stream:
        subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    result=json.loads((case/'decoder_stack_result.json').read_text())
    driver_wall=time.perf_counter()-driver_start
    match=re.search(r'\[PASS\] SST simulation\s+wall ([0-9.]+)s',(case/'driver.log').read_text())
    if not match:raise ValueError('missing SST wall timing')
    (case/'simulation_wall_time.json').write_text(json.dumps({'status':'PASS','mpi_ranks':a.mpi_ranks,
        'cpu_set':a.cpu_set or '0-'+str(a.mpi_ranks-1),'sst_wall_seconds':float(match[1]),
        'driver_wall_seconds':driver_wall,'timing_scope':'SST wall excludes input generation and host validation; driver wall covers run_sst and validation, excluding prepare before launch'},indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('status','shape','dma_timing','final_norm','lm_head','embedding') if key in result},indent=2))

if __name__=='__main__':main()
