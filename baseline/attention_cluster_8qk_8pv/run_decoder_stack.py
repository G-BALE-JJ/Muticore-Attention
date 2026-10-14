#!/usr/bin/env python3
"""Two decoder layers in one SST run; shared allocator and independent CPU oracle."""
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


@lru_cache(maxsize=None)
def memory_layout(hq, hkv, count=2, final_norm=False, vocab=0, node_bytes=128*2**20,embedding=False):
    # C++ is the single source of offsets. Python never duplicates its arithmetic.
    with tempfile.TemporaryDirectory() as directory:
        exe=Path(directory)/'layout'
        subprocess.run(['g++','-std=c++17','-O2',str(GUEST_DIR/'dump_decoder_layout.cpp'),'-o',str(exe)],check=True)
        result=subprocess.run([str(exe),'1024',str(hq),str(hkv),'64',str(count),str(int(final_norm)),str(vocab),str(node_bytes),str(int(embedding))],check=True,capture_output=True,text=True)
        return json.loads(result.stdout)


def sparse_weight(rng,n,k,scale):
    w=np.zeros((n,k),dtype='<f2')
    for row in range(n):
        cols=sorted(set(((row*17+3)%k,(row*31+79)%k,(row*7+137)%k,k-1)))
        w[row,cols]=rng.uniform(-scale,scale,len(cols)).astype('<f2')
    return w


def prepare(case,hq,hkv,final_norm=False,vocab=0,node_bytes=128*2**20,embedding=False):
    layout=memory_layout(hq,hkv,final_norm=final_norm,vocab=vocab,node_bytes=node_bytes,embedding=embedding); hidden=hq*64
    rng=np.random.default_rng(20261015)
    preloads=[]
    def preload(name,path,node,offset):
        preloads.append({'name':name,'path':str(path),'node':node,'offset':offset,
                         'bytes':path.stat().st_size,'sha256':digest(path)})
    x=rng.normal(0,.3,(1024,hidden)).astype('<f2'); x.tofile(case/'projection_x.bin')
    for node in range(1,5):
        path=case/f'x_node{node}.bin';x[(node-1)*256:node*256].tofile(path)
        if not embedding:preload('X0',path,node,layout['activations'][0])
    for index,l in enumerate(layout['layers'],1):
        for name in ('gamma1','gamma2'):
            data=rng.uniform(.8,1.2,hidden).astype('<f2')
            path=case/f'layer{index}_{name}.bin';data.tofile(path)
            for node in range(1,5):preload(name,path,node,l[name])
        for name,n,k,scale in [('qkv',hidden+2*hkv*64,hidden,.06),('wo',hidden,hidden,.25),
                              ('gate',4*hidden,hidden,.25),('up',4*hidden,hidden,.25),
                              ('down',hidden,4*hidden,.25)]:
            w=sparse_weight(rng,n,k,scale)
            np.save(case/f'layer{index}_{name}.npy',w)
            panels=w.reshape(n//64,64,k//64,64).transpose(0,2,1,3).copy()
            for node in range(1,5):
                path=case/f'layer{index}_{name}_node{node}.bin'
                (panels if name=='qkv' else panels[node-1::4]).tofile(path)
                preload(name,path,node,l[name+'_weight'])
    sys.path.insert(0,str(GUEST_DIR))
    from attention_case import generate_rope_table
    path=case/'rope_64.bin';generate_rope_table(path,1024,64)
    preload('rope',path,1,layout['shared']['rope'])
    # Poisoned flags prove that the guest clears both sets before layer 1.
    path=case/'poison_flags.bin';np.full(4,0xfeedface,dtype='<u8').tofile(path)
    for node in range(1,5):
        for name in ('projection_done','decoder_done'):
            preload(name,path,node,layout['shared'][name])
    if final_norm:
        gamma=case/'final_gamma.bin';rng.uniform(.7,1.3,hidden).astype('<f2').tofile(gamma)
        output=case/'poison_final_output.bin';np.full((256,hidden),np.nan,dtype='<f2').tofile(output)
        for node in range(1,5):
            preload('final_gamma',gamma,node,layout['final_norm']['gamma'])
            preload('final_done',path,node,layout['final_norm']['done'])
            preload('final_output',output,node,layout['final_norm']['output'])
    manifest={'sequence':1024,'hidden':hidden,'num_query_heads':hq,'num_kv_heads':hkv,
              'layers':2,'weight_kind':'distinct sparse synthetic matrices; all dense hardware tiles execute',
              'layout':layout,'preloads':preloads}
    if final_norm:manifest['final_norm']=True
    if vocab:
        columns=((vocab+63)//64)*64
        # Keep all Decoder inputs identical. A separate seed defines the head.
        w=sparse_weight(np.random.default_rng(20261016),columns,hidden,.15)
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
        tokens=rng_embedding.integers(0,vocab,1024,dtype=np.uint32)
        tokens[:4]=np.arange(4)%vocab;tokens[4:8]=tokens[:4];tokens[-1]=vocab-1
        tokens.astype('<u4').tofile(case/'input_token_ids.bin')
        x=table[tokens];x.tofile(case/'projection_x.bin')
        for node in range(1,5):
            weight=case/f'embedding_weight_node{node}.bin';table[node-1::4].tofile(weight)
            ids=case/f'embedding_tokens_node{node}.bin';tokens[(node-1)*256:node*256].astype('<u4').tofile(ids)
            preload('embedding_weight',weight,node,layout['embedding']['weight'])
            preload('embedding_tokens',ids,node,layout['embedding']['tokens'])
            preload('embedding_done',case/'poison_flags.bin',node,layout['embedding']['done'])
            preload('embedding_output',case/'poison_final_output.bin',node,layout['embedding']['output'])
        manifest['embedding']=True
    (case/'decoder_stack_inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def rms(x,gamma):
    f=x.astype(np.float32); square=np.zeros(len(x),dtype=np.float32)
    for col in range(x.shape[1]):square+=f[:,col]*f[:,col]
    return storage_half(f*(1/np.sqrt(square/np.float32(x.shape[1])+np.float32(1e-6)))[:,None]*gamma.astype(np.float32))


def attention_reference(x,qkv,hq,hkv,table):
    sys.path.insert(0,str(GUEST_DIR))
    from verify_fused_attention_scale_output import rotate_rope,compute_attention_blocked
    p=linear_reference(x,qkv).reshape(1024,hq+2*hkv,64)
    output=np.empty((1024,hq,64),dtype='<f2')
    for head in range(hq):
        kv=head//(hq//hkv)
        q=rotate_rope(p[:,head],1024,64,table,simulator_storage=True)
        k=rotate_rope(p[:,hq+kv],1024,64,table,simulator_storage=True)
        output[:,head]=storage_half(np.array(compute_attention_blocked(q,k,p[:,hq+hkv+kv],1024,1024,64,causal=True)).reshape(1024,64))
    return output.reshape(1024,-1)


def compare(got,want,tolerance,name):
    error=np.abs(got.astype(np.float32)-want.astype(np.float32))
    maximum=float(error.max())
    if not np.isfinite(got).all() or not np.isfinite(want).all() or maximum>tolerance:
        raise ValueError(f'{name}: max_abs_error={maximum}, tolerance={tolerance}')
    return {'status':'PASS','max_abs_error':maximum,'tolerance':tolerance,'elements':int(got.size)}


def verify_numerics(case,m):
    hidden=m['hidden']; hq=m['num_query_heads'];hkv=m['num_kv_heads'];s=m['layout']['shared']
    x=np.fromfile(case/'projection_x.bin',dtype='<f2').reshape(1024,hidden)
    table=np.fromfile(case/'rope_64.bin',dtype='<f2')
    results={}; expected=x
    # Compose both layers without consulting intermediate device tensors.
    for index,l in enumerate(m['layout']['layers'],1):
        weights={n:np.load(case/f'layer{index}_{n}.npy') for n in ('qkv','wo','gate','up','down')}
        gamma1=np.fromfile(case/f'layer{index}_gamma1.bin',dtype='<f2')
        gamma2=np.fromfile(case/f'layer{index}_gamma2.bin',dtype='<f2')
        o=attention_reference(rms(expected,gamma1),weights['qkv'],hq,hkv,table)
        h1=storage_half(expected.astype('f4')+linear_reference(o,weights['wo']).astype('f4'))
        norm=rms(h1,gamma2); gate=linear_reference(norm,weights['gate']);up=linear_reference(norm,weights['up'])
        g=gate.astype('f4');act=storage_half(g/(np.float32(1)+np.exp(-g)))
        gated=storage_half(act.astype('f4')*up.astype('f4'))
        expected=storage_half(linear_reference(gated,weights['down']).astype('f4')+h1.astype('f4'))
        got=read_bands(case,l['output'],hidden)
        results[f'layer{index}_composed']=compare(got,expected,4e-3,f'layer{index}_composed')
    if m.get('final_norm'):
        final=m['layout']['final_norm'];gamma=np.fromfile(case/'final_gamma.bin',dtype='<f2')
        got=read_bands(case,final['output'],hidden)
        final_expected=rms(expected,gamma)
        results['final_norm_composed']=compare(got,final_expected,6e-3,'final_norm_composed')
        if m.get('lm_head'):np.save(case/'final_reference_last_token.npy',final_expected[-1:])
        results['final_norm_local']=compare(got,rms(read_bands(case,m['layout']['layers'][-1]['output'],hidden),gamma),2e-3,'final_norm_local')
    # The last layer's scratch survives. Check its nine stages against device
    # inputs so errors cannot cancel between operators or across layers.
    l=m['layout']['layers'][-1]; index=m['layers']
    x=read_bands(case,l['input'],hidden);o=read_bands(case,s['o'],hidden)
    results['last_attention']=compare(o,attention_reference(rms(x,gamma1),weights['qkv'],hq,hkv,table),3e-4,'last_attention')
    actual={}
    def check(name,want,tolerance=0):
        actual[name]=read_bands(case,s[name],want.shape[1])
        results['last_'+name]=compare(actual[name],want,tolerance,name)
    check('wo',linear_reference(o,weights['wo']))
    check('h1',storage_half(x.astype('f4')+actual['wo'].astype('f4')))
    check('norm2',rms(actual['h1'],gamma2),2e-3)
    check('gate',linear_reference(actual['norm2'],weights['gate']))
    check('up',linear_reference(actual['norm2'],weights['up']))
    g=actual['gate'].astype('f4');check('act',storage_half(g/(np.float32(1)+np.exp(-g))),5e-4)
    check('gated',storage_half(actual['act'].astype('f4')*actual['up'].astype('f4')))
    check('down',linear_reference(actual['gated'],weights['down']))
    results['last_residual2']=compare(read_bands(case,l['output'],hidden),
        storage_half(actual['down'].astype('f4')+actual['h1'].astype('f4')),0,'last_residual2')
    for node in range(1,5):
        with (case/'hbm'/f'hbm_out_node{node}.bin').open('rb') as stream:
            for flag in ('projection_done','decoder_done'):
                stream.seek(s[flag]); values=np.frombuffer(stream.read(32),dtype='<u8')
                if values.size!=4 or not np.all(values==2):raise ValueError('stale final completion flags')
            if m.get('final_norm'):
                stream.seek(m['layout']['final_norm']['done']);values=np.frombuffer(stream.read(32),dtype='<u8')
                if values.size!=4 or not np.all(values==m['layers']+1):raise ValueError('stale final RMSNorm completion flags')
    return results


def fields(line):
    return {k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)}


def verify_lifecycle(case,m):
    log=next((case/'logs').glob('attention-sst*.log'))
    accepts=[]; ops=[];resets=[];scratch=[]
    with log.open() as stream:
        for line in stream:
            if '[CONTROL_EVENT] stage=projection_accept ' in line:accepts.append(fields(line))
            if '[DECODER_OP] event=done ' in line:ops.append(fields(line))
            if '[DECODER_SYNC] event=flag_write_ack ' in line:resets.append(fields(line))
            if '[DECODER_SYNC] event=scratch_retire ' in line:scratch.append(fields(line))
    if m.get('final_norm'):
        final_flags={node*m['layout']['node_stride']+m['layout']['final_norm']['done']+lane*8 for node in range(1,5) for lane in range(4)}
        if m.get('lm_head'):
            final_flags.update(node*m['layout']['node_stride']+m['layout']['lm_head']['done']+lane*8 for node in range(1,5) for lane in range(4))
        if m.get('embedding'):
            final_flags.update(node*m['layout']['node_stride']+m['layout']['embedding']['done']+lane*8 for node in range(1,5) for lane in range(4))
        ops=[e for e in ops if e['stage']!=10 and not (m.get('lm_head') and e['stage']==11) and not (m.get('embedding') and e['stage']==12)]
        resets=[e for e in resets if e['address'] not in final_flags]
    if len(accepts)!=32 or len(ops)!=288 or len(resets)!=64 or len(scratch)!=64:raise ValueError('missing or duplicate layer jobs/reset ACKs')
    begin={};end={};stages={};s=m['layout']['shared'];stride=m['layout']['node_stride']
    flags={node*stride+s[name]+lane*8 for node in range(1,5) for name in ('projection_done','decoder_done') for lane in range(4)}
    for epoch in (1,2):
        group=[e for e in accepts if e['epoch']==epoch]
        if {e['core'] for e in group}!=set(range(4,20)) or len(group)!=16:raise ValueError('projection worker generation mismatch')
        for e in group:
            node=1+(e['core']-4)%4;l=m['layout']['layers'][epoch-1]
            if e['input']!=node*stride+l['input'] or e['weights']!=node*stride+l['qkv_weight']:
                raise ValueError('layer input or weight address mismatch')
        begin[epoch]=min(e['tick'] for e in group)
        stage_summary={}
        for stage in range(1,10):
            group=[e for e in ops if e['epoch']==epoch and e['stage']==stage]
            if len(group)!=16 or {e['core'] for e in group}!=set(range(4,20)) or any(e['status'] for e in group):
                raise ValueError('incomplete/failed decoder stages')
            stage_summary[str(stage)]={'start_tick':min(e['start_tick'] for e in group),'end_tick':max(e['tick'] for e in group),
                'span_cycles':(max(e['tick'] for e in group)-min(e['start_tick'] for e in group))/1000,
                'workers':16,'hbm_read_bytes':sum(e['hbm_read'] for e in group),'hbm_write_bytes':sum(e['hbm_write'] for e in group)}
        stages[str(epoch)]=stage_summary;end[epoch]=stage_summary['9']['end_tick']
        group=[e for e in resets if (epoch==1 or e['tick']>=end[epoch-1]) and e['tick']<begin[epoch]]
        if len(group)!=32 or {e['address'] for e in group}!=flags or any(e['value'] or e['status'] for e in group):
            raise ValueError('flag reset not ACKed before projection release')
    for core in range(4,20):
        retires=sorted((e for e in scratch if e['core']==core),key=lambda e:e['tick'])
        if len(retires)!=4:raise ValueError('missing worker scratch ownership transitions')
        for epoch in (1,2):
            accepted=next(e for e in accepts if e['core']==core and e['epoch']==epoch)
            wo=next(e for e in ops if e['core']==core and e['epoch']==epoch and e['stage']==1)
            if not (retires[2*epoch-2]['tick']<=accepted['tick']<retires[2*epoch-1]['tick']<=wo['start_tick']):
                raise ValueError('operand cache invalidation order mismatch')
    if begin[2]<=end[1]:raise ValueError('next layer started before previous layer completion')
    stdout='\n'.join(p.read_text() for p in (case/'stdout').rglob('stdout-*'))
    completed=re.findall(r'DECODER_LAYER status=(\d+) stage=4 hidden=\d+ intermediate=\d+ layer=(\d+)',stdout)
    if sorted(completed)!=[('0','1'),('0','2')]:raise ValueError('root completion missing/failed')
    attention=re.findall(r'FUSED_ATTENTION status=(\d+) job=\d+ manager=(\d+)',stdout)
    if len(attention)!=8 or any(status!='0' for status,manager in attention):raise ValueError('manager Attention completion failed')
    # Reuse established event patterns but verify each layer separately. SFU
    # totals also contain FFN SiLU, so global single-Attention counters cannot
    # serve as a multilayer acceptance criterion.
    from report_sst import QK,SFU,PV,E2E
    text=log.read_text();cluster={}
    expected_jobs=4*m['num_kv_heads']*2
    expected_pv=m['num_query_heads']*sum((block*64+255)//256 for block in range(1,17))
    expected_tiles=m['num_query_heads']*16*17//2
    for epoch in (1,2):
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
                            'completed_jobs':len(done),'qk_tiles':expected_tiles,'root_tags':sorted(tags)}
    if set(cluster['1']['root_tags']) & set(cluster['2']['root_tags']):raise ValueError('layer Attention tag reused')
    (case/'decoder_stack_attention.json').write_text(json.dumps(cluster,indent=2)+'\n')
    return log,begin,end,{'status':'PASS','projection_jobs':32,'decoder_ops':288,'reset_ack_count':64,
        'layers':2,'scratch_retire_count':64,'distinct_attention_tags':True,'all_attention_jobs_checked':sum(e['completed_jobs'] for e in cluster.values()),'stage_spans':stages}


def verify_dma(log,m,begin,end):
    from report_dma_e2e import contained
    from analyze_resource_timeline import FIELDS,merged
    stride=m['layout']['node_stride'];qbytes=256*m['hidden']*2;s=m['layout']['shared']
    layers={epoch:{'first_input':None,'last_o':0,'last_y':0,'o_writes':[],'y_writes':[],'o_bytes':0,'y_bytes':0} for epoch in (1,2)}
    ranges=lambda offset:[(node*stride+offset,node*stride+offset+qbytes) for node in range(1,5)]
    with log.with_suffix('.resources.jsonl').open() as stream:
        for line in stream:
            if 'resource=dma_read_transport ' not in line and 'resource=dma_write_transport ' not in line:continue
            e={k:int(v) for k,v in FIELDS.findall(line) if k not in ('domain','resource')}
            epoch=next((n for n in (1,2) if begin[n]<=e['start_tick']<=end[n]),None)
            if epoch is None:continue
            state=layers[epoch];l=m['layout']['layers'][epoch-1]
            if 'resource=dma_read_transport ' in line and contained(e['address'],e['bytes'],ranges(l['input'])):
                state['first_input']=e['start_tick'] if state['first_input'] is None else min(state['first_input'],e['start_tick'])
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
            if merged(state[key+'_writes'])!=merged(ranges(offset)) or state[key+'_bytes']!=4*qbytes:
                raise ValueError('incomplete or duplicate layer output ACK coverage')
        if state['last_o']>min(v['start_tick'] for v in m['_lifecycle']['stage_spans'][str(epoch)].values()):
            raise ValueError('Wo started before all Attention O ACKs')
        result[str(epoch)]={'first_input_dma_tick':state['first_input'],'last_attention_o_ack_tick':state['last_o'],
            'last_y_ack_tick':state['last_y'],'attention_cycles':(state['last_o']-state['first_input'])/1000,
            'post_attention_cycles':(state['last_y']-state['last_o'])/1000,
            'layer_cycles':(state['last_y']-state['first_input'])/1000,'output_ack_bytes':state['y_bytes']}
    if result['2']['first_input_dma_tick']<=result['1']['last_y_ack_tick']:raise ValueError('layer input read before previous Y ACK')
    return {'contract':'first layer X input DMA send to second layer Y final DMA ACK',
        'ticks_per_cycle':1000,'layers':result,
        'layer_handoff_cycles':(result['2']['first_input_dma_tick']-result['1']['last_y_ack_tick'])/1000,
        'end_to_end_cycles':(result['2']['last_y_ack_tick']-result['1']['first_input_dma_tick'])/1000,
        'note':'Stage spans overlap; cannot be summed. Includes handoff/reset between layers; excludes host preload/startup and final exit.'}


def verify_final_norm(case,m,log,end,timing):
    from analyze_resource_timeline import FIELDS,merged
    final=m['layout']['final_norm'];stride=m['layout']['node_stride'];hidden=m['hidden']
    row_bytes=64*hidden*2;epoch=m['layers']+1
    flags={node*stride+final['done']+lane*8 for node in range(1,5) for lane in range(4)}
    ops=[];resets=[]
    with log.open() as stream:
        for line in stream:
            if '[DECODER_OP] event=done ' in line:
                e=fields(line)
                if e['stage']==10:ops.append(e)
            if '[DECODER_SYNC] event=flag_write_ack ' in line:
                e=fields(line)
                if e['address'] in flags:resets.append(e)
    if (len(ops)!=16 or {e['core'] for e in ops}!=set(range(4,20)) or
        any(e['status'] or e['op']!=3 or e['epoch']!=epoch for e in ops)):
        raise ValueError('missing/duplicate/failed final RMSNorm worker generation')
    start=min(e['start_tick'] for e in ops)
    if (len(resets)!=16 or {e['address'] for e in resets}!=flags or
        any(e['status'] or e['value'] or not end[2]<=e['tick']<start for e in resets)):
        raise ValueError('final RMSNorm flag reset ACK/barrier failed')
    states={e['core']:{'op':e,'reads':[],'writes':[],'read_bytes':0,'write_bytes':0,
        'gamma_bytes':0,'last_output':0,'flag_acks':[]} for e in ops}
    first_input=None
    with log.with_suffix('.resources.jsonl').open() as stream:
        for line in stream:
            if 'resource=dma_read_transport ' not in line and 'resource=dma_write_transport ' not in line:continue
            e={k:int(v) for k,v in FIELDS.findall(line) if k not in ('domain','resource')}
            if e['core'] not in states:continue
            state=states[e['core']];op=state['op']
            if not op['start_tick']<=e['start_tick']<=op['tick']:continue
            node=1+(e['core']-4)%4;lane=(e['core']-4)//4
            input_base=node*stride+m['layout']['layers'][-1]['output']+lane*row_bytes
            output_base=node*stride+final['output']+lane*row_bytes
            address=e['address'];stop=address+e['bytes']
            if 'resource=dma_read_transport ' in line:
                if input_base<=address<stop<=input_base+row_bytes:
                    if e['ok']!=1 or e['start_tick']<=timing['layers']['2']['last_y_ack_tick']:
                        raise ValueError('final RMSNorm input read before final layer Y ACK')
                    state['reads'].append((address,stop));state['read_bytes']+=e['bytes']
                    first_input=e['start_tick'] if first_input is None else min(first_input,e['start_tick'])
                elif node*stride+final['gamma']<=address<stop<=node*stride+final['gamma']+hidden*2:
                    if e['ok']!=1:raise ValueError('final gamma DMA failure')
                    state['gamma_bytes']+=e['bytes']
                else:raise ValueError('unexpected final RMSNorm input DMA address')
            else:
                if output_base<=address<stop<=output_base+row_bytes:
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
            base=node*stride+offset+lane*row_bytes
            if merged(state[key])!=[(base,base+row_bytes)]:raise ValueError('final RMSNorm DMA coverage incomplete')
        gamma_bytes=((64*hidden+8191)//8192)*hidden*2
        if (state['read_bytes']!=row_bytes or state['write_bytes']!=row_bytes or state['gamma_bytes']!=gamma_bytes or
            op['hbm_read']!=row_bytes+gamma_bytes or op['hbm_write']!=row_bytes+8):
            raise ValueError('final RMSNorm duplicate/missing data or gamma DMA')
        if (len(state['flag_acks'])!=1 or state['flag_acks'][0]['start_tick']<state['last_output'] or
            not state['last_output']<=state['flag_acks'][0]['done_tick']<=op['tick']):
            raise ValueError('final RMSNorm completed before output/flag DMA ACK')
    stdout='\n'.join(p.read_text() for p in (case/'stdout').rglob('stdout-*'))
    if re.findall(r'DECODER_FINAL_NORM status=(\d+) hidden=(\d+) epoch=(\d+)',stdout)!=[('0',str(hidden),str(epoch))]:
        raise ValueError('final RMSNorm root completion missing/failed')
    last_output=max(s['last_output'] for s in states.values())
    return {'status':'PASS','workers':16,'epoch':epoch,'reset_ack_count':16,
        'first_input_dma_tick':first_input,'last_output_ack_tick':last_output,
        'last_completion_ack_tick':max(s['flag_acks'][0]['done_tick'] for s in states.values()),
        'output_ack_bytes':16*row_bytes,'rmsnorm_cycles':(last_output-first_input)/1000,
        'handoff_cycles':(first_input-timing['layers']['2']['last_y_ack_tick'])/1000,
        'tail_cycles':(last_output-timing['layers']['2']['last_y_ack_tick'])/1000,
        'end_to_end_cycles':(last_output-timing['layers']['1']['first_input_dma_tick'])/1000,
        'contract':'first layer X input DMA send to final RMSNorm output DMA ACK',
        'epsilon':1e-6,'io_dtype':'fp16','internal_dtype':'fp32'}


def validate_inputs(case):
    manifest=json.loads((case/'decoder_stack_inputs.json').read_text())
    expected=memory_layout(manifest['num_query_heads'],manifest['num_kv_heads'],final_norm=manifest.get('final_norm',False),
        vocab=manifest['layout'].get('lm_head',{}).get('vocab',0),node_bytes=manifest['layout']['node_stride'],embedding=manifest.get('embedding',False))
    if manifest['layout']!=expected:raise ValueError('manifest differs from authoritative layout')
    for entry in manifest['preloads']:
        path=Path(entry['path'])
        if path.stat().st_size!=entry['bytes'] or digest(path)!=entry['sha256']:raise ValueError('input provenance mismatch')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root',type=Path,required=True)
    p.add_argument('--shape',choices=('small','formal'),default='small')
    p.add_argument('--cpu-set',help='default: one physical core per rank, starting at core 0')
    p.add_argument('--mpi-ranks',type=int,choices=(2,4,8),default=4)
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--validate-inputs',action='store_true')
    p.add_argument('--final-norm',action='store_true',help='append independent final RMSNorm after both layers')
    p.add_argument('--lm-head',action='store_true',help='append final norm and last-token LM head')
    p.add_argument('--vocab',type=int,help='default: 8201 for small, 128256 for formal')
    p.add_argument('--node-mib',type=int,help='HBM capacity per node; default 2048 for LM head, otherwise 128')
    p.add_argument('--embedding',action='store_true',help='on-architecture token lookup before the complete prefill path')
    a=p.parse_args();case=a.artifact_root.resolve()
    if a.validate_inputs:validate_inputs(case);return
    if a.verify_only:
        m=validate_inputs(case)
        log,begin,end,lifecycle=verify_lifecycle(case,m)
        m['_lifecycle']=lifecycle
        result={'status':'PASS','shape':{'S':1024,'hidden':m['hidden'],'intermediate':4*m['hidden'],'layers':2},
            'numerics':verify_numerics(case,m),'lifecycle':lifecycle,'dma_timing':verify_dma(log,m,begin,end),
            'memory':{'per_node_used_bytes':m['layout']['used_bytes'],'per_node_capacity_bytes':m['layout']['node_stride'],
                      'activation_slots':2,'shared_scratch':True,'weight_sets':2}}
        if m.get('final_norm'):
            result['final_norm']=verify_final_norm(case,m,log,end,result['dma_timing'])
        if m.get('lm_head'):
            from verify_lm_head import verify_lm_head
            result['lm_head']=verify_lm_head(case,m,log,result['final_norm'],result['dma_timing'])
        if m.get('embedding'):
            from verify_embedding import verify_embedding
            result['embedding']=verify_embedding(case,m,log,begin,result['lm_head'])
        (case/'decoder_stack_result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2));return
    case.mkdir(parents=True,exist_ok=False)
    if a.embedding:a.lm_head=True
    vocab=(a.vocab if a.vocab is not None else (128256 if a.shape=='formal' else 8201)) if a.lm_head else 0
    if a.vocab is not None and not a.lm_head:p.error('--vocab requires --lm-head')
    if a.lm_head and not 1<=vocab<=1048576:p.error('vocab must be in [1,1048576]')
    node_mib=a.node_mib if a.node_mib is not None else (2048 if a.lm_head else 128)
    if node_mib<128 or node_mib & (node_mib-1):p.error('--node-mib must be a power of two >=128')
    node_bytes=node_mib*2**20
    m=prepare(case,*( (32,8) if a.shape=='formal' else (4,2)),final_norm=a.final_norm or a.lm_head,vocab=vocab,node_bytes=node_bytes,embedding=a.embedding)
    env=environment(case,(1,1,1),'none',1024,a.shape)
    env.update({'GOLEM_DECODER_LAYERS':'2','GOLEM_DECODER_STACK_MANIFEST':str(case/'decoder_stack_inputs.json'),
        'GOLEM_ATTENTION_GUEST_EXECUTABLE':str(GUEST_DIR/'riscv64'/('decoder-embedding' if a.embedding else 'decoder-lm-head' if a.lm_head else 'decoder-final-norm' if a.final_norm else 'decoder-two-layer')/'fused_attention'),
        'GOLEM_RESOURCE_TRACE_DOMAINS':'gm','GOLEM_CRITICAL_PATH_TRACE':'1','OPENBLAS_NUM_THREADS':'1',
        'GOLEM_MPI_RANKS':str(a.mpi_ranks),
        'GOLEM_MPI_ARGS':f'--bind-to core --cpu-set {a.cpu_set or "0-"+str(a.mpi_ranks-1)} --output-filename {case/"mpi_rank_output"}'})
    env['GOLEM_DECODER_NODE_BYTES']=str(node_bytes)
    if a.lm_head:env['GOLEM_DECODER_VOCAB']=str(vocab)
    command=[str(HERE/'run_sst.sh'),'--projection','--query-length','1024','--kv-length','1024',
        '--num-query-heads',str(m['num_query_heads']),'--num-kv-heads',str(m['num_kv_heads']),
        '--head-dim','64','--causal','--rope','--artifact-root',str(case)]
    sources=[ROOT/'src/sst/elements/golem'/name for name in (
        'sfu/sfu.cc','sfu/sfu.h','sfu/vector_math.h','sfu/hardware_pipeline.h','sfu/vector_pipeline.h',
        'tests/architecture/cpu_builder.py',
        'attention/decoderMemoryLayout.h','attention/projectionJobAbi.h','attention/projectionJob.h',
        'attention/decoderOpAbi.h','attention/decoderOpJob.h','rocc/roccAnalog.h','workercmdproc/workercmdproc.h',
        'tests/small/muticore_attention/golem_projection_runtime.h',
        'tests/small/muticore_attention/golem_decoder_runtime.h',
        'tests/small/muticore_attention/golem_attention_runtime.cpp',
        'tests/small/muticore_attention/dump_decoder_layout.cpp',
        'tests/small/muticore_attention/run_fused_attention_scale.sh','tests/tools/gen_hbm_init.py',
        'tests/architecture/ncores_selfcom_dma_ctrl.py')]
    sources.append(Path(__file__).resolve())
    sources.extend([HERE/'run_sst.sh',HERE/'merge_mpi_rank_logs.py',ROOT/'src/sst/elements/golem/tests/architecture/partition_planner.py'])
    if a.lm_head:sources.extend([HERE/'verify_lm_head.py',HERE/'select_next_token.py'])
    if a.embedding:sources.append(HERE/'verify_embedding.py')
    for source in sources:
        target=case/'source_snapshot'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    (case/'decoder_stack_provenance.json').write_text(json.dumps({'command':command,
        'source_sha256':{str(s.relative_to(ROOT)):digest(s) for s in sources},
        'guest_sha256':digest(Path(env['GOLEM_ATTENTION_GUEST_EXECUTABLE'])),
        'sst_library_sha256':digest(ROOT/'install/lib/sst-elements-library/libgolem.so'),
        'environment':{k:v for k,v in env.items() if k.startswith(('GOLEM_','VANADIS_'))}},indent=2)+'\n')
    print(f'Running two layers: {a.shape}, {m["layout"]["used_bytes"]/2**20:.3f} MiB/data node',flush=True)
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
