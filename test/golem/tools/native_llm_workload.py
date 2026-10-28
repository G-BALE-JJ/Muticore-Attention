"""Explicit packed LLM matrix workloads on existing TE / NoC / HBM resources."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np
from tools.native_workload_plan import build_plan, panel_locations

MODES={'int4':(0,4),'int8':(1,8),'fp16':(2,16),'bf16':(3,16),
       'tf32':(4,32),'fp32':(5,32),'fp8e4m3':(6,8),'fp8e5m2':(7,8)}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=8)
def load_config(path):
    config=json.loads(Path(path).read_text())
    if config.get('schema')!='llm.native-operator' or config.get('precision') not in MODES:
        raise ValueError('unsupported native operator/precision contract')
    plan=config['plan'];g=plan['geometry']
    expected=build_plan(m_tiles=g['m_tiles'],n_tiles=g['n_tiles'],k_tiles=g['k_tiles'],
        reuse_m=g['reuse_m'],reuse_n=g['reuse_n'],workers=16,groups=4,data_nodes=4,
        instances=len(plan['instances']),panel_mapping=plan['panel_mapping'],replicate_b=plan['replicate_b'],
        block_m=g['block_m'],block_n=g['block_n'],block_k=g['block_k'],elem_bytes=4,
        node_bytes=plan['node_bytes'],input_bits=MODES[config['precision']][1])
    if plan!=expected:raise ValueError('native plan differs from canonical physical partition / packed layout')
    if (g['block_m'],g['block_n'],g['block_k'])!=(64,64,64):
        raise ValueError('native LLM operators retain the existing 64x64 arrays')
    # The existing guest must dispatch every physical worker before the TE bridge.
    macro=((g['m_tiles']+g['reuse_m']-1)//g['reuse_m'])*((g['n_tiles']+g['reuse_n']-1)//g['reuse_n'])
    if macro<16:raise ValueError('native workload requires at least sixteen guest macro tasks')
    return config


def engine_parameters(path,core_id,groups=4):
    c=load_config(str(Path(path).resolve()));p=c['plan']
    if groups!=4:raise ValueError('native LLM workload requires the fixed four-manager machine')
    if core_id<groups:return {}
    worker=core_id-groups
    owners=[j for j in p['instances'] if worker in j['physical_workers']]
    if len(owners)!=1:raise ValueError('worker has no unique native instance owner')
    j=owners[0];r=p['regions']
    return dict(native_precision_mode=MODES[c['precision']][0],
        native_worker_slot=j['physical_workers'].index(worker),native_worker_count=len(j['physical_workers']),
        native_data_nodes=j['data_nodes'],native_task_nodes=j['task_nodes'],
        native_a_nodes=j['a_tile_nodes'],native_b_nodes=j['b_tile_nodes'],
        native_replicate_b=int(p['replicate_b']),native_a_offset=r['a']['offset'],
        native_b_offset=r['b']['offset'],native_c_offset=r['c']['offset'])


def encode(values,precision):
    """Encode supported native operands. Integer operands must fit their native signed width; FP8/BF16/TF32 validation operands are exact -2..2."""
    x=np.asarray(values,dtype='<f4');bits=MODES[precision][1]
    if not np.all(np.isfinite(x)):raise ValueError('native operands must be finite')
    if precision=='fp16':raw=x.astype('<f2').view('<u2')
    elif precision=='fp32':raw=x.view('<u4')
    else:
        if precision.startswith('int'):
            limit=1<<(bits-1)
            if not np.all(x==np.floor(x)) or np.any(x < -limit) or np.any(x >= limit):
                raise ValueError('integer operands exceed the signed native precision range')
        elif not np.all(np.isin(x,[-2,-1,0,1,2])):
            raise ValueError('this native validation encoder requires exact -2..2 values for '+precision)
        if precision in ('int4','int8'):raw=x.astype('<i4').view('<u4')&((1<<bits)-1)
        elif precision=='bf16':raw=x.view('<u4')>>16
        elif precision=='tf32':raw=x.view('<u4')
        else:
            one=0x38 if precision=='fp8e4m3' else 0x3c
            raw=np.where(x==0,0,np.where(abs(x)==1,one,0x40)|(np.where(x<0,0x80,0))).astype('uint32')
    row_bytes=((x.shape[-1]*bits+31)//32)*4
    if bits==4:
        out=np.zeros((*x.shape[:-1],row_bytes),dtype='uint8')
        for k in range(x.shape[-1]):out[...,k//2]|=(raw[...,k].astype('uint8')<<(4*(k%2)))
    else:
        out=np.zeros((*x.shape[:-1],row_bytes),dtype='uint8')
        payload=np.ascontiguousarray(raw,dtype={8:'u1',16:'<u2',32:'<u4'}[bits]).view('uint8').reshape(*x.shape[:-1],-1)
        out[...,:payload.shape[-1]]=payload
    return out.tobytes()


def write_payloads(plan,inputs,precision,directory):
    target=Path(directory);target.mkdir(parents=True,exist_ok=False)
    g=plan['geometry'];entries=[]
    for instance,(a,b) in enumerate(inputs):
        for kind,tile,k,node,offset in panel_locations(plan,instance):
            ks=slice(k*64,(k+1)*64)
            values=a[tile*64:(tile+1)*64,ks] if kind=='a' else b[ks,tile*64:(tile+1)*64].T
            data=encode(values,precision)
            path=target/f'i{instance}_{kind}{tile}_k{k}_n{node}.bin';path.write_bytes(data)
            entries.append(dict(node=node,offset=offset,bytes=len(data),path=str(path.resolve()),sha256=digest(path)))
    return entries


def preload_native_images(path,node_buffers,node_bytes):
    c=load_config(str(Path(path).resolve()));p=c['plan']
    if p['node_bytes']!=node_bytes:raise ValueError('native plan / physical HBM capacity mismatch')
    for e in c['payloads']:
        data=Path(e['path']).read_bytes();node=e['node'];offset=e['offset']
        if node==0 or node not in node_buffers or len(data)!=e['bytes'] or hashlib.sha256(data).hexdigest()!=e['sha256']:
            raise ValueError('native payload hash/size/node mismatch')
        if offset<0 or offset+len(data)>p['regions']['c']['offset']:
            raise ValueError('native operand payload overlaps output region or exceeds capacity')
        node_buffers[node].write_block(offset,data)
