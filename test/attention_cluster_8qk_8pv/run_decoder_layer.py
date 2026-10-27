#!/usr/bin/env python3
"""Opt-in S=1024 decoder checkpoints on the unchanged 20-core baseline."""
import argparse
import json
import re
import subprocess
import shutil
from pathlib import Path
import numpy as np
from run_batch_b import environment
from run_projection_overlap import ROOT, HERE, digest

OFFSETS = {'wo_weight':0x02800000, 'gate_weight':0x03000000,
           'up_weight':0x04200000, 'down_weight':0x05200000,
           'wo':0x02400000, 'h1':0x02500000, 'norm':0x02600000,
           'gamma':0x02700000, 'down':0x02a00000, 'y':0x02b00000,
           'gate':0x03800000, 'up':0x03c00000, 'act':0x04a00000, 'gated':0x06200000}
STAGES = {'wo':1, 'residual':2, 'ffn':3, 'layer':4}
GUEST_DIR = ROOT/'test/golem/small/muticore_attention'



def validate_layout(hidden, kv_heads):
    if hidden not in (256,2048) or kv_heads not in (2,8):
        raise ValueError('unsupported decoder baseline shape')
    qbytes=256*hidden*2; kvbytes=256*kv_heads*64*2
    align=lambda x:(x+0xfffff)&~0xfffff
    norm_end=0x01100000+qbytes
    k=align(0x02000000+qbytes);v=align(k+kvbytes);o=align(v+kvbytes)
    regions=[('X',0,qbytes),('gamma1',0x01000000,hidden*2),
        ('norm1',0x01100000,qbytes),('qkv_weights',align(norm_end),(hidden+2*kv_heads*64)*hidden*2),
        ('projection_flags',0x01ff0000,32),('decoder_flags',0x01ff0100,32),
        ('Q',0x02000000,qbytes),('K',k,kvbytes),('V',v,kvbytes),('O',o,qbytes),
        ('Qpanel',0x04000000,qbytes),('Kpanel',0x05000000,kvbytes),
        ('Vpanel',0x06000000,kvbytes),('rope',0x07000000,1024*64*2)]
    for name,n,k in [('wo',hidden,hidden),('gate',4*hidden,hidden),
                      ('up',4*hidden,hidden),('down',hidden,4*hidden)]:
        regions.append((name+'_weight',OFFSETS[name+'_weight'],n*k*2//4))
    for name in ('wo','h1','norm','gate','up','act','gated','down','y'):
        width=4*hidden if name in ('gate','up','act','gated') else hidden
        regions.append((name,OFFSETS[name],256*width*2))
    regions.append(('gamma2',OFFSETS['gamma'],hidden*2))
    regions.sort(key=lambda r:r[1])
    for i,(name,start,size) in enumerate(regions):
        if size<=0 or start+size>0x08000000:raise ValueError('decoder HBM capacity overflow')
        if i and regions[i-1][1]+regions[i-1][2]>start:
            raise ValueError(f'HBM overlap: {regions[i-1][0]} / {name}')
    return [{'name':name,'offset':start,'bytes':size} for name,start,size in regions]


def prepare(case, hidden):
    rng = np.random.default_rng(20261014)
    regions=validate_layout(hidden,2 if hidden==256 else 8)
    preloads = []
    for name, n, k in [('wo',hidden,hidden), ('gate',4*hidden,hidden),
                       ('up',4*hidden,hidden), ('down',hidden,4*hidden)]:
        # Nontrivial sparse synthetic weights exercise all input tiles and
        # output-shard ownership, including the final 8192-wide down tile.
        matrix = np.zeros((n,k), dtype='<f2')
        for row in range(n):
            cols = sorted(set([(row*17+3)%k, (row*31+79)%k, (row*7+137)%k, k-1]))
            matrix[row,cols] = rng.uniform(-.25,.25,len(cols)).astype('<f2')
        panels = matrix.reshape(n//64,64,k//64,64).transpose(0,2,1,3).copy()
        for node in range(1,5):
            data = panels[node-1::4].tobytes()
            path = case/f'decoder_{name}_node{node}.bin'; path.write_bytes(data)
            preloads.append({'name':f'decoder_{name}', 'node':node,
                'offset':OFFSETS[name+'_weight'], 'path':str(path), 'bytes':len(data)})
        np.save(case/f'decoder_{name}_weight.npy', matrix)
    gamma = rng.uniform(.8,1.2,hidden).astype('<f2')
    path = case/'decoder_gamma.bin'; path.write_bytes(gamma.tobytes())
    for node in range(1,5):
        preloads.append({'name':'decoder_gamma', 'node':node, 'offset':OFFSETS['gamma'],
                         'path':str(path), 'bytes':gamma.nbytes})
    manifest = {'sequence':1024, 'hidden':hidden, 'intermediate':4*hidden,
                'weight_kind':'sparse synthetic; hardware still executes every dense tile',
                'regions_per_node':regions, 'preloads':preloads, 'sha256':{Path(e['path']).name:digest(Path(e['path'])) for e in preloads}}
    path = case/'decoder_inputs.json'; path.write_text(json.dumps(manifest,indent=2)+'\n')
    return path


def read_bands(case, offset, width, heads=0, rows=256):
    pieces=[]
    for node in range(1,5):
        path=case/'hbm'/f'hbm_out_node{node}.bin'
        with path.open('rb') as f:
            f.seek(offset); data=np.frombuffer(f.read(rows*width*2),dtype='<f2').copy()
        if data.size!=rows*width: raise ValueError(f'short HBM output: {path}')
        if heads: data=data.reshape(heads,rows,width//heads).transpose(1,0,2).reshape(rows,width)
        else: data=data.reshape(rows,width)
        pieces.append(data)
    return np.concatenate(pieces)



def storage_half(values):
    # Match the current project's conversion helper, including its tiny-value
    # cutoff below 2**-24 (not fully IEEE rounding near half-min-subnormal).
    values = np.array(values, dtype=np.float32, copy=True)
    tiny = np.abs(values) < 2**-24
    values[tiny] = np.copysign(np.float32(0), values[tiny])
    return values.astype('<f2')


def sparse_linear_reference(x, w):
    """Independent sparse reference to array FP16 products, dot and tile accumulation."""
    result=np.zeros((len(x),len(w)),dtype='<f2')
    for output,row in enumerate(w):
        nz=np.flatnonzero(row)
        for tile in sorted(set(nz//64)):
            dot=np.zeros(len(x),dtype='<f2')
            for col in nz[nz//64==tile]:
                product=storage_half(x[:,col].astype(np.float32)*np.float32(row[col]))
                dot=storage_half(dot.astype(np.float32)+product.astype(np.float32))
            result[:,output]=storage_half(result[:,output].astype(np.float32)+dot.astype(np.float32))
    return result


def linear_reference(x, w):
    """Use the independent scalar host oracle for dense numerical stress."""
    if x.ndim!=2 or w.ndim!=2 or x.shape[1]!=w.shape[1] or x.shape[1]%64:
        raise ValueError('linear reference shape mismatch')
    if np.count_nonzero(w)>len(w)*8:
        import struct
        from dense_reference import executable
        result=subprocess.run([str(executable())],input=struct.pack('<III',len(x),len(w),x.shape[1])+
                              np.asarray(x,dtype='<f2').tobytes()+np.asarray(w,dtype='<f2').tobytes(),
                              capture_output=True,check=True).stdout
        if len(result)!=len(x)*len(w)*2:raise ValueError('short dense reference output')
        return np.frombuffer(result,dtype='<f2').reshape(len(x),len(w))
    return sparse_linear_reference(x,w)


def verify(case, hidden, heads, stage):
    result={}; actual={}
    def check(name, expected, tolerance=0):
        got=read_bands(case,OFFSETS[name],expected.shape[1]); actual[name]=got
        error=np.abs(got.astype(np.float32)-expected.astype(np.float32))
        okay=np.isfinite(got).all() and np.max(error)<=tolerance
        result[name]={'status':'PASS' if okay else 'FAIL', 'max_abs_error':float(np.max(error)),
                      'tolerance':tolerance, 'elements':int(got.size)}
        if not okay: raise ValueError(f'{name} verification failed: {result[name]}')
    # Each stage is checked against its actual device-produced input, with a
    # second composed layer check below to detect accumulated numerical drift.
    # The baseline aligns each Q/K/V region to 1 MiB.
    align=lambda x:(x+0xfffff)&~0xfffff
    qbytes=256*hidden*2; kvbytes=256*(heads[1]*64)*2
    o_offset=align(align(0x02000000+qbytes)+kvbytes); o_offset=align(o_offset+kvbytes)
    attention=read_bands(case,o_offset,hidden)
    weights={n:np.load(case/f'decoder_{n}_weight.npy') for n in ('wo','gate','up','down')}
    wo=linear_reference(attention,weights['wo']); check('wo',wo)
    if stage==1:return result
    x=np.fromfile(case/'projection_x.bin',dtype='<f2').reshape(1024,hidden)
    h1=storage_half(x.astype(np.float32)+wo.astype(np.float32)); check('h1',h1)
    if stage==2:return result
    gamma=np.fromfile(case/'decoder_gamma.bin',dtype='<f2').astype(np.float32)
    f=h1.astype(np.float32); square=np.zeros(len(f),dtype=np.float32)
    for col in range(hidden):square+=f[:,col]*f[:,col]
    norm=(f*(1/np.sqrt(square/np.float32(hidden)+np.float32(1e-6)))[:,None]*gamma).astype('<f2')
    check('norm',norm,2e-3)
    gate=linear_reference(actual['norm'],weights['gate']);check('gate',gate)
    up=linear_reference(actual['norm'],weights['up']);check('up',up)
    g=gate.astype(np.float32);act=storage_half(g/(np.float32(1)+np.exp(-g)));check('act',act,5e-4)
    gated=storage_half(actual['act'].astype(np.float32)*up.astype(np.float32));check('gated',gated)
    down=linear_reference(gated,weights['down']);check('down',down)
    if stage==4:
        y=storage_half(down.astype(np.float32)+h1.astype(np.float32));check('y',y)
        # Compose entirely from Attention O, independently of intermediate HBM.
        cg=linear_reference(norm,weights['gate']).astype(np.float32)
        cu=linear_reference(norm,weights['up']).astype(np.float32)
        ca=storage_half(cg/(1+np.exp(-cg)))
        cm=storage_half(ca.astype(np.float32)*cu)
        cy=storage_half(linear_reference(cm,weights['down']).astype(np.float32)+h1.astype(np.float32))
        error=np.abs(actual['y'].astype(np.float32)-cy.astype(np.float32))
        if np.max(error)>4e-3:raise ValueError('composed decoder layer check failed')
        result['composed_layer']={'status':'PASS','max_abs_error':float(np.max(error)), 'tolerance':4e-3}
    return result


def stage_cycles(case):
    events=[]
    for path in (case/'logs').glob('attention-sst*.log'):
        for line in path.read_text(errors='replace').splitlines():
            if '[DECODER_OP] event=done' in line:
                events.append({k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)})
    unique={(e['core'],e['stage']):e for e in events};events=list(unique.values())
    summary={}
    for stage in sorted({e['stage'] for e in events}):
        group=[e for e in events if e['stage']==stage]
        if len(group)!=16 or any(e['status'] for e in group):raise ValueError('missing/failed decoder workers')
        start=min(e['start_tick'] for e in group);end=max(e['tick'] for e in group)
        summary[str(stage)]={'workers':16,'start_cycle':start/1000,'end_cycle':end/1000,
            'span_cycles':(end-start)/1000, 'hbm_read_bytes':sum(e['hbm_read'] for e in group),
            'hbm_write_bytes':sum(e['hbm_write'] for e in group)}
    if not summary:raise ValueError('no decoder stage timing')
    summaries=[]
    for path in (case/'stdout').rglob('stdout-*'):
        summaries.extend(re.findall(r'DECODER_LAYER status=(\d+) stage=(\d+) hidden=(\d+) intermediate=(\d+)',path.read_text()))
    if len(summaries)!=1 or int(summaries[0][0])!=0:
        raise ValueError('missing or failed root decoder completion')
    checkpoint=int(summaries[0][1]);last={1:1,2:2,3:8,4:9}[checkpoint]
    if sorted(map(int,summary))!=list(range(1,last+1)):
        raise ValueError('decoder stage coverage incomplete')
    return summary



def dma_cycles(case, hidden, checkpoint):
    from report_dma_e2e import report as attention_report, contained
    from analyze_resource_timeline import FIELDS, merged
    log=next((case/'logs').glob('attention-sst*.log'))
    if not log.with_suffix('.resources.jsonl').stat().st_size:
        return {'available':False, 'reason':'resource trace disabled in this run'}
    attention=attention_report(case)
    (case/'dma_e2e_result.json').write_text(json.dumps(attention,indent=2)+'\n')
    name={'wo':'wo','residual':'h1','ffn':'down','layer':'y'}[checkpoint]
    ranges=[(node*0x08000000+OFFSETS[name],node*0x08000000+OFFSETS[name]+256*hidden*2)
            for node in range(1,5)]
    writes=[];end=0;total=0
    with log.with_suffix('.resources.jsonl').open() as stream:
        for line in stream:
            if 'resource=dma_write_transport ' not in line:continue
            e={k:int(v) for k,v in FIELDS.findall(line) if k not in ('domain','resource')}
            if contained(e['address'],e['bytes'],ranges):
                if e['ok']!=1:raise ValueError('decoder output DMA failed')
                writes.append((e['address'],e['address']+e['bytes']))
                end=max(end,e['done_tick']);total+=e['bytes']
    if merged(writes)!=merged(ranges) or total!=1024*hidden*2:
        raise ValueError('decoder final ACK coverage incomplete or duplicated')
    origin=attention['first_data_dma_send_tick'];o_end=attention['last_O_ack_tick']
    return {'available':True,'contract':'first input data DMA send to final checkpoint output ACK',
        'ticks_per_cycle':1000,'first_data_dma_send_tick':origin,'last_attention_O_ack_tick':o_end,
        'last_checkpoint_ack_tick':end,'attention_cycles':(o_end-origin)/1000,
        'post_attention_cycles':(end-o_end)/1000,'end_to_end_cycles':(end-origin)/1000,
        'output_ack_bytes':total,'includes':'all compute, data transport, and handoffs within this interval',
        'excludes':'startup before first data DMA; final flag polling/printing/exit'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root',type=Path,required=True)
    p.add_argument('--shape',choices=('small','formal'),default='small')
    p.add_argument('--checkpoint',choices=STAGES,default='wo')
    p.add_argument('--reference',type=Path)
    p.add_argument('--cpu-set')
    p.add_argument('--mpi-ranks',type=int,choices=(4,8),default=4)
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--output-panel-tiles',type=int,choices=range(17),default=16)
    p.add_argument('--output-write-window',type=int,choices=range(1,17),default=8)
    p.add_argument('--gemm-prefetch',action='store_true')
    args=p.parse_args();case=args.artifact_root.resolve()
    heads=(32,8) if args.shape=='formal' else (4,2);hidden=heads[0]*64;stage=STAGES[args.checkpoint]
    if not args.verify_only:
        case.mkdir(parents=True,exist_ok=False)
        manifest=prepare(case,hidden)
        env=environment(case,(1,1,1),'none',1024,args.shape)
        guest=GUEST_DIR/f'riscv64/decoder-{args.checkpoint}/fused_attention'
        env['GOLEM_ATTENTION_GUEST_EXECUTABLE']=str(guest)
        env['GOLEM_DECODER_INPUT_MANIFEST']=str(manifest)
        env['GOLEM_RESOURCE_TRACE_DOMAINS']='gm';env['GOLEM_CRITICAL_PATH_TRACE']='1'
        env['GOLEM_DECODER_OUTPUT_PANEL_TILES']=str(args.output_panel_tiles)
        env['GOLEM_DECODER_OUTPUT_WRITE_WINDOW']=str(args.output_write_window)
        env['GOLEM_DECODER_GEMM_PREFETCH']=str(int(args.gemm_prefetch))
        env['GOLEM_MPI_RANKS']=str(args.mpi_ranks)
        env['GOLEM_MPI_ARGS']=f'--bind-to core --cpu-set {args.cpu_set or "0-"+str(args.mpi_ranks-1)} --output-filename {case/"mpi_rank_output"}'
        if args.reference:env['GOLEM_PROJECTION_INPUT_REFERENCE']=str(args.reference.resolve())
        command=[str(HERE/'run_sst.sh'),'--projection','--query-length','1024','--kv-length','1024',
            '--num-query-heads',str(heads[0]),'--num-kv-heads',str(heads[1]),'--head-dim','64',
            '--causal','--rope','--artifact-root',str(case)]
        source_paths=[ROOT/path for path in (
            'src/sst/elements/golem/attention/decoderOpAbi.h',
            'src/sst/elements/golem/attention/decoderMemoryLayout.h',
            'src/sst/elements/golem/attention/decoderOpJob.h',
            'src/sst/elements/golem/rocc/roccAnalog.h',
            'src/sst/elements/golem/sfu/vector_math.h',
            'src/sst/elements/golem/sfu/sfu.cc',
            'test/golem/small/muticore_attention/golem_decoder_runtime.h',
            'test/golem/small/muticore_attention/golem_attention_runtime.cpp',
            'test/golem/tools/gen_hbm_init.py',
            'test/golem/architecture/ramulator/hbm2e_2500.yaml',
            'test/attention_cluster_8qk_8pv/run_decoder_layer.py')]
        source_hashes={str(path.relative_to(ROOT)):digest(path) for path in source_paths}
        for source in source_paths:
            target=case/'source_snapshot'/source.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        (case/'decoder_provenance.json').write_text(json.dumps({'command':command,'guest_sha256':digest(guest),
            'source_sha256':source_hashes,
            'sst_library_sha256':digest(ROOT/'install/lib/sst-elements-library/libgolem.so'),
            'environment':{k:v for k,v in env.items() if k.startswith(('GOLEM_','VANADIS_'))}},indent=2)+'\n')
        print(f'Running {args.checkpoint} {args.shape}',flush=True)
        with (case/'driver.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    result={'status':'PASS','checkpoint':args.checkpoint,'shape':{'S':1024,'hidden':hidden,'intermediate':4*hidden},
            'numerics':verify(case,hidden,heads,stage),'stages':stage_cycles(case),
            'dma_timing':dma_cycles(case,hidden,args.checkpoint),
            'timing_note':'Stage spans overlap across workers; do not sum. Serialized per worker baseline.'}
    projection=json.loads((case/'projection_e2e_result.json').read_text())
    result['legacy_attention_job_to_worker_cycles']=projection['end_to_end_cycles']
    spans=result['stages'];result['post_attention_span_cycles']=max(v['end_cycle'] for v in spans.values())-min(v['start_cycle'] for v in spans.values())
    (case/'decoder_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
