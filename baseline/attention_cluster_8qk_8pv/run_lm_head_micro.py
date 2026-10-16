#!/usr/bin/env python3
"""Isolated SST LM head: screen N parallelism before full-stack acceptance."""
import argparse
from bisect import bisect_right
import json
from pathlib import Path
import re
import subprocess
import time
import numpy as np
from run_decoder_micro import configured_environment, save, TESTS
from run_decoder_stack import memory_layout, sparse_weight
from run_decoder_layer import GUEST_DIR, linear_reference
from run_projection_overlap import ROOT, digest
from merge_mpi_rank_logs import merge
from artifact_storage import preflight, finalize
from verify_decoder_dataflow_resources import verify as verify_resources, FIELDS
from analyze_resource_timeline import merged


def prepare(root, hidden, vocab):
    layout=memory_layout(hidden//64,2 if hidden==256 else 8,final_norm=True,vocab=vocab,node_bytes=256*2**20)
    columns=(vocab+63)//64*64
    rng=np.random.default_rng(20261022)
    x=rng.normal(0,.3,(1,hidden)).astype('<f2');np.save(root/'input.npy',x)
    w=sparse_weight(rng,columns,hidden,.15);w[vocab:]=0;np.save(root/'weight.npy',w)
    np.save(root/'reference.npy',linear_reference(x,w).reshape(-1))
    tiles=w.reshape(columns//64,64,hidden//64,64).transpose(0,2,1,3).copy()
    preloads=[]
    def add(name,node,offset,data):
        path=root/f'{name}_node{node}.bin';data.tofile(path)
        preloads.append(dict(node=node,offset=offset,path=str(path),bytes=path.stat().st_size,sha256=digest(path)))
    add('input',4,layout['final_norm']['output']+255*hidden*2,x)
    for node in range(1,5):
        add('weight',node,layout['lm_head']['weight'],tiles[node-1::4])
        add('poison_output',node,layout['lm_head']['output'],np.full(columns,np.nan,dtype='<f2'))
        add('poison_done',node,layout['lm_head']['done'],np.full(4,0xfeedface,dtype='<u8'))
    m=dict(hidden=hidden,vocab=vocab,columns=columns,layout=layout,preloads=preloads,
           weight_kind='sparse synthetic values; every dense hardware tile executes')
    save(root/'inputs.json',m);return m


def audit(case,inputs,m,workers,parallel):
    log=case/'logs/attention-sst.log';jobs=[];flows=[];arrays={}
    for line in log.open():
        if '[DECODER_OP] event=done ' in line:jobs.append({k:int(v) for k,v in FIELDS.findall(line)})
        if '[DECODER_LM_FLOW]' in line:flows.append({k:int(v) for k,v in FIELDS.findall(line)})
        if 'GOLEM_ARRAY_BUFFER_STATS' in line:
            f={k:int(v) for k,v in FIELDS.findall(line)};assert f['core'] not in arrays;arrays[f['core']]=f
    chunks=(m['columns']+8191)//8192
    keys={(core,4+chunk) for core in range(4,4+workers) for chunk in range(chunks)}
    assert len(jobs)==len(keys) and {(j['core'],j['epoch']) for j in jobs}==keys
    assert len(flows)==len(keys) and {(f['core'],f['epoch']) for f in flows}==keys
    per_core={c:sorted([j for j in jobs if j['core']==c],key=lambda j:j['start_tick']) for c in range(4,4+workers)}
    for js in per_core.values():
        assert all(a['tick']<b['start_tick'] for a,b in zip(js,js[1:]))
    starts={c:[j['start_tick'] for j in js] for c,js in per_core.items()}
    states={key:dict(inputs=[],weights=[],writes=[],flags=[],last_output_done=0) for key in keys}
    stride=m['layout']['node_stride'];lm=m['layout']['lm_head'];h=m['hidden']
    input_address=4*stride+m['layout']['final_norm']['output']+255*h*2
    first=None;last=0
    for line in log.with_suffix('.resources.jsonl').open():
        read='resource=dma_read_transport ' in line
        if not read and 'resource=dma_write_transport ' not in line:continue
        e={k:int(v) for k,v in FIELDS.findall(line)};c=e['core']
        if c not in starts:continue
        index=bisect_right(starts[c],e['start_tick'])-1
        if index<0:continue
        j=per_core[c][index];s=states[c,j['epoch']]
        assert j['start_tick']<=e['start_tick']<=e['done_tick']<=j['tick'] and e['ok']==1
        interval=(e['address'],e['address']+e['bytes'])
        if read:
            if e['address']==input_address and e['bytes']==h*2:
                s['inputs'].append(interval);first=e['start_tick'] if first is None else min(first,e['start_tick'])
            else:s['weights'].append(interval)
        else:
            flag=(1+(c-4)%4)*stride+lm['done']+(c-4)//4*8
            if e['address']==flag:
                assert e['bytes']==8;s['flags'].append(e)
            else:s['writes'].append(interval);last=max(last,e['done_tick']);s['last_output_done']=max(s['last_output_done'],e['done_tick'])
    flowmap={(f['core'],f['epoch']):f for f in flows}
    tiles_total=0;launches_total=0
    for j in jobs:
        c=j['core'];chunk=j['epoch']-4;s=states[c,j['epoch']]
        begin=chunk*128;end=min(begin+128,m['columns']//64)
        tiles=list(range(begin+c-4,end,16));tiles_total+=len(tiles)
        want=[((1+t%4)*stride+lm['weight']+(t//4*(h//64)+k)*8192,
               (1+t%4)*stride+lm['weight']+(t//4*(h//64)+k+1)*8192) for t in tiles for k in range(h//64)]
        writes=[((1+(c-4)%4)*stride+lm['output']+t*128,(1+(c-4)%4)*stride+lm['output']+(t+1)*128) for t in tiles]
        for name,expected in [('weights',want),('writes',writes)]:
            assert merged(s[name])==merged(expected) and sum(b-a for a,b in s[name])==sum(b-a for a,b in expected),(c,chunk,name)
        assert len(s['inputs'])==bool(tiles) and len(s['flags'])==(chunk==chunks-1)
        assert j['op']==1 and j['stage']==11 and j['status']==0
        assert j['hbm_read']==len(tiles)*h*128+bool(tiles)*h*2
        assert j['hbm_write']==len(tiles)*128+(8 if chunk==chunks-1 else 0)
        for ack in s['flags']:assert s['last_output_done']<=ack['start_tick']<=ack['done_tick']<=j['tick']
        f=flowmap[c,j['epoch']];assert f['n_parallel']==parallel
        expected_launches=((len(tiles)+parallel-1)//parallel)*(h//64) if parallel>1 else 0
        assert f['launches']==expected_launches and f['array_k_tiles']==(len(tiles)*(h//64) if parallel>1 else 0)
        launches_total+=expected_launches
    stdout='\n'.join(p.read_text() for p in (case/'stdout').rglob('stdout-*'))
    completions=re.findall(r'LM_MICRO core=(\d+) status=(\d+) workers=(\d+) hidden=(\d+) vocab=(\d+)',stdout)
    assert len(completions)==workers and set(completions)=={(str(c),'0',str(workers),str(h),str(m['vocab'])) for c in range(4,4+workers)}
    reference=np.load(inputs/'reference.npy');logits=np.full(m['columns'],np.nan,dtype='<f2');owned=[]
    for node in range(1,5):
        with (case/'hbm'/f'hbm_out_node{node}.bin').open('rb') as stream:
            stream.seek(lm['output']);data=np.frombuffer(stream.read(m['columns']*2),dtype='<f2')
            for tile in range(node-1,m['columns']//64,4):
                if tile%16<workers:
                    logits[tile*64:(tile+1)*64]=data[tile*64:(tile+1)*64];owned.extend(range(tile*64,(tile+1)*64))
            stream.seek(lm['done']);flags=np.frombuffer(stream.read(32),dtype='<u8')
            for lane in range(4):assert flags[lane]==(3+chunks if node-1+lane*4<workers else 0xfeedface)
    assert np.array_equal(logits[owned].view('<u2'),reference[owned].view('<u2')) and np.isfinite(logits[owned]).all()
    np.save(case/'owned_logits.npy',logits[owned])
    assert set(arrays)==set(range(4,20))
    active_arrays={c:arrays[c] for c in range(4,4+workers)}
    assert sum(a['matrix_broadcast_requests'] for a in active_arrays.values())==tiles_total*(h//64)
    assert sum(a['matrix_broadcast_ingress_bytes'] for a in active_arrays.values())==tiles_total*(h//64)*8192
    assert sum(a['input_scatter_bytes'] for a in active_arrays.values())==tiles_total*(h//64)*128
    return dict(status='PASS',cycles=(last-first)/1000,first_input_tick=first,last_output_ack_tick=last,
                jobs=len(jobs),dense_weight_tiles=tiles_total*(h//64),group_launches=launches_total,
                hbm_read_bytes=sum(j['hbm_read'] for j in jobs),hbm_write_bytes=sum(j['hbm_write'] for j in jobs),
                fp16_bitwise_equal=True,verified_elements=len(owned),array_stats=active_arrays)


def lm_environment(case,args,m,parallel):
    env=configured_environment(case,args,0,16)
    env.update(GOLEM_MEM_NODE_SIZE_BYTES=str(256*2**20),VANADIS_EXE=str(GUEST_DIR/'riscv64/decoder-lm-micro/lm'),
               GOLEM_DECODER_MICRO_CONFIG=f'{args.workers},{m["hidden"]},{m["vocab"]},256',GOLEM_DECODER_LM_N_PARALLEL=str(parallel))
    return env


def run(case,inputs,m,args,parallel):
    preflight(case)
    case.mkdir()
    for d in ('hbm','logs','work','stdout'):(case/d).mkdir()
    for node in range(1,5):
        with (case/'hbm'/f'hbm_init_node{node}.bin').open('wb') as stream:
            stream.truncate(m['layout']['node_stride'])
            for e in m['preloads']:
                if e['node']!=node:continue
                p=Path(e['path']);assert p.stat().st_size==e['bytes'] and digest(p)==e['sha256']
                stream.seek(e['offset']);stream.write(p.read_bytes())
    env=lm_environment(case,args,m,parallel)
    command=[env['SST_CORE_PREFIX']+'/bin/sst','--lib-path='+str(ROOT/'install/lib/sst-elements-library'),'--no-env-config','--num-threads=1']
    if args.mpi_ranks>1:
        command+=['--partitioner=sst.self']
        command=['mpirun','--bind-to','core','--cpu-set',args.cpu_set or f'0-{args.mpi_ranks-1}',
                 '--output-filename',str(case/'mpi_rank_output'),'-np',str(args.mpi_ranks)]+command
    elif args.cpu_set:command=['taskset','-c',args.cpu_set]+command
    command.append(str(TESTS/'architecture/ncores_selfcom_dma_ctrl.py'))
    sources=[Path(__file__).resolve(),Path(__file__).with_name('artifact_storage.py'),GUEST_DIR/'golem_decoder_lm_micro.cpp',GUEST_DIR/'golem_decoder_runtime.h',
             ROOT/'src/sst/elements/golem/attention/decoderOpJob.h',ROOT/'src/sst/elements/golem/attention/decoderMemoryLayout.h',
             TESTS/'architecture/ncores_selfcom_dma_ctrl.py',TESTS/'architecture/cpu_builder.py',TESTS/'architecture/ramulator/hbm2e_2500.yaml']+sorted((TESTS/'configs').glob('*.env'))
    prov=dict(command=command,guest_sha256=digest(Path(env['VANADIS_EXE'])),sst_library_sha256=digest(ROOT/'install/lib/sst-elements-library/libgolem.so'),
              input_sha256=digest(inputs/'inputs.json'),source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},
              environment={k:v for k,v in env.items() if k.startswith(('GOLEM_','VANADIS_'))})
    save(case/'provenance.json',prov);log=case/'logs/attention-sst.log';start=time.perf_counter()
    with log.open('w') as stream:subprocess.run(command,cwd=case/'work',env=env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=args.timeout)
    wall=time.perf_counter()-start
    for pattern in ('stdout-*','stderr-*'):
        for p in (case/'work').glob(pattern):p.rename(case/'stdout'/p.name)
    if args.mpi_ranks>1:merge(case/'mpi_rank_output',log,args.mpi_ranks)
    else:
        log.rename(log.with_suffix('.raw'))
        with log.with_suffix('.raw').open() as source,log.open('w') as main,log.with_suffix('.resources.jsonl').open('w') as resources:
            for line in source:(resources if '[RESOURCE_INTERVAL]' in line else main).write(line)
    r=dict(status='PASS',parallel=parallel,shape=dict(hidden=m['hidden'],vocab=m['vocab'],workers=args.workers),
           measurement=audit(case,inputs,m,args.workers,parallel),resources=verify_resources(case),sst_wall_seconds=wall,provenance=prov)
    save(case/'result.json',r)
    finalize(case)
    return r


def compare(results):
    base=results['1'];ignored={'GOLEM_ARTIFACT_ROOT','GOLEM_HBM_DIR','GOLEM_STDOUT_DIR','GOLEM_STATS_DIR','GOLEM_MPI_RANK_LOG_ROOT','GOLEM_MPI_ARGS','GOLEM_DECODER_LM_N_PARALLEL'}
    def hardware(r):return {k:v for k,v in r['provenance']['environment'].items() if k not in ignored}
    for r in results.values():
        assert r['shape']==base['shape'] and hardware(r)==hardware(base)
        def placement(result):
            prov=result['provenance'];root=prov['environment'].get('GOLEM_ARTIFACT_ROOT','')
            return [word.replace(root,'<case>') if root else word for word in prov.get('command',[])]
        assert placement(r)==placement(base), 'execution placement changed'
        for k in ('input_sha256','source_sha256','guest_sha256','sst_library_sha256'):assert r['provenance'][k]==base['provenance'][k],k
        for k in ('dense_weight_tiles','hbm_read_bytes','hbm_write_bytes','verified_elements'):assert r['measurement'][k]==base['measurement'][k]
    return dict(status='PASS',shape=base['shape'],cases={k:dict(cycles=r['measurement']['cycles'],saved_percent=100*(1-r['measurement']['cycles']/base['measurement']['cycles']),
               sst_wall_seconds=r['sst_wall_seconds']) for k,r in results.items()},scope='isolated first last-token input DMA to final logits ACK; full-stack acceptance required')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact-root',type=Path,required=True)
    p.add_argument('--shape',choices=('small','formal'),default='small');p.add_argument('--workers',type=int,choices=(1,16),default=1)
    p.add_argument('--mpi-ranks',type=int,choices=(1,4,8),default=1);p.add_argument('--cpu-set');p.add_argument('--vocab',type=int,default=128256)
    p.add_argument('--parallel',type=int,choices=(1,2,4,8),nargs='+',default=[1,2,4,8]);p.add_argument('--timeout',type=int,default=900)
    p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    if len(set(a.parallel))!=len(a.parallel) or 1 not in a.parallel:p.error('unique parallel choices including baseline 1 required')
    if not 1<=a.vocab<=128256:p.error('vocab must be in [1,128256]')
    a.no_prefetch=False;a.prefetch_slots=4;a.weight_trace=True;a.with_down=False;a.with_residual=False
    root=a.artifact_root.resolve();inputs=root/'inputs';results={}
    if a.verify_only:
        m=json.loads((inputs/'inputs.json').read_text())
        for n in a.parallel:
            case=root/f'n{n}';r=json.loads((case/'result.json').read_text())
            r['measurement']=audit(case,inputs,m,r['shape']['workers'],n);r['resources']=verify_resources(case);results[str(n)]=r
    else:
        preflight(root)
        root.mkdir(parents=True,exist_ok=False);inputs.mkdir();m=prepare(inputs,2048 if a.shape=='formal' else 256,a.vocab)
        subprocess.run(['make','-C',str(GUEST_DIR),'decoder-lm-micro'],check=True)
        for n in a.parallel:
            print(f'Running LM local N={n}, {a.shape}, workers={a.workers}',flush=True)
            results[str(n)]=run(root/f'n{n}',inputs,m,a,n)
            save(root/'summary.json',compare(results));print(json.dumps(compare(results)['cases'][str(n)]),flush=True)
    summary=compare(results);save(root/'summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
