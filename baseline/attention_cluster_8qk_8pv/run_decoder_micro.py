#!/usr/bin/env python3
"""Real SST FFN screening: shared inputs/hardware, independent numerics and DMA ACK audit."""
import argparse
from bisect import bisect_right
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
import numpy as np

from run_batch_b import environment
from run_decoder_layer import GUEST_DIR, linear_reference, storage_half
from run_decoder_stack import memory_layout, sparse_weight, scalar_silu_reference
from run_projection_overlap import ROOT, digest
from merge_mpi_rank_logs import merge
from artifact_storage import preflight, finalize
from verify_decoder_dataflow_resources import verify as verify_resources, residual_audit, FIELDS

TESTS=ROOT/'src/sst/elements/golem/tests'
VARIANTS={'baseline':(0,16),'fused16':(1,16),'fused8':(1,8),'down16':(2,16),'down8':(2,8)}
VARIANTS.update({f'down16q{slots}':(2,16) for slots in (2,4,8)})
VARIANTS['residual16q4']=(3,16)
VARIANTS['groups2residual16q4']=(3,16)
for name in ('layout1','layout2','layout1reuse','layout2reuse'):
    VARIANTS[name]=(3,16)


def save(path, data):
    path.write_text(json.dumps(data,indent=2)+'\n')


def prepare(root, hidden):
    layout=memory_layout(hidden//64,2 if hidden==256 else 8)
    layer=layout['layers'][0]
    rng=np.random.default_rng(20261021)
    x=rng.normal(0,.3,(4,256,hidden)).astype('<f2')
    np.save(root/'input.npy',x)
    residual=rng.normal(0,.2,x.shape).astype('<f2');np.save(root/'residual.npy',residual)
    preloads=[]
    def add(name, node, offset, data):
        path=root/f'{name}_node{node}.bin';data.tofile(path)
        preloads.append({'name':name,'node':node,'offset':offset,'path':str(path),
                         'bytes':path.stat().st_size,'sha256':digest(path)})
    for node in range(1,5):
        add('input',node,layout['shared']['norm2'],x[node-1])
        add('residual',node,layout['shared']['h1'],residual[node-1])
    for name,n,k in [('gate',4*hidden,hidden),('up',4*hidden,hidden),('down',hidden,4*hidden)]:
        w=sparse_weight(rng,n,k,.25);np.save(root/f'{name}.npy',w)
        tiles=w.reshape(n//64,64,k//64,64).transpose(0,2,1,3).copy()
        for node in range(1,5):add(name,node,layer[name+'_weight'],tiles[node-1::4])
    for name in ('gate','up','gated','down'):
        width=hidden if name=='down' else 4*hidden
        for node in range(1,5):add('poison_'+name,node,layout['shared'][name],np.full((256,width),np.nan,dtype='<f2'))
    manifest={'hidden':hidden,'layout':layout,'preloads':preloads,
              'weight_kind':'sparse synthetic; every dense hardware tile executes'}
    save(root/'inputs.json',manifest)
    return manifest


def configured_environment(case, args, fused, panel):
    env=environment(case,(1,1,1),'none',1024,args.shape)
    env.update({'GOLEM_LOCAL_GM_MODEL':getattr(args, 'gm_model', 'legacy'),
        'GOLEM_LOCAL_GM_RESPONSE_CYCLES':str(getattr(args, 'gm_response_cycles', 1)),
        'GOLEM_LOCAL_GM_RESPONSE_CAPACITY':'32'})
    env.update({
        'VANADIS_NUM_CORES':'20','GOLEM_TOTAL_CORES':'20','GOLEM_TOTAL_GEMM_CORES':'20',
        'GOLEM_GLOBAL_STRIDE_KB':'2048','GOLEM_MATMUL_DTYPE':'fp16',
        'GOLEM_ROCC_TYPE':'golem.RoCCAnalogFloat','GOLEM_ARRAY_TYPE':'golem.MVMFloatArray',
        'GOLEM_SFU_ENABLE':'1','GOLEM_ARRAY_OPERAND_CONTEXT_BANKS':'1',
        'GOLEM_ARRAY_OUTPUT_READ_CREDITS':'1','GOLEM_ARRAY_OUTPUT_READ_BANKS':'1',
        'GOLEM_ARRAY_BUFFER_PORTS':'1','GOLEM_ARRAY_BUFFER_BASE_LATENCY_CYCLES':'1',
        'GOLEM_MATRIX_BROADCAST_MAX_FANOUT':'64','GOLEM_ATTENTION_KV_BUFFER_COUNT':'1',
        'GOLEM_ATTENTION_KV_SECOND_LOOKAHEAD':'0','GOLEM_ATTENTION_GENERIC_GEMM_ENABLE':'1',
        'GOLEM_NOC_INPUT_BUF_SIZE':'128KB','GOLEM_NOC_OUTPUT_BUF_SIZE':'128KB',
        'GOLEM_GM_BUFFER_LENGTH':'128KB','GOLEM_PROGRESS_HEARTBEAT':'0',
        'GOLEM_DMA_READ_RETRY_TICKS':'4096','GOLEM_DMA_READ_MAX_RETRIES':'32',
        'GOLEM_WORKER_START_BARRIER_ENABLE':'0','GOLEM_WORKER_START_GUARD_CYCLES':'0',
        'GOLEM_MEM_NODE_SIZE_BYTES':str(128*2**20),'GOLEM_MEMORY_BACKEND':'ramulator2',
        'GOLEM_RAMULATOR2_CONFIG':str(TESTS/'architecture/ramulator/hbm2e_2500.yaml'),
        'GOLEM_ARTIFACT_ROOT':str(case),'GOLEM_HBM_DIR':str(case/'hbm'),
        'GOLEM_STDOUT_DIR':str(case/'stdout'),'GOLEM_STATS_DIR':str(case/'stats'),
        'GOLEM_MPI_RANKS':str(args.mpi_ranks),'GOLEM_MPI_PARTITIONING':str(int(args.mpi_ranks>1)),
        'VANADIS_EXE':str(GUEST_DIR/'riscv64/decoder-micro/ffn'),
        'GOLEM_RESOURCE_TRACE_DOMAINS':'gm','GOLEM_CRITICAL_PATH_TRACE':'1',
        'GOLEM_DECODER_MICRO_CONFIG':f'{args.workers},{256 if args.workers==16 else 64},{2048 if args.shape=="formal" else 256},{int(fused)},{int(args.with_down)},{int(getattr(args,"with_residual",False))}',
        'GOLEM_DECODER_GEMM_PREFETCH':str(int(not args.no_prefetch)),
        'GOLEM_DECODER_OUTPUT_PANEL_TILES':'16','GOLEM_DECODER_OUTPUT_WRITE_WINDOW':'8',
        'GOLEM_DECODER_FFN_PANEL_TILES':str(panel),
        'GOLEM_DECODER_PREFETCH_SLOTS':str(getattr(args,'prefetch_slots',2)),
        'GOLEM_DECODER_WEIGHT_TRACE':str(int(getattr(args,'weight_trace',False))),
        'GOLEM_DECODER_ARRAY_GROUPS':'1',
    })
    # Use the same versioned hardware presets as the full simulator wrapper.
    # Capture in memory; inherited credentials/environment are never logged.
    command='\n'.join([
        'source '+shlex.quote(str(ROOT/'scripts/env_local_install.sh'))+' >/dev/null',
        'source '+shlex.quote(str(TESTS/'configs/default.env')),
        'env -0'])
    raw=subprocess.run(['bash','-c',command],env=env,check=True,capture_output=True).stdout
    configured=dict(item.decode().split('=',1) for item in raw.split(b'\0') if item)
    # memHierarchy links DRAMSim3 even when the selected backend is Ramulator2.
    configured['LD_LIBRARY_PATH']=configured['SST_DRAMSIM3_PREFIX']+':'+configured.get('LD_LIBRARY_PATH','')
    return configured


def active_rows(workers):
    return {node:(256 if workers==16 else 64) for node in range(1,5 if workers>1 else 2)}


def read_output(case, manifest, name, workers):
    width=manifest['hidden'] if name=='down' else 4*manifest['hidden']
    result=[]
    for node,rows in active_rows(workers).items():
        with (case/'hbm'/f'hbm_out_node{node}.bin').open('rb') as stream:
            stream.seek(manifest['layout']['shared'][name]);data=stream.read(rows*width*2)
        if len(data)!=rows*width*2:raise ValueError('short HBM output')
        result.append(np.frombuffer(data,dtype='<f2').reshape(rows,width))
    return np.concatenate(result)


def check_numerics(case, inputs, m, workers, down, fused=0, residual=False):
    x=np.load(inputs/'input.npy')
    x=np.concatenate([x[node-1,:rows] for node,rows in active_rows(workers).items()])
    gate=linear_reference(x,np.load(inputs/'gate.npy'))
    up=linear_reference(x,np.load(inputs/'up.npy'))
    gated=storage_half(scalar_silu_reference(gate).astype('f4')*up.astype('f4'))
    expected={} if fused>=2 else {'gated':gated}
    if down:
        expected['down']=linear_reference(gated,np.load(inputs/'down.npy'))
        if residual:
            h=np.load(inputs/'residual.npy');h=np.concatenate([h[node-1,:rows] for node,rows in active_rows(workers).items()])
            expected['down']=storage_half(expected['down'].astype('f4')+h.astype('f4'))
    report={}
    for name,want in expected.items():
        got=read_output(case,m,name,workers)
        if not np.isfinite(got).all() or not np.array_equal(got.view('<u2'),want.view('<u2')):
            raise ValueError(f'{name}: output differs from independent FP16 oracle')
        report[name]={'status':'PASS','elements':int(got.size),'fp16_bitwise_equal':True}
    return report


def timing(case, m, workers, fused, down, residual=False):
    log=case/'logs/attention-sst.log'
    jobs=[]
    for line in log.read_text().splitlines():
        if '[DECODER_OP] event=done' in line:jobs.append({k:int(v) for k,v in FIELDS.findall(line)})
    expected_stages={4:9} if fused>=2 else {4:8} if fused else {4:1,5:1,7:7}
    if down and fused<2:expected_stages[8]=1
    if residual and fused!=3:expected_stages[9]=2
    actual={(j['core'],j['stage']):j for j in jobs}
    if len(actual)!=len(jobs) or set(actual)!={(core,stage) for core in range(4,4+workers) for stage in expected_stages}:
        raise ValueError('missing/duplicate/unexpected decoder job')
    stdout='\n'.join(path.read_text(errors='replace') for path in (case/'stdout').rglob('stdout-*'))
    completions=re.findall(r'DECODER_MICRO core=(\d+) status=(\d+) workers=(\d+) rows=(\d+) hidden=(\d+) fused=(\d+) down=(\d+)',stdout)
    expected_completions={(str(core),'0',str(workers),str(256 if workers==16 else 64),str(m['hidden']),str(int(fused)),str(int(down))) for core in range(4,4+workers)}
    if len(completions)!=workers or set(completions)!=expected_completions:
        raise ValueError('missing/failed guest completion')
    hidden=m['hidden'];width=4*hidden;weights=hidden*width*2
    for j in jobs:
        stage=j['stage']
        if j['op']!=expected_stages[stage] or j['status']:raise ValueError('failed or wrong decoder op')
        expected_read={4:(3*weights+64*hidden*2 if fused>=2 else 2*weights+64*hidden*2 if fused else weights+64*hidden*2),
                       5:weights+64*hidden*2,7:2*64*width*2,8:weights+64*width*2,9:2*64*hidden*2}[stage]
        if fused==3 and stage==4:expected_read+=64*hidden*2
        expected_write=64*(hidden if stage in (8,9) or fused>=2 else width)*2
        if (j['hbm_read'],j['hbm_write'])!=(expected_read,expected_write):
            raise ValueError('missing dense weight/input traffic or extra output write')
    by_core={core:sorted([j for j in jobs if j['core']==core],key=lambda j:j['start_tick']) for core in range(4,4+workers)}
    starts={core:[j['start_tick'] for j in js] for core,js in by_core.items()}
    first=None;last=0;covered={};final_stage=9 if residual and fused!=3 else 4 if fused>=2 else 8 if down else 4 if fused else 7
    for node,rows in active_rows(workers).items():
        offset=m['layout']['shared']['down' if down else 'gated']
        base=node*m['layout']['node_stride']+offset
        size=rows*(hidden if down else width)*2
        covered[node]=(base,base+size,[])
    with log.with_suffix('.resources.jsonl').open() as stream:
        for line in stream:
            match=re.search(r'resource=(dma_read_transport|dma_write_transport)\b',line)
            if not match:continue
            e={k:int(v) for k,v in FIELDS.findall(line)}
            core=e['core']
            if core not in starts:continue
            index=bisect_right(starts[core],e['start_tick'])-1
            if index<0:continue
            j=by_core[core][index]
            if not j['start_tick']<=e['start_tick']<=e['done_tick']<=j['tick'] or e['ok']!=1:
                raise ValueError('failed DMA or ACK outside job boundary')
            input_base=(1+(core-4)%4)*m['layout']['node_stride']+m['layout']['shared']['norm2']
            if (match[1]=='dma_read_transport' and j['stage']==4 and
                input_base<=e['address']<e['address']+e['bytes']<=input_base+256*hidden*2):
                first=e['start_tick'] if first is None else min(first,e['start_tick'])
            if match[1]=='dma_write_transport' and j['stage']==final_stage:
                node=1+(core-4)%4;base,end,intervals=covered[node]
                address=e['address'];stop=address+e['bytes']
                if not base<=address<stop<=end:raise ValueError('unexpected final output address')
                intervals.append((address,stop));last=max(last,e['done_tick'])
    for base,end,intervals in covered.values():
        cursor=base
        for begin,stop in sorted(intervals):
            if begin!=cursor:raise ValueError('duplicate or incomplete final output ACK coverage')
            cursor=stop
        if cursor!=end:raise ValueError('missing final output ACK')
    if first is None or last<=first:raise ValueError('missing DMA timing')
    stages={str(stage):{'span_cycles':(max(j['tick'] for j in jobs if j['stage']==stage)-min(j['start_tick'] for j in jobs if j['stage']==stage))/1000,
        'hbm_read_bytes':sum(j['hbm_read'] for j in jobs if j['stage']==stage),
        'hbm_write_bytes':sum(j['hbm_write'] for j in jobs if j['stage']==stage)} for stage in expected_stages}
    residual_flow=residual_audit(case,{(core,1,4) for core in range(4,4+workers)},hidden) if fused==3 else None
    return {'first_data_dma_tick':first,'last_output_ack_tick':last,'cycles':(last-first)/1000,'residual_flow':residual_flow,
        'stages':stages,'hbm_read_bytes':sum(j['hbm_read'] for j in jobs),'hbm_write_bytes':sum(j['hbm_write'] for j in jobs),
        'contract':'first Norm2 input DMA send to final gated/down output ACK; includes dispatch gaps; excludes CPU startup/exit'}


def down_partial_audit(case, hidden, workers, panel, reuse=False):
    flows=[]
    for line in (case/'logs/attention-sst.log').read_text().splitlines():
        if '[DECODER_DOWN_FLOW]' in line:flows.append({k:int(v) for k,v in FIELDS.findall(line)})
    panels=(4*hidden//64+panel-1)//panel;partial_bytes=64*hidden*2
    if len(flows)!=workers or {e['core'] for e in flows}!=set(range(4,4+workers)):
        raise ValueError('missing/duplicate down partial lifecycle')
    passes=4*hidden//64 if reuse else panels
    for e in flows:
        if (e['panels_consumed']!=panels or e['cbuffer_bytes']!=partial_bytes or partial_bytes>2**20 or
            e['cbuffer_read_bytes']!=(passes-1)*partial_bytes or e['cbuffer_write_bytes']!=(passes-1)*partial_bytes or
            e['restores']!=(passes-1)*hidden//64):raise ValueError('down partial SRAM traffic or restoration mismatch')
    return {'status':'PASS','workers':workers,'panels_per_worker':panels,'cbuffer_capacity_bytes':2**20,
            'required_per_worker_bytes':partial_bytes,'input_reuse':reuse,'flows':flows}


def sfu_work(case):
    work={}
    for line in (case/'logs/attention-sst.log').read_text().splitlines():
        match=re.search(r'GOLEM_SFU_HW_PIPELINE core=(\d+) unit=(exp|reciprocal) .*accepted_tokens=(\d+)',line)
        if match:
            core,unit,tokens=match.groups();key=core+'.'+unit
            if key in work:raise ValueError('duplicate SFU statistics')
            work[key]=int(tokens)
    if set(work)!={str(core)+'.'+unit for core in range(4,20) for unit in ('exp','reciprocal')}:
        raise ValueError('missing actual SFU EXP/reciprocal statistics')
    return work


def group_audit(case, hidden, workers, groups, reuse=False):
    flows=[]
    for line in (case/'logs/attention-sst.log').read_text().splitlines():
        if '[DECODER_GROUP_FLOW]' in line:flows.append({k:int(v) for k,v in FIELDS.findall(line)})
    if len(flows)!=workers or {e['core'] for e in flows}!=set(range(4,4+workers)):
        raise ValueError('missing/duplicate array group lifecycle')
    tiles=(2 if reuse else 3)*(hidden//64)*(4*hidden//64)
    for e in flows:
        if e['groups']!=groups or e['launches']!=(2*tiles if groups==2 else 0):
            raise ValueError('array subset launches missing/duplicated')
        if groups==2 and (e['compute_array_cycles']!=tiles*64*67 or
                          e['program_cycles']!=2*tiles*39):
            raise ValueError('array subset MAC or broadcast service changed')
    return {'status':'PASS','groups':groups,'full_row_tile_count_per_worker':tiles,
            'scope':'64 existing arrays, one context each; MAC callbacks include one WCP cycle',
            'flows':flows}


def run(case, inputs, m, args, variant):
    fused,panel=VARIANTS[variant]
    preflight(case)
    case.mkdir();(case/'hbm').mkdir();(case/'logs').mkdir();(case/'work').mkdir();(case/'stdout').mkdir()
    # mmap backing is sparse and only 128 MiB/node; no unused model weights.
    for node in range(1,5):
        with (case/'hbm'/f'hbm_init_node{node}.bin').open('wb') as stream:
            stream.truncate(m['layout']['node_stride'])
            for entry in m['preloads']:
                if entry['node']!=node:continue
                source=Path(entry['path'])
                if source.stat().st_size!=entry['bytes'] or digest(source)!=entry['sha256']:raise ValueError('preload changed')
                stream.seek(entry['offset']);stream.write(source.read_bytes())
    env=configured_environment(case,args,fused,panel)
    if variant.startswith(('down16q','residual16q')):env['GOLEM_DECODER_PREFETCH_SLOTS']=variant[-1]
    if variant in ('groups2residual16q4','layout1','layout2','layout1reuse','layout2reuse'):
        env['GOLEM_DECODER_PREFETCH_SLOTS']='4'
        env['GOLEM_DECODER_ARRAY_GROUPS']='2'
    env['GOLEM_DECODER_FFN_LAYOUT']=str(1 if variant.startswith('layout1') else 2 if variant.startswith('layout2') else 0)
    env['GOLEM_DECODER_DOWN_INPUT_REUSE']=str(int(variant.endswith('reuse')))
    command=[env['SST_CORE_PREFIX']+'/bin/sst','--lib-path='+str(ROOT/'install/lib/sst-elements-library'),'--no-env-config','--num-threads=1']
    if args.mpi_ranks>1:
        command+=['--partitioner=sst.self']
        command=['mpirun','--bind-to','core','--cpu-set',args.cpu_set or f'0-{args.mpi_ranks-1}',
                 '--output-filename',str(case/'mpi_rank_output'),'-np',str(args.mpi_ranks)]+command
    elif args.cpu_set:command=['taskset','-c',args.cpu_set]+command
    command.append(str(TESTS/'architecture/ncores_selfcom_dma_ctrl.py'))
    sources=[Path(__file__).resolve(),Path(__file__).with_name('artifact_storage.py'),GUEST_DIR/'golem_decoder_micro.cpp',GUEST_DIR/'golem_decoder_runtime.h',
             ROOT/'src/sst/elements/golem/attention/decoderOpJob.h',ROOT/'src/sst/elements/golem/attention/decoderMemoryLayout.h',
             ROOT/'src/sst/elements/golem/sfu/vector_math.h',ROOT/'src/sst/elements/golem/sfu/vector_pipeline.h',
             ROOT/'src/sst/elements/golem/sfu/hardware_pipeline.h',ROOT/'src/sst/elements/golem/fp16.h',
             TESTS/'architecture/ncores_selfcom_dma_ctrl.py',TESTS/'architecture/cpu_builder.py',
             TESTS/'architecture/ramulator/hbm2e_2500.yaml']+sorted((TESTS/'configs').glob('*.env'))
    sources.extend(ROOT/'src/sst/elements/golem/globalmemory'/name for name in
                   ('globalmemory.cc','globalmemory.h','localGmBankScheduler.h'))
    provenance={'command':command,'guest_sha256':digest(Path(env['VANADIS_EXE'])),
        'sst_library_sha256':digest(ROOT/'install/lib/sst-elements-library/libgolem.so'),
        'sst_memhierarchy_sha256':digest(ROOT/'install/lib/sst-elements-library/libmemHierarchy.so'),
        'input_sha256':digest(inputs/'inputs.json'),
        'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in sources},
        'environment':{k:v for k,v in env.items() if k.startswith(('GOLEM_','VANADIS_'))}}
    save(case/'provenance.json',provenance)
    log=case/'logs/attention-sst.log'
    start=time.perf_counter()
    with log.open('w') as stream:subprocess.run(command,cwd=case/'work',env=env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=args.timeout)
    wall=time.perf_counter()-start
    for pattern in ('stdout-*','stderr-*'):
        for path in (case/'work').glob(pattern):path.rename(case/'stdout'/path.name)
    if args.mpi_ranks>1:merge(case/'mpi_rank_output',log,args.mpi_ranks)
    else:
        log.rename(log.with_suffix('.raw'))
        with log.with_suffix('.raw').open() as source,log.open('w') as main,log.with_suffix('.resources.jsonl').open('w') as resources:
            for line in source:(resources if '[RESOURCE_INTERVAL]' in line else main).write(line)
    result={'status':'PASS','variant':variant,'shape':{'hidden':m['hidden'],'workers':args.workers,
        'rows_per_node':256 if args.workers==16 else 64,'with_down':args.with_down,'with_residual':getattr(args,'with_residual',False)},
        'dma_timing':timing(case,m,args.workers,fused,args.with_down,getattr(args,'with_residual',False)),
        'numerics':check_numerics(case,inputs,m,args.workers,args.with_down,fused,getattr(args,'with_residual',False)),
        'resources':verify_resources(case),'sfu_work':sfu_work(case),'sst_wall_seconds':wall,
        'provenance':provenance,'wall_scope':'SST process including initialization/output dump; excludes input preparation and host validation'}
    if fused>=2:result['down_partial']=down_partial_audit(case,m['hidden'],args.workers,panel,variant.endswith('reuse'))
    if fused==3:result['array_groups']=group_audit(case,m['hidden'],args.workers,int(env['GOLEM_DECODER_ARRAY_GROUPS']),variant.endswith('reuse'))
    save(case/'result.json',result)
    finalize(case)
    return result


def compare(results, reference_variant='baseline'):
    base=results.get(reference_variant);reference=base or next(iter(results.values()))
    # Same input, build, modeled hardware and execution placement throughout.
    ignored={'GOLEM_ARTIFACT_ROOT','GOLEM_HBM_DIR','GOLEM_STDOUT_DIR','GOLEM_STATS_DIR',
             'GOLEM_MPI_RANK_LOG_ROOT','GOLEM_MPI_ARGS','GOLEM_DECODER_MICRO_CONFIG','GOLEM_DECODER_FFN_PANEL_TILES',
             'GOLEM_DECODER_PREFETCH_SLOTS','GOLEM_DECODER_ARRAY_GROUPS','GOLEM_DECODER_FFN_LAYOUT','GOLEM_DECODER_DOWN_INPUT_REUSE'}
    def hardware(r):return {k:v for k,v in r['provenance']['environment'].items() if k not in ignored}
    for r in results.values():
        if r['status']!='PASS' or r['shape']!=reference['shape']:raise ValueError('failed or unmatched shape')
        for key in ('guest_sha256','sst_library_sha256','source_sha256','input_sha256'):
            if r['provenance'][key]!=reference['provenance'][key]:raise ValueError('unmatched build or inputs')
        if hardware(r)!=hardware(reference):raise ValueError('unmatched hardware/environment')
        if r['provenance'].get('sst_memhierarchy_sha256')!=reference['provenance'].get('sst_memhierarchy_sha256'):raise ValueError('unmatched memory library')
        if r['sfu_work']!=reference['sfu_work']:raise ValueError('SFU EXP/reciprocal work changed')
    return {'status':'PASS','reference_variant':reference_variant,'shape':reference['shape'],'cases':{name:{'cycles':r['dma_timing']['cycles'],
        'sst_wall_seconds':r['sst_wall_seconds'],'hbm_read_bytes':r['dma_timing']['hbm_read_bytes'],
        'hbm_write_bytes':r['dma_timing']['hbm_write_bytes'],
        'saved_percent':100*(1-r['dma_timing']['cycles']/base['dma_timing']['cycles']) if base else None}
        for name,r in results.items()},'scope':'isolated FFN screening; full-model speedup requires end-to-end confirmation'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root',type=Path,required=True)
    p.add_argument('--gm-response-cycles',type=int,choices=(1,2,4),default=1)
    p.add_argument('--gm-model',choices=('legacy','banked'),default='banked',help='local GM timing contract; banked is reference-aligned candidate')
    p.add_argument('--shape',choices=('small','formal'),default='small')
    p.add_argument('--workers',type=int,choices=(1,4,16),default=1)
    p.add_argument('--mpi-ranks',type=int,choices=(1,4,8),default=1)
    p.add_argument('--cpu-set');p.add_argument('--with-down',action='store_true')
    p.add_argument('--with-residual',action='store_true',help='compare final down+residual outputs, including the standalone Add baseline')
    p.add_argument('--verify-only',action='store_true',help='recheck existing outputs/ACKs; do not simulate')
    p.add_argument('--no-prefetch',action='store_true')
    p.add_argument('--prefetch-slots',type=int,choices=(2,4,8),default=2)
    p.add_argument('--weight-trace',action='store_true')
    p.add_argument('--variants',choices=tuple(VARIANTS),nargs='+',default=['baseline','fused16','fused8'])
    p.add_argument('--reference-variant',choices=tuple(VARIANTS),default='baseline')
    p.add_argument('--timeout',type=int,default=900,help='maximum wall seconds per SST invocation')
    args=p.parse_args()
    if len(set(args.variants))!=len(args.variants):p.error('duplicate variants')
    if any(VARIANTS[name][0]>=2 for name in args.variants) and not (args.with_down or args.verify_only):p.error('down fusion requires --with-down')
    if not args.verify_only:
        if args.with_residual and not args.with_down:p.error('--with-residual requires --with-down')
        if any(VARIANTS[v][0]==3 for v in args.variants) and not args.with_residual:p.error('residual variants require --with-residual')
    root=args.artifact_root.resolve()
    if args.verify_only:
        inputs=root/'inputs';m=json.loads((inputs/'inputs.json').read_text());results={}
        for variant in args.variants:
            case=root/variant;r=json.loads((case/'result.json').read_text());fused,panel=VARIANTS[variant]
            shape=r['shape'];workers=shape['workers'];down=shape['with_down']
            residual=shape.get('with_residual',False)
            r['numerics']=check_numerics(case,inputs,m,workers,down,fused,residual)
            r['dma_timing']=timing(case,m,workers,fused,down,residual)
            r['resources']=verify_resources(case);r['sfu_work']=sfu_work(case)
            if fused>=2:r['down_partial']=down_partial_audit(case,m['hidden'],workers,panel,variant.endswith('reuse'))
            if fused==3:r['array_groups']=group_audit(case,m['hidden'],workers,int(r['provenance']['environment']['GOLEM_DECODER_ARRAY_GROUPS']),variant.endswith('reuse'))
            save(case/'result.json',r);results[variant]=r
        save(root/'summary.json',compare(results,args.reference_variant));print(json.dumps(compare(results,args.reference_variant),indent=2));return
    preflight(root)
    root.mkdir(parents=True,exist_ok=False)
    inputs=root/'inputs';inputs.mkdir();m=prepare(inputs,2048 if args.shape=='formal' else 256)
    subprocess.run(['make','-C',str(GUEST_DIR),'decoder-micro'],check=True)
    results={}
    for variant in args.variants:
        print(f'Running local FFN: {variant}, {args.shape}, workers={args.workers}',flush=True)
        results[variant]=run(root/variant,inputs,m,args,variant)
        save(root/'summary.json',compare(results,args.reference_variant))
        print(json.dumps(compare(results,args.reference_variant)['cases'][variant]),flush=True)


if __name__=='__main__':main()
