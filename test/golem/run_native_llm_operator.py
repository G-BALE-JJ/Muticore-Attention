#!/usr/bin/env python3
"""Run an explicitly selected LLM GEMM operator through real TE/NoC/HBM.

This is an operator experiment, not a replacement numerical contract for the
FP16 decoder. It keeps 16 workers, 64 arrays/worker and four HBM data nodes.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from tools.native_workload_plan import build_plan
from tools.native_llm_workload import MODES,digest,engine_parameters,load_config,write_payloads
ROOT=Path(__file__).resolve().parents[2]


def geometry(operator,shape,sequence,vocab):
    hidden,kv=(128,64) if shape=='small' else (2048,512)
    return {'qkv':(sequence,hidden+2*kv,hidden),'wo':(sequence,hidden,hidden),
            'gate':(sequence,4*hidden,hidden),'up':(sequence,4*hidden,hidden),
            'down':(sequence,hidden,4*hidden),'lm_head':(1,vocab,hidden)}[operator]


def prepare(args):
    logical=geometry(args.operator,args.shape,args.sequence,args.vocab)
    m,n,k=(((x+63)//64)*64 for x in logical)
    plan=build_plan(m_tiles=m//64,n_tiles=n//64,k_tiles=k//64,reuse_m=args.reuse,reuse_n=args.reuse,
        workers=16,groups=4,data_nodes=4,instances=args.instances,panel_mapping=args.panel_mapping,
        replicate_b=args.replicate_b,node_bytes=128*1024**2,elem_bytes=4,input_bits=MODES[args.precision][1])
    root=args.artifact_root.resolve();root.mkdir(parents=True,exist_ok=False)
    inputs=[]
    if bool(args.input_a)!=bool(args.input_b):raise ValueError('provide both --input-a and --input-b')
    for instance in range(args.instances):
        rng=np.random.default_rng(20261028+instance)
        if args.input_a:
            if args.instances!=1:raise ValueError('external tensors require --instances 1; independent synthetic instances use distinct inputs')
            if args.precision not in ('int4','int8','fp16','fp32'):raise ValueError('external tensors currently support native INT4/INT8 integers or FP16/FP32 values')
            a=np.load(args.input_a);b=np.load(args.input_b)
            if a.shape!=(logical[0],logical[2]) or b.shape!=(logical[2],logical[1]):
                raise ValueError('external tensors differ from the selected LLM operator geometry')
            a=np.pad(a,((0,m-a.shape[0]),(0,k-a.shape[1])));b=np.pad(b,((0,k-b.shape[0]),(0,n-b.shape[1])))
        else:
            a=rng.integers(-2,3,size=(m,k)).astype('<f4');b=rng.integers(-2,3,size=(k,n)).astype('<f4')
            a[logical[0]:]=0;b[:,logical[1]:]=0
        if args.precision=='fp16':a=a.astype('<f2').astype('<f4');b=b.astype('<f2').astype('<f4')
        inputs.append((a,b));np.save(root/f'input_a_{instance}.npy',a);np.save(root/f'input_b_{instance}.npy',b)
    config=dict(schema='llm.native-operator',operator=args.operator,shape=args.shape,precision=args.precision,
                logical_shape=logical,padded_shape=[m,n,k],plan=plan,
                numerical_contract='INT32 or FP32 packed accumulator; separate from legacy decoder FP16 semantics',
                payloads=write_payloads(plan,inputs,args.precision,root/'payloads'))
    path=root/'native_config.json';path.write_text(json.dumps(config,indent=2)+'\n');load_config(str(path))
    return root,path,config


def verify(root,config,mode):
    p=config['plan'];g=p['geometry'];r=p['regions'];m,n,k=config['padded_shape']
    log=next((root/'simulation/logs').glob('*.log'));text=log.read_text(errors='replace')
    records=re.findall(r'\[NATIVE_WORKLOAD\] core=(\d+) mode=(\d+) physical_worker=(\d+) logical_worker=(\d+) workers=(\d+) tasks=(\d+) replicate_b=(\d+)',text)
    if len(records)!=16 or {int(e[0]) for e in records}!=set(range(4,20)):
        raise ValueError('all sixteen physical workers must execute the native TE path')
    for record in records:
        core,precision,physical,logical,workers,tasks,replicated=map(int,record)
        expected=engine_parameters(root/'native_config.json',core)
        macro=len(expected['native_task_nodes'])
        count=(macro-1-logical)//workers+1 if logical<macro else 0
        if (precision,physical,logical,workers,tasks,replicated)!=(
                expected['native_precision_mode'],core-4,expected['native_worker_slot'],
                expected['native_worker_count'],count,expected['native_replicate_b']):
            raise ValueError('observed precision/physical/logical partition differs from native plan')
    modes=re.findall(r'\[ARRAY_NUMERICAL_DATA\] core=(\d+) functional=(\d+)',text)
    if {int(c) for c,f in modes if int(f)==int(mode=='full-functional')}!=set(range(4,20)):
        raise ValueError('array numerical mode differs from requested mode')
    errors=[];tiles=0
    reuse=g['reuse_m']
    mg=(g['m_tiles']+reuse-1)//reuse;ng=(g['n_tiles']+reuse-1)//reuse
    for instance,job in enumerate(p['instances']):
        a=np.load(root/f'input_a_{instance}.npy');b=np.load(root/f'input_b_{instance}.npy')
        expected=((a.astype('int64')@b.astype('int64')).astype('<i4') if config['precision'].startswith('int') else a@b) if mode=='full-functional' else None
        maximum=0.
        for task,node in enumerate(job['task_nodes']):
            slot=job['task_nodes'][:task].count(node);mi=task%mg;ni=(task//mg+mi)%ng
            with (root/'simulation/hbm'/f'hbm_out_node{node}.bin').open('rb') as stream:
                for im in range(min(reuse,g['m_tiles']-mi*reuse)):
                    for jn in range(min(reuse,g['n_tiles']-ni*reuse)):
                        offset=r['c']['offset']+(slot*reuse*reuse+im*reuse+jn)*r['c']['stride']
                        stream.seek(offset);data=stream.read(64*64*4)
                        if len(data)!=64*64*4:raise ValueError('short output tile')
                        got=np.frombuffer(data,dtype='<i4' if config['precision'].startswith('int') else '<f4').reshape(64,64).T
                        if expected is not None:
                            want=expected[(mi*reuse+im)*64:(mi*reuse+im+1)*64,(ni*reuse+jn)*64:(ni*reuse+jn+1)*64]
                            error=float(np.max(np.abs(got-want)));maximum=max(maximum,error)
                            if not np.allclose(got,want,rtol=2e-5,atol=2e-5):raise ValueError(f'instance {instance} output differs from quantized-input FP32/INT32 reference: {error}')
                        tiles+=1
        errors.append(dict(instance=instance,status='PASS' if expected is not None else 'SKIPPED',max_abs_error=maximum if expected is not None else None))
    summaries=list((root/'simulation').glob('**/execution_summary.csv'))
    if len(summaries)!=1:raise ValueError('missing unique execution summary')
    metrics={row['metric']:row['value'] for row in csv.DictReader(summaries[0].open())}
    if mode=='full-timing':
        sys.path.insert(0,str(ROOT/'test/attention_cluster_8qk_8pv'))
        from timing_mode import verify_timing_backend
        verify_timing_backend(log)
    # Each WCP completion is emitted after its final timed C write ACK.
    done=re.findall(r'\[Core (\d+)\] \[wcp\] LATENCY\(cycles\):',text)
    if set(map(int,done))!=set(range(4,20)):raise ValueError('missing timed native worker completion')
    return dict(status='PASS',sim_mode=mode,operator=config['operator'],precision=config['precision'],
        logical_shape=config['logical_shape'],padded_shape=config['padded_shape'],instances=len(p['instances']),
        input_sha256=[dict(a=digest(root/f'input_a_{i}.npy'),b=digest(root/f'input_b_{i}.npy'))
                      for i in range(len(p['instances']))],
        panel_mapping=p['panel_mapping'],replicate_b=p['replicate_b'],
        native_workers=16,output_tiles=tiles,numerics=errors,latency_unit='cycles',
        cycles=float(metrics['gemm_system_latency_cycles']),
        scope='first guest descriptor start through last native WCP completion after final C write ACK, inclusive +1',
        library_sha256=digest(ROOT/'install/lib/sst-elements-library/libgolem.so'),
        memhierarchy_sha256=digest(ROOT/'install/lib/sst-elements-library/libmemHierarchy.so'),
        vanadis_sha256=digest(ROOT/'install/lib/sst-elements-library/libvanadis.so'),
        config_sha256=digest(root/'native_config.json'),real_noc_hbm=True,
        output_contract='32-bit outputs; padded work is timed; no decoder next-token claim')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root',type=Path,required=True)
    parser.add_argument('--operator',choices=('qkv','wo','gate','up','down','lm_head'),default='qkv')
    parser.add_argument('--shape',choices=('small','formal'),default='small')
    parser.add_argument('--sequence',type=int,default=1024)
    parser.add_argument('--vocab',type=int,default=8201)
    parser.add_argument('--precision',choices=MODES,default='fp16')
    parser.add_argument('--instances',type=int,choices=(1,2,4),default=1)
    parser.add_argument('--reuse',type=int,choices=(1,2),default=2)
    parser.add_argument('--panel-mapping',choices=('group-affine','reuse-striped'),default='group-affine')
    parser.add_argument('--replicate-b',action='store_true')
    parser.add_argument('--sim-mode',choices=('full-functional','full-timing'),default='full-functional')
    parser.add_argument('--mpi-ranks',type=int,choices=(1,2,4),default=1)
    parser.add_argument('--cpu-set',default='4-7')
    parser.add_argument('--input-a',type=Path);parser.add_argument('--input-b',type=Path)
    parser.add_argument('--abi-selftest',action='store_true')
    parser.add_argument('--launcher',choices=('fixed','legacy'),default='fixed')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.sequence<=0 or args.vocab<=0:parser.error('sequence/vocab must be positive')
    root,path,config=prepare(args)
    native_bundle=None
    if args.launcher=='fixed':
        from tools.native_descriptor import write_bundle
        native_bundle=write_bundle(path,root)
    if args.prepare_only:print(path);sys.exit(0)
    m,n,k=config['padded_shape']
    command=['bash',str(ROOT/'test/golem/run_noc_dma_pipeline.sh'),'--groups','4','--num-cores','20','--gemm-cores','20',
        '--num-mem-nodes','5','--mem-node-size',str(config['plan']['node_bytes']),'--mesh-dim-x','4','--gemm-m',str(m),'--gemm-n',str(n),'--gemm-k',str(k),
        '--gemm-block-m','64','--gemm-block-n','64','--gemm-block-k','64','--dtype','fp32',
        '--tensor-source','synthetic','--output-mode','hbm','--memory-backend','ramulator2',
        '--mpi-ranks',str(args.mpi_ranks),'--sim-mode',args.sim_mode]
    env={k:v for k,v in os.environ.items() if not k.startswith('GOLEM_')}
    (root/'guest').mkdir()
    env.update(GOLEM_TEST_BINARY=str(root/'guest/test_noc_dma'),GOLEM_ATTENTION_KV_BUFFER_COUNT='1',
        GOLEM_ATTENTION_KV_SECOND_LOOKAHEAD='0',GOLEM_ARCH_SCRIPT='architecture/ncores_selfcom_dma_ctrl.py',GOLEM_CTRL_LINK_ENABLE='1',
        GOLEM_GROUP_MANAGER_ENABLE='1',GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE='1',GOLEM_REQUEST_SCHEDULER_ENABLE='1',
        GOLEM_NATIVE_WORKLOAD_CONFIG=str(path),GOLEM_ARTIFACT_ROOT=str(root/'simulation'),
        GOLEM_A_REUSE_N_TILES=str(args.reuse),GOLEM_B_REUSE_M_TILES=str(args.reuse),
        GOLEM_ARRAY_INPUT_SIZE='64',GOLEM_ARRAY_OUTPUT_SIZE='64',GOLEM_NUM_ARRAYS='64',
        GOLEM_GLOBAL_STRIDE_BYTES='2097152',GOLEM_LOCAL_GM_MODEL='banked',
        GOLEM_LOCAL_GM_BANKS='8',GOLEM_LOCAL_GM_LINE_BYTES='128',GOLEM_LOCAL_GM_RESPONSE_CYCLES='1',
        GOLEM_LOCAL_GM_RESPONSE_CAPACITY='32',GOLEM_NOC_LINK_BW='512GB/s',GOLEM_NOC_XBAR_BW='1024GB/s',
        GOLEM_NOC_INPUT_BUF_SIZE='128KB',GOLEM_NOC_OUTPUT_BUF_SIZE='128KB',GOLEM_GM_BUFFER_LENGTH='128KB',
        GOLEM_REQUEST_SCHEDULER_EVENT_DRIVEN_WORKER='0',GOLEM_VANADIS_ROCC_WAIT_FASTPATH='0',
        GOLEM_MPI_ARGS='--bind-to core --cpu-set '+args.cpu_set,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    guest=None
    if native_bundle:
        # This target depends only on runtime sources/ABI, never the workload.
        subprocess.run(['make','-C',str(ROOT/'software/native')],check=True)
        guest=ROOT/'build/native-guest/launcher'
        env.update(GOLEM_SKIP_DEFAULT_GUEST_BUILD='1',VANADIS_EXE=str(guest),GOLEM_TEST_BINARY=str(guest),
                   GOLEM_NATIVE_NODE0_INIT=native_bundle['node0_image'],GOLEM_ARRAY_TYPE='golem.TileUnitArray')
    if args.abi_selftest:
        if not guest:parser.error('--abi-selftest requires the fixed launcher')
        env['GOLEM_NATIVE_ABI_SELFTEST']='1'
    guest_before=digest(guest) if guest else None
    before=digest(ROOT/'install/lib/sst-elements-library/libgolem.so')
    with (root/'run.log').open('w') as stream:subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    if before!=digest(ROOT/'install/lib/sst-elements-library/libgolem.so'):raise ValueError('library changed during native run')
    result=verify(root,config,args.sim_mode);result['command']=command
    result['launcher']=args.launcher
    if guest:
        if digest(guest)!=guest_before:raise ValueError('fixed guest changed during run')
        log='\n'.join(p.read_text(errors='replace') for p in (root/'simulation/stdout').rglob('stdout-*'))
        done=re.findall(r'\[NATIVE_GUEST_DONE\] core=(\d+) token=(\d+) status=(\d+)',log)
        if len(done)!=16 or {int(c) for c,t,s in done if int(t)>0 and s=='0'}!=set(range(4,20)):
            raise ValueError('fixed guest token/completion failure')
        if args.abi_selftest:
            probes=re.findall(r'\[NATIVE_ABI_SELFTEST\] core=(\d+) status=PASS',log)
            if len(probes)!=16 or set(map(int,probes))!=set(range(4,20)):
                raise ValueError('not all workers passed native ABI self-tests')
            result['abi_selftest']='PASS'
        result.update(guest_sha256=guest_before,bundle_sha256=native_bundle['bundle_sha256'],
                      descriptor_status='PASS',native_tu=True)

    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
