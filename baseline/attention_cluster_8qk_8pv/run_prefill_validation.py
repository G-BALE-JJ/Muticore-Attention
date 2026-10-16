#!/usr/bin/env python3
"""Screen generalization locally, then validate formal dimensions and 16 layers."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import time
from run_projection_overlap import ROOT, digest
from run_decoder_layer import read_bands
import numpy as np


CASES={
    'local':[
        ('dense_l1_s1','small',1,1,'dense',65,True,8),
        ('dense_l1_s33','small',1,33,'dense',65,True,8),
        ('dense_l3_s65','small',3,65,'dense',8201,True,8),
        ('dense_l3_s257','small',3,257,'dense',65,True,8),
        ('sparse_l3_s1025','small',3,1025,'sparse',8201,False,8),
        ('dense_l1_s1_mpi2','small',1,1,'dense',65,True,2)],
    'formal':[('formal_l2_s1024','formal',2,1024,'sparse',128256,False,8)],
    'target':[
        ('dense_l16_s65_retained','small',16,65,'dense',8201,True,8),
        ('dense_l16_s65_pingpong','small',16,65,'dense',8201,False,8)]}


def simulation_source_ast(source):
    """Exclude only host numerical/lifecycle checking and the replay branch.

    Input generation, command construction, environment and simulation launch
    must remain identical before an old simulation can be reused.
    """
    tree=ast.parse(source)
    tree.body=[node for node in tree.body if not (
        isinstance(node,ast.FunctionDef) and node.name in
        ('compare','compare_normalized_chain','verify_numerics','verify_lifecycle'))]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name=='main':
            node.body=[statement for statement in node.body if not (
                isinstance(statement,ast.If) and isinstance(statement.test,ast.Attribute)
                and isinstance(statement.test.value,ast.Name)
                and statement.test.value.id=='a' and statement.test.attr=='verify_only')]
    return ast.dump(tree,include_attributes=False)


def source_matches(case,path,sha):
    current=ROOT/path
    if digest(current)==sha:return True
    if path!='baseline/attention_cluster_8qk_8pv/run_decoder_stack.py':return False
    snapshot=case/'source_snapshot'/path
    replay=case/'verification_provenance.json'
    if not snapshot.exists() or not replay.exists() or digest(snapshot)!=sha:return False
    if json.loads(replay.read_text()).get('verifier_source_sha256')!=digest(current):return False
    return simulation_source_ast(snapshot.read_text())==simulation_source_ast(current.read_text())


def compare_outputs(a,b):
    manifests=[json.loads((case/'decoder_stack_inputs.json').read_text()) for case in (a,b)]
    for name in manifests[0]['oracle_sha256']:
        if manifests[0]['oracle_sha256'][name]!=manifests[1]['oracle_sha256'][name]:
            raise ValueError('comparison oracle mismatch: '+name)
    for name in ('input_token_ids.bin','projection_x.bin'):
        if digest(a/name)!=digest(b/name):raise ValueError('comparison input mismatch')
    for key in ('final_output','final_norm'):
        values=[]
        for case,m in zip((a,b),manifests):
            offset=m['layout']['layers'][-1]['output'] if key=='final_output' else m['layout']['final_norm']['output']
            values.append(read_bands(case,offset,m['hidden'],rows=m['sequence']//4))
        if not np.array_equal(values[0].view('<u2'),values[1].view('<u2')):raise ValueError('comparison FP16 mismatch: '+key)
    if not np.array_equal(np.load(a/'last_token_logits.npy').view('<u2'),np.load(b/'last_token_logits.npy').view('<u2')):
        raise ValueError('comparison logits mismatch')
    cycles=[json.loads((case/'decoder_stack_result.json').read_text())['embedding']['end_to_end_cycles'] for case in (a,b)]
    mpi_pair=manifests[0]['retain_outputs']==manifests[1]['retain_outputs']
    if mpi_pair and cycles[0]!=cycles[1]:raise ValueError('MPI comparison cycle mismatch')
    return {'status':'PASS','all_executed_rows_and_logits_bitwise_equal':True,
            'cycles':cycles,'mpi_cycle_equality_checked':mpi_pair,'first':str(a),'second':str(b)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root',type=Path,required=True)
    p.add_argument('--stage',choices=(*CASES,'all'),default='all')
    p.add_argument('--cpu-set',default='0-7')
    a=p.parse_args();root=a.artifact_root.resolve();root.mkdir(parents=True,exist_ok=True)
    build_file=root/'experiment_build.json'
    if not build_file.exists():
        # A fresh validation batch pins the functional/timing models, host
        # verifiers, guest and libraries before starting the first case.
        paths=set(Path(__file__).parent.glob('*.py'))|set(Path(__file__).parent.glob('*.cpp'))
        for subdir in ('attention','globalmemory','workercmdproc','array','rocc','sfu'):
            for suffix in ('*.h','*.cc'):paths.update((ROOT/'src/sst/elements/golem'/subdir).glob(suffix))
        paths.update(ROOT/path for path in (
            'src/sst/elements/golem/fp16.h',
            'src/sst/elements/golem/tests/architecture/ncores_selfcom_dma_ctrl.py',
            'src/sst/elements/golem/tests/architecture/cpu_builder.py',
            'src/sst/elements/golem/tests/architecture/ramulator/hbm2e_2500.yaml',
            'src/sst/elements/golem/tests/small/muticore_attention/golem_attention_runtime.cpp',
            'src/sst/elements/golem/tests/small/muticore_attention/golem_decoder_runtime.h',
            'src/sst/elements/golem/tests/small/muticore_attention/golem_projection_runtime.h',
            'src/sst/elements/golem/tests/small/muticore_attention/run_fused_attention_scale.sh',
            'src/sst/elements/golem/tests/small/muticore_attention/riscv64/decoder-embedding/fused_attention',
            'install/lib/sst-elements-library/libgolem.so',
            'install/lib/sst-elements-library/libmemHierarchy.so'))
        build_file.write_text(json.dumps({'source_sha256':{str(path.relative_to(ROOT)):digest(path) for path in sorted(paths)},
            'GM_bytes_per_worker':2**21,'hardware_resources_changed':False},indent=2)+'\n')
    frozen=json.loads((root/'experiment_build.json').read_text())['source_sha256']
    def check_build():
        if any(digest(ROOT/name)!=sha for name,sha in frozen.items()):raise ValueError('frozen sources or libraries changed')
    report={'status':'RUNNING','stage':a.stage,'cases':{}}
    destination=root/f'{a.stage}_validation.json'
    def save():destination.write_text(json.dumps(report,indent=2)+'\n')
    save()
    for stage in CASES if a.stage=='all' else [a.stage]:
        for name,shape,layers,sequence,weights,vocab,retain,ranks in CASES[stage]:
            check_build();case=root/name
            result_file=case/'decoder_stack_result.json'
            if case.exists():
                if not result_file.exists():raise ValueError('incomplete existing case: '+str(case))
                result=json.loads(result_file.read_text())
                if result.get('status')!='PASS' or result.get('gm_banks',{}).get('status')!='PASS':
                    raise ValueError('existing case lacks acceptance audits')
                provenance=json.loads((case/'decoder_stack_provenance.json').read_text())
                if any(not source_matches(case,path,sha) for path,sha in provenance['source_sha256'].items()):
                    raise ValueError('existing case source provenance differs')
                for key,library in (('sst_library_sha256','libgolem.so'),('sst_memhierarchy_sha256','libmemHierarchy.so')):
                    if provenance[key]!=digest(ROOT/'install/lib/sst-elements-library'/library):
                        raise ValueError('existing case library provenance differs')
            else:
                command=[sys.executable,str(Path(__file__).with_name('run_decoder_stack.py')),
                    '--artifact-root',str(case),'--shape',shape,'--layers',str(layers),'--sequence',str(sequence),
                    '--weights',weights,'--embedding','--vocab',str(vocab),'--mpi-ranks',str(ranks),
                    '--cpu-set',a.cpu_set if ranks==8 else a.cpu_set.split('-')[0]+'-'+str(int(a.cpu_set.split('-')[0])+ranks-1),
                    '--gemm-prefetch','--prefetch-slots','4','--weight-trace','--ffn-down-fusion',
                    '--down-residual-fusion','--ffn-panel-tiles','16','--ffn-array-groups','2',
                    '--ffn-layout','2','--lm-n-parallel','4']
                if retain:command+=['--retain-layer-outputs']
                print('Running '+name,flush=True);start=time.monotonic()
                with (root/(name+'.log')).open('w') as log:
                    subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                print(f'PASS {name}, wall={time.monotonic()-start:.1f}s',flush=True)
                check_build();result=json.loads(result_file.read_text())
            report['cases'][name]={'status':result['status'],'shape':result['shape'],
                'cycles':result['embedding']['end_to_end_cycles'],'token_id':result['lm_head']['next_token']['token_id'],
                'sst_wall_seconds':json.loads((case/'simulation_wall_time.json').read_text())['sst_wall_seconds'],
                'gm_untimed_tensor_bytes':result['gm_banks']['untimed_tensor_bytes']}
            save()
    pairs=[('dense_l1_s1','dense_l1_s1_mpi2'),('dense_l16_s65_retained','dense_l16_s65_pingpong')]
    report['comparisons']=[]
    for first,second in pairs:
        if (root/first/'decoder_stack_result.json').exists() and (root/second/'decoder_stack_result.json').exists():
            report['comparisons'].append(compare_outputs(root/first,root/second))
    report['status']='PASS';save();print(json.dumps(report,indent=2))


if __name__=='__main__':main()
