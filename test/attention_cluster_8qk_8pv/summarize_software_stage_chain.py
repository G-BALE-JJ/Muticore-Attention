#!/usr/bin/env python3
"""Audit software-only GM residency against identical-hardware full-stack runs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import numpy as np
from summarize_reuse_window_optimization import ROOT,read,digest,comparable_inputs,compare_outputs,scopes
from summarize_gm_prefetch import flow_stats
from run_all_gemm_reuse_window_baseline import summarize,save_report,BASELINE_RESULT
from verify_decoder_dataflow_resources import verify
from analyze_reuse_window_critical_path import analyze


def normalized_inputs(case):
    m=comparable_inputs(case);m.pop('gm_stage_chain',None);return m


def output_bits(a,b):
    bits=compare_outputs(a,b)
    x,y=[np.load(p/'last_token_logits.npy').astype('<f2').view('<u2') for p in (a,b)]
    if x.shape!=y.shape:raise ValueError('logit shape changed')
    bits['logits']=dict(elements=x.size,fp16_bit_mismatches=int(np.count_nonzero(x!=y)))
    if any(v['fp16_bit_mismatches'] for v in bits.values()):raise ValueError('output bit mismatch')
    return bits


def intermediate_traffic(case):
    m=read(case/'decoder_stack_inputs.json');s=m['layout']['shared'];stride=m['layout']['node_stride']
    size=m['sequence']//4*m['hidden']*2
    ranges=[(name,node*stride+s[name],node*stride+s[name]+size) for name in ('wo','h1','norm2') for node in range(1,5)]
    counts=Counter()
    log=next((case/'logs').glob('attention-sst*.log')).with_suffix('.resources.jsonl')
    with log.open() as stream:
        for line in stream:
            if 'resource=dma_read_transport ' not in line and 'resource=dma_write_transport ' not in line:continue
            f=dict(re.findall(r'(\w+)=([^\s]+)',line));a=int(f['address']);n=int(f['bytes'])
            for name,start,end in ranges:
                if a<end and start<a+n:
                    if a<start or a+n>end:raise ValueError('DMA straddles retained intermediate')
                    counts[name+'_'+('read' if f['resource']=='dma_read_transport' else 'write')]+=n
    return {name+'_'+direction:counts[name+'_'+direction] for name in ('wo','h1','norm2') for direction in ('read','write')}


def job_totals(case):
    counts=Counter()
    with next((case/'logs').glob('attention-sst*.log')).open() as stream:
        for line in stream:
            if '[DECODER_OP] event=done' not in line:continue
            f=dict(re.findall(r'(\w+)=([^\s]+)',line))
            for k in ('hbm_read','hbm_write'):counts[k]+=int(f[k])
    return dict(counts)


def audit_pair(root,prefix):
    cases={name:root/(prefix+'_'+name)/'reuse_window_probe' for name in ('control','chain')}
    d={};p={};audits={};flows={};profiles={}
    for name,case in cases.items():
        d[name]=read(case/'decoder_stack_result.json');p[name]=read(case/'decoder_stack_provenance.json')
        if d[name]['status']!='PASS':raise ValueError('full-stack numerical/lifecycle audit failed')
        report=summarize(case,BASELINE_RESULT);save_report(case.parent,report)
        if report['status']!='PASS':raise ValueError('GEMM reuse-window coverage audit failed')
        audits[name]=verify(case);flows[name]=flow_stats(case,p[name]['environment']);profiles[name]=analyze(case)
        (root/(prefix+'_critical_'+name+'.json')).write_text(json.dumps(profiles[name],indent=2)+'\n')
    if normalized_inputs(cases['control'])!=normalized_inputs(cases['chain']):raise ValueError('inputs/layout changed')
    for key in ('guest_sha256','sst_library_sha256','sst_memhierarchy_sha256'):
        if p['control'][key]!=p['chain'][key]:raise ValueError('measured binaries differ')
    def env(name):
        e={k:v.replace(str(cases[name]),'<case>') for k,v in p[name]['environment'].items()}
        if 'GOLEM_MPI_ARGS' in e:e['GOLEM_MPI_ARGS']=re.sub(r'--cpu-set \S+','--cpu-set <placement>',e['GOLEM_MPI_ARGS'])
        return e
    before,after=env('control'),env('chain')
    changes={k:dict(control=before.get(k),chain=after.get(k)) for k in set(before)|set(after) if before.get(k)!=after.get(k)}
    if changes!={'GOLEM_DECODER_GM_STAGE_CHAIN':dict(control='0',chain='1')}:raise ValueError('unexpected software/hardware config difference: '+repr(changes))
    if flows['control']['traffic']!=flows['chain']['traffic']:raise ValueError('dense GEMM window traffic or MAC launches changed')
    matrix=read(cases['chain']/'decoder_stack_inputs.json');tensor=matrix['sequence']*matrix['hidden']*2*matrix['layers']
    control=intermediate_traffic(cases['control']);chain=intermediate_traffic(cases['chain'])
    expected={name+'_'+direction:tensor*(2 if name=='h1' and direction=='read' else 1) for name in ('wo','h1','norm2') for direction in ('read','write')}
    if control!=expected or any(chain.values()):raise ValueError('actual elided HBM region traffic differs from contract')
    totals={k:job_totals(case) for k,case in cases.items()}
    for key,factor in (('hbm_read',4),('hbm_write',3)):
        if totals['control'][key]-totals['chain'][key]!=factor*tensor:raise ValueError('decoder job HBM saving mismatch')
    a,b=scopes(d['control']),scopes(d['chain'])
    timing={k:dict(control_cycles=a[k],chain_cycles=v,reduction_percent=100*(1-v/a[k])) for k,v in b.items()}
    return dict(status='PASS',timing=timing,bits=output_bits(cases['control'],cases['chain']),environment_changes=changes,
                intermediate_hbm_traffic=dict(control=control,chain=chain),decoder_job_hbm_totals=totals,
                saved_hbm_read_bytes=4*tensor,saved_hbm_write_bytes=3*tensor,
                resources=audits,gemm=flows,binaries={k:p['chain'][k] for k in ('guest_sha256','sst_library_sha256','sst_memhierarchy_sha256')})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root',type=Path,default=ROOT/'results/software_stage_chain_20261021')
    a=parser.parse_args();r=a.experiment_root.resolve()
    small=audit_pair(r,'small');formal=audit_pair(r,'formal')
    old=ROOT/'results/reuse_window_gm_schedule_20261021/formal_a_prefetch_lm1/reuse_window_probe'
    candidate=r/'formal_chain/reuse_window_probe';control=r/'formal_control/reuse_window_probe'
    if normalized_inputs(old)!=normalized_inputs(candidate):raise ValueError('previous best inputs changed')
    old_bits=output_bits(old,candidate);old_result=read(old/'decoder_stack_result.json')
    for scope in ('两层 decoder','完整端到端'):
        if scopes(old_result)[scope]!=formal['timing'][scope]['control_cycles']:raise ValueError('control did not reproduce previous best timing')
    fixed=read(r/'fixed_hardware_sources.json')
    for name,sha in fixed.items():
        if digest(ROOT/name)!=sha:raise ValueError('hardware or architecture source changed: '+name)
    source_state=read(r/'before_state.json');original=ROOT / 'Muticore-Attention'
    if subprocess.check_output(['git','diff','HEAD'],cwd=original,text=True)!=source_state['original_diff'] or digest(original/'install/lib/sst-elements-library/libgolem.so')!=source_state['original_library']:
        raise ValueError('original baseline changed')
    frozen=read(r/'measured_binaries.json')
    if digest(ROOT/'src/sst/elements/golem/attention/decoderOpJob.h')!=frozen['decoder_job'] or digest(ROOT/'install/lib/sst-elements-library/libgolem.so')!=frozen['library']:raise ValueError('measured implementation changed')
    selected='formal_chain' if all(formal['timing'][scope]['reduction_percent']>0 for scope in ('两层 decoder','完整端到端')) else 'previous_default'
    result=dict(status='PASS',selected=selected,software_only=True,hardware_sources_unchanged=len(fixed),original_preserved=True,
                small=small,formal=formal,previous_best_bits=old_bits)
    (r/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 现有硬件上的软件 GM 阶段衔接','',
      '硬件固定：16 worker、每 worker 64 阵列、单操作数上下文、2 MiB GM；端口、队列、带宽、时钟和内存时序均不变。51 个硬件/架构源文件校验一致，控制与候选的运行环境只改变 GOLEM_DECODER_GM_STAGE_CHAIN。',
      '正式形状：S=1024、H=2048、FFN=8192、2 层、词表 128256；两者均启用 A 预读、LM N=1。','',
      '| 范围 | 保留的上一轮最佳 / 本轮同构建对照 ms | 软件 GM 衔接 ms | 延迟降低 |','|---|---:|---:|---:|']
    lines += [f"| {k} | {v['control_cycles']/1e6:.6f} | {v['chain_cycles']/1e6:.6f} | {v['reduction_percent']:.3f}% |" for k,v in formal['timing'].items()]
    lines += ['',f'采用决定：`{selected}`。','',
      '数据路径：Wo 阵列输出通过现有端口写入 GM 0xb0000 区域；Add 读取该区域和 HBM 中的原始 X，将 H1 原位写回；Norm2 读完整隐藏维度，通过现有 SFU 计算，再写入现有 FFN 输入区的 K-major 布局；FFN 从该区消费，第二处残差从保留的 H1 读取。所有 GM 读写经过计时端口、重试和 ACK。',
      'Wo/H1 共用一份现有缓冲；Norm2 占用原有输入区。没有增加张量缓存、GM 地址空间、阵列上下文或物理请求队列。仅保留软件元数据；错 epoch/地址/worker/形状的消费被拒绝。',
      '本轮保留 Wo、Add、Norm2、FFN 的原有 guest 描述符和提交边界；融合的是数据路径，没有宣称已消除 CPU 提交/切换开销。',
      '适用范围为每节点最多 256 行（当前 S<=1024）、H<=2048 的 all-window/layout2 私有权重路径；更大形状在生产前回退原路径。与旧 Norm 交接、FFN 双窗口和权重共享互斥。','',
      f"实际资源事件确认，正式两层减少 HBM 读取 {formal['saved_hbm_read_bytes']:,} 字节、写入 {formal['saved_hbm_write_bytes']:,} 字节，合计 {(formal['saved_hbm_read_bytes']+formal['saved_hbm_write_bytes'])/2**20:.0f} MiB。三个中间区域的 DMA 读写均为 0；GEMM 窗口 MAC launch、权重读取、C 部分和工作量相同。",
      '完整设备输出 Y、final Norm、完整 logits 对同构建对照及上一轮最佳逐位一致。被消除的 Wo/H1/Norm2 不再从 HBM 检查；其 GM 中间位模式由有限资源 C++ 测试逐项比较，完整 SST 输出仍作独立数学参考验证。',
      '小尺寸同样为 S1024、16 worker、两层：详细数字见 summary.json small。完成延迟以模拟周期计，宿主机 wall time 仅为执行成本。',
      '测试、负例、时序与资源审计见 checks/，硬件固定证据见 fixed_hardware_sources.json，原始基线保护证据见 before_state.json / final_validation.json。','',
      '## 复现','', '```bash',
      'OPENBLAS_NUM_THREADS=1 python3 baseline/attention_cluster_8qk_8pv/run_all_gemm_reuse_window_baseline.py \\',
      '  --artifact-root results/software_stage_chain_reproduction --shape formal --run \\',
      '  --window-a-prefetch 1 --lm-n-parallel 1 --gm-stage-chain --cpu-set 0-7','```','',
      '移除 --gm-stage-chain 得到同构建对照。原始 Muticore-Attention 仓库保持不变。']
    (r/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='PASS',selected=selected,timing=formal['timing'],bits=formal['bits'],saved_read=formal['saved_hbm_read_bytes'],saved_write=formal['saved_hbm_write_bytes']),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
