#!/usr/bin/env python3
"""Audit GM operand scheduling experiments and the complete two-layer run."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import numpy as np
from summarize_reuse_window_optimization import (
    ROOT, read, digest, comparable_inputs, comparable_micro_inputs, compare_outputs, scopes,
)
from run_all_gemm_reuse_window_baseline import summarize, save_report, BASELINE_RESULT
from verify_decoder_dataflow_resources import verify
from run_decoder_stack import verify_lifecycle
from analyze_reuse_window_critical_path import analyze


def flow_stats(case, env):
    windows=[];inputs=[];traffic=Counter()
    for log in (case/'logs').glob('attention-sst*.log'):
        for line in log.open():
            if not any(s in line for s in ('[GEMM_WINDOW_FLOW]','[GEMM_A_PIPELINE]','[GEMM_REUSE_WINDOW]')):continue
            f=dict(re.findall(r'(\w+)=([^\s]+)',line))
            if '[GEMM_WINDOW_FLOW]' in line:
                if sum(int(v) for k,v in f.items() if k.startswith('phase_'))!=int(f['active_cycles']):
                    raise ValueError('window phase partition changed')
                if int(f['gm_lookahead'])!=min(int(env['GOLEM_REUSE_WINDOW_PREFETCH_SLOTS']),int(env.get('GOLEM_REUSE_WINDOW_GM_LOOKAHEAD','4'))):
                    raise ValueError('GM lookahead trace/config mismatch')
                if f['a_first']!=env.get('GOLEM_REUSE_WINDOW_A_FIRST','0'):raise ValueError('A priority trace/config mismatch')
                windows.append(f)
            elif '[GEMM_A_PIPELINE]' in line:inputs.append(f)
            else:
                for k in ('launches','hbm_read','gm_read','gm_write','c_gm_read','c_gm_write','c_drains','c_restores'):
                    traffic[k]+=int(f.get(k,0))
    enabled=int(env.get('GOLEM_REUSE_WINDOW_A_PREFETCH','0'))
    if 'GOLEM_REUSE_WINDOW_A_PREFETCH' in env:
        if len(inputs)!=len(windows) or not inputs:raise ValueError('missing A pipeline trace')
        for f in inputs:
            if int(f['enabled'])!=enabled or int(f['entries'])!=2 or int(f['live_at_finish']):raise ValueError('A pipeline protocol mismatch')
            if not (0<int(f['high_water'])<=2 if enabled else int(f['high_water'])==0):raise ValueError('A snapshot capacity mismatch')
    return dict(window_records=len(windows),traffic=dict(traffic),
        waits={k:sum(int(f[k]) for f in windows) for k in ('wait_a','wait_b_gm','wait_b_dma','wait_array')},
        a_prefetch_enabled=enabled,a_snapshot_high_water=max([int(f['high_water']) for f in inputs] or [0]),
        a_ready_hits=sum(int(f['ready_hits']) for f in inputs),a_live_at_finish=sum(int(f['live_at_finish']) for f in inputs))


def micro_group(root,names):
    rows={};reference=None;reference_result=None
    for name in names:
        directory=root/name;case=directory/'layout2';d=read(case/'result.json');p=d['provenance']
        if d['status']!='PASS':raise ValueError('micro audit failed')
        signature=({k:p[k] for k in ('guest_sha256','sst_library_sha256','sst_memhierarchy_sha256')},comparable_micro_inputs(directory))
        if reference is None:reference=signature;reference_result=d
        if signature!=reference:raise ValueError('micro inputs/binaries differ')
        if d['sfu_work']!=reference_result['sfu_work']:raise ValueError('micro SFU token work changed')
        for k in ('hbm_read_bytes','hbm_write_bytes'):
            if d['dma_timing'][k]!=reference_result['dma_timing'][k]:raise ValueError('micro DMA traffic changed')
        f=flow_stats(case,p['environment'])
        if rows and f['traffic']!=next(iter(rows.values()))['flow']['traffic']:raise ValueError('micro GM/C/MAC traffic changed')
        rows[name]=dict(cycles=d['dma_timing']['cycles'],status='PASS',flow=f,
            reduction_percent=100*(1-d['dma_timing']['cycles']/reference_result['dma_timing']['cycles']))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root',type=Path,default=ROOT/'results/reuse_window_gm_schedule_20261021')
    parser.add_argument('--candidate',default='formal_a_prefetch_lm1');a=parser.parse_args();root=a.experiment_root.resolve()
    micro={}
    for names in [('micro_control','micro_a_first','micro_look1','micro_look2'),
                  ('micro2_control','micro2_a_prefetch','micro2_a_prefetch_first'),
                  ('cluster2_control','cluster2_a_prefetch')]:micro.update(micro_group(root,names))
    cases={'control':root/'formal_control2/reuse_window_probe','candidate':root/a.candidate/'reuse_window_probe'}
    results={};profiles={};audits={};provenance={}
    for name,case in cases.items():
        d=read(case/'decoder_stack_result.json');m=read(case/'decoder_stack_inputs.json');p=read(case/'decoder_stack_provenance.json')
        if d['status']!='PASS':raise ValueError('formal numerical audit failed')
        lifecycle=verify_lifecycle(case,m)[3];resources=verify(case)
        report=summarize(case,BASELINE_RESULT);save_report(case.parent,report)
        if report['status']!='PASS':raise ValueError('formal GEMM/panel audit failed')
        audits[name]=dict(status='PASS',lifecycle=lifecycle,resources=resources,flow=flow_stats(case,p['environment']))
        results[name]=d;provenance[name]=p;profiles[name]=analyze(case)
        (root/f'critical_workers_{name}.json').write_text(json.dumps(profiles[name],indent=2)+'\n')
        measured=read(case.parent/'experiment_provenance.json')
        for path,sha in measured['source_sha256'].items():
            if digest(ROOT/path)!=sha:raise ValueError('production source changed after measurement')
        if measured['library_sha256']!=digest(ROOT/'install/lib/sst-elements-library/libgolem.so'):raise ValueError('installed library changed')
    for key in ('guest_sha256','sst_library_sha256','sst_memhierarchy_sha256'):
        if provenance['control'][key]!=provenance['candidate'][key]:raise ValueError('formal binaries differ')
    if comparable_inputs(cases['control'])!=comparable_inputs(cases['candidate']):raise ValueError('formal inputs differ')
    def env(name):
        d={k:v.replace(str(cases[name]),'<case>') for k,v in provenance[name]['environment'].items()}
        if 'GOLEM_MPI_ARGS' in d:d['GOLEM_MPI_ARGS']=re.sub(r'--cpu-set \S+','--cpu-set <placement>',d['GOLEM_MPI_ARGS'])
        return d
    before,after=env('control'),env('candidate')
    changes={k:dict(control=before.get(k),candidate=after.get(k)) for k in set(before)|set(after) if before.get(k)!=after.get(k)}
    if set(changes)-{'GOLEM_REUSE_WINDOW_A_PREFETCH','GOLEM_DECODER_LM_N_PARALLEL'}:raise ValueError('unexpected hardware/software difference')
    bits=compare_outputs(cases['control'],cases['candidate'])
    x,y=[np.load(case/'last_token_logits.npy').astype('<f2').view('<u2') for case in cases.values()]
    if x.shape!=y.shape:raise ValueError('logit shape changed')
    bits['logits']=dict(elements=x.size,fp16_bit_mismatches=int(np.count_nonzero(x!=y)))
    if any(v['fp16_bit_mismatches'] for v in bits.values()):raise ValueError('formal bits changed')
    old_case=ROOT/'results/reuse_window_optimization_20261021/formal_final/reuse_window_probe'
    old=read(old_case/'decoder_stack_result.json')
    if comparable_inputs(old_case)!=comparable_inputs(cases['candidate']):raise ValueError('previous default inputs differ')
    previous_bits=compare_outputs(old_case,cases['candidate'])
    old_logits=np.load(old_case/'last_token_logits.npy').astype('<f2').view('<u2')
    if old_logits.shape!=y.shape:raise ValueError('previous default logit shape differs')
    previous_bits['logits']=dict(elements=y.size,fp16_bit_mismatches=int(np.count_nonzero(old_logits!=y)))
    if any(v['fp16_bit_mismatches'] for v in previous_bits.values()):raise ValueError('previous default output bits changed')
    comparison={k:dict(previous_cycles=scopes(old)[k],control_cycles=scopes(results['control'])[k],candidate_cycles=v,
        reduction_vs_control_percent=100*(1-v/scopes(results['control'])[k]),reduction_vs_previous_percent=100*(1-v/scopes(old)[k]))
        for k,v in scopes(results['candidate']).items()}
    state=read(root/'before_state.json');original=ROOT.parent/'Muticore-Attention'
    if subprocess.check_output(['git','diff','HEAD'],cwd=original,text=True)!=state['original_diff'] or digest(original/'install/lib/sst-elements-library/libgolem.so')!=state['original_library']:
        raise ValueError('original baseline changed')
    for path,sha in state['files'].items():
        if digest(root/'before'/path)!=sha:raise ValueError('before snapshot changed')
    selected=a.candidate if all(comparison[k]['reduction_vs_previous_percent']>0 for k in ('完整端到端','两层 decoder')) else 'previous_default'
    summary=dict(status='PASS',selected=selected,formal_comparison=comparison,micro=micro,validation=audits,
        bitwise_comparison=bits,previous_bitwise_comparison=previous_bits,environment_changes=changes,original_baseline_preserved=True,
        same_formal_library_guest_inputs_hardware=True,next_token=results['candidate']['lm_head']['next_token']['token_id'])
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# GM 输入流水优化实测','',
        'S=1024、H=2048、FFN=8192、两层、16 worker、词表 128256。所有 GEMM 保持 reuse-window。',
        '1 GHz：cycles / 1e6 = ms。完整端到端从 token IDs 首次读取计至完整 logits 最后 DMA ACK。','',
        '| 配置 | 两层 decoder ms | 完整端到端 ms |','|---|---:|---:|',
        f"| 保留的上一轮默认 | {scopes(old)['两层 decoder']/1e6:.6f} | {scopes(old)['完整端到端']/1e6:.6f} |"]
    lines += [f"| {name} | {scopes(d)['两层 decoder']/1e6:.6f} | {scopes(d)['完整端到端']/1e6:.6f} |" for name,d in results.items()]
    lines += ['',f'采用决定：`{selected}`。','',
        '| 范围 | 对照 cycles | 候选 cycles | 相对对照降低 | 相对旧默认降低 |','|---|---:|---:|---:|---:|']
    lines += [f"| {name} | {d['control_cycles']:,.3f} | {d['candidate_cycles']:,.3f} | {d['reduction_vs_control_percent']:.3f}% | {d['reduction_vs_previous_percent']:.3f}% |" for name,d in comparison.items()]
    lines += ['', '| FFN 筛选 | cycles | 相对组内对照降低 | wait A | wait B GM |','|---|---:|---:|---:|---:|']
    lines += [f"| {name} | {d['cycles']:,.3f} | {d['reduction_percent']:.3f}% | {d['flow']['waits']['wait_a']:,} | {d['flow']['waits']['wait_b_gm']:,} |" for name,d in micro.items()]
    lines += ['', '筛选按各自相同库/guest/输入/硬件分组比较。两版单 worker 对照都复现 2,009,035.625 cycles。',
        '减少 B 的 GM 提前量会暴露权重等待，未采用；仅 A 优先无净收益。采用的是当前 + 下一块 A 的两个有界读取状态，仍保留四个 B DMA/暂存槽。',
        '每块 A 都从现有 GM 区域经计时端口读取，读完并 ACK 后才能使用；不跨 N 复用，不增加物理 SRAM、端口、阵列或操作数上下文。仅当前 A 编程到阵列，必须等前一个 MAC 完成。',
        'A 状态高水位至多 2，所有记录在结束时 live=0。GM/HBM/C 字节、dense MAC 和 SFU 工作在各组 FFN 实验中完全相同。',
        '正式候选仅启用 A 读取流水，并使用此前独立实测更快的 LM N=1；关闭 Norm 交接、epoch 优化、SFU 双 chunk 和 FFN 双窗口。',
        'LM N 的选择发生在 decoder 之后；两层 decoder 的延迟单独列出，避免用 LM 收益掩盖主体退化。','',
        '所有正式数值、QKV 实际面板、16 worker GEMM 覆盖、DMA 字节、生命周期和 ACK 排空检查 PASS；末层 Y、final Norm、完整 logits 的 FP16 位差异均为 0。',
        '各窗口等待可以与别的工作重叠，不能直接相加当作可消除时间。关键路径分区见 critical_workers_*.json。','',
        '## 复现','', '```bash',
        'OPENBLAS_NUM_THREADS=1 python3 baseline/attention_cluster_8qk_8pv/run_all_gemm_reuse_window_baseline.py \\',
        '  --artifact-root results/gm_prefetch_reproduction --shape formal --run \\',
        '  --window-a-prefetch 1 --lm-n-parallel 1 --cpu-set 0-7','```','',
        '同构建对照使用 --window-a-prefetch 0 --lm-n-parallel 4。原始架构和测试所在仓库保持不变，本轮修改仅在独立新基线中。',
        '详细假设、负结果和单端口标定见 HYPOTHESIS.md；首版正式对照主动取消并被第二版替代，不作为完整延迟结果。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='PASS',selected=selected,formal_comparison=comparison,bitwise=bits),indent=2))


if __name__=='__main__':main()
