#!/usr/bin/env python3
"""Audit and compare the next-step experiments, including rejected candidates."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import numpy as np
from summarize_reuse_window_optimization import (
    ROOT, read, digest, comparable_inputs, comparable_micro_inputs, compare_outputs, scopes,
)
from analyze_reuse_window_critical_path import analyze
from run_all_gemm_reuse_window_baseline import summarize, save_report, BASELINE_RESULT
from run_decoder_stack import verify_lifecycle
from verify_decoder_dataflow_resources import verify


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-root',type=Path,default=ROOT/'results/reuse_window_next_steps_20261021')
    p.add_argument('--candidate',default='auto');a=p.parse_args();root=a.experiment_root.resolve()
    candidate_names=['formal_combined','formal_epoch_only']
    if a.candidate=='auto':
        a.candidate=min(candidate_names,key=lambda n:read(root/n/'reuse_window_probe/decoder_stack_result.json')['embedding']['end_to_end_cycles'])
    control=root/'formal_control_final/reuse_window_probe';candidate=root/a.candidate/'reuse_window_probe'
    cases={'control':control,'candidate':candidate}
    cases.update({n:root/n/'reuse_window_probe' for n in candidate_names if n!=a.candidate})
    results={};profiles={};validation={}
    for name,case in cases.items():
        result=read(case/'decoder_stack_result.json')
        if result['status']!='PASS':raise ValueError('formal numerical audit failed')
        manifest=read(case/'decoder_stack_inputs.json')
        lifecycle=verify_lifecycle(case,manifest)[3]
        resources=verify(case)
        report=summarize(case,BASELINE_RESULT)
        if report['status']!='PASS':raise ValueError('fresh all-GEMM/panel audit failed')
        save_report(case.parent,report)
        profiles[name]=analyze(case);results[name]=result
        validation[name]=dict(status='PASS',lifecycle=lifecycle,resources=resources,
                              all_gemm_coverage=report['coverage']['status'],projection_panels=report['projection_panels']['status'])
        (root/f'critical_workers_{name}.json').write_text(json.dumps(profiles[name],indent=2)+'\n')
    prov={name:read(case/'decoder_stack_provenance.json') for name,case in cases.items()}
    for name,case in cases.items():
        for key in ('guest_sha256','sst_library_sha256','sst_memhierarchy_sha256'):
            if prov['control'][key]!=prov[name][key]:raise ValueError(f'comparison changed {key}')
        if comparable_inputs(control)!=comparable_inputs(case):raise ValueError('inputs/layout changed')
    allowed={'GOLEM_DECODER_NORM_FFN_HANDOFF','GOLEM_DECODER_EPOCH_FLAGS','GOLEM_DECODER_LM_N_PARALLEL',
             'GOLEM_REUSE_WINDOW_LATE_LEASE','GOLEM_DECODER_FFN_SFU_STREAM','GOLEM_DECODER_FFN_SFU_CHUNK',
             'GOLEM_DECODER_FFN_BLOCK_PIPELINE','GOLEM_DECODER_FFN_PIPELINE_POLICY'}
    def environment(name):
        env={k:v.replace(str(cases[name]),'<case>') for k,v in prov[name]['environment'].items()}
        if 'GOLEM_MPI_ARGS' in env:env['GOLEM_MPI_ARGS']=re.sub(r'--cpu-set \S+','--cpu-set <placement>',env['GOLEM_MPI_ARGS'])
        return env
    ea,eb=environment('control'),environment('candidate')
    changes={k:dict(control=ea.get(k),candidate=eb.get(k)) for k in set(ea)|set(eb) if ea.get(k)!=eb.get(k)}
    if set(changes)-allowed:raise ValueError(f'unexpected environment differences: {set(changes)-allowed}')
    for name in cases:
        other=environment(name)
        if {k for k in set(ea)|set(other) if ea.get(k)!=other.get(k)}-allowed:
            raise ValueError('other formal candidate changed hardware/environment')
    for case in cases.values():
        measured=read(case.parent/'experiment_provenance.json')
        for path,sha in measured['source_sha256'].items():
            if digest(ROOT/path)!=sha:raise ValueError(f'production source changed after measurement: {path}')
        if measured['library_sha256']!=digest(ROOT/'install/lib/sst-elements-library/libgolem.so'):
            raise ValueError('installed library differs from measured library')
    bits=compare_outputs(control,candidate)
    logits=[np.load(c/'last_token_logits.npy') for c in (control,candidate)]
    if logits[0].shape!=logits[1].shape:raise ValueError('logits shape changed')
    bits['logits']=dict(elements=logits[0].size,fp16_bit_mismatches=int(np.count_nonzero(
        logits[0].astype('<f2').view('<u2')!=logits[1].astype('<f2').view('<u2'))))
    if any(v['fp16_bit_mismatches'] for v in bits.values()):raise ValueError('formal output bits changed')
    other_bits={}
    for name,case in cases.items():
        if name in ('control','candidate'):continue
        other_bits[name]=compare_outputs(control,case)
        other_logits=np.load(case/'last_token_logits.npy')
        other_bits[name]['logits']=dict(elements=other_logits.size,fp16_bit_mismatches=int(np.count_nonzero(
            logits[0].astype('<f2').view('<u2')!=other_logits.astype('<f2').view('<u2'))))
        if any(v['fp16_bit_mismatches'] for v in other_bits[name].values()):raise ValueError('other candidate output bits changed')
    micro={}
    for path in sorted(root.glob('micro*/layout2/result.json')):
        d=read(path)
        if d['status']!='PASS':raise ValueError(f'failed micro: {path}')
        if comparable_micro_inputs(path.parents[1])!=comparable_micro_inputs(root/'micro_control'):
            raise ValueError('micro inputs changed')
        base=read(root/'micro_control/layout2/result.json')
        if d['provenance']['sst_library_sha256']!=base['provenance']['sst_library_sha256']:
            raise ValueError('micro library changed')
        for field in ('hbm_read_bytes','hbm_write_bytes'):
            if d['dma_timing'][field]!=base['dma_timing'][field]:raise ValueError('micro dense traffic changed')
        if d['sfu_work']!=base['sfu_work']:raise ValueError('SFU token work changed')
        micro[path.parents[1].name]=dict(cycles=d['dma_timing']['cycles'],status='PASS',
            saved_percent=100*(1-d['dma_timing']['cycles']/base['dma_timing']['cycles']),
            optimizations=d['resources'].get('optimizations',{}))
    before,after=results['control'],results['candidate'];rows=[]
    for name,value in scopes(after).items():
        old=scopes(before)[name];rows.append(dict(scope=name,control_cycles=old,candidate_cycles=value,
                                                saved_cycles=old-value,saved_percent=100*(1-value/old)))
    state=read(root/'baseline_state.json');original=ROOT / 'test/history/legacy-primary'
    if digest(root/'baseline_guest')!=state['baseline_guest_sha256']:
        raise ValueError('preserved baseline guest changed')
    if subprocess.check_output(['git','diff','HEAD'],cwd=original,text=True)!=state['original_diff']:
        raise ValueError('original baseline diff changed')
    if digest(original/'install/lib/sst-elements-library/libgolem.so')!=state['original_lib_sha256']:
        raise ValueError('original baseline library changed')
    for path,sha in state['files'].items():
        snapshot=root/('baseline_libgolem.so' if path.endswith('libgolem.so') else 'baseline_snapshot/'+path)
        if digest(snapshot)!=sha:raise ValueError(f'baseline snapshot changed: {path}')
    phase_comparison={name:dict(control_cycles=d['cycles'],
        candidate_cycles=profiles['candidate']['additive_phases'][name]['cycles'],
        saved_cycles=d['cycles']-profiles['candidate']['additive_phases'][name]['cycles'])
        for name,d in profiles['control']['additive_phases'].items()}
    stage_comparison={name:dict(control_span_cycles=d['span_cycles'],
        candidate_span_cycles=profiles['candidate']['stages'][name]['span_cycles'],
        saved_span_cycles=d['span_cycles']-profiles['candidate']['stages'][name]['span_cycles'],
        control_last_core=d['last_core'],candidate_last_core=profiles['candidate']['stages'][name]['last_core'])
        for name,d in profiles['control']['stages'].items()}
    previous_profile=read(root/'critical_workers_before.json')
    previous_to_control={name:dict(previous_cycles=d['cycles'],
        control_cycles=profiles['control']['additive_phases'][name]['cycles'],
        increase_cycles=profiles['control']['additive_phases'][name]['cycles']-d['cycles'])
        for name,d in previous_profile['additive_phases'].items()}
    lm=read(root/'lm_tuning/summary.json')
    formal_candidates={(a.candidate if name=='candidate' else 'formal_control_final' if name=='control' else name):dict(cycles=d['embedding']['end_to_end_cycles'],decoder_cycles=d['dma_timing']['end_to_end_cycles'],status=d['status'],
                                norm_handoff=prov[name]['environment'].get('GOLEM_DECODER_NORM_FFN_HANDOFF','0'),
                                epoch_flags=prov[name]['environment'].get('GOLEM_DECODER_EPOCH_FLAGS','0'),
                                lm_parallel=prov[name]['environment']['GOLEM_DECODER_LM_N_PARALLEL'])
                       for name,d in results.items()}
    current_selected=a.candidate if rows[0]['saved_cycles']>0 else 'formal_control_final'
    current_best=min(rows[0]['control_cycles'],rows[0]['candidate_cycles'])
    summary=dict(status='PASS',candidate=a.candidate,comparison=rows,micro=micro,lm_tuning=lm,formal_candidates=formal_candidates,
                 additive_phase_comparison=phase_comparison,stage_comparison=stage_comparison,
                 previous_to_new_control=previous_to_control,
                 validation=validation,bitwise_comparison=bits,other_bitwise_comparison=other_bits,environment_changes=changes,
                 next_token=after['lm_head']['next_token']['token_id'],original_baseline_preserved=True,
                 previous_default_cycles=6674795.75,previous_default_decoder_cycles=5801221.375,
                 previous_best_optional_cycles=6672043.375,
                 same_library_guest_inputs_hardware=True,current_selected=current_selected,
                 reduction_vs_previous_default_percent=100*(1-current_best/6674795.75),
                 selected=current_selected if current_best<6674795.75 else 'previous_default')
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Reuse-window 后续优化实测','',
           '正式配置：S=1024、H=2048、FFN=8192、2 层、词表 128256、16 worker、8 MPI ranks。',
           '计时：token IDs 首次读取到完整 logits 最后 DMA ACK。1 GHz 下 cycles / 1e6 = ms。','',
           f'最佳非对照候选：`{a.candidate}`。','',
           '| 正式方案 | 完整端到端 cycles | 两层 decoder cycles | GM 交接 | epoch 标志 | LM N 并行度 |','|---|---:|---:|---:|---:|---:|',
           '| 保留的旧默认（旧 guest） | 6,674,795.750 | 5,801,221.375 | 0 | 0 | 4 |']
    lines += [f"| {name} | {d['cycles']:,.3f} | {d['decoder_cycles']:,.3f} | {d['norm_handoff']} | {d['epoch_flags']} | {d['lm_parallel']} |" for name,d in formal_candidates.items()]
    lines += ['', '| 范围 | 同构建对照 cycles | 候选 cycles | 降低 |','|---|---:|---:|---:|']
    lines += [f"| {r['scope']} | {r['control_cycles']:,.3f} | {r['candidate_cycles']:,.3f} | {r['saved_percent']:.3f}% |" for r in rows]
    lines += ['', '| 可加和端到端区间 | 对照 cycles | 候选 cycles | 减少 cycles |',
              '|---|---:|---:|---:|']
    lines += [f"| {name} | {d['control_cycles']:,.3f} | {d['candidate_cycles']:,.3f} | {d['saved_cycles']:,.3f} |"
              for name,d in phase_comparison.items()]
    lines += ['',f"完整端到端：{rows[0]['control_cycles']/1e6:.6f} → {rows[0]['candidate_cycles']/1e6:.6f} ms。",
              f"本轮最低延迟配置：`{current_selected}`；计入保留的旧默认后选择：`{summary['selected']}`。",
              f"相对上一轮默认 6.674796 ms 降低 {summary['reduction_vs_previous_default_percent']:.3f}%。本轮 guest 增加可选 epoch 协议，开关收益以本轮同 guest 对照为准。",'',
              '| FFN 单 worker 筛选 | cycles | 相对串行对照降低 |','|---|---:|---:|']
    lines += [f"| {name} | {d['cycles']:,.3f} | {d['saved_percent']:.3f}% |" for name,d in micro.items()]
    lines += ['', '延后阵列占用与原 policy 1 的结果相同；SFU 双 chunk 的 I/O 阶段与在途 SFU 请求存在交叠，但计数含等待，筛选未获得净收益，默认关闭。',
              '', '| LM head N 并行度 | cycles |','|---|---:|']
    lines += [f"| {n} | {d['cycles']:,.3f} |" for n,d in lm['cases'].items()]
    lines += ['', 'LM 微测试为独立输入及内存节点布局，绝对延迟不能直接与正式 LM 阶段比较。formal_combined 验证 N=1，formal_epoch_only 保留 N=4。',
              '', '实现：Norm2 在现有 GM 空闲区域生成 K-major 输入，FFN 直接读取；Norm2 的 HBM 写出保留以执行原有独立中间数值检查。',
              '仅在单个本地行组、相同 epoch/形状/地址/worker 且直接相邻的 Norm2→FFN 作业间命中；其余情况回退。',
              '首层仍清零带毒完成标志；后续层保留精确 epoch 等待与前层输出/完成 ACK，省掉重复的 32 次清零。',
              '所有 GEMM 保留 reuse-window；阵列、单操作数上下文、GM 容量和端口、时钟、NoC/HBM 时序、FP16 舍入与 dense 工作量不变。',
              '', f"逐位检查：末层 Y、final Norm、完整 logits 的差异均为 0；next token={summary['next_token']}。",
              '正式数值、实际 QKV 面板、16 worker 全部 dense GEMM、C 流量、每作业 DMA 字节和完成前 ACK 排空均 PASS。',
              '', '关键路径见 critical_workers_control.json / critical_workers_candidate.json；阶段跨度和后台等待不能相加为可消除时间。',
              '未把所有候选方案当作已完成：跨 panel 改变累加精度、跨层 attention 重排、扩大硬件和重新设计权重共享不属于本轮实现。',
              '运行细节、开关、筛选范围及未采用方向见 IMPLEMENTATION.md。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='PASS',selected=summary['selected'],comparison=rows[0],bitwise_comparison=bits),indent=2))


if __name__=='__main__':main()
