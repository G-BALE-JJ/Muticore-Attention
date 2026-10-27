#!/usr/bin/env python3
"""Compare audited experiments without rerunning SST or changing tolerances."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'results/reuse_window_optimization_20261021'
PREVIOUS=ROOT/'results/all_gemm_reuse_window_v4_formal_groups1_l2_s1024_mpi8_20261020'
ORIGINAL=ROOT / 'Muticore-Attention/results/prefill_generalization_20261016/formal_l2_s1024'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparable_inputs(case):
    manifest=read(case/'decoder_stack_inputs.json')
    for entry in manifest['preloads']:entry.pop('path',None)
    return manifest


def comparable_micro_inputs(directory):
    manifest=read(directory/'inputs/inputs.json')
    for entry in manifest['preloads']:entry.pop('path',None)
    return manifest


def flow_summary(case):
    flows=defaultdict(lambda:defaultdict(int));sharing=defaultdict(int)
    for log in (case/'logs').glob('attention-sst*.log'):
        for line in log.open():
            if '[GEMM_WINDOW_FLOW]' in line:
                fields=dict(re.findall(r'(\w+)=([^\s]+)',line))
                values={k:int(v) for k,v in fields.items() if k!='operator'}
                if sum(v for k,v in values.items() if k.startswith('phase_'))!=values['active_cycles']:
                    raise ValueError('window phase accounting does not partition active cycles')
                target=flows[fields['operator']];target['records']+=1
                for k,v in values.items():
                    if k not in ('core','epoch','stage','groups','early_a','output_overlap'):target[k]+=v
            if '[GEMM_WEIGHT_SHARE]' in line:
                values={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)}
                sharing['records']+=1
                for k in ('requests','server_requests','cache_hits','hbm_bytes','gm_read','gm_write'):
                    sharing[k]+=values[k]
    return dict(window_flows=dict(flows),weight_sharing=dict(sharing))


def scopes(result):
    out={'完整端到端':result['embedding']['end_to_end_cycles'],
         '两层 decoder':result['dma_timing']['end_to_end_cycles'],
         '最终 RMSNorm':result['final_norm']['rmsnorm_cycles'],
         'LM head':result['lm_head']['lm_head_cycles']}
    for layer,stages in result['lifecycle']['stage_spans'].items():
        out[f'第 {layer} 层 FFN span']=stages['4']['span_cycles']
        out[f'第 {layer} 层 attention']=result['dma_timing']['layers'][layer]['attention_cycles']
    return out


def compare_outputs(previous_case,case):
    import numpy as np
    from run_decoder_layer import read_bands
    m=read(case/'decoder_stack_inputs.json');hidden=m['hidden'];rows=m['sequence']//4
    offsets={'last_layer_y':m['layout']['layers'][-1]['output'],
             'final_norm':m['layout']['final_norm']['output']}
    out={}
    for name,offset in offsets.items():
        before=read_bands(previous_case,offset,hidden,rows=rows)
        after=read_bands(case,offset,hidden,rows=rows)
        out[name]=dict(elements=int(after.size),
            fp16_bit_mismatches=int(np.count_nonzero(before.view('u2')!=after.view('u2'))))
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root',type=Path,default=DEFAULT)
    parser.add_argument('--selected',default='formal_final')
    args=parser.parse_args();root=args.experiment_root.resolve()
    selected=root/args.selected;case=selected/'reuse_window_probe';previous_case=PREVIOUS/'reuse_window_probe'
    report=read(selected/'all_gemm_reuse_window_report.json')
    previous_report=read(PREVIOUS/'all_gemm_reuse_window_report.json')
    original=read(ORIGINAL/'decoder_stack_result.json')
    old=read(previous_case/'decoder_stack_result.json');new=read(case/'decoder_stack_result.json')
    if any(r['status']!='PASS' for r in (report,previous_report,original,old,new)):
        raise ValueError('all compared formal results must pass existing audits')
    if comparable_inputs(case)!=comparable_inputs(previous_case):raise ValueError('v4 inputs/layout changed')
    if comparable_inputs(case)!=comparable_inputs(ORIGINAL):raise ValueError('original inputs/layout changed')
    a=read(case/'decoder_stack_provenance.json');b=read(previous_case/'decoder_stack_provenance.json')
    for key in ('guest_sha256','sst_memhierarchy_sha256'):
        if a[key]!=b[key]:raise ValueError(f'comparison changed {key}')
    allowed_env={'GOLEM_REUSE_WINDOW_EARLY_A','GOLEM_REUSE_WINDOW_OUTPUT_OVERLAP',
        'GOLEM_REUSE_WINDOW_QKV_INPUT_PACK','GOLEM_DECODER_GAMMA_REUSE','GOLEM_REUSE_WEIGHT_COHORT',
        'GOLEM_MPI_ARGS','GOLEM_MPI_RANK_LOG_ROOT','GOLEM_DECODER_STACK_MANIFEST'}
    env_changes={k:{'previous':b['environment'].get(k),'selected':a['environment'].get(k)}
        for k in set(a['environment'])|set(b['environment'])
        if a['environment'].get(k)!=b['environment'].get(k)}
    if set(env_changes)-allowed_env:raise ValueError(f'hardware/environment changed: {set(env_changes)-allowed_env}')
    for path,sha in read(root/'baseline_sha256.json').items():
        if digest(root/'baseline_snapshot'/Path(path).name)!=sha:raise ValueError(f'v4 snapshot changed: {path}')
    software={'decoderOpJob.h','projectionJob.h','decoderMemoryLayout.h','roccAnalog.h','globalmemory.h'}
    for path,old_hash in b['source_sha256'].items():
        if '/.all_gemm_reuse_runner_' in path or Path(path).name in software:continue
        if a['source_sha256'].get(path)!=old_hash:raise ValueError(f'non-software source changed: {path}')
    recorded=read(selected/'experiment_provenance.json')
    for path,sha in recorded['source_sha256'].items():
        if digest(ROOT/path)!=sha:raise ValueError(f'measured source changed: {path}')
    if digest(ROOT/'install/lib/sst-elements-library/libgolem.so')!=recorded['library_sha256']:
        raise ValueError('installed library differs from selected simulation')
    rows=[]
    for name,cycles in scopes(new).items():
        before=scopes(old)[name]
        rows.append(dict(scope=name,before_cycles=before,after_cycles=cycles,
                         reduction_percent=100*(before-cycles)/before))
    micros=[]
    micro_reference={}
    for name in ('micro_final_control','micro_final_early_a','micro_final_output','micro_final_combined','micro16_control','micro16_cohort4'):
        result=read(root/name/'layout2/result.json')
        if result['status']!='PASS':raise ValueError(f'failed micro: {name}')
        workers=result['shape']['workers'];provenance=result['provenance']
        if workers==1 and provenance['sst_library_sha256']!=recorded['library_sha256']:
            raise ValueError(f'micro differs from final build: {name}')
        signature=({k:provenance[k] for k in ('guest_sha256','sst_library_sha256','sst_memhierarchy_sha256')},
                   comparable_micro_inputs(root/name))
        if workers not in micro_reference:micro_reference[workers]=signature
        if micro_reference[workers]!=signature:raise ValueError(f'unmatched micro inputs/build: {name}')
        micros.append(dict(name=name,workers=result['shape']['workers'],cycles=result['dma_timing']['cycles'],
                           hbm_read_bytes=result['dma_timing']['hbm_read_bytes'],
                           **flow_summary(root/name/'layout2')))
    single=micros[0]['cycles'];multi=micros[-2]['cycles']
    for row in micros:row['change_percent']=100*(row['cycles']/(single if row['workers']==1 else multi)-1)
    old_cycles=old['embedding']['end_to_end_cycles'];cycles=new['embedding']['end_to_end_cycles']
    original_cycles=original['embedding']['end_to_end_cycles']
    summary=dict(status='PASS',selected=args.selected,formal_comparison=rows,micro_experiments=micros,
        original_cycles=original_cycles,previous_cycles=old_cycles,selected_cycles=cycles,
        reduction_vs_previous_percent=100*(old_cycles-cycles)/old_cycles,
        reduction_vs_original_percent=100*(original_cycles-cycles)/original_cycles,
        inputs_layout_guest_memory_model_match=True,environment_changes=env_changes,
        next_token=new['lm_head']['next_token']['token_id'],flow_evidence=flow_summary(case),
        pre_ack_fix_cycles=read(root/'formal_local/reuse_window_probe/decoder_stack_result.json')['embedding']['end_to_end_cycles'],
        output_comparison_to_v4=compare_outputs(previous_case,case),
        formal_report=str(selected/'all_gemm_reuse_window_report.json'))
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Reuse-window 流水优化实测（2026-10-21）','',
        f'正式测试：S=1024，H=2048，FFN=8192，2 层，vocab=128256，16 worker，8 MPI ranks。',
        '完整端到端边界：token IDs 首次读取到最终 logits 的 DMA ACK；1 GHz 下 cycles / 1e6 = ms。',
        '原基线目录及 v4 结果保留。所有 GEMM 保留 reuse-window，数组/GM 容量、端口、时钟、NoC/HBM 时序未调整。','',
        '| 方案 | 完整端到端 cycles | 延迟 ms |','|---|---:|---:|',
        f'| 原方案 | {original_cycles:,.3f} | {original_cycles/1e6:.6f} |',
        f'| 上轮 v4 | {old_cycles:,.3f} | {old_cycles/1e6:.6f} |',
        f'| 本轮 {args.selected} | {cycles:,.3f} | {cycles/1e6:.6f} |','',
        f"较 v4 降低 **{summary['reduction_vs_previous_percent']:.3f}%**，较原方案降低 **{summary['reduction_vs_original_percent']:.3f}%**。",'',
        '采用：提前读取 A；Wo/Gate/Up 的独立输出写回允许下一块提前准备；QKV 每行组一次成对重排并复用 GM 输入；每个 RMSNorm 作业只从 HBM 读一次 gamma。',
        '实际 output_compute_overlap 为 0：本轮输出侧收益来自写回期间提前准备下一块，并未观测到 MAC 与输出写回同时运行。',
        'Down 保留初始 C 和输出同步；同一阵列的 program/scatter/MAC 仍遵守单上下文互斥。所有消费依赖真实 GM/NoC/数组 ACK。','',
        '| 范围 | v4 cycles | 本轮 cycles | 降低 |','|---|---:|---:|---:|']
    lines += [f"| {r['scope']} | {r['before_cycles']:,.3f} | {r['after_cycles']:,.3f} | {r['reduction_percent']:.3f}% |" for r in rows]
    lines += ['', '各范围存在覆盖，不能相加。主要改善集中在 attention/projection；FFN 提升有限，第二层 FFN 和 LM head 略有回退，最终配置依据完整端到端结果选取。','',
        '| FFN + Down + residual 微测试 | worker | cycles | 相对同 worker 对照变化 |',
        '|---|---:|---:|---:|']
    lines += [f"| {r['name']} | {r['workers']} | {r['cycles']:,.3f} | {r['change_percent']:+.3f}% |" for r in micros]
    sharing=micros[-1]['weight_sharing']
    lines += ['',f"共享微测试实际权重 HBM 字节：{sharing.get('hbm_bytes',0):,}；共享服务显式 GM 读/写：{sharing.get('gm_read',0):,} / {sharing.get('gm_write',0):,}。",
        '消费端 GM 写替代原权重 DMA 的落地写；净增 GM 流量是 leader 缓存的 DMA 落地和再读，不能把服务写入量全部算作净增流量。',
        f"权重 HBM 降幅 {100*(1-sharing['hbm_bytes']/sharing['gm_write']):.3f}%；净增 GM 字节 {sharing['hbm_bytes']+sharing['gm_read']:,}。",
        '跨 worker 共享使用 4 槽真实 GM 缓存及现有 NoC，作为独立实验保留；未纳入推荐配置。缓存容量、进度差和分发流量都会限制收益，减少 HBM 字节不能直接当作延迟收益。','',
        '验证：正式端到端数值、实际 Q/K/V 面板逐元素严格检查、16 worker GEMM 覆盖、C 流量、每作业 DMA 字节核对、完成前 ACK 排空全部 PASS；输入内容、布局、guest 和内存模型库与旧结果匹配。',
        f"next token = {summary['next_token']}。窗口独占阶段计数之和逐条等于 active_cycles；后台等待/覆盖计数不能直接相加为关键路径。",'',
        '对 v4 的末层 Y 和最终 RMSNorm 各 2,097,152 个 FP16 元素逐位比较，差异数分别为 '
        f"{summary['output_comparison_to_v4']['last_layer_y']['fp16_bit_mismatches']} 和 "
        f"{summary['output_comparison_to_v4']['final_norm']['fp16_bit_mismatches']}。",'',
        '共享实验的 wait_b_dma 表示外部 fetch 完成前的等待，包含缓存、NoC 和消费端 GM 落地，并非纯 HBM RTT。','',
        '这些优化仍未消除单上下文阵列的串行编程/计算；也未把全模型改成逐块跨层流水。当前最优仅指本轮已测配置。','',
        '复现正式配置：','```bash','source scripts/env_local_install.sh >/dev/null',
        'OPENBLAS_NUM_THREADS=1 python3 baseline/attention_cluster_8qk_8pv/run_all_gemm_reuse_window_baseline.py \\',
        '  --artifact-root results/reuse_window_optimized_reproduction --shape formal --run \\',
        '  --window-array-groups 1 --window-early-a 1 --window-output-overlap 1 \\',
        '  --qkv-input-pack 1 --gamma-reuse 1 --weight-cohort 0 --cpu-set 0-7','```','',
        f'原始证据：{args.selected}/、formal_local/、micro_*/、micro16_*/；summary.json 含可机器读取的比较与计数。',
        '单 worker 消融与正式最终结果使用相同构建；16 worker 共享/不共享使用同一 build3，两者均已通过独立审计。',
        'formal_local 是 ACK 边界修复前的 K=32 测量；formal_final 使用修复后的最终构建。短 K/乱序 ACK 回归保存在 checks/ack_regression.log。']
    lines += [f"两次正式完整测量的周期数：{summary['pre_ack_fix_cycles']:,.3f} / {cycles:,.3f}；最终构建与其记录的源码、动态库哈希一致。"]
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('flow_evidence','micro_experiments')},indent=2))


if __name__=='__main__':main()
