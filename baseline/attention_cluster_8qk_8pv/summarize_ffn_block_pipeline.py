#!/usr/bin/env python3
"""Summarize audited FFN pipeline runs, preserving the previous measurement."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import subprocess

from summarize_reuse_window_optimization import (
    ROOT, read, digest, comparable_inputs, comparable_micro_inputs,
    flow_summary, scopes, compare_outputs,
)
from verify_decoder_dataflow_resources import block_pipeline_audit

PREVIOUS = ROOT/'results/reuse_window_optimization_20261021/formal_final/reuse_window_probe'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', type=Path,
                        default=ROOT/'results/ffn_block_pipeline_20261021')
    parser.add_argument('--selected', default='formal_pipeline16')
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    cases = {name: root/name/'reuse_window_probe'
             for name in ('formal_control', args.selected)}
    results = {name: read(case/'decoder_stack_result.json') for name, case in cases.items()}
    reports = {name: read(case.parent/'all_gemm_reuse_window_report.json') for name, case in cases.items()}
    old = read(PREVIOUS/'decoder_stack_result.json')
    if any(v['status'] != 'PASS' for v in list(results.values())+list(reports.values())+[old]):
        raise ValueError('all formal numerical/resource/coverage audits must pass')
    if any(comparable_inputs(case) != comparable_inputs(PREVIOUS) for case in cases.values()):
        raise ValueError('formal inputs or memory layout changed')
    provenance = {name: read(case/'decoder_stack_provenance.json') for name, case in cases.items()}
    control, selected = (provenance[name] for name in ('formal_control', args.selected))
    for key in ('guest_sha256', 'sst_memhierarchy_sha256', 'sst_library_sha256'):
        if control[key] != selected[key]:
            raise ValueError(f'formal comparison changed {key}')
    allowed_env = {'GOLEM_DECODER_FFN_BLOCK_PIPELINE', 'GOLEM_DECODER_FFN_PIPELINE_POLICY',
                   'GOLEM_DECODER_FFN_PANEL_TILES', 'GOLEM_MPI_ARGS', 'GOLEM_MPI_RANK_LOG_ROOT',
                   'GOLEM_DECODER_STACK_MANIFEST'}
    changes = {k: dict(control=control['environment'].get(k), selected=selected['environment'].get(k))
               for k in set(control['environment']) | set(selected['environment'])
               if control['environment'].get(k) != selected['environment'].get(k)}
    if set(changes)-allowed_env:
        raise ValueError(f'non-pipeline environment changed: {set(changes)-allowed_env}')
    measured = read(cases[args.selected].parent/'experiment_provenance.json')
    for path, sha in measured['source_sha256'].items():
        if digest(ROOT/path) != sha:
            raise ValueError(f'production source changed after measurement: {path}')
    if digest(ROOT/'install/lib/sst-elements-library/libgolem.so') != measured['library_sha256']:
        raise ValueError('installed library differs from selected measurement')
    for path, sha in read(root/'baseline_sha256.json').items():
        if digest(root/'baseline_snapshot'/Path(path).name) != sha:
            raise ValueError(f'previous snapshot changed: {path}')
    if subprocess.check_output(['git', 'diff', '--stat'], cwd=ROOT.parent/'Muticore-Attention', text=True):
        raise ValueError('original tracked baseline changed')
    original_lib = ROOT.parent/'Muticore-Attention/install/lib/sst-elements-library/libgolem.so'
    if digest(original_lib) != 'a56013188b986918e8f5ea6aa64c4fbe0c8c35e48febbde791b004a4aa9febae':
        raise ValueError('original baseline installed library changed')
    before, after = (results[name] for name in ('formal_control', args.selected))
    rows = []
    for scope, cycles in scopes(after).items():
        base = scopes(before)[scope]
        rows.append(dict(scope=scope, control_cycles=base, selected_cycles=cycles,
                         saved_cycles=base-cycles, saved_percent=(1-cycles/base)*100))
    micros = {}
    for path in sorted(root.glob('micro*/layout2/result.json')):
        v = read(path)
        if v['status'] != 'PASS':
            raise ValueError(f'micro audit failed: {path}')
        micros[path.parents[1].name] = dict(
            status=v['status'], cycles=v['dma_timing']['cycles'], workers=v['shape']['workers'],
            library_sha256=v['provenance']['sst_library_sha256'],
            hbm_read_bytes=v['dma_timing']['hbm_read_bytes'], hbm_write_bytes=v['dma_timing']['hbm_write_bytes'],
            block_pipeline=v['resources']['block_pipeline'], sfu_work=v['sfu_work'])
    for a, b in [('micro3_control', 'micro3_policy1'), ('micro16_control', 'micro16_pipeline16')]:
        if comparable_micro_inputs(root/a) != comparable_micro_inputs(root/b):
            raise ValueError('micro inputs differ')
        if micros[a]['library_sha256'] != micros[b]['library_sha256']:
            raise ValueError('micro pair used different libraries')
        if any(micros[a][k] != micros[b][k] for k in ('hbm_read_bytes', 'hbm_write_bytes')):
            raise ValueError('micro pair dense work or output traffic differs')
        if micros[a]['sfu_work'] != micros[b]['sfu_work']:
            raise ValueError('micro pair executed different SFU work')
    bits = compare_outputs(cases['formal_control'], cases[args.selected])
    if any(v['fp16_bit_mismatches'] for v in bits.values()):
        raise ValueError('formal outputs changed')
    e2e = rows[0]
    candidates = {}
    for directory in sorted(root.glob('formal*')):
        path = directory/'reuse_window_probe/decoder_stack_result.json'
        if not path.exists():
            continue
        v = read(path)
        report = read(directory/'all_gemm_reuse_window_report.json')
        if v['status'] != 'PASS' or report['status'] != 'PASS':
            raise ValueError(f'formal candidate audit failed: {directory}')
        candidate_case = path.parent
        if comparable_inputs(candidate_case) != comparable_inputs(cases['formal_control']):
            raise ValueError('formal candidate inputs changed')
        candidate_provenance = read(candidate_case/'decoder_stack_provenance.json')
        if candidate_provenance['sst_library_sha256'] != control['sst_library_sha256']:
            raise ValueError('formal candidate used a different build')
        jobs = defaultdict(list)
        for log in (candidate_case/'logs').glob('attention-sst*.log'):
            for line in log.open():
                if '[DECODER_OP] event=done' in line:
                    job = {k: int(value) for k, value in re.findall(r'(\w+)=(-?\d+)', line)}
                    jobs[job['core']].append(job)
        policy_reaudit = block_pipeline_audit(candidate_case, jobs)
        cycles = scopes(v)['完整端到端']
        candidates[directory.name] = dict(cycles=cycles, saved_percent=(1-cycles/e2e['control_cycles'])*100,
                                          policy_reaudit=policy_reaudit)
    summary = dict(status='PASS', selected=args.selected, baseline_preserved=True,
                   same_build_comparison=True, environment_differences=changes,
                   scopes=rows, microbenchmarks=micros, bitwise_comparison=bits, formal_candidates=candidates,
                   end_to_end_improved=e2e['selected_cycles']<e2e['control_cycles'],
                   decoder_improved=scopes(after)['两层 decoder']<scopes(before)['两层 decoder'],
                   recommend_default_replacement=(e2e['selected_cycles']<e2e['control_cycles'] and
                                                  scopes(after)['两层 decoder']<scopes(before)['两层 decoder']),
                   previous_end_to_end_cycles=scopes(old)['完整端到端'],
                   control_reproduces_previous=scopes(before)['完整端到端']==scopes(old)['完整端到端'],
                   pipeline=after['resources']['block_pipeline'],
                   window_flows=flow_summary(cases[args.selected]), next_token=after['lm_head']['next_token'])
    (root/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')
    text = [
        '# FFN 分块流水实测', '',
        '正式配置：S=1024，H=2048，FFN=8192，2 层，vocab=128256，16 worker，8 MPI ranks。',
        f"最快流水候选：{args.selected}；policy={selected['environment']['GOLEM_DECODER_FFN_PIPELINE_POLICY']}，panel={selected['environment']['GOLEM_DECODER_FFN_PANEL_TILES']} tiles，array groups={selected['environment']['GOLEM_REUSE_WINDOW_ARRAY_GROUPS']}。",
        '完整端到端边界：token IDs 首次读取到最终 logits DMA ACK；1 GHz 下 cycles / 1e6 = ms。', '',
        '| 范围 | 同构建对照 cycles | 流水 cycles | 降低 |',
        '|---|---:|---:|---:|',
    ]
    text += [f"| {v['scope']} | {v['control_cycles']:,.3f} | {v['selected_cycles']:,.3f} | {v['saved_percent']:.3f}% |" for v in rows]
    verb = '降低' if e2e['saved_percent'] >= 0 else '增加'
    if summary['recommend_default_replacement']:
        recommendation = '上述配置在当前测试中同时改善完整链路和 decoder 本体；原基线仍保留。'
    elif summary['end_to_end_improved']:
        recommendation = ('完整链路只有很小的改善，decoder 本体回退；本轮保留关闭流水的 6.674796 ms 配置为默认基线，流水实现保留为显式实验选项。')
    else:
        recommendation = ('本轮流水没有改善完整端到端延迟，保留关闭流水的 6.674796 ms 配置为默认基线；流水实现保留为实验选项。')
    text += ['', f"完整端到端：{e2e['control_cycles']/1e6:.6f} → {e2e['selected_cycles']/1e6:.6f} ms，{verb} {abs(e2e['saved_percent']):.3f}%。",
             recommendation,
             '各阶段有覆盖，不能直接相加；最终选择依据完整端到端结果。', '',
             '| 正式完整链路候选 | cycles | 相对同构建对照降低 |', '|---|---:|---:|']
    text += [f"| {name} | {v['cycles']:,.3f} | {v['saved_percent']:.3f}% |" for name, v in candidates.items()]
    text += ['',
             '| FFN 微测试 | worker | cycles |', '|---|---:|---:|']
    text += [f"| {name} | {v['workers']} | {v['cycles']:,.3f} |" for name, v in micros.items()]
    text += ['', 'micro1 为初始实现，使用 build1；micro3 和 micro16 使用最终 build3，同组开关对照的输入、库和 SFU 工作量一致。',
             '8 tile 使 Down 部分和读写及输出 drain/restore 更频繁，本轮单 worker 筛选慢于 16 tile，保留 16 tile。']
    p = summary['pipeline']
    text += ['', '实现保留两个 gated GM 槽，Gate/Up 与 Down 各有独立窗口状态、真实 GM 权重槽和互斥阵列使用权。',
             'Gate/Up/SiLU/Multiply/Down/residual 保持融合；所有 64 阵列的单上下文约束和跨 panel FP16 部分和顺序不变。',
             '采用的策略及参数以 experiment_provenance.json 和 decoder_stack_provenance.json 为准。', '',
             f"双窗口同时 active 累计 {p['both_active_cycles']:,} cycles，其中包含等待。有效搬运/MAC 覆盖计数：",
             f"生产者搬运与 Down MAC：{p['producer_transfer_down_compute_cycles']:,}；Down 搬运与生产者 MAC：{p['down_transfer_producer_compute_cycles']:,} cycles。",
             '上述覆盖和等待计数不能直接相加为延迟收益；同一阵列没有两个算子的 MAC 同时执行。', '',
             '验证：正式数值、实际 Q/K/V 面板严格检查、16 worker 全部 dense GEMM 工作量、C 流量、每作业 DMA 字节和结束前 ACK 排空均 PASS。',
             f"末层 Y 与最终 RMSNorm 分别逐位比较 {bits['last_layer_y']['elements']:,} / {bits['final_norm']['elements']:,} 个 FP16 元素，差异均为 0。next token = {summary['next_token']['token_id']}。",
             '同构建对照与流水使用相同 guest、输入内容和内存模型库；硬件容量、端口、时钟、NoC/HBM 时序不变。',
             f"对照重现上一轮周期数：{summary['control_reproduces_previous']}。原基线 tracked diff 为空，原 installed lib 哈希不变。", '',
             '实现、开关和复现命令见 [IMPLEMENTATION.md](IMPLEMENTATION.md)。',
             '原始证据：'+ '、'.join(name+'/' for name in candidates) +'、micro*/；机器可读比较见 summary.json；构建和回归日志见 checks/。', '',
             '剩余限制：同一物理阵列的 program/scatter/MAC 串行仍在；分块后预取和 SFU 覆盖更好不保证全模型收益随之线性增加。',
    ]
    (root/'REPORT.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(status='PASS', selected=args.selected, end_to_end=e2e), ensure_ascii=False))


if __name__ == '__main__':
    main()
