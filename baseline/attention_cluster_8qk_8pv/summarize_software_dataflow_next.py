#!/usr/bin/env python3
"""Select measured software schedules with fixed hardware and bitwise outputs."""
import argparse
import json
from pathlib import Path
import re
import subprocess

from summarize_reuse_window_optimization import ROOT, read, digest, comparable_inputs, scopes
from summarize_software_stage_chain import output_bits, job_totals
from summarize_gm_prefetch import flow_stats
from run_all_gemm_reuse_window_baseline import summarize, save_report, BASELINE_RESULT
from verify_decoder_dataflow_resources import verify
from analyze_reuse_window_critical_path import analyze


FLAGS = ('layer_residency', 'lm_input_residency')
ENV_FLAGS = {'GOLEM_DECODER_LAYER_RESIDENCY', 'GOLEM_DECODER_LM_INPUT_RESIDENCY',
             'GOLEM_REUSE_WINDOW_B_WINDOW_BRIDGE'}


def inputs(case):
    d = comparable_inputs(case)
    for k in FLAGS:
        d.pop(k, None)
    return d


def environment(p, case):
    d = {k: v.replace(str(case), '<case>') for k, v in p['environment'].items()}
    if 'GOLEM_MPI_ARGS' in d:
        d['GOLEM_MPI_ARGS'] = re.sub(r'--cpu-set \S+', '--cpu-set <placement>', d['GOLEM_MPI_ARGS'])
    return d


def audit(root):
    names = ('formal_control', 'formal_resident_lm', 'formal_all')
    cases = {n: root/n/'reuse_window_probe' for n in names}
    results = {}; provenance = {}; audits = {}; profiles = {}; flows = {}; totals = {}
    for n, case in cases.items():
        d = read(case/'decoder_stack_result.json')
        if d['status'] != 'PASS':
            raise ValueError('full numerical/lifecycle audit failed: '+n)
        report = summarize(case, BASELINE_RESULT)
        save_report(case.parent, report)
        if report['status'] != 'PASS':
            raise ValueError('all-GEMM coverage failed: '+n)
        results[n] = d
        provenance[n] = read(case/'decoder_stack_provenance.json')
        audits[n] = verify(case)
        flows[n] = flow_stats(case, provenance[n]['environment'])
        profiles[n] = analyze(case)
        (root/(n+'_critical_path.json')).write_text(json.dumps(profiles[n], indent=2)+'\n')
        totals[n] = job_totals(case)
    control = names[0]; ref = provenance[control]; base_env = environment(ref, cases[control])
    matrix = read(cases[control]/'decoder_stack_inputs.json')
    tensor = matrix['sequence']*matrix['hidden']*2
    layer_saved = tensor*matrix['layers']
    lm_saved = (results[control]['lm_head']['chunks']-1)*16*matrix['hidden']*2
    changes = {}; bits = {}; timing = {}
    for n in names[1:]:
        if inputs(cases[n]) != inputs(cases[control]):
            raise ValueError('input tensors or layout differ: '+n)
        for key in ('guest_sha256', 'sst_library_sha256', 'sst_memhierarchy_sha256'):
            if provenance[n][key] != ref[key]:
                raise ValueError('measured binaries differ: '+key)
        env = environment(provenance[n], cases[n])
        changes[n] = {k: dict(control=base_env.get(k), candidate=env.get(k))
                      for k in set(base_env)|set(env) if base_env.get(k) != env.get(k)}
        expected = {'GOLEM_DECODER_LAYER_RESIDENCY', 'GOLEM_DECODER_LM_INPUT_RESIDENCY'}
        if n == 'formal_all':
            expected.add('GOLEM_REUSE_WINDOW_B_WINDOW_BRIDGE')
        if set(changes[n]) != expected or any(v != dict(control='0', candidate='1') for v in changes[n].values()):
            raise ValueError('unexpected hardware/software configuration difference: '+repr(changes[n]))
        if flows[n]['traffic'] != flows[control]['traffic']:
            raise ValueError('dense GEMM MAC/A/B/C work changed: '+n)
        opt = audits[n]['optimizations']['layer_residency']
        if opt['saved_hbm_read_bytes'] != layer_saved:
            raise ValueError('layer residency read saving differs')
        payload = opt['actual_projection_payload']
        if payload['saved_hbm_read_bytes'] != tensor*(matrix['layers']-1):
            raise ValueError('actual projection GM byte count differs')
        for epoch, tick in payload['first_input_ticks'].items():
            if results[n]['dma_timing']['layers'][str(epoch)]['first_input_dma_tick'] != tick:
                raise ValueError('projection timing boundary differs from real GM payload')
        if totals[control]['hbm_read']-totals[n]['hbm_read'] != tensor+lm_saved:
            raise ValueError('decoder/final-Norm/LM DMA savings differ')
        if totals[control]['hbm_write'] != totals[n]['hbm_write']:
            raise ValueError('Y/logits output materialization changed')
        if results[control]['lm_head']['input_and_weight_dma_bytes']-results[n]['lm_head']['input_and_weight_dma_bytes'] != lm_saved:
            raise ValueError('LM input DMA savings differ')
        bits[n] = output_bits(cases[control], cases[n])
        before = scopes(results[control])
        timing[n] = {k: dict(control_cycles=before[k], candidate_cycles=v,
                            reduction_percent=100*(1-v/before[k]))
                     for k, v in scopes(results[n]).items()}
    state = read(root/'before_state.json')
    old = ROOT/state['baseline']; old_result = read(old/'decoder_stack_result.json')
    if inputs(old) != inputs(cases[control]):
        raise ValueError('previous selected inputs differ')
    for scope in ('两层 decoder', '完整端到端'):
        if scopes(old_result)[scope] != scopes(results[control])[scope]:
            raise ValueError('same-build control did not reproduce previous best')
    eligible = [n for n in names[1:] if all(timing[n][k]['reduction_percent'] > 0
                                          for k in ('两层 decoder', '完整端到端'))]
    selected = min(eligible, key=lambda n: scopes(results[n])['完整端到端']) if eligible else control
    previous_bits = output_bits(old, cases[selected])
    fixed = read(root/'fixed_hardware_sources.json')
    for name, sha in fixed.items():
        if digest(ROOT/name) != sha:
            raise ValueError('hardware/architecture source changed: '+name)
    original = ROOT.parent/'Muticore-Attention'
    if (subprocess.check_output(['git', 'diff', 'HEAD'], cwd=original, text=True) != state['original_diff'] or
        digest(original/'install/lib/sst-elements-library/libgolem.so') != state['original_library']):
        raise ValueError('original baseline changed')
    frozen = read(root/'measured_binaries.json')
    for name, sha in frozen['sources'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('measured runtime source changed: '+name)
    if digest(ROOT/'install/lib/sst-elements-library/libgolem.so') != frozen['library'] or frozen['library'] != ref['sst_library_sha256']:
        raise ValueError('measured/installed library differs')
    small = {}
    for n in ('small_control', 'small_layer_only', 'small_lm_only', 'small_resident_lm', 'small_all'):
        case = root/n/'reuse_window_probe'; d = read(case/'decoder_stack_result.json')
        if d['status'] != 'PASS':
            raise ValueError('small screening did not pass: '+n)
        small[n] = dict(timing=scopes(d), library=read(case/'decoder_stack_provenance.json')['sst_library_sha256'])
    micro = {}
    for n in ('micro_control', 'micro_wave', 'micro_stream', 'micro_panel8', 'micro_bridge'):
        d = read(root/n/'layout2/result.json')
        if d['status'] != 'PASS':
            raise ValueError('micro screening did not pass: '+n)
        micro[n] = dict(cycles=d['dma_timing']['cycles'], library=d['provenance']['sst_library_sha256'])
    out = dict(status='PASS', selected=selected, formal_comparison=timing,
               selected_timing=scopes(results[selected]), bitwise_comparison=bits,
               previous_selected_bitwise_comparison=previous_bits, environment_changes=changes,
               software_only=True, hardware_sources_unchanged=len(fixed), original_preserved=True,
               same_formal_binaries_inputs_hardware=True, measured_binaries=frozen,
               saved_hbm_read_bytes=layer_saved+lm_saved, saved_hbm_write_bytes=0,
               layer_saved_hbm_read_bytes=layer_saved, lm_saved_hbm_read_bytes=lm_saved,
               decoder_job_hbm_totals=totals, resources=audits, gemm_flows=flows,
               small_screening=small, micro_screening=micro,
               notes=['Formal selection uses identical binaries; micro bridge and the first micro control use different builds.',
                      'Small tests select candidates; formal shape is the acceptance criterion.',
                      'HBM Y output remains materialized; cached Y removes downstream rereads only.',
                      'All timing values are simulated cycles at 1 GHz, not host execution time.'])
    (root/'summary.json').write_text(json.dumps(out, indent=2, ensure_ascii=False)+'\n')
    return out


def report(root, d):
    selected = d['selected']; base = read(root/'formal_control/reuse_window_probe/decoder_stack_result.json')
    rows = [('上轮最佳／同构建对照', scopes(base))]
    rows += [(n, {k: v['candidate_cycles'] for k, v in row.items()}) for n, row in d['formal_comparison'].items()]
    lines = ['# 固定硬件上的下一轮软件数据流优化', '',
             '正式形状：S=1024、H=2048、FFN=8192、两层、词表 128256，16 worker，8 MPI rank，1 GHz。',
             '两层 decoder：第一层输入首次读取至第二层 Y 最后 DMA ACK。完整端到端：token IDs 首次读取至完整 logits 最后 DMA ACK。', '',
             '| 配置 | 两层 decoder ms | 完整端到端 ms |', '|---|---:|---:|']
    lines += [f"| {n} | {r['两层 decoder']/1e6:.6f} | {r['完整端到端']/1e6:.6f} |" for n, r in rows]
    lines += ['', f'本轮选择：`{selected}`。', '',
              '| 范围 | 对照 cycles | 选中配置 cycles | 降低 |', '|---|---:|---:|---:|']
    if selected in d['formal_comparison']:
        lines += [f"| {k} | {v['control_cycles']:,.3f} | {v['candidate_cycles']:,.3f} | {v['reduction_percent']:.3f}% |"
                  for k, v in d['formal_comparison'][selected].items()]
    lines += ['', '阶段 span 可覆盖，不能相加；关键路径分区见 formal_*_critical_path.json。', '',
              '## 实现与边界', '',
              '1. FFN 残差 Add 消费完 H1 的每个片段后，将最终 Y 原位写入现有 GM 0xb0000。下一层已有的 Norm1→QKV 融合路径从 GM 读取完整隐藏维度，省去 Y 的 HBM 重读；最终 RMSNorm 也从该区读取。Y 仍写回 HBM，保留原始残差路径和输出检查。',
              '2. LM head 的相邻词表分块复用现有 GM 0x10000 中已排布的最后一个 token。匹配 input 地址、维度、worker、连续 epoch 和连续输出区间后，省去重复输入 DMA 和打包；权重、GEMM、logits 输出保持原工作量。',
              '3. 跨 N 窗口延续原有四个 B 预取槽，避免符合条件的单 K 窗口在每四个 N tile 后排空。多 K 窗口回退原调度。没有增加预取槽或阵列操作数上下文；是否采用由上述完整正式测试决定。',
              '4. 实现并筛选 32+32 阵列分波 scatter/MAC、SFU 小块和 panel=8；这些 FFN 微测试更慢，正式配置关闭 scatter wave，保留原 panel/SFU 方案。', '',
              f"候选均额外减少 HBM 读取 **{d['saved_hbm_read_bytes']:,} 字节（{d['saved_hbm_read_bytes']/2**20:.4f} MiB）**；其中层间／最终 Norm 为 {d['layer_saved_hbm_read_bytes']:,} 字节，LM 为 {d['lm_saved_hbm_read_bytes']:,} 字节。HBM 写入量不变。GEMM 的 dense MAC、权重、A/C 窗口工作量完全相同。",
              'GM 0x3400 的 64 字节普通软件所有权记录包含 epoch、地址、形状和 worker，发布/读取均经过现有计时端口与 ACK。Y 复用 Wo/H1 的既有缓冲，Norm1 读完才允许 QKV C 暂存覆盖该区。没有新增张量缓存或硬件存储。',
              f"本轮 {d['hardware_sources_unchanged']} 个硬件/架构源文件哈希一致；控制与候选的 library、guest、内存模型、输入与布局一致，仅软件开关不同。原始 Muticore-Attention 仓库和库文件保持不变。", '',
              '## 尚存的边界', '',
              'Wo→Add→Norm2 的数据已在 GM 衔接，但原 guest 描述符和提交边界保留。本轮没有实现逐行跨描述符提前启动，也没有将所有阶段合成一个硬件作业。',
              'PV 的 8 worker 到 Wo 的 16 worker 仍有既有数据重分布。原输出 DMA/完成协议保持不变；不能把远端 GM 视作免费共享缓存，因此本轮没有宣称消除这一往返。',
              '最终 Norm 的输出仍写 HBM；LM 输入只在首次读取后于相邻词表块间复用。单上下文阵列的 program/scatter/MAC 依赖、完整行 RMSNorm 依赖、现有端口争用仍存在，不能称为完美流水。',
              '因此这轮完成了有收益候选的实现、筛选和正式验收；并非所有可设想的融合方向都已消除边界。详细决策见 IMPLEMENTATION.md。', '',
              '## 验证', '',
              '正式三组数值、实际 QKV 面板、所有 GEMM reuse-window 覆盖、16 worker、DMA 字节、有限资源、元数据所有权及完成前 ACK 排空检查通过。末层 Y、最终 RMSNorm、完整 logits 与对照及上一轮最佳的 FP16 位差异均为 0。',
              '另有延迟 ACK、拒绝提交、尾块、初始 C、多 K 窗口回退、错误 epoch、输入地址变化等负例和回归测试，日志保留在 checks/。',
              '正式比较使用同一冻结构建。micro_bridge 与最早 micro_control 使用不同构建，仅作筛选线索；是否采用 bridge 只依据同构建正式比较。一次 formal_all 的宿主机 CPU 放置失败在模拟启动前退出，保留为 formal_all_host_placement_failed，不纳入结果。', '',
              '## 复现', '', '```bash',
              'OPENBLAS_NUM_THREADS=1 python3 baseline/attention_cluster_8qk_8pv/run_all_gemm_reuse_window_baseline.py \\',
              '  --artifact-root results/software_dataflow_next_reproduction --shape formal --run \\',
              '  --window-a-prefetch 1 --lm-n-parallel 1 --gm-stage-chain \\',
              '  --layer-residency --lm-input-residency'+(' --b-window-bridge 1' if selected=='formal_all' else '')+' --cpu-set 0-7',
              '```', '', '同构建对照移除 --layer-residency、--lm-input-residency、--b-window-bridge 1；新选项默认关闭。',
              '数值、字节及选择依据见 summary.json；源码归档和逐文件哈希见 source_after/、final_validation.json。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-root', type=Path, default=ROOT/'results/software_dataflow_next_20261021')
    a = p.parse_args(); root = a.experiment_root.resolve()
    d = audit(root); report(root, d)
    print(json.dumps(dict(status=d['status'], selected=d['selected'], formal_comparison=d['formal_comparison'],
                          saved_hbm_read_bytes=d['saved_hbm_read_bytes']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
