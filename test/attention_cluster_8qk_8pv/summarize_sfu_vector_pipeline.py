#!/usr/bin/env python3
"""Audit actual SFU reservations and compare verified, identical two-layer inputs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re


def fields(line):
    return {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', line)}


def read_log(case):
    markers = ('[SFU_VECTOR_OP] ', '[SFU_VECTOR_PIPELINE] ',
               '[DECODER_OP] event=done ', 'GOLEM_SFU_HW_PIPELINE ')
    with next((case / 'logs').glob('attention-sst*.log')).open() as stream:
        return ''.join(line for line in stream if any(marker in line for marker in markers))


def hardware(text):
    result = {}
    for line in text.splitlines():
        if 'GOLEM_SFU_HW_PIPELINE ' in line:
            f = fields(line)
            unit = re.search(r'unit=(\w+)', line)[1]
            result[f['core'], unit] = f
    return result


def audit(case, baseline):
    result = json.loads((case / 'decoder_stack_result.json').read_text())
    old = json.loads((baseline / 'decoder_stack_result.json').read_text())
    if result['status'] != 'PASS' or old['status'] != 'PASS':
        raise ValueError('both runs must pass the full stack verifier')
    new_inputs = json.loads((case / 'decoder_stack_inputs.json').read_text())
    old_inputs = json.loads((baseline / 'decoder_stack_inputs.json').read_text())
    signature = lambda m: sorted((e['name'], e['node'], e['offset'], e['bytes'], e['sha256'])
                                 for e in m['preloads'])
    if signature(new_inputs) != signature(old_inputs) or new_inputs['layout'] != old_inputs['layout']:
        raise ValueError('before/after inputs or memory layout differ')
    text = read_log(case)
    new_hw, old_hw = hardware(text), hardware(read_log(baseline))
    jobs = [fields(line) for line in text.splitlines() if '[SFU_VECTOR_OP] ' in line]
    summaries = [fields(line) for line in text.splitlines() if '[SFU_VECTOR_PIPELINE] ' in line]
    if {e['core'] for e in summaries} != set(range(4, 20)) or len(summaries) != 16:
        raise ValueError('missing or duplicate worker scheduler statistics')
    hidden = result['shape']['hidden']
    expected_jobs = {3: hidden // 2, 4: hidden, 5: hidden}
    if Counter(j['op'] for j in jobs) != expected_jobs:
        raise ValueError('missing vector operation completions')
    if len({(j['core'], j['job']) for j in jobs}) != len(jobs):
        raise ValueError('duplicate vector request IDs')
    ops = [fields(line) for line in text.splitlines() if '[DECODER_OP] event=done ' in line]
    per_stage = {}
    for op in ops:
        if op['stage'] not in (2, 6, 7, 9):
            continue
        opcode = {2: 3, 6: 5, 7: 4, 9: 3}[op['stage']]
        width = hidden if op['stage'] in (2, 9) else 4 * hidden
        owned = [j for j in jobs if j['core'] == op['core'] and j['op'] == opcode and
                 op['start_tick'] <= j['submit_tick'] < j['done_tick'] <= op['tick']]
        if len(owned) != 64 * width // 8192 or sum(j['elements'] for j in owned) != 64 * width:
            raise ValueError('vector result publication not covered by decoder stage lifetime')
        per_stage[op['core'], op['epoch'], op['stage']] = owned
    if sum(len(group) for group in per_stage.values()) != len(jobs):
        raise ValueError('unowned or multiply-owned vector completion')
    for summary in summaries:
        if summary['fifo_high_water'] > summary['fifo_elements'] or summary['request_high_water'] != 1:
            raise ValueError('FIFO capacity or serial decoder request ownership violated')
        core = summary['core']
        owned = [j for j in jobs if j['core'] == core]
        activations = [j for j in owned if j['op'] == 5]
        vector = new_hw[core, 'vector_op']
        for job in owned:
            minimum = vector['latency'] + ((job['elements']-1)//vector['lanes'])*vector['ii']
            if job['op'] == 5:
                exp, reciprocal = new_hw[core, 'exp'], new_hw[core, 'reciprocal']
                minimum = (3*vector['latency'] + exp['latency'] + reciprocal['latency'] +
                           (job['elements']-1)*reciprocal['ii'])
            if job['cycles'] < minimum:
                raise ValueError('completion faster than physical pipeline dependency/throughput limit')
        vector_tokens = sum(((j['elements'] + vector['lanes'] - 1) // vector['lanes']) *
                            (3 if j['op'] == 5 else 1) for j in owned)
        if vector['accepted_tokens'] != vector_tokens:
            raise ValueError('vector instructions not actually reserved')
        for unit in ('exp', 'reciprocal'):
            new, previous = new_hw[core, unit], old_hw[core, unit]
            if any(new[key] != previous[key] for key in ('lanes', 'latency', 'ii', 'depth')):
                raise ValueError('shared hardware configuration changed')
            extra = sum((j['elements'] + new['lanes'] - 1) // new['lanes'] for j in activations)
            if new['accepted_tokens'] - previous['accepted_tokens'] != extra:
                raise ValueError('SiLU did not reserve the actual shared hardware pipeline')
        for unit in ('vector_op', 'exp', 'reciprocal'):
            if new_hw[core, unit]['max_occupancy'] > new_hw[core, unit]['depth']:
                raise ValueError('hardware queue depth exceeded')
    cycles = {str(op): dict(sorted(Counter(j['cycles'] for j in jobs if j['op'] == op).items()))
              for op in (3, 4, 5)}
    v, e, r = (new_hw[4, unit] for unit in ('vector_op', 'exp', 'reciprocal'))
    chunks = (8192 + v['lanes'] - 1) // v['lanes']
    old_silu_formula = (2*v['latency'] + 3*chunks*v['ii'] + e['latency'] +
                        ((8192+e['lanes']-1)//e['lanes'])*e['ii'] + r['latency'] + 8192*r['ii'])
    before, after = old['dma_timing']['end_to_end_cycles'], result['dma_timing']['end_to_end_cycles']
    return {'status': 'PASS', 'identical_inputs_and_layout': True,
            'shape': result['shape'], 'vector_completions': len(jobs),
            'cycles_per_request': cycles, 'old_silu_formula_cycles_8192': old_silu_formula,
            'shared_hardware': {unit: new_hw[4, unit] for unit in ('vector_op', 'exp', 'reciprocal')},
            'scheduler': {'max_fifo_elements': max(s['fifo_high_water'] for s in summaries),
                          'fifo_backpressure_events': sum(s['fifo_backpressure_events'] for s in summaries),
                          'resource_backpressure_events': sum(s['resource_backpressure_events'] for s in summaries)},
            'before_end_to_end_cycles': before, 'after_end_to_end_cycles': after,
            'end_to_end_reduction_percent': 100*(before-after)/before,
            'silu_stage_spans': {layer: {'before': old['lifecycle']['stage_spans'][layer]['6']['span_cycles'],
                                        'after': result['lifecycle']['stage_spans'][layer]['6']['span_cycles']}
                                 for layer in ('1', '2')}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root', type=Path, required=True)
    p.add_argument('--baseline-root', type=Path, required=True)
    a = p.parse_args()
    root = a.artifact_root.resolve()
    cases = {}
    for name, previous in (('two_layer_small', 'two_layer_small_v4'), ('two_layer_formal', 'two_layer_formal')):
        if (root / name / 'decoder_stack_result.json').exists():
            cases[name] = audit(root / name, a.baseline_root / previous)
    if not cases:
        raise ValueError('no completed cases')
    (root / 'sfu_pipeline_audit.json').write_text(json.dumps({'status': 'PASS', 'cases': cases}, indent=2)+'\n')
    lines = ['# SFU 向量流水线前后对比', '',
             '相同输入、权重、内存布局和共享 EXP/倒数配置；所有 case 通过两层完整数值、同步和 DMA ACK 检查。', '',
             '| Case | 原两层 cycle | 新两层 cycle | 总周期下降 | SiLU 原跨度 L1/L2 | 新跨度 L1/L2 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, c in cases.items():
        spans = c['silu_stage_spans']
        lines.append(f'| {name} | {c["before_end_to_end_cycles"]:,.2f} | {c["after_end_to_end_cycles"]:,.2f} | '
                     f'{c["end_to_end_reduction_percent"]:.3f}% | '
                     f'{spans["1"]["before"]:,.0f} / {spans["2"]["before"]:,.0f} | '
                     f'{spans["1"]["after"]:,.0f} / {spans["2"]["after"]:,.0f} |')
    lines += ['', '实际端到端脚本配置：vector=16 lanes、EXP=16 lanes、reciprocal=1 lane；组件 EXP 默认值为 4，不能混用两种配置。',
              '8192 元素：原 SiLU 串行计费公式 10,262 cycles；新 SST 调度 8,217 cycles。加法/乘法仍为 515 cycles。',
              '单独 C++ 调度器从 cycle 0 发射，新 SiLU 为 8,216 cycles；SST 首次 pump 多 1 cycle。',
              'SiLU 的单算子计算下降约 19.93%；端到端仍包含相同的 GM/DMA/HBM 和控制开销。算子跨度跨 worker 重叠，不能相加。', '',
              '每个 worker 的新增 EXP、reciprocal accepted_tokens 与 SiLU 工作量精确相等；vector_op 的 accepted_tokens 与全部 Add/Multiply/SiLU 指令数相等。有限队列及 FIFO 容量均未超限。',
              '当前 decoder 在 Attention 和 FFN 之间有 barrier；共享竞争由生产调度器的独立测试覆盖，端到端回归不声称观察到这两个阶段的实际交叠。', '']
    (root / 'sfu_pipeline_comparison.md').write_text('\n'.join(lines))
    print(root / 'sfu_pipeline_comparison.md')


if __name__ == '__main__':
    main()
