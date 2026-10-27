#!/usr/bin/env python3
"""Summarize verified prefill stacks without adding overlapping stage spans."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root',type=Path,required=True)
    p.add_argument('--accepted-matrix',action='store_true',help='only cases in the successful all_validation.json matrix')
    a=p.parse_args();root=a.artifact_root.resolve()
    accepted=None
    if a.accepted_matrix:
        matrix=json.loads((root/'all_validation.json').read_text())
        if matrix['status']!='PASS':raise ValueError('matrix is not accepted')
        accepted=set(matrix['cases'])
    results={}
    for path in sorted(root.glob('*/decoder_stack_result.json')):
        if accepted is not None and path.parent.name not in accepted:continue
        r=json.loads(path.read_text())
        if r['status']!='PASS':raise ValueError(f'failed case: {path.parent}')
        if accepted is not None and r.get('gm_banks',{}).get('status')!='PASS':raise ValueError('missing GM audit')
        results[path.parent.name]=r
    if not results:raise ValueError('no verified stack results')
    summary={'status':'PASS','measurement':'explicit DMA send/ACK boundaries; per-case total includes optional Embedding/final norm/LM head',
             'cases':results,'scope':'synthetic FP16 prefill; parameterized layers and logical/padded sequence; fixed 20-core hardware'}
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Prefill 验证与周期统计','',
           '所有结果必须通过数值、标志、scratch 生命周期和 DMA ACK 时序检查。补齐行的工作计入设备 cycle。','',
           '| Case | 逻辑 / 执行 S | 层数 | hidden / FFN | 层间交接合计 cycle | Decoder 总 cycle | 完整入口总 cycle |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,r in results.items():
        d=r['dma_timing'];s=r['shape']
        total=r.get('embedding',r.get('lm_head',r.get('final_norm',d)))['end_to_end_cycles']
        lines.append(f'| {name} | {s["S"]} / {s.get("padded_S",s["S"])} | {s["layers"]} | {s["hidden"]} / {s["intermediate"]} | {d["layer_handoff_cycles"]:,.2f} | {d["end_to_end_cycles"]:,.2f} | {total:,.2f} |')
    lines+=['','## 连续且互不重叠的阶段边界','',
            '| Case / layer | RMSNorm + QKV + Attention cycle | Wo + 两次残差 + RMSNorm + FFN cycle | Y ACK 字节 |',
            '| --- | ---: | ---: | ---: |']
    for name,r in results.items():
        for layer,d in r['dma_timing']['layers'].items():
            lines.append(f'| {name} / {layer} | {d["attention_cycles"]:,.2f} | {d["post_attention_cycles"]:,.2f} | {d["output_ack_bytes"]:,} |')
    lines+=['','## 逐算子跨度（跨 worker 重叠，不可相加）','',
            '| Case / layer | Wo | residual1 | RMSNorm2 | gate | up | SiLU | multiply | down | residual2 |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,r in results.items():
        for layer,stages in r['lifecycle']['stage_spans'].items():
            values=' | '.join(f'{stages[str(i)]["span_cycles"]:,.0f}' if str(i) in stages else "融合省略" for i in range(1,10))
            lines.append(f'| {name} / {layer} | {values} |')
    lines+=['','## 实际生命周期计数','',
            '| Case | Projection jobs | Decoder ops | Reset ACK | Scratch retire |',
            '| --- | ---: | ---: | ---: | ---: |']
    for name,r in results.items():
        l=r['lifecycle']
        lines.append(f'| {name} | {l["projection_jobs"]} | {l["decoder_ops"]} | {l["reset_ack_count"]} | {l["scratch_retire_count"]} |')
    lines+=['','各层使用独立合成权重与 gamma，所有 dense 硬件 tiles 均执行。',
            '双缓冲模式只保留最后两层输出；retain-layer-outputs 模式保留并验证每层输出。',
            '交接含 guest 轮询、清零与 release；不等同于纯硬件 barrier。',
            '1 GHz 下 cycle / 1e6 为 ms。不同形状、padding 和验证布局的 cycle 不能直接当作软件优化收益。','']
    (root/'summary.md').write_text('\n'.join(lines))
    print(root/'summary.md')

if __name__=='__main__':main()
