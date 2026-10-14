#!/usr/bin/env python3
"""Summarize verified two-layer runs without adding overlapping stage spans."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root',type=Path,required=True)
    a=p.parse_args();root=a.artifact_root.resolve()
    results={}
    for path in sorted(root.glob('*/decoder_stack_result.json')):
        r=json.loads(path.read_text())
        if r['status']!='PASS':raise ValueError(f'failed case: {path.parent}')
        results[path.parent.name]=r
    if not results:raise ValueError('no verified stack results')
    summary={'status':'PASS','measurement':'first layer input DMA send to second layer output final DMA ACK',
             'cases':results,'scope':'synthetic FP16 prefill, two layers; fixed 20-core hardware'}
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# 两层 Decoder 验证与周期统计','',
           '同一次 SST 仿真内连续执行两层；第二层直接读取第一层输出。所有结果必须通过数值、标志、缓存交接和 DMA ACK 时序检查。','',
           '| Case | hidden / FFN | 第一层 cycle | 层间交接 cycle | 第二层 cycle | 两层总 cycle | 1 GHz 延迟 |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for name,r in results.items():
        d=r['dma_timing'];s=r['shape']
        lines.append(f'| {name} | {s["hidden"]} / {s["intermediate"]} | {d["layers"]["1"]["layer_cycles"]:,.2f} | {d["layer_handoff_cycles"]:,.2f} | {d["layers"]["2"]["layer_cycles"]:,.2f} | {d["end_to_end_cycles"]:,.2f} | {d["end_to_end_cycles"]/1e6:.6f} ms |')
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
            values=' | '.join(f'{stages[str(i)]["span_cycles"]:,.0f}' for i in range(1,10))
            lines.append(f'| {name} / {layer} | {values} |')
    lines+=['','每个 case：32 个 projection job、288 个 decoder op、64 个 flag reset ACK、64 个 scratch retire，两个 root decoder 完成及 8 个 manager Attention 完成。两层 Attention job/tag 独立，最终 32 个 HBM 完成标志均为 2。','',
            '两层使用不同合成权重、不同两组 gamma；CPU 参考从 X0 独立组合两层，第二层逐算子再与设备输入对齐。FP16 合成权重仍执行完整 dense tile。',
            '交接含 guest 完成轮询/控制流、清零和 release，不能等同于纯硬件 barrier。排除最初 input DMA 以前的启动和最终 Y ACK 之后的打印/退出。','']
    (root/'summary.md').write_text('\n'.join(lines))
    print(root/'summary.md')

if __name__=='__main__':main()
