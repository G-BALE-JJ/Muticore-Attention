#!/usr/bin/env python3
"""Summarize verified Decoder stacks with an independent final RMSNorm."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root',type=Path,required=True)
    root=parser.parse_args().artifact_root.resolve()
    cases={}
    for path in sorted(root.glob('*/decoder_stack_result.json')):
        result=json.loads(path.read_text())
        if result['status']!='PASS' or result.get('final_norm',{}).get('status')!='PASS':
            raise ValueError(f'unverified final RMSNorm: {path}')
        cases[path.parent.name]=result
    if not cases:raise ValueError('no verified final RMSNorm results')
    summary={'status':'PASS','contract':'first layer X input DMA send to final RMSNorm output DMA ACK',
             'scope':'two-layer synthetic FP16 prefill plus final RMSNorm; FP32 internal norm; no LM head',
             'cases':cases}
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# 两层 Decoder + final RMSNorm','',
           '各阶段按连续 DMA 边界计时。交接包括 guest 控制、标志清零 ACK 和 worker release。1 GHz 下每 cycle 为 1 ns。','',
           '| Case | hidden / FFN | 两层 cycle | final 交接 cycle | final RMSNorm cycle | 含 final 总 cycle | 延迟 ms |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for name,r in cases.items():
        f=r['final_norm'];shape=r['shape']
        lines.append(f'| {name} | {shape["hidden"]} / {shape["intermediate"]} | {r["dma_timing"]["end_to_end_cycles"]:,.2f} | '
                     f'{f["handoff_cycles"]:,.2f} | {f["rmsnorm_cycles"]:,.2f} | {f["end_to_end_cycles"]:,.2f} | {f["end_to_end_cycles"]/1e6:.6f} |')
    lines+=['','| Case | final 独立组合参考最大误差 | final 局部参考最大误差 | 输出 ACK bytes | 每节点占用 MiB |',
            '| --- | ---: | ---: | ---: | ---: |']
    for name,r in cases.items():
        lines.append(f'| {name} | {r["numerics"]["final_norm_composed"]["max_abs_error"]:.9f} | '
                     f'{r["numerics"]["final_norm_local"]["max_abs_error"]:.9f} | {r["final_norm"]["output_ack_bytes"]:,} | '
                     f'{r["memory"]["per_node_used_bytes"]/2**20:.6f} |')
    lines+=['','独立组合参考从 X0 计算两层及 final norm；局部参考只取第二层实际 Y 和独立 final gamma，单独检查归一化。',
            '每个 case 检查原两层 32 个 projection job、288 个 decoder op、64 个 reset ACK、64 次 scratch retire；',
            '再检查 16 个 final RMSNorm job、16 个独立 flag reset ACK，以及每个 worker 的 Y/gamma 输入、完整输出、完成标志 DMA ACK 顺序。',
            '原两层标志为 epoch=2，final 标志为 epoch=3。final 输出初始为 NaN，所有元素必须完成且有限。',
            '端到端边界包含两层间及 final 交接，排除 host preload、最初启动和 final 输出 ACK 后的通知/退出。',
            '当前仍为合成权重和两层 prefill；尚未接 LM head、token 选择及真实完整模型权重。','']
    (root/'summary.md').write_text('\n'.join(lines))
    print(root/'summary.md')


if __name__=='__main__':main()
