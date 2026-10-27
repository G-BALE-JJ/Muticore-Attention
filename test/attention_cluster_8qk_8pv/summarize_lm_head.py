#!/usr/bin/env python3
"""Summarize verified last-token logits runs with their DMA timing boundary."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root',type=Path,required=True)
    root=p.parse_args().artifact_root.resolve();cases={}
    for path in sorted(root.glob('*/decoder_stack_result.json')):
        r=json.loads(path.read_text())
        if r['status']!='PASS' or r.get('lm_head',{}).get('status')!='PASS':raise ValueError(f'unverified head: {path}')
        cases[path.parent.name]=r
    if not cases:raise ValueError('no verified LM head results')
    (root/'summary.json').write_text(json.dumps({'status':'PASS','scope':'two synthetic Decoder layers, final RMSNorm and last-token LM head',
        'contract':'first layer X input DMA send to last-token logits final DMA ACK','cases':cases},indent=2)+'\n')
    lines=['# 两层 prefill + final RMSNorm + 最后一个 token 的 LM head','',
           '全部 case 完成数值、epoch、输入/权重/输出字节覆盖及 DMA ACK 检查。1 GHz 下每 cycle 为 1 ns。','',
           '| Case | hidden / vocab | node 容量 MiB | node 使用 MiB | norm ACK 总 cycle | head 交接 cycle | head cycle | logits ACK 总 cycle | 延迟 ms |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,r in cases.items():
        h=r['lm_head'];memory=r['memory']
        lines.append(f'| {name} | {r["shape"]["hidden"]} / {h["vocab"]} | {memory["per_node_capacity_bytes"]/2**20:,.0f} | '
                     f'{memory["per_node_used_bytes"]/2**20:.6f} | {r["final_norm"]["end_to_end_cycles"]:,.2f} | '
                     f'{h["handoff_cycles"]:,.2f} | {h["lm_head_cycles"]:,.2f} | {h["end_to_end_cycles"]:,.2f} | {h["end_to_end_cycles"]/1e6:.6f} |')
    lines+=['','| Case | chunks / 尾块列数 | head 局部参考误差 | 完整组合参考误差 | logits ACK bytes |',
            '| --- | --- | ---: | ---: | ---: |']
    for name,r in cases.items():
        h=r['lm_head'];lines.append(f'| {name} | {h["chunks"]} / {h["tail_chunk_columns"]} | {h["local_reference"]["max_abs_error"]:.9f} | '
                                  f'{h["composed_reference"]["max_abs_error"]:.9f} | {h["output_ack_bytes"]:,} |')
    lines+=['','每个 worker 只读取 token 1023 的 hidden 向量，M=1，不读取其余 63 个填充 token。每块 N<=8192，尾块补齐到 64 列；host 仅返回有效 vocab logits。',
            '每个 worker 按 core 分配词表 tile，只激活一条 array 行；所有 weight tile 经现有 DMA、GM、WCP 和真实 array 执行。',
            'head 完成标志只在最后词表块输出 ACK 后发布；不同 worker 可以交叠不同块，逐块时间不能直接相加。',
            '计时包含层间、final norm 和 LM head 交接及数据搬运，排除 host preload、启动及最终 logits ACK 后的通知/退出。',
            '2 GiB/node 是容量；SST functional backing 与地址 stride 同步扩大，链路带宽和 Ramulator geometry 不随容量扩大。',
            '原两层使用合成权重，尚未加载真实 checkpoint；输出保存为 last_token_logits.npy，尚未接 token 采样。','']
    (root/'summary.md').write_text('\n'.join(lines));print(root/'summary.md')


if __name__=='__main__':main()
