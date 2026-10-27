#!/usr/bin/env python3
"""Audit and compare matched 4/8-rank embedding-to-token prefill runs."""
import argparse
import json
from pathlib import Path
import numpy as np
from run_projection_overlap import digest


def summarize(root):
    cases={};pairs={}
    for path in sorted(root.glob('*/decoder_stack_result.json')):
        case=path.parent;r=json.loads(path.read_text())
        if not r.get('embedding'):continue
        wall=json.loads((case/'simulation_wall_time.json').read_text())
        provenance=json.loads((case/'decoder_stack_provenance.json').read_text())
        manifest=json.loads((case/'decoder_stack_inputs.json').read_text())
        if any(r[key]['status']!='PASS' for key in ('embedding','lm_head','final_norm')):raise ValueError('incomplete verification')
        if r['lm_head']['next_token']['status']!='PASS':raise ValueError('unverified next token')
        for name,sha in provenance['source_sha256'].items():
            if digest(case/'source_snapshot'/name)!=sha:raise ValueError('snapshot hash mismatch')
        for entry in manifest['preloads']:
            if digest(Path(entry['path']))!=entry['sha256']:raise ValueError('input hash mismatch')
        placement=json.loads((case/'attention_mpi_placement.json').read_text())
        if placement['mpi_ranks']!=wall['mpi_ranks']:raise ValueError('placement/rank mismatch')
        ranks={placement['component_ranks'][f'core{core}'] for core in range(4,20)}
        if ranks!=set(range(wall['mpi_ranks'])):raise ValueError('unused worker rank')
        e=r['embedding'];layers=r['dma_timing']['layers'];norm=r['final_norm'];head=r['lm_head']
        parts={'embedding':e['embedding_cycles'],'embedding_handoff':e['handoff_cycles'],
            'layer1_norm_qkv_attention':layers['1']['attention_cycles'],'layer1_post_attention':layers['1']['post_attention_cycles'],
            'layer_handoff':(layers['2']['first_input_dma_tick']-layers['1']['last_y_ack_tick'])/1000,
            'layer2_norm_qkv_attention':layers['2']['attention_cycles'],'layer2_post_attention':layers['2']['post_attention_cycles'],
            'final_norm_handoff':norm['handoff_cycles'],'final_norm':norm['rmsnorm_cycles'],
            'lm_head_handoff':head['handoff_cycles'],'lm_head':head['lm_head_cycles']}
        if sum(parts.values())!=e['end_to_end_cycles'] or min(parts.values())<0:raise ValueError('cycle boundary accounting mismatch')
        cases[case.name]={'result':r,'wall':wall,'provenance':provenance,'cycle_breakdown':parts}
        shape='formal' if r['shape']['hidden']==2048 else 'small'
        pairs.setdefault(shape,{})[wall['mpi_ranks']]=(case,r,wall,provenance,manifest)
    comparison={}
    for shape,group in pairs.items():
        if set(group)!={4,8}:raise ValueError(f'{shape}: requires matched 4/8 ranks')
        four,eight=group[4],group[8]
        input_key=lambda m:sorted((e['name'],e['node'],e['offset'],e['bytes'],e['sha256']) for e in m['preloads'])
        if input_key(four[4])!=input_key(eight[4]) or four[4]['layout']!=eight[4]['layout']:raise ValueError('unmatched inputs/layout')
        for key in ('guest_sha256','sst_library_sha256'):
            if four[3][key]!=eight[3][key]:raise ValueError('unmatched executable')
        a=np.load(four[0]/'last_token_logits.npy');b=np.load(eight[0]/'last_token_logits.npy')
        token4=four[1]['lm_head']['next_token']['token_id'];token8=eight[1]['lm_head']['next_token']['token_id']
        if token4!=token8:raise ValueError('rank count changed selected token')
        comparison[shape]={'status':'PASS','identical_input_preloads':True,'identical_guest_and_library':True,
            'source_differences':[k for k,v in four[3]['source_sha256'].items() if v!=eight[3]['source_sha256'].get(k)],
            'rank4_wall_seconds':four[2]['sst_wall_seconds'],'rank8_wall_seconds':eight[2]['sst_wall_seconds'],
            'wall_speedup':four[2]['sst_wall_seconds']/eight[2]['sst_wall_seconds'],
            'wall_reduction_percent':100*(1-eight[2]['sst_wall_seconds']/four[2]['sst_wall_seconds']),
            'rank4_cycles':four[1]['embedding']['end_to_end_cycles'],'rank8_cycles':eight[1]['embedding']['end_to_end_cycles'],
            'logits_max_abs_difference':float(np.max(np.abs(a.astype('f4')-b.astype('f4')))),
            'logits_exactly_equal':bool(np.array_equal(a,b)),'next_token_id':token4}
    report={'status':'PASS','scope':'synthetic two-layer prefill from token IDs through timed embedding and last-token LM head; host greedy selection',
            'comparisons':comparison,'cases':cases}
    (root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Embedding → 两层 prefill → logits → greedy token','',
        'SST wall 单独计时，排除 host 初始化和验证；4/8 rank 顺序运行。每组输入、内存布局、guest 和 SST library hashes 相同。','',
        '| 规模 | rank 4 wall s | rank 8 wall s | 加速比 | wall 减少 | rank 4 cycle | rank 8 cycle | token ID |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for shape,c in comparison.items():
        lines.append(f'| {shape} | {c["rank4_wall_seconds"]:.6f} | {c["rank8_wall_seconds"]:.6f} | {c["wall_speedup"]:.3f}× | {c["wall_reduction_percent"]:.2f}% | {c["rank4_cycles"]:,.3f} | {c["rank8_cycles"]:,.3f} | {c["next_token_id"]} |')
    lines+=['','| Case | embedding cycle | 首层交接 cycle | 全链路 cycle | 1 GHz ms | node 使用 MiB | argmax host µs |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,c in cases.items():
        r=c['result'];e=r['embedding']
        lines.append(f'| {name} | {e["embedding_cycles"]:,.3f} | {e["handoff_cycles"]:,.3f} | {e["end_to_end_cycles"]:,.3f} | {e["end_to_end_cycles"]/1e6:.6f} | {r["memory"]["per_node_used_bytes"]/2**20:.6f} | {r["lm_head"]["next_token"]["host_selection_seconds"]*1e6:.3f} |')
    if 'formal_rank4' in cases and 'formal_rank8' in cases:
        lines+=['','| 阶段（互不重叠，含数据搬运） | rank 4 cycle | rank 8 cycle | rank 4 占比 |','| --- | ---: | ---: | ---: |']
        labels={'embedding':'Embedding','embedding_handoff':'Embedding → 首层交接','layer1_norm_qkv_attention':'第一层 Norm1/QKV/Attention',
            'layer1_post_attention':'第一层 Wo/残差/Norm2/FFN/残差','layer_handoff':'层间交接','layer2_norm_qkv_attention':'第二层 Norm1/QKV/Attention',
            'layer2_post_attention':'第二层 Wo/残差/Norm2/FFN/残差','final_norm_handoff':'final norm 交接','final_norm':'final RMSNorm',
            'lm_head_handoff':'LM head 交接','lm_head':'LM head'}
        for key,value in cases['formal_rank4']['cycle_breakdown'].items():
            lines.append(f'| {labels[key]} | {value:,.3f} | {cases["formal_rank8"]["cycle_breakdown"][key]:,.3f} | {100*value/cases["formal_rank4"]["result"]["embedding"]["end_to_end_cycles"]:.3f}% |')
    lines+=['','Embedding 数值要求逐元素精确匹配；检查 token/表/输出 DMA 地址与字节覆盖、独立 flag reset/completion ACK、第一层在全局完成后读取。',
        '模拟 cycle 的起点为首个 token-ID DMA send，终点为最后 logits output DMA ACK。host argmax 不计入加速器 cycle。',
        'MPI rank 数只改变仿真任务分区，硬件仍为 20 cores / 16 workers / 4 data HBM nodes，各 node 2 GiB，GM 2 MiB，带宽未修改。',
        '8 rank 将同一列的 QK 与 PV worker 分到成对 rank；每 rank 两个 worker。此对比每种配置各运行一次，wall time 可能受宿主机波动影响。',
        '使用合成权重和 1024 token ID，未接 tokenizer、真实 checkpoint、完整模型层数或自回归 decode。','']
    (root/'summary.md').write_text('\n'.join(lines))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact-root',type=Path,required=True)
    print(json.dumps(summarize(p.parse_args().artifact_root.resolve())['comparisons'],indent=2))
