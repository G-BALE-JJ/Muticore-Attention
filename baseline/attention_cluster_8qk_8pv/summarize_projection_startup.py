#!/usr/bin/env python3
"""Validate the four frozen formal startup cases and compare paired metrics."""
import argparse
import json
from pathlib import Path

from analyze_projection_startup import compare
from run_projection_overlap import digest, ROOT


def combine(shard_roots):
    indexed, hashes = {}, None
    for root in map(Path, shard_roots):
        summary = json.loads((root / 'summary.json').read_text())
        if summary['status'] != 'PASS':
            raise ValueError('incomplete startup shard')
        current = json.loads((root / 'source_sha256.json').read_text())
        if hashes is not None and hashes != current:
            raise ValueError('startup source or binary versions differ')
        hashes = current
        for row in summary['cases']:
            key = (row['shape']['query_length'], row['variant'])
            if key in indexed:
                raise ValueError('duplicate startup case')
            if row['status'] != 'PASS' or row['byte_exact']['status'] != 'PASS':
                raise ValueError('startup output verification missing')
            provenance = json.loads((Path(row['artifact_root']) / 'provenance.json').read_text())
            env = provenance['environment']
            expected = {'GOLEM_PROJECTION_ARRAYS': '64', 'GOLEM_ARRAY_OPERAND_CONTEXT_BANKS': '1',
                'GOLEM_LOCAL_GM_READ_PORTS': '1', 'GOLEM_PROJECTION_ARRAY_GROUPS': '1',
                'GOLEM_PROJECTION_INPUT_PAIR_PREFETCH': '1', 'GOLEM_PROJECTION_LOCAL_WEIGHT_PREFETCH': '1',
                'GOLEM_DIAGNOSTIC_ORACLE': 'none', 'VANADIS_CPU_CLOCK': '1.0GHz'}
            if any(env.get(k) != v for k, v in expected.items()):
                raise ValueError('startup hardware, oracle, or dataflow changed')
            guest = Path(env['GOLEM_ATTENTION_GUEST_EXECUTABLE'])
            if digest(guest) != provenance['guest_sha256'] or digest(guest) != current[str(guest.relative_to(ROOT))]:
                raise ValueError('selected guest does not match frozen binary')
            indexed[key] = row
    if set(indexed) != {(s, v) for s in (1024, 2048) for v in ('baseline', 'batch_release')}:
        raise ValueError('startup formal comparison requires all four cases')
    comparisons = {}
    for sequence in (1024, 2048):
        baseline, candidate = (indexed[(sequence, v)] for v in ('baseline', 'batch_release'))
        if baseline['weight_programs'] != candidate['weight_programs']:
            raise ValueError('Projection weight programming work changed')
        for core in baseline['dataflow_by_worker']:
            for counter in ('array_launches', 'compute_busy_cycles'):
                if baseline['dataflow_by_worker'][core][counter] != candidate['dataflow_by_worker'][core][counter]:
                    raise ValueError('Projection computation work changed')
        comparisons[str(sequence)] = compare(baseline, candidate)
    if any(digest(ROOT / p) != value for p, value in hashes.items()):
        raise ValueError('current sources or binaries changed after startup simulation')
    return {'status': 'PASS', 'completed_cases': 4,
            'cases': [indexed[(s, v)] for s in (1024, 2048) for v in ('baseline', 'batch_release')],
            'comparisons': comparisons, 'source_sha256': hashes,
            'scope': 'Real guest release order change, with all guest logs retained. Absolute simulation-zero output and process-exit times include preparation and teardown.'}


def markdown(result):
    lines = ['# Worker 启动错峰优化：正式实验', '',
        '基线为输入＋权重预取（IW）；实验仅将 manager 的逐 worker 日志延后到四次真实 release 之后。', '',
        '| 指标（cycles） | S1024 对照 | S1024 优化 | S2048 对照 | S2048 优化 |',
        '|---|---:|---:|---:|---:|']
    for key, name in (
        ('reported_e2e_cycles', '原定义 E2E'),
        ('worker_accept_span_cycles', '全部 worker 接受时间跨度'),
        ('release_to_attention_complete_cycles', '首个 release → Attention 完成'),
        ('last_output_ack_cycle_from_simulation_zero', '仿真起点 → 最终 O ACK'),
        ('root_complete_cycle_from_simulation_zero', '仿真起点 → root 完成'),
        ('simulation_end_cycles_from_zero', '仿真起点 → 全部进程退出')):
        a, b = (result['comparisons'][str(s)][key] for s in (1024, 2048))
        lines.append(f'| {name} | {a["baseline"]:,} | {a["candidate"]:,} | {b["baseline"]:,} | {b["candidate"]:,} |')
    lines += ['', '真实输出提前判断：' + ', '.join(
        f'S{s}={result["comparisons"][str(s)]["real_completion_improves"]}' for s in (1024, 2048)),
        '', '全部输出需通过基线逐字节对照和 NumPy 校验，计算量不变，硬件资源不变。详细起点、服务、版本与配置见 JSON。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard-roots', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = combine(args.shard_roots)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    args.output.with_suffix('.md').write_text(markdown(result))
    print('PASS four formal cases: frozen versions, output, workload, startup and completion accounting')


if __name__ == '__main__':
    main()
