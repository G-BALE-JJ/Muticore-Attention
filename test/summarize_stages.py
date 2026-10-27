#!/usr/bin/env python3
"""Collect four completed SST runs and audit their latency boundaries."""
import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def load(path, passed=False):
    data = json.loads(path.read_text())
    if passed and data.get('status') != 'PASS':
        raise ValueError(f'Unvalidated result: {path}')
    return data


def metrics(path):
    return {row['metric']: row['value'] for row in csv.DictReader(path.open())}


def unique(paths):
    paths = list(paths)
    if len(paths) != 1:
        raise ValueError(f'Expected one artifact, found {paths}')
    return paths[0]


def load_selection(configuration=None):
    public_path = ROOT / 'test/attention_cluster_8qk_8pv/SELECTED_PREFILL_2026-10-26.json'
    public = load(public_path, passed=True)
    if configuration is None:
        return public
    candidate = load(configuration, passed=True)
    # A rebuilt candidate may replace only the installed-library pin. Guest,
    # workload, schedule, numerical references and latency contract stay fixed.
    for key, value in public.items():
        if key != 'measured_library_sha256' and candidate.get(key) != value:
            raise ValueError(f'Candidate changes selected prefill contract: {key}')
    if not candidate.get('candidate_validation', {}).get('public_selection_unchanged'):
        raise ValueError('Candidate configuration lacks explicit validation provenance')
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--gemm-run-name', help='optional GEMM run with explicit A@B verification')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--configuration', type=Path,
                        help='explicit rebuilt candidate pin; preserves the public workload and guest')
    args = parser.parse_args()
    library = digest(ROOT / 'install/lib/sst-elements-library/libgolem.so')
    rows, invocations, cases, evidence = [], {}, {}, {}
    for stage in ('GEMM', 'Attention', 'GQA', 'prefill'):
        name = args.gemm_run_name if stage == 'GEMM' and args.gemm_run_name else args.run_name
        if Path(name).name != name:
            parser.error('run names must be directory names')
        case = ROOT / 'test' / stage / 'results' / name
        meta = case / 'stage_invocation.json'
        invocation = load(meta, passed=True)
        if invocation['exit_code'] != 0 or invocation['mode'] != 'run':
            raise ValueError(f'Expected a successful formal run: {meta}')
        if invocation['library_sha256'] != library:
            raise ValueError(f'Installed library differs from the measured run: {meta}')
        invocations[stage], cases[stage] = invocation, case
        evidence[stage] = {str(meta.relative_to(ROOT)): digest(meta)}
        stats = unique(case.glob('**/stats/overlap0/*/run_config.env')).parent
        config = stats.joinpath('run_config.env').read_text()
        if not re.search(r'^\s*VANADIS_CPU_CLOCK=1\.0GHz\s*$', config, re.M):
            raise ValueError(f'Expected the recorded 1 GHz reference clock: {stats}')
        if stage == 'GEMM':
            import numpy as np
            parameters = dict(re.findall(r'^\s*(\w+)=(.*)$', config, re.M))
            m, n, k = (int(parameters['GOLEM_GEMM_' + dim]) for dim in 'MNK')
            if parameters['GOLEM_MATMUL_DTYPE'] != 'fp32' or parameters['GOLEM_BIAS_ENABLE'] != '0':
                raise ValueError('This full-element GEMM audit requires unbiased fp32')
            a_path, b_path = (Path(parameters['GOLEM_TENSOR_' + dim + '_FILE']) for dim in 'AB')
            a = np.fromfile(a_path, dtype='<f4').reshape(m, k)
            b = np.fromfile(b_path, dtype='<f4').reshape(k, n)
            if parameters.get('GOLEM_MATMUL_TRANSPOSE_B', '0') != '0':
                raise ValueError('This GEMM audit requires non-transposed B')
            c_path = stats / 'c_out.csv'
            c = np.loadtxt(c_path, delimiter=',', dtype=np.float32)
            expected = a @ b
            if c.shape != expected.shape or not np.isfinite(c).all():
                raise ValueError('Invalid GEMM output shape or nonfinite values')
            mismatch = int(np.count_nonzero(~np.isclose(c, expected, atol=1e-5, rtol=1e-4)))
            if mismatch:
                raise ValueError(f'GEMM full-output mismatches: {mismatch}')
            verification = dict(status='PASS', checked_elements=int(c.size), mismatches=mismatch,
                                max_abs_error=float(np.max(np.abs(c - expected))), atol=1e-5, rtol=1e-4,
                                inputs_sha256={str(p): digest(p) for p in (a_path, b_path, c_path)})
            path = case / 'gemm_numerical_verification.json'
            path.write_text(json.dumps(verification, indent=2) + '\n')
            timing = metrics(stats / 'execution_summary.csv')
            terminal = metrics(stats / 'terminal_summary.csv')
            if terminal['c_writeback_complete_ratio'] != '4/4':
                raise ValueError('GEMM final C writes incomplete for the current 128-cubed baseline')
            start, end, duration = (float(timing[key]) for key in
                                   ('gemm_system_start_cycle', 'gemm_system_end_cycle', 'gemm_system_latency_cycles'))
            if duration != end - start + 1:
                raise ValueError('GEMM inclusive cycle boundary mismatch')
            scope = 'first descriptor start -> last WCP completion (includes C writeback), inclusive +1'
            shape = dict(M=m, N=n, K=k, dtype='fp32')
            result_paths = [path, stats / 'execution_summary.csv', stats / 'terminal_summary.csv']
            validation = verification
        elif stage in ('Attention', 'GQA'):
            path = case / 'sst_qk_bridge_result.json'
            timing = load(path, passed=True)
            result_paths = [path, *(case / f for f in ('fused_attention_result.json',
                            'attention_mpi_partition.json', 'attention_hbm_layout.json'))]
            validation = {p.stem: load(p, passed=True)['status'] for p in result_paths}
            start, end, duration = (timing[k] for k in ('start_cycle', 'end_cycle', 'end_to_end_cycles'))
            if duration != end - start:
                raise ValueError(f'{stage} cycle boundary mismatch')
            scope, shape = timing['scope'], timing['shape']
        else:
            campaign = load(case / 'all_gemm_reuse_window_report.json', passed=True)
            path = case / 'reuse_window_probe/decoder_stack_result.json'
            prefill = load(path, passed=True)
            result_paths = [path, case / 'all_gemm_reuse_window_report.json']
            validation = {k: prefill[k]['status'] for k in
                          ('lifecycle', 'resources', 'gm_banks', 'final_norm', 'lm_head', 'embedding')}
            validation.update(coverage=campaign['coverage']['status'],
                              projection_panels=campaign['projection_panels']['status'])
            if any(value != 'PASS' for value in validation.values()):
                raise ValueError('prefill resource or numerical validation incomplete')
            factor = prefill['dma_timing']['ticks_per_cycle']
            if factor != 1000:
                raise ValueError('Expected 1 ps ticks and 1 GHz reference cycles')
            start = prefill['embedding']['first_token_ids_dma_tick'] / factor
            end = prefill['lm_head']['last_logits_ack_tick'] / factor
            duration = prefill['embedding']['end_to_end_cycles']
            if duration != end - start:
                raise ValueError('prefill complete DMA boundary mismatch')
            scope, shape = prefill['embedding']['contract'], prefill['shape']
        if duration <= 0:
            raise ValueError(f'Invalid latency: {stage}')
        for p in result_paths:
            evidence[stage][str(p.relative_to(ROOT))] = digest(p)
        summaries = list(case.glob('**/stats/run_summary.csv'))
        host_seconds = None
        if len(summaries) == 1:
            records = list(csv.DictReader(summaries[0].open()))
            host_seconds = float(records[-1]['wall_time_sec'])
        rows.append(dict(stage=stage, status='PASS', shape=shape, start_cycle=start, end_cycle=end,
                         end_to_end_cycles=duration, boundary=scope, reference_clock_hz=10**9,
                         host_sst_wall_seconds=host_seconds, validation=validation,
                         artifacts=str(case.relative_to(ROOT))))

    # Disjoint intervals, including handoffs. Eager final norm overlaps decoder;
    # only its tail after final Y is added to avoid double counting.
    d = prefill['dma_timing']
    points = [('embedding', prefill['embedding']['last_embedding_output_ack_tick']),
              ('embedding -> decoder handoff', d['layers']['1']['first_input_dma_tick'])]
    for index, layer in sorted(d['layers'].items(), key=lambda item: int(item[0])):
        points += [(f'layer {index} norm/QKV/RoPE/Attention to O ACK', layer['last_attention_o_ack_tick']),
                   (f'layer {index} Wo/residual/norm/FFN to Y ready', layer['last_y_ack_tick'])]
        if str(int(index) + 1) in d['layers']:
            points.append((f'layer {index} -> {int(index) + 1} handoff',
                           d['layers'][str(int(index) + 1)]['first_input_dma_tick']))
    points += [('final RMSNorm tail after decoder Y', prefill['final_norm']['last_output_ack_tick']),
               ('final norm -> LM head handoff', prefill['lm_head']['first_input_dma_tick']),
               ('last-token LM head, all vocabulary logits', prefill['lm_head']['last_logits_ack_tick'])]
    cursor, breakdown = prefill['embedding']['first_token_ids_dma_tick'], []
    for label, stop in points:
        if stop < cursor:
            raise ValueError(f'Overlapping boundary in disjoint latency partition: {label}')
        breakdown.append(dict(stage=label, start_tick=cursor, end_tick=stop, cycles=(stop - cursor) / factor))
        cursor = stop
    if sum(part['cycles'] for part in breakdown) != prefill['embedding']['end_to_end_cycles']:
        raise ValueError('Incomplete prefill latency partition')
    selection = load_selection(args.configuration)
    selected_guest = selection['measured_guest_sha256']
    provenance = load(cases['prefill'] / 'reuse_window_probe/decoder_stack_provenance.json')
    if library != selection['measured_library_sha256'] or provenance['guest_sha256'] != selected_guest:
        raise ValueError('Run differs from selected prefill guest/library')
    token = prefill['lm_head']['next_token']
    if token['status'] != 'PASS' or not token['composed_reference_agrees']:
        raise ValueError('Next-token selection does not match the independent references')
    report = dict(status='PASS', generated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                  latency_unit='cycles', reference_clock_hz=10**9, ticks_per_cycle=factor,
                  library_sha256=library, guest_sha256=selected_guest, stages=rows,
                  prefill_disjoint_breakdown=breakdown,
                  prefill_decoder_cycles=d['end_to_end_cycles'],
                  prefill_operator_spans=prefill['lifecycle']['stage_spans'],
                  prefill_numerical_checks=prefill['numerics'],
                  prefill_verified_decoder_jobs=prefill['resources']['verified_decoder_jobs'],
                  prefill_memory=prefill['memory'],
                  final_norm_span_cycles=prefill['final_norm']['rmsnorm_cycles'],
                  final_norm_overlaps_decoder=(prefill['final_norm']['first_input_dma_tick'] <
                                               d['layers'][str(shape['layers'])]['last_y_ack_tick']),
                  selected_reference_cycles=selection['full_cycles'],
                  delta_from_selected_cycles=prefill['embedding']['end_to_end_cycles'] - selection['full_cycles'],
                  next_token=prefill['lm_head']['next_token'], evidence_sha256=evidence,
                  invocations=invocations,
                  notes=['Four standalone workloads have different shapes and boundaries; their latencies must not be added.',
                         'prefill uses synthetic sparse-valued weights; all dense hardware tiles execute. This is a two-layer workload.',
                         'Host preparation, simulation wall time, verification, and host argmax are excluded from accelerator cycles.',
                         'Operator activity spans overlap; use the disjoint DMA boundary partition for the complete latency sum.'])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    buffer = io.StringIO()
    columns = ('stage', 'status', 'start_cycle', 'end_cycle', 'end_to_end_cycles', 'boundary', 'artifacts')
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    (output / 'summary.csv').write_text(buffer.getvalue())
    lines = ['# 四阶段正式重跑与端到端延迟', '',
             '全部使用同一安装库，数值与相应资源检查通过。延迟单位为 1 GHz 参考 cycles；SST tick 为 1 ps。', '',
             '| 测试 | 状态 | 端到端 cycles | 边界 |', '| --- | --- | ---: | --- |']
    for row in rows:
        lines.append(f"| {row['stage']} | PASS | {row['end_to_end_cycles']:,.3f} | {row['boundary']} |")
    lines += ['', '## prefill 完整分解', '',
              'S1024 / H2048 / FFN8192 / 2 层 / vocab128256，16 workers、MPI8；合成输入与权重，所有稠密 tile 实际执行。', '',
              '| 互不重叠的区间 | cycles |', '| --- | ---: |']
    for part in breakdown:
        lines.append(f"| {part['stage']} | {part['cycles']:,.3f} |")
    lines += [f"| **合计** | **{prefill['embedding']['end_to_end_cycles']:,.3f}** |", '',
              f"两层 decoder：{d['end_to_end_cycles']:,.3f} cycles。",
              f"相对已选配置差值：{report['delta_from_selected_cycles']:,.3f} cycles。",
              f"最终 greedy token ID：{report['next_token']['token_id']}，local/composed reference 检查通过。", '',
              'Final RMSNorm 提前启动并与 decoder 尾部重叠，上表只累计 decoder 完成后的剩余尾部。',
              '完整延迟从首次 token-ID DMA 发送到最后有效 token 的全词表 logits DMA ACK；主机 argmax 不计入。',
              '四个独立算子的配置和测量边界不同，不能相加得到 prefill 延迟。原始形状、命令、检查、证据哈希和主机时间见 summary.json。', '',
              '## decoder 算子活动跨度', '',
              '下表为跨 worker 的算子活动跨度，彼此可能重叠，仅作诊断，不能累加为端到端延迟。', '',
              '| 层 | 算子 | span cycles |', '| --- | --- | ---: |']
    labels = {'1': 'Wo', '2': 'residual Add / Norm2 fusion', '4': 'Gate/Up/SiLU/Multiply/Down/residual fusion'}
    for layer, spans in prefill['lifecycle']['stage_spans'].items():
        for stage, span in spans.items():
            lines.append(f"| {layer} | {labels.get(stage, 'stage ' + stage)} | {span['span_cycles']:,.3f} |")
    lines.append('')
    (output / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({k: report[k] for k in ('status', 'delta_from_selected_cycles')}, indent=2))
    for row in rows:
        print(row['stage'], row['end_to_end_cycles'], 'cycles')


if __name__ == '__main__':
    main()
