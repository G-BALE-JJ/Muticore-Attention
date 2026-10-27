#!/usr/bin/env python3
"""Audit archived logged guests and the new clean performance baseline."""
import argparse
import json
from pathlib import Path

from analyze_projection_startup import compare
from run_projection_overlap import ROOT, digest


def validate_snapshot(snapshot, hashes):
    for name, expected in hashes.items():
        if digest(snapshot / name) != expected:
            raise ValueError(f'archived source/binary hash mismatch: {name}')


def combine(root):
    root = Path(root)
    logged = json.loads((root / 'logged_release_summary.json').read_text())
    if logged['status'] != 'PASS' or logged['completed_cases'] != 4:
        raise ValueError('logged release comparison is incomplete')
    rows = {}
    environments = {}
    for row in logged['cases']:
        sequence, variant = row['shape']['query_length'], row['variant']
        case = Path(row['artifact_root'])
        batch_root = case.parent.parent
        manifest = json.loads((batch_root / 'source_sha256.json').read_text())
        if manifest != logged['source_sha256'] or row['sha256'] != manifest:
            raise ValueError('logged case source version differs from validated summary')
        validate_snapshot(batch_root / 'source_snapshot', manifest)
        provenance = json.loads((case / 'provenance.json').read_text())
        environments[(sequence, variant)] = provenance['environment']
        rows[(sequence, variant)] = row
    comparisons = {}
    clean_versions = None
    for sequence in (1024, 2048):
        case = root / f'clean_formal_s{sequence}'
        row = json.loads((case / 'experiment_result.json').read_text())
        if row['status'] != 'PASS' or row['byte_exact']['status'] != 'PASS':
            raise ValueError('clean startup output verification missing')
        if row['shape']['query_length'] != sequence or row['variant'] != 'clean_startup':
            raise ValueError('clean startup case identity mismatch')
        provenance = json.loads((case / 'provenance.json').read_text())
        if provenance['compile_time_startup_diagnostics'] != 0 or provenance['compile_time_batch_release'] != 0:
            raise ValueError('clean guest has unexpected compile-time settings')
        validate_snapshot(case / 'source_snapshot', provenance['sha256'])
        if clean_versions is not None and clean_versions != provenance['sha256']:
            raise ValueError('clean startup source versions differ')
        clean_versions = provenance['sha256']
        allowed = {'src/sst/elements/golem/tests/small/muticore_attention/golem_attention_runtime.cpp',
                   'src/sst/elements/golem/tests/small/muticore_attention/golem_projection_runtime.h',
                   'src/sst/elements/golem/tests/small/muticore_attention/Makefile'}
        shared = set(clean_versions) & set(logged['source_sha256'])
        differences = {name for name in shared if clean_versions[name] != logged['source_sha256'][name]}
        if differences - allowed:
            raise ValueError(f'non-diagnostic model change: {differences - allowed}')
        ignored_env = {'GOLEM_ATTENTION_GUEST_EXECUTABLE', 'GOLEM_MPI_ARGS',
                       'GOLEM_MPI_RANK_LOG_ROOT', 'GOLEM_PROJECTION_INPUT_REFERENCE'}
        env = {k: v for k, v in provenance['environment'].items() if k not in ignored_env}
        for variant in ('baseline', 'batch_release'):
            previous = rows[(sequence, variant)]
            previous_env = {k: v for k, v in environments[(sequence, variant)].items() if k not in ignored_env}
            if env != previous_env:
                raise ValueError('clean guest hardware/dataflow environment differs')
            if previous['weight_programs'] != row['weight_programs']:
                raise ValueError('Projection programming work changed')
            for core in previous['dataflow_by_worker']:
                for name in ('array_launches', 'compute_busy_cycles'):
                    if previous['dataflow_by_worker'][core][name] != row['dataflow_by_worker'][core][name]:
                        raise ValueError('Projection computation work changed')
        rows[(sequence, 'clean_startup')] = row
        comparisons[str(sequence)] = {
            'batch_release_vs_logged_baseline': compare(rows[(sequence, 'baseline')], rows[(sequence, 'batch_release')]),
            'clean_vs_logged_baseline': compare(rows[(sequence, 'baseline')], row),
            'clean_vs_batch_release': compare(rows[(sequence, 'batch_release')], row)}
    for name, expected in clean_versions.items():
        if digest(ROOT / name) != expected:
            raise ValueError('clean experiment source/binary changed after simulation')
    quiet = ROOT / 'src/sst/elements/golem/tests/small/muticore_attention/riscv64/clean_startup/fused_attention'
    quiet_batch = quiet.parent.parent / 'clean_batch/fused_attention'
    if digest(quiet) != digest(quiet_batch):
        raise ValueError('quiet batching equivalence check failed')
    return {'status': 'PASS', 'completed_formal_cases': 6,
            'cases': [rows[(s, v)] for s in (1024, 2048) for v in ('baseline', 'batch_release', 'clean_startup')],
            'comparisons': comparisons, 'logged_source_sha256': logged['source_sha256'],
            'clean_source_sha256': clean_versions,
            'quiet_batch_binary_equivalent': True,
            'interpretation': 'Removing startup diagnostics establishes a clean performance baseline. Report its difference as removal of measurement overhead, not an accelerator/dataflow speedup.'}


def markdown(result):
    lines = ['# 启动实验与干净性能基线', '',
             '同一 IW 数据流、原硬件资源、相同输入；纯启动诊断在 clean guest 中编译移除。', '',
             '| 配置 | S1024 原 E2E | S2048 原 E2E | S1024 最终 O ACK（从仿真起点） | S2048 最终 O ACK（从仿真起点） |',
             '|---|---:|---:|---:|---:|']
    for variant, label in (('baseline', '原日志对照'), ('batch_release', '保留日志、先连续 release'),
                           ('clean_startup', '关闭启动诊断，新性能基线')):
        a, b = (next(r for r in result['cases'] if r['variant'] == variant and r['shape']['query_length'] == s) for s in (1024, 2048))
        lines.append(f'| {label} | {a["end_to_end_cycles"]:,} | {b["end_to_end_cycles"]:,} | {a["startup"]["last_output_ack_cycle_from_simulation_zero"]:,} | {b["startup"]["last_output_ack_cycle_from_simulation_zero"]:,} |')
    lines += ['', '关闭诊断后，batch-release 与原循环的 guest 二进制完全一致；无需两套性能调度。', '',
              '错误报告、最终完成报告、NumPy/HBM 检查、宿主侧控制/资源 trace 继续保留。所有正式 case 输出逐字节一致，计算量不变。', '',
              '新基线清除了测量路径的诊断开销，不能把这项差值包装为算法或硬件加速。以后优化须在新基线上重新做端到端对照。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, required=True)
    args = parser.parse_args()
    result = combine(args.artifact_root)
    (args.artifact_root / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.artifact_root / 'summary.md').write_text(markdown(result))
    print('PASS six formal cases; archived versions, hardware, data, workload and completion accounting')


if __name__ == '__main__':main()
