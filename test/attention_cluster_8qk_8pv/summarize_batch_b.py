#!/usr/bin/env python3
"""Validate independent Batch B shards and produce a complete comparison."""
import argparse
import json
import re
from pathlib import Path

from analyze_critical_path import analyze
from analyze_batch_b_waits import analyze as analyze_waits
from run_batch_b import ORACLES, MATRIX, cases_for, report


def combine(baseline_root, shard_roots, sequences=(1024,2048)):
    baseline_root = Path(baseline_root)
    rows, indexed, provenance = [], {}, {}
    for sequence in sequences:
        case = baseline_root / f's{sequence}'
        row = json.loads((case/'experiment_result.json').read_text())
        if row['case'] != 'baseline': raise ValueError('reference case is not baseline')
        indexed[(sequence,'baseline')] = row
        provenance[sequence] = json.loads((case/'baseline_provenance.json').read_text())
        rows.append(row)
    for root in map(Path,shard_roots):
        summary = json.loads((root/'summary.json').read_text())
        if summary['status'] != 'PASS': raise ValueError(f'incomplete shard: {root}')
        for row in summary['cases']:
            sequence = row['shape']['query_length']
            key = (sequence,row['case'])
            if key in indexed: raise ValueError(f'duplicate case: {key}')
            if sequence not in sequences: raise ValueError(f'unexpected sequence: {sequence}')
            current = json.loads((Path(row['artifact_root'])/'baseline_provenance.json').read_text())
            previous = provenance[sequence]
            if current['input_sha256'] != previous['input_sha256']:
                raise ValueError(f'inputs changed: {key}')
            # Baseline S1024 predates shard CLI options, but has identical
            # simulator/guest/model sources and libraries. Record that single
            # runner difference rather than silently equating source versions.
            differences = [name for name,value in previous['sha256'].items()
                           if current['sha256'].get(name) != value]
            allowed = ['test/attention_cluster_8qk_8pv/run_batch_b.py']
            if any(name not in allowed for name in differences):
                raise ValueError(f'model/guest changed: {key} {differences}')
            row['provenance_difference_from_baseline'] = differences
            if row['status'] != 'PASS' or row.get('byte_exact',{}).get('status') != 'PASS':
                raise ValueError(f'case correctness missing: {key}')
            if row['fixed_origin_cycle'] != indexed[(sequence,'baseline')]['fixed_origin_cycle']:
                raise ValueError(f'origin moved: {key}')
            for name in ('GOLEM_PROJECTION_ARRAYS','GOLEM_ARRAY_OPERAND_CONTEXT_BANKS',
                         'GOLEM_LOCAL_GM_READ_PORTS','GOLEM_PROJECTION_RETAIN_OUTPUT'):
                if current['environment'].get(name) != previous['environment'].get(name):
                    raise ValueError(f'hardware/ownership mismatch: {key} {name}')
            indexed[key]=row
            rows.append(row)
    expected = {(s,n) for s in sequences for n,_,_ in cases_for('all')}
    if set(indexed)!=expected:
        raise ValueError(f'missing/unexpected cases: missing={expected-set(indexed)}, extra={set(indexed)-expected}')
    ordered=[indexed[(s,n)] for s in sequences for n,_,_ in cases_for('all')]
    for row in ordered:
        critical = analyze(row['artifact_root'])
        if critical['reconstructed_e2e_cycles'] != row['end_to_end_cycles']:
            raise ValueError('phase/control E2E reconstruction mismatch')
        case=Path(row['artifact_root'])
        (case/'critical_path_analysis.json').write_text(json.dumps(critical,indent=2)+'\n')
        row['phase_critical_worker']=critical['critical_worker']
        row['worker_start_span']=critical['worker_start_span']
        row['projection_to_attention_handoff']=critical['projection_to_attention_handoff']
        log = next((case/'logs').glob('attention-sst*.log'))
        phases={}
        with log.open(errors='replace') as stream:
            for line in stream:
                if '[PROJECTION_CYCLE_BREAKDOWN]' in line:
                    fields={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)}
                    core=fields.pop('core')
                    phases[str(core)]=fields
        if set(phases) != {str(c) for c in range(4,20)}:
            raise ValueError('missing per-worker phase breakdown')
        row['projection_phase_cycles_by_worker']=phases
        waits=analyze_waits(case)
        (case/'projection_resource_waits.json').write_text(json.dumps(waits,indent=2)+'\n')
        row['projection_resource_waits']=waits
    result=report(ordered,len(expected))
    for sequence in sequences:
        for name,oracle in result['oracles'][str(sequence)].items():
            oracle['scope_limited']=True
            oracle['scope']={
                'gm_read':'Projection input/weight local service only; other reads unchanged',
                'matrix_program':'Projection matrix-broadcast service only',
                'input_scatter':'Projection input-scatter service only',
                'weight_arrival':'Immediate immutable weight supply in current GM slots; removes DMA traffic',
                'worker_start':'16 simultaneous Projection starts at frozen baseline origin',
                'projection_barrier':'Post-all-producer notification/submission gap only; not per-head/tile overlap',
                'attention_barrier':'Ready scheduler transition cadence only; not true data/resource barriers',
            }[name]
    result['dependency_model']='Measured simulator interventions plus validated phase/control chains; no complete DAG or slack claim.'
    return result


def markdown(result):
    lines=['# 批次 B 正式结果','',
           '所有 case 保留原报告结束边界，oracle 使用各序列 baseline 的固定起点。以下 oracle 是诊断干预，不是可部署优化。','',
           '| Oracle | S1024 E2E | 节省 cycles | S2048 E2E | 节省 cycles |',
           '|---|---:|---:|---:|---:|']
    for name in ORACLES:
        a,b=(result['oracles'][str(s)][name] for s in (1024,2048))
        lines.append(f'| {name} | {a["oracle_cycles"]:,} | {a["saved_cycles"]:,} | {b["oracle_cycles"]:,} | {b["saved_cycles"]:,} |')
    lines += ['', '两个 barrier oracle 的范围只包括本文注明的轮询/提交间隙；不能把小收益外推为跨 head/tile 流水没有收益。','',
              '| 组合 | S1024 E2E | S2048 E2E |','|---|---:|---:|']
    for name in MATRIX:
        a,b=(result['factorial'][str(s)]['e2e_cycles'][name] for s in (1024,2048))
        lines.append(f'| {name} | {a:,} | {b:,} |')
    lines += ['', '完整资源时间、交互效应、起点、输出检查和版本差异见同目录 JSON。','']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root',type=Path,required=True)
    p.add_argument('--shard-roots',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=combine(args.baseline_root,args.shard_roots)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    args.output.with_suffix('.md').write_text(markdown(result))
    print(f'PASS: {result["completed_cases"]} formal cases, complete oracle and factorial results')


if __name__=='__main__':main()
