#!/usr/bin/env python3
"""Reconcile the retained-output baseline's measured latency and compute floor."""
import argparse
import json
from pathlib import Path
import re


JOB = re.compile(r'\[PROJECTION_JOB\] core=(\d+) start=(\d+) end=(\d+) cycles=(\d+)')
LOCAL_WAIT = re.compile(r'\[PROJECTION_SYNC\] core=(\d+) stage=local_wait cycle=(\d+) status=0')


def partition_service_model(groups):
    """Port/compute-only illustration for the existing 64 physical arrays."""
    if groups not in (1, 2):
        raise ValueError('illustration supports one or two fixed groups')
    width = 64 // groups
    matrix = 1 + 8192 // 256 + (width - 1).bit_length()
    scatter = 1 + width * 128 // 256
    port_ready, group_ready, tile_counts = 0, [0] * groups, [0] * groups
    while min(tile_counts) < 32:
        group = min((g for g in range(groups) if tile_counts[g] < 32),
                    key=lambda g: (max(port_ready, group_ready[g]), g))
        port_ready = max(port_ready, group_ready[group]) + matrix + scatter
        group_ready[group] = port_ready + 66
        tile_counts[group] += 1
    return {'physical_arrays': 64, 'groups': groups, 'rows_per_group': width,
            'hidden_tiles': 32, 'matrix_service_per_group': matrix,
            'scatter_service_per_group': scatter, 'mac_per_group': 66,
            'port_service_cycles': groups * 32 * (matrix + scatter),
            'makespan_cycles': max(group_ready),
            'scope': 'Illustration only: excludes GM, output, WCP command latency, head boundaries and contention; not SST evidence.'}


def analyze(case):
    case = Path(case)
    report = json.loads((case / 'projection_e2e_result.json').read_text())
    if report['status'] != 'PASS' or not report.get('single_operand_context_valid'):
        raise ValueError('requires a verified single-context baseline')
    projection = report['stages']['projection']
    logs = list((case / 'logs').glob('attention-sst*.log'))
    if len(logs) != 1:
        raise ValueError('requires exactly one merged SST log')
    log_text = logs[0].read_text(errors='replace')
    jobs = [tuple(map(int, values)) for values in JOB.findall(log_text)]
    if sorted(job[0] for job in jobs) != list(range(4, 20)):
        raise ValueError('requires one job for each worker')
    completions = [tuple(map(int, values)) for values in LOCAL_WAIT.findall(log_text)]
    if sorted(core for core, _ in completions) != list(range(4, 20)):
        raise ValueError('requires worker completion timing')
    critical_core, completion = max(completions, key=lambda entry: entry[1])
    core, start, end, duration = next(job for job in jobs if job[0] == critical_core)
    if completion != projection['end_cycle'] or end - start != duration or end > completion:
        raise ValueError('critical worker timing mismatch')
    breakdown = projection['cycle_breakdown_by_worker'][str(core)]
    if projection['array_count'] != 64 or not projection['accumulation_by_worker'][str(core)]['retain_output']:
        raise ValueError('requires the 64-array retained-output baseline')
    floor = report['floor_components_cycles']
    stages = {
        'worker_start_skew': start - projection['start_cycle'],
        'norm_vector': breakdown['norm_wait'],
        'foreground_local_gm_read': breakdown['local_read'],
        'matrix_program_and_submit': breakdown['program_weight'] + breakdown['program_weight_ready'],
        'input_scatter_and_prepare': breakdown['program_input'] + breakdown['load_input'],
        'array_launch_and_compute': breakdown['launch'],
        'output_read': breakdown['read_output'],
    }
    stages['other_projection_phases'] = projection['elapsed_cycles'] - sum(stages.values())
    if stages['other_projection_phases'] < 0 or stages['norm_vector'] != floor['rmsnorm_vector']:
        raise ValueError('phase reconciliation failed')
    gap = dict(stages)
    del gap['norm_vector']
    gap['array_launch_over_compute_floor'] = gap.pop('array_launch_and_compute') - floor['projection_array']
    gap['projection_to_attention_handoff'] = report['stage_handoff_cycles']['projection_to_attention']
    gap['attention_over_exp_floor'] = report['stages']['attention']['elapsed_cycles'] - floor['attention_sfu_exp']
    expected_gap = report['end_to_end_cycles'] - report['optimistic_resource_floor_cycles']
    if min(gap.values()) < 0 or sum(gap.values()) != expected_gap:
        raise ValueError('end-to-end gap reconciliation failed')
    flow = projection['input_flow_by_worker'][str(core)]
    pipeline = projection['pipeline_by_worker'][str(core)]
    return {
        'artifact_root': str(case), 'shape': report['shape'],
        'critical_worker': core, 'projection_cycles': projection['elapsed_cycles'],
        'end_to_end_cycles': report['end_to_end_cycles'],
        'compute_only_serial_reference_cycles': report['optimistic_resource_floor_cycles'],
        'gap_cycles': expected_gap, 'projection_partition_cycles': stages,
        'gap_partition_cycles': gap,
        'local_gm_operand_read_cycles': {
            'weight': breakdown['local_weight_read'], 'input': breakdown['local_input_read'],
            'fusion': breakdown['local_fusion_read']},
        'input_flow': flow, 'pipeline': pipeline,
        'scope': 'Additive measured critical-worker phase partition; not independent savings or a new overlapping-schedule lower bound.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {'status': 'PASS', 'cases': [analyze(args.artifact_root / f's{s}') for s in (1024, 2048)],
              'array_partition_service_model': [partition_service_model(g) for g in (1, 2)]}
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
