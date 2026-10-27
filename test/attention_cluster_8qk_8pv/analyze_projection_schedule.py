#!/usr/bin/env python3
"""Compare fixed-resource projection scheduling using exclusive controller cycles."""
import argparse
import csv
import json
from pathlib import Path
from run_llama_projection import compare_reference, summarize

PHASES = ('idle load_block load_input load_weight load_weight_dma program_weight '
          'program_weight_ready program_input launch wait_input_prefetch read_output '
          'restore_output restore_output_ready prepare_output norm_batch norm_wait '
          'rope_wait local_read local_write stage_partial write_output write_drain '
          'write_completion failed complete').split()


def partition(fields):
    """Controller residence: each sampled cycle belongs to exactly one bucket."""
    groups = {
        'array_compute_and_launch': fields['launch'],
        'foreground_input_gm_read': fields['local_input_read'],
        'foreground_weight_gm_read': fields['local_weight_read'],
        'foreground_fusion_gm_read': fields['local_fusion_read'],
        'foreground_other_gm_read': fields['local_other_read'],
        'foreground_gm_write': fields['local_write'],
        'array_output_read': fields['read_output'],
        'array_output_restore': fields['restore_output'] + fields['restore_output_ready'],
        'array_weight_program': fields['program_weight'] + fields['program_weight_ready'],
        'array_input_scatter': fields['program_input'] + fields['wait_input_prefetch'],
        'weight_hbm_load': fields['load_weight'] + fields['load_weight_dma'],
        'rmsnorm_sfu': fields['norm_wait'],
        'rope_sfu': fields['rope_wait'],
        'norm_input_dma_and_dispatch': fields['norm_batch'] + fields['load_block'],
        'output_write_submission_and_drain': fields['write_output'] + fields['write_drain'] + fields['write_completion'],
        'partial_store_dispatch': fields['stage_partial'],
        'input_and_output_dispatch': fields['load_input'] + fields['prepare_output'],
        'inactive_or_failure': fields['idle'] + fields['failed'] + fields['complete'],
    }
    if sum(groups.values()) != sum(fields[k] for k in PHASES):
        raise ValueError('cycle partition does not cover the exclusive state counters')
    if fields['local_read'] != sum(fields[k] for k in (
            'local_input_read', 'local_weight_read', 'local_fusion_read', 'local_other_read')):
        raise ValueError('local read role counters do not partition local_read')
    return groups


def output_fabric_stats(case):
    keys = {'output_scatter_requests', 'output_gather_requests',
            'output_scatter_gather_bytes', 'output_scatter_gather_transfer_cycles'}
    values = {str(core): {} for core in range(4)}
    for path in case.glob('stats/**/stats_selfcom*.txt'):
        for row in csv.reader(path.open()):
            if len(row) > 6 and row[1] in keys:
                for core in values:
                    if row[0] == f'core{core}:rocc:array':
                        values[core][row[1]] = max(values[core].get(row[1], 0), int(row[6]))
    if any(set(stats) != keys for stats in values.values()):
        raise ValueError('missing array output fabric statistics')
    return values


def analyze(root, variant='wrap_prefetch'):
    if variant not in ('wrap_prefetch', 'restore_ahead'):
        raise ValueError('unknown projection schedule variant')
    schedule_flag = 'GOLEM_PROJECTION_WRAP_PREFETCH' if variant == 'wrap_prefetch' else 'GOLEM_PROJECTION_PARTIAL_RESTORE_AHEAD'
    rows, partitions = [], {}
    for sequence in (1024, 2048):
        before = root / 'control' / f's{sequence}'
        after = root / variant / f's{sequence}'
        cases = [summarize(p) for p in (before, after)]
        comparison = compare_reference(after, before)
        prov = [c['provenance'] for c in cases]
        binary = 'install/lib/sst-elements-library/libgolem.so'
        if prov[0]['sha256'][binary] != prov[1]['sha256'][binary]:
            raise ValueError('comparison requires the same simulator library')
        for key in ('projection_x.bin', 'projection_gamma.bin', 'projection_weights.bin', 'rope_64.bin'):
            if prov[0]['input_sha256'][key] != prov[1]['input_sha256'][key]:
                raise ValueError(f'input mismatch: {key}')
        exclusions = {schedule_flag, 'GOLEM_MPI_ARGS',
                      'GOLEM_PROJECTION_INPUT_REFERENCE'}
        envs = [{k: v for k, v in p['environment_overrides'].items() if k not in exclusions} for p in prov]
        if envs[0] != envs[1]:
            raise ValueError('hardware or execution configuration mismatch')
        for name, case in zip(('control', variant), cases):
            pipeline = case['pipeline']
            stage = pipeline['stages']['projection']
            detail = stage['cycle_breakdown_by_manager']
            grouped = {core: partition(fields) for core, fields in detail.items()}
            if stage['end_cycle'] - pipeline['chain_start_cycle'] + pipeline['stages']['attention']['elapsed_cycles'] + pipeline['stage_handoff_cycles']['projection_to_attention'] != pipeline['end_to_end_cycles']:
                raise ValueError('E2E stage partition mismatch')
            expected = int(name != 'control')
            schedule_detail = detail if variant == 'wrap_prefetch' else stage['partial_movement_by_manager']
            schedule_field = 'wrap_prefetch' if variant == 'wrap_prefetch' else 'restore_ahead'
            if set(detail) != set(map(str, range(4))) or any(f[schedule_field] != expected for f in schedule_detail.values()):
                raise ValueError('incomplete or incorrect scheduling evidence')
            partitions[f'{name}/s{sequence}'] = {
                'by_manager': grouped,
                'controller_ticks_by_manager': {core: sum(g.values()) for core, g in grouped.items()},
                'raw_state_counters': detail,
            }
            rows.append(dict(variant=name, sequence=sequence,
                e2e_cycles=pipeline['end_to_end_cycles'],
                producer_span_cycles=stage['end_cycle'] - pipeline['chain_start_cycle'],
                attention_span_cycles=pipeline['stages']['attention']['elapsed_cycles'],
                handoff_cycles=pipeline['stage_handoff_cycles']['projection_to_attention'],
                foreground_input_gm_cycles_max=max(g['foreground_input_gm_read'] for g in grouped.values()),
                foreground_local_gm_cycles_max=max(f['local_read'] for f in detail.values()),
                local_read_bytes=sum(g['read_bytes'] for g in stage['local_gm_by_manager'].values()),
                cbuffer_read_bytes=sum(g['cbuffer_read_bytes'] for g in stage['schedule_by_manager'].values()),
                cbuffer_write_bytes=sum(g['cbuffer_write_bytes'] for g in stage['schedule_by_manager'].values()),
                array_compute_cycles_max=max(g['array_compute_and_launch'] for g in grouped.values()),
                O_max_abs_error=case['max_abs_errors']['O'],
                library_sha256=prov[0]['sha256'][binary]))
            if variant == 'restore_ahead':
                movement = stage['partial_movement_by_manager']
                partitions[f'{name}/s{sequence}']['partial_movement_by_manager'] = movement
                for key in ('partial_reads', 'final_reads', 'partial_read_cycles', 'final_read_cycles', 'restores_foreground', 'restores_ahead', 'restore_cbuffer_wait', 'restore_submit', 'restore_transfer', 'restore_ahead_wait', 'restore_ahead_consume', 'cbuffer_prefetch_residence', 'cbuffer_demand_residence', 'cbuffer_store_residence'):
                    rows[-1][key + '_max'] = max(f[key] for f in movement.values())
                rows[-1]['optimistic_floor_cycles'] = pipeline['optimistic_resource_floor_cycles']
                rows[-1]['actual_to_floor_ratio'] = pipeline['actual_to_floor_ratio']
        if variant == 'restore_ahead':
            fabric = [output_fabric_stats(p) for p in (before, after)]
            if fabric[0] != fabric[1]:
                raise ValueError('partial scheduling changed physical output traffic/service cost')
            for core, stats in fabric[0].items():
                f = cases[0]['pipeline']['stages']['projection']['partial_movement_by_manager'][core]
                width = cases[0]['pipeline']['stages']['projection']['array_count']
                if (stats['output_gather_requests'] != f['partial_reads'] + f['final_reads'] or
                    stats['output_scatter_requests'] != f['restores_foreground'] + f['restores_ahead'] or
                    stats['output_scatter_gather_bytes'] != (2*f['partial_reads'] + f['final_reads']) * width * 128):
                    raise ValueError('partial counters do not match physical output fabric work')
            for name in ('control', variant):
                partitions[f'{name}/s{sequence}']['output_fabric_by_manager'] = fabric[0]
        for key in ('local_read_bytes', 'cbuffer_read_bytes', 'cbuffer_write_bytes', 'array_compute_cycles_max'):
            if rows[-1][key] != rows[-2][key]:
                raise ValueError(f'scheduling changed work/traffic: {key}')
        rows[-1]['saved_cycles'] = comparison['end_to_end']['saved_cycles']
        rows[-1]['reduction_percent'] = comparison['end_to_end']['reduction_percent']
        rows[-1]['exact_checked_regions'] = len(comparison['byte_exact_regions'])
        rows[-2].update(saved_cycles=0, reduction_percent=0.0, exact_checked_regions=0)
    result = dict(status='PASS', rows=rows, exclusive_cycle_partitions=partitions,
        note='State counters are exclusive controller tick samples; asynchronous background transfers overlap them. Local-read role counters subdivide local_read, never add them twice. Controller samples include completion DMA, whereas legacy PROJECTION_JOB end captures completion submission; stage E2E uses manager-observed completion.',
        optimization=('Reuse the existing future-input/partial slot to prefetch the first row group of the next hidden pair while the last row group computes.' if variant == 'wrap_prefetch' else 'Restore the next group partial into the sole array output context after the current output has been completely drained, overlapping existing operand preparation.') + ' No extra SRAM, ports, arrays, DMA credits, output contexts, or hidden tile accumulation changes.')
    (root / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    with (root / 'analysis.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--variant', choices=('wrap_prefetch', 'restore_ahead'), default='wrap_prefetch')
    args = parser.parse_args()
    report = analyze(args.artifact_root.resolve(), args.variant)
    for row in report['rows']:
        print(row['variant'], row['sequence'], row['e2e_cycles'], row['reduction_percent'])
