#!/usr/bin/env python3
"""Compare fixed-resource input layouts and bounded prefetch schedules."""
import argparse
import csv
import json
from pathlib import Path

from analyze_projection_schedule import output_fabric_stats, partition
from run_llama_projection import compare_reference, summarize


PLANS = {'control': (0, 0), 'budget4': (0, 4), 'budget8': (0, 8),
         'packed': (1, 0), 'packed_budget4': (1, 4),
         'packed_budget1': (1, 1), 'packed_budget2': (1, 2)}


def analyze(root, variants, sequences):
    rows, details, comparisons = [], {}, {}
    for sequence in sequences:
        before = root / 'control' / f's{sequence}'
        control = summarize(before)
        for name in ['control', *variants]:
            case_path = root / name / f's{sequence}'
            case = control if name == 'control' else summarize(case_path)
            prov, ref = case['provenance'], control['provenance']
            binary = 'install/lib/sst-elements-library/libgolem.so'
            if prov['sha256'][binary] != ref['sha256'][binary]:
                raise ValueError('simulator library mismatch')
            for key in ('projection_x.bin', 'projection_gamma.bin', 'projection_weights.bin', 'rope_64.bin'):
                if prov['input_sha256'][key] != ref['input_sha256'][key]:
                    raise ValueError(f'input mismatch: {key}')
            exclusions = {'GOLEM_PROJECTION_PACKED_NORM_INPUT', 'GOLEM_PROJECTION_INPUT_PREFETCH_BUDGET',
                          'GOLEM_MPI_ARGS', 'GOLEM_PROJECTION_INPUT_REFERENCE'}
            envs = [{k: v for k, v in p['environment_overrides'].items() if k not in exclusions}
                    for p in (prov, ref)]
            if envs[0] != envs[1]:
                raise ValueError('hardware or execution configuration mismatch')
            pipeline = case['pipeline']
            stage = pipeline['stages']['projection']
            reference = control['pipeline']['stages']['projection']
            flows = stage['input_flow_by_manager']
            packed, budget = PLANS[name]
            if set(flows) != set(map(str, range(4))) or any(
                    (f['packed_input'], f['input_prefetch_budget']) != (packed, budget)
                    for f in flows.values()):
                raise ValueError('missing or incorrect input-flow execution evidence')
            for field in ('local_gm_by_manager', 'schedule_by_manager', 'pipeline_by_manager'):
                if field == 'local_gm_by_manager':
                    keys = ('read_bytes', 'write_bytes', 'timed', 'paired_weights', 'reuse_block')
                elif field == 'schedule_by_manager':
                    keys = ('cbuffer_read_bytes', 'cbuffer_write_bytes', 'cbuffer_bytes')
                else:
                    keys = ('staging_bytes', 'array_count', 'input_prefetches', 'partial_prefetches')
                for core in flows:
                    if any(stage[field][core][k] != reference[field][core][k] for k in keys):
                        raise ValueError(f'changed work, bytes or storage: {field}')
            for core in flows:
                for field in ('local_read_bytes', 'local_write_bytes', 'norm_vector_cycles', 'rope_vector_cycles'):
                    if pipeline['fusion_by_manager'][core][field] != control['pipeline']['fusion_by_manager'][core][field]:
                        raise ValueError('fusion bytes or computation changed')
            states = stage['cycle_breakdown_by_manager']
            if any(states[c]['launch'] != reference['cycle_breakdown_by_manager'][c]['launch'] for c in flows):
                raise ValueError('array compute work changed')
            fabric = output_fabric_stats(case_path)
            if fabric != output_fabric_stats(before):
                raise ValueError('output fabric work or service cost changed')
            comparison = compare_reference(case_path, before) if name != 'control' else None
            if comparison:
                comparisons[f'{name}/s{sequence}'] = comparison
            groups = {core: partition(fields) for core, fields in states.items()}
            details[f'{name}/s{sequence}'] = {'by_manager': groups, 'raw_state_counters': states,
                'input_flow_by_manager': flows, 'partial_movement_by_manager': stage['partial_movement_by_manager'],
                'output_fabric_by_manager': fabric}
            producer = stage['end_cycle'] - pipeline['chain_start_cycle']
            handoff = pipeline['stage_handoff_cycles']['projection_to_attention']
            attention = pipeline['stages']['attention']['elapsed_cycles']
            if producer + handoff + attention != pipeline['end_to_end_cycles']:
                raise ValueError('E2E critical-path partition mismatch')
            total_requests = [f['input_read_requests'] + f['background_input_read_requests'] for f in flows.values()]
            rows.append({'variant': name, 'sequence': sequence, 'e2e_cycles': pipeline['end_to_end_cycles'],
                'producer_span_cycles': producer, 'attention_span_cycles': attention, 'handoff_cycles': handoff,
                'foreground_weight_gm_cycles_max': max(s['local_weight_read'] for s in states.values()),
                'foreground_input_gm_cycles_max': max(s['local_input_read'] for s in states.values()),
                'foreground_norm_gm_write_cycles_max': max(s['local_write'] for s in states.values()),
                'input_read_requests_per_manager_max': max(total_requests),
                'norm_write_requests_per_manager_max': max(f['norm_write_requests'] for f in flows.values()),
                'projection_local_read_bytes': sum(f['read_bytes'] for f in stage['local_gm_by_manager'].values()),
                'array_compute_cycles_max': max(s['launch'] for s in states.values()),
                'optimistic_floor_cycles': pipeline['optimistic_resource_floor_cycles'],
                'actual_to_floor_ratio': pipeline['actual_to_floor_ratio'],
                'O_max_abs_error': case['max_abs_errors']['O'], 'library_sha256': prov['sha256'][binary],
                'saved_cycles': comparison['end_to_end']['saved_cycles'] if comparison else 0,
                'reduction_percent': comparison['end_to_end']['reduction_percent'] if comparison else 0,
                'exact_checked_regions': len(comparison['byte_exact_regions']) if comparison else 0})
    result = {'status': 'PASS', 'rows': rows, 'cycle_partitions': details, 'comparisons': comparisons,
              'scope': 'Fixed GM service model, resources, payload bytes and computation; input request counts/layout or submission timing may differ. Resource service spans overlap controller residence.'}
    (root / 'input_flow_analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    with (root / 'input_flow_analysis.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', choices=[p for p in PLANS if p != 'control'], required=True)
    parser.add_argument('--sequences', nargs='+', type=int, default=[1024, 2048])
    args = parser.parse_args()
    result = analyze(args.artifact_root.resolve(), args.variants, args.sequences)
    for row in result['rows']:
        print(row['variant'], row['sequence'], row['e2e_cycles'], row['reduction_percent'])
