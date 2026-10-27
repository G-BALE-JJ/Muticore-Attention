#!/usr/bin/env python3
"""Compare real guest release scheduling without hiding preparation time."""
import json
from pathlib import Path
import re


def startup_metrics(projection, resources, simulation_end_cycles):
    controls = resources['worker_control']
    if set(controls) != {str(core) for core in range(4, 20)}:
        raise ValueError('missing worker startup controls')
    releases = [v['release_issue_cycle'] for v in controls.values()]
    accepts = [v['job_accept_cycle'] for v in controls.values()]
    end = projection['stages']['attention']['end_cycle']
    first_release = min(releases)
    return {
        'first_release_cycle': first_release,
        'first_projection_start_cycle': projection['chain_start_cycle'],
        'worker_accept_span_cycles': max(accepts) - min(accepts),
        'manager_release_span_cycles': {str(m): max(times) - min(times)
            for m in range(4)
            for times in [[controls[str(4 + 4 * lane + m)]['release_issue_cycle']
                           for lane in range(4)]]},
        'release_to_observed_cycles_by_worker': {
            core: v['release_to_observed_cycles'] for core, v in controls.items()},
        'observed_to_accept_cycles_by_worker': {
            core: v['observed_to_accept_cycles'] for core, v in controls.items()},
        'attention_complete_cycle_from_simulation_zero': end,
        'release_to_attention_complete_cycles': end - first_release,
        'last_output_ack_cycle_from_simulation_zero': resources['last_attention_output_ack_cycle'],
        'root_complete_cycle_from_simulation_zero': resources['last_root_tensor_complete_cycle'],
        'simulation_end_cycles_from_zero': simulation_end_cycles,
        'reported_e2e_cycles': projection['end_to_end_cycles'],
    }


def analyze(case, resources):
    case = Path(case)
    log = next((case / 'logs').glob('attention-sst*.log'))
    dispatches, simulation_end = [], None
    for guest_log in sorted((case / 'stdout').rglob('stderr-*')):
        for match in re.finditer(r'\[PROJECTION_DISPATCH\] manager=(\d+) worker=(\d+) row_offset=(\d+) row_count=64', guest_log.read_text()):
            dispatches.append(tuple(map(int, match.groups())))
    with log.open(errors='replace') as stream:
        for line in stream:
            match = re.search(r'Simulation is complete, simulated time:\s*([\d.]+)\s*([munp]?s)', line)
            if match:
                # This runner fixes the existing platform at 1 GHz.
                simulation_end = float(match[1]) * {'s': 1e9, 'ms': 1e6,
                    'us': 1e3, 'ns': 1, 'ps': 1e-3}[match[2]]
    expected = {(m, 4 + 4 * lane + m, lane * 64)
                for m in range(4) for lane in range(4)}
    if set(dispatches) != expected or len(dispatches) != 16:
        raise ValueError('dispatch logs changed or releases missing')
    if simulation_end is None:
        raise ValueError('simulation completion time missing')
    projection = json.loads((case / 'projection_e2e_result.json').read_text())
    return startup_metrics(projection, resources, simulation_end)


def compare(baseline, candidate):
    if baseline['input_sha256'] != candidate['input_sha256']:
        raise ValueError('comparison inputs changed')
    result = {}
    for metric in ('reported_e2e_cycles', 'release_to_attention_complete_cycles',
                   'attention_complete_cycle_from_simulation_zero',
                   'last_output_ack_cycle_from_simulation_zero',
                   'root_complete_cycle_from_simulation_zero',
                   'simulation_end_cycles_from_zero', 'worker_accept_span_cycles'):
        old, new = baseline['startup'][metric], candidate['startup'][metric]
        result[metric] = {'baseline': old, 'candidate': new, 'saved_cycles': old - new,
                         'reduction_percent': 100 * (old - new) / old}
    result['manager_release_span_cycles'] = {
        m: {'baseline': baseline['startup']['manager_release_span_cycles'][m],
            'candidate': candidate['startup']['manager_release_span_cycles'][m]}
        for m in baseline['startup']['manager_release_span_cycles']}
    result['real_completion_improves'] = all(result[key]['saved_cycles'] > 0 for key in (
        'last_output_ack_cycle_from_simulation_zero',
        'root_complete_cycle_from_simulation_zero', 'simulation_end_cycles_from_zero'))
    return result
