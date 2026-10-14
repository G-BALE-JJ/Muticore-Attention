#!/usr/bin/env python3
"""Auditable capacity bounds for fixed archived work, with explicit constraints.

These are necessary bounds, not an optimal achievable schedule. No queue waits
or measured inter-request gaps are charged as indispensable resource service.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

from analyze_resource_timeline import FIELDS, intersection
from report_dma_e2e import report

EXP = re.compile(r'GOLEM_SFU_HW_PIPELINE core=(\d+) unit=exp lanes=(\d+) latency=(\d+) ii=(\d+) depth=(\d+) accepted_tokens=(\d+)')


def bounds(projection, attention, serialized):
    """Different resources overlap; only proven phase ordering permits a sum."""
    p = max(projection.values())
    a = max(attention.values())
    return {'barrier_capacity_cycles': p+a,
            'group1_serialized_and_barrier_cycles': max(p, serialized)+a,
            'projection_capacity_cycles': p, 'attention_capacity_cycles': a}


def analyze(case):
    case = Path(case)
    timing = report(case)
    projection = json.loads((case/'projection_e2e_result.json').read_text())
    attention = json.loads((case/'sst_qk_bridge_result.json').read_text())
    env = json.loads((case/'provenance.json').read_text())['environment']
    if env.get('GOLEM_DIAGNOSTIC_ORACLE') != 'none':
        raise ValueError('oracle work cannot define baseline capacity')
    if env.get('GOLEM_LOCAL_GM_READ_PORTS', '1') != '1' or env.get('GOLEM_ARRAY_BUFFER_PORTS', '1') != '1':
        raise ValueError('bound requires one read port and one array buffer port')
    if any(v['array_groups'] != 1 for v in projection['stages']['projection']['dataflow_by_worker'].values()):
        raise ValueError('group1 schedule assumption differs')
    lower = timing['first_data_dma_send_tick']
    upper = timing['last_O_ack_tick']
    phase_end = projection['stages']['projection']['end_cycle'] * 1000
    jobs, exp = {}, {}
    log = next((case/'logs').glob('attention-sst*.log'))
    with log.open(errors='replace') as stream:
        for line in stream:
            m = re.search(r'\[PROJECTION_JOB\] core=(\d+) start=(\d+) end=(\d+)', line)
            if m:
                core, start, end = map(int, m.groups()); jobs[core] = (start*1000,end*1000)
            m = EXP.search(line)
            if m:
                core, lanes, latency, ii, depth, tokens = map(int,m.groups())
                if core in attention['qk_cores']:
                    exp[core] = dict(lanes=lanes, latency=latency, ii=ii, tokens=tokens,
                                     elements=lanes*tokens, issue_work_cycles=tokens*ii)
    if len(jobs) != 16 or set(exp) != set(attention['qk_cores']):
        raise ValueError('missing projection/SFU work evidence')
    if sum(e['elements'] for e in exp.values()) != attention['sfu_exp_work']['observed_elements']:
        raise ValueError('EXP tokens disagree with executed tile work')
    if any(e['latency'] < e['ii'] for e in exp.values()):
        raise ValueError('token*II bound needs latency at least II')
    totals = {stage: defaultdict(lambda: defaultdict(float)) for stage in ('projection','attention','global')}
    spans = defaultdict(list)
    sidecar = log.with_suffix('.resources.jsonl')
    with (sidecar if sidecar.exists() else log).open(errors='replace') as stream:
        for line in stream:
            if '[RESOURCE_INTERVAL]' not in line:
                continue
            raw = dict(FIELDS.findall(line))
            r = {k:v if k in ('domain','resource') else int(v) for k,v in raw.items()}
            start,end,core = r['start_tick'],r['done_tick'],r['core']
            if core not in jobs or start < lower or end > upper:
                continue
            resource = r['domain']+'.'+r['resource']
            if resource not in ('gm.local_read','gm.local_write','array.compute',
                    'array.matrix_broadcast','array.input_scatter','array.output_scatter_gather'):
                continue
            # Whole service intervals, not RTT, wait, or overlapping WCP proxy.
            cycles = (end-start)/1000
            totals['global'][core][resource] += cycles
            if jobs[core][0] <= start and end <= jobs[core][1]:
                stage = 'projection'
                spans[(core,resource)].append((start,end))
            elif start >= phase_end:
                stage = 'attention'
            else:
                continue
            totals[stage][core][resource] += cycles
    def capacity(stage):
        cores = totals[stage]
        return {
            'array_compute': max(v.get('array.compute',0) for v in cores.values()),
            'single_GM_read_port': max(v.get('gm.local_read',0) for v in cores.values()),
            'single_GM_write_port': max(v.get('gm.local_write',0) for v in cores.values()),
            'shared_array_buffer_port': max(sum(v.get(k,0) for k in
                ('array.matrix_broadcast','array.input_scatter','array.output_scatter_gather')) for v in cores.values()),
        }
    p,a = capacity('projection'),capacity('attention')
    # Independent of ordering between fused RMSNorm and projection MAC.
    p['RMSNorm_vector'] = max(v['norm_vector_cycles'] for v in projection['fusion_by_worker'].values())
    a['EXP_issue'] = max(v['issue_work_cycles'] for v in exp.values())
    serialized_by_core = {}
    for core,v in totals['projection'].items():
        resources = ('array.matrix_broadcast','array.input_scatter','array.compute','array.output_scatter_gather')
        groups = [spans[(core,k)] for k in resources]
        if any(intersection(groups[i],groups[j]) for i in range(4) for j in range(i)):
            raise ValueError('group1 array work overlaps; serialized bound assumption invalid')
        serialized_by_core[str(core)] = sum(v.get(k,0) for k in resources)
    result = bounds(p,a,max(serialized_by_core.values()))
    global_cap = capacity('global')
    # array.compute is a group interval, not array-lane occupancy. Summing QK/PV
    # group intervals would incorrectly forbid using disjoint existing arrays.
    group_service = global_cap.pop('array_compute')
    global_cap['projection_full_array_compute'] = p['array_compute']
    global_cap['EXP_issue'] = a['EXP_issue']
    result.update({
        'status':'PASS', 'timing':timing,
        'projection_resource_bounds_cycles':p,'attention_resource_bounds_cycles':a,
        'streaming_global_resource_bounds_cycles':global_cap,
        'streaming_capacity_cycles':max(global_cap.values()),
        'global_array_group_service_diagnostic_cycles':group_service,
        'group1_serialized_array_service_by_core':serialized_by_core,
        'EXP_work_by_core':exp,
        'actual_to_group1_bound_ratio':timing['end_to_end_cycles']/result['group1_serialized_and_barrier_cycles'],
        'actual_to_barrier_capacity_ratio':timing['end_to_end_cycles']/result['barrier_capacity_cycles'],
        'assumptions': ['same actual resource work/traffic, ownership and operation sizes as archived IW',
            'one local GM read port and one shared array buffer port per worker',
            'barrier bounds retain global Projection-to-Attention producer barrier',
            'group1 bound additionally retains whole-array program/scatter/compute/drain sequencing',
            'streaming bound permits phase overlap but retains QK/PV core ownership and archived work'],
        'limitations': ['necessary resource lower bounds, not a proven attainable optimum',
            'full dependency DAG, HBM bank constraints, NoC cuts, SFU other pipelines and minimal fill/drain not yet jointly optimized',
            'changing traffic reuse/chunking/masked-element computation requires recomputing work bounds',
            'streaming EXP bound omits RMSNorm and other SFU work; separate engines are not assumed shared without evidence'],
        'measured_service_cycles_by_stage':totals,
    })
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); result=analyze(args.case)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print({k:result[k] for k in ('barrier_capacity_cycles','group1_serialized_and_barrier_cycles','streaming_capacity_cycles')})


if __name__ == '__main__':main()
