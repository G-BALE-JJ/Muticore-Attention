#!/usr/bin/env python3
"""Audit descriptor routing, causal work, transfers and fixed-resource E2E."""
import argparse
import json
import re
from pathlib import Path

from run_llama_projection import summarize, compare_reference


ROUTE = re.compile(r'\[ATTENTION_TASK_ROUTE\] manager=(\d+) head=(\d+) slot=(\d+) worker=(\d+) row=(\d+) rows=(\d+) query_length=(\d+)')
QK_TRANSFER = re.compile(r'\[ATTENTION_QK_TRANSFER\] core=(\d+) txn=(\d+) q_panels=(\d+) k_panels=(\d+) q_bytes=(\d+) k_bytes=(\d+)')
V_TRANSFER = re.compile(r'\[ATTENTION_V_TRANSFER\] core=(\d+) source=(\d+) bytes=(\d+)')
SFU = re.compile(r'GOLEM_SFU_HW_PIPELINE core=(\d+) unit=exp lanes=(\d+) latency=(\d+) ii=(\d+) depth=(\d+) accepted_tokens=(\d+) issue_wait_cycles=(\d+)')
GM = re.compile(r'GOLEM_LOCAL_GM_STATS core=(\d+) ([^\n]+)')


def audit_routes(text, shape, balanced):
    routes = [tuple(map(int, r)) for r in ROUTE.findall(text)]
    heads, group = shape['Hkv'], shape['Hq'] // shape['Hkv']
    band_rows = shape['query_length'] // 4
    expected = {(m, h, s) for m in range(4) for h in range(heads) for s in range(2)}
    if len(routes) != len(expected) or {r[:3] for r in routes} != expected:
        raise ValueError('missing, duplicated or unknown descriptor dispatch')
    costs = {str(c): {'qk_tiles': 0, 'pv_windows': 0, 'jobs': 0} for c in range(4, 12)}
    for manager, head, slot, worker, begin, rows, query_length in routes:
        destination = 3 - manager if balanced and head % 2 else manager
        if (worker != 4 + slot * 4 + destination or query_length != band_rows or
                rows != band_rows * group // 2 or
                begin != manager * band_rows * group + slot * rows):
            raise ValueError('invalid ownership, coverage or executor mapping')
        cost = costs[str(worker)]
        cost['jobs'] += 1
        for offset in range(0, rows, 64):
            last_query = manager * band_rows + (slot * rows + offset + 63) % band_rows
            cost['qk_tiles'] += last_query // 64 + 1
            cost['pv_windows'] += last_query // 256 + 1
    return {'descriptors': len(routes), 'by_core': costs,
            'coverage': 'each original head/band/slot exactly once',
            'ownership': 'original query band, unchanged Q/O addresses and completion manager'}


def metrics(path, balanced):
    case = summarize(path)
    logs = list((path / 'logs').glob('attention*.log'))
    if len(logs) != 1:
        raise ValueError('missing unique SST log')
    text = logs[0].read_text()
    attention = case['attention']
    route = audit_routes(text, attention['shape'], balanced)
    if sum(c['qk_tiles'] for c in route['by_core'].values()) != attention['qk_tiles']['observed']:
        raise ValueError('causal cost does not match executed QK tiles')
    if sum(c['pv_windows'] for c in route['by_core'].values()) != attention['pv_windows']:
        raise ValueError('causal cost does not match executed PV windows')
    qk = [tuple(map(int, r)) for r in QK_TRANSFER.findall(text)]
    v = [tuple(map(int, r)) for r in V_TRANSFER.findall(text)]
    if not qk or not v:
        raise ValueError('missing operand transfer evidence')
    transfer = {name: sum(r[index] for r in qk) for name, index in
                [('q_panels', 2), ('k_panels', 3), ('q_bytes', 4), ('k_bytes', 5)]}
    transfer.update(v_loads=len(v), v_bytes=sum(r[2] for r in v))
    transfer['note'] = ('Q/K: nonzero accepted WCP window transactions; each panel enqueues one HBM read. '
                        'V: actual broadcast-cache DMA loads, including reloads; excludes NoC multicast copies.')
    sfu = {c: {'elements': int(lanes) * int(tokens), 'issue_tokens': int(tokens),
               'issue_wait_cycles': int(wait), 'lanes': int(lanes), 'ii': int(ii)}
           for c, lanes, _, ii, _, tokens, wait in SFU.findall(text) if 4 <= int(c) < 12}
    if sum(s['elements'] for s in sfu.values()) != attention['sfu_exp_work']['observed_elements']:
        raise ValueError('SFU execution work mismatch')
    if any(sfu[c]['elements'] != cost['qk_tiles'] * 64 * 64
           for c, cost in route['by_core'].items()):
        raise ValueError('per-core SFU work does not match assigned causal windows')
    gm = {c: {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', fields)}
          for c, fields in GM.findall(text)}
    return {'e2e_cycles': case['pipeline']['end_to_end_cycles'],
            'projection_cycles': case['pipeline']['stages']['projection']['elapsed_cycles'],
            'attention_cycles': attention['end_to_end_cycles'],
            'stage_spans': attention['stage_spans'], 'route_audit': route,
            'qk_by_core': attention['qk_wcp_cycles_by_core'],
            'sfu_exp_by_core': sfu, 'pv_busy_by_core': attention['pv_service']['core_busy_cycles'],
            'sfu_exp_issue_floor_cycles': max(s['issue_tokens'] * s['ii'] for s in sfu.values()),
            'sfu_exp_balanced_issue_floor_cycles': sum(s['issue_tokens'] * s['ii'] for s in sfu.values()) // len(sfu),
            'pv_waits': attention['pv_transport_waits'], 'transfers': transfer,
            'local_gm': gm, 'v_cache': attention['v_cache'],
            'max_abs_errors': case['max_abs_errors'], 'provenance': case['provenance']}


def analyze(root):
    result = {'status': 'PASS', 'scope': 'fixed 4 managers + 8 QK/SFU + 8 PV; no resource/model changes', 'cases': []}
    for sequence in (1024, 2048):
        control_path, balanced_path = (root / name / f's{sequence}' for name in ('control', 'balanced'))
        control, balanced = metrics(control_path, False), metrics(balanced_path, True)
        a, b = control['provenance'], balanced['provenance']
        if a['sha256'] != b['sha256'] or a['input_sha256'] != b['input_sha256']:
            raise ValueError('source, library, guest binary or input mismatch')
        exclude = {'GOLEM_ATTENTION_CAUSAL_BALANCE', 'GOLEM_MPI_ARGS'}
        envs = [{k: v for k, v in p['environment_overrides'].items() if k not in exclude} for p in (a, b)]
        if envs[0] != envs[1]:
            raise ValueError('hardware or non-schedule configuration mismatch')
        comparison = compare_reference(balanced_path, control_path)
        if sum(v['compute'] for v in control['qk_by_core'].values()) != sum(v['compute'] for v in balanced['qk_by_core'].values()):
            raise ValueError('QK compute work changed')
        if balanced['e2e_cycles'] >= control['e2e_cycles']:
            raise ValueError('no net E2E benefit')
        transfer_delta = {k: balanced['transfers'][k] - control['transfers'][k]
                          for k in control['transfers'] if k != 'note'}
        result['cases'].append({'sequence': sequence, 'control': control, 'balanced': balanced,
                                'comparison': comparison, 'transfer_delta': transfer_delta})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.artifact_root)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for row in result['cases']:
        c, b = row['control'], row['balanced']
        print(f"S{row['sequence']}: {c['e2e_cycles']:,} -> {b['e2e_cycles']:,}; "
              f"saved {row['comparison']['end_to_end']['reduction_percent']:.2f}%; "
              f"transfer delta {row['transfer_delta']}")
