#!/usr/bin/env python3
"""Exercise worker-only projection lifecycle, fallback, and shared Attention arrays."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from run_llama_projection import GM_PROFILE
from run_causal_balance import BEST

HERE = Path(__file__).resolve().parent
CASES = [
    ('three_blocks', 3072, 4, 2, 64, True, {}),
    ('fused_rope', 1024, 4, 2, 64, True, {'GOLEM_FUSION_PROJECTION_ROPE': '1'}),
    ('unfused_norm_gm_partial', 1024, 4, 2, 64, True, {
        'GOLEM_FUSION_NORM_PROJECTION': '0', 'GOLEM_FUSION_SKIP_AUDIT_OUTPUT': '0',
        'GOLEM_PROJECTION_PACKED_NORM_INPUT': '0', 'GOLEM_PROJECTION_CBUFFER_PARTIAL': '0',
        'GOLEM_LOCAL_GM_BYTES_PER_CYCLE': '64', 'GOLEM_LOCAL_GM_BASE_LATENCY_CYCLES': '3',
        'GOLEM_LOCAL_GM_QUEUE_DEPTH': '64'}),
    ('attention_d128', 512, 8, 2, 128, False, {}),
    ('single_context_32_cbuffer', 1024, 4, 2, 64, True, {
        'GOLEM_PROJECTION_ARRAYS': '32'}),
    ('single_context_32_gm_partial', 1024, 4, 2, 64, True, {
        'GOLEM_PROJECTION_ARRAYS': '32', 'GOLEM_PROJECTION_CBUFFER_PARTIAL': '0',
        'GOLEM_LOCAL_GM_BYTES_PER_CYCLE': '64', 'GOLEM_LOCAL_GM_BASE_LATENCY_CYCLES': '3',
        'GOLEM_LOCAL_GM_QUEUE_DEPTH': '64'}),
    ('single_context_generic_hidden128', 1024, 2, 1, 64, True, {
        'GOLEM_FUSION_NORM_PROJECTION': '0', 'GOLEM_FUSION_SKIP_AUDIT_OUTPUT': '0',
        'GOLEM_PROJECTION_PACKED_NORM_INPUT': '0'}),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    results = []
    for name, sequence, hq, hkv, dim, projection, overrides in CASES:
        path = root / name
        if not args.summarize_only:
            path.mkdir(parents=True, exist_ok=False)
        env = {**os.environ, **GM_PROFILE, **BEST,
               'GOLEM_ATTENTION_CAUSAL_BALANCE': '1', **overrides}
        if not projection:
            for flag in ('GOLEM_FUSION_NORM_PROJECTION', 'GOLEM_FUSION_PROJECTION_ROPE',
                         'GOLEM_FUSION_SKIP_AUDIT_OUTPUT'):
                env[flag] = '0'
        command = [str(HERE / 'run_sst.sh'), '--query-length', str(sequence),
                   '--kv-length', str(sequence), '--num-query-heads', str(hq),
                   '--num-kv-heads', str(hkv), '--head-dim', str(dim), '--causal',
                   '--rope', '--artifact-root', str(path)]
        if projection:
            command.append('--projection')
        if not args.summarize_only:
            with (path / 'regression_driver.log').open('w') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        numerical = json.loads((path / 'fused_attention_result.json').read_text())
        logs = list((path / 'logs').glob('attention-sst*.log'))
        if len(logs) != 1:
            raise ValueError('missing SST resource evidence')
        text = logs[0].read_text(errors='replace')
        resources = {int(c): (int(a), int(s)) for c, a, s in re.findall(
            r'\[CORE_RESOURCE_OWNERSHIP\] core=(\d+) array_present=(\d+) sfu_present=(\d+)', text)}
        if sorted(resources) != list(range(20)) or any(
                resources[c] != ((0, 0) if c < 4 else (1, 1)) for c in resources):
            raise ValueError('resource topology violation')
        operands = [tuple(map(int, e)) for e in re.findall(
            r'\[ARRAY_OPERAND_STORAGE\] core=(\d+) arrays=(\d+) operand_banks=(\d+) matrix_contexts=(\d+) input_contexts=(\d+)', text)]
        if sorted(e[0] for e in operands) != list(range(4, 20)) or any(
                e[1:] != (64, 1, 64, 64) for e in operands):
            raise ValueError('single array operand storage violation')
        timing = json.loads((path / ('projection_e2e_result.json' if projection else
                                     'sst_qk_bridge_result.json')).read_text())
        if numerical['status'] != 'PASS' or timing['status'] != 'PASS' or (
                projection and not timing.get('resource_topology_valid')):
            raise ValueError('failed worker-only verification')
        results.append({'name': name, 'status': 'PASS', 'resource_topology_valid': True, 'array_operand_banks': 1,
                        'end_to_end_cycles': timing['end_to_end_cycles'],
                        'max_abs_error': numerical['max_abs_error']})
        print(name, 'PASS', flush=True)
    (root / 'summary.json').write_text(json.dumps({'status': 'PASS', 'cases': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
