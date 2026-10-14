#!/usr/bin/env python3
"""Matched NumPy and byte-exact Attention regressions beyond the Llama shape."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from run_llama_projection import GM_PROFILE, digest

HERE = Path(__file__).resolve().parent
SHAPES = [('odd_heads', 6, 3, 64, True), ('gqa1', 4, 4, 64, True),
          ('d128', 8, 2, 128, True), ('noncausal', 8, 2, 64, False)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    results = []
    for name, hq, hkv, dim, causal in SHAPES:
        cases = []
        for enabled in ('0', '1'):
            path = root / name / enabled
            path.mkdir(parents=True, exist_ok=False)
            env = {**os.environ, **GM_PROFILE, 'GOLEM_ATTENTION_CAUSAL_BALANCE': enabled}
            command = [str(HERE / 'run_sst.sh'), '--query-length', '512', '--kv-length', '512',
                       '--num-query-heads', str(hq), '--num-kv-heads', str(hkv),
                       '--head-dim', str(dim), '--rope', '--artifact-root', str(path)]
            if causal:
                command.append('--causal')
            with (path / 'regression_driver.log').open('w') as handle:
                subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
            report = json.loads((path / 'fused_attention_result.json').read_text())
            timing = json.loads((path / 'sst_qk_bridge_result.json').read_text())
            if report['status'] != 'PASS' or timing['status'] != 'PASS':
                raise ValueError('failed standalone verification')
            cases.append((path, report, timing))
        control, balanced = cases
        for source in control[0].glob('*.bin'):
            if digest(source) != digest(balanced[0] / source.name):
                raise ValueError('standalone input mismatch')
        output_bytes = 128 * hq * dim * 2
        for node in range(1, 5):
            with (control[0] / 'hbm' / f'hbm_out_node{node}.bin').open('rb') as a, \
                 (balanced[0] / 'hbm' / f'hbm_out_node{node}.bin').open('rb') as b:
                a.seek(0x02300000); b.seek(0x02300000)
                if a.read(output_bytes) != b.read(output_bytes):
                    raise ValueError(f'{name}: migrated O is not byte-exact')
        results.append({'name': name, 'Hq': hq, 'Hkv': hkv, 'D': dim, 'causal': causal,
                        'status': 'PASS', 'byte_exact_O_bytes': 4 * output_bytes,
                        'max_abs_error': balanced[1]['max_abs_error'],
                        'control_cycles': control[2]['end_to_end_cycles'],
                        'balanced_cycles': balanced[2]['end_to_end_cycles']})
        print(name, 'PASS', flush=True)
    (root / 'summary.json').write_text(json.dumps({'status': 'PASS', 'cases': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
