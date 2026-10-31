"""Independent local RMSNorm diagnosis; does not certify the full decoder chain."""
import argparse
import json
from pathlib import Path

import numpy as np


def half(value):
    return np.asarray(value, dtype=np.float16).astype(np.float64)


def read_bands(case, offset, width, rows):
    bands = []
    for node in range(1, 5):
        with (case / 'hbm' / f'hbm_out_node{node}.bin').open('rb') as stream:
            stream.seek(offset)
            bands.append(np.frombuffer(stream.read(rows * width * 2), dtype='<f2').reshape(rows, width))
    return np.concatenate(bands)


def diagnose(case):
    manifest = json.loads((case / 'decoder_stack_inputs.json').read_text())
    width = manifest['hidden']
    rows = manifest['sequence'] // 4
    logical = manifest.get('logical_sequence', manifest['sequence'])
    layout = manifest['layout']
    x = half(read_bands(case, layout['layers'][-1]['output'], width, rows)[:logical])
    got = read_bands(case, layout['final_norm']['output'], width, rows)[:logical]
    gamma = half(np.fromfile(case / 'final_gamma.bin', dtype='<f2'))
    total = half(np.zeros(len(x)))
    for column in x.T:
        total = half(total + half(column * column))
    mean = half(total / half(width))
    inverse = half(1 / np.sqrt(half(mean + half(1e-6))))
    want = half(half(x * inverse[:, None]) * gamma)
    old_x = x.astype(np.float32)
    old_total = np.zeros(len(x), dtype=np.float32)
    for column in old_x.T:
        old_total += column * column
    old = old_x * (1 / np.sqrt(old_total / np.float32(width) + np.float32(1e-6)))[:, None] * gamma.astype(np.float32)
    # The legacy tensor converter flushes magnitudes below 2^-24.
    old[np.abs(old) < 2**-24] = np.copysign(np.float32(0), old[np.abs(old) < 2**-24])
    old = old.astype(np.float16)
    mismatch = int(np.count_nonzero(got != want))
    return dict(status='PASS' if mismatch == 0 and np.isfinite(got).all() else 'FAIL',
                operator='final RMSNorm', elements=int(got.size),
                fp16_reference_mismatches=mismatch,
                fp16_reference_max_abs_error=float(np.max(np.abs(got.astype(np.float64) - want))),
                legacy_fp32_reference_max_abs_error=float(np.max(np.abs(got.astype(np.float64) - old))),
                legacy_tolerance=0.002, sfu_internal_precision='fp16',
                reference='Independent NumPy binary16 operands and every square, sequential add, division, epsilon add, rsqrt and two multiplies',
                accelerator_latency_claim=False, full_decoder_acceptance=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.case)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
