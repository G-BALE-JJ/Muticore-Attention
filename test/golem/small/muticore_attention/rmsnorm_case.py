#!/usr/bin/env python3
"""Prepare and verify the FP16 RMSNorm HBM workload."""

import argparse
import math
import re
import struct
from pathlib import Path


OUTPUT_OFFSET = 0x02200000


def values(rows, cols):
    x = [((index * 17) % 29 - 14) / 8 for index in range(rows * cols)]
    x[:cols] = [0.0] * cols
    gamma = [0.75 + (index % 11) / 32 for index in range(cols)]
    return x, gamma


def pack_half(values_to_pack):
    return struct.pack(f"<{len(values_to_pack)}e", *values_to_pack)


def prepare(root, rows, cols):
    root.mkdir(parents=True, exist_ok=True)
    x, gamma = values(rows, cols)
    (root / "x.bin").write_bytes(pack_half(x))
    (root / "gamma.bin").write_bytes(pack_half(gamma))


def verify(root, hbm_dir, rows, cols, epsilon, log=None):
    if log is not None:
        matches = re.findall(r"\[SFU_RMSNORM\].*\bstatus=0\b", log.read_text())
        if len(matches) != 1:
            raise ValueError(f"expected one successful device RMSNorm job, got {len(matches)}")
        print(matches[0])
    x = struct.unpack(f"<{rows * cols}e", (root / "x.bin").read_bytes())
    gamma = struct.unpack(f"<{cols}e", (root / "gamma.bin").read_bytes())
    with (hbm_dir / "hbm_out_node1.bin").open("rb") as output_file:
        output_file.seek(OUTPUT_OFFSET)
        raw = output_file.read(rows * cols * 2)
    if len(raw) != rows * cols * 2:
        raise ValueError("RMSNorm HBM output is missing or truncated")
    actual = struct.unpack(f"<{rows * cols}e", raw)
    max_error = 0.0
    for row in range(rows):
        base = row * cols
        inverse_rms = 1.0 / math.sqrt(
            sum(value * value for value in x[base:base + cols]) / cols + epsilon)
        for col in range(cols):
            expected = x[base + col] * inverse_rms * gamma[col]
            got = actual[base + col]
            if not math.isfinite(got):
                raise ValueError(f"nonfinite RMSNorm output at row={row} col={col}")
            error = abs(got - expected)
            max_error = max(max_error, error)
            if error > 0.003:
                raise ValueError(
                    f"RMSNorm mismatch row={row} col={col}: got={got} expected={expected}")
    print(f"RMSNorm PASS: rows={rows} cols={cols} max_abs_error={max_error:.6g}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--hbm-dir", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--cols", type=int, default=128)
    parser.add_argument("--epsilon", type=float, default=1.0e-6)
    args = parser.parse_args()
    if args.rows <= 0 or args.cols <= 0 or args.epsilon <= 0:
        parser.error("rows, cols, and epsilon must be positive")
    if args.action == "prepare":
        prepare(args.case_dir, args.rows, args.cols)
    elif args.hbm_dir is None:
        parser.error("--hbm-dir is required for verification")
    else:
        verify(args.case_dir, args.hbm_dir, args.rows, args.cols,
               args.epsilon, args.log)


if __name__ == "__main__":
    main()
