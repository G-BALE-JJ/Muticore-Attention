#!/usr/bin/env python3

import argparse
import csv
import os
import random
import sys

if __package__ in {None, ""}:
    _tests_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _tests_dir not in sys.path:
        sys.path.insert(0, _tests_dir)

from golem_dtype import (
    default_tolerance,
    elem_nbytes,
    normalize_dtype,
    numpy_dtype_name,
    parse_scalar_text,
    unpack_values,
    values_close,
)


def load_matrix(path: str, rows: int, cols: int, name: str, dtype: str):
    if path.endswith(".csv"):
        vals = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                for item in row:
                    item = item.strip()
                    if item:
                        vals.append(parse_scalar_text(dtype, item))
    elif path.endswith(".npy"):
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                f"{name} uses .npy but numpy is unavailable: {path}"
            ) from exc
        arr = np.load(path)
        if arr.shape != (rows, cols):
            raise ValueError(
                f"{name} shape mismatch, expected ({rows},{cols}), got {arr.shape}"
            )
        return arr.astype(numpy_dtype_name(dtype), copy=False).tolist()
    else:
        with open(path, "rb") as f:
            data = f.read()
        expected = rows * cols * elem_nbytes(dtype)
        if len(data) != expected:
            raise ValueError(
                f"{name} binary size mismatch, expected {expected} bytes, got {len(data)}"
            )
        vals = unpack_values(dtype, data)

    if len(vals) != rows * cols:
        raise ValueError(
            f"{name} element count mismatch, expected {rows * cols}, got {len(vals)}"
        )
    out = []
    for r in range(rows):
        out.append(vals[r * cols : (r + 1) * cols])
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Verify unpacked C against A@B golden")
    parser.add_argument("--a-file", required=True)
    parser.add_argument("--b-file", required=True)
    parser.add_argument("--c-file", required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--transpose-b", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--dtype",
        default=os.getenv("GOLEM_MATMUL_DTYPE", "int32"),
        help="Tensor dtype: int32|fp16|fp32",
    )
    parser.add_argument(
        "--bias-enable",
        type=int,
        default=0,
        help="Whether scalar bias is enabled in matmul kernel (0/1)",
    )
    parser.add_argument(
        "--bias-value",
        default="0",
        help="Scalar bias value added to each C element when bias is enabled",
    )
    parser.add_argument(
        "--c-rows",
        type=int,
        default=None,
        help="Rows stored in C file (default: use --m)",
    )
    parser.add_argument(
        "--c-cols",
        type=int,
        default=None,
        help="Cols stored in C file (default: use --n)",
    )
    parser.add_argument(
        "--crop-m",
        type=int,
        default=None,
        help="Crop C to first crop-m rows before compare (default: no extra crop)",
    )
    parser.add_argument(
        "--crop-n",
        type=int,
        default=None,
        help="Crop C to first crop-n cols before compare (default: no extra crop)",
    )
    parser.add_argument(
        "--seed", type=int, default=2026, help="Random seed for column sampling"
    )
    parser.add_argument(
        "--atol", type=float, default=None, help="Absolute tolerance for fp32 compare"
    )
    parser.add_argument(
        "--rtol", type=float, default=None, help="Relative tolerance for fp32 compare"
    )
    args = parser.parse_args(argv)
    dtype = normalize_dtype(args.dtype)

    c_rows = args.c_rows if args.c_rows is not None else args.m
    c_cols = args.c_cols if args.c_cols is not None else args.n
    if c_rows <= 0 or c_cols <= 0:
        raise ValueError(f"C shape must be positive, got ({c_rows},{c_cols})")
    crop_m = args.crop_m if args.crop_m is not None else args.m
    crop_n = args.crop_n if args.crop_n is not None else args.n
    if crop_m <= 0 or crop_n <= 0:
        raise ValueError(f"crop shape must be positive, got ({crop_m},{crop_n})")
    if crop_m > c_rows or crop_n > c_cols:
        raise ValueError(
            f"crop shape ({crop_m},{crop_n}) exceeds loaded C shape ({c_rows},{c_cols})"
        )
    if crop_m != args.m or crop_n != args.n:
        raise ValueError(
            f"current verify expects crop shape == (--m,--n), got crop=({crop_m},{crop_n}) vs m/n=({args.m},{args.n})"
        )

    a = load_matrix(args.a_file, args.m, args.k, "A", dtype)
    b_rows = args.n if args.transpose_b else args.k
    b_cols = args.k if args.transpose_b else args.n
    b = load_matrix(args.b_file, b_rows, b_cols, "B", dtype)
    c_loaded = load_matrix(args.c_file, c_rows, c_cols, "C", dtype)
    c = [row[:crop_n] for row in c_loaded[:crop_m]]
    atol, rtol = default_tolerance(dtype)
    if args.atol is not None:
        atol = args.atol
    if args.rtol is not None:
        rtol = args.rtol
    bias_value = parse_scalar_text(dtype, args.bias_value)

    rng = random.Random(args.seed)

    mismatches = 0
    first = None
    max_abs = 0
    sampled = 0
    for i in range(args.m):
        j = rng.randrange(args.n)
        ref_ij = 0.0 if dtype in {"fp16", "fp32"} else 0
        for kk in range(args.k):
            b_value = b[j][kk] if args.transpose_b else b[kk][j]
            ref_ij += a[i][kk] * b_value

        if args.bias_enable != 0:
            ref_ij += bias_value

        diff = c[i][j] - ref_ij
        ad = abs(diff)
        sampled += 1
        if ad > max_abs:
            max_abs = ad
        if not values_close(dtype, c[i][j], ref_ij, atol=atol, rtol=rtol):
            mismatches += 1
            if first is None:
                first = (i, j, c[i][j], ref_ij, diff)

    if mismatches > 0:
        i, j, cv, rv, dv = first
        print(
            f"[VERIFY-C] FAIL dtype={dtype} sampled={sampled} mismatches={mismatches} max_abs_diff={max_abs} atol={atol} rtol={rtol} seed={args.seed}"
        )
        print(f"[VERIFY-C] first_mismatch at ({i},{j}): C={cv}, REF={rv}, diff={dv}")
        return 1

    print(
        f"[VERIFY-C] PASS dtype={dtype} sampled={sampled} mismatches=0 max_abs_diff={max_abs} atol={atol} rtol={rtol} seed={args.seed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
