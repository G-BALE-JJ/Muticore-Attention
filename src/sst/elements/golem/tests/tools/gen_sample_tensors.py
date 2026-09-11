#!/usr/bin/env python3

import argparse
import os
import struct


def build_a(m: int, k: int):
    vals = []
    for i in range(m):
        for j in range(k):
            vals.append(((i * 3 + j * 5) % 17) - 8)
    return vals


def build_b(k: int, n: int):
    vals = []
    for i in range(k):
        for j in range(n):
            vals.append(((i * 7 + j * 2) % 19) - 9)
    return vals


def write_i32_bin(path: str, values):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}i", *values))


def write_f32_bin(path: str, values):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}f", *[float(v) for v in values]))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate sample tensors for GOLEM matmul"
    )
    parser.add_argument("--m", type=int, required=True, help="Rows of A")
    parser.add_argument("--n", type=int, required=True, help="Cols of B")
    parser.add_argument("--k", type=int, required=True, help="Cols of A / Rows of B")
    parser.add_argument(
        "--a-out", default="data/a.bin", help="Output path for A (.bin int32)"
    )
    parser.add_argument(
        "--b-out", default="data/b.bin", help="Output path for B (.bin int32)"
    )
    parser.add_argument(
        "--dtype",
        default="int32",
        choices=["int32", "fp16", "fp32"],
        help="Sample tensor binary dtype",
    )
    args = parser.parse_args(argv)

    if args.m <= 0 or args.n <= 0 or args.k <= 0:
        raise ValueError("m/n/k must be positive")

    a_vals = build_a(args.m, args.k)
    b_vals = build_b(args.k, args.n)

    if args.dtype == "fp32":
        write_f32_bin(args.a_out, a_vals)
        write_f32_bin(args.b_out, b_vals)
    elif args.dtype == "fp16":
        os.makedirs(os.path.dirname(os.path.abspath(args.a_out)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.b_out)), exist_ok=True)
        with open(args.a_out, "wb") as f:
            f.write(struct.pack(f"<{len(a_vals)}e", *[float(v) for v in a_vals]))
        with open(args.b_out, "wb") as f:
            f.write(struct.pack(f"<{len(b_vals)}e", *[float(v) for v in b_vals]))
    else:
        write_i32_bin(args.a_out, a_vals)
        write_i32_bin(args.b_out, b_vals)

    print(
        f"[GEN] A -> {args.a_out} (shape={args.m}x{args.k}, elems={len(a_vals)}, dtype={args.dtype})"
    )
    print(
        f"[GEN] B -> {args.b_out} (shape={args.k}x{args.n}, elems={len(b_vals)}, dtype={args.dtype})"
    )


if __name__ == "__main__":
    main()
