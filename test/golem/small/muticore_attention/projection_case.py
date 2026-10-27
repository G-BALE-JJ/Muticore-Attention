#!/usr/bin/env python3
"""Generate FP16 RMSNorm and tiled Q/K/V projection inputs and reference tensors."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from attention_case import generate_rope_table


def generate(sequence, hq, hkv, dim, root):
    if sequence % 1024 or dim % 64 or hq < 1 or hkv < 1:
        raise ValueError("projection requires S divisible by 1024 and D by 64")
    hidden = hq * dim
    if hidden > 2048:
        raise ValueError("projection hidden dimension exceeds 2048")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1742)
    x = rng.normal(0, 0.3, (sequence, hidden)).astype("<f2")
    gamma = rng.uniform(0.9, 1.1, hidden).astype("<f2")
    xf = x.astype(np.float32)
    norm = (xf * np.reciprocal(np.sqrt(
        np.mean(xf * xf, axis=1, keepdims=True) + 1.0e-6)) *
        gamma.astype(np.float32)).astype("<f2")
    weights = rng.normal(0, 0.07 / np.sqrt(hidden),
                         (hq + 2 * hkv, dim, hidden)).astype("<f2")
    tiled = weights.reshape(hq + 2 * hkv, dim // 64, 64,
                            hidden // 64, 64).transpose(0, 1, 3, 2, 4)
    x.tofile(root / "projection_x.bin")
    gamma.tofile(root / "projection_gamma.bin")
    norm.tofile(root / "projection_norm.bin")
    tiled.tofile(root / "projection_weights.bin")
    names = (("q", 0, hq), ("k", hq, hkv), ("v", hq + hkv, hkv))
    for name, first, heads in names:
        projected = np.empty((heads, sequence, dim), dtype="<f2")
        for head in range(heads):
            output = np.zeros((sequence, dim), dtype=np.float16)
            for tile in range(hidden // 64):
                partial = np.zeros_like(output)
                for col in range(64):
                    product = (norm[:, tile * 64 + col, None] *
                               weights[first + head, :, tile * 64 + col]).astype(np.float16)
                    partial = (partial + product).astype(np.float16)
                output = (output + partial).astype(np.float16)
            projected[head] = output
        projected.tofile(root / f"{name}_{heads}x{sequence}x{dim}.bin")
    generate_rope_table(root / f"rope_{dim}.bin", sequence, dim)


def reuse_verified_inputs(reference, sequence, hq, hkv, dim, root):
    """Reuse immutable golden inputs; does not reuse HBM state or simulator output."""
    reference, root = Path(reference), Path(root)
    report = json.loads((reference / "projection_e2e_result.json").read_text())
    shape = report["shape"]
    if report["status"] != "PASS" or (shape["query_length"], shape["Hq"], shape["Hkv"], shape["head_dim"]) != (sequence, hq, hkv, dim):
        raise ValueError("cached projection input shape/status mismatch")
    hashes = json.loads((reference / "baseline_provenance.json").read_text())["input_sha256"]
    names = ["projection_x.bin", "projection_gamma.bin", "projection_norm.bin",
             "projection_weights.bin", f"rope_{dim}.bin",
             f"q_{hq}x{sequence}x{dim}.bin", f"k_{hkv}x{sequence}x{dim}.bin",
             f"v_{hkv}x{sequence}x{dim}.bin"]
    for name in names:
        source = reference / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != hashes.get(name):
            raise ValueError(f"cached projection input hash mismatch: {name}")
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copyfile(reference / name, root / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--query-heads", type=int, required=True)
    parser.add_argument("--kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reuse-from", type=Path)
    args = parser.parse_args()
    if args.reuse_from:
        reuse_verified_inputs(args.reuse_from, args.sequence, args.query_heads, args.kv_heads, args.head_dim, args.output_dir)
        return
    generate(args.sequence, args.query_heads, args.kv_heads,
             args.head_dim, args.output_dir)


if __name__ == "__main__":
    main()
