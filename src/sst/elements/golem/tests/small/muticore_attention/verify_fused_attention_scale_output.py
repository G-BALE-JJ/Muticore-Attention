#!/usr/bin/env python3
"""Verify a four-node striped S256 fused Attention output."""

import argparse
import json
import math
from pathlib import Path

import attention_case


def compute_attention_blocked(
    q, k, v, query_length, kv_length, head_dim, query_tile_rows=64
):
    """Compute a bounded-memory NumPy reference for scale-point verification."""
    import numpy as np

    if query_tile_rows <= 0:
        raise ValueError("query_tile_rows must be positive")
    q_matrix = np.asarray(q, dtype=np.float64).reshape(query_length, head_dim)
    k_matrix = np.asarray(k, dtype=np.float64).reshape(kv_length, head_dim)
    v_matrix = np.asarray(v, dtype=np.float64).reshape(kv_length, head_dim)
    scale = 1.0 / math.sqrt(head_dim)
    output = []
    for begin in range(0, query_length, query_tile_rows):
        scores = q_matrix[begin:begin + query_tile_rows] @ k_matrix.T
        scores *= scale
        scores -= np.max(scores, axis=1, keepdims=True)
        np.exp(scores, out=scores)
        scores /= np.sum(scores, axis=1, keepdims=True)
        output.extend((scores @ v_matrix).ravel().tolist())
    return output


def verify(q_file, k_file, v_file, hbm_dir, output_offset,
           query_length, kv_length, num_query_heads, num_kv_heads,
           head_dim, band_rows):
    if (num_query_heads <= 0 or num_kv_heads <= 0 or
            num_query_heads % num_kv_heads != 0):
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    q = attention_case._read_f32(
        q_file, num_query_heads * query_length * head_dim
    )
    k = attention_case._read_f32(k_file, num_kv_heads * kv_length * head_dim)
    v = attention_case._read_f32(v_file, num_kv_heads * kv_length * head_dim)
    expected = []
    q_head_values = query_length * head_dim
    kv_head_values = kv_length * head_dim
    group_size = num_query_heads // num_kv_heads
    for head in range(num_query_heads):
        kv_head = head // group_size
        expected.extend(compute_attention_blocked(
            q[head * q_head_values:(head + 1) * q_head_values],
            k[kv_head * kv_head_values:(kv_head + 1) * kv_head_values],
            v[kv_head * kv_head_values:(kv_head + 1) * kv_head_values],
            query_length, kv_length, head_dim,
        ))
    actual = []
    band_values = band_rows * head_dim
    node_outputs = [
        attention_case._read_f32(
            Path(hbm_dir) / f"hbm_out_node{node}.bin",
            num_query_heads * band_values, output_offset,
        )
        for node in range(1, 5)
    ]
    for head in range(num_query_heads):
        for node_data in node_outputs:
            for query in range(band_rows):
                begin = (query * num_query_heads + head) * head_dim
                actual.extend(node_data[begin:begin + head_dim])
    mismatches = 0
    max_abs_error = 0.0
    first_mismatch = None
    for index, (got, want) in enumerate(zip(actual, expected)):
        error = abs(got - want)
        max_abs_error = max(max_abs_error, error)
        if not math.isclose(got, want, rel_tol=2.0e-4, abs_tol=2.0e-4):
            mismatches += 1
            if first_mismatch is None:
                first_mismatch = {
                    "head": index // q_head_values,
                    "query": (index % q_head_values) // head_dim,
                    "dim": index % head_dim,
                    "actual": got,
                    "expected": want,
                    "abs_error": error,
                }
    return {
        "status": "PASS" if mismatches == 0 else "FAIL",
        "checked": len(expected),
        "mismatches": mismatches,
        "max_abs_error": max_abs_error,
        "first_mismatch": first_mismatch,
        "shape": {
            "num_query_heads": num_query_heads, "num_kv_heads": num_kv_heads,
            "gqa_group_size": group_size,
            "query_length": query_length, "kv_length": kv_length,
            "head_dim": head_dim,
        },
        "hbm_output_nodes": [1, 2, 3, 4],
        "output_layout": "query_major_[query_length,num_query_heads,head_dim]",
        "score_probability_hbm_bytes": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-file", required=True)
    parser.add_argument("--k-file", required=True)
    parser.add_argument("--v-file", required=True)
    parser.add_argument("--hbm-dir", required=True)
    parser.add_argument("--output-offset", type=lambda value: int(value, 0), required=True)
    parser.add_argument("--query-length", "--queries", dest="query_length",
                        type=int, required=True)
    parser.add_argument("--kv-length", "--keys", dest="kv_length",
                        type=int, required=True)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--num-query-heads", "--query-heads",
                        dest="num_query_heads", type=int)
    parser.add_argument("--num-kv-heads", "--kv-heads",
                        dest="num_kv_heads", type=int)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--band-rows", type=int, required=True)
    parser.add_argument("--result-json")
    args = parser.parse_args()
    num_query_heads = args.num_query_heads or args.heads or 1
    num_kv_heads = args.num_kv_heads or (
        args.heads if args.heads is not None else num_query_heads
    )
    result = verify(args.q_file, args.k_file, args.v_file, args.hbm_dir,
                    args.output_offset, args.query_length, args.kv_length,
                    num_query_heads, num_kv_heads, args.head_dim, args.band_rows)
    print(json.dumps(result, indent=2))
    if args.result_json:
        Path(args.result_json).write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
