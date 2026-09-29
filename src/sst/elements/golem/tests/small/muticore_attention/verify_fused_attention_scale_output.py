#!/usr/bin/env python3
"""Verify a four-node striped S256 fused Attention output."""

import argparse
import json
import math
from pathlib import Path

import attention_case


def compute_attention_blocked(
    q, k, v, query_length, kv_length, head_dim, query_tile_rows=64,
    causal=False,
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
        if causal:
            rows = np.arange(begin, min(begin + query_tile_rows, query_length))
            scores[np.arange(kv_length)[None, :] > rows[:, None]] = -np.inf
        scores -= np.max(scores, axis=1, keepdims=True)
        np.exp(scores, out=scores)
        scores /= np.sum(scores, axis=1, keepdims=True)
        output.extend((scores @ v_matrix).ravel().tolist())
    return output


def rotate_rope(values, positions, head_dim, table):
    import numpy as np

    source = np.asarray(values, dtype=np.float32).reshape(positions, head_dim)
    coefficients = np.asarray(table, dtype=np.float32).reshape(-1, head_dim)
    cosine = coefficients[:positions, 0::2]
    sine = coefficients[:positions, 1::2]
    rotated = np.empty_like(source)
    rotated[:, 0::2] = source[:, 0::2] * cosine - source[:, 1::2] * sine
    rotated[:, 1::2] = source[:, 0::2] * sine + source[:, 1::2] * cosine
    return rotated.astype(np.float16).astype(np.float64).ravel().tolist()


def verify(q_file, k_file, v_file, hbm_dir, output_offset,
           query_length, kv_length, num_query_heads, num_kv_heads,
           head_dim, band_rows, dtype="fp32", causal=False, rope_table_file=None):
    if (num_query_heads <= 0 or num_kv_heads <= 0 or
            num_query_heads % num_kv_heads != 0):
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    q = attention_case._read_tensor(
        q_file, num_query_heads * query_length * head_dim, dtype
    )
    k = attention_case._read_tensor(k_file, num_kv_heads * kv_length * head_dim, dtype)
    v = attention_case._read_tensor(v_file, num_kv_heads * kv_length * head_dim, dtype)
    if rope_table_file:
        if dtype != "fp16":
            raise ValueError("RoPE reference requires fp16")
        table = attention_case._read_tensor(
            rope_table_file, max(query_length, kv_length) * head_dim, "fp16")
    expected = []
    q_head_values = query_length * head_dim
    kv_head_values = kv_length * head_dim
    group_size = num_query_heads // num_kv_heads
    for head in range(num_query_heads):
        kv_head = head // group_size
        q_head = q[head * q_head_values:(head + 1) * q_head_values]
        k_head = k[kv_head * kv_head_values:(kv_head + 1) * kv_head_values]
        if rope_table_file:
            q_head = rotate_rope(q_head, query_length, head_dim, table)
            k_head = rotate_rope(k_head, kv_length, head_dim, table)
        expected.extend(compute_attention_blocked(
            q_head,
            k_head,
            v[kv_head * kv_head_values:(kv_head + 1) * kv_head_values],
            query_length, kv_length, head_dim, causal=causal,
        ))
    actual = []
    band_values = band_rows * head_dim
    node_outputs = [
        attention_case._read_tensor(
            Path(hbm_dir) / f"hbm_out_node{node}.bin",
            num_query_heads * band_values, dtype, output_offset,
        )
        for node in range(1, 5)
    ]
    for head in range(num_query_heads):
        for node_data in node_outputs:
            for query in range(band_rows):
                begin = (query * num_query_heads + head) * head_dim
                actual.extend(node_data[begin:begin + head_dim])
    atol = (2.0e-4 if causal else 2.0e-5) if dtype == "fp16" else 2.0e-4
    rtol = 2.0e-3 if dtype == "fp16" else 2.0e-4
    mismatches = 0
    max_abs_error = 0.0
    first_mismatch = None
    for index, (got, want) in enumerate(zip(actual, expected)):
        error = abs(got - want)
        max_abs_error = max(max_abs_error, error)
        if not math.isfinite(got) or not math.isclose(got, want, rel_tol=rtol, abs_tol=atol):
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
        "status": "PASS" if mismatches == 0 and any(actual) else "FAIL",
        "dtype": dtype,
        "causal": causal,
        "rope": bool(rope_table_file),
        "atol": atol,
        "rtol": rtol,
        "output_nonzero": any(actual),
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
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--rope-table-file")
    parser.add_argument("--result-json")
    args = parser.parse_args()
    num_query_heads = args.num_query_heads or args.heads or 1
    num_kv_heads = args.num_kv_heads or (
        args.heads if args.heads is not None else num_query_heads
    )
    result = verify(args.q_file, args.k_file, args.v_file, args.hbm_dir,
                    args.output_offset, args.query_length, args.kv_length,
                    num_query_heads, num_kv_heads, args.head_dim, args.band_rows,
                    args.dtype, args.causal, args.rope_table_file)
    print(json.dumps(result, indent=2))
    if args.result_json:
        Path(args.result_json).write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
