#!/usr/bin/env python3
"""Generate and verify deterministic QK^T attention inputs."""

import argparse
import json
import math
import struct
from pathlib import Path


def native_k_index(key: int, dim: int, head_dim: int) -> int:
    """Return the flat index for K stored in native [key, dim] order."""
    return key * head_dim + dim


def compute_qk(q, k, queries: int, keys: int, head_dim: int):
    """Compute Q[queries, D] times K[keys, D]^T in row-major order."""
    if len(q) != queries * head_dim:
        raise ValueError("Q element count does not match [queries, head_dim]")
    if len(k) != keys * head_dim:
        raise ValueError("K element count does not match native [keys, head_dim]")

    output = []
    for query in range(queries):
        q_base = query * head_dim
        for key in range(keys):
            k_base = native_k_index(key, 0, head_dim)
            acc = 0.0
            for dim in range(head_dim):
                acc += q[q_base + dim] * k[k_base + dim]
            output.append(acc)
    return output


def compute_attention(q, k, v, queries: int, keys: int, head_dim: int, causal: bool):
    """Compute scaled dot-product attention with native K/V row-major storage."""
    if len(v) != keys * head_dim:
        raise ValueError("V element count does not match [keys, head_dim]")
    scores = compute_qk(q, k, queries, keys, head_dim)
    scale = 1.0 / math.sqrt(head_dim)
    output = [0.0] * (queries * head_dim)
    for query in range(queries):
        row = [scores[query * keys + key] * scale for key in range(keys)]
        if causal:
            row = [value if key <= query else -math.inf for key, value in enumerate(row)]
        row_max = max(row)
        weights = [math.exp(value - row_max) for value in row]
        denominator = sum(weights)
        for key, weight in enumerate(weights):
            probability = weight / denominator
            v_base = key * head_dim
            o_base = query * head_dim
            for dim in range(head_dim):
                output[o_base + dim] += probability * v[v_base + dim]
    return output


def attention_logical_hbm_traffic(queries: int, keys: int, head_dim: int, fused: bool):
    """Count logical tensor traffic, excluding protocol and control metadata."""
    elem_bytes = 4
    q_bytes = queries * head_dim * elem_bytes
    kv_bytes = keys * head_dim * elem_bytes
    score_bytes = queries * keys * elem_bytes
    output_bytes = queries * head_dim * elem_bytes
    if fused:
        traffic = {
            "qkt_bytes": q_bytes + kv_bytes,
            "softmax_bytes": 0,
            "pv_bytes": kv_bytes + output_bytes,
            "score_tensor_write_bytes": 0,
            "score_tensor_read_bytes": 0,
            "probability_tensor_write_bytes": 0,
            "probability_tensor_read_bytes": 0,
        }
    else:
        traffic = {
            "qkt_bytes": q_bytes + kv_bytes + score_bytes,
            "softmax_bytes": 2 * score_bytes,
            "pv_bytes": score_bytes + kv_bytes + output_bytes,
            "score_tensor_write_bytes": score_bytes,
            "score_tensor_read_bytes": score_bytes,
            "probability_tensor_write_bytes": score_bytes,
            "probability_tensor_read_bytes": score_bytes,
        }
    traffic["total_logical_hbm_bytes"] = sum(
        traffic[name] for name in ("qkt_bytes", "softmax_bytes", "pv_bytes")
    )
    return traffic


def merge_online_tile(m, l, tile_m, tile_l, final_tile=False):
    """Return the stable online Softmax state and output scaling factors."""
    m_new = max(m, tile_m)
    alpha = 0.0 if math.isinf(m) and m < 0 else math.exp(m - m_new)
    beta = math.exp(tile_m - m_new)
    l_new = l * alpha + tile_l * beta
    normalization = 1.0 / l_new if final_tile else 1.0
    return {
        "m": m_new,
        "l": l_new,
        "old_output_scale": alpha * normalization,
        "tile_weight_scale": beta * normalization,
    }


def _read_f32(path, expected_count: int, offset_bytes: int = 0):
    data = Path(path).read_bytes()
    expected_bytes = expected_count * 4
    if offset_bytes < 0 or len(data) < offset_bytes + expected_bytes:
        raise ValueError(
            f"{path}: expected {expected_bytes} bytes at offset {offset_bytes}, "
            f"found {len(data)} total bytes"
        )
    if offset_bytes == 0 and len(data) != expected_bytes:
        raise ValueError(f"{path}: expected {expected_bytes} bytes, found {len(data)}")
    payload = data[offset_bytes:offset_bytes + expected_bytes]
    return list(struct.unpack(f"<{expected_count}f", payload))


def _write_f32(path, values):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(struct.pack(f"<{len(values)}f", *values))


def generate_case(
    queries: int, keys: int, head_dim: int, q_path, k_path, storage_keys=None,
    v_path=None, extreme_logits=False, heads: int = 1, kv_heads=None,
):
    if heads <= 0:
        raise ValueError("heads must be positive")
    kv_heads = heads if kv_heads is None else kv_heads
    if kv_heads <= 0 or heads % kv_heads != 0:
        raise ValueError("query heads must be divisible by K/V heads")
    storage_keys = keys if storage_keys is None else storage_keys
    if storage_keys < keys:
        raise ValueError("storage_keys cannot be smaller than logical keys")
    if extreme_logits:
        if (queries, keys, head_dim) != (64, 64, 64):
            raise ValueError("extreme_logits requires queries=keys=head_dim=64")
        q = [1.0 if dim == 0 else 0.0
             for _head in range(heads)
             for _query in range(queries) for dim in range(head_dim)]
        logical_k = [(-800.0 if key < 32 else 800.0) if dim == 0 else 0.0
                     for _head in range(kv_heads)
                     for key in range(keys) for dim in range(head_dim)]
    else:
        q = [
            (((head * 19 + query * 17 + dim * 5) % 29) - 14) / 32.0
            for head in range(heads)
            for query in range(queries)
            for dim in range(head_dim)
        ]
        logical_k = [
            (((head * 23 + key * 11 + dim * 7 + 3) % 31) - 15) / 32.0
            for head in range(kv_heads)
            for key in range(keys)
            for dim in range(head_dim)
        ]
    if storage_keys == keys:
        k = logical_k
    else:
        head_values = keys * head_dim
        head_padding = (storage_keys - keys) * head_dim
        k = []
        for head in range(kv_heads):
            begin = head * head_values
            k.extend(logical_k[begin:begin + head_values])
            k.extend([0.0] * head_padding)
    _write_f32(q_path, q)
    _write_f32(k_path, k)
    if v_path is not None:
        v = [
            (((head * 29 + key * 13 + dim * 3 + 5) % 37) - 18) / 32.0
            for head in range(kv_heads)
            for key in range(keys)
            for dim in range(head_dim)
        ]
        _write_f32(v_path, v)


def verify_qk(
    q_path,
    k_path,
    output_path,
    queries: int,
    keys: int,
    head_dim: int,
    atol: float = 1.0e-4,
    rtol: float = 1.0e-4,
    storage_keys=None,
):
    storage_keys = keys if storage_keys is None else storage_keys
    if storage_keys < keys:
        raise ValueError("storage_keys cannot be smaller than logical keys")
    q = _read_f32(q_path, queries * head_dim)
    k_data = Path(k_path).read_bytes()
    if len(k_data) < keys * head_dim * 4 or len(k_data) % 4 != 0:
        raise ValueError(f"{k_path}: invalid native K storage size {len(k_data)}")
    k = list(struct.unpack(f"<{len(k_data) // 4}f", k_data))[: keys * head_dim]
    stored_output = _read_f32(output_path, queries * storage_keys)
    actual = [
        stored_output[query * storage_keys + key]
        for query in range(queries)
        for key in range(keys)
    ]
    expected = compute_qk(q, k, queries, keys, head_dim)

    mismatches = 0
    max_abs_error = 0.0
    first_mismatch = None
    for index, (got, want) in enumerate(zip(actual, expected)):
        error = abs(got - want)
        max_abs_error = max(max_abs_error, error)
        if not math.isclose(got, want, rel_tol=rtol, abs_tol=atol):
            mismatches += 1
            if first_mismatch is None:
                first_mismatch = {
                    "query": index // keys,
                    "key": index % keys,
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
        "shape": [queries, keys],
        "head_dim": head_dim,
        "k_layout": "native_key_major_[keys,head_dim]",
        "transpose_b": 1,
    }


def verify_attention(
    q_path, k_path, v_path, output_path, queries: int, keys: int,
    head_dim: int, causal: bool, atol: float = 2.0e-4, rtol: float = 2.0e-4,
    output_offset: int = 0, fused: bool = False, extreme_logits: bool = False,
):
    q = _read_f32(q_path, queries * head_dim)
    k = _read_f32(k_path, keys * head_dim)
    v = _read_f32(v_path, keys * head_dim)
    actual = _read_f32(output_path, queries * head_dim, output_offset)
    expected = compute_attention(q, k, v, queries, keys, head_dim, causal)
    mismatches = 0
    max_abs_error = 0.0
    first_mismatch = None
    for index, (got, want) in enumerate(zip(actual, expected)):
        error = abs(got - want)
        max_abs_error = max(max_abs_error, error)
        if not math.isclose(got, want, rel_tol=rtol, abs_tol=atol):
            mismatches += 1
            if first_mismatch is None:
                first_mismatch = {
                    "query": index // head_dim,
                    "dim": index % head_dim,
                    "actual": got,
                    "expected": want,
                    "abs_error": error,
                }
    traffic = attention_logical_hbm_traffic(queries, keys, head_dim, fused)
    return {
        "status": "PASS" if mismatches == 0 else "FAIL",
        "checked": len(expected),
        "mismatches": mismatches,
        "max_abs_error": max_abs_error,
        "first_mismatch": first_mismatch,
        "shape": {"queries": queries, "keys": keys, "head_dim": head_dim},
        "causal": causal,
        "fused": fused,
        "input_profile": "extreme_tile_jump" if extreme_logits else "default",
        "output_offset_bytes": output_offset,
        "scale": 1.0 / math.sqrt(head_dim),
        "logical_hbm_traffic": traffic,
    }


def _positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--query-length", "--queries", dest="query_length",
                        type=_positive_int, required=True)
    common.add_argument("--kv-length", "--keys", dest="kv_length",
                        type=_positive_int, required=True)
    common.add_argument("--heads", type=_positive_int)
    common.add_argument("--num-query-heads", "--query-heads",
                        dest="num_query_heads", type=_positive_int)
    common.add_argument("--num-kv-heads", "--kv-heads",
                        dest="num_kv_heads", type=_positive_int)
    common.add_argument("--head-dim", type=_positive_int, required=True)
    common.add_argument("--q-file", required=True)
    common.add_argument("--k-file", required=True)
    common.add_argument("--storage-keys", type=_positive_int)

    generate = subparsers.add_parser("generate", parents=[common])
    generate.add_argument("--manifest")
    generate.add_argument("--v-file")
    generate.add_argument("--extreme-logits", action="store_true")

    verify = subparsers.add_parser("verify", parents=[common])
    verify.add_argument("--output-file", required=True)
    verify.add_argument("--result-json")
    verify.add_argument("--atol", type=float, default=1.0e-4)
    verify.add_argument("--rtol", type=float, default=1.0e-4)

    verify_attention_parser = subparsers.add_parser("verify-attention", parents=[common])
    verify_attention_parser.add_argument("--v-file", required=True)
    verify_attention_parser.add_argument("--output-file", required=True)
    verify_attention_parser.add_argument("--causal", type=int, choices=[0, 1], default=0)
    verify_attention_parser.add_argument("--result-json")
    verify_attention_parser.add_argument("--atol", type=float, default=2.0e-4)
    verify_attention_parser.add_argument("--rtol", type=float, default=2.0e-4)
    verify_attention_parser.add_argument("--output-offset", type=int, default=0)
    verify_attention_parser.add_argument("--fused", action="store_true")
    verify_attention_parser.add_argument("--extreme-logits", action="store_true")
    return parser


def main():
    args = _build_parser().parse_args()
    if args.head_dim not in (64, 128):
        raise SystemExit("head_dim must be 64 or 128 for the Phase A Attention path")

    query_heads = args.num_query_heads or args.heads or 1
    kv_heads = args.num_kv_heads or (args.heads if args.heads is not None else query_heads)
    if query_heads % kv_heads != 0:
        raise SystemExit("query heads must be divisible by K/V heads")

    if args.command == "generate":
        storage_keys = args.storage_keys or args.kv_length
        generate_case(
            args.query_length,
            args.kv_length,
            args.head_dim,
            args.q_file,
            args.k_file,
            storage_keys,
            args.v_file,
            args.extreme_logits,
            query_heads,
            kv_heads,
        )
        result = {
            "queries": args.query_length,
            "keys": args.kv_length,
            "query_heads": query_heads,
            "kv_heads": kv_heads,
            "gqa_group_size": query_heads // kv_heads,
            "storage_keys": storage_keys,
            "head_dim": args.head_dim,
            "q_layout": "head_major_[query_heads,queries,head_dim]",
            "k_layout": "head_major_[kv_heads,keys,head_dim]",
            "transpose_b": 1,
            "input_profile": "extreme_tile_jump" if args.extreme_logits else "default",
            "q_file": str(Path(args.q_file).resolve()),
            "k_file": str(Path(args.k_file).resolve()),
        }
        if args.v_file:
            result["v_file"] = str(Path(args.v_file).resolve())
        if args.manifest:
            manifest = Path(args.manifest)
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    elif args.command == "verify":
        result = verify_qk(
            args.q_file,
            args.k_file,
            args.output_file,
            args.query_length,
            args.kv_length,
            args.head_dim,
            args.atol,
            args.rtol,
            args.storage_keys,
        )
        if args.result_json:
            result_path = Path(args.result_json)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    else:
        if args.storage_keys not in (None, args.kv_length):
            raise SystemExit("verify-attention requires unpadded native K storage")
        result = verify_attention(
            args.q_file, args.k_file, args.v_file, args.output_file,
            args.query_length, args.kv_length, args.head_dim, bool(args.causal),
            args.atol, args.rtol, args.output_offset, args.fused, args.extreme_logits,
        )
        if args.result_json:
            result_path = Path(args.result_json)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")

    print(json.dumps(result, indent=2))
    if result.get("status") == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
