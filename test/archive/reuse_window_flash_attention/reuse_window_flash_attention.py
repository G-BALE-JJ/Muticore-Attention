#!/usr/bin/env python3
"""Functional GQA FlashAttention and reference-WCP reuse-window contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


FP32_BYTES = 4
TILE = 64
ARRAYS_PER_WORKER = 64
ARRAY_OUTPUTS = 64
MACS_PER_ARRAY_OUTPUT_PER_CYCLE = 1
TOTAL_WORKERS = 16


def positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def validate(args: argparse.Namespace) -> None:
    if args.num_query_heads % args.num_kv_heads:
        raise ValueError("Hq must be divisible by Hkv")
    group = args.num_query_heads // args.num_kv_heads
    if group not in (1, 2, 4):
        raise ValueError("GQA group size Hq/Hkv must be 1, 2, or 4")
    for name in ("query_length", "kv_length", "head_dim"):
        if getattr(args, name) % TILE:
            raise ValueError(f"{name.replace('_', '-')} must be divisible by {TILE}")
    if args.window_m_tiles != 2 or args.window_n_tiles != 4:
        raise ValueError("this baseline fixes the selected reuse window at 2x4")
    if args.query_length % (args.window_m_tiles * TILE):
        raise ValueError("Sq must divide evenly over the reuse window M dimension")
    if args.kv_length % (args.window_n_tiles * TILE):
        raise ValueError("Skv must divide evenly over the reuse window N dimension")


def make_inputs(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    q = rng.normal(0.0, 0.2, (args.num_query_heads, args.query_length, args.head_dim)).astype(np.float32)
    k = rng.normal(0.0, 0.2, (args.num_kv_heads, args.kv_length, args.head_dim)).astype(np.float32)
    v = rng.normal(0.0, 0.2, (args.num_kv_heads, args.kv_length, args.head_dim)).astype(np.float32)
    return q, k, v


def online_window_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    row_window: int,
    key_window: int,
) -> np.ndarray:
    output = np.empty((q.shape[0], v.shape[1]), dtype=np.float32)
    scale = np.float32(1.0 / math.sqrt(q.shape[1]))
    for row_begin in range(0, q.shape[0], row_window):
        q_window = q[row_begin : row_begin + row_window]
        summaries: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for key_begin in range(0, k.shape[0], key_window):
            k_window = k[key_begin : key_begin + key_window]
            v_window = v[key_begin : key_begin + key_window]
            scores = (q_window @ k_window.T) * scale
            window_max = np.max(scores, axis=1)
            probabilities = np.exp(scores - window_max[:, None]).astype(np.float32)
            summaries.append((
                window_max,
                np.sum(probabilities, axis=1),
                probabilities @ v_window,
            ))
        while len(summaries) > 1:
            merged: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            for index in range(0, len(summaries), 2):
                if index + 1 == len(summaries):
                    merged.append(summaries[index])
                    continue
                left_max, left_sum, left_out = summaries[index]
                right_max, right_sum, right_out = summaries[index + 1]
                combined_max = np.maximum(left_max, right_max)
                left_scale = np.exp(left_max - combined_max).astype(np.float32)
                right_scale = np.exp(right_max - combined_max).astype(np.float32)
                merged.append((
                    combined_max,
                    left_sum * left_scale + right_sum * right_scale,
                    left_out * left_scale[:, None] + right_out * right_scale[:, None],
                ))
            summaries = merged
        _, normalizer, numerator = summaries[0]
        output[row_begin : row_begin + q_window.shape[0]] = numerator / normalizer[:, None]
    return output


def full_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    scores = (q @ k.T) * np.float32(1.0 / math.sqrt(q.shape[1]))
    scores -= np.max(scores, axis=1, keepdims=True)
    probabilities = np.exp(scores).astype(np.float32)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    return probabilities @ v


def build_schedule(args: argparse.Namespace) -> list[dict[str, Any]]:
    schedule: list[dict[str, Any]] = []
    row_window = args.window_m_tiles * TILE
    key_window = args.window_n_tiles * TILE
    group_size = args.num_query_heads // args.num_kv_heads
    for query_head in range(args.num_query_heads):
        kv_head = query_head // group_size
        query_within_group = query_head % group_size
        gemm_submission = f"qk_h{query_head}_kv{kv_head}"
        window_in_gemm = 0
        for row_begin in range(0, args.query_length, row_window):
            for key_begin in range(0, args.kv_length, key_window):
                window_id = len(schedule) // 2
                worker = window_in_gemm % TOTAL_WORKERS
                schedule.append({
                    "window_id": window_id,
                    "op": "qk",
                    "gemm_submission": gemm_submission,
                    "logical_gemm": {
                        "m": args.query_length,
                        "n": args.kv_length,
                        "k": args.head_dim,
                    },
                    "query_head": query_head,
                    "kv_head": kv_head,
                    "query_order_within_kv_group": query_within_group,
                    "window_index_within_gemm": window_in_gemm,
                    "worker_core": worker,
                    "row_begin": row_begin,
                    "key_begin": key_begin,
                    "gemm": {"m": row_window, "n": key_window, "k": args.head_dim},
                    "output_tile_window": {"m_tiles": 2, "n_tiles": 4},
                    "wcp": {
                        "a_reuse_n_tiles": 4,
                        "b_reuse_m_tiles": 2,
                        "window_k_tiles": args.head_dim // TILE,
                    },
                    "score_bytes": row_window * key_window * FP32_BYTES,
                    "score_destination": "bounded_fusion_c_buffer",
                })
                schedule.append({
                    "window_id": window_id,
                    "op": "pv",
                    "parent_qk_gemm_submission": gemm_submission,
                    "query_head": query_head,
                    "kv_head": kv_head,
                    "query_order_within_kv_group": query_within_group,
                    "window_index_within_gemm": window_in_gemm,
                    "worker_core": worker,
                    "row_begin": row_begin,
                    "key_begin": key_begin,
                    "gemm": {"m": row_window, "n": args.head_dim, "k": key_window},
                    "output_tile_window": {"m_tiles": 2, "n_tiles": args.head_dim // TILE},
                    "wcp": {
                        "a_reuse_n_tiles": args.head_dim // TILE,
                        "b_reuse_m_tiles": 2,
                        "window_k_tiles": key_window // TILE,
                    },
                    "probability_source": "bounded_online_softmax_window",
                })
                window_in_gemm += 1
    return schedule


def traffic_and_bounds(args: argparse.Namespace, schedule: list[dict[str, Any]]) -> dict[str, Any]:
    q_bytes = args.num_query_heads * args.query_length * args.head_dim * FP32_BYTES
    k_bytes = args.num_kv_heads * args.kv_length * args.head_dim * FP32_BYTES
    v_bytes = k_bytes
    o_bytes = q_bytes
    hbm_bytes = q_bytes + k_bytes + v_bytes + o_bytes
    score_window_bytes = args.window_m_tiles * args.window_n_tiles * TILE * TILE * FP32_BYTES
    qk_k_tiles = args.head_dim // TILE
    qk_slots = (args.prefetch_windows + 1) * qk_k_tiles * max(
        args.window_m_tiles, args.window_n_tiles
    )
    pv_n_tiles = args.head_dim // TILE
    pv_k_tiles = args.window_n_tiles
    pv_slots = (args.prefetch_windows + 1) * pv_k_tiles * max(
        args.window_m_tiles, pv_n_tiles
    )
    pv_partial_c_bytes = args.window_m_tiles * pv_n_tiles * TILE * TILE * FP32_BYTES

    total_macs = 2 * args.num_query_heads * args.query_length * args.kv_length * args.head_dim
    active_workers = TOTAL_WORKERS
    worker_macs_per_cycle = ARRAYS_PER_WORKER * ARRAY_OUTPUTS * MACS_PER_ARRAY_OUTPUT_PER_CYCLE
    compute_floor_cycles = math.ceil(total_macs / (active_workers * worker_macs_per_cycle))
    hbm_bytes_per_cycle = args.hbm_nodes * args.hbm_node_bytes_per_cycle
    hbm_floor_cycles = math.ceil(hbm_bytes / hbm_bytes_per_cycle)

    group_hbm_bytes = (
        (args.num_query_heads // args.num_kv_heads) * args.query_length * args.head_dim * FP32_BYTES * 2
        + args.kv_length * args.head_dim * FP32_BYTES * 2
    )
    group_macs = (
        2 * (args.num_query_heads // args.num_kv_heads) * args.query_length * args.kv_length * args.head_dim
    )
    group_compute_floor = math.ceil(group_macs / (active_workers * worker_macs_per_cycle))
    group_required_bpc = group_hbm_bytes / group_compute_floor

    return {
        "physical_hbm": {
            "q_bytes": q_bytes,
            "k_bytes": k_bytes,
            "v_bytes": v_bytes,
            "o_bytes": o_bytes,
            "score_probability_bytes": 0,
            "total_bytes": hbm_bytes,
            "kv_counting": "once per KV head; GQA receivers use multicast",
        },
        "bounded_storage": {
            "qk_score_window_bytes_per_worker": score_window_bytes,
            "pv_partial_c_bytes_per_worker": pv_partial_c_bytes,
            "configured_c_buffer_bytes": args.c_buffer_bytes,
            "fits": max(score_window_bytes, pv_partial_c_bytes) <= args.c_buffer_bytes,
        },
        "wcp_capacity": {
            "prefetch_windows": args.prefetch_windows,
            "qk_required_local_slots": qk_slots,
            "pv_required_local_slots": pv_slots,
            "configured_local_slots": args.local_slots,
            "fits": max(qk_slots, pv_slots) <= args.local_slots,
        },
        "lower_bounds": {
            "total_macs": total_macs,
            "active_workers_per_gqa_group": active_workers,
            "macs_per_worker_cycle": worker_macs_per_cycle,
            "compute_floor_cycles": compute_floor_cycles,
            "hbm_floor_cycles": hbm_floor_cycles,
            "combined_floor_cycles": max(compute_floor_cycles, hbm_floor_cycles),
        },
        "supply_proof_per_gqa_group": {
            "physical_hbm_bytes": group_hbm_bytes,
            "compute_floor_cycles": group_compute_floor,
            "required_bytes_per_cycle": group_required_bpc,
            "available_bytes_per_cycle": hbm_bytes_per_cycle,
            "headroom_ratio": hbm_bytes_per_cycle / group_required_bpc,
            "supply_keeps_up": group_required_bpc <= hbm_bytes_per_cycle,
        },
        "window_counts": {
            "qk": sum(item["op"] == "qk" for item in schedule),
            "pv": sum(item["op"] == "pv" for item in schedule),
            "qk_gemm_submissions": len({
                item["gemm_submission"] for item in schedule if item["op"] == "qk"
            }),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-length", type=positive, default=1024)
    parser.add_argument("--kv-length", type=positive, default=1024)
    parser.add_argument("--num-query-heads", type=positive, default=4)
    parser.add_argument("--num-kv-heads", type=positive, default=2)
    parser.add_argument("--head-dim", type=positive, default=128)
    parser.add_argument("--window-m-tiles", type=positive, default=2)
    parser.add_argument("--window-n-tiles", type=positive, default=4)
    parser.add_argument("--hbm-nodes", type=positive, default=2)
    parser.add_argument("--hbm-node-bytes-per-cycle", type=positive, default=320)
    parser.add_argument("--prefetch-windows", type=positive, default=2)
    parser.add_argument("--local-slots", type=positive, default=24)
    parser.add_argument("--c-buffer-bytes", type=positive, default=1024 * 1024)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--artifact-root", type=Path, default=Path(__file__).parent / "artifacts" / "latest")
    parser.add_argument("--no-numerical", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate(args)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    schedule = build_schedule(args)
    traffic = traffic_and_bounds(args, schedule)
    if not traffic["bounded_storage"]["fits"] or not traffic["wcp_capacity"]["fits"]:
        raise RuntimeError("reuse window exceeds the configured bounded resources")

    numerical: dict[str, Any] = {"status": "SKIPPED"}
    if not args.no_numerical:
        q, k, v = make_inputs(args)
        output = np.empty((args.query_length, args.num_query_heads, args.head_dim), dtype=np.float32)
        max_abs_error = 0.0
        group_size = args.num_query_heads // args.num_kv_heads
        for query_head in range(args.num_query_heads):
            kv_head = query_head // group_size
            actual = online_window_attention(
                q[query_head], k[kv_head], v[kv_head],
                args.window_m_tiles * TILE, args.window_n_tiles * TILE,
            )
            expected = full_attention(q[query_head], k[kv_head], v[kv_head])
            max_abs_error = max(max_abs_error, float(np.max(np.abs(actual - expected))))
            output[:, query_head, :] = actual
        output.tofile(args.artifact_root / "output.bin")
        numerical = {
            "status": "PASS" if max_abs_error <= 2e-5 else "FAIL",
            "max_abs_error": max_abs_error,
            "tolerance": 2e-5,
            "output_layout": [args.query_length, args.num_query_heads, args.head_dim],
        }

    result = {
        "schema_version": 1,
        "baseline": "reuse_window_flash_attention",
        "status": "PASS" if numerical["status"] in ("PASS", "SKIPPED") else "FAIL",
        "shape": {
            "Hq": args.num_query_heads, "Hkv": args.num_kv_heads,
            "Sq": args.query_length, "Skv": args.kv_length, "Dh": args.head_dim,
        },
        "gqa": {
            "group_size": args.num_query_heads // args.num_kv_heads,
            "query_to_kv_head": [
                h // (args.num_query_heads // args.num_kv_heads)
                for h in range(args.num_query_heads)
            ],
            "execution_policy": (
                "one complete Query-head GEMM at a time; Query heads sharing a KV head "
                "execute sequentially and retain the shared K/V"
            ),
        },
        "reuse_windows": {
            "qk": {
                "m_tiles": args.window_m_tiles,
                "n_tiles": args.window_n_tiles,
                "gemm": [
                    args.window_m_tiles * TILE,
                    args.window_n_tiles * TILE,
                    args.head_dim,
                ],
            },
            "pv": {
                "m_tiles": args.window_m_tiles,
                "n_tiles": args.head_dim // TILE,
                "gemm": [
                    args.window_m_tiles * TILE,
                    args.head_dim,
                    args.window_n_tiles * TILE,
                ],
            },
        },
        "engine_contract": {
            "request_scheduler_enable": 1,
            "submission_policy": "one full Query-head GEMM at a time",
            "qk_logical_gemm": [args.query_length, args.kv_length, args.head_dim],
            "qk_windows_distributed_across_workers": TOTAL_WORKERS,
            "softmax_window_merge": "associative_m_l_O_reduction",
            "qk_a_reuse_n_tiles": 4,
            "qk_b_reuse_m_tiles": 2,
            "qk_window_k_tiles": args.head_dim // TILE,
            "pv_a_reuse_n_tiles": args.head_dim // TILE,
            "pv_b_reuse_m_tiles": 2,
            "pv_window_k_tiles": args.window_n_tiles,
            "output_mode": "fusion",
            "sst_integration_status": "contract_ready_not_wired_to_attention_rocc",
        },
        "traffic": traffic,
        "numerical": numerical,
    }
    (args.artifact_root / "schedule.json").write_text(json.dumps(schedule, indent=2) + "\n")
    (args.artifact_root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
