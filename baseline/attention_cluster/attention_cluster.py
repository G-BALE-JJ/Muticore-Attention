#!/usr/bin/env python3
"""Functional and scheduled partitioned QK/SFU + PV Attention cluster."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


TILE = 64
ELEM_BYTES = 4
ARRAY_MACS_PER_CYCLE = 64 * 64


def positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def validate(args: argparse.Namespace) -> None:
    if args.num_query_heads % args.num_kv_heads:
        raise ValueError("Hq must be divisible by Hkv")
    if args.num_query_heads // args.num_kv_heads not in (1, 2, 4):
        raise ValueError("Hq/Hkv must be 1, 2, or 4")
    if args.query_length % 128 or args.kv_length % 256:
        raise ValueError("Sq must be divisible by 128 and Skv by 256")
    if args.head_dim % TILE:
        raise ValueError("Dh must be divisible by 64")
    if args.head_dim != 128:
        raise ValueError("the first cluster bridge fixes Dh=128")
    if args.qk_workers_per_manager not in (1, 2):
        raise ValueError("qk-workers-per-manager must be 1 or 2")


def make_inputs(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    q = rng.normal(0, 0.2, (
        args.num_query_heads, args.query_length, args.head_dim)).astype(np.float32)
    k = rng.normal(0, 0.2, (
        args.num_kv_heads, args.kv_length, args.head_dim)).astype(np.float32)
    v = rng.normal(0, 0.2, (
        args.num_kv_heads, args.kv_length, args.head_dim)).astype(np.float32)
    return q, k, v


def cluster_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Online-softmax reference; never materializes a complete score matrix."""
    hq, sq, dh = q.shape
    hkv, skv, _ = k.shape
    group_size = hq // hkv
    output = np.zeros((hq, sq, dh), dtype=np.float32)
    scale = np.float32(1.0 / math.sqrt(dh))
    for q_head in range(hq):
        kv_head = q_head // group_size
        for row_begin in range(0, sq, TILE):
            rows = q[q_head, row_begin:row_begin + TILE]
            running_max = np.full(TILE, -np.inf, dtype=np.float32)
            running_sum = np.zeros(TILE, dtype=np.float32)
            running_out = np.zeros((TILE, dh), dtype=np.float32)
            for key_begin in range(0, skv, 256):
                keys = k[kv_head, key_begin:key_begin + 256]
                values = v[kv_head, key_begin:key_begin + 256]
                scores = (rows @ keys.T) * scale
                tile_max = np.max(scores, axis=1)
                merged_max = np.maximum(running_max, tile_max)
                old_scale = np.exp(running_max - merged_max).astype(np.float32)
                probabilities = np.exp(scores - merged_max[:, None]).astype(np.float32)
                running_sum = running_sum * old_scale + np.sum(probabilities, axis=1)
                running_out = running_out * old_scale[:, None] + probabilities @ values
                running_max = merged_max
            output[q_head, row_begin:row_begin + TILE] = (
                running_out / running_sum[:, None]
            )
    return output


def full_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    hq, sq, dh = q.shape
    group_size = hq // k.shape[0]
    output = np.empty_like(q)
    scale = np.float32(1.0 / math.sqrt(dh))
    for head in range(hq):
        kv_head = head // group_size
        score = (q[head] @ k[kv_head].T) * scale
        score -= np.max(score, axis=1, keepdims=True)
        probability = np.exp(score).astype(np.float32)
        probability /= np.sum(probability, axis=1, keepdims=True)
        output[head] = probability @ v[kv_head]
    return output


def _duration(m: int, n: int, k: int) -> int:
    return math.ceil(m * n * k / ARRAY_MACS_PER_CYCLE)


def build_schedule(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a dependency-respecting cluster timeline in accelerator cycles."""
    qk_workers_per_manager = getattr(args, "qk_workers_per_manager", 1)
    qk_workers = 4 * qk_workers_per_manager
    pv_workers = 16 - qk_workers
    group_size = args.num_query_heads // args.num_kv_heads
    row_blocks_per_group = group_size * args.query_length // TILE
    row_windows_per_group = row_blocks_per_group // 2
    key_windows = args.kv_length // 256
    qk_ready = [0] * qk_workers
    sfu_ready = [0] * qk_workers
    score_slot_ready = [[0, 0] for _ in range(qk_workers)]
    pv_ready = [0] * pv_workers
    previous_pv_done: dict[tuple[int, int], int] = defaultdict(int)
    events: list[dict[str, Any]] = []
    pv_inputs: dict[tuple[int, int, int], list[dict[str, int]]] = defaultdict(list)

    q_bytes_per_worker = math.ceil(
        group_size * args.query_length / qk_workers) * args.head_dim * ELEM_BYTES
    q_prefetch_cycles = math.ceil(q_bytes_per_worker / args.local_bytes_per_cycle)
    k_window_bytes = 256 * args.head_dim * ELEM_BYTES
    k_broadcast_cycles = math.ceil(k_window_bytes / args.broadcast_bytes_per_cycle)

    for kv_head in range(args.num_kv_heads):
        group_base = kv_head * group_size * args.query_length
        group_q_prefetch_done = [
            max(qk_ready[worker], min(qk_ready)) + q_prefetch_cycles
            for worker in range(qk_workers)
        ]
        for worker, end in enumerate(group_q_prefetch_done):
            events.append({
                "stage": "q_prefetch", "kv_head": kv_head,
                "worker": worker, "worker_core": 4 + worker,
                "start": end - q_prefetch_cycles, "end": end,
                "bytes": q_bytes_per_worker, "buffer": kv_head & 1,
            })

        k_buffer_ready = [0] * key_windows
        for key_window in range(key_windows):
            previous = k_buffer_ready[key_window - 1] if key_window else min(group_q_prefetch_done)
            start = max(previous, min(group_q_prefetch_done))
            k_buffer_ready[key_window] = start + k_broadcast_cycles
            events.append({
                "stage": "k_prefetch_broadcast", "kv_head": kv_head,
                "key_window": key_window, "start": start,
                "end": k_buffer_ready[key_window], "bytes": k_window_bytes,
                "fanout": qk_workers, "buffer": key_window & 1,
            })

        # Key-major order retains one K window while all row windows consume it.
        for key_window in range(key_windows):
            for row_window in range(row_windows_per_group):
                qk_worker = row_window % qk_workers
                qk_cycles = _duration(128, 256, args.head_dim)
                score_slot = min(
                    range(2), key=lambda slot: score_slot_ready[qk_worker][slot]
                )
                # A slot is needed only when QK completes, so computation may
                # overlap the tail of the preceding SFU consumer.
                start = max(
                    qk_ready[qk_worker], group_q_prefetch_done[qk_worker],
                    k_buffer_ready[key_window],
                    score_slot_ready[qk_worker][score_slot] - qk_cycles,
                )
                qk_end = start + qk_cycles
                qk_ready[qk_worker] = qk_end
                global_row_block = group_base // TILE + row_window * 2
                events.append({
                    "stage": "qk_gemm", "kv_head": kv_head,
                    "key_window": key_window, "row_window": row_window,
                    "worker": qk_worker, "worker_core": 4 + qk_worker,
                    "start": start, "end": qk_end,
                    "gemm": [128, 256, args.head_dim],
                    "reuse_window": [2, 4], "q_reuse": 4, "k_reuse": 2,
                    "score_fifo_slot": score_slot,
                })

                # One SFU follows each QK worker, so QK and SFU overlap across windows.
                sfu_start = max(qk_end, sfu_ready[qk_worker])
                sfu_cycles = args.sfu_window_cycles
                sfu_end = sfu_start + sfu_cycles
                sfu_ready[qk_worker] = sfu_end
                score_slot_ready[qk_worker][score_slot] = sfu_end
                events.append({
                    "stage": "online_softmax", "kv_head": kv_head,
                    "key_window": key_window, "row_window": row_window,
                    "worker": qk_worker, "worker_core": 4 + qk_worker,
                    "start": sfu_start, "end": sfu_end,
                    "shape": [128, 256], "score_fifo_slot": score_slot,
                })
                for half in range(2):
                    row_block = global_row_block + half
                    pv_worker = row_block % pv_workers
                    pv_inputs[(kv_head, key_window, pv_worker)].append({
                        "row_block": row_block,
                        "ready": sfu_end + math.ceil(
                            64 * 256 * ELEM_BYTES / args.cluster_link_bytes_per_cycle),
                        "qk_worker": qk_worker,
                    })

        # Pair P row blocks on a PV worker: the generic GEMM sees a 2x2 window.
        for key_window in range(key_windows):
            v_bytes = 256 * args.head_dim * ELEM_BYTES
            v_start = min(pv_ready)
            v_end = v_start + math.ceil(v_bytes / args.broadcast_bytes_per_cycle)
            events.append({
                "stage": "v_prefetch_broadcast", "kv_head": kv_head,
                "key_window": key_window, "start": v_start, "end": v_end,
                "bytes": v_bytes, "fanout": pv_workers,
                "buffer": key_window & 1,
            })
            for worker in range(pv_workers):
                inputs = pv_inputs.get((kv_head, key_window, worker), [])
                inputs.sort(key=lambda item: item["row_block"])
                for offset in range(0, len(inputs), 2):
                    batch = inputs[offset:offset + 2]
                    rows = TILE * len(batch)
                    dependency = max(
                        previous_pv_done[(worker, item["row_block"])]
                        for item in batch
                    )
                    start = max(pv_ready[worker], v_end, dependency,
                                max(item["ready"] for item in batch))
                    end = start + _duration(rows, args.head_dim, 256)
                    pv_ready[worker] = end
                    for item in batch:
                        previous_pv_done[(worker, item["row_block"])] = end
                    events.append({
                        "stage": "pv_gemm", "kv_head": kv_head,
                        "key_window": key_window, "worker": worker,
                        "worker_core": 4 + qk_workers + worker,
                        "start": start, "end": end,
                        "row_blocks": [item["row_block"] for item in batch],
                        "gemm": [rows, args.head_dim, 256],
                        "reuse_window": [len(batch), 2],
                        "p_reuse": 2, "v_reuse": len(batch),
                        "p_fifo_slot": key_window & 1,
                    })

    end_cycle = max(max(qk_ready), max(sfu_ready), max(pv_ready))
    hbm = {
        "q_bytes": args.num_query_heads * args.query_length * args.head_dim * ELEM_BYTES,
        "k_bytes": args.num_kv_heads * args.kv_length * args.head_dim * ELEM_BYTES,
        "v_bytes": args.num_kv_heads * args.kv_length * args.head_dim * ELEM_BYTES,
        "o_bytes": args.num_query_heads * args.query_length * args.head_dim * ELEM_BYTES,
        "score_probability_bytes": 0,
    }
    hbm["total_bytes"] = sum(hbm[key] for key in ("q_bytes", "k_bytes", "v_bytes", "o_bytes"))
    qk_macs = args.num_query_heads * args.query_length * args.kv_length * args.head_dim
    pv_macs = qk_macs
    softmax_elements = args.num_query_heads * args.query_length * args.kv_length
    qk_floor = math.ceil(qk_macs / (qk_workers * ARRAY_MACS_PER_CYCLE))
    pv_floor = math.ceil(pv_macs / (pv_workers * ARRAY_MACS_PER_CYCLE))
    softmax_exp_floor = math.ceil(softmax_elements / (qk_workers * 16))
    ideal_cluster_floor = max(qk_floor, pv_floor, softmax_exp_floor)
    summary = {
        "status": "PASS",
        "shape": {
            "Hq": args.num_query_heads, "Hkv": args.num_kv_heads,
            "Sq": args.query_length, "Skv": args.kv_length, "Dh": args.head_dim,
        },
        "clusters": {
            "qk_softmax_workers": list(range(4, 4 + qk_workers)),
            "pv_workers": list(range(4 + qk_workers, 20)),
        },
        "end_to_end_modeled_cycles": end_cycle,
        "stage_last_cycle": {
            "qk": max(qk_ready), "softmax": max(sfu_ready), "pv": max(pv_ready),
        },
        "compute_floor_cycles": {
            "qk": qk_floor,
            "softmax_exp_only": softmax_exp_floor,
            "pv": pv_floor,
            "perfect_pipeline": ideal_cluster_floor,
        },
        "modeled_efficiency_vs_perfect_pipeline": ideal_cluster_floor / end_cycle,
        "hbm": hbm,
        "hbm_floor_cycles": math.ceil(hbm["total_bytes"] / args.hbm_bytes_per_cycle),
        "average_hbm_bytes_per_cycle": hbm["total_bytes"] / end_cycle,
        "hbm_supply_keeps_up": (
            hbm["total_bytes"] / end_cycle <= args.hbm_bytes_per_cycle
        ),
        "internal_transport": {
            "probability_bytes_qk_to_pv": (
                args.num_query_heads * args.query_length * args.kv_length * ELEM_BYTES
            ),
            "k_broadcast_fanout": qk_workers,
            "v_broadcast_fanout": pv_workers,
        },
        "reuse": {
            "qk": "2x4 output window: Q reused 4, K reused 2",
            "pv": "up to 2x2 output window: P reused 2, V reused by 2 row tiles",
        },
        "prefetch": {
            "q": "two group buffers per QK worker",
            "k": f"two 256xDh broadcast buffers shared by {qk_workers} QK workers",
            "v": f"two 256xDh broadcast buffers replicated to {pv_workers} PV workers",
            "score_fifo_depth": 2,
            "p_fifo_depth": 2,
        },
        "event_counts": dict(
            sorted((stage, sum(event["stage"] == stage for event in events))
                   for stage in {event["stage"] for event in events})
        ),
    }
    return events, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--query-length", type=positive, default=1024)
    parser.add_argument("--kv-length", type=positive, default=1024)
    parser.add_argument("--num-query-heads", type=positive, default=4)
    parser.add_argument("--num-kv-heads", type=positive, default=2)
    parser.add_argument("--head-dim", type=positive, default=128)
    parser.add_argument("--qk-workers-per-manager", type=positive, choices=(1, 2), default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--hbm-bytes-per-cycle", type=positive, default=1280)
    parser.add_argument("--local-bytes-per-cycle", type=positive, default=256)
    parser.add_argument("--broadcast-bytes-per-cycle", type=positive, default=256)
    parser.add_argument("--cluster-link-bytes-per-cycle", type=positive, default=256)
    parser.add_argument("--sfu-window-cycles", type=positive, default=3064)
    parser.add_argument("--no-numerical", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate(args)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    schedule, result = build_schedule(args)
    if not args.no_numerical:
        q, k, v = make_inputs(args)
        actual = cluster_attention(q, k, v)
        expected = full_attention(q, k, v)
        max_error = float(np.max(np.abs(actual - expected)))
        result["numerical"] = {
            "status": "PASS" if max_error <= 3e-5 else "FAIL",
            "max_abs_error": max_error,
            "output_shape": list(actual.shape),
        }
        actual.transpose(1, 0, 2).astype(np.float32).tofile(
            args.artifact_root / "output_query_major.bin")
        if result["numerical"]["status"] != "PASS":
            result["status"] = "FAIL"
    (args.artifact_root / "schedule.json").write_text(
        json.dumps(schedule, indent=2) + "\n")
    (args.artifact_root / "result.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
