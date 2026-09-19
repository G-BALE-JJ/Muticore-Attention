#!/usr/bin/env python3
"""Verify exact Phase E four-manager/sixteen-worker activity."""

import argparse
import csv
import json
import math
import re
from decimal import Decimal
from pathlib import Path


def make_attention_activity(queries, keys, head_dim, key_block_rows=32):
    if queries <= 0 or queries % 256 != 0:
        raise ValueError("queries must be a positive multiple of 256")
    if key_block_rows <= 0 or key_block_rows % 16 != 0:
        raise ValueError("key_block_rows must be a positive multiple of 16")
    if keys <= 0 or keys % key_block_rows != 0:
        raise ValueError("keys must be divisible by key_block_rows")
    if head_dim <= 0 or head_dim % 16 != 0:
        raise ValueError("head_dim must be a positive multiple of 16")
    query_blocks_per_worker = queries // 256
    key_tiles = keys // key_block_rows
    jobs = query_blocks_per_worker * key_tiles
    rows = jobs * 16
    return {
        "qk": jobs * key_block_rows,
        "pv": jobs * head_dim,
        "jobs": jobs,
        "qblocks": query_blocks_per_worker,
        "rows": rows,
        "scaled": rows * key_block_rows,
        "dimension_panels": head_dim // 16,
        "v_tile_bytes": key_block_rows * head_dim * 4,
        "key_block_rows": key_block_rows,
    }


def make_sequential_64_activity(queries, keys, head_dim, key_block_rows=64):
    if queries <= 0 or queries % 256 != 0:
        raise ValueError("queries must be a positive multiple of 256")
    if key_block_rows != 64:
        raise ValueError("sequential-64 requires 64 key rows per tile")
    if keys <= 0 or keys % key_block_rows != 0:
        raise ValueError("keys must be divisible by 64")
    if head_dim <= 0 or head_dim % 64 != 0:
        raise ValueError("sequential-64 head_dim must be divisible by 64")
    worker_rows = queries // 16
    query_block_rows = 64
    query_blocks_per_worker = (
        worker_rows + query_block_rows - 1
    ) // query_block_rows
    key_tiles = keys // key_block_rows
    jobs = query_blocks_per_worker * key_tiles
    reduction_panels = head_dim // 64
    dimension_panels = head_dim // 64
    row_tiles = worker_rows * key_tiles
    return {
        "qk": jobs * reduction_panels * 64,
        "pv": row_tiles * dimension_panels,
        "jobs": jobs,
        "qblocks": query_blocks_per_worker,
        "rows": row_tiles,
        "scaled": row_tiles * key_block_rows,
        "dimension_panels": dimension_panels,
        "reduction_panels": reduction_panels,
        "v_tile_bytes": key_block_rows * head_dim * 4,
        "key_block_rows": key_block_rows,
        "key_tiles": key_tiles,
        "query_rows_per_worker": worker_rows,
        "query_block_rows": query_block_rows,
        "head_dim": head_dim,
    }


def scale_activity_for_heads(activity, query_heads, kv_heads=None):
    kv_heads = query_heads if kv_heads is None else kv_heads
    if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads != 0:
        raise ValueError("query heads must be divisible by K/V heads")
    scaled = dict(activity)
    for field in ("qk", "pv", "jobs", "qblocks", "rows", "scaled"):
        scaled[field] *= query_heads
    scaled["heads"] = query_heads
    scaled["query_heads"] = query_heads
    scaled["kv_heads"] = kv_heads
    scaled["gqa_group_size"] = query_heads // kv_heads
    return scaled


def manager_slot_stalls_valid(ideal_lookahead_loads, lookahead_loads,
                              slot_stalls, workers_per_group=4):
    missed_lookaheads = max(ideal_lookahead_loads - lookahead_loads, 0)
    return 0 <= slot_stalls <= missed_lookaheads * workers_per_group


def attention_cluster_ii_sample_counts(qblocks, key_tiles, group_size):
    if qblocks <= 0 or key_tiles <= 0 or group_size <= 0:
        raise ValueError("qblocks, key_tiles, and group_size must be positive")
    full_groups, tail = divmod(qblocks, group_size)
    steady_per_key = full_groups * max(group_size - 2, 0)
    if tail:
        steady_per_key += max(tail - 2, 0)
    physical_tiles = (full_groups + (1 if tail else 0)) * key_tiles
    return {
        "steady": steady_per_key * key_tiles,
        "boundary": max(physical_tiles - 1, 0),
    }


def summarize_attention_cluster_resource_profile(
        observed, maxima, critical_worker_core, accelerator_clock_hz,
        timebase_ticks_per_second):
    """Separate per-worker interval unions from cross-worker accumulated work."""
    resources = {
        "qk_array_active": ("rocc", "attention_cluster_qk_array", "ticks"),
        "pv_array_active": ("rocc", "attention_cluster_pv_array", "ticks"),
        "array_buffer_ports": (
            "rocc:array", "attention_cluster_buffer", "ticks"
        ),
        "local_gm_read_ports": (
            "rocc:global_memory", "attention_cluster_local_read", "ticks"
        ),
        "local_gm_write_ports": (
            "rocc:global_memory", "attention_cluster_local_write", "ticks"
        ),
        "sfu": ("rocc:sfu", "attention_cluster_sfu", "ticks"),
        "o_read_port": ("rocc", "attention_cluster_o_read", "cycles"),
        "o_write_port": ("rocc", "attention_cluster_o_write", "cycles"),
        "o_alu": ("rocc", "attention_cluster_o_alu", "cycles"),
    }
    result = {
        "critical_worker_core": critical_worker_core,
        "interval_semantics": (
            "Each worker entry is a union of real [start,end) service intervals. "
            "worker_totals are sums across workers and are not end-to-end latency."
        ),
        "resources": {},
    }
    for name, (suffix, prefix, unit) in resources.items():
        workers = []
        for core in range(4, 20):
            component = f"core{core}:{suffix}"
            busy = observed.get((component, f"{prefix}_busy_union_{unit}"), 0)
            span = observed.get((component, f"{prefix}_busy_span_{unit}"), 0)
            idle = observed.get((component, f"{prefix}_idle_gap_{unit}"), 0)
            if unit == "ticks":
                busy_ticks, span_ticks, idle_ticks = busy, span, idle
                busy_cycles = ticks_to_cycles(
                    busy, accelerator_clock_hz, timebase_ticks_per_second
                )
                span_cycles = ticks_to_cycles(
                    span, accelerator_clock_hz, timebase_ticks_per_second
                )
                idle_cycles = ticks_to_cycles(
                    idle, accelerator_clock_hz, timebase_ticks_per_second
                )
            else:
                busy_cycles, span_cycles, idle_cycles = busy, span, idle
                busy_ticks = busy * timebase_ticks_per_second // accelerator_clock_hz
                span_ticks = span * timebase_ticks_per_second // accelerator_clock_hz
                idle_ticks = idle * timebase_ticks_per_second // accelerator_clock_hz
            worker = {
                "core": core,
                "busy_ticks": busy_ticks,
                "busy_cycles": busy_cycles,
                "span_ticks": span_ticks,
                "span_cycles": span_cycles,
                "idle_ticks": idle_ticks,
                "idle_cycles": idle_cycles,
                "busy_fraction_of_span": (
                    busy / span if span else 0.0
                ),
                "max_concurrency": maxima.get(
                    (component, f"{prefix}_max_concurrency"), 0
                ),
                "source_unit": unit,
            }
            workers.append(worker)
        critical = next(
            worker for worker in workers if worker["core"] == critical_worker_core
        )
        result["resources"][name] = {
            "critical_worker": critical,
            "worker_totals": {
                key: sum(worker[key] for worker in workers)
                for key in ("busy_ticks", "busy_cycles", "span_ticks",
                            "span_cycles", "idle_ticks", "idle_cycles")
            },
            "max_worker_concurrency": max(
                worker["max_concurrency"] for worker in workers
            ),
            "workers": workers,
        }
    return result


def ceil_log2(value):
    stages = 0
    covered = 1
    while covered < value:
        covered *= 2
        stages += 1
    return stages


def expected_matrix_broadcast_activity(
        activity, pv_enabled, qk_enabled, qk_dataflow_transpose,
        max_fanout, bytes_per_cycle, base_latency_cycles,
        stage_latency_cycles, pv_active_k=False):
    fanout = 16
    pv_requests = activity["pv"] // fanout if pv_enabled else 0
    qk_requests = 0
    if qk_enabled:
        qk_requests = (
            activity["qk"] // fanout
            if qk_dataflow_transpose else activity["jobs"]
        )
    requests = pv_requests + qk_requests
    qk_payload_bytes = activity["v_tile_bytes"] // 2
    pv_payload_bytes = 16 * 32 * 4 if pv_active_k else qk_payload_bytes
    qk_transfer_cycles = (
        base_latency_cycles
        + (qk_payload_bytes + bytes_per_cycle - 1) // bytes_per_cycle
        + ceil_log2(fanout) * stage_latency_cycles
    )
    pv_transfer_cycles = (
        base_latency_cycles
        + (pv_payload_bytes + bytes_per_cycle - 1) // bytes_per_cycle
        + ceil_log2(fanout) * stage_latency_cycles
    )
    payload_bytes = qk_payload_bytes if qk_requests else pv_payload_bytes
    transfer_cycles = qk_transfer_cycles if qk_requests else pv_transfer_cycles
    return {
        "enabled": requests > 0,
        "fanout": fanout,
        "max_fanout": max_fanout,
        "payload_bytes": payload_bytes,
        "qk_payload_bytes": qk_payload_bytes,
        "pv_payload_bytes": pv_payload_bytes,
        "pv_requests": pv_requests,
        "qk_requests": qk_requests,
        "requests": requests,
        "rejected": 0,
        "ingress_bytes": (
            qk_requests * qk_payload_bytes + pv_requests * pv_payload_bytes
        ),
        "sink_bytes": (
            qk_requests * qk_payload_bytes + pv_requests * pv_payload_bytes
        ) * fanout,
        "transfer_cycles": (
            qk_requests * qk_transfer_cycles +
            pv_requests * pv_transfer_cycles
        ),
        "cycles_per_request": transfer_cycles,
        "qk_cycles_per_request": qk_transfer_cycles,
        "pv_cycles_per_request": pv_transfer_cycles,
        "tree_stages": ceil_log2(fanout),
        "fanout_sum": requests * fanout,
        "min_fanout": fanout,
        "max_observed_fanout_expected": fanout,
    }


def expected_sequential_64_broadcast_activity(
        activity, bytes_per_cycle, base_latency_cycles,
        stage_latency_cycles):
    def transfer_cycles(payload_bytes, fanout):
        return (
            base_latency_cycles
            + (payload_bytes + bytes_per_cycle - 1) // bytes_per_cycle
            + ceil_log2(fanout) * stage_latency_cycles
        )

    jobs = activity["jobs"]
    panels = activity["dimension_panels"]
    reduction_panels = activity.get("reduction_panels", 1)
    worker_rows = activity.get("query_rows_per_worker", 16)
    query_block_rows = activity.get("query_block_rows", 16)
    key_tiles = activity.get(
        "key_tiles", activity["jobs"] // activity["qblocks"]
    )
    full_blocks, tail_rows = divmod(worker_rows, query_block_rows)
    block_rows = [query_block_rows] * full_blocks
    if tail_rows:
        block_rows.append(tail_rows)
    qk_requests = jobs * reduction_panels
    pv_matrix_requests = jobs * panels
    qk_bytes = 64 * 64 * 4
    pv_matrix_bytes = 64 * activity["key_block_rows"] * 4
    request_groups = [(qk_requests, qk_bytes, 64)]
    request_groups.extend(
        (activity.get("heads", 1) * key_tiles * panels,
         pv_matrix_bytes, rows) for rows in block_rows
    )
    qk_cycles = transfer_cycles(qk_bytes, 64)
    pv_matrix_cycles = transfer_cycles(pv_matrix_bytes, query_block_rows)
    return {
        "enabled": True,
        "fanout": 64,
        "max_fanout": 64,
        "payload_bytes": qk_bytes,
        "qk_payload_bytes": qk_bytes,
        "pv_payload_bytes": pv_matrix_bytes,
        "qk_requests": qk_requests,
        "pv_requests": pv_matrix_requests,
        "cycles_per_request": qk_cycles,
        "qk_cycles_per_request": qk_cycles,
        "pv_cycles_per_request": pv_matrix_cycles,
        "tree_stages": 6,
        "requests": sum(count for count, _, _ in request_groups),
        "rejected": 0,
        "ingress_bytes": sum(count * size for count, size, _ in request_groups),
        "sink_bytes": sum(
            count * size * fanout for count, size, fanout in request_groups
        ),
        "transfer_cycles": sum(
            count * transfer_cycles(size, fanout)
            for count, size, fanout in request_groups
        ),
        "fanout_sum": sum(
            count * fanout for count, _, fanout in request_groups
        ),
        "min_fanout": min(fanout for _, _, fanout in request_groups),
        "max_observed_fanout_expected": 64,
    }


def expected_sequential_64_scatter_activity(
        activity, bytes_per_cycle, base_latency_cycles):
    payload_bytes = 64 * 64 * 4
    qk_requests = activity["jobs"] * activity.get("reduction_panels", 1)
    pv_requests = activity["jobs"]
    requests = qk_requests + pv_requests
    qk_cycles_per_request = (
        base_latency_cycles
        + (payload_bytes + bytes_per_cycle - 1) // bytes_per_cycle
    )
    worker_rows = activity.get("query_rows_per_worker", 64)
    query_block_rows = activity.get("query_block_rows", 64)
    key_tiles = activity.get("key_tiles", activity["jobs"] // activity["qblocks"])
    full_blocks, tail_rows = divmod(worker_rows, query_block_rows)
    block_rows = [query_block_rows] * full_blocks
    if tail_rows:
        block_rows.append(tail_rows)
    heads = activity.get("heads", 1)
    pv_bytes = heads * sum(rows * 64 * 4 * key_tiles for rows in block_rows)
    pv_cycles = heads * sum(
        key_tiles * (
            base_latency_cycles
            + (rows * 64 * 4 + bytes_per_cycle - 1) // bytes_per_cycle
        )
        for rows in block_rows
    )
    pv_destinations = heads * sum(rows * key_tiles for rows in block_rows)
    return {
        "enabled": True,
        "lanes": 64,
        "bytes_per_cycle": bytes_per_cycle,
        "payload_bytes": payload_bytes,
        "qk_requests": qk_requests,
        "pv_requests": pv_requests,
        "requests": requests,
        "bytes": qk_requests * payload_bytes + pv_bytes,
        "cycles_per_request": qk_cycles_per_request,
        "transfer_cycles": qk_requests * qk_cycles_per_request + pv_cycles,
        "destinations": qk_requests * 64 + pv_destinations,
    }


def expected_sequential_64_o_scatter_gather_activity(
        activity, bytes_per_cycle, base_latency_cycles):
    payload_bytes = 64 * 64 * 4
    panels = activity["dimension_panels"]
    worker_rows = activity.get("query_rows_per_worker", 64)
    query_block_rows = activity.get("query_block_rows", 64)
    key_tiles = activity.get("key_tiles", activity["jobs"] // activity["qblocks"])
    full_blocks, tail_rows = divmod(worker_rows, query_block_rows)
    block_rows = [query_block_rows] * full_blocks
    if tail_rows:
        block_rows.append(tail_rows)
    heads = activity.get("heads", 1)
    scatter_requests = heads * (key_tiles - 1) * panels * len(block_rows)
    gather_requests = heads * key_tiles * panels * len(block_rows)
    requests = scatter_requests + gather_requests
    bytes_total = heads * sum(
        ((key_tiles - 1) + key_tiles) * panels * rows * 64 * 4
        for rows in block_rows
    )
    transfer_cycles = heads * sum(
        ((key_tiles - 1) + key_tiles) * panels * (
            base_latency_cycles
            + (rows * 64 * 4 + bytes_per_cycle - 1) // bytes_per_cycle
        )
        for rows in block_rows
    )
    destinations = heads * sum(
        ((key_tiles - 1) + key_tiles) * panels * rows
        for rows in block_rows
    )
    return {
        "enabled": True,
        "lanes": 64,
        "bytes_per_cycle": bytes_per_cycle,
        "payload_bytes": payload_bytes,
        "scatter_requests": scatter_requests,
        "gather_requests": gather_requests,
        "requests": requests,
        "bytes": bytes_total,
        "cycles_per_request": (
            base_latency_cycles
            + (payload_bytes + bytes_per_cycle - 1) // bytes_per_cycle
        ),
        "transfer_cycles": transfer_cycles,
        "destinations": destinations,
    }


def expected_sequential_qk_score_readout(activity):
    tile_bytes = 64 * 64 * 4
    return {
        "requests": activity["jobs"],
        "bytes": activity["jobs"] * tile_bytes,
    }


def expected_attention_cluster_broadcast_activity(
        activity, bytes_per_cycle, base_latency_cycles,
        stage_latency_cycles, physical_kv_jobs=None, qk_arrays=16):
    def transfer_cycles(payload_bytes, fanout):
        return (
            base_latency_cycles
            + (payload_bytes + bytes_per_cycle - 1) // bytes_per_cycle
            + ceil_log2(fanout) * stage_latency_cycles
        )

    if physical_kv_jobs is None:
        physical_kv_jobs = activity["jobs"]
    qk_fanout = min(qk_arrays // 2, 16)
    pv_fanout = min((64 - qk_arrays) // 2, 16)
    key_block_rows = activity.get("key_block_rows", 32)
    k_payload_bytes = 64 * 64 * 4
    q_payload_bytes = 64 * 4
    pv_payload_bytes = key_block_rows * 4
    v_payload_bytes = 64 * key_block_rows * 4
    k_requests = physical_kv_jobs * 2
    v_requests = physical_kv_jobs * 2
    q_requests = activity["jobs"] * 32
    pv_requests = activity["jobs"] * 16
    k_cycles = transfer_cycles(k_payload_bytes, qk_fanout)
    v_cycles = transfer_cycles(v_payload_bytes, pv_fanout)
    q_cycles = transfer_cycles(q_payload_bytes, 1)
    pv_cycles = transfer_cycles(pv_payload_bytes, 2)
    max_fanout = max(qk_fanout, pv_fanout, 2)
    return {
        "enabled": True,
        "fanout": max_fanout,
        "min_fanout": 1,
        "max_observed_fanout_expected": max_fanout,
        "max_fanout": max_fanout,
        "payload_bytes": None,
        "qk_payload_bytes": k_payload_bytes,
        "q_pair_payload_bytes": q_payload_bytes,
        "pv_payload_bytes": pv_payload_bytes,
        "pv_requests": pv_requests,
        "qk_requests": k_requests,
        "q_pair_requests": q_requests,
        "requests": k_requests + v_requests + q_requests + pv_requests,
        "rejected": 0,
        "ingress_bytes": (
            k_requests * k_payload_bytes
            + v_requests * v_payload_bytes
            + q_requests * q_payload_bytes
            + pv_requests * pv_payload_bytes
        ),
        "sink_bytes": (
            k_requests * k_payload_bytes * qk_fanout
            + v_requests * v_payload_bytes * pv_fanout
            + q_requests * q_payload_bytes
            + pv_requests * pv_payload_bytes * 2
        ),
        "transfer_cycles": (
            k_requests * k_cycles
            + v_requests * v_cycles
            + q_requests * q_cycles
            + pv_requests * pv_cycles
        ),
        "cycles_per_request": None,
        "qk_cycles_per_request": k_cycles,
        "q_pair_cycles_per_request": q_cycles,
        "pv_cycles_per_request": pv_cycles,
        "tree_stages": ceil_log2(max_fanout),
        "fanout_sum": (
            k_requests * qk_fanout + v_requests * pv_fanout +
            q_requests + pv_requests * 2
        ),
    }

INTER_TILE_PHASE_STATS = (
    ("output_dma", "attention_worker_intertile_output_dma_ticks"),
    ("query_load", "attention_worker_intertile_query_load_ticks"),
    ("kv_load", "attention_worker_intertile_kv_load_ticks"),
    ("q_local_read", "attention_worker_intertile_q_local_read_ticks"),
    ("qk_matrix_program", "attention_worker_intertile_qk_matrix_program_ticks"),
    ("qk_input_program", "attention_worker_intertile_qk_input_program_ticks"),
    ("qk_compute_readout", "attention_worker_intertile_qk_compute_readout_ticks"),
)

TILE_PIPELINE_PHASE_STATS = (
    ("kv_load", "attention_worker_tile_kv_load_ticks"),
    ("q_local_read", "attention_worker_tile_q_local_read_ticks"),
    ("qk_matrix_program", "attention_worker_tile_qk_matrix_program_ticks"),
    ("qk_input_program", "attention_worker_tile_qk_input_program_ticks"),
    ("qk_compute_readout", "attention_worker_tile_qk_compute_readout_ticks"),
    ("softmax", "attention_worker_tile_softmax_ticks"),
    ("pv_matrix_program", "attention_worker_tile_pv_matrix_program_ticks"),
    ("pv_input_program", "attention_worker_tile_pv_input_program_ticks"),
    ("pv_restore_output", "attention_worker_tile_pv_restore_output_ticks"),
    ("pv_compute", "attention_worker_tile_pv_compute_ticks"),
    ("pv_output_readwrite", "attention_worker_tile_pv_output_readwrite_ticks"),
)


def parse_frequency_hz(value):
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([kKmMgGtT]?)Hz\s*", value)
    if not match:
        raise argparse.ArgumentTypeError(f"unsupported frequency: {value}")
    scale = {"": 1, "k": 10**3, "m": 10**6, "g": 10**9, "t": 10**12}
    hz = Decimal(match.group(1)) * scale[match.group(2).lower()]
    if hz != hz.to_integral_value() or hz <= 0:
        raise argparse.ArgumentTypeError(f"frequency must be a positive integer Hz: {value}")
    return int(hz)


def parse_dma_runtime_invariants(path, expected_data_nodes=4,
                                 expected_credit_cap=None):
    """Parse end-of-run DMA summaries and validate resource conservation."""
    credit_summaries = []
    compact_credit_summaries = []
    response_summaries = []
    compact_response_summaries = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if "GOLEM_MEMNIC_DMA_CREDIT_CONSERVATION" in line:
            fields = dict(re.findall(r"([a-z0-9_]+)=([^\s]+)", line))
            compact_credit_summaries.append(fields)
        elif "CREDIT_OWNER_SUMMARY" in line:
            fields = dict(re.findall(r"([a-z0-9_]+)=([^\s]+)", line))
            credit_summaries.append(fields)
        elif "GOLEM_MEMNIC_DMA_RESPONSE_CONSERVATION" in line:
            fields = dict(re.findall(r"([a-z0-9_]+)=([^\s]+)", line))
            compact_response_summaries.append(fields)
        elif "GOLEM_MEMNIC_DMA_RESPONSE_STATS" in line:
            fields = dict(re.findall(r"([a-z0-9_]+)=([^\s]+)", line))
            response_summaries.append(fields)

    if compact_credit_summaries:
        credit_summaries = compact_credit_summaries
    if compact_response_summaries:
        response_summaries = compact_response_summaries

    mismatches = {}
    credit_components = [fields.get("name", "unknown") for fields in credit_summaries]
    response_components = [
        fields.get("component", "unknown") for fields in response_summaries
    ]
    expected_credit_components = {
        f"dirctrl_{node}:highlink" for node in range(expected_data_nodes + 1)
    }
    expected_response_components = {
        f"dirctrl_{node}:highlink" for node in range(1, expected_data_nodes + 1)
    }
    if expected_credit_cap == 0 and credit_summaries:
        mismatches["dma_credit/runtime_summary"] = {
            "expected": 0, "actual": len(credit_summaries),
        }
    elif expected_credit_cap != 0 and set(credit_components) != expected_credit_components:
        mismatches["dma_credit/runtime_summary"] = {
            "expected": sorted(expected_credit_components),
            "actual": sorted(set(credit_components)),
        }
    if len(credit_components) != len(set(credit_components)):
        mismatches["dma_credit/runtime_duplicates"] = {
            "expected": "unique components", "actual": credit_components,
        }
    if set(response_components) != expected_response_components:
        mismatches["dma_response/runtime_summary"] = {
            "expected": sorted(expected_response_components),
            "actual": sorted(set(response_components)),
        }
    if len(response_components) != len(set(response_components)):
        mismatches["dma_response/runtime_duplicates"] = {
            "expected": "unique components", "actual": response_components,
        }
    for fields in credit_summaries:
        component = fields.get("name", "unknown")
        numeric = {key: int(fields[key]) for key in (
            "cap", "available", "admitted", "released", "pending"
        )}
        if expected_credit_cap is not None and numeric["cap"] != expected_credit_cap:
            mismatches[f"{component}/dma_credit_cap"] = {
                "expected": expected_credit_cap, "actual": numeric["cap"],
            }
        if numeric["available"] != numeric["cap"]:
            mismatches[f"{component}/dma_credit_available"] = {
                "expected": numeric["cap"], "actual": numeric["available"],
            }
        if numeric["admitted"] != numeric["released"]:
            mismatches[f"{component}/dma_credit_conservation"] = {
                "expected": numeric["admitted"], "actual": numeric["released"],
            }
        if numeric["pending"] != 0:
            mismatches[f"{component}/dma_admission_pending"] = {
                "expected": 0, "actual": numeric["pending"],
            }
    for fields in response_summaries:
        component = fields.get("component", "unknown")
        numeric = {key: int(fields[key]) for key in (
            "attempted", "immediate", "enqueued", "drained", "pending",
            "responses_d0", "responses_d1", "responses_d2", "responses_far",
        )}
        completed = numeric["immediate"] + numeric["enqueued"]
        if numeric["attempted"] != completed:
            mismatches[f"{component}/dma_response_classification"] = {
                "expected": numeric["attempted"], "actual": completed,
            }
        if numeric["enqueued"] != numeric["drained"]:
            mismatches[f"{component}/dma_response_drain"] = {
                "expected": numeric["enqueued"], "actual": numeric["drained"],
            }
        if numeric["pending"] != 0:
            mismatches[f"{component}/dma_response_pending"] = {
                "expected": 0, "actual": numeric["pending"],
            }
        distance_responses = sum(
            numeric[key] for key in (
                "responses_d0", "responses_d1", "responses_d2", "responses_far"
            )
        )
        if distance_responses != numeric["attempted"]:
            mismatches[f"{component}/dma_response_distance_classification"] = {
                "expected": numeric["attempted"], "actual": distance_responses,
            }
    if expected_credit_cap != 0:
        credit_by_component = {
            fields.get("name", "unknown"): fields for fields in credit_summaries
        }
        response_by_component = {
            fields.get("component", "unknown"): fields
            for fields in response_summaries
        }
        for component in expected_response_components:
            if component not in credit_by_component or component not in response_by_component:
                continue
            admitted = int(credit_by_component[component]["admitted"])
            attempted = int(response_by_component[component]["attempted"])
            if admitted != attempted:
                mismatches[f"{component}/dma_credit_response_conservation"] = {
                    "expected": admitted, "actual": attempted,
                }
    return {
        "credit_owners": credit_summaries,
        "response_nodes": response_summaries,
        "mismatches": mismatches,
    }


def read_model_clocks_from_run_config(path):
    keys = {
        "VANADIS_CPU_CLOCK": "cpu",
        "GOLEM_ARRAY_CLOCK": "array",
        "GOLEM_MEMCTRL_CLOCK": "memctrl",
    }
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"\s*([A-Z0-9_]+)=([^\s#]+)\s*", line)
        if match and match.group(1) in keys:
            values[keys[match.group(1)]] = parse_frequency_hz(match.group(2))
    missing = sorted(set(keys.values()) - set(values))
    if missing:
        raise ValueError(
            f"run configuration is missing model clocks: {', '.join(missing)}"
        )
    return values


def read_integer_from_run_config(path, key):
    pattern = re.compile(rf"\b{re.escape(key)}=([0-9]+)\s*$")
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            return int(match.group(1))
    raise ValueError(f"run configuration is missing {key}")


def ticks_to_cycles(ticks, clock_hz, timebase_ticks_per_second):
    return (ticks * clock_hz + timebase_ticks_per_second - 1) // timebase_ticks_per_second


def ticks_to_milliseconds(ticks, timebase_ticks_per_second):
    return ticks * 1000.0 / timebase_ticks_per_second


def make_clock_contract(normalization_clock_hz, timebase_ticks_per_second,
                        model_cpu_clock_hz, model_array_clock_hz,
                        model_memctrl_clock_hz, model_platform_clock_hz):
    return {
        "sst_timebase_ticks_per_second": timebase_ticks_per_second,
        "normalization_clock_hz": normalization_clock_hz,
        "normalized_cycle_definition": (
            "ceil(sst_timebase_ticks * normalization_clock_hz / "
            "sst_timebase_ticks_per_second)"
        ),
        "normalized_cycles_are_model_native_cycles": (
            normalization_clock_hz == model_cpu_clock_hz
            and normalization_clock_hz == model_array_clock_hz
        ),
        "model_clocks_hz": {
            "vanadis_cpu_rocc_sfu_local_gm": model_cpu_clock_hz,
            "mvm_array": model_array_clock_hz,
            "memory_controller": model_memctrl_clock_hz,
            "archive_platform": model_platform_clock_hz,
        },
    }


def expected_v_tile_buffer_activity(activity, enabled, capacity_bytes,
                                    hit_ticks, bytes_per_cycle=64,
                                    group_retention=False,
                                    physical_jobs=None):
    if not enabled:
        return {"hits": 0, "misses": 0, "bytes_read": 0,
                "bytes_reused": 0, "wait_ticks": 0,
                "capacity_rejections": 0, "group_hits": 0}
    panels = activity["dimension_panels"]
    jobs = activity["jobs"]
    tile_bytes = activity["v_tile_bytes"]
    if capacity_bytes < tile_bytes:
        return {
            "hits": 0,
            "misses": jobs * panels,
            "bytes_read": jobs * panels * tile_bytes,
            "bytes_reused": 0,
            "wait_ticks": 0,
            "capacity_rejections": jobs * panels,
            "group_hits": 0,
        }
    retained_jobs = physical_jobs if group_retention else jobs
    if retained_jobs is None:
        retained_jobs = jobs
    hits = jobs * panels - retained_jobs
    group_hits = jobs - retained_jobs if group_retention else 0
    panel_bytes = 32 * 16 * 4
    access_ticks = hit_ticks + (
        panel_bytes + bytes_per_cycle - 1
    ) // bytes_per_cycle
    return {
        "hits": hits,
        "misses": retained_jobs,
        "bytes_read": retained_jobs * tile_bytes,
        "bytes_reused": hits * panel_bytes,
        "wait_ticks": hits * access_ticks,
        "capacity_rejections": 0,
        "group_hits": group_hits,
    }


def summarize_intertile_breakdown(observed, counts, worker_core,
                                  accelerator_clock_hz,
                                  timebase_ticks_per_second):
    component = f"core{worker_core}:rocc"
    total_stat = "attention_worker_intertile_total_ticks"
    total_ticks = observed.get((component, total_stat))
    if total_ticks is None:
        return {}
    phase_ticks = {
        label: observed.get((component, statistic), 0)
        for label, statistic in INTER_TILE_PHASE_STATS
    }
    attributed_ticks = sum(phase_ticks.values())
    unattributed_ticks = total_ticks - attributed_ticks
    return {
        "worker_core": worker_core,
        "transition_count": counts.get((component, total_stat), 0),
        "total_ticks": total_ticks,
        "total_cycles": ticks_to_cycles(
            total_ticks, accelerator_clock_hz, timebase_ticks_per_second
        ),
        "phase_ticks": phase_ticks,
        "phase_cycles": {
            label: ticks_to_cycles(
                ticks, accelerator_clock_hz, timebase_ticks_per_second
            )
            for label, ticks in phase_ticks.items()
        },
        "phase_counts": {
            label: counts.get((component, statistic), 0)
            for label, statistic in INTER_TILE_PHASE_STATS
        },
        "attributed_ticks": attributed_ticks,
        "unattributed_ticks": unattributed_ticks,
        "conservation_valid": unattributed_ticks == 0,
    }


def summarize_tile_pipeline_breakdown(observed, counts, worker_core,
                                      accelerator_clock_hz,
                                      timebase_ticks_per_second):
    component = f"core{worker_core}:rocc"
    total_stat = "attention_worker_tile_total_ticks"
    total_ticks = observed.get((component, total_stat))
    if total_ticks is None:
        return {}
    phase_ticks = {
        label: observed.get((component, statistic), 0)
        for label, statistic in TILE_PIPELINE_PHASE_STATS
    }
    attributed_ticks = sum(phase_ticks.values())
    unattributed_ticks = total_ticks - attributed_ticks
    return {
        "worker_core": worker_core,
        "tile_count": counts.get((component, total_stat), 0),
        "total_ticks": total_ticks,
        "total_cycles": ticks_to_cycles(
            total_ticks, accelerator_clock_hz, timebase_ticks_per_second
        ),
        "phase_ticks": phase_ticks,
        "phase_cycles": {
            label: ticks_to_cycles(
                ticks, accelerator_clock_hz, timebase_ticks_per_second
            )
            for label, ticks in phase_ticks.items()
        },
        "phase_counts": {
            label: counts.get((component, statistic), 0)
            for label, statistic in TILE_PIPELINE_PHASE_STATS
        },
        "attributed_ticks": attributed_ticks,
        "unattributed_ticks": unattributed_ticks,
        "conservation_valid": unattributed_ticks == 0,
    }


def summarize_kv_second_lookahead(observed, counts, maxima, worker_cores,
                                  max_candidates, accelerator_clock_hz,
                                  timebase_ticks_per_second):
    components = [f"core{core}:rocc" for core in worker_cores]
    timing_stats = {
        "k_release": "attention_kv_k_release_ticks",
        "v_release": "attention_kv_v_release_ticks",
        "available_lead": "attention_kv_second_lookahead_lead_ticks",
    }
    timing_ticks = {
        label: sum(observed.get((component, statistic), 0)
                   for component in components)
        for label, statistic in timing_stats.items()
    }
    timing_counts = {
        label: sum(counts.get((component, statistic), 0)
                   for component in components)
        for label, statistic in timing_stats.items()
    }
    candidates = sum(observed.get(
        (component, "attention_kv_second_lookahead_candidates"), 0
    ) for component in components)
    prefetches = sum(observed.get(
        (component, "attention_kv_second_lookahead_prefetches"), 0
    ) for component in components)
    ready_at_release = sum(observed.get(
        (component, "attention_kv_next_ready_at_release_tiles"), 0
    ) for component in components)
    max_lead_ticks = max((maxima.get(
        (component, "attention_kv_second_lookahead_lead_ticks"), 0
    ) for component in components), default=0)
    return {
        "max_candidates": max_candidates,
        "candidates": candidates,
        "prefetches": prefetches,
        "candidate_rate": candidates / max_candidates if max_candidates else 0.0,
        "ready_at_release": ready_at_release,
        "ready_after_release_before_boundary": candidates - ready_at_release,
        "timing_ticks": timing_ticks,
        "timing_counts": timing_counts,
        "mean_cycles": {
            label: (
                ticks_to_cycles(timing_ticks[label], accelerator_clock_hz,
                                timebase_ticks_per_second) / count
                if count else 0.0
            )
            for label, count in timing_counts.items()
        },
        "max_available_lead_cycles": ticks_to_cycles(
            max_lead_ticks, accelerator_clock_hz, timebase_ticks_per_second
        ),
    }


def summarize_worker_critical_path(observed, minima, maxima, worker_cores,
                                   accelerator_clock_hz,
                                   timebase_ticks_per_second):
    suffixes = (
        ("dispatch_accept", "attention_worker_dispatch_accept_tick", minima),
        ("final_qk_tile_complete", "attention_worker_qk_tile_complete_tick", maxima),
        ("final_softmax_tile_complete",
         "attention_worker_softmax_tile_complete_tick", maxima),
        ("final_pv_tile_complete", "attention_worker_pv_tile_complete_tick", maxima),
        ("final_output_dma_ack", "attention_worker_output_dma_ack_tick", maxima),
    )
    paths = {}
    for core in worker_cores:
        component = f"core{core}:rocc"
        milestones = {
            label: values.get((component, statistic))
            for label, statistic, values in suffixes
        }
        if all(value is not None for value in milestones.values()):
            paths[core] = milestones
    if not paths:
        return {}

    slowest_core = max(paths, key=lambda core: paths[core]["final_output_dma_ack"])
    milestones = paths[slowest_core]
    ordered = list(milestones.values())
    if ordered != sorted(ordered):
        return {
            "slowest_worker_core": slowest_core,
            "milestone_ticks": milestones,
            "order_valid": False,
        }

    stage_pairs = (
        ("dispatch_to_final_qk", "dispatch_accept", "final_qk_tile_complete"),
        ("final_qk_to_final_softmax", "final_qk_tile_complete",
         "final_softmax_tile_complete"),
        ("final_softmax_to_final_pv", "final_softmax_tile_complete",
         "final_pv_tile_complete"),
        ("final_pv_to_output_dma_ack", "final_pv_tile_complete",
         "final_output_dma_ack"),
    )
    stage_ticks = {
        label: milestones[end] - milestones[start]
        for label, start, end in stage_pairs
    }
    component = f"core{slowest_core}:rocc"
    qk_sum = observed[(component, "attention_worker_qk_tile_complete_tick")]
    softmax_sum = observed[(component, "attention_worker_softmax_tile_complete_tick")]
    pv_sum = observed[(component, "attention_worker_pv_tile_complete_tick")]
    aggregate_ticks = {
        "dispatch_to_first_qk": minima[
            (component, "attention_worker_qk_tile_complete_tick")
        ] - milestones["dispatch_accept"],
        "all_qk_to_softmax": softmax_sum - qk_sum,
        "all_softmax_to_pv": pv_sum - softmax_sum,
        "inter_tile_pv_to_next_qk": (
            qk_sum - minima[(component, "attention_worker_qk_tile_complete_tick")]
            - pv_sum + milestones["final_pv_tile_complete"]
        ),
        "final_pv_to_output_dma_ack": (
            milestones["final_output_dma_ack"] -
            milestones["final_pv_tile_complete"]
        ),
    }
    return {
        "slowest_worker_core": slowest_core,
        "milestone_ticks": milestones,
        "stage_ticks": stage_ticks,
        "stage_cycles": {
            label: ticks_to_cycles(
                ticks, accelerator_clock_hz, timebase_ticks_per_second
            )
            for label, ticks in stage_ticks.items()
        },
        "aggregate_online_pipeline_cycles": {
            label: ticks_to_cycles(
                ticks, accelerator_clock_hz, timebase_ticks_per_second
            )
            for label, ticks in aggregate_ticks.items()
        },
        "aggregate_interpretation": (
            "PV-to-next-QK includes next-tile KV/QK preparation and any "
            "intermediate query-block output DMA."
        ),
        "order_valid": True,
    }


def summarize_system_frontier(observed, minima, maxima, accelerator_clock_hz,
                              timebase_ticks_per_second):
    root = "core0:rocc"
    milestones = {
        "root_descriptor_accept": minima.get(
            (root, "attention_manager_descriptor_accept_tick")
        ),
        "manager_dispatch_complete": max(
            maxima.get((f"core{core}:rocc", "attention_manager_dispatch_tick"), 0)
            for core in range(4)
        ),
        "worker_dispatch_accept_complete": max(
            maxima.get((f"core{core}:rocc", "attention_worker_dispatch_accept_tick"), 0)
            for core in range(4, 20)
        ),
        "final_qk_tile_complete": max(
            maxima.get((f"core{core}:rocc", "attention_worker_qk_tile_complete_tick"), 0)
            for core in range(4, 20)
        ),
        "final_softmax_tile_complete": max(
            maxima.get((f"core{core}:rocc", "attention_worker_softmax_tile_complete_tick"), 0)
            for core in range(4, 20)
        ),
        "final_pv_tile_complete": max(
            maxima.get((f"core{core}:rocc", "attention_worker_pv_tile_complete_tick"), 0)
            for core in range(4, 20)
        ),
        "final_output_dma_ack": max(
            maxima.get((f"core{core}:rocc", "attention_worker_output_dma_ack_tick"), 0)
            for core in range(4, 20)
        ),
        "manager_local_complete": max(
            maxima.get((f"core{core}:rocc", "attention_manager_local_complete_tick"), 0)
            for core in range(4)
        ),
        "root_tensor_complete": maxima.get((root, "attention_tensor_complete_tick")),
        "software_wait_observed": maxima.get(
            (root, "attention_manager_wait_observed_tick")
        ),
    }
    if any(value in (None, 0) for value in milestones.values()):
        return {}
    ordered = list(milestones.values())
    if ordered != sorted(ordered):
        return {"milestone_ticks": milestones, "order_valid": False}

    labels = list(milestones)
    stage_ticks = {
        f"{labels[index]}_to_{labels[index + 1]}":
            milestones[labels[index + 1]] - milestones[labels[index]]
        for index in range(len(labels) - 1)
    }
    accelerator_stage_ticks = {
        label: ticks
        for label, ticks in stage_ticks.items()
        if not label.startswith("root_tensor_complete_to_")
    }
    accelerator_total_ticks = (
        milestones["root_tensor_complete"] - milestones["root_descriptor_accept"]
    )
    attributed_ticks = sum(accelerator_stage_ticks.values())
    return {
        "milestone_ticks": milestones,
        "milestone_cycles_from_root_accept": {
            label: ticks_to_cycles(
                tick - milestones["root_descriptor_accept"],
                accelerator_clock_hz, timebase_ticks_per_second,
            )
            for label, tick in milestones.items()
        },
        "stage_cycles": {
            label: ticks_to_cycles(
                ticks, accelerator_clock_hz, timebase_ticks_per_second
            )
            for label, ticks in stage_ticks.items()
        },
        "stage_ticks": stage_ticks,
        "accelerator_attribution": {
            "total_ticks": accelerator_total_ticks,
            "attributed_ticks": attributed_ticks,
            "unattributed_ticks": accelerator_total_ticks - attributed_ticks,
            "coverage_ratio": (
                attributed_ticks / accelerator_total_ticks
                if accelerator_total_ticks else 0.0
            ),
            "conservation_valid": attributed_ticks == accelerator_total_ticks,
        },
        "order_valid": True,
        "interpretation": (
            "Frontier deltas describe when all parallel workers cross each milestone; "
            "dispatch-to-final-QK includes earlier online QK-Softmax-PV tiles."
        ),
    }


def verify(path, case_id, accelerator_clock_hz=1_000_000_000,
           timebase_ticks_per_second=10**12, pv_matrix_broadcast=False,
           qk_matrix_broadcast=False, qk_dataflow_transpose=False,
           kv_double_buffer=False, pv_input_pipeline=False,
           pv_matrix_softmax_overlap=False, pv_restore_pipeline=False,
           pv_output_pipeline=False, pv_early_compute=False,
           pv_v_tile_reuse=False, pv_v_tile_buffer_bytes=16384,
           pv_v_tile_buffer_hit_ticks=1,
           pv_v_tile_buffer_bytes_per_cycle=64,
           model_cpu_clock_hz=2_300_000_000,
           model_array_clock_hz=2_300_000_000,
           model_memctrl_clock_hz=2_300_000_000,
           model_platform_clock_hz=2_000_000_000,
           generic_gemm=False,
           matrix_broadcast_max_fanout=16,
           matrix_broadcast_bytes_per_cycle=64,
           matrix_broadcast_base_latency_cycles=1,
           matrix_broadcast_stage_latency_cycles=1,
           wcp_gemm_proxy_completion_latency_cycles=1,
           wcp_gemm_proxy_queue_depth=32,
           wcp_gemm_proxy_issue_width=1,
           wcp_gemm_proxy_command_latency_cycles=1,
           activity=None,
           kv_second_lookahead=True, kv_cross_query_prefetch=False,
           kv_pair_reuse=False, kv_query_group_size=2,
           pv_v_tile_group_retention=False,
           pv_active_k=False,
           array_mac_per_cu_per_cycle=1.0, array_pipeline_depth=2,
           qk_early_compute=False, qk_input_pipeline=False,
           qk_readout_overlap=False,
           qk_readout_window=2, qk_score_row_burst=False,
           pv_input_residency=False, o_accumulator_cbuffer=False,
           attention_tile_storage_banks=16,
           attention_tile_storage_bank_bytes_per_cycle=64,
           runtime_log=None,
           dma_credit_cap=None, dma_data_nodes=4,
           cross_tile_operand_pipeline=False, attention_cluster=False,
           sequential_64=False,
           attention_cluster_qk_arrays=16,
           pv_o_row_fusion=False,
           cluster_pv_row_wavefront=False,
           cluster_qk_matrix_lookahead=False,
           cluster_pv_matrix_lookahead=False,
           near_array_output_bytes_per_cycle=512,
           array_buffer_base_latency_cycles=1,
           kv_distribution=False, kv_manager_lookahead=False,
           input_scatter_bytes_per_cycle=256,
           output_scatter_gather_bytes_per_cycle=256):
    if activity is None:
        raise ValueError("dimension-derived activity is required")
    query_head_count = activity.get("query_heads", activity.get("heads", 1))
    attention_job_count = activity.get("kv_heads", query_head_count)
    observed = {}
    counts = {}
    minima = {}
    maxima = {}
    with Path(path).open(newline="", encoding="ascii") as stream:
        for row in csv.DictReader(stream):
            key = (row["ComponentName"], row["StatisticName"])
            observed[key] = int(row["Sum.u64"])
            counts[key] = int(row["Count.u64"])
            minima[key] = int(row["Min.u64"])
            maxima[key] = int(row["Max.u64"])
    expected = {}
    expected_counts = {}
    active_columns = activity.get("key_block_rows", 32)
    key_tiles = activity["jobs"] // activity["qblocks"]
    effective_kv_query_group_size = (
        kv_query_group_size if kv_pair_reuse else 1
    )
    kv_query_groups = (
        activity["qblocks"] + effective_kv_query_group_size - 1
    ) // effective_kv_query_group_size
    reused_queries = activity["qblocks"] - kv_query_groups
    ahead_operand_tiles = (
        reused_queries * max(key_tiles - 1, 0)
        if cross_tile_operand_pipeline else 0
    )
    pair_reuse_tiles = reused_queries * key_tiles
    physical_kv_jobs = kv_query_groups * key_tiles
    cluster_ii_counts = attention_cluster_ii_sample_counts(
        activity["qblocks"], key_tiles, effective_kv_query_group_size
    )
    full_columns = 64 if (attention_cluster or sequential_64) else (
        activity["v_tile_bytes"] // (active_columns * 4)
    )
    effective_pv_active_k = pv_active_k and active_columns < full_columns
    broadcast = (
        expected_attention_cluster_broadcast_activity(
            activity, matrix_broadcast_bytes_per_cycle,
            matrix_broadcast_base_latency_cycles,
            matrix_broadcast_stage_latency_cycles,
            physical_kv_jobs,
            attention_cluster_qk_arrays,
        )
        if attention_cluster else expected_sequential_64_broadcast_activity(
            activity, matrix_broadcast_bytes_per_cycle,
            matrix_broadcast_base_latency_cycles,
            matrix_broadcast_stage_latency_cycles,
        ) if sequential_64 else expected_matrix_broadcast_activity(
            activity, pv_matrix_broadcast, qk_matrix_broadcast,
            qk_dataflow_transpose, matrix_broadcast_max_fanout,
            matrix_broadcast_bytes_per_cycle,
            matrix_broadcast_base_latency_cycles,
            matrix_broadcast_stage_latency_cycles,
            effective_pv_active_k,
        )
    )
    scatter = expected_sequential_64_scatter_activity(
        activity, input_scatter_bytes_per_cycle,
        array_buffer_base_latency_cycles,
    ) if sequential_64 else {"enabled": False}
    output_scatter_gather = (
        expected_sequential_64_o_scatter_gather_activity(
            activity, output_scatter_gather_bytes_per_cycle,
            array_buffer_base_latency_cycles,
        ) if sequential_64 else {"enabled": False}
    )
    for core in range(4):
        component = f"core{core}:rocc"
        expected[(component, "attention_manager_jobs_issued")] = attention_job_count
        expected[(component, "attention_manager_jobs_completed")] = attention_job_count
        expected[(component, "attention_manager_bands_completed")] = attention_job_count
        expected[(component, "attention_manager_band_completions_received")] = (
            4 * attention_job_count if core == 0 else 0
        )
        expected[(component, "attention_tensor_jobs_completed")] = (
            attention_job_count if core == 0 else 0
        )
        expected[(f"core{core}:rocc:sfu", "sfu_attention_jobs")] = 0
        for stat in (
            "attention_manager_descriptor_accept_tick",
            "attention_manager_dispatch_tick",
            "attention_manager_local_complete_tick",
            "attention_manager_wait_observed_tick",
        ):
            expected_counts[(component, stat)] = attention_job_count
        expected_counts[(component, "attention_manager_band_completion_received_tick")] = (
            4 * attention_job_count if core == 0 else 0
        )
        expected_counts[(component, "attention_tensor_complete_tick")] = (
            attention_job_count if core == 0 else 0
        )
        if kv_distribution:
            groupctrl = f"core{core}:rocc:group_ctrl"
            ideal_lookahead_loads = (
                kv_query_groups * max(key_tiles - 2, 0)
                if kv_manager_lookahead else 0
            )
            lookahead_loads = observed.get(
                (groupctrl, "attention_kv_manager_lookahead_loads"), 0
            ) if kv_manager_lookahead else 0
            slot_stalls = observed.get(
                (groupctrl, "attention_kv_distribution_slot_stalls"), 0
            )
            if (kv_manager_lookahead and ideal_lookahead_loads > 0 and
                    not 0 < lookahead_loads <= ideal_lookahead_loads):
                expected[(groupctrl,
                          "attention_kv_manager_lookahead_loads.range")] = (
                    ideal_lookahead_loads
                )
                observed[(groupctrl,
                          "attention_kv_manager_lookahead_loads.range")] = (
                    lookahead_loads
                )
            manager_stats = {
                "attention_kv_distribution_requests": 0,
                "attention_kv_distribution_manager_loads": physical_kv_jobs,
                "attention_kv_distribution_manager_bytes": (
                    physical_kv_jobs * 2 * activity["v_tile_bytes"]
                ),
                "attention_kv_distribution_coalesced": (
                    physical_kv_jobs * 3 + lookahead_loads
                ),
                "attention_kv_distribution_deliveries": physical_kv_jobs * 4,
                "attention_kv_distribution_delivery_bytes": (
                    physical_kv_jobs * 4 * 2 * activity["v_tile_bytes"]
                ),
                "attention_kv_distribution_slot_stalls": slot_stalls,
                "attention_kv_distribution_cancels": 0,
                "attention_kv_manager_lookahead_loads": lookahead_loads,
                "attention_kv_manager_lookahead_hits": lookahead_loads,
            }
            for statistic, value in manager_stats.items():
                expected[(groupctrl, statistic)] = value
            if not manager_slot_stalls_valid(
                    ideal_lookahead_loads, lookahead_loads, slot_stalls):
                expected[(groupctrl,
                          "attention_kv_distribution_slot_stalls.range")] = (
                    max(ideal_lookahead_loads - lookahead_loads, 0) * 4
                )
                observed[(groupctrl,
                          "attention_kv_distribution_slot_stalls.range")] = (
                    slot_stalls
                )
            expected_max = 2 if physical_kv_jobs > 1 else 1
            actual_max = maxima.get(
                (groupctrl, "attention_kv_distribution_max_slots"), 0
            )
            if actual_max != expected_max:
                expected[(groupctrl, "attention_kv_distribution_max_slots.max")] = (
                    expected_max
                )
                observed[(groupctrl, "attention_kv_distribution_max_slots.max")] = (
                    actual_max
                )
    for core in range(4, 20):
        component = f"core{core}:rocc"
        qk_lookahead_launches = observed.get(
            (component, "attention_cluster_qk_matrix_lookahead_launches"), 0
        )
        qk_lookahead_min = physical_kv_jobs - kv_query_groups
        qk_lookahead_max = (
            physical_kv_jobs - 1 if kv_cross_query_prefetch
            else qk_lookahead_min
        )
        if (cluster_qk_matrix_lookahead and qk_lookahead_max > 0 and
                not qk_lookahead_min <= qk_lookahead_launches <=
                    qk_lookahead_max):
            expected[(component,
                      "attention_cluster_qk_matrix_lookahead_launches.range")] = (
                qk_lookahead_max
            )
            observed[(component,
                      "attention_cluster_qk_matrix_lookahead_launches.range")] = (
                qk_lookahead_launches
            )
        pv_lookahead_candidates = physical_kv_jobs - kv_query_groups
        pv_lookahead_launches = observed.get(
            (component, "attention_cluster_pv_matrix_lookahead_launches"), 0
        )
        if (cluster_pv_matrix_lookahead and pv_lookahead_candidates > 0 and
                not 0 < pv_lookahead_launches <= pv_lookahead_candidates):
            expected[(component,
                      "attention_cluster_pv_matrix_lookahead_launches.range")] = (
                pv_lookahead_candidates
            )
            observed[(component,
                      "attention_cluster_pv_matrix_lookahead_launches.range")] = (
                pv_lookahead_launches
            )
        if kv_distribution:
            groupctrl = f"core{core}:rocc:group_ctrl"
            expected[(groupctrl, "attention_kv_distribution_requests")] = (
                physical_kv_jobs
            )
            for statistic in (
                "attention_kv_distribution_manager_loads",
                "attention_kv_distribution_manager_bytes",
                "attention_kv_distribution_coalesced",
                "attention_kv_distribution_deliveries",
                "attention_kv_distribution_delivery_bytes",
                "attention_kv_distribution_slot_stalls",
                "attention_kv_distribution_cancels",
                "attention_kv_distribution_max_slots",
                "attention_kv_manager_lookahead_loads",
                "attention_kv_manager_lookahead_hits",
            ):
                expected[(groupctrl, statistic)] = 0
        if attention_cluster and pv_v_tile_reuse:
            v_tile = {
                "hits": 0,
                "misses": physical_kv_jobs,
                "bytes_read": physical_kv_jobs * activity["v_tile_bytes"],
                "bytes_reused": 0,
                "wait_ticks": 0,
                "capacity_rejections": 0,
                "group_hits": 0,
            }
        else:
            v_tile = expected_v_tile_buffer_activity(
                activity, pv_v_tile_reuse, pv_v_tile_buffer_bytes,
                pv_v_tile_buffer_hit_ticks, pv_v_tile_buffer_bytes_per_cycle,
                pv_v_tile_group_retention, physical_kv_jobs,
            )
        for suffix, value in v_tile.items():
            expected[(component, f"attention_pv_v_tile_buffer_{suffix}")] = value
        qk_array_ops = activity["jobs"] * 32 if attention_cluster else activity["qk"]
        pv_array_ops = activity["jobs"] * 32 if attention_cluster else activity["pv"]
        expected[(component, "attention_qk_array_ops")] = qk_array_ops
        expected[(component, "attention_pv_array_ops")] = pv_array_ops
        expected[(component, "attention_generic_gemm_qk_ops")] = (
            qk_array_ops if generic_gemm else 0
        )
        expected[(component, "attention_generic_gemm_pv_ops")] = (
            pv_array_ops if generic_gemm else 0
        )
        sequential_qk_waves = (
            activity["jobs"] * activity.get("reduction_panels", 1)
            if sequential_64 else 0
        )
        sequential_pv_waves = (
            activity["jobs"] * activity["dimension_panels"]
            if sequential_64 else 0
        )
        expected[(component, "attention_sequential_qk_waves")] = (
            sequential_qk_waves
        )
        expected[(component, "attention_sequential_pv_waves")] = sequential_pv_waves
        expected[(component, "attention_sequential_qk_active_arrays")] = (
            qk_array_ops if sequential_64 else 0
        )
        expected[(component, "attention_sequential_pv_active_arrays")] = (
            pv_array_ops if sequential_64 else 0
        )
        if sequential_64:
            expected_counts[(component, "attention_sequential_qk_active_arrays")] = (
                sequential_qk_waves
            )
            expected_counts[(component, "attention_sequential_pv_active_arrays")] = (
                sequential_pv_waves
            )
        expected[(component, "attention_pv_active_k_launches")] = (
            pv_array_ops if effective_pv_active_k else 0
        )
        expected[(component, "attention_pv_active_k_columns")] = (
            pv_array_ops * active_columns if effective_pv_active_k else 0
        )
        expected[(component, "attention_pv_active_k_matrix_elements")] = (
            ((physical_kv_jobs - pv_lookahead_launches) * 2 * 64
             * active_columns)
            if (attention_cluster and effective_pv_active_k and
                cluster_pv_matrix_lookahead) else
            (physical_kv_jobs * 2 * 64 * active_columns
             if attention_cluster and effective_pv_active_k else
            (activity["pv"] * 16 * active_columns
             if effective_pv_active_k else 0))
        )
        resident_panels = activity["jobs"] * (
            activity["dimension_panels"] - 1
        )
        resident_rows = resident_panels * 16
        accumulator_rows = activity["pv"] * (
            activity["jobs"] - activity["qblocks"]
        ) // activity["jobs"]
        expected[(component, "attention_pv_input_residency_hits")] = (
            resident_panels if pv_input_residency else 0
        )
        expected[(component, "attention_pv_input_residency_rows_reused")] = (
            resident_rows if pv_input_residency else 0
        )
        expected[(component, "attention_pv_input_residency_invalidations")] = (
            activity["jobs"] - 1 if pv_input_residency else 0
        )
        expected[(component, "attention_o_accumulator_stores")] = (
            accumulator_rows if o_accumulator_cbuffer else 0
        )
        expected[(component, "attention_o_accumulator_restores")] = (
            accumulator_rows if o_accumulator_cbuffer else 0
        )
        expected[(component, "attention_o_accumulator_bytes")] = (
            accumulator_rows * 2 * 64 if o_accumulator_cbuffer else 0
        )
        expected[(component, "attention_kv_pair_reuse_tiles")] = pair_reuse_tiles
        expected[(component, "attention_kv_pair_reuse_bytes")] = (
            pair_reuse_tiles * 2 * activity["v_tile_bytes"]
        )
        array_component = f"{component}:array"
        active_cycles = math.ceil(
            active_columns / array_mac_per_cu_per_cycle
        ) + array_pipeline_depth
        full_cycles = math.ceil(
            full_columns / array_mac_per_cu_per_cycle
        ) + array_pipeline_depth
        expected[(array_component, "active_k_launches")] = (
            pv_array_ops if effective_pv_active_k else 0
        )
        expected[(array_component, "active_k_columns")] = (
            pv_array_ops * active_columns if effective_pv_active_k else 0
        )
        expected[(array_component, "active_k_compute_cycles")] = (
            pv_array_ops * active_cycles if effective_pv_active_k else 0
        )
        expected[(array_component, "active_k_full_width_cycles_avoided")] = (
            pv_array_ops * max(full_cycles - active_cycles, 0)
            if effective_pv_active_k else 0
        )
        if generic_gemm:
            wcp_component = f"{component}:worker_command_processor"
            gemm_completions = qk_array_ops + pv_array_ops
            if attention_cluster:
                qk_lanes = attention_cluster_qk_arrays // 2
                pv_lanes = (64 - attention_cluster_qk_arrays) // 2
                qk_waves = (16 + qk_lanes - 1) // qk_lanes
                pv_waves = (16 + pv_lanes - 1) // pv_lanes
                gemm_launches = activity["jobs"] * (
                    qk_waves + (16 if cluster_pv_row_wavefront else pv_waves)
                )
            elif sequential_64:
                gemm_launches = sequential_qk_waves + sequential_pv_waves
            else:
                gemm_launches = gemm_completions
            expected[(wcp_component, "gemm_proxy_launch_commands")] = gemm_launches
            expected[(wcp_component, "gemm_proxy_completion_callbacks")] = gemm_completions
            expected[(wcp_component, "gemm_proxy_completion_delay_cycles")] = (
                gemm_completions * wcp_gemm_proxy_completion_latency_cycles
            )
            expected[(wcp_component, "gemm_proxy_queue_full_stalls")] = 0
            expected_counts[(wcp_component, "gemm_proxy_launch_commands")] = gemm_launches
            expected_counts[(wcp_component, "gemm_proxy_completion_callbacks")] = gemm_completions
            expected_counts[(wcp_component, "gemm_proxy_completion_delay_cycles")] = (
                gemm_completions
                if wcp_gemm_proxy_completion_latency_cycles > 0 else 0
            )
            qk_score_tiles = activity["qk"] // 16 if qk_score_row_burst else 0
            qk_storage_ops = activity["qk"] if qk_score_row_burst else 0
            expected[(wcp_component, "attention_tile_storage_acquires")] = qk_score_tiles
            expected[(wcp_component, "attention_tile_storage_releases")] = qk_score_tiles
            expected[(wcp_component, "attention_tile_storage_mode_conflicts")] = 0
            expected[(wcp_component, "attention_tile_storage_capacity_rejections")] = 0
            expected[(wcp_component, "attention_tile_storage_column_writes")] = qk_storage_ops
            expected[(wcp_component, "attention_tile_storage_row_reads")] = qk_storage_ops
            expected[(wcp_component, "attention_tile_storage_write_bytes")] = qk_storage_ops * 64
            expected[(wcp_component, "attention_tile_storage_read_bytes")] = qk_storage_ops * 64
            expected_counts[(wcp_component, "attention_tile_storage_acquires")] = qk_score_tiles
            expected_counts[(wcp_component, "attention_tile_storage_releases")] = qk_score_tiles
            expected_counts[(wcp_component, "attention_tile_storage_mode_conflicts")] = 0
            expected_counts[(wcp_component, "attention_tile_storage_capacity_rejections")] = 0
            expected_counts[(wcp_component, "attention_tile_storage_column_writes")] = qk_storage_ops
            expected_counts[(wcp_component, "attention_tile_storage_row_reads")] = qk_storage_ops
            expected_counts[(wcp_component, "attention_tile_storage_write_bytes")] = qk_storage_ops
            expected_counts[(wcp_component, "attention_tile_storage_read_bytes")] = qk_storage_ops
            expected_counts[(wcp_component, "attention_tile_storage_write_wait_cycles")] = qk_storage_ops
            expected_counts[(wcp_component, "attention_tile_storage_read_wait_cycles")] = qk_storage_ops
            session_count = activity["qblocks"] if o_accumulator_cbuffer else 0
            accumulator_ops = accumulator_rows if o_accumulator_cbuffer else 0
            expected[(wcp_component, "attention_storage_session_acquires")] = session_count
            expected[(wcp_component, "attention_storage_session_releases")] = session_count
            expected[(wcp_component, "attention_accumulator_row_writes")] = accumulator_ops
            expected[(wcp_component, "attention_accumulator_row_reads")] = accumulator_ops
            expected[(wcp_component, "attention_accumulator_write_bytes")] = accumulator_ops * 64
            expected[(wcp_component, "attention_accumulator_read_bytes")] = accumulator_ops * 64
            expected_counts[(wcp_component, "attention_storage_session_acquires")] = session_count
            expected_counts[(wcp_component, "attention_storage_session_releases")] = session_count
            expected_counts[(wcp_component, "attention_accumulator_row_writes")] = accumulator_ops
            expected_counts[(wcp_component, "attention_accumulator_row_reads")] = accumulator_ops
            expected_counts[(wcp_component, "attention_accumulator_write_bytes")] = accumulator_ops
            expected_counts[(wcp_component, "attention_accumulator_read_bytes")] = accumulator_ops
            expected_counts[(wcp_component, "attention_accumulator_write_wait_cycles")] = accumulator_ops
            expected_counts[(wcp_component, "attention_accumulator_read_wait_cycles")] = accumulator_ops
        expected[(component, "attention_sp_hbm_bytes")] = 0
        expected[(component, "attention_qk_early_compute_arrays")] = (
            0 if attention_cluster else
            (activity["rows"] if qk_dataflow_transpose else
             activity["qk"] - ahead_operand_tiles * 16)
            if qk_early_compute else 0
        )
        expected[(component, "attention_qk_input_pipeline_rows_fetched")] = (
            activity["qk"] - ahead_operand_tiles * 16
            if qk_input_pipeline else 0
        )
        expected[(component, "attention_qk_input_pipeline_rows_programmed")] = (
            activity["qk"] - ahead_operand_tiles * 16
            if qk_input_pipeline else 0
        )
        expected[(component, "attention_qk_input_pipeline_tag_mismatches")] = 0
        expected[(component, "attention_qk_input_pipeline_max_depth")] = (
            activity["qk"] // 8 - ahead_operand_tiles * 2
            if qk_input_pipeline else 0
        )
        expected_counts[(component, "attention_qk_input_pipeline_max_depth")] = (
            activity["qk"] // 16 - ahead_operand_tiles
            if qk_input_pipeline else 0
        )
        expected[(component, "attention_cross_tile_operand_candidates")] = (
            ahead_operand_tiles
        )
        expected[(component, "attention_cross_tile_operand_launches")] = (
            ahead_operand_tiles
        )
        expected[(component, "attention_cross_tile_operand_matrix_programs")] = (
            ahead_operand_tiles
        )
        expected[(component, "attention_cross_tile_operand_input_rows")] = (
            ahead_operand_tiles * 16
        )
        expected[(component, "attention_cross_tile_operand_promotions")] = (
            ahead_operand_tiles
        )
        expected[(component, "attention_cross_tile_operand_tag_mismatches")] = 0
        expected_prefetches = physical_kv_jobs - kv_query_groups
        if kv_double_buffer and kv_cross_query_prefetch:
            expected_prefetches += max(kv_query_groups - attention_job_count, 0)
        expected[(component, "attention_kv_prefetch_tiles")] = (
            expected_prefetches if kv_double_buffer else 0
        )
        expected_counts[(component, "attention_kv_prefetch_dma_ticks")] = (
            expected_prefetches if kv_double_buffer else 0
        )
        expected_counts[(component, "attention_kv_k_release_ticks")] = (
            (physical_kv_jobs if attention_cluster else activity["jobs"])
            if kv_double_buffer else 0
        )
        expected_counts[(component, "attention_kv_v_release_ticks")] = (
            activity["jobs"] if kv_double_buffer else 0
        )
        programmed_input_rows = (
            activity["jobs"] * 16 if attention_cluster else
            activity["jobs"] * 16 if pv_input_residency else activity["pv"]
        )
        expected[(component, "attention_pv_input_pipeline_rows")] = (
            programmed_input_rows if pv_input_pipeline else 0
        )
        restore_rows = activity["pv"] * (
            activity["jobs"] - activity["qblocks"]
        ) // activity["jobs"]
        expected[(component, "attention_pv_restore_pipeline_rows")] = (
            restore_rows if pv_restore_pipeline else 0
        )
        expected[(component, "attention_pv_output_pipeline_rows")] = (
            activity["pv"] if pv_output_pipeline else 0
        )
        expected[(component, "attention_pv_early_compute_arrays")] = (
            activity["pv"] if pv_early_compute else 0
        )
        expected[(component, "attention_pv_matrix_overlap_tiles")] = (
            activity["jobs"] if pv_matrix_softmax_overlap else 0
        )
        expected[(component, "attention_qk_matrix_broadcasts")] = (
            physical_kv_jobs * 2 if attention_cluster else
            activity["jobs"] * activity.get("reduction_panels", 1)
            if sequential_64 and qk_matrix_broadcast else
            (activity["qk"] // 16 if qk_dataflow_transpose else
             activity["jobs"] - ahead_operand_tiles)
            if qk_matrix_broadcast else 0
        )
        expected[(component, "attention_pv_matrix_broadcasts")] = (
            0 if attention_cluster else
            activity["jobs"] * activity["dimension_panels"]
            if sequential_64 and pv_matrix_broadcast else
            activity["pv"] // 16 if pv_matrix_broadcast else 0
        )
        if attention_cluster:
            cluster_contexts = activity["jobs"] * 3 + activity["qblocks"]
            cluster_stats = {
                "attention_cluster_worker_jobs": 1,
                "attention_cluster_contexts_issued": cluster_contexts,
                "attention_cluster_contexts_completed": cluster_contexts,
                "attention_cluster_contexts_cancelled": 0,
                "attention_cluster_bank_refs_cancelled": 0,
                "attention_cluster_memory_requests_cancelled": 0,
                "attention_cluster_stale_callbacks": 0,
                "attention_cluster_illegal_transitions": 0,
                "attention_cluster_score_slot_reservations": activity["jobs"],
                "attention_cluster_score_slot_releases": activity["jobs"],
                "attention_cluster_score_slot_full_stalls": 0,
                "attention_cluster_qk_k_panel_broadcasts": (
                    physical_kv_jobs * 2
                ),
                "attention_cluster_qk_k_panel_bytes": (
                    physical_kv_jobs * 2 * 64 * 64 * 4
                ),
                "attention_cluster_qk_q_pair_multicasts": activity["jobs"] * 32,
                "attention_cluster_qk_q_pair_bytes": (
                    activity["jobs"] * 32 * 64 * 4
                ),
                "attention_cluster_qk_score_beats": activity["jobs"] * 16,
                "attention_cluster_qk_score_bytes": activity["qk"] * 64,
                "attention_cluster_ahead_contexts_launched": (
                    activity["jobs"] - physical_kv_jobs
                ),
                "attention_cluster_ahead_contexts_completed": (
                    activity["jobs"] - physical_kv_jobs
                ),
                "attention_cluster_ahead_contexts_promoted": (
                    activity["jobs"] - physical_kv_jobs
                ),
                "attention_cluster_qk_tile_starts": activity["jobs"],
                "attention_cluster_o_context_reservations": activity["qblocks"],
                "attention_cluster_o_context_releases": activity["qblocks"],
                "attention_cluster_o_context_cancelled": 0,
                "attention_cluster_o_scale_segments": (
                    (activity["jobs"] - activity["qblocks"])
                    * 16 * activity["dimension_panels"]
                ),
                "attention_cluster_o_accumulate_segments": (
                    activity["jobs"] * 16 * activity["dimension_panels"]
                ),
                "attention_cluster_o_fused_rows": (
                    activity["jobs"] * 16 if pv_o_row_fusion else 0
                ),
                "attention_cluster_o_fused_bytes": (
                    activity["jobs"] * 16 * 128 * 4
                    if pv_o_row_fusion else 0
                ),
                "attention_cluster_pv_wavefront_rows": (
                    activity["jobs"] * 16 if cluster_pv_row_wavefront else 0
                ),
                "attention_cluster_qk_matrix_lookahead_launches": (
                    qk_lookahead_launches if cluster_qk_matrix_lookahead else 0
                ),
                "attention_cluster_qk_matrix_lookahead_hits": (
                    qk_lookahead_launches if cluster_qk_matrix_lookahead else 0
                ),
                "attention_cluster_pv_matrix_lookahead_launches": (
                    pv_lookahead_launches
                    if cluster_pv_matrix_lookahead else 0
                ),
                "attention_cluster_pv_matrix_lookahead_hits": (
                    pv_lookahead_launches
                    if cluster_pv_matrix_lookahead else 0
                ),
                "attention_cluster_o_drain_requests": activity["qblocks"],
                "attention_cluster_o_drain_bytes": (
                    activity["qblocks"] * 16 * activity["dimension_panels"] * 64
                ),
            }
            for statistic, value in cluster_stats.items():
                expected[(component, statistic)] = value
            expected_counts[(component, "attention_cluster_config_fingerprint")] = 1
            classified_array_stats = {
                "attention_cluster_qk_k_matrix_requests": (
                    physical_kv_jobs * 2
                ),
                "attention_cluster_qk_k_matrix_bytes": (
                    physical_kv_jobs * 2 * 64 * 64 * 4
                ),
                "attention_cluster_qk_q_pair_requests": activity["jobs"] * 32,
                "attention_cluster_qk_q_pair_bytes": (
                    activity["jobs"] * 32 * 64 * 4
                ),
                "attention_cluster_qk_score_out_requests": activity["jobs"] * 16,
                "attention_cluster_qk_score_out_bytes": (
                    activity["jobs"] * 16 * 2 * 64 * 4
                ),
                "attention_cluster_pv_group_drains": activity["jobs"] * 16,
                "attention_cluster_pv_group_drain_bytes": (
                    activity["jobs"] * 16 * activity["dimension_panels"] * 64
                ),
                "attention_cluster_pv_group_drain_cycles": (
                    activity["jobs"] * 16 * (
                        array_buffer_base_latency_cycles +
                        (
                            activity["dimension_panels"] * 64 +
                            near_array_output_bytes_per_cycle - 1
                        ) // near_array_output_bytes_per_cycle
                    )
                ),
            }
            for statistic, value in classified_array_stats.items():
                expected[(array_component, statistic)] = value
        elif sequential_64:
            score_readout = expected_sequential_qk_score_readout(activity)
            expected[(
                array_component,
                "attention_cluster_qk_score_out_requests",
            )] = score_readout["requests"]
            expected[(
                array_component,
                "attention_cluster_qk_score_out_bytes",
            )] = score_readout["bytes"]
            scatter_stats = {
                "input_scatter_requests": scatter["requests"],
                "input_scatter_rejected": 0,
                "input_scatter_bytes": scatter["bytes"],
                "input_scatter_transfer_cycles": scatter["transfer_cycles"],
                "input_scatter_destinations": scatter["destinations"],
            }
            for statistic, value in scatter_stats.items():
                expected[(array_component, statistic)] = value
            expected_counts[(array_component, "input_scatter_destinations")] = (
                scatter["requests"]
            )
            output_stats = {
                "output_scatter_requests":
                    output_scatter_gather["scatter_requests"],
                "output_gather_requests":
                    output_scatter_gather["gather_requests"],
                "output_scatter_gather_rejected": 0,
                "output_scatter_gather_bytes":
                    output_scatter_gather["bytes"],
                "output_scatter_gather_transfer_cycles":
                    output_scatter_gather["transfer_cycles"],
                "output_scatter_gather_destinations":
                    output_scatter_gather["destinations"],
            }
            for statistic, value in output_stats.items():
                expected[(array_component, statistic)] = value
            expected_counts[(
                array_component, "output_scatter_gather_destinations"
            )] = output_scatter_gather["requests"]
        if broadcast["enabled"]:
            broadcast_stats = {
                "matrix_broadcast_requests": broadcast["requests"],
                "matrix_broadcast_rejected": broadcast["rejected"],
                "matrix_broadcast_ingress_bytes": broadcast["ingress_bytes"],
                "matrix_broadcast_sink_bytes": broadcast["sink_bytes"],
                "matrix_broadcast_transfer_cycles": broadcast["transfer_cycles"],
                "matrix_broadcast_fanout": broadcast["fanout_sum"],
            }
            for statistic, value in broadcast_stats.items():
                expected[(array_component, statistic)] = value
            expected_counts[(array_component, "matrix_broadcast_fanout")] = (
                broadcast["requests"]
            )
        expected_counts[(component, "attention_worker_dispatch_accept_tick")] = attention_job_count
        expected_counts[(component, "attention_worker_qk_tile_complete_tick")] = (
            activity["jobs"]
        )
        expected_counts[(component, "attention_worker_softmax_tile_complete_tick")] = (
            activity["jobs"]
        )
        expected_counts[(component, "attention_worker_pv_tile_complete_tick")] = (
            activity["jobs"]
        )
        expected_counts[(component, "attention_worker_output_dma_ack_tick")] = (
            activity["qblocks"]
        )
        for stat in (
            "attention_worker_intertile_total_ticks",
            *(statistic for _, statistic in INTER_TILE_PHASE_STATS),
        ):
            expected_counts[(component, stat)] = (
                physical_kv_jobs - attention_job_count if attention_cluster
                else activity["jobs"] - attention_job_count
            )
        for stat in (
            "attention_worker_tile_total_ticks",
            *(statistic for _, statistic in TILE_PIPELINE_PHASE_STATS),
        ):
            expected_counts[(component, stat)] = activity["jobs"]
        expected[(f"core{core}:rocc:sfu", "sfu_attention_jobs")] = activity["jobs"]
        expected[(f"core{core}:rocc:sfu", "sfu_softmax_rows")] = activity["rows"]
        expected[(f"core{core}:rocc:sfu", "sfu_attention_scaled_elements")] = activity["scaled"]
        if attention_cluster:
            sfu_component = f"core{core}:rocc:sfu"
            expected[(sfu_component, "attention_cluster_score_fifo_producer_writes")] = (
                activity["jobs"] * 16
            )
            expected[(sfu_component, "attention_cluster_score_fifo_producer_bytes")] = (
                activity["qk"] * 64
            )
            expected[(sfu_component, "attention_cluster_score_fifo_reads")] = (
                activity["qk"] * 3
            )
            expected[(sfu_component, "attention_cluster_score_fifo_read_bytes")] = (
                activity["qk"] * 3 * 64
            )
            expected[(sfu_component, "attention_cluster_score_fifo_internal_writes")] = (
                activity["qk"] * 2
            )
            expected[(sfu_component, "attention_cluster_score_fifo_internal_write_bytes")] = (
                activity["qk"] * 2 * 64
            )
            expected[(sfu_component, "attention_cluster_score_fifo_backpressure_stalls")] = 0
            p_reads = activity["jobs"] * 16
            p_stats = {
                "attention_cluster_p_fifo_reservations": activity["jobs"],
                "attention_cluster_p_fifo_releases": activity["jobs"],
                "attention_cluster_p_fifo_cancelled": 0,
                "attention_cluster_p_fifo_writes": (
                    activity["jobs"] * 16 * active_columns // 16
                ),
                "attention_cluster_p_fifo_write_bytes": (
                    activity["jobs"] * 16 * active_columns * 4
                ),
                "attention_cluster_p_fifo_reads": p_reads,
                "attention_cluster_p_fifo_read_bytes": p_reads * active_columns * 4,
                "attention_cluster_p_fifo_backpressure_stalls": 0,
            }
            for statistic, value in p_stats.items():
                if statistic != "attention_cluster_p_fifo_backpressure_stalls":
                    expected[(sfu_component, statistic)] = value
            expected_counts[(component, "attention_cluster_qk_tile_ii_cycles")] = (
                cluster_ii_counts["steady"]
            )
            expected_counts[(component, "attention_cluster_qk_boundary_ii_cycles")] = (
                cluster_ii_counts["boundary"]
            )
    mismatches = {
        f"{component}/{stat}": {"expected": value, "actual": observed.get((component, stat))}
        for (component, stat), value in expected.items()
        if observed.get((component, stat)) != value
    }
    mismatches.update({
        f"{component}/{stat}.Count": {
            "expected": value,
            "actual": counts.get((component, stat)),
        }
        for (component, stat), value in expected_counts.items()
        if counts.get((component, stat)) != value
    })
    if not attention_cluster:
        for (component, statistic), value in observed.items():
            if ((component, statistic) not in expected and
                    "attention_cluster" in statistic and value != 0):
                mismatches[f"{component}/{statistic}.disabled"] = {
                    "expected": 0, "actual": value,
                }
    else:
        def require_interval(component, prefix, suffix):
            union = observed.get((component, f"{prefix}_busy_union_{suffix}"), 0)
            span = observed.get((component, f"{prefix}_busy_span_{suffix}"), 0)
            idle = observed.get((component, f"{prefix}_idle_gap_{suffix}"), 0)
            concurrency = observed.get((component, f"{prefix}_max_concurrency"), 0)
            if union <= 0 or span < union or idle != span - union or concurrency <= 0:
                mismatches[f"{component}/{prefix}_interval"] = {
                    "expected": "union>0, span>=union, idle=span-union, concurrency>0",
                    "actual": {
                        "union": union, "span": span, "idle": idle,
                        "concurrency": concurrency,
                    },
                }

        for core in range(4, 20):
            component = f"core{core}:rocc"
            fingerprint = observed.get(
                (component, "attention_cluster_config_fingerprint"), 0
            )
            if fingerprint <= 0:
                mismatches[f"{component}/attention_cluster_config_fingerprint"] = {
                    "expected": ">0", "actual": fingerprint,
                }
            for statistic in (
                "attention_cluster_context_high_water",
                "attention_cluster_score_slot_high_water",
            ):
                maximum = maxima.get((component, statistic), 0)
                # Context high-water includes the resident O context plus
                # transient QK/score/PV contexts.  The cluster has four of
                # each context kind, hence 16 total; score FIFO remains a
                # separate four-slot contract checked by its own statistic.
                upper_bound = 16 if statistic.endswith("context_high_water") else 4
                if not 1 <= maximum <= upper_bound:
                    mismatches[f"{component}/{statistic}.bound"] = {
                        "expected": f"1..{upper_bound}", "actual": maximum,
                    }
            expected_concurrency = {
                "attention_cluster_qk_array_max_concurrency": min(
                    attention_cluster_qk_arrays, 32
                ),
            }
            if not cluster_pv_row_wavefront:
                expected_concurrency["attention_cluster_pv_array_max_concurrency"] = min(
                    64 - attention_cluster_qk_arrays, 32
                )
            for statistic, expected_max in expected_concurrency.items():
                actual = observed.get((component, statistic), 0)
                if actual != expected_max:
                    mismatches[f"{component}/{statistic}"] = {
                        "expected": expected_max, "actual": actual,
                    }
            if cluster_pv_row_wavefront:
                pv_concurrency = observed.get(
                    (component, "attention_cluster_pv_array_max_concurrency"), 0
                )
                pv_limit = min(64 - attention_cluster_qk_arrays, 32)
                if not 1 <= pv_concurrency <= pv_limit:
                    mismatches[
                        f"{component}/attention_cluster_pv_array_max_concurrency.bound"
                    ] = {
                        "expected": f"1..{pv_limit}", "actual": pv_concurrency,
                    }
            o_high_water = maxima.get(
                (component, "attention_cluster_o_context_high_water"), 0
            )
            expected_o_high_water = min(
                activity["qblocks"], effective_kv_query_group_size, 4
            )
            if o_high_water != expected_o_high_water:
                mismatches[
                    f"{component}/attention_cluster_o_context_high_water"
                ] = {"expected": expected_o_high_water, "actual": o_high_water}
            require_interval(component, "attention_cluster_qk_array", "ticks")
            require_interval(component, "attention_cluster_pv_array", "ticks")
            array_component = f"{component}:array"
            require_interval(
                array_component, "attention_cluster_buffer", "ticks"
            )
            memory_component = f"{component}:global_memory"
            require_interval(
                memory_component, "attention_cluster_local_read", "ticks"
            )
            require_interval(
                memory_component, "attention_cluster_local_write", "ticks"
            )
            require_interval(
                f"{component}:sfu", "attention_cluster_sfu", "ticks"
            )
            require_interval(component, "attention_cluster_o_read", "cycles")
            require_interval(component, "attention_cluster_o_write", "cycles")
            require_interval(component, "attention_cluster_o_alu", "cycles")
            ahead_contexts = activity["jobs"] - physical_kv_jobs
            promotion_waits = observed.get(
                (component, "attention_cluster_promotion_waits"), 0
            )
            if promotion_waits > ahead_contexts:
                mismatches[f"{component}/attention_cluster_promotion_waits.bound"] = {
                    "expected": f"0..{ahead_contexts}", "actual": promotion_waits,
                }
            # Row fusion removes the serialized O-submit tail that previously
            # kept PV active into the following SFU interval. QK/PV overlap
            # remains the durable cross-tile pipeline contract.
            pair_overlap_stats = ["attention_cluster_qk_pv_overlap_cycles"]
            if not pv_o_row_fusion:
                pair_overlap_stats.append(
                    "attention_cluster_sfu_pv_overlap_cycles"
                )
            for statistic in pair_overlap_stats:
                actual = observed.get((component, statistic), 0)
                if ahead_contexts > 0 and actual <= 0:
                    mismatches[f"{component}/{statistic}"] = {
                        "expected": ">0", "actual": actual,
                    }
                elif ahead_contexts == 0 and actual != 0:
                    mismatches[f"{component}/{statistic}"] = {
                        "expected": 0, "actual": actual,
                    }
        three_stage_overlap = sum(
            observed.get(
                (f"core{core}:rocc",
                 "attention_cluster_three_stage_overlap_cycles"), 0
            )
            for core in range(4, 20)
        )
        if activity["jobs"] == physical_kv_jobs:
            if three_stage_overlap != 0:
                mismatches["attention_cluster_three_stage_overlap_cycles.total"] = {
                    "expected": 0, "actual": three_stage_overlap,
                }
        # With a 64-cycle QK wave, QK can complete between SFU sampling
        # points.  Keep three-stage overlap as a reported efficiency metric;
        # the required concurrency contracts are the two durable PV overlaps.
    for core in range(4, 20):
        component = f"core{core}:rocc"
        ready = observed.get(
            (component, "attention_cross_tile_operand_ready_hits"), 0
        )
        waits = observed.get(
            (component, "attention_cross_tile_operand_waits"), 0
        )
        if ready + waits != ahead_operand_tiles:
            mismatches[f"{component}/attention_cross_tile_operand_retirement"] = {
                "expected": ahead_operand_tiles,
                "actual": {"ready": ready, "waits": waits},
            }
        wait_tick_count = counts.get(
            (component, "attention_cross_tile_operand_wait_ticks"), 0
        )
        if wait_tick_count != waits:
            mismatches[f"{component}/attention_cross_tile_operand_wait_ticks.Count"] = {
                "expected": waits,
                "actual": wait_tick_count,
            }
    if broadcast["enabled"]:
        for core in range(4, 20):
            component = f"core{core}:rocc:array"
            key = (component, "matrix_broadcast_fanout")
            actual_range = [minima.get(key), maxima.get(key)]
            expected_range = [
                broadcast["min_fanout"],
                broadcast["max_observed_fanout_expected"],
            ]
            if actual_range != expected_range:
                mismatches[f"{component}/matrix_broadcast_fanout.range"] = {
                    "expected": expected_range,
                    "actual": actual_range,
                }
    if sequential_64:
        worker_rows = activity.get("query_rows_per_worker", 64)
        query_block_rows = activity.get("query_block_rows", 64)
        tail_rows = worker_rows % query_block_rows
        pv_min = tail_rows if tail_rows else query_block_rows
        pv_max = min(worker_rows, query_block_rows)
        for core in range(4, 20):
            component = f"core{core}:rocc"
            for statistic in (
                    "attention_sequential_qk_active_arrays",
                    "attention_sequential_pv_active_arrays"):
                key = (component, statistic)
                actual_range = [minima.get(key), maxima.get(key)]
                expected_active_range = (
                    [64, 64] if statistic.endswith("qk_active_arrays")
                    else [pv_min, pv_max]
                )
                if actual_range != expected_active_range:
                    mismatches[f"{component}/{statistic}.range"] = {
                        "expected": expected_active_range,
                        "actual": actual_range,
                    }
    if generic_gemm:
        for core in range(4, 20):
            component = f"core{core}:rocc:worker_command_processor"
            commands = observed.get((component, "gemm_proxy_commands_issued"))
            completions = activity["qk"] + activity["pv"]
            launches = (
                activity["jobs"] *
                (activity["qk"] // activity["jobs"] // 8 + 16)
                if attention_cluster else
                activity["jobs"] * (
                    activity.get("reduction_panels", 1) +
                    activity["dimension_panels"]
                ) if sequential_64 else completions
            )
            if commands is None or commands < launches:
                mismatches[f"{component}/gemm_proxy_commands_issued"] = {
                    "expected": f">={launches}",
                    "actual": commands,
                }
                continue
            queue_wait_count = counts.get(
                (component, "gemm_proxy_queue_wait_cycles")
            )
            if queue_wait_count != commands:
                mismatches[f"{component}/gemm_proxy_queue_wait_cycles.Count"] = {
                    "expected": commands,
                    "actual": queue_wait_count,
                }
            queue_wait_sum = observed.get(
                (component, "gemm_proxy_queue_wait_cycles"), 0
            )
            minimum_wait_sum = commands * wcp_gemm_proxy_command_latency_cycles
            if queue_wait_sum < minimum_wait_sum:
                mismatches[f"{component}/gemm_proxy_queue_wait_cycles.Sum"] = {
                    "expected": f">={minimum_wait_sum}",
                    "actual": queue_wait_sum,
                }
            minimum_burst_wait = 0 if qk_early_compute else (
                wcp_gemm_proxy_command_latency_cycles
                if sequential_64 else
                wcp_gemm_proxy_command_latency_cycles
                + (16 + wcp_gemm_proxy_issue_width - 1)
                // wcp_gemm_proxy_issue_width
                - 1
            )
            queue_wait_max = maxima.get(
                (component, "gemm_proxy_queue_wait_cycles"), 0
            )
            if queue_wait_max < minimum_burst_wait:
                mismatches[f"{component}/gemm_proxy_issue_width"] = {
                    "expected_max_queue_wait_at_least": minimum_burst_wait,
                    "actual_max_queue_wait": queue_wait_max,
                }
    qk_readout_max_ahead = 0
    for core in range(4, 20):
        key = (f"core{core}:rocc", "attention_qk_readout_ahead_depth")
        actual_max = maxima.get(key, 0)
        qk_readout_max_ahead = max(qk_readout_max_ahead, actual_max)
        actual_count = counts.get(key, 0)
        if qk_readout_overlap and (
                actual_count != activity["qk"] or
                not 1 <= actual_max <= qk_readout_window):
            mismatches[f"{key[0]}/{key[1]}.bound"] = {
                "expected": {
                    "count": activity["qk"],
                    "max": f"1..{qk_readout_window}",
                },
                "actual": {"count": actual_count, "max": actual_max},
            }
        if not qk_readout_overlap and actual_count != 0:
            mismatches[f"{key[0]}/{key[1]}.disabled"] = {
                "expected": 0, "actual": actual_count,
            }
    for core in range(4, 20):
        component = f"core{core}:rocc"
        expected_consumed = (
            physical_kv_jobs - kv_query_groups if kv_double_buffer else 0
        )
        if kv_double_buffer and kv_cross_query_prefetch:
            expected_consumed += max(kv_query_groups - attention_job_count, 0)
        hits = observed.get((component, "attention_kv_prefetch_hits"), 0)
        waits = observed.get((component, "attention_kv_prefetch_waits"), 0)
        if hits + waits != expected_consumed:
            mismatches[f"{component}/attention_kv_prefetch_consumed"] = {
                "expected": expected_consumed,
                "actual": hits + waits,
            }
        timing_counts = (
            ("attention_kv_prefetch_ready_lead_ticks", hits),
            ("attention_kv_prefetch_wait_ticks", waits),
        )
        for statistic, expected_count in timing_counts:
            if counts.get((component, statistic)) != expected_count:
                mismatches[f"{component}/{statistic}.Count"] = {
                    "expected": expected_count,
                    "actual": counts.get((component, statistic)),
                }
        candidates = observed.get(
            (component, "attention_kv_second_lookahead_candidates"), 0
        )
        second_prefetches = observed.get(
            (component, "attention_kv_second_lookahead_prefetches"), 0
        )
        ready_at_release = observed.get(
            (component, "attention_kv_next_ready_at_release_tiles"), 0
        )
        candidate_count = counts.get(
            (component, "attention_kv_second_lookahead_lead_ticks"), 0
        )
        max_candidates = max(physical_kv_jobs - 2 * kv_query_groups, 0)
        observation_enabled = kv_double_buffer and kv_second_lookahead
        if not observation_enabled:
            max_candidates = 0
        if candidates > max_candidates or ready_at_release > candidates or \
                candidate_count != candidates or second_prefetches != candidates:
            mismatches[f"{component}/attention_kv_second_lookahead_window"] = {
                "expected": {
                    "candidates_at_most": max_candidates,
                    "ready_at_release_at_most": candidates,
                    "lead_count": candidates,
                    "prefetches": candidates,
                },
                "actual": {
                    "candidates": candidates,
                    "ready_at_release": ready_at_release,
                    "lead_count": candidate_count,
                    "prefetches": second_prefetches,
                },
            }
        cross_prefetches = observed.get(
            (component, "attention_kv_cross_query_prefetches"), 0
        )
        cross_hits = observed.get(
            (component, "attention_kv_cross_query_hits"), 0
        )
        cross_waits = observed.get(
            (component, "attention_kv_cross_query_waits"), 0
        )
        cross_wait_ticks = observed.get(
            (component, "attention_kv_cross_query_wait_ticks"), 0
        )
        expected_cross = max(kv_query_groups - attention_job_count, 0) \
            if kv_double_buffer and kv_cross_query_prefetch else 0
        if cross_prefetches != expected_cross or cross_hits + cross_waits != expected_cross:
            mismatches[f"{component}/attention_kv_cross_query_prefetch"] = {
                "expected": {
                    "prefetches": expected_cross,
                    "consumed": expected_cross,
                },
                "actual": {
                    "prefetches": cross_prefetches,
                    "hits": cross_hits,
                    "waits": cross_waits,
                },
            }
        if counts.get((component, "attention_kv_cross_query_wait_ticks"), 0) != cross_waits:
            mismatches[f"{component}/attention_kv_cross_query_wait_ticks.Count"] = {
                "expected": cross_waits,
                "actual": counts.get((component, "attention_kv_cross_query_wait_ticks"), 0),
            }
        overlap_hits = observed.get(
            (component, "attention_pv_matrix_overlap_hits"), 0
        )
        overlap_waits = observed.get(
            (component, "attention_pv_matrix_overlap_waits"), 0
        )
        expected_overlaps = activity["jobs"] if pv_matrix_softmax_overlap else 0
        if overlap_hits + overlap_waits != expected_overlaps:
            mismatches[f"{component}/attention_pv_matrix_overlap_consumed"] = {
                "expected": expected_overlaps,
                "actual": overlap_hits + overlap_waits,
            }
    for core in range(4, 20):
        breakdown = summarize_intertile_breakdown(
            observed, counts, core, accelerator_clock_hz,
            timebase_ticks_per_second,
        )
        if breakdown and not breakdown["conservation_valid"]:
            mismatches[f"core{core}:rocc/attention_worker_intertile_conservation"] = {
                "expected": 0,
                "actual": breakdown["unattributed_ticks"],
            }
        tile_breakdown = summarize_tile_pipeline_breakdown(
            observed, counts, core, accelerator_clock_hz,
            timebase_ticks_per_second,
        )
        if tile_breakdown and not tile_breakdown["conservation_valid"]:
            mismatches[f"core{core}:rocc/attention_worker_tile_conservation"] = {
                "expected": 0,
                "actual": tile_breakdown["unattributed_ticks"],
            }
    for core in range(4, 20):
        key = (f"core{core}:rocc:sfu", "sfu_attention_rsqrt_ready_tick")
        if counts.get(key) != attention_job_count:
            mismatches[f"{key[0]}/{key[1]}.Count"] = {
                "expected": attention_job_count, "actual": counts.get(key)
            }
        worker = f"core{core}:rocc"
        worker_ticks = [
            minima.get((worker, "attention_worker_dispatch_accept_tick")),
            maxima.get((worker, "attention_worker_qk_tile_complete_tick")),
            maxima.get((worker, "attention_worker_softmax_tile_complete_tick")),
            maxima.get((worker, "attention_worker_pv_tile_complete_tick")),
            maxima.get((worker, "attention_worker_output_dma_ack_tick")),
        ]
        if all(tick is not None for tick in worker_ticks) and worker_ticks != sorted(worker_ticks):
            mismatches[f"{worker}/attention_worker_lifecycle_order"] = {
                "expected": "dispatch <= final_qk <= final_softmax <= final_pv <= output_ack",
                "actual": worker_ticks,
            }
    lifecycle_stats = (
        "attention_manager_descriptor_accept_tick",
        "attention_manager_dispatch_tick",
        "attention_manager_local_complete_tick",
        "attention_manager_wait_observed_tick",
    )
    manager_ticks = {
        stat: [
            (minima if stat == "attention_manager_descriptor_accept_tick" else maxima).get(
                (f"core{core}:rocc", stat)
            )
            for core in range(4)
        ]
        for stat in lifecycle_stats
    }
    for core in range(4):
        ticks = [manager_ticks[stat][core] for stat in lifecycle_stats]
        if all(tick is not None for tick in ticks) and ticks != sorted(ticks):
            mismatches[f"core{core}:rocc/attention_lifecycle_order"] = {
                "expected": "accept <= dispatch <= local_complete <= wait",
                "actual": ticks,
            }

    root = "core0:rocc"
    root_accept = minima.get((root, "attention_manager_descriptor_accept_tick"))
    tensor_complete = maxima.get((root, "attention_tensor_complete_tick"))
    root_wait = maxima.get((root, "attention_manager_wait_observed_tick"))
    received_key = (root, "attention_manager_band_completion_received_tick")
    lifecycle = {}
    all_manager_ticks_present = all(
        value is not None
        for values in manager_ticks.values()
        for value in values
    )
    if all(value is not None for value in (root_accept, tensor_complete, root_wait)) and \
            all_manager_ticks_present:
        if not root_accept <= tensor_complete <= root_wait:
            mismatches[f"{root}/attention_tensor_lifecycle_order"] = {
                "expected": "accept <= tensor_complete <= wait",
                "actual": [root_accept, tensor_complete, root_wait],
            }
        accept_skew_ticks = max(manager_ticks[lifecycle_stats[0]]) - min(
            manager_ticks[lifecycle_stats[0]]
        )
        local_complete_skew_ticks = max(manager_ticks[lifecycle_stats[2]]) - min(
            manager_ticks[lifecycle_stats[2]]
        )
        accelerator_completion_ticks = tensor_complete - root_accept
        wait_return_ticks = root_wait - root_accept
        lifecycle = {
            "clock_contract": make_clock_contract(
                accelerator_clock_hz, timebase_ticks_per_second,
                model_cpu_clock_hz, model_array_clock_hz,
                model_memctrl_clock_hz, model_platform_clock_hz,
            ),
            # Retained for compatibility. This is the normalization clock, not
            # a claim that all modeled accelerator components share one clock.
            "accelerator_clock_hz": accelerator_clock_hz,
            "sst_timebase_ticks_per_second": timebase_ticks_per_second,
            "root_descriptor_accept_tick": root_accept,
            "root_tensor_complete_tick": tensor_complete,
            "root_wait_observed_tick": root_wait,
            "accelerator_completion_ticks": accelerator_completion_ticks,
            "wait_return_ticks": wait_return_ticks,
            "accelerator_completion_cycles": ticks_to_cycles(
                accelerator_completion_ticks, accelerator_clock_hz,
                timebase_ticks_per_second
            ),
            "accelerator_completion_milliseconds": ticks_to_milliseconds(
                accelerator_completion_ticks, timebase_ticks_per_second
            ),
            "wait_return_cycles": ticks_to_cycles(
                wait_return_ticks, accelerator_clock_hz,
                timebase_ticks_per_second
            ),
            "wait_return_milliseconds": ticks_to_milliseconds(
                wait_return_ticks, timebase_ticks_per_second
            ),
            "manager_descriptor_accept_skew_ticks": accept_skew_ticks,
            "manager_descriptor_accept_skew_cycles": ticks_to_cycles(
                accept_skew_ticks, accelerator_clock_hz, timebase_ticks_per_second
            ),
            "manager_local_complete_skew_ticks": local_complete_skew_ticks,
            "manager_local_complete_skew_cycles": ticks_to_cycles(
                local_complete_skew_ticks, accelerator_clock_hz,
                timebase_ticks_per_second
            ),
            "root_band_receive_first_tick": minima.get(received_key),
            "root_band_receive_last_tick": maxima.get(received_key),
            "worker_critical_path": summarize_worker_critical_path(
                observed, minima, maxima, range(4, 20), accelerator_clock_hz,
                timebase_ticks_per_second,
            ),
            "system_frontier": summarize_system_frontier(
                observed, minima, maxima, accelerator_clock_hz,
                timebase_ticks_per_second,
            ),
            "kv_second_lookahead_window": summarize_kv_second_lookahead(
                observed, counts, maxima, range(4, 20),
                16 * max(activity["jobs"] - 2 * activity["qblocks"], 0)
                if kv_double_buffer and kv_second_lookahead
                else 0,
                accelerator_clock_hz, timebase_ticks_per_second,
            ),
        }
    elif all(value is not None for value in (root_accept, tensor_complete, root_wait)):
        mismatches["attention_manager_lifecycle_complete"] = {
            "expected": "all four manager lifecycle timestamp sets",
            "actual": {
                stat: [index for index, value in enumerate(values) if value is None]
                for stat, values in manager_ticks.items()
                if any(value is None for value in values)
            },
        }
    if lifecycle:
        lifecycle["qk_readout_overlap"] = {
            "enabled": qk_readout_overlap,
            "window": qk_readout_window,
            "max_observed_ahead": qk_readout_max_ahead,
            "ordered_local_gm_retirement": True,
        }
        critical_path = lifecycle["worker_critical_path"]
        if critical_path:
            critical_path["inter_tile_breakdown"] = summarize_intertile_breakdown(
                observed, counts, critical_path["slowest_worker_core"],
                accelerator_clock_hz, timebase_ticks_per_second,
            )
            critical_path["tile_pipeline_breakdown"] = \
                summarize_tile_pipeline_breakdown(
                    observed, counts, critical_path["slowest_worker_core"],
                    accelerator_clock_hz, timebase_ticks_per_second,
                )
            component = f"core{critical_path['slowest_worker_core']}:rocc"
            prefetch_stats = {
                "dma": "attention_kv_prefetch_dma_ticks",
                "ready_lead": "attention_kv_prefetch_ready_lead_ticks",
                "wait": "attention_kv_prefetch_wait_ticks",
            }
            critical_path["kv_prefetch_timing"] = {
                "ticks": {
                    label: observed.get((component, statistic), 0)
                    for label, statistic in prefetch_stats.items()
                },
                "cycles": {
                    label: ticks_to_cycles(
                        observed.get((component, statistic), 0),
                        accelerator_clock_hz, timebase_ticks_per_second,
                    )
                    for label, statistic in prefetch_stats.items()
                },
                "counts": {
                    label: counts.get((component, statistic), 0)
                    for label, statistic in prefetch_stats.items()
                },
            }
        if critical_path and not critical_path.get("order_valid", False):
            mismatches["attention_worker_critical_path_order"] = {
                "expected": "dispatch <= final_qk <= final_softmax <= final_pv <= output_ack",
                "actual": critical_path["milestone_ticks"],
            }
        system_frontier = lifecycle["system_frontier"]
        if system_frontier and not system_frontier.get("order_valid", False):
            mismatches["attention_system_frontier_order"] = {
                "expected": "monotonic system milestone frontiers",
                "actual": system_frontier["milestone_ticks"],
            }
        if system_frontier and not system_frontier.get(
            "accelerator_attribution", {}
        ).get("conservation_valid", False):
            mismatches["attention_system_frontier_conservation"] = {
                "expected": 0,
                "actual": system_frontier.get(
                    "accelerator_attribution", {}
                ).get("unattributed_ticks"),
            }
    if lifecycle and generic_gemm:
        wcp_stats = (
            "gemm_proxy_commands_issued",
            "gemm_proxy_queue_full_stalls",
            "gemm_proxy_queue_wait_cycles",
            "gemm_proxy_launch_commands",
            "gemm_proxy_completion_callbacks",
            "gemm_proxy_completion_delay_cycles",
        )
        lifecycle["wcp_gemm_proxy"] = {
            "queue_depth": wcp_gemm_proxy_queue_depth,
            "issue_width": wcp_gemm_proxy_issue_width,
            "command_latency_cycles": wcp_gemm_proxy_command_latency_cycles,
            "completion_latency_cycles": wcp_gemm_proxy_completion_latency_cycles,
            "max_worker_queue_wait_cycles": max(
                maxima.get(
                    (f"core{core}:rocc:worker_command_processor",
                     "gemm_proxy_queue_wait_cycles"), 0
                )
                for core in range(4, 20)
            ),
            "worker_totals": {
                statistic: sum(
                    observed.get(
                        (f"core{core}:rocc:worker_command_processor", statistic), 0
                    )
                    for core in range(4, 20)
                )
                for statistic in wcp_stats
            },
        }
        qk_storage_stats = (
            "attention_tile_storage_acquires",
            "attention_tile_storage_releases",
            "attention_tile_storage_mode_conflicts",
            "attention_tile_storage_capacity_rejections",
            "attention_tile_storage_column_writes",
            "attention_tile_storage_row_reads",
            "attention_tile_storage_write_bytes",
            "attention_tile_storage_read_bytes",
            "attention_tile_storage_write_wait_cycles",
            "attention_tile_storage_read_wait_cycles",
        )
        accumulator_storage_stats = (
            "attention_storage_session_acquires",
            "attention_storage_session_releases",
            "attention_accumulator_row_writes",
            "attention_accumulator_row_reads",
            "attention_accumulator_write_bytes",
            "attention_accumulator_read_bytes",
            "attention_accumulator_write_wait_cycles",
            "attention_accumulator_read_wait_cycles",
        )
        lifecycle["qk_score_row_burst"] = {
            "enabled": qk_score_row_burst,
            "mode": "ATTENTION_TILE_STORAGE" if qk_score_row_burst else "disabled",
            "capacity_bytes": 1024,
            "banks": attention_tile_storage_banks,
            "bank_bytes_per_cycle": attention_tile_storage_bank_bytes_per_cycle,
            "row_burst_bytes": 64,
            "worker_totals": {
                statistic: sum(
                    observed.get(
                        (f"core{core}:rocc:worker_command_processor", statistic), 0
                    )
                    for core in range(4, 20)
                )
                for statistic in qk_storage_stats
            },
        }
        lifecycle["attention_o_accumulator_cbuffer"] = {
            "enabled": o_accumulator_cbuffer,
            "mode": "ATTENTION_TILE_STORAGE" if o_accumulator_cbuffer else "disabled",
            "qk_scratch_bytes": 1024,
            "accumulator_bytes": 16 * activity["dimension_panels"] * 64,
            "worker_totals": {
                statistic: sum(
                    observed.get(
                        (f"core{core}:rocc:worker_command_processor", statistic), 0
                    )
                    for core in range(4, 20)
                )
                for statistic in accumulator_storage_stats
            },
        }
    if lifecycle:
        broadcast_stats = (
            "matrix_broadcast_requests",
            "matrix_broadcast_rejected",
            "matrix_broadcast_ingress_bytes",
            "matrix_broadcast_sink_bytes",
            "matrix_broadcast_transfer_cycles",
            "matrix_broadcast_fanout",
        )
        lifecycle["matrix_broadcast_fabric"] = {
            "enabled": broadcast["enabled"],
            "pv_enabled": pv_matrix_broadcast,
            "qk_enabled": qk_matrix_broadcast,
            "topology": "binary_tree",
            "shares_array_buffer_ports": not attention_cluster,
            "max_fanout": matrix_broadcast_max_fanout,
            "bytes_per_cycle": matrix_broadcast_bytes_per_cycle,
            "base_latency_cycles": matrix_broadcast_base_latency_cycles,
            "stage_latency_cycles": matrix_broadcast_stage_latency_cycles,
            "tree_stages": broadcast["tree_stages"],
            "fanout": broadcast["fanout"],
            "payload_bytes": broadcast["payload_bytes"],
            "qk_payload_bytes": broadcast["qk_payload_bytes"],
            "q_pair_payload_bytes": broadcast.get("q_pair_payload_bytes"),
            "pv_payload_bytes": broadcast["pv_payload_bytes"],
            "cycles_per_request": broadcast["cycles_per_request"],
            "qk_cycles_per_request": broadcast["qk_cycles_per_request"],
            "q_pair_cycles_per_request": broadcast.get(
                "q_pair_cycles_per_request"
            ),
            "pv_cycles_per_request": broadcast["pv_cycles_per_request"],
            "qk_requests_per_worker": broadcast["qk_requests"],
            "q_pair_requests_per_worker": broadcast.get("q_pair_requests", 0),
            "pv_requests_per_worker": broadcast["pv_requests"],
            "max_observed_fanout": max(
                maxima.get(
                    (f"core{core}:rocc:array", "matrix_broadcast_fanout"), 0
                )
                for core in range(4, 20)
            ),
            "worker_totals": {
                statistic: sum(
                    observed.get(
                        (f"core{core}:rocc:array", statistic), 0
                    )
                    for core in range(4, 20)
                )
                for statistic in broadcast_stats
            },
        }
        lifecycle["input_scatter_fabric"] = {
            **scatter,
            "topology": "64_lane_destination_specific",
            "shares_array_buffer_ports": True,
            "base_latency_cycles": array_buffer_base_latency_cycles,
            "worker_totals": {
                statistic: sum(
                    observed.get((f"core{core}:rocc:array", statistic), 0)
                    for core in range(4, 20)
                )
                for statistic in (
                    "input_scatter_requests", "input_scatter_rejected",
                    "input_scatter_bytes", "input_scatter_transfer_cycles",
                    "input_scatter_destinations",
                )
            },
        }
        lifecycle["output_scatter_gather_fabric"] = {
            **output_scatter_gather,
            "topology": "64_lane_accumulator_scatter_gather",
            "shares_array_buffer_ports": True,
            "base_latency_cycles": array_buffer_base_latency_cycles,
            "worker_totals": {
                statistic: sum(
                    observed.get((f"core{core}:rocc:array", statistic), 0)
                    for core in range(4, 20)
                )
                for statistic in (
                    "output_scatter_requests", "output_gather_requests",
                    "output_scatter_gather_rejected",
                    "output_scatter_gather_bytes",
                    "output_scatter_gather_transfer_cycles",
                    "output_scatter_gather_destinations",
                )
            },
        }
        lifecycle["attention_cluster"] = {
            "enabled": attention_cluster,
            "physical_array_contract": {
                "arrays_per_worker": 64,
                "input_rows_per_array": 64,
                "output_cus_per_array": 64,
                "operand_banks_per_array": 2,
            },
            "array_partition": {
                "qk": [0, attention_cluster_qk_arrays - 1],
                "pv": [attention_cluster_qk_arrays, 63],
            },
            "owned_arrays": {
                "qk": attention_cluster_qk_arrays,
                "pv": 64 - attention_cluster_qk_arrays,
            },
            "contexts_per_worker": activity["jobs"] * 3 + activity["qblocks"],
            "q_physical_inputs_per_tile": 32,
            "qk_pair_reductions_per_tile": 16,
            "score_beats_per_tile": 16,
            "direct_p": {
                "slots": 4,
                "bytes_per_slot": 16 * active_columns * 4,
                "write_bytes_per_cycle": 64,
                "read_bytes_per_cycle": 64,
                "reservations_per_worker": activity["jobs"],
                "writes_per_worker": activity["jobs"] * active_columns // 16,
                "reads_per_worker": activity["jobs"] * 16,
            },
            "group_pipeline": {
                "group_size": effective_kv_query_group_size,
                "physical_tiles_per_worker": physical_kv_jobs,
                "logical_tiles_per_worker": activity["jobs"],
                "ahead_contexts_per_worker": (
                    activity["jobs"] - physical_kv_jobs
                ),
                "tile_ii_target_cycles": 625,
                "tile_ii_applicable": cluster_ii_counts["steady"] > 0,
                "tile_ii_accepted": (
                    all(
                        observed.get(
                            (f"core{core}:rocc",
                             "attention_cluster_qk_steady_ii_over_target"), 0
                        ) * 20 <= counts.get(
                            (f"core{core}:rocc",
                             "attention_cluster_qk_tile_ii_cycles"), 0
                        )
                        for core in range(4, 20)
                    ) if cluster_ii_counts["steady"] > 0 else None
                ),
                "three_stage_overlap_cycles_total": sum(
                    observed.get(
                        (f"core{core}:rocc",
                         "attention_cluster_three_stage_overlap_cycles"), 0
                    )
                    for core in range(4, 20)
                ),
                "workers": [
                    {
                        "core": core,
                        "ahead_launched": observed.get(
                            (f"core{core}:rocc",
                             "attention_cluster_ahead_contexts_launched"), 0
                        ),
                        "ahead_completed": observed.get(
                            (f"core{core}:rocc",
                             "attention_cluster_ahead_contexts_completed"), 0
                        ),
                        "ahead_promoted": observed.get(
                            (f"core{core}:rocc",
                             "attention_cluster_ahead_contexts_promoted"), 0
                        ),
                        "promotion_waits": observed.get(
                            (f"core{core}:rocc",
                             "attention_cluster_promotion_waits"), 0
                        ),
                        "initial_enqueue_retries": observed.get(
                            (f"core{core}:rocc",
                             "attention_cluster_initial_enqueue_retries"), 0
                        ),
                        "tile_ii": {
                            "count": counts.get(
                                (f"core{core}:rocc",
                                 "attention_cluster_qk_tile_ii_cycles"), 0
                            ),
                            "sum_cycles": observed.get(
                                (f"core{core}:rocc",
                                 "attention_cluster_qk_tile_ii_cycles"), 0
                            ),
                            "min_cycles": minima.get(
                                (f"core{core}:rocc",
                                 "attention_cluster_qk_tile_ii_cycles"), 0
                            ),
                            "max_cycles": maxima.get(
                                (f"core{core}:rocc",
                                 "attention_cluster_qk_tile_ii_cycles"), 0
                            ),
                            "over_target_count": observed.get(
                                (f"core{core}:rocc",
                                 "attention_cluster_qk_steady_ii_over_target"), 0
                            ),
                            "p95_relation": (
                                "<=625" if observed.get(
                                    (f"core{core}:rocc",
                                     "attention_cluster_qk_steady_ii_over_target"), 0
                                ) * 20 <= counts.get(
                                    (f"core{core}:rocc",
                                     "attention_cluster_qk_tile_ii_cycles"), 0
                                ) else ">625"
                            ),
                        },
                        "boundary_ii": {
                            "count": counts.get(
                                (f"core{core}:rocc",
                                 "attention_cluster_qk_boundary_ii_cycles"), 0
                            ),
                            "sum_cycles": observed.get(
                                (f"core{core}:rocc",
                                 "attention_cluster_qk_boundary_ii_cycles"), 0
                            ),
                            "min_cycles": minima.get(
                                (f"core{core}:rocc",
                                 "attention_cluster_qk_boundary_ii_cycles"), 0
                            ),
                            "max_cycles": maxima.get(
                                (f"core{core}:rocc",
                                 "attention_cluster_qk_boundary_ii_cycles"), 0
                            ),
                        },
                        "overlap_cycles": {
                            name: observed.get(
                                (f"core{core}:rocc",
                                 f"attention_cluster_{name}_overlap_cycles"), 0
                            )
                            for name in ("qk_sfu", "sfu_pv", "qk_pv", "three_stage")
                        },
                    }
                    for core in range(4, 20)
                ],
            },
        }
        if attention_cluster:
            critical_worker_core = lifecycle["worker_critical_path"][
                "slowest_worker_core"
            ]
            lifecycle["attention_cluster"]["resource_profile"] = (
                summarize_attention_cluster_resource_profile(
                    observed, maxima, critical_worker_core,
                    accelerator_clock_hz, timebase_ticks_per_second,
                )
            )
        o_stats = (
            "attention_cluster_o_context_reservations",
            "attention_cluster_o_context_releases",
            "attention_cluster_o_context_cancelled",
            "attention_cluster_o_scale_segments",
            "attention_cluster_o_accumulate_segments",
            "attention_cluster_o_fused_rows",
            "attention_cluster_o_fused_bytes",
            "attention_cluster_o_drain_requests",
            "attention_cluster_o_drain_bytes",
            "attention_cluster_o_read_wait_cycles",
            "attention_cluster_o_write_wait_cycles",
            "attention_cluster_o_alu_wait_cycles",
            "attention_cluster_o_bank_conflict_cycles",
            "attention_cluster_o_drain_wait_cycles",
        )
        lifecycle["attention_cluster"]["resident_o"] = {
            "contexts": 4,
            "bytes_per_context": 8192,
            "banks": 16,
            "pv_row_fusion": pv_o_row_fusion,
            "max_context_high_water": max(
                maxima.get(
                    (f"core{core}:rocc",
                     "attention_cluster_o_context_high_water"), 0
                )
                for core in range(4, 20)
            ),
            "worker_totals": {
                statistic: sum(
                    observed.get((f"core{core}:rocc", statistic), 0)
                    for core in range(4, 20)
                )
                for statistic in o_stats
            },
        }
        active_k_stats = (
            "active_k_launches",
            "active_k_columns",
            "active_k_compute_cycles",
            "active_k_full_width_cycles_avoided",
        )
        lifecycle["pv_active_k"] = {
            "configured": pv_active_k,
            "enabled": effective_pv_active_k,
            "active_columns": active_columns,
            "mac_per_cu_per_cycle": array_mac_per_cu_per_cycle,
            "pipeline_depth": array_pipeline_depth,
            "worker_totals": {
                statistic: sum(
                    observed.get(
                        (f"core{core}:rocc:array", statistic), 0
                    )
                    for core in range(4, 20)
                )
                for statistic in active_k_stats
            },
        }
        lifecycle["qk_early_compute"] = {
            "configured": qk_early_compute,
            "enabled": qk_early_compute,
            "worker_totals": {
                "arrays": sum(
                    observed.get(
                        (f"core{core}:rocc",
                         "attention_qk_early_compute_arrays"), 0
                    )
                    for core in range(4, 20)
                ),
            },
        }
        qk_input_pipeline_stats = (
            "rows_fetched",
            "rows_programmed",
            "overlap_ticks",
            "slot_full_stalls",
            "max_depth",
            "tag_mismatches",
        )
        lifecycle["qk_input_pipeline"] = {
            "configured": qk_input_pipeline,
            "enabled": qk_input_pipeline,
            "depth": 2,
            "row_bytes": 512,
            "capacity_bytes": 1024,
            "worker_totals": {
                statistic: sum(
                    observed.get(
                        (f"core{core}:rocc",
                         f"attention_qk_input_pipeline_{statistic}"), 0
                    )
                    for core in range(4, 20)
                )
                for statistic in qk_input_pipeline_stats
            },
        }
        cross_tile_operand_stats = (
            "candidates",
            "launches",
            "matrix_programs",
            "input_rows",
            "ready_hits",
            "waits",
            "wait_ticks",
            "promotions",
            "tag_mismatches",
        )
        lifecycle["cross_tile_operand_pipeline"] = {
            "configured": cross_tile_operand_pipeline,
            "enabled": cross_tile_operand_pipeline and ahead_operand_tiles > 0,
            "depth": 2,
            "max_ahead": 1,
            "operand_banks": 2 if cross_tile_operand_pipeline else 1,
            "capacity_bytes_per_worker": 139264 if cross_tile_operand_pipeline else 0,
            "worker_totals": {
                statistic: sum(
                    observed.get(
                        (f"core{core}:rocc",
                         f"attention_cross_tile_operand_{statistic}"), 0
                    )
                    for core in range(4, 20)
                )
                for statistic in cross_tile_operand_stats
            },
        }
        v_tile_stats = (
            "hits",
            "misses",
            "bytes_read",
            "bytes_reused",
            "wait_ticks",
            "capacity_rejections",
            "group_hits",
        )
        lifecycle["pv_v_tile_buffer"] = {
            "enabled": pv_v_tile_reuse,
            "group_retention_enabled": pv_v_tile_group_retention,
            "capacity_bytes": pv_v_tile_buffer_bytes,
            "bytes_per_cycle": pv_v_tile_buffer_bytes_per_cycle,
            "base_latency_cycles": pv_v_tile_buffer_hit_ticks,
            "panel_bytes": 32 * 16 * 4,
            "worker_totals": {
                statistic: sum(
                    observed.get(
                        (f"core{core}:rocc",
                         f"attention_pv_v_tile_buffer_{statistic}"), 0
                    )
                    for core in range(4, 20)
                )
                for statistic in v_tile_stats
            },
        }
        cross_query_stats = (
            "prefetches", "hits", "waits", "wait_ticks",
        )
        lifecycle["kv_cross_query_prefetch"] = {
            "enabled": kv_double_buffer and kv_cross_query_prefetch,
            "worker_totals": {
                statistic: sum(
                    observed.get(
                        (f"core{core}:rocc",
                         f"attention_kv_cross_query_{statistic}"), 0
                    )
                    for core in range(4, 20)
                )
                for statistic in cross_query_stats
            },
        }
        lifecycle["kv_pair_reuse"] = {
            "enabled": kv_pair_reuse,
            "query_group_size": effective_kv_query_group_size,
            "query_state_slots": effective_kv_query_group_size,
            "online_softmax_rows": 16 * effective_kv_query_group_size,
            "worker_totals": {
                "tiles": sum(
                    observed.get(
                        (f"core{core}:rocc", "attention_kv_pair_reuse_tiles"), 0
                    )
                    for core in range(4, 20)
                ),
                "bytes": sum(
                    observed.get(
                        (f"core{core}:rocc", "attention_kv_pair_reuse_bytes"), 0
                    )
                    for core in range(4, 20)
                ),
            },
        }
    if runtime_log is not None:
        dma_runtime = parse_dma_runtime_invariants(
            runtime_log, dma_data_nodes, dma_credit_cap)
        mismatches.update(dma_runtime.pop("mismatches"))
        lifecycle["dma_runtime_invariants"] = dma_runtime
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "mismatches": mismatches,
        "lifecycle": lifecycle,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id")
    parser.add_argument("--query-length", "--queries", dest="query_length", type=int)
    parser.add_argument("--kv-length", "--keys", dest="kv_length", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--num-query-heads", "--query-heads",
                        dest="num_query_heads", type=int)
    parser.add_argument("--num-kv-heads", "--kv-heads",
                        dest="num_kv_heads", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--key-block-rows", type=int, default=32)
    parser.add_argument(
        "--normalization-clock", "--accelerator-clock",
        dest="normalization_clock", type=parse_frequency_hz,
        default=parse_frequency_hz("1.0GHz"),
        help="clock used only to convert SST timebase ticks to normalized cycles",
    )
    parser.add_argument("--model-cpu-clock", type=parse_frequency_hz,
                        default=parse_frequency_hz("2.3GHz"))
    parser.add_argument("--model-array-clock", type=parse_frequency_hz,
                        default=parse_frequency_hz("2.3GHz"))
    parser.add_argument("--model-memctrl-clock", type=parse_frequency_hz,
                        default=parse_frequency_hz("2.3GHz"))
    parser.add_argument("--model-platform-clock", type=parse_frequency_hz,
                        default=parse_frequency_hz("2.0GHz"))
    parser.add_argument(
        "--run-config",
        help=(
            "resolved run_config.env; when supplied, its CPU, array, and "
            "memory-controller clocks override model-clock arguments"
        ),
    )
    parser.add_argument("--timebase-ticks-per-second", type=int, default=10**12)
    parser.add_argument("--pv-matrix-broadcast", action="store_true")
    parser.add_argument("--qk-matrix-broadcast", action="store_true")
    parser.add_argument("--qk-dataflow-transpose", action="store_true")
    parser.add_argument(
        "--qk-early-compute", action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--qk-input-pipeline", action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--qk-readout-overlap", action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--qk-readout-window", type=int, default=2)
    parser.add_argument(
        "--qk-score-row-burst", "--qk-panel-row-burst",
        dest="qk_score_row_burst", action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--cross-tile-operand-pipeline",
        action=argparse.BooleanOptionalAction, default=False,
    )
    parser.add_argument("--attention-tile-storage-banks", type=int, default=16)
    parser.add_argument(
        "--attention-tile-storage-bank-bytes-per-cycle", type=int, default=64,
    )
    parser.add_argument("--runtime-log")
    parser.add_argument("--kv-double-buffer", action="store_true")
    parser.add_argument(
        "--kv-second-lookahead", action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--kv-cross-query-prefetch", action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--kv-distribution", action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--kv-manager-lookahead", action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--kv-pair-reuse", action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--kv-query-group-size", type=int, default=2)
    parser.add_argument("--pv-v-tile-reuse", action="store_true")
    parser.add_argument(
        "--pv-v-tile-group-retention", action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--pv-v-tile-buffer-bytes", type=int, default=16384)
    parser.add_argument("--pv-v-tile-buffer-hit-ticks", type=int, default=1)
    parser.add_argument("--pv-v-tile-buffer-bytes-per-cycle", type=int, default=64)
    parser.add_argument("--pv-input-pipeline", action="store_true")
    parser.add_argument(
        "--pv-input-residency", action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--o-accumulator-cbuffer", action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--pv-restore-pipeline", action="store_true")
    parser.add_argument("--pv-output-pipeline", action="store_true")
    parser.add_argument("--pv-early-compute", action="store_true")
    parser.add_argument("--pv-matrix-softmax-overlap", action="store_true")
    parser.add_argument("--pv-active-k", action="store_true")
    parser.add_argument("--sequential-64", action="store_true")
    parser.add_argument("--array-mac-per-cu-per-cycle", type=float, default=1.0)
    parser.add_argument("--array-pipeline-depth", type=int, default=2)
    parser.add_argument(
        "--near-array-output-bytes-per-cycle", type=int, default=512
    )
    parser.add_argument(
        "--array-buffer-base-latency-cycles", type=int, default=1
    )
    parser.add_argument("--generic-gemm", action="store_true")
    parser.add_argument("--matrix-broadcast-max-fanout", type=int, default=16)
    parser.add_argument("--matrix-broadcast-bytes-per-cycle", type=int, default=256)
    parser.add_argument("--input-scatter-bytes-per-cycle", type=int, default=256)
    parser.add_argument(
        "--output-scatter-gather-bytes-per-cycle", type=int, default=256
    )
    parser.add_argument(
        "--matrix-broadcast-base-latency-cycles", type=int, default=1
    )
    parser.add_argument(
        "--matrix-broadcast-stage-latency-cycles", type=int, default=1
    )
    parser.add_argument(
        "--wcp-gemm-proxy-completion-latency-cycles", type=int, default=1
    )
    parser.add_argument("--wcp-gemm-proxy-queue-depth", type=int, default=32)
    parser.add_argument("--wcp-gemm-proxy-issue-width", type=int, default=1)
    parser.add_argument(
        "--wcp-gemm-proxy-command-latency-cycles", type=int, default=1
    )
    parser.add_argument("--result-json")
    parser.add_argument("stats_file")
    args = parser.parse_args()
    query_heads = args.num_query_heads or args.heads or 1
    kv_heads = args.num_kv_heads or (args.heads if args.heads is not None else query_heads)
    explicit_shape = (args.query_length, args.kv_length, args.head_dim)
    if any(value is not None for value in explicit_shape):
        if not all(value is not None for value in explicit_shape):
            parser.error("--queries, --keys, and --head-dim must be supplied together")
        try:
            activity_factory = (
                make_sequential_64_activity
                if args.sequential_64 else make_attention_activity
            )
            activity = activity_factory(
                *explicit_shape, key_block_rows=args.key_block_rows
            )
            activity = scale_activity_for_heads(activity, query_heads, kv_heads)
        except ValueError as error:
            parser.error(str(error))
        case_id = args.case_id or (
            f"fused_attention_q{args.query_length}_k{args.kv_length}_d{args.head_dim}"
            + (f"_hq{query_heads}_hkv{kv_heads}"
               if query_heads > 1 or kv_heads > 1 else "")
        )
    else:
        parser.error("explicit dimensions are required")
    if args.run_config:
        try:
            resolved_clocks = read_model_clocks_from_run_config(args.run_config)
        except (OSError, ValueError, argparse.ArgumentTypeError) as error:
            parser.error(str(error))
        args.model_cpu_clock = resolved_clocks["cpu"]
        args.model_array_clock = resolved_clocks["array"]
        args.model_memctrl_clock = resolved_clocks["memctrl"]
        try:
            dma_credit_cap = read_integer_from_run_config(
                args.run_config, "GOLEM_DMA_NODE_CHUNK_CREDITS")
        except (OSError, ValueError) as error:
            parser.error(str(error))
    else:
        dma_credit_cap = None
    if (args.pv_matrix_broadcast or args.qk_matrix_broadcast) and \
            args.matrix_broadcast_max_fanout < 16:
        parser.error("matrix broadcast max fanout must be at least 16")
    if args.matrix_broadcast_bytes_per_cycle <= 0:
        parser.error("matrix broadcast bytes per cycle must be positive")
    if args.input_scatter_bytes_per_cycle <= 0:
        parser.error("input scatter bytes per cycle must be positive")
    if args.output_scatter_gather_bytes_per_cycle <= 0:
        parser.error("output scatter/gather bytes per cycle must be positive")
    if args.matrix_broadcast_base_latency_cycles <= 0:
        parser.error("matrix broadcast base latency must be positive")
    if args.matrix_broadcast_stage_latency_cycles < 0:
        parser.error("matrix broadcast stage latency cannot be negative")
    if args.array_mac_per_cu_per_cycle <= 0:
        parser.error("array MACs per CU per cycle must be positive")
    if args.array_pipeline_depth < 0:
        parser.error("array pipeline depth cannot be negative")
    if args.near_array_output_bytes_per_cycle <= 0:
        parser.error("near-array output bytes per cycle must be positive")
    if args.array_buffer_base_latency_cycles < 0:
        parser.error("array buffer base latency cannot be negative")
    if not 1 <= args.qk_readout_window <= 16:
        parser.error("QK readout window must be from 1 through 16")
    if args.qk_input_pipeline and args.qk_dataflow_transpose:
        parser.error("QK input pipeline requires non-transposed QK")
    if args.attention_tile_storage_banks <= 0 or \
            args.attention_tile_storage_bank_bytes_per_cycle <= 0:
        parser.error("Attention tile-storage bank resources must be positive")
    if args.pv_v_tile_buffer_bytes_per_cycle <= 0:
        parser.error("PV V tile buffer bytes per cycle must be positive")
    if args.pv_v_tile_group_retention and not args.pv_v_tile_reuse:
        parser.error("PV V-tile group retention requires PV V-tile reuse")
    if args.kv_query_group_size not in (1, 2, 4):
        parser.error("KV query group size must be 1, 2, or 4")
    if args.kv_pair_reuse and args.kv_query_group_size == 1:
        parser.error("KV pair reuse requires query group size 2 or 4")
    if args.cross_tile_operand_pipeline and not args.kv_pair_reuse:
        parser.error("cross-tile operand pipeline requires KV pair reuse")
    result = verify(args.stats_file, case_id, args.normalization_clock,
                    args.timebase_ticks_per_second, args.pv_matrix_broadcast,
                    args.qk_matrix_broadcast, args.qk_dataflow_transpose,
                    args.kv_double_buffer, args.pv_input_pipeline,
                    args.pv_matrix_softmax_overlap, args.pv_restore_pipeline,
                    args.pv_output_pipeline, args.pv_early_compute,
                    args.pv_v_tile_reuse, args.pv_v_tile_buffer_bytes,
                    args.pv_v_tile_buffer_hit_ticks,
                    args.pv_v_tile_buffer_bytes_per_cycle,
                    args.model_cpu_clock,
                    args.model_array_clock, args.model_memctrl_clock,
                    args.model_platform_clock, args.generic_gemm,
                    args.matrix_broadcast_max_fanout,
                    args.matrix_broadcast_bytes_per_cycle,
                    args.matrix_broadcast_base_latency_cycles,
                    args.matrix_broadcast_stage_latency_cycles,
                    args.wcp_gemm_proxy_completion_latency_cycles,
                    args.wcp_gemm_proxy_queue_depth,
                    args.wcp_gemm_proxy_issue_width,
                    args.wcp_gemm_proxy_command_latency_cycles,
                    activity, args.kv_second_lookahead,
                    args.kv_cross_query_prefetch, args.kv_pair_reuse,
                    args.kv_query_group_size,
                    args.pv_v_tile_group_retention,
                    args.pv_active_k,
                    args.array_mac_per_cu_per_cycle,
                    args.array_pipeline_depth, args.qk_early_compute,
                    args.qk_input_pipeline,
                    args.qk_readout_overlap, args.qk_readout_window,
                    args.qk_score_row_burst,
                    args.pv_input_residency, args.o_accumulator_cbuffer,
                    args.attention_tile_storage_banks,
                    args.attention_tile_storage_bank_bytes_per_cycle,
                    args.runtime_log, dma_credit_cap,
                    cross_tile_operand_pipeline=
                        args.cross_tile_operand_pipeline,
                    attention_cluster=False,
                    sequential_64=args.sequential_64,
                    attention_cluster_qk_arrays=16,
                    pv_o_row_fusion=False,
                    cluster_pv_row_wavefront=False,
                    cluster_qk_matrix_lookahead=False,
                    cluster_pv_matrix_lookahead=False,
                    near_array_output_bytes_per_cycle=
                        args.near_array_output_bytes_per_cycle,
                    input_scatter_bytes_per_cycle=
                        args.input_scatter_bytes_per_cycle,
                    output_scatter_gather_bytes_per_cycle=
                        args.output_scatter_gather_bytes_per_cycle,
                    array_buffer_base_latency_cycles=
                        args.array_buffer_base_latency_cycles,
                    kv_distribution=args.kv_distribution,
                    kv_manager_lookahead=args.kv_manager_lookahead)
    print(json.dumps(result, indent=2))
    if args.result_json:
        output = Path(args.result_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
