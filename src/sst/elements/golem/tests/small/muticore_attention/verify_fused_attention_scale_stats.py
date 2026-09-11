#!/usr/bin/env python3
"""Verify exact Phase E four-manager/sixteen-worker activity."""

import argparse
import csv
import json
import math
import re
from decimal import Decimal
from pathlib import Path


PROFILES = {
    "e2": {
        "qk": 256, "pv": 512, "jobs": 8, "qblocks": 1,
        "rows": 128, "scaled": 4096, "dimension_panels": 4,
        "v_tile_bytes": 8192,
    },
    "e3": {
        "qk": 4096,
        "pv": 16384,
        "jobs": 128,
        "qblocks": 4,
        "rows": 2048,
        "scaled": 65536,
        "dimension_panels": 8,
        "v_tile_bytes": 16384,
    },
    "e4": {
        "qk": 16384,
        "pv": 65536,
        "jobs": 512,
        "qblocks": 8,
        "rows": 8192,
        "scaled": 262144,
        "dimension_panels": 8,
        "v_tile_bytes": 16384,
    },
    "e5": {
        "qk": 65536,
        "pv": 262144,
        "jobs": 2048,
        "qblocks": 16,
        "rows": 32768,
        "scaled": 1048576,
        "dimension_panels": 8,
        "v_tile_bytes": 16384,
    },
}


def make_attention_activity(queries, keys, head_dim):
    if queries <= 0 or queries % 256 != 0:
        raise ValueError("queries must be a positive multiple of 256")
    if keys <= 0 or keys % 32 != 0:
        raise ValueError("keys must be a positive multiple of 32")
    if head_dim <= 0 or head_dim % 16 != 0:
        raise ValueError("head_dim must be a positive multiple of 16")
    query_blocks_per_worker = queries // 256
    key_tiles = keys // 32
    jobs = query_blocks_per_worker * key_tiles
    rows = jobs * 16
    return {
        "qk": jobs * 32,
        "pv": jobs * head_dim,
        "jobs": jobs,
        "qblocks": query_blocks_per_worker,
        "rows": rows,
        "scaled": rows * 32,
        "dimension_panels": head_dim // 16,
        "v_tile_bytes": 32 * head_dim * 4,
    }


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
    response_summaries = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if "CREDIT_OWNER_SUMMARY" in line:
            fields = dict(re.findall(r"([a-z0-9_]+)=([^\s]+)", line))
            credit_summaries.append(fields)
        elif "GOLEM_MEMNIC_DMA_RESPONSE_STATS" in line:
            fields = dict(re.findall(r"([a-z0-9_]+)=([^\s]+)", line))
            response_summaries.append(fields)

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
        "normalized_cycles_are_model_native_cycles": False,
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
        ("dispatch_accept", "attention_worker_dispatch_accept_tick", observed),
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


def summarize_system_frontier(observed, maxima, accelerator_clock_hz,
                              timebase_ticks_per_second):
    root = "core0:rocc"
    milestones = {
        "root_descriptor_accept": observed.get(
            (root, "attention_manager_descriptor_accept_tick")
        ),
        "manager_dispatch_complete": max(
            observed.get((f"core{core}:rocc", "attention_manager_dispatch_tick"), 0)
            for core in range(4)
        ),
        "worker_dispatch_accept_complete": max(
            observed.get((f"core{core}:rocc", "attention_worker_dispatch_accept_tick"), 0)
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
            observed.get((f"core{core}:rocc", "attention_manager_local_complete_tick"), 0)
            for core in range(4)
        ),
        "root_tensor_complete": observed.get((root, "attention_tensor_complete_tick")),
        "software_wait_observed": observed.get(
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


def verify(path, profile, accelerator_clock_hz=1_000_000_000,
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
           qk_readout_window=2, qk_panel_row_burst=False,
           pv_input_residency=False, o_accumulator_cbuffer=False,
           attention_tile_storage_banks=16,
           attention_tile_storage_bank_bytes_per_cycle=64,
           runtime_log=None,
           dma_credit_cap=None, dma_data_nodes=4,
           cross_tile_operand_pipeline=False):
    if activity is None:
        activity = PROFILES[profile]
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
    active_columns = 32
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
    full_columns = activity["v_tile_bytes"] // (active_columns * 4)
    effective_pv_active_k = pv_active_k and active_columns < full_columns
    broadcast = expected_matrix_broadcast_activity(
        activity, pv_matrix_broadcast, qk_matrix_broadcast,
        qk_dataflow_transpose, matrix_broadcast_max_fanout,
        matrix_broadcast_bytes_per_cycle,
        matrix_broadcast_base_latency_cycles,
        matrix_broadcast_stage_latency_cycles,
        effective_pv_active_k,
    )
    for core in range(4):
        component = f"core{core}:rocc"
        expected[(component, "attention_manager_jobs_issued")] = 1
        expected[(component, "attention_manager_jobs_completed")] = 1
        expected[(component, "attention_manager_bands_completed")] = 1
        expected[(component, "attention_manager_band_completions_received")] = (
            4 if core == 0 else 0
        )
        expected[(component, "attention_tensor_jobs_completed")] = (
            1 if core == 0 else 0
        )
        expected[(f"core{core}:rocc:sfu", "sfu_attention_jobs")] = 0
        for stat in (
            "attention_manager_descriptor_accept_tick",
            "attention_manager_dispatch_tick",
            "attention_manager_local_complete_tick",
            "attention_manager_wait_observed_tick",
        ):
            expected_counts[(component, stat)] = 1
        expected_counts[(component, "attention_manager_band_completion_received_tick")] = (
            4 if core == 0 else 0
        )
        expected_counts[(component, "attention_tensor_complete_tick")] = (
            1 if core == 0 else 0
        )
    for core in range(4, 20):
        component = f"core{core}:rocc"
        v_tile = expected_v_tile_buffer_activity(
            activity, pv_v_tile_reuse, pv_v_tile_buffer_bytes,
            pv_v_tile_buffer_hit_ticks, pv_v_tile_buffer_bytes_per_cycle,
            pv_v_tile_group_retention, physical_kv_jobs,
        )
        for suffix, value in v_tile.items():
            expected[(component, f"attention_pv_v_tile_buffer_{suffix}")] = value
        expected[(component, "attention_qk_array_ops")] = activity["qk"]
        expected[(component, "attention_pv_array_ops")] = activity["pv"]
        expected[(component, "attention_generic_gemm_qk_ops")] = (
            activity["qk"] if generic_gemm else 0
        )
        expected[(component, "attention_generic_gemm_pv_ops")] = (
            activity["pv"] if generic_gemm else 0
        )
        expected[(component, "attention_pv_active_k_launches")] = (
            activity["pv"] if effective_pv_active_k else 0
        )
        expected[(component, "attention_pv_active_k_columns")] = (
            activity["pv"] * active_columns if effective_pv_active_k else 0
        )
        expected[(component, "attention_pv_active_k_matrix_elements")] = (
            activity["pv"] * 16 * active_columns if effective_pv_active_k else 0
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
            activity["pv"] if effective_pv_active_k else 0
        )
        expected[(array_component, "active_k_columns")] = (
            activity["pv"] * active_columns if effective_pv_active_k else 0
        )
        expected[(array_component, "active_k_compute_cycles")] = (
            activity["pv"] * active_cycles if effective_pv_active_k else 0
        )
        expected[(array_component, "active_k_full_width_cycles_avoided")] = (
            activity["pv"] * max(full_cycles - active_cycles, 0)
            if effective_pv_active_k else 0
        )
        if generic_gemm:
            wcp_component = f"{component}:worker_command_processor"
            gemm_launches = activity["qk"] + activity["pv"]
            expected[(wcp_component, "gemm_proxy_launch_commands")] = gemm_launches
            expected[(wcp_component, "gemm_proxy_completion_callbacks")] = gemm_launches
            expected[(wcp_component, "gemm_proxy_completion_delay_cycles")] = (
                gemm_launches * wcp_gemm_proxy_completion_latency_cycles
            )
            expected[(wcp_component, "gemm_proxy_queue_full_stalls")] = 0
            expected_counts[(wcp_component, "gemm_proxy_launch_commands")] = gemm_launches
            expected_counts[(wcp_component, "gemm_proxy_completion_callbacks")] = gemm_launches
            expected_counts[(wcp_component, "gemm_proxy_completion_delay_cycles")] = (
                gemm_launches
                if wcp_gemm_proxy_completion_latency_cycles > 0 else 0
            )
            qk_panels = activity["qk"] // 16 if qk_panel_row_burst else 0
            qk_storage_ops = activity["qk"] if qk_panel_row_burst else 0
            expected[(wcp_component, "attention_tile_storage_acquires")] = qk_panels
            expected[(wcp_component, "attention_tile_storage_releases")] = qk_panels
            expected[(wcp_component, "attention_tile_storage_mode_conflicts")] = 0
            expected[(wcp_component, "attention_tile_storage_capacity_rejections")] = 0
            expected[(wcp_component, "attention_tile_storage_column_writes")] = qk_storage_ops
            expected[(wcp_component, "attention_tile_storage_row_reads")] = qk_storage_ops
            expected[(wcp_component, "attention_tile_storage_write_bytes")] = qk_storage_ops * 64
            expected[(wcp_component, "attention_tile_storage_read_bytes")] = qk_storage_ops * 64
            expected_counts[(wcp_component, "attention_tile_storage_acquires")] = qk_panels
            expected_counts[(wcp_component, "attention_tile_storage_releases")] = qk_panels
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
            expected_prefetches += max(kv_query_groups - 1, 0)
        expected[(component, "attention_kv_prefetch_tiles")] = (
            expected_prefetches if kv_double_buffer else 0
        )
        expected_counts[(component, "attention_kv_prefetch_dma_ticks")] = (
            expected_prefetches if kv_double_buffer else 0
        )
        expected_counts[(component, "attention_kv_k_release_ticks")] = (
            activity["jobs"] if kv_double_buffer else 0
        )
        expected_counts[(component, "attention_kv_v_release_ticks")] = (
            activity["jobs"] if kv_double_buffer else 0
        )
        programmed_input_rows = (
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
            (activity["qk"] // 16 if qk_dataflow_transpose else
             activity["jobs"] - ahead_operand_tiles)
            if qk_matrix_broadcast else 0
        )
        expected[(component, "attention_pv_matrix_broadcasts")] = (
            activity["pv"] // 16 if pv_matrix_broadcast else 0
        )
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
        expected_counts[(component, "attention_worker_dispatch_accept_tick")] = 1
        for stat in (
            "attention_worker_qk_tile_complete_tick",
            "attention_worker_softmax_tile_complete_tick",
            "attention_worker_pv_tile_complete_tick",
        ):
            expected_counts[(component, stat)] = activity["jobs"]
        expected_counts[(component, "attention_worker_output_dma_ack_tick")] = (
            activity["qblocks"]
        )
        for stat in (
            "attention_worker_intertile_total_ticks",
            *(statistic for _, statistic in INTER_TILE_PHASE_STATS),
        ):
            expected_counts[(component, stat)] = activity["jobs"] - 1
        for stat in (
            "attention_worker_tile_total_ticks",
            *(statistic for _, statistic in TILE_PIPELINE_PHASE_STATS),
        ):
            expected_counts[(component, stat)] = activity["jobs"]
        expected[(f"core{core}:rocc:sfu", "sfu_attention_jobs")] = activity["jobs"]
        expected[(f"core{core}:rocc:sfu", "sfu_softmax_rows")] = activity["rows"]
        expected[(f"core{core}:rocc:sfu", "sfu_attention_scaled_elements")] = activity["scaled"]
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
            if actual_range != [broadcast["fanout"], broadcast["fanout"]]:
                mismatches[f"{component}/matrix_broadcast_fanout.range"] = {
                    "expected": [broadcast["fanout"], broadcast["fanout"]],
                    "actual": actual_range,
                }
    if generic_gemm:
        for core in range(4, 20):
            component = f"core{core}:rocc:worker_command_processor"
            commands = observed.get((component, "gemm_proxy_commands_issued"))
            launches = activity["qk"] + activity["pv"]
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
            expected_consumed += max(kv_query_groups - 1, 0)
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
        expected_cross = max(kv_query_groups - 1, 0) \
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
        if counts.get(key) != 1:
            mismatches[f"{key[0]}/{key[1]}.Count"] = {
                "expected": 1, "actual": counts.get(key)
            }
        worker = f"core{core}:rocc"
        worker_ticks = [
            observed.get((worker, "attention_worker_dispatch_accept_tick")),
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
        stat: [observed.get((f"core{core}:rocc", stat)) for core in range(4)]
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
    root_accept = observed.get((root, "attention_manager_descriptor_accept_tick"))
    tensor_complete = observed.get((root, "attention_tensor_complete_tick"))
    root_wait = observed.get((root, "attention_manager_wait_observed_tick"))
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
                observed, maxima, accelerator_clock_hz,
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
        lifecycle["qk_panel_row_burst"] = {
            "enabled": qk_panel_row_burst,
            "mode": "ATTENTION_TILE_STORAGE" if qk_panel_row_burst else "disabled",
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
            "shares_array_buffer_ports": True,
            "max_fanout": matrix_broadcast_max_fanout,
            "bytes_per_cycle": matrix_broadcast_bytes_per_cycle,
            "base_latency_cycles": matrix_broadcast_base_latency_cycles,
            "stage_latency_cycles": matrix_broadcast_stage_latency_cycles,
            "tree_stages": broadcast["tree_stages"],
            "fanout": broadcast["fanout"],
            "payload_bytes": broadcast["payload_bytes"],
            "qk_payload_bytes": broadcast["qk_payload_bytes"],
            "pv_payload_bytes": broadcast["pv_payload_bytes"],
            "cycles_per_request": broadcast["cycles_per_request"],
            "qk_cycles_per_request": broadcast["qk_cycles_per_request"],
            "pv_cycles_per_request": broadcast["pv_cycles_per_request"],
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
        active_k_stats = (
            "active_k_launches",
            "active_k_columns",
            "active_k_compute_cycles",
            "active_k_full_width_cycles_avoided",
        )
        lifecycle["pv_active_k"] = {
            "configured": pv_active_k,
            "enabled": effective_pv_active_k,
            "active_columns": 32,
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
    parser.add_argument("--profile", choices=sorted(PROFILES))
    parser.add_argument("--case-id")
    parser.add_argument("--queries", type=int)
    parser.add_argument("--keys", type=int)
    parser.add_argument("--head-dim", type=int)
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
        "--qk-panel-row-burst", action=argparse.BooleanOptionalAction,
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
    parser.add_argument("--array-mac-per-cu-per-cycle", type=float, default=1.0)
    parser.add_argument("--array-pipeline-depth", type=int, default=2)
    parser.add_argument("--generic-gemm", action="store_true")
    parser.add_argument("--matrix-broadcast-max-fanout", type=int, default=16)
    parser.add_argument("--matrix-broadcast-bytes-per-cycle", type=int, default=64)
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
    explicit_shape = (args.queries, args.keys, args.head_dim)
    if any(value is not None for value in explicit_shape):
        if not all(value is not None for value in explicit_shape):
            parser.error("--queries, --keys, and --head-dim must be supplied together")
        try:
            activity = make_attention_activity(*explicit_shape)
        except ValueError as error:
            parser.error(str(error))
        case_id = args.case_id or (
            f"fused_attention_q{args.queries}_k{args.keys}_d{args.head_dim}"
        )
    elif args.profile:
        activity = PROFILES[args.profile]
        case_id = args.profile
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
    if args.matrix_broadcast_base_latency_cycles <= 0:
        parser.error("matrix broadcast base latency must be positive")
    if args.matrix_broadcast_stage_latency_cycles < 0:
        parser.error("matrix broadcast stage latency cannot be negative")
    if args.array_mac_per_cu_per_cycle <= 0:
        parser.error("array MACs per CU per cycle must be positive")
    if args.array_pipeline_depth < 0:
        parser.error("array pipeline depth cannot be negative")
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
                    args.qk_panel_row_burst,
                    args.pv_input_residency, args.o_accumulator_cbuffer,
                    args.attention_tile_storage_banks,
                    args.attention_tile_storage_bank_bytes_per_cycle,
                    args.runtime_log, dma_credit_cap,
                    cross_tile_operand_pipeline=
                        args.cross_tile_operand_pipeline)
    print(json.dumps(result, indent=2))
    if args.result_json:
        output = Path(args.result_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
