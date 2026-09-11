#!/usr/bin/env python3
"""Build a concise Attention timing report from verified lifecycle statistics."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path


def _stage_rows(frontier):
    stage_ticks = frontier.get("stage_ticks", {})
    stage_cycles = frontier.get("stage_cycles", {})
    return [
        {
            "name": name,
            "duration_ticks": ticks,
            "duration_cycles": stage_cycles.get(name),
        }
        for name, ticks in stage_ticks.items()
    ]


def _load_verification_result(path, label):
    try:
        return json.loads(Path(path).read_text(encoding="ascii"))
    except (OSError, json.JSONDecodeError) as error:
        return {"status": "FAIL", "artifact_error": f"{label}: {error}"}


def build_report(lifecycle_result, numerical_result, *, profile, mpi_ranks,
                 sst_wall_seconds, pipeline_wall_seconds, generic_gemm,
                 mpi_partition_result=None, baseline_result=None):
    lifecycle = lifecycle_result.get("lifecycle", {})
    clock = lifecycle.get("clock_contract", {})
    frontier = lifecycle.get("system_frontier", {})
    critical = lifecycle.get("worker_critical_path", {})
    wcp = dict(lifecycle.get("wcp_gemm_proxy", {})) if generic_gemm else {}
    broadcast = dict(lifecycle.get("matrix_broadcast_fabric", {}))
    active_k = dict(lifecycle.get("pv_active_k", {}))
    qk_early = dict(lifecycle.get("qk_early_compute", {}))
    qk_input_pipeline = dict(lifecycle.get("qk_input_pipeline", {}))
    qk_row_burst = dict(lifecycle.get("qk_panel_row_burst", {}))
    v_tile = dict(lifecycle.get("pv_v_tile_buffer", {}))
    cross_query = dict(lifecycle.get("kv_cross_query_prefetch", {}))
    if wcp:
        wcp["clock_domain"] = "worker_command_processor_component_cycles"
        wcp["clock_hz"] = clock.get("model_clocks_hz", {}).get(
            "vanadis_cpu_rocc_sfu_local_gm"
        )
        wcp["aggregation"] = (
            "worker_totals cycle metrics are sums of WCP component cycles "
            "across workers, not normalized end-to-end latency"
        )
    if broadcast:
        broadcast["clock_domain"] = "mvm_array_component_cycles"
        broadcast["clock_hz"] = clock.get("model_clocks_hz", {}).get(
            "mvm_array"
        )
        broadcast["aggregation"] = (
            "worker_totals cycle metrics are sums of array component cycles "
            "across workers, not normalized end-to-end latency"
        )
    if active_k:
        active_k["clock_domain"] = "mvm_array_component_cycles"
        active_k["clock_hz"] = clock.get("model_clocks_hz", {}).get(
            "mvm_array"
        )
        active_k["aggregation"] = (
            "worker_totals are sums across workers and arrays, not "
            "normalized end-to-end latency"
        )
    if qk_early:
        qk_early["clock_domain"] = "rocc_component_cycles"
        qk_early["clock_hz"] = clock.get("model_clocks_hz", {}).get(
            "vanadis_cpu_rocc_sfu_local_gm"
        )
        qk_early["aggregation"] = (
            "worker_totals are sums across workers, not normalized "
            "end-to-end latency"
        )
    if qk_input_pipeline:
        qk_input_pipeline["clock_domain"] = "rocc_component_cycles"
        qk_input_pipeline["clock_hz"] = clock.get("model_clocks_hz", {}).get(
            "vanadis_cpu_rocc_sfu_local_gm"
        )
        qk_input_pipeline["aggregation"] = (
            "worker_totals are sums across workers, not normalized "
            "end-to-end latency"
        )
    if qk_row_burst:
        qk_row_burst["clock_domain"] = "worker_command_processor_component_cycles"
        qk_row_burst["clock_hz"] = clock.get("model_clocks_hz", {}).get(
            "vanadis_cpu_rocc_sfu_local_gm"
        )
        qk_row_burst["aggregation"] = (
            "worker_totals cycle metrics are sums of WCP component cycles "
            "across workers, not normalized end-to-end latency"
        )
    if v_tile:
        v_tile["clock_domain"] = "rocc_component_cycles"
        v_tile["clock_hz"] = clock.get("model_clocks_hz", {}).get(
            "vanadis_cpu_rocc_sfu_local_gm"
        )
        v_tile["aggregation"] = (
            "worker_totals are sums across workers, not normalized "
            "end-to-end latency"
        )
    if cross_query:
        cross_query["clock_domain"] = "rocc_component_cycles"
        cross_query["clock_hz"] = clock.get("model_clocks_hz", {}).get(
            "vanadis_cpu_rocc_sfu_local_gm"
        )
        cross_query["aggregation"] = (
            "worker_totals are sums across workers, not normalized "
            "end-to-end latency"
        )
    lifecycle_status = lifecycle_result.get("status", "FAIL")
    numerical_status = numerical_result.get("status", "FAIL")
    mpi_required = mpi_ranks > 1
    mpi_partition_result = mpi_partition_result or {}
    mpi_status = (
        mpi_partition_result.get("status", "FAIL")
        if mpi_required else "NOT_APPLICABLE"
    )
    mpi_summary = {
        name: mpi_partition_result.get(name)
        for name in (
            "status", "mpi_ranks", "missing_rank_files", "file_rank_mismatches",
            "missing_components", "unexpected_components", "misplaced_components",
            "conflicts", "router_conflicts", "artifact_error",
        )
        if name in mpi_partition_result
    }
    if mpi_required and "status" not in mpi_summary:
        mpi_summary["status"] = "FAIL"
        mpi_summary["artifact_error"] = "MPI partition result is missing"
    baseline_status = (
        baseline_result.get("status", "FAIL")
        if baseline_result is not None else "NOT_APPLICABLE"
    )

    return {
        "schema_version": 1,
        "status": (
            "PASS"
            if lifecycle_status == "PASS" and numerical_status == "PASS"
            and (not mpi_required or mpi_status == "PASS")
            and baseline_status in ("PASS", "NOT_APPLICABLE")
            else "FAIL"
        ),
        "configuration": {
            "profile": profile,
            "mpi_ranks": mpi_ranks,
            "generic_gemm": bool(generic_gemm),
            "wcp_enabled": bool(generic_gemm),
            "pv_matrix_broadcast": bool(broadcast.get("pv_enabled", False)),
            "pv_active_k": bool(active_k.get("enabled", False)),
            "qk_early_compute": bool(qk_early.get("enabled", False)),
            "qk_input_pipeline": bool(
                qk_input_pipeline.get("enabled", False)
            ),
            "qk_panel_row_burst": bool(qk_row_burst.get("enabled", False)),
            "pv_v_tile_buffer": bool(v_tile.get("enabled", False)),
            "kv_cross_query_prefetch": bool(cross_query.get("enabled", False)),
        },
        "host_timing": {
            "sst_wall_seconds": sst_wall_seconds,
            "pipeline_wall_seconds": pipeline_wall_seconds,
        },
        "time_domains": {
            "sst_timebase_ticks_per_second": clock.get(
                "sst_timebase_ticks_per_second",
                lifecycle.get("sst_timebase_ticks_per_second"),
            ),
            "normalization_clock_hz": clock.get("normalization_clock_hz"),
            "normalized_cycle_definition": clock.get(
                "normalized_cycle_definition"
            ),
            "normalized_cycles_are_model_native_cycles": clock.get(
                "normalized_cycles_are_model_native_cycles"
            ),
            "model_clocks_hz": clock.get("model_clocks_hz", {}),
        },
        "simulated_timing": {
            "total_ticks": lifecycle.get("accelerator_completion_ticks"),
            "total_cycles": lifecycle.get("accelerator_completion_cycles"),
            "total_milliseconds": lifecycle.get(
                "accelerator_completion_milliseconds"
            ),
            "wait_return_ticks": lifecycle.get("wait_return_ticks"),
            "wait_return_cycles": lifecycle.get("wait_return_cycles"),
            "wait_return_milliseconds": lifecycle.get(
                "wait_return_milliseconds"
            ),
            "manager_descriptor_accept_skew_cycles": lifecycle.get(
                "manager_descriptor_accept_skew_cycles"
            ),
            "manager_local_complete_skew_cycles": lifecycle.get(
                "manager_local_complete_skew_cycles"
            ),
        },
        "system_milestones": frontier.get("milestone_ticks", {}),
        "system_stages": _stage_rows(frontier),
        "system_attribution": frontier.get("accelerator_attribution", {}),
        "critical_worker": {
            "core": critical.get("slowest_worker_core"),
            "milestone_ticks": critical.get("milestone_ticks", {}),
            "stage_ticks": critical.get("stage_ticks", {}),
            "stage_cycles": critical.get("stage_cycles", {}),
            "aggregate_online_pipeline_cycles": critical.get(
                "aggregate_online_pipeline_cycles", {}
            ),
            "inter_tile_breakdown": critical.get("inter_tile_breakdown", {}),
            "tile_pipeline_breakdown": critical.get(
                "tile_pipeline_breakdown", {}
            ),
            "kv_prefetch_timing": critical.get("kv_prefetch_timing", {}),
        },
        "kv_second_lookahead_window": lifecycle.get(
            "kv_second_lookahead_window", {}
        ),
        "wcp": wcp,
        "matrix_broadcast_fabric": broadcast,
        "pv_active_k": active_k,
        "qk_early_compute": qk_early,
        "qk_input_pipeline": qk_input_pipeline,
        "qk_panel_row_burst": qk_row_burst,
        "pv_v_tile_buffer": v_tile,
        "kv_cross_query_prefetch": cross_query,
        "numerical_verification": numerical_result,
        "mpi_partition_verification": mpi_summary,
        "baseline_verification": baseline_result or {},
        "source_status": {
            "lifecycle": lifecycle_status,
            "numerical": numerical_status,
            "mpi_partition": mpi_status,
            "baseline": baseline_status,
        },
        "source_errors": {
            name: result["artifact_error"]
            for name, result in (
                ("lifecycle", lifecycle_result),
                ("numerical", numerical_result),
                ("mpi_partition", mpi_partition_result),
                ("baseline", baseline_result or {}),
            )
            if "artifact_error" in result
        },
    }


def _csv_rows(report):
    simulated = report["simulated_timing"]
    host = report["host_timing"]
    rows = [
        ("status", "overall", report["status"], "label"),
        ("status", "lifecycle", report["source_status"]["lifecycle"], "label"),
        ("status", "numerical", report["source_status"]["numerical"], "label"),
        ("status", "mpi_partition", report["source_status"]["mpi_partition"],
         "label"),
        ("status", "baseline", report["source_status"]["baseline"], "label"),
        ("host", "sst_wall_seconds", host["sst_wall_seconds"], "seconds"),
        ("host", "pipeline_wall_seconds", host["pipeline_wall_seconds"], "seconds"),
        ("summary", "total_ticks", simulated["total_ticks"], "sst_ticks"),
        ("summary", "total_cycles", simulated["total_cycles"],
         "normalized_cycles"),
        ("summary", "total_milliseconds", simulated["total_milliseconds"],
         "milliseconds"),
        ("summary", "wait_return_cycles", simulated["wait_return_cycles"],
         "normalized_cycles"),
    ]
    rows.extend(
        ("system_stage", stage["name"], stage["duration_cycles"],
         "normalized_cycles")
        for stage in report["system_stages"]
    )
    rows.extend(
        ("critical_worker_stage", name, value, "normalized_cycles")
        for name, value in report["critical_worker"]["stage_cycles"].items()
    )
    rows.extend(
        ("critical_worker_pipeline", name, value, "normalized_cycles")
        for name, value in report["critical_worker"][
            "aggregate_online_pipeline_cycles"
        ].items()
    )
    prefetch = report["critical_worker"].get("kv_prefetch_timing", {})
    rows.extend(
        ("kv_prefetch", f"{name}_cycles", value, "normalized_cycles")
        for name, value in prefetch.get("cycles", {}).items()
    )
    rows.extend(
        ("kv_prefetch", f"{name}_count", value, "count")
        for name, value in prefetch.get("counts", {}).items()
    )
    lookahead = report.get("kv_second_lookahead_window", {})
    for name in (
        "max_candidates", "candidates", "prefetches", "ready_at_release",
        "ready_after_release_before_boundary", "max_available_lead_cycles",
    ):
        if name in lookahead:
            unit = "normalized_cycles" if name == "max_available_lead_cycles" else "count"
            rows.append(("kv_second_lookahead", name, lookahead[name], unit))
    if "candidate_rate" in lookahead:
        rows.append((
            "kv_second_lookahead", "candidate_rate",
            lookahead["candidate_rate"], "ratio",
        ))
    wcp = report["wcp"]
    for name in (
        "queue_depth", "issue_width", "command_latency_cycles",
        "completion_latency_cycles", "max_worker_queue_wait_cycles",
    ):
        if name in wcp:
            unit = "entries" if name == "queue_depth" else (
                "commands_per_wcp_component_cycle" if name == "issue_width"
                else "wcp_component_cycles"
            )
            rows.append(("wcp", name, wcp[name], unit))
    if wcp.get("clock_domain") is not None:
        rows.append(("wcp", "clock_domain", wcp["clock_domain"], "label"))
    if wcp.get("clock_hz") is not None:
        rows.append(("wcp", "clock_hz", wcp["clock_hz"], "hertz"))
    wcp_cycle_metrics = {
        "gemm_proxy_queue_wait_cycles",
        "gemm_proxy_completion_delay_cycles",
    }
    rows.extend(
        (
            "wcp", name, value,
            "wcp_component_cycles" if name in wcp_cycle_metrics else "count",
        )
        for name, value in wcp.get("worker_totals", {}).items()
    )
    broadcast = report.get("matrix_broadcast_fabric", {})
    broadcast_units = {
        "max_fanout": "arrays",
        "bytes_per_cycle": "bytes_per_array_component_cycle",
        "base_latency_cycles": "array_component_cycles",
        "stage_latency_cycles": "array_component_cycles",
        "tree_stages": "stages",
        "fanout": "arrays",
        "payload_bytes": "bytes",
        "cycles_per_request": "array_component_cycles",
        "max_observed_fanout": "arrays",
    }
    for name, unit in broadcast_units.items():
        if name in broadcast:
            rows.append(("matrix_broadcast", name, broadcast[name], unit))
    broadcast_total_units = {
        "matrix_broadcast_requests": "broadcasts",
        "matrix_broadcast_rejected": "broadcasts",
        "matrix_broadcast_ingress_bytes": "bytes",
        "matrix_broadcast_sink_bytes": "bytes",
        "matrix_broadcast_transfer_cycles": "array_component_cycles",
        "matrix_broadcast_fanout": "array_destinations",
    }
    rows.extend(
        ("matrix_broadcast", name, value, broadcast_total_units[name])
        for name, value in broadcast.get("worker_totals", {}).items()
        if name in broadcast_total_units
    )
    active_k = report.get("pv_active_k", {})
    active_k_units = {
        "active_columns": "columns",
        "mac_per_cu_per_cycle": "macs_per_array_component_cycle",
        "pipeline_depth": "array_component_cycles",
        "clock_hz": "hertz",
        "clock_domain": "label",
    }
    for name, unit in active_k_units.items():
        if name in active_k:
            rows.append(("pv_active_k", name, active_k[name], unit))
    active_k_total_units = {
        "active_k_launches": "count",
        "active_k_columns": "columns",
        "active_k_compute_cycles": "array_component_cycles",
        "active_k_full_width_cycles_avoided": "array_component_cycles",
    }
    rows.extend(
        ("pv_active_k", name, value, active_k_total_units[name])
        for name, value in active_k.get("worker_totals", {}).items()
        if name in active_k_total_units
    )
    qk_early = report.get("qk_early_compute", {})
    for name, unit in {
        "enabled": "boolean",
        "configured": "boolean",
        "clock_hz": "hertz",
        "clock_domain": "label",
    }.items():
        if name in qk_early:
            rows.append(("qk_early_compute", name, qk_early[name], unit))
    if "arrays" in qk_early.get("worker_totals", {}):
        rows.append((
            "qk_early_compute", "arrays",
            qk_early["worker_totals"]["arrays"], "count",
        ))
    qk_input_pipeline = report.get("qk_input_pipeline", {})
    for name, unit in {
        "enabled": "boolean",
        "configured": "boolean",
        "depth": "slots",
        "row_bytes": "bytes",
        "capacity_bytes": "bytes",
        "clock_hz": "hertz",
        "clock_domain": "label",
    }.items():
        if name in qk_input_pipeline:
            rows.append((
                "qk_input_pipeline", name, qk_input_pipeline[name], unit,
            ))
    rows.extend(
        (
            "qk_input_pipeline", name, value,
            "rocc_component_cycles" if name == "overlap_ticks" else "count",
        )
        for name, value in qk_input_pipeline.get("worker_totals", {}).items()
    )
    qk_row_burst = report.get("qk_panel_row_burst", {})
    for name, unit in {
        "enabled": "boolean",
        "mode": "label",
        "capacity_bytes": "bytes",
        "banks": "banks",
        "bank_bytes_per_cycle": "bytes_per_wcp_component_cycle",
        "row_burst_bytes": "bytes",
        "clock_domain": "label",
        "clock_hz": "hertz",
    }.items():
        if name in qk_row_burst:
            rows.append(("qk_panel_row_burst", name, qk_row_burst[name], unit))
    qk_row_burst_cycle_metrics = {
        "attention_tile_storage_write_wait_cycles",
        "attention_tile_storage_read_wait_cycles",
    }
    rows.extend(
        (
            "qk_panel_row_burst", name, value,
            "wcp_component_cycles"
            if name in qk_row_burst_cycle_metrics else
            ("bytes" if name.endswith("_bytes") else "count"),
        )
        for name, value in qk_row_burst.get("worker_totals", {}).items()
    )
    v_tile = report.get("pv_v_tile_buffer", {})
    v_tile_units = {
        "capacity_bytes": "bytes",
        "bytes_per_cycle": "bytes_per_rocc_component_cycle",
        "base_latency_cycles": "rocc_component_cycles",
        "panel_bytes": "bytes",
        "clock_hz": "hertz",
        "clock_domain": "label",
    }
    for name, unit in v_tile_units.items():
        if name in v_tile:
            rows.append(("pv_v_tile_buffer", name, v_tile[name], unit))
    v_tile_total_units = {
        "hits": "count",
        "misses": "count",
        "bytes_read": "bytes",
        "bytes_reused": "bytes",
        "wait_ticks": "rocc_component_cycles",
        "capacity_rejections": "count",
    }
    rows.extend(
        ("pv_v_tile_buffer", name, value, v_tile_total_units[name])
        for name, value in v_tile.get("worker_totals", {}).items()
        if name in v_tile_total_units
    )
    verification = report["numerical_verification"]
    for name in ("checked", "mismatches", "max_abs_error"):
        if name in verification:
            rows.append(("numerical", name, verification[name], "value"))
    return rows


def write_csv(report, path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="ascii") as stream:
        writer = csv.writer(stream)
        writer.writerow(("scope", "metric", "value", "unit"))
        writer.writerows(_csv_rows(report))


def _color_enabled():
    if os.environ.get("GOLEM_ATTENTION_COLOR") == "1":
        return True
    if os.environ.get("GOLEM_ATTENTION_COLOR") == "0":
        return False
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def _color(code, value):
    text = str(value)
    return f"\033[{code}m{text}\033[0m" if _color_enabled() else text


def _result_metric(label, value, *, value_color="1;35"):
    print(f"  {_color('1;36', f'{label:<15}')} {_color(value_color, value)}")


def _formatted(value, format_spec, suffix=""):
    return "n/a" if value is None else f"{value:{format_spec}}{suffix}"


def _critical_worker_operator_cycles(critical):
    phase_cycles = critical.get("tile_pipeline_breakdown", {}).get(
        "phase_cycles", {}
    )
    groups = (
        ("Input movement", ("kv_load", "q_local_read")),
        ("QK", (
            "qk_matrix_program", "qk_input_program", "qk_compute_readout",
        )),
        ("Softmax", ("softmax",)),
        ("PV", (
            "pv_matrix_program", "pv_input_program", "pv_restore_output",
            "pv_compute", "pv_output_readwrite",
        )),
    )
    return [
        (label, sum(phase_cycles[name] for name in names))
        for label, names in groups
        if all(phase_cycles.get(name) is not None for name in names)
    ]


def print_summary(report):
    timing = report["simulated_timing"]
    status = report["status"]
    print()
    print(_color("1;36", "== RESULT =="))
    _result_metric("Status", status, value_color="1;32" if status == "PASS" else "1;31")
    configuration = report["configuration"]
    gemm_mode = "generic GEMM + WCP" if configuration["generic_gemm"] else "direct GEMM"
    _result_metric(
        "Configuration",
        f"{configuration['profile'].upper()} | MPI {configuration['mpi_ranks']} | {gemm_mode}",
        value_color="1;35",
    )
    source_status = report["source_status"]
    verification_parts = [
        f"numerical={source_status['numerical']}",
        f"lifecycle={source_status['lifecycle']}",
    ]
    if source_status["mpi_partition"] != "NOT_APPLICABLE":
        verification_parts.append(
            f"mpi_partition={source_status['mpi_partition']}"
        )
    if source_status["baseline"] != "NOT_APPLICABLE":
        verification_parts.append(f"baseline={source_status['baseline']}")
    _result_metric(
        "Verification", ", ".join(verification_parts),
        value_color="1;32" if all(value == "PASS" for value in source_status.values()
                                    if value != "NOT_APPLICABLE") else "1;31",
    )
    _result_metric(
        "Wall time",
        f"SST {report['host_timing']['sst_wall_seconds']:.3f}s | "
        f"pre-report pipeline {report['host_timing']['pipeline_wall_seconds']:.3f}s",
        value_color="1;33",
    )
    _result_metric(
        "Simulated", _formatted(timing["total_milliseconds"], ".6f", " ms"),
        value_color="1;32",
    )
    _result_metric(
        "Total cycles", _formatted(
            timing["total_cycles"], ",", " normalized cycles"
        ), value_color="1;32",
    )
    _result_metric("SST ticks", _formatted(timing["total_ticks"], ","), value_color="0;36")
    _result_metric(
        "Wait return",
        f"{_formatted(timing['wait_return_cycles'], ',', ' normalized cycles')} | "
        f"{_formatted(timing['wait_return_milliseconds'], '.6f', ' ms')}",
        value_color="1;33",
    )
    numerical = report["numerical_verification"]
    _result_metric(
        "Numerical",
        f"checked={numerical.get('checked', 0):,} | "
        f"mismatches={numerical.get('mismatches', 0):,} | "
        f"max_abs_error={numerical.get('max_abs_error', 0):.3e}",
        value_color="1;32"
        if source_status["numerical"] == "PASS"
        and numerical.get("mismatches", 0) == 0
        else "1;31",
    )
    critical = report["critical_worker"]
    print(f"  {_color('1;36', 'Critical-worker operator cycles')}")
    for label, cycles in _critical_worker_operator_cycles(critical):
        print(
            f"    {_color('2;37', f'{label:<14}')} "
            f"{_color('1;33', f'{cycles:>12,}')}"
        )
    _result_metric("Critical worker", f"core{critical['core']}", value_color="1;35")
    prefetch = critical.get("kv_prefetch_timing", {})
    prefetch_counts = prefetch.get("counts", {})
    prefetch_cycles = prefetch.get("cycles", {})
    if prefetch_counts:
        _result_metric(
            "K/V prefetch",
            f"DMA={prefetch_counts.get('dma', 0):,} | "
            f"hits={prefetch_counts.get('ready_lead', 0):,} | "
            f"waits={prefetch_counts.get('wait', 0):,} | "
            f"wait={prefetch_cycles.get('wait', 0):,} cycles",
            value_color="1;33",
        )
    lookahead = report.get("kv_second_lookahead_window", {})
    if lookahead.get("max_candidates", 0):
        _result_metric(
            "N+2 lookahead",
            f"launched={lookahead.get('prefetches', 0):,}/"
            f"{lookahead.get('max_candidates', 0):,} | "
            f"rate={100.0 * lookahead.get('candidate_rate', 0.0):.1f}% | "
            f"ready={lookahead.get('ready_at_release', 0):,} | "
            f"late-ready="
            f"{lookahead.get('ready_after_release_before_boundary', 0):,}",
            value_color="1;34",
        )
    cross_query = report.get("kv_cross_query_prefetch", {})
    if cross_query.get("enabled"):
        totals = cross_query.get("worker_totals", {})
        _result_metric(
            "K/V cross-query",
            f"prefetch={totals.get('prefetches', 0):,} | "
            f"hits={totals.get('hits', 0):,} | "
            f"waits={totals.get('waits', 0):,} | "
            f"wait={totals.get('wait_ticks', 0):,} cycles",
            value_color="1;32",
        )
    broadcast = report.get("matrix_broadcast_fabric", {})
    if broadcast.get("enabled"):
        totals = broadcast.get("worker_totals", {})
        enabled_operators = [
            name for name, enabled in (
                ("QK", broadcast.get("qk_enabled")),
                ("PV", broadcast.get("pv_enabled")),
            ) if enabled
        ]
        broadcast_label = "+".join(enabled_operators) or "Matrix"
        _result_metric(
            f"{broadcast_label} broadcast",
            f"binary tree | fanout={broadcast.get('fanout', 0)} | "
            f"{broadcast.get('bytes_per_cycle', 0)} B/array-cycle | "
            f"{broadcast.get('tree_stages', 0)} stages",
            value_color="1;34",
        )
        _result_metric(
            "Broadcast work",
            f"requests={totals.get('matrix_broadcast_requests', 0):,} | "
            f"ingress={totals.get('matrix_broadcast_ingress_bytes', 0):,} B | "
            f"sinks={totals.get('matrix_broadcast_sink_bytes', 0):,} B | "
            f"transfer={totals.get('matrix_broadcast_transfer_cycles', 0):,} "
            "array-cycle sum",
            value_color="1;35",
        )
    else:
        _result_metric("Matrix broadcast", "disabled")
    active_k = report.get("pv_active_k", {})
    if active_k.get("enabled"):
        totals = active_k.get("worker_totals", {})
        _result_metric(
            "PV active-K",
            f"launches={totals.get('active_k_launches', 0):,} | "
            f"active={active_k.get('active_columns', 0)} columns | "
            f"compute={totals.get('active_k_compute_cycles', 0):,} | "
            f"avoided={totals.get('active_k_full_width_cycles_avoided', 0):,} "
            "array-cycle sum",
            value_color="1;34",
        )
    qk_early = report.get("qk_early_compute", {})
    if qk_early.get("enabled"):
        totals = qk_early.get("worker_totals", {})
        _result_metric(
            "QK early compute",
            f"arrays={totals.get('arrays', 0):,} | "
            "input-programming overlap enabled",
            value_color="1;32",
        )
    qk_input_pipeline = report.get("qk_input_pipeline", {})
    if qk_input_pipeline.get("enabled"):
        totals = qk_input_pipeline.get("worker_totals", {})
        _result_metric(
            "QK input pipe",
            f"storage={qk_input_pipeline.get('capacity_bytes', 0):,} B | "
            f"depth={qk_input_pipeline.get('depth', 0)} | "
            f"rows={totals.get('rows_programmed', 0):,} | "
            f"overlap={totals.get('overlap_ticks', 0):,} | "
            f"slot-full={totals.get('slot_full_stalls', 0):,} | "
            f"mismatch={totals.get('tag_mismatches', 0):,}",
            value_color="1;32",
        )
    qk_row_burst = report.get("qk_panel_row_burst", {})
    if qk_row_burst.get("enabled"):
        totals = qk_row_burst.get("worker_totals", {})
        _result_metric(
            "QK row burst",
            f"storage={qk_row_burst.get('capacity_bytes', 0):,} B | "
            f"banks={qk_row_burst.get('banks', 0)} | "
            f"columns={totals.get('attention_tile_storage_column_writes', 0):,} | "
            f"row bursts={totals.get('attention_tile_storage_row_reads', 0):,} | "
            f"read/write={totals.get('attention_tile_storage_read_bytes', 0):,}/"
            f"{totals.get('attention_tile_storage_write_bytes', 0):,} B",
            value_color="1;32",
        )
    v_tile = report.get("pv_v_tile_buffer", {})
    if v_tile.get("enabled"):
        totals = v_tile.get("worker_totals", {})
        _result_metric(
            "PV V-tile",
            f"hits={totals.get('hits', 0):,} | "
            f"misses={totals.get('misses', 0):,} | "
            f"fill={totals.get('bytes_read', 0):,} B | "
            f"reused={totals.get('bytes_reused', 0):,} B | "
            f"wait={totals.get('wait_ticks', 0):,} RoCC component-cycle sum",
            value_color="1;35",
        )
    if report["wcp"]:
        wcp = report["wcp"]
        totals = wcp.get("worker_totals", {})
        _result_metric(
            "WCP GEMM proxy",
            f"depth={wcp.get('queue_depth', 0)} | "
            f"issue={wcp.get('issue_width', 0)}/cycle | "
            f"command={wcp.get('command_latency_cycles', 0)} cycles | "
            f"completion={wcp.get('completion_latency_cycles', 0)} cycles",
            value_color="1;34",
        )
        _result_metric(
            "WCP activity",
            f"commands={totals.get('gemm_proxy_commands_issued', 0):,} | "
            f"queue_wait={totals.get('gemm_proxy_queue_wait_cycles', 0):,} | "
            f"completion_delay="
            f"{totals.get('gemm_proxy_completion_delay_cycles', 0):,} "
            "component-cycle sum",
            value_color="1;35",
        )
    else:
        _result_metric("WCP GEMM proxy", "disabled")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lifecycle-json", required=True)
    parser.add_argument("--numerical-json", required=True)
    parser.add_argument("--mpi-partition-json")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--mpi-ranks", type=int, required=True)
    parser.add_argument("--sst-wall-seconds", type=float, required=True)
    parser.add_argument("--pipeline-wall-seconds", type=float, required=True)
    parser.add_argument("--baseline-json")
    parser.add_argument("--generic-gemm", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    lifecycle = _load_verification_result(args.lifecycle_json, "lifecycle")
    numerical = _load_verification_result(args.numerical_json, "numerical")
    mpi_partition = None
    if args.mpi_ranks > 1:
        mpi_partition = {}
        if args.mpi_partition_json:
            try:
                mpi_partition = json.loads(
                    Path(args.mpi_partition_json).read_text(encoding="ascii")
                )
            except (OSError, json.JSONDecodeError) as error:
                mpi_partition = {
                    "status": "FAIL",
                    "artifact_error": str(error),
                }
    baseline = None
    if args.baseline_json:
        try:
            baseline = json.loads(
                Path(args.baseline_json).read_text(encoding="ascii")
            )
        except (OSError, json.JSONDecodeError) as error:
            baseline = {"status": "FAIL", "error": str(error)}
    report = build_report(
        lifecycle, numerical, profile=args.profile, mpi_ranks=args.mpi_ranks,
        sst_wall_seconds=args.sst_wall_seconds,
        pipeline_wall_seconds=args.pipeline_wall_seconds,
        generic_gemm=args.generic_gemm,
        mpi_partition_result=mpi_partition,
        baseline_result=baseline,
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    write_csv(report, args.output_csv)
    print_summary(report)
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
