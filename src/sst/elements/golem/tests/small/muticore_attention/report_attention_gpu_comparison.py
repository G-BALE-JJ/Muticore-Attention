#!/usr/bin/env python3
"""Build a comparable SST/GPU Attention latency and stage report."""

import argparse
import json
import math
import statistics
from pathlib import Path


PROFILE_SHAPES = {
    "e3": {"batch": 1, "heads": 1, "sequence": 1024, "head_dim": 128},
    "e4": {"batch": 1, "heads": 1, "sequence": 2048, "head_dim": 128},
}

GPU_STAGE_STREAM = "current default stream"
GPU_STAGE_METHOD = "consecutive CUDA Events on one stream; one final synchronize"
GPU_BENCHMARK = "single-head FP32 non-causal scaled dot-product attention"
GPU_DEVICE = "NVIDIA GeForce RTX 5060"

SST_STAGE_GROUPS = {
    "input_movement": ("kv_load", "q_local_read"),
    "qk": ("qk_matrix_program", "qk_input_program", "qk_compute_readout"),
    "scale_softmax": ("softmax",),
    "pv": (
        "pv_matrix_program",
        "pv_input_program",
        "pv_restore_output",
        "pv_compute",
        "pv_output_readwrite",
    ),
}


def load_json(path):
    with Path(path).open(encoding="ascii") as stream:
        return json.load(stream)


def timing_from_ticks(ticks, timebase_hz, normalization_hz):
    milliseconds = ticks * 1000.0 / timebase_hz
    normalized_cycles = (
        ticks * normalization_hz + timebase_hz - 1
    ) // timebase_hz
    return {
        "sst_timebase_ticks": ticks,
        "milliseconds": milliseconds,
        "normalized_cycles": normalized_cycles,
    }


def build_sst_profile(profile, document):
    if document.get("status") != "PASS":
        raise ValueError(f"{profile}: SST lifecycle status is not PASS")
    lifecycle = document.get("lifecycle", {})
    clock = lifecycle.get("clock_contract")
    if not clock:
        raise ValueError(f"{profile}: lifecycle is missing clock_contract")
    timebase_hz = clock["sst_timebase_ticks_per_second"]
    normalization_hz = clock["normalization_clock_hz"]
    critical = lifecycle.get("worker_critical_path", {})
    pipeline = critical.get("tile_pipeline_breakdown", {})
    frontier = lifecycle.get("system_frontier", {})
    attribution = frontier.get("accelerator_attribution", {})
    if not pipeline.get("conservation_valid"):
        raise ValueError(f"{profile}: worker tile pipeline does not conserve ticks")
    if not attribution.get("conservation_valid"):
        raise ValueError(f"{profile}: system frontier does not conserve ticks")

    phase_ticks = pipeline["phase_ticks"]
    grouped_ticks = {
        group: sum(phase_ticks[name] for name in members)
        for group, members in SST_STAGE_GROUPS.items()
    }
    grouped = {
        name: timing_from_ticks(ticks, timebase_hz, normalization_hz)
        for name, ticks in grouped_ticks.items()
    }
    completion_ticks = lifecycle["accelerator_completion_ticks"]
    worker_ticks = pipeline["total_ticks"]
    milestones = critical.get("milestone_ticks", {})
    worker_interval_ticks = (
        milestones.get("final_output_dma_ack", 0)
        - milestones.get("dispatch_accept", 0)
    )
    if worker_interval_ticks <= 0:
        raise ValueError(f"{profile}: invalid slowest-worker milestone interval")
    return {
        "shape": PROFILE_SHAPES[profile],
        "clock_contract": clock,
        "accelerator_completion": timing_from_ticks(
            completion_ticks, timebase_hz, normalization_hz
        ),
        "system_frontier": {
            "stage_ticks": frontier["stage_ticks"],
            "attribution": attribution,
            "interpretation": frontier["interpretation"],
        },
        "slowest_worker_work": {
            "core": critical["slowest_worker_core"],
            "tile_count": pipeline["tile_count"],
            "grouped_stages": grouped,
            "detailed_phase_ticks": phase_ticks,
            "total": timing_from_ticks(worker_ticks, timebase_hz, normalization_hz),
            "coverage_of_accelerator_completion_ratio": (
                worker_ticks / completion_ticks if completion_ticks else 0.0
            ),
            "worker_interval": timing_from_ticks(
                worker_interval_ticks, timebase_hz, normalization_hz
            ),
            "coverage_of_worker_interval_ratio": worker_ticks / worker_interval_ticks,
            "conservation_valid": pipeline["conservation_valid"],
            "interpretation": (
                "Non-overlapping accumulated work on the slowest worker. Do not "
                "sum this value across parallel workers."
            ),
        },
    }


def validate_samples(samples, expected_count, label):
    if not isinstance(samples, list) or len(samples) != expected_count:
        raise ValueError(f"{label}: expected {expected_count} raw samples")
    if any(not isinstance(value, (int, float)) or value <= 0 for value in samples):
        raise ValueError(f"{label}: samples must be positive numbers")


def validate_median(samples, expected, label):
    if not math.isclose(statistics.median(samples), expected,
                        rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise ValueError(f"{label}: median does not match raw samples")


def validate_gpu_measurement(document):
    measurement = document.get("measurement", {})
    if measurement.get("timer") != "CUDA Event":
        raise ValueError("GPU measurement must use CUDA Event timing")
    for field in ("warmup", "iterations"):
        if not isinstance(measurement.get(field), int) or measurement[field] < 1:
            raise ValueError(f"GPU measurement {field} must be a positive integer")
    for field in ("h2d_included", "d2h_included"):
        if measurement.get(field) is not False:
            raise ValueError(f"GPU measurement must set {field}=false")
    if measurement.get("dtype_conversion_included") is not False:
        raise ValueError("GPU measurement must set dtype_conversion_included=false")
    return measurement


def validate_gpu_profile(profile, data, measurement):
    expected = PROFILE_SHAPES[profile]
    if data.get("shape") != expected:
        raise ValueError(f"{profile}: GPU shape does not match {expected}")
    fp32 = data.get("fp32_math", {})
    scope_a = fp32.get("scope_a", {})
    median_ms = scope_a.get("median_ms")
    if not isinstance(median_ms, (int, float)) or median_ms <= 0:
        raise ValueError(f"{profile}: invalid GPU FP32 Scope A median")
    if scope_a.get("normalization_clock_hz") != 1_000_000_000:
        raise ValueError(f"{profile}: GPU result must use 1 GHz normalization")
    if scope_a.get("median_normalized_cycles") != round(median_ms * 1_000_000):
        raise ValueError(f"{profile}: normalized cycles do not match Scope A median")
    evidence = fp32.get("evidence_level")
    chain = fp32.get("stage_chain", {})
    if evidence == "summary_only":
        if scope_a.get("samples_ms") is not None:
            raise ValueError(f"{profile}: summary-only Scope A samples must be null")
        if chain.get("status") != "pending_external_measurement":
            raise ValueError(f"{profile}: summary-only input cannot claim measured stages")
        return fp32
    if evidence != "raw_samples":
        raise ValueError(f"{profile}: unknown FP32 evidence_level")
    if measurement.get("tf32_allowed") is not False:
        raise ValueError("measured GPU FP32 stages require tf32_allowed=false")
    if measurement.get("stream") != GPU_STAGE_STREAM:
        raise ValueError(
            f"{profile}: measured GPU stages require stream={GPU_STAGE_STREAM!r}"
        )
    if chain.get("status") != "measured":
        raise ValueError(f"{profile}: raw-sample input must contain measured stages")
    if chain.get("method") != GPU_STAGE_METHOD:
        raise ValueError(
            f"{profile}: measured GPU stages require method={GPU_STAGE_METHOD!r}"
        )

    floor_iterations = measurement.get("empty_event_iterations")
    if not isinstance(floor_iterations, int) or floor_iterations < 1:
        raise ValueError("measured GPU stages require empty-event iterations")
    floor_samples = measurement.get("empty_event_samples_ms")
    validate_samples(floor_samples, floor_iterations, "GPU empty Event")
    floor_median = measurement.get("empty_event_median_ms")
    if not isinstance(floor_median, (int, float)) or floor_median <= 0:
        raise ValueError("GPU empty Event median must be positive")
    validate_median(floor_samples, floor_median, "GPU empty Event")

    correctness = fp32.get("correctness", {})
    if correctness.get("passed") is not True or correctness.get("finite") is not True:
        raise ValueError(f"{profile}: GPU correctness must pass and be finite")
    expected_output_shape = [
        expected["batch"], expected["heads"],
        expected["sequence"], expected["head_dim"],
    ]
    if correctness.get("output_shape") != expected_output_shape:
        raise ValueError(f"{profile}: GPU output shape is invalid")
    tolerance = correctness.get("tolerance")
    if not isinstance(tolerance, (int, float)) or tolerance != 1.0e-6:
        raise ValueError(f"{profile}: GPU correctness tolerance must be 1e-6")
    errors = (
        correctness.get("scope_a_max_abs_error"),
        correctness.get("stage_chain_max_abs_error"),
        correctness.get("max_abs_error"),
    )
    if any(not isinstance(value, (int, float)) or value < 0 or value > tolerance
           for value in errors):
        raise ValueError(f"{profile}: GPU FP32 error exceeds tolerance")

    iterations = measurement["iterations"]
    scope_samples = scope_a.get("samples_ms")
    validate_samples(scope_samples, iterations, f"{profile} Scope A")
    validate_median(scope_samples, median_ms, f"{profile} Scope A")
    stage_medians = chain.get("stage_median_ms", {})
    stage_samples = chain.get("stage_samples_ms", {})
    for stage in ("qk", "scale", "softmax", "pv"):
        samples = stage_samples.get(stage) if isinstance(stage_samples, dict) else None
        validate_samples(samples, iterations, f"{profile} {stage}")
        median = stage_medians.get(stage)
        if not isinstance(median, (int, float)) or median <= 0:
            raise ValueError(f"{profile} {stage}: invalid stage median")
        validate_median(samples, median, f"{profile} {stage}")
    chain_samples = chain.get("end_to_end_samples_ms")
    validate_samples(chain_samples, iterations, f"{profile} stage-chain end-to-end")
    validate_median(
        chain_samples, chain.get("end_to_end_median_ms"),
        f"{profile} stage-chain end-to-end",
    )
    if chain.get("output_writeback") != "included in the PV matmul event interval":
        raise ValueError(f"{profile}: GPU output writeback boundary is not explicit")
    return fp32


def build_stage_comparison(sst, fp32):
    chain = fp32.get("stage_chain", {})
    if chain.get("status") != "measured":
        return {
            "status": "pending_external_measurement",
            "reason": "GPU FP32 single-stream stage-chain samples are not available",
        }
    gpu_stages = chain.get("stage_median_ms", {})
    required = ("qk", "scale", "softmax", "pv")
    if any(not isinstance(gpu_stages.get(name), (int, float))
           or gpu_stages[name] <= 0 for name in required):
        raise ValueError("GPU measured stage chain is missing a positive stage median")
    common_gpu = {
        "qk": gpu_stages["qk"],
        "scale_softmax": gpu_stages["scale"] + gpu_stages["softmax"],
        "pv": gpu_stages["pv"],
    }
    sst_stages = sst["slowest_worker_work"]["grouped_stages"]
    comparisons = {}
    for name, gpu_ms in common_gpu.items():
        sst_ms = sst_stages[name]["milliseconds"]
        comparisons[name] = {
            "sst_work_ms": sst_ms,
            "gpu_stage_ms": gpu_ms,
            "sst_over_gpu_ratio": sst_ms / gpu_ms,
        }
    return {
        "status": "measured",
        "stages": comparisons,
        "gpu_stage_chain_sum_ms": sum(gpu_stages[name] for name in required),
        "gpu_stage_chain_end_to_end_median_ms": chain.get("end_to_end_median_ms"),
        "interpretation": (
            "SST values are non-overlapping accumulated work on its slowest worker; "
            "GPU values are consecutive events on one CUDA stream."
        ),
    }


def build_report(sst_documents, gpu_document):
    if gpu_document.get("schema_version") != 1:
        raise ValueError("unsupported GPU result schema_version")
    if gpu_document.get("benchmark") != GPU_BENCHMARK:
        raise ValueError(f"GPU benchmark must be {GPU_BENCHMARK!r}")
    if gpu_document.get("hardware", {}).get("gpu") != GPU_DEVICE:
        raise ValueError(f"GPU device must be {GPU_DEVICE!r}")
    measurement = validate_gpu_measurement(gpu_document)
    profiles = {}
    for profile in ("e3", "e4"):
        sst = build_sst_profile(profile, sst_documents[profile])
        fp32 = validate_gpu_profile(
            profile, gpu_document["profiles"][profile], measurement
        )
        gpu_ms = fp32["scope_a"]["median_ms"]
        sst_ms = sst["accelerator_completion"]["milliseconds"]
        coverage = sst["slowest_worker_work"][
            "coverage_of_accelerator_completion_ratio"
        ]
        interval_coverage = sst["slowest_worker_work"][
            "coverage_of_worker_interval_ratio"
        ]
        coverage_pass = (
            0.95 <= coverage <= 1.0 + 1.0e-9
            and 0.95 <= interval_coverage <= 1.0 + 1.0e-9
        )
        profiles[profile] = {
            "sst": sst,
            "gpu_fp32_scope_a": fp32["scope_a"],
            "end_to_end_gap": {
                "sst_over_gpu_ratio": sst_ms / gpu_ms,
                "gpu_faster_than_sst": sst_ms > gpu_ms,
            },
            "stage_comparison": build_stage_comparison(sst, fp32),
            "sst_coverage_gate": {
                "minimum_ratio": 0.95,
                "maximum_ratio": 1.0,
                "observed_ratio": coverage,
                "worker_interval_ratio": interval_coverage,
                "pass": coverage_pass,
            },
        }
    stage_statuses = {
        data["stage_comparison"]["status"] for data in profiles.values()
    }
    return {
        "schema_version": 1,
        "comparison": "single-head FP32 non-causal scaled dot-product attention",
        "cycle_semantics": "1 GHz normalized latency, not native GPU/model cycles",
        "sst_report_status": (
            "PASS" if all(p["sst_coverage_gate"]["pass"] for p in profiles.values())
            else "FAIL"
        ),
        "gpu_stage_status": (
            "measured" if stage_statuses == {"measured"}
            else "pending_external_measurement"
        ),
        "profiles": profiles,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sst-e3", required=True)
    parser.add_argument("--sst-e4", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--result-json")
    args = parser.parse_args()
    report = build_report(
        {"e3": load_json(args.sst_e3), "e4": load_json(args.sst_e4)},
        load_json(args.gpu),
    )
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.result_json:
        output = Path(args.result_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="ascii")
    if report["sst_report_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
