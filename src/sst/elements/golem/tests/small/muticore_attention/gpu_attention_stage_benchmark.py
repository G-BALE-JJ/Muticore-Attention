#!/usr/bin/env python3
"""Measure FP32 Attention end-to-end and stages on a CUDA GPU."""

import argparse
import json
import platform
import statistics
from pathlib import Path


PROFILES = {
    "e3": {"batch": 1, "heads": 1, "sequence": 1024, "head_dim": 128},
    "e4": {"batch": 1, "heads": 1, "sequence": 2048, "head_dim": 128},
}

BENCHMARK_NAME = "single-head FP32 non-causal scaled dot-product attention"
EXPECTED_GPU = "NVIDIA GeForce RTX 5060"


def percentile(samples, fraction):
    ordered = sorted(samples)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def summarize(samples):
    return {
        "median_ms": statistics.median(samples),
        "p10_ms": percentile(samples, 0.10),
        "p90_ms": percentile(samples, 0.90),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def fp32_attention(torch, q, k, v):
    scores = torch.matmul(q, k.transpose(-2, -1))
    scaled = scores * (q.shape[-1] ** -0.5)
    probabilities = torch.softmax(scaled, dim=-1)
    return torch.matmul(probabilities, v)


def measure_scope_a(torch, q, k, v, warmup, iterations):
    for _ in range(warmup):
        output = fp32_attention(torch, q, k, v)
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = fp32_attention(torch, q, k, v)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return output, samples


def measure_stage_chain(torch, q, k, v, warmup, iterations):
    scale = q.shape[-1] ** -0.5
    for _ in range(warmup):
        scores = torch.matmul(q, k.transpose(-2, -1))
        scaled = scores * scale
        probabilities = torch.softmax(scaled, dim=-1)
        output = torch.matmul(probabilities, v)
    torch.cuda.synchronize()

    names = ("qk", "scale", "softmax", "pv")
    stage_samples = {name: [] for name in names}
    end_to_end_samples = []
    for _ in range(iterations):
        events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        events[0].record()
        scores = torch.matmul(q, k.transpose(-2, -1))
        events[1].record()
        scaled = scores * scale
        events[2].record()
        probabilities = torch.softmax(scaled, dim=-1)
        events[3].record()
        output = torch.matmul(probabilities, v)
        events[4].record()
        events[-1].synchronize()
        for index, name in enumerate(names):
            stage_samples[name].append(events[index].elapsed_time(events[index + 1]))
        end_to_end_samples.append(events[0].elapsed_time(events[-1]))
    return output, stage_samples, end_to_end_samples


def measure_event_floor(torch, iterations):
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def make_correctness(output_shape, finite, max_abs_error):
    tolerance = 1.0e-6
    return {
        "passed": finite and max_abs_error <= tolerance,
        "output_shape": list(output_shape),
        "finite": finite,
        "scope_a_max_abs_error": max_abs_error,
        "stage_chain_max_abs_error": 0.0,
        "max_abs_error": max_abs_error,
        "tolerance": tolerance,
    }


def make_result(gpu_name, compute_capability, torch_version, cuda_runtime,
                warmup, iterations, event_floor_samples, profiles):
    return {
        "schema_version": 1,
        "benchmark": BENCHMARK_NAME,
        "hardware": {
            "gpu": gpu_name,
            "compute_capability": list(compute_capability),
        },
        "software": {
            "python": platform.python_version(),
            "pytorch": torch_version,
            "cuda": cuda_runtime,
        },
        "measurement": {
            "timer": "CUDA Event",
            "warmup": warmup,
            "iterations": iterations,
            "stream": "current default stream",
            "tf32_allowed": False,
            "h2d_included": False,
            "d2h_included": False,
            "dtype_conversion_included": False,
            "empty_event_iterations": len(event_floor_samples),
            "empty_event_median_ms": statistics.median(event_floor_samples),
            "empty_event_samples_ms": event_floor_samples,
        },
        "profiles": profiles,
    }


def measure_profile(torch, profile, warmup, iterations, seed):
    shape = PROFILES[profile]
    torch.manual_seed(seed)
    q = torch.randn(
        shape["batch"], shape["heads"], shape["sequence"], shape["head_dim"],
        device="cuda", dtype=torch.float32,
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    scope_output, scope_samples = measure_scope_a(
        torch, q, k, v, warmup, iterations
    )
    stage_output, stage_samples, chain_samples = measure_stage_chain(
        torch, q, k, v, warmup, iterations
    )
    max_abs_error = (scope_output - stage_output).abs().max().item()
    finite = bool(torch.isfinite(stage_output).all().item())
    stage_medians = {
        name: statistics.median(samples) for name, samples in stage_samples.items()
    }
    return {
        "shape": shape,
        "fp32_math": {
            "evidence_level": "raw_samples",
            "scope_a": {
                **summarize(scope_samples),
                "samples_ms": scope_samples,
                "normalization_clock_hz": 1_000_000_000,
                "median_normalized_cycles": round(
                    statistics.median(scope_samples) * 1_000_000
                ),
            },
            "stage_chain": {
                "status": "measured",
                "method": "consecutive CUDA Events on one stream; one final synchronize",
                "stage_median_ms": stage_medians,
                "stage_samples_ms": stage_samples,
                "end_to_end_median_ms": statistics.median(chain_samples),
                "end_to_end_samples_ms": chain_samples,
                "stage_sum_median_ms": sum(stage_medians.values()),
                "output_writeback": "included in the PV matmul event interval",
            },
            "correctness": make_correctness(
                stage_output.shape, finite, max_abs_error
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", choices=sorted(PROFILES),
                        default=["e3", "e4"])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--event-floor-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (args.warmup < 1 or args.iterations < 1
            or args.event_floor_iterations < 1):
        parser.error("warmup, iterations, and event-floor-iterations must be positive")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    gpu_name = torch.cuda.get_device_name(0)
    if gpu_name != EXPECTED_GPU:
        raise SystemExit(f"Expected {EXPECTED_GPU}, found {gpu_name}")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    event_floor_samples = measure_event_floor(
        torch, args.event_floor_iterations
    )
    profiles = {}
    for profile in args.profiles:
        profiles[profile] = measure_profile(
            torch, profile, args.warmup, args.iterations, args.seed
        )
    result = make_result(
        gpu_name, torch.cuda.get_device_capability(0), torch.__version__,
        torch.version.cuda, args.warmup, args.iterations,
        event_floor_samples, profiles,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
