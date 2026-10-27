#!/usr/bin/env python3
"""Verify one Attention result against its frozen numerical/architecture baseline."""

import argparse
import json
import math
import sys
from pathlib import Path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="ascii"))


def compare_expected(checks, prefix, actual, expected):
    """Compare every baseline-declared leaf while allowing new report fields."""
    if isinstance(expected, dict):
        actual_mapping = actual if isinstance(actual, dict) else {}
        for name, value in expected.items():
            compare_expected(
                checks,
                f"{prefix}.{name}" if prefix else name,
                actual_mapping.get(name),
                value,
            )
        return
    checks[prefix] = actual == expected


def build_checks(baseline, result, lifecycle, mpi_ranks, partition=None):
    lifecycle_data = lifecycle.get("lifecycle", {})
    baseline_architecture = baseline["architecture"]
    actual_wcp = lifecycle_data.get("wcp_gemm_proxy", {})
    actual_fabric = lifecycle_data.get("matrix_broadcast_fabric", {})
    actual_worker_path = lifecycle_data.get("worker_critical_path", {})
    actual_kv_prefetch = actual_worker_path.get("kv_prefetch_timing", {})
    checks = {
        "verification.status": (
            result.get("status") == baseline["verification"]["status"]
        ),
        "verification.checked": (
            result.get("checked") == baseline["verification"]["checked"]
        ),
        "verification.mismatches": (
            result.get("mismatches") == baseline["verification"]["mismatches"]
        ),
        "verification.shape": result.get("shape") == {
            "queries": baseline["shape"]["queries"],
            "keys": baseline["shape"]["keys"],
            "head_dim": baseline["shape"]["head_dim"],
        },
        "verification.score_probability_hbm_bytes": (
            result.get("score_probability_hbm_bytes")
            == baseline["verification"]["score_probability_hbm_bytes"]
        ),
        "verification.max_abs_error": math.isclose(
            result.get("max_abs_error", math.inf),
            baseline["verification"]["max_abs_error"],
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ),
        "lifecycle.status": (
            lifecycle.get("status") == baseline["lifecycle"]["status"]
        ),
        "lifecycle.order_valid": lifecycle_data.get(
            "worker_critical_path", {}
        ).get("order_valid") is True,
        "lifecycle.conservation_valid": lifecycle_data.get(
            "worker_critical_path", {}
        ).get("inter_tile_breakdown", {}).get("conservation_valid") is True,
        "lifecycle.accelerator_completion_cycles": (
            lifecycle_data.get("accelerator_completion_cycles")
            == baseline["lifecycle"]["accelerator_completion_cycles"]
        ),
        "lifecycle.wait_return_cycles": (
            lifecycle_data.get("wait_return_cycles")
            == baseline["lifecycle"]["wait_return_cycles"]
        ),
        "topology.mpi_ranks": (
            baseline.get("topology", {}).get("mpi_ranks") == mpi_ranks
        ),
        "architecture.generic_gemm_wcp": (
            bool(actual_wcp)
            == baseline_architecture["generic_gemm_wcp"]
        ),
        "architecture.pv_matrix_broadcast": (
            actual_fabric.get("pv_enabled")
            == baseline_architecture["pv_matrix_broadcast"]
        ),
        "architecture.qk_matrix_broadcast": (
            actual_fabric.get("qk_enabled")
            == baseline_architecture["qk_matrix_broadcast"]
        ),
        "architecture.kv_double_buffer": (
            bool(actual_kv_prefetch.get("counts", {}).get("dma", 0))
            == baseline_architecture["kv_double_buffer"]
        ),
    }
    compare_expected(
        checks,
        "architecture.wcp_gemm_proxy",
        actual_wcp,
        baseline_architecture["wcp_gemm_proxy"],
    )
    compare_expected(
        checks,
        "architecture.matrix_broadcast_fabric",
        actual_fabric,
        baseline_architecture["matrix_broadcast_fabric"],
    )
    compare_expected(
        checks,
        "lifecycle.worker_critical_path.kv_prefetch_timing",
        actual_kv_prefetch,
        baseline["lifecycle"]["worker_critical_path"]["kv_prefetch_timing"],
    )
    if mpi_ranks > 1:
        partition = partition or {}
        checks.update({
            "partition.status": partition.get("status") == "PASS",
            "partition.mpi_ranks": partition.get("mpi_ranks") == mpi_ranks,
            "partition.core_ranks": (
                partition.get("observed_core_ranks")
                == partition.get("expected_core_ranks")
            ),
            "partition.stats_files": (
                len(partition.get("ranked_stats_files", [])) == mpi_ranks
            ),
        })
    return checks


def preflight_checks(baseline, queries, keys, head_dim, mpi_ranks):
    shape = baseline.get("shape", {})
    return {
        "baseline.shape": shape == {
            "queries": queries, "keys": keys, "head_dim": head_dim,
        },
        "baseline.mpi_ranks": (
            baseline.get("topology", {}).get("mpi_ranks") == mpi_ranks
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--result")
    parser.add_argument("--lifecycle")
    parser.add_argument("--partition")
    parser.add_argument("--mpi-ranks", required=True, type=int)
    parser.add_argument("--query-length", "--queries", dest="query_length", type=int)
    parser.add_argument("--kv-length", "--keys", dest="kv_length", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--result-json")
    args = parser.parse_args()
    if args.mpi_ranks > 1 and not args.partition:
        if not args.preflight_only:
            parser.error("--partition is required when --mpi-ranks is greater than 1")
    if args.preflight_only and any(
            value is None for value in (args.query_length, args.kv_length, args.head_dim)):
        parser.error("preflight requires --queries, --keys, and --head-dim")
    if not args.preflight_only and (not args.result or not args.lifecycle):
        parser.error("verification requires --result and --lifecycle")

    try:
        baseline = load_json(args.baseline)
        if args.preflight_only:
            checks = preflight_checks(
                baseline, args.query_length, args.kv_length, args.head_dim, args.mpi_ranks
            )
        else:
            partition = load_json(args.partition) if args.partition else None
            checks = build_checks(
                baseline,
                load_json(args.result),
                load_json(args.lifecycle),
                args.mpi_ranks,
                partition,
            )
        profile = str(baseline.get("profile", Path(args.baseline).stem)).upper()
        error = None
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        checks = {"baseline.readable": False}
        profile = Path(args.baseline).stem.upper()
        error = str(exc)
    failed = [name for name, passed in checks.items() if not passed]
    verification = {
        "status": "FAIL" if failed else "PASS",
        "profile": profile,
        "checks": checks,
        "failed": failed,
    }
    if error is not None:
        verification["error"] = error
    if args.result_json:
        output = Path(args.result_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(verification, indent=2) + "\n", encoding="ascii")
    if failed:
        if args.result_json:
            message = (
                f"[FLASH] {profile} baseline FAIL: {len(failed)} check(s); "
                "see attention_baseline_verification.json"
            )
        else:
            message = f"[FLASH] {profile} baseline FAIL: " + ", ".join(failed)
        print(message, file=sys.stderr)
        raise SystemExit(1)
    action = "preflight PASS" if args.preflight_only else "MATCH"
    print(f"[FLASH] {profile} {args.mpi_ranks}-rank baseline {action}")


if __name__ == "__main__":
    main()
