#!/usr/bin/env python3
"""Check that selected simulator results are identical across parallel runs."""

import argparse
import csv
import sys
from pathlib import Path


DEFAULT_FIELDS = (
    "simulated_time",
    "exec_total_cycles",
    "gemm_system_latency_cycles",
    "gemm_system_start_cycle",
    "gemm_system_end_cycle",
    "exec_breakdown_compute_active_time",
    "exec_breakdown_prefetch_wait_time",
    "exec_breakdown_writeback_wait_time",
    "exec_breakdown_control_other_time",
    "dma_timeout_retry_sum",
    "dma_read_issue_count_sum",
    "dma_write_issue_count_sum",
    "dma_read_bytes_total_sum",
    "dma_write_bytes_total_sum",
    "dma_write_timeout_retry_sum",
    "dma_completion_sum",
    "dma_write_completion_sum",
    "dma_wait_count_sum",
    "dma_strict_rtt_samples_sum",
    "dma_strict_rtt_cycles_sum",
    "dma_strict_e2e_rtt_samples_sum",
    "dma_strict_e2e_rtt_cycles_sum",
    "noc_total_xbar_stalls",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare deterministic fields for selected run_summary.csv rows."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument(
        "--run-id",
        action="append",
        required=True,
        help="Run ID to compare; repeat in baseline-first order.",
    )
    parser.add_argument(
        "--field",
        action="append",
        dest="fields",
        help="Field to compare; repeat to override the default field set.",
    )
    parser.add_argument("--report", type=Path, help="Optional detailed comparison CSV.")
    return parser.parse_args()


def load_runs(summary: Path, run_ids: list[str]) -> tuple[list[str], dict[str, dict[str, str]]]:
    if not summary.is_file():
        raise ValueError(f"summary file does not exist: {summary}")

    with summary.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "run_id" not in reader.fieldnames:
            raise ValueError(f"missing run_id column: {summary}")
        rows: dict[str, dict[str, str]] = {}
        duplicates: set[str] = set()
        wanted = set(run_ids)
        for row in reader:
            run_id = row.get("run_id", "")
            if run_id not in wanted:
                continue
            if run_id in rows:
                duplicates.add(run_id)
            rows[run_id] = row

    missing = [run_id for run_id in run_ids if run_id not in rows]
    if missing:
        raise ValueError(f"run IDs not found in {summary}: {', '.join(missing)}")
    if duplicates:
        raise ValueError(f"duplicate run IDs in {summary}: {', '.join(sorted(duplicates))}")
    return reader.fieldnames, rows


def write_report(path: Path, comparisons: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("run_id", "mpi_ranks", "field", "baseline", "actual", "status"),
        )
        writer.writeheader()
        writer.writerows(comparisons)


def main() -> int:
    args = parse_args()
    if len(args.run_id) < 2:
        print("[ERROR] at least two --run-id arguments are required", file=sys.stderr)
        return 2
    if len(set(args.run_id)) != len(args.run_id):
        print("[ERROR] --run-id values must be unique", file=sys.stderr)
        return 2

    fields = tuple(args.fields or DEFAULT_FIELDS)
    try:
        headers, rows = load_runs(args.summary, args.run_id)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    unknown = [field for field in fields if field not in headers]
    if unknown:
        print(f"[ERROR] fields missing from summary: {', '.join(unknown)}", file=sys.stderr)
        return 2

    baseline_id = args.run_id[0]
    baseline = rows[baseline_id]
    comparisons: list[dict[str, str]] = []
    failures: list[str] = []

    for run_id in args.run_id:
        row = rows[run_id]
        for field in fields:
            expected = baseline.get(field, "").strip()
            actual = row.get(field, "").strip()
            status = "PASS"
            if not expected or not actual:
                status = "MISSING"
            elif actual != expected:
                status = "MISMATCH"
            comparisons.append(
                {
                    "run_id": run_id,
                    "mpi_ranks": row.get("mpi_ranks", ""),
                    "field": field,
                    "baseline": expected,
                    "actual": actual,
                    "status": status,
                }
            )
            if status != "PASS":
                failures.append(
                    f"{run_id}: {field}: baseline={expected!r}, actual={actual!r} ({status})"
                )

    if args.report:
        write_report(args.report, comparisons)

    print("run_id,mpi_ranks,simulated_time,exec_total_cycles,gemm_system_end_cycle")
    for run_id in args.run_id:
        row = rows[run_id]
        print(
            ",".join(
                (
                    run_id,
                    row.get("mpi_ranks", ""),
                    row.get("simulated_time", ""),
                    row.get("exec_total_cycles", ""),
                    row.get("gemm_system_end_cycle", ""),
                )
            )
        )

    if failures:
        print(f"[FAIL] {len(failures)} deterministic field comparisons failed", file=sys.stderr)
        for failure in failures[:20]:
            print(f"  {failure}", file=sys.stderr)
        if len(failures) > 20:
            print(f"  ... {len(failures) - 20} more; see report CSV", file=sys.stderr)
        return 1

    print(
        f"[PASS] {len(fields)} deterministic fields match across "
        f"{len(args.run_id)} runs (baseline={baseline_id})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
