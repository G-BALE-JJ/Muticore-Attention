#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path
from statistics import mean, median


def parse_stats_sum(stats_path: Path, stat_name: str):
    total = 0.0
    if not stats_path.exists():
        return total
    for line in stats_path.read_text(errors="ignore").splitlines():
        parts = line.split(",")
        if len(parts) < 7:
            continue
        if parts[1] != stat_name:
            continue
        try:
            total += float(parts[6])
        except ValueError:
            continue
    return total


PATTERN = re.compile(
    r"\[Core\s+(?P<core>\d+)\](?:\s+\[(?P<dtype>[^\]]+)\])?\s+(?:CTRL\s+)?LATENCY\(cycles\):\s+"
    r"dma_issue=(?P<dma_issue>\d+)\s+"
    r"dma_wait=(?P<dma_wait>\d+)\s+"
    r"dma_total=(?P<dma_total>\d+)\s+"
    r"compute=(?P<compute>\d+)\s+"
    r"(?:compute_submit=(?P<compute_submit>\d+)\s+)?"
    r"(?:compute_wait=(?P<compute_wait>\d+)\s+)?"
    r"(?:sched_protocol=(?P<sched_protocol>\d+)\s+)?"
    r"(?:c_store=(?P<c_store>\d+)\s+)?"
    r"(?:tile_ready_wait=(?P<tile_ready_wait>\d+)\s+)?"
    r"(?:txn_wait=(?P<txn_wait>\d+)\s+)?"
    r"(?:writeback_wait=(?P<writeback_wait>\d+)\s+)?"
    r"(?:wait_2d_activate=(?P<wait_2d_activate>\d+)\s+)?"
    r"(?:wait_2d_active_not_ready=(?P<wait_2d_active_not_ready>\d+)\s+)?"
    r"(?:wait_non2d_txn=(?P<wait_non2d_txn>\d+)\s+)?"
    r"(?:wait_no_active_txn=(?P<wait_no_active_txn>\d+)\s+)?"
    r"(?:window_submit_active=(?P<window_submit_active>\d+)\s+)?"
    r"(?:window_submit_prefetch=(?P<window_submit_prefetch>\d+)\s+)?"
    r"(?:window_activate=(?P<window_activate>\d+)\s+)?"
    r"(?:window_advance_wait_prefetch=(?P<window_advance_wait_prefetch>\d+)\s+)?"
    r"(?:group_wait=(?P<group_wait>\d+)\s+)?"
    r"(?:poll_iters=(?P<poll_iters>\d+)\s+)?"
    r"(?:overlap_issue=(?P<overlap_issue>\d+)\s+)?"
    r"(?:overlap_wait=(?P<overlap_wait>\d+)\s+)?"
    r"(?:issue_block_q=(?P<issue_block_q>\d+)\s+)?"
    r"(?:issue_write=(?P<issue_write>\d+)\s+)?"
    r"(?:ov_issue_block_q=(?P<ov_issue_block_q>\d+)\s+)?"
    r"(?:ov_issue_write=(?P<ov_issue_write>\d+)\s+)?"
    r"(?:task_desc=(?P<task_desc>\d+)\s+)?"
    r"(?:nloop=(?P<nloop>\d+)\s+)?"
    r"(?:task_loop=(?P<task_loop>\d+)\s+)?"
    r"(?:submit_pack=(?P<submit_pack>\d+)\s+)?"
    r"(?:finish_publish=(?P<finish_publish>\d+)\s+)?"
    r"(?:c_buffer_reads=(?P<c_buffer_reads>\d+)\s+)?"
    r"(?:c_buffer_writes=(?P<c_buffer_writes>\d+)\s+)?"
    r"(?:c_buffer_read_wait=(?P<c_buffer_read_wait>\d+)\s+)?"
    r"total=(?P<total>\d+)"
    r"(?:\s+descriptor_start_cycle=(?P<descriptor_start_cycle>\d+))?"
    r"(?:\s+start_cycle=(?P<start_cycle>\d+)\s+end_cycle=(?P<end_cycle>\d+))?"
)

FIELDS = [
    "core",
    "dtype",
    "dma_issue",
    "dma_wait",
    "dma_total",
    "compute",
    "compute_submit",
    "compute_wait",
    "sched_protocol",
    "c_store",
    "tile_ready_wait",
    "txn_wait",
    "writeback_wait",
    "wait_2d_activate",
    "wait_2d_active_not_ready",
    "wait_non2d_txn",
    "wait_no_active_txn",
    "window_submit_active",
    "window_submit_prefetch",
    "window_activate",
    "window_advance_wait_prefetch",
    "group_wait",
    "poll_iters",
    "overlap_issue",
    "overlap_wait",
    "issue_block_q",
    "issue_write",
    "ov_issue_block_q",
    "ov_issue_write",
    "task_desc",
    "nloop",
    "submit_pack",
    "finish_publish",
    "c_buffer_reads",
    "c_buffer_writes",
    "c_buffer_read_wait",
    "total",
    "descriptor_start_cycle",
    "start_cycle",
    "end_cycle",
]


def _p95(values):
    if not values:
        return 0
    vals = sorted(values)
    idx = int(round(0.95 * (len(vals) - 1)))
    return vals[idx]


def resolve_inputs(log_path: Path | None, log_dir: Path, stdout_glob: str):
    inputs = []
    if log_dir.exists() and log_dir.is_dir():
        inputs.extend(sorted(p for p in log_dir.glob(stdout_glob) if p.is_file()))
    if log_path is not None and log_path.exists() and log_path.is_file():
        # Main simulation log usually contains the latest aggregate lines (e.g. WCP summary).
        # Parse it last so it can override earlier per-core stdout records.
        inputs.append(log_path)
    if not inputs and log_path is not None:
        raise SystemExit(
            f"[ERROR] no input logs found (log={log_path}, dir={log_dir}, glob={stdout_glob})"
        )
    return inputs


def parse_logs(log_paths):
    # Keep latest measurement per core in case logs contain repeated runs
    latest = {}
    for log_path in log_paths:
        for line in log_path.read_text(errors="ignore").splitlines():
            m = PATTERN.search(line)
            if not m:
                continue
            raw = m.groupdict()
            task_loop = raw.get("task_loop")
            rec = {
                k: int(v)
                for k, v in raw.items()
                if k not in ("dtype", "task_loop") and v is not None
            }
            rec["dtype"] = m.groupdict().get("dtype") or "unknown"
            rec.setdefault("sched_protocol", 0)
            rec.setdefault("compute_submit", 0)
            rec.setdefault("compute_wait", 0)
            rec.setdefault("c_store", 0)
            rec.setdefault("tile_ready_wait", 0)
            rec.setdefault("txn_wait", 0)
            rec.setdefault("writeback_wait", 0)
            rec.setdefault("wait_2d_activate", 0)
            rec.setdefault("wait_2d_active_not_ready", 0)
            rec.setdefault("wait_non2d_txn", 0)
            rec.setdefault("wait_no_active_txn", 0)
            rec.setdefault("window_submit_active", 0)
            rec.setdefault("window_submit_prefetch", 0)
            rec.setdefault("window_activate", 0)
            rec.setdefault("window_advance_wait_prefetch", 0)
            rec.setdefault("group_wait", 0)
            rec.setdefault("poll_iters", 0)
            rec.setdefault("overlap_issue", 0)
            rec.setdefault("overlap_wait", 0)
            rec.setdefault("issue_block_q", 0)
            rec.setdefault("issue_write", 0)
            rec.setdefault("ov_issue_block_q", 0)
            rec.setdefault("ov_issue_write", 0)
            rec.setdefault("task_desc", 0)
            rec.setdefault("nloop", 0)
            rec.setdefault("submit_pack", 0)
            rec.setdefault("finish_publish", 0)
            rec.setdefault("c_buffer_reads", 0)
            rec.setdefault("c_buffer_writes", 0)
            rec.setdefault("c_buffer_read_wait", 0)
            rec.setdefault("descriptor_start_cycle", 0)
            rec.setdefault("start_cycle", 0)
            rec.setdefault("end_cycle", 0)
            if task_loop is not None and rec["task_desc"] == 0 and rec["nloop"] == 0:
                rec["nloop"] = int(task_loop)
            latest[rec["core"]] = rec
    return [latest[k] for k in sorted(latest.keys())]


def write_core_csv(records, path: Path):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for rec in records:
            w.writerow(rec)


def write_debug_summary_csv(records, path: Path):
    metric_fields = [
        "dma_issue",
        "dma_wait",
        "dma_total",
        "compute",
        "compute_submit",
        "compute_wait",
        "sched_protocol",
        "c_store",
        "tile_ready_wait",
        "txn_wait",
        "writeback_wait",
        "wait_2d_activate",
        "wait_2d_active_not_ready",
        "wait_non2d_txn",
        "wait_no_active_txn",
        "window_submit_active",
        "window_submit_prefetch",
        "window_activate",
        "window_advance_wait_prefetch",
        "group_wait",
        "poll_iters",
        "overlap_issue",
        "overlap_wait",
        "issue_block_q",
        "issue_write",
        "ov_issue_block_q",
        "ov_issue_write",
        "task_desc",
        "nloop",
        "submit_pack",
        "finish_publish",
        "c_buffer_reads",
        "c_buffer_writes",
        "c_buffer_read_wait",
        "total",
    ]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "median", "p95", "min", "max"])
        for name in metric_fields:
            vals = [r[name] for r in records]
            if not vals:
                w.writerow([name, 0, 0, 0, 0, 0])
                continue
            w.writerow(
                [
                    name,
                    int(mean(vals)),
                    int(median(vals)),
                    int(_p95(vals)),
                    min(vals),
                    max(vals),
                ]
            )

        totals = [r["total"] for r in records]
        if totals:
            mean_total = mean(totals)
            mean_compute = mean(r["compute"] for r in records)
            mean_compute_submit = mean(r["compute_submit"] for r in records)
            mean_compute_wait = mean(r["compute_wait"] for r in records)
            mean_dma = mean(r["dma_total"] for r in records)
            mean_sched_protocol = mean(r["sched_protocol"] for r in records)
            mean_group_wait = mean(r["group_wait"] for r in records)
            mean_nloop = mean(r["nloop"] for r in records)
            mean_dma_wait = mean(r["dma_wait"] for r in records)
            overlap_ratio_pct = 0.0
            if (mean_compute + mean_dma_wait) > 0:
                overlap_ratio_pct = (
                    100.0 * mean_compute / (mean_compute + mean_dma_wait)
                )
            for name, value in [
                ("compute_share_pct_runtime", 100.0 * mean_compute / mean_total),
                (
                    "compute_submit_share_pct_runtime",
                    100.0 * mean_compute_submit / mean_total,
                ),
                (
                    "compute_wait_share_pct_runtime",
                    100.0 * mean_compute_wait / mean_total,
                ),
                ("dma_share_pct_runtime", 100.0 * mean_dma / mean_total),
                (
                    "sched_protocol_share_pct_runtime",
                    100.0 * mean_sched_protocol / mean_total,
                ),
                ("group_wait_share_pct_runtime", 100.0 * mean_group_wait / mean_total),
                ("loop_other_share_pct_runtime", 100.0 * mean_nloop / mean_total),
                ("overlap_ratio_pct_runtime", overlap_ratio_pct),
            ]:
                w.writerow([name, f"{value:.4f}", "", "", "", ""])


def write_summary_csv(records, path: Path, hardware_context):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        if not records:
            return

        mean_total = mean(r["total"] for r in records)
        total_values = [r["total"] for r in records]
        mean_dma = mean(r["dma_total"] for r in records)
        mean_compute = mean(r["compute"] for r in records)
        mean_c_store = mean(r["c_store"] for r in records)
        mean_tile_ready_wait = mean(r["tile_ready_wait"] for r in records)
        mean_txn_wait = mean(r["txn_wait"] for r in records)
        mean_writeback_wait = mean(r["writeback_wait"] for r in records)
        mean_sched_protocol = mean(r["sched_protocol"] for r in records)
        mean_group_wait = mean(r["group_wait"] for r in records)

        m = hardware_context["gemm_m"]
        n = hardware_context["gemm_n"]
        k = hardware_context["gemm_k"]
        active_worker_cores = hardware_context["active_worker_cores"]
        num_arrays = hardware_context["num_arrays"]
        array_num_cu = hardware_context["array_num_cu"]
        mac_per_cu_per_cycle = hardware_context["mac_per_cu_per_cycle"]
        stats_path = hardware_context["stats_path"]

        effective_ops = float(2 * m * n * k)
        avg_throughput = effective_ops / mean_total if mean_total > 0 else 0.0
        peak_throughput = float(
            active_worker_cores * num_arrays * array_num_cu * mac_per_cu_per_cycle * 2
        )
        array_utilization_pct = (
            100.0 * avg_throughput / peak_throughput if peak_throughput > 0 else 0.0
        )
        wcp_valid_windows = [
            (r["start_cycle"], r["end_cycle"])
            for r in records
            if r.get("start_cycle", 0) > 0 and r.get("end_cycle", 0) >= r.get("start_cycle", 0)
        ]
        descriptor_start_cycles = [
            r["descriptor_start_cycle"]
            for r in records
            if r.get("descriptor_start_cycle", 0) > 0
        ]
        wcp_start_cycles = [s for s, _ in wcp_valid_windows]
        gemm_descriptor_start_cycle = min(descriptor_start_cycles, default=0)
        gemm_descriptor_start_skew_cycles = (
            max(descriptor_start_cycles) - min(descriptor_start_cycles)
            if descriptor_start_cycles
            else 0
        )
        gemm_wcp_start_min_cycle = min(wcp_start_cycles, default=0)
        gemm_wcp_start_max_cycle = max(wcp_start_cycles, default=0)
        gemm_wcp_start_skew_cycles = (
            gemm_wcp_start_max_cycle - gemm_wcp_start_min_cycle
            if wcp_start_cycles
            else 0
        )
        gemm_system_start_cycle = gemm_descriptor_start_cycle or gemm_wcp_start_min_cycle
        gemm_system_end_cycle = max((e for _, e in wcp_valid_windows), default=0)
        gemm_system_latency_cycles = (
            gemm_system_end_cycle - gemm_system_start_cycle + 1
            if gemm_system_start_cycle > 0 and gemm_system_end_cycle >= gemm_system_start_cycle
            else 0
        )
        system_avg_throughput = (
            effective_ops / gemm_system_latency_cycles if gemm_system_latency_cycles > 0 else 0.0
        )
        system_array_utilization_pct = (
            100.0 * system_avg_throughput / peak_throughput if peak_throughput > 0 else 0.0
        )

        # Mutually exclusive runtime breakdown for the current WCP pipeline:
        # - compute_active_time: array compute in flight
        # - prefetch_wait_time: waiting for next tile pair to become ready
        # - writeback_wait_time: final C-tile writeback completion wait
        # - control_other_time: residual runtime not covered above
        compute_active_time = mean_compute
        prefetch_wait_time = mean_tile_ready_wait + mean_txn_wait
        writeback_wait_time = mean_writeback_wait
        control_other_time = max(
            mean_total - compute_active_time - prefetch_wait_time - writeback_wait_time,
            0.0,
        )

        rows = [
            ("total_cycles", mean_total),
            ("worker_avg_total_cycles", mean_total),
            ("worker_p95_total_cycles", _p95(total_values)),
            ("worker_max_total_cycles", max(total_values) if total_values else 0),
            ("gemm_descriptor_start_cycle", gemm_descriptor_start_cycle),
            ("gemm_descriptor_start_skew_cycles", gemm_descriptor_start_skew_cycles),
            ("gemm_wcp_start_min_cycle", gemm_wcp_start_min_cycle),
            ("gemm_wcp_start_max_cycle", gemm_wcp_start_max_cycle),
            ("gemm_wcp_start_skew_cycles", gemm_wcp_start_skew_cycles),
            ("gemm_system_start_cycle", gemm_system_start_cycle),
            ("gemm_system_end_cycle", gemm_system_end_cycle),
            ("gemm_system_latency_cycles", gemm_system_latency_cycles),
            ("avg_throughput_ops_per_cycle", avg_throughput),
            ("worker_avg_throughput_ops_per_cycle", avg_throughput),
            ("system_avg_throughput_ops_per_cycle", system_avg_throughput),
            ("peak_throughput_ops_per_cycle", peak_throughput),
            ("array_utilization_pct", array_utilization_pct),
            ("worker_avg_array_efficiency_pct", array_utilization_pct),
            ("system_array_utilization_pct", system_array_utilization_pct),
            ("compute_active_time", compute_active_time),
            ("prefetch_wait_time", prefetch_wait_time),
            ("writeback_wait_time", writeback_wait_time),
            ("control_other_time", control_other_time),
            (
                "compute_active_time_share_pct",
                100.0 * compute_active_time / mean_total
                if mean_total > 0
                else 0.0,
            ),
            (
                "prefetch_wait_time_share_pct",
                100.0 * prefetch_wait_time / mean_total if mean_total > 0 else 0.0,
            ),
            (
                "writeback_wait_time_share_pct",
                100.0 * writeback_wait_time / mean_total if mean_total > 0 else 0.0,
            ),
            (
                "control_other_time_share_pct",
                100.0 * control_other_time / mean_total if mean_total > 0 else 0.0,
            ),
            ("debug_sched_protocol_mean", mean_sched_protocol),
            ("debug_group_wait_mean", mean_group_wait),
        ]
        for metric, value in rows:
            w.writerow([metric, f"{value:.6f}"])


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-core execution metrics from logs into CSV files."
    )
    parser.add_argument(
        "--log",
        default="",
        help="Optional path to main simulation log (e.g., test.log)",
    )
    parser.add_argument(
        "--log-dir",
        default=".",
        help="Directory to scan for sharded logs (default: current dir)",
    )
    parser.add_argument(
        "--stdout-glob",
        default="stdout-*",
        help="Glob for sharded stdout logs (default: stdout-*)",
    )
    parser.add_argument("--out", default="", help="Optional output per-core CSV path")
    parser.add_argument(
        "--summary", default="execution_summary.csv", help="Output summary CSV path"
    )
    parser.add_argument("--debug-summary", default="execution_debug_summary.csv")
    parser.add_argument("--stats-file", default="", help="stats_selfcom.txt path")
    parser.add_argument("--gemm-m", type=int, default=0)
    parser.add_argument("--gemm-n", type=int, default=0)
    parser.add_argument("--gemm-k", type=int, default=0)
    parser.add_argument("--active-worker-cores", type=int, default=0)
    parser.add_argument("--num-arrays", type=int, default=0)
    parser.add_argument("--array-num-cu", type=int, default=0)
    parser.add_argument("--mac-per-cu-per-cycle", type=int, default=0)
    args = parser.parse_args()

    log_path = Path(args.log) if args.log else None
    log_dir = Path(args.log_dir)
    out_path = Path(args.out) if args.out else None
    sum_path = Path(args.summary)
    debug_sum_path = Path(args.debug_summary)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    sum_path.parent.mkdir(parents=True, exist_ok=True)
    debug_sum_path.parent.mkdir(parents=True, exist_ok=True)

    input_logs = resolve_inputs(log_path, log_dir, args.stdout_glob)
    records = parse_logs(input_logs)
    if not records:
        if out_path is not None:
            write_core_csv([], out_path)
        write_debug_summary_csv([], debug_sum_path)
        write_summary_csv(
            [],
            sum_path,
            {
                "gemm_m": args.gemm_m,
                "gemm_n": args.gemm_n,
                "gemm_k": args.gemm_k,
                "active_worker_cores": args.active_worker_cores,
                "num_arrays": args.num_arrays,
                "array_num_cu": args.array_num_cu,
                "mac_per_cu_per_cycle": args.mac_per_cu_per_cycle,
                "stats_path": Path(args.stats_file) if args.stats_file else Path(""),
            },
        )
        print(
            "[WARN] no LATENCY(cycles) records found in log; wrote empty CSV templates"
        )
        print("[INFO] searched logs:")
        for p in input_logs:
            print(f"  - {p}")
        if out_path is not None:
            print(f"[OK] wrote {out_path}")
        print(f"[OK] wrote {sum_path}")
        return

    if out_path is not None:
        write_core_csv(records, out_path)
    write_debug_summary_csv(records, debug_sum_path)
    write_summary_csv(
        records,
        sum_path,
        {
            "gemm_m": args.gemm_m,
            "gemm_n": args.gemm_n,
            "gemm_k": args.gemm_k,
            "active_worker_cores": args.active_worker_cores,
            "num_arrays": args.num_arrays,
            "array_num_cu": args.array_num_cu,
            "mac_per_cu_per_cycle": args.mac_per_cu_per_cycle,
            "stats_path": Path(args.stats_file) if args.stats_file else Path(""),
        },
    )

    print("[OK] parsed logs:")
    for p in input_logs:
        print(f"  - {p}")
    if out_path is not None:
        print(f"[OK] wrote {out_path}")
    print(f"[OK] wrote {sum_path}")
    print(f"[OK] wrote {debug_sum_path}")
    print(f"[OK] cores: {len(records)}")


if __name__ == "__main__":
    main()
