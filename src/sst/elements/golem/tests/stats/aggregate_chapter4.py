#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path


def metrics_csv(path):
    out = {}
    if not path.exists():
        return out
    with path.open(newline="") as stream:
        for row in csv.reader(stream):
            if len(row) >= 2 and row[0] != "metric":
                out[row[0]] = row[1]
    return out


def number(row, key, default=0.0):
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = list(csv.DictReader((args.root / "plan.tsv").open(newline=""), delimiter="\t"))
    records = []
    for item in plan:
        case_root = args.root / "cases" / f"{int(item['order']):02d}_{item['case']}"
        summaries = list(csv.DictReader((case_root / "run_summary.csv").open(newline=""))) if (case_root / "run_summary.csv").exists() else []
        run = summaries[-1] if summaries else {}
        exec_summary = metrics_csv(case_root / "artifacts/stats/overlap0" / f"ch4_{args.root.name}_{item['order']}_{item['case']}" / "execution_summary.csv")
        # The pipeline's automatic run id directory is not stable across wrapper versions;
        # prefer the run summary fields and discover the actual stats directory when needed.
        stats_dirs = list((case_root / "artifacts/stats").glob("overlap*/ch4_*"))
        stats = stats_dirs[-1] if stats_dirs else None
        if stats:
            exec_summary = metrics_csv(stats / "execution_summary.csv")
            memory = metrics_csv(stats / "memory_summary.csv")
            queue = metrics_csv(stats / "memory_queue_summary.csv")
            credit = metrics_csv(stats / "credit_owner_summary.csv")
            sched = metrics_csv(stats / "sched_pressure_summary.csv")
        else:
            memory = queue = credit = sched = {}
        m = number(run, "gemm_m", 1024); n = number(run, "gemm_n", 1024); k = number(run, "gemm_k", 1024)
        bm = number(run, "block_m", 64); bn = number(run, "block_n", 64); bk = number(run, "block_k", 64)
        rn = max(1, int(item["reuse_n"])); rm = max(1, int(item["reuse_m"]))
        mt = math.ceil(m / bm); nt = math.ceil(n / bn); kt = math.ceil(k / bk)
        mg = math.ceil(mt / rm); ng = math.ceil(nt / rn)
        elem = 4
        a_bytes = mg * ng * kt * rm * bm * bk * elem
        b_bytes = mg * ng * kt * rn * bn * bk * elem
        useful = a_bytes + b_bytes
        ops = 2.0 * m * n * k
        cycles = number(run, "gemm_system_latency_cycles", number(exec_summary, "gemm_system_latency_cycles"))
        roof_bpc = 4 * 16 * 64 / 2
        rec = dict(item)
        rec.update({
            "case_root": str(case_root),
            "status": "PASS" if run else "MISSING",
            "gemm_m": int(m), "gemm_n": int(n), "gemm_k": int(k),
            "block_m": int(bm), "block_n": int(bn), "block_k": int(bk),
            "exec_total_cycles": run.get("exec_total_cycles", ""),
            "gemm_system_latency_cycles": cycles,
            "exec_system_array_utilization_pct": run.get("exec_system_array_utilization_pct", ""),
            "hbm_useful_read_bytes": int(useful),
            "a_read_bytes": int(a_bytes), "b_read_bytes": int(b_bytes),
            "arithmetic_intensity_ops_per_byte": ops / useful if useful else 0,
            "hbm_roofline_bytes_per_cycle": roof_bpc,
            "hbm_read_pressure_pct": 100 * useful / (cycles * roof_bpc) if cycles else 0,
            "hbm_utilization_pct": run.get("hbm_utilization_pct", ""),
            "hbm_data_node_count": run.get("hbm_data_node_count", memory.get("data_node_count", "")),
            "hbm_data_channel_count": run.get("hbm_data_channel_count", memory.get("channel_count", "")),
            "hbm_aggregate_bandwidth_gbps": memory.get("hbm_aggregate_bandwidth_gbps", run.get("hbm_aggregate_bandwidth_gbps", "")),
            "hbm_backend_service_window_utilization_pct": run.get("hbm_backend_service_window_utilization_pct", ""),
            "hbm_backend_active_utilization_pct": run.get("hbm_backend_active_utilization_pct", ""),
            "hbm_tccdl_roofline_bytes_per_cycle": run.get("hbm_tccdl_roofline_bytes_per_cycle", roof_bpc),
            "memory_queue_delay_avg_cycles": queue.get("memory_queue_delay_avg_cycles", run.get("memory_queue_delay_avg_cycles", "")),
            "memory_queue_delay_p99_cycles": queue.get("memory_queue_delay_p99_cycles", run.get("memory_queue_delay_p99_cycles", "")),
            "memory_backend_read_latency_avg_cycles": queue.get("memory_backend_read_latency_avg_cycles", run.get("memory_backend_read_latency_avg_cycles", "")),
            "memory_backend_read_latency_p99_cycles": queue.get("memory_backend_read_latency_p99_cycles", run.get("memory_backend_read_latency_p99_cycles", "")),
            "credit_owner_total_blocked_requests": credit.get("credit_owner_total_blocked_requests", ""),
            "credit_owner_max_used": credit.get("credit_owner_max_used", ""),
            "credit_owner_max_queue": credit.get("credit_owner_max_queue", ""),
            "sched_blocked_issue_pace_max": sched.get("sched_blocked_issue_pace_max", ""),
            "sched_blocked_worker_credit_max": sched.get("sched_blocked_worker_credit_max", ""),
        })
        records.append(rec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0]) if records else []
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(records)
    print(f"[OK] wrote {args.output} ({len(records)} cases)")


if __name__ == "__main__":
    main()
