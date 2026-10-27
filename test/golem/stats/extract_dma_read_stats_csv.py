#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path
from statistics import mean, median

PATTERN = re.compile(
    r"GlobalMemory core=(?P<core>\d+) DMA READ stats: "
    r"immediate_send=(?P<immediate_send>\d+) "
    r"queued_send=(?P<queued_send>\d+) "
    r"flushed_send=(?P<flushed_send>\d+) "
    r"read_issue_count=(?P<read_issue_count>\d+) "
    r"write_issue_count=(?P<write_issue_count>\d+) "
    r"read_bytes_total=(?P<read_bytes_total>\d+) "
    r"write_bytes_total=(?P<write_bytes_total>\d+) "
    r"timeout_retry=(?P<timeout_retry>\d+) "
    r"timeout_exhausted=(?P<timeout_exhausted>\d+) "
    r"write_timeout_retry=(?P<write_timeout_retry>\d+) "
    r"completion=(?P<completion>\d+) "
    r"write_completion=(?P<write_completion>\d+) "
    r"completion_no_pending=(?P<completion_no_pending>\d+) "
    r"wait_count=(?P<wait_count>\d+) "
    r"avg_rtt_ticks=(?P<avg_rtt_ticks>\d+) "
    r"max_rtt_ticks=(?P<max_rtt_ticks>\d+) "
    r"avg_rtt_cycles=(?P<avg_rtt_cycles>\d+) "
    r"max_rtt_cycles=(?P<max_rtt_cycles>\d+) "
    r"(?:avg_e2e_rtt_ticks=(?P<avg_e2e_rtt_ticks>\d+) "
    r"max_e2e_rtt_ticks=(?P<max_e2e_rtt_ticks>\d+) "
    r"avg_e2e_rtt_cycles=(?P<avg_e2e_rtt_cycles>\d+) "
    r"max_e2e_rtt_cycles=(?P<max_e2e_rtt_cycles>\d+) )?"
    r"(?:strict_avg_rtt_cycles=(?P<strict_avg_rtt_cycles>\d+) "
    r"strict_max_rtt_cycles=(?P<strict_max_rtt_cycles>\d+) "
    r"strict_avg_e2e_rtt_cycles=(?P<strict_avg_e2e_rtt_cycles>\d+) "
    r"strict_max_e2e_rtt_cycles=(?P<strict_max_e2e_rtt_cycles>\d+) "
    r"request_avg_submit_ready_cycles=(?P<request_avg_submit_ready_cycles>\d+) "
    r"request_max_submit_ready_cycles=(?P<request_max_submit_ready_cycles>\d+) )?"
    r"(?:write_rtt_samples=(?P<write_rtt_samples>\d+) "
    r"write_rtt_cycles_sum=(?P<write_rtt_cycles_sum>\d+) "
    r"write_avg_rtt_cycles=(?P<write_avg_rtt_cycles>\d+) "
    r"write_max_rtt_cycles=(?P<write_max_rtt_cycles>\d+) "
    r"write_first_issue_cycle=(?P<write_first_issue_cycle>\d+) "
    r"write_last_issue_cycle=(?P<write_last_issue_cycle>\d+) "
    r"write_last_complete_cycle=(?P<write_last_complete_cycle>\d+) )?"
    r"send_retry_q_max=(?P<send_retry_q_max>\d+)"
)

PATTERN_LEGACY = re.compile(
    r"GlobalMemory core=(?P<core>\d+) DMA READ stats: "
    r"immediate_send=(?P<immediate_send>\d+) "
    r"queued_send=(?P<queued_send>\d+) "
    r"flushed_send=(?P<flushed_send>\d+) "
    r"timeout_retry=(?P<timeout_retry>\d+) "
    r"timeout_exhausted=(?P<timeout_exhausted>\d+) "
    r"completion=(?P<completion>\d+) "
    r"completion_no_pending=(?P<completion_no_pending>\d+) "
    r"avg_rtt_ticks=(?P<avg_rtt_ticks>\d+) "
    r"max_rtt_ticks=(?P<max_rtt_ticks>\d+) "
    r"send_retry_q_max=(?P<send_retry_q_max>\d+)"
)

STRICT_RTT_PATTERN = re.compile(
    r"\[RequestScheduler\]\[core=(?P<core>\d+)\] STRICT_RTT_SUMMARY\(cycles\): "
    r"strict_rtt_samples=(?P<strict_rtt_samples>\d+) "
    r"strict_rtt_cycles_sum=(?P<strict_rtt_cycles_sum>\d+) "
    r"strict_avg_rtt_cycles=(?P<strict_avg_rtt_cycles>\d+) "
    r"strict_max_rtt_cycles=(?P<strict_max_rtt_cycles>\d+) "
    r"strict_e2e_rtt_samples=(?P<strict_e2e_rtt_samples>\d+) "
    r"strict_e2e_rtt_cycles_sum=(?P<strict_e2e_rtt_cycles_sum>\d+) "
    r"strict_avg_e2e_rtt_cycles=(?P<strict_avg_e2e_rtt_cycles>\d+) "
    r"strict_max_e2e_rtt_cycles=(?P<strict_max_e2e_rtt_cycles>\d+)"
)

FIELDS = [
    "core",
    "immediate_send",
    "queued_send",
    "flushed_send",
    "read_issue_count",
    "write_issue_count",
    "read_bytes_total",
    "write_bytes_total",
    "timeout_retry",
    "timeout_exhausted",
    "write_timeout_retry",
    "completion",
    "write_completion",
    "completion_no_pending",
    "wait_count",
    "avg_rtt_ticks",
    "max_rtt_ticks",
    "avg_rtt_cycles",
    "max_rtt_cycles",
    "avg_e2e_rtt_ticks",
    "max_e2e_rtt_ticks",
    "avg_e2e_rtt_cycles",
    "max_e2e_rtt_cycles",
    "strict_rtt_samples",
    "strict_rtt_cycles_sum",
    "strict_avg_rtt_cycles",
    "strict_max_rtt_cycles",
    "strict_e2e_rtt_samples",
    "strict_e2e_rtt_cycles_sum",
    "strict_avg_e2e_rtt_cycles",
    "strict_max_e2e_rtt_cycles",
    "request_avg_submit_ready_cycles",
    "request_max_submit_ready_cycles",
    "write_rtt_samples",
    "write_rtt_cycles_sum",
    "write_avg_rtt_cycles",
    "write_max_rtt_cycles",
    "write_first_issue_cycle",
    "write_last_issue_cycle",
    "write_last_complete_cycle",
    "send_retry_q_max",
]

STRICT_ACTIVE_FIELDS = {
    "strict_rtt_samples": "strict_rtt_samples",
    "strict_rtt_cycles_sum": "strict_rtt_samples",
    "strict_avg_rtt_cycles": "strict_rtt_samples",
    "strict_max_rtt_cycles": "strict_rtt_samples",
    "strict_e2e_rtt_samples": "strict_e2e_rtt_samples",
    "strict_e2e_rtt_cycles_sum": "strict_e2e_rtt_samples",
    "strict_avg_e2e_rtt_cycles": "strict_e2e_rtt_samples",
    "strict_max_e2e_rtt_cycles": "strict_e2e_rtt_samples",
    "write_avg_rtt_cycles": "write_rtt_samples",
    "write_max_rtt_cycles": "write_rtt_samples",
    "write_first_issue_cycle": "write_rtt_samples",
    "write_last_issue_cycle": "write_rtt_samples",
    "write_last_complete_cycle": "write_rtt_samples",
}

STRICT_WEIGHTED_AVG_FIELDS = {
    "strict_avg_rtt_cycles": ("strict_rtt_cycles_sum", "strict_rtt_samples"),
    "strict_avg_e2e_rtt_cycles": (
        "strict_e2e_rtt_cycles_sum",
        "strict_e2e_rtt_samples",
    ),
    "write_avg_rtt_cycles": ("write_rtt_cycles_sum", "write_rtt_samples"),
}


def _p95(values):
    if not values:
        return 0
    vals = sorted(values)
    idx = int(round(0.95 * (len(vals) - 1)))
    return vals[idx]


def resolve_inputs(log_path: Path | None, log_dir: Path, stdout_glob: str):
    inputs = []
    if log_path is not None and log_path.exists() and log_path.is_file():
        inputs.append(log_path)
    if log_dir.exists() and log_dir.is_dir():
        inputs.extend(sorted(p for p in log_dir.glob(stdout_glob) if p.is_file()))
    if not inputs and log_path is not None:
        raise SystemExit(
            f"[ERROR] no input logs found (log={log_path}, dir={log_dir}, glob={stdout_glob})"
        )
    return inputs


def empty_record(core):
    rec = {name: 0 for name in FIELDS}
    rec["core"] = core
    return rec


def fill_strict_defaults(rec):
    for name in (
        "strict_rtt_samples",
        "strict_rtt_cycles_sum",
        "strict_e2e_rtt_samples",
        "strict_e2e_rtt_cycles_sum",
    ):
        rec.setdefault(name, 0)

    if rec["strict_rtt_samples"] == 0 and (
        rec.get("strict_avg_rtt_cycles", 0) or rec.get("strict_max_rtt_cycles", 0)
    ):
        rec["strict_rtt_samples"] = max(1, rec.get("read_issue_count", 0))
        rec["strict_rtt_cycles_sum"] = (
            rec.get("strict_avg_rtt_cycles", 0) * rec["strict_rtt_samples"]
        )
    if rec["strict_e2e_rtt_samples"] == 0 and (
        rec.get("strict_avg_e2e_rtt_cycles", 0)
        or rec.get("strict_max_e2e_rtt_cycles", 0)
    ):
        rec["strict_e2e_rtt_samples"] = max(1, rec.get("read_issue_count", 0))
        rec["strict_e2e_rtt_cycles_sum"] = (
            rec.get("strict_avg_e2e_rtt_cycles", 0)
            * rec["strict_e2e_rtt_samples"]
        )


def parse_logs(log_paths):
    latest = {}
    strict_latest = {}
    for log_path in log_paths:
        for line in log_path.read_text(errors="ignore").splitlines():
            strict = STRICT_RTT_PATTERN.search(line)
            if strict:
                rec = {k: int(v) for k, v in strict.groupdict().items()}
                strict_latest[rec["core"]] = rec
                continue

            m = PATTERN.search(line)
            if m:
                rec = {
                    k: int(v) if v is not None else 0 for k, v in m.groupdict().items()
                }
            else:
                old = PATTERN_LEGACY.search(line)
                if not old:
                    continue
                rec = {k: int(v) for k, v in old.groupdict().items()}
                rec["read_issue_count"] = 0
                rec["write_issue_count"] = 0
                rec["read_bytes_total"] = 0
                rec["write_bytes_total"] = 0
                rec["write_timeout_retry"] = 0
                rec["write_completion"] = 0
                rec["wait_count"] = 0
                rec["avg_rtt_cycles"] = 0
                rec["max_rtt_cycles"] = 0
                rec["avg_e2e_rtt_ticks"] = 0
                rec["max_e2e_rtt_ticks"] = 0
                rec["avg_e2e_rtt_cycles"] = 0
                rec["max_e2e_rtt_cycles"] = 0
                rec["strict_avg_rtt_cycles"] = 0
                rec["strict_max_rtt_cycles"] = 0
                rec["strict_avg_e2e_rtt_cycles"] = 0
                rec["strict_max_e2e_rtt_cycles"] = 0
                rec["request_avg_submit_ready_cycles"] = 0
                rec["request_max_submit_ready_cycles"] = 0
                rec["write_rtt_samples"] = 0
                rec["write_rtt_cycles_sum"] = 0
                rec["write_avg_rtt_cycles"] = 0
                rec["write_max_rtt_cycles"] = 0
                rec["write_first_issue_cycle"] = 0
                rec["write_last_issue_cycle"] = 0
                rec["write_last_complete_cycle"] = 0
            fill_strict_defaults(rec)
            latest[rec["core"]] = rec
    for core, strict in strict_latest.items():
        has_strict_samples = (
            strict["strict_rtt_samples"] > 0 or strict["strict_e2e_rtt_samples"] > 0
        )
        if core not in latest:
            if not has_strict_samples:
                continue
            latest[core] = empty_record(core)
        rec = latest[core]
        if strict["strict_rtt_samples"] > 0:
            for name in (
                "strict_rtt_samples",
                "strict_rtt_cycles_sum",
                "strict_avg_rtt_cycles",
                "strict_max_rtt_cycles",
            ):
                rec[name] = strict[name]
        if strict["strict_e2e_rtt_samples"] > 0:
            for name in (
                "strict_e2e_rtt_samples",
                "strict_e2e_rtt_cycles_sum",
                "strict_avg_e2e_rtt_cycles",
                "strict_max_e2e_rtt_cycles",
            ):
                rec[name] = strict[name]
    return [latest[k] for k in sorted(latest.keys())]


def write_core_csv(records, path: Path):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for rec in records:
            w.writerow(rec)


def write_summary_csv(records, path: Path):
    metric_fields = FIELDS[1:]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "median", "p95", "min", "max", "sum"])
        for name in metric_fields:
            active_field = STRICT_ACTIVE_FIELDS.get(name)
            active_records = records
            if active_field is not None:
                active_records = [r for r in records if r.get(active_field, 0) > 0]
            vals = [r[name] for r in active_records]
            if not vals:
                w.writerow([name, 0, 0, 0, 0, 0, 0])
                continue
            mean_value = int(mean(vals))
            if name in STRICT_WEIGHTED_AVG_FIELDS:
                sum_field, sample_field = STRICT_WEIGHTED_AVG_FIELDS[name]
                total_samples = sum(r.get(sample_field, 0) for r in active_records)
                total_cycles = sum(r.get(sum_field, 0) for r in active_records)
                mean_value = int(total_cycles / total_samples) if total_samples else 0
            w.writerow(
                [
                    name,
                    mean_value,
                    int(median(vals)),
                    int(_p95(vals)),
                    min(vals),
                    max(vals),
                    sum(vals),
                ]
            )


def main():
    parser = argparse.ArgumentParser(
        description="Extract GlobalMemory DMA READ stats into CSV files."
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
        "--summary",
        default="dma_read_stats_summary.csv",
        help="Output summary CSV path",
    )
    args = parser.parse_args()

    log_path = Path(args.log) if args.log else None
    log_dir = Path(args.log_dir)
    out_path = Path(args.out) if args.out else None
    sum_path = Path(args.summary)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    sum_path.parent.mkdir(parents=True, exist_ok=True)

    input_logs = resolve_inputs(log_path, log_dir, args.stdout_glob)
    records = parse_logs(input_logs)
    if not records:
        if out_path is not None:
            write_core_csv([], out_path)
        write_summary_csv([], sum_path)
        print(
            "[WARN] no DMA READ stats records found in log; wrote empty CSV templates"
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
    write_summary_csv(records, sum_path)

    print("[OK] parsed logs:")
    for p in input_logs:
        print(f"  - {p}")
    if out_path is not None:
        print(f"[OK] wrote {out_path}")
    print(f"[OK] wrote {sum_path}")
    print(f"[OK] cores: {len(records)}")


if __name__ == "__main__":
    main()
