#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


PRESSURE_RE = re.compile(
    r"\[RequestScheduler\]\[core=(?P<core>\d+)\] SCHED_PRESSURE: "
    r"ticks=(?P<ticks>\d+) pending_nonempty_ticks=(?P<pending_nonempty_ticks>\d+) "
    r"no_issue_ticks=(?P<no_issue_ticks>\d+) issued_requests=(?P<issued_requests>\d+) "
    r"pending_q\(n=(?P<pending_q_n>\d+) mean=(?P<pending_q_mean>[0-9.]+) "
    r"p50=(?P<pending_q_p50>\d+) p95=(?P<pending_q_p95>\d+) max=(?P<pending_q_max>\d+)\)"
    r" worker_used_max\(n=(?P<worker_used_n>\d+) mean=(?P<worker_used_mean>[0-9.]+) "
    r"p50=(?P<worker_used_p50>\d+) p95=(?P<worker_used_p95>\d+) max=(?P<worker_used_max>\d+) cap=(?P<worker_credit_cap>\d+)\)"
    r" node_used_max\(n=(?P<node_used_n>\d+) mean=(?P<node_used_mean>[0-9.]+) "
    r"p50=(?P<node_used_p50>\d+) p95=(?P<node_used_p95>\d+) max=(?P<node_used_max>\d+) cap=(?P<node_credit_cap>\d+)"
    r"(?: chunk_bytes=(?P<node_credit_chunk_bytes>\d+))?"
    r"(?: panel_chunk_bytes=(?P<panel_chunk_bytes>\d+))?\)"
    r" blocked\(worker_credit=(?P<blocked_worker_credit>\d+) node_credit=(?P<blocked_node_credit>\d+)"
    r"(?: issue_pace=(?P<blocked_issue_pace>\d+))?"
    r"(?: network_send=(?P<blocked_network_send>\d+))?"
    r"\)"
    r"(?: priority_pair\(attempts=(?P<priority_pair_attempts>\d+) issued=(?P<priority_pair_issued>\d+)\))?"
)


def resolve_inputs(log_path: Path | None, log_dir: Path, stdout_glob: str):
    inputs = []
    if log_dir.exists() and log_dir.is_dir():
        inputs.extend(sorted(p for p in log_dir.glob(stdout_glob) if p.is_file()))
    if log_path is not None and log_path.exists() and log_path.is_file():
        inputs.append(log_path)
    return inputs


def parse_records(paths):
    latest = {}
    for path in paths:
        for line in path.read_text(errors="ignore").splitlines():
            m = PRESSURE_RE.search(line)
            if not m:
                continue
            rec = m.groupdict()
            if rec.get("node_credit_chunk_bytes") is None:
                rec["node_credit_chunk_bytes"] = "0"
            if rec.get("panel_chunk_bytes") is None:
                rec["panel_chunk_bytes"] = "0"
            if rec.get("blocked_issue_pace") is None:
                rec["blocked_issue_pace"] = "0"
            if rec.get("blocked_network_send") is None:
                rec["blocked_network_send"] = "0"
            if rec.get("priority_pair_attempts") is None:
                rec["priority_pair_attempts"] = "0"
            if rec.get("priority_pair_issued") is None:
                rec["priority_pair_issued"] = "0"
            core = int(rec["core"])
            latest[core] = rec
    return [latest[k] for k in sorted(latest)]


def write_summary(records, path: Path):
    metrics = {}
    if records:
        numeric_keys = [k for k in records[0] if k != "core"]
        for key in numeric_keys:
            vals = [float(r[key]) for r in records]
            metrics[f"sched_{key}_mean"] = sum(vals) / len(vals)
            metrics[f"sched_{key}_max"] = max(vals)
        ticks = sum(float(r["ticks"]) for r in records)
        pending_nonempty = sum(float(r["pending_nonempty_ticks"]) for r in records)
        no_issue = sum(float(r["no_issue_ticks"]) for r in records)
        issued = sum(float(r["issued_requests"]) for r in records)
        metrics["sched_pending_nonempty_tick_share_pct"] = 100.0 * pending_nonempty / ticks if ticks else 0.0
        metrics["sched_no_issue_when_pending_share_pct"] = 100.0 * no_issue / pending_nonempty if pending_nonempty else 0.0
        metrics["sched_issued_requests_per_tick"] = issued / ticks if ticks else 0.0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for key in sorted(metrics):
            w.writerow([key, f"{metrics[key]:.6f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path)
    ap.add_argument("--log-dir", type=Path, default=Path("."))
    ap.add_argument("--stdout-glob", default="*.out")
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--table", type=Path)
    args = ap.parse_args()

    records = parse_records(resolve_inputs(args.log, args.log_dir, args.stdout_glob))
    write_summary(records, args.summary)
    if args.table:
        args.table.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(records[0].keys()) if records else ["core"]
        with args.table.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for rec in records:
                w.writerow(rec)


if __name__ == "__main__":
    main()
