#!/usr/bin/env python3
import argparse
import csv
import math
import re
from pathlib import Path


def _to_float(v: str) -> float:
    if v is None:
        return math.nan
    s = str(v).strip()
    if s == "":
        return math.nan
    try:
        return float(s)
    except ValueError:
        return math.nan


def _extract_run_id(log_file: str) -> str:
    m = re.search(r"(run_\d{8}_\d{6}_\d+)", log_file or "")
    return m.group(1) if m else ""


def _dominates(a, b, metrics):
    """All metrics <= and at least one < (minimization)."""
    any_strict = False
    for metric in metrics:
        av = a[metric]
        bv = b[metric]
        if math.isnan(av) or math.isnan(bv):
            return False
        if av > bv:
            return False
        if av < bv:
            any_strict = True
    return any_strict


def _pareto_front(rows, metrics):
    front = []
    for i, row_i in enumerate(rows):
        dominated = False
        for j, row_j in enumerate(rows):
            if i == j:
                continue
            if _dominates(row_j, row_i, metrics):
                dominated = True
                break
        if not dominated:
            front.append(row_i)
    return front


def _normalized_score(rows, metrics, weights):
    mins = {m: math.inf for m in metrics}
    maxs = {m: -math.inf for m in metrics}
    for row in rows:
        for m in metrics:
            v = row[m]
            if math.isnan(v):
                continue
            mins[m] = min(mins[m], v)
            maxs[m] = max(maxs[m], v)

    for row in rows:
        score = 0.0
        for m in metrics:
            v = row[m]
            w = weights[m]
            if math.isnan(v) or mins[m] is math.inf or maxs[m] == -math.inf:
                score += w
                continue
            span = maxs[m] - mins[m]
            norm = 0.0 if span <= 1e-12 else (v - mins[m]) / span
            score += w * norm
        row["weighted_score"] = score


def main():
    p = argparse.ArgumentParser(description="Select Pareto-optimal runs from run_summary.csv")
    p.add_argument("--input", required=True, help="Path to run_summary.csv")
    p.add_argument("--output", required=True, help="Output CSV for Pareto front")
    p.add_argument("--topk-output", default="", help="Optional output CSV for top-k by weighted score")
    p.add_argument("--topk", type=int, default=10, help="Top-k count for weighted ranking")
    p.add_argument("--overlap", default="", help="Filter overlap, e.g. overlap0 or overlap1")
    p.add_argument("--require-fields", default="exec_total_mean,memory_queue_delay_p99_cycles,dma_timeout_retry_sum")
    p.add_argument("--weights", default="exec_total_mean=0.5,memory_queue_delay_p99_cycles=0.3,dma_timeout_retry_sum=0.2")
    args = p.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    topk_path = Path(args.topk_output) if args.topk_output else None

    required_metrics = [s.strip() for s in args.require_fields.split(",") if s.strip()]
    weights = {}
    for item in args.weights.split(","):
        item = item.strip()
        if not item:
            continue
        k, v = item.split("=", 1)
        weights[k.strip()] = float(v.strip())
    for m in required_metrics:
        weights.setdefault(m, 1.0 / max(1, len(required_metrics)))

    with input_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    candidates = []
    for r in rows:
        if args.overlap and r.get("overlap", "") != args.overlap:
            continue
        entry = dict(r)
        entry["run_id"] = _extract_run_id(r.get("log_file", ""))
        ok = True
        for m in required_metrics:
            entry[m] = _to_float(r.get(m, ""))
            if math.isnan(entry[m]):
                ok = False
        if ok:
            candidates.append(entry)

    if not candidates:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as f:
            csv.writer(f).writerow(["message"])
            csv.writer(f).writerow(["no valid candidate rows"])
        print(f"[WARN] no valid candidate rows in {input_path}")
        return

    front = _pareto_front(candidates, required_metrics)
    _normalized_score(front, required_metrics, weights)
    front.sort(key=lambda x: x["weighted_score"])

    fields = [
        "run_id",
        "timestamp",
        "overlap",
        "dim",
        "gemm_m",
        "gemm_n",
        "gemm_k",
        "dma_max_inflight",
        "dma_retry_ticks",
        "dma_burst_bytes",
        "group_max_inflight_per_node",
        "ctrl_overlap_ab",
        "noc_link_bw",
        "exec_total_mean",
        "memory_queue_delay_p99_cycles",
        "dma_timeout_retry_sum",
        "noc_total_xbar_stalls",
        "weighted_score",
        "log_file",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in front:
            w.writerow({k: row.get(k, "") for k in fields})

    if topk_path is not None:
        ranked = sorted(candidates, key=lambda x: x[required_metrics[0]])
        _normalized_score(ranked, required_metrics, weights)
        ranked.sort(key=lambda x: x["weighted_score"])
        topk_path.parent.mkdir(parents=True, exist_ok=True)
        with topk_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in ranked[: max(1, args.topk)]:
                w.writerow({k: row.get(k, "") for k in fields})

    print(f"[OK] pareto rows={len(front)} written: {output_path}")
    if topk_path is not None:
        print(f"[OK] topk written: {topk_path}")


if __name__ == "__main__":
    main()
