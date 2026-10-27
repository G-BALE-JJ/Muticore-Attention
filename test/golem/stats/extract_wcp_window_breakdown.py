#!/usr/bin/env python3
"""Extract per-core, per-window WCP and frontend timing breakdowns."""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


WINDOW_RE = re.compile(
    r"\[Core\s+(?P<core>\d+)\].*?WINDOW_BREAKDOWN"
    r" task_idx=(?P<task_idx>\d+) task=(?P<task>\d+) macro=(?P<macro>\d+)"
    r" window=(?P<window>\d+) k_begin=(?P<k_begin>\d+) k_count=(?P<k_count>\d+)"
    r" tile_count=(?P<tile_count>\d+) buffer=(?P<buffer>\d+) txn=(?P<txn>\d+)"
    r" submit=(?P<submit>\d+) first_data=(?P<first_data>\d+)"
    r" first_tile_ready=(?P<first_tile_ready>\d+) ready=(?P<ready>\d+)"
    r" activate=(?P<activate>\d+) compute_start=(?P<compute_start>\d+)"
    r" compute_end=(?P<compute_end>\d+) next_ready=(?P<next_ready>\d+)"
    r"(?: compute_segments=(?P<compute_segments>\d+) compute_active=(?P<compute_active>\d+)"
    r" last_segment_end=(?P<last_segment_end>\d+)"
    r" intra_window_transitions=(?P<intra_window_transitions>\d+)"
    r" intra_window_bubbles=(?P<intra_window_bubbles>\d+)"
    r" intra_window_bubble_cycles=(?P<intra_window_bubble_cycles>\d+))?"
)
FRONTEND_RE = re.compile(
    r"\[Core\s+(?P<core>\d+)\].*?FRONTEND_BREAKDOWN"
    r" barrier=(?P<barrier>\d+) descriptor_mm2gm_start=(?P<desc_start>\d+)"
    r" descriptor_mm2gm_end=(?P<desc_end>\d+)"
)
MVM_PROGRESS_RE = re.compile(
    r"RoCC core=\d+ MVM_PROGRESS: completed=\d+/\d+ \(\d+%\) cycle=\d+\r?\n"
)

# Every compute transition includes one cycle of WCP/array event handoff.
# Bubble metrics report only the interval beyond that mandatory handoff.
MANDATORY_HANDOFF_CYCLES = 1

DETAIL_FIELDS = [
    "core", "task_idx", "task", "macro", "window", "k_begin", "k_count",
    "tile_count", "buffer", "txn", "barrier_cycle", "descriptor_mm2gm_start",
    "descriptor_mm2gm_end", "wcp_first_tick", "submit_cycle", "first_data_cycle",
    "first_tile_ready_cycle", "ready_cycle", "activate_cycle", "compute_start_cycle",
    "compute_end_cycle", "next_window_ready_cycle", "startup_overhead_cycles",
    "descriptor_mm2gm_cycles", "frontend_after_descriptor_cycles", "fill_latency_cycles",
    "window_ready_latency_cycles", "compute_span_cycles", "exposed_gap_cycles",
    "compute_segment_count", "compute_active_cycles", "last_compute_segment_end_cycle",
    "compute_segment_span_cycles", "intra_window_transition_count",
    "intra_window_bubble_count", "intra_window_bubble_cycles",
]


def read_inputs(log: Path, log_dir: Path):
    # The run directory can contain copied summaries and auxiliary files.  Only
    # consume text output fragments plus the canonical combined log.
    paths = sorted(p for p in log_dir.glob("stdout-*") if p.is_file()) if log_dir.exists() else []
    if log.exists():
        paths.append(log)
    return paths


def read_log_lines(path: Path):
    # mpirun can splice an asynchronous MVM progress message into a longer WCP
    # record. Removing that complete message before splitting restores the two
    # fragments (for example, "buffe" + "r=2") to their original record.
    text = path.read_text(errors="ignore")
    return MVM_PROGRESS_RE.sub("", text).splitlines()


def p95(values):
    if not values:
        return 0
    vals = sorted(values)
    return vals[int(round(0.95 * (len(vals) - 1)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--core-summary", type=Path, required=True)
    args = ap.parse_args()

    frontend = {}
    window_records = {}
    paths = read_inputs(args.log, args.log_dir)
    lines = []
    for path in paths:
        lines.extend(read_log_lines(path))

    # Parse frontend and WCP aggregate records in separate passes so file order
    # cannot make a window inherit stale per-core metadata.
    for line in lines:
        fm = FRONTEND_RE.search(line)
        if fm:
            frontend[int(fm.group("core"))] = {
                k: int(v) for k, v in fm.groupdict().items() if k != "core"
            }
    for line in lines:
        wm = WINDOW_RE.search(line)
        if wm:
            raw = {k: int(v) if v is not None else 0 for k, v in wm.groupdict().items()}
            key = (raw["core"], raw["txn"])
            rec = {
                "core": raw["core"], "task_idx": raw["task_idx"],
                "task": raw["task"], "macro": raw["macro"],
                "window": raw["window"], "k_begin": raw["k_begin"],
                "k_count": raw["k_count"], "tile_count": raw["tile_count"],
                "buffer": raw["buffer"], "txn": raw["txn"],
                "submit_cycle": raw["submit"],
                "first_data_cycle": raw["first_data"],
                "first_tile_ready_cycle": raw["first_tile_ready"],
                "ready_cycle": raw["ready"],
                "activate_cycle": raw["activate"],
                "compute_start_cycle": raw["compute_start"],
                "compute_end_cycle": raw["compute_end"],
                "next_window_ready_cycle": raw["next_ready"],
                "compute_segment_count": raw["compute_segments"],
                "compute_active_cycles": raw["compute_active"],
                "last_compute_segment_end_cycle": raw["last_segment_end"],
                "intra_window_transition_count": raw["intra_window_transitions"],
                "intra_window_bubble_count": raw["intra_window_bubbles"],
                "intra_window_bubble_cycles": raw["intra_window_bubble_cycles"],
            }
            meta = frontend.get(raw["core"], {})
            rec["barrier_cycle"] = meta.get("barrier", 0)
            rec["descriptor_mm2gm_start"] = meta.get("desc_start", 0)
            rec["descriptor_mm2gm_end"] = meta.get("desc_end", 0)
            # Prefer the combined log's complete record over an incomplete
            # fragment if both contain the same transaction.
            old = window_records.get(key)
            if old is None or sum(rec[k] != 0 for k in (
                "submit_cycle", "first_data_cycle", "ready_cycle",
                "compute_start_cycle", "compute_end_cycle", "compute_segment_count")) >= sum(old[k] != 0 for k in (
                    "submit_cycle", "first_data_cycle", "ready_cycle",
                    "compute_start_cycle", "compute_end_cycle", "compute_segment_count")):
                window_records[key] = rec
    windows = list(window_records.values())

    # WCP emits its aggregate line separately; use the latest start_cycle per core.
    start_re = re.compile(r"\[Core\s+(\d+)\].*?LATENCY\(cycles\):.*?\sstart_cycle=(\d+)")
    first_ticks = {}
    for line in lines:
        m = start_re.search(line)
        if m:
            first_ticks[int(m.group(1))] = int(m.group(2))
    for rec in windows:
        rec["wcp_first_tick"] = first_ticks.get(rec["core"], 0)

    for rec in windows:
        barrier = rec["barrier_cycle"]
        first_tick = rec["wcp_first_tick"]
        desc_start = rec["descriptor_mm2gm_start"]
        desc_end = rec["descriptor_mm2gm_end"]
        rec["startup_overhead_cycles"] = max(0, first_tick - barrier) if first_tick and barrier else 0
        rec["descriptor_mm2gm_cycles"] = max(0, desc_end - desc_start) if desc_end and desc_start else 0
        rec["frontend_after_descriptor_cycles"] = max(0, first_tick - desc_end) if first_tick and desc_end else 0
        rec["fill_latency_cycles"] = max(0, rec["first_tile_ready_cycle"] - rec["submit_cycle"]) if rec["first_tile_ready_cycle"] else 0
        rec["window_ready_latency_cycles"] = max(0, rec["ready_cycle"] - rec["submit_cycle"]) if rec["ready_cycle"] else 0
        rec["compute_span_cycles"] = max(0, rec["compute_end_cycle"] - rec["compute_start_cycle"]) if rec["compute_end_cycle"] and rec["compute_start_cycle"] else 0
        rec["compute_segment_span_cycles"] = max(0, rec["last_compute_segment_end_cycle"] - rec["compute_start_cycle"]) if rec["last_compute_segment_end_cycle"] and rec["compute_start_cycle"] else rec["compute_span_cycles"]
        rec["exposed_gap_cycles"] = max(0, rec["next_window_ready_cycle"] - rec["compute_end_cycle"]) if rec["next_window_ready_cycle"] and rec["compute_end_cycle"] else 0

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        writer.writerows({field: rec.get(field, 0) for field in DETAIL_FIELDS} for rec in windows)

    grouped = defaultdict(list)
    for rec in windows:
        grouped[rec["core"]].append(rec)
    summary_fields = [
        "core", "window_count", "startup_overhead_cycles", "first_window_fill_cycles",
        "fill_mean_cycles", "fill_p95_cycles", "ready_mean_cycles", "compute_mean_cycles",
        "exposed_gap_total_cycles", "exposed_gap_mean_cycles", "exposed_gap_p95_cycles",
        "compute_segment_count", "compute_active_cycles",
        "intra_window_transition_count", "intra_window_bubble_count",
        "intra_window_bubble_cycles", "inter_window_transition_count",
        "inter_window_bubble_count", "inter_window_bubble_cycles",
        # Compatibility aliases. These old fields measured only transitions
        # between windows in the same macro.
        "intra_macro_transition_count", "intra_macro_bubble_count",
        "intra_macro_bubble_cycles", "inter_macro_transition_count",
        "inter_macro_bubble_count", "inter_macro_bubble_cycles",
        "legacy_inter_macro_transition_count", "legacy_inter_macro_bubble_count",
        "legacy_inter_macro_bubble_cycles",
    ]
    with args.core_summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for core in sorted(grouped):
            rows = grouped[core]
            fills = [r["fill_latency_cycles"] for r in rows if r["fill_latency_cycles"]]
            ready = [r["window_ready_latency_cycles"] for r in rows if r["window_ready_latency_cycles"]]
            compute = [r["compute_span_cycles"] for r in rows if r["compute_span_cycles"]]
            gaps = [r["exposed_gap_cycles"] for r in rows]
            first = next((r["fill_latency_cycles"] for r in sorted(rows, key=lambda x: x["window"])), 0)
            compute_segments = sum(r["compute_segment_count"] for r in rows)
            compute_active_cycles = sum(r["compute_active_cycles"] for r in rows)
            intra_window_transitions = sum(r["intra_window_transition_count"] for r in rows)
            intra_window_bubbles = sum(r["intra_window_bubble_count"] for r in rows)
            intra_window_bubble_cycles = sum(r["intra_window_bubble_cycles"] for r in rows)
            inter_window_transitions = 0
            inter_window_bubbles = 0
            inter_window_bubble_cycles = 0
            inter_transitions = 0
            inter_bubbles = 0
            inter_bubble_cycles = 0
            legacy_intra_transitions = 0
            legacy_intra_bubbles = 0
            legacy_intra_bubble_cycles = 0
            legacy_inter_transitions = 0
            legacy_inter_bubbles = 0
            legacy_inter_bubble_cycles = 0
            ordered = sorted(
                (r for r in rows if r["compute_start_cycle"] and r["compute_end_cycle"]),
                key=lambda r: (r["compute_start_cycle"], r["compute_end_cycle"]),
            )
            for previous, current in zip(ordered, ordered[1:]):
                previous_segment_end = (
                    previous["last_compute_segment_end_cycle"] or previous["compute_end_cycle"]
                )
                gap = max(0, current["compute_start_cycle"] - previous_segment_end)
                legacy_gap = max(0, current["compute_start_cycle"] - previous["compute_end_cycle"])
                exposed_gap = max(0, gap - MANDATORY_HANDOFF_CYCLES)
                exposed_legacy_gap = max(0, legacy_gap - MANDATORY_HANDOFF_CYCLES)
                same_macro = (
                    current["task_idx"] == previous["task_idx"]
                    and current["macro"] == previous["macro"]
                )
                if same_macro:
                    inter_window_transitions += 1
                    inter_window_bubble_cycles += exposed_gap
                    inter_window_bubbles += int(exposed_gap > 0)
                    legacy_intra_transitions += 1
                    legacy_intra_bubble_cycles += exposed_legacy_gap
                    legacy_intra_bubbles += int(exposed_legacy_gap > 0)
                else:
                    inter_transitions += 1
                    inter_bubble_cycles += exposed_gap
                    inter_bubbles += int(exposed_gap > 0)
                    legacy_inter_transitions += 1
                    legacy_inter_bubble_cycles += exposed_legacy_gap
                    legacy_inter_bubbles += int(exposed_legacy_gap > 0)
            writer.writerow({
                "core": core,
                "window_count": len(rows),
                "startup_overhead_cycles": rows[0]["startup_overhead_cycles"],
                "first_window_fill_cycles": first,
                "fill_mean_cycles": sum(fills) / len(fills) if fills else 0,
                "fill_p95_cycles": p95(fills),
                "ready_mean_cycles": sum(ready) / len(ready) if ready else 0,
                "compute_mean_cycles": sum(compute) / len(compute) if compute else 0,
                "exposed_gap_total_cycles": sum(gaps),
                "exposed_gap_mean_cycles": sum(gaps) / len(gaps) if gaps else 0,
                "exposed_gap_p95_cycles": p95(gaps),
                "compute_segment_count": compute_segments,
                "compute_active_cycles": compute_active_cycles,
                "intra_window_transition_count": intra_window_transitions,
                "intra_window_bubble_count": intra_window_bubbles,
                "intra_window_bubble_cycles": intra_window_bubble_cycles,
                "inter_window_transition_count": inter_window_transitions,
                "inter_window_bubble_count": inter_window_bubbles,
                "inter_window_bubble_cycles": inter_window_bubble_cycles,
                "intra_macro_transition_count": legacy_intra_transitions,
                "intra_macro_bubble_count": legacy_intra_bubbles,
                "intra_macro_bubble_cycles": legacy_intra_bubble_cycles,
                "inter_macro_transition_count": inter_transitions,
                "inter_macro_bubble_count": inter_bubbles,
                "inter_macro_bubble_cycles": inter_bubble_cycles,
                "legacy_inter_macro_transition_count": legacy_inter_transitions,
                "legacy_inter_macro_bubble_count": legacy_inter_bubbles,
                "legacy_inter_macro_bubble_cycles": legacy_inter_bubble_cycles,
            })


if __name__ == "__main__":
    main()
