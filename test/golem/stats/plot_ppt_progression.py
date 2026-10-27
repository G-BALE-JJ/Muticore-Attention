#!/usr/bin/env python3
import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {
    "ink": "#18212f",
    "muted": "#64748b",
    "grid": "#e2e8f0",
    "blue": "#2563eb",
    "green": "#16a34a",
    "orange": "#f97316",
    "red": "#dc2626",
    "purple": "#7c3aed",
    "teal": "#0891b2",
    "gray": "#94a3b8",
}


STAGES = [
    {
        "label": "Naive DMA\nchunk resp",
        "match": "internal_resp_chunk_8kb",
        "note": "logical request + response chunks",
    },
    {
        "label": "Partial\ncredit",
        "match": "partial_credit_8kb",
        "note": "chunk-level node credit return",
    },
    {
        "label": "Prefetch\ndepth 8",
        "match": "20260428_152959_2710000",
        "note": "more in-flight prefetch",
    },
    {
        "label": "Credit\ntuned",
        "match": "20260428_171730_2795700",
        "note": "node chunk credits = 82",
    },
    {
        "label": "DONE\nseparate",
        "match": "20260429_002540_2927358",
        "note": "mat/vec credit recycle split",
    },
    {
        "label": "Response\nVN1",
        "match": "20260429_005110_2962972",
        "note": "DMA response virtual network",
    },
]


def setup_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "font.family": "DejaVu Sans",
            "savefig.bbox": "tight",
            "savefig.dpi": 180,
        }
    )


def parse_float(row, key, default=float("nan")):
    try:
        value = row.get(key, "")
        if value in (None, ""):
            return default
        return float(value)
    except ValueError:
        return default


def parse_int(row, key, default=0):
    try:
        value = row.get(key, "")
        if value in (None, ""):
            return default
        return int(float(value))
    except ValueError:
        return default


def short_run_id(row):
    log_file = row.get("log_file", "")
    match = re.search(r"run_(\d{8})_(\d{6})_(\d+)", log_file)
    if not match:
        return ""
    return f"{match.group(2)}_{match.group(3)}"


def load_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def select_stages(rows):
    selected = []
    for stage in STAGES:
        match = stage["match"]
        found = None
        for row in rows:
            if match in row.get("log_file", ""):
                found = row
        if found is None:
            continue
        item = dict(stage)
        item["row"] = found
        selected.append(item)
    return selected


def hbm_read_util_pct(row):
    existing = parse_float(row, "hbm_utilization_pct")
    if existing == existing:
        return existing

    total_cycles = parse_float(row, "exec_total_cycles")
    if not total_cycles or total_cycles != total_cycles:
        return float("nan")

    elem_bytes = 4
    m = parse_int(row, "gemm_m")
    n = parse_int(row, "gemm_n")
    k = parse_int(row, "gemm_k")
    bm = parse_int(row, "block_m")
    bn = parse_int(row, "block_n")
    bk = parse_int(row, "block_k")
    if min(m, n, k, bm, bn, bk) <= 0:
        return float("nan")

    # The 1024 progression uses 4x4 reuse. Older summaries did not persist these fields.
    a_reuse_n = parse_int(row, "a_reuse_n_tiles", 4) or 4
    b_reuse_m = parse_int(row, "b_reuse_m_tiles", 4) or 4
    m_tiles = math.ceil(m / bm)
    n_tiles = math.ceil(n / bn)
    k_tiles = math.ceil(k / bk)
    m_groups = math.ceil(m_tiles / b_reuse_m)
    n_groups = math.ceil(n_tiles / a_reuse_n)
    useful_read_bytes = (
        m_groups
        * n_groups
        * k_tiles
        * elem_bytes
        * (b_reuse_m * bm * bk + a_reuse_n * bn * bk)
    )
    data_nodes = max(1, parse_int(row, "num_mem_nodes", 5) - 1)
    roofline_bpc = data_nodes * 16 * 64 / 3
    return 100.0 * useful_read_bytes / (total_cycles * roofline_bpc)


def annotate_values(ax, xs, ys, fmt, dy=4, color=COLORS["ink"]):
    for x, y in zip(xs, ys):
        if y != y:
            continue
        ax.annotate(
            fmt.format(y),
            (x, y),
            textcoords="offset points",
            xytext=(0, dy),
            ha="center",
            color=color,
            fontsize=9,
        )


def save(fig, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    svg = out_dir / f"{name}.svg"
    fig.savefig(png)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def plot_progression(stages, out_dir):
    labels = [s["label"] for s in stages]
    xs = list(range(len(stages)))
    cycles = [parse_float(s["row"], "exec_total_cycles") for s in stages]
    util = [parse_float(s["row"], "exec_array_utilization_pct") for s in stages]
    base_cycles = cycles[0]
    speedup = [base_cycles / c if c and c == c else float("nan") for c in cycles]

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.plot(xs, cycles, color=COLORS["blue"], marker="o", linewidth=2.8, markersize=7)
    ax.fill_between(xs, cycles, color=COLORS["blue"], alpha=0.08)
    ax.set_title("Optimization Progression: 1024x1024 GEMM")
    ax.set_ylabel("Total cycles (lower is better)")
    ax.set_xticks(xs, labels)
    ax.grid(axis="y")
    annotate_values(ax, xs, cycles, "{:.0f}", dy=8, color=COLORS["blue"])

    ax2 = ax.twinx()
    ax2.plot(xs, util, color=COLORS["green"], marker="s", linewidth=2.2, markersize=6)
    ax2.set_ylabel("Array utilization (%)")
    ax2.tick_params(axis="y", colors=COLORS["green"])
    annotate_values(ax2, xs, util, "{:.1f}%", dy=-16, color=COLORS["green"])

    for x, sp in zip(xs, speedup):
        if sp == sp:
            ax.text(x, min(cycles) * 0.985, f"{sp:.2f}x", ha="center", va="top", color=COLORS["muted"], fontsize=9)

    save(fig, out_dir, "01_progression_cycles_util")


def plot_breakdown(stages, out_dir):
    labels = [s["label"] for s in stages]
    xs = list(range(len(stages)))
    compute = [parse_float(s["row"], "exec_breakdown_compute_active_time", 0.0) for s in stages]
    prefetch = [parse_float(s["row"], "exec_breakdown_prefetch_wait_time", 0.0) for s in stages]
    writeback = [parse_float(s["row"], "exec_breakdown_writeback_wait_time", 0.0) for s in stages]
    other = [parse_float(s["row"], "exec_breakdown_control_other_time", 0.0) for s in stages]

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.bar(xs, compute, color=COLORS["green"], label="Compute active")
    ax.bar(xs, prefetch, bottom=compute, color=COLORS["orange"], label="Prefetch wait")
    bottom = [a + b for a, b in zip(compute, prefetch)]
    ax.bar(xs, writeback, bottom=bottom, color=COLORS["purple"], label="Writeback wait")
    bottom = [a + b for a, b in zip(bottom, writeback)]
    ax.bar(xs, other, bottom=bottom, color=COLORS["gray"], label="Other")
    ax.set_title("Where Cycles Go")
    ax.set_ylabel("Cycles")
    ax.set_xticks(xs, labels)
    ax.grid(axis="y")
    ax.legend(loc="upper right")
    save(fig, out_dir, "02_cycle_breakdown")


def plot_memory_roofline(stages, out_dir):
    labels = [s["label"] for s in stages]
    xs = list(range(len(stages)))
    hbm_util = [hbm_read_util_pct(s["row"]) for s in stages]
    mem_avg = [parse_float(s["row"], "memory_backend_read_latency_avg_cycles") for s in stages]
    mem_p99 = [parse_float(s["row"], "memory_backend_read_latency_p99_cycles") for s in stages]

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.plot(xs, hbm_util, color=COLORS["teal"], marker="o", linewidth=2.8, markersize=7, label="HBM read util (tCCD_L)")
    ax.axhline(100, color=COLORS["red"], linewidth=1.2, linestyle="--", alpha=0.7, label="Read roofline")
    ax.set_title("HBM Read Roofline Utilization")
    ax.set_ylabel("Read-only tCCD_L utilization (%)")
    ax.set_xticks(xs, labels)
    ax.set_ylim(0, max(110, max(v for v in hbm_util if v == v) * 1.12))
    ax.grid(axis="y")
    annotate_values(ax, xs, hbm_util, "{:.1f}%", dy=8, color=COLORS["teal"])

    ax2 = ax.twinx()
    ax2.plot(xs, mem_avg, color=COLORS["blue"], marker="s", linewidth=1.8, markersize=5, label="Backend avg latency")
    ax2.plot(xs, mem_p99, color=COLORS["purple"], marker="^", linewidth=1.8, markersize=5, label="Backend p99 latency")
    ax2.set_ylabel("Backend read latency (cycles)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    save(fig, out_dir, "03_hbm_roofline")


def plot_return_path(stages, out_dir):
    labels = [s["label"] for s in stages]
    xs = list(range(len(stages)))
    mat = [parse_float(s["row"], "causal_return_path_mat_mean_cycles") for s in stages]
    vec = [parse_float(s["row"], "causal_return_path_vec_mean_cycles") for s in stages]
    service = [parse_float(s["row"], "causal_memory_service_mean_cycles") for s in stages]

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.plot(xs, service, color=COLORS["green"], marker="o", linewidth=2.2, markersize=6, label="HBM service")
    ax.plot(xs, mat, color=COLORS["orange"], marker="s", linewidth=2.2, markersize=6, label="Return path mat")
    ax.plot(xs, vec, color=COLORS["purple"], marker="^", linewidth=2.2, markersize=6, label="Return path vec")
    ax.set_title("Request-Level Causal Latency")
    ax.set_ylabel("Mean cycles")
    ax.set_xticks(xs, labels)
    ax.grid(axis="y")
    ax.legend(loc="upper right")
    save(fig, out_dir, "04_causal_return_path")


def write_stage_table(stages, out_dir):
    out = out_dir / "ppt_progression_table.csv"
    fields = [
        "stage",
        "run_id",
        "note",
        "total_cycles",
        "speedup_vs_first",
        "array_util_pct",
        "hbm_read_util_tccdl_pct",
        "prefetch_wait_cycles",
        "writeback_wait_cycles",
        "memory_service_mean_cycles",
        "return_path_mat_mean_cycles",
        "return_path_vec_mean_cycles",
    ]
    first = parse_float(stages[0]["row"], "exec_total_cycles") if stages else float("nan")
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for stage in stages:
            row = stage["row"]
            cycles = parse_float(row, "exec_total_cycles")
            writer.writerow(
                {
                    "stage": stage["label"].replace("\n", " "),
                    "run_id": short_run_id(row),
                    "note": stage["note"],
                    "total_cycles": f"{cycles:.6f}",
                    "speedup_vs_first": f"{first / cycles:.6f}" if cycles == cycles else "",
                    "array_util_pct": f"{parse_float(row, 'exec_array_utilization_pct'):.6f}",
                    "hbm_read_util_tccdl_pct": f"{hbm_read_util_pct(row):.6f}",
                    "prefetch_wait_cycles": row.get("exec_breakdown_prefetch_wait_time", ""),
                    "writeback_wait_cycles": row.get("exec_breakdown_writeback_wait_time", ""),
                    "memory_service_mean_cycles": row.get("causal_memory_service_mean_cycles", ""),
                    "return_path_mat_mean_cycles": row.get("causal_return_path_mat_mean_cycles", ""),
                    "return_path_vec_mean_cycles": row.get("causal_return_path_vec_mean_cycles", ""),
                }
            )
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate PPT-ready GOLEM optimization progression plots")
    parser.add_argument("--run-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    setup_style()
    rows = load_rows(args.run_summary)
    stages = select_stages(rows)
    if len(stages) < 2:
        raise SystemExit("Not enough matching stages in run_summary.csv")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_progression(stages, args.output_dir)
    plot_breakdown(stages, args.output_dir)
    plot_memory_roofline(stages, args.output_dir)
    plot_return_path(stages, args.output_dir)
    table = write_stage_table(stages, args.output_dir)
    print(f"[OK] wrote plots and {table}")


if __name__ == "__main__":
    main()
