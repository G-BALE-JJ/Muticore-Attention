#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


COLORS = {
    "total": "#1f2937",
    "compute": "#4e79a7",
    "dma": "#f28e2b",
    "sync": "#e15759",
    "xbar": "#76b7b2",
    "avg": "#59a14f",
    "p99": "#b07aa1",
    "retry": "#edc948",
    "poll": "#9c755f",
    "hotspot": "#76b7b2",
    "imbalance": "#bab0ab",
    "util": "#2f4858",
}

MEMORY_FIELDS = [
    ("memory_avg_read_latency_cycles", "Avg read latency", COLORS["avg"]),
    ("memory_backend_read_latency_avg_cycles", "Backend read avg", COLORS["compute"]),
    ("memory_backend_read_latency_p99_cycles", "Backend read p99", COLORS["p99"]),
]


def setup_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#d1d5db",
            "axes.linewidth": 0.8,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "grid.color": "#e5e7eb",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "font.family": "DejaVu Sans",
            "savefig.bbox": "tight",
        }
    )


def parse_float(row, key):
    value = row.get(key, "")
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_rows(csv_path: Path, x_field: str):
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r[x_field]))
    return rows


def values(rows, key):
    out = []
    for row in rows:
        val = parse_float(row, key)
        out.append(float("nan") if val is None else float(val))
    return out


def values0(rows, key):
    out = []
    for row in rows:
        val = parse_float(row, key)
        out.append(0.0 if val is None else float(val))
    return out


def read_metric_csv(path: Path):
    if not path.exists():
        return {}
    out = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            metric = row.get("metric", "").strip()
            if not metric:
                continue
            try:
                if row.get("mean", "") not in (None, ""):
                    out[metric] = float(row.get("mean", "") or 0.0)
                elif row.get("value", "") not in (None, ""):
                    out[metric] = float(row.get("value", "") or 0.0)
                else:
                    continue
            except ValueError:
                continue
    return out


def resolve_stats_dir_from_row(row):
    log_file = (row.get("log_file") or "").strip()
    overlap = (row.get("overlap") or "overlap0").strip()
    if not log_file:
        return None

    log_path = Path(log_file)
    run_id = log_path.stem.replace("test_default_", "").replace("test_", "")
    return log_path.parent.parent / "stats" / overlap / run_id


def get_exec_metric_strict(row, metric, cache):
    stats_dir = resolve_stats_dir_from_row(row)
    if stats_dir is None:
        return None

    key = str(stats_dir)
    if key not in cache:
        cache[key] = read_metric_csv(stats_dir / "execution_summary.csv")
    val = cache[key].get(metric)
    return val


def get_memory_metric_strict(row, metric, cache):
    stats_dir = resolve_stats_dir_from_row(row)
    if stats_dir is None:
        return None

    key = str(stats_dir)
    if key not in cache:
        cache[key] = read_metric_csv(stats_dir / "memory_summary.csv")
    return cache[key].get(metric)


def sci_y(ax):
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(fmt)


def hide_offset_text(ax):
    ax.yaxis.get_offset_text().set_visible(False)


def base_axis(ax, xpos, labels, ylabel, x_label, *, sci=True):
    ax.set_xlabel(x_label)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if sci:
        sci_y(ax)


def annotate_point(ax, x, y, text, dx, dy):
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=8.5,
        color="#374151",
        arrowprops={"arrowstyle": "-", "color": "#9ca3af", "lw": 0.8},
        bbox={
            "boxstyle": "round,pad=0.2",
            "fc": "#ffffff",
            "ec": "#e5e7eb",
            "alpha": 0.95,
        },
    )


def parse_noc_router_hotspots(log_file: Path, top_k=10):
    pattern = re.compile(r"rtr_(\d+),xbar_stalls,port\d+,Accumulator,\d+,\d+,(\d+),")
    router_stalls = {}
    if not log_file.exists():
        return []
    for line in log_file.read_text(errors="ignore").splitlines():
        m = pattern.match(line)
        if not m:
            continue
        router = int(m.group(1))
        value = int(m.group(2))
        router_stalls[router] = router_stalls.get(router, 0) + value

    hottest = sorted(router_stalls.items(), key=lambda item: item[1], reverse=True)[
        :top_k
    ]
    return hottest


def draw_mesh_hotspot_inset(ax, stats_file: Path, dim_label: str):
    hottest = parse_noc_router_hotspots(stats_file)
    inset = ax.inset_axes([0.72, 0.74, 0.24, 0.22])
    inset.set_title("Router hotspots", fontsize=8, pad=2)
    xs = [c for r in range(5) for c in range(4)]
    ys = [4 - r for r in range(5) for c in range(4)]
    base_weights = {(r * 4 + c): 0.0 for r in range(5) for c in range(4)}

    max_val = hottest[0][1] if hottest else 1
    for router, val in hottest:
        base_weights[router] = val / max_val

    node_weights = [base_weights[r * 4 + c] for r in range(5) for c in range(4)]
    max_weight = max(node_weights) if node_weights else 1.0
    if max_weight <= 0:
        max_weight = 1.0
    order = sorted(
        range(len(node_weights)), key=lambda i: node_weights[i], reverse=True
    )
    inset.scatter(xs, ys, s=10, color="#cbd5e1", alpha=0.60, zorder=2)

    hot_x = []
    hot_y = []
    hot_s = []
    hot_c = []
    for rank, i in enumerate(order):
        w = node_weights[i]
        if w <= 0:
            continue
        hot_x.append(xs[i])
        hot_y.append(ys[i])
        if rank < 3:
            hot_s.append(95.0)
            hot_c.append("#115e59")
        elif rank < 6:
            hot_s.append(68.0)
            hot_c.append("#2a9d8f")
        elif rank < 10:
            hot_s.append(42.0)
            hot_c.append("#7fcdbb")
        else:
            hot_s.append(24.0)
            hot_c.append("#bfe6df")

    if hot_x:
        inset.scatter(
            hot_x,
            hot_y,
            s=hot_s,
            c=hot_c,
            alpha=0.78,
            edgecolors="#134e4a",
            linewidths=0.6,
            zorder=3,
        )

    inset.set_xlim(-0.4, 3.4)
    inset.set_ylim(-0.4, 4.4)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_visible(False)
    inset.set_facecolor((1, 1, 1, 0.60))


def plot_exec(ax, xpos, labels, rows, x_label):
    strict_cache = {}
    total = []
    compute = []
    dma = []
    dma_issue = []
    sched = []
    c_store = []
    group_wait = []
    task_desc = []
    nloop = []
    submit_pack = []
    finish_publish = []

    # Strict mode: prefer per-run execution_summary.csv; fallback to sweep CSV columns.
    for row in rows:
        c = get_exec_metric_strict(row, "compute", strict_cache)
        d = get_exec_metric_strict(row, "dma_wait", strict_cache)
        di = get_exec_metric_strict(row, "dma_issue", strict_cache)
        s = get_exec_metric_strict(row, "sched_protocol", strict_cache)
        cs = get_exec_metric_strict(row, "c_store", strict_cache)
        gw = get_exec_metric_strict(row, "group_wait", strict_cache)
        td = get_exec_metric_strict(row, "task_desc", strict_cache)
        nl = get_exec_metric_strict(row, "nloop", strict_cache)
        sp = get_exec_metric_strict(row, "submit_pack", strict_cache)
        fp = get_exec_metric_strict(row, "finish_publish", strict_cache)

        if c is None:
            c = parse_float(row, "exec_compute_mean")
        if d is None:
            d = parse_float(row, "exec_dma_wait_mean")
        if di is None:
            di = parse_float(row, "exec_dma_issue_mean") or 0.0
        if s is None:
            s = parse_float(row, "exec_sched_protocol_mean") or 0.0
        if cs is None:
            cs = parse_float(row, "exec_c_store_mean") or 0.0
        if gw is None:
            gw = parse_float(row, "exec_group_wait_mean") or 0.0
        if td is None:
            td = parse_float(row, "exec_other_task_desc_mean") or 0.0
        if nl is None:
            nl = parse_float(row, "exec_other_nloop_mean") or 0.0
        if sp is None:
            sp = parse_float(row, "exec_other_submit_pack_mean") or 0.0
        if fp is None:
            fp = parse_float(row, "exec_other_finish_publish_mean") or 0.0

        compute.append(float("nan") if c is None else float(c))
        dma.append(float("nan") if d is None else float(d))
        dma_issue.append(float(di))
        sched.append(float(s))
        c_store.append(float(cs))
        group_wait.append(float(gw))
        task_desc.append(float(td))
        nloop.append(float(nl))
        submit_pack.append(float(sp))
        finish_publish.append(float(fp))

    task_desc_only = task_desc
    total = [
        compute[i] + dma[i] + task_desc_only[i] + dma_issue[i]
        for i in range(len(xpos))
    ]

    ax.bar(
        [p - 0.27 for p in xpos],
        compute,
        width=0.24,
        color=COLORS["compute"],
        alpha=0.72,
        label="Compute",
    )
    ax.bar(
        [p for p in xpos],
        dma,
        width=0.24,
        color=COLORS["dma"],
        alpha=0.72,
        label="DMA wait",
    )
    ax.bar(
        [p + 0.27 for p in xpos],
        task_desc_only,
        width=0.24,
        color=COLORS["sync"],
        alpha=0.72,
        label="Task description",
    )
    ax.plot(
        xpos,
        total,
        color=COLORS["total"],
        marker="o",
        linewidth=2.5,
        markersize=6,
        label="Total",
    )
    ax.set_title("Execution Breakdown")
    base_axis(ax, xpos, labels, "Cycles", x_label)
    ax.legend(ncol=2, loc="upper left")
    annotate_point(
        ax,
        xpos[-1],
        total[-1],
        "grow with scale",
        -90,
        -22,
    )


def plot_noc(ax, xpos, labels, rows):
    raise RuntimeError("Use plot_noc_broken instead")


def plot_noc_clean(ax, xpos, labels, rows, x_label):
    ax2 = ax.twinx()
    stalls = values(rows, "noc_total_xbar_stalls")
    avg = values(rows, "noc_avg_packet_latency_ns")
    p99 = values(rows, "noc_p99_packet_latency_ns")

    ax.bar(
        xpos,
        stalls,
        width=0.40,
        color=COLORS["xbar"],
        alpha=0.45,
        edgecolor=COLORS["xbar"],
        linewidth=1.2,
        label="Xbar stalls",
    )
    ax2.plot(
        xpos,
        avg,
        color=COLORS["avg"],
        marker="o",
        linewidth=2.3,
        markersize=6,
        label="Avg packet latency",
    )
    ax2.plot(
        xpos,
        p99,
        color=COLORS["p99"],
        marker="o",
        linewidth=2.3,
        markersize=6,
        label="P99 packet latency",
    )
    ax.set_title("NoC Analysis")
    base_axis(ax, xpos, labels, "Xbar stalls", x_label, sci=False)
    ax2.set_ylabel("Packet latency (ns)")
    ax2.spines["top"].set_visible(False)
    hide_offset_text(ax2)
    ax.set_ylabel("Xbar stalls", color=COLORS["xbar"])
    ax.tick_params(axis="y", colors=COLORS["xbar"])
    ax.spines["left"].set_color(COLORS["xbar"])
    ax2.set_ylabel("Packet latency (ns)", color=COLORS["avg"])
    ax2.tick_params(axis="y", colors=COLORS["avg"])
    ax2.spines["right"].set_color(COLORS["avg"])
    ax2.set_ylim(0, max(p99) * 1.35)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", bbox_to_anchor=(0.02, 0.98))

    if rows and rows[-1].get("log_file"):
        log_path = Path(rows[-1]["log_file"])
        run_id = log_path.stem.replace("test_default_", "").replace("test_", "")
        overlap_label = rows[-1].get("overlap", "overlap0")
        stats_file = (
            log_path.parent.parent
            / "stats"
            / overlap_label
            / run_id
            / "stats_selfcom.txt"
        )
        if stats_file.exists():
            draw_mesh_hotspot_inset(ax, stats_file, rows[-1].get("gemm_m", "?"))


def plot_noc_broken(fig, gs, xpos, labels, rows):
    top = fig.add_subplot(gs[0])
    bottom = fig.add_subplot(gs[1], sharex=top)
    stalls = values(rows, "noc_total_xbar_stalls")
    avg = values(rows, "noc_avg_packet_latency_ns")
    p99 = values(rows, "noc_p99_packet_latency_ns")

    for ax in (top, bottom):
        ax.plot(
            xpos,
            stalls,
            color=COLORS["xbar"],
            marker="o",
            linewidth=2.3,
            markersize=6,
            label="Xbar stalls",
        )
        ax.plot(
            xpos,
            avg,
            color=COLORS["avg"],
            marker="o",
            linewidth=2.3,
            markersize=6,
            label="Avg packet latency",
        )
        ax.plot(
            xpos,
            p99,
            color=COLORS["p99"],
            marker="o",
            linewidth=2.3,
            markersize=6,
            label="P99 packet latency",
        )
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        hide_offset_text(ax)

    bottom.set_ylim(0, 260)
    top.set_ylim(1800, max(stalls) * 1.12)
    top.spines["bottom"].set_visible(False)
    bottom.spines["top"].set_visible(False)
    top.tick_params(labelbottom=False)

    bottom.set_xticks(xpos)
    bottom.set_xticklabels(labels)
    bottom.set_xlabel("GEMM dimension (M = K, N = 64)")
    bottom.set_ylabel("Value")
    top.set_title("NoC Analysis")
    top.legend(loc="upper left")
    bottom.set_yticks([50, 100, 150, 200])
    top.set_yticks([2000, 4000, 8000, 12000])
    top.ticklabel_format(style="plain", axis="y")
    bottom.ticklabel_format(style="plain", axis="y")


def plot_memory(fig, gs, xpos, labels, rows):
    top = fig.add_subplot(gs[0])
    bottom = fig.add_subplot(gs[1], sharex=top)
    series = [
        (key, label, color, values(rows, key)) for key, label, color in MEMORY_FIELDS
    ]

    for ax in (top, bottom):
        for _, label, color, y in series:
            ax.plot(
                xpos,
                y,
                color=color,
                marker="o",
                linewidth=2.3,
                markersize=6,
                label=label,
            )
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        hide_offset_text(ax)

    bottom.set_ylim(0, 24)
    p99 = series[-1][3]
    top.set_ylim(60, max(p99) * 1.12)
    top.spines["bottom"].set_visible(False)
    bottom.spines["top"].set_visible(False)
    top.tick_params(labelbottom=False)

    bottom.set_xticks(xpos)
    bottom.set_xticklabels(labels)
    bottom.set_xlabel("GEMM dimension (M = K, N = 64)")
    bottom.set_ylabel("Cycles")
    top.set_title("Memory Analysis")
    top.legend(loc="upper left")
    bottom.set_yticks([0, 5, 10, 15, 20])
    top.set_yticks([70, 80, 90, 100, 110])
    top.ticklabel_format(style="plain", axis="y")
    bottom.ticklabel_format(style="plain", axis="y")


def plot_memory_clean(ax, xpos, labels, rows, x_label):
    ax2 = ax.twinx()
    mem_cache = {}
    series = [
        (key, label, color, values(rows, key)) for key, label, color in MEMORY_FIELDS
    ]
    for _, label, color, y in series:
        ax.plot(
            xpos,
            y,
            color=color,
            marker="o",
            linewidth=2.3,
            markersize=6,
            label=label,
        )

    utilization_pct = []
    for row in rows:
        hbm_util = parse_float(row, "hbm_utilization_pct")
        if hbm_util is not None:
            utilization_pct.append(hbm_util)
            continue

        avg_bw = get_memory_metric_strict(row, "hbm_avg_bandwidth", mem_cache)
        peak_bw = get_memory_metric_strict(row, "hbm_peak_bandwidth", mem_cache)
        if avg_bw is None:
            avg_bw = parse_float(row, "hbm_avg_bandwidth")
        if peak_bw is None:
            peak_bw = parse_float(row, "hbm_peak_bandwidth")

        if avg_bw is not None and peak_bw is not None and peak_bw > 0:
            util = 100.0 * float(avg_bw) / float(peak_bw)
        elif avg_bw is not None and float(avg_bw) <= 1.0:
            util = 100.0 * float(avg_bw)
        else:
            util = float("nan")
        utilization_pct.append(util)

    ax2.plot(
        xpos,
        utilization_pct,
        color=COLORS["util"],
        marker="s",
        linewidth=2.0,
        markersize=5,
        linestyle="-.",
        label="HBM utilization (%)",
    )

    ax.set_title("Memory Analysis")
    base_axis(ax, xpos, labels, "Cycles", x_label, sci=False)
    ax2.set_ylabel("HBM utilization (%)", color=COLORS["util"])
    ax2.tick_params(axis="y", colors=COLORS["util"])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(COLORS["util"])
    finite = [v for v in utilization_pct if v == v]
    util_max = max(finite) if finite else 100.0
    ax2.set_ylim(0, max(100.0, util_max * 1.10))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")


def plot_pressure(ax, xpos, labels, rows, x_label):
    ax2 = ax.twinx()
    retries = values(rows, "dma_timeout_retry_sum")
    poll = values(rows, "exec_poll_iters_mean")

    ax.plot(
        xpos,
        retries,
        color=COLORS["retry"],
        marker="o",
        linewidth=2.3,
        markersize=6,
        label="DMA retries",
    )
    ax2.plot(
        xpos,
        poll,
        color=COLORS["poll"],
        marker="o",
        linewidth=2.3,
        markersize=6,
        label="Poll iterations",
    )

    ax.set_title("Pressure Indicators")
    base_axis(ax, xpos, labels, "DMA retries", x_label, sci=False)
    ax.set_ylabel("DMA retries", color=COLORS["retry"])
    ax2.set_ylabel("Poll iterations", color=COLORS["poll"])
    ax2.spines["top"].set_visible(False)
    ax.tick_params(axis="y", colors=COLORS["retry"])
    ax.spines["left"].set_color(COLORS["retry"])
    ax2.tick_params(axis="y", colors=COLORS["poll"])
    ax2.spines["right"].set_color(COLORS["poll"])
    lines = ax.get_lines() + ax2.get_lines()
    labels2 = [line.get_label() for line in lines]
    ax.legend(lines, labels2, ncol=2, loc="upper left")


def save_single_plot(output_dir: Path, name: str, x_values, rows, x_label):
    labels = [str(v) for v in x_values]
    xpos = list(range(len(x_values)))

    if name == "memory":
        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        plot_memory_clean(ax, xpos, labels, rows, x_label)
        out = output_dir / "dim_sweep_memory.png"
        fig.savefig(out, dpi=240)
        plt.close(fig)
        return out

    if name == "noc":
        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        plot_noc_clean(ax, xpos, labels, rows, x_label)
        out = output_dir / "dim_sweep_noc.png"
        fig.savefig(out, dpi=240)
        plt.close(fig)
        return out

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    if name == "exec":
        plot_exec(ax, xpos, labels, rows, x_label)
    elif name == "pressure":
        plot_pressure(ax, xpos, labels, rows, x_label)
    fig.tight_layout()
    out = output_dir / f"dim_sweep_{name}.png"
    fig.savefig(out, dpi=240)
    plt.close(fig)
    return out


def save_dashboard(output_dir: Path, x_values, rows, x_label):
    labels = [str(v) for v in x_values]
    xpos = list(range(len(x_values)))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_exec(axes[0, 0], xpos, labels, rows, x_label)
    plot_noc_clean(axes[0, 1], xpos, labels, rows, x_label)
    plot_memory_clean(axes[1, 0], xpos, labels, rows, x_label)

    plot_pressure(axes[1, 1], xpos, labels, rows, x_label)
    fig.tight_layout()
    out = output_dir / "dim_sweep_dashboard.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description="Plot dimension sweep CSV for课题一")
    parser.add_argument("--input", required=True, help="Input dim_sweep CSV")
    parser.add_argument("--output-dir", required=True, help="Output plot directory")
    parser.add_argument("--x-field", default="gemm_m", help="CSV field for x-axis sorting/labels")
    parser.add_argument("--x-label", default="GEMM dimension (M = K, N = 64)", help="X-axis label")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_style()
    rows = load_rows(input_path, args.x_field)
    x_values = [int(row[args.x_field]) for row in rows]

    outputs = [save_dashboard(output_dir, x_values, rows, args.x_label)]
    for name in ["exec", "noc", "memory", "pressure"]:
        outputs.append(save_single_plot(output_dir, name, x_values, rows, args.x_label))

    print("[OK] wrote plots:")
    for out in outputs:
        print(f"  {out}")


if __name__ == "__main__":
    main()
