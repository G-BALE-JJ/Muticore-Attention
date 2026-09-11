#!/usr/bin/env python3
import argparse
import csv
import math
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


COLORS = {
    "base": "#4e79a7",
    "opt": "#f28e2b",
    "grid": "#e5e7eb",
    "spine": "#d1d5db",
    "text": "#1f2937",
}

REPO_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_STATS_ROOT = REPO_ROOT / "src/sst/elements/golem/tests/artifacts/stats"
TESTS_DIR = Path(__file__).resolve().parents[1]


def setup_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": COLORS["spine"],
            "axes.linewidth": 0.8,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.family": "DejaVu Sans",
            "savefig.bbox": "tight",
        }
    )


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())


def _maybe_num(text):
    if text is None:
        return None
    s = str(text).strip()
    if s == "":
        return None
    try:
        if any(ch in s for ch in [".", "e", "E"]):
            return float(s)
        return int(s)
    except ValueError:
        return s


def _same_value(a, b):
    av = _maybe_num(a)
    bv = _maybe_num(b)
    if isinstance(av, float) or isinstance(bv, float):
        try:
            return math.isclose(float(av), float(bv), rel_tol=1e-9, abs_tol=1e-9)
        except Exception:
            return str(a) == str(b)
    return av == bv


def _extract_run_id(path: Path) -> str:
    m = re.search(r"(run_\d{8}_\d{6}_\d+)", path.name)
    return m.group(1) if m else path.name


def short_run_id(run_id: str) -> str:
    m = re.match(r"run_(\d{8})_(\d{6})_(\d+)", run_id)
    if not m:
        return run_id
    return f"{m.group(2)}_{m.group(3)}"


def load_run_summary_index(run_summary_path: Path):
    out = {}
    if not run_summary_path.exists():
        return out
    with run_summary_path.open(newline="") as f:
        for row in csv.DictReader(f):
            log_file = row.get("log_file", "")
            m = re.search(r"(run_\d{8}_\d{6}_\d+)", log_file)
            if m:
                out[m.group(1)] = row
    return out


def _compact_key_name(key: str) -> str:
    aliases = {
        "overlap": "ov",
        "dim": "dim",
        "gemm_m": "M",
        "gemm_n": "N",
        "gemm_k": "K",
        "block_m": "BM",
        "block_n": "BN",
        "block_k": "BK",
        "num_cores": "cores",
        "gemm_cores": "gcores",
        "num_mem_nodes": "mem",
        "wcp_prefetch_windows": "wcpWin",
        "submit_batch_size": "batch",
        "dma_max_inflight": "inflight",
        "dma_retry_ticks": "retry",
        "dma_burst_bytes": "burst",
        "dma_stagger_cycles": "stagger",
        "group_max_inflight_per_node": "grpNode",
        "ctrl_overlap_ab": "ovAB",
        "noc_link_bw": "noc",
        "noc_xbar_bw": "xbar",
        "noc_flit_size": "flit",
        "dirctrl_highlink_bw": "dirHi",
    }
    return aliases.get(key, key)


def build_feature_label(row, fallback_name: str):
    if not row:
        return fallback_name
    ordered_keys = [
        "overlap",
        "gemm_m",
        "gemm_n",
        "gemm_k",
        "block_m",
        "block_n",
        "block_k",
        "num_cores",
        "num_mem_nodes",
        "wcp_prefetch_windows",
        "submit_batch_size",
        "dma_max_inflight",
        "dma_retry_ticks",
        "dma_burst_bytes",
        "dma_stagger_cycles",
        "group_max_inflight_per_node",
        "ctrl_overlap_ab",
        "noc_link_bw",
        "noc_xbar_bw",
        "noc_flit_size",
        "dirctrl_highlink_bw",
    ]
    parts = []
    for key in ordered_keys:
        val = row.get(key, "")
        if val != "":
            parts.append(f"{_compact_key_name(key)}={val}")
    return ", ".join(parts) if parts else fallback_name


def build_diff_labels(base_row, opt_row, base_name: str, opt_name: str):
    if not base_row or not opt_row:
        return base_name, opt_name
    ordered_keys = [
        "overlap",
        "dim",
        "gemm_m",
        "gemm_n",
        "gemm_k",
        "block_m",
        "block_n",
        "block_k",
        "num_cores",
        "gemm_cores",
        "num_mem_nodes",
        "wcp_prefetch_windows",
        "submit_batch_size",
        "dma_max_inflight",
        "dma_retry_ticks",
        "dma_burst_bytes",
        "dma_stagger_cycles",
        "group_max_inflight_per_node",
        "ctrl_overlap_ab",
        "noc_link_bw",
        "noc_xbar_bw",
        "noc_flit_size",
        "dirctrl_highlink_bw",
    ]
    diffs = [
        k
        for k in ordered_keys
        if not _same_value(base_row.get(k, ""), opt_row.get(k, ""))
    ]
    if not diffs:
        return base_name, opt_name
    base_parts = [f"{_compact_key_name(k)}={base_row.get(k, '')}" for k in diffs]
    opt_parts = [f"{_compact_key_name(k)}={opt_row.get(k, '')}" for k in diffs]
    return ", ".join(base_parts), ", ".join(opt_parts)


def build_display_labels(base_row, opt_row, base_run_id: str, opt_run_id: str):
    base_full = build_feature_label(base_row, base_run_id)
    opt_full = build_feature_label(opt_row, opt_run_id)
    base_diff, opt_diff = build_diff_labels(base_row, opt_row, base_full, opt_full)
    base_short = short_run_id(base_run_id)
    opt_short = short_run_id(opt_run_id)
    if base_diff != base_full or opt_diff != opt_full:
        return f"{base_short} | {base_diff}", f"{opt_short} | {opt_diff}"
    return f"{base_short} | matched-config", f"{opt_short} | matched-config"


def build_title_suffix(base_row, opt_row):
    if not base_row or not opt_row:
        return ""
    ordered_keys = [
        "overlap",
        "dim",
        "gemm_m",
        "gemm_n",
        "gemm_k",
        "block_m",
        "block_n",
        "block_k",
        "num_cores",
        "gemm_cores",
        "num_mem_nodes",
        "wcp_prefetch_windows",
        "submit_batch_size",
        "dma_max_inflight",
        "dma_retry_ticks",
        "dma_burst_bytes",
        "dma_stagger_cycles",
        "group_max_inflight_per_node",
        "ctrl_overlap_ab",
        "noc_link_bw",
        "noc_xbar_bw",
        "noc_flit_size",
        "dirctrl_highlink_bw",
    ]
    diffs = [
        k
        for k in ordered_keys
        if not _same_value(base_row.get(k, ""), opt_row.get(k, ""))
    ]
    if diffs:
        chunks = [
            f"{_compact_key_name(k)}: {base_row.get(k, '')} -> {opt_row.get(k, '')}"
            for k in diffs
        ]
        return "Changed: " + "; ".join(chunks)
    stable = [
        "gemm_m",
        "gemm_n",
        "gemm_k",
        "block_m",
        "block_n",
        "block_k",
        "wcp_prefetch_windows",
        "submit_batch_size",
        "dma_retry_ticks",
        "dma_burst_bytes",
        "group_max_inflight_per_node",
        "ctrl_overlap_ab",
        "noc_link_bw",
        "dirctrl_highlink_bw",
    ]
    parts = [
        f"{_compact_key_name(k)}={base_row.get(k, '')}"
        for k in stable
        if base_row.get(k, "") != ""
    ]
    return "Matched config: " + ", ".join(parts[:8])


def parse_metric_csv(path: Path):
    out = {}
    if not path.exists():
        return out
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return out
    for row in rows:
        metric = row.get("metric", "")
        if not metric:
            continue
        if "value" in row:
            value = row.get("value", "")
        else:
            value = row.get("mean", "")
        if value in ("", None):
            continue
        try:
            out[metric] = float(value)
        except ValueError:
            out[metric] = value
    return out


def pick(metrics, key):
    v = metrics.get(key)
    if v is None:
        return float("nan")
    try:
        return float(v)
    except Exception:
        return float("nan")


def pick_text(metrics, key):
    v = metrics.get(key)
    return "" if v is None else str(v)


def pick0(metrics, key):
    v = metrics.get(key)
    if v is None:
        return 0.0
    try:
        return float(v)
    except Exception:
        return 0.0


def exec_breakdown(metrics):
    current_keys = [
        "compute_active_time",
        "prefetch_wait_time",
        "writeback_wait_time",
        "control_other_time",
        "total_cycles",
    ]
    if all(k in metrics for k in current_keys):
        return {
            "array_compute_active_time": pick(metrics, "compute_active_time"),
            "array_load_active_time": 0.0,
            "data_movement_time": pick(metrics, "prefetch_wait_time")
            + pick(metrics, "writeback_wait_time"),
            "control_overhead_time": pick(metrics, "control_other_time"),
            "unclassified_time": 0.0,
            "total": pick(metrics, "total_cycles"),
        }

    new_keys = [
        "array_compute_active_time",
        "array_load_active_time",
        "data_movement_time",
        "control_overhead_time",
        "unclassified_time",
        "total_cycles",
    ]
    if all(k in metrics for k in new_keys):
        return {
            "array_compute_active_time": pick(metrics, "array_compute_active_time"),
            "array_load_active_time": pick(metrics, "array_load_active_time"),
            "data_movement_time": pick(metrics, "data_movement_time"),
            "control_overhead_time": pick(metrics, "control_overhead_time"),
            "unclassified_time": pick(metrics, "unclassified_time"),
            "total": pick(metrics, "total_cycles"),
        }

    total = pick(metrics, "total")
    compute = pick(metrics, "compute")
    dma_issue = pick(metrics, "dma_issue")
    dma_wait = pick(metrics, "dma_wait")
    sched_protocol = pick(metrics, "sched_protocol")
    group_wait = pick(metrics, "group_wait")
    c_store = pick0(metrics, "c_store")
    overlap_issue = pick0(metrics, "overlap_issue")
    overlap_wait = pick0(metrics, "overlap_wait")
    task_desc = pick0(metrics, "task_desc")
    nloop = pick0(metrics, "nloop")
    submit_pack = pick0(metrics, "submit_pack")
    finish_publish = pick0(metrics, "finish_publish")
    issue_block_q = pick0(metrics, "issue_block_q")
    issue_write = pick0(metrics, "issue_write")

    if any(
        math.isnan(v)
        for v in [total, compute, dma_issue, dma_wait, sched_protocol, group_wait]
    ):
        return {
            "compute": float("nan"),
            "dma_issue_non_overlap": float("nan"),
            "token_issue_non_overlap": float("nan"),
            "overlap_issue": float("nan"),
            "dma_wait_non_overlap": float("nan"),
            "overlap_wait": float("nan"),
            "sched_protocol": float("nan"),
            "group_wait": float("nan"),
            "c_store": float("nan"),
            "task_desc": float("nan"),
            "nloop": float("nan"),
            "submit_pack": float("nan"),
            "finish_publish": float("nan"),
            "other_uncat": float("nan"),
            "other_total": float("nan"),
            "total": float("nan"),
        }

    # Support both accounting styles:
    # 1) mutually-exclusive (new): dma_total = dma_issue + dma_wait + overlap_issue + overlap_wait
    # 2) legacy (overlap also included in dma_issue/dma_wait): dma_total = dma_issue + dma_wait
    dma_total = pick(metrics, "dma_total")
    full_dma = dma_issue + dma_wait + overlap_issue + overlap_wait
    if not math.isnan(dma_total) and math.isclose(
        dma_total, full_dma, rel_tol=1e-6, abs_tol=2.0
    ):
        issue_non_overlap = dma_issue
        wait_non_overlap = dma_wait
    else:
        issue_non_overlap = max(0.0, dma_issue - overlap_issue)
        wait_non_overlap = max(0.0, dma_wait - overlap_wait)
    token_issue_non_overlap = max(0.0, issue_non_overlap - issue_block_q - issue_write)
    classified_wo_group = (
        compute
        + issue_non_overlap
        + wait_non_overlap
        + sched_protocol
        + task_desc
        + nloop
        + submit_pack
    )

    best_residual = None
    best_classified = classified_wo_group
    for include_group_wait in (False, True):
        classified = classified_wo_group
        if include_group_wait:
            classified += group_wait
        residual = abs(total - classified)
        if best_residual is None or residual < best_residual:
            best_residual = residual
            best_classified = classified
    other_uncat = total - best_classified

    return {
        "compute": compute,
        "dma_issue_non_overlap": issue_non_overlap,
        "token_issue_non_overlap": token_issue_non_overlap,
        "overlap_issue": overlap_issue,
        "dma_wait_non_overlap": wait_non_overlap,
        "overlap_wait": overlap_wait,
        "sched_protocol": sched_protocol,
        "group_wait": group_wait,
        "c_store": c_store,
        "task_desc": task_desc,
        "nloop": nloop,
        "submit_pack": submit_pack,
        "finish_publish": finish_publish,
        "other_uncat": other_uncat,
        "other_total": c_store
        + task_desc
        + nloop
        + submit_pack
        + finish_publish
        + other_uncat,
        "total": total,
    }


def style_axis(ax, title, subtitle=None):
    ax.set_title(title, pad=26)
    if subtitle:
        ax.text(
            0.0,
            1.10,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color=COLORS["text"],
        )
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _fmt_bar(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    av = abs(v)
    if av >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if av >= 1_000:
        return f"{v / 1_000:.1f}k"
    return f"{v:.1f}"


def add_bar_labels(ax, container):
    labels = [_fmt_bar(p.get_height()) for p in container.patches]
    ax.bar_label(container, labels=labels, padding=2, fontsize=8, rotation=0)


def add_headroom(ax, ratio=0.22):
    low, high = ax.get_ylim()
    if high <= 0:
        return
    if low >= 0:
        ax.set_ylim(0, high * (1.0 + ratio))
    else:
        span = high - low
        ax.set_ylim(low, high + span * ratio)


def plot_execution_dual_axis(
    out_path: Path,
    base_exec,
    opt_exec,
    base_label,
    opt_label,
    title_suffix=None,
    clean: bool = False,
):
    base_bd = exec_breakdown(base_exec)
    opt_bd = exec_breakdown(opt_exec)
    if clean:
        left_labels = [
            "array_compute",
            "array_load",
            "data_move",
            "control",
            "unclassified",
        ]
        right_labels = ["total"]
        base_left = [
            base_bd.get("array_compute_active_time", float("nan")),
            base_bd.get("array_load_active_time", float("nan")),
            base_bd.get("data_movement_time", float("nan")),
            base_bd.get("control_overhead_time", float("nan")),
            base_bd.get("unclassified_time", float("nan")),
        ]
        opt_left = [
            opt_bd.get("array_compute_active_time", float("nan")),
            opt_bd.get("array_load_active_time", float("nan")),
            opt_bd.get("data_movement_time", float("nan")),
            opt_bd.get("control_overhead_time", float("nan")),
            opt_bd.get("unclassified_time", float("nan")),
        ]
        base_right = [sum(v for v in base_left if not math.isnan(v))]
        opt_right = [sum(v for v in opt_left if not math.isnan(v))]
        chart_title = "Execution Breakdown Comparison"
    else:
        left_labels = [
            "array_compute",
            "array_load",
            "data_move",
            "control",
            "unclassified",
        ]
        right_labels = ["total"]
        base_left = [
            base_bd.get("array_compute_active_time", float("nan")),
            base_bd.get("array_load_active_time", float("nan")),
            base_bd.get("data_movement_time", float("nan")),
            base_bd.get("control_overhead_time", float("nan")),
            base_bd.get("unclassified_time", float("nan")),
        ]
        opt_left = [
            opt_bd.get("array_compute_active_time", float("nan")),
            opt_bd.get("array_load_active_time", float("nan")),
            opt_bd.get("data_movement_time", float("nan")),
            opt_bd.get("control_overhead_time", float("nan")),
            opt_bd.get("unclassified_time", float("nan")),
        ]
        base_right = [pick(base_exec, "total_cycles")]
        opt_right = [pick(opt_exec, "total_cycles")]
        chart_title = "Execution Breakdown + Total Comparison"

    left_x = list(range(len(left_labels)))
    right_x = [len(left_labels) + i for i in range(len(right_labels))]
    width = 0.34

    fig, ax = plt.subplots(figsize=(13.2, 5.2))
    b1 = ax.bar(
        [i - width / 2 for i in left_x],
        base_left,
        width=width,
        color=COLORS["base"],
        alpha=0.85,
        label=f"{base_label} (left)",
    )
    b2 = ax.bar(
        [i + width / 2 for i in left_x],
        opt_left,
        width=width,
        color=COLORS["opt"],
        alpha=0.85,
        label=f"{opt_label} (left)",
    )
    ax.set_ylabel("Cycles")
    sci_fmt = ScalarFormatter(useMathText=True)
    sci_fmt.set_scientific(True)
    sci_fmt.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(sci_fmt)
    style_axis(ax, chart_title, title_suffix)

    ax2 = ax.twinx()
    b3 = ax2.bar(
        [i - width / 2 for i in right_x],
        base_right,
        width=width,
        color="#1f2937",
        alpha=0.62,
        label=f"{base_label} (right)",
    )
    b4 = ax2.bar(
        [i + width / 2 for i in right_x],
        opt_right,
        width=width,
        color="#9c755f",
        alpha=0.62,
        label=f"{opt_label} (right)",
    )
    ax2.set_ylabel("Total cycles")
    ax2.spines["top"].set_visible(False)

    add_bar_labels(ax, b1)
    add_bar_labels(ax, b2)
    add_bar_labels(ax2, b3)
    add_bar_labels(ax2, b4)
    add_headroom(ax)
    add_headroom(ax2)

    all_x = left_x + right_x
    all_labels = left_labels + right_labels
    ax.set_xlim(-0.6, right_x[-1] + 0.6)
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, rotation=20, ha="right")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", ncol=2)
    fig.subplots_adjust(top=0.82)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_noc_dual_axis(
    out_path: Path,
    base_noc,
    opt_noc,
    base_nocl,
    opt_nocl,
    base_label,
    opt_label,
    title_suffix=None,
):
    left_labels = ["xbar_stalls"]
    right_labels = ["avg_pkt_lat_ns", "p99_pkt_lat_ns", "max_port_util_pct"]
    left_x = [0]
    right_x = [1, 2, 3]
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.8, 4.9))
    xbar_base = [pick(base_noc, "total_xbar_stalls")]
    xbar_opt = [pick(opt_noc, "total_xbar_stalls")]
    b1 = ax.bar(
        [i - width / 2 for i in left_x],
        xbar_base,
        width=width,
        color=COLORS["base"],
        alpha=0.82,
        label=f"{base_label} (left)",
    )
    b2 = ax.bar(
        [i + width / 2 for i in left_x],
        xbar_opt,
        width=width,
        color=COLORS["opt"],
        alpha=0.82,
        label=f"{opt_label} (left)",
    )
    ax.set_ylabel("Xbar stalls")
    style_axis(ax, "Interconnect Comparison", title_suffix)

    ax2 = ax.twinx()
    right_base = [
        pick(base_nocl, "noc_avg_packet_latency_ns"),
        pick(base_nocl, "noc_p99_packet_latency_ns"),
        pick(base_noc, "max_port_util_pct"),
    ]
    right_opt = [
        pick(opt_nocl, "noc_avg_packet_latency_ns"),
        pick(opt_nocl, "noc_p99_packet_latency_ns"),
        pick(opt_noc, "max_port_util_pct"),
    ]
    b3 = ax2.bar(
        [i - width / 2 for i in right_x],
        right_base,
        width=width,
        color="#59a14f",
        alpha=0.62,
        label=f"{base_label} (right)",
    )
    b4 = ax2.bar(
        [i + width / 2 for i in right_x],
        right_opt,
        width=width,
        color="#b07aa1",
        alpha=0.62,
        label=f"{opt_label} (right)",
    )
    ax2.set_ylabel("Latency (ns) / Util (%)")
    ax2.spines["top"].set_visible(False)

    add_bar_labels(ax, b1)
    add_bar_labels(ax, b2)
    add_bar_labels(ax2, b3)
    add_bar_labels(ax2, b4)
    add_headroom(ax)
    add_headroom(ax2)

    all_x = left_x + right_x
    all_labels = left_labels + right_labels
    ax.set_xlim(-0.6, right_x[-1] + 0.6)
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, rotation=15, ha="right")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", ncol=2)
    fig.subplots_adjust(top=0.82)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_memory_dual_axis(
    out_path: Path,
    base_mem,
    opt_mem,
    base_mq,
    opt_mq,
    base_label,
    opt_label,
    title_suffix=None,
):
    left_labels = ["mem_avg_read_lat", "backend_read_avg"]
    right_labels = ["tail_ge_100_pct", "bw_imbalance"]
    left_x = [0, 1]
    right_x = [2, 3]
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.8, 4.9))

    left_base = [
        pick(base_mem, "mem_avg_read_latency_cycles"),
        pick(base_mq, "memory_backend_read_latency_avg_cycles"),
    ]
    left_opt = [
        pick(opt_mem, "mem_avg_read_latency_cycles"),
        pick(opt_mq, "memory_backend_read_latency_avg_cycles"),
    ]
    b1 = ax.bar(
        [i - width / 2 for i in left_x],
        left_base,
        width=width,
        color=COLORS["base"],
        alpha=0.82,
        label=f"{base_label} (left)",
    )
    b2 = ax.bar(
        [i + width / 2 for i in left_x],
        left_opt,
        width=width,
        color=COLORS["opt"],
        alpha=0.82,
        label=f"{opt_label} (left)",
    )
    ax.set_ylabel("Latency (cycles)")
    style_axis(ax, "Memory System Comparison", title_suffix)

    ax2 = ax.twinx()
    right_base = [
        pick(base_mem, "mem_read_tail_ge_100_pct"),
        pick(base_mem, "hbm_channel_bandwidth_imbalance"),
    ]
    right_opt = [
        pick(opt_mem, "mem_read_tail_ge_100_pct"),
        pick(opt_mem, "hbm_channel_bandwidth_imbalance"),
    ]
    b3 = ax2.bar(
        [i - width / 2 for i in right_x],
        right_base,
        width=width,
        color="#e15759",
        alpha=0.62,
        label=f"{base_label} (right)",
    )
    b4 = ax2.bar(
        [i + width / 2 for i in right_x],
        right_opt,
        width=width,
        color="#edc948",
        alpha=0.62,
        label=f"{opt_label} (right)",
    )
    ax2.set_ylabel("Tail/Imbalance/Queue")
    ax2.spines["top"].set_visible(False)

    add_bar_labels(ax, b1)
    add_bar_labels(ax, b2)
    add_bar_labels(ax2, b3)
    add_bar_labels(ax2, b4)
    add_headroom(ax)
    add_headroom(ax2)

    all_x = left_x + right_x
    all_labels = left_labels + right_labels
    ax.set_xlim(-0.6, right_x[-1] + 0.6)
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, rotation=15, ha="right")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", ncol=2)
    fig.subplots_adjust(top=0.82)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def pct_delta(base, opt):
    if math.isnan(base) or math.isnan(opt) or base == 0:
        return "NA"
    return f"{(opt / base - 1.0) * 100.0:+.2f}%"


def _as_float_or_nan(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def _fmt_summary_value(v):
    vf = _as_float_or_nan(v)
    if not math.isnan(vf):
        return f"{vf:.6f}"
    return "" if v is None else str(v)


def write_summary_csv(path: Path, rows):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "baseline", "optimized", "delta", "delta_pct"])
        for metric, b, o in rows:
            bf = _as_float_or_nan(b)
            of = _as_float_or_nan(o)
            d = "" if (math.isnan(bf) or math.isnan(of)) else f"{of - bf:.6f}"
            w.writerow(
                [
                    metric,
                    _fmt_summary_value(b),
                    _fmt_summary_value(o),
                    d,
                    pct_delta(bf, of),
                ]
            )


def main():
    p = argparse.ArgumentParser(
        description="Plot baseline vs optimized run comparison."
    )
    p.add_argument("--base-run-dir", required=True)
    p.add_argument("--opt-run-dir", required=True)
    p.add_argument("--base-label", default="baseline")
    p.add_argument("--opt-label", default="optimized")
    p.add_argument(
        "--base-feature-label",
        default="",
        help="Manual legend label for the base run, used to override auto-detected feature text.",
    )
    p.add_argument(
        "--opt-feature-label",
        default="",
        help="Manual legend label for the optimized run, used to override auto-detected feature text.",
    )
    p.add_argument(
        "--title-suffix",
        default="",
        help="Manual subtitle describing the experiment change; overrides auto-generated change text.",
    )
    p.add_argument(
        "--out-dir",
        default=str(DEFAULT_STATS_ROOT / "analysis/comparison"),
    )
    p.add_argument(
        "--run-summary",
        default=str(DEFAULT_STATS_ROOT / "run_summary.csv"),
    )
    args = p.parse_args()

    setup_style()

    base_dir = Path(args.base_run_dir)
    opt_dir = Path(args.opt_run_dir)
    out_root = Path(args.out_dir)
    tag = f"{safe_name(base_dir.name)}_vs_{safe_name(opt_dir.name)}"
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    run_summary_idx = load_run_summary_index(Path(args.run_summary))
    base_run_id = _extract_run_id(base_dir)
    opt_run_id = _extract_run_id(opt_dir)
    base_row = run_summary_idx.get(base_run_id)
    opt_row = run_summary_idx.get(opt_run_id)

    auto_base_label, auto_opt_label = build_display_labels(
        base_row, opt_row, base_run_id, opt_run_id
    )
    auto_title_suffix = build_title_suffix(base_row, opt_row)
    base_label = args.base_label if args.base_label != "baseline" else auto_base_label
    opt_label = args.opt_label if args.opt_label != "optimized" else auto_opt_label
    if args.base_feature_label.strip():
        base_label = f"{short_run_id(base_run_id)} | {args.base_feature_label.strip()}"
    if args.opt_feature_label.strip():
        opt_label = f"{short_run_id(opt_run_id)} | {args.opt_feature_label.strip()}"
    title_suffix = (
        args.title_suffix.strip() if args.title_suffix.strip() else auto_title_suffix
    )

    base_exec = parse_metric_csv(base_dir / "execution_summary.csv")
    opt_exec = parse_metric_csv(opt_dir / "execution_summary.csv")
    base_noc = parse_metric_csv(base_dir / "noc_summary.csv")
    opt_noc = parse_metric_csv(opt_dir / "noc_summary.csv")
    base_mem = parse_metric_csv(base_dir / "memory_summary.csv")
    opt_mem = parse_metric_csv(opt_dir / "memory_summary.csv")
    base_mq = parse_metric_csv(base_dir / "memory_queue_summary.csv")
    opt_mq = parse_metric_csv(opt_dir / "memory_queue_summary.csv")
    base_nocl = parse_metric_csv(base_dir / "noc_latency_summary.csv")
    opt_nocl = parse_metric_csv(opt_dir / "noc_latency_summary.csv")
    base_causal = parse_metric_csv(base_dir / "submit_ready_causal_summary.csv")
    opt_causal = parse_metric_csv(opt_dir / "submit_ready_causal_summary.csv")
    base_hotspot = parse_metric_csv(base_dir / "noc_hotspot_summary.csv")
    opt_hotspot = parse_metric_csv(opt_dir / "noc_hotspot_summary.csv")

    plot_execution_dual_axis(
        out_dir
        / f"execution_breakdown_comparison_{safe_name(base_label)}_vs_{safe_name(opt_label)}.png",
        base_exec,
        opt_exec,
        base_label,
        opt_label,
        title_suffix,
    )

    plot_execution_dual_axis(
        out_dir
        / f"throughput_utilization_comparison_{safe_name(base_label)}_vs_{safe_name(opt_label)}.png",
        base_exec,
        opt_exec,
        base_label,
        opt_label,
        title_suffix,
        clean=True,
    )

    plot_noc_dual_axis(
        out_dir
        / f"interconnect_comparison_{safe_name(base_label)}_vs_{safe_name(opt_label)}.png",
        base_noc,
        opt_noc,
        base_nocl,
        opt_nocl,
        base_label,
        opt_label,
        title_suffix,
    )

    plot_memory_dual_axis(
        out_dir
        / f"memory_system_comparison_{safe_name(base_label)}_vs_{safe_name(opt_label)}.png",
        base_mem,
        opt_mem,
        base_mq,
        opt_mq,
        base_label,
        opt_label,
        title_suffix,
    )

    # archive source csvs
    archive_dir = out_dir / "source_csv"
    archive_dir.mkdir(exist_ok=True)
    for src in [
        base_dir / "execution_summary.csv",
        base_dir / "noc_summary.csv",
        base_dir / "noc_latency_summary.csv",
        base_dir / "memory_summary.csv",
        base_dir / "memory_queue_summary.csv",
        base_dir / "submit_ready_causal_summary.csv",
        base_dir / "noc_hotspot_summary.csv",
        opt_dir / "execution_summary.csv",
        opt_dir / "noc_summary.csv",
        opt_dir / "noc_latency_summary.csv",
        opt_dir / "memory_summary.csv",
        opt_dir / "memory_queue_summary.csv",
        opt_dir / "submit_ready_causal_summary.csv",
        opt_dir / "noc_hotspot_summary.csv",
    ]:
        if src.exists():
            prefix = "base" if src.parts[-2] == base_dir.name else "opt"
            shutil.copy2(src, archive_dir / f"{prefix}_{src.name}")

    base_exec_bd = exec_breakdown(base_exec)
    opt_exec_bd = exec_breakdown(opt_exec)

    summary_rows = [
        (
            "execution_total_cycles",
            pick(base_exec, "total_cycles"),
            pick(opt_exec, "total_cycles"),
        ),
        (
            "avg_throughput_ops_per_cycle",
            pick(base_exec, "avg_throughput_ops_per_cycle"),
            pick(opt_exec, "avg_throughput_ops_per_cycle"),
        ),
        (
            "peak_throughput_ops_per_cycle",
            pick(base_exec, "peak_throughput_ops_per_cycle"),
            pick(opt_exec, "peak_throughput_ops_per_cycle"),
        ),
        (
            "array_utilization_pct",
            pick(base_exec, "array_utilization_pct"),
            pick(opt_exec, "array_utilization_pct"),
        ),
        (
            "execution_compute_active_time",
            pick(base_exec, "compute_active_time"),
            pick(opt_exec, "compute_active_time"),
        ),
        (
            "execution_compute_active_time_share_pct",
            pick(base_exec, "compute_active_time_share_pct"),
            pick(opt_exec, "compute_active_time_share_pct"),
        ),
        (
            "execution_prefetch_wait_time",
            pick(base_exec, "prefetch_wait_time"),
            pick(opt_exec, "prefetch_wait_time"),
        ),
        (
            "execution_prefetch_wait_time_share_pct",
            pick(base_exec, "prefetch_wait_time_share_pct"),
            pick(opt_exec, "prefetch_wait_time_share_pct"),
        ),
        (
            "execution_writeback_wait_time",
            pick(base_exec, "writeback_wait_time"),
            pick(opt_exec, "writeback_wait_time"),
        ),
        (
            "execution_writeback_wait_time_share_pct",
            pick(base_exec, "writeback_wait_time_share_pct"),
            pick(opt_exec, "writeback_wait_time_share_pct"),
        ),
        (
            "execution_control_other_time",
            pick(base_exec, "control_other_time"),
            pick(opt_exec, "control_other_time"),
        ),
        (
            "breakdown_array_compute_active_time",
            base_exec_bd.get("array_compute_active_time", float("nan")),
            opt_exec_bd.get("array_compute_active_time", float("nan")),
        ),
        (
            "breakdown_array_load_active_time",
            base_exec_bd.get("array_load_active_time", float("nan")),
            opt_exec_bd.get("array_load_active_time", float("nan")),
        ),
        (
            "breakdown_data_movement_time",
            base_exec_bd.get("data_movement_time", float("nan")),
            opt_exec_bd.get("data_movement_time", float("nan")),
        ),
        (
            "breakdown_control_overhead_time",
            base_exec_bd.get("control_overhead_time", float("nan")),
            opt_exec_bd.get("control_overhead_time", float("nan")),
        ),
        (
            "breakdown_unclassified_time",
            base_exec_bd.get("unclassified_time", float("nan")),
            opt_exec_bd.get("unclassified_time", float("nan")),
        ),
        (
            "causal_event_full_match_count",
            pick(base_causal, "event_full_match_count"),
            pick(opt_causal, "event_full_match_count"),
        ),
        (
            "causal_event_invalid_order_count",
            pick(base_causal, "event_invalid_order_count"),
            pick(opt_causal, "event_invalid_order_count"),
        ),
        (
            "causal_submit_to_issue_mat_mean_cycles",
            pick(base_causal, "causal_submit_to_issue_mat_mean_cycles"),
            pick(opt_causal, "causal_submit_to_issue_mat_mean_cycles"),
        ),
        (
            "causal_submit_to_issue_vec_mean_cycles",
            pick(base_causal, "causal_submit_to_issue_vec_mean_cycles"),
            pick(opt_causal, "causal_submit_to_issue_vec_mean_cycles"),
        ),
        (
            "causal_submit_to_issue_mat_p95_cycles",
            pick(base_causal, "causal_submit_to_issue_mat_p95_cycles"),
            pick(opt_causal, "causal_submit_to_issue_mat_p95_cycles"),
        ),
        (
            "causal_submit_to_issue_vec_p95_cycles",
            pick(base_causal, "causal_submit_to_issue_vec_p95_cycles"),
            pick(opt_causal, "causal_submit_to_issue_vec_p95_cycles"),
        ),
        (
            "causal_issue_to_pending_mat_mean_cycles",
            pick(base_causal, "causal_issue_to_pending_mat_mean_cycles"),
            pick(opt_causal, "causal_issue_to_pending_mat_mean_cycles"),
        ),
        (
            "causal_issue_to_pending_vec_mean_cycles",
            pick(base_causal, "causal_issue_to_pending_vec_mean_cycles"),
            pick(opt_causal, "causal_issue_to_pending_vec_mean_cycles"),
        ),
        (
            "causal_issue_to_pending_mat_p95_cycles",
            pick(base_causal, "causal_issue_to_pending_mat_p95_cycles"),
            pick(opt_causal, "causal_issue_to_pending_mat_p95_cycles"),
        ),
        (
            "causal_issue_to_pending_vec_p95_cycles",
            pick(base_causal, "causal_issue_to_pending_vec_p95_cycles"),
            pick(opt_causal, "causal_issue_to_pending_vec_p95_cycles"),
        ),
        (
            "causal_forward_to_memnic_mean_cycles",
            pick(base_causal, "causal_forward_to_memnic_mean_cycles"),
            pick(opt_causal, "causal_forward_to_memnic_mean_cycles"),
        ),
        (
            "causal_forward_to_memnic_p95_cycles",
            pick(base_causal, "causal_forward_to_memnic_p95_cycles"),
            pick(opt_causal, "causal_forward_to_memnic_p95_cycles"),
        ),
        (
            "causal_memory_service_mean_cycles",
            pick(base_causal, "causal_memory_service_mean_cycles"),
            pick(opt_causal, "causal_memory_service_mean_cycles"),
        ),
        (
            "causal_memory_service_p95_cycles",
            pick(base_causal, "causal_memory_service_p95_cycles"),
            pick(opt_causal, "causal_memory_service_p95_cycles"),
        ),
        (
            "causal_return_path_mat_mean_cycles",
            pick(base_causal, "causal_return_path_mat_mean_cycles"),
            pick(opt_causal, "causal_return_path_mat_mean_cycles"),
        ),
        (
            "causal_return_path_vec_mean_cycles",
            pick(base_causal, "causal_return_path_vec_mean_cycles"),
            pick(opt_causal, "causal_return_path_vec_mean_cycles"),
        ),
        (
            "causal_return_path_mat_p95_cycles",
            pick(base_causal, "causal_return_path_mat_p95_cycles"),
            pick(opt_causal, "causal_return_path_mat_p95_cycles"),
        ),
        (
            "causal_return_path_vec_p95_cycles",
            pick(base_causal, "causal_return_path_vec_p95_cycles"),
            pick(opt_causal, "causal_return_path_vec_p95_cycles"),
        ),
        (
            "causal_return_path_mat_mean_share_pct",
            pick(base_causal, "causal_return_path_mat_mean_share_pct"),
            pick(opt_causal, "causal_return_path_mat_mean_share_pct"),
        ),
        (
            "causal_return_path_vec_mean_share_pct",
            pick(base_causal, "causal_return_path_vec_mean_share_pct"),
            pick(opt_causal, "causal_return_path_vec_mean_share_pct"),
        ),
        (
            "noc_total_xbar_stalls",
            pick(base_hotspot, "total_xbar_stalls"),
            pick(opt_hotspot, "total_xbar_stalls"),
        ),
        (
            "noc_total_output_port_stalls",
            pick(base_hotspot, "total_output_port_stalls"),
            pick(opt_hotspot, "total_output_port_stalls"),
        ),
        (
            "noc_top1_router",
            pick_text(base_hotspot, "top1_router"),
            pick_text(opt_hotspot, "top1_router"),
        ),
        (
            "noc_top1_router_xbar_share_pct",
            pick(base_hotspot, "top1_router_xbar_share_pct"),
            pick(opt_hotspot, "top1_router_xbar_share_pct"),
        ),
        (
            "noc_top1_port_router",
            pick_text(base_hotspot, "top1_port_router"),
            pick_text(opt_hotspot, "top1_port_router"),
        ),
        (
            "noc_top1_port",
            pick_text(base_hotspot, "top1_port"),
            pick_text(opt_hotspot, "top1_port"),
        ),
        (
            "noc_top1_port_xbar_share_pct",
            pick(base_hotspot, "top1_port_xbar_share_pct"),
            pick(opt_hotspot, "top1_port_xbar_share_pct"),
        ),
        (
            "noc_avg_packet_latency_ns",
            pick(base_nocl, "noc_avg_packet_latency_ns"),
            pick(opt_nocl, "noc_avg_packet_latency_ns"),
        ),
        (
            "noc_p99_packet_latency_ns",
            pick(base_nocl, "noc_p99_packet_latency_ns"),
            pick(opt_nocl, "noc_p99_packet_latency_ns"),
        ),
        (
            "memory_avg_read_latency_cycles",
            pick(base_mem, "mem_avg_read_latency_cycles"),
            pick(opt_mem, "mem_avg_read_latency_cycles"),
        ),
        (
            "memory_p95_read_latency_bucket_cycles",
            pick(base_mem, "mem_p95_read_latency_bucket_cycles"),
            pick(opt_mem, "mem_p95_read_latency_bucket_cycles"),
        ),
        (
            "memory_backend_read_latency_p95_cycles",
            pick(base_mq, "memory_backend_read_latency_p95_cycles"),
            pick(opt_mq, "memory_backend_read_latency_p95_cycles"),
        ),
        (
            "memory_backend_read_latency_p99_cycles",
            pick(base_mq, "memory_backend_read_latency_p99_cycles"),
            pick(opt_mq, "memory_backend_read_latency_p99_cycles"),
        ),
        (
            "memory_queue_delay_p99_cycles",
            pick(base_mq, "memory_queue_delay_p99_cycles"),
            pick(opt_mq, "memory_queue_delay_p99_cycles"),
        ),
    ]

    debug_rows = [
        (
            "debug_runtime_compute",
            pick(base_exec, "compute"),
            pick(opt_exec, "compute"),
        ),
        (
            "debug_runtime_compute_submit",
            pick(base_exec, "compute_submit"),
            pick(opt_exec, "compute_submit"),
        ),
        (
            "debug_runtime_compute_wait",
            pick(base_exec, "compute_wait"),
            pick(opt_exec, "compute_wait"),
        ),
        (
            "debug_runtime_dma_total",
            pick(base_exec, "dma_total"),
            pick(opt_exec, "dma_total"),
        ),
        (
            "debug_sched_protocol_mean",
            pick(base_exec, "debug_sched_protocol_mean"),
            pick(opt_exec, "debug_sched_protocol_mean"),
        ),
        (
            "debug_group_wait_mean",
            pick(base_exec, "debug_group_wait_mean"),
            pick(opt_exec, "debug_group_wait_mean"),
        ),
        (
            "debug_runtime_loop_other",
            pick(base_exec, "nloop"),
            pick(opt_exec, "nloop"),
        ),
        (
            "noc_total_xbar_stalls",
            pick(base_noc, "total_xbar_stalls"),
            pick(opt_noc, "total_xbar_stalls"),
        ),
        (
            "noc_avg_packet_latency_ns",
            pick(base_nocl, "noc_avg_packet_latency_ns"),
            pick(opt_nocl, "noc_avg_packet_latency_ns"),
        ),
        (
            "memory_avg_read_latency_cycles",
            pick(base_mem, "mem_avg_read_latency_cycles"),
            pick(opt_mem, "mem_avg_read_latency_cycles"),
        ),
        (
            "memory_backend_read_latency_avg_cycles",
            pick(base_mq, "memory_backend_read_latency_avg_cycles"),
            pick(opt_mq, "memory_backend_read_latency_avg_cycles"),
        ),
    ]
    write_summary_csv(out_dir / "comparison_summary.csv", summary_rows)
    write_summary_csv(out_dir / "comparison_debug_summary.csv", debug_rows)

    with (out_dir / "comparison_metadata.txt").open("w") as f:
        f.write(f"base_run_id={base_run_id}\n")
        f.write(f"opt_run_id={opt_run_id}\n")
        f.write(f"base_label={base_label}\n")
        f.write(f"opt_label={opt_label}\n")
        if base_row:
            f.write(f"base_features={build_feature_label(base_row, base_run_id)}\n")
        if opt_row:
            f.write(f"opt_features={build_feature_label(opt_row, opt_run_id)}\n")

    print(f"[OK] comparison archived: {out_dir}")


if __name__ == "__main__":
    main()
