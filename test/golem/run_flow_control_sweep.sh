#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="${GOLEM_ARTIFACT_ROOT:-$SCRIPT_DIR/artifacts}"
SWEEP_ROOT="${GOLEM_SWEEP_ROOT:-$ARTIFACT_ROOT/stats/sweeps/flow_control_4x4_bk64}"
SWEEP_TAG="${GOLEM_SWEEP_TAG:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_RUN_DIR="${GOLEM_SWEEP_RUN_DIR:-$SWEEP_ROOT/run_$SWEEP_TAG}"
RUN_SUMMARY_CSV="$SWEEP_RUN_DIR/run_summary.csv"
STATUS_RAW_CSV="$SWEEP_RUN_DIR/flow_control_status_raw.csv"
SWEEP_CSV="$SWEEP_RUN_DIR/flow_control_sweep_4x4_bk64.csv"
PLOTS_DIR="$SWEEP_RUN_DIR/plots"

mkdir -p "$SWEEP_RUN_DIR" "$PLOTS_DIR"

GEMM_M="${GOLEM_SWEEP_GEMM_M:-1024}"
GEMM_N="${GOLEM_SWEEP_GEMM_N:-1024}"
GEMM_K="${GOLEM_SWEEP_GEMM_K:-1024}"
BLOCK_K="${GOLEM_SWEEP_GEMM_BLOCK_K:-64}"
RUN_TIMEOUT="${GOLEM_SWEEP_TIMEOUT:-300s}"
BASE_NODE_CREDITS="${GOLEM_FLOW_BASE_NODE_CHUNK_CREDITS:-84}"
BASE_WCP_PREFETCH_WINDOWS="${GOLEM_FLOW_BASE_WCP_PREFETCH_WINDOWS:-2}"
NODE_CREDITS_SET=(${GOLEM_FLOW_NODE_CREDITS_SET:-32 48 64 82 84 96 128 256})
WCP_PREFETCH_SET=(${GOLEM_FLOW_WCP_PREFETCH_WINDOWS_SET:-1 2 3})

printf 'run_order,phase,label,dma_node_chunk_credits,wcp_prefetch_windows,status,exit_code\n' > "$STATUS_RAW_CSV"

run_order=0
run_case() {
  local phase="$1"
  local label="$2"
  local node_credits="$3"
  local wcp_prefetch_windows="$4"

  run_order=$((run_order + 1))
  echo "[SWEEP] phase=${phase} label=${label} node_chunk_credits=${node_credits} wcp_prefetch_windows=${wcp_prefetch_windows}"
  set +e
  GOLEM_RUN_SUMMARY_CSV="$RUN_SUMMARY_CSV" \
  GOLEM_GEMM_M="$GEMM_M" \
  GOLEM_GEMM_N="$GEMM_N" \
  GOLEM_GEMM_K="$GEMM_K" \
  GOLEM_GEMM_BLOCK_K="$BLOCK_K" \
  GOLEM_A_REUSE_N_TILES=4 \
  GOLEM_B_REUSE_M_TILES=4 \
  GOLEM_DMA_NODE_CHUNK_CREDITS="$node_credits" \
  GOLEM_WCP_PREFETCH_WINDOWS="$wcp_prefetch_windows" \
  timeout "$RUN_TIMEOUT" "$SCRIPT_DIR/run_noc_dma_pipeline.sh"
  rc=$?
  set -e

  if [[ "$rc" -eq 0 ]]; then
    status="PASS"
  elif [[ "$rc" -eq 124 ]]; then
    status="TIMEOUT"
  else
    status="FAIL"
  fi
  printf '%s,%s,%s,%s,%s,%s,%s\n' \
    "$run_order" "$phase" "$label" "$node_credits" "$wcp_prefetch_windows" "$status" "$rc" >> "$STATUS_RAW_CSV"
  if [[ "$status" != "PASS" ]]; then
    echo "[WARN] label=${label} ended with status=${status} exit_code=${rc}; continuing sweep"
  fi
}

for credits in "${NODE_CREDITS_SET[@]}"; do
  run_case "node_credits" "node_${credits}" "$credits" "$BASE_WCP_PREFETCH_WINDOWS"
done

for windows in "${WCP_PREFETCH_SET[@]}"; do
  run_case "wcp_prefetch_windows" "wcp_${windows}" "$BASE_NODE_CREDITS" "$windows"
done

python3 - "$RUN_SUMMARY_CSV" "$STATUS_RAW_CSV" "$SWEEP_CSV" "$PLOTS_DIR" <<'PY'
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt

run_summary = Path(sys.argv[1])
status_raw = Path(sys.argv[2])
out_csv = Path(sys.argv[3])
plots_dir = Path(sys.argv[4])
plots_dir.mkdir(parents=True, exist_ok=True)

summary_rows = list(csv.DictReader(run_summary.open(newline=""))) if run_summary.exists() else []
status_rows = list(csv.DictReader(status_raw.open(newline="")))
pass_count = sum(1 for row in status_rows if row.get("status") == "PASS")
pass_summary_rows = summary_rows[-pass_count:] if pass_count > 0 else []
pass_iter = iter(pass_summary_rows)

fieldnames = [
    "run_order", "phase", "label", "status", "exit_code",
    "dma_node_chunk_credits", "dma_credit_chunk_bytes", "wcp_prefetch_windows", "timestamp", "log_file",
    "gemm_m", "gemm_n", "gemm_k", "block_m", "block_n", "block_k",
    "exec_total_cycles", "speedup_vs_phase_baseline", "exec_array_utilization_pct",
    "exec_breakdown_compute_active_time", "exec_breakdown_prefetch_wait_time",
    "exec_breakdown_writeback_wait_time", "exec_breakdown_control_other_time",
    "hbm_utilization_pct", "hbm_useful_read_bytes", "dma_read_bytes_total_sum",
    "dma_read_issue_count_sum", "dma_avg_rtt_cycles_mean", "dma_max_rtt_cycles_p95",
    "noc_total_xbar_stalls", "memory_queue_delay_avg_cycles", "memory_queue_delay_p99_cycles",
    "causal_return_path_mat_mean_cycles", "causal_return_path_vec_mean_cycles",
    "debug_sched_protocol_mean", "debug_group_wait_mean",
]

records = []
phase_baseline = {}
for status_row in status_rows:
    row = next(pass_iter) if status_row.get("status") == "PASS" else {}
    phase = status_row.get("phase", "")
    total_cycles = float(row.get("exec_total_cycles") or 0) if row else 0.0
    if total_cycles and phase not in phase_baseline:
        phase_baseline[phase] = total_cycles
    speedup = phase_baseline.get(phase, 0.0) / total_cycles if total_cycles and phase in phase_baseline else 0.0
    rec = {
        "run_order": status_row.get("run_order", ""),
        "phase": phase,
        "label": status_row.get("label", ""),
        "status": status_row.get("status", ""),
        "exit_code": status_row.get("exit_code", ""),
        "dma_node_chunk_credits": status_row.get("dma_node_chunk_credits", ""),
        "wcp_prefetch_windows": status_row.get("wcp_prefetch_windows", ""),
        "speedup_vs_phase_baseline": f"{speedup:.6f}" if total_cycles else "",
    }
    for key in fieldnames:
        if key not in rec:
            rec[key] = row.get(key, "") if row else ""
    records.append(rec)

with out_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)

def num(row, key):
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return math.nan

def save_speedup_util(rows, name, title, xlabel):
    labels = [row["label"] for row in rows]
    x = list(range(len(rows)))
    speedup = [num(row, "speedup_vs_phase_baseline") for row in rows]
    util = [num(row, "exec_array_utilization_pct") for row in rows]
    colors = ["#4e79a7" if row["status"] == "PASS" else "#d1d5db" for row in rows]
    fig, ax1 = plt.subplots(figsize=(7.6, 4.2))
    bars = ax1.bar(x, [0.0 if math.isnan(v) else v for v in speedup], color=colors, width=0.62, label="Speedup")
    for bar, val, row in zip(bars, speedup, rows):
        text = row["status"] if row["status"] != "PASS" else f"{val:.2f}x"
        ax1.text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 0.02), text, ha="center", va="bottom", fontsize=8.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Speedup vs phase baseline")
    ax1.set_title(title)
    ax1.grid(True, axis="y")
    ax1.grid(False, axis="x")
    ax1.spines["top"].set_visible(False)
    ax2 = ax1.twinx()
    ax2.plot(x, util, color="#f28e2b", marker="o", linewidth=2, label="Array utilization")
    ax2.set_ylabel("Array utilization (%)")
    ax2.spines["top"].set_visible(False)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(plots_dir / f"{name}.{ext}", dpi=180)
    plt.close(fig)

def save_cycle_breakdown(rows, name, title):
    labels = [row["label"] for row in rows]
    x = list(range(len(rows)))
    total = [num(row, "exec_total_cycles") for row in rows]
    compute = [num(row, "exec_breakdown_compute_active_time") for row in rows]
    prefetch = [num(row, "exec_breakdown_prefetch_wait_time") for row in rows]
    writeback = [num(row, "exec_breakdown_writeback_wait_time") for row in rows]
    other = []
    for t, c, p, w in zip(total, compute, prefetch, writeback):
        known = sum(v for v in (c, p, w) if math.isfinite(v))
        other.append(max(0.0, t - known) if math.isfinite(t) else math.nan)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    bottom = [0.0] * len(rows)
    for label, vals, color in [
        ("Compute", compute, "#4e79a7"),
        ("Prefetch wait", prefetch, "#f28e2b"),
        ("Writeback", writeback, "#59a14f"),
        ("Other", other, "#bab0ab"),
    ]:
        safe = [0.0 if not math.isfinite(v) else v for v in vals]
        ax.bar(x, safe, bottom=bottom, label=label, color=color, width=0.62)
        bottom = [b + v for b, v in zip(bottom, safe)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Cycles")
    ax.set_title(title)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(plots_dir / f"{name}.{ext}", dpi=180)
    plt.close(fig)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#d1d5db",
    "grid.color": "#e5e7eb",
    "legend.frameon": False,
    "font.family": "DejaVu Sans",
})

for phase, title, xlabel in [
    ("node_credits", "Node Chunk Credit Budget Sweep", "Node chunk credits"),
    ("wcp_prefetch_windows", "WCP Prefetch Window Sweep", "WCP prefetch windows"),
]:
    phase_rows = [row for row in records if row["phase"] == phase]
    if phase_rows:
        save_speedup_util(phase_rows, f"{phase}_speedup_util", title, xlabel)
        save_cycle_breakdown(phase_rows, f"{phase}_cycle_breakdown", f"{title}: Cycle Breakdown")

print(f"[OK] wrote sweep CSV: {out_csv}")
print(f"[OK] wrote plots: {plots_dir}")
PY

echo "[OK] sweep directory: $SWEEP_RUN_DIR"
