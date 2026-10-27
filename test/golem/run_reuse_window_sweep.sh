#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="${GOLEM_ARTIFACT_ROOT:-$SCRIPT_DIR/artifacts}"
SWEEP_ROOT="${GOLEM_SWEEP_ROOT:-$ARTIFACT_ROOT/stats/sweeps/reuse_window_bk128}"
SWEEP_TAG="${GOLEM_SWEEP_TAG:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_RUN_DIR="${GOLEM_SWEEP_RUN_DIR:-$SWEEP_ROOT/run_$SWEEP_TAG}"
RUN_SUMMARY_CSV="$SWEEP_RUN_DIR/run_summary.csv"
SWEEP_CSV="$SWEEP_RUN_DIR/reuse_window_sweep_bk128.csv"
STATUS_RAW_CSV="$SWEEP_RUN_DIR/reuse_window_status_raw.csv"
PLOTS_DIR="$SWEEP_RUN_DIR/plots"

mkdir -p "$SWEEP_RUN_DIR" "$PLOTS_DIR"

GEMM_M="${GOLEM_SWEEP_GEMM_M:-1024}"
GEMM_N="${GOLEM_SWEEP_GEMM_N:-1024}"
GEMM_K="${GOLEM_SWEEP_GEMM_K:-1024}"
BLOCK_K="${GOLEM_SWEEP_GEMM_BLOCK_K:-128}"
RUN_TIMEOUT="${GOLEM_SWEEP_TIMEOUT:-300s}"

REUSE_CONFIGS=(
  "1x1:1:1"
  "1x4:1:4"
  "4x1:4:1"
  "2x2:2:2"
  "4x4:4:4"
)

printf 'reuse_order,reuse_config,a_reuse_n_tiles,b_reuse_m_tiles,status,exit_code\n' > "$STATUS_RAW_CSV"

reuse_order=0
for cfg in "${REUSE_CONFIGS[@]}"; do
  reuse_order=$((reuse_order + 1))
  IFS=: read -r label a_reuse_n b_reuse_m <<<"$cfg"
  echo "[SWEEP] Running reuse=${label} A_REUSE_N_TILES=${a_reuse_n} B_REUSE_M_TILES=${b_reuse_m} block_k=${BLOCK_K}"
  set +e
  GOLEM_RUN_SUMMARY_CSV="$RUN_SUMMARY_CSV" \
  GOLEM_GEMM_M="$GEMM_M" \
  GOLEM_GEMM_N="$GEMM_N" \
  GOLEM_GEMM_K="$GEMM_K" \
  GOLEM_GEMM_BLOCK_K="$BLOCK_K" \
  GOLEM_A_REUSE_N_TILES="$a_reuse_n" \
  GOLEM_B_REUSE_M_TILES="$b_reuse_m" \
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
  printf '%s,%s,%s,%s,%s,%s\n' "$reuse_order" "$label" "$a_reuse_n" "$b_reuse_m" "$status" "$rc" >> "$STATUS_RAW_CSV"
  if [[ "$status" != "PASS" ]]; then
    echo "[WARN] reuse=${label} ended with status=${status} exit_code=${rc}; continuing sweep"
  fi
done

python3 - "$RUN_SUMMARY_CSV" "$STATUS_RAW_CSV" "$SWEEP_CSV" <<'PY'
import csv
import sys
from pathlib import Path

run_summary = Path(sys.argv[1])
status_raw = Path(sys.argv[2])
out_csv = Path(sys.argv[3])

rows = list(csv.DictReader(run_summary.open(newline=""))) if run_summary.exists() else []
status_rows = list(csv.DictReader(status_raw.open(newline="")))
pass_rows = iter(rows[-sum(1 for row in status_rows if row.get("status") == "PASS"):])
fieldnames = [
    "reuse_order",
    "reuse_config",
    "a_reuse_n_tiles",
    "b_reuse_m_tiles",
    "status",
    "exit_code",
    "timestamp",
    "log_file",
    "gemm_m",
    "gemm_n",
    "gemm_k",
    "block_m",
    "block_n",
    "block_k",
    "exec_total_cycles",
    "speedup_vs_1x1",
    "exec_array_utilization_pct",
    "exec_breakdown_compute_active_time",
    "exec_breakdown_prefetch_wait_time",
    "exec_breakdown_writeback_wait_time",
    "hbm_utilization_pct",
    "hbm_useful_read_bytes",
    "dma_read_bytes_total_sum",
    "dma_write_bytes_total_sum",
    "dma_read_issue_count_sum",
    "dma_write_issue_count_sum",
    "memory_queue_delay_avg_cycles",
    "memory_queue_delay_p99_cycles",
    "noc_total_xbar_stalls",
]

baseline_cycles = None
mapped = []
for status_row in status_rows:
    row = next(pass_rows) if status_row.get("status") == "PASS" else {}
    total_cycles = float(row.get("exec_total_cycles", "0") or 0)
    if baseline_cycles is None and total_cycles:
        baseline_cycles = total_cycles
    speedup = baseline_cycles / total_cycles if baseline_cycles and total_cycles else 0.0
    rec = {
        "reuse_order": status_row.get("reuse_order", ""),
        "reuse_config": status_row.get("reuse_config", ""),
        "a_reuse_n_tiles": status_row.get("a_reuse_n_tiles", ""),
        "b_reuse_m_tiles": status_row.get("b_reuse_m_tiles", ""),
        "status": status_row.get("status", ""),
        "exit_code": status_row.get("exit_code", ""),
        "speedup_vs_1x1": f"{speedup:.6f}" if total_cycles else "",
    }
    for name in fieldnames:
        if name not in rec:
            rec[name] = row.get(name, "")
    mapped.append(rec)

out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(mapped)

print(f"[OK] wrote sweep CSV: {out_csv}")
PY

python3 - "$SWEEP_CSV" "$PLOTS_DIR" <<'PY'
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt

csv_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

with csv_path.open(newline="") as f:
    rows = list(csv.DictReader(f))

def number(row, key):
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan

labels = [row["reuse_config"] for row in rows]
xpos = list(range(len(rows)))
speedup = [number(row, "speedup_vs_1x1") for row in rows]
util = [number(row, "exec_array_utilization_pct") for row in rows]
cycles = [number(row, "exec_total_cycles") for row in rows]
compute = [number(row, "exec_breakdown_compute_active_time") for row in rows]
prefetch = [number(row, "exec_breakdown_prefetch_wait_time") for row in rows]
writeback = [number(row, "exec_breakdown_writeback_wait_time") for row in rows]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#d1d5db",
    "axes.grid": True,
    "grid.color": "#e5e7eb",
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "font.family": "DejaVu Sans",
})

fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
bars = ax1.bar(xpos, speedup, color="#4e79a7", width=0.62, label="Speedup vs 1x1")
ax1.set_ylabel("Speedup vs 1x1")
ax1.set_xticks(xpos)
ax1.set_xticklabels(labels)
ax1.set_xlabel("2D reuse window")
ax1.spines["top"].set_visible(False)
ax1.grid(True, axis="y")
ax1.grid(False, axis="x")
for bar, val in zip(bars, speedup):
    if math.isfinite(val):
        ax1.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}x", ha="center", va="bottom", fontsize=9)

ax2 = ax1.twinx()
ax2.plot(xpos, util, color="#f28e2b", marker="o", linewidth=2.0, label="Array utilization")
ax2.set_ylabel("Array utilization (%)")
ax2.spines["top"].set_visible(False)

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
ax1.set_title("2D Reuse Window Ablation")
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(out_dir / f"reuse_window_speedup_util.{ext}", dpi=180)
plt.close(fig)

other = []
for total, c, p, w in zip(cycles, compute, prefetch, writeback):
    known = sum(v for v in (c, p, w) if math.isfinite(v))
    other.append(max(0.0, total - known) if math.isfinite(total) else math.nan)

fig, ax = plt.subplots(figsize=(7.2, 4.2))
bottom = [0.0] * len(rows)
for name, vals, color in [
    ("Compute active", compute, "#4e79a7"),
    ("Prefetch wait", prefetch, "#f28e2b"),
    ("Writeback wait", writeback, "#59a14f"),
    ("Other", other, "#bab0ab"),
]:
    safe_vals = [0.0 if not math.isfinite(v) else v for v in vals]
    ax.bar(xpos, safe_vals, bottom=bottom, label=name, color=color, width=0.62)
    bottom = [b + v for b, v in zip(bottom, safe_vals)]
ax.set_xticks(xpos)
ax.set_xticklabels(labels)
ax.set_xlabel("2D reuse window")
ax.set_ylabel("Cycles")
ax.set_title("Execution Cycle Breakdown")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, axis="y")
ax.grid(False, axis="x")
ax.legend(loc="upper right")
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(out_dir / f"reuse_window_cycle_breakdown.{ext}", dpi=180)
plt.close(fig)

print(f"[OK] wrote plots: {out_dir}")
PY

echo "[OK] sweep directory: $SWEEP_RUN_DIR"
