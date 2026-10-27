#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="${GOLEM_ARTIFACT_ROOT:-$SCRIPT_DIR/artifacts}"
PARAM_FILE="${GOLEM_LOW_SYSUTI_PARAM_FILE:-$SCRIPT_DIR/low_sysuti_iter_round1.csv}"
SWEEP_ROOT="${GOLEM_SWEEP_ROOT:-$ARTIFACT_ROOT/stats/sweeps/low_sysuti_iterative}"
SWEEP_TAG="${GOLEM_SWEEP_TAG:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_RUN_DIR="${GOLEM_SWEEP_RUN_DIR:-$SWEEP_ROOT/run_$SWEEP_TAG}"
RUN_SUMMARY_CSV="$SWEEP_RUN_DIR/run_summary.csv"
PLAN_TSV="$SWEEP_RUN_DIR/low_sysuti_plan.tsv"
STATUS_RAW_CSV="$SWEEP_RUN_DIR/low_sysuti_status_raw.csv"
SWEEP_CSV="$SWEEP_RUN_DIR/low_sysuti_sweep.csv"
SHARED_HBM_DIR="$SWEEP_RUN_DIR/hbm"
RUN_TIMEOUT="${GOLEM_LOW_SYSUTI_SWEEP_TIMEOUT:-${GOLEM_SWEEP_TIMEOUT:-}}"

if [[ ! -f "$PARAM_FILE" ]]; then
  echo "[ERR] Missing parameter file: $PARAM_FILE" >&2
  exit 1
fi

mkdir -p "$SWEEP_RUN_DIR" "$SHARED_HBM_DIR"

python3 - "$PARAM_FILE" "$PLAN_TSV" "$SWEEP_TAG" <<'PY'
import csv
import re
import sys
from pathlib import Path

param_path = Path(sys.argv[1])
plan_path = Path(sys.argv[2])
tag = sys.argv[3]

required = [
    "label", "shape_m", "shape_k", "shape_n", "shape_b",
    "block_m", "block_n", "block_k",
    "a_reuse_n_tiles", "b_reuse_m_tiles",
    "dma_window_k_tiles", "dma_slot_count",
    "dma_node_chunk_credits", "wcp_prefetch_windows",
]

def positive_int(row, key):
    raw = (row.get(key) or "").strip()
    if not raw.isdigit() or int(raw) <= 0:
        raise SystemExit(f"[ERR] {key} must be a positive integer for label={row.get('label')}: {raw!r}")
    return int(raw)

def safe_label(label):
    return re.sub(r"[^A-Za-z0-9_]+", "_", label).strip("_")

rows = []
with param_path.open(newline="") as f:
    reader = csv.DictReader(f)
    missing = [key for key in required if key not in (reader.fieldnames or [])]
    if missing:
        raise SystemExit(f"[ERR] missing columns: {missing}")
    for idx, row in enumerate(reader, start=1):
        label = (row.get("label") or "").strip()
        if not label:
            continue
        rec = {"run_order": idx, "label": label}
        for key in required[1:]:
            rec[key] = positive_int(row, key)
        rec["selection_note"] = (row.get("selection_note") or "").strip()
        if rec["shape_m"] % rec["block_m"] or rec["shape_n"] % rec["block_n"] or rec["shape_k"] % rec["block_k"]:
            raise SystemExit(
                f"[ERR] shape {label} [{rec['shape_m']},{rec['shape_k']},{rec['shape_n']}] "
                f"is not divisible by block [{rec['block_m']},{rec['block_k']},{rec['block_n']}]"
            )
        rec["run_id"] = (
            f"low_sysuti_{tag}_{idx}_{safe_label(label)}_"
            f"m{rec['shape_m']}_n{rec['shape_n']}_k{rec['shape_k']}_"
            f"bk{rec['block_k']}_rN{rec['a_reuse_n_tiles']}_rM{rec['b_reuse_m_tiles']}_"
            f"w{rec['dma_window_k_tiles']}_s{rec['dma_slot_count']}_c{rec['dma_node_chunk_credits']}"
        )
        rows.append(rec)

if not rows:
    raise SystemExit(f"[ERR] no rows parsed from {param_path}")

fieldnames = [
    "run_order", "label", "run_id",
    "shape_m", "shape_k", "shape_n", "shape_b",
    "block_m", "block_n", "block_k",
    "a_reuse_n_tiles", "b_reuse_m_tiles",
    "dma_window_k_tiles", "dma_slot_count",
    "dma_node_chunk_credits", "wcp_prefetch_windows",
    "selection_note",
]
with plan_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

print(f"[OK] wrote plan TSV: {plan_path}")
print(f"[OK] planned rows: {len(rows)}")
PY

printf 'run_order,label,run_id,status,exit_code,shape_m,shape_k,shape_n,shape_b,block_m,block_n,block_k,a_reuse_n_tiles,b_reuse_m_tiles,dma_window_k_tiles,dma_slot_count,dma_node_chunk_credits,wcp_prefetch_windows,selection_note\n' > "$STATUS_RAW_CSV"

echo "[INFO] parameter file: $PARAM_FILE"
echo "[INFO] output dir: $SWEEP_RUN_DIR"

planned_count=0
launched_count=0

read -r _ < "$PLAN_TSV"
while IFS=$'\t' read -r run_order label run_id shape_m shape_k shape_n shape_b block_m block_n block_k a_reuse_n b_reuse_m dma_window_k dma_slot_count dma_node_chunk_credits wcp_prefetch_windows selection_note; do
  planned_count=$((planned_count + 1))
  launched_count=$((launched_count + 1))
  log_label="${label//[^A-Za-z0-9_]/_}"
  log_name="low_sysuti_${run_order}_${log_label}_m${shape_m}_n${shape_n}_k${shape_k}_bk${block_k}_rN${a_reuse_n}_rM${b_reuse_m}_w${dma_window_k}_s${dma_slot_count}_c${dma_node_chunk_credits}.log"

  echo "[SWEEP] ${label}: shape=[${shape_m},${shape_k},${shape_n},${shape_b}] block=${block_m}/${block_n}/${block_k} reuseN/M=${a_reuse_n}/${b_reuse_m} winK=${dma_window_k} slots=${dma_slot_count} nodeCredits=${dma_node_chunk_credits}"

  set +e
  if [[ -n "$RUN_TIMEOUT" && "$RUN_TIMEOUT" != "0" ]]; then
    env \
      GOLEM_RUN_ID="$run_id" \
      GOLEM_RUN_SUMMARY_CSV="$RUN_SUMMARY_CSV" \
      GOLEM_ARTIFACT_ROOT="$SWEEP_RUN_DIR/artifacts" \
      GOLEM_HBM_DIR="$SHARED_HBM_DIR" \
      GOLEM_GEMM_M="$shape_m" \
      GOLEM_GEMM_N="$shape_n" \
      GOLEM_GEMM_K="$shape_k" \
      GOLEM_ORIG_M="$shape_m" \
      GOLEM_ORIG_N="$shape_n" \
      GOLEM_ORIG_K="$shape_k" \
      GOLEM_GEMM_BLOCK_M="$block_m" \
      GOLEM_GEMM_BLOCK_N="$block_n" \
      GOLEM_GEMM_BLOCK_K="$block_k" \
      GOLEM_A_REUSE_N_TILES="$a_reuse_n" \
      GOLEM_B_REUSE_M_TILES="$b_reuse_m" \
      GOLEM_DMA_WINDOW_K_TILES="$dma_window_k" \
      GOLEM_DMA_SLOT_COUNT="$dma_slot_count" \
      GOLEM_DMA_NODE_CHUNK_CREDITS="$dma_node_chunk_credits" \
      GOLEM_WCP_PREFETCH_WINDOWS="$wcp_prefetch_windows" \
      GOLEM_PRESET_LOG="$log_name" \
      GOLEM_TENSOR_SOURCE=synthetic \
      GOLEM_VERIFY_C=0 \
      GOLEM_DUMP_C_FILE= \
      GOLEM_HBM_DUMP_OUTPUT=0 \
      GOLEM_MEM_NODE_SIZE_BYTES=auto \
      GOLEM_BENCH_QUIET_LOGS=1 \
      timeout "$RUN_TIMEOUT" "$SCRIPT_DIR/run_noc_dma_pipeline.sh"
  else
    env \
      GOLEM_RUN_ID="$run_id" \
      GOLEM_RUN_SUMMARY_CSV="$RUN_SUMMARY_CSV" \
      GOLEM_ARTIFACT_ROOT="$SWEEP_RUN_DIR/artifacts" \
      GOLEM_HBM_DIR="$SHARED_HBM_DIR" \
      GOLEM_GEMM_M="$shape_m" \
      GOLEM_GEMM_N="$shape_n" \
      GOLEM_GEMM_K="$shape_k" \
      GOLEM_ORIG_M="$shape_m" \
      GOLEM_ORIG_N="$shape_n" \
      GOLEM_ORIG_K="$shape_k" \
      GOLEM_GEMM_BLOCK_M="$block_m" \
      GOLEM_GEMM_BLOCK_N="$block_n" \
      GOLEM_GEMM_BLOCK_K="$block_k" \
      GOLEM_A_REUSE_N_TILES="$a_reuse_n" \
      GOLEM_B_REUSE_M_TILES="$b_reuse_m" \
      GOLEM_DMA_WINDOW_K_TILES="$dma_window_k" \
      GOLEM_DMA_SLOT_COUNT="$dma_slot_count" \
      GOLEM_DMA_NODE_CHUNK_CREDITS="$dma_node_chunk_credits" \
      GOLEM_WCP_PREFETCH_WINDOWS="$wcp_prefetch_windows" \
      GOLEM_PRESET_LOG="$log_name" \
      GOLEM_TENSOR_SOURCE=synthetic \
      GOLEM_VERIFY_C=0 \
      GOLEM_DUMP_C_FILE= \
      GOLEM_HBM_DUMP_OUTPUT=0 \
      GOLEM_MEM_NODE_SIZE_BYTES=auto \
      GOLEM_BENCH_QUIET_LOGS=1 \
      "$SCRIPT_DIR/run_noc_dma_pipeline.sh"
  fi
  rc=$?
  set -e

  if [[ "$rc" -eq 0 ]]; then
    status="PASS"
  elif [[ "$rc" -eq 124 ]]; then
    status="TIMEOUT"
  else
    status="FAIL"
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$run_order" "$label" "$run_id" "$status" "$rc" \
    "$shape_m" "$shape_k" "$shape_n" "$shape_b" "$block_m" "$block_n" "$block_k" \
    "$a_reuse_n" "$b_reuse_m" "$dma_window_k" "$dma_slot_count" "$dma_node_chunk_credits" "$wcp_prefetch_windows" "$selection_note" >> "$STATUS_RAW_CSV"
  if [[ "$status" != "PASS" ]]; then
    echo "[WARN] ${label} ended with status=${status} exit_code=${rc}; continuing sweep"
  fi
done < <(tail -n +2 "$PLAN_TSV")

python3 - "$RUN_SUMMARY_CSV" "$STATUS_RAW_CSV" "$SWEEP_CSV" <<'PY'
import csv
import sys
from pathlib import Path

run_summary = Path(sys.argv[1])
status_raw = Path(sys.argv[2])
out_csv = Path(sys.argv[3])

summary_rows = list(csv.DictReader(run_summary.open(newline=""))) if run_summary.exists() else []
status_rows = list(csv.DictReader(status_raw.open(newline="")))
summary_by_run_id = {row.get("run_id", "").strip(): row for row in summary_rows if row.get("run_id", "").strip()}

metric_fields = [
    "timestamp", "log_file", "wall_time_sec", "simulated_time",
    "gemm_m", "gemm_n", "gemm_k", "block_m", "block_n", "block_k",
    "dma_node_chunk_credits", "wcp_prefetch_windows",
    "exec_total_cycles", "gemm_system_latency_cycles",
    "exec_avg_throughput_ops_per_cycle", "exec_system_avg_throughput_ops_per_cycle",
    "exec_array_utilization_pct", "exec_system_array_utilization_pct",
    "exec_worker_p95_total_cycles", "exec_worker_max_total_cycles",
    "exec_breakdown_compute_active_time", "exec_breakdown_prefetch_wait_time",
    "exec_breakdown_writeback_wait_time", "exec_breakdown_control_other_time",
    "hbm_utilization_pct", "hbm_pressure_vs_gemm_system_pct",
    "hbm_backend_service_window_utilization_pct", "hbm_backend_active_utilization_pct",
    "hbm_useful_read_bytes", "dma_strict_avg_rtt_cycles_mean", "dma_strict_max_rtt_cycles_max",
    "noc_total_xbar_stalls", "noc_hotspot_top5pct_port_util_pct", "noc_max_port_util_pct",
    "memory_queue_delay_avg_cycles", "memory_queue_delay_p99_cycles",
    "memory_backend_read_latency_avg_cycles", "memory_backend_read_latency_p99_cycles",
    "causal_issue_to_pending_mat_mean_cycles", "causal_issue_to_pending_vec_mean_cycles",
    "causal_return_path_mat_mean_cycles", "causal_return_path_vec_mean_cycles",
]
fieldnames = list(status_rows[0].keys()) + [f for f in metric_fields if f not in status_rows[0]]

records = []
for status_row in status_rows:
    run_id = status_row.get("run_id", "").strip()
    summary = summary_by_run_id.get(run_id, {})
    rec = dict(status_row)
    for field in metric_fields:
        if field not in rec:
            rec[field] = summary.get(field, "")
    records.append(rec)

with out_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)

print(f"[OK] wrote sweep CSV: {out_csv}")
PY

echo "[OK] planned rows: $planned_count"
echo "[OK] launched runs: $launched_count"
echo "[OK] plan TSV: $PLAN_TSV"
echo "[OK] status CSV: $STATUS_RAW_CSV"
echo "[OK] sweep CSV: $SWEEP_CSV"
echo "[OK] sweep directory: $SWEEP_RUN_DIR"
