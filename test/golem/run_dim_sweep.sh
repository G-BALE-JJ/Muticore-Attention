#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="${GOLEM_ARTIFACT_ROOT:-$SCRIPT_DIR/artifacts}"
SWEEP_ROOT="${GOLEM_SWEEP_ROOT:-$ARTIFACT_ROOT/stats/sweeps/dim_power2_synthetic}"
SWEEP_TAG="${GOLEM_SWEEP_TAG:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_RUN_DIR="${GOLEM_SWEEP_RUN_DIR:-$SWEEP_ROOT/run_$SWEEP_TAG}"
RUN_SUMMARY_CSV="$SWEEP_RUN_DIR/run_summary.csv"
STATUS_RAW_CSV="$SWEEP_RUN_DIR/dim_status_raw.csv"
SWEEP_CSV="$SWEEP_RUN_DIR/dim_sweep.csv"
PLOTS_DIR="$SWEEP_RUN_DIR/plots"
SHARED_HBM_DIR="$SWEEP_RUN_DIR/hbm"
RUN_TIMEOUT="${GOLEM_DIM_SWEEP_TIMEOUT:-}"
DIMS=(${GOLEM_DIM_SWEEP_DIMS:-128 256 512 1024 2048 4096 8192 16384 32768})

mkdir -p "$SWEEP_RUN_DIR" "$PLOTS_DIR" "$SHARED_HBM_DIR"
printf 'run_order,dim,status,exit_code\n' > "$STATUS_RAW_CSV"

run_order=0
for dim in "${DIMS[@]}"; do
  run_order=$((run_order + 1))
  echo "[SWEEP] dim=${dim} synthetic verify=0 hbm_dump_output=0 mem_node_size=auto"
  cmd=(
    "$SCRIPT_DIR/run_noc_dma_pipeline.sh"
    --gemm-m "$dim"
    --gemm-n "$dim"
    --gemm-k "$dim"
    --orig-m "$dim"
    --orig-n "$dim"
    --orig-k "$dim"
    --mem-node-size auto
    --no-hbm-dump-output
    --log "dim_${dim}.log"
  )

  set +e
  if [[ -n "$RUN_TIMEOUT" && "$RUN_TIMEOUT" != "0" ]]; then
    env \
      GOLEM_RUN_ID="dim_${dim}_${SWEEP_TAG}" \
      GOLEM_RUN_SUMMARY_CSV="$RUN_SUMMARY_CSV" \
      GOLEM_ARTIFACT_ROOT="$SWEEP_RUN_DIR/artifacts" \
      GOLEM_HBM_DIR="$SHARED_HBM_DIR" \
      GOLEM_TENSOR_SOURCE=synthetic \
      GOLEM_VERIFY_C=0 \
      GOLEM_DUMP_C_FILE= \
      GOLEM_HBM_DUMP_OUTPUT=0 \
      GOLEM_MEM_NODE_SIZE_BYTES=auto \
      GOLEM_BENCH_QUIET_LOGS=1 \
      timeout "$RUN_TIMEOUT" "${cmd[@]}"
  else
    env \
      GOLEM_RUN_ID="dim_${dim}_${SWEEP_TAG}" \
      GOLEM_RUN_SUMMARY_CSV="$RUN_SUMMARY_CSV" \
      GOLEM_ARTIFACT_ROOT="$SWEEP_RUN_DIR/artifacts" \
      GOLEM_HBM_DIR="$SHARED_HBM_DIR" \
      GOLEM_TENSOR_SOURCE=synthetic \
      GOLEM_VERIFY_C=0 \
      GOLEM_DUMP_C_FILE= \
      GOLEM_HBM_DUMP_OUTPUT=0 \
      GOLEM_MEM_NODE_SIZE_BYTES=auto \
      GOLEM_BENCH_QUIET_LOGS=1 \
      "${cmd[@]}"
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
  printf '%s,%s,%s,%s\n' "$run_order" "$dim" "$status" "$rc" >> "$STATUS_RAW_CSV"
  if [[ "$status" != "PASS" ]]; then
    echo "[WARN] dim=${dim} ended with status=${status} exit_code=${rc}; continuing sweep"
  fi
done

python3 - "$RUN_SUMMARY_CSV" "$STATUS_RAW_CSV" "$SWEEP_CSV" <<'PY'
import csv
import sys
from pathlib import Path

run_summary = Path(sys.argv[1])
status_raw = Path(sys.argv[2])
out_csv = Path(sys.argv[3])

summary_rows = list(csv.DictReader(run_summary.open(newline=""))) if run_summary.exists() else []
status_rows = list(csv.DictReader(status_raw.open(newline="")))
pass_count = sum(1 for row in status_rows if row.get("status") == "PASS")
pass_summary_rows = summary_rows[-pass_count:] if pass_count > 0 else []
pass_iter = iter(pass_summary_rows)

fieldnames = [
    "run_order", "dim", "status", "exit_code",
    "timestamp", "run_id", "log_file", "gemm_m", "gemm_n", "gemm_k",
    "block_m", "block_n", "block_k", "num_cores", "gemm_cores",
    "num_mem_nodes", "mem_node_size_bytes", "hbm_dump_output",
    "wall_time_sec", "simulated_time",
    "exec_total_cycles", "gemm_system_latency_cycles",
    "exec_array_utilization_pct", "exec_system_array_utilization_pct",
    "exec_worker_avg_array_efficiency_pct",
    "exec_breakdown_compute_active_time",
    "exec_breakdown_prefetch_wait_time",
    "exec_breakdown_writeback_wait_time",
    "exec_breakdown_control_other_time",
    "hbm_utilization_pct", "hbm_useful_read_bytes",
    "hbm_tccdl_roofline_bytes_per_cycle",
    "dma_read_bytes_total_sum", "dma_write_bytes_total_sum",
    "dma_read_issue_count_sum", "dma_write_issue_count_sum",
    "dma_timeout_retry_sum", "dma_avg_rtt_cycles_mean", "dma_max_rtt_cycles_p95",
    "noc_total_xbar_stalls", "noc_hotspot_top5pct_port_util_pct",
    "noc_max_port_util_pct", "noc_avg_packet_latency_ns", "noc_p99_packet_latency_ns",
    "memory_avg_read_latency_cycles", "memory_p95_read_latency_bucket_cycles",
    "memory_read_tail_ge_100_pct", "hbm_channel_bandwidth_imbalance",
    "memory_queue_delay_avg_cycles", "memory_queue_delay_p99_cycles",
    "memory_backend_read_latency_avg_cycles", "memory_backend_read_latency_p99_cycles",
]

records = []
for status_row in status_rows:
    row = next(pass_iter) if status_row.get("status") == "PASS" else {}
    rec = {
        "run_order": status_row.get("run_order", ""),
        "dim": status_row.get("dim", ""),
        "status": status_row.get("status", ""),
        "exit_code": status_row.get("exit_code", ""),
    }
    for key in fieldnames:
        if key not in rec:
            rec[key] = row.get(key, "") if row else ""
    records.append(rec)

out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)

print(f"[OK] wrote sweep CSV: {out_csv}")
PY

if grep -q ',PASS,' "$STATUS_RAW_CSV"; then
  python3 "$SCRIPT_DIR/stats/plot_dim_sweep.py" \
    --input "$SWEEP_CSV" \
    --output-dir "$PLOTS_DIR" \
    --x-field dim \
    --x-label "GEMM dimension (M=N=K)"
fi

echo "[OK] sweep directory: $SWEEP_RUN_DIR"
echo "[OK] sweep CSV: $SWEEP_CSV"
