#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="${GOLEM_ARTIFACT_ROOT:-$SCRIPT_DIR/artifacts}"
SWEEP_ROOT="${GOLEM_SWEEP_ROOT:-$ARTIFACT_ROOT/stats/sweeps/numarrays_mnk1024_bn_eq_na}"
RUN_SUMMARY_CSV="${GOLEM_RUN_SUMMARY_CSV:-$SWEEP_ROOT/run_summary.csv}"
PLOTS_DIR="$SWEEP_ROOT/plots"
SWEEP_CSV="$SWEEP_ROOT/numarrays_sweep_mnk1024_bn_eq_na.csv"

mkdir -p "$SWEEP_ROOT" "$PLOTS_DIR"

ARRAY_SET=(8 16 32 64)
GEMM_M="${GOLEM_SWEEP_GEMM_M:-1024}"
GEMM_N="${GOLEM_SWEEP_GEMM_N:-1024}"
GEMM_K="${GOLEM_SWEEP_GEMM_K:-1024}"

for na in "${ARRAY_SET[@]}"; do
  echo "[SWEEP] Running GOLEM_NUM_ARRAYS=$na (block_n=$na)"
  GOLEM_RUN_SUMMARY_CSV="$RUN_SUMMARY_CSV" \
  GOLEM_GEMM_M="$GEMM_M" \
  GOLEM_GEMM_N="$GEMM_N" \
  GOLEM_GEMM_K="$GEMM_K" \
  GOLEM_NUM_ARRAYS="$na" \
  GOLEM_GEMM_BLOCK_N="$na" \
  GOLEM_MATMUL_BLOCK_N="$na" \
  "$SCRIPT_DIR/run_noc_dma_pipeline.sh"
done

python3 - "$RUN_SUMMARY_CSV" "$SWEEP_CSV" <<'PY'
import csv
import sys
from pathlib import Path

run_summary = Path(sys.argv[1])
out_csv = Path(sys.argv[2])

target_arrays = {8, 16, 32, 64}
rows = list(csv.DictReader(run_summary.open(newline="")))

selected = {}
for row in rows:
    try:
        gemm_m = int(row.get("gemm_m", "0") or 0)
        gemm_n = int(row.get("gemm_n", "0") or 0)
        gemm_k = int(row.get("gemm_k", "0") or 0)
        num_arrays = int(row.get("array_input_size", "0") or 0)
        array_output_size = int(row.get("array_output_size", "0") or 0)
        block_m = int(row.get("block_m", "0") or 0)
        block_n = int(row.get("block_n", "0") or 0)
        block_k = int(row.get("block_k", "0") or 0)
    except ValueError:
        continue

    # run_summary currently records only array_input_size/array_output_size; num_arrays is not persisted,
    # so recover it from block_n for this sweep where block_n == num_arrays by construction.
    num_arrays = block_n
    if gemm_m <= 0 or gemm_n <= 0 or gemm_k <= 0:
        continue
    if num_arrays not in target_arrays:
        continue
    if block_n != num_arrays:
        continue
    selected[num_arrays] = row

fieldnames = [
    "timestamp", "log_file", "overlap", "num_arrays", "gemm_m", "gemm_n", "gemm_k",
    "block_m", "block_n", "block_k", "bias_enable", "bias_value", "num_cores", "gemm_cores",
    "num_mem_nodes", "dma_node_credits", "wcp_prefetch_windows", "submit_batch_size", "done_batch_size", "dma_retry_ticks",
    "dma_burst_bytes", "dma_stagger_cycles", "ctrl_overlap_ab", "noc_link_bw", "noc_xbar_bw",
    "noc_flit_size", "dirctrl_highlink_bw", "wall_time_sec", "simulated_time",
    "exec_total_mean", "exec_array_utilization_pct", "exec_compute_active_time",
    "exec_prefetch_wait_time", "exec_writeback_wait_time", "hbm_utilization_pct",
    "hbm_useful_read_bytes", "hbm_tccdl_roofline_bytes_per_cycle", "dma_timeout_retry_sum",
    "dma_read_issue_count_sum", "dma_write_issue_count_sum", "dma_read_bytes_total_sum",
    "dma_write_bytes_total_sum", "dma_write_timeout_retry_sum", "dma_completion_sum",
    "dma_write_completion_sum", "dma_wait_count_sum", "dma_avg_rtt_cycles_mean", "dma_max_rtt_cycles_p95",
    "noc_total_xbar_stalls", "noc_hotspot_top5pct_port_util_pct", "noc_max_port_util_pct",
    "noc_avg_packet_latency_ns", "noc_p99_packet_latency_ns", "memory_avg_read_latency_cycles",
    "memory_p95_read_latency_bucket_cycles", "memory_read_tail_ge_100_pct", "hbm_channel_bandwidth_imbalance",
    "memory_queue_delay_avg_cycles", "memory_queue_delay_p99_cycles", "memory_backend_read_latency_avg_cycles",
    "memory_backend_read_latency_p99_cycles",
]

mapped = []
for na in sorted(selected):
    row = selected[na]
    rec = {
        "timestamp": row.get("timestamp", ""),
        "log_file": row.get("log_file", ""),
        "overlap": row.get("overlap", ""),
        "num_arrays": str(na),
        "gemm_m": row.get("gemm_m", ""),
        "gemm_n": row.get("gemm_n", ""),
        "gemm_k": row.get("gemm_k", ""),
        "block_m": row.get("block_m", ""),
        "block_n": row.get("block_n", ""),
        "block_k": row.get("block_k", ""),
        "bias_enable": row.get("bias_enable", ""),
        "bias_value": row.get("bias_value", ""),
        "num_cores": row.get("num_cores", ""),
        "gemm_cores": row.get("gemm_cores", ""),
        "num_mem_nodes": row.get("num_mem_nodes", ""),
        "dma_node_credits": row.get("dma_node_credits", ""),
        "wcp_prefetch_windows": row.get("wcp_prefetch_windows", ""),
        "submit_batch_size": row.get("submit_batch_size", ""),
        "done_batch_size": row.get("done_batch_size", ""),
        "dma_retry_ticks": row.get("dma_retry_ticks", ""),
        "dma_burst_bytes": row.get("dma_burst_bytes", ""),
        "dma_stagger_cycles": row.get("dma_stagger_cycles", ""),
        "ctrl_overlap_ab": row.get("ctrl_overlap_ab", ""),
        "noc_link_bw": row.get("noc_link_bw", ""),
        "noc_xbar_bw": row.get("noc_xbar_bw", ""),
        "noc_flit_size": row.get("noc_flit_size", ""),
        "dirctrl_highlink_bw": row.get("dirctrl_highlink_bw", ""),
        "wall_time_sec": row.get("wall_time_sec", ""),
        "simulated_time": row.get("simulated_time", ""),
        "exec_total_mean": row.get("exec_total_cycles", ""),
        "exec_array_utilization_pct": row.get("exec_array_utilization_pct", ""),
        "exec_compute_active_time": row.get("exec_breakdown_compute_active_time", ""),
        "exec_prefetch_wait_time": row.get("exec_breakdown_prefetch_wait_time", ""),
        "exec_writeback_wait_time": row.get("exec_breakdown_writeback_wait_time", ""),
        "hbm_utilization_pct": row.get("hbm_utilization_pct", ""),
        "hbm_useful_read_bytes": row.get("hbm_useful_read_bytes", ""),
        "hbm_tccdl_roofline_bytes_per_cycle": row.get("hbm_tccdl_roofline_bytes_per_cycle", ""),
        "dma_timeout_retry_sum": row.get("dma_timeout_retry_sum", ""),
        "dma_read_issue_count_sum": row.get("dma_read_issue_count_sum", ""),
        "dma_write_issue_count_sum": row.get("dma_write_issue_count_sum", ""),
        "dma_read_bytes_total_sum": row.get("dma_read_bytes_total_sum", ""),
        "dma_write_bytes_total_sum": row.get("dma_write_bytes_total_sum", ""),
        "dma_write_timeout_retry_sum": row.get("dma_write_timeout_retry_sum", ""),
        "dma_completion_sum": row.get("dma_completion_sum", ""),
        "dma_write_completion_sum": row.get("dma_write_completion_sum", ""),
        "dma_wait_count_sum": row.get("dma_wait_count_sum", ""),
        "dma_avg_rtt_cycles_mean": row.get("dma_avg_rtt_cycles_mean", ""),
        "dma_max_rtt_cycles_p95": row.get("dma_max_rtt_cycles_p95", ""),
        "noc_total_xbar_stalls": row.get("noc_total_xbar_stalls", ""),
        "noc_hotspot_top5pct_port_util_pct": row.get("noc_hotspot_top5pct_port_util_pct", ""),
        "noc_max_port_util_pct": row.get("noc_max_port_util_pct", ""),
        "noc_avg_packet_latency_ns": row.get("noc_avg_packet_latency_ns", ""),
        "noc_p99_packet_latency_ns": row.get("noc_p99_packet_latency_ns", ""),
        "memory_avg_read_latency_cycles": row.get("memory_avg_read_latency_cycles", ""),
        "memory_p95_read_latency_bucket_cycles": row.get("memory_p95_read_latency_bucket_cycles", ""),
        "memory_read_tail_ge_100_pct": row.get("memory_read_tail_ge_100_pct", ""),
        "hbm_channel_bandwidth_imbalance": row.get("hbm_channel_bandwidth_imbalance", ""),
        "memory_queue_delay_avg_cycles": row.get("memory_queue_delay_avg_cycles", ""),
        "memory_queue_delay_p99_cycles": row.get("memory_queue_delay_p99_cycles", ""),
        "memory_backend_read_latency_avg_cycles": row.get("memory_backend_read_latency_avg_cycles", ""),
        "memory_backend_read_latency_p99_cycles": row.get("memory_backend_read_latency_p99_cycles", ""),
    }
    mapped.append(rec)

out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(mapped)

print(f"[OK] wrote sweep CSV: {out_csv}")
PY

python3 "$SCRIPT_DIR/stats/plot_dim_sweep.py" \
  --input "$SWEEP_CSV" \
  --output-dir "$PLOTS_DIR" \
  --x-field num_arrays \
  --x-label "GOLEM_NUM_ARRAYS (= block_n)"

python3 "$SCRIPT_DIR/stats/plot_numarrays_scaling.py" \
  --input "$SWEEP_CSV" \
  --output "$PLOTS_DIR/numarrays_density_scaling.png" \
  --title "Density Scaling" \
  --baseline 8

echo "[OK] sweep directory: $SWEEP_ROOT"
