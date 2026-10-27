#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="$SCRIPT_DIR/run_noc_dma_pipeline.sh"
DIM="${1:-1024}"
ROOT="${GOLEM_CH4_BATCH_ROOT:-$SCRIPT_DIR/artifacts/chapter4_dim_batch_$(date +%Y%m%d_%H%M%S)_${DIM}}"
TIMEOUT="${GOLEM_CH4_TIMEOUT:-900s}"
MPI_RANKS=4
SLOTS=4

case "$DIM" in
  1024|2048) ;;
  *) echo "[ERROR] dimension must be 1024 or 2048" >&2; exit 2 ;;
esac

mkdir -p "$ROOT/cases"

# label, A reuse, B reuse, node chunk credit
CASES=(
  "baseline_1x1_cred0:1:1:0"
  "reuse_4x4_cred0:4:4:0"
  "reuse_4x4_cred32:4:4:32"
  "reuse_4x4_cred64:4:4:64"
  "reuse_4x4_cred128:4:4:128"
  "reuse_4x1_cred128:4:1:128"
  "reuse_1x4_cred128:1:4:128"
  "full_4x4_cred128:4:4:128"
)

mapfile -t PHYSICAL_CPUS < <(lscpu -p=CPU,CORE,SOCKET | awk -F, '$1 !~ /^#/ {key=$3 ":" $2; if (!seen[key]++) print $1}' | head -n $((SLOTS * MPI_RANKS)))
if (( ${#PHYSICAL_CPUS[@]} < SLOTS * MPI_RANKS )); then
  echo "[ERROR] fewer than 16 physical CPUs available" >&2
  exit 2
fi

printf 'case_index\tlabel\tstatus\tcase_root\tcpus\ta_reuse_n\tb_reuse_m\tnode_chunk_credit\n' > "$ROOT/status.tsv"

run_case() {
  local index="$1" spec="$2" slot="$3"
  local label reuse_n reuse_m credit case_root run_id cpu_list
  IFS=: read -r label reuse_n reuse_m credit <<< "$spec"
  case_root="$ROOT/cases/$(printf '%02d_%s' "$index" "$label")"
  run_id="ch4_${DIM}_${index}_${label}"
  cpu_list="${PHYSICAL_CPUS[$((slot * MPI_RANKS))]},${PHYSICAL_CPUS[$((slot * MPI_RANKS + 1))]},${PHYSICAL_CPUS[$((slot * MPI_RANKS + 2))]},${PHYSICAL_CPUS[$((slot * MPI_RANKS + 3))]}"
  mkdir -p "$case_root"
  echo "[START] dim=$DIM case=$label slot=$slot cpus=$cpu_list reuse=${reuse_n}x${reuse_m} credit=$credit"
  (
    export GOLEM_ARTIFACT_ROOT="$case_root/artifacts"
    export GOLEM_HBM_DIR="$case_root/hbm"
    export GOLEM_TENSOR_DIR="$case_root/tensors"
    export GOLEM_SST_WORK_DIR="$case_root/work"
    export GOLEM_TEST_BINARY="$case_root/test_noc_dma"
    export GOLEM_BUILD_METADATA_FILE="$case_root/test_noc_dma.build.env"
    export GOLEM_RUN_SUMMARY_CSV="$case_root/run_summary.csv"
    export GOLEM_RUN_ID="$run_id"
    export GOLEM_GEMM_M="$DIM" GOLEM_GEMM_N="$DIM" GOLEM_GEMM_K="$DIM"
    export GOLEM_ORIG_M="$DIM" GOLEM_ORIG_N="$DIM" GOLEM_ORIG_K="$DIM"
    export GOLEM_GEMM_BLOCK_M=64 GOLEM_GEMM_BLOCK_N=64 GOLEM_GEMM_BLOCK_K=64
    export GOLEM_ARRAY_INPUT_SIZE=64 GOLEM_ARRAY_OUTPUT_SIZE=64
    export GOLEM_NUM_ARRAYS=64 GOLEM_TOTAL_CORES=20 GOLEM_TOTAL_GEMM_CORES=20
    export GOLEM_TOTAL_GROUPS=4 GOLEM_NUM_MEMORY_NODES=5
    export GOLEM_GLOBAL_STRIDE_KB=4096
    export GOLEM_A_REUSE_N_TILES="$reuse_n" GOLEM_B_REUSE_M_TILES="$reuse_m"
    export GOLEM_DMA_NODE_CREDITS=4 GOLEM_DMA_NODE_CHUNK_CREDITS="$credit"
    export GOLEM_DMA_SLOT_COUNT=16 GOLEM_DMA_WINDOW_K_TILES=4
    export GOLEM_WCP_PREFETCH_WINDOWS=1
    export GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE=1
    export GOLEM_WORKER_START_BARRIER_ENABLE=1 GOLEM_WORKER_START_GUARD_CYCLES=32768
    export GOLEM_CTRL_OVERLAP_AB=1 GOLEM_DMA_OVERLAP=0 GOLEM_DMA_STAGGER_CYCLES=0
    export GOLEM_SIM_MODE=full-timing GOLEM_TENSOR_SOURCE=synthetic
    export GOLEM_VERIFY_C=0 GOLEM_HBM_DUMP_OUTPUT=0 GOLEM_BENCH_QUIET_LOGS=1
    export GOLEM_MPI_PARTITIONER=sst.self GOLEM_EXPLICIT_PARTITION=1
    export GOLEM_MPI_ARGS="--bind-to cpu-list:ordered --cpu-list $cpu_list"
    export GOLEM_SST_THREADS=1
    taskset -c "$cpu_list" timeout "$TIMEOUT" "$PIPELINE" \
      --mpi-ranks "$MPI_RANKS" --sst-threads 1 --cpu-binding-mode batch_physical \
      --sim-mode full-timing --log "${run_id}.log"
  ) > "$case_root/pipeline.log" 2>&1
}

failed=0
for ((base = 0; base < ${#CASES[@]}; base += SLOTS)); do
  pids=()
  indices=()
  for ((offset = 0; offset < SLOTS && base + offset < ${#CASES[@]}; offset++)); do
    index=$((base + offset))
    run_case "$index" "${CASES[$index]}" "$offset" &
    pids+=("$!")
    indices+=("$index")
  done
  for pos in "${!pids[@]}"; do
    status=0
    wait "${pids[$pos]}" || status=$?
    index="${indices[$pos]}"
    spec="${CASES[$index]}"
    label="${spec%%:*}"
    case_root="$ROOT/cases/$(printf '%02d_%s' "$index" "$label")"
    IFS=: read -r _ reuse_n reuse_m credit <<< "$spec"
    cpu_start=$((pos * MPI_RANKS))
    cpu_list="${PHYSICAL_CPUS[$cpu_start]},${PHYSICAL_CPUS[$((cpu_start + 1))]},${PHYSICAL_CPUS[$((cpu_start + 2))]},${PHYSICAL_CPUS[$((cpu_start + 3))]}"
    if (( status == 0 )); then state=PASS; else state=FAIL; failed=1; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$index" "$label" "$state" "$case_root" "$cpu_list" "$reuse_n" "$reuse_m" "$credit" >> "$ROOT/status.tsv"
    echo "[$state] dim=$DIM case=$label root=$case_root"
  done
done

aggregate="$ROOT/run_summary.csv"
first=1
for ((index = 0; index < ${#CASES[@]}; index++)); do
  label="${CASES[$index]%%:*}"
  summary="$ROOT/cases/$(printf '%02d_%s' "$index" "$label")/run_summary.csv"
  if [[ -f "$summary" ]]; then
    if (( first )); then head -n 1 "$summary" > "$aggregate"; first=0; fi
    tail -n 1 "$summary" >> "$aggregate"
  fi
done

if (( first )); then
  echo "[ERROR] no successful run summaries; see $ROOT/status.tsv" >&2
  exit 1
fi
echo "[OK] dimension=$DIM aggregate=$aggregate status=$ROOT/status.tsv"
if (( failed )); then exit 1; fi
