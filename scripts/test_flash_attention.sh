#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/test_flash_attention.sh [options]

Run one Attention test from this worktree. The current performance and cycle
acceptance point is fixed at Sq=Skv=1024, Dh=128; dimension overrides are retained
for correctness-contract use only.

  --query-length N     Query sequence length Sq (default: 1024)
  --kv-length N        Shared K/V sequence length Skv (default: 1024)
  --num-query-heads N  Number of Query heads Hq (default: 1)
  --num-kv-heads N     Number of shared K/V heads Hkv (default: Hq)
  --queries/--keys     Deprecated aliases for query/KV length
  --query-heads/--kv-heads
                       Deprecated aliases for the two head counts
  --heads N            Compatibility alias: set Query and K/V heads to N
  --head-dim N         Per-head dimension Dh: 64 or 128 (default: 128)
  --mpi-ranks N        SST MPI ranks: 1, 2, or 4 (default: 4)
  --timeout SEC        Test timeout (default: 7200)
  --artifact-root DIR  Output directory (default: /tmp/<case-id>[_mpiN])
  --baseline FILE      Optionally compare against an explicit frozen baseline
  --show-config        Print resolved parameters without running the test

The public runner fixes the verified sequential-64 dataflow and its QK/PV
broadcast, streaming K/V, SFU, and active-K/V-tile PV configuration.
USAGE
}

QUERY_LENGTH=1024
KV_LENGTH=1024
NUM_QUERY_HEADS=1
NUM_KV_HEADS=1
HEAD_DIM=128
TIMEOUT=7200
MPI_RANKS=4
ARTIFACT_ROOT=""
BASELINE_JSON=""
SHOW_CONFIG=0
GOLEM_MATRIX_BROADCAST_MAX_FANOUT=16
GOLEM_MATRIX_BROADCAST_BYTES_PER_CYCLE=256
GOLEM_INPUT_SCATTER_BYTES_PER_CYCLE=256
GOLEM_MATRIX_BROADCAST_BASE_LATENCY_CYCLES=1
GOLEM_MATRIX_BROADCAST_STAGE_LATENCY_CYCLES=1
GOLEM_ATTENTION_PV_V_TILE_BUFFER_BYTES="${GOLEM_ATTENTION_PV_V_TILE_BUFFER_BYTES:-16384}"
GOLEM_ATTENTION_PV_V_TILE_BUFFER_HIT_TICKS="${GOLEM_ATTENTION_PV_V_TILE_BUFFER_HIT_TICKS:-1}"
GOLEM_ATTENTION_PV_V_TILE_BUFFER_BYTES_PER_CYCLE="${GOLEM_ATTENTION_PV_V_TILE_BUFFER_BYTES_PER_CYCLE:-64}"
GOLEM_ARRAY_MAC_PER_CU_PER_CYCLE="${GOLEM_ARRAY_MAC_PER_CU_PER_CYCLE:-1.0}"
GOLEM_ARRAY_PIPELINE_DEPTH="${GOLEM_ARRAY_PIPELINE_DEPTH:-2}"
KV_PAIR_REUSE="${GOLEM_ATTENTION_KV_PAIR_REUSE:-1}"
KV_PAIR_REUSE_EXPLICIT=0
if [[ -n "${GOLEM_ATTENTION_KV_PAIR_REUSE+x}" ]]; then
  KV_PAIR_REUSE_EXPLICIT=1
fi
KV_QUERY_GROUP_SIZE="${GOLEM_ATTENTION_KV_QUERY_GROUP_SIZE:-4}"
KV_QUERY_GROUP_SIZE_EXPLICIT=0
if [[ -n "${GOLEM_ATTENTION_KV_QUERY_GROUP_SIZE+x}" ]]; then
  KV_QUERY_GROUP_SIZE_EXPLICIT=1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --query-length|--kv-length|--num-query-heads|--num-kv-heads|--queries|--keys|--heads|--query-heads|--kv-heads|--head-dim|--timeout|--mpi-ranks|--artifact-root|--baseline)
      option="$1"
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $option" >&2
        usage >&2
        exit 2
      fi
      case "$option" in
        --query-length|--queries) QUERY_LENGTH="$2" ;;
        --kv-length|--keys) KV_LENGTH="$2" ;;
        --heads) NUM_QUERY_HEADS="$2"; NUM_KV_HEADS="$2" ;;
        --num-query-heads|--query-heads) NUM_QUERY_HEADS="$2" ;;
        --num-kv-heads|--kv-heads) NUM_KV_HEADS="$2" ;;
        --head-dim) HEAD_DIM="$2" ;;
        --timeout) TIMEOUT="$2" ;;
        --mpi-ranks) MPI_RANKS="$2" ;;
        --artifact-root) ARTIFACT_ROOT="$2" ;;
        --baseline) BASELINE_JSON="$2" ;;
      esac
      shift 2
      ;;
    --show-config) SHOW_CONFIG=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for pair in "query-length:$QUERY_LENGTH" "kv-length:$KV_LENGTH" \
            "num-query-heads:$NUM_QUERY_HEADS" "num-kv-heads:$NUM_KV_HEADS" \
            "head-dim:$HEAD_DIM" \
            "timeout:$TIMEOUT"; do
  name="${pair%%:*}"
  value="${pair#*:}"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "--$name must be a positive integer" >&2
    exit 2
  fi
done
for value in "$QUERY_LENGTH" "$KV_LENGTH" "$NUM_QUERY_HEADS" "$NUM_KV_HEADS" "$HEAD_DIM"; do
  if (( ${#value} > 10 )) ||
      { (( ${#value} == 10 )) && [[ "$value" > "4294967295" ]]; }; then
    echo "Attention dimensions must fit uint32" >&2
    exit 2
  fi
done
if (( QUERY_LENGTH % 256 != 0 )); then
  echo "--query-length must be divisible by 256 for the fixed four-manager/four-worker mapping" >&2
  exit 2
fi
if (( KV_LENGTH % 128 != 0 )); then
  echo "--kv-length must be divisible by 128 for four striped K/V memory nodes" >&2
  exit 2
fi
if (( HEAD_DIM != 64 && HEAD_DIM != 128 )); then
  echo "--head-dim must be 64 or 128 for the Attention workload" >&2
  exit 2
fi
if (( NUM_QUERY_HEADS > 1024 || NUM_KV_HEADS > 1024 )); then
  echo "head counts must not exceed 1024" >&2
  exit 2
fi
if (( NUM_QUERY_HEADS % NUM_KV_HEADS != 0 )); then
  echo "--num-query-heads must be divisible by --num-kv-heads" >&2
  exit 2
fi
GQA_GROUP_SIZE=$((NUM_QUERY_HEADS / NUM_KV_HEADS))
if (( GQA_GROUP_SIZE != 1 && GQA_GROUP_SIZE != 2 && GQA_GROUP_SIZE != 4 )); then
  echo "GQA group size must be 1, 2, or 4 for the four-worker mapping" >&2
  exit 2
fi
if [[ "$MPI_RANKS" != "1" && "$MPI_RANKS" != "2" && "$MPI_RANKS" != "4" ]]; then
  echo "--mpi-ranks must be 1, 2, or 4" >&2
  exit 2
fi
if [[ "$KV_PAIR_REUSE" != 0 && "$KV_PAIR_REUSE" != 1 ]]; then
  echo "GOLEM_ATTENTION_KV_PAIR_REUSE must be 0 or 1" >&2
  exit 2
fi
if [[ "$KV_QUERY_GROUP_SIZE" != 1 && "$KV_QUERY_GROUP_SIZE" != 2 &&
      "$KV_QUERY_GROUP_SIZE" != 4 ]]; then
  echo "GOLEM_ATTENTION_KV_QUERY_GROUP_SIZE must be 1, 2, or 4" >&2
  exit 2
fi
if (( KV_QUERY_GROUP_SIZE == 1 )); then
  if (( KV_PAIR_REUSE && KV_PAIR_REUSE_EXPLICIT )); then
    echo "K/V pair reuse requires query group size 2 or 4" >&2
    exit 2
  fi
  KV_PAIR_REUSE=0
elif (( KV_QUERY_GROUP_SIZE_EXPLICIT && !KV_PAIR_REUSE_EXPLICIT )); then
  KV_PAIR_REUSE=1
fi

MANAGER_QUERY_ROWS=$((QUERY_LENGTH / 4))
if (( MANAGER_QUERY_ROWS * HEAD_DIM * 4 > 1024 * 1024 )); then
  echo "query band exceeds the 1 MiB per-manager tensor window" >&2
  exit 2
fi
if (( (KV_LENGTH / 4) * HEAD_DIM * 4 > 1024 * 1024 )); then
  echo "K/V shard exceeds the 1 MiB per-memory-node tensor window" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
WORKTREE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
ATTENTION_DIR="$WORKTREE_ROOT/src/sst/elements/golem/tests/small/muticore_attention"
CASE_ID="fused_attention_q${QUERY_LENGTH}_k${KV_LENGTH}_d${HEAD_DIM}"
if (( NUM_QUERY_HEADS > 1 || NUM_KV_HEADS > 1 )); then
  CASE_ID="${CASE_ID}_hq${NUM_QUERY_HEADS}_hkv${NUM_KV_HEADS}"
fi
if [[ -z "$ARTIFACT_ROOT" ]]; then
  ARTIFACT_ROOT="${TMPDIR:-/tmp}/$CASE_ID"
  if (( MPI_RANKS > 1 )); then ARTIFACT_ROOT="${ARTIFACT_ROOT}_mpi${MPI_RANKS}"; fi
fi

DISPLAY_PV_V_TILE_REUSE=1
DISPLAY_PV_INPUT_PIPELINE=1
DISPLAY_PV_RESTORE_PIPELINE=1
DISPLAY_PV_OUTPUT_PIPELINE=1
DISPLAY_PV_EARLY_COMPUTE=1
DISPLAY_PV_MATRIX_SOFTMAX_OVERLAP=1
DISPLAY_PV_V_TILE_REUSE=0
DISPLAY_PV_INPUT_PIPELINE=0
DISPLAY_PV_RESTORE_PIPELINE=0
DISPLAY_PV_OUTPUT_PIPELINE=0
DISPLAY_PV_EARLY_COMPUTE=0
DISPLAY_PV_MATRIX_SOFTMAX_OVERLAP=0
GOLEM_MATRIX_BROADCAST_MAX_FANOUT=64
KV_PAIR_REUSE=0
KV_QUERY_GROUP_SIZE=1
DISPLAY_QK_SCORE_ROW_BURST=0

if [[ "$SHOW_CONFIG" == "1" ]]; then
  printf '%s\n' \
    "CASE_ID=$CASE_ID" \
    "QUERY_LENGTH=$QUERY_LENGTH" \
    "KV_LENGTH=$KV_LENGTH" \
    "QUERIES=$QUERY_LENGTH" \
    "KEYS=$KV_LENGTH" \
    "HEADS=$NUM_QUERY_HEADS" \
    "NUM_QUERY_HEADS=$NUM_QUERY_HEADS" \
    "NUM_KV_HEADS=$NUM_KV_HEADS" \
    "QUERY_HEADS=$NUM_QUERY_HEADS" \
    "KV_HEADS=$NUM_KV_HEADS" \
    "GQA_GROUP_SIZE=$GQA_GROUP_SIZE" \
    "HEAD_DIM=$HEAD_DIM" \
    "MANAGER_QUERY_ROWS=$MANAGER_QUERY_ROWS" \
    "MANAGER_QUERIES=$MANAGER_QUERY_ROWS" \
    "TIMEOUT=$TIMEOUT" \
    "MPI_RANKS=$MPI_RANKS" \
    "BASELINE_JSON=$BASELINE_JSON" \
    "ARTIFACT_ROOT=$ARTIFACT_ROOT" \
    "GENERIC_GEMM_WCP=1" \
    "PV_MATRIX_BROADCAST=1" \
    "QK_MATRIX_BROADCAST=1" \
    "QK_SCORE_ROW_BURST=$DISPLAY_QK_SCORE_ROW_BURST" \
    "QK_PANEL_ROW_BURST=$DISPLAY_QK_SCORE_ROW_BURST" \
    "ATTENTION_DATAFLOW=sequential_64" \
    "KV_DOUBLE_BUFFER=1" \
    "KV_SECOND_LOOKAHEAD=1" \
    "KV_PAIR_REUSE=$KV_PAIR_REUSE" \
    "KV_QUERY_GROUP_SIZE=$KV_QUERY_GROUP_SIZE" \
    "PV_V_TILE_REUSE=$DISPLAY_PV_V_TILE_REUSE" \
    "PV_INPUT_PIPELINE=$DISPLAY_PV_INPUT_PIPELINE" \
    "PV_COMPACT_INPUT=1" \
    "PV_RESTORE_PIPELINE=$DISPLAY_PV_RESTORE_PIPELINE" \
    "PV_OUTPUT_PIPELINE=$DISPLAY_PV_OUTPUT_PIPELINE" \
    "PV_EARLY_COMPUTE=$DISPLAY_PV_EARLY_COMPUTE" \
    "PV_MATRIX_SOFTMAX_OVERLAP=$DISPLAY_PV_MATRIX_SOFTMAX_OVERLAP" \
    "PV_ACTIVE_K=1" \
    "GOLEM_ATTENTION_PV_V_TILE_BUFFER_BYTES=$GOLEM_ATTENTION_PV_V_TILE_BUFFER_BYTES" \
    "GOLEM_ATTENTION_PV_V_TILE_BUFFER_HIT_TICKS=$GOLEM_ATTENTION_PV_V_TILE_BUFFER_HIT_TICKS" \
    "GOLEM_ATTENTION_PV_V_TILE_BUFFER_BYTES_PER_CYCLE=$GOLEM_ATTENTION_PV_V_TILE_BUFFER_BYTES_PER_CYCLE" \
    "GOLEM_ARRAY_MAC_PER_CU_PER_CYCLE=$GOLEM_ARRAY_MAC_PER_CU_PER_CYCLE" \
    "GOLEM_ARRAY_PIPELINE_DEPTH=$GOLEM_ARRAY_PIPELINE_DEPTH" \
    "GOLEM_MATRIX_BROADCAST_MAX_FANOUT=$GOLEM_MATRIX_BROADCAST_MAX_FANOUT" \
    "GOLEM_MATRIX_BROADCAST_BYTES_PER_CYCLE=$GOLEM_MATRIX_BROADCAST_BYTES_PER_CYCLE" \
    "GOLEM_INPUT_SCATTER_BYTES_PER_CYCLE=$GOLEM_INPUT_SCATTER_BYTES_PER_CYCLE" \
    "GOLEM_MATRIX_BROADCAST_BASE_LATENCY_CYCLES=$GOLEM_MATRIX_BROADCAST_BASE_LATENCY_CYCLES" \
    "GOLEM_MATRIX_BROADCAST_STAGE_LATENCY_CYCLES=$GOLEM_MATRIX_BROADCAST_STAGE_LATENCY_CYCLES"
  exit 0
fi

BASELINE_ARGS=()
if [[ -n "$BASELINE_JSON" ]]; then
  if (( NUM_QUERY_HEADS != 1 || NUM_KV_HEADS != 1 )); then
    echo "[ERROR] Frozen single-head baselines require one Query and K/V head" >&2
    exit 1
  fi
  if [[ ! -f "$BASELINE_JSON" ]]; then
    echo "[ERROR] Baseline does not exist: $BASELINE_JSON" >&2
    exit 1
  fi
  python3 "$ATTENTION_DIR/verify_flash_attention_baseline.py" \
    --baseline "$BASELINE_JSON" --mpi-ranks "$MPI_RANKS" \
    --query-length "$QUERY_LENGTH" --kv-length "$KV_LENGTH" \
    --head-dim "$HEAD_DIM" \
    --preflight-only
  BASELINE_ARGS=(--baseline "$BASELINE_JSON")
fi

# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_local_install.sh"
export SST_LIB_PATH="$WORKTREE_ROOT/install/lib/sst-elements-library"
if [[ ! -f "$SST_LIB_PATH/libgolem.so" ]]; then
  echo "[ERROR] Missing local element library: $SST_LIB_PATH/libgolem.so" >&2
  echo "        Build it first with scripts/build_and_install_local.sh" >&2
  exit 1
fi

env -u GOLEM_ATTENTION_PV_INPUT_RESIDENCY \
    -u GOLEM_ATTENTION_O_ACCUMULATOR_CBUFFER \
    -u GOLEM_ATTENTION_KV_PAIR_REUSE \
    -u GOLEM_ATTENTION_KV_QUERY_GROUP_SIZE \
    -u GOLEM_WCP_GEMM_PROXY_QUEUE_DEPTH \
    -u GOLEM_WCP_GEMM_PROXY_ISSUE_WIDTH \
    -u GOLEM_WCP_GEMM_PROXY_COMMAND_LATENCY_CYCLES \
    -u GOLEM_WCP_GEMM_PROXY_COMPLETION_LATENCY_CYCLES \
  python3 -m unittest \
  "$ATTENTION_DIR/test_flash_attention_baseline_contract.py" \
  "$ATTENTION_DIR/test_attention_metrics_report.py"
bash -n "$ATTENTION_DIR/run_flash_attention.sh" "$ATTENTION_DIR/run_fused_attention_scale.sh"

echo "[ATTENTION] Running Hq=$NUM_QUERY_HEADS Hkv=$NUM_KV_HEADS Sq=$QUERY_LENGTH Skv=$KV_LENGTH Dh=$HEAD_DIM with $MPI_RANKS MPI rank(s)"
GOLEM_MPI_RANKS="$MPI_RANKS" \
GOLEM_MATRIX_BROADCAST_MAX_FANOUT="$GOLEM_MATRIX_BROADCAST_MAX_FANOUT" \
GOLEM_MATRIX_BROADCAST_BYTES_PER_CYCLE="$GOLEM_MATRIX_BROADCAST_BYTES_PER_CYCLE" \
GOLEM_INPUT_SCATTER_BYTES_PER_CYCLE="$GOLEM_INPUT_SCATTER_BYTES_PER_CYCLE" \
GOLEM_MATRIX_BROADCAST_BASE_LATENCY_CYCLES="$GOLEM_MATRIX_BROADCAST_BASE_LATENCY_CYCLES" \
GOLEM_MATRIX_BROADCAST_STAGE_LATENCY_CYCLES="$GOLEM_MATRIX_BROADCAST_STAGE_LATENCY_CYCLES" \
GOLEM_ATTENTION_KV_PAIR_REUSE="$KV_PAIR_REUSE" \
GOLEM_ATTENTION_KV_QUERY_GROUP_SIZE="$KV_QUERY_GROUP_SIZE" \
GOLEM_ATTENTION_SEQUENTIAL_64_ENABLE=1 \
  "$ATTENTION_DIR/run_flash_attention.sh" \
  --query-length "$QUERY_LENGTH" --kv-length "$KV_LENGTH" \
  --num-query-heads "$NUM_QUERY_HEADS" --num-kv-heads "$NUM_KV_HEADS" \
  --head-dim "$HEAD_DIM" \
  --timeout "$TIMEOUT" --artifact-root "$ARTIFACT_ROOT" \
  --generic-gemm --pv-matrix-broadcast --qk-matrix-broadcast --kv-double-buffer \
  --kv-second-lookahead --no-qk-score-row-burst --sequential-64 \
  "${BASELINE_ARGS[@]}"

echo "[ATTENTION] $CASE_ID PASS"
