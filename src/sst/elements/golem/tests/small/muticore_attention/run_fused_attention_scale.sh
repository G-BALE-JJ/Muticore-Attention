#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=attention_terminal_ui.sh
source "$SCRIPT_DIR/attention_terminal_ui.sh"
TESTS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKTREE_ROOT="$(cd "$TESTS_DIR/../../../../.." && pwd -P)"
BASE_RUNNER="$TESTS_DIR/run_noc_dma_pipeline.sh"
LOCAL_ELEMENT_LIB="$WORKTREE_ROOT/install/lib/sst-elements-library"
ARTIFACT_ROOT=""
BASELINE_JSON=""
TIMEOUT_SECONDS=7200
DRY_RUN=0
ATTENTION_CLUSTER="${GOLEM_ATTENTION_CLUSTER_ENABLE:-0}"
ATTENTION_CLUSTER_QK_ARRAYS="${GOLEM_ATTENTION_CLUSTER_QK_ARRAYS:-16}"
GENERIC_GEMM=1
PV_MATRIX_BROADCAST=1
QK_MATRIX_BROADCAST=1
QK_DATAFLOW_TRANSPOSE=0
QK_EARLY_COMPUTE=1
QK_INPUT_PIPELINE="${GOLEM_ATTENTION_QK_INPUT_PIPELINE:-1}"
QK_INPUT_PIPELINE_EXPLICIT=0
if [[ -n "${GOLEM_ATTENTION_QK_INPUT_PIPELINE+x}" ]]; then
  QK_INPUT_PIPELINE_EXPLICIT=1
fi
QK_READOUT_OVERLAP="${GOLEM_ATTENTION_QK_READOUT_OVERLAP:-0}"
QK_READOUT_WINDOW="${GOLEM_ATTENTION_QK_READOUT_WINDOW:-2}"
QK_PANEL_ROW_BURST="${GOLEM_ATTENTION_QK_PANEL_ROW_BURST:-1}"
QK_PANEL_ROW_BURST_EXPLICIT=0
if [[ -n "${GOLEM_ATTENTION_QK_PANEL_ROW_BURST+x}" ]]; then
  QK_PANEL_ROW_BURST_EXPLICIT=1
fi
CROSS_TILE_OPERAND_PIPELINE="${GOLEM_ATTENTION_CROSS_TILE_OPERAND_PIPELINE:-1}"
CROSS_TILE_OPERAND_PIPELINE_EXPLICIT=0
if [[ -n "${GOLEM_ATTENTION_CROSS_TILE_OPERAND_PIPELINE+x}" ]]; then
  CROSS_TILE_OPERAND_PIPELINE_EXPLICIT=1
fi
KV_TILE_ROTATION=0
KV_DOUBLE_BUFFER=1
KV_BUFFER_COUNT="${GOLEM_ATTENTION_KV_BUFFER_COUNT:-2}"
KV_DISTRIBUTION="${GOLEM_ATTENTION_KV_DISTRIBUTION_ENABLE:-0}"
KV_DISTRIBUTION_EXPLICIT=0
if [[ -n "${GOLEM_ATTENTION_KV_DISTRIBUTION_ENABLE+x}" ]]; then
  KV_DISTRIBUTION_EXPLICIT=1
fi
KV_DISTRIBUTION_SLOTS="${GOLEM_ATTENTION_KV_DISTRIBUTION_SLOTS:-2}"
KV_DISTRIBUTION_SCRATCH_OFFSET="${GOLEM_ATTENTION_KV_DISTRIBUTION_SCRATCH_OFFSET:-0x40000}"
KV_MANAGER_LOOKAHEAD="${GOLEM_ATTENTION_KV_MANAGER_LOOKAHEAD:-0}"
KV_MANAGER_LOOKAHEAD_EXPLICIT=0
if [[ -n "${GOLEM_ATTENTION_KV_MANAGER_LOOKAHEAD+x}" ]]; then
  KV_MANAGER_LOOKAHEAD_EXPLICIT=1
fi
KEY_BLOCK_ROWS="${GOLEM_ATTENTION_KEY_BLOCK_ROWS:-0}"
KV_SECOND_LOOKAHEAD=1
KV_CROSS_QUERY_PREFETCH=1
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
PV_V_TILE_REUSE=1
PV_V_TILE_GROUP_RETENTION="${GOLEM_ATTENTION_PV_V_TILE_GROUP_RETENTION:-1}"
PV_INPUT_PIPELINE="${GOLEM_ATTENTION_PV_INPUT_PIPELINE:-1}"
PV_INPUT_PIPELINE_EXPLICIT=0
if [[ -n "${GOLEM_ATTENTION_PV_INPUT_PIPELINE+x}" ]]; then
  PV_INPUT_PIPELINE_EXPLICIT=1
fi
CLUSTER_PV_ROW_WAVEFRONT="${GOLEM_ATTENTION_CLUSTER_PV_ROW_WAVEFRONT:-0}"
CLUSTER_PV_ROW_WAVEFRONT_EXPLICIT=0
if [[ -n "${GOLEM_ATTENTION_CLUSTER_PV_ROW_WAVEFRONT+x}" ]]; then
  CLUSTER_PV_ROW_WAVEFRONT_EXPLICIT=1
fi
CLUSTER_QK_MATRIX_LOOKAHEAD="${GOLEM_ATTENTION_CLUSTER_QK_MATRIX_LOOKAHEAD:-0}"
CLUSTER_QK_MATRIX_LOOKAHEAD_EXPLICIT=0
if [[ -n "${GOLEM_ATTENTION_CLUSTER_QK_MATRIX_LOOKAHEAD+x}" ]]; then
  CLUSTER_QK_MATRIX_LOOKAHEAD_EXPLICIT=1
fi
CLUSTER_PV_MATRIX_LOOKAHEAD="${GOLEM_ATTENTION_CLUSTER_PV_MATRIX_LOOKAHEAD:-0}"
CLUSTER_PV_MATRIX_LOOKAHEAD_EXPLICIT=0
if [[ -n "${GOLEM_ATTENTION_CLUSTER_PV_MATRIX_LOOKAHEAD+x}" ]]; then
  CLUSTER_PV_MATRIX_LOOKAHEAD_EXPLICIT=1
fi
PV_COMPACT_INPUT=1
PV_INPUT_RESIDENCY="${GOLEM_ATTENTION_PV_INPUT_RESIDENCY:-1}"
PV_INPUT_RESIDENCY_EXPLICIT=0
if [[ -n "${GOLEM_ATTENTION_PV_INPUT_RESIDENCY+x}" ]]; then
  PV_INPUT_RESIDENCY_EXPLICIT=1
fi
O_ACCUMULATOR_CBUFFER="${GOLEM_ATTENTION_O_ACCUMULATOR_CBUFFER:-0}"
PV_RESTORE_PIPELINE=1
PV_OUTPUT_PIPELINE=1
PV_O_ROW_FUSION="${GOLEM_ATTENTION_PV_O_ROW_FUSION:-0}"
PV_O_ROW_FUSION_EXPLICIT=0
if [[ -n "${GOLEM_ATTENTION_PV_O_ROW_FUSION+x}" ]]; then
  PV_O_ROW_FUSION_EXPLICIT=1
fi
PV_EARLY_COMPUTE=1
PV_MATRIX_SOFTMAX_OVERLAP=1
PV_ACTIVE_K=1
MPI_RANKS="${GOLEM_MPI_RANKS:-1}"
MPI_PARTITIONER=sst.simple
ATTENTION_SST_ARGS="${GOLEM_ATTENTION_SST_ARGS:-}"
WCP_GEMM_PROXY_QUEUE_DEPTH="${GOLEM_WCP_GEMM_PROXY_QUEUE_DEPTH:-32}"
WCP_GEMM_PROXY_ISSUE_WIDTH="${GOLEM_WCP_GEMM_PROXY_ISSUE_WIDTH:-1}"
WCP_GEMM_PROXY_COMMAND_LATENCY_CYCLES="${GOLEM_WCP_GEMM_PROXY_COMMAND_LATENCY_CYCLES:-1}"
WCP_GEMM_PROXY_COMPLETION_LATENCY_CYCLES="${GOLEM_WCP_GEMM_PROXY_COMPLETION_LATENCY_CYCLES:-1}"
MATRIX_BROADCAST_MAX_FANOUT="${GOLEM_MATRIX_BROADCAST_MAX_FANOUT:-16}"
MATRIX_BROADCAST_BYTES_PER_CYCLE="${GOLEM_MATRIX_BROADCAST_BYTES_PER_CYCLE:-64}"
MATRIX_BROADCAST_BASE_LATENCY_CYCLES="${GOLEM_MATRIX_BROADCAST_BASE_LATENCY_CYCLES:-1}"
MATRIX_BROADCAST_STAGE_LATENCY_CYCLES="${GOLEM_MATRIX_BROADCAST_STAGE_LATENCY_CYCLES:-1}"
DMA_RESPONSE_VN="${GOLEM_DMA_RESPONSE_VN:-1}"
ATTENTION_MILESTONE_TRACE="${GOLEM_ATTENTION_MILESTONE_TRACE:-1}"
ATTENTION_TILE_TRACE="${GOLEM_ATTENTION_TILE_TRACE:-0}"
ATTENTION_TERMINAL_VERBOSE="${GOLEM_ATTENTION_TERMINAL_VERBOSE:-0}"
ARRAY_MAC_PER_CU_PER_CYCLE="${GOLEM_ARRAY_MAC_PER_CU_PER_CYCLE:-1}"
ARRAY_PIPELINE_DEPTH="${GOLEM_ARRAY_PIPELINE_DEPTH:-2}"
ARRAY_OUTPUT_READ_CREDITS="${GOLEM_ARRAY_OUTPUT_READ_CREDITS:-1}"
ARRAY_OUTPUT_READ_BANKS="${GOLEM_ARRAY_OUTPUT_READ_BANKS:-1}"
ARRAY_BUFFER_PORTS="${GOLEM_ARRAY_BUFFER_PORTS:-1}"
ARRAY_BUFFER_BASE_LATENCY_CYCLES="${GOLEM_ARRAY_BUFFER_BASE_LATENCY_CYCLES:-1}"
LOCAL_GM_READ_PORTS="${GOLEM_LOCAL_GM_READ_PORTS:-1}"
NEAR_ARRAY_OUTPUT_BYTES_PER_CYCLE="${GOLEM_ATTENTION_NEAR_ARRAY_OUTPUT_BYTES_PER_CYCLE:-512}"
NEAR_ARRAY_OUTPUT_CREDITS="${GOLEM_ATTENTION_NEAR_ARRAY_OUTPUT_CREDITS:-2}"
PV_V_TILE_BUFFER_BYTES="${GOLEM_ATTENTION_PV_V_TILE_BUFFER_BYTES:-16384}"
PV_V_TILE_BUFFER_HIT_TICKS="${GOLEM_ATTENTION_PV_V_TILE_BUFFER_HIT_TICKS:-1}"
PV_V_TILE_BUFFER_BYTES_PER_CYCLE="${GOLEM_ATTENTION_PV_V_TILE_BUFFER_BYTES_PER_CYCLE:-64}"
ATTENTION_TILE_STORAGE_BANKS="${GOLEM_ATTENTION_TILE_STORAGE_BANKS:-16}"
ATTENTION_TILE_STORAGE_BANK_BPC="${GOLEM_ATTENTION_TILE_STORAGE_BANK_BPC:-64}"
TOTAL_QUERIES=1024
KEYS=1024
HEAD_DIM=128

while [[ $# -gt 0 ]]; do
  case "$1" in
    --artifact-root|--baseline|--timeout|--queries|--keys|--head-dim|--key-block-rows|--kv-query-group-size)
      option="$1"
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $option" >&2
        exit 2
      fi
      case "$option" in
        --artifact-root) ARTIFACT_ROOT="$2" ;;
        --baseline) BASELINE_JSON="$2" ;;
        --timeout) TIMEOUT_SECONDS="$2" ;;
        --queries) TOTAL_QUERIES="$2" ;;
        --keys) KEYS="$2" ;;
        --head-dim) HEAD_DIM="$2" ;;
        --key-block-rows) KEY_BLOCK_ROWS="$2" ;;
        --kv-query-group-size)
          KV_QUERY_GROUP_SIZE="$2"
          KV_QUERY_GROUP_SIZE_EXPLICIT=1
          ;;
      esac
      shift 2
      ;;
    --pv-matrix-broadcast) PV_MATRIX_BROADCAST=1; shift ;;
    --no-pv-matrix-broadcast) PV_MATRIX_BROADCAST=0; shift ;;
    --qk-matrix-broadcast) QK_MATRIX_BROADCAST=1; shift ;;
    --no-qk-matrix-broadcast) QK_MATRIX_BROADCAST=0; shift ;;
    --qk-dataflow-transpose) QK_DATAFLOW_TRANSPOSE=1; QK_MATRIX_BROADCAST=1; shift ;;
    --qk-early-compute) QK_EARLY_COMPUTE=1; shift ;;
    --no-qk-early-compute) QK_EARLY_COMPUTE=0; shift ;;
    --qk-input-pipeline) QK_INPUT_PIPELINE=1; QK_INPUT_PIPELINE_EXPLICIT=1; shift ;;
    --no-qk-input-pipeline) QK_INPUT_PIPELINE=0; QK_INPUT_PIPELINE_EXPLICIT=1; shift ;;
    --qk-readout-overlap) QK_READOUT_OVERLAP=1; shift ;;
    --no-qk-readout-overlap) QK_READOUT_OVERLAP=0; shift ;;
    --qk-panel-row-burst) QK_PANEL_ROW_BURST=1; QK_PANEL_ROW_BURST_EXPLICIT=1; shift ;;
    --no-qk-panel-row-burst) QK_PANEL_ROW_BURST=0; QK_PANEL_ROW_BURST_EXPLICIT=1; shift ;;
    --cross-tile-operand-pipeline) CROSS_TILE_OPERAND_PIPELINE=1; CROSS_TILE_OPERAND_PIPELINE_EXPLICIT=1; shift ;;
    --no-cross-tile-operand-pipeline) CROSS_TILE_OPERAND_PIPELINE=0; CROSS_TILE_OPERAND_PIPELINE_EXPLICIT=1; shift ;;
    --kv-tile-rotation) KV_TILE_ROTATION=1; shift ;;
    --kv-double-buffer) KV_DOUBLE_BUFFER=1; shift ;;
    --no-kv-double-buffer) KV_DOUBLE_BUFFER=0; shift ;;
    --kv-distribution) KV_DISTRIBUTION=1; shift ;;
    --no-kv-distribution) KV_DISTRIBUTION=0; shift ;;
    --kv-manager-lookahead) KV_MANAGER_LOOKAHEAD=1; KV_MANAGER_LOOKAHEAD_EXPLICIT=1; shift ;;
    --no-kv-manager-lookahead) KV_MANAGER_LOOKAHEAD=0; KV_MANAGER_LOOKAHEAD_EXPLICIT=1; shift ;;
    --kv-second-lookahead) KV_SECOND_LOOKAHEAD=1; shift ;;
    --no-kv-second-lookahead) KV_SECOND_LOOKAHEAD=0; shift ;;
    --kv-cross-query-prefetch) KV_CROSS_QUERY_PREFETCH=1; shift ;;
    --no-kv-cross-query-prefetch) KV_CROSS_QUERY_PREFETCH=0; shift ;;
    --kv-pair-reuse) KV_PAIR_REUSE=1; KV_PAIR_REUSE_EXPLICIT=1; shift ;;
    --no-kv-pair-reuse) KV_PAIR_REUSE=0; KV_PAIR_REUSE_EXPLICIT=1; shift ;;
    --pv-v-tile-reuse) PV_V_TILE_REUSE=1; shift ;;
    --no-pv-v-tile-reuse) PV_V_TILE_REUSE=0; shift ;;
    --pv-v-tile-group-retention) PV_V_TILE_GROUP_RETENTION=1; shift ;;
    --no-pv-v-tile-group-retention) PV_V_TILE_GROUP_RETENTION=0; shift ;;
    --pv-input-pipeline) PV_INPUT_PIPELINE=1; PV_INPUT_PIPELINE_EXPLICIT=1; shift ;;
    --no-pv-input-pipeline) PV_INPUT_PIPELINE=0; PV_INPUT_PIPELINE_EXPLICIT=1; shift ;;
    --cluster-pv-row-wavefront) CLUSTER_PV_ROW_WAVEFRONT=1; CLUSTER_PV_ROW_WAVEFRONT_EXPLICIT=1; shift ;;
    --no-cluster-pv-row-wavefront) CLUSTER_PV_ROW_WAVEFRONT=0; CLUSTER_PV_ROW_WAVEFRONT_EXPLICIT=1; shift ;;
    --cluster-qk-matrix-lookahead) CLUSTER_QK_MATRIX_LOOKAHEAD=1; CLUSTER_QK_MATRIX_LOOKAHEAD_EXPLICIT=1; shift ;;
    --no-cluster-qk-matrix-lookahead) CLUSTER_QK_MATRIX_LOOKAHEAD=0; CLUSTER_QK_MATRIX_LOOKAHEAD_EXPLICIT=1; shift ;;
    --cluster-pv-matrix-lookahead) CLUSTER_PV_MATRIX_LOOKAHEAD=1; CLUSTER_PV_MATRIX_LOOKAHEAD_EXPLICIT=1; shift ;;
    --no-cluster-pv-matrix-lookahead) CLUSTER_PV_MATRIX_LOOKAHEAD=0; CLUSTER_PV_MATRIX_LOOKAHEAD_EXPLICIT=1; shift ;;
    --pv-compact-input) PV_COMPACT_INPUT=1; shift ;;
    --no-pv-compact-input) PV_COMPACT_INPUT=0; shift ;;
    --pv-input-residency) PV_INPUT_RESIDENCY=1; PV_INPUT_RESIDENCY_EXPLICIT=1; shift ;;
    --no-pv-input-residency) PV_INPUT_RESIDENCY=0; PV_INPUT_RESIDENCY_EXPLICIT=1; shift ;;
    --o-accumulator-cbuffer) O_ACCUMULATOR_CBUFFER=1; shift ;;
    --no-o-accumulator-cbuffer) O_ACCUMULATOR_CBUFFER=0; shift ;;
    --pv-restore-pipeline) PV_RESTORE_PIPELINE=1; shift ;;
    --no-pv-restore-pipeline) PV_RESTORE_PIPELINE=0; shift ;;
    --pv-output-pipeline) PV_OUTPUT_PIPELINE=1; shift ;;
    --no-pv-output-pipeline) PV_OUTPUT_PIPELINE=0; shift ;;
    --pv-o-row-fusion) PV_O_ROW_FUSION=1; PV_O_ROW_FUSION_EXPLICIT=1; shift ;;
    --no-pv-o-row-fusion) PV_O_ROW_FUSION=0; PV_O_ROW_FUSION_EXPLICIT=1; shift ;;
    --pv-early-compute) PV_EARLY_COMPUTE=1; shift ;;
    --no-pv-early-compute) PV_EARLY_COMPUTE=0; shift ;;
    --pv-matrix-softmax-overlap) PV_MATRIX_SOFTMAX_OVERLAP=1; shift ;;
    --no-pv-matrix-softmax-overlap) PV_MATRIX_SOFTMAX_OVERLAP=0; shift ;;
    --pv-active-k) PV_ACTIVE_K=1; shift ;;
    --no-pv-active-k) PV_ACTIVE_K=0; shift ;;
    --attention-cluster)
      ATTENTION_CLUSTER=1
      if (( ! KV_DISTRIBUTION_EXPLICIT )); then
        KV_DISTRIBUTION=1
      fi
      GENERIC_GEMM=1
      KV_DOUBLE_BUFFER=1
      KV_PAIR_REUSE=1
      KV_QUERY_GROUP_SIZE=4
      QK_PANEL_ROW_BURST=0
      QK_INPUT_PIPELINE=0
      QK_READOUT_OVERLAP=0
      CROSS_TILE_OPERAND_PIPELINE=0
      PV_INPUT_PIPELINE=0
      PV_INPUT_RESIDENCY=0
      PV_RESTORE_PIPELINE=0
      PV_OUTPUT_PIPELINE=0
      PV_EARLY_COMPUTE=0
      PV_MATRIX_SOFTMAX_OVERLAP=0
      shift
      ;;
    --no-attention-cluster) ATTENTION_CLUSTER=0; shift ;;
    --generic-gemm) GENERIC_GEMM=1; shift ;;
    --direct-gemm) GENERIC_GEMM=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      echo "Cluster mode: [--attention-cluster|--no-attention-cluster]"
      echo "Key tile override: [--key-block-rows 32|64]"
      echo "Usage: run_fused_attention_scale.sh [--queries N] [--keys N] [--head-dim N] [--baseline FILE] [--generic-gemm|--direct-gemm] [--pv-matrix-broadcast|--no-pv-matrix-broadcast] [--qk-matrix-broadcast|--no-qk-matrix-broadcast] [--qk-dataflow-transpose] [--qk-early-compute|--no-qk-early-compute] [--qk-input-pipeline|--no-qk-input-pipeline] [--qk-readout-overlap|--no-qk-readout-overlap] [--qk-panel-row-burst|--no-qk-panel-row-burst] [--cross-tile-operand-pipeline|--no-cross-tile-operand-pipeline] [--kv-tile-rotation] [--kv-double-buffer|--no-kv-double-buffer] [--kv-distribution|--no-kv-distribution] [--kv-second-lookahead|--no-kv-second-lookahead] [--kv-cross-query-prefetch|--no-kv-cross-query-prefetch] [--kv-pair-reuse|--no-kv-pair-reuse] [--kv-query-group-size 1|2|4] [--pv-v-tile-reuse|--no-pv-v-tile-reuse] [--pv-v-tile-group-retention|--no-pv-v-tile-group-retention] [--pv-input-pipeline|--no-pv-input-pipeline] [--pv-compact-input|--no-pv-compact-input] [--pv-input-residency|--no-pv-input-residency] [--o-accumulator-cbuffer|--no-o-accumulator-cbuffer] [--pv-restore-pipeline|--no-pv-restore-pipeline] [--pv-output-pipeline|--no-pv-output-pipeline] [--pv-o-row-fusion|--no-pv-o-row-fusion] [--pv-early-compute|--no-pv-early-compute] [--pv-matrix-softmax-overlap|--no-pv-matrix-softmax-overlap] [--pv-active-k|--no-pv-active-k] [--artifact-root DIR] [--timeout SEC] [--dry-run]"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if (( ATTENTION_CLUSTER )); then
  GENERIC_GEMM=1
  KV_DOUBLE_BUFFER=1
  KV_PAIR_REUSE=1
  KV_QUERY_GROUP_SIZE=4
  QK_PANEL_ROW_BURST=0
  QK_INPUT_PIPELINE=0
  QK_READOUT_OVERLAP=0
  CROSS_TILE_OPERAND_PIPELINE=0
  PV_V_TILE_REUSE=1
  PV_V_TILE_GROUP_RETENTION=1
  if (( ! PV_INPUT_PIPELINE_EXPLICIT )); then
    PV_INPUT_PIPELINE=1
  fi
  if (( ! KV_MANAGER_LOOKAHEAD_EXPLICIT )); then
    KV_MANAGER_LOOKAHEAD=$KV_DISTRIBUTION
  fi
  if (( ! CLUSTER_QK_MATRIX_LOOKAHEAD_EXPLICIT )); then
    CLUSTER_QK_MATRIX_LOOKAHEAD=1
  fi
  if (( ! CLUSTER_PV_MATRIX_LOOKAHEAD_EXPLICIT )); then
    CLUSTER_PV_MATRIX_LOOKAHEAD=1
  fi
  PV_INPUT_RESIDENCY=0
  PV_RESTORE_PIPELINE=0
  PV_OUTPUT_PIPELINE=0
  if (( ! PV_O_ROW_FUSION_EXPLICIT )); then
    PV_O_ROW_FUSION=1
  fi
  PV_EARLY_COMPUTE=0
  PV_MATRIX_SOFTMAX_OVERLAP=0
  PV_ACTIVE_K=1
  PV_V_TILE_BUFFER_BYTES=32768
  ARRAY_BUFFER_PORTS=2
  LOCAL_GM_READ_PORTS=2
  ARRAY_OUTPUT_READ_CREDITS=8
  ARRAY_OUTPUT_READ_BANKS=8
fi

if [[ "$PV_O_ROW_FUSION" != 0 && "$PV_O_ROW_FUSION" != 1 ]]; then
  echo "GOLEM_ATTENTION_PV_O_ROW_FUSION must be 0 or 1" >&2
  exit 2
fi
if [[ "$CLUSTER_PV_ROW_WAVEFRONT" != 0 && "$CLUSTER_PV_ROW_WAVEFRONT" != 1 ]]; then
  echo "GOLEM_ATTENTION_CLUSTER_PV_ROW_WAVEFRONT must be 0 or 1" >&2
  exit 2
fi
if (( CLUSTER_PV_ROW_WAVEFRONT && ! ATTENTION_CLUSTER )); then
  echo "Cluster PV row wavefront requires --attention-cluster" >&2
  exit 2
fi
if [[ "$CLUSTER_QK_MATRIX_LOOKAHEAD" != 0 && "$CLUSTER_QK_MATRIX_LOOKAHEAD" != 1 ]]; then
  echo "GOLEM_ATTENTION_CLUSTER_QK_MATRIX_LOOKAHEAD must be 0 or 1" >&2
  exit 2
fi
if (( CLUSTER_QK_MATRIX_LOOKAHEAD && ! ATTENTION_CLUSTER )); then
  echo "Cluster QK matrix lookahead requires --attention-cluster" >&2
  exit 2
fi
if (( CLUSTER_PV_MATRIX_LOOKAHEAD && ! ATTENTION_CLUSTER )); then
  echo "PV matrix lookahead requires --attention-cluster" >&2
  exit 2
fi
if [[ "$KV_MANAGER_LOOKAHEAD" != 0 && "$KV_MANAGER_LOOKAHEAD" != 1 ]]; then
  echo "GOLEM_ATTENTION_KV_MANAGER_LOOKAHEAD must be 0 or 1" >&2
  exit 2
fi
if (( KV_MANAGER_LOOKAHEAD && ! KV_DISTRIBUTION )); then
  echo "Manager K/V lookahead requires Attention K/V distribution" >&2
  exit 2
fi
if (( PV_O_ROW_FUSION && ! ATTENTION_CLUSTER )); then
  echo "PV-to-O row fusion requires --attention-cluster" >&2
  exit 2
fi

if [[ "$CROSS_TILE_OPERAND_PIPELINE" != 0 &&
      "$CROSS_TILE_OPERAND_PIPELINE" != 1 ]]; then
  echo "GOLEM_ATTENTION_CROSS_TILE_OPERAND_PIPELINE must be 0 or 1" >&2
  exit 2
fi
if [[ "$ATTENTION_CLUSTER" != 0 && "$ATTENTION_CLUSTER" != 1 ]]; then
  echo "GOLEM_ATTENTION_CLUSTER_ENABLE must be 0 or 1" >&2
  exit 2
fi
if [[ "$KV_QUERY_GROUP_SIZE" != 1 && "$KV_QUERY_GROUP_SIZE" != 2 &&
      "$KV_QUERY_GROUP_SIZE" != 4 ]]; then
  echo "K/V query group size must be 1, 2, or 4" >&2
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

if (( !GENERIC_GEMM && !QK_PANEL_ROW_BURST_EXPLICIT )); then
  QK_PANEL_ROW_BURST=0
fi
if (( !GENERIC_GEMM && !PV_INPUT_RESIDENCY_EXPLICIT )); then
  PV_INPUT_RESIDENCY=0
fi
if (( QK_DATAFLOW_TRANSPOSE && !QK_INPUT_PIPELINE_EXPLICIT )); then
  QK_INPUT_PIPELINE=0
fi
if (( !GENERIC_GEMM && (PV_INPUT_RESIDENCY || O_ACCUMULATOR_CBUFFER) )); then
  echo "PV input residency and O accumulator C-buffer require generic GEMM/WCP" >&2
  exit 2
fi
if (( ATTENTION_CLUSTER && O_ACCUMULATOR_CBUFFER )); then
  echo "attention cluster O accumulator C-buffer is not compatible with group-4 pair reuse" >&2
  exit 2
fi
if (( KV_PAIR_REUSE && (!GENERIC_GEMM || !KV_DOUBLE_BUFFER ||
      O_ACCUMULATOR_CBUFFER ) )); then
  if (( KV_PAIR_REUSE_EXPLICIT )); then
    echo "K/V pair reuse requires generic GEMM/WCP and K/V double buffering, and is incompatible with O accumulator C-buffer" >&2
    exit 2
  fi
  KV_PAIR_REUSE=0
fi
if (( CROSS_TILE_OPERAND_PIPELINE &&
      (!GENERIC_GEMM || !QK_MATRIX_BROADCAST || QK_DATAFLOW_TRANSPOSE) )); then
  if (( CROSS_TILE_OPERAND_PIPELINE_EXPLICIT )); then
    echo "Cross-tile operand pipeline requires grouped reuse and generic, non-transposed broadcast QK" >&2
    exit 2
  fi
  CROSS_TILE_OPERAND_PIPELINE=0
fi
if (( CROSS_TILE_OPERAND_PIPELINE && !KV_PAIR_REUSE )); then
  if (( CROSS_TILE_OPERAND_PIPELINE_EXPLICIT )); then
    echo "Cross-tile operand pipeline requires grouped K/V reuse" >&2
    exit 2
  fi
  CROSS_TILE_OPERAND_PIPELINE=0
fi

if ! [[ "$MPI_RANKS" =~ ^[1-9][0-9]*$ ]]; then
  echo "GOLEM_MPI_RANKS must be a positive integer" >&2
  exit 2
fi
for trace_value in "$ATTENTION_MILESTONE_TRACE" "$ATTENTION_TILE_TRACE" \
                   "$ATTENTION_TERMINAL_VERBOSE" \
                   "$PV_V_TILE_GROUP_RETENTION"; do
  if [[ "$trace_value" != 0 && "$trace_value" != 1 ]]; then
    echo "Attention trace and query-group optimization settings must be 0 or 1" >&2
    exit 2
  fi
done
if (( PV_V_TILE_GROUP_RETENTION && !PV_V_TILE_REUSE )); then
  echo "PV V-tile group retention requires PV V-tile reuse" >&2
  exit 2
fi
if (( MPI_RANKS > 4 || 4 % MPI_RANKS != 0 )); then
  echo "GOLEM_MPI_RANKS must divide the four query-manager bands (supported: 1, 2, 4)" >&2
  exit 2
fi
if ! [[ "$DMA_RESPONSE_VN" =~ ^[0-2]$ ]]; then
  echo "GOLEM_DMA_RESPONSE_VN must be 0, 1, or 2" >&2
  exit 2
fi
if [[ "$QK_READOUT_OVERLAP" != 0 && "$QK_READOUT_OVERLAP" != 1 ]]; then
  echo "GOLEM_ATTENTION_QK_READOUT_OVERLAP must be 0 or 1" >&2
  exit 2
fi
if ! [[ "$QK_READOUT_WINDOW" =~ ^[1-9][0-9]*$ ]] ||
    (( 10#$QK_READOUT_WINDOW > 16 )); then
  echo "GOLEM_ATTENTION_QK_READOUT_WINDOW must be an integer from 1 through 16" >&2
  exit 2
fi
if (( QK_READOUT_OVERLAP && QK_DATAFLOW_TRANSPOSE )); then
  echo "QK readout overlap is not supported with transposed QK dataflow" >&2
  exit 2
fi
if [[ "$QK_INPUT_PIPELINE" != 0 && "$QK_INPUT_PIPELINE" != 1 ]]; then
  echo "GOLEM_ATTENTION_QK_INPUT_PIPELINE must be 0 or 1" >&2
  exit 2
fi
if (( QK_INPUT_PIPELINE && QK_DATAFLOW_TRANSPOSE )); then
  echo "QK input pipeline is not supported with transposed QK dataflow" >&2
  exit 2
fi
if [[ "$QK_PANEL_ROW_BURST" != 0 && "$QK_PANEL_ROW_BURST" != 1 ]]; then
  echo "GOLEM_ATTENTION_QK_PANEL_ROW_BURST must be 0 or 1" >&2
  exit 2
fi
if (( QK_PANEL_ROW_BURST &&
      (QK_DATAFLOW_TRANSPOSE || QK_READOUT_OVERLAP || !GENERIC_GEMM) )); then
  echo "QK panel row burst requires generic GEMM/WCP, non-transposed QK, and disabled QK readout overlap" >&2
  exit 2
fi
if ! [[ "$ATTENTION_TILE_STORAGE_BANKS" =~ ^[1-9][0-9]*$ ]] ||
    (( ${#ATTENTION_TILE_STORAGE_BANKS} > 2 )) ||
    (( 10#$ATTENTION_TILE_STORAGE_BANKS > 16 )); then
  echo "GOLEM_ATTENTION_TILE_STORAGE_BANKS must be an integer from 1 through 16" >&2
  exit 2
fi
if ! [[ "$ATTENTION_TILE_STORAGE_BANK_BPC" =~ ^[1-9][0-9]*$ ]]; then
  echo "GOLEM_ATTENTION_TILE_STORAGE_BANK_BPC must be a positive integer" >&2
  exit 2
fi
if (( GENERIC_GEMM )); then
  if ! [[ "$WCP_GEMM_PROXY_QUEUE_DEPTH" =~ ^[1-9][0-9]*$ ]] ||
      (( 10#$WCP_GEMM_PROXY_QUEUE_DEPTH < 16 )); then
    echo "GOLEM_WCP_GEMM_PROXY_QUEUE_DEPTH must be an integer >= 16 for a bounded Attention group launch" >&2
    exit 2
  fi
  if ! [[ "$WCP_GEMM_PROXY_ISSUE_WIDTH" =~ ^[1-9][0-9]*$ ]]; then
    echo "GOLEM_WCP_GEMM_PROXY_ISSUE_WIDTH must be a positive integer" >&2
    exit 2
  fi
  for latency in "$WCP_GEMM_PROXY_COMMAND_LATENCY_CYCLES" \
                 "$WCP_GEMM_PROXY_COMPLETION_LATENCY_CYCLES"; do
    if ! [[ "$latency" =~ ^[1-9][0-9]*$ ]]; then
      echo "WCP GEMM proxy latency values must be positive integers" >&2
      exit 2
    fi
  done
fi
if ! [[ "$MATRIX_BROADCAST_MAX_FANOUT" =~ ^[1-9][0-9]*$ ]] ||
    (( 10#$MATRIX_BROADCAST_MAX_FANOUT < 16 )); then
  echo "GOLEM_MATRIX_BROADCAST_MAX_FANOUT must be an integer >= 16" >&2
  exit 2
fi
if ! [[ "$MATRIX_BROADCAST_BYTES_PER_CYCLE" =~ ^[1-9][0-9]*$ ]]; then
  echo "GOLEM_MATRIX_BROADCAST_BYTES_PER_CYCLE must be a positive integer" >&2
  exit 2
fi
if ! [[ "$MATRIX_BROADCAST_BASE_LATENCY_CYCLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "GOLEM_MATRIX_BROADCAST_BASE_LATENCY_CYCLES must be a positive integer" >&2
  exit 2
fi
if ! [[ "$MATRIX_BROADCAST_STAGE_LATENCY_CYCLES" =~ ^[0-9]+$ ]]; then
  echo "GOLEM_MATRIX_BROADCAST_STAGE_LATENCY_CYCLES must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$PV_V_TILE_BUFFER_BYTES" =~ ^[1-9][0-9]*$ ]]; then
  echo "GOLEM_ATTENTION_PV_V_TILE_BUFFER_BYTES must be a positive integer" >&2
  exit 2
fi
if (( ${#PV_V_TILE_BUFFER_BYTES} > 6 )) ||
    (( 10#$PV_V_TILE_BUFFER_BYTES > 262144 )); then
  echo "GOLEM_ATTENTION_PV_V_TILE_BUFFER_BYTES must fit the 256 KiB local-GM window" >&2
  exit 2
fi
if [[ "$KV_DISTRIBUTION" != 0 && "$KV_DISTRIBUTION" != 1 ]]; then
  echo "GOLEM_ATTENTION_KV_DISTRIBUTION_ENABLE must be 0 or 1" >&2
  exit 2
fi
if (( KV_DISTRIBUTION && ! ATTENTION_CLUSTER )); then
  echo "Attention K/V distribution requires --attention-cluster" >&2
  exit 2
fi
if ! [[ "$KV_DISTRIBUTION_SLOTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "GOLEM_ATTENTION_KV_DISTRIBUTION_SLOTS must be positive" >&2
  exit 2
fi
if ! [[ "$PV_V_TILE_BUFFER_HIT_TICKS" =~ ^[0-9]+$ ]]; then
  echo "GOLEM_ATTENTION_PV_V_TILE_BUFFER_HIT_TICKS must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$PV_V_TILE_BUFFER_BYTES_PER_CYCLE" =~ ^[1-9][0-9]*$ ]]; then
  echo "GOLEM_ATTENTION_PV_V_TILE_BUFFER_BYTES_PER_CYCLE must be a positive integer" >&2
  exit 2
fi
if (( MPI_RANKS > 1 )); then
  MPI_PARTITIONER=sst.self
fi
if [[ " $ATTENTION_SST_ARGS " == *" --partitioner"* ||
      " $ATTENTION_SST_ARGS " == *" --lib-path"* ||
      " $ATTENTION_SST_ARGS " == *" --add-lib-path"* ||
      " $ATTENTION_SST_ARGS " == *" --timebase"* ]]; then
  echo "GOLEM_ATTENTION_SST_ARGS cannot override the partitioner, element library path, or timebase" >&2
  exit 2
fi
if [[ ! -f "$LOCAL_ELEMENT_LIB/libgolem.so" ]]; then
  echo "Missing local element library: $LOCAL_ELEMENT_LIB/libgolem.so" >&2
  echo "Build it first with scripts/build_and_install_local.sh" >&2
  exit 1
fi

for shape in "$TOTAL_QUERIES" "$KEYS" "$HEAD_DIM"; do
  if ! [[ "$shape" =~ ^[1-9][0-9]*$ ]]; then
    echo "Attention dimensions must be positive integers" >&2
    exit 2
  fi
done
for shape in "$TOTAL_QUERIES" "$KEYS" "$HEAD_DIM"; do
  if (( ${#shape} > 10 )) ||
      { (( ${#shape} == 10 )) && [[ "$shape" > "4294967295" ]]; }; then
    echo "Attention dimensions must fit uint32" >&2
    exit 2
  fi
done
if (( TOTAL_QUERIES % 256 != 0 )); then
  echo "--queries must be divisible by 256 for balanced worker placement" >&2
  exit 2
fi
if (( KEYS % 128 != 0 )); then
  echo "--keys must be divisible by 128 for four striped K/V memory nodes" >&2
  exit 2
fi
if (( HEAD_DIM != 64 && HEAD_DIM != 128 )); then
  echo "--head-dim must be 64 or 128" >&2
  exit 2
fi
if (( ATTENTION_CLUSTER && HEAD_DIM != 128 )); then
  echo "Attention cluster L2-L4 requires head dimension 128" >&2
  exit 2
fi
if [[ "$ATTENTION_CLUSTER_QK_ARRAYS" != 16 &&
      "$ATTENTION_CLUSTER_QK_ARRAYS" != 24 &&
      "$ATTENTION_CLUSTER_QK_ARRAYS" != 32 &&
      "$ATTENTION_CLUSTER_QK_ARRAYS" != 40 ]]; then
  echo "GOLEM_ATTENTION_CLUSTER_QK_ARRAYS must be 16, 24, 32, or 40" >&2
  exit 2
fi
MANAGER_QUERIES=$((TOTAL_QUERIES / 4))
if (( MANAGER_QUERIES * HEAD_DIM * 4 > 1024 * 1024 )); then
  echo "query band exceeds the 1 MiB per-manager tensor window" >&2
  exit 2
fi
if (( (KEYS / 4) * HEAD_DIM * 4 > 1024 * 1024 )); then
  echo "K/V shard exceeds the 1 MiB per-memory-node tensor window" >&2
  exit 2
fi
ARRAY_INPUT=$HEAD_DIM
ARRAY_OUTPUT=16
NUM_ARRAYS=16
if (( ATTENTION_CLUSTER )); then
  ARRAY_INPUT=64
  ARRAY_OUTPUT=64
  NUM_ARRAYS=64
fi
RUN_ID="fused_attention_q${TOTAL_QUERIES}_k${KEYS}_d${HEAD_DIM}"
GUEST_NAME=fused_attention

if (( !ATTENTION_CLUSTER )); then
  KEY_BLOCK_ROWS=32
elif (( KEY_BLOCK_ROWS == 0 )); then
  KEY_BLOCK_ROWS=$((KEYS <= 128 ? 64 : 32))
fi
if (( KEY_BLOCK_ROWS != 32 && KEY_BLOCK_ROWS != 64 )); then
  echo "--key-block-rows must be 32 or 64" >&2
  exit 2
fi
KV_BUFFER_COUNT_EFFECTIVE=$((KV_DOUBLE_BUFFER ? KV_BUFFER_COUNT : 1))
KV_DISTRIBUTION_TILE_BYTES=$((KEY_BLOCK_ROWS * HEAD_DIM * 4))
KV_DISTRIBUTION_SCRATCH_BYTES=$((2 * KV_DISTRIBUTION_SLOTS * KV_DISTRIBUTION_TILE_BYTES))
if (( KV_DISTRIBUTION )) &&
   (( KV_DISTRIBUTION_SCRATCH_OFFSET < 0x2000 ||
      KV_DISTRIBUTION_SCRATCH_OFFSET + KV_DISTRIBUTION_SCRATCH_BYTES > 1024 * 1024 - 64 )); then
  echo "Attention K/V distribution scratch exceeds the manager Local-GM window" >&2
  exit 2
fi
if (( ATTENTION_CLUSTER )); then
  ATTENTION_WINDOW_OFFSET=0xC0000
  ATTENTION_WINDOW_BYTES=$((128 + 5 * 16 * HEAD_DIM * 4 +
    16 * KEY_BLOCK_ROWS * 4 +
    2 * KV_BUFFER_COUNT_EFFECTIVE * KEY_BLOCK_ROWS * HEAD_DIM * 4))
else
  ATTENTION_WINDOW_OFFSET=0xC0000
  ATTENTION_WINDOW_BYTES=$((KV_DOUBLE_BUFFER ? 0x14880 : 0x10000))
fi
if (( KV_PAIR_REUSE && !ATTENTION_CLUSTER )); then
  ATTENTION_WINDOW_BYTES=$((ATTENTION_WINDOW_BYTES + \
    (KV_QUERY_GROUP_SIZE - 1) * 2 * 16 * HEAD_DIM * 4))
fi
if (( PV_V_TILE_REUSE && !ATTENTION_CLUSTER )); then
  ATTENTION_WINDOW_BYTES=$((ATTENTION_WINDOW_BYTES + PV_V_TILE_BUFFER_BYTES))
fi
if (( ATTENTION_WINDOW_BYTES > 0x40000 )); then
  echo "Attention window plus PV V-tile buffer exceeds the 256 KiB local-GM window" >&2
  exit 2
fi

ARTIFACT_ROOT="${ARTIFACT_ROOT:-${TMPDIR:-/tmp}/$RUN_ID}"
Q_FILE="$ARTIFACT_ROOT/q_${TOTAL_QUERIES}x${HEAD_DIM}.bin"
K_FILE="$ARTIFACT_ROOT/k_${KEYS}x${HEAD_DIM}.bin"
V_FILE="$ARTIFACT_ROOT/v_${KEYS}x${HEAD_DIM}.bin"
RESULT_JSON="$ARTIFACT_ROOT/fused_attention_result.json"
LIFECYCLE_JSON="$ARTIFACT_ROOT/attention_lifecycle.json"
METRICS_JSON="$ARTIFACT_ROOT/attention_metrics.json"
METRICS_CSV="$ARTIFACT_ROOT/attention_metrics.csv"
DRIVER_LOG_DIR="$ARTIFACT_ROOT/driver_logs"
DRIVER_MILESTONE_LOG="$DRIVER_LOG_DIR/milestones.log"
SST_LOG_BASENAME=attention-sst.log
SST_RUNTIME_LOG="$ARTIFACT_ROOT/logs/${SST_LOG_BASENAME%.log}_${RUN_ID}.log"
MPI_PARTITION_JSON="$ARTIFACT_ROOT/attention_mpi_partition.json"
MPI_PLACEMENT_JSON="$ARTIFACT_ROOT/attention_mpi_placement.json"
GUEST="$SCRIPT_DIR/riscv64/$GUEST_NAME"
HBM_DIR="$ARTIFACT_ROOT/hbm"
STATS_FILE="$ARTIFACT_ROOT/stats/overlap0/$RUN_ID/stats_selfcom.txt"
RUN_CONFIG_FILE="$ARTIFACT_ROOT/stats/overlap0/$RUN_ID/run_config.env"
BASELINE_RESULT_JSON="$ARTIFACT_ROOT/attention_baseline_verification.json"
MEM_NODE_SIZE=134217728
MODEL_PLATFORM_CLOCK="${VANADIS_CPU_CLOCK:-2.0GHz}"
NORMALIZATION_CLOCK="${GOLEM_ATTENTION_NORMALIZATION_CLOCK:-1.0GHz}"
Q_OFFSET=$((0x02000000))
K_OFFSET=$((0x02100000))
V_OFFSET=$((0x02200000))
O_OFFSET=$((0x02300000))

if [[ ! -x "$GUEST" ]]; then
  echo "Missing FlashAttention guest: $GUEST" >&2
  echo "Build it first with scripts/build_and_install_local.sh" >&2
  exit 1
fi

GENERATE_CMD=(python3 "$SCRIPT_DIR/attention_case.py" generate
  --queries "$TOTAL_QUERIES" --keys "$KEYS" --head-dim "$HEAD_DIM"
  --q-file "$Q_FILE" --k-file "$K_FILE" --v-file "$V_FILE")

RUN_CMD=(timeout "$TIMEOUT_SECONDS" env
  "PYTHONPATH=$TESTS_DIR${PYTHONPATH:+:$PYTHONPATH}"
  "SST_LIB_PATH=$LOCAL_ELEMENT_LIB"
  "GOLEM_SST_ARGS=--lib-path=$LOCAL_ELEMENT_LIB${ATTENTION_SST_ARGS:+ $ATTENTION_SST_ARGS}"
  "GOLEM_RUN_ID=$RUN_ID"
  "GOLEM_ARTIFACT_ROOT=$ARTIFACT_ROOT"
  "VANADIS_EXE=$GUEST"
  GOLEM_SKIP_DEFAULT_GUEST_BUILD=1
  GOLEM_ARCH_SCRIPT=architecture/archive/ncores_selfcom_dma.py
  GOLEM_ATTENTION_FUSED=1
  GOLEM_ATTENTION_HBM_STRIPED=1
  "GOLEM_ATTENTION_QUERY_BLOCK_MPI=$((MPI_RANKS > 1 ? 1 : 0))"
  "GOLEM_ATTENTION_PLACEMENT_FILE=$MPI_PLACEMENT_JSON"
  "GOLEM_ATTENTION_QUERIES=$TOTAL_QUERIES"
  "GOLEM_ATTENTION_KEYS=$KEYS"
  "GOLEM_ATTENTION_HEAD_DIM=$HEAD_DIM"
  "GOLEM_ATTENTION_GUEST_MANAGER_QUERIES=$MANAGER_QUERIES"
  "GOLEM_ATTENTION_GUEST_KEYS=$KEYS"
  "GOLEM_ATTENTION_GUEST_HEAD_DIM=$HEAD_DIM"
  "GOLEM_ATTENTION_GUEST_KEY_BLOCK_ROWS=$KEY_BLOCK_ROWS"
  "GOLEM_ATTENTION_Q_FILE=$Q_FILE"
  "GOLEM_ATTENTION_K_FILE=$K_FILE"
  "GOLEM_ATTENTION_V_FILE=$V_FILE"
  "GOLEM_ATTENTION_Q_OFFSET=$Q_OFFSET"
  "GOLEM_ATTENTION_K_OFFSET=$K_OFFSET"
  "GOLEM_ATTENTION_V_OFFSET=$V_OFFSET"
  "GOLEM_ATTENTION_WINDOW_OFFSET=$ATTENTION_WINDOW_OFFSET"
  "GOLEM_ATTENTION_WINDOW_BYTES=$ATTENTION_WINDOW_BYTES"
  "GOLEM_ATTENTION_CLUSTER_ENABLE=$ATTENTION_CLUSTER"
  "GOLEM_ATTENTION_CLUSTER_QK_ARRAYS=$ATTENTION_CLUSTER_QK_ARRAYS"
  "GOLEM_ATTENTION_QK_DATAFLOW_TRANSPOSE=$QK_DATAFLOW_TRANSPOSE"
  "GOLEM_ATTENTION_QK_MATRIX_BROADCAST=$QK_MATRIX_BROADCAST"
  "GOLEM_ATTENTION_PV_MATRIX_BROADCAST=$PV_MATRIX_BROADCAST"
  "GOLEM_ATTENTION_KV_TILE_ROTATION=$KV_TILE_ROTATION"
  "GOLEM_ATTENTION_QK_EARLY_COMPUTE=$QK_EARLY_COMPUTE"
  "GOLEM_ATTENTION_QK_INPUT_PIPELINE=$QK_INPUT_PIPELINE"
  "GOLEM_ATTENTION_QK_READOUT_OVERLAP=$QK_READOUT_OVERLAP"
  "GOLEM_ATTENTION_QK_READOUT_WINDOW=$QK_READOUT_WINDOW"
  "GOLEM_ATTENTION_QK_PANEL_ROW_BURST=$QK_PANEL_ROW_BURST"
  "GOLEM_ATTENTION_CROSS_TILE_OPERAND_PIPELINE=$CROSS_TILE_OPERAND_PIPELINE"
  "GOLEM_ARRAY_OPERAND_CONTEXT_BANKS=$(((CROSS_TILE_OPERAND_PIPELINE || ATTENTION_CLUSTER) ? 2 : 1))"
  "GOLEM_ATTENTION_KV_DOUBLE_BUFFER=$KV_DOUBLE_BUFFER"
  "GOLEM_ATTENTION_KV_BUFFER_COUNT=$KV_BUFFER_COUNT"
  "GOLEM_ATTENTION_KV_DISTRIBUTION_ENABLE=$KV_DISTRIBUTION"
  "GOLEM_ATTENTION_KV_MANAGER_LOOKAHEAD=$KV_MANAGER_LOOKAHEAD"
  "GOLEM_ATTENTION_KV_DISTRIBUTION_SLOTS=$KV_DISTRIBUTION_SLOTS"
  "GOLEM_ATTENTION_KV_DISTRIBUTION_TILE_BYTES=$KV_DISTRIBUTION_TILE_BYTES"
  "GOLEM_ATTENTION_KV_DISTRIBUTION_SCRATCH_OFFSET=$KV_DISTRIBUTION_SCRATCH_OFFSET"
  GOLEM_ATTENTION_KV_DISTRIBUTION_EXPECTED_WORKERS=4
  "GOLEM_ATTENTION_KEY_BLOCK_ROWS=$KEY_BLOCK_ROWS"
  "GOLEM_DMA_RESPONSE_VN=$DMA_RESPONSE_VN"
  "GOLEM_ATTENTION_KV_SECOND_LOOKAHEAD=$KV_SECOND_LOOKAHEAD"
  "GOLEM_ATTENTION_KV_CROSS_QUERY_PREFETCH=$KV_CROSS_QUERY_PREFETCH"
  "GOLEM_ATTENTION_KV_PAIR_REUSE=$KV_PAIR_REUSE"
  "GOLEM_ATTENTION_KV_QUERY_GROUP_SIZE=$KV_QUERY_GROUP_SIZE"
  "GOLEM_ATTENTION_PV_V_TILE_REUSE=$PV_V_TILE_REUSE"
  "GOLEM_ATTENTION_PV_V_TILE_GROUP_RETENTION=$PV_V_TILE_GROUP_RETENTION"
  "GOLEM_ATTENTION_PV_INPUT_PIPELINE=$PV_INPUT_PIPELINE"
  "GOLEM_ATTENTION_CLUSTER_PV_ROW_WAVEFRONT=$CLUSTER_PV_ROW_WAVEFRONT"
  "GOLEM_ATTENTION_CLUSTER_QK_MATRIX_LOOKAHEAD=$CLUSTER_QK_MATRIX_LOOKAHEAD"
  "GOLEM_ATTENTION_CLUSTER_PV_MATRIX_LOOKAHEAD=$CLUSTER_PV_MATRIX_LOOKAHEAD"
  "GOLEM_ATTENTION_PV_COMPACT_INPUT=$PV_COMPACT_INPUT"
  "GOLEM_ATTENTION_PV_INPUT_RESIDENCY=$PV_INPUT_RESIDENCY"
  "GOLEM_ATTENTION_O_ACCUMULATOR_CBUFFER=$O_ACCUMULATOR_CBUFFER"
  "GOLEM_ATTENTION_PV_RESTORE_PIPELINE=$PV_RESTORE_PIPELINE"
  "GOLEM_ATTENTION_PV_OUTPUT_PIPELINE=$PV_OUTPUT_PIPELINE"
  "GOLEM_ATTENTION_PV_O_ROW_FUSION=$PV_O_ROW_FUSION"
  "GOLEM_ATTENTION_PV_EARLY_COMPUTE=$PV_EARLY_COMPUTE"
  "GOLEM_ATTENTION_PV_MATRIX_SOFTMAX_OVERLAP=$PV_MATRIX_SOFTMAX_OVERLAP"
  "GOLEM_ATTENTION_PV_ACTIVE_K=$PV_ACTIVE_K"
  "GOLEM_ARRAY_MAC_PER_CU_PER_CYCLE=$ARRAY_MAC_PER_CU_PER_CYCLE"
  "GOLEM_ARRAY_PIPELINE_DEPTH=$ARRAY_PIPELINE_DEPTH"
  "GOLEM_ARRAY_OUTPUT_READ_CREDITS=$ARRAY_OUTPUT_READ_CREDITS"
  "GOLEM_ARRAY_OUTPUT_READ_BANKS=$ARRAY_OUTPUT_READ_BANKS"
  "GOLEM_ARRAY_BUFFER_PORTS=$ARRAY_BUFFER_PORTS"
  "GOLEM_ARRAY_BUFFER_BASE_LATENCY_CYCLES=$ARRAY_BUFFER_BASE_LATENCY_CYCLES"
  "GOLEM_LOCAL_GM_READ_PORTS=$LOCAL_GM_READ_PORTS"
  "GOLEM_ATTENTION_NEAR_ARRAY_OUTPUT_BYTES_PER_CYCLE=$NEAR_ARRAY_OUTPUT_BYTES_PER_CYCLE"
  "GOLEM_ATTENTION_NEAR_ARRAY_OUTPUT_CREDITS=$NEAR_ARRAY_OUTPUT_CREDITS"
  "GOLEM_ATTENTION_PV_V_TILE_BUFFER_BYTES=$PV_V_TILE_BUFFER_BYTES"
  "GOLEM_ATTENTION_PV_V_TILE_BUFFER_HIT_TICKS=$PV_V_TILE_BUFFER_HIT_TICKS"
  "GOLEM_ATTENTION_PV_V_TILE_BUFFER_BYTES_PER_CYCLE=$PV_V_TILE_BUFFER_BYTES_PER_CYCLE"
  "GOLEM_ATTENTION_GENERIC_GEMM_ENABLE=$GENERIC_GEMM"
  "GOLEM_ATTENTION_MILESTONE_TRACE=$ATTENTION_MILESTONE_TRACE"
  "GOLEM_ATTENTION_TILE_TRACE=$ATTENTION_TILE_TRACE"
  GOLEM_PROGRESS_HEARTBEAT=0
  "GOLEM_WCP_GEMM_PROXY_QUEUE_DEPTH=$WCP_GEMM_PROXY_QUEUE_DEPTH"
  "GOLEM_WCP_GEMM_PROXY_ISSUE_WIDTH=$WCP_GEMM_PROXY_ISSUE_WIDTH"
  "GOLEM_WCP_GEMM_PROXY_COMMAND_LATENCY_CYCLES=$WCP_GEMM_PROXY_COMMAND_LATENCY_CYCLES"
  "GOLEM_WCP_GEMM_PROXY_COMPLETION_LATENCY_CYCLES=$WCP_GEMM_PROXY_COMPLETION_LATENCY_CYCLES"
  "GOLEM_ATTENTION_TILE_STORAGE_BANKS=$ATTENTION_TILE_STORAGE_BANKS"
  "GOLEM_ATTENTION_TILE_STORAGE_BANK_BPC=$ATTENTION_TILE_STORAGE_BANK_BPC"
  "GOLEM_MATRIX_BROADCAST_MAX_FANOUT=$MATRIX_BROADCAST_MAX_FANOUT"
  "GOLEM_MATRIX_BROADCAST_BYTES_PER_CYCLE=$MATRIX_BROADCAST_BYTES_PER_CYCLE"
  "GOLEM_MATRIX_BROADCAST_BASE_LATENCY_CYCLES=$MATRIX_BROADCAST_BASE_LATENCY_CYCLES"
  "GOLEM_MATRIX_BROADCAST_STAGE_LATENCY_CYCLES=$MATRIX_BROADCAST_STAGE_LATENCY_CYCLES"
  GOLEM_SFU_ROW_CONTEXTS=16
  GOLEM_DMA_READ_RETRY_TICKS=4096
  GOLEM_DMA_READ_MAX_RETRIES=32
  GOLEM_GROUP_MANAGER_ENABLE=1
  GOLEM_SFU_MANAGER_COORDINATOR=1
  "GOLEM_CTRL_LINK_ENABLE=$KV_DISTRIBUTION"
  GOLEM_REQUEST_SCHEDULER_ENABLE=0
  "GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE=$GENERIC_GEMM"
  GOLEM_SFU_ENABLE=1
  GOLEM_SFU_DISTRIBUTED_REDUCTION_TRANSPORT=explicit_noc
  bash "$BASE_RUNNER"
  --dtype fp32 --tensor-source file --tensor-a "$Q_FILE" --tensor-b "$K_FILE"
  --transpose-b 1 --hbm-dump-output 1
  --gemm-m "$TOTAL_QUERIES" --gemm-n "$KEYS" --gemm-k "$HEAD_DIM"
  --orig-m "$TOTAL_QUERIES" --orig-n "$KEYS" --orig-k "$HEAD_DIM"
  --gemm-block-m 16 --gemm-block-n 16 --gemm-block-k "$HEAD_DIM"
  --array-in "$ARRAY_INPUT" --array-out "$ARRAY_OUTPUT" --num-arrays "$NUM_ARRAYS"
  --groups 4 --num-cores 20 --gemm-cores 20 --num-mem-nodes 5 --mesh-dim-x 4
  --global-stride-kb 1024 --mem-node-size "$MEM_NODE_SIZE"
  --log "$SST_LOG_BASENAME"
  --mpi-ranks "$MPI_RANKS" --mpi-partitioner "$MPI_PARTITIONER")

VERIFY_CMD=(python3 "$SCRIPT_DIR/verify_fused_attention_scale_output.py"
  --q-file "$Q_FILE" --k-file "$K_FILE" --v-file "$V_FILE"
  --queries "$TOTAL_QUERIES" --keys "$KEYS" --head-dim "$HEAD_DIM"
  --band-rows "$MANAGER_QUERIES" --hbm-dir "$HBM_DIR"
  --output-offset "$O_OFFSET" --result-json "$RESULT_JSON")
VERIFY_STATS_CMD=(python3 "$SCRIPT_DIR/verify_fused_attention_scale_stats.py"
  --case-id "$RUN_ID"
  --queries "$TOTAL_QUERIES" --keys "$KEYS" --head-dim "$HEAD_DIM"
  --key-block-rows "$KEY_BLOCK_ROWS"
  --attention-cluster-qk-arrays "$ATTENTION_CLUSTER_QK_ARRAYS"
  --normalization-clock "$NORMALIZATION_CLOCK"
  --run-config "$RUN_CONFIG_FILE"
  --model-platform-clock "$MODEL_PLATFORM_CLOCK"
  --wcp-gemm-proxy-queue-depth "$WCP_GEMM_PROXY_QUEUE_DEPTH"
  --wcp-gemm-proxy-issue-width "$WCP_GEMM_PROXY_ISSUE_WIDTH"
  --wcp-gemm-proxy-command-latency-cycles "$WCP_GEMM_PROXY_COMMAND_LATENCY_CYCLES"
  --wcp-gemm-proxy-completion-latency-cycles "$WCP_GEMM_PROXY_COMPLETION_LATENCY_CYCLES"
  --matrix-broadcast-max-fanout "$MATRIX_BROADCAST_MAX_FANOUT"
  --matrix-broadcast-bytes-per-cycle "$MATRIX_BROADCAST_BYTES_PER_CYCLE"
  --matrix-broadcast-base-latency-cycles "$MATRIX_BROADCAST_BASE_LATENCY_CYCLES"
  --matrix-broadcast-stage-latency-cycles "$MATRIX_BROADCAST_STAGE_LATENCY_CYCLES"
  --array-mac-per-cu-per-cycle "$ARRAY_MAC_PER_CU_PER_CYCLE"
  --array-pipeline-depth "$ARRAY_PIPELINE_DEPTH"
  --near-array-output-bytes-per-cycle "$NEAR_ARRAY_OUTPUT_BYTES_PER_CYCLE"
  --array-buffer-base-latency-cycles "$ARRAY_BUFFER_BASE_LATENCY_CYCLES"
  --pv-v-tile-buffer-bytes "$PV_V_TILE_BUFFER_BYTES"
  --pv-v-tile-buffer-hit-ticks "$PV_V_TILE_BUFFER_HIT_TICKS"
  --pv-v-tile-buffer-bytes-per-cycle "$PV_V_TILE_BUFFER_BYTES_PER_CYCLE"
  --runtime-log "$SST_RUNTIME_LOG"
  --qk-readout-window "$QK_READOUT_WINDOW"
  --attention-tile-storage-banks "$ATTENTION_TILE_STORAGE_BANKS"
  --attention-tile-storage-bank-bytes-per-cycle "$ATTENTION_TILE_STORAGE_BANK_BPC"
  --timebase-ticks-per-second 1000000000000
  --result-json "$LIFECYCLE_JSON" "$STATS_FILE")
VERIFY_MPI_CMD=(python3 "$SCRIPT_DIR/verify_attention_mpi_partition.py"
  --stats-file "$STATS_FILE" --mpi-ranks "$MPI_RANKS"
  --placement-file "$MPI_PLACEMENT_JSON"
  --result-json "$MPI_PARTITION_JSON")
REPORT_CMD=(python3 "$SCRIPT_DIR/report_attention_metrics.py"
  --lifecycle-json "$LIFECYCLE_JSON" --numerical-json "$RESULT_JSON"
  --mpi-partition-json "$MPI_PARTITION_JSON"
  --profile "$RUN_ID" --mpi-ranks "$MPI_RANKS"
  --sst-wall-seconds 0 --pipeline-wall-seconds 0
  --output-json "$METRICS_JSON" --output-csv "$METRICS_CSV")
BASELINE_VERIFY_CMD=()
if [[ -n "$BASELINE_JSON" ]]; then
  if [[ ! -f "$BASELINE_JSON" ]]; then
    echo "Missing Attention baseline: $BASELINE_JSON" >&2
    exit 1
  fi
  if [[ "$(realpath -m "$BASELINE_JSON")" == \
        "$(realpath -m "$BASELINE_RESULT_JSON")" ]]; then
    echo "Attention baseline input conflicts with generated verification path" >&2
    exit 2
  fi
  python3 "$SCRIPT_DIR/verify_flash_attention_baseline.py" \
    --baseline "$BASELINE_JSON" --mpi-ranks "$MPI_RANKS" \
    --queries "$TOTAL_QUERIES" --keys "$KEYS" --head-dim "$HEAD_DIM" \
    --preflight-only
  BASELINE_VERIFY_CMD=(python3 "$SCRIPT_DIR/verify_flash_attention_baseline.py"
    --baseline "$BASELINE_JSON" --result "$RESULT_JSON"
    --lifecycle "$LIFECYCLE_JSON" --mpi-ranks "$MPI_RANKS"
    --result-json "$BASELINE_RESULT_JSON")
  if (( MPI_RANKS > 1 )); then
    BASELINE_VERIFY_CMD+=(--partition "$MPI_PARTITION_JSON")
  fi
  REPORT_CMD+=(--baseline-json "$BASELINE_RESULT_JSON")
fi
if (( PV_MATRIX_BROADCAST )); then
  VERIFY_STATS_CMD+=(--pv-matrix-broadcast)
fi
if (( QK_MATRIX_BROADCAST )); then
  VERIFY_STATS_CMD+=(--qk-matrix-broadcast)
fi
if (( QK_DATAFLOW_TRANSPOSE )); then
  VERIFY_STATS_CMD+=(--qk-dataflow-transpose)
fi
if (( QK_EARLY_COMPUTE )); then
  VERIFY_STATS_CMD+=(--qk-early-compute)
else
  VERIFY_STATS_CMD+=(--no-qk-early-compute)
fi
if (( QK_INPUT_PIPELINE )); then
  VERIFY_STATS_CMD+=(--qk-input-pipeline)
else
  VERIFY_STATS_CMD+=(--no-qk-input-pipeline)
fi
if (( QK_READOUT_OVERLAP )); then
  VERIFY_STATS_CMD+=(--qk-readout-overlap)
else
  VERIFY_STATS_CMD+=(--no-qk-readout-overlap)
fi
if (( QK_PANEL_ROW_BURST )); then
  VERIFY_STATS_CMD+=(--qk-panel-row-burst)
else
  VERIFY_STATS_CMD+=(--no-qk-panel-row-burst)
fi
if (( CROSS_TILE_OPERAND_PIPELINE )); then
  VERIFY_STATS_CMD+=(--cross-tile-operand-pipeline)
else
  VERIFY_STATS_CMD+=(--no-cross-tile-operand-pipeline)
fi
if (( KV_DOUBLE_BUFFER )); then
  VERIFY_STATS_CMD+=(--kv-double-buffer)
fi
if (( KV_DISTRIBUTION )); then
  VERIFY_STATS_CMD+=(--kv-distribution)
else
  VERIFY_STATS_CMD+=(--no-kv-distribution)
fi
if (( KV_MANAGER_LOOKAHEAD )); then
  VERIFY_STATS_CMD+=(--kv-manager-lookahead)
else
  VERIFY_STATS_CMD+=(--no-kv-manager-lookahead)
fi
if (( ATTENTION_CLUSTER )); then
  VERIFY_STATS_CMD+=(--attention-cluster)
fi
if (( KV_SECOND_LOOKAHEAD )); then
  VERIFY_STATS_CMD+=(--kv-second-lookahead)
else
  VERIFY_STATS_CMD+=(--no-kv-second-lookahead)
fi
if (( KV_CROSS_QUERY_PREFETCH )); then
  VERIFY_STATS_CMD+=(--kv-cross-query-prefetch)
else
  VERIFY_STATS_CMD+=(--no-kv-cross-query-prefetch)
fi
if (( KV_PAIR_REUSE )); then
  VERIFY_STATS_CMD+=(--kv-pair-reuse)
else
  VERIFY_STATS_CMD+=(--no-kv-pair-reuse)
fi
VERIFY_STATS_CMD+=(--kv-query-group-size "$KV_QUERY_GROUP_SIZE")
if (( PV_V_TILE_REUSE )); then
  VERIFY_STATS_CMD+=(--pv-v-tile-reuse)
fi
if (( PV_V_TILE_GROUP_RETENTION )); then
  VERIFY_STATS_CMD+=(--pv-v-tile-group-retention)
else
  VERIFY_STATS_CMD+=(--no-pv-v-tile-group-retention)
fi
if (( PV_INPUT_PIPELINE )); then
  VERIFY_STATS_CMD+=(--pv-input-pipeline)
fi
if (( CLUSTER_PV_ROW_WAVEFRONT )); then
  VERIFY_STATS_CMD+=(--cluster-pv-row-wavefront)
else
  VERIFY_STATS_CMD+=(--no-cluster-pv-row-wavefront)
fi
if (( CLUSTER_QK_MATRIX_LOOKAHEAD )); then
  VERIFY_STATS_CMD+=(--cluster-qk-matrix-lookahead)
else
  VERIFY_STATS_CMD+=(--no-cluster-qk-matrix-lookahead)
fi
if (( CLUSTER_PV_MATRIX_LOOKAHEAD )); then
  VERIFY_STATS_CMD+=(--cluster-pv-matrix-lookahead)
else
  VERIFY_STATS_CMD+=(--no-cluster-pv-matrix-lookahead)
fi
if (( PV_INPUT_RESIDENCY )); then
  VERIFY_STATS_CMD+=(--pv-input-residency)
else
  VERIFY_STATS_CMD+=(--no-pv-input-residency)
fi
if (( O_ACCUMULATOR_CBUFFER )); then
  VERIFY_STATS_CMD+=(--o-accumulator-cbuffer)
else
  VERIFY_STATS_CMD+=(--no-o-accumulator-cbuffer)
fi
if (( PV_RESTORE_PIPELINE )); then
  VERIFY_STATS_CMD+=(--pv-restore-pipeline)
fi
if (( PV_OUTPUT_PIPELINE )); then
  VERIFY_STATS_CMD+=(--pv-output-pipeline)
fi
if (( PV_O_ROW_FUSION )); then
  VERIFY_STATS_CMD+=(--pv-o-row-fusion)
else
  VERIFY_STATS_CMD+=(--no-pv-o-row-fusion)
fi
if (( PV_EARLY_COMPUTE )); then
  VERIFY_STATS_CMD+=(--pv-early-compute)
fi
if (( PV_MATRIX_SOFTMAX_OVERLAP )); then
  VERIFY_STATS_CMD+=(--pv-matrix-softmax-overlap)
fi
if (( PV_ACTIVE_K )); then
  VERIFY_STATS_CMD+=(--pv-active-k)
fi
if (( GENERIC_GEMM )); then
  VERIFY_STATS_CMD+=(--generic-gemm)
  REPORT_CMD+=(--generic-gemm)
fi

if (( DRY_RUN )); then
  printf '%q ' "${GENERATE_CMD[@]}"; printf '\n'
  printf '%q ' "${RUN_CMD[@]}"; printf '\n'
  printf '%q ' "${VERIFY_CMD[@]}"; printf '\n'
  printf '%q ' "${VERIFY_STATS_CMD[@]}"; printf '\n'
  if (( MPI_RANKS > 1 )); then
    printf '%q ' "${VERIFY_MPI_CMD[@]}"; printf '\n'
  fi
  if (( ${#BASELINE_VERIFY_CMD[@]} > 0 )); then
    printf '%q ' "${BASELINE_VERIFY_CMD[@]}"; printf '\n'
  fi
  printf '%q ' "${REPORT_CMD[@]}"; printf '\n'
  exit 0
fi

attention_elapsed_seconds() {
  awk -v start="$1" -v end="$2" 'BEGIN { printf "%.6f", (end - start) / 1000000000.0 }'
}

attention_test_milestone() {
  local stage="$1"
  local status="$2"
  local elapsed="${3:-0}"
  local record
  printf -v record \
    '[ATTENTION_TEST_MILESTONE] stage=%s status=%s wall_epoch_ns=%s elapsed_seconds=%s' \
    "$stage" "$status" "$(date +%s%N)" "$elapsed"
  printf '%s\n' "$record" >> "$DRIVER_MILESTONE_LOG"
  if (( ATTENTION_TERMINAL_VERBOSE )); then
    printf '%s\n' "$record"
  fi
}

ATTENTION_STAGE_SECONDS=0
attention_stage_label() {
  case "$1" in
    generate) echo "input generation" ;;
    sst) echo "SST simulation" ;;
    numerical_verify) echo "numerical verification" ;;
    lifecycle_verify) echo "lifecycle verification" ;;
    mpi_partition_verify) echo "MPI partition verification" ;;
    baseline_verify) echo "baseline verification" ;;
    metrics_report) echo "metrics report" ;;
    *) echo "$1" ;;
  esac
}

run_attention_stage_command() {
  local stage="$1"
  shift
  local stage_log="$DRIVER_LOG_DIR/$stage.log"
  if (( ATTENTION_TERMINAL_VERBOSE )); then
    "$@" 2>&1 | tee "$stage_log"
  elif [[ "$stage" == sst ]]; then
    ATTENTION_UI_PROGRESS_LOG="$SST_RUNTIME_LOG" \
    ATTENTION_UI_MILESTONE_FILTER="$SCRIPT_DIR/attention_progress.awk" \
      attention_ui_run_sst \
      "$stage_log" "$SCRIPT_DIR/attention_terminal_progress.awk" "$@"
  elif [[ "$stage" == metrics_report ]]; then
    if attention_ui_supports_color; then
      GOLEM_ATTENTION_COLOR=1 "$@" 2>&1 | tee "$stage_log"
    else
      "$@" 2>&1 | tee "$stage_log"
    fi
  else
    "$@" > "$stage_log" 2>&1
  fi
}

run_attention_stage() {
  local stage="$1"
  shift
  local start_ns end_ns elapsed
  start_ns="$(date +%s%N)"
  attention_test_milestone "$stage" start
  if run_attention_stage_command "$stage" "$@"; then
    end_ns="$(date +%s%N)"
    elapsed="$(attention_elapsed_seconds "$start_ns" "$end_ns")"
    ATTENTION_STAGE_SECONDS="$elapsed"
    attention_test_milestone "$stage" done "$elapsed"
    if [[ "$stage" != metrics_report ]]; then
      attention_ui_stage_result \
        "$(attention_stage_label "$stage")" PASS "${elapsed}s"
    fi
    return 0
  else
    local rc=$?
    end_ns="$(date +%s%N)"
    elapsed="$(attention_elapsed_seconds "$start_ns" "$end_ns")"
    ATTENTION_STAGE_SECONDS="$elapsed"
    attention_test_milestone "$stage" fail "$elapsed"
    attention_ui_stage_result \
      "$(attention_stage_label "$stage")" FAIL "${elapsed}s" >&2
    if (( ! ATTENTION_TERMINAL_VERBOSE )) && [[ "$stage" != metrics_report ]]; then
      tail -n 20 "$DRIVER_LOG_DIR/$stage.log" >&2 || true
    fi
    return "$rc"
  fi
}

TEST_START_NS="$(date +%s%N)"
mkdir -p "$ARTIFACT_ROOT" "$DRIVER_LOG_DIR"
rm -f "$RESULT_JSON" "$LIFECYCLE_JSON" "$METRICS_JSON" "$METRICS_CSV" \
  "$MPI_PARTITION_JSON" "$MPI_PLACEMENT_JSON" "$STATS_FILE" \
  "$BASELINE_RESULT_JSON" \
  "$DRIVER_MILESTONE_LOG" "$DRIVER_LOG_DIR/generate.log" \
  "$DRIVER_LOG_DIR/sst.log" "$DRIVER_LOG_DIR/numerical_verify.log" \
  "$DRIVER_LOG_DIR/lifecycle_verify.log" \
  "$DRIVER_LOG_DIR/mpi_partition_verify.log" \
  "$DRIVER_LOG_DIR/baseline_verify.log" \
  "$DRIVER_LOG_DIR/metrics_report.log"
rm -f "$SST_RUNTIME_LOG"
: > "$DRIVER_MILESTONE_LOG"
for rank in 0 1 2 3; do
  rm -f "${STATS_FILE%.txt}_${rank}.txt"
done
run_attention_stage generate "${GENERATE_CMD[@]}"
run_attention_stage sst "${RUN_CMD[@]}"
SST_WALL_SECONDS="$ATTENTION_STAGE_SECONDS"
ATTENTION_TEST_RC=0
if run_attention_stage numerical_verify "${VERIFY_CMD[@]}"; then
  :
else
  ATTENTION_TEST_RC=$?
fi
if run_attention_stage lifecycle_verify "${VERIFY_STATS_CMD[@]}"; then
  :
else
  stage_rc=$?
  if (( ATTENTION_TEST_RC == 0 )); then ATTENTION_TEST_RC="$stage_rc"; fi
fi
if (( MPI_RANKS > 1 )); then
  if run_attention_stage mpi_partition_verify "${VERIFY_MPI_CMD[@]}"; then
    :
  else
    stage_rc=$?
    if (( ATTENTION_TEST_RC == 0 )); then ATTENTION_TEST_RC="$stage_rc"; fi
  fi
fi
if (( ${#BASELINE_VERIFY_CMD[@]} > 0 )); then
  if run_attention_stage baseline_verify "${BASELINE_VERIFY_CMD[@]}"; then
    :
  else
    stage_rc=$?
    if (( ATTENTION_TEST_RC == 0 )); then ATTENTION_TEST_RC="$stage_rc"; fi
  fi
fi
TEST_END_NS="$(date +%s%N)"
PIPELINE_WALL_SECONDS="$(attention_elapsed_seconds "$TEST_START_NS" "$TEST_END_NS")"
REPORT_CMD=(python3 "$SCRIPT_DIR/report_attention_metrics.py"
  --lifecycle-json "$LIFECYCLE_JSON" --numerical-json "$RESULT_JSON"
  --mpi-partition-json "$MPI_PARTITION_JSON"
  --profile "$RUN_ID" --mpi-ranks "$MPI_RANKS"
  --sst-wall-seconds "$SST_WALL_SECONDS"
  --pipeline-wall-seconds "$PIPELINE_WALL_SECONDS"
  --output-json "$METRICS_JSON" --output-csv "$METRICS_CSV")
if [[ -n "$BASELINE_JSON" ]]; then
  REPORT_CMD+=(--baseline-json "$BASELINE_RESULT_JSON")
fi
if [[ -f "$LIFECYCLE_JSON" && -f "$RESULT_JSON" ]]; then
  if (( GENERIC_GEMM )); then
    REPORT_CMD+=(--generic-gemm)
  fi
  if run_attention_stage metrics_report "${REPORT_CMD[@]}"; then
    :
  else
    stage_rc=$?
    if (( ATTENTION_TEST_RC == 0 )); then ATTENTION_TEST_RC="$stage_rc"; fi
  fi
else
  attention_test_milestone metrics_report fail
  echo "Cannot generate Attention metrics: verifier JSON artifact missing" >&2
  if (( ATTENTION_TEST_RC == 0 )); then ATTENTION_TEST_RC=1; fi
fi

if (( ATTENTION_TEST_RC != 0 )); then
  echo "Fused Attention $RUN_ID FAIL: verification or reporting failed" >&2
fi
attention_ui_key_value "Artifacts" "$ARTIFACT_ROOT"
attention_ui_key_value "Attention metrics JSON" "$METRICS_JSON"
attention_ui_key_value "Attention metrics CSV" "$METRICS_CSV"
exit "$ATTENTION_TEST_RC"
