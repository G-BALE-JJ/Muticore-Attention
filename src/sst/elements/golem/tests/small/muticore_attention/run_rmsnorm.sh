#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT="$(cd "$TESTS_DIR/../../../../.." && pwd)"
ARTIFACT_ROOT="${GOLEM_ARTIFACT_ROOT:-$ROOT/results/rmsnorm_$(date +%Y%m%d_%H%M%S)}"
ROWS="${GOLEM_SFU_RMSNORM_ROWS:-16}"
COLS="${GOLEM_SFU_RMSNORM_COLS:-128}"
EPSILON="${GOLEM_SFU_RMSNORM_EPSILON:-1.0e-6}"
CASE_DIR="$ARTIFACT_ROOT/case"
GUEST="$SCRIPT_DIR/riscv64/rmsnorm"

mkdir -p "$CASE_DIR"
python3 "$SCRIPT_DIR/rmsnorm_case.py" prepare \
  --case-dir "$CASE_DIR" --rows "$ROWS" --cols "$COLS"
make -C "$SCRIPT_DIR" rmsnorm CFLAGS="-DGOLEM_SFU_RMSNORM_ROWS=$ROWS -DGOLEM_SFU_RMSNORM_COLS=$COLS -DGOLEM_SFU_RMSNORM_EPSILON=$EPSILON" -B

env \
  GOLEM_ARTIFACT_ROOT="$ARTIFACT_ROOT" \
  LOG_FILE="$ARTIFACT_ROOT/sst.log" \
  GOLEM_SFU_RMSNORM_HBM_STREAM=1 \
  GOLEM_SFU_RMSNORM_ROWS="$ROWS" \
  GOLEM_SFU_RMSNORM_COLS="$COLS" \
  GOLEM_SFU_RMSNORM_X_FILE="$CASE_DIR/x.bin" \
  GOLEM_SFU_RMSNORM_GAMMA_FILE="$CASE_DIR/gamma.bin" \
  GOLEM_ATTENTION_FUSED=0 \
  GOLEM_ATTENTION_KV_BUFFER_COUNT=1 \
  GOLEM_SFU_ENABLE=1 \
  GOLEM_GEMM_M=64 GOLEM_GEMM_N=64 GOLEM_GEMM_K=64 \
  GOLEM_SKIP_DEFAULT_GUEST_BUILD=1 VANADIS_EXE="$GUEST" \
  GOLEM_VERIFY_MVM=0 GOLEM_HBM_DUMP_OUTPUT=1 \
  GOLEM_OUTPUT_MODE=hbm GOLEM_MPI_RANKS=1 \
  "$TESTS_DIR/run_noc_dma_pipeline.sh" "$@"

python3 "$SCRIPT_DIR/rmsnorm_case.py" verify \
  --case-dir "$CASE_DIR" --hbm-dir "$ARTIFACT_ROOT/hbm" \
  --log "$ARTIFACT_ROOT/sst.log" \
  --rows "$ROWS" --cols "$COLS" --epsilon "$EPSILON"
printf 'Artifacts: %s\n' "$ARTIFACT_ROOT"
