#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR"
while [[ ! -f "$ROOT/configure.ac" && "$ROOT" != / ]]; do ROOT="$(dirname "$ROOT")"; done
[[ -f "$ROOT/configure.ac" ]] || { echo "Cannot find project root" >&2; exit 1; }
export GOLEM_ATTENTION_SEQUENTIAL_64_ENABLE=0
ATTENTION_RUNNER="$ROOT/test/golem/small/muticore_attention/run_fused_attention_scale.sh"
ARTIFACT_ROOT="${GOLEM_ARTIFACT_ROOT:-$SCRIPT_DIR/artifacts/sst_qk_bridge}"

export GOLEM_ATTENTION_REUSE_WINDOW_QK_BRIDGE=1
export GOLEM_MPI_RANKS="${GOLEM_MPI_RANKS:-4}"
export GOLEM_OUTPUT_MODE=fusion
export GOLEM_FINAL_C_WRITE_ENABLE=0
export GOLEM_HBM_DUMP_OUTPUT=1
export GOLEM_KEEP_HBM_DUMP_IN_FUSION=1

exec "$ATTENTION_RUNNER" \
  --artifact-root "$ARTIFACT_ROOT" \
  --query-length "${GOLEM_QUERY_LENGTH:-1024}" \
  --kv-length "${GOLEM_KV_LENGTH:-1024}" \
  --num-query-heads "${GOLEM_NUM_QUERY_HEADS:-4}" \
  --num-kv-heads "${GOLEM_NUM_KV_HEADS:-2}" \
  --head-dim "${GOLEM_HEAD_DIM:-128}" \
  --timeout "${GOLEM_TIMEOUT_SECONDS:-7200}"
