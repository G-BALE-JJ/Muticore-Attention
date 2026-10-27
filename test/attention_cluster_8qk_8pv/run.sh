#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARTIFACT_ROOT="${GOLEM_ARTIFACT_ROOT:-$SCRIPT_DIR/artifacts/latest}"

exec python3 "$SCRIPT_DIR/attention_cluster_model.py" \
  --artifact-root "$ARTIFACT_ROOT" \
  --query-length "${GOLEM_QUERY_LENGTH:-1024}" \
  --kv-length "${GOLEM_KV_LENGTH:-1024}" \
  --num-query-heads "${GOLEM_NUM_QUERY_HEADS:-4}" \
  --num-kv-heads "${GOLEM_NUM_KV_HEADS:-2}" \
  --head-dim "${GOLEM_HEAD_DIM:-128}" \
  --qk-workers-per-manager 2 \
  "$@"
