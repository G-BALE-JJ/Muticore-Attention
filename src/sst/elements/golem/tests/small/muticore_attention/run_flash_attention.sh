#!/usr/bin/env bash
set -euo pipefail

# Canonical local Attention entry point. Workload dimensions are forwarded to
# the scale-capable implementation; no named scale profile is selected here.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd "$SCRIPT_DIR/../../../../../../.." && pwd -P)"
exec "$WORKTREE_ROOT/baseline/attention_cluster_8qk_8pv/run_sst.sh" "$@"
