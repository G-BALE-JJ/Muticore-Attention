#!/usr/bin/env bash
set -euo pipefail

# Canonical local Attention entry point. Workload dimensions are forwarded to
# the scale-capable implementation; no named scale profile is selected here.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$SCRIPT_DIR"
while [[ ! -f "$WORKTREE_ROOT/configure.ac" && "$WORKTREE_ROOT" != / ]]; do
  WORKTREE_ROOT="$(dirname "$WORKTREE_ROOT")"
done
[[ -f "$WORKTREE_ROOT/configure.ac" ]] || { echo "Cannot find project root" >&2; exit 1; }
exec "$WORKTREE_ROOT/test/attention_cluster_8qk_8pv/run_sst.sh" "$@"
