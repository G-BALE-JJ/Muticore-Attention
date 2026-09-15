#!/usr/bin/env bash
set -euo pipefail

# Canonical local Attention entry point. Workload dimensions are forwarded to
# the scale-capable implementation; no named scale profile is selected here.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${GOLEM_ATTENTION_SEQUENTIAL_64_ENABLE:=1}"
export GOLEM_ATTENTION_SEQUENTIAL_64_ENABLE
exec "$SCRIPT_DIR/run_fused_attention_scale.sh" "$@"
