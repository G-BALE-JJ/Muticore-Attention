#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
TESTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
WORKTREE_ROOT="$(cd "$TESTS_DIR/../../../../.." && pwd -P)"
RAMULATOR2_ROOT="${SST_RAMULATOR2_PREFIX:-$WORKTREE_ROOT/deps/ramulator2-2.1}"
OUT_DIR="${GOLEM_RAMULATOR2_MICROBENCH_DIR:-$TESTS_DIR/artifacts/ramulator2_microbench}"
ONE_CHANNEL_CONFIG="$TESTS_DIR/architecture/ramulator/hbm2e_2500_one_channel.yaml"
STACK_CONFIG="$TESTS_DIR/architecture/ramulator/hbm2e_2500.yaml"
ROOFLINE_CONFIG="$TESTS_DIR/architecture/ramulator/hbm2e_2500_stack_norefresh.yaml"
BINARY="$OUT_DIR/ramulator2_hbm_microbench"

mkdir -p "$OUT_DIR"
g++ -std=c++20 -O2 \
	-I"$RAMULATOR2_ROOT/src" \
	-I"$RAMULATOR2_ROOT/ext/yaml-cpp/include" \
	-I"$RAMULATOR2_ROOT/ext/fmt/include" \
	"$SCRIPT_DIR/ramulator2_hbm_microbench.cpp" \
	-L"$RAMULATOR2_ROOT" -Wl,-rpath,"$RAMULATOR2_ROOT" -lramulator \
	-o "$BINARY"

{
	"$BINARY" "$ONE_CHANNEL_CONFIG" tccd-l
	"$BINARY" "$ONE_CHANNEL_CONFIG" tccd-s
	"$BINARY" "$ONE_CHANNEL_CONFIG" pseudo-channel
	"$BINARY" "$ROOFLINE_CONFIG" stack-roofline "${GOLEM_RAMULATOR2_STACK_REQUESTS:-262144}"
	"$BINARY" "$STACK_CONFIG" stack-refresh "${GOLEM_RAMULATOR2_STACK_REQUESTS:-262144}"
} | tee "$OUT_DIR/results.txt"

awk '
function value(name,    i,pair) {
    for (i = 1; i <= NF; ++i) {
        split($i, pair, "=")
        if (pair[1] == name) return pair[2] + 0
    }
    return -1
}
$1 == "mode=tccd-l" && (value("avg_command_interval_cycles") < 3.95 || value("avg_command_interval_cycles") > 4.05) { exit 1 }
$1 == "mode=tccd-s" && (value("avg_command_interval_cycles") < 1.95 || value("avg_command_interval_cycles") > 2.10) { exit 1 }
$1 == "mode=pseudo-channel" && (value("avg_command_interval_cycles") < 0.98 || value("avg_command_interval_cycles") > 1.05) { exit 1 }
$1 == "mode=stack-roofline" && value("bandwidth_GBps") < 315.0 { exit 1 }
' "$OUT_DIR/results.txt" || {
	echo "[ERROR] Ramulator2 HBM timing/roofline validation failed" >&2
	exit 1
}

echo "[OK] Ramulator2 HBM microbenchmark: $OUT_DIR/results.txt"
