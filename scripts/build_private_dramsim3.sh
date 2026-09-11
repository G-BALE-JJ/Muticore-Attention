#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
WORKTREE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
SOURCE_DIR="${DRAMSIM3_SOURCE_DIR:-/local/packages/dramsim3}"
PATCH_FILE="$SCRIPT_DIR/patches/dramsim3_read_issue_stats.patch"
INSTALL_DIR="${GOLEM_PRIVATE_DRAMSIM3_LIB_DIR:-$WORKTREE_ROOT/install/lib/dramsim3}"
ELEMENT_LIB_DIR="${SST_ELEMENTS_INSTALL_PREFIX:-$WORKTREE_ROOT/install}/lib/sst-elements-library"

if [[ ! -f "$SOURCE_DIR/src/controller.cc" ]]; then
	echo "[ERROR] Invalid DRAMSim3 source directory: $SOURCE_DIR" >&2
	exit 2
fi
if [[ ! -f "$PATCH_FILE" ]]; then
	echo "[ERROR] Missing patch: $PATCH_FILE" >&2
	exit 2
fi

TEMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/golem-dramsim3.XXXXXX")"
trap 'rm -rf -- "$TEMP_ROOT"' EXIT
PRIVATE_SOURCE="$TEMP_ROOT/source"
PRIVATE_BUILD="$TEMP_ROOT/build"

mkdir -p "$PRIVATE_SOURCE" "$PRIVATE_BUILD" "$INSTALL_DIR" "$ELEMENT_LIB_DIR"
cp -a "$SOURCE_DIR/." "$PRIVATE_SOURCE/"
patch --directory="$PRIVATE_SOURCE" --strip=1 --input="$PATCH_FILE"
cmake -S "$PRIVATE_SOURCE" -B "$PRIVATE_BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$PRIVATE_BUILD" --target dramsim3 --parallel "${GOLEM_BUILD_JOBS:-4}"
cp -f "$PRIVATE_SOURCE/libdramsim3.so" "$INSTALL_DIR/libdramsim3.so"
cp -f "$PRIVATE_SOURCE/libdramsim3.so" "$ELEMENT_LIB_DIR/libdramsim3.so"

echo "[OK] Private DRAMSim3 library: $INSTALL_DIR/libdramsim3.so"
echo "[OK] SST-visible DRAMSim3 library: $ELEMENT_LIB_DIR/libdramsim3.so"
