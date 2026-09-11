#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOF'
Usage: scripts/prepare_local_build.sh [BUILD_ROOT]

Prepare a local SST source/build tree under this full-layout worktree.

Defaults:
  BUILD_ROOT          ./build/sst-elements
  SST_CORE_PREFIX     /local/sstcore
  SST_DRAMSIM3_PREFIX /local/packages/dramsim3
  SST_RAMULATOR2_PREFIX ./deps/ramulator2-2.1

The prepared tree copies this worktree into BUILD_ROOT so generated files stay
inside the local build tree.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
	usage
	exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
WORKTREE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
ELEMENTS_SOURCE="$WORKTREE_ROOT/src/sst/elements"
BUILD_ROOT="${1:-$WORKTREE_ROOT/build/sst-elements}"
INSTALL_PREFIX="$WORKTREE_ROOT/install"
SST_CORE_PREFIX="${SST_CORE_PREFIX:-/local/sstcore}"
SST_DRAMSIM3_PREFIX="${SST_DRAMSIM3_PREFIX:-/local/packages/dramsim3}"
SST_RAMULATOR2_PREFIX="${SST_RAMULATOR2_PREFIX:-$WORKTREE_ROOT/deps/ramulator2-2.1}"

if [[ ! -d "$ELEMENTS_SOURCE/golem" || ! -f "$WORKTREE_ROOT/autogen.sh" || ! -f "$WORKTREE_ROOT/configure.ac" ]]; then
	echo "[ERROR] This must be run from a full SST source layout with src/sst/elements." >&2
	exit 1
fi

if [[ -e "$BUILD_ROOT" ]]; then
	if [[ -d "$BUILD_ROOT/src/sst/elements" && ! -L "$BUILD_ROOT/src/sst/elements" && -d "$BUILD_ROOT/src/sst/elements/golem" ]]; then
		echo "[INFO] Refreshing local build tree: $BUILD_ROOT"
	elif [[ -L "$BUILD_ROOT/src/sst/elements" ]]; then
		current_target="$(readlink -f "$BUILD_ROOT/src/sst/elements")"
		if [[ "$current_target" == "$ELEMENTS_SOURCE" ]]; then
			echo "[INFO] Replacing legacy symlink-based build tree: $BUILD_ROOT"
			rm -rf "$BUILD_ROOT"
		else
			echo "[ERROR] BUILD_ROOT already exists with an unexpected elements symlink: $BUILD_ROOT" >&2
			echo "        $BUILD_ROOT/src/sst/elements -> $current_target" >&2
			echo "        Choose another path or rerun scripts/build_and_install_local.sh --clean after checking its contents." >&2
			exit 1
		fi
	else
		echo "[ERROR] BUILD_ROOT already exists and is not prepared for this worktree: $BUILD_ROOT" >&2
		echo "        Choose another path or rerun scripts/build_and_install_local.sh --clean after checking its contents." >&2
		exit 1
	fi
else
	mkdir -p "$(dirname "$BUILD_ROOT")"
	echo "[INFO] Preparing local build tree: $BUILD_ROOT"
fi

rsync -a \
	--exclude='.git/' \
	--exclude='build/' \
	--exclude='install/' \
	--exclude='deps/' \
	--exclude='autom4te.cache/' \
	--exclude='config.log' \
	--exclude='config.status' \
	--exclude='libtool' \
	--exclude='aclocal.m4' \
	--exclude='configure' \
	--exclude='configure~' \
	--exclude='Makefile' \
	--exclude='Makefile.in' \
	--exclude='src/libltdl/' \
	--exclude='*.o' \
	--exclude='*.lo' \
	--exclude='*.la' \
	--exclude='.deps/' \
	--exclude='.libs/' \
	"$WORKTREE_ROOT/" "$BUILD_ROOT/"

# The golem small test rebuilds `test_noc_dma` in place and needs this
# handwritten Makefile. Keep the source copy clean, then restore just this file.
if [[ -f "$WORKTREE_ROOT/src/sst/elements/golem/tests/small/mvm_noc_int_array/Makefile" ]]; then
	install -D -m 644 \
		"$WORKTREE_ROOT/src/sst/elements/golem/tests/small/mvm_noc_int_array/Makefile" \
		"$BUILD_ROOT/src/sst/elements/golem/tests/small/mvm_noc_int_array/Makefile"
fi

cat <<EOF
[OK] Prepared local build tree:
  $BUILD_ROOT

Use it when you want to build:
  cd "$BUILD_ROOT"
  ./autogen.sh
  ./configure --prefix="$INSTALL_PREFIX" --with-sst-core="$SST_CORE_PREFIX" --with-dramsim3="$SST_DRAMSIM3_PREFIX" --with-ramulator2="$SST_RAMULATOR2_PREFIX"
  make -j
  make install

Before running this worktree's experiments:
  cd "$WORKTREE_ROOT"
  source scripts/env_local_install.sh

Elements source copied into that build tree:
  $BUILD_ROOT/src/sst/elements

Source snapshot used for the copy:
  $ELEMENTS_SOURCE
EOF
