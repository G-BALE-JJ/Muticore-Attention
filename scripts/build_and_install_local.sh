#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOF'
Usage: scripts/build_and_install_local.sh [--clean] [--reconfigure] [--no-autogen] [--jobs N]

Build and install this full-layout worktree's SST elements into a worktree-local
prefix.

Defaults:
  BUILD_ROOT          ./build/sst-elements
  INSTALL_PREFIX      ./install
  SST_CORE_PREFIX     /local/sstcore
  SST_DRAMSIM3_PREFIX /local/packages/dramsim3
  SST_RAMULATOR2_PREFIX ./deps/ramulator2-2.1
  JOBS                nproc

Override defaults with environment variables, for example:
  SST_CORE_PREFIX=/path/to/sst_core_install \
  SST_DRAMSIM3_PREFIX=/path/to/DRAMsim3 \
  SST_RAMULATOR2_PREFIX=/path/to/ramulator2 \
  scripts/build_and_install_local.sh

Options:
  --clean        Remove BUILD_ROOT and INSTALL_PREFIX before building.
  --reconfigure  Remove configure outputs in BUILD_ROOT before configuring.
  --no-autogen   Skip ./autogen.sh.
  --jobs N       Parallel make jobs.
EOF
}

clean=0
reconfigure=0
run_autogen=1
jobs="${JOBS:-}"

while [[ $# -gt 0 ]]; do
	case "$1" in
		-h|--help)
			usage
			exit 0
			;;
		--clean)
			clean=1
			shift
			;;
		--reconfigure)
			reconfigure=1
			shift
			;;
		--no-autogen)
			run_autogen=0
			shift
			;;
		--jobs)
			if [[ $# -lt 2 || ! "$2" =~ ^[0-9]+$ || "$2" -le 0 ]]; then
				echo "[ERROR] --jobs requires a positive integer" >&2
				exit 1
			fi
			jobs="$2"
			shift 2
			;;
		*)
			echo "[ERROR] Unknown option: $1" >&2
			usage >&2
			exit 1
			;;
	esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
WORKTREE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
BUILD_ROOT="${BUILD_ROOT:-$WORKTREE_ROOT/build/sst-elements}"
INSTALL_PREFIX="${INSTALL_PREFIX:-$WORKTREE_ROOT/install}"
SST_CORE_PREFIX="${SST_CORE_PREFIX:-/local/sstcore}"
SST_DRAMSIM3_PREFIX="${SST_DRAMSIM3_PREFIX:-/local/packages/dramsim3}"
SST_RAMULATOR2_PREFIX="${SST_RAMULATOR2_PREFIX:-$WORKTREE_ROOT/deps/ramulator2-2.1}"
INSTALL_HOME="$BUILD_ROOT/.sst-home"
INSTALL_HOME_CONFIG="$INSTALL_HOME/.sst/sstsimulator.conf"
ATTENTION_DIR="$WORKTREE_ROOT/src/sst/elements/golem/tests/small/muticore_attention"

restore_handwritten_test_makefiles() {
	local rel
	for rel in \
		"src/sst/elements/golem/tests/small/mvm_noc_int_array/Makefile"; do
		if [[ -f "$WORKTREE_ROOT/$rel" ]]; then
			install -D -m 644 "$WORKTREE_ROOT/$rel" "$BUILD_ROOT/$rel"
		fi
	done
}

if [[ -z "$jobs" ]]; then
	jobs="$(nproc 2>/dev/null || echo 1)"
fi

if [[ ! -x "$SST_CORE_PREFIX/bin/sst-config" ]]; then
	echo "[ERROR] Missing SST core install: $SST_CORE_PREFIX/bin/sst-config" >&2
	exit 1
fi

if [[ ! -d "$SST_DRAMSIM3_PREFIX" ]]; then
	echo "[ERROR] Missing DRAMSim3 prefix: $SST_DRAMSIM3_PREFIX" >&2
	exit 1
fi
if [[ ! -f "$SST_RAMULATOR2_PREFIX/libramulator.so" ]]; then
	echo "[ERROR] Missing Ramulator2 library: $SST_RAMULATOR2_PREFIX/libramulator.so" >&2
	exit 1
fi

if [[ "$clean" -eq 1 ]]; then
	echo "[INFO] Removing local build/install directories"
	rm -rf "$BUILD_ROOT" "$INSTALL_PREFIX"
fi

"$SCRIPT_DIR/prepare_local_build.sh" "$BUILD_ROOT"

if [[ ! -d "$BUILD_ROOT/src/sst/elements/golem" || -L "$BUILD_ROOT/src/sst/elements" ]]; then
	echo "[ERROR] $BUILD_ROOT/src/sst/elements is not a copied elements source tree" >&2
	exit 1
fi

cd "$BUILD_ROOT"

if [[ "$reconfigure" -eq 1 ]]; then
	echo "[INFO] Removing configure outputs"
	rm -f Makefile config.log config.status libtool
	find . -name Makefile -type f -delete
	restore_handwritten_test_makefiles
fi

if [[ "$run_autogen" -eq 1 ]]; then
	echo "[1/5] Running autogen.sh"
	./autogen.sh
else
	echo "[1/5] Skipping autogen.sh"
fi

echo "[2/5] Configuring local install"
./configure \
	--prefix="$INSTALL_PREFIX" \
	--with-sst-core="$SST_CORE_PREFIX" \
	--with-dramsim3="$SST_DRAMSIM3_PREFIX" \
	--with-ramulator2="$SST_RAMULATOR2_PREFIX"

echo "[3/5] Building SST elements with $jobs jobs"
make -j"$jobs"

echo "[4/5] Installing to $INSTALL_PREFIX"
mkdir -p "$INSTALL_HOME/.sst"
if [[ ! -f "$INSTALL_HOME_CONFIG" ]]; then
	cp "$SST_CORE_PREFIX/etc/sst/sstsimulator.conf" "$INSTALL_HOME_CONFIG"
fi
HOME="$INSTALL_HOME" make install

echo "[5/5] Building FlashAttention RISC-V guests"
make -C "$ATTENTION_DIR" -j"$jobs" scale

cat <<EOF
[OK] Build and install complete.

Before running experiments from this worktree:
  cd "$WORKTREE_ROOT"
  source scripts/env_local_install.sh
  cd "$BUILD_ROOT/src/sst/elements/golem/tests"

Installed element libraries:
  $INSTALL_PREFIX/lib/sst-elements-library
EOF
