#!/usr/bin/env bash
# Source this file before running experiments from this worktree:
#   source scripts/env_local_install.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
	echo "source this script instead of executing it:" >&2
	echo "  source scripts/env_local_install.sh" >&2
	exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
WORKTREE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"

export SST_CORE_PREFIX="${SST_CORE_PREFIX:-/local/sstcore}"
export SST_DRAMSIM3_PREFIX="${SST_DRAMSIM3_PREFIX:-/local/packages/dramsim3}"
export SST_RAMULATOR2_PREFIX="${SST_RAMULATOR2_PREFIX:-$WORKTREE_ROOT/deps/ramulator2-2.1}"
export RISCV_TOOLCHAIN_BIN="${RISCV_TOOLCHAIN_BIN:-/local/scratch/src/riscv64-linux-musl-cross/bin}"
export SST_ELEMENTS_INSTALL_PREFIX="${SST_ELEMENTS_INSTALL_PREFIX:-$WORKTREE_ROOT/install}"
export SST_PYTHON_LIB_DIR="${SST_PYTHON_LIB_DIR:-/usr/lib/x86_64-linux-gnu}"
export GOLEM_PRIVATE_DRAMSIM3_LIB_DIR="${GOLEM_PRIVATE_DRAMSIM3_LIB_DIR:-$WORKTREE_ROOT/install/lib/dramsim3}"
export GOLEM_PRIVATE_RAMULATOR2_LIB_DIR="${GOLEM_PRIVATE_RAMULATOR2_LIB_DIR:-$WORKTREE_ROOT/deps/ramulator2-2.1}"

for bin_dir in "$RISCV_TOOLCHAIN_BIN" "$SST_CORE_PREFIX/bin"; do
	case ":${PATH:-}:" in
		*":$bin_dir:"*) ;;
		*) export PATH="$bin_dir${PATH:+:$PATH}" ;;
	esac
done

for lib_dir in "$SST_PYTHON_LIB_DIR" "$GOLEM_PRIVATE_DRAMSIM3_LIB_DIR" "$GOLEM_PRIVATE_RAMULATOR2_LIB_DIR" "$SST_ELEMENTS_INSTALL_PREFIX/lib" "$SST_CORE_PREFIX/lib"; do
	[[ -d "$lib_dir" ]] || continue
	case ":${LD_LIBRARY_PATH:-}:" in
		*":$lib_dir:"*) ;;
		*) export LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
	esac
done

elements_lib_path="$SST_ELEMENTS_INSTALL_PREFIX/lib/sst-elements-library"
read -r -a requested_sst_args <<< "${GOLEM_SST_ARGS:-}"
filtered_sst_args=()
skip_sst_arg=0
for arg in "${requested_sst_args[@]}"; do
	if [[ "$skip_sst_arg" -eq 1 ]]; then
		skip_sst_arg=0
		continue
	fi
	case "$arg" in
		--lib-path|--add-lib-path)
			skip_sst_arg=1
			;;
		--lib-path=*|--add-lib-path=*|--no-env-config)
			;;
		*)
			filtered_sst_args+=("$arg")
			;;
	esac
done
export GOLEM_SST_ARGS="--lib-path=$elements_lib_path --no-env-config"
if [[ "${#filtered_sst_args[@]}" -gt 0 ]]; then
	export GOLEM_SST_ARGS+=" ${filtered_sst_args[*]}"
fi

echo "[OK] SST core: $SST_CORE_PREFIX"
echo "[OK] SST elements install: $SST_ELEMENTS_INSTALL_PREFIX"
echo "[OK] GOLEM_SST_ARGS=$GOLEM_SST_ARGS"
