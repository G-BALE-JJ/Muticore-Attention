#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
WORKTREE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
RAMULATOR2_ROOT="${SST_RAMULATOR2_PREFIX:-$WORKTREE_ROOT/deps/ramulator2-2.1}"
RAMULATOR2_COMMIT="${RAMULATOR2_COMMIT:-72427a1bba3771564c4fb0e494ba02242fd1eaa7}"
RAMULATOR2_GIT_URL="${RAMULATOR2_GIT_URL:-https://gh-proxy.com/https://github.com/CMU-SAFARI/ramulator2.git}"
RAMULATOR2_ARCHIVE_SHA256="${RAMULATOR2_ARCHIVE_SHA256:-4d296240d0478ee91896f93daa0a623696e1c2c8b63888b51ed35c01c86af19b}"
JOBS="${JOBS:-$(nproc)}"
RAMULATOR2_COMMIT_FILE="$RAMULATOR2_ROOT/.ramulator2-commit"
RAMULATOR2_MANIFEST_FILE="$RAMULATOR2_ROOT/.ramulator2-source.sha256"

generate_source_manifest() {
	local root="$1"
	(
		cd "$root"
		find . -type f \
			! -path './.git/*' \
			! -path './ext/*' \
			! -path './build-sst/*' \
			! -name 'libramulator.so' \
			! -name '.ramulator2-commit' \
			! -name '.ramulator2-source.sha256' \
			-print0 | LC_ALL=C sort -z | xargs -0 sha256sum
	)
}

if [[ -d "$RAMULATOR2_ROOT/.git" ]]; then
	if [[ -n "$(git -C "$RAMULATOR2_ROOT" status --porcelain --untracked-files=no)" ]]; then
		echo "[ERROR] Ramulator2 checkout has local changes: $RAMULATOR2_ROOT" >&2
		exit 1
	fi
	if [[ "$(git -C "$RAMULATOR2_ROOT" rev-parse HEAD)" != "$RAMULATOR2_COMMIT" ]]; then
		git -C "$RAMULATOR2_ROOT" fetch --depth 1 origin "$RAMULATOR2_COMMIT"
		git -C "$RAMULATOR2_ROOT" checkout --detach "$RAMULATOR2_COMMIT"
	fi
elif [[ -f "$RAMULATOR2_COMMIT_FILE" ]]; then
	if [[ "$(<"$RAMULATOR2_COMMIT_FILE")" != "$RAMULATOR2_COMMIT" ]]; then
		echo "[ERROR] Ramulator2 archive commit does not match $RAMULATOR2_COMMIT: $RAMULATOR2_ROOT" >&2
		exit 1
	fi
	if [[ ! -f "$RAMULATOR2_MANIFEST_FILE" ]] || ! generate_source_manifest "$RAMULATOR2_ROOT" | cmp -s - "$RAMULATOR2_MANIFEST_FILE"; then
		echo "[ERROR] Ramulator2 archive source has changed: $RAMULATOR2_ROOT" >&2
		exit 1
	fi
elif [[ -e "$RAMULATOR2_ROOT" ]]; then
	echo "[ERROR] Existing Ramulator2 directory has no Git metadata or commit marker: $RAMULATOR2_ROOT" >&2
	exit 1
else
	ramulator2_parent="$(dirname "$RAMULATOR2_ROOT")"
	mkdir -p "$ramulator2_parent"
	temp_dir="$(mktemp -d "$ramulator2_parent/.ramulator2-source.XXXXXX")"
	trap 'rm -rf -- "$temp_dir"' EXIT
	candidate="$temp_dir/source"
	if git clone --no-checkout "$RAMULATOR2_GIT_URL" "$candidate" && \
		git -C "$candidate" checkout --detach "$RAMULATOR2_COMMIT" && \
		[[ "$(git -C "$candidate" rev-parse HEAD)" == "$RAMULATOR2_COMMIT" ]]; then
		mv "$candidate" "$RAMULATOR2_ROOT"
	else
		echo "[WARN] Mirror clone failed; downloading the pinned GitHub source archive" >&2
		rm -rf -- "$candidate"
		curl -L --fail \
			"https://codeload.github.com/CMU-SAFARI/ramulator2/tar.gz/$RAMULATOR2_COMMIT" \
			-o "$temp_dir/ramulator2.tar.gz"
		echo "$RAMULATOR2_ARCHIVE_SHA256  $temp_dir/ramulator2.tar.gz" | sha256sum -c -
		tar -xzf "$temp_dir/ramulator2.tar.gz" -C "$temp_dir"
		candidate="$temp_dir/ramulator2-$RAMULATOR2_COMMIT"
		printf '%s\n' "$RAMULATOR2_COMMIT" > "$candidate/.ramulator2-commit"
		generate_source_manifest "$candidate" > "$candidate/.ramulator2-source.sha256"
		mv "$candidate" "$RAMULATOR2_ROOT"
	fi
	rm -rf -- "$temp_dir"
	trap - EXIT
fi

fetch_dependency() {
	local name="$1" version="$2" url="$3"
	local target="$RAMULATOR2_ROOT/ext/$name"
	if [[ -f "$target/CMakeLists.txt" ]]; then return; fi
	if [[ -e "$target" ]]; then
		echo "[ERROR] Incomplete dependency directory: $target" >&2
		exit 1
	fi
	local temp_dir
	temp_dir="$(mktemp -d)"
	curl -L --fail "$url" -o "$temp_dir/archive.tar.gz"
	mkdir -p "$temp_dir/source"
	tar -xzf "$temp_dir/archive.tar.gz" --strip-components=1 -C "$temp_dir/source"
	mv "$temp_dir/source" "$target"
	rm -rf "$temp_dir"
	echo "[OK] fetched $name $version"
}

fetch_dependency yaml-cpp 0.9.0 \
	https://github.com/jbeder/yaml-cpp/archive/refs/tags/yaml-cpp-0.9.0.tar.gz
fetch_dependency fmt 10.2.1 \
	https://github.com/fmtlib/fmt/archive/refs/tags/10.2.1.tar.gz

RAMULATOR2_BUILD_DIR="$RAMULATOR2_ROOT/build-sst"
RAMULATOR2_CMAKE_CACHE="$RAMULATOR2_BUILD_DIR/CMakeCache.txt"
if [[ -f "$RAMULATOR2_CMAKE_CACHE" ]] && \
	! grep -Fqx "CMAKE_HOME_DIRECTORY:INTERNAL=$RAMULATOR2_ROOT" "$RAMULATOR2_CMAKE_CACHE"; then
	echo "[INFO] Removing relocated Ramulator2 CMake cache: $RAMULATOR2_BUILD_DIR"
	rm -rf -- "$RAMULATOR2_BUILD_DIR"
fi

cmake -S "$RAMULATOR2_ROOT" -B "$RAMULATOR2_BUILD_DIR" \
	-DCMAKE_BUILD_TYPE=Release \
	-DRAMULATOR_PYTHON_BINDINGS=OFF \
	-DFETCHCONTENT_FULLY_DISCONNECTED=ON
cmake --build "$RAMULATOR2_BUILD_DIR" --parallel "$JOBS"
test -f "$RAMULATOR2_ROOT/libramulator.so"
echo "[OK] Ramulator2 $RAMULATOR2_COMMIT: $RAMULATOR2_ROOT/libramulator.so"
