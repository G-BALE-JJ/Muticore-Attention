#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ $# -ne 1 || ! "$1" =~ ^exp-[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Usage: scripts/new_experiment.sh exp-feature-name" >&2
  exit 2
fi
name="$1"
destination="$PROJECT_ROOT/$name"
if [[ -e "$destination" || -L "$destination" ]]; then
  echo "Workspace already exists: $destination" >&2
  exit 1
fi
if [[ -n "$(git -C "$PROJECT_ROOT" status --porcelain --untracked-files=no)" ]]; then
  echo "Commit the public architecture changes before creating a feature workspace." >&2
  exit 1
fi
git -C "$PROJECT_ROOT" worktree add -b "$name" "$destination" HEAD
# Only dependencies and archived evidence are shared. Builds and installs are private.
ln -s "$PROJECT_ROOT/deps" "$destination/deps"
ln -s "$PROJECT_ROOT/test/history" "$destination/test/history"
ln -s "$PROJECT_ROOT/test/history/Muticore-Attention-all-gemm/results" "$destination/results"
cat <<EOF
Created: $destination
Branch:  $name

Build the feature workspace:
  cd "$destination"
  scripts/build_and_install_local.sh

Run the four stages against it:
  test/GEMM/run.sh --check
  test/Attention/run.sh --check
  test/GQA/run.sh --check
  test/prefill/run.sh --check

After simulation and paired validation, review and merge from the public root:
  git -C "$PROJECT_ROOT" diff main..."$name"
  git -C "$PROJECT_ROOT" merge --no-ff "$name"
EOF
