#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="${GOLEM_ARTIFACT_ROOT:-$SCRIPT_DIR/artifacts}"
RUN_SUMMARY_CSV="${GOLEM_RUN_SUMMARY_CSV:-$ARTIFACT_ROOT/stats/run_summary.csv}"
OVERLAP_ROOT="${GOLEM_OVERLAP_ROOT:-$ARTIFACT_ROOT/stats/overlap0}"
OUT_DIR="${GOLEM_COMPARISON_OUT_DIR:-$ARTIFACT_ROOT/stats/analysis/comparison}"
PLOT_SCRIPT="$SCRIPT_DIR/stats/plot_run_comparison.py"

BASE_RUN_ID=""
OPT_RUN_ID=""
BASE_FEATURE_LABEL=""
OPT_FEATURE_LABEL=""
TITLE_SUFFIX=""

usage() {
    cat <<'EOF'
Usage:
  run_latest_comparison.sh [options]

Default behavior:
  - pick the latest two runs recorded in run_summary.csv
  - compare their stats directories with stats/plot_run_comparison.py

Options:
  --base-run-id ID            Explicit base run id (e.g. run_20260328_181824_939149)
  --opt-run-id ID             Explicit optimized run id
  --base-feature-label TEXT   Manual feature label for base legend
  --opt-feature-label TEXT    Manual feature label for opt legend
  --title-suffix TEXT         Manual subtitle for the comparison plots
  -h, --help                  Show this message

Environment:
  GOLEM_RUN_SUMMARY_CSV       Override run summary path
  GOLEM_OVERLAP_ROOT          Override overlap stats root
  GOLEM_COMPARISON_OUT_DIR    Override comparison output directory
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-run-id)
            BASE_RUN_ID="$2"
            shift 2
            ;;
        --opt-run-id)
            OPT_RUN_ID="$2"
            shift 2
            ;;
        --base-feature-label)
            BASE_FEATURE_LABEL="$2"
            shift 2
            ;;
        --opt-feature-label)
            OPT_FEATURE_LABEL="$2"
            shift 2
            ;;
        --title-suffix)
            TITLE_SUFFIX="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERR] Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ ! -f "$PLOT_SCRIPT" ]]; then
    echo "[ERR] Missing plot script: $PLOT_SCRIPT" >&2
    exit 1
fi

if [[ -z "$BASE_RUN_ID" || -z "$OPT_RUN_ID" ]]; then
    if [[ ! -f "$RUN_SUMMARY_CSV" ]]; then
        echo "[ERR] Missing run summary: $RUN_SUMMARY_CSV" >&2
        exit 1
    fi

    mapfile -t AUTO_RUN_IDS < <(
        python3 - "$RUN_SUMMARY_CSV" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = list(csv.DictReader(path.open(newline="")))
seen = []
for row in reversed(rows):
    log_file = row.get("log_file", "")
    if not log_file:
        continue

    # Stats directory name is derived from log filename stem, e.g.
    # test_default_run_...log -> run_...
    # test_default_sweep_mk_...log -> sweep_mk_...
    stem = Path(log_file).stem
    if stem.startswith("test_default_"):
        run_id = stem[len("test_default_"):]
    elif stem.startswith("test_"):
        run_id = stem[len("test_"):]
    else:
        run_id = stem

    if not run_id:
        continue

    if run_id not in seen:
        seen.append(run_id)
    if len(seen) == 2:
        break
for run_id in reversed(seen):
    print(run_id)
PY
    )

    if [[ ${#AUTO_RUN_IDS[@]} -lt 2 ]]; then
        echo "[ERR] Could not identify two runs from $RUN_SUMMARY_CSV" >&2
        exit 1
    fi

    if [[ -z "$BASE_RUN_ID" ]]; then
        BASE_RUN_ID="${AUTO_RUN_IDS[0]}"
    fi
    if [[ -z "$OPT_RUN_ID" ]]; then
        OPT_RUN_ID="${AUTO_RUN_IDS[1]}"
    fi
fi

BASE_RUN_DIR="$OVERLAP_ROOT/$BASE_RUN_ID"
OPT_RUN_DIR="$OVERLAP_ROOT/$OPT_RUN_ID"

if [[ ! -d "$BASE_RUN_DIR" ]]; then
    echo "[ERR] Base run dir not found: $BASE_RUN_DIR" >&2
    exit 1
fi
if [[ ! -d "$OPT_RUN_DIR" ]]; then
    echo "[ERR] Opt run dir not found: $OPT_RUN_DIR" >&2
    exit 1
fi

CMD=(
    python3 "$PLOT_SCRIPT"
    --base-run-dir "$BASE_RUN_DIR"
    --opt-run-dir "$OPT_RUN_DIR"
    --out-dir "$OUT_DIR"
)

if [[ -n "$BASE_FEATURE_LABEL" ]]; then
    CMD+=(--base-feature-label "$BASE_FEATURE_LABEL")
fi
if [[ -n "$OPT_FEATURE_LABEL" ]]; then
    CMD+=(--opt-feature-label "$OPT_FEATURE_LABEL")
fi
if [[ -n "$TITLE_SUFFIX" ]]; then
    CMD+=(--title-suffix "$TITLE_SUFFIX")
fi

echo "[INFO] base run: $BASE_RUN_ID"
echo "[INFO] opt  run: $OPT_RUN_ID"
if [[ -n "$BASE_FEATURE_LABEL" ]]; then
    echo "[INFO] base label override: $BASE_FEATURE_LABEL"
fi
if [[ -n "$OPT_FEATURE_LABEL" ]]; then
    echo "[INFO] opt label override: $OPT_FEATURE_LABEL"
fi
if [[ -n "$TITLE_SUFFIX" ]]; then
    echo "[INFO] title suffix override: $TITLE_SUFFIX"
fi

"${CMD[@]}"
