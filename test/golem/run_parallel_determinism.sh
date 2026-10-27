#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="$SCRIPT_DIR/run_noc_dma_pipeline.sh"
CHECKER="$SCRIPT_DIR/stats/check_parallel_determinism.py"

RANK_LIST="${GOLEM_DETERMINISM_RANKS:-1,4,16}"
THREAD_LIST="${GOLEM_DETERMINISM_THREADS:-1}"
ARTIFACT_ROOT="${GOLEM_ARTIFACT_ROOT:-$SCRIPT_DIR/artifacts}"
EXTRA_SST_ARGS="${GOLEM_DETERMINISM_EXTRA_SST_ARGS:-}"
PREPARE=0
PROFILE="${GOLEM_DETERMINISM_PROFILE:-0}"
VERBOSE="${GOLEM_DETERMINISM_VERBOSE:-0}"
EXPLICIT_PARTITION=0
PIPELINE_ARGS=()

usage() {
	cat <<'EOF'
Usage: run_parallel_determinism.sh [options] [-- pipeline-options]

Run the same workload with a rank/thread matrix and require identical
simulation-time and architectural-cycle results.

Options:
  --ranks LIST          Comma-separated MPI rank counts (default: 1,4,16)
  --threads LIST        Comma-separated SST thread counts (default: 1)
  --artifact-root PATH  Reuse build/HBM artifacts under PATH
  --extra-sst-args ARGS Additional arguments passed to SST
  --prepare             Generate tensors/HBM and build on the first run
  --profile             Save SST timing JSON and partition maps for each case
  --verbose             Stream full pipeline output instead of per-case logs
  --explicit-partition  Use the manager-preserving sst.self partition map
  -h, --help            Show this help

Without --prepare, an existing test binary and HBM image are required.
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--ranks)
			RANK_LIST="$2"
			shift 2
			;;
		--threads)
			THREAD_LIST="$2"
			shift 2
			;;
		--artifact-root)
			ARTIFACT_ROOT="$2"
			shift 2
			;;
		--extra-sst-args)
			EXTRA_SST_ARGS="$2"
			shift 2
			;;
		--prepare)
			PREPARE=1
			shift
			;;
		--profile)
			PROFILE=1
			shift
			;;
		--verbose)
			VERBOSE=1
			shift
			;;
		--explicit-partition)
			EXPLICIT_PARTITION=1
			shift
			;;
		-h|--help)
			usage
			exit 0
			;;
		--)
			shift
			PIPELINE_ARGS=("$@")
			break
			;;
		*)
			echo "[ERROR] unknown option: $1" >&2
			usage >&2
			exit 2
			;;
	esac
done

validate_list() {
	local label="$1"
	local value="$2"
	local item
	IFS=',' read -r -a items <<< "$value"
	if [[ ${#items[@]} -eq 0 ]]; then
		echo "[ERROR] $label list must not be empty" >&2
		exit 2
	fi
	for item in "${items[@]}"; do
		if [[ ! "$item" =~ ^[1-9][0-9]*$ ]]; then
			echo "[ERROR] invalid $label value: $item" >&2
			exit 2
		fi
	done
}

validate_list "rank" "$RANK_LIST"
validate_list "thread" "$THREAD_LIST"
IFS=',' read -r -a RANKS <<< "$RANK_LIST"
IFS=',' read -r -a THREADS <<< "$THREAD_LIST"

STAMP="$(date +%Y%m%d_%H%M%S)_$$"
STATS_ROOT="$ARTIFACT_ROOT/stats/parallel_determinism"
SUMMARY="$STATS_ROOT/${STAMP}_run_summary.csv"
REPORT="$STATS_ROOT/${STAMP}_comparison.csv"
PROFILE_ROOT="$STATS_ROOT/${STAMP}_profile"
mkdir -p "$STATS_ROOT"
if [[ "$PROFILE" == "1" ]]; then
	mkdir -p "$PROFILE_ROOT"
fi

RUN_IDS=()
CREDIT_OWNER_TABLES=()
FIRST_RUN=1
for threads in "${THREADS[@]}"; do
	for ranks in "${RANKS[@]}"; do
		run_id="det_${STAMP}_r${ranks}_t${threads}"
		RUN_IDS+=("$run_id")
		skip_prepare=1
		if [[ "$PREPARE" == "1" && "$FIRST_RUN" == "1" ]]; then
			skip_prepare=0
		fi

		echo "[RUN] ranks=$ranks threads=$threads run_id=$run_id"
			sst_args="$EXTRA_SST_ARGS"
		case_partitioner="${GOLEM_MPI_PARTITIONER:-sst.simple}"
		if [[ "$EXPLICIT_PARTITION" == "1" ]]; then
			case_partitioner="sst.self"
		fi
		if [[ "$PROFILE" == "1" ]]; then
			sst_args+=" --print-timing-info=2"
			sst_args+=" --timing-info-json=$PROFILE_ROOT/${run_id}_timing.json"
			sst_args+=" --output-json=$PROFILE_ROOT/${run_id}_graph.json"
			sst_args+=" --output-partition"
		fi
		pipeline_log="$STATS_ROOT/${run_id}_pipeline.log"
		run_case() {
			GOLEM_ARTIFACT_ROOT="$ARTIFACT_ROOT" \
			GOLEM_RUN_SUMMARY_CSV="$SUMMARY" \
			GOLEM_RUN_ID="$run_id" \
			GOLEM_EXPLICIT_PARTITION="$EXPLICIT_PARTITION" \
				GOLEM_MPI_PARTITIONER="$case_partitioner" \
				GOLEM_SST_THREADS="$threads" \
			GOLEM_SST_ARGS="$sst_args" \
			GOLEM_SKIP_TENSOR_GEN="$skip_prepare" \
			GOLEM_SKIP_HBM_GEN="$skip_prepare" \
			GOLEM_SKIP_BUILD="$skip_prepare" \
			GOLEM_BENCH_QUIET_LOGS=1 \
			"$PIPELINE" --mpi-ranks "$ranks" --log "${run_id}.log" "${PIPELINE_ARGS[@]}"
		}
		if [[ "$VERBOSE" == "1" ]]; then
			run_case
		elif ! run_case >"$pipeline_log" 2>&1; then
			echo "[FAIL] ranks=$ranks threads=$threads; tail of $pipeline_log:" >&2
			tail -n 80 "$pipeline_log" >&2
			exit 1
		else
			echo "[DONE] ranks=$ranks threads=$threads log=$pipeline_log"
		fi
		owner_table="$(find "$ARTIFACT_ROOT/stats" -type f -path "*/$run_id/credit_owner_table.csv" -print -quit)"
		if [[ -z "$owner_table" ]]; then
			echo "[FAIL] missing credit owner table for $run_id" >&2
			exit 1
		fi
		CREDIT_OWNER_TABLES+=("$owner_table")
		FIRST_RUN=0
	done
done

echo "[OK] summary: $SUMMARY"
if [[ ${#RUN_IDS[@]} -ge 2 ]]; then
	checker_args=(--summary "$SUMMARY" --report "$REPORT")
	for run_id in "${RUN_IDS[@]}"; do
		checker_args+=(--run-id "$run_id")
	done
	python3 "$CHECKER" "${checker_args[@]}"
	owner_baseline="${CREDIT_OWNER_TABLES[0]}"
	for owner_table in "${CREDIT_OWNER_TABLES[@]:1}"; do
		if ! cmp -s "$owner_baseline" "$owner_table"; then
			echo "[FAIL] memory-node credit owner state differs: $owner_table" >&2
			diff -u "$owner_baseline" "$owner_table" >&2 || true
			exit 1
		fi
	done
	echo "[PASS] memory-node credit owner tables match across ${#CREDIT_OWNER_TABLES[@]} runs"
	echo "[OK] comparison: $REPORT"
else
	echo "[INFO] one case selected; deterministic comparison skipped"
fi
if [[ "$PROFILE" == "1" ]]; then
	echo "[OK] SST profiles: $PROFILE_ROOT"
fi
