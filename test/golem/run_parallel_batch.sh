#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="$SCRIPT_DIR/run_noc_dma_pipeline.sh"
SLOTS=4
REPEAT=4
RANKS_PER_CASE=4
BASE_ARTIFACT_ROOT="${GOLEM_ARTIFACT_ROOT:-$SCRIPT_DIR/artifacts}"
BATCH_ROOT=""
CASE_FILE=""
DRY_RUN=0
PIPELINE_ARGS=()

usage() {
	cat <<'EOF'
Usage: run_parallel_batch.sh [options] [-- common-pipeline-options]

Run a queue of independent simulations on fixed, disjoint physical-CPU slots.
Each case receives private HBM, log, stdout, and statistics directories. When
a case completes, the next queued case immediately reuses the released slot.

Options:
  --slots N                   Concurrent CPU slots (default: 4)
  --jobs N                    Deprecated alias for --slots
  --repeat N                  Identical cases when --case-file is absent (default: 4)
  --case-file PATH            TSV manifest: case name, then pipeline arguments
  --ranks-per-case N          MPI ranks/physical CPUs per case (currently must be 4)
  --ranks-per-job N           Deprecated alias for --ranks-per-case
  --base-artifact-root PATH   Source artifact tree with prepared HBM files
  --batch-root PATH           Output root (default: artifacts/batches/TIMESTAMP)
  --dry-run                   Print and validate the queue without preparing files
  -h, --help                  Show this help

Blank lines and lines beginning with # are ignored in the case file. Fields
must be separated by literal tab characters; shell quoting is neither needed
nor evaluated. Example:

  baseline<TAB>--sst-threads<TAB>1
  threaded<TAB>--sst-threads<TAB>2

Common options are applied first and per-case options second. All cases must be
compatible with the prepared test binary and HBM contract; the pipeline checks
their metadata before launching SST. The queue owns --mpi-ranks, --sst-threads,
--cpuset, and --cpu-binding-mode; cases may not override those resource options.
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--slots|--jobs)
			SLOTS="$2"
			shift 2
			;;
		--repeat)
			REPEAT="$2"
			shift 2
			;;
		--case-file)
			CASE_FILE="$2"
			shift 2
			;;
		--ranks-per-case|--ranks-per-job)
			RANKS_PER_CASE="$2"
			shift 2
			;;
		--base-artifact-root)
			BASE_ARTIFACT_ROOT="$2"
			shift 2
			;;
		--batch-root)
			BATCH_ROOT="$2"
			shift 2
			;;
		--dry-run)
			DRY_RUN=1
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

if [[ ! "$SLOTS" =~ ^[1-9][0-9]*$ ]]; then
	echo "[ERROR] --slots must be a positive integer" >&2
	exit 2
fi
if [[ ! "$REPEAT" =~ ^[1-9][0-9]*$ ]]; then
	echo "[ERROR] --repeat must be a positive integer" >&2
	exit 2
fi
if [[ "$RANKS_PER_CASE" != "4" ]]; then
	echo "[ERROR] the manager-preserving partition currently requires --ranks-per-case 4" >&2
	exit 2
fi

CASE_NAMES=()
CASE_ROWS=()
declare -A SEEN_CASE_NAMES=()

if [[ -n "$CASE_FILE" ]]; then
	if [[ ! -f "$CASE_FILE" ]]; then
		echo "[ERROR] case file not found: $CASE_FILE" >&2
		exit 2
	fi
	line_number=0
	while IFS= read -r row || [[ -n "$row" ]]; do
		line_number=$((line_number + 1))
		row="${row%$'\r'}"
		[[ -z "$row" || "$row" =~ ^[[:space:]]*# ]] && continue
		IFS=$'\t' read -r -a fields <<< "$row"
		case_name="${fields[0]}"
		if [[ ! "$case_name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
			echo "[ERROR] invalid case name at $CASE_FILE:$line_number: $case_name" >&2
			exit 2
		fi
		if [[ -n "${SEEN_CASE_NAMES[$case_name]+x}" ]]; then
			echo "[ERROR] duplicate case name at $CASE_FILE:$line_number: $case_name" >&2
			exit 2
		fi
		SEEN_CASE_NAMES[$case_name]=1
		CASE_NAMES+=("$case_name")
		CASE_ROWS+=("$row")
	done < "$CASE_FILE"
	if (( ${#CASE_NAMES[@]} == 0 )); then
		echo "[ERROR] case file contains no cases: $CASE_FILE" >&2
		exit 2
	fi
else
	for ((case_index = 0; case_index < REPEAT; case_index++)); do
		printf -v case_name 'repeat%03d' "$case_index"
		CASE_NAMES+=("$case_name")
		CASE_ROWS+=("$case_name")
	done
fi

validate_case_args() {
	local label="$1"
	shift
	local arg
	for arg in "$@"; do
		case "$arg" in
			--mpi-ranks|--sst-threads|--cpuset|--cpu-binding-mode)
				echo "[ERROR] $label may not override queue-owned resource option: $arg" >&2
				exit 2
				;;
		esac
	done
}

validate_case_args "common pipeline arguments" "${PIPELINE_ARGS[@]}"
if [[ -n "$CASE_FILE" ]]; then
	for case_index in "${!CASE_ROWS[@]}"; do
		IFS=$'\t' read -r -a fields <<< "${CASE_ROWS[$case_index]}"
		validate_case_args "case ${CASE_NAMES[$case_index]}" "${fields[@]:1}"
	done
fi

declare -A SEEN_CORES=()
PHYSICAL_CPUS=()
while IFS=, read -r cpu core socket; do
	[[ "$cpu" == \#* || -z "$cpu" ]] && continue
	key="${socket}:${core}"
	if [[ -z "${SEEN_CORES[$key]+x}" ]]; then
		SEEN_CORES[$key]=1
		PHYSICAL_CPUS+=("$cpu")
	fi
done < <(lscpu -p=CPU,CORE,SOCKET)

required_cpus=$((SLOTS * RANKS_PER_CASE))
if (( required_cpus > ${#PHYSICAL_CPUS[@]} )); then
	echo "[ERROR] need $required_cpus physical CPUs, found ${#PHYSICAL_CPUS[@]}" >&2
	exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)_$$"
BATCH_ROOT="${BATCH_ROOT:-$BASE_ARTIFACT_ROOT/batches/$STAMP}"
BASE_HBM_DIR="$BASE_ARTIFACT_ROOT/hbm"

if [[ "$DRY_RUN" != "1" && ! -d "$BASE_HBM_DIR" ]]; then
	echo "[ERROR] prepared HBM directory not found: $BASE_HBM_DIR" >&2
	exit 2
fi

CASE_ROOTS=()
CASE_RUN_IDS=()
CASE_STATUSES=()
ACTIVE_PIDS=()
declare -A PID_TO_SLOT=()
declare -A PID_TO_CASE=()

case_args_for_index() {
	local case_index="$1"
	local -a fields=()
	CASE_ARGS=("${PIPELINE_ARGS[@]}")
	if [[ -n "$CASE_FILE" ]]; then
		IFS=$'\t' read -r -a fields <<< "${CASE_ROWS[$case_index]}"
		CASE_ARGS+=("${fields[@]:1}")
	fi
}

slot_cpu_list() {
	local slot="$1"
	local start=$((slot * RANKS_PER_CASE))
	local -a cpus=("${PHYSICAL_CPUS[@]:start:RANKS_PER_CASE}")
	local IFS=,
	SLOT_CPU_LIST="${cpus[*]}"
}

launch_case() {
	local case_index="$1"
	local slot="$2"
	local case_name="${CASE_NAMES[$case_index]}"
	local case_prefix run_id case_root case_log pid
	printf -v case_prefix '%03d_%s' "$case_index" "$case_name"
	run_id="batch_${STAMP}_${case_prefix}"
	case_root="$BATCH_ROOT/$case_prefix"
	case_log="$case_root/pipeline.log"
	slot_cpu_list "$slot"
	case_args_for_index "$case_index"

	CASE_ROOTS[$case_index]="$case_root"
	CASE_RUN_IDS[$case_index]="$run_id"
	echo "[START] case=$case_index name=$case_name slot=$slot ranks=$RANKS_PER_CASE cpus=$SLOT_CPU_LIST"

	mkdir -p "$case_root/hbm"
	cp -a --reflink=auto "$BASE_HBM_DIR/." "$case_root/hbm/"
	(
		export GOLEM_ARTIFACT_ROOT="$case_root"
		export GOLEM_HBM_DIR="$case_root/hbm"
		export GOLEM_RUN_ID="$run_id"
		export GOLEM_EXPLICIT_PARTITION=1
		export GOLEM_MPI_PARTITIONER=sst.self
		export GOLEM_MPI_ARGS="--bind-to cpu-list:ordered --cpu-list $SLOT_CPU_LIST"
		export GOLEM_SKIP_TENSOR_GEN=1
		export GOLEM_SKIP_HBM_GEN=1
		export GOLEM_SKIP_BUILD=1
		export GOLEM_BENCH_QUIET_LOGS=1
		"$PIPELINE" --mpi-ranks "$RANKS_PER_CASE" --sst-threads 1 \
			--cpu-binding-mode batch_physical --log "${run_id}.log" "${CASE_ARGS[@]}"
	) >"$case_log" 2>&1 &
	pid=$!
	ACTIVE_PIDS+=("$pid")
	PID_TO_SLOT[$pid]="$slot"
	PID_TO_CASE[$pid]="$case_index"
}

terminate_children() {
	local pid
	for pid in "${ACTIVE_PIDS[@]}"; do
		kill "$pid" 2>/dev/null || true
	done
	wait || true
	exit 130
}
trap terminate_children INT TERM

case_count=${#CASE_NAMES[@]}
echo "[QUEUE] cases=$case_count slots=$SLOTS ranks_per_case=$RANKS_PER_CASE physical_cpus=$required_cpus"

if [[ "$DRY_RUN" == "1" ]]; then
	for ((case_index = 0; case_index < case_count; case_index++)); do
		if (( case_index < SLOTS )); then
			slot_cpu_list "$case_index"
			echo "[PLAN] case=$case_index name=${CASE_NAMES[$case_index]} initial_slot=$case_index cpus=$SLOT_CPU_LIST"
		else
			echo "[PLAN] case=$case_index name=${CASE_NAMES[$case_index]} queued_for=next_free_slot"
		fi
	done
	echo "[OK] dry run: $case_count cases, $SLOTS slots, $required_cpus distinct physical CPUs"
	exit 0
fi

next_case=0
initial_cases=$SLOTS
if (( case_count < initial_cases )); then
	initial_cases=$case_count
fi
for ((slot = 0; slot < initial_cases; slot++)); do
	launch_case "$next_case" "$slot"
	next_case=$((next_case + 1))
done

failed=0
while (( ${#ACTIVE_PIDS[@]} > 0 )); do
	done_pid=""
	if wait -n -p done_pid "${ACTIVE_PIDS[@]}"; then
		status=0
	else
		status=$?
	fi
	done_slot="${PID_TO_SLOT[$done_pid]}"
	done_case="${PID_TO_CASE[$done_pid]}"
	case_name="${CASE_NAMES[$done_case]}"
	case_root="${CASE_ROOTS[$done_case]}"
	CASE_STATUSES[$done_case]="$status"

	remaining_pids=()
	for pid in "${ACTIVE_PIDS[@]}"; do
		[[ "$pid" != "$done_pid" ]] && remaining_pids+=("$pid")
	done
	ACTIVE_PIDS=("${remaining_pids[@]}")
	unset 'PID_TO_SLOT[$done_pid]' 'PID_TO_CASE[$done_pid]'

	if [[ "$status" == "0" ]]; then
		echo "[DONE] case=$done_case name=$case_name slot=$done_slot root=$case_root"
	else
		echo "[FAIL] case=$done_case name=$case_name slot=$done_slot status=$status" >&2
		tail -n 80 "$case_root/pipeline.log" >&2 || true
		failed=1
	fi

	if (( next_case < case_count )); then
		launch_case "$next_case" "$done_slot"
		next_case=$((next_case + 1))
	fi
done
trap - INT TERM

STATUS_FILE="$BATCH_ROOT/case_status.tsv"
mkdir -p "$BATCH_ROOT"
{
	printf 'case_index\tcase_name\texit_status\trun_id\tartifact_root\n'
	for ((case_index = 0; case_index < case_count; case_index++)); do
		printf '%d\t%s\t%s\t%s\t%s\n' \
			"$case_index" "${CASE_NAMES[$case_index]}" "${CASE_STATUSES[$case_index]}" \
			"${CASE_RUN_IDS[$case_index]}" "${CASE_ROOTS[$case_index]}"
	done
} > "$STATUS_FILE"

if [[ "$failed" == "1" ]]; then
	echo "[ERROR] one or more cases failed; status: $STATUS_FILE" >&2
	exit 1
fi

AGGREGATE="$BATCH_ROOT/run_summary.csv"
first=1
for case_root in "${CASE_ROOTS[@]}"; do
	summary="$case_root/stats/run_summary.csv"
	if [[ "$first" == "1" ]]; then
		head -n 1 "$summary" > "$AGGREGATE"
		first=0
	fi
	tail -n 1 "$summary" >> "$AGGREGATE"
done

echo "[PASS] $case_count cases completed through $SLOTS slots on $required_cpus physical CPUs"
echo "[OK] aggregate summary: $AGGREGATE"
echo "[OK] case status: $STATUS_FILE"
