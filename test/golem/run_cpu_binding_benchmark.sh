#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="$SCRIPT_DIR/run_noc_dma_pipeline.sh"
CHECKER="$SCRIPT_DIR/stats/check_parallel_determinism.py"
SUMMARIZER="$SCRIPT_DIR/stats/summarize_parallel_scaling.py"

RANKS=8
THREADS=2
REPEATS=3
ARTIFACT_ROOT="${GOLEM_ARTIFACT_ROOT:-$SCRIPT_DIR/artifacts}"
PIPELINE_ARGS=()

usage() {
	cat <<'EOF'
Usage: run_cpu_binding_benchmark.sh [options] [-- pipeline-options]

Compare the same SST rank/thread configuration on distinct physical cores and
on SMT siblings. Runs alternate physical/SMT order to reduce thermal bias.

Options:
  --ranks N             MPI ranks (default: 8)
  --threads N           SST threads per rank (default: 2)
  --repeats N           Repeats per binding mode (default: 3)
  --artifact-root PATH  Prepared artifact root
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--ranks) RANKS="$2"; shift 2 ;;
		--threads) THREADS="$2"; shift 2 ;;
		--repeats) REPEATS="$2"; shift 2 ;;
		--artifact-root) ARTIFACT_ROOT="$2"; shift 2 ;;
		-h|--help) usage; exit 0 ;;
		--) shift; PIPELINE_ARGS=("$@"); break ;;
		*) echo "[ERROR] unknown option: $1" >&2; usage >&2; exit 2 ;;
	esac
done

for value in "$RANKS" "$THREADS" "$REPEATS"; do
	if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
		echo "[ERROR] ranks, threads, and repeats must be positive integers" >&2
		exit 2
	fi
done

declare -A CORE_CPUS=()
CORE_KEYS=()
while IFS=, read -r cpu core socket; do
	[[ "$cpu" == \#* || -z "$cpu" ]] && continue
	key="${socket}:${core}"
	if [[ -z "${CORE_CPUS[$key]+x}" ]]; then
		CORE_KEYS+=("$key")
		CORE_CPUS[$key]="$cpu"
	else
		CORE_CPUS[$key]+=",$cpu"
	fi
done < <(lscpu -p=CPU,CORE,SOCKET)

logical_workers=$((RANKS * THREADS))
if (( logical_workers > ${#CORE_KEYS[@]} )); then
	echo "[ERROR] physical mode needs $logical_workers cores; found ${#CORE_KEYS[@]}" >&2
	exit 2
fi

physical_cpus=()
for ((index = 0; index < logical_workers; index++)); do
	IFS=, read -r first_cpu _ <<< "${CORE_CPUS[${CORE_KEYS[$index]}]}"
	physical_cpus+=("$first_cpu")
done

smt_cpus=()
for key in "${CORE_KEYS[@]}"; do
	IFS=, read -r -a siblings <<< "${CORE_CPUS[$key]}"
	if (( ${#siblings[@]} < 2 )); then
		continue
	fi
	for cpu in "${siblings[@]}"; do
		smt_cpus+=("$cpu")
		if (( ${#smt_cpus[@]} == logical_workers )); then
			break 2
		fi
	done
done
if (( ${#smt_cpus[@]} != logical_workers )); then
	echo "[ERROR] SMT mode needs $logical_workers sibling CPUs; found ${#smt_cpus[@]}" >&2
	exit 2
fi

join_cpus() {
	local IFS=,
	JOINED_CPUS="$*"
}
join_cpus "${physical_cpus[@]}"
PHYSICAL_CPUSET="$JOINED_CPUS"
join_cpus "${smt_cpus[@]}"
SMT_CPUSET="$JOINED_CPUS"

STAMP="$(date +%Y%m%d_%H%M%S)_$$"
BENCH_ROOT="$ARTIFACT_ROOT/stats/cpu_binding/$STAMP"
SUMMARY="$BENCH_ROOT/run_summary.csv"
REPORT="$BENCH_ROOT/binding_summary.csv"
COMPARISON="$BENCH_ROOT/determinism.csv"
mkdir -p "$BENCH_ROOT/pipeline_logs" "$BENCH_ROOT/timing"

echo "[PLAN] ranks=$RANKS threads=$THREADS repeats=$REPEATS"
echo "[PLAN] physical cpuset=$PHYSICAL_CPUSET"
echo "[PLAN] smt cpuset=$SMT_CPUSET"

RUN_IDS=()
for ((repeat = 1; repeat <= REPEATS; repeat++)); do
	for mode in physical smt; do
		if [[ "$mode" == "physical" ]]; then
			cpuset="$PHYSICAL_CPUSET"
		else
			cpuset="$SMT_CPUSET"
		fi
		run_id="bind_${STAMP}_${mode}_rep${repeat}"
		RUN_IDS+=("$run_id")
		echo "[RUN] mode=$mode repeat=$repeat run_id=$run_id"
		pipeline_log="$BENCH_ROOT/pipeline_logs/${run_id}.log"
		if ! GOLEM_ARTIFACT_ROOT="$ARTIFACT_ROOT" \
			GOLEM_RUN_SUMMARY_CSV="$SUMMARY" \
			GOLEM_RUN_ID="$run_id" \
			GOLEM_EXPLICIT_PARTITION=1 \
			GOLEM_MPI_PARTITIONER=sst.self \
			GOLEM_MPI_ARGS="--bind-to none" \
			GOLEM_SST_ARGS="--print-timing-info=2 --timing-info-json=$BENCH_ROOT/timing/${run_id}.json" \
			GOLEM_SKIP_TENSOR_GEN=1 \
			GOLEM_SKIP_HBM_GEN=1 \
			GOLEM_SKIP_BUILD=1 \
			GOLEM_BENCH_QUIET_LOGS=1 \
			"$PIPELINE" --mpi-ranks "$RANKS" --sst-threads "$THREADS" \
				--cpuset "$cpuset" --cpu-binding-mode "$mode" \
				--log "${run_id}.log" "${PIPELINE_ARGS[@]}" >"$pipeline_log" 2>&1; then
			echo "[FAIL] mode=$mode repeat=$repeat; tail of $pipeline_log:" >&2
			tail -n 80 "$pipeline_log" >&2
			exit 1
		fi
		echo "[DONE] mode=$mode repeat=$repeat"
	done
done

checker_args=(--summary "$SUMMARY" --report "$COMPARISON")
for run_id in "${RUN_IDS[@]}"; do
	checker_args+=(--run-id "$run_id")
done
python3 "$CHECKER" "${checker_args[@]}"
python3 "$SUMMARIZER" --summary "$SUMMARY" --output "$REPORT"
echo "[OK] raw summary: $SUMMARY"
echo "[OK] binding summary: $REPORT"
