#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
WORKTREE_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd -P)"
PIPELINE="$SCRIPT_DIR/run_noc_dma_pipeline.sh"
ROOT="${GOLEM_MEMORY_CROSS_ROOT:-$SCRIPT_DIR/artifacts/memory_backend_cross_$(date +%Y%m%d_%H%M%S)}"
TIMEOUT="${GOLEM_MEMORY_CROSS_TIMEOUT:-2h}"
DIMS=(2048 4096 6144)
BACKENDS=(dramsim3 ramulator2)

# shellcheck source=/dev/null
source "$WORKTREE_ROOT/scripts/env_local_install.sh" >/dev/null
mapfile -t PHYSICAL_CPUS < <(
	lscpu -p=CPU,CORE,SOCKET | awk -F, '$1 !~ /^#/ {key=$3 ":" $2; if (!seen[key]++) print $1}' | head -n 8
)
if (( ${#PHYSICAL_CPUS[@]} < 8 )); then
	echo "[ERROR] fewer than eight physical CPUs available" >&2
	exit 2
fi
CPU_SETS=(
	"${PHYSICAL_CPUS[0]},${PHYSICAL_CPUS[1]},${PHYSICAL_CPUS[2]},${PHYSICAL_CPUS[3]}"
	"${PHYSICAL_CPUS[4]},${PHYSICAL_CPUS[5]},${PHYSICAL_CPUS[6]},${PHYSICAL_CPUS[7]}"
)

mkdir -p "$ROOT"
printf 'dimension\tbackend\tstatus\tcpus\tcase_root\n' > "$ROOT/status.tsv"

run_case() {
	local dim="$1" backend="$2" slot="$3" skip_build="$4"
	local case_root="$ROOT/s${dim}/${backend}"
	local shared_binary="$ROOT/s${dim}/bin/test_noc_dma"
	local cpus="${CPU_SETS[$slot]}"
	mkdir -p "$case_root"
	(
		export GOLEM_ARTIFACT_ROOT="$case_root/artifacts"
		export GOLEM_TEST_BINARY="$shared_binary"
		export GOLEM_BUILD_METADATA_FILE="${shared_binary}.build.env"
		export GOLEM_RUN_SUMMARY_CSV="$case_root/run_summary.csv"
		export GOLEM_RUN_ID="cross_s${dim}_${backend}"
		export GOLEM_MEMORY_BACKEND="$backend"
		export GOLEM_SKIP_BUILD="$skip_build"
		export GOLEM_HBM_DUMP_OUTPUT=0
		export GOLEM_MPI_ARGS="--bind-to cpu-list:ordered --cpu-list $cpus"
		taskset -c "$cpus" timeout "$TIMEOUT" "$PIPELINE" \
			--dim "s${dim}" --sim-mode full-timing --mpi-ranks 4 --sst-threads 1 \
			--memory-backend "$backend" --cpu-binding-mode cross_validation \
			--no-hbm-dump-output --log "cross_s${dim}_${backend}.log"
	) > "$case_root/pipeline.log" 2>&1
}

for dim in "${DIMS[@]}"; do
	mkdir -p "$ROOT/s${dim}/bin"
	echo "[START] s${dim}: dramsim3 on ${CPU_SETS[0]}"
	run_case "$dim" dramsim3 0 0 &
	first_pid=$!
	first_log="$ROOT/s${dim}/dramsim3/pipeline.log"
	while kill -0 "$first_pid" 2>/dev/null; do
		if [[ -f "$first_log" ]] && grep -q '^\[OK\] Test binary ready' "$first_log"; then
			break
		fi
		sleep 1
	done
	if ! kill -0 "$first_pid" 2>/dev/null; then
		wait "$first_pid" || true
		echo "[FAIL] s${dim} dramsim3 failed before shared binary was ready" >&2
		printf '%s\t%s\tFAIL\t%s\t%s\n' "$dim" dramsim3 "${CPU_SETS[0]}" "$ROOT/s${dim}/dramsim3" >> "$ROOT/status.tsv"
		exit 1
	fi

	echo "[START] s${dim}: ramulator2 on ${CPU_SETS[1]}"
	run_case "$dim" ramulator2 1 1 &
	second_pid=$!
	first_status=0
	second_status=0
	wait "$first_pid" || first_status=$?
	wait "$second_pid" || second_status=$?
	for entry in "dramsim3:$first_status:0" "ramulator2:$second_status:1"; do
		IFS=: read -r backend status slot <<< "$entry"
		state=PASS
		if (( status != 0 )); then state=FAIL; fi
		printf '%s\t%s\t%s\t%s\t%s\n' \
			"$dim" "$backend" "$state" "${CPU_SETS[$slot]}" "$ROOT/s${dim}/${backend}" \
			>> "$ROOT/status.tsv"
		echo "[$state] s${dim} $backend"
	done
	if (( first_status != 0 || second_status != 0 )); then
		exit 1
	fi
done

python3 - "$ROOT" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
wanted = [
    "gemm_m", "memory_backend", "gemm_system_latency_cycles",
    "exec_system_array_utilization_pct", "hbm_read_command_utilization_pct",
    "hbm_backend_service_window_utilization_pct",
    "memory_avg_read_latency_cycles", "memory_p95_read_latency_bucket_cycles",
    "pipeline_efficiency_pct", "pipeline_avg_bubble_cycles_per_core",
    "pipeline_intra_window_avg_bubble_cycles_per_core",
    "pipeline_inter_window_avg_bubble_cycles_per_core",
    "pipeline_inter_macro_avg_bubble_cycles_per_core",
]
rows = []
for dim in (2048, 4096, 6144):
    for backend in ("dramsim3", "ramulator2"):
        path = root / f"s{dim}" / backend / "run_summary.csv"
        with path.open(newline="") as handle:
            row = list(csv.DictReader(handle))[-1]
        rows.append({key: row.get(key, "") for key in wanted})
with (root / "cross_validation.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=wanted)
    writer.writeheader()
    writer.writerows(rows)
print(f"[OK] cross-validation summary: {root / 'cross_validation.csv'}")
PY

echo "[OK] all backend cross-validation runs complete: $ROOT"
