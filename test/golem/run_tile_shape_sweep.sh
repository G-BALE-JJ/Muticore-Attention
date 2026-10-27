#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARAM_FILE="${1:-${GOLEM_TILE_SHAPE_PARAM_FILE:-$SCRIPT_DIR/tile_shape_sweep.csv}}"
SWEEP_TAG="${GOLEM_SWEEP_TAG:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_ROOT="${GOLEM_SWEEP_ROOT:-$SCRIPT_DIR/artifacts/stats/sweeps/tile_shape}"
SWEEP_DIR="${GOLEM_SWEEP_RUN_DIR:-$SWEEP_ROOT/run_$SWEEP_TAG}"
PLAN_TSV="$SWEEP_DIR/tile_shape_plan.tsv"
RESULT_CSV="$SWEEP_DIR/tile_shape_sweep.csv"
PARETO_CSV="$SWEEP_DIR/tile_shape_pareto.csv"
JOBS="${GOLEM_SWEEP_JOBS:-4}"
MPI_RANKS="${GOLEM_SWEEP_MPI_RANKS:-4}"
SST_THREADS="${GOLEM_SWEEP_SST_THREADS:-1}"
SIM_MODE="${GOLEM_SWEEP_SIM_MODE:-full-timing}"
RUN_TIMEOUT="${GOLEM_TILE_SHAPE_SWEEP_TIMEOUT:-${GOLEM_SWEEP_TIMEOUT:-}}"
DRY_RUN="${GOLEM_SWEEP_DRY_RUN:-0}"
CPU_SETS_RAW="${GOLEM_SWEEP_CPU_SETS:-0,1,2,3;4,5,6,7;8,9,10,11;12,13,14,15}"
COLOR_MODE="${GOLEM_SWEEP_COLOR:-auto}"
SWEEP_START_EPOCH="$(date +%s)"

UI_COLOR_ENABLED=0
if [[ "$COLOR_MODE" == "always" ]] ||
   { [[ "$COLOR_MODE" == "auto" ]] && [[ -t 1 ]] && [[ "${TERM:-dumb}" != "dumb" ]] && [[ -z "${NO_COLOR:-}" ]]; }; then
	UI_COLOR_ENABLED=1
fi

ui_color() {
	local code="$1"
	shift
	if [[ "$UI_COLOR_ENABLED" -eq 1 ]]; then
		printf '\033[%sm%s\033[0m' "$code" "$*"
	else
		printf '%s' "$*"
	fi
}

ui_header() {
	printf '\n%s\n' "$(ui_color '1;36' "== $* ==")"
}

ui_success() {
	printf '%s %s\n' "$(ui_color '1;32' '[OK]')" "$*"
}

format_elapsed() {
	local total="${1:-0}"
	printf '%02d:%02d:%02d' "$((total / 3600))" "$(((total % 3600) / 60))" "$((total % 60))"
}

display_path() {
	local path="$1"
	if [[ "$path" == "$SCRIPT_DIR" ]]; then
		printf '.'
	elif [[ "$path" == "$SCRIPT_DIR/"* ]]; then
		printf './%s' "${path#"$SCRIPT_DIR/"}"
	else
		printf '%s' "$path"
	fi
}

status_cell() {
	local status="$1"
	local padded
	printf -v padded '%-7s' "$status"
	case "$status" in
		PASS) ui_color '1;32' "$padded" ;;
		TIMEOUT) ui_color '1;33' "$padded" ;;
		*) ui_color '1;31' "$padded" ;;
	esac
}

print_info_table() {
	local border='+------------+--------------------------------------------------------------------------------------+'
	printf '%s\n' "$(ui_color '2;37' "$border")"
	printf '| %s | %s |\n' "$(ui_color '1;36' "$(printf '%-10s' 'Field')")" "$(ui_color '1;36' "$(printf '%-84s' 'Value')")"
	printf '%s\n' "$(ui_color '2;37' "$border")"
	printf '| %-10s | %-84s |\n' 'Cases' "${#PLAN_ROWS[@]}"
	printf '| %-10s | %-84s |\n' 'Parallel' "$JOBS jobs x $MPI_RANKS MPI ranks x $SST_THREADS SST thread(s)"
	printf '| %-10s | %-84s |\n' 'Mode' "$SIM_MODE"
	printf '| %-10s | %-84s |\n' 'Dry run' "$DRY_RUN"
	printf '| %-10s | %-84s |\n' 'Parameters' "$(display_path "$PARAM_FILE")"
	printf '| %-10s | %-84s |\n' 'Output' "$(display_path "$SWEEP_DIR")"
	printf '%s\n' "$(ui_color '2;37' "$border")"
}

print_plan_table() {
	local border='+-----+----------------------+----------------+----------+-------------+'
	printf '%s\n' "$(ui_color '2;37' "$border")"
	printf '| %s | %s | %s | %s | %s |\n' \
		"$(ui_color '1;36' "$(printf '%-3s' '#')")" \
		"$(ui_color '1;36' "$(printf '%-20s' 'Case')")" \
		"$(ui_color '1;36' "$(printf '%-14s' 'Block MxNxK')")" \
		"$(ui_color '1;36' "$(printf '%-8s' 'CPU slot')")" \
		"$(ui_color '1;36' "$(printf '%-11s' 'State')")"
	printf '%s\n' "$(ui_color '2;37' "$border")"
	local row run_order label safe_label run_id gemm_m gemm_n gemm_k block_m block_n block_k note slot
	for row in "${PLAN_ROWS[@]}"; do
		IFS=$'\t' read -r run_order label safe_label run_id gemm_m gemm_n gemm_k block_m block_n block_k note <<< "$row"
		slot=$(( (run_order - 1) % JOBS ))
		printf '| %3s | %-20s | %-14s | %-8s | %s |\n' \
			"$run_order" "$label" "${block_m}x${block_n}x${block_k}" "$((slot + 1))" \
			"$(ui_color '1;33' "$(printf '%-11s' 'QUEUED')")"
	done
	printf '%s\n' "$(ui_color '2;37' "$border")"
}

usage() {
	cat <<EOF
Usage: $(basename "$0") [tile_shape_sweep.csv]

Edit the CSV and run this script. Enabled rows are scheduled dynamically on
four isolated 4-rank x 1-thread SST slots by default.

Environment overrides:
  GOLEM_SWEEP_JOBS=4
  GOLEM_SWEEP_MPI_RANKS=4
  GOLEM_SWEEP_SST_THREADS=1
  GOLEM_SWEEP_SIM_MODE=full-timing
  GOLEM_SWEEP_CPU_SETS='0,1,2,3;4,5,6,7;8,9,10,11;12,13,14,15'
  GOLEM_SWEEP_TIMEOUT=2h
  GOLEM_SWEEP_DRY_RUN=1
  GOLEM_SWEEP_COLOR=auto|always|never
  GOLEM_SWEEP_RUN_DIR=/path/to/output
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
	usage
	exit 0
fi
if [[ ! -f "$PARAM_FILE" ]]; then
	echo "[ERROR] Missing parameter file: $PARAM_FILE" >&2
	exit 1
fi
for value_name in JOBS MPI_RANKS SST_THREADS; do
	value="${!value_name}"
	if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
		echo "[ERROR] $value_name must be a positive integer, got: $value" >&2
		exit 1
	fi
done
if [[ "$SIM_MODE" != "full-timing" && "$SIM_MODE" != "full-functional" ]]; then
	echo "[ERROR] GOLEM_SWEEP_SIM_MODE must be full-timing or full-functional" >&2
	exit 1
fi

IFS=';' read -r -a CPU_SETS <<< "$CPU_SETS_RAW"
if (( JOBS > ${#CPU_SETS[@]} )); then
	echo "[ERROR] GOLEM_SWEEP_JOBS=$JOBS but only ${#CPU_SETS[@]} CPU sets were provided" >&2
	exit 1
fi
for ((slot = 0; slot < JOBS; slot++)); do
	if [[ -z "${CPU_SETS[$slot]}" ]]; then
		echo "[ERROR] CPU set $slot is empty" >&2
		exit 1
	fi
done

mkdir -p "$SWEEP_DIR/candidates"

python3 - "$PARAM_FILE" "$PLAN_TSV" "$SWEEP_TAG" <<'PY'
import csv
import re
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
tag = sys.argv[3]
required = ["enabled", "label", "gemm_m", "gemm_n", "gemm_k", "block_m", "block_n", "block_k"]

def positive(row, key):
    raw = (row.get(key) or "").strip()
    if not raw.isdigit() or int(raw) <= 0:
        raise SystemExit(f"[ERROR] {key} must be a positive integer for {row.get('label')!r}: {raw!r}")
    return int(raw)

def safe_label(label):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_.-")
    if not value:
        raise SystemExit(f"[ERROR] label has no usable filename characters: {label!r}")
    return value

records = []
seen_labels = set()
with source.open(newline="") as handle:
    reader = csv.DictReader(handle)
    missing = [name for name in required if name not in (reader.fieldnames or [])]
    if missing:
        raise SystemExit(f"[ERROR] missing CSV columns: {', '.join(missing)}")
    for source_row, row in enumerate(reader, 2):
        enabled = (row.get("enabled") or "").strip().lower()
        if enabled in {"", "0", "false", "no", "off"}:
            continue
        if enabled not in {"1", "true", "yes", "on"}:
            raise SystemExit(f"[ERROR] invalid enabled value on CSV row {source_row}: {enabled!r}")
        label = (row.get("label") or "").strip()
        if not label:
            raise SystemExit(f"[ERROR] empty label on CSV row {source_row}")
        safe = safe_label(label)
        if safe in seen_labels:
            raise SystemExit(f"[ERROR] duplicate/sanitized label: {safe}")
        seen_labels.add(safe)
        values = {name: positive(row, name) for name in required[2:]}
        for dim, block in (("gemm_m", "block_m"), ("gemm_n", "block_n"), ("gemm_k", "block_k")):
            if values[dim] % values[block]:
                raise SystemExit(
                    f"[ERROR] {label}: {dim}={values[dim]} is not divisible by {block}={values[block]}"
                )
        order = len(records) + 1
        records.append({
            "run_order": order,
            "label": label.replace("\t", " "),
            "safe_label": safe,
            "run_id": f"tile_{tag}_{order:03d}_{safe}",
            **values,
            "note": (row.get("note") or "").replace("\t", " ").strip(),
        })

if not records:
    raise SystemExit("[ERROR] parameter file contains no enabled candidates")

fields = ["run_order", "label", "safe_label", "run_id", "gemm_m", "gemm_n", "gemm_k",
          "block_m", "block_n", "block_k", "note"]
target.parent.mkdir(parents=True, exist_ok=True)
with target.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(records)
PY

mapfile -t PLAN_ROWS < <(tail -n +2 "$PLAN_TSV")
declare -a SLOT_PIDS SLOT_LABELS SLOT_DIRS

cleanup_sweep_children() {
	local exit_code=$?
	trap - INT TERM EXIT
	for pid in "${SLOT_PIDS[@]:-}"; do
		if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
			kill -TERM "$pid" 2>/dev/null || true
		fi
	done
	for pid in "${SLOT_PIDS[@]:-}"; do
		if [[ -n "$pid" ]]; then
			wait "$pid" 2>/dev/null || true
		fi
	done
	exit "$exit_code"
}
trap 'exit 130' INT
trap 'exit 143' TERM
trap cleanup_sweep_children EXIT

run_candidate() {
	local row="$1"
	local slot="$2"
	local cpu_list="${CPU_SETS[$slot]}"
	local run_order label safe_label run_id gemm_m gemm_n gemm_k block_m block_n block_k note
	IFS=$'\t' read -r run_order label safe_label run_id gemm_m gemm_n gemm_k block_m block_n block_k note <<< "$row"
	local candidate_dir="$SWEEP_DIR/candidates/${run_order}_${safe_label}"
	local console_log="$candidate_dir/console.log"
	local result_file="$candidate_dir/result.tsv"
	local -a command=("$SCRIPT_DIR/run_noc_dma_pipeline.sh")
	if [[ "$DRY_RUN" == "1" ]]; then
		command+=(--dry-run)
	fi
	if [[ -n "$RUN_TIMEOUT" && "$RUN_TIMEOUT" != "0" ]]; then
		command=(timeout "$RUN_TIMEOUT" "${command[@]}")
	fi

	mkdir -p "$candidate_dir/bin" "$candidate_dir/hbm" "$candidate_dir/artifacts" "$candidate_dir/work"
	set +e
	env \
		GOLEM_RUN_ID="$run_id" \
		GOLEM_GEMM_M="$gemm_m" GOLEM_GEMM_N="$gemm_n" GOLEM_GEMM_K="$gemm_k" \
		GOLEM_ORIG_M="$gemm_m" GOLEM_ORIG_N="$gemm_n" GOLEM_ORIG_K="$gemm_k" \
		GOLEM_GEMM_BLOCK_M="$block_m" GOLEM_GEMM_BLOCK_N="$block_n" GOLEM_GEMM_BLOCK_K="$block_k" \
		GOLEM_SIM_MODE="$SIM_MODE" \
		GOLEM_MPI_RANKS="$MPI_RANKS" GOLEM_SST_THREADS="$SST_THREADS" \
		GOLEM_MPI_ARGS="--bind-to cpu-list:ordered --cpu-list $cpu_list" \
		GOLEM_CPU_BINDING_MODE="tile_sweep_slot_${slot}" \
		GOLEM_TEST_BINARY="$candidate_dir/bin/test_noc_dma" \
		GOLEM_BUILD_METADATA_FILE="$candidate_dir/bin/test_noc_dma.build.env" \
		GOLEM_HBM_DIR="$candidate_dir/hbm" \
		GOLEM_ARTIFACT_ROOT="$candidate_dir/artifacts" \
		GOLEM_SST_WORK_DIR="$candidate_dir/work" \
		GOLEM_RUN_SUMMARY_CSV="$candidate_dir/run_summary.csv" \
		GOLEM_PRESET_LOG="sst.log" \
		GOLEM_TENSOR_SOURCE=synthetic GOLEM_VERIFY_C=0 GOLEM_DUMP_C_FILE= \
		GOLEM_HBM_DUMP_OUTPUT=0 GOLEM_MEM_NODE_SIZE_BYTES=auto \
		GOLEM_BENCH_QUIET_LOGS=1 \
		"${command[@]}" >"$console_log" 2>&1 &
	local pipeline_pid=$!
	trap 'kill -TERM "$pipeline_pid" 2>/dev/null || true; wait "$pipeline_pid" 2>/dev/null || true; exit 130' INT
	trap 'kill -TERM "$pipeline_pid" 2>/dev/null || true; wait "$pipeline_pid" 2>/dev/null || true; exit 143' TERM
	wait "$pipeline_pid"
	local rc=$?
	trap - INT TERM
	set -e
	local status=FAIL
	if (( rc == 0 )); then
		status=PASS
	elif (( rc == 124 )); then
		status=TIMEOUT
	fi
	printf '%s\t%s\t%s\t%s\n' "$status" "$rc" "$cpu_list" "$console_log" > "$result_file"
	return "$rc"
}

launch_slot() {
	local slot="$1"
	local row="$2"
	local run_order label safe_label rest
	IFS=$'\t' read -r run_order label safe_label rest <<< "$row"
	local candidate_dir="$SWEEP_DIR/candidates/${run_order}_${safe_label}"
	run_candidate "$row" "$slot" &
	SLOT_PIDS[$slot]=$!
	SLOT_LABELS[$slot]="$label"
	SLOT_DIRS[$slot]="$candidate_dir"
	printf '%s slot=%d cpus=%-11s case=%s\n' \
		"$(ui_color '1;36' '[START]')" "$((slot + 1))" "${CPU_SETS[$slot]}" "$(ui_color '1;35' "$label")"
}

ui_header "TileMC TILE SHAPE SWEEP"
print_info_table
ui_header "CASE PLAN"
print_plan_table
printf '\n%s\n' "$(ui_color '1;36' '== RUNNING ==')"

next_row=0
running=0
completed=0
while (( next_row < ${#PLAN_ROWS[@]} || running > 0 )); do
	for ((slot = 0; slot < JOBS && next_row < ${#PLAN_ROWS[@]}; slot++)); do
		if [[ -z "${SLOT_PIDS[$slot]:-}" ]]; then
			launch_slot "$slot" "${PLAN_ROWS[$next_row]}"
			next_row=$((next_row + 1))
			running=$((running + 1))
		fi
	done
	if (( running == 0 )); then
		continue
	fi
	finished_pid=""
	set +e
	wait -n -p finished_pid
	set -e
	for ((slot = 0; slot < JOBS; slot++)); do
		if [[ -n "${SLOT_PIDS[$slot]:-}" && "${SLOT_PIDS[$slot]}" == "$finished_pid" ]]; then
			result_file="${SLOT_DIRS[$slot]}/result.tsv"
			status=FAIL
			rc=unknown
			if [[ -f "$result_file" ]]; then
				IFS=$'\t' read -r status rc _ < "$result_file"
			fi
			completed=$((completed + 1))
			printf '%s %d/%d case=%-20s exit=%s\n' \
				"$(status_cell "$status")" "$completed" "${#PLAN_ROWS[@]}" "${SLOT_LABELS[$slot]}" "$rc"
			unset 'SLOT_PIDS[slot]' 'SLOT_LABELS[slot]' 'SLOT_DIRS[slot]'
			running=$((running - 1))
			break
		fi
	done
done

python3 - "$PLAN_TSV" "$SWEEP_DIR/candidates" "$RESULT_CSV" "$PARETO_CSV" <<'PY'
import csv
import math
import sys
from pathlib import Path

plan_file, candidates_dir, result_file, pareto_file = map(Path, sys.argv[1:])
plan = list(csv.DictReader(plan_file.open(newline=""), delimiter="\t"))
records = []
summary_fields = []
for item in plan:
    candidate_dir = candidates_dir / f"{item['run_order']}_{item['safe_label']}"
    status_path = candidate_dir / "result.tsv"
    status, exit_code, cpu_list, console_log = "FAIL", "missing", "", str(candidate_dir / "console.log")
    if status_path.exists():
        status, exit_code, cpu_list, console_log = status_path.read_text().rstrip("\n").split("\t", 3)
    summary_path = candidate_dir / "run_summary.csv"
    summary = {}
    if summary_path.exists():
        rows = list(csv.DictReader(summary_path.open(newline="")))
        if rows:
            summary = rows[-1]
            for field in summary:
                if field not in summary_fields:
                    summary_fields.append(field)
    record = dict(item)
    record.update({
        "status": status,
        "exit_code": exit_code,
        "cpu_list": cpu_list,
        "candidate_dir": str(candidate_dir),
        "console_log": console_log,
    })
    record.update(summary)
    records.append(record)

base_fields = list(plan[0]) + ["status", "exit_code", "cpu_list", "candidate_dir", "console_log"]
fields = base_fields + [field for field in summary_fields if field not in base_fields]
with result_file.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)

def number(row, name):
    try:
        return float(row.get(name, ""))
    except (TypeError, ValueError):
        return math.nan

metric_latency = "gemm_system_latency_cycles"
metric_util = "exec_system_array_utilization_pct"
valid = [row for row in records if row["status"] == "PASS" and not math.isnan(number(row, metric_latency))]
front = []
for candidate in valid:
    workload = tuple(candidate[name] for name in ("gemm_m", "gemm_n", "gemm_k"))
    latency = number(candidate, metric_latency)
    util = number(candidate, metric_util)
    dominated = False
    for other in valid:
        if other is candidate or tuple(other[name] for name in ("gemm_m", "gemm_n", "gemm_k")) != workload:
            continue
        other_latency = number(other, metric_latency)
        other_util = number(other, metric_util)
        if math.isnan(util) or math.isnan(other_util):
            if other_latency < latency:
                dominated = True
                break
        elif other_latency <= latency and other_util >= util and (other_latency < latency or other_util > util):
            dominated = True
            break
    if not dominated:
        front.append(candidate)
front.sort(key=lambda row: (number(row, metric_latency), -number(row, metric_util)))
with pareto_file.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(front)

PY

pass_count="$(awk -F '\t' '$1 == "PASS" {count++} END {print count + 0}' "$SWEEP_DIR"/candidates/*/result.tsv 2>/dev/null || true)"
fail_count=$(( ${#PLAN_ROWS[@]} - pass_count ))
mapfile -t DISPLAY_ROWS < <(python3 - "$RESULT_CSV" <<'PY'
import csv
import math
import sys

def integer(value):
    try:
        return str(int(round(float(value))))
    except (TypeError, ValueError):
        return "-"

def decimal(value, suffix=""):
    try:
        number = float(value)
        if math.isnan(number):
            return "-"
        return f"{number:.2f}{suffix}"
    except (TypeError, ValueError):
        return "-"

def elapsed(value):
    try:
        seconds = int(round(float(value)))
    except (TypeError, ValueError):
        return "-"
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"

with open(sys.argv[1], newline="") as handle:
    for row in csv.DictReader(handle):
        try:
            array_in = int(row.get("array_input_size") or 0)
            block_k = int(row.get("block_k") or 0)
            k_steps = (
                str(max(1, (block_k + array_in - 1) // array_in))
                if array_in and (block_k <= array_in or block_k % array_in == 0)
                else "-"
            )
        except ValueError:
            k_steps = "-"
        values = [
            row.get("run_order", "-"), row.get("label", "-"),
            f"{row.get('block_m', '-')}x{row.get('block_n', '-')}x{row.get('block_k', '-')}",
            k_steps, row.get("status", "FAIL"), elapsed(row.get("wall_time_sec")),
            row.get("simulated_time") or "-", integer(row.get("gemm_system_latency_cycles")),
            decimal(row.get("exec_system_array_utilization_pct"), "%"),
        ]
        print("\t".join(values))
PY
)

ui_header "SWEEP RESULT"
result_border='+-----+----------------------+----------------+--------+---------+----------+---------------+--------------+----------+'
printf '%s\n' "$(ui_color '2;37' "$result_border")"
printf '| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' \
	"$(ui_color '1;36' "$(printf '%-3s' '#')")" \
	"$(ui_color '1;36' "$(printf '%-20s' 'Case')")" \
	"$(ui_color '1;36' "$(printf '%-14s' 'Block MxNxK')")" \
	"$(ui_color '1;36' "$(printf '%-6s' 'K-step')")" \
	"$(ui_color '1;36' "$(printf '%-7s' 'Status')")" \
	"$(ui_color '1;36' "$(printf '%-8s' 'Wall')")" \
	"$(ui_color '1;36' "$(printf '%-13s' 'Simulated')")" \
	"$(ui_color '1;36' "$(printf '%-12s' 'Latency')")" \
	"$(ui_color '1;36' "$(printf '%-8s' 'Util')")"
printf '%s\n' "$(ui_color '2;37' "$result_border")"
for row in "${DISPLAY_ROWS[@]}"; do
	IFS=$'\t' read -r order label block k_steps status wall simulated latency util <<< "$row"
	printf '| %3s | %-20s | %-14s | %s | %s | %8s | %13s | %12s | %s |\n' \
		"$order" "$label" "$block" "$(ui_color '1;33' "$(printf '%6s' "$k_steps")")" "$(status_cell "$status")" \
		"$wall" "$simulated" "$latency" "$(ui_color '1;32' "$(printf '%8s' "$util")")"
done
printf '%s\n' "$(ui_color '2;37' "$result_border")"

summary_border='+------------+--------------------------------------------------------------------------------------+'
printf '%s\n' "$(ui_color '2;37' "$summary_border")"
printf '| %-10s | %s |\n' 'Passed' "$(ui_color '1;32' "$(printf '%-84s' "$pass_count/${#PLAN_ROWS[@]}")")"
printf '| %-10s | %s |\n' 'Sweep time' "$(ui_color '1;33' "$(printf '%-84s' "$(format_elapsed "$(( $(date +%s) - SWEEP_START_EPOCH ))")")")"
printf '| %-10s | %-84s |\n' 'Results' "$(display_path "$RESULT_CSV")"
printf '| %-10s | %-84s |\n' 'Pareto' "$(display_path "$PARETO_CSV")"
printf '| %-10s | %-84s |\n' 'Details' "$(display_path "$SWEEP_DIR/candidates/<candidate>/console.log")"
printf '%s\n' "$(ui_color '2;37' "$summary_border")"
if (( fail_count > 0 )); then
	exit 1
fi
