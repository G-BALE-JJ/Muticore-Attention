#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="${GOLEM_ARTIFACT_ROOT:-$SCRIPT_DIR/artifacts}"
SHAPES_FILE="${GOLEM_LLM_SHAPES_FILE:-}"
SWEEP_ROOT="${GOLEM_SWEEP_ROOT:-$ARTIFACT_ROOT/stats/sweeps/llm_operator_shapes}"
SWEEP_TAG="${GOLEM_SWEEP_TAG:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_RUN_DIR="${GOLEM_SWEEP_RUN_DIR:-$SWEEP_ROOT/run_$SWEEP_TAG}"
ALIGN="${GOLEM_LLM_SWEEP_ALIGN:-64}"
RUN_TIMEOUT="${GOLEM_LLM_SWEEP_TIMEOUT:-}"
LIMIT=""
EXECUTE=1
RESUME=1
DISABLE_SST_STATS="${GOLEM_BENCH_DISABLE_SST_STATS:-0}"
PROGRESS_HEARTBEAT="${GOLEM_LLM_SWEEP_PROGRESS_HEARTBEAT:-${GOLEM_PROGRESS_HEARTBEAT:-1}}"
PROGRESS_INTERVAL_CYCLES="${GOLEM_LLM_SWEEP_PROGRESS_INTERVAL_CYCLES:-${GOLEM_PROGRESS_INTERVAL_CYCLES:-1000000}}"
OUTPUT_MODE="${GOLEM_LLM_OUTPUT_MODE:-fusion}"

usage() {
	cat <<'EOF'
Usage: run_llm_operator_sweep.sh [options]

Sweep unique LLM GEMM operator shapes from gpu_baseline/llm_gemm_operator_shapes.txt
on the GOLEM SST architecture. The input file uses [M,K,N,B]; this script runs
unique padded architecture shapes as --gemm-m M --gemm-n N --gemm-k K, then
weights the summaries by B.

Options:
  --shapes-file PATH       Shape file (or set GOLEM_LLM_SHAPES_FILE)
  --sweep-root DIR         Sweep root directory
  --sweep-run-dir DIR      Exact output directory for this run
  --tag TAG                Sweep tag, default timestamp
  --align N                Pad M/N/K up to this multiple, default 64
  --timeout SECONDS        Per-shape timeout, passed to timeout(1)
  --limit N                Run only first N unique shapes, useful for smoke tests
  --dry-run                Generate plan and print commands, but do not run SST
  --no-execute             Alias for --dry-run
  --no-resume              Do not skip shapes already marked PASS in shape_status.tsv
  --disable-sst-stats      Disable SST all-stats for faster latency-only runs
  --enable-sst-stats       Keep SST stats enabled, default
  --progress-heartbeat N   Enable lightweight SST progress heartbeats, default 1
  --progress-interval-cycles N  Heartbeat interval in simulated cycles, default 1000000
  --output-mode MODE        Final C destination: fusion|hbm, default fusion
  -h, --help               Show this help

Output files:
  unique_shapes.tsv        Unique padded architecture runs
  operator_map.csv         Original model operators mapped to unique runs
  shape_status.tsv         Per-shape PASS/FAIL/TIMEOUT status
  operator_summary.csv     Per-operator weighted cycles/TOPS
  model_summary.csv        Per-model weighted cycles/TOPS
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--shapes-file)
			SHAPES_FILE="$2"; shift 2 ;;
		--sweep-root)
			SWEEP_ROOT="$2"; shift 2 ;;
		--sweep-run-dir)
			SWEEP_RUN_DIR="$2"; shift 2 ;;
		--tag)
			SWEEP_TAG="$2"; shift 2 ;;
		--align)
			ALIGN="$2"; shift 2 ;;
		--timeout)
			RUN_TIMEOUT="$2"; shift 2 ;;
		--limit)
			LIMIT="$2"; shift 2 ;;
		--dry-run|--no-execute)
			EXECUTE=0; shift ;;
		--no-resume)
			RESUME=0; shift ;;
		--disable-sst-stats)
			DISABLE_SST_STATS=1; shift ;;
		--enable-sst-stats)
			DISABLE_SST_STATS=0; shift ;;
		--progress-heartbeat)
			PROGRESS_HEARTBEAT="$2"; shift 2 ;;
		--progress-interval-cycles)
			PROGRESS_INTERVAL_CYCLES="$2"; shift 2 ;;
		--output-mode)
			OUTPUT_MODE="$2"; shift 2 ;;
		-h|--help)
			usage; exit 0 ;;
		*)
			echo "[ERROR] Unknown option: $1" >&2
			usage >&2
			exit 1 ;;
	esac
done

if [[ -z "$SHAPES_FILE" || ! -f "$SHAPES_FILE" ]]; then
	echo "[ERROR] Missing shapes file: $SHAPES_FILE" >&2
	exit 1
fi
if ! [[ "$ALIGN" =~ ^[0-9]+$ ]] || [[ "$ALIGN" -le 0 ]]; then
	echo "[ERROR] --align must be a positive integer, got: $ALIGN" >&2
	exit 1
fi
if [[ -n "$LIMIT" ]] && { ! [[ "$LIMIT" =~ ^[0-9]+$ ]] || [[ "$LIMIT" -le 0 ]]; }; then
	echo "[ERROR] --limit must be a positive integer, got: $LIMIT" >&2
	exit 1
fi
if [[ "$PROGRESS_HEARTBEAT" != "0" && "$PROGRESS_HEARTBEAT" != "1" ]]; then
	echo "[ERROR] --progress-heartbeat must be 0 or 1, got: $PROGRESS_HEARTBEAT" >&2
	exit 1
fi
if ! [[ "$PROGRESS_INTERVAL_CYCLES" =~ ^[0-9]+$ ]] || [[ "$PROGRESS_INTERVAL_CYCLES" -le 0 ]]; then
	echo "[ERROR] --progress-interval-cycles must be a positive integer, got: $PROGRESS_INTERVAL_CYCLES" >&2
	exit 1
fi
if [[ "$OUTPUT_MODE" != "fusion" && "$OUTPUT_MODE" != "hbm" ]]; then
	echo "[ERROR] --output-mode must be fusion or hbm, got: $OUTPUT_MODE" >&2
	exit 1
fi

mkdir -p "$SWEEP_RUN_DIR" "$SWEEP_RUN_DIR/driver_logs"

UNIQUE_SHAPES_TSV="$SWEEP_RUN_DIR/unique_shapes.tsv"
OPERATOR_MAP_CSV="$SWEEP_RUN_DIR/operator_map.csv"
STATUS_TSV="$SWEEP_RUN_DIR/shape_status.tsv"
RUN_SUMMARY_CSV="$SWEEP_RUN_DIR/run_summary.csv"
OPERATOR_SUMMARY_CSV="$SWEEP_RUN_DIR/operator_summary.csv"
MODEL_SUMMARY_CSV="$SWEEP_RUN_DIR/model_summary.csv"
METADATA_TXT="$SWEEP_RUN_DIR/metadata.txt"

python3 - "$SHAPES_FILE" "$ALIGN" "$UNIQUE_SHAPES_TSV" "$OPERATOR_MAP_CSV" <<'PY'
import csv
import re
import sys
from collections import OrderedDict
from pathlib import Path

shape_path = Path(sys.argv[1])
align = int(sys.argv[2])
unique_out = Path(sys.argv[3])
op_out = Path(sys.argv[4])

def pad_up(x, a):
    return ((x + a - 1) // a) * a

def slug(text):
    text = text.strip().lower()
    text = text.replace("+", "plus")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "shape"

current_model = None
ops = []
line_re = re.compile(r"^([^#\[]\S*)\s+\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]")

for raw in shape_path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
    line = raw.strip()
    if not line or line.startswith("Notation:") or line.startswith("Assumption:") or line.startswith("GEMM operator"):
        continue
    if line.endswith(":"):
        current_model = line[:-1].strip()
        continue
    m = line_re.match(line)
    if not m:
        continue
    if current_model is None:
        raise SystemExit(f"[ERROR] shape row before model header: {line}")
    op_name, m_raw, k_raw, n_raw, b_raw = m.groups()
    orig_m, orig_k, orig_n, repeat_b = map(int, (m_raw, k_raw, n_raw, b_raw))
    run_m = pad_up(orig_m, align)
    run_n = pad_up(orig_n, align)
    run_k = pad_up(orig_k, align)
    ops.append({
        "model": current_model,
        "model_slug": slug(current_model),
        "operator": op_name,
        "operator_slug": slug(op_name),
        "orig_m": orig_m,
        "orig_k": orig_k,
        "orig_n": orig_n,
        "repeat_b": repeat_b,
        "run_m": run_m,
        "run_n": run_n,
        "run_k": run_k,
    })

if not ops:
    raise SystemExit(f"[ERROR] no operator shapes parsed from {shape_path}")

unique = OrderedDict()
for op in ops:
    key = (op["run_m"], op["run_n"], op["run_k"])
    if key not in unique:
        idx = len(unique) + 1
        unique[key] = {
            "shape_id": f"s{idx:02d}_{op['model_slug']}_{op['operator_slug']}_m{op['run_m']}_n{op['run_n']}_k{op['run_k']}",
            "run_m": op["run_m"],
            "run_n": op["run_n"],
            "run_k": op["run_k"],
            "orig_m": op["orig_m"],
            "orig_n": op["orig_n"],
            "orig_k": op["orig_k"],
            "operators": [],
            "total_b": 0,
        }
    rec = unique[key]
    rec["operators"].append(f"{op['model_slug']}/{op['operator_slug']}x{op['repeat_b']}")
    rec["total_b"] += op["repeat_b"]
    op["shape_id"] = rec["shape_id"]

unique_out.parent.mkdir(parents=True, exist_ok=True)
with unique_out.open("w", newline="") as f:
    f.write("shape_id\trun_m\trun_n\trun_k\torig_m\torig_n\torig_k\ttotal_b\toperators\n")
    for rec in unique.values():
        f.write(
            "\t".join(
                str(x)
                for x in (
                    rec["shape_id"], rec["run_m"], rec["run_n"], rec["run_k"],
                    rec["orig_m"], rec["orig_n"], rec["orig_k"], rec["total_b"],
                    ";".join(rec["operators"]),
                )
            )
            + "\n"
        )

fieldnames = [
    "model", "operator", "model_slug", "operator_slug", "shape_id",
    "orig_m", "orig_k", "orig_n", "repeat_b", "run_m", "run_n", "run_k",
    "pad_m", "pad_n", "pad_k", "original_flops", "padded_flops",
]
with op_out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for op in ops:
        original_flops = 2 * op["orig_m"] * op["orig_n"] * op["orig_k"]
        padded_flops = 2 * op["run_m"] * op["run_n"] * op["run_k"]
        writer.writerow({
            **op,
            "pad_m": op["run_m"] - op["orig_m"],
            "pad_n": op["run_n"] - op["orig_n"],
            "pad_k": op["run_k"] - op["orig_k"],
            "original_flops": original_flops,
            "padded_flops": padded_flops,
        })

print(f"[OK] parsed {len(ops)} operators into {len(unique)} unique padded shapes")
PY

{
	echo "date=$(date -Is)"
	echo "script=$0"
	echo "script_dir=$SCRIPT_DIR"
	echo "shapes_file=$SHAPES_FILE"
	echo "sweep_tag=$SWEEP_TAG"
	echo "sweep_run_dir=$SWEEP_RUN_DIR"
	echo "align=$ALIGN"
	echo "timeout=${RUN_TIMEOUT:-<none>}"
	echo "execute=$EXECUTE"
	echo "resume=$RESUME"
	echo "disable_sst_stats=$DISABLE_SST_STATS"
	echo "progress_heartbeat=$PROGRESS_HEARTBEAT"
	echo "progress_interval_cycles=$PROGRESS_INTERVAL_CYCLES"
	echo "tensor_source=synthetic"
	echo "quiet_logs=1"
	echo "hbm_dump_output=0"
	echo "verify_c=0"
	echo "output_mode=$OUTPUT_MODE"
	echo "clock_ref=${GOLEM_STAGE_CLOCK_GHZ:-${VANADIS_CPU_CLOCK:-1.0GHz}}"
} > "$METADATA_TXT"

if [[ ! -f "$STATUS_TSV" || "$RESUME" -eq 0 ]]; then
	printf 'run_order\tshape_id\trun_id\tstatus\texit_code\tdriver_log\n' > "$STATUS_TSV"
fi

shape_has_passed() {
	local shape_id="$1"
	awk -F '\t' -v id="$shape_id" '$2 == id && $4 == "PASS" { found=1 } END { exit(found ? 0 : 1) }' "$STATUS_TSV"
}

run_order=0
while IFS=$'\t' read -r shape_id run_m run_n run_k orig_m orig_n orig_k total_b operators; do
	if [[ "$shape_id" == "shape_id" ]]; then
		continue
	fi
	run_order=$((run_order + 1))
	if [[ -n "$LIMIT" && "$run_order" -gt "$LIMIT" ]]; then
		break
	fi

	run_id="${shape_id}_${SWEEP_TAG}"
	driver_log="$SWEEP_RUN_DIR/driver_logs/${shape_id}.driver.log"
	sst_log="${shape_id}.sst.log"

	if [[ "$RESUME" -eq 1 ]] && shape_has_passed "$shape_id"; then
		echo "[SKIP] $shape_id already PASS"
		continue
	fi

	echo "[SWEEP] #$run_order $shape_id run=($run_m,$run_n,$run_k) orig=($orig_m,$orig_n,$orig_k) total_B=$total_b"
	echo "        operators=$operators"

	cmd=(
		"$SCRIPT_DIR/run_noc_dma_pipeline.sh"
		--gemm-m "$run_m"
		--gemm-n "$run_n"
		--gemm-k "$run_k"
		--orig-m "$orig_m"
		--orig-n "$orig_n"
		--orig-k "$orig_k"
		--mem-node-size auto
		--output-mode "$OUTPUT_MODE"
		--no-hbm-dump-output
		--log "$sst_log"
	)

	if [[ "$EXECUTE" -eq 0 ]]; then
		printf '[DRY-RUN] env GOLEM_RUN_ID=%q GOLEM_RUN_SUMMARY_CSV=%q GOLEM_ARTIFACT_ROOT=%q GOLEM_BENCH_QUIET_LOGS=1 GOLEM_BENCH_DISABLE_SST_STATS=%q GOLEM_PROGRESS_HEARTBEAT=%q GOLEM_PROGRESS_INTERVAL_CYCLES=%q GOLEM_TENSOR_SOURCE=synthetic GOLEM_VERIFY_C=0 GOLEM_HBM_DUMP_OUTPUT=0 GOLEM_MVM_DUMP_ENABLE=0 ' \
			"$run_id" "$RUN_SUMMARY_CSV" "$SWEEP_RUN_DIR/artifacts" "$DISABLE_SST_STATS" "$PROGRESS_HEARTBEAT" "$PROGRESS_INTERVAL_CYCLES"
		printf '%q ' "${cmd[@]}"
		printf '> %q 2>&1\n' "$driver_log"
		continue
	fi

	set +e
	if [[ -n "$RUN_TIMEOUT" && "$RUN_TIMEOUT" != "0" ]]; then
		env \
			GOLEM_RUN_ID="$run_id" \
			GOLEM_RUN_SUMMARY_CSV="$RUN_SUMMARY_CSV" \
			GOLEM_ARTIFACT_ROOT="$SWEEP_RUN_DIR/artifacts" \
			GOLEM_TENSOR_SOURCE=synthetic \
			GOLEM_VERIFY_C=0 \
			GOLEM_DUMP_C_FILE= \
			GOLEM_HBM_DUMP_OUTPUT=0 \
			GOLEM_MEM_NODE_SIZE_BYTES=auto \
			GOLEM_BENCH_QUIET_LOGS=1 \
			GOLEM_BENCH_DISABLE_SST_STATS="$DISABLE_SST_STATS" \
			GOLEM_PROGRESS_HEARTBEAT="$PROGRESS_HEARTBEAT" \
			GOLEM_PROGRESS_INTERVAL_CYCLES="$PROGRESS_INTERVAL_CYCLES" \
			GOLEM_MVM_DUMP_ENABLE=0 \
			GOLEM_EXPORT_NOC_HEATMAPS=0 \
			GOLEM_SKIP_TENSOR_GEN=0 \
			GOLEM_SKIP_HBM_GEN=0 \
			GOLEM_SKIP_BUILD=0 \
			timeout "$RUN_TIMEOUT" "${cmd[@]}" > "$driver_log" 2>&1
	else
		env \
			GOLEM_RUN_ID="$run_id" \
			GOLEM_RUN_SUMMARY_CSV="$RUN_SUMMARY_CSV" \
			GOLEM_ARTIFACT_ROOT="$SWEEP_RUN_DIR/artifacts" \
			GOLEM_TENSOR_SOURCE=synthetic \
			GOLEM_VERIFY_C=0 \
			GOLEM_DUMP_C_FILE= \
			GOLEM_HBM_DUMP_OUTPUT=0 \
			GOLEM_MEM_NODE_SIZE_BYTES=auto \
			GOLEM_BENCH_QUIET_LOGS=1 \
			GOLEM_BENCH_DISABLE_SST_STATS="$DISABLE_SST_STATS" \
			GOLEM_PROGRESS_HEARTBEAT="$PROGRESS_HEARTBEAT" \
			GOLEM_PROGRESS_INTERVAL_CYCLES="$PROGRESS_INTERVAL_CYCLES" \
			GOLEM_MVM_DUMP_ENABLE=0 \
			GOLEM_EXPORT_NOC_HEATMAPS=0 \
			GOLEM_SKIP_TENSOR_GEN=0 \
			GOLEM_SKIP_HBM_GEN=0 \
			GOLEM_SKIP_BUILD=0 \
			"${cmd[@]}" > "$driver_log" 2>&1
	fi
	rc=$?
	set -e

	if [[ "$rc" -eq 0 ]]; then
		status="PASS"
	elif [[ "$rc" -eq 124 ]]; then
		status="TIMEOUT"
	else
		status="FAIL"
	fi
	printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$run_order" "$shape_id" "$run_id" "$status" "$rc" "$driver_log" >> "$STATUS_TSV"
	if [[ "$status" != "PASS" ]]; then
		echo "[WARN] $shape_id ended with status=$status exit_code=$rc; see $driver_log"
	else
		echo "[OK] $shape_id PASS"
	fi
done < "$UNIQUE_SHAPES_TSV"

if [[ "$EXECUTE" -eq 0 ]]; then
	echo "[OK] dry-run complete"
	echo "[OK] sweep directory: $SWEEP_RUN_DIR"
	echo "[OK] unique shapes: $UNIQUE_SHAPES_TSV"
	echo "[OK] operator map: $OPERATOR_MAP_CSV"
	exit 0
fi

python3 - "$OPERATOR_MAP_CSV" "$UNIQUE_SHAPES_TSV" "$STATUS_TSV" "$RUN_SUMMARY_CSV" "$OPERATOR_SUMMARY_CSV" "$MODEL_SUMMARY_CSV" <<'PY'
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

op_map_path, unique_path, status_path, run_summary_path, op_out_path, model_out_path = map(Path, sys.argv[1:])

def to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def parse_clock_ghz(raw):
    raw = (raw or "1.0GHz").strip()
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)([GMK]?Hz)?$", raw, re.I)
    if not m:
        return 1.0
    val = float(m.group(1))
    unit = (m.group(2) or "GHz").lower()
    if unit == "ghz":
        return val
    if unit == "mhz":
        return val / 1000.0
    if unit == "khz":
        return val / 1_000_000.0
    if unit == "hz":
        return val / 1_000_000_000.0
    return val

clock_ghz = parse_clock_ghz(__import__("os").environ.get("GOLEM_STAGE_CLOCK_GHZ") or __import__("os").environ.get("VANADIS_CPU_CLOCK") or "1.0GHz")

latest_status = {}
if status_path.exists():
    with status_path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            latest_status[row["shape_id"]] = row

run_rows_by_id = {}
if run_summary_path.exists():
    with run_summary_path.open(newline="") as f:
        for row in csv.DictReader(f):
            run_rows_by_id[row.get("run_id", "")] = row

operators = list(csv.DictReader(op_map_path.open(newline="")))

op_fields = [
    "model", "operator", "shape_id", "status", "exit_code", "repeat_b",
    "orig_m", "orig_k", "orig_n", "run_m", "run_n", "run_k",
    "original_flops_weighted", "padded_flops_weighted",
    "gemm_system_latency_cycles", "weighted_cycles", "latency_seconds",
    "effective_original_tops", "padded_tops", "padding_overhead_pct",
    "run_id", "driver_log", "sst_log_file",
]
op_out_path.parent.mkdir(parents=True, exist_ok=True)
model_acc = defaultdict(lambda: {
    "operators": 0,
    "passed_operators": 0,
    "original_flops": 0.0,
    "padded_flops": 0.0,
    "weighted_cycles": 0.0,
})

with op_out_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=op_fields)
    writer.writeheader()
    for op in operators:
        shape_id = op["shape_id"]
        status = latest_status.get(shape_id, {})
        run_id = status.get("run_id", "")
        run_row = run_rows_by_id.get(run_id, {})
        b = int(op["repeat_b"])
        original_flops = int(op["original_flops"]) * b
        padded_flops = int(op["padded_flops"]) * b
        cycles = to_float(run_row.get("gemm_system_latency_cycles", ""), 0.0)
        weighted_cycles = cycles * b if cycles > 0 else 0.0
        latency_seconds = weighted_cycles / (clock_ghz * 1e9) if weighted_cycles > 0 and clock_ghz > 0 else 0.0
        effective_original_tops = original_flops / latency_seconds / 1e12 if latency_seconds > 0 else 0.0
        padded_tops = padded_flops / latency_seconds / 1e12 if latency_seconds > 0 else 0.0
        padding_overhead_pct = 100.0 * (padded_flops - original_flops) / original_flops if original_flops > 0 else 0.0

        model = op["model"]
        model_acc[model]["operators"] += 1
        model_acc[model]["original_flops"] += original_flops
        model_acc[model]["padded_flops"] += padded_flops
        if status.get("status") == "PASS" and weighted_cycles > 0:
            model_acc[model]["passed_operators"] += 1
            model_acc[model]["weighted_cycles"] += weighted_cycles

        writer.writerow({
            "model": model,
            "operator": op["operator"],
            "shape_id": shape_id,
            "status": status.get("status", "MISSING"),
            "exit_code": status.get("exit_code", ""),
            "repeat_b": b,
            "orig_m": op["orig_m"],
            "orig_k": op["orig_k"],
            "orig_n": op["orig_n"],
            "run_m": op["run_m"],
            "run_n": op["run_n"],
            "run_k": op["run_k"],
            "original_flops_weighted": int(original_flops),
            "padded_flops_weighted": int(padded_flops),
            "gemm_system_latency_cycles": f"{cycles:.0f}" if cycles else "",
            "weighted_cycles": f"{weighted_cycles:.0f}" if weighted_cycles else "",
            "latency_seconds": f"{latency_seconds:.9f}" if latency_seconds else "",
            "effective_original_tops": f"{effective_original_tops:.6f}" if effective_original_tops else "",
            "padded_tops": f"{padded_tops:.6f}" if padded_tops else "",
            "padding_overhead_pct": f"{padding_overhead_pct:.6f}",
            "run_id": run_id,
            "driver_log": status.get("driver_log", ""),
            "sst_log_file": run_row.get("log_file", ""),
        })

model_fields = [
    "model", "status", "operators", "passed_operators", "original_flops", "padded_flops",
    "weighted_cycles", "latency_seconds", "effective_original_tops", "padded_tops",
    "padding_overhead_pct", "clock_ghz",
]
with model_out_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=model_fields)
    writer.writeheader()
    for model, acc in model_acc.items():
        status = "PASS" if acc["operators"] == acc["passed_operators"] else "INCOMPLETE"
        latency_seconds = acc["weighted_cycles"] / (clock_ghz * 1e9) if acc["weighted_cycles"] > 0 and clock_ghz > 0 else 0.0
        effective_original_tops = acc["original_flops"] / latency_seconds / 1e12 if latency_seconds > 0 else 0.0
        padded_tops = acc["padded_flops"] / latency_seconds / 1e12 if latency_seconds > 0 else 0.0
        padding_overhead_pct = 100.0 * (acc["padded_flops"] - acc["original_flops"]) / acc["original_flops"] if acc["original_flops"] > 0 else 0.0
        writer.writerow({
            "model": model,
            "status": status,
            "operators": acc["operators"],
            "passed_operators": acc["passed_operators"],
            "original_flops": int(acc["original_flops"]),
            "padded_flops": int(acc["padded_flops"]),
            "weighted_cycles": f"{acc['weighted_cycles']:.0f}" if acc["weighted_cycles"] else "",
            "latency_seconds": f"{latency_seconds:.9f}" if latency_seconds else "",
            "effective_original_tops": f"{effective_original_tops:.6f}" if effective_original_tops else "",
            "padded_tops": f"{padded_tops:.6f}" if padded_tops else "",
            "padding_overhead_pct": f"{padding_overhead_pct:.6f}",
            "clock_ghz": f"{clock_ghz:.6f}",
        })

print(f"[OK] wrote operator summary: {op_out_path}")
print(f"[OK] wrote model summary: {model_out_path}")
PY

echo "[OK] sweep directory: $SWEEP_RUN_DIR"
echo "[OK] unique shapes: $UNIQUE_SHAPES_TSV"
echo "[OK] operator map: $OPERATOR_MAP_CSV"
echo "[OK] status: $STATUS_TSV"
echo "[OK] run summary: $RUN_SUMMARY_CSV"
echo "[OK] operator summary: $OPERATOR_SUMMARY_CSV"
echo "[OK] model summary: $MODEL_SUMMARY_CSV"
