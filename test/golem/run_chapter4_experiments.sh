#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="$SCRIPT_DIR/run_noc_dma_pipeline.sh"
MANIFEST="${1:-$SCRIPT_DIR/chapter4_experiments.tsv}"
STAMP="${GOLEM_CH4_TAG:-$(date +%Y%m%d_%H%M%S)_$$}"
ROOT="${GOLEM_CH4_ROOT:-$SCRIPT_DIR/artifacts/chapter4_20260901/$STAMP}"
M=${GOLEM_CH4_M:-1024}; N=${GOLEM_CH4_N:-1024}; K=${GOLEM_CH4_K:-1024}
BK=${GOLEM_CH4_BLOCK_K:-64}; MPI_RANKS=${GOLEM_CH4_MPI_RANKS:-4}
SIM_MODE=${GOLEM_CH4_SIM_MODE:-full-timing}; TIMEOUT=${GOLEM_CH4_TIMEOUT:-600s}
mkdir -p "$ROOT/cases"
if [[ ! -e "$ROOT/status.tsv" ]]; then
    printf 'order\tphase\tcase\trc\tcase_root\treuse_n\treuse_m\tnode_credit\n' > "$ROOT/status.tsv"
fi

if [[ ! -f "$MANIFEST" ]]; then echo "[ERROR] missing manifest: $MANIFEST" >&2; exit 2; fi

python3 - "$MANIFEST" "$ROOT/plan.tsv" <<'PY'
import csv, re, sys
from pathlib import Path
source, target = map(Path, sys.argv[1:])
rows=[]
with source.open() as f:
    for line_no, raw in enumerate(f, 1):
        raw=raw.strip()
        if not raw or raw.startswith('#'): continue
        fields=raw.split()
        if len(fields) != 7:
            raise SystemExit(f"manifest line {line_no}: expected 7 fields")
        phase, case, rn, rm, credit, prefetch, note=fields
        for name, value in (("reuse_n",rn),("reuse_m",rm),("node_credit",credit),("prefetch",prefetch)):
            if not value.isdigit(): raise SystemExit(f"manifest line {line_no}: {name} must be integer")
        rows.append(dict(order=len(rows)+1, phase=phase, case=case, reuse_n=rn, reuse_m=rm,
                         node_credit=credit, prefetch=prefetch, note=note))
if not rows: raise SystemExit("manifest has no cases")
target.parent.mkdir(parents=True, exist_ok=True)
with target.open('w', newline='') as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t'); w.writeheader(); w.writerows(rows)
PY

echo "[CH4] root=$ROOT mode=$SIM_MODE workload=${M}x${N}x${K} block_k=$BK"
set +e
while IFS=$'\t' read -r order phase case reuse_n reuse_m node_credit prefetch note; do
    [[ "$order" == "order" ]] && continue
    case_root="$ROOT/cases/$(printf '%02d_%s' "$order" "$case")"
    mkdir -p "$case_root"
    echo "[START] $phase/$case reuse=${reuse_n}x${reuse_m} credit=$node_credit prefetch=$prefetch"
    env \
      GOLEM_RUN_ID="ch4_${STAMP}_${order}_${case}" \
      GOLEM_GEMM_M="$M" GOLEM_GEMM_N="$N" GOLEM_GEMM_K="$K" \
      GOLEM_ORIG_M="$M" GOLEM_ORIG_N="$N" GOLEM_ORIG_K="$K" \
      GOLEM_GEMM_BLOCK_M=64 GOLEM_GEMM_BLOCK_N=64 GOLEM_GEMM_BLOCK_K="$BK" \
      GOLEM_ARRAY_INPUT_SIZE=64 GOLEM_ARRAY_OUTPUT_SIZE=64 GOLEM_NUM_ARRAYS=64 \
      GOLEM_A_REUSE_N_TILES="$reuse_n" GOLEM_B_REUSE_M_TILES="$reuse_m" \
      GOLEM_DMA_NODE_CHUNK_CREDITS="$node_credit" GOLEM_WCP_PREFETCH_WINDOWS="$prefetch" \
      GOLEM_DMA_SLOT_COUNT=64 GOLEM_DMA_WINDOW_K_TILES=4 \
      GOLEM_GLOBAL_STRIDE_KB=4096 GOLEM_SIM_MODE="$SIM_MODE" \
      GOLEM_MPI_RANKS="$MPI_RANKS" GOLEM_SST_THREADS=1 \
      GOLEM_MPI_PARTITIONER=sst.self GOLEM_EXPLICIT_PARTITION=1 \
      GOLEM_TEST_BINARY="$case_root/test_noc_dma" \
      GOLEM_BUILD_METADATA_FILE="$case_root/test_noc_dma.build.env" \
      GOLEM_ARTIFACT_ROOT="$case_root/artifacts" GOLEM_HBM_DIR="$case_root/hbm" \
      GOLEM_SST_WORK_DIR="$case_root/work" GOLEM_RUN_SUMMARY_CSV="$case_root/run_summary.csv" \
      GOLEM_TENSOR_SOURCE=synthetic GOLEM_VERIFY_C=0 GOLEM_DUMP_C_FILE= \
      GOLEM_HBM_DUMP_OUTPUT=0 GOLEM_MEM_NODE_SIZE_BYTES=auto GOLEM_BENCH_QUIET_LOGS=1 \
      timeout "$TIMEOUT" "$PIPELINE" >"$case_root/pipeline.log" 2>&1
    rc=$?
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$order" "$phase" "$case" "$rc" "$case_root" "$reuse_n" "$reuse_m" "$node_credit" >> "$ROOT/status.tsv"
    if (( rc == 0 )); then echo "[DONE] $case"; else echo "[FAIL] $case rc=$rc"; fi
done < "$ROOT/plan.tsv"
set -e

python3 "$SCRIPT_DIR/stats/aggregate_chapter4.py" --root "$ROOT" --output "$ROOT/chapter4_results.csv"
python3 "$SCRIPT_DIR/stats/plot_chapter4.py" --input "$ROOT/chapter4_results.csv" --output-dir "$ROOT/plots"
echo "[CH4] results=$ROOT/chapter4_results.csv plots=$ROOT/plots"
