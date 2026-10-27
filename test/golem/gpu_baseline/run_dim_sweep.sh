#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.6}
NVCC=${NVCC:-${CUDA_HOME}/bin/nvcc}
DEVICE=${CUDA_VISIBLE_DEVICES:-0}
GPU_DEVICE_INDEX=${GPU_DEVICE_INDEX:-0}
PEAK_TFLOPS=${GPU_BASELINE_PEAK_TFLOPS:-15.7}
WARMUP=${GPU_BASELINE_WARMUP:-20}
ITERS=${GPU_BASELINE_ITERS:-100}
BATCH=${GPU_BASELINE_BATCH:-10}
DIMS=${GPU_BASELINE_DIMS:-"256 512 1024 2048"}
OUT_DIR=${GPU_BASELINE_OUT_DIR:-${SCRIPT_DIR}/results/$(date +%Y%m%d_%H%M%S)}
BIN=${SCRIPT_DIR}/gemm_bench

if [[ ! -x "${NVCC}" ]]; then
  echo "nvcc not found or not executable: ${NVCC}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

"${NVCC}" -O3 -std=c++17 "${SCRIPT_DIR}/gemm_bench.cu" -lcublas -o "${BIN}"

SUMMARY=${OUT_DIR}/gpu_dim_sweep.csv
META=${OUT_DIR}/metadata.txt

{
  echo "date=$(date -Is)"
  echo "cuda_home=${CUDA_HOME}"
  echo "nvcc=${NVCC}"
  echo "cuda_visible_devices=${DEVICE}"
  echo "gpu_device_index=${GPU_DEVICE_INDEX}"
  echo "peak_tflops_ref=${PEAK_TFLOPS}"
  echo "warmup=${WARMUP}"
  echo "iters=${ITERS}"
  echo "batch=${BATCH}"
  echo "dims=${DIMS}"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true
} > "${META}"

first=1
for dim in ${DIMS}; do
  tmp=${OUT_DIR}/dim_${dim}.csv
  "${BIN}" --dim "${dim}" --warmup "${WARMUP}" --iters "${ITERS}" --batch "${BATCH}" \
    --device "${GPU_DEVICE_INDEX}" --peak-tflops "${PEAK_TFLOPS}" --csv > "${tmp}"
  if [[ ${first} -eq 1 ]]; then
    cat "${tmp}" > "${SUMMARY}"
    first=0
  else
    awk 'NR > 1' "${tmp}" >> "${SUMMARY}"
  fi
done

echo "summary=${SUMMARY}"
cat "${SUMMARY}"
