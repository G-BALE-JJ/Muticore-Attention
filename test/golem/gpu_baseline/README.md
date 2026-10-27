# GPU Baseline

This directory contains a CUDA/cuBLAS FP32 square GEMM baseline for comparing GOLEM matmul runs against a real GPU.

Default sweep:

```bash
./run_dim_sweep.sh
```

The script measures `C = A * B` for square dimensions `256 512 1024 2048` and writes:

```text
results/<timestamp>/gpu_dim_sweep.csv
results/<timestamp>/metadata.txt
```

Metrics:

- `latency_avg_ms`: average CUDA event latency over timed iterations.
- `latency_min_ms`: best observed latency, used for peak throughput.
- `avg_tflops`: `2 * dim^3 / latency_avg`.
- `peak_tflops`: `2 * dim^3 / latency_min`.
- `avg_util_pct`: `avg_tflops / ref_peak_tflops`.
- `peak_util_pct`: `peak_tflops / ref_peak_tflops`.

Defaults target the current machine's Tesla V100 FP32 peak:

```bash
GPU_BASELINE_PEAK_TFLOPS=15.7
CUDA_HOME=/usr/local/cuda-12.6
GPU_BASELINE_DIMS="256 512 1024 2048"
GPU_BASELINE_WARMUP=20
GPU_BASELINE_ITERS=100
GPU_BASELINE_BATCH=10
```

Each timed sample runs `GPU_BASELINE_BATCH` consecutive SGEMMs and divides elapsed time by the batch size. This reduces single-launch jitter while preserving per-GEMM latency and throughput units.
