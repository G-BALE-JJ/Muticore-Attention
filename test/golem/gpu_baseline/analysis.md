# V100 GPU Baseline Cycle Conversion

Run directory:

```text
results/20260426_165835_batch10
```

Measurement setup:

```text
GPU: Tesla V100-SXM2-32GB
CUDA: /usr/local/cuda-12.6
Benchmark: FP32 cuBLAS SGEMM, C = A * B
Matrix shape: dim x dim square GEMM
warmup: 20 samples
iters: 100 timed samples
batch: 10 SGEMMs per timed sample
reference FP32 peak: 15.7 TFLOP/s
cycle conversion clock: 1530 MHz max SM clock
```

Cycle conversion:

```text
cycles = latency_ms * 1e-3 s/ms * 1.53e9 cycles/s
       = latency_ms * 1.53e6 cycles
```

| dim | avg latency ms | min latency ms | max latency ms | avg cycles @1.53GHz | min cycles @1.53GHz | max cycles @1.53GHz | avg TFLOP/s | peak TFLOP/s | avg util | peak util |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.018803 | 0.012288 | 0.412262 | 28769 | 18801 | 630761 | 1.784555 | 2.730667 | 11.367% | 17.393% |
| 512 | 0.037225 | 0.034099 | 0.249037 | 56954 | 52171 | 381027 | 7.211069 | 7.872192 | 45.930% | 50.141% |
| 1024 | 0.194244 | 0.184422 | 0.607744 | 297193 | 282166 | 929848 | 11.055622 | 11.644375 | 70.418% | 74.168% |
| 2048 | 1.175474 | 1.161216 | 1.580237 | 1797475 | 1776660 | 2417763 | 14.615267 | 14.794723 | 93.091% | 94.234% |

Interpretation:

```text
avg cycles: average steady-state per-GEMM device execution cycles
min cycles: best observed per-GEMM device execution cycles, used for peak throughput
max cycles: worst observed per-GEMM sample, includes runtime/GPU scheduling jitter
```

The GPU event interval includes GPU-side memory access from device memory/cache hierarchy to compute units, compute execution, and writeback to device memory. It excludes host-to-device copies, device-to-host copies, allocation, input initialization, and CPU launch overhead.
