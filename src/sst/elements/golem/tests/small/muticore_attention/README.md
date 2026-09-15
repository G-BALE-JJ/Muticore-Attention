# FlashAttention Workload

This directory contains the active FP32, non-causal, single-head Attention
workload and its SST runner, verification, reporting, and GPU comparison tools.
Performance work is fixed to `B=1,H=1,Q=K=1024,D=128`; other dimensions are
used only for correctness contracts until this case is fully optimized.

## Run

Build and install the local SST elements and RISC-V guest first. Then run the
verified sequential-64 configuration from the repository root:

```bash
scripts/test_flash_attention.sh --sequential-64 \
  --queries 1024 --keys 1024 --head-dim 128
```

MPI partitions simulator work without changing simulated completion latency:

```bash
scripts/test_flash_attention.sh --sequential-64 \
  --queries 1024 --keys 1024 --head-dim 128 --mpi-ranks 4
```

Use `--attention-cluster` only for the historical cluster path. Use an explicit
`--artifact-root` outside the source tree for retained local runs.

## Current Sequential Contract

- Four managers and 16 workers.
- 64 physical 64x64 arrays and 64 CUs per worker.
- All 64 arrays execute QK, then all 64 execute PV after online softmax.
- Br64/Bc64/D128 uses two QK D64 reduction slices and two PV D64 output slices.
- Matrix broadcast, vector scatter, and grouped O scatter/gather are independently
  modeled at 256 B/cycle aggregate.
- Shared memory nodes coalesce identical 16 KiB K/V chunks, retain completed
  chunks for late consumers, and multicast them to all 16 workers.
- K readiness can start QK independently; V readiness is required only at PV.

Key architecture controls include:

- `GOLEM_ATTENTION_CLUSTER_QK_ARRAYS`
- `GOLEM_ATTENTION_KEY_BLOCK_ROWS`
- `GOLEM_ATTENTION_KV_BUFFER_COUNT`
- `GOLEM_ATTENTION_KV_SHARED_STREAM_ENABLE`
- `GOLEM_ATTENTION_KV_STREAM_BYTES_PER_CYCLE`
- `GOLEM_WCP_GEMM_PROXY_*`
- `GOLEM_ATTENTION_NEAR_ARRAY_OUTPUT_*`

The accepted default is sequential-64, Bc64, two worker K/V buffers, shared K/V
streaming, WCP issue width one, and grouped PV O accumulation. Three complete
K/V pairs do not fit the 256 KiB Attention window.

## Verification

The runner checks the instantiated memory backend, numerical output, and
lifecycle/resource conservation from SST statistics.

| Case | Cycles | Status |
|---|---:|---|
| Q1024/K1024/D128 | **31,197** | backend/numerical/lifecycle PASS |

The current theoretical lower bound is 20,600 cycles. K/V performs exactly 64
physical HBM chunk reads for 1 MiB of unique data, and the critical worker has
zero exposed K/V wait. Softmax is the largest remaining gap.

## Key Files

- `run_fused_attention_scale.sh`: SST execution and artifact pipeline.
- `golem_attention_runtime.cpp`: RISC-V guest runtime.
- `verify_fused_attention_scale_stats.py`: lifecycle/resource verifier.
- `report_attention_metrics.py`: JSON/CSV metrics and terminal summary.
- `test_flash_attention_baseline_contract.py`: runner and contract tests.
- `gpu_attention_stage_benchmark.py`: external GPU measurement tool.
- `report_attention_gpu_comparison.py`: SST/GPU comparison report.
- `GPU_COMPETITIVE_ROADMAP.md`: current target, evidence, and next steps.

Large HBM images, tensors, logs, CSV statistics, and run directories are
regenerable and intentionally not stored in Git.
