# FlashAttention Workload

This directory contains the active FP32, non-causal, single-head Attention
workload and its SST runner, verification, reporting, and GPU comparison tools.
The primary contract is `B=1,H=1,D=128` with E3 Q=K=1024 and E4 Q=K=2048.

## Run

Build and install the local SST elements and RISC-V guest first. Then run the
verified cluster configuration from the repository root:

```bash
scripts/test_flash_attention.sh --attention-cluster \
  --queries 1024 --keys 1024 --head-dim 128

scripts/test_flash_attention.sh --attention-cluster \
  --queries 2048 --keys 2048 --head-dim 128
```

MPI partitions simulator work without changing simulated completion latency:

```bash
scripts/test_flash_attention.sh --attention-cluster \
  --queries 1024 --keys 1024 --head-dim 128 --mpi-ranks 4
```

Use `--no-attention-cluster` for the feature-off regression path. Use an
explicit `--artifact-root` outside the source tree for retained local runs.

## Current Cluster Contract

- Four managers and 16 workers.
- 64 physical 64x64 arrays and 64 CUs per worker.
- Default `16 QK + 48 PV` ownership, paired D64 paths for D128, Br16/Bc32.
- Two operand banks, bounded score/P/O contexts, group-four K/V reuse, direct
  score/P flow, tagged ahead scheduling, cancellation, and retry conservation.
- A timed 16-lane, two-cycle FP32 O FMA with finite read/write/ALU resources.

The following controls are architecture experiments rather than independent
performance claims:

- `GOLEM_ATTENTION_CLUSTER_QK_ARRAYS`
- `GOLEM_ATTENTION_KEY_BLOCK_ROWS`
- `GOLEM_ATTENTION_KV_BUFFER_COUNT`
- `GOLEM_WCP_GEMM_PROXY_*`
- `GOLEM_ATTENTION_NEAR_ARRAY_OUTPUT_*`

The accepted default is 16/48, Bc32, two worker K/V buffers, two manager K/V
distribution slots with demand-paced lookahead, one-row P read/program overlap,
WCP issue width one, and PV-to-O row fusion.
Bc64, a third K/V buffer, and WCP issue width two were measured and rejected.

## Verification

The runner executes Python contracts before simulation, checks numerical output,
and verifies lifecycle/resource conservation from SST statistics. Final R6
evidence is stored under the project-root `attention_cluster/results/` and is
hashed by `r6_final_manifest.sha256`.

| Case | Cycles | Status |
|---|---:|---|
| Q256/K128 | 16,112 | PASS |
| Q256/K1024 | 119,054 | PASS |
| E3 | 127,589 | PASS |
| E4 | 430,131 | PASS |
| WCP depth 16 | 16,112 | PASS |

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
