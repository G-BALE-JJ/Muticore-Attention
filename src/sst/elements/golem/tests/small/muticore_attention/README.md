# FlashAttention Workload

This directory contains the active FP32, non-causal GQA workload and its SST
runner, verification, and reporting tools.
The optimized point is `B=1,Sq=Skv=1024,Dh=128`. The active interface exposes
independent `Hq` and `Hkv`.

## Run

Build and install the local SST elements and RISC-V guest first. Then run the
8+8 worker-cluster configuration from the repository root. Attention tests
use four SST MPI ranks by default:

```bash
scripts/test_flash_attention.sh \
  --query-length 1024 --kv-length 1024 \
  --num-query-heads 4 --num-kv-heads 1 --head-dim 128
```

Override the simulator partition count with `--mpi-ranks 1`, `2`, or `4`.
MPI partitions simulator work without changing simulated completion latency:

```bash
scripts/test_flash_attention.sh \
  --query-length 1024 --kv-length 1024 --head-dim 128 --mpi-ranks 2
```

Use an explicit `--artifact-root` outside the source tree for retained local
runs.

## GQA Dataflow

Input storage is head-major: Q is `[Hq,Sq,Dh]`, while K/V are
`[Hkv,Skv,Dh]`. One composite job shares each K/V head across `Hq/Hkv` Query
heads. Query heads in a group are placed across workers, while each issued
`QK^T` or PV operation still uses all 64 arrays of its worker:

```text
HBM Q[hq], K[hkv], V[hkv]
          |
          v
  8 QK/SFU workers: QK^T
          |
          v
  row-dispatched online Softmax
          |
          v
  8 PV workers: P V  --->  HBM O[h]
          |
          v
 direct query-major O[Sq,Hq,Dh]
```

K/V data remains striped across four HBM nodes. Q and O remain partitioned
across four managers. Multiple K/V-group jobs are queued without a guest-side
barrier. `--heads N` remains an MHA compatibility alias for `Hq=Hkv=N`.

## Current 8+8 Contract

- Four managers and 16 workers: two QK/SFU and two PV workers per manager.
- QK outputs flow through bounded Score/P FIFOs into pipelined PV windows.
- Row-priority SFU scheduling exposes complete P windows earlier.
- Each V window is fetched once and broadcast into the existing PV local SRAM.
- The stable default uses static row-sticky PV placement; dynamic PV remains an
  opt-in experiment.

Key architecture controls include:

- `GOLEM_ATTENTION_CLUSTER_QK_ARRAYS`
- `GOLEM_ATTENTION_KEY_BLOCK_ROWS`
- `GOLEM_ATTENTION_KV_BUFFER_COUNT`
- `GOLEM_ATTENTION_KV_SHARED_STREAM_ENABLE`
- `GOLEM_ATTENTION_KV_STREAM_BYTES_PER_CYCLE`
- `GOLEM_WCP_GEMM_PROXY_*`
- `GOLEM_ATTENTION_NEAR_ARRAY_OUTPUT_*`

The accepted default is 8+8, with row priority, V broadcast, four-tile PV
windows, and static PV placement.

## Verification

The 8+8 SST bridge reports QK jobs, PV windows, assigned worker cores, fusion
completion, and end-to-end cycles from simulator events. The standalone
functional model checks online softmax against a full-attention reference.
The SST bridge currently exits after its cluster report; it does not run the
separate HBM-output numerical and lifecycle verifiers.

| Case | Resource floor | Frozen SST cycles | Status |
|---|---:|---:|---|
| Hq4/Hkv2/Sq1024/Skv1024/Dh128 | 32,768 | **47,193** | cluster report PASS |

Historical sequential-64 measurements are in
`../../../../../../../archive/attention_sequential_64/` and the existing
`ATTENTION_DIMENSION_SWEEP.md`.

## Key Files

- `run_fused_attention_scale.sh`: SST execution and artifact pipeline.
- `golem_attention_runtime.cpp`: RISC-V guest runtime.
- `verify_fused_attention_scale_stats.py`: lifecycle/resource verifier.
- `report_attention_metrics.py`: JSON/CSV metrics and terminal summary.
- `test_flash_attention_baseline_contract.py`: runner and contract tests.

Large HBM images, tensors, logs, CSV statistics, and run directories are
regenerable and intentionally not stored in Git.
