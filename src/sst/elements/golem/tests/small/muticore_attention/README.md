# FlashAttention Workload

This directory contains the active FP32, non-causal GQA workload and its SST
runner, verification, and reporting tools.
The optimized point is `B=1,Sq=Skv=1024,Dh=128`. The active interface exposes
independent `Hq` and `Hkv`.

## Run

Build and install the local SST elements and RISC-V guest first. Then run the
verified sequential-64 configuration from the repository root. Attention tests
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
  64 arrays/worker: QK^T
          |
          v
  row-dispatched online Softmax
          |
          v
  64 arrays/worker: P V  --->  HBM O[h]
          |
          v
 direct query-major O[Sq,Hq,Dh]
```

K/V data remains striped across four HBM nodes. Q and O remain partitioned
across four managers. Multiple K/V-group jobs are queued without a guest-side
barrier. `--heads N` remains an MHA compatibility alias for `Hq=Hkv=N`.

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
- The SFU is an event-driven physical operator model with separately configurable
  Scale, compare, MAX/SUM reduction, EXP, online merge, reciprocal, and
  normalize resources. See
  [SFU_HARDWARE_MODEL.md](SFU_HARDWARE_MODEL.md).
- Sequential SFU traffic uses one 16 KiB score read and one 16 KiB P write per
  key tile through an independently configurable 256 B/cycle SRAM stream.
- The SFU row dispatcher launches one Br64 row every four cycles by default,
  matching four 16-lane vector beats; its interval is independently configurable.
- Canonical names and deprecated aliases are listed in
  [ATTENTION_TERMINOLOGY.md](../../../../../../../attention_sequential_64/ATTENTION_TERMINOLOGY.md).

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

| Case | Theory | Measured | Gap | Status |
|---|---:|---:|---:|---|
| H1/Q1024/K1024/D128 | 24,256 | **27,186** | 12.08% | all checks PASS |
| H2/Q1024/K1024/D128 | 48,512 | **54,278** | 11.89% | all checks PASS |
| Hq4/Hkv1/Sq1024/Skv1024/Dh128 | 97,024 | **106,939** | 10.22% | all checks PASS |

For one Q1024/K1024/D128 head, the dependency-preserving lower bound is 24,256
cycles. Because the current design deliberately gives all 64 arrays to one head
and serializes heads, its multi-head lower bound is `H * 24,256` cycles. This is
the latency of computing every head on one Attention engine, not the lower bound
of a hypothetical design with a separate 64-array engine for every head.

The verified Q=K 256/512/1024/2048/4096 cycle sweep and dimension-aware lower
bound are recorded in [ATTENTION_DIMENSION_SWEEP.md](ATTENTION_DIMENSION_SWEEP.md).

The arithmetic-only Softmax floor is 4,856 cycles with ideal beat streaming, or
6,008 cycles while retaining the current beat-at-a-time controller dependency.
Including two serialized 65-cycle tile streams across 16 key tiles raises those
bounds to 6,936 and 8,088 cycles; measurement is 8,870 cycles. K/V performs
exactly 64 physical HBM chunk reads for 1 MiB of unique data, and the critical
worker has zero exposed K/V wait.

## Key Files

- `run_fused_attention_scale.sh`: SST execution and artifact pipeline.
- `golem_attention_runtime.cpp`: RISC-V guest runtime.
- `verify_fused_attention_scale_stats.py`: lifecycle/resource verifier.
- `report_attention_metrics.py`: JSON/CSV metrics and terminal summary.
- `test_flash_attention_baseline_contract.py`: runner and contract tests.

Large HBM images, tensors, logs, CSV statistics, and run directories are
regenerable and intentionally not stored in Git.
