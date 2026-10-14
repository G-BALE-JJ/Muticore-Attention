# Projection scheduling with the existing input-prefetch slot

> **架构纠正（2026-10-10）：** 本文的历史投影/E2E 测量或重叠估计使用了 manager 本地 arrays，不符合「manager 只控制、16 个 worker 拥有阵列」的目标拓扑，不能作为该架构的性能证据。新的 worker 执行基准见 [WORKER_PROJECTION_ARCHITECTURE_FIX.md](WORKER_PROJECTION_ARCHITECTURE_FIX.md)。

The best measured fusion plan (RMSNorm/projection fusion, no RoPE fusion, no
intermediate audit writeback) now has an additional opt-in scheduling change:
`GOLEM_PROJECTION_WRAP_PREFETCH=1`. It changes request timing inside ProjectionJob,
not accelerator resources or operator interfaces. The default remains zero.

## Bottleneck and exclusive cycle breakdown

New `[PROJECTION_CYCLE_BREAKDOWN]` evidence records every controller state and
subdivides its foreground GM read state by input, weights and fusion operands.
These counters count exclusive controller tick samples, not summed resource
busy time. Background transfers overlap them and must not be added again.

For S1024, the original **foreground GM read state occupies 204,672 ticks per
manager**, 17.55% of manager 0's 1,166,101 sampled ticks. This is the largest
single transfer-wait state: output read occupies 145,920, and output restore's
ready state 123,840. The compute/launch state occupies 417,792 ticks, representing
productive array work and dispatch rather than an avoidable data wait.
GM reads split into weights 99,840, input 99,072, and fused operands 5,760.
The input portion exposes a scheduling gap with no need for additional storage.

The following is the complete, exclusive partition for manager 0 at S1024.
GM subcategories below partition the read state; they are not extra cycles.
Other managers have identical compute/GM work and small DMA/control differences.

| Controller residence | Control ticks | Scheduled ticks |
| --- | ---: | ---: |
| Array compute and launch | 417,792 | 417,792 |
| Foreground GM input read | 99,072 | 6,192 |
| Foreground GM weight read | 99,840 | 99,840 |
| Foreground GM fusion operand read | 5,760 | 5,760 |
| Foreground GM write | 4,608 | 4,608 |
| Array output read | 145,920 | 163,920 |
| Output restore, including dispatch | 126,720 | 103,680 |
| Weight programming, including dispatch | 64,512 | 64,512 |
| Input scatter | 58,368 | 33,888 |
| Weight HBM load/cache check | 8,523 | 10,902 |
| RMSNorm SFU | 111,808 | 111,808 |
| Normalization input DMA and dispatch | 13,635 | 13,635 |
| Output submission/completion/drain | 327 | 315 |
| Partial-store dispatch | 2,880 | 2,880 |
| Input/output dispatch | 6,336 | 6,336 |
| **Total sampled controller ticks** | **1,166,101** | **1,046,068** |

The full raw counters and all four managers' partitions for both lengths are
in [`projection_schedule_results.json`](projection_schedule_results.json).
The legacy PROJECTION_JOB end timestamp captures completion-DMA submission;
these new samples include that DMA's wait. Neither should be substituted for
the benchmark's manager-observed producer completion. E2E and stage spans below
retain the original observation boundaries.

## One scheduling change

The existing paired traversal keeps two weight tiles in the physical operand
banks and processes four 64-token row groups within a 256-token input block.
It already prefetches the next row group's input and partial while the current
row group computes. At the last row group, that future-input slot was unused;
the first group of the next hidden-tile pair performed a demand GM gather.

The new schedule uses that same slot during the last group's even-tile compute
to gather row zero of the next pair. Its source is the existing normalized input
block. Its previous partial is already available in the existing C-buffer or
GM partial region. The existing inactive-bank scatter path consumes the fetched
input. Weight programming, output restore and single-context accumulation still
use their existing interfaces and ordering.

This happens only when another pair exists within the same head and block.
Head/block boundaries retain their original behavior. The number of first-group
demand gathers falls from 16 to 1 per head/block at hidden2048; foreground input
read residence correspondingly falls by 93.75%. The complete GM input byte count
is unchanged: reads moved to the background rather than disappearing.

No extra SRAM, snapshot slot, array, output context, operand bank, port,
bandwidth, DMA credit, NoC link or ready-message interface is introduced.
The original pipeline capacity/eligibility checks still apply. Existing disabled
and non-paired paths retain their schedule.

## End-to-end evidence

All four formal cases use one identical simulator library (SHA256 recorded in
the results), identical input hashes, the same hardware configuration and the
same fusion/C-buffer settings. Controls reproduce the earlier best-fusion
cycles exactly. Host CPU affinity differs only to run independent experiments;
it is excluded from accelerator latency.

Shape: FP16 causal GQA, Hq32/Hkv8/D64, hidden2048, seed1742. The workload ends at O
HBM and excludes Wo, residual and MLP.

| Sequence | Control E2E cycles | Scheduled E2E cycles | Saved cycles | Reduction |
| --- | ---: | ---: | ---: | ---: |
| 1024 | 1,418,196 | **1,298,385** | 119,811 | **8.45%** |
| 2048 | 3,278,429 | **3,039,112** | 239,317 | **7.30%** |

| Critical-path segment, scheduled minus control | S1024 cycles | S2048 cycles |
| --- | ---: | ---: |
| Producer span | -120,061 | -239,667 |
| Handoff | -18 | -453 |
| Attention | +268 | +803 |
| **E2E** | **-119,811** | **-239,317** |

The gain is in projection. Attention remains effectively unchanged; small
variations arise from request timing. Some projection waits increase, including
output read (+18,000 sampled ticks per manager at S1024), because earlier input
traffic changes contention. The full exclusive partition includes those costs.

Every optimized case passed independent NumPy numerical verification, panel
layout checks, MPI partition checks, backend/lifecycle checks and work/traffic
accounting. All 16 retained Q/K/V panel and O regions are byte-identical to the
matched control (10 MiB at S1024, 20 MiB at S2048). Final O maximum absolute error
against the independent NumPy reference remains 0.0001220703125.

Across all managers at S1024, projection local reads remain 240 MiB, and
C-buffer partial reads/writes remain 90 MiB each. At S2048 they double. Array
compute/launch residence remains 417,792 / 835,584 ticks per manager. Numerical
precision, hidden-tile accumulation order and output layouts are unchanged.

## Regression and stress checks

The final Python suite passes 132 tests, including exclusive cycle-partition
checks. Both runner shell syntax checks and git diff checks pass.

An additional Hq4/Hkv2/D64/S1024 run disables C-buffer partial storage and uses
GM partials, a 64-slot GM queue, 64 B/cycle GM bandwidth, three-cycle GM latency,
and the existing 4096-byte request cap. It passes the same E2E numerical/layout,
backend, MPI and accounting checks (137,954 cycles). Each manager records
544–552 background-read retries, exercising the new prefetch under backpressure.
This altered stress profile is not included in the formal performance comparison.

A more extreme two-slot GM queue allows projection to finish but fails the
existing Attention LoadingKv path with this optimization both off and on.
Those matched failure logs are retained under
`results/projection_schedule/backpressure_supported{,_control}`. The original
standalone-runner and failed launch experiments are also retained. No Attention
queue handling or hardware capacity was changed to make the stress case pass.

## Reproduction

Use fresh output roots. The reference below supplies only verified inputs;
every HBM image and simulation is regenerated.

```bash
source scripts/env_local_install.sh
scripts/build_and_install_local.sh --no-autogen --jobs 8

GOLEM_PROJECTION_CBUFFER_PARTIAL=1 GOLEM_FUSION_NORM_PROJECTION=1 \
GOLEM_FUSION_SKIP_AUDIT_OUTPUT=1 GOLEM_FUSION_PROJECTION_ROPE=0 \
GOLEM_PROJECTION_WRAP_PREFETCH=0 \
python3 baseline/attention_cluster_8qk_8pv/run_llama_projection.py \
  --sequences 1024 2048 --artifact-root results/fresh_schedule/control \
  --reference-root results/operator_fusion_final/norm_production

GOLEM_PROJECTION_CBUFFER_PARTIAL=1 GOLEM_FUSION_NORM_PROJECTION=1 \
GOLEM_FUSION_SKIP_AUDIT_OUTPUT=1 GOLEM_FUSION_PROJECTION_ROPE=0 \
GOLEM_PROJECTION_WRAP_PREFETCH=1 \
python3 baseline/attention_cluster_8qk_8pv/run_llama_projection.py \
  --sequences 1024 2048 --artifact-root results/fresh_schedule/wrap_prefetch \
  --reference-root results/fresh_schedule/control

python3 baseline/attention_cluster_8qk_8pv/analyze_projection_schedule.py \
  --artifact-root results/fresh_schedule
```

The analyzer verifies matching inputs, library, hardware/work/traffic and exact
outputs before writing `analysis.json` and `analysis.csv`. Formal results are
under `results/projection_schedule`; the compact tracked results preserve all
exclusive state counters and the supplementary stress outcome.
