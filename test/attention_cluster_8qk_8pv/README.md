# P0–P4 cycle 优化验收（2026-10-24）

当前已采用同构建复现的 groups=2：完整端到端 **6,084,167.75 cycles**，
两层 decoder **5,343,797.75 cycles**。相对 groups=1，完整端到端减少
**50,610.625 cycles（0.825%）**。最终 Y / Norm / 全部 128256 logits 逐位一致。
所有新延迟使用 1 GHz 参考 cycles；宿主 wall time 单独记录。

全局与 FFN 专用 lookahead=2 虽通过局部筛选，但正式全链分别退步
31,004.25 / 58,466 cycles，均不采用；保持原深度 4。
PV→Wo landing 直读已实现并通过生命周期、bitwise、traffic 和 lease 审计；正式矩阵中比 control 慢，保留为 opt-in，不改变默认配置。
当前源码回归 **250 项 host + 176 项生产路径全部通过**。

[本轮执行结果与采用判断](PROGRESS_2026-10-24.md) ·
[原定 P0–P4 计划](OPTIMIZATION_PLAN_2026-10-24.md) ·
[PV→Wo 直读契约](PV_WO_DIRECT_INPUT_2026-10-24.md)。
下文为历史报告，保留其原始单位与测量边界。

# GM 模型校准

**软件融合整组优化（2026-10-22，独立 all-gemm-reuse-window 分支）：**
在既有片上驻留基础上实现 Add/RMSNorm2 融合、Norm2 GM 写请求合并、末层
FFN/final Norm 的 worker 级流水，并把不可变配置解析移出消费交接路径。
正式 S=1024 两层 decoder **5.458952 → 5.400835 ms（−1.06%）**；
含 embedding/final Norm/LM 的端到端 **6.288746 → 6.134778 ms（−2.45%）**。
相对本次同构建对照，端到端再下降 0.49%；减少 8 MiB GM 读取和 75% 的
Norm2 写请求，新增 HBM 节省为 0 B。240 项回归及两组正式数值/资源审计通过。
保持原硬件和全部 GEMM 的 reuse-window；完整 S=1024、两层、16 worker
配对结果、数值/资源审计、复现命令和仍存在的 HBM 边界统一记录在
[整组实施与测量报告](SOFTWARE_FUSION_BATCH_2026-10-22.md)。

**RTX 5060 GPU 实验交接（2026-10-16）：**
[SSH 查阅、精简传输、计算图与测量契约](GPU_PREFILL_HANDOFF_2026-10-16.md)。
当前通用性回归正式两层为 **7,424,562 cycles**；GPU prefill runner 尚待实现。
交接清单包含当前工作树哈希、冻结输入和普通数组参考，无需复制完整 HBM/build。

**宿主磁盘产物管理（2026-10-16）：** 主要 runner 默认在验收成功后核对并
删除重复 MPI 日志、回收张量全零块；生成输入前检查至少 24 GiB 剩余空间。
失败实验保留原始日志。构建复制排除所有 `results/` 并保留稀疏文件。
显式归档清理及完整保留选项见 [产物保留策略](ARTIFACT_STORAGE.md)。

当前 banked GM 下，FFN 内部保持 tile-major 并直接写最终 row-major down panel，正式两层 MPI8 从同构建 **7,483,267.75 → 7,362,979.75 cycles（−1.61%）**。新增 `--ffn-layout 2`，保留四槽预取、16-tile FFN/down/残差融合、两组数组与 LM N=4。硬件未改，每 worker 仍为 2 MiB GM；输出/logits 比特相同，302 项回归和完整资源审计通过。输入复用实验因 C-buffer 开销变慢，保留关闭。参见 [FFN GM 四步实验及正式验收](FFN_GM_FLOW_2026-10-16.md)。

GM 校准补齐入口/bank 冲突与 WCP DMA 落地，校准对照为 **7,483,267.75 cycles**；张量未计时绕行均为 0 B。参见 [GM banked 实现与验证](GM_ALIGNMENT_VALIDATION_2026-10-16.md)；下文历史优化结果属于 legacy 模型。

# 8 QK/SFU + 8 PV attention cluster

**GM model alignment review (2026-10-16):** retain **2 MiB per worker**.
The [review and candidate contract](GM_ALIGNMENT_REVIEW_2026-10-16.md) separate
the reference SST abstraction from its HDL resource model, and bound client
ingress, bank conflicts, read/write ordering and response backpressure together.
The banked implementation and acceptance evidence are in the
[validation report](GM_ALIGNMENT_VALIDATION_2026-10-16.md). Formal runners now
default to banked; the historical results below use legacy timing.
Model-calibration differences must not be reported as software scheduling speedups.

**Latest LM head N parallelism (2026-10-15):** map distinct vocabulary tiles to
existing single-context arrays with `--lm-n-parallel 1|2|4|8`. Local formal
16-worker / MPI8 screening selects **4**, reducing LM cycles by **4.68%**;
2 regresses and 8 is slightly slower than 4. Matched full two-layer prefill
reduces LM head **633,901.75 → 602,504.75 cycles (−4.95%)**, and total
**6,262,408.75 → 6,231,011.75 cycles (−0.50%)**. All upstream cycles are identical,
outputs/logits are bit-identical, and dense weight/DMA/SFU work is unchanged.
No hardware changes; 302 regression tests, 12 local SST cases and both full
cases pass. Keep the fusion/prefetch/FFN settings below and explicitly add
`--lm-n-parallel 4`; default remains 1. See the
[implementation, local screening and full acceptance](LM_N_PARALLEL_2026-10-15.md).

**Previous FFN array subset scheduling (2026-10-15):** interleave two disjoint
32-array subsets of the existing 64 arrays, keeping one operand context per
array and all hardware parameters unchanged. Formal single-worker / 16-worker
FFN+residual cycles decrease by **8.56% / 4.60%**. Matched two-layer MPI8 prefill
decreases **6,519,275.75 → 6,262,408.75 cycles (−3.94%)**, with bit-identical
outputs/logits and equal DMA/GM and SFU work. All 299 regression tests pass.
Use `--ffn-array-groups 2` with four-slot prefetch and the fusion settings below;
the default remains one group. Extra broadcasts are charged to the existing
shared port. See the [implementation and evidence](ARRAY_GROUP_PIPELINE_2026-10-15.md).

**Previous down-to-residual pipeline (2026-10-15):** existing down panels feed the
shared SFU Add before final layer-output DMA. Formal MPI8 prefill decreases from
the matched current-build **6,575,251.75** to **6,519,275.75 cycles (−0.85%)**;
formal 16-worker local FFN+residual decreases by 1.55%. Two layers save 16 MiB
of HBM traffic and 32 standalone jobs. Outputs and logits are bit-identical,
with equal EXP/reciprocal/vector-op work. All 297 tests pass. No hardware resource
or timing parameters change: 8 KiB scratch reuses the existing GM input arena.
Enable explicitly with `--down-residual-fusion` in addition to the settings below.
See the [implementation and matched evidence](RESIDUAL_PIPELINE_2026-10-15.md).

**Previous weight prefetch experiments (2026-10-15):** real 2/4/8-slot queues,
foreground DMA/GM wait instrumentation, bounded cross-worker reuse replay, and
local-to-full validation are complete. Four slots reduce formal 16-worker FFN
cycles by **31.85%** and full two-layer prefill from **8,827,863**
to **6,559,174.75 cycles (−25.70%)**, with bit-identical outputs
and logits. Use `--gemm-prefetch --ffn-down-fusion --ffn-panel-tiles 16 --prefetch-slots 4`.
The controller schedule uses the existing 2 MiB GM (maximum layout 1.945 MiB),
64 arrays and one operand bank. All 295 tests pass. Reuse replay estimates bytes;
it does not implement multicast or claim sharing cycle savings. See the
[complete experiments](WEIGHT_PREFETCH_2026-10-15.md).

**Previous gated-to-down fusion (2026-10-15):** op9 consumes gated panels in GM and
preserves K-order FP16 partials in the existing timed C-buffer. The formal 8-rank
two-layer prefill measures **8,827,863 cycles**, down **1.96%** from the matched
current-build op8 baseline (9,003,939). Use `--ffn-down-fusion --ffn-panel-tiles 16`
for this shape. Outputs and logits are bit-identical; two layers save 64 MiB of
HBM traffic without increasing SRAM capacity. All 292 regression tests pass.
See the [down fusion experiments](FFN_DOWN_FUSION_2026-10-15.md).

**Previous FFN producer fusion (2026-10-14):** gate/up producer panels now feed the shared
SiLU/Multiply SFU directly, with bounded GM ownership and final gated ACKs.
The formal 8-rank two-layer prefill measures **8,807,050.75 cycles** with 8-tile
panels, down **14.29%** from the matched current-binary baseline. Layer outputs
and logits are bit-identical; two layers save 136 MiB of HBM traffic. See the
[FFN panel implementation and validation](FFN_PANEL_FUSION_2026-10-14.md).

**Previous dataflow progress (2026-10-14):** generic GEMM output panels, bounded weight
prefetch, and chunked SiLU/Multiply fusion are validated through timed Embedding,
two Decoder layers, final RMSNorm, and last-token LM head. The formal 8-rank run
measures **10,210,612.75 cycles**, down 49.37% from the pre-optimization archive,
with bit-identical layer outputs and logits. See the
[dataflow implementation and matched comparisons](DECODER_DATAFLOW_2026-10-14.md)
and [current progress](PROGRESS_2026-10-14.md).

The earlier 4/8-rank archive has identical logits and 20,167,497.75 cycles;
8 ranks reduce simulator wall time by 12.08% in that matched single-run comparison.
See the [archived results](progress_results/2026-10-14/embedding_token_summary.md).

See the [full-prefill cycle and lower-bound analysis](DECODER_E2E_ANALYSIS_2026-10-14.md)
for pre-optimization additive phase timings, overlapping worker profiles, resource bounds,
and the evidence for optimizing generic GEMM output DMA and prefetch before FFN fusion.

**Canonical DMA timing (2026-10-12):** the primary report is `dma_e2e_result.json`, contract `hbmdma_to_O_ack_v1`: first workload data DMA sent → last final O write ACK. Clean IW S1024/S2048 measure **473,902.75 / 1,238,814.75 cycles**. Legacy job-to-worker fields below retain historical boundaries. Current group1 + producer barrier resource lower bounds are **352,816 / 967,776 cycles**, necessary bounds rather than attained optimal schedules. See [timing and lower-bound analysis](DMA_LOWER_BOUND_2026-10-12.md).

**Clean startup measurements (2026-10-12):** guest diagnostic `fprintf`
calls were serializing worker releases. Build `make clean-startup` and select
`riscv64/clean_startup/fused_attention` for performance experiments. The IW
path now measures S1024/S2048 E2E at 473,914/1,238,826 cycles; output arrival
from simulation zero is 551,750.75/1,316,662.75 cycles. This removes benchmark
diagnostic overhead, not accelerator work. Error/completion reporting and
host resource traces remain enabled. See
[`STARTUP_RELEASE_2026-10-12.md`](STARTUP_RELEASE_2026-10-12.md) for the paired
experiments, timing boundaries and the observed S2048 schedule sensitivity.

**Read/program/compute overlap (2026-10-11):** matched formal experiments
reduce S1024/S2048 E2E from 573,772/1,354,890 to 544,931/1,304,066 cycles
(5.03%/3.75%) using the existing future-input slot and a local-weight snapshot
within the original staging budget. Use `GOLEM_PROJECTION_INPUT_PAIR_PREFETCH=1`,
`GOLEM_PROJECTION_LOCAL_WEIGHT_PREFETCH=1`, and `GOLEM_PROJECTION_ARRAY_GROUPS=1`
with the 64-array retained-output pipeline. Splitting the existing arrays into
two groups overlaps programming with MAC, but the extra broadcasts cancel its
additional benefit in both formal shapes. The new options are opt-in. See the
[complete experiment report](PROJECTION_READ_PROGRAM_COMPUTE_OVERLAP_2026-10-11.md)
and [matched results](projection_overlap_results.json).

**Dataflow direction (2026-10-11):** the 32-array weight-stationary prototype
has been retired at the user's request. The supported path retains outputs
on the existing 64 arrays and uses one operand context per array. See the
[latency reanalysis](LATENCY_REANALYSIS_WITHOUT_WEIGHT_STATIONARY_2026-10-11.md)
for reconciled timing evidence and the new software overlap experiments.

**Architecture correction (2026-10-10):** dedicated managers have no arrays or
SFUs. Projection and RMSNorm now execute on cores 4..19. Historical projection
and fusion measurements that executed on manager-local arrays are invalid as
performance evidence for this target topology, even where numerical checks
passed. See `WORKER_PROJECTION_ARCHITECTURE_FIX.md` for the corrected baseline.

**Single operand context (2026-10-10):** extra array-local operand banks have
been removed. Every worker retains 64 physical arrays, with one matrix and one
input context per array; managers have no arrays or SFUs. Existing GM capacity,
ports, bandwidth, and output accumulators are unchanged. The older corrected
worker and retained-output measurements used two operand contexts and are
historical references. See [`SINGLE_OPERAND_CONTEXT.md`](SINGLE_OPERAND_CONTEXT.md)
for the current dataflow and validation.

This is the final worker-cluster architecture. The archived 4:12 comparison is
in `../../archive/baseline_attention_cluster_4qk_12pv/`.

## Current SST verification

The 2026-09-28 long-sequence changes reduce the Hq=4/Hkv=2/D=128/S=4096
critical path from 1,161,522 to 1,015,210 cycles (12.60%). S=2048 falls from
282,982 to 241,792 cycles (14.56%); S=512 and S=1024 retain their cycle counts.
The implementation now supports D=64 as well as D=128. See
[`LONG_SEQUENCE_ANALYSIS.md`](LONG_SEQUENCE_ANALYSIS.md) for the root cause,
controlled comparisons, remaining bottlenecks, and Llama Hq=32/Hkv=8/D=64
measurements. The baseline runner and sweep default to 4 MPI ranks and
local-GM queue depth 256.

The Llama 3.2 1B head/dimension sweep (Hq=32, Hkv=8, D=64) passes at
S=512/1024/2048/4096 with 70,158 / 267,869 / 1,054,344 / 4,200,595 cycles.
At S=4096 this is 1.00150 times the dense SFU resource floor. Results are in
`results/llama32_8_d64/summary-latest.csv`. These are FP32, noncausal attention
kernel measurements, not full Llama inference or numerical-output validation.

A 2026-09-24 rerun of Hq4/Hkv2/Q1024/K1024/D128 completed all 16 QK jobs,
256 PV windows, and 16 end-to-end jobs on cores 4..11 and 12..19. The
measured critical path was 50,329 cycles (`start=63,668`, `end=113,997`).
The PV WCP stall was caused by requesting a 16 KiB V panel through a local
SRAM interface limited to 4 KiB per request; the WCP now reads each panel in
bounded chunks. The `47,193`-cycle value below remains the frozen
2026-09-19 comparison, not a measurement of this worktree. Current runs also
write the PV output to HBM and check it numerically, so their cycle counts
include output DMA and are not directly comparable to that frozen value.

## FP16 end-to-end validation

The default SST run stores Q/K/V, the QK-to-PV probability payload, and O in
FP16. QK accumulation, online softmax state, and PV accumulation remain FP32.
The runner checks the completed worker jobs, compares every O element against
a NumPy attention reference, rejects all-zero or non-finite output, verifies
4-rank component placement, and checks every striped HBM Q/K/V tensor and
packed panel against its source file. A case is PASS only when all checks pass.

```bash
source scripts/env_local_install.sh
baseline/attention_cluster_8qk_8pv/run_sst.sh \
  --query-length 512 --kv-length 512 --head-dim 64 \
  --num-query-heads 4 --num-kv-heads 2 --dtype fp16 \
  --artifact-root /tmp/attention_fp16_e2e
python3 scripts/sweep_attention.py --head-dim 64 \
  --output-root results/fp16_e2e_d64
```

The FP16 D64 and D128 sweeps with `Hq:Hkv = 1:1, 2:1, 4:2, 4:1` and
`Q=K=512,1024,2048,4096` each passed all 16 cases using 4 MPI ranks.
Their cycle, numerical-error, MPI, and HBM-layout results are in
`results/fp16_e2e_d64/summary-latest.csv` and
`results/fp16_e2e_d128/summary-latest.csv`. The largest absolute output
errors across the two sweeps were `4.13e-6` and `4.36e-6`, respectively.

| Q=K=4096, Hq:Hkv | FP16 D64 cycles | FP16 D128 cycles |
| --- | ---: | ---: |
| 1:1 | 135,311 | 142,611 |
| 2:1 | 277,355 | 275,878 |
| 4:2 | 528,420 | 538,635 |
| 4:1 | 538,283 | 549,231 |

The packed K panel address now
selects the HBM node for each 64-key tile, including reuse windows that cross
a node boundary. D128 PV consumes all four contiguous P panels in each
256-key window. Both fixes are required for numerically correct multi-head
and D128 runs.

The FP16 `Hq=32,Hkv=8,D=64` kernel sweep also passed at
`Q=K=512,1024,2048,4096`: end-to-end cycles were `69,957`, `266,857`,
`1,054,104`, and `4,199,892`. The maximum absolute output error was
`5.30e-6`; detailed results are in
`results/fp16_e2e_llama32_8_d64/summary-latest.csv`. This is noncausal
attention-kernel validation, not full Llama inference.

## RMSNorm and projection end-to-end

Fixed-resource RMSNorm/projection and projection/RoPE fusion is available as
an opt-in execution plan. See [OPERATOR_FUSION.md](OPERATOR_FUSION.md) for
storage ownership, controls, validation and cycle interpretation.

The latest measured Llama-shape projection pipeline and input sharing results
are in [LLAMA_PROJECTION_PIPELINE.md](LLAMA_PROJECTION_PIPELINE.md). Its
comparison reference is the two-bank paired-weight implementation at `adb912f`.

The current audited Llama-shape baseline is documented in
[LLAMA_PROJECTION_BASELINE.md](LLAMA_PROJECTION_BASELINE.md). The historical
measurements below precede the projection/QK/RoPE local-GM timing correction
and the configured-array resource-floor correction.

The subsequent D64/hidden2048 optimization is documented in
[LLAMA_PROJECTION_PAIR_REUSE.md](LLAMA_PROJECTION_PAIR_REUSE.md): two resident
weight tiles process a bounded row block with timed FP16 partial staging.

The verified 16/32/64-array comparison, address mapping repair, and reproduction
commands are in [LLAMA_PROJECTION_ARRAYS.md](LLAMA_PROJECTION_ARRAYS.md).
The dedicated baseline defaults to 64 projection arrays and bounded weight
prefetch; see [LLAMA_PROJECTION_WEIGHT_PREFETCH.md](LLAMA_PROJECTION_WEIGHT_PREFETCH.md)
for the measured results and the output-context design study.

The `--projection` path starts from FP16 X, gamma, and projection weights in
HBM. The four manager cores run RMSNorm, Q/K/V projection and the causal RoPE
attention kernel. It verifies the projected raw and panel layouts, final O,
memory backend, and MPI placement. Projection uses FP16 multiplication and
accumulation; the existing attention workers still use FP32 for QK/softmax/PV
internal accumulation while storing tensor payloads in FP16.

```bash
baseline/attention_cluster_8qk_8pv/run_sst.sh --projection \
  --query-length 1024 --kv-length 1024 --num-query-heads 1 \
  --num-kv-heads 1 --head-dim 128 \
  --artifact-root /tmp/attention_projection_e2e
```

`projection_e2e_result.json` records RMSNorm, projection, and attention stage
spans, handoff gaps, weight reuse, total cycles, and an optimistic resource
floor. A negative RMSNorm-to-projection gap means different managers overlap
those stages. The floor uses RMSNorm vector work, the configured 64x64 arrays per
manager, and 16 exp lanes per QK worker; it omits data movement, control, and
handoff costs. The stages can overlap across managers and should not be added
as elapsed times.

The projection manager retains up to 64 weight tiles (512 KiB) in local GM.
For D=128, two operand banks allow the next input tile to be scattered while
the first tile computes. Up to four output DMAs may remain in flight as the
next tile starts; all writes drain before the completion flag. Raw Q/K/V for
each 16-token group is written with one DMA after all dimension tiles complete.
The guest prepares attention descriptors while the projection job runs, then
waits for all four managers before dispatching attention.

Before the two-bank/pipelined writeback change, all four `S=1024` projection
cases passed numerical, HBM layout, backend, and MPI checks. These archived
measurements run from the first RMSNorm issue to the last attention worker
completion:

| Hq:Hkv | D | End-to-end cycles | Optimistic floor | Actual / floor |
| --- | ---: | ---: | ---: | ---: |
| 1:1 | 64 | 50,685 | 15,670 | 3.24x |
| 1:1 | 128 | 98,333 | 19,916 | 4.94x |
| 2:1 | 64 | 85,246 | 24,204 | 3.52x |
| 2:1 | 128 | 209,088 | 31,768 | 6.58x |

The single-head D64 case took 73,514 cycles before weight reuse and descriptor
overlap; D128 took 185,132 cycles before those optimizations and the grouped
raw writeback. The gap to the resource floor is primarily projection transfer
and programming overhead, especially at D128, followed by guest descriptor
setup and attention scheduling.

### Projection pipeline validation and remaining cost

The 2026-10-04 rebuild and 4-rank reruns passed numerical output, projected
HBM layout, memory backend, MPI placement, and stage reporting. The table
compares the previous measurements above with the current two-bank/pipelined
writeback implementation. All S=1024 runs use causal RoPE attention and FP16
projection accumulation.

| Hq:Hkv | D | Previous cycles | Current cycles | Projection span | Projection weight loads / programs | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1:1 | 64 | 50,685 | 45,814 | 12,116 | 12 / 12 | 9.6% |
| 1:1 | 128 | 98,333 | 79,867 | 33,352 | 48 / 768 | 18.8% |
| 2:1 | 64 | 85,246 | 68,633 | 20,504 | 32 / 32 | 19.5% |
| 2:1 | 128 | 209,088 | 149,497 | 79,966 | 128 / 2,048 | 28.5% |

An additional `S=2048,Hq:Hkv=1:1,D=128` run passed with 159,877 total
cycles and a 63,334-cycle projection span. These are simulator cycles, not
host wall time. Stage spans overlap across managers and are not additive.

At S=1024, D=128 projection is the largest measured stage: 33,352 of 79,867
cycles for 1:1 and 79,966 of 149,497 for 2:1. The projection-to-attention
handoff adds 8,950 and 8,520 cycles, respectively; about 7,000 cycles of each
handoff lie between the last projection synchronization and descriptor issue.
The optimistic array-only projection floors are 192 and 512 cycles, so they
do not model the dominant transfer, programming, and controller work.

The D=128 weight cache removes repeated HBM weight reads, but the row-major
job order still alternates dimension/input tiles and reprograms weights for
every 16-token group. It records zero array-resident weight reuses in both
D=128 cases. Per manager, array-buffer transfer counters report 7,104 of
9,696 transfer cycles in matrix broadcast for 1:1 and 18,944 of 24,704 for
2:1. Input scatter and output gather account for the rest of those transfer
counters. These are summed transfer-service cycles, not an isolated critical
path contribution, but the counts identify repeated matrix programming as
the next projection-specific target. Reordering or retaining weight tiles
across 16-token groups needs a controlled comparison that preserves the
bounded local GM and output layout.

### 256-row weight reuse and handoff

The next implementation batches D=128 projection in 256-row windows when
`hidden_dim <= 512` and the local-GM scratch range fits. It stages each window's
normalized input once, runs each weight tile across the window's 16-token
groups, and stores FP16 partial outputs and raw rows in disjoint local-GM
regions. The existing two array operand banks and HBM output layout are
unchanged. Larger D=128 shapes keep the previous row-major path.

The post-change rebuild passed numerical, memory-backend, MPI, HBM raw/panel,
and projection-stage checks with four ranks. The baseline column below is the
preceding two-bank/pipelined writeback result; all S=1024 cases use causal RoPE
and FP16 projection accumulation.

| S | Hq:Hkv | D | Baseline cycles | Current cycles | Projection span | Weight programs | Reduction |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1:1 | 64 | 45,814 | 39,310 | 11,376 | 12 | 14.2% |
| 1024 | 1:1 | 128 | 79,867 | 63,471 | 22,941 | 48 | 20.5% |
| 1024 | 2:1 | 64 | 68,633 | 61,346 | 19,682 | 32 | 10.6% |
| 1024 | 2:1 | 128 | 149,497 | 122,813 | 60,950 | 128 | 17.8% |
| 2048 | 1:1 | 128 | 159,877 | 135,438 | 43,387 | 96 | 15.3% |

At S=1024, D=128 array weight programs fell from 768 to 48 for 1:1 and
from 2,048 to 128 for 2:1. The corresponding array-resident reuse counters
are 720 and 1,920. In 2:1, manager 0 spent 1,248 cycles in matrix programming,
4,224 restoring partial outputs, 5,632 scattering inputs, 5,632 reading
outputs, and 34,816 in array compute. These phase counters count residence
in the job state, including issue/wait time; they are not independent hardware
utilization or additive end-to-end costs. Array compute and partial-result
movement are now the main projection-specific costs at that shape.

Removing one guest status print after the projection flag barrier reduced the
per-manager flag-to-descriptor interval from roughly 7,000 cycles to 51 cycles.
The 2:1/D128 projection-to-attention gap fell from 8,520 to 1,533 cycles.
For S=2048/D128, 12,696 cycles remain between the last descriptor and first
Attention job start; worker dispatch occurs within about 20 cycles of the
descriptor, so this is Attention startup rather than projection synchronization.

The 4:2/D128/S=1024 case also passes after fixing an HBM initialization
collision: generic LeNet ready/partial zeroing had overwritten part of the
fourth Q head's projection weights. With intact weight images, the row-major
control run took 396,899 cycles and the reuse path took 338,531 cycles, with
8,192 to 512 total weight programs. Projection HBM layout verification now
checks the initialized weight bytes as well as the produced tensors.

## Causal prefill

The public runner accepts `--causal` for `Sq=Skv`. The QK scheduler omits
fully future 64-key tiles, the online SFU masks the remaining diagonal tiles,
and PV stops each query row after its final causal 256-key window. Distinct
query heads keep independent online softmax state even when they share a KV
head. The verifier compares all FP16 output elements against a causal NumPy
reference; MPI placement and HBM layout checks still run.

```bash
scripts/test_flash_attention.sh --causal --query-length 1024 --kv-length 1024 \
  --num-query-heads 4 --num-kv-heads 2 --head-dim 128
python3 scripts/sweep_attention.py --causal --pairs 32:8 --head-dim 64 \
  --lengths 512,1024,2048,4096 --output-root results/causal_llama32_8_d64
```

The `Hq=32,Hkv=8,D=64` causal runs passed all numerical, MPI, and HBM checks:

| Sq=Skv | SST cycles | QK tiles issued | QK tiles skipped | PV windows issued | PV windows skipped |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 940,552 | 16,896 | 15,872 | 4,608 | 3,584 |
| 4096 | 3,709,114 | 66,560 | 64,512 | 17,408 | 15,360 |

Skipped counts are relative to dense attention at the same shape. RMSNorm and
RoPE are not part of this kernel test.

```text
managers 0..3
   |-- QK/SFU columns 0..1 -> worker cores 4..11   (8 cores)
   `-- PV columns 2..3     -> worker cores 12..19  (8 cores)
```

The tensor shape, 64x64 arrays, QK `2x4` reuse window, PV `1x2` edge reuse
window, online softmax, Score/P FIFOs, 512 KiB V region in each PV core's
existing local SRAM, 256 B/cycle local-SRAM port, HBM/NoC parameters, and SST
completion boundary are shared with the 4:12 baseline.

The 8:8 path adds two PV pipeline contexts.  While context A computes, context
B prefetches the next 128 KiB V window into a different entry of the existing
four-entry V SRAM region.  Within a window, a second 16 KiB payload buffer reads
panel `i+1` from local SRAM while panel `i` is on the arrays.  The request queue
is expanded from 64 to 2048 entries.  The memory-node admission limit remains
128 x 4 KiB credits: testing 256 credits reduced blocked speculative requests
but regressed end-to-end time by 659 cycles because they competed with demand
traffic.  These changes do not alter the 4:12 runner;
`GOLEM_ATTENTION_PV_PANEL_PREFETCH=0` disables local panel lookahead, while the
credit and issue environment variables remain available for ablation runs.

Two additional policies are enabled by default. First, the SFU scheduler gives
priority to row contexts that have already completed more key tiles. This
finishes rows earlier and exposes complete P windows to PV instead of leaving
many rows partially complete. Second, each distinct 128 KiB V entry is fetched
from HBM once by a PV leader, divided into 16 KiB NoC packets, and distributed
through a binary tree to the other PV cores. A prefetch hint is sent as soon as
online softmax starts forming a P window, so V distribution overlaps the
remaining QK/SFU work. Receivers install the assembled entry in the existing
512 KiB per-core V SRAM; this does not add another cache.

The policies can be ablated independently with
`GOLEM_ATTENTION_WORKER_CLUSTER_ROW_PRIORITY=0` and
`GOLEM_ATTENTION_WORKER_CLUSTER_V_BROADCAST=0`.

An optional row-sticky dynamic PV allocator can be enabled with
`GOLEM_ATTENTION_WORKER_CLUSTER_DYNAMIC_PV=1`. Allocation waits until a row's
first complete P window is ready, chooses between the two PV cores attached to
the same manager, and keeps all four windows on that core so online PV
accumulation remains local. It is an experimental ablation and is disabled by
default because it does not improve this regular shape.

Each manager dispatches its 512 flattened GQA rows to two QK workers. Each QK
worker receives 256 rows (one query head for this GQA shape). A completed
64x256 P window is round-robin routed to one of the manager's two PV workers.

Run the functional schedule model:

```bash
baseline/attention_cluster_8qk_8pv/run.sh --no-numerical
```

Run SST for `Hq=4,Hkv=2,Sq=Skv=1024,Dh=128`:

```bash
baseline/attention_cluster_8qk_8pv/run_sst.sh \
  --artifact-root /tmp/attention_cluster_8qk_8pv \
  --query-length 1024 --kv-length 1024 \
  --num-query-heads 4 --num-kv-heads 2 --head-dim 128
```

Run the multi-shape/head sweep (4 GQA pairs x 4 sequence lengths by default):

```bash
python3 scripts/sweep_attention.py
```

The default sweep uses `Hq:Hkv = 1:1, 2:1, 4:2, 4:1` and
`Q/SKV = 512, 1024, 2048, 4096`. Each case gets an independent directory under
`results/sweeps/`; `summary-latest.json` and timestamped CSV/JSON summaries
record status, end-to-end cycles, numerical error, MPI and HBM mapping checks,
stage spans, PV queue wait, and the current 8+8 theoretical resource floor.
Use `--resume` to retain completed cases and
`--dry-run` to inspect the matrix without launching SST. Query lengths must be
divisible by 256 and KV lengths by 128, matching the worker placement checks.
The baseline and sweep use 4 MPI ranks by default; `--mpi-ranks 1` is available
for a single-rank control run. The sweep uses a local-GM queue depth of 256
for every shape (`--local-gm-queue-depth` overrides it). Long KV sequences
need this capacity while V windows are installed and PV panels are read.
Record the queue depth with each cycle result; earlier runs at depth 32 are
not directly comparable.

The 4-rank `Q=K=4096`, `D=128`, local-GM queue-depth-256 sweep completed with:

| Hq:Hkv | End-to-end cycles | 8+8 resource floor | Actual/floor |
| --- | ---: | ---: | ---: |
| 1:1 | 344,392 | 131,072 | 2.628 |
| 2:1 | 617,748 | 262,144 | 2.357 |
| 4:2 | 1,161,522 | 524,288 | 2.215 |
| 4:1 | 1,186,068 | 524,288 | 2.262 |

The complete 16-case sweep (`512, 1024, 2048, 4096`) passed. Its combined
cycle table is in `results/sweeps_4096_q256/summary-latest.csv` and the
per-case stage data is in `summary-latest.json`. At `4096`, the measured
end-to-end time is 2.22-2.63 times the resource floor. For `Hq:Hkv=4:2`,
PV receive-to-start wait averages 204,991 cycles and the PV stage spans
1,157,156 cycles; the resource floor assumes perfect overlap and does not
include this queueing delay.

For the 4:2, 1024 baseline, the report verifies 16 QK jobs, 256 PV windows,
QK cores 4..11, and PV cores 12..19. The terminal summary shows QK, online SFU, and PV cycle spans,
end-to-end cycles, P dispatch/receive/start wait times, and SST/total wall
time. Stage spans overlap and should not be added together. Color is enabled
on interactive terminals; set
`GOLEM_ATTENTION_COLOR=1` to force it or `NO_COLOR=1` to disable it. The full
result is saved as `sst_qk_bridge_result.json` under the artifact root. The
theoretical resource floors and measured comparison are generated by
`compare.py` after both SST runs complete.

The frozen stable result and its complete parameter manifest are archived in
`artifacts/golden/sst_result.json` and `artifacts/golden/manifest.json`.
Generated files under `artifacts/latest` are intentionally ignored.

## Default-shape comparison

Both variants below use the same build, tensor shape, bandwidths, local-SRAM
model, and SST timing boundary.

| split | theoretical resource floor | SST end-to-end | excess over floor |
|---|---:|---:|---:|
| 4 QK/SFU + 12 PV | 65,536 | 73,122 | 7,586 (11.6%) |
| 8 QK/SFU + 8 PV | 32,768 | 47,193 | 14,425 (44.0%) |

Before these pipeline changes, the 8:8 split took 81,397 cycles and was 2,354
cycles slower than 4:12.  Cross-window V prefetch, the prepared context, and
local panel double buffering first reduced it to 72,352 cycles.  Bypassing the
redundant matrix DMA for the P operand then reduces it to 53,434 cycles: another
18,918 cycles (26.15%), or 27,963 cycles (34.35%) against the original 8:8
implementation.  The full panel-pipeline change below reaches 48,756 cycles.
Row-completion priority then reaches 47,100 cycles. Enabling both row priority
and the V broadcast path gives 47,193 cycles while reducing V HBM traffic from
8.375 MiB to 1 MiB. The final default is 25,929 cycles faster than the corrected
4:12 path, a 1.549x speedup.

P is already delivered through the Score/P FIFO and held in the worker's
resident payload.  The old request path nevertheless fetched four 16 KiB P
tiles for every PV window.  Removing those unused requests eliminates 1,024
memory-node requests and 16 MiB of redundant off-chip traffic.  On V-cache-hit
windows, mean PV latency first falls from 1,847.3 to 800 cycles.

The PV path now uses one four-K-tile transaction for the whole 256-row V
window, and its local-SRAM panel lookahead crosses the two output-column reuse
groups.  This reduces a cache-hit PV window from 800 to 598 cycles: 528 compute
cycles plus one 66-cycle startup panel read and fixed completion cost.  The
final run observes 16 V misses and 240 hits; mean latency over all 256 PV
windows is 907.1 cycles.  End-to-end time falls from 53,434 to 48,756 cycles,
another 4,678 cycles (8.75%).  Against the original 81,397-cycle 8:8 path, the
combined reduction is 32,641 cycles (40.10%).

Against the previous 48,756-cycle baseline, the two-policy default saves 1,563
cycles (3.21%). The latency-only optimum is row priority by itself at 47,100
cycles, 93 cycles faster than the bandwidth-saving default. The V tree removes
7.375 MiB (88.1%) of off-chip V reads, but its on-chip distribution consumes
enough NoC capacity to offset 93 cycles of the scheduling gain. This is an
explicit bandwidth/latency tradeoff rather than an unreported regression.

The SST ablation separates the gains: cross-window V prefetch plus the prepared
context reduces 81,397 to 73,602 cycles (7,795 saved); local 16 KiB panel
double-buffering then reduces 73,602 to 72,352 cycles (another 1,250 saved).
Changing the request queue from 64 to 2048 does not change this workload's
cycle count, but prevents the dual-context design from being tied to the old
single-context queue limit.

The theoretical resource floor is still 32,768 cycles, set by the per-QK-core
SFU issue load. With both policies enabled, the measured QK/SFU production span
is 37,487 cycles; the schedule-aware floor including manager dispatch skew is
about 34,433 cycles. After the last QK job finishes, the slowest PV queue needs
another 9,706 cycles to drain. The remaining 14,425-cycle gap over the resource
floor is concentrated in SFU/QK scheduling gaps, NoC V/P contention, and static
PV-core queue imbalance rather than redundant off-chip V reads.

## Dynamic-PV experiment

Dynamic row-to-PV assignment was implemented and measured rather than enabled
on assumption. A global least-active-row allocator took 48,286 cycles: routing
P and V outside the producing manager's local pair added NoC contention and
made it 1,093 cycles slower than static mapping. Constraining allocation to the
same manager removed those long PV transactions and reached 47,890 cycles, but
was still 697 cycles slower than the 47,193-cycle static baseline.

For this shape every row contributes exactly four equal PV windows. Static
round-robin therefore gives every PV core exactly 32 windows. The dynamic
active-row heuristic instead produced 28--36 windows per core and paid an
allocation request/response before the first window. It moved the tail rather
than reducing it. Dynamic mode remains available for irregular or masked
workloads, but static mapping is the correct default for the regular
`Hq=4,Hkv=2,Sq=Skv=1024,Dh=128` comparison.

V prefetch is deliberately tied to the first online-softmax tile of each
window. Restricting hints to only the first window was measured at 51,174
cycles because later V entries returned to the demand path. Sending all hints
on the existing low-priority VN was measured at 47,337 cycles; keeping them on
the normal control VN is faster at 47,193 cycles. Neither rejected policy is
enabled in this baseline.

Reproduce the comparison after running both variants:

```bash
python3 baseline/attention_cluster_8qk_8pv/compare.py \
  --baseline-4qk-12pv /tmp/attention_cluster_4qk_12pv/sst_qk_bridge_result.json \
  --experiment-8qk-8pv /tmp/attention_cluster_8qk_8pv/sst_qk_bridge_result.json
```

Projection cycle breakdown and the opt-in next-pair input prefetch schedule:
[`PROJECTION_SCHEDULE.md`](PROJECTION_SCHEDULE.md), with matched results in
[`projection_schedule_results.json`](projection_schedule_results.json).

Partial transfer bandwidth audit and the opt-in early-restore schedule:
[`PROJECTION_BANDWIDTH.md`](PROJECTION_BANDWIDTH.md),
[`PROJECTION_PARTIAL_RESTORE.md`](PROJECTION_PARTIAL_RESTORE.md), with matched
results in [`projection_partial_restore_results.json`](projection_partial_restore_results.json).

GM queue acceptance, port initiation and response timing are audited in
[`GM_TIMING_CONTRACT.md`](GM_TIMING_CONTRACT.md), with an executable
[`probe_gm_timing.py`](probe_gm_timing.py).

Latest fixed-resource projection input optimization:
[`PROJECTION_INPUT_FLOW.md`](PROJECTION_INPUT_FLOW.md), with the full matched
results in [`projection_input_flow_results.json`](projection_input_flow_results.json).
Recommended opt-in: `GOLEM_PROJECTION_PACKED_NORM_INPUT=1`,
`GOLEM_PROJECTION_INPUT_PREFETCH_BUDGET=0`; the latter can be set to 4 for the
original row-major input plan.

Fixed-resource causal Attention load balancing:
[`CAUSAL_BALANCE.md`](CAUSAL_BALANCE.md). Use
`GOLEM_ATTENTION_CAUSAL_BALANCE=1` to pair complementary query bands through
existing per-head topology descriptors. The implementation preserves tensor
ownership and accumulation order and adds no accelerator resources.
Reproduce and audit with [`run_causal_balance.py`](run_causal_balance.py) and
[`analyze_causal_balance.py`](analyze_causal_balance.py).

RMSNorm/projection overlap analysis using the latest measured batches:
[`NORM_PROJECTION_OVERLAP.md`](NORM_PROJECTION_OVERLAP.md), with
[`norm_projection_overlap_analysis.json`](norm_projection_overlap_analysis.json).
The proposed cross-block prefix schedule is an estimate, not an implemented
or benchmarked optimization.

Fine resource timestamps (GM/DMA/HBM, array transfers/MAC, WCP, SFU,
snapshot ready/consume, worker release/observed/accept) and the distinction
between Batch B diagnostic oracles and the full 2³ software experiment matrix:
[`FINE_RESOURCE_TIMELINE_2026-10-12.md`](FINE_RESOURCE_TIMELINE_2026-10-12.md).
Analyze completed traces with [`analyze_resource_timeline.py`](analyze_resource_timeline.py).
Resource overlap is measured by interval intersection, not by summing phase totals.

Batch B controlled diagnostic oracles and the full fixed-hardware 2³ matrix:
[`BATCH_B_2026-10-12.md`](BATCH_B_2026-10-12.md). Reproduce with
[`run_batch_b_parallel.py`](run_batch_b_parallel.py), validate independent shards
with [`summarize_batch_b.py`](summarize_batch_b.py), and inspect Projection-only
queue/service times with [`analyze_batch_b_waits.py`](analyze_batch_b_waits.py).
Diagnostic oracles are scoped interventions; their savings are not additive
and are not deployable optimization results.

2026-10-12：新增 [Projection 逐 tile 因果分析](PROJECTION_CAUSAL_ANALYSIS_2026-10-12.md)。`GOLEM_PROJECTION_CAUSAL_TRACE=1` 与资源追踪一起记录完整 tile 标识，MPI 合并后保存在 `.causal.jsonl`；分析器校验 tag/地址/时间戳关联并闭合 MAC 间隙。正式数据在 `results/projection_causal_20261012/summary.json`，包含两个规模的对照及 input-budget / weight-lookahead 单变量试验。

## 单层 decoder prefill 检查点（2026-10-14）

[接入说明和容量分析](DECODER_LAYER_2026-10-14.md)记录了 Wo、第一残差、
第二次 RMSNorm、SwiGLU FFN 和第二残差的独立检查点。使用
`run_decoder_layer.py --checkpoint wo|residual|ffn|layer --shape small|formal`
验证 S=1024 的单层；默认 Attention baseline guest 不受这些 opt-in 变体影响。
本轮保持每 worker 2 MiB GM 和 4×128 MiB 数据 HBM，down 支持 8192→2048 分块。
实测输出、逐阶段误差和 DMA ACK 时间边界见
`results/decoder_layer_20261014/summary.md`。

两层 Decoder 重复执行、统一内存布局和 completion epoch：
[DECODER_STACK_2026-10-14.md](DECODER_STACK_2026-10-14.md)。

参数化多层、变长前缀、稠密权重与 FP32 误差测量：
[PREFILL_GENERALIZATION_2026-10-16.md](PREFILL_GENERALIZATION_2026-10-16.md)。
`run_prefill_validation.py` 按 local / formal / target 分级验证，支持重用已验收结果。
变长 S 当前通过软件补齐到 1024 的倍数，所有补齐工作计入 cycle；GM 保持每 worker 2 MiB。

Add/Multiply/SiLU 已接入实际 SFU 管线预占、有限 FIFO 和反压；SiLU 与
Attention 使用同一个 EXP/倒数资源对象。精度契约、竞争测试及相同输入的
两层周期对比见 [SFU_VECTOR_PIPELINE_2026-10-14.md](SFU_VECTOR_PIPELINE_2026-10-14.md)。

两层输出已追加独立 gamma 的 final RMSNorm，保持 FP16 I/O 和 FP32 内部计算；检查独立标志清零、epoch=3、完整输出及 completion DMA ACK。
复现和计时边界见 [FINAL_RMSNORM_2026-10-14.md](FINAL_RMSNORM_2026-10-14.md)。

最后一个 token 的 LM head 支持 M=1、8192 列词表分块及尾块，LM head runner 默认每 HBM 节点容量 2 GiB。
数据路径和验证见 [LM_HEAD_2026-10-14.md](LM_HEAD_2026-10-14.md)。

已接入 worker 定时 Embedding gather 和 host greedy token 选择；4/8 MPI rank 的相同输入对比见
[EMBEDDING_TOKEN_MPI_2026-10-14.md](EMBEDDING_TOKEN_MPI_2026-10-14.md)。

后续优化采用 [局部 SST 筛选与端到端验收](LOCAL_TEST_WORKFLOW_2026-10-14.md)。
`run_decoder_micro.py` 从预置 Norm2 开始，直接验证实际 FFN 数据通路，支持 1/4/16 worker、正式 K/N 宽度、可选 down 和配对收益报告。

独立 all-GEMM reuse-window 基线及后续 GM 软件驻留优化见
[Software Input Residency](SOFTWARE_INPUT_RESIDENCY_2026-10-22.md)。
`--input-residency` 在已有 GM 中保留 Norm1 的原始输入，供残差 Add 复用；
保留原始硬件资源及输出 ACK 协议，配置约束与正式端到端结果见该报告。

2026-10-24 的默认配置和实测周期由
[SELECTED_PREFILL_2026-10-24.json](SELECTED_PREFILL_2026-10-24.json) 保存，
`run_selected_prefill.py --artifact-root <目录>` 在运行前重建并核对已测 guest 哈希。
软件 tail 实验使用独立 guest：在相同 all-GEMM 配置上，
`--tail-schedule-experiment` 为关闭对照，追加 `--final-norm-lm-stream`
启用 producer 经现有 remote_store 通知其余 worker，继续追加
`--final-norm-producer-first` 独立测试 producer 的整个 64 行 Norm 作业优先。
实验要求 `--eager-final-norm`，只支持 padded sequence 不超过 1024；不减少张量 HBM IO。
结果、采用门槛及默认恢复证据见 [当日进展 P7](PROGRESS_2026-10-24.md)。

2026-10-26 完成 [FFN 关键路径预算与需求优先实验](FFN_CRITICAL_PATH_2026-10-25.md)：
正式 16-worker micro 配对均为 1,967,583.5 cycles，实际资源 trace 相同，候选保持关闭。
新库默认路径完整端到端精确复现 6,084,167.75 cycles，输出 FP16 位差为零。
当前复现入口使用 [SELECTED_PREFILL_2026-10-26.json](SELECTED_PREFILL_2026-10-26.json)，
启动前核对 guest 和 libgolem 哈希；历史配置保留原实测库哈希。
