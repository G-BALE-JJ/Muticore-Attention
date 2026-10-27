# 2D Reuse Three-Stage Plan

## Control Knobs

Use the existing reuse knobs for all stages:

```text
GOLEM_B_REUSE_M_TILES = reuse height in M
GOLEM_A_REUSE_N_TILES = reuse width in N
```

Modes:

```text
M=1,N=1 -> baseline
M=1,N>1 -> existing A reuse across N
M>1,N=1 -> existing B reuse across M
M>1,N>1 -> 2D reuse group
```

No new `GOLEM_2D_*` knobs are needed. No `K_WINDOW_TILES` knob is added; the later window size is derived from slots and prefetch depth.

## Stage 1: 2x2 Full-K Correctness

Goal: make the 2D layout and WCP execution correct before chasing the larger 4x4 benefit.

Default target:

```text
GOLEM_GEMM_M/N/K = 512/512/512
block_m/n/k = 64/64/128
GOLEM_B_REUSE_M_TILES=2
GOLEM_A_REUSE_N_TILES=2
GOLEM_DMA_SLOT_COUNT=8
GOLEM_GLOBAL_STRIDE_KB>=1024
```

Derived layout:

```text
M tiles = 8
N tiles = 8
K tiles = 4
macro groups = (8/2) * (8/2) = 16
active workers = 16
```

Local full-K slots:

```text
mat slots = M_REUSE * K_TILES = 2 * 4 = 8
vec slots = N_REUSE * K_TILES = 2 * 4 = 8
```

Stage 1 execution policy:

```text
1. Prefetch all A(m_offset,k) and B(n_offset,k) for the macro group.
2. Compute all C(m_offset,n_offset) tiles using full-K resident A/B.
3. Write each C tile once after full K accumulation.
```

Success criteria:

```text
default A=1/B=1 remains VERIFY-C PASS
existing A4 remains VERIFY-C PASS
existing B4 remains VERIFY-C PASS
2x2 full-K VERIFY-C PASS
A/B read requests drop versus one-dimensional reuse
```

## Stage 2: 4x4 Full-K Benefit

Goal: validate the upper-bound traffic reduction from 2D full-K reuse.

Targets:

```text
1024x1024x512:
  GOLEM_B_REUSE_M_TILES=4
  GOLEM_A_REUSE_N_TILES=4
  GOLEM_DMA_SLOT_COUNT>=16
  GOLEM_GLOBAL_STRIDE_KB>=2048

1024x1024x1024:
  GOLEM_B_REUSE_M_TILES=4
  GOLEM_A_REUSE_N_TILES=4
  GOLEM_DMA_SLOT_COUNT>=32
  GOLEM_GLOBAL_STRIDE_KB>=4096
```

Expected full-K read reduction:

```text
2x2 -> 50% A/B read reduction versus baseline
4x4 -> 75% A/B read reduction versus baseline
```

Stage 2 keeps strict full-prefetch-before-compute. If traffic drops but total cycles are limited by cold-start wait, overlap is handled in Stage 3 or later progressive scheduling.

## Stage 3: Slot-Driven K Window

Goal: maintain hardware constraints for large K and small block_k without adding a manual window knob.

Derived resident window:

```text
resident_k_tiles = min(
  GOLEM_DMA_WINDOW_K_TILES,
  local_slot_count / ((GOLEM_WCP_PREFETCH_WINDOWS + 1) * GOLEM_B_REUSE_M_TILES),
  local_slot_count / ((GOLEM_WCP_PREFETCH_WINDOWS + 1) * GOLEM_A_REUSE_N_TILES)
)
```

When `K_TILES > resident_k_tiles`, split K into windows.

Required additional state:

```text
partial C scratch = M_REUSE * N_REUSE output tiles
```

Execution policy:

```text
clear partial C tiles
for each K window:
  prefetch resident A/B window
  compute all C tiles into partials
write final C tiles once
```

Stage 3 is intentionally after full-K stages because it adds partial C traffic and more correctness surface area.

## Stage 1 Execution Notes

Implemented with the existing knobs only:

```text
GOLEM_B_REUSE_M_TILES=2
GOLEM_A_REUSE_N_TILES=2
GOLEM_DMA_SLOT_COUNT=8
GOLEM_GLOBAL_STRIDE_KB=1024
```

Validation runs:

```text
run_20260427_2d_stage1_default_final: VERIFY-C PASS
run_20260427_2d_stage1_2x2_fix3: VERIFY-C PASS
```

Debug iterations:

```text
run_20260427_2d_stage1_2x2:
  VERIFY-C FAIL, C unpacked as all zero.
  GM stats showed only 3 C writebacks per 2x2 group.
  Root cause: when advancing reuseMIndex, reuseNIndex was not reset to 0.

run_20260427_2d_stage1_2x2_fix1:
  VERIFY-C FAIL, C still unpacked as all zero.
  MVM dump showed nonzero matrix/vector inputs and nonzero array outputs.
  GM verbose run showed C writeback went to 0x630000-style offsets, while Python layout expected 0x800000.
  Root cause: runtime gemm_off_out_base(cfg) still used B-reuse one-dimensional vec_slots=1 logic.

run_20260427_2d_stage1_2x2_fix3:
  VERIFY-C PASS after using vec_slots=A_REUSE_N_TILES when A reuse is enabled.
```

Benefit versus final default baseline:

```text
total_cycles:              27409.8125 -> 15407.8750  (-43.79%)
avg_throughput_ops/cycle:   9793.4072 -> 17421.9648 (+77.90%)
array_utilization_pct:         7.4718 ->    13.2919 (+77.90%)
prefetch_wait_time:        13657.0625 -> 10521.1875 (-22.96%)
writeback_wait_time:        7483.8750 ->  2737.6875 (-63.42%)
```

Logical A/B read traffic expectation:

```text
baseline: 64 C tiles * 4 K tiles * (1 A + 1 B) = 512 logical tile reads
2x2:      16 macro groups * 4 K tiles * (2 A + 2 B) = 256 logical tile reads
expected read reduction = 50%
```

The Stage 1 total-cycle reduction is slightly below the ideal read reduction because the strict full-K implementation waits for the whole 2x2 group to become resident before compute and does not yet pipeline K windows or overlap macro groups.

## Stage 2 Execution Notes

Validated the full-worker 4x4 case at `1024x1024x512`:

```text
baseline run: run_20260427_2d_stage2_baseline_1024x1024x512
4x4 run:      run_20260427_2d_stage2_4x4_1024x1024x512

GOLEM_GEMM_M/N/K=1024/1024/512
GOLEM_B_REUSE_M_TILES=4
GOLEM_A_REUSE_N_TILES=4
GOLEM_DMA_SLOT_COUNT=16
GOLEM_GLOBAL_STRIDE_KB=2048
```

Validation:

```text
baseline: VERIFY-C PASS
4x4:      VERIFY-C PASS
```

Benefit:

```text
total_cycles:              106539.5625 -> 38449.2500  (-63.91%, 2.77x speedup)
avg_throughput_ops/cycle:   10078.3390 -> 27926.2098 (+177.09%)
array_utilization_pct:          7.6892 ->    21.3060 (+177.09%)
prefetch_wait_time:         60646.4375 -> 22478.3750 (-62.94%)
writeback_wait_time:        37412.1250 ->  7377.8750 (-80.28%)
```

Logical A/B read traffic expectation:

```text
baseline: 256 C tiles * 4 K tiles * (1 A + 1 B) = 2048 logical tile reads
4x4:       16 groups  * 4 K tiles * (4 A + 4 B) = 512 logical tile reads
expected read reduction = 75%
```

Observed scheduler read completions in logs are close to this ratio:

```text
baseline TRACE_REQ_DONE slot0+slot1 ~= 2007
4x4 TRACE_REQ_DONE slot0+slot1      ~= 489
observed reduction ~= 75.6%
```

The total-cycle reduction is lower than the ideal 75% read-traffic reduction because Stage 2 still uses a strict full-K resident barrier. Even so, the full-worker 4x4 case confirms the expected Stage 2 benefit shape: large prefetch-wait reduction, significantly higher array utilization, and a 2.77x end-to-end speedup over 1x1 at the same GEMM size.
