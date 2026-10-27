# B Reuse Across M Tiles Plan

## Goal

Reduce repeated B-tile reads across adjacent M tiles while preserving the current C-stationary behavior:

```text
C partial sums stay in array output state.
Each C tile is written once.
B tiles stay in worker local GM vec slots and are reused across M tiles.
A tiles remain per-M-tile reads in the first implementation.
```

## First Implementation Scope

The first implementation is intentionally symmetric to A reuse and mutually exclusive with it:

```text
Path: WCP runtime path only
Default target: k128 baseline, GEMM_K_TILES <= local_slot_count
Reuse factor: GOLEM_B_REUSE_M_TILES, default 1, test value 4
C accumulation: one C tile at a time, no C partial spill
B reuse granularity: one macro task = (m_group, n_tile)
Mutual exclusion: do not enable A reuse and B reuse at the same time
```

For the default 512x512, block 64x64x128 case:

```text
M tiles = 8
N tiles = 8
K tiles = 4
reuse_m = 4
macro tasks = ceil(8 / 4) * 8 = 16
```

This keeps all 16 workers occupied while reducing B reads by 4x within each M group.

## Macro Task Layout

With `reuse_m = R`:

```text
m_groups = ceil(m_tiles / R)
macro_id = m_group * n_tiles + n_tile
m_begin = m_group * R
m_count = min(R, m_tiles - m_begin)
```

The per-node slot becomes a macro slot, not an original C tile slot.

Offsets:

```text
B offset = off_vec + macro_slot * k_tiles * block_n * vec_stride
A offset = off_mat + (macro_slot * R + m_offset) * k_tiles * mat_stride
C offset = off_out + (macro_slot * R + m_offset) * out_stride
```

For `m_offset=0`, WCP reads A+B. For later M offsets in the same macro task, WCP reads only A and reuses B from local vec slots.

## Required Code Changes

```text
tests/small/mvm_noc_int_array/pipeline_config.h
tests/small/mvm_noc_int_array/gemm_matmul_op.h
tests/small/mvm_noc_int_array/gemm_matmul_op_ctrl.h
```

## Expected Benefit

For `R=4`, four C tiles originally read:

```text
4A + 4B = 8 units
```

After B reuse:

```text
4A + 1B = 5 units
```

Expected read traffic reduction:

```text
37.5%
```

Primary success signals:

```text
VERIFY-C PASS
vec read request/event count decreases
memory_service pressure decreases
prefetch_wait_time decreases
array_utilization increases
```

## Guardrails

If `reuse_m > 1` and `k_tiles > local_slot_count`, fail fast for this first implementation. That prevents vec slots from being overwritten before reuse completes.

If both `GOLEM_A_REUSE_N_TILES > 1` and `GOLEM_B_REUSE_M_TILES > 1`, fail fast. Supporting both together is a two-dimensional tile group problem and needs a separate scheduling design.
