# A Reuse Across N Tiles Plan

## Goal

Reduce repeated A-tile reads across adjacent N tiles while preserving the current C-stationary behavior:

```text
C partial sums stay in array output state.
Each C tile is written once.
A tiles stay in worker local GM mat slots and are reused across N tiles.
B tiles remain per-N-tile reads in the first implementation.
```

## First Implementation Scope

The first implementation is intentionally narrow:

```text
Path: WCP runtime path only
Default target: k128 baseline, GEMM_K_TILES <= local_slot_count
Reuse factor: GOLEM_A_REUSE_N_TILES, default 1, test value 4
C accumulation: one C tile at a time, no C partial spill
A reuse granularity: one macro task = (m_tile, n_group)
```

For the default 512x512, block 64x64x128 case:

```text
M tiles = 8
N tiles = 8
K tiles = 4
reuse_n = 4
macro tasks = 8 * ceil(8 / 4) = 16
```

This keeps all 16 workers occupied while reducing A reads by 4x within each N group.

## Current Code Shape

Current task mapping:

```text
task_id = m_tile * GEMM_N_TILES + n_tile
```

Current HBM layout stores A per C tile task:

```text
A offset = off_mat + task_slot * k_tiles * mat_stride
B offset = off_vec + task_slot * k_tiles * block_n * vec_stride
C offset = off_out + task_slot * out_stride
```

Current WCP deriveTask mirrors that layout, so adjacent N tiles reread the same A.

## New Macro Task Layout

With `reuse_n = R`:

```text
n_groups = ceil(n_tiles / R)
macro_id = m_tile * n_groups + n_group
n_begin = n_group * R
n_count = min(R, n_tiles - n_begin)
```

The per-node slot becomes a macro slot, not an original C tile slot.

New offsets:

```text
A offset = off_mat + macro_slot * k_tiles * mat_stride
B offset = off_vec + (macro_slot * R + n_offset) * k_tiles * block_n * vec_stride
C offset = off_out + (macro_slot * R + n_offset) * out_stride
```

For `n_offset=0`, WCP reads A+B. For later N offsets in the same macro task, WCP reads only B and reuses A from local mat slots.

## Required Code Changes

```text
tests/small/mvm_noc_int_array/pipeline_config.h
tests/small/mvm_noc_int_array/gemm_matmul_op.h
tests/small/mvm_noc_int_array/gemm_matmul_op_ctrl.h
golem/workercmdproc/workercmdproc.h
golem/requestscheduler/requestscheduler.h
golem/requestscheduler/requestscheduler.cc
tests/tools/gen_hbm_init.py
tests/tools/unpack_c_from_hbm.py
```

## Expected Benefit

For `R=4`, four C tiles originally read:

```text
4A + 4B = 8 units
```

After A reuse:

```text
1A + 4B = 5 units
```

Expected read traffic reduction:

```text
37.5%
```

Primary success signals:

```text
VERIFY-C PASS
mat read request/event count decreases
memory_service pressure decreases
prefetch_wait_time decreases
array_utilization increases
```

## Guardrails

If `reuse_n > 1` and `k_tiles > local_slot_count`, fail fast for this first implementation. That prevents mat slots from being overwritten before reuse completes.
