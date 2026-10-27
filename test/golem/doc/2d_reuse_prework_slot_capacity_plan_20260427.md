# 2D Reuse Prework: Local Slots and Capacity

## Goal

Prepare the runtime for future 2D full-K reuse without implementing the 2D dataflow yet.

The prework makes local mat/vec slots configurable and verifies the capacity requirements that 2D full-K will need.

## Scope

```text
Use existing GOLEM_DMA_SLOT_COUNT as the WCP local slot count.
Keep current A-reuse and B-reuse behavior unchanged.
Keep full-K one-dimensional reuse guards, but use slot_count instead of hard-coded 4.
Do not add K_WINDOW_TILES.
Do not implement 2D reuse in this stage.
```

## Local GM Layout

The local GM layout changes from a fixed 4-slot layout to a slot-count-driven layout:

```text
mat_base = LOCAL_DATA_BASE
vec_base = mat_base + slot_count * mat_slot_bytes
out      = vec_base + slot_count * vec_slot_bytes
accum    = out + out_scratch_bytes
```

Existing names such as `mat_ping`, `mat_pong`, `mat_slot2`, `mat_slot3`, `vec_ping`, `vec_pong`, `vec_slot2`, and `vec_slot3` remain as aliases for slot0-slot3 to avoid disrupting old code paths.

## Required Capacity

For full-K 2D reuse:

```text
mat_slots_needed = reuse_m * k_tiles
vec_slots_needed = reuse_n * k_tiles
```

The current prework only makes these slot counts possible; later 2D stages will add guards for the actual 2D reuse factors.

## Example Capacity Checks

Assume fp32, block_m=64, block_n=64, block_k=128.

### 2x2 full-K, 512^3

```text
K tiles = 4
mat slots = 2 * 4 = 8
vec slots = 2 * 4 = 8
mat bytes = 8 * 32KB = 256KB
vec bytes = 8 * 32KB = 256KB
recommended GOLEM_GLOBAL_STRIDE_KB = 1024
HBM node size 128MB is sufficient
```

### 4x4 full-K, 1024x1024x512

```text
K tiles = 4
mat slots = 4 * 4 = 16
vec slots = 4 * 4 = 16
mat bytes = 16 * 32KB = 512KB
vec bytes = 16 * 32KB = 512KB
recommended GOLEM_GLOBAL_STRIDE_KB = 2048
HBM node size 128MB is sufficient
```

### 4x4 full-K, 1024^3

```text
K tiles = 8
mat slots = 4 * 8 = 32
vec slots = 4 * 8 = 32
mat bytes = 32 * 32KB = 1024KB
vec bytes = 32 * 32KB = 1024KB
recommended GOLEM_GLOBAL_STRIDE_KB = 4096
HBM node size 128MB is sufficient
```

The HBM estimate remains well below 128MB per data node for these cases because each worker owns one macro group at 1024^3/4x4 and stores roughly 2MB of A/B plus 1MB of C outputs per worker-local macro group layout.

## Verification Plan

```text
1. Build and install golem element library.
2. Run default A=1/B=1 regression with GOLEM_DMA_SLOT_COUNT=4.
3. Run A-reuse N4 with slot_count=4.
4. Run B-reuse M4 with slot_count=4.
5. Run default A=1/B=1 with GOLEM_DMA_SLOT_COUNT=8 to verify expanded local layout.
```

Success criteria:

```text
VERIFY-C PASS for all regressions.
No local GM stride overlap error.
Generated configuration prints the expanded slot count and stride.
```

## Verification Results

Completed on 2026-04-27.

```text
make: PASS
make install: PASS
```

Regression runs:

```text
run_20260427_slotpre_default
  GOLEM_DMA_SLOT_COUNT=4
  GOLEM_A_REUSE_N_TILES=1
  GOLEM_B_REUSE_M_TILES=1
  VERIFY-C PASS

run_20260427_slotpre_areuse4_retry
  GOLEM_DMA_SLOT_COUNT=4
  GOLEM_A_REUSE_N_TILES=4
  GOLEM_B_REUSE_M_TILES=1
  VERIFY-C PASS

run_20260427_slotpre_breuse4
  GOLEM_DMA_SLOT_COUNT=4
  GOLEM_A_REUSE_N_TILES=1
  GOLEM_B_REUSE_M_TILES=4
  VERIFY-C PASS

run_20260427_slotpre_default_slot8
  GOLEM_DMA_SLOT_COUNT=8
  GOLEM_A_REUSE_N_TILES=1
  GOLEM_B_REUSE_M_TILES=1
  GOLEM_GLOBAL_STRIDE_BYTES auto-expanded from 524288 to 549440
  VERIFY-C PASS
```

One A-reuse run was first launched concurrently with a B-reuse run and failed in `memHierarchy::BackingMMAP` with a bus error before simulation startup. The concurrent runs shared `artifacts/hbm`, so the likely cause was backing-file truncate/mmap contention. The same A-reuse configuration passed when rerun serially.

## Capacity Conclusion

Current default HBM/node and local GM settings are sufficient for the planned stages with explicit slot/stride settings:

```text
Stage 1, 2x2 full-K, 512^3:
  GOLEM_DMA_SLOT_COUNT >= 8
  recommended GOLEM_GLOBAL_STRIDE_KB >= 1024
  128MB HBM node size is sufficient

Stage 2, 4x4 full-K, 1024x1024x512:
  GOLEM_DMA_SLOT_COUNT >= 16
  recommended GOLEM_GLOBAL_STRIDE_KB >= 2048
  128MB HBM node size is sufficient

Stage 2, 4x4 full-K, 1024^3:
  GOLEM_DMA_SLOT_COUNT >= 32
  recommended GOLEM_GLOBAL_STRIDE_KB >= 4096
  128MB HBM node size is sufficient
```

The prework now supports slot counts above 4 in the local layout and WCP buffering. The 2D stages still need their own mat/vec slot allocation policy and full-K reuse guards.
