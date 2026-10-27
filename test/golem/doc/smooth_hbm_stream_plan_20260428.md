# Smooth HBM Stream Plan 20260428

## Goal

Reduce synchronized A/B read bursts from the four workers mapped to the same HBM data node, while keeping the 600GB/s NoC/HBM path busy enough to approach the local theoretical compute feed rate.

## Local Model

One HBM data node serves four GEMM workers. For the current 4x4 reuse macro:

- `block_m = 64`, `block_n = 64`, `block_k = 128`
- `array_input_size = 64`, `array_output_size = 64`
- one micro-step costs `64 + pipeline_depth(2) = 66 cycles`
- one `block_k=128` k-tile costs two micro-steps, or `132 cycles` per C tile
- one worker has 16 C tiles per k-tile, so compute per k-tile is `16 * 132 = 2112 cycles`
- A/B bytes per worker per k-tile are `4 * 32KiB + 4 * 32KiB = 256KiB`
- four workers need `1MiB / 2112 cycles = 497 B/cycle`

With `GOLEM_NOC_LINK_BW=600GB/s` at 1GHz, the local path is about `600 B/cycle`. This is enough for A/B reads in the average case, but only if requests are smoothed. Burst submission still creates scheduler, NoC, and HBM tail latency.

## Current Problem

The WCP/scheduler path tends to submit a whole window from multiple workers at similar times. Async writeback removed C-store wait, but it also removed natural throttling, causing A/B requests to concentrate earlier. The visible symptom is high `prefetch_wait`, `submit_to_issue`, and `memory_service_p95`.

## Implementation Plan

### Phase 1: Work-Conserving Manager Smoothing

Implement this first because it is low risk and directly targets burst issue at the HBM-node manager.

- Add per-target-node token buckets in `RequestSchedulerEndpoint`.
- Use byte tokens, not request-count tokens.
- Add a burst cap to prevent a node from saving too many tokens and issuing a large burst later.
- Issue with per-node worker round-robin so one worker cannot consume all available tokens while others are queued.
- Keep the scheduler work-conserving: if a chosen worker has no issueable request, scan other workers.
- Keep existing worker and node outstanding credits as safety limits.

Initial default parameters:

- `GOLEM_SCHED_SMOOTH_ENABLE=1`
- `GOLEM_SCHED_READ_BW_BYTES_PER_CYCLE=600`
- `GOLEM_SCHED_WRITE_BW_BYTES_PER_CYCLE=40`
- `GOLEM_SCHED_TOKEN_BURST_BYTES=65536`

The scheduler token bucket currently gates request-scheduler A/B reads only. Async C writeback bypasses this scheduler path, so the default read budget uses the full 600 B/cycle NoC setting. A separate write budget is documented for a future phase that moves C writeback under the same smoother.

### Phase 2: Slice Large Transfers

Only add this if Phase 1 leaves `submit_to_issue_p95` or NoC p99 high.

- Split `32KiB` A/B tile transfers into `4KiB` or `8KiB` slices.
- Mark a tile ready only after all slices complete.
- This gives the token bucket finer granularity and enables true interleaving across workers.

### Phase 3: WCP Watermark Submit

Only add this after Phase 1/2 data shows manager-side smoothing is not enough.

- Replace whole-window prefetch burst with low/high watermark submission.
- Keep one or two k-tiles inflight per worker.
- Submit more only when ready+inflight drops below the low watermark.

## Validation

Run order:

1. `1024x1024x1024` correctness and regression.
2. `2048x2048x1024` correctness and metrics.

Pass/fail metrics:

- `VERIFY-C PASS` must hold.
- `prefetch_wait_time` should drop versus `run_20260428_prog_asyncwb_2048_verify`.
- `submit_to_issue_{mat,vec}_mean/p95` should drop.
- `memory_service_p95` should drop or stay flat.
- NoC `p99` and `xbar_stalls` should drop or stay flat.
- `writeback_wait_time` should remain low; async writeback must not regress.

## Expected Outcome

At 600GB/s, one HBM-node group has enough average bandwidth to feed four workers' A/B reads. The first target is not perfect compute dominance, but to move from burst-limited behavior toward the local average-bandwidth bound by reducing scheduler and memory tail latency.

## Phase 1 Result And Direction Change

The manager-side token bucket reduced NoC burst symptoms but did not improve the compute critical path at 600GB/s:

- 2048 600GB/s no smoothing: `total_cycles=217442.125`, `array_utilization_pct=30.139514`, `noc_p99=414ns`, `xbar_stalls=504203`
- 2048 600GB/s manager smoothing: `total_cycles=218828.5`, `array_utilization_pct=29.948567`, `noc_p99=190ns`, `xbar_stalls=338396`

This means the network became smoother, but critical A/B readiness did not improve. The smoother was too far downstream: the manager sees requests but does not know which request is on the compute critical path.

## WCP Streaming Direction

The GPU-like mechanism should live in WCP:

- Keep the manager fast and mostly work-conserving.
- Make WCP produce A/B requests at a steady k-tile cadence.
- Keep the next k-tile in flight while the current k-tile is consumed by the 4x4 reuse macro.
- Later, expand the ready work pool from the current `(reuseM,reuseN)` order to a ready queue over all 16 C tiles in the macro.

The first implementation step was intentionally small:

- Add `GOLEM_WCP_STREAM_KTILE_WINDOWS=1`.
- When enabled, force the 2D reuse resident K-window to one k-tile.
- This changes the WCP pattern from `submit k[0..1], compute, submit k[2..3]` to `submit k0, compute k0 while k1 is in flight, compute k1 while k2 is in flight`.
- Disable manager token smoothing by default because it improved NoC p99 but slightly slowed the compute path.

The result showed that `k-window=1` made the compute cover too short. A single 4x4 k-tile gives only about `16 * 132 = 2112 cycles`, while request service is roughly `3000+ cycles`. Therefore the final direction is not to shrink the compute window. The compute window should stay large, while the request submission granularity becomes small.

Current default direction:

- `GOLEM_WCP_STREAM_KTILE_WINDOWS=0`
- `GOLEM_DMA_WINDOW_K_TILES=4`
- active compute window remains four k-tiles
- WCP submits that window as four independent one-k-tile scheduler transactions

This gives the active window about `4 * 16 * 132 = 8448 cycles` of compute cover while avoiding a single large scheduler transaction for the full k-window.

## Phase 2: Ready Reuse Queue

After k-tile streaming, the next source of bubbles is the fixed 4x4 macro traversal order. If the current `(reuseM,reuseN)` C tile waits for A/B readiness, another C tile in the same k-window may already be ready. GPU schedulers hide memory latency by switching to another ready warp; the WCP equivalent is to switch to another ready C tile in the active reuse macro.

Implementation:

- Add `GOLEM_WCP_READY_REUSE_QUEUE=1`.
- Track `windowReuseDone_` for the active k-window, one bit per `(reuseM,reuseN)` C tile.
- After a C tile finishes the active k-window, save partial C or issue final C writeback, mark that reuse tile done, then choose the next unfinished reuse tile with ready A/B.
- If no unfinished reuse tile is ready, choose the first unfinished tile as a fallback and wait there.
- Keep per-C-tile K accumulation ordered. This does not compute k+1 before k for the same C tile.

This is still conservative: it only reorders C tiles within one active k-window. It does not reorder K for any C tile and does not change the final output layout.
