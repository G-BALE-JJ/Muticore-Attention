# GPU-Competitive FlashAttention Plan

## Goal

For FP32, non-causal `B=1,H=1,D=128` Attention, reduce SST accelerator
completion latency below the RTX 5060 Scope A measurements. MPI is used only
to reduce simulator wall time. Multi-head Attention and MoE remain deferred.

## Current Baseline

The verified cluster architecture has 20 cores: four control-only managers and
16 workers. Each worker has 64 physical 64x64 arrays with 64 CUs per array,
two operand banks, and a default 16-QK + 48-PV ownership split. Br16/Bc32 is
the measured default.

| Profile | SST cycles | RTX 5060 Scope A | Ratio | Gate |
|---|---:|---:|---:|---|
| E3, Q=K=1024 | 330,125 | 97,568 | 3.384x | FAIL |
| E4, Q=K=2048 | 1,007,628 | 345,984 | 2.912x | FAIL |

All frozen R6 numerical and lifecycle cases pass. Independent review is closed
at zero Critical and zero Important findings. See
`attention_cluster/R1_R6_OPTIMIZATION_RESULTS.md`.

## Completed

- [complete] Local SST/Golem and RISC-V guest build/install workflow.
- [complete] Single-head online FlashAttention numerical and lifecycle baseline.
- [complete] Query-group MPI 1/2/4-rank placement and determinism validation.
- [complete] Direct score/P storage, K/V residency, resident O, bounded tagged
  scheduling, cancellation, backpressure, and interval-union observability.
- [complete] H64 alignment to 64 physical 64x64 arrays and 64 CUs per array.
- [complete] R1-R6 ownership sweep, Bc sweep, O FMA, supply-path profiling,
  complete verification matrix, and independent review.
- [complete] R7-A through R7-C controlled scheduler and K/V buffer experiments;
  rejected changes were fully reverted and the R6 baseline was reproduced.
- [complete] Remove regenerable experiment artifacts and superseded 8+8/H64
  transition documents; retain only current summaries and final verification JSON.

## Current Work

- [pending] R7-D: add explicit score/SFU/PV operand-bank consumer lifetime,
  generation, cancellation, and release accounting before attempting any
  further QK multi-producer overlap.
- [pending] After R7-D, evaluate a bounded four-worker K/V distribution and
  reassembly engine with four 64 B/cycle destinations and two resident contexts.
- [pending] Re-run Q256/K128, Q256/K1024, E3, E4, pressure, cancellation, MPI,
  review, and GPU comparison gates after the next accepted architecture change.
- [pending] Pass E3 `<97,568` and E4 `<345,984` cycles before multi-head work.

## Repository Publication

- [complete] P1. Remove regenerable Attention run directories and obsolete docs.
- [complete] P2. Rewrite project status around the current 16-QK + 48-PV R6
  baseline and R7-D next step.
- [complete] P3. Verify retained hashes, documentation, tests, build, and size.
- [complete] P4. Commit the curated cleanup and status update for publication on
  the `softmax-update` GitHub branch.

## Rejected R7 Experiments

- Moving the ahead scheduler before SFU issue was a measured no-op.
- A partial score-bank retain/release change failed E3 lifecycle and was reverted.
- Bank-aware multi-producer QK caused 9,404 E3 mismatches and was reverted.
- Three K/V buffers passed correctness but regressed E3 from 330,125 to 330,300
  cycles and increased K/V wait, so the two-buffer model remains.

## Next Step

Proceed with R7-D after publication; the safe baseline remains R6.
