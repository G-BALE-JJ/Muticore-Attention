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
| E3, Q=K=1024 | 148,146 | 97,568 | 1.518x | FAIL |
| E4, Q=K=2048 | 517,020 | 345,984 | 1.494x | FAIL |

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
- [complete] Confirmed the two-versus-three K/V buffer A/B on both E3 and E4.
  Three buffers passed correctness at 330,300/997,866 cycles: E3 regressed
  0.05%, while E4 improved 0.97%, so the default remains two buffers.
- [complete] Remove regenerable experiment artifacts and superseded 8+8/H64
  transition documents; retain only current summaries and final verification JSON.
- [complete] R8 manager-level K/V distribution and its feature-off, multi-tile,
  E3, E4, pressure, cancellation-contract, and MPI validation matrix.

## Current Work

- [complete] R11-A: audit and complete tagged two-bank QK/score/PV operand
  lifetimes, including generation-safe cancellation and exact lease release.
- [complete] R11-B: overlap next grouped-query/tile QK matrix and input
  programming with current SFU/PV work, with an independent feature-off path.
- [complete] R11-C: overlap next physical-tile PV V-matrix programming across
  group boundaries without weakening current four-query V retention.
- [complete] R11-D: convert cluster P input and PV launch to a bounded row
  wavefront, preserving row order, backpressure, and resident-O accounting.
- [complete] R11-E: run focused contracts, build/install, feature A/B, E3/E4,
  pressure, MPI, lifecycle, and GPU comparison; retain only correct improvements.

- [complete] R10-A: trace manager K/V request timing, distributor slot
  ownership, PV matrix identity, P-input lifetime, and group-boundary release.
- [complete] R10-B: implement a bounded early manager K/V request path behind an
  independent feature flag and verify request/delivery/cancellation conservation.
- [complete] R10-C: implement only proven cross-tile PV matrix or P-input
  retention/overlap opportunities behind independent feature flags.
- [complete] R10-D: build and run feature-off, Q256/K1024, E3, E4, pressure, and
  MPI A/B; retain only numerically correct end-to-end improvements.

- [complete] R9-A: trace PV output readout, P/V movement, resident-O
  accumulation, bank ownership, and completion dependencies using R8 metrics.
- [complete] R9-B: implement a bounded timed PV-to-O accumulation path that
  removes redundant cross-component staging without weakening conservation.
- [complete] R9-C: add feature controls, statistics, cancellation/lifecycle
  contracts, and feature-off A/B coverage.
- [complete] R9-D: build and run Q256/K128, Q256/K1024, E3, E4, pressure, and MPI;
  accept only a numerically correct measured improvement.

- [complete] R8-A: extend the existing Manager/Worker GroupCtrl transport
  with a bounded, tagged Attention K/V distribution protocol while preserving
  the legacy mailbox scheduler protocol.
- [complete] R8-B: route worker K/V tile requests through the distributor with
  two resident manager slots, per-K/V readiness, four-consumer completion, and
  generation-safe cancellation/fallback.
- [complete] R8-C: wire feature flags, 64 B/cycle per-destination delivery,
  statistics, runner controls, and contract coverage.
- [complete] R8-D: build and run Q256/K128, Q256/K1024, E3, E4, pressure,
  cancellation, and MPI gates; accept only a correct measured improvement.
- [complete] R7-D: add explicit score/SFU/PV operand-bank consumer lifetime,
  generation, cancellation, and release accounting before attempting any
  further QK multi-producer overlap.
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
- Three K/V buffers passed correctness at E3/E4. E3 regressed from 330,125 to
  330,300 cycles and K/V wait increased from 105,272 to 105,443 cycles. E4
  improved from 1,007,628 to 997,866 cycles and K/V wait fell from 317,331 to
  271,017 cycles. The mixed, sub-1% end-to-end result does not justify changing
  the unconditional two-buffer default.

## Next Step

Reduce the E4 cross-query-group K/V delivery tail; only revisit P-row wavefront
after PV output readout and O commit can stream at the same granularity.

## Errors Encountered
| 2026-09-14 | QK lookahead initially launched zero tiles because it waited for pair-ready | Add worker-local K-ready signaling and overlap QK programming with V delivery |
| 2026-09-14 | E4 verifier assumed every speculative manager lookahead must hit | Preserve exact formal traffic counts and check missed speculation against slot stalls |
| 2026-09-14 | WCP pressure run requested unsupported queue depth 2 | Use the cluster contract's minimum bounded depth of 16 |
| 2026-09-14 | New independent-switch test omitted `--attention-cluster` | Enable the required architecture in the test fixture |
| 2026-09-14 | Initial P-row wavefront lifecycle expected six batch launches and PV concurrency 32 | Root cause was a stale verifier assumption; express 36 row launches and bounded variable concurrency for the explicit A/B path |
| 2026-09-14 | Initial R11 planning patch used an imprecise context anchor | Re-read the planning files with line numbers and applied a narrower patch |
| 2026-09-14 | Fused accumulator contract initially expected 16 O read-busy cycles | Final context drain also consumes 128 segment reads; corrected the expected total to 144 |
| 2026-09-14 | Tried to run `attention_cluster/test_attention_cluster_contract` directly | The contract test is a source-only target; rebuild it under `/tmp` before execution |
| 2026-09-14 | Contract test compile used `-Isrc/sst/elements` | Include is rooted at `src`; rebuild with `-Isrc` |
| 2026-09-14 | Combined planning update/build call could not match old checklist syntax | Read the current plan, patched its actual `[status]` syntax, then start the build separately |
| 2026-09-14 | E3/E4 row-fusion lifecycle rejected zero per-worker SFU/PV overlap | Row fusion removes the old serialized O-submit tail that created this overlap; retain QK/PV overlap plus interval/O conservation gates, and require SFU/PV only on the legacy path |
| 2026-09-14 | R9 results document patch had one unprefixed added line | Correct the patch syntax and reapply; no file changes were made by the failed patch |
| 2026-09-14 | R10 P-overlap Q256/K1024 first run failed lifecycle despite zero numerical mismatches | Cluster rows were verified with the generic PV-array count; add a regression test and use `jobs * 16` for cluster P rows |
| 2026-09-14 | R10 runner contract failed after lookahead defaulting and a new test raised `NameError: ROOT` | Make non-explicit lookahead follow distribution and use the existing repository path expression |
| 2026-09-14 | Broad `/tmp` artifact search reported permission errors in unrelated private directories | Restrict subsequent reads to the known R9 artifact directory |
| 2026-09-14 | Findings current-performance patch did not match its broad context | Re-read the exact paragraph and apply a smaller scoped replacement |

| Error | Attempt | Resolution |
|---|---:|---|
| `jq` rejected `$label` in the comparison filter | 1 | Rename the argument to a non-keyword variable and rerun extraction; simulations and artifacts are unaffected. |
| `make -C src/sst/elements/golem groupctrl/groupctrl.lo` has no such target | 1 | Use the repository's standard build/install script after integration instead of retrying an unsupported partial target. |
| Q256/K128 model construction required missing scheduler endpoints | 1 | The runner had selected the legacy control architecture; investigate its compatibility before rerunning. |
| Q256/K128 completed with zero Attention and HBM activity | 2 | The legacy control architecture omitted the four fused-Attention guest arguments. Wire GroupCtrl into the proven Attention architecture and keep that architecture selected. |
| Focused K/V distribution contract raised `NameError` | 1 | Reuse the existing `ARCHIVE_ARCH` path constant in the new architecture assertions. |
| Direct scale runner rejected `--mpi-ranks` | 1 | Set its documented `GOLEM_MPI_RANKS=2` environment variable; the CLI flag belongs to the outer wrapper. |
