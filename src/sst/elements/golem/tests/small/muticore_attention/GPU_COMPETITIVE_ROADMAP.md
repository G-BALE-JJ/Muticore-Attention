# GPU-Competitive FlashAttention Roadmap

## Objective And Measurement

The project targets FP32, non-causal `B=1,H=1,D=128` Attention. SST latency is
measured from descriptor acceptance through accelerator tensor completion and
normalized at 1 GHz. RTX 5060 Scope A is a warm-cache CUDA device-timeline
measurement. MPI affects host simulation wall time only.

| Profile | SST R6 | RTX 5060 Scope A | Ratio | Target |
|---|---:|---:|---:|---:|
| E3, Q=K=1024 | 330,125 | 97,568 | 3.384x | `<97,568` |
| E4, Q=K=2048 | 1,007,628 | 345,984 | 2.912x | `<345,984` |

The GPU gate remains failed. Multi-head Attention and MoE stay deferred until
both same-precision gates pass or a measured resource lower bound proves that
the current architecture cannot pass them.

## Current Architecture

The verified model uses four managers plus 16 workers. Each worker has 64
physical 64x64 arrays, 64 CUs per array, two operand banks, and a fixed default
partition of 16 QK plus 48 PV arrays. D128 is mapped through paired D64 paths.
Br16/Bc32, two K/V buffers, WCP issue width one, and the timed 16-lane O FMA are
the accepted R6 configuration.

R6 improves E3/E4 by 9.58%/16.02% over the initial H64 mapping and by
9.37%/4.03% over L12. Numerical/lifecycle gates pass and independent review is
closed at zero Critical/Important findings.

## Evidence

- Measured steady II is 679.5/730.1 cycles; the array-buffer conservation floor
  is 348 cycles/tile.
- K/V critical-worker wait is 105,272/317,331 cycles.
- HBM queue p95 is one cycle and NoC maximum port utilization is 2.01%/3.03%,
  so additional global bandwidth is not currently justified.
- O resource interval utilization is below 8%; another O lane alone is not an
  evidence-backed end-to-end optimization.
- Bc64, three K/V buffers, and WCP issue width two were measured and rejected.

## Ordered Next Work

1. R7-D: explicitly model every score/SFU/PV operand-bank consumer under the
   generation/tag/lease/cancel contract.
2. Re-attempt QK producer overlap only after bank lifetime conservation is
   directly verified under Q256, E3, pressure, and cancellation cases.
3. Evaluate a bounded four-worker K/V distribution/reassembly engine that loads
   each striped tile once and delivers at least 64 B/cycle to four destinations,
   with two resident contexts and separate K/V completion credits.
4. Re-profile physical K/V traffic, K/V-ready wait, steady II, HBM, NoC, array
   buffer, SFU, and O resources before considering more arrays or O lanes.
5. Run the full Q256/K128, Q256/K1024, E3, E4, pressure, MPI, review, and GPU
   comparison matrix for every accepted architecture revision.

## Rejected R7 Hypotheses

- Moving the ahead scheduler call before SFU issue was a no-op.
- Partial score-bank retain/release broke E3 lifecycle.
- Bank-aware multi-producer scheduling caused 9,404 E3 mismatches because the
  visible QK lease did not cover downstream bank consumers.
- Three-buffer K/V was correct but regressed E3 from 330,125 to 330,300 cycles.

All rejected changes were reverted. The safe implementation baseline remains R6.
