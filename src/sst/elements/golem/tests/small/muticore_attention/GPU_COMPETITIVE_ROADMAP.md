# GPU-Competitive FlashAttention Roadmap

## Objective And Measurement

The project targets FP32, non-causal `B=1,H=1,D=128` Attention. SST latency is
measured from descriptor acceptance through accelerator tensor completion and
normalized at 1 GHz. RTX 5060 Scope A is a warm-cache CUDA device-timeline
measurement. MPI affects host simulation wall time only.

| Profile | SST R11 | RTX 5060 Scope A | Ratio | Target |
|---|---:|---:|---:|---:|
| E3, Q=K=1024 | 127,589 | 97,568 | 1.308x | `<97,568` |
| E4, Q=K=2048 | 430,131 | 345,984 | 1.243x | `<345,984` |

The GPU gate remains failed. Multi-head Attention and MoE stay deferred until
both same-precision gates pass or a measured resource lower bound proves that
the current architecture cannot pass them.

## Current Architecture

The verified model uses four managers plus 16 workers. Each worker has 64
physical 64x64 arrays, 64 CUs per array, two operand banks, and a fixed default
partition of 16 QK plus 48 PV arrays. D128 is mapped through paired D64 paths.
Br16/Bc32, two worker K/V buffers, WCP issue width one, the timed 16-lane O FMA,
and a two-slot manager K/V distributor are retained. R9 adds a 128-lane
PV-to-resident-O row path implemented as eight parallel 16-lane banks. R10
uses freed manager slots for one-tile K/V lookahead and overlaps consecutive P
FIFO reads with PV array input programming.

R10 improves E3/E4 another 13.46%/20.05% over R9. Numerical/lifecycle, WCP
pressure, feature-off fallback, and MPI determinism gates pass.
R11 adds independent QK and PV next-tile matrix lookahead in the existing two
operand banks, improving E3/E4 another 13.88%/16.81% over R10.

## Evidence

- PV output/read-write falls from 59,708/239,736 to 6,652/26,616 cycles,
  approximately 88.9%, and measures about 52 cycles per PV tile.
- K/V critical-worker ready wait is now zero on E3/E4. Each manager still loads
  every physical tile once; 30/124 lookahead loads per manager all hit.
- HBM queue p95 is one cycle and NoC maximum port utilization is 2.01%/3.03%,
  so additional global bandwidth is not currently justified.
- The fused O path reports zero read/write/ALU waits and bank conflicts; further
  O widening is not an evidence-backed end-to-end optimization.
- Bc64, three K/V buffers, and WCP issue width two were measured and rejected.

## Ordered Next Work

1. Remove the E4 cross-group boundary long tail (maximum 17,309 cycles) while
   preserving manager slot and score/P context conservation.
2. Carry P-row wavefront through PV readout and O commit before enabling it.
3. Re-profile steady II, boundary II, and array-buffer/local-GM contention.
4. Run the full Q256/K128, Q256/K1024, E3, E4, pressure, MPI, and GPU
   comparison matrix for every accepted architecture revision.

## Rejected R7 Hypotheses

- Moving the ahead scheduler call before SFU issue was a no-op.
- Partial score-bank retain/release broke E3 lifecycle.
- Bank-aware multi-producer scheduling caused 9,404 E3 mismatches because the
  visible QK lease did not cover downstream bank consumers.
- Three-buffer K/V was correct but regressed E3 from 330,125 to 330,300 cycles.

All rejected changes remain disabled. The verified implementation baseline is R11.
