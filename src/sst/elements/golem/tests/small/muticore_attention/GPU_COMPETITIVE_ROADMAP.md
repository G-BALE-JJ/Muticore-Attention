# GPU-Competitive FlashAttention Roadmap

## Objective And Measurement

The project targets FP32, non-causal `B=1,H=1,D=128` Attention. SST latency is
measured from descriptor acceptance through accelerator tensor completion and
normalized at 1 GHz. RTX 5060 Scope A is a warm-cache CUDA device-timeline
measurement. MPI affects host simulation wall time only.

| Profile | SST R28 sequential-64 | RTX 5060 Scope A | Ratio | Target |
|---|---:|---:|---:|---:|
| E3, Q=K=1024 | **31,197** | 97,568 | 0.320x | `<97,568` |
| E4, Q=K=2048 | deferred | 345,984 | - | after QK1024 optimization |

QK1024 now passes the same-precision latency gate in the modeled architecture.
Larger performance cases, multi-head Attention, and MoE remain deferred until
the fixed QK1024 path is fully optimized and its assumptions are frozen.

## Current Architecture

The verified model uses four managers plus 16 workers. Each worker has 64
physical 64x64 arrays and reuses all 64 in dependency order: QK, online
softmax, then PV. Br64/Bc64/D128 maps QK and PV through two D64 slices.

R24 models independent 256 B/cycle matrix broadcast and vector scatter. R26
groups old-O restore and output movement into 64-array, 256 B/cycle transfers.
R28 coalesces identical K/V chunks at each shared memory node, retains completed
chunks for launch-skewed consumers, and multicasts them at 256 B/cycle. K can
start QK before V completes; V is checked only at the PV boundary.

## Evidence

- QK1024 falls from the R26 44,790-cycle baseline to 31,197 cycles; numerical,
  lifecycle, and instantiated-backend verification pass.
- Critical-worker stages are 137 input, 6,736 QK, 12,567 softmax, and 9,848 PV
  cycles. All 15 K/V prefetches hit with zero exposed wait.
- The four data nodes issue exactly 64 physical K/V reads for 1 MiB of unique
  data and retire every completed cache entry.
- The current lower bound is 20,600 cycles. Softmax contributes 8,135 of the
  remaining 10,597-cycle gap.
- The Ramulator2 microbenchmark reaches 316.67 GB/s without refresh and 295.43
  GB/s with per-bank refresh for one eight-channel stack.

## Ordered Next Work

1. Reconcile the 12,567-cycle SFU path with its 4,432-cycle lower bound.
2. Optimize only fixed Q=K=1024/D=128 until that gap is closed.
3. Freeze the QK1024 architecture assumptions and rerun the broader dimension,
   pressure, MPI, and GPU comparison matrix.

## Rejected R7 Hypotheses

- Moving the ahead scheduler call before SFU issue was a no-op.
- Partial score-bank retain/release broke E3 lifecycle.
- Bank-aware multi-producer scheduling caused 9,404 E3 mismatches because the
  visible QK lease did not cover downstream bank consumers.
- Three-buffer K/V was correct but regressed E3 from 330,125 to 330,300 cycles.

All rejected changes remain disabled. The verified implementation baseline is
R28 sequential-64 with Ramulator2 HBM2E-2500.
