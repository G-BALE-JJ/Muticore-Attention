# R11 QK/PV Matrix Lookahead

R11 keeps the R10 16-QK + 48-PV array split, two operand banks, two worker K/V
buffers, and two manager distributor slots. It changes scheduling and readiness
signaling, not physical capacity.

## Design

- GroupCtrl emits a K-ready callback after the worker-local K write completes,
  while V delivery continues in the background.
- The worker reads that K tile and programs the next physical tile into the
  inactive QK operand bank. Full generation/job/group/query/tile tags guard
  promotion and release.
- After the complete K/V callback, V is read and programmed into the next PV
  operand bank while QK and softmax execute.
- QK and PV lookahead have independent feature switches. A tile that arrives
  too late uses the unchanged demand path.
- The experimental P-row-only wavefront is retained off by default: it is
  numerically correct but regresses Q256/K128 by 15 cycles because output drain
  remains a whole-wave barrier.

## Results

| Case | R10 cycles | R11 cycles | Change | Verification |
|---|---:|---:|---:|---|
| Q256/K128 | 17,676 | 16,112 | -8.85% | numerical/lifecycle PASS |
| Q256/K1024 | 123,129 | 119,054 | -3.31% | numerical/lifecycle PASS |
| E3 Q1024/K1024 | 148,146 | 127,589 | -13.88% | numerical/lifecycle PASS |
| E4 Q2048/K2048 | 517,020 | 430,131 | -16.81% | numerical/lifecycle PASS |
| Q256/K128 WCP depth 16 | 18,306 | 16,112 | -11.99% | numerical/lifecycle PASS |
| Q256/K128 MPI 2 | 17,676 | 16,112 | -8.85% | numerical/lifecycle/MPI PASS |

The Q256/K128 component A/B is 17,156 cycles for QK-only and 16,632 for
PV-only. Their combined 16,112-cycle result shows that the disjoint QK/PV array
groups overlap without cancelling each other's benefit.

Against RTX 5060 FP32 Scope A, E3 is now 1.308x and E4 is 1.243x slower. The
remaining gaps are 30,021 and 84,147 cycles. E4 still has a 17,309-cycle maximum
group boundary and nine cross-query K/V waits, so group-boundary delivery is the
next target; compute-only P wavefront work is not justified without streamed PV
readout/O commit.

## Verification

- Full SST/Golem build and local install: PASS.
- 89 Python workload contracts: PASS.
- Warnings-as-errors C++ cluster contract: PASS.
- QK-only, PV-only, combined, multi-tile, E3, E4, WCP pressure, and MPI 2: PASS.
- Numerical mismatches: zero in every runtime case.
