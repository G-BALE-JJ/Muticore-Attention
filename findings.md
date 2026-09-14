# Findings

## Current Architecture

- The implemented Attention contract uses four managers and 16 workers.
- Each worker models 64 physical 64x64 arrays, 64 CUs per array, FP32, two
  operand banks, and a default 16-QK + 48-PV ownership split.
- D128 uses paired D64 paths. QK executes two waves; PV uses 32 arrays in one
  wave. The O engine provides a timed 16-lane, two-cycle FP32 FMA.

## Current Performance

- E3/E4 complete in 330,125/1,007,628 normalized cycles.
- This is 9.58%/16.02% faster than the initial H64 result and 9.37%/4.03%
  faster than L12, but still 3.384x/2.912x the RTX 5060 Scope A result.
- Array-buffer resource work gives a 348-cycle/tile conservation floor. Measured
  steady II is 679.5/730.1 cycles because K/V readiness and boundary dependencies
  are not hidden by the finite two-context pipeline.
- HBM and NoC are not saturated: memory-controller queue p95 is one cycle and
  maximum NoC port utilization is 2.01%/3.03%.
- Current evidence favors bounded four-worker K/V distribution/reassembly over
  more arrays, wider WCP issue, extra HBM channels, or an extra O lane.

## R7 Result

- Moving the ahead scheduler call did not change cycles or overlap.
- An incomplete score-bank ownership change passed Q256 but failed E3 lifecycle.
- Removing the global ahead-QK producer gate caused 9,404 E3 mismatches. The
  visible QK lease does not cover all score/SFU/PV operand-bank consumers.
- All unsafe experiments were reverted. Fresh E3 returned to 330,125 cycles,
  zero mismatches, and numerical/lifecycle PASS.
- A real three-buffer K/V run passed correctness but regressed to 330,300 cycles.

## Cleanup

- The removed experiment tree occupied about 32 GiB and contained 5,639
  untracked files, primarily HBM images, tensors, logs, and raw statistics.
- Publication retains five final R6 cases with three JSON files each and a
  SHA-256 manifest. Superseded 8+8 and H64 transition documents were removed.
- Generated run directories are ignored so future experiments do not pollute
  repository status.
