# R12 Ramulator2 HBM2E Backend

R12 corrects the fused-Attention memory-backend wiring. Earlier R11 resolved
configuration files named Ramulator2, but the selected archive architecture
unconditionally instantiated DRAMSim3. The Attention runner now explicitly
selects `memHierarchy.ramulator2` with `hbm2e_2500.yaml`, and a mandatory
post-SST gate requires Ramulator2 runtime summaries and rejects DRAMSim3
statistics.

## HBM2E Configuration

- Ramulator2 2.1, GenericDRAM, eight HBM2 controllers/channels.
- 2500 MT/s, 0.8 ns tCK, two pseudo-channels, 128-bit channel width.
- One rank, four bank groups, four banks per group, open-row policy.
- FR-FCFS row-hit scheduling, per-bank refresh, 64-entry read/write queues.
- 32-byte SST request width and 128 MiB exposed capacity per memory node.
- Four striped Attention data nodes plus one separate OS node.

The timing microbenchmark passes at 39.57 GB/s per channel, 316.67 GB/s for an
eight-channel stack without refresh, and 295.43 GB/s with per-bank refresh.

## Verified Results

| Case | R11 DRAMSim3 | R12 Ramulator2 | Change | Verification |
|---|---:|---:|---:|---|
| Q256/K128 | 16,112 | 15,565 | -3.39% | numerical/lifecycle/backend PASS |
| E3 Q1024/K1024 | 127,589 | 142,737 | +11.87% | numerical/lifecycle/backend PASS |
| E4 Q2048/K2048 | 430,131 | 518,096 | +20.45% | numerical/lifecycle/backend PASS |

These changes are memory-model effects, not an architecture optimization A/B.
Against RTX 5060 FP32 Scope A, E3 is 1.463x and E4 is 1.497x slower, leaving
45,169 and 172,112 normalized cycles respectively.

Ramulator2 exposes 10,283/29,004 critical-worker K/V wait cycles on E3/E4. The
E4 maximum group-boundary interval rises to 60,243 cycles. This makes bounded
manager delivery and the cross-group tail the next optimization target under
the corrected backend.

## Verification

- 90 Python workload contracts: PASS.
- Ramulator2 timing and stack-bandwidth microbenchmark: PASS.
- Q256/K128, E3, and E4 numerical verification: zero mismatches.
- Q256/K128, E3, and E4 lifecycle and instantiated-backend gates: PASS.
