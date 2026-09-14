# R9 PV-to-O Row Fusion

## Architecture

The PV arrays still drain one 128-element FP32 row through the modeled
512 B/cycle near-array path. After that timed transfer, the row is submitted
once to eight parallel 16-lane resident-O banks instead of being split into
eight serialized RoCC-to-accumulator submissions.

```text
                  128 FP32 values = 512 B
PV array group ========= timed row drain =========>
                                                   |
                    +------------------------------+
                    | one tagged row transaction
                    v
        +------+------+------+------+------+------+------+------+
        | O[0:15] | O[16:31] | ...                  | O[112:127] |
        | 16-lane | 16-lane  |     8 parallel banks| 16-lane    |
        +------+------+------+------+------+------+------+------+
                    | FP32: O = alpha * O + PV
                    v
              resident O context, then final drain
```

Four resident O contexts, generation tags, key-tile ordering, cancellation,
two-cycle FP32 FMA latency, and final context drain are unchanged. The
`--no-pv-o-row-fusion` option selects the legacy eight-segment path in the same
binary.

## A/B Results

All cases use FP32, non-causal `B=1,H=1,D=128`, 16 QK + 48 PV arrays,
Br16/Bc32, two K/V buffers, and R8 manager K/V distribution.

| Case | R8 segmented O | R9 row fusion | Change | Verification |
|---|---:|---:|---:|---|
| Q256/K128 | 18,169 | 17,751 | -2.30% | numerical/lifecycle PASS |
| Q256/K1024 | 160,087 | 159,345 | -0.46% | numerical/lifecycle PASS |
| E3 Q1024/K1024 | 176,313 | 171,186 | -2.91% | numerical/lifecycle PASS |
| E4 Q2048/K2048 | 687,791 | 646,670 | -5.98% | numerical/lifecycle PASS |
| Q256/K128 WCP pressure | 18,862 | 18,381 | -2.55% | numerical/lifecycle PASS |
| Q256/K128 MPI 2 | 18,169 | 17,751 | deterministic | numerical/lifecycle/MPI PASS |

E3/E4 remain above the RTX 5060 FP32 Scope A results of 97,568/345,984 cycles:
1.755x and 1.869x respectively.

## Resource Effect

Each PV tile contains 16 rows and eight D-panel segments per row. The logical
work remains 128 segments per tile, but physical accumulator issue count falls
from 128 segment transactions to 16 row transactions. For Q256/K128, O write
and ALU busy time falls from 256 to 32 cycles per worker; O read busy time falls
from 256 to 144 cycles because the unchanged final context drain costs 128
reads.

PV output/read-write falls from 59,708 to 6,652 cycles on E3 and from 239,736
to 26,616 on E4, both approximately 88.9%. The new path measures 51.97 cycles
per PV tile, close to the modeled row-level floor: 32 near-array transfer cycles
plus 16 row issues and pipeline fill.

The runtime conserves 32,768/131,072 fused rows and 16/64 MiB of fused input
bytes on E3/E4. O context reservations equal releases; read, write, ALU, bank,
and drain wait counters are all zero.

## Updated Bottleneck

The shorter O tail exposes K/V readiness that was previously hidden. Critical
worker K/V wait is now 40,297 cycles on E3 and 140,175 on E4. PV matrix and
P-input programming together cost 33,279/133,118 cycles. The next optimization
should launch manager K/V delivery earlier and retain or overlap PV matrix/P
programming, rather than widening O again.

## Verification

- Standard SST/Golem build and local install: PASS.
- 69 Python Attention contracts: PASS.
- C++ accumulator contract with `-Wall -Wextra -Werror`: PASS.
- Feature-off, multi-tile, E3, E4, WCP pressure, and MPI 2: PASS.
- Numerical mismatches: zero in every runtime case.
