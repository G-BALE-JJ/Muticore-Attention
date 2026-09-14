# R8 Manager-Level K/V Distribution

## Architecture

Each of the four managers owns a bounded two-slot K/V distributor for its four
workers. Requests are keyed by job generation, query group, and key tile. The
manager loads striped K and V data once into reserved local scratch, reads the
completed pair in bounded chunks, and sends it over the existing GroupCtrl
links. Each worker performs a timed 64 B/cycle local-GM write before its RoCC
callback becomes ready.

```text
                         one striped K/V DMA pair
                                   |
                                   v
HBM nodes ---> manager local scratch (2 resident slots)
                                   |
                      coalesce 4 tagged requests
                   +---------------+---------------+
                   |               |               |
                   v               v               v
              worker 0         worker 1    ... worker 3
              K/V buffer       K/V buffer       K/V buffer
                   |               |               |
                   +-------- ACK / generation cancel -------+
```

Slots are released only after all four subscribers acknowledge delivery.
Generation cancellation removes pending callbacks and stale subscribers. The
feature-off path retains the original per-worker DMA implementation.

## A/B Results

All cases use FP32, non-causal `B=1,H=1,D=128`, 16 QK + 48 PV arrays, Br16/Bc32,
two worker K/V buffers, and one SST rank unless stated otherwise.

| Case | R6 worker DMA | R8 manager distribution | Change | Verification |
|---|---:|---:|---:|---|
| Q256/K128 | 30,607 | 18,169 | -40.6% | numerical/lifecycle PASS |
| Q256/K1024 | 308,481 | 160,087 | -48.1% | numerical/lifecycle PASS |
| E3 Q1024/K1024 | 330,125 | 176,313 | -46.6% | numerical/lifecycle PASS |
| E4 Q2048/K2048 | 1,007,628 | 687,791 | -31.7% | numerical/lifecycle PASS |
| Q256/K128 WCP pressure | 31,657 | 18,862 | -40.4% | numerical/lifecycle PASS |
| Q256/K128 MPI 2 | - | 18,169 | deterministic | numerical/lifecycle/MPI PASS |

The Q256/K128 A/B rerun reproduced 30,607 cycles with distribution disabled and
18,169 cycles enabled. Across four managers, the enabled run recorded 32 worker
requests, eight manager K/V loads, 24 coalesced requests, 32 deliveries, and a
two-slot high-water mark. Manager HBM traffic was 512 KiB versus 2 MiB of
delivered worker-local data, confirming the intended 4:1 physical-read reduction.

## Updated Bottleneck

Critical-worker K/V wait fell from 105,272 to 8,798 cycles on E3 and from
317,331 to 38,985 cycles on E4. E3/E4 are now 1.807x/1.988x the RTX 5060 Scope A
measurements, so the GPU gate remains open. PV work dominates the critical
worker at 97,595/391,286 cycles, and query-group boundary II averages
2,947.0/3,180.2 cycles. The next optimization should target PV and explicit
score/SFU/PV operand-bank lifetime across group boundaries.

## Verification

- Standard SST/Golem build and local install: PASS.
- 68 Python Attention contracts: PASS.
- C++ finite-resource/cancellation contract with `-Wall -Wextra -Werror`: PASS.
- Feature-off fallback, multi-tile, E3, E4, WCP pressure, and MPI 2: PASS.
- Numerical mismatches: zero in every runtime case above.
