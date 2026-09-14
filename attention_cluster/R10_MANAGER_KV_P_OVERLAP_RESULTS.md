# R10 Manager K/V Lookahead And P-Row Overlap

## Architecture

R10 keeps the existing two worker K/V buffers and two manager distributor
slots. When all four workers acknowledge a physical K/V tile, the manager first
serves queued demand and then uses the released slot to stage at most one next
sequential tile. The speculative slot has no worker destination until a real
request coalesces into it; delivery still uses the timed 64 B/cycle worker-local
write path and normal ACK/cancellation accounting.

```text
worker buffers:       current N | requested N+1       (still two)
                              ACK all four workers
                                      |
manager slots:        release N | load farthest+1 ----+--> resident K/V
                                                   real N+2 requests arrive
                                                          |
                                                          v
                                          timed local writes + normal ACK
```

The PV V matrix was already retained across the four queries sharing a physical
tile. R10 therefore does not duplicate matrix storage. Instead, its one-entry P
buffer reads row N+1 from the tagged SFU FIFO while row N is being programmed
into the PV arrays. The first row remains demand-read, and prefetch never crosses
a PV wave boundary.

Both paths are independently ablatable with
`GOLEM_ATTENTION_KV_MANAGER_LOOKAHEAD` and
`GOLEM_ATTENTION_PV_INPUT_PIPELINE`.

## A/B Results

| Case | R9 | R10 | Change | Verification |
|---|---:|---:|---:|---|
| Q256/K128 | 17,751 | 17,676 | -0.42% | numerical/lifecycle PASS |
| Q256/K1024 | 159,345 | 123,129 | -22.73% | numerical/lifecycle PASS |
| E3 Q1024/K1024 | 171,186 | 148,146 | -13.46% | numerical/lifecycle PASS |
| E4 Q2048/K2048 | 646,670 | 517,020 | -20.05% | numerical/lifecycle PASS |
| Q256/K128 pressure | 18,381 | 18,306 | -0.41% | numerical/lifecycle PASS |
| Q256/K128 MPI 2 | 17,751 | 17,676 | deterministic | numerical/lifecycle/MPI PASS |

On Q256/K1024, manager lookahead alone reaches 123,821 cycles. Adding P-row
overlap reaches 123,129, so the measured contributions are 35,524 and 692 cycles
respectively versus the strict 159,345-cycle feature-off reproduction.

## Critical-Worker Phase Cycles

Raw SST phase ticks are converted at 1,000 ticks per normalized cycle.

| Phase | E3 | E4 |
|---|---:|---:|
| Initial/unhidden K/V load | 2,152 | 2,152 |
| Q local read | 4,160 | 16,640 |
| QK matrix program | 25,023 | 100,095 |
| QK input program | 6,144 | 24,576 |
| QK compute/readout | 7,424 | 29,696 |
| Softmax | 6,023 | 24,071 |
| PV matrix program | 16,896 | 67,584 |
| PV P-input program | 11,440 | 45,855 |
| PV compute | 4,638 | 18,556 |
| PV output/O accumulation | 6,683 | 26,742 |

The reporting-level operator totals are QK 38,592/154,367 cycles and PV
39,657/158,738 cycles. E3/E4 K/V ready wait is zero. E3 steady p95 is at the
625-cycle target, while the remaining boundary average is 1,695 cycles. E4 has
a 1,843-cycle boundary average and a 20,535-cycle maximum, now the main tail.

## Traffic And Capacity Conservation

Per E3 manager, 32 physical tiles produce 32 manager loads, 30 lookahead loads,
30 lookahead hits, 128 deliveries, 126 coalesced requests, zero slot stalls, and
a maximum of two occupied slots. Per E4 manager the corresponding counts are
128, 124, 124, 512, and 508. Manager HBM bytes remain exactly one K/V pair per
physical tile; no extra worker or manager buffer was added.

Lifecycle verification now checks these request/load/byte/delivery/lookahead
relations automatically, together with zero cancel/stall counts on successful
runs and the two-slot maximum.

## GPU Gap And Next Bottleneck

RTX 5060 FP32 Scope A remains 97,568 cycles for E3 and 345,984 for E4. R10 is
1.518x and 1.494x those baselines, leaving gaps of 50,578 and 171,036 cycles.
The next evidence-backed target is balanced QK/PV work plus the E4 group-boundary
tail, not additional K/V bandwidth or more K/V buffers.

## Verification

- Incremental SST/Golem build and local install: PASS.
- 88 Python workload contracts: PASS.
- C++ cluster contract with `-Wall -Wextra -Werror`: PASS.
- Feature-off, Q256/K128, Q256/K1024, E3, E4, pressure, and MPI 2: PASS.
- Numerical mismatches: zero in every runtime case.
