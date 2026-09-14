# H64 R1-R6 Attention Optimization Results

## Outcome

The fixed worker budget remains 64 physical 64x64 arrays and 64 CUs per array.
The measured default partition is now 16 QK arrays (0-15) plus 48 PV arrays
(16-63). D128 remains two D64 physical paths. QK uses eight row pairs in each
of two waves; PV uses 16 of its 24 row pairs in one wave. No manager array,
zero-delay callback, unbounded queue, or free SRAM port was added.

One real hardware capability was added: the 16-lane FP32 O engine now has a
two-cycle fused multiply-add datapath for `alpha * old_o + pv`, replacing its
separate multiply and add issues. The capability is part of the architecture
configuration/fingerprint and numerical execution uses `std::fma`. It still has
one read issue, one write issue, one ALU issue, 16 row banks, finite pending
operations, and the original generation/tag/context checks.

All reported E3/E4 results pass numerical and lifecycle verification. The GPU
gate still fails.

| Profile | New cycles | H64 32/32 | L12 | Delta vs L12 | RTX 5060 Scope A | New/GPU |
|---|---:|---:|---:|---:|---:|---:|
| E3, Q=K=1024 | 330,125 | 365,119 | 364,266 | -34,141 (-9.37%) | 97,568 | 3.384x |
| E4, Q=K=2048 | 1,007,628 | 1,199,780 | 1,049,928 | -42,300 (-4.03%) | 345,984 | 2.912x |

The table uses the single default 16/48 partition. Profile tuning can reach
326,185 cycles for E3 with 40/24 and 1,004,777 cycles for E4 with 32/32, but
those are not mixed into the default headline. Final default Q256 results are
recorded after the last rebuild in the verification section below.

## Ownership And Tile Sweep

The initial pre-FMA Q256/K128 ownership-isolation cases were real timed
simulations with identical math:

| QK/PV arrays | Cycles | Observed QK/PV max concurrency | Broadcast sink bytes |
|---|---:|---:|---:|
| 16/48 | 30,863 | 16 / 32 | 25,690,112 |
| 24/40 | 30,867 | 24 / 32 | 29,884,416 |
| 32/32 | 30,731 | 32 / 32 | 34,078,720 |
| 40/24 | 30,635 | 32 / 24 | 29,884,416 |

With the final fused O engine, all requested ownership cases were rerun under
identical Bc32 timing and numerical/lifecycle gates:

| QK/PV arrays | E3 cycles | E4 cycles | E3 steady II | E4 steady II |
|---|---:|---:|---:|---:|
| 16/48 | 330,125 | 1,007,628 | 679.5 | 730.1 |
| 24/40 | 330,173 | 1,016,875 | 679.5 | 704.4 |
| 32/32 | 332,263 | 1,004,777 | 677.0 | 693.5 |
| 40/24 | 326,185 | 1,031,493 | 661.5 | 755.1 |

The default is 16/48: it strictly dominates 24/40, has the best mean E3/E4 GPU
ratio, and reduces broadcast replication. The 32/32 E4 advantage is only 2,851
cycles, while 40/24 remains an explicit E3 profile-tuning option. Bc64 regressed E3 to
436,401 cycles at 16/48 and 436,279 at 32/32 because K/V wait rose to about
176K cycles. Bc32 is retained. Br16 remains the physical row/context granularity;
there is no honest parameter-only Br32 mapping. Time-multiplexing all 64 arrays
was not promoted because it requires reprogramming the same bounded operand banks
between stages and cannot improve the measured feed/readiness bottleneck.

## Critical Resources

The table is the critical worker's interval union, not a sum of parallel worker
time. Idle is the gap inside that resource's first-to-last active span.

| Resource | E3 busy / idle / utilization | E4 busy / idle / utilization | Max concurrency E3/E4 |
|---|---:|---:|---:|
| QK arrays | 17,408 / 232,173 / 6.97% | 69,632 / 856,137 / 7.52% | 16 / 16 |
| PV arrays | 4,608 / 246,367 / 1.84% | 18,432 / 908,730 / 1.99% | 32 / 32 |
| Array-buffer ports | 44,449 / 208,286 / 17.59% | 177,830 / 751,093 / 19.14% | 2 / 2 |
| Local-GM read | 33,670 / 220,407 / 13.25% | 134,030 / 796,236 / 14.41% | 2 / 2 |
| Local-GM write | 17,680 / 309,691 / 5.40% | 68,640 / 937,331 / 6.82% | 1 / 1 |
| SFU | 24,071 / 226,047 / 9.62% | 96,263 / 830,043 / 10.39% | 1 / 1 |
| O read | 16,384 / 201,467 / 7.52% | 65,536 / 792,172 / 7.64% | 1 / 1 |
| O write | 16,384 / 235,020 / 6.52% | 65,536 / 862,055 / 7.06% | 1 / 1 |
| O FMA issue | 16,384 / 235,020 / 6.52% | 65,536 / 862,055 / 7.06% | 1 / 1 |

The E3/E4 steady-tile II is 679.5/730.1 cycles. K/V critical-worker wait is
105,272/317,331 cycles. PV output/O is 59,708/239,736 cycles, 18.09%/23.79% of
end-to-end latency; the complete PV path is 97,596/391,286 cycles.

Broadcast useful ingress is 46,137,344/184,549,376 bytes and physical sink
traffic is 293,601,280/1,174,405,120 bytes. WCP issues 172,032/688,128 commands;
aggregate queue residence is 172,049/688,148 cycles, maximum per-command wait is
two cycles, and increasing issue width from one to two changes E3 by zero cycles.

The generated DMA summaries report 17,301,504/68,157,440 physical read bytes and
524,288/1,048,576 write bytes. Unique logical K/V tensor payload is only
1,048,576/2,097,152 bytes. Memory-controller queue delay is one cycle at p95;
backend read p95 is 190/178 cycles. NoC crossbar stalls are 1,635/2,658, output
port stalls are zero, and maximum port utilization is 2.01%/3.03%. DRAMSim
aggregate bandwidth is 1.593/0.612 Gbit/s, with peak channel bandwidth
0.234/0.090 Gbit/s.

Counter availability is explicit: array, array-buffer, Local-GM, SFU, and O have
critical-worker busy/idle interval counters above. O additionally reports
aggregate read wait 6,190,080/25,159,680 cycles, ALU wait 145,920/291,840,
write wait zero, and row-bank conflict 47,993,344/192,783,360 worker-cycle sums.
WCP exposes cumulative issue-busy and queue wait but no critical-path idle union.
DMA exposes bytes, retries (zero), response queue depth/wait, and RTT but no DMA
engine busy/idle union or bank-conflict counter. HBM exposes bandwidth and queue
latency but no controller busy/idle-union or useful-byte counter. NoC exposes
physical sent bits, utilization and crossbar/output stalls, but no logical
useful-byte or busy/idle-union counter. Array-buffer and Local-GM queue-wait,
bank-conflict, and byte counters are not exported separately; their service
interval unions are therefore reported without invented values. SFU score/P SRAM
port wait and bytes are present in raw stats, while a distinct SFU bank-conflict
counter is not applicable to that model.

## Lower Bound And Next Hardware

For default Bc32, the actual concurrent array service per tile is 136 QK cycles
(two waves) and 36 PV cycles (one 32-array wave), not 544 serialized PV cycles.
The strictest measured per-tile resource-work constraint is the array-buffer:
44,449/128 and 177,830/512 = 347.3 cycles, hence an analytical conservation
floor of 348 cycles/tile. This gives lower bounds of 44,544 E3 cycles and 178,176
E4 cycles before fill/dependency penalties. Actual latency is 7.41x/5.66x those
floors, leaving 285,581/829,452 cycles. The observed minimum steady interval is
554 cycles for 16/48; average II is higher because K/V readiness and boundary
dependencies cannot be hidden by the finite two-context pipeline.

The next real hardware priority is not more arrays, WCP width, HBM channels, or
an extra generic DMA. It is a bounded K/V distribution/reassembly engine that
loads each striped K/V tile once per four-worker group and multicasts it into the
existing operand SRAM under generation/lease control. Current DMA reads are
17.30/68.16 MB; ideal four-worker sharing would target roughly 4.33/17.04 MB,
about 75% less physical K/V traffic. It needs at least 64 B/cycle sustained local
delivery per destination, four destinations, two resident tile contexts, and
separate completion credits per K and V segment. Only after that should a second
O read/FMA/write lane be considered; current O ports are below 8% interval
utilization, so adding lanes alone has no supported end-to-end benefit.

## Module Changes

- `attentionCluster.h`: dynamic ownership contract plus the explicit 16-lane,
  two-cycle fused FP32 O engine capability and scheduling.
- `roccAnalog*.h`: ownership-aware wave mapping, resource statistics, and 16/48
  default.
- `mvmComputeArray.h` and `workercmdproc.h`: dynamic lower-layer traffic-class
  admission, positive control latency, and deferred completion while preserving
  topology, bank, and bounded-queue checks.
- `cpu_builder.py` and `run_fused_attention_scale.sh`: end-to-end ownership
  propagation and measured default.
- verifier/reporter/tests: ownership-aware launch/broadcast contracts, critical
  interval resource profiles, O port profiles, and numerical/lifecycle checks.

## Completion And Verification

- Architecture modification: complete for R1-R5. Cross-worker K/V sharing was
  evaluated but intentionally not fabricated from existing hardware; it remains
  the next explicit hardware addition described above.
- Functional correctness: complete for the tested D128 non-causal contract. All
  final numerical runs have zero mismatches and all lifecycle gates pass.
- Performance optimization: successful versus H64 and L12, by 9.58%/16.02% and
  9.37%/4.03% on E3/E4 respectively.
- GPU gate: failed. Default E3/E4 remain 3.384x/2.912x RTX 5060 Scope A.
- Independent review: first pass found 0 Critical and 6 Important; all six were
  fixed. Targeted follow-up found 0 Critical and 0 Important.

Fresh post-review artifacts are under `attention_cluster/results/`:

| Verification case | Cycles | Result |
|---|---:|---|
| `r6_final_default_q256_k128` | 30,607 | numerical/lifecycle PASS |
| `r6_final_default_q256_k1024` | 308,481 | numerical/lifecycle PASS |
| `r6_final_default_e3` | 330,125 | numerical/lifecycle PASS |
| `r6_final_default_e4` | 1,007,628 | numerical/lifecycle PASS |
| `r6_final_pressure_q256_k128` | 31,657 | numerical/lifecycle PASS |

The pressure case uses WCP queue depth 16 and eight-cycle command/completion
latencies. Cancellation conservation is exercised deterministically by the C++
contract for generation fences, score FIFO, pending O operations, contexts, and
bank leases; the runner has no runtime cancel-injection interface, so no claim of
an end-to-end injected-cancel simulation is made.
