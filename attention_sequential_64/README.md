# 64-Array Sequential Worker Attention

## Final Architecture

The public Attention runner selects the sequential-64 path by default. The
legacy Attention cluster is used only with `--attention-cluster`.

For `Q=K=1024`, `D=128`, FP32:

- four managers each dispatch four workers;
- 16 workers execute concurrently;
- one worker owns `Qw[64,128]` and `Ow[64,128]`;
- each worker has 64 physical `64 input x 64 output` arrays;
- the key dimension is tiled into 16 `K/V[64,128]` tiles;
- one worker serially executes QK, online softmax, and PV for every key tile.

```text
4 HBM2E data nodes
        |
        | striped Q/O DMA; read-once K/V multicast prefetch
        v
+---------------- worker LocalGM ----------------+
| Q[64,128] | K/V ping | K/V pong | S/P | O      |
+------------------------+------------------------+
                         |
        +----------------+----------------+
        |                                 |
        v                                 |
  QK D64 reduction slice 0: D[0:64]       |
  retain 64x64 partial sums in arrays      |
  QK D64 reduction slice 1: D[64:128], add|
        |                                 |
        | one 64-array group read (16 KiB)|
        | array-major -> row-major transpose
        | one LocalGM tile write (16 KiB) |
        v                                 |
  S[64,64] -> 64-context SFU SRAM          |
  max -> exp/sum -> normalize              |
        |                                 |
        v                                 |
  P[64,64]                                 |
        |                                 |
        +---------> PV D64 output slice 0 |
                    PV D64 output slice 1 |
                              |            |
                              v            |
                   online accumulate O ----+
                              |
                    after tile 15: DMA O
                              v
                             HBM
```

### Unified D64 Slice Terminology

`D64 slice` is the common partitioning term for both phases. The qualifier
describes how the slices are combined:

- a QK D64 **reduction slice** contracts one 64-element part of D; both slices
  update the same `S[64,64]`, so their partial dot products are added;
- a PV D64 **output slice** produces a disjoint 64-element range of O; the two
  slices are concatenated along D and are never reduced together.

Calling both operations only a reduction would be mathematically wrong, while
calling both only a panel would hide the QK accumulation dependency.

### QK Mapping

For one key tile, array `j` represents key row `j`.

```text
common matrix on all arrays: Qhalf[64 outputs, 64 inputs]
array j input:               K[j, half*64 : (half+1)*64]
array j output:              64 query scores for key j
```

The first launch overwrites the output accumulator and the second launch adds
the other QK D64 reduction slice. All 64 arrays run in parallel. One launch is
`64 MAC cycles + 2 pipeline cycles = 66 cycles`; QK therefore needs 132 pure
array-compute cycles per key tile. It is not multiplied by 64.

The score output is consumed only after the second launch. The first D64 slice
does not leave the arrays. The final physical layout is array-major
`array[key][query]`; one group request reads all `64*64` FP32 values, the RoCC
callback transposes them to `S[query][key]`, and one logical LocalGM request
writes the complete score tile. Thus one key tile now performs one score read
and one score write, rather than 64 array reads followed by 4,096 scalar
writes.

For the requested sufficient-port-bandwidth experiment, the sequential runner
defaults both paths to 16 KiB/cycle and allows a 16 KiB LocalGM request. Both
ports retain the timing rule `base + ceil(bytes/bandwidth)`, with a one-cycle
base latency:

```text
64-array group read = 1 + ceil(16 KiB / 16 KiB/cycle) = 2 cycles
LocalGM tile write  = 1 + ceil(16 KiB / 16 KiB/cycle) = 2 cycles
ideal QK tile path  = 2*66 compute + 2 read + 2 write  = 136 cycles
ideal 16-tile QK    = 16*136                           = 2,176 cycles
```

These are service/dependency lower bounds. The measured QK phase also includes
WCP command issue, state-machine callbacks, matrix/input programming, and any
queue wait. Port widths remain configurable through
`GOLEM_ATTENTION_NEAR_ARRAY_OUTPUT_BYTES_PER_CYCLE`,
`GOLEM_LOCAL_GM_BYTES_PER_CYCLE`, and
`GOLEM_LOCAL_GM_MAX_REQUEST_BYTES`.

### Optimized SFU Mapping

The SFU owns 64 online row contexts, 16 vector lanes, and 16 EXP lanes. Each
context retains all 64 values of its row in SFU scratch SRAM across Max,
Exp/Sum, and Normalize. Intermediate values are no longer round-tripped through
LocalGM. LocalGM is used only to ingest S and emit P. The online `(m,l)` state
is retained across the 16 key tiles.

### PV Mapping and D64 Output Slices

A PV D64 output slice is a 64-element output range of the head dimension. D128
therefore has exactly two slices: `D[0:64]` and `D[64:128]`.

For output slice `p`, array `i` represents query row `i`:

```text
common matrix on all arrays: Vp^T[64 output dims, 64 key inputs]
array i input:               P[i,0:64]
array i output:              O contribution[i,p*64:(p+1)*64]
```

The complete `V[64,128]` tile is read once and retained while both output
slices are programmed. This prevents the K/V prefetcher from overwriting slice
1's source. Each slice costs 66 pure array cycles, so PV costs 132 cycles per
key tile.

## Cycle Lower Bounds

All numbers below are native 1 GHz accelerator cycles.

For one `Br=64, Bc=64, D=128` tile:

```text
QK   = 2 * (64 + 2) + 2 group read + 2 tile write = 136 cycles
SFU  = (4+4+4) + 63*max(4,4,4) + (4+8+1)
     = 277 cycles
PV   = 2 * (64 + 2) = 132 cycles
tile = 136 + 277 + 132 = 545 cycles
```

Each worker owns one Q block and traverses 16 key tiles:

```text
compute/dataflow dependency floor = 16 * 545 = 8,720 cycles
```

The 64 arrays and 16 workers are already parallel, so neither factor is
multiplied into latency.

Total HBM traffic is 17 MiB:

```text
Q read + O write + 16 workers * (K read + V read)
= 0.5 MiB + 0.5 MiB + 16 * (0.5 MiB + 0.5 MiB)
= 17 MiB
```

Each HBM data node and its DirectoryController attachment are now both modeled
at 320 GB/s. Four nodes therefore provide 1,280 B per native cycle. The
read-response direction carries 16.5 MiB and has an endpoint-link floor of
`ceil(16.5 MiB / 1280) = 13,517 cycles`. If reads and writes share the same
payload budget, all 17 MiB require `ceil(17 MiB / 1280) = 13,927 cycles`, equal
to the ideal HBM-media floor. The configured tCCD command roofline is 1,024
B/native-cycle, giving a separate 16.5 MiB read-command floor of 16,896 cycles.

The worker array-operand ingress has a 64 B/cycle raw floor:

```text
Q resident + 16 * (K tile + V tile + P tile)
= 32 KiB + 16 * (32 KiB + 32 KiB + 16 KiB)
= 1,312 KiB
1,312 KiB / 64 B/cycle = 20,992 cycles
```

The end-to-end roofline lower bound is therefore

```text
max(8,720 compute/dataflow,
    13,517 node-link reads,
    13,927 HBM/aggregate link traffic,
    16,896 HBM read-command issue,
    20,992 operand ingress)
= 20,992 cycles.
```

This is a strict resource lower bound, not an estimate of the current state
machine. It excludes NoC arbitration, LocalGM request startup, matrix
programming protocol, serialized score readout, queueing, and control skew.

## Optimization Test Policy

Until the QK path is fully optimized, all performance experiments, phase-cycle
reports, and end-to-end acceptance measurements use only
`Q=1024, K=1024, D=128` ("QK1024"). Small-shape source/unit/contract tests may
still be used to catch correctness errors, but they are not performance
experiments and must not be used for optimization claims. Larger dimensions
are deferred until the QK1024 optimization gate is complete.

## Measured Q1024 Result

Ramulator2 HBM2E, one MPI rank, and the default sequential-64 path:

| Critical-worker phase | Before | One-time QK I/O, 25 GB/s | 320 GB/s/node |
|---|---:|---:|---:|
| Input movement | 96,904 | 257,004 | 3,481 |
| QK total | 311,721 | 27,080 | 27,088 |
| QK matrix program | 4,219 | 4,216 | 4,224 |
| QK input program | 22,528 | 16,384 | 16,384 |
| QK compute/readout | 284,974 | 6,480 | 6,480 |
| Softmax | 12,567 | 12,567 | 12,567 |
| PV | 85,682 | 58,816 | 58,817 |
| PV matrix program | - | - | 8,512 |
| PV input program | - | - | 16,384 |
| PV old-O restore | - | - | 15,360 |
| PV compute | - | - | 2,176 |
| PV output read/write | - | - | 16,385 |
| Worker tile pipeline | 506,874 | 355,467 | 101,953 |
| End-to-end | 576,657 | 425,172 | 103,757 |

Numerical verification checked all 131,072 outputs: zero mismatches and maximum
absolute error `1.713e-09`. QK and PV both use exactly 64 active arrays per
wave. Every worker reports exactly 16 final QK group reads and 262,144 bytes,
which is one 16 KiB score read per key tile. The old serialized score path is
therefore absent from the measured run.

At 320 GB/s/node, all 15 K/V prefetches hit and exposed K/V wait falls from
221,935 to zero cycles. The end-to-end result improves by 75.60% (4.098x) from
425,172 to 103,757 cycles. Aggregate measured HBM bandwidth is 529.97 GB/s;
NoC maximum port utilization is 4.34%, with 215 crossbar stalls and zero output
port stalls.

The strict system roofline remains 20,992 cycles because the independent
64 B/cycle array-operand ingress floor has not been widened. This LLM-fork
broadcast rate is configurable with
`GOLEM_MATRIX_BROADCAST_BYTES_PER_CYCLE`; the reference tree has no equivalent
timed broadcast port and assigns no separate cycle cost to its functional WCP
matrix writes. At the 64 B/cycle default, one 16 KiB V-matrix multicast costs
`1 + ceil(16,384/64) + ceil(log2(64)) = 263` cycles, excluding the one-cycle WCP
boundary. Across 32 broadcasts, V LocalGM reads and that WCP boundary, the
current-interface PV matrix floor and measured count are both 8,512 cycles.

The measured end-to-end result is 4.943x the 20,992-cycle bound. Corrected
counter boundaries split the former 40,256-cycle aggregate into 8,512 PV matrix,
16,384 P-input, and 15,360 old-O restore cycles. The remaining PV compute and
output phases are 2,176 and 16,385 cycles. This is an accounting correction;
PV total and end-to-end latency remain 58,817 and 103,757 cycles.

## Current 256 B/cycle Broadcast + Scatter Architecture

This section supersedes the earlier 64 B/cycle operand-ingress bound and
measured table above. The SST model is the architectural contract; no RTL
interface is required.

```text
                         one worker, one Br64 x Bc64 tile

 LocalGM Q half (16 KiB) -- 256 B/cyc common-data broadcast --> matrix bank[0..63]
 LocalGM K tile (32 KiB) -- pack K[0..63, half*64:+64] -------+
                                                                  |
                         256 B/cyc aggregate scatter               |
                  beat = 64 FP32 destination lanes                 v
                    lane i -> input bank[i]                 array[0..63]
                                                                  |
                         two K64 launches, 64 arrays in parallel   v
                                                            S[64,64]
                                                                  |
                                                        16-lane SFU
                                                                  v
                                                            P[64,64]
                                                                  |
 LocalGM V slice (16 KiB) -- 256 B/cyc common-data broadcast --> matrix bank[0..63]
 LocalGM P tile (16 KiB) -- one 64-lane scatter -------------> input bank[0..63]
                                                                  |
                         P input remains resident for slice 1      |
                         two D64 launches, 64 arrays in parallel   v
                                                            O[64,128]
```

Broadcast and scatter use independent parameters even though both defaults are
256 B/cycle. Broadcast replicates one 16 KiB matrix to 64 sinks, so its modeled
service is `1 + ceil(16384/256) + log2(64) = 71 cycles`. Scatter transfers 64
different 64-element FP32 vectors, 16 KiB aggregate, so its service is
`1 + ceil(16384/256) = 65 cycles`. Aggregate means the 256-byte beat is split
into 64 FP32 lanes; it is not 256 B/cycle per lane and is not multiplied by 64.

For one tile, QK needs two Q broadcasts and two destination-specific K
scatters. PV needs two V broadcasts but only one P scatter because the P input
bank remains valid across both D64 output slices:

```text
matrix broadcast = 4 * 71                         = 284 cycles/tile
vector scatter   = (2 K + 1 P) * 65               = 195 cycles/tile
QK compute/read/write + SFU + PV compute           = 545 cycles/tile
dependency floor = 284 + 195 + 545                = 1,024 cycles/tile
16 key tiles                                           16,384 cycles
```

The unchanged HBM bounds are 13,927 cycles for aggregate bytes and 16,896
cycles for read-command issue. The current end-to-end theoretical resource
lower bound is therefore:

```text
max(16,384 worker dependency path,
    13,927 HBM aggregate traffic,
    16,896 HBM read-command issue)
= 16,896 cycles
```

This lower bound excludes NoC arbitration, command/control skew, and the
current serialized old-O restore and PV output read/write implementation.

The fixed QK1024 run (`Q=K=1024,D=128`) passes all 131,072 output checks and
all lifecycle contracts at 62,900 accelerator cycles (3.723x the lower bound):

| Critical-worker phase | Measured cycles |
|---|---:|
| Input movement | 4,375 |
| QK matrix program | 1,152 |
| QK input program | 2,176 |
| QK compute/readout | 3,408 |
| QK total | 6,736 |
| Softmax | 12,569 |
| PV matrix program | 2,368 |
| PV input program | 1,088 |
| PV old-O restore | 15,360 |
| PV compute | 2,176 |
| PV output read/write | 16,386 |
| PV total | 37,378 |
| End-to-end | 62,900 |

Per worker, matrix broadcast records 64 requests and 4,544 service cycles;
scatter records 48 requests, 786,432 bytes, 3,120 service cycles, 64 maximum
destinations, and zero rejects. The two largest remaining serialized phases are
PV output read/write and old-O restore, totaling 31,746 cycles.

## Grouped 256 B/cycle O Scatter/Gather

The sequential PV state machine no longer restores or drains one array at a
time. A D64 output slice is one 64-array group transfer:

```text
                         256 B/cycle aggregate O fabric
                                      |
 LocalGM row-major O cache -- scatter +--> accumulator[0..63] -- PV MAC
                                      |                              |
                                      +---- gather <-----------------+

 group payload = 64 arrays * 64 FP32 = 16 KiB
 group service = 1 + ceil(16 KiB / 256 B/cycle) = 65 cycles
```

The O tensor is row-major, so one D64 slice is strided in LocalGM. Each key
tile reads the complete 32 KiB O image through two 16 KiB LocalGM requests,
uses the image for both grouped restores and gathers, and writes it through two
16 KiB requests after slice 1. This preserves the final output DMA layout.

For QK1024, old O is restored for 15 noninitial key tiles and output is gathered
for all 16 tiles:

```text
restore = 15 * (2*2 LocalGM + 2*(1 WCP + 65 scatter)) = 2,040 cycles
output  = 16 * (2*(1 WCP + 65 gather) + 2*2 LocalGM) = 2,176 cycles
O total                                                     4,216 cycles
```

Adding this complete O path to the previous 16,384-cycle worker dependency
floor gives 20,600 cycles. It exceeds the 16,896-cycle HBM command floor, so
the updated complete end-to-end theoretical lower bound is 20,600 cycles.

The fixed QK1024 Ramulator2 run passes numerical and lifecycle verification at
44,790 accelerator cycles:

| Critical-worker phase | Before grouped O | Grouped O |
|---|---:|---:|
| Input movement | 4,375 | 13,654 |
| QK total | 6,736 | 6,732 |
| Softmax | 12,569 | 12,581 |
| PV matrix program | 2,368 | 2,368 |
| PV input program | 1,088 | 1,088 |
| PV old-O restore | 15,360 | 2,040 |
| PV compute | 2,176 | 2,176 |
| PV output read/write | 16,386 | 2,176 |
| PV total | 37,378 | 9,848 |
| End-to-end | 62,900 | 44,790 |

The critical worker reports 30 restore scatters, 32 output gathers, 1,015,808
bytes, 4,030 array-fabric service cycles, 64 maximum destinations, and zero
rejects. Faster PV exposes 10,173 cycles of K/V-prefetch wait, which explains
why input movement grows and the 27,530-cycle PV reduction yields an 18,110-cycle
end-to-end improvement.

## Shared Streaming K/V Supply

R28 removes the K/V duplication exposed by the grouped-O optimization. Each
32 KiB K or V tile is already split into two 16 KiB DMA chunks. At each shared
memory node, the first request for a chunk creates one HBM read; concurrent
requests join its subscriber list, and later requests reuse the completed
resident chunk. The chunk is then fanned out at 256 B/cycle. A node retains it
until all 16 workers have consumed it.

```text
16 worker requests for one K/V chunk
                 |
                 v
       shared memory-node MemNIC
          |                 |
     in-flight join    completed-cache hit
          +--------+--------+
                   |
          one 16 KiB HBM read
                   |
        256 B/cycle multicast
                   |
          16 worker LocalGMs
```

For 16 key tiles, K and V contain 1 MiB of unique data. Four striped nodes each
serve 256 KiB, or 16 chunks. One chunk occupies
`16 KiB / 256 B/cycle = 64 cycles`, so the parallel system supply floor is
`16 * 64 = 1,024 cycles`. This resource is hidden below the 20,600-cycle worker
dependency path and therefore does not increase the end-to-end lower bound.

The worker also tracks K and V readiness independently. K completion can start
QK while V remains in flight; V is checked only at the PV boundary. The active
descriptor is retained until tile completion so the two-pair buffer layout
cannot overwrite a live operand.

The final fixed QK1024 run passes backend, numerical, and lifecycle verification:

| Metric | R26 grouped O | R28 shared K/V |
|---|---:|---:|
| Input movement | 13,654 | 137 |
| QK | 6,732 | 6,736 |
| Softmax | 12,581 | 12,567 |
| PV | 9,848 | 9,848 |
| Exposed K/V wait | 10,173 | 0 |
| End-to-end | 44,790 | **31,197** |

Every data node reports 16 physical reads, 256 logical receivers, 256 KiB of
unique multicast data, and zero resident chunks at completion. Globally this
is 64 physical reads and 1 MiB of unique K/V data. The 31,197-cycle result is
10,597 cycles above the 20,600-cycle bound; softmax alone contributes 8,135 of
that gap and is the next fixed-QK1024 optimization target.

## Running

```bash
src/sst/elements/golem/tests/small/muticore_attention/run_flash_attention.sh \
  --queries 1024 --keys 1024 --head-dim 128
```
