# Attention cluster baseline

This baseline preserves `reuse_window_flash_attention` and models a separate
worker-cluster dataflow for `Hq=4,Hkv=2,Sq=Skv=1024,Dh=128`.

```text
                       bounded score FIFO
  group 0 workers       (two windows/core)
  cores 4..7          +---------------------+
 Q/K -> 2x4 QK WCP -> | online softmax SFU  | ---- P + alpha ----+
        [128,256]      +---------------------+                    |
                                                                  v
 group 1..3 workers       bounded P FIFO                    cores 8..19
 V -- broadcast/prefetch ------------------------------> 2x2 PV WCP
                                                         [128,256]x[256,128]
                                                                  |
                                                        resident online O
```

## Ownership

- Four QK workers compute QK and online softmax only.
- Twelve PV workers compute PV only.
- A 64-row Query block is permanently assigned to one PV worker. All four
  256-key windows for those rows therefore update the same local online output.
- Row blocks are round-robin distributed over all twelve PV workers.

## Reuse and broadcast

QK uses the reference generic-GEMM `2x4` output window:

```text
Q [128,128] x K^T [128,256] -> S [128,256]
Q panel reuse = 4; K panel reuse = 2; reduction tiles = 2
```

PV batches two ready 64-row probability blocks when possible and uses `2x2`:

```text
P [128,256] x V [256,128] -> partial O [128,128]
P panel reuse = 2; V panel reuse = 2; reduction tiles = 4
```

Q is prefetched once into the owning QK worker. Each 256-key K window is read
once and broadcast to four QK workers; each V window is read once and broadcast
to twelve PV workers. K and V use ping-pong buffers. Score and P use bounded
on-chip FIFOs, and neither is written to HBM.

The scheduler uses key-major order so a broadcast K/V window remains resident
while all relevant row windows consume it. QK, SFU, P transfer, and PV are
independent resources and overlap subject to their FIFO dependencies.

## Run

```bash
baseline/attention_cluster/run.sh
```

Run the physical worker split through SST:

```bash
GOLEM_MPI_RANKS=1 baseline/attention_cluster/run_sst.sh \
  --artifact-root /tmp/attention_cluster_sst \
  --query-length 1024 --kv-length 1024 \
  --num-query-heads 4 --num-kv-heads 2 --head-dim 128
```

The SST bridge maps QK/SFU to cores 4-7 and PV to cores 8-19. Each completed
`[64,256]` P window and its 64 online-rescale factors travel over the modeled
NoC. The receiving PV core executes generic GEMM
`[64,256] x [256,128]`, using a `1x2` edge reuse window. The report measures
from the first QK WCP launch through the final PV completion ACK.

The SST path streams QK output tiles directly into the SFU: it does not wait
for the complete `[512,1024]` score matrix. A four-tile probability window is
sent as soon as online softmax finishes it, so QK, SFU, NoC transfer, and PV
execute concurrently. The runner defaults to the reference fabric parameters
(`512GB/s` links, `1024GB/s` crossbar, `256GB/s` GM endpoint, and `512GB/s`
directory highlink).

The QK/SFU bridge splits each `64x64` score tile into sixteen `4x64` row slices.
The existing 64 SFU row contexts therefore run up to sixteen independent
four-row tiles at once. This exposes ready work from more tiles to the separate
Max, Exp, and Normalize pipelines. Each slice keeps its own next-KV index,
online scale, and bounded `4x256` P state. Sixteen completed row slices are
assembled back into the unchanged `64x256` PV interface. Total live Score/P
storage is unchanged from the former `4 slots x 16 rows` organization.

Score and P use independent sixteen-entry slot allocators. A row slice may consume
Score slot `i` and produce into P slot `j`; the Score slot is released as soon
as SFU processing completes, while the P slot remains owned until RoCC drains
the normalized tile. Online-softmax state is keyed by `(job,row)`, so a
physical four-row slice can preserve multiple logical row contexts. The scheduler
immediately lends a physical slice whose next Score tile is unavailable to a
logical row whose next tile is ready. This ready-first policy fills QK producer
gaps without moving online-softmax state out of its logical row context.
`[ATTENTION_FIFO_DECOUPLED]` reports unique-cycle Score/P allocation stalls and
the number of logical row-context switches.

QK output is written directly to the Score FIFO. The SFU reads each 64-element
score row once, retains the row across scale/max, exp/sum, and normalize, then
writes the normalized row directly to the P FIFO. RoCC reads P from that FIFO
for PV-window assembly. Consequently score and P no longer make a round trip
through local GM. Attention scale remains in the SFU Max-stage dataflow; this
baseline does not pre-scale Q. The Score and P FIFO ports retain their existing
default `64 B/cycle` bandwidth and one-cycle base latency.

P-window dispatch is round-robin across the three PV lanes and paced at the
modeled `256 B/cycle` GM endpoint rate; a temporarily rejected PV WCP window
remains queued and is retried.

The receiving PV WCP accepts the assembled P window as a resident matrix
operand. It no longer stages P through the PV core's local GM. Each PV core
reserves 512 KiB of its existing 2 MiB local GM SRAM as four 128 KiB V-window
entries. A miss DMA lands the complete `[256,128]` window directly in the
selected entry; an exact source/layout match skips only the off-chip DMA and
reads V from the same SRAM address. The cluster runner fixes the modeled
local-GM port at 256 B/cycle.

When another KV-head dispatch is queued, the idle QK WCP also computes it into
an ahead context while the current job drains through SFU/PV. Once the current
job releases all Score/P row contexts, the ahead job can enter SFU while the
old job continues P/NoC dispatch and PV completion in a separate draining
context. Completion is still reported only after the corresponding final PV
ACK; the optimization does not weaken manager or software wait semantics.

Outputs under `baseline/attention_cluster/artifacts/latest/`:

- `result.json`: role assignment, modeled cycles, traffic, reuse and prefetch;
- `schedule.json`: dependency-respecting event timeline;
- `output_query_major.bin`: verified FP32 `[Sq,Hq,Dh]` output.

`run.sh` remains the numerical functional/event model. `run_sst.sh` is the
cycle-accurate SST bridge. Its current timing boundary stops at final PV ACK;
the PV result is resident at the worker and the final output HBM DMA is not yet
included. Consequently the SST runner intentionally reports cycle completion
without running the old HBM-output numerical checker.

For the default shape, each SFU has 65,536 scale/EXP/normalize issues, giving a
65,536-cycle initiation-interval floor. The modeled QK WCP compute floor is
33,792 cycles per critical core, and the busiest PV worker has a 12,672-cycle
compute floor. The current PV path has 96 compulsory V-window misses and 160
cache hits, so it reads 12 MiB of V panels. Its aggregate four-node
`1280 B/cycle` HBM floor is 9,831 cycles. Because these resources overlap, the
optimistic end-to-end resource floor remains 65,536 cycles.

The verified `Sq=Skv=1024, Hq=4, Hkv=2, Dh=128` SST run completes in 79,043
cycles (1.21x the resource floor). This includes 2,048 timed 16 KiB V-panel
reads (32 MiB total) from the existing per-core local GM at 256 B/cycle. The
older 73,902-cycle result used the same 96-miss/160-hit policy but bypassed the
local-SRAM port on a hit, making it 5,141 cycles too optimistic. Relative to the
former 88,240-cycle implementation, the fully timed path saves 9,197 cycles
(10.4%); relative to the earlier 103,343-cycle baseline it saves 24,300 cycles
(23.5%). All eight Attention jobs and all 256 PV windows complete through the
modeled SST components.
