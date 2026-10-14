# Projection transfer bandwidth audit

> **架构纠正（2026-10-10）：** 本文的历史投影/E2E 测量或重叠估计使用了 manager 本地 arrays，不符合「manager 只控制、16 个 worker 拥有阵列」的目标拓扑，不能作为该架构的性能证据。新的 worker 执行基准见 [WORKER_PROJECTION_ARCHITECTURE_FIX.md](WORKER_PROJECTION_ARCHITECTURE_FIX.md)。

This audit covers the current best fixed-resource workload: S1024,
Hq32/Hkv8/D64/hidden2048, FP16, four projection managers, 64 arrays per manager,
1 GHz CPU/array clocks, fused RMSNorm, no RoPE fusion, no raw audit materialization,
C-buffer partial storage and next-pair input prefetch.

The request count is determined by traversal/materialization, not bandwidth.
There is no evidence that projection grouped output transfers accidentally use
the generic 64 B/cycle array-buffer rate. They use the dedicated class parameter
`GOLEM_OUTPUT_SCATTER_GATHER_BYTES_PER_CYCLE=256`. Whether these rates and shared
ports accurately represent physical RTL is a separate hardware-model assumption;
this audit verifies the executable configuration and timing formulas.

## Actual paths and rates

At 1 GHz, 256 B/cycle is 256 GB/s (decimal units). Each figure below is per
manager/component, not a rate multiplied by its 64 destination arrays.

| Path | Rate used by the model | Port/latency semantics |
| --- | ---: | --- |
| Local GM reads | 256 B/cycle | One read port; each request occupies it for `1 + ceil(bytes/256)` cycles |
| Local GM writes / DMA landing | 256 B/cycle | One independent write port; same service formula |
| GM/WCP snapshot -> array weight broadcast | 256 B/cycle ingress | `1 + ceil(bytes/256) + ceil(log2(fanout))`; shared array-buffer port |
| Input scatter to arrays | 256 B/cycle aggregate | `1 + ceil(total_bytes/256)`; shared array-buffer port |
| Grouped array output read and restore | 256 B/cycle aggregate | `1 + ceil(total_bytes/256)`; same shared array-buffer port |
| C-buffer partial reads | 256 B/cycle | Independent read schedule; transfer occupancy plus one-cycle completion latency |
| C-buffer partial writes | 256 B/cycle | Independent write schedule; reads wait for matching writes to become ready |
| Generic single-array buffer transfers | 64 B/cycle | This is not the ProjectionOutput grouped path |
| Attention near-array output path | 512 B/cycle default | This is not the ProjectionOutput grouped path |

Weight broadcast, input scatter and grouped output transfers all enter
`enqueueModeledBufferTransfer`, with `GOLEM_ARRAY_BUFFER_PORTS=1`. Their class
bandwidths differ from the generic bandwidth, but they share port scheduling.
A group read can therefore wait for an input scatter or weight broadcast.
Increasing output bandwidth alone would change a modeled resource; it would
not remove these requests or prove a scheduling improvement.

External DMA traverses separate network and DRAM resources. The baseline wrapper
sets GM link bandwidth to 256 GB/s, NoC link bandwidth to 512 GB/s and crossbar
bandwidth to 1024 GB/s. HBM uses Ramulator2's `hbm2e_2500.yaml`, with a 1.25 GHz
memory-controller clock. Those network rates are not an HBM bandwidth measurement.
DRAM bank/channel timing, request scheduling, network contention and GM landing
all affect achieved DMA throughput. In particular, a local C-buffer partial
round trip is not an HBM round trip.

## Reconcile rates with actual counters

For 64 arrays, one grouped FP16 output transfer is
`64 arrays * 64 elements * 2 bytes = 8192 bytes`.
The configured 256 B/cycle grouped path takes `1 + 8192/256 = 33` service cycles.

The archived S1024 manager-0 array counters are:

| Traffic | Accepted requests | Payload bytes | Modeled service cycles | Cycles/request |
| --- | ---: | ---: | ---: | ---: |
| Weight broadcast | 1,536 | 12,582,912 ingress | 59,904 | 39 |
| Input scatter | 6,144 | 50,331,648 | 202,752 | 33 |
| Output gather + restore | 3,072 + 2,880 | 48,758,784 | 196,416 | 33 |

For the broadcast, `39 = 1 + 32 + 6 tree stages`. Its 64 sink copies total
768 MiB, but only 12 MiB is ingress. It would be incorrect to count all sink
copies as repeated HBM weight loads.

These resource-service counters differ from foreground controller residence:
output read is 163,920 ticks and restore including dispatch is 103,680 ticks.
Together, 267,600 controller ticks exceed the 196,416 resource service cycles
by 71,184 ticks. Shared-port queueing, WCP dispatch and controller transitions
contribute to the difference. These overlapping resource/controller metrics
must not be added into one E2E total.

## Why so many partial transfers?

Each manager handles 256 rows across 48 Q/K/V heads. There are four 64-row groups
per head and 32 hidden tiles (16 pairs). Thus:

- Output groups: `48 * 4 = 192`.
- Output gathers: `192 * 16 = 3072`.
- Partial gathers/stores: `192 * 15 = 2880`.
- Final output gathers: `192`.
- Partial restores: `2880`.

The paired traversal keeps two weight tiles resident while processing all row
groups. Between pairs, partials must survive reuse of the single output context
for another row group. They are rounded to FP16 and stored in the existing
C-buffer, then restored before continuing accumulation.

Per manager, partial readout, C-buffer stores, C-buffer reads and array restore
are each 22.5 MiB. Final output readout is another 1.5 MiB. Counts and bytes
would remain the same if the configured bandwidth doubled; only service time
would change. Reducing them requires a different traversal/lifetime plan, with
weight-programming and rounding tradeoffs, rather than correcting a rate.

## A bandwidth-model assumption that matters

The strided GM input gather reads two adjacent tiles per row: 256 bytes per
request, 64 requests per group. The GM port remains occupied for the entire
`1 + ceil(256/256) = 2` cycles. Consequently the effective sustained payload rate
for this request size is **128 B/cycle**, despite the configured 256 B/cycle.
A 4096-byte request takes 17 cycles, or approximately 241 B/cycle before queueing.

This is a consequence of the current non-pipelined per-request port occupancy,
not evidence that the environment variable was ignored. If physical SRAM can
accept a new 256-byte request every cycle while returning it after two cycles,
this occupancy model would be conservative and would need a separately justified
model correction. No latency, rate, port count or occupancy rule was changed in
the partial scheduling experiment.

## Evidence locations

- Rates and parameter mapping: `tests/architecture/cpu_builder.py`.
- GM timing/port release: `globalmemory/globalmemory.cc`.
- Group transfer formulas and shared scheduling: `array/computeArray.h`.
- ProjectionOutput selection: `array/mvmComputeArray.h`.
- C-buffer timing and read-after-write readiness: `workercmdproc/workercmdproc.h`.
- Measured configuration/counters: `results/projection_schedule/wrap_prefetch/s1024`.

All source paths above are relative to `src/sst/elements/golem`.

A follow-up distinguishes queue acceptance, port initiation and completion,
with an executable source probe and hardware evidence limits:
[`GM_TIMING_CONTRACT.md`](GM_TIMING_CONTRACT.md). The production model remains unchanged.
