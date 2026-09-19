# 8 QK/SFU + 8 PV attention cluster

This experiment changes only the worker split of `attention_cluster`:

```text
managers 0..3
   |-- QK/SFU columns 0..1 -> worker cores 4..11   (8 cores)
   `-- PV columns 2..3     -> worker cores 12..19  (8 cores)
```

The tensor shape, 64x64 arrays, QK `2x4` reuse window, PV `1x2` edge reuse
window, online softmax, Score/P FIFOs, 512 KiB V region in each PV core's
existing local SRAM, 256 B/cycle local-SRAM port, HBM/NoC parameters, and SST
completion boundary are shared with the 4:12 baseline.

The 8:8 path adds two PV pipeline contexts.  While context A computes, context
B prefetches the next 128 KiB V window into a different entry of the existing
four-entry V SRAM region.  Within a window, a second 16 KiB payload buffer reads
panel `i+1` from local SRAM while panel `i` is on the arrays.  The request queue
is expanded from 64 to 2048 entries.  The memory-node admission limit remains
128 x 4 KiB credits: testing 256 credits reduced blocked speculative requests
but regressed end-to-end time by 659 cycles because they competed with demand
traffic.  These changes do not alter the 4:12 runner;
`GOLEM_ATTENTION_PV_PANEL_PREFETCH=0` disables local panel lookahead, while the
credit and issue environment variables remain available for ablation runs.

Two additional policies are enabled by default. First, the SFU scheduler gives
priority to row contexts that have already completed more key tiles. This
finishes rows earlier and exposes complete P windows to PV instead of leaving
many rows partially complete. Second, each distinct 128 KiB V entry is fetched
from HBM once by a PV leader, divided into 16 KiB NoC packets, and distributed
through a binary tree to the other PV cores. A prefetch hint is sent as soon as
online softmax starts forming a P window, so V distribution overlaps the
remaining QK/SFU work. Receivers install the assembled entry in the existing
512 KiB per-core V SRAM; this does not add another cache.

The policies can be ablated independently with
`GOLEM_ATTENTION_WORKER_CLUSTER_ROW_PRIORITY=0` and
`GOLEM_ATTENTION_WORKER_CLUSTER_V_BROADCAST=0`.

An optional row-sticky dynamic PV allocator can be enabled with
`GOLEM_ATTENTION_WORKER_CLUSTER_DYNAMIC_PV=1`. Allocation waits until a row's
first complete P window is ready, chooses between the two PV cores attached to
the same manager, and keeps all four windows on that core so online PV
accumulation remains local. It is an experimental ablation and is disabled by
default because it does not improve this regular shape.

Each manager dispatches its 512 flattened GQA rows to two QK workers. Each QK
worker receives 256 rows (one query head for this GQA shape). A completed
64x256 P window is round-robin routed to one of the manager's two PV workers.

Run the functional schedule model:

```bash
baseline/attention_cluster_8qk_8pv/run.sh --no-numerical
```

Run SST for `Hq=4,Hkv=2,Sq=Skv=1024,Dh=128`:

```bash
GOLEM_MPI_RANKS=1 baseline/attention_cluster_8qk_8pv/run_sst.sh \
  --artifact-root /tmp/attention_cluster_8qk_8pv \
  --query-length 1024 --kv-length 1024 \
  --num-query-heads 4 --num-kv-heads 2 --head-dim 128
```

The report verifies 16 QK jobs, 256 PV windows, QK cores 4..11, and PV cores
12..19. The theoretical resource floors and measured comparison are generated
by `compare.py` after both SST runs complete.

The frozen stable result and its complete parameter manifest are archived in
`artifacts/golden/sst_result.json` and `artifacts/golden/manifest.json`.
Generated files under `artifacts/latest` are intentionally ignored.

## Default-shape comparison

Both variants below use the same build, tensor shape, bandwidths, local-SRAM
model, and SST timing boundary.

| split | theoretical resource floor | SST end-to-end | excess over floor |
|---|---:|---:|---:|
| 4 QK/SFU + 12 PV | 65,536 | 73,122 | 7,586 (11.6%) |
| 8 QK/SFU + 8 PV | 32,768 | 47,193 | 14,425 (44.0%) |

Before these pipeline changes, the 8:8 split took 81,397 cycles and was 2,354
cycles slower than 4:12.  Cross-window V prefetch, the prepared context, and
local panel double buffering first reduced it to 72,352 cycles.  Bypassing the
redundant matrix DMA for the P operand then reduces it to 53,434 cycles: another
18,918 cycles (26.15%), or 27,963 cycles (34.35%) against the original 8:8
implementation.  The full panel-pipeline change below reaches 48,756 cycles.
Row-completion priority then reaches 47,100 cycles. Enabling both row priority
and the V broadcast path gives 47,193 cycles while reducing V HBM traffic from
8.375 MiB to 1 MiB. The final default is 25,929 cycles faster than the corrected
4:12 path, a 1.549x speedup.

P is already delivered through the Score/P FIFO and held in the worker's
resident payload.  The old request path nevertheless fetched four 16 KiB P
tiles for every PV window.  Removing those unused requests eliminates 1,024
memory-node requests and 16 MiB of redundant off-chip traffic.  On V-cache-hit
windows, mean PV latency first falls from 1,847.3 to 800 cycles.

The PV path now uses one four-K-tile transaction for the whole 256-row V
window, and its local-SRAM panel lookahead crosses the two output-column reuse
groups.  This reduces a cache-hit PV window from 800 to 598 cycles: 528 compute
cycles plus one 66-cycle startup panel read and fixed completion cost.  The
final run observes 16 V misses and 240 hits; mean latency over all 256 PV
windows is 907.1 cycles.  End-to-end time falls from 53,434 to 48,756 cycles,
another 4,678 cycles (8.75%).  Against the original 81,397-cycle 8:8 path, the
combined reduction is 32,641 cycles (40.10%).

Against the previous 48,756-cycle baseline, the two-policy default saves 1,563
cycles (3.21%). The latency-only optimum is row priority by itself at 47,100
cycles, 93 cycles faster than the bandwidth-saving default. The V tree removes
7.375 MiB (88.1%) of off-chip V reads, but its on-chip distribution consumes
enough NoC capacity to offset 93 cycles of the scheduling gain. This is an
explicit bandwidth/latency tradeoff rather than an unreported regression.

The SST ablation separates the gains: cross-window V prefetch plus the prepared
context reduces 81,397 to 73,602 cycles (7,795 saved); local 16 KiB panel
double-buffering then reduces 73,602 to 72,352 cycles (another 1,250 saved).
Changing the request queue from 64 to 2048 does not change this workload's
cycle count, but prevents the dual-context design from being tied to the old
single-context queue limit.

The theoretical resource floor is still 32,768 cycles, set by the per-QK-core
SFU issue load. With both policies enabled, the measured QK/SFU production span
is 37,487 cycles; the schedule-aware floor including manager dispatch skew is
about 34,433 cycles. After the last QK job finishes, the slowest PV queue needs
another 9,706 cycles to drain. The remaining 14,425-cycle gap over the resource
floor is concentrated in SFU/QK scheduling gaps, NoC V/P contention, and static
PV-core queue imbalance rather than redundant off-chip V reads.

## Dynamic-PV experiment

Dynamic row-to-PV assignment was implemented and measured rather than enabled
on assumption. A global least-active-row allocator took 48,286 cycles: routing
P and V outside the producing manager's local pair added NoC contention and
made it 1,093 cycles slower than static mapping. Constraining allocation to the
same manager removed those long PV transactions and reached 47,890 cycles, but
was still 697 cycles slower than the 47,193-cycle static baseline.

For this shape every row contributes exactly four equal PV windows. Static
round-robin therefore gives every PV core exactly 32 windows. The dynamic
active-row heuristic instead produced 28--36 windows per core and paid an
allocation request/response before the first window. It moved the tail rather
than reducing it. Dynamic mode remains available for irregular or masked
workloads, but static mapping is the correct default for the regular
`Hq=4,Hkv=2,Sq=Skv=1024,Dh=128` comparison.

V prefetch is deliberately tied to the first online-softmax tile of each
window. Restricting hints to only the first window was measured at 51,174
cycles because later V entries returned to the demand path. Sending all hints
on the existing low-priority VN was measured at 47,337 cycles; keeping them on
the normal control VN is faster at 47,193 cycles. Neither rejected policy is
enabled in this baseline.

Reproduce the comparison after running both variants:

```bash
python3 baseline/attention_cluster_8qk_8pv/compare.py \
  --baseline-4qk-12pv /tmp/attention_cluster_4qk_12pv/sst_qk_bridge_result.json \
  --experiment-8qk-8pv /tmp/attention_cluster_8qk_8pv/sst_qk_bridge_result.json
```
