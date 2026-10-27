# DMA Retry Root-Cause Archive (2026-03-28)

## Scope

- Entry: `tests/run_noc_dma_pipeline.sh`
- Default config with `GOLEM_DMA_READ_RETRY_TICKS=32`
- Reference run: `tests/artifacts/stats/overlap0/run_20260328_171355_879601`

## Key Conclusion

- Current DMA retries are primarily caused by first-wave tail latency, not by NoC saturation or HBM backend saturation.
- Under the default mapping, each data node receives a synchronized first-wave burst from 4 workers.
- With `dma_read_max_inflight=8` and `dma_burst_bytes=512`, each node can see up to 32 in-flight DMA read chunks in the first wave.
- The configured retry window is about `32 * 30ns = 960ns`, which is at or below the expected tail service time for that burst.

## Default Parameters Used in Analysis

- `GOLEM_DMA_MAX_INFLIGHT=8` in `tests/configs/20_dma.env`
- `GOLEM_DMA_READ_RETRY_TICKS=32` in `tests/configs/20_dma.env`
- `GOLEM_DMA_BURST_BYTES=512` in `tests/configs/20_dma.env`
- `GOLEM_GROUP_MAX_INFLIGHT_PER_NODE=4` in `tests/configs/30_network.env`
- GM network link bandwidth `50GB/s` in `tests/architecture/cpu_builder.py`
- Directory highlink bandwidth `25GB/s` in `tests/architecture/ncores_selfcom_dma_ctrl.py`
- `GOLEM_GM_TRANS_LATENCY=30ns` default in `tests/architecture/cpu_builder.py`
- CPU clock `2.3GHz` in `tests/architecture/cpu_builder.py`

## First-Wave Concurrency Accounting

- Per worker, first `k_tile=0` loads:
  - A tile: `64 x 64 x 4B = 16384B`
  - B block: `64 x 4 x 4B = 1024B`
  - Total first batch: `17408B`
- With `512B` burst size:
  - A chunks: `32`
  - B chunks: `2`
  - Total chunks per worker for first batch: `34`
- With `dma_read_max_inflight=8`, each worker initially exposes up to `8` chunks to the network.
- Default mapping places `4` workers on the same target data node concurrently.
- Therefore, first-wave concurrency per node is up to `4 x 8 = 32` in-flight chunks.

## Retry Window

- Retry tick is not 1 CPU cycle.
- In this codebase, `dma_retry_tick_cpu_cycles` is derived from `GOLEM_GM_TRANS_LATENCY`.
- With `30ns` GM latency and `32` retry ticks, retry window is approximately `960ns`.

## Theoretical Serialization Budget

Approximate per-chunk wire cost:

- request metadata at `25GB/s`: about `0.96ns`
- reply payload (`512B` data plus metadata) at `25GB/s`: about `21.12ns`
- reply payload at GM side `50GB/s`: about `10.56ns`

For a first-wave node burst of `32` chunks:

- directory-side reply serialization: `32 x 21.12ns = 675.84ns`
- source GM-side reply serialization: `32 x 10.56ns = 337.92ns`
- request-side serialization: about `30.72ns`
- NoC average round trip: about `58ns`
- backend average read: about `10ns`
- short fixed links plus arbitration margin: tens of ns

Conservative total tail budget is around `1.1us`, which already exceeds the configured `960ns` retry window.

## Corroborating Log Evidence

- Fast chunk example:
  - issue `src_pa=0x8033e00 -> gm_dst=0x405e00`
  - completion after about `96ns`
- Slow chunk example from same worker wave:
  - issue `src_pa=0x8033a00 -> gm_dst=0x405a00`
  - completion after about `1085ns`
  - exceeds retry window, so retry is expected
- Worse tail example:
  - `src_pa=0x8003a00 -> gm_dst=0x105a00`
  - completion after about `4049ns`
  - multiple retries observed before final completion

Relevant files:

- `tests/artifacts/logs/test_default_run_20260328_171355_879601.log`
- `tests/artifacts/stats/overlap0/run_20260328_171355_879601/dma_summary.csv`
- `tests/artifacts/stats/overlap0/run_20260328_171355_879601/noc_summary.csv`
- `tests/artifacts/stats/overlap0/run_20260328_171355_879601/memory_queue_summary.csv`

## Interpretation

- The dominant issue is not total NoC bandwidth.
- The dominant issue is first-wave burst concentration at each data node combined with small DMA chunks and a retry threshold that is lower than normal tail service time.
- Therefore, many retries are normal consequences of the current configuration rather than evidence of packet loss or protocol failure.

## Most Relevant Knobs Going Forward

Priority order for follow-up experiments:

1. Increase memory-side directory highlink bandwidth from `25GB/s`.
2. Reduce per-node burst concurrency by lowering `GOLEM_GROUP_MAX_INFLIGHT_PER_NODE`.
3. Increase `GOLEM_DMA_BURST_BYTES` to reduce chunk count and reply count.

## Bandwidth Role Map

The current default architecture has three bandwidth layers: source-side injection, NoC fabric transport, and memory-side injection/ejection.

```text
                    source / injection                              fabric                                  sink / memory side

   [CPU core]
       |
       v
   [L1/L2 cache]
       |
       |  L2 MemNIC = 50GB/s
       |  tests/architecture/cpu_builder.py:423
       v
  ---------------------> [ Mesh Router / Mesh Links ] ---------------------> [ DirectoryController highlink ]
                         link_bw = 100GB/s                                  dir_hi network_bw = 25GB/s
                         xbar_bw = 100GB/s                                  tests/architecture/ncores_selfcom_dma_ctrl.py:312
                         tests/configs/30_network.env:13
                         tests/configs/30_network.env:14
                                                                                         |
                                                                                         v
                                                                                [DirectoryController]
                                                                                         |
                                                                                         v
                                                                                  [MemController]
                                                                                         |
                                                                                         v
                                                                                  [DRAMSim3 / HBM]
```

In parallel, the accelerator-local source path is:

```text
   [RoCC / GlobalMemory]
          |
          |  GlobalMemory link_bw = 50GB/s
          |  tests/architecture/cpu_builder.py:534
          v
  ---------------------> [ Mesh Router / Mesh Links ] ---------------------> [ DirectoryController highlink ]
                         link_bw = 100GB/s                                  dir_hi network_bw = 25GB/s
                         xbar_bw = 100GB/s
```

Condensed view:

| Bandwidth item | Default value | Architectural role | File reference |
|---|---:|---|---|
| `L2 MemNIC network_bw` | `50GB/s` | source-side injection | `tests/architecture/cpu_builder.py:423` |
| `GlobalMemory link_bw` | `50GB/s` | source-side injection | `tests/architecture/cpu_builder.py:534` |
| `NoC link_bw` | `100GB/s` | fabric link transport | `tests/configs/30_network.env:13` |
| `NoC xbar_bw` | `100GB/s` | fabric router crossbar | `tests/configs/30_network.env:14` |
| `dir_hi network_bw` | `25GB/s` | memory-side ingress/egress | `tests/architecture/ncores_selfcom_dma_ctrl.py:312` |

Interpretation:

- source side (`50GB/s`) is wider than memory-side service (`25GB/s`)
- fabric (`100GB/s`) is wider than both endpoint classes
- therefore the first likely concentration point is the memory-node network port rather than the mesh core fabric

## Read Request vs Read Completion Sensitivity

The read path is asymmetric. The request packet is small, but the read-completion packet carries the data payload. As a result, the two phases do not stress the same links equally.

### Read Request Path

Conceptually:

```text
[worker core / GM]
  -> GlobalMemory local network port
  -> mesh router / mesh links
  -> DirectoryController highlink
  -> DirectoryController
  -> MemController
  -> DRAM backend request service
```

Sensitivity by bandwidth:

- `GlobalMemory link_bw`
  - affects how fast requests can be injected from each worker into the NoC
- `NoC link_bw` and `NoC xbar_bw`
  - affect transport through the mesh
- `dir_hi network_bw`
  - affects how fast requests can enter the memory node

But for this workload, request packets are relatively small. Therefore the request path is usually not the dominant contributor to long-tail latency unless the network is already badly congested.

### Read Completion Path

Conceptually:

```text
[DRAM backend]
  -> MemController
  -> DirectoryController
  -> DirectoryController highlink
  -> mesh router / mesh links
  -> GlobalMemory local network port
  -> local GM writeback
```

Sensitivity by bandwidth:

- `dir_hi network_bw`
  - directly limits how quickly full data-bearing completion packets can leave a memory node
- `NoC link_bw` and `NoC xbar_bw`
  - determine whether many simultaneous completion packets can traverse the mesh without added queuing
- `GlobalMemory link_bw`
  - determines how quickly a core can absorb completion packets back into its local GM window

### Why Completion Is More Sensitive in This Workload

For a DMA read chunk:

- request size is metadata only
- completion size includes the actual returned data payload

So the returned packet is much larger than the outgoing request packet. This means:

- request-side serialization cost is modest
- completion-side serialization cost dominates

Architecturally, this makes the memory-side ejection bandwidth (`dir_hi network_bw`) more critical than request injection bandwidth for a read-heavy tiled GEMM flow.

### Practical Implication for Current Default Config

With current defaults:

- request injection from the source side can happen at `50GB/s`
- the mesh can transport at `100GB/s`
- but each memory node ejects reply traffic at only `25GB/s`

Therefore the most sensitive phase is usually:

- `HBM node -> dir_hi -> NoC` on read completion

rather than:

- `worker GM -> NoC -> HBM node` on read request

This is why a read-heavy workload with many small chunks can show long completion tails even when overall NoC utilization and backend memory queue depth are still modest.

## Notes About Recent Protocol Patch

- Recent DMA protocol changes added `txnId`, response-VN separation, and duplicate-response caches in `globalmemory`.
- Those changes did not materially change the current retry profile.
- This indicates the hot-path bottleneck is still first-wave tail latency rather than the protocol bug class that was targeted.
