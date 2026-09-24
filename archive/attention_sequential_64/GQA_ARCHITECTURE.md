# Parallel GQA dataflow

This worktree models grouped-query attention with a fixed four-manager,
16-worker topology. Every worker still owns 64 arrays, and every issued QK or
PV operation uses all 64 arrays. GQA changes which tensors are shared and how
the rows are placed; it does not split one MAC into smaller array operations.

## Shape and ownership

`Hq` is the number of Query heads and `Hkv` is the number of K/V heads. The
supported group sizes are `Hq/Hkv = 1, 2, 4`. Query input is head-major
`[Hq,Sq,Dh]`; K and V are head-major `[Hkv,Skv,Dh]`. A K/V head is submitted as one
composite job, and its Query group is flattened into rows for manager
partitioning:

```text
KV head g
  K[g], V[g]  -------------------- shared by ---------------------+
                                                                  |
  Q[g*G + 0] -> worker rows    Q[g*G + 1] -> worker rows          |
  ...                         Q[g*G + G-1] -> worker rows         |
                                                                  v
                 each worker: QK(all 64 arrays) -> SFU -> PV(all 64 arrays)
```

The four managers partition each group spatially. A group of one uses four
workers per Query head, a group of two uses two workers per head, and a group
of four uses one worker per head. Thus heads in one group run concurrently,
while each worker keeps its full array width. Multiple K/V-group jobs are
submitted before the guest waits; a bounded worker FIFO starts the next group
as soon as the current worker context completes.

## Direct output placement

The worker does not produce a head-major temporary followed by a concatenation
copy. For a completed local row `r`, the output DMA destination is computed as

```text
O_node + ((query_row + r) * Hq + query_head) * Dh * sizeof(float)
```

This writes the final query-major layout `[Sq,Hq,Dh]` directly. The row DMA has a
16-row in-flight window because the modeled LocalGM DMA admission queue has a
finite depth; callbacks refill the window. This is a parallel 2-D scatter,
not a 64-row burst that can overrun the queue and not a serial post-copy.

## Sq=Skv=1024 cycle model

For `Sq=Skv=1024,Dh=128`, one single-head dependency chain has a lower bound of
`24,256 cycle` under the active 64-array, 256 B/cycle on-chip fabric and the
configured SFU pipeline. The fixed 16-worker topology gives the aggregate
arithmetic bound

```text
T_theory(Hq,Hkv) = Hq * 24,256 cycle
```

`Hkv` changes K/V traffic and sharing, but not the Query-side arithmetic work.
The measured direct-output results are:

| Hq | Hkv | lower bound | measured | gap | relative gap |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 24,256 | 27,430 | 3,174 | 13.09% |
| 4 | 1 | 97,024 | 106,939 | 9,915 | 10.22% |
| 4 | 4 | 97,024 | 109,137 | 12,113 | 12.49% |
| 8 | 2 | 194,048 | 212,098 | 18,050 | 9.30% |

All listed runs use MPI rank 4 and pass backend, numerical, lifecycle, and
placement checks. The remaining gap is mostly SFU pipeline admission, DMA/NoC
queueing, and manager/worker synchronization; it is not a missing output
concatenation phase.

## Reproduction

```bash
scripts/test_flash_attention.sh \
  --num-query-heads 4 --num-kv-heads 1 \
  --query-length 1024 --kv-length 1024 --head-dim 128 \
  --mpi-ranks 4
```

The legacy `--heads N` spelling remains an MHA compatibility alias for
`Hq=Hkv=N`.
