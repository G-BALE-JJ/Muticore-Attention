# Projection pipeline and shared input blocks

This change overlaps input-pair reads, inactive-bank input scatter, next-row
partial reads, and partial stores with the existing two-bank projection
computation. It also traverses input blocks before Q/K/V heads so that all
48 heads use the same staged normalized input block. The default flags are
`GOLEM_PROJECTION_PIPELINE=1` and `GOLEM_PROJECTION_SHARED_INPUT=1`.
Both apply only to the paired D64 path. D128 and row-major fallbacks retain
their traversal and timing.

## Measured results

The comparison reference is the paired-weight implementation at `adb912f`,
documented in [LLAMA_PROJECTION_PAIR_REUSE.md](LLAMA_PROJECTION_PAIR_REUSE.md).
The shape is Hq=32, Hkv=8, D=64, hidden=2048 with synthetic deterministic
FP16 inputs, gamma, and weights (seed 1742). The measured endpoint is the
Attention O writeback; Wo, residual, MLP, and a full decoder are outside scope.

| S | Paired reference E2E | Pipeline only E2E | Pipeline + shared input E2E | Saved E2E cycles | E2E reduction | New projection cycles |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 3,848,933 | 3,030,330 | 2,820,198 | 1,028,735 | 26.7% | 2,380,859 (30.2% lower) |
| 2048 | 7,951,808 | 6,303,320 | 6,075,613 | 1,876,195 | 23.6% | 4,754,102 (28.3% lower) |

At S1024, pipelining alone saves 818,603 E2E cycles. Sharing the input block
saves another 210,132 cycles. At S2048, these savings are 1,648,488 and
227,707 cycles respectively. The incremental shared-input measurements
include its extra weight DMA traffic and changes in contention.

All four nodes' RMSNorm, raw Q/K/V, and Q/K/V panels match the paired reference
byte for byte for both optimized and pipeline-only runs at both lengths.
All numerical, weight-image, layout, MPI, and QK/SFU/PV work checks pass.
The final O maximum absolute error is 0.0001220703125 at both lengths, with
the prior numerical tolerances retained. S1024 was repeated using a different
host CPU binding and completed in exactly 2,820,198 cycles. Its four numerical
and performance reports are identical; the MPI partition report also matches
when artifact paths are normalized. Source, simulator binary, and generated
input hashes match across repetitions. Hardware timing overrides and input
hashes match the paired reference at both lengths.

The 92 small contract/report tests and 10 baseline/runner tests pass. D64
hidden256/Hq4/Hkv2 decreases from 140,252 to 119,708 cycles; disabling both
flags reproduces 140,252 cycles. D128 hidden256/Hq2/Hkv1 remains at 161,566.
Compact results and provenance are frozen in `llama_projection_pipeline.json`.

## Scheduling and ownership

The first tile in a pair computes using bank 0 while preparing bank 1 input.
The second computes using bank 1 while preparing the next row group's bank 0
input. A scatter is issued only to the inactive bank, and its row/tile tag
must match before the controller consumes it. The first row of each pair
still programs both matrices normally. No next-pair matrix overwrites a bank
in use, and no additional array banks or output contexts are introduced.

A bounded prefetch holds exactly one future 16-row input pair (4 KiB) and one
future partial (2 KiB). Reads use timed GM requests, share the existing read
port and queue, split to the API request limit, and retry queue rejection.
The controller waits if a prefetch is not ready. Partial reads cannot race a
pending store to the same address.

After the current output read completes, a separate 2 KiB snapshot enters a
single bounded partial-store slot. The next row can proceed while that store
uses the timed GM write port. A second store waits for the slot to drain.
Completion waits for outstanding stores and output DMA. Restoration to the
shared array output remains strictly after the preceding output read and
before the next accumulation launch. FP16 accumulation order is preserved.
The additional 8 KiB transfer snapshots are reserved against the existing
local-GM capacity check, excluding the 64-byte DMA status tail. If this
budget does not fit, pipelining is disabled.

## Traffic and timing evidence

| Per manager | Paired S1024 | New S1024 | Paired S2048 | New S2048 |
| --- | ---: | ---: | ---: | ---: |
| Input blocks loaded from HBM | 48 | 1 | 96 | 2 |
| Input payload from HBM | 48 MiB | 1 MiB | 96 MiB | 2 MiB |
| Weight payload from HBM | 12 MiB | 12 MiB | 12 MiB | 24 MiB |
| Weight programs | 1,536 | 1,536 | 3,072 | 3,072 |
| Projection local-GM reads | 82.5 MiB | 82.5 MiB | 165 MiB | 165 MiB |
| Projection partial writes | 22.5 MiB | 22.5 MiB | 45 MiB | 45 MiB |

Sharing inputs changes the traversal, so S2048 reloads weights for its second
input block. This cost is included in the measured improvement. Projection
local bytes and arithmetic work remain identical. The report independently
checks input blocks, weight loads/programs/reuses, prefetch counts, and SRAM
bytes against the shape.

For S1024 manager 0, serial local-read residence drops from 576,000 to 87,840
cycles, and input-scatter residence drops from 270,336 to 39,936. The new
background read state is active for 477,360 cycles and the store state for
115,200 cycles. These background counters overlap computation and are not
exclusive SRAM-port utilization or additional E2E time. A zero serial
`local_gm.write_cycles` means writes moved to the background state, not that
writes became free. `PROJECTION_PIPELINE` records these counters and queue
retry counts separately.

Array launch/compute residence remains 1,671,168 cycles at S1024 and
3,342,336 at S2048. Full-width array compute remains 66 cycles, with 1 MAC
per CU per cycle and pipeline depth 2 at 1 GHz. GM remains 2 MiB with one
read port, one write port, 256 B/cycle, base latency 1, maximum request 4096,
and queue depth 256. SFU, NoC, HBM, and operand/output transfer timing are
retained. The gains come from overlap and avoided input DMA.

## Backpressure boundary and remaining bottlenecks

A separate projection stress check uses queue depth 2, 16 B/cycle, and
request limits 128/1024/4096 with S2048/Hq4/Hkv2/D64. All managers complete
projection with correct byte counts and numerical/layout checks; every
projection region matches a disabled-pipeline control byte for byte.
Retries are recorded in raw logs. These are projection-only stress results.

The complete Attention chain fails under this queue-depth-2 profile in both
the optimized run and the disabled control. The existing
`readAttentionRopeTableLocal` submits an entire batch and treats a rejected
request as failure instead of retrying. This is a remaining Attention
backpressure limitation; none of these stress runs are counted as passing
E2E measurements. Default-profile E2E runs pass.

Array computation now occupies about 70% of projection residence. S1024
manager 0 still spends 204,667 phase cycles in weight DMA/control, 126,720
restoring shared output, and 135,168 reading output. Those output operations
cannot safely overlap the next computation with the current shared output
register. Reducing actual array compute latency or removing these shared
output round trips requires a separately measured hardware-capacity or
throughput change. Prefetching future weight DMA into bounded scratch is a
possible next software scheduling experiment.

## Reproduce

```bash
source scripts/env_local_install.sh
scripts/build_and_install_local.sh --no-autogen --jobs 8
python3 baseline/attention_cluster_8qk_8pv/run_llama_projection.py \
  --artifact-root results/fresh_projection_pipeline \
  --reference-root results/llama_projection_pair_final
```

The reference argument is optional. With it, the runner checks matching
inputs and byte-exact RMSNorm/QKV/panels and writes `reference_comparison.json`
with measured E2E/projection savings. Every run archives inputs, resolved
config, source/binary hashes, dirty-tree evidence, raw logs, and reports.
Use a fresh artifact root to retain prior evidence.

For the ablation, prefix the runner with `GOLEM_PROJECTION_SHARED_INPUT=0`.
To reproduce the paired reference scheduling with the new binary, set both
`GOLEM_PROJECTION_PIPELINE=0 GOLEM_PROJECTION_SHARED_INPUT=0`.
Independent repeats used `GOLEM_MPI_ARGS='--bind-to core --cpu-set 8-11'`;
the primary and ablation runs used CPU sets 0-3 and 4-7.
