# FlashAttention

This directory contains the active local FlashAttention workload. One unified
runner accepts the tensor shape and MPI count as terminal parameters. Its
default is `B1,H1,Q1024,K1024,D128`.

Run the current verified default configuration:

```bash
scripts/test_flash_attention.sh
```

Run the same shape with query-block MPI partitioning:

```bash
scripts/test_flash_attention.sh --mpi-ranks 2
scripts/test_flash_attention.sh --mpi-ranks 4
```

Select a larger shape directly instead of choosing an E-numbered test:

```bash
scripts/test_flash_attention.sh --queries 2048 --keys 2048 --head-dim 128
scripts/test_flash_attention.sh --queries 2048 --keys 2048 --head-dim 128 --mpi-ranks 2
scripts/test_flash_attention.sh --queries 2048 --keys 2048 --head-dim 128 --mpi-ranks 4
```

An optional larger pressure run uses the same interface:

```bash
scripts/test_flash_attention.sh --queries 4096 --keys 4096 --head-dim 128 --timeout 28800
```

The active path uses the locally built SST/Golem library and the RISC-V guest
binary built by the local `Makefile`. MPI supports 2 or 4 ranks. Each manager
and its four workers remain colocated, so the four query bands are assigned
two-per-rank or one-per-rank.

QK and PV use the shared GEMM/WCP compute interface by default. QK retains its
full-width array mapping. PV uses the same WCP path with 32 active input
columns, so compact probability rows and V panels occupy and compute only the
valid K dimension instead of paying the full physical-array-width latency.
Attention retains its existing transpose, online Softmax, KV buffering,
tiling, accumulation order, and dependencies. Use `--direct-gemm` with
`run_fused_attention_scale.sh` only for regression comparison.

QK input programming is overlapped with array execution by default:
`--qk-early-compute` starts an array GEMM as soon as that array's input
programming callback completes, while the remaining arrays continue to be
programmed. The transposed dataflow applies this to the first panel and keeps
later panels on the existing reuse path. Disable it only for an apples-to-apples
control with `--no-qk-early-compute`; numerical ordering and output collection
remain unchanged. The RESULT report includes the number of arrays started by
this mechanism.

The shared path models WCP control timing independently of array transfer and
compute latency. Defaults and environment overrides are:

- `GOLEM_WCP_GEMM_PROXY_QUEUE_DEPTH=32`;
- `GOLEM_WCP_GEMM_PROXY_ISSUE_WIDTH=1`;
- `GOLEM_WCP_GEMM_PROXY_COMMAND_LATENCY_CYCLES=1`;
- `GOLEM_WCP_GEMM_PROXY_COMPLETION_LATENCY_CYCLES=1`.

The Attention runner requires queue depth at least 16 because one QK/PV step
can enqueue a launch for all 16 arrays before the next WCP issue cycle.

Array output reads now have explicit lower-level resource parameters:
`GOLEM_ARRAY_OUTPUT_READ_CREDITS` limits output reads in flight and
`GOLEM_ARRAY_OUTPUT_READ_BANKS` models independent output-read banks. The
default `1/1` setting preserves the original single-port behavior; increasing
these values is only meaningful after the corresponding physical buffer ports
are provided and should be evaluated with QK readout statistics.

PV matrix programming uses the modeled array-cluster broadcast fabric by
default. One V panel enters the fabric once and is distributed to the 16 array
matrix banks through a bounded binary tree. The transfer occupies the same
array-buffer port and queue used by ordinary matrix/input/output transfers, so
it participates in resource contention. Functional bank updates occur only
after the modeled transfer completes; the Attention tiling, online Softmax,
PV accumulation order, and WCP GEMM launch flow are unchanged.

K/V Local-GM uses two modeled buffers. Once tile N has released both operands,
the released buffer can issue the N+2 DMA while N+1 is resident in the other
buffer. The DMA still traverses GlobalMemory queues, bandwidth, and
backpressure. Use `run_fused_attention_scale.sh --no-kv-second-lookahead` for
an explicit lower-level N+1-only diagnostic control; the public runner fixes
the latest N+2 mechanism on.

At query-block boundaries, the default runner also uses the released double
buffer to prefetch tile 0 for the next query block while the current block's
output DMA is in flight. This bounded temporal residency adds no storage,
preserves the QK/PV order, and still pays GlobalMemory queueing and bandwidth.
Disable it for an apples-to-apples control with
`run_fused_attention_scale.sh --no-kv-cross-query-prefetch`. RESULT and
lifecycle JSON report cross-query prefetches, hits, waits, and wait ticks
separately from the intra-block N+2 counters.

Adjacent non-causal query blocks also reuse each resident K/V tile by default.
The public runner advances four queries before the key-tile ordinal, while
retaining independent Q/O tiles and online-Softmax state. For D128 this uses
four Q tiles, four O tiles, and 64 online row contexts per worker; score scratch
and the two K/V buffers remain shared. Against the verified two-query path,
four-query reuse reduces Q1024/K1024/D128 from 663,804 to 620,168 normalized
cycles (6.57%) and Q2048/K2048/D128 from 2,393,428 to 2,289,536 cycles (4.34%),
with exact Q1024 repeatability and zero numerical mismatches. Use
`--kv-query-group-size 2` for the former pair schedule, or
`--kv-query-group-size 1`/`--no-kv-pair-reuse` for the query-major control.
Component-level configurations retain group size 2 as their compatibility
default. Causal Attention and the experimental O-accumulator C-buffer path
reject grouped reuse rather than weakening their state contract.

The default PV path also uses one bounded 16 KiB V-tile buffer per worker.
The first panel fills it through the modeled Local-GM path; later 2 KiB panels
reuse the tagged tile through a 64 B/cycle modeled port plus one base cycle.
The staging identity is `(worker generation, query-group owner, key-tile
ordinal)`, so follower queries in the same physical group also reuse panel 0;
a new generation, group, or key tile invalidates the payload. Group1 and final
partial groups follow the same identity rule without allocating more storage.
Capacity, tag, bandwidth wait, hits, misses, fill bytes, reused bytes,
cross-query hits, and rejections are all checked from component statistics.
For group4, this reduces V fills from 32 MiB to 8 MiB at Q1024 and from 128 MiB
to 32 MiB at Q2048, improving the verified end-to-end results from 620,168 to
578,914 cycles and from 2,289,536 to 2,160,533 cycles. Use
`--no-pv-v-tile-group-retention` for the previous per-query lifetime. Input
programming, restore, output, early-compute, and matrix/Softmax overlap
pipelines remain enabled as one verified bundle; lower-runner `--no-pv-*`
controls are diagnostics and do not define the public default.

QK panel input uses a bounded two-slot producer/consumer pipeline by default.
Each 512 B slot is tagged by generation, query, key tile, panel, array, and
transfer identity. It overlaps one modeled Local-GM read with one modeled
array-buffer input program while retaining one request per existing port and
strict array order. This reduces the fixed QK-input phase from 608 to 368 cycles
per tile and moves Q1024/Q2048 group4 to 545,851/2,056,077 cycles. Use
`--no-qk-input-pipeline` for the serial input-programming control.

PV input residency is also enabled. Within one PV job, probability inputs are
programmed on panel 0 and retained for the remaining dimension panels, with
explicit job/panel ownership and unchanged FP32 accumulation order. Its final
re-evaluation moves Q1024/Q2048 group4 to 507,341/1,950,847 cycles. The older
O-accumulator C-buffer remains an off-by-default experiment because its extra
restore/output traffic did not improve end-to-end latency consistently.

The current generic, non-transposed grouped path adds one inactive QK
matrix/input operand bank. During the current tile's Softmax/PV window it
prepares at most one same-group follower for the same physical key tile; the
normal ordered path then promotes the tagged bank and launches panel 0. The
added capacity is exactly 136 KiB per worker. Compute issue, outputs, score
storage, WCP issue width, Local-GM ports, and K/V buffers are not duplicated.
Group1, direct GEMM, and transposed QK use one bank automatically. Use
`--no-cross-tile-operand-pipeline` for ablation.

The final verified rank-1 matrix is:

| Workload | Group | Cycles | 1 GHz latency |
|---|---:|---:|---:|
| Q256/K256 | 1/2/4 | 71,916 | 0.071915 ms |
| Q512/K512, two-member tail | 4 | 168,547 | 0.168546 ms |
| Q768/K768, three-member tail | 4 | 311,542 | 0.311541 ms |
| Q1024/K1024 | 2 / 4 | 560,875 / **491,072** | 0.560874 / **0.491071 ms** |
| Q2048/K2048 | 2 / 4 | 1,967,488 / **1,899,765** | 1.967487 / **1.899764 ms** |

All configurations pass numerical, lifecycle, storage, and statistics checks.
Repeated Q1024/Q2048 group4 runs have identical cycles and SST ticks.

The fabric defaults and environment overrides are:

- `GOLEM_MATRIX_BROADCAST_MAX_FANOUT=16`;
- `GOLEM_MATRIX_BROADCAST_BYTES_PER_CYCLE=64`;
- `GOLEM_MATRIX_BROADCAST_BASE_LATENCY_CYCLES=1`;
- `GOLEM_MATRIX_BROADCAST_STAGE_LATENCY_CYCLES=1`.

For payload size `B` and fanout `F`, latency is
`base + ceil(B / bytes_per_cycle) + ceil(log2(F)) * stage_latency` array
component cycles. Use `--no-pv-matrix-broadcast` only for a serial-programming
control run. The verifier requires exact accepted/rejected transaction counts,
ingress bytes, aggregate sink bytes, transfer cycles, and fanout range from the
SST compute-array component.

The lifecycle JSON reports aggregate WCP command, queue-wait, launch,
completion, and completion-delay metrics under `wcp_gemm_proxy`. This proxy
timing does not enable the RequestScheduler/HBM descriptor path.

Attention tests print host-stage markers while they run and enable concise
RoCC lifecycle markers by default. RoCC records use this parseable format:

```text
[ATTENTION_MILESTONE] stage=final_qk_tile_complete status=done sst_tick=... rocc_cycle=... core=... role=worker job=... tag=... query_block=... key_tile=...
```

Set `GOLEM_ATTENTION_MILESTONE_TRACE=0` to disable these records. Set
`GOLEM_ATTENTION_TILE_TRACE=1` to print every QK, Softmax, PV, and output-DMA
tile milestone; the default is `0`, which prints job/frontier milestones and
only the final tile on each worker.

After verification, every run writes `attention_metrics.json` and
`attention_metrics.csv` in the artifact directory and prints the same key
metrics to the terminal. The report includes SST and pre-report pipeline wall time,
accelerator completion and software-wait latency, every system-frontier stage,
the slowest worker stages, numerical verification, WCP control metrics,
matrix-broadcast activity, PV active-K work avoided, and V-tile buffer activity.
`total_cycles` and stage cycles are normalized using the configured report
clock (1 GHz by default); they are not RoCC-, array-, or memory-native cycles.
The corresponding raw SST timebase ticks and model clocks remain in the JSON.

Terminal output is concise by default: generic DMA/MVM heartbeat records are
disabled, repeated worker counts are collapsed into phase changes, and full
generator/verifier output (including non-fatal base-runner warnings) is stored
under `driver_logs/`. Set
`GOLEM_ATTENTION_TERMINAL_VERBOSE=1` to restore complete command output for
debugging. This setting changes presentation only; RoCC milestone records and
JSON/CSV metrics remain enabled.

On an interactive terminal, the SST phase uses a colored progress bar, spinner,
elapsed timer, and in-place refresh matching the reference runner style. The
component label changes through SETUP, DISPATCH, QK, SOFTMAX, PV, DMA, and
FINALIZE; concurrent worker pipelines are represented by live QK/Softmax/PV/DMA
frontier counts rather than pretending those stages are globally serial. Piped
or redirected output automatically falls back to one plain line per component. The
final colored `== RESULT ==` section retains configuration, validation, wall and
simulated time, total cycles/ticks, wait return, numerical error, grouped stage
cycles, critical worker, and WCP metrics. `NO_COLOR=1` disables automatic color;
`GOLEM_ATTENTION_COLOR=1` explicitly enables it. Stage names, PASS/FAIL status,
wall times, RESULT labels/values, and artifact paths use semantic colors too.
Set `GOLEM_ATTENTION_TERMINAL_REFRESH_SECONDS` to override the interactive
refresh period (default `0.1` seconds); this affects display polling only.

When `--baseline FILE` is supplied, the runner validates shape and MPI metadata
before SST starts, then includes the post-run baseline match in the same
`Verification` line and overall report status.

For example, run Q1024/K1024/D128 on two MPI ranks with the default WCP,
broadcast, rolling prefetch, active-K/V-tile PV path, and milestone reporting
using:

```bash
scripts/test_flash_attention.sh --queries 1024 --keys 1024 --head-dim 128 \
  --mpi-ranks 2 --timeout 600
```

The historical MPI2 checkpoint before the later rank-1 optimization ladder was
1,163,611 cycles for
Q1024/K1024/D128 and 4,112,550 cycles for Q2048/K2048/D128. At Q1024, the
same-binary active-K and V-tile ablations show 7.67% and 9.84% individual
end-to-end contributions; the combined default is 32.64% below the former
1,727,387-cycle unified/N+2 result.

Key files:

- `scripts/test_flash_attention.sh`: canonical parameter-driven entry point;
- `run_flash_attention.sh`: thin in-directory forwarding entry point;
- `GPU_COMPETITIVE_ROADMAP.md`: current GPU comparison contract and ordered
  architecture optimization plan;
- `PHASE_AB_RESULTS.md`: audited clocks, SST stage results, and the Phase C
  priority decision;
- `gpu_attention_stage_benchmark.py`: standalone CUDA FP32 Scope A and
  single-stream stage measurement for the external GPU host;
- `gpu_attention_stage_schema.json`: import contract for GPU stage samples;
- `report_attention_gpu_comparison.py`: combines E3/E4 SST lifecycle files with
  the GPU result into one comparison report;
- `run_fused_attention_scale.sh`: SST execution and verification pipeline;
- `golem_attention_runtime.{h,cpp}`: RISC-V guest runtime;
- `attention_case.py`: deterministic Q/K/V generation;
- `verify_fused_attention_scale_output.py`: numerical verification;
- `verify_fused_attention_scale_stats.py`: lifecycle/statistics verification.
- `report_attention_metrics.py`: terminal, JSON, and CSV timing summary.
- `verify_attention_mpi_partition.py`: query-block rank placement verification.

The checked-in `baseline/e3/lifecycle.json`, `baseline/e4/lifecycle.json`, and
`baseline/gpu_attention_rtx5060.json` are the persistent inputs for regenerating
the historical Phase A/B `baseline/attention_gpu_comparison.json`; reproduction
does not depend on `/tmp`. Its SST values predate the optimization ladder above
and must not be used as the current SST baseline.

From the repository root, reproduce the checked-in SST/GPU comparison with:

```bash
python3 src/sst/elements/golem/tests/small/muticore_attention/report_attention_gpu_comparison.py \
  --sst-e3 baseline/e3/lifecycle.json \
  --sst-e4 baseline/e4/lifecycle.json \
  --gpu baseline/gpu_attention_rtx5060.json \
  --result-json baseline/attention_gpu_comparison.json
```

On the RTX host, repeat the FP32 stage measurement with:

```bash
python3 gpu_attention_stage_benchmark.py \
  --profiles e3 e4 --warmup 50 --iterations 200 \
  --output gpu_attention_rtx5060_stages.json
```
