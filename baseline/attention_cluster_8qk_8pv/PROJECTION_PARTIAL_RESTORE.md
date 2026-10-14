# Partial movement breakdown and early restore

> **架构纠正（2026-10-10）：** 本文的历史投影/E2E 测量或重叠估计使用了 manager 本地 arrays，不符合「manager 只控制、16 个 worker 拥有阵列」的目标拓扑，不能作为该架构的性能证据。新的 worker 执行基准见 [WORKER_PROJECTION_ARCHITECTURE_FIX.md](WORKER_PROJECTION_ARCHITECTURE_FIX.md)。

The best prior configuration now supports one more opt-in schedule:
`GOLEM_PROJECTION_PARTIAL_RESTORE_AHEAD=1` (default zero). This applies to the
existing paired pipeline with C-buffer partial storage. It changes request
timing, with no additional SRAM, output context, array, operand bank, port,
bandwidth or DMA credit. Other execution paths retain their existing schedule.

## Bandwidth and request counts

The executable configuration and actual service counters agree: grouped
ProjectionOutput reads/restores use **256 B/cycle aggregate**, not the generic
64 B/cycle rate. An 8 KiB group takes 33 service cycles. Output, input scatter
and weight broadcast share one array-buffer port. The detailed audit, including
GM small-request effective bandwidth and external DMA resources, is in
[`PROJECTION_BANDWIDTH.md`](PROJECTION_BANDWIDTH.md).

At S1024, each manager has 192 output groups and 16 hidden-tile pairs.
Consequently it performs 2,880 partial reads, 192 final reads and 2,880 partial
restores. Each partial read/C-buffer store/C-buffer read/array restore leg
carries 22.5 MiB. Final readout carries 1.5 MiB. Counts come from partial
materialization between pairs while reusing the single output context.
Changing bandwidth would change time, not these counts.

## Exclusive controller subdivision

New `[PROJECTION_PARTIAL_MOVEMENT]` records partition ReadOutput into partial
and final reads, and RestoreOutputReady into C-buffer readiness, transfer
submission, foreground transfer completion, ahead-transfer completion and
ahead-transfer consumption. These subdivisions are checked against the existing
exclusive controller state totals. C-buffer issue-to-ready residence is recorded
separately; it overlaps foreground work and must not be added to controller ticks.

S1024 per-manager measurements:

| Controller residence | Control ticks | Early-restore ticks |
| --- | ---: | ---: |
| Partial output read | 154,800 | 154,800 |
| Final output read | 9,120 | 9,120 |
| Restore dispatch | 2,880 | 2,880 |
| RestoreReady: C-buffer readiness wait | 0 | 0 |
| RestoreReady: foreground submission | 2,880 | 0 |
| RestoreReady: foreground transfer wait | 97,920 | 0 |
| RestoreReady: ahead-transfer wait | 0 | 66,816 |
| RestoreReady: ahead-transfer consumption | 0 | 2,880 |
| **Output read + restore total** | **267,600** | **236,496** |

C-buffer prefetch reads and stores each have 95,040 issue-to-ready residence
cycles in both cases; there are no demand reads in this formal configuration.
Physical output traffic also stays identical: 3,072 gathers, 2,880 restores,
48,758,784 bytes and 196,416 array-buffer service cycles per manager. The matched
analyzer verifies these physical counters for all managers at both lengths.
S2048 counts and these residence totals double.

## Schedule and ordering

After the entire current output has been drained into the existing output
snapshot, the sole array output context is available. Once the existing next
partial prefetch reaches its timed C-buffer ready cycle, the scheduler restores
that partial immediately through the existing ProjectionOutput interface.
It overlaps restoration with current partial storage and subsequent operand
preparation, rather than waiting until input/weight preparation finishes.

The scheduler tracks the target row/tile and consumes that restore only for the
matching computation. Launch waits for restore completion. It retries rejected
transfer submissions without losing the C-buffer snapshot. Late or absent
prefetches use the original foreground restore. Output draining and timed
C-buffer readiness are mandatory fences. Hidden-tile order, FP16 partial
rounding and layouts are unchanged.

## End-to-end results

Formal shape: FP16 causal Hq32/Hkv8/D64/hidden2048, seed1742. The workload covers
RMSNorm, QKV projection, RoPE and Attention through O HBM; Wo, residual and MLP
are excluded. Both variants use norm fusion, skip audit writeback, no RoPE
fusion, C-buffer partials and next-pair input prefetch.

| Sequence | Previous best E2E | Early restore E2E | Saved cycles | Reduction |
| --- | ---: | ---: | ---: | ---: |
| 1024 | 1,298,385 | **1,265,454** | 32,931 | **2.54%** |
| 2048 | 3,039,112 | **2,974,207** | 64,905 | **2.14%** |

| Critical-path change, optimized minus control | S1024 | S2048 |
| --- | ---: | ---: |
| Producer span | -33,168 | -64,689 |
| Handoff | -24 | +3 |
| Attention span | +261 | -219 |
| **E2E** | **-32,931** | **-64,905** |

Every formal case passed independent NumPy O verification, panel layouts, MPI
partition, lifecycle/backend and work accounting. All 16 retained panel/O
regions are byte-identical between matched variants: 10 MiB at S1024 and
20 MiB at S2048. O maximum absolute error remains 0.0001220703125. All four cases
use the same simulator-library SHA256, identical input hashes and identical
hardware/environment parameters except the schedule flag and host CPU affinity.
Local GM and C-buffer bytes, output fabric service cost and array compute work
are unchanged. The control reproduces the previous best cycles exactly.

Against the existing incomplete optimistic resource floor, ratios improve from
1.978 to 1.927 at S1024 and from 1.929 to 1.888 at S2048. This floor omits complete
QK/PV/RoPE/transfer/control costs and is not a reachable-performance promise.

## Regression and delayed-C-buffer check

The Python suites pass 134 tests (122 Attention/report tests and 12 baseline
tests). New accounting tests reject missing partial work and unaccounted
read/restore cycles. Shell syntax and diff checks pass.

An additional Hq4/Hkv2/D64/S1024 test uses 64 B/cycle C-buffer read/write service,
seven-cycle completion latency, a 64-slot GM queue, and disables next-pair input
prefetch. It passes all E2E checks at 86,402 cycles. Each manager executes
24 ahead restores and eight foreground restores, with 1,072 foreground C-buffer
wait ticks. This checks timed readiness and fallback with mixed restore paths;
its altered resources are excluded from the formal performance comparison.

## Reproduction and artifacts

Build the local library with `scripts/build_and_install_local.sh --no-autogen
--jobs 8` and source `scripts/env_local_install.sh`. Use fresh artifact roots:

```bash
GOLEM_PROJECTION_CBUFFER_PARTIAL=1 GOLEM_PROJECTION_WRAP_PREFETCH=1 \
GOLEM_FUSION_NORM_PROJECTION=1 GOLEM_FUSION_PROJECTION_ROPE=0 \
GOLEM_FUSION_SKIP_AUDIT_OUTPUT=1 GOLEM_PROJECTION_PARTIAL_RESTORE_AHEAD=0 \
python3 baseline/attention_cluster_8qk_8pv/run_llama_projection.py \
  --sequences 1024 2048 --artifact-root results/fresh_partial/control \
  --reference-root results/projection_schedule/wrap_prefetch

GOLEM_PROJECTION_CBUFFER_PARTIAL=1 GOLEM_PROJECTION_WRAP_PREFETCH=1 \
GOLEM_FUSION_NORM_PROJECTION=1 GOLEM_FUSION_PROJECTION_ROPE=0 \
GOLEM_FUSION_SKIP_AUDIT_OUTPUT=1 GOLEM_PROJECTION_PARTIAL_RESTORE_AHEAD=1 \
python3 baseline/attention_cluster_8qk_8pv/run_llama_projection.py \
  --sequences 1024 2048 --artifact-root results/fresh_partial/restore_ahead \
  --reference-root results/fresh_partial/control

python3 baseline/attention_cluster_8qk_8pv/analyze_projection_schedule.py \
  --artifact-root results/fresh_partial --variant restore_ahead
```

Full logs/configurations/counters are under `results/projection_partial_restore`.
The tracked [`projection_partial_restore_results.json`](projection_partial_restore_results.json)
preserves all exclusive state partitions, partial subdivisions, physical output
counters, library hash and stress evidence.
