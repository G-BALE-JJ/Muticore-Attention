# Bounded projection weight prefetch

The dedicated baseline now defaults to 64 projection arrays per manager and
`GOLEM_PROJECTION_WEIGHT_PREFETCH=1`. Both settings can be overridden through
the environment. The general scale runner still defaults to 16 arrays, and
non-paired projection paths continue using 16 arrays with weight prefetch off.

## Implementation and ownership

A manager may submit one background weight DMA at a time. It looks up to three
input tiles ahead within the current head (the other operand bank and next
pair), directly into their existing 8 KiB slots in the 512 KiB local-GM weight
cache. It never changes array matrices, operand banks, or output registers.
No extra SRAM, staging snapshot, array context, bandwidth or DMA credit is
added. Foreground demands and the single background request use the same
timed DMA/NoC/HBM/GM path.

The cache tag becomes valid only after DMA completion. If a demanded tile is
currently being prefetched, the foreground waits for its completion rather
than submitting a duplicate read. A prefetched tag is consumed on the first
matrix-program request. A head's last input tile must finish loading before
the traversal can change head/block, so a background request cannot outlive
its head or the job. `PROJECTION_WEIGHT_PREFETCH` records requests, consumed
hits, demand wait cycles, and remaining in-flight requests. The report checks
all four managers, requires requests == hits at completion, and rejects
extra requests beyond total weight loads or disabled-mode activity.

## Measured results

Same synthetic FP16 Llama Hq32/Hkv8/D64/hidden2048 and seed 1742 as the
[array-width comparison](LLAMA_PROJECTION_ARRAYS.md). Scope: RMSNorm -> QKV ->
RoPE -> causal GQA -> O HBM; excludes Wo, residual and MLP. All numbers are
simulator cycles, with 1 GHz clocks and 66-cycle full-width array computation.

| S | 64-array reference E2E | With weight prefetch E2E | Saved E2E cycles | Incremental reduction | New projection cycles |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1,611,982 | 1,503,249 | 108,733 | 6.75% | 1,063,695 |
| 2048 | 3,664,580 | 3,447,833 | 216,747 | 5.91% | 2,126,268 |

Relative to the verified 16-array pipeline baseline (2,820,198 / 6,075,613),
64 arrays plus weight prefetch reduce E2E cycles by 46.70% / 43.25%.

Both complete cases pass numerical, backend, MPI, HBM layout, attention work,
projection work/byte and timestamp-order checks. RMSNorm, Q/K/V and all panels
match the prior 64-array reference byte for byte (16 MiB at S1024, 32 MiB at
S2048). Input hashes match. Final O maximum absolute error remains
0.0001220703125. Current source and installed simulator binary hashes match
the archived provenance.

Per manager, S1024 has 1,488 prefetched tiles and 48 foreground tiles, totaling
the original 1,536 weight loads/programs. S2048 has 2,976 prefetched and 96
foreground tiles, totaling the original 3,072. Every prefetch is consumed;
demand wait cycles and remaining in-flight requests are zero in these runs.
This is measured for these shapes, not an assumption that DMA is instantaneous.

Small Hq4/Hkv2/D64/S1024 E2E is 93,685 cycles at 64 arrays and 99,836 at 32.
Disabling weight prefetch at 64 arrays reproduces the earlier 95,631 cycles.
RMSNorm, Q/K/V and panels in both enabled cases are byte-exact against this
disabled control. D128/Hq2/Hkv1 falls back to 16 arrays and remains at 161,566
cycles with all checks passing. The 89 contract, three projection report and
ten baseline runner tests pass. The report tests reject unconsumed prefetches,
in-flight completion, excess loads and missing manager evidence.

Results and provenance are frozen in
[llama_projection_weight_prefetch.json](llama_projection_weight_prefetch.json).
Artifact roots are `results/llama_weight_prefetch/s1024` and `s2048`.

## Remaining cost and output-context design

S1024 manager 0's serial weight-DMA/control phase falls from 186,905 to 9,245
cycles. Computation remains 417,792 cycles; output restore remains 100,800 and
output read remains 145,920. Foreground local-read residence increases from
180,288 to 249,312 cycles under the overlapping traffic. Thus moving DMA off
the serial path saves fewer elapsed cycles than the serial-phase reduction:
contention and overlap still have a cost. SRAM read/write bytes remain
86,507,520 / 23,592,960 per manager; none of the partial traffic was removed.

The current array API has one output vector per array; its two banks hold
operands, not independent accumulators. To retain the existing weight reuse
and avoid partial round trips, the next candidate is four timed output
contexts per array for the four 64-row groups in a 256-row input block:

1. Map local row group to context `(row % 256) / 64`, keeping array i mapped
   to row `row+i`. The first input tile overwrites that context; subsequent
   tiles accumulate into it, preserving the existing FP16 rounding/order.
2. Retain all four contexts while processing the 16 weight pairs. Read each
   context only after the last pair; preserve existing raw/panel writeback
   and the 64-row V transpose/flush rules.
3. Model context capacity, switching latency, ports, read bandwidth and
   backpressure explicitly. Select a context through the WCP launch/read API;
   reject conflicting or out-of-range requests and drain before completion.
4. Compare one-context control with four contexts using byte-exact projection
   layouts, final O tolerance, total physical bytes and E2E cycles. Include
   context wraparound at head and 256-row block boundaries.

For D64, four contexts require 64 arrays * 64 values * 2 bytes * 4 = 32 KiB
per manager, versus the current 8 KiB: **24 KiB additional physical output
storage per manager**, 96 KiB across four managers. Existing CBuffer modes are
GEMM partial-C and attention tile storage; they provide no projection
accumulator-context API. Reusing their capacity would require a timed transfer
and ownership design, rather than treating a software vector as free storage.

At S1024, each manager currently reads 24 MiB of array outputs and restores
22.5 MiB. Keeping four contexts would leave the final 1.5 MiB read and could
remove 45 MiB of intermediate array transfers plus 45 MiB of local-GM partial
reads/writes. A phase-count estimate is 100,800 restore cycles plus 15/16 of
145,920 read cycles = 237,600 cycles before context costs. This is an estimate
of removable controller work, **not a measured E2E speedup**; port contention
and overlap prevent adding phase counts into an elapsed-cycle prediction.
Output contexts are a documented next hardware-model experiment, not part
of the implemented or measured weight-prefetch optimization.

## Reproduce

Use fresh artifact directories. Setting `GOLEM_PROJECTION_ARRAYS=16|32|64`
retains the allocation controls; disabling weight prefetch reconstructs the
64-array reference:

```bash
source scripts/env_local_install.sh
scripts/build_and_install_local.sh --no-autogen --jobs 8
GOLEM_PROJECTION_WEIGHT_PREFETCH=0 \
  python3 baseline/attention_cluster_8qk_8pv/run_llama_projection.py \
  --sequences 1024 2048 --artifact-root results/repro_weight_control
python3 baseline/attention_cluster_8qk_8pv/run_llama_projection.py \
  --sequences 1024 2048 --artifact-root results/repro_weight_prefetch \
  --reference-root results/repro_weight_control
```
