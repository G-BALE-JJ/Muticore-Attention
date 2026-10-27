# Projection row scheduling and existing C-buffer experiment

The measured workload is synthetic FP16 Llama Hq32/Hkv8/D64, hidden 2048,
sequence 1024 or 2048, seed 1742. It covers RMSNorm, QKV projection, RoPE,
causal GQA and O HBM. It excludes Wo, residual and MLP. All runs use four
managers, 64 projection arrays per manager, 1 GHz clocks and the same bounded
weight prefetch as `results/llama_weight_prefetch`.

## Results

| Schedule | S1024 E2E | S1024 projection | S2048 E2E | S2048 projection |
| --- | ---: | ---: | ---: | ---: |
| Weight-prefetch reference | 1,503,249 | 1,063,695 | 3,447,833 | 2,126,268 |
| Row-first, full K per 64 rows | 2,192,722 | 1,753,435 | 4,823,845 | 3,502,275 |
| Weight reuse, C-buffer without partial prefetch | 1,544,364 | 1,105,183 | 3,530,760 | 2,209,326 |
| Weight reuse, C-buffer with partial prefetch | **1,474,856** | **1,035,819** | **3,391,766** | **2,070,630** |

Against the reference, row-first adds 689,473 / 1,376,012 E2E cycles.
Prefetched C-buffer saves 28,393 / 56,067 E2E cycles (1.89% / 1.63%) and
27,876 / 55,638 projection cycles. The unprefetched C-buffer case shows why
transfer placement matters: it removes local-GM partial traffic but waits for
the C-buffer read on the serial output-restore path. Prefetching the next row
group during array compute recovers that wait.

All four complete cases pass numerical, backend, MPI, layout, projection byte
accounting and timestamp checks at both sequence lengths. Each candidate uses
the same input hashes as the reference and matches its RMSNorm, Q, K, V, Q/K/V
panels byte for byte: 28 HBM regions, 16 MiB at S1024 and 32 MiB at S2048.
Final O maximum absolute error is 0.0001220703125 for both candidate schedules
at both lengths. The C-buffer path also passes a small D64/S1024 test with
pipeline prefetch disabled.

## Storage and traffic

| Resource per manager | Reference | Row-first | C-buffer partial |
| --- | ---: | ---: | ---: |
| Physical WCP C-buffer capacity | 1 MiB | 1 MiB | 1 MiB |
| Projection occupancy in that buffer | 0 | 0 | 32 KiB |
| Physical array output registers | 8 KiB | 8 KiB | 8 KiB |
| Incremental physical capacity | 0 | 0 | 0 |
| Partial read + write, S1024 | 22.5 + 22.5 MiB local GM | 0 | 22.5 + 22.5 MiB C-buffer |
| Partial read + write, S2048 | 45 + 45 MiB local GM | 0 | 45 + 45 MiB C-buffer |

The 32 KiB is four 64-row FP16 partial slots within the WCP's existing 1 MiB
C-buffer, with its existing 256-byte/cycle read and write ports and one-cycle
latency. Projection reserves it exclusively, then releases it before attention
uses the same physical buffer. This changes traffic location, not total
physical SRAM capacity. The 2 MiB local GM and its reserved scratch address
range also remain physically present. Array output transfers retain their
timed read/write path in both schedules. The software prefetch snapshot is
bounded to one 8 KiB row group and is not modeled as a new hardware SRAM.

Row-first avoids partial transfers but repeats matrix programming four times
per 256-row block: S2048 rises from 12,288 to 49,152 programs across managers.
It also loses the cross-head input-block sharing, so per-manager local-GM input
traffic and input DMA increase. It is a useful negative control, not the
preferred projection schedule.

## Output-context decision

The prefetched C-buffer run still reads array output after each weight pair and
restores partials before the next pair. At S1024, manager 0 spends 145,920
cycles in array-output read and 123,840 in output restore, while compute takes
417,792. Four output contexts would retain one accumulator for each 64-row
group across all 16 weight pairs. They could eliminate 15/16 of the output
reads (136,800 phase cycles) and the intermediate restores (123,840 phase
cycles), as well as the 45 MiB of C-buffer partial traffic per manager. Those
phase counts overlap other work and are not an E2E speedup prediction.

Four contexts require 32 KiB of physical array output storage per manager,
versus 8 KiB today: **24 KiB additional per manager, 96 KiB across four**.
The measured C-buffer gain is modest, while substantial output read/restore
work remains. A four-context prototype is therefore worth testing, provided
its context-select timing, output ports and backpressure are modeled. Keep the
one-context baseline as control and require numerical, byte-exact layout,
physical-capacity and E2E-cycle checks before adopting it.

## Reproduce

Build once, then use fresh artifact roots. The options are opt-in; the default
projection schedule and storage path remain unchanged.

```bash
source scripts/env_local_install.sh
scripts/build_and_install_local.sh --no-autogen --jobs 8
GOLEM_PROJECTION_ARRAYS=64 GOLEM_PROJECTION_ROW_FIRST=1 \
  python3 baseline/attention_cluster_8qk_8pv/run_llama_projection.py \
  --sequences 1024 2048 --artifact-root results/repro_rowfirst \
  --reference-root results/llama_weight_prefetch
GOLEM_PROJECTION_ARRAYS=64 GOLEM_PROJECTION_CBUFFER_PARTIAL=1 \
  python3 baseline/attention_cluster_8qk_8pv/run_llama_projection.py \
  --sequences 1024 2048 --artifact-root results/repro_cbuffer \
  --reference-root results/llama_weight_prefetch
```

Measured artifacts: `results/llama_rowfirst_verified`,
`results/llama_cbuffer_partial` (unprefetched), and
`results/llama_cbuffer_prefetch`. Each `summary.json` contains the stage
timings, numerical errors, provenance and byte-exact reference comparison.
