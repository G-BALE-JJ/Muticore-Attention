# Verified 16/32/64-array projection

`GOLEM_PROJECTION_ARRAYS=16|32|64` selects the number of parallel token rows
per manager in the paired D64 projection path. The general scale runner defaults
to 16; the dedicated baseline now defaults to 64 plus
[bounded weight prefetch](LLAMA_PROJECTION_WEIGHT_PREFETCH.md). The measurements
below predate weight prefetch; disable it to reproduce this comparison.
Other projection paths use 16 arrays. There are still 64 physical arrays per
core, two operand banks, and one shared output vector per array. This change
uses more of the existing arrays during projection; it does not reduce the
66-cycle array latency or increase the MAC rate, clocks, or fabric bandwidth.
The runner raises the matrix broadcast fanout cap to the requested width and
records the resolved configuration. Actual fanout and transfer costs are timed.

## Address mapping repair

Let A be the array count, r the first local row in a group, and i the array ID.
Array i produces row r+i, with 64 FP16 values (128 bytes) per D64 row:

| Buffer | Address / offset | Group size |
| --- | --- | --- |
| Input pair for input tile t | input_block + (r-block_start+i) * hidden * 2 + t * 128 | A * 256 bytes, gathered at the full hidden row stride |
| Array output read/restore | element i * 64 + dim | A * 64 FP16 values |
| Local partial | scratch + 0xE0000 + (r-block_start+i) * 128 | A * 128 bytes |
| Raw Q/K/V | raw_base + (head * rows_per_node + r+i) * 128 | A * 128 bytes |
| Q/K panel | panel_base + (head * (rows_per_node/64) + r/64) * 8192 + (r%64+i) * 128 | A * 128 bytes |
| V panel | existing panel tile base + (dim * 64 + key) * 2 | one complete transposed 64-row tile |

Row-group traversal, foreground reads, input scatter, output vectors, partial
stores/restores, and background prefetch sizes all use A. The input-tile byte
offset remains t * 128 regardless of A. Input prefetch must gather A rows;
retaining the 16-row byte count leaves the other lanes without valid inputs.

V panel completion used to require r%64 == 48, which works only for a 16-row
group. It now requires r%64 + A == 64: the flush starts at row 48 for A=16,
32 for A=32, and 0 for A=64. This fixes the missing V panels that caused final
attention output to be zero. Whole-group output reads/writes support A=32/64;
no output API change or subdivision is necessary.

The input prefetch, partial prefetch, and partial-store snapshot require
A * (256+128+128) bytes: 8/16/32 KiB. This budget is checked against the same
2 MiB local GM, excluding its DMA status tail. SRAM requests still split at
the configured 4096-byte maximum and share the existing timed ports.

## Validation and cycles

Small Hq4/Hkv2/D64/hidden256/S1024 cases first checked the row mapping. All
three widths pass final numerical, HBM layout, MPI, backend, and cycle/work
accounting checks. For both wider cases, all four nodes' Q/K/V and panels
match the 16-array control byte for byte (2,097,152 bytes each comparison).
The 64-array serial control with `GOLEM_PROJECTION_PIPELINE=0` also passes.
Small E2E cycles are 119,708 / 102,684 / 95,631 for 16 / 32 / 64 arrays.

Full synthetic Llama uses Hq32/Hkv8/D64/hidden2048, seed 1742, FP16 inputs,
gamma and weights. The scope is RMSNorm -> QKV -> RoPE -> causal GQA -> O HBM;
Wo, residual and MLP are excluded. Results are simulator cycles at 1 GHz:

| S | Arrays / manager | Projection cycles | E2E cycles | E2E saved vs 16 | E2E reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1024 | 16 | 2,380,859 | 2,820,198 | 0 | 0% |
| 1024 | 32 | 1,499,571 | 1,938,289 | 881,909 | 31.27% |
| 1024 | 64 | 1,172,732 | 1,611,982 | 1,208,216 | 42.84% |
| 2048 | 16 | 4,754,102 | 6,075,613 | 0 | 0% |
| 2048 | 32 | 2,998,429 | 4,320,105 | 1,755,508 | 28.89% |
| 2048 | 64 | 2,343,594 | 3,664,580 | 2,411,033 | 39.68% |

All six cases pass numerical, layout, backend, MPI, attention work, projection
work/byte accounting and timestamp-order checks. Both wider variants match
the 16-array control's RMSNorm, Q/K/V and panels byte for byte: 16 MiB checked
at S1024 and 32 MiB at S2048 per comparison. Generated inputs also have
identical hashes. Final O maximum absolute error is 0.0001220703125 in every
case, with the original tolerances. The 16-array rerun reproduces the earlier
2,820,198 / 6,075,613 cycles exactly.

Build, 89 baseline contract tests, three projection report tests (including
32/64 count, staging-budget, and mixed-manager rejection cases), and ten
baseline runner tests pass. Compact reports, source/binary/input hashes,
resolved configurations and artifact locations are frozen in
[llama_projection_arrays.json](llama_projection_arrays.json). Runs were
performed before committing; source and binary hashes identify the tested
implementation, while git revision identifies its parent.

## Remaining bottlenecks

S1024 manager 0 phase residence illustrates why doubling arrays has
diminishing returns. Background activity overlaps these phases; these are
controller phase counts, not exclusive resource utilization:

| Phase | 16 arrays | 32 arrays | 64 arrays |
| --- | ---: | ---: | ---: |
| Array launch/compute | 1,671,168 | 835,584 | 417,792 |
| Weight DMA/control | 204,667 | 187,406 | 186,905 |
| Matrix program | 59,904 | 61,440 | 62,976 |
| Input scatter | 39,936 | 39,936 | 58,368 |
| Output restore | 126,720 | 109,440 | 100,800 |
| Output read | 135,168 | 116,736 | 145,920 |

Compute is halved with each width doubling, while total input, weight, and
partial-result bytes remain constant. At 64 arrays, weight DMA/control and
shared-output restoration/readback together consume about as much phase
time as compute. Output read and input scatter get slower from 32 to 64
arrays because each request is wider and overlap changes. The next measured
software experiment should prefetch weight DMA into bounded scratch; removing
output round trips requires a separate output-context/accumulator design.

64 arrays is fastest for the tested isolated projection schedule. It reserves
all arrays on each manager during projection; these results do not establish
the best allocation when other workloads share the arrays.

## Reproduce

Use fresh artifact directories. The runner retains exact input/layout and
cycle comparisons for both wider variants:

```bash
source scripts/env_local_install.sh
scripts/build_and_install_local.sh --no-autogen --jobs 8
export GOLEM_PROJECTION_WEIGHT_PREFETCH=0
GOLEM_PROJECTION_ARRAYS=16 python3 baseline/attention_cluster_8qk_8pv/run_llama_projection.py \
  --sequences 1024 2048 --artifact-root results/repro_arrays16
GOLEM_PROJECTION_ARRAYS=32 python3 baseline/attention_cluster_8qk_8pv/run_llama_projection.py \
  --sequences 1024 2048 --artifact-root results/repro_arrays32 \
  --reference-root results/repro_arrays16
GOLEM_PROJECTION_ARRAYS=64 python3 baseline/attention_cluster_8qk_8pv/run_llama_projection.py \
  --sequences 1024 2048 --artifact-root results/repro_arrays64 \
  --reference-root results/repro_arrays16
```
