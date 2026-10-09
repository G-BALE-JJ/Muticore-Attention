# Llama projection handoff (2026-10-09)

Read this first in a new session. The repository is
`/data/jjgong/LLM/Muticore-Attention`, branch `softmax-update`. At archive
time the latest implementation commit was `ee69c6e`, one commit ahead of
`origin/softmax-update`. It was committed locally but **not pushed**. This
handoff document is a subsequent commit; recheck `git status --short --branch`
before editing.

## Workload and measurement scope

The current end-to-end task is synthetic FP16 Llama prefill with Hq=32,
Hkv=8, head dimension 64, hidden dimension 2048, and sequence lengths 1024
and 2048. Four managers each have 64 projection arrays. The measured path is
RMSNorm -> QKV projection -> RoPE -> causal GQA -> O HBM. It does **not**
include output projection Wo, residual addition, MLP, or a complete Llama
layer. Inputs use seed 1742. The full-width array computation takes 66 cycles
in this 1 GHz simulator profile.

The reproducible runner is `baseline/attention_cluster_8qk_8pv/run_llama_projection.py`.
Its reference comparator checks identical input hashes and byte-exact RMSNorm,
Q/K/V raw output and Q/K/V panel regions against a chosen artifact root.
Artifacts under `results/` are local and git-ignored. The checked-in
measurement narrative is `LLAMA_PROJECTION_SCHEDULE_CBUFFER.md` in this
directory; earlier baselines and design history are in the neighboring
`LLAMA_PROJECTION_*.md` files.

## Verified results

| Configuration | S1024 E2E | S1024 projection | S2048 E2E | S2048 projection |
| --- | ---: | ---: | ---: | ---: |
| 64 arrays + bounded weight prefetch (`results/llama_weight_prefetch`) | 1,503,249 | 1,063,695 | 3,447,833 | 2,126,268 |
| Row-first control (`results/llama_rowfirst_verified`) | 2,192,722 | 1,753,435 | 4,823,845 | 3,502,275 |
| Existing C-buffer, no partial prefetch (`results/llama_cbuffer_partial`) | 1,544,364 | 1,105,183 | 3,530,760 | 2,209,326 |
| Existing C-buffer, partial prefetch (`results/llama_cbuffer_prefetch`) | **1,474,856** | **1,035,819** | **3,391,766** | **2,070,630** |

The prefetched C-buffer experiment saves 28,393 / 56,067 end-to-end cycles
(1.89% / 1.63%) versus the weight-prefetch reference. Row-first is slower
because it reprograms each weight tile for the four row groups and loses
cross-head input-block sharing. Keep it as a negative control only. The
unprefetched C-buffer case is also slower: moving partials without moving
their reads off the serial restore path increases stalls.

All four configurations pass full numerical, backend, MPI, HBM layout and
projection-accounting checks at both lengths. Each candidate matches the
reference across all 28 selected HBM regions byte for byte (16 MiB at S1024,
32 MiB at S2048). The final attention O maximum absolute error is
0.0001220703125 in the measured candidate runs. The C-buffer path also passes
a small D64/S1024 case with its pipeline disabled. Compare source and binary
hashes in each case's `baseline_provenance.json` when reproducing.

## Actual latency versus optimistic floor

| S | RMSNorm actual / floor | Projection actual / floor | Attention actual / exp floor | E2E actual / floor |
| --- | ---: | ---: | ---: | ---: |
| 1024 | 187,834 / 111,808 | 1,035,819 / 405,504 | 249,667 / 139,264 | **1,474,856 / 656,576 (2.246x)** |
| 2048 | 374,612 / 223,616 | 2,070,630 / 811,008 | 943,006 / 540,672 | **3,391,766 / 1,575,296 (2.153x)** |

The floor sums RMSNorm vector work, projection array launches at full width,
and causal attention exponential-lane work. It excludes DMA, NoC, controller
work, resource contention and handoffs; it is an optimistic resource floor,
not a claim about attainable full-system latency. Actual exceeds it by
818,280 / 1,816,470 cycles. Projection accounts for 630,315 / 1,259,622
of that gap (about 77% / 69%). Do not sum per-phase cycle counters as an
end-to-end speedup prediction: stages and background transfers overlap.

## Current implementation and physical storage

`src/sst/elements/golem/attention/projectionJob.h` owns the projection
traversal, output read/restore, weight reuse and opt-in switches:

- `GOLEM_PROJECTION_ROW_FIRST=1` processes all K tiles for one 64-row group
  before changing rows. It removes partial spills but is much slower.
- `GOLEM_PROJECTION_CBUFFER_PARTIAL=1` keeps weight reuse and stages each
  partial in the existing timed WCP C-buffer. It prefetches the next row
  group's partial during array compute. This is the measured faster variant.
- With both unset, the prior local-GM partial path remains the default. The
  dedicated full-shape runner defaults to 64 arrays and weight prefetch.

`src/sst/elements/golem/workercmdproc/workercmdproc.h` reserves/releases the
C-buffer for projection and uses its existing 256-byte/cycle read and write
ports and one-cycle latency. The WCP already physically allocates 1 MiB per
manager. Projection uses four 8 KiB partial slots, **32 KiB of that existing
capacity**, so it adds no physical SRAM. The 2 MiB local GM remains present;
moving partials does not shrink its capacity. Existing array output storage
is 8 KiB per manager. At S1024, the new path transfers 22.5 MiB of partial
reads and 22.5 MiB of partial writes per manager through the C-buffer; at
S2048 each direction is 45 MiB. Array output read/write transfers still run
through the timed array API.

The report parser and validation are in
`src/sst/elements/golem/tests/small/muticore_attention/report_projection_e2e.py`.
The runner propagates the switches through `run_fused_attention_scale.sh`.
Focused parser tests are in `test_report_projection_e2e.py`. Before archive,
the full S1024/S2048 comparison, small serial C-buffer case, three report
tests, `bash -n` and `git diff --check` passed.

## Next candidate: four output contexts

The existing K-major traversal programs one weight pair, then computes four
successive 64-row groups of a 256-row block. Today each array has one output
vector. Before another group overwrites it, the partial must be read out to
C-buffer; before that group computes the next weight pair, it must be restored.
There are 16 weight pairs for hidden dimension 2048. Output contexts would
give each row group its own retained accumulator while retaining weight reuse:

| Context | Rows within the 256-row block |
| --- | --- |
| 0 | 0-63 |
| 1 | 64-127 |
| 2 | 128-191 |
| 3 | 192-255 |

Each of the 64 arrays holds 64 FP16 output values per context. Four contexts
therefore need 32 KiB array output storage per manager instead of 8 KiB:
**24 KiB incremental per manager, 96 KiB across four managers**. This is
real new physical array storage, unlike C-buffer reuse. The existing two
operand banks are matrix/input banks and are not output contexts.

At S1024, manager 0 still spends 145,920 phase cycles reading array output
and 123,840 restoring partials. Keeping four accumulators could remove
15/16 of the intermediate output reads (136,800 phase cycles) and the
intermediate restores (123,840 phase cycles), plus the 45 MiB C-buffer
partial traffic per manager. These are removable work estimates, **not**
measured end-to-end savings. Switching contexts, extra storage and ports must
be timed and may offset part of the gain.

Suggested implementation/verification order:

1. Add an explicit context selector to the compute-array/WCP launch and
   output-read API. Model four finite output slots, switching delay, read
   bandwidth, port contention and backpressure. Do not use unbounded software
   vectors as free hardware storage.
2. Map `(row - blockStart) / 64` to context 0-3. Initialize a context on its
   first K tile, accumulate later tiles, and read it only after the last pair.
   Check context reuse at head and 256-row-block boundaries, Q/K/V raw rows,
   Q/K panels and V transpose layout. Preserve the current FP16 rounding.
3. First test a small shape and address mapping independently. Then run full
   S1024 and S2048 against the same reference with numerical, byte-exact HBM
   layout, physical-capacity and end-to-end cycle checks. Keep one-context
   behavior as control. Do not enable the four-context path by default until
   all checks pass and measured E2E improves.

## Reproduce the current fastest experiment

Use a fresh artifact root. The local `results/llama_weight_prefetch` root is
the byte-exact reference; recreate it first if unavailable in a new workspace.

```bash
cd /data/jjgong/LLM/Muticore-Attention
source scripts/env_local_install.sh
scripts/build_and_install_local.sh --no-autogen --jobs 8
GOLEM_PROJECTION_ARRAYS=64 GOLEM_PROJECTION_CBUFFER_PARTIAL=1 \
  python3 baseline/attention_cluster_8qk_8pv/run_llama_projection.py \
  --sequences 1024 2048 --artifact-root results/repro_cbuffer_new \
  --reference-root results/llama_weight_prefetch
```

The latest experiment commit `ee69c6e` was not pushed at archive time.
Decide whether to push it before work that depends on another checkout.
