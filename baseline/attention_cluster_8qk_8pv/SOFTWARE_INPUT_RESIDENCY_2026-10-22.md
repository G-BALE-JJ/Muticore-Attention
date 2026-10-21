# Software input residency in the all-GEMM baseline

This branch preserves the original `Muticore-Attention` worktree and implements
software dataflow scheduling in the isolated `Muticore-Attention-all-gemm`
worktree. The existing 16 workers, 64 arrays per worker, one operand context,
2 MiB worker GM, ports, queues, clocks and memory timing remain fixed.

## Retained data and lifetimes

`--input-residency` extends the previously selected software GM stage chain:

1. The first layer's fused Norm1/QKV producer loads its original input directly
   into existing GM `0xb0000..0xf0000`. Norm1 consumes the same DMA payload.
   There is no second copy, additional input DMA or new SRAM allocation.
2. After QKV output ACKs, a timed 64-byte write publishes an ordinary GM
   ownership record at `0x3400`. It identifies epoch, HBM address, shape,
   manager and lane. Completion is signalled only after that write ACK.
3. Wo retains its output at `0x70000..0xb0000`, a software lifetime alias of
   otherwise idle scratch. The selected stage-chain configuration excludes
   concurrent weight-cohort and FFN block-pipeline users of that interval.
4. Residual Add checks the ownership record, reads Wo and original input
   through the existing timed GM ports, and writes H1 over the consumed input
   chunks only after the corresponding reads and SFU operation complete.
5. Norm2 reads H1 from GM and writes the existing packed FFN input. The fused
   Gate/Up/SiLU/Multiply/Down path consumes that input; its residual epilogue
   consumes H1 and replaces each consumed chunk with final Y in GM. Y is also
   materialized in HBM with the existing completion/ACK contract.
6. The next layer consumes retained Y in Norm1 and keeps it for that layer's
   residual Add. Final RMSNorm also consumes retained Y. The separate LM input
   option keeps its last-token input across adjacent vocabulary descriptors.

The new path removes the original-X HBM reread in each residual Add. For
S=1024, H=2048 and two layers this is exactly `1024*2048*2*2 = 8,388,608`
additional bytes, or 8 MiB. HBM output writes and dense GEMM work are unchanged.
Existing GM traffic remains timed and subject to shared-port contention.

## Supported experiment

The new switch is opt-in and requires `--gm-stage-chain --layer-residency`.
Residency applies to one 64-row group per worker (S=1024, H<=2048). Larger
sequences use the established non-resident fallback. The runner requires the
QKV K window to cover the complete hidden dimension so that intermediate C
spills cannot overwrite the retained payload. It rejects incompatible cohort,
block-pipeline and legacy Norm-handoff configurations.

The full-stack experiment uses S=1024, H=2048, FFN=8192, two layers,
vocabulary=128256, 16 workers and 8 MPI ranks. Sparse deterministic weight
values are test fixtures; dense MAC and weight/window traffic are verified.
Times are simulated cycles at 1 GHz, excluding host preparation and validation.

The original-X HBM traffic is removed; guest descriptor boundaries, PV-to-Wo
HBM redistribution, final output materialization and the first LM input load
remain. This change does not claim complete cross-operator streaming.

## Verified paired result, 2026-10-22

Both corrected runs pass full-stack numerical, lifecycle, timed-resource and
all-GEMM reuse-window coverage checks. The control exactly reproduces the
previously selected result. The candidate is selected because both decoder
and full end-to-end latency improve.

| Scope | Control cycles | Input-resident cycles | Latency reduction |
| --- | ---: | ---: | ---: |
| Token IDs through final logits ACK | 6,314,118 | 6,303,926 | 0.1614% |
| Two-layer decoder | 5,478,964.75 | 5,463,352.75 | 0.2849% |
| Final RMSNorm | 44,640.375 | 43,832.375 | 1.8100% |
| LM head | 663,626 | 671,248 | -1.1485% |
| Layer 1 residual Add span | 38,107 | 30,996 | 18.6606% |
| Layer 2 residual Add span | 39,082 | 32,290 | 17.3788% |

At 1 GHz, the complete end-to-end result is **6.314118 -> 6.303926 ms**,
a reduction of **10.192 us**. Stage spans can overlap and must not be summed.
The LM head is 7,622 cycles slower in this paired run, offsetting part of the
earlier-stage gain; the change does not accelerate every stage.

All 32 residual Add jobs consume their complete input through timed GM reads,
with zero HBM reads. The actual payload audit verifies 8,388,608 saved bytes,
no overwrite between Norm1 and Add, and read ACKs preceding in-place writes.
Decoder-op job counters (including LM descriptors, excluding projection and
attention jobs) change from 4,036,235,264 to 4,027,846,656 HBM-read bytes;
HBM-write bytes remain 17,034,368. The saving is about 0.208% of the measured
job-read total, so input residency alone has limited end-to-end impact.

Last-layer Y (2,097,152 elements), final Norm (2,097,152 elements) and logits
(128,256 elements) have zero FP16 bit mismatches against both the paired
control and the previous selected configuration. Greedy token remains 99037.

The selected command enables `--window-a-prefetch 1 --lm-n-parallel 1
--gm-stage-chain --layer-residency --lm-input-residency --input-residency`;
the runner retains K=32, four weight slots and one 64-array window group.
All new residency switches remain opt-in so older experiments are reproducible.

The machine-readable measured comparison, binary/source hashes and actual
payload audit are published in
[`software_input_residency_summary.json`](progress_results/2026-10-22/software_input_residency_summary.json).
The measured SST library SHA256 is
`21fec61f74e20ce64163327d53dabae736bc1d19e0d206c495b2d3d75afa98fb`.

## Verification

- Control and candidate use identical library, guest, memory model, inputs and
  layouts. Only `GOLEM_DECODER_INPUT_RESIDENCY` changes; host CPU placement is
  normalized in the comparison.
- Every residual Add must report zero HBM reads and complete timed GM coverage
  for its retained input. Logs are checked for premature overwrites between
  Norm1 and Add and for in-place writes preceding input-read ACKs.
- End-to-end numerical/lifecycle audits and all-GEMM reuse-window coverage must
  pass. Last-layer Y, final RMSNorm and all logits must match control FP16 bits.
- 48 underlying hardware/architecture source hashes remain unchanged. The
  software allocator adds only the Wo scratch alias and its bounds assertion;
  all prior allocation and capacity definitions remain byte-for-byte identical.
- The original worktree diff and its installed runtime library are checked
  against the saved pre-optimization state.
- The full baseline host suite passes 200 tests. Thirteen C++ regression tests
  cover the production window scheduler, staged GM lifetimes, residual input
  reuse, layer and LM residency, pipeline variants and bounded weight sharing.

An initial candidate was stopped because the asynchronous metadata-write ACK
callback did not restore `WriteCompletion`. That run is excluded from timing.
The corrected producer advances only after its timed metadata write ACK.

The software weight-cohort, block-pipeline, scatter-wave and array-window bridge
experiments remain available for comparison, but are disabled in the selected
residency configuration. They do not add GM capacity, array contexts or ports.

## Reproduction

With the repository's SST/RISC-V dependencies and built runtime available:

```bash
OPENBLAS_NUM_THREADS=1 python3 baseline/attention_cluster_8qk_8pv/run_all_gemm_reuse_window_baseline.py \
  --artifact-root results/input_residency_reproduction/fixed_candidate \
  --baseline-result results/input_residency_reproduction/fixed_control/reuse_window_probe/decoder_stack_result.json \
  --shape formal --run --window-a-prefetch 1 --lm-n-parallel 1 \
  --gm-stage-chain --layer-residency --lm-input-residency --input-residency \
  --cpu-set 8-15
```

First run the control with the same options, omitting `--input-residency`,
using `fixed_control` as its artifact root and a previously validated result
as `--baseline-result`. Both cases must use the same built library and guest.

```bash
python3 baseline/attention_cluster_8qk_8pv/summarize_input_residency.py \
  --experiment-root results/software_input_residency_next_20261021
```

Full local evidence is under `results/software_input_residency_next_20261021`.
Generated HBM images, binaries and multi-gigabyte logs are intentionally excluded
from Git; the verified comparison is published alongside this report.
