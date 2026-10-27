# Projection reuse with two resident weight tiles

The D64 projection path now keeps a pair of adjacent 64-column weight tiles
in the two physical operand banks while processing a 256-row block. After
each 16-row group, it saves FP16 partial outputs in timed local GM. The next
weight pair restores those partials and continues accumulation in the same
input-column order. Only the final pair writes raw Q/K/V and packed panels
to HBM.

This path is enabled for D64 with an even number of input tiles, hidden>=256,
and sufficient local GM scratch. Existing row-major and D128 paths remain
available for other configurations. At hidden=2048, the scratch fits within
2 MiB per manager: weights occupy 512 KiB, partial outputs 32 KiB, and the
256-row input block 1 MiB. D64 reuses the split-dimension raw staging region
for input, leaving 64 KiB free at the GM tail. The capacity check excludes
the final 64 bytes reserved for DMA sequence and status words. No additional array banks
or output contexts are assumed.

Each input gather reads two adjacent 64-element tiles per row (256 bytes),
with 16 separate SRAM requests per row group. These requests use the existing
GM queue and shared read port; requests exceeding the configured limit are
split, and queue backpressure retries on subsequent ticks. The operand
snapshot holds 4096 bytes. It avoids full hidden-dimension rereads during
each reuse iteration.

Pairing halves the number of partial-result round trips compared with
reusing one weight tile at a time. For hidden=2048 there are 16 weight pairs:
15 stores and 15 restores per 16-row group, rather than 31 of each. The
existing SFU, HBM, NoC, array ingress, compute latency, and output-DMA timing
models also apply to this path.

## Reproduce

```bash
source scripts/env_local_install.sh
scripts/build_and_install_local.sh --jobs 8
python3 baseline/attention_cluster_8qk_8pv/run_llama_projection.py \
  --artifact-root results/llama_projection_pair_reproduction
```

The runner preserves executable and input hashes, resolved config, numerical
checks, and raw logs. It supports both the frozen row-major baseline and the
new paired path. The projection report independently checks the expected
number of weight programs, reuses, input blocks, and SRAM bytes.

The measured reference is [LLAMA_PROJECTION_BASELINE.md](LLAMA_PROJECTION_BASELINE.md)
and its JSON, committed at `046f778`. It includes the prior SRAM timing audit.
Inputs are deterministic synthetic tensors with the Llama 32:8/D64/hidden2048
shape. This measures RMSNorm -> projection -> RoPE -> causal GQA -> O;
Wo, residual, and MLP are outside this experiment.

S1024 was independently repeated with different host CPU bindings. Both runs
completed in **3,848,933 simulated cycles**; all four verification/performance
reports match exactly. All four nodes' RMSNorm and raw Q/K/V bytes also match
the frozen row-major baseline bit for bit at both S1024 and S2048, preserving
FP16 accumulation order.

At S1024, projection decreases from 4,623,334 to 3,409,481 cycles (26.3%), and
the full chain decreases from 5,062,292 to 3,848,933 cycles (24.0%). The final
O maximum absolute error remains 0.0001220703125. Compact frozen results and
provenance are stored in `llama_projection_pair.json` beside this document.

| S | Baseline E2E cycles | Optimized E2E cycles | Reduction | Optimized projection cycles | Weight programs, before -> after |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1024 | 5,062,292 | 3,848,933 | 24.0% | 3,409,481 | 98,304 -> 6,144 |
| 2048 | 10,379,121 | 7,951,808 | 23.4% | 6,630,789 | 196,608 -> 12,288 |

S2048 projection decreases by 26.8%. Its final O maximum absolute error is
also 0.0001220703125. All numerical, panel-layout, weight-image, MPI, QK/SFU/PV
work-count, and projection SRAM-byte checks pass at both sequence lengths.
The 101 related unit tests pass; D128/hidden256 remains at 161,566 cycles.

## Traffic accounting

At S=1024, each manager processes 48 projection heads and 16 row groups per
head. Weight programs decrease from 24,576 to 1,536 per manager, a 16x
reduction. The four-manager total decreases from 98,304 to 6,144.

| Per-manager SRAM traffic, S1024 | Row-major | Paired reuse |
| --- | ---: | ---: |
| Input reads | 48 MiB | 48 MiB |
| Weight reads for programming | 192 MiB | 12 MiB |
| Partial restores | 0 | 22.5 MiB |
| Total reads | 240 MiB | 82.5 MiB |
| Partial stores | 0 | 22.5 MiB |

S2048 doubles these amounts. HBM weight loads remain 12 MiB per manager for
both sequence lengths: the weight cache already avoided repeated HBM loads,
and the optimization removes repeated local reads and array programming.
All partial stores/restores are included in measured cycles.

The full-shape numerical check caught an initial input-staging overlap with
the DMA status words. Errors were confined to each 256-row block's last row;
small hidden dimensions had not filled that region. Moving D64 input into
the unused raw staging region and excluding the reserved tail removes this
overlap. Full Q/K/V checking is required even when the final Attention output
is within tolerance.

## Remaining costs

At S1024, manager 0 spends 1,671,168 phase cycles in launch/compute, unchanged
from the row-major baseline. Matrix programming drops from 958,464 to 59,904
cycles. Local-GM read residence decreases from 1,105,920 to 576,000 cycles;
the new partial stores take 115,200 cycles. Restoring partial output to arrays
takes another 126,720 cycles, and output readback takes 135,168 cycles.
These counters include setup and waiting and overlap across managers.

Array computation is now the largest projection phase. Future work should
measure overlap of SRAM reads, array input scatter, and partial transfers,
while retaining the same two-bank capacity and 66-cycle array compute model.
