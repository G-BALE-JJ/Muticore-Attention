# LLM mechanism migration

Six features plus an isolated native API repair pass seven gates: 28 full functional runs. Each gate passed
before the following feature was applied. Accelerator latency is in cycles.

| Mechanism | GEMM | Attention | GQA | prefill |
| --- | ---: | ---: | ---: | ---: |
| Partial-C readiness | 3500.0 | 12852 | 37928 | 6084167.75 |
| HBM statistics v2 | 3500.0 | 12852 | 37928 | 6084167.75 |
| CU/TU/TE separation | 3500.0 | 12852 | 37928 | 6084167.75 |
| Explicit packed precision | 3500.0 | 12852 | 37928 | 6084167.75 |
| Explicit workload/HBM placement | 3500.0 | 12852 | 37928 | 6084167.75 |
| Native TU API registration repair | 3500.0 | 12852 | 37928 | 6084167.75 |
| Timing numerical-data fast path | 3500.0 | 12852 | 37928 | 6084167.75 |

Every delta from the public baseline is zero. Hardware resources,
legacy arithmetic, the pinned guest and the selected schedule are retained.
The four workloads have different shapes and boundaries; do not add their latencies.

Prefill uses synthetic S1024/H2048/FFN8192, two layers, vocabulary 128256,
16 workers and MPI8. Sparse-valued inputs execute all dense tiles.
Its boundary is the first token-ID DMA send through the final valid-token
full-vocabulary logits DMA ACK. Host preparation, validation and argmax are excluded.

The timing-only GEMM before and after the numerical-data fast path is
3500 cycles with 4/4 output writebacks complete; the paired delta is zero.
A same-build unoptimized control additionally matches all recorded execution,
terminal, memory, DMA, NoC and HBM command statistics byte for byte.
Numerical correctness is established by the functional gates.

Packed precision and workload placement are explicit native opt-ins.
Installed-library TU/TE fixture tests cover the new interfaces, arithmetic,
cross-K accumulation, B replication and functional/timing event equivalence.
They do not establish NoC/HBM performance for a new opt-in mapping.
No TileLang/TileBin components were introduced.

Interface contracts and restrictions: 
[NATIVE_INTERFACES.md](../../maintenance/mechanism_alignment/NATIVE_INTERFACES.md).
The manifest records per-phase source commits, library hashes and evidence hashes.
Raw artifacts remain in the experimental worktree paths recorded there.
Interrupted phase 01 and initial phase 04 runs are retained separately and are not counted.
The phase 01 report finalization, phase 04 active-K and native API repairs are explained in
[RECOVERY.md](../../maintenance/mechanism_alignment/RECOVERY.md).
