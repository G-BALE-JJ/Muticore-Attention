# GEMM mechanism alignment

Source: /data/jjgong/RISC-V-CIM-Manycore-SST, cc1b454 plus inspected local build fixes.
Base: LLM main 3f519e9. Worktree: exp-gemm-mechanism-alignment.

Preserve default hardware, arithmetic, guest ABI, Attention/GQA/decoder callbacks,
selected guest hash and full functional workload. New precision or placement
contracts are explicit opt-ins. No TileLang compiler/TileBin dependency.

Each feature is committed independently and must pass all four formal functional
stages before the next feature is applied. Record checks, exact installed-library
hash, numerical evidence, cycles and source commit. A failed gate stops rollout.

1. Partial-C next-window prefetch readiness guard; targeted out-of-order test.
2. Per-pseudo-channel HBM read-command statistics v2; pure observation.
3. CU/TU/TE separation with compatibility interfaces and exact legacy arithmetic.
4. Packed eight-mode precision and calibrated PE timing, opt-in versioned contract.
5. Explicit workload partitions / HBM panel placement without compiler dependency.
6. Timing numerical-data fast path; functional path unchanged, paired timing checks.

Four formal baseline cycles: GEMM 3500; Attention 12852; GQA 37928;
prefill 6084167.75. Full prefill boundary: first token-ID DMA send through
last valid-token full-vocabulary logits final DMA ACK; host argmax excluded.

The selected public library pin stays immutable until all gates pass. Each gate
writes a private configuration pinned to its actual installed library, retaining
the selected guest digest and schedule; this is recorded as candidate evidence,
not an unvalidated replacement for the public selection.
