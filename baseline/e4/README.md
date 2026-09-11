# Archived FlashAttention Q2048/K2048/D128 Baseline

This directory preserves the pre-N+2 frozen larger result formerly named E4:

```text
B1,H1,S2048,D128,FP32
4 manager cores, 16 worker cores, 1, 2, or 4 MPI ranks
```

Build once, then run the same shape through the unified interface:

```bash
scripts/build_and_install_local.sh --reconfigure --jobs 16
scripts/test_flash_attention.sh --queries 2048 --keys 2048 --head-dim 128
scripts/test_flash_attention.sh --queries 2048 --keys 2048 --head-dim 128 --mpi-ranks 2
scripts/test_flash_attention.sh --queries 2048 --keys 2048 --head-dim 128 --mpi-ranks 4
```

The numerical gate checks all 262144 output values. The lifecycle gate checks
QK, online Softmax, PV, output DMA, manager completion, ordering, and cycle
conservation. MPI runs additionally verify every ranked statistics file and
the 200-component placement manifest.

Each rank-mode record locks the archived architecture: generic GEMM/WCP,
modeled QK+PV binary-tree broadcast, unchanged QK/PV tiling and Attention
mapping, and N+1 K/V DMA overlap through two Local-GM buffers. The fabric uses
fanout 16, 64 B/cycle, one base cycle, and one cycle per tree stage. Exact
broadcast traffic and critical-worker prefetch DMA/hit/wait timing must match
the frozen result rather than merely producing numerically correct output. Its
N+1 result is 6,560,597 normalized cycles for rank 1, 2, and 4. The current
unified runner additionally enables rolling N+2 prefetch, so these files are
historical comparison fixtures rather than the default pass/fail target.

Wall-clock timing is intentionally excluded from the frozen correctness gate;
it depends on host load and MPI communication conditions.
