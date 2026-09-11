# Archived FlashAttention Q1024/K1024/D128 Baseline

This directory preserves the pre-N+2 frozen result formerly named E3:

```text
B1,H1,S1024,D128,FP32
4 manager cores, 16 worker cores, 1, 2, or 4 MPI ranks
```

Run the same shape through the current unified command:

```bash
scripts/build_and_install_local.sh --reconfigure --jobs 16
scripts/test_flash_attention.sh --queries 1024 --keys 1024 --head-dim 128
scripts/test_flash_attention.sh --queries 1024 --keys 1024 --head-dim 128 --mpi-ranks 2
scripts/test_flash_attention.sh --queries 1024 --keys 1024 --head-dim 128 --mpi-ranks 4
```

The build command installs SST elements and builds the runtime-parameterized
FlashAttention RISC-V guest. The test command never invokes a compiler. Repeat only the test command
when the local install and guest binaries are unchanged.
Generated logs, HBM images, tensors, and detailed statistics stay under the
artifact directory and are not part of this baseline record.

The numerical gate checks all 131072 output values. The lifecycle gate checks
manager dispatch, worker QK/Softmax/PV completion ordering, output DMA ACK, and
the single tensor completion.

The archived result freezes the architecture path: all Attention GEMMs use the
generic WCP flow, QK and PV matrix programming use the modeled binary-tree
broadcast, and N+1 K/V DMA overlaps the current tile through two Local-GM
buffers. QK/PV tiling and Attention mapping remain unchanged. The fabric
contract is fanout 16, 64 B/cycle, one base cycle, and one cycle per tree stage.
Exact broadcast traffic plus critical-worker prefetch DMA/hit/wait timing are
part of every rank-mode record. Its N+1 result is 1,747,850 normalized cycles
for rank 1, 2, and 4. The current unified runner additionally enables rolling
N+2 prefetch, so this directory is a historical comparison fixture rather than
the default pass/fail target.

The single-rank result is recorded in `result.json`; the query-block MPI result
is recorded in `mpi2/result.json`, and the one-group-per-rank result is in
`mpi4/result.json`. The unified runner also forces
the current worktree `install/` library so a stale build-tree library cannot
silently replace the tested artifact.
