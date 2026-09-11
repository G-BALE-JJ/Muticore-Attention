# FlashAttention Baseline

`scripts/test_flash_attention.sh` is the canonical entry point for the locally
built FlashAttention workload. It accepts dimensions and MPI ranks directly
and defaults to `Q1024,K1024,D128`:

```bash
scripts/test_flash_attention.sh
```

Use explicit values for larger or smaller runs; E3/E4 names are retained only
for checked-in historical baseline artifacts:

```bash
# From the repository root:
scripts/test_flash_attention.sh --queries 2048 --keys 2048 --head-dim 128
scripts/test_flash_attention.sh --queries 4096 --keys 4096 --head-dim 128 \
  --timeout 28800
```

The scale/archive architecture supports query-block MPI with 2 or 4 ranks:

```bash
scripts/test_flash_attention.sh --queries 1024 --keys 1024 --head-dim 128 \
  --mpi-ranks 2
scripts/test_flash_attention.sh --queries 1024 --keys 1024 --head-dim 128 \
  --mpi-ranks 4
```

It uses `sst.self` so the explicit manager/worker placement in the archive
configuration is retained. The post-run verifier rejects missing, duplicated,
or misplaced cores.

The wrapper still accepts scale-runner options, for example `--dry-run`,
`--artifact-root`, and the optimization flags documented by
`run_fused_attention_scale.sh`.

The default run enables generic GEMM/WCP for QK and PV, uses the modeled
binary-tree matrix broadcast for both operators, and rolls N+2 K/V DMA into a
released physical buffer while N+1 remains resident. PV additionally uses a
32-column active-K array launch, a bounded/bandwidth-modeled 16 KiB V-tile
buffer, and the verified input/restore/output/early-compute overlap bundle.
QK/PV tiling, numerical order, and Attention mapping are unchanged. The lower
runner retains
`--no-pv-matrix-broadcast`, `--no-qk-matrix-broadcast`, and a run without
`--kv-double-buffer` or `--kv-second-lookahead` for paired controls, plus
`--no-pv-*` switches for single-mechanism ablations. Frozen
baseline comparison is opt-in with `--baseline FILE` because arbitrary shapes
do not have a matching checked-in baseline.

Current verified MPI2 totals are 1,163,611 cycles at Q1024/K1024/D128 and
4,112,550 cycles at Q2048/K2048/D128. Checked-in E3/E4 baseline artifacts that
carry older cycle values remain historical records until regenerated through
the full baseline-freeze workflow; the parameter-driven runner does not select
them implicitly.
