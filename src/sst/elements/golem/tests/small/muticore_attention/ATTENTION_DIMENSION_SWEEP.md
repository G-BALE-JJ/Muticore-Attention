# Sequential-64 Attention Dimension Sweep

## Scope

All runs use FP32, non-causal `B=1`, `H=1`, `D=128`, `Q=K`, Br64/Bc64,
16 workers, 64 arrays per worker, Ramulator2 HBM, 256 B/cycle on-chip
broadcast/scatter/O fabrics, the calibrated SFU defaults, and one SST MPI rank.
Every reported point passes memory-backend, numerical, and lifecycle checks.

## Theoretical End-to-End Lower Bound

For one worker query block, let `R` be its active query rows (`1..64`),
`L=ceil(log2(R))` the PV broadcast-tree depth, and `T=K/64` its number of key
tiles. The current dependency-preserving bound is:

```text
QK/PV operands and arrays       = (735 + 2*L)*T
SFU compute plus score/P stream = (6*R + 121)*T + 8
old-O restore plus O output     = (2*T - 1)*(2*R + 8)

one-block lower bound = T*(872 + 2*L + 10*R) - 2*R
```

The final `8` cycles are the last-tile reciprocal. Full Br64 query blocks are
serialized within a worker, so Q2048 and Q4096 multiply the one-block result by
two and four. The deduplicated HBM resource floor is lower than this worker
dependency chain for all five points and does not raise the bound.

## Results

| Q=K | Worker blocks | Theory | Actual | Gap | Over theory | Input | QK | Softmax | PV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 1 x Br16 | 4,128 | 5,541 | 1,413 | 34.23% | 82 | 1,684 | 899 | 1,480 |
| 512 | 1 x Br32 | 9,552 | 11,128 | 1,576 | 16.50% | 89 | 3,368 | 2,560 | 3,624 |
| 1,024 | 1 x Br64 | 24,256 | 27,529 | 3,273 | 13.49% | 137 | 6,736 | 8,870 | 9,848 |
| 2,048 | 2 x Br64 | 97,280 | 105,232 | 7,952 | 8.17% | 329 | 26,944 | 35,468 | 39,664 |
| 4,096 | 4 x Br64 | 389,632 | 415,058 | 25,426 | 6.53% | 1,393 | 107,775 | 141,848 | 159,200 |

The phase columns are measured critical-worker intervals. They do not include
root/manager dispatch, completion return, and phase-boundary gaps, so they need
not sum to the end-to-end cycle.

## Interpretation

The relative gap falls monotonically from 34.23% to 6.53%. Fixed root/manager
startup and completion overhead dominate Q256: its measured phase sum is 4,145
cycles, 17 cycles above the 4,128-cycle worker bound, while end-to-end is 1,396
cycles beyond that phase sum. As dimensions grow, fixed overhead is
amortized. The remaining large-size residual is mostly repeated controller and
event boundaries: at Q4096, measured phases sum to 410,216 cycles, 20,584 above
the worker arithmetic/data-movement floor, plus 4,842 cycles outside the four
reported phases.

The sweep also exposed and fixed three tail/scale contracts:

- grouped QK output now accepts Br16/Br32 active rows while reading the fixed
  64-row physical array output stride;
- vector scatter and grouped PV O restore/drain treat 64 lanes as a maximum,
  allowing consecutive 16/32-lane tail groups;
- deferred K-first cross-query consumption now classifies both hit and wait
  outcomes, closing Q4096 lifecycle accounting without changing timing.

## Artifacts

The retained local artifacts are:

```text
/tmp/r33_attention_sweep/q256_final
/tmp/r33_attention_sweep/q512_final
/tmp/r33_attention_sweep/q1024_final
/tmp/r33_attention_sweep/q2048_final
/tmp/r33_attention_sweep/q4096_final2
```
