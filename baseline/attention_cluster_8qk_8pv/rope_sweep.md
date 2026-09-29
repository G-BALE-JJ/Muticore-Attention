# Causal FP16 RoPE sweep

All runs use four SST ranks, the 8 QK/SFU + 8 PV mapping, full-head
interleaved RoPE, and base 10000. `E2E` starts when QK windows begin and ends
at final PV completion. `Worker` starts at the first worker dispatch, so it
also includes any exposed RoPE table loading. The static EXP floor counts
the causal QK tiles assigned to the busiest fixed QK worker and assumes 16
EXP lanes; other work and transfer latency can only raise completion time.

| Hq/Hkv | S | D | EXP floor | E2E no RoPE | E2E RoPE | Worker RoPE |
|---|---:|---:|---:|---:|---:|---:|
| 1/1 | 2048 | 64 | 31,232 | 36,046 | 39,385 | 40,612 |
| 1/1 | 4096 | 64 | 123,904 | 128,727 | 127,272 | 140,055 |
| 2/1 | 1024 | 64 | 14,848 | 18,722 | 21,057 | 21,755 |
| 2/1 | 2048 | 64 | 58,368 | 63,501 | 67,695 | 68,922 |
| 2/1 | 4096 | 64 | 231,424 | 271,245 | 234,975 | 247,758 |
| 2/1 | 1024 | 128 | 14,848 | 20,488 | 30,106 | 31,320 |
| 2/1 | 2048 | 128 | 58,368 | 65,728 | 64,766 | 77,549 |
| 2/1 | 4096 | 128 | 231,424 | 239,753 | 240,680 | 266,885 |
| 32/8 | 2048 | 64 | 933,888 | 940,702 | 943,860 | 945,073 |

Every listed run passed SST, numerical, MPI placement, and HBM layout
verification. A shorter RoPE E2E than the no-RoPE run at some lengths is a
scheduling effect: the two runs do not issue QK windows at identical times.
It does not imply that rotation has negative compute cost. In the target
`32/8` case, RoPE E2E is 1.011 times the static EXP floor. Further QK
arithmetic tuning has little room to improve that full request.

Without cross-job retention, the `32/8` run loaded the same 256 KiB table 64
times (16 MiB). Retaining it in each QK worker reduced this to 8 loads
(2 MiB) and 56 local hits. Numerical output remained valid; E2E stayed at
943,860 cycles because EXP already dominated. The largest single table load
was 6,711 cycles. The vector unit issued 2,320 RoPE operations covering
9,502,720 FP16 elements. The transformed Q/K panel cache is limited to
320 KiB per QK worker; longer windows can evict panels and issue more vector
operations. For example, `2/1, S=4096, D=128` issued 1,664 operations under
this limit.

Reproduce the sweep with `scripts/sweep_attention.py --causal --rope`
and the same command without `--rope`, varying `--head-dim`, `--pairs`, and
`--lengths`. The script reports both the perfect-overlap resource floor and
the static EXP floor. Large simulator artifacts are retained locally outside
the Git repository.
