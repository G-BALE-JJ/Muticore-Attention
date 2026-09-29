# Long-sequence optimization and Llama attention shapes

Measured on 2026-09-28 with 4 MPI ranks, 8 QK/SFU workers, 8 PV workers,
FP32 operands, and a local-GM queue depth of 256. SRAM capacity, arrays,
HBM placement policy, and fabric bandwidth are held constant in the
before/after comparison.

## Measurement scope

The cycle boundary covers QK WCP, online SFU, P transport, PV WCP, and final
PV acknowledgements. It excludes model weight projections, RoPE, output
projection, other Transformer operators, and the other layers. The runner
uses dense, noncausal attention. Llama shapes reproduce its head counts and
head dimension; they are not a complete Llama inference benchmark.

For these historical FP32 measurements, PASS verifies completion, worker
placement, QK fusion tile counts, PV window counts, and the observed SFU exp
element count when hardware counters are available. These runs preceded the
HBM-output numerical verifier now used by the FP16 end-to-end sweeps.

## Root cause and changes

1. RoCC's V residency set recorded broadcast delivery history indefinitely.
   The WCP has only four cache entries and evicts entries with LRU. At
   S=4096, the old schedule started 1,918 of 4,096 PV windows with a real
   cache miss despite the upper layer treating the source as resident.
   Mean PV service grew from about 614 cycles at S=1024 to over 1,600 cycles.
2. The WCP now exposes actual cache residency. Scheduling checks that state,
   and an evicted, previously delivered V entry is refilled through the
   existing DMA prefetch path. A hit measured at PV start therefore does not
   mean that no HBM refill was needed before the window started.
3. QK's diagonal macro order interleaved key windows across query row groups.
   For KV lengths greater than 1024, key-window-major order keeps each key
   window together across row groups. Per-row softmax/PV window order remains
   increasing. Set `GOLEM_ATTENTION_WORKER_CLUSTER_WINDOW_MAJOR=0` to ablate.
4. PV headers, V head/window offsets, output accumulation, and completion
   tile counts now use D=64 or D=128. A V window is 64 KiB at D=64 and
   128 KiB at D=128. The HBM node rule remains
   `node = 1 + window / ceil((Skv / 256) / 4)`; only the dimension-dependent
   byte strides change.

The baseline runner now defaults to local-GM queue depth 256, matching the
sweep. The sweep records head dimension in its case ID, element-library
SHA-256, MPI ranks, queue depth, and scheduling policies. Resume checks these
fields to avoid reusing results from a different binary or policy.

## Controlled before/after comparison

All rows below use Hq=4, Hkv=2, D=128, and Sq=Skv=S.

| S | Before cycles | Optimized cycles | Reduction | Resource floor | Optimized/floor |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 18,590 | 18,590 | 0.00% | 8,192 | 2.2693 |
| 1024 | 47,817 | 47,817 | 0.00% | 32,768 | 1.4593 |
| 2048 | 282,982 | 241,792 | 14.56% | 131,072 | 1.8447 |
| 4096 | 1,161,522 | 1,015,210 | 12.60% | 524,288 | 1.9364 |

The 2048 reference is the previously completed queue-depth-256 sweep; 1024
and 4096 were reproduced in this session before editing the binary.

For S=4096:

| Metric | Before | Residency fix only | Window-major schedule |
| --- | ---: | ---: | ---: |
| End-to-end cycles | 1,161,522 | 1,101,227 | 1,015,210 |
| Mean PV receive-to-start wait | 204,991.5 | 25,221.6 | 457.0 |
| PV tail after final SFU completion | 210,519 | 17,109 | 1,171 |
| V misses at PV compute start | 1,918 | 0 | 0 |

## Remaining bottleneck for D=128

The final S=4096 QK span is 1,013,085 cycles and the SFU span is 1,013,508;
PV completion trails SFU by only 1,171 cycles. Removing PV stalls has exposed
the QK input supply path. Per QK worker, summed WCP compute time is 270,336
cycles and summed DMA wait averages 741,599.5 cycles across its two jobs.
The active input-not-ready component averages 573,509 cycles. These counters
are nested descriptions of the same waits and must not be added together.

Window-major scheduling trades V reuse for concentrated accesses to the
current K shard. The wait counters establish an input-readiness bottleneck;
they do not isolate HBM, NoC, scheduler admission, and local-SRAM installation
as independent causes. Further work should separate those components and
consider K fanout/reuse across QK workers, with SRAM and bandwidth held fixed.

From S=1024 to S=4096, the optimized cycles still grow by 21.23x for 16x the
attention work, versus 24.29x before. The abnormal PV backlog is substantially
reduced, but D=128 long-sequence scaling is not fully solved. Dense attention
itself necessarily grows quadratically when both Q and K lengths increase.

## Rejected alternatives

- Disabling row-completion priority with the original diagonal order took
  1,263,032 cycles at S=4096, compared with 1,161,522 before the fixes.
- Enabling existing cross-macro prefetch with the optimized order took the
  same 47,817 and 1,015,210 cycles at S=1024 and S=4096. It submitted and
  adopted prefetches but did not shorten the critical path; it remains off.

## Llama 3.2 1B head/dimension results

Hq=32, Hkv=8, D=64, batch=1, Sq=Skv=S. All four cases pass, each
completing 64 QK/SFU/end-to-end jobs with the expected PV windows and exp work.

| S | Cycles | Resource floor | Actual/floor | PV windows | Mean PV queue wait | PV tail after SFU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 70,158 | 65,536 | 1.07053 | 512 | 429.5 | 468 |
| 1024 | 267,869 | 262,144 | 1.02184 | 2,048 | 469.2 | 493 |
| 2048 | 1,054,344 | 1,048,576 | 1.00550 | 8,192 | 703.9 | 663 |
| 4096 | 4,200,595 | 4,194,304 | 1.00150 | 32,768 | 518.6 | 1,187 |

For S=4096, the SFU exp issue bound is 4,194,304 cycles, while the array
MAC bound is 1,048,576 cycles per QK or PV stage. The measured SFU span is
4,198,956 cycles; QK spans 4,165,276 cycles and final PV completion trails
SFU by only 1,187 cycles. The expected 536,870,912 exp elements are observed
in the hardware counters. These facts make SFU throughput the principal
limit in this dense D=64 workload, with little end-to-end gain available by
removing just the remaining PV tail.

Each PV worker is busy for approximately 1.401 million cycles, about 33.4%
of the end-to-end interval. QK compute totals 1,081,344 cycles per worker,
about 25.7% of that interval. These are stage-specific compute-time ratios,
not general hardware utilization measurements. QK WCP wait counters are
large, but removing those waits alone cannot beat the unchanged SFU bound;
they must be interpreted with the overlapped production/consumption path.

Successive sequence doublings increase cycles by about 3.818x, 3.936x,
and 3.984x, versus 4x attention work. At S=4096 the excess over the floor is
6,291 cycles (0.15%). The worsening actual/floor trend observed for the
earlier small-head D=128 sweep is absent in these four Llama shape cases.

For further speedup at these shapes, evaluate SFU lanes/issue throughput and
the QK:PV resource split with an explicitly updated theoretical floor. More
PV capacity is unlikely to improve the current critical path. Causal prefill
with actual masked-work skipping, BF16 operand transport, and Q=1 decode need
separate implementation and measurements before claiming Llama inference
performance. FP16 numerical-output validation was added after these historical
FP32 measurements.

## Reproduction and artifacts

```bash
python3 scripts/sweep_attention.py --pairs 4:2 \
  --lengths 512,1024,2048,4096 \
  --output-root results/long_sequence_window_major --timeout 1800

python3 scripts/sweep_attention.py --pairs 32:8 --head-dim 64 \
  --lengths 512,1024,2048,4096 \
  --output-root results/llama32_8_d64 --timeout 3600
```

Artifact roots relative to the repository:

- `results/long_sequence_before/`: fresh original-binary reference.
- `results/long_sequence_residency/`: residency fix alone.
- `results/long_sequence_window_major/`: optimized D=128 sweep.
- `results/long_sequence_row_priority_off/`: row-priority ablation.
- `results/long_sequence_cross_macro/`: cross-macro-prefetch ablation.
- `results/llama32_8_d64/`: Llama head/dimension sweep.

Each root contains `summary-latest.csv` and `summary-latest.json`; each case
contains `sst_qk_bridge_result.json`, runtime logs, and simulator statistics.

The optimistic resource floor is
`max(ceil(Hq*Sq*Skv*D/(8*4096)), ceil(Hq*Sq*Skv/(8*16)))`.
It excludes finite transfer/startup time and assumes perfect pipeline overlap.
Actual/floor is an overhead ratio, not a speedup or measured hardware
utilization. Stage spans overlap and must not be summed.
