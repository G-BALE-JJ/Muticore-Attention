# SFU Hardware Model

The Attention SFU uses the same architectural abstraction as the Golem MVM
arrays: host code evaluates functional values, but a value becomes visible only
after a finite physical resource accepts the operation and its completion event
arrives. This is a cycle/resource model, not gate-level RTL simulation.

## Physical Resources

The fixed sequential-64 QK1024 preset instantiates these resources in every
worker:

| Unit | Default lanes | Latency | II | Role |
|---|---:|---:|---:|---|
| Scale MUL | 16 | 3 | 1 | multiply scores by `1/sqrt(D)` |
| MAX compare | 16 | 1 | 1 | compare one 16-value beat |
| MAX reduction | 16 inputs | 4 | 1 | reduce a beat to one maximum |
| Vector EXP | 16 | 8 | 1 | shifted exponentials |
| SUM reduction | 16 inputs | 8 | 1 | FP32 sum of one EXP beat |
| Online MAX | 1 | 1 | 1 | merge old and tile maxima |
| Online EXP | 1 | 8 | 1 | compute the non-trivial rescale factor |
| Online MUL | 2 | 3 | 1 | two independent rescale products |
| Online ADD | 1 | 3 | 1 | merge the two partial sums |
| Reciprocal | 1 | 8 | 1 | final `1/l` operation |
| Normalize MUL | 16 | 3 | 1 | multiply probabilities by `1/l` |

Every resource has a finite accepted-token capacity, defaulting to 16. A token
records its issue and completion cycle. Latency controls result availability;
II controls the next issue opportunity. Full resources backpressure the owning
row context until capacity becomes available. `1/sqrt(D)` is descriptor/setup
state and is not charged as a per-tile RSQRT operation.

## Functional Commit

The simulator evaluates FP32 vector results and the existing online-state
arithmetic with host operations only when the matching completion event fires.
Host execution time never contributes to simulated time, and no result is
published at issue time.

This matches the MVM model's abstraction boundary. It does not claim bit-exact
equivalence to a particular vendor EXP or reciprocal IP. A future bit-accurate
mode can replace the host functions with a selected LUT or polynomial without
changing the physical pipeline model.

## Configuration

Every operator latency and initiation interval is independently configurable:

```text
GOLEM_SFU_SCALE_LATENCY                 GOLEM_SFU_SCALE_II
GOLEM_SFU_MAX_COMPARE_LATENCY           GOLEM_SFU_MAX_COMPARE_II
GOLEM_SFU_MAX_REDUCTION_LATENCY         GOLEM_SFU_MAX_REDUCTION_II
GOLEM_SFU_EXP_LATENCY                   GOLEM_SFU_EXP_II
GOLEM_SFU_SUM_REDUCTION_LATENCY         GOLEM_SFU_SUM_REDUCTION_II
GOLEM_SFU_ONLINE_MAX_LATENCY            GOLEM_SFU_ONLINE_MAX_II
GOLEM_SFU_ONLINE_EXP_LATENCY            GOLEM_SFU_ONLINE_EXP_II
GOLEM_SFU_ONLINE_MUL_LATENCY            GOLEM_SFU_ONLINE_MUL_II
GOLEM_SFU_ONLINE_ADD_LATENCY            GOLEM_SFU_ONLINE_ADD_II
GOLEM_SFU_RECIPROCAL_LATENCY            GOLEM_SFU_RECIPROCAL_II
GOLEM_SFU_NORMALIZE_VECTOR_LATENCY      GOLEM_SFU_NORMALIZE_VECTOR_II
GOLEM_SFU_PIPELINE_QUEUE_DEPTH
GOLEM_SFU_TILE_STREAM_ENABLE
GOLEM_SFU_TILE_STREAM_BYTES_PER_CYCLE
GOLEM_SFU_TILE_STREAM_BASE_LATENCY_CYCLES
GOLEM_SFU_ROW_DISPATCH_INTERVAL_CYCLES
```

Lane counts remain configurable through `GOLEM_SFU_VECTOR_LANES` and
`GOLEM_SFU_EXP_LANES`; the sequential-64 runner sets both to 16. For example:

```bash
GOLEM_SFU_EXP_LATENCY=10 \
GOLEM_SFU_EXP_II=2 \
GOLEM_SFU_PIPELINE_QUEUE_DEPTH=32 \
scripts/test_flash_attention.sh --sequential-64 \
  --queries 1024 --keys 1024 --head-dim 128
```

`GOLEM_SFU_HW_PIPELINE` reports accepted tokens, aggregate issue wait,
backpressure, and maximum occupancy for each unit. `GOLEM_TENSOR_LOCAL_STATS`
separately reports SRAM traffic and local-memory retry behavior.

## QK1024 Compute Lower Bound

One worker handles 16 Bc64 key tiles. Each tile contains 64 rows of 64 scores.
A 16-lane vector unit therefore consumes four beats per row.

For an ideal beat-streamed implementation, the first non-final row takes:

```text
MAX path       = 3 + 1 + 4 + 3 issue gaps = 11 cycles
EXP/SUM path   = 8 + 8 + 3 issue gaps     = 19 cycles
online merge   = 1 + 8 + 3 + 3            = 15 cycles
normalize      = 3 + 3 issue gaps          =  6 cycles
first row                                      51 cycles
steady row II = max(4 vector beats, 1 scalar token) = 4 cycles
```

The final key tile adds an 8-cycle reciprocal. Under the current worker/SFU
tile barrier, but with ideal beat streaming and zero SRAM/controller cost:

```text
non-final tile = 51 + 63 * 4 = 303 cycles
final tile     = 51 + 8 + 63 * 4 = 311 cycles
16-tile floor  = 15 * 303 + 311 = 4,856 cycles
```

The current controller does not stream the four beats of one row at II=1. It
waits for each beat's completion before requesting the next beat. Keeping that
dependency, but removing all SRAM and queue delay, gives a second, implementation-
constrained floor:

```text
MAX            = 4 * (3 + 1 + 4) = 32 cycles
EXP/SUM        = 4 * (8 + 8)     = 64 cycles
online merge                         15 cycles
normalize      = 4 * 3           = 12 cycles
first row                           123 cycles
non-final tile = 123 + 63 * 4 = 375 cycles
final tile     = 375 + 8       = 383 cycles
16-tile floor  = 15 * 375 + 383 = 6,008 cycles
```

Each 16 KiB score read or P write costs `1 + 16384 / 256 = 65` cycles. The
current worker dependency serializes both transfers around every tile, adding
`16 * 2 * 65 = 2,080` cycles. The resulting bounds are 6,936 cycles for ideal
beat-streamed arithmetic and 8,088 cycles for the current controller. They are
Softmax bounds, not a new end-to-end Attention bound.

## QK1024 Measurement

The default fixed run passes backend, numerical, and lifecycle verification:

| Metric | Result |
|---|---:|
| End-to-end Attention | 27,529 cycles |
| Input movement | 137 cycles |
| QK | 6,736 cycles |
| Softmax | 8,870 cycles |
| PV | 9,848 cycles |
| Checked FP32 outputs | 131,072 |
| Mismatches | 0 |
| Maximum absolute error | 1.713e-09 |

The measured Softmax is 782 cycles above the 8,088-cycle current-controller
plus stream floor (`1.097x`, or 9.67% overhead), and 1,934 cycles above the
6,936-cycle ideal streamed floor (`1.279x`, or 27.88% overhead).

The critical worker records the exact operator demand:

| Unit | Tokens | Issue wait | Backpressure |
|---|---:|---:|---:|
| Scale MUL | 4,096 | 23,711 | 2,364 |
| MAX compare | 4,096 | 0 | 0 |
| MAX reduction | 4,096 | 0 | 0 |
| Vector EXP | 4,096 | 7,569 | 124 |
| SUM reduction | 4,096 | 0 | 0 |
| Online MAX/EXP/MUL/ADD | 1,024 each | 0 | 0 |
| Reciprocal | 64 | 0 | 0 |
| Normalize MUL | 4,096 | 9,071 | 0 |

Issue-wait values are sums over overlapping contexts and must not be added to
the 8,870-cycle critical path.

## Why Measurement Is Larger

The coalesced path performs exactly 16 score reads and 16 P writes, preserves
262,144 bytes in each direction, and reports zero SFU LocalGM retries. Across
all traffic, LocalGM queue high-water falls from 32 to 4 and rejected requests
fall from 143,763 to zero.

The dispatcher launches row zero immediately and rows 1-63 every four cycles,
matching four 16-lane beats per row. Fixed QK1024 therefore records exactly
1,024 row dispatches per critical worker. Total hardware backpressure falls
from 43,974 to 2,488 events (94.34%); Scale falls from 42,832 to 2,364 and
Normalize reaches zero.

Softmax nevertheless rises by 46 cycles from R31. The removed retry polls were
mostly overlapping bookkeeping, while fixed pacing adds a small drain tail for
the final rows. The unchanged 8,088-cycle lower bound already assumed a
four-cycle steady row II, so dispatch does not lower it. The remaining
782-cycle gap includes event/controller boundaries and aligned downstream
completion bursts. True within-row beat streaming and downstream-ready launch
would be the next refinement, but R32 shows that dispatcher polling was not a
material end-to-end bottleneck.
