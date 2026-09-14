# Findings

## R11 Initial Constraints

- The modeled cluster already declares two operand banks per array and exposes
  bank-addressable program/launch methods, so the first revision should exploit
  existing capacity rather than add arrays or another programming port.
- R7 bank-aware multi-producer scheduling produced 9,404 E3 mismatches because
  the visible QK lease did not cover all downstream consumers. R11 must separate
  array-bank lifetime from score/P context lifetime and prove exact release.
- The accepted R10 path retains V across four grouped queries and already uses a
  one-entry P-row read lookahead. R11 must target cross-group V readiness and
  row-level P program/compute overlap rather than duplicate existing retention.
- The existing cluster ahead engine already transfers ownership at the correct
  boundary: it writes all QK output beats into the tagged score FIFO, releases
  the QK array lease, and only later releases the independent score context when
  SFU completes. It overlaps later grouped-query QK/SFU work with current PV.
- The remaining QK opportunity is therefore the physical key-tile boundary,
  not grouped-query QK. The safest first measured change is cluster P-row
  wavefront because its rows map to disjoint PV array pairs.
- A compute-only P-row wavefront is insufficient. Q256/K128 remains numerically
  exact but regresses 17,676 -> 17,691 cycles: it creates 36 versus six WCP
  launch commands per worker, reaches variable PV concurrency 18, and still
  waits for all rows before output drain. Future row streaming must carry the
  wavefront through PV readout/O, not stop after compute launch.
- Splitting worker-local K-ready from the final K/V callback exposes a useful
  520-cycle QK matrix overlap on Q256/K128. V completion then exposes a separate
  1,044-cycle PV matrix overlap; combined latency is 16,112 cycles.
- R11 E3/E4 are 127,589/430,131 cycles, 1.308x/1.243x the GPU baselines. The
  remaining gaps are 30,021/84,147 cycles. E4's maximum group boundary is now
  17,309 cycles, with nine cross-query K/V waits on the critical worker.

## Current Architecture

- The implemented Attention contract uses four managers and 16 workers.
- Each worker models 64 physical 64x64 arrays, 64 CUs per array, FP32, two
  operand banks, and a default 16-QK + 48-PV ownership split.
- D128 uses paired D64 paths. QK executes two waves; PV uses 32 arrays in one
  wave. The O engine provides a timed 16-lane, two-cycle FP32 FMA.

## R10 Performance Baseline

- R10 E3/E4 complete in 148,146/517,020 normalized cycles, 1.518x/1.494x
  the RTX 5060 Scope A result. The remaining gaps are 50,578/171,036 cycles.
- Critical-worker K/V ready wait is zero in both profiles. Per manager, all
  30/124 sequential lookahead loads hit while total HBM loads remain exactly
  one per physical tile and maximum distributor occupancy remains two slots.
- QK and PV are now balanced: 38,592/39,657 cycles on E3 and
  154,367/158,738 on E4. E4's 20,535-cycle maximum group boundary is the
  remaining long-tail target.
- Earlier R6 measurements of 330,125/1,007,628 cycles and 679.5/730.1 steady II
  are retained below as historical evidence, not the current baseline.

## R7 Result

- Moving the ahead scheduler call did not change cycles or overlap.
- An incomplete score-bank ownership change passed Q256 but failed E3 lifecycle.
- Removing the global ahead-QK producer gate caused 9,404 E3 mismatches. The
  visible QK lease does not cover all score/SFU/PV operand-bank consumers.
- All unsafe experiments were reverted. Fresh E3 returned to 330,125 cycles,
  zero mismatches, and numerical/lifecycle PASS.
- The completed two-versus-three K/V buffer A/B passed numerical and lifecycle
  verification with zero mismatches for both E3 and E4. E3 changed from 330,125
  to 330,300 cycles (+0.05%); E4 changed from 1,007,628 to 997,866 cycles
  (-0.97%).
- On the critical worker, E3 K/V wait changed from 105,272 to 105,443 cycles and
  steady II from 679.53 to 679.61. E4 K/V wait fell from 317,331 to 271,017
  cycles and steady II from 730.10 to 688.82, but boundary II slightly increased
  from 5,231.97 to 5,248.70 cycles.
- Three buffers reduce E4 prefetch long-tail pressure but do not solve boundary
  dependencies. The extra 32 KiB per worker is not justified as an unconditional
  default for a mixed sub-1% end-to-end result; it remains a candidate adaptive
  policy for longer sequences.

## Cleanup

- The removed experiment tree occupied about 32 GiB and contained 5,639
  untracked files, primarily HBM images, tensors, logs, and raw statistics.
- Publication retains five final R6 cases with three JSON files each and a
  SHA-256 manifest. Superseded 8+8 and H64 transition documents were removed.
- Generated run directories are ignored so future experiments do not pollute
  repository status.

## R8 K/V Distribution Design

- The existing `GroupCtrlEndpoint` already has dedicated bidirectional links
  between each manager and its four workers and binds each endpoint's local
  `GlobalMemory`, but Attention runs currently set `GOLEM_CTRL_LINK_ENABLE=0`.
- GroupCtrl's current event and state machine implement the legacy guest-mailbox
  REQUEST/GRANT/DONE/FINISHED protocol. The K/V distributor must extend it
  without changing those message semantics.
- A real implementation must enable the links for Attention, coalesce four
  worker requests by generation/group/tile, fetch each K/V segment once at the
  manager, and deliver it into each worker's local K/V buffer. Merely suppressing
  worker DMA counters would not model the requested architecture.
- The selected data path is: manager DMA K/V into two bounded local scratch
  slots, manager local-read the completed pair, GroupCtrl delivers the pair to
  the four requesting endpoints, and each worker endpoint performs timed
  `localWriteAsync` operations into its own destination K/V buffer before
  invoking the RoCC callback.
- Whole-tile local accesses exceed the modeled Local-GM maximum request size, so
  manager reads and worker writes must be chunked. Worker-local 64 B/cycle ports
  then provide the requested per-destination delivery timing without inventing
  a zero-time data copy.
- Manager scratch must be explicitly reserved inside the configured Attention
  window; using the end of GlobalMemory directly would risk collision with its
  internal DMA completion slots.
- The first runtime integration selected the legacy control architecture. That
  script always starts guests with `argc=2`, omitting the four fused-Attention
  shape arguments, so every process exited normally without issuing work.
- GroupCtrl can instead be wired into the proven archive Attention architecture:
  `CPU_Builder` already returns the endpoint at port index 6, so only the
  manager/worker links and the control-plane compatibility check are required.
- The first valid distributed Q256/K128 run passed numerical and lifecycle
  verification with zero mismatches at 18,169 cycles, versus 30,607 cycles for
  the R6 worker-DMA baseline (40.6% lower latency).
- R8 E3/E4 complete at 176,313/687,791 cycles, 46.6%/31.7% below R6. Their
  critical-worker K/V wait is 8,798/38,985 cycles; PV work and query-group
  boundary dependencies are now the leading optimization targets.
- Q256/K128 distribution statistics conserve traffic: 32 worker requests become
  eight manager loads plus 24 coalesced requests, with 512 KiB manager HBM data
  delivered as 2 MiB across worker-local buffers.

## R9 PV-to-O Investigation

- The cluster PV path already uses the dedicated 512 B/cycle near-array output
  transfer and permits the next row read while prior O commits are pending.
  Increasing the existing output credit alone would therefore not remove the
  main serialization.
- Each PV row is materialized at the RoCC callback, split into eight 16-value
  segments, and submitted through one global 16-lane O read/ALU/write pipeline.
  The eight dimension panels consequently serialize despite addressing
  independent accumulator regions.
- Per key tile, near-array transfer service is only 32 cycles, while O read,
  ALU, and write service are each 128 cycles. Measured PV-output latency is
  357-470 cycles/tile.
- R9 will test a bounded row-fused accumulator with eight independent 16-lane
  panel banks. It retains the timed 512-byte near-array row transfer but replaces
  eight cross-component segment submissions with one tagged row operation.
- The legacy lifecycle gate treated SFU/PV overlap as architectural, but its
  measured duration came from the serialized eight-segment O-submit tail. Row
  fusion legitimately reduces it to zero on most E3/E4 workers while preserving
  124/504 cycles of QK/PV overlap per worker and exact O conservation.

## R10 Early K/V And PV Programming Investigation

- The cluster path already retains the programmed V matrix across the four
  logical query contexts that share `(generation, group owner, key tile)` via
  `clusterPvMatrixResident`; a second V-matrix retention mechanism would be
  redundant.
- P inputs differ for every query block, so cross-query reuse is invalid. The
  viable P-side optimization is bounded prefetch/program overlap, not retention.
- R10 must trace the existing N+1/N+2 K/V descriptor launch points and the PV
  output completion transition before moving requests earlier; distributor
  slots remain bounded at two and must retain cancellation/tag conservation.
- Existing N+2 launch is gated on both current K/V operand releases and on N+1
  already being ready. With two worker buffers, this often delays the request
  until the current tile is nearly retired.
- Manager K/V slots are released only after all four worker-local chunked writes
  acknowledge. Moving a worker request earlier is useful only when both a safe
  destination buffer and a manager slot are available; otherwise it merely
  moves the same bounded stall.
- `prefetchAttentionClusterPvInput()` supports one next P row, but no cluster
  hot-path call currently starts it. The measured cluster P-input phase is thus
  still a serial read-then-program chain across 16 rows.
- Cross-tile P prefetch cannot use the current `clusterPContext`: each next tile
  owns a different SFU P slot/tag. A safe first step is a bounded current-tile P
  read FIFO that issues multiple row reads after PV matrix readiness and commits
  them in row order to distinct PV arrays.
- Runtime counters confirm why worker-side N+2 never starts: E3/E4 report zero
  second-lookahead candidates because the active tile and ready N+1 descriptor
  occupy both worker buffers throughout a four-query group.
- A useful early-delivery mechanism therefore belongs in the manager: after a
  completed distributed slot is released, the free manager scratch slot can
  stage the next sequential K/V tile without a worker destination. Later worker
  requests coalesce into that resident slot and perform the normal timed local
  delivery. This needs explicit speculative-slot cancellation accounting.
- The existing one-entry `clusterPvNextInput` buffer is sufficient for the first
  P optimization. Starting row N+1's SFU read after row N has been accepted for
  array programming overlaps two already modeled operations without changing P
  identity or array ownership.
- SFU P FIFO reads use a bounded pending queue and a dedicated serialized read
  port schedule, separate from the write port. One-row lookahead therefore
  preserves modeled transfer latency/backpressure while allowing the array
  input program for row N to cover row N+1's FIFO read latency.
- Manager lookahead must be demand-paced: one completed real slot release may
  create at most one sequential speculative slot, and a speculative slot cannot
  trigger another lookahead merely by finishing its DMA/read. This prevents an
  unbounded scan through the sequence and keeps occupancy at the configured two
  scratch slots.
- The accepted one-row P overlap changes Q256/K1024 from 159,345 to 159,220
  cycles (-125, -0.08%). It reduces the critical worker PV phase from 23,807 to
  22,367 cycles, but shifts tile boundaries enough to increase measured K/V wait
  from 77,711 to 79,026 cycles; manager-side K/V readiness remains the larger
  opportunity.
- Manager-resident lookahead changes Q256/K1024 from 159,345 to 123,821 cycles
  (-35,524, -22.3%) with P overlap disabled. Per manager, all 32 physical K/V
  tiles are still loaded exactly once; 30 are speculative loads and all 30 hit,
  with 128 worker deliveries, zero slot stalls, and a two-slot maximum.
- The critical worker's 31 N+1 consumptions change from 0 hits/31 waits to
  25 hits/6 waits. Worker N+2 launch rate rises from 8.3% to 77.5% because the
  manager has already staged data when local storage becomes reusable.
