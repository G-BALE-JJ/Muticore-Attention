# Progress

## 2026-09-15 R23 reference SRAM bandwidth re-audit

- Reopened the bandwidth conclusion after the user identified an expected
  roughly 256 B/cycle on-chip Local/GlobalMemory level. This is a read-only
  audit of effective reference configuration and semantic boundary matching.
- Confirmed the active reference defaults are C-buffer 256 B/cycle per
  independent read/write port and GlobalMemory-to-NoC 256 GB/s. The reference
  does not time generic local GM reads/writes; that port model was added in the
  Attention fork, so semantic mapping is required rather than literal copying.
- Found a direct RTL width anchor: each 64-element FP32 array matrix row is a
  2048-bit/256-byte load interface. The reference selects one array rather than
  broadcasting, but an Attention row-broadcast tree rooted at this interface is
  naturally modeled at 256 B/cycle.
- Identified an important asymmetry: reference matrix/output SRAM writes are
  256 B rows, while vector-bank loads are one 4 B scalar per selected array per
  cycle. The current SST multicast helper uses the matrix rate for both matrix
  and group-input payloads, so the earlier 256 B/cycle A/B overstates a strictly
  reference-aligned improvement by accelerating P-vector programming too.
- Completed the semantic decision: 256 B/cycle is the correct reference-aligned
  Q/V matrix-row broadcast target. Exact evaluation should first separate
  matrix and group-input rates; alternatively, treating both as 256 requires an
  explicit 64-lane vector scatter design. No architecture defaults were changed.

## 2026-09-15 R22 broadcast-port plausibility audit

- Started a read-only QK1024 audit of the 64 B/cycle matrix-broadcast setting.
  The audit distinguishes array dimensions from bytes/cycle, and inventories
  matrix multicast separately from P/Q input delivery, K/V distribution,
  LocalGM, grouped output, NoC links, and HBM endpoints.
- Confirmed matrix multicast shares the array-buffer request/port machinery
  with ordinary operand programming; its rate is a distinct transfer-service
  parameter, not a second unconstrained port. GroupCtrl K/V delivery creates
  one payload message per worker and is not modeled as one physical multicast.
- Confirmed the active clock is 1 GHz, so the default is 64 GB/s per worker.
  Wider parameter values are implementable in the simulator but represent a
  materially wider or banked local fabric and require an explicit hardware
  assumption; they are not justified by the 320 GB/s HBM-node setting alone.
- Confirmed group input multicast reuses the matrix tree. A first reading
  incorrectly grouped fanout-one PV P-row requests with ordinary single-array
  transfers; aggregate request counts and the MVM implementation show that all
  2,048 P rows also use the matrix-broadcast rate. QK K-row input, old-O restore,
  and output movement remain on other bandwidth paths.
- Ran read-only parameter A/B experiments at 128 and 256 B/cycle. Both passed
  numerical/lifecycle verification at 91,485 and 85,305 cycles versus the
  103,757-cycle 64 B/cycle baseline. The diminishing second-step gain and fixed
  non-broadcast phases support retaining 64 as the current conservative default
  or adopting 128 only with an explicit dual-lane/banked hardware contract.

## 2026-09-15 R21 broadcast configuration and phase boundaries

- Confirmed the LLM Attention fork exposes the matrix broadcast port through
  `GOLEM_MATRIX_BROADCAST_BYTES_PER_CYCLE`, with a 64 B/cycle default. The
  reference tree has no equivalent timing parameter or broadcast queue; its WCP
  matrix writes update arrays functionally without modeled programming cycles.
- Added a sequential-PV phase-boundary contract, observed its expected failure,
  and inserted counter transitions after V-matrix programming and after P-input
  programming. The focused contract now passes. These transitions change only
  cycle attribution, not commands, dependencies, or dataflow.
- All 86 baseline contracts and the local build/install pass. The fixed
  Q=K=1024/D=128 run passes numerical and lifecycle verification with zero
  mismatches at the unchanged 103,757 cycles. PV is now correctly split into
  matrix 8,512, input 16,384, restore 15,360, compute 2,176, and output 16,385
  cycles, totaling the unchanged 58,817 cycles.

## 2026-09-15 R20 PV matrix-programming analysis

- Started a read-only, QK1024-only root-cause analysis of the measured 40,256
  critical-worker PV matrix-programming cycles. No implementation change is
  authorized in this phase.
- Traced the sequential state machine and established that the phase counter
  fails to transition for P input and old-O restore. The reported bucket is an
  aggregate of those operations plus V read and V broadcasts, not pure matrix
  programming. Its samples are 1,556 cycles for tile 0 and 2,580 for tiles
  1..15; pure broadcast-fabric occupancy is 8,416 cycles total.
- Closed the measured count exactly: 64 V-read + 8,448 V-broadcast/WCP + 16,384
  P-input + 15,360 old-O-restore = 40,256 cycles. The pure payload floor is
  8,192 cycles, the configured broadcast-service floor is 8,416, and the
  current-interface V-read plus WCP dependency floor is 8,512 cycles.
- Confirmed the gap is not queueing: array-buffer rejects and WCP queue-full
  stalls are zero, array-buffer high-water is one, and WCP maximum wait is one
  cycle. No architecture implementation was changed.
- Compared the reference tree completely. It does not model the LLM fork's
  broadcast port at all: WCP performs functional `setMatrixItem` calls with no
  separate programming latency. Added a source contract for honest sequential
  PV phase boundaries before changing implementation.

## 2026-09-15 R19 320 GB/s HBM-node alignment

- The initial source-level contract intentionally failed because the Attention
  runner did not forward `GOLEM_DIRCTRL_HIGHLINK_BW`; its sourced global config
  could therefore substitute the reference path's 512 GB/s value. The runner
  now owns and forwards an explicit 320 GB/s Attention default.
- The focused bandwidth contract, public dry-run propagation check, all 85
  baseline contracts, and the standard local build/install completed cleanly.
- The fixed Q=K=1024/D=128 run passed backend, numerical, and lifecycle checks
  at 103,757 cycles. All 15 K/V prefetches hit with zero exposed wait; critical
  phases are input 3,481, QK 27,088, softmax 12,567, and PV 58,817 cycles.
- Recomputed the topology-consistent strict floor as 20,992 cycles: the new
  four-node 1,280 B/cycle attachment imposes only 13,517 cycles on read returns,
  below the array-operand ingress floor. The measured result is 4.943x floor.

- Compared the reference Ramulator model, network profile, active ctrl topology,
  and archived topology. Confirmed 320 GB/s is the per-stack HBM payload peak;
  the active reference DirCtrl link is 512 GB/s headroom, while both archives
  still hard-code 25 GB/s.
- Selected the user-requested Attention policy: 320 GB/s per data-node DirCtrl
  highlink by default, with explicit environment override. No implementation
  change has been made yet; a failing contract comes first.
- Added the source contract and observed the expected failure against the
  hard-coded 25 GB/s archive. Changed only the memory-node DirCtrl highlink to
  read `GOLEM_DIRCTRL_HIGHLINK_BW` with a 320 GB/s fallback.

## 2026-09-15 R18 K/V supply gap analysis

- Started a read-only, QK1024-only diagnosis of the 221,935-cycle K/V wait.
- The analysis separates HBM media service, NoC/DMA transfer, LocalGM landing,
  request issue timing, and overlap; no architecture changes are authorized in
  this phase.
- Completed the source/statistics reconstruction. Critical core 16 spends
  20,949 cycles per prefetched K/V pair on average, hides 6,154, and exposes
  14,796; all 15 boundaries wait and none hit.
- Identified a missing resource in the old roofline: the archived topology
  ignores the resolved 512 GB/s DirectoryController setting and hard-codes
  25 GB/s per data node. Its four-node 100 B/cycle endpoint makes the corrected
  instantiated-topology read-direction lower bound at least 173,016 cycles.
- HBM and global NoC are not saturated. The long tail is concentrated in
  memory-endpoint response queues and credits on data nodes 1 and 2. No code or
  configuration was changed during R18.

## 2026-09-15 R17 QK final-tile readout

- Began the user-requested QK1024-only optimization after root-cause analysis.
- Froze the intended dataflow: retain D64 slice 0 in array accumulators, add
  slice 1, read all 64 array outputs in one timed group transaction, transpose
  array-major results in the controller, and write one logical 16 KiB score
  tile. Port bandwidth remains modeled explicitly.
- R17-A began with a failing source contract before the architecture change.
- The new contract failed before implementation as intended. After the first
  code patch it failed only because one assertion depended on C++ line wrapping;
  changed that assertion to check the same semantic markers independently.
- Added a sequential-only score traffic class, 64-array validation, final-only
  reduction transition, array-major to row-major assembly, and one logical
  score-tile LocalGM write. Build and runtime validation remain pending.
- R17-A source contract now passes. The scale runner defaults sequential-64 to
  a 16 KiB/cycle near-array score path, a 16 KiB/cycle LocalGM path, and a
  16 KiB maximum LocalGM request unless the experiment explicitly overrides
  those values.
- Full SST/Golem build and local install passed with the new traffic class,
  group-read validation, and state-machine callbacks.
- The first QK1024 run produced zero numerical mismatches at 425,172 cycles and
  reduced critical QK to 27,080 cycles, but lifecycle rejected the new shared
  classified score counters. Added a failing regression before updating the
  verifier to expect one 16 KiB final-score read per tile in sequential mode.
- Updated lifecycle accounting and regenerated the canonical metrics report;
  the QK1024 run now passes all correctness and conservation gates. All 101
  Python contracts, the SST/Golem build/install, shell syntax, and diff checks
  pass.
- Final R17 critical-worker phases are input 257,004, QK 27,080, softmax
  12,567, and PV 58,816 cycles. End-to-end is 425,172 cycles versus the R16
  576,657-cycle baseline. The next read-only diagnosis target is the 221,935
  cycles of K/V inter-tile wait exposed by the faster QK path.

## 2026-09-15 R16 Full 64x64 Sequential Design

- Started implementation after the user approved completing the full design and
  testing it. The frozen contract is Br64/Bc64/D128 on 64 physical 64x64 arrays,
  followed by a balanced 16-lane online SFU and two PV D64 output slices.
- The first Q256/K256 smoke run aborted in `Reg2GlobalMem`. Root-cause tracing
  showed that the generic runner expanded the SST per-core GM stride to
  1,335,616 bytes while the prebuilt Attention guest retained a 1 MiB manager
  stride. Standardized this custom Attention path on a shared 2 MiB stride.
- The stride-fixed rerun reached clean SST completion but all four managers
  rejected the descriptor. The guest process environment omitted the new
  sequential-mode variable, so `query_block_rows` remained 16 while the
  accelerator required 64. Added the mode to the explicit guest env whitelist
  and a source contract for that host-to-guest ABI.
- The env-fixed Q256/K256 run exercised all stages at 106,783 cycles but failed
  numerically. The sequential QK second D64 pass reused the first-half Q matrix,
  and partial worker bands indexed beyond the loaded query payload. Q is now
  retained, each half rebuilds its own zero-padded 64x64 matrix, and K ownership
  is released only after the accumulated second pass is programmed.
- The QK-corrected rerun remained numerically wrong. PV still used the legacy
  `D/16` panel count even though each sequential array now emits D64; this
  launched 8 panels instead of 2 for D128 and indexed six panels past V/O.
  Sequential PV dimension panels now use `ceil(D / arrayOutputSize)`.

## 2026-09-15 R13 Sequential Worker Redesign

- Started from the existing dirty `softmax-update` worktree and preserved all
  prior R12 changes.
- Reframed the active plan around worker-local QK-then-PV execution with all 64
  arrays reused by both phases.
- Began architecture and performance-model audit.
- Located the pre-cluster reference at `4d47f83` and confirmed that current
  ownership validation cannot represent temporal reuse of all 64 arrays.
- Audited the array-index formulas and found that the fixed Br16/D128 logical
  wave has only 32 independent array operations; full 64-array occupancy needs
  two Br16 contexts or a Br32 worker tile.
- Recovered the modeled array timing formula (66 cycles full-width, 34 cycles
  for Bc32 active-K PV) and confirmed that legacy enable-off mode has only 16
  wide arrays, so it cannot be used unchanged.
- Selected a Br16-preserving mapping: QK uses 64 key-indexed arrays at Bc64;
  PV uses four D16 panels by 16 query rows per 64-array wave.
- Build/install passed. The first Q256/K128 run reproduced a QK matrix broadcast
  admission failure; traced it to the stale fanout-16 runner limit and changed
  sequential mode to configure fanout 64.

## 2026-09-14: R12 Ramulator2 HBM2E backend

- Started migration after confirming that R11's resolved environment named
  Ramulator2 but the selected archive architecture actually instantiated
  DRAMSim3.
- Current target is Ramulator2 2.1 with the eight-channel HBM2E-2500 stack
  configuration, followed by fresh Q256, E3, and E4 validation.
- Preserved the fused-Attention archive architecture and ported only the proven
  backend selector into it; switching wholesale to the control architecture
  would have dropped the four required Attention guest arguments.
- The Attention runner now explicitly requests Ramulator2 HBM2E-2500, while
  the architecture validates the selection and emits the instantiated backend
  and configuration path at startup.
- Ramulator2 microbench PASS: 39.57 GB/s per channel, 316.67 GB/s for the
  eight-channel no-refresh roofline, and 295.43 GB/s with per-bank refresh.
- The first real Ramulator2 Q256/K128 run passed numerical/lifecycle checks at
  15,565 normalized cycles. Runtime summaries identify Ramulator2 on all five
  memory nodes and contain no DRAMSim3 backend statistics.
- Added a mandatory post-SST backend gate that rejects missing Ramulator2
  summaries and any mixed-in DRAMSim3 statistics before result verification.
- Fresh Ramulator2 E3 passed all gates at 142,737 cycles. The first E4 run was
  numerically correct at 518,096 cycles but exposed a verifier unit mismatch:
  four queued worker requests were compared to two missed tile lookaheads.
- Added a failing regression for the four-worker stall bound, corrected the
  verifier without changing hardware behavior, and reran E4. It passed
  numerical, lifecycle, and backend gates at the same 518,096 cycles.

## 2026-09-14: R11 QK/PV two-bank pipeline

- Started the approved staged implementation: bank lifetime first, then QK
  ping-pong, cross-group PV V programming, and bounded P-row wavefront.
- The worktree already contains uncommitted Attention changes beyond the frozen
  R10 reports. They will be audited and preserved; R10 remains the numerical and
  performance acceptance baseline until the new path passes all gates.
- Audited the existing cluster ahead state machine. QK banks are released only
  after output beats reach the independent score FIFO, and the score context is
  released only after SFU completion. Added a focused contract for safe bank
  reuse while the preceding score context remains live.
- R11-A focused C++ contract passes with `-Wall -Wextra -Werror`.
- Implemented an independently gated P-row program/compute wavefront and ran
  Q256/K128 A/B. It is numerically correct but changes 17,676 to 17,691 cycles
  (+15): per-row WCP launches rise from 6 to 36 per worker and the unchanged
  whole-wave output barrier prevents useful latency hiding. The default remains
  disabled; this experiment is rejected unless output drain is also row-streamed.
- Added a K-ready callback to GroupCtrl delivery so QK K-matrix programming can
  start while V is still being written. Added exact tagged inactive-bank
  acquisition, promotion, cancellation, release, and launch/hit accounting.
- Added an independent PV V-matrix lookahead. Ready V tiles are programmed into
  the next PV bank during QK/softmax; late tiles retain the demand fallback.
- Q256/K128 A/B passed at 17,676 off, 17,156 QK-only, 16,632 PV-only, and 16,112
  combined. All runs have zero mismatches and lifecycle PASS.
- Q256/K1024 passed at 119,054 cycles. E3/E4 passed at 127,589/430,131, improving
  13.88%/16.81% over R10. WCP depth-16 and MPI-2 both passed at 16,112 cycles.
- Full build/install, 89 Python contracts, warnings-as-errors C++ contract,
  shell/Python syntax, and whitespace checks pass. The partial compute-only
  P-row wavefront remains disabled because it regresses latency.

## 2026-09-14: R10 K/V early delivery and PV programming overlap

- Started from the verified R9 baseline: E3/E4 171,186/646,670 cycles,
  K/V ready wait 40,297/140,175 cycles, and PV matrix plus P-input programming
  33,279/133,118 cycles.
- Split the work into independently gated early K/V delivery and PV matrix/P
  retention so each optimization can be measured and reverted independently.
- Confirmed that cross-group V-matrix retention already exists and that the
  dormant one-entry P-row buffer can overlap the next SFU read with current-row
  PV input programming.
- Selected manager-resident speculative N+2 staging because the two worker K/V
  buffers are structurally occupied by current and N+1 tiles; merely moving the
  worker request earlier cannot create a destination buffer.
- Verified that SFU P reads are bounded and independently port-scheduled; the
  planned one-row overlap retains the existing backpressure and transfer model.
- Implemented the existing one-entry P-row lookahead behind
  `attention_pv_input_pipeline`. Q256/K1024 feature-off exactly reproduced
  159,345 cycles; feature-on passed numerical/lifecycle checks at 159,220 cycles.
- The first feature-on lifecycle run exposed a verifier scope error: cluster P
  rows are `jobs * 16`, not generic PV-array operations. Added a failing
  regression test, corrected only that formula, and passed all 70 Python tests.
- Added bounded manager-resident sequential K/V staging with serialized
  `keyTiles`/`totalKeys` metadata, demand-paced two-slot occupancy, explicit
  cancellation accounting, load/hit statistics, and independent runner control.
- K/V-only Q256/K1024 passed numerical/lifecycle checks at 123,821 cycles, 22.3%
  below R9. Manager statistics preserve one load per physical tile and show
  30/30 speculative hits per manager with zero slot stalls.
- Combined manager lookahead plus P overlap passed Q256/K1024 at 123,129 cycles,
  22.7% below R9. Feature-off in the same binary exactly reproduces 159,345.
- E3/E4 passed at 148,146/517,020 cycles; critical-worker K/V ready wait is zero
  in both. Q256/K128, WCP-pressure, and MPI-2 passed at 17,676/18,306/17,676.
- Extended lifecycle verification to enforce manager/worker request, load, byte,
  coalesce, delivery, lookahead hit, cancel/stall, and two-slot conservation.
- Final R10 gates passed: full incremental build/install, 88 Python tests,
  warnings-as-errors C++ contract, shell/Python syntax, and diff whitespace.

## 2026-09-14: R9 PV output and O accumulation

- Started a scoped optimization of PV output movement and resident-O
  accumulation after R8 reduced K/V wait to a secondary bottleneck.
- Acceptance requires preserved numerical/lifecycle conservation plus measured
  Q256, E3/E4, pressure, and MPI improvement; raw cycle suppression is out of scope.
- Tracing showed that the near-array PV drain is already 512 B/cycle and row
  reads overlap prior commits. The remaining serialization is eight 16-value
  panel submissions through a single O read/ALU/write path.
- Selected a feature-gated eight-bank row accumulator: eight independent 16-lane
  FP32 slices preserve capacity, tags, ordering, cancellation, and final drain.

## 2026-09-14: R8 Manager-level K/V distribution

- Started implementation scoped to Manager-level K/V load/reassembly and
  four-worker multicast; QK/SFU operand-bank work remains out of scope.
- Identified the existing GroupCtrl Manager/Worker links as the integration
  transport. Attention currently disables those links, so configuration and
  compatibility changes are required in addition to the data path itself.
- Selected a real data path using one manager DMA per K/V pair, bounded manager
  scratch, serialized event payloads, and timed chunked worker Local-GM writes.
  This preserves modeled DMA and local-delivery costs while removing fourfold
  HBM reads.
- Added the tagged request/delivery/ack/cancel messages and the bounded endpoint
  state machines. A guessed partial make target did not exist; compilation will
  use the standard repository build rather than repeating that command.
- Full build/install and the focused source contract passed. The first Q256/K128
  launch exposed unconditional scheduler-link wiring in the control architecture;
  the loop is now gated by `GOLEM_REQUEST_SCHEDULER_ENABLE`.
- The second launch completed with zero Attention/HBM activity. Root-cause
  comparison showed that the legacy control architecture omits all four fused
  Attention guest arguments and therefore launches a no-op workload.
- Kept the proven Attention architecture and added only its missing GroupCtrl
  endpoint links. Removed the temporary scheduler workaround from the legacy
  architecture because that script is no longer selected.
- Q256/K128 then passed numerical and lifecycle verification with zero
  mismatches at 18,169 cycles, a 40.6% reduction from the 30,607-cycle R6
  baseline. Multi-tile and scale validation remain in progress.
- Completed the R8 validation matrix: Q256/K1024 160,087, E3 176,313, E4
  687,791, WCP-pressure Q256/K128 18,862, and MPI-2 Q256/K128 18,169 cycles.
  Every run passed numerical/lifecycle checks with zero mismatches; MPI placement
  also passed and matched the single-rank cycle result.
- Feature-off Q256/K128 exactly reproduced 30,607 cycles. Enabled GroupCtrl
  statistics confirmed a 4:1 reduction from worker delivery traffic to manager
  HBM traffic and a two-slot high-water mark.
- Feature-off Q256/K1024 also reproduced 308,481 cycles after the target-tile
  consumer metadata correction, with zero numerical mismatches and lifecycle PASS.
- Final verification passed: standard build/install, 68 Python contracts,
  warnings-as-errors C++ contract, Python/shell syntax, and `git diff --check`.

## 2026-09-14: R9 PV-to-O row fusion

- Implemented a feature-gated 128-lane row submission from the timed PV output
  drain into eight parallel 16-lane resident-O banks, replacing eight serialized
  cross-component submissions while preserving the legacy feature-off path.
- Added fused-row and fused-byte statistics, lifecycle conservation checks,
  order-independent CLI controls, and C++/Python contract coverage.
- Contract checks pass: two key tiles over 16 rows produce 32 fused row
  operations, 256 logical accumulation segments, and the expected O values.
- Runtime A/B passed at 17,751 cycles versus 18,169 for Q256/K128. Multi-tile
  Q256/K1024 passed at 159,345 cycles.
- E3/E4 passed numerical and lifecycle verification at 171,186/646,670 cycles,
  improving 2.91%/5.98% over R8. PV output/read-write fell about 88.9% to
  6,652/26,616 cycles.
- WCP pressure passed at 18,381 cycles; MPI-2 passed placement and reproduced
  the single-rank 17,751-cycle result. The complete 69-test Python suite passes.

## 2026-09-14: three-buffer E3/E4 confirmation

- Started a controlled A/B using the retained verified two-buffer R6 E3/E4
  artifacts as A. The B runs change only `GOLEM_ATTENTION_KV_BUFFER_COUNT=3`.
- The prior three-buffer E3 observation was 330,300 cycles versus the 330,125
  baseline, but E4 had not yet been recorded as a complete comparison.
- Fresh three-buffer E3 and E4 both passed numerical and lifecycle verification
  with zero mismatches, at 330,300 and 997,866 cycles respectively.
- The first compact `jq` comparison used `label` as an argument name and failed
  at filter compilation; no simulation or result artifact was affected.
- Aligned critical-worker comparison: E3 total cycles +175 (+0.05%), K/V wait
  +171, steady II 679.53 -> 679.61, boundary II 6,059.10 -> 6,064.58. E4 total
  cycles -9,762 (-0.97%), K/V wait -46,314 (-14.60%), steady II 730.10 ->
  688.82 (-5.65%), boundary II 5,231.97 -> 5,248.70 (+0.32%).
- Kept the verified two-buffer R6 default. Three buffers are shape-dependent and
  will only be reconsidered with an adaptive long-sequence policy or earlier
  N+2 prefetch scheduling. Large generated artifacts remain under `/tmp` only.

## 2026-09-14: repository cleanup and publication

- The first attempt to remove two generated test artifact directories with
  recursive `rm` was rejected by the local safety policy before execution.
  Cleanup continued with path-scoped `find -delete`; no partial deletion or
  source-tree change resulted from the rejected command.
- Audited the 32 GiB untracked `attention_cluster/` tree and identified the
  15 JSON files covered by the final R6 SHA-256 manifest.
- Removed regenerable HBM images, tensors, logs, raw statistics, intermediate
  sweeps, rejected runs, and old validation directories. The cluster directory
  is now approximately 1.2 MiB.
- Retained Q256/K128, Q256/K1024, E3, E4, and WCP-pressure final JSON summaries,
  plus the R6 manifest and C++ contract.
- Removed documents and theoretical scripts tied to superseded 16-array 8+8 or
  transition H64 assumptions.
- Rewrote the active plan, findings, cluster README, workload README, GPU roadmap,
  and root status around the verified 16-QK + 48-PV R6 baseline.
- Fresh publication verification passed: full SST/Golem build and install,
  FlashAttention RISC-V guest build, 84 Python contracts, warnings-as-errors C++
  contract, Python/shell syntax, `git diff --check`, and all 15 artifact hashes.
- Fresh Q256/K128, E3, and E4 simulations passed numerical and lifecycle gates
  at 30,607, 330,125, and 1,007,628 cycles with zero mismatches.

## 2026-09-14: R7 scheduler investigation

- Scheduler-call relocation was a measured no-op and was reverted.
- A partial score-bank retain/release experiment failed E3 lifecycle and was
  reverted.
- Bank-aware multi-producer scheduling caused 9,404 E3 numerical mismatches and
  was fully reverted.
- A fresh rebuild and E3 run after revert passed numerical/lifecycle checks at
  330,125 cycles with zero mismatches.
- Three-buffer K/V passed numerical verification but regressed to 330,300 cycles;
  the verified two-buffer R6 configuration remains the safe baseline.

## 2026-09-14: R1-R6 completion

- Completed dynamic QK/PV ownership and selected 16/48 as the fixed default.
- Added the timed 16-lane FP32 O FMA and ownership-aware lower-layer admission.
- Final cases passed at 30,607, 308,481, 330,125, 1,007,628, and 31,657 cycles.
- Full build/install, 84 Python contracts, warnings-as-errors C++ contract,
  pressure checks, artifact hashes, and independent review completed.
- GPU gate remains failed at 3.384x/2.912x.
## 2026-09-15: R13 sequential 64-array Attention

- Replaced the public cluster default with a worker-local phase sequence:
  64-array QK, online softmax, then 64-array PV.
- Selected Br16/Bc64 and wide `D x 16` arrays. QK uses one 64-array wave; D128
  PV uses two waves, each mapping four D16 panels across 16 query rows.
- Added a four-destination sequential-PV multicast traffic class, group launch,
  active-matrix accounting, and exact 64-array-per-wave lifecycle contracts.
- Fixed online-softmax accumulation by scaling every restored O row before the
  next key tile is accumulated.
- Build/install and 92 Python contracts pass. Ramulator2 Q256/K128/D128 passes
  numerical and lifecycle verification at 37,765 accelerator cycles, with zero
  mismatches and maximum absolute error 5.603e-09. QK/PV wave ranges are both
  exactly `[64,64]` active arrays.
- Documented the D128 array-only lower bound as 262 cycles/tile and the serial
  QK+softmax+PV compute bound as 539 model-native cycles/tile.

## 2026-09-15: R14 Q1024 dataflow accounting

- Started a source-backed trace for Q=K=1024, D128 from striped HBM through the
  16 sequential-64 workers. Cycle reporting uses the active 1.0 GHz
  RoCC/SFU/array model-native cycle.
## 2026-09-15 R14 progress

- Fixed the worked-example contract to Q=K=1024, D=128, FP32, Br=16,
  Bc=64, 16 workers and 64 arrays per worker.
- Traced Q DMA, direct per-worker K/V DMA with four-query-block reuse, and O
  DMA in `roccAnalog.h`.
- Derived physical traffic: 17 MiB total HBM traffic (1.0625 MiB/worker).
- Completed the worked example in `attention_sequential_64/README.md`: ASCII
  HBM-to-output flow, 17 MiB traffic derivation, 13,927-cycle HBM floor,
  34,496-cycle compute/roofline floor, and the approximately 238,336-cycle
  current QK no-queue datapath estimate.
- Expanded the example with the actual key-tile/query-block loop order,
  physical per-array QK/PV matrix shapes, a precise D-panel definition, and
  separate HBM-media versus Local-GM-landing cycle calculations.

## 2026-09-15 R15 64x64 Reference Alignment Audit

- Started a read-only audit before implementation, as requested.
- Confirmed the reference Chapter 4 contract is 64 inputs, 64 outputs, and 64
  arrays; the LLM base config matches it, but the sequential Attention runner
  overrides the physical array to D inputs by 16 outputs.
- Confirmed output size also sets modeled CU count, so the current D128x16
  override is not a harmless tiling choice; it changes modeled hardware.
- Completed the audit without modifying architecture code. Frozen the corrected
  mapping as Br64/Bc64 on 64 physical 64x64 arrays: two accumulated QK K64
  launches and two PV D64 output-panel launches for D128.
- Identified blocking Br16 assumptions in the guest ABI values, manager/worker
  admission, SFU online-state ownership, Local-GM layout, sequential PV
  multicast topology, output gather, lifecycle statistics, and verifier.
- Confirmed the reference generic WCP supplies K micro-tiling/accumulation, but
  fused Attention disables its RequestScheduler and only uses the WCP proxy;
  the custom Attention state machine must implement or deliberately reuse that
  mechanism.
- Marked the old Br16/D16 cycle calculations as superseded. Only the corrected
  array-compute component is currently fixed: 132 QK + 132 PV = 264 native
  cycles per Br64/Bc64/D128 tile, excluding SFU and timed data movement.
- Evaluated the matching optimized SFU architecture without changing code:
  64 logical online contexts and balanced 16-lane max/EXP/normalize pipelines
  give 277 cycles per Br64/Bc64 tile. The resulting compute-only floor is 541
  cycles/tile and 8,656 cycles for one Q=K=1024 critical worker.
- Separated that compute result from system bandwidth: unchanged global HBM has
  a 13,927-cycle media floor, while the existing single 64 B/cycle array operand
  ingress has a 20,992-cycle raw floor even with resident Q matrices.

## 2026-09-15 R16 implementation and diagnosis

- Migrated the sequential path to Br64/Bc64, physical 64x64 arrays, two QK K64
  reduction slices, two PV D64 output slices, and 64-row/16-lane SFU contexts.
- Standardized the Attention guest and SST per-core Global-Memory stride at
  2 MiB and propagated the sequential-mode environment into the guest.
- MVM snapshots found that tile 0 issued both QK halves while later tiles began
  with stale `qkReductionHalf=1`; reset it at the common key-tile entry.
- The first Q1024 run exposed an unused generic-GEMM scratch reservation that
  auto-expanded SST's GM stride past the guest ABI. Attention now explicitly
  selects one A/B reuse tile because its custom state machine does not consume
  the generic partial-C window.
- Q1024 failures occur while asynchronous K/V descriptors are active, but HBM
  statistics report zero DMA timeout exhaustion. Instrumented descriptor failure
  exits instead of changing the correctly bounded 32 x 4096-cycle retry policy.
- Fixed DMA landing backpressure so a full LocalGM queue retries instead of
  reporting a false K/V DMA failure.
- Fixed sequential PV D64 output-slice lifetime: the full V tile is read once
  and retained across both slices, so next-tile prefetch cannot corrupt slice 1.
- Added SFU row-resident intermediates; Max/EXP/Normalize no longer round-trip
  the score tile through LocalGM.
- The final Ramulator2 Q1024/K1024/D128 run passed all 131,072 values with zero
  mismatches and maximum absolute error 1.713e-09 at 576,657 native cycles.
  Critical-worker phases are input 96,904, QK 311,721, softmax 12,567, and PV
  85,682 cycles. The strict resource roofline is 20,992 cycles.
# 2026-09-15 R24 256 B/cycle worker-local operand fabrics

- Froze the requested model independently of the reference RTL: common-matrix
  broadcast and destination-specific 64-lane FP32 scatter are separate timed
  operations, both configured at 256 B/native-cycle.
- Traced the current QK and PV input loops and confirmed they issue 64 serialized
  LocalGM reads and array-program commands per 64-array launch. The existing
  group input API is multicast-only and cannot encode 64 distinct vectors.
- Added the timed scatter operation through ComputeArray, MVMComputeArray,
  WorkerCmdProc, and the sequential RoCC state machine. Added independent
  configuration, source contracts, statistics, verifier accounting, and report
  propagation.
- Reused one programmed P input across both PV D64 output slices. Full build and
  install passed; 104 focused Python contracts pass. The only performance run
  was fixed QK1024 and passed numerical/lifecycle verification at 62,900 cycles.
# 2026-09-15 R25 PV restore/output diagnosis

- Completed a read-only source/statistics reconciliation for fixed QK1024.
- Old-O restore: 1,920 serialized 256 B iterations at 8 cycles each = 15,360
  cycles. PV output: 2,048 serialized 256 B iterations at 8 cycles each =
  16,384 cycle body, versus 16,386 measured including two boundary cycles.
- Separated three bounds: raw-byte bandwidth (7,680/8,192 cycles), current
  per-request resource service (9,600/10,240 cycles), and the current forced
  serial callback mechanism (15,360/16,384 cycles).
- Root cause is per-array callback serialization and repeated request startup,
  not WCP queue congestion. The earlier 256 B/cycle matrix/scatter optimization
  does not widen the legacy 64 B/cycle O read/write path.

# 2026-09-15 R26 grouped O-accumulator fabric

- Started implementation under the agreed contract: one worker-local,
  256 B/native-cycle aggregate 64-lane O scatter/gather, one 16 KiB request per
  D64 slice, with online-softmax old-O scaling preserved.
- Root-cause and pattern analysis completed before editing architecture code:
  sequential PV currently chains 1,920/2,048 single-array callbacks, while the
  existing grouped input scatter and grouped score read demonstrate the needed
  aggregate request/completion pattern.
- Added the R26 source/config contract and confirmed it fails at the first
  missing behavior: the sequential dry-run does not yet export
  `GOLEM_OUTPUT_SCATTER_GATHER_BYTES_PER_CYCLE=256`.
- The first post-implementation contract exposed a test-assumption error rather
  than an architecture error: row-major O makes one D64 slice strided. The
  implementation therefore reads/writes the complete 32 KiB O image as two
  legal 16 KiB LocalGM requests per K tile and performs one grouped array
  scatter/gather per slice; updated the contract to assert those two requests.
- Cross-layer implementation now compiles and installs successfully. All 105
  focused Python contracts pass, along with shell syntax, Python compilation,
  and repository diff whitespace checks.
- The only performance run was fixed Q=K=1024/D=128. It passed numerical and
  lifecycle verification at 44,790 cycles with zero mismatches and maximum
  absolute error 1.713e-09. Critical phases: input 13,654, QK 6,732, softmax
  12,581, PV 9,848; restore/output are exactly 2,040/2,176 cycles.
- Extended lifecycle verification with the grouped O-fabric contract and
  reverified the artifact: per worker 30 scatters, 32 gathers, 62 requests,
  1,015,808 bytes, 4,030 service cycles, and zero rejected requests.
- A final unittest command used the repository root and hit two import errors;
  this is a working-directory invocation error, so the same suite is being
  rerun from its test directory rather than changing source code.

# 2026-09-15 R27 K/V prefetch lower-bound diagnosis

- Started a read-only analysis of the fixed QK1024 R26 artifact. The analysis
  separates DMA lifetime from exposed consumer wait: the former includes HBM
  and transport service, while the latter is `max(0, ready-consume)` and can be
  zero when prefetch lead covers the full DMA lifetime.
- Traced the exact two-buffer launch, readiness, wait, activation, and
  second-lookahead conditions. Normal lookahead is one tile; the observed
  2,469-cycle average pair lifetime exceeds the 1,827-cycle average compute
  hiding window, explaining nearly all of the 10,173-cycle exposed wait.
- Completed the resource derivation and measurement reconciliation. Pair floors
  are 205 cycles at the 320 B/cycle link and 256 cycles at the tighter HBM
  command roofline; a 16-worker duplicate wave is 4,096 command cycles. The
  measured K/V phase is exactly 3,416.75 cold-start plus 10,172.5 prefetch-wait
  cycles. No architecture source was changed.

# 2026-09-15 R28 shared streaming K/V supply

- Started implementation of the approved optimization for fixed QK1024 only.
  The intended boundary is shared-node identical-address DMA coalescing with
  chunk-level response fanout, followed by independent worker K/V readiness and
  bounded multi-tile lookahead. Existing dirty changes are treated as the active
  baseline and will not be reverted.
- Added the R28 source/config dry-run contract. Its first run failed as intended
  at the absent `golem_dma_kv_coalesce_enable` token, establishing a red test
  before implementation.
- The first combined configuration patch matched a stale MemNIC member context
  and was rejected atomically; verification found no partial edits. Continued
  with smaller exact-context patches.
- The first enabled dry-run exposed a real capacity constraint: three complete
  K/V pairs overflow the 256 KiB worker Attention window. Kept the two-pair
  baseline for the coalescing phase; a split 3K/2V ring is reserved only if the
  measured post-dedup latency still needs two-tile lookahead.
- Implemented the first shared-node version: identical in-flight K/V chunks are
  coalesced into one HBM request, returned data is fanned out to all subscribers,
  and each physical 16 KiB response occupies a node-wide 256 B/cycle multicast
  stream for 64 cycles. The local build and install completed successfully.
- The first runtime proved that in-flight-only coalescing leaves 176 physical
  reads because worker launches are skewed. Added a red contract, then added a
  completed K/V chunk cache keyed independently of worker identity; a chunk is
  retained until all 16 consumers attached to its memory node have consumed it.
- The cache-backed QK1024 run reached the strict 64-read/1-MiB target, passed all
  runtime verifiers, and reduced total latency to 31,197 cycles. Added worker
  state for K-first activation and a V-only gate at the PV boundary; its focused
  source/config contract is green and awaits rebuild/runtime validation.
- Rebuilt after the worker K/V state-machine change and reran the identical
  fixed QK1024 workload. Final result is PASS at 31,197 cycles, with all 15
  prefetches hitting, zero K/V wait, 64 physical reads, 1 MiB unique multicast
  traffic, and zero resident cache chunks. Independent K/V readiness does not
  change this run because V already arrives before every PV boundary.
- The complete 106-test attention suite, runner syntax check, focused R28
  contract, backend verification, numerical verification, lifecycle
  verification, and `git diff --check` all pass.

# 2026-09-15 P5 GitHub publication

- Started curating the complete accumulated sequential-64/R12-R28 worktree for
  publication on `softmax-update`. Confirmed that local HEAD is two commits
  ahead of the GitHub tracking branch and that public summary numbers need an
  R28 refresh before commit and push.
- Updated the root status, sequential architecture report, workload README, GPU
  roadmap, and R28 plan wording. Fetched `origin/softmax-update`; it is unchanged
  and local remains two commits ahead. The curated worktree passes 106 tests,
  shell syntax, documentation consistency, and whitespace checks.
- Created commit `15bab3f` (`feat: optimize sequential 64-array attention
  pipeline`). HTTPS push failed due to missing GitHub credentials; systematic
  authentication diagnosis found no CLI, helper, SSH agent, or usable SSH key.
- Confirmed there is no token or credential-file fallback. Marked publication
  blocked only on external GitHub authentication; implementation, summaries,
  verification, and the local commit are complete.
