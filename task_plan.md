# 64-Array Sequential Worker Attention Plan

## Goal

For FP32, non-causal `B=1,H=1,D=128` Attention, replace the Attention-cluster
16-QK + 48-PV ownership split with a worker-local, phase-serial architecture:
each worker first executes QK on all 64 arrays, completes the online-softmax
dependency, and then executes PV on the same 64 arrays. Derive and document the
theoretical lower bound for one Attention operation under the modeled hardware
parameters. MPI is used only to reduce simulator wall time.

## Current Baseline

The verified cluster architecture has 20 cores: four control-only managers and
16 workers. Each worker has 64 physical 64x64 arrays with 64 CUs per array,
two operand banks, and a default 16-QK + 48-PV ownership split. Br16/Bc32 is
the measured default.

| Profile | SST R12 Ramulator2 | RTX 5060 Scope A | Ratio | Gate |
|---|---:|---:|---:|---|
| E3, Q=K=1024 | 142,737 | 97,568 | 1.463x | FAIL |
| E4, Q=K=2048 | 518,096 | 345,984 | 1.497x | FAIL |

All frozen R6 numerical and lifecycle cases pass. Independent review is closed
at zero Critical and zero Important findings. See
`attention_cluster/R1_R6_OPTIMIZATION_RESULTS.md`.

## Completed

- [complete] Local SST/Golem and RISC-V guest build/install workflow.
- [complete] Single-head online FlashAttention numerical and lifecycle baseline.
- [complete] Query-group MPI 1/2/4-rank placement and determinism validation.
- [complete] Direct score/P storage, K/V residency, resident O, bounded tagged
  scheduling, cancellation, backpressure, and interval-union observability.
- [complete] H64 alignment to 64 physical 64x64 arrays and 64 CUs per array.
- [complete] R1-R6 ownership sweep, Bc sweep, O FMA, supply-path profiling,
  complete verification matrix, and independent review.
- [complete] R7-A through R7-C controlled scheduler and K/V buffer experiments;
  rejected changes were fully reverted and the R6 baseline was reproduced.
- [complete] Confirmed the two-versus-three K/V buffer A/B on both E3 and E4.
  Three buffers passed correctness at 330,300/997,866 cycles: E3 regressed
  0.05%, while E4 improved 0.97%, so the default remains two buffers.
- [complete] Remove regenerable experiment artifacts and superseded 8+8/H64
  transition documents; retain only current summaries and final verification JSON.
- [complete] R8 manager-level K/V distribution and its feature-off, multi-tile,
  E3, E4, pressure, cancellation-contract, and MPI validation matrix.

## Current Work

- [complete] R28-A: freeze a failing QK1024 contract for global K/V DMA
  coalescing, chunk-streamed fanout, independent K/V readiness, and bounded
  lookahead; trace the Directory/GlobalMemory request and response ownership
  before editing implementation code.
- [complete] R28-B: implement identical-address K/V read coalescing at the shared
  memory-node boundary, retain completed chunks for launch-skewed consumers,
  and fan each returned chunk out to all waiting workers,
  without changing worker-local 64-array QK/SFU/PV ownership.
- [complete] R28-C: allow sequential workers to enter QK when K is complete while
  V continues in flight, retaining the active descriptor in the capacity-safe
  two-pair buffer layout without overwriting live operands.
- [complete] R28-D: build and run focused contracts, then run only fixed
  Q=K=1024/D=128 through numerical, lifecycle, traffic, and cycle verification;
  compare against the 44,790-cycle R26 baseline.

- [complete] R27-A: trace one fixed-QK1024 sequential K/V prefetch from
  release/launch through DMA bursts, striped HBM nodes, landing, readiness, and
  consumer boundary; recover all active bandwidth/credit/timing parameters.
- [complete] R27-B: derive byte, command-issue, DRAM-latency, and overlap-aware
  theoretical cycle bounds for one tile and all 15 prefetched tiles.
- [complete] R27-C: reconcile the measured 37,034 DMA-lifetime cycles and 10,173
  exposed-wait cycles, rank root causes, and document the read-only diagnosis.

- [completed] R26-A: freeze failing source/config contracts for sequential
  64-array O-accumulator scatter/gather at 256 B/native-cycle aggregate, with
  one 16 KiB group request per D64 slice and no per-array callback loop.
- [completed] R26-B: implement grouped O restore/drain through ComputeArray,
  MVMComputeArray, WorkerCmdProc, RoCC, configuration, and statistics while
  preserving online-softmax scaling and output layout.
- [completed] R26-C: build and run focused contracts, then run only fixed
  Q=K=1024/D=128 numerical, lifecycle, and cycle verification; reconcile the
  measured restore/output phases against the 1,950/2,080-cycle modeled bounds.

- [completed] R25-A: trace the fixed-QK1024 sequential PV old-O restore and
  output read/write paths from phase boundaries through WCP, array-buffer, and
  LocalGM; recover exact request sizes and loop counts.
- [completed] R25-B: derive byte-only, port-service, command-serialized, and
  current-state-machine cycle lower bounds for both phases, then reconcile them
  with the measured 15,360 and 16,386 cycles.
- [completed] R25-C: identify the dominant root causes and remaining queue wait,
  document the read-only diagnosis, and name the next optimization target
  without modifying architecture code.

- [complete] R24-A: freeze the architectural contract for two independent
  256 B/cycle worker-local fabrics: common-data matrix broadcast and
  destination-specific 64-lane FP32 vector scatter; add failing source/config
  contracts before implementation.
- [complete] R24-B: add the timed 64-destination scatter primitive through the
  ComputeArray, MVM array, and WorkerCmdProc layers with independent bandwidth
  configuration and statistics.
- [complete] R24-C: replace sequential QK K-vector and PV P-row per-array loops
  with one 16 KiB aggregate scatter request per 64-array launch, preserving
  numerical layout, bank lifetime, and phase accounting.
- [complete] R24-D: build, run focused contracts, then run only fixed
  Q=K=1024/D=128; report theoretical and measured cycle breakdowns.

- [complete] R23-A: trace the reference project's effective configuration
  chain for worker-local GlobalMemory, C-buffer, array operand, GM/NoC, and HBM
  bandwidths; distinguish defaults from active overrides.
- [complete] R23-B: map the Attention matrix multicast onto the closest reference
  SRAM boundary and decide whether 64, 256, or another B/cycle value is the
  aligned assumption.
- [complete] R23-C: update the recommendation and QK1024 cycle interpretation;
  keep this phase read-only unless the user explicitly requests implementation.

- [complete] R22-A: inventory every broadcast/multicast-like path in the
  sequential-64 Attention architecture and record its modeled bandwidth,
  fanout, clock domain, and physical-resource relationship.
- [complete] R22-B: judge whether the 64 B/cycle matrix-broadcast default is
  hardware-plausible and derive the effects and consistency limits of wider
  128/256/512 B/cycle alternatives for fixed QK1024.
- [complete] R22-C: document the read-only recommendation; do not change the
  architecture until a target physical implementation contract is selected.

- [complete] R21-A: establish that the LLM 64 B/cycle matrix-broadcast port is
  configurable and has no bandwidth-modeled counterpart in the reference tree;
  add a failing contract for the sequential PV phase boundaries.
- [complete] R21-B: transition the sequential tile counter from V matrix to
  P input and then to old-O restore without changing dataflow or latency.
- [complete] R21-C: rebuild, run contracts, and rerun only QK1024 to prove phase
  redistribution, numerical/lifecycle correctness, and unchanged total cycles.

- [complete] R20-A: trace QK1024 sequential PV matrix programming from the
  worker state machine through LocalGM, WCP, broadcast/scatter, and array
  operand-buffer completion; define exactly what the phase counter includes.
- [complete] R20-B: derive the per-slice, per-key-tile, and critical-worker
  theoretical cycle lower bounds from actual bytes, ports, fanout, and command
  dependencies.
- [complete] R20-C: reconcile the 40,256 measured cycles with those bounds and
  identify the dominant serialized mechanism without changing implementation.

- [complete] R19-A: confirm the reference project's 320 GB/s layer and add a
  failing contract that the Attention archive exposes a 320 GB/s per-node
  DirectoryController highlink default with environment override.
- [complete] R19-B: implement only that node-bandwidth alignment, rebuild, and
  run the complete focused contract suite.
- [complete] R19-C: run Q=K=1024/D=128 only, verify numerical/lifecycle results,
  recompute the topology-consistent cycle floor, and compare stage latency.

- [complete] R18-A: define K/V supply precisely and reconstruct the QK1024
  HBM -> NoC/DMA -> worker LocalGM -> array timeline from source and statistics.
- [complete] R18-B: compare the 20,992-cycle resource lower-bound assumptions
  against measured bandwidth, request timing, overlap, queueing, and worker
  synchronization without changing the architecture.
- [complete] R18-C: identify and rank the proven sources of the theoretical gap,
  explicitly separating useful service time from exposed wait and overlap loss.

- [complete] R17-A: add a failing contract for sequential QK final-only
  64-array group readout and one logical 16 KiB score-tile write.
- [complete] R17-B: retain the first D64 partial sum in the arrays, then add a
  timed 64-array group read and ordered array-major to row-major tile assembly.
- [complete] R17-C: issue the assembled score tile as one logical LocalGM write,
  with latency still derived from configured port bandwidth.
- [complete] R17-D: build, run focused contracts, then run only
  Q=K=1024/D=128 and report correctness plus stage-cycle changes.

- [complete] R16-A: migrate the sequential worker ABI, Local-GM window, and
  SFU online-state contract from Br16 to Br64.
- [complete] R16-B: implement reference-consistent 64x64 QK D64 reduction-slice
  accumulation and PV D64 output-slice mapping with all 64 arrays active.
- [complete] R16-C: add balanced 16-lane SFU configuration, row-resident flow,
  corrected output gathering/order, and cycle observability.
- [complete] R16-D: build, run contracts, verify a focused numerical case, then
  run Q=K=1024/D128 and report native-cycle stage breakdowns.

- [complete] R15-A: audit the reference 64x64x64-array contract against the
  sequential Attention data path without modifying implementation code.
- [complete] R15-B: enumerate every Br16, D16, wide-D input, operand-bank,
  Local-GM, SFU, scheduler, and verifier dependency that must change for a
  correct 64x64 worker-local QK-then-PV design.
- [complete] R15-C: produce a reviewed migration boundary and corrected cycle
  model before authorizing implementation.

- [complete] R14-A: trace the Q=K=1024 sequential-64 path from striped HBM
  through DMA, Local-GM, QK, online softmax, PV, and final O writeback.
- [complete] R14-B: derive model-native cycle floors for each stage and the
  dependency-constrained critical worker, separating strict bounds from
  Ramulator2/NoC latency that requires simulation.

- [complete] R13-A: recover the original non-cluster worker QK/PV dataflow and
  identify every configuration, scheduler, verifier, and documentation contract
  tied to the 16-QK + 48-PV Attention-cluster split.
- [complete] R13-B: implement 64-array phase reuse so one worker uses arrays 0-63
  for QK and, only after the required dependency boundary, arrays 0-63 for PV.
- [complete] R13-C: add explicit utilization/ordering contracts and update runner
  defaults so the sequential 64-array architecture is the supported path.
- [complete] R13-D: build and run focused correctness/lifecycle tests and at least
  one end-to-end Attention case.
- [complete] R13-E: derive the compute and dependency lower bounds from repository
  parameters, and explicitly separate them from excluded data movement and
  end-to-end latency.

- [complete] R12-A: replace the fused-Attention archive DRAMSim3 backend
  with the selectable control architecture and require Ramulator2 HBM2E-2500.
- [complete] R12-B: add a runtime/configuration contract proving the instantiated
  memory backend matches `GOLEM_MEMORY_BACKEND`.
- [complete] R12-C: run focused contracts and the Ramulator2 HBM microbenchmark,
  then verify Q256/K128 numerical and lifecycle behavior.
- [complete] R12-D: rerun E3/E4 under Ramulator2 and update the performance/GPU
  comparison without mixing DRAMSim3 and Ramulator2 results.

- [complete] R11-A: audit and complete tagged two-bank QK/score/PV operand
  lifetimes, including generation-safe cancellation and exact lease release.
- [complete] R11-B: overlap next grouped-query/tile QK matrix and input
  programming with current SFU/PV work, with an independent feature-off path.
- [complete] R11-C: overlap next physical-tile PV V-matrix programming across
  group boundaries without weakening current four-query V retention.
- [complete] R11-D: convert cluster P input and PV launch to a bounded row
  wavefront, preserving row order, backpressure, and resident-O accounting.
- [complete] R11-E: run focused contracts, build/install, feature A/B, E3/E4,
  pressure, MPI, lifecycle, and GPU comparison; retain only correct improvements.

- [complete] R10-A: trace manager K/V request timing, distributor slot
  ownership, PV matrix identity, P-input lifetime, and group-boundary release.
- [complete] R10-B: implement a bounded early manager K/V request path behind an
  independent feature flag and verify request/delivery/cancellation conservation.
- [complete] R10-C: implement only proven cross-tile PV matrix or P-input
  retention/overlap opportunities behind independent feature flags.
- [complete] R10-D: build and run feature-off, Q256/K1024, E3, E4, pressure, and
  MPI A/B; retain only numerically correct end-to-end improvements.

- [complete] R9-A: trace PV output readout, P/V movement, resident-O
  accumulation, bank ownership, and completion dependencies using R8 metrics.
- [complete] R9-B: implement a bounded timed PV-to-O accumulation path that
  removes redundant cross-component staging without weakening conservation.
- [complete] R9-C: add feature controls, statistics, cancellation/lifecycle
  contracts, and feature-off A/B coverage.
- [complete] R9-D: build and run Q256/K128, Q256/K1024, E3, E4, pressure, and MPI;
  accept only a numerically correct measured improvement.

- [complete] R8-A: extend the existing Manager/Worker GroupCtrl transport
  with a bounded, tagged Attention K/V distribution protocol while preserving
  the legacy mailbox scheduler protocol.
- [complete] R8-B: route worker K/V tile requests through the distributor with
  two resident manager slots, per-K/V readiness, four-consumer completion, and
  generation-safe cancellation/fallback.
- [complete] R8-C: wire feature flags, 64 B/cycle per-destination delivery,
  statistics, runner controls, and contract coverage.
- [complete] R8-D: build and run Q256/K128, Q256/K1024, E3, E4, pressure,
  cancellation, and MPI gates; accept only a correct measured improvement.
- [complete] R7-D: add explicit score/SFU/PV operand-bank consumer lifetime,
  generation, cancellation, and release accounting before attempting any
  further QK multi-producer overlap.
- [pending] Re-run Q256/K128, Q256/K1024, E3, E4, pressure, cancellation, MPI,
  review, and GPU comparison gates after the next accepted architecture change.
- [pending] Pass E3 `<97,568` and E4 `<345,984` cycles before multi-head work.

## Repository Publication

- [complete] P1. Remove regenerable Attention run directories and obsolete docs.
- [complete] P2. Rewrite project status around the current 16-QK + 48-PV R6
  baseline and R7-D next step.
- [complete] P3. Verify retained hashes, documentation, tests, build, and size.
- [complete] P4. Commit the curated cleanup and status update for publication on
  the `softmax-update` GitHub branch.
- [blocked] P5. Refresh the public architecture/results summary through R28,
  verify the complete cumulative worktree, commit it on `softmax-update`, and
  push the branch to `origin`.

## Rejected R7 Experiments

- Moving the ahead scheduler before SFU issue was a measured no-op.
- A partial score-bank retain/release change failed E3 lifecycle and was reverted.
- Bank-aware multi-producer QK caused 9,404 E3 mismatches and was reverted.
- Three K/V buffers passed correctness at E3/E4. E3 regressed from 330,125 to
  330,300 cycles and K/V wait increased from 105,272 to 105,443 cycles. E4
  improved from 1,007,628 to 997,866 cycles and K/V wait fell from 317,331 to
  271,017 cycles. The mixed, sub-1% end-to-end result does not justify changing
  the unconditional two-buffer default.

## Next Step

Authenticate this environment to GitHub, then run
`git push origin softmax-update`. The verified publication commit is complete;
only remote authentication is missing.

## Errors Encountered
| 2026-09-15 | R28 dry-run rejected three full K/V buffers because Q/O + score + K/V require about 272 KiB in a 256 KiB window | Keep two pair buffers for coalescing; use a split 3K/2V layout only if post-dedup measurements require it |
| 2026-09-15 | Combined R28 response-fanout patch missed a shifted statistics-member context and applied nothing | Apply response ownership, queue timing, fanout, and statistics as four independent patches |
| 2026-09-15 | First combined R28 config patch found a stale MemNIC member context and applied nothing | Split runner, architecture, and MemNIC changes into exact-context patches |
| 2026-09-15 | Initial R28 shared-stream contract failed on the missing MemNIC coalescing control, as intended | Implement the shared-node subscriber/fanout path and rerun the same focused test |
| 2026-09-15 | Used the cell-wait API with an exec session identifier while polling the R28 build | Poll exec sessions with `write_stdin`; the original build continued and completed successfully |
| 2026-09-15 | Looked for the R28 SST log at the artifact root although the runner stores it under `logs/` | Resolve the emitted artifact path first and inspect `logs/attention-sst_*.log` |
| 2026-09-15 | Referenced the archived architecture from `tests/archive/`, but its actual path is outside `tests/small/` | Locate it with `rg --files` and patch only the resolved file |
| 2026-09-15 | Initial R28 worker search included a nonexistent `golem/processor/processor.*` path | Use the resolved implementation at `golem/rocc/roccAnalog.h` |
| 2026-09-15 | A broad lower-bound search included a nonexistent top-level `docs/` path, causing `rg` to exit 2 after returning useful matches | Restrict searches to existing planning files and the attention test directory |
| 2026-09-15 | `git push origin softmax-update` failed because HTTPS could not obtain a GitHub username | Diagnosed authentication surfaces before retrying: `gh` is absent, no credential helper is configured, no SSH agent is available, and GitHub rejects the available SSH identity |
| 2026-09-15 | Final R26 unittest invocation ran from the repository root and could not import test-directory modules | Re-run the identical 105-test suite from `tests/small/muticore_attention`, where its local imports resolve |
| 2026-09-15 | R26 grouped O-fabric contract failed on the missing 256 B/cycle configuration, as intended | Proceed with the cross-layer grouped scatter/gather implementation, then rerun the same focused contract |
| 2026-09-15 | Initial R26 contract assumed a D64 O slice was one contiguous 16 KiB LocalGM region | Preserve row-major O and final-DMA compatibility by caching the full 32 KiB O through two 16 KiB requests; keep array scatter/gather grouped per slice |
| 2026-09-15 | First R24 source contract failed on the old 64 B/cycle default, as intended | Implemented independent 256 B/cycle broadcast/scatter configuration and reran the contract to PASS |
| 2026-09-15 | First R24 lifecycle verification expected a minimum broadcast fanout of one after P-row traffic moved to scatter | Derive minimum fanout only from remaining matrix broadcasts; QK1024 then passed lifecycle verification |
| 2026-09-15 | A combined P-input-reuse patch included verifier context under the wrong file target and applied nothing | Split the patch by exact file context, then rebuilt and reran fixed QK1024 |
| 2026-09-15 | Tried to diff reference HDL against the same path in the Attention fork, but the fork has no `architecture/HDL` directory | Treat the reference RTL as the physical width anchor and explicitly state that the Attention broadcast tree currently exists only in the SST model |
| 2026-09-15 | Initial R22 inventory classified fanout-one sequential PV P-row group writes as ordinary array-buffer-bandwidth transfers | Traced `programGemmInputGroupBankAsync` through `MVMComputeArray`; it always uses `enqueueMatrixBroadcastTransfer`, and 2,080 PV requests/worker confirm 32 matrix plus 2,048 P-row requests |
| 2026-09-15 | Final evidence search used two stale guessed runner/builder paths | Enumerated repository files first, then verified the actual `run_fused_attention_scale.sh` and shared `tests/architecture/cpu_builder.py` paths |
| 2026-09-15 | First phase-boundary patch used stale worker-phase spelling and omitted the immediate call visible in the current source | Re-read both exact transition sites and applied the two counter-only transitions against the current implementation |
| 2026-09-15 | Initial reference bandwidth search included a nonexistent top-level `configs/` path and interleaved parallel output led to a premature attribution | Re-ran against the exact reference paths and diffed the two `computeArray.h` files; confirmed broadcast timing exists only in the LLM fork |
| 2026-09-15 | First R20 planning update failed to match the findings-file header context | Re-read the exact file prefix and apply a narrower header replacement; no files were changed by the failed patch |
| 2026-09-15 | The first 320 GB/s contract passed the architecture default but failed runner propagation | The public runner sources a 512 GB/s reference-network default first; make the Attention scale runner explicitly resolve and forward its own 320 GB/s default |
| 2026-09-15 | First R17 QK1024 run passed numerical verification but failed lifecycle on 16 classified score reads per worker | Add an explicit sequential final-tile readout conservation formula; preserve the zero gate for every unclaimed cluster statistic |
| 2026-09-15 | R17 source contract matched a traffic-class comparison across a fixed line layout | Keep the contract semantic by checking the class and group-read helper independently of C++ formatting |
| 2026-09-15 | R17 source contract expected the traffic enum inside the tile function although it lives in its group-read wrapper | Check the wrapper call in the tile function; the enum-to-array validation is already checked separately |
| 2026-09-14 | R11 resolved config claimed Ramulator2 while runtime used DRAMSim3 | The selected archive architecture hard-codes `memHierarchy.dramsim3`; migrate the runner and add an instantiated-backend contract |
| 2026-09-14 | First verifier patch used an imprecise function-return context | Re-read the exact helper and manager-stat blocks, then applied a narrower patch |
| 2026-09-14 | First Ramulator2 E4 run was numerically correct but lifecycle reported 4 stalls versus 2 | `slot_stalls` counts queued worker requests, not missed tile lookaheads; add a regression and validate the four-worker upper bound |
| 2026-09-14 | Final shell audit compared an empty zero-match `rg -c` result as an integer | Use boolean `rg -q` assertions for required and forbidden backend markers |
| 2026-09-14 | QK lookahead initially launched zero tiles because it waited for pair-ready | Add worker-local K-ready signaling and overlap QK programming with V delivery |
| 2026-09-14 | E4 verifier assumed every speculative manager lookahead must hit | Preserve exact formal traffic counts and check missed speculation against slot stalls |
| 2026-09-14 | WCP pressure run requested unsupported queue depth 2 | Use the cluster contract's minimum bounded depth of 16 |
| 2026-09-14 | New independent-switch test omitted `--attention-cluster` | Enable the required architecture in the test fixture |
| 2026-09-14 | Initial P-row wavefront lifecycle expected six batch launches and PV concurrency 32 | Root cause was a stale verifier assumption; express 36 row launches and bounded variable concurrency for the explicit A/B path |
| 2026-09-14 | Initial R11 planning patch used an imprecise context anchor | Re-read the planning files with line numbers and applied a narrower patch |
| 2026-09-14 | Fused accumulator contract initially expected 16 O read-busy cycles | Final context drain also consumes 128 segment reads; corrected the expected total to 144 |
| 2026-09-14 | Tried to run `attention_cluster/test_attention_cluster_contract` directly | The contract test is a source-only target; rebuild it under `/tmp` before execution |
| 2026-09-14 | Contract test compile used `-Isrc/sst/elements` | Include is rooted at `src`; rebuild with `-Isrc` |
| 2026-09-14 | Combined planning update/build call could not match old checklist syntax | Read the current plan, patched its actual `[status]` syntax, then start the build separately |
| 2026-09-14 | E3/E4 row-fusion lifecycle rejected zero per-worker SFU/PV overlap | Row fusion removes the old serialized O-submit tail that created this overlap; retain QK/PV overlap plus interval/O conservation gates, and require SFU/PV only on the legacy path |
| 2026-09-14 | R9 results document patch had one unprefixed added line | Correct the patch syntax and reapply; no file changes were made by the failed patch |
| 2026-09-14 | R10 P-overlap Q256/K1024 first run failed lifecycle despite zero numerical mismatches | Cluster rows were verified with the generic PV-array count; add a regression test and use `jobs * 16` for cluster P rows |
| 2026-09-14 | R10 runner contract failed after lookahead defaulting and a new test raised `NameError: ROOT` | Make non-explicit lookahead follow distribution and use the existing repository path expression |
| 2026-09-14 | Broad `/tmp` artifact search reported permission errors in unrelated private directories | Restrict subsequent reads to the known R9 artifact directory |
| 2026-09-14 | Findings current-performance patch did not match its broad context | Re-read the exact paragraph and apply a smaller scoped replacement |

| Error | Attempt | Resolution |
|---|---:|---|
| `jq` rejected `$label` in the comparison filter | 1 | Rename the argument to a non-keyword variable and rerun extraction; simulations and artifacts are unaffected. |
| `make -C src/sst/elements/golem groupctrl/groupctrl.lo` has no such target | 1 | Use the repository's standard build/install script after integration instead of retrying an unsupported partial target. |
| Q256/K128 model construction required missing scheduler endpoints | 1 | The runner had selected the legacy control architecture; investigate its compatibility before rerunning. |
| Q256/K128 completed with zero Attention and HBM activity | 2 | The legacy control architecture omitted the four fused-Attention guest arguments. Wire GroupCtrl into the proven Attention architecture and keep that architecture selected. |
| Sequential Q256/K256 aborted in `Reg2GlobalMem` | 1 | Generic geometry expanded SST GM stride beyond the prebuilt guest's fixed 1 MiB ABI; standardize both on a 2 MiB per-core stride. |
| Focused K/V distribution contract raised `NameError` | 1 | Reuse the existing `ARCHIVE_ARCH` path constant in the new architecture assertions. |
| Direct scale runner rejected `--mpi-ranks` | 1 | Set its documented `GOLEM_MPI_RANKS=2` environment variable; the CLI flag belongs to the outer wrapper. |
