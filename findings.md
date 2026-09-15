# Findings

## R23 Reference On-Chip SRAM Bandwidth Re-audit

- The reference `configs/default.env` is an active aggregator loaded by
  `run_noc_dma_pipeline.sh`; it sources `20_dma.env` and `30_network.env`, and
  their `:=` assignments preserve caller overrides while supplying real default
  values. The 256 values are not dead examples.
- Reference defaults include a 1 MiB worker-private partial-C SRAM with
  independent 1R1W ports, each at 256 B/cycle and one-cycle latency. The source
  comment defines 256 B as one 64-array accumulator row of 64 int32 values per
  cycle.
- Reference defaults also set the GlobalMemory NetworkIF link to 256 GB/s,
  alongside 512 GB/s NoC links, 1024 GB/s router crossbar, 512 GB/s memory-side
  attachment, and a 320 GB/s physical HBM-stack peak. This establishes a
  deliberate 256-byte/cycle-class worker/GM injection boundary at a 1 GHz data
  plane.
- The reference GlobalMemory implementation does not time ordinary local
  `rd_from_globalmem`/`wr_to_globalmem` accesses with a bytes/cycle port. The
  `local_access_bytes_per_cycle` queue and port model exists only in the LLM
  Attention fork. Therefore the reference supports a 256-class architectural
  intent, but does not directly measure every local SRAM movement at 256.
- The reference synthesis RTL defines an array matrix row as
  `ARRAY_M * ELEM_W = 64 * 32 = 2048 bits = 256 B` and exposes that complete row
  on `array_load_row_i` in one cycle while selecting `array_load_array_i`. The
  RTL does not implement a 64-destination broadcast tree, but it provides a
  much closer physical width anchor for the Attention tree root than the generic
  64 B/cycle SST buffer default.
- Mapping the Attention common-matrix broadcast to this row interface makes
  256 B/cycle the natural reference-aligned payload rate: one 64-FP32 matrix row
  per cycle, 64 payload cycles for a 16 KiB 64x64 matrix, plus setup/tree delay.
- The RTL GM has eight banks and a 128 B line. A 256 B/cycle worker-local path
  therefore corresponds to two GM lines per cycle feeding one 256 B matrix row;
  this is internally consistent with the reference's 256 GB/s GM link and
  256 B/cycle C-buffer ports at a 1 GHz data plane.
- The Attention fork does not contain the reference `architecture/HDL` tree, so
  its 64 B/cycle broadcast extension has no separate updated RTL. The physical
  comparison must use the reference RTL row/GM widths plus the fork's SST
  multicast semantics, not claim an implemented Attention broadcast RTL.
- Reference array storage is asymmetric. Matrix and output banks are 2048-bit
  wide SRAM words, so their load/seed interface carries 256 B/cycle. Each input
  vector bank is instead 32 bits wide at depth 64, and the load interface writes
  one scalar to one selected array/address per cycle: 4 B/cycle/array.
- The Attention SST extension incorrectly couples matrix multicast and group
  input multicast through the single `matrixBroadcastBytesPerCycle` parameter.
  Raising it to 256 is reference-aligned for Q/V matrix rows, but also turns a
  256 B PV P-vector request into one payload cycle, which is not supported by
  the reference RTL scalar vector-load interface.
- Consequently the measured 85,305-cycle 256 B/cycle run is a valid hypothetical
  wide-matrix plus wide-group-input architecture, not a clean reference-aligned
  matrix-only result. A faithful model should split matrix-row broadcast rate
  from vector-input multicast/programming rate before changing the default.
- Revised recommendation: use 256 B/cycle for the common Q/V matrix-row
  broadcast when aligning to the reference on-chip SRAM organization. Keep a
  separate vector-program parameter unless the architecture explicitly adds a
  64-lane scatter/wide vector-load path. With such a vector-path redesign, an
  aggregate 256 B/cycle is also plausible as 64 arrays each accepting one FP32
  scalar per cycle, but that is not the current reference RTL controller.
- The 16 KiB common matrix then has 64 payload cycles and
  `1 + 64 + 6 = 71` total broadcast cycles. The existing coupled 256 experiment
  measures 85,305 cycles. Holding P-vector timing at the 64 B/cycle behavior
  while widening only Q/V matrices would analytically land near 91.5k cycles;
  an exact value requires the rate split and a fresh QK1024 simulation.

## R22 Broadcast-Port Plausibility Audit

- The sequential-64 path has a true modeled matrix multicast operation: one
  matrix payload enters `ComputeArray`, is functionally installed in 64 array
  banks, and is charged once at
  `base + ceil(payload/matrixBroadcastBytesPerCycle) + log2(fanout)*stage`.
  It records ingress bytes once and sink bytes multiplied by fanout.
- The matrix-bandwidth parameter is logically distinct but does not represent a
  fully independent physical port: matrix broadcasts, single-array input/output
  programming, and non-near-array reads all enter the same bounded
  `arrayBuffer` request machinery and contend for `arrayBufferPorts` (one in the
  QK1024 configuration).
- The QK1024 sequential runner uses 64 B/cycle matrix broadcast and 64 B/cycle
  per array-buffer transfer. It separately raises LocalGM and the dedicated
  grouped near-array output path to 16,384 B/cycle. The dormant PV V-tile reuse
  buffer defaults to 64 B/cycle but is disabled on this path.
- GroupCtrl K/V distribution is not a physical multicast in the model. The
  manager loops over four subscribers and sends four payload-bearing response
  messages; each worker then serially writes K followed by V through that
  worker's LocalGM port. In the active sequential-64 configuration distribution
  is disabled, so all 16 workers independently DMA K/V from memory.
- The fixed QK1024 run sets CPU, RoCC, SFU, LocalGM, and array clocks to 1 GHz.
  Thus 64 B/cycle is 64 GB/s per worker, while 128/256/512 B/cycle would mean
  128/256/512 GB/s per worker and a 1024/2048/4096-bit-wide one-cycle ingress
  equivalent. These rates must not be compared directly to the four-node HBM
  aggregate without accounting for 16 replicated worker fabrics.
- At 64 B/cycle, the setting is a conservative, physically plausible shared
  worker operand-fabric baseline. A larger rate is legitimate only as an
  explicit wider/banked fabric contract; changing the number alone otherwise
  creates bandwidth unsupported by the modeled port count and storage banks.
- `programInputGroupBankAsync` reuses `enqueueMatrixBroadcastTransfer`, so both
  genuine input multicast and fanout-one group input use the matrix-broadcast
  rate. Sequential PV P rows use this API even though each D64 wave targets one
  array per row. The measured 2,080 PV requests/worker close as 32 V-matrix plus
  2,048 P-row requests. Raising matrix-broadcast bandwidth therefore reduces PV
  P-input time as well as V-matrix time. Sequential QK K rows use ordinary
  single-array input programming and remain on `arrayBufferBytesPerCycle`.
- A 16 KiB common matrix sent to 64 arrays produces 1 MiB of aggregate sink
  writes. At the current 263-cycle modeled service, that is about 3,987 B/cycle
  across all leaves, or roughly 62.3 B/cycle into every array bank. The model
  does not separately constrain those 64 destination SRAM write ports; wider
  root rates implicitly assume proportionally wider per-array sinks as well.
- Controlled QK1024 parameter runs passed numerical and lifecycle verification:
  64/128/256 B/cycle produce 103,757/91,485/85,305 end-to-end cycles. Relative
  to 64, 128 saves 12,272 cycles (11.83%) and 256 saves 18,452 (17.78%); doubling
  from 128 to 256 saves only another 6,180 cycles.
- At 128 B/cycle, critical QK/PV are 22,992/50,625 cycles; at 256 they are
  20,944/46,528. The invariant costs include 16,384 QK input, 15,360 old-O
  restore, 2,176 PV compute, and about 16,384 PV output cycles. PV P-input falls
  from 16,384 to 12,288 at 128 and reaches 10,240 at 256, after which its
  LocalGM/base/WCP overhead prevents further bandwidth benefit.
- The defensible default for the currently modeled one-port, 64 B/cycle ordinary
  array-buffer interface is 64 B/cycle matrix ingress. A 128 B/cycle design is
  plausible as an explicit two-lane/dual-bank 1024-bit multicast fabric. A
  256 B/cycle default is aggressive: it represents a 2048-bit root and 64
  simultaneous destination-bank writes per worker, and should require a stated
  SRAM banking/wiring/energy contract rather than a parameter-only speedup.
- The implementation applies the 64 destination `matrixData` copies
  functionally when the modeled transfer completes; it does not model per-cell
  analog programming pulses or destination SRAM bank timing. Thus 64 B/cycle is
  plausible if this boundary is a staged SRAM/operand-buffer multicast. If it
  represents direct RRAM/PCM conductance programming, even 64 B/cycle may be
  optimistic and no larger setting is defensible without a device model.

## R21 Broadcast Configuration and Corrected PV Boundaries

- The LLM Attention broadcast ingress is configured by
  `GOLEM_MATRIX_BROADCAST_BYTES_PER_CYCLE` and defaults to 64 B/cycle. For
  payload `B`, fanout `F`, one setup cycle, and one cycle per binary-tree stage,
  its modeled service is `1 + ceil(B/rate) + ceil(log2(F))` cycles.
- The reference project has no corresponding bandwidth setting, timed matrix
  broadcast queue, or matrix-program cycle statistic. Its direct
  `setMatrixItem` loop is a functional state update, so the only accurate
  reference-project answer is "not modeled", not 64 B/cycle or another inferred
  physical rate.
- Sequential PV now transitions its tile-pipeline counter from V-matrix
  programming to P-input programming, then to old-O restoration for key tiles
  1..15. Commands and dependencies are unchanged.
- The rebuilt QK1024 run passes numerical and lifecycle verification at the
  unchanged 103,757 end-to-end cycles. The corrected critical-worker PV counts
  are matrix 8,512, input 16,384, restore 15,360, compute 2,176, and output
  read/write 16,385 cycles; their sum remains 58,817 cycles.

## R20 PV Matrix-Programming Root Cause

- In sequential-64 PV, one D64 output slice constructs the transposed
  `Vpanel^T[64,64]` matrix and broadcasts its 16 KiB payload to all 64 arrays.
  D128 has two slices per key tile and QK1024 has 16 key tiles, so each worker
  issues exactly 32 PV V-matrix broadcasts.
- The measured `attention_worker_tile_pv_matrix_program_ticks=40,256,000`
  ticks is not a pure V-matrix counter. At the start of PV the tile-pipeline
  phase changes to `PvMatrixProgram`, but the sequential functions later change
  only the logical worker phase to `PvProgramInputs`; they never call
  `transitionAttentionTilePipeline(PvInputProgram)` or
  `transitionAttentionTilePipeline(PvRestoreOutput)`.
- Consequently the 40,256-cycle bucket contains: one full 32 KiB V LocalGM read
  per key tile; 32 V-matrix broadcasts; 2,048 P-row LocalGM-read plus array-input
  operations; and, for key tiles 1..15, 1,920 old-O-row LocalGM-read plus array
  output-restore operations. The separate measured PV input and restore buckets
  are both zero, confirming this attribution leak.
- Per-tile samples are exact and deterministic: the first key tile costs 1,556
  cycles, while each of the 15 later tiles costs 2,580 cycles. The 1,024-cycle
  difference is old-O restore work, not additional V-matrix programming:
  `1,556 + 15*2,580 = 40,256`.
- The configured broadcast fabric accepts 64 destinations, has 64 B/cycle
  ingress, one setup cycle, and six binary-tree stages. A 16 KiB V matrix thus
  occupies `1 + 16,384/64 + 6 = 263` component cycles. Pure matrix-broadcast
  occupancy is only `32*263 = 8,416` cycles per worker before WCP boundary
  overhead, far below the mislabeled 40,256-cycle bucket.
- The entire measured bucket closes exactly under the instantiated timing:
  V LocalGM reads are `16 tiles * 2 chunks * 2 cycles = 64`; each V broadcast
  adds one WCP issue cycle, so matrix transfers are `32*(1+263)=8,448`; each of
  2,048 P rows serializes a two-cycle 256 B LocalGM read with one WCP issue plus
  a five-cycle single-array transfer, giving `2,048*8=16,384`; and 1,920 old-O
  rows use the same `2+1+5=8` cycle chain, giving 15,360. Therefore
  `64+8,448+16,384+15,360=40,256` with no unexplained queue time.
- The pure payload lower bound for PV V-matrix ingress is
  `16 tiles * 2 slices * 16 KiB / 64 B/cycle = 8,192 cycles`. Including the
  configured one-cycle broadcast setup and six tree stages gives 8,416 cycles.
  Including the current unavoidable one-cycle WCP boundary and the sequential
  32 KiB LocalGM V read gives a current-interface dependency floor of 8,512
  cycles. The 64-way fanout is not multiplied into payload service; the fabric
  already models one multicast transaction and records 64 destination sinks.
- There is no evidence that queueing inflates this path: the array buffer has
  zero rejects and high-water mark one; WCP has zero queue-full stalls and a
  one-cycle maximum wait. The 31,744-cycle difference between 40,256 and 8,512
  is entirely P input plus old-O restore, not matrix traffic or contention.
- `GOLEM_MATRIX_BROADCAST_BYTES_PER_CYCLE` is an LLM-fork extension and is
  configurable; the scale runner and CPU builder default it to 64 B/cycle.
  The reference project has no `matrixBroadcastBytesPerCycle`, modeled
  broadcast queue, or corresponding statistic. Its WCP directly calls
  `setMatrixItem` for each functional matrix element without assigning those
  writes a separate cycle cost. The reference therefore has no numeric
  broadcast-port bandwidth to copy; calling it 64 B/cycle would be incorrect.

## R19 Reference HBM Node Bandwidth Alignment

- The reference project's Ramulator model explicitly describes one eight-channel
  HBM2E stack as 320 GB/s: 8 channels * 32 B/DRAM-cycle * 1.25 GHz.
- Its current default `ncores_selfcom_dma_ctrl.py` topology reads
  `GOLEM_DIRCTRL_HIGHLINK_BW` and the network profile defaults that link to
  512 GB/s to leave protocol headroom above the 320 GB/s HBM stack.
- Both the reference archive and the Attention archive retain a stale hard-coded
  25 GB/s DirectoryController MemNIC. Thus 320 GB/s is the reference HBM-node
  payload rate, while 512 GB/s is the reference's non-limiting NoC attachment.
- Per the requested Attention contract, the archive will use 320 GB/s/node as
  its default DirectoryController highlink and accept an environment override.
  This makes the node attachment match the HBM stack rather than artificially
  limiting it to 25 GB/s.
- The Attention scale runner must explicitly forward its own 320 GB/s default.
  It sources the shared network profile, whose 512 GB/s default would otherwise
  override the archive fallback and make recorded intent differ from the
  requested experiment.
- With four 320 GB/s endpoints, 16.5 MiB of read responses have a 13,517-cycle
  endpoint floor; all 17 MiB need 13,927 cycles if directions share capacity.
  The 1,024 B/cycle HBM command roofline gives a 16,896-cycle reads-only floor.
  All are below the unchanged 20,992-cycle array-operand ingress floor, so the
  recomputed strict system roofline is still 20,992 native cycles.
- The QK1024 run passes numerical and lifecycle verification at 103,757 cycles,
  with zero mismatches and maximum absolute error 1.713e-09. This is 75.60%
  lower, or 4.098x faster, than the 425,172-cycle 25 GB/s run and is 4.943x the
  strict roofline.
- Critical-worker phase totals are input 3,481, QK 27,088, softmax 12,567, and
  PV 58,817 cycles. All 15 K/V prefetches hit; their exposed wait falls from
  221,935 cycles to zero. Aggregate measured HBM bandwidth rises from 64.91 to
  529.97 GB/s. NoC maximum port utilization is 4.34%, with 215 crossbar stalls
  and zero output-port stalls.

## R18 K/V Supply Versus Theoretical Floor

- K/V supply means a complete `K[64,128]` plus `V[64,128]` pair becoming ready
  in the worker's selected LocalGM ping-pong buffer. It includes HBM service,
  memory-controller/DirectoryController handling, MemNIC response admission,
  NoC return, GlobalMemory DMA completion, and LocalGM landing.
- Each K or V tile is 32 KiB, so one pair is 64 KiB. Distribution is disabled;
  all 16 workers independently fetch the same complete K/V tensor. Actual per
  worker reads are exactly 1,081,344 B = 32 KiB Q + 16 * 64 KiB K/V, and the
  system read total is 17,301,504 B. This matches the intended traffic formula;
  the gap is not caused by accidental extra K/V reads.
- For critical core 16, the first K/V pair costs 35,005 cycles. The next 15
  prefetches take 314,240 cycles in total, or 20,949 cycles/pair. Current-tile
  work hides 92,305 cycles, or 6,154 cycles/pair, leaving exactly 221,935
  exposed cycles, or 14,796 cycles at every tile boundary. There are zero hits
  and 15 waits.
- The existing 20,992-cycle roofline omits the actual memory-node injection
  ports. The selected archived topology hard-codes every DirectoryController
  MemNIC to 25 GB/s even though the resolved environment and CSV record
  `GOLEM_DIRCTRL_HIGHLINK_BW=512GB/s`. Four data nodes therefore provide only
  100 B/native-cycle of payload egress. The 16.5 MiB read-response traffic alone
  therefore needs `ceil(16.5 MiB/100)=173,016` cycles before packet overhead or
  queueing. Summing the opposite-direction 0.5 MiB O writes would require proof
  that the link shares directional serialization, so 173,016 is the strict
  bound used here. The previously stated 20,992-cycle system lower bound is
  incomplete for the instantiated topology.
- The four HBM stacks still have a 1,280 B/cycle physical ideal floor of 13,927
  cycles. The statistics tool separately reports a 1,024 B/cycle tCCD_L command
  roofline, which would give 17,408 cycles for 17 MiB. Both are hidden behind
  the much narrower 100 B/cycle DirectoryController endpoint.
- HBM itself is not saturated: measured aggregate bandwidth is 64.913 GB/s,
  tCCD_L utilization is 6.339%, backend average read latency is 92.5 cycles,
  and P95 is 139 cycles. The global NoC routers are also lightly utilized
  (maximum port utilization 1.595%, 172 xbar stalls, zero output-port stalls).
- Congestion is localized at the memory endpoint response path. Data nodes 1
  and 2 hit all 128 DMA credits, record 207/142 blocked requests, response queue
  high-water marks 32/32, and maximum response holds of 66,321/32,233 cycles.
  Nodes 3 and 4 do not credit-block. This endpoint queuing plus the 25 GB/s
  egress cap explains long DMA RTT despite low HBM and global-NoC utilization.
- Critical core 16 has zero DMA landing backpressure retries, so its 221,935
  K/V wait is not caused by the widened LocalGM landing port. LocalGM has other
  request retries, but those are not on this critical DMA landing boundary.
- A topology-consistent strict resource lower bound is at least
  `max(8,720 compute/dataflow, 13,927 raw HBM, 20,992 array ingress,
  173,016 memory-endpoint read egress) = 173,016 cycles`. The measured 425,172
  cycles are 2.457x this corrected bound, not 20.25x.

## R17 Sequential QK Tile Readout

- The critical worker spends 284,974 of 311,721 QK cycles in the combined
  compute/readout phase; pure QK array compute is only 2,112 cycles.
- The ordinary sequential path reads one array output and then issues 64
  callback-chained 4-byte LocalGM writes before reading the next array.
- It performs this complete 64x64 read/write traversal after both D64 slices,
  even though the first traversal materializes an intermediate partial sum that
  no consumer reads.
- For QK1024 this creates 16 tiles x 2 slices x 64 arrays x 64 lanes = 131,072
  LocalGM writes. At one base cycle plus one transfer cycle per 4-byte request,
  those writes alone impose 262,144 cycles.
- The corrected contract is one final-only 64-array group read, followed by one
  logical 16 KiB row-major score-tile write. Both operations remain timed by
  their configured port bandwidth; "one-time" means one transaction, not
  bandwidth-free movement.
- The array layer already exposes a timed group-output API, but the MVM
  implementation currently validates only two adjacent arrays and only the
  legacy cluster QK/PV traffic classes. Sequential-64 needs an explicit
  64-array final-score traffic class rather than weakening cluster validation.
- LocalGM currently limits one request to 4 KiB, so `attentionLocalWrite` would
  split a 16 KiB score tile into four requests. The sequential-64 performance
  configuration must raise this limit to 16 KiB while retaining the normal
  `base + ceil(bytes / bytes_per_cycle)` latency formula.
- The first R17 QK1024 simulation is numerically exact and reduces critical
  QK from 311,721 to 27,080 cycles. Its only lifecycle mismatches are the old
  non-cluster expectation of zero classified QK score reads; every worker
  reports exactly 16 requests and 262,144 bytes, matching one 16 KiB read per
  key tile.
- After updating that conservation formula, the same QK1024 artifact passes
  numerical, lifecycle, and metrics verification at 425,172 end-to-end cycles.
  QK compute/readout falls from 284,974 to 6,480 cycles; total QK falls from
  311,721 to 27,080 cycles.
- The sufficient-bandwidth contract uses one 16 KiB/cycle group-output port and
  one 16 KiB/cycle LocalGM port. With one base cycle on each, the final score
  group read and tile write have two-cycle service lower bounds apiece.
- The largest remaining critical-worker cost is input movement: 257,004 cycles,
  including 221,935 cycles waiting on K/V prefetch over all 15 inter-tile
  boundaries. Shortening QK removed overlap that previously hid this wait.

## R13 Sequential 64-Array Worker Architecture

- The requested target is `/data/jjgong/LLM/RISC-V-CIM-Manycore-SST`; it already
  contains uncommitted R12 backend/documentation work that must be preserved.
- The current cluster contract statically partitions each worker's 64 physical
  arrays into 16 QK and 48 PV arrays and permits cross-stage overlap. The new
  contract instead requires temporal reuse: QK owns all 64 arrays first, then PV
  owns the same 64 arrays after the dependency boundary.
- The repository also retains a legacy non-cluster Attention path. Its mapping
  and launch granularity must be audited before deciding whether to extend it or
  simplify the current cluster implementation.
- The cluster was introduced by commit `1140dec`; `4d47f83` is the immediately
  preceding implementation baseline and is the best available reference for the
  user's "original idea."
- `AttentionClusterConfig` currently accepts only QK ownership 16/24/32/40 and
  derives PV as the remainder. `AttentionArrayOwnership` mirrors that static
  partition. Setting QK to 64 would therefore be rejected and would leave zero
  PV arrays even if validation were bypassed.
- A faithful 64-array design needs phase ownership (`all QK` -> dependency ->
  `all PV`) and must disable the ahead pipeline's QK/PV array overlap. This is a
  semantic scheduler change, not a configuration-only adjustment.
- With the current fixed `Br=16`, `D=128`, 64x64 arrays, QK maps each query row
  to two D64 arrays, so one logical QK wave contains only 32 useful array ops.
  Merely giving the phase ownership of 64 arrays does not make all 64 active.
  Full simultaneous utilization requires two Br16 query contexts (or changing
  the worker tile to Br32); this is an architectural constraint that the final
  implementation and lower-bound derivation must state explicitly.
- Cluster PV uses the same two-array-per-query-row mapping and iterates output
  dimension panels. Thus its current Br16 wave also activates 32, not 64,
  arrays even though the static PV partition contains 48 arrays.
- The project's own R1-R6 report explicitly says there is no parameter-only
  Br32 mapping and that time-multiplexing all 64 arrays requires reprogramming
  bounded operand banks. The redesign must therefore change mapping/scheduling,
  and claims of "full" use must be verified by 64-way launch counters.
- The legacy path configures 16 arrays of shape `head_dim x 16`; the cluster path
  configures 64 physical 64x64 arrays. Reverting the enable flag alone would
  revert to 16 logical wide arrays, not satisfy the requested physical design.
- Default compute latency for a 64-column launch is `ceil(64/1)+2 = 66` array
  cycles at 2.3 GHz; a 32-active-column PV launch is 34 cycles. These are
  operation latencies, while end-to-end lower bounds must additionally respect
  operand programming/readout bandwidth and QK->softmax->PV serialization.
- A Br16-compatible full-occupancy mapping exists in the original wide-array
  dataflow and is preferable to Br32: use `Bc=64`, map QK key rows 0-63 to
  arrays 0-63 in one wave, and map PV `(D16 panel within a four-panel wave,
  query row)` to `array = panel_in_wave * 16 + row`. D128 PV then needs two
  64-array waves. This preserves the manager/worker query partition.
- The sequential mapping uses 64 arrays shaped `128x16` for D128, rather than
  the cluster's `64x64` shape. Each QK wave performs 64 independent MVMs; each
  PV wave performs 4 panels x 16 rows = 64 independent MVMs.
- Existing QK row-burst storage is sized around a 16-key panel and cannot
  directly hold a 16x64 score tile. The new path should use the ordinary or
  bounded pipelined readout and disable row-burst mode unless its C-buffer is
  deliberately enlarged.
- First end-to-end R13 run failed deterministically in `QkProgramMatrix` before
  any GEMM command. The array recorded exactly one broadcast rejection: the new
  QK wave requested fanout 64 while the runner still configured maximum fanout
  16. This is a configuration propagation defect, not a numerical defect.

## R12 Memory Backend Migration

- R11 `run_config.env` records `GOLEM_MEMORY_BACKEND=ramulator2` and
  `hbm2e_2500.yaml`, but the selected
  `architecture/archive/ncores_selfcom_dma.py` unconditionally instantiates
  `memHierarchy.dramsim3` with `HBM_4Gb_x128.ini`.
- The R11 runtime log confirms the actual backend through
  `DRAMSIM3_BACKEND_READ_LATENCY_GLOBAL`; the published R11 cycles must remain
  labeled as DRAMSim3 until the full matrix is rerun.
- `architecture/ncores_selfcom_dma_ctrl.py` implements the required
  `dramsim3`/`ramulator2` selection and passes a 32-byte request width plus the
  configured HBM2E-2500 YAML to `memHierarchy.ramulator2`.
- R12 Ramulator2 passes at 15,565 cycles for Q256/K128, 142,737 for E3, and
  518,096 for E4 with zero numerical mismatches and all lifecycle/backend gates.
- Relative to the historical DRAMSim3 R11 values, E3/E4 increase 11.87%/20.45%.
  Critical-worker K/V wait is 10,283/29,004 cycles and E4's maximum boundary
  interval is 60,243 cycles under the corrected memory timing.
- Ramulator2 manager slot stalls count queued worker requests. A missed
  speculative tile can contribute up to four stalls; treating stalls as a
  one-to-one count of missed tile lookaheads was an invalid verifier assumption.

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
## R13 Verified Result

- The original wide-MVM layout can use all arrays without changing Br16: Bc64
  assigns one key to each QK array, while PV maps `(D16 panel, query row)` onto
  64 arrays and needs two waves for D128.
- Sixty-four individual launch commands overflow the bounded WCP depth of 32;
  one group-launch command preserves the physical concurrency and callback
  conservation.
- Sequential PV probability multicast requires a distinct four-array topology;
  reusing the cluster two-array `PvPInput` class is correctly rejected.
- The first numerically complete run reproduced the first tile's unnormalized
  partial O. Root cause was omitted `oldOutputScale[row]` during restored-O
  programming; applying it before accumulation produced zero mismatches.
- Verified Q256/K128/D128 statistics per worker: two QK waves and four PV waves,
  with active-array min/max equal to 64 for both phases.
- The canonical metric for R13 and later is the model-native accelerator cycle.
  The active preset runs RoCC/SFU/array at 1.0 GHz, so its 1 GHz normalization
  already produces native cycles; wall-time units are excluded from new comparisons.
- Under the modeled array timing, D128 Br16/Bc64 has QK 130 cycles, PV 132 cycles,
  and array-only floor 262 cycles/tile. The 16-row/64-column Row Engine floor is
  277 cycles, giving a strictly serial compute floor of 539 cycles/tile.
# R14: Q=K=1024 HBM-to-attention trace (2026-09-15)

- Assumption requested by the current example: `Q=K=1024`, `D=128`, FP32,
  `Br=16`, `Bc=64`, 16 workers, 64 arrays/worker.
- Tensor sizes: Q/K/V/O are each 512 KiB. Each worker owns 64 query rows, so
  its Q and O slices are 32 KiB. One K or V key tile is 32 KiB.
- With K/V query-group reuse of four query blocks, every worker fetches each of
  the 16 K tiles and 16 V tiles once: 1 MiB K+V per worker. Including Q read and
  O write, HBM traffic is 1.0625 MiB/worker and 17 MiB globally.
- Placement is balanced across four data nodes: each manager's 256 Q/O rows
  live on its node, while K/V use four 256-row bands (four Bc64 tiles/node).
  Each node therefore serves 4.25 MiB for the complete operation.
- Current native architectural cycle is the 1.0 GHz RoCC/array/SFU cycle. The
  HBM controller runs at 1.25 GHz and must be converted to this cycle basis.
- The strict compute-only bound per logical `(Br=16,Bc=64)` tile at D=128 is
  QK 130 + online softmax 277 + PV 132 = 539 cycles. A critical worker has
  4 query blocks x 16 key tiles = 64 logical tiles, hence 34,496 cycles.
- QK's implementation path includes Q local read + 64-array broadcast, 64
  serialized K-vector local reads/programs, 130-cycle compute, and 64 output
  reads followed by scalar score writes. The simple modeled sum is about 3,724
  cycles/tile, consistent with the Q256/K128 regression (7,448 QK cycles for
  two tiles). For Q1024 this makes the worker QK path about 238,336 cycles
  before HBM startup/queue contention.
- Ramulator HBM2E config: 32-byte data payload, 20 memory-cycle read latency,
  1.25 GHz memory controller; local GM: 64 B/native-cycle, 1-cycle base,
  one read and one write port, max request 4096 B.
- Source confirms direct K and V DMAs are issued together, into the active
  double-buffer pair. The next pair is prefetched while the current tile runs.
  Query DMA occurs once at the start of each 16-row query block; final O DMA
  occurs once after all 16 key tiles for that block.
- Sequential64 QK broadcasts one 16x128 Q matrix to arrays 0..63, then reads
  and programs one 128-float K vector per array. All 64 arrays launch as one
  group and therefore the utilization invariant is exactly 64/64.
- HBM peak floor: each memory node has 8 HBM12 controllers, each configured
  for a 32-byte payload/cycle. Four balanced memory nodes provide 1024 B/HBM
  cycle, or 1280 B per 1 GHz native cycle at the 1.25 GHz HBM clock. Therefore
  17 MiB needs ceil(17*2^20/1280)=13,927 native cycles at unattainable-perfect
  utilization. The configured 20-HBM-cycle read latency is 16 native cycles,
  but DRAM timing/network/queue costs prevent treating that as the full DMA
  latency.
- With double buffering, the resource/roofline lower bound is not the sum of
  HBM and compute floors: max(13,927, 34,496)=34,496 native cycles. A fully
  serialized ideal reference would be 48,423 cycles, but that is not the
  architecture's lower bound.
- Current QK no-queue service arithmetic per tile is 130 cycles Q local read,
  135 Q matrix broadcast, 64*(9+9)=1,152 K local-read/input-program cycles,
  130 compute cycles, and 64*(2+16*2)=2,176 readout/score-write cycles: 3,723
  cycles plus a one-cycle stage boundary in measured accounting (3,724).
  Across 64 tiles this is approximately 238,336 cycles for QK alone, explaining
  why the compute-only 34,496-cycle floor is not an expected simulator result.
- Combining the exact 3,723-cycle QK service sum with only the softmax/PV
  compute floors gives a current-path lower bound of
  64*(3,723+277+132)=264,448 cycles. It remains optimistic because SFU local
  traffic, all PV preparation/readout, HBM fill/drain, queues, and sync remain
  excluded.
- Clarified the loop nest from source: key tile is outermost and the four query
  blocks in the reuse group are inner, so one K/V DMA pair serves four QK/SFU/PV
  tile executions. Q blocks are loaded on the first key tile and retained.
- Clarified D panel: it is a width-16 slice of the D output dimension. D128 has
  eight panels; sequential PV maps four panels x 16 query rows to 64 arrays per
  wave, hence two waves.
- Per-node media-only DMA examples at ideal 320 B/native-cycle are about 42
  cycles for 8 KiB Q and 119 cycles separately for 32 KiB K or V. Concurrent
  K+V share bandwidth, so their aggregate media floor is about 221 cycles. The
  separate Local-GM landing service is 130 cycles for Q and 1,040 cycles for a
  K/V pair; exact issue-to-ack latency needs Ramulator/NoC simulation.
- The verified sequential64 Q256/K128 critical-worker two-tile breakdown is:
  K/V load 5,110, Q local read 260, QK matrix program 272, QK input program
  2,432, QK compute/readout 4,744, softmax 1,009, PV matrix/preparation 4,672,
  PV compute 272, and PV output read/write 1,280 cycles. WCP maximum queue wait
  is one cycle with no full stalls, and K/V ready wait is zero. Thus current
  latency is dominated by service/data-motion phases, not queue residence.
- `programming cycles` mean time from a WCP operand command until the array
  operand/output state is ready: matrix programming, input-vector programming,
  or restored-O programming. It excludes MAC execution. In the current phase
  counters, PV `matrix_program` also spans V Local-GM reads, P input multicast,
  and later-tile O restore, so that label is broader than matrix bytes alone.
# R15: 64x64 reference-alignment audit (2026-09-15)

- Reference `configs/10_core_gemm.env` specifies array input 64, output 64,
  count 64, and GEMM block M/N/K 64. Chapter 4 runners explicitly repeat the
  64/64/64 contract.
- The LLM base config still matches. Sequential Attention alone overrides the
  array to `HEAD_DIM x 16` and count 64. This uncommitted override is the source
  of the mismatch; cluster mode already selects 64x64 arrays.
- In `cpu_builder.py`, modeled `num_cu` equals array output size. Therefore
  output16 models 16 parallel dot-product lanes per array, while reference
  output64 models 64. Correcting the geometry changes compute resources as well
  as buffer shape.
- Reference K64 semantics require D128 operations to use two micro-tiles and
  accumulation. The custom Attention path currently programs D-wide operands
  directly and needs an explicit two-pass QK implementation rather than only a
  configuration change.
- The guest descriptor hard-codes `query_block_rows=16`; worker admission also
  requires exactly 16 and sequential64 construction explicitly requires array
  output16. These are separate gates, not one runner override.
- Sequential PV topology assumes four D16 panels times 16 query rows and uses
  array offsets +16/+32/+48. QK readout validates exactly 16 outputs. All must
  be replaced for a 64-row/64-output tile.
- SFU row contexts default to 16, and local online Attention explicitly rejects
  multi-key-tile requests when rows exceed row contexts. A Br64 tile therefore
  needs either 64 persistent online row contexts or a redesigned batched state
  store; simply letting 16 contexts recycle would alias online `(m,l)` state.
- Reference `block_k=64` uses the generic WCP micro-tiler and partial-C
  accumulation machinery. Fused Attention disables the generic
  RequestScheduler by default and drives WCP array proxy operations from its
  custom RoCC state machine, so reference micro-tiling is not inherited.
- FP32 is aligned with the current reference run profile (`60_run.env`), but
  Attention-specific SFU, Local-GM port timing, operand banks, broadcast fabric,
  near-array output path, and GroupCtrl K/V mechanisms do not exist in the
  reference base and must be justified as explicit extensions, not called
  reference-aligned behavior.
- The reference-consistent Attention mapping is now frozen at Br64/Bc64. QK
  broadcasts each `Q[64,64]` K-slice as the common array matrix and sends one
  `K[key,64]` vector to each key-indexed array; D128 therefore needs two K64
  launches with overwrite then accumulate. The final 64 arrays x 64 outputs are
  the complete `S[64,64]` tile in array-major (key-major) order.
- PV uses the same GEMM convention rather than the old row-indexed topology:
  broadcast `P[64,64]` as the common matrix, send one `V[:,d]` vector to each
  D-indexed array, and obtain 64 query outputs per array. D128 is two D64 output
  panels, each one 64-array launch. A D panel now means 64 output dimensions,
  not the superseded width-16 panel.
- Br64 eliminates the four Br16 query blocks previously used to amortize one
  K/V tile. The sequential path should use query-group size one and remove its
  four-way Q/O storage and K/V pair-reuse assumptions; K/V is still fetched once
  per worker because that worker's single Br64 block consumes every key tile.
- The existing group-output operation is not a usable 64-array gather: it only
  admits adjacent pairs and only the cluster QK/PV ownership ranges. The new
  mapping needs a bounded 64-array score/O gather or an explicitly modeled banked
  alternative, followed by the required key-major-to-row-major transpose.
- Reference array operand writes are direct `setMatrixItem`/`setVectorItem`
  operations inside WCP and have no separate array-buffer programming latency.
  The LLM fork adds timed 64 B/cycle array-buffer ports, multicast/broadcast
  trees, output credits, and near-array drains. These timing mechanisms must be
  retained and parameterized as LLM extensions or removed from the comparison;
  they cannot be described as reference behavior.
- The current matrix-broadcast API can fan out a common matrix, which matches
  both corrected QK and PV. Unique K/V column vectors are not multicast data;
  they need 64 destination-specific input writes. Any proposed parallel scatter
  needs its own physical bandwidth/port model and cannot be treated as a free
  broadcast.
- QK D-reduction accumulation and online-Attention accumulation are distinct.
  QK retains array C across its two K64 launches. Across key tiles, softmax
  rescales the old O before PV accumulation; because the arrays are reprogrammed
  for QK between PV tiles, O cannot remain implicitly resident in the array.
- The runner also disables RequestScheduler, forces one operand bank and 16 SFU
  row contexts, advertises generic block M/N=16, and fixes the Attention window
  formula to Br16. These settings must be changed coherently rather than only
  switching the three array geometry variables.
- Required verification migration includes guest/manager/worker descriptor
  admission, window sizing/layout, SFU result vectors, QK/PV wave definitions,
  traffic classes, output ordering, lifecycle conservation, utilization checks,
  numerical tests, and every verifier formula that currently encodes 16 rows,
  16 output values, four D16 PV panels, or four-query K/V reuse.
- With the reference latency parameters, one full 64x64 MVM launch is 66 native
  cycles (`64` MAC cycles plus pipeline depth `2`). Thus array-compute work per
  Br64/Bc64 tile is QK `2*66=132` cycles and PV `2*66=132` cycles for D128.
  The 264-cycle array-only figure is valid, but the full Attention lower bound is
  intentionally not frozen until the Br64 SFU and 64-array gather/scatter timing
  are specified. All earlier 539/tile, 34,496, 264,448, and width-16 D-panel
  results describe the superseded D128x16 design.
- A balanced Br64 SFU should expose 64 logical online `(m,l)` contexts while
  retaining a pipelined 16-value max path, 16-value EXP/sum path, and 16-value
  normalize path. For a 64x64 score tile this changes the modeled stage times to
  4/4/4 cycles per row and gives `12 + 63*4 + 13 = 277` native cycles. Expanding
  EXP beyond 16 lanes has almost no benefit because max/normalize then limit II
  to four cycles; 64 EXP lanes would only reduce 277 to 274.
- The optimized SFU boundary needs a 64-bank key-major-to-query-major transpose
  SRAM. At the existing 512 B/cycle near-array drain a 16 KiB score tile enters
  in 32 cycles, while the 16-lane SFU consumes/emits 64 B/cycle. P should stream
  directly from normalize into the 64-array matrix-broadcast/program path,
  avoiding score and P round-trips through Local-GM.
- The corresponding compute-only tile floor is QK 132 + SFU 277 + PV 132 =
  541 native cycles. A Q=K=1024 critical worker has 16 Bc64 tiles, hence 8,656
  compute cycles. The unchanged 17 MiB global HBM media floor is 13,927 cycles,
  so the ideal resource roofline is at least 13,927 cycles.
- This 13,927-cycle number requires a widened/banked array operand supply. With
  the current single 64 B/cycle array-buffer ingress, even keeping both Q slices
  resident, K vectors (32 KiB), V vectors (32 KiB), and P matrix (16 KiB) require
  1,280 raw ingress cycles per key tile. Across 16 tiles plus the initial 32 KiB
  Q load, the operand-ingress bandwidth floor alone is 20,992 cycles before
  command setup, output traffic, or pipeline boundaries.
- Multi-tile MVM diagnostics exposed a sequential-state bug: tile 0 executed
  both K64 reductions, but an ordinary next-tile transition retained
  `qkReductionHalf=1`, so tile 1 and later skipped the low half. The correct
  reset point is the common `beginAttentionKeyTile()` entry used by normal DMA,
  prefetch-hit, and grouped-query paths.
- For Q1024 the base runner's default generic `A/B reuse=8` reserved 64 partial-C
  tiles and expanded the SST Global-Memory stride beyond 2 MiB. Fused Attention
  disables the generic RequestScheduler and never uses that scratch, so its
  runner must select reuse 1 explicitly to keep the SST/guest 2 MiB ABI stable.
- LocalGM admission failure is transient backpressure, not a failed DMA. Keeping
  a tagged landing chunk pending and retrying it on the 1 GHz local-access link
  removes asynchronous K/V false failures without increasing the DMA timeout.
- Sequential PV must retain the complete V tile after panel 0. Re-reading V for
  panel 1 after releasing the ping-pong buffer races next-tile prefetch and
  corrupts only D[64:128], which is why the initial Q1024 error clustered at D64
  boundaries while QK, softmax, and D[0:64] remained correct.
- The optimized non-cluster SFU now retains each 64-value row across Max,
  EXP/Sum, and Normalize. On Q1024 this reduced the measured softmax total from
  25,115 to 12,567 cycles and LocalGM retry events on the critical worker from
  roughly 388k to 117,520 while preserving exact numerical verification.
- Final Q1024/K1024/D128: 576,657 end-to-end cycles; critical worker input
  96,904, QK 311,721, softmax 12,567, PV 85,682; zero of 131,072 values mismatch,
  max absolute error 1.713e-09. The resource roofline is 20,992 cycles, and the
  largest remaining cost is the serialized QK score readout/LocalGM path.
# R24: 256 B/cycle matrix broadcast and 64-lane vector scatter (2026-09-15)

- The reference RTL is not an interface requirement for this SST architecture.
  It was useful only as prior width evidence; the requested worker-local model
  is defined directly by its own bandwidth, fanout, payload, and timing contract.
- Matrix broadcast and vector scatter are different operations. Broadcast reads
  one common 64x64 FP32 matrix (16 KiB ingress) and replicates it to 64 matrix
  banks. Scatter reads 64 destination-specific K or P vectors (64x64 FP32,
  16 KiB aggregate) and sends lane `i` to input bank `i` without replication.
- Both fabrics target 256 B/native-cycle, but need independent parameters and
  statistics so tuning one cannot silently retime the other. One full scatter
  has a raw bandwidth floor of 16 KiB / 256 B/cycle = 64 cycles; fanout is not
  multiplied again because 256 B/cycle is the aggregate 64-lane width.
- Current sequential QK performs 64 separate 256 B LocalGM reads and 64 WCP /
  single-array input writes per K64 reduction slice. Current sequential PV
  similarly performs 64 P-row reads and group-input calls per D64 wave. The
  existing group-input primitive replicates one common vector and therefore
  cannot represent destination-specific 64-lane scatter.
- Implemented independent 256 B/cycle matrix-broadcast and input-scatter
  parameters. QK now reads one complete K tile and issues two 64-array K64
  scatters; PV reads P once, scatters it once, and reuses the input bank across
  both D64 output slices. Matrix and scatter still share the modeled array-buffer
  port, so their service cannot overlap for free.
- The final fixed QK1024 run passes numerical and lifecycle verification at
  62,900 cycles. Critical-worker phases are input 4,375, QK 6,736, softmax
  12,569, and PV 37,378 cycles. Per worker scatter accounting is exactly 48
  requests, 786,432 bytes, 3,120 service cycles, and zero rejects.
- The revised intended dataflow floor is 1,024 cycles/key tile: four 71-cycle
  matrix broadcasts, three 65-cycle scatters, and the existing 545-cycle
  QK/SFU/PV compute/read-write chain. Across 16 key tiles this is 16,384 cycles;
  the unchanged 16,896-cycle HBM read-command floor therefore sets the current
  end-to-end theoretical resource lower bound.
# R25: Sequential PV old-O restore and output-path diagnosis (2026-09-15)

- Fixed QK1024 has 16 key tiles, two PV D64 output slices per tile, and 64
  arrays per slice. Old O is absent on tile 0, so restore executes
  `15 * 2 * 64 = 1,920` strictly serialized per-array iterations. The measured
  15,360 cycles are exactly 8 cycles/iteration.
- PV output is drained after both slices on every tile, so it executes
  `16 * 2 * 64 = 2,048` per-array iterations. The measured 16,386 cycles are
  `2,048 * 8 + 2` phase-boundary cycles.
- Restore source flow is one 256 B LocalGM read, FP32 scale in the callback,
  then one 256 B array-output write, followed by recursive issue of the next
  array. Output flow still needs exact source confirmation, but its measured
  eight-cycle cadence already matches the reverse per-array read/write chain.
- WCP evidence rules out queue congestion as the cause: maximum worker proxy
  queue wait is one cycle and queue-full stalls are zero. The latency is exposed
  serialized service/setup repeated 1,920/2,048 times, not accumulated waiting
  behind a deep queue.
- The restore callback chain is one 256 B LocalGM read, one WCP
  `WRITE_OUTPUT` command, and one 256 B array-output write. With LocalGM at
  16 KiB/cycle and the array-output path at 64 B/cycle, this is
  `(1+ceil(256/16384)) + 1 + (1+ceil(256/64)) = 2+1+5 = 8` cycles per array,
  exactly explaining `1,920 * 8 = 15,360` cycles.
- The output chain reverses those resources: one WCP `READ_OUTPUT` command,
  one 256 B array-output read, then one 256 B LocalGM write. Its request body is
  also `1+5+2 = 8` cycles, explaining 16,384 of the measured 16,386 cycles.
- A worker's final O is only `64*128*4 = 32 KiB`, but online-softmax tiling
  restores it after 15 key tiles (480 KiB) and writes it after all 16 key tiles
  (512 KiB). Thus these paths move 992 KiB of old/updated O per worker.

# R26: Grouped 256 B/cycle O scatter/gather (2026-09-15)

- The array-side unit of work is one D64 slice: 64 arrays x 64 FP32 values =
  16 KiB, modeled as one shared-port group transfer at
  `1 + ceil(16384/256) = 65` cycles.
- LocalGM O is row-major, so one D64 slice is strided rather than contiguous.
  To preserve the existing final-output DMA ABI, each K tile caches the complete
  32 KiB O using two legal 16 KiB LocalGM requests, updates each slice via one
  grouped scatter/gather, then commits the complete cache after slice 1.
- The full-cache approach predicts 136 restore cycles per noninitial K tile and
  136 output cycles per K tile after WCP and LocalGM timing: 2,040 and 2,176
  cycles respectively for QK1024, before any phase-boundary residue.
- QK1024 measures exactly 2,040 restore and 2,176 output cycles. Critical-worker
  array statistics are exactly 30 scatters, 32 gathers, 1,015,808 bytes, 4,030
  service cycles, 64 maximum destinations, and zero rejects.
- Numerical and lifecycle verification pass at 44,790 cycles with zero of
  131,072 mismatches and maximum absolute error 1.713e-09. PV falls from 37,378
  to 9,848 cycles, while end-to-end falls from 62,900 to 44,790 cycles.
- The shorter PV path exposes K/V prefetch latency: critical-worker K/V wait is
  now 10,173 cycles and input movement rises from 4,375 to 13,654 cycles. This
  is why a 27,530-cycle PV reduction becomes an 18,110-cycle end-to-end gain.
- The complete dependency/resource lower bound is now 20,600 cycles: the prior
  16,384-cycle operand/compute path plus 4,216 cycles of grouped O state
  movement. It exceeds the unchanged 16,896-cycle HBM command floor.

# R27: K/V prefetch theoretical-cycle diagnosis (2026-09-15)

- Fixed QK1024 uses 1 GHz worker/array clocks, 1.25 GHz memory-controller
  clocks, four data-memory nodes, 320 GB/s high links, 512 GB/s NoC links,
  16 KiB DMA chunks, and a 256-chunk per-core DMA in-flight cap.
- One worker consumes one K tile and one V tile per key step. Each is
  `64*128*4 = 32 KiB`, so one K/V pair is 64 KiB and becomes four 16 KiB DMA
  requests. The in-flight cap therefore does not serialize a single pair.
- The measured critical-worker counters distinguish 15 prefetch lifetimes
  totaling 37,034 cycles from only 10,173 exposed wait cycles across 11 late
  prefetches; four prefetches were ready early with 2,127 cycles total lead.
- The normal two-buffer state machine launches tile `i+1` only after the initial
  pair or an activated prefetched pair has completed. Its hiding window is
  therefore approximately the non-K/V work of tile `i`, not the entire job.
  Second-lookahead additionally requires both current K and V operands released,
  tile `i+1` already ready, and a reusable/free buffer; it triggered only three
  times in the fixed run.
- Critical-worker non-K/V work is `(64 + 6,732 + 12,581 + 9,848)/16 =
  1,826.6` cycles/tile, versus `37,034/15 = 2,468.9` cycles average DMA
  lifetime. The simple steady-state deficit is therefore about 642.3 cycles per
  prefetch, or 9,635 cycles across 15 transitions, close to the measured 10,173
  exposed cycles; arbitration variance and issue/consume boundary effects account
  for the remainder.
- `attentionKvHostAddrForTile` stripes by 256-row bands, while a K/V tile is 64
  rows. Each complete pair therefore comes from one data node; four consecutive
  tiles target the same node. With distribution disabled, all 16 workers issue
  duplicate reads of that same pair rather than one HBM read plus multicast.
- At 1 GHz, the 320 GB/s node link carries 320 B/cycle, giving a raw pair floor
  of `ceil(65,536/320)=205` cycles. The HBM command roofline is tighter: eight
  channels times 32 B/cycle = 256 B/cycle/node, hence 256 cycles/pair before
  fixed DRAM, NoC, response, and LocalGM landing latency.
- A same-tile 16-worker wave reads 1 MiB from one node and therefore has a
  4,096-cycle node-command service floor. Across all QK1024 K/V data, each of
  four nodes receives four such waves (4 MiB/node), so the no-distribution job
  has a 16,384-cycle K/V command-resource floor. Ideal read-once distribution
  would reduce this to 1,024 cycles (256 KiB/node).
- The measured tile K/V phase closes exactly: 13,589.25 cycles equals the fully
  exposed initial-pair cold start of 3,416.75 cycles plus 10,172.5 cycles of
  exposed prefetch waits. It is not the 37,033.75-cycle sum of asynchronous
  prefetch lifetimes.
- Current per-prefetch lifetime is 2,468.9 cycles average (1,272.875 minimum,
  3,525.5 maximum). The 11 late cases expose 924.8 cycles on average; across all
  15 transitions, exposed wait averages 678.2 cycles. Four hits arrive 531.7
  cycles early on average.
- Request injection is not the dominant limiter: the critical worker's DMA
  sends are immediate with no source queue. The dominant mechanisms are HBM
  duplicate-read contention, coarse node concentration, one-tile lookahead, and
  return-side credit/admission pressure. Global memory queue delay is 129 cycles
  average but 1,882/1,999 cycles at p95/p99, consistent with the tail waits.

# R28: shared streaming K/V supply (2026-09-15)

- Worker GlobalMemory already splits each 32 KiB K or V tile into two 16 KiB
  requests. Chunk completion is therefore the natural streaming boundary; no
  new sub-tile packetization is required.
- All 16 worker requests converge in the data-node `MemNIC` bridge before an HBM
  `GetS` is created. A physical read currently owns exactly one
  `returnEndpoint/requestId`, so identical-address requests cannot share its
  response without an explicit subscriber list keyed by host address and size.
- The existing GroupCtrl distributor is not the final requested mechanism: it
  coalesces only four workers per manager, waits for full K and V tiles in
  manager LocalGM, then sends four independent full-data deliveries. It reduces
  HBM duplication fourfold but neither spans all 16 workers nor streams chunks.
- The selected R28 boundary is therefore the shared data-node bridge. The first
  eligible Attention K/V chunk creates one HBM read; later identical chunks join
  its subscriber list. On completion, the chunk is fanned out immediately. The
  NoC accounting must represent one full data injection plus broadcast-branch
  control, otherwise sixteen ordinary unicast payloads would merely move the
  bottleneck from HBM to the sender link.
- Three complete pair buffers do not fit the existing 256 KiB worker Attention
  window: K/V would use 192 KiB, Q/O 64 KiB, and scores 16 KiB. R28 therefore
  keeps two pair buffers for the first coalescing measurement. If a second
  lookahead remains necessary after deduplication, the capacity-correct form is
  three 32 KiB K slots plus two 32 KiB V slots (160 KiB total), not three full
  64 KiB pairs.
- The first compiled fixed-QK1024 run passed numerical and lifecycle checks and
  completed in 31,770 cycles versus the R26 44,790-cycle baseline: 13,020 cycles
  lower (29.1%). Critical-worker K/V wait fell from 10,172.5 to 0 cycles; all 15
  prefetches were hits. Reported operator stages were input 533, QK 6,736,
  softmax 12,567, and PV 9,848 cycles.
- In-flight coalescing alone does not meet the read-once contract. The four data
  nodes reported 176 physical reads (46/44/42/44), 848 coalesced followers, and
  1,024 receivers. The strict unique-chunk target is 64 physical reads: 16 K/V
  tile pairs x four 16 KiB chunks per pair. Worker launch skew lets an HBM
  response retire before every worker has issued its matching request, so a
  completed shared-node chunk cache is required for late consumers.
- The fixed 64-worker topology has four shared memory nodes, so each unique K/V
  chunk has 16 consumers per node. This count is now an explicit architecture
  parameter rather than an implicit cache-lifetime assumption.
- The completed-cache run met the exact traffic contract: every node reported
  16 physical reads and 262,144 unique multicast bytes; totals are 64 reads and
  1 MiB. The 1,024 logical consumers split into 609 in-flight followers, 351
  completed-cache hits, and 64 physical owners. All cache entries retired.
- That run completed in 31,197 cycles with input 137, QK 6,736, softmax 12,567,
  and PV 9,848; all 15 critical-worker prefetches hit with zero exposed wait.
  The worker still gates ordinary QK activation on pair-level `ready`, however;
  its existing `kReady` callback only enables cluster-mode matrix lookahead, so
  independent K-before-V consumption remains an implementation gap even though
  it is off the measured critical path after global deduplication.
- Independent readiness can reuse the existing two-buffer lifecycle safely if
  the active descriptor is retained instead of cleared at QK activation. K
  readiness permits QK to start; V pair readiness completes later in the same
  descriptor; the active descriptor is released only at tile completion. The
  alternate buffer remains available for the next prefetch throughout.
- After implementing K-first activation and the PV-side V gate, the final
  QK1024 run remains exactly 31,197 cycles and passes numerical, backend, and
  lifecycle verification. The unchanged cycle count is expected: all 15 pair
  prefetches are already ready before use, so independent readiness removes a
  possible dependency but has zero exposed cycles in this workload.
- Final critical-worker tile accounting is 29,288 cycles with exact phase
  conservation: K/V load 73, Q-local read 64, QK matrix 1,152, QK input 2,176,
  QK compute/readout 3,408, softmax 12,567, PV matrix 2,368, PV input 1,088,
  PV restore 2,040, PV compute 2,176, and PV output 2,176 cycles.
- The updated dependency/resource lower bound remains 20,600 cycles because the
  unique K/V stream requires only 1,024 cycles across the four 256 B/cycle
  nodes and is hidden below the 16,384-cycle operand/compute chain. Adding
  grouped O movement gives 16,384 + 4,216 = 20,600. Relative to 31,197 actual,
  the residual is 10,597 cycles (51.4% over lower bound); 8,135 cycles of it are
  the softmax gap (12,567 actual versus 4,432 at 277 cycles/tile).

# P5: GitHub progress publication (2026-09-15)

- The working branch is `softmax-update`; `origin/softmax-update` is currently
  at `4d47f83`, while local HEAD is two commits ahead at `ad5642f`.
- The cumulative worktree contains the full sequential-64 evolution and R28,
  not only the latest MemNIC patch. The two intentional untracked documentation
  additions are `attention_sequential_64/README.md` and the R12 Ramulator2
  report; no generated run directory is present in the Git status.
- Root `README.md` still advertises the older 103,757-cycle R19 result and must
  be refreshed to 31,197 cycles, 64 physical K/V reads, zero exposed K/V wait,
  and the 20,600-cycle current lower bound before publication.
- `attention_sequential_64/README.md` contains the R24/R26 evolution after its
  older R19 table but has no R28 section. The attention test README still calls
  the historical 16/48 cluster the accepted default. Public documentation must
  identify sequential-64 as default and distinguish historical measurements
  from the final R28 result.
- The actual topology remains four managers and 16 workers, with 64 arrays per
  worker. Each unique K/V chunk is consumed by all 16 workers; striping assigns
  different chunks to four HBM nodes, which is why every node reports 16 unique
  chunks and 256 logical receivers. Public wording must not confuse 64 arrays
  per worker with 64 workers.
- The sequential architecture document already contains the R24 256 B/cycle
  operand fabrics and R26 grouped-O result. A final R28 section can supersede
  its historical tables without deleting the useful optimization progression.
- The GPU roadmap is the last public-facing file that incorrectly labels R12
  cluster as current. It should retain R12 as history but make the fixed
  QK1024 R28 result the active baseline; E4 remains deliberately unmeasured
  under the user's QK1024-only optimization policy.
- After `git fetch`, `origin/softmax-update` has not advanced: the branch is
  zero behind and two local commits ahead. No rebase or merge is required.
- The publication candidate passes all 106 focused Attention tests, shell
  syntax checks, and `git diff --check`. The stale-current-value scan returns
  no matches in public Attention Markdown files.
- Publication commit `15bab3f` was created successfully. The first push failed
  before contacting repository authorization because the HTTPS remote has no
  credential source. There is no `gh` executable, configured Git credential
  helper, active SSH agent, or accepted GitHub SSH key in this environment.
- Environment fallback checks confirm `GH_TOKEN` and `GITHUB_TOKEN` are unset,
  no GitHub-related askpass variable is present, and no user git-credentials
  file exists. Remote publication therefore requires user-side authentication;
  retrying the same unauthenticated push cannot succeed.
