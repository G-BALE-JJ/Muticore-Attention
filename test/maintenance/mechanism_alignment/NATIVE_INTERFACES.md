# Native mechanism interfaces

The default LLM guest, task-list ABI, hardware resources and scheduling flags
remain the compatibility contract. The interfaces below are explicit native
runtime entry points; the selected LLM guest does not automatically use them.
No TileLang compiler, TileBin loader or compiler task descriptor is required.

The public `prefill --native-operator` experiment now connects these interfaces
to LLM-shaped QKV, WO, gate/up/down and LM-head matrices through the existing
GEMM guest and real production NoC/HBM. `native_llm_workload.py` builds packed
images; explicit per-TE parameters preserve physical scheduler slots while
selecting logical instance slots, precision and placement. Default decoder
execution retains its original FP16 contract. Native operator outputs use
INT32/FP32; this entry does not claim full-decoder eight-precision inference.
See `test/prefill/README.md` for padding and input-encoding limits.

## CU, TU and TE

`computeunit/computeUnit.h` contains scalar reductions.
`tileunit/tileUnit.h` composes reductions while retaining the legacy FP16
rounding rules. `golem.TileUnitArray` exposes the existing physical array as a
native TU with `numTileUnits`, `tuKCapacity` and `cusPerTileUnit` aliases.
The existing MVM array names and parameters remain supported.

`golem.TileEngineLocal` implements the execution engine. The historical
`golem.WorkerCommandProcessorLocal` delegates to the same implementation and
retains its API, events and callbacks, including Attention, GQA and decoder
paths. Neither interface creates additional compute units or buffers.

## Packed precision

Native TU clients select an explicit `XTileMC::PrecisionSpecV02` with
`configurePrecision(spec, activeK)`, load raw operand bits using
`setMatrixBits` / `setVectorBits`, and read or seed 32-bit output bits using
`outputBits` / `setOutputBits`. `useLegacyPrecision()` restores the original
arithmetic and latency model.

Native TE clients use `startPrecisionWindow(header, spec)`. Supported modes are
INT4, INT8, FP16, BF16, TF32, FP32, FP8 E4M3 and FP8 E5M2 under the donor's
versioned arithmetic contract. This includes its rounding and INT32/FP32
accumulation semantics; it does not change legacy LLM FP16 arithmetic.
Packed TE windows require row-major panel layout, a K block equal to TU K
capacity, packed row strides and 4-byte output elements. Unsupported contracts
are rejected rather than silently converted.

`computeunit/peTiming.def` is shared by the C++ model and
`test/golem/architecture/pe_timing.py`. Calibrated timing applies only after
explicit precision selection. All reported accelerator latencies use cycles.

## Independent workloads and HBM panel placement

`test/golem/tools/native_workload_plan.py` partitions existing worker groups and
HBM data nodes into independent instances. Node 0 remains OS-only. Plans can
use group-affine or reuse-striped panel mapping and optional B replication.
The planner checks tensor regions against each existing HBM node's capacity.
Its `write_images` function accepts distinct A/B tensors for each instance and
writes panel images for ordinary element-byte layouts. It is not a packed
precision encoder.

Native runtime code supplies each instance's maps through `WorkloadPlacement`
and calls `startPlacedWindow(header, placement, optionalPrecision)`. The
header's worker slot is logical within the instance; the scheduler's worker
slot remains physical. The runtime must use the planned physical worker and
node partitions. The original task-list ABI is unchanged.

Explicit placement supports no reuse or reuse in both M and N dimensions.
One-sided reuse uses a separate legacy resident protocol and is rejected by
the native placement entry point. Explicit DMA source lists let neighbouring
panels reside on different HBM nodes. Default windows retain their original
address derivation.

## Timing-only execution and evidence

With `functionalCompute=0`, the fast path avoids materializing numerical
operands and partial results while retaining transfer sizes, buffer requests,
compute completion events and output completion callbacks. Numerical checking
requires functional execution.
`timingNumericalFastPath=0` retains numerical materialization for an unoptimized
timing control in the same installed build. The baseline architecture exposes
this validation switch as `GOLEM_TIMING_NUMERICAL_FAST_PATH=0`; its default is 1.
Functional execution always materializes numerical data.

`test_native_packed_array.py` exercises actual installed SST TU arithmetic and
calibrated completion events. `test_native_engine.py` exercises the production
TE/TU path with deterministic memory/scheduler fixtures, including cross-K
accumulation, placement, B replication and equal functional/timing event
cycles. It also runs two native engines concurrently with distinct tensor
values and disjoint mapped HBM nodes, separating logical and physical worker
slots. These fixtures do not establish NoC/HBM performance for a new mapping.
`test_native_workload_placement.py` checks disjoint partitions, tensor image
contents and agreement between Python layouts and C++ runtime addresses.

Each migration phase additionally requires the four full functional gates
in `PLAN.md`; their default LLM runs exercise the real NoC/HBM system.
`validate_timing.py` records a paired timing-only GEMM before and after the
numerical-data fast path, then compares the optimized path with a control in
the identical build. The same-build comparison checks execution, output,
memory, DMA, NoC and HBM command statistics. Neither unchanged default cycles nor these fixture
checks establish a performance improvement for an unmeasured opt-in workload.

`test/run_native_llm_regression.py` additionally runs functional/timing pairs
through the real public native operator entry on one installed build. It covers
all eight precisions, all six operator shapes, separate stripe/replication
controls and two/four independent workloads, including MPI. Functional outputs
are compared with each instance's own input product; timing numerics are skipped.
Every pair requires equal cycles, output counts and physical worker completion.
