# Native TU implementation and legacy adapters

`TileUnitArrayModel<T>` owns the physical operand banks, output state, CU
reductions and MV completion events. It derives from `TileUnitAPI`, a shared
transport/API layer over `ComputeArray`; it has no dependency on legacy MVM
implementation headers. `TileUnitArray` registers the native floating bank.

The historical `MVMComputeArray<T>` is a thin compatibility adapter over this
native model. Existing MVMFloatArray/MVMFp16Array/MVMIntArray names and CrossSim
interfaces remain available. TE, Attention and decoder still use their existing
timed port APIs and callbacks. The split adds no buffers, arrays, events or
clocks, and preserves legacy FP16 rounding and FP32 reduction order.

Packed precision, active-K behavior, timing materialization control and the
shared PE timing table live in the same native model. Default decoder paths
continue to select their original arithmetic. The fixed native descriptor
runner explicitly selects `golem.TileUnitArray`.

`test_native_packed_array.py` exercises both native and MVM registrations in
functional and timing modes. Its C++ driver statically checks that the native
TU does not derive from the legacy MVM adapter. `test_native_engine.py` covers
the production TE with packed arithmetic, placement, B replication, partial-C
accumulation and concurrent distinct workloads. Four full functional stage
gates additionally protect the existing decoder contract and cycles.
