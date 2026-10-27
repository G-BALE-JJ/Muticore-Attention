# Compile Boundaries

## Goal

Avoid unnecessary full-workspace rebuilds and avoid leaving the install tree in a mixed ABI state.

## Rule 1

If a change touches `globalmemory/globalmemory.h` or `globalmemory/globalmemory.cc`, do a **full clean rebuild + install** of the whole workspace.

Reason:

- `globalmemory.h` is directly included by files in `libgolem`
- `globalmemory.h` is also directly included by `memHierarchy/memNICBase.h`
- `sst` loads these libraries independently at runtime, so a partial rebuild can leave incompatible class layouts or symbol expectations across libraries

Required commands:

```bash
cd /data4/lishun/pkg/sst-elements
make clean
./configure --prefix=/data4/lishun/pkg/sst_install --with-dramsim3=/data4/lishun/pkg/DRAMsim3
make -j4
make install
```

## Rule 2

If a change only touches files under `src/sst/elements/golem/` except `globalmemory/*`, rebuild **only `libgolem`**.

Examples:

- `rocc/roccAnalog.h`
- `requestscheduler/requestscheduler.{h,cc}`
- `groupctrl/groupctrl.{h,cc}`
- `workercmdproc/workercmdproc.h`

Commands:

```bash
cd /data4/lishun/pkg/sst-elements/src/sst/elements/golem
make -j4
make install
```

## Rule 3

If a change only touches test/runtime sources under:

- `tests/small/mvm_noc_int_array/*`
- `tests/configs/*.env`
- `tests/run_noc_dma_pipeline.sh`
- `tests/stats/*.py`
- `tests/tools/*.py`

do **not** rebuild SST element libraries unless a shared library header/API changed.

Behavior:

- C/C++ test runtime sources are rebuilt by the test script when it rebuilds `riscv64/test_noc_dma`
- Python/shell/env changes need no library rebuild

## Direct include boundary for `globalmemory.h`

Current direct includes found in the tree:

- `src/sst/elements/golem/globalmemory/globalmemory.cc`
- `src/sst/elements/golem/groupctrl/groupctrl.h`
- `src/sst/elements/golem/requestscheduler/requestscheduler.h`
- `src/sst/elements/golem/rocc/roccAnalog.h`
- `src/sst/elements/golem/workercmdproc/workercmdproc.h`
- `src/sst/elements/memHierarchy/memNICBase.h`

Library impact:

- `libgolem.la`
- `libmemHierarchy.la`

So a `globalmemory` header change is **never** a `libgolem`-only rebuild.

## ASan rule

Never leave the install tree half-ASan, half-normal.

If any full-workspace ASan build was performed, restore the install tree with a **full clean normal rebuild** before running normal regressions.

Symptoms of violation:

- `undefined symbol: __asan_option_detect_stack_use_after_return`
- runtime library load failures like `unable to find "merlin" element library`

## Run script rule

Never run `run_noc_dma_pipeline.sh` in parallel.

Reason:

- it shares `artifacts/hbm`
- it shares some top-level generated outputs
- parallel runs can corrupt backing files and invalidate results

Always run regressions serially, one config at a time.

## Practical decision table

### Full workspace rebuild required

- touched `globalmemory/*`
- touched anything in `memHierarchy/*`
- recovering from ASan/full-workspace instrumentation

### `libgolem`-only rebuild required

- touched `rocc/*`
- touched `requestscheduler/*`
- touched `groupctrl/*`
- touched `workercmdproc/*`
- touched other files only compiled into `libgolem`

### No library rebuild required

- touched `tests/configs/*`
- touched `tests/stats/*`
- touched `tests/tools/*`
- touched `run_noc_dma_pipeline.sh`
- touched only RISC-V test sources, where the script already rebuilds the test binary
