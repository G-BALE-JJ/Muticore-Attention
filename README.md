![SST](http://sst-simulator.org/img/sst-logo-small.png)

# Structural Simulation Toolkit (SST)

#### Copyright (c) 2009-2025, National Technology and Engineering Solutions of Sandia, LLC (NTESS)
Portions are copyright of other developers:
See the file CONTRIBUTORS.TXT in the top level directory
of this repository for more information.

---

The Structural Simulation Toolkit (SST) was developed to explore innovations in highly concurrent systems where the ISA, microarchitecture, and memory interact with the programming model and communications system. The package provides two novel capabilities. The first is a fully modular design that enables extensive exploration of an individual system parameter without the need for intrusive changes to the simulator. The second is a parallel simulation environment based on MPI. This provides a high level of performance and the ability to look at large systems. The framework has been successfully used to model concepts ranging from processing in memory to conventional processors connected by conventional network interfaces and running MPI.

---

Visit [sst-simulator.org](http://sst-simulator.org) to learn more about SST.

See [SST Elements Documentation](http://sst-simulator.org/SSTPages/SSTDeveloperElementSummaryInfo/) for an overview of the simulation capabilities in this repository.

See [Contributing](https://github.com/sstsimulator/sst-elements/blob/devel/CONTRIBUTING.md) to learn how to contribute to SST.

##### [LICENSE](https://github.com/sstsimulator/sst-elements/blob/devel/LICENSE.md)

## Muticore-Attention status

The integrated development branch is `softmax-update` in
[`G-BALE-JJ/Muticore-Attention`](https://github.com/G-BALE-JJ/Muticore-Attention).
It combines the Golem generic GEMM/WCP path, deterministic MPI partitioning,
Ramulator2-backed memory timing, bounded resource models, and the current
single-head FP32 FlashAttention dataflow.

The primary Attention comparison is `B=1,Hq=Hkv=1,Dh=128`, non-causal, with SST
normalized cycles interpreted at 1 GHz. Host wall time is simulator execution
time and is not accelerator latency.

The current default is a worker-local sequential architecture with four
control-only managers and 16 workers. Every worker has 64 physical 64x64 arrays
and reuses all of them in the dependency order `QK^T` -> online softmax -> PV.
Br64/Bc64/D128 uses two accumulated QK D64 reduction slices and two PV D64
output slices. The SFU has 64 online row contexts, balanced 16-lane vector/EXP
paths, and row-resident intermediate storage. This is the only supported
Attention worker dataflow.

| Workload | Current SST cycles | Status |
|---|---:|---|
| Sq=Skv=1024 | **31,197** | numerical/lifecycle/backend PASS |

The implementation uses worker-local 64-array `QK^T` -> softmax -> PV
execution, grouped score readout, 256 B/cycle matrix/vector/O fabrics, grouped
running-output restore/writeback, and shared-node K/V delivery.

The current multi-head path supports GQA with independent `Hq` and `Hkv`
(`Hq/Hkv` equal to 1, 2, or 4). One composite job shares each K/V head across
its Query-head group, heads are spatially distributed over the 16 workers, and
each worker still uses all 64 arrays for every QK and PV operation. Output is
written directly as `[Sq,Hq,Dh]`, so there is no serial concatenation copy.

For Q1024/K1024, every 32 KiB K or V tile is split into two 16 KiB chunks.
Identical requests from all 16 workers are coalesced at the shared memory node;
completed chunks remain resident for launch-skewed consumers and are delivered
through a 256 B/cycle multicast stream. The final run performs exactly 64
physical K/V reads for 1 MiB of unique data, versus 1,024 logical deliveries.
All 15 critical-worker prefetches hit and exposed K/V wait is zero.

The current dependency/resource lower bound is 20,600 cycles: 16,384 cycles for
the 16-tile operand/compute path plus 4,216 cycles for grouped O movement. The
measured critical-worker phases are 137 input, 6,736 QK, 12,567 softmax, and
9,848 PV cycles. End-to-end latency is 31,197 cycles, down 30.35% from the
44,790-cycle grouped-O baseline. Softmax accounts for 8,135 of the remaining
10,597 cycles above the bound.
Performance experiments and cycle acceptance remain fixed to
`Sq=Skv=1024,Dh=128`;
other shapes remain available only for correctness contracts.

See
[`src/sst/elements/golem/tests/small/muticore_attention/README.md`](src/sst/elements/golem/tests/small/muticore_attention/README.md)
for the runner contract,
[`attention_sequential_64/README.md`](attention_sequential_64/README.md)
for the single-head architecture evolution,
[`attention_sequential_64/GQA_ARCHITECTURE.md`](attention_sequential_64/GQA_ARCHITECTURE.md)
for the parallel GQA dataflow, cycle derivation, and measurements,
[`baseline/reuse_window_flash_attention/README.md`](baseline/reuse_window_flash_attention/README.md)
for the independent `2x4` generic-GEMM reuse-window FlashAttention baseline,
[`baseline/attention_cluster/README.md`](baseline/attention_cluster/README.md)
for the 4 QK/SFU + 12 PV worker-cluster baseline, and
[`baseline/attention_cluster_8qk_8pv/README.md`](baseline/attention_cluster_8qk_8pv/README.md)
for the frozen 8 QK/SFU + 8 PV comparison and its `47,193`-cycle SST archive,
[`attention_sequential_64/ATTENTION_TERMINOLOGY.md`](attention_sequential_64/ATTENTION_TERMINOLOGY.md)
for the canonical terminology and compatibility map.

## Portable Muticore-Attention build

The current integrated CIM/WCP implementation is maintained on the
`softmax-update` branch of `G-BALE-JJ/Muticore-Attention`. Clone that
branch explicitly in a new
environment:

```bash
git clone \
  --branch softmax-update \
  --single-branch \
  https://github.com/G-BALE-JJ/Muticore-Attention.git \
  Muticore-Attention
cd Muticore-Attention
```

### Prerequisites and paths

This repository contains SST elements, not SST core, DRAMSim3, or Ramulator2. The build
needs a host C/C++ compiler, Autotools, `rsync`, `make`, Python 3, and an MPI
launcher. It also needs an SST core installation and a DRAMSim3 prefix. Define
their locations once; do not copy paths from another host:

```bash
export REPO_ROOT="$(pwd -P)"
export SST_CORE_PREFIX=/absolute/path/to/sst-core-install
export SST_DRAMSIM3_PREFIX=/absolute/path/to/DRAMsim3
export SST_RAMULATOR2_PREFIX=/absolute/path/to/ramulator2-2.1
export RISCV_TOOLCHAIN_BIN=/absolute/path/to/riscv64-musl-toolchain/bin
export DRAMSIM3_LIB_DIR="$SST_DRAMSIM3_PREFIX"
export PATH="$SST_CORE_PREFIX/bin:$RISCV_TOOLCHAIN_BIN:$PATH"
```

`SST_CORE_PREFIX` must contain `bin/sst-config`, `bin/sst`, headers, and
runtime libraries. `SST_DRAMSIM3_PREFIX` may be an installed prefix or a
built DRAMSim3 checkout containing the headers and library required by SST.
`DRAMSIM3_LIB_DIR` is the directory that directly contains `libdramsim3.so`;
for the validated source layout it is the same as `SST_DRAMSIM3_PREFIX`.
Adjust it separately if an installed prefix places the library under `lib/`.
The validated DRAMSim3 compatibility point is release `1.0.0`; when using a
Git checkout, record and verify its tag/commit in that environment instead of
reusing an absolute path from the original development machine.

Validated dependency versions:

| Dependency | Validated version |
| --- | --- |
| SST Core | `v15.1.0_Final` (`a7eb776d6845f340e0448535e65bcee29c90d445`) |
| DRAMSim3 | `1.0.0` (`29817593b3389f1337235d63cac515024ab8fd6e`) |
| Ramulator2 | `2.1` snapshot (`72427a1bba3771564c4fb0e494ba02242fd1eaa7`) |
| RISC-V musl compiler | GCC `9.4.0`, target `riscv64-linux-musl` |
| MPI | Open MPI `4.1.2` |

Nearby versions may work, but use the versions above when reproducing the
validated environment or diagnosing an unexplained build/runtime difference.

If the dependencies are already installed, check the environment before
configuring. Otherwise, run the bootstrap below and then return to this check:

```bash
test -x "$SST_CORE_PREFIX/bin/sst-config"
test -x "$SST_CORE_PREFIX/bin/sst"
test -d "$SST_DRAMSIM3_PREFIX"
test -f "$DRAMSIM3_LIB_DIR/libdramsim3.so"
command -v git autoconf automake libtoolize cmake rsync make gcc g++ python3
command -v mpicc mpic++ mpirun
command -v riscv64-linux-musl-gcc riscv64-linux-musl-g++
grep -q '^#define SST_CONFIG_HAVE_MPI 1' \
  "$SST_CORE_PREFIX/include/sst/core/sst_config.h"
```

If a check fails, fix the dependency or its `PATH` entry before continuing.
CrossSim is optional for the default Golem tests; only CrossSim-backed analog
experiments require `python3 -c 'import simulator'` to succeed.

If SST Core and DRAMSim3 are not already installed, the following is a minimal
bootstrap using the validated releases. It keeps dependencies outside this
repository:

```bash
export DEPS_ROOT="$(cd "$REPO_ROOT/.." && pwd -P)/golem-deps"
mkdir -p "$DEPS_ROOT"

git clone --branch v15.1.0_Final --single-branch \
  https://github.com/sstsimulator/sst-core.git "$DEPS_ROOT/sst-core-src"
(cd "$DEPS_ROOT/sst-core-src" && ./autogen.sh)
mkdir -p "$DEPS_ROOT/sst-core-build"
(cd "$DEPS_ROOT/sst-core-build" && \
  MPICC="$(command -v mpicc)" MPICXX="$(command -v mpic++)" \
  "$DEPS_ROOT/sst-core-src/configure" \
    --prefix="$DEPS_ROOT/sst-core-install" && \
  make -j"$(nproc)" && make install)

git clone --branch 1.0.0 --single-branch \
  https://github.com/umd-memsys/DRAMsim3.git "$DEPS_ROOT/DRAMsim3"
cmake -S "$DEPS_ROOT/DRAMsim3" -B "$DEPS_ROOT/DRAMsim3/build"
cmake --build "$DEPS_ROOT/DRAMsim3/build" -j "$(nproc)"

export SST_CORE_PREFIX="$DEPS_ROOT/sst-core-install"
export SST_DRAMSIM3_PREFIX="$DEPS_ROOT/DRAMsim3"
export DRAMSIM3_LIB_DIR="$SST_DRAMSIM3_PREFIX"
export PATH="$SST_CORE_PREFIX/bin:$RISCV_TOOLCHAIN_BIN:$PATH"
```

This bootstrap still assumes `mpicc`, `mpic++`, CMake, Autotools, a C++20-capable
host compiler, and the RISC-V musl compiler are installed by the host environment.
Ramulator2 is built privately with C++20 and fixed dependency versions:

```bash
scripts/build_private_ramulator2.sh
export SST_RAMULATOR2_PREFIX="$REPO_ROOT/deps/ramulator2-2.1"
```

### Toolchain roles

There are two different C++ toolchains in this workflow:

```text
SST/Golem host libraries: gcc/g++ (C++17; memHierarchy uses C++20 with Ramulator2)
RISC-V worker binaries:  riscv64-linux-musl-g++, used by the worker test Makefile
```

The worker Makefile uses `ARCH=riscv64` and resolves the compiler names as
`riscv64-linux-musl-g++` and `riscv64-linux-musl-gcc` from `PATH`. Build the
worker test binary from the copied build tree with:

```bash
scripts/prepare_local_build.sh
make -C "$REPO_ROOT/build/sst-elements/src/sst/elements/golem/tests/small/mvm_noc_int_array" \
  ARCH=riscv64
```

The RISC-V cross compiler is not used to build the SST/Golem host shared
library. Conversely, the host `g++` is not suitable for the RISC-V worker
binary.

### Build and install

Run the worktree-local build/install script after exporting the variables above:

```bash
cd "$REPO_ROOT"
scripts/build_and_install_local.sh --jobs "$(nproc)"
```

The script copies source files from `src/sst/elements` into
`build/sst-elements/src/sst/elements`, builds incrementally, and installs the
result into `install/`.

The source tree is authoritative. Do not edit files under `build/`; rerun the
script after changing source files so the build copy is refreshed. For a
source/config/script-only update that does not require compilation, refresh the
copy with:

```bash
cd "$REPO_ROOT"
scripts/prepare_local_build.sh
```

Use a clean rebuild only when configure state or generated files are stale:

```bash
cd "$REPO_ROOT"
scripts/build_and_install_local.sh --clean --reconfigure --jobs "$(nproc)"
```

`--clean` removes the worktree-local `build/sst-elements` and `install`
directories. It does not remove the source tree.

### Clean checkout and Ramulator2.1 build

For the reproducible TileMC/Ramulator2.1 configuration on the current integrated
branch, keep generated build and result files outside the tracked source files.
The following sequence builds Ramulator2.1 at the pinned commit, builds the SST
elements against that private library, and runs a timing smoke test:

```bash
git clone --branch softmax-update --single-branch \
  https://github.com/G-BALE-JJ/Muticore-Attention.git Muticore-Attention
cd Muticore-Attention

export REPO_ROOT="$PWD"
export DEPS_ROOT="$REPO_ROOT/deps"
export SST_CORE_PREFIX=/absolute/path/to/sst-core-install
export SST_DRAMSIM3_PREFIX=/absolute/path/to/DRAMsim3
export SST_RAMULATOR2_PREFIX="$DEPS_ROOT/ramulator2-2.1-72427a1b"
export DRAMSIM3_LIB_DIR="$SST_DRAMSIM3_PREFIX"

scripts/build_private_ramulator2.sh
scripts/build_and_install_local.sh --jobs "$(nproc)"
source scripts/env_local_install.sh

cd build/sst-elements/src/sst/elements/golem/tests
./run_noc_dma_pipeline.sh --dim s2048 --sim-mode full-timing \
  --mpi-ranks 4 --memory-backend ramulator2
```

The three dependency prefixes must be set explicitly for each host. Do not
install Ramulator2 into a shared mutable system prefix or overwrite another
worktree's `libramulator.so`. A shared Ramulator2 directory is safe only when
it is read-only and pinned to the same commit. The run wrapper invokes
`mpirun` internally; do not wrap the wrapper itself in another `mpirun`.

Before accepting a build, run the backend microbenchmark and deterministic
gate:

```bash
src/sst/elements/golem/tests/microbench/run_ramulator2_hbm_microbench.sh
src/sst/elements/golem/tests/run_parallel_determinism.sh \
  --ranks 1,4,16 --explicit-partition
```

### Run Golem tests

Load the worktree-local SST elements and run from the generated build copy:

```bash
cd "$REPO_ROOT"
source scripts/env_local_install.sh
cd build/sst-elements/src/sst/elements/golem/tests
./run_noc_dma_pipeline.sh --help
```

The normal MPI smoke/regression command is:

```bash
./run_noc_dma_pipeline.sh --mpi-ranks 4
```

The default timing backend is Ramulator2. Use
`--memory-backend dramsim3` only for an explicit DRAMSim3 comparison.

Select the timing backend explicitly when comparing memory models. Node 0–4
all use the selected library; HBM utilization excludes node 0 and reports only
the four data stacks:

```bash
./run_noc_dma_pipeline.sh --dim s2048 --sim-mode full-timing --mpi-ranks 4 \
  --memory-backend ramulator2
src/sst/elements/golem/tests/microbench/run_ramulator2_hbm_microbench.sh
src/sst/elements/golem/tests/run_memory_backend_cross_validation.sh
```

For Ramulator2 runs, the terminal reports two different read-utilization
metrics. `HBM service utilization` is the primary metric: completed read bytes
divided by the four data stacks' capacity from each stack's first read arrival
through its last read completion. `HBM RD issue utilization` is a command-level
diagnostic: issued RD commands divided by the available channel-cycle slots
from first RD issue through last RD issue. Both are capacity-weighted aggregates
and are followed by the per-stack array `[node1,node2,node3,node4]`.

The project-side Ramulator2 observer also checks that every completed read/write
has exactly one RD/WR command, that row-hit/miss/conflict totals equal RD count,
that byte totals equal command count times the 32-byte transaction size, that
all eight channels per stack are present with no pending requests, and that
observed same-pseudo-channel gaps do not violate the configured `tCCD_S` or
`tCCD_L`. A failed check rejects the memory summary instead of reporting a
partial utilization number.
The observer and the SST adapter live in this repository; the pinned Ramulator2
source is not patched.

The wrapper starts `mpirun` internally. Do not invoke the wrapper itself under
`mpirun`; preparation and post-processing must run once. Run one test case at a
time because cases share generated HBM/artifact locations. Override the output
root for an independent run:

```bash
GOLEM_ARTIFACT_ROOT="$REPO_ROOT/artifacts/my-run" \
./run_noc_dma_pipeline.sh --mpi-ranks 4
```

The explicit map distributes CPUs, routers, memory controllers, and request
schedulers across `(rank, thread)` lanes. Each directory-side memory-node NIC
owns one shared DMA credit pool. Requests consume credit when admitted to the
memory hierarchy and release it when the existing memory response is created;
same-cycle requests are ordered by stable request metadata. Managers therefore
share capacity without process-global state, host locks, or fixed per-manager
shards. Run the deterministic gate before accepting a partitioning change:

```bash
./run_parallel_determinism.sh \
  --ranks 4,8 --threads 1,2 --explicit-partition --profile
```

The checker requires identical simulated time, GEMM boundaries, execution
cycles, DMA counts, and NoC stalls. `--profile` also writes SST timing JSON and
a partition-annotated graph JSON below
`artifacts/stats/parallel_determinism/`. The explicit map currently supports
arbitrary positive rank/thread counts and uses `sst.self`. The 2048 GEMM gate
must produce identical values across the matrix, including simulated time,
execution cycles, GEMM boundaries, DMA counts, and NoC stalls.
Each normal pipeline run also writes `credit_owner_summary.csv` and
`credit_owner_table.csv`; a completed run must have admitted equal to released,
available equal to cap, and zero pending requests for every memory node.

For experiment throughput on a 16-physical-core host, run four independent
four-rank jobs on disjoint CPU sets:

```bash
./run_parallel_batch.sh
./run_parallel_batch.sh --slots 2 \
  --case-file configs/parallel_cases.example.tsv
```

The batch wrapper is a dynamic queue: each completed case immediately releases
its fixed physical-CPU slot to the next manifest row. It creates private
HBM/statistics directories and writes both `case_status.tsv` and an aggregate
`run_summary.csv`. Manifest fields are literal tab-separated arguments and are
never evaluated as shell. Queue cases must share the prepared binary/HBM
contract; build-affecting differences need separate prepared batches.

To compare physical cores with SMT siblings under the same 8-rank, 2-thread SST
graph, run:

```bash
./run_cpu_binding_benchmark.sh --ranks 8 --threads 2 --repeats 3
```

The benchmark discovers sibling topology with `lscpu`, applies a whole-job
cpuset, verifies 23 deterministic fields, and reports wall-time throughput in
`artifacts/stats/cpu_binding/`. On the validated 16-core/32-thread host, 16
distinct physical cores averaged 62.67 seconds while 16 SMT threads on eight
cores averaged 94.33 seconds. Use physical cores for one latency-sensitive run;
use the remaining SMT siblings for additional independent jobs only after all
physical cores are occupied.

After changing source files, rerun `scripts/prepare_local_build.sh` (or the
build script) before executing a test from `build/`; otherwise the test may run
an older copied snapshot. Results are written below the selected artifact root,
including logs, HBM files, MPI-rank statistics, and CSV summaries.

### Source of truth and portability

The tracked source is under `src/`. `build/`, `install/`, and `artifacts/` are
machine-local generated directories and should not be copied between hosts.
Historical notes under `src/sst/elements/golem/tests/doc/` may contain absolute
paths from earlier experiments; those paths are archival examples, not build
requirements. On a new host, use the environment variables and relative paths
in this README instead.
