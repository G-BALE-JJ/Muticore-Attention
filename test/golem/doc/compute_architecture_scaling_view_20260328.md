# GOLEM Compute Architecture View for Scaling Analysis

## Purpose

This document describes the current GOLEM default architecture from a computer-architecture perspective for compute-capacity analysis and forward-looking scaling studies. The emphasis is on module function, hierarchy, and connectivity, so the model can be compared against GPUs or other advanced throughput architectures. It intentionally avoids low-level DMA protocol details.

Scope and default reference:

- Entry flow: `tests/run_noc_dma_pipeline.sh`
- Architecture script: `tests/architecture/ncores_selfcom_dma_ctrl.py`
- CPU builder: `tests/architecture/cpu_builder.py`
- Default config chain: `tests/configs/default.env`

## Default System Instance

Under the current default configuration, the instantiated system is:

- `20` total CPU cores
- `20` total GEMM-capable cores
- `4` logical groups
- `4` compute arrays per core complex
- `5` memory nodes total
  - `1` OS / system memory node
  - `4` data memory nodes
- `4 x 7` mesh NoC
  - `4` columns
  - `5` CPU rows
  - `1` top memory row
  - `1` bottom OS row

Relevant defaults:

- `GOLEM_TOTAL_CORES=20` in `tests/configs/10_core_gemm.env`
- `GOLEM_TOTAL_GROUPS=4` in `tests/configs/10_core_gemm.env`
- `GOLEM_NUM_MEMORY_NODES=5` in `tests/configs/30_network.env`
- `GOLEM_MESH_DIM_X=4` in `tests/configs/30_network.env`
- `GOLEM_CTRL_LINK_ENABLE=1` in `tests/configs/60_run.env`
- `GOLEM_GROUP_MANAGER_ENABLE=1` in `tests/configs/60_run.env`

## Architectural Intent

The architecture is best understood as a throughput-oriented tiled matmul machine built from many lightweight core complexes connected by a packet NoC and backed by distributed memory nodes. Each core complex combines:

- a general-purpose scalar control core,
- a cache hierarchy,
- a RoCC-attached matrix/vector accelerator,
- a private software-visible scratch/global-memory window,
- and access to a distributed shared memory fabric.

This places the design between a conventional manycore CPU and a GPU-like tiled accelerator:

- more distributed and software-explicit than a monolithic GPU SM cluster,
- but more accelerator-centric than a pure cache-coherent CPU cluster.

## Top-Level Module Graph

The main connectivity is:

`Process -> Vanadis CPU -> L1/L2 cache hierarchy -> NoC`

In parallel, each core also has:

`Vanadis CPU -> RoCC -> Compute Array`

and:

`RoCC -> GlobalMemory -> NoC`

System-side services are:

- `NodeOS + MMU + OS L1 -> NoC`
- `DirectoryController + MemController + DRAMSim3 -> NoC`

So the machine is not a single shared-bus accelerator. It is a distributed compute fabric where control, cache-coherent traffic, accelerator-local storage traffic, and memory traffic meet at the mesh network.

## Core Complex

Each of the 20 cores is a complete compute complex.

### 1. Scalar Control Core

Each core uses a `vanadis` RISC-V CPU model. Its role is:

- instruction sequencing,
- loop/control execution,
- issuing accelerator commands,
- coordinating data movement and synchronization,
- running one process per core.

Architecturally, this acts like the scalar front-end of a throughput processor.

### 2. Cache Hierarchy

Per core:

- L1D: `32KB`, `8-way`, `2 cycles`
- L1I: `32KB`, `8-way`, `2 cycles`
- L2: `1MB`, `16-way`, `14 cycles`

The cache hierarchy supports the scalar CPU path and provides conventional memory-system behavior for code, control data, and coherence-managed traffic.

The L2 cache connects to the NoC through a `MemNIC`.

This makes the L2 the coherent memory gateway of each core complex.

### 3. RoCC Accelerator Front-End

Each core instantiates a RoCC subcomponent, which is the architectural bridge between the scalar core and the matrix accelerator backend.

Its role is to:

- receive accelerator instructions from the CPU,
- interpret the command stream,
- configure and trigger array execution,
- connect accelerator-side storage/control to the rest of the system.

In architectural terms, RoCC behaves like a per-core accelerator command processor.

### 4. Compute Array

Each RoCC complex attaches one compute array backend.

With current defaults:

- array dimension: `64`
- array input size: `64`
- array output size: `64`
- dtype path default: `fp32`

This means the natural compute tile is centered around a `64 x 64` inner product structure, with blocked GEMM shapes chosen to match the array dimensions.

Default blocked GEMM shape:

- block `M=64`
- block `N=4`
- block `K=64`

So each active worker computes a small output tile whose inner accumulation dimension is matched to the array width.

For scaling analysis, the compute array is the main dense-math engine and should be treated as the analogue of a GPU tensor-core or matrix-core block.

### 5. Per-Core GlobalMemory Window

Each core has a `GlobalMemory` subcomponent with a private address window.

Default parameters:

- per-core window size: `256KB`
- base for core `i`: `GLOBAL_BASE + i * GLOBAL_STRIDE`

Architecturally, this is a software-addressable local storage region associated with each accelerator complex. For scaling studies, it is best viewed as a scratchpad-like local operand/control store rather than a cache.

Its role in the architecture is to:

- hold per-core working sets,
- stage accelerator-visible operands/results,
- provide a private memory namespace for each core complex,
- act as the software-managed local memory plane beside the cache hierarchy.

That makes the design closer to scratchpad-based accelerators than to pure cache-only systems.

## Grouped Execution Organization

The 20 cores are organized into `4` logical groups.

Under the default mapping:

- cores are distributed across `4` mesh columns,
- each group aligns naturally with one mesh column,
- each group contains one manager-like control role plus worker roles.

For compute analysis, the important property is not the exact runtime policy, but the structural one:

- the machine already has a hierarchical execution model,
- work is not launched as 20 fully independent peers,
- instead, the design groups cores into local compute domains.

This is architecturally similar to the way GPUs organize lanes into warps and warps into SM-local scheduling domains, although here the grouping is exposed more explicitly in software/control structure.

## On-Chip Network

### Topology

The NoC is a Merlin mesh built from `hr_router` routers.

Default topology:

- mesh size: `4 x 7`
- total routers: `28`
- local ports per router: `3`

Placement:

- top row: data memory nodes
- middle rows: CPU/core complexes
- bottom row: OS/system node

This gives the architecture a clear spatial layout in which compute occupies the center, bulk data memory sits on one edge, and system services sit on another edge.

### Link Parameters

Default NoC parameters:

- link bandwidth: `100GB/s`
- crossbar bandwidth: `100GB/s`
- flit size: `128B`
- input buffer: `64KB`
- output buffer: `64KB`
- virtual networks: `3`

For scaling studies, this NoC should be treated as the global on-chip transport fabric between:

- coherent cache endpoints,
- accelerator-local memory windows,
- system software services,
- distributed DRAM directories/controllers.

### Architectural Role

The NoC is not just a memory interconnect. It is the integrating backbone of the entire machine. Any future scaling in core count, array count, memory-node count, or memory bandwidth must therefore be evaluated against:

- bisection bandwidth,
- per-column concentration,
- router local-port pressure,
- and endpoint injection/ejection balance.

## System Software and MMU Plane

The design includes a `VanadisNodeOS` instance and MMU subsystem.

System-side components:

- NodeOS
- simpleMMU
- OS-side L1 cache

The OS cache also connects into the NoC through a `MemNIC`.

Architecturally, this forms the control and virtual-memory service plane of the machine. It makes the platform more like a full heterogeneous computer than a bare accelerator array.

For throughput comparisons with GPUs, this is one of the important differences:

- GPU-like machines often centralize or hide much of this software service path,
- while this architecture keeps a more explicit CPU/OS/VM execution model in the loop.

## Memory Subsystem

### Memory Nodes

There are `5` memory nodes total.

- node 0: OS/system memory role
- nodes 1-4: data memory roles

Each node owns a contiguous physical address range.

Default per-node capacity:

- `128MiB` each

### Per-Node Structure

Each memory node is built from:

- `DirectoryController`
- `MemController`
- `DRAMSim3` backend

This means the memory system is distributed and coherence-aware, not a single shared memory controller.

### Memory-Side NoC Attachment

Each directory controller attaches to the mesh through a `MemNIC` highlink.

Default memory-side network bandwidth:

- directory highlink: `25GB/s`

This is important for scaling because the compute side injects into a higher-bandwidth fabric, but the memory-side service point is narrower. Even when total NoC bandwidth is high, memory-node ingress/egress bandwidth remains a first-order architectural limit.

## Connection Summary by Module

### CPU Path

- `CPU -> L1I`
- `CPU -> LSQ -> L1D`
- `L1I/L1D -> processor bus -> L2`
- `L2 -> MemNIC -> NoC`

Functionally, this is the conventional scalar/coherent path.

### Accelerator Path

- `CPU -> RoCC`
- `RoCC -> Compute Array`
- `RoCC -> GlobalMemory`
- `GlobalMemory -> SimpleNetwork -> NoC`

Functionally, this is the throughput-acceleration path.

### System Path

- `CPU cores -> NodeOS`
- `NodeOS -> MMU`
- `NodeOS -> OS L1 -> NoC`

Functionally, this is the software and address-translation support path.

### Memory Path

- `NoC -> DirectoryController`
- `DirectoryController -> MemController`
- `MemController -> DRAMSim3 backend`

Functionally, this is the distributed memory service path.

## How To Read This Architecture Against GPU-Like Designs

For forward-looking compute scaling, the cleanest mapping is:

- `Vanadis CPU` ~ scalar/control front-end
- `RoCC` ~ per-core accelerator command processor
- `Compute Array` ~ tensor-core / matrix-core equivalent
- `GlobalMemory window` ~ local scratchpad / software-managed shared memory analogue
- `L2 + coherent path` ~ cache-backed general data path
- `Mesh NoC` ~ on-chip fabric / cross-SM interconnect
- `Directory + DRAM nodes` ~ distributed memory partitions

The main architectural differences from a modern GPU are:

- this machine keeps stronger CPU-style control and OS visibility,
- it distributes accelerator front-ends per core instead of concentrating many lanes inside an SM,
- and it exposes a more explicit local-memory plus networked-memory model.

The main architectural similarities are:

- tiled matrix compute,
- distributed memory partitions,
- dependence on on-chip network scaling,
- and the need to balance local storage, compute density, and memory-node bandwidth.

## Key Quantities for Future Compute Scaling

If the goal is to match or approach GPU-class throughput, the most important architectural scaling variables are:

- number of active compute arrays,
- array dimension and datatype support,
- sustained operations per array per cycle,
- local storage capacity per compute complex,
- number of cores per group and groups per chip,
- NoC bisection bandwidth,
- memory-node count,
- memory-node bandwidth at the network attachment point,
- and the ratio of scalar-control resources to accelerator resources.

In other words, future compute scaling will not be governed by the matrix array alone. It will be governed by whether the full architecture can scale as a balanced machine:

- enough array throughput,
- enough local operand capacity,
- enough network transport,
- and enough distributed memory service bandwidth.

## Recommended Use of This Document

This description should be used as the architectural baseline for:

- peak-throughput modeling,
- roofline-style analysis,
- array-count expansion studies,
- memory-node scaling studies,
- NoC dimensioning studies,
- and comparisons against GPU, TPU, or AI-accelerator organizations.

For lower-level dataflow, runtime scheduling, or detailed transfer behavior, use separate mechanism-focused documents rather than overloading this architecture summary.
