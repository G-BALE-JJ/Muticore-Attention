#!/usr/bin/env python3

import os
import re
import sys
import sst

if __package__ in {None, ""}:
    _tests_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _tests_dir not in sys.path:
        sys.path.insert(0, _tests_dir)

from architecture.noc_builder import MeshNoCBuilder
from architecture.partition_planner import (
    build_weighted_topology_plan,
    write_partition_audit,
)
from architecture.cpu_builder import (
    CPU_Builder,
    TESTS_DIR,
    cpu_clock,
    protocol,
    mh_debug,
    mh_debug_level,
    numCpus,
    numThreads,
)


def _parse_size_to_int(size_str: str) -> int:
    size_str = size_str.strip()
    if not size_str:
        raise ValueError("Empty size string")
    try:
        return int(size_str, 0)
    except (ValueError, TypeError):
        pass
    match = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*([A-Za-z]+)", size_str)
    if not match:
        raise ValueError(f"Unsupported size format: {size_str}")
    value = float(match.group(1))
    unit = match.group(2).upper()
    factors = {
        "B": 1,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "TB": 1000**4,
        "KIB": 1024,
        "MIB": 1024**2,
        "GIB": 1024**3,
        "TIB": 1024**4,
        "K": 1000,
        "M": 1000**2,
        "G": 1000**3,
        "T": 1000**4,
        "KI": 1024,
        "MI": 1024**2,
        "GI": 1024**3,
        "TI": 1024**4,
    }
    if unit not in factors:
        raise ValueError(f"Unsupported size unit: {unit}")
    return int(value * factors[unit])


def addParamsPrefix(prefix, params):
    out = {}
    for key, value in params.items():
        out[prefix + "." + key] = value
    return out


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


ARTIFACT_ROOT = os.getenv("GOLEM_ARTIFACT_ROOT", os.path.join(TESTS_DIR, "artifacts"))
HBM_DIR = os.getenv("GOLEM_HBM_DIR", os.path.join(ARTIFACT_ROOT, "hbm"))
HBM_DUMP_OUTPUT = _env_flag("GOLEM_HBM_DUMP_OUTPUT", True)
STATS_DIR = os.getenv("GOLEM_STATS_DIR", os.path.join(ARTIFACT_ROOT, "stats"))
STATS_FILE = os.getenv(
    "GOLEM_STATS_FILE", os.path.join(STATS_DIR, "stats_selfcom_ctrl.txt")
)
DRAMSIM3_OUT_DIR = os.getenv("GOLEM_DRAMSIM3_OUT_DIR", os.path.join(STATS_DIR, "dramsim3"))
MPI_PARTITIONING = _env_flag("GOLEM_MPI_PARTITIONING", False)
EXPLICIT_PARTITION = _env_flag("GOLEM_EXPLICIT_PARTITION", False)
MPI_RANKS = int(os.getenv("GOLEM_MPI_RANKS", "1"))
SST_THREADS = int(os.getenv("GOLEM_SST_THREADS", "1"))
PARTITION_STRATEGY = os.getenv(
    "GOLEM_PARTITION_STRATEGY", "weighted_topology"
).strip().lower()
PARTITION_WEIGHT_PROFILE = os.getenv("GOLEM_PARTITION_WEIGHT_PROFILE", "").strip()
PARTITION_AUDIT_FILE = os.getenv(
    "GOLEM_PARTITION_AUDIT_FILE", os.path.join(STATS_DIR, "partition_plan.csv")
)
if MPI_RANKS <= 0 or SST_THREADS <= 0:
    raise ValueError("GOLEM_MPI_RANKS and GOLEM_SST_THREADS must be positive")
if PARTITION_STRATEGY not in {"round_robin", "weighted_topology"}:
    raise ValueError(f"Unsupported GOLEM_PARTITION_STRATEGY={PARTITION_STRATEGY}")
DRAMSIM3_CONFIG = os.getenv(
    "GOLEM_DRAMSIM3_CONFIG",
    os.path.join(TESTS_DIR, "architecture", "dram", "HBM_4Gb_x128.ini"),
)
MEMORY_BACKEND = os.getenv("GOLEM_MEMORY_BACKEND", "dramsim3").strip().lower()
if MEMORY_BACKEND not in {"dramsim3", "ramulator2"}:
    raise ValueError(
        f"Unsupported GOLEM_MEMORY_BACKEND={MEMORY_BACKEND}; expected dramsim3 or ramulator2"
    )
RAMULATOR2_CONFIG = os.getenv(
    "GOLEM_RAMULATOR2_CONFIG",
    os.path.join(TESTS_DIR, "architecture", "ramulator", "hbm2e_2500.yaml"),
)
memctrl_clock = os.getenv("GOLEM_MEMCTRL_CLOCK", cpu_clock)

os.makedirs(HBM_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)

NUM_MEMORY_NODES = int(os.getenv("GOLEM_NUM_MEMORY_NODES", "5"))
OS_MEMORY_NODE_INDEX = 0
default_node_size_bytes = int(os.getenv("GOLEM_MEM_NODE_SIZE_BYTES", str(64 * 1024**2)))
default_phys_mem_size = (
    f"{(default_node_size_bytes * NUM_MEMORY_NODES) // (1024**2)}MiB"
)
physMemSize = os.getenv("GOLEM_PHYS_MEM_SIZE", default_phys_mem_size)
physMemBytes = _parse_size_to_int(physMemSize)
memBytesPerNode = physMemBytes // NUM_MEMORY_NODES
memSizePerNode = f"{memBytesPerNode // (1024**2)}MiB"
os.environ["GOLEM_MEM_NODE_SIZE"] = f"{memBytesPerNode}B"

sst.setProgramOption("timebase", "1ps")
sst.setStatisticOutput("sst.statOutputCSV", {"filepath": STATS_FILE, "separator": ","})
sst.setStatisticLoadLevel(int(os.getenv("GOLEM_SST_STAT_LOAD_LEVEL", "16")))
if int(os.getenv("GOLEM_SST_ENABLE_ALL_STATS", "1")) != 0:
    sst.enableAllStatisticsForAllComponents({"type": "sst.AccumulatorStatistic"})

MESH_DIM_X = int(os.getenv("GOLEM_MESH_DIM_X", "4"))
if EXPLICIT_PARTITION and MESH_DIM_X != 4:
    raise ValueError("GOLEM_EXPLICIT_PARTITION currently requires GOLEM_MESH_DIM_X=4")
cpu_rows = (numCpus + MESH_DIM_X - 1) // MESH_DIM_X
DATA_MEMORY_NODE_COUNT = NUM_MEMORY_NODES - 1
MEMORY_LAYOUT = os.getenv("GOLEM_MEMORY_LAYOUT", "top_hbm").strip().lower()
if MEMORY_LAYOUT not in {"top_hbm", "bottom_hbm"}:
    raise ValueError(f"Unsupported GOLEM_MEMORY_LAYOUT={MEMORY_LAYOUT}")

if MEMORY_LAYOUT == "top_hbm":
    DATA_MEMORY_ROW_INDEX = 0
    CPU_ROW_START = 1
    OS_MEMORY_ROW_INDEX = cpu_rows + 1
else:
    DATA_MEMORY_ROW_INDEX = cpu_rows
    CPU_ROW_START = 0
    OS_MEMORY_ROW_INDEX = cpu_rows + 1

MESH_DIM_Y = cpu_rows + 2
LOCAL_PORTS = 3


def _evenly_spaced_columns(num_cols: int, count: int):
    if count <= 0:
        return []
    if count == 1:
        return [0]
    return [int(round(i * (num_cols - 1) / (count - 1))) for i in range(count)]


osParams = {
    "processDebugLevel": 0,
    "dbgLevel": os.getenv("VANADIS_OS_VERBOSE", os.getenv("VANADIS_VERBOSE", 0)),
    "dbgMask": 8,
    "cores": numCpus,
    "hardwareThreadCount": numThreads,
    "page_size": 4096,
    "physMemSize": physMemSize,
    "useMMU": True,
    "checkpointDir": "",
    "checkpoint": "",
}

full_exe_name = os.getenv(
    "VANADIS_EXE",
    os.path.join(TESTS_DIR, "small", "mvm_noc_int_array", "riscv64", "test_noc_dma"),
)
exe_name = full_exe_name.split("/")[-1]
process_env_keys = [
    "GOLEM_MATMUL_M",
    "GOLEM_MATMUL_N",
    "GOLEM_MATMUL_K",
    "GOLEM_MATMUL_BLOCK_M",
    "GOLEM_MATMUL_BLOCK_N",
    "GOLEM_MATMUL_BLOCK_K",
    "GOLEM_MATMUL_DTYPE",
    "GOLEM_MATMUL_LAYOUT",
    "GOLEM_MATMUL_TRANSPOSE_A",
    "GOLEM_MATMUL_TRANSPOSE_B",
    "GOLEM_CTRL_LINK_ENABLE",
    "GOLEM_SFU_INTERLEAVE_GEMM",
    "GOLEM_SFU_STANDALONE_SOFTMAX",
    "GOLEM_SFU_JOB_SOFTMAX",
    "GOLEM_SFU_JOB_SOFTMAX_CHUNK_ELEMS",
    "GOLEM_SFU_JOB_SOFTMAX_WORKER_CORES",
    "GOLEM_SFU_JOB_SOFTMAX_STAGING_ROWS",
    "GOLEM_SFU_JOB_SOFTMAX_TRACE_BANDS",
    "GOLEM_SFU_JOB_SOFTMAX_BAND_CORES",
    "GOLEM_SFU_JOB_SOFTMAX_JOB_ROWS",
    "GOLEM_SFU_JOB_SOFTMAX_DIRECT_ROWMAJOR_HBM",
    "GOLEM_SFU_JOB_SOFTMAX_DISTRIBUTED_COLUMNS",
    "GOLEM_SFU_PRIMITIVE_SOFTMAX",
    "GOLEM_SFU_PRIMITIVE_SOFTMAX_ROWS",
    "GOLEM_SFU_PRIMITIVE_SOFTMAX_DIM",
    "GOLEM_SFU_PRIMITIVE_SOFTMAX_CHUNK_ELEMS",
    "GOLEM_SFU_PRIMITIVE_SOFTMAX_WORKER_CORES",
    "GOLEM_SFU_PRIMITIVE_SOFTMAX_MULTICORE_MIN_DIM",
    "GOLEM_SFU_PRIMITIVE_SOFTMAX_VERIFY",
    "GOLEM_SFU_PRIMITIVE_SOFTMAX_ROW_BLOCK",
    "GOLEM_SFU_PRIMITIVE_SOFTMAX_PIPELINE_DEPTH",
    "GOLEM_SFU_PRIMITIVE_SOFTMAX_NBPOLL",
    "GOLEM_SFU_PRIMITIVE_HBM_STREAM",
    "GOLEM_SFU_PRIMITIVE_HBM_ELEMS",
    "GOLEM_SFU_PRIMITIVE_HBM_CHUNK_ELEMS",
    "GOLEM_SFU_PRIMITIVE_HBM_OPS",
    "GOLEM_SFU_PRIMITIVE_HBM_BATCH",
    "GOLEM_SFU_PRIMITIVE_SMOKE",
    "GOLEM_SFU_PRIMITIVE_SMOKE_ELEMS",
    "GOLEM_SFU_PRIMITIVE_SMOKE_CHUNK_ELEMS",
    "GOLEM_STAGE_PROGRESS",
    "GOLEM_RUNTIME_SILENT",
    "GOLEM_SILENT",
]
process_env_entries = [
    f"{key}={os.environ[key]}" for key in process_env_keys if key in os.environ
]
processList = []
for core_id in range(numCpus):
    process_params = {
        "env_count": len(process_env_entries),
        "exe": full_exe_name,
        "arg0": exe_name,
        "arg1": str(core_id),
        "argc": 2,
    }
    for idx, env_entry in enumerate(process_env_entries):
        process_params[f"env{idx}"] = env_entry
    processList.append((1, process_params))

osl1cacheParams = {
    "access_latency_cycles": "2",
    "cache_frequency": cpu_clock,
    "replacement_policy": "lru",
    "coherence_protocol": protocol,
    "cache_type": "inclusive",
    "associativity": "8",
    "cache_line_size": "64",
    "cache_size": "32 KB",
    "L1": "1",
    "debug": mh_debug,
    "debug_level": mh_debug_level,
}

mmuType = "simpleMMU"
mmuParams = {
    "debug_level": 0,
    "num_cores": numCpus,
    "num_threads": numThreads,
    "page_size": 4096,
}

noc_inter_router_no_cut = _env_flag("GOLEM_NOC_INTER_ROUTER_NO_CUT", False)
noc_local_no_cut = _env_flag("GOLEM_NOC_LOCAL_NO_CUT", False)
noc_memory_local_no_cut = _env_flag("GOLEM_NOC_MEMORY_LOCAL_NO_CUT", False)
noc_scheduler_local_no_cut = _env_flag("GOLEM_NOC_SCHED_LOCAL_NO_CUT", False)
noc_vn_priority_enable = _env_flag("GOLEM_NOC_VN_PRIORITY_ENABLE", True)
noc_vn_priority_order = (
    os.getenv("GOLEM_NOC_VN_PRIORITY_ORDER", "1,0,2")
    if noc_vn_priority_enable
    else ""
)
noc_vn_starvation_vn = int(os.getenv("GOLEM_NOC_VN_STARVATION_VN", "2"))
noc_vn_max_starvation_cycles = int(
    os.getenv("GOLEM_NOC_VN_MAX_STARVATION_CYCLES", "4096")
)
if EXPLICIT_PARTITION:
    noc_inter_router_no_cut = False
print(
    f"[NoC] inter_router_no_cut={int(noc_inter_router_no_cut)}, "
    f"local_no_cut={int(noc_local_no_cut)}, "
    f"memory_local_no_cut={int(noc_memory_local_no_cut)}, "
    f"scheduler_local_no_cut={int(noc_scheduler_local_no_cut)}"
)
print(
    f"[MPI] partitioning={int(MPI_PARTITIONING)} "
    f"explicit_partition={int(EXPLICIT_PARTITION)} ranks={MPI_RANKS} "
    f"threads={SST_THREADS} "
    f"strategy={PARTITION_STRATEGY} "
    f"dramsim3_out_dir={DRAMSIM3_OUT_DIR}"
)
print(f"[MPI] partitioning={int(MPI_PARTITIONING)} dramsim3_out_dir={DRAMSIM3_OUT_DIR}")

noc = MeshNoCBuilder(
    dim_x=MESH_DIM_X,
    dim_y=MESH_DIM_Y,
    local_ports=LOCAL_PORTS,
    link_bw=os.getenv("GOLEM_NOC_LINK_BW", "25GB/s"),
    xbar_bw=os.getenv("GOLEM_NOC_XBAR_BW", "25GB/s"),
    flit_size=os.getenv("GOLEM_NOC_FLIT_SIZE", "128B"),
    directional_link_latency="1ns",
    local_link_latency="1ns",
    input_buf_size=os.getenv("GOLEM_NOC_INPUT_BUF_SIZE", "8KB"),
    output_buf_size=os.getenv("GOLEM_NOC_OUTPUT_BUF_SIZE", "8KB"),
    num_vns=3,
    vn_priority_order=noc_vn_priority_order,
    vn_starvation_vn=noc_vn_starvation_vn,
    vn_max_starvation_cycles=noc_vn_max_starvation_cycles,
    inter_router_no_cut=noc_inter_router_no_cut,
    local_no_cut=noc_local_no_cut,
    debug=1,
)
noc.build()

data_row_start = DATA_MEMORY_ROW_INDEX * MESH_DIM_X
os_row_start = OS_MEMORY_ROW_INDEX * MESH_DIM_X
if DATA_MEMORY_NODE_COUNT > MESH_DIM_X:
    raise ValueError("DATA_MEMORY_NODE_COUNT exceeds mesh columns")
data_memory_columns = _evenly_spaced_columns(MESH_DIM_X, DATA_MEMORY_NODE_COUNT)
DATA_MEMORY_ROUTERS = [data_row_start + col for col in data_memory_columns]
OS_ROUTER = os_row_start
MEMORY_ROUTERS = [OS_ROUTER] + DATA_MEMORY_ROUTERS
os.environ["GOLEM_MEMORY_ROUTERS"] = ",".join(str(r) for r in MEMORY_ROUTERS)

cpu_routers = []
for row in range(CPU_ROW_START, CPU_ROW_START + cpu_rows):
    for col in range(MESH_DIM_X):
        router_id = row * MESH_DIM_X + col
        if router_id in MEMORY_ROUTERS:
            continue
        cpu_routers.append(router_id)
        if len(cpu_routers) == numCpus:
            break
    if len(cpu_routers) == numCpus:
        break

if len(cpu_routers) != numCpus:
    raise RuntimeError(
        f"insufficient CPU routers: need {numCpus}, got {len(cpu_routers)}"
    )


def _round_robin_placement(index: int):
    if not EXPLICIT_PARTITION or MPI_RANKS == 1:
        return 0, index % SST_THREADS
    return index % MPI_RANKS, (index // MPI_RANKS) % SST_THREADS


router_to_core = {router_id: core_id for core_id, router_id in enumerate(cpu_routers)}
if EXPLICIT_PARTITION and PARTITION_STRATEGY == "weighted_topology":
    manager_count = (
        min(numCpus, int(os.getenv("GOLEM_TOTAL_GROUPS", str(MESH_DIM_X))))
        if _env_flag("GOLEM_GROUP_MANAGER_ENABLE", False)
        else 0
    )
    partition_plan = build_weighted_topology_plan(
        dim_x=MESH_DIM_X,
        dim_y=MESH_DIM_Y,
        num_ranks=MPI_RANKS,
        num_threads=SST_THREADS,
        cpu_by_router=router_to_core,
        data_memory_by_router={
            router_id: node_id
            for node_id, router_id in enumerate(DATA_MEMORY_ROUTERS, start=1)
        },
        os_router=OS_ROUTER,
        manager_cores=range(manager_count),
        router_weight=float(os.getenv("GOLEM_PARTITION_ROUTER_WEIGHT", "0.1")),
        cpu_weight=float(os.getenv("GOLEM_PARTITION_CPU_WEIGHT", "1.0")),
        manager_weight=float(os.getenv("GOLEM_PARTITION_MANAGER_WEIGHT", "0.5")),
        data_memory_weight=float(
            os.getenv("GOLEM_PARTITION_DATA_MEMORY_WEIGHT", "5.0")
        ),
        os_weight=float(os.getenv("GOLEM_PARTITION_OS_WEIGHT", "0.5")),
        affinity_edge_weight=float(
            os.getenv("GOLEM_PARTITION_AFFINITY_EDGE_WEIGHT", "2.0")
        ),
        profile_path=PARTITION_WEIGHT_PROFILE or None,
    )
    router_partition_placements = list(partition_plan.placements)
    write_partition_audit(PARTITION_AUDIT_FILE, partition_plan)
    print(
        f"[MPI] partition mesh cuts rank={partition_plan.cross_rank_edges}/"
        f"{partition_plan.total_mesh_edges} thread={partition_plan.cross_thread_edges}; "
        f"affinity cuts rank={partition_plan.cross_rank_affinity_edges} "
        f"thread={partition_plan.cross_thread_affinity_edges}; "
        f"rank weights={partition_plan.rank_weights(MPI_RANKS)}; "
        f"audit={PARTITION_AUDIT_FILE}"
    )
else:
    router_partition_placements = []
    for router_id in range(noc.num_nodes):
        if router_id in router_to_core:
            placement = _round_robin_placement(router_to_core[router_id])
        elif router_id == OS_ROUTER:
            placement = (0, 0)
        else:
            placement = _round_robin_placement(router_id)
        router_partition_placements.append(placement)

cpu_partition_placements = [
    router_partition_placements[router_id] for router_id in cpu_routers
]
for router_id, placement in enumerate(router_partition_placements):
    if EXPLICIT_PARTITION:
        noc.get_router(router_id).setRank(*placement)

if EXPLICIT_PARTITION:
    lane_core_counts = {
        f"r{rank}t{thread}": cpu_partition_placements.count((rank, thread))
        for rank in range(MPI_RANKS)
        for thread in range(SST_THREADS)
    }
    lane_router_counts = {
        f"r{rank}t{thread}": router_partition_placements.count((rank, thread))
        for rank in range(MPI_RANKS)
        for thread in range(SST_THREADS)
    }
    print(
        f"[MPI] explicit core lane counts={lane_core_counts} "
        f"router lane counts={lane_router_counts}"
    )

builder = CPU_Builder()
cpu_ports = []
ctrl_eps = {}
sched_eps = {}
sched_net_eps = {}
for core_id in range(numCpus):
    ports = builder.build(
        f"core{core_id}",
        core_id,
        core_id,
        add_l2_cache=True,
        add_rocc_golem=True,
        rank=cpu_partition_placements[core_id][0] if EXPLICIT_PARTITION else None,
        thread=cpu_partition_placements[core_id][1],
    )
    cpu_ports.append(ports)
    ctrl_eps[core_id] = ports[6] if len(ports) > 6 else None
    sched_eps[core_id] = ports[7] if len(ports) > 7 else None
    sched_net_eps[core_id] = ports[5] if len(ports) > 5 else None

for core_id, router_id in enumerate(cpu_routers):
    l2_mem = cpu_ports[core_id][1]
    gm = cpu_ports[core_id][4]
    noc.attach_local(
        router_id, l2_mem, link_name=f"link_core{core_id}_l2_to_rtr{router_id}"
    )
    noc.attach_local(
        router_id, gm, link_name=f"link_core{core_id}_gm_to_rtr{router_id}"
    )
    sched_net = sched_net_eps[core_id]
    if sched_net is not None:
        noc.attach_local(
            router_id,
            sched_net,
            link_name=f"link_core{core_id}_sched_to_rtr{router_id}",
            no_cut=noc_scheduler_local_no_cut,
        )

node_os = sst.Component("os", "vanadis.VanadisNodeOS")
if EXPLICIT_PARTITION:
    node_os.setRank(*router_partition_placements[OS_ROUTER])
node_os.addParams(osParams)
num = 0
for i, process in processList:
    for _ in range(i):
        node_os.addParams(addParamsPrefix("process" + str(num), process))
        num += 1

node_os_mmu = node_os.setSubComponent("mmu", "mmu." + mmuType)
node_os_mmu.addParams(mmuParams)
node_os_mem_if = node_os.setSubComponent(
    "mem_interface", "memHierarchy.standardInterface"
)

os_l1 = sst.Component("node.os_l1cache", "memHierarchy.Cache")
if EXPLICIT_PARTITION:
    os_l1.setRank(*router_partition_placements[OS_ROUTER])
os_l1.addParams(osl1cacheParams)
os_l1_hi = os_l1.setSubComponent("highlink", "memHierarchy.MemLink")
os_l1_lo = os_l1.setSubComponent("lowlink", "memHierarchy.MemNIC")
memory_destinations = ",".join(str(100 + idx) for idx in range(NUM_MEMORY_NODES))
os_l1_lo.addParams(
    {
        "group": 1,
        "destinations": memory_destinations,
        "network_bw": "100GB/s",
        "num_vns": 3,
        "network_input_buffer_size": os.getenv(
            "GOLEM_NOC_MEMNIC_INPUT_BUF_SIZE", "128KB"
        ),
        "network_output_buffer_size": os.getenv(
            "GOLEM_NOC_MEMNIC_OUTPUT_BUF_SIZE", "128KB"
        ),
        "network_vn_priority_order": noc_vn_priority_order,
        "network_vn_starvation_vn": noc_vn_starvation_vn,
        "network_vn_max_starvation_cycles": noc_vn_max_starvation_cycles,
    }
)

link_os_to_l1 = sst.Link("link_node_os_to_l1")
link_os_to_l1.connect((node_os_mem_if, "lowlink", "1ns"), (os_l1_hi, "port", "1ns"))
link_os_to_l1.setNoCut()
noc.attach_local(
    OS_ROUTER, (os_l1_lo, "port", "1ns"), link_name=f"link_node_osl1_to_rtr{OS_ROUTER}"
)

for idx, router_id in enumerate(MEMORY_ROUTERS):
    addr_start = idx * memBytesPerNode
    addr_end = (idx + 1) * memBytesPerNode - 1

    dirctrl = sst.Component(f"dirctrl_{idx}", "memHierarchy.DirectoryController")
    if EXPLICIT_PARTITION:
        dirctrl.setRank(*router_partition_placements[router_id])
    dirctrl.addParams(
        {
            "coherence_protocol": protocol,
            "entry_cache_size": "256",
            "debug": mh_debug,
            "debug_level": mh_debug_level,
            "addr_range_start": hex(addr_start),
            "addr_range_end": hex(addr_end),
        }
    )
    dir_hi = dirctrl.setSubComponent("highlink", "memHierarchy.MemNIC")
    dir_hi.addParams(
        {
            "group": 100 + idx,
            "sources": "1",
            "network_bw": os.getenv("GOLEM_DIRCTRL_HIGHLINK_BW", "100GB/s"),
            "num_vns": 3,
            "network_input_buffer_size": os.getenv(
                "GOLEM_NOC_MEMNIC_INPUT_BUF_SIZE", "128KB"
            ),
            "network_output_buffer_size": os.getenv(
                "GOLEM_NOC_MEMNIC_OUTPUT_BUF_SIZE", "128KB"
            ),
            "network_vn_priority_order": noc_vn_priority_order,
            "network_vn_starvation_vn": noc_vn_starvation_vn,
            "network_vn_max_starvation_cycles": noc_vn_max_starvation_cycles,
            "golem_dma_write_response_vn": os.getenv("GOLEM_DMA_WRITE_VN", "2"),
            "golem_dma_response_vn": os.getenv("GOLEM_DMA_RESPONSE_VN", "1"),
            "golem_dma_response_drain_limit": os.getenv(
                "GOLEM_DMA_RESPONSE_DRAIN_LIMIT", "0"
            ),
            "golem_dma_trace": os.getenv("GOLEM_DMA_TRACE", "0"),
            "golem_dma_credit_cap": os.getenv(
                "GOLEM_DMA_NODE_CHUNK_CREDITS", "128"
            ),
            "golem_dma_credit_chunk_bytes": os.getenv(
                "GOLEM_DMA_CREDIT_CHUNK_BYTES", "16384"
            ),
            "golem_dma_admission_limit": os.getenv(
                "GOLEM_DMA_ADMISSION_LIMIT", "0"
            ),
            "golem_dma_window_priority_enable": os.getenv(
                "GOLEM_DMA_WINDOW_PRIORITY_ENABLE", "0"
            ),
            "golem_dma_window_reorder_cycles": os.getenv(
                "GOLEM_DMA_WINDOW_REORDER_CYCLES", "512"
            ),
            "golem_dma_tile_chunk_quantum": os.getenv(
                "GOLEM_DMA_TILE_CHUNK_QUANTUM", "1"
            ),
            "golem_dma_response_tile_priority_enable": os.getenv(
                "GOLEM_DMA_RESPONSE_TILE_PRIORITY_ENABLE", "0"
            ),
            "golem_dma_response_reorder_cycles": os.getenv(
                "GOLEM_DMA_RESPONSE_REORDER_CYCLES", "0"
            ),
            "golem_dma_response_max_starvation_cycles": os.getenv(
                "GOLEM_DMA_RESPONSE_MAX_STARVATION_CYCLES", "65536"
            ),
            "golem_dma_admission_max_starvation_cycles": os.getenv(
                "GOLEM_DMA_ADMISSION_MAX_STARVATION_CYCLES", "4096"
            ),
        }
    )
    dir_lo = dirctrl.setSubComponent("lowlink", "memHierarchy.MemLink")

    memctrl = sst.Component(f"memory_{idx}", "memHierarchy.MemController")
    if EXPLICIT_PARTITION:
        memctrl.setRank(*router_partition_placements[router_id])
    mem_params = {
        "clock": memctrl_clock,
        "backend.mem_size": memSizePerNode,
        "addr_range_start": addr_start,
        "addr_range_end": addr_end,
        "debug_level": mh_debug_level,
        "debug": mh_debug,
    }
    if idx == OS_MEMORY_NODE_INDEX:
        mem_params.update({"backing": "malloc", "initBacking": 0})
    else:
        hbm_init_file = os.path.join(HBM_DIR, f"hbm_init_node{idx}.bin")
        hbm_out_file = (
            os.path.join(HBM_DIR, f"hbm_out_node{idx}.bin")
            if HBM_DUMP_OUTPUT
            else hbm_init_file
        )
        mem_params.update(
            {
                "backing": "mmap",
                "initBacking": 1,
                "backing_in_file": hbm_init_file,
                "backing_out_file": hbm_out_file,
            }
        )
    memctrl.addParams(mem_params)
    mem_hi = memctrl.setSubComponent("highlink", "memHierarchy.MemLink")
    node_memory_backend = MEMORY_BACKEND
    mem_backend = memctrl.setSubComponent(
        "backend", f"memHierarchy.{node_memory_backend}"
    )
    if node_memory_backend == "dramsim3":
        backend_params = {"mem_size": memSizePerNode, "config_ini": DRAMSIM3_CONFIG}
        if MPI_PARTITIONING:
            # DRAMSim3 writes fixed filenames; isolate each memory node when ranks
            # are active so different SST ranks cannot overwrite one another.
            node_output_dir = os.path.join(DRAMSIM3_OUT_DIR, f"node{idx}")
            os.makedirs(node_output_dir, exist_ok=True)
            backend_params["output_dir"] = node_output_dir
    else:
        backend_params = {
            "mem_size": memSizePerNode,
            "configFile": RAMULATOR2_CONFIG,
            "request_width": 32,
            # MemController presents node-local addresses to its backend.
            "address_offset": 0,
            "backend_id": idx,
        }
    mem_backend.addParams(backend_params)

    link_dir_mem = sst.Link(f"link_dir{idx}_to_mem{idx}")
    link_dir_mem.connect((dir_lo, "port", "1ns"), (mem_hi, "port", "1ns"))
    link_dir_mem.setNoCut()
    noc.attach_local(
        router_id,
        (dir_hi, "port", "1ns"),
        link_name=f"link_dir{idx}_to_rtr{router_id}",
        no_cut=noc_memory_local_no_cut,
    )

for core_id, ports in enumerate(cpu_ports):
    os_link, dtlb, itlb = ports[0], ports[2], ports[3]
    link_core_os = sst.Link(f"link_core{core_id}_os")
    link_core_os.connect(os_link, (node_os, f"core{core_id}", "5ns"))
    if not MPI_PARTITIONING:
        link_core_os.setNoCut()

    link_mmu_dtlb = sst.Link(f"link_core{core_id}_mmu_dtlb")
    link_mmu_dtlb.connect((node_os_mmu, f"core{core_id}.dtlb", "1ns"), dtlb)
    if not MPI_PARTITIONING:
        link_mmu_dtlb.setNoCut()

    link_mmu_itlb = sst.Link(f"link_core{core_id}_mmu_itlb")
    link_mmu_itlb.connect((node_os_mmu, f"core{core_id}.itlb", "1ns"), itlb)
    if not MPI_PARTITIONING:
        link_mmu_itlb.setNoCut()


def group_id_of_core(core_id: int) -> int:
    return core_id % MESH_DIM_X


def worker_slot_for_core(core_id: int) -> int:
    return (core_id // MESH_DIM_X) - 1


for worker_core in range(MESH_DIM_X, numCpus):
    gid = group_id_of_core(worker_core)
    manager_core = gid
    slot = worker_slot_for_core(worker_core)
    worker_ep = ctrl_eps[worker_core]
    manager_ep = ctrl_eps[manager_core]
    if worker_ep is None or manager_ep is None:
        raise RuntimeError("control endpoint missing while GOLEM_CTRL_LINK_ENABLE=1")

    req_link = sst.Link(f"ctrl_req_{worker_core}_to_{manager_core}")
    req_link.connect(
        (worker_ep, "req_out", "2ns"), (manager_ep, f"req_in_{slot}", "2ns")
    )

    rsp_link = sst.Link(f"ctrl_rsp_{manager_core}_to_{worker_core}")
    rsp_link.connect(
        (manager_ep, f"rsp_out_{slot}", "2ns"), (worker_ep, "rsp_in", "2ns")
    )

for worker_core in range(MESH_DIM_X, numCpus):
    gid = group_id_of_core(worker_core)
    manager_core = gid
    slot = worker_slot_for_core(worker_core)
    worker_ep = sched_eps[worker_core]
    manager_ep = sched_eps[manager_core]
    if worker_ep is None or manager_ep is None:
        raise RuntimeError(
            "request scheduler endpoint missing while GOLEM_REQUEST_SCHEDULER_ENABLE=1"
        )

    req_link = sst.Link(f"sched_req_{worker_core}_to_{manager_core}")
    req_link.connect(
        (worker_ep, "req_out", "2ns"), (manager_ep, f"req_in_{slot}", "2ns")
    )

print("[CTRL] Control-link architecture configured via RoCC subcomponents")
