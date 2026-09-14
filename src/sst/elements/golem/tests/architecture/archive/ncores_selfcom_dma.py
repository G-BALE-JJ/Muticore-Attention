#!/usr/bin/env python3
"""
N-Core 流水线 Mailbox 测试 - 多进程版本
架构: N cores，每个 core 1 个硬件线程
通信: Mailbox (NoC)
优势: 每个进程独占 1 个核心，彻底避开 pthread/futex/LLSC
"""

import json
import os
import re
import sys
from pathlib import Path
import sst

if __package__ in {None, ""}:
    _tests_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _tests_dir not in sys.path:
        sys.path.insert(0, _tests_dir)

from architecture.noc_builder import MeshNoCBuilder
from architecture.cpu_builder import CPU_Builder


# --- 1. 帮助函数 ---
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
    unit_factors = {
        "B": 1,
        "K": 1000,
        "KB": 1000,
        "M": 1000**2,
        "MB": 1000**2,
        "G": 1000**3,
        "GB": 1000**3,
        "T": 1000**4,
        "TB": 1000**4,
        "KI": 1024,
        "KIB": 1024,
        "MI": 1024**2,
        "MIB": 1024**2,
        "GI": 1024**3,
        "GIB": 1024**3,
        "TI": 1024**4,
        "TIB": 1024**4,
    }

    factor = unit_factors.get(unit)
    if factor is None:
        raise ValueError(f"Unsupported size unit: {size_str}")

    return int(value * factor)


def addParamsPrefix(prefix, params):
    ret = {}
    for key, value in params.items():
        ret[prefix + "." + key] = value
    return ret


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


_attention_fused = _env_flag("GOLEM_ATTENTION_FUSED", False)
_attention_local_wcp = _env_flag(
    "GOLEM_ATTENTION_GENERIC_GEMM_ENABLE", _attention_fused
)
_control_plane_defaults = {
    "GOLEM_GROUP_MANAGER_ENABLE": False,
    "GOLEM_CTRL_LINK_ENABLE": False,
    "GOLEM_REQUEST_SCHEDULER_ENABLE": not _attention_fused,
    "GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE": _attention_local_wcp,
}
_manager_rocc_only = _env_flag("GOLEM_SFU_MANAGER_COORDINATOR", False)
_enabled_control_plane = [
    name
    for name, default in _control_plane_defaults.items()
    if _env_flag(name, default)
    and not (name == "GOLEM_GROUP_MANAGER_ENABLE" and _manager_rocc_only)
    and not (name == "GOLEM_CTRL_LINK_ENABLE" and _attention_fused)
    and not (name == "GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE" and _attention_local_wcp)
]
if _enabled_control_plane:
    raise ValueError(
        "archive/no-ctrl architecture does not wire control-plane endpoints; "
        "set GOLEM_GROUP_MANAGER_ENABLE=0, GOLEM_CTRL_LINK_ENABLE=0, "
        "GOLEM_REQUEST_SCHEDULER_ENABLE=0, and "
        "GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE=0 (enabled: "
        + ", ".join(_enabled_control_plane)
        + ")"
    )


TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRAMSIM3_CONFIG = os.getenv(
    "GOLEM_DRAMSIM3_CONFIG",
    os.path.join(TESTS_DIR, "architecture", "dram", "HBM_4Gb_x128.ini"),
)
ARTIFACT_ROOT = os.getenv("GOLEM_ARTIFACT_ROOT", os.path.join(TESTS_DIR, "artifacts"))
HBM_DIR = os.getenv("GOLEM_HBM_DIR", os.path.join(ARTIFACT_ROOT, "hbm"))
HBM_DUMP_OUTPUT = _env_flag("GOLEM_HBM_DUMP_OUTPUT", True)
STATS_DIR = os.getenv("GOLEM_STATS_DIR", os.path.join(ARTIFACT_ROOT, "stats"))
STATS_FILE = os.getenv("GOLEM_STATS_FILE", os.path.join(STATS_DIR, "stats_selfcom.txt"))
DRAMSIM3_OUT_DIR = os.getenv(
    "GOLEM_DRAMSIM3_OUT_DIR", os.path.join(STATS_DIR, "dramsim3")
)
MPI_PARTITIONING = _env_flag("GOLEM_MPI_PARTITIONING", False)
ATTENTION_QUERY_BLOCK_MPI = _env_flag("GOLEM_ATTENTION_QUERY_BLOCK_MPI", False)
MPI_RANKS = int(os.getenv("GOLEM_MPI_RANKS", "1"))
ATTENTION_PLACEMENT_FILE = os.getenv("GOLEM_ATTENTION_PLACEMENT_FILE", "")
attention_component_ranks = {}

if ATTENTION_QUERY_BLOCK_MPI and (not MPI_PARTITIONING or MPI_RANKS not in (2, 4)):
    raise ValueError(
        "GOLEM_ATTENTION_QUERY_BLOCK_MPI requires MPI partitioning with 2 or 4 ranks"
    )

os.makedirs(HBM_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)


# --- 2. 全局参数 ---
mh_debug = 0
mh_debug_level = 16
cpu_clock = os.getenv("VANADIS_CPU_CLOCK", "2.0GHz")
memctrl_clock = os.getenv("GOLEM_MEMCTRL_CLOCK", cpu_clock)
protocol = "MESI"
numCpus = int(os.getenv("VANADIS_NUM_CORES", 16))
os.environ["VANADIS_NUM_CORES"] = str(numCpus)
if ATTENTION_QUERY_BLOCK_MPI and numCpus != 20:
    raise ValueError("Attention query-block MPI requires exactly 20 cores")


def attention_rank_for_core(core_id: int) -> int:
    manager_id = core_id if core_id < 4 else core_id % 4
    return manager_id % MPI_RANKS


def attention_rank_for_memory_node(node_index: int) -> int:
    return 0 if node_index == OS_MEMORY_NODE_INDEX else (node_index - 1) % MPI_RANKS


def set_attention_component_rank(component, component_name: str, rank: int) -> None:
    component.setRank(rank)
    attention_component_ranks[component_name] = rank


def set_attention_cpu_rank(prefix: str, rank: int) -> None:
    for suffix in (
        "",
        ".processorBus",
        ".l1dcache",
        ".l1icache",
        ".dtlb",
        ".itlb",
        ".bus",
        ".l2cache",
    ):
        component_name = prefix + suffix
        component = sst.findComponentByName(component_name)
        if component is None:
            raise RuntimeError(f"missing CPU component for MPI placement: {component_name}")
        set_attention_component_rank(component, component_name, rank)


# 关键修正：每个核心 1 个硬件线程
numThreads = 1
verbosity = int(os.getenv("VANADIS_VERBOSE", 0))
os.environ["VANADIS_VERBOSE"] = str(verbosity)
os_verbosity = os.getenv("VANADIS_OS_VERBOSE", verbosity)
checkpointDir = ""
checkpoint = ""

# NUMA 配置
NUM_MEMORY_NODES = int(os.getenv("GOLEM_NUM_MEMORY_NODES", "4"))
if NUM_MEMORY_NODES < 2:
    raise ValueError("GOLEM_NUM_MEMORY_NODES must be >= 2")

OS_MEMORY_NODE_INDEX = 0
default_node_size_bytes = int(os.getenv("GOLEM_MEM_NODE_SIZE_BYTES", str(64 * 1024**2)))
default_phys_mem_size = (
    f"{(default_node_size_bytes * NUM_MEMORY_NODES) // (1024**2)}MiB"
)

# 物理内存大小（默认保持每节点 64MiB，随节点数线性扩展）
physMemSize = os.getenv("GOLEM_PHYS_MEM_SIZE", default_phys_mem_size)
try:
    physMemBytes = _parse_size_to_int(physMemSize)
except ValueError:
    fallback_size = "256MiB"  # 4 nodes * 64MB = 256MB total
    physMemBytes = _parse_size_to_int(fallback_size)
    print(
        f"Warning: unable to parse GOLEM_PHYS_MEM_SIZE='{physMemSize}', using {fallback_size}"
    )

if physMemBytes <= 0:
    raise ValueError("GOLEM_PHYS_MEM_SIZE must be positive")

if physMemBytes % NUM_MEMORY_NODES != 0:
    raise ValueError(
        f"GOLEM_PHYS_MEM_SIZE({physMemBytes}B) must be divisible by GOLEM_NUM_MEMORY_NODES({NUM_MEMORY_NODES})"
    )

memBytesPerNode = physMemBytes // NUM_MEMORY_NODES
memSizePerNode = f"{memBytesPerNode // (1024**2)}MiB"

configured_node_size = int(os.getenv("GOLEM_MEM_NODE_SIZE_BYTES", str(memBytesPerNode)))
if configured_node_size != memBytesPerNode:
    print(
        f"Warning: GOLEM_MEM_NODE_SIZE_BYTES={configured_node_size} differs from runtime memBytesPerNode={memBytesPerNode}; "
        f"compile-time address mapping may mismatch"
    )

# 传递每节点内存大小给 GlobalMemory，用于 DMA 地址映射
os.environ["GOLEM_MEM_NODE_SIZE"] = f"{memBytesPerNode}B"

print(f"[PTHREAD] 总内存: {physMemSize}")
print(f"[PTHREAD] 每节点: {memSizePerNode} ({memBytesPerNode} bytes)")
print(f"[PTHREAD] 核心数: {numCpus}")
print(f"[PTHREAD] 每核心硬件线程数: {numThreads}")
print(f"[PTHREAD] 模式: 多进程单线程（共享地址空间）")
print(f"[PTHREAD] HBM目录: {HBM_DIR}")
print(f"[PTHREAD] 统计文件: {STATS_FILE}")
print(
    f"[MPI] partitioning={int(MPI_PARTITIONING)} "
    f"dramsim3_out_dir={DRAMSIM3_OUT_DIR}"
)

# MMU 类型
mmuType = "simpleMMU"

# SST 核心选项
sst.setProgramOption("timebase", "1ps")
sst.setStatisticOutput(
    "sst.statOutputCSV",
    {
        "filepath": STATS_FILE,
        "separator": ",",
    },
)
sst.setStatisticLoadLevel(16)
sst.enableAllStatisticsForAllComponents({"type": "sst.AccumulatorStatistic"})

# 可执行文件
full_exe_name = os.getenv(
    "VANADIS_EXE",
    os.path.join(TESTS_DIR, "small", "mvm_noc_int_array", "riscv64", "test_noc_dma"),
)
exe_name = full_exe_name.split("/")[-1]

# --- 3. NodeOS 参数 ---
osParams = {
    "processDebugLevel": 0,
    "dbgLevel": os_verbosity,
    "dbgMask": 8,
    "cores": numCpus,
    "hardwareThreadCount": numThreads,
    "page_size": 4096,
    "physMemSize": physMemSize,
    "useMMU": True,
    "checkpointDir": checkpointDir,
    "checkpoint": checkpoint,
}

# 进程列表 - 多进程单线程模式
print(f"[PTHREAD] 为每个核心创建独立进程...")
processList = []
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
    "GOLEM_STAGE_PROGRESS",
    "GOLEM_RUNTIME_SILENT",
    "GOLEM_SILENT",
]

process_env_entries = []
for key in process_env_keys:
    if key in os.environ:
        process_env_entries.append(f"{key}={os.environ[key]}")

attention_guest_args = []
if _attention_fused:
    attention_guest_arg_names = (
        "GOLEM_ATTENTION_GUEST_MANAGER_QUERIES",
        "GOLEM_ATTENTION_GUEST_KEYS",
        "GOLEM_ATTENTION_GUEST_HEAD_DIM",
        "GOLEM_ATTENTION_GUEST_KEY_BLOCK_ROWS",
    )
    missing_attention_args = [
        name for name in attention_guest_arg_names if name not in os.environ
    ]
    if missing_attention_args:
        raise ValueError(
            "missing Attention guest arguments: " + ", ".join(missing_attention_args)
        )
    attention_guest_args = [os.environ[name] for name in attention_guest_arg_names]

for core_id in range(numCpus):
    process_params = {
        "env_count": len(process_env_entries),
        "exe": full_exe_name,
        "arg0": exe_name,
        "arg1": str(core_id),
        "argc": 2 + len(attention_guest_args),
    }
    for idx, argument in enumerate(attention_guest_args, start=2):
        process_params[f"arg{idx}"] = argument
    for idx, env_entry in enumerate(process_env_entries):
        process_params[f"env{idx}"] = env_entry
    processList.append((1, process_params))
    print(f"  进程 core{core_id}: {exe_name} (core_id={core_id})")

# NodeOS L1 缓存参数
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

# MMU 参数
mmuParams = {
    "debug_level": 0,
    "num_cores": numCpus,
    "num_threads": numThreads,
    "page_size": 4096,
}

# --- 4. 构建自适应 Mesh NoC ---
MESH_DIM_X = int(os.getenv("GOLEM_MESH_DIM_X", "4"))
if MESH_DIM_X <= 0:
    raise ValueError("GOLEM_MESH_DIM_X must be positive")
cpu_rows = (numCpus + MESH_DIM_X - 1) // MESH_DIM_X
DATA_MEMORY_NODE_COUNT = NUM_MEMORY_NODES - 1
MEMORY_LAYOUT = os.getenv("GOLEM_MEMORY_LAYOUT", "bottom_hbm").strip().lower()
if MEMORY_LAYOUT not in {"bottom_hbm", "top_hbm"}:
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
required_router_count = MESH_DIM_X * MESH_DIM_Y
LOCAL_PORTS = 3


def _evenly_spaced_columns(num_cols: int, count: int):
    if count <= 0:
        return []
    if count == 1:
        return [0]
    return [int(round(i * (num_cols - 1) / (count - 1))) for i in range(count)]


noc_input_buf_size = os.getenv("GOLEM_NOC_INPUT_BUF_SIZE", "8KB")
noc_output_buf_size = os.getenv("GOLEM_NOC_OUTPUT_BUF_SIZE", "8KB")
noc_link_bw = os.getenv("GOLEM_NOC_LINK_BW", "25GB/s")
noc_xbar_bw = os.getenv("GOLEM_NOC_XBAR_BW", "25GB/s")
noc_flit_size = os.getenv("GOLEM_NOC_FLIT_SIZE", "128B")
noc_inter_router_no_cut = _env_flag("GOLEM_NOC_INTER_ROUTER_NO_CUT", False)
noc_local_no_cut = _env_flag("GOLEM_NOC_LOCAL_NO_CUT", False)
print(
    f"[NoC] input_buf_size={noc_input_buf_size}, output_buf_size={noc_output_buf_size}, "
    f"link_bw={noc_link_bw}, xbar_bw={noc_xbar_bw}, flit_size={noc_flit_size}"
)
print(
    f"[NoC] inter_router_no_cut={int(noc_inter_router_no_cut)}, "
    f"local_no_cut={int(noc_local_no_cut)}"
)
print(f"[NoC] memory_layout={MEMORY_LAYOUT}")
print(f"[NoC] required_router_count={required_router_count}")

print(f"[SST] 构建 {MESH_DIM_X}x{MESH_DIM_Y} Mesh 网络...")
noc = MeshNoCBuilder(
    dim_x=MESH_DIM_X,
    dim_y=MESH_DIM_Y,
    local_ports=LOCAL_PORTS,
    link_bw=noc_link_bw,
    xbar_bw=noc_xbar_bw,
    flit_size=noc_flit_size,
    directional_link_latency="1ns",
    local_link_latency="1ns",
    input_buf_size=noc_input_buf_size,
    output_buf_size=noc_output_buf_size,
    num_vns=3,
    inter_router_no_cut=noc_inter_router_no_cut,
    local_no_cut=noc_local_no_cut,
    debug=1,
)
noc.build()

# --- 6. NUMA 节点分配 ---
data_row_start = DATA_MEMORY_ROW_INDEX * MESH_DIM_X
os_row_start = OS_MEMORY_ROW_INDEX * MESH_DIM_X
if DATA_MEMORY_NODE_COUNT > MESH_DIM_X:
    raise ValueError(
        f"DATA_MEMORY_NODE_COUNT({DATA_MEMORY_NODE_COUNT}) cannot exceed GOLEM_MESH_DIM_X({MESH_DIM_X})"
    )
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

if ATTENTION_QUERY_BLOCK_MPI:
    for router_id, router in enumerate(noc.routers):
        set_attention_component_rank(router, f"rtr_{router_id}", 0)
    for node_index, router_id in enumerate(MEMORY_ROUTERS):
        set_attention_component_rank(
            noc.get_router(router_id),
            f"rtr_{router_id}",
            attention_rank_for_memory_node(node_index),
        )
    for core_id, router_id in enumerate(cpu_routers):
        set_attention_component_rank(
            noc.get_router(router_id),
            f"rtr_{router_id}",
            attention_rank_for_core(core_id),
        )

# CPU_Builder creates GlobalMemory, which reads GOLEM_MEMORY_ROUTERS at build time.
print("[SST] 实例化 CPU_Builder...")
builder = CPU_Builder()

cpu_ports = []
ctrl_eps = {}
for core_id in range(numCpus):
    print(f"[SST] 构建 core{core_id}...")
    ports = builder.build(
        f"core{core_id}", core_id, core_id, add_l2_cache=True, add_rocc_golem=True
    )
    if ATTENTION_QUERY_BLOCK_MPI:
        set_attention_cpu_rank(
            f"core{core_id}", attention_rank_for_core(core_id)
        )
    cpu_ports.append(ports)
    ctrl_eps[core_id] = ports[6] if len(ports) > 6 else None

print("[SST] CPU 模块构建完成。")

print(f"[NUMA] 数据内存节点: {DATA_MEMORY_ROUTERS}")
print(f"[NUMA] OS 节点: {OS_ROUTER}")
print(f"[NUMA] CPU 节点: {cpu_routers}")

# --- 7. 连接 CPU 核心到 NoC ---
for core_id, router_id in enumerate(cpu_routers):
    print(f"[SST] 连接 Core {core_id} 到 rtr_{router_id}...")
    l2_mem = cpu_ports[core_id][1]
    gm = cpu_ports[core_id][4]  # GlobalMemory (包含 RDMA + DMA，复用 link_control)
    noc.attach_local(
        router_id, l2_mem, link_name=f"link_core{core_id}_l2_to_rtr{router_id}"
    )
    noc.attach_local(
        router_id, gm, link_name=f"link_core{core_id}_gm_to_rtr{router_id}"
    )
    # DMA 复用 GlobalMemory 的 link_control，不需要额外连接

# --- 8. 创建 NodeOS ---
print("[SST] 创建 NodeOS...")
node_os = sst.Component("os", "vanadis.VanadisNodeOS")
node_os.addParams(osParams)
if ATTENTION_QUERY_BLOCK_MPI:
    set_attention_component_rank(node_os, "os", 0)

# 添加所有进程
num = 0
for i, process in processList:
    for y in range(i):
        node_os.addParams(addParamsPrefix("process" + str(num), process))
        num += 1

# NodeOS MMU
node_os_mmu = node_os.setSubComponent("mmu", "mmu." + mmuType)
node_os_mmu.addParams(mmuParams)

# NodeOS 内存接口
node_os_mem_if = node_os.setSubComponent(
    "mem_interface", "memHierarchy.standardInterface"
)

# OS L1 Cache
os_l1 = sst.Component("node.os_l1cache", "memHierarchy.Cache")
os_l1.addParams(osl1cacheParams)
if ATTENTION_QUERY_BLOCK_MPI:
    set_attention_component_rank(os_l1, "node.os_l1cache", 0)
os_l1_hi = os_l1.setSubComponent("highlink", "memHierarchy.MemLink")
os_l1_lo = os_l1.setSubComponent("lowlink", "memHierarchy.MemNIC")
memory_destinations = ",".join(str(100 + idx) for idx in range(NUM_MEMORY_NODES))
os_l1_lo.addParams(
    {
        "group": 1,
        "destinations": memory_destinations,
        "network_bw": "25GB/s",
        "num_vns": 3,
    }
)

# 连接 NodeOS -> OS L1
link_os_to_l1 = sst.Link("link_node_os_to_l1")
link_os_to_l1.connect((node_os_mem_if, "lowlink", "1ns"), (os_l1_hi, "port", "1ns"))
link_os_to_l1.setNoCut()

# OS L1 连接到 OS_ROUTER
noc.attach_local(
    OS_ROUTER,
    (os_l1_lo, "port", "1ns"),
    link_name=f"link_node_osl1_to_rtr{OS_ROUTER}",
)


# --- 9. 创建分布式内存节点 ---
print(f"[NUMA] 创建 {NUM_MEMORY_NODES} 个分布式内存节点...")

for idx, router_id in enumerate(MEMORY_ROUTERS):
    addr_start = idx * memBytesPerNode
    addr_end = (idx + 1) * memBytesPerNode - 1
    print(f"[NUMA] 创建内存节点 {idx} 在 rtr_{router_id}...")
    print(f"  地址范围: 0x{addr_start:x} - 0x{addr_end:x}")

    dirctrl = sst.Component(f"dirctrl_{idx}", "memHierarchy.DirectoryController")
    if ATTENTION_QUERY_BLOCK_MPI:
        set_attention_component_rank(
            dirctrl, f"dirctrl_{idx}", attention_rank_for_memory_node(idx)
        )
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
            "network_bw": "25GB/s",
            "num_vns": 3,
            "network_input_buffer_size": os.getenv(
                "GOLEM_DIRCTRL_HIGHLINK_INPUT_BUF_SIZE", "64KB"
            ),
            "network_output_buffer_size": os.getenv(
                "GOLEM_DIRCTRL_HIGHLINK_OUTPUT_BUF_SIZE", "64KB"
            ),
            "golem_dma_response_chunk_bytes": os.getenv("GOLEM_DMA_RESPONSE_CHUNK_BYTES", "0"),
            "golem_dma_response_vn": os.getenv("GOLEM_DMA_RESPONSE_VN", "1"),
            "golem_dma_response_drain_limit": os.getenv(
                "GOLEM_DMA_RESPONSE_DRAIN_LIMIT", "0"
            ),
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
            "golem_dma_trace": os.getenv("GOLEM_DMA_TRACE", "0"),
        }
    )
    dir_lo = dirctrl.setSubComponent("lowlink", "memHierarchy.MemLink")

    memctrl = sst.Component(f"memory_{idx}", "memHierarchy.MemController")
    if ATTENTION_QUERY_BLOCK_MPI:
        set_attention_component_rank(
            memctrl, f"memory_{idx}", attention_rank_for_memory_node(idx)
        )
    mem_params = {
        "clock": memctrl_clock,
        "backend.mem_size": memSizePerNode,
        "addr_range_start": addr_start,
        "addr_range_end": addr_end,
        "debug_level": mh_debug_level,
        "debug": mh_debug,
        "checkpointDir": checkpointDir,
        "checkpoint": checkpoint,
    }

    if idx == OS_MEMORY_NODE_INDEX:
        mem_params.update(
            {
                "backing": "malloc",
                "initBacking": 0,
            }
        )
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
    mem_backend = memctrl.setSubComponent("backend", "memHierarchy.dramsim3")
    backend_params = {
        "mem_size": memSizePerNode,
        "config_ini": DRAMSIM3_CONFIG,
    }
    if MPI_PARTITIONING:
        node_output_dir = os.path.join(DRAMSIM3_OUT_DIR, f"node{idx}")
        os.makedirs(node_output_dir, exist_ok=True)
        backend_params["output_dir"] = node_output_dir
    mem_backend.addParams(backend_params)
    mem_backend.enableAllStatistics()

    link_dir_mem = sst.Link(f"link_dir{idx}_to_mem{idx}")
    link_dir_mem.connect((dir_lo, "port", "1ns"), (mem_hi, "port", "1ns"))
    link_dir_mem.setNoCut()
    noc.attach_local(
        router_id,
        (dir_hi, "port", "1ns"),
        link_name=f"link_dir{idx}_to_rtr{router_id}",
    )

# --- 10. 连接 CPU 核心到 NodeOS/MMU ---
print("[SST] 连接 Cores 到 NodeOS 和 MMU...")

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

if _env_flag("GOLEM_CTRL_LINK_ENABLE", False):
    for worker_core in range(MESH_DIM_X, numCpus):
        manager_core = worker_core % MESH_DIM_X
        worker_slot = (worker_core // MESH_DIM_X) - 1
        worker_ep = ctrl_eps[worker_core]
        manager_ep = ctrl_eps[manager_core]
        if worker_ep is None or manager_ep is None:
            raise RuntimeError(
                "control endpoint missing while GOLEM_CTRL_LINK_ENABLE=1"
            )

        req_link = sst.Link(f"ctrl_req_{worker_core}_to_{manager_core}")
        req_link.connect(
            (worker_ep, "req_out", "2ns"),
            (manager_ep, f"req_in_{worker_slot}", "2ns"),
        )

        rsp_link = sst.Link(f"ctrl_rsp_{manager_core}_to_{worker_core}")
        rsp_link.connect(
            (manager_ep, f"rsp_out_{worker_slot}", "2ns"),
            (worker_ep, "rsp_in", "2ns"),
        )

if ATTENTION_QUERY_BLOCK_MPI:
    if not ATTENTION_PLACEMENT_FILE:
        raise ValueError(
            "GOLEM_ATTENTION_PLACEMENT_FILE is required for query-block MPI verification"
        )
    placement_path = Path(ATTENTION_PLACEMENT_FILE)
    placement_path.parent.mkdir(parents=True, exist_ok=True)
    placement_tmp = placement_path.with_name(
        f"{placement_path.name}.{os.getpid()}.tmp"
    )
    placement_tmp.write_text(
        json.dumps(
            {
                "mpi_ranks": MPI_RANKS,
                "component_ranks": attention_component_ranks,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(placement_tmp, placement_path)

# --- 11. 总结 ---
print("\n" + "=" * 60)
print("Pthread 流水线架构配置完成")
print("=" * 60)
print(f"总内存: {physMemSize}")
print(f"内存节点数: {NUM_MEMORY_NODES}")
print(f"每节点内存: {memSizePerNode}")
print(f"核心数: {numCpus}")
print(f"每核心硬件线程数: {numThreads}")
print(f"模式: {numCpus} 个进程 × 1 线程/进程")
print(f"通信方式: Mailbox (NoC)")
print("=" * 60)
