import os
import re
import math
import sst

TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_ROOT = os.getenv("GOLEM_ARTIFACT_ROOT", os.path.join(TESTS_DIR, "artifacts"))

# debug参数设置（默认关闭，必要时可通过环境变量开启）
mh_debug = int(os.getenv("GOLEM_MH_DEBUG", "0"))
mh_debug_level = int(os.getenv("GOLEM_MH_DEBUG_LEVEL", "0"))

dbgAddr = 0
stopDbg = 0

checkpointDir = ""
checkpoint = ""

pythonDebug = False
enable_all_stats = int(os.getenv("GOLEM_SST_ENABLE_ALL_STATS", "1")) != 0

# 环境参数配置
vanadis_isa = os.getenv("VANADIS_ISA", "RISCV64")
isa = "riscv64"

loader_mode = os.getenv("VANADIS_LOADER_MODE", "0")

tlbType = "simpleTLB"

verbosity = int(os.getenv("VANADIS_VERBOSE", 0))
os_verbosity = os.getenv("VANADIS_OS_VERBOSE", verbosity)
pipe_trace_file = os.getenv("VANADIS_PIPE_TRACE", "vanadis_trace.txt")
lsq_ld_entries = os.getenv("VANADIS_LSQ_LD_ENTRIES", 16)
lsq_st_entries = os.getenv("VANADIS_LSQ_ST_ENTRIES", 8)
llsc_backoff_enabled = int(os.getenv("VANADIS_LLSC_BACKOFF_ENABLED", "1"))
llsc_backoff_base_cycles = int(os.getenv("VANADIS_LLSC_BACKOFF_BASE_CYCLES", "2"))
llsc_backoff_max_exp = int(os.getenv("VANADIS_LLSC_BACKOFF_MAX_EXP", "4"))
llsc_core_offset_cycles = int(os.getenv("VANADIS_LLSC_CORE_OFFSET_CYCLES", "10"))
llsc_ll_stagger_cycles = int(os.getenv("VANADIS_LLSC_LL_STAGGER_CYCLES", "0"))
llsc_trace = int(os.getenv("GOLEM_LLSC_TRACE", "0"))

rob_slots = os.getenv("VANADIS_ROB_SLOTS", 64)
retires_per_cycle = os.getenv("VANADIS_RETIRES_PER_CYCLE", 4)
issues_per_cycle = os.getenv("VANADIS_ISSUES_PER_CYCLE", 4)
decodes_per_cycle = os.getenv("VANADIS_DECODES_PER_CYCLE", 4)

integer_arith_cycles = int(os.getenv("VANADIS_INTEGER_ARITH_CYCLES", 2))
integer_arith_units = int(os.getenv("VANADIS_INTEGER_ARITH_UNITS", 2))
fp_arith_cycles = int(os.getenv("VANADIS_FP_ARITH_CYCLES", 8))
fp_arith_units = int(os.getenv("VANADIS_FP_ARITH_UNITS", 2))
branch_arith_cycles = int(os.getenv("VANADIS_BRANCH_ARITH_CYCLES", 2))

cpu_clock = os.getenv("VANADIS_CPU_CLOCK", "2.3GHz")
gm_trans_latency = os.getenv("GOLEM_GM_TRANS_LATENCY", "30ns")
local_gm_base_latency_cycles = int(os.getenv("GOLEM_LOCAL_GM_BASE_LATENCY_CYCLES", "1"))
local_gm_bytes_per_cycle = int(os.getenv("GOLEM_LOCAL_GM_BYTES_PER_CYCLE", "64"))
local_gm_read_ports = int(os.getenv("GOLEM_LOCAL_GM_READ_PORTS", "1"))
local_gm_write_ports = int(os.getenv("GOLEM_LOCAL_GM_WRITE_PORTS", "1"))
local_gm_queue_depth = int(os.getenv("GOLEM_LOCAL_GM_QUEUE_DEPTH", "32"))
local_gm_max_request_bytes = int(os.getenv("GOLEM_LOCAL_GM_MAX_REQUEST_BYTES", "4096"))


def _parse_frequency_hz(freq_str: str) -> float:
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([kKmMgGtT]?)(?:Hz)\s*$", freq_str)
    if not m:
        raise ValueError(f"Unsupported frequency format: {freq_str}")
    value = float(m.group(1))
    unit = m.group(2).upper()
    scale = {
        "": 1.0,
        "K": 1e3,
        "M": 1e6,
        "G": 1e9,
        "T": 1e12,
    }[unit]
    return value * scale


def _parse_time_ns(time_str: str) -> float:
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(ps|ns|us|ms|s)\s*$", time_str)
    if not m:
        raise ValueError(f"Unsupported time format: {time_str}")
    value = float(m.group(1))
    unit = m.group(2)
    scale = {
        "ps": 1e-3,
        "ns": 1.0,
        "us": 1e3,
        "ms": 1e6,
        "s": 1e9,
    }[unit]
    return value * scale


gm_retry_tick_cpu_cycles = max(
    1,
    int(
        round((_parse_time_ns(gm_trans_latency) * _parse_frequency_hz(cpu_clock)) / 1e9)
    ),
)

numCpus = int(os.getenv("VANADIS_NUM_CORES", 16))
numThreads = int(os.getenv("VANADIS_NUM_HW_THREADS", 1))
total_gemm_cores = int(os.getenv("GOLEM_TOTAL_GEMM_CORES", str(numCpus)))
group_manager_enabled = int(os.getenv("GOLEM_GROUP_MANAGER_ENABLE", "0")) != 0
total_groups = int(os.getenv("GOLEM_TOTAL_GROUPS", "1"))
dedicated_manager_cores = total_groups if group_manager_enabled else 0
active_worker_cores = total_gemm_cores - dedicated_manager_cores
progress_heartbeat = int(os.getenv("GOLEM_PROGRESS_HEARTBEAT", "0"))
progress_interval_cycles = int(os.getenv("GOLEM_PROGRESS_INTERVAL_CYCLES", "50000"))

vanadis_cpu_type = "vanadis."
vanadis_cpu_type += os.getenv("VANADIS_CPU_ELEMENT_NAME", "dbg_VanadisCPU")

if verbosity > 0:
    print(
        "Verbosity: "
        + str(verbosity)
        + " -> loading Vanadis CPU type: "
        + vanadis_cpu_type
    )

vanadis_decoder = "vanadis.Vanadis" + vanadis_isa + "Decoder"
vanadis_os_hdlr = "vanadis.Vanadis" + vanadis_isa + "OSHandler"


def _normalize_matmul_dtype(dtype: str) -> str:
    value = (dtype or "int32").strip().lower()
    aliases = {
        "int": "int32",
        "i32": "int32",
        "int32": "int32",
        "float": "fp32",
        "float32": "fp32",
        "fp32": "fp32",
        "half": "fp16",
        "float16": "fp16",
        "fp16": "fp16",
    }
    value = aliases.get(value, value)
    if value not in {"int32", "fp16", "fp32"}:
        raise ValueError(f"Unsupported GOLEM_MATMUL_DTYPE: {dtype}")
    return value


def _default_rocc_type(dtype: str) -> str:
    if dtype in {"fp16", "fp32"}:
        return "golem.RoCCAnalogFloat"
    return "golem.RoCCAnalogInt"


def _default_array_type(dtype: str) -> str:
    if dtype in {"fp16", "fp32"}:
        return "golem.MVMFloatArray"
    return "golem.MVMIntArray"


matmul_dtype = _normalize_matmul_dtype(os.getenv("GOLEM_MATMUL_DTYPE", "int32"))
rocc_type = os.getenv("GOLEM_ROCC_TYPE", _default_rocc_type(matmul_dtype))
array_type = os.getenv("GOLEM_ARRAY_TYPE", _default_array_type(matmul_dtype))
gm_buffer_length = os.getenv("GOLEM_GM_BUFFER_LENGTH", "64KB")
gm_link_bw = os.getenv(
    "GOLEM_GM_LINK_BW", os.getenv("GOLEM_NOC_LINK_BW", "128GB/s")
)
gm_c_buffer_bytes = int(os.getenv("GOLEM_GM_C_BUFFER_BYTES", str(1024 * 1024)))
gm_c_buffer_read_bpc = int(os.getenv("GOLEM_GM_C_BUFFER_READ_BPC", "256"))
gm_c_buffer_write_bpc = int(os.getenv("GOLEM_GM_C_BUFFER_WRITE_BPC", "256"))
gm_c_buffer_latency_cycles = int(os.getenv("GOLEM_GM_C_BUFFER_LATENCY_CYCLES", "1"))
attention_tile_storage_banks = int(
    os.getenv("GOLEM_ATTENTION_TILE_STORAGE_BANKS", "16")
)
attention_tile_storage_bank_bpc = int(
    os.getenv("GOLEM_ATTENTION_TILE_STORAGE_BANK_BPC", "64")
)
final_c_write_enable = int(os.getenv("GOLEM_FINAL_C_WRITE_ENABLE", "1"))
output_mode = os.getenv(
    "GOLEM_OUTPUT_MODE", "hbm" if final_c_write_enable else "fusion"
).strip().lower()
fusion_dump_enable = int(os.getenv("GOLEM_FUSION_DUMP_ENABLE", "0"))
fusion_dump_dir = os.getenv("GOLEM_FUSION_DUMP_DIR", "")
dma_write_vn = int(os.getenv("GOLEM_DMA_WRITE_VN", "2"))
noc_vn_priority_enable = int(os.getenv("GOLEM_NOC_VN_PRIORITY_ENABLE", "1"))
noc_vn_priority_order = (
    os.getenv("GOLEM_NOC_VN_PRIORITY_ORDER", "1,0,2")
    if noc_vn_priority_enable
    else ""
)
noc_vn_starvation_vn = int(os.getenv("GOLEM_NOC_VN_STARVATION_VN", "2"))
noc_vn_max_starvation_cycles = int(
    os.getenv("GOLEM_NOC_VN_MAX_STARVATION_CYCLES", "4096")
)
gm_dma_max_inflight = int(os.getenv("GOLEM_DMA_MAX_INFLIGHT", "256"))
gm_dma_retry_ticks = int(os.getenv("GOLEM_DMA_READ_RETRY_TICKS", "96"))
gm_dma_max_retries = int(os.getenv("GOLEM_DMA_READ_MAX_RETRIES", "8"))
mvm_dump_enable = int(os.getenv("GOLEM_MVM_DUMP_ENABLE", "0"))
mvm_dump_dir = os.getenv("GOLEM_MVM_DUMP_DIR", os.path.join(ARTIFACT_ROOT, "mvm_dumps"))
mvm_dump_mode = os.getenv("GOLEM_MVM_DUMP_MODE", "overwrite")
simulation_mode = os.getenv("GOLEM_SIM_MODE", "full-functional")
if simulation_mode not in {"full-functional", "full-timing"}:
    raise ValueError(f"Unsupported GOLEM_SIM_MODE: {simulation_mode}")
functional_compute = int(simulation_mode == "full-functional")
attention_fused = int(os.getenv("GOLEM_ATTENTION_FUSED", "0")) != 0
attention_generic_gemm_enable = int(os.getenv(
    "GOLEM_ATTENTION_GENERIC_GEMM_ENABLE", "1" if attention_fused else "0"
))
gm_dma_burst_bytes = int(os.getenv("GOLEM_DMA_BURST_BYTES", "64"))
mvm_latency_ovec2gm = int(os.getenv("GOLEM_LATENCY_MVM_OVEC2GM", "10"))
mvm_latency_gm2ivec = int(os.getenv("GOLEM_LATENCY_MVM_GM2IVEC", "10"))
mvm_latency_gm2imat = int(os.getenv("GOLEM_LATENCY_MVM_GM2IMAT", "10"))
array_clock = os.getenv("GOLEM_ARRAY_CLOCK", cpu_clock)
array_buffer_base_latency_cycles = int(
    os.getenv("GOLEM_ARRAY_BUFFER_BASE_LATENCY_CYCLES", "1")
)
array_buffer_bytes_per_cycle = int(
    os.getenv("GOLEM_ARRAY_BUFFER_BYTES_PER_CYCLE", "64")
)
array_buffer_ports = int(os.getenv("GOLEM_ARRAY_BUFFER_PORTS", "1"))
array_operand_context_banks = int(
    os.getenv("GOLEM_ARRAY_OPERAND_CONTEXT_BANKS", "1")
)
array_buffer_queue_depth = int(
    os.getenv("GOLEM_ARRAY_BUFFER_QUEUE_DEPTH", "64")
)
array_output_read_credits = int(
    os.getenv("GOLEM_ARRAY_OUTPUT_READ_CREDITS", "1")
)
attention_near_array_output_bytes_per_cycle = int(
    os.getenv("GOLEM_ATTENTION_NEAR_ARRAY_OUTPUT_BYTES_PER_CYCLE", "512")
)
attention_near_array_output_credits = int(
    os.getenv("GOLEM_ATTENTION_NEAR_ARRAY_OUTPUT_CREDITS", "2")
)
array_output_read_banks = int(
    os.getenv("GOLEM_ARRAY_OUTPUT_READ_BANKS", "1")
)
matrix_broadcast_max_fanout = int(
    os.getenv("GOLEM_MATRIX_BROADCAST_MAX_FANOUT", "16")
)
matrix_broadcast_bytes_per_cycle = int(
    os.getenv("GOLEM_MATRIX_BROADCAST_BYTES_PER_CYCLE", "64")
)
matrix_broadcast_base_latency_cycles = int(
    os.getenv("GOLEM_MATRIX_BROADCAST_BASE_LATENCY_CYCLES", "1")
)
matrix_broadcast_stage_latency_cycles = int(
    os.getenv("GOLEM_MATRIX_BROADCAST_STAGE_LATENCY_CYCLES", "1")
)
rocc_verbose = int(os.getenv("GOLEM_ROCC_VERBOSE", "0"))
gm_verbose = int(os.getenv("GOLEM_GM_VERBOSE", "0"))
gm_dump_data = int(os.getenv("GOLEM_GM_DUMP_DATA", "0"))
ctrl_link_enable = int(os.getenv("GOLEM_CTRL_LINK_ENABLE", "0"))
ctrl_link_latency = os.getenv("GOLEM_CTRL_LINK_LATENCY", "2ns")
ctrl_link_queue_depth = os.getenv("GOLEM_CTRL_QUEUE_DEPTH", "32")
ctrl_link_max_grants_per_schedule = os.getenv(
    "GOLEM_GROUP_MAX_GRANTS_PER_SCHEDULE", "1"
)
ctrl_link_verbose = os.getenv("GOLEM_CTRL_VERBOSE", "0")
request_scheduler_enable = int(os.getenv(
    "GOLEM_REQUEST_SCHEDULER_ENABLE", "0" if attention_fused else "1"
))
request_scheduler_queue_depth = os.getenv("GOLEM_REQUEST_SCHEDULER_QUEUE_DEPTH", "64")
request_scheduler_initial_chunk_credit_env = os.getenv("GOLEM_DMA_NODE_CHUNK_CREDITS")
request_scheduler_legacy_node_credit_env = os.getenv(
    "GOLEM_DMA_NODE_CREDITS",
    os.getenv("GOLEM_REQUEST_SCHEDULER_INITIAL_CREDIT", "4"),
)
request_scheduler_node_credit_chunk_bytes = os.getenv(
    "GOLEM_DMA_CREDIT_CHUNK_BYTES",
    str(gm_dma_burst_bytes),
)
request_scheduler_panel_chunk_bytes = os.getenv(
    "GOLEM_DMA_PANEL_CHUNK_BYTES",
    request_scheduler_node_credit_chunk_bytes,
)
request_scheduler_issue_budget_per_tick = os.getenv(
    "GOLEM_SCHED_ISSUE_BUDGET_PER_TICK",
    "2",
)
request_scheduler_worker_credit_cap = os.getenv(
    "GOLEM_SCHED_WORKER_CREDIT_CAP", "0"
)
wcp_prefetch_windows = os.getenv("GOLEM_WCP_PREFETCH_WINDOWS", "2")
wcp_cross_macro_prefetch = os.getenv("GOLEM_WCP_CROSS_MACRO_PREFETCH_ENABLE", "0")
request_scheduler_submit_batch_size = os.getenv("GOLEM_SCHED_SUBMIT_BATCH_SIZE", "4")
request_scheduler_done_batch_size = os.getenv("GOLEM_SCHED_DONE_BATCH_SIZE", "4")
request_scheduler_verbose = os.getenv("GOLEM_REQUEST_SCHEDULER_VERBOSE", "0")
request_scheduler_trace = os.getenv("GOLEM_REQUEST_SCHEDULER_TRACE", "0")
request_scheduler_group_round_robin = os.getenv("GOLEM_DMA_GROUP_RR_ENABLE", "0")
request_scheduler_tile_chunk_quantum = os.getenv("GOLEM_DMA_TILE_CHUNK_QUANTUM", "1")
worker_command_processor_enable = int(
    os.getenv("GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE",
              str(attention_generic_gemm_enable))
)
request_scheduler_event_driven_worker_env = os.getenv(
    "GOLEM_REQUEST_SCHEDULER_EVENT_DRIVEN_WORKER", ""
).strip()
request_scheduler_event_driven_worker = (
    int(request_scheduler_event_driven_worker_env)
    if request_scheduler_event_driven_worker_env
    else int(simulation_mode == "full-timing" and worker_command_processor_enable)
)
vanadis_rocc_wait_fastpath_env = os.getenv(
    "GOLEM_VANADIS_ROCC_WAIT_FASTPATH", ""
).strip()
vanadis_rocc_wait_fastpath = (
    int(vanadis_rocc_wait_fastpath_env)
    if vanadis_rocc_wait_fastpath_env
    else int(simulation_mode == "full-timing" and worker_command_processor_enable)
)
vanadis_rocc_wait_fastpath_threshold = int(
    os.getenv("GOLEM_VANADIS_ROCC_WAIT_FASTPATH_THRESHOLD", "256")
)
worker_command_processor_verbose = os.getenv(
    "GOLEM_WORKER_COMMAND_PROCESSOR_VERBOSE", "0"
)
wcp_gemm_proxy_queue_depth = os.getenv("GOLEM_WCP_GEMM_PROXY_QUEUE_DEPTH", "32")
wcp_gemm_proxy_issue_width = os.getenv("GOLEM_WCP_GEMM_PROXY_ISSUE_WIDTH", "1")
wcp_gemm_proxy_command_latency_cycles = os.getenv(
    "GOLEM_WCP_GEMM_PROXY_COMMAND_LATENCY_CYCLES", "1"
)
wcp_gemm_proxy_completion_latency_cycles = os.getenv(
    "GOLEM_WCP_GEMM_PROXY_COMPLETION_LATENCY_CYCLES", "1"
)
sfu_enable = int(os.getenv("GOLEM_SFU_ENABLE", "0")) != 0
sfu_max_inflight = os.getenv("GOLEM_SFU_MAX_INFLIGHT", "8")
sfu_stats_latency = os.getenv("GOLEM_SFU_STATS_LATENCY", "1")
sfu_merge_latency = os.getenv("GOLEM_SFU_MERGE_LATENCY", "1")
sfu_normalize_latency = os.getenv("GOLEM_SFU_NORMALIZE_LATENCY", "1")
sfu_vector_lanes = os.getenv("GOLEM_SFU_VECTOR_LANES", "16")
sfu_exp_lanes = os.getenv("GOLEM_SFU_EXP_LANES", "4")
sfu_reduction_tree_latency = os.getenv("GOLEM_SFU_REDUCTION_TREE_LATENCY", "4")
sfu_exp_latency = os.getenv("GOLEM_SFU_EXP_LATENCY", "8")
sfu_reciprocal_latency = os.getenv("GOLEM_SFU_RECIPROCAL_LATENCY", "1")
attention_kv_pair_reuse = int(
    os.getenv("GOLEM_ATTENTION_KV_PAIR_REUSE", "0")
)
attention_kv_query_group_size = int(
    os.getenv("GOLEM_ATTENTION_KV_QUERY_GROUP_SIZE", "2")
)
attention_kv_distribution_enable = int(
    os.getenv("GOLEM_ATTENTION_KV_DISTRIBUTION_ENABLE", "0")
)
if attention_kv_query_group_size not in (1, 2, 4):
    raise ValueError(
        "GOLEM_ATTENTION_KV_QUERY_GROUP_SIZE must be 1, 2, or 4"
    )
if attention_kv_pair_reuse and attention_kv_query_group_size == 1:
    raise ValueError(
        "K/V query reuse requires GOLEM_ATTENTION_KV_QUERY_GROUP_SIZE 2 or 4"
    )
sfu_row_contexts = os.getenv(
    "GOLEM_SFU_ROW_CONTEXTS", "16" if attention_kv_pair_reuse else "4"
)
if attention_kv_pair_reuse and int(sfu_row_contexts) < 16:
    raise ValueError(
        "K/V pair reuse requires GOLEM_SFU_ROW_CONTEXTS >= 16"
    )
sfu_scratchpad_bytes = os.getenv("GOLEM_SFU_SCRATCHPAD_BYTES", "65536")
sfu_distributed_reduction_transport = os.getenv("GOLEM_SFU_DISTRIBUTED_REDUCTION_TRANSPORT", "shared")
sfu_reduction_vn = os.getenv("GOLEM_SFU_REDUCTION_VN", "")
sfu_verbose = os.getenv("GOLEM_SFU_VERBOSE", "0")

num_arrays = int(os.getenv("GOLEM_NUM_ARRAYS", 1))
attention_cluster_qk_arrays = int(
    os.getenv("GOLEM_ATTENTION_CLUSTER_QK_ARRAYS", "16")
)
array_input_size = int(os.getenv("GOLEM_ARRAY_INPUT_SIZE", "4"))
array_output_size = int(os.getenv("GOLEM_ARRAY_OUTPUT_SIZE", "4"))
array_mac_per_cu_per_cycle = float(os.getenv("GOLEM_ARRAY_MAC_PER_CU_PER_CYCLE", "1"))
array_pipeline_depth = int(os.getenv("GOLEM_ARRAY_PIPELINE_DEPTH", "0"))
if array_mac_per_cu_per_cycle <= 0:
    raise ValueError("GOLEM_ARRAY_MAC_PER_CU_PER_CYCLE must be positive")
if array_pipeline_depth < 0:
    raise ValueError("GOLEM_ARRAY_PIPELINE_DEPTH must be non-negative")
num_memory_nodes = int(os.getenv("GOLEM_NUM_MEMORY_NODES", "4"))
if num_memory_nodes <= 0:
    raise ValueError("GOLEM_NUM_MEMORY_NODES must be positive")
memory_destinations = ",".join(str(100 + idx) for idx in range(num_memory_nodes))

matmul_m = int(
    os.getenv("GOLEM_MATMUL_M", os.getenv("GOLEM_GEMM_M", str(array_output_size)))
)
matmul_n = int(os.getenv("GOLEM_MATMUL_N", os.getenv("GOLEM_GEMM_N", str(num_arrays))))
matmul_k = int(
    os.getenv("GOLEM_MATMUL_K", os.getenv("GOLEM_GEMM_K", str(array_input_size)))
)
matmul_block_m = int(os.getenv("GOLEM_MATMUL_BLOCK_M", str(array_output_size)))
matmul_block_n = int(os.getenv("GOLEM_MATMUL_BLOCK_N", str(num_arrays)))
matmul_block_k = int(os.getenv("GOLEM_MATMUL_BLOCK_K", str(array_input_size)))
matmul_elem_bytes = 2 if matmul_dtype == "fp16" else 4


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


request_sched_slot0_bytes = matmul_block_m * matmul_block_k * matmul_elem_bytes
# B vectors are packed without padding; the DMA payload is the useful vector data.
request_sched_vec_stride_bytes = matmul_block_k * matmul_elem_bytes
request_sched_slot1_bytes = matmul_block_n * request_sched_vec_stride_bytes
request_sched_credit_chunk_bytes_int = max(1, int(request_scheduler_node_credit_chunk_bytes))
request_sched_slot0_chunks = max(
    1,
    math.ceil(request_sched_slot0_bytes / request_sched_credit_chunk_bytes_int),
)
request_sched_slot1_chunks = max(
    1,
    math.ceil(request_sched_slot1_bytes / request_sched_credit_chunk_bytes_int),
)
request_sched_prefetch_windows = max(1, int(wcp_prefetch_windows))
request_sched_window_buffers = request_sched_prefetch_windows + 1
request_sched_window_k_tiles = max(1, _env_int("GOLEM_DMA_WINDOW_K_TILES", 4))
request_sched_total_k_tiles = max(1, math.ceil(matmul_k / max(1, matmul_block_k)))
request_sched_resident_k_tiles = min(request_sched_window_k_tiles, request_sched_total_k_tiles)
request_sched_a_reuse_n = max(1, _env_int("GOLEM_A_REUSE_N_TILES", 1))
request_sched_b_reuse_m = max(1, _env_int("GOLEM_B_REUSE_M_TILES", 1))
request_sched_slot_count = max(1, _env_int("GOLEM_DMA_SLOT_COUNT", 2))
if request_sched_a_reuse_n > 1 and request_sched_b_reuse_m > 1:
    request_sched_mat_resident_k = request_sched_slot_count // (
        request_sched_window_buffers * request_sched_b_reuse_m
    )
    request_sched_vec_resident_k = request_sched_slot_count // (
        request_sched_window_buffers * request_sched_a_reuse_n
    )
    request_sched_slot_resident_k = min(
        request_sched_mat_resident_k, request_sched_vec_resident_k
    )
    if request_sched_slot_resident_k > 0:
        request_sched_resident_k_tiles = min(
            request_sched_resident_k_tiles, request_sched_slot_resident_k
        )
request_sched_active_workers = max(1, active_worker_cores)
request_sched_per_worker_ab_chunks = request_sched_resident_k_tiles * (
    request_sched_b_reuse_m * request_sched_slot0_chunks
    + request_sched_a_reuse_n * request_sched_slot1_chunks
)
request_sched_derived_node_chunk_credit = (
    request_sched_active_workers
    * request_sched_window_buffers
    * request_sched_per_worker_ab_chunks
)
if request_scheduler_initial_chunk_credit_env is not None and request_scheduler_initial_chunk_credit_env.strip() != "":
    request_scheduler_initial_chunk_credit = request_scheduler_initial_chunk_credit_env
else:
    request_scheduler_initial_chunk_credit = str(request_sched_derived_node_chunk_credit)

# Current MAC-array model:
# - num_cu semantically equals hardware output size
# - input-size semantics follow hardware array width; WCP expands logical block_k via micro-tiling
array_num_cu = array_output_size
runtime_array_input_size = array_input_size
runtime_array_active_k = min(matmul_block_k, array_input_size)
mvm_latency_compute_cycles = int(
    math.ceil(runtime_array_active_k / array_mac_per_cu_per_cycle) + array_pipeline_depth
)
array_latency_ns = (mvm_latency_compute_cycles / _parse_frequency_hz(array_clock)) * 1e9
array_latency = f"{array_latency_ns:.6f}ns"


def _tasks_for_core(core_id: int) -> int:
    if core_id < 0:
        return 0
    if active_worker_cores <= 0:
        return 0

    if group_manager_enabled:
        if core_id < total_groups or core_id >= total_gemm_cores:
            return 0
        worker_slot = core_id - total_groups
    else:
        if core_id >= total_gemm_cores:
            return 0
        worker_slot = core_id

    if min(matmul_block_m, matmul_block_n, matmul_block_k) <= 0:
        return 0
    if (
        (matmul_m % matmul_block_m) != 0
        or (matmul_n % matmul_block_n) != 0
        or (matmul_k % matmul_block_k) != 0
    ):
        return 0
    total_tasks = (matmul_m // matmul_block_m) * (matmul_n // matmul_block_n)
    if worker_slot >= total_tasks:
        return 0
    return ((total_tasks - 1 - worker_slot) // active_worker_cores) + 1


def _expected_mvm_ops_for_core(core_id: int) -> int:
    if matmul_block_k <= 0 or (matmul_k % matmul_block_k) != 0:
        return 0
    k_tiles = matmul_k // matmul_block_k
    return _tasks_for_core(core_id) * k_tiles * matmul_block_n


def _expected_dma_chunks_for_core(core_id: int) -> int:
    if matmul_block_k <= 0 or (matmul_k % matmul_block_k) != 0:
        return 0
    k_tiles = matmul_k // matmul_block_k
    return _tasks_for_core(core_id) * k_tiles * (1 + matmul_block_n)


gm_progress_interval_ticks = max(
    1, progress_interval_cycles // gm_retry_tick_cpu_cycles
)

protocol = "MESI"

# CPU相关参数配置
tlbParams = {
    "debug_level": 0,
    "hitLatency": 1,
    "num_hardware_threads": numThreads,
    "num_tlb_entries_per_thread": 1024,
    "tlb_set_size": 1024,
}

tlbWrapperParams = {
    "debug_level": 0,
}

decoderParams = {
    "loader_mode": loader_mode,
    "uop_cache_entries": 1536,
    "predecode_cache_entries": 4,
}

osHdlrParams = {}

branchPredParams = {"branch_entries": 32}

cpuParams = {
    "clock": cpu_clock,
    "verbose": verbosity,
    "hardware_threads": numThreads,
    "physical_fp_registers": 168 * numThreads,
    "physical_integer_registers": 180 * numThreads,
    "integer_arith_cycles": integer_arith_cycles,
    "integer_arith_units": integer_arith_units,
    "fp_arith_cycles": fp_arith_cycles,
    "fp_arith_units": fp_arith_units,
    "branch_unit_cycles": branch_arith_cycles,
    "print_int_reg": False,
    "print_fp_reg": False,
    "pipeline_trace_file": pipe_trace_file,
    "reorder_slots": rob_slots,
    "decodes_per_cycle": decodes_per_cycle,
    "issues_per_cycle": issues_per_cycle,
    "retires_per_cycle": retires_per_cycle,
    "pause_when_retire_address": os.getenv("VANADIS_HALT_AT_ADDRESS", 0),
    "start_verbose_when_issue_address": dbgAddr,
    "stop_verbose_when_retire_address": stopDbg,
    "print_rob": False,
    "checkpointDir": checkpointDir,
    "checkpoint": checkpoint,
    "rocc_wait_fastpath": vanadis_rocc_wait_fastpath,
    "rocc_wait_fastpath_threshold": vanadis_rocc_wait_fastpath_threshold,
}

lsqParams = {
    "verbose": verbosity,
    "address_mask": 0xFFFFFFFFFFFFFFFF,  # 修复：使用64位地址掩码以支持 RISCV64
    "max_stores": 16,  # 增加 store 队列容量从 8 到 16
    "max_loads": 32,  # 增加 load 队列容量从 16 到 32
    # LLSC backoff 启用以防止活锁
    "llsc_backoff_enabled": llsc_backoff_enabled,
    "llsc_backoff_base_cycles": llsc_backoff_base_cycles,
    "llsc_backoff_max_exp": llsc_backoff_max_exp,
    "llsc_core_offset_cycles": llsc_core_offset_cycles,
    "llsc_ll_stagger_cycles": llsc_ll_stagger_cycles,
    "llsc_trace": llsc_trace,
}

roccParams = {
    "clock": cpu_clock,
    "verbose": rocc_verbose,
    "max_instructions": 8,
    "progress_heartbeat": progress_heartbeat,
    "progress_interval_cycles": progress_interval_cycles,
}

roccarrayParams = {
    "inputOperandSize": matmul_elem_bytes,
    "outputOperandSize": matmul_elem_bytes,
    "vectorStrideBytes": request_sched_vec_stride_bytes,
    "latency_mvm_ovec2gm": mvm_latency_ovec2gm,
    "latency_mvm_gm2ivec": mvm_latency_gm2ivec,
    "latency_mvm_gm2imat": mvm_latency_gm2imat,
    "latency_remote_st": 10,
    "latency_remote_ld": 10,
    "enable_async_array_load": int(os.getenv("GOLEM_ENABLE_ASYNC_ARRAY_LOAD", "1")),
    "workerCommandProcessorEnable": worker_command_processor_enable,
    "attention_window_offset": int(os.getenv("GOLEM_ATTENTION_WINDOW_OFFSET", "0xC0000"), 0),
    "attention_window_bytes": int(os.getenv("GOLEM_ATTENTION_WINDOW_BYTES", "0x10000"), 0),
    "attention_cluster_enable": int(
        os.getenv("GOLEM_ATTENTION_CLUSTER_ENABLE", "0")
    ),
    "attention_cluster_qk_arrays": attention_cluster_qk_arrays,
    "attention_kv_tile_rotation": int(
        os.getenv("GOLEM_ATTENTION_KV_TILE_ROTATION", "0")
    ),
    "attention_kv_double_buffer": int(
        os.getenv("GOLEM_ATTENTION_KV_DOUBLE_BUFFER", "0")
    ),
    "attention_kv_buffer_count": int(
        os.getenv("GOLEM_ATTENTION_KV_BUFFER_COUNT", "2")
    ),
    "attention_kv_distribution_enable": attention_kv_distribution_enable,
    "attention_key_block_rows": int(
        os.getenv("GOLEM_ATTENTION_KEY_BLOCK_ROWS", "32")
    ),
    "attention_kv_second_lookahead": int(
        os.getenv("GOLEM_ATTENTION_KV_SECOND_LOOKAHEAD", "1")
    ),
    "attention_kv_cross_query_prefetch": int(
        os.getenv("GOLEM_ATTENTION_KV_CROSS_QUERY_PREFETCH", "0")
    ),
    "attention_kv_pair_reuse": attention_kv_pair_reuse,
    "attention_kv_query_group_size": attention_kv_query_group_size,
    "attention_pv_v_tile_reuse": int(
        os.getenv("GOLEM_ATTENTION_PV_V_TILE_REUSE", "0")
    ),
    "attention_pv_v_tile_group_retention": int(
        os.getenv("GOLEM_ATTENTION_PV_V_TILE_GROUP_RETENTION", "0")
    ),
    "attention_pv_v_tile_buffer_bytes": int(
        os.getenv("GOLEM_ATTENTION_PV_V_TILE_BUFFER_BYTES", "16384")
    ),
    "attention_pv_v_tile_buffer_hit_ticks": int(
        os.getenv("GOLEM_ATTENTION_PV_V_TILE_BUFFER_HIT_TICKS", "1")
    ),
    "attention_pv_v_tile_buffer_bytes_per_cycle": int(
        os.getenv("GOLEM_ATTENTION_PV_V_TILE_BUFFER_BYTES_PER_CYCLE", "64")
    ),
    "attention_pv_v_tile_buffer_offset": int(
        os.getenv("GOLEM_ATTENTION_PV_V_TILE_BUFFER_OFFSET", "0"), 0
    ),
    "attention_pv_input_pipeline": int(
        os.getenv("GOLEM_ATTENTION_PV_INPUT_PIPELINE", "0")
    ),
    "attention_cluster_pv_row_wavefront": int(
        os.getenv("GOLEM_ATTENTION_CLUSTER_PV_ROW_WAVEFRONT", "0")
    ),
    "attention_cluster_qk_matrix_lookahead": int(
        os.getenv("GOLEM_ATTENTION_CLUSTER_QK_MATRIX_LOOKAHEAD", "0")
    ),
    "attention_cluster_pv_matrix_lookahead": int(
        os.getenv("GOLEM_ATTENTION_CLUSTER_PV_MATRIX_LOOKAHEAD", "0")
    ),
    "attention_pv_compact_input": int(
        os.getenv("GOLEM_ATTENTION_PV_COMPACT_INPUT", "0")
    ),
    "attention_pv_input_residency": int(
        os.getenv("GOLEM_ATTENTION_PV_INPUT_RESIDENCY", "0")
    ),
    "attention_o_accumulator_cbuffer": int(
        os.getenv("GOLEM_ATTENTION_O_ACCUMULATOR_CBUFFER", "0")
    ),
    "attention_pv_restore_pipeline": int(
        os.getenv("GOLEM_ATTENTION_PV_RESTORE_PIPELINE", "0")
    ),
    "attention_pv_output_pipeline": int(
        os.getenv("GOLEM_ATTENTION_PV_OUTPUT_PIPELINE", "0")
    ),
    "attention_pv_o_row_fusion": int(
        os.getenv("GOLEM_ATTENTION_PV_O_ROW_FUSION", "0")
    ),
    "attention_pv_early_compute": int(
        os.getenv("GOLEM_ATTENTION_PV_EARLY_COMPUTE", "0")
    ),
    "attention_pv_matrix_softmax_overlap": int(
        os.getenv("GOLEM_ATTENTION_PV_MATRIX_SOFTMAX_OVERLAP", "0")
    ),
    "attention_pv_active_k": int(
        os.getenv("GOLEM_ATTENTION_PV_ACTIVE_K", "0")
    ),
    "attention_qk_dataflow_transpose": int(
        os.getenv("GOLEM_ATTENTION_QK_DATAFLOW_TRANSPOSE", "0")
    ),
    "attention_qk_early_compute": int(
        os.getenv("GOLEM_ATTENTION_QK_EARLY_COMPUTE", "0")
    ),
    "attention_qk_input_pipeline": int(
        os.getenv("GOLEM_ATTENTION_QK_INPUT_PIPELINE", "0")
    ),
    "attention_qk_readout_overlap": int(
        os.getenv("GOLEM_ATTENTION_QK_READOUT_OVERLAP", "0")
    ),
    "attention_qk_readout_window": int(
        os.getenv("GOLEM_ATTENTION_QK_READOUT_WINDOW", "2")
    ),
    "attention_qk_panel_row_burst": int(
        os.getenv("GOLEM_ATTENTION_QK_PANEL_ROW_BURST", "0")
    ),
    "attention_cross_tile_operand_pipeline": int(
        os.getenv("GOLEM_ATTENTION_CROSS_TILE_OPERAND_PIPELINE", "0")
    ),
    "attention_operand_context_banks": array_operand_context_banks,
    "attention_qk_matrix_broadcast": int(
        os.getenv("GOLEM_ATTENTION_QK_MATRIX_BROADCAST", "0")
    ),
    "attention_pv_matrix_broadcast": int(
        os.getenv("GOLEM_ATTENTION_PV_MATRIX_BROADCAST", "0")
    ),
    "attention_generic_gemm_enable": attention_generic_gemm_enable,
    "attention_milestone_trace": int(
        os.getenv("GOLEM_ATTENTION_MILESTONE_TRACE", "0")
    ),
    "attention_tile_trace": int(os.getenv("GOLEM_ATTENTION_TILE_TRACE", "0")),
}

arrayParams = {
    "arrayLatency": array_latency,
    "clock": array_clock,
    "modeledComputeCycles": mvm_latency_compute_cycles,
    "arrayMacPerCuPerCycle": array_mac_per_cu_per_cycle,
    "arrayPipelineDepth": array_pipeline_depth,
    "max_instructions": 8,
    "verbose": 0,
    "mmioAddr": 0,
    "numArrays": num_arrays,
    "arrayInputSize": runtime_array_input_size,
    "arrayOutputSize": array_output_size,
    "arrayBufferBaseLatencyCycles": array_buffer_base_latency_cycles,
    "arrayBufferBytesPerCycle": array_buffer_bytes_per_cycle,
    "arrayBufferPorts": array_buffer_ports,
    "operandContextBanks": array_operand_context_banks,
    "attention_cluster_qk_arrays": attention_cluster_qk_arrays,
    "arrayBufferQueueDepth": array_buffer_queue_depth,
    "arrayOutputReadCredits": array_output_read_credits,
    "arrayOutputReadBanks": array_output_read_banks,
    "attentionNearArrayOutputBytesPerCycle": attention_near_array_output_bytes_per_cycle,
    "attentionNearArrayOutputCredits": attention_near_array_output_credits,
    "matrixBroadcastMaxFanout": matrix_broadcast_max_fanout,
    "matrixBroadcastBytesPerCycle": matrix_broadcast_bytes_per_cycle,
    "matrixBroadcastBaseLatencyCycles": matrix_broadcast_base_latency_cycles,
    "matrixBroadcastStageLatencyCycles": matrix_broadcast_stage_latency_cycles,
    # "CrossSimJSONParameters" : crosssim_json_params
}

# globalmemory参数设置
_global_stride_kb = int(os.getenv("GOLEM_GLOBAL_STRIDE_KB", "64"), 0)
GLOBAL_STRIDE = int(
    os.getenv("GOLEM_GLOBAL_STRIDE_BYTES", str(_global_stride_kb * 1024)), 0
)
GLOBAL_BASE = 0x00000  # 0x00000
GLOBAL_TOTAL = numCpus * GLOBAL_STRIDE
GLOBAL_END = GLOBAL_BASE + GLOBAL_TOTAL - 1  # inclusive

# Split base for RoCC address decoding: addresses >= SPLIT_BASE are treated as
# shared buffer / identity-mapped host memory (VA==PA window configured above).
# Identity Window: 0x04000000 - 0x0fffffff (64MB - 256MB, Node 1-3)
# 默认值 64MB，可通过环境变量 GOLEM_IDENTITY_BASE 配置
SPLIT_BASE = int(os.getenv("GOLEM_IDENTITY_BASE", "0x04000000"), 0)

if verbosity > 0:
    print(
        f"[GOLEM] Array config: input={runtime_array_input_size}, output={array_output_size}, numArrays={num_arrays}"
    )
    print(f"[GOLEM] Matmul dtype={matmul_dtype}")
    print(f"[GOLEM] RoCC type={rocc_type}")
    print(f"[GOLEM] Array type={array_type}")
    print(f"[GOLEM] GlobalMemory per-core stride={GLOBAL_STRIDE} bytes")
    print(f"[GOLEM] GlobalMemory link_bw={gm_link_bw}")
    print(f"[GOLEM] GlobalMemory link buffer_length={gm_buffer_length}")
    print(f"[GOLEM] GlobalMemory dma_read_max_inflight={gm_dma_max_inflight}")
    print(f"[GOLEM] GlobalMemory dma_read_retry_ticks={gm_dma_retry_ticks}")
    print(f"[GOLEM] GlobalMemory dma_read_max_retries={gm_dma_max_retries}")
    print(f"[GOLEM] GlobalMemory dma_burst_bytes={gm_dma_burst_bytes}")
    print(f"[GOLEM] Scheduler panel_chunk_bytes={request_scheduler_panel_chunk_bytes}")
    print(
        f"[GOLEM] Scheduler node_chunk_credit={request_scheduler_initial_chunk_credit} "
        f"(residentK={request_sched_resident_k_tiles}, active_workers={request_sched_active_workers}, "
        f"window_buffers={request_sched_window_buffers})"
    )
    print(f"[GOLEM] Scheduler issue_budget_per_tick={request_scheduler_issue_budget_per_tick}")
    print(f"[GOLEM] GlobalMemory trans_latency={gm_trans_latency}")
    print(
        f"[GOLEM] MVM op latency cycles: gm2imat={mvm_latency_gm2imat}, gm2ivec={mvm_latency_gm2ivec}, ovec2gm={mvm_latency_ovec2gm}"
    )
    print(
        f"[GOLEM] MVM compute latency cycles={mvm_latency_compute_cycles} (array_clock={array_clock}, arrayLatency={array_latency})"
    )
    print(
        f"[GOLEM] CU latency model: num_cu={array_num_cu}, mac_per_cu_per_cycle={array_mac_per_cu_per_cycle}, "
        f"pipeline_depth={array_pipeline_depth}, physical_input_size={array_input_size}, "
        f"logical_block_k={matmul_block_k}, active_k_per_micro_step={runtime_array_active_k}"
    )
    print(f"[GOLEM] GlobalMemory retry_tick_cpu_cycles={gm_retry_tick_cpu_cycles}")
    print(
        f"[GOLEM] Progress heartbeat={progress_heartbeat}, interval_cycles={progress_interval_cycles}, gm_interval_ticks={gm_progress_interval_ticks}"
    )
    print(f"[GOLEM] RoCC verbose level={rocc_verbose}")
    print(f"[GOLEM] GlobalMemory verbose level={gm_verbose}")
    print(f"[GOLEM] GlobalMemory data dump={gm_dump_data}")
    print(
        f"[GOLEM] Trace switches: dma={os.getenv('GOLEM_DMA_TRACE', '0')}, "
        f"scheduler={request_scheduler_trace}, llsc={llsc_trace}"
    )

# Keep DMA MMIO windows in low, OS-unused physical space just above GlobalMemory.
DMA_MMIO_BASE = GLOBAL_END + 1

roccParams.update(roccarrayParams)
arrayParams.update(roccarrayParams)
roccParams.update(arrayParams)

l1dcacheParams = {
    "access_latency_cycles": "2",
    "cache_frequency": cpu_clock,
    "replacement_policy": "lru",
    "coherence_protocol": protocol,
    "cache_type": "inclusive",  # L1 必须是 inclusive（SST 硬性要求）
    "associativity": "8",
    "cache_line_size": "64",
    "cache_size": "32 KB",
    "L1": "1",
    "debug": mh_debug,
    "debug_level": mh_debug_level,
    "mshr_num_entries": "32",  # 增加 MSHR 条目数
    "mshr_latency_cycles": "2",  # MSHR 访问延迟
}

l1icacheParams = {
    "access_latency_cycles": "2",
    "cache_frequency": cpu_clock,
    "replacement_policy": "lru",
    "coherence_protocol": protocol,
    "cache_type": "inclusive",  # L1 必须是 inclusive（SST 硬性要求）
    "associativity": "8",
    "cache_line_size": "64",
    "cache_size": "32 KB",
    "prefetcher": "cassini.NextBlockPrefetcher",
    "prefetcher.reach": 1,
    "L1": "1",
    "debug": mh_debug,
    "debug_level": mh_debug_level,
    "mshr_num_entries": "16",  # I-cache 需要较少 MSHR
    "mshr_latency_cycles": "2",
}

l2cacheParams = {
    "access_latency_cycles": "14",
    "cache_frequency": cpu_clock,
    "replacement_policy": "lru",
    "coherence_protocol": protocol,
    "cache_type": "inclusive",  # L2 使用 inclusive
    "associativity": "16",
    "cache_line_size": "64",
    "cache_size": "1MB",
    "mshr_num_entries": "64",  # L2 需要更多 MSHR 处理多核请求
    "mshr_latency_cycles": 3,
    "debug": mh_debug,
    "debug_level": mh_debug_level,
}
busParams = {
    "bus_frequency": cpu_clock,
}

l2memLinkParams = {
    "group": 1,
    "destinations": memory_destinations,
    "network_bw": "50GB/s",  # 增加到 50GB/s 与 NoC 保持一致
    "num_vns": 3,  # 与 NoC 的 num_vns 保持一致
}


class CPU_Builder:
    def __init__(self):
        pass

    # CPU
    def build(
        self,
        prefix,
        nodeId,
        cpuId,
        add_l2_cache: bool = True,
        add_rocc_golem: bool = True,
        rank=None,
        thread=0,
    ):
        if pythonDebug:
            print(f"build {prefix} (L2: {add_l2_cache}, Golem: {add_rocc_golem})")

        # CPU
        cpu = sst.Component(prefix, vanadis_cpu_type)
        if rank is not None:
            cpu.setRank(rank, thread)
        cpu.addParams(cpuParams)
        cpu.addParam("core_id", cpuId)
        if enable_all_stats:
            cpu.enableAllStatistics()

        # CPU.decoder (总是构建)
        for n in range(numThreads):
            decode = cpu.setSubComponent("decoder" + str(n), vanadis_decoder)
            decode.addParams(decoderParams)
            if enable_all_stats:
                decode.enableAllStatistics()
            os_hdlr = decode.setSubComponent("os_handler", vanadis_os_hdlr)
            os_hdlr.addParams(osHdlrParams)
            branch_pred = decode.setSubComponent(
                "branch_unit", "vanadis.VanadisBasicBranchUnit"
            )
            branch_pred.addParams(branchPredParams)
            if enable_all_stats:
                branch_pred.enableAllStatistics()

        # CPU.lsq (总是构建)
        cpu_lsq = cpu.setSubComponent("lsq", "vanadis.VanadisBasicLoadStoreQueue")
        cpu_lsq.addParams(lsqParams)
        if enable_all_stats:
            cpu_lsq.enableAllStatistics()

        # Processors to L1 bus (总是构建)
        processor_bus = sst.Component(prefix + ".processorBus", "memHierarchy.Bus")
        if rank is not None:
            processor_bus.setRank(rank, thread)
        processor_bus.addParams(busParams)

        # CPU.lsq mem interface (总是构建)
        cpuDcacheIf = cpu_lsq.setSubComponent(
            "memory_interface", "memHierarchy.standardInterface"
        )

        # CPU.mem interface for I-cache (总是构建)
        cpuIcacheIf = cpu.setSubComponent(
            "mem_interface_inst", "memHierarchy.standardInterface"
        )

        # <--- Conditional RoCC/Golem block --->
        groupCtrl = None
        requestScheduler = None
        if add_rocc_golem:
            if pythonDebug:
                print(f"  {prefix}: Building RoCC/Golem components...")

            cpu_rocc = cpu.setSubComponent("rocc", rocc_type, 0)
            cpu_rocc.addParams(roccParams)
            # Keep the storage width explicit on the RoCC instance.  This is
            # required for the legacy direct path, which serializes array
            # outputs itself instead of going through the WCP capture path.
            cpu_rocc.addParam("inputOperandSize", matmul_elem_bytes)
            cpu_rocc.addParam("outputOperandSize", matmul_elem_bytes)
            cpu_rocc.addParam("core_id", cpuId)
            cpu_rocc.addParam("globalMemBase", GLOBAL_BASE)
            cpu_rocc.addParam("globalMemStride", GLOBAL_STRIDE)
            cpu_rocc.addParam(
                "globalMemEnd", hex(GLOBAL_BASE + numCpus * GLOBAL_STRIDE - 1)
            )
            cpu_rocc.addParam("groupCtrlEnable", ctrl_link_enable)
            cpu_rocc.addParam("requestSchedulerEnable", request_scheduler_enable)
            cpu_rocc.addParam("sfuEnable", 1 if sfu_enable else 0)
            cpu_rocc.addParam("active_worker_cores", active_worker_cores)
            cpu_rocc.addParam("groupCtrlLatency", ctrl_link_latency)
            cpu_rocc.addParam("groupCtrlQueueDepth", ctrl_link_queue_depth)
            cpu_rocc.addParam(
                "groupCtrlMaxGrantsPerSchedule", ctrl_link_max_grants_per_schedule
            )
            cpu_rocc.addParam(
                "groupCtrlNumMemoryNodes", os.getenv("GOLEM_NUM_MEMORY_NODES", "5")
            )
            cpu_rocc.addParam("groupCtrlVerbose", ctrl_link_verbose)
            cpu_rocc.addParam("hbmBase", hex(SPLIT_BASE))
            cpu_rocc.addParam("hbmPhysBase", hex(SPLIT_BASE))
            cpu_rocc.addParam(
                "progress_total_mvm_ops", _expected_mvm_ops_for_core(cpuId)
            )
            if enable_all_stats:
                cpu_rocc.enableAllStatistics()

            computeArray = cpu_rocc.setSubComponent("array", array_type)
            computeArray.addParams(arrayParams)
            computeArray.addParam("inputOperandSize", matmul_elem_bytes)
            computeArray.addParam("outputOperandSize", matmul_elem_bytes)
            computeArray.addParam("core_id", cpuId)
            computeArray.addParam("mvm_dump_enable", mvm_dump_enable)
            computeArray.addParam("mvm_dump_dir", mvm_dump_dir)
            computeArray.addParam("mvm_dump_mode", mvm_dump_mode)
            computeArray.addParam("functionalCompute", functional_compute)
            if enable_all_stats:
                computeArray.enableAllStatistics()

            GlobalMemory = cpu_rocc.setSubComponent(
                "global_memory", "golem.GlobalMemory"
            )
            gm_params = {
                "baseAddr": hex(GLOBAL_BASE + cpuId * GLOBAL_STRIDE),
                "size": hex(GLOBAL_STRIDE),
                "src_id": cpuId,
                "link_bw": gm_link_bw,
                "buffer_length": gm_buffer_length,
                "num_vns": 3,  # 与 NoC 保持一致
                "dma_write_vn": dma_write_vn,
                "network_vn_priority_order": noc_vn_priority_order,
                "network_vn_starvation_vn": noc_vn_starvation_vn,
                "network_vn_max_starvation_cycles": noc_vn_max_starvation_cycles,
                "identityWindowBase": hex(SPLIT_BASE),
                "dma_read_max_inflight": gm_dma_max_inflight,
                "dma_read_retry_ticks": gm_dma_retry_ticks,
                "dma_read_max_retries": gm_dma_max_retries,
                "dma_burst_bytes": gm_dma_burst_bytes,
                "globalMemTransLatency": gm_trans_latency,
                "attention_cluster_enable": int(
                    os.getenv("GOLEM_ATTENTION_CLUSTER_ENABLE", "0")
                ),
                "local_access_clock": cpu_clock,
                "local_access_base_latency_cycles": local_gm_base_latency_cycles,
                "local_access_bytes_per_cycle": local_gm_bytes_per_cycle,
                "local_access_read_ports": local_gm_read_ports,
                "local_access_write_ports": local_gm_write_ports,
                "local_access_queue_depth": local_gm_queue_depth,
                "local_access_max_request_bytes": local_gm_max_request_bytes,
                "dma_retry_tick_cpu_cycles": gm_retry_tick_cpu_cycles,
                "dma_progress_heartbeat": progress_heartbeat,
                "dma_progress_interval_ticks": gm_progress_interval_ticks,
                "dma_progress_total_chunks": _expected_dma_chunks_for_core(cpuId),
                "verbose": gm_verbose,
                "dump_data": gm_dump_data,
                "dma_trace": os.getenv("GOLEM_DMA_TRACE", "0"),
                "memoryRouters": os.getenv("GOLEM_MEMORY_ROUTERS", ""),
            }
            mem_node_size = os.getenv("GOLEM_MEM_NODE_SIZE", "")
            if mem_node_size:
                gm_params["memNodeSize"] = mem_node_size
            if sfu_reduction_vn:
                gm_params["reduction_vn"] = sfu_reduction_vn
            GlobalMemory.addParams(gm_params)
            if enable_all_stats:
                GlobalMemory.enableAllStatistics()

            if sfu_enable:
                sfu = cpu_rocc.setSubComponent("sfu", "golem.SFU")
                sfu.addParams(
                    {
                        "core_id": cpuId,
                        "active_worker_cores": active_worker_cores,
                        "max_inflight": sfu_max_inflight,
                        "stats_latency": sfu_stats_latency,
                        "merge_latency": sfu_merge_latency,
                        "normalize_latency": sfu_normalize_latency,
                        "accelerator_clock_hz": int(_parse_frequency_hz(cpu_clock)),
                        "vector_lanes": sfu_vector_lanes,
                        "exp_lanes": sfu_exp_lanes,
                        "reduction_tree_latency": sfu_reduction_tree_latency,
                        "exp_latency": sfu_exp_latency,
                        "reciprocal_latency": sfu_reciprocal_latency,
                        "row_contexts": sfu_row_contexts,
                        "attention_kv_pair_reuse": attention_kv_pair_reuse,
                        "attention_kv_query_group_size": attention_kv_query_group_size,
                        "attention_cluster_enable": int(
                            os.getenv("GOLEM_ATTENTION_CLUSTER_ENABLE", "0")
                        ),
                        "scratchpad_bytes": sfu_scratchpad_bytes,
                        "distributed_reduction_transport": sfu_distributed_reduction_transport,
                        "verbose": sfu_verbose,
                    }
                )
                if enable_all_stats:
                    sfu.enableAllStatistics()

            if ctrl_link_enable:
                groupCtrl = cpu_rocc.setSubComponent(
                    "group_ctrl", "golem.GroupCtrlEndpoint"
                )
                groupCtrl.addParams(
                    {
                        "core_id": cpuId,
                        "group_id": cpuId % 4,
                        "worker_slot": -1 if cpuId < 4 else (cpuId // 4) - 1,
                        "role": "manager" if cpuId < 4 else "worker",
                        "queue_depth": ctrl_link_queue_depth,
                        "max_grants_per_schedule": ctrl_link_max_grants_per_schedule,
                        "num_memory_nodes": os.getenv("GOLEM_NUM_MEMORY_NODES", "5"),
                        "ctrl_latency": ctrl_link_latency,
                        "gm_base_addr": GLOBAL_BASE + cpuId * GLOBAL_STRIDE,
                        "gm_size": GLOBAL_STRIDE,
                        "attention_kv_distribution_enable": attention_kv_distribution_enable,
                        "attention_kv_manager_lookahead": int(
                            os.getenv("GOLEM_ATTENTION_KV_MANAGER_LOOKAHEAD", "0")
                        ),
                        "attention_kv_distribution_slots": int(
                            os.getenv("GOLEM_ATTENTION_KV_DISTRIBUTION_SLOTS", "2")
                        ),
                        "attention_kv_distribution_tile_bytes": int(
                            os.getenv("GOLEM_ATTENTION_KV_DISTRIBUTION_TILE_BYTES", "16384")
                        ),
                        "attention_kv_distribution_scratch_offset": int(
                            os.getenv("GOLEM_ATTENTION_KV_DISTRIBUTION_SCRATCH_OFFSET", "0x40000"), 0
                        ),
                        "attention_kv_distribution_expected_workers": int(
                            os.getenv("GOLEM_ATTENTION_KV_DISTRIBUTION_EXPECTED_WORKERS", "4")
                        ),
                        "verbose": ctrl_link_verbose,
                    }
                )
                if enable_all_stats:
                    groupCtrl.enableAllStatistics()

            if request_scheduler_enable:
                requestScheduler = cpu_rocc.setSubComponent(
                    "request_scheduler", "golem.RequestSchedulerEndpoint"
                )
                requestScheduler.addParams(
                    {
                        "core_id": cpuId,
                        "group_id": cpuId % 4,
                        "worker_slot": -1 if cpuId < 4 else (cpuId // 4) - 1,
                        "role": "manager" if cpuId < 4 else "worker",
                        "queue_depth": request_scheduler_queue_depth,
                        "panel_chunk_bytes": request_scheduler_panel_chunk_bytes,
                        "worker_credit_cap": request_scheduler_worker_credit_cap,
                        "manager_issue_budget_per_tick": request_scheduler_issue_budget_per_tick,
                        "tile_chunk_quantum": request_scheduler_tile_chunk_quantum,
                        "submit_batch_size": request_scheduler_submit_batch_size,
                        "done_batch_size": request_scheduler_done_batch_size,
                        "prefetch_windows": wcp_prefetch_windows,
                        "local_slot_count": os.getenv("GOLEM_DMA_SLOT_COUNT", "2"),
                        "a_reuse_n_tiles": os.getenv("GOLEM_A_REUSE_N_TILES", "1"),
                        "b_reuse_m_tiles": os.getenv("GOLEM_B_REUSE_M_TILES", "1"),
                        "window_k_tiles": os.getenv("GOLEM_DMA_WINDOW_K_TILES", "4"),
                        "num_memory_nodes": os.getenv("GOLEM_NUM_MEMORY_NODES", "5"),
                        "ctrl_latency": ctrl_link_latency,
                        "gm_base_addr": GLOBAL_BASE + cpuId * GLOBAL_STRIDE,
                        "gm_size": GLOBAL_STRIDE,
                        "link_bw": os.getenv("GOLEM_NOC_LINK_BW", "100GB/s"),
                        "buffer_length": os.getenv("GOLEM_GM_BUFFER_LENGTH", "64KB"),
                        "infer_submit_bytes": 1,
                        "slot0_bytes": request_sched_slot0_bytes,
                        "slot1_bytes": request_sched_slot1_bytes,
                        "verbose": request_scheduler_verbose,
                        "trace_events": request_scheduler_trace,
                        "event_driven_worker": request_scheduler_event_driven_worker,
                        "group_round_robin": request_scheduler_group_round_robin,
                    }
                )
            if worker_command_processor_enable:
                workerCommandProcessor = cpu_rocc.setSubComponent(
                    "worker_command_processor", "golem.WorkerCommandProcessorLocal"
                )
                workerCommandProcessor.addParams(
                    {
                        "verbose": worker_command_processor_verbose,
                        "dtype_is_float": 1 if "Float" in rocc_type else 0,
                        "stage3_trace": os.getenv("GOLEM_WCP_STAGE3_TRACE", "0"),
                        "attention_cluster_qk_arrays": attention_cluster_qk_arrays,
                        "prefetch_windows": wcp_prefetch_windows,
                        "cross_macro_prefetch": wcp_cross_macro_prefetch,
                        "window_k_tiles": os.getenv("GOLEM_DMA_WINDOW_K_TILES", "4"),
                        "gemm_proxy_queue_depth": wcp_gemm_proxy_queue_depth,
                        "gemm_proxy_issue_width": wcp_gemm_proxy_issue_width,
                        "gemm_proxy_command_latency_cycles": wcp_gemm_proxy_command_latency_cycles,
                        "gemm_proxy_completion_latency_cycles": wcp_gemm_proxy_completion_latency_cycles,
                        "c_buffer_bytes": gm_c_buffer_bytes,
                        "c_buffer_read_bytes_per_cycle": gm_c_buffer_read_bpc,
                        "c_buffer_write_bytes_per_cycle": gm_c_buffer_write_bpc,
                        "c_buffer_latency_cycles": gm_c_buffer_latency_cycles,
                        "attention_tile_storage_banks": attention_tile_storage_banks,
                        "attention_tile_storage_bank_bytes_per_cycle": attention_tile_storage_bank_bpc,
                        "final_c_write_enable": final_c_write_enable,
                        "output_mode": output_mode,
                        "fusion_dump_enable": fusion_dump_enable,
                        "fusion_dump_dir": fusion_dump_dir,
                    }
                )
                if enable_all_stats:
                    workerCommandProcessor.enableAllStatistics()

            roccDcacheIf = cpu_rocc.setSubComponent(
                "memory_interface", "memHierarchy.standardInterface"
            )

            link_rocc_l1dcache_link = sst.Link(prefix + ".link_rocc_dbus_link")
            link_rocc_l1dcache_link.connect(
                (roccDcacheIf, "lowlink", "1ns"),
                (processor_bus, "high_network_1", "1ns"),
            )
            link_rocc_l1dcache_link.setNoCut()

        # L1 Caches (总是构建)
        cpu_l1dcache = sst.Component(prefix + ".l1dcache", "memHierarchy.Cache")
        if rank is not None:
            cpu_l1dcache.setRank(rank, thread)
        cpu_l1dcache.addParams(l1dcacheParams)
        cpu_l1dcache.enableStatistics(["CacheHits", "CacheMisses"])
        l1dcache_2_cpu = cpu_l1dcache.setSubComponent(
            "highlink", "memHierarchy.MemLink"
        )

        cpu_l1icache = sst.Component(prefix + ".l1icache", "memHierarchy.Cache")
        if rank is not None:
            cpu_l1icache.setRank(rank, thread)
        cpu_l1icache.addParams(l1icacheParams)
        cpu_l1icache.enableStatistics(["CacheHits", "CacheMisses"])
        l1icache_2_cpu = cpu_l1icache.setSubComponent(
            "highlink", "memHierarchy.MemLink"
        )

        # TLBs (总是构建)
        dtlbWrapper = sst.Component(prefix + ".dtlb", "mmu.tlb_wrapper")
        if rank is not None:
            dtlbWrapper.setRank(rank, thread)
        dtlbWrapper.addParams(tlbWrapperParams)
        dtlb = dtlbWrapper.setSubComponent("tlb", "mmu." + tlbType)
        dtlb.addParams(tlbParams)

        itlbWrapper = sst.Component(prefix + ".itlb", "mmu.tlb_wrapper")
        if rank is not None:
            itlbWrapper.setRank(rank, thread)
        itlbWrapper.addParams(tlbWrapperParams)
        itlbWrapper.addParam("exe", True)
        itlb = itlbWrapper.setSubComponent("tlb", "mmu." + tlbType)
        itlb.addParams(tlbParams)

        # --- L2 Cache or L1-to-Network-NICs ---
        if add_l2_cache:
            # --- CASE 1: WITH L2 (左侧图) ---
            if pythonDebug:
                print(f"  {prefix}: Building L2 Cache. L1s connect to bus.")

            # L1s connect to bus via MemLink
            l1dcache_2_bus = cpu_l1dcache.setSubComponent(
                "lowlink", "memHierarchy.MemLink"
            )
            l1icache_2_bus = cpu_l1icache.setSubComponent(
                "lowlink", "memHierarchy.MemLink"
            )

            # L1 to L2 bus
            cache_bus = sst.Component(prefix + ".bus", "memHierarchy.Bus")
            if rank is not None:
                cache_bus.setRank(rank, thread)
            cache_bus.addParams(busParams)

            # L2 cache
            cpu_l2cache = sst.Component(prefix + ".l2cache", "memHierarchy.Cache")
            if rank is not None:
                cpu_l2cache.setRank(rank, thread)
            cpu_l2cache.addParams(l2cacheParams)
            cpu_l2cache.enableStatistics(["CacheHits", "CacheMisses"])

            l2cache_2_l1caches = cpu_l2cache.setSubComponent(
                "highlink", "memHierarchy.MemLink"
            )
            l2cache_2_mem = cpu_l2cache.setSubComponent(
                "lowlink", "memHierarchy.MemNIC"
            )
            l2cache_2_mem.addParams(l2memLinkParams)

            # L1D -> bus
            link_l1dcache_l2cache_link = sst.Link(
                prefix + ".link_l1dcache_l2cache_link"
            )
            link_l1dcache_l2cache_link.connect(
                (l1dcache_2_bus, "port", "1ns"), (cache_bus, "high_network_0", "1ns")
            )
            link_l1dcache_l2cache_link.setNoCut()

            # L1I -> bus
            link_l1icache_l2cache_link = sst.Link(
                prefix + ".link_l1icache_l2cache_link"
            )
            link_l1icache_l2cache_link.connect(
                (l1icache_2_bus, "port", "1ns"), (cache_bus, "high_network_1", "1ns")
            )
            link_l1icache_l2cache_link.setNoCut()

            # BUS -> L2
            link_bus_l2cache_link = sst.Link(prefix + ".link_bus_l2cache_link")
            link_bus_l2cache_link.connect(
                (cache_bus, "low_network_0", "1ns"), (l2cache_2_l1caches, "port", "1ns")
            )
            link_bus_l2cache_link.setNoCut()

        else:
            # --- CASE 2: NO L2  ---
            if pythonDebug:
                print(f"  {prefix}: No L2 Cache. L1D and L1I get separate MemNICs.")

            # L1D's lowlink is now a NIC
            l1dcache_2_mem = cpu_l1dcache.setSubComponent(
                "lowlink", "memHierarchy.MemNIC"
            )
            l1dcache_2_mem.addParams(l2memLinkParams)  # Use same network params

            # L1I's lowlink is now a NIC
            l1icache_2_mem = cpu_l1icache.setSubComponent(
                "lowlink", "memHierarchy.MemNIC"
            )
            l1icache_2_mem.addParams(l2memLinkParams)  # Use same network params

        # --- Unconditional Links (TLBs, CPU-to-Bus) ---

        # CPU (data) -> processor_bus
        link_lsq_l1dcache_link = sst.Link(prefix + ".link_cpu_dbus_link")
        link_lsq_l1dcache_link.connect(
            (cpuDcacheIf, "lowlink", "1ns"), (processor_bus, "high_network_0", "1ns")
        )
        link_lsq_l1dcache_link.setNoCut()

        # processor_bus -> data TLB
        link_bus_l1cache_link = sst.Link(prefix + ".link_bus_l1cache_link")
        link_bus_l1cache_link.connect(
            (processor_bus, "low_network_0", "1ns"), (dtlbWrapper, "cpu_if", "1ns")
        )
        link_bus_l1cache_link.setNoCut()

        # data TLB -> data L1
        link_cpu_l1dcache_link = sst.Link(prefix + ".link_cpu_l1dcache_link")
        link_cpu_l1dcache_link.connect(
            (dtlbWrapper, "cache_if", "1ns"), (l1dcache_2_cpu, "port", "1ns")
        )
        link_cpu_l1dcache_link.setNoCut()

        # CPU (instruction) -> instruction TLB
        link_cpu_itlb_link = sst.Link(prefix + ".link_cpu_itlb_link")
        link_cpu_itlb_link.connect(
            (cpuIcacheIf, "lowlink", "1ns"), (itlbWrapper, "cpu_if", "1ns")
        )
        link_cpu_itlb_link.setNoCut()

        # instruction TLB -> instruction L1
        link_cpu_l1icache_link = sst.Link(prefix + ".link_cpu_l1icache_link")
        link_cpu_l1icache_link.connect(
            (itlbWrapper, "cache_if", "1ns"), (l1icache_2_cpu, "port", "1ns")
        )
        link_cpu_l1icache_link.setNoCut()

        # 返回接口 (DMA 复用 link_control，所以只有 6 个接口)
        if add_l2_cache and add_rocc_golem:
            if ctrl_link_enable and groupCtrl is not None:
                return (
                    (cpu, "os_link", "5ns"),
                    (l2cache_2_mem, "port", "1ns"),
                    (dtlb, "mmu", "1ns"),
                    (itlb, "mmu", "1ns"),
                    (GlobalMemory, "rtr", "1ns"),
                    (requestScheduler, "rtr", "1ns")
                    if (
                        request_scheduler_enable
                        and cpuId < 4
                        and requestScheduler is not None
                    )
                    else None,
                    groupCtrl,
                    requestScheduler,
                )
            return (
                (cpu, "os_link", "5ns"),
                (l2cache_2_mem, "port", "1ns"),
                (dtlb, "mmu", "1ns"),
                (itlb, "mmu", "1ns"),
                (GlobalMemory, "rtr", "1ns"),
                (requestScheduler, "rtr", "1ns")
                if (
                    request_scheduler_enable
                    and cpuId < 4
                    and requestScheduler is not None
                )
                else None,
                groupCtrl,
                requestScheduler,
            )
        if not add_l2_cache and add_rocc_golem:
            if ctrl_link_enable and groupCtrl is not None:
                return (
                    (cpu, "os_link", "5ns"),
                    (l1icache_2_mem, "port", "1ns"),
                    (l1dcache_2_mem, "port", "1ns"),
                    (dtlb, "mmu", "1ns"),
                    (itlb, "mmu", "1ns"),
                    (GlobalMemory, "rtr", "1ns"),
                    groupCtrl,
                )
            return (
                (cpu, "os_link", "5ns"),
                (l1icache_2_mem, "port", "1ns"),
                (l1dcache_2_mem, "port", "1ns"),
                (dtlb, "mmu", "1ns"),
                (itlb, "mmu", "1ns"),
                (GlobalMemory, "rtr", "1ns"),
            )
        if not add_l2_cache and not add_rocc_golem:
            return (
                (cpu, "os_link", "5ns"),
                (l1icache_2_mem, "port", "1ns"),
                (l1dcache_2_mem, "port", "1ns"),
                (dtlb, "mmu", "1ns"),
                (itlb, "mmu", "1ns"),
            )
        if add_l2_cache and not add_rocc_golem:
            return (
                (cpu, "os_link", "5ns"),
                (l2cache_2_mem, "port", "1ns"),
                (dtlb, "mmu", "1ns"),
                (itlb, "mmu", "1ns"),
            )
