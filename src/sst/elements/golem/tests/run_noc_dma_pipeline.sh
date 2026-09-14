#!/usr/bin/env bash
set -euo pipefail

# 统一运行脚本：
# 1) 设置阵列硬件参数 / RoCC 阵列参数
# 2) 生成 HBM 初始化文件
# 3) 编译 test_noc_dma
# 4) 运行 SST 架构配置脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd -P)"
cd "$SCRIPT_DIR"

# The wrapper performs preparation and post-processing once.  Calling the
# wrapper itself under mpirun would duplicate those steps and race on HBM files.
if [[ "${OMPI_COMM_WORLD_SIZE:-1}" != "1" || "${PMI_SIZE:-1}" != "1" ]]; then
	echo "[ERROR] Do not launch run_noc_dma_pipeline.sh under mpirun; use --mpi-ranks N or GOLEM_MPI_RANKS=N." >&2
	exit 2
fi

RUN_START_EPOCH="$(date +%s)"
RUN_ID="${GOLEM_RUN_ID:-run_$(date +%Y%m%d_%H%M%S)_$$}"
if [[ "$(basename "$REPO_ROOT")" == "sst-elements" &&
      "$(basename "$(dirname "$REPO_ROOT")")" == "build" ]]; then
	REPO_ROOT="$(cd "$REPO_ROOT/../.." && pwd -P)"
fi
WORKTREE_ROOT="$REPO_ROOT"
GOLEM_PRIVATE_DRAMSIM3_LIB_DIR="${GOLEM_PRIVATE_DRAMSIM3_LIB_DIR:-$REPO_ROOT/install/lib/dramsim3}"
GOLEM_PRIVATE_RAMULATOR2_LIB_DIR="${GOLEM_PRIVATE_RAMULATOR2_LIB_DIR:-$REPO_ROOT/deps/ramulator2-2.1}"
DRAMSIM3_LIB_DIR="${DRAMSIM3_LIB_DIR:-${SST_DRAMSIM3_PREFIX:-/local/packages/dramsim3}}"
RISCV_MUSL_TOOLCHAIN_BIN="${RISCV_MUSL_TOOLCHAIN_BIN:-/local/scratch/src/riscv64-linux-musl-cross/bin}"
SST_CORE_HOME="${SST_CORE_HOME:-/local/sstcore}"
SST_ELEMENTS_HOME="${SST_ELEMENTS_HOME:-$REPO_ROOT/install}"
SST_BUILD_LIB_PATH="${SST_BUILD_LIB_PATH:-$REPO_ROOT/build/sst-elements/src/sst/elements/golem/.libs}"
SST_INSTALL_LIB_PATH="${SST_INSTALL_LIB_PATH:-$REPO_ROOT/install/lib/sst-elements-library}"
if [[ -z "${SST_LIB_PATH+x}" ]]; then
	if [[ -f "$SST_BUILD_LIB_PATH/libgolem.so" ]]; then
		SST_LIB_PATH="$SST_BUILD_LIB_PATH"
	else
		SST_LIB_PATH="$SST_INSTALL_LIB_PATH"
	fi
fi
CONDA_LIB_DIR="${CONDA_LIB_DIR:-/usr/lib/x86_64-linux-gnu}"
REAL_SST_BIN="${REAL_SST_BIN:-$SST_CORE_HOME/bin/sst}"
export RISCV_MUSL_TOOLCHAIN_BIN

if [[ -f "$GOLEM_PRIVATE_DRAMSIM3_LIB_DIR/libdramsim3.so" ]]; then
	export LD_LIBRARY_PATH="$GOLEM_PRIVATE_DRAMSIM3_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
elif [[ -d "$DRAMSIM3_LIB_DIR" ]]; then
	export LD_LIBRARY_PATH="$DRAMSIM3_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ -f "$GOLEM_PRIVATE_RAMULATOR2_LIB_DIR/libramulator.so" ]]; then
	export LD_LIBRARY_PATH="$GOLEM_PRIVATE_RAMULATOR2_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export SST_CORE_HOME
export SST_ELEMENTS_HOME
export SST_LIB_PATH
export SST_SOFTMAX_LD_LIBRARY_PATH="${SST_SOFTMAX_LD_LIBRARY_PATH:-$CONDA_LIB_DIR:$SST_LIB_PATH:$SST_INSTALL_LIB_PATH:$SST_CORE_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
export LD_LIBRARY_PATH="$SST_SOFTMAX_LD_LIBRARY_PATH"

DEFAULT_PRESET_FILE="$SCRIPT_DIR/configs/default.env"
AUTO_PRESET_FILE=""
CALLER_SET_GOLEM_ARCH_SCRIPT=0
CALLER_GOLEM_ARCH_SCRIPT=""
if [[ -n "${GOLEM_ARCH_SCRIPT+x}" ]]; then
	CALLER_SET_GOLEM_ARCH_SCRIPT=1
	CALLER_GOLEM_ARCH_SCRIPT="$GOLEM_ARCH_SCRIPT"
fi
if [[ -f "$DEFAULT_PRESET_FILE" ]]; then
	# shellcheck source=/dev/null
	source "$DEFAULT_PRESET_FILE"
	AUTO_PRESET_FILE="$DEFAULT_PRESET_FILE"
fi
if [[ "$CALLER_SET_GOLEM_ARCH_SCRIPT" -eq 1 ]]; then
	GOLEM_ARCH_SCRIPT="$CALLER_GOLEM_ARCH_SCRIPT"
fi

STATS_DIR_FROM_ENV=0
STATS_FILE_FROM_ENV=0
CORE_MAP_FILE_FROM_ENV=0
MVM_VERIFY_SUMMARY_FROM_ENV=0
MVM_DUMP_DIR_FROM_ENV=0
STDOUT_DIR_FROM_ENV=0

if [[ -n "${GOLEM_STATS_DIR+x}" ]]; then
	STATS_DIR_FROM_ENV=1
fi
if [[ -n "${GOLEM_STATS_FILE+x}" ]]; then
	STATS_FILE_FROM_ENV=1
fi
if [[ -n "${GOLEM_CORE_MAP_FILE+x}" ]]; then
	CORE_MAP_FILE_FROM_ENV=1
fi
if [[ -n "${GOLEM_MVM_VERIFY_SUMMARY_FILE+x}" ]]; then
	MVM_VERIFY_SUMMARY_FROM_ENV=1
fi
if [[ -n "${GOLEM_MVM_DUMP_DIR+x}" ]]; then
	MVM_DUMP_DIR_FROM_ENV=1
fi
if [[ -n "${GOLEM_STDOUT_DIR+x}" ]]; then
	STDOUT_DIR_FROM_ENV=1
fi

ARTIFACT_ROOT="${GOLEM_ARTIFACT_ROOT:-$SCRIPT_DIR/artifacts}"
LOG_DIR="${GOLEM_LOG_DIR:-$ARTIFACT_ROOT/logs}"
HBM_DIR="${GOLEM_HBM_DIR:-$ARTIFACT_ROOT/hbm}"
STDOUT_DIR="${GOLEM_STDOUT_DIR:-$ARTIFACT_ROOT/stdout}"
STATS_DIR="${GOLEM_STATS_DIR:-$ARTIFACT_ROOT/stats}"
GOLEM_MVM_DUMP_DIR="${GOLEM_MVM_DUMP_DIR:-$ARTIFACT_ROOT/mvm_dumps}"
GOLEM_SST_WORK_DIR="${GOLEM_SST_WORK_DIR:-$ARTIFACT_ROOT/work/$RUN_ID}"
STATS_FILE="${GOLEM_STATS_FILE:-$STATS_DIR/stats_selfcom.txt}"
CORE_MAP_FILE="${GOLEM_CORE_MAP_FILE:-$STATS_DIR/core_memory_map.csv}"
MVM_VERIFY_SUMMARY_FILE="${GOLEM_MVM_VERIFY_SUMMARY_FILE:-$STATS_DIR/mvm_verify_summary.csv}"
RUN_SUMMARY_CSV="${GOLEM_RUN_SUMMARY_CSV:-$ARTIFACT_ROOT/stats/run_summary.csv}"
TEST_BINARY="${GOLEM_TEST_BINARY:-${VANADIS_EXE:-$SCRIPT_DIR/small/mvm_noc_int_array/riscv64/test_noc_dma}}"
if [[ "$TEST_BINARY" != /* ]]; then
	TEST_BINARY="$SCRIPT_DIR/$TEST_BINARY"
fi
BUILD_METADATA_FILE="${GOLEM_BUILD_METADATA_FILE:-${TEST_BINARY}.build.env}"
if [[ "$BUILD_METADATA_FILE" != /* ]]; then
	BUILD_METADATA_FILE="$SCRIPT_DIR/$BUILD_METADATA_FILE"
fi
DRAMSIM_STATS_DIR="$STATS_DIR/dramsim3"
EXEC_SUMMARY_FILE="$STATS_DIR/execution_summary.csv"
EXEC_DEBUG_SUMMARY_FILE="$STATS_DIR/execution_debug_summary.csv"
DMA_SUMMARY_FILE="$STATS_DIR/dma_summary.csv"
NOC_SUMMARY_FILE="$STATS_DIR/noc_summary.csv"
MEMORY_SUMMARY_FILE="$STATS_DIR/memory_summary.csv"
HBM_READ_COMMAND_SUMMARY_FILE="$STATS_DIR/hbm_read_command_summary.csv"
HBM_READ_COMMAND_NODE_FILE="$STATS_DIR/hbm_read_command_nodes.csv"
NOC_LATENCY_SUMMARY_FILE="$STATS_DIR/noc_latency_summary.csv"
MEMORY_QUEUE_SUMMARY_FILE="$STATS_DIR/memory_queue_summary.csv"
CAUSAL_SUMMARY_FILE="$STATS_DIR/submit_ready_causal_summary.csv"
CAUSAL_TABLE_FILE="$STATS_DIR/submit_ready_causal_table.csv"
SCHED_PRESSURE_SUMMARY_FILE="$STATS_DIR/sched_pressure_summary.csv"
SCHED_PRESSURE_TABLE_FILE="$STATS_DIR/sched_pressure_table.csv"
CREDIT_OWNER_SUMMARY_FILE="$STATS_DIR/credit_owner_summary.csv"
CREDIT_OWNER_TABLE_FILE="$STATS_DIR/credit_owner_table.csv"
WCP_WINDOW_BREAKDOWN_FILE="$STATS_DIR/wcp_window_breakdown.csv"
WCP_WINDOW_CORE_SUMMARY_FILE="$STATS_DIR/wcp_window_core_summary.csv"
TERMINAL_SUMMARY_FILE="$STATS_DIR/terminal_summary.csv"
NOC_HOTSPOT_SUMMARY_FILE="$STATS_DIR/noc_hotspot_summary.csv"
NOC_HOTSPOT_ROUTER_FILE="$STATS_DIR/noc_hotspot_router_table.csv"
NOC_HOTSPOT_PORT_FILE="$STATS_DIR/noc_hotspot_port_table.csv"
EXEC_SUMMARY_FILE="$STATS_DIR/execution_summary.csv"
DMA_SUMMARY_FILE="$STATS_DIR/dma_summary.csv"
NOC_SUMMARY_FILE="$STATS_DIR/noc_summary.csv"
MEMORY_SUMMARY_FILE="$STATS_DIR/memory_summary.csv"

# ===== 默认值（可被环境变量或命令行覆盖） =====
GOLEM_TOTAL_GROUPS="${GOLEM_TOTAL_GROUPS:-4}"

GOLEM_ARRAY_INPUT_SIZE="${GOLEM_ARRAY_INPUT_SIZE:-4}"
GOLEM_ARRAY_OUTPUT_SIZE="${GOLEM_ARRAY_OUTPUT_SIZE:-4}"
GOLEM_NUM_ARRAYS="${GOLEM_NUM_ARRAYS:-1}"
GOLEM_TOTAL_CORES="${GOLEM_TOTAL_CORES:-${VANADIS_NUM_CORES:-16}}"
GOLEM_TOTAL_GEMM_CORES="${GOLEM_TOTAL_GEMM_CORES:-16}"
GOLEM_NUM_MEMORY_NODES="${GOLEM_NUM_MEMORY_NODES:-5}"
GOLEM_MEMORY_LAYOUT="${GOLEM_MEMORY_LAYOUT:-top_hbm}"
GOLEM_MEM_NODE_SIZE_BYTES="${GOLEM_MEM_NODE_SIZE_BYTES:-67108864}"
GOLEM_GLOBAL_STRIDE_KB="${GOLEM_GLOBAL_STRIDE_KB:-64}"
GOLEM_GM_C_BUFFER_BYTES="${GOLEM_GM_C_BUFFER_BYTES:-1048576}"
GOLEM_GM_C_BUFFER_READ_BPC="${GOLEM_GM_C_BUFFER_READ_BPC:-256}"
GOLEM_GM_C_BUFFER_WRITE_BPC="${GOLEM_GM_C_BUFFER_WRITE_BPC:-256}"
GOLEM_GM_C_BUFFER_LATENCY_CYCLES="${GOLEM_GM_C_BUFFER_LATENCY_CYCLES:-1}"
GOLEM_FINAL_C_WRITE_ENABLE="${GOLEM_FINAL_C_WRITE_ENABLE:-1}"
GOLEM_OUTPUT_MODE="${GOLEM_OUTPUT_MODE:-}"
GOLEM_DMA_WRITE_VN="${GOLEM_DMA_WRITE_VN:-2}"
GOLEM_DMA_STAGGER_CYCLES="${GOLEM_DMA_STAGGER_CYCLES:-0}"
GOLEM_DMA_OVERLAP="${GOLEM_DMA_OVERLAP:-0}"
GOLEM_CTRL_OVERLAP_AB="${GOLEM_CTRL_OVERLAP_AB:-1}"
GOLEM_GROUP_MANAGER_ENABLE="${GOLEM_GROUP_MANAGER_ENABLE:-1}"
GOLEM_CTRL_LINK_ENABLE="${GOLEM_CTRL_LINK_ENABLE:-0}"
GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE="${GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE:-0}"
GOLEM_WORKER_START_BARRIER_ENABLE="${GOLEM_WORKER_START_BARRIER_ENABLE:-1}"
GOLEM_WORKER_START_GUARD_CYCLES="${GOLEM_WORKER_START_GUARD_CYCLES:-32768}"
GOLEM_A_REUSE_N_TILES="${GOLEM_A_REUSE_N_TILES:-1}"
GOLEM_B_REUSE_M_TILES="${GOLEM_B_REUSE_M_TILES:-1}"
GOLEM_ARCH_SCRIPT="${GOLEM_ARCH_SCRIPT:-architecture/ncores_selfcom_dma_ctrl.py}"
GOLEM_MEMORY_BACKEND="${GOLEM_MEMORY_BACKEND:-ramulator2}"
GOLEM_DRAMSIM3_CONFIG="${GOLEM_DRAMSIM3_CONFIG:-$SCRIPT_DIR/architecture/dram/HBM_4Gb_x128.ini}"
GOLEM_RAMULATOR2_CONFIG="${GOLEM_RAMULATOR2_CONFIG:-$SCRIPT_DIR/architecture/ramulator/hbm2e_2500.yaml}"
GOLEM_DMA_NODE_CREDITS="${GOLEM_DMA_NODE_CREDITS:-4}"
GOLEM_DMA_NODE_CHUNK_CREDITS="${GOLEM_DMA_NODE_CHUNK_CREDITS:-}"
GOLEM_WCP_PREFETCH_WINDOWS="${GOLEM_WCP_PREFETCH_WINDOWS:-2}"
GOLEM_DMA_WINDOW_K_TILES="${GOLEM_DMA_WINDOW_K_TILES:-4}"
GOLEM_SCHED_SUBMIT_BATCH_SIZE="${GOLEM_SCHED_SUBMIT_BATCH_SIZE:-4}"
GOLEM_SCHED_DONE_BATCH_SIZE="${GOLEM_SCHED_DONE_BATCH_SIZE:-8}"
GOLEM_GEMM_M="${GOLEM_GEMM_M:-$GOLEM_ARRAY_OUTPUT_SIZE}"
GOLEM_GEMM_N="${GOLEM_GEMM_N:-$GOLEM_NUM_ARRAYS}"
GOLEM_GEMM_K="${GOLEM_GEMM_K:-$GOLEM_ARRAY_INPUT_SIZE}"
GOLEM_ORIG_M="${GOLEM_ORIG_M:-}"
GOLEM_ORIG_N="${GOLEM_ORIG_N:-}"
GOLEM_ORIG_K="${GOLEM_ORIG_K:-}"
GOLEM_GEMM_BLOCK_M="${GOLEM_GEMM_BLOCK_M:-$GOLEM_ARRAY_OUTPUT_SIZE}"
GOLEM_GEMM_BLOCK_N="${GOLEM_GEMM_BLOCK_N:-$GOLEM_NUM_ARRAYS}"
GOLEM_GEMM_BLOCK_K="${GOLEM_GEMM_BLOCK_K:-$GOLEM_ARRAY_INPUT_SIZE}"
GOLEM_MATMUL_DTYPE="${GOLEM_MATMUL_DTYPE:-fp32}"
GOLEM_MATMUL_TRANSPOSE_B="${GOLEM_MATMUL_TRANSPOSE_B:-0}"
GOLEM_BIAS_ENABLE="${GOLEM_BIAS_ENABLE:-0}"
GOLEM_BIAS_VALUE="${GOLEM_BIAS_VALUE:-0}"
GOLEM_DMA_READ_RETRY_TICKS="${GOLEM_DMA_READ_RETRY_TICKS:-256}"
GOLEM_DMA_READ_MAX_RETRIES="${GOLEM_DMA_READ_MAX_RETRIES:-8}"
GOLEM_DMA_MAX_INFLIGHT="${GOLEM_DMA_MAX_INFLIGHT:-256}"
GOLEM_DMA_BURST_BYTES="${GOLEM_DMA_BURST_BYTES:-16384}"
GOLEM_DMA_PANEL_CHUNK_BYTES="${GOLEM_DMA_PANEL_CHUNK_BYTES:-16384}"
GOLEM_DMA_CREDIT_CHUNK_BYTES="${GOLEM_DMA_CREDIT_CHUNK_BYTES:-8192}"
GOLEM_DMA_ADMISSION_LIMIT="${GOLEM_DMA_ADMISSION_LIMIT:-0}"
GOLEM_DMA_WINDOW_PRIORITY_ENABLE="${GOLEM_DMA_WINDOW_PRIORITY_ENABLE:-0}"
GOLEM_DMA_WINDOW_REORDER_CYCLES="${GOLEM_DMA_WINDOW_REORDER_CYCLES:-512}"
GOLEM_DMA_RESPONSE_DRAIN_LIMIT="${GOLEM_DMA_RESPONSE_DRAIN_LIMIT:-0}"
GOLEM_DMA_RESPONSE_TILE_PRIORITY_ENABLE="${GOLEM_DMA_RESPONSE_TILE_PRIORITY_ENABLE:-0}"
GOLEM_DMA_RESPONSE_VN="${GOLEM_DMA_RESPONSE_VN:-1}"
GOLEM_DMA_RESPONSE_REORDER_CYCLES="${GOLEM_DMA_RESPONSE_REORDER_CYCLES:-0}"
GOLEM_DMA_RESPONSE_MAX_STARVATION_CYCLES="${GOLEM_DMA_RESPONSE_MAX_STARVATION_CYCLES:-65536}"
GOLEM_DMA_ADMISSION_MAX_STARVATION_CYCLES="${GOLEM_DMA_ADMISSION_MAX_STARVATION_CYCLES:-4096}"
GOLEM_SCHED_ISSUE_BUDGET_PER_TICK="${GOLEM_SCHED_ISSUE_BUDGET_PER_TICK:-2}"
GOLEM_SCHED_WORKER_CREDIT_CAP="${GOLEM_SCHED_WORKER_CREDIT_CAP:-0}"
GOLEM_DMA_GROUP_RR_ENABLE="${GOLEM_DMA_GROUP_RR_ENABLE:-0}"
GOLEM_DMA_TILE_CHUNK_QUANTUM="${GOLEM_DMA_TILE_CHUNK_QUANTUM:-1}"
GOLEM_WCP_CROSS_MACRO_PREFETCH_ENABLE="${GOLEM_WCP_CROSS_MACRO_PREFETCH_ENABLE:-0}"
GOLEM_LATENCY_MVM_GM2IMAT="${GOLEM_LATENCY_MVM_GM2IMAT:-10}"
GOLEM_LATENCY_MVM_GM2IVEC="${GOLEM_LATENCY_MVM_GM2IVEC:-10}"
GOLEM_LATENCY_MVM_OVEC2GM="${GOLEM_LATENCY_MVM_OVEC2GM:-10}"
GOLEM_ARRAY_NUM_CU="${GOLEM_ARRAY_NUM_CU:-$GOLEM_ARRAY_OUTPUT_SIZE}"
GOLEM_ARRAY_MAC_PER_CU_PER_CYCLE="${GOLEM_ARRAY_MAC_PER_CU_PER_CYCLE:-1}"
GOLEM_ARRAY_PIPELINE_DEPTH="${GOLEM_ARRAY_PIPELINE_DEPTH:-0}"
GOLEM_ARRAY_CLOCK="${GOLEM_ARRAY_CLOCK:-${VANADIS_CPU_CLOCK:-2.3GHz}}"
GOLEM_MEMCTRL_CLOCK="${GOLEM_MEMCTRL_CLOCK:-${VANADIS_CPU_CLOCK:-2.3GHz}}"
GOLEM_ARRAY_BUFFER_BASE_LATENCY_CYCLES="${GOLEM_ARRAY_BUFFER_BASE_LATENCY_CYCLES:-1}"
GOLEM_ARRAY_BUFFER_BYTES_PER_CYCLE="${GOLEM_ARRAY_BUFFER_BYTES_PER_CYCLE:-64}"
GOLEM_ARRAY_BUFFER_PORTS="${GOLEM_ARRAY_BUFFER_PORTS:-1}"
GOLEM_ARRAY_BUFFER_QUEUE_DEPTH="${GOLEM_ARRAY_BUFFER_QUEUE_DEPTH:-64}"
GOLEM_MATRIX_BROADCAST_MAX_FANOUT="${GOLEM_MATRIX_BROADCAST_MAX_FANOUT:-16}"
GOLEM_MATRIX_BROADCAST_BYTES_PER_CYCLE="${GOLEM_MATRIX_BROADCAST_BYTES_PER_CYCLE:-64}"
GOLEM_MATRIX_BROADCAST_BASE_LATENCY_CYCLES="${GOLEM_MATRIX_BROADCAST_BASE_LATENCY_CYCLES:-1}"
GOLEM_MATRIX_BROADCAST_STAGE_LATENCY_CYCLES="${GOLEM_MATRIX_BROADCAST_STAGE_LATENCY_CYCLES:-1}"

# 可选：RoCC/Array 类型
GOLEM_ROCC_TYPE="${GOLEM_ROCC_TYPE:-golem.RoCCAnalogInt}"
GOLEM_ARRAY_TYPE="${GOLEM_ARRAY_TYPE:-golem.MVMIntArray}"

GOLEM_NOC_INPUT_BUF_SIZE="${GOLEM_NOC_INPUT_BUF_SIZE:-8KB}"
GOLEM_NOC_OUTPUT_BUF_SIZE="${GOLEM_NOC_OUTPUT_BUF_SIZE:-8KB}"
GOLEM_NOC_MEMNIC_INPUT_BUF_SIZE="${GOLEM_NOC_MEMNIC_INPUT_BUF_SIZE:-128KB}"
GOLEM_NOC_MEMNIC_OUTPUT_BUF_SIZE="${GOLEM_NOC_MEMNIC_OUTPUT_BUF_SIZE:-128KB}"
GOLEM_NOC_LINK_BW="${GOLEM_NOC_LINK_BW:-25GB/s}"
GOLEM_NOC_XBAR_BW="${GOLEM_NOC_XBAR_BW:-25GB/s}"
GOLEM_NOC_FLIT_SIZE="${GOLEM_NOC_FLIT_SIZE:-128B}"
GOLEM_GM_LINK_BW="${GOLEM_GM_LINK_BW:-128GB/s}"
GOLEM_NOC_VN_PRIORITY_ENABLE="${GOLEM_NOC_VN_PRIORITY_ENABLE:-1}"
GOLEM_NOC_VN_PRIORITY_ORDER="${GOLEM_NOC_VN_PRIORITY_ORDER:-1,0,2}"
GOLEM_NOC_VN_STARVATION_VN="${GOLEM_NOC_VN_STARVATION_VN:-2}"
GOLEM_NOC_VN_MAX_STARVATION_CYCLES="${GOLEM_NOC_VN_MAX_STARVATION_CYCLES:-4096}"
GOLEM_NOC_INTER_ROUTER_NO_CUT="${GOLEM_NOC_INTER_ROUTER_NO_CUT:-1}"
GOLEM_NOC_LOCAL_NO_CUT="${GOLEM_NOC_LOCAL_NO_CUT:-0}"
GOLEM_NOC_MEMORY_LOCAL_NO_CUT="${GOLEM_NOC_MEMORY_LOCAL_NO_CUT:-1}"
GOLEM_NOC_SCHED_LOCAL_NO_CUT="${GOLEM_NOC_SCHED_LOCAL_NO_CUT:-1}"
GOLEM_EXPLICIT_PARTITION="${GOLEM_EXPLICIT_PARTITION:-1}"
GOLEM_PARTITION_STRATEGY="${GOLEM_PARTITION_STRATEGY:-weighted_topology}"
GOLEM_PARTITION_WEIGHT_PROFILE="${GOLEM_PARTITION_WEIGHT_PROFILE:-}"
GOLEM_PARTITION_ROUTER_WEIGHT="${GOLEM_PARTITION_ROUTER_WEIGHT:-0.1}"
GOLEM_PARTITION_CPU_WEIGHT="${GOLEM_PARTITION_CPU_WEIGHT:-1.0}"
GOLEM_PARTITION_MANAGER_WEIGHT="${GOLEM_PARTITION_MANAGER_WEIGHT:-0.5}"
GOLEM_PARTITION_DATA_MEMORY_WEIGHT="${GOLEM_PARTITION_DATA_MEMORY_WEIGHT:-5.0}"
GOLEM_PARTITION_OS_WEIGHT="${GOLEM_PARTITION_OS_WEIGHT:-0.5}"
GOLEM_PARTITION_AFFINITY_EDGE_WEIGHT="${GOLEM_PARTITION_AFFINITY_EDGE_WEIGHT:-2.0}"
GOLEM_MESH_DIM_X="${GOLEM_MESH_DIM_X:-4}"
GOLEM_GM_BUFFER_LENGTH="${GOLEM_GM_BUFFER_LENGTH:-64KB}"
GOLEM_ROCC_VERBOSE="${GOLEM_ROCC_VERBOSE:-0}"
GOLEM_GM_VERBOSE="${GOLEM_GM_VERBOSE:-0}"
GOLEM_GM_DUMP_DATA="${GOLEM_GM_DUMP_DATA:-0}"
GOLEM_DMA_TRACE="${GOLEM_DMA_TRACE:-0}"
GOLEM_REQUEST_SCHEDULER_TRACE="${GOLEM_REQUEST_SCHEDULER_TRACE:-0}"
GOLEM_LLSC_TRACE="${GOLEM_LLSC_TRACE:-0}"
GOLEM_MVM_DUMP_ENABLE="${GOLEM_MVM_DUMP_ENABLE:-0}"
GOLEM_MVM_DUMP_MODE="${GOLEM_MVM_DUMP_MODE:-overwrite}"
GOLEM_SIM_MODE="${GOLEM_SIM_MODE:-full-functional}"
GOLEM_REQUEST_SCHEDULER_EVENT_DRIVEN_WORKER="${GOLEM_REQUEST_SCHEDULER_EVENT_DRIVEN_WORKER:-}"
GOLEM_VANADIS_ROCC_WAIT_FASTPATH="${GOLEM_VANADIS_ROCC_WAIT_FASTPATH:-}"
GOLEM_VANADIS_ROCC_WAIT_FASTPATH_THRESHOLD="${GOLEM_VANADIS_ROCC_WAIT_FASTPATH_THRESHOLD:-256}"
GOLEM_PROGRESS_HEARTBEAT="${GOLEM_PROGRESS_HEARTBEAT:-1}"
GOLEM_PROGRESS_INTERVAL_CYCLES="${GOLEM_PROGRESS_INTERVAL_CYCLES:-50000}"
GOLEM_TERMINAL_VERBOSE="${GOLEM_TERMINAL_VERBOSE:-0}"
GOLEM_TERMINAL_REFRESH_SECONDS="${GOLEM_TERMINAL_REFRESH_SECONDS:-1}"
read -r -a REQUESTED_SST_ARGS <<< "${GOLEM_SST_ARGS:-}"
FILTERED_SST_ARGS=()
SKIP_SST_ARG=0
for SST_ARG in "${REQUESTED_SST_ARGS[@]}"; do
	if [[ "$SKIP_SST_ARG" -eq 1 ]]; then
		SKIP_SST_ARG=0
		continue
	fi
	case "$SST_ARG" in
		--lib-path|--add-lib-path)
			SKIP_SST_ARG=1
			;;
		--lib-path=*|--add-lib-path=*|--no-env-config)
			;;
		*)
			FILTERED_SST_ARGS+=("$SST_ARG")
			;;
	esac
done
GOLEM_SST_ARGS="--lib-path=$WORKTREE_ROOT/install/lib/sst-elements-library --no-env-config"
if [[ "${#FILTERED_SST_ARGS[@]}" -gt 0 ]]; then
	GOLEM_SST_ARGS+=" ${FILTERED_SST_ARGS[*]}"
fi
GOLEM_SST_THREADS="${GOLEM_SST_THREADS:-1}"
GOLEM_CPUSET="${GOLEM_CPUSET:-}"
GOLEM_CPU_BINDING_MODE="${GOLEM_CPU_BINDING_MODE:-default}"
GOLEM_MPI_RANKS="${GOLEM_MPI_RANKS:-8}"
GOLEM_MPI_LAUNCHER="${GOLEM_MPI_LAUNCHER:-mpirun}"
GOLEM_MPI_ARGS_WAS_SET="${GOLEM_MPI_ARGS+x}"
GOLEM_MPI_ARGS="${GOLEM_MPI_ARGS:---bind-to core --map-by core}"
GOLEM_MPI_PARTITIONER="${GOLEM_MPI_PARTITIONER:-}"
if [[ -z "$GOLEM_MPI_PARTITIONER" ]]; then
	if [[ "$GOLEM_EXPLICIT_PARTITION" == "1" ]]; then
		GOLEM_MPI_PARTITIONER="sst.self"
	else
		GOLEM_MPI_PARTITIONER="sst.simple"
	fi
fi
GOLEM_SST_STAT_LOAD_LEVEL="${GOLEM_SST_STAT_LOAD_LEVEL:-16}"
GOLEM_SST_ENABLE_ALL_STATS="${GOLEM_SST_ENABLE_ALL_STATS:-1}"
GOLEM_EXPORT_NOC_HEATMAPS="${GOLEM_EXPORT_NOC_HEATMAPS:-0}"
GOLEM_SKIP_TENSOR_GEN="${GOLEM_SKIP_TENSOR_GEN:-0}"
GOLEM_SKIP_HBM_GEN="${GOLEM_SKIP_HBM_GEN:-0}"
GOLEM_SKIP_BUILD="${GOLEM_SKIP_BUILD:-0}"
GOLEM_SKIP_DEFAULT_GUEST_BUILD="${GOLEM_SKIP_DEFAULT_GUEST_BUILD:-0}"
GOLEM_HBM_DUMP_OUTPUT="${GOLEM_HBM_DUMP_OUTPUT:-1}"
GOLEM_BENCH_QUIET_LOGS="${GOLEM_BENCH_QUIET_LOGS:-0}"
GOLEM_BENCH_DISABLE_SST_STATS="${GOLEM_BENCH_DISABLE_SST_STATS:-0}"

LOG_FILE="${LOG_FILE:-${GOLEM_PRESET_LOG:-test.log}}"
TENSOR_A_FILE="${GOLEM_TENSOR_A_FILE:-}"
TENSOR_B_FILE="${GOLEM_TENSOR_B_FILE:-}"
DUMP_C_FILE="${GOLEM_DUMP_C_FILE:-}"
DUMP_C_AUTO=0
TENSOR_SOURCE="${GOLEM_TENSOR_SOURCE:-synthetic}"
TENSOR_DIR="${GOLEM_TENSOR_DIR:-$SCRIPT_DIR/data}"

ARRAY_IN_SET=0
ARRAY_OUT_SET=0
DRY_RUN=0
PRINT_CORE_MAP=0
VERIFY_MVM="${GOLEM_VERIFY_MVM:-0}"
VERIFY_C="${GOLEM_VERIFY_C:-0}"
TIMELINE="${GOLEM_TIMELINE:-0}"
TOTAL_STAGES=4
PROGRESS_WIDTH=32
CONFIG_NOTICES=()

supports_fancy_output() {
	[[ -t 1 ]]
}

UI_COLOR_ENABLED=0
if [[ -t 1 ]] && [[ "${TERM:-dumb}" != "dumb" ]] && [[ -z "${NO_COLOR:-}" ]]; then
	UI_COLOR_ENABLED=1
fi

supports_color_output() {
	[[ "$UI_COLOR_ENABLED" -eq 1 ]]
}

ui_color() {
	local code="$1"
	shift
	if supports_color_output; then
		printf '\033[%sm%s\033[0m' "$code" "$*"
	else
		printf '%s' "$*"
	fi
}

ui_header() {
	printf '\n%s\n' "$(ui_color '1;36' "== $* ==")"
}

ui_kv() {
	printf '  '
	ui_color '2;37' "$(printf '%-12s' "$1")"
	printf ' %s\n' "$(ui_color '1;35' "$2")"
}

ui_stage() {
	local index="$1"
	shift
	printf '\n%s %s\n' "$(ui_color '1;33' "[$index/$TOTAL_STAGES]")" "$(ui_color '1;36' "$*")"
}

ui_success() {
	printf '%s %s\n' "$(ui_color '1;32' '[OK]')" "$*"
}

format_elapsed() {
	local total="${1:-0}"
	printf '%02d:%02d:%02d' "$((total / 3600))" "$(((total % 3600) / 60))" "$((total % 60))"
}

csv_metric() {
	local file="$1"
	local metric="$2"
	[[ -f "$file" ]] || return 0
	awk -F, -v key="$metric" '$1 == key {
		value = substr($0, length($1) + 2)
		sub(/\r$/, "", value)
		if (value ~ /^".*"$/) {
			sub(/^"/, "", value)
			sub(/"$/, "", value)
			gsub(/""/, "\"", value)
		}
		print value
		exit
	}' "$file"
}

display_path() {
	local path="$1"
	if [[ "$path" == "$SCRIPT_DIR/"* ]]; then
		printf './%s' "${path#"$SCRIPT_DIR/"}"
	else
		printf '%s' "$path"
	fi
}

run_archived() {
	{
		printf '\n$'
		printf ' %q' "$@"
		printf '\n'
	} >> "$PIPELINE_DETAIL_LOG"
	local status
	if [[ "$GOLEM_TERMINAL_VERBOSE" -eq 1 ]]; then
		if "$@" 2>&1 | tee -a "$PIPELINE_DETAIL_LOG"; then
			return 0
		else
			status=$?
		fi
	else
		if "$@" >> "$PIPELINE_DETAIL_LOG" 2>&1; then
			return 0
		else
			status=$?
		fi
	fi
	echo "[ERROR] Command failed (exit $status); details: $(display_path "$PIPELINE_DETAIL_LOG")" >&2
	return "$status"
}

dtype_nbytes() {
	case "$1" in
		int32|fp32)
			printf '4\n' ;;
		fp16)
			printf '2\n' ;;
		*)
			printf '0\n' ;;
	esac
}

file_size_bytes() {
	local path="$1"
	stat -c '%s' "$path"
}

metadata_get_value() {
	local file="$1"
	local key="$2"
	awk -F= -v k="$key" '$1 == k { print substr($0, index($0, "=") + 1) }' "$file" | tail -n 1
}

write_metadata_file() {
	local file="$1"
	shift
	mkdir -p "$(dirname "$file")"
	: > "$file"
	for key in "$@"; do
		printf '%s=%s\n' "$key" "${!key}" >> "$file"
	done
}

validate_metadata_file() {
	local file="$1"
	local label="$2"
	shift 2
	if [[ ! -f "$file" ]]; then
		echo "[ERROR] $label metadata 不存在: $file" >&2
		echo "        关闭对应 skip 重新生成一次，之后才能安全复用。" >&2
		return 1
	fi
	local mismatch=0
	for key in "$@"; do
		local expected="${!key}"
		local actual
		actual="$(metadata_get_value "$file" "$key")"
		if [[ -z "$actual" && "$expected" != "" ]]; then
			echo "[ERROR] $label metadata 缺少 $key，期望 $expected" >&2
			mismatch=1
		elif [[ "$actual" != "$expected" ]]; then
			echo "[ERROR] $label metadata 不匹配: $key 当前=$expected 复用文件=$actual" >&2
			mismatch=1
		fi
	done
	if [[ "$mismatch" -ne 0 ]]; then
		echo "        当前配置已经变化，请关闭对应 skip 重新生成。" >&2
		return 1
	fi
	return 0
}

validate_tensor_file_size() {
	local path="$1"
	local label="$2"
	local expected="$3"
	if [[ ! -f "$path" ]]; then
		echo "[ERROR] 缺少 $label tensor: $path" >&2
		return 1
	fi
	local actual
	actual="$(file_size_bytes "$path")"
	if [[ "$actual" != "$expected" ]]; then
		echo "[ERROR] $label tensor 大小不匹配: $path 当前文件=${actual}B 期望=${expected}B" >&2
		echo "        维度或 dtype 已变化，请关闭 GOLEM_SKIP_TENSOR_GEN 重新生成 tensor。" >&2
		return 1
	fi
	return 0
}

validate_hbm_contract_fallback() {
	local file="$1"
	if [[ ! -f "$file" ]]; then
		echo "[ERROR] HBM metadata 不存在，且找不到兼容检查文件: $file" >&2
		echo "        请关闭 GOLEM_SKIP_HBM_GEN 重新生成 HBM。" >&2
		return 1
	fi
	python3 - "$file" \
		"$GOLEM_GEMM_M" "$GOLEM_GEMM_N" "$GOLEM_GEMM_K" \
		"$GOLEM_GEMM_BLOCK_M" "$GOLEM_GEMM_BLOCK_N" "$GOLEM_GEMM_BLOCK_K" \
		"$GOLEM_MATMUL_DTYPE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = {
    "m": int(sys.argv[2]),
    "n": int(sys.argv[3]),
    "k": int(sys.argv[4]),
    "block_m": int(sys.argv[5]),
    "block_n": int(sys.argv[6]),
    "block_k": int(sys.argv[7]),
    "dtype": sys.argv[8],
}
try:
    actual = json.loads(path.read_text())
except Exception as exc:
    print(f"[ERROR] 读取 HBM 兼容 metadata 失败: {path}: {exc}", file=sys.stderr)
    sys.exit(1)

bad = []
for key, want in expected.items():
    got = actual.get(key)
    if got != want:
        bad.append(f"{key} 当前={want} 复用文件={got}")
if bad:
    for item in bad:
        print(f"[ERROR] HBM contract 不匹配: {item}", file=sys.stderr)
    print("        当前矩阵配置已变化，请关闭 GOLEM_SKIP_HBM_GEN 重新生成 HBM。", file=sys.stderr)
    sys.exit(1)
print(f"[WARN] {path} 只覆盖 M/N/K/block/dtype；建议下次关闭 GOLEM_SKIP_HBM_GEN 生成完整 hbm_config.env。")
PY
}

align_up_int() {
	local value="$1"
	local align="$2"
	if (( align <= 0 )); then
		printf '%s\n' "$value"
		return
	fi
	printf '%s\n' $(( ((value + align - 1) / align) * align ))
}

derive_auto_mem_node_size() {
	python3 - "$@" <<'PY'
import sys

(
    gemm_m,
    gemm_n,
    gemm_k,
    block_m,
    block_n,
    block_k,
    elem_bytes,
    num_memory_nodes,
    total_groups,
    total_gemm_cores,
    group_manager_enable,
    a_reuse_n,
    b_reuse_m,
) = [int(x) for x in sys.argv[1:]]


def align_up(value, align):
    return ((value + align - 1) // align) * align


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def next_power_of_two(value):
    return 1 << (value - 1).bit_length()


mm_align = 0x100
m_tiles = gemm_m // block_m
n_tiles = gemm_n // block_n
k_tiles = gemm_k // block_k
m_groups = ceil_div(m_tiles, b_reuse_m)
n_groups = ceil_div(n_tiles, a_reuse_n)
total_macro_tasks = m_groups * n_groups
data_nodes = list(range(1, num_memory_nodes))
dedicated_manager_cores = total_groups if group_manager_enable else 0
active_gemm_cores = total_gemm_cores - dedicated_manager_cores
first_worker_core = dedicated_manager_cores


def owner_core_for_task(task_id):
    if active_gemm_cores <= 0:
        return first_worker_core
    return first_worker_core + (task_id % active_gemm_cores)


def group_id_for_core(core_id):
    return core_id % total_groups if total_groups > 0 else 0


def data_node_for_task(task_id):
    if not data_nodes:
        return 1
    group_id = group_id_for_core(owner_core_for_task(task_id))
    return data_nodes[group_id % len(data_nodes)]


def a_data_node_for_m_tile(m_tile):
    return data_node_for_task(m_tile // b_reuse_m)


def b_data_node_for_n_tile(n_tile):
    return data_node_for_task(n_tile // a_reuse_n)


def max_count(items, predicate):
    out = 0
    for node_idx in data_nodes:
        count = sum(1 for item in items if predicate(node_idx, item))
        out = max(out, count)
    return out or 1


max_a_m_tiles = max_count(range(m_tiles), lambda node, tile: a_data_node_for_m_tile(tile) == node)
max_b_n_tiles = max_count(range(n_tiles), lambda node, tile: b_data_node_for_n_tile(tile) == node)
max_macro_tasks = max_count(range(total_macro_tasks), lambda node, task: data_node_for_task(task) == node)

mat_stride = align_up(block_m * block_k * elem_bytes, mm_align)
vec_stride = align_up(block_k * elem_bytes, mm_align)
out_stride = align_up(block_m * block_n * elem_bytes, mm_align)
bias_stride = align_up(gemm_n * elem_bytes, mm_align)
mat_region_end = max_a_m_tiles * k_tiles * mat_stride
vec_region_end = mat_region_end + max_b_n_tiles * k_tiles * block_n * vec_stride
out_reuse_slots = max(1, a_reuse_n) * max(1, b_reuse_m)
data_region_end = vec_region_end + max_macro_tasks * out_reuse_slots * out_stride
fixed_aux_region_end = 0x01300000
required = max(data_region_end + bias_stride, fixed_aux_region_end, 64 * 1024**2)
chosen = next_power_of_two(required)
print(chosen, required, data_region_end, bias_stride)
PY
}

sst_log_has_fatal() {
	local log_file="$1"
	[[ -f "$log_file" ]] || return 1
	grep -Eqi \
		'corrupted size vs\. prev_size|Segmentation fault|Signal:[[:space:]]+(Aborted|Segmentation fault)|Assertion .* failed|(^|[^[:alnum:]_])(fatal|panic)([^[:alnum:]_]|$)' \
		< <(tail -n 2000 "$log_file" 2>/dev/null)
}

sst_log_has_complete() {
	local log_file="$1"
	[[ -f "$log_file" ]] || return 1
	grep -q 'Simulation is complete, simulated time:' < <(tail -n 200 "$log_file" 2>/dev/null)
}

print_sst_failure_context() {
	local log_file="$1"
	echo "[ERROR] SST failed. Recent log tail:"
	python3 - "$log_file" <<'PY'
import pathlib
import sys

log_path = pathlib.Path(sys.argv[1])
try:
    lines = log_path.read_text(errors="ignore").splitlines()
except OSError as exc:
    print(f"[WARN] Unable to read log: {exc}")
    sys.exit(0)

for line in lines[-40:]:
    print(line)
PY
}

terminate_sst_process_tree() {
	local pid="$1"
	[[ -n "$pid" ]] || return 0
	local pgid
	pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]')"
	if [[ -n "$pgid" ]]; then
		kill -TERM -- "-$pgid" 2>/dev/null || true
		sleep 0.2
		kill -KILL -- "-$pgid" 2>/dev/null || true
	else
		kill -TERM "$pid" 2>/dev/null || true
		sleep 0.2
		kill -KILL "$pid" 2>/dev/null || true
	fi
}

SST_PID=""

cleanup_sst_on_exit() {
	local status="$?"
	if [[ -n "$SST_PID" ]] && sst_pid_is_running "$SST_PID"; then
		echo "[WARN] Script interrupted or exiting; terminating SST pid=$SST_PID"
		terminate_sst_process_tree "$SST_PID"
		wait "$SST_PID" 2>/dev/null || true
	fi
	return "$status"
}

trap cleanup_sst_on_exit INT TERM EXIT

wait_for_sst_exit() {
	local pid="$1"
	local timeout_sec="${2:-5}"
	local waited=0
	while sst_pid_is_running "$pid"; do
		if (( waited >= timeout_sec * 10 )); then
			return 1
		fi
		sleep 0.1
		((waited++))
	done
	wait "$pid" 2>/dev/null || true
	return 0
}

sst_pid_is_running() {
	local pid="$1"
	[[ -n "$pid" ]] || return 1
	local stat
	stat="$(ps -o stat= -p "$pid" 2>/dev/null | tr -d '[:space:]')"
	[[ -n "$stat" && "${stat:0:1}" != "Z" ]]
}

estimate_sst_progress_info() {
	local log_file="$1"
	if [[ ! -f "$log_file" ]]; then
		echo "0|启动 SST"
		return
	fi

	# Fused Attention progress uses unique participating cores so a first-worker
	# event cannot be mistaken for the all-worker system frontier.
	local attention_ms
	attention_ms="$(tail -n 6000 "$log_file" 2>/dev/null |
		awk -f "$SCRIPT_DIR/small/muticore_attention/attention_progress.awk")"
	if [[ "$attention_ms" != "none" ]]; then
		echo "$attention_ms"
		return
	fi

	# 优先使用里程碑日志（lenet_conv12）：
	# [MILESTONE] stage=conv1|conv2|fc1|fc2|fc3 status=start|done|fail cycle=...
	local ms
	ms="$(tail -n 6000 "$log_file" 2>/dev/null | awk '
		function token_value(key,    i,n,a) {
			n = split($0, a, /[[:space:]]+/);
			for (i = 1; i <= n; i++) {
				if (index(a[i], key "=") == 1) return substr(a[i], length(key) + 2);
			}
			return "";
		}
		/\[MILESTONE\]/ {
			stage = token_value("stage"); st = token_value("status");
			if (stage != "" && st != "") {
				seen=1;
				if (st == "start") start[stage]=1;
				if (st == "done") done[stage]=1;
				if (st == "fail") fail=stage;
			}
		}
		END {
			if (seen != 1) {
				print "none";
				exit;
			}
			if (fail != "") {
				printf("98|网络阶段失败:%s", fail);
				exit;
			}
			if (done["fc3"] == 1) {
				print "98|网络阶段完成收尾";
				exit;
			}
			if (start["fc3"] == 1) {
				print "90|网络阶段 fc3";
				exit;
			}
			if (done["fc2"] == 1) {
				print "85|网络阶段 fc2 完成";
				exit;
			}
			if (start["fc2"] == 1) {
				print "75|网络阶段 fc2";
				exit;
			}
			if (done["fc1"] == 1) {
				print "70|网络阶段 fc1 完成";
				exit;
			}
			if (start["fc1"] == 1) {
				print "60|网络阶段 fc1";
				exit;
			}
			if (done["conv2"] == 1) {
				print "55|网络阶段 conv2 完成";
				exit;
			}
			if (start["conv2"] == 1) {
				print "45|网络阶段 conv2";
				exit;
			}
			if (done["conv1"] == 1) {
				print "35|网络阶段 conv1 完成";
				exit;
			}
			if (start["conv1"] == 1) {
				print "20|网络阶段 conv1";
				exit;
			}
			print "5|网络工作负载初始化";
		}
	')"
	if [[ "$ms" != "none" ]]; then
		echo "$ms"
		return
	fi

	# 优先使用轻量心跳日志估算真实进度（需要 --progress-heartbeat 1）
	local hb
	hb="$(tail -n 4000 "$log_file" 2>/dev/null | awk '
		function token_value(key,    i,n,a,v) {
			n = split($0, a, /[[:space:]]+/);
			for (i = 1; i <= n; i++) {
				if (index(a[i], key "=") == 1) return substr(a[i], length(key) + 2);
			}
			return "";
		}
		BEGIN {
			dma_sum_c = 0; dma_sum_t = 0;
			mvm_sum_c = 0; mvm_sum_t = 0;
		}
		/GlobalMemory core=[0-9]+ DMA_PROGRESS(_FINAL)?:/ {
			core = token_value("core"); completed = token_value("completed");
			if (core != "" && completed != "") {
				split(completed, p, "/");
				dma_c[core] = p[1];
				dma_t[core] = p[2];
			}
		}
		/RoCC core=[0-9]+ MVM_PROGRESS:/ {
			core = token_value("core"); completed = token_value("completed");
			if (core != "" && completed != "") {
				split(completed, p, "/");
				mvm_c[core] = p[1];
				mvm_t[core] = p[2];
			}
		}
		END {
			for (k in dma_t) { dma_sum_c += dma_c[k]; dma_sum_t += dma_t[k]; }
			for (k in mvm_t) { mvm_sum_c += mvm_c[k]; mvm_sum_t += mvm_t[k]; }
			dma_pct = (dma_sum_t > 0) ? int((dma_sum_c * 100) / dma_sum_t) : -1;
			mvm_pct = (mvm_sum_t > 0) ? int((mvm_sum_c * 100) / mvm_sum_t) : -1;
			printf("%d|%d", dma_pct, mvm_pct);
		}
	')"

	local dma_pct="${hb%%|*}"
	local mvm_pct="${hb##*|}"
	if [[ "$dma_pct" =~ ^-?[0-9]+$ && "$mvm_pct" =~ ^-?[0-9]+$ ]]; then
		if [[ "$mvm_pct" -ge 0 ]]; then
			# MVM completion is the workload-wide progress signal. Keep startup and
			# SST shutdown headroom without inventing a separate 50% DMA phase.
			local p=$(( 5 + (mvm_pct * 93 / 100) ))
			if [[ "$dma_pct" -ge 0 ]]; then
				echo "$p|计算 | DMA ${dma_pct}% | MVM ${mvm_pct}%"
			else
				echo "$p|计算 | MVM ${mvm_pct}%"
			fi
			return
		fi
		if [[ "$dma_pct" -ge 0 ]]; then
			local p=$(( 8 + (dma_pct * 47 / 100) ))
			echo "$p|DMA | ${dma_pct}%"
			return
		fi
	fi

	if grep -q "mvm.ovec2gm\|MVM compute" < <(tail -n 2000 "$log_file" 2>/dev/null); then
		echo "70|计算（等待下一次心跳）"
	elif grep -q "DMA_READ_COMPLETE\|RemoteLoad issued\|remote_ld" < <(tail -n 2000 "$log_file" 2>/dev/null); then
		echo "20|DMA（等待下一次心跳）"
	elif grep -q "readBinaryELFInfo\|ELF Information" < <(tail -n 2000 "$log_file" 2>/dev/null); then
		echo "5|装载工作负载"
	elif grep -q "init phase=\|Creating CPU core\|Configuring for .* core links" < <(tail -n 2000 "$log_file" 2>/dev/null); then
		echo "2|构建仿真架构"
	else
		echo "0|启动 SST"
	fi
}

print_progress_bar() {
	local stage="$1"
	local label="$2"
	local filled=$(( stage * PROGRESS_WIDTH / TOTAL_STAGES ))
	local empty=$(( PROGRESS_WIDTH - filled ))
	local bar
	bar="$(printf '%*s' "$filled" '' | tr ' ' '#')$(printf '%*s' "$empty" '' | tr ' ' '-')"
	local pct=$(( stage * 100 / TOTAL_STAGES ))
	printf '%s [%s] %3d%%  %s\n' "$(ui_color '1;36' 'PIPELINE')" "$bar" "$pct" "$label"
}

render_inline_progress_bar() {
	local pct="$1"
	local width="$2"
	local filled=$(( pct * width / 100 ))
	local empty=$(( width - filled ))
	local filled_seg empty_seg
	filled_seg="$(printf '%*s' "$filled" '' | tr ' ' '=')"
	empty_seg="$(printf '%*s' "$empty" '' | tr ' ' '.')"
	if supports_color_output; then
		printf '[\033[38;5;48m%s\033[2;37m%s\033[0m]' "$filled_seg" "$empty_seg"
	else
		printf '[%s%s]' "$filled_seg" "$empty_seg"
	fi
}

merge_ranked_stats() {
	local base_file="$1"
	local base_dir="${base_file%/*}"
	local base_name="${base_file##*/}"
	local stem="${base_name%.*}"
	local ext="${base_name##*.}"
	local merged_file="${base_file}.merge.$$"
	local files=()

	shopt -s nullglob
	files=("$base_dir/${stem}"_*."$ext")
	shopt -u nullglob
	if [[ "${#files[@]}" -eq 0 ]]; then
		echo "[ERROR] MPI 统计文件不存在: $base_dir/${stem}_*.${ext}" >&2
		return 1
	fi

	# SST writes the CSV header in every rank file; retain only the first one.
	awk 'FNR == 1 && NR != 1 { next } { print }' "${files[@]}" > "$merged_file"
	mv "$merged_file" "$base_file"
	echo "[INFO] Merged ${#files[@]} MPI statistic files into $base_file"
}

merge_ranked_stats() {
	local base_file="$1"
	local base_dir="${base_file%/*}"
	local base_name="${base_file##*/}"
	local stem="${base_name%.*}"
	local ext="${base_name##*.}"
	local merged_file="${base_file}.merge.$$"
	local files=()

	shopt -s nullglob
	files=("$base_dir/${stem}"_*."$ext")
	shopt -u nullglob
	if [[ "${#files[@]}" -eq 0 ]]; then
		echo "[ERROR] MPI 统计文件不存在: $base_dir/${stem}_*.${ext}" >&2
		return 1
	fi

	# SST writes one CSV header per rank; keep only the first header.
	awk 'FNR == 1 && NR != 1 { next } { print }' "${files[@]}" > "$merged_file"
	mv "$merged_file" "$base_file"
	echo "[INFO] Merged ${#files[@]} MPI statistic files into $base_file"
}

resolve_stage_clock_ghz() {
	if [[ -n "${GOLEM_STAGE_CLOCK_GHZ:-}" ]]; then
		echo "$GOLEM_STAGE_CLOCK_GHZ"
		return
	fi
	local raw="${VANADIS_CPU_CLOCK:-2.3GHz}"
	if [[ "$raw" =~ ^([0-9]+([.][0-9]+)?)GHz$ ]]; then
		echo "${BASH_REMATCH[1]}"
		return
	fi
	if [[ "$raw" =~ ^([0-9]+([.][0-9]+)?)MHz$ ]]; then
		python3 - <<PY
v=float("${BASH_REMATCH[1]}")
print(v/1000.0)
PY
		return
	fi
	if [[ "$raw" =~ ^([0-9]+([.][0-9]+)?)kHz$ ]]; then
		python3 - <<PY
v=float("${BASH_REMATCH[1]}")
print(v/1000000.0)
PY
		return
	fi
	if [[ "$raw" =~ ^([0-9]+([.][0-9]+)?)Hz$ ]]; then
		python3 - <<PY
v=float("${BASH_REMATCH[1]}")
print(v/1000000000.0)
PY
		return
	fi
	echo "2.3"
}

show_help() {
	cat <<'EOF'
Usage:
	./run_noc_dma_pipeline.sh [options]

Options:
	(自动加载) 若存在 configs/default.env，会在启动时自动 source；
	           default.env 会继续加载分类文件：
	           configs/10_core_gemm.env
	           configs/20_dma.env
	           configs/30_network.env
	           configs/40_debug_io.env
	           configs/50_tensor_verify.env
	--groups N           组数量（默认: 4）
	--array-in N         RoCC 阵列输入长度（默认: 4）
	--array-out N        RoCC 阵列输出长度（默认: 4）
	--num-arrays N       RoCC 阵列实例数（默认: 1）
	--gemm-cores N       GEMM 并发核心数（默认: 16）
	--num-cores N        总核心数（同时用于 SST 与 C++ TOTAL_CORES，默认: 16）
	--num-mem-nodes N    内存节点总数（首节点挂 OS，默认: 4）
	--mem-node-size N|auto 单个内存节点大小（字节），auto 按当前 GEMM/HBM 布局选 2 的幂
	--global-stride-kb N  每核 GM 窗口大小（KB，默认: 64）
	--dim sN             设置方阵 GEMM 维度 M=N=K=N，例如 --dim s2048
	--gemm-m N           GEMM 的 M 维（默认: 跟 --array-out）
	--gemm-n N           GEMM 的 N 维（默认: 跟 --num-arrays）
	--gemm-k N           GEMM 的 K 维（默认: 跟 --array-in）
	--orig-m N           原始请求 M（仅记录/元数据，不参与内核执行）
	--orig-n N           原始请求 N（仅记录/元数据，不参与内核执行）
	--orig-k N           原始请求 K（仅记录/元数据，不参与内核执行）
	--gemm-block-m N     GEMM block_M（默认: 跟 --array-out，phase-1 要求等于 --array-out）
	--gemm-block-n N     GEMM block_N（默认: 跟 --num-arrays，phase-1 要求 <= --num-arrays）
	--gemm-block-k N     GEMM block_K（默认: 跟 --array-in；WCP 支持不大于 array-in 的窄 K，或其整数倍）
	--dtype TYPE         matmul 数据类型：int32|fp16|fp32（默认: int32）
	--transpose-b N      B 的逻辑转置：0 使用 [K,N]，1 使用原生 [N,K]（默认: 0）
	--bias-enable N      可选后处理bias开关（0:关闭,1:开启，默认: 0）
	--bias-value N       bias常量值（int32/fp16/fp32，默认: 0）
	--bias-file FILE     bias向量文件（.bin/.csv/.npy），透传给 gen_hbm_init.py
	--pool1-file FILE    预置 pool1 张量文件（6x12x12 fp32 .bin），透传给 gen_hbm_init.py
	--conv2-bpack-file FILE  conv2 B打包文件（3x16x64 fp32 .bin），透传给 gen_hbm_init.py
	--conv2-bias-file FILE   conv2 bias文件（16 fp32 .bin），透传给 gen_hbm_init.py
	--fc1-weight-file FILE   fc1分片权重文件（4x2x64x64 fp32 .bin），透传给 gen_hbm_init.py
	--fc1-bias-file FILE     fc1 bias文件（120 fp32 .bin），透传给 gen_hbm_init.py
	--fc2-weight-file FILE   fc2权重文件（2x2x64x64 fp32 .bin），透传给 gen_hbm_init.py
	--fc2-bias-file FILE     fc2 bias文件（84 fp32 .bin），透传给 gen_hbm_init.py
	--fc3-weight-file FILE   fc3权重文件（2x2x64x64 fp32 .bin），透传给 gen_hbm_init.py
	--fc3-bias-file FILE     fc3 bias文件（10 fp32 .bin），透传给 gen_hbm_init.py
	--dma-stagger-cycles N  每核启动DMA错峰周期（默认: 0，建议dim16从2000起试）
	--dma-overlap N      DMA/计算重叠开关（0:关闭, 1:开启，默认: 0）
	--dma-max-inflight N 每核DMA读请求在途上限（默认: 8）
	--dma-read-retry-ticks N DMA读chunk重试超时tick（默认: 96）
	--dma-read-max-retries N DMA读chunk最大重试次数（默认: 8）
	--dma-burst-bytes N  DMA分块大小（字节，默认: 64）
	--dma-panel-chunk-bytes N  manager 每次发送的 panel chunk 大小
	--dma-tile-chunk-quantum N  同一 worker tile 连续准入的 chunk 数（默认: 1）
	--dma-response-tile-priority N  返回侧 tile-completion-aware 调度（0/1）
	--dma-response-reorder-cycles N  返回响应的最小收集周期
	--dma-response-max-starvation-cycles N  返回侧 FIFO fallback 上界（0 关闭）
	--dma-admission-max-starvation-cycles N  准入侧 oldest-admissible fallback 上界（0 关闭）
	--progress-heartbeat N   轻量进度心跳（0关闭,1开启，默认: 1）
	--progress-interval-cycles N 进度心跳周期（cycles，默认: 50000）
	--verbose-output     同时在终端显示构建与后处理的完整内部输出
	--rocc-type TYPE     RoCC 类型（默认: golem.RoCCAnalogInt）
	--array-type TYPE    阵列类型（默认: golem.MVMIntArray）
	--noc-buf SIZE       同时设置 NoC input/output buffer 大小（默认: 8KB）
	--noc-in-buf SIZE    仅设置 NoC input buffer 大小（默认: 8KB）
	--noc-out-buf SIZE   仅设置 NoC output buffer 大小（默认: 8KB）
	--noc-memnic-buf SIZE      同时设置 MemNIC input/output buffer 大小（默认: 128KB）
	--noc-memnic-in-buf SIZE   仅设置 MemNIC input buffer 大小（默认: 128KB）
	--noc-memnic-out-buf SIZE  仅设置 MemNIC output buffer 大小（默认: 128KB）
	--noc-link-bw BW     NoC 链路带宽（默认: 25GB/s）
	--noc-xbar-bw BW     NoC 路由器 xbar 带宽（默认: 25GB/s）
	--noc-flit-size SIZE NoC flit 大小（默认: 128B）
	--mesh-dim-x N       Mesh 列数 GOLEM_MESH_DIM_X（默认: 4）
	--gm-buf SIZE        GlobalMemory networkIF buffer_length（默认: 64KB）
	--rocc-verbose N     RoCC 日志级别（默认: -1，静默）
	--gm-verbose N       GlobalMemory 日志级别（默认: 0，关键 DMA 仍保留）
	--gm-dump-data N     GlobalMemory 数据hex打印（0关闭，1开启，默认: 0）
	--mvm-dump           打开 MVM 结果按核落盘（等价 GOLEM_MVM_DUMP_ENABLE=1）
	--mvm-dump-dir DIR   MVM 结果输出根目录（默认: mvm_dumps）
	--sim-mode MODE      仿真模式: full-functional|full-timing（默认: full-functional）
	--memory-backend B   内存时序后端: dramsim3|ramulator2（默认: ramulator2）
	--ramulator2-config FILE Ramulator2 完整 YAML 配置
	--timeline           生成 window timeline SVG，并在结果区打印路径
	--print-core-map     生成 core->memory-node 映射 CSV（归档到 artifacts/stats）
	--verify-mvm         运行 Python 离线矩阵结果校验（依赖 mvm dump）
	--verify-c           运行 C=AxB 端到端校验（需要 --tensor-a/--tensor-b）
	--output-mode MODE   最终 C 去向: hbm|fusion（默认: hbm）
	--tensor-source MODE 输入来源: synthetic|file|sample（默认: synthetic）
	--tensor-dir DIR     sample 模式输出目录（默认: tests/data）
	--tensor-a FILE      外部输入 A 矩阵文件（.bin/.csv/.npy，按 --dtype 解释）传给 gen_hbm_init.py
	--tensor-b FILE      外部输入 B 矩阵文件（.bin/.csv/.npy，按 --dtype 解释）传给 gen_hbm_init.py
	--dump-c FILE        从 HBM 输出或 fusion sink 重组 C 到 FILE（.bin 或 .csv）
	--hbm-dump-output N  是否生成 hbm_out_node*.bin（0/1，默认: 1）
	--no-hbm-dump-output 等价 --hbm-dump-output 0
	--log FILE           SST 输出日志文件名或绝对路径（默认: test.log，存放到 artifacts/logs）
	--mpi-ranks N        SST MPI rank 数（默认: 8；N>1 时由 mpirun 启动 SST）
	--sst-threads N       每个 MPI rank 的 SST 线程数（默认: 1）
	--partition-strategy N  显式分区策略：weighted_topology|round_robin
	--partition-weight-profile FILE  可选 router_id,weight CSV 权重覆盖
	--cpuset LIST         用 taskset 将整个 SST/MPI 作业限制到逻辑 CPU 列表
	--cpu-binding-mode N  写入 run summary 的绑定模式标签
	--mpi-launcher FILE  MPI 启动器（默认: mpirun）
	--mpi-args STRING    传给 MPI 启动器的参数（默认: --bind-to core --map-by core）
	--mpi-partitioner N  MPI 分区器（显式分区默认: sst.self；否则 sst.simple）
	--dry-run            仅打印配置与命令，不实际执行
	环境变量 GOLEM_SKIP_TENSOR_GEN=1 可跳过 sample tensor 生成
	环境变量 GOLEM_SKIP_HBM_GEN=1 可复用现有 hbm_init/out_node*.bin
	环境变量 GOLEM_TEST_BINARY 可指定测试二进制输出路径（并发任务必须隔离）
	环境变量 GOLEM_BUILD_METADATA_FILE 可指定对应的构建元数据路径
	环境变量 GOLEM_SKIP_BUILD=1 可复用 GOLEM_TEST_BINARY 指定的二进制
	环境变量 GOLEM_SKIP_DEFAULT_GUEST_BUILD=1 要求 VANADIS_EXE 指向可执行 custom guest
	-h, --help           显示帮助

Output Layout (默认):
	artifacts/logs       主日志与 trace
	artifacts/stdout/overlap0 or overlap1   stdout-* / stderr-* 分片输出
	artifacts/hbm        hbm_init_node*.bin / hbm_out_node*.bin
	artifacts/mvm_dumps/overlap0 or overlap1   MVM dump snapshots
	artifacts/stats/overlap0 or overlap1   stats_selfcom.txt / execution_summary.csv / dma_summary.csv / noc_summary.csv / memory_summary.csv / per-run NoC heatmaps

Priority:
	命令行参数 > 环境变量 > 脚本默认值

Examples:
	./run_noc_dma_pipeline.sh --array-in 8 --array-out 8
	./run_noc_dma_pipeline.sh --array-in 16 --array-out 16 --log test_dim16.log
	./run_noc_dma_pipeline.sh --dim s2048 --timeline --sim-mode full-timing --mpi-ranks 4
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--groups)
			GOLEM_TOTAL_GROUPS="$2"; shift 2 ;;
		--array-in)
			GOLEM_ARRAY_INPUT_SIZE="$2"; ARRAY_IN_SET=1; shift 2 ;;
		--array-out)
			GOLEM_ARRAY_OUTPUT_SIZE="$2"; ARRAY_OUT_SET=1; shift 2 ;;
		--num-arrays)
			GOLEM_NUM_ARRAYS="$2"; shift 2 ;;
		--gemm-cores)
			GOLEM_TOTAL_GEMM_CORES="$2"; shift 2 ;;
		--num-cores)
			GOLEM_TOTAL_CORES="$2"; shift 2 ;;
		--num-mem-nodes)
			GOLEM_NUM_MEMORY_NODES="$2"; shift 2 ;;
		--mem-node-size)
			GOLEM_MEM_NODE_SIZE_BYTES="$2"; shift 2 ;;
		--global-stride-kb)
			GOLEM_GLOBAL_STRIDE_KB="$2"; shift 2 ;;
		--dim)
			if [[ "${2:-}" =~ ^s([1-9][0-9]*)$ ]]; then
				GOLEM_GEMM_M="${BASH_REMATCH[1]}"
				GOLEM_GEMM_N="${BASH_REMATCH[1]}"
				GOLEM_GEMM_K="${BASH_REMATCH[1]}"
				shift 2
			else
				echo "[ERROR] --dim 仅支持方阵格式 s<正整数>，例如 --dim s2048；收到: ${2:-<missing>}" >&2
				exit 1
			fi ;;
		--gemm-m)
			GOLEM_GEMM_M="$2"; shift 2 ;;
		--gemm-n)
			GOLEM_GEMM_N="$2"; shift 2 ;;
		--gemm-k)
			GOLEM_GEMM_K="$2"; shift 2 ;;
		--orig-m)
			GOLEM_ORIG_M="$2"; shift 2 ;;
		--orig-n)
			GOLEM_ORIG_N="$2"; shift 2 ;;
		--orig-k)
			GOLEM_ORIG_K="$2"; shift 2 ;;
		--gemm-block-m)
			GOLEM_GEMM_BLOCK_M="$2"; shift 2 ;;
		--gemm-block-n)
			GOLEM_GEMM_BLOCK_N="$2"; shift 2 ;;
		--gemm-block-k)
			GOLEM_GEMM_BLOCK_K="$2"; shift 2 ;;
		--dtype)
			GOLEM_MATMUL_DTYPE="$2"; shift 2 ;;
		--transpose-b)
			GOLEM_MATMUL_TRANSPOSE_B="$2"; shift 2 ;;
		--bias-enable)
			GOLEM_BIAS_ENABLE="$2"; shift 2 ;;
		--bias-value)
			GOLEM_BIAS_VALUE="$2"; shift 2 ;;
		--bias-file)
			GOLEM_BIAS_FILE="$2"; shift 2 ;;
		--pool1-file)
			GOLEM_POOL1_FILE="$2"; shift 2 ;;
		--conv2-bpack-file)
			GOLEM_CONV2_BPACK_FILE="$2"; shift 2 ;;
		--conv2-bias-file)
			GOLEM_CONV2_BIAS_FILE="$2"; shift 2 ;;
		--fc1-weight-file)
			GOLEM_FC1_WEIGHT_FILE="$2"; shift 2 ;;
		--fc1-bias-file)
			GOLEM_FC1_BIAS_FILE="$2"; shift 2 ;;
		--fc2-weight-file)
			GOLEM_FC2_WEIGHT_FILE="$2"; shift 2 ;;
		--fc2-bias-file)
			GOLEM_FC2_BIAS_FILE="$2"; shift 2 ;;
		--fc3-weight-file)
			GOLEM_FC3_WEIGHT_FILE="$2"; shift 2 ;;
		--fc3-bias-file)
			GOLEM_FC3_BIAS_FILE="$2"; shift 2 ;;
		--dma-stagger-cycles)
			GOLEM_DMA_STAGGER_CYCLES="$2"; shift 2 ;;
		--dma-overlap)
			GOLEM_DMA_OVERLAP="$2"; shift 2 ;;
		--dma-max-inflight)
			GOLEM_DMA_MAX_INFLIGHT="$2"; shift 2 ;;
		--dma-read-retry-ticks)
			GOLEM_DMA_READ_RETRY_TICKS="$2"; shift 2 ;;
		--dma-read-max-retries)
			GOLEM_DMA_READ_MAX_RETRIES="$2"; shift 2 ;;
		--dma-burst-bytes)
			GOLEM_DMA_BURST_BYTES="$2"; shift 2 ;;
		--dma-panel-chunk-bytes)
			GOLEM_DMA_PANEL_CHUNK_BYTES="$2"; shift 2 ;;
		--dma-tile-chunk-quantum)
			GOLEM_DMA_TILE_CHUNK_QUANTUM="$2"; shift 2 ;;
		--dma-response-tile-priority)
			GOLEM_DMA_RESPONSE_TILE_PRIORITY_ENABLE="$2"; shift 2 ;;
		--dma-response-reorder-cycles)
			GOLEM_DMA_RESPONSE_REORDER_CYCLES="$2"; shift 2 ;;
		--dma-response-max-starvation-cycles)
			GOLEM_DMA_RESPONSE_MAX_STARVATION_CYCLES="$2"; shift 2 ;;
		--dma-admission-max-starvation-cycles)
			GOLEM_DMA_ADMISSION_MAX_STARVATION_CYCLES="$2"; shift 2 ;;
		--dma-response-drain-limit)
			GOLEM_DMA_RESPONSE_DRAIN_LIMIT="$2"; shift 2 ;;
		--progress-heartbeat)
			GOLEM_PROGRESS_HEARTBEAT="$2"; shift 2 ;;
		--progress-interval-cycles)
			GOLEM_PROGRESS_INTERVAL_CYCLES="$2"; shift 2 ;;
		--verbose-output)
			GOLEM_TERMINAL_VERBOSE=1; shift ;;
		--rocc-type)
			GOLEM_ROCC_TYPE="$2"; shift 2 ;;
		--array-type)
			GOLEM_ARRAY_TYPE="$2"; shift 2 ;;
		--noc-buf)
			GOLEM_NOC_INPUT_BUF_SIZE="$2"; GOLEM_NOC_OUTPUT_BUF_SIZE="$2"; shift 2 ;;
		--noc-in-buf)
			GOLEM_NOC_INPUT_BUF_SIZE="$2"; shift 2 ;;
		--noc-out-buf)
			GOLEM_NOC_OUTPUT_BUF_SIZE="$2"; shift 2 ;;
		--noc-memnic-buf)
			GOLEM_NOC_MEMNIC_INPUT_BUF_SIZE="$2"; GOLEM_NOC_MEMNIC_OUTPUT_BUF_SIZE="$2"; shift 2 ;;
		--noc-memnic-in-buf)
			GOLEM_NOC_MEMNIC_INPUT_BUF_SIZE="$2"; shift 2 ;;
		--noc-memnic-out-buf)
			GOLEM_NOC_MEMNIC_OUTPUT_BUF_SIZE="$2"; shift 2 ;;
		--noc-link-bw)
			GOLEM_NOC_LINK_BW="$2"; shift 2 ;;
		--noc-xbar-bw)
			GOLEM_NOC_XBAR_BW="$2"; shift 2 ;;
		--noc-flit-size)
			GOLEM_NOC_FLIT_SIZE="$2"; shift 2 ;;
		--mesh-dim-x)
			GOLEM_MESH_DIM_X="$2"; shift 2 ;;
		--gm-buf)
			GOLEM_GM_BUFFER_LENGTH="$2"; shift 2 ;;
		--rocc-verbose)
			GOLEM_ROCC_VERBOSE="$2"; shift 2 ;;
		--gm-verbose)
			GOLEM_GM_VERBOSE="$2"; shift 2 ;;
		--gm-dump-data)
			GOLEM_GM_DUMP_DATA="$2"; shift 2 ;;
		--mvm-dump)
			GOLEM_MVM_DUMP_ENABLE=1; shift ;;
		--mvm-dump-dir)
			GOLEM_MVM_DUMP_DIR="$2"; shift 2 ;;
		--sim-mode)
			GOLEM_SIM_MODE="$2"; shift 2 ;;
		--memory-backend)
			GOLEM_MEMORY_BACKEND="$2"; shift 2 ;;
		--ramulator2-config)
			GOLEM_RAMULATOR2_CONFIG="$2"; shift 2 ;;
		--timeline)
			TIMELINE=1; shift ;;
		--print-core-map)
			PRINT_CORE_MAP=1; shift ;;
		--verify-mvm)
			VERIFY_MVM=1; shift ;;
		--verify-c)
			VERIFY_C=1; shift ;;
		--output-mode)
			GOLEM_OUTPUT_MODE="$2"; shift 2 ;;
		--tensor-source)
			TENSOR_SOURCE="$2"; shift 2 ;;
		--tensor-dir)
			TENSOR_DIR="$2"; shift 2 ;;
		--tensor-a)
			TENSOR_A_FILE="$2"; shift 2 ;;
		--tensor-b)
			TENSOR_B_FILE="$2"; shift 2 ;;
		--dump-c)
			DUMP_C_FILE="$2"; shift 2 ;;
		--hbm-dump-output)
			GOLEM_HBM_DUMP_OUTPUT="$2"; shift 2 ;;
		--no-hbm-dump-output)
			GOLEM_HBM_DUMP_OUTPUT=0; shift ;;
		--log)
			LOG_FILE="$2"; shift 2 ;;
			--mpi-ranks)
				GOLEM_MPI_RANKS="$2"; shift 2 ;;
			--sst-threads)
				GOLEM_SST_THREADS="$2"; shift 2 ;;
			--partition-strategy)
				GOLEM_PARTITION_STRATEGY="$2"; shift 2 ;;
			--partition-weight-profile)
				GOLEM_PARTITION_WEIGHT_PROFILE="$2"; shift 2 ;;
			--cpuset)
				GOLEM_CPUSET="$2"; shift 2 ;;
			--cpu-binding-mode)
				GOLEM_CPU_BINDING_MODE="$2"; shift 2 ;;
		--mpi-launcher)
			GOLEM_MPI_LAUNCHER="$2"; shift 2 ;;
		--mpi-args)
			GOLEM_MPI_ARGS="$2"; shift 2 ;;
		--mpi-partitioner)
			GOLEM_MPI_PARTITIONER="$2"; shift 2 ;;
		--dry-run)
			DRY_RUN=1; shift ;;
		-h|--help)
			show_help; exit 0 ;;
		*)
			echo "[ERROR] Unknown option: $1" >&2
			show_help
			exit 1 ;;
	esac
done

if ! [[ "$GOLEM_MPI_RANKS" =~ ^[0-9]+$ ]] || [[ "$GOLEM_MPI_RANKS" -le 0 ]]; then
	echo "[ERROR] --mpi-ranks/GOLEM_MPI_RANKS 必须为正整数，收到: $GOLEM_MPI_RANKS" >&2
	exit 1
fi
if ! [[ "$GOLEM_SST_THREADS" =~ ^[0-9]+$ ]] || [[ "$GOLEM_SST_THREADS" -le 0 ]]; then
	echo "[ERROR] --sst-threads/GOLEM_SST_THREADS 必须为正整数，收到: $GOLEM_SST_THREADS" >&2
	exit 1
fi
if [[ -z "$GOLEM_MPI_ARGS_WAS_SET" && "$GOLEM_SST_THREADS" -gt 1 ]]; then
	GOLEM_MPI_ARGS="--bind-to core --map-by slot:PE=$GOLEM_SST_THREADS"
fi
if [[ "$GOLEM_MPI_RANKS" -gt 1 ]] && ! command -v "$GOLEM_MPI_LAUNCHER" >/dev/null 2>&1; then
	echo "[ERROR] MPI 启动器不可执行: $GOLEM_MPI_LAUNCHER" >&2
	exit 1
fi
if [[ "$GOLEM_PARTITION_STRATEGY" != "weighted_topology" && "$GOLEM_PARTITION_STRATEGY" != "round_robin" ]]; then
	echo "[ERROR] --partition-strategy 必须为 weighted_topology 或 round_robin，收到: $GOLEM_PARTITION_STRATEGY" >&2
	exit 1
fi
if [[ "$GOLEM_SIM_MODE" != "full-functional" && "$GOLEM_SIM_MODE" != "full-timing" ]]; then
	echo "[ERROR] --sim-mode/GOLEM_SIM_MODE 必须为 full-functional 或 full-timing，收到: $GOLEM_SIM_MODE" >&2
	exit 1
fi
if [[ "$GOLEM_MEMORY_BACKEND" != "dramsim3" && "$GOLEM_MEMORY_BACKEND" != "ramulator2" ]]; then
	echo "[ERROR] --memory-backend/GOLEM_MEMORY_BACKEND 必须为 dramsim3 或 ramulator2，收到: $GOLEM_MEMORY_BACKEND" >&2
	exit 1
fi
if [[ "$GOLEM_MEMORY_BACKEND" == "ramulator2" && ! -f "$GOLEM_RAMULATOR2_CONFIG" ]]; then
	echo "[ERROR] Ramulator2 配置不存在: $GOLEM_RAMULATOR2_CONFIG" >&2
	exit 1
fi
if [[ "$TIMELINE" != "0" && "$TIMELINE" != "1" ]]; then
	echo "[ERROR] --timeline/GOLEM_TIMELINE 必须为 0 或 1，收到: $TIMELINE" >&2
	exit 1
fi
if [[ -n "$GOLEM_PARTITION_WEIGHT_PROFILE" ]]; then
	if [[ ! -f "$GOLEM_PARTITION_WEIGHT_PROFILE" ]]; then
		echo "[ERROR] partition weight profile 不存在: $GOLEM_PARTITION_WEIGHT_PROFILE" >&2
		exit 1
	fi
	GOLEM_PARTITION_WEIGHT_PROFILE="$(realpath "$GOLEM_PARTITION_WEIGHT_PROFILE")"
fi

if [[ "$ARRAY_IN_SET" -eq 0 ]]; then
	GOLEM_ARRAY_INPUT_SIZE="${GOLEM_ARRAY_INPUT_SIZE:-4}"
fi
if [[ "$ARRAY_OUT_SET" -eq 0 ]]; then
	GOLEM_ARRAY_OUTPUT_SIZE="${GOLEM_ARRAY_OUTPUT_SIZE:-4}"
fi

for n in "$GOLEM_TOTAL_GROUPS" "$GOLEM_ARRAY_INPUT_SIZE" "$GOLEM_ARRAY_OUTPUT_SIZE" "$GOLEM_NUM_ARRAYS" "$GOLEM_TOTAL_CORES" "$GOLEM_TOTAL_GEMM_CORES" "$GOLEM_NUM_MEMORY_NODES" "$GOLEM_GLOBAL_STRIDE_KB"; do
	if ! [[ "$n" =~ ^[0-9]+$ ]] || [[ "$n" -le 0 ]]; then
		echo "[ERROR] 参数必须为正整数，收到: $n" >&2
		exit 1
	fi
done

for n in "$GOLEM_GM_C_BUFFER_BYTES" "$GOLEM_GM_C_BUFFER_READ_BPC" "$GOLEM_GM_C_BUFFER_WRITE_BPC"; do
	if ! [[ "$n" =~ ^[0-9]+$ ]] || [[ "$n" -le 0 ]]; then
		echo "[ERROR] C-buffer capacity/bandwidth must be positive integers, got: $n" >&2
		exit 1
	fi
done
if ! [[ "$GOLEM_GM_C_BUFFER_LATENCY_CYCLES" =~ ^[0-9]+$ ]]; then
	echo "[ERROR] GOLEM_GM_C_BUFFER_LATENCY_CYCLES must be a non-negative integer, got: $GOLEM_GM_C_BUFFER_LATENCY_CYCLES" >&2
	exit 1
fi

if [[ "$GOLEM_MEM_NODE_SIZE_BYTES" != "auto" ]]; then
	if ! [[ "$GOLEM_MEM_NODE_SIZE_BYTES" =~ ^[0-9]+$ ]] || [[ "$GOLEM_MEM_NODE_SIZE_BYTES" -le 0 ]]; then
		echo "[ERROR] GOLEM_MEM_NODE_SIZE_BYTES 必须为正整数或 auto，收到: $GOLEM_MEM_NODE_SIZE_BYTES" >&2
		exit 1
	fi
fi

# Backward compatibility: allow GOLEM_GLOBAL_STRIDE_BYTES override.
if [[ -n "${GOLEM_GLOBAL_STRIDE_BYTES+x}" ]]; then
	if ! [[ "$GOLEM_GLOBAL_STRIDE_BYTES" =~ ^[0-9]+$ ]] || [[ "$GOLEM_GLOBAL_STRIDE_BYTES" -le 0 ]]; then
		echo "[ERROR] GOLEM_GLOBAL_STRIDE_BYTES 必须为正整数，收到: $GOLEM_GLOBAL_STRIDE_BYTES" >&2
		exit 1
	fi
	if (( GOLEM_GLOBAL_STRIDE_BYTES % 1024 != 0 )); then
		echo "[ERROR] GOLEM_GLOBAL_STRIDE_BYTES 必须是 1024 的整数倍，收到: $GOLEM_GLOBAL_STRIDE_BYTES" >&2
		exit 1
	fi
	GOLEM_GLOBAL_STRIDE_KB=$(( GOLEM_GLOBAL_STRIDE_BYTES / 1024 ))
fi

GOLEM_GLOBAL_STRIDE_BYTES=$(( GOLEM_GLOBAL_STRIDE_KB * 1024 ))

for n in "$GOLEM_GEMM_M" "$GOLEM_GEMM_N" "$GOLEM_GEMM_K"; do
	if ! [[ "$n" =~ ^[0-9]+$ ]] || [[ "$n" -le 0 ]]; then
		echo "[ERROR] GEMM 维度必须为正整数，收到: $n" >&2
		exit 1
	fi
done

for n in "$GOLEM_ORIG_M" "$GOLEM_ORIG_N" "$GOLEM_ORIG_K"; do
	if [[ -n "$n" ]] && { ! [[ "$n" =~ ^[0-9]+$ ]] || [[ "$n" -le 0 ]]; }; then
		echo "[ERROR] 原始维度 --orig-m/--orig-n/--orig-k 必须为正整数，收到: $n" >&2
		exit 1
	fi
done

if [[ "$TENSOR_SOURCE" != "synthetic" && "$TENSOR_SOURCE" != "file" && "$TENSOR_SOURCE" != "sample" ]]; then
	echo "[ERROR] --tensor-source 只能是 synthetic|file|sample，收到: $TENSOR_SOURCE" >&2
	exit 1
fi

if [[ "$GOLEM_MATMUL_TRANSPOSE_B" != "0" && "$GOLEM_MATMUL_TRANSPOSE_B" != "1" ]]; then
	echo "[ERROR] --transpose-b 必须为 0 或 1，收到: $GOLEM_MATMUL_TRANSPOSE_B" >&2
	exit 1
fi

if [[ "$TENSOR_SOURCE" == "synthetic" && ( -n "$TENSOR_A_FILE" || -n "$TENSOR_B_FILE" ) ]]; then
	TENSOR_SOURCE="file"
fi

if [[ "$TENSOR_SOURCE" == "sample" ]]; then
	if [[ -z "$TENSOR_A_FILE" ]]; then
		TENSOR_A_FILE="$TENSOR_DIR/a.bin"
	fi
	if [[ -z "$TENSOR_B_FILE" ]]; then
		TENSOR_B_FILE="$TENSOR_DIR/b.bin"
	fi
fi

if [[ "$TENSOR_SOURCE" == "file" ]] && { [[ -n "$TENSOR_A_FILE" && -z "$TENSOR_B_FILE" ]] || [[ -z "$TENSOR_A_FILE" && -n "$TENSOR_B_FILE" ]]; }; then
	echo "[ERROR] --tensor-a 和 --tensor-b 必须同时提供" >&2
	exit 1
fi

if [[ "$VERIFY_C" -eq 1 ]]; then
	if [[ "$TENSOR_SOURCE" == "synthetic" ]]; then
		echo "[ERROR] --verify-c 需要 --tensor-source file 或 sample（synthetic 模式无输入文件可对齐）" >&2
		exit 1
	fi
	if [[ -z "$TENSOR_A_FILE" || -z "$TENSOR_B_FILE" ]]; then
		echo "[ERROR] --verify-c 需要同时提供 --tensor-a 和 --tensor-b" >&2
		exit 1
	fi
	if [[ -z "$DUMP_C_FILE" ]]; then
		DUMP_C_FILE="$STATS_DIR/c_out.csv"
		DUMP_C_AUTO=1
	fi
fi

if [[ "$GOLEM_HBM_DUMP_OUTPUT" != "0" && "$GOLEM_HBM_DUMP_OUTPUT" != "1" ]]; then
	echo "[ERROR] GOLEM_HBM_DUMP_OUTPUT 必须为 0 或 1，收到: $GOLEM_HBM_DUMP_OUTPUT" >&2
	exit 1
fi

if [[ "$GOLEM_FINAL_C_WRITE_ENABLE" != "0" && "$GOLEM_FINAL_C_WRITE_ENABLE" != "1" ]]; then
	echo "[ERROR] GOLEM_FINAL_C_WRITE_ENABLE 必须为 0 或 1，收到: $GOLEM_FINAL_C_WRITE_ENABLE" >&2
	exit 1
fi


if [[ -z "$GOLEM_OUTPUT_MODE" ]]; then
	if [[ "$GOLEM_FINAL_C_WRITE_ENABLE" -eq 1 ]]; then
		GOLEM_OUTPUT_MODE="hbm"
	else
		GOLEM_OUTPUT_MODE="fusion"
	fi
fi
GOLEM_OUTPUT_MODE="${GOLEM_OUTPUT_MODE,,}"
if [[ "$GOLEM_OUTPUT_MODE" != "hbm" && "$GOLEM_OUTPUT_MODE" != "fusion" ]]; then
	echo "[ERROR] GOLEM_OUTPUT_MODE 必须为 hbm 或 fusion，收到: $GOLEM_OUTPUT_MODE" >&2
	exit 1
fi
if [[ "$GOLEM_OUTPUT_MODE" == "hbm" ]]; then
	GOLEM_FINAL_C_WRITE_ENABLE=1
else
	GOLEM_FINAL_C_WRITE_ENABLE=0
	GOLEM_HBM_DUMP_OUTPUT=0
fi

if ! [[ "$GOLEM_DMA_WRITE_VN" =~ ^[0-2]$ ]]; then
	echo "[ERROR] GOLEM_DMA_WRITE_VN 必须为 0、1 或 2，收到: $GOLEM_DMA_WRITE_VN" >&2
	exit 1
fi

if [[ "$GOLEM_NOC_VN_PRIORITY_ENABLE" != "0" && "$GOLEM_NOC_VN_PRIORITY_ENABLE" != "1" ]]; then
	echo "[ERROR] GOLEM_NOC_VN_PRIORITY_ENABLE 必须为 0 或 1，收到: $GOLEM_NOC_VN_PRIORITY_ENABLE" >&2
	exit 1
fi
if [[ "$GOLEM_NOC_VN_PRIORITY_ENABLE" == "1" &&
      ! "$GOLEM_NOC_VN_PRIORITY_ORDER" =~ ^(0,1,2|0,2,1|1,0,2|1,2,0|2,0,1|2,1,0)$ ]]; then
	echo "[ERROR] GOLEM_NOC_VN_PRIORITY_ORDER 必须是三个 VN 的逗号分隔顺序，例如 1,0,2" >&2
	exit 1
fi
if ! [[ "$GOLEM_NOC_VN_STARVATION_VN" =~ ^[0-2]$ ]]; then
	echo "[ERROR] GOLEM_NOC_VN_STARVATION_VN 必须为 0、1 或 2，收到: $GOLEM_NOC_VN_STARVATION_VN" >&2
	exit 1
fi
if ! [[ "$GOLEM_NOC_VN_MAX_STARVATION_CYCLES" =~ ^[0-9]+$ ]]; then
	echo "[ERROR] GOLEM_NOC_VN_MAX_STARVATION_CYCLES 必须为非负整数" >&2
	exit 1
fi

if [[ "$GOLEM_OUTPUT_MODE" == "hbm" && "$GOLEM_HBM_DUMP_OUTPUT" -eq 0 && -n "$DUMP_C_FILE" ]]; then
	echo "[ERROR] GOLEM_HBM_DUMP_OUTPUT=0 时不会生成 hbm_out_node*.bin，不能使用 --dump-c/--verify-c" >&2
	exit 1
fi

if [[ "$GOLEM_SIM_MODE" == "full-timing" ]]; then
	if [[ "$VERIFY_MVM" -eq 1 || "$VERIFY_C" -eq 1 || "$GOLEM_MVM_DUMP_ENABLE" -eq 1 || -n "$DUMP_C_FILE" ]]; then
		echo "[ERROR] full-timing 跳过数值 MAC，不能使用 --verify-mvm/--verify-c/--mvm-dump/--dump-c" >&2
		exit 1
	fi
	CONFIG_NOTICES+=("full-timing: host numerical MACs disabled; output values are placeholders")
fi

for n in "$GOLEM_GEMM_BLOCK_M" "$GOLEM_GEMM_BLOCK_N" "$GOLEM_GEMM_BLOCK_K"; do
	if ! [[ "$n" =~ ^[0-9]+$ ]] || [[ "$n" -le 0 ]]; then
		echo "[ERROR] GEMM block 维度必须为正整数，收到: $n" >&2
		exit 1
	fi
done

if (( GOLEM_GEMM_M % GOLEM_GEMM_BLOCK_M != 0 || GOLEM_GEMM_N % GOLEM_GEMM_BLOCK_N != 0 || GOLEM_GEMM_K % GOLEM_GEMM_BLOCK_K != 0 )); then
	echo "[ERROR] GEMM M/N/K 必须可被 block_M/N/K 整除。收到 M/N/K=${GOLEM_GEMM_M}/${GOLEM_GEMM_N}/${GOLEM_GEMM_K}, block=${GOLEM_GEMM_BLOCK_M}/${GOLEM_GEMM_BLOCK_N}/${GOLEM_GEMM_BLOCK_K}" >&2
	exit 1
fi

GOLEM_ELEM_BYTES="$(dtype_nbytes "$GOLEM_MATMUL_DTYPE")"
if (( GOLEM_ELEM_BYTES <= 0 )); then
	echo "[ERROR] 不支持的 dtype: $GOLEM_MATMUL_DTYPE" >&2
	exit 1
fi

if [[ -z "$GOLEM_ORIG_M" ]]; then
	GOLEM_ORIG_M="$GOLEM_GEMM_M"
fi
if [[ -z "$GOLEM_ORIG_N" ]]; then
	GOLEM_ORIG_N="$GOLEM_GEMM_N"
fi
if [[ -z "$GOLEM_ORIG_K" ]]; then
	GOLEM_ORIG_K="$GOLEM_GEMM_K"
fi

if [[ "$GOLEM_ATTENTION_FUSED" != "1" ]] &&
   (( GOLEM_GEMM_BLOCK_M % GOLEM_ARRAY_OUTPUT_SIZE != 0 ||
      (GOLEM_GEMM_BLOCK_K > GOLEM_ARRAY_INPUT_SIZE &&
       GOLEM_GEMM_BLOCK_K % GOLEM_ARRAY_INPUT_SIZE != 0) )); then
	echo "[ERROR] 当前运行要求 block_M 是 ARRAY_OUTPUT 的整数倍；block_K 必须不大于 ARRAY_INPUT，或是 ARRAY_INPUT 的整数倍。收到 block_M=$GOLEM_GEMM_BLOCK_M block_K=$GOLEM_GEMM_BLOCK_K, ARRAY_OUTPUT/INPUT=$GOLEM_ARRAY_OUTPUT_SIZE/$GOLEM_ARRAY_INPUT_SIZE" >&2
	exit 1
fi

if (( GOLEM_GEMM_BLOCK_K < GOLEM_ARRAY_INPUT_SIZE )) && [[ "$GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE" != "1" ]]; then
	echo "[ERROR] block_K 小于 ARRAY_INPUT 时需要 GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE=1，以完成有效 lane 装载和其余 lane 清零。" >&2
	exit 1
fi

if [[ "$GOLEM_ATTENTION_FUSED" != "1" &&
      "$GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE" == "1" &&
      "$GOLEM_GEMM_BLOCK_M" -ne "$GOLEM_ARRAY_OUTPUT_SIZE" ]]; then
	echo "[ERROR] 当前 WorkerCommandProcessor 仅支持 block_M == ARRAY_OUTPUT_SIZE。" >&2
	echo "        收到 block_M=$GOLEM_GEMM_BLOCK_M, ARRAY_OUTPUT_SIZE=$GOLEM_ARRAY_OUTPUT_SIZE。" >&2
	echo "        block_M 大于硬件输出宽度的 M 向 micro-tiling 尚未实现；继续运行会使 worker 停在 wcp_wait。" >&2
	exit 1
fi
if ! [[ "$GOLEM_WORKER_START_BARRIER_ENABLE" =~ ^[01]$ ]]; then
	echo "[ERROR] GOLEM_WORKER_START_BARRIER_ENABLE 必须为 0 或 1，收到: $GOLEM_WORKER_START_BARRIER_ENABLE" >&2
	exit 1
fi
if ! [[ "$GOLEM_WORKER_START_GUARD_CYCLES" =~ ^[0-9]+$ ]] || [[ "$GOLEM_WORKER_START_GUARD_CYCLES" -lt 1 ]]; then
	echo "[ERROR] GOLEM_WORKER_START_GUARD_CYCLES 必须为正整数，收到: $GOLEM_WORKER_START_GUARD_CYCLES" >&2
	exit 1
fi

if [[ "$GOLEM_GEMM_BLOCK_N" -gt "$GOLEM_NUM_ARRAYS" ]]; then
	echo "[ERROR] 当前 phase-1 要求 block_N <= GOLEM_NUM_ARRAYS($GOLEM_NUM_ARRAYS)，收到 block_N=$GOLEM_GEMM_BLOCK_N" >&2
	exit 1
fi

if ! [[ "$GOLEM_DMA_SLOT_COUNT" =~ ^[0-9]+$ ]] || [[ "$GOLEM_DMA_SLOT_COUNT" -lt 4 ]]; then
	echo "[ERROR] GOLEM_DMA_SLOT_COUNT 必须为 >=4 的正整数，收到: $GOLEM_DMA_SLOT_COUNT" >&2
	exit 1
fi

# Keep per-core local GM layout within the reserved global stride.
ELEM_BYTES="$GOLEM_ELEM_BYTES"
LOCAL_ALIGN=256
LOCAL_DATA_BASE_DEC=$((0x2000))
mat_slot_bytes=$(align_up_int $(( GOLEM_GEMM_BLOCK_M * GOLEM_GEMM_BLOCK_K * ELEM_BYTES )) $LOCAL_ALIGN)
vec_slot_bytes=$(align_up_int $(( GOLEM_GEMM_BLOCK_N * GOLEM_GEMM_BLOCK_K * ELEM_BYTES )) $LOCAL_ALIGN)
out_tile_bytes=$(align_up_int $(( GOLEM_GEMM_BLOCK_M * GOLEM_GEMM_BLOCK_N * ELEM_BYTES )) $LOCAL_ALIGN)
out_vec_bytes=$(align_up_int $(( GOLEM_ARRAY_OUTPUT_SIZE * ELEM_BYTES )) $LOCAL_ALIGN)
out_scratch_bytes=$(( out_vec_bytes > out_tile_bytes ? out_vec_bytes : out_tile_bytes ))
gemm_m_tiles=$(( GOLEM_GEMM_M / GOLEM_GEMM_BLOCK_M ))
gemm_n_tiles=$(( GOLEM_GEMM_N / GOLEM_GEMM_BLOCK_N ))
partial_rows=$(( GOLEM_B_REUSE_M_TILES < gemm_m_tiles ? GOLEM_B_REUSE_M_TILES : gemm_m_tiles ))
partial_cols=$(( GOLEM_A_REUSE_N_TILES < gemm_n_tiles ? GOLEM_A_REUSE_N_TILES : gemm_n_tiles ))
partial_tile_count=$(( partial_rows * partial_cols ))
required_global_stride_bytes=$(( LOCAL_DATA_BASE_DEC + GOLEM_DMA_SLOT_COUNT * mat_slot_bytes + GOLEM_DMA_SLOT_COUNT * vec_slot_bytes + out_scratch_bytes + partial_tile_count * out_tile_bytes + 0x40 + LOCAL_ALIGN ))
if (( GOLEM_GLOBAL_STRIDE_BYTES < required_global_stride_bytes )); then
	CONFIG_NOTICES+=("Expanded global stride ${GOLEM_GLOBAL_STRIDE_BYTES} -> ${required_global_stride_bytes} bytes (slots=${GOLEM_DMA_SLOT_COUNT})")
	GOLEM_GLOBAL_STRIDE_BYTES=$required_global_stride_bytes
	GOLEM_GLOBAL_STRIDE_KB=$(( (GOLEM_GLOBAL_STRIDE_BYTES + 1023) / 1024 ))
fi

if [[ "$GOLEM_BIAS_ENABLE" != "0" && "$GOLEM_BIAS_ENABLE" != "1" ]]; then
	echo "[ERROR] --bias-enable 仅支持 0/1，收到: $GOLEM_BIAS_ENABLE" >&2
	exit 1
fi

if [[ "$GOLEM_MATMUL_DTYPE" != "int32" && "$GOLEM_MATMUL_DTYPE" != "fp16" && "$GOLEM_MATMUL_DTYPE" != "fp32" ]]; then
	echo "[ERROR] --dtype 仅支持 int32|fp16|fp32，收到: $GOLEM_MATMUL_DTYPE" >&2
	exit 1
fi

# FP16 4x4 has half-sized panels but the same fixed per-window control and
# response latency as FP32.  With 32 slots, three resident buffers leave only
# K=2, which exposes a return-path bubble at 2048.  Use two buffers and K=4 so
# the compute span covers the observed RTT; explicit caller settings win.
if [[ "$GOLEM_MATMUL_DTYPE" == "int32" ]]; then
	if ! [[ "$GOLEM_BIAS_VALUE" =~ ^-?[0-9]+$ ]]; then
		echo "[ERROR] int32 模式下 --bias-value 必须为整数，收到: $GOLEM_BIAS_VALUE" >&2
		exit 1
	fi
elif ! [[ "$GOLEM_BIAS_VALUE" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
	echo "[ERROR] fp32 模式下 --bias-value 必须为数字，收到: $GOLEM_BIAS_VALUE" >&2
	exit 1
fi

if [[ -z "${GOLEM_ROCC_TYPE+x}" || "$GOLEM_ROCC_TYPE" == "golem.RoCCAnalogInt" ]]; then
	if [[ "$GOLEM_MATMUL_DTYPE" == "fp32" || "$GOLEM_MATMUL_DTYPE" == "fp16" ]]; then
		GOLEM_ROCC_TYPE="golem.RoCCAnalogFloat"
	else
		GOLEM_ROCC_TYPE="golem.RoCCAnalogInt"
	fi
fi

if [[ -z "${GOLEM_ARRAY_TYPE+x}" || "$GOLEM_ARRAY_TYPE" == "golem.MVMIntArray" ]]; then
	if [[ "$GOLEM_MATMUL_DTYPE" == "fp32" || "$GOLEM_MATMUL_DTYPE" == "fp16" ]]; then
		GOLEM_ARRAY_TYPE="golem.MVMFloatArray"
	else
		GOLEM_ARRAY_TYPE="golem.MVMIntArray"
	fi
fi

DERIVED_GEMM_K_TILES=$(( GOLEM_GEMM_K / GOLEM_GEMM_BLOCK_K ))
if [[ "$DERIVED_GEMM_K_TILES" -le 0 ]]; then
	echo "[ERROR] 派生 GEMM K tiles 必须为正整数，收到: $DERIVED_GEMM_K_TILES" >&2
	exit 1
fi

if ! [[ "$GOLEM_A_REUSE_N_TILES" =~ ^[0-9]+$ ]] || [[ "$GOLEM_A_REUSE_N_TILES" -le 0 ]]; then
	echo "[ERROR] GOLEM_A_REUSE_N_TILES 必须为正整数，收到: $GOLEM_A_REUSE_N_TILES" >&2
	exit 1
fi

if ! [[ "$GOLEM_B_REUSE_M_TILES" =~ ^[0-9]+$ ]] || [[ "$GOLEM_B_REUSE_M_TILES" -le 0 ]]; then
	echo "[ERROR] GOLEM_B_REUSE_M_TILES 必须为正整数，收到: $GOLEM_B_REUSE_M_TILES" >&2
	exit 1
fi

if [[ "$GOLEM_A_REUSE_N_TILES" -gt 1 && "$GOLEM_B_REUSE_M_TILES" -gt 1 ]]; then
	required_c_buffer_bytes=$(( GOLEM_A_REUSE_N_TILES * GOLEM_B_REUSE_M_TILES * GOLEM_GEMM_BLOCK_M * GOLEM_GEMM_BLOCK_N * GOLEM_ELEM_BYTES ))
	if [[ "$GOLEM_GM_C_BUFFER_BYTES" -lt "$required_c_buffer_bytes" ]]; then
		echo "[ERROR] GOLEM_GM_C_BUFFER_BYTES=$GOLEM_GM_C_BUFFER_BYTES is too small for ${GOLEM_B_REUSE_M_TILES}x${GOLEM_A_REUSE_N_TILES} reuse; required=$required_c_buffer_bytes" >&2
		exit 1
	fi
fi

DERIVED_GEMM_M_TILES=$(( GOLEM_GEMM_M / GOLEM_GEMM_BLOCK_M ))
DERIVED_GEMM_N_TILES=$(( GOLEM_GEMM_N / GOLEM_GEMM_BLOCK_N ))
DERIVED_GEMM_TOTAL_TASKS=$(( DERIVED_GEMM_M_TILES * DERIVED_GEMM_N_TILES ))
DERIVED_GEMM_M_GROUPS=$(( (DERIVED_GEMM_M_TILES + GOLEM_B_REUSE_M_TILES - 1) / GOLEM_B_REUSE_M_TILES ))
DERIVED_GEMM_N_GROUPS=$(( (DERIVED_GEMM_N_TILES + GOLEM_A_REUSE_N_TILES - 1) / GOLEM_A_REUSE_N_TILES ))
DERIVED_GEMM_TOTAL_MACRO_TASKS=$(( DERIVED_GEMM_M_GROUPS * DERIVED_GEMM_N_GROUPS ))

GOLEM_MEM_NODE_SIZE_BYTES_AUTO=0
if [[ "$GOLEM_MEM_NODE_SIZE_BYTES" == "auto" ]]; then
	GOLEM_MEM_NODE_SIZE_BYTES_AUTO=1
	read -r GOLEM_MEM_NODE_SIZE_BYTES GOLEM_AUTO_MEM_REQUIRED_BYTES GOLEM_AUTO_MEM_DATA_END_BYTES GOLEM_AUTO_MEM_BIAS_STRIDE_BYTES < <(
		derive_auto_mem_node_size \
			"$GOLEM_GEMM_M" "$GOLEM_GEMM_N" "$GOLEM_GEMM_K" \
			"$GOLEM_GEMM_BLOCK_M" "$GOLEM_GEMM_BLOCK_N" "$GOLEM_GEMM_BLOCK_K" \
			"$GOLEM_ELEM_BYTES" "$GOLEM_NUM_MEMORY_NODES" "$GOLEM_TOTAL_GROUPS" \
			"$GOLEM_TOTAL_GEMM_CORES" "$GOLEM_GROUP_MANAGER_ENABLE" \
			"$GOLEM_A_REUSE_N_TILES" "$GOLEM_B_REUSE_M_TILES"
	)
	echo "[INFO] Auto-selected GOLEM_MEM_NODE_SIZE_BYTES=$GOLEM_MEM_NODE_SIZE_BYTES (required=$GOLEM_AUTO_MEM_REQUIRED_BYTES, data_end=$GOLEM_AUTO_MEM_DATA_END_BYTES, bias_stride=$GOLEM_AUTO_MEM_BIAS_STRIDE_BYTES)"
fi

if ! [[ "$GOLEM_MEM_NODE_SIZE_BYTES" =~ ^[0-9]+$ ]] || [[ "$GOLEM_MEM_NODE_SIZE_BYTES" -le 0 ]]; then
	echo "[ERROR] GOLEM_MEM_NODE_SIZE_BYTES 必须为正整数或 auto，收到: $GOLEM_MEM_NODE_SIZE_BYTES" >&2
	exit 1
fi

GOLEM_IDENTITY_BASE="$GOLEM_MEM_NODE_SIZE_BYTES"

if ! [[ "$GOLEM_DMA_WINDOW_K_TILES" =~ ^[0-9]+$ ]] || [[ "$GOLEM_DMA_WINDOW_K_TILES" -le 0 ]]; then
	echo "[ERROR] GOLEM_DMA_WINDOW_K_TILES 必须为正整数，收到: $GOLEM_DMA_WINDOW_K_TILES" >&2
	exit 1
fi


DERIVED_WCP_RESIDENT_K_TILES=$GOLEM_DMA_WINDOW_K_TILES
if [[ "$DERIVED_WCP_RESIDENT_K_TILES" -gt "$DERIVED_GEMM_K_TILES" ]]; then
	DERIVED_WCP_RESIDENT_K_TILES=$DERIVED_GEMM_K_TILES
fi

if [[ "$GOLEM_A_REUSE_N_TILES" -gt 1 && "$GOLEM_B_REUSE_M_TILES" -gt 1 ]]; then
	# The worker command processor uses slot-driven K windows for 2D reuse, so local slots
	# only need to hold active+prefetch resident windows rather than all K tiles at once.
	if [[ "${GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE:-1}" != "0" ]]; then
		wcp_prefetch_windows=$GOLEM_WCP_PREFETCH_WINDOWS
		if ! [[ "$wcp_prefetch_windows" =~ ^[0-9]+$ ]] || [[ "$wcp_prefetch_windows" -le 0 ]]; then
			echo "[ERROR] GOLEM_WCP_PREFETCH_WINDOWS 必须为正整数，收到: $GOLEM_WCP_PREFETCH_WINDOWS" >&2
			exit 1
		fi
		wcp_window_buffers=$(( wcp_prefetch_windows + 1 ))
		if [[ "$wcp_window_buffers" -lt 2 ]]; then
			wcp_window_buffers=2
		fi
		mat_resident_k=$(( GOLEM_DMA_SLOT_COUNT / (wcp_window_buffers * GOLEM_B_REUSE_M_TILES) ))
		vec_resident_k=$(( GOLEM_DMA_SLOT_COUNT / (wcp_window_buffers * GOLEM_A_REUSE_N_TILES) ))
		if [[ "$mat_resident_k" -le 0 || "$vec_resident_k" -le 0 ]]; then
			echo "[ERROR] 当前 2D K-window 要求 local_slot_count 至少容纳 active+prefetch residentK=1，收到 slots=$GOLEM_DMA_SLOT_COUNT A=$GOLEM_A_REUSE_N_TILES B=$GOLEM_B_REUSE_M_TILES prefetch_windows=$GOLEM_WCP_PREFETCH_WINDOWS buffers=$wcp_window_buffers" >&2
			exit 1
		fi
		slot_resident_k=$mat_resident_k
		if [[ "$vec_resident_k" -lt "$slot_resident_k" ]]; then
			slot_resident_k=$vec_resident_k
		fi
		if [[ "$DERIVED_WCP_RESIDENT_K_TILES" -gt "$slot_resident_k" ]]; then
			DERIVED_WCP_RESIDENT_K_TILES=$slot_resident_k
		fi
	else
		mat_slots_needed=$(( GOLEM_B_REUSE_M_TILES * DERIVED_GEMM_K_TILES ))
		vec_slots_needed=$(( GOLEM_A_REUSE_N_TILES * DERIVED_GEMM_K_TILES ))
		if [[ "$mat_slots_needed" -gt "$GOLEM_DMA_SLOT_COUNT" || "$vec_slots_needed" -gt "$GOLEM_DMA_SLOT_COUNT" ]]; then
			echo "[ERROR] 当前 2D full-K 要求 reuse_m*k_tiles 和 reuse_n*k_tiles <= local_slot_count($GOLEM_DMA_SLOT_COUNT)，收到 mat=$mat_slots_needed vec=$vec_slots_needed" >&2
			exit 1
		fi
	fi
elif [[ "$GOLEM_A_REUSE_N_TILES" -gt 1 && "$DERIVED_GEMM_K_TILES" -gt "$GOLEM_DMA_SLOT_COUNT" ]]; then
	echo "[ERROR] 当前 A-reuse 第一版要求 GEMM_K_TILES <= local_slot_count($GOLEM_DMA_SLOT_COUNT)，收到 K tiles=$DERIVED_GEMM_K_TILES" >&2
	exit 1
elif [[ "$GOLEM_B_REUSE_M_TILES" -gt 1 && "$DERIVED_GEMM_K_TILES" -gt "$GOLEM_DMA_SLOT_COUNT" ]]; then
	echo "[ERROR] 当前 B-reuse 第一版要求 GEMM_K_TILES <= local_slot_count($GOLEM_DMA_SLOT_COUNT)，收到 K tiles=$DERIVED_GEMM_K_TILES" >&2
	exit 1
fi

if [[ "$GOLEM_NUM_MEMORY_NODES" -lt 2 ]]; then
	echo "[ERROR] --num-mem-nodes 至少为 2（1 个 OS 节点 + 至少 1 个数据节点）" >&2
	exit 1
fi

DATA_MEM_NODES=$(( GOLEM_NUM_MEMORY_NODES - 1 ))
if [[ "$DATA_MEM_NODES" -gt "$GOLEM_MESH_DIM_X" ]]; then
	echo "[ERROR] 数据节点数($DATA_MEM_NODES) 不能大于 --mesh-dim-x($GOLEM_MESH_DIM_X)，否则无法在单独数据行按列对齐放置 4 个 HBM 节点" >&2
	exit 1
fi

if [[ "$(basename "$GOLEM_ARCH_SCRIPT")" == "ncores_selfcom_dma_ctrl.py" &&
      "$GOLEM_EXPLICIT_PARTITION" == "1" && "$GOLEM_MESH_DIM_X" -ne 4 ]]; then
	echo "[ERROR] 当前显式分区布局要求 mesh_dim_x=4；收到 mesh_dim_x=$GOLEM_MESH_DIM_X。" >&2
	exit 1
fi

if [[ "$GOLEM_GROUP_MANAGER_ENABLE" == "1" && "$GOLEM_CTRL_LINK_ENABLE" == "1" && "$GOLEM_TOTAL_CORES" == "16" && "$GOLEM_TOTAL_GEMM_CORES" == "16" ]]; then
	GOLEM_TOTAL_CORES=$((GOLEM_TOTAL_CORES + GOLEM_MESH_DIM_X))
	GOLEM_TOTAL_GEMM_CORES="$GOLEM_TOTAL_CORES"
	CONFIG_NOTICES+=("Group-manager mode added one CPU row (${GOLEM_TOTAL_CORES} total cores)")
fi

if [[ "$(basename "$GOLEM_ARCH_SCRIPT")" == "ncores_selfcom_dma_ctrl.py" &&
      "$GOLEM_GROUP_MANAGER_ENABLE" == "1" && "$GOLEM_CTRL_LINK_ENABLE" == "1" ]]; then
	if [[ "$GOLEM_TOTAL_GROUPS" -ne 4 || "$GOLEM_MESH_DIM_X" -ne 4 ]]; then
		echo "[ERROR] 当前控制链路布局要求 total_groups=4 且 mesh_dim_x=4；收到 total_groups=$GOLEM_TOTAL_GROUPS, mesh_dim_x=$GOLEM_MESH_DIM_X。" >&2
		exit 1
	fi

	if [[ "$GOLEM_NUM_ARRAYS" == "1" ]]; then
		GOLEM_NUM_ARRAYS="$GOLEM_GEMM_BLOCK_N"
		CONFIG_NOTICES+=("Aligned numArrays to block_n (${GOLEM_NUM_ARRAYS})")
	fi

	if [[ "$GOLEM_NUM_ARRAYS" -lt "$GOLEM_GEMM_BLOCK_N" ]]; then
		echo "[ERROR] numArrays($GOLEM_NUM_ARRAYS) 必须 >= block_n($GOLEM_GEMM_BLOCK_N) 才能按 n_col 对齐 array_id" >&2
		exit 1
	fi

	required_active_workers=$((GOLEM_TOTAL_GROUPS * 4))
	actual_active_workers=$((GOLEM_TOTAL_GEMM_CORES - GOLEM_TOTAL_GROUPS))
	if [[ "$actual_active_workers" -ne "$required_active_workers" ]]; then
		echo "[ERROR] 当前 GroupCtrlEndpoint 管理器固定 4 个 worker slot (req_in_0..3)，需要 active_workers=${required_active_workers}。" >&2
		echo "        但当前 active_workers=${actual_active_workers} (gemm_cores=${GOLEM_TOTAL_GEMM_CORES}, groups=${GOLEM_TOTAL_GROUPS})。" >&2
		echo "        建议设置 GOLEM_TOTAL_GEMM_CORES=$((GOLEM_TOTAL_GROUPS * 5))（例如 groups=4 时为 20）。" >&2
		exit 1
	fi
fi

if [[ "$GOLEM_TOTAL_GEMM_CORES" -gt "$GOLEM_TOTAL_CORES" ]]; then
	echo "[ERROR] --gemm-cores($GOLEM_TOTAL_GEMM_CORES) 不能大于 --num-cores($GOLEM_TOTAL_CORES)" >&2
	exit 1
fi

if ! [[ "$GOLEM_DMA_NODE_CREDITS" =~ ^[0-9]+$ ]] || [[ "$GOLEM_DMA_NODE_CREDITS" -le 0 ]]; then
	echo "[ERROR] GOLEM_DMA_NODE_CREDITS 必须为正整数，收到: $GOLEM_DMA_NODE_CREDITS" >&2
	exit 1
fi

if ! [[ "$GOLEM_DMA_READ_RETRY_TICKS" =~ ^[0-9]+$ ]] || [[ "$GOLEM_DMA_READ_RETRY_TICKS" -le 0 ]]; then
	echo "[ERROR] --dma-read-retry-ticks 必须为正整数，收到: $GOLEM_DMA_READ_RETRY_TICKS" >&2
	exit 1
fi

if ! [[ "$GOLEM_DMA_READ_MAX_RETRIES" =~ ^[0-9]+$ ]]; then
	echo "[ERROR] --dma-read-max-retries 必须为非负整数，收到: $GOLEM_DMA_READ_MAX_RETRIES" >&2
	exit 1
fi

if ! [[ "$GOLEM_DMA_MAX_INFLIGHT" =~ ^[0-9]+$ ]] || [[ "$GOLEM_DMA_MAX_INFLIGHT" -le 0 ]]; then
	echo "[ERROR] --dma-max-inflight 必须为正整数，收到: $GOLEM_DMA_MAX_INFLIGHT" >&2
	exit 1
fi

if [[ "$GOLEM_PROGRESS_HEARTBEAT" != "0" && "$GOLEM_PROGRESS_HEARTBEAT" != "1" ]]; then
	echo "[ERROR] --progress-heartbeat 仅支持 0/1，收到: $GOLEM_PROGRESS_HEARTBEAT" >&2
	exit 1
fi

if ! [[ "$GOLEM_PROGRESS_INTERVAL_CYCLES" =~ ^[0-9]+$ ]] || [[ "$GOLEM_PROGRESS_INTERVAL_CYCLES" -le 0 ]]; then
	echo "[ERROR] --progress-interval-cycles 必须为正整数，收到: $GOLEM_PROGRESS_INTERVAL_CYCLES" >&2
	exit 1
fi

if [[ "$GOLEM_TERMINAL_VERBOSE" != "0" && "$GOLEM_TERMINAL_VERBOSE" != "1" ]]; then
	echo "[ERROR] GOLEM_TERMINAL_VERBOSE 仅支持 0/1，收到: $GOLEM_TERMINAL_VERBOSE" >&2
	exit 1
fi

if ! [[ "$GOLEM_TERMINAL_REFRESH_SECONDS" =~ ^[0-9]+$ ]] || [[ "$GOLEM_TERMINAL_REFRESH_SECONDS" -le 0 ]]; then
	echo "[ERROR] GOLEM_TERMINAL_REFRESH_SECONDS 必须为正整数，收到: $GOLEM_TERMINAL_REFRESH_SECONDS" >&2
	exit 1
fi

if ! [[ "$GOLEM_DMA_BURST_BYTES" =~ ^[0-9]+$ ]] || [[ "$GOLEM_DMA_BURST_BYTES" -le 0 ]]; then
	echo "[ERROR] --dma-burst-bytes 必须为正整数，收到: $GOLEM_DMA_BURST_BYTES" >&2
	exit 1
fi

if ! [[ "$GOLEM_DMA_PANEL_CHUNK_BYTES" =~ ^[0-9]+$ ]] || [[ "$GOLEM_DMA_PANEL_CHUNK_BYTES" -le 0 ]]; then
	echo "[ERROR] GOLEM_DMA_PANEL_CHUNK_BYTES 必须为正整数，收到: $GOLEM_DMA_PANEL_CHUNK_BYTES" >&2
	exit 1
fi

if ! [[ "$GOLEM_DMA_CREDIT_CHUNK_BYTES" =~ ^[0-9]+$ ]] || [[ "$GOLEM_DMA_CREDIT_CHUNK_BYTES" -le 0 ]]; then
	echo "[ERROR] GOLEM_DMA_CREDIT_CHUNK_BYTES 必须为正整数，收到: $GOLEM_DMA_CREDIT_CHUNK_BYTES" >&2
	exit 1
fi

if ! [[ "$GOLEM_DMA_WINDOW_PRIORITY_ENABLE" =~ ^[01]$ ]]; then
	echo "[ERROR] GOLEM_DMA_WINDOW_PRIORITY_ENABLE 必须为 0 或 1，收到: $GOLEM_DMA_WINDOW_PRIORITY_ENABLE" >&2
	exit 1
fi

if ! [[ "$GOLEM_DMA_WINDOW_REORDER_CYCLES" =~ ^[0-9]+$ ]]; then
	echo "[ERROR] GOLEM_DMA_WINDOW_REORDER_CYCLES 必须为非负整数，收到: $GOLEM_DMA_WINDOW_REORDER_CYCLES" >&2
	exit 1
fi

if ! [[ "$GOLEM_DMA_TILE_CHUNK_QUANTUM" =~ ^[0-9]+$ ]] || [[ "$GOLEM_DMA_TILE_CHUNK_QUANTUM" -le 0 ]]; then
	echo "[ERROR] GOLEM_DMA_TILE_CHUNK_QUANTUM 必须为正整数，收到: $GOLEM_DMA_TILE_CHUNK_QUANTUM" >&2
	exit 1
fi

if ! [[ "$GOLEM_DMA_RESPONSE_TILE_PRIORITY_ENABLE" =~ ^[01]$ ]]; then
	echo "[ERROR] GOLEM_DMA_RESPONSE_TILE_PRIORITY_ENABLE 必须为 0 或 1，收到: $GOLEM_DMA_RESPONSE_TILE_PRIORITY_ENABLE" >&2
	exit 1
fi

if ! [[ "$GOLEM_DMA_RESPONSE_REORDER_CYCLES" =~ ^[0-9]+$ ]]; then
	echo "[ERROR] GOLEM_DMA_RESPONSE_REORDER_CYCLES 必须为非负整数，收到: $GOLEM_DMA_RESPONSE_REORDER_CYCLES" >&2
	exit 1
fi

if ! [[ "$GOLEM_DMA_RESPONSE_MAX_STARVATION_CYCLES" =~ ^[0-9]+$ ]]; then
	echo "[ERROR] GOLEM_DMA_RESPONSE_MAX_STARVATION_CYCLES 必须为非负整数，收到: $GOLEM_DMA_RESPONSE_MAX_STARVATION_CYCLES" >&2
	exit 1
fi

if ! [[ "$GOLEM_DMA_ADMISSION_MAX_STARVATION_CYCLES" =~ ^[0-9]+$ ]]; then
	echo "[ERROR] GOLEM_DMA_ADMISSION_MAX_STARVATION_CYCLES 必须为非负整数，收到: $GOLEM_DMA_ADMISSION_MAX_STARVATION_CYCLES" >&2
	exit 1
fi

if ! [[ "$GOLEM_DMA_RESPONSE_DRAIN_LIMIT" =~ ^[0-9]+$ ]]; then
	echo "[ERROR] GOLEM_DMA_RESPONSE_DRAIN_LIMIT 必须为非负整数，收到: $GOLEM_DMA_RESPONSE_DRAIN_LIMIT" >&2
	exit 1
fi

if ! [[ "$GOLEM_SCHED_ISSUE_BUDGET_PER_TICK" =~ ^[0-9]+$ ]] || [[ "$GOLEM_SCHED_ISSUE_BUDGET_PER_TICK" -le 0 ]]; then
	echo "[ERROR] GOLEM_SCHED_ISSUE_BUDGET_PER_TICK 必须为正整数，收到: $GOLEM_SCHED_ISSUE_BUDGET_PER_TICK" >&2
	exit 1
fi

if [[ -z "$GOLEM_DMA_NODE_CHUNK_CREDITS" ]]; then
	mat_sched_transfer_bytes=$(( GOLEM_GEMM_BLOCK_M * GOLEM_GEMM_BLOCK_K * ELEM_BYTES ))
	vec_sched_transfer_bytes=$(( GOLEM_GEMM_BLOCK_N * GOLEM_GEMM_BLOCK_K * ELEM_BYTES ))
	mat_transfer_credit_chunks=$(( (mat_sched_transfer_bytes + GOLEM_DMA_CREDIT_CHUNK_BYTES - 1) / GOLEM_DMA_CREDIT_CHUNK_BYTES ))
	vec_transfer_credit_chunks=$(( (vec_sched_transfer_bytes + GOLEM_DMA_CREDIT_CHUNK_BYTES - 1) / GOLEM_DMA_CREDIT_CHUNK_BYTES ))
	wcp_credit_window_buffers=$(( GOLEM_WCP_PREFETCH_WINDOWS + 1 ))
	if [[ "$wcp_credit_window_buffers" -lt 1 ]]; then
		wcp_credit_window_buffers=1
	fi
	if [[ "$GOLEM_GROUP_MANAGER_ENABLE" == "1" ]]; then
		derived_active_workers=$(( GOLEM_TOTAL_GEMM_CORES - GOLEM_TOTAL_GROUPS ))
	else
		derived_active_workers=$GOLEM_TOTAL_GEMM_CORES
	fi
	if [[ "$derived_active_workers" -le 0 ]]; then
		echo "[ERROR] 推导 DMA node chunk credit 需要 active worker > 0，收到 gemm_cores=$GOLEM_TOTAL_GEMM_CORES groups=$GOLEM_TOTAL_GROUPS group_manager=$GOLEM_GROUP_MANAGER_ENABLE" >&2
		exit 1
	fi
	per_worker_ab_credit_chunks=$(( DERIVED_WCP_RESIDENT_K_TILES * (GOLEM_B_REUSE_M_TILES * mat_transfer_credit_chunks + GOLEM_A_REUSE_N_TILES * vec_transfer_credit_chunks) ))
	GOLEM_DMA_NODE_CHUNK_CREDITS=$(( derived_active_workers * wcp_credit_window_buffers * per_worker_ab_credit_chunks ))
fi

if ! [[ "$GOLEM_DMA_NODE_CHUNK_CREDITS" =~ ^[0-9]+$ ]]; then
	echo "[ERROR] GOLEM_DMA_NODE_CHUNK_CREDITS 必须为非负整数（0 表示关闭 node admission credit），收到: $GOLEM_DMA_NODE_CHUNK_CREDITS" >&2
	exit 1
fi

if ! [[ "$GOLEM_SCHED_SUBMIT_BATCH_SIZE" =~ ^[0-9]+$ ]] || [[ "$GOLEM_SCHED_SUBMIT_BATCH_SIZE" -le 0 ]]; then
	echo "[ERROR] GOLEM_SCHED_SUBMIT_BATCH_SIZE 必须为正整数，收到: $GOLEM_SCHED_SUBMIT_BATCH_SIZE" >&2
	exit 1
fi

if ! [[ "$GOLEM_SCHED_DONE_BATCH_SIZE" =~ ^[0-9]+$ ]] || [[ "$GOLEM_SCHED_DONE_BATCH_SIZE" -le 0 ]]; then
	echo "[ERROR] GOLEM_SCHED_DONE_BATCH_SIZE 必须为正整数，收到: $GOLEM_SCHED_DONE_BATCH_SIZE" >&2
	exit 1
fi

if ! [[ "$GOLEM_DMA_STAGGER_CYCLES" =~ ^[0-9]+$ ]]; then
	echo "[ERROR] --dma-stagger-cycles 必须为非负整数，收到: $GOLEM_DMA_STAGGER_CYCLES" >&2
	exit 1
fi

if [[ "$GOLEM_DMA_OVERLAP" != "0" && "$GOLEM_DMA_OVERLAP" != "1" ]]; then
	echo "[ERROR] --dma-overlap 必须为 0 或 1，收到: $GOLEM_DMA_OVERLAP" >&2
	exit 1
fi

for flag_name in GOLEM_SKIP_TENSOR_GEN GOLEM_SKIP_HBM_GEN GOLEM_SKIP_BUILD GOLEM_SKIP_DEFAULT_GUEST_BUILD; do
	flag_value="${!flag_name}"
	if [[ "$flag_value" != "0" && "$flag_value" != "1" ]]; then
		echo "[ERROR] $flag_name 必须为 0 或 1，收到: $flag_value" >&2
		exit 1
	fi
done

if [[ "$GOLEM_CTRL_OVERLAP_AB" != "0" && "$GOLEM_CTRL_OVERLAP_AB" != "1" ]]; then
	echo "[ERROR] GOLEM_CTRL_OVERLAP_AB 必须为 0 或 1，收到: $GOLEM_CTRL_OVERLAP_AB" >&2
	exit 1
fi

if [[ "$GOLEM_ARCH_SCRIPT" == "architecture/ncores_selfcom_dma.py" ]]; then
	if [[ "$GOLEM_CTRL_LINK_ENABLE" == "1" ]]; then
		GOLEM_ARCH_SCRIPT="architecture/ncores_selfcom_dma_ctrl.py"
	else
		GOLEM_ARCH_SCRIPT="architecture/archive/ncores_selfcom_dma.py"
	fi
fi

AUTO_STATS_SUBDIR="overlap${GOLEM_DMA_OVERLAP}/$RUN_ID"
DEFAULT_STATS_ROOT="$ARTIFACT_ROOT/stats"
DEFAULT_MVM_DUMP_ROOT="$ARTIFACT_ROOT/mvm_dumps"
DEFAULT_STDOUT_ROOT="$ARTIFACT_ROOT/stdout"
if [[ "$STATS_DIR_FROM_ENV" -eq 0 ]]; then
	STATS_DIR="$DEFAULT_STATS_ROOT/$AUTO_STATS_SUBDIR"
fi
if [[ "$STATS_FILE_FROM_ENV" -eq 0 ]]; then
	STATS_FILE="$STATS_DIR/stats_selfcom.txt"
fi
if [[ "$CORE_MAP_FILE_FROM_ENV" -eq 0 ]]; then
	CORE_MAP_FILE="$STATS_DIR/core_memory_map.csv"
fi
if [[ "$MVM_VERIFY_SUMMARY_FROM_ENV" -eq 0 ]]; then
	MVM_VERIFY_SUMMARY_FILE="$STATS_DIR/mvm_verify_summary.csv"
fi
HEATMAP_PREFIX="noc_m${GOLEM_GEMM_M}_n${GOLEM_GEMM_N}_k${GOLEM_GEMM_K}_${GOLEM_MATMUL_DTYPE}_ov${GOLEM_DMA_OVERLAP}_${RUN_ID}"
CONTRACT_RESOLVED_FILE="$ARTIFACT_ROOT/contracts/matmul_op_desc_resolved.json"
HBM_METADATA_FILE="$HBM_DIR/hbm_config.env"
EXEC_SUMMARY_FILE="$STATS_DIR/execution_summary.csv"
EXEC_DEBUG_SUMMARY_FILE="$STATS_DIR/execution_debug_summary.csv"
DMA_SUMMARY_FILE="$STATS_DIR/dma_summary.csv"
NOC_SUMMARY_FILE="$STATS_DIR/noc_summary.csv"
MEMORY_SUMMARY_FILE="$STATS_DIR/memory_summary.csv"
HBM_READ_COMMAND_SUMMARY_FILE="$STATS_DIR/hbm_read_command_summary.csv"
HBM_READ_COMMAND_NODE_FILE="$STATS_DIR/hbm_read_command_nodes.csv"
NOC_LATENCY_SUMMARY_FILE="$STATS_DIR/noc_latency_summary.csv"
MEMORY_QUEUE_SUMMARY_FILE="$STATS_DIR/memory_queue_summary.csv"
CAUSAL_SUMMARY_FILE="$STATS_DIR/submit_ready_causal_summary.csv"
CAUSAL_TABLE_FILE="$STATS_DIR/submit_ready_causal_table.csv"
SCHED_PRESSURE_SUMMARY_FILE="$STATS_DIR/sched_pressure_summary.csv"
SCHED_PRESSURE_TABLE_FILE="$STATS_DIR/sched_pressure_table.csv"
CREDIT_OWNER_SUMMARY_FILE="$STATS_DIR/credit_owner_summary.csv"
CREDIT_OWNER_TABLE_FILE="$STATS_DIR/credit_owner_table.csv"
WCP_WINDOW_BREAKDOWN_FILE="$STATS_DIR/wcp_window_breakdown.csv"
WCP_WINDOW_CORE_SUMMARY_FILE="$STATS_DIR/wcp_window_core_summary.csv"
TERMINAL_SUMMARY_FILE="$STATS_DIR/terminal_summary.csv"
GOLEM_FUSION_DUMP_DIR="$STATS_DIR/fusion_c"
if [[ "$DUMP_C_AUTO" -eq 1 ]]; then
	DUMP_C_FILE="$STATS_DIR/c_out.csv"
fi
TIMELINE_FILE="$STATS_DIR/window_timeline.svg"
NOC_HOTSPOT_SUMMARY_FILE="$STATS_DIR/noc_hotspot_summary.csv"
NOC_HOTSPOT_ROUTER_FILE="$STATS_DIR/noc_hotspot_router_table.csv"
NOC_HOTSPOT_PORT_FILE="$STATS_DIR/noc_hotspot_port_table.csv"
GOLEM_PARTITION_AUDIT_FILE="${GOLEM_PARTITION_AUDIT_FILE:-$STATS_DIR/partition_plan.csv}"
if [[ "$MVM_DUMP_DIR_FROM_ENV" -eq 0 ]]; then
	GOLEM_MVM_DUMP_DIR="$DEFAULT_MVM_DUMP_ROOT/$AUTO_STATS_SUBDIR"
fi
if [[ "$STDOUT_DIR_FROM_ENV" -eq 0 ]]; then
	STDOUT_DIR="$DEFAULT_STDOUT_ROOT/$AUTO_STATS_SUBDIR"
fi
DRAMSIM_STATS_DIR="$STATS_DIR/dramsim3"
RUN_CONFIG_FILE="$STATS_DIR/run_config.env"
RUN_COMMAND_FILE="$STATS_DIR/run_command.txt"
PIPELINE_DETAIL_LOG="$STATS_DIR/pipeline_detail.log"

if ! [[ "$GOLEM_MESH_DIM_X" =~ ^[0-9]+$ ]] || [[ "$GOLEM_MESH_DIM_X" -le 0 ]]; then
	echo "[ERROR] --mesh-dim-x 必须为正整数，收到: $GOLEM_MESH_DIM_X" >&2
	exit 1
fi

if [[ "$VERIFY_MVM" -eq 1 ]]; then
	GOLEM_MVM_DUMP_ENABLE=1
fi

	mkdir -p "$ARTIFACT_ROOT" "$LOG_DIR" "$HBM_DIR" "$STDOUT_DIR" "$STATS_DIR" "$GOLEM_MVM_DUMP_DIR" "$DRAMSIM_STATS_DIR" "$GOLEM_SST_WORK_DIR"
	: > "$PIPELINE_DETAIL_LOG"

GOLEM_FUSION_DUMP_ENABLE=0
if [[ "$GOLEM_OUTPUT_MODE" == "fusion" && "$VERIFY_C" -eq 1 ]]; then
	GOLEM_FUSION_DUMP_ENABLE=1
	mkdir -p "$GOLEM_FUSION_DUMP_DIR"
fi

if [[ "$GOLEM_MVM_DUMP_ENABLE" -eq 1 ]]; then
	rm -rf "$GOLEM_MVM_DUMP_DIR"
	mkdir -p "$GOLEM_MVM_DUMP_DIR"
fi

if [[ "$VERIFY_MVM" -eq 1 ]]; then
	rm -f "$MVM_VERIFY_SUMMARY_FILE"
fi

HBM_METADATA_KEYS=(
	GOLEM_GEMM_M
	GOLEM_GEMM_N
	GOLEM_GEMM_K
	GOLEM_GEMM_BLOCK_M
	GOLEM_GEMM_BLOCK_N
	GOLEM_GEMM_BLOCK_K
	GOLEM_MATMUL_DTYPE
	GOLEM_MATMUL_TRANSPOSE_B
	GOLEM_ARRAY_INPUT_SIZE
	GOLEM_ARRAY_OUTPUT_SIZE
	GOLEM_NUM_ARRAYS
	GOLEM_TOTAL_GROUPS
	GOLEM_TOTAL_CORES
	GOLEM_TOTAL_GEMM_CORES
	GOLEM_NUM_MEMORY_NODES
	GOLEM_MEM_NODE_SIZE_BYTES
	GOLEM_HBM_DUMP_OUTPUT
	GOLEM_GLOBAL_STRIDE_BYTES
	GOLEM_A_REUSE_N_TILES
	GOLEM_B_REUSE_M_TILES
	GOLEM_DMA_SLOT_COUNT
)

BUILD_METADATA_KEYS=(
	GOLEM_ARRAY_INPUT_SIZE
	GOLEM_ARRAY_OUTPUT_SIZE
	GOLEM_TOTAL_GROUPS
	GOLEM_TOTAL_CORES
	GOLEM_TOTAL_GEMM_CORES
	GOLEM_NUM_ARRAYS
	GOLEM_NUM_MEMORY_NODES
	GOLEM_MEM_NODE_SIZE_BYTES
	GOLEM_GLOBAL_STRIDE_BYTES
	GOLEM_GEMM_M
	GOLEM_GEMM_N
	GOLEM_GEMM_K
	GOLEM_GEMM_BLOCK_M
	GOLEM_GEMM_BLOCK_N
	GOLEM_GEMM_BLOCK_K
	GOLEM_MATMUL_DTYPE
	GOLEM_DMA_STAGGER_CYCLES
	GOLEM_DMA_OVERLAP
	GOLEM_CTRL_OVERLAP_AB
	GOLEM_GROUP_MANAGER_ENABLE
	GOLEM_CTRL_LINK_ENABLE
	GOLEM_A_REUSE_N_TILES
	GOLEM_B_REUSE_M_TILES
	GOLEM_DMA_SLOT_COUNT
	GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE
	GOLEM_WORKER_START_BARRIER_ENABLE
	GOLEM_WORKER_START_GUARD_CYCLES
	GOLEM_BIAS_ENABLE
	GOLEM_BIAS_VALUE
)

if [[ "$LOG_FILE" = /* ]]; then
	LOG_PATH="$LOG_FILE"
else
	log_base="${LOG_FILE##*/}"
	log_ext=""
	log_stem="$log_base"
	if [[ "$log_base" == *.* ]]; then
		log_ext=".${log_base##*.}"
		log_stem="${log_base%.*}"
	fi
	LOG_PATH="$LOG_DIR/${log_stem}_${RUN_ID}${log_ext}"
fi

export GOLEM_ARTIFACT_ROOT="$ARTIFACT_ROOT"
export GOLEM_HBM_DIR="$HBM_DIR"
export GOLEM_TEST_BINARY="$TEST_BINARY"
export GOLEM_BUILD_METADATA_FILE="$BUILD_METADATA_FILE"
export VANADIS_EXE="$TEST_BINARY"
export GOLEM_SST_WORK_DIR
export GOLEM_STATS_DIR="$STATS_DIR"
export GOLEM_STATS_FILE="$STATS_FILE"
export GOLEM_STDOUT_DIR="$STDOUT_DIR"
export GOLEM_RUN_ID="$RUN_ID"
export GOLEM_DRAMSIM3_OUT_DIR="$DRAMSIM_STATS_DIR"
export GOLEM_CORE_MAP_FILE="$CORE_MAP_FILE"
export GOLEM_PRINT_CORE_MAP="$PRINT_CORE_MAP"
export GOLEM_MVM_VERIFY_SUMMARY_FILE="$MVM_VERIFY_SUMMARY_FILE"
export GOLEM_VERIFY_MVM="$VERIFY_MVM"
export GOLEM_VERIFY_C="$VERIFY_C"
export GOLEM_TIMELINE="$TIMELINE"
export GOLEM_DRAMSIM3_CONFIG
export GOLEM_MEMORY_BACKEND
export GOLEM_RAMULATOR2_CONFIG
export VANADIS_PIPE_TRACE="${VANADIS_PIPE_TRACE:-$LOG_DIR/vanadis_trace.txt}"

export GOLEM_TOTAL_GROUPS
export GOLEM_ARRAY_INPUT_SIZE
export GOLEM_ARRAY_OUTPUT_SIZE
export GOLEM_NUM_ARRAYS
export GOLEM_TOTAL_CORES
export GOLEM_TOTAL_GEMM_CORES
export VANADIS_NUM_CORES="$GOLEM_TOTAL_CORES"
export GOLEM_NUM_MEMORY_NODES
export GOLEM_MEMORY_LAYOUT
export GOLEM_MEM_NODE_SIZE_BYTES
export GOLEM_IDENTITY_BASE
export GOLEM_HBM_DUMP_OUTPUT
export GOLEM_GLOBAL_STRIDE_KB
export GOLEM_GLOBAL_STRIDE_BYTES
export GOLEM_DMA_STAGGER_CYCLES
export GOLEM_DMA_OVERLAP
export GOLEM_CTRL_OVERLAP_AB
export GOLEM_GROUP_MANAGER_ENABLE
export GOLEM_CTRL_LINK_ENABLE
export GOLEM_WORKER_START_BARRIER_ENABLE
export GOLEM_WORKER_START_GUARD_CYCLES
export GOLEM_A_REUSE_N_TILES
export GOLEM_B_REUSE_M_TILES
export GOLEM_DMA_NODE_CREDITS
export GOLEM_DMA_NODE_CHUNK_CREDITS
export GOLEM_DMA_PANEL_CHUNK_BYTES
export GOLEM_DMA_CREDIT_CHUNK_BYTES
export GOLEM_DMA_ADMISSION_LIMIT
export GOLEM_DMA_WINDOW_PRIORITY_ENABLE
export GOLEM_DMA_WINDOW_REORDER_CYCLES
export GOLEM_DMA_RESPONSE_DRAIN_LIMIT
export GOLEM_DMA_RESPONSE_TILE_PRIORITY_ENABLE
export GOLEM_DMA_RESPONSE_VN
export GOLEM_DMA_RESPONSE_REORDER_CYCLES
export GOLEM_DMA_RESPONSE_MAX_STARVATION_CYCLES
export GOLEM_DMA_ADMISSION_MAX_STARVATION_CYCLES
export GOLEM_SCHED_ISSUE_BUDGET_PER_TICK
export GOLEM_SCHED_WORKER_CREDIT_CAP
export GOLEM_DMA_GROUP_RR_ENABLE
export GOLEM_DMA_TILE_CHUNK_QUANTUM
export GOLEM_WCP_CROSS_MACRO_PREFETCH_ENABLE
export GOLEM_DMA_SLOT_COUNT
export GOLEM_DMA_WINDOW_K_TILES
export GOLEM_GEMM_M
export GOLEM_GEMM_N
export GOLEM_GEMM_K
export GOLEM_ORIG_M
export GOLEM_ORIG_N
export GOLEM_ORIG_K
export GOLEM_GEMM_BLOCK_M
export GOLEM_GEMM_BLOCK_N
export GOLEM_GEMM_BLOCK_K
export GOLEM_BIAS_ENABLE
export GOLEM_BIAS_VALUE
export GOLEM_MATMUL_M="$GOLEM_GEMM_M"
export GOLEM_MATMUL_N="$GOLEM_GEMM_N"
export GOLEM_MATMUL_K="$GOLEM_GEMM_K"
export GOLEM_MATMUL_BLOCK_M="$GOLEM_GEMM_BLOCK_M"
export GOLEM_MATMUL_BLOCK_N="$GOLEM_GEMM_BLOCK_N"
export GOLEM_MATMUL_BLOCK_K="$GOLEM_GEMM_BLOCK_K"
export GOLEM_MATMUL_DTYPE
export GOLEM_GEMM_OUT_LAYOUT="colmajor_tile"
export GOLEM_MATMUL_LAYOUT="row_major"
export GOLEM_MATMUL_TRANSPOSE_A="0"
export GOLEM_MATMUL_TRANSPOSE_B="$GOLEM_MATMUL_TRANSPOSE_B"
export GOLEM_DMA_READ_RETRY_TICKS
export GOLEM_DMA_READ_MAX_RETRIES
export GOLEM_DMA_MAX_INFLIGHT
export GOLEM_DMA_BURST_BYTES
export GOLEM_LATENCY_MVM_GM2IMAT
export GOLEM_LATENCY_MVM_GM2IVEC
export GOLEM_LATENCY_MVM_OVEC2GM
export GOLEM_ARRAY_NUM_CU
export GOLEM_ARRAY_MAC_PER_CU_PER_CYCLE
export GOLEM_ARRAY_PIPELINE_DEPTH
export GOLEM_ARRAY_CLOCK
export GOLEM_MEMCTRL_CLOCK
export GOLEM_ARRAY_BUFFER_BASE_LATENCY_CYCLES
export GOLEM_ARRAY_BUFFER_BYTES_PER_CYCLE
export GOLEM_ARRAY_BUFFER_PORTS
export GOLEM_ARRAY_BUFFER_QUEUE_DEPTH
export GOLEM_MATRIX_BROADCAST_MAX_FANOUT
export GOLEM_MATRIX_BROADCAST_BYTES_PER_CYCLE
export GOLEM_MATRIX_BROADCAST_BASE_LATENCY_CYCLES
export GOLEM_MATRIX_BROADCAST_STAGE_LATENCY_CYCLES
export GOLEM_PROGRESS_HEARTBEAT
export GOLEM_PROGRESS_INTERVAL_CYCLES
export GOLEM_SST_ARGS
export GOLEM_SST_THREADS
export GOLEM_CPUSET
export GOLEM_CPU_BINDING_MODE
export GOLEM_MPI_RANKS
export GOLEM_MPI_PARTITIONER
export GOLEM_MPI_LAUNCHER
export GOLEM_MPI_ARGS
export GOLEM_MPI_PARTITIONING=0
export GOLEM_EXPLICIT_PARTITION
export GOLEM_PARTITION_STRATEGY
export GOLEM_PARTITION_WEIGHT_PROFILE
export GOLEM_PARTITION_AUDIT_FILE
export GOLEM_PARTITION_ROUTER_WEIGHT
export GOLEM_PARTITION_CPU_WEIGHT
export GOLEM_PARTITION_MANAGER_WEIGHT
export GOLEM_PARTITION_DATA_MEMORY_WEIGHT
export GOLEM_PARTITION_OS_WEIGHT
export GOLEM_PARTITION_AFFINITY_EDGE_WEIGHT
export GOLEM_SST_STAT_LOAD_LEVEL
export GOLEM_SST_ENABLE_ALL_STATS
export GOLEM_EXPORT_NOC_HEATMAPS
export GOLEM_SKIP_TENSOR_GEN
export GOLEM_SKIP_HBM_GEN
export GOLEM_SKIP_BUILD
export GOLEM_SKIP_DEFAULT_GUEST_BUILD
export GOLEM_BENCH_QUIET_LOGS
export GOLEM_BENCH_DISABLE_SST_STATS
export GOLEM_ROCC_TYPE
export GOLEM_ARRAY_TYPE
export GOLEM_NOC_INPUT_BUF_SIZE
export GOLEM_NOC_OUTPUT_BUF_SIZE
export GOLEM_NOC_MEMNIC_INPUT_BUF_SIZE
export GOLEM_NOC_MEMNIC_OUTPUT_BUF_SIZE
export GOLEM_NOC_LINK_BW
export GOLEM_NOC_XBAR_BW
export GOLEM_NOC_FLIT_SIZE
export GOLEM_GM_LINK_BW
export GOLEM_NOC_VN_PRIORITY_ENABLE
export GOLEM_NOC_VN_PRIORITY_ORDER
export GOLEM_NOC_VN_STARVATION_VN
export GOLEM_NOC_VN_MAX_STARVATION_CYCLES
export GOLEM_NOC_INTER_ROUTER_NO_CUT
export GOLEM_NOC_LOCAL_NO_CUT
export GOLEM_NOC_MEMORY_LOCAL_NO_CUT
export GOLEM_NOC_SCHED_LOCAL_NO_CUT
export GOLEM_MESH_DIM_X
export GOLEM_GM_BUFFER_LENGTH
export GOLEM_GM_C_BUFFER_BYTES
export GOLEM_GM_C_BUFFER_READ_BPC
export GOLEM_GM_C_BUFFER_WRITE_BPC
export GOLEM_GM_C_BUFFER_LATENCY_CYCLES
export GOLEM_FINAL_C_WRITE_ENABLE
export GOLEM_OUTPUT_MODE
export GOLEM_FUSION_DUMP_ENABLE
export GOLEM_FUSION_DUMP_DIR
export GOLEM_DMA_WRITE_VN
export GOLEM_ROCC_VERBOSE
export GOLEM_GM_VERBOSE
export GOLEM_GM_DUMP_DATA
export GOLEM_MVM_DUMP_ENABLE
export GOLEM_MVM_DUMP_DIR
export GOLEM_MVM_DUMP_MODE
export GOLEM_SIM_MODE
export GOLEM_REQUEST_SCHEDULER_EVENT_DRIVEN_WORKER
export GOLEM_VANADIS_ROCC_WAIT_FASTPATH
export GOLEM_VANADIS_ROCC_WAIT_FASTPATH_THRESHOLD
export GOLEM_WCP_PREFETCH_WINDOWS
export GOLEM_SCHED_SUBMIT_BATCH_SIZE
export GOLEM_SCHED_DONE_BATCH_SIZE
export GOLEM_TENSOR_A_FILE="$TENSOR_A_FILE"
export GOLEM_TENSOR_B_FILE="$TENSOR_B_FILE"
export GOLEM_DUMP_C_FILE="$DUMP_C_FILE"
export GOLEM_TENSOR_SOURCE="$TENSOR_SOURCE"
export GOLEM_TENSOR_DIR="$TENSOR_DIR"

{
echo "[RUN] Configuration:"
if [[ -n "$AUTO_PRESET_FILE" ]]; then
	echo "  AUTO_PRESET_FILE=$AUTO_PRESET_FILE"
fi
echo "  GOLEM_TOTAL_GROUPS=$GOLEM_TOTAL_GROUPS"
echo "  GOLEM_ARRAY_INPUT_SIZE=$GOLEM_ARRAY_INPUT_SIZE"
echo "  GOLEM_ARRAY_OUTPUT_SIZE=$GOLEM_ARRAY_OUTPUT_SIZE"
echo "  GOLEM_NUM_ARRAYS=$GOLEM_NUM_ARRAYS"
echo "  GOLEM_TOTAL_CORES=$GOLEM_TOTAL_CORES"
echo "  GOLEM_TOTAL_GEMM_CORES=$GOLEM_TOTAL_GEMM_CORES"
echo "  GOLEM_NUM_MEMORY_NODES=$GOLEM_NUM_MEMORY_NODES"
echo "  GOLEM_MEMORY_LAYOUT=$GOLEM_MEMORY_LAYOUT"
echo "  GOLEM_MEMORY_BACKEND=$GOLEM_MEMORY_BACKEND"
echo "  GOLEM_RAMULATOR2_CONFIG=$GOLEM_RAMULATOR2_CONFIG"
echo "  GOLEM_MEM_NODE_SIZE_BYTES=$GOLEM_MEM_NODE_SIZE_BYTES"
echo "  GOLEM_IDENTITY_BASE=${GOLEM_IDENTITY_BASE:-<unset>}"
echo "  GOLEM_HBM_DUMP_OUTPUT=$GOLEM_HBM_DUMP_OUTPUT"
echo "  GOLEM_GLOBAL_STRIDE_KB=$GOLEM_GLOBAL_STRIDE_KB"
echo "  GOLEM_GLOBAL_STRIDE_BYTES=$GOLEM_GLOBAL_STRIDE_BYTES"
echo "  GOLEM_GM_C_BUFFER_BYTES=$GOLEM_GM_C_BUFFER_BYTES"
echo "  GOLEM_GM_C_BUFFER_READ_BPC=$GOLEM_GM_C_BUFFER_READ_BPC"
echo "  GOLEM_GM_C_BUFFER_WRITE_BPC=$GOLEM_GM_C_BUFFER_WRITE_BPC"
echo "  GOLEM_GM_C_BUFFER_LATENCY_CYCLES=$GOLEM_GM_C_BUFFER_LATENCY_CYCLES"
echo "  GOLEM_FINAL_C_WRITE_ENABLE=$GOLEM_FINAL_C_WRITE_ENABLE"
echo "  GOLEM_DMA_WRITE_VN=$GOLEM_DMA_WRITE_VN"
echo "  GOLEM_DMA_STAGGER_CYCLES=$GOLEM_DMA_STAGGER_CYCLES"
echo "  GOLEM_DMA_OVERLAP=$GOLEM_DMA_OVERLAP"
echo "  GOLEM_CTRL_OVERLAP_AB=$GOLEM_CTRL_OVERLAP_AB"
echo "  GOLEM_GROUP_MANAGER_ENABLE=$GOLEM_GROUP_MANAGER_ENABLE"
echo "  GOLEM_CTRL_LINK_ENABLE=$GOLEM_CTRL_LINK_ENABLE"
echo "  GOLEM_WORKER_START_BARRIER_ENABLE=$GOLEM_WORKER_START_BARRIER_ENABLE"
echo "  GOLEM_WORKER_START_GUARD_CYCLES=$GOLEM_WORKER_START_GUARD_CYCLES"
echo "  GOLEM_A_REUSE_N_TILES=$GOLEM_A_REUSE_N_TILES"
echo "  GOLEM_B_REUSE_M_TILES=$GOLEM_B_REUSE_M_TILES"
echo "  GOLEM_DMA_NODE_CREDITS(legacy transfer credits)=$GOLEM_DMA_NODE_CREDITS"
echo "  GOLEM_DMA_NODE_CHUNK_CREDITS=$GOLEM_DMA_NODE_CHUNK_CREDITS"
echo "  GOLEM_DMA_PANEL_CHUNK_BYTES=$GOLEM_DMA_PANEL_CHUNK_BYTES"
echo "  GOLEM_DMA_CREDIT_CHUNK_BYTES=$GOLEM_DMA_CREDIT_CHUNK_BYTES"
echo "  GOLEM_DMA_ADMISSION_LIMIT=$GOLEM_DMA_ADMISSION_LIMIT"
echo "  GOLEM_DMA_WINDOW_PRIORITY_ENABLE=$GOLEM_DMA_WINDOW_PRIORITY_ENABLE"
echo "  GOLEM_DMA_WINDOW_REORDER_CYCLES=$GOLEM_DMA_WINDOW_REORDER_CYCLES"
echo "  GOLEM_DMA_RESPONSE_DRAIN_LIMIT=$GOLEM_DMA_RESPONSE_DRAIN_LIMIT"
echo "  GOLEM_DMA_RESPONSE_TILE_PRIORITY_ENABLE=$GOLEM_DMA_RESPONSE_TILE_PRIORITY_ENABLE"
echo "  GOLEM_DMA_RESPONSE_REORDER_CYCLES=$GOLEM_DMA_RESPONSE_REORDER_CYCLES"
echo "  GOLEM_DMA_RESPONSE_MAX_STARVATION_CYCLES=$GOLEM_DMA_RESPONSE_MAX_STARVATION_CYCLES"
echo "  GOLEM_DMA_ADMISSION_MAX_STARVATION_CYCLES=$GOLEM_DMA_ADMISSION_MAX_STARVATION_CYCLES"
echo "  GOLEM_SCHED_ISSUE_BUDGET_PER_TICK=$GOLEM_SCHED_ISSUE_BUDGET_PER_TICK"
echo "  GOLEM_SCHED_WORKER_CREDIT_CAP=$GOLEM_SCHED_WORKER_CREDIT_CAP"
echo "  GOLEM_DMA_GROUP_RR_ENABLE=$GOLEM_DMA_GROUP_RR_ENABLE"
echo "  GOLEM_DMA_TILE_CHUNK_QUANTUM=$GOLEM_DMA_TILE_CHUNK_QUANTUM"
echo "  GOLEM_DMA_SLOT_COUNT=$GOLEM_DMA_SLOT_COUNT"
echo "  GOLEM_DMA_WINDOW_K_TILES=$GOLEM_DMA_WINDOW_K_TILES"
echo "  GOLEM_WCP_PREFETCH_WINDOWS=$GOLEM_WCP_PREFETCH_WINDOWS"
echo "  GOLEM_WCP_CROSS_MACRO_PREFETCH_ENABLE=$GOLEM_WCP_CROSS_MACRO_PREFETCH_ENABLE"
echo "  GOLEM_WCP_RESIDENT_K_TILES(derived)=$DERIVED_WCP_RESIDENT_K_TILES"
echo "  GOLEM_SCHED_SUBMIT_BATCH_SIZE=$GOLEM_SCHED_SUBMIT_BATCH_SIZE"
echo "  GOLEM_SCHED_DONE_BATCH_SIZE=$GOLEM_SCHED_DONE_BATCH_SIZE"
echo "  GOLEM_GEMM_K_TILES(derived)=$DERIVED_GEMM_K_TILES"
echo "  GOLEM_GEMM_M_TILES(derived)=$DERIVED_GEMM_M_TILES"
echo "  GOLEM_GEMM_N_TILES(derived)=$DERIVED_GEMM_N_TILES"
echo "  GOLEM_GEMM_TOTAL_TASKS(derived)=$DERIVED_GEMM_TOTAL_TASKS"
echo "  GOLEM_GEMM_TOTAL_MACRO_TASKS(derived)=$DERIVED_GEMM_TOTAL_MACRO_TASKS"
echo "  GOLEM_GEMM_M=$GOLEM_GEMM_M"
echo "  GOLEM_GEMM_N=$GOLEM_GEMM_N"
echo "  GOLEM_GEMM_K=$GOLEM_GEMM_K"
echo "  ORIG_M/N/K=$GOLEM_ORIG_M/$GOLEM_ORIG_N/$GOLEM_ORIG_K"
echo "  PADDED_M/N/K=$GOLEM_GEMM_M/$GOLEM_GEMM_N/$GOLEM_GEMM_K"
echo "  GOLEM_MATMUL_BLOCK_M=$GOLEM_MATMUL_BLOCK_M"
echo "  GOLEM_MATMUL_BLOCK_N=$GOLEM_MATMUL_BLOCK_N"
echo "  GOLEM_MATMUL_BLOCK_K=$GOLEM_MATMUL_BLOCK_K"
echo "  GOLEM_BIAS_ENABLE=$GOLEM_BIAS_ENABLE"
echo "  GOLEM_BIAS_VALUE=$GOLEM_BIAS_VALUE"
echo "  GOLEM_MATMUL_DTYPE=$GOLEM_MATMUL_DTYPE"
echo "  VANADIS_CPU_CLOCK=${VANADIS_CPU_CLOCK:-2.3GHz}"
echo "  GOLEM_ARRAY_CLOCK=$GOLEM_ARRAY_CLOCK"
echo "  GOLEM_MEMCTRL_CLOCK=$GOLEM_MEMCTRL_CLOCK"
echo "  GOLEM_MATMUL_LAYOUT=$GOLEM_MATMUL_LAYOUT"
echo "  GOLEM_DMA_READ_RETRY_TICKS=$GOLEM_DMA_READ_RETRY_TICKS"
echo "  GOLEM_DMA_READ_MAX_RETRIES=$GOLEM_DMA_READ_MAX_RETRIES"
echo "  GOLEM_DMA_MAX_INFLIGHT=$GOLEM_DMA_MAX_INFLIGHT"
echo "  GOLEM_DMA_BURST_BYTES=$GOLEM_DMA_BURST_BYTES"
echo "  GOLEM_LATENCY_MVM_GM2IMAT=$GOLEM_LATENCY_MVM_GM2IMAT"
echo "  GOLEM_LATENCY_MVM_GM2IVEC=$GOLEM_LATENCY_MVM_GM2IVEC"
echo "  GOLEM_LATENCY_MVM_OVEC2GM=$GOLEM_LATENCY_MVM_OVEC2GM"
echo "  GOLEM_ARRAY_NUM_CU=$GOLEM_ARRAY_NUM_CU"
echo "  GOLEM_ARRAY_MAC_PER_CU_PER_CYCLE=$GOLEM_ARRAY_MAC_PER_CU_PER_CYCLE"
echo "  GOLEM_ARRAY_PIPELINE_DEPTH=$GOLEM_ARRAY_PIPELINE_DEPTH"
echo "  GOLEM_ARRAY_BUFFER_BASE_LATENCY_CYCLES=$GOLEM_ARRAY_BUFFER_BASE_LATENCY_CYCLES"
echo "  GOLEM_ARRAY_BUFFER_BYTES_PER_CYCLE=$GOLEM_ARRAY_BUFFER_BYTES_PER_CYCLE"
echo "  GOLEM_ARRAY_BUFFER_PORTS=$GOLEM_ARRAY_BUFFER_PORTS"
echo "  GOLEM_ARRAY_BUFFER_QUEUE_DEPTH=$GOLEM_ARRAY_BUFFER_QUEUE_DEPTH"
echo "  GOLEM_MATRIX_BROADCAST_MAX_FANOUT=$GOLEM_MATRIX_BROADCAST_MAX_FANOUT"
echo "  GOLEM_MATRIX_BROADCAST_BYTES_PER_CYCLE=$GOLEM_MATRIX_BROADCAST_BYTES_PER_CYCLE"
echo "  GOLEM_MATRIX_BROADCAST_BASE_LATENCY_CYCLES=$GOLEM_MATRIX_BROADCAST_BASE_LATENCY_CYCLES"
echo "  GOLEM_MATRIX_BROADCAST_STAGE_LATENCY_CYCLES=$GOLEM_MATRIX_BROADCAST_STAGE_LATENCY_CYCLES"
echo "  GOLEM_PROGRESS_HEARTBEAT=$GOLEM_PROGRESS_HEARTBEAT"
echo "  GOLEM_PROGRESS_INTERVAL_CYCLES=$GOLEM_PROGRESS_INTERVAL_CYCLES"
echo "  GOLEM_TERMINAL_VERBOSE=$GOLEM_TERMINAL_VERBOSE"
echo "  GOLEM_TERMINAL_REFRESH_SECONDS=$GOLEM_TERMINAL_REFRESH_SECONDS"
echo "  GOLEM_SST_ARGS=${GOLEM_SST_ARGS:-<none>}"
echo "  GOLEM_SST_THREADS=$GOLEM_SST_THREADS"
echo "  GOLEM_CPUSET=${GOLEM_CPUSET:-<none>}"
echo "  GOLEM_CPU_BINDING_MODE=$GOLEM_CPU_BINDING_MODE"
echo "  GOLEM_MPI_RANKS=$GOLEM_MPI_RANKS"
echo "  GOLEM_MPI_LAUNCHER=$GOLEM_MPI_LAUNCHER"
echo "  GOLEM_MPI_ARGS=${GOLEM_MPI_ARGS:-<none>}"
echo "  GOLEM_MPI_PARTITIONER=$GOLEM_MPI_PARTITIONER"
echo "  GOLEM_EXPLICIT_PARTITION=$GOLEM_EXPLICIT_PARTITION"
echo "  GOLEM_PARTITION_STRATEGY=$GOLEM_PARTITION_STRATEGY"
echo "  GOLEM_PARTITION_WEIGHT_PROFILE=${GOLEM_PARTITION_WEIGHT_PROFILE:-<none>}"
echo "  GOLEM_PARTITION_AUDIT_FILE=$GOLEM_PARTITION_AUDIT_FILE"
echo "  GOLEM_PARTITION_WEIGHTS=router:$GOLEM_PARTITION_ROUTER_WEIGHT,cpu:$GOLEM_PARTITION_CPU_WEIGHT,manager:$GOLEM_PARTITION_MANAGER_WEIGHT,data_memory:$GOLEM_PARTITION_DATA_MEMORY_WEIGHT,os:$GOLEM_PARTITION_OS_WEIGHT,affinity_edge:$GOLEM_PARTITION_AFFINITY_EDGE_WEIGHT"
echo "  GOLEM_SST_STAT_LOAD_LEVEL=$GOLEM_SST_STAT_LOAD_LEVEL"
echo "  GOLEM_SST_ENABLE_ALL_STATS=$GOLEM_SST_ENABLE_ALL_STATS"
echo "  GOLEM_EXPORT_NOC_HEATMAPS=$GOLEM_EXPORT_NOC_HEATMAPS"
echo "  GOLEM_SKIP_TENSOR_GEN=$GOLEM_SKIP_TENSOR_GEN"
echo "  GOLEM_SKIP_HBM_GEN=$GOLEM_SKIP_HBM_GEN"
echo "  GOLEM_SKIP_BUILD=$GOLEM_SKIP_BUILD"
echo "  GOLEM_SKIP_DEFAULT_GUEST_BUILD=$GOLEM_SKIP_DEFAULT_GUEST_BUILD"
echo "  GOLEM_BENCH_QUIET_LOGS=$GOLEM_BENCH_QUIET_LOGS"
echo "  GOLEM_BENCH_DISABLE_SST_STATS=$GOLEM_BENCH_DISABLE_SST_STATS"
echo "  GOLEM_ROCC_TYPE=$GOLEM_ROCC_TYPE"
echo "  GOLEM_ARRAY_TYPE=$GOLEM_ARRAY_TYPE"
	echo "  GOLEM_NOC_INPUT_BUF_SIZE=$GOLEM_NOC_INPUT_BUF_SIZE"
	echo "  GOLEM_NOC_OUTPUT_BUF_SIZE=$GOLEM_NOC_OUTPUT_BUF_SIZE"
	echo "  GOLEM_NOC_MEMNIC_INPUT_BUF_SIZE=$GOLEM_NOC_MEMNIC_INPUT_BUF_SIZE"
	echo "  GOLEM_NOC_MEMNIC_OUTPUT_BUF_SIZE=$GOLEM_NOC_MEMNIC_OUTPUT_BUF_SIZE"
echo "  GOLEM_NOC_LINK_BW=$GOLEM_NOC_LINK_BW"
echo "  GOLEM_NOC_XBAR_BW=$GOLEM_NOC_XBAR_BW"
echo "  GOLEM_NOC_FLIT_SIZE=$GOLEM_NOC_FLIT_SIZE"
echo "  GOLEM_NOC_VN_PRIORITY_ENABLE=$GOLEM_NOC_VN_PRIORITY_ENABLE"
echo "  GOLEM_NOC_VN_PRIORITY_ORDER=$GOLEM_NOC_VN_PRIORITY_ORDER"
echo "  GOLEM_NOC_VN_STARVATION_VN=$GOLEM_NOC_VN_STARVATION_VN"
echo "  GOLEM_NOC_VN_MAX_STARVATION_CYCLES=$GOLEM_NOC_VN_MAX_STARVATION_CYCLES"
echo "  GOLEM_NOC_INTER_ROUTER_NO_CUT=$GOLEM_NOC_INTER_ROUTER_NO_CUT"
echo "  GOLEM_NOC_LOCAL_NO_CUT=$GOLEM_NOC_LOCAL_NO_CUT"
echo "  GOLEM_NOC_MEMORY_LOCAL_NO_CUT=$GOLEM_NOC_MEMORY_LOCAL_NO_CUT"
echo "  GOLEM_NOC_SCHED_LOCAL_NO_CUT=$GOLEM_NOC_SCHED_LOCAL_NO_CUT"
echo "  GOLEM_MESH_DIM_X=$GOLEM_MESH_DIM_X"
echo "  GOLEM_GM_BUFFER_LENGTH=$GOLEM_GM_BUFFER_LENGTH"
echo "  GOLEM_ROCC_VERBOSE=$GOLEM_ROCC_VERBOSE"
echo "  GOLEM_GM_VERBOSE=$GOLEM_GM_VERBOSE"
echo "  GOLEM_GM_DUMP_DATA=$GOLEM_GM_DUMP_DATA"
echo "  GOLEM_DMA_TRACE=$GOLEM_DMA_TRACE"
echo "  GOLEM_REQUEST_SCHEDULER_TRACE=$GOLEM_REQUEST_SCHEDULER_TRACE"
echo "  GOLEM_LLSC_TRACE=$GOLEM_LLSC_TRACE"
echo "  GOLEM_MVM_DUMP_ENABLE=$GOLEM_MVM_DUMP_ENABLE"
echo "  GOLEM_MVM_DUMP_DIR=$GOLEM_MVM_DUMP_DIR"
echo "  GOLEM_MVM_DUMP_MODE=$GOLEM_MVM_DUMP_MODE"
echo "  GOLEM_SIM_MODE=$GOLEM_SIM_MODE"
echo "  GOLEM_OUTPUT_MODE=$GOLEM_OUTPUT_MODE"
echo "  GOLEM_FUSION_DUMP_ENABLE=$GOLEM_FUSION_DUMP_ENABLE"
echo "  GOLEM_FUSION_DUMP_DIR=$GOLEM_FUSION_DUMP_DIR"
echo "  GOLEM_REQUEST_SCHEDULER_EVENT_DRIVEN_WORKER=${GOLEM_REQUEST_SCHEDULER_EVENT_DRIVEN_WORKER:-auto}"
echo "  GOLEM_VANADIS_ROCC_WAIT_FASTPATH=${GOLEM_VANADIS_ROCC_WAIT_FASTPATH:-auto}"
echo "  GOLEM_VANADIS_ROCC_WAIT_FASTPATH_THRESHOLD=$GOLEM_VANADIS_ROCC_WAIT_FASTPATH_THRESHOLD"
echo "  GOLEM_HBM_DIR=$GOLEM_HBM_DIR"
echo "  GOLEM_STATS_DIR=$GOLEM_STATS_DIR"
echo "  GOLEM_STATS_FILE=$GOLEM_STATS_FILE"
echo "  GOLEM_PRINT_CORE_MAP=$GOLEM_PRINT_CORE_MAP"
echo "  GOLEM_CORE_MAP_FILE=$GOLEM_CORE_MAP_FILE"
echo "  GOLEM_VERIFY_MVM=$GOLEM_VERIFY_MVM"
echo "  GOLEM_VERIFY_C=$GOLEM_VERIFY_C"
echo "  GOLEM_TIMELINE=$GOLEM_TIMELINE"
echo "  GOLEM_TENSOR_SOURCE=$GOLEM_TENSOR_SOURCE"
echo "  GOLEM_TENSOR_DIR=$GOLEM_TENSOR_DIR"
echo "  GOLEM_MVM_VERIFY_SUMMARY_FILE=$GOLEM_MVM_VERIFY_SUMMARY_FILE"
echo "  GOLEM_TENSOR_A_FILE=${GOLEM_TENSOR_A_FILE:-<none>}"
echo "  GOLEM_TENSOR_B_FILE=${GOLEM_TENSOR_B_FILE:-<none>}"
echo "  GOLEM_BIAS_FILE=${GOLEM_BIAS_FILE:-<none>}"
echo "  GOLEM_POOL1_FILE=${GOLEM_POOL1_FILE:-<none>}"
echo "  GOLEM_CONV2_BPACK_FILE=${GOLEM_CONV2_BPACK_FILE:-<none>}"
echo "  GOLEM_CONV2_BIAS_FILE=${GOLEM_CONV2_BIAS_FILE:-<none>}"
echo "  GOLEM_FC1_WEIGHT_FILE=${GOLEM_FC1_WEIGHT_FILE:-<none>}"
echo "  GOLEM_FC1_BIAS_FILE=${GOLEM_FC1_BIAS_FILE:-<none>}"
echo "  GOLEM_FC2_WEIGHT_FILE=${GOLEM_FC2_WEIGHT_FILE:-<none>}"
echo "  GOLEM_FC2_BIAS_FILE=${GOLEM_FC2_BIAS_FILE:-<none>}"
echo "  GOLEM_FC3_WEIGHT_FILE=${GOLEM_FC3_WEIGHT_FILE:-<none>}"
echo "  GOLEM_FC3_BIAS_FILE=${GOLEM_FC3_BIAS_FILE:-<none>}"
echo "  GOLEM_DUMP_C_FILE=${GOLEM_DUMP_C_FILE:-<none>}"
	echo "  GOLEM_STDOUT_DIR=$GOLEM_STDOUT_DIR"
	echo "  GOLEM_RUN_ID=$GOLEM_RUN_ID"
	echo "  GOLEM_SST_WORK_DIR=$GOLEM_SST_WORK_DIR"
	echo "  GOLEM_DRAMSIM3_OUT_DIR=$DRAMSIM_STATS_DIR"
	echo "  LOG_PATH=$LOG_PATH"
} > "$RUN_CONFIG_FILE"

ui_header "TileMC SST SIMULATION"
ui_kv "Run" "$RUN_ID"
ui_kv "Mode" "$GOLEM_SIM_MODE"
ui_kv "Memory" "$GOLEM_MEMORY_BACKEND"
ui_kv "Workload" "${GOLEM_GEMM_M}x${GOLEM_GEMM_N}x${GOLEM_GEMM_K} ${GOLEM_MATMUL_DTYPE} | block ${GOLEM_GEMM_BLOCK_M}x${GOLEM_GEMM_BLOCK_N}x${GOLEM_GEMM_BLOCK_K}"
ui_kv "Tasks" "${DERIVED_GEMM_TOTAL_TASKS} tasks | ${DERIVED_GEMM_TOTAL_MACRO_TASKS} macro tasks"
ui_kv "System" "${GOLEM_TOTAL_CORES} cores | ${GOLEM_NUM_MEMORY_NODES} memory nodes | mesh-x ${GOLEM_MESH_DIM_X}"
ui_kv "Parallel" "${GOLEM_MPI_RANKS} ranks x ${GOLEM_SST_THREADS} threads | ${GOLEM_PARTITION_STRATEGY} / ${GOLEM_MPI_PARTITIONER}"
ui_kv "Heartbeat" "every ${GOLEM_PROGRESS_INTERVAL_CYCLES} cycles"
ui_kv "Archive" "$(display_path "$STATS_DIR")"
for notice in "${CONFIG_NOTICES[@]}"; do
	printf '  %s %s\n' "$(ui_color '1;36' '[config]')" "$notice"
done
printf '  %s\n' "$(ui_color '2;37' "Full configuration: $(display_path "$RUN_CONFIG_FILE")")"

GEN_HBM_CMD=(python3 tools/gen_hbm_init.py)
if [[ "$GOLEM_ARCH_SCRIPT" = /* ]]; then
	GOLEM_ARCH_SCRIPT_PATH="$GOLEM_ARCH_SCRIPT"
else
	GOLEM_ARCH_SCRIPT_PATH="$SCRIPT_DIR/$GOLEM_ARCH_SCRIPT"
fi
SST_BASE_CMD=("$REAL_SST_BIN")
if [[ -n "$GOLEM_SST_ARGS" ]]; then
	read -r -a SST_EXTRA_ARGS <<< "$GOLEM_SST_ARGS"
	SST_BASE_CMD+=("${SST_EXTRA_ARGS[@]}")
fi
if [[ " $GOLEM_SST_ARGS " != *"--num-threads"* ]]; then
	SST_BASE_CMD+=("--num-threads=$GOLEM_SST_THREADS")
fi
if [[ "$GOLEM_MPI_RANKS" -gt 1 ]]; then
	if [[ " $GOLEM_SST_ARGS " != *"--partitioner"* ]]; then
		SST_BASE_CMD+=("--partitioner=$GOLEM_MPI_PARTITIONER")
	fi
	MPI_EXTRA_ARGS=()
	if [[ -n "$GOLEM_MPI_ARGS" ]]; then
		read -r -a MPI_EXTRA_ARGS <<< "$GOLEM_MPI_ARGS"
	fi
	SST_CMD=("$GOLEM_MPI_LAUNCHER" "${MPI_EXTRA_ARGS[@]}" -np "$GOLEM_MPI_RANKS" "${SST_BASE_CMD[@]}" "$GOLEM_ARCH_SCRIPT_PATH")
	GOLEM_MPI_PARTITIONING=1
	export GOLEM_MPI_PARTITIONING
else
	SST_CMD=("${SST_BASE_CMD[@]}" "$GOLEM_ARCH_SCRIPT_PATH")
fi
if [[ -n "$GOLEM_CPUSET" ]]; then
	if ! command -v taskset >/dev/null 2>&1; then
		echo "[ERROR] --cpuset requires taskset" >&2
		exit 1
	fi
	SST_CMD=(taskset --cpu-list "$GOLEM_CPUSET" "${SST_CMD[@]}")
fi
SST_DISPLAY_CMD=("${SST_CMD[@]}")
SST_DISPLAY_CMD[$(( ${#SST_DISPLAY_CMD[@]} - 1 ))]="$GOLEM_ARCH_SCRIPT"
SST_CMD_DISPLAY="$(printf '%q ' "${SST_DISPLAY_CMD[@]}")"
SST_CMD_DISPLAY="${SST_CMD_DISPLAY% }"
printf '%s\n' "$SST_CMD_DISPLAY" > "$RUN_COMMAND_FILE"
SAMPLE_TENSOR_CMD=()
if [[ "$TENSOR_SOURCE" == "sample" ]]; then
	SAMPLE_TENSOR_CMD=(python3 tools/gen_sample_tensors.py --m "$GOLEM_GEMM_M" --n "$GOLEM_GEMM_N" --k "$GOLEM_GEMM_K" --dtype "$GOLEM_MATMUL_DTYPE" --a-out "$TENSOR_A_FILE" --b-out "$TENSOR_B_FILE")
fi
if [[ "$TENSOR_SOURCE" == "file" || "$TENSOR_SOURCE" == "sample" ]]; then
	GEN_HBM_CMD+=(--a-file "$TENSOR_A_FILE")
	GEN_HBM_CMD+=(--b-file "$TENSOR_B_FILE")
fi
if [[ -n "${GOLEM_BIAS_FILE:-}" ]]; then
	GEN_HBM_CMD+=(--bias-file "$GOLEM_BIAS_FILE")
fi
if [[ -n "${GOLEM_POOL1_FILE:-}" ]]; then
	GEN_HBM_CMD+=(--pool1-file "$GOLEM_POOL1_FILE")
fi
if [[ -n "${GOLEM_CONV2_BPACK_FILE:-}" ]]; then
	GEN_HBM_CMD+=(--conv2-bpack-file "$GOLEM_CONV2_BPACK_FILE")
fi
if [[ -n "${GOLEM_CONV2_BIAS_FILE:-}" ]]; then
	GEN_HBM_CMD+=(--conv2-bias-file "$GOLEM_CONV2_BIAS_FILE")
fi
if [[ -n "${GOLEM_FC1_WEIGHT_FILE:-}" ]]; then
	GEN_HBM_CMD+=(--fc1-weight-file "$GOLEM_FC1_WEIGHT_FILE")
fi
if [[ -n "${GOLEM_FC1_BIAS_FILE:-}" ]]; then
	GEN_HBM_CMD+=(--fc1-bias-file "$GOLEM_FC1_BIAS_FILE")
fi
if [[ -n "${GOLEM_FC2_WEIGHT_FILE:-}" ]]; then
	GEN_HBM_CMD+=(--fc2-weight-file "$GOLEM_FC2_WEIGHT_FILE")
fi
if [[ -n "${GOLEM_FC2_BIAS_FILE:-}" ]]; then
	GEN_HBM_CMD+=(--fc2-bias-file "$GOLEM_FC2_BIAS_FILE")
fi
if [[ -n "${GOLEM_FC3_WEIGHT_FILE:-}" ]]; then
	GEN_HBM_CMD+=(--fc3-weight-file "$GOLEM_FC3_WEIGHT_FILE")
fi
if [[ -n "${GOLEM_FC3_BIAS_FILE:-}" ]]; then
	GEN_HBM_CMD+=(--fc3-bias-file "$GOLEM_FC3_BIAS_FILE")
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
	{
	echo "[DRY-RUN] Commands to execute:"
	echo "Pipeline start"
	if [[ "$TENSOR_SOURCE" == "sample" ]]; then
		if [[ "$GOLEM_SKIP_TENSOR_GEN" -eq 1 || "$GOLEM_SKIP_HBM_GEN" -eq 1 ]]; then
			echo "  skip sample tensor generation"
		else
			echo "  ${SAMPLE_TENSOR_CMD[*]}"
		fi
	fi
	if [[ "$GOLEM_SKIP_HBM_GEN" -eq 1 ]]; then
		echo "  skip HBM generation and reuse $HBM_DIR/hbm_init_node*.bin"
	else
		echo "  ${GEN_HBM_CMD[*]}"
	fi
	if [[ "$GOLEM_SKIP_DEFAULT_GUEST_BUILD" -eq 1 ]]; then
		if [[ -z "${VANADIS_EXE:-}" || ! -x "$VANADIS_EXE" ]]; then
			echo "[ERROR] GOLEM_SKIP_DEFAULT_GUEST_BUILD=1 requires an executable VANADIS_EXE" >&2
			exit 1
		fi
		echo "  skip default GEMM guest build; use custom VANADIS_EXE=$VANADIS_EXE"
	elif [[ "$GOLEM_SKIP_BUILD" -eq 1 ]]; then
		echo "  skip build and reuse $TEST_BINARY"
	else
		echo "  (cd small/mvm_noc_int_array && make ARCH=riscv64 -B PROG7_OUT=$TEST_BINARY CFLAGS=\"...\")"
	fi
	echo "  GOLEM_MESH_DIM_X=$GOLEM_MESH_DIM_X GOLEM_MVM_DUMP_ENABLE=$GOLEM_MVM_DUMP_ENABLE GOLEM_MVM_DUMP_DIR=$GOLEM_MVM_DUMP_DIR GOLEM_MVM_DUMP_MODE=$GOLEM_MVM_DUMP_MODE GOLEM_CTRL_LINK_ENABLE=$GOLEM_CTRL_LINK_ENABLE GOLEM_MPI_PARTITIONING=$GOLEM_MPI_PARTITIONING $SST_CMD_DISPLAY > $LOG_PATH 2>&1"
	echo "  mv dramsim3*.{txt,json} $DRAMSIM_STATS_DIR/ (if generated)"
	echo "  mv stdout-*/stderr-* $STDOUT_DIR/"
	if [[ "$VERIFY_MVM" -eq 1 ]]; then
		echo "  python3 stats/verify_mvm_dumps.py --dump-dir $GOLEM_MVM_DUMP_DIR --summary $MVM_VERIFY_SUMMARY_FILE"
	fi
	if [[ -n "$DUMP_C_FILE" ]]; then
		if [[ "$GOLEM_OUTPUT_MODE" == "fusion" ]]; then
			echo "  python3 tools/unpack_c_from_hbm.py --fusion-dir $GOLEM_FUSION_DUMP_DIR --out-file $DUMP_C_FILE"
		else
			echo "  python3 tools/unpack_c_from_hbm.py --out-file $DUMP_C_FILE"
		fi
	fi
	if [[ "$VERIFY_C" -eq 1 ]]; then
		echo "  python3 verify/verify_c_against_golden.py --dtype $GOLEM_MATMUL_DTYPE --a-file $TENSOR_A_FILE --b-file $TENSOR_B_FILE --c-file $DUMP_C_FILE --m $GOLEM_GEMM_M --n $GOLEM_GEMM_N --k $GOLEM_GEMM_K --transpose-b $GOLEM_MATMUL_TRANSPOSE_B --bias-enable $GOLEM_BIAS_ENABLE --bias-value $GOLEM_BIAS_VALUE"
	fi
	echo "  python3 stats/extract_latency_csv.py --log $LOG_PATH --log-dir $STDOUT_DIR --summary $EXEC_SUMMARY_FILE"
	echo "  python3 stats/extract_dma_read_stats_csv.py --log $LOG_PATH --log-dir $STDOUT_DIR --summary $DMA_SUMMARY_FILE"
	echo "  python3 stats/extract_noc_summary_csv.py --input-file $GOLEM_STATS_FILE --link-bw $GOLEM_NOC_LINK_BW --output $NOC_SUMMARY_FILE"
	echo "  python3 stats/extract_noc_hotspot_csv.py --input-file $GOLEM_STATS_FILE --link-bw $GOLEM_NOC_LINK_BW --summary $NOC_HOTSPOT_SUMMARY_FILE --router-table $NOC_HOTSPOT_ROUTER_FILE --port-table $NOC_HOTSPOT_PORT_FILE"
	echo "  python3 stats/extract_noc_latency_summary_csv.py --log $LOG_PATH --log-dir $STDOUT_DIR --output $NOC_LATENCY_SUMMARY_FILE"
	if [[ "$GOLEM_MPI_RANKS" -gt 1 ]]; then
		echo "  python3 stats/extract_memory_summary_csv.py --json $DRAMSIM_STATS_DIR/node*/dramsim3.json --txt $DRAMSIM_STATS_DIR/node*/dramsim3.txt --output $MEMORY_SUMMARY_FILE"
	else
		echo "  python3 stats/extract_memory_summary_csv.py --json $DRAMSIM_STATS_DIR/dramsim3.json --txt $DRAMSIM_STATS_DIR/dramsim3.txt --output $MEMORY_SUMMARY_FILE"
	fi
	echo "  python3 stats/extract_memory_queue_summary_csv.py --log $LOG_PATH --log-dir $STDOUT_DIR --output $MEMORY_QUEUE_SUMMARY_FILE"
	if [[ "$TIMELINE" -eq 1 ]]; then
		echo "  python3 stats/plot_window_timeline.py --input $WCP_WINDOW_BREAKDOWN_FILE --output $TIMELINE_FILE"
	fi
	echo "  python3 stats/extract_submit_ready_causal_csv.py --log $LOG_PATH --log-dir $STDOUT_DIR --noc-latency-summary $NOC_LATENCY_SUMMARY_FILE --memory-queue-summary $MEMORY_QUEUE_SUMMARY_FILE --sched-clock 1GHz --memory-clock $GOLEM_MEMCTRL_CLOCK --summary $CAUSAL_SUMMARY_FILE --table $CAUSAL_TABLE_FILE"
	echo "  append run summary -> $RUN_SUMMARY_CSV"
	if [[ "$PRINT_CORE_MAP" -eq 1 ]]; then
		echo "  core map file: $CORE_MAP_FILE"
	fi
	} >> "$PIPELINE_DETAIL_LOG"
	ui_header "DRY RUN"
	ui_kv "MPI partition" "GOLEM_MPI_PARTITIONING=$GOLEM_MPI_PARTITIONING"
	ui_kv "Configuration" "$(display_path "$RUN_CONFIG_FILE")"
	ui_kv "SST command" "$SST_CMD_DISPLAY"
	ui_kv "Command file" "$(display_path "$RUN_COMMAND_FILE")"
	ui_kv "Full plan" "$(display_path "$PIPELINE_DETAIL_LOG")"
	ui_success "Validation complete; no commands executed"
	exit 0
fi

ui_stage 1 "Preparing HBM images"
if [[ "$GOLEM_SKIP_HBM_GEN" -eq 1 ]]; then
	missing_hbm=0
	for ((node_idx = 1; node_idx < GOLEM_NUM_MEMORY_NODES; node_idx++)); do
		if [[ ! -f "$HBM_DIR/hbm_init_node${node_idx}.bin" ]]; then
			echo "[ERROR] GOLEM_SKIP_HBM_GEN=1 但缺少 $HBM_DIR/hbm_init_node${node_idx}.bin" >&2
			missing_hbm=1
		fi
	done
	if [[ "$missing_hbm" -ne 0 ]]; then
		exit 1
	fi
	if [[ -f "$HBM_METADATA_FILE" ]]; then
		validate_metadata_file "$HBM_METADATA_FILE" "HBM" "${HBM_METADATA_KEYS[@]}"
	else
		validate_hbm_contract_fallback "$CONTRACT_RESOLVED_FILE"
	fi
	echo "Reused HBM images from $HBM_DIR" >> "$PIPELINE_DETAIL_LOG"
elif [[ "$TENSOR_SOURCE" == "sample" && "$GOLEM_SKIP_TENSOR_GEN" -eq 1 ]]; then
	expected_a_bytes=$(( GOLEM_GEMM_M * GOLEM_GEMM_K * GOLEM_ELEM_BYTES ))
	expected_b_bytes=$(( GOLEM_GEMM_K * GOLEM_GEMM_N * GOLEM_ELEM_BYTES ))
	validate_tensor_file_size "$TENSOR_A_FILE" "A" "$expected_a_bytes"
	validate_tensor_file_size "$TENSOR_B_FILE" "B" "$expected_b_bytes"
	echo "Reusing sample tensors: A=$TENSOR_A_FILE B=$TENSOR_B_FILE" >> "$PIPELINE_DETAIL_LOG"
	run_archived "${GEN_HBM_CMD[@]}"
	write_metadata_file "$HBM_METADATA_FILE" "${HBM_METADATA_KEYS[@]}"
elif [[ "$TENSOR_SOURCE" == "sample" ]]; then
	run_archived "${SAMPLE_TENSOR_CMD[@]}"
	run_archived "${GEN_HBM_CMD[@]}"
	write_metadata_file "$HBM_METADATA_FILE" "${HBM_METADATA_KEYS[@]}"
else
	run_archived "${GEN_HBM_CMD[@]}"
	write_metadata_file "$HBM_METADATA_FILE" "${HBM_METADATA_KEYS[@]}"
fi
ui_success "HBM images ready"

ui_stage 2 "Building test binary"
if [[ "$GOLEM_SKIP_DEFAULT_GUEST_BUILD" -eq 1 ]]; then
	if [[ -z "${VANADIS_EXE:-}" || ! -x "$VANADIS_EXE" ]]; then
		echo "[ERROR] GOLEM_SKIP_DEFAULT_GUEST_BUILD=1 requires an executable VANADIS_EXE" >&2
		exit 1
	fi
	echo "[INFO] Skipping default GEMM guest build; using custom VANADIS_EXE=$VANADIS_EXE"
elif [[ "$GOLEM_SKIP_BUILD" -eq 1 ]]; then
	if [[ ! -x "$TEST_BINARY" ]]; then
		echo "[ERROR] GOLEM_SKIP_BUILD=1 但缺少可执行文件 $TEST_BINARY" >&2
		exit 1
	fi
	validate_metadata_file "$BUILD_METADATA_FILE" "test_noc_dma build" "${BUILD_METADATA_KEYS[@]}"
	echo "Reused test binary: $TEST_BINARY" >> "$PIPELINE_DETAIL_LOG"
else
	mkdir -p "$(dirname "$TEST_BINARY")" "$(dirname "$BUILD_METADATA_FILE")"
	pushd small/mvm_noc_int_array >/dev/null
	compile_dtype_flag="-DGOLEM_COMPILE_ELEM_BYTES=${GOLEM_ELEM_BYTES}"
 run_archived make -B ARCH=riscv64 PROG7_OUT="$TEST_BINARY" CFLAGS="-DGOLEM_ARRAY_INPUT_SIZE=${GOLEM_ARRAY_INPUT_SIZE} -DGOLEM_ARRAY_OUTPUT_SIZE=${GOLEM_ARRAY_OUTPUT_SIZE} -DGOLEM_TOTAL_GROUPS=${GOLEM_TOTAL_GROUPS} -DGOLEM_TOTAL_CORES=${GOLEM_TOTAL_CORES} -DGOLEM_TOTAL_GEMM_CORES=${GOLEM_TOTAL_GEMM_CORES} -DGOLEM_NUM_ARRAYS=${GOLEM_NUM_ARRAYS} -DGOLEM_NUM_MEMORY_NODES=${GOLEM_NUM_MEMORY_NODES} -DGOLEM_MEM_NODE_SIZE_BYTES=${GOLEM_MEM_NODE_SIZE_BYTES} -DGOLEM_GLOBAL_STRIDE_BYTES=${GOLEM_GLOBAL_STRIDE_BYTES} -DGOLEM_GEMM_M=${GOLEM_GEMM_M} -DGOLEM_GEMM_N=${GOLEM_GEMM_N} -DGOLEM_GEMM_K=${GOLEM_GEMM_K} -DGOLEM_GEMM_BLOCK_M=${GOLEM_GEMM_BLOCK_M} -DGOLEM_GEMM_BLOCK_N=${GOLEM_GEMM_BLOCK_N} -DGOLEM_GEMM_BLOCK_K=${GOLEM_GEMM_BLOCK_K} -DGOLEM_DMA_STAGGER_CYCLES=${GOLEM_DMA_STAGGER_CYCLES} -DGOLEM_DMA_OVERLAP=${GOLEM_DMA_OVERLAP} -DGOLEM_CTRL_OVERLAP_AB=${GOLEM_CTRL_OVERLAP_AB} -DGOLEM_GROUP_MANAGER_ENABLE=${GOLEM_GROUP_MANAGER_ENABLE} -DGOLEM_CTRL_LINK_ENABLE=${GOLEM_CTRL_LINK_ENABLE} -DGOLEM_A_REUSE_N_TILES=${GOLEM_A_REUSE_N_TILES} -DGOLEM_B_REUSE_M_TILES=${GOLEM_B_REUSE_M_TILES} -DGOLEM_DMA_SLOT_COUNT=${GOLEM_DMA_SLOT_COUNT} -DGOLEM_WORKER_COMMAND_PROCESSOR_ENABLE=${GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE:-0} -DGOLEM_WORKER_START_BARRIER_ENABLE=${GOLEM_WORKER_START_BARRIER_ENABLE} -DGOLEM_WORKER_START_GUARD_CYCLES=${GOLEM_WORKER_START_GUARD_CYCLES} -DGOLEM_BIAS_ENABLE=${GOLEM_BIAS_ENABLE} -DGOLEM_BIAS_VALUE=${GOLEM_BIAS_VALUE} ${compile_dtype_flag}"
	popd >/dev/null
	write_metadata_file "$BUILD_METADATA_FILE" "${BUILD_METADATA_KEYS[@]}"
fi
ui_success "Test binary ready"

ui_stage 3 "Running SST"
rm -f "$GOLEM_SST_WORK_DIR"/dramsim3.txt "$GOLEM_SST_WORK_DIR"/dramsim3.json "$GOLEM_SST_WORK_DIR"/dramsim3epoch.json
(
	cd "$GOLEM_SST_WORK_DIR"
	exec setsid "${SST_CMD[@]}"
) > "$LOG_PATH" 2>&1 &
SST_PID=$!
SST_FATAL=0
SST_COMPLETE=0
progress=0
bar_width=40
phase="启动 SST"
sst_start_epoch="$(date +%s)"
last_plain_progress=-10
last_plain_phase=""
while sst_pid_is_running "$SST_PID"; do
	if sst_log_has_fatal "$LOG_PATH"; then
		SST_FATAL=1
		phase="检测到致命错误"
		break
	fi
	if sst_log_has_complete "$LOG_PATH"; then
		SST_COMPLETE=1
		phase="仿真完成"
		break
	fi
	info="$(estimate_sst_progress_info "$LOG_PATH")"
	new_progress="${info%%|*}"
	new_phase="${info#*|}"
	if [[ "$new_progress" -gt "$progress" ]]; then
		progress="$new_progress"
	fi
	phase="$new_phase"
	elapsed=$(( $(date +%s) - sst_start_epoch ))
	if [[ "$progress" -le 5 && "$elapsed" -ge 15 ]]; then
		phase="事件循环 | 等待首个工作负载心跳"
	fi
	if supports_fancy_output; then
		bar="$(render_inline_progress_bar "$progress" "$bar_width")"
		printf "\r  %s %s %3d%%  %-42s  %s %s\033[K" \
			"$(ui_color '1;36' 'SIM')" "$bar" "$progress" "$phase" \
			"$(ui_color '2;37' 'elapsed')" "$(format_elapsed "$elapsed")"
	elif (( progress >= last_plain_progress + 5 )) || [[ "$phase" != "$last_plain_phase" ]]; then
		printf '  SIM %3d%%  %-42s  elapsed %s\n' "$progress" "$phase" "$(format_elapsed "$elapsed")"
		last_plain_progress="$progress"
		last_plain_phase="$phase"
	fi
	sleep "$GOLEM_TERMINAL_REFRESH_SECONDS"
done

if [[ "$SST_FATAL" -eq 1 ]]; then
	terminate_sst_process_tree "$SST_PID"
	wait_for_sst_exit "$SST_PID" 2 || true
	if supports_fancy_output; then
		printf "\n"
	fi
	print_sst_failure_context "$LOG_PATH"
	exit 1
fi

if [[ "$SST_COMPLETE" -eq 1 ]]; then
	if ! wait_for_sst_exit "$SST_PID" 2; then
		echo "[WARN] SST completed in log but process is still exiting; continue to post-processing."
	fi
	if supports_fancy_output; then
		bar="$(render_inline_progress_bar 100 "$bar_width")"
		printf "\r  %s %s 100%%  %-42s  %s %s\033[K\n" "$(ui_color '1;36' 'SIM')" "$bar" "完成" "$(ui_color '2;37' 'elapsed')" "$(format_elapsed "$(( $(date +%s) - sst_start_epoch ))")"
	fi
else
	if ! wait "$SST_PID"; then
		if supports_fancy_output; then
			printf "\n"
		fi
		print_sst_failure_context "$LOG_PATH"
		exit 1
	fi

	if supports_fancy_output; then
		bar="$(render_inline_progress_bar 100 "$bar_width")"
		printf "\r  %s %s 100%%  %-42s  %s %s\033[K\n" "$(ui_color '1;36' 'SIM')" "$bar" "完成" "$(ui_color '2;37' 'elapsed')" "$(format_elapsed "$(( $(date +%s) - sst_start_epoch ))")"
	fi
fi

shopt -s nullglob
if [[ "$GOLEM_MPI_RANKS" -eq 1 ]]; then
	for f in "$GOLEM_SST_WORK_DIR"/dramsim3.txt "$GOLEM_SST_WORK_DIR"/dramsim3.json "$GOLEM_SST_WORK_DIR"/dramsim3epoch.json; do
		if [[ -f "$f" ]]; then
			mv "$f" "$DRAMSIM_STATS_DIR/"
		fi
	done
fi
shopt -u nullglob
ui_success "SST simulation complete"

shopt -s nullglob
for f in "$GOLEM_SST_WORK_DIR"/stdout-* "$GOLEM_SST_WORK_DIR"/stderr-*; do
	mv "$f" "$STDOUT_DIR/"
done
shopt -u nullglob

if [[ "$GOLEM_MPI_RANKS" -gt 1 && "$GOLEM_BENCH_DISABLE_SST_STATS" -eq 0 ]]; then
	if ! merge_ranked_stats "$GOLEM_STATS_FILE" >> "$PIPELINE_DETAIL_LOG" 2>&1; then
		echo "[ERROR] Failed to merge MPI statistics; see $PIPELINE_DETAIL_LOG" >&2
		exit 1
	fi
fi

ui_stage 4 "Generating reports"
if [[ "$VERIFY_MVM" -eq 1 ]]; then
	echo "Verifying MVM dumps" >> "$PIPELINE_DETAIL_LOG"
	run_archived python3 "$SCRIPT_DIR/stats/verify_mvm_dumps.py" --dump-dir "$GOLEM_MVM_DUMP_DIR" --summary "$MVM_VERIFY_SUMMARY_FILE"
fi

if [[ -n "$DUMP_C_FILE" ]]; then
	if [[ "$GOLEM_OUTPUT_MODE" == "fusion" ]]; then
		echo "Unpacking C tensor from fusion sink" >> "$PIPELINE_DETAIL_LOG"
		run_archived python3 "$SCRIPT_DIR/tools/unpack_c_from_hbm.py" \
			--fusion-dir "$GOLEM_FUSION_DUMP_DIR" --out-file "$DUMP_C_FILE"
	else
		echo "Unpacking C tensor from HBM output" >> "$PIPELINE_DETAIL_LOG"
		run_archived python3 "$SCRIPT_DIR/tools/unpack_c_from_hbm.py" --out-file "$DUMP_C_FILE"
	fi
fi

if [[ "$VERIFY_C" -eq 1 ]]; then
	echo "Verifying C against A@B golden" >> "$PIPELINE_DETAIL_LOG"
	run_archived python3 "$SCRIPT_DIR/verify/verify_c_against_golden.py" \
		--dtype "$GOLEM_MATMUL_DTYPE" \
		--a-file "$TENSOR_A_FILE" \
		--b-file "$TENSOR_B_FILE" \
		--c-file "$DUMP_C_FILE" \
		--m "$GOLEM_GEMM_M" \
		--n "$GOLEM_GEMM_N" \
		--k "$GOLEM_GEMM_K" \
		--transpose-b "$GOLEM_MATMUL_TRANSPOSE_B" \
		--bias-enable "$GOLEM_BIAS_ENABLE" \
		--bias-value "$GOLEM_BIAS_VALUE"
fi

echo "Exporting execution summary" >> "$PIPELINE_DETAIL_LOG"
if [[ -f "$SCRIPT_DIR/stats/extract_latency_csv.py" ]]; then
	if ! run_archived python3 "$SCRIPT_DIR/stats/extract_latency_csv.py" \
		--log "$LOG_PATH" \
		--log-dir "$STDOUT_DIR" \
		--debug-summary "$EXEC_DEBUG_SUMMARY_FILE" \
		--stats-file "$GOLEM_STATS_FILE" \
		--gemm-m "$GOLEM_GEMM_M" \
		--gemm-n "$GOLEM_GEMM_N" \
		--gemm-k "$GOLEM_GEMM_K" \
		--active-worker-cores "$((GOLEM_TOTAL_GEMM_CORES - GOLEM_TOTAL_GROUPS))" \
		--num-arrays "$GOLEM_NUM_ARRAYS" \
		--array-num-cu "$GOLEM_ARRAY_NUM_CU" \
		--mac-per-cu-per-cycle "$GOLEM_ARRAY_MAC_PER_CU_PER_CYCLE" \
		--summary "$EXEC_SUMMARY_FILE"; then
		echo "[WARN] Execution summary extraction failed. Ensure test.log contains LATENCY(cycles) lines."
	fi
else
	echo "[WARN] stats/extract_latency_csv.py not found, skip execution summary extraction."
fi

echo "Exporting DMA summary" >> "$PIPELINE_DETAIL_LOG"
if [[ -f "$SCRIPT_DIR/stats/extract_dma_read_stats_csv.py" ]]; then
	if ! run_archived python3 "$SCRIPT_DIR/stats/extract_dma_read_stats_csv.py" \
		--log "$LOG_PATH" \
		--log-dir "$STDOUT_DIR" \
		--summary "$DMA_SUMMARY_FILE"; then
		echo "[WARN] DMA summary extraction failed. Ensure log contains 'DMA READ stats' lines."
	fi
else
	echo "[WARN] stats/extract_dma_read_stats_csv.py not found, skip DMA summary extraction."
fi

echo "Exporting NoC summary" >> "$PIPELINE_DETAIL_LOG"
if [[ -f "$SCRIPT_DIR/stats/extract_noc_summary_csv.py" ]]; then
	if [[ -f "$GOLEM_STATS_FILE" ]]; then
		if ! run_archived python3 "$SCRIPT_DIR/stats/extract_noc_summary_csv.py" \
			--input-file "$GOLEM_STATS_FILE" \
			--link-bw "$GOLEM_NOC_LINK_BW" \
			--output "$NOC_SUMMARY_FILE"; then
			echo "[WARN] NoC summary extraction failed."
		fi
	else
		echo "[WARN] $GOLEM_STATS_FILE not found, skip NoC summary extraction."
	fi
else
	echo "[WARN] stats/extract_noc_summary_csv.py not found, skip NoC summary extraction."
fi

echo "Exporting NoC hotspot tables" >> "$PIPELINE_DETAIL_LOG"
if [[ -f "$SCRIPT_DIR/stats/extract_noc_hotspot_csv.py" ]]; then
	if [[ -f "$GOLEM_STATS_FILE" ]]; then
		if ! run_archived python3 "$SCRIPT_DIR/stats/extract_noc_hotspot_csv.py" \
			--input-file "$GOLEM_STATS_FILE" \
			--link-bw "$GOLEM_NOC_LINK_BW" \
			--summary "$NOC_HOTSPOT_SUMMARY_FILE" \
			--router-table "$NOC_HOTSPOT_ROUTER_FILE" \
			--port-table "$NOC_HOTSPOT_PORT_FILE"; then
			echo "[WARN] NoC hotspot extraction failed."
		fi
	else
		echo "[WARN] $GOLEM_STATS_FILE not found, skip NoC hotspot extraction."
	fi
else
	echo "[WARN] stats/extract_noc_hotspot_csv.py not found, skip NoC hotspot extraction."
fi

echo "Exporting NoC latency summary" >> "$PIPELINE_DETAIL_LOG"
if [[ -f "$SCRIPT_DIR/stats/extract_noc_latency_summary_csv.py" ]]; then
	if ! run_archived python3 "$SCRIPT_DIR/stats/extract_noc_latency_summary_csv.py" \
		--log "$LOG_PATH" \
		--log-dir "$STDOUT_DIR" \
		--output "$NOC_LATENCY_SUMMARY_FILE"; then
		echo "[WARN] NoC latency summary extraction failed."
	fi
else
	echo "[WARN] stats/extract_noc_latency_summary_csv.py not found, skip NoC latency summary extraction."
fi

if [[ "${GOLEM_EXPORT_NOC_HEATMAPS:-0}" -eq 1 ]]; then
	echo "Exporting NoC heatmaps" >> "$PIPELINE_DETAIL_LOG"
	if [[ -f "$SCRIPT_DIR/stats/visualize_noc_routers.py" ]]; then
		if [[ -f "$GOLEM_STATS_FILE" ]]; then
			if ! run_archived python3 "$SCRIPT_DIR/stats/visualize_noc_routers.py" \
				--input-file "$GOLEM_STATS_FILE" \
				--output-dir "$STATS_DIR" \
				--output-prefix "$HEATMAP_PREFIX"; then
				echo "[WARN] NoC heatmap generation failed."
			fi
		else
			echo "[WARN] $GOLEM_STATS_FILE not found, skip NoC heatmap generation."
		fi
	else
		echo "[WARN] stats/visualize_noc_routers.py not found, skip NoC heatmap generation."
	fi
fi

echo "Exporting memory summary" >> "$PIPELINE_DETAIL_LOG"
if [[ "$GOLEM_MEMORY_BACKEND" == "ramulator2" ]]; then
	RAMULATOR2_LOG_FILES=("$LOG_PATH")
	shopt -s nullglob
	RAMULATOR2_LOG_FILES+=("$STDOUT_DIR"/stdout-* "$STDOUT_DIR"/stderr-*)
	shopt -u nullglob
	RAMULATOR2_SUMMARY_CMD=(
		python3 "$SCRIPT_DIR/stats/extract_ramulator2_summary.py"
		--exclude-node 0
		--memory-summary "$MEMORY_SUMMARY_FILE"
		--hbm-summary "$HBM_READ_COMMAND_SUMMARY_FILE"
		--hbm-nodes "$HBM_READ_COMMAND_NODE_FILE"
	)
	for f in "${RAMULATOR2_LOG_FILES[@]}"; do
		RAMULATOR2_SUMMARY_CMD+=(--log "$f")
	done
	if ! run_archived "${RAMULATOR2_SUMMARY_CMD[@]}"; then
		echo "[WARN] Ramulator2 memory summary extraction failed."
	fi
elif [[ -f "$SCRIPT_DIR/stats/extract_memory_summary_csv.py" ]]; then
	DRAMSIM_JSON_FILES=()
	DRAMSIM_TXT_FILES=()
	shopt -s nullglob
	if [[ "$GOLEM_MPI_RANKS" -gt 1 ]]; then
		DRAMSIM_JSON_FILES=("$DRAMSIM_STATS_DIR"/node*/dramsim3.json)
		DRAMSIM_TXT_FILES=("$DRAMSIM_STATS_DIR"/node*/dramsim3.txt)
	else
		DRAMSIM_JSON_FILES=("$DRAMSIM_STATS_DIR/dramsim3.json")
		DRAMSIM_TXT_FILES=("$DRAMSIM_STATS_DIR/dramsim3.txt")
	fi
	shopt -u nullglob
	if [[ "${#DRAMSIM_JSON_FILES[@]}" -gt 0 && "${#DRAMSIM_JSON_FILES[@]}" -eq "${#DRAMSIM_TXT_FILES[@]}" ]]; then
		MEMORY_SUMMARY_CMD=(python3 "$SCRIPT_DIR/stats/extract_memory_summary_csv.py")
		for f in "${DRAMSIM_JSON_FILES[@]}"; do
			MEMORY_SUMMARY_CMD+=(--json "$f")
		done
		for f in "${DRAMSIM_TXT_FILES[@]}"; do
			MEMORY_SUMMARY_CMD+=(--txt "$f")
		done
		if [[ "$GOLEM_MPI_RANKS" -gt 1 ]]; then
			MEMORY_SUMMARY_CMD+=(--exclude-node 0)
		fi
		MEMORY_SUMMARY_CMD+=(--output "$MEMORY_SUMMARY_FILE")
		if ! run_archived "${MEMORY_SUMMARY_CMD[@]}"; then
			echo "[WARN] Memory summary extraction failed."
		fi
	else
		echo "[WARN] DRAMSim3 stats files not found, skip memory summary extraction."
	fi
else
	echo "[WARN] stats/extract_memory_summary_csv.py not found, skip memory summary extraction."
fi

echo "Exporting HBM READ-command utilization" >> "$PIPELINE_DETAIL_LOG"
if [[ "$GOLEM_MEMORY_BACKEND" == "ramulator2" ]]; then
	: # Produced together with the Ramulator2 memory summary above.
elif [[ -f "$SCRIPT_DIR/stats/extract_hbm_read_command_util.py" && "${#DRAMSIM_JSON_FILES[@]}" -gt 0 ]]; then
	HBM_READ_COMMAND_CMD=(python3 "$SCRIPT_DIR/stats/extract_hbm_read_command_util.py" --config "$GOLEM_DRAMSIM3_CONFIG")
	for f in "${DRAMSIM_JSON_FILES[@]}"; do
		HBM_READ_COMMAND_CMD+=(--json "$f")
	done
	if [[ "$GOLEM_MPI_RANKS" -gt 1 ]]; then
		HBM_READ_COMMAND_CMD+=(--exclude-node 0)
	fi
	HBM_READ_COMMAND_CMD+=(--summary "$HBM_READ_COMMAND_SUMMARY_FILE" --nodes "$HBM_READ_COMMAND_NODE_FILE")
	if ! run_archived "${HBM_READ_COMMAND_CMD[@]}"; then
		echo "[WARN] HBM READ-command utilization extraction failed."
	fi
else
	echo "[WARN] READ-command extractor or DRAMSim3 JSON files not found."
fi

echo "Exporting memory queue summary" >> "$PIPELINE_DETAIL_LOG"
if [[ -f "$SCRIPT_DIR/stats/extract_memory_queue_summary_csv.py" ]]; then
	if ! run_archived python3 "$SCRIPT_DIR/stats/extract_memory_queue_summary_csv.py" \
		--log "$LOG_PATH" \
		--log-dir "$STDOUT_DIR" \
		--output "$MEMORY_QUEUE_SUMMARY_FILE"; then
		echo "[WARN] Memory queue summary extraction failed."
	fi
else
	echo "[WARN] stats/extract_memory_queue_summary_csv.py not found, skip memory queue summary extraction."
fi

echo "Exporting submit-ready causal breakdown" >> "$PIPELINE_DETAIL_LOG"
if [[ -f "$SCRIPT_DIR/stats/extract_submit_ready_causal_csv.py" ]]; then
	if ! run_archived python3 "$SCRIPT_DIR/stats/extract_submit_ready_causal_csv.py" \
		--log "$LOG_PATH" \
		--log-dir "$STDOUT_DIR" \
		--noc-latency-summary "$NOC_LATENCY_SUMMARY_FILE" \
		--memory-queue-summary "$MEMORY_QUEUE_SUMMARY_FILE" \
		--sched-clock "1GHz" \
		--memory-clock "$GOLEM_MEMCTRL_CLOCK" \
		--summary "$CAUSAL_SUMMARY_FILE" \
		--table "$CAUSAL_TABLE_FILE"; then
		echo "[WARN] submit-ready causal breakdown extraction failed."
	fi
else
	echo "[WARN] stats/extract_submit_ready_causal_csv.py not found, skip causal breakdown extraction."
fi

echo "Exporting WCP window breakdown" >> "$PIPELINE_DETAIL_LOG"
if [[ -f "$SCRIPT_DIR/stats/extract_wcp_window_breakdown.py" ]]; then
	if ! run_archived python3 "$SCRIPT_DIR/stats/extract_wcp_window_breakdown.py" \
		--log "$LOG_PATH" \
		--log-dir "$STDOUT_DIR" \
		--summary "$WCP_WINDOW_BREAKDOWN_FILE" \
		--core-summary "$WCP_WINDOW_CORE_SUMMARY_FILE"; then
		echo "[WARN] WCP window breakdown extraction failed."
	fi
fi

if [[ "$TIMELINE" -eq 1 ]]; then
	echo "Generating WCP window timeline" >> "$PIPELINE_DETAIL_LOG"
	if [[ ! -s "$WCP_WINDOW_BREAKDOWN_FILE" ]]; then
		echo "[ERROR] --timeline requested, but WCP window breakdown is missing or empty: $WCP_WINDOW_BREAKDOWN_FILE" >&2
		exit 1
	fi
	if [[ ! -f "$SCRIPT_DIR/stats/plot_window_timeline.py" ]]; then
		echo "[ERROR] --timeline requested, but plotter is missing: $SCRIPT_DIR/stats/plot_window_timeline.py" >&2
		exit 1
	fi
		if ! run_archived python3 "$SCRIPT_DIR/stats/plot_window_timeline.py" \
			--input "$WCP_WINDOW_BREAKDOWN_FILE" \
			--output "$TIMELINE_FILE" \
			--total-groups "$GOLEM_TOTAL_GROUPS" \
			--title "${GOLEM_GEMM_M}x${GOLEM_GEMM_N}x${GOLEM_GEMM_K} GEMM window timeline" \
		--panel-chunk "${GOLEM_DMA_PANEL_CHUNK_BYTES} B"; then
		echo "[ERROR] Failed to generate timeline SVG." >&2
		exit 1
	fi
fi

echo "Exporting scheduler pressure summary" >> "$PIPELINE_DETAIL_LOG"
if [[ -f "$SCRIPT_DIR/stats/extract_sched_pressure_csv.py" ]]; then
    if ! run_archived python3 "$SCRIPT_DIR/stats/extract_sched_pressure_csv.py" \
        --log "$LOG_PATH" \
        --log-dir "$STDOUT_DIR" \
        --summary "$SCHED_PRESSURE_SUMMARY_FILE" \
        --table "$SCHED_PRESSURE_TABLE_FILE"; then
        echo "[WARN] scheduler pressure extraction failed."
    fi
else
    echo "[WARN] stats/extract_sched_pressure_csv.py not found, skip scheduler pressure extraction."
fi

echo "Exporting memory-node credit owner summary" >> "$PIPELINE_DETAIL_LOG"
if [[ -f "$SCRIPT_DIR/stats/extract_credit_owner_csv.py" ]]; then
    if ! run_archived python3 "$SCRIPT_DIR/stats/extract_credit_owner_csv.py" \
        --log "$LOG_PATH" \
        --log-dir "$STDOUT_DIR" \
        --summary "$CREDIT_OWNER_SUMMARY_FILE" \
        --table "$CREDIT_OWNER_TABLE_FILE" \
        --allow-empty; then
        echo "[ERROR] memory-node credit owner extraction or conservation check failed." >&2
        exit 1
    fi
else
    echo "[ERROR] stats/extract_credit_owner_csv.py not found." >&2
    exit 1
fi

echo "Appending run summary CSV" >> "$PIPELINE_DETAIL_LOG"
run_archived env RUN_START_EPOCH="$RUN_START_EPOCH" RUN_SUMMARY_CSV="$RUN_SUMMARY_CSV" LOG_PATH="$LOG_PATH" STATS_DIR="$STATS_DIR" GOLEM_MPI_RANKS="$GOLEM_MPI_RANKS" GOLEM_MPI_PARTITIONER="$GOLEM_MPI_PARTITIONER" python3 - <<'PY'
import csv
import datetime as dt
import math
import os
import re
from pathlib import Path

run_start = int(os.environ.get("RUN_START_EPOCH", "0") or "0")
run_end = int(dt.datetime.now().timestamp())
wall_time_sec = max(0, run_end - run_start)

log_path = Path(os.environ["LOG_PATH"])
stats_dir = Path(os.environ["STATS_DIR"])
execution_summary = stats_dir / "execution_summary.csv"
dma_summary = stats_dir / "dma_summary.csv"
noc_summary = stats_dir / "noc_summary.csv"
memory_summary = stats_dir / "memory_summary.csv"
hbm_read_command_summary = stats_dir / "hbm_read_command_summary.csv"
noc_latency_summary = stats_dir / "noc_latency_summary.csv"
memory_queue_summary = stats_dir / "memory_queue_summary.csv"
causal_summary = stats_dir / "submit_ready_causal_summary.csv"
noc_hotspot_summary = stats_dir / "noc_hotspot_summary.csv"
wcp_window_core_summary = stats_dir / "wcp_window_core_summary.csv"
out_csv = Path(os.environ["RUN_SUMMARY_CSV"])


def read_metric_value_csv(path: Path):
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return {}
    header = rows[0]
    out = {}
    if header == ["metric", "value"]:
        for row in rows[1:]:
            if len(row) >= 2:
                out[row[0]] = row[1]
        return out
    if header[:2] == ["metric", "mean"]:
        for row in rows[1:]:
            if len(row) >= 2 and row[0].endswith("_share_pct"):
                out[row[0]] = row[1]
            if len(row) >= 2:
                out[f"{row[0]}_mean"] = row[1]
            if len(row) >= 4:
                out[f"{row[0]}_p95"] = row[3]
            if len(row) >= 5:
                out[f"{row[0]}_min"] = row[4]
            if len(row) >= 6:
                out[f"{row[0]}_max"] = row[5]
            if len(row) >= 7:
                out[f"{row[0]}_sum"] = row[6]
        return out
    return {}


def _to_int(value, default=0):
    try:
        return int(str(value), 0)
    except (TypeError, ValueError):
        return default


def _to_float(value, default=0.0):
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _read_ini_number(path: Path, key: str, default=0.0):
    if not path.exists():
        return default
    pat = re.compile(rf"^\s*{re.escape(key)}\s*=\s*([^;#\s]+)")
    for line in path.read_text(errors="ignore").splitlines():
        m = pat.match(line)
        if not m:
            continue
        return _to_float(m.group(1), default)
    return default


def _hbm_read_cmd_bytes(default=64.0):
    if os.environ.get("GOLEM_MEMORY_BACKEND", "ramulator2") == "ramulator2":
        return 32.0
    if "GOLEM_HBM_READ_BYTES_PER_CMD" in os.environ:
        return _to_float(os.environ.get("GOLEM_HBM_READ_BYTES_PER_CMD"), default)
    config = Path(os.environ.get("GOLEM_DRAMSIM3_CONFIG", ""))
    bl = _read_ini_number(config, "BL", 4.0)
    bus_width = _read_ini_number(config, "bus_width", 128.0)
    if bl > 0 and bus_width > 0:
        return bl * bus_width / 8.0
    return default


def _hbm_tccd_l_cycles(default=3.0):
    if os.environ.get("GOLEM_MEMORY_BACKEND", "ramulator2") == "ramulator2":
        # Two independently timed pseudo-channels can sustain one 32 B transfer
        # per physical-channel CK when requests alternate between them.
        return 1.0
    if "GOLEM_HBM_TCCD_L_CYCLES" in os.environ:
        return _to_float(os.environ.get("GOLEM_HBM_TCCD_L_CYCLES"), default)
    config = Path(os.environ.get("GOLEM_DRAMSIM3_CONFIG", ""))
    return _read_ini_number(config, "tCCD_L", default)


def _hbm_channels_per_node(default=16):
    if os.environ.get("GOLEM_MEMORY_BACKEND", "ramulator2") == "ramulator2":
        return 8
    config = Path(os.environ.get("GOLEM_DRAMSIM3_CONFIG", ""))
    return max(1, int(_read_ini_number(config, "channels", default)))


def compute_hbm_readonly_tccdl_util_pct(execution, memory, backend_window_cycles, backend_active_cycles):
    elem_bytes = 4
    m = _to_int(os.environ.get("GOLEM_GEMM_M", ""))
    n = _to_int(os.environ.get("GOLEM_GEMM_N", ""))
    k = _to_int(os.environ.get("GOLEM_GEMM_K", ""))
    block_m = _to_int(os.environ.get("GOLEM_GEMM_BLOCK_M", ""))
    block_n = _to_int(os.environ.get("GOLEM_GEMM_BLOCK_N", ""))
    block_k = _to_int(os.environ.get("GOLEM_GEMM_BLOCK_K", ""))
    a_reuse_n = max(1, _to_int(os.environ.get("GOLEM_A_REUSE_N_TILES", "1"), 1))
    b_reuse_m = max(1, _to_int(os.environ.get("GOLEM_B_REUSE_M_TILES", "1"), 1))
    if min(m, n, k, block_m, block_n, block_k) <= 0:
        return "", "", "", "", "", "", ""

    m_tiles = math.ceil(m / block_m)
    n_tiles = math.ceil(n / block_n)
    k_tiles = math.ceil(k / block_k)
    m_groups = math.ceil(m_tiles / b_reuse_m)
    n_groups = math.ceil(n_tiles / a_reuse_n)
    useful_read_bytes = (
        m_groups
        * n_groups
        * k_tiles
        * elem_bytes
        * (
            b_reuse_m * block_m * block_k
            + a_reuse_n * block_n * block_k
        )
    )

    data_nodes = max(1, _to_int(os.environ.get("GOLEM_NUM_MEMORY_NODES", "1"), 1) - 1)
    channels_per_node = _hbm_channels_per_node(16)
    bytes_per_read_cmd = _hbm_read_cmd_bytes(64.0)
    tccd_l_cycles = _hbm_tccd_l_cycles(3.0)
    if bytes_per_read_cmd <= 0 or tccd_l_cycles <= 0:
        return "", "", "", "", "", str(useful_read_bytes), "", ""

    roofline_bytes_per_cycle = data_nodes * channels_per_node * bytes_per_read_cmd / tccd_l_cycles
    gemm_system_cycles = _to_float(execution.get("gemm_system_latency_cycles", ""), 0.0)
    system_pressure_pct = ""
    if gemm_system_cycles > 0:
        system_pressure_pct = f"{100.0 * useful_read_bytes / (gemm_system_cycles * roofline_bytes_per_cycle):.6f}"

    worker_total_cycles = _to_float(execution.get("total_cycles", ""), 0.0)
    worker_pressure_pct = ""
    if worker_total_cycles > 0:
        worker_pressure_pct = f"{100.0 * useful_read_bytes / (worker_total_cycles * roofline_bytes_per_cycle):.6f}"

    backend_window_util_pct = ""
    if backend_window_cycles > 0:
        backend_window_util_pct = f"{100.0 * useful_read_bytes / (backend_window_cycles * roofline_bytes_per_cycle):.6f}"

    backend_util_pct = ""
    if backend_active_cycles > 0:
        backend_util_pct = f"{100.0 * useful_read_bytes / (backend_active_cycles * roofline_bytes_per_cycle):.6f}"

    # DRAMSim3 reports aggregate average bandwidth in GB/s.  This is the
    # measured physical HBM throughput and must be separated from the
    # useful-byte roofline pressure computed above.
    measured_bw_gbps = _to_float(memory.get("hbm_aggregate_bandwidth_gbps", ""), 0.0)
    peak_bw_gbps = roofline_bytes_per_cycle
    measured_util_pct = ""
    if measured_bw_gbps > 0 and peak_bw_gbps > 0:
        measured_util_pct = f"{100.0 * measured_bw_gbps / peak_bw_gbps:.6f}"

    return (
        measured_util_pct,
        system_pressure_pct,
        backend_window_util_pct,
        backend_util_pct,
        worker_pressure_pct,
        str(useful_read_bytes),
        f"{roofline_bytes_per_cycle:.6f}",
        f"{tccd_l_cycles:.6f}",
    )

simulated_time = ""
backend_read_window_cycles = 0
backend_read_active_cycles = 0
backend_read_service_records = []
if log_path.exists():
    sim_pat = re.compile(r"Simulation is complete, simulated time:\s*(.+)$")
    backend_window_pat = re.compile(
        r"DRAMSIM3_BACKEND_READ_SERVICE_WINDOW_GLOBAL\b.*\bcount=(\d+)"
        r".*\bfirst_arrival_cycle=(\d+).*\blast_complete_cycle=(\d+)"
        r".*\bwindow_cycles=(\d+)"
    )
    backend_active_pat = re.compile(r"DRAMSIM3_BACKEND_READ_ACTIVE_WINDOW_GLOBAL\b.*\bactive_cycles=(\d+)")
    for line in log_path.read_text(errors="ignore").splitlines():
        m = sim_pat.search(line)
        if m:
            simulated_time = m.group(1).strip()
        m = backend_window_pat.search(line)
        if m:
            backend_read_service_records.append(tuple(_to_int(value, 0) for value in m.groups()))
        m = backend_active_pat.search(line)
        if m:
            backend_read_active_cycles = _to_int(m.group(1), 0)

execution = read_metric_value_csv(execution_summary)
dma = read_metric_value_csv(dma_summary)
noc = read_metric_value_csv(noc_summary)
memory = read_metric_value_csv(memory_summary)
hbm_read_command = read_metric_value_csv(hbm_read_command_summary)
noc_latency = read_metric_value_csv(noc_latency_summary)
memory_queue = read_metric_value_csv(memory_queue_summary)
causal = read_metric_value_csv(causal_summary)
hotspot = read_metric_value_csv(noc_hotspot_summary)

if not backend_read_service_records:
    backend_read_window_cycles = _to_int(memory.get("backend_read_window_cycles", ""), 0)


def read_core_bubble_summary(path: Path):
    totals = {
        "core_count": 0,
        "compute_segments": 0,
        "compute_active_cycles": 0,
        "intra_window_transitions": 0,
        "intra_window_bubbles": 0,
        "intra_window_cycles": 0,
        "inter_window_transitions": 0,
        "inter_window_bubbles": 0,
        "inter_window_cycles": 0,
        "inter_macro_transitions": 0,
        "inter_macro_bubbles": 0,
        "inter_macro_cycles": 0,
        "legacy_intra_macro_transitions": 0,
        "legacy_intra_macro_bubbles": 0,
        "legacy_intra_macro_cycles": 0,
        "legacy_inter_macro_transitions": 0,
        "legacy_inter_macro_bubbles": 0,
        "legacy_inter_macro_cycles": 0,
        "descriptor_cold_start_cycles": [],
    }
    if not path.exists():
        return totals
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    totals["core_count"] = len(rows)
    for row in rows:
        cold_start = _to_int(row.get("startup_overhead_cycles", ""))
        if cold_start > 0:
            totals["descriptor_cold_start_cycles"].append(cold_start)
        totals["compute_segments"] += _to_int(row.get("compute_segment_count", ""))
        totals["compute_active_cycles"] += _to_int(row.get("compute_active_cycles", ""))
        totals["intra_window_transitions"] += _to_int(row.get("intra_window_transition_count", ""))
        totals["intra_window_bubbles"] += _to_int(row.get("intra_window_bubble_count", ""))
        totals["intra_window_cycles"] += _to_int(row.get("intra_window_bubble_cycles", ""))
        totals["inter_window_transitions"] += _to_int(
            row.get("inter_window_transition_count", "") or row.get("intra_macro_transition_count", "")
        )
        totals["inter_window_bubbles"] += _to_int(
            row.get("inter_window_bubble_count", "") or row.get("intra_macro_bubble_count", "")
        )
        totals["inter_window_cycles"] += _to_int(
            row.get("inter_window_bubble_cycles", "") or row.get("intra_macro_bubble_cycles", "")
        )
        totals["inter_macro_transitions"] += _to_int(row.get("inter_macro_transition_count", ""))
        totals["inter_macro_bubbles"] += _to_int(row.get("inter_macro_bubble_count", ""))
        totals["inter_macro_cycles"] += _to_int(row.get("inter_macro_bubble_cycles", ""))
        totals["legacy_intra_macro_transitions"] += _to_int(row.get("intra_macro_transition_count", ""))
        totals["legacy_intra_macro_bubbles"] += _to_int(row.get("intra_macro_bubble_count", ""))
        totals["legacy_intra_macro_cycles"] += _to_int(row.get("intra_macro_bubble_cycles", ""))
        totals["legacy_inter_macro_transitions"] += _to_int(
            row.get("legacy_inter_macro_transition_count", "") or row.get("inter_macro_transition_count", "")
        )
        totals["legacy_inter_macro_bubbles"] += _to_int(
            row.get("legacy_inter_macro_bubble_count", "") or row.get("inter_macro_bubble_count", "")
        )
        totals["legacy_inter_macro_cycles"] += _to_int(
            row.get("legacy_inter_macro_bubble_cycles", "") or row.get("inter_macro_bubble_cycles", "")
        )
    return totals


bubbles = read_core_bubble_summary(wcp_window_core_summary)
core_count = bubbles["core_count"]
cold_start_cycles = bubbles["descriptor_cold_start_cycles"]
descriptor_cold_start_avg = (
    f"{sum(cold_start_cycles) / len(cold_start_cycles):.6f}" if cold_start_cycles else ""
)
descriptor_cold_start_min = str(min(cold_start_cycles)) if cold_start_cycles else ""
descriptor_cold_start_max = str(max(cold_start_cycles)) if cold_start_cycles else ""
descriptor_cold_start_array = (
    "[" + ",".join(str(value) for value in cold_start_cycles) + "]"
    if cold_start_cycles else ""
)
intra_window_avg_per_core = (
    f"{bubbles['intra_window_cycles'] / core_count:.6f}" if core_count else ""
)
inter_window_avg_per_core = (
    f"{bubbles['inter_window_cycles'] / core_count:.6f}" if core_count else ""
)
inter_macro_avg_per_core = (
    f"{bubbles['inter_macro_cycles'] / core_count:.6f}" if core_count else ""
)
pipeline_transitions = bubbles["legacy_intra_macro_transitions"] + bubbles["legacy_inter_macro_transitions"]
pipeline_bubble_count = bubbles["legacy_intra_macro_bubbles"] + bubbles["legacy_inter_macro_bubbles"]
pipeline_covered = pipeline_transitions - pipeline_bubble_count
pipeline_bubble_cycles = (
    bubbles["intra_window_cycles"]
    + bubbles["inter_window_cycles"]
    + bubbles["inter_macro_cycles"]
)
pipeline_accounted_cycles = bubbles["compute_active_cycles"] + pipeline_bubble_cycles
pipeline_efficiency_pct = (
    f"{100.0 * bubbles['compute_active_cycles'] / pipeline_accounted_cycles:.6f}"
    if pipeline_accounted_cycles else ""
)
pipeline_avg_bubble_cycles_per_core = (
    f"{pipeline_bubble_cycles / core_count:.6f}" if core_count else ""
)
gemm_system_start_cycle = int(_to_float(execution.get("gemm_system_start_cycle", ""), 0.0))
data_service_records = [
    record for record in backend_read_service_records
    if gemm_system_start_cycle <= 0 or record[1] >= gemm_system_start_cycle
]
if data_service_records:
    backend_read_window_cycles = max(record[2] for record in data_service_records) - min(
        record[1] for record in data_service_records
    )
hbm_util_pct, hbm_pressure_pct, hbm_backend_window_util_pct, hbm_backend_active_util_pct, hbm_worker_pressure_pct, hbm_useful_read_bytes, hbm_roofline_bpc, hbm_tccd_l_cycles = compute_hbm_readonly_tccdl_util_pct(
    execution, memory, backend_read_window_cycles, backend_read_active_cycles
)
if os.environ.get("GOLEM_MEMORY_BACKEND", "ramulator2") == "ramulator2":
    # Ramulator2's adapter records each node's first request arrival through
    # last completion.  Use the capacity-weighted aggregate of those service
    # windows as the primary HBM utilization rather than a global loose span.
    hbm_backend_window_util_pct = memory.get("hbm_read_utilization_pct", "")

record = {
    "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
    "run_id": os.environ.get("GOLEM_RUN_ID", ""),
    "log_file": str(log_path),
    "mpi_ranks": os.environ.get("GOLEM_MPI_RANKS", "1"),
    "sst_threads": os.environ.get("GOLEM_SST_THREADS", "1"),
    "cpu_binding_mode": os.environ.get("GOLEM_CPU_BINDING_MODE", "default"),
    "host_cpuset": os.environ.get("GOLEM_CPUSET", ""),
    "mpi_partitioner": os.environ.get("GOLEM_MPI_PARTITIONER", ""),
    "partition_strategy": os.environ.get("GOLEM_PARTITION_STRATEGY", ""),
    "partition_weight_profile": os.environ.get("GOLEM_PARTITION_WEIGHT_PROFILE", ""),
    "memory_backend": os.environ.get("GOLEM_MEMORY_BACKEND", "ramulator2"),
    "output_mode": os.environ.get("GOLEM_OUTPUT_MODE", "hbm"),
    "overlap": f"overlap{os.environ.get('GOLEM_DMA_OVERLAP', '0')}",
    "array_input_size": os.environ.get("GOLEM_ARRAY_INPUT_SIZE", ""),
    "array_output_size": os.environ.get("GOLEM_ARRAY_OUTPUT_SIZE", ""),
    "gemm_m": os.environ.get("GOLEM_GEMM_M", ""),
    "gemm_n": os.environ.get("GOLEM_GEMM_N", ""),
    "gemm_k": os.environ.get("GOLEM_GEMM_K", ""),
    "block_m": os.environ.get("GOLEM_GEMM_BLOCK_M", ""),
    "block_n": os.environ.get("GOLEM_GEMM_BLOCK_N", ""),
    "block_k": os.environ.get("GOLEM_GEMM_BLOCK_K", ""),
    "bias_enable": os.environ.get("GOLEM_BIAS_ENABLE", ""),
    "bias_value": os.environ.get("GOLEM_BIAS_VALUE", ""),
    "num_cores": os.environ.get("GOLEM_TOTAL_CORES", ""),
    "gemm_cores": os.environ.get("GOLEM_TOTAL_GEMM_CORES", ""),
    "num_mem_nodes": os.environ.get("GOLEM_NUM_MEMORY_NODES", ""),
    "mem_node_size_bytes": os.environ.get("GOLEM_MEM_NODE_SIZE_BYTES", ""),
    "hbm_dump_output": os.environ.get("GOLEM_HBM_DUMP_OUTPUT", ""),
    "dma_node_credits": os.environ.get("GOLEM_DMA_NODE_CREDITS", ""),
    "dma_node_chunk_credits": os.environ.get("GOLEM_DMA_NODE_CHUNK_CREDITS", ""),
    "dma_panel_chunk_bytes": os.environ.get("GOLEM_DMA_PANEL_CHUNK_BYTES", ""),
    "dma_credit_chunk_bytes": os.environ.get("GOLEM_DMA_CREDIT_CHUNK_BYTES", ""),
    "dma_admission_limit": os.environ.get("GOLEM_DMA_ADMISSION_LIMIT", ""),
    "dma_window_priority_enable": os.environ.get("GOLEM_DMA_WINDOW_PRIORITY_ENABLE", ""),
    "dma_window_reorder_cycles": os.environ.get("GOLEM_DMA_WINDOW_REORDER_CYCLES", ""),
    "dma_tile_chunk_quantum": os.environ.get("GOLEM_DMA_TILE_CHUNK_QUANTUM", ""),
    "dma_response_tile_priority_enable": os.environ.get("GOLEM_DMA_RESPONSE_TILE_PRIORITY_ENABLE", ""),
    "dma_response_reorder_cycles": os.environ.get("GOLEM_DMA_RESPONSE_REORDER_CYCLES", ""),
    "dma_response_max_starvation_cycles": os.environ.get("GOLEM_DMA_RESPONSE_MAX_STARVATION_CYCLES", ""),
    "dma_admission_max_starvation_cycles": os.environ.get("GOLEM_DMA_ADMISSION_MAX_STARVATION_CYCLES", ""),
    "wcp_prefetch_windows": os.environ.get("GOLEM_WCP_PREFETCH_WINDOWS", ""),
    "wcp_cross_macro_prefetch_enable": os.environ.get("GOLEM_WCP_CROSS_MACRO_PREFETCH_ENABLE", ""),
    "submit_batch_size": os.environ.get("GOLEM_SCHED_SUBMIT_BATCH_SIZE", ""),
    "done_batch_size": os.environ.get("GOLEM_SCHED_DONE_BATCH_SIZE", ""),
    "sched_worker_credit_cap": os.environ.get("GOLEM_SCHED_WORKER_CREDIT_CAP", ""),
    "dma_retry_ticks": os.environ.get("GOLEM_DMA_READ_RETRY_TICKS", ""),
    "dma_burst_bytes": os.environ.get("GOLEM_DMA_BURST_BYTES", ""),
    "dma_response_drain_limit": os.environ.get("GOLEM_DMA_RESPONSE_DRAIN_LIMIT", ""),
    "dma_stagger_cycles": os.environ.get("GOLEM_DMA_STAGGER_CYCLES", ""),
    "ctrl_overlap_ab": os.environ.get("GOLEM_CTRL_OVERLAP_AB", ""),
    "noc_link_bw": os.environ.get("GOLEM_NOC_LINK_BW", ""),
    "noc_xbar_bw": os.environ.get("GOLEM_NOC_XBAR_BW", ""),
    "noc_flit_size": os.environ.get("GOLEM_NOC_FLIT_SIZE", ""),
    "noc_vn_priority_enable": os.environ.get("GOLEM_NOC_VN_PRIORITY_ENABLE", ""),
    "noc_vn_priority_order": os.environ.get("GOLEM_NOC_VN_PRIORITY_ORDER", ""),
    "noc_vn_starvation_vn": os.environ.get("GOLEM_NOC_VN_STARVATION_VN", ""),
    "noc_vn_max_starvation_cycles": os.environ.get("GOLEM_NOC_VN_MAX_STARVATION_CYCLES", ""),
    "gm_link_bw": os.environ.get("GOLEM_GM_LINK_BW", ""),
    "gm_c_buffer_bytes": os.environ.get("GOLEM_GM_C_BUFFER_BYTES", ""),
    "gm_c_buffer_read_bytes_per_cycle": os.environ.get("GOLEM_GM_C_BUFFER_READ_BPC", ""),
    "gm_c_buffer_write_bytes_per_cycle": os.environ.get("GOLEM_GM_C_BUFFER_WRITE_BPC", ""),
    "gm_c_buffer_latency_cycles": os.environ.get("GOLEM_GM_C_BUFFER_LATENCY_CYCLES", ""),
    "final_c_write_enable": os.environ.get("GOLEM_FINAL_C_WRITE_ENABLE", ""),
    "dma_write_vn": os.environ.get("GOLEM_DMA_WRITE_VN", ""),
    "dirctrl_highlink_bw": os.environ.get("GOLEM_DIRCTRL_HIGHLINK_BW", ""),
    "wall_time_sec": str(wall_time_sec),
     "simulated_time": simulated_time,
     "exec_total_cycles": execution.get("total_cycles", ""),
     "gemm_system_latency_cycles": execution.get("gemm_system_latency_cycles", ""),
     "gemm_system_start_cycle": execution.get("gemm_system_start_cycle", ""),
     "gemm_system_end_cycle": execution.get("gemm_system_end_cycle", ""),
     "exec_avg_throughput_ops_per_cycle": execution.get("avg_throughput_ops_per_cycle", ""),
     "exec_system_avg_throughput_ops_per_cycle": execution.get("system_avg_throughput_ops_per_cycle", ""),
     "exec_peak_throughput_ops_per_cycle": execution.get("peak_throughput_ops_per_cycle", ""),
     "exec_array_utilization_pct": execution.get("array_utilization_pct", ""),
     "exec_system_array_utilization_pct": execution.get("system_array_utilization_pct", ""),
     "exec_worker_avg_array_efficiency_pct": execution.get("worker_avg_array_efficiency_pct", ""),
     "exec_worker_p95_total_cycles": execution.get("worker_p95_total_cycles", ""),
     "exec_worker_max_total_cycles": execution.get("worker_max_total_cycles", ""),
     "exec_breakdown_compute_active_time": execution.get("compute_active_time", ""),
     "exec_breakdown_prefetch_wait_time": execution.get("prefetch_wait_time", ""),
     "exec_breakdown_writeback_wait_time": execution.get("writeback_wait_time", ""),
    "exec_breakdown_control_other_time": execution.get("control_other_time", ""),
    "pipeline_compute_segment_count": str(bubbles["compute_segments"]),
    "descriptor_cold_start_avg_cycles_per_core": descriptor_cold_start_avg,
    "descriptor_cold_start_min_cycles": descriptor_cold_start_min,
    "descriptor_cold_start_max_cycles": descriptor_cold_start_max,
    "descriptor_cold_start_cycles_array": descriptor_cold_start_array,
    "pipeline_compute_active_cycles": str(bubbles["compute_active_cycles"]),
    "pipeline_bubble_cycles": str(pipeline_bubble_cycles),
    "pipeline_accounted_cycles": str(pipeline_accounted_cycles),
    "pipeline_efficiency_pct": pipeline_efficiency_pct,
    "pipeline_avg_bubble_cycles_per_core": pipeline_avg_bubble_cycles_per_core,
    "pipeline_intra_window_transition_count": str(bubbles["intra_window_transitions"]),
    "pipeline_intra_window_bubble_count": str(bubbles["intra_window_bubbles"]),
    "pipeline_intra_window_bubble_cycles": str(bubbles["intra_window_cycles"]),
    "pipeline_intra_window_avg_bubble_cycles_per_core": intra_window_avg_per_core,
    "pipeline_inter_window_transition_count": str(bubbles["inter_window_transitions"]),
    "pipeline_inter_window_bubble_count": str(bubbles["inter_window_bubbles"]),
    "pipeline_inter_window_bubble_cycles": str(bubbles["inter_window_cycles"]),
    "pipeline_inter_window_avg_bubble_cycles_per_core": inter_window_avg_per_core,
    "pipeline_inter_macro_transition_count": str(bubbles["inter_macro_transitions"]),
    "pipeline_inter_macro_bubble_count": str(bubbles["inter_macro_bubbles"]),
    "pipeline_inter_macro_bubble_cycles": str(bubbles["inter_macro_cycles"]),
    "pipeline_inter_macro_avg_bubble_cycles_per_core": inter_macro_avg_per_core,
    # Deprecated compatibility aliases: old intra-macro meant inter-window.
    "pipeline_intra_macro_transition_count": str(bubbles["legacy_intra_macro_transitions"]),
    "pipeline_intra_macro_bubble_count": str(bubbles["legacy_intra_macro_bubbles"]),
    "pipeline_intra_macro_bubble_cycles": str(bubbles["legacy_intra_macro_cycles"]),
    "pipeline_intra_macro_avg_bubble_cycles_per_core": (
        f"{bubbles['legacy_intra_macro_cycles'] / core_count:.6f}" if core_count else ""
    ),
     "debug_sched_protocol_mean": execution.get("debug_sched_protocol_mean", ""),
     "debug_group_wait_mean": execution.get("debug_group_wait_mean", ""),
     "dma_timeout_retry_sum": dma.get("timeout_retry_sum", ""),
    "dma_read_issue_count_sum": dma.get("read_issue_count_sum", ""),
    "dma_write_issue_count_sum": dma.get("write_issue_count_sum", ""),
    "dma_read_bytes_total_sum": dma.get("read_bytes_total_sum", ""),
    "dma_write_bytes_total_sum": dma.get("write_bytes_total_sum", ""),
    "dma_write_timeout_retry_sum": dma.get("write_timeout_retry_sum", ""),
    "dma_completion_sum": dma.get("completion_sum", ""),
    "dma_write_completion_sum": dma.get("write_completion_sum", ""),
    "dma_write_rtt_samples_sum": dma.get("write_rtt_samples_sum", ""),
    "dma_write_rtt_cycles_sum": dma.get("write_rtt_cycles_sum_sum", ""),
    "dma_write_avg_rtt_cycles": dma.get("write_avg_rtt_cycles_mean", ""),
    "dma_write_max_rtt_cycles": dma.get("write_max_rtt_cycles_max", ""),
    "dma_write_first_issue_cycle": dma.get("write_first_issue_cycle_min", ""),
    "dma_write_last_issue_cycle": dma.get("write_last_issue_cycle_max", ""),
    "dma_write_last_complete_cycle": dma.get("write_last_complete_cycle_max", ""),
    "dma_wait_count_sum": dma.get("wait_count_sum", ""),
    "dma_avg_rtt_cycles_mean": dma.get("avg_rtt_cycles_mean", ""),
    "dma_max_rtt_cycles_p95": dma.get("max_rtt_cycles_p95", ""),
    "dma_strict_rtt_samples_sum": dma.get("strict_rtt_samples_sum", ""),
    "dma_strict_rtt_cycles_sum": dma.get("strict_rtt_cycles_sum_sum", ""),
    "dma_strict_avg_rtt_cycles_mean": dma.get("strict_avg_rtt_cycles_mean", ""),
    "dma_strict_max_rtt_cycles_max": dma.get("strict_max_rtt_cycles_max", ""),
    "dma_strict_e2e_rtt_samples_sum": dma.get("strict_e2e_rtt_samples_sum", ""),
    "dma_strict_e2e_rtt_cycles_sum": dma.get("strict_e2e_rtt_cycles_sum_sum", ""),
    "dma_strict_avg_e2e_rtt_cycles_mean": dma.get("strict_avg_e2e_rtt_cycles_mean", ""),
    "dma_strict_max_e2e_rtt_cycles_max": dma.get("strict_max_e2e_rtt_cycles_max", ""),
    "noc_total_xbar_stalls": noc.get("total_xbar_stalls", ""),
    "noc_hotspot_top5pct_port_util_pct": noc.get("hotspot_top5pct_port_util_pct", ""),
    "noc_max_port_util_pct": noc.get("max_port_util_pct", ""),
    "noc_total_output_port_stalls": hotspot.get("total_output_port_stalls", ""),
    "noc_hotspot_top1_router": hotspot.get("top1_router", ""),
    "noc_hotspot_top1_router_xbar_share_pct": hotspot.get("top1_router_xbar_share_pct", ""),
    "noc_hotspot_top2_router": hotspot.get("top2_router", ""),
    "noc_hotspot_top2_router_xbar_share_pct": hotspot.get("top2_router_xbar_share_pct", ""),
    "noc_hotspot_top3_router": hotspot.get("top3_router", ""),
    "noc_hotspot_top3_router_xbar_coverage_pct": hotspot.get("top3_router_xbar_coverage_pct", ""),
    "noc_hotspot_top1_port_router": hotspot.get("top1_port_router", ""),
    "noc_hotspot_top1_port": hotspot.get("top1_port", ""),
    "noc_hotspot_top1_port_xbar_share_pct": hotspot.get("top1_port_xbar_share_pct", ""),
    "noc_avg_packet_latency_ns": noc_latency.get("noc_avg_packet_latency_ns", ""),
    "noc_p99_packet_latency_ns": noc_latency.get("noc_p99_packet_latency_ns", ""),
    "memory_avg_read_latency_cycles": memory.get("mem_avg_read_latency_cycles", ""),
    "memory_p95_read_latency_bucket_cycles": memory.get("mem_p95_read_latency_bucket_cycles", ""),
    "memory_read_tail_ge_100_pct": memory.get("mem_read_tail_ge_100_pct", ""),
    "memory_avg_write_latency_cycles": memory.get("mem_avg_write_latency_cycles", ""),
    "memory_p95_write_latency_cycles": memory.get("mem_p95_write_latency_cycles", ""),
    "hbm_data_node_count": memory.get("data_node_count", ""),
    "hbm_data_channel_count": memory.get("channel_count", ""),
    "hbm_aggregate_bandwidth_gbps": memory.get("hbm_aggregate_bandwidth_gbps", ""),
    "hbm_utilization_pct": hbm_util_pct,
    "hbm_measured_bandwidth_gbps": memory.get("hbm_aggregate_bandwidth_gbps", ""),
    "hbm_peak_bandwidth_gbps": hbm_roofline_bpc,
    "hbm_pressure_vs_gemm_system_pct": hbm_pressure_pct,
    "hbm_backend_service_window_utilization_pct": hbm_backend_window_util_pct,
    "hbm_backend_service_window_node_utilization_pct_array": memory.get(
        "hbm_read_service_window_node_utilization_pct_array", ""
    ),
    "hbm_backend_active_utilization_pct": hbm_backend_active_util_pct,
    "hbm_backend_read_window_cycles": str(backend_read_window_cycles) if backend_read_window_cycles > 0 else "",
    "hbm_backend_read_active_cycles": str(backend_read_active_cycles) if backend_read_active_cycles > 0 else "",
    "hbm_pressure_vs_worker_avg_pct": hbm_worker_pressure_pct,
    "hbm_useful_read_bytes": hbm_useful_read_bytes,
    "hbm_tccdl_roofline_bytes_per_cycle": hbm_roofline_bpc,
    "hbm_tccd_l_cycles": hbm_tccd_l_cycles,
    "hbm_channel_bandwidth_imbalance": memory.get("hbm_channel_bandwidth_imbalance", ""),
    "hbm_read_command_utilization_pct": hbm_read_command.get("hbm_read_command_utilization_pct", ""),
    "hbm_read_command_node_utilization_pct_array": hbm_read_command.get("hbm_read_command_node_utilization_pct_array", ""),
    "hbm_read_command_count": hbm_read_command.get("hbm_read_command_count", ""),
    "hbm_read_command_burst_cycles": hbm_read_command.get("hbm_read_command_burst_cycles", ""),
    "hbm_read_command_occupied_data_bus_cycles": hbm_read_command.get("hbm_read_command_occupied_data_bus_cycles", ""),
    "hbm_read_command_capacity_slots": hbm_read_command.get("hbm_read_command_capacity_slots", ""),
    "hbm_read_command_tccd_s_pair_pct": hbm_read_command.get("hbm_read_command_tccd_s_pair_pct", ""),
    "hbm_read_command_tccd_l_pair_pct": hbm_read_command.get("hbm_read_command_tccd_l_pair_pct", ""),
    "hbm_read_command_cross_pseudo_channel_pair_pct": hbm_read_command.get(
        "hbm_read_command_cross_pseudo_channel_pair_pct", ""
    ),
    "hbm_read_command_tccd_s_avg_gap_cycles": hbm_read_command.get(
        "hbm_read_command_tccd_s_avg_gap_cycles", ""
    ),
    "hbm_read_command_tccd_l_avg_gap_cycles": hbm_read_command.get(
        "hbm_read_command_tccd_l_avg_gap_cycles", ""
    ),
    "hbm_read_row_hit_pct": hbm_read_command.get("hbm_read_row_hit_pct", ""),
    "hbm_read_write_switch_count": hbm_read_command.get("hbm_read_write_switch_count", ""),
    "ramulator2_dependency_read_blocked": hbm_read_command.get("ramulator2_dependency_read_blocked", ""),
    "ramulator2_dependency_write_blocked": hbm_read_command.get("ramulator2_dependency_write_blocked", ""),
    "ramulator2_concurrent_alias_reads": hbm_read_command.get("ramulator2_concurrent_alias_reads", ""),
    "memory_queue_delay_avg_cycles": memory_queue.get("memory_queue_delay_avg_cycles", ""),
    "memory_queue_delay_p99_cycles": memory_queue.get("memory_queue_delay_p99_cycles", ""),
    "memory_backend_read_latency_avg_cycles": (
        memory_queue.get("memory_backend_read_latency_avg_cycles", "")
        or memory.get("memory_backend_read_latency_avg_cycles", "")
    ),
    "memory_backend_read_latency_p99_cycles": (
        memory_queue.get("memory_backend_read_latency_p99_cycles", "")
        or memory.get("memory_backend_read_latency_p99_cycles", "")
    ),
    "causal_model_source": causal.get("causal_model_source", ""),
    "causal_memnic_cycle_scale": causal.get("memnic_cycle_scale", ""),
    "causal_event_full_match_count": causal.get("event_full_match_count", ""),
    "causal_event_invalid_order_count": causal.get("event_invalid_order_count", ""),
    "causal_issue_to_pending_mat_mean_cycles": causal.get("causal_issue_to_pending_mat_mean_cycles", ""),
    "causal_issue_to_pending_vec_mean_cycles": causal.get("causal_issue_to_pending_vec_mean_cycles", ""),
    "causal_forward_to_memnic_mean_cycles": causal.get("causal_forward_to_memnic_mean_cycles", ""),
    "causal_memory_service_mean_cycles": causal.get("causal_memory_service_mean_cycles", ""),
    "causal_return_path_mat_mean_cycles": causal.get("causal_return_path_mat_mean_cycles", ""),
    "causal_return_path_vec_mean_cycles": causal.get("causal_return_path_vec_mean_cycles", ""),
    "causal_return_path_mat_mean_share_pct": causal.get("causal_return_path_mat_mean_share_pct", ""),
    "causal_return_path_vec_mean_share_pct": causal.get("causal_return_path_vec_mean_share_pct", ""),
    "causal_issue_to_pending_mat_p95_cycles": causal.get("causal_issue_to_pending_mat_p95_cycles", ""),
    "causal_issue_to_pending_vec_p95_cycles": causal.get("causal_issue_to_pending_vec_p95_cycles", ""),
    "causal_forward_to_memnic_p95_cycles": causal.get("causal_forward_to_memnic_p95_cycles", ""),
    "causal_memory_service_p95_cycles": causal.get("causal_memory_service_p95_cycles", ""),
    "causal_return_path_mat_p95_cycles": causal.get("causal_return_path_mat_p95_cycles", ""),
    "causal_return_path_vec_p95_cycles": causal.get("causal_return_path_vec_p95_cycles", ""),
}

out_csv.parent.mkdir(parents=True, exist_ok=True)
fieldnames = list(record.keys())
write_header = (not out_csv.exists()) or out_csv.stat().st_size == 0
if out_csv.exists() and out_csv.stat().st_size > 0:
    with out_csv.open(newline="") as f:
        existing_rows = list(csv.reader(f))
    existing_header = existing_rows[0] if existing_rows else []
    if existing_header != fieldnames:
        backup_path = out_csv.with_name(out_csv.stem + ".legacy_backup.csv")
        out_csv.replace(backup_path)
        print(f"[INFO] run summary header changed; archived legacy file to {backup_path}")
        write_header = True
with out_csv.open("a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        w.writeheader()
    w.writerow(record)

terminal_metrics = [
    ("output_mode", os.environ.get("GOLEM_OUTPUT_MODE", "hbm")),
    ("dma_write_issue_count_sum", dma.get("write_issue_count_sum", "")),
    ("dma_write_completion_sum", dma.get("write_completion_sum", "")),
    ("dma_write_first_issue_cycle", dma.get("write_first_issue_cycle_min", "")),
    ("dma_write_last_issue_cycle", dma.get("write_last_issue_cycle_max", "")),
    ("dma_write_last_complete_cycle", dma.get("write_last_complete_cycle_max", "")),
    ("c_writeback_complete_ratio", ""),
    ("c_writeback_tail_latency_pct", ""),
    ("c_fusion_consumed_ratio", ""),
    ("hbm_backend_service_window_utilization_pct", hbm_backend_window_util_pct),
    ("hbm_backend_service_window_node_utilization_pct_array", memory.get("hbm_read_service_window_node_utilization_pct_array", "")),
    ("hbm_read_command_utilization_pct", hbm_read_command.get("hbm_read_command_utilization_pct", "")),
    ("hbm_read_command_node_utilization_pct_array", hbm_read_command.get("hbm_read_command_node_utilization_pct_array", "")),
    ("hbm_read_command_tccd_s_pair_pct", hbm_read_command.get("hbm_read_command_tccd_s_pair_pct", "")),
    ("hbm_read_command_tccd_l_pair_pct", hbm_read_command.get("hbm_read_command_tccd_l_pair_pct", "")),
    ("hbm_read_command_cross_pseudo_channel_pair_pct", hbm_read_command.get("hbm_read_command_cross_pseudo_channel_pair_pct", "")),
    ("hbm_read_row_hit_pct", hbm_read_command.get("hbm_read_row_hit_pct", "")),
    ("pipeline_covered_transitions", str(pipeline_covered) if pipeline_transitions > 0 else ""),
    ("pipeline_total_transitions", str(pipeline_transitions) if pipeline_transitions > 0 else ""),
    ("pipeline_efficiency_pct", pipeline_efficiency_pct),
    ("pipeline_core_count", str(core_count) if core_count else ""),
    ("descriptor_cold_start_avg_cycles_per_core", descriptor_cold_start_avg),
    ("descriptor_cold_start_min_cycles", descriptor_cold_start_min),
    ("descriptor_cold_start_max_cycles", descriptor_cold_start_max),
    ("descriptor_cold_start_cycles_array", descriptor_cold_start_array),
    ("pipeline_compute_active_cycles", str(bubbles["compute_active_cycles"]) if core_count else ""),
    ("pipeline_bubble_cycles", str(pipeline_bubble_cycles) if core_count else ""),
    ("pipeline_accounted_cycles", str(pipeline_accounted_cycles) if core_count else ""),
    ("pipeline_avg_bubble_cycles_per_core", pipeline_avg_bubble_cycles_per_core),
    ("intra_window_transition_count", str(bubbles["intra_window_transitions"]) if core_count else ""),
    ("intra_window_bubble_count", str(bubbles["intra_window_bubbles"]) if core_count else ""),
    ("intra_window_avg_bubble_cycles_per_core", intra_window_avg_per_core),
    ("inter_window_transition_count", str(bubbles["inter_window_transitions"]) if core_count else ""),
    ("inter_window_bubble_count", str(bubbles["inter_window_bubbles"]) if core_count else ""),
    ("inter_window_avg_bubble_cycles_per_core", inter_window_avg_per_core),
    ("inter_macro_transition_count", str(bubbles["inter_macro_transitions"]) if core_count else ""),
    ("inter_macro_bubble_count", str(bubbles["inter_macro_bubbles"]) if core_count else ""),
    ("inter_macro_avg_bubble_cycles_per_core", inter_macro_avg_per_core),
    ("intra_macro_transition_count", str(bubbles["legacy_intra_macro_transitions"]) if core_count else ""),
    ("intra_macro_bubble_count", str(bubbles["legacy_intra_macro_bubbles"]) if core_count else ""),
    ("intra_macro_avg_bubble_cycles_per_core", (
        f"{bubbles['legacy_intra_macro_cycles'] / core_count:.6f}" if core_count else ""
    )),
]
with (stats_dir / "terminal_summary.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerows(terminal_metrics)

print(f"[OK] run summary appended: {out_csv}")
PY

{
	echo "SST log: $LOG_PATH"
	echo "stdout/stderr: $STDOUT_DIR"
	echo "HBM: $HBM_DIR"
	echo "DRAMSim3: $DRAMSIM_STATS_DIR"
	echo "execution: $EXEC_SUMMARY_FILE"
	echo "DMA: $DMA_SUMMARY_FILE"
	echo "NoC: $NOC_SUMMARY_FILE"
	echo "memory: $MEMORY_SUMMARY_FILE"
	echo "HBM READ commands: $HBM_READ_COMMAND_SUMMARY_FILE"
	echo "HBM READ command nodes: $HBM_READ_COMMAND_NODE_FILE"
	echo "terminal summary: $TERMINAL_SUMMARY_FILE"
	echo "run summary: $RUN_SUMMARY_CSV"
	if [[ "$TIMELINE" -eq 1 ]]; then
		echo "timeline: $TIMELINE_FILE"
	fi
} >> "$PIPELINE_DETAIL_LOG"

run_end_epoch="$(date +%s)"
wall_time_sec=$((run_end_epoch - RUN_START_EPOCH))
simulated_time="$(grep 'Simulation is complete, simulated time:' "$LOG_PATH" 2>/dev/null | tail -n 1 | sed 's/.*simulated time:[[:space:]]*//' || true)"
gemm_latency="$(csv_metric "$EXEC_SUMMARY_FILE" gemm_system_latency_cycles)"
if [[ "$gemm_latency" =~ ^([0-9]+)[.]0+$ ]]; then
	gemm_latency="${BASH_REMATCH[1]}"
fi
system_utilization_raw="$(csv_metric "$EXEC_SUMMARY_FILE" system_array_utilization_pct)"
system_utilization="not reported"
if [[ "$system_utilization_raw" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
	printf -v system_utilization '%.2f%%' "$system_utilization_raw"
fi
descriptor_cold_start_avg="$(csv_metric "$TERMINAL_SUMMARY_FILE" descriptor_cold_start_avg_cycles_per_core)"
descriptor_cold_start_min="$(csv_metric "$TERMINAL_SUMMARY_FILE" descriptor_cold_start_min_cycles)"
descriptor_cold_start_max="$(csv_metric "$TERMINAL_SUMMARY_FILE" descriptor_cold_start_max_cycles)"
descriptor_cold_start="not reported"
if [[ "$descriptor_cold_start_avg" =~ ^[0-9]+([.][0-9]+)?$ &&
      "$descriptor_cold_start_min" =~ ^[0-9]+$ && "$descriptor_cold_start_max" =~ ^[0-9]+$ ]]; then
	printf -v descriptor_cold_start '%.2f cycles/core (min %s, max %s)' \
		"$descriptor_cold_start_avg" "$descriptor_cold_start_min" "$descriptor_cold_start_max"
fi
if [[ "$GOLEM_MEMORY_BACKEND" == "ramulator2" ]]; then
	hbm_service_utilization_raw="$(csv_metric "$TERMINAL_SUMMARY_FILE" hbm_backend_service_window_utilization_pct)"
	hbm_service_node_utilization="$(csv_metric "$TERMINAL_SUMMARY_FILE" hbm_backend_service_window_node_utilization_pct_array)"
else
	hbm_service_utilization_raw="$(csv_metric "$TERMINAL_SUMMARY_FILE" hbm_read_command_utilization_pct)"
	hbm_service_node_utilization="$(csv_metric "$TERMINAL_SUMMARY_FILE" hbm_read_command_node_utilization_pct_array)"
fi
hbm_command_utilization_raw="$(csv_metric "$TERMINAL_SUMMARY_FILE" hbm_read_command_utilization_pct)"
hbm_command_node_utilization="$(csv_metric "$TERMINAL_SUMMARY_FILE" hbm_read_command_node_utilization_pct_array)"
hbm_service_utilization="not reported"
hbm_command_utilization="not reported"
if [[ "$hbm_service_utilization_raw" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
	printf -v hbm_service_utilization '%.2f%% %s' "$hbm_service_utilization_raw" "$hbm_service_node_utilization"
fi
if [[ "$hbm_command_utilization_raw" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
	printf -v hbm_command_utilization '%.2f%% %s' "$hbm_command_utilization_raw" "$hbm_command_node_utilization"
fi
dma_write_issue_count="$(csv_metric "$TERMINAL_SUMMARY_FILE" dma_write_issue_count_sum)"
dma_write_completion_count="$(csv_metric "$TERMINAL_SUMMARY_FILE" dma_write_completion_sum)"
dma_write_last_issue="$(csv_metric "$TERMINAL_SUMMARY_FILE" dma_write_last_issue_cycle)"
dma_write_last_complete="$(csv_metric "$TERMINAL_SUMMARY_FILE" dma_write_last_complete_cycle)"
writeback_complete_ratio="not reported"
writeback_tail_latency_pct="not reported"
fusion_consumed_ratio="not reported"
if [[ "$dma_write_issue_count" =~ ^[0-9]+$ && "$dma_write_completion_count" =~ ^[0-9]+$ ]]; then
	printf -v writeback_complete_ratio '%s/%s' "$dma_write_completion_count" "$dma_write_issue_count"
	if [[ "$dma_write_last_issue" =~ ^[0-9]+$ && "$dma_write_last_complete" =~ ^[0-9]+$ ]]; then
		tail_latency=$((dma_write_last_complete - dma_write_last_issue))
		if (( tail_latency < 0 )); then
			tail_latency=0
		fi
		if [[ "$gemm_latency" =~ ^[0-9]+([.][0-9]+)?$ ]] &&
		   awk -v latency="$gemm_latency" 'BEGIN { exit !(latency > 0) }'; then
			writeback_tail_latency_pct="$(awk -v tail="$tail_latency" -v latency="$gemm_latency" \
				'BEGIN { printf "%.2f", 100.0 * tail / latency }')"
		fi
	fi
fi
if [[ "$GOLEM_OUTPUT_MODE" == "fusion" ]]; then
	fusion_matches="$(rg --no-filename -o 'fusion_consumed=[0-9]+' "$STDOUT_DIR" 2>/dev/null || true)"
	if [[ -z "$fusion_matches" ]]; then
		fusion_matches="$(rg --no-filename -o 'fusion_consumed=[0-9]+' "$LOG_PATH" 2>/dev/null || true)"
	fi
	fusion_consumed_count="$(awk -F= '{ total += $2 } END { print total + 0 }' <<< "$fusion_matches")"
	printf -v fusion_consumed_ratio '%s/%s' "$fusion_consumed_count" "$DERIVED_GEMM_TOTAL_TASKS"
fi
python3 - "$TERMINAL_SUMMARY_FILE" "$writeback_complete_ratio" "$writeback_tail_latency_pct" "$fusion_consumed_ratio" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
updates = {
    "c_writeback_complete_ratio": sys.argv[2],
    "c_writeback_tail_latency_pct": sys.argv[3],
    "c_fusion_consumed_ratio": sys.argv[4],
}
rows = []
if path.exists():
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
if not rows:
    rows = [["metric", "value"]]
index = {row[0]: i for i, row in enumerate(rows[1:], 1) if row}
for name, value in updates.items():
    if name in index:
        rows[index[name]] = [name, value]
    else:
        rows.append([name, value])
with path.open("w", newline="") as f:
    csv.writer(f).writerows(rows)
PY
if [[ "$GOLEM_OUTPUT_MODE" == "fusion" ]]; then
	c_output_status="${fusion_consumed_ratio} consumed; NoC/HBM write bypassed"
else
	c_output_status="${writeback_complete_ratio} complete; tail latency ${writeback_tail_latency_pct}%"
fi
pipeline_covered="$(csv_metric "$TERMINAL_SUMMARY_FILE" pipeline_covered_transitions)"
pipeline_total="$(csv_metric "$TERMINAL_SUMMARY_FILE" pipeline_total_transitions)"
pipeline_efficiency_raw="$(csv_metric "$TERMINAL_SUMMARY_FILE" pipeline_efficiency_pct)"
pipeline_core_count="$(csv_metric "$TERMINAL_SUMMARY_FILE" pipeline_core_count)"
pipeline_active_cycles="$(csv_metric "$TERMINAL_SUMMARY_FILE" pipeline_compute_active_cycles)"
pipeline_accounted_cycles="$(csv_metric "$TERMINAL_SUMMARY_FILE" pipeline_accounted_cycles)"
pipeline_avg_bubble="$(csv_metric "$TERMINAL_SUMMARY_FILE" pipeline_avg_bubble_cycles_per_core)"
pipeline_efficiency="not reported"
if [[ "$pipeline_efficiency_raw" =~ ^[0-9]+([.][0-9]+)?$ &&
      "$pipeline_active_cycles" =~ ^[0-9]+$ && "$pipeline_accounted_cycles" =~ ^[1-9][0-9]*$ &&
      "$pipeline_core_count" =~ ^[1-9][0-9]*$ &&
      "$pipeline_avg_bubble" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
	printf -v pipeline_efficiency '%.2f%% (%s/%s active cycles, avg bubble %.2f cycles/core, %s cores)' \
		"$pipeline_efficiency_raw" "$pipeline_active_cycles" "$pipeline_accounted_cycles" \
		"$pipeline_avg_bubble" "$pipeline_core_count"
fi
intra_window_transitions="$(csv_metric "$TERMINAL_SUMMARY_FILE" intra_window_transition_count)"
intra_window_bubbles="$(csv_metric "$TERMINAL_SUMMARY_FILE" intra_window_bubble_count)"
intra_window_avg="$(csv_metric "$TERMINAL_SUMMARY_FILE" intra_window_avg_bubble_cycles_per_core)"
inter_window_transitions="$(csv_metric "$TERMINAL_SUMMARY_FILE" inter_window_transition_count)"
inter_window_bubbles="$(csv_metric "$TERMINAL_SUMMARY_FILE" inter_window_bubble_count)"
inter_window_avg="$(csv_metric "$TERMINAL_SUMMARY_FILE" inter_window_avg_bubble_cycles_per_core)"
inter_macro_transitions="$(csv_metric "$TERMINAL_SUMMARY_FILE" inter_macro_transition_count)"
inter_macro_bubbles="$(csv_metric "$TERMINAL_SUMMARY_FILE" inter_macro_bubble_count)"
inter_macro_avg="$(csv_metric "$TERMINAL_SUMMARY_FILE" inter_macro_avg_bubble_cycles_per_core)"
intra_window_bubble="not reported"
inter_window_bubble="not reported"
inter_macro_bubble="not reported"
if [[ "$intra_window_bubbles" =~ ^[0-9]+$ && "$intra_window_transitions" =~ ^[0-9]+$ &&
      "$intra_window_avg" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
	printf -v intra_window_bubble '%s/%s nonzero (%.2f cycles/core)' \
		"$intra_window_bubbles" "$intra_window_transitions" "$intra_window_avg"
fi
if [[ "$inter_window_bubbles" =~ ^[0-9]+$ && "$inter_window_transitions" =~ ^[0-9]+$ &&
      "$inter_window_avg" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
	printf -v inter_window_bubble '%s/%s nonzero (%.2f cycles/core)' \
		"$inter_window_bubbles" "$inter_window_transitions" "$inter_window_avg"
fi
if [[ "$inter_macro_bubbles" =~ ^[0-9]+$ && "$inter_macro_transitions" =~ ^[0-9]+$ &&
      "$inter_macro_avg" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
	printf -v inter_macro_bubble '%s/%s nonzero (%.2f cycles/core)' \
		"$inter_macro_bubbles" "$inter_macro_transitions" "$inter_macro_avg"
fi

ui_success "Reports ready"
ui_header "RESULT"
ui_kv "Wall time" "$(format_elapsed "$wall_time_sec") (${wall_time_sec}s)"
ui_kv "Mode" "$GOLEM_SIM_MODE"
ui_kv "Simulated" "${simulated_time:-not reported}"
ui_kv "GEMM latency" "${gemm_latency:-not reported} cycles"
ui_kv "System utilization" "$system_utilization"
ui_kv "Descriptor cold start" "$descriptor_cold_start"
ui_kv "HBM service utilization" "$hbm_service_utilization"
if [[ "$GOLEM_MEMORY_BACKEND" == "ramulator2" ]]; then
	ui_kv "HBM RD issue utilization" "$hbm_command_utilization"
fi
ui_kv "C output" "$c_output_status"
ui_kv "Pipeline efficiency" "$pipeline_efficiency"
ui_kv "Intra-window bubble" "$intra_window_bubble"
ui_kv "Inter-window bubble" "$inter_window_bubble"
ui_kv "Inter-macro bubble" "$inter_macro_bubble"
ui_kv "Artifacts" "$(display_path "$STATS_DIR")"
ui_kv "SST log" "$(display_path "$LOG_PATH")"
ui_kv "Details" "$(display_path "$PIPELINE_DETAIL_LOG")"
if [[ "$TIMELINE" -eq 1 ]]; then
	ui_kv "Timeline SVG" "$(display_path "$TIMELINE_FILE")"
fi
