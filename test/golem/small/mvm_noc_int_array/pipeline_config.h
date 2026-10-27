#pragma once

#include <algorithm>
#include <cstdint>

#ifndef GOLEM_ARRAY_INPUT_SIZE
#define GOLEM_ARRAY_INPUT_SIZE 16
#endif

#ifndef GOLEM_ARRAY_OUTPUT_SIZE
#define GOLEM_ARRAY_OUTPUT_SIZE 16
#endif

#ifndef GOLEM_TOTAL_GEMM_CORES
#define GOLEM_TOTAL_GEMM_CORES 16
#endif

#ifndef GOLEM_NUM_ARRAYS
#define GOLEM_NUM_ARRAYS 1
#endif

#ifndef GOLEM_GROUP_GRANT_WINDOW
#define GOLEM_GROUP_GRANT_WINDOW 1
#endif

#ifndef GOLEM_TOTAL_GROUPS
#define GOLEM_TOTAL_GROUPS 4
#endif

#ifndef GOLEM_DMA_STAGGER_CYCLES
#define GOLEM_DMA_STAGGER_CYCLES 0
#endif

#ifndef GOLEM_DMA_OVERLAP
#define GOLEM_DMA_OVERLAP 0
#endif

#ifndef GOLEM_GROUP_MANAGER_ENABLE
#define GOLEM_GROUP_MANAGER_ENABLE 1
#endif

#ifndef GOLEM_GROUP_MAX_INFLIGHT_PER_NODE
#define GOLEM_GROUP_MAX_INFLIGHT_PER_NODE 2
#endif

#ifndef GOLEM_CTRL_LINK_ENABLE
#define GOLEM_CTRL_LINK_ENABLE 0
#endif

#ifndef GOLEM_CTRL_OVERLAP_AB
#define GOLEM_CTRL_OVERLAP_AB 0
#endif

#ifndef GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE
#define GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE 0
#endif

#ifndef GOLEM_WORKER_START_BARRIER_ENABLE
#define GOLEM_WORKER_START_BARRIER_ENABLE 1
#endif

#ifndef GOLEM_WORKER_START_GUARD_CYCLES
#define GOLEM_WORKER_START_GUARD_CYCLES 32768
#endif

#ifndef GOLEM_A_REUSE_N_TILES
#define GOLEM_A_REUSE_N_TILES 1
#endif

#ifndef GOLEM_B_REUSE_M_TILES
#define GOLEM_B_REUSE_M_TILES 1
#endif

#ifndef GOLEM_DMA_SLOT_COUNT
#define GOLEM_DMA_SLOT_COUNT 4
#endif



#ifndef GOLEM_GEMM_M
#define GOLEM_GEMM_M GOLEM_ARRAY_OUTPUT_SIZE
#endif

#ifndef GOLEM_GEMM_N
#define GOLEM_GEMM_N GOLEM_NUM_ARRAYS
#endif

#ifndef GOLEM_GEMM_K
#define GOLEM_GEMM_K GOLEM_ARRAY_INPUT_SIZE
#endif

#ifndef GOLEM_GEMM_BLOCK_M
#define GOLEM_GEMM_BLOCK_M GOLEM_ARRAY_OUTPUT_SIZE
#endif

#ifndef GOLEM_GEMM_BLOCK_N
#define GOLEM_GEMM_BLOCK_N GOLEM_NUM_ARRAYS
#endif

#ifndef GOLEM_GEMM_BLOCK_K
#define GOLEM_GEMM_BLOCK_K GOLEM_ARRAY_INPUT_SIZE
#endif

#ifndef GOLEM_BIAS_ENABLE
#define GOLEM_BIAS_ENABLE 0
#endif

#ifndef GOLEM_BIAS_VALUE
#define GOLEM_BIAS_VALUE 0
#endif

#ifndef GOLEM_NUM_MEMORY_NODES
#define GOLEM_NUM_MEMORY_NODES 5
#endif

#ifndef GOLEM_MEM_NODE_SIZE_BYTES
#define GOLEM_MEM_NODE_SIZE_BYTES 0x04000000ULL
#endif

#ifndef GOLEM_GLOBAL_STRIDE_BYTES
#define GOLEM_GLOBAL_STRIDE_BYTES 65536
#endif

constexpr uint64_t align_up_constexpr(uint64_t value, uint64_t align) {
    return ((value + align - 1) / align) * align;
}

// ============================
// 1) 并行拓扑参数
// ============================
constexpr int TOTAL_GROUPS = GOLEM_TOTAL_GROUPS;
constexpr int GROUP_SIZE = 4;
constexpr int TOTAL_PIPELINE_CORES = TOTAL_GROUPS * GROUP_SIZE;
constexpr int TOTAL_GEMM_CORES = GOLEM_TOTAL_GEMM_CORES;
constexpr int NUM_ARRAYS = GOLEM_NUM_ARRAYS;
constexpr uint32_t DMA_STAGGER_CYCLES = static_cast<uint32_t>(GOLEM_DMA_STAGGER_CYCLES);
constexpr bool DMA_OVERLAP_ENABLED = (GOLEM_DMA_OVERLAP != 0);
constexpr bool GROUP_MANAGER_ENABLED = (GOLEM_GROUP_MANAGER_ENABLE != 0);
constexpr bool CTRL_LINK_ENABLED = (GOLEM_CTRL_LINK_ENABLE != 0);
constexpr bool CTRL_OVERLAP_AB_ENABLED = (GOLEM_CTRL_OVERLAP_AB != 0);
constexpr bool WORKER_COMMAND_PROCESSOR_ENABLED = (GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE != 0);
constexpr bool WORKER_START_BARRIER_ENABLED = (GOLEM_WORKER_START_BARRIER_ENABLE != 0);
constexpr uint64_t WORKER_START_GUARD_CYCLES = static_cast<uint64_t>(GOLEM_WORKER_START_GUARD_CYCLES);
constexpr int GROUP_MAX_INFLIGHT_PER_NODE = GOLEM_GROUP_MAX_INFLIGHT_PER_NODE;

constexpr int TILE_M = GOLEM_ARRAY_OUTPUT_SIZE;
constexpr int TILE_K = GOLEM_ARRAY_INPUT_SIZE;
constexpr int TILE_N_MAX = GOLEM_NUM_ARRAYS;
constexpr int GEMM_M = GOLEM_GEMM_M;
constexpr int GEMM_N = GOLEM_GEMM_N;
constexpr int GEMM_K = GOLEM_GEMM_K;
constexpr int GEMM_BLOCK_M = GOLEM_GEMM_BLOCK_M;
constexpr int GEMM_BLOCK_N = GOLEM_GEMM_BLOCK_N;
constexpr int GEMM_BLOCK_K = GOLEM_GEMM_BLOCK_K;
constexpr bool BIAS_ENABLED = (GOLEM_BIAS_ENABLE != 0);
constexpr int32_t BIAS_VALUE = static_cast<int32_t>(GOLEM_BIAS_VALUE);
constexpr int GEMM_M_TILES = (GEMM_M >= GEMM_BLOCK_M) ? (GEMM_M / GEMM_BLOCK_M) : 1;
constexpr int GEMM_N_TILES = (GEMM_N >= GEMM_BLOCK_N) ? (GEMM_N / GEMM_BLOCK_N) : 1;
constexpr int GEMM_K_TILES = (GEMM_K >= GEMM_BLOCK_K) ? (GEMM_K / GEMM_BLOCK_K) : 1;
constexpr int TOTAL_GEMM_TASKS = GEMM_M_TILES * GEMM_N_TILES;
constexpr int A_REUSE_N_TILES_RAW = GOLEM_A_REUSE_N_TILES;
constexpr int A_REUSE_N_TILES = (A_REUSE_N_TILES_RAW > 0) ? A_REUSE_N_TILES_RAW : 1;
constexpr int B_REUSE_M_TILES_RAW = GOLEM_B_REUSE_M_TILES;
constexpr int B_REUSE_M_TILES = (B_REUSE_M_TILES_RAW > 0) ? B_REUSE_M_TILES_RAW : 1;
constexpr bool A_REUSE_ENABLED = (A_REUSE_N_TILES > 1);
constexpr bool B_REUSE_ENABLED = (B_REUSE_M_TILES > 1);
constexpr int LOCAL_SLOT_COUNT = (GOLEM_DMA_SLOT_COUNT > 0) ? GOLEM_DMA_SLOT_COUNT : 4;
constexpr int GEMM_M_GROUPS = (GEMM_M_TILES + B_REUSE_M_TILES - 1) / B_REUSE_M_TILES;
constexpr int GEMM_N_GROUPS = (GEMM_N_TILES + A_REUSE_N_TILES - 1) / A_REUSE_N_TILES;
constexpr int TOTAL_GEMM_MACRO_TASKS = GEMM_M_GROUPS * GEMM_N_GROUPS;
constexpr int GEMM_MAT_REUSE_SLOTS = B_REUSE_ENABLED ? B_REUSE_M_TILES : 1;
constexpr int GEMM_VEC_REUSE_SLOTS = A_REUSE_ENABLED ? A_REUSE_N_TILES : 1;
constexpr int GEMM_OUT_REUSE_SLOTS = GEMM_MAT_REUSE_SLOTS * GEMM_VEC_REUSE_SLOTS;
constexpr int DEDICATED_MANAGER_CORES = GROUP_MANAGER_ENABLED ? TOTAL_GROUPS : 0;
constexpr int ACTIVE_GEMM_CORES = TOTAL_GEMM_CORES - DEDICATED_MANAGER_CORES;

// ============================
// 2) 计算规模参数
//    由阵列硬件 step 推导 micro-tile 尺寸
// ============================
constexpr uint64_t MAT_ROWS = static_cast<uint64_t>(TILE_M);
constexpr uint64_t MAT_COLS = static_cast<uint64_t>(TILE_K);
#ifndef GOLEM_COMPILE_ELEM_BYTES
#define GOLEM_COMPILE_ELEM_BYTES 4
#endif
constexpr uint64_t ELEM_BYTES = static_cast<uint64_t>(GOLEM_COMPILE_ELEM_BYTES);
constexpr uint64_t MAT_ELEMS = MAT_ROWS * MAT_COLS;
constexpr uint64_t VEC_ELEMS = MAT_COLS;
constexpr uint64_t MAT_BYTES = MAT_ELEMS * ELEM_BYTES;
constexpr uint64_t VEC_BYTES = VEC_ELEMS * ELEM_BYTES;
constexpr uint64_t OUT_VEC_ELEMS = MAT_ROWS;
constexpr uint64_t OUT_VEC_BYTES = OUT_VEC_ELEMS * ELEM_BYTES;
constexpr uint64_t GEMM_BLOCK_MAT_BYTES = static_cast<uint64_t>(GEMM_BLOCK_M) * static_cast<uint64_t>(GEMM_BLOCK_K) * ELEM_BYTES;
constexpr uint64_t GEMM_BLOCK_VEC_BYTES = static_cast<uint64_t>(GEMM_BLOCK_K) * ELEM_BYTES;
constexpr uint64_t GEMM_BLOCK_VEC_PACK_BYTES = static_cast<uint64_t>(GEMM_BLOCK_N) * GEMM_BLOCK_VEC_BYTES;
constexpr uint64_t GEMM_BLOCK_OUT_TILE_BYTES = static_cast<uint64_t>(GEMM_BLOCK_M) * static_cast<uint64_t>(GEMM_BLOCK_N) * ELEM_BYTES;

// ============================
// 3) 主存(Identity Window)源地址配置
//    remote_load(src>=IDENTITY_BASE, dst_gm) 自动走 DMA
//    地址随 DIM 自动推导
// ============================
constexpr int NUM_MEMORY_NODES = GOLEM_NUM_MEMORY_NODES;
constexpr int OS_MEMORY_NODE_INDEX = 0;
constexpr int DATA_MEMORY_NODE_COUNT = NUM_MEMORY_NODES - 1;
constexpr uint64_t MEM_NODE_SIZE = static_cast<uint64_t>(GOLEM_MEM_NODE_SIZE_BYTES);
constexpr uint64_t IDENTITY_BASE = MEM_NODE_SIZE;

constexpr int gemm_const_owner_core_for_task(int task_id) {
    if (ACTIVE_GEMM_CORES <= 0) {
        return DEDICATED_MANAGER_CORES;
    }
    return DEDICATED_MANAGER_CORES + (task_id % ACTIVE_GEMM_CORES);
}

constexpr int gemm_const_group_id_for_core(int core_id) {
    return (TOTAL_GROUPS <= 0) ? 0 : (core_id % TOTAL_GROUPS);
}

constexpr int gemm_const_data_node_for_task(int task_id) {
    if (DATA_MEMORY_NODE_COUNT <= 0) {
        return 1;
    }
    return 1 + (gemm_const_group_id_for_core(gemm_const_owner_core_for_task(task_id)) % DATA_MEMORY_NODE_COUNT);
}

constexpr int gemm_const_macro_tasks_for_data_node(int node_idx) {
    int count = 0;
    for (int task_id = 0; task_id < TOTAL_GEMM_MACRO_TASKS; ++task_id) {
        if (gemm_const_data_node_for_task(task_id) == node_idx) {
            ++count;
        }
    }
    return count;
}

constexpr int gemm_const_max_macro_tasks_per_data_node() {
    int max_count = 0;
    for (int node_idx = 1; node_idx <= DATA_MEMORY_NODE_COUNT; ++node_idx) {
        const int count = gemm_const_macro_tasks_for_data_node(node_idx);
        if (count > max_count) {
            max_count = count;
        }
    }
    return max_count > 0 ? max_count : 1;
}

constexpr int MAX_GEMM_MACRO_TASKS_PER_DATA_NODE = gemm_const_max_macro_tasks_per_data_node();

constexpr uint64_t node_base_addr(int node_idx) {
    return static_cast<uint64_t>(node_idx) * MEM_NODE_SIZE;
}

constexpr uint64_t MM_ALIGN = 0x100;
constexpr uint64_t MM_MAT_STRIDE = align_up_constexpr(GEMM_BLOCK_MAT_BYTES, MM_ALIGN);
constexpr uint64_t stage_mat_src_mm(int stage_id) {
    return IDENTITY_BASE + static_cast<uint64_t>(stage_id) * MM_MAT_STRIDE;
}
constexpr uint64_t INIT_VEC_SRC_MM = IDENTITY_BASE + TOTAL_GROUPS * MM_MAT_STRIDE;

// GEMM模式：A/B packed once, C remains one output tile per logical task.
// A tile storage key: (m_tile, k_tile). B tile storage key: (n_tile, k_tile, n_col).
// C tile storage key: (m_tile, n_tile), packed by macro-task slot plus reuse offset.
// B vectors are packed back-to-back.  Their payload width, rather than the
// 256B request alignment, determines the address stride so FP16 uses all bytes.
constexpr uint64_t GEMM_VEC_STRIDE_MM = GEMM_BLOCK_VEC_BYTES;
constexpr uint64_t OFF_GEMM_MAT_BASE = 0x0;
constexpr uint64_t GEMM_OUT_STRIDE_MM = align_up_constexpr(GEMM_BLOCK_OUT_TILE_BYTES, MM_ALIGN);
constexpr uint64_t GEMM_BIAS_STRIDE_MM = align_up_constexpr(static_cast<uint64_t>(GEMM_N) * ELEM_BYTES, MM_ALIGN);
constexpr uint64_t OFF_GEMM_BIAS_BASE = MEM_NODE_SIZE - GEMM_BIAS_STRIDE_MM;

constexpr int gemm_const_m_group_for_m_tile(int m_tile) {
    return m_tile / B_REUSE_M_TILES;
}

constexpr int gemm_const_n_group_for_n_tile(int n_tile) {
    return n_tile / A_REUSE_N_TILES;
}

constexpr int gemm_const_macro_task_for_group(int m_group, int n_group) {
    const int n_band = (n_group - m_group + GEMM_N_GROUPS) % GEMM_N_GROUPS;
    return n_band * GEMM_M_GROUPS + m_group;
}

constexpr int gemm_const_macro_task_for_tile(int m_tile, int n_tile) {
    return gemm_const_macro_task_for_group(
        gemm_const_m_group_for_m_tile(m_tile),
        gemm_const_n_group_for_n_tile(n_tile));
}

constexpr int gemm_const_a_data_node_for_m_tile(int m_tile) {
    return gemm_const_data_node_for_task(
        gemm_const_m_group_for_m_tile(m_tile));
}

constexpr int gemm_const_b_data_node_for_n_tile(int n_tile) {
    return gemm_const_data_node_for_task(
        gemm_const_n_group_for_n_tile(n_tile));
}

constexpr int gemm_const_a_m_tiles_for_data_node(int node_idx) {
    int count = 0;
    for (int m_tile = 0; m_tile < GEMM_M_TILES; ++m_tile) {
        if (gemm_const_a_data_node_for_m_tile(m_tile) == node_idx) {
            ++count;
        }
    }
    return count;
}

constexpr int gemm_const_b_n_tiles_for_data_node(int node_idx) {
    int count = 0;
    for (int n_tile = 0; n_tile < GEMM_N_TILES; ++n_tile) {
        if (gemm_const_b_data_node_for_n_tile(n_tile) == node_idx) {
            ++count;
        }
    }
    return count;
}

constexpr int gemm_const_max_a_m_tiles_per_data_node() {
    int max_count = 0;
    for (int node_idx = 1; node_idx <= DATA_MEMORY_NODE_COUNT; ++node_idx) {
        const int count = gemm_const_a_m_tiles_for_data_node(node_idx);
        if (count > max_count) {
            max_count = count;
        }
    }
    return max_count > 0 ? max_count : 1;
}

constexpr int gemm_const_max_b_n_tiles_per_data_node() {
    int max_count = 0;
    for (int node_idx = 1; node_idx <= DATA_MEMORY_NODE_COUNT; ++node_idx) {
        const int count = gemm_const_b_n_tiles_for_data_node(node_idx);
        if (count > max_count) {
            max_count = count;
        }
    }
    return max_count > 0 ? max_count : 1;
}

constexpr int MAX_GEMM_A_M_TILES_PER_DATA_NODE = gemm_const_max_a_m_tiles_per_data_node();
constexpr int MAX_GEMM_B_N_TILES_PER_DATA_NODE = gemm_const_max_b_n_tiles_per_data_node();

constexpr uint64_t OFF_GEMM_VEC_BASE =
    OFF_GEMM_MAT_BASE + static_cast<uint64_t>(MAX_GEMM_A_M_TILES_PER_DATA_NODE * GEMM_K_TILES) * MM_MAT_STRIDE;
constexpr uint64_t OFF_GEMM_OUT_BASE =
    OFF_GEMM_VEC_BASE + static_cast<uint64_t>(MAX_GEMM_B_N_TILES_PER_DATA_NODE * GEMM_K_TILES * GEMM_BLOCK_N) * GEMM_VEC_STRIDE_MM;
constexpr uint64_t GEMM_DATA_REGION_END = OFF_GEMM_OUT_BASE + static_cast<uint64_t>(MAX_GEMM_MACRO_TASKS_PER_DATA_NODE * GEMM_OUT_REUSE_SLOTS) * GEMM_OUT_STRIDE_MM;

struct MatmulRuntimeConfig {
    int m;
    int n;
    int k;
    int block_m;
    int block_n;
    int block_k;
};

inline MatmulRuntimeConfig default_matmul_runtime_config() {
    return {
        .m = GEMM_M,
        .n = GEMM_N,
        .k = GEMM_K,
        .block_m = GEMM_BLOCK_M,
        .block_n = GEMM_BLOCK_N,
        .block_k = GEMM_BLOCK_K,
    };
}

struct GemmTaskDescriptor {
    int core_id;
    int task_id;
    int m_tile;
    int n_tile;
    int data_node_idx;
    int task_slot_in_node;
    int m;
    int n;
    int k;
    int block_m;
    int block_n;
    int block_k;
    int k_tiles;
    uint64_t a_base_mm;
    uint64_t b_pack_base_mm;
    uint64_t c_base_mm;
    uint64_t bias_base_mm;
};

inline int gemm_m_tiles(const MatmulRuntimeConfig& cfg) {
    return cfg.m / cfg.block_m;
}

inline int gemm_n_tiles(const MatmulRuntimeConfig& cfg) {
    return cfg.n / cfg.block_n;
}

inline int gemm_k_tiles(const MatmulRuntimeConfig& cfg) {
    return cfg.k / cfg.block_k;
}

inline int gemm_total_tasks(const MatmulRuntimeConfig& cfg) {
    return gemm_m_tiles(cfg) * gemm_n_tiles(cfg);
}

inline int gemm_a_reuse_n_tiles() {
    return A_REUSE_N_TILES;
}

inline int gemm_b_reuse_m_tiles() {
    return B_REUSE_M_TILES;
}

inline bool gemm_b_reuse_enabled() {
    return B_REUSE_ENABLED;
}

inline int gemm_m_groups(const MatmulRuntimeConfig& cfg) {
    const int m_tiles = gemm_m_tiles(cfg);
    const int reuse_m = gemm_b_reuse_m_tiles();
    return (m_tiles + reuse_m - 1) / reuse_m;
}

inline int gemm_n_groups(const MatmulRuntimeConfig& cfg) {
    const int n_tiles = gemm_n_tiles(cfg);
    const int reuse_n = gemm_a_reuse_n_tiles();
    return (n_tiles + reuse_n - 1) / reuse_n;
}

inline int gemm_total_macro_tasks(const MatmulRuntimeConfig& cfg) {
    return gemm_m_groups(cfg) * gemm_n_groups(cfg);
}

inline uint64_t gemm_off_vec_base(const MatmulRuntimeConfig& cfg) {
    (void)cfg;
    return OFF_GEMM_MAT_BASE + static_cast<uint64_t>(MAX_GEMM_A_M_TILES_PER_DATA_NODE * GEMM_K_TILES) * MM_MAT_STRIDE;
}

inline uint64_t gemm_out_stride_mm(const MatmulRuntimeConfig& cfg) {
    const uint64_t out_tile_bytes = static_cast<uint64_t>(cfg.block_m) * static_cast<uint64_t>(cfg.block_n) * ELEM_BYTES;
    return align_up_constexpr(out_tile_bytes, MM_ALIGN);
}

inline uint64_t gemm_off_out_base(const MatmulRuntimeConfig& cfg) {
    return gemm_off_vec_base(cfg) + static_cast<uint64_t>(MAX_GEMM_B_N_TILES_PER_DATA_NODE * GEMM_K_TILES * cfg.block_n) * GEMM_VEC_STRIDE_MM;
}

inline int gemm_task_id_for_core(int core_id, const MatmulRuntimeConfig& cfg) {
    const int total_tasks = gemm_total_macro_tasks(cfg);
    if (total_tasks <= 0) {
        return 0;
    }
    const int worker_slot = GROUP_MANAGER_ENABLED ? (core_id - TOTAL_GROUPS) : core_id;
    return worker_slot % total_tasks;
}

inline int gemm_m_group_of_macro_task(int macro_task_id, const MatmulRuntimeConfig& cfg);
inline int gemm_n_group_of_macro_task(int macro_task_id, const MatmulRuntimeConfig& cfg);

inline int gemm_m_tile_of_task(int task_id, const MatmulRuntimeConfig& cfg) {
    return gemm_m_group_of_macro_task(task_id, cfg) * gemm_b_reuse_m_tiles();
}

inline int gemm_n_tile_of_task(int task_id, const MatmulRuntimeConfig& cfg) {
    return gemm_n_group_of_macro_task(task_id, cfg) * gemm_a_reuse_n_tiles();
}

inline int gemm_n_group_of_macro_task(int macro_task_id, const MatmulRuntimeConfig& cfg) {
    const int m_groups = gemm_m_groups(cfg);
    const int n_groups = gemm_n_groups(cfg);
    const int m_group = macro_task_id % m_groups;
    return ((macro_task_id / m_groups) + m_group) % n_groups;
}

inline int gemm_n_count_for_group(int n_group, const MatmulRuntimeConfig& cfg) {
    const int n_begin = n_group * gemm_a_reuse_n_tiles();
    const int remain = gemm_n_tiles(cfg) - n_begin;
    return remain < gemm_a_reuse_n_tiles() ? remain : gemm_a_reuse_n_tiles();
}

inline int gemm_m_group_of_macro_task(int macro_task_id, const MatmulRuntimeConfig& cfg) {
    return macro_task_id % gemm_m_groups(cfg);
}

inline int gemm_m_count_for_group(int m_group, const MatmulRuntimeConfig& cfg) {
    const int m_begin = m_group * gemm_b_reuse_m_tiles();
    const int remain = gemm_m_tiles(cfg) - m_begin;
    return remain < gemm_b_reuse_m_tiles() ? remain : gemm_b_reuse_m_tiles();
}

inline bool is_gemm_worker_core(int core_id) {
    if (!GROUP_MANAGER_ENABLED) {
        return core_id < ACTIVE_GEMM_CORES;
    }
    return core_id >= TOTAL_GROUPS && core_id < TOTAL_GEMM_CORES;
}

inline int gemm_worker_slot_for_core(int core_id) {
    if (!is_gemm_worker_core(core_id)) {
        return -1;
    }
    return GROUP_MANAGER_ENABLED ? (core_id - TOTAL_GROUPS) : core_id;
}

inline int gemm_worker_core_for_slot(int worker_slot) {
    return GROUP_MANAGER_ENABLED ? (worker_slot + TOTAL_GROUPS) : worker_slot;
}

inline int gemm_owner_core_for_task(int task_id) {
    if (ACTIVE_GEMM_CORES <= 0) {
        return 0;
    }
    return gemm_worker_core_for_slot(task_id % ACTIVE_GEMM_CORES);
}

inline int gemm_group_id_for_core(int core_id) {
    if (TOTAL_GROUPS <= 0) {
        return 0;
    }
    return core_id % TOTAL_GROUPS;
}

inline int gemm_primary_data_node_for_group(int group_id) {
    if (DATA_MEMORY_NODE_COUNT <= 0) {
        return 1;
    }
    return 1 + (group_id % DATA_MEMORY_NODE_COUNT);
}

inline int gemm_data_node_for_task(int task_id, const MatmulRuntimeConfig& cfg) {
    (void)cfg;
    const int owner_core = gemm_owner_core_for_task(task_id);
    const int group_id = gemm_group_id_for_core(owner_core);
    return gemm_primary_data_node_for_group(group_id);
}

inline int gemm_local_task_slot(int task_id, const MatmulRuntimeConfig& cfg) {
    const int node_idx = gemm_data_node_for_task(task_id, cfg);
    int slot = 0;
    for (int i = 0; i < task_id; ++i) {
        if (gemm_data_node_for_task(i, cfg) == node_idx) {
            ++slot;
        }
    }
    return slot;
}

inline int gemm_macro_task_for_tile(int m_tile, int n_tile, const MatmulRuntimeConfig& cfg) {
    const int m_group = m_tile / gemm_b_reuse_m_tiles();
    const int n_group = n_tile / gemm_a_reuse_n_tiles();
    const int n_groups = gemm_n_groups(cfg);
    const int n_band = (n_group - m_group + n_groups) % n_groups;
    return n_band * gemm_m_groups(cfg) + m_group;
}

inline int gemm_a_data_node_for_m_tile(int m_tile, const MatmulRuntimeConfig& cfg) {
    (void)cfg;
    const int m_group = m_tile / gemm_b_reuse_m_tiles();
    return gemm_data_node_for_task(m_group, cfg);
}

inline int gemm_b_data_node_for_n_tile(int n_tile, const MatmulRuntimeConfig& cfg) {
    const int n_group = n_tile / gemm_a_reuse_n_tiles();
    return gemm_data_node_for_task(n_group, cfg);
}

inline int gemm_a_slot_for_m_tile(int m_tile, const MatmulRuntimeConfig& cfg) {
    const int node_idx = gemm_a_data_node_for_m_tile(m_tile, cfg);
    int slot = 0;
    for (int i = 0; i < m_tile; ++i) {
        if (gemm_a_data_node_for_m_tile(i, cfg) == node_idx) {
            ++slot;
        }
    }
    return slot;
}

inline int gemm_b_slot_for_n_tile(int n_tile, const MatmulRuntimeConfig& cfg) {
    const int node_idx = gemm_b_data_node_for_n_tile(n_tile, cfg);
    int slot = 0;
    for (int i = 0; i < n_tile; ++i) {
        if (gemm_b_data_node_for_n_tile(i, cfg) == node_idx) {
            ++slot;
        }
    }
    return slot;
}

inline GemmTaskDescriptor gemm_task_desc_for_task(int core_id, int task_id, const MatmulRuntimeConfig& cfg) {
    const int n_tiles = gemm_n_tiles(cfg);
    const int m_tile = task_id / n_tiles;
    const int n_tile = task_id % n_tiles;
    const int macro_task_id = gemm_macro_task_for_tile(m_tile, n_tile, cfg);
    const int node_idx = gemm_data_node_for_task(macro_task_id, cfg);
    const int task_slot = gemm_local_task_slot(macro_task_id, cfg);
    const int a_node_idx = gemm_a_data_node_for_m_tile(m_tile, cfg);
    const int b_node_idx = gemm_b_data_node_for_n_tile(n_tile, cfg);
    const int a_slot = gemm_a_slot_for_m_tile(m_tile, cfg);
    const int b_slot = gemm_b_slot_for_n_tile(n_tile, cfg);
    const int mat_slots = gemm_b_reuse_enabled() ? gemm_b_reuse_m_tiles() : 1;
    const int vec_slots = gemm_a_reuse_n_tiles() > 1 ? gemm_a_reuse_n_tiles() : 1;
    const int reuse_offset = (m_tile % mat_slots) * vec_slots + (n_tile % vec_slots);
    const int k_tiles = gemm_k_tiles(cfg);
    const uint64_t task_slot_u64 = static_cast<uint64_t>(task_slot);
    const uint64_t off_vec_base = gemm_off_vec_base(cfg);
    const uint64_t off_out_base = gemm_off_out_base(cfg);
    const uint64_t out_stride = gemm_out_stride_mm(cfg);
    return {
        .core_id = core_id,
        .task_id = task_id,
        .m_tile = m_tile,
        .n_tile = n_tile,
        .data_node_idx = node_idx,
        .task_slot_in_node = task_slot,
        .m = cfg.m,
        .n = cfg.n,
        .k = cfg.k,
        .block_m = cfg.block_m,
        .block_n = cfg.block_n,
        .block_k = cfg.block_k,
        .k_tiles = k_tiles,
        .a_base_mm = node_base_addr(a_node_idx) + OFF_GEMM_MAT_BASE + static_cast<uint64_t>(a_slot * k_tiles) * MM_MAT_STRIDE,
        .b_pack_base_mm = node_base_addr(b_node_idx) + off_vec_base + static_cast<uint64_t>(b_slot * k_tiles * cfg.block_n) * GEMM_VEC_STRIDE_MM,
        .c_base_mm = node_base_addr(node_idx) + off_out_base + (task_slot_u64 * static_cast<uint64_t>(mat_slots * vec_slots) + static_cast<uint64_t>(reuse_offset)) * out_stride,
        .bias_base_mm = node_base_addr(node_idx) + OFF_GEMM_BIAS_BASE,
    };
}

inline GemmTaskDescriptor gemm_task_desc_for_core(int core_id, const MatmulRuntimeConfig& cfg) {
    return gemm_task_desc_for_task(core_id, gemm_task_id_for_core(core_id, cfg), cfg);
}

constexpr uint64_t gemm_desc_a_src_mm(const GemmTaskDescriptor& desc, int k_tile) {
    return desc.a_base_mm + static_cast<uint64_t>(k_tile) * MM_MAT_STRIDE;
}

constexpr uint64_t gemm_desc_b_pack_src_mm(const GemmTaskDescriptor& desc, int k_tile, int n_col_in_tile) {
    const uint64_t vec_slot = static_cast<uint64_t>(k_tile) * static_cast<uint64_t>(desc.block_n) + static_cast<uint64_t>(n_col_in_tile);
    return desc.b_pack_base_mm + vec_slot * GEMM_VEC_STRIDE_MM;
}

constexpr uint64_t gemm_desc_bias_src_mm(const GemmTaskDescriptor& desc, int n_col_in_tile) {
    const uint64_t global_col = static_cast<uint64_t>(desc.n_tile) * static_cast<uint64_t>(desc.block_n) + static_cast<uint64_t>(n_col_in_tile);
    return desc.bias_base_mm + global_col * ELEM_BYTES;
}

// ============================
// 4) 本地 GM 地址布局（每个组长核心一致）
//    地址随 DIM 自动推导，避免重叠
// ============================
struct LocalLayout {
    uint64_t tmp;
    uint64_t mat;
    uint64_t mat_ping;
    uint64_t mat_pong;
    uint64_t mat_slot2;
    uint64_t mat_slot3;
    uint64_t vec_ping;
    uint64_t vec_pong;
    uint64_t vec_slot2;
    uint64_t vec_slot3;
    uint64_t vec_in;
    uint64_t out;
    uint64_t accum;
};

constexpr uint64_t LOCAL_TMP_OFFSET = 0x0800;
constexpr uint64_t WORKER_START_READY_BASE = 0x1D00;
constexpr uint64_t WORKER_START_RELEASE_CYCLE_OFF = 0x1F00;
constexpr uint64_t WORKER_START_RELEASE_EPOCH_OFF = 0x1F08;
constexpr uint64_t LOCAL_DATA_BASE = 0x2000;
constexpr uint64_t LOCAL_ALIGN = 0x100;
constexpr uint64_t LOCAL_MAT_BYTES_ALIGNED = align_up_constexpr(GEMM_BLOCK_MAT_BYTES, LOCAL_ALIGN);
constexpr uint64_t LOCAL_VEC_BYTES_ALIGNED = align_up_constexpr(GEMM_BLOCK_VEC_PACK_BYTES, LOCAL_ALIGN);
constexpr uint64_t LOCAL_OUT_TILE_BYTES_ALIGNED = align_up_constexpr(GEMM_BLOCK_OUT_TILE_BYTES, LOCAL_ALIGN);
constexpr uint64_t LOCAL_OUT_SCRATCH_BYTES_ALIGNED =
    align_up_constexpr(std::max(OUT_VEC_BYTES, GEMM_BLOCK_OUT_TILE_BYTES), LOCAL_ALIGN);
constexpr uint64_t LOCAL_PARTIAL_TILE_COUNT =
    static_cast<uint64_t>(std::min(A_REUSE_N_TILES, GEMM_N_TILES)) *
    static_cast<uint64_t>(std::min(B_REUSE_M_TILES, GEMM_M_TILES));

constexpr LocalLayout LOCAL_LAYOUT = {
    .tmp = LOCAL_TMP_OFFSET,
    .mat = LOCAL_DATA_BASE,
    .mat_ping = LOCAL_DATA_BASE,
    .mat_pong = LOCAL_DATA_BASE + LOCAL_MAT_BYTES_ALIGNED,
    .mat_slot2 = LOCAL_DATA_BASE + 2 * LOCAL_MAT_BYTES_ALIGNED,
    .mat_slot3 = LOCAL_DATA_BASE + 3 * LOCAL_MAT_BYTES_ALIGNED,
    .vec_ping = LOCAL_DATA_BASE + static_cast<uint64_t>(LOCAL_SLOT_COUNT) * LOCAL_MAT_BYTES_ALIGNED,
    .vec_pong = LOCAL_DATA_BASE + static_cast<uint64_t>(LOCAL_SLOT_COUNT) * LOCAL_MAT_BYTES_ALIGNED + LOCAL_VEC_BYTES_ALIGNED,
    .vec_slot2 = LOCAL_DATA_BASE + static_cast<uint64_t>(LOCAL_SLOT_COUNT) * LOCAL_MAT_BYTES_ALIGNED + 2 * LOCAL_VEC_BYTES_ALIGNED,
    .vec_slot3 = LOCAL_DATA_BASE + static_cast<uint64_t>(LOCAL_SLOT_COUNT) * LOCAL_MAT_BYTES_ALIGNED + 3 * LOCAL_VEC_BYTES_ALIGNED,
    .vec_in = LOCAL_DATA_BASE + static_cast<uint64_t>(LOCAL_SLOT_COUNT) * LOCAL_MAT_BYTES_ALIGNED,
    .out = LOCAL_DATA_BASE + static_cast<uint64_t>(LOCAL_SLOT_COUNT) * LOCAL_MAT_BYTES_ALIGNED + static_cast<uint64_t>(LOCAL_SLOT_COUNT) * LOCAL_VEC_BYTES_ALIGNED,
    .accum = LOCAL_DATA_BASE + static_cast<uint64_t>(LOCAL_SLOT_COUNT) * LOCAL_MAT_BYTES_ALIGNED + static_cast<uint64_t>(LOCAL_SLOT_COUNT) * LOCAL_VEC_BYTES_ALIGNED + LOCAL_OUT_SCRATCH_BYTES_ALIGNED,
};

// ============================
// 5) Mailbox 同步地址布局（避开 DMA flag 尾部区域）
// ============================
struct MailboxLayout {
    uint64_t seq;
    uint64_t ack;
};

constexpr MailboxLayout MBOX_LAYOUT = {
    .seq = 0x10,
    .ack = 0x40,
};

static_assert(TOTAL_GROUPS > 0, "TOTAL_GROUPS must be positive");
static_assert(TILE_M > 0, "TILE_M must be positive");
static_assert(TILE_K > 0, "TILE_K must be positive");
static_assert(GEMM_M > 0 && GEMM_N > 0 && GEMM_K > 0, "GEMM M/N/K must be positive");
static_assert(GEMM_K_TILES > 0, "GEMM_K_TILES must be positive");
static_assert(NUM_MEMORY_NODES >= 2, "NUM_MEMORY_NODES must be >= 2");
static_assert(OS_MEMORY_NODE_INDEX == 0, "OS memory node must stay at node 0");
static_assert(DATA_MEMORY_NODE_COUNT > 0, "DATA_MEMORY_NODE_COUNT must be positive");
static_assert(GEMM_BIAS_STRIDE_MM < MEM_NODE_SIZE, "bias vector region exceeds memory node size");
static_assert(GEMM_DATA_REGION_END <= OFF_GEMM_BIAS_BASE, "GEMM HBM data layout exceeds per-node memory size");
static_assert(
    LOCAL_LAYOUT.accum + LOCAL_PARTIAL_TILE_COUNT * LOCAL_OUT_TILE_BYTES_ALIGNED <
        (GOLEM_GLOBAL_STRIDE_BYTES - 0x40),
    "本地 GM 布局接近/覆盖 DMA flag 尾部区域，请调整 LOCAL_DATA_BASE/对齐策略"
);

static_assert(TOTAL_GEMM_CORES > 0, "TOTAL_GEMM_CORES must be positive");
static_assert(ACTIVE_GEMM_CORES > 0, "ACTIVE_GEMM_CORES must be positive");
static_assert(TOTAL_GEMM_TASKS > 0, "TOTAL_GEMM_TASKS must be positive");
static_assert(LOCAL_SLOT_COUNT >= 4, "LOCAL_SLOT_COUNT must be at least 4 for existing slot aliases");
static_assert(
    WORKER_START_READY_BASE + static_cast<uint64_t>(ACTIVE_GEMM_CORES) * sizeof(uint64_t) <=
        WORKER_START_RELEASE_CYCLE_OFF,
    "worker start barrier ready slots overlap the release record"
);
static_assert(
    WORKER_START_RELEASE_EPOCH_OFF + sizeof(uint64_t) <= LOCAL_DATA_BASE,
    "worker start barrier overlaps local GEMM data"
);
