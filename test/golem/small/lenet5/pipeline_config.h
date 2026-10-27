#pragma once

#include <cstdint>

#ifndef GOLEM_DIM
#define GOLEM_DIM 16
#endif

#ifndef GOLEM_TOTAL_GEMM_CORES
#define GOLEM_TOTAL_GEMM_CORES 16
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

#ifndef GOLEM_GEMM_M
#define GOLEM_GEMM_M GOLEM_DIM
#endif

#ifndef GOLEM_GEMM_N
#define GOLEM_GEMM_N GOLEM_DIM
#endif

#ifndef GOLEM_GEMM_K
#define GOLEM_GEMM_K GOLEM_DIM
#endif

#ifndef GOLEM_BIAS_ENABLE
#define GOLEM_BIAS_ENABLE 0
#endif

#ifndef GOLEM_BIAS_VALUE
#define GOLEM_BIAS_VALUE 0
#endif

#ifndef GOLEM_NUM_MEMORY_NODES
#define GOLEM_NUM_MEMORY_NODES 4
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
constexpr uint32_t DMA_STAGGER_CYCLES = static_cast<uint32_t>(GOLEM_DMA_STAGGER_CYCLES);
constexpr bool DMA_OVERLAP_ENABLED = (GOLEM_DMA_OVERLAP != 0);

constexpr int TILE_DIM = GOLEM_DIM;
constexpr int GEMM_M = GOLEM_GEMM_M;
constexpr int GEMM_N = GOLEM_GEMM_N;
constexpr int GEMM_K = GOLEM_GEMM_K;
constexpr bool BIAS_ENABLED = (GOLEM_BIAS_ENABLE != 0);
constexpr int32_t BIAS_VALUE = static_cast<int32_t>(GOLEM_BIAS_VALUE);
constexpr int GEMM_M_TILES = (GEMM_M >= TILE_DIM) ? (GEMM_M / TILE_DIM) : 1;
constexpr int GEMM_N_TILES = (GEMM_N >= TILE_DIM) ? (GEMM_N / TILE_DIM) : 1;
constexpr int GEMM_K_TILES = (GEMM_K >= TILE_DIM) ? (GEMM_K / TILE_DIM) : 1;
constexpr int TOTAL_GEMM_TASKS = GEMM_M_TILES * GEMM_N_TILES;
constexpr int ACTIVE_GEMM_CORES = TOTAL_GEMM_CORES;

// ============================
// 2) 计算规模参数
//    仅需修改 DIM
// ============================
constexpr uint64_t DIM = static_cast<uint64_t>(GOLEM_DIM);
constexpr uint64_t ELEM_BYTES = sizeof(int32_t);
constexpr uint64_t MAT_ELEMS = DIM * DIM;
constexpr uint64_t VEC_ELEMS = DIM;
constexpr uint64_t MAT_BYTES = MAT_ELEMS * ELEM_BYTES;
constexpr uint64_t VEC_BYTES = VEC_ELEMS * ELEM_BYTES;

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

constexpr uint64_t node_base_addr(int node_idx) {
    return static_cast<uint64_t>(node_idx) * MEM_NODE_SIZE;
}

constexpr uint64_t MM_ALIGN = 0x100;
constexpr uint64_t MM_MAT_STRIDE = align_up_constexpr(MAT_BYTES, MM_ALIGN);
constexpr uint64_t stage_mat_src_mm(int stage_id) {
    return IDENTITY_BASE + static_cast<uint64_t>(stage_id) * MM_MAT_STRIDE;
}
constexpr uint64_t INIT_VEC_SRC_MM = IDENTITY_BASE + TOTAL_GROUPS * MM_MAT_STRIDE;

// GEMM模式：按 tile-task 分配。
// task_id = m_tile * GEMM_N_TILES + n_tile
// 每个 task 对应 K 方向 GEMM_K_TILES 个 A/vec 源片段。
constexpr uint64_t GEMM_VEC_STRIDE_MM = align_up_constexpr(VEC_BYTES, MM_ALIGN);
constexpr uint64_t OFF_GEMM_MAT_BASE = 0x0;
constexpr uint64_t OFF_GEMM_VEC_BASE = OFF_GEMM_MAT_BASE + static_cast<uint64_t>(TOTAL_GEMM_TASKS * GEMM_K_TILES) * MM_MAT_STRIDE;
constexpr uint64_t GEMM_OUT_STRIDE_MM = align_up_constexpr(MAT_BYTES, MM_ALIGN);
constexpr uint64_t OFF_GEMM_OUT_BASE = OFF_GEMM_VEC_BASE + static_cast<uint64_t>(TOTAL_GEMM_TASKS * GEMM_K_TILES * TILE_DIM) * GEMM_VEC_STRIDE_MM;
constexpr uint64_t GEMM_BIAS_STRIDE_MM = align_up_constexpr(static_cast<uint64_t>(GEMM_N) * ELEM_BYTES, MM_ALIGN);
constexpr uint64_t OFF_GEMM_BIAS_BASE = MEM_NODE_SIZE - GEMM_BIAS_STRIDE_MM;

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
        .block_m = TILE_DIM,
        .block_n = TILE_DIM,
        .block_k = TILE_DIM,
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

inline uint64_t gemm_off_vec_base(const MatmulRuntimeConfig& cfg) {
    return OFF_GEMM_MAT_BASE + static_cast<uint64_t>(gemm_total_tasks(cfg) * gemm_k_tiles(cfg)) * MM_MAT_STRIDE;
}

inline uint64_t gemm_out_stride_mm(const MatmulRuntimeConfig& cfg) {
    const uint64_t out_tile_bytes = static_cast<uint64_t>(cfg.block_m) * static_cast<uint64_t>(cfg.block_n) * ELEM_BYTES;
    return align_up_constexpr(out_tile_bytes, MM_ALIGN);
}

inline uint64_t gemm_off_out_base(const MatmulRuntimeConfig& cfg) {
    return gemm_off_vec_base(cfg) + static_cast<uint64_t>(gemm_total_tasks(cfg) * gemm_k_tiles(cfg) * cfg.block_n) * GEMM_VEC_STRIDE_MM;
}

inline int gemm_task_id_for_core(int core_id, const MatmulRuntimeConfig& cfg) {
    const int total_tasks = gemm_total_tasks(cfg);
    if (total_tasks <= 0) {
        return 0;
    }
    return core_id % total_tasks;
}

inline int gemm_m_tile_of_task(int task_id, const MatmulRuntimeConfig& cfg) {
    return task_id / gemm_n_tiles(cfg);
}

inline int gemm_n_tile_of_task(int task_id, const MatmulRuntimeConfig& cfg) {
    return task_id % gemm_n_tiles(cfg);
}

inline int gemm_data_node_for_task(int task_id, const MatmulRuntimeConfig& cfg) {
    const int total_tasks = gemm_total_tasks(cfg);
    if (total_tasks <= 0) {
        return 1;
    }
    return 1 + ((task_id * DATA_MEMORY_NODE_COUNT) / total_tasks);
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

inline GemmTaskDescriptor gemm_task_desc_for_task(int core_id, int task_id, const MatmulRuntimeConfig& cfg) {
    const int node_idx = gemm_data_node_for_task(task_id, cfg);
    const int task_slot = gemm_local_task_slot(task_id, cfg);
    const int k_tiles = gemm_k_tiles(cfg);
    const uint64_t task_slot_u64 = static_cast<uint64_t>(task_slot);
    const uint64_t off_vec_base = gemm_off_vec_base(cfg);
    const uint64_t off_out_base = gemm_off_out_base(cfg);
    const uint64_t out_stride = gemm_out_stride_mm(cfg);
    return {
        .core_id = core_id,
        .task_id = task_id,
        .m_tile = gemm_m_tile_of_task(task_id, cfg),
        .n_tile = gemm_n_tile_of_task(task_id, cfg),
        .data_node_idx = node_idx,
        .task_slot_in_node = task_slot,
        .m = cfg.m,
        .n = cfg.n,
        .k = cfg.k,
        .block_m = cfg.block_m,
        .block_n = cfg.block_n,
        .block_k = cfg.block_k,
        .k_tiles = k_tiles,
        .a_base_mm = node_base_addr(node_idx) + OFF_GEMM_MAT_BASE + task_slot_u64 * static_cast<uint64_t>(k_tiles) * MM_MAT_STRIDE,
        .b_pack_base_mm = node_base_addr(node_idx) + off_vec_base + task_slot_u64 * static_cast<uint64_t>(k_tiles * cfg.block_n) * GEMM_VEC_STRIDE_MM,
        .c_base_mm = node_base_addr(node_idx) + off_out_base + task_slot_u64 * out_stride,
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
    uint64_t vec_in;
    uint64_t out;
};

constexpr uint64_t LOCAL_TMP_OFFSET = 0x0800;
constexpr uint64_t LOCAL_DATA_BASE = 0x2000;
constexpr uint64_t LOCAL_ALIGN = 0x100;
constexpr uint64_t LOCAL_MAT_BYTES_ALIGNED = align_up_constexpr(MAT_BYTES, LOCAL_ALIGN);
constexpr uint64_t LOCAL_VEC_BYTES_ALIGNED = align_up_constexpr(VEC_BYTES, LOCAL_ALIGN);
constexpr uint64_t LOCAL_OUT_TILE_BYTES_ALIGNED = align_up_constexpr(MAT_BYTES, LOCAL_ALIGN);

constexpr LocalLayout LOCAL_LAYOUT = {
    .tmp = LOCAL_TMP_OFFSET,
    .mat = LOCAL_DATA_BASE,
    .mat_ping = LOCAL_DATA_BASE,
    .mat_pong = LOCAL_DATA_BASE + LOCAL_MAT_BYTES_ALIGNED,
    .vec_in = LOCAL_DATA_BASE + 2 * LOCAL_MAT_BYTES_ALIGNED,
    .out = LOCAL_DATA_BASE + 2 * LOCAL_MAT_BYTES_ALIGNED + LOCAL_VEC_BYTES_ALIGNED,
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
static_assert(TILE_DIM > 0, "TILE_DIM must be positive");
static_assert(GEMM_M > 0 && GEMM_N > 0 && GEMM_K > 0, "GEMM M/N/K must be positive");
static_assert(GEMM_K_TILES > 0, "GEMM_K_TILES must be positive");
static_assert(NUM_MEMORY_NODES >= 2, "NUM_MEMORY_NODES must be >= 2");
static_assert(OS_MEMORY_NODE_INDEX == 0, "OS memory node must stay at node 0");
static_assert(DATA_MEMORY_NODE_COUNT > 0, "DATA_MEMORY_NODE_COUNT must be positive");
static_assert(GEMM_BIAS_STRIDE_MM < MEM_NODE_SIZE, "bias vector region exceeds memory node size");
static_assert(
    LOCAL_LAYOUT.out + LOCAL_OUT_TILE_BYTES_ALIGNED < (GOLEM_GLOBAL_STRIDE_BYTES - 0x20),
    "本地 GM 布局接近/覆盖 DMA flag 尾部区域，请调整 LOCAL_DATA_BASE/对齐策略"
);

static_assert(TOTAL_GEMM_CORES > 0, "TOTAL_GEMM_CORES must be positive");
static_assert(TOTAL_GEMM_TASKS > 0, "TOTAL_GEMM_TASKS must be positive");
