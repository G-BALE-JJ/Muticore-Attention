#pragma once

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>
#include <sched.h>

#include "gemm_postop_hook.h"
#include "pipeline_config.h"
#include "operators.h"

struct GemmTileRuntimeContext {
    uint64_t local_mat;
    uint64_t local_mat_ping;
    uint64_t local_mat_pong;
    uint64_t local_vec_in;
    uint64_t local_out;
};

struct GemmKernelStats {
    uint64_t dma_mat_cycles;
    uint64_t dma_vec_cycles;
    uint64_t compute_cycles;
    uint64_t overlap_issue_cycles;
    uint64_t overlap_wait_cycles;
};

template <typename T>
struct MatmulTensorBindingsT {
    const T* a;
    const T* b;
    T* c;
    int64_t a_stride0;
    int64_t a_stride1;
    int64_t b_stride0;
    int64_t b_stride1;
    int64_t c_stride0;
    int64_t c_stride1;
};

using MatmulTensorBindings = MatmulTensorBindingsT<int32_t>;
using MatmulTensorBindingsFP32 = MatmulTensorBindingsT<float>;

template <typename T>
static inline T zero_value() {
    return static_cast<T>(0);
}

template <typename T>
static inline T scalar_from_gm_reg(uint64_t gm_addr);

template <>
inline int32_t scalar_from_gm_reg<int32_t>(uint64_t gm_addr) {
    const uint64_t raw = gm2reg(gm_addr);
    return static_cast<int32_t>(raw);
}

template <>
inline float scalar_from_gm_reg<float>(uint64_t gm_addr) {
    const uint64_t raw = gm2reg(gm_addr);
    const uint32_t raw32 = static_cast<uint32_t>(raw & 0xffffffffu);
    float value = 0.0f;
    std::memcpy(&value, &raw32, sizeof(value));
    return value;
}

template <typename T>
static inline const char* dtype_label();

template <>
inline const char* dtype_label<int32_t>() {
    return "int32";
}

template <>
inline const char* dtype_label<float>() {
    return "fp32";
}

static inline uint64_t read_cycle_counter() {
    uint64_t cycles;
    asm volatile (
        "rdcycle %0"
        : "=r"(cycles)
        :
        :
    );
    return cycles;
}

static inline GemmTileRuntimeContext make_gemm_runtime_context(int core_id) {
    return {
        .local_mat = gm_addr(core_id, LOCAL_LAYOUT.mat),
        .local_mat_ping = gm_addr(core_id, LOCAL_LAYOUT.mat_ping),
        .local_mat_pong = gm_addr(core_id, LOCAL_LAYOUT.mat_pong),
        .local_vec_in = gm_addr(core_id, LOCAL_LAYOUT.vec_in),
        .local_out = gm_addr(core_id, LOCAL_LAYOUT.out),
    };
}

template <typename T>
static inline void accumulate_c_tile_column(
    const GemmTaskDescriptor& desc,
    const GemmTileRuntimeContext& rt,
    uint64_t local_mat_addr,
    int n_col,
    std::vector<T>& c_tile,
    std::vector<T>& out_vec
) {
    run_mvm_stage(local_mat_addr, rt.local_vec_in, rt.local_out);
    set_len(VEC_BYTES);
    gm2mm(out_vec.data(), rt.local_out);
    for (int row = 0; row < desc.block_m; ++row) {
        c_tile[static_cast<size_t>(row) * desc.block_n + n_col] += out_vec[row];
    }
}

template <typename T>
static inline void store_c_tile(const GemmTaskDescriptor& desc, const GemmTileRuntimeContext& rt, const std::vector<T>& c_tile) {
    const uint64_t out_tile_bytes = static_cast<uint64_t>(desc.block_m) * static_cast<uint64_t>(desc.block_n) * ELEM_BYTES;
    set_len(out_tile_bytes);
    mm2gm(const_cast<T*>(c_tile.data()), rt.local_out);
    remote_store(rt.local_out, desc.c_base_mm);
}

static inline void maybe_apply_task_side_ops(const GemmTaskDescriptor&, std::vector<int32_t>&) {
}

static inline void maybe_apply_task_side_ops(const GemmTaskDescriptor& desc, std::vector<float>& c_tile) {
    const GolemFp32TilePostOpHook hook = golem_get_fp32_tile_postop_hook();
    if (hook.fn != nullptr) {
        hook.fn(desc, c_tile.data(), c_tile.size(), hook.user);
    }
}

template <typename T>
static inline void apply_optional_bias(
    int core_id,
    const GemmTaskDescriptor& desc,
    const GemmTileRuntimeContext& rt,
    std::vector<T>& c_tile
) {
    if (!BIAS_ENABLED) {
        return;
    }

    for (int n_col = 0; n_col < desc.block_n; ++n_col) {
        const uint64_t bias_src = gemm_desc_bias_src_mm(desc, n_col);
        dma_remote_load_to_gm(core_id, bias_src, rt.local_vec_in, ELEM_BYTES);
        const T bias = scalar_from_gm_reg<T>(rt.local_vec_in);
        for (int row = 0; row < desc.block_m; ++row) {
            c_tile[static_cast<size_t>(row) * desc.block_n + n_col] += bias;
        }
    }
}

template <typename T>
static inline void gemm_tiled_baseline(
    int core_id,
    const GemmTaskDescriptor& desc,
    const GemmTileRuntimeContext& rt,
    GemmKernelStats* stats
) {
    std::vector<T> c_tile(static_cast<size_t>(desc.block_m) * desc.block_n, zero_value<T>());
    std::vector<T> out_vec(desc.block_m, zero_value<T>());

    for (int k = 0; k < desc.k_tiles; ++k) {
        const uint64_t dma_mat_begin = read_cycle_counter();
        dma_remote_load_to_gm(core_id, gemm_desc_mat_src_mm(desc, k), rt.local_mat, MAT_BYTES);
        const uint64_t dma_mat_end = read_cycle_counter();
        stats->dma_mat_cycles += (dma_mat_end - dma_mat_begin);

        for (int n_col = 0; n_col < desc.block_n; ++n_col) {
            const uint64_t dma_vec_begin = read_cycle_counter();
            dma_remote_load_to_gm(core_id, gemm_desc_vec_src_mm(desc, k, n_col), rt.local_vec_in, VEC_BYTES);
            const uint64_t dma_vec_end = read_cycle_counter();
            stats->dma_vec_cycles += (dma_vec_end - dma_vec_begin);

            const uint64_t compute_begin = read_cycle_counter();
            accumulate_c_tile_column<T>(desc, rt, rt.local_mat, n_col, c_tile, out_vec);
            const uint64_t compute_end = read_cycle_counter();
            stats->compute_cycles += (compute_end - compute_begin);
        }
    }

    apply_optional_bias<T>(core_id, desc, rt, c_tile);
    maybe_apply_task_side_ops(desc, c_tile);
    store_c_tile<T>(desc, rt, c_tile);
}

template <typename T>
static inline void gemm_tiled_overlap(
    int core_id,
    const GemmTaskDescriptor& desc,
    const GemmTileRuntimeContext& rt,
    GemmKernelStats* stats
) {
    std::vector<T> c_tile(static_cast<size_t>(desc.block_m) * desc.block_n, zero_value<T>());
    std::vector<T> out_vec(desc.block_m, zero_value<T>());

    const uint64_t dma_mat_begin = read_cycle_counter();
    uint64_t mat_rd_seq = dma_remote_load_issue(core_id, gemm_desc_mat_src_mm(desc, 0), rt.local_mat_ping, MAT_BYTES);
    dma_remote_load_wait(core_id, mat_rd_seq);
    const uint64_t dma_mat_end = read_cycle_counter();
    stats->dma_mat_cycles += (dma_mat_end - dma_mat_begin);

    uint64_t local_mat_active = rt.local_mat_ping;
    uint64_t local_mat_next = rt.local_mat_pong;

    for (int k = 0; k < desc.k_tiles; ++k) {
        const bool has_next_k = (k + 1) < desc.k_tiles;
        uint64_t next_rd_seq = 0;

        for (int n_col = 0; n_col < desc.block_n; ++n_col) {
            const uint64_t dma_vec_begin = read_cycle_counter();
            dma_remote_load_to_gm(core_id, gemm_desc_vec_src_mm(desc, k, n_col), rt.local_vec_in, VEC_BYTES);
            const uint64_t dma_vec_end = read_cycle_counter();
            stats->dma_vec_cycles += (dma_vec_end - dma_vec_begin);

            const bool is_last_col = (n_col + 1) == desc.block_n;
            if (has_next_k && is_last_col) {
                const uint64_t overlap_issue_begin = read_cycle_counter();
                next_rd_seq = dma_remote_load_issue(core_id, gemm_desc_mat_src_mm(desc, k + 1), local_mat_next, MAT_BYTES);
                const uint64_t overlap_issue_end = read_cycle_counter();
                stats->overlap_issue_cycles += (overlap_issue_end - overlap_issue_begin);
            }

            const uint64_t compute_begin = read_cycle_counter();
            accumulate_c_tile_column<T>(desc, rt, local_mat_active, n_col, c_tile, out_vec);
            const uint64_t compute_end = read_cycle_counter();
            stats->compute_cycles += (compute_end - compute_begin);
        }

        if (has_next_k) {
            const uint64_t overlap_wait_begin = read_cycle_counter();
            dma_remote_load_wait(core_id, next_rd_seq);
            const uint64_t overlap_wait_end = read_cycle_counter();
            stats->overlap_wait_cycles += (overlap_wait_end - overlap_wait_begin);
            std::swap(local_mat_active, local_mat_next);
        }
    }

    apply_optional_bias<T>(core_id, desc, rt, c_tile);
    maybe_apply_task_side_ops(desc, c_tile);
    store_c_tile<T>(desc, rt, c_tile);
}

template <typename T>
static inline void gemm_tiled(
    int core_id,
    const GemmTaskDescriptor& desc,
    const GemmTileRuntimeContext& rt,
    GemmKernelStats* stats
) {
    if (DMA_OVERLAP_ENABLED) {
        gemm_tiled_overlap<T>(core_id, desc, rt, stats);
        return;
    }
    gemm_tiled_baseline<T>(core_id, desc, rt, stats);
}

template <typename T>
static inline void load_a_tile_from_tensor(
    const GemmTaskDescriptor& desc,
    const MatmulTensorBindingsT<T>& tensors,
    int k_tile,
    std::vector<T>& mat_tile
) {
    const int m_base = desc.m_tile * desc.block_m;
    const int k_base = k_tile * desc.block_k;
    for (int r = 0; r < desc.block_m; ++r) {
        for (int c = 0; c < desc.block_k; ++c) {
            const int64_t src_idx = static_cast<int64_t>(m_base + r) * tensors.a_stride0 + static_cast<int64_t>(k_base + c) * tensors.a_stride1;
            mat_tile[static_cast<size_t>(r) * desc.block_k + c] = tensors.a[src_idx];
        }
    }
}

template <typename T>
static inline void load_b_col_from_tensor(
    const GemmTaskDescriptor& desc,
    const MatmulTensorBindingsT<T>& tensors,
    int k_tile,
    int n_col,
    std::vector<T>& vec
) {
    const int k_base = k_tile * desc.block_k;
    const int n_base = desc.n_tile * desc.block_n + n_col;
    for (int i = 0; i < desc.block_k; ++i) {
        const int64_t src_idx = static_cast<int64_t>(k_base + i) * tensors.b_stride0 + static_cast<int64_t>(n_base) * tensors.b_stride1;
        vec[i] = tensors.b[src_idx];
    }
}

template <typename T>
static inline void write_c_tile_to_tensor(
    const GemmTaskDescriptor& desc,
    const MatmulTensorBindingsT<T>& tensors,
    const std::vector<T>& c_tile
) {
    const int m_base = desc.m_tile * desc.block_m;
    const int n_base = desc.n_tile * desc.block_n;
    for (int r = 0; r < desc.block_m; ++r) {
        for (int c = 0; c < desc.block_n; ++c) {
            const int64_t dst_idx = static_cast<int64_t>(m_base + r) * tensors.c_stride0 + static_cast<int64_t>(n_base + c) * tensors.c_stride1;
            tensors.c[dst_idx] = c_tile[static_cast<size_t>(r) * desc.block_n + c];
        }
    }
}

template <typename T>
static inline void gemm_tiled_from_tensors(
    int core_id,
    const GemmTaskDescriptor& desc,
    const GemmTileRuntimeContext& rt,
    GemmKernelStats* stats,
    const MatmulTensorBindingsT<T>& tensors
) {
    std::vector<T> c_tile(static_cast<size_t>(desc.block_m) * desc.block_n, zero_value<T>());
    std::vector<T> out_vec(desc.block_m, zero_value<T>());
    std::vector<T> mat_tile(static_cast<size_t>(desc.block_m) * desc.block_k, zero_value<T>());
    std::vector<T> vec(desc.block_k, zero_value<T>());

    const uint64_t mat_bytes = static_cast<uint64_t>(desc.block_m) * static_cast<uint64_t>(desc.block_k) * ELEM_BYTES;
    const uint64_t vec_bytes = static_cast<uint64_t>(desc.block_k) * ELEM_BYTES;

    for (int k = 0; k < desc.k_tiles; ++k) {
        const uint64_t dma_mat_begin = read_cycle_counter();
        load_a_tile_from_tensor<T>(desc, tensors, k, mat_tile);
        set_len(mat_bytes);
        mm2gm(mat_tile.data(), rt.local_mat);
        const uint64_t dma_mat_end = read_cycle_counter();
        stats->dma_mat_cycles += (dma_mat_end - dma_mat_begin);

        for (int n_col = 0; n_col < desc.block_n; ++n_col) {
            const uint64_t dma_vec_begin = read_cycle_counter();
            load_b_col_from_tensor<T>(desc, tensors, k, n_col, vec);
            set_len(vec_bytes);
            mm2gm(vec.data(), rt.local_vec_in);
            const uint64_t dma_vec_end = read_cycle_counter();
            stats->dma_vec_cycles += (dma_vec_end - dma_vec_begin);

            const uint64_t compute_begin = read_cycle_counter();
            accumulate_c_tile_column<T>(desc, rt, rt.local_mat, n_col, c_tile, out_vec);
            const uint64_t compute_end = read_cycle_counter();
            stats->compute_cycles += (compute_end - compute_begin);
        }
    }

    apply_optional_bias<T>(core_id, desc, rt, c_tile);
    maybe_apply_task_side_ops(desc, c_tile);
    write_c_tile_to_tensor<T>(desc, tensors, c_tile);
    store_c_tile<T>(desc, rt, c_tile);
}

static inline bool validate_matmul_call(const MatmulRuntimeConfig& cfg) {
    if (cfg.m <= 0 || cfg.n <= 0 || cfg.k <= 0) {
        return false;
    }
    if (cfg.block_m <= 0 || cfg.block_n <= 0 || cfg.block_k <= 0) {
        return false;
    }
    if ((cfg.m % cfg.block_m) != 0 || (cfg.n % cfg.block_n) != 0 || (cfg.k % cfg.block_k) != 0) {
        return false;
    }
    return true;
}

template <typename T>
static inline void matmul_for_core_t(int core_id, const MatmulRuntimeConfig& cfg, const MatmulTensorBindingsT<T>* tensors) {
    if (!validate_matmul_call(cfg)) {
        printf("[Core %d] [ERROR] invalid matmul config: M/N/K=(%d,%d,%d), block=(%d,%d,%d)\n",
               core_id, cfg.m, cfg.n, cfg.k, cfg.block_m, cfg.block_n, cfg.block_k);
        return;
    }
    if (cfg.block_m != TILE_DIM || cfg.block_k != TILE_DIM) {
        printf("[Core %d] [ERROR] block_M/block_K must equal GOLEM_DIM(%d), got (%d,%d)\n",
               core_id, TILE_DIM, cfg.block_m, cfg.block_k);
        return;
    }
    if (cfg.block_n > TILE_DIM) {
        printf("[Core %d] [ERROR] block_N(%d) exceeds current packed-vector width GOLEM_DIM(%d)\n",
               core_id, cfg.block_n, TILE_DIM);
        return;
    }
    const GemmTileRuntimeContext rt = make_gemm_runtime_context(core_id);
    const int total_tasks = gemm_total_tasks(cfg);
    if (core_id >= total_tasks) {
        printf("[Core %d] GEMM 核心空闲（total_tasks=%d, active_cores=%d）。\n",
               core_id, total_tasks, ACTIVE_GEMM_CORES);
        return;
    }

    const uint64_t total_begin = read_cycle_counter();
    if (DMA_STAGGER_CYCLES > 0) {
        delay_cycles(DMA_STAGGER_CYCLES * static_cast<uint32_t>(core_id));
    }

    GemmKernelStats stats = {};
    int first_task = -1;
    int last_task = -1;
    int tasks_done = 0;

    for (int task_id = core_id; task_id < total_tasks; task_id += ACTIVE_GEMM_CORES) {
        if (first_task < 0) {
            first_task = task_id;
        }
        last_task = task_id;
        tasks_done++;

        const GemmTaskDescriptor desc = gemm_descriptor_for_task(core_id, task_id, cfg);
        if (tensors != nullptr && tensors->a != nullptr && tensors->b != nullptr && tensors->c != nullptr) {
            gemm_tiled_from_tensors<T>(core_id, desc, rt, &stats, *tensors);
        } else {
            gemm_tiled<T>(core_id, desc, rt, &stats);
        }
    }

    const uint64_t total_end = read_cycle_counter();
    const uint64_t dma_total_cycles = stats.dma_mat_cycles + stats.dma_vec_cycles;
    const uint64_t total_cycles = total_end - total_begin;
    printf("[Core %d] [%s] LATENCY(cycles): dma_mat=%" PRIu64
           " dma_vec=%" PRIu64
           " dma_total=%" PRIu64
           " compute=%" PRIu64
           " total=%" PRIu64 "\n",
            core_id,
            dtype_label<T>(),
            stats.dma_mat_cycles,
            stats.dma_vec_cycles,
            dma_total_cycles,
           stats.compute_cycles,
           total_cycles);

    if (DMA_OVERLAP_ENABLED) {
        printf("[Core %d] [%s] OVERLAP(cycles): issue=%" PRIu64 " wait=%" PRIu64 "\n",
               core_id,
               dtype_label<T>(),
               stats.overlap_issue_cycles,
               stats.overlap_wait_cycles);
        printf("[Core %d] [%s] GEMM/MM summary(overlap=1): tasks_done=%d first_task=%d last_task=%d task_stride=%d total_tasks=%d block=(%d,%d,%d) k_tiles=%d mat_ping=0x%" PRIx64 ", mat_pong=0x%" PRIx64 ", vec=0x%" PRIx64 ", out=0x%" PRIx64 "\n",
               core_id, dtype_label<T>(), tasks_done, first_task, last_task, ACTIVE_GEMM_CORES, total_tasks,
               cfg.block_m, cfg.block_n, cfg.block_k, gemm_k_tiles(cfg),
               rt.local_mat_ping, rt.local_mat_pong, rt.local_vec_in, rt.local_out);
        return;
    }

    printf("[Core %d] [%s] GEMM/MM summary: tasks_done=%d first_task=%d last_task=%d task_stride=%d total_tasks=%d block=(%d,%d,%d) k_tiles=%d mat=0x%" PRIx64 ", vec=0x%" PRIx64 ", out=0x%" PRIx64 "\n",
           core_id, dtype_label<T>(), tasks_done, first_task, last_task, ACTIVE_GEMM_CORES, total_tasks,
           cfg.block_m, cfg.block_n, cfg.block_k, gemm_k_tiles(cfg),
           rt.local_mat, rt.local_vec_in, rt.local_out);
}

static inline void matmul_for_core(int core_id, const MatmulRuntimeConfig& cfg, const MatmulTensorBindings* tensors) {
    matmul_for_core_t<int32_t>(core_id, cfg, tensors);
}

inline void matmul(int M, int N, int K, int block_M, int block_N, int block_K) {
    const int core_id = sched_getcpu();
    if (core_id < 0 || core_id >= TOTAL_CORES) {
        printf("[ERROR] invalid runtime core id=%d, TOTAL_CORES=%d\n", core_id, TOTAL_CORES);
        return;
    }
    if (core_id >= ACTIVE_GEMM_CORES) {
        printf("[Core %d] 非 GEMM 核心（当前 GEMM 使用前 %d 核，任务总数=%d）。\n", core_id, ACTIVE_GEMM_CORES, TOTAL_GEMM_TASKS);
        return;
    }
    const MatmulRuntimeConfig cfg = {
        .m = M,
        .n = N,
        .k = K,
        .block_m = block_M,
        .block_n = block_N,
        .block_k = block_K,
    };
    matmul_for_core(core_id, cfg, nullptr);
}

inline void matmul_with_tensors(
    int M,
    int N,
    int K,
    int block_M,
    int block_N,
    int block_K,
    const MatmulTensorBindings& tensors
) {
    const int core_id = sched_getcpu();
    if (core_id < 0 || core_id >= TOTAL_CORES) {
        printf("[ERROR] invalid runtime core id=%d, TOTAL_CORES=%d\n", core_id, TOTAL_CORES);
        return;
    }
    if (core_id >= ACTIVE_GEMM_CORES) {
        printf("[Core %d] 非 GEMM 核心（当前 GEMM 使用前 %d 核，任务总数=%d）。\n", core_id, ACTIVE_GEMM_CORES, TOTAL_GEMM_TASKS);
        return;
    }
    const MatmulRuntimeConfig cfg = {
        .m = M,
        .n = N,
        .k = K,
        .block_m = block_M,
        .block_n = block_N,
        .block_k = block_K,
    };
    matmul_for_core_t<int32_t>(core_id, cfg, &tensors);
}

inline void matmul_fp32(int M, int N, int K, int block_M, int block_N, int block_K) {
    const int core_id = sched_getcpu();
    if (core_id < 0 || core_id >= TOTAL_CORES) {
        printf("[ERROR] invalid runtime core id=%d, TOTAL_CORES=%d\n", core_id, TOTAL_CORES);
        return;
    }
    if (core_id >= ACTIVE_GEMM_CORES) {
        printf("[Core %d] 非 GEMM 核心（当前 GEMM 使用前 %d 核，任务总数=%d）。\n", core_id, ACTIVE_GEMM_CORES, TOTAL_GEMM_TASKS);
        return;
    }
    const MatmulRuntimeConfig cfg = {
        .m = M,
        .n = N,
        .k = K,
        .block_m = block_M,
        .block_n = block_N,
        .block_k = block_K,
    };
    matmul_for_core_t<float>(core_id, cfg, nullptr);
}

inline void matmul_with_tensors_fp32(
    int M,
    int N,
    int K,
    int block_M,
    int block_N,
    int block_K,
    const MatmulTensorBindingsFP32& tensors
) {
    const int core_id = sched_getcpu();
    if (core_id < 0 || core_id >= TOTAL_CORES) {
        printf("[ERROR] invalid runtime core id=%d, TOTAL_CORES=%d\n", core_id, TOTAL_CORES);
        return;
    }
    if (core_id >= ACTIVE_GEMM_CORES) {
        printf("[Core %d] 非 GEMM 核心（当前 GEMM 使用前 %d 核，任务总数=%d）。\n", core_id, ACTIVE_GEMM_CORES, TOTAL_GEMM_TASKS);
        return;
    }
    const MatmulRuntimeConfig cfg = {
        .m = M,
        .n = N,
        .k = K,
        .block_m = block_M,
        .block_n = block_N,
        .block_k = block_K,
    };
    matmul_for_core_t<float>(core_id, cfg, &tensors);
}
