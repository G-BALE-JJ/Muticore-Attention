#pragma once

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <vector>
#include <sched.h>

#include "../../../fp16.h"

#include "pipeline_config.h"
#include "operators.h"

struct GemmTileRuntimeContext {
    uint64_t local_mat;
    uint64_t local_mat_ping;
    uint64_t local_mat_pong;
    uint64_t local_mat_slot2;
    uint64_t local_mat_slot3;
    uint64_t local_vec_ping;
    uint64_t local_vec_pong;
    uint64_t local_vec_slot2;
    uint64_t local_vec_slot3;
    uint64_t local_vec_in;
    uint64_t local_out;
    uint64_t local_accum;
};

struct WorkerTaskListHeaderRuntime {
    uint32_t worker_slot;
    uint32_t task_count;
    uint32_t active_worker_cores;
    uint32_t total_groups;
    uint32_t data_memory_node_count;
    uint64_t mem_node_size;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t hw_input_size;
    uint32_t hw_output_size;
    uint32_t block_m;
    uint32_t block_n;
    uint32_t block_k;
    uint32_t elem_bytes;
    uint64_t mat_stride_bytes;
    uint64_t vec_stride_bytes;
    uint64_t off_gemm_mat_base;
    uint64_t off_gemm_vec_base;
    uint64_t off_gemm_out_base;
    uint64_t local_mat_ping_gm_addr;
    uint64_t local_mat_pong_gm_addr;
    uint64_t local_mat_slot2_gm_addr;
    uint64_t local_mat_slot3_gm_addr;
    uint64_t local_vec_ping_gm_addr;
    uint64_t local_vec_pong_gm_addr;
    uint64_t local_vec_slot2_gm_addr;
    uint64_t local_vec_slot3_gm_addr;
    uint64_t local_mat_slot_stride_bytes;
    uint64_t local_vec_slot_stride_bytes;
    uint32_t local_slot_count;
    uint64_t local_accum_gm_addr;
    uint64_t local_out_gm_addr;
    uint64_t finished_mailbox_addr;
    uint32_t a_reuse_n_tiles;
    uint32_t n_group_count;
    uint32_t b_reuse_m_tiles;
    uint32_t m_group_count;
    uint32_t data_node_map_mode;
    uint64_t descriptor_start_cycle;
};

constexpr uint64_t WCP_DESC_GM_ADDR = LOCAL_TMP_OFFSET;
static_assert((sizeof(WorkerTaskListHeaderRuntime) % sizeof(uint64_t)) == 0,
              "WorkerTaskListHeaderRuntime must be written as whole u64 words");

static inline void write_worker_task_list_header_at(int core_id, uint64_t base, const WorkerTaskListHeaderRuntime& desc) {
    (void)core_id;
    memory_barrier();
    set_len(sizeof(WorkerTaskListHeaderRuntime));
    mm2gm(&desc, base);
}

static inline void write_worker_task_list_header(int core_id, const WorkerTaskListHeaderRuntime& desc) {
    write_worker_task_list_header_at(core_id, gm_addr(core_id, WCP_DESC_GM_ADDR), desc);
}

static inline uint64_t read_cycle_counter();

static inline uint64_t synchronize_worker_descriptor_start(
    int core_id,
    int worker_slot,
    int participant_count) {
    if (!WORKER_START_BARRIER_ENABLED || participant_count <= 1) {
        return read_cycle_counter();
    }

    static uint64_t local_epoch = 0;
    const uint64_t epoch = ++local_epoch;
    const int leader_core = gemm_worker_core_for_slot(0);
    const uint64_t ready_addr =
        gm_addr(leader_core, WORKER_START_READY_BASE + static_cast<uint64_t>(worker_slot) * sizeof(uint64_t));
    const uint64_t entry_cycle = read_cycle_counter();

    if (core_id == leader_core) {
        reg2gm(epoch, ready_addr);
    } else {
        remote_write_u64(core_id, epoch, ready_addr);
    }

    const uint64_t local_release_cycle = gm_addr(core_id, WORKER_START_RELEASE_CYCLE_OFF);
    const uint64_t local_release_epoch = gm_addr(core_id, WORKER_START_RELEASE_EPOCH_OFF);
    if (core_id == leader_core) {
        for (int slot = 0; slot < participant_count; ++slot) {
            adaptive_wait_eq(
                gm_addr(leader_core, WORKER_START_READY_BASE + static_cast<uint64_t>(slot) * sizeof(uint64_t)),
                epoch);
        }

        const uint64_t start_cycle = read_cycle_counter() + WORKER_START_GUARD_CYCLES;
        for (int slot = 0; slot < participant_count; ++slot) {
            const int target_core = gemm_worker_core_for_slot(slot);
            const uint64_t target_release = gm_addr(target_core, WORKER_START_RELEASE_CYCLE_OFF);
            if (target_core == core_id) {
                reg2gm(start_cycle, local_release_cycle);
                reg2gm(epoch, local_release_epoch);
                continue;
            }

            const uint64_t src = tmp_addr(core_id);
            reg2gm(start_cycle, src);
            reg2gm(epoch, src + sizeof(uint64_t));
            set_len(2 * sizeof(uint64_t));
            remote_store(src, target_release);
        }
    } else {
        adaptive_wait_eq(local_release_epoch, epoch);
    }

    uint64_t start_cycle = gm2reg(local_release_cycle);
    while (start_cycle <= entry_cycle) {
        delay_cycles(8);
        start_cycle = gm2reg(local_release_cycle);
    }
    while (read_cycle_counter() < start_cycle) {
        delay_cycles(8);
    }
    return start_cycle;
}

struct GemmKernelStats {
    // Buckets are intended to be mutually exclusive:
    // - dma_issue_cycles / dma_wait_cycles: non-overlap DMA path
    // - overlap_issue_cycles / overlap_wait_cycles: overlap-only path
    // - sched_protocol_cycles: control protocol overhead not charged to DMA issue/wait
    uint64_t dma_issue_cycles;
    uint64_t dma_wait_cycles;
    uint64_t compute_cycles;
    uint64_t compute_submit_cycles;
    uint64_t compute_wait_cycles;
    uint64_t sched_protocol_cycles;
    uint64_t c_store_cycles;
    uint64_t group_wait_cycles;
    uint64_t poll_iters;
    uint64_t overlap_issue_cycles;
    uint64_t overlap_wait_cycles;
    uint64_t issue_block_submitq_cycles;
    uint64_t issue_write_submitq_cycles;
    uint64_t ov_issue_block_submitq_cycles;
    uint64_t ov_issue_write_submitq_cycles;
    uint64_t task_desc_overhead_cycles;
    uint64_t nloop_overhead_cycles;
    uint64_t submit_pack_cycles;
    uint64_t finish_publish_cycles;
    uint64_t exec_window_begin_cycles;
    uint64_t exec_window_end_cycles;
    uint64_t exec_window_started;
    uint64_t descriptor_barrier_cycle;
    uint64_t descriptor_mm2gm_start_cycle;
    uint64_t descriptor_mm2gm_end_cycle;
};

static inline void mark_exec_window_begin(GemmKernelStats* stats, uint64_t cycle) {
    if (stats == nullptr || stats->exec_window_started != 0) {
        return;
    }
    stats->exec_window_started = 1;
    stats->exec_window_begin_cycles = cycle;
    stats->exec_window_end_cycles = cycle;
}

static inline void mark_exec_window_end(GemmKernelStats* stats, uint64_t cycle) {
    if (stats == nullptr || stats->exec_window_started == 0) {
        return;
    }
    if (cycle > stats->exec_window_end_cycles) {
        stats->exec_window_end_cycles = cycle;
    }
}

static inline uint64_t exec_window_cycles(const GemmKernelStats& stats) {
    if (stats.exec_window_started == 0 || stats.exec_window_end_cycles < stats.exec_window_begin_cycles) {
        return 0;
    }
    return stats.exec_window_end_cycles - stats.exec_window_begin_cycles;
}

static inline void account_nloop_overhead(
    GemmKernelStats* stats,
    uint64_t loop_begin,
    uint64_t loop_end,
    uint64_t compute_before
) {
    if (stats == nullptr || loop_end <= loop_begin) {
        return;
    }
    const uint64_t loop_total = loop_end - loop_begin;
    const uint64_t active_delta = stats->compute_cycles - compute_before;
    if (loop_total > active_delta) {
        stats->nloop_overhead_cycles += (loop_total - active_delta);
    }
}

template <typename T>
struct MatmulTensorBindingsT {
    const T* a;
    const T* b;
    T* c;
    int64_t a_stride0;
    int64_t a_stride1;
    int64_t b_stride0;
    int64_t b_stride1;
    bool transpose_b;
    int64_t c_stride0;
    int64_t c_stride1;
};

using MatmulTensorBindings = MatmulTensorBindingsT<int32_t>;
using MatmulTensorBindingsFP32 = MatmulTensorBindingsT<float>;
using MatmulTensorBindingsFP16 = MatmulTensorBindingsT<SST::Golem::GolemFp16>;

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

template <>
inline SST::Golem::GolemFp16 scalar_from_gm_reg<SST::Golem::GolemFp16>(uint64_t gm_addr) {
    const uint64_t raw = gm2reg(gm_addr);
    return SST::Golem::GolemFp16(
        SST::Golem::golem_fp16_to_float(static_cast<uint16_t>(raw & 0xffffu)));
}

template <typename T>
static inline const char* dtype_label();

inline bool runtime_silent_enabled();

inline bool stage_progress_enabled() {
    if (runtime_silent_enabled()) {
        return false;
    }
    static int cached = -1;
    if (cached < 0) {
        const char* env = std::getenv("GOLEM_STAGE_PROGRESS");
        cached = (env == nullptr || env[0] == '\0' || env[0] != '0') ? 1 : 0;
    }
    return cached != 0;
}

inline bool runtime_silent_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* env = std::getenv("GOLEM_RUNTIME_SILENT");
        if (env == nullptr || env[0] == '\0') {
            env = std::getenv("GOLEM_SILENT");
        }
        cached = (env != nullptr && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }
    return cached != 0;
}

inline bool runtime_info_enabled() {
    return !runtime_silent_enabled();
}

template <>
inline const char* dtype_label<int32_t>() {
    return "int32";
}

template <>
inline const char* dtype_label<float>() {
    return "fp32";
}

template <>
inline const char* dtype_label<SST::Golem::GolemFp16>() {
    return "fp16";
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
        .local_mat_slot2 = gm_addr(core_id, LOCAL_LAYOUT.mat_slot2),
        .local_mat_slot3 = gm_addr(core_id, LOCAL_LAYOUT.mat_slot3),
        .local_vec_ping = gm_addr(core_id, LOCAL_LAYOUT.vec_ping),
        .local_vec_pong = gm_addr(core_id, LOCAL_LAYOUT.vec_pong),
        .local_vec_slot2 = gm_addr(core_id, LOCAL_LAYOUT.vec_slot2),
        .local_vec_slot3 = gm_addr(core_id, LOCAL_LAYOUT.vec_slot3),
        .local_vec_in = gm_addr(core_id, LOCAL_LAYOUT.vec_in),
        .local_out = gm_addr(core_id, LOCAL_LAYOUT.out),
        .local_accum = gm_addr(core_id, LOCAL_LAYOUT.accum),
    };
}

static inline uint64_t mat_buf_addr(const GemmTileRuntimeContext& rt, int buf_idx) {
    return (buf_idx & 1) ? rt.local_mat_pong : rt.local_mat_ping;
}

static inline uint64_t vec_buf_base_addr(const GemmTileRuntimeContext& rt, int buf_idx) {
    return (buf_idx & 1) ? rt.local_vec_pong : rt.local_vec_ping;
}

static inline uint64_t accum_col_addr(const GemmTaskDescriptor& desc, const GemmTileRuntimeContext& rt, int n_col) {
    return rt.local_accum + static_cast<uint64_t>(n_col) * static_cast<uint64_t>(desc.block_m) * ELEM_BYTES;
}

static inline uint64_t vec_col_addr(const GemmTileRuntimeContext& rt, int n_col) {
    return rt.local_vec_in + static_cast<uint64_t>(n_col) * VEC_BYTES;
}

static inline uint64_t vec_col_addr(const GemmTileRuntimeContext& rt, int buf_idx, int n_col) {
    return vec_buf_base_addr(rt, buf_idx) + static_cast<uint64_t>(n_col) * VEC_BYTES;
}

static inline uint64_t vec_block_addr(const GemmTileRuntimeContext& rt, int block_idx, int block_n) {
    return rt.local_vec_in + static_cast<uint64_t>(block_idx) * static_cast<uint64_t>(block_n) * VEC_BYTES;
}

static inline uint64_t vec_col_addr_from_block(uint64_t vec_block_base, int n_col) {
    return vec_block_base + static_cast<uint64_t>(n_col) * VEC_BYTES;
}

template <typename T>
static inline void zero_local_accum_tile(const GemmTaskDescriptor& desc, const GemmTileRuntimeContext& rt) {
    const uint64_t elems = static_cast<uint64_t>(desc.block_m) * static_cast<uint64_t>(desc.block_n);
    const uint64_t bytes = elems * ELEM_BYTES;
    std::vector<T> zeros(static_cast<size_t>(elems), zero_value<T>());
    set_len(bytes);
    mm2gm(zeros.data(), rt.local_accum);
}

template <typename T>
static inline void run_mvm_compute_only(uint64_t mat_gm, uint64_t vec_gm, uint64_t array_id = 0) {
    inputmatrixload(mat_gm, array_id);
    inputvectorload(vec_gm, array_id);
    mvm_compute(array_id);
}

template <typename T>
static inline void run_mvm_load_matrix_only(uint64_t mat_gm, uint64_t array_id = 0) {
    inputmatrixload(mat_gm, array_id);
}

template <typename T>
static inline void run_mvm_load_matrix_only_async(uint64_t mat_gm, uint64_t array_id = 0) {
    inputmatrixload_async(mat_gm, array_id);
}

template <typename T>
static inline void run_mvm_load_matrix_batch_async(uint64_t mat_gm, uint64_t count) {
    tile_gm2imat_broadcast(mat_gm, count);
}

template <typename T>
static inline void run_mvm_load_vector_only(uint64_t vec_gm, uint64_t array_id = 0) {
    inputvectorload(vec_gm, array_id);
}

template <typename T>
static inline void run_mvm_load_vector_only_async(uint64_t vec_gm, uint64_t array_id = 0) {
    inputvectorload_async(vec_gm, array_id);
}

template <typename T>
static inline void run_mvm_load_vector_batch_async(uint64_t vec_base_gm, uint64_t count) {
    tile_gm2ivec_batch(vec_base_gm, count);
}

template <typename T>
static inline void run_mvm_compute_only_no_load(uint64_t array_id = 0) {
    mvm_compute(array_id);
}

template <typename T>
static inline void run_mvm_compute_only_no_load_async(uint64_t array_id = 0) {
    mvm_compute_async(array_id);
}

template <typename T>
static inline void run_mvm_compute_batch_async(uint64_t start_array, uint64_t count) {
    tile_mvm_batch_async(start_array, count);
}

template <typename T>
static inline void run_mvm_compute_batch_wait(uint64_t start_array, uint64_t count) {
    tile_mvm_batch_wait(start_array, count);
}

static inline void run_worker_window_start(uint64_t desc_gm_addr) {
    wcp_start(desc_gm_addr);
}

static inline void run_worker_window_wait() {
    wcp_wait();
}

template <typename T>
static inline void accumulate_c_tile_column_host(
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

static inline void store_c_tile_from_gm(const GemmTaskDescriptor& desc, const GemmTileRuntimeContext& rt) {
    const uint64_t out_tile_bytes = static_cast<uint64_t>(desc.block_m) * static_cast<uint64_t>(desc.block_n) * ELEM_BYTES;
    set_len(out_tile_bytes);
    remote_store(rt.local_accum, desc.c_base_mm);
}

static inline void profiled_store_c_tile_from_gm(
    const GemmTaskDescriptor& desc,
    const GemmTileRuntimeContext& rt,
    uint64_t* c_store_cycles,
    GemmKernelStats* stats
) {
    const uint64_t begin = read_cycle_counter();
    store_c_tile_from_gm(desc, rt);
    const uint64_t end = read_cycle_counter();
    if (c_store_cycles != nullptr) {
        *c_store_cycles += (end - begin);
    }
    mark_exec_window_end(stats, end);
}

template <typename T>
static inline void apply_optional_bias_gm_fast_path(int core_id) {
    if (BIAS_ENABLED) {
        if (runtime_info_enabled()) {
            printf("[Core %d] [WARN] bias is unsupported in GM-accum fast path; results exclude bias\n", core_id);
        }
    }
}

template <typename T>
static inline void store_c_tile(const GemmTaskDescriptor& desc, const GemmTileRuntimeContext& rt, const std::vector<T>& c_tile) {
    const uint64_t out_tile_bytes = static_cast<uint64_t>(desc.block_m) * static_cast<uint64_t>(desc.block_n) * ELEM_BYTES;
    set_len(out_tile_bytes);
    mm2gm(const_cast<T*>(c_tile.data()), rt.local_out);
    remote_store(rt.local_out, desc.c_base_mm);
}

static inline void profiled_dma_remote_load_to_gm(
    int core_id,
    uint64_t mm_src_addr,
    uint64_t gm_dst_addr,
    uint64_t bytes,
    uint64_t* issue_cycles,
    uint64_t* wait_cycles,
    uint64_t* poll_iters
) {
    const uint64_t issue_begin = read_cycle_counter();
    const uint64_t rd_seq = dma_remote_load_issue(core_id, mm_src_addr, gm_dst_addr, bytes);
    const uint64_t issue_end = read_cycle_counter();
    const WaitProfileStats wait_stats = dma_remote_load_wait_profiled(core_id, rd_seq);

    if (issue_cycles != nullptr) {
        *issue_cycles += (issue_end - issue_begin);
    }
    if (wait_cycles != nullptr) {
        *wait_cycles += wait_stats.wait_cycles;
    }
    if (poll_iters != nullptr) {
        *poll_iters += wait_stats.poll_iters;
    }
}

static inline void group_managed_profiled_dma_remote_load_to_gm(
    int core_id,
    uint64_t mm_src_addr,
    uint64_t gm_dst_addr,
    uint64_t bytes,
    uint64_t* dma_issue_cycles,
    uint64_t* dma_wait_cycles,
    uint64_t* group_wait_cycles,
    uint64_t* poll_iters
) {
    group_manager_service(core_id);
    const WaitProfileStats grant_wait = group_request_dma_token_profiled(core_id, mm_src_addr, gm_dst_addr, bytes);
    if (group_wait_cycles != nullptr) {
        *group_wait_cycles += grant_wait.wait_cycles;
    }
    if (poll_iters != nullptr) {
        *poll_iters += grant_wait.poll_iters;
    }
    profiled_dma_remote_load_to_gm(core_id, mm_src_addr, gm_dst_addr, bytes, dma_issue_cycles, dma_wait_cycles, poll_iters);
    group_request_dma_done(core_id);
    group_manager_service(core_id);
}

static inline WaitProfileStats group_request_dma_batch_token(
    int core_id,
    uint64_t mm_src_addr,
    uint64_t total_batch_bytes
) {
    group_manager_service(core_id);
    return group_request_dma_token_profiled(core_id, mm_src_addr, 0, total_batch_bytes);
}

static inline void group_request_dma_batch_done(int core_id) {
    group_request_dma_done(core_id);
    group_manager_service(core_id);
}

struct DmaTicket {
    uint64_t seq;
    uint64_t bytes;
    int slot;
};

static inline DmaTicket issue_dma_read_ticket_slot(int core_id, int slot, uint64_t mm_src_addr, uint64_t gm_dst_addr, uint64_t bytes) {
    return {.seq = dma_remote_load_issue_slot(core_id, slot, mm_src_addr, gm_dst_addr, bytes), .bytes = bytes, .slot = slot};
}

static inline DmaTicket prepare_dma_read_ticket_slot(int core_id, int slot, uint64_t bytes) {
    return {.seq = dma_remote_load_prepare_slot(core_id, slot), .bytes = bytes, .slot = slot};
}

static inline DmaTicket issue_dma_read_ticket(int core_id, uint64_t mm_src_addr, uint64_t gm_dst_addr, uint64_t bytes) {
    return issue_dma_read_ticket_slot(core_id, 0, mm_src_addr, gm_dst_addr, bytes);
}

static inline void wait_dma_read_ticket(
    int core_id,
    const DmaTicket& ticket,
    uint64_t* issue_cycles,
    uint64_t* wait_cycles,
    uint64_t* poll_iters
) {
    const WaitProfileStats wait_stats = dma_remote_load_wait_profiled_slot(core_id, ticket.slot, ticket.seq);
    (void)issue_cycles;
    if (wait_cycles != nullptr) {
        *wait_cycles += wait_stats.wait_cycles;
    }
    if (poll_iters != nullptr) {
        *poll_iters += wait_stats.poll_iters;
    }
}

static inline void profiled_dma_remote_load_block_to_gm(
    int core_id,
    uint64_t mm_src_addr,
    uint64_t gm_dst_addr,
    uint64_t bytes,
    uint64_t* issue_cycles,
    uint64_t* wait_cycles,
    uint64_t* poll_iters
) {
    const uint64_t issue_begin = read_cycle_counter();
    const DmaTicket ticket = issue_dma_read_ticket(core_id, mm_src_addr, gm_dst_addr, bytes);
    const uint64_t issue_end = read_cycle_counter();
    if (issue_cycles != nullptr) {
        *issue_cycles += (issue_end - issue_begin);
    }
    wait_dma_read_ticket(core_id, ticket, nullptr, wait_cycles, poll_iters);
}

// NOTE: The current per-core DMA completion protocol provides only one read-completion
// sequence/flag pair per core. Issuing multiple remote.ld operations concurrently on the
// same core is therefore unsafe: a later issue overwrites the completion state of an
// earlier issue. For now, we keep reads coarse-grained (block DMA) but still wait in-order.

template <typename T>
static inline void apply_optional_bias(
    int core_id,
    const GemmTaskDescriptor& desc,
    const GemmTileRuntimeContext& rt,
    std::vector<T>& c_tile,
    GemmKernelStats* stats
) {
    if (!BIAS_ENABLED) {
        return;
    }

    for (int n_col = 0; n_col < desc.block_n; ++n_col) {
        const uint64_t bias_src = gemm_desc_bias_src_mm(desc, n_col);
        profiled_dma_remote_load_to_gm(
            core_id,
            bias_src,
            rt.local_vec_in,
            ELEM_BYTES,
            &stats->dma_issue_cycles,
            &stats->dma_wait_cycles,
            &stats->poll_iters
        );
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
    zero_local_accum_tile<T>(desc, rt);
    bool first_dma_ready_reported = false;
    bool first_before_mvm_reported = false;
    bool first_after_mvm_reported = false;

    for (int n_col = 0; n_col < desc.block_n; ++n_col) {
        configure_output_mode(0, 1, true);
        for (int k = 0; k < desc.k_tiles; ++k) {
            const WaitProfileStats batch_grant_wait = group_request_dma_batch_token(
                core_id,
                gemm_desc_mat_src_mm(desc, k),
                MAT_BYTES + VEC_BYTES
            );
            stats->group_wait_cycles += batch_grant_wait.wait_cycles;
            stats->poll_iters += batch_grant_wait.poll_iters;
            mark_exec_window_begin(stats, read_cycle_counter());
            profiled_dma_remote_load_to_gm(
                core_id,
                gemm_desc_mat_src_mm(desc, k),
                rt.local_mat,
                MAT_BYTES,
                &stats->dma_issue_cycles,
                &stats->dma_wait_cycles,
                &stats->poll_iters
            );
            profiled_dma_remote_load_to_gm(
                core_id,
                gemm_desc_vec_src_mm(desc, k, n_col),
                rt.local_vec_in,
                VEC_BYTES,
                &stats->dma_issue_cycles,
                &stats->dma_wait_cycles,
                &stats->poll_iters
            );
            const uint64_t batch_end = read_cycle_counter();

            if (!first_dma_ready_reported && stage_progress_enabled()) {
                printf("[Core %d] [%s] STAGE_PROGRESS: A/B DMA done k_tile=%d cycle=%" PRIu64 "\n",
                       core_id, dtype_label<T>(), k, batch_end);
                fflush(stdout);
                first_dma_ready_reported = true;
            }

            const uint64_t n_loop_begin = read_cycle_counter();
            const uint64_t compute_before = stats->compute_cycles;
            const uint64_t compute_begin = read_cycle_counter();
            if (!first_before_mvm_reported && stage_progress_enabled()) {
                printf("[Core %d] [%s] STAGE_PROGRESS: before first mvm n_col=%d k_tile=%d cycle=%" PRIu64 "\n",
                       core_id, dtype_label<T>(), n_col, k, compute_begin);
                fflush(stdout);
                first_before_mvm_reported = true;
            }
            run_mvm_compute_only<T>(rt.local_mat, rt.local_vec_in, 0);
            const uint64_t compute_end = read_cycle_counter();
            stats->compute_cycles += (compute_end - compute_begin);
            if (!first_after_mvm_reported && stage_progress_enabled()) {
                printf("[Core %d] [%s] STAGE_PROGRESS: after first mvm n_col=%d k_tile=%d cycle=%" PRIu64 "\n",
                       core_id, dtype_label<T>(), n_col, k, compute_end);
                fflush(stdout);
                first_after_mvm_reported = true;
            }
            account_nloop_overhead(stats, n_loop_begin, read_cycle_counter(), compute_before);
            group_request_dma_batch_done(core_id);
        }
        const uint64_t store_begin = read_cycle_counter();
        set_len(VEC_BYTES);
        outputvectorstore(accum_col_addr(desc, rt, n_col), 0);
        const uint64_t store_end = read_cycle_counter();
        stats->compute_cycles += (store_end - store_begin);
    }
    configure_output_mode(0, 0, true);
    apply_optional_bias_gm_fast_path<T>(core_id);
    profiled_store_c_tile_from_gm(desc, rt, &stats->c_store_cycles, stats);
}

template <typename T>
static inline void gemm_tiled_overlap(
    int core_id,
    const GemmTaskDescriptor& desc,
    const GemmTileRuntimeContext& rt,
    GemmKernelStats* stats
) {
    zero_local_accum_tile<T>(desc, rt);

    for (int n_col = 0; n_col < desc.block_n; ++n_col) {
        configure_output_mode(0, 1, true);
        const uint64_t first_mat_begin = read_cycle_counter();
        mark_exec_window_begin(stats, first_mat_begin);
        uint64_t mat_rd_seq = dma_remote_load_issue(core_id, gemm_desc_mat_src_mm(desc, 0), rt.local_mat_ping, MAT_BYTES);
        const WaitProfileStats first_mat_wait = dma_remote_load_wait_profiled(core_id, mat_rd_seq);
        const uint64_t first_mat_end = read_cycle_counter();
        stats->dma_issue_cycles += (first_mat_end - first_mat_begin) - first_mat_wait.wait_cycles;
        stats->dma_wait_cycles += first_mat_wait.wait_cycles;
        stats->poll_iters += first_mat_wait.poll_iters;

        uint64_t local_mat_active = rt.local_mat_ping;
        uint64_t local_mat_next = rt.local_mat_pong;

        for (int k = 0; k < desc.k_tiles; ++k) {
            const bool has_next_k = (k + 1) < desc.k_tiles;
            uint64_t next_rd_seq = 0;
            profiled_dma_remote_load_to_gm(
                core_id,
                gemm_desc_vec_src_mm(desc, k, n_col),
                rt.local_vec_in,
                VEC_BYTES,
                &stats->dma_issue_cycles,
                &stats->dma_wait_cycles,
                &stats->poll_iters
            );

            if (has_next_k) {
                const uint64_t overlap_issue_begin = read_cycle_counter();
                next_rd_seq = dma_remote_load_issue(core_id, gemm_desc_mat_src_mm(desc, k + 1), local_mat_next, MAT_BYTES);
                const uint64_t overlap_issue_end = read_cycle_counter();
                stats->overlap_issue_cycles += (overlap_issue_end - overlap_issue_begin);
            }

            const uint64_t n_loop_begin = read_cycle_counter();
            const uint64_t compute_before = stats->compute_cycles;
            const uint64_t compute_begin = read_cycle_counter();
            run_mvm_compute_only<T>(local_mat_active, rt.local_vec_in, 0);
            const uint64_t compute_end = read_cycle_counter();
            stats->compute_cycles += (compute_end - compute_begin);
            account_nloop_overhead(stats, n_loop_begin, compute_end, compute_before);

            if (has_next_k) {
                const WaitProfileStats overlap_wait = dma_remote_load_wait_profiled(core_id, next_rd_seq);
                stats->overlap_wait_cycles += overlap_wait.wait_cycles;
                stats->poll_iters += overlap_wait.poll_iters;
                std::swap(local_mat_active, local_mat_next);
            }
        }
        const uint64_t store_begin = read_cycle_counter();
        set_len(VEC_BYTES);
        outputvectorstore(accum_col_addr(desc, rt, n_col), 0);
        const uint64_t store_end = read_cycle_counter();
        stats->compute_cycles += (store_end - store_begin);
    }

    configure_output_mode(0, 0, true);
    apply_optional_bias_gm_fast_path<T>(core_id);
    profiled_store_c_tile_from_gm(desc, rt, &stats->c_store_cycles, stats);
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
        const int64_t src_idx = tensors.transpose_b
            ? static_cast<int64_t>(n_base) * tensors.b_stride0 + static_cast<int64_t>(k_base + i) * tensors.b_stride1
            : static_cast<int64_t>(k_base + i) * tensors.b_stride0 + static_cast<int64_t>(n_base) * tensors.b_stride1;
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
        mark_exec_window_begin(stats, read_cycle_counter());
        const uint64_t dma_mat_begin = read_cycle_counter();
        load_a_tile_from_tensor<T>(desc, tensors, k, mat_tile);
        set_len(mat_bytes);
        mm2gm(mat_tile.data(), rt.local_mat);
        const uint64_t dma_mat_end = read_cycle_counter();
        stats->dma_issue_cycles += (dma_mat_end - dma_mat_begin);

        const uint64_t n_loop_begin = read_cycle_counter();
        const uint64_t compute_before = stats->compute_cycles;
        for (int n_col = 0; n_col < desc.block_n; ++n_col) {
            const uint64_t dma_vec_begin = read_cycle_counter();
            load_b_col_from_tensor<T>(desc, tensors, k, n_col, vec);
            set_len(vec_bytes);
            mm2gm(vec.data(), rt.local_vec_in);
            const uint64_t dma_vec_end = read_cycle_counter();
            stats->dma_issue_cycles += (dma_vec_end - dma_vec_begin);

            const uint64_t compute_begin = read_cycle_counter();
            accumulate_c_tile_column_host<T>(desc, rt, rt.local_mat, n_col, c_tile, out_vec);
            const uint64_t compute_end = read_cycle_counter();
            stats->compute_cycles += (compute_end - compute_begin);
        }
        account_nloop_overhead(stats, n_loop_begin, read_cycle_counter(), compute_before);
    }

      apply_optional_bias<T>(core_id, desc, rt, c_tile, stats);
      write_c_tile_to_tensor<T>(desc, tensors, c_tile);
      const uint64_t c_store_begin = read_cycle_counter();
      store_c_tile<T>(desc, rt, c_tile);
      const uint64_t c_store_end = read_cycle_counter();
      stats->c_store_cycles += (c_store_end - c_store_begin);
      mark_exec_window_end(stats, c_store_end);
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
    const bool block_k_supported =
        (cfg.block_k <= TILE_K) || ((cfg.block_k % TILE_K) == 0);
    if ((cfg.block_m % TILE_M) != 0 || !block_k_supported) {
        printf("[Core %d] [ERROR] block_M must be a multiple of ARRAY_OUTPUT(%d); block_K must be <= ARRAY_INPUT(%d) or an integer multiple of it, got (%d,%d)\n",
               core_id, TILE_M, TILE_K, cfg.block_m, cfg.block_k);
        return;
    }
    if (cfg.block_k < TILE_K && !WORKER_COMMAND_PROCESSOR_ENABLED) {
        printf("[Core %d] [ERROR] block_K(%d) < ARRAY_INPUT(%d) requires WorkerCommandProcessor lane padding\n",
               core_id, cfg.block_k, TILE_K);
        return;
    }
    if (cfg.block_n > TILE_N_MAX) {
        printf("[Core %d] [ERROR] block_N(%d) exceeds current packed-vector width NUM_ARRAYS(%d)\n",
               core_id, cfg.block_n, TILE_N_MAX);
        return;
    }
    const GemmTileRuntimeContext rt = make_gemm_runtime_context(core_id);
    const int total_tasks = gemm_total_tasks(cfg);
    group_manager_prepare(core_id);
    if (GROUP_MANAGER_ENABLED && is_group_leader(core_id)) {
        GemmKernelStats stats = {};
        const uint64_t total_begin = read_cycle_counter();
        const WaitProfileStats drain_wait = group_manager_drain_until_group_complete(core_id);
        stats.group_wait_cycles += drain_wait.wait_cycles;
        stats.poll_iters += drain_wait.poll_iters;
        const uint64_t total_end = read_cycle_counter();
        if (runtime_info_enabled()) {
            printf("[Core %d] [%s] MANAGER summary: group_wait=%" PRIu64 " poll_iters=%" PRIu64 " total=%" PRIu64 "\n",
                   core_id, dtype_label<T>(), stats.group_wait_cycles, stats.poll_iters, total_end - total_begin);
            fflush(stdout);
        }
        return;
    }
    const int worker_slot = gemm_worker_slot_for_core(core_id);
    if (worker_slot < 0 || worker_slot >= total_tasks) {
        if (runtime_info_enabled()) {
            printf("[Core %d] GEMM 核心空闲（total_tasks=%d, active_cores=%d）。\n",
                   core_id, total_tasks, ACTIVE_GEMM_CORES);
        }
        group_mark_worker_finished(core_id);
        const WaitProfileStats group_wait = group_wait_for_group_done_profiled(core_id);
        (void)group_wait;
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

    for (int task_id = worker_slot; task_id < total_tasks; task_id += ACTIVE_GEMM_CORES) {
        if (first_task < 0) {
            first_task = task_id;
        }
        last_task = task_id;
        tasks_done++;

        const uint64_t task_loop_begin = read_cycle_counter();
        const GemmTaskDescriptor desc = gemm_descriptor_for_task(core_id, task_id, cfg);
        const uint64_t task_loop_end = read_cycle_counter();
        stats.task_desc_overhead_cycles += (task_loop_end - task_loop_begin);
        if (tensors != nullptr && tensors->a != nullptr && tensors->b != nullptr && tensors->c != nullptr) {
            gemm_tiled_from_tensors<T>(core_id, desc, rt, &stats, *tensors);
        } else {
            gemm_tiled<T>(core_id, desc, rt, &stats);
        }
    }

    const uint64_t finish_publish_begin = read_cycle_counter();
    group_mark_worker_finished(core_id);
    const uint64_t finish_publish_end = read_cycle_counter();
    stats.finish_publish_cycles += (finish_publish_end - finish_publish_begin);
    const WaitProfileStats group_wait = group_wait_for_group_done_profiled(core_id);
    stats.group_wait_cycles += group_wait.wait_cycles;
    stats.poll_iters += group_wait.poll_iters;

    const uint64_t total_end = read_cycle_counter();
    const uint64_t dma_total_cycles =
        stats.dma_issue_cycles +
        stats.dma_wait_cycles +
        stats.overlap_issue_cycles +
        stats.overlap_wait_cycles;
    uint64_t total_cycles = exec_window_cycles(stats);
    if (total_cycles == 0) {
        const uint64_t full_cycles = total_end - total_begin;
        total_cycles = (full_cycles > stats.group_wait_cycles) ? (full_cycles - stats.group_wait_cycles) : 0;
    }
    printf("[Core %d] [%s] LATENCY(cycles): dma_issue=%" PRIu64
           " dma_wait=%" PRIu64
           " dma_total=%" PRIu64
           " compute=%" PRIu64
           " sched_protocol=%" PRIu64
           " c_store=%" PRIu64
           " group_wait=%" PRIu64
           " poll_iters=%" PRIu64
           " overlap_issue=%" PRIu64
           " overlap_wait=%" PRIu64
           " issue_block_q=%" PRIu64
           " issue_write=%" PRIu64
           " ov_issue_block_q=%" PRIu64
           " ov_issue_write=%" PRIu64
           " task_desc=%" PRIu64
           " nloop=%" PRIu64
           " submit_pack=%" PRIu64
           " finish_publish=%" PRIu64
           " total=%" PRIu64 "\n",
              core_id,
              dtype_label<T>(),
                stats.dma_issue_cycles,
                stats.dma_wait_cycles,
                dma_total_cycles,
                stats.compute_cycles,
                stats.sched_protocol_cycles,
                stats.c_store_cycles,
                stats.group_wait_cycles,
                stats.poll_iters,
                stats.overlap_issue_cycles,
                stats.overlap_wait_cycles,
                stats.issue_block_submitq_cycles,
                stats.issue_write_submitq_cycles,
                stats.ov_issue_block_submitq_cycles,
                stats.ov_issue_write_submitq_cycles,
                stats.task_desc_overhead_cycles,
                stats.nloop_overhead_cycles,
                stats.submit_pack_cycles,
                stats.finish_publish_cycles,
                total_cycles);
    fflush(stdout);

    if (DMA_OVERLAP_ENABLED) {
        if (runtime_info_enabled()) {
            printf("[Core %d] [%s] OVERLAP(cycles): issue=%" PRIu64 " wait=%" PRIu64 "\n",
                   core_id,
                   dtype_label<T>(),
                   stats.overlap_issue_cycles,
                   stats.overlap_wait_cycles);
            printf("[Core %d] [%s] GEMM/MM summary(overlap=1): tasks_done=%d first_task=%d last_task=%d task_stride=%d total_tasks=%d block=(%d,%d,%d) k_tiles=%d mat_ping=0x%" PRIx64 ", mat_pong=0x%" PRIx64 ", vec=0x%" PRIx64 ", out=0x%" PRIx64 "\n",
                   core_id, dtype_label<T>(), tasks_done, first_task, last_task, ACTIVE_GEMM_CORES, total_tasks,
                   cfg.block_m, cfg.block_n, cfg.block_k, gemm_k_tiles(cfg),
                   rt.local_mat_ping, rt.local_mat_pong, rt.local_vec_in, rt.local_out);
        }
        return;
    }

    if (runtime_info_enabled()) {
        printf("[Core %d] [%s] GEMM/MM summary: tasks_done=%d first_task=%d last_task=%d task_stride=%d total_tasks=%d block=(%d,%d,%d) k_tiles=%d mat=0x%" PRIx64 ", vec=0x%" PRIx64 ", out=0x%" PRIx64 "\n",
               core_id, dtype_label<T>(), tasks_done, first_task, last_task, ACTIVE_GEMM_CORES, total_tasks,
               cfg.block_m, cfg.block_n, cfg.block_k, gemm_k_tiles(cfg),
               rt.local_mat, rt.local_vec_in, rt.local_out);
    }
}

static inline void matmul_for_core(int core_id, const MatmulRuntimeConfig& cfg, const MatmulTensorBindings* tensors) {
    matmul_for_core_t<int32_t>(core_id, cfg, tensors);
}

inline void matmul_fp16(int M, int N, int K, int block_M, int block_N, int block_K) {
    const int core_id = sched_getcpu();
    if (core_id < 0 || core_id >= TOTAL_CORES) {
        printf("[ERROR] invalid runtime core id=%d, TOTAL_CORES=%d\n", core_id, TOTAL_CORES);
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
    // FP16 is stored in GM/HBM while the configured float array performs
    // the numerical accumulation in FP32.
    matmul_for_core_t<float>(core_id, cfg, nullptr);
}

inline void matmul_with_tensors_fp16(
    int M,
    int N,
    int K,
    int block_M,
    int block_N,
    int block_K,
    const MatmulTensorBindingsFP16& tensors
) {
    const int core_id = sched_getcpu();
    if (core_id < 0 || core_id >= TOTAL_CORES) {
        printf("[ERROR] invalid runtime core id=%d, TOTAL_CORES=%d\n", core_id, TOTAL_CORES);
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
    matmul_for_core_t<SST::Golem::GolemFp16>(core_id, cfg, &tensors);
}

inline void matmul(int M, int N, int K, int block_M, int block_N, int block_K) {
    const int core_id = sched_getcpu();
    if (core_id < 0 || core_id >= TOTAL_CORES) {
        printf("[ERROR] invalid runtime core id=%d, TOTAL_CORES=%d\n", core_id, TOTAL_CORES);
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
