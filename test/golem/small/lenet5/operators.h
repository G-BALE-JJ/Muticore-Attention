#pragma once

#include <cstdint>

#include "gm_config.h"
#include "ex_instr.h"
#include "pipeline_config.h"

// ---------- 基础等待与屏障 ----------
inline void delay_cycles(volatile uint32_t cycles) {
    while (cycles--) {
        __asm__ volatile ("" ::: "memory");
    }
}

inline void memory_barrier() {
    MEMORY_BARRIER();
}

inline void adaptive_wait_eq(uint64_t local_addr, uint64_t expected) {
    uint32_t backoff = 8;
    while (gm2reg(local_addr) != expected) {
        delay_cycles(backoff);
        if (backoff < 2048) {
            backoff <<= 1;
        }
    }
}

// ---------- 拓扑辅助 ----------
inline int group_id_of_core(int core_id) {
    return core_id / GROUP_SIZE;
}

inline int leader_core_of_group(int group_id) {
    return group_id * GROUP_SIZE;
}

inline bool is_group_leader(int core_id) {
    return (core_id % GROUP_SIZE) == 0;
}

// ---------- 地址辅助 ----------
inline uint64_t gm_addr(int core_id, uint64_t offset) {
    return get_core_data_addr(core_id) + offset;
}

inline uint64_t tmp_addr(int core_id) {
    return gm_addr(core_id, LOCAL_LAYOUT.tmp);
}

inline uint64_t seq_addr_for_group(int dst_group) {
    return get_core_mailbox_addr(leader_core_of_group(dst_group)) + MBOX_LAYOUT.seq;
}

inline uint64_t ack_addr_for_group(int src_group) {
    return get_core_mailbox_addr(leader_core_of_group(src_group)) + MBOX_LAYOUT.ack;
}

// ---------- 通用通信/同步原语 ----------
inline void remote_write_u64(int src_core, uint64_t value, uint64_t remote_addr) {
    reg2gm(value, tmp_addr(src_core));
    set_len(8);
    remote_store(tmp_addr(src_core), remote_addr);
}

inline void init_sync(int core_id) {
    int gid = group_id_of_core(core_id);
    if (gid > 0) {
        reg2gm(0, seq_addr_for_group(gid));
    }
    if (gid < TOTAL_GROUPS - 1) {
        reg2gm(0, ack_addr_for_group(gid));
    }
}

inline uint64_t dma_remote_load_issue(int core_id, uint64_t mm_src_addr, uint64_t gm_dst_addr, uint64_t bytes) {
    uint64_t rd_seq_addr = get_core_read_seq_addr(core_id);
    uint64_t rd_flag_addr = get_core_read_flag_addr(core_id);
    uint64_t rd_seq = gm2reg(rd_seq_addr) + 1;

    reg2gm(rd_seq, rd_seq_addr);
    reg2gm(0, rd_flag_addr);
    set_len(bytes);
    remote_load(mm_src_addr, gm_dst_addr);
    return rd_seq;
}

inline void dma_remote_load_wait(int core_id, uint64_t expected_rd_seq) {
    uint64_t rd_flag_addr = get_core_read_flag_addr(core_id);
    adaptive_wait_eq(rd_flag_addr, expected_rd_seq);
}

inline void dma_remote_load_to_gm(int core_id, uint64_t mm_src_addr, uint64_t gm_dst_addr, uint64_t bytes) {
    uint64_t rd_seq = dma_remote_load_issue(core_id, mm_src_addr, gm_dst_addr, bytes);
    dma_remote_load_wait(core_id, rd_seq);
}

inline void run_mvm_stage(uint64_t mat_gm, uint64_t vec_gm, uint64_t out_gm, uint64_t array_id = 0) {
    inputmatrixload(mat_gm, array_id);
    inputvectorload(vec_gm, array_id);
    mvm_compute(array_id);
    outputvectorstore(out_gm, array_id);
}

inline GemmTaskDescriptor gemm_descriptor_for_task(int core_id, int task_id, const MatmulRuntimeConfig& cfg) {
    return gemm_task_desc_for_task(core_id, task_id, cfg);
}

inline uint64_t gemm_desc_mat_src_mm(const GemmTaskDescriptor& desc, int k_tile) {
    return gemm_desc_a_src_mm(desc, k_tile);
}

inline uint64_t gemm_desc_vec_src_mm(const GemmTaskDescriptor& desc, int k_tile, int n_col_in_tile) {
    return gemm_desc_b_pack_src_mm(desc, k_tile, n_col_in_tile);
}

inline void transfer_vector_to_next_group(uint64_t producer_vec_gm, uint64_t next_vec_gm, uint64_t bytes) {
    set_len(bytes);
    remote_store(producer_vec_gm, next_vec_gm);
    memory_barrier();
}

inline void notify_next_group_ready(int my_core, int my_group) {
    if (my_group < TOTAL_GROUPS - 1) {
        remote_write_u64(my_core, 1, seq_addr_for_group(my_group + 1));
    }
}

inline void notify_prev_group_done(int my_core, int my_group) {
    if (my_group > 0) {
        remote_write_u64(my_core, 1, ack_addr_for_group(my_group - 1));
    }
}
