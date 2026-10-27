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

inline bool is_group_leader(int core_id);
inline void group_manager_service(int core_id);
inline void group_manager_prepare(int core_id);

inline void adaptive_wait_eq(uint64_t local_addr, uint64_t expected) {
    uint32_t backoff = 8;
    while (gm2reg(local_addr) != expected) {
        delay_cycles(backoff);
        if (backoff < 2048) {
            backoff <<= 1;
        }
    }
}

struct WaitProfileStats {
    uint64_t wait_cycles;
    uint64_t poll_iters;
};

constexpr int GROUP_GRANT_WINDOW = GOLEM_GROUP_GRANT_WINDOW;
constexpr int GROUP_PINGPONG_SLOTS = 2;

inline void group_mark_worker_finished(int core_id);
inline WaitProfileStats group_request_dma_token_profiled(int core_id, uint64_t mm_src_addr, uint64_t gm_dst_addr, uint64_t bytes);
inline WaitProfileStats group_wait_for_group_done_profiled(int core_id);
inline WaitProfileStats group_manager_drain_until_group_complete(int core_id);

inline WaitProfileStats adaptive_wait_eq_profiled(int core_id, uint64_t local_addr, uint64_t expected) {
    const uint64_t begin = read_cycles();
    uint64_t polls = 0;
    uint32_t backoff = 8;
    while (true) {
        if (GROUP_MANAGER_ENABLED && is_group_leader(core_id)) {
            group_manager_service(core_id);
        }
        polls++;
        if (gm2reg(local_addr) == expected) {
            break;
        }
        delay_cycles(backoff);
        if (backoff < 2048) {
            backoff <<= 1;
        }
    }
    const uint64_t end = read_cycles();
    return {
        .wait_cycles = end - begin,
        .poll_iters = polls,
    };
}

// ---------- 拓扑辅助 ----------
inline int group_id_of_core(int core_id) {
    return core_id % TOTAL_GROUPS;
}

inline int leader_core_of_group(int group_id) {
    return group_id;
}

inline bool is_group_leader(int core_id) {
    return core_id < TOTAL_GROUPS;
}

inline int local_index_in_group(int core_id) {
    return core_id / TOTAL_GROUPS;
}

inline int core_id_in_group(int group_id, int local_index) {
    return group_id + local_index * TOTAL_GROUPS;
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

inline uint64_t dma_remote_load_monotonic_next_seq_slot(int core_id, int slot) {
    const uint64_t rd_seq_addr = get_core_read_seq_addr_slot(core_id, slot);
    const uint64_t rd_flag_addr = get_core_read_flag_addr_slot(core_id, slot);
    const uint64_t cur_seq = gm2reg(rd_seq_addr);
    const uint64_t cur_flag = gm2reg(rd_flag_addr);
    return ((cur_seq > cur_flag) ? cur_seq : cur_flag) + 1;
}

inline uint64_t dma_remote_load_issue_slot(int core_id, int slot, uint64_t mm_src_addr, uint64_t gm_dst_addr, uint64_t bytes) {
    uint64_t rd_seq_addr = get_core_read_seq_addr_slot(core_id, slot);
    uint64_t rd_flag_addr = get_core_read_flag_addr_slot(core_id, slot);
    uint64_t rd_slot_sel_addr = get_core_read_slot_selector_addr(core_id);
    uint64_t rd_seq = gm2reg(rd_seq_addr) + 1;

    reg2gm(static_cast<uint64_t>(slot), rd_slot_sel_addr);
    reg2gm(rd_seq, rd_seq_addr);
    reg2gm(0, rd_flag_addr);
    set_len(bytes);
    remote_load(mm_src_addr, gm_dst_addr);
    return rd_seq;
}

inline uint64_t dma_remote_load_prepare_slot(int core_id, int slot) {
    uint64_t rd_seq_addr = get_core_read_seq_addr_slot(core_id, slot);
    uint64_t rd_seq = dma_remote_load_monotonic_next_seq_slot(core_id, slot);
    reg2gm(rd_seq, rd_seq_addr);
    return rd_seq;
}

inline uint64_t dma_remote_load_prepare_pair_slots(int core_id, int slot_a, int slot_b) {
    uint64_t seq_addr_a = get_core_read_seq_addr_slot(core_id, slot_a);
    uint64_t seq_addr_b = get_core_read_seq_addr_slot(core_id, slot_b);
    uint64_t flag_addr_a = get_core_read_flag_addr_slot(core_id, slot_a);
    uint64_t flag_addr_b = get_core_read_flag_addr_slot(core_id, slot_b);
    const uint64_t cur_a = gm2reg(seq_addr_a);
    const uint64_t cur_b = gm2reg(seq_addr_b);
    const uint64_t flag_a = gm2reg(flag_addr_a);
    const uint64_t flag_b = gm2reg(flag_addr_b);
    const uint64_t seq_max = (cur_a > cur_b) ? cur_a : cur_b;
    const uint64_t flag_max = (flag_a > flag_b) ? flag_a : flag_b;
    const uint64_t shared_seq = ((seq_max > flag_max) ? seq_max : flag_max) + 1;
    reg2gm(shared_seq, seq_addr_a);
    reg2gm(shared_seq, seq_addr_b);
    return shared_seq;
}

inline uint64_t dma_remote_load_next_seq_slot(int core_id, int slot) {
    return dma_remote_load_monotonic_next_seq_slot(core_id, slot);
}

inline uint64_t dma_remote_load_issue(int core_id, uint64_t mm_src_addr, uint64_t gm_dst_addr, uint64_t bytes) {
    return dma_remote_load_issue_slot(core_id, 0, mm_src_addr, gm_dst_addr, bytes);
}

inline void dma_remote_load_wait_slot(int core_id, int slot, uint64_t expected_rd_seq) {
    uint64_t rd_flag_addr = get_core_read_flag_addr_slot(core_id, slot);
    adaptive_wait_eq(rd_flag_addr, expected_rd_seq);
}

inline void dma_remote_load_wait(int core_id, uint64_t expected_rd_seq) {
    dma_remote_load_wait_slot(core_id, 0, expected_rd_seq);
}

inline WaitProfileStats dma_remote_load_wait_profiled_slot(int core_id, int slot, uint64_t expected_rd_seq) {
    uint64_t rd_flag_addr = get_core_read_flag_addr_slot(core_id, slot);
    return adaptive_wait_eq_profiled(core_id, rd_flag_addr, expected_rd_seq);
}

inline WaitProfileStats dma_remote_load_wait_profiled(int core_id, uint64_t expected_rd_seq) {
    return dma_remote_load_wait_profiled_slot(core_id, 0, expected_rd_seq);
}

inline bool dma_remote_load_test_slot(int core_id, int slot, uint64_t expected_rd_seq) {
    uint64_t rd_flag_addr = get_core_read_flag_addr_slot(core_id, slot);
    return gm2reg(rd_flag_addr) == expected_rd_seq;
}

inline bool dma_remote_load_test(int core_id, uint64_t expected_rd_seq) {
    return dma_remote_load_test_slot(core_id, 0, expected_rd_seq);
}

inline void dma_remote_load_to_gm(int core_id, uint64_t mm_src_addr, uint64_t gm_dst_addr, uint64_t bytes) {
    uint64_t rd_seq = dma_remote_load_issue(core_id, mm_src_addr, gm_dst_addr, bytes);
    dma_remote_load_wait(core_id, rd_seq);
}

inline void run_mvm_stage(uint64_t mat_gm, uint64_t vec_gm, uint64_t out_gm, uint64_t array_id = 0) {
    inputmatrixload(mat_gm, array_id);
    inputvectorload(vec_gm, array_id);
    mvm_compute(array_id);
    set_len(VEC_BYTES);
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

enum : uint64_t {
    GROUP_REQ_STATE_EMPTY = 0,
    GROUP_REQ_STATE_PENDING = 1,
    GROUP_REQ_STATE_GRANTED = 2,
    GROUP_REQ_STATE_DONE = 3,
};

inline uint64_t group_slot_field_addr(int manager_core_id, int local_index, uint64_t field_off) {
    return get_group_ctrl_slot_addr(manager_core_id, local_index) + field_off;
}

inline uint64_t group_req_entry_field_addr(int manager_core_id, int local_index, int ring_index, uint64_t field_off) {
    return get_group_req_entry_addr(manager_core_id, local_index, ring_index) + field_off;
}

inline uint64_t local_group_meta_field_addr(int core_id, uint64_t field_off) {
    return get_group_ctrl_base_addr(core_id) + GROUP_META_BASE_OFF + field_off;
}

inline uint64_t group_manager_ready_addr(int core_id) {
    return local_group_meta_field_addr(core_id, GROUP_READY_OFF);
}

inline uint64_t group_drain_addr(int core_id) {
    return local_group_meta_field_addr(core_id, GROUP_DRAIN_OFF);
}

inline uint64_t group_done_addr(int core_id) {
    return local_group_meta_field_addr(core_id, GROUP_DONE_OFF);
}

inline uint64_t worker_request_slot_addr(int worker_core) {
    const int gid = group_id_of_core(worker_core);
    const int manager = leader_core_of_group(gid);
    const int lidx = local_index_in_group(worker_core);
    return get_group_ctrl_slot_addr(manager, lidx);
}

inline uint64_t worker_request_field_addr(int worker_core, uint64_t field_off) {
    return worker_request_slot_addr(worker_core) + field_off;
}

inline uint64_t worker_request_entry_addr(int worker_core, int ring_index) {
    const int gid = group_id_of_core(worker_core);
    const int manager = leader_core_of_group(gid);
    const int lidx = local_index_in_group(worker_core);
    return get_group_req_entry_addr(manager, lidx, ring_index);
}

inline uint64_t worker_request_entry_field_addr(int worker_core, int ring_index, uint64_t field_off) {
    return worker_request_entry_addr(worker_core, ring_index) + field_off;
}

inline uint64_t worker_slot_state_addr(int worker_core, int ring_index) {
    return worker_request_entry_field_addr(worker_core, ring_index, GROUP_SLOT_STATE_OFF);
}

inline uint64_t worker_slot_done_addr(int worker_core, int ring_index) {
    return worker_request_entry_field_addr(worker_core, ring_index, GROUP_SLOT_DONE_OFF);
}

inline void clear_worker_request_slot_local(int manager_core, int worker_core) {
    const int gid = group_id_of_core(manager_core);
    const int manager = leader_core_of_group(gid);
    const int lidx = local_index_in_group(worker_core);
    const uint64_t slot_base = get_group_ctrl_slot_addr(manager, lidx);
    const uint64_t entry_base = get_group_req_entry_addr(manager, lidx, 0);
    reg2gm(0, entry_base + GROUP_REQ_SEQ_OFF);
    reg2gm(0, entry_base + GROUP_OP_OFF);
    reg2gm(0, entry_base + GROUP_SRC_OFF);
    reg2gm(0, entry_base + GROUP_DST_OFF);
    reg2gm(0, entry_base + GROUP_BYTES_OFF);
    reg2gm(0, entry_base + GROUP_NODE_OFF);
    reg2gm(GROUP_REQ_STATE_EMPTY, entry_base + GROUP_SLOT_STATE_OFF);
    reg2gm(0, entry_base + GROUP_SLOT_DONE_OFF);
    reg2gm(0, slot_base + GROUP_REQ_HEAD_OFF);
    reg2gm(0, slot_base + GROUP_REQ_TAIL_OFF);
    reg2gm(0, slot_base + GROUP_GRANTED_SEQ_OFF);
    reg2gm(0, slot_base + GROUP_DONE_SEQ_OFF);
    reg2gm(GROUP_REQ_STATE_EMPTY, slot_base + GROUP_STATE_OFF);
    reg2gm(0, slot_base + GROUP_WORKER_FINISHED_OFF);
}

inline void group_manager_prepare(int core_id) {
    if (!GROUP_MANAGER_ENABLED) {
        return;
    }

    static bool prepared = false;
    if (prepared) {
        return;
    }

    reg2gm(0, get_core_group_grant_addr(core_id));
    reg2gm(0, group_manager_ready_addr(core_id));
    reg2gm(0, group_drain_addr(core_id));
    reg2gm(0, group_done_addr(core_id));

    if (is_group_leader(core_id)) {
        const int gid = group_id_of_core(core_id);
        for (int lidx = 1; lidx < GROUP_SIZE; ++lidx) {
            const int worker_core = core_id_in_group(gid, lidx);
            clear_worker_request_slot_local(core_id, worker_core);
            remote_write_u64(core_id, 0, get_core_group_grant_addr(worker_core));
            remote_write_u64(core_id, 0, group_done_addr(worker_core));
            remote_write_u64(core_id, 1, group_manager_ready_addr(worker_core));
        }
        reg2gm(1, group_manager_ready_addr(core_id));
        reg2gm(0, group_done_addr(core_id));
    } else {
        adaptive_wait_eq(group_manager_ready_addr(core_id), 1);
    }

    prepared = true;
}

inline void group_mark_worker_finished(int core_id) {
    if (!GROUP_MANAGER_ENABLED || is_group_leader(core_id)) {
        return;
    }
    group_manager_prepare(core_id);
    remote_write_u64(core_id, 1, worker_request_field_addr(core_id, GROUP_WORKER_FINISHED_OFF));
}

inline WaitProfileStats group_wait_for_group_done_profiled(int core_id) {
    if (!GROUP_MANAGER_ENABLED || is_group_leader(core_id)) {
        return {.wait_cycles = 0, .poll_iters = 0};
    }
    group_manager_prepare(core_id);
    return adaptive_wait_eq_profiled(core_id, group_done_addr(core_id), 1);
}

inline WaitProfileStats group_manager_drain_until_group_complete(int core_id) {
    if (!GROUP_MANAGER_ENABLED || !is_group_leader(core_id)) {
        return {.wait_cycles = 0, .poll_iters = 0};
    }
    group_manager_prepare(core_id);
    const uint64_t begin = read_cycles();
    uint64_t polls = 0;
    while (gm2reg(group_done_addr(core_id)) != 1) {
        group_manager_service(core_id);
        polls++;
        delay_cycles(8);
    }
    const uint64_t end = read_cycles();
    return {.wait_cycles = end - begin, .poll_iters = polls};
}

inline void publish_worker_request(int core_id, uint64_t req_seq, uint64_t mm_src_addr, uint64_t gm_dst_addr, uint64_t bytes) {
    remote_write_u64(core_id, req_seq, worker_request_entry_field_addr(core_id, 0, GROUP_REQ_SEQ_OFF));
    remote_write_u64(core_id, 0, worker_request_entry_field_addr(core_id, 0, GROUP_OP_OFF));
    remote_write_u64(core_id, mm_src_addr, worker_request_entry_field_addr(core_id, 0, GROUP_SRC_OFF));
    remote_write_u64(core_id, gm_dst_addr, worker_request_entry_field_addr(core_id, 0, GROUP_DST_OFF));
    remote_write_u64(core_id, bytes, worker_request_entry_field_addr(core_id, 0, GROUP_BYTES_OFF));
    remote_write_u64(core_id, mm_src_addr / MEM_NODE_SIZE, worker_request_entry_field_addr(core_id, 0, GROUP_NODE_OFF));
    remote_write_u64(core_id, 0, worker_slot_done_addr(core_id, 0));
    remote_write_u64(core_id, GROUP_REQ_STATE_PENDING, worker_slot_state_addr(core_id, 0));
    remote_write_u64(core_id, req_seq, worker_request_field_addr(core_id, GROUP_REQ_TAIL_OFF));
}

inline void group_request_dma_token(int core_id, uint64_t mm_src_addr, uint64_t gm_dst_addr, uint64_t bytes) {
    if (!GROUP_MANAGER_ENABLED || is_group_leader(core_id)) {
        return;
    }
    static uint64_t request_seq[GOLEM_TOTAL_CORES] = {0};
    group_manager_prepare(core_id);
    adaptive_wait_eq(get_core_group_grant_addr(core_id), 0);
    request_seq[core_id]++;
    publish_worker_request(core_id, request_seq[core_id], mm_src_addr, gm_dst_addr, bytes);
    adaptive_wait_eq(get_core_group_grant_addr(core_id), request_seq[core_id]);
}

inline WaitProfileStats group_request_dma_token_profiled(int core_id, uint64_t mm_src_addr, uint64_t gm_dst_addr, uint64_t bytes) {
    if (!GROUP_MANAGER_ENABLED || is_group_leader(core_id)) {
        return {.wait_cycles = 0, .poll_iters = 0};
    }
    static uint64_t request_seq[GOLEM_TOTAL_CORES] = {0};
    group_manager_prepare(core_id);
    const WaitProfileStats slot_wait = adaptive_wait_eq_profiled(core_id, get_core_group_grant_addr(core_id), 0);
    request_seq[core_id]++;
    publish_worker_request(core_id, request_seq[core_id], mm_src_addr, gm_dst_addr, bytes);
    const WaitProfileStats grant_wait = adaptive_wait_eq_profiled(core_id, get_core_group_grant_addr(core_id), request_seq[core_id]);
    return {.wait_cycles = slot_wait.wait_cycles + grant_wait.wait_cycles,
            .poll_iters = slot_wait.poll_iters + grant_wait.poll_iters};
}

inline void group_request_dma_done(int core_id) {
    if (!GROUP_MANAGER_ENABLED || is_group_leader(core_id)) {
        return;
    }
    const uint64_t granted = gm2reg(get_core_group_grant_addr(core_id));
    remote_write_u64(core_id, granted, worker_slot_done_addr(core_id, 0));
    remote_write_u64(core_id, GROUP_REQ_STATE_DONE, worker_slot_state_addr(core_id, 0));
    remote_write_u64(core_id, granted, worker_request_field_addr(core_id, GROUP_DONE_SEQ_OFF));
}

inline void group_manager_service(int core_id) {
    if (!GROUP_MANAGER_ENABLED || !is_group_leader(core_id)) {
        return;
    }
    group_manager_prepare(core_id);

    static bool inited = false;
    static uint64_t inflight_seq[GROUP_SIZE] = {0};
    static uint64_t inflight_node[GROUP_SIZE] = {0};
    static uint64_t inflight_per_node[NUM_MEMORY_NODES] = {0};
    static int queue_worker[GROUP_SIZE * 8] = {0};
    static uint64_t queue_seq[GROUP_SIZE * 8] = {0};
    static int q_head = 0;
    static int q_tail = 0;
    static uint64_t seen_req_seq[GROUP_SIZE] = {0};

    if (!inited) {
        for (int i = 0; i < GROUP_SIZE; ++i) {
            inflight_seq[i] = 0;
            inflight_node[i] = 0;
            seen_req_seq[i] = 0;
        }
        for (int node = 0; node < NUM_MEMORY_NODES; ++node) {
            inflight_per_node[node] = 0;
        }
        q_head = 0;
        q_tail = 0;
        inited = true;
    }

    const int gid = group_id_of_core(core_id);

    for (int lidx = 1; lidx < GROUP_SIZE; ++lidx) {
        const int worker_core = core_id_in_group(gid, lidx);
        if (inflight_seq[lidx] == 0) {
            continue;
        }
        const uint64_t done_seq = gm2reg(worker_slot_done_addr(worker_core, 0));
        const uint64_t slot_state = gm2reg(worker_slot_state_addr(worker_core, 0));
        if (done_seq == inflight_seq[lidx] && slot_state == GROUP_REQ_STATE_DONE) {
            const uint64_t node = inflight_node[lidx];
            if (node < NUM_MEMORY_NODES && inflight_per_node[node] > 0) {
                inflight_per_node[node]--;
            }
            reg2gm(done_seq, worker_request_field_addr(worker_core, GROUP_GRANTED_SEQ_OFF));
            reg2gm(done_seq, worker_request_field_addr(worker_core, GROUP_REQ_HEAD_OFF));
            reg2gm(GROUP_REQ_STATE_EMPTY, worker_slot_state_addr(worker_core, 0));
            reg2gm(0, worker_slot_done_addr(worker_core, 0));
            remote_write_u64(core_id, 0, get_core_group_grant_addr(worker_core));
            inflight_seq[lidx] = 0;
            inflight_node[lidx] = 0;
        }
    }

    for (int lidx = 1; lidx < GROUP_SIZE; ++lidx) {
        if (inflight_seq[lidx] != 0) {
            continue;
        }
        const int worker_core = core_id_in_group(gid, lidx);
        const uint64_t req_seq = gm2reg(worker_request_entry_field_addr(worker_core, 0, GROUP_REQ_SEQ_OFF));
        const uint64_t slot_state = gm2reg(worker_slot_state_addr(worker_core, 0));
        if (slot_state != GROUP_REQ_STATE_PENDING || req_seq == 0 || req_seq <= seen_req_seq[lidx]) {
            continue;
        }
        seen_req_seq[lidx] = req_seq;
        queue_worker[q_tail] = lidx;
        queue_seq[q_tail] = req_seq;
        q_tail = (q_tail + 1) % (GROUP_SIZE * 8);
    }

    while (q_head != q_tail) {
        const int lidx = queue_worker[q_head];
        const uint64_t req_seq = queue_seq[q_head];
        const int worker_core = core_id_in_group(gid, lidx);
        const uint64_t slot_req_seq = gm2reg(worker_request_entry_field_addr(worker_core, 0, GROUP_REQ_SEQ_OFF));
        const uint64_t slot_state = gm2reg(worker_slot_state_addr(worker_core, 0));
        if (slot_req_seq != req_seq || slot_state != GROUP_REQ_STATE_PENDING || inflight_seq[lidx] != 0) {
            q_head = (q_head + 1) % (GROUP_SIZE * 8);
            continue;
        }
        const uint64_t target_node = gm2reg(worker_request_entry_field_addr(worker_core, 0, GROUP_NODE_OFF));
        if (target_node >= NUM_MEMORY_NODES || inflight_per_node[target_node] >= static_cast<uint64_t>(GROUP_MAX_INFLIGHT_PER_NODE)) {
            break;
        }
        inflight_per_node[target_node]++;
        inflight_seq[lidx] = req_seq;
        inflight_node[lidx] = target_node;
        reg2gm(req_seq, worker_request_field_addr(worker_core, GROUP_GRANTED_SEQ_OFF));
        reg2gm(GROUP_REQ_STATE_GRANTED, worker_slot_state_addr(worker_core, 0));
        remote_write_u64(core_id, req_seq, get_core_group_grant_addr(worker_core));
        q_head = (q_head + 1) % (GROUP_SIZE * 8);
    }

    bool all_finished = true;
    bool queue_empty = (q_head == q_tail);
    bool inflight_empty = true;
    for (int lidx = 1; lidx < GROUP_SIZE; ++lidx) {
        const int worker_core = core_id_in_group(gid, lidx);
        if (gm2reg(worker_request_field_addr(worker_core, GROUP_WORKER_FINISHED_OFF)) != 1) {
            all_finished = false;
        }
        if (inflight_seq[lidx] != 0) {
            inflight_empty = false;
        }
        if (gm2reg(worker_slot_state_addr(worker_core, 0)) != GROUP_REQ_STATE_EMPTY) {
            inflight_empty = false;
        }
    }

    bool nodes_drained = true;
    for (int node = 0; node < NUM_MEMORY_NODES; ++node) {
        if (inflight_per_node[node] != 0) {
            nodes_drained = false;
            break;
        }
    }

    if (all_finished) {
        reg2gm(1, group_drain_addr(core_id));
        if (queue_empty && inflight_empty && nodes_drained) {
            reg2gm(1, group_done_addr(core_id));
            for (int lidx = 1; lidx < GROUP_SIZE; ++lidx) {
                remote_write_u64(core_id, 1, group_done_addr(core_id_in_group(gid, lidx)));
            }
        }
    }
}
