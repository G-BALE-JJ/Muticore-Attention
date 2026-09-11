#pragma once

#include "gemm_matmul_op.h"
#include "group_ctrl_runtime.h"
#include "request_scheduler_runtime.h"

inline DmaTicket scheduler_submit_read_ticket_slot(
    int core_id,
    int slot,
    uint64_t mm_src_addr,
    uint64_t gm_dst_addr,
    uint64_t bytes,
    uint64_t node
) {
    if (!CTRL_LINK_ENABLED) {
        (void)node;
        return issue_dma_read_ticket_slot(core_id, slot, mm_src_addr, gm_dst_addr, bytes);
    }

    // In control-link mode the manager-side scheduler owns DMA issue. Workers only
    // prepare the completion slot locally, then submit a transaction to the scheduler.
    const DmaTicket ticket = prepare_dma_read_ticket_slot(core_id, slot, bytes);
    const uint64_t request_id = sched_compose_request_id(core_id, slot, ticket.seq, node);
    sched_publish_submit_local(core_id, request_id, mm_src_addr, gm_dst_addr, bytes);
    return ticket;
}

inline DmaTicket scheduler_submit_read_ticket_slot_profiled(
    int core_id,
    int slot,
    uint64_t mm_src_addr,
    uint64_t gm_dst_addr,
    uint64_t bytes,
    uint64_t node,
    GemmKernelStats* stats
) {
    const DmaTicket ticket = scheduler_submit_read_ticket_slot(
        core_id, slot, mm_src_addr, gm_dst_addr, bytes, node);
    (void)stats;
    return ticket;
}

inline void scheduler_submit_read_ticket_pair(
    int core_id,
    int slot_a,
    uint64_t mm_src_a,
    uint64_t gm_dst_a,
    uint64_t bytes_a,
    int slot_b,
    uint64_t mm_src_b,
    uint64_t gm_dst_b,
    uint64_t bytes_b,
    uint64_t node,
    DmaTicket* ticket_a,
    DmaTicket* ticket_b,
    SchedSubmitProfile* submit_profile = nullptr
) {
    if (!CTRL_LINK_ENABLED) {
        const uint64_t begin = read_cycle_counter();
        if (ticket_a != nullptr) {
            *ticket_a = issue_dma_read_ticket_slot(core_id, slot_a, mm_src_a, gm_dst_a, bytes_a);
        }
        if (ticket_b != nullptr) {
            *ticket_b = issue_dma_read_ticket_slot(core_id, slot_b, mm_src_b, gm_dst_b, bytes_b);
        }
        if (submit_profile != nullptr) {
            submit_profile->queue_wait_cycles = 0;
            submit_profile->queue_wait_polls = 0;
            submit_profile->write_cycles = read_cycle_counter() - begin;
        }
        return;
    }

    const uint64_t begin = read_cycle_counter();
    DmaTicket local_a{};
    DmaTicket local_b{};
    if (slot_a == 0 && slot_b == 1) {
        const uint64_t shared_seq = dma_remote_load_prepare_pair_slots(core_id, slot_a, slot_b);
        local_a = {.seq = shared_seq, .bytes = bytes_a, .slot = slot_a};
        local_b = {.seq = shared_seq, .bytes = bytes_b, .slot = slot_b};
    } else {
        local_a = prepare_dma_read_ticket_slot(core_id, slot_a, bytes_a);
        local_b = prepare_dma_read_ticket_slot(core_id, slot_b, bytes_b);
    }

    const SchedSubmitProfile mailbox_profile =
        (slot_a == 0 && slot_b == 1)
            ? sched_publish_submit_pair_compact_local_profiled(
                core_id,
                sched_compose_pair_request_id(core_id, local_a.seq, node),
                mm_src_a,
                gm_dst_a,
                mm_src_b,
                gm_dst_b)
            : sched_publish_submit_pair_local_profiled(
                core_id,
                SchedSubmitEntry{
                    .request_id = sched_compose_request_id(core_id, slot_a, local_a.seq, node),
                    .src_addr = mm_src_a,
                    .dst_addr = gm_dst_a,
                },
                SchedSubmitEntry{
                    .request_id = sched_compose_request_id(core_id, slot_b, local_b.seq, node),
                    .src_addr = mm_src_b,
                    .dst_addr = gm_dst_b,
                });

    if (ticket_a != nullptr) {
        *ticket_a = local_a;
    }
    if (ticket_b != nullptr) {
        *ticket_b = local_b;
    }

    if (submit_profile != nullptr) {
        submit_profile->queue_wait_cycles = mailbox_profile.queue_wait_cycles;
        submit_profile->queue_wait_polls = mailbox_profile.queue_wait_polls;
        const uint64_t total_cycles = read_cycle_counter() - begin;
        submit_profile->write_cycles =
            (total_cycles > mailbox_profile.queue_wait_cycles)
                ? (total_cycles - mailbox_profile.queue_wait_cycles)
                : mailbox_profile.write_cycles;
    }
}

inline void scheduler_complete_ticket(int core_id, int slot, uint64_t seq, uint64_t node) {
    if (!CTRL_LINK_ENABLED) {
        (void)core_id;
        (void)slot;
        (void)seq;
        (void)node;
        return;
    }
    sched_publish_done_local(core_id, sched_compose_request_id(core_id, slot, seq, node));
}

inline void scheduler_complete_ticket_profiled(
    int core_id,
    int slot,
    uint64_t seq,
    uint64_t node,
    GemmKernelStats* stats
) {
    const uint64_t begin = read_cycle_counter();
    scheduler_complete_ticket(core_id, slot, seq, node);
    const uint64_t end = read_cycle_counter();
    if (CTRL_LINK_ENABLED && stats != nullptr) {
        stats->sched_protocol_cycles += (end - begin);
    }
}

inline WaitProfileStats ctrl_request_dma_batch_token_profiled(
    int core_id,
    uint64_t req_seq,
    uint64_t mm_src_addr,
    uint64_t gm_dst_addr,
    uint64_t bytes,
    uint64_t node,
    uint64_t window
) {
    if (CTRL_LINK_ENABLED) {
        (void)core_id;
        (void)req_seq;
        (void)mm_src_addr;
        (void)gm_dst_addr;
        (void)bytes;
        (void)node;
        (void)window;
        return {.wait_cycles = 0, .poll_iters = 0};
    }

    ctrl_publish_request_local(
        core_id,
        req_seq,
        mm_src_addr,
        gm_dst_addr,
        bytes,
        node,
        window);
    return adaptive_wait_eq_profiled(core_id, ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_GRANT_SEQ_OFF), req_seq);
}

inline void ctrl_request_dma_batch_done(int core_id, uint64_t req_seq) {
    if (CTRL_LINK_ENABLED) {
        (void)core_id;
        (void)req_seq;
        return;
    }
    ctrl_publish_done_local(core_id, req_seq);
}

inline void ctrl_manager_runtime_loop(int manager_core_id) {
    if (CTRL_LINK_ENABLED) {
        return;
    }
    group_manager_prepare(manager_core_id);
    while (gm2reg(group_done_addr(manager_core_id)) != 1) {
        group_manager_service(manager_core_id);
        delay_cycles(8);
    }
}

template <typename T>
static inline void gemm_tiled_baseline_ctrl(
    int core_id,
    const GemmTaskDescriptor& desc,
    const GemmTileRuntimeContext& rt,
    GemmKernelStats* stats,
    uint64_t& ctrl_req_seq
) {
    if (GOLEM_NUM_ARRAYS < desc.block_n) {
        printf("[Core %d] [ERROR] numArrays(%d) < block_n(%d), cannot map n_col to array_id\n",
               core_id, GOLEM_NUM_ARRAYS, desc.block_n);
        return;
    }
    if (desc.k_tiles <= 0) {
        printf("[Core %d] [ERROR] invalid k_tiles=%d\n", core_id, desc.k_tiles);
        return;
    }

    bool first_dma_ready_reported = false;
    bool first_before_mvm_reported = false;
    bool first_after_mvm_reported = false;
    bool first_overlap_issue_reported = false;
    bool first_overlap_vec_issue_reported = false;
    bool first_overlap_wait_reported = false;

    for (int n_col = 0; n_col < desc.block_n; ++n_col) {
        configure_output_mode(static_cast<uint32_t>(n_col), 1, true);
    }

    const uint64_t vec_block_bytes = static_cast<uint64_t>(desc.block_n) * VEC_BYTES;
    if (WORKER_COMMAND_PROCESSOR_ENABLED) {
        if (stage_progress_enabled()) {
            printf("[Core %d] [%s] WCP runtime path active\n", core_id, dtype_label<T>());
            fflush(stdout);
        }
        WorkerTaskListHeaderRuntime header = {
            .worker_slot = static_cast<uint32_t>(core_id - TOTAL_GROUPS),
            .task_count = 1,
            .active_worker_cores = static_cast<uint32_t>(ACTIVE_GEMM_CORES),
            .total_groups = static_cast<uint32_t>(TOTAL_GROUPS),
            .data_memory_node_count = static_cast<uint32_t>(DATA_MEMORY_NODE_COUNT),
            .mem_node_size = MEM_NODE_SIZE,
            .m = static_cast<uint32_t>(desc.m),
            .n = static_cast<uint32_t>(desc.n),
            .k = static_cast<uint32_t>(desc.k),
            .hw_input_size = static_cast<uint32_t>(TILE_K),
            .hw_output_size = static_cast<uint32_t>(TILE_M),
            .block_m = static_cast<uint32_t>(desc.block_m),
            .block_n = static_cast<uint32_t>(desc.block_n),
            .block_k = static_cast<uint32_t>(desc.block_k),
            .elem_bytes = static_cast<uint32_t>(ELEM_BYTES),
            .mat_stride_bytes = MM_MAT_STRIDE,
            .vec_stride_bytes = GEMM_VEC_STRIDE_MM,
            .off_gemm_mat_base = OFF_GEMM_MAT_BASE,
            .off_gemm_vec_base = gemm_off_vec_base(default_matmul_runtime_config()),
            .off_gemm_out_base = gemm_off_out_base(default_matmul_runtime_config()),
            .local_mat_ping_gm_addr = rt.local_mat_ping,
            .local_mat_pong_gm_addr = rt.local_mat_pong,
            .local_mat_slot2_gm_addr = rt.local_mat_slot2,
            .local_mat_slot3_gm_addr = rt.local_mat_slot3,
            .local_vec_ping_gm_addr = rt.local_vec_ping,
            .local_vec_pong_gm_addr = rt.local_vec_pong,
            .local_vec_slot2_gm_addr = rt.local_vec_slot2,
            .local_vec_slot3_gm_addr = rt.local_vec_slot3,
            .local_mat_slot_stride_bytes = LOCAL_MAT_BYTES_ALIGNED,
            .local_vec_slot_stride_bytes = LOCAL_VEC_BYTES_ALIGNED,
            .local_slot_count = static_cast<uint32_t>(LOCAL_SLOT_COUNT),
            .local_accum_gm_addr = rt.local_accum,
            .local_out_gm_addr = rt.local_out,
            .finished_mailbox_addr = ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_FINISHED_OFF),
            .a_reuse_n_tiles = static_cast<uint32_t>(A_REUSE_N_TILES),
            .n_group_count = static_cast<uint32_t>(gemm_n_groups(default_matmul_runtime_config())),
            .b_reuse_m_tiles = static_cast<uint32_t>(B_REUSE_M_TILES),
            .m_group_count = static_cast<uint32_t>(gemm_m_groups(default_matmul_runtime_config())),
            .descriptor_start_cycle = 0,
        };
        write_worker_task_list_header(core_id, header);
        mark_exec_window_begin(stats, read_cycle_counter());
        const uint64_t n_loop_begin = read_cycle_counter();
        const uint64_t compute_before = stats->compute_cycles;
        const uint64_t compute_begin = read_cycle_counter();
        run_worker_window_start(gm_addr(core_id, WCP_DESC_GM_ADDR));
        const uint64_t submit_end = read_cycle_counter();
        stats->compute_submit_cycles += (submit_end - compute_begin);
        stats->compute_cycles += (submit_end - compute_begin);
        const uint64_t compute_wait_begin = read_cycle_counter();
        run_worker_window_wait();
        const uint64_t compute_end = read_cycle_counter();
        stats->compute_wait_cycles += (compute_end - compute_wait_begin);
        stats->compute_cycles += (compute_end - compute_wait_begin);
        account_nloop_overhead(stats, n_loop_begin, read_cycle_counter(), compute_before);
        return;
    }

    if (!CTRL_OVERLAP_AB_ENABLED) {

        constexpr int kWindowSpan = 2;
        for (int k = 0; k < desc.k_tiles; k += kWindowSpan) {
            const int window_k_count = std::min(kWindowSpan, desc.k_tiles - k);
            ctrl_req_seq++;
            const uint64_t issue_begin = read_cycle_counter();
            const WaitProfileStats batch_grant_wait = ctrl_request_dma_batch_token_profiled(
                core_id,
                ctrl_req_seq,
                gemm_desc_mat_src_mm(desc, k),
                rt.local_mat,
                static_cast<uint64_t>(window_k_count) * (MAT_BYTES + vec_block_bytes),
                gemm_desc_mat_src_mm(desc, k) / MEM_NODE_SIZE,
                GROUP_GRANT_WINDOW);
            const uint64_t issue_end = read_cycle_counter();
            const uint64_t issue_cycles = issue_end - issue_begin;
            stats->dma_issue_cycles +=
                (issue_cycles > batch_grant_wait.wait_cycles)
                    ? (issue_cycles - batch_grant_wait.wait_cycles)
                    : 0;
            stats->group_wait_cycles += batch_grant_wait.wait_cycles;
            stats->poll_iters += batch_grant_wait.poll_iters;
            mark_exec_window_begin(stats, read_cycle_counter());
            for (int wk = 0; wk < window_k_count; ++wk) {
                const int k_tile = k + wk;
                SchedSubmitProfile submit_profile = {0, 0, 0};
                DmaTicket mat_ticket{};
                DmaTicket vec_ticket{};
                const uint64_t submit_pack_begin = read_cycle_counter();
                scheduler_submit_read_ticket_pair(
                    core_id,
                    0,
                    gemm_desc_mat_src_mm(desc, k_tile),
                    rt.local_mat,
                    MAT_BYTES,
                    1,
                    gemm_desc_vec_src_mm(desc, k_tile, 0),
                    rt.local_vec_in,
                    vec_block_bytes,
                    gemm_desc_mat_src_mm(desc, k_tile) / MEM_NODE_SIZE,
                    &mat_ticket,
                    &vec_ticket,
                    &submit_profile);
                const uint64_t submit_pack_end = read_cycle_counter();
                const uint64_t submit_classified = submit_profile.queue_wait_cycles + submit_profile.write_cycles;
                const uint64_t submit_total = submit_pack_end - submit_pack_begin;
                if (submit_total > submit_classified) {
                    stats->submit_pack_cycles += (submit_total - submit_classified);
                }
                stats->dma_issue_cycles += (submit_profile.queue_wait_cycles + submit_profile.write_cycles);
                stats->issue_block_submitq_cycles += submit_profile.queue_wait_cycles;
                stats->issue_write_submitq_cycles += submit_profile.write_cycles;
                stats->poll_iters += submit_profile.queue_wait_polls;
                wait_dma_read_ticket(core_id, mat_ticket, nullptr, &stats->dma_wait_cycles, &stats->poll_iters);
                scheduler_complete_ticket_profiled(core_id, 0, mat_ticket.seq, gemm_desc_mat_src_mm(desc, k_tile) / MEM_NODE_SIZE, stats);
                wait_dma_read_ticket(core_id, vec_ticket, nullptr, &stats->dma_wait_cycles, &stats->poll_iters);
                scheduler_complete_ticket_profiled(core_id, 1, vec_ticket.seq, gemm_desc_mat_src_mm(desc, k_tile) / MEM_NODE_SIZE, stats);

                const uint64_t batch_end = read_cycle_counter();
                if (!first_dma_ready_reported && stage_progress_enabled()) {
                    printf("[Core %d] [%s] CTRL STAGE_PROGRESS: coupled A/B DMA done k_tile=%d cycle=%" PRIu64 "\n",
                           core_id, dtype_label<T>(), k_tile, batch_end);
                    fflush(stdout);
                    first_dma_ready_reported = true;
                }

                const uint64_t n_loop_begin = read_cycle_counter();
                const uint64_t compute_before = stats->compute_cycles;
                run_mvm_load_matrix_batch_async<T>(rt.local_mat, static_cast<uint64_t>(desc.block_n));
                run_mvm_load_vector_batch_async<T>(rt.local_vec_in, static_cast<uint64_t>(desc.block_n));
                const uint64_t compute_begin = read_cycle_counter();
                if (!first_before_mvm_reported && stage_progress_enabled()) {
                    printf("[Core %d] [%s] CTRL STAGE_PROGRESS: coupled before batch mvm count=%d k_tile=%d cycle=%" PRIu64 "\n",
                           core_id, dtype_label<T>(), desc.block_n, k_tile, compute_begin);
                    fflush(stdout);
                    first_before_mvm_reported = true;
                }

                run_mvm_compute_batch_async<T>(0, static_cast<uint64_t>(desc.block_n));
                const uint64_t submit_end = read_cycle_counter();
                stats->compute_submit_cycles += (submit_end - compute_begin);
                stats->compute_cycles += (submit_end - compute_begin);

                const uint64_t compute_wait_begin = read_cycle_counter();
                run_mvm_compute_batch_wait<T>(0, static_cast<uint64_t>(desc.block_n));
                const uint64_t compute_end = read_cycle_counter();
                stats->compute_wait_cycles += (compute_end - compute_wait_begin);
                stats->compute_cycles += (compute_end - compute_wait_begin);
                if (!first_after_mvm_reported && stage_progress_enabled()) {
                    printf("[Core %d] [%s] CTRL STAGE_PROGRESS: coupled after batch mvm count=%d k_tile=%d cycle=%" PRIu64 "\n",
                           core_id, dtype_label<T>(), desc.block_n, k_tile, compute_end);
                    fflush(stdout);
                    first_after_mvm_reported = true;
                }
                account_nloop_overhead(stats, n_loop_begin, read_cycle_counter(), compute_before);
            }

            ctrl_request_dma_batch_done(core_id, ctrl_req_seq);
        }

        for (int n_col = 0; n_col < desc.block_n; ++n_col) {
            const uint64_t store_begin = read_cycle_counter();
            set_len(VEC_BYTES);
            outputvectorstore(accum_col_addr(desc, rt, n_col), static_cast<uint64_t>(n_col));
            const uint64_t store_end = read_cycle_counter();
            stats->compute_wait_cycles += (store_end - store_begin);
            stats->compute_cycles += (store_end - store_begin);
            configure_output_mode(static_cast<uint32_t>(n_col), 0, true);
        }

        apply_optional_bias_gm_fast_path<T>(core_id);
        profiled_store_c_tile_from_gm(desc, rt, &stats->c_store_cycles, stats);
        return;
    }

    const uint64_t ping_mat = mat_buf_addr(rt, 0);
    const uint64_t pong_mat = mat_buf_addr(rt, 1);
    const uint64_t ping_vec = vec_buf_base_addr(rt, 0);
    const uint64_t pong_vec = vec_buf_base_addr(rt, 1);
    if (ping_mat == pong_mat || ping_vec == pong_vec) {
        printf("[Core %d] [ERROR] CTRL ping-pong buffers overlap: mat(0x%" PRIx64 "/0x%" PRIx64 ") vec(0x%" PRIx64 "/0x%" PRIx64 ")\n",
               core_id, ping_mat, pong_mat, ping_vec, pong_vec);
        return;
    }

    int cur_buf = 0;

    ctrl_req_seq++;
    const WaitProfileStats first_grant_wait = ctrl_request_dma_batch_token_profiled(
        core_id,
        ctrl_req_seq,
        gemm_desc_mat_src_mm(desc, 0),
        mat_buf_addr(rt, cur_buf),
        MAT_BYTES + vec_block_bytes,
        gemm_desc_mat_src_mm(desc, 0) / MEM_NODE_SIZE,
        GROUP_GRANT_WINDOW);
    stats->group_wait_cycles += first_grant_wait.wait_cycles;
    stats->poll_iters += first_grant_wait.poll_iters;
    SchedSubmitProfile first_submit_profile = {0, 0, 0};
    DmaTicket first_mat_ticket{};
    DmaTicket first_vec_ticket{};
    mark_exec_window_begin(stats, read_cycle_counter());
    const uint64_t first_submit_pack_begin = read_cycle_counter();
    scheduler_submit_read_ticket_pair(
        core_id,
        0,
        gemm_desc_mat_src_mm(desc, 0),
        mat_buf_addr(rt, cur_buf),
        MAT_BYTES,
        1,
        gemm_desc_vec_src_mm(desc, 0, 0),
        vec_buf_base_addr(rt, cur_buf),
        vec_block_bytes,
        gemm_desc_mat_src_mm(desc, 0) / MEM_NODE_SIZE,
        &first_mat_ticket,
        &first_vec_ticket,
        &first_submit_profile);
    const uint64_t first_submit_pack_end = read_cycle_counter();
    const uint64_t first_submit_classified = first_submit_profile.queue_wait_cycles + first_submit_profile.write_cycles;
    const uint64_t first_submit_total = first_submit_pack_end - first_submit_pack_begin;
    if (first_submit_total > first_submit_classified) {
        stats->submit_pack_cycles += (first_submit_total - first_submit_classified);
    }
    stats->dma_issue_cycles += (first_submit_profile.queue_wait_cycles + first_submit_profile.write_cycles);
    stats->issue_block_submitq_cycles += first_submit_profile.queue_wait_cycles;
    stats->issue_write_submitq_cycles += first_submit_profile.write_cycles;
    stats->poll_iters += first_submit_profile.queue_wait_polls;
    wait_dma_read_ticket(core_id, first_mat_ticket, nullptr, &stats->dma_wait_cycles, &stats->poll_iters);
    scheduler_complete_ticket_profiled(core_id, 0, first_mat_ticket.seq, gemm_desc_mat_src_mm(desc, 0) / MEM_NODE_SIZE, stats);
    wait_dma_read_ticket(core_id, first_vec_ticket, nullptr, &stats->dma_wait_cycles, &stats->poll_iters);
    scheduler_complete_ticket_profiled(core_id, 1, first_vec_ticket.seq, gemm_desc_mat_src_mm(desc, 0) / MEM_NODE_SIZE, stats);

    const uint64_t first_batch_end = read_cycle_counter();
    if (!first_dma_ready_reported && stage_progress_enabled()) {
        printf("[Core %d] [%s] CTRL STAGE_PROGRESS: A/B DMA done k_tile=%d buf=%d cycle=%" PRIu64 "\n",
               core_id, dtype_label<T>(), 0, cur_buf, first_batch_end);
        fflush(stdout);
        first_dma_ready_reported = true;
    }
    ctrl_request_dma_batch_done(core_id, ctrl_req_seq);

    for (int k = 0; k < desc.k_tiles; ++k) {
        const bool has_next_k = (k + 1) < desc.k_tiles;
        const int next_buf = cur_buf ^ 1;
        uint64_t next_req_seq = 0;
        DmaTicket next_mat_ticket = {.seq = 0, .bytes = 0, .slot = 0};
        DmaTicket next_vec_ticket = {.seq = 0, .bytes = 0, .slot = 1};

        if (has_next_k) {
            next_req_seq = ctrl_req_seq + 1;
            const WaitProfileStats next_grant_wait = ctrl_request_dma_batch_token_profiled(
                core_id,
                next_req_seq,
                gemm_desc_mat_src_mm(desc, k + 1),
                mat_buf_addr(rt, next_buf),
                MAT_BYTES + vec_block_bytes,
                gemm_desc_mat_src_mm(desc, k + 1) / MEM_NODE_SIZE,
                GROUP_GRANT_WINDOW);
            stats->group_wait_cycles += next_grant_wait.wait_cycles;
            stats->poll_iters += next_grant_wait.poll_iters;
            SchedSubmitProfile overlap_submit_profile = {0, 0, 0};
            const uint64_t overlap_submit_pack_begin = read_cycle_counter();
            scheduler_submit_read_ticket_pair(
                core_id,
                0,
                gemm_desc_mat_src_mm(desc, k + 1),
                mat_buf_addr(rt, next_buf),
                MAT_BYTES,
                1,
                gemm_desc_vec_src_mm(desc, k + 1, 0),
                vec_buf_base_addr(rt, next_buf),
                vec_block_bytes,
                gemm_desc_mat_src_mm(desc, k + 1) / MEM_NODE_SIZE,
                &next_mat_ticket,
                &next_vec_ticket,
                &overlap_submit_profile);
            const uint64_t overlap_submit_pack_end = read_cycle_counter();
            const uint64_t overlap_submit_classified = overlap_submit_profile.queue_wait_cycles + overlap_submit_profile.write_cycles;
            const uint64_t overlap_submit_total = overlap_submit_pack_end - overlap_submit_pack_begin;
            if (overlap_submit_total > overlap_submit_classified) {
                stats->submit_pack_cycles += (overlap_submit_total - overlap_submit_classified);
            }
            const uint64_t overlap_issue_cycles = overlap_submit_profile.queue_wait_cycles + overlap_submit_profile.write_cycles;
            stats->overlap_issue_cycles += overlap_issue_cycles;
            stats->ov_issue_block_submitq_cycles += overlap_submit_profile.queue_wait_cycles;
            stats->ov_issue_write_submitq_cycles += overlap_submit_profile.write_cycles;
            stats->poll_iters += overlap_submit_profile.queue_wait_polls;
            if (!first_overlap_issue_reported && stage_progress_enabled()) {
                printf("[Core %d] [%s] CTRL STAGE_PROGRESS: overlap issue next-mat k_tile=%d->%d buf=%d cycle=%" PRIu64 "\n",
                       core_id, dtype_label<T>(), k, k + 1, next_buf, read_cycle_counter());
                fflush(stdout);
                first_overlap_issue_reported = true;
            }
            if (!first_overlap_vec_issue_reported && stage_progress_enabled()) {
                printf("[Core %d] [%s] CTRL STAGE_PROGRESS: overlap issue next-vec k_tile=%d->%d buf=%d cycle=%" PRIu64 "\n",
                       core_id, dtype_label<T>(), k, k + 1, next_buf, read_cycle_counter());
                fflush(stdout);
                first_overlap_vec_issue_reported = true;
            }
            ctrl_req_seq = next_req_seq;
        }

        const uint64_t n_loop_begin = read_cycle_counter();
        const uint64_t compute_before = stats->compute_cycles;
        run_mvm_load_matrix_batch_async<T>(mat_buf_addr(rt, cur_buf), static_cast<uint64_t>(desc.block_n));
        run_mvm_load_vector_batch_async<T>(vec_buf_base_addr(rt, cur_buf), static_cast<uint64_t>(desc.block_n));
        const uint64_t compute_begin = read_cycle_counter();
        if (!first_before_mvm_reported && stage_progress_enabled()) {
            printf("[Core %d] [%s] CTRL STAGE_PROGRESS: before batch mvm count=%d k_tile=%d buf=%d cycle=%" PRIu64 "\n",
                   core_id, dtype_label<T>(), desc.block_n, k, cur_buf, compute_begin);
            fflush(stdout);
            first_before_mvm_reported = true;
        }

        run_mvm_compute_batch_async<T>(0, static_cast<uint64_t>(desc.block_n));
        const uint64_t submit_end = read_cycle_counter();
        stats->compute_submit_cycles += (submit_end - compute_begin);
        stats->compute_cycles += (submit_end - compute_begin);

        const uint64_t compute_wait_begin = read_cycle_counter();
        run_mvm_compute_batch_wait<T>(0, static_cast<uint64_t>(desc.block_n));
        const uint64_t compute_end = read_cycle_counter();
        stats->compute_wait_cycles += (compute_end - compute_wait_begin);
        stats->compute_cycles += (compute_end - compute_wait_begin);
        if (!first_after_mvm_reported && stage_progress_enabled()) {
            printf("[Core %d] [%s] CTRL STAGE_PROGRESS: after batch mvm count=%d k_tile=%d buf=%d cycle=%" PRIu64 "\n",
                   core_id, dtype_label<T>(), desc.block_n, k, cur_buf, compute_end);
            fflush(stdout);
            first_after_mvm_reported = true;
        }
        account_nloop_overhead(stats, n_loop_begin, read_cycle_counter(), compute_before);

        if (!has_next_k) {
            continue;
        }

        const WaitProfileStats mat_wait = dma_remote_load_wait_profiled_slot(core_id, next_mat_ticket.slot, next_mat_ticket.seq);
        stats->overlap_wait_cycles += mat_wait.wait_cycles;
        stats->poll_iters += mat_wait.poll_iters;
        scheduler_complete_ticket_profiled(core_id, next_mat_ticket.slot, next_mat_ticket.seq, gemm_desc_mat_src_mm(desc, k + 1) / MEM_NODE_SIZE, stats);
        if (!first_overlap_wait_reported && stage_progress_enabled()) {
            const uint64_t overlap_wait_done = read_cycle_counter();
            printf("[Core %d] [%s] CTRL STAGE_PROGRESS: overlap wait next-mat ready k_tile=%d buf=%d cycle=%" PRIu64 "\n",
                   core_id, dtype_label<T>(), k + 1, next_buf, overlap_wait_done);
            fflush(stdout);
            first_overlap_wait_reported = true;
        }

        const WaitProfileStats vec_wait = dma_remote_load_wait_profiled_slot(core_id, next_vec_ticket.slot, next_vec_ticket.seq);
        stats->overlap_wait_cycles += vec_wait.wait_cycles;
        stats->poll_iters += vec_wait.poll_iters;
        scheduler_complete_ticket_profiled(core_id, next_vec_ticket.slot, next_vec_ticket.seq, gemm_desc_mat_src_mm(desc, k + 1) / MEM_NODE_SIZE, stats);

        const uint64_t next_batch_end = read_cycle_counter();
        if (!first_dma_ready_reported && stage_progress_enabled()) {
            printf("[Core %d] [%s] CTRL STAGE_PROGRESS: A/B DMA done k_tile=%d buf=%d cycle=%" PRIu64 "\n",
                   core_id, dtype_label<T>(), k + 1, next_buf, next_batch_end);
            fflush(stdout);
            first_dma_ready_reported = true;
        }
        ctrl_request_dma_batch_done(core_id, next_req_seq);
        cur_buf = next_buf;
    }

    for (int n_col = 0; n_col < desc.block_n; ++n_col) {
        const uint64_t store_begin = read_cycle_counter();
        set_len(VEC_BYTES);
        outputvectorstore(accum_col_addr(desc, rt, n_col), static_cast<uint64_t>(n_col));
        const uint64_t store_end = read_cycle_counter();
        stats->compute_wait_cycles += (store_end - store_begin);
        stats->compute_cycles += (store_end - store_begin);
        configure_output_mode(static_cast<uint32_t>(n_col), 0, true);
    }

    apply_optional_bias_gm_fast_path<T>(core_id);
    profiled_store_c_tile_from_gm(desc, rt, &stats->c_store_cycles, stats);
}

template <typename T>
inline void matmul_for_core_ctrl_t(int core_id, const MatmulRuntimeConfig& cfg, const MatmulTensorBindingsT<T>* tensors) {
    ctrl_local_mailbox_init(core_id);
    sched_local_mailbox_init(core_id);

    if (!validate_matmul_call(cfg)) {
        printf("[Core %d] [ERROR] invalid matmul config: M/N/K=(%d,%d,%d), block=(%d,%d,%d)\n",
               core_id, cfg.m, cfg.n, cfg.k, cfg.block_m, cfg.block_n, cfg.block_k);
        return;
    }

    if (GROUP_MANAGER_ENABLED && is_group_leader(core_id)) {
        if (CTRL_LINK_ENABLED) {
            if (runtime_info_enabled()) {
                printf("[Core %d] [%s] CTRL MANAGER delegated to GroupCtrl component\n", core_id, dtype_label<T>());
                fflush(stdout);
            }
            return;
        }
        if (runtime_info_enabled()) {
            printf("[Core %d] [%s] CTRL MANAGER service loop\n", core_id, dtype_label<T>());
            fflush(stdout);
        }
        ctrl_manager_runtime_loop(core_id);
        return;
    }

    const GemmTileRuntimeContext rt = make_gemm_runtime_context(core_id);
    const int total_tasks = gemm_total_tasks(cfg);
    const int wcp_tasks = gemm_total_macro_tasks(cfg);
    const int scheduled_tasks = (WORKER_COMMAND_PROCESSOR_ENABLED && tensors == nullptr) ? wcp_tasks : total_tasks;
    const int worker_slot = gemm_worker_slot_for_core(core_id);
    if (worker_slot < 0 || worker_slot >= scheduled_tasks) {
        ctrl_publish_finished_local(core_id);
        adaptive_wait_eq(ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_GROUP_DONE_OFF), 1);
        return;
    }

    GemmKernelStats stats = {};
    uint64_t ctrl_req_seq = 0;
    const uint64_t total_begin = read_cycle_counter();
    int first_task = -1;
    int last_task = -1;
    int tasks_done = 0;

    if (WORKER_COMMAND_PROCESSOR_ENABLED && tensors == nullptr) {
        const uint64_t desc_base = gm_addr(core_id, WCP_DESC_GM_ADDR);
        for (int task_id = worker_slot; task_id < wcp_tasks; task_id += ACTIVE_GEMM_CORES) {
            if (first_task < 0) {
                first_task = task_id;
            }
            last_task = task_id;
            tasks_done++;
        }
        if (tasks_done > 0) {
            const int barrier_participants = std::min(ACTIVE_GEMM_CORES, wcp_tasks);
            const uint64_t descriptor_start_cycle = synchronize_worker_descriptor_start(
                core_id,
                worker_slot,
                barrier_participants);
            stats.descriptor_barrier_cycle = descriptor_start_cycle;
            const uint64_t task_loop_begin = read_cycle_counter();
            WorkerTaskListHeaderRuntime header = {
                .worker_slot = static_cast<uint32_t>(worker_slot),
                .task_count = static_cast<uint32_t>(tasks_done),
                .active_worker_cores = static_cast<uint32_t>(ACTIVE_GEMM_CORES),
                .total_groups = static_cast<uint32_t>(TOTAL_GROUPS),
                .data_memory_node_count = static_cast<uint32_t>(DATA_MEMORY_NODE_COUNT),
                .mem_node_size = MEM_NODE_SIZE,
                .m = static_cast<uint32_t>(cfg.m),
                .n = static_cast<uint32_t>(cfg.n),
                .k = static_cast<uint32_t>(cfg.k),
                .hw_input_size = static_cast<uint32_t>(TILE_K),
                .hw_output_size = static_cast<uint32_t>(TILE_M),
                .block_m = static_cast<uint32_t>(cfg.block_m),
                .block_n = static_cast<uint32_t>(cfg.block_n),
                .block_k = static_cast<uint32_t>(cfg.block_k),
                .elem_bytes = static_cast<uint32_t>(ELEM_BYTES),
                .mat_stride_bytes = MM_MAT_STRIDE,
                .vec_stride_bytes = GEMM_VEC_STRIDE_MM,
                .off_gemm_mat_base = OFF_GEMM_MAT_BASE,
                .off_gemm_vec_base = gemm_off_vec_base(cfg),
                .off_gemm_out_base = gemm_off_out_base(cfg),
                .local_mat_ping_gm_addr = rt.local_mat_ping,
                .local_mat_pong_gm_addr = rt.local_mat_pong,
                .local_mat_slot2_gm_addr = rt.local_mat_slot2,
                .local_mat_slot3_gm_addr = rt.local_mat_slot3,
                .local_vec_ping_gm_addr = rt.local_vec_ping,
                .local_vec_pong_gm_addr = rt.local_vec_pong,
                .local_vec_slot2_gm_addr = rt.local_vec_slot2,
                .local_vec_slot3_gm_addr = rt.local_vec_slot3,
                .local_mat_slot_stride_bytes = LOCAL_MAT_BYTES_ALIGNED,
                .local_vec_slot_stride_bytes = LOCAL_VEC_BYTES_ALIGNED,
                .local_slot_count = static_cast<uint32_t>(LOCAL_SLOT_COUNT),
                .local_accum_gm_addr = rt.local_accum,
                .local_out_gm_addr = rt.local_out,
                .finished_mailbox_addr = ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_FINISHED_OFF),
                .a_reuse_n_tiles = static_cast<uint32_t>(A_REUSE_N_TILES),
                .n_group_count = static_cast<uint32_t>(gemm_n_groups(cfg)),
                .b_reuse_m_tiles = static_cast<uint32_t>(B_REUSE_M_TILES),
                .m_group_count = static_cast<uint32_t>(gemm_m_groups(cfg)),
                .descriptor_start_cycle = descriptor_start_cycle,
            };
            stats.descriptor_mm2gm_start_cycle = read_cycle_counter();
            write_worker_task_list_header_at(core_id, desc_base, header);
            stats.descriptor_mm2gm_end_cycle = read_cycle_counter();
            const uint64_t task_loop_end = read_cycle_counter();
            stats.task_desc_overhead_cycles += (task_loop_end - task_loop_begin);

            mark_exec_window_begin(&stats, read_cycle_counter());
            const uint64_t n_loop_begin = read_cycle_counter();
            const uint64_t compute_before = stats.compute_cycles;
            const uint64_t compute_begin = read_cycle_counter();
            run_worker_window_start(desc_base);
            const uint64_t submit_end = read_cycle_counter();
            stats.compute_submit_cycles += (submit_end - compute_begin);
            stats.compute_cycles += (submit_end - compute_begin);
            const uint64_t compute_wait_begin = read_cycle_counter();
            run_worker_window_wait();
            const uint64_t compute_end = read_cycle_counter();
            stats.compute_wait_cycles += (compute_end - compute_wait_begin);
            stats.compute_cycles += (compute_end - compute_wait_begin);
            account_nloop_overhead(&stats, n_loop_begin, read_cycle_counter(), compute_before);
        }

        const WaitProfileStats group_wait = adaptive_wait_eq_profiled(
            core_id,
            ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_GROUP_DONE_OFF),
            1);
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
            total_cycles = total_end - total_begin;
        }
        printf("[Core %d] [%s] CTRL LATENCY(cycles): dma_issue=%" PRIu64
               " dma_wait=%" PRIu64
               " dma_total=%" PRIu64
               " compute=%" PRIu64
               " compute_submit=%" PRIu64
               " compute_wait=%" PRIu64
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
               stats.compute_submit_cycles,
               stats.compute_wait_cycles,
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
        printf("[Core %d] [%s] FRONTEND_BREAKDOWN barrier=%" PRIu64
               " descriptor_mm2gm_start=%" PRIu64
               " descriptor_mm2gm_end=%" PRIu64 "\n",
               core_id, dtype_label<T>(), stats.descriptor_barrier_cycle,
               stats.descriptor_mm2gm_start_cycle,
               stats.descriptor_mm2gm_end_cycle);
        return;
    }

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
            gemm_tiled_baseline_ctrl<T>(core_id, desc, rt, &stats, ctrl_req_seq);
        }
    }

    const uint64_t finish_publish_begin = read_cycle_counter();
    ctrl_publish_finished_local(core_id);
    const uint64_t finish_publish_end = read_cycle_counter();
    stats.finish_publish_cycles += (finish_publish_end - finish_publish_begin);
    const WaitProfileStats group_wait = adaptive_wait_eq_profiled(core_id, ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_GROUP_DONE_OFF), 1);
    stats.group_wait_cycles += group_wait.wait_cycles;
    stats.poll_iters += group_wait.poll_iters;

    const uint64_t total_end = read_cycle_counter();
    const uint64_t dma_total_cycles =
        stats.dma_issue_cycles +
        stats.dma_wait_cycles +
        stats.overlap_issue_cycles +
        stats.overlap_wait_cycles;
    printf("[Core %d] [%s] CTRL LATENCY(cycles): dma_issue=%" PRIu64
           " dma_wait=%" PRIu64
           " dma_total=%" PRIu64
           " compute=%" PRIu64
           " compute_submit=%" PRIu64
           " compute_wait=%" PRIu64
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
           stats.compute_submit_cycles,
           stats.compute_wait_cycles,
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
            (exec_window_cycles(stats) != 0)
                ? exec_window_cycles(stats)
                : (((total_end - total_begin) > stats.group_wait_cycles)
                      ? ((total_end - total_begin) - stats.group_wait_cycles)
                      : 0));
    if (runtime_info_enabled()) {
        printf("[Core %d] [%s] CTRL GEMM/MM summary: tasks_done=%d first_task=%d last_task=%d task_stride=%d total_tasks=%d block=(%d,%d,%d) k_tiles=%d mat=0x%" PRIx64 ", vec=0x%" PRIx64 ", out=0x%" PRIx64 "\n",
               core_id, dtype_label<T>(), tasks_done, first_task, last_task, ACTIVE_GEMM_CORES, total_tasks,
               cfg.block_m, cfg.block_n, cfg.block_k, gemm_k_tiles(cfg), rt.local_mat, rt.local_vec_in, rt.local_out);
        fflush(stdout);
    }
}

inline void matmul_ctrl(int M, int N, int K, int block_M, int block_N, int block_K) {
    const int core_id = sched_getcpu();
    if (core_id < 0 || core_id >= TOTAL_CORES) {
        printf("[ERROR] invalid runtime core id=%d, TOTAL_CORES=%d\n", core_id, TOTAL_CORES);
        return;
    }
    const MatmulRuntimeConfig cfg = {.m = M, .n = N, .k = K, .block_m = block_M, .block_n = block_N, .block_k = block_K};
    matmul_for_core_ctrl_t<int32_t>(core_id, cfg, nullptr);
}

inline void matmul_with_tensors_ctrl(int M, int N, int K, int block_M, int block_N, int block_K, const MatmulTensorBindings& tensors) {
    const int core_id = sched_getcpu();
    if (core_id < 0 || core_id >= TOTAL_CORES) {
        printf("[ERROR] invalid runtime core id=%d, TOTAL_CORES=%d\n", core_id, TOTAL_CORES);
        return;
    }
    const MatmulRuntimeConfig cfg = {.m = M, .n = N, .k = K, .block_m = block_M, .block_n = block_N, .block_k = block_K};
    matmul_for_core_ctrl_t<int32_t>(core_id, cfg, &tensors);
}

inline void matmul_ctrl_fp32(int M, int N, int K, int block_M, int block_N, int block_K) {
    const int core_id = sched_getcpu();
    if (core_id < 0 || core_id >= TOTAL_CORES) {
        printf("[ERROR] invalid runtime core id=%d, TOTAL_CORES=%d\n", core_id, TOTAL_CORES);
        return;
    }
    const MatmulRuntimeConfig cfg = {.m = M, .n = N, .k = K, .block_m = block_M, .block_n = block_N, .block_k = block_K};
    matmul_for_core_ctrl_t<float>(core_id, cfg, nullptr);
}

inline void matmul_with_tensors_ctrl_fp32(int M, int N, int K, int block_M, int block_N, int block_K, const MatmulTensorBindingsFP32& tensors) {
    const int core_id = sched_getcpu();
    if (core_id < 0 || core_id >= TOTAL_CORES) {
        printf("[ERROR] invalid runtime core id=%d, TOTAL_CORES=%d\n", core_id, TOTAL_CORES);
        return;
    }
    const MatmulRuntimeConfig cfg = {.m = M, .n = N, .k = K, .block_m = block_M, .block_n = block_N, .block_k = block_K};
    matmul_for_core_ctrl_t<float>(core_id, cfg, &tensors);
}

inline void matmul_ctrl_fp16(int M, int N, int K, int block_M, int block_N, int block_K) {
    const int core_id = sched_getcpu();
    if (core_id < 0 || core_id >= TOTAL_CORES) {
        printf("[ERROR] invalid runtime core id=%d, TOTAL_CORES=%d\n", core_id, TOTAL_CORES);
        return;
    }
    const MatmulRuntimeConfig cfg = {.m = M, .n = N, .k = K, .block_m = block_M, .block_n = block_N, .block_k = block_K};
    matmul_for_core_ctrl_t<float>(core_id, cfg, nullptr);
}

inline void matmul_with_tensors_ctrl_fp16(int M, int N, int K, int block_M, int block_N, int block_K, const MatmulTensorBindingsFP16& tensors) {
    const int core_id = sched_getcpu();
    if (core_id < 0 || core_id >= TOTAL_CORES) {
        printf("[ERROR] invalid runtime core id=%d, TOTAL_CORES=%d\n", core_id, TOTAL_CORES);
        return;
    }
    const MatmulRuntimeConfig cfg = {.m = M, .n = N, .k = K, .block_m = block_M, .block_n = block_N, .block_k = block_K};
    matmul_for_core_ctrl_t<SST::Golem::GolemFp16>(core_id, cfg, &tensors);
}
