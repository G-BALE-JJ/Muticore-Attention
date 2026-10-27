#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>

#include "golem_matmul_runtime.h"
#include "gm_config.h"
#include "lenet5_layout.h"
#include "operators.h"
#include "pipeline_config.h"

namespace conv1_ops {

constexpr int kBands = 12;
constexpr int kBandRows = 64;
constexpr int kBandValidRows = 48;
constexpr int kChannels = 6;
constexpr int kOutH = 24;
constexpr int kOutW = 24;
constexpr int kPoolH = 12;
constexpr int kPoolW = 12;

inline MatmulRuntimeConfig conv1_cfg() {
    return MatmulRuntimeConfig{
        .m = 768,
        .n = 6,
        .k = 64,
        .block_m = 64,
        .block_n = 6,
        .block_k = 64,
    };
}

inline void log_stage(int core_id, const char* stage, const char* status) {
    if (core_id != 0) {
        return;
    }
    const uint64_t cyc = read_cycles();
    std::fprintf(stdout, "[MILESTONE] stage=%s status=%s cycle=%llu\n",
                 stage, status, static_cast<unsigned long long>(cyc));
    std::fflush(stdout);
}

inline bool is_conv1_task(const GemmTaskDescriptor& desc) {
    return desc.m == 768 && desc.n == 6 && desc.k == 64 &&
           desc.block_m == 64 && desc.block_n == 6 && desc.block_k == 64 &&
           desc.n_tile == 0 && desc.m_tile >= 0 && desc.m_tile < kBands;
}

inline void load_c_tile(const GemmTaskDescriptor& desc, float* c_tile) {
    const uint64_t local_out_gm = gm_addr(desc.core_id, LOCAL_LAYOUT.out);
    const uint64_t bytes = static_cast<uint64_t>(kBandRows * kChannels) * sizeof(float);
    dma_remote_load_to_gm(desc.core_id, desc.c_base_mm, local_out_gm, bytes);
    set_len(bytes);
    gm2mm(c_tile, local_out_gm);
}

inline void relu_store_to_conv1_hbm(const GemmTaskDescriptor& desc, const float* c_tile) {
    const uint64_t local_tmp_gm = gm_addr(desc.core_id, LOCAL_LAYOUT.tmp);
    const uint64_t node_base = node_base_addr(desc.data_node_idx);
    std::array<float, kOutW> row{};

    for (int oc = 0; oc < kChannels; ++oc) {
        for (int r = 0; r < 2; ++r) {
            for (int ow = 0; ow < kOutW; ++ow) {
                const int src_row = r * kOutW + ow;
                const float v = c_tile[src_row * kChannels + oc];
                row[ow] = (v > 0.0f) ? v : 0.0f;
            }
            const int oh = desc.m_tile * 2 + r;
            const uint64_t elem_index =
                static_cast<uint64_t>(oc) * static_cast<uint64_t>(kOutH * kOutW) +
                static_cast<uint64_t>(oh) * static_cast<uint64_t>(kOutW);
            const uint64_t dst_remote =
                node_base + lenet5_layout::CONV1_OFF + elem_index * sizeof(float);
            set_len(static_cast<uint64_t>(kOutW) * sizeof(float));
            mm2gm(row.data(), local_tmp_gm);
            remote_store(local_tmp_gm, dst_remote);
        }
    }
}

inline void pool_from_conv1_relu_to_pool1(const GemmTaskDescriptor& desc) {
    const uint64_t local_tmp_gm = gm_addr(desc.core_id, LOCAL_LAYOUT.tmp);
    const uint64_t node_base = node_base_addr(desc.data_node_idx);
    std::array<float, kOutW> r0{};
    std::array<float, kOutW> r1{};
    std::array<float, kPoolW> pooled{};

    for (int oc = 0; oc < kChannels; ++oc) {
        const int oh0 = desc.m_tile * 2;
        const int oh1 = oh0 + 1;
        const uint64_t idx0 =
            static_cast<uint64_t>(oc) * static_cast<uint64_t>(kOutH * kOutW) +
            static_cast<uint64_t>(oh0) * static_cast<uint64_t>(kOutW);
        const uint64_t idx1 =
            static_cast<uint64_t>(oc) * static_cast<uint64_t>(kOutH * kOutW) +
            static_cast<uint64_t>(oh1) * static_cast<uint64_t>(kOutW);
        dma_remote_load_to_gm(desc.core_id, node_base + lenet5_layout::CONV1_OFF + idx0 * sizeof(float),
                              local_tmp_gm, static_cast<uint64_t>(kOutW) * sizeof(float));
        set_len(static_cast<uint64_t>(kOutW) * sizeof(float));
        gm2mm(r0.data(), local_tmp_gm);
        dma_remote_load_to_gm(desc.core_id, node_base + lenet5_layout::CONV1_OFF + idx1 * sizeof(float),
                              local_tmp_gm, static_cast<uint64_t>(kOutW) * sizeof(float));
        set_len(static_cast<uint64_t>(kOutW) * sizeof(float));
        gm2mm(r1.data(), local_tmp_gm);

        for (int pw = 0; pw < kPoolW; ++pw) {
            const int ow0 = pw * 2;
            pooled[pw] = std::max(std::max(r0[ow0], r0[ow0 + 1]), std::max(r1[ow0], r1[ow0 + 1]));
        }

        const uint64_t pool_elem_index =
            static_cast<uint64_t>(oc) * static_cast<uint64_t>(kPoolH * kPoolW) +
            static_cast<uint64_t>(desc.m_tile) * static_cast<uint64_t>(kPoolW);
        const uint64_t dst_remote = node_base + lenet5_layout::POOL1_OFF + pool_elem_index * sizeof(float);
        set_len(static_cast<uint64_t>(kPoolW) * sizeof(float));
        mm2gm(pooled.data(), local_tmp_gm);
        remote_store(local_tmp_gm, dst_remote);
    }

    const uint64_t ready_addr =
        node_base + lenet5_layout::POOL1_READY_OFF + static_cast<uint64_t>(desc.m_tile) * sizeof(uint64_t);
    remote_write_u64(desc.core_id, 1, ready_addr);
}

inline void relu_for_core(int core_id) {
    const MatmulRuntimeConfig cfg = conv1_cfg();
    const int total_tasks = gemm_total_tasks(cfg);
    for (int task_id = core_id; task_id < total_tasks; task_id += ACTIVE_GEMM_CORES) {
        const GemmTaskDescriptor desc = gemm_task_desc_for_task(core_id, task_id, cfg);
        if (!is_conv1_task(desc)) {
            continue;
        }
        std::array<float, kBandRows * kChannels> c_tile{};
        load_c_tile(desc, c_tile.data());
        relu_store_to_conv1_hbm(desc, c_tile.data());
    }
}

inline void pool_for_core(int core_id) {
    const MatmulRuntimeConfig cfg = conv1_cfg();
    const int total_tasks = gemm_total_tasks(cfg);
    for (int task_id = core_id; task_id < total_tasks; task_id += ACTIVE_GEMM_CORES) {
        const GemmTaskDescriptor desc = gemm_task_desc_for_task(core_id, task_id, cfg);
        if (!is_conv1_task(desc)) {
            continue;
        }
        pool_from_conv1_relu_to_pool1(desc);
    }
}

inline int run_conv1(int core_id, const golem_matmul_op_desc_t& op_desc, golem_dtype_t dtype) {
    golem_tensor_desc_t a_desc = {
        .data = nullptr,
        .ndim = 2,
        .shape = {op_desc.m, op_desc.k},
        .stride = {op_desc.k, 1},
        .dtype = dtype,
        .layout = GOLEM_LAYOUT_ROW_MAJOR,
    };
    golem_tensor_desc_t b_desc = {
        .data = nullptr,
        .ndim = 2,
        .shape = {op_desc.k, op_desc.n},
        .stride = {op_desc.n, 1},
        .dtype = dtype,
        .layout = GOLEM_LAYOUT_ROW_MAJOR,
    };
    golem_tensor_desc_t c_desc = {
        .data = nullptr,
        .ndim = 2,
        .shape = {op_desc.m, op_desc.n},
        .stride = {op_desc.n, 1},
        .dtype = dtype,
        .layout = GOLEM_LAYOUT_ROW_MAJOR,
    };

    golem_kernel_handle_t kernel = nullptr;
    golem_status_t st = golemCreateMatmulKernel(&op_desc, &kernel);
    if (st != GOLEM_STATUS_OK) {
        std::fprintf(stderr, "[ERROR] golemCreateMatmulKernel failed: %s\n", golemGetLastErrorString());
        return 1;
    }

    log_stage(core_id, "conv1_gemm", "start");
    st = golemRunMatmul(kernel, &a_desc, &b_desc, &c_desc);
    if (st != GOLEM_STATUS_OK) {
        std::fprintf(stderr, "[ERROR] golemRunMatmul failed: %s\n", golemGetLastErrorString());
        golemDestroyKernel(kernel);
        log_stage(core_id, "conv1_gemm", "fail");
        return 1;
    }
    log_stage(core_id, "conv1_gemm", "done");

    st = golemDestroyKernel(kernel);
    if (st != GOLEM_STATUS_OK) {
        std::fprintf(stderr, "[ERROR] golemDestroyKernel failed: %s\n", golemGetLastErrorString());
        return 1;
    }

    log_stage(core_id, "conv1_relu", "start");
    relu_for_core(core_id);
    log_stage(core_id, "conv1_relu", "done");

    log_stage(core_id, "pool1", "start");
    pool_for_core(core_id);
    log_stage(core_id, "pool1", "done");

    return 0;
}

}  // namespace conv1_ops
