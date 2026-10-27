#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "golem_matmul_runtime.h"
#include "gm_config.h"
#include "lenet5_layout.h"
#include "operators.h"
#include "pipeline_config.h"

namespace conv2_ops {

constexpr int kInChannels = 6;
constexpr int kOutChannels = 16;
constexpr int kConv2OutH = 8;
constexpr int kConv2OutW = 8;
constexpr int kBands = 4;
constexpr int kBandRows = 64;
constexpr int kBandValidRows = 16;
constexpr int kPoolH = 4;
constexpr int kPoolW = 4;
constexpr int kPool1H = 12;
constexpr int kPool1W = 12;
constexpr int kKernel = 5;
constexpr int kConv2K = kInChannels * kKernel * kKernel;
constexpr int kConv2KPad = 192;
constexpr int kConv2KTiles = 3;

inline bool is_conv2_banded_task(const GemmTaskDescriptor& desc) {
    return desc.m == 256 && desc.n == 16 && desc.k == 192 &&
           desc.block_m == 64 && desc.block_n == 16 && desc.block_k == 64 &&
           desc.n_tile == 0 && desc.m_tile >= 0 && desc.m_tile < kBands;
}

inline int conv1_pool_row_node(int pool_h) {
    MatmulRuntimeConfig conv1_cfg = {
        .m = 768,
        .n = 6,
        .k = 64,
        .block_m = 64,
        .block_n = 6,
        .block_k = 64,
    };
    return gemm_data_node_for_task(pool_h, conv1_cfg);
}

inline uint64_t conv1_pool_row_addr(int channel, int pool_h) {
    const int node_idx = conv1_pool_row_node(pool_h);
    const uint64_t elem_index =
        static_cast<uint64_t>(channel) * static_cast<uint64_t>(kPool1H * kPool1W) +
        static_cast<uint64_t>(pool_h) * static_cast<uint64_t>(kPool1W);
    return node_base_addr(node_idx) + lenet5_layout::POOL1_OFF + elem_index * sizeof(float);
}

inline uint64_t conv1_pool_ready_addr(int pool_h) {
    const int node_idx = conv1_pool_row_node(pool_h);
    return node_base_addr(node_idx) + lenet5_layout::POOL1_READY_OFF +
           static_cast<uint64_t>(pool_h) * sizeof(uint64_t);
}

inline void wait_pool1_row_ready(int core_id, int pool_h) {
    const uint64_t local_tmp_gm = gm_addr(core_id, LOCAL_LAYOUT.tmp);
    const uint64_t ready_remote = conv1_pool_ready_addr(pool_h);
    while (true) {
        dma_remote_load_to_gm(core_id, ready_remote, local_tmp_gm, sizeof(uint64_t));
        const uint64_t ready = gm2reg(local_tmp_gm);
        if (ready == 1) {
            return;
        }
        delay_cycles(64);
    }
}

inline void load_pool1_row_to_mm(int core_id, int channel, int pool_h, float* row12) {
    const uint64_t src_remote = conv1_pool_row_addr(channel, pool_h);
    const uint64_t local_tmp_gm = gm_addr(core_id, LOCAL_LAYOUT.tmp);
    dma_remote_load_to_gm(core_id, src_remote, local_tmp_gm, static_cast<uint64_t>(kPool1W) * sizeof(float));
    gm2mm(row12, local_tmp_gm);
}

inline void repack_conv2_task_a(const GemmTaskDescriptor& desc) {
    const int band = desc.m_tile;
    const int oh_base = band * 2;
    const int pool_h_start = oh_base;
    const int pool_h_end = oh_base + kKernel;
    const uint64_t local_row_gm = gm_addr(desc.core_id, LOCAL_LAYOUT.out);
    std::array<float, kPool1W> pool1_row{};
    float pool1_cache[kInChannels][kKernel + 1][kPool1W] = {};
    std::array<float, kConv2KPad> a_row{};

    // Wait once for all rows this band needs, then cache all rows locally.
    for (int ph = pool_h_start; ph <= pool_h_end; ++ph) {
        wait_pool1_row_ready(desc.core_id, ph);
    }
    for (int ic = 0; ic < kInChannels; ++ic) {
        for (int ph = pool_h_start; ph <= pool_h_end; ++ph) {
            load_pool1_row_to_mm(desc.core_id, ic, ph, pool1_row.data());
            const int cached_ph = ph - pool_h_start;
            for (int pw = 0; pw < kPool1W; ++pw) {
                pool1_cache[ic][cached_ph][pw] = pool1_row[pw];
            }
        }
    }

    for (int local_row = 0; local_row < kBandRows; ++local_row) {
        a_row.fill(0.0f);
        if (local_row < kBandValidRows) {
            const int oh = oh_base + (local_row / kConv2OutW);
            const int ow = local_row % kConv2OutW;

            int k_idx = 0;
            for (int ic = 0; ic < kInChannels; ++ic) {
                for (int kh = 0; kh < kKernel; ++kh) {
                    const int cache_ph = (oh - pool_h_start) + kh;
                    for (int kw = 0; kw < kKernel; ++kw) {
                        a_row[k_idx++] = pool1_cache[ic][cache_ph][ow + kw];
                    }
                }
            }
        }

        for (int kt = 0; kt < kConv2KTiles; ++kt) {
            const uint64_t dst_remote =
                desc.a_base_mm +
                static_cast<uint64_t>(kt) * MM_MAT_STRIDE +
                static_cast<uint64_t>(local_row) * static_cast<uint64_t>(desc.block_k) * sizeof(float);
            set_len(static_cast<uint64_t>(desc.block_k) * sizeof(float));
            mm2gm(a_row.data() + static_cast<size_t>(kt) * static_cast<size_t>(desc.block_k), local_row_gm);
            remote_store(local_row_gm, dst_remote);
        }
    }
}

inline void prepare_conv2_weights_and_bias_for_task(const GemmTaskDescriptor& desc) {
    const uint64_t node_base = node_base_addr(desc.data_node_idx);
    const uint64_t src_bpack_base = node_base + lenet5_layout::CONV2_BPACK_OFF;
    const uint64_t src_bias_base = node_base + lenet5_layout::CONV2_BIAS_OFF;
    const uint64_t local_vec_gm = gm_addr(desc.core_id, LOCAL_LAYOUT.vec_in);

    set_len(static_cast<uint64_t>(desc.block_k) * sizeof(float));
    for (int kt = 0; kt < kConv2KTiles; ++kt) {
        for (int nc = 0; nc < kOutChannels; ++nc) {
            const uint64_t vec_off =
                static_cast<uint64_t>(kt * kOutChannels + nc) *
                static_cast<uint64_t>(desc.block_k) * sizeof(float);
            dma_remote_load_to_gm(
                desc.core_id,
                src_bpack_base + vec_off,
                local_vec_gm,
                static_cast<uint64_t>(desc.block_k) * sizeof(float));
            remote_store(local_vec_gm, desc.b_pack_base_mm + vec_off);
        }
    }

    dma_remote_load_to_gm(
        desc.core_id,
        src_bias_base,
        local_vec_gm,
        static_cast<uint64_t>(kOutChannels) * sizeof(float));
    remote_store(local_vec_gm, desc.bias_base_mm);
}

inline void prepare_conv2_weights_and_bias_for_core(int core_id) {
    MatmulRuntimeConfig cfg = {
        .m = 256,
        .n = 16,
        .k = 192,
        .block_m = 64,
        .block_n = 16,
        .block_k = 64,
    };
    const int total_tasks = gemm_total_tasks(cfg);
    for (int task_id = core_id; task_id < total_tasks; task_id += ACTIVE_GEMM_CORES) {
        const GemmTaskDescriptor desc = gemm_task_desc_for_task(core_id, task_id, cfg);
        prepare_conv2_weights_and_bias_for_task(desc);
    }
}

inline void repack_conv2_a_for_core(int core_id) {
    MatmulRuntimeConfig cfg = {
        .m = 256,
        .n = 16,
        .k = 192,
        .block_m = 64,
        .block_n = 16,
        .block_k = 64,
    };
    const int total_tasks = gemm_total_tasks(cfg);
    for (int task_id = core_id; task_id < total_tasks; task_id += ACTIVE_GEMM_CORES) {
        const GemmTaskDescriptor desc = gemm_task_desc_for_task(core_id, task_id, cfg);
        repack_conv2_task_a(desc);
    }
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

inline void load_c_tile(const GemmTaskDescriptor& desc, float* c_tile) {
    const uint64_t local_out_gm = gm_addr(desc.core_id, LOCAL_LAYOUT.out);
    const uint64_t bytes = static_cast<uint64_t>(kBandRows * kOutChannels) * sizeof(float);
    dma_remote_load_to_gm(desc.core_id, desc.c_base_mm, local_out_gm, bytes);
    set_len(bytes);
    gm2mm(c_tile, local_out_gm);
}

inline void relu_store_to_conv2_hbm(const GemmTaskDescriptor& desc, const float* c_tile) {
    const uint64_t local_tmp_gm = gm_addr(desc.core_id, LOCAL_LAYOUT.tmp);
    const uint64_t node_base = node_base_addr(desc.data_node_idx);
    std::array<float, kConv2OutW> row{};
    for (int oc = 0; oc < kOutChannels; ++oc) {
        for (int r = 0; r < 2; ++r) {
            for (int ow = 0; ow < kConv2OutW; ++ow) {
                const int src_row = r * kConv2OutW + ow;
                const float v = c_tile[src_row * kOutChannels + oc];
                row[ow] = (v > 0.0f) ? v : 0.0f;
            }
            const int oh = desc.m_tile * 2 + r;
            const uint64_t elem_index =
                static_cast<uint64_t>(oc) * static_cast<uint64_t>(kConv2OutH * kConv2OutW) +
                static_cast<uint64_t>(oh) * static_cast<uint64_t>(kConv2OutW);
            const uint64_t dst = node_base + lenet5_layout::CONV2_OFF + elem_index * sizeof(float);
            set_len(static_cast<uint64_t>(kConv2OutW) * sizeof(float));
            mm2gm(row.data(), local_tmp_gm);
            remote_store(local_tmp_gm, dst);
        }
    }
}

inline void pool_from_conv2_relu_to_hbm(const GemmTaskDescriptor& desc) {
    const uint64_t local_tmp_gm = gm_addr(desc.core_id, LOCAL_LAYOUT.tmp);
    const uint64_t node_base = node_base_addr(desc.data_node_idx);
    std::array<float, kConv2OutW> r0{};
    std::array<float, kConv2OutW> r1{};
    std::array<float, kPoolW> pooled{};
    for (int oc = 0; oc < kOutChannels; ++oc) {
        const int oh0 = desc.m_tile * 2;
        const int oh1 = oh0 + 1;
        const uint64_t idx0 =
            static_cast<uint64_t>(oc) * static_cast<uint64_t>(kConv2OutH * kConv2OutW) +
            static_cast<uint64_t>(oh0) * static_cast<uint64_t>(kConv2OutW);
        const uint64_t idx1 =
            static_cast<uint64_t>(oc) * static_cast<uint64_t>(kConv2OutH * kConv2OutW) +
            static_cast<uint64_t>(oh1) * static_cast<uint64_t>(kConv2OutW);
        dma_remote_load_to_gm(desc.core_id, node_base + lenet5_layout::CONV2_OFF + idx0 * sizeof(float),
                              local_tmp_gm, static_cast<uint64_t>(kConv2OutW) * sizeof(float));
        set_len(static_cast<uint64_t>(kConv2OutW) * sizeof(float));
        gm2mm(r0.data(), local_tmp_gm);
        dma_remote_load_to_gm(desc.core_id, node_base + lenet5_layout::CONV2_OFF + idx1 * sizeof(float),
                              local_tmp_gm, static_cast<uint64_t>(kConv2OutW) * sizeof(float));
        set_len(static_cast<uint64_t>(kConv2OutW) * sizeof(float));
        gm2mm(r1.data(), local_tmp_gm);
        for (int pw = 0; pw < kPoolW; ++pw) {
            const int ow0 = pw * 2;
            pooled[pw] = std::max(std::max(r0[ow0], r0[ow0 + 1]), std::max(r1[ow0], r1[ow0 + 1]));
        }
        const uint64_t pool_elem_index =
            static_cast<uint64_t>(oc) * static_cast<uint64_t>(kPoolH * kPoolW) +
            static_cast<uint64_t>(desc.m_tile) * static_cast<uint64_t>(kPoolW);
        const uint64_t dst = node_base + lenet5_layout::POOL2_OFF + pool_elem_index * sizeof(float);
        set_len(static_cast<uint64_t>(kPoolW) * sizeof(float));
        mm2gm(pooled.data(), local_tmp_gm);
        remote_store(local_tmp_gm, dst);
    }
}

inline void relu_for_core(int core_id) {
    MatmulRuntimeConfig cfg = {
        .m = 256,
        .n = 16,
        .k = 192,
        .block_m = 64,
        .block_n = 16,
        .block_k = 64,
    };
    const int total_tasks = gemm_total_tasks(cfg);
    for (int task_id = core_id; task_id < total_tasks; task_id += ACTIVE_GEMM_CORES) {
        const GemmTaskDescriptor desc = gemm_task_desc_for_task(core_id, task_id, cfg);
        if (!is_conv2_banded_task(desc)) {
            continue;
        }
        std::array<float, kBandRows * kOutChannels> c_tile{};
        load_c_tile(desc, c_tile.data());
        relu_store_to_conv2_hbm(desc, c_tile.data());
    }
}

inline void pool_for_core(int core_id) {
    MatmulRuntimeConfig cfg = {
        .m = 256,
        .n = 16,
        .k = 192,
        .block_m = 64,
        .block_n = 16,
        .block_k = 64,
    };
    const int total_tasks = gemm_total_tasks(cfg);
    for (int task_id = core_id; task_id < total_tasks; task_id += ACTIVE_GEMM_CORES) {
        const GemmTaskDescriptor desc = gemm_task_desc_for_task(core_id, task_id, cfg);
        if (!is_conv2_banded_task(desc)) {
            continue;
        }
        pool_from_conv2_relu_to_hbm(desc);
    }
}

inline int run_conv2(int core_id, const golem_matmul_op_desc_t& op_desc, golem_dtype_t dtype) {
    if (dtype != GOLEM_DTYPE_FP32) {
        std::fprintf(stderr, "[ERROR] conv2 currently requires fp32 dtype\n");
        return 1;
    }

    MatmulRuntimeConfig cfg = {
        .m = 256,
        .n = 16,
        .k = 192,
        .block_m = 64,
        .block_n = 16,
        .block_k = 64,
    };
    const int total_tasks = gemm_total_tasks(cfg);
    if (core_id >= total_tasks) {
        return 0;
    }

    prepare_conv2_weights_and_bias_for_core(core_id);
    log_stage(core_id, "conv2_im2col", "start");
    repack_conv2_a_for_core(core_id);
    log_stage(core_id, "conv2_im2col", "done");

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

    log_stage(core_id, "conv2_gemm", "start");
    st = golemRunMatmul(kernel, &a_desc, &b_desc, &c_desc);
    if (st != GOLEM_STATUS_OK) {
        std::fprintf(stderr, "[ERROR] golemRunMatmul failed: %s\n", golemGetLastErrorString());
        golemDestroyKernel(kernel);
        log_stage(core_id, "conv2_gemm", "fail");
        return 1;
    }
    log_stage(core_id, "conv2_gemm", "done");

    st = golemDestroyKernel(kernel);
    if (st != GOLEM_STATUS_OK) {
        std::fprintf(stderr, "[ERROR] golemDestroyKernel failed: %s\n", golemGetLastErrorString());
        return 1;
    }

    log_stage(core_id, "conv2_relu", "start");
    relu_for_core(core_id);
    log_stage(core_id, "conv2_relu", "done");

    log_stage(core_id, "pool2", "start");
    pool_for_core(core_id);
    log_stage(core_id, "pool2", "done");

    return 0;
}

}  // namespace conv2_ops
