#pragma once

#include <array>
#include <cstdint>

#include "golem_matmul_runtime.h"
#include "lenet5_layout.h"
#include "operators.h"
#include "pipeline_config.h"

namespace fc1_ops {

constexpr int kTasks = 4;
constexpr int kInChannels = 16;
constexpr int kPool2H = 4;
constexpr int kPool2W = 4;
constexpr int kSliceK = 64;
constexpr int kOut = 120;
constexpr int kOutPad = 128;
constexpr int kPartialStrideBytes = kOutPad * static_cast<int>(sizeof(float));
constexpr int kMatChunks = 2;

inline MatmulRuntimeConfig conv2_cfg() {
    return MatmulRuntimeConfig{
        .m = 256,
        .n = 16,
        .k = 192,
        .block_m = 64,
        .block_n = 16,
        .block_k = 64,
    };
}

inline int fc1_task_node(int task_id) {
    return gemm_data_node_for_task(task_id, conv2_cfg());
}

inline uint64_t fc1_partial_addr(int task_id) {
    return node_base_addr(fc1_task_node(task_id)) + lenet5_layout::FC1_PARTIAL_OFF +
           static_cast<uint64_t>(task_id) * static_cast<uint64_t>(kPartialStrideBytes);
}

inline uint64_t fc1_ready_addr(int task_id) {
    return node_base_addr(fc1_task_node(task_id)) + lenet5_layout::FC1_READY_OFF +
           static_cast<uint64_t>(task_id) * sizeof(uint64_t);
}

inline uint64_t fc1_l1_ready_addr(int group_id) {
    return node_base_addr(fc1_task_node(group_id * 2)) + lenet5_layout::FC1_READY_OFF +
           static_cast<uint64_t>(kTasks + group_id) * sizeof(uint64_t);
}

inline uint64_t fc1_weight_chunk_addr(int task_id, int chunk_id) {
    const uint64_t mat_bytes = MAT_BYTES;
    const uint64_t task_stride = static_cast<uint64_t>(kMatChunks) * mat_bytes;
    return node_base_addr(fc1_task_node(task_id)) + lenet5_layout::FC1_WSLICE_OFF +
           static_cast<uint64_t>(task_id) * task_stride +
           static_cast<uint64_t>(chunk_id) * mat_bytes;
}

inline uint64_t fc1_bias_addr() {
    return node_base_addr(fc1_task_node(0)) + lenet5_layout::FC1_BIAS_OFF;
}

inline uint64_t fc1_out_addr() {
    return node_base_addr(fc1_task_node(0)) + lenet5_layout::FC1_OUT_OFF;
}

inline uint64_t pool2_value_addr(int task_id, int oc, int pw) {
    const uint64_t elem_index =
        static_cast<uint64_t>(oc) * static_cast<uint64_t>(kPool2H * kPool2W) +
        static_cast<uint64_t>(task_id) * static_cast<uint64_t>(kPool2W) +
        static_cast<uint64_t>(pw);
    return node_base_addr(fc1_task_node(task_id)) + lenet5_layout::POOL2_OFF +
           elem_index * sizeof(float);
}

inline void wait_ready_u64(int core_id, uint64_t remote_addr) {
    const uint64_t local_tmp_gm = gm_addr(core_id, LOCAL_LAYOUT.tmp);
    while (true) {
        dma_remote_load_to_gm(core_id, remote_addr, local_tmp_gm, sizeof(uint64_t));
        if (gm2reg(local_tmp_gm) == 1) {
            return;
        }
        delay_cycles(64);
    }
}

inline void load_partial(int core_id, int from_task_id, float* dst120) {
    const uint64_t local_buf_gm = gm_addr(core_id, LOCAL_LAYOUT.out);
    dma_remote_load_to_gm(
        core_id,
        fc1_partial_addr(from_task_id),
        local_buf_gm,
        static_cast<uint64_t>(kOut) * sizeof(float));
    gm2mm(dst120, local_buf_gm);
}

inline void store_partial(int core_id, int task_id, const float* src120) {
    const uint64_t local_buf_gm = gm_addr(core_id, LOCAL_LAYOUT.out);
    set_len(static_cast<uint64_t>(kOut) * sizeof(float));
    mm2gm(const_cast<float*>(src120), local_buf_gm);
    remote_store(local_buf_gm, fc1_partial_addr(task_id));
}

inline void load_pool2_x_slice(int core_id, int task_id, float* x64) {
    const uint64_t local_vec_gm = gm_addr(core_id, LOCAL_LAYOUT.vec_in);
    std::array<float, 4> row4{};
    for (int oc = 0; oc < kInChannels; ++oc) {
        dma_remote_load_to_gm(
            core_id,
            pool2_value_addr(task_id, oc, 0),
            local_vec_gm,
            static_cast<uint64_t>(kPool2W) * sizeof(float));
        gm2mm(row4.data(), local_vec_gm);
        for (int pw = 0; pw < kPool2W; ++pw) {
            x64[oc * kPool2W + pw] = row4[pw];
        }
    }
}

inline void compute_partial(int core_id, int task_id, float* partial120) {
    std::array<float, kSliceK> x64{};
    std::array<float, TILE_DIM> out64{};
    const uint64_t local_mat_gm = gm_addr(core_id, LOCAL_LAYOUT.mat);
    const uint64_t local_vec_gm = gm_addr(core_id, LOCAL_LAYOUT.vec_in);
    const uint64_t local_out_gm = gm_addr(core_id, LOCAL_LAYOUT.out);

    for (int n = 0; n < kOut; ++n) {
        partial120[n] = 0.0f;
    }

    load_pool2_x_slice(core_id, task_id, x64.data());
    set_len(static_cast<uint64_t>(kSliceK) * sizeof(float));
    mm2gm(x64.data(), local_vec_gm);

    for (int chunk = 0; chunk < kMatChunks; ++chunk) {
        dma_remote_load_to_gm(core_id, fc1_weight_chunk_addr(task_id, chunk), local_mat_gm, MAT_BYTES);
        run_mvm_stage(local_mat_gm, local_vec_gm, local_out_gm);
        set_len(static_cast<uint64_t>(TILE_DIM) * sizeof(float));
        gm2mm(out64.data(), local_out_gm);

        const int out_base = chunk * 64;
        const int valid = (out_base + 64 <= kOut) ? 64 : (kOut - out_base);
        for (int i = 0; i < valid; ++i) {
            partial120[out_base + i] = out64[i];
        }
    }
}

inline void add_inplace(float* dst120, const float* src120) {
    for (int n = 0; n < kOut; ++n) {
        dst120[n] += src120[n];
    }
}

inline void add_bias_relu(int core_id, float* out120) {
    const uint64_t local_buf_gm = gm_addr(core_id, LOCAL_LAYOUT.out);
    std::array<float, kOut> bias{};
    dma_remote_load_to_gm(
        core_id,
        fc1_bias_addr(),
        local_buf_gm,
        static_cast<uint64_t>(kOut) * sizeof(float));
    gm2mm(bias.data(), local_buf_gm);
    for (int n = 0; n < kOut; ++n) {
        const float v = out120[n] + bias[n];
        out120[n] = (v > 0.0f) ? v : 0.0f;
    }
}

inline void store_fc1_out(int core_id, const float* out120) {
    const uint64_t local_buf_gm = gm_addr(core_id, LOCAL_LAYOUT.out);
    set_len(static_cast<uint64_t>(kOut) * sizeof(float));
    mm2gm(const_cast<float*>(out120), local_buf_gm);
    remote_store(local_buf_gm, fc1_out_addr());
}

inline int run_fc1_distributed(int core_id) {
    if (core_id >= kTasks) {
        return 0;
    }

    const int task_id = core_id;
    std::array<float, kOut> partial{};
    std::array<float, kOut> peer{};

    compute_partial(core_id, task_id, partial.data());
    store_partial(core_id, task_id, partial.data());
    remote_write_u64(core_id, 1, fc1_ready_addr(task_id));

    if (task_id == 0) {
        wait_ready_u64(core_id, fc1_ready_addr(1));
        load_partial(core_id, 1, peer.data());
        add_inplace(partial.data(), peer.data());
        store_partial(core_id, 0, partial.data());
        remote_write_u64(core_id, 1, fc1_l1_ready_addr(0));

        wait_ready_u64(core_id, fc1_l1_ready_addr(1));
        load_partial(core_id, 2, peer.data());
        add_inplace(partial.data(), peer.data());
        add_bias_relu(core_id, partial.data());
        store_fc1_out(core_id, partial.data());
    } else if (task_id == 2) {
        wait_ready_u64(core_id, fc1_ready_addr(3));
        load_partial(core_id, 3, peer.data());
        add_inplace(partial.data(), peer.data());
        store_partial(core_id, 2, partial.data());
        remote_write_u64(core_id, 1, fc1_l1_ready_addr(1));
    }

    return 0;
}

}  // namespace fc1_ops
