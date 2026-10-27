#pragma once

#include <array>
#include <cstdint>

#include "lenet5_layout.h"
#include "operators.h"
#include "pipeline_config.h"

namespace fc23_ops {

constexpr int kVecPad = 128;
constexpr int kInChunks = 2;
constexpr int kOutChunks = 2;

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

inline int node_for_core0_outputs() {
    return gemm_data_node_for_task(0, conv2_cfg());
}

inline uint64_t node_base0() {
    return node_base_addr(node_for_core0_outputs());
}

inline void load_vec_padded(int core_id, uint64_t src_remote, int src_len, float* vec128) {
    for (int i = 0; i < kVecPad; ++i) {
        vec128[i] = 0.0f;
    }
    const uint64_t local_tmp_gm = gm_addr(core_id, LOCAL_LAYOUT.tmp);
    set_len(static_cast<uint64_t>(src_len) * sizeof(float));
    dma_remote_load_to_gm(core_id, src_remote, local_tmp_gm, static_cast<uint64_t>(src_len) * sizeof(float));
    gm2mm(vec128, local_tmp_gm);
}

inline void load_bias(int core_id, uint64_t bias_remote, int out_len, float* bias) {
    const uint64_t local_tmp_gm = gm_addr(core_id, LOCAL_LAYOUT.tmp);
    set_len(static_cast<uint64_t>(out_len) * sizeof(float));
    dma_remote_load_to_gm(core_id, bias_remote, local_tmp_gm, static_cast<uint64_t>(out_len) * sizeof(float));
    gm2mm(bias, local_tmp_gm);
}

inline void store_output(int core_id, uint64_t dst_remote, int out_len, const float* out) {
    const uint64_t local_tmp_gm = gm_addr(core_id, LOCAL_LAYOUT.tmp);
    set_len(static_cast<uint64_t>(out_len) * sizeof(float));
    mm2gm(const_cast<float*>(out), local_tmp_gm);
    remote_store(local_tmp_gm, dst_remote);
}

inline void run_single_fc_layer(
    int core_id,
    uint64_t in_remote,
    int in_len,
    uint64_t w_remote,
    uint64_t b_remote,
    uint64_t out_remote,
    int out_len,
    bool relu) {
    std::array<float, kVecPad> vec{};
    std::array<float, 64> vec_chunk{};
    std::array<float, 64> out_chunk{};
    std::array<float, kVecPad> out_full{};
    std::array<float, kVecPad> bias{};

    load_vec_padded(core_id, in_remote, in_len, vec.data());

    const uint64_t local_mat_gm = gm_addr(core_id, LOCAL_LAYOUT.mat);
    const uint64_t local_vec_gm = gm_addr(core_id, LOCAL_LAYOUT.vec_in);
    const uint64_t local_out_gm = gm_addr(core_id, LOCAL_LAYOUT.out);

    for (int out_chunk_id = 0; out_chunk_id < kOutChunks; ++out_chunk_id) {
        for (int i = 0; i < 64; ++i) {
            out_full[out_chunk_id * 64 + i] = 0.0f;
        }
        for (int in_chunk_id = 0; in_chunk_id < kInChunks; ++in_chunk_id) {
            for (int i = 0; i < 64; ++i) {
                vec_chunk[i] = vec[in_chunk_id * 64 + i];
            }
            set_len(64 * sizeof(float));
            mm2gm(vec_chunk.data(), local_vec_gm);

            const uint64_t w_off =
                static_cast<uint64_t>(out_chunk_id * kInChunks + in_chunk_id) * MAT_BYTES;
            dma_remote_load_to_gm(core_id, w_remote + w_off, local_mat_gm, MAT_BYTES);
            run_mvm_stage(local_mat_gm, local_vec_gm, local_out_gm);

            set_len(64 * sizeof(float));
            gm2mm(out_chunk.data(), local_out_gm);
            for (int i = 0; i < 64; ++i) {
                out_full[out_chunk_id * 64 + i] += out_chunk[i];
            }
        }
    }

    load_bias(core_id, b_remote, out_len, bias.data());
    for (int i = 0; i < out_len; ++i) {
        float v = out_full[i] + bias[i];
        if (relu && v < 0.0f) {
            v = 0.0f;
        }
        out_full[i] = v;
    }

    store_output(core_id, out_remote, out_len, out_full.data());
}

inline int run_fc2_core0(int core_id) {
    if (core_id != 0) {
        return 0;
    }

    const uint64_t base = node_base0();
    run_single_fc_layer(
        core_id,
        base + lenet5_layout::FC1_OUT_OFF,
        120,
        base + lenet5_layout::FC2_WPACK_OFF,
        base + lenet5_layout::FC2_BIAS_OFF,
        base + lenet5_layout::FC2_OUT_OFF,
        84,
        true);

    return 0;
}

inline int run_fc3_core0(int core_id) {
    if (core_id != 0) {
        return 0;
    }

    const uint64_t base = node_base0();

    run_single_fc_layer(
        core_id,
        base + lenet5_layout::FC2_OUT_OFF,
        84,
        base + lenet5_layout::FC3_WPACK_OFF,
        base + lenet5_layout::FC3_BIAS_OFF,
        base + lenet5_layout::FC3_OUT_OFF,
        10,
        false);

    return 0;
}

inline int run_fc23_core0(int core_id) {
    const int st2 = run_fc2_core0(core_id);
    if (st2 != 0) {
        return st2;
    }
    return run_fc3_core0(core_id);
}

}  // namespace fc23_ops
