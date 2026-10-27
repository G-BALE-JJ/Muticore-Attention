#pragma once

#include <cstdint>

#ifndef GOLEM_DIM
#define GOLEM_DIM 16
#endif

namespace lenet5_layout {

constexpr int IMAGE_H = 28;
constexpr int IMAGE_W = 28;
constexpr int IMAGE_SIZE = IMAGE_H * IMAGE_W;

constexpr int CONV1_OUT_CH = 6;
constexpr int CONV2_OUT_CH = 16;
constexpr int KERNEL = 5;
constexpr int FC1_OUT = 120;
constexpr int FC2_OUT = 84;
constexpr int FC3_OUT = 10;

constexpr uint64_t HBM_ALIGN = 0x1000;
constexpr uint64_t LAYER_BASE = 0x01000000;
constexpr uint64_t INPUT_OFF = LAYER_BASE;
constexpr uint64_t CONV1_OFF = INPUT_OFF + 0x00002000;
constexpr uint64_t POOL1_OFF = CONV1_OFF + 0x00004000;
constexpr uint64_t POOL1_READY_OFF = POOL1_OFF + 0x00001000;
constexpr uint64_t CONV2_OFF = POOL1_OFF + 0x00002000;
constexpr uint64_t POOL2_OFF = CONV2_OFF + 0x00002000;
constexpr uint64_t FC1_OFF = POOL2_OFF + 0x00001000;
constexpr uint64_t FC2_OFF = FC1_OFF + 0x00001000;
constexpr uint64_t FC3_OFF = FC2_OFF + 0x00001000;
constexpr uint64_t CONV1_IM2COL_OFF = FC3_OFF + 0x00001000;
constexpr uint64_t CONV2_IM2COL_OFF = CONV1_IM2COL_OFF + 0x00010000;

constexpr uint64_t PRELOAD_BASE = 0x01200000;
constexpr uint64_t CONV2_BPACK_OFF = PRELOAD_BASE + 0x00040000;
constexpr uint64_t CONV2_BIAS_OFF = CONV2_BPACK_OFF + 0x00008000;
constexpr uint64_t FC1_WSLICE_OFF = PRELOAD_BASE + 0x00050000;
constexpr uint64_t FC1_BIAS_OFF = PRELOAD_BASE + 0x00070000;
constexpr uint64_t FC1_PARTIAL_OFF = PRELOAD_BASE + 0x00071000;
constexpr uint64_t FC1_READY_OFF = PRELOAD_BASE + 0x00072000;
constexpr uint64_t FC1_OUT_OFF = PRELOAD_BASE + 0x00073000;
constexpr uint64_t FC2_WPACK_OFF = PRELOAD_BASE + 0x00074000;
constexpr uint64_t FC2_BIAS_OFF = PRELOAD_BASE + 0x00085000;
constexpr uint64_t FC2_OUT_OFF = PRELOAD_BASE + 0x00086000;
constexpr uint64_t FC3_WPACK_OFF = PRELOAD_BASE + 0x00087000;
constexpr uint64_t FC3_BIAS_OFF = PRELOAD_BASE + 0x00098000;
constexpr uint64_t FC3_OUT_OFF = PRELOAD_BASE + 0x00099000;

constexpr int B_CONV1_COUNT = CONV1_OUT_CH;
constexpr int W_CONV1_K = KERNEL * KERNEL;
constexpr int W_CONV1_N = CONV1_OUT_CH;
constexpr int B_CONV2_COUNT = CONV2_OUT_CH;
constexpr int W_CONV2_K = CONV1_OUT_CH * KERNEL * KERNEL;
constexpr int W_CONV2_N = CONV2_OUT_CH;
constexpr int B_FC1_COUNT = FC1_OUT;
constexpr int W_FC1_K = 256;
constexpr int W_FC1_N = FC1_OUT;
constexpr int B_FC2_COUNT = FC2_OUT;
constexpr int W_FC2_K = FC1_OUT;
constexpr int W_FC2_N = FC2_OUT;
constexpr int B_FC3_COUNT = FC3_OUT;
constexpr int W_FC3_K = FC2_OUT;
constexpr int W_FC3_N = FC3_OUT;

constexpr uint64_t align_up(uint64_t value, uint64_t align) {
    return ((value + align - 1) / align) * align;
}

struct WeightOffsets {
    uint64_t b_conv1_off;
    uint64_t w_conv1_off;
    uint64_t b_conv2_off;
    uint64_t w_conv2_off;
    uint64_t b_fc1_off;
    uint64_t w_fc1_off;
    uint64_t b_fc2_off;
    uint64_t w_fc2_off;
    uint64_t b_fc3_off;
    uint64_t w_fc3_off;
};

inline WeightOffsets make_weight_offsets() {
    const uint64_t tile = static_cast<uint64_t>(GOLEM_DIM);
    const uint64_t w_conv1_bytes = align_up(static_cast<uint64_t>(W_CONV1_K), tile) * static_cast<uint64_t>(W_CONV1_N) * 4ULL;
    const uint64_t w_conv2_bytes = align_up(static_cast<uint64_t>(W_CONV2_K), tile) * static_cast<uint64_t>(W_CONV2_N) * 4ULL;
    const uint64_t w_fc1_bytes = align_up(static_cast<uint64_t>(W_FC1_K), tile) * static_cast<uint64_t>(W_FC1_N) * 4ULL;
    const uint64_t w_fc2_bytes = align_up(static_cast<uint64_t>(W_FC2_K), tile) * static_cast<uint64_t>(W_FC2_N) * 4ULL;
    const uint64_t w_fc3_bytes = align_up(static_cast<uint64_t>(W_FC3_K), tile) * static_cast<uint64_t>(W_FC3_N) * 4ULL;

    WeightOffsets off{};
    off.b_conv1_off = PRELOAD_BASE;
    off.w_conv1_off = align_up(off.b_conv1_off + static_cast<uint64_t>(B_CONV1_COUNT) * 4ULL, HBM_ALIGN);
    off.b_conv2_off = align_up(off.w_conv1_off + w_conv1_bytes, HBM_ALIGN);
    off.w_conv2_off = align_up(off.b_conv2_off + static_cast<uint64_t>(B_CONV2_COUNT) * 4ULL, HBM_ALIGN);
    off.b_fc1_off = align_up(off.w_conv2_off + w_conv2_bytes, HBM_ALIGN);
    off.w_fc1_off = align_up(off.b_fc1_off + static_cast<uint64_t>(B_FC1_COUNT) * 4ULL, HBM_ALIGN);
    off.b_fc2_off = align_up(off.w_fc1_off + w_fc1_bytes, HBM_ALIGN);
    off.w_fc2_off = align_up(off.b_fc2_off + static_cast<uint64_t>(B_FC2_COUNT) * 4ULL, HBM_ALIGN);
    off.b_fc3_off = align_up(off.w_fc2_off + w_fc2_bytes, HBM_ALIGN);
    off.w_fc3_off = align_up(off.b_fc3_off + static_cast<uint64_t>(B_FC3_COUNT) * 4ULL, HBM_ALIGN);
    (void)w_fc3_bytes;
    return off;
}

}  // namespace lenet5_layout
