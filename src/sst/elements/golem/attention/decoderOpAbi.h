#ifndef SST_GOLEM_DECODER_OP_ABI_H
#define SST_GOLEM_DECODER_OP_ABI_H
#include <cstdint>
namespace SST { namespace Golem {
constexpr uint32_t GOLEM_DECODER_OP_MAGIC = 0x44454331;
enum class DecoderOp : uint32_t { Linear = 1, Add = 2, RmsNorm = 3, Silu = 4, Multiply = 5, Embedding = 6, SiluMultiply = 7 };
// FP16 storage and arithmetic contract; 64-row/64-column linear tiles. Weights
// are output-tile striped across four HBM nodes, never replicated in full GM.
struct DecoderOpDesc {
    uint32_t magic, size_bytes, op, stage;
    uint32_t rows, input_dim, output_dim, lane;
    uint32_t manager, head_dim, epoch, output_tile_begin;
    uint64_t input_addr, input1_addr, output_addr, weights_offset;
    uint64_t node_stride, completion_addr;
};
static_assert(sizeof(DecoderOpDesc) == 96, "decoder ABI size");
constexpr uint64_t DEC_WO_WEIGHT = 0x02800000ull;
constexpr uint64_t DEC_GATE_WEIGHT = 0x03000000ull;
constexpr uint64_t DEC_UP_WEIGHT = 0x04200000ull;
constexpr uint64_t DEC_DOWN_WEIGHT = 0x05200000ull;
constexpr uint64_t DEC_WO = 0x02400000ull, DEC_H1 = 0x02500000ull;
constexpr uint64_t DEC_NORM = 0x02600000ull, DEC_GAMMA = 0x02700000ull;
constexpr uint64_t DEC_DOWN = 0x02a00000ull, DEC_Y = 0x02b00000ull;
constexpr uint64_t DEC_GATE = 0x03800000ull, DEC_UP = 0x03c00000ull;
constexpr uint64_t DEC_ACT = 0x04a00000ull, DEC_GATED = 0x06200000ull;
constexpr uint64_t DEC_DONE = 0x01ff0100ull;
} }
#endif
