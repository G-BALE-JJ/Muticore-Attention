#ifndef TILEMC_PRECISION_H
#define TILEMC_PRECISION_H
#include <cstdint>
namespace XTileMC {
enum class ComputeMode : uint32_t { Int4=0, Int8=1, Fp16=2, Bf16=3, Tf32=4, Fp32=5, Fp8E4M3=6, Fp8E5M2=7 };
constexpr uint32_t inputBits(ComputeMode m) {
    return m == ComputeMode::Int4 ? 4 :
        (m == ComputeMode::Int8 || m == ComputeMode::Fp8E4M3 || m == ComputeMode::Fp8E5M2) ? 8 :
        (m == ComputeMode::Fp16 || m == ComputeMode::Bf16) ? 16 : 32;
}
constexpr bool integerMode(ComputeMode m) { return m == ComputeMode::Int4 || m == ComputeMode::Int8; }
constexpr uint32_t lanesPerBeat(ComputeMode m) { return 32 / inputBits(m); }
constexpr uint64_t packedRowBytes(uint64_t k, ComputeMode m) { return (k * inputBits(m) + 7) / 8; }
// Functional v1: PE packing/reduction topology, corrected arithmetic, FP32 RN
// accumulation. This is a functional contract, not bit-exact unfixed RTL.
struct PrecisionSpecV02 {
    uint32_t compute_mode = 5;
    uint32_t input_bits = 32;
    uint32_t output_kind = 1; // 0: INT32, 1: FP32
    uint32_t accumulator_bits = 32;
    uint32_t arithmetic_version = 1;
    uint32_t reserved[3] = {};
};
static_assert(sizeof(PrecisionSpecV02) == 32, "Precision wire layout changed");
constexpr bool validPrecision(const PrecisionSpecV02& p) {
    return p.compute_mode <= 7 && p.input_bits == inputBits(static_cast<ComputeMode>(p.compute_mode)) &&
        p.output_kind == (integerMode(static_cast<ComputeMode>(p.compute_mode)) ? 0u : 1u) &&
        p.accumulator_bits == 32 && p.arithmetic_version == 1 &&
        !p.reserved[0] && !p.reserved[1] && !p.reserved[2];
}
}
#endif
