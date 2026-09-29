#ifndef SST_GOLEM_SFU_VECTOR_MATH_H
#define SST_GOLEM_SFU_VECTOR_MATH_H

#include <cmath>
#include <cstdint>
#include <vector>

#include <sst/elements/golem/fp16.h>

namespace SST {
namespace Golem {

enum class SFUVectorOp : uint8_t {
    Rope = 1,
    RmsNorm = 2,
};

struct SFUVectorRequest {
    // Operands are borrowed only for the synchronous issue call.
    SFUVectorOp op = SFUVectorOp::Rope;
    const std::vector<double>* input = nullptr;
    const std::vector<double>* coefficients = nullptr;
    const std::vector<uint32_t>* positions = nullptr;
    uint32_t rowWidth = 0;
    uint32_t activeWidth = 0;
    uint32_t coefficientStride = 0;
    uint32_t coefficientOffset = 0;
    float epsilon = 1.0e-6f;
};

inline float vectorFp16Value(double value) {
    return golem_fp16_to_float(golem_float_to_fp16(static_cast<float>(value)));
}

inline bool evaluateSFUVectorOp(const SFUVectorRequest& request,
                                std::vector<double>* output) {
    if (output == nullptr || request.input == nullptr ||
        request.coefficients == nullptr || request.rowWidth == 0 ||
        request.input->empty() ||
        request.input->size() % request.rowWidth != 0) return false;
    const auto& input = *request.input;
    const auto& coefficients = *request.coefficients;
    const size_t rows = input.size() / request.rowWidth;

    if (request.op == SFUVectorOp::Rope) {
        if (request.positions == nullptr || request.positions->size() != rows ||
            request.activeWidth == 0 || request.activeWidth > request.rowWidth ||
            (request.activeWidth & 1u) != 0 || request.coefficientStride == 0 ||
            static_cast<uint64_t>(request.coefficientOffset) +
                request.activeWidth > request.coefficientStride ||
            coefficients.size() % request.coefficientStride != 0) return false;
        const size_t tableRows = coefficients.size() / request.coefficientStride;
        for (uint32_t position : *request.positions) {
            if (position >= tableRows) return false;
        }
        *output = input;
        for (size_t row = 0; row < rows; ++row) {
            const size_t base = row * request.rowWidth;
            const size_t tableBase =
                static_cast<size_t>((*request.positions)[row]) *
                    request.coefficientStride + request.coefficientOffset;
            for (uint32_t dim = 0; dim < request.activeWidth; dim += 2) {
                const float even = vectorFp16Value(input[base + dim]);
                const float odd = vectorFp16Value(input[base + dim + 1]);
                const float cosine = vectorFp16Value(coefficients[tableBase + dim]);
                const float sine = vectorFp16Value(coefficients[tableBase + dim + 1]);
                (*output)[base + dim] = vectorFp16Value(
                    even * cosine - odd * sine);
                (*output)[base + dim + 1] = vectorFp16Value(
                    even * sine + odd * cosine);
            }
        }
        return true;
    }

    if (request.op == SFUVectorOp::RmsNorm) {
        if (coefficients.size() != request.rowWidth ||
            !std::isfinite(request.epsilon) || request.epsilon <= 0.0f)
            return false;
        output->resize(input.size());
        for (size_t row = 0; row < rows; ++row) {
            const size_t base = row * request.rowWidth;
            float squareSum = 0.0f;
            for (uint32_t dim = 0; dim < request.rowWidth; ++dim) {
                const float value = vectorFp16Value(input[base + dim]);
                squareSum += value * value;
            }
            const float inverseRms = 1.0f / std::sqrt(
                squareSum / request.rowWidth + request.epsilon);
            for (uint32_t dim = 0; dim < request.rowWidth; ++dim) {
                const float value = vectorFp16Value(input[base + dim]);
                const float gamma = vectorFp16Value(coefficients[dim]);
                (*output)[base + dim] = vectorFp16Value(
                    value * inverseRms * gamma);
            }
        }
        return true;
    }
    return false;
}

} // namespace Golem
} // namespace SST

#endif
