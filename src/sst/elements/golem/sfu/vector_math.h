#ifndef SST_GOLEM_SFU_VECTOR_MATH_H
#define SST_GOLEM_SFU_VECTOR_MATH_H

#include <cmath>
#include <cstdint>
#include <vector>

#include <sst/elements/golem/fp16.h>
#include <sst/elements/golem/sfu/arithmetic.h>
#include <sst/elements/golem/sfu/norm_compensated.h>

namespace SST {
namespace Golem {

enum class SFUVectorOp : uint8_t {
    Rope = 1,
    RmsNorm = 2,
    Add = 3,
    Multiply = 4,
    Silu = 5,
    SiluMultiply = 6,
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
                                std::vector<double>* output,
                                const SFUArithmetic& arithmetic = SFUArithmetic{},
                                bool treeReduction = true) {
    if (output == nullptr || request.input == nullptr ||
        request.coefficients == nullptr || request.rowWidth == 0 ||
        request.input->empty() ||
        request.input->size() % request.rowWidth != 0) return false;
    const auto& input = *request.input;
    const auto& coefficients = *request.coefficients;
    const size_t rows = input.size() / request.rowWidth;

    if (request.op == SFUVectorOp::Add || request.op == SFUVectorOp::Multiply ||
        request.op == SFUVectorOp::Silu || request.op == SFUVectorOp::SiluMultiply) {
        if (request.op != SFUVectorOp::Silu && coefficients.size() != input.size()) return false;
        output->resize(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            const float x = vectorFp16Value(input[i]);
            // FP16 operands/output; arithmetic uses the selected internal precision.
            // SiLU matches the hardware
            // stage sequence EXP -> add-one -> reciprocal -> multiply.
            float y;
            if (request.op == SFUVectorOp::Silu || request.op == SFUVectorOp::SiluMultiply) {
                if (arithmetic.fp16() && arithmetic.backend()==SFUNonlinearBackend::Lut) {
                    y = arithmetic.silu(x);
                } else {
                    const float exponential = arithmetic.exp(arithmetic.fp16() ? -std::abs(x) : -x);
                    const float denominator = arithmetic.add(1.0f, exponential);
                    const float reciprocal = arithmetic.reciprocal(denominator);
                    y = arithmetic.mul(arithmetic.fp16() && x<0 ? arithmetic.mul(x,exponential) : x, reciprocal);
                }
                if (request.op == SFUVectorOp::SiluMultiply)
                    y = arithmetic.mul(vectorFp16Value(y), vectorFp16Value(coefficients[i]));
            } else {
                const float coefficient = vectorFp16Value(coefficients[i]);
                y = request.op == SFUVectorOp::Add ? arithmetic.add(x, coefficient)
                                                  : arithmetic.mul(x, coefficient);
            }
            (*output)[i] = vectorFp16Value(y);
        }
        return true;
    }

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
                    arithmetic.sub(arithmetic.mul(even, cosine), arithmetic.mul(odd, sine)));
                (*output)[base + dim + 1] = vectorFp16Value(
                    arithmetic.add(arithmetic.mul(even, sine), arithmetic.mul(odd, cosine)));
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
            float squareSum = 0.0f, inputScale=1.f;
            if(arithmetic.fp16()) {
                float maximum=0;for(uint32_t dim=0;dim<request.rowWidth;++dim)maximum=std::max(maximum,std::abs(vectorFp16Value(input[base+dim])));
                int exponent=0;std::frexp(maximum,&exponent);inputScale=std::ldexp(1.f,std::max(0,exponent-1));
            }
            if(arithmetic.fp16() && treeReduction) {
                std::vector<SFUNormPair> p(request.rowWidth);
                for(size_t i=0;i<p.size();++i)p[i]=normSquare(arithmetic,arithmetic.div(vectorFp16Value(input[base+i]),inputScale));
                for(size_t n=p.size();n>1;n=(n+1)/2)
                    for(size_t i=0;i<(n+1)/2;++i)if(2*i+1<n)p[i]=normAdd(arithmetic,p[2*i],p[2*i+1]);else p[i]=p[2*i];
                SFUNormPair m{arithmetic.div(p[0].hi,float(request.rowWidth)),arithmetic.div(p[0].lo,float(request.rowWidth))};
                const float eps=arithmetic.div(arithmetic.div(request.epsilon,inputScale),inputScale);
                m=normAdd(arithmetic,m,{eps,0});
                const auto root=normRoot(arithmetic,m);
                for(size_t i=0;i<p.size();++i)(*output)[base+i]=normGamma(arithmetic,normDivide(arithmetic,arithmetic.div(vectorFp16Value(input[base+i]),inputScale),root),vectorFp16Value(coefficients[i]));
                continue;
            }
            std::vector<float> partials(request.rowWidth);
            for (uint32_t dim = 0; dim < request.rowWidth; ++dim) {
                const float value = vectorFp16Value(input[base + dim]);
                const float scaled=arithmetic.fp16() ? arithmetic.div(value,inputScale) : value;
                partials[dim] = arithmetic.mul(scaled, scaled);
                if (!treeReduction) squareSum = arithmetic.add(squareSum, partials[dim]);
            }
            if (treeReduction) {
                for (size_t size = partials.size(); size > 1; size = (size + 1) / 2)
                    for (size_t i = 0; i < (size + 1) / 2; ++i)
                        partials[i] = 2*i+1 < size ? arithmetic.add(partials[2*i], partials[2*i+1]) : partials[2*i];
                squareSum = partials[0];
            }
            const float epsilon=arithmetic.fp16() ? arithmetic.div(arithmetic.div(request.epsilon,inputScale),inputScale) : request.epsilon;
            const float mean=arithmetic.add(arithmetic.div(squareSum,static_cast<float>(request.rowWidth)),epsilon);
            const float root=arithmetic.fp16() ? arithmetic.sqrt(mean) : arithmetic.rsqrt(mean);
            for (uint32_t dim = 0; dim < request.rowWidth; ++dim) {
                const float value = vectorFp16Value(input[base + dim]);
                const float gamma = vectorFp16Value(coefficients[dim]);
                (*output)[base + dim] = vectorFp16Value(
                    arithmetic.mul(arithmetic.fp16() ? arithmetic.div(arithmetic.div(value,inputScale),root) : arithmetic.mul(value,root), gamma));
            }
        }
        return true;
    }
    return false;
}

} // namespace Golem
} // namespace SST

#endif
