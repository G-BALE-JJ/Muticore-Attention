#ifndef SST_GOLEM_SFU_ARITHMETIC_H
#define SST_GOLEM_SFU_ARITHMETIC_H

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <sst/elements/golem/fp16.h>
#include <sst/elements/golem/sfu/lut.h>

namespace SST { namespace Golem {

enum class SFUInternalPrecision { FP32, FP16 };
enum class SFUNonlinearBackend { Libm, Lut };

// Host carriers may be float/double. In FP16 mode every operand and operation
// result is binary16, RNE, with gradual underflow and IEEE infinities/NaNs.
// Libm and exact FP16 ROM share the v2 arithmetic contract. ROM lookup invokes
// no runtime transcendental; LOG/TANH remain explicitly on the libm path.
class SFUArithmetic {
public:
    explicit SFUArithmetic(SFUInternalPrecision precision = SFUInternalPrecision::FP16,
                           SFUNonlinearBackend backend = SFUNonlinearBackend::Libm)
        : precision_(precision), backend_(backend) {
        if(backend==SFUNonlinearBackend::Lut && precision!=SFUInternalPrecision::FP16)
            throw std::invalid_argument("SFU nonlinear_backend=lut requires internal_precision=fp16");
    }

    static SFUNonlinearBackend parseBackend(const std::string& name) {
        if(name=="libm")return SFUNonlinearBackend::Libm;
        if(name=="lut")return SFUNonlinearBackend::Lut;
        throw std::invalid_argument("SFU nonlinear_backend must be libm or lut");
    }
    SFUNonlinearBackend backend() const {return backend_;}
    const char* backendName() const {return backend_==SFUNonlinearBackend::Lut ? "lut" : "libm";}

    static SFUInternalPrecision parse(const std::string& name) {
        if (name == "fp32") return SFUInternalPrecision::FP32;
        if (name == "fp16") return SFUInternalPrecision::FP16;
        throw std::invalid_argument("SFU internal_precision must be fp32 or fp16");
    }
    bool fp16() const { return precision_ == SFUInternalPrecision::FP16; }
    SFUInternalPrecision precision() const { return precision_; }
    const char* name() const { return fp16() ? "fp16" : "fp32"; }

    template<class T> T round(T value) const {
        if (!fp16()) return value;
        const double x = static_cast<double>(value);
        if (!std::isfinite(x) || x == 0.0) return value;
        const double magnitude = std::abs(x);
        if (magnitude >= 65520.0)
            return static_cast<T>(std::copysign(INFINITY, x));
        int exponent = 0;
        std::frexp(magnitude, &exponent);
        // Normal: 11 significant bits. Subnormal: fixed 2^-24 spacing.
        const int shift = std::max(-24, exponent - 11);
        const double scaled = std::ldexp(magnitude, -shift);
        double integral = std::floor(scaled);
        const double fraction = scaled - integral;
        if (fraction > 0.5 || (fraction == 0.5 && static_cast<unsigned>(integral) % 2))
            ++integral;
        // Round directly from the host carrier, avoiding double -> float ->
        // half double-rounding at half-ULP ties (including overflow).
        return static_cast<T>(std::copysign(std::ldexp(integral, shift), x));
    }
    template<class T, class U> auto add(T a, U b) const -> std::common_type_t<T,U> {
        if (fp16()) return round(static_cast<double>(round(a)) + static_cast<double>(round(b)));
        return a + b;
    }
    template<class T, class U> auto sub(T a, U b) const -> std::common_type_t<T,U> {
        if (fp16()) return round(static_cast<double>(round(a)) - static_cast<double>(round(b)));
        return a - b;
    }
    template<class T, class U> auto mul(T a, U b) const -> std::common_type_t<T,U> {
        if (fp16()) return round(static_cast<double>(round(a)) * static_cast<double>(round(b)));
        return a * b;
    }
    template<class T, class U> auto div(T a, U b) const -> std::common_type_t<T,U> {
        if (fp16()) return round(static_cast<double>(round(a)) / static_cast<double>(round(b)));
        return a / b;
    }
    template<class T> T fma(T a,T b,T c) const {
        return fp16() ? round(std::fma(double(round(a)),double(round(b)),double(round(c)))) : std::fma(a,b,c);
    }
    template<class T> T silu(T x) const {
        if(backend_==SFUNonlinearBackend::Lut && fp16()) return round(SFULut::silu(round(x)));
        const T input=fp16() ? round(x) : x;
        const T e=exp(fp16() ? -std::abs(input) : -input);
        const T numerator=fp16() && input<0 ? mul(input,e) : input;
        return mul(numerator,reciprocal(add(T(1),e)));
    }
    template<class T> T exp(T x) const {
        if(backend_==SFUNonlinearBackend::Lut)return round(SFULut::exp(round(x)));
        return fp16() ? round(std::exp(static_cast<double>(round(x)))) : std::exp(x);
    }
    template<class T> T log(T x) const {
        return fp16() ? round(std::log(static_cast<double>(round(x)))) : std::log(x);
    }
    template<class T> T tanh(T x) const {
        return fp16() ? round(std::tanh(static_cast<double>(round(x)))) : std::tanh(x);
    }
    template<class T> T reciprocal(T x) const {
        if(backend_==SFUNonlinearBackend::Lut)return round(SFULut::reciprocal(round(x)));
        return div(T(1), x);
    }
    template<class T> T rsqrt(T x) const {
        if(backend_==SFUNonlinearBackend::Lut)return round(SFULut::root(round(x),true));
        // RSQRT is one SFU operation, with one result rounding.
        return fp16() ? round(1.0 / std::sqrt(static_cast<double>(round(x))))
                      : T(1) / std::sqrt(x);
    }
    template<class T> T sqrt(T x) const {
        if(backend_==SFUNonlinearBackend::Lut)return round(SFULut::root(round(x),false));
        return fp16() ? round(std::sqrt(static_cast<double>(round(x)))) : std::sqrt(x);
    }
    template<class T> T treeSum(std::vector<T> values) const {
        if (values.empty()) return T(0);
        for (size_t n=values.size(); n>1; n=(n+1)/2)
            for (size_t i=0; i<(n+1)/2; ++i)
                values[i]=2*i+1<n ? add(values[2*i],values[2*i+1]) : values[2*i];
        return round(values[0]);
    }
private:
    SFUInternalPrecision precision_;
    SFUNonlinearBackend backend_;
};

} }
#endif
