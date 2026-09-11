#ifndef SST_GOLEM_FP16_H
#define SST_GOLEM_FP16_H

#include <cstdint>
#include <cstring>

namespace SST {
namespace Golem {

// IEEE-754 binary16 conversion helpers.  FP16 is used as the storage format;
// the modeled floating-point array accumulates in FP32.
inline float golem_fp16_to_float(uint16_t value) {
    const uint32_t sign = (static_cast<uint32_t>(value & 0x8000u)) << 16;
    const uint32_t exponent = (value >> 10) & 0x1fu;
    const uint32_t fraction = value & 0x03ffu;
    uint32_t bits = 0;

    if (exponent == 0) {
        if (fraction == 0) {
            bits = sign;
        } else {
            uint32_t normalized = fraction;
            int exp = -14;
            while ((normalized & 0x0400u) == 0) {
                normalized <<= 1;
                --exp;
            }
            normalized &= 0x03ffu;
            bits = sign | (static_cast<uint32_t>(exp + 127) << 23) |
                   (normalized << 13);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (fraction << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (fraction << 13);
    }

    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline uint16_t golem_float_to_fp16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
    const uint32_t exponent = (bits >> 23) & 0xffu;
    const uint32_t fraction = bits & 0x7fffffu;

    if (exponent == 0xffu) {
        if (fraction == 0) {
            return static_cast<uint16_t>(sign | 0x7c00u);
        }
        const uint16_t payload = static_cast<uint16_t>(fraction >> 13);
        return static_cast<uint16_t>(sign | 0x7c00u | payload | 0x0200u);
    }

    const int32_t unbiased = static_cast<int32_t>(exponent) - 127;
    if (unbiased > 15) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    if (unbiased < -14) {
        if (unbiased < -24) {
            return sign;
        }
        const uint32_t mantissa = fraction | 0x00800000u;
        const int shift = -unbiased - 14;
        uint32_t half_fraction = mantissa >> (shift + 13);
        const uint32_t remainder_mask = (1u << (shift + 13)) - 1u;
        const uint32_t remainder = mantissa & remainder_mask;
        const uint32_t halfway = 1u << (shift + 12);
        if (remainder > halfway || (remainder == halfway && (half_fraction & 1u))) {
            ++half_fraction;
        }
        return static_cast<uint16_t>(sign | half_fraction);
    }

    uint32_t half_exponent = static_cast<uint32_t>(unbiased + 15);
    uint32_t half_fraction = fraction >> 13;
    const uint32_t remainder = fraction & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half_fraction & 1u))) {
        ++half_fraction;
        if (half_fraction == 0x400u) {
            half_fraction = 0;
            ++half_exponent;
            if (half_exponent >= 0x1fu) {
                return static_cast<uint16_t>(sign | 0x7c00u);
            }
        }
    }
    return static_cast<uint16_t>(sign | (half_exponent << 10) | half_fraction);
}

struct GolemFp16 {
    uint16_t bits = 0;

    GolemFp16() = default;
    GolemFp16(float value) : bits(golem_float_to_fp16(value)) {}
    GolemFp16(double value) : bits(golem_float_to_fp16(static_cast<float>(value))) {}
    GolemFp16(int value) : bits(golem_float_to_fp16(static_cast<float>(value))) {}

    operator float() const { return golem_fp16_to_float(bits); }

    GolemFp16& operator+=(const GolemFp16& rhs) {
        bits = golem_float_to_fp16(static_cast<float>(*this) + static_cast<float>(rhs));
        return *this;
    }
};

static_assert(sizeof(GolemFp16) == 2, "GolemFp16 must remain a 16-bit storage type");

} // namespace Golem
} // namespace SST

#endif
