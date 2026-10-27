#ifndef TILEMC_MULTI_PRECISION_CU_H
#define TILEMC_MULTI_PRECISION_CU_H
#include <sst/elements/golem/isa/precision.h>
#include <sst/elements/golem/fp16.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace SST { namespace Golem {
inline float floatFromBits(uint32_t bits) { float v; std::memcpy(&v, &bits, 4); return v; }
inline uint32_t floatToBits(float v) { uint32_t bits; std::memcpy(&bits, &v, 4); return bits; }
inline uint32_t packedElement(const uint8_t* row, size_t k, XTileMC::ComputeMode mode) {
    const uint32_t bits = XTileMC::inputBits(mode);
    if (bits == 4) return (row[k/2] >> ((k%2)*4)) & 15;
    uint32_t result = 0;
    for (uint32_t byte=0; byte<bits/8; ++byte) result |= uint32_t(row[k*(bits/8)+byte]) << (byte*8);
    return result;
}
inline float decodePrecision(uint32_t raw, XTileMC::ComputeMode mode) {
    using M = XTileMC::ComputeMode;
    switch (mode) {
        case M::Fp16: return golem_fp16_to_float(static_cast<uint16_t>(raw));
        case M::Bf16: return floatFromBits(raw << 16);
        case M::Tf32: return floatFromBits(raw & 0xffffe000u);
        case M::Fp32: return floatFromBits(raw);
        case M::Fp8E4M3: case M::Fp8E5M2: {
            const bool e5 = mode == M::Fp8E5M2;
            const int frac = e5 ? 2 : 3, bias = e5 ? 15 : 7;
            const int exponent = (raw & 127) >> frac;
            if (exponent == 0) return 0.0f; // PE flush-to-zero input policy.
            return std::ldexp(float((1 << frac) + (raw & ((1 << frac)-1))), exponent-bias-frac) * ((raw & 128) ? -1.0f : 1.0f);
        }
        default: return 0;
    }
}
class MultiPrecisionComputeUnit {
public:
    uint32_t dot(XTileMC::ComputeMode mode, const uint32_t* a, const uint32_t* b,
                 size_t k, uint32_t seed = 0) const {
        return reduce(mode, [a](size_t i) { return a[i]; },
                      [b](size_t i) { return b[i]; }, k, seed);
    }
    // Each word is one 32-bit input packet. validK masks the last packet's
    // unused lanes; packetCount is explicit and must cover validK elements.
    uint32_t dotPackets(XTileMC::ComputeMode mode, const uint32_t* a,
                        const uint32_t* b, size_t packetCount, size_t validK,
                        uint32_t seed = 0) const {
        const size_t lanes = XTileMC::lanesPerBeat(mode);
        if (packetCount != (validK + lanes - 1) / lanes)
            throw std::invalid_argument("CU packet count does not match valid K");
        const uint32_t bits = XTileMC::inputBits(mode);
        const uint32_t mask = bits == 32 ? 0xffffffffu : (1u << bits) - 1;
        const auto element = [lanes,bits,mask](const uint32_t* words, size_t i) {
            return (words[i/lanes] >> ((i%lanes)*bits)) & mask;
        };
        return reduce(mode, [&](size_t i) { return element(a,i); },
                      [&](size_t i) { return element(b,i); }, validK, seed);
    }
private:
    template<class A, class B>
    uint32_t reduce(XTileMC::ComputeMode mode, A a, B b,
                    size_t k, uint32_t seed) const {
        using M = XTileMC::ComputeMode;
        if (XTileMC::integerMode(mode)) {
            uint32_t acc = seed;
            const int bits = XTileMC::inputBits(mode), sign = 1 << (bits-1), mask = (1 << bits)-1;
            for (size_t i=0; i<k; ++i) {
                const int av = (int(a(i)) & mask) ^ sign, bv = (int(b(i)) & mask) ^ sign;
                acc += uint32_t((av-sign)*(bv-sign)); // Defined modulo-2^32 accumulation.
            }
            return acc;
        }
        float acc0 = floatFromBits(seed), acc1 = 0.0f;
        if (mode == M::Fp16 || mode == M::Bf16) {
            for (size_t i=0; i<k; i+=2) {
                acc0 = add(acc0, multiply(decodePrecision(a(i),mode), decodePrecision(b(i),mode)));
                if (i+1 < k) acc1 = add(acc1, multiply(decodePrecision(a(i+1),mode), decodePrecision(b(i+1),mode)));
            }
            return floatToBits(add(acc0, acc1));
        }
        if (mode == M::Fp8E4M3 || mode == M::Fp8E5M2) {
            for (size_t i=0; i<k; i+=4) {
                const size_t count = std::min(size_t(4),k-i);
                uint32_t av[4]={}, bv[4]={};
                for (size_t lane=0; lane<count; ++lane) { av[lane]=a(i+lane); bv[lane]=b(i+lane); }
                acc0 = add(acc0, dot4(mode,av,bv,count));
            }
            return floatToBits(acc0);
        }
        for (size_t i=0; i<k; ++i) acc0 = add(acc0, multiply(decodePrecision(a(i),mode),decodePrecision(b(i),mode)));
        return floatToBits(acc0);
    }
private:
    static float add(float a,float b) { volatile float v = a+b; return v; }
    static float multiply(float a,float b) { volatile float v = a*b; return v; }
    static float dot4(XTileMC::ComputeMode mode,const uint32_t* a,const uint32_t* b,size_t count) {
        const bool e5 = mode == XTileMC::ComputeMode::Fp8E5M2;
        const int frac=e5?2:3, bias=e5?15:7;
        int exp[4]={}, prod[4]={}, maxexp=-64;
        for (size_t lane=0;lane<count;++lane) {
            const int ea=(a[lane]&127)>>frac, eb=(b[lane]&127)>>frac;
            if (!ea || !eb) continue;
            exp[lane]=ea+eb-2*bias;
            const int ma=((1<<frac)+(a[lane]&((1<<frac)-1))) << (3-frac);
            const int mb=((1<<frac)+(b[lane]&((1<<frac)-1))) << (3-frac);
            prod[lane]=ma*mb;
            maxexp=std::max(maxexp,exp[lane]);
        }
        int total=0;
        for (size_t lane=0;lane<count;++lane) {
            const int shift=maxexp-exp[lane];
            const int term=prod[lane] && shift<16 ? (prod[lane]<<7)>>shift : 0;
            total += ((a[lane]^b[lane])&128) ? -term : term;
        }
        return std::ldexp(float(total),maxexp-13);
    }
};
}}
#endif
