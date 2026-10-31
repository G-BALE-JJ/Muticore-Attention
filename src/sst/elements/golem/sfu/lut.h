#ifndef SST_GOLEM_SFU_LUT_H
#define SST_GOLEM_SFU_LUT_H

#include <cmath>
#include <cstdint>
#include <limits>
#include <sst/elements/golem/fp16.h>
#include <sst/elements/golem/sfu/lut_tables.h>
#include <sst/elements/golem/sfu/silu_lut_tables.h>

namespace SST { namespace Golem {

// All inputs have already been rounded to binary16 by SFUArithmetic. Float64
// carries exact power-of-two exponent shifts; it does not perform widened
// nonlinear arithmetic. Final output rounding belongs to SFUArithmetic.
class SFULut {
public:
    static constexpr uint64_t siluBytes=SFURom::SiLUBytes;
    static constexpr uint64_t expBytes=sizeof(SFURom::ExpPositiveBase)+sizeof(SFURom::ExpPositiveDelta)+
        sizeof(SFURom::ExpNegativeBase)+sizeof(SFURom::ExpNegativeDelta);
    static constexpr uint64_t expDecodeLatency=1;
    static constexpr const char* implementation="fp16-rom-v3-silu-direct";
    static constexpr uint64_t reciprocalBytes=sizeof(SFURom::Reciprocal);
    static constexpr uint64_t sqrtBytes=sizeof(SFURom::Sqrt);
    static constexpr uint64_t rsqrtBytes=sizeof(SFURom::Rsqrt);
    static uint16_t bits(double x) { return golem_float_to_fp16(static_cast<float>(x)); }
    static double decode(uint16_t x) { return golem_fp16_to_float(x); }
    static double silu(double x) { return decode(SFURom::SiLU[bits(x)]); }
    static double exp(double x) {
        const uint16_t raw=bits(x), mag=raw&0x7fff;
        if(mag>0x7c00) return std::numeric_limits<double>::quiet_NaN();
        if(raw&0x8000) {
            if(mag<SFURom::ExpNegativeBegin)return 1.;
            if(mag>=SFURom::ExpNegativeEnd)return 0.;
            const uint32_t index=mag-SFURom::ExpNegativeBegin;
            return decode(SFURom::ExpNegativeBase[index/SFURom::ExpBlockEntries]-SFURom::ExpNegativeDelta[index]);
        }
        if(mag<SFURom::ExpPositiveBegin)return 1.;
        if(mag>=SFURom::ExpPositiveEnd)return std::numeric_limits<double>::infinity();
        const uint32_t index=mag-SFURom::ExpPositiveBegin;
        return decode(SFURom::ExpPositiveBase[index/SFURom::ExpBlockEntries]+SFURom::ExpPositiveDelta[index]);
    }
    static double reciprocal(double x) {
        const uint16_t raw=bits(x),mag=raw&0x7fff;
        if(mag>0x7c00)return std::numeric_limits<double>::quiet_NaN();
        if(mag==0)return std::copysign(std::numeric_limits<double>::infinity(),x);
        if(mag==0x7c00)return std::copysign(0.,x);
        int exponent;uint16_t mantissa;normalize(mag,exponent,mantissa);
        const uint32_t bank=exponent>=14 ? exponent-13 : 0;
        return std::copysign(std::ldexp(decode(SFURom::Reciprocal[bank*1024+mantissa]),-exponent),x);
    }
    static double root(double x,bool inverse) {
        const uint16_t raw=bits(x),mag=raw&0x7fff;
        if(mag>0x7c00 || ((raw&0x8000) && mag))return std::numeric_limits<double>::quiet_NaN();
        if(mag==0)return inverse ? std::copysign(std::numeric_limits<double>::infinity(),x) : x;
        if(mag==0x7c00)return inverse ? 0. : x;
        int exponent;uint16_t mantissa;normalize(mag,exponent,mantissa);
        const int parity=(exponent%2+2)%2,shift=(exponent-parity)/2;
        const uint32_t index=parity*1024+mantissa;
        return std::ldexp(decode(inverse ? SFURom::Rsqrt[index] : SFURom::Sqrt[index]),inverse ? -shift : shift);
    }
private:
    static void normalize(uint16_t raw,int& exponent,uint16_t& mantissa) {
        exponent=(raw>>10)-15;mantissa=raw&0x3ff;
        if((raw>>10)==0) {
            exponent=-14;
            while(mantissa<1024){mantissa<<=1;--exponent;}
            mantissa&=0x3ff;
        }
    }
};

} }
#endif
