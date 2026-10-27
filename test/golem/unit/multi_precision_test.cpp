#include <sst/elements/golem/computeunit/multiPrecisionComputeUnit.h>
#include <sst/elements/golem/computeunit/peTiming.h>
#include <cassert>
#include <iostream>
using namespace SST::Golem;
using M = XTileMC::ComputeMode;
int main() {
    MultiPrecisionComputeUnit cu;
    for (uint32_t code=0; code<8; ++code) {
        M mode=static_cast<M>(code);
        XTileMC::PrecisionSpecV02 spec{code,XTileMC::inputBits(mode),XTileMC::integerMode(mode)?0u:1u,32,1,{0,0,0}};
        const uint32_t expectedL[] = {5,4,5,5,5,6,5,5};
        const uint32_t expectedGap[] = {0,0,1,1,0,0,0,0};
        const auto timing = peTiming(mode);
        assert(timing.packetII == 1);
        for (uint64_t packets : {1u, 2u, 16u, 64u, 96u}) {
            assert(timing.completionCycles(packets) == packets - 1 + expectedL[code]);
            assert(timing.startInterval(packets) == packets + expectedGap[code]);
            // Earliest second output for two independent dots.
            assert(timing.startInterval(packets) + timing.completionCycles(packets)
                   == 2*packets + expectedGap[code] + expectedL[code] - 1);
        }
        bool rejectedEmpty = false;
        try { timing.completionCycles(0); } catch (const std::invalid_argument&) { rejectedEmpty = true; }
        assert(rejectedEmpty);
        assert(XTileMC::validPrecision(spec));
        spec.input_bits=3; assert(!XTileMC::validPrecision(spec));
        const uint32_t one[]={1,1,0x3c00,0x3f80,0x3f800000,0x3f800000,0x38,0x3c};
        for (size_t k : {1u,2u,3u,4u,7u,8u,9u,64u,65u}) {
            std::vector<uint32_t> a(k,one[code]),b(k,one[code]);
            const uint32_t seed=XTileMC::integerMode(mode)?7:floatToBits(7.0f);
            uint32_t bits=cu.dot(mode,a.data(),b.data(),k,seed);
            const uint32_t lanes=XTileMC::lanesPerBeat(mode), width=XTileMC::inputBits(mode);
            const size_t packets=(k+lanes-1)/lanes;
            std::vector<uint32_t> ap(packets,0xffffffffu),bp(packets,0xffffffffu);
            const uint32_t mask=width==32?0xffffffffu:(1u<<width)-1;
            for (size_t i=0;i<k;++i) {
                const uint32_t shift=(i%lanes)*width;
                ap[i/lanes]=(ap[i/lanes]&~(mask<<shift))|((a[i]&mask)<<shift);
                bp[i/lanes]=(bp[i/lanes]&~(mask<<shift))|((b[i]&mask)<<shift);
            }
            assert(cu.dotPackets(mode,ap.data(),bp.data(),packets,k,seed)==bits);
            if (XTileMC::integerMode(mode)) assert(bits==k+7);
            else assert(floatFromBits(bits)==k+7);
        }
    }
    uint32_t a[]={8,7,15},b[]={7,8,1}; // INT4: -8*7 + 7*-8 + -1*1 = -113.
    assert(cu.dot(M::Int4,a,b,3)==uint32_t(-113));
    uint32_t neg[]={128}, pos[]={127};
    assert(cu.dot(M::Int8,neg,pos,1)==uint32_t(-16256));
    uint32_t one[]={1};
    assert(cu.dot(M::Int8,one,one,1,0x7fffffff)==0x80000000u);
    uint32_t half[]={0x3c00,0x3c00,0x3c00,0x3c00};
    assert(floatFromBits(cu.dot(M::Fp16,half,half,4,floatToBits(16777216.0f)))==16777218.0f);
    uint32_t bf[]={0x3f80,0x3f80};
    assert(floatFromBits(cu.dot(M::Bf16,bf,bf,2))==2.0f);
    uint32_t cancel_a[]={floatToBits(1),floatToBits(-1)}, cancel_b[]={floatToBits(1),floatToBits(1)};
    assert(cu.dot(M::Fp32,cancel_a,cancel_b,2)==0);
    uint32_t tf[]={0x3f801fff},tfb[]={0x3f800000};
    assert(cu.dot(M::Tf32,tf,tfb,1)==0x3f800000);
    uint32_t f8a[]={0x78,0x78,0x08,0x08}, f8b[]={0x78,0xf8,0x08,0x08};
    assert(cu.dot(M::Fp8E4M3,f8a,f8b,4)==0); // Small lanes fall outside shared alignment window.
    uint8_t packed[]={0xf8,0x07};
    assert(packedElement(packed,0,M::Int4)==8);
    assert(packedElement(packed,1,M::Int4)==15);
    assert(packedElement(packed,2,M::Int4)==7);
    std::cout << "[PASS] RTL PE timing (including two independent dots), eight PE modes, packed tails, seed topology, INT32 wrap, cancellation, TF32 and FP8 alignment\n";
}
