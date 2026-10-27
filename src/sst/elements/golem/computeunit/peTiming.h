#ifndef TILEMC_PE_TIMING_H
#define TILEMC_PE_TIMING_H
#include <sst/elements/golem/isa/precision.h>
#include <stdexcept>
namespace SST { namespace Golem {
struct PeTiming {
    uint32_t packetII, singlePacketLatency, interDotGap;
    uint64_t completionCycles(uint64_t packets) const {
        if (!packets) throw std::invalid_argument("PE dot requires at least one packet");
        return (packets - 1) * packetII + singlePacketLatency;
    }
    // Hardware capability for independent same-mode dots. TE scheduling may
    // impose a larger interval, including waiting for dependent partial sums.
    uint64_t startInterval(uint64_t packets) const {
        if (!packets) throw std::invalid_argument("PE dot requires at least one packet");
        return packets * packetII + interDotGap;
    }
};
inline PeTiming peTiming(XTileMC::ComputeMode mode) {
    switch (mode) {
#define TILEMC_PE_TIMING(code, name, ii, latency, gap) \
        case XTileMC::ComputeMode::code: return {ii, latency, gap};
#include <sst/elements/golem/computeunit/peTiming.def>
#undef TILEMC_PE_TIMING
    }
    throw std::invalid_argument("Unsupported PE timing mode");
}
}}
#endif
