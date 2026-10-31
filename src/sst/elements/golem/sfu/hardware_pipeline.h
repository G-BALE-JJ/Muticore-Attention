#ifndef SST_GOLEM_HARDWARE_PIPELINE_H
#define SST_GOLEM_HARDWARE_PIPELINE_H

#include <algorithm>
#include <cstdint>
#include <deque>

namespace SST { namespace Golem {

// Architectural pipeline model, at the same abstraction level as an MVM array:
// accepted tokens occupy finite hardware state and become visible only when the
// completion event fires. Functional arithmetic is evaluated by the host then.
struct TensorHardwarePipeline {
    uint64_t latency = 1;
    uint64_t initiationInterval = 1;
    uint32_t queueDepth = 1;
    uint64_t nextIssueCycle = 0;
    std::deque<uint64_t> completionCycles;
    uint64_t acceptedTokens = 0;
    uint64_t issueWaitCycles = 0;
    uint64_t backpressureEvents = 0;
    uint64_t maxOccupancy = 0;

    void configure(uint64_t configuredLatency, uint64_t configuredII,
                   uint32_t configuredQueueDepth) {
        latency = std::max<uint64_t>(1, configuredLatency);
        initiationInterval = std::max<uint64_t>(1, configuredII);
        queueDepth = std::max<uint32_t>(1, configuredQueueDepth);
    }

    void retire(uint64_t now) {
        while (!completionCycles.empty() && completionCycles.front() <= now) {
            completionCycles.pop_front();
        }
    }

    bool reserve(uint64_t now, uint64_t* completionCycle) {
        retire(now);
        if (completionCycle == nullptr || completionCycles.size() >= queueDepth) {
            ++backpressureEvents;
            return false;
        }
        const uint64_t issue = std::max(now, nextIssueCycle);
        *completionCycle = issue + latency;
        nextIssueCycle = issue + initiationInterval;
        completionCycles.push_back(*completionCycle);
        ++acceptedTokens;
        issueWaitCycles += issue - now;
        maxOccupancy = std::max<uint64_t>(
            maxOccupancy, completionCycles.size());
        return true;
    }

    uint64_t retryDelay(uint64_t now) {
        retire(now);
        if (completionCycles.size() < queueDepth) return 1;
        return std::max<uint64_t>(1, completionCycles.front() - now + 1);
    }
};

// Arbitration follows successful VPU grants, not wall-clock parity. With II=2,
// a parity policy otherwise gives the same client every eligible issue slot.
// A client with no ready work does not consume its turn or block the other one.
struct SFUSharedVectorArbiter {
    bool grantBased=true, preferRows=true;
    uint64_t rowGrants=0, vectorGrants=0;
    template<class Rows,class Vectors>
    void step(uint64_t now,TensorHardwarePipeline& resource,Rows rows,Vectors vectors) {
        const auto runRows=[&] {
            const auto before=resource.acceptedTokens; rows();
            if(resource.acceptedTokens!=before){rowGrants+=resource.acceptedTokens-before;preferRows=false;}
        };
        const auto runVectors=[&] {
            const auto before=resource.acceptedTokens; vectors();
            if(resource.acceptedTokens!=before){vectorGrants+=resource.acceptedTokens-before;preferRows=true;}
        };
        const bool first=grantBased ? preferRows : bool(now&1);
        if(first){runRows();runVectors();}else{runVectors();runRows();}
    }
};

} }
#endif
