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

} }
#endif
