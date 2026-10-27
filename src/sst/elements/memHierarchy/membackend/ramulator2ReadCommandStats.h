#ifndef SST_MEMHIERARCHY_RAMULATOR2_READ_COMMAND_STATS_H
#define SST_MEMHIERARCHY_RAMULATOR2_READ_COMMAND_STATS_H

#include <algorithm>
#include <cstdint>
#include <limits>
#include <unordered_map>

namespace SST { namespace MemHierarchy {

// Observation only: never modifies requests, DRAM readiness or scheduling.
struct Ramulator2ReadCommandStats {
    struct Pairs {
        uint64_t count = 0;
        uint64_t gapSum = 0;
        uint64_t minGap = std::numeric_limits<uint64_t>::max();
        uint64_t violations = 0;

        void record(uint64_t gap, uint64_t minimum = 0) {
            ++count;
            gapSum += gap;
            minGap = std::min(minGap, gap);
            if (gap < minimum) ++violations;
        }
    };

    Pairs longPairs;
    Pairs shortPairs;
    Pairs crossPcPairs;
    uint64_t pcReadSequences = 0;
    uint64_t globalReadPairs = 0;

    void recordRead(uint64_t cycle, int pc, int bg, uint64_t shortMinimum,
                    uint64_t longMinimum, bool previousColumnWasRead) {
        // Cross-PC alternation retains its physical-channel, adjacent-column
        // scope. Its denominator is independent of the per-PC L/S pairs.
        if (haveLastGlobalRead_ && previousColumnWasRead) {
            ++globalReadPairs;
            if (pc != lastGlobalPc_) crossPcPairs.record(cycle - lastGlobalCycle_);
        }
        haveLastGlobalRead_ = true;
        lastGlobalPc_ = pc;
        lastGlobalCycle_ = cycle;

        auto found = lastReadByPc_.find(pc);
        if (found == lastReadByPc_.end()) {
            ++pcReadSequences;
        } else {
            const auto gap = cycle - found->second.cycle;
            if (bg == found->second.bg) longPairs.record(gap, longMinimum);
            else shortPairs.record(gap, shortMinimum);
        }
        lastReadByPc_[pc] = {cycle, bg};
    }

private:
    struct Read { uint64_t cycle; int bg; };
    std::unordered_map<int, Read> lastReadByPc_;
    bool haveLastGlobalRead_ = false;
    int lastGlobalPc_ = -1;
    uint64_t lastGlobalCycle_ = 0;
};

}} // namespace SST::MemHierarchy
#endif
