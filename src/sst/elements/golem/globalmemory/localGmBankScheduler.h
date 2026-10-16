#ifndef GOLEM_LOCAL_GM_BANK_SCHEDULER_H
#define GOLEM_LOCAL_GM_BANK_SCHEDULER_H

#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <stdexcept>
#include <vector>

namespace SST { namespace Golem {
// Physical beat arbitration only. Data sampling/commit and API completion belong
// to GlobalMemory. Two ingress FIFOs: compute/control (0), DMA read+write (1).
class LocalGmBankScheduler {
public:
    struct Config {
        uint32_t banks = 8, lineBytes = 128, responseCycles = 1;
        uint32_t responseCapacity = 32, requestCapacity = 256;
    };
    struct Request {
        uint64_t id, address, length, eligibleCycle;
        uint64_t offset = 0;
    };
    struct Grant {
        uint64_t id, address, offset, bytes, readyCycle;
        uint32_t lane, bank;
        bool last;
    };
    struct Stats {
        std::array<uint64_t, 2> laneBeats{{0, 0}};
        std::vector<uint64_t> bankBeats;
        uint64_t payloadBytes = 0, bankConflictCycles = 0;
        uint64_t responseBlockedLaneCycles = 0, responseHighWater = 0;
    };
    explicit LocalGmBankScheduler(Config c) : config_(c) {
        auto pow2 = [](uint32_t x) { return x && !(x & (x - 1)); };
        if (!pow2(c.banks) || !pow2(c.lineBytes) || !c.responseCycles ||
            !c.responseCapacity || !c.requestCapacity)
            throw std::invalid_argument("invalid local GM bank configuration");
        stats_.bankBeats.resize(c.banks);
    }
    bool enqueue(uint32_t lane, Request r) {
        if (lane >= 2 || !r.length || queued() >= config_.requestCapacity) return false;
        queues_[lane].push_back(r);
        return true;
    }
    std::vector<Grant> step(uint64_t cycle) {
        if (stepped_ && cycle <= lastCycle_)
            throw std::logic_error("GM bank edges must be strictly increasing");
        stepped_ = true; lastCycle_ = cycle;
        while (!responses_.empty() && responses_.front() <= cycle) responses_.pop_front();
        std::vector<Grant> result;
        int grantedBank = -1;
        // Fixed compute priority is the reference HDL client-index priority.
        for (uint32_t lane = 0; lane < 2; ++lane) {
            if (queues_[lane].empty() || queues_[lane].front().eligibleCycle > cycle) continue;
            auto& r = queues_[lane].front();
            const uint64_t address = r.address + r.offset;
            const uint32_t bank = (address / config_.lineBytes) % config_.banks;
            if (static_cast<int>(bank) == grantedBank) { ++stats_.bankConflictCycles; continue; }
            if (responses_.size() >= config_.responseCapacity) {
                ++stats_.responseBlockedLaneCycles; continue;
            }
            const uint64_t bytes = std::min<uint64_t>(r.length - r.offset,
                config_.lineBytes - address % config_.lineBytes);
            Grant g{r.id, address, r.offset, bytes, cycle + config_.responseCycles,
                    lane, bank, r.offset + bytes == r.length};
            result.push_back(g);
            responses_.push_back(g.readyCycle);
            ++stats_.laneBeats[lane]; ++stats_.bankBeats[bank];
            stats_.payloadBytes += bytes;
            stats_.responseHighWater = std::max<uint64_t>(stats_.responseHighWater, responses_.size());
            grantedBank = static_cast<int>(bank);
            r.offset += bytes;
            if (g.last) queues_[lane].pop_front();
        }
        return result;
    }
    size_t queued() const { return queues_[0].size() + queues_[1].size(); }
    bool busy() const { return queued() || !responses_.empty(); }
    const Stats& stats() const { return stats_; }
    const Config& config() const { return config_; }
private:
    Config config_;
    Stats stats_;
    std::array<std::deque<Request>, 2> queues_;
    std::deque<uint64_t> responses_;
    uint64_t lastCycle_ = 0;
    bool stepped_ = false;
};
}} // namespace SST::Golem
#endif
