#ifndef _GOLEM_ATTENTION_CLUSTER_H
#define _GOLEM_ATTENTION_CLUSTER_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <map>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

namespace SST {
namespace Golem {

class AttentionGenerationCancelRegistry {
public:
    static uint64_t token(uint64_t owner, uint64_t generation) {
        return (owner << 32) ^ generation;
    }
    static void cancel(uint64_t owner, uint64_t generation,
                       size_t pendingWrites = 1) {
        if (pendingWrites != 0) {
            cancelled_[token(owner, generation)] += pendingWrites;
        }
    }
    static bool cancelledToken(uint64_t generationToken) {
        return consume(generationToken);
    }
    static bool discardToken(uint64_t generationToken) {
        return consume(generationToken);
    }

private:
    static bool consume(uint64_t generationToken) {
        auto it = cancelled_.find(generationToken);
        if (generationToken == 0 || it == cancelled_.end()) return false;
        if (it->second <= 1) cancelled_.erase(it);
        else --it->second;
        return true;
    }
    inline static std::unordered_map<uint64_t, size_t> cancelled_;
};

enum class AttentionClusterTrafficClass : uint8_t {
    Legacy = 0,
    QkKMatrix,
    QkQPair,
    QkScoreOut,
    SequentialQkScoreOut,
    SequentialQkInputScatter,
    PvVMatrix,
    PvPInput,
    SequentialPvInput,
    SequentialPvInputScatter,
    SequentialPvORestore,
    SequentialPvOOutput,
    PvOFinalDrain,
    ScoreSfuRead,
    SfuPWrite,
    OScale,
    OAccumulate,
    Count,
};

inline const char* attentionClusterTrafficClassName(
        AttentionClusterTrafficClass trafficClass) {
    switch (trafficClass) {
    case AttentionClusterTrafficClass::Legacy: return "legacy";
    case AttentionClusterTrafficClass::QkKMatrix: return "qk_k_matrix";
    case AttentionClusterTrafficClass::QkQPair: return "qk_q_pair";
    case AttentionClusterTrafficClass::QkScoreOut: return "qk_score_out";
    case AttentionClusterTrafficClass::SequentialQkScoreOut:
        return "sequential_qk_score_out";
    case AttentionClusterTrafficClass::SequentialQkInputScatter:
        return "sequential_qk_input_scatter";
    case AttentionClusterTrafficClass::PvVMatrix: return "pv_v_matrix";
    case AttentionClusterTrafficClass::PvPInput: return "pv_p_input";
    case AttentionClusterTrafficClass::SequentialPvInput:
        return "sequential_pv_input";
    case AttentionClusterTrafficClass::SequentialPvInputScatter:
        return "sequential_pv_input_scatter";
    case AttentionClusterTrafficClass::SequentialPvORestore:
        return "sequential_pv_o_restore";
    case AttentionClusterTrafficClass::SequentialPvOOutput:
        return "sequential_pv_o_output";
    case AttentionClusterTrafficClass::PvOFinalDrain: return "pv_o_final_drain";
    case AttentionClusterTrafficClass::ScoreSfuRead: return "score_sfu_read";
    case AttentionClusterTrafficClass::SfuPWrite: return "sfu_p_write";
    case AttentionClusterTrafficClass::OScale: return "o_scale";
    case AttentionClusterTrafficClass::OAccumulate: return "o_accumulate";
    case AttentionClusterTrafficClass::Count: break;
    }
    return "invalid";
}

class BusyIntervalUnion {
public:
    bool add(uint64_t start, uint64_t end) {
        if (end <= start || (hasInterval_ && start < lastStart_)) return false;
        while (!activeEnds_.empty() && activeEnds_.top() <= start) {
            activeEnds_.pop();
        }
        activeEnds_.push(end);
        maxConcurrency_ = std::max<uint64_t>(maxConcurrency_, activeEnds_.size());

        if (!hasInterval_) {
            firstStart_ = start;
            mergedEnd_ = end;
            unionCycles_ = end - start;
            hasInterval_ = true;
        } else if (start <= mergedEnd_) {
            if (end > mergedEnd_) {
                unionCycles_ += end - mergedEnd_;
                mergedEnd_ = end;
            }
        } else {
            idleCycles_ += start - mergedEnd_;
            unionCycles_ += end - start;
            mergedEnd_ = end;
        }
        lastStart_ = start;
        lastEnd_ = std::max(lastEnd_, end);
        operations_ += 1;
        return true;
    }

    bool empty() const { return !hasInterval_; }
    uint64_t unionCycles() const { return unionCycles_; }
    uint64_t spanCycles() const {
        return hasInterval_ ? lastEnd_ - firstStart_ : 0;
    }
    uint64_t idleCycles() const { return idleCycles_; }
    uint64_t maxConcurrency() const { return maxConcurrency_; }
    uint64_t operations() const { return operations_; }

    void reset() { *this = BusyIntervalUnion(); }

private:
    bool hasInterval_ = false;
    uint64_t firstStart_ = 0;
    uint64_t lastStart_ = 0;
    uint64_t mergedEnd_ = 0;
    uint64_t lastEnd_ = 0;
    uint64_t unionCycles_ = 0;
    uint64_t idleCycles_ = 0;
    uint64_t maxConcurrency_ = 0;
    uint64_t operations_ = 0;
    std::priority_queue<uint64_t, std::vector<uint64_t>, std::greater<uint64_t>>
        activeEnds_;
};

class BusyActivityTracker {
public:
    bool enter(uint64_t cycle) {
        if (hasEvent_ && cycle < lastEventCycle_) return false;
        hasEvent_ = true;
        lastEventCycle_ = cycle;
        if (active_ == 0) {
            openCycle_ = cycle;
            if (!hasSpan_) {
                firstCycle_ = cycle;
                hasSpan_ = true;
            }
        }
        ++active_;
        maxConcurrency_ = std::max(maxConcurrency_, active_);
        return true;
    }

    bool leave(uint64_t cycle) {
        if (active_ == 0 || (hasEvent_ && cycle < lastEventCycle_)) return false;
        lastEventCycle_ = cycle;
        --active_;
        if (active_ == 0) {
            unionCycles_ += cycle - openCycle_;
            lastCloseCycle_ = cycle;
        }
        return true;
    }

    uint64_t unionCycles(uint64_t now = 0) const {
        if (active_ == 0) return unionCycles_;
        const uint64_t end = std::max(now, lastEventCycle_);
        return unionCycles_ + end - openCycle_;
    }
    uint64_t spanCycles(uint64_t now = 0) const {
        if (!hasSpan_) return 0;
        const uint64_t end = active_ == 0
            ? lastCloseCycle_ : std::max(now, lastEventCycle_);
        return end - firstCycle_;
    }
    uint64_t idleCycles(uint64_t now = 0) const {
        return spanCycles(now) - unionCycles(now);
    }
    uint64_t maxConcurrency() const { return maxConcurrency_; }
    uint64_t active() const { return active_; }

private:
    bool hasEvent_ = false;
    bool hasSpan_ = false;
    uint64_t active_ = 0;
    uint64_t maxConcurrency_ = 0;
    uint64_t firstCycle_ = 0;
    uint64_t openCycle_ = 0;
    uint64_t lastCloseCycle_ = 0;
    uint64_t lastEventCycle_ = 0;
    uint64_t unionCycles_ = 0;
};

struct AttentionClusterConfig {
    uint32_t arrays = 64;
    uint32_t qkArrays = 16;
    uint32_t pvArrays = 48;
    uint32_t arrayInputs = 64;
    uint32_t arrayOutputs = 64;
    uint32_t operandBanks = 2;
    uint32_t scoreSlots = 4;
    uint32_t pSlots = 4;
    uint32_t oContexts = 4;
    uint32_t headDim = 128;
    uint32_t queryBlockRows = 16;
    uint32_t keyBlockRows = 64;
    uint32_t groupSize = 4;
    uint32_t elemBytes = 4;
    uint32_t oFmaLanes = 16;
    uint32_t oFmaLatencyCycles = 2;
    bool oFp32Fused = true;
    bool causal = false;

    std::string validate() const {
        if (arrays != 64 || qkArrays + pvArrays != arrays ||
            (qkArrays != 16 && qkArrays != 24 && qkArrays != 32 &&
             qkArrays != 40))
            return "cluster requires 64 arrays with QK ownership 16, 24, 32, or 40";
        if ((qkArrays % 2) != 0 || (pvArrays % 2) != 0)
            return "cluster D128 ownership must contain complete D64 array pairs";
        if (arrayInputs != 64 || arrayOutputs != 64)
            return "cluster requires 64x64 arrays";
        if (operandBanks != 2) return "cluster requires two operand banks";
        if (scoreSlots != 4 || pSlots != 4 || oContexts != 4)
            return "cluster requires four score/P/O contexts";
        if (headDim != 128 || queryBlockRows != 16 ||
            (keyBlockRows != 32 && keyBlockRows != 64))
            return "cluster requires D128 Br16 and Bc32 or Bc64";
        if (groupSize != 4) return "cluster requires query group size four";
        if (elemBytes != 4) return "cluster requires FP32 operands";
        if (oFmaLanes != 16 || oFmaLatencyCycles != 2 || !oFp32Fused)
            return "cluster requires a 16-lane two-cycle fused FP32 O engine";
        if (causal) return "cluster phase L2-L4 supports non-causal attention only";
        return {};
    }

    uint64_t fingerprint() const {
        uint64_t hash = 1469598103934665603ULL;
        const std::array<uint64_t, 18> values = {{
            arrays, qkArrays, pvArrays, arrayInputs, arrayOutputs, operandBanks,
            scoreSlots, pSlots, oContexts, headDim, queryBlockRows, keyBlockRows,
            groupSize, elemBytes, oFmaLanes, oFmaLatencyCycles,
            oFp32Fused ? 1ULL : 0ULL, causal ? 1ULL : 0ULL,
        }};
        for (uint64_t value : values) {
            for (unsigned byte = 0; byte < sizeof(value); ++byte) {
                hash ^= (value >> (byte * 8)) & 0xffULL;
                hash *= 1099511628211ULL;
            }
        }
        return hash;
    }
};

struct AttentionClusterTag {
    uint64_t generation = 0;
    uint64_t jobId = 0;
    uint32_t group = 0;
    uint32_t queryContext = 0;
    uint32_t queryBlock = 0;
    uint32_t keyTile = 0;
    uint32_t sequence = 0;

    bool operator==(const AttentionClusterTag& other) const {
        return generation == other.generation && jobId == other.jobId &&
            group == other.group && queryContext == other.queryContext &&
            queryBlock == other.queryBlock && keyTile == other.keyTile &&
            sequence == other.sequence;
    }
};

class AttentionScoreFifo {
public:
    static constexpr uint32_t kSlots = 4;
    static constexpr size_t kElementsPerSlot = 16 * 64;

    bool reserve(uint32_t slot, const AttentionClusterTag& tag, size_t elements) {
        if (slot >= kSlots || elements == 0 || elements > kElementsPerSlot ||
            slots_[slot].valid) return false;
        Slot& target = slots_[slot];
        target.valid = true;
        target.tag = tag;
        target.elements.assign(elements, 0.0f);
        target.elementValid.assign(elements, 0);
        target.elementReserved.assign(elements, 0);
        target.validElements = 0;
        return true;
    }

    bool write(uint32_t slot, const AttentionClusterTag& tag, size_t offset,
               const std::vector<float>& values) {
        if (!rangeAvailable(slot, tag, offset, values.size())) return false;
        Slot& target = slots_[slot];
        for (size_t index = 0; index < values.size(); ++index) {
            target.elementValid[offset + index] = 1;
            ++target.validElements;
            target.elements[offset + index] = values[index];
        }
        return true;
    }

    bool reserveWrite(uint32_t slot, const AttentionClusterTag& tag,
                      size_t offset, size_t count) {
        if (!rangeAvailable(slot, tag, offset, count)) return false;
        Slot& target = slots_[slot];
        std::fill(target.elementReserved.begin() + offset,
                  target.elementReserved.begin() + offset + count, 1);
        return true;
    }

    bool commitWrite(uint32_t slot, const AttentionClusterTag& tag,
                     size_t offset, const std::vector<float>& values) {
        if (!matches(slot, tag) || values.empty()) return false;
        Slot& target = slots_[slot];
        if (!rangeInBounds(target, offset, values.size()) ||
            std::find(target.elementReserved.begin() + offset,
                      target.elementReserved.begin() + offset + values.size(), 0) !=
                target.elementReserved.begin() + offset + values.size() ||
            std::find(target.elementValid.begin() + offset,
                      target.elementValid.begin() + offset + values.size(), 1) !=
                target.elementValid.begin() + offset + values.size()) return false;
        for (size_t index = 0; index < values.size(); ++index) {
            target.elementReserved[offset + index] = 0;
            target.elementValid[offset + index] = 1;
            target.elements[offset + index] = values[index];
            ++target.validElements;
        }
        return true;
    }

    bool overwrite(uint32_t slot, const AttentionClusterTag& tag, size_t offset,
                   const std::vector<float>& values) {
        if (!canRead(slot, tag, offset, values.size())) return false;
        Slot& target = slots_[slot];
        std::copy(values.begin(), values.end(), target.elements.begin() + offset);
        return true;
    }

    bool canRead(uint32_t slot, const AttentionClusterTag& tag, size_t offset,
                 size_t count) const {
        if (!matches(slot, tag) || count == 0) return false;
        const Slot& source = slots_[slot];
        if (offset > source.elements.size() ||
            count > source.elements.size() - offset) return false;
        return std::find(source.elementValid.begin() + offset,
                         source.elementValid.begin() + offset + count, 0) ==
            source.elementValid.begin() + offset + count;
    }

    bool read(uint32_t slot, const AttentionClusterTag& tag, size_t offset,
              size_t count, std::vector<float>& values) const {
        if (!canRead(slot, tag, offset, count)) return false;
        const Slot& source = slots_[slot];
        values.assign(source.elements.begin() + offset,
                      source.elements.begin() + offset + count);
        return true;
    }

    bool ready(uint32_t slot, const AttentionClusterTag& tag) const {
        return matches(slot, tag) &&
            slots_[slot].validElements == slots_[slot].elements.size();
    }

    bool reserved(uint32_t slot, const AttentionClusterTag& tag) const {
        return matches(slot, tag);
    }

    bool occupied(uint32_t slot) const {
        return slot < kSlots && slots_[slot].valid;
    }

    bool release(uint32_t slot, const AttentionClusterTag& tag) {
        if (!matches(slot, tag)) return false;
        slots_[slot] = {};
        return true;
    }

    uint32_t cancelGeneration(uint64_t generation) {
        uint32_t cancelled = 0;
        for (Slot& slot : slots_) {
            if (slot.valid && slot.tag.generation == generation) {
                slot = {};
                ++cancelled;
            }
        }
        return cancelled;
    }

    uint32_t occupied() const {
        return static_cast<uint32_t>(std::count_if(
            slots_.begin(), slots_.end(),
            [](const Slot& slot) { return slot.valid; }));
    }

private:
    struct Slot {
        bool valid = false;
        AttentionClusterTag tag = {};
        std::vector<float> elements;
        std::vector<uint8_t> elementValid;
        std::vector<uint8_t> elementReserved;
        size_t validElements = 0;
    };

    static bool rangeInBounds(const Slot& slot, size_t offset, size_t count) {
        return count != 0 && offset <= slot.elements.size() &&
            count <= slot.elements.size() - offset;
    }

    bool rangeAvailable(uint32_t slot, const AttentionClusterTag& tag,
                        size_t offset, size_t count) const {
        if (!matches(slot, tag)) return false;
        const Slot& target = slots_[slot];
        if (!rangeInBounds(target, offset, count)) return false;
        return std::find(target.elementValid.begin() + offset,
                         target.elementValid.begin() + offset + count, 1) ==
                   target.elementValid.begin() + offset + count &&
            std::find(target.elementReserved.begin() + offset,
                      target.elementReserved.begin() + offset + count, 1) ==
                   target.elementReserved.begin() + offset + count;
    }

    bool matches(uint32_t slot, const AttentionClusterTag& tag) const {
        return slot < kSlots && slots_[slot].valid && slots_[slot].tag == tag;
    }

    std::array<Slot, kSlots> slots_ = {};
};

class AttentionOAccumulator {
public:
    static constexpr uint32_t kContexts = 4;
    static constexpr uint32_t kRows = 16;
    static constexpr uint32_t kPanels = 8;
    static constexpr uint32_t kValuesPerSegment = 16;
    static constexpr size_t kValuesPerContext =
        kRows * kPanels * kValuesPerSegment;
    static constexpr uint32_t kFmaLanes = 16;
    static constexpr uint32_t kRowFmaBanks = kPanels;
    static constexpr uint32_t kRowFmaLanes =
        kRowFmaBanks * kFmaLanes;
    static constexpr uint64_t kFmaLatencyCycles = 2;
    static constexpr bool kFp32Fused = true;

    struct Completion {
        uint64_t id = 0;
        bool ok = false;
        bool drain = false;
        uint32_t slot = 0;
        AttentionClusterTag tag = {};
        std::vector<float> values;
    };

    bool reserve(uint32_t slot, const AttentionClusterTag& tag,
                 uint32_t keyTiles) {
        if (slot >= kContexts || tag.generation == 0 || keyTiles == 0 ||
            contexts_[slot].valid) return false;
        Context& context = contexts_[slot];
        context.valid = true;
        context.tag = tag;
        context.keyTiles = keyTiles;
        context.values.fill(0.0f);
        context.segmentState.fill(0);
        ++occupied_;
        highWater_ = std::max(highWater_, occupied_);
        return true;
    }

    bool submitSegment(uint32_t slot, const AttentionClusterTag& tag,
                       uint32_t keyTile, uint32_t row, uint32_t panel,
                       float alpha, const std::vector<float>& values,
                       uint64_t now, uint64_t* id, uint64_t* readyCycle) {
        if (!matches(slot, tag) || row >= kRows || panel >= kPanels ||
            values.size() != kValuesPerSegment || !std::isfinite(alpha) ||
            id == nullptr || readyCycle == nullptr) return false;
        Context& context = contexts_[slot];
        const size_t segment = static_cast<size_t>(row) * kPanels + panel;
        if (context.draining || keyTile != context.expectedKeyTile ||
            context.segmentState[segment] != 0) return false;

        uint64_t cursor = std::max(now, bankReadyCycle_[row]);
        bankConflictCycles_ += cursor - now;
        if (keyTile != 0) {
            const uint64_t readStart = std::max(cursor, nextReadCycle_);
            readWaitCycles_ += readStart - cursor;
            nextReadCycle_ = readStart + 1;
            intervalOrderValid_ = readIntervals_.add(readStart, readStart + 1) &&
                intervalOrderValid_;
            const uint64_t fmaIssue = std::max(readStart + 1, nextAluCycle_);
            aluWaitCycles_ += fmaIssue - (readStart + 1);
            nextAluCycle_ = fmaIssue + 1;
            intervalOrderValid_ = aluIntervals_.add(fmaIssue, fmaIssue + 1) &&
                intervalOrderValid_;
            cursor = fmaIssue + fmaLatency_;
            ++scaleOperations_;
        } else {
            const uint64_t accumulateIssue = std::max(cursor, nextAluCycle_);
            aluWaitCycles_ += accumulateIssue - cursor;
            nextAluCycle_ = accumulateIssue + 1;
            intervalOrderValid_ =
                aluIntervals_.add(accumulateIssue, accumulateIssue + 1) &&
                intervalOrderValid_;
            cursor = accumulateIssue + addLatency_;
        }
        const uint64_t writeStart = std::max(
            cursor, nextWriteCycle_);
        writeWaitCycles_ += writeStart - cursor;
        nextWriteCycle_ = writeStart + 1;
        intervalOrderValid_ = writeIntervals_.add(writeStart, writeStart + 1) &&
            intervalOrderValid_;
        bankReadyCycle_[row] = writeStart + 1;

        Pending pending;
        pending.id = nextId_++;
        pending.readyCycle = writeStart + 1;
        pending.slot = slot;
        pending.tag = tag;
        pending.keyTile = keyTile;
        pending.row = row;
        pending.panel = panel;
        pending.alpha = alpha;
        std::copy(values.begin(), values.end(), pending.values.begin());
        context.segmentState[segment] = 1;
        pending_.emplace(pending.id, pending);
        *id = pending.id;
        *readyCycle = pending.readyCycle;
        ++accumulateOperations_;
        return true;
    }

    bool submitRow(uint32_t slot, const AttentionClusterTag& tag,
                   uint32_t keyTile, uint32_t row, float alpha,
                   const std::vector<float>& values, uint64_t now,
                   uint64_t* id, uint64_t* readyCycle) {
        if (!matches(slot, tag) || row >= kRows ||
            values.size() != kRowFmaLanes || !std::isfinite(alpha) ||
            id == nullptr || readyCycle == nullptr) return false;
        Context& context = contexts_[slot];
        const size_t firstSegment = static_cast<size_t>(row) * kPanels;
        if (context.draining || keyTile != context.expectedKeyTile ||
            std::any_of(context.segmentState.begin() + firstSegment,
                        context.segmentState.begin() + firstSegment + kPanels,
                        [](uint8_t state) { return state != 0; })) return false;

        uint64_t cursor = std::max(now, bankReadyCycle_[row]);
        if (keyTile != 0) {
            const uint64_t readStart = std::max(cursor, nextReadCycle_);
            readWaitCycles_ += readStart - cursor;
            nextReadCycle_ = readStart + 1;
            intervalOrderValid_ = readIntervals_.add(readStart, readStart + 1) &&
                intervalOrderValid_;
            const uint64_t fmaIssue = std::max(readStart + 1, nextAluCycle_);
            aluWaitCycles_ += fmaIssue - (readStart + 1);
            nextAluCycle_ = fmaIssue + 1;
            intervalOrderValid_ = aluIntervals_.add(fmaIssue, fmaIssue + 1) &&
                intervalOrderValid_;
            cursor = fmaIssue + fmaLatency_;
            scaleOperations_ += kPanels;
        } else {
            const uint64_t accumulateIssue = std::max(cursor, nextAluCycle_);
            aluWaitCycles_ += accumulateIssue - cursor;
            nextAluCycle_ = accumulateIssue + 1;
            intervalOrderValid_ =
                aluIntervals_.add(accumulateIssue, accumulateIssue + 1) &&
                intervalOrderValid_;
            cursor = accumulateIssue + addLatency_;
        }
        const uint64_t writeStart = std::max(cursor, nextWriteCycle_);
        writeWaitCycles_ += writeStart - cursor;
        nextWriteCycle_ = writeStart + 1;
        intervalOrderValid_ = writeIntervals_.add(writeStart, writeStart + 1) &&
            intervalOrderValid_;
        bankReadyCycle_[row] = writeStart + 1;

        Pending pending;
        pending.id = nextId_++;
        pending.readyCycle = writeStart + 1;
        pending.slot = slot;
        pending.tag = tag;
        pending.keyTile = keyTile;
        pending.row = row;
        pending.rowFused = true;
        pending.alpha = alpha;
        std::copy(values.begin(), values.end(), pending.rowValues.begin());
        std::fill(context.segmentState.begin() + firstSegment,
                  context.segmentState.begin() + firstSegment + kPanels, 1);
        pending_.emplace(pending.id, pending);
        *id = pending.id;
        *readyCycle = pending.readyCycle;
        accumulateOperations_ += kPanels;
        ++rowOperations_;
        return true;
    }

    bool requestDrain(uint32_t slot, const AttentionClusterTag& tag,
                      uint64_t now, uint64_t* id, uint64_t* readyCycle) {
        if (!matches(slot, tag) || id == nullptr || readyCycle == nullptr)
            return false;
        Context& context = contexts_[slot];
        if (context.draining || context.expectedKeyTile != context.keyTiles ||
            context.committedSegments != 0) return false;
        uint64_t cursor = now;
        for (uint32_t row = 0; row < kRows; ++row) {
            for (uint32_t panel = 0; panel < kPanels; ++panel) {
                const uint64_t start = std::max(
                    std::max(cursor, nextReadCycle_), bankReadyCycle_[row]);
                drainWaitCycles_ += start - cursor;
                nextReadCycle_ = start + 1;
                bankReadyCycle_[row] = start + 1;
                intervalOrderValid_ = readIntervals_.add(start, start + 1) &&
                    intervalOrderValid_;
                cursor = start + 1;
            }
        }
        Pending pending;
        pending.id = nextId_++;
        pending.readyCycle = cursor;
        pending.slot = slot;
        pending.tag = tag;
        pending.drain = true;
        pending_.emplace(pending.id, pending);
        context.draining = true;
        *id = pending.id;
        *readyCycle = pending.readyCycle;
        ++drainOperations_;
        return true;
    }

    std::vector<Completion> progress(uint64_t now) {
        std::vector<Completion> completions;
        for (auto it = pending_.begin(); it != pending_.end();) {
            if (it->second.readyCycle > now) {
                ++it;
                continue;
            }
            Pending pending = it->second;
            it = pending_.erase(it);
            Completion completion;
            completion.id = pending.id;
            completion.slot = pending.slot;
            completion.tag = pending.tag;
            completion.drain = pending.drain;
            completion.ok = matches(pending.slot, pending.tag);
            if (!completion.ok) {
                completions.push_back(std::move(completion));
                continue;
            }
            Context& context = contexts_[pending.slot];
            if (pending.drain) {
                completion.values.assign(
                    context.values.begin(), context.values.end());
                completions.push_back(std::move(completion));
                continue;
            }
            const size_t segment =
                static_cast<size_t>(pending.row) * kPanels + pending.panel;
            completion.ok = pending.keyTile == context.expectedKeyTile;
            const uint32_t committed = pending.rowFused ? kPanels : 1;
            for (uint32_t panel = 0; completion.ok && panel < committed; ++panel) {
                completion.ok = context.segmentState[segment + panel] == 1;
            }
            if (completion.ok) {
                for (uint32_t panel = 0; panel < committed; ++panel) {
                    const size_t targetSegment = segment + panel;
                    const size_t offset = targetSegment * kValuesPerSegment;
                    for (size_t lane = 0; lane < kValuesPerSegment; ++lane) {
                        const float value = pending.rowFused
                            ? pending.rowValues[panel * kValuesPerSegment + lane]
                            : pending.values[lane];
                        context.values[offset + lane] = pending.keyTile == 0
                            ? value
                            : std::fma(pending.alpha,
                                       context.values[offset + lane], value);
                    }
                    context.segmentState[targetSegment] = 2;
                }
                context.committedSegments += committed;
                if (context.committedSegments == kRows * kPanels) {
                    ++context.expectedKeyTile;
                    context.committedSegments = 0;
                    context.segmentState.fill(0);
                }
            }
            completions.push_back(std::move(completion));
        }
        return completions;
    }

    bool release(uint32_t slot, const AttentionClusterTag& tag) {
        if (!matches(slot, tag) || !contexts_[slot].draining ||
            hasPending(slot, tag)) return false;
        contexts_[slot] = {};
        --occupied_;
        return true;
    }

    uint32_t cancelGeneration(uint64_t generation) {
        uint32_t cancelled = 0;
        for (auto it = pending_.begin(); it != pending_.end();) {
            if (it->second.tag.generation == generation) it = pending_.erase(it);
            else ++it;
        }
        for (Context& context : contexts_) {
            if (context.valid && context.tag.generation == generation) {
                context = {};
                --occupied_;
                ++cancelled;
            }
        }
        return cancelled;
    }

    bool drained() const { return occupied_ == 0 && pending_.empty(); }
    uint32_t occupied() const { return occupied_; }
    uint32_t highWater() const { return highWater_; }
    uint64_t scaleOperations() const { return scaleOperations_; }
    uint64_t accumulateOperations() const { return accumulateOperations_; }
    uint64_t rowOperations() const { return rowOperations_; }
    uint64_t drainOperations() const { return drainOperations_; }
    uint64_t readWaitCycles() const { return readWaitCycles_; }
    uint64_t writeWaitCycles() const { return writeWaitCycles_; }
    uint64_t aluWaitCycles() const { return aluWaitCycles_; }
    uint64_t bankConflictCycles() const { return bankConflictCycles_; }
    uint64_t drainWaitCycles() const { return drainWaitCycles_; }
    uint64_t readBusyCycles() const { return readIntervals_.unionCycles(); }
    uint64_t readBusySpanCycles() const { return readIntervals_.spanCycles(); }
    uint64_t readIdleCycles() const { return readIntervals_.idleCycles(); }
    uint64_t readMaxConcurrency() const {
        return readIntervals_.maxConcurrency();
    }
    uint64_t writeBusyCycles() const { return writeIntervals_.unionCycles(); }
    uint64_t writeBusySpanCycles() const { return writeIntervals_.spanCycles(); }
    uint64_t writeIdleCycles() const { return writeIntervals_.idleCycles(); }
    uint64_t writeMaxConcurrency() const {
        return writeIntervals_.maxConcurrency();
    }
    uint64_t aluBusyCycles() const { return aluIntervals_.unionCycles(); }
    uint64_t aluBusySpanCycles() const { return aluIntervals_.spanCycles(); }
    uint64_t aluIdleCycles() const { return aluIntervals_.idleCycles(); }
    uint64_t aluMaxConcurrency() const { return aluIntervals_.maxConcurrency(); }
    bool intervalOrderValid() const { return intervalOrderValid_; }

private:
    struct Context {
        bool valid = false;
        bool draining = false;
        AttentionClusterTag tag = {};
        uint32_t keyTiles = 0;
        uint32_t expectedKeyTile = 0;
        uint32_t committedSegments = 0;
        std::array<float, kValuesPerContext> values = {};
        std::array<uint8_t, kRows * kPanels> segmentState = {};
    };

    struct Pending {
        uint64_t id = 0;
        uint64_t readyCycle = 0;
        uint32_t slot = 0;
        AttentionClusterTag tag = {};
        uint32_t keyTile = 0;
        uint32_t row = 0;
        uint32_t panel = 0;
        float alpha = 1.0f;
        bool drain = false;
        bool rowFused = false;
        std::array<float, kValuesPerSegment> values = {};
        std::array<float, kRowFmaLanes> rowValues = {};
    };

    bool matches(uint32_t slot, const AttentionClusterTag& tag) const {
        return slot < kContexts && contexts_[slot].valid &&
            contexts_[slot].tag == tag;
    }

    bool hasPending(uint32_t slot, const AttentionClusterTag& tag) const {
        return std::any_of(pending_.begin(), pending_.end(),
            [slot, &tag](const auto& item) {
                return item.second.slot == slot && item.second.tag == tag;
            });
    }

    static constexpr uint64_t fmaLatency_ = kFmaLatencyCycles;
    static constexpr uint64_t addLatency_ = 2;
    uint64_t nextId_ = 1;
    uint64_t nextReadCycle_ = 0;
    uint64_t nextWriteCycle_ = 0;
    uint64_t nextAluCycle_ = 0;
    std::array<uint64_t, kRows> bankReadyCycle_ = {};
    std::array<Context, kContexts> contexts_ = {};
    std::map<uint64_t, Pending> pending_;
    uint32_t occupied_ = 0;
    uint32_t highWater_ = 0;
    uint64_t scaleOperations_ = 0;
    uint64_t accumulateOperations_ = 0;
    uint64_t rowOperations_ = 0;
    uint64_t drainOperations_ = 0;
    uint64_t readWaitCycles_ = 0;
    uint64_t writeWaitCycles_ = 0;
    uint64_t aluWaitCycles_ = 0;
    uint64_t bankConflictCycles_ = 0;
    uint64_t drainWaitCycles_ = 0;
    BusyIntervalUnion readIntervals_;
    BusyIntervalUnion writeIntervals_;
    BusyIntervalUnion aluIntervals_;
    bool intervalOrderValid_ = true;
};

class AttentionGenerationFence {
public:
    bool begin(uint64_t generation) {
        if (generation == 0 || activeGeneration_ != 0) return false;
        activeGeneration_ = generation;
        return true;
    }

    bool matches(uint64_t generation) const {
        return generation != 0 && generation == activeGeneration_;
    }

    bool retire(uint64_t generation) {
        if (!matches(generation)) return false;
        activeGeneration_ = 0;
        return true;
    }

    bool cancel(uint64_t generation) {
        return retire(generation);
    }

    uint64_t active() const { return activeGeneration_; }

private:
    uint64_t activeGeneration_ = 0;
};

enum class AttentionArrayOwner : uint8_t { None, Qk, Pv };
enum class AttentionClusterContextKind : uint8_t { Qk, Score, Pv, O };

struct AttentionClusterContext {
    bool valid = false;
    AttentionClusterTag tag = {};
    uint32_t operandBank = 0;
    uint32_t creditsHeld = 0;
};

class AttentionClusterState {
public:
    static constexpr uint32_t kArrayCount = 64;
    static constexpr uint32_t kQkArrayCount = 16;
    static constexpr uint32_t kPvArrayCount = 48;
    static constexpr uint32_t kOperandBanks = 2;
    static constexpr uint32_t kScoreSlots = 4;
    static constexpr uint32_t kPSlots = 4;
    static constexpr uint32_t kOContexts = 4;

    bool configure(uint64_t generation, uint32_t qkArrays = kQkArrayCount) {
        if (generation == 0 ||
            (qkArrays != 16 && qkArrays != 24 && qkArrays != 32 &&
             qkArrays != 40)) return false;
        cancel();
        enabled_ = true;
        generation_ = generation;
        for (uint32_t id = 0; id < kArrayCount; ++id) {
            owners_[id] = id < qkArrays ? AttentionArrayOwner::Qk
                                        : AttentionArrayOwner::Pv;
        }
        return true;
    }

    bool enabled() const { return enabled_; }
    uint64_t generation() const { return generation_; }
    bool owns(uint32_t arrayId, AttentionArrayOwner owner) const {
        return enabled_ && arrayId < owners_.size() && owners_[arrayId] == owner;
    }

    int reserve(AttentionClusterContextKind kind, const AttentionClusterTag& tag,
                uint32_t operandBank) {
        if (!enabled_ || tag.generation != generation_ ||
            operandBank >= kOperandBanks) return -1;
        auto& contexts = contextArray(kind);
        for (size_t i = 0; i < capacity(kind); ++i) {
            if (!contexts[i].valid) {
                contexts[i].valid = true;
                contexts[i].tag = tag;
                contexts[i].operandBank = operandBank;
                contexts[i].creditsHeld = 0;
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    bool callbackMatches(AttentionClusterContextKind kind, uint32_t slot,
                         const AttentionClusterTag& tag) const {
        const auto& contexts = contextArray(kind);
        return slot < contexts.size() && contexts[slot].valid &&
            contexts[slot].tag == tag && tag.generation == generation_;
    }

    bool release(AttentionClusterContextKind kind, uint32_t slot,
                 const AttentionClusterTag& tag) {
        if (!callbackMatches(kind, slot, tag)) return false;
        contextArray(kind)[slot] = {};
        return true;
    }

    bool acquireBank(uint32_t arrayId, uint32_t bank,
                     AttentionArrayOwner owner, const AttentionClusterTag& tag) {
        if (!owns(arrayId, owner) || bank >= kOperandBanks) return false;
        BankLease& lease = bankLeases_[arrayId][bank];
        if (lease.refs != 0 || tag.generation != generation_) return false;
        lease.tag = tag;
        lease.refs = 1;
        return true;
    }

    bool retainBank(uint32_t arrayId, uint32_t bank,
                    AttentionArrayOwner owner, const AttentionClusterTag& tag) {
        if (!owns(arrayId, owner) || bank >= kOperandBanks ||
            bankLeases_[arrayId][bank].refs == 0 ||
            !(bankLeases_[arrayId][bank].tag == tag)) return false;
        bankLeases_[arrayId][bank].refs += 1;
        return true;
    }

    bool releaseBank(uint32_t arrayId, uint32_t bank,
                     AttentionArrayOwner owner, const AttentionClusterTag& tag) {
        if (!owns(arrayId, owner) || bank >= kOperandBanks ||
            bankLeases_[arrayId][bank].refs == 0 ||
            !(bankLeases_[arrayId][bank].tag == tag)) return false;
        BankLease& lease = bankLeases_[arrayId][bank];
        lease.refs -= 1;
        if (lease.refs == 0) lease = {};
        return true;
    }

    uint32_t bankRefcount(uint32_t arrayId, uint32_t bank) const {
        return arrayId < kArrayCount && bank < kOperandBanks
            ? bankLeases_[arrayId][bank].refs : 0;
    }

    bool bankMatches(uint32_t arrayId, uint32_t bank,
                     AttentionArrayOwner owner,
                     const AttentionClusterTag& tag) const {
        return owns(arrayId, owner) && bank < kOperandBanks &&
            bankLeases_[arrayId][bank].refs != 0 &&
            bankLeases_[arrayId][bank].tag == tag;
    }

    uint32_t liveBankRefs() const {
        uint32_t refs = 0;
        for (const auto& leases : bankLeases_) {
            for (const BankLease& lease : leases) refs += lease.refs;
        }
        return refs;
    }

    uint32_t liveContexts(AttentionClusterContextKind kind) const {
        const auto& contexts = contextArray(kind);
        return static_cast<uint32_t>(std::count_if(
            contexts.begin(), contexts.begin() + capacity(kind),
            [](const AttentionClusterContext& context) { return context.valid; }));
    }

    uint32_t liveContexts() const {
        return liveContexts(AttentionClusterContextKind::Qk) +
            liveContexts(AttentionClusterContextKind::Score) +
            liveContexts(AttentionClusterContextKind::Pv) +
            liveContexts(AttentionClusterContextKind::O);
    }

    uint32_t cancel() {
        uint32_t cancelled = 0;
        for (auto* contexts : {&qkContexts_, &scoreContexts_, &pvContexts_, &oContexts_}) {
            for (auto& context : *contexts) {
                if (context.valid) cancelled += 1;
                context = {};
            }
        }
        for (auto& leases : bankLeases_) leases.fill({});
        owners_.fill(AttentionArrayOwner::None);
        enabled_ = false;
        generation_ = 0;
        return cancelled;
    }

    bool drained() const {
        const auto noLive = [](const auto& contexts) {
            return std::none_of(contexts.begin(), contexts.end(),
                                [](const AttentionClusterContext& context) {
                                    return context.valid;
                                });
        };
        const bool noBanks = std::all_of(
            bankLeases_.begin(), bankLeases_.end(), [](const auto& leases) {
                return leases[0].refs == 0 && leases[1].refs == 0;
            });
        return noLive(qkContexts_) && noLive(scoreContexts_) &&
            noLive(pvContexts_) && noLive(oContexts_) && noBanks;
    }

private:
    using ContextArray = std::array<AttentionClusterContext, 4>;
    struct BankLease {
        AttentionClusterTag tag = {};
        uint32_t refs = 0;
    };

    static constexpr size_t capacity(AttentionClusterContextKind kind) {
        return kind == AttentionClusterContextKind::Qk ||
               kind == AttentionClusterContextKind::Pv ? 2 : 4;
    }

    ContextArray& contextArray(AttentionClusterContextKind kind) {
        switch (kind) {
        case AttentionClusterContextKind::Qk: return qkContexts_;
        case AttentionClusterContextKind::Score: return scoreContexts_;
        case AttentionClusterContextKind::Pv: return pvContexts_;
        case AttentionClusterContextKind::O: return oContexts_;
        }
        return qkContexts_;
    }

    const ContextArray& contextArray(AttentionClusterContextKind kind) const {
        return const_cast<AttentionClusterState*>(this)->contextArray(kind);
    }

    bool enabled_ = false;
    uint64_t generation_ = 0;
    std::array<AttentionArrayOwner, kArrayCount> owners_ = {};
    std::array<std::array<BankLease, kOperandBanks>, kArrayCount> bankLeases_ = {};
    ContextArray qkContexts_ = {};
    ContextArray scoreContexts_ = {};
    ContextArray pvContexts_ = {};
    ContextArray oContexts_ = {};
};

} // namespace Golem
} // namespace SST

#endif
