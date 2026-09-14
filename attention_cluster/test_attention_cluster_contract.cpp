#include "sst/elements/golem/attention/attentionCluster.h"

#include <cassert>
#include <cstdint>

using namespace SST::Golem;

int main() {
    BusyIntervalUnion empty;
    assert(empty.empty());
    assert(empty.unionCycles() == 0);
    assert(empty.spanCycles() == 0);

    BusyIntervalUnion intervals;
    assert(intervals.add(10, 20));
    assert(intervals.add(12, 18));
    assert(intervals.add(20, 25));
    assert(intervals.add(30, 35));
    assert(intervals.unionCycles() == 20);
    assert(intervals.spanCycles() == 25);
    assert(intervals.idleCycles() == 5);
    assert(intervals.maxConcurrency() == 2);
    assert(intervals.operations() == 4);
    assert(!intervals.add(29, 31));

    BusyActivityTracker activity;
    assert(activity.enter(10));
    assert(activity.enter(12));
    assert(activity.leave(15));
    assert(activity.leave(20));
    assert(activity.enter(25));
    assert(activity.leave(30));
    assert(activity.unionCycles() == 15);
    assert(activity.spanCycles() == 20);
    assert(activity.idleCycles() == 5);
    assert(activity.maxConcurrency() == 2);
    assert(activity.active() == 0);
    assert(!activity.leave(31));
    assert(!activity.enter(29));

    BusyActivityTracker liveActivity;
    assert(liveActivity.enter(40));
    assert(liveActivity.enter(45));
    assert(liveActivity.unionCycles() == 5);
    assert(liveActivity.spanCycles() == 5);
    assert(liveActivity.idleCycles() == 0);

    BusyIntervalUnion qk;
    BusyIntervalUnion pv;
    BusyIntervalUnion combined;
    assert(qk.add(0, 520));
    assert(pv.add(100, 644));
    assert(combined.add(0, 520));
    assert(combined.add(100, 644));
    assert(qk.unionCycles() == 520);
    assert(pv.unionCycles() == 544);
    assert(combined.unionCycles() == 644);
    assert(combined.maxConcurrency() == 2);

    AttentionClusterConfig config;
    assert(config.validate().empty());
    assert(config.fingerprint() == AttentionClusterConfig().fingerprint());
    config.operandBanks = 1;
    assert(!config.validate().empty());
    config = {};
    for (uint32_t qkArrays : {16u, 24u, 32u, 40u}) {
        config.qkArrays = qkArrays;
        config.pvArrays = config.arrays - qkArrays;
        assert(config.validate().empty());
    }
    config.qkArrays = 20;
    config.pvArrays = 44;
    assert(!config.validate().empty());
    config = {};
    config.oFmaLanes = 8;
    assert(!config.validate().empty());
    assert(AttentionOAccumulator::kFmaLanes == 16);
    assert(AttentionOAccumulator::kFmaLatencyCycles == 2);
    assert(AttentionOAccumulator::kFp32Fused);

    AttentionClusterState state;
    assert(state.configure(7));
    for (uint32_t id = 0; id < 16; ++id)
        assert(state.owns(id, AttentionArrayOwner::Qk));
    for (uint32_t id = 16; id < 64; ++id)
        assert(state.owns(id, AttentionArrayOwner::Pv));

    AttentionClusterTag tag;
    tag.generation = 7;
    tag.jobId = 11;
    tag.queryBlock = 2;
    tag.keyTile = 3;
    const int slot = state.reserve(AttentionClusterContextKind::Qk, tag, 0);
    assert(slot >= 0);
    assert(state.acquireBank(0, 0, AttentionArrayOwner::Qk, tag));
    assert(!state.acquireBank(0, 0, AttentionArrayOwner::Qk, tag));
    assert(!state.acquireBank(32, 0, AttentionArrayOwner::Qk, tag));

    AttentionClusterTag stale = tag;
    stale.generation = 6;
    assert(!state.callbackMatches(
        AttentionClusterContextKind::Qk, static_cast<uint32_t>(slot), stale));
    assert(!state.release(
        AttentionClusterContextKind::Qk, static_cast<uint32_t>(slot), stale));
    assert(!state.releaseBank(0, 0, AttentionArrayOwner::Qk, stale));
    assert(state.bankRefcount(0, 0) == 1);
    assert(state.liveBankRefs() == 1);
    assert(state.releaseBank(0, 0, AttentionArrayOwner::Qk, tag));
    assert(state.release(
        AttentionClusterContextKind::Qk, static_cast<uint32_t>(slot), tag));
    assert(state.drained());

    assert(state.reserve(AttentionClusterContextKind::Qk, tag, 0) == 0);
    tag.sequence += 1;
    assert(state.reserve(AttentionClusterContextKind::Qk, tag, 1) == 1);
    tag.sequence += 1;
    assert(state.reserve(AttentionClusterContextKind::Qk, tag, 0) == -1);
    assert(state.liveContexts(AttentionClusterContextKind::Qk) == 2);
    assert(state.cancel() == 2);
    assert(state.configure(7));

    assert(state.reserve(AttentionClusterContextKind::Score, tag, 1) >= 0);
    assert(state.acquireBank(33, 1, AttentionArrayOwner::Pv, tag));
    assert(state.liveBankRefs() == 1);
    assert(state.cancel() == 1);
    assert(state.liveBankRefs() == 0);
    assert(state.drained());
    assert(!state.enabled());

    // Array operands may be handed to the next tagged QK operation once the
    // prior output is committed, while the prior score context remains live.
    assert(state.configure(9));
    AttentionClusterTag producerTag = tag;
    producerTag.generation = 9;
    producerTag.queryBlock = 0;
    producerTag.sequence = 1;
    const int producerQk = state.reserve(
        AttentionClusterContextKind::Qk, producerTag, 0);
    const int producerScore = state.reserve(
        AttentionClusterContextKind::Score, producerTag, 0);
    assert(producerQk >= 0 && producerScore >= 0);
    assert(state.acquireBank(
        0, 0, AttentionArrayOwner::Qk, producerTag));
    assert(state.releaseBank(
        0, 0, AttentionArrayOwner::Qk, producerTag));
    assert(state.release(
        AttentionClusterContextKind::Qk,
        static_cast<uint32_t>(producerQk), producerTag));
    assert(state.callbackMatches(
        AttentionClusterContextKind::Score,
        static_cast<uint32_t>(producerScore), producerTag));

    AttentionClusterTag nextProducerTag = producerTag;
    nextProducerTag.queryBlock = 1;
    nextProducerTag.queryContext = 1;
    nextProducerTag.sequence = 2;
    const int nextQk = state.reserve(
        AttentionClusterContextKind::Qk, nextProducerTag, 0);
    assert(nextQk >= 0);
    assert(state.acquireBank(
        0, 0, AttentionArrayOwner::Qk, nextProducerTag));
    assert(!state.releaseBank(
        0, 0, AttentionArrayOwner::Qk, producerTag));
    assert(state.callbackMatches(
        AttentionClusterContextKind::Score,
        static_cast<uint32_t>(producerScore), producerTag));
    assert(state.release(
        AttentionClusterContextKind::Score,
        static_cast<uint32_t>(producerScore), producerTag));
    assert(state.releaseBank(
        0, 0, AttentionArrayOwner::Qk, nextProducerTag));
    assert(state.release(
        AttentionClusterContextKind::Qk,
        static_cast<uint32_t>(nextQk), nextProducerTag));
    assert(state.drained());
    assert(state.cancel() == 0);

    for (uint32_t qkArrays : {16u, 24u, 32u, 40u}) {
        assert(state.configure(8, qkArrays));
        for (uint32_t id = 0; id < qkArrays; ++id)
            assert(state.owns(id, AttentionArrayOwner::Qk));
        for (uint32_t id = qkArrays; id < 64; ++id)
            assert(state.owns(id, AttentionArrayOwner::Pv));
        assert(state.cancel() == 0);
    }
    assert(!state.configure(8, 0));
    assert(!state.configure(8, 20));
    assert(!state.configure(8, 64));
    assert(!state.configure(0, 24));

    AttentionScoreFifo scoreFifo;
    assert(scoreFifo.reserve(0, tag, 32));
    assert(!scoreFifo.reserve(0, tag, 32));
    std::vector<float> first(16, 1.0f);
    std::vector<float> second(16, 2.0f);
    assert(scoreFifo.reserveWrite(0, tag, 0, first.size()));
    assert(!scoreFifo.reserveWrite(0, tag, 8, first.size()));
    assert(!scoreFifo.write(0, tag, 0, first));
    assert(scoreFifo.commitWrite(0, tag, 0, first));
    assert(!scoreFifo.commitWrite(0, tag, 0, first));
    assert(scoreFifo.canRead(0, tag, 0, 16));
    assert(!scoreFifo.canRead(0, tag, 8, 16));
    assert(!scoreFifo.ready(0, tag));
    assert(scoreFifo.write(0, tag, 16, second));
    assert(!scoreFifo.write(0, tag, 16, second));
    assert(scoreFifo.ready(0, tag));
    std::vector<float> scoreRead;
    assert(scoreFifo.read(0, tag, 8, 16, scoreRead));
    assert(scoreRead.size() == 16 && scoreRead.front() == 1.0f &&
           scoreRead.back() == 2.0f);
    assert(!scoreFifo.read(0, stale, 0, 16, scoreRead));
    assert(scoreFifo.cancelGeneration(stale.generation) == 0);
    assert(scoreFifo.release(0, tag));
    assert(scoreFifo.occupied() == 0);
    assert(scoreFifo.reserve(1, tag, 16));
    assert(scoreFifo.cancelGeneration(tag.generation) == 1);
    assert(scoreFifo.occupied() == 0);

    AttentionGenerationFence generationFence;
    assert(generationFence.begin(11));
    assert(!generationFence.begin(12));
    assert(generationFence.matches(11));
    assert(!generationFence.matches(12));
    assert(generationFence.cancel(11));
    assert(!generationFence.matches(11));
    assert(generationFence.begin(12));
    assert(!generationFence.retire(11));
    assert(generationFence.retire(12));
    const uint64_t cancelToken =
        AttentionGenerationCancelRegistry::token(3, 99);
    assert(!AttentionGenerationCancelRegistry::cancelledToken(cancelToken));
    AttentionGenerationCancelRegistry::cancel(3, 99);
    assert(AttentionGenerationCancelRegistry::cancelledToken(cancelToken));
    assert(!AttentionGenerationCancelRegistry::cancelledToken(
        AttentionGenerationCancelRegistry::token(4, 99)));

    AttentionOAccumulator accumulator;
    AttentionClusterTag oTag = tag;
    oTag.keyTile = 0;
    assert(accumulator.reserve(0, oTag, 2));
    assert(!accumulator.reserve(0, oTag, 2));
    uint64_t maxReady = 0;
    std::vector<float> pvSegment(16, 1.0f);
    for (uint32_t keyTile = 0; keyTile < 2; ++keyTile) {
        for (uint32_t row = 0; row < 16; ++row) {
            for (uint32_t panel = 0; panel < 8; ++panel) {
                uint64_t operation = 0;
                uint64_t ready = 0;
                assert(accumulator.submitSegment(
                    0, oTag, keyTile, row, panel, 0.5f, pvSegment,
                    maxReady, &operation, &ready));
                assert(operation != 0 && ready > maxReady);
                maxReady = ready;
            }
        }
        const auto completed = accumulator.progress(maxReady);
        assert(completed.size() == 128);
        assert(std::all_of(completed.begin(), completed.end(),
            [](const AttentionOAccumulator::Completion& item) {
                return item.ok && !item.drain;
            }));
        std::fill(pvSegment.begin(), pvSegment.end(), 2.0f);
    }
    uint64_t drainId = 0;
    uint64_t drainReady = 0;
    assert(accumulator.requestDrain(
        0, oTag, maxReady, &drainId, &drainReady));
    assert(accumulator.progress(drainReady - 1).empty());
    const auto drained = accumulator.progress(drainReady);
    assert(drained.size() == 1 && drained.front().ok && drained.front().drain);
    assert(drained.front().id == drainId);
    assert(drained.front().values.size() == 16 * 128);
    assert(std::all_of(drained.front().values.begin(),
                       drained.front().values.end(),
        [](float value) { return value == 2.5f; }));
    assert(accumulator.release(0, oTag));
    assert(accumulator.drained());
    assert(accumulator.scaleOperations() == 128);
    assert(accumulator.accumulateOperations() == 256);
    assert(accumulator.drainOperations() == 1);
    assert(accumulator.intervalOrderValid());
    assert(accumulator.readBusyCycles() == 256);
    assert(accumulator.writeBusyCycles() == 256);
    assert(accumulator.aluBusyCycles() == 256);
    assert(accumulator.readBusySpanCycles() ==
           accumulator.readBusyCycles() + accumulator.readIdleCycles());
    assert(accumulator.writeBusySpanCycles() ==
           accumulator.writeBusyCycles() + accumulator.writeIdleCycles());
    assert(accumulator.aluBusySpanCycles() ==
           accumulator.aluBusyCycles() + accumulator.aluIdleCycles());
    assert(accumulator.readMaxConcurrency() == 1);
    assert(accumulator.writeMaxConcurrency() == 1);
    assert(accumulator.aluMaxConcurrency() == 1);
    assert(accumulator.intervalOrderValid());
    assert(accumulator.reserve(1, oTag, 1));
    assert(accumulator.cancelGeneration(oTag.generation) == 1);
    assert(accumulator.drained());

    AttentionOAccumulator fusedAccumulator;
    assert(AttentionOAccumulator::kRowFmaBanks == 8);
    assert(AttentionOAccumulator::kRowFmaLanes == 128);
    assert(fusedAccumulator.reserve(0, oTag, 2));
    std::vector<float> pvRow(128, 1.0f);
    maxReady = 0;
    for (uint32_t keyTile = 0; keyTile < 2; ++keyTile) {
        for (uint32_t row = 0; row < 16; ++row) {
            uint64_t operation = 0;
            uint64_t ready = 0;
            assert(fusedAccumulator.submitRow(
                0, oTag, keyTile, row, 0.5f, pvRow, maxReady,
                &operation, &ready));
            assert(operation != 0 && ready > maxReady);
            maxReady = ready;
        }
        const auto completed = fusedAccumulator.progress(maxReady);
        assert(completed.size() == 16);
        assert(std::all_of(completed.begin(), completed.end(),
            [](const AttentionOAccumulator::Completion& item) {
                return item.ok && !item.drain;
            }));
        std::fill(pvRow.begin(), pvRow.end(), 2.0f);
    }
    assert(fusedAccumulator.requestDrain(
        0, oTag, maxReady, &drainId, &drainReady));
    const auto fusedDrain = fusedAccumulator.progress(drainReady);
    assert(fusedDrain.size() == 1 && fusedDrain.front().ok &&
           fusedDrain.front().drain);
    assert(std::all_of(fusedDrain.front().values.begin(),
                       fusedDrain.front().values.end(),
        [](float value) { return value == 2.5f; }));
    assert(fusedAccumulator.release(0, oTag));
    assert(fusedAccumulator.rowOperations() == 32);
    assert(fusedAccumulator.scaleOperations() == 128);
    assert(fusedAccumulator.accumulateOperations() == 256);
    // Sixteen fused row reads for the second key tile, plus 128 segment reads
    // when the completed O context is drained.
    assert(fusedAccumulator.readBusyCycles() == 144);
    assert(fusedAccumulator.writeBusyCycles() == 32);
    assert(fusedAccumulator.aluBusyCycles() == 32);
    assert(fusedAccumulator.intervalOrderValid());
    return 0;
}
