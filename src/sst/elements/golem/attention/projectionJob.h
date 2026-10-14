#ifndef SST_GOLEM_PROJECTION_JOB_H
#define SST_GOLEM_PROJECTION_JOB_H

#include <sst/elements/golem/attention/projectionJobAbi.h>
#include <sst/elements/golem/fp16.h>
#include <sst/elements/golem/attention/resourceTrace.h>
#include <sst/elements/golem/attention/diagnosticOracle.h>
#include <sst/elements/golem/globalmemory/globalmemory.h>
#include <sst/elements/golem/workercmdproc/workercmdproc.h>

#include <array>
#include <algorithm>
#include <functional>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <utility>
#include <vector>

namespace SST { namespace Golem {

// A bounded, descriptor-driven producer of the raw and panel Q/K/V layouts.
// The row-major path holds one array-width token group; the reuse path holds one
// 256-token input block in bounded local GM. Both retain one V tile.
class ProjectionJob {
public:
    ProjectionJob(GlobalMemoryAPI* memory, WorkerCommandProcessorAPI* processor, SFUAPI* vectorEngine)
        : vectorEngine_(vectorEngine), memory_(memory), processor_(processor) {
        const char* width = std::getenv("GOLEM_PROJECTION_ARRAYS");
        uint32_t count = width ? static_cast<uint32_t>(std::strtoul(width, nullptr, 10)) : 16;
        if (count != 16 && count != 32 && count != 64) count = 16;
        for (uint32_t id = 0; id < count; ++id) arrays_.push_back(id);
    }

    std::vector<uint8_t> completionBytes() const {
        const uint64_t value=desc_.completion_value ? desc_.completion_value : 1;
        std::vector<uint8_t> bytes(8); std::memcpy(bytes.data(),&value,8); return bytes;
    }

    bool start(const ProjectionJobDesc& desc) {
        if (phase_ != Phase::Idle || processor_ == nullptr || memory_ == nullptr ||
            processor_->isBusy() || desc.magic != GOLEM_PROJECTION_JOB_MAGIC ||
            desc.size_bytes != sizeof(desc) || desc.hidden_dim == 0 ||
            desc.hidden_dim % 64 || desc.hidden_dim > 2048 ||
            desc.head_dim == 0 || desc.head_dim % 64 ||
            desc.query_heads == 0 || desc.kv_heads == 0 ||
            desc.rows_per_node == 0 || desc.rows_per_node % 256 ||
            desc.manager_slot >= 4 || desc.reserved < 1 || desc.reserved > 4 || desc.scratch_addr == 0 ||
            desc.input_addr == 0 || desc.weights_addr == 0 ||
            desc.q_addr == 0 || desc.k_addr == 0 || desc.v_addr == 0 ||
            desc.q_panel_addr == 0 || desc.k_panel_addr == 0 ||
            desc.v_panel_addr == 0 || desc.completion_addr == 0) return false;
        desc_ = desc;
        std::printf("[PROJECTION_WORKER_OWNER] core=%u manager=%u lane=%u row_offset=%u row_count=64\n",
            executionCore(), desc_.manager_slot, desc_.reserved - 1, (desc_.reserved - 1) * 64);
        kind_ = head_ = dimTile_ = inputTile_ = 0;
        row_ = (desc.reserved - 1) * 64;
        status_ = 0;
        criticalPathTrace_ = enabledOptIn("GOLEM_CRITICAL_PATH_TRACE");
        causalTrace_ = enabledOptIn("GOLEM_PROJECTION_CAUSAL_TRACE");
        tracePhase_ = Phase::Idle;
        traceSequence_ = 0;
        tracePredecessor_ = 0;
        traceCycle_ = 0;
        residentTiles_.fill(~uint64_t{0});
        activeOperandBank_ = 0;
        cachedTiles_.fill(~uint64_t{0});
        weightLoads_ = weightPrograms_ = weightReuses_ = 0;
        inputLoads_ = 0;
        localReadBytes_ = localWriteBytes_ = 0;
        inputRaw_.clear();
        localInputRow_ = ~uint32_t{0};
        phaseCycles_.fill(0);
        localRoleCycles_.fill(0);
        // The reuse path stages 256 input rows and per-row FP16 partials in
        // local GM, using the array's single weight/input operand context.
        const uint64_t gmBase = memory_->getBaseAddr();
        const uint64_t gmSize = memory_->getSize();
        pairedWeights_ = desc.head_dim == 64 && desc.hidden_dim >= 256 &&
            desc.hidden_dim % 128 == 0;
        // D64 does not stage split-dimension raw rows. Reuse that region
        // for input, and exclude the GM tail's 64 DMA status bytes.
        const uint64_t reuseBytes = (pairedWeights_ ? 0xE8000 : 0xF8000) +
            static_cast<uint64_t>(256) * desc.hidden_dim * 2;
        reuseBlock_ = (pairedWeights_ ||
            (desc.head_dim == 128 && desc.hidden_dim <= 512)) &&
            gmSize >= 64 &&
            desc.scratch_addr >= gmBase &&
            desc.scratch_addr - gmBase <= gmSize - 64 &&
            reuseBytes <= gmSize - 64 - (desc.scratch_addr - gmBase);
        pairedWeights_ = pairedWeights_ && reuseBlock_;
        fuseNorm_ = (desc.flags & GOLEM_PROJECTION_FUSE_NORM) != 0;
        fuseRope_ = (desc.flags & GOLEM_PROJECTION_FUSE_ROPE) != 0;
        audit_ = (desc.flags & GOLEM_PROJECTION_AUDIT_OUTPUT) != 0;
        // Reuse existing scratch below the weight cache. Restrict this execution
        // plan to the established paired/shared-input traversal, not hardware.
        if ((desc.flags & ~7u) || ((fuseNorm_ || fuseRope_) &&
            (!pairedWeights_ || !enabled("GOLEM_PROJECTION_SHARED_INPUT") ||
             enabledOptIn("GOLEM_PROJECTION_ROW_FIRST") || vectorEngine_ == nullptr)) ||
            (fuseNorm_ && (!desc.gamma_addr || !desc.norm_output_addr)) ||
            (fuseRope_ && !desc.rope_table_addr)) return false;
        gammaReady_ = false;
        ropeCacheBlock_ = ~uint32_t{0};
        normRow_ = 0;
        fusionReadBytes_ = fusionWriteBytes_ = normVectorCycles_ = ropeVectorCycles_ = 0;
        normJobs_ = ropeJobs_ = normAuditBytes_ = rawAuditBytes_ = panelBytes_ = 0;
        vectorResult_ = SFUVectorResult{};
        if (!pairedWeights_ && rowGroup() != 16) arrays_.resize(16);
        rowFirst_ = pairedWeights_ && enabledOptIn("GOLEM_PROJECTION_ROW_FIRST");
        // One row group owns the existing output vector for the entire dot
        // product. No other rows overwrite it between hidden-tile pairs.
        retainOutput_ = pairedWeights_ && !rowFirst_ && rowGroup() == 64 &&
            enabledOptIn("GOLEM_PROJECTION_RETAIN_OUTPUT");
        retainedPairs_ = 0;
        cBufferPartial_ = pairedWeights_ && !rowFirst_ && !retainOutput_ &&
            enabledOptIn("GOLEM_PROJECTION_CBUFFER_PARTIAL");
        // Retired scheduling must not silently fall back in direct SST runs.
        if (enabledOptIn("GOLEM_PROJECTION_WEIGHT_STATIONARY")) return false;
        if (cBufferPartial_ && !processor_->beginProjectionPartialBuffer(256 * 128))
            return false;
        cBufferReadyCycle_ = cBufferReadReadyCycle_ = 0;
        cBufferReadBytes_ = cBufferWriteBytes_ = 0;
        cBufferPrefetchRow_ = cBufferPrefetchTile_ = ~uint32_t{0};
        // Budget width-dependent transfer snapshots against the same
        // local-GM capacity, including the reserved DMA tail. This is not an
        // additional array bank or an unbounded software cache.
        pipeline_ = pairedWeights_ && enabled("GOLEM_PROJECTION_PIPELINE") &&
            !rowFirst_ &&
            reuseBytes + stagingBytes() <= gmSize - 64 - (desc.scratch_addr - gmBase);
        inputPairPrefetch_ = enabledOptIn("GOLEM_PROJECTION_INPUT_PAIR_PREFETCH");
        localWeightPrefetch_ = enabledOptIn("GOLEM_PROJECTION_LOCAL_WEIGHT_PREFETCH");
        const char* groups = std::getenv("GOLEM_PROJECTION_ARRAY_GROUPS");
        arrayGroups_ = groups ? static_cast<uint32_t>(std::strtoul(groups, nullptr, 10)) : 1;
        overlapTrace_ = enabledOptIn("GOLEM_PROJECTION_OVERLAP_TRACE");
        if ((arrayGroups_ != 1 && arrayGroups_ != 2) ||
            ((inputPairPrefetch_ || localWeightPrefetch_ || arrayGroups_ == 2) &&
             (!pipeline_ || !retainOutput_ || !enabled("GOLEM_PROJECTION_SHARED_INPUT")))) return false;
        for (uint32_t group = 0; group < 2; ++group) {
            groupArrays_[group].assign(arrays_.begin() + (arrayGroups_ == 2 ? group * 32 : 0),
                arrays_.begin() + (arrayGroups_ == 2 ? (group + 1) * 32 : 0));
        }
        groupRemaining_.fill(0);
        groupCompletedTile_.fill(~uint32_t{0});
        programGroup_ = 0;
        groupBufferPending_ = false;
        arrayLaunches_ = computeBusyCycles_ = bufferComputeOverlapCycles_ = gmComputeOverlapCycles_ = 0;
        localWeightPrefetches_ = localWeightHits_ = localWeightWaitCycles_ = 0;
        localWeightTile_ = ~uint64_t{0};
        prefetchWeight_ = AsyncRead{};
        const char* budget = std::getenv("GOLEM_PROJECTION_INPUT_PREFETCH_BUDGET");
        inputPrefetchBudget_ = pipeline_ && budget ?
            static_cast<uint32_t>(std::min<unsigned long>(64, std::strtoul(budget, nullptr, 10))) : 0;
        // A scratch layout choice, with the same GM footprint and snapshots.
        // Keep unsupported batch/group shapes and audit paths on row-major.
        const uint32_t normBatchRows = 8192 / desc.hidden_dim;
        packedInput_ = pairedWeights_ && fuseNorm_ && !audit_ && !rowFirst_ &&
            enabledOptIn("GOLEM_PROJECTION_PACKED_NORM_INPUT") &&
            8192 % desc.hidden_dim == 0 && normBatchRows != 0 &&
            rowGroup() % normBatchRows == 0;
        inputReadRequests_ = backgroundInputReadRequests_ = weightReadRequests_ = normWriteRequests_ = 0;
        inputBudgetStalls_ = 0;
        localPackedNormWrite_ = localNormWrite_ = false;
        wrapPrefetch_ = pipeline_ && enabledOptIn("GOLEM_PROJECTION_WRAP_PREFETCH");
        restoreAheadEnabled_ = pipeline_ && cBufferPartial_ &&
            enabledOptIn("GOLEM_PROJECTION_PARTIAL_RESTORE_AHEAD");
        outputDrained_ = restoreAheadIssued_ = restoreAheadReady_ = false;
        restoreAheadRow_ = restoreAheadTile_ = ~uint32_t{0};
        partialReadCycles_ = finalReadCycles_ = partialReads_ = finalReads_ = 0;
        restoreCBufferWaitCycles_ = restoreSubmitCycles_ = restoreTransferCycles_ = 0;
        restoreAheadWaitCycles_ = restoreAheadConsumeCycles_ = 0;
        restoresForeground_ = restoresAhead_ = restoreAheadRetries_ = 0;
        cBufferPrefetchReads_ = cBufferDemandReads_ = cBufferStores_ = 0;
        cBufferPrefetchResidence_ = cBufferDemandResidence_ = cBufferStoreResidence_ = 0;
        sharedInput_ = pairedWeights_ && !rowFirst_ && enabled("GOLEM_PROJECTION_SHARED_INPUT");
        weightPrefetchEnabled_ = pairedWeights_ && enabled("GOLEM_PROJECTION_WEIGHT_PREFETCH");
        const char* lookahead = std::getenv("GOLEM_PROJECTION_WEIGHT_PREFETCH_DEPTH");
        weightPrefetchDepth_ = lookahead ? static_cast<uint32_t>(std::min<unsigned long>(8, std::max<unsigned long>(1, std::strtoul(lookahead, nullptr, 10)))) : 4;
        weightPrefetchBusy_ = false;
        weightPrefetchTile_ = ~uint64_t{0};
        prefetchedTiles_.fill(~uint64_t{0});
        weightPrefetchRequests_ = weightPrefetchHits_ = weightPrefetchWaitCycles_ = 0;
        prefetchInput_ = AsyncRead{};
        prefetchPartial_ = AsyncRead{};
        partialStore_ = AsyncStore{};
        scatterIssued_ = scatterReady_ = computing_ = false;
        inputPrefetches_ = partialPrefetches_ = scatterPrefetches_ = 0;
        backgroundReadCycles_ = backgroundWriteCycles_ = 0;
        backgroundReadRejects_ = backgroundWriteRejects_ = 0;
        blockStart_ = 0;
        blockLoaded_ = false;
        writeOutstanding_ = 0;
        writes_.clear();
        nextInputIssued_ = false;
        nextInputReady_ = false;
        nextInputFailed_ = false;
        loadedInputRow_ = ~uint32_t{0};
        startCycle_ = 0;
        phase_ = reuseBlock_ ? Phase::LoadBlock : Phase::LoadInput;
        vTile_.assign(64 * desc.head_dim, 0);
        rawTile_.assign(rowGroup() * desc.head_dim * 2, 0);
        return true;
    }

    bool complete() const { return phase_ == Phase::Complete; }
    bool active() const { return phase_ != Phase::Idle && phase_ != Phase::Complete; }
    uint64_t status() const { return status_; }
    const ProjectionJobDesc& descriptor() const { return desc_; }
    void retire() {
        if (cBufferPartial_) processor_->endProjectionPartialBuffer();
        cBufferPartial_ = false;
        phase_ = Phase::Idle;
    }

    void tick(uint64_t cycle, uint64_t nowTick) {
        traceCycle_ = cycle;
        tracePhaseTransition(cycle);
        if (!active()) return;
        nowTick_ = nowTick;
        if (startCycle_ == 0) startCycle_ = cycle;
        ++phaseCycles_[static_cast<size_t>(phase_)];
        if (phase_ == Phase::LocalRead) ++localRoleCycles_[static_cast<size_t>(localReadRole_)];
        if (phase_ == Phase::ReadOutput) {
            if (partialOutput()) ++partialReadCycles_;
            else ++finalReadCycles_;
        }
        if (phase_ == Phase::RestoreOutputReady) {
            if (matchingAheadRestore()) {
                if (restoreAheadReady_) ++restoreAheadConsumeCycles_;
                else ++restoreAheadWaitCycles_;
            } else if (cBufferPartial_ && cycle < cBufferReadReadyCycle_) {
                ++restoreCBufferWaitCycles_;
            } else if (pending_) ++restoreTransferCycles_;
            else ++restoreSubmitCycles_;
        }
        pumpWeightPrefetch();
        pumpLocalWeightPrefetch();
        const bool arrayBusy = arrayGroups_ == 2 ?
            groupRemaining_[0] || groupRemaining_[1] : computing_;
        if (arrayBusy) {
            ++computeBusyCycles_;
            if (groupBufferPending_ || (pending_ &&
                (phase_ == Phase::ProgramWeightReady || phase_ == Phase::ProgramInput)))
                ++bufferComputeOverlapCycles_;
            if (prefetchInput_.active || prefetchWeight_.active || phase_ == Phase::LocalRead)
                ++gmComputeOverlapCycles_;
        }
        // Bound the existing future-input lane and submit foreground chunks
        // first. The underlying GM FIFO, ports and latency are unchanged.
        if (inputPrefetchBudget_) pumpLocalAccess();
        if (pipeline_) pumpBackground(cycle);
        pumpAheadRestore(cycle);
        if (!inputPrefetchBudget_) pumpLocalAccess();
        if (pending_) return;
        switch (phase_) {
        case Phase::LoadBlock: {
            if (fuseNorm_) { beginNormBlock(); break; }
            const uint64_t addr = desc_.input_addr +
                static_cast<uint64_t>(blockRowStart()) * desc_.hidden_dim * 2;
            const size_t bytes = static_cast<size_t>(64) * desc_.hidden_dim * 2;
            pending_ = true;
            memory_->dma_read_from_host_to_globalmem(addr, bytes, inputBlockAddr() + static_cast<uint64_t>((desc_.reserved - 1) * 64) * desc_.hidden_dim * 2,
                [this](bool ok) {
                    pending_ = false;
                    if (!ok) { fail(); return; }
                    ++inputLoads_;
                    blockLoaded_ = true;
                    phase_ = Phase::LoadInput;
                });
            break;
        }
        case Phase::NormBatch:
            beginNormBatch();
            break;
        case Phase::NormWait:
            if (nowTick_ >= vectorResult_.readyTick) finishNormBatch();
            break;
        case Phase::RopeWait:
            if (nowTick_ >= vectorResult_.readyTick) {
                std::vector<uint8_t> packed(vectorResult_.values.size() * 2);
                for (size_t i = 0; i < vectorResult_.values.size(); ++i)
                    encode(packed, i, vectorResult_.values[i]);
                beginFusionWrite(desc_.scratch_addr + 0x30000, std::move(packed),
                    [this]() {
                        beginFusionRead(desc_.scratch_addr + 0x30000, outputValues_.size() * 2,
                            [this](const std::vector<uint8_t>& raw) {
                                panelValues_.resize(raw.size() / 2);
                                for (size_t i = 0; i < panelValues_.size(); ++i)
                                    panelValues_[i] = decode(raw, i);
                                finishOutput();
                            });
                    });
            }
            break;
        case Phase::LoadInput: {
            if (reuseBlock_) { prepareInput(); break; }
            if (loadedInputRow_ == row_) {
                prepareInput();
                break;
            }
            const uint64_t addr = desc_.input_addr +
                static_cast<uint64_t>(row_) * desc_.hidden_dim * 2;
            const size_t bytes = static_cast<size_t>(rowGroup()) * desc_.hidden_dim * 2;
            pending_ = true;
            memory_->dma_read_from_host_to_globalmem(addr, bytes, desc_.scratch_addr,
                [this](bool ok) {
                    if (!ok) { pending_ = false; fail(); return; }
                    loadedInputRow_ = row_;
                    ++inputLoads_;
                    pending_ = false;
                    phase_ = Phase::LoadInput;
                });
            break;
        }
        case Phase::LoadWeight: {
            const uint64_t tile = weightTile();
            if (!cacheTileEnabled()) {
                phase_ = Phase::LoadWeightDma;
                break;
            }
            if (cachedTiles_[cacheSlot()] == tile) {
                if (prefetchedTiles_[cacheSlot()] == tile) {
                    ++weightPrefetchHits_;
                    prefetchedTiles_[cacheSlot()] = ~uint64_t{0};
                }
                phase_ = Phase::ProgramWeight;
                break;
            }
            // The background DMA owns this cache slot until completion.
            // Wait for it instead of submitting a duplicate demand read.
            if (weightPrefetchBusy_ && weightPrefetchTile_ == tile) {
                ++weightPrefetchWaitCycles_;
                break;
            }
            const uint64_t addr = desc_.weights_addr + tile * 8192;
            pending_ = true;
            readWeightToGM(addr, 8192, cachedWeightAddr(),
                [this, tile](bool ok) {
                    pending_ = false;
                    if (ok) {
                        cachedTiles_[cacheSlot()] = tile;
                        ++weightLoads_;
                    }
                    phase_ = ok ? Phase::ProgramWeight : Phase::Failed;
                    if (!ok) status_ = 1;
                });
            break;
        }
        case Phase::LoadWeightDma: {
            const uint64_t tile = weightTile();
            const uint64_t addr = desc_.weights_addr + tile * 8192;
            pending_ = true;
            readWeightToGM(addr, 8192, desc_.scratch_addr + 0x2000,
                [this](bool ok) {
                    pending_ = false;
                    if (ok) ++weightLoads_;
                    phase_ = ok ? Phase::ProgramWeight : Phase::Failed;
                    if (!ok) status_ = 1;
                });
            break;
        }
        case Phase::ProgramWeight: {
            if (localWeightPrefetch_ && localWeightTile_ == weightTile() &&
                (prefetchWeight_.active || prefetchWeight_.ready)) {
                if (!prefetchWeight_.ready) { ++localWeightWaitCycles_; break; }
                weightRaw_ = std::move(prefetchWeight_.data);
                traceSnapshotConsume(prefetchWeight_, "weight_snapshot_residence");
                prefetchWeight_ = AsyncRead{};
                ++localWeightHits_;
                phase_ = Phase::ProgramWeightReady;
                break;
            }
            beginLocalRead(cacheTileEnabled() ? cachedWeightAddr() :
                desc_.scratch_addr + 0x2000, 8192,
                [this](const std::vector<uint8_t>& raw) {
                    weightRaw_ = raw;
                    phase_ = Phase::ProgramWeightReady;
                }, LocalReadRole::Weight);
            break;
        }
        case Phase::ProgramWeightReady: {
            if (arrayGroups_ == 2) {
                programGroup_ = 0;
                phase_ = Phase::GroupProgramWeight;
                break;
            }
            const CausalKey key = causalKey();
            traceCausal("decode_start", "weight", key);
            std::vector<double> matrix(4096);
            for (size_t i = 0; i < matrix.size(); ++i) matrix[i] = decode(weightRaw_, i);
            traceCausal("decode_done", "weight", key);
            pending_ = true;
            const uint64_t commandTag = ++tag_;
            traceCausal("matrix_submit", "weight", key, commandTag);
            if (!processor_->programGemmMatrixGroupClassBankAsync(arrays_, activeOperandBank_, matrix, 2,
                    AttentionClusterTrafficClass::ProjectionWeights, commandTag, cycle,
                    [this, key, commandTag](bool ok, uint64_t) {
                        traceCausal(ok ? "matrix_done" : "matrix_failed", "weight", key, commandTag);
                        pending_ = false;
                        if (ok) {
                            residentTiles_[activeOperandBank_] = weightTile();
                            ++weightPrograms_;
                        }
                        phase_ = ok ? Phase::ProgramInput : Phase::Failed;
                        if (!ok) status_ = 1;
                    })) { pending_ = false; break; }
            break;
        }
        case Phase::ProgramInput: {
            if (pipeline_ && scatterIssued_ && scatterRow_ == row_ &&
                scatterTile_ == inputTile_) {
                if (!scatterReady_) break;
                scatterIssued_ = scatterReady_ = false;
                phase_ = !retainOutput_ && inputTile_ != 0 && inputTile_ % 2 == 0 ?
                    Phase::RestoreOutput : Phase::Launch;
                break;
            }
            if (inputTile_ == 1 && nextInputReady_) {
                nextInputReady_ = false;
                nextInputIssued_ = false;
                phase_ = Phase::Launch;
                break;
            }
            const CausalKey key = causalKey();
            std::vector<double> values(input_.begin(), input_.end());
            pending_ = true;
            const uint64_t commandTag = ++tag_;
            traceCausal("scatter_submit", "input", key, commandTag);
            if (!processor_->programGemmInputScatterBankAsync(arrays_, activeOperandBank_, values, 2,
                    AttentionClusterTrafficClass::ProjectionInputScatter, commandTag, cycle,
                    [this, key, commandTag](bool ok, uint64_t) {
                        traceCausal(ok ? "scatter_done" : "scatter_failed", "input", key, commandTag);
                        pending_ = false;
                        if (inputTile_ == 1) {
                            nextInputIssued_ = false;
                            nextInputReady_ = false;
                        }
                        phase_ = ok ? (reuseBlock_ && !rowFirst_ && !retainOutput_ && inputTile_ != 0 &&
                            (!pairedWeights_ || inputTile_ % 2 == 0) ?
                            Phase::RestoreOutput : Phase::Launch) : Phase::Failed;
                        if (!ok) status_ = 1;
                    })) { pending_ = false; break; }
            break;
        }
        case Phase::WaitInputPrefetch:
            if (nextInputFailed_) {
                nextInputIssued_ = false;
                nextInputFailed_ = false;
                phase_ = Phase::LoadWeight;
            } else if (nextInputReady_) {
                nextInputIssued_ = false;
                nextInputReady_ = false;
                phase_ = Phase::LoadWeight;
            }
            break;
        case Phase::Launch: {
            const CausalKey key = causalKey();
            traceCausal("mac_submit", "compute", key);
            pending_ = true;
            remaining_ = rowGroup();
            if (!processor_->launchGemmArrayGroupActiveBank(arrays_, activeOperandBank_,
                    inputTile_ == 0 ? 0 : 1, 64, cycle,
                    [this, key](uint32_t, uint64_t) {
                        if (--remaining_ == 0) {
                            traceCausal("mac_done", "compute", key);
                            pending_ = false;
                            computing_ = false;
                            if (pairedWeights_ && inputTile_ % 2 == 0) {
                                ++inputTile_;
                                phase_ = Phase::LoadInput;
                                return;
                            }
                            if (retainOutput_ && inputTile_ + 1 < desc_.hidden_dim / 64) {
                                ++inputTile_;
                                ++retainedPairs_;
                                phase_ = Phase::LoadInput;
                                return;
                            }
                            if (rowFirst_ && ++inputTile_ < desc_.hidden_dim / 64) {
                                phase_ = Phase::LoadInput;
                                return;
                            }
                            phase_ = reuseBlock_ ? Phase::ReadOutput :
                                (++inputTile_ < desc_.hidden_dim / 64 ?
                                    Phase::LoadInput : Phase::ReadOutput);
                        }
                    })) { pending_ = false; break; }
            ++arrayLaunches_;
            if (pipeline_) {
                outputDrained_ = false;
                computing_ = true;
                computeBank_ = activeOperandBank_;
                const bool nextRow = row_ + rowGroup() < blockRowEnd();
                const bool nextPair = (wrapPrefetch_ && rowGroup() < 64 || inputPairPrefetch_) && !nextRow &&
                    inputTile_ + 2 < desc_.hidden_dim / 64;
                if (inputTile_ % 2 == 0 && (nextRow || nextPair)) {
                    // At the last row group, the same bounded future-input slot
                    // can fetch row zero of the next pair. Its GM source and
                    // previous partial are already resident; no new storage.
                    prefetchRow_ = nextRow ? row_ + rowGroup() : blockRowStart();
                    prefetchTile_ = nextRow ? inputTile_ : inputTile_ + 2;
                    armRead(prefetchInput_, inputPairAddr(prefetchRow_, prefetchTile_),
                        rowGroup() * 256, packedInput_ ? rowGroup() * 256 : 256);
                    prefetchInput_.key.row = prefetchRow_;
                    prefetchInput_.key.tile = prefetchTile_;
                    ++inputPrefetches_;
                    if (prefetchTile_ != 0 && !retainOutput_ && !cBufferPartial_) {
                        armRead(prefetchPartial_, desc_.scratch_addr + 0xE0000 +
                            static_cast<uint64_t>(prefetchRow_ - blockStart_) * 128,
                            rowGroup() * 64 * 2, rowGroup() * 64 * 2);
                        ++partialPrefetches_;
                    }
                    if (prefetchTile_ != 0 && cBufferPartial_) {
                        uint64_t ready = 0;
                        if (!processor_->readProjectionPartial(
                                static_cast<uint64_t>(prefetchRow_ - blockStart_) * 128,
                                rowGroup() * 128, cycle, cBufferPrefetchRaw_, ready)) {
                            fail();
                            break;
                        }
                        cBufferPrefetchReadyCycle_ = ready;
                        cBufferPrefetchRow_ = prefetchRow_;
                        cBufferPrefetchTile_ = prefetchTile_;
                        cBufferReadBytes_ += cBufferPrefetchRaw_.size();
                        ++cBufferPrefetchReads_;
                        cBufferPrefetchResidence_ += ready - cycle;
                        ++partialPrefetches_;
                    }
                }
            }
            break;
        }
        case Phase::GroupProgramWeight: {
            // Only this group's physical operands may be programmed. The other
            // group retains its own output and may still be computing.
            if (groupRemaining_[programGroup_]) break;
            if (inputTile_ && groupCompletedTile_[programGroup_] != inputTile_ - 1) { fail(); break; }
            std::vector<double> matrix(4096);
            for (size_t i = 0; i < matrix.size(); ++i) matrix[i] = decode(weightRaw_, i);
            pending_ = true;
            if (!processor_->programGemmMatrixGroupClassBankAsync(groupArrays_[programGroup_], 0, matrix, 2,
                    AttentionClusterTrafficClass::ProjectionWeights, ++tag_, cycle,
                    [this](bool ok, uint64_t) {
                        pending_ = false;
                        groupBufferPending_ = false;
                        if (!ok) { fail(); return; }
                        ++weightPrograms_;
                        phase_ = Phase::GroupProgramInput;
                    })) { pending_ = false; break; }
            groupBufferPending_ = true;
            traceGroup("program", programGroup_, inputTile_, cycle);
            break;
        }
        case Phase::GroupProgramInput: {
            const auto first = input_.begin() + programGroup_ * 32 * 64;
            std::vector<double> values(first, first + 32 * 64);
            pending_ = true;
            if (!processor_->programGemmInputScatterBankAsync(groupArrays_[programGroup_], 0, values, 2,
                    AttentionClusterTrafficClass::ProjectionInputScatter, ++tag_, cycle,
                    [this](bool ok, uint64_t) {
                        pending_ = false;
                        groupBufferPending_ = false;
                        if (!ok) { fail(); return; }
                        phase_ = Phase::GroupLaunch;
                    })) { pending_ = false; break; }
            groupBufferPending_ = true;
            break;
        }
        case Phase::GroupLaunch: {
            const uint32_t group = programGroup_, tile = inputTile_;
            if (groupRemaining_[group]) { fail(); break; }
            groupRemaining_[group] = 32;
            if (!processor_->launchGemmArrayGroupActiveBank(groupArrays_[group], 0,
                    tile == 0 ? 0 : 1, 64, cycle,
                    [this, group, tile](uint32_t, uint64_t doneCycle) {
                        if (--groupRemaining_[group] == 0) {
                            groupCompletedTile_[group] = tile;
                            traceGroup("done", group, tile, doneCycle);
                        }
                    })) { groupRemaining_[group] = 0; break; }
            ++arrayLaunches_;
            traceGroup("launch", group, tile, cycle);
            if (group == 0 && inputPairPrefetch_ && tile % 2 == 0 && tile + 2 < desc_.hidden_dim / 64) {
                if (prefetchInput_.active || prefetchInput_.ready) { fail(); break; }
                prefetchRow_ = row_;
                prefetchTile_ = tile + 2;
                armRead(prefetchInput_, inputPairAddr(row_, tile + 2),
                    rowGroup() * 256, packedInput_ ? rowGroup() * 256 : 256);
                ++inputPrefetches_;
            }
            if (++programGroup_ < 2) phase_ = Phase::GroupProgramWeight;
            else if (tile + 1 == desc_.hidden_dim / 64) phase_ = Phase::GroupDrain;
            else {
                if (tile % 2) ++retainedPairs_;
                ++inputTile_;
                phase_ = Phase::LoadInput;
            }
            break;
        }
        case Phase::GroupDrain:
            if (!groupRemaining_[0] && !groupRemaining_[1]) phase_ = Phase::ReadOutput;
            break;
        case Phase::ReadOutput:
            pending_ = true;
            if (!processor_->readGemmOutputGroupClassAsync(arrays_, 2,
                    AttentionClusterTrafficClass::ProjectionOutput, ++tag_, cycle,
                    [this](bool ok, uint64_t completedCycle, const std::vector<double>& values) {
                        pending_ = false;
                        if (!ok || values.size() != this->rowGroup() * 64) { fail(); return; }
                        outputDrained_ = true;
                        if (partialOutput()) ++partialReads_;
                        else ++finalReads_;
                        if (reuseBlock_ && inputTile_ + 1 < desc_.hidden_dim / 64) {
                            if (pipeline_) {
                                outputValues_ = values;
                                phase_ = Phase::StagePartial;
                            } else stagePartial(values, completedCycle);
                        } else {
                            outputValues_ = values;
                            phase_ = Phase::PrepareOutput;
                        }
                    })) { pending_ = false; break; }
            break;
        case Phase::StagePartial:
            if (cBufferPartial_) {
                stagePartial(outputValues_, cycle);
                break;
            }
            if (!partialStore_.active) {
                partialStore_.addr = partialAddr();
                partialStore_.data.resize(rowGroup() * 64 * 2);
                for (size_t i = 0; i < outputValues_.size(); ++i)
                    encode(partialStore_.data, i, outputValues_[i]);
                partialStore_.offset = 0;
                partialStore_.inFlight = false;
                partialStore_.active = true;
                advance();
            }
            break;
        case Phase::RestoreOutput: {
            if (matchingAheadRestore()) {
                phase_ = Phase::RestoreOutputReady;
                break;
            }
            if (cBufferPartial_) {
                if (cBufferPrefetchRow_ == row_ && cBufferPrefetchTile_ == inputTile_) {
                    partialRaw_ = std::move(cBufferPrefetchRaw_);
                    cBufferReadReadyCycle_ = cBufferPrefetchReadyCycle_;
                    cBufferPrefetchRow_ = cBufferPrefetchTile_ = ~uint32_t{0};
                    phase_ = Phase::RestoreOutputReady;
                    break;
                }
                if (!processor_->readProjectionPartial(partialOffset(), rowGroup() * 128,
                        cycle, partialRaw_, cBufferReadReadyCycle_)) { fail(); break; }
                cBufferReadBytes_ += partialRaw_.size();
                ++cBufferDemandReads_;
                cBufferDemandResidence_ += cBufferReadReadyCycle_ - cycle;
                phase_ = Phase::RestoreOutputReady;
                break;
            }
            if (pipeline_ && prefetchRow_ == row_ && prefetchTile_ == inputTile_ &&
                (prefetchPartial_.active || prefetchPartial_.ready)) {
                if (!prefetchPartial_.ready) break;
                partialRaw_ = std::move(prefetchPartial_.data);
                traceSnapshotConsume(prefetchPartial_, "partial_snapshot_residence");
                prefetchPartial_ = AsyncRead{};
                phase_ = Phase::RestoreOutputReady;
                break;
            }
            // A store owns its snapshot until timed SRAM completion. Do not
            // read a matching address, even under slow-write/backpressure profiles.
            if (pipeline_ && partialStore_.active && partialStore_.addr == partialAddr()) break;
            beginLocalRead(partialAddr(), rowGroup() * 64 * 2,
                [this](const std::vector<uint8_t>& raw) {
                    partialRaw_ = raw;
                    phase_ = Phase::RestoreOutputReady;
                });
            break;
        }
        case Phase::RestoreOutputReady: {
            if (matchingAheadRestore()) {
                if (!restoreAheadReady_) break;
                restoreAheadIssued_ = restoreAheadReady_ = false;
                phase_ = Phase::Launch;
                break;
            }
            if (cBufferPartial_ && cycle < cBufferReadReadyCycle_) break;
            std::vector<double> values(rowGroup() * 64);
            for (size_t i = 0; i < values.size(); ++i) values[i] = decode(partialRaw_, i);
            pending_ = true;
            if (!processor_->writeGemmOutputGroupClassAsync(arrays_, values, 2,
                    AttentionClusterTrafficClass::ProjectionOutput, ++tag_, cycle,
                    [this](bool ok, uint64_t) {
                        pending_ = false;
                        phase_ = ok ? Phase::Launch : Phase::Failed;
                        if (!ok) status_ = 1;
                    })) { pending_ = false; break; }
            ++restoresForeground_;
            break;
        }
        case Phase::PrepareOutput:
            if (fuseRope_ && kind_ != 2) { beginRopeOutput(); break; }
            panelValues_ = outputValues_;
            if (reuseBlock_ && dimTile_ != 0) {
                beginLocalRead(rawBlockAddr(), rawTile_.size(),
                    [this](const std::vector<uint8_t>& raw) {
                        rawTile_ = raw;
                        finishOutput();
                    });
            } else finishOutput();
            break;
        case Phase::WriteOutput:
            issueWritebacks();
            if (!writes_.empty()) break;  // Bound staging to this output group.
            // Output DMA owns copies of the payload vectors, so the next
            // input/array tile can proceed while bounded writebacks drain.
            advance();
            break;
        case Phase::WriteDrain:
            issueWritebacks();
            if (writes_.empty() && writeOutstanding_ == 0 && !partialStore_.active) phase_ = Phase::WriteCompletion;
            break;
        case Phase::WriteCompletion:
            if (cBufferPartial_ && cycle < cBufferReadyCycle_) break;
            pending_ = true;
            memory_->dma_write_to_host(desc_.completion_addr, 8,
                completionBytes(),
                [this, cycle](bool ok) {
                    pending_ = false;
                    phase_ = Phase::Complete;
                    status_ = (ok && status_ == 0) ? 0 : 1;
                    std::printf("[PROJECTION_JOB] core=%u start=%llu end=%llu cycles=%llu weight_loads=%llu weight_programs=%llu weight_reuses=%llu input_loads=%llu status=%llu\n",
                        executionCore(),
                        static_cast<unsigned long long>(startCycle_),
                        static_cast<unsigned long long>(cycle),
                        static_cast<unsigned long long>(cycle - startCycle_),
                        static_cast<unsigned long long>(weightLoads_),
                        static_cast<unsigned long long>(weightPrograms_),
                        static_cast<unsigned long long>(weightReuses_),
                        static_cast<unsigned long long>(inputLoads_),
                        static_cast<unsigned long long>(status_));
                    static const char* phaseNames[] = {"idle", "load_block", "load_input", "load_weight", "load_weight_dma", "program_weight",
                        "program_weight_ready", "program_input", "launch", "wait_input_prefetch", "read_output",
                        "restore_output", "restore_output_ready", "prepare_output", "norm_batch", "norm_wait", "rope_wait", "local_read", "local_write", "stage_partial", "write_output",
                        "write_drain", "write_completion", "failed", "complete",
                        "group_program_weight", "group_program_input", "group_launch", "group_drain"};
                    static_assert(sizeof(phaseNames) / sizeof(*phaseNames) == static_cast<size_t>(Phase::Count), "phase names");
                    std::printf("[PROJECTION_CYCLE_BREAKDOWN] core=%u wrap_prefetch=%u", executionCore(), wrapPrefetch_);
                    for (size_t i = 0; i < phaseCycles_.size(); ++i)
                        std::printf(" %s=%llu", phaseNames[i], (unsigned long long)phaseCycles_[i]);
                    std::printf(" local_weight_read=%llu local_input_read=%llu local_fusion_read=%llu local_other_read=%llu\n",
                        (unsigned long long)localRoleCycles_[static_cast<size_t>(LocalReadRole::Weight)],
                        (unsigned long long)localRoleCycles_[static_cast<size_t>(LocalReadRole::Input)],
                        (unsigned long long)localRoleCycles_[static_cast<size_t>(LocalReadRole::Fusion)],
                        (unsigned long long)localRoleCycles_[static_cast<size_t>(LocalReadRole::Other)]);
                    std::printf("[PROJECTION_PHASE] core=%u input_dma=%llu weight_dma=%llu matrix_program=%llu input_scatter=%llu output_restore=%llu compute=%llu output_read=%llu write_drain=%llu\n",
                        executionCore(),
                        static_cast<unsigned long long>(phaseCycles_[static_cast<size_t>(Phase::LoadBlock)] + phaseCycles_[static_cast<size_t>(Phase::LoadInput)]),
                        static_cast<unsigned long long>(phaseCycles_[static_cast<size_t>(Phase::LoadWeight)] + phaseCycles_[static_cast<size_t>(Phase::LoadWeightDma)]),
                        static_cast<unsigned long long>(phaseCycles_[static_cast<size_t>(Phase::ProgramWeightReady)] + phaseCycles_[static_cast<size_t>(Phase::GroupProgramWeight)]),
                        static_cast<unsigned long long>(phaseCycles_[static_cast<size_t>(Phase::ProgramInput)] + phaseCycles_[static_cast<size_t>(Phase::WaitInputPrefetch)] + phaseCycles_[static_cast<size_t>(Phase::GroupProgramInput)]),
                        static_cast<unsigned long long>(phaseCycles_[static_cast<size_t>(Phase::RestoreOutputReady)]),
                        static_cast<unsigned long long>(arrayGroups_ == 2 ? computeBusyCycles_ : phaseCycles_[static_cast<size_t>(Phase::Launch)]),
                        static_cast<unsigned long long>(phaseCycles_[static_cast<size_t>(Phase::ReadOutput)]),
                        static_cast<unsigned long long>(phaseCycles_[static_cast<size_t>(Phase::WriteDrain)]));
                    std::printf("[PROJECTION_LOCAL_GM] core=%u read_bytes=%llu write_bytes=%llu read_cycles=%llu write_cycles=%llu timed=1 reuse_block=%u paired_weights=%u\n",
                        executionCore(),
                        static_cast<unsigned long long>(localReadBytes_),
                        static_cast<unsigned long long>(localWriteBytes_),
                        static_cast<unsigned long long>(phaseCycles_[static_cast<size_t>(Phase::LocalRead)]),
                        static_cast<unsigned long long>(phaseCycles_[static_cast<size_t>(Phase::LocalWrite)]),
                        reuseBlock_ ? 1u : 0u, pairedWeights_ ? 1u : 0u);
                    std::printf("[PROJECTION_PIPELINE] core=%u enabled=%u shared_input=%u input_prefetches=%llu partial_prefetches=%llu scatter_prefetches=%llu background_read_cycles=%llu background_write_cycles=%llu staging_bytes=%u array_count=%u read_retries=%llu write_retries=%llu\n",
                        executionCore(), pipeline_ ? 1u : 0u, sharedInput_ ? 1u : 0u,
                        static_cast<unsigned long long>(inputPrefetches_),
                        static_cast<unsigned long long>(partialPrefetches_),
                        static_cast<unsigned long long>(scatterPrefetches_),
                        static_cast<unsigned long long>(backgroundReadCycles_),
                        static_cast<unsigned long long>(backgroundWriteCycles_),
                        pipeline_ ? stagingBytes() : 0u, rowGroup(),
                        static_cast<unsigned long long>(backgroundReadRejects_),
                        static_cast<unsigned long long>(backgroundWriteRejects_));
                    std::printf("[PROJECTION_WEIGHT_PREFETCH] core=%u enabled=%u requests=%llu hits=%llu wait_cycles=%llu inflight=%u\n",
                        executionCore(), weightPrefetchEnabled_ ? 1u : 0u,
                        static_cast<unsigned long long>(weightPrefetchRequests_),
                        static_cast<unsigned long long>(weightPrefetchHits_),
                        static_cast<unsigned long long>(weightPrefetchWaitCycles_),
                        weightPrefetchBusy_ ? 1u : 0u);
                    std::printf("[PROJECTION_PREFETCH_CONFIG] core=%u weight_lookahead=%u input_single_slot=%u partial_single_slot=%u weight_single_slot=%u\n",
                        executionCore(), weightPrefetchDepth_, pipeline_ ? 1u : 0u,
                        pipeline_ && !localWeightPrefetch_ ? 1u : 0u, localWeightPrefetch_ ? 1u : 0u);
                    std::printf("[PROJECTION_FUSION] core=%u norm=%u rope=%u audit=%u norm_jobs=%llu rope_jobs=%llu norm_vector_cycles=%llu rope_vector_cycles=%llu local_read_bytes=%llu local_write_bytes=%llu norm_audit_bytes=%llu raw_audit_bytes=%llu panel_bytes=%llu\n",
                        executionCore(), fuseNorm_, fuseRope_, audit_,
                        (unsigned long long)normJobs_, (unsigned long long)ropeJobs_,
                        (unsigned long long)normVectorCycles_, (unsigned long long)ropeVectorCycles_,
                        (unsigned long long)fusionReadBytes_, (unsigned long long)fusionWriteBytes_,
                        (unsigned long long)normAuditBytes_, (unsigned long long)rawAuditBytes_,
                        (unsigned long long)panelBytes_);
                    std::printf("[PROJECTION_SCHEDULE] core=%u row_first=%u cbuffer_partial=%u cbuffer_bytes=%u cbuffer_read_bytes=%llu cbuffer_write_bytes=%llu\n",
                        executionCore(), rowFirst_ ? 1u : 0u,
                        cBufferPartial_ ? 1u : 0u, cBufferPartial_ ? 256u * 128u : 0u,
                        static_cast<unsigned long long>(cBufferReadBytes_),
                        static_cast<unsigned long long>(cBufferWriteBytes_));
                    std::printf("[PROJECTION_ACCUMULATION] core=%u retain_output=%u retained_pairs=%llu\n",
                        executionCore(), retainOutput_, (unsigned long long)retainedPairs_);
                    std::printf("[PROJECTION_DATAFLOW] core=%u input_pair_prefetch=%u local_weight_prefetch=%u array_groups=%u local_weight_prefetches=%llu local_weight_hits=%llu local_weight_wait_cycles=%llu array_launches=%llu compute_busy_cycles=%llu buffer_compute_overlap_cycles=%llu gm_compute_overlap_cycles=%llu\n",
                        executionCore(), inputPairPrefetch_, localWeightPrefetch_, arrayGroups_,
                        (unsigned long long)localWeightPrefetches_, (unsigned long long)localWeightHits_,
                        (unsigned long long)localWeightWaitCycles_, (unsigned long long)arrayLaunches_,
                        (unsigned long long)computeBusyCycles_, (unsigned long long)bufferComputeOverlapCycles_,
                        (unsigned long long)gmComputeOverlapCycles_);
                    std::printf("[PROJECTION_INPUT_FLOW] core=%u packed_input=%u input_prefetch_budget=%u max_request_bytes=%zu input_read_requests=%llu background_input_read_requests=%llu weight_read_requests=%llu norm_write_requests=%llu budget_stalls=%llu\n",
                        executionCore(), packedInput_, inputPrefetchBudget_, memory_->localMaxRequestBytes(),
                        (unsigned long long)inputReadRequests_, (unsigned long long)backgroundInputReadRequests_,
                        (unsigned long long)weightReadRequests_, (unsigned long long)normWriteRequests_,
                        (unsigned long long)inputBudgetStalls_);
                    std::printf("[PROJECTION_PARTIAL_MOVEMENT] core=%u restore_ahead=%u partial_reads=%llu final_reads=%llu partial_read_cycles=%llu final_read_cycles=%llu restores_foreground=%llu restores_ahead=%llu restore_cbuffer_wait=%llu restore_submit=%llu restore_transfer=%llu restore_ahead_wait=%llu restore_ahead_consume=%llu ahead_retries=%llu cbuffer_prefetch_reads=%llu cbuffer_demand_reads=%llu cbuffer_stores=%llu cbuffer_prefetch_residence=%llu cbuffer_demand_residence=%llu cbuffer_store_residence=%llu\n",
                        executionCore(), restoreAheadEnabled_,
                        (unsigned long long)partialReads_, (unsigned long long)finalReads_,
                        (unsigned long long)partialReadCycles_, (unsigned long long)finalReadCycles_,
                        (unsigned long long)restoresForeground_, (unsigned long long)restoresAhead_,
                        (unsigned long long)restoreCBufferWaitCycles_, (unsigned long long)restoreSubmitCycles_,
                        (unsigned long long)restoreTransferCycles_, (unsigned long long)restoreAheadWaitCycles_,
                        (unsigned long long)restoreAheadConsumeCycles_, (unsigned long long)restoreAheadRetries_,
                        (unsigned long long)cBufferPrefetchReads_, (unsigned long long)cBufferDemandReads_,
                        (unsigned long long)cBufferStores_, (unsigned long long)cBufferPrefetchResidence_,
                        (unsigned long long)cBufferDemandResidence_, (unsigned long long)cBufferStoreResidence_);
                });
            break;
        case Phase::Failed: phase_ = Phase::Complete; break;
        default: break;
        }
    }

private:
    enum class Phase;
    static const char* phaseName(Phase phase) {
        static const char* names[] = {"idle", "load_block", "load_input", "load_weight", "load_weight_dma", "program_weight",
            "program_weight_ready", "program_input", "launch", "wait_input_prefetch", "read_output",
            "restore_output", "restore_output_ready", "prepare_output", "norm_batch", "norm_wait", "rope_wait", "local_read", "local_write", "stage_partial", "write_output",
            "write_drain", "write_completion", "failed", "complete", "group_program_weight", "group_program_input", "group_launch", "group_drain"};
        const size_t index = static_cast<size_t>(phase);
        return index < sizeof(names) / sizeof(*names) ? names[index] : "unknown";
    }

    static const char* phaseResource(Phase phase) {
        switch (phase) {
        case Phase::LocalRead: return "local_gm_read";
        case Phase::LocalWrite: return "local_gm_write";
        case Phase::ProgramWeight:
        case Phase::ProgramWeightReady:
        case Phase::GroupProgramWeight: return "array_matrix";
        case Phase::ProgramInput:
        case Phase::GroupProgramInput: return "array_input";
        case Phase::Launch:
        case Phase::GroupLaunch: return "array_compute";
        case Phase::ReadOutput:
        case Phase::RestoreOutput:
        case Phase::RestoreOutputReady: return "array_output";
        case Phase::NormBatch:
        case Phase::NormWait:
        case Phase::RopeWait: return "sfu";
        default: return "control";
        }
    }

    void tracePhaseTransition(uint64_t cycle) {
        if (!criticalPathTrace_ || phase_ == tracePhase_) return;
        if (tracePhase_ != Phase::Idle) {
            const uint64_t event = ++traceSequence_;
            std::printf("[CRITICAL_EVENT] event_id=%llu predecessor=%llu kind=phase_done phase=%s resource=%s cycle=%llu core=%u job=%llu block=%u head=%u tile=%u generation=%llu\n",
                static_cast<unsigned long long>(event), static_cast<unsigned long long>(tracePredecessor_),
                phaseName(tracePhase_), phaseResource(tracePhase_), static_cast<unsigned long long>(cycle),
                executionCore(), static_cast<unsigned long long>(desc_.completion_addr), blockStart_, head_, inputTile_,
                static_cast<unsigned long long>(tag_));
            tracePredecessor_ = event;
        }
        const uint64_t event = ++traceSequence_;
        std::printf("[CRITICAL_EVENT] event_id=%llu predecessor=%llu kind=phase_start phase=%s resource=%s cycle=%llu core=%u job=%llu block=%u head=%u tile=%u generation=%llu\n",
            static_cast<unsigned long long>(event), static_cast<unsigned long long>(tracePredecessor_),
            phaseName(phase_), phaseResource(phase_), static_cast<unsigned long long>(cycle),
            executionCore(), static_cast<unsigned long long>(desc_.completion_addr), blockStart_, head_, inputTile_,
            static_cast<unsigned long long>(tag_));
        tracePredecessor_ = event;
        tracePhase_ = phase_;
    }

    struct CausalKey {
        uint32_t block, kind, head, dim, row, tile;
    };
    CausalKey causalKey() const {
        return {blockStart_, kind_, head_, dimTile_, row_, inputTile_};
    }
    void traceCausal(const char* event, const char* role, const CausalKey& key,
                     uint64_t tag = 0, uint64_t addr = 0, size_t bytes = 0) const {
        if (!causalTrace_) return;
        std::printf("[PROJECTION_CAUSAL] core=%u job=%llu event=%s role=%s tick=%llu block=%u kind=%u head=%u dim=%u row=%u tile=%u tag=%llu address=%llu bytes=%zu\n",
            executionCore(), (unsigned long long)desc_.completion_addr, event, role,
            (unsigned long long)memory_->getCurrentSimCycle(), key.block, key.kind,
            key.head, key.dim, key.row, key.tile, (unsigned long long)tag,
            (unsigned long long)addr, bytes);
    }
    bool causalLocalRead(uint64_t addr, size_t bytes, LocalMemoryClient client,
                         uint64_t tag, LocalReadCallback callback,
                         const char* role, CausalKey key) {
        const bool accepted = memory_->localReadAsync(addr, bytes, client, tag,
            [this, addr, bytes, tag, callback, role, key](bool ok, uint64_t id,
                                                       const std::vector<uint8_t>& raw) {
                traceCausal(ok ? "read_done" : "read_failed", role, key, tag, addr, bytes);
                callback(ok, id, raw);
            });
        if (accepted) traceCausal("read_submit", role, key, tag, addr, bytes);
        return accepted;
    }
    uint32_t executionCore() const { return 4 + (desc_.reserved - 1) * 4 + desc_.manager_slot; }
    uint32_t blockRowStart() const { return blockStart_ + (desc_.reserved - 1) * 64; }
    uint32_t blockRowEnd() const { return blockRowStart() + 64; }
    bool partialOutput() const {
        return reuseBlock_ && inputTile_ + 1 < desc_.hidden_dim / 64;
    }
    bool matchingAheadRestore() const {
        return restoreAheadIssued_ && restoreAheadRow_ == row_ &&
            restoreAheadTile_ == inputTile_;
    }
    void pumpAheadRestore(uint64_t cycle) {
        // The sole output context cannot be overwritten until its entire old
        // result has been read into the existing bounded output snapshot.
        // Consume only the existing next-row partial snapshot after its timed
        // C-buffer read completes. No new output context or payload slot.
        if (!restoreAheadEnabled_ || !outputDrained_ || restoreAheadIssued_ ||
            cBufferPrefetchRow_ == ~uint32_t{0} ||
            cycle < cBufferPrefetchReadyCycle_ || phase_ == Phase::Failed ||
            phase_ == Phase::Complete) return;
        std::vector<double> values(cBufferPrefetchRaw_.size() / 2);
        for (size_t i = 0; i < values.size(); ++i) values[i] = decode(cBufferPrefetchRaw_, i);
        if (!processor_->writeGemmOutputGroupClassAsync(arrays_, values, 2,
                AttentionClusterTrafficClass::ProjectionOutput, ++tag_, cycle,
                [this](bool ok, uint64_t) {
                    if (!ok) { fail(); return; }
                    restoreAheadReady_ = true;
                })) { ++restoreAheadRetries_; return; }
        restoreAheadIssued_ = true;
        restoreAheadReady_ = false;
        restoreAheadRow_ = cBufferPrefetchRow_;
        restoreAheadTile_ = cBufferPrefetchTile_;
        cBufferPrefetchRow_ = cBufferPrefetchTile_ = ~uint32_t{0};
        cBufferPrefetchRaw_.clear();
        ++restoresAhead_;
    }
    void pumpWeightPrefetch() {
        if (!weightPrefetchEnabled_ || weightPrefetchBusy_ || !blockLoaded_ ||
            kind_ >= 3 || phase_ == Phase::Failed || phase_ == Phase::WriteDrain ||
            phase_ == Phase::WriteCompletion) return;
        // Use existing cache slots for the same head only. A single future
        // DMA may look up to three tiles ahead (the other bank and next pair).
        // It never programs array operands or overwrites the current slot.
        const uint32_t end = std::min(desc_.hidden_dim / 64, inputTile_ + weightPrefetchDepth_);
        for (uint32_t next = inputTile_ + 1; next < end; ++next) {
            const uint64_t tile = weightTileForInputTile(next);
            const size_t slot = static_cast<size_t>(dimTile_) * (desc_.hidden_dim / 64) + next;
            if (cachedTiles_[slot] == tile) continue;
            weightPrefetchBusy_ = true;
            weightPrefetchTile_ = tile;
            ++weightPrefetchRequests_;
            readWeightToGM(desc_.weights_addr + tile * 8192,
                8192, desc_.scratch_addr + 0x60000 + slot * 8192,
                [this, tile, slot](bool ok) {
                    weightPrefetchBusy_ = false;
                    if (!ok) { fail(); return; }
                    cachedTiles_[slot] = tile;
                    prefetchedTiles_[slot] = tile;
                    ++weightLoads_;
                });
            return;
        }
    }
    void traceGroup(const char* event, uint32_t group, uint32_t tile, uint64_t cycle) {
        if (overlapTrace_) std::printf("[PROJECTION_GROUP_TRACE] core=%u block=%u head=%u group=%u tile=%u event=%s cycle=%llu\n",
            executionCore(), blockStart_, weightHead(), group, tile, event, (unsigned long long)cycle);
    }
    void pumpLocalWeightPrefetch() {
        if (!localWeightPrefetch_ || prefetchWeight_.active || prefetchWeight_.ready ||
            inputTile_ + 1 >= desc_.hidden_dim / 64) return;
        if (phase_ != Phase::ProgramWeightReady && phase_ != Phase::ProgramInput &&
            phase_ != Phase::Launch && phase_ != Phase::GroupProgramWeight &&
            phase_ != Phase::GroupProgramInput && phase_ != Phase::GroupLaunch) return;
        const size_t slot = inputTile_ + 1;
        const uint64_t tile = weightTileForInputTile(inputTile_ + 1);
        if (cachedTiles_[slot] != tile) return;
        // Retention leaves the original 8 KiB future-partial snapshot unused.
        // Reassign it to one future weight, within the same 32 KiB staging budget.
        localWeightTile_ = tile;
        armRead(prefetchWeight_, desc_.scratch_addr + 0x60000 + slot * 8192, 8192, 8192);
        prefetchWeight_.key.tile = inputTile_ + 1;
        ++localWeightPrefetches_;
    }
    uint32_t rowGroup() const { return static_cast<uint32_t>(arrays_.size()); }
    // One future input pair, one future partial, and one output snapshot.
    // Each lane requires 256 + 128 + 128 bytes; no extra array contexts.
    uint32_t stagingBytes() const { return rowGroup() * 512; }
    struct AsyncRead {
        uint64_t addr = 0;
        size_t laneBytes = 0, submitted = 0, completed = 0;
        uint64_t traceId = 0, submitTick = 0, readyTick = 0;
        CausalKey key{};
        bool active = false, ready = false;
        uint32_t inFlight = 0;
        std::vector<uint8_t> data;
    };
    struct AsyncStore {
        uint64_t addr = 0;
        size_t offset = 0;
        bool active = false, inFlight = false;
        std::vector<uint8_t> data;
    };
    static bool enabled(const char* name) {
        const char* value = std::getenv(name);
        return value == nullptr || std::strcmp(value, "0") != 0;
    }
    static bool enabledOptIn(const char* name) {
        const char* value = std::getenv(name);
        return value != nullptr && std::strcmp(value, "0") != 0;
    }
    uint64_t snapshotSequence_ = 0;
    LocalMemoryClient operandReadClient(bool operand = true) const {
        return operand && oracleEnabled("gm_read") ?
            LocalMemoryClient::ProjectionOracleOperand : LocalMemoryClient::RoCC;
    }
    void readWeightToGM(uint64_t address, size_t bytes, uint64_t destination,
                        std::function<void(bool)> done) {
        CausalKey key = causalKey();
        key.tile = ((address - desc_.weights_addr) / 8192) % (desc_.hidden_dim / 64);
        traceCausal("dma_submit", "weight", key, 0, address, bytes);
        auto originalDone = std::move(done);
        done = [this, address, bytes, key, originalDone](bool ok) {
            traceCausal(ok ? "dma_done" : "dma_failed", "weight", key, 0, address, bytes);
            originalDone(ok);
        };
        if (!oracleEnabled("weight_arrival")) {
            memory_->dma_read_from_host_to_globalmem(address, bytes, destination, std::move(done));
            return;
        }
        // Ideal supply of the exact immutable input tile, into the SAME
        // bounded GM cache slot. No extra resident weight bank is created.
        const char* path = std::getenv("GOLEM_PROJECTION_WEIGHTS_FILE");
        std::ifstream input(path ? path : "", std::ios::binary);
        std::vector<uint8_t> raw(bytes);
        if (address < desc_.weights_addr || !input) { done(false); return; }
        input.seekg(address - desc_.weights_addr);
        input.read(reinterpret_cast<char*>(raw.data()), bytes);
        if (static_cast<size_t>(input.gcount()) != bytes || destination < memory_->getBaseAddr() ||
            bytes > memory_->getSize() || destination - memory_->getBaseAddr() > memory_->getSize() - bytes) {
            done(false); return;
        }
        memory_->wr_to_globalmem(destination, bytes, raw);
        done(true);
    }
    void traceSnapshotConsume(const AsyncRead& slot, const char* resource) {
        traceCausal("snapshot_consume", resource, causalKey(), 0, slot.addr, slot.completed);
        traceResourceInterval("projection", resource, executionCore(), slot.traceId, 0,
            slot.traceId, slot.submitTick, slot.readyTick, slot.readyTick,
            memory_->getCurrentSimCycle(), slot.completed, 0, slot.addr);
    }
    void armRead(AsyncRead& transfer, uint64_t addr, size_t bytes, size_t laneBytes) {
        transfer = AsyncRead{};
        transfer.addr = addr;
        transfer.laneBytes = laneBytes;
        transfer.data.assign(bytes, 0);
        transfer.active = true;
        transfer.traceId = ++snapshotSequence_;
        transfer.submitTick = memory_->getCurrentSimCycle();
        transfer.key = causalKey();
    }
    void pumpRead(AsyncRead& transfer) {
        if (!transfer.active) return;
        const size_t limit = memory_->localMaxRequestBytes();
        if (limit == 0) { fail(); return; }
        while (transfer.submitted < transfer.data.size()) {
            if (&transfer == &prefetchInput_ && inputPrefetchBudget_ &&
                transfer.inFlight >= inputPrefetchBudget_) { ++inputBudgetStalls_; break; }
            const size_t offset = transfer.submitted;
            const size_t chunk = std::min(limit, transfer.laneBytes - offset % transfer.laneBytes);
            const uint64_t addr = transfer.addr + (transfer.laneBytes == 256 ?
                (offset / 256) * desc_.hidden_dim * 2 + offset % 256 : offset);
            AsyncRead* slot = &transfer;
            if (!causalLocalRead(addr, chunk,
                operandReadClient(slot != &prefetchPartial_), ++tag_,
                [this, slot, offset, chunk](bool ok, uint64_t, const std::vector<uint8_t>& raw) {
                    if (!ok || raw.size() != chunk) { fail(); return; }
                    std::copy(raw.begin(), raw.end(), slot->data.begin() + offset);
                    localReadBytes_ += chunk;
                    slot->completed += chunk;
                    --slot->inFlight;
                    if (slot->completed == slot->data.size()) {
                        slot->active = false;
                        slot->ready = true;
                        slot->readyTick = memory_->getCurrentSimCycle();
                    }
                }, &transfer == &prefetchInput_ ? "background_input" :
                   &transfer == &prefetchWeight_ ? "background_weight" : "background_partial",
                   transfer.key)) { ++backgroundReadRejects_; break; }
            transfer.submitted += chunk;
            ++transfer.inFlight;
            if (&transfer == &prefetchInput_) ++backgroundInputReadRequests_;
            if (&transfer == &prefetchWeight_) ++weightReadRequests_;
        }
    }
    void pumpBackground(uint64_t cycle) {
        if (prefetchInput_.active || prefetchPartial_.active || prefetchWeight_.active) ++backgroundReadCycles_;
        pumpRead(prefetchInput_);
        pumpRead(prefetchWeight_);
        // Different row addresses cannot alias the current row's store.
        if (!partialStore_.active || partialStore_.addr != prefetchPartial_.addr)
            pumpRead(prefetchPartial_);
        if (partialStore_.active) {
            ++backgroundWriteCycles_;
            if (!partialStore_.inFlight) {
                const size_t limit = memory_->localMaxRequestBytes();
                if (limit == 0) { fail(); return; }
                const size_t chunk = std::min(limit, partialStore_.data.size() - partialStore_.offset);
                std::vector<uint8_t> payload(partialStore_.data.begin() + partialStore_.offset,
                    partialStore_.data.begin() + partialStore_.offset + chunk);
                if (memory_->localWriteAsync(partialStore_.addr + partialStore_.offset, payload,
                    LocalMemoryClient::RoCC, ++tag_, [this, chunk](bool ok, uint64_t) {
                        partialStore_.inFlight = false;
                        if (!ok) { fail(); return; }
                        localWriteBytes_ += chunk;
                        partialStore_.offset += chunk;
                        if (partialStore_.offset == partialStore_.data.size())
                            partialStore_.active = false;
                    })) partialStore_.inFlight = true;
                else ++backgroundWriteRejects_;
            }
        }
        // Prefetch only into existing GM. The foreground ProgramInput phase
        // owns the sole array input context, including both tiles of a pair.
    }
    enum class Phase { Idle, LoadBlock, LoadInput, LoadWeight, LoadWeightDma, ProgramWeight,
        ProgramWeightReady, ProgramInput, Launch, WaitInputPrefetch, ReadOutput,
        RestoreOutput, RestoreOutputReady, PrepareOutput, NormBatch, NormWait, RopeWait, LocalRead, LocalWrite, StagePartial, WriteOutput,
        WriteDrain, WriteCompletion, Failed, Complete,
        GroupProgramWeight, GroupProgramInput, GroupLaunch, GroupDrain, Count };
    static double decode(const std::vector<uint8_t>& bytes, size_t index) {
        uint16_t bits;
        std::memcpy(&bits, bytes.data() + index * 2, 2);
        return golem_fp16_to_float(bits);
    }
    static void encode(std::vector<uint8_t>& bytes, size_t index, double value) {
        const uint16_t bits = golem_float_to_fp16(static_cast<float>(value));
        std::memcpy(bytes.data() + index * 2, &bits, 2);
    }
    uint64_t inputPairAddr(uint32_t row, uint32_t tile) const {
        const uint64_t localRow = row - blockStart_;
        if (!packedInput_) return inputBlockAddr() + localRow * desc_.hidden_dim * 2 + tile * 128;
        return inputBlockAddr() + (localRow / rowGroup()) * rowGroup() * desc_.hidden_dim * 2 +
            (tile / 2) * rowGroup() * 256 + (localRow % rowGroup()) * 256;
    }
    enum class LocalReadRole { Other, Weight, Input, Fusion, Count };
    // Bound every request to the SRAM API limit and retry queue backpressure.
    // One chunk is in flight; all producers share the existing GM ports.
    void beginLocalRead(uint64_t addr, size_t bytes,
                        std::function<void(const std::vector<uint8_t>&)> done,
                        LocalReadRole role = LocalReadRole::Other) {
        localPackedNormWrite_ = localNormWrite_ = false;
        localReadRole_ = role;
        localAddr_ = addr;
        localData_.assign(bytes, 0);
        localOffset_ = 0;
        localGather_ = false;
        localReadDone_ = std::move(done);
        localWriteDone_ = {};
        pending_ = true;
        phase_ = Phase::LocalRead;
    }
    void beginLocalInputPairRead(uint64_t addr,
                        std::function<void(const std::vector<uint8_t>&)> done) {
        // Gather exactly two adjacent 64-element tiles from each input row.
        // Each lane is a separate timed SRAM request, with shared-port
        // contention and queue backpressure. No full-hidden row reread.
        beginLocalRead(addr, rowGroup() * 256, std::move(done), LocalReadRole::Input);
        localGather_ = !packedInput_;
        gatherSubmitted_ = gatherCompleted_ = 0;
    }
    void beginLocalWrite(uint64_t addr, std::vector<uint8_t> data,
                         std::function<void()> done) {
        localPackedNormWrite_ = localNormWrite_ = false;
        localAddr_ = addr;
        localData_ = std::move(data);
        localOffset_ = 0;
        localWriteDone_ = std::move(done);
        localReadDone_ = {};
        pending_ = true;
        phase_ = Phase::LocalWrite;
    }
    void pumpLocalAccess() {
        if (phase_ == Phase::LocalRead && localGather_) {
            const size_t limit = memory_->localMaxRequestBytes();
            if (limit == 0) { pending_ = false; fail(); return; }
            while (gatherSubmitted_ < localData_.size()) {
                const size_t offset = gatherSubmitted_;
                const size_t chunk = std::min(limit, 256 - offset % 256);
                const uint64_t addr = localAddr_ +
                    (offset / 256) * desc_.hidden_dim * 2 + offset % 256;
                const bool accepted = causalLocalRead(addr, chunk,
                    operandReadClient(), ++tag_,
                    [this, offset, chunk](bool ok, uint64_t,
                                          const std::vector<uint8_t>& raw) {
                        if (phase_ != Phase::LocalRead) return;
                        if (!ok || raw.size() != chunk) { pending_ = false; fail(); return; }
                        std::copy(raw.begin(), raw.end(), localData_.begin() + offset);
                        localReadBytes_ += chunk;
                        gatherCompleted_ += chunk;
                        if (gatherCompleted_ == localData_.size()) {
                            pending_ = false;
                            localGather_ = false;
                            auto done = std::move(localReadDone_);
                            done(localData_);
                        }
                    }, "foreground_input", causalKey());
                if (!accepted) break;
                gatherSubmitted_ += chunk;
                ++inputReadRequests_;
            }
            return;
        }
        if ((phase_ != Phase::LocalRead && phase_ != Phase::LocalWrite) ||
            localInFlight_) return;
        const size_t limit = memory_->localMaxRequestBytes();
        if (limit == 0) { pending_ = false; fail(); return; }
        const uint32_t normRows = localPackedNormWrite_ ? localData_.size() / (desc_.hidden_dim * 2) : 0;
        const size_t segmentBytes = normRows * 256;
        const size_t chunk = localPackedNormWrite_ ?
            std::min(limit, segmentBytes - localOffset_ % segmentBytes) :
            std::min(limit, localData_.size() - localOffset_);
        const size_t offset = localOffset_;
        localInFlight_ = true;
        bool accepted;
        if (phase_ == Phase::LocalRead) {
            accepted = causalLocalRead(localAddr_ + offset, chunk,
                operandReadClient(localReadRole_ == LocalReadRole::Input || localReadRole_ == LocalReadRole::Weight), ++tag_,
                [this, offset, chunk](bool ok, uint64_t,
                                      const std::vector<uint8_t>& raw) {
                    localInFlight_ = false;
                    if (!ok || raw.size() != chunk) { pending_ = false; fail(); return; }
                    std::copy(raw.begin(), raw.end(), localData_.begin() + offset);
                    localReadBytes_ += chunk;
                    localOffset_ += chunk;
                    if (localOffset_ == localData_.size()) {
                        pending_ = false;
                        auto done = std::move(localReadDone_);
                        done(localData_);
                    }
                }, localReadRole_ == LocalReadRole::Weight ? "foreground_weight" :
                   localReadRole_ == LocalReadRole::Input ? "foreground_input" :
                   localReadRole_ == LocalReadRole::Fusion ? "foreground_fusion" : "foreground_other",
                   causalKey());
        } else {
            std::vector<uint8_t> data(chunk);
            uint64_t writeAddr = localAddr_ + offset;
            if (localPackedNormWrite_) {
                const size_t pair = offset / segmentBytes;
                const size_t inSegment = offset % segmentBytes;
                writeAddr = inputPairAddr(blockStart_ + normRow_, pair * 2) + inSegment;
                for (size_t i = 0; i < chunk; ++i) {
                    const size_t pos = inSegment + i;
                    data[i] = localData_[(pos / 256) * desc_.hidden_dim * 2 + pair * 256 + pos % 256];
                }
            } else std::copy(localData_.begin() + offset, localData_.begin() + offset + chunk, data.begin());
            accepted = memory_->localWriteAsync(writeAddr, data,
                LocalMemoryClient::RoCC, ++tag_,
                [this, chunk](bool ok, uint64_t) {
                    localInFlight_ = false;
                    if (!ok) { pending_ = false; fail(); return; }
                    localWriteBytes_ += chunk;
                    localOffset_ += chunk;
                    if (localOffset_ == localData_.size()) {
                        pending_ = false;
                        auto done = std::move(localWriteDone_);
                        done();
                    }
                });
        }
        if (!accepted) localInFlight_ = false;
        else if (phase_ == Phase::LocalRead) {
            if (localReadRole_ == LocalReadRole::Input) ++inputReadRequests_;
            if (localReadRole_ == LocalReadRole::Weight) ++weightReadRequests_;
        } else if (localNormWrite_) ++normWriteRequests_;
    }
    void beginFusionRead(uint64_t addr, size_t bytes,
            std::function<void(const std::vector<uint8_t>&)> done) {
        beginLocalRead(addr, bytes, [this, bytes, done](const std::vector<uint8_t>& raw) {
            localReadBytes_ -= bytes;
            fusionReadBytes_ += bytes;
            done(raw);
        }, LocalReadRole::Fusion);
    }
    void beginFusionWrite(uint64_t addr, std::vector<uint8_t> raw,
            std::function<void()> done) {
        const size_t bytes = raw.size();
        beginLocalWrite(addr, std::move(raw), [this, bytes, done]() {
            localWriteBytes_ -= bytes;
            fusionWriteBytes_ += bytes;
            done();
        });
    }
    void beginNormBlock() {
        normRow_ = (desc_.reserved - 1) * 64;
        if (gammaReady_) { phase_ = Phase::NormBatch; return; }
        pending_ = true;
        memory_->dma_read_from_host_to_globalmem(desc_.gamma_addr,
            desc_.hidden_dim * 2, desc_.scratch_addr + 0x50000, [this](bool ok) {
                pending_ = false;
                if (!ok) { fail(); return; }
                gammaReady_ = true;
                phase_ = Phase::NormBatch;
            });
    }
    void beginNormBatch() {
        const uint32_t rows = std::min<uint32_t>(8192 / desc_.hidden_dim, (desc_.reserved * 64) - normRow_);
        const size_t bytes = rows * desc_.hidden_dim * 2;
        normIssueTick_ = nowTick_;
        pending_ = true;
        memory_->dma_read_from_host_to_globalmem(
            desc_.input_addr + static_cast<uint64_t>(blockStart_ + normRow_) * desc_.hidden_dim * 2,
            bytes, desc_.scratch_addr + 0x40000, [this, bytes](bool ok) {
                pending_ = false;
                if (!ok) { fail(); return; }
                beginFusionRead(desc_.scratch_addr + 0x40000, bytes,
                    [this](const std::vector<uint8_t>& raw) {
                        normInput_.resize(raw.size() / 2);
                        for (size_t i = 0; i < normInput_.size(); ++i) normInput_[i] = decode(raw, i);
                        // Gamma is cached in GM, not in free persistent SFU registers.
                        // Every batch pays its existing shared GM read port.
                        beginFusionRead(desc_.scratch_addr + 0x50000, desc_.hidden_dim * 2,
                            [this](const std::vector<uint8_t>& coefficients) {
                                gamma_.resize(desc_.hidden_dim);
                                for (size_t i = 0; i < gamma_.size(); ++i) gamma_[i] = decode(coefficients, i);
                                SFUVectorRequest request{};
                                request.op = SFUVectorOp::RmsNorm;
                                request.input = &normInput_;
                                request.coefficients = &gamma_;
                                request.rowWidth = desc_.hidden_dim;
                                request.epsilon = 1.0e-6f;
                                vectorResult_ = {};
                                if (!vectorEngine_->issueVectorOp(request, &vectorResult_)) { fail(); return; }
                                normVectorCycles_ += vectorResult_.modeledCycles;
                                ++normJobs_;
                                phase_ = Phase::NormWait;
                            });
                    });
            });
    }
    void finishNormBatch() {
        std::vector<uint8_t> raw(vectorResult_.values.size() * 2);
        for (size_t i = 0; i < vectorResult_.values.size(); ++i) encode(raw, i, vectorResult_.values[i]);
        const uint64_t addr = inputBlockAddr() + static_cast<uint64_t>(normRow_) * desc_.hidden_dim * 2;
        const size_t bytes = raw.size();
        beginFusionWrite(addr, std::move(raw), [this, addr, bytes]() {
            if (!audit_) { advanceNormBatch(); return; }
            pending_ = true;
            normAuditBytes_ += bytes;
            memory_->dma_write_from_globalmem_to_host(addr,
                desc_.norm_output_addr + static_cast<uint64_t>(blockStart_ + normRow_) * desc_.hidden_dim * 2,
                bytes, [this](bool ok) {
                    pending_ = false;
                    if (!ok) { fail(); return; }
                    advanceNormBatch();
                });
        });
        localNormWrite_ = true;
        localPackedNormWrite_ = packedInput_;
    }
    void advanceNormBatch() {
        // Same evidence format as standalone SFU jobs; timestamps are SST ticks.
        std::printf("[SFU_RMSNORM] core=%u tag=%u rows=%u cols=%u issue_tick=%llu vector_ready_tick=%llu complete_tick=%llu elapsed_cycles=%llu vector_cycles=%llu status=0\n",
            executionCore(), blockStart_ + normRow_, static_cast<uint32_t>(vectorResult_.values.size() / desc_.hidden_dim), desc_.hidden_dim,
            (unsigned long long)normIssueTick_, (unsigned long long)vectorResult_.readyTick,
            (unsigned long long)nowTick_, (unsigned long long)((nowTick_ - normIssueTick_) / 1000),
            (unsigned long long)vectorResult_.modeledCycles);
        normRow_ += vectorResult_.values.size() / desc_.hidden_dim;
        vectorResult_.values.clear();
        normInput_.clear();
        gamma_.clear();
        if (normRow_ == desc_.reserved * 64) {
            ++inputLoads_;
            blockLoaded_ = true;
            phase_ = Phase::LoadInput;
        } else phase_ = Phase::NormBatch;
    }
    void beginRopeOutput() {
        const size_t bytes = rowGroup() * desc_.head_dim * 2;
        if (ropeCacheBlock_ != blockStart_) {
            const uint64_t globalRow = static_cast<uint64_t>(desc_.manager_slot) * desc_.rows_per_node + blockStart_;
            pending_ = true;
            memory_->dma_read_from_host_to_globalmem(desc_.rope_table_addr + globalRow * desc_.head_dim * 2,
                256 * desc_.head_dim * 2, desc_.scratch_addr + 0x20000, [this](bool ok) {
                    pending_ = false;
                    if (!ok) { fail(); return; }
                    ropeCacheBlock_ = blockStart_;
                    phase_ = Phase::PrepareOutput;
                });
            return;
        }
        beginFusionRead(desc_.scratch_addr + 0x20000 +
                static_cast<uint64_t>(row_ - blockStart_) * desc_.head_dim * 2, bytes,
                    [this](const std::vector<uint8_t>& raw) {
                        std::vector<double> coefficients(raw.size() / 2);
                        for (size_t i = 0; i < coefficients.size(); ++i) coefficients[i] = decode(raw, i);
                        // Quantize before rotation, preserving the former HBM transport boundary.
                        std::vector<double> input(outputValues_.size());
                        for (size_t i = 0; i < input.size(); ++i)
                            input[i] = golem_fp16_to_float(golem_float_to_fp16((float)outputValues_[i]));
                        std::vector<uint32_t> positions(rowGroup());
                        for (size_t i = 0; i < positions.size(); ++i) positions[i] = i;
                        SFUVectorRequest request{};
                        request.op = SFUVectorOp::Rope;
                        request.input = &input;
                        request.coefficients = &coefficients;
                        request.positions = &positions;
                        request.rowWidth = request.activeWidth = 64;
                        request.coefficientStride = desc_.head_dim;
                        request.coefficientOffset = dimTile_ * 64;
                        vectorResult_ = {};
                        if (!vectorEngine_->issueVectorOp(request, &vectorResult_)) { fail(); return; }
                        ropeVectorCycles_ += vectorResult_.modeledCycles;
                        ++ropeJobs_;
                        phase_ = Phase::RopeWait;
                    });
    }
    void prepareInput() {
        if (pairedWeights_) {
            if (inputTile_ % 2 != 0) { prepareInputReady(); return; }
            if (pipeline_ && prefetchRow_ == row_ && prefetchTile_ == inputTile_ &&
                (prefetchInput_.active || prefetchInput_.ready)) {
                if (!prefetchInput_.ready) return;
                inputRaw_ = std::move(prefetchInput_.data);
                traceSnapshotConsume(prefetchInput_, "input_snapshot_residence");
                prefetchInput_ = AsyncRead{};
                prepareInputReady();
                return;
            }
            beginLocalInputPairRead(inputPairAddr(row_, inputTile_),
                [this](const std::vector<uint8_t>& raw) {
                    inputRaw_ = raw;
                    prepareInputReady();
                });
            return;
        }
        const size_t bytes = static_cast<size_t>(rowGroup()) * desc_.hidden_dim * 2;
        if (reuseBlock_ || localInputRow_ != row_) {
            beginLocalRead(reuseBlock_ ? inputBlockAddr() +
                static_cast<uint64_t>(row_ - blockStart_) * desc_.hidden_dim * 2 :
                desc_.scratch_addr, bytes,
                [this](const std::vector<uint8_t>& raw) {
                    inputRaw_ = raw;
                    localInputRow_ = row_;
                    prepareInputReady();
                });
        } else prepareInputReady();
    }
    void prepareInputReady() {
        input_.resize(rowGroup() * 64);
        for (uint32_t lane = 0; lane < rowGroup(); ++lane)
            for (uint32_t col = 0; col < 64; ++col)
                input_[lane * 64 + col] = decode(inputRaw_,
                    pairedWeights_ ? lane * 128 + (inputTile_ % 2) * 64 + col :
                    lane * desc_.hidden_dim + inputTile_ * 64 + col);
        // Each input tile owns the sole operand context until its result is
        // consumed.  The first tile overwrites the shared output vector;
        // later tiles accumulate into it before the single readback.
        activeOperandBank_ = 0;
        if (!reuseBlock_ && inputTile_ == 1 && nextInputIssued_ && !nextInputReady_ &&
            !nextInputFailed_) {
            phase_ = Phase::WaitInputPrefetch;
            return;
        }
        const uint64_t tile = weightTile();
        if (residentTiles_[activeOperandBank_] == tile) {
            ++weightReuses_;
            phase_ = Phase::ProgramInput;
        } else {
            phase_ = Phase::LoadWeight;
        }
    }
    uint32_t headCount() const { return kind_ == 0 ? desc_.query_heads : desc_.kv_heads; }
    uint32_t weightHead() const {
        return (kind_ == 0 ? 0 : kind_ == 1 ? desc_.query_heads :
            desc_.query_heads + desc_.kv_heads) + head_;
    }
    uint64_t weightTile() const {
        return weightTileForInputTile(inputTile_);
    }
    uint64_t weightTileForInputTile(uint32_t inputTile) const {
        return ((static_cast<uint64_t>(weightHead()) * (desc_.head_dim / 64) +
                 dimTile_) * (desc_.hidden_dim / 64) + inputTile);
    }
    uint64_t cachedWeightAddr() const {
        // Keep the 512 KiB cache above the RMSNorm scratch at GM+0x40000.
        return desc_.scratch_addr + 0x60000 +
            cacheSlot() * 8192;
    }
    uint64_t partialAddr() const {
        // Cache ends at +0xE0000; partials, raw rows, and input occupy
        // disjoint bounded regions above it.
        return desc_.scratch_addr + 0xE0000 +
            static_cast<uint64_t>(row_ - blockStart_) * 128;
    }
    uint64_t rawBlockAddr() const {
        return desc_.scratch_addr + 0xE8000 +
            static_cast<uint64_t>(row_ - blockStart_) * desc_.head_dim * 2;
    }
    uint64_t inputBlockAddr() const {
        return desc_.scratch_addr + (pairedWeights_ ? 0xE8000 : 0xF8000);
    }
    size_t cacheSlot() const {
        const size_t inputTiles = desc_.hidden_dim / 64;
        return static_cast<size_t>(dimTile_) * inputTiles + inputTile_;
    }
    bool cacheTileEnabled() const {
        // Keep one complete hidden-dimension tile row resident. The 64-entry
        // cache covers the largest supported hidden dimension (2048) and both
        // D=64/D=128 dimension tiles without aliasing cache slots.
        return desc_.head_dim == 64 || desc_.head_dim == 128;
    }
    uint64_t rawBase() const {
        return kind_ == 0 ? desc_.q_addr : kind_ == 1 ? desc_.k_addr : desc_.v_addr;
    }
    void fail() { status_ = 1; pending_ = false; phase_ = Phase::Failed; }
    void issueWritebacks() {
        constexpr uint32_t kWritebackSlots = 4;
        while (!writes_.empty() && writeOutstanding_ < kWritebackSlots) {
            auto write = std::move(writes_.front());
            writes_.pop_front();
            ++writeOutstanding_;
            memory_->dma_write_to_host(write.first, write.second.size(),
                std::move(write.second), [this](bool ok) {
                    if (!ok) status_ = 1;
                    if (writeOutstanding_ > 0) --writeOutstanding_;
                });
        }
    }
    uint64_t partialOffset() const {
        return static_cast<uint64_t>(row_ - blockStart_) * 128;
    }
    void stagePartial(const std::vector<double>& values, uint64_t cycle) {
        std::vector<uint8_t> raw(rowGroup() * 64 * 2);
        for (size_t i = 0; i < values.size(); ++i) encode(raw, i, values[i]);
        if (cBufferPartial_) {
            uint64_t ready = 0;
            if (!processor_->writeProjectionPartial(partialOffset(), raw, cycle, ready)) {
                fail();
                return;
            }
            cBufferWriteBytes_ += raw.size();
            ++cBufferStores_;
            cBufferStoreResidence_ += ready - cycle;
            cBufferReadyCycle_ = std::max(cBufferReadyCycle_, ready);
            advance();
            return;
        }
        beginLocalWrite(partialAddr(), std::move(raw), [this]() { advance(); });
    }
    void finishOutput() {
        queueOutput(outputValues_);
        if (reuseBlock_ && desc_.head_dim == 128 && dimTile_ == 0)
            beginLocalWrite(rawBlockAddr(), rawTile_,
                [this]() { phase_ = Phase::WriteOutput; });
        else phase_ = Phase::WriteOutput;
    }
    void queueOutput(const std::vector<double>& values) {
        const uint64_t rowStride = static_cast<uint64_t>(desc_.head_dim) * 2;
        const uint64_t headStride = static_cast<uint64_t>(desc_.rows_per_node) * rowStride;
        const uint64_t tileBytes = 8192;
        const uint64_t panelBase = kind_ == 0 ? desc_.q_panel_addr :
            kind_ == 1 ? desc_.k_panel_addr : desc_.v_panel_addr;
        const uint32_t dimTiles = desc_.head_dim / 64;
        std::vector<uint8_t> grouped(rowGroup() * 128);
        for (uint32_t lane = 0; lane < rowGroup(); ++lane) {
            std::vector<uint8_t> packed(128);
            for (uint32_t dim = 0; dim < 64; ++dim) {
                const double value = values[lane * 64 + dim];
                encode(packed, dim, value);
                if (kind_ == 2) {
                    vTile_[(row_ % 64 + lane) * desc_.head_dim + dimTile_ * 64 + dim] =
                        golem_float_to_fp16(static_cast<float>(value));
                }
            }
            std::memcpy(rawTile_.data() + lane * rowStride + dimTile_ * 128,
                        packed.data(), 128);
            if (kind_ != 2) {
                for (uint32_t dim = 0; dim < 64; ++dim)
                    encode(grouped, lane * 64 + dim, panelValues_[lane * 64 + dim]);
            }
        }
        if (audit_ && dimTile_ + 1 == dimTiles) {
            writes_.emplace_back(rawBase() + head_ * headStride + row_ * rowStride, rawTile_);
            rawAuditBytes_ += rawTile_.size();
        }
        if (kind_ != 2) {
            const uint64_t panel = panelBase +
                ((static_cast<uint64_t>(head_) * (desc_.rows_per_node / 64) +
                  row_ / 64) * dimTiles + dimTile_) * tileBytes +
                (row_ % 64) * 128;
            panelBytes_ += grouped.size();
            writes_.emplace_back(panel, std::move(grouped));
        }
        if (kind_ == 2 && row_ % 64 + rowGroup() == 64 &&
            (reuseBlock_ || dimTile_ + 1 == dimTiles)) {
            for (uint32_t tile = reuseBlock_ ? dimTile_ : 0;
                 tile < (reuseBlock_ ? dimTile_ + 1 : dimTiles); ++tile) {
                std::vector<uint8_t> packed(tileBytes);
                for (uint32_t dim = 0; dim < 64; ++dim)
                    for (uint32_t key = 0; key < 64; ++key) {
                        const uint16_t bits = vTile_[key * desc_.head_dim + tile * 64 + dim];
                        std::memcpy(packed.data() + (dim * 64 + key) * 2, &bits, 2);
                    }
                const uint64_t panel = panelBase +
                    (((static_cast<uint64_t>(head_) * (desc_.rows_per_node / 256) +
                       row_ / 256) * dimTiles + tile) * 4 +
                     (row_ % 256) / 64) * tileBytes;
                panelBytes_ += packed.size();
                writes_.emplace_back(panel, std::move(packed));
            }
        }
    }
    void advance() {
        if (reuseBlock_) {
            if (rowFirst_) {
                inputTile_ = 0;
                if ((row_ += rowGroup()) == blockRowEnd()) {
                    blockStart_ += 256;
                    row_ = blockRowStart();
                    blockLoaded_ = false;
                    localInputRow_ = ~uint32_t{0};
                    if (blockStart_ == desc_.rows_per_node) {
                        blockStart_ = 0; row_ = blockRowStart();
                        if (++head_ == headCount()) {
                            head_ = 0;
                            if (++kind_ == 3) {
                                phase_ = (writeOutstanding_ == 0 && !partialStore_.active) ?
                                    Phase::WriteCompletion : Phase::WriteDrain;
                                return;
                            }
                        }
                    }
                }
                phase_ = blockLoaded_ ? Phase::LoadInput : Phase::LoadBlock;
                return;
            }
            // ReadOutput leaves a paired traversal on its odd input tile.
            // Keep the same two weights while moving to the next row group.
            if (pairedWeights_) --inputTile_;
            if ((row_ += rowGroup()) == blockRowEnd()) {
                row_ = blockRowStart();
                inputTile_ += pairedWeights_ ? 2 : 1;
                if (inputTile_ == desc_.hidden_dim / 64) {
                    inputTile_ = 0;
                    if (++dimTile_ == desc_.head_dim / 64) {
                        dimTile_ = 0;
                        if (sharedInput_) {
                            if (++head_ == headCount()) {
                                head_ = 0;
                                ++kind_;
                            }
                            if (kind_ != 3) {
                                phase_ = Phase::LoadInput;
                                return;
                            }
                            head_ = kind_ = 0;
                        }
                        blockStart_ += 256;
                        row_ = blockRowStart();
                        blockLoaded_ = false;
                        localInputRow_ = ~uint32_t{0};
                        if (blockStart_ == desc_.rows_per_node) {
                            if (sharedInput_) {
                                phase_ = (writeOutstanding_ == 0 && !partialStore_.active) ?
                                    Phase::WriteCompletion : Phase::WriteDrain;
                                return;
                            }
                            blockStart_ = 0; row_ = blockRowStart();
                            if (++head_ == headCount()) {
                                head_ = 0;
                                if (++kind_ == 3) {
                                    phase_ = (writeOutstanding_ == 0 && !partialStore_.active) ? Phase::WriteCompletion : Phase::WriteDrain;
                                    return;
                                }
                            }
                        }
                    }
                }
            }
            phase_ = blockLoaded_ ? Phase::LoadInput : Phase::LoadBlock;
            return;
        }
        inputTile_ = 0;
        if (++dimTile_ == desc_.head_dim / 64) {
            dimTile_ = 0;
            if ((row_ += rowGroup()) == blockRowEnd()) {
                blockStart_ += 256;
                row_ = blockRowStart();
                if (blockStart_ != desc_.rows_per_node) { phase_ = Phase::LoadInput; return; }
                blockStart_ = 0; row_ = blockRowStart();
                localInputRow_ = ~uint32_t{0};
                if (++head_ == headCount()) {
                    head_ = 0;
                    if (++kind_ == 3) {
                        phase_ = writeOutstanding_ == 0 ? Phase::WriteCompletion :
                            Phase::WriteDrain;
                        return;
                    }
                }
            }
        }
        phase_ = Phase::LoadInput;
    }

    SFUAPI* vectorEngine_;
    bool causalTrace_ = false;
    bool fuseNorm_ = false, fuseRope_ = false, audit_ = true, gammaReady_ = false;
    uint32_t normRow_ = 0, ropeCacheBlock_ = ~uint32_t{0};
    uint64_t nowTick_ = 0, normIssueTick_ = 0;
    uint64_t fusionReadBytes_ = 0, fusionWriteBytes_ = 0;
    uint64_t normVectorCycles_ = 0, ropeVectorCycles_ = 0;
    uint64_t normJobs_ = 0, ropeJobs_ = 0, normAuditBytes_ = 0, rawAuditBytes_ = 0, panelBytes_ = 0;
    SFUVectorResult vectorResult_;
    std::vector<double> gamma_, normInput_, panelValues_;
    GlobalMemoryAPI* memory_;
    WorkerCommandProcessorAPI* processor_;
    ProjectionJobDesc desc_{};
    Phase phase_ = Phase::Idle;
    std::vector<uint32_t> arrays_;
    std::vector<double> input_;
    std::vector<double> outputValues_;
    std::vector<uint8_t> inputRaw_, weightRaw_, partialRaw_;
    uint32_t localInputRow_ = ~uint32_t{0};
    std::vector<uint16_t> vTile_;
    std::vector<uint8_t> rawTile_;
    std::deque<std::pair<uint64_t, std::vector<uint8_t>>> writes_;
    uint32_t kind_ = 0, head_ = 0, row_ = 0, dimTile_ = 0, inputTile_ = 0;
    uint32_t remaining_ = 0;
    uint64_t tag_ = 0, status_ = 0;
    std::array<uint64_t, 1> residentTiles_{};
    uint32_t activeOperandBank_ = 0;
    uint32_t loadedInputRow_ = ~uint32_t{0};
    std::array<uint64_t, 64> cachedTiles_{};
    std::array<uint64_t, 64> prefetchedTiles_{};
    bool weightPrefetchEnabled_ = false, weightPrefetchBusy_ = false;
    uint32_t weightPrefetchDepth_ = 4;
    uint64_t weightPrefetchTile_ = ~uint64_t{0};
    uint64_t weightPrefetchRequests_ = 0, weightPrefetchHits_ = 0, weightPrefetchWaitCycles_ = 0;
    uint64_t weightLoads_ = 0, weightPrograms_ = 0, weightReuses_ = 0;
    uint64_t inputLoads_ = 0;
    uint64_t localReadBytes_ = 0, localWriteBytes_ = 0, localAddr_ = 0;
    size_t localOffset_ = 0;
    bool localInFlight_ = false;
    bool localGather_ = false;
    bool packedInput_ = false, localPackedNormWrite_ = false, localNormWrite_ = false;
    uint32_t inputPrefetchBudget_ = 0;
    uint64_t inputReadRequests_ = 0, backgroundInputReadRequests_ = 0;
    uint64_t weightReadRequests_ = 0, normWriteRequests_ = 0, inputBudgetStalls_ = 0;
    size_t gatherSubmitted_ = 0, gatherCompleted_ = 0;
    std::vector<uint8_t> localData_;
    std::function<void(const std::vector<uint8_t>&)> localReadDone_;
    std::function<void()> localWriteDone_;
    std::array<uint64_t, static_cast<size_t>(Phase::Count)> phaseCycles_{};
    std::array<uint64_t, static_cast<size_t>(LocalReadRole::Count)> localRoleCycles_{};
    LocalReadRole localReadRole_ = LocalReadRole::Other;
    bool wrapPrefetch_ = false;
    bool restoreAheadEnabled_ = false, outputDrained_ = false;
    bool restoreAheadIssued_ = false, restoreAheadReady_ = false;
    uint32_t restoreAheadRow_ = ~uint32_t{0}, restoreAheadTile_ = ~uint32_t{0};
    uint64_t partialReadCycles_ = 0, finalReadCycles_ = 0, partialReads_ = 0, finalReads_ = 0;
    uint64_t restoreCBufferWaitCycles_ = 0, restoreSubmitCycles_ = 0, restoreTransferCycles_ = 0;
    uint64_t restoreAheadWaitCycles_ = 0, restoreAheadConsumeCycles_ = 0;
    uint64_t restoresForeground_ = 0, restoresAhead_ = 0, restoreAheadRetries_ = 0;
    uint64_t cBufferPrefetchReads_ = 0, cBufferDemandReads_ = 0, cBufferStores_ = 0;
    uint64_t cBufferPrefetchResidence_ = 0, cBufferDemandResidence_ = 0, cBufferStoreResidence_ = 0;
    bool pipeline_ = false, sharedInput_ = false;
    bool computing_ = false;
    uint32_t computeBank_ = 0;
    AsyncRead prefetchInput_, prefetchPartial_, prefetchWeight_;
    bool inputPairPrefetch_ = false, localWeightPrefetch_ = false, overlapTrace_ = false;
    uint32_t arrayGroups_ = 1, programGroup_ = 0;
    std::array<std::vector<uint32_t>, 2> groupArrays_;
    std::array<uint32_t, 2> groupRemaining_{}, groupCompletedTile_{};
    bool groupBufferPending_ = false;
    uint64_t localWeightTile_ = ~uint64_t{0};
    uint64_t localWeightPrefetches_ = 0, localWeightHits_ = 0, localWeightWaitCycles_ = 0;
    uint64_t arrayLaunches_ = 0, computeBusyCycles_ = 0;
    uint64_t bufferComputeOverlapCycles_ = 0, gmComputeOverlapCycles_ = 0;
    AsyncStore partialStore_;
    uint32_t prefetchRow_ = ~uint32_t{0}, prefetchTile_ = ~uint32_t{0};
    bool scatterIssued_ = false, scatterReady_ = false;
    uint32_t scatterRow_ = 0, scatterTile_ = 0;
    uint64_t inputPrefetches_ = 0, partialPrefetches_ = 0, scatterPrefetches_ = 0;
    uint64_t backgroundReadCycles_ = 0, backgroundWriteCycles_ = 0;
    uint64_t backgroundReadRejects_ = 0, backgroundWriteRejects_ = 0;
    bool reuseBlock_ = false;
    bool pairedWeights_ = false;
    bool rowFirst_ = false;
    bool retainOutput_ = false;
    uint64_t retainedPairs_ = 0;
    bool cBufferPartial_ = false;
    uint64_t cBufferReadyCycle_ = 0, cBufferReadReadyCycle_ = 0;
    uint64_t cBufferPrefetchReadyCycle_ = 0;
    uint32_t cBufferPrefetchRow_ = ~uint32_t{0}, cBufferPrefetchTile_ = ~uint32_t{0};
    std::vector<uint8_t> cBufferPrefetchRaw_;
    uint64_t cBufferReadBytes_ = 0, cBufferWriteBytes_ = 0;
    bool blockLoaded_ = false;
    uint32_t blockStart_ = 0;
    uint32_t writeOutstanding_ = 0;
    uint64_t startCycle_ = 0;
    bool pending_ = false;
    bool nextInputIssued_ = false;
    bool nextInputReady_ = false;
    bool nextInputFailed_ = false;
    bool criticalPathTrace_ = false;
    Phase tracePhase_ = Phase::Idle;
    uint64_t traceSequence_ = 0, tracePredecessor_ = 0, traceCycle_ = 0;
};

}} // namespace SST::Golem
#endif
