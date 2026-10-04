#ifndef SST_GOLEM_PROJECTION_JOB_H
#define SST_GOLEM_PROJECTION_JOB_H

#include <sst/elements/golem/attention/projectionJobAbi.h>
#include <sst/elements/golem/fp16.h>
#include <sst/elements/golem/globalmemory/globalmemory.h>
#include <sst/elements/golem/workercmdproc/workercmdproc.h>

#include <array>
#include <algorithm>
#include <functional>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <utility>
#include <vector>

namespace SST { namespace Golem {

// A bounded, descriptor-driven producer of the raw and panel Q/K/V layouts.
// The row-major path holds one 16-token group; the reuse path holds one
// 256-token input block in bounded local GM. Both retain one V tile.
class ProjectionJob {
public:
    ProjectionJob(GlobalMemoryAPI* memory, WorkerCommandProcessorAPI* processor)
        : memory_(memory), processor_(processor) {
        for (uint32_t id = 0; id < 16; ++id) arrays_.push_back(id);
    }

    bool start(const ProjectionJobDesc& desc) {
        if (phase_ != Phase::Idle || processor_ == nullptr || memory_ == nullptr ||
            processor_->isBusy() || desc.magic != GOLEM_PROJECTION_JOB_MAGIC ||
            desc.size_bytes != sizeof(desc) || desc.hidden_dim == 0 ||
            desc.hidden_dim % 64 || desc.hidden_dim > 2048 ||
            desc.head_dim == 0 || desc.head_dim % 64 ||
            desc.query_heads == 0 || desc.kv_heads == 0 ||
            desc.rows_per_node == 0 || desc.rows_per_node % 256 ||
            desc.manager_slot >= 4 || desc.scratch_addr == 0 ||
            desc.input_addr == 0 || desc.weights_addr == 0 ||
            desc.q_addr == 0 || desc.k_addr == 0 || desc.v_addr == 0 ||
            desc.q_panel_addr == 0 || desc.k_panel_addr == 0 ||
            desc.v_panel_addr == 0 || desc.completion_addr == 0) return false;
        desc_ = desc;
        kind_ = head_ = row_ = dimTile_ = inputTile_ = 0;
        status_ = 0;
        residentTiles_.fill(~uint64_t{0});
        activeOperandBank_ = 0;
        cachedTiles_.fill(~uint64_t{0});
        weightLoads_ = weightPrograms_ = weightReuses_ = 0;
        inputLoads_ = 0;
        localReadBytes_ = localWriteBytes_ = 0;
        inputRaw_.clear();
        localInputRow_ = ~uint32_t{0};
        phaseCycles_.fill(0);
        // The reuse path stages 256 input rows and per-row FP16 partials in
        // local GM, leaving the physical two-bank array configuration intact.
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
        // Budget the additional 8 KiB of transfer snapshots against the same
        // local-GM capacity, including the reserved DMA tail. This is not an
        // additional array bank or an unbounded software cache.
        pipeline_ = pairedWeights_ && enabled("GOLEM_PROJECTION_PIPELINE") &&
            reuseBytes + 8192 <= gmSize - 64 - (desc.scratch_addr - gmBase);
        sharedInput_ = pairedWeights_ && enabled("GOLEM_PROJECTION_SHARED_INPUT");
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
        rawTile_.assign(16 * desc.head_dim * 2, 0);
        return true;
    }

    bool complete() const { return phase_ == Phase::Complete; }
    bool active() const { return phase_ != Phase::Idle && phase_ != Phase::Complete; }
    uint64_t status() const { return status_; }
    void retire() { phase_ = Phase::Idle; }

    void tick(uint64_t cycle) {
        if (!active()) return;
        if (startCycle_ == 0) startCycle_ = cycle;
        ++phaseCycles_[static_cast<size_t>(phase_)];
        if (pipeline_) pumpBackground(cycle);
        pumpLocalAccess();
        if (pending_) return;
        switch (phase_) {
        case Phase::LoadBlock: {
            const uint64_t addr = desc_.input_addr +
                static_cast<uint64_t>(blockStart_) * desc_.hidden_dim * 2;
            const size_t bytes = static_cast<size_t>(256) * desc_.hidden_dim * 2;
            pending_ = true;
            memory_->dma_read_from_host_to_globalmem(addr, bytes, inputBlockAddr(),
                [this](bool ok) {
                    pending_ = false;
                    if (!ok) { fail(); return; }
                    ++inputLoads_;
                    blockLoaded_ = true;
                    phase_ = Phase::LoadInput;
                });
            break;
        }
        case Phase::LoadInput: {
            if (reuseBlock_) { prepareInput(); break; }
            if (loadedInputRow_ == row_) {
                prepareInput();
                break;
            }
            const uint64_t addr = desc_.input_addr +
                static_cast<uint64_t>(row_) * desc_.hidden_dim * 2;
            const size_t bytes = static_cast<size_t>(16) * desc_.hidden_dim * 2;
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
                phase_ = Phase::ProgramWeight;
                break;
            }
            const uint64_t addr = desc_.weights_addr + tile * 8192;
            pending_ = true;
            memory_->dma_read_from_host_to_globalmem(addr, 8192, cachedWeightAddr(),
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
            memory_->dma_read_from_host_to_globalmem(addr, 8192, desc_.scratch_addr + 0x2000,
                [this](bool ok) {
                    pending_ = false;
                    if (ok) ++weightLoads_;
                    phase_ = ok ? Phase::ProgramWeight : Phase::Failed;
                    if (!ok) status_ = 1;
                });
            break;
        }
        case Phase::ProgramWeight: {
            beginLocalRead(cacheTileEnabled() ? cachedWeightAddr() :
                desc_.scratch_addr + 0x2000, 8192,
                [this](const std::vector<uint8_t>& raw) {
                    weightRaw_ = raw;
                    phase_ = Phase::ProgramWeightReady;
                });
            break;
        }
        case Phase::ProgramWeightReady: {
            std::vector<double> matrix(4096);
            for (size_t i = 0; i < matrix.size(); ++i) matrix[i] = decode(weightRaw_, i);
            pending_ = true;
            if (!processor_->programGemmMatrixGroupClassBankAsync(arrays_, activeOperandBank_, matrix, 2,
                    AttentionClusterTrafficClass::ProjectionWeights, ++tag_, cycle,
                    [this](bool ok, uint64_t) {
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
                phase_ = inputTile_ != 0 && inputTile_ % 2 == 0 ?
                    Phase::RestoreOutput : Phase::Launch;
                break;
            }
            if (inputTile_ == 1 && nextInputReady_) {
                nextInputReady_ = false;
                nextInputIssued_ = false;
                phase_ = Phase::Launch;
                break;
            }
            std::vector<double> values(input_.begin(), input_.end());
            pending_ = true;
            if (!processor_->programGemmInputScatterBankAsync(arrays_, activeOperandBank_, values, 2,
                    AttentionClusterTrafficClass::ProjectionInputScatter, ++tag_, cycle,
                    [this](bool ok, uint64_t) {
                        pending_ = false;
                        if (inputTile_ == 1) {
                            nextInputIssued_ = false;
                            nextInputReady_ = false;
                        }
                        phase_ = ok ? (reuseBlock_ && inputTile_ != 0 &&
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
        case Phase::Launch:
            if (!reuseBlock_ && inputTile_ == 0 && desc_.hidden_dim == 128 &&
                !nextInputIssued_) {
                issueNextInputPrefetch(cycle);
            }
            pending_ = true;
            remaining_ = 16;
            if (!processor_->launchGemmArrayGroupActiveBank(arrays_, activeOperandBank_,
                    inputTile_ == 0 ? 0 : 1, 64, cycle,
                    [this](uint32_t, uint64_t) {
                        if (--remaining_ == 0) {
                            pending_ = false;
                            computing_ = false;
                            if (pairedWeights_ && inputTile_ % 2 == 0) {
                                ++inputTile_;
                                phase_ = Phase::LoadInput;
                                return;
                            }
                            phase_ = reuseBlock_ ? Phase::ReadOutput :
                                (++inputTile_ < desc_.hidden_dim / 64 ?
                                    Phase::LoadInput : Phase::ReadOutput);
                        }
                    })) { pending_ = false; break; }
            if (pipeline_) {
                computing_ = true;
                computeBank_ = activeOperandBank_;
                if (inputTile_ % 2 == 0 && row_ + 16 < blockStart_ + 256) {
                    prefetchRow_ = row_ + 16;
                    prefetchTile_ = inputTile_;
                    armRead(prefetchInput_, inputBlockAddr() +
                        static_cast<uint64_t>(prefetchRow_ - blockStart_) * desc_.hidden_dim * 2 +
                        static_cast<uint64_t>(inputTile_) * 128, 4096, 256);
                    ++inputPrefetches_;
                    if (inputTile_ != 0) {
                        armRead(prefetchPartial_, desc_.scratch_addr + 0xE0000 +
                            static_cast<uint64_t>(prefetchRow_ - blockStart_) * 128,
                            2048, 2048);
                        ++partialPrefetches_;
                    }
                }
            }
            break;
        case Phase::ReadOutput:
            pending_ = true;
            if (!processor_->readGemmOutputGroupClassAsync(arrays_, 2,
                    AttentionClusterTrafficClass::ProjectionOutput, ++tag_, cycle,
                    [this](bool ok, uint64_t, const std::vector<double>& values) {
                        pending_ = false;
                        if (!ok || values.size() != 1024) { fail(); return; }
                        if (reuseBlock_ && inputTile_ + 1 < desc_.hidden_dim / 64) {
                            if (pipeline_) {
                                outputValues_ = values;
                                phase_ = Phase::StagePartial;
                            } else stagePartial(values);
                        } else {
                            outputValues_ = values;
                            phase_ = Phase::PrepareOutput;
                        }
                    })) { pending_ = false; break; }
            break;
        case Phase::StagePartial:
            if (!partialStore_.active) {
                partialStore_.addr = partialAddr();
                partialStore_.data.resize(2048);
                for (size_t i = 0; i < outputValues_.size(); ++i)
                    encode(partialStore_.data, i, outputValues_[i]);
                partialStore_.offset = 0;
                partialStore_.inFlight = false;
                partialStore_.active = true;
                advance();
            }
            break;
        case Phase::RestoreOutput: {
            if (pipeline_ && prefetchRow_ == row_ && prefetchTile_ == inputTile_ &&
                (prefetchPartial_.active || prefetchPartial_.ready)) {
                if (!prefetchPartial_.ready) break;
                partialRaw_ = std::move(prefetchPartial_.data);
                prefetchPartial_ = AsyncRead{};
                phase_ = Phase::RestoreOutputReady;
                break;
            }
            // A store owns its snapshot until timed SRAM completion. Do not
            // read a matching address, even under slow-write/backpressure profiles.
            if (pipeline_ && partialStore_.active && partialStore_.addr == partialAddr()) break;
            beginLocalRead(partialAddr(), 2048,
                [this](const std::vector<uint8_t>& raw) {
                    partialRaw_ = raw;
                    phase_ = Phase::RestoreOutputReady;
                });
            break;
        }
        case Phase::RestoreOutputReady: {
            std::vector<double> values(1024);
            for (size_t i = 0; i < values.size(); ++i) values[i] = decode(partialRaw_, i);
            pending_ = true;
            if (!processor_->writeGemmOutputGroupClassAsync(arrays_, values, 2,
                    AttentionClusterTrafficClass::ProjectionOutput, ++tag_, cycle,
                    [this](bool ok, uint64_t) {
                        pending_ = false;
                        phase_ = ok ? Phase::Launch : Phase::Failed;
                        if (!ok) status_ = 1;
                    })) { pending_ = false; break; }
            break;
        }
        case Phase::PrepareOutput:
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
            // Output DMA owns copies of the payload vectors, so the next
            // input/array tile can proceed while bounded writebacks drain.
            advance();
            break;
        case Phase::WriteDrain:
            if (writeOutstanding_ == 0 && !partialStore_.active) phase_ = Phase::WriteCompletion;
            break;
        case Phase::WriteCompletion:
            pending_ = true;
            memory_->dma_write_to_host(desc_.completion_addr, 8,
                std::vector<uint8_t>{1, 0, 0, 0, 0, 0, 0, 0},
                [this, cycle](bool ok) {
                    pending_ = false;
                    phase_ = Phase::Complete;
                    status_ = (ok && status_ == 0) ? 0 : 1;
                    std::printf("[PROJECTION_JOB] manager=%u start=%llu end=%llu cycles=%llu weight_loads=%llu weight_programs=%llu weight_reuses=%llu input_loads=%llu status=%llu\n",
                        desc_.manager_slot,
                        static_cast<unsigned long long>(startCycle_),
                        static_cast<unsigned long long>(cycle),
                        static_cast<unsigned long long>(cycle - startCycle_),
                        static_cast<unsigned long long>(weightLoads_),
                        static_cast<unsigned long long>(weightPrograms_),
                        static_cast<unsigned long long>(weightReuses_),
                        static_cast<unsigned long long>(inputLoads_),
                        static_cast<unsigned long long>(status_));
                    std::printf("[PROJECTION_PHASE] manager=%u input_dma=%llu weight_dma=%llu matrix_program=%llu input_scatter=%llu output_restore=%llu compute=%llu output_read=%llu write_drain=%llu\n",
                        desc_.manager_slot,
                        static_cast<unsigned long long>(phaseCycles_[static_cast<size_t>(Phase::LoadBlock)] + phaseCycles_[static_cast<size_t>(Phase::LoadInput)]),
                        static_cast<unsigned long long>(phaseCycles_[static_cast<size_t>(Phase::LoadWeight)] + phaseCycles_[static_cast<size_t>(Phase::LoadWeightDma)]),
                        static_cast<unsigned long long>(phaseCycles_[static_cast<size_t>(Phase::ProgramWeightReady)]),
                        static_cast<unsigned long long>(phaseCycles_[static_cast<size_t>(Phase::ProgramInput)] + phaseCycles_[static_cast<size_t>(Phase::WaitInputPrefetch)]),
                        static_cast<unsigned long long>(phaseCycles_[static_cast<size_t>(Phase::RestoreOutputReady)]),
                        static_cast<unsigned long long>(phaseCycles_[static_cast<size_t>(Phase::Launch)]),
                        static_cast<unsigned long long>(phaseCycles_[static_cast<size_t>(Phase::ReadOutput)]),
                        static_cast<unsigned long long>(phaseCycles_[static_cast<size_t>(Phase::WriteDrain)]));
                    std::printf("[PROJECTION_LOCAL_GM] manager=%u read_bytes=%llu write_bytes=%llu read_cycles=%llu write_cycles=%llu timed=1 reuse_block=%u paired_weights=%u\n",
                        desc_.manager_slot,
                        static_cast<unsigned long long>(localReadBytes_),
                        static_cast<unsigned long long>(localWriteBytes_),
                        static_cast<unsigned long long>(phaseCycles_[static_cast<size_t>(Phase::LocalRead)]),
                        static_cast<unsigned long long>(phaseCycles_[static_cast<size_t>(Phase::LocalWrite)]),
                        reuseBlock_ ? 1u : 0u, pairedWeights_ ? 1u : 0u);
                    std::printf("[PROJECTION_PIPELINE] manager=%u enabled=%u shared_input=%u input_prefetches=%llu partial_prefetches=%llu scatter_prefetches=%llu background_read_cycles=%llu background_write_cycles=%llu staging_bytes=%u read_retries=%llu write_retries=%llu\n",
                        desc_.manager_slot, pipeline_ ? 1u : 0u, sharedInput_ ? 1u : 0u,
                        static_cast<unsigned long long>(inputPrefetches_),
                        static_cast<unsigned long long>(partialPrefetches_),
                        static_cast<unsigned long long>(scatterPrefetches_),
                        static_cast<unsigned long long>(backgroundReadCycles_),
                        static_cast<unsigned long long>(backgroundWriteCycles_),
                        pipeline_ ? 8192u : 0u,
                        static_cast<unsigned long long>(backgroundReadRejects_),
                        static_cast<unsigned long long>(backgroundWriteRejects_));
                });
            break;
        case Phase::Failed: phase_ = Phase::Complete; break;
        default: break;
        }
    }

private:
    // One future input pair, one future partial, and one output snapshot.
    // These bounded transfer buffers add 8 KiB of staging, not array contexts.
    struct AsyncRead {
        uint64_t addr = 0;
        size_t laneBytes = 0, submitted = 0, completed = 0;
        bool active = false, ready = false;
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
    void armRead(AsyncRead& transfer, uint64_t addr, size_t bytes, size_t laneBytes) {
        transfer = AsyncRead{};
        transfer.addr = addr;
        transfer.laneBytes = laneBytes;
        transfer.data.assign(bytes, 0);
        transfer.active = true;
    }
    void pumpRead(AsyncRead& transfer) {
        if (!transfer.active) return;
        const size_t limit = memory_->localMaxRequestBytes();
        if (limit == 0) { fail(); return; }
        while (transfer.submitted < transfer.data.size()) {
            const size_t offset = transfer.submitted;
            const size_t chunk = std::min(limit, transfer.laneBytes - offset % transfer.laneBytes);
            const uint64_t addr = transfer.addr + (transfer.laneBytes == 256 ?
                (offset / 256) * desc_.hidden_dim * 2 + offset % 256 : offset);
            AsyncRead* slot = &transfer;
            if (!memory_->localReadAsync(addr, chunk, LocalMemoryClient::RoCC, ++tag_,
                [this, slot, offset, chunk](bool ok, uint64_t, const std::vector<uint8_t>& raw) {
                    if (!ok || raw.size() != chunk) { fail(); return; }
                    std::copy(raw.begin(), raw.end(), slot->data.begin() + offset);
                    localReadBytes_ += chunk;
                    slot->completed += chunk;
                    if (slot->completed == slot->data.size()) {
                        slot->active = false;
                        slot->ready = true;
                    }
                })) { ++backgroundReadRejects_; break; }
            transfer.submitted += chunk;
        }
    }
    void prefetchScatter(uint32_t row, uint32_t tile,
                         const std::vector<uint8_t>& raw, uint64_t cycle) {
        if (scatterIssued_) return;
        std::vector<double> values(1024);
        for (uint32_t lane = 0; lane < 16; ++lane)
            for (uint32_t col = 0; col < 64; ++col)
                values[lane * 64 + col] = decode(raw, lane * 128 + (tile % 2) * 64 + col);
        if (processor_->programGemmInputScatterBankAsync(arrays_, tile & 1u, values, 2,
            AttentionClusterTrafficClass::ProjectionInputScatter, ++tag_, cycle,
            [this](bool ok, uint64_t) {
                if (!ok) { fail(); return; }
                scatterReady_ = true;
            })) {
            scatterIssued_ = true;
            scatterReady_ = false;
            scatterRow_ = row;
            scatterTile_ = tile;
            ++scatterPrefetches_;
        }
    }
    void pumpBackground(uint64_t cycle) {
        if (prefetchInput_.active || prefetchPartial_.active) ++backgroundReadCycles_;
        pumpRead(prefetchInput_);
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
        // Only the inactive input bank is written. Output restore remains on
        // the serial path, after the preceding output read has completed.
        if (computing_ && !scatterIssued_) {
            if (computeBank_ == 0 &&
                residentTiles_[1] == weightTileForInputTile(inputTile_ + 1))
                prefetchScatter(row_, inputTile_ + 1, inputRaw_, cycle);
            else if (computeBank_ == 1 && prefetchInput_.ready)
                prefetchScatter(prefetchRow_, prefetchTile_, prefetchInput_.data, cycle);
        }
    }
    enum class Phase { Idle, LoadBlock, LoadInput, LoadWeight, LoadWeightDma, ProgramWeight,
        ProgramWeightReady, ProgramInput, Launch, WaitInputPrefetch, ReadOutput,
        RestoreOutput, RestoreOutputReady, PrepareOutput, LocalRead, LocalWrite, StagePartial, WriteOutput,
        WriteDrain, WriteCompletion, Failed, Complete, Count };
    static double decode(const std::vector<uint8_t>& bytes, size_t index) {
        uint16_t bits;
        std::memcpy(&bits, bytes.data() + index * 2, 2);
        return golem_fp16_to_float(bits);
    }
    static void encode(std::vector<uint8_t>& bytes, size_t index, double value) {
        const uint16_t bits = golem_float_to_fp16(static_cast<float>(value));
        std::memcpy(bytes.data() + index * 2, &bits, 2);
    }
    // Bound every request to the SRAM API limit and retry queue backpressure.
    // One chunk is in flight; all producers share the existing GM ports.
    void beginLocalRead(uint64_t addr, size_t bytes,
                        std::function<void(const std::vector<uint8_t>&)> done) {
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
        beginLocalRead(addr, 16 * 256, std::move(done));
        localGather_ = true;
        gatherSubmitted_ = gatherCompleted_ = 0;
    }
    void beginLocalWrite(uint64_t addr, std::vector<uint8_t> data,
                         std::function<void()> done) {
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
                const bool accepted = memory_->localReadAsync(addr, chunk,
                    LocalMemoryClient::RoCC, ++tag_,
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
                    });
                if (!accepted) break;
                gatherSubmitted_ += chunk;
            }
            return;
        }
        if ((phase_ != Phase::LocalRead && phase_ != Phase::LocalWrite) ||
            localInFlight_) return;
        const size_t limit = memory_->localMaxRequestBytes();
        if (limit == 0) { pending_ = false; fail(); return; }
        const size_t chunk = std::min(limit, localData_.size() - localOffset_);
        const size_t offset = localOffset_;
        localInFlight_ = true;
        bool accepted;
        if (phase_ == Phase::LocalRead) {
            accepted = memory_->localReadAsync(localAddr_ + offset, chunk,
                LocalMemoryClient::RoCC, ++tag_,
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
                });
        } else {
            std::vector<uint8_t> data(localData_.begin() + offset,
                                      localData_.begin() + offset + chunk);
            accepted = memory_->localWriteAsync(localAddr_ + offset, data,
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
    }
    void prepareInput() {
        if (pairedWeights_) {
            if (inputTile_ % 2 != 0) { prepareInputReady(); return; }
            if (pipeline_ && prefetchRow_ == row_ && prefetchTile_ == inputTile_ &&
                (prefetchInput_.active || prefetchInput_.ready)) {
                if (!prefetchInput_.ready) return;
                inputRaw_ = std::move(prefetchInput_.data);
                prefetchInput_ = AsyncRead{};
                prepareInputReady();
                return;
            }
            beginLocalInputPairRead(inputBlockAddr() +
                static_cast<uint64_t>(row_ - blockStart_) * desc_.hidden_dim * 2 +
                static_cast<uint64_t>(inputTile_) * 128,
                [this](const std::vector<uint8_t>& raw) {
                    inputRaw_ = raw;
                    prepareInputReady();
                });
            return;
        }
        const size_t bytes = static_cast<size_t>(16) * desc_.hidden_dim * 2;
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
        input_.resize(1024);
        for (uint32_t lane = 0; lane < 16; ++lane)
            for (uint32_t col = 0; col < 64; ++col)
                input_[lane * 64 + col] = decode(inputRaw_,
                    pairedWeights_ ? lane * 128 + (inputTile_ % 2) * 64 + col :
                    lane * desc_.hidden_dim + inputTile_ * 64 + col);
        // Each input tile owns one operand bank until its partial result is
        // consumed.  The first tile overwrites the shared output vector;
        // later tiles accumulate into it before the single readback.
        activeOperandBank_ = inputTile_ & 1u;
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
    void issueNextInputPrefetch(uint64_t cycle) {
        // The current row group was already fetched through the timed GM port.
        std::vector<double> values(1024, 0.0);
        for (uint32_t lane = 0; lane < 16; ++lane)
            for (uint32_t col = 0; col < 64; ++col)
                values[lane * 64 + col] = decode(inputRaw_,
                    lane * desc_.hidden_dim + 64 + col);
        nextInputIssued_ = true;
        nextInputReady_ = false;
        nextInputFailed_ = false;
        if (!processor_->programGemmInputScatterBankAsync(
                arrays_, 1, values, 2,
                AttentionClusterTrafficClass::ProjectionInputScatter,
                ++tag_, cycle,
                [this](bool ok, uint64_t) {
                    nextInputReady_ = ok;
                    nextInputFailed_ = !ok;
                })) nextInputFailed_ = true;
    }
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
    void stagePartial(const std::vector<double>& values) {
        std::vector<uint8_t> raw(2048);
        for (size_t i = 0; i < values.size(); ++i) encode(raw, i, values[i]);
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
        std::vector<uint8_t> grouped(16 * 128);
        for (uint32_t lane = 0; lane < 16; ++lane) {
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
            if (kind_ != 2)
                std::memcpy(grouped.data() + lane * 128, packed.data(), 128);
        }
        if (dimTile_ + 1 == dimTiles)
            writes_.emplace_back(rawBase() + head_ * headStride + row_ * rowStride,
                                 rawTile_);
        if (kind_ != 2) {
            const uint64_t panel = panelBase +
                ((static_cast<uint64_t>(head_) * (desc_.rows_per_node / 64) +
                  row_ / 64) * dimTiles + dimTile_) * tileBytes +
                (row_ % 64) * 128;
            writes_.emplace_back(panel, std::move(grouped));
        }
        if (kind_ == 2 && row_ % 64 == 48 &&
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
                writes_.emplace_back(panel, std::move(packed));
            }
        }
    }
    void advance() {
        if (reuseBlock_) {
            // ReadOutput leaves a paired traversal on its odd input tile.
            // Keep the same two weights while moving to the next row group.
            if (pairedWeights_) --inputTile_;
            if ((row_ += 16) == blockStart_ + 256) {
                row_ = blockStart_;
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
                        row_ = blockStart_;
                        blockLoaded_ = false;
                        localInputRow_ = ~uint32_t{0};
                        if (blockStart_ == desc_.rows_per_node) {
                            if (sharedInput_) {
                                phase_ = (writeOutstanding_ == 0 && !partialStore_.active) ?
                                    Phase::WriteCompletion : Phase::WriteDrain;
                                return;
                            }
                            blockStart_ = row_ = 0;
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
            if ((row_ += 16) == desc_.rows_per_node) {
                row_ = 0;
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
    std::array<uint64_t, 2> residentTiles_{};
    uint32_t activeOperandBank_ = 0;
    uint32_t loadedInputRow_ = ~uint32_t{0};
    std::array<uint64_t, 64> cachedTiles_{};
    uint64_t weightLoads_ = 0, weightPrograms_ = 0, weightReuses_ = 0;
    uint64_t inputLoads_ = 0;
    uint64_t localReadBytes_ = 0, localWriteBytes_ = 0, localAddr_ = 0;
    size_t localOffset_ = 0;
    bool localInFlight_ = false;
    bool localGather_ = false;
    size_t gatherSubmitted_ = 0, gatherCompleted_ = 0;
    std::vector<uint8_t> localData_;
    std::function<void(const std::vector<uint8_t>&)> localReadDone_;
    std::function<void()> localWriteDone_;
    std::array<uint64_t, static_cast<size_t>(Phase::Count)> phaseCycles_{};
    bool pipeline_ = false, sharedInput_ = false;
    bool computing_ = false;
    uint32_t computeBank_ = 0;
    AsyncRead prefetchInput_, prefetchPartial_;
    AsyncStore partialStore_;
    uint32_t prefetchRow_ = ~uint32_t{0}, prefetchTile_ = ~uint32_t{0};
    bool scatterIssued_ = false, scatterReady_ = false;
    uint32_t scatterRow_ = 0, scatterTile_ = 0;
    uint64_t inputPrefetches_ = 0, partialPrefetches_ = 0, scatterPrefetches_ = 0;
    uint64_t backgroundReadCycles_ = 0, backgroundWriteCycles_ = 0;
    uint64_t backgroundReadRejects_ = 0, backgroundWriteRejects_ = 0;
    bool reuseBlock_ = false;
    bool pairedWeights_ = false;
    bool blockLoaded_ = false;
    uint32_t blockStart_ = 0;
    uint32_t writeOutstanding_ = 0;
    uint64_t startCycle_ = 0;
    bool pending_ = false;
    bool nextInputIssued_ = false;
    bool nextInputReady_ = false;
    bool nextInputFailed_ = false;
};

}} // namespace SST::Golem
#endif
