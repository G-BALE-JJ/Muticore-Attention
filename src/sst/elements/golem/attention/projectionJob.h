#ifndef SST_GOLEM_PROJECTION_JOB_H
#define SST_GOLEM_PROJECTION_JOB_H

#include <sst/elements/golem/attention/projectionJobAbi.h>
#include <sst/elements/golem/fp16.h>
#include <sst/elements/golem/globalmemory/globalmemory.h>
#include <sst/elements/golem/workercmdproc/workercmdproc.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <deque>
#include <utility>
#include <vector>

namespace SST { namespace Golem {

// A bounded, descriptor-driven producer of the raw and panel Q/K/V layouts.
// One 16-token group, a bounded local-GM weight tile cache, and one V tile are resident.
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
        writeOutstanding_ = 0;
        writes_.clear();
        nextInputIssued_ = false;
        nextInputReady_ = false;
        nextInputFailed_ = false;
        loadedInputRow_ = ~uint32_t{0};
        startCycle_ = 0;
        phase_ = Phase::LoadInput;
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
        if (pending_) return;
        switch (phase_) {
        case Phase::LoadInput: {
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
                    prepareInput();
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
            std::vector<uint8_t> raw;
            memory_->rd_from_globalmem(cacheTileEnabled() ? cachedWeightAddr() :
                desc_.scratch_addr + 0x2000, 8192, raw);
            if (raw.size() != 8192) { fail(); break; }
            std::vector<double> matrix(4096);
            for (size_t i = 0; i < matrix.size(); ++i) matrix[i] = decode(raw, i);
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
                        phase_ = ok ? Phase::Launch : Phase::Failed;
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
            if (inputTile_ == 0 && desc_.hidden_dim == 128 &&
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
                            phase_ = ++inputTile_ < desc_.hidden_dim / 64 ?
                                Phase::LoadInput : Phase::ReadOutput;
                        }
                    })) { pending_ = false; break; }
            break;
        case Phase::ReadOutput:
            pending_ = true;
            if (!processor_->readGemmOutputGroupClassAsync(arrays_, 2,
                    AttentionClusterTrafficClass::ProjectionOutput, ++tag_, cycle,
                    [this](bool ok, uint64_t, const std::vector<double>& values) {
                        pending_ = false;
                        if (!ok || values.size() != 1024) { fail(); return; }
                        queueOutput(values);
                        phase_ = Phase::WriteOutput;
                    })) { pending_ = false; break; }
            break;
        case Phase::WriteOutput:
            issueWritebacks();
            // Output DMA owns copies of the payload vectors, so the next
            // input/array tile can proceed while bounded writebacks drain.
            advance();
            break;
        case Phase::WriteDrain:
            if (writeOutstanding_ == 0) phase_ = Phase::WriteCompletion;
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
                });
            break;
        case Phase::Failed: phase_ = Phase::Complete; break;
        default: break;
        }
    }

private:
    enum class Phase { Idle, LoadInput, LoadWeight, LoadWeightDma, ProgramWeight, ProgramInput,
        Launch, WaitInputPrefetch, ReadOutput, WriteOutput, WriteDrain, WriteCompletion, Failed, Complete };
    static double decode(const std::vector<uint8_t>& bytes, size_t index) {
        uint16_t bits;
        std::memcpy(&bits, bytes.data() + index * 2, 2);
        return golem_fp16_to_float(bits);
    }
    static void encode(std::vector<uint8_t>& bytes, size_t index, double value) {
        const uint16_t bits = golem_float_to_fp16(static_cast<float>(value));
        std::memcpy(bytes.data() + index * 2, &bits, 2);
    }
    void prepareInput() {
        const size_t bytes = static_cast<size_t>(16) * desc_.hidden_dim * 2;
        std::vector<uint8_t> raw;
        memory_->rd_from_globalmem(desc_.scratch_addr, bytes, raw);
        if (raw.size() != bytes) { fail(); return; }
        input_.resize(1024);
        for (uint32_t lane = 0; lane < 16; ++lane)
            for (uint32_t col = 0; col < 64; ++col)
                input_[lane * 64 + col] = decode(raw,
                    lane * desc_.hidden_dim + inputTile_ * 64 + col);
        // Each input tile owns one operand bank until its partial result is
        // consumed.  The first tile overwrites the shared output vector;
        // later tiles accumulate into it before the single readback.
        activeOperandBank_ = inputTile_ & 1u;
        if (inputTile_ == 1 && nextInputIssued_ && !nextInputReady_ &&
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
    void fail() { status_ = 1; phase_ = Phase::Failed; }
    void issueNextInputPrefetch(uint64_t cycle) {
        const size_t bytes = static_cast<size_t>(16) * desc_.hidden_dim * 2;
        std::vector<uint8_t> raw;
        memory_->rd_from_globalmem(desc_.scratch_addr, bytes, raw);
        if (raw.size() != bytes) { nextInputFailed_ = true; return; }
        std::vector<double> values(1024, 0.0);
        for (uint32_t lane = 0; lane < 16; ++lane)
            for (uint32_t col = 0; col < 64; ++col)
                values[lane * 64 + col] = decode(raw,
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
        if (kind_ == 2 && row_ % 64 == 48 && dimTile_ + 1 == dimTiles) {
            for (uint32_t tile = 0; tile < dimTiles; ++tile) {
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
        inputTile_ = 0;
        if (++dimTile_ == desc_.head_dim / 64) {
            dimTile_ = 0;
            if ((row_ += 16) == desc_.rows_per_node) {
                row_ = 0;
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
    uint32_t writeOutstanding_ = 0;
    uint64_t startCycle_ = 0;
    bool pending_ = false;
    bool nextInputIssued_ = false;
    bool nextInputReady_ = false;
    bool nextInputFailed_ = false;
};

}} // namespace SST::Golem
#endif
