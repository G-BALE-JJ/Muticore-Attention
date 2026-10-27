#ifndef SST_GOLEM_PV_WO_HANDOFF_H
#define SST_GOLEM_PV_WO_HANDOFF_H
#include <sst/elements/golem/attention/decoderMemoryLayout.h>
#include <sst/elements/golem/attention/pvWoInputInterface.h>
#include <sst/elements/golem/globalmemory/globalmemory.h>
#include <algorithm>
#include <cstdio>
#include <deque>
#include <functional>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace SST { namespace Golem {
// S=1024/H=2048/Dh=64 software transfer through the existing control NoC.
// A PV producer owns one unacknowledged 8 KiB output. Each landing queue is
// bounded by eight PV producers; a head is published only after timed GM ACKs.
class PvWoHandoff : public PvWoInputInterface {
    using Message = ControlTransportMessage;
    using Kind = ControlTransportMessageKind;
    using Key = std::tuple<uint64_t, uint64_t, uint32_t>;
    struct Pending {
        uint32_t target = 0;
        std::function<void(bool)> done;
    };
    struct Landing {
        Message message;
        size_t offset = 0;
    };
public:
    PvWoHandoff(GlobalMemoryAPI* gm, uint32_t core) : gm_(gm), core_(core) {
        if (!gm_ || core_ < 4 || core_ >= 20 ||
            gm_->getSize() < DecoderMemoryLayout::decoder_pv_wo_landing_gm + workerTileBytes() ||
            !gm_->localMaxRequestBytes())
            throw std::invalid_argument("PV-Wo handoff worker or GM capacity");
    }
    static bool accepts(Kind kind) {
        return kind == Kind::AttentionPvWoData || kind == Kind::AttentionPvWoAck;
    }
    static constexpr uint32_t payloadBytes() { return 8192; }
    static constexpr uint32_t headBlockBytes() { return 64 * 64 * 2; }
    static constexpr uint32_t workerTileBytes() { return 64 * 2048 * 2; }

    bool dispatch(const Message& source, const std::vector<uint8_t>& bytes,
                  std::function<void(bool)> done) {
        if (core_ < 12 || !pending_.empty() || !done ||
            bytes.size() != headBlockBytes() || source.headDim != 64 ||
            source.queryLength != 256 || source.kvLength != 1024 ||
            source.numQueryHeads != 32 || source.numKvHeads != 8 ||
            source.groupQueryRowBegin % 64 || source.row / 1024 > 3 ||
            source.kvHeadIndex >= 8 ||
            (source.jobId >> 32) != source.kvHeadIndex) return false;
        const uint32_t generation = static_cast<uint32_t>(source.jobId);
        if (generation < 0xD1000401u || (generation - 0xD1000401u) % 65536 ||
            (generation - 0xD1000401u) / 65536 >= 32) return false;
        const uint32_t manager = source.row / 1024;
        const uint32_t localBegin = source.row - manager * 1024 + source.groupQueryRowBegin;
        if (localBegin + 64 > 1024 || localBegin / 256 != (localBegin + 63) / 256) return false;
        const uint32_t query = localBegin % 256;
        const uint32_t head = source.kvHeadIndex * 4 + localBegin / 256;
        const uint32_t target = 4 + manager + 4 * (query / 64);
        const Key key{source.jobId, source.tag, source.groupQueryRowBegin};
        pending_.emplace(key, Pending{target, std::move(done)});
        Message data{};
        data.kind = Kind::AttentionPvWoData;
        data.jobId = source.jobId;
        data.tag = source.tag;
        data.ownerCore = core_;
        data.workerCore = target;
        data.workerSlot = head;
        data.outputAddr = manager;
        data.inputAddr = (generation - 0xD1000401u) / 65536 + 1;
        data.groupQueryRowBegin = source.groupQueryRowBegin;
        data.headDim = 64;
        data.queryLength = 256;
        data.tensorPayload = bytes;
        enqueue(target, std::move(data));
        return true;
    }

    void receive(const Message& message) {
        if (message.kind == Kind::AttentionPvWoAck) {
            const Key key{message.jobId, message.tag, message.groupQueryRowBegin};
            auto it = pending_.find(key);
            if (it == pending_.end() || message.ownerCore != core_ ||
                message.workerCore != it->second.target || message.value != 1.0)
                throw std::logic_error("invalid PV-Wo landing ACK");
            auto done = std::move(it->second.done);
            pending_.erase(it);
            done(true);
            return;
        }
        if (leased_ || message.kind != Kind::AttentionPvWoData || message.workerCore != core_ ||
            message.ownerCore < 12 || message.ownerCore >= 20 || message.workerSlot >= 32 ||
            message.tensorPayload.size() != payloadBytes() ||
            message.queryLength != 256 || message.headDim != 64 ||
            message.outputAddr != (core_ - 4) % 4 ||
            message.inputAddr != consumedEpoch_ + 1 || incoming_.size() >= 8)
            throw std::logic_error("invalid PV-Wo data payload, generation or queue bound");
        const uint32_t bit = uint32_t{1} << message.workerSlot;
        if (received_ & bit) throw std::logic_error("duplicate PV-Wo head landing");
        received_ |= bit;
        incoming_.push_back({message, 0});
    }

    bool consume(uint32_t epoch, uint32_t manager, uint32_t lane) override {
        if (!ready(epoch,manager,lane)) return false;
        trace(epoch,workerTileBytes());
        retireGeneration();
        return true;
    }

    bool acquire(uint32_t epoch, uint32_t manager, uint32_t lane) override {
        if (!ready(epoch,manager,lane)) return false;
        leased_=true;
        trace(epoch,0);
        return true;
    }

    bool release(uint32_t epoch, uint32_t manager, uint32_t lane) override {
        if (!leased_ || epoch!=consumedEpoch_+1 || core_!=4+manager+4*lane) return false;
        leased_=false;
        retireGeneration();
        return true;
    }

private:
    bool ready(uint32_t epoch,uint32_t manager,uint32_t lane) const {
        return !leased_ && epoch==consumedEpoch_+1 && core_==4+manager+4*lane &&
            received_==UINT32_MAX && landed_==UINT32_MAX && !landingPending_ && incoming_.empty();
    }
    void retireGeneration() {
        ++consumedEpoch_;
        received_=landed_=0;
    }
    void trace(uint32_t epoch,uint32_t relocation) const {
        std::printf("[PV_WO_HANDOFF] core=%u epoch=%u heads=32 payloads=32 payload_bytes=%u gm_write_bytes=%u relocation_read_bytes=%u relocation_write_bytes=%u\n",
            core_,epoch,workerTileBytes(),workerTileBytes(),relocation,relocation);
    }

public:
    void tick() {
        pumpOutgoing();
        if (landingPending_ || incoming_.empty()) return;
        auto& landing = incoming_.front();
        const size_t length = std::min(gm_->localMaxRequestBytes(), payloadBytes() - landing.offset);
        const uint64_t address = gm_->getBaseAddr() + DecoderMemoryLayout::decoder_pv_wo_landing_gm +
            uint64_t(landing.message.workerSlot) * payloadBytes() + landing.offset;
        std::vector<uint8_t> bytes(landing.message.tensorPayload.begin() + landing.offset,
                                  landing.message.tensorPayload.begin() + landing.offset + length);
        landingPending_ = true;
        if (!gm_->localWriteAsync(address, bytes, LocalMemoryClient::RoCC, ++tag_,
            [this, length](bool ok, uint64_t) {
                landingPending_ = false;
                if (!ok || incoming_.empty()) throw std::runtime_error("PV-Wo GM landing failed");
                auto& current = incoming_.front();
                current.offset += length;
                if (current.offset != payloadBytes()) return;
                landed_ |= uint32_t{1} << current.message.workerSlot;
                Message ack{};
                ack.kind = Kind::AttentionPvWoAck;
                ack.jobId = current.message.jobId;
                ack.tag = current.message.tag;
                ack.ownerCore = current.message.ownerCore;
                ack.workerCore = core_;
                ack.groupQueryRowBegin = current.message.groupQueryRowBegin;
                ack.value = 1.0;
                enqueue(ack.ownerCore, std::move(ack));
                incoming_.pop_front();
            })) landingPending_ = false;
    }

private:
    void enqueue(uint32_t target, Message message) {
        if (outgoing_.size() >= 16) throw std::logic_error("PV-Wo outgoing queue overflow");
        outgoing_.emplace_back(target, std::move(message));
    }
    void pumpOutgoing() {
        if (outgoing_.empty()) return;
        auto& item = outgoing_.front();
        if (item.first == core_) {
            auto message = std::move(item.second);
            outgoing_.pop_front();
            receive(message);
        } else if (gm_->sendControlMessage(item.first, item.second)) outgoing_.pop_front();
    }
    GlobalMemoryAPI* gm_;
    uint32_t core_;
    uint64_t tag_ = 0;
    uint32_t consumedEpoch_ = 0, received_ = 0, landed_ = 0;
    bool landingPending_ = false, leased_ = false;
    std::map<Key, Pending> pending_;
    std::deque<Landing> incoming_;
    std::deque<std::pair<uint32_t, Message>> outgoing_;
};
}}
#endif
