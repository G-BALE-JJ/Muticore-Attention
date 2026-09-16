#include <sst/core/link.h>

#include <algorithm>
#include <stdexcept>

#include <sst/elements/golem/groupctrl/groupctrl.h>

namespace {
constexpr uint64_t CTRL_LOCAL_MAILBOX_BASE = 0x1900;
constexpr uint64_t CTRL_LOCAL_REQ_SEQ_OFF = 0x00;
constexpr uint64_t CTRL_LOCAL_REQ_VALID_OFF = 0x08;
constexpr uint64_t CTRL_LOCAL_GRANT_SEQ_OFF = 0x10;
constexpr uint64_t CTRL_LOCAL_GRANT_WINDOW_OFF = 0x18;
constexpr uint64_t CTRL_LOCAL_DONE_SEQ_OFF = 0x20;
constexpr uint64_t CTRL_LOCAL_DONE_VALID_OFF = 0x28;
constexpr uint64_t CTRL_LOCAL_FINISHED_OFF = 0x30;
constexpr uint64_t CTRL_LOCAL_GROUP_DONE_OFF = 0x38;
constexpr uint64_t CTRL_LOCAL_REQ_SRC_OFF = 0x40;
constexpr uint64_t CTRL_LOCAL_REQ_DST_OFF = 0x48;
constexpr uint64_t CTRL_LOCAL_REQ_BYTES_OFF = 0x50;
constexpr uint64_t CTRL_LOCAL_REQ_NODE_OFF = 0x58;
constexpr uint64_t CTRL_LOCAL_REQ_WINDOW_OFF = 0x60;
constexpr uint64_t GOLEM_WCP_COARSE_FINISHED_FLAG = 0x8000000000000000ULL;

SST::Golem::AttentionKvRequest attentionKvRequestFromMessage(
    const SST::Golem::GroupCtrlMsg& message) {
    SST::Golem::AttentionKvRequest request;
    request.generation = message.generation;
    request.jobTag = message.jobTag;
    request.queryGroup = message.queryGroup;
    request.kvTileIndex = message.kvTileIndex;
    request.physicalKvTileIndex = message.physicalKvTileIndex;
    request.numKvTiles = message.numKvTiles;
    request.kvLength = message.kvLength;
    request.kvTileRows = message.kvTileRows;
    request.tileRows = message.tileRows;
    request.rowsPerBand = message.rowsPerBand;
    request.headDim = message.headDim;
    request.nodeStrideBytes = message.nodeStrideBytes;
    request.kAddr = message.kAddr;
    request.vAddr = message.vAddr;
    request.kDstAddr = message.kDstAddr;
    request.vDstAddr = message.vDstAddr;
    return request;
}

void fillAttentionKvMessage(
    SST::Golem::GroupCtrlMsg& message,
    const SST::Golem::AttentionKvRequest& request) {
    message.generation = request.generation;
    message.jobTag = request.jobTag;
    message.queryGroup = request.queryGroup;
    message.kvTileIndex = request.kvTileIndex;
    message.physicalKvTileIndex = request.physicalKvTileIndex;
    message.numKvTiles = request.numKvTiles;
    message.kvLength = request.kvLength;
    message.kvTileRows = request.kvTileRows;
    message.tileRows = request.tileRows;
    message.rowsPerBand = request.rowsPerBand;
    message.headDim = request.headDim;
    message.nodeStrideBytes = request.nodeStrideBytes;
    message.kAddr = request.kAddr;
    message.vAddr = request.vAddr;
    message.kDstAddr = request.kDstAddr;
    message.vDstAddr = request.vDstAddr;
}
}

namespace SST {
namespace Golem {

GroupCtrlEndpoint::GroupCtrlEndpoint(SST::ComponentId_t id, SST::Params& params)
    : GroupCtrlAPI(id, params),
      role_(parseRole(params.find<std::string>("role", "worker"))),
      coreId_(parseU32Param(params, "core_id", 0)),
      groupId_(parseU32Param(params, "group_id", 0)),
      workerSlot_(parseI32Param(params, "worker_slot", -1)),
      queueDepth_(parseU32Param(params, "queue_depth", 32)),
      maxInflightPerNode_(parseU32Param(params, "max_inflight_per_node", 2)),
      maxGrantsPerSchedule_(parseU32Param(params, "max_grants_per_schedule", 1)),
      numMemoryNodes_(parseU32Param(params, "num_memory_nodes", 5)),
      ctrlLatency_(params.find<std::string>("ctrl_latency", "2ns")),
      gmBaseAddr_(parseU64Param(params, "gm_base_addr", 0)),
      gmSize_(parseU64Param(params, "gm_size", 0)),
      attentionKvDistributionEnable_(
          params.find<bool>("attention_kv_distribution_enable", false)),
      attentionKvManagerLookahead_(
          params.find<bool>("attention_kv_manager_lookahead", false)),
      attentionKvDistributionSlots_(parseU32Param(
          params, "attention_kv_distribution_slots", 2)),
      attentionKvDistributionTileBytes_(parseU32Param(
          params, "attention_kv_distribution_tile_bytes", 16384)),
      attentionKvDistributionScratchOffset_(parseU64Param(
          params, "attention_kv_distribution_scratch_offset", 0x40000)),
      attentionKvDistributionExpectedWorkers_(parseU32Param(
          params, "attention_kv_distribution_expected_workers", 4)),
      verbose_(parseI32Param(params, "verbose", 0)),
      output_("GroupCtrlEndpoint[@p:@l]: ", verbose_, 0, SST::Output::STDOUT),
      reqOut_(nullptr),
      rspIn_(nullptr),
      scheduleCursor_(0),
      localGrantSeq_(0),
      localGrantWindow_(0),
      localGroupDone_(false),
      localReqSeqSeen_(0),
      localDoneSeqSeen_(0),
      localFinishedSeen_(false),
      gm_(nullptr),
      gmBoundLogged_(false),
      nextKvRequestId_(1),
      nextKvSlotEpoch_(1),
      nextKvLocalTag_(1) {
    if (maxInflightPerNode_ == 0) {
        maxInflightPerNode_ = 1;
    }
    if (maxGrantsPerSchedule_ == 0) {
        maxGrantsPerSchedule_ = 1;
    }
    if (attentionKvDistributionSlots_ == 0 ||
        attentionKvDistributionTileBytes_ == 0 ||
        attentionKvDistributionExpectedWorkers_ == 0 ||
        attentionKvDistributionExpectedWorkers_ > 4) {
        throw std::runtime_error(
            "GroupCtrlEndpoint invalid Attention K/V distribution dimensions");
    }
    reqIn_.resize(4, nullptr);
    rspOut_.resize(4, nullptr);
    workers_.resize(4);
    inflightPerNode_.resize(numMemoryNodes_, 0);
    kvSlots_.resize(attentionKvDistributionSlots_);
    statKvRequests_ = registerStatistic<uint64_t>(
        "attention_kv_distribution_requests");
    statKvManagerLoads_ = registerStatistic<uint64_t>(
        "attention_kv_distribution_manager_loads");
    statKvManagerBytes_ = registerStatistic<uint64_t>(
        "attention_kv_distribution_manager_bytes");
    statKvCoalesced_ = registerStatistic<uint64_t>(
        "attention_kv_distribution_coalesced");
    statKvDeliveries_ = registerStatistic<uint64_t>(
        "attention_kv_distribution_deliveries");
    statKvDeliveryBytes_ = registerStatistic<uint64_t>(
        "attention_kv_distribution_delivery_bytes");
    statKvSlotStalls_ = registerStatistic<uint64_t>(
        "attention_kv_distribution_slot_stalls");
    statKvCancels_ = registerStatistic<uint64_t>(
        "attention_kv_distribution_cancels");
    statKvMaxSlots_ = registerStatistic<uint64_t>(
        "attention_kv_distribution_max_slots");
    statKvManagerLookaheadLoads_ = registerStatistic<uint64_t>(
        "attention_kv_manager_lookahead_loads");
    statKvManagerLookaheadHits_ = registerStatistic<uint64_t>(
        "attention_kv_manager_lookahead_hits");
    configureLinks();
    registerClock("1GHz", new SST::Clock::Handler<GroupCtrlEndpoint>(this, &GroupCtrlEndpoint::tick));
}

uint32_t GroupCtrlEndpoint::parseU32Param(SST::Params& params, const std::string& key, uint32_t defaultValue)
{
    bool found = false;
    std::string value = params.find<std::string>(key, "", found);
    if (!found || value.empty()) {
        return defaultValue;
    }
    try {
        return static_cast<uint32_t>(std::stoull(value, nullptr, 0));
    } catch (const std::exception& e) {
        throw std::runtime_error("GroupCtrlEndpoint invalid u32 param '" + key + "' = '" + value + "': " + e.what());
    }
}

int32_t GroupCtrlEndpoint::parseI32Param(SST::Params& params, const std::string& key, int32_t defaultValue)
{
    bool found = false;
    std::string value = params.find<std::string>(key, "", found);
    if (!found || value.empty()) {
        return defaultValue;
    }
    try {
        return static_cast<int32_t>(std::stoll(value, nullptr, 0));
    } catch (const std::exception& e) {
        throw std::runtime_error("GroupCtrlEndpoint invalid i32 param '" + key + "' = '" + value + "': " + e.what());
    }
}

uint64_t GroupCtrlEndpoint::parseU64Param(SST::Params& params, const std::string& key, uint64_t defaultValue)
{
    bool found = false;
    std::string value = params.find<std::string>(key, "", found);
    if (!found || value.empty()) {
        return defaultValue;
    }
    try {
        return static_cast<uint64_t>(std::stoull(value, nullptr, 0));
    } catch (const std::exception& e) {
        throw std::runtime_error("GroupCtrlEndpoint invalid u64 param '" + key + "' = '" + value + "': " + e.what());
    }
}

GroupCtrlRole GroupCtrlEndpoint::parseRole(const std::string& role) const {
    if (role == "manager") {
        return GroupCtrlRole::MANAGER;
    }
    return GroupCtrlRole::WORKER;
}

void GroupCtrlEndpoint::configureLinks() {
    auto* tc = getTimeConverter(ctrlLatency_);
    if (role_ == GroupCtrlRole::WORKER) {
        reqOut_ = configureLink("req_out", tc);
        rspIn_ = configureLink("rsp_in", tc, new SST::Event::Handler<GroupCtrlEndpoint>(this, &GroupCtrlEndpoint::handleRsp));
        return;
    }

    for (int slot = 0; slot < 4; ++slot) {
        SST::Event::HandlerBase* handler = nullptr;
        if (slot == 0) {
            handler = new SST::Event::Handler<GroupCtrlEndpoint>(this, &GroupCtrlEndpoint::handleReq0);
        } else if (slot == 1) {
            handler = new SST::Event::Handler<GroupCtrlEndpoint>(this, &GroupCtrlEndpoint::handleReq1);
        } else if (slot == 2) {
            handler = new SST::Event::Handler<GroupCtrlEndpoint>(this, &GroupCtrlEndpoint::handleReq2);
        } else {
            handler = new SST::Event::Handler<GroupCtrlEndpoint>(this, &GroupCtrlEndpoint::handleReq3);
        }
        reqIn_[slot] = configureLink(
            "req_in_" + std::to_string(slot),
            tc,
            handler);
        rspOut_[slot] = configureLink("rsp_out_" + std::to_string(slot), tc);
    }
}

void GroupCtrlEndpoint::handleReq0(SST::Event* ev) { handleReq(ev, 0); }
void GroupCtrlEndpoint::handleReq1(SST::Event* ev) { handleReq(ev, 1); }
void GroupCtrlEndpoint::handleReq2(SST::Event* ev) { handleReq(ev, 2); }
void GroupCtrlEndpoint::handleReq3(SST::Event* ev) { handleReq(ev, 3); }

void GroupCtrlEndpoint::init(unsigned int phase) {
    (void)phase;
}

void GroupCtrlEndpoint::bindGlobalMemory(GlobalMemoryAPI* globalMemory) {
    gm_ = dynamic_cast<GlobalMemoryImplement*>(globalMemory);
    if (globalMemory != nullptr && gm_ == nullptr) {
        output_.fatal(CALL_INFO, -1,
            "core=%u group control requires golem.GlobalMemory, not the local fallback\n",
            coreId_);
    }
}

void GroupCtrlEndpoint::setup() {
    if (role_ == GroupCtrlRole::WORKER) {
        output_.verbose(CALL_INFO, 1, 0,
            "core=%u link_status req_out=%s rsp_in=%s\n",
            coreId_,
            reqOut_ ? "connected" : "MISSING",
            rspIn_ ? "connected" : "MISSING");
        if (reqOut_ == nullptr) {
            output_.fatal(CALL_INFO, -1,
                "core=%u worker missing required req_out link\n",
                coreId_);
        }
        if (rspIn_ == nullptr) {
            output_.fatal(CALL_INFO, -1,
                "core=%u worker missing required rsp_in link\n",
                coreId_);
        }
    } else {
        for (int slot = 0; slot < static_cast<int>(reqIn_.size()); ++slot) {
            output_.verbose(CALL_INFO, 1, 0,
                "core=%u link_status req_in_%d=%s rsp_out_%d=%s\n",
                coreId_,
                slot,
                reqIn_[slot] ? "connected" : "MISSING",
                slot,
                rspOut_[slot] ? "connected" : "MISSING");
            if (reqIn_[slot] == nullptr) {
                output_.fatal(CALL_INFO, -1,
                    "core=%u manager missing required req_in_%d link\n",
                    coreId_, slot);
            }
            if (rspOut_[slot] == nullptr) {
                output_.fatal(CALL_INFO, -1,
                    "core=%u manager missing required rsp_out_%d link\n",
                    coreId_, slot);
            }
        }
    }

    if (gm_ != nullptr) {
        gmBoundLogged_ = true;
        output_.verbose(CALL_INFO, 1, 0,
            "core=%u role=%s bound local GM in setup mailbox_base=0x%" PRIx64 "\n",
            coreId_, role_ == GroupCtrlRole::MANAGER ? "manager" : "worker", mailboxAddr(0));
    }
    if (attentionKvDistributionEnable_ && gm_ == nullptr) {
        output_.fatal(CALL_INFO, -1,
            "core=%u Attention K/V distribution requires GlobalMemory\n", coreId_);
    }
    if (attentionKvDistributionEnable_ && role_ == GroupCtrlRole::MANAGER) {
        const uint64_t scratchBytes =
            static_cast<uint64_t>(attentionKvDistributionSlots_) * 2 *
            attentionKvDistributionTileBytes_;
        if (attentionKvDistributionScratchOffset_ < 0x2000 ||
            attentionKvDistributionScratchOffset_ > gmSize_ ||
            scratchBytes > gmSize_ - attentionKvDistributionScratchOffset_ ||
            attentionKvDistributionScratchOffset_ + scratchBytes > gmSize_ - 64) {
            output_.fatal(CALL_INFO, -1,
                "core=%u invalid Attention K/V scratch offset=0x%" PRIx64
                " bytes=%" PRIu64 " gm_size=0x%" PRIx64 "\n",
                coreId_, attentionKvDistributionScratchOffset_, scratchBytes,
                gmSize_);
        }
    }
    output_.verbose(CALL_INFO, 1, 0,
        "core=%u group=%u role=%s worker_slot=%d queueDepth=%u maxInflightPerNode=%u maxGrantsPerSchedule=%u gm_base=0x%" PRIx64 " gm_size=0x%" PRIx64 "\n",
        coreId_, groupId_, role_ == GroupCtrlRole::MANAGER ? "manager" : "worker",
        workerSlot_, queueDepth_, maxInflightPerNode_, maxGrantsPerSchedule_, gmBaseAddr_, gmSize_);
}

void GroupCtrlEndpoint::finish() {
    if (role_ == GroupCtrlRole::WORKER) {
        output_.verbose(CALL_INFO, 1, 0,
            "worker core=%u final grant_seq=%" PRIu64 " grant_window=%u group_done=%d\n",
            coreId_, localGrantSeq_, localGrantWindow_, localGroupDone_ ? 1 : 0);
        return;
    }

    output_.verbose(CALL_INFO, 1, 0,
        "manager core=%u pending_q=%zu inflight_nodes=%zu\n",
        coreId_, pendingQ_.size(), inflightPerNode_.size());
}

void GroupCtrlEndpoint::handleReq(SST::Event* ev, int slot) {
    auto* msg = dynamic_cast<GroupCtrlMsg*>(ev);
    if (msg == nullptr) {
        output_.fatal(CALL_INFO, -1, "core=%u manager received non-GroupCtrlMsg on slot=%d\n", coreId_, slot);
    }

    if (slot < 0 || slot >= static_cast<int>(workers_.size())) {
        output_.fatal(CALL_INFO, -1, "core=%u invalid manager slot=%d\n", coreId_, slot);
    }

    switch (msg->type) {
    case GroupCtrlMsgType::ATTENTION_KV_REQUEST:
        handleAttentionKvRequest(*msg, slot);
        break;
    case GroupCtrlMsgType::ATTENTION_KV_ACK:
        handleAttentionKvAck(*msg, slot);
        break;
    case GroupCtrlMsgType::ATTENTION_KV_CANCEL:
        handleAttentionKvCancel(*msg, slot);
        break;
    case GroupCtrlMsgType::REQUEST: {
        if (pendingQ_.size() >= queueDepth_) {
            output_.verbose(CALL_INFO, 1, 0,
                "core=%u queue full dropping request seq=%" PRIu64 " slot=%d\n",
                coreId_, msg->reqSeq, slot);
            break;
        }
        PendingReq req = {
            .workerSlot = static_cast<uint8_t>(slot),
            .reqSeq = msg->reqSeq,
            .window = msg->window,
            .srcAddr = msg->srcAddr,
            .dstAddr = msg->dstAddr,
            .bytes = msg->bytes,
            .targetNode = msg->targetNode,
        };
        workers_[slot].lastReqSeq = msg->reqSeq;
        pendingQ_.push_back(req);
        output_.verbose(CALL_INFO, 1, 0,
            "manager core=%u recv REQUEST slot=%d req=%" PRIu64 " node=%u q=%zu\n",
            coreId_, slot, msg->reqSeq, msg->targetNode, pendingQ_.size());
        trySchedule();
        break;
    }
    case GroupCtrlMsgType::DONE:
        workers_[slot].lastDoneSeq = msg->reqSeq;
        workers_[slot].inflight = false;
        if (workers_[slot].inflightNode < inflightPerNode_.size() && inflightPerNode_[workers_[slot].inflightNode] > 0) {
            inflightPerNode_[workers_[slot].inflightNode]--;
        }
        workers_[slot].inflightNode = 0;
        output_.verbose(CALL_INFO, 1, 0,
            "manager core=%u recv DONE slot=%d req=%" PRIu64 "\n",
            coreId_, slot, msg->reqSeq);
        trySchedule();
        maybeSendGroupDone();
        break;
    case GroupCtrlMsgType::FINISHED:
        workers_[slot].finished = true;
        if ((workers_[slot].lastGrantSeq & GOLEM_WCP_COARSE_FINISHED_FLAG) != 0) {
            if (workers_[slot].inflight) {
                const auto node = workers_[slot].inflightNode;
                if (node < inflightPerNode_.size() && inflightPerNode_[node] > 0) {
                    inflightPerNode_[node]--;
                }
            }
            workers_[slot].inflight = false;
            workers_[slot].inflightNode = 0;
        }
        output_.verbose(CALL_INFO, 1, 0,
            "manager core=%u recv FINISHED slot=%d\n",
            coreId_, slot);
        maybeSendGroupDone();
        break;
    default:
        output_.verbose(CALL_INFO, 1, 0,
            "core=%u manager received unsupported msg type=%u on slot=%d\n",
            coreId_, static_cast<unsigned>(msg->type), slot);
        break;
    }

    delete msg;
}

bool GroupCtrlEndpoint::tick(SST::Cycle_t)
{
    if (gm_ == nullptr) {
        return false;
    }

    if (role_ == GroupCtrlRole::WORKER) {
        std::vector<uint64_t> deliveryIds;
        deliveryIds.reserve(workerKvDeliveries_.size());
        for (const auto& entry : workerKvDeliveries_) {
            deliveryIds.push_back(entry.first);
        }
        for (const uint64_t requestId : deliveryIds) {
            pumpWorkerAttentionKvDelivery(requestId);
        }
        const uint64_t reqValid = readMailboxU32(CTRL_LOCAL_REQ_VALID_OFF);
        const uint64_t reqSeq = readMailboxU32(CTRL_LOCAL_REQ_SEQ_OFF);
        if (reqValid != 0 && reqSeq > localReqSeqSeen_) {
            output_.verbose(CALL_INFO, 1, 0,
                "worker core=%u observed mailbox req_valid=%" PRIu64 " req_seq=%" PRIu64 " req_addr=0x%" PRIx64 " valid_addr=0x%" PRIx64 "\n",
                coreId_, reqValid, reqSeq,
                mailboxAddr(CTRL_LOCAL_REQ_SEQ_OFF),
                mailboxAddr(CTRL_LOCAL_REQ_VALID_OFF));
        }
        if (reqValid != 0 && reqSeq > localReqSeqSeen_) {
            auto* req = new GroupCtrlMsg(GroupCtrlMsgType::REQUEST);
            req->groupId = static_cast<uint8_t>(groupId_);
            req->workerSlot = static_cast<uint8_t>(workerSlot_);
            req->reqSeq = reqSeq;
            req->srcAddr = readMailbox(CTRL_LOCAL_REQ_SRC_OFF);
            req->dstAddr = readMailbox(CTRL_LOCAL_REQ_DST_OFF);
            req->bytes = static_cast<uint32_t>(readMailbox(CTRL_LOCAL_REQ_BYTES_OFF));
            req->targetNode = static_cast<uint16_t>(readMailbox(CTRL_LOCAL_REQ_NODE_OFF));
            req->window = static_cast<uint8_t>(readMailbox(CTRL_LOCAL_REQ_WINDOW_OFF));
            if (reqOut_ != nullptr) {
                reqOut_->send(req);
                localReqSeqSeen_ = reqSeq;
                writeMailboxU32(CTRL_LOCAL_REQ_VALID_OFF, 0);
                output_.verbose(CALL_INFO, 1, 0,
                    "worker core=%u send REQUEST req=%" PRIu64 " node=%u\n",
                    coreId_, reqSeq, req->targetNode);
            } else {
                delete req;
                output_.fatal(CALL_INFO, -1,
                    "worker core=%u req_out link missing while sending REQUEST req=%" PRIu64 "\n",
                    coreId_, reqSeq);
            }
        }

        const uint64_t doneValid = readMailboxU32(CTRL_LOCAL_DONE_VALID_OFF);
        const uint64_t doneSeq = readMailboxU32(CTRL_LOCAL_DONE_SEQ_OFF);
        if (doneValid != 0 && doneSeq > localDoneSeqSeen_) {
            auto* done = new GroupCtrlMsg(GroupCtrlMsgType::DONE);
            done->groupId = static_cast<uint8_t>(groupId_);
            done->workerSlot = static_cast<uint8_t>(workerSlot_);
            done->reqSeq = doneSeq;
            done->targetNode = static_cast<uint16_t>(readMailbox(CTRL_LOCAL_REQ_NODE_OFF));
            if (reqOut_ != nullptr) {
                reqOut_->send(done);
                localDoneSeqSeen_ = doneSeq;
                writeMailboxU32(CTRL_LOCAL_DONE_VALID_OFF, 0);
                output_.verbose(CALL_INFO, 1, 0,
                    "worker core=%u send DONE req=%" PRIu64 "\n",
                    coreId_, doneSeq);
            } else {
                delete done;
                output_.fatal(CALL_INFO, -1,
                    "worker core=%u req_out link missing while sending DONE req=%" PRIu64 "\n",
                    coreId_, doneSeq);
            }
        }

        const bool finished = readMailbox(CTRL_LOCAL_FINISHED_OFF) != 0;
        if (finished && !localFinishedSeen_) {
            auto* fin = new GroupCtrlMsg(GroupCtrlMsgType::FINISHED);
            fin->groupId = static_cast<uint8_t>(groupId_);
            fin->workerSlot = static_cast<uint8_t>(workerSlot_);
            if (reqOut_ != nullptr) {
                reqOut_->send(fin);
                localFinishedSeen_ = true;
                writeMailbox(CTRL_LOCAL_FINISHED_OFF, 0);
            } else {
                delete fin;
                output_.fatal(CALL_INFO, -1,
                    "worker core=%u req_out link missing while sending FINISHED\n",
                    coreId_);
            }
        }
    } else if (attentionKvDistributionEnable_) {
        for (size_t slotIndex = 0; slotIndex < kvSlots_.size(); ++slotIndex) {
            pumpAttentionKvManagerRead(slotIndex);
        }
    }

    return false;
}

void GroupCtrlEndpoint::handleRsp(SST::Event* ev) {
    auto* msg = dynamic_cast<GroupCtrlMsg*>(ev);
    if (msg == nullptr) {
        output_.fatal(CALL_INFO, -1, "core=%u worker received non-GroupCtrlMsg\n", coreId_);
    }

    switch (msg->type) {
    case GroupCtrlMsgType::ATTENTION_KV_DELIVERY:
        handleAttentionKvDelivery(msg);
        return;
    case GroupCtrlMsgType::GRANT:
        localGrantSeq_ = msg->reqSeq;
        localGrantWindow_ = msg->window;
        writeMailboxU32(CTRL_LOCAL_GRANT_SEQ_OFF, static_cast<uint32_t>(localGrantSeq_));
        writeMailboxU32(CTRL_LOCAL_GRANT_WINDOW_OFF, static_cast<uint32_t>(localGrantWindow_));
        output_.verbose(CALL_INFO, 1, 0,
            "worker core=%u recv GRANT req=%" PRIu64 " window=%u\n",
            coreId_, localGrantSeq_, localGrantWindow_);
        break;
    case GroupCtrlMsgType::GROUP_DONE:
        localGroupDone_ = true;
        writeMailboxU32(CTRL_LOCAL_GROUP_DONE_OFF, 1);
        output_.verbose(CALL_INFO, 1, 0,
            "worker core=%u recv GROUP_DONE\n",
            coreId_);
        break;
    default:
        output_.verbose(CALL_INFO, 1, 0,
            "core=%u worker received unsupported msg type=%u\n",
            coreId_, static_cast<unsigned>(msg->type));
        break;
    }

    delete msg;
}

bool GroupCtrlEndpoint::requestAttentionKvPair(
    const AttentionKvRequest& request,
    std::function<void(bool)> callback,
    std::function<void(bool)> kReadyCallback) {
    if (!attentionKvDistributionEnable_ || role_ != GroupCtrlRole::WORKER ||
        reqOut_ == nullptr || !callback || request.generation == 0 ||
        request.tileRows == 0 || request.rowsPerBand == 0 ||
        request.headDim == 0 || request.tileRows > request.kvTileRows ||
        workerSlot_ < 0 || workerSlot_ >= 4 ||
        workerKvCallbacks_.size() >= queueDepth_) {
        return false;
    }
    const uint64_t bytes = static_cast<uint64_t>(request.tileRows) *
        request.headDim * sizeof(float);
    if (bytes == 0 || bytes > attentionKvDistributionTileBytes_ ||
        bytes > UINT32_MAX) {
        return false;
    }

    const uint64_t requestId = nextKvRequestId_++;
    workerKvCallbacks_.emplace(
        requestId, WorkerKvCallback{
            request.generation, std::move(callback),
            std::move(kReadyCallback), false});
    auto* message = new GroupCtrlMsg(GroupCtrlMsgType::ATTENTION_KV_REQUEST);
    message->groupId = static_cast<uint8_t>(groupId_);
    message->workerSlot = static_cast<uint8_t>(workerSlot_);
    message->reqSeq = requestId;
    message->bytes = static_cast<uint32_t>(bytes);
    fillAttentionKvMessage(*message, request);
    reqOut_->send(message);
    statKvRequests_->addData(1);
    return true;
}

uint32_t GroupCtrlEndpoint::cancelAttentionKvGeneration(uint64_t generation) {
    if (!attentionKvDistributionEnable_ || generation == 0) return 0;
    uint32_t cancelled = 0;
    if (role_ == GroupCtrlRole::WORKER) {
        for (auto it = workerKvCallbacks_.begin();
             it != workerKvCallbacks_.end();) {
            if (it->second.generation == generation) {
                workerKvDeliveries_.erase(it->first);
                it = workerKvCallbacks_.erase(it);
                ++cancelled;
            } else {
                ++it;
            }
        }
        if (reqOut_ != nullptr) {
            auto* message = new GroupCtrlMsg(GroupCtrlMsgType::ATTENTION_KV_CANCEL);
            message->groupId = static_cast<uint8_t>(groupId_);
            message->workerSlot = static_cast<uint8_t>(workerSlot_);
            message->generation = generation;
            reqOut_->send(message);
        }
        if (cancelled != 0) statKvCancels_->addData(cancelled);
    }
    return cancelled;
}

int GroupCtrlEndpoint::findAttentionKvSlot(const GroupCtrlMsg& message) const {
    for (size_t index = 0; index < kvSlots_.size(); ++index) {
        const KvSlot& slot = kvSlots_[index];
        if (slot.occupied && slot.jobTag == message.jobTag &&
            slot.queryGroup == message.queryGroup &&
            slot.kvTileIndex == message.kvTileIndex &&
            slot.physicalKvTileIndex == message.physicalKvTileIndex) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

int GroupCtrlEndpoint::findFreeAttentionKvSlot() const {
    for (size_t index = 0; index < kvSlots_.size(); ++index) {
        if (!kvSlots_[index].occupied) return static_cast<int>(index);
    }
    return -1;
}

void GroupCtrlEndpoint::handleAttentionKvRequest(
    const GroupCtrlMsg& message, int workerSlot) {
    if (!attentionKvDistributionEnable_ || role_ != GroupCtrlRole::MANAGER ||
        workerSlot < 0 || workerSlot >= 4 || message.generation == 0 ||
        message.bytes == 0 || message.numKvTiles == 0 ||
        message.kvLength == 0 || message.kvTileIndex >= message.numKvTiles ||
        message.bytes > attentionKvDistributionTileBytes_) {
        auto* failure = new GroupCtrlMsg(GroupCtrlMsgType::ATTENTION_KV_DELIVERY);
        failure->reqSeq = message.reqSeq;
        failure->generation = message.generation;
        failure->status = 0;
        sendRsp(workerSlot, failure);
        return;
    }

    int slotIndex = findAttentionKvSlot(message);
    if (slotIndex >= 0) {
        KvSlot& slot = kvSlots_[static_cast<size_t>(slotIndex)];
        const bool compatible = slot.bytes == message.bytes &&
            slot.kvTileRows == message.kvTileRows &&
            slot.numKvTiles == message.numKvTiles &&
            slot.kvLength == message.kvLength &&
            slot.tileRows == message.tileRows &&
            slot.rowsPerBand == message.rowsPerBand &&
            slot.headDim == message.headDim &&
            slot.nodeStrideBytes == message.nodeStrideBytes &&
            slot.kAddr == message.kAddr && slot.vAddr == message.vAddr;
        if (!compatible || slot.subscribers[workerSlot].present) {
            auto* failure = new GroupCtrlMsg(GroupCtrlMsgType::ATTENTION_KV_DELIVERY);
            failure->reqSeq = message.reqSeq;
            failure->generation = message.generation;
            failure->status = 0;
            sendRsp(workerSlot, failure);
            return;
        }
        statKvCoalesced_->addData(1);
        addAttentionKvSubscriber(
            static_cast<size_t>(slotIndex), message, workerSlot);
        deliverAttentionKvSlot(static_cast<size_t>(slotIndex));
        return;
    }

    slotIndex = findFreeAttentionKvSlot();
    if (slotIndex < 0) {
        if (pendingKvRequests_.size() >= queueDepth_) {
            auto* failure = new GroupCtrlMsg(GroupCtrlMsgType::ATTENTION_KV_DELIVERY);
            failure->reqSeq = message.reqSeq;
            failure->generation = message.generation;
            failure->status = 0;
            sendRsp(workerSlot, failure);
            return;
        }
        pendingKvRequests_.push_back(PendingKvRequest{
            workerSlot, message.reqSeq, attentionKvRequestFromMessage(message)});
        statKvSlotStalls_->addData(1);
        return;
    }

    startAttentionKvSlot(static_cast<size_t>(slotIndex), message);
    addAttentionKvSubscriber(static_cast<size_t>(slotIndex), message, workerSlot);
    issueAttentionKvDma(static_cast<size_t>(slotIndex), false);
    issueAttentionKvDma(static_cast<size_t>(slotIndex), true);
}

void GroupCtrlEndpoint::startAttentionKvSlot(
    size_t slotIndex, const GroupCtrlMsg& message) {
    KvSlot slot;
    slot.occupied = true;
    slot.epoch = nextKvSlotEpoch_++;
    slot.generation = message.generation;
    slot.jobTag = message.jobTag;
    slot.queryGroup = message.queryGroup;
    slot.kvTileIndex = message.kvTileIndex;
    slot.physicalKvTileIndex = message.physicalKvTileIndex;
    slot.numKvTiles = message.numKvTiles;
    slot.kvLength = message.kvLength;
    slot.kvTileRows = message.kvTileRows;
    slot.tileRows = message.tileRows;
    slot.rowsPerBand = message.rowsPerBand;
    slot.headDim = message.headDim;
    slot.nodeStrideBytes = message.nodeStrideBytes;
    slot.kAddr = message.kAddr;
    slot.vAddr = message.vAddr;
    slot.bytes = message.bytes;
    slot.kData.reserve(slot.bytes);
    slot.vData.reserve(slot.bytes);
    kvSlots_[slotIndex] = std::move(slot);
    statKvManagerLoads_->addData(1);
    statKvManagerBytes_->addData(static_cast<uint64_t>(2) * message.bytes);
    uint64_t occupied = 0;
    for (const KvSlot& candidate : kvSlots_) occupied += candidate.occupied ? 1 : 0;
    statKvMaxSlots_->addData(occupied);
}

void GroupCtrlEndpoint::addAttentionKvSubscriber(
    size_t slotIndex, const GroupCtrlMsg& message, int workerSlot) {
    KvSlot& slot = kvSlots_[slotIndex];
    if (slot.speculative) {
        slot.speculative = false;
        statKvManagerLookaheadHits_->addData(1);
    }
    KvSubscriber& subscriber = slot.subscribers[workerSlot];
    subscriber.present = true;
    subscriber.requestId = message.reqSeq;
    subscriber.generation = message.generation;
    subscriber.kDstAddr = message.kDstAddr;
    subscriber.vDstAddr = message.vDstAddr;
    slot.requestedMask |= static_cast<uint8_t>(1u << workerSlot);
}

void GroupCtrlEndpoint::issueAttentionKvDma(
    size_t slotIndex, bool valueOperand) {
    KvSlot& slot = kvSlots_[slotIndex];
    uint32_t& pending = valueOperand ? slot.vLoadsPending : slot.kLoadsPending;
    const uint64_t tensorBase = valueOperand ? slot.vAddr : slot.kAddr;
    const uint64_t scratchBase = gmBaseAddr_ +
        attentionKvDistributionScratchOffset_ +
        slotIndex * static_cast<uint64_t>(2) * attentionKvDistributionTileBytes_ +
        (valueOperand ? attentionKvDistributionTileBytes_ : 0);
    const uint32_t firstRow = slot.physicalKvTileIndex * slot.kvTileRows;
    uint32_t rowsIssued = 0;
    while (rowsIssued < slot.tileRows) {
        const uint32_t globalRow = firstRow + rowsIssued;
        const uint32_t nodeBand = globalRow / slot.rowsPerBand;
        const uint32_t rowInBand = globalRow % slot.rowsPerBand;
        const uint32_t rows = std::min(
            slot.tileRows - rowsIssued, slot.rowsPerBand - rowInBand);
        const uint64_t rowBytes =
            static_cast<uint64_t>(slot.headDim) * sizeof(float);
        const uint64_t source = tensorBase +
            static_cast<uint64_t>(nodeBand) * slot.nodeStrideBytes +
            static_cast<uint64_t>(rowInBand) * rowBytes;
        const uint64_t destination = scratchBase +
            static_cast<uint64_t>(rowsIssued) * rowBytes;
        const size_t bytes = static_cast<size_t>(rows) * rowBytes;
        ++pending;
        const uint64_t epoch = slot.epoch;
        gm_->dma_read_from_host_to_globalmem(
            source, bytes, destination,
            [this, slotIndex, epoch, valueOperand](bool ok) {
                completeAttentionKvDma(slotIndex, epoch, valueOperand, ok);
            }, DmaRequestKind::AttentionKvPrefetch);
        rowsIssued += rows;
    }
}

void GroupCtrlEndpoint::completeAttentionKvDma(
    size_t slotIndex, uint64_t epoch, bool valueOperand, bool ok) {
    if (slotIndex >= kvSlots_.size()) return;
    KvSlot& slot = kvSlots_[slotIndex];
    if (!slot.occupied || slot.epoch != epoch) return;
    uint32_t& pending = valueOperand ? slot.vLoadsPending : slot.kLoadsPending;
    if (pending == 0) return;
    slot.failed = slot.failed || !ok;
    --pending;
    if (slot.kLoadsPending == 0 && slot.vLoadsPending == 0) {
        if (slot.failed) deliverAttentionKvSlot(slotIndex);
        else pumpAttentionKvManagerRead(slotIndex);
    }
}

void GroupCtrlEndpoint::pumpAttentionKvManagerRead(size_t slotIndex) {
    if (slotIndex >= kvSlots_.size()) return;
    KvSlot& slot = kvSlots_[slotIndex];
    if (!slot.occupied || slot.failed || slot.readInflight ||
        slot.kLoadsPending != 0 || slot.vLoadsPending != 0) return;
    std::vector<uint8_t>& destination = slot.readingV ? slot.vData : slot.kData;
    if (slot.readOffset == slot.bytes) {
        if (!slot.readingV) {
            slot.readingV = true;
            slot.readOffset = 0;
            pumpAttentionKvManagerRead(slotIndex);
        } else {
            deliverAttentionKvSlot(slotIndex);
        }
        return;
    }
    const size_t chunk = std::min(
        static_cast<size_t>(slot.bytes - slot.readOffset),
        gm_->localMaxRequestBytes());
    const uint64_t source = gmBaseAddr_ +
        attentionKvDistributionScratchOffset_ +
        slotIndex * static_cast<uint64_t>(2) * attentionKvDistributionTileBytes_ +
        (slot.readingV ? attentionKvDistributionTileBytes_ : 0) +
        slot.readOffset;
    const uint64_t tag = nextKvLocalTag_++;
    const uint64_t epoch = slot.epoch;
    const bool accepted = gm_->localReadAsync(
        source, chunk, LocalMemoryClient::Control, tag,
        [this, slotIndex, epoch, tag](
            bool ok, uint64_t callbackTag, const std::vector<uint8_t>& bytes) {
            if (slotIndex >= kvSlots_.size()) return;
            KvSlot& callbackSlot = kvSlots_[slotIndex];
            if (!callbackSlot.occupied || callbackSlot.epoch != epoch) return;
            callbackSlot.readInflight = false;
            if (!ok || callbackTag != tag || bytes.empty()) {
                callbackSlot.failed = true;
                deliverAttentionKvSlot(slotIndex);
                return;
            }
            std::vector<uint8_t>& data = callbackSlot.readingV
                ? callbackSlot.vData : callbackSlot.kData;
            data.insert(data.end(), bytes.begin(), bytes.end());
            callbackSlot.readOffset += bytes.size();
            pumpAttentionKvManagerRead(slotIndex);
        });
    if (accepted) slot.readInflight = true;
}

void GroupCtrlEndpoint::deliverAttentionKvSlot(size_t slotIndex) {
    if (slotIndex >= kvSlots_.size()) return;
    KvSlot& slot = kvSlots_[slotIndex];
    const bool ready = slot.failed ||
        (slot.kData.size() == slot.bytes && slot.vData.size() == slot.bytes);
    if (!slot.occupied || !ready) return;
    for (int workerSlot = 0; workerSlot < 4; ++workerSlot) {
        const uint8_t bit = static_cast<uint8_t>(1u << workerSlot);
        const KvSubscriber& subscriber = slot.subscribers[workerSlot];
        if (!subscriber.present || (slot.deliveredMask & bit) != 0) continue;
        auto* delivery = new GroupCtrlMsg(GroupCtrlMsgType::ATTENTION_KV_DELIVERY);
        delivery->groupId = static_cast<uint8_t>(groupId_);
        delivery->workerSlot = static_cast<uint8_t>(workerSlot);
        delivery->reqSeq = subscriber.requestId;
        delivery->generation = subscriber.generation;
        delivery->kDstAddr = subscriber.kDstAddr;
        delivery->vDstAddr = subscriber.vDstAddr;
        delivery->status = slot.failed ? 0 : 1;
        if (!slot.failed) {
            delivery->kData = slot.kData;
            delivery->vData = slot.vData;
        }
        sendRsp(workerSlot, delivery);
        slot.deliveredMask |= bit;
        statKvDeliveries_->addData(1);
        if (!slot.failed) {
            statKvDeliveryBytes_->addData(
                static_cast<uint64_t>(slot.kData.size() + slot.vData.size()));
        }
    }
}

void GroupCtrlEndpoint::handleAttentionKvDelivery(GroupCtrlMsg* message) {
    const uint64_t requestId = message->reqSeq;
    auto callbackIt = workerKvCallbacks_.find(requestId);
    if (!attentionKvDistributionEnable_ || role_ != GroupCtrlRole::WORKER ||
        callbackIt == workerKvCallbacks_.end() ||
        callbackIt->second.generation != message->generation ||
        message->status == 0 || message->kData.empty() ||
        message->kData.size() != message->vData.size()) {
        if (callbackIt != workerKvCallbacks_.end()) {
            completeWorkerAttentionKvDelivery(requestId, false);
        }
        delete message;
        return;
    }
    WorkerKvDelivery delivery;
    delivery.generation = message->generation;
    delivery.requestId = requestId;
    delivery.kDstAddr = message->kDstAddr;
    delivery.vDstAddr = message->vDstAddr;
    delivery.kData = std::move(message->kData);
    delivery.vData = std::move(message->vData);
    workerKvDeliveries_.emplace(requestId, std::move(delivery));
    delete message;
    pumpWorkerAttentionKvDelivery(requestId);
}

void GroupCtrlEndpoint::pumpWorkerAttentionKvDelivery(uint64_t requestId) {
    auto deliveryIt = workerKvDeliveries_.find(requestId);
    if (deliveryIt == workerKvDeliveries_.end() || gm_ == nullptr) return;
    WorkerKvDelivery& delivery = deliveryIt->second;
    if (delivery.inflight) return;
    const std::vector<uint8_t>& source =
        delivery.writingV ? delivery.vData : delivery.kData;
    if (delivery.offset == source.size()) {
        if (!delivery.writingV) {
            auto callbackIt = workerKvCallbacks_.find(requestId);
            if (callbackIt == workerKvCallbacks_.end()) {
                completeWorkerAttentionKvDelivery(requestId, false);
                return;
            }
            if (!callbackIt->second.kReadySent) {
                callbackIt->second.kReadySent = true;
                if (callbackIt->second.kReadyCallback) {
                    callbackIt->second.kReadyCallback(true);
                }
            }
            delivery.writingV = true;
            delivery.offset = 0;
            pumpWorkerAttentionKvDelivery(requestId);
        } else {
            completeWorkerAttentionKvDelivery(requestId, true);
        }
        return;
    }
    const size_t chunk = std::min(
        source.size() - delivery.offset, gm_->localMaxRequestBytes());
    std::vector<uint8_t> bytes(
        source.begin() + delivery.offset,
        source.begin() + delivery.offset + chunk);
    const uint64_t destination =
        (delivery.writingV ? delivery.vDstAddr : delivery.kDstAddr) +
        delivery.offset;
    const uint64_t tag = nextKvLocalTag_++;
    const bool accepted = gm_->localWriteAsync(
        destination, bytes, LocalMemoryClient::Control, tag,
        [this, requestId, tag, chunk](bool ok, uint64_t callbackTag) {
            auto it = workerKvDeliveries_.find(requestId);
            if (it == workerKvDeliveries_.end()) return;
            it->second.inflight = false;
            if (!ok || callbackTag != tag) {
                completeWorkerAttentionKvDelivery(requestId, false);
                return;
            }
            it->second.offset += chunk;
            pumpWorkerAttentionKvDelivery(requestId);
        });
    if (accepted) delivery.inflight = true;
}

void GroupCtrlEndpoint::completeWorkerAttentionKvDelivery(
    uint64_t requestId, bool ok) {
    auto callbackIt = workerKvCallbacks_.find(requestId);
    if (callbackIt == workerKvCallbacks_.end()) {
        workerKvDeliveries_.erase(requestId);
        return;
    }
    const uint64_t generation = callbackIt->second.generation;
    const bool kReadySent = callbackIt->second.kReadySent;
    auto kReadyCallback = std::move(callbackIt->second.kReadyCallback);
    auto callback = std::move(callbackIt->second.callback);
    workerKvCallbacks_.erase(callbackIt);
    workerKvDeliveries_.erase(requestId);
    sendAttentionKvAck(requestId, generation);
    if (!kReadySent && kReadyCallback) kReadyCallback(false);
    callback(ok);
}

void GroupCtrlEndpoint::sendAttentionKvAck(
    uint64_t requestId, uint64_t generation) {
    if (reqOut_ == nullptr) return;
    auto* ack = new GroupCtrlMsg(GroupCtrlMsgType::ATTENTION_KV_ACK);
    ack->groupId = static_cast<uint8_t>(groupId_);
    ack->workerSlot = static_cast<uint8_t>(workerSlot_);
    ack->reqSeq = requestId;
    ack->generation = generation;
    reqOut_->send(ack);
}

void GroupCtrlEndpoint::handleAttentionKvAck(
    const GroupCtrlMsg& message, int workerSlot) {
    if (!attentionKvDistributionEnable_ || role_ != GroupCtrlRole::MANAGER ||
        workerSlot < 0 || workerSlot >= 4) return;
    for (size_t index = 0; index < kvSlots_.size(); ++index) {
        KvSlot& slot = kvSlots_[index];
        const KvSubscriber& subscriber = slot.subscribers[workerSlot];
        if (!slot.occupied || !subscriber.present ||
            subscriber.requestId != message.reqSeq ||
            subscriber.generation != message.generation) continue;
        slot.completedMask |= static_cast<uint8_t>(1u << workerSlot);
        maybeReleaseAttentionKvSlot(index);
        return;
    }
}

void GroupCtrlEndpoint::handleAttentionKvCancel(
    const GroupCtrlMsg& message, int workerSlot) {
    if (!attentionKvDistributionEnable_ || role_ != GroupCtrlRole::MANAGER ||
        workerSlot < 0 || workerSlot >= 4 || message.generation == 0) return;
    uint32_t cancelled = 0;
    for (size_t index = 0; index < kvSlots_.size(); ++index) {
        KvSlot& slot = kvSlots_[index];
        KvSubscriber& subscriber = slot.subscribers[workerSlot];
        if (!slot.occupied || slot.generation != message.generation) continue;
        const uint8_t bit = static_cast<uint8_t>(1u << workerSlot);
        if (subscriber.present && subscriber.generation == message.generation) {
            slot.completedMask |= bit;
            ++cancelled;
        } else {
            slot.cancelledMask |= bit;
        }
        maybeReleaseAttentionKvSlot(index, false);
    }
    for (auto it = pendingKvRequests_.begin();
         it != pendingKvRequests_.end();) {
        if (it->workerSlot == workerSlot &&
            it->request.generation == message.generation) {
            it = pendingKvRequests_.erase(it);
            ++cancelled;
        } else {
            ++it;
        }
    }
    if (cancelled != 0) statKvCancels_->addData(cancelled);
}

void GroupCtrlEndpoint::maybeReleaseAttentionKvSlot(
    size_t slotIndex, bool allowLookahead) {
    if (slotIndex >= kvSlots_.size() || !kvSlots_[slotIndex].occupied) return;
    const uint8_t expectedMask = static_cast<uint8_t>(
        (1u << attentionKvDistributionExpectedWorkers_) - 1u);
    const KvSlot& slot = kvSlots_[slotIndex];
    const uint8_t accountedMask = static_cast<uint8_t>(
        slot.requestedMask | slot.cancelledMask);
    const uint8_t doneMask = static_cast<uint8_t>(
        slot.completedMask | slot.cancelledMask);
    if ((accountedMask & expectedMask) != expectedMask ||
        (doneMask & expectedMask) != expectedMask) return;
    KvSlot released = std::move(kvSlots_[slotIndex]);
    kvSlots_[slotIndex] = KvSlot{};
    processPendingAttentionKvRequests();
    if (allowLookahead) maybeStartAttentionKvLookahead(released);
}

void GroupCtrlEndpoint::maybeStartAttentionKvLookahead(const KvSlot& released) {
    if (!attentionKvManagerLookahead_ || role_ != GroupCtrlRole::MANAGER ||
        released.failed || released.numKvTiles == 0 || released.kvLength == 0 ||
        !pendingKvRequests_.empty()) return;
    const int freeSlot = findFreeAttentionKvSlot();
    if (freeSlot < 0) return;

    uint32_t farthestOrdinal = released.kvTileIndex;
    for (const KvSlot& slot : kvSlots_) {
        if (slot.occupied && slot.jobTag == released.jobTag &&
            slot.queryGroup == released.queryGroup &&
            slot.numKvTiles == released.numKvTiles) {
            farthestOrdinal = std::max(farthestOrdinal, slot.kvTileIndex);
        }
    }
    const uint32_t nextOrdinal = farthestOrdinal + 1;
    if (nextOrdinal >= released.numKvTiles) return;
    const uint32_t ordinalDelta = nextOrdinal - released.kvTileIndex;
    const uint32_t nextTile =
        (released.physicalKvTileIndex + ordinalDelta) % released.numKvTiles;
    const uint64_t firstRow =
        static_cast<uint64_t>(nextTile) * released.kvTileRows;
    if (firstRow >= released.kvLength) return;
    const uint32_t tileRows = static_cast<uint32_t>(std::min<uint64_t>(
        released.kvTileRows, released.kvLength - firstRow));

    GroupCtrlMsg message(GroupCtrlMsgType::ATTENTION_KV_REQUEST);
    message.generation = released.generation;
    message.jobTag = released.jobTag;
    message.queryGroup = released.queryGroup;
    message.kvTileIndex = nextOrdinal;
    message.physicalKvTileIndex = nextTile;
    message.numKvTiles = released.numKvTiles;
    message.kvLength = released.kvLength;
    message.kvTileRows = released.kvTileRows;
    message.tileRows = tileRows;
    message.rowsPerBand = released.rowsPerBand;
    message.headDim = released.headDim;
    message.nodeStrideBytes = released.nodeStrideBytes;
    message.kAddr = released.kAddr;
    message.vAddr = released.vAddr;
    message.bytes = tileRows * released.headDim * sizeof(float);
    startAttentionKvSlot(static_cast<size_t>(freeSlot), message);
    kvSlots_[static_cast<size_t>(freeSlot)].speculative = true;
    statKvManagerLookaheadLoads_->addData(1);
    issueAttentionKvDma(static_cast<size_t>(freeSlot), false);
    issueAttentionKvDma(static_cast<size_t>(freeSlot), true);
}

void GroupCtrlEndpoint::processPendingAttentionKvRequests() {
    while (!pendingKvRequests_.empty()) {
        PendingKvRequest pending = std::move(pendingKvRequests_.front());
        GroupCtrlMsg message(GroupCtrlMsgType::ATTENTION_KV_REQUEST);
        message.reqSeq = pending.requestId;
        message.bytes = static_cast<uint32_t>(
            static_cast<uint64_t>(pending.request.tileRows) *
            pending.request.headDim * sizeof(float));
        fillAttentionKvMessage(message, pending.request);
        const int matching = findAttentionKvSlot(message);
        if (matching < 0 && findFreeAttentionKvSlot() < 0) return;
        pendingKvRequests_.pop_front();
        handleAttentionKvRequest(message, pending.workerSlot);
    }
}

void GroupCtrlEndpoint::trySchedule() {
    if (role_ != GroupCtrlRole::MANAGER) {
        return;
    }

    uint32_t grantsIssued = 0;
    while (!pendingQ_.empty() && grantsIssued < maxGrantsPerSchedule_) {
        if (scheduleCursor_ >= pendingQ_.size()) {
            scheduleCursor_ = 0;
        }

        bool scheduled = false;
        const size_t qsize = pendingQ_.size();
        for (size_t scanned = 0; scanned < qsize; ++scanned) {
            const size_t idx = (scheduleCursor_ + scanned) % qsize;
            const auto& req = pendingQ_[idx];
            const auto slot = static_cast<size_t>(req.workerSlot);
            if (slot >= workers_.size()) {
                continue;
            }
            if (workers_[slot].inflight) {
                continue;
            }
            if (req.targetNode >= inflightPerNode_.size()) {
                continue;
            }
            if (inflightPerNode_[req.targetNode] >= maxInflightPerNode_) {
                continue;
            }

            auto* grant = new GroupCtrlMsg(GroupCtrlMsgType::GRANT);
            grant->groupId = static_cast<uint8_t>(groupId_);
            grant->workerSlot = req.workerSlot;
            grant->reqSeq = req.reqSeq;
            grant->window = req.window == 0 ? 1 : req.window;
            grant->targetNode = req.targetNode;
            sendRsp(req.workerSlot, grant);
            output_.verbose(CALL_INFO, 1, 0,
                "manager core=%u send GRANT slot=%u req=%" PRIu64 " node=%u\n",
                coreId_, req.workerSlot, req.reqSeq, req.targetNode);

            workers_[slot].lastGrantSeq = req.reqSeq;
            if (req.window > 1) {
                workers_[slot].lastGrantSeq |= GOLEM_WCP_COARSE_FINISHED_FLAG;
            }
            workers_[slot].inflight = true;
            workers_[slot].inflightNode = req.targetNode;
            inflightPerNode_[req.targetNode]++;
            pendingQ_.erase(pendingQ_.begin() + idx);
            scheduleCursor_ = pendingQ_.empty() ? 0 : (idx % pendingQ_.size());
            scheduled = true;
            ++grantsIssued;
            break;
        }

        if (!scheduled) {
            break;
        }
    }
}

bool GroupCtrlEndpoint::allWorkersFinished() const {
    for (const auto& worker : workers_) {
        if (!worker.finished) {
            return false;
        }
    }
    return true;
}

bool GroupCtrlEndpoint::groupDrained() const {
    if (!pendingQ_.empty()) {
        return false;
    }
    for (const auto& worker : workers_) {
        if (worker.inflight) {
            return false;
        }
    }
    for (const auto inflight : inflightPerNode_) {
        if (inflight != 0) {
            return false;
        }
    }
    return true;
}

void GroupCtrlEndpoint::maybeSendGroupDone() {
    if (role_ != GroupCtrlRole::MANAGER) {
        return;
    }
    if (!allWorkersFinished() || !groupDrained()) {
        return;
    }
    output_.verbose(CALL_INFO, 1, 0,
        "manager core=%u send GROUP_DONE\n",
        coreId_);
    for (int slot = 0; slot < static_cast<int>(rspOut_.size()); ++slot) {
        auto* done = new GroupCtrlMsg(GroupCtrlMsgType::GROUP_DONE);
        done->groupId = static_cast<uint8_t>(groupId_);
        done->workerSlot = static_cast<uint8_t>(slot);
        sendRsp(slot, done);
    }
}

void GroupCtrlEndpoint::sendRsp(int slot, GroupCtrlMsg* msg) {
    if (slot < 0 || slot >= static_cast<int>(rspOut_.size()) || rspOut_[slot] == nullptr) {
        delete msg;
        return;
    }
    if (msg->type == GroupCtrlMsgType::GRANT && msg->reqSeq == 0) {
        msg->window = 0;
    }
    rspOut_[slot]->send(msg);
}

uint64_t GroupCtrlEndpoint::mailboxAddr(uint64_t off) const
{
    return gmBaseAddr_ + CTRL_LOCAL_MAILBOX_BASE + off;
}

uint64_t GroupCtrlEndpoint::readMailbox(uint64_t off) const
{
    return gm_->ctrlReadLocalU64(mailboxAddr(off));
}

void GroupCtrlEndpoint::writeMailbox(uint64_t off, uint64_t value)
{
    gm_->ctrlWriteLocalU64(mailboxAddr(off), value);
}

uint32_t GroupCtrlEndpoint::readMailboxU32(uint64_t off) const
{
    return static_cast<uint32_t>(readMailbox(off) & 0xffffffffULL);
}

void GroupCtrlEndpoint::writeMailboxU32(uint64_t off, uint32_t value)
{
    const uint64_t baseOff = off & ~0x7ULL;
    const uint64_t shift = (off - baseOff) * 8ULL;
    const uint64_t mask = 0xffffffffULL << shift;
    const uint64_t cur = readMailbox(baseOff);
    const uint64_t next = (cur & ~mask) | (static_cast<uint64_t>(value) << shift);
    writeMailbox(baseOff, next);
}

} // namespace Golem
} // namespace SST
