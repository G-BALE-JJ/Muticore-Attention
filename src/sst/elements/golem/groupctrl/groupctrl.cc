#include <sst/core/link.h>

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
      verbose_(parseI32Param(params, "verbose", 0)),
      output_("GroupCtrlEndpoint[@p:@l]: ", verbose_, 0, SST::Output::STDOUT),
      reqOut_(nullptr),
      rspIn_(nullptr),
      localGrantSeq_(0),
      localGrantWindow_(0),
      localGroupDone_(false),
      localReqSeqSeen_(0),
      localDoneSeqSeen_(0),
      localFinishedSeen_(false),
      gm_(nullptr),
      gmBoundLogged_(false),
      scheduleCursor_(0) {
    if (maxInflightPerNode_ == 0) {
        maxInflightPerNode_ = 1;
    }
    if (maxGrantsPerSchedule_ == 0) {
        maxGrantsPerSchedule_ = 1;
    }
    reqIn_.resize(4, nullptr);
    rspOut_.resize(4, nullptr);
    workers_.resize(4);
    inflightPerNode_.resize(numMemoryNodes_, 0);
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
    }

    return false;
}

void GroupCtrlEndpoint::handleRsp(SST::Event* ev) {
    auto* msg = dynamic_cast<GroupCtrlMsg*>(ev);
    if (msg == nullptr) {
        output_.fatal(CALL_INFO, -1, "core=%u worker received non-GroupCtrlMsg\n", coreId_);
    }

    switch (msg->type) {
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
