#ifndef _H_GOLEM_GROUPCTRL_ENDPOINT
#define _H_GOLEM_GROUPCTRL_ENDPOINT

#include <deque>
#include <array>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>
#include <cinttypes>

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/output.h>
#include <sst/core/params.h>
#include <sst/core/serialization/serializable.h>

#include <sst/elements/golem/globalmemory/globalmemory.h>

namespace SST {
namespace Golem {

enum class GroupCtrlMsgType : uint8_t {
    REQUEST = 0,
    GRANT = 1,
    DONE = 2,
    FINISHED = 3,
    GROUP_DONE = 4,
    ATTENTION_KV_REQUEST = 5,
    ATTENTION_KV_DELIVERY = 6,
    ATTENTION_KV_ACK = 7,
    ATTENTION_KV_CANCEL = 8,
};

enum class GroupCtrlRole : uint8_t {
    WORKER = 0,
    MANAGER = 1,
};

class GroupCtrlMsg : public SST::Event {
public:
    GroupCtrlMsg()
        : SST::Event(), type(GroupCtrlMsgType::REQUEST), groupId(0), workerSlot(0),
          window(0), status(0), reqSeq(0), srcAddr(0), dstAddr(0), bytes(0),
          targetNode(0), generation(0), jobTag(0), queryGroup(0),
          kvTileIndex(0), physicalKvTileIndex(0), numKvTiles(0), kvLength(0),
          kvTileRows(0), tileRows(0),
          rowsPerBand(0), headDim(0), nodeStrideBytes(0), kAddr(0), vAddr(0),
          kDstAddr(0), vDstAddr(0) {}

    explicit GroupCtrlMsg(GroupCtrlMsgType t)
        : GroupCtrlMsg() {
        type = t;
    }

    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        SST::Event::serialize_order(ser);
        ser & type;
        ser & groupId;
        ser & workerSlot;
        ser & window;
        ser & status;
        ser & reqSeq;
        ser & srcAddr;
        ser & dstAddr;
        ser & bytes;
        ser & targetNode;
        ser & generation;
        ser & jobTag;
        ser & queryGroup;
        ser & kvTileIndex;
        ser & physicalKvTileIndex;
        ser & numKvTiles;
        ser & kvLength;
        ser & kvTileRows;
        ser & tileRows;
        ser & rowsPerBand;
        ser & headDim;
        ser & nodeStrideBytes;
        ser & kAddr;
        ser & vAddr;
        ser & kDstAddr;
        ser & vDstAddr;
        ser & kData;
        ser & vData;
    }

    ImplementSerializable(SST::Golem::GroupCtrlMsg);

public:

    GroupCtrlMsgType type;
    uint8_t groupId;
    uint8_t workerSlot;
    uint8_t window;
    uint8_t status;
    uint64_t reqSeq;
    uint64_t srcAddr;
    uint64_t dstAddr;
    uint32_t bytes;
    uint16_t targetNode;
    uint64_t generation;
    uint64_t jobTag;
    uint32_t queryGroup;
    uint32_t kvTileIndex;
    uint32_t physicalKvTileIndex;
    uint32_t numKvTiles;
    uint32_t kvLength;
    uint32_t kvTileRows;
    uint32_t tileRows;
    uint32_t rowsPerBand;
    uint32_t headDim;
    uint64_t nodeStrideBytes;
    uint64_t kAddr;
    uint64_t vAddr;
    uint64_t kDstAddr;
    uint64_t vDstAddr;
    std::vector<uint8_t> kData;
    std::vector<uint8_t> vData;
};

struct AttentionKvRequest {
    uint64_t generation = 0;
    uint64_t jobTag = 0;
    uint32_t queryGroup = 0;
    uint32_t kvTileIndex = 0;
    uint32_t physicalKvTileIndex = 0;
    uint32_t numKvTiles = 0;
    uint32_t kvLength = 0;
    uint32_t kvTileRows = 0;
    uint32_t tileRows = 0;
    uint32_t rowsPerBand = 0;
    uint32_t headDim = 0;
    uint64_t nodeStrideBytes = 0;
    uint64_t kAddr = 0;
    uint64_t vAddr = 0;
    uint64_t kDstAddr = 0;
    uint64_t vDstAddr = 0;
};

class GroupCtrlAPI : public SST::SubComponent {
public:
    SST_ELI_REGISTER_SUBCOMPONENT_API(SST::Golem::GroupCtrlAPI)

    GroupCtrlAPI(ComponentId_t id, SST::Params& params) : SST::SubComponent(id) {}
    virtual ~GroupCtrlAPI() = default;
    virtual void bindGlobalMemory(GlobalMemoryAPI* globalMemory) = 0;
    virtual bool attentionKvDistributionEnabled() const = 0;
    virtual bool requestAttentionKvPair(
        const AttentionKvRequest& request,
        std::function<void(bool)> callback,
        std::function<void(bool)> kReadyCallback = {}) = 0;
    virtual uint32_t cancelAttentionKvGeneration(uint64_t generation) = 0;
};

class GroupCtrlEndpoint : public GroupCtrlAPI {
public:
    SST_ELI_REGISTER_SUBCOMPONENT(
        GroupCtrlEndpoint,
        "golem",
        "GroupCtrlEndpoint",
        SST_ELI_ELEMENT_VERSION(1, 0, 0),
        "Group-local lightweight control endpoint skeleton",
        SST::Golem::GroupCtrlAPI)

    SST_ELI_DOCUMENT_PARAMS(
        {"core_id", "Owning core id", "0"},
        {"group_id", "Owning group id", "0"},
        {"worker_slot", "Worker slot in group; manager uses -1", "-1"},
        {"role", "worker or manager", "worker"},
        {"gm_base_addr", "Local GM base address aligned with data endpoint", "0"},
        {"gm_size", "Local GM window size aligned with data endpoint", "0"},
        {"ctrl_latency", "Control link latency", "2ns"},
        {"queue_depth", "Manager pending queue depth", "32"},
        {"max_inflight_per_node", "Manager per-memory-node inflight cap", "2"},
        {"max_grants_per_schedule", "Max GRANTs issued per scheduling pass", "1"},
        {"num_memory_nodes", "HBM/data node count", "5"},
        {"attention_kv_distribution_enable", "Enable manager-level Attention K/V distribution", "0"},
        {"attention_kv_manager_lookahead", "Stage one sequential K/V tile in a free manager slot", "0"},
        {"attention_kv_distribution_slots", "Manager resident K/V tile slots", "2"},
        {"attention_kv_distribution_tile_bytes", "Maximum bytes per K or V tile", "16384"},
        {"attention_kv_distribution_scratch_offset", "Manager-local scratch offset", "0x40000"},
        {"attention_kv_distribution_expected_workers", "Consumers required before slot release", "4"},
        {"verbose", "Verbosity", "0"})

    SST_ELI_DOCUMENT_STATISTICS(
        {"attention_kv_distribution_requests", "Worker K/V pair requests", "requests", 1},
        {"attention_kv_distribution_manager_loads", "K/V pairs loaded once by a manager", "pairs", 1},
        {"attention_kv_distribution_manager_bytes", "Manager HBM bytes for K/V pairs", "bytes", 1},
        {"attention_kv_distribution_coalesced", "Worker requests coalesced into an existing manager slot", "requests", 1},
        {"attention_kv_distribution_deliveries", "K/V pair deliveries to workers", "deliveries", 1},
        {"attention_kv_distribution_delivery_bytes", "Worker-local K/V delivery bytes", "bytes", 1},
        {"attention_kv_distribution_slot_stalls", "Requests queued because both manager slots were occupied", "requests", 1},
        {"attention_kv_distribution_cancels", "Worker K/V requests cancelled by generation", "requests", 1},
        {"attention_kv_distribution_max_slots", "Maximum simultaneously occupied manager slots", "slots", 1},
        {"attention_kv_manager_lookahead_loads", "K/V pairs speculatively staged by a manager", "pairs", 1},
        {"attention_kv_manager_lookahead_hits", "Worker requests served from speculative manager slots", "requests", 1})

    SST_ELI_DOCUMENT_PORTS(
        {"req_out", "Worker request output", {"SST::Golem::GroupCtrlMsg"}},
        {"rsp_in", "Worker response input", {"SST::Golem::GroupCtrlMsg"}},
        {"req_in_0", "Manager request input from worker slot 0", {"SST::Golem::GroupCtrlMsg"}},
        {"req_in_1", "Manager request input from worker slot 1", {"SST::Golem::GroupCtrlMsg"}},
        {"req_in_2", "Manager request input from worker slot 2", {"SST::Golem::GroupCtrlMsg"}},
        {"req_in_3", "Manager request input from worker slot 3", {"SST::Golem::GroupCtrlMsg"}},
        {"rsp_out_0", "Manager response output to worker slot 0", {"SST::Golem::GroupCtrlMsg"}},
        {"rsp_out_1", "Manager response output to worker slot 1", {"SST::Golem::GroupCtrlMsg"}},
        {"rsp_out_2", "Manager response output to worker slot 2", {"SST::Golem::GroupCtrlMsg"}},
        {"rsp_out_3", "Manager response output to worker slot 3", {"SST::Golem::GroupCtrlMsg"}})

    GroupCtrlEndpoint(SST::ComponentId_t id, SST::Params& params);
    ~GroupCtrlEndpoint() override = default;

    void init(unsigned int phase) override;
    void setup() override;
    void finish() override;
    void bindGlobalMemory(GlobalMemoryAPI* globalMemory) override;
    bool attentionKvDistributionEnabled() const override {
        return attentionKvDistributionEnable_;
    }
    bool requestAttentionKvPair(
        const AttentionKvRequest& request,
        std::function<void(bool)> callback,
        std::function<void(bool)> kReadyCallback = {}) override;
    uint32_t cancelAttentionKvGeneration(uint64_t generation) override;

private:
    struct PendingReq {
        uint8_t workerSlot;
        uint64_t reqSeq;
        uint8_t window;
        uint64_t srcAddr;
        uint64_t dstAddr;
        uint32_t bytes;
        uint16_t targetNode;
    };

    struct WorkerState {
        uint64_t lastReqSeq = 0;
        uint64_t lastGrantSeq = 0;
        uint64_t lastDoneSeq = 0;
        bool finished = false;
        bool inflight = false;
        uint16_t inflightNode = 0;
    };

    struct KvSubscriber {
        bool present = false;
        uint64_t requestId = 0;
        uint64_t generation = 0;
        uint64_t kDstAddr = 0;
        uint64_t vDstAddr = 0;
    };

    struct KvSlot {
        bool occupied = false;
        bool failed = false;
        bool speculative = false;
        bool readInflight = false;
        bool readingV = false;
        uint64_t epoch = 0;
        uint64_t generation = 0;
        uint64_t jobTag = 0;
        uint32_t queryGroup = 0;
        uint32_t kvTileIndex = 0;
        uint32_t physicalKvTileIndex = 0;
        uint32_t numKvTiles = 0;
        uint32_t kvLength = 0;
        uint32_t kvTileRows = 0;
        uint32_t tileRows = 0;
        uint32_t rowsPerBand = 0;
        uint32_t headDim = 0;
        uint64_t nodeStrideBytes = 0;
        uint64_t kAddr = 0;
        uint64_t vAddr = 0;
        uint32_t bytes = 0;
        uint32_t kLoadsPending = 0;
        uint32_t vLoadsPending = 0;
        uint8_t requestedMask = 0;
        uint8_t deliveredMask = 0;
        uint8_t completedMask = 0;
        uint8_t cancelledMask = 0;
        size_t readOffset = 0;
        std::array<KvSubscriber, 4> subscribers = {};
        std::vector<uint8_t> kData;
        std::vector<uint8_t> vData;
    };

    struct WorkerKvCallback {
        uint64_t generation = 0;
        std::function<void(bool)> callback;
        std::function<void(bool)> kReadyCallback;
        bool kReadySent = false;
    };

    struct PendingKvRequest {
        int workerSlot = -1;
        uint64_t requestId = 0;
        AttentionKvRequest request;
    };

    struct WorkerKvDelivery {
        uint64_t generation = 0;
        uint64_t requestId = 0;
        uint64_t kDstAddr = 0;
        uint64_t vDstAddr = 0;
        size_t offset = 0;
        bool writingV = false;
        bool inflight = false;
        std::vector<uint8_t> kData;
        std::vector<uint8_t> vData;
    };

    GroupCtrlRole role_;
    uint32_t coreId_;
    uint32_t groupId_;
    int32_t workerSlot_;
    uint32_t queueDepth_;
    uint32_t maxInflightPerNode_;
    uint32_t maxGrantsPerSchedule_;
    uint32_t numMemoryNodes_;
    std::string ctrlLatency_;
    uint64_t gmBaseAddr_;
    uint64_t gmSize_;
    bool attentionKvDistributionEnable_;
    bool attentionKvManagerLookahead_;
    uint32_t attentionKvDistributionSlots_;
    uint32_t attentionKvDistributionTileBytes_;
    uint64_t attentionKvDistributionScratchOffset_;
    uint32_t attentionKvDistributionExpectedWorkers_;
    int verbose_;
    SST::Output output_;

    SST::Link* reqOut_;
    SST::Link* rspIn_;
    std::vector<SST::Link*> reqIn_;
    std::vector<SST::Link*> rspOut_;

    std::deque<PendingReq> pendingQ_;
    std::vector<WorkerState> workers_;
    std::vector<uint32_t> inflightPerNode_;
    size_t scheduleCursor_;

    uint64_t localGrantSeq_;
    uint8_t localGrantWindow_;
    bool localGroupDone_;
    uint64_t localReqSeqSeen_;
    uint64_t localDoneSeqSeen_;
    bool localFinishedSeen_;
    GlobalMemoryImplement* gm_;
    bool gmBoundLogged_;
    uint64_t nextKvRequestId_;
    uint64_t nextKvSlotEpoch_;
    uint64_t nextKvLocalTag_;
    std::vector<KvSlot> kvSlots_;
    std::deque<PendingKvRequest> pendingKvRequests_;
    std::unordered_map<uint64_t, WorkerKvCallback> workerKvCallbacks_;
    std::unordered_map<uint64_t, WorkerKvDelivery> workerKvDeliveries_;

    Statistics::Statistic<uint64_t>* statKvRequests_;
    Statistics::Statistic<uint64_t>* statKvManagerLoads_;
    Statistics::Statistic<uint64_t>* statKvManagerBytes_;
    Statistics::Statistic<uint64_t>* statKvCoalesced_;
    Statistics::Statistic<uint64_t>* statKvDeliveries_;
    Statistics::Statistic<uint64_t>* statKvDeliveryBytes_;
    Statistics::Statistic<uint64_t>* statKvSlotStalls_;
    Statistics::Statistic<uint64_t>* statKvCancels_;
    Statistics::Statistic<uint64_t>* statKvMaxSlots_;
    Statistics::Statistic<uint64_t>* statKvManagerLookaheadLoads_;
    Statistics::Statistic<uint64_t>* statKvManagerLookaheadHits_;

    GroupCtrlRole parseRole(const std::string& role) const;
    static uint32_t parseU32Param(SST::Params& params, const std::string& key, uint32_t defaultValue);
    static int32_t parseI32Param(SST::Params& params, const std::string& key, int32_t defaultValue);
    static uint64_t parseU64Param(SST::Params& params, const std::string& key, uint64_t defaultValue);
    void configureLinks();
    void handleReq0(SST::Event* ev);
    void handleReq1(SST::Event* ev);
    void handleReq2(SST::Event* ev);
    void handleReq3(SST::Event* ev);
    void handleReq(SST::Event* ev, int slot);
    void handleRsp(SST::Event* ev);
    bool tick(SST::Cycle_t cycle);
    void trySchedule();
    bool allWorkersFinished() const;
    bool groupDrained() const;
    void maybeSendGroupDone();
    void sendRsp(int slot, GroupCtrlMsg* msg);
    void handleAttentionKvRequest(const GroupCtrlMsg& message, int slot);
    void handleAttentionKvAck(const GroupCtrlMsg& message, int slot);
    void handleAttentionKvCancel(const GroupCtrlMsg& message, int slot);
    int findAttentionKvSlot(const GroupCtrlMsg& message) const;
    int findFreeAttentionKvSlot() const;
    void startAttentionKvSlot(size_t slotIndex, const GroupCtrlMsg& message);
    void addAttentionKvSubscriber(size_t slotIndex, const GroupCtrlMsg& message, int workerSlot);
    void issueAttentionKvDma(size_t slotIndex, bool valueOperand);
    void completeAttentionKvDma(size_t slotIndex, uint64_t epoch,
                                bool valueOperand, bool ok);
    void pumpAttentionKvManagerRead(size_t slotIndex);
    void deliverAttentionKvSlot(size_t slotIndex);
    void maybeReleaseAttentionKvSlot(size_t slotIndex, bool allowLookahead = true);
    void maybeStartAttentionKvLookahead(const KvSlot& released);
    void processPendingAttentionKvRequests();
    void handleAttentionKvDelivery(GroupCtrlMsg* message);
    void pumpWorkerAttentionKvDelivery(uint64_t requestId);
    void completeWorkerAttentionKvDelivery(uint64_t requestId, bool ok);
    void sendAttentionKvAck(uint64_t requestId, uint64_t generation);
    uint64_t mailboxAddr(uint64_t off) const;
    uint64_t readMailbox(uint64_t off) const;
    void writeMailbox(uint64_t off, uint64_t value);
    uint32_t readMailboxU32(uint64_t off) const;
    void writeMailboxU32(uint64_t off, uint32_t value);
};

} // namespace Golem
} // namespace SST

#endif
