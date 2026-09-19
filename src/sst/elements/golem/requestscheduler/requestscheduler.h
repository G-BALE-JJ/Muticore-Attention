#ifndef _H_GOLEM_REQUEST_SCHEDULER_ENDPOINT
#define _H_GOLEM_REQUEST_SCHEDULER_ENDPOINT

#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/output.h>
#include <sst/core/params.h>
#include <sst/core/serialization/serializable.h>
#include <sst/core/interfaces/simpleNetwork.h>

#include <sst/elements/golem/globalmemory/globalmemory.h>

namespace SST {
namespace Golem {

enum class RequestSchedulerMsgType : uint8_t {
    SUBMIT = 0,
    DONE = 1,
};

enum class RequestSchedulerRole : uint8_t {
    WORKER = 0,
    MANAGER = 1,
};

class RequestSchedulerMsg : public SST::Event {
public:
    RequestSchedulerMsg()
        : SST::Event(), type(RequestSchedulerMsgType::SUBMIT), groupId(0), workerSlot(0),
          srcCoreId(0), requestId(0), srcAddr(0), dstAddr(0), bytes(0), targetNode(0),
          completionFlagAddr(0), completionValue(0), pendingClearCycle(0), batchCount(0) {}

    explicit RequestSchedulerMsg(RequestSchedulerMsgType t) : RequestSchedulerMsg() { type = t; }

    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        SST::Event::serialize_order(ser);
        ser & type;
        ser & groupId;
        ser & workerSlot;
        ser & srcCoreId;
        ser & requestId;
        ser & srcAddr;
        ser & dstAddr;
        ser & bytes;
        ser & targetNode;
        ser & completionFlagAddr;
        ser & completionValue;
        ser & pendingClearCycle;
        ser & batchCount;
        ser & batchRequestIds;
        ser & batchSrcAddrs;
        ser & batchDstAddrs;
        ser & batchBytes;
        ser & batchTargetNodes;
        ser & batchCompletionFlagAddrs;
        ser & batchCompletionValues;
        ser & batchPendingClearCycles;
    }

    ImplementSerializable(SST::Golem::RequestSchedulerMsg);

public:
    RequestSchedulerMsgType type;
    uint8_t groupId;
    uint8_t workerSlot;
    uint16_t srcCoreId;
    uint64_t requestId;
    uint64_t srcAddr;
    uint64_t dstAddr;
    uint32_t bytes;
    uint16_t targetNode;
    uint64_t completionFlagAddr;
    uint64_t completionValue;
    uint64_t pendingClearCycle;
    uint16_t batchCount;
    std::vector<uint64_t> batchRequestIds;
    std::vector<uint64_t> batchSrcAddrs;
    std::vector<uint64_t> batchDstAddrs;
    std::vector<uint32_t> batchBytes;
    std::vector<uint16_t> batchTargetNodes;
    std::vector<uint64_t> batchCompletionFlagAddrs;
    std::vector<uint64_t> batchCompletionValues;
    std::vector<uint64_t> batchPendingClearCycles;
};

struct WcpWindowTransaction {
    uint32_t workerSlot = 0;
    uint32_t taskId = 0;
    uint32_t windowId = 0;
    uint32_t kBegin = 0;
    uint32_t kTiles = 0;
    uint32_t blockN = 0;
    uint32_t blockK = 0;
    uint32_t elemBytes = 0;
    uint32_t hwInputSize = 0;
    uint64_t memNodeSize = 0;
    uint64_t matBaseAddr = 0;
    uint64_t vecBaseAddr = 0;
    uint64_t localMatBaseAddr = 0;
    uint64_t localMatAltBaseAddr = 0;
    uint64_t localVecBaseAddr = 0;
    uint64_t localVecAltBaseAddr = 0;
    uint64_t localMatSlotStrideBytes = 0;
    uint64_t localVecSlotStrideBytes = 0;
    uint32_t slotCount = 2;
    bool skipMatRead = false;
    bool skipVecRead = false;
    uint64_t matStrideBytes = 0;
    uint64_t vecStrideBytes = 0;
    bool useIndependentMatVecTiles = false;
    uint32_t matTileCount = 0;
    uint32_t vecTileCount = 0;
    uint32_t kWindowTiles = 0;
    uint32_t totalKTileCount = 0;
    bool rowMajorPanels = false;
    uint32_t matPanelRows = 0;
    uint32_t vecPanelRows = 0;
    uint64_t matSourceRowStrideBytes = 0;
    uint64_t vecSourceRowStrideBytes = 0;
    uint64_t matGroupStrideBytes = 0;
    uint64_t vecGroupStrideBytes = 0;
    uint64_t kSliceStrideBytes = 0;
};

struct WcpTileTimelineDebug {
    uint64_t submitCycle = 0;
    uint64_t matDoneCycle = 0;
    uint64_t vecDoneCycle = 0;
    uint64_t readyCycle = 0;
};

class RequestSchedulerAPI : public SST::SubComponent {
public:
    SST_ELI_REGISTER_SUBCOMPONENT_API(SST::Golem::RequestSchedulerAPI)
    RequestSchedulerAPI(ComponentId_t id, SST::Params& params) : SST::SubComponent(id) {}
    virtual ~RequestSchedulerAPI() = default;

    virtual void bindGlobalMemory(GlobalMemoryAPI* globalMemory) = 0;
    virtual uint64_t submitWindowTransaction(const WcpWindowTransaction& txn) = 0;
    virtual bool isTileReady(uint64_t txnId, uint32_t localTileIdx) const = 0;
    virtual bool getTileKStepReadiness(
        uint64_t txnId,
        uint32_t localTileIdx,
        uint32_t kStep,
        bool& matReady,
        bool& vecReady) const = 0;
    virtual bool getTileTimeline(uint64_t txnId, uint32_t localTileIdx, WcpTileTimelineDebug& out) const = 0;
    virtual uint64_t getTimelineCycle() const = 0;
    virtual void retireTileReady(uint64_t txnId, uint32_t localTileIdx) = 0;
    virtual bool isTransactionDone(uint64_t txnId) const = 0;
    virtual void retireTransaction(uint64_t txnId) = 0;
};

class RequestSchedulerEndpoint : public RequestSchedulerAPI {
public:
    SST_ELI_REGISTER_SUBCOMPONENT(
        RequestSchedulerEndpoint,
        "golem",
        "RequestSchedulerEndpoint",
        SST_ELI_ELEMENT_VERSION(1, 0, 0),
        "Transfer request scheduler endpoint",
        SST::Golem::RequestSchedulerAPI)

    SST_ELI_DOCUMENT_PARAMS(
        {"core_id", "Owning core id", "0"},
        {"group_id", "Owning group id", "0"},
        {"worker_slot", "Worker slot in group; manager uses -1", "-1"},
        {"role", "worker or manager", "worker"},
        {"gm_base_addr", "Local GM base address", "0"},
        {"gm_size", "Local GM size", "0"},
        {"ctrl_latency", "Control link latency", "2ns"},
        {"queue_depth", "Manager queue depth", "64"},
        {"panel_chunk_bytes", "Scheduler issue chunk size for one logical A/B panel", "2048"},
        {"worker_credit_cap", "Optional per-worker outstanding panel cap; 0 derives it from slot/window capacity", "0"},
        {"manager_issue_budget_per_tick", "Max manager-issued read requests per scheduler tick", "2"},
        {"tile_chunk_quantum", "A/B chunk rounds issued for one worker tile before group round-robin advances", "1"},
        {"prefetch_windows", "WCP prefetch windows; 2D worker issue credit is derived from active+prefetch windows", "1"},
        {"local_slot_count", "WCP local panel slot count used to derive worker credits", "2"},
        {"a_reuse_n_tiles", "A reuse fanout used to derive worker credits", "1"},
        {"b_reuse_m_tiles", "B reuse fanout used to derive worker credits", "1"},
        {"submit_batch_size", "Max logical submits packed into one worker->manager ctrl msg", "4"},
        {"done_batch_size", "Max local done entries packed into one worker->manager ctrl msg", "4"},
        {"num_memory_nodes", "HBM/data node count", "5"},
        {"infer_submit_bytes", "Infer submit bytes from request slot instead of mailbox bytes", "0"},
        {"slot0_bytes", "Inferred bytes for slot0 requests", "0"},
        {"slot1_bytes", "Inferred bytes for slot1 requests", "0"},
        {"event_driven_worker", "Skip idle worker scheduler scans until direct work or DMA completion", "0"},
        {"group_round_robin", "Manager issues one chunk per group worker in round-robin order", "0"},
        {"link_bw", "Bandwidth of the router link", "50GB/s"},
        {"buffer_length", "Network buffer length", "64KB"},
        {"verbose", "Verbosity", "0"})

    SST_ELI_DOCUMENT_PORTS(
        {"req_out", "Worker request output", {"SST::Golem::RequestSchedulerMsg"}},
        {"req_in_0", "Manager request input from worker slot 0", {"SST::Golem::RequestSchedulerMsg"}},
        {"req_in_1", "Manager request input from worker slot 1", {"SST::Golem::RequestSchedulerMsg"}},
        {"req_in_2", "Manager request input from worker slot 2", {"SST::Golem::RequestSchedulerMsg"}},
        {"req_in_3", "Manager request input from worker slot 3", {"SST::Golem::RequestSchedulerMsg"}},
        {"rtr", "Manager scheduler network port", {"merlin.RtrEvent", "merlin.credit_event"}})

    SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS(
        {"networkIF", "Network interface (SimpleNetwork)", "SST::Interfaces::SimpleNetwork"})

    RequestSchedulerEndpoint(SST::ComponentId_t id, SST::Params& params);
    ~RequestSchedulerEndpoint() override = default;

    void init(unsigned int phase) override;
    void setup() override;
    void finish() override;
    void bindGlobalMemory(GlobalMemoryAPI* globalMemory) override;

    uint64_t submitWindowTransaction(const WcpWindowTransaction& txn) override;
    bool isTileReady(uint64_t txnId, uint32_t localTileIdx) const override;
    bool getTileKStepReadiness(
        uint64_t txnId,
        uint32_t localTileIdx,
        uint32_t kStep,
        bool& matReady,
        bool& vecReady) const override;
    bool getTileTimeline(uint64_t txnId, uint32_t localTileIdx, WcpTileTimelineDebug& out) const override;
    uint64_t getTimelineCycle() const override;
    void retireTileReady(uint64_t txnId, uint32_t localTileIdx) override;
    bool isTransactionDone(uint64_t txnId) const override;
    void retireTransaction(uint64_t txnId) override;

private:
    struct PendingTransfer {
        uint8_t workerSlot;
        uint16_t workerCoreId;
        uint64_t requestId;
        uint64_t srcAddr;
        uint64_t dstAddr;
        uint32_t bytes;
        uint16_t targetNode;
        uint64_t completionFlagAddr;
        uint64_t completionValue;
        uint32_t issuedBytes = 0;
    };

    struct WorkerTileRequestState {
        uint64_t submitCycle = 0;
        uint64_t matRequestId = 0;
        uint64_t vecRequestId = 0;
        std::vector<uint64_t> matRequestIds;
        std::vector<uint64_t> vecRequestIds;
        std::vector<uint8_t> matDoneSentKStep;
        std::vector<uint8_t> vecDoneSentKStep;
        uint64_t matDoneCycle = 0;
        uint64_t vecDoneCycle = 0;
        uint64_t readyCycle = 0;
        uint32_t kStepCount = 0;
        bool issued = false;
        bool matDoneSent = false;
        bool vecDoneSent = false;
        bool retired = false;
    };

    struct WorkerWindowTxnState {
        uint64_t txnId = 0;
        WcpWindowTransaction txn;
        std::vector<WorkerTileRequestState> tiles;
    };

    struct RequestTraceState {
        uint64_t submitRecvCycle = 0;
        uint64_t issueCycle = 0;
        uint64_t pendingClearCycle = 0;
        uint64_t doneAckCycle = 0;
        uint8_t requestSlot = 0;
        uint8_t workerSlot = 0;
        uint16_t targetNode = 0;
        bool submitSeen = false;
        bool issueSeen = false;
        bool pendingClearSeen = false;
        bool doneSeen = false;
    };

    RequestSchedulerRole parseRole(const std::string& role) const;
    static uint32_t parseU32Param(SST::Params& params, const std::string& key, uint32_t def);
    static int32_t parseI32Param(SST::Params& params, const std::string& key, int32_t def);
    static uint64_t parseU64Param(SST::Params& params, const std::string& key, uint64_t def);
    void configureLinks(SST::Params& params);
    void handleReq0(SST::Event* ev);
    void handleReq1(SST::Event* ev);
    void handleReq2(SST::Event* ev);
    void handleReq3(SST::Event* ev);
    void handleReq(SST::Event* ev, int slot);
    bool handleRecv(int vn);
    bool handleSendAvailable(int vn);
    bool tick(SST::Cycle_t cycle);
    void pollWorkerMailbox();
    void flushWorkerDirectSubmits();
    void flushWorkerDirectDones();
    void flushDoneForWindowTransaction(WorkerWindowTxnState& state);
    void enqueueWindowTiles(WorkerWindowTxnState& state);
    void refillWorkerWindows();
    void tryIssue();
    bool issueTransfer(const PendingTransfer& req);
    uint32_t computeWorkerResidentK() const;
    uint32_t computeWorkerCreditCap() const;
    uint32_t computeWorkerSoftIssueCap() const;
    uint64_t composeWindowRequestId(uint8_t slot, uint16_t targetNode, uint64_t txnId,
                                    uint32_t tileIdx) const;
    uint32_t chunkBytesForTransfer(const PendingTransfer& req) const;
    size_t managerPendingCount() const;
    void traceSubmitAtManager(uint64_t requestId, uint8_t workerSlot, uint16_t targetNode);
    void traceIssueAtManager(uint64_t requestId);
    void traceDoneAtManager(uint64_t requestId, uint64_t pendingClearCycle);
    void emitManagerTraceSummary() const;
    void sampleManagerPressure();
    void emitManagerPressureSummary() const;
    static uint8_t decodeRequestSlot(uint64_t requestId);
    static uint64_t decodeSiblingRequestId(uint64_t requestId);
    uint32_t inferBytesForSlot(uint8_t slot, uint32_t mailboxBytes) const;
    uint64_t mailboxAddr(uint64_t off) const;
    uint64_t readMailbox(uint64_t off) const;
    void writeMailbox(uint64_t off, uint64_t value);

    RequestSchedulerRole role_;
    uint32_t coreId_;
    uint32_t groupId_;
    int32_t workerSlot_;
    uint32_t queueDepth_;
    uint32_t panelChunkBytes_;
    uint32_t workerCreditCap_;
    uint32_t workerSoftIssueCap_;
    uint32_t managerIssueBudgetPerTick_;
    uint32_t tileChunkQuantum_;
    uint32_t submitBatchSize_;
    uint32_t doneBatchSize_;
    uint32_t prefetchWindowDepth_;
    uint32_t localSlotCount_;
    uint32_t aReuseNTiles_;
    uint32_t bReuseMTiles_;
    uint32_t numMemoryNodes_;
    bool inferSubmitBytes_;
    bool traceEvents_;
    bool eventDrivenWorker_;
    bool groupRoundRobin_;
    bool workerNeedsService_;
    uint32_t slot0Bytes_;
    uint32_t slot1Bytes_;
    std::string ctrlLatency_;
    uint64_t gmBaseAddr_;
    uint64_t gmSize_;
    int verbose_;
    SST::Output output_;

    SST::Link* reqOut_;
    std::vector<SST::Link*> reqIn_;
    SST::Interfaces::SimpleNetwork* linkControl_;
    int networkId_;
    GlobalMemoryImplement* gm_;

    std::deque<PendingTransfer> workerSubmitQ_;
    std::vector<std::vector<std::deque<PendingTransfer>>> nodeWorkerQueues_;
    std::vector<uint32_t> workerCredits_;
    size_t nodeIssueCursor_;
    std::vector<size_t> nodeWorkerIssueCursor_;
    size_t groupWorkerIssueCursor_;
    std::vector<size_t> workerNodeIssueCursor_;
    bool submitSeen_;
    bool doneSeen_;
    uint32_t windowKtiles_;
    uint64_t nextWindowTxnId_;
    uint64_t lastTickCycle_;
    uint64_t issueBudgetCycle_;
    uint32_t issuedThisCycle_;
    std::unordered_map<uint64_t, WorkerWindowTxnState> workerWindowTxns_;
    std::unordered_map<uint64_t, RequestTraceState> requestTrace_;
    std::unordered_set<uint64_t> tracePairIssueCounted_;
    std::unordered_set<uint64_t> tracePairDoneCounted_;
    std::vector<uint64_t> traceSubmitToIssueMat_;
    std::vector<uint64_t> traceSubmitToIssueVec_;
    std::vector<uint64_t> traceIssueToPendingClearMat_;
    std::vector<uint64_t> traceIssueToPendingClearVec_;
    std::vector<uint64_t> tracePendingClearToDoneAckMat_;
    std::vector<uint64_t> tracePendingClearToDoneAckVec_;
    std::vector<uint64_t> traceIssueToDoneMat_;
    std::vector<uint64_t> traceIssueToDoneVec_;
    std::vector<uint64_t> traceSubmitToDoneMat_;
    std::vector<uint64_t> traceSubmitToDoneVec_;
    std::vector<uint64_t> tracePairIssueGap_;
    std::vector<uint64_t> tracePairDoneGap_;
    std::vector<uint64_t> pressurePendingQSamples_;
    std::vector<uint64_t> pressureWorkerUsedMaxSamples_;
    std::vector<uint64_t> pressureNodeUsedMaxSamples_;
    uint64_t pressureTicks_ = 0;
    uint64_t pressurePendingNonEmptyTicks_ = 0;
    uint64_t pressureNoIssueTicks_ = 0;
    uint64_t pressureIssuedRequests_ = 0;
    uint64_t pressureWorkerCreditBlocked_ = 0;
    uint64_t pressureNodeCreditBlocked_ = 0;
    uint64_t pressureIssuePaceBlocked_ = 0;
    uint64_t pressureNetworkSendBlocked_ = 0;
    uint64_t pressurePriorityPairAttempts_ = 0;
    uint64_t pressurePriorityPairIssued_ = 0;
};

} // namespace Golem
} // namespace SST

#endif
