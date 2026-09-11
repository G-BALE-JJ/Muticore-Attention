#ifndef _H_GOLEM_GROUPCTRL_ENDPOINT
#define _H_GOLEM_GROUPCTRL_ENDPOINT

#include <deque>
#include <string>
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
          targetNode(0) {}

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
};

class GroupCtrlAPI : public SST::SubComponent {
public:
    SST_ELI_REGISTER_SUBCOMPONENT_API(SST::Golem::GroupCtrlAPI)

    GroupCtrlAPI(ComponentId_t id, SST::Params& params) : SST::SubComponent(id) {}
    virtual ~GroupCtrlAPI() = default;
    virtual void bindGlobalMemory(GlobalMemoryAPI* globalMemory) = 0;
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
        {"verbose", "Verbosity", "0"})

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
    uint64_t mailboxAddr(uint64_t off) const;
    uint64_t readMailbox(uint64_t off) const;
    void writeMailbox(uint64_t off, uint64_t value);
    uint32_t readMailboxU32(uint64_t off) const;
    void writeMailboxU32(uint64_t off, uint32_t value);
};

} // namespace Golem
} // namespace SST

#endif
