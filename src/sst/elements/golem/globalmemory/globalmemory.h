#ifndef _GLOBAL_MEMORY_H
#define _GLOBAL_MEMORY_H

#include <sst/core/subcomponent.h>
#include <vector>
#include <string>
#include <cassert>
#include <unordered_map>
#include <functional>
#include <memory>
#include <cctype>
#include <deque>
#include <utility>

#include <sst/core/interfaces/simpleNetwork.h>
#include <sst/core/interfaces/stringEvent.h>
#include <map>
#include <unistd.h>
#include <climits>
#include <signal.h>
#include <sst/core/params.h>
#include <sst/core/timeLord.h>
#include <sst/core/output.h>
#include <sst/core/serialization/serializer.h>
#include <sst/core/serialization/serializable.h>
#include <sst/core/interfaces/stdMem.h>
#include <sst/elements/memHierarchy/memLinkBase.h>
#include <sst/elements/memHierarchy/util.h>
#include <sst/elements/golem/attention/attentionCluster.h>

namespace SST {
namespace Golem {

// DMA 相关的全局常量（默认值，运行时可通过参数覆盖）
static constexpr uint64_t MEM_NODE_SIZE = 0x04000000ULL;  // 256MB / 4 = 64MB per node
// Legacy fallback used when memoryRouters param is absent.
static constexpr uint8_t MEMORY_ROUTERS[4] = {16, 17, 18, 19};

using DmaCallback = std::function<void(bool)>;

enum class LocalMemoryClient : uint8_t {
    Dma,
    RoCC,
    SFU,
    WCP,
    Control,
};

enum class DmaRequestKind : uint8_t {
    Unknown,
    AttentionQuery,
    AttentionKv,
    AttentionKvPrefetch,
    AttentionOutput,
};

enum class DmaOperand : uint8_t {
    Unknown,
    AttentionK,
    AttentionV,
};

struct DmaConsumerMetadata {
    uint8_t valid = 0;
    uint64_t jobId = 0;
    uint32_t worker = 0;
    uint32_t consumerQueryTile = 0;
    uint32_t consumerKvTileIndex = 0;
    uint32_t targetQueryTile = 0;
    uint32_t targetKvTileIndex = 0;
    DmaOperand operand = DmaOperand::Unknown;

    void serialize_order(SST::Core::Serialization::serializer& ser) {
        ser & valid;
        ser & jobId;
        ser & worker;
        ser & consumerQueryTile;
        ser & consumerKvTileIndex;
        ser & targetQueryTile;
        ser & targetKvTileIndex;
        ser & operand;
    }
};

using LocalReadCallback =
    std::function<void(bool, uint64_t, const std::vector<uint8_t>&)>;
using LocalWriteCallback = std::function<void(bool, uint64_t)>;

class LocalMemoryAccessEvent : public SST::Event {
public:
    LocalMemoryAccessEvent() = default;
    explicit LocalMemoryAccessEvent(uint64_t requestId) : requestId_(requestId) {}

    uint64_t requestId() const { return requestId_; }

    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        ser & requestId_;
    }

    ImplementSerializable(SST::Golem::LocalMemoryAccessEvent);

private:
    uint64_t requestId_ = 0;
};

enum class ControlTransportMessageKind {
    MaxRequest,
    MaxResponse,
    SumRequest,
    SumResponse,
    TensorRowDispatch,
    TensorRowComplete,
    AttentionDispatch,
    AttentionComplete,
    AttentionManagerComplete,
};

struct ControlTransportMessage {
    ControlTransportMessageKind kind = ControlTransportMessageKind::MaxRequest;
    uint64_t jobId = 0;
    uint64_t tag = 0;
    uint32_t ownerCore = 0;
    uint32_t workerSlot = 0;
    uint32_t workerCore = 0;
    uint32_t row = 0;
    uint32_t expectedWorkers = 0;
    uint32_t expectedRows = 0;
    uint32_t expectedCols = 0;
    uint64_t sendCycle = 0;
    double value = 0.0;
    uint64_t inputAddr = 0;
    uint64_t outputAddr = 0;
    uint64_t nodeStrideBytes = 0;
    uint32_t dataNodeMask = 0;
    uint32_t rowsPerBand = 0;
    uint32_t headDim = 0;
    uint32_t queryTileRows = 0;
    uint32_t kvTileRows = 0;
    uint32_t flags = 0;
    uint64_t qAddr = 0;
    uint64_t kAddr = 0;
    uint64_t vAddr = 0;
    uint64_t oAddr = 0;
    uint32_t numQueryHeads = 1;
    uint32_t numKvHeads = 1;
    uint32_t kvHeadIndex = 0;
    uint32_t queryLength = 0;
    uint32_t groupQueryRowBegin = 0;

    void serialize_order(SST::Core::Serialization::serializer& ser) {
        ser & kind;
        ser & jobId;
        ser & tag;
        ser & ownerCore;
        ser & workerSlot;
        ser & workerCore;
        ser & row;
        ser & expectedWorkers;
        ser & expectedRows;
        ser & expectedCols;
        ser & sendCycle;
        ser & value;
        ser & inputAddr;
        ser & outputAddr;
        ser & nodeStrideBytes;
        ser & dataNodeMask;
        ser & rowsPerBand;
        ser & headDim;
        ser & queryTileRows;
        ser & kvTileRows;
        ser & flags;
        ser & qAddr;
        ser & kAddr;
        ser & vAddr;
        ser & oAddr;
        ser & numQueryHeads;
        ser & numKvHeads;
        ser & kvHeadIndex;
        ser & queryLength;
        ser & groupQueryRowBegin;
    }
};

class ControlTransportEvent : public SST::Event {
public:
    ControlTransportEvent() = default;
    explicit ControlTransportEvent(const ControlTransportMessage& message)
        : message_(message) {}

    const ControlTransportMessage& getMessage() const { return message_; }

    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        ser & message_;
    }

    ImplementSerializable(SST::Golem::ControlTransportEvent);

private:
    ControlTransportMessage message_;
};

class NetworkDataEvent : public Event {
public:
    // 枚举类型表示事件类型: READ=读请求, WRITE=写请求或读回复, DMA_WRITE=DMA写请求,
    // DMA_READ_COMPLETE=DMA读完成, DMA_WRITE_COMPLETE=DMA写完成
    enum Type { READ, WRITE, DMA_WRITE, DMA_READ_COMPLETE, DMA_WRITE_COMPLETE,
                DMA_CONSUMER_PROGRESS };

    // 默认构造函数 (序列化需要)
    NetworkDataEvent() : Event(), type(READ), addr(0), length(0), data(), returnAddr(0), returnEndpoint(-1), completionFlagAddr(0), completionValue(0), requestId(0), dmaRequestKind(DmaRequestKind::Unknown), dmaConsumer(), ownerGenerationToken(0) {}

    // 带参数构造函数
    NetworkDataEvent(Type type, uint64_t addr, size_t length, const std::vector<uint8_t>& data)
        : NetworkDataEvent(type, addr, length, data, addr) {}

    NetworkDataEvent(Type type, uint64_t addr, size_t length, const std::vector<uint8_t>& data, uint64_t returnAddr)
        : Event(), type(type), addr(addr), length(length), data(data), returnAddr(returnAddr),
          returnEndpoint(-1), completionFlagAddr(0), completionValue(0), requestId(0),
          dmaRequestKind(DmaRequestKind::Unknown), dmaConsumer(), ownerGenerationToken(0) {}

    NetworkDataEvent(Type type, uint64_t addr, size_t length, const std::vector<uint8_t>& data,
                     uint64_t returnAddr, int returnEndpoint, uint64_t completionFlagAddr, uint64_t completionValue,
                     uint64_t requestId = 0, DmaRequestKind dmaRequestKind = DmaRequestKind::Unknown,
                     const DmaConsumerMetadata& dmaConsumer = DmaConsumerMetadata())
        : Event(), type(type), addr(addr), length(length), data(data), returnAddr(returnAddr),
          returnEndpoint(returnEndpoint), completionFlagAddr(completionFlagAddr), completionValue(completionValue), requestId(requestId), dmaRequestKind(dmaRequestKind), dmaConsumer(dmaConsumer), ownerGenerationToken(0) {}

    // 获取事件类型
    Type getType() const { return type; }
    // 获取远端全局内存地址
    uint64_t getAddr() const { return addr; }
    // 获取数据长度
    size_t getLength() const { return length; }
    // 获取数据内容 (READ请求时数据可能为空)
    const std::vector<uint8_t>& getData() const { return data; }
    // 获取读请求返回后需要写入的目标地址（缺省为 addr 本身）
    uint64_t getReturnAddr() const { return returnAddr; }
    int getReturnEndpoint() const { return returnEndpoint; }
    uint64_t getCompletionFlagAddr() const { return completionFlagAddr; }
    uint64_t getCompletionValue() const { return completionValue; }
    uint64_t getRequestId() const { return requestId; }
    DmaRequestKind getDmaRequestKind() const { return dmaRequestKind; }
    const DmaConsumerMetadata& getDmaConsumerMetadata() const { return dmaConsumer; }
    uint64_t getOwnerGenerationToken() const { return ownerGenerationToken; }
    void setOwnerGenerationToken(uint64_t token) { ownerGenerationToken = token; }
    // 序列化函数: 序列化所有字段以支持跨节点传输
    void serialize_order(SST::Core::Serialization::serializer &ser) override {
        Event::serialize_order(ser);
        ser & type;
        ser & addr;
        ser & length;
        ser & data;
        ser & returnAddr;
        ser & returnEndpoint;
        ser & completionFlagAddr;
        ser & completionValue;
        ser & requestId;
        ser & dmaRequestKind;
        ser & dmaConsumer;
        ser & ownerGenerationToken;
    }

    ImplementSerializable(SST::Golem::NetworkDataEvent);

private:
    Type type;                      // 事件类型 (READ 或 WRITE)
    uint64_t addr;                  // 远端全局内存地址
    size_t length;                  // 数据长度
    std::vector<uint8_t> data;      // 数据内容 (WRITE请求包含写入数据; READ请求则为空)
    uint64_t returnAddr;            // READ 请求回复时应写回的地址
    int returnEndpoint;             // optional override endpoint for response routing
    uint64_t completionFlagAddr;    // optional direct-completion flag address on receiver
    uint64_t completionValue;       // optional direct-completion flag value on receiver
    uint64_t requestId;             // optional scheduler transaction identifier
    DmaRequestKind dmaRequestKind;  // semantic tag for DMA response trace attribution
    DmaConsumerMetadata dmaConsumer;
    uint64_t ownerGenerationToken;
};


/**
 * GlobalMemory API 接口类：声明读/写操作的纯虚函数 
 * 并通过 SST_ELI_REGISTER_SUBCOMPONENT_API 注册为子组件接口
 */
class GlobalMemoryAPI : public SST::SubComponent {
public:
    SST_ELI_REGISTER_SUBCOMPONENT_API(SST::Golem::GlobalMemoryAPI)

    GlobalMemoryAPI(ComponentId_t id, Params& params) : SubComponent(id) {}
    virtual ~GlobalMemoryAPI() {}

    // 定义虚函数
    using DmaCallback = ::SST::Golem::DmaCallback;
    using LocalReadCallback = ::SST::Golem::LocalReadCallback;
    using LocalWriteCallback = ::SST::Golem::LocalWriteCallback;
    using ControlMessageHandler = std::function<void(const ControlTransportMessage&)>;
    virtual void wr_to_globalmem(uint64_t wr_addr, size_t length, const std::vector<uint8_t>& wr_data) = 0;
    virtual void rd_from_globalmem(uint64_t rd_addr, size_t length, std::vector<uint8_t>& rd_data) = 0;
    virtual bool localReadAsync(uint64_t addr, size_t length, LocalMemoryClient client,
                                uint64_t tag, LocalReadCallback cb) = 0;
    virtual bool localWriteAsync(uint64_t addr, const std::vector<uint8_t>& data,
                                 LocalMemoryClient client, uint64_t tag,
                                 LocalWriteCallback cb) = 0;
    virtual size_t localMaxRequestBytes() const = 0;
    virtual void wr_to_network(uint64_t wr_addr, size_t length, std::vector<uint8_t>& wr_data) = 0;
    virtual void rd_to_network(uint64_t rd_addr, size_t length, uint64_t returnAddr) = 0;
    virtual void setBaseAddr(uint64_t addr) = 0;
    virtual uint64_t getBaseAddr() const = 0;
    virtual uint64_t getSize() const = 0;
    virtual void dma_write_to_host(uint64_t dst_pa, size_t length, const std::vector<uint8_t>& data, DmaCallback cb,
                                   DmaRequestKind kind = DmaRequestKind::Unknown) = 0;
    virtual void dma_write_from_globalmem_to_host(uint64_t gm_src_addr,
                                                   uint64_t dst_pa,
                                                   size_t length,
                                                   DmaCallback cb,
                                                   DmaRequestKind kind = DmaRequestKind::Unknown) = 0;
    virtual void dma_read_from_host_to_globalmem(uint64_t src_pa, size_t length, uint64_t gm_dst_addr, DmaCallback cb,
                                                 DmaRequestKind kind = DmaRequestKind::Unknown,
                                                 const DmaConsumerMetadata& consumer = DmaConsumerMetadata()) = 0;
    virtual void dma_update_consumer_progress(uint64_t firstHostAddr,
                                              uint64_t secondHostAddr,
                                              const DmaConsumerMetadata& consumer) = 0;
    virtual uint64_t dma_write_to_host_async(uint64_t dst_pa, size_t length,
                                             const std::vector<uint8_t>& data) = 0;
    virtual bool dma_completion_done(uint64_t token) const = 0;
    virtual void dma_completion_retire(uint64_t token) = 0;
    virtual bool controlNetworkAvailable() const = 0;
    virtual bool sendControlMessage(uint32_t destinationCore,
                                      const ControlTransportMessage& message) = 0;
    virtual void setControlMessageHandler(ControlMessageHandler handler) = 0;
    virtual bool beginAttentionGeneration(uint64_t generation) {
        return generation != 0;
    }
    virtual bool attentionGenerationDrained(uint64_t) const { return true; }
    virtual bool retireAttentionGeneration(uint64_t generation) {
        return generation != 0;
    }
    virtual uint32_t cancelAttentionGeneration(uint64_t) { return 0; }
};


/**
 * GlobalMemory 实现类：继承 GlobalMemoryAPI 并实现具体读写逻辑。
 * 使用 SST_ELI_REGISTER_SUBCOMPONENT 注册该实现，指定接口类为 GlobalMemoryAPI
 */
class GlobalMemoryImplement : public GlobalMemoryAPI {
public:
    SST_ELI_REGISTER_SUBCOMPONENT(
        GlobalMemoryImplement,                      // C++ 类名
        "golem", "GlobalMemory",                    // 元素库名与子组件名称
        SST_ELI_ELEMENT_VERSION(1,0,0),
        "Local global memory storage for Golem",    // 描述信息
        SST::Golem::GlobalMemoryAPI                 // 接口类全名
    )

    SST_ELI_DOCUMENT_PORTS(
        {"rtr", "Port that hooks up to router.", { "merlin.RtrEvent", "merlin.credit_event" } }
    )

    SST_ELI_DOCUMENT_PARAMS({
        {"baseAddr", "起始物理地址", "0"},
        {"size",     "内存字节容量", "0"},
        {"src_id",   "Logical core index for this GlobalMemory instance."},
        {"num_vns",  "Number of requested virtual networks."},
        {"dma_write_vn", "Virtual network used by DMA writes. Defaults to VN2 when available.", "2"},
        {"network_vn_priority_order", "Comma-separated VN injection priority order. Empty preserves round-robin.", ""},
        {"network_vn_starvation_vn", "VN allowed to bypass injection priority after a bounded wait.", "-1"},
        {"network_vn_max_starvation_cycles", "Maximum injection queue age before starvation bypass. 0 disables bypass.", "0"},
        {"verbose",  "GlobalMemory verbose level", "1"},
        {"dump_data", "Enable detailed data hex dump logs", "0"},
        {"link_bw",  "Bandwidth of the router link (e.g., \"1GB/s\")."},
        {"globalMemTransLatency", "Latency per bytes transferred in global memory operation", "5ns"},
        {"local_access_clock", "Clock used by the modeled local-memory scheduler", "1GHz"},
        {"local_access_base_latency_cycles", "Base cycles for an accepted local-memory request", "1"},
        {"local_access_bytes_per_cycle", "Transfer bandwidth per local-memory port", "64"},
        {"local_access_read_ports", "Number of modeled local-memory read ports", "1"},
        {"local_access_write_ports", "Number of modeled local-memory write ports", "1"},
        {"local_access_queue_depth", "Maximum queued plus in-flight local-memory requests", "32"},
        {"local_access_max_request_bytes", "Maximum bytes accepted by one local-memory request", "4096"},
        {"identityWindowBase", "Base address of Identity Window for DMA access to main memory", "0x04000000"},
        {"dma_read_retry_ticks", "Retry timeout ticks per DMA READ chunk (in selfLink ticks)", "96"},
        {"dma_read_max_retries", "Maximum retry attempts per DMA READ chunk", "8"},
        {"dma_read_max_inflight", "Maximum in-flight DMA READ chunks per core", "8"},
        {"dma_burst_bytes", "DMA chunk size in bytes for read/write burst splitting", "64"},
        {"dma_retry_tick_cpu_cycles", "CPU cycles represented by one DMA retry tick", "1"},
        {"control_vn", "Virtual network for control and collective messages (defaults to request VN).", ""},
        {"reduction_vn", "Deprecated alias for control_vn.", ""},
        {"memoryRouters", "Comma-separated memory router IDs for DMA fallback routing (e.g., \"24,0,1,2,3\")", ""}
    })

    SST_ELI_DOCUMENT_STATISTICS(
        {"gmem_control_send_immediate", "Control messages sent immediately", "messages", 1},
        {"gmem_control_send_queued", "Control messages queued for later send", "messages", 1},
        {"gmem_control_send_rejected", "Control messages rejected before transport ownership", "messages", 1},
        {"gmem_control_received", "Control messages received from the network", "messages", 1},
        {"attention_cluster_local_read_busy_union_ticks", "Union of Local-GM read-port intervals while cluster mode is enabled", "ticks", 1},
        {"attention_cluster_local_read_busy_span_ticks", "Span of Local-GM read-port intervals while cluster mode is enabled", "ticks", 1},
        {"attention_cluster_local_read_idle_gap_ticks", "Idle gaps within the cluster Local-GM read span", "ticks", 1},
        {"attention_cluster_local_read_max_concurrency", "Maximum concurrent cluster Local-GM reads", "requests", 1},
        {"attention_cluster_local_write_busy_union_ticks", "Union of Local-GM write-port intervals while cluster mode is enabled", "ticks", 1},
        {"attention_cluster_local_write_busy_span_ticks", "Span of Local-GM write-port intervals while cluster mode is enabled", "ticks", 1},
        {"attention_cluster_local_write_idle_gap_ticks", "Idle gaps within the cluster Local-GM write span", "ticks", 1},
        {"attention_cluster_local_write_max_concurrency", "Maximum concurrent cluster Local-GM writes", "requests", 1})

    SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS(
        {"networkIF", "Network interface (SimpleNetwork)", "SST::Interfaces::SimpleNetwork"},
        {"dma_iface", "StandardMem interface for DMA-like host memory access (identity window)", "SST::Interfaces::StandardMem"}
    )

    GlobalMemoryImplement(ComponentId_t id, Params& params);
    ~GlobalMemoryImplement();

    // 重写 SST 组件生命周期方法
    void init(unsigned int phase) override;
    void setup() override;
    void complete(unsigned int phase) override;
    void finish() override;

    // 将数据写入本地 GlobalMemory 存储
    void wr_to_globalmem(uint64_t wr_addr, size_t length, const std::vector<uint8_t>& wr_data) override;
    // 从本地 GlobalMemory 读取数据
    void rd_from_globalmem(uint64_t rd_addr, size_t length, std::vector<uint8_t>& rd_data) override;
    bool localReadAsync(uint64_t addr, size_t length, LocalMemoryClient client,
                        uint64_t tag, LocalReadCallback cb) override;
    bool localWriteAsync(uint64_t addr, const std::vector<uint8_t>& data,
                         LocalMemoryClient client, uint64_t tag,
                         LocalWriteCallback cb) override;
    size_t localMaxRequestBytes() const override { return localAccessMaxRequestBytes_; }
    // 发起对远端 GlobalMemory 节点的写请求
    void wr_to_network(uint64_t wr_addr, size_t length, std::vector<uint8_t>& wr_data) override;
    // 发起对远端 GlobalMemory 节点的读请求
    void rd_to_network(uint64_t rd_addr, size_t length, uint64_t returnAddr) override;

    // 设置全局内存基地址
    void setBaseAddr(uint64_t addr) override;
    // 获取全局内存基地址
    uint64_t getBaseAddr() const override;
    // 获取全局内存容量大小
    uint64_t getSize() const override;

    void dma_write_to_host(uint64_t dst_pa, size_t length, const std::vector<uint8_t>& data, DmaCallback cb,
                           DmaRequestKind kind = DmaRequestKind::Unknown) override;
    void dma_write_from_globalmem_to_host(uint64_t gm_src_addr,
                                           uint64_t dst_pa,
                                           size_t length,
                                           DmaCallback cb,
                                           DmaRequestKind kind = DmaRequestKind::Unknown) override;
    void dma_read_from_host_to_globalmem(uint64_t src_pa, size_t length, uint64_t gm_dst_addr, DmaCallback cb,
                                         DmaRequestKind kind = DmaRequestKind::Unknown,
                                         const DmaConsumerMetadata& consumer = DmaConsumerMetadata()) override;
    void dma_update_consumer_progress(uint64_t firstHostAddr, uint64_t secondHostAddr,
                                      const DmaConsumerMetadata& consumer) override;
    bool controlNetworkAvailable() const override;
    bool sendControlMessage(uint32_t destinationCore,
                              const ControlTransportMessage& message) override;
    void setControlMessageHandler(ControlMessageHandler handler) override;
    bool beginAttentionGeneration(uint64_t generation) override;
    bool attentionGenerationDrained(uint64_t generation) const override;
    bool retireAttentionGeneration(uint64_t generation) override;
    uint32_t cancelAttentionGeneration(uint64_t generation) override;
    uint64_t dma_write_to_host_async(uint64_t dst_pa, size_t length,
                                     const std::vector<uint8_t>& data) override;
    uint64_t dma_read_from_host_to_globalmem_async(uint64_t src_pa, size_t length, uint64_t gm_dst_addr,
                                                  DmaRequestKind kind = DmaRequestKind::Unknown);
    bool dma_completion_done(uint64_t token) const override;
    void dma_completion_retire(uint64_t token) override;

    uint64_t ctrlReadLocalU64(uint64_t addr) const;
    void ctrlWriteLocalU64(uint64_t addr, uint64_t value);
    int ctrlLookupMemNicEndpointId(uint64_t phys_addr);
    uint8_t ctrlLookupDmaTargetRouter(uint64_t phys_addr);
    int ctrlResolveEndpointForAddress(uint64_t addr);
    uint64_t ctrlGetReadFlagAddr(uint8_t readSlot) const;
    void ctrlRegisterPendingReadRequest(uint64_t requestId, uint64_t gmDstAddr,
                                        uint64_t completionFlagAddr, uint64_t completionValue,
                                        size_t totalLength = 0);
    bool ctrlIsReadRequestPending(uint64_t requestId) const;
    void ctrlSetReadCompletionCallback(std::function<void(uint64_t)> callback);
    uint32_t getRequestVn() const { return request_vn; }

    SST::Output* output;

    // 网络接收回调函数：处理收到的远端请求
    bool handle_receives(int vn);
    bool handle_send_available(int vn);

    virtual SimTime_t getLatency(uint32_t arrayID);
    virtual void handleSelfEvent(Event* ev);
    void handleLocalAccessEvent(Event* ev);
    //globalmemory传递数据的延迟参数
    SST::Link* selfLink = nullptr;
    UnitAlgebra globalMemTransLatency;
    TimeConverter* latencyTC = nullptr;
    SST::Event::HandlerBase* tileHandler = nullptr;

private:
    class StdMemHandlers : public SST::Interfaces::StandardMem::RequestHandler {
    public:
        StdMemHandlers(GlobalMemoryImplement* parent, SST::Output* output)
            : SST::Interfaces::StandardMem::RequestHandler(output), parent(parent) {}

        void handle(SST::Interfaces::StandardMem::ReadResp* ev) override;
        void handle(SST::Interfaces::StandardMem::WriteResp* ev) override;

    private:
        GlobalMemoryImplement* parent;
    };

    struct PendingDmaOp {
        enum Kind { READ_TO_GM, WRITE_TO_HOST } kind;
        uint64_t request_id = 0;
        uint64_t host_addr = 0;
        uint64_t gm_dst_addr = 0;
        size_t length = 0;
        DmaRequestKind dmaRequestKind = DmaRequestKind::Unknown;
        DmaConsumerMetadata dmaConsumer;
        DmaCallback cb;
        struct DmaContext {
            uint64_t host_base = 0;
            uint64_t gm_base = 0;
            size_t total_len = 0;
            size_t burst_size = 0;
            size_t remaining = 0;
            bool ok = true;
            DmaCallback cb;
            bool completion_enabled = false;
            uint64_t completion_flag_addr = 0;
            uint64_t completion_value = 0;
            uint64_t completion_token = 0;
        };
        std::shared_ptr<DmaContext> ctx;
        bool completion_enabled = false;
        uint64_t completion_flag_addr = 0;
        uint64_t completion_value = 0;
        uint64_t completion_token = 0;
        uint32_t retry_ticks_left = 0;
        uint32_t retry_attempts = 0;
        bool request_issued = false;
        bool request_sent = false;
        bool first_send_seen = false;
        uint64_t first_send_tick = 0;
        uint64_t last_send_tick = 0;
        uint64_t first_send_cycle = 0;
        uint64_t last_send_cycle = 0;
        uint64_t issue_cycle = 0;
        uint64_t ownerGeneration = 0;
        uint64_t ownerGenerationToken = 0;
    };

    struct PendingReadReply {
        uint64_t completion_flag_addr = 0;
        uint64_t completion_value = 0;
    };

    struct PendingReadRequest {
        uint64_t gm_dst_addr = 0;
        uint64_t completion_flag_addr = 0;
        uint64_t completion_value = 0;
        size_t total_len = 0;
        size_t received_len = 0;
        uint64_t submit_cycle = 0;
    };

    struct PendingLocalAccess {
        enum class Kind : uint8_t { Read, Write };
        Kind kind = Kind::Read;
        uint64_t requestId = 0;
        uint64_t addr = 0;
        size_t length = 0;
        LocalMemoryClient client = LocalMemoryClient::Control;
        uint64_t tag = 0;
        uint64_t submitCycle = 0;
        uint64_t ownerGeneration = 0;
        bool issued = false;
        bool cancelled = false;
        std::vector<uint8_t> writeData;
        LocalReadCallback readCallback;
        LocalWriteCallback writeCallback;
    };

    struct PendingDmaLanding {
        PendingDmaOp op;
        std::vector<uint8_t> data;
        size_t offset = 0;
        bool chunkInFlight = false;
    };

    uint64_t baseAddr = 0;
    uint64_t size = 0;
    uint64_t globalBase = 0;
    std::vector<uint8_t> storage;                 // 本地内存存储区域

    int num_vns;
    int core_id;
    int network_id = -1;
    uint64_t identityWindowBase = 0x04000000ULL;  // Identity Window base address for DMA
    uint64_t memNodeSize = MEM_NODE_SIZE;         // DMA 物理地址到节点的分段大小
    std::vector<uint8_t> memoryRouters;           // Fallback router table indexed by memory node id
    std::map<std::string, int> IDMap;             // 端点名称到ID的映射表
    std::unordered_map<int, int> coreEndpointMap; // 核心索引到网络端点 ID 的映射
    SST::Interfaces::SimpleNetwork* link_control; // SimpleNetwork 网络接口 (RDMA + DMA 共用)
    bool init_broadcast_done = false;             // 是否已经完成名称广播

    // 移除独立的 DMA NIC，复用 link_control
    // int dma_nic_vc = 0;  // DMA 使用的虚拟通道

    SST::Interfaces::StandardMem* dma_iface = nullptr;
    StdMemHandlers* dma_handlers = nullptr;
    std::unordered_map<uint64_t, PendingDmaOp> dma_pending;
    std::unordered_map<uint64_t, bool> dma_completion_tokens;
    std::unordered_map<uint64_t, PendingReadReply> read_pending;
    std::unordered_map<uint64_t, PendingReadRequest> request_pending;
    std::function<void(uint64_t)> ctrl_read_completion_callback;
    uint64_t next_dma_completion_token = 1;
    uint64_t next_dma_request_id = 1;

    bool localAccessValid(uint64_t addr, size_t length) const;
    void tryIssueLocalAccesses();
    void issueLocalAccess(uint64_t requestId);
    uint64_t localAccessDelayCycles(size_t length) const;
    void completeDmaReadToLocalMemory(PendingDmaOp op,
                                      const std::vector<uint8_t>& data);
    void issueNextDmaLandingChunk(uint64_t landingTag);
    void scheduleDmaLandingRetry();
    void retryDmaLandingChunks();
    void finishDmaReadLanding(PendingDmaOp& op, bool ok);

    SST::Link* localAccessSelfLink_ = nullptr;
    TimeConverter* localAccessTC_ = nullptr;
    uint32_t localAccessBaseLatencyCycles_ = 1;
    uint32_t localAccessBytesPerCycle_ = 64;
    uint32_t localAccessReadPorts_ = 1;
    uint32_t localAccessWritePorts_ = 1;
    uint32_t localAccessQueueDepth_ = 32;
    uint32_t localAccessMaxRequestBytes_ = 4096;
    uint32_t localReadsInFlight_ = 0;
    uint32_t localWritesInFlight_ = 0;
    uint64_t nextLocalAccessRequestId_ = 1;
    uint64_t nextLocalDmaTag_ = 1;
    uint64_t localAccessQueueHighWater_ = 0;
    uint64_t localReadRequests_ = 0;
    uint64_t localWriteRequests_ = 0;
    uint64_t localReadBytes_ = 0;
    uint64_t localWriteBytes_ = 0;
    uint64_t localQueueRejected_ = 0;
    uint64_t localReadQueueCycles_ = 0;
    uint64_t localWriteQueueCycles_ = 0;
    uint64_t dmaLandingBackpressureRetries_ = 0;
    bool dmaLandingRetryScheduled_ = false;
    bool attentionClusterEnable_ = false;
    AttentionGenerationFence attentionGenerationFence_;
    BusyIntervalUnion attentionClusterLocalReadIntervals_;
    BusyIntervalUnion attentionClusterLocalWriteIntervals_;
    std::deque<uint64_t> localReadQueue_;
    std::deque<uint64_t> localWriteQueue_;
    std::unordered_map<uint64_t, PendingLocalAccess> localAccessPending_;
    std::unordered_map<uint64_t, PendingDmaLanding> pendingDmaLandings_;

    void handleDmaMemEvent(SST::Interfaces::StandardMem::Request* req);
    void notifyCtrlReadComplete(uint64_t requestId);

    void registerEndpoint(const std::string& name, int endpointId);
    int extractCoreId(const std::string& name) const;
    int resolveEndpointForAddress(uint64_t addr);

    // DMA 网络辅助函数
    uint8_t getDmaTargetNode(uint64_t phys_addr);
    uint8_t getDmaTargetRouter(uint64_t phys_addr);
    int getMemNicEndpointId(uint64_t phys_addr);
    void sendDmaRequest(SST::Interfaces::SimpleNetwork::Request* req);
    bool handleDmaReceives(int vn);
    bool try_send_or_queue(SST::Interfaces::SimpleNetwork::Request* req, const char* context);
    bool flush_send_retry_queue();
    void schedule_dma_retry_event();
    void process_dma_read_retries();
    bool mark_dma_read_sent(SST::Interfaces::SimpleNetwork::Request* req);
    void track_send_queue_stats(SST::Interfaces::SimpleNetwork::Request* req, bool immediate_send);
    void issue_dma_read_chunk(PendingDmaOp& op, const char* reason);
    size_t count_issued_dma_reads() const;
    void issue_pending_dma_read_window();
    uint64_t allocate_dma_completion_token();
    void dma_write_to_host_impl(uint64_t dst_pa, size_t length, const std::vector<uint8_t>& data, DmaCallback cb,
                                uint64_t completion_token, DmaRequestKind kind = DmaRequestKind::Unknown);
    void dma_read_from_host_to_globalmem_impl(uint64_t src_pa, size_t length, uint64_t gm_dst_addr, DmaCallback cb,
                                              uint64_t completion_token, DmaRequestKind kind,
                                              const DmaConsumerMetadata& consumer = DmaConsumerMetadata());

    uint64_t read_u64_from_storage(uint64_t addr) const;
    void write_u64_to_storage(uint64_t addr, uint64_t value);
    void get_dma_flag_addrs(bool is_write, uint8_t read_slot, uint64_t& seq_addr, uint64_t& flag_addr) const;

private:
    // DMA 接收回调
    bool (*dma_notify_receive)(int);

    std::deque<SST::Interfaces::SimpleNetwork::Request*> send_retry_queue;
    uint32_t dma_read_retry_ticks = 96;
    uint32_t dma_read_max_retries = 8;
    uint32_t dma_read_max_inflight = 8;
    uint32_t dma_burst_bytes = 64;
    uint64_t dma_retry_tick_cpu_cycles = 1;
    bool gm_dump_data = false;
    bool dma_trace = false;
    bool dma_retry_event_scheduled = false;
    uint32_t request_vn = 0;
    uint32_t response_vn = 0;
    uint32_t control_vn = 0;
    uint32_t write_vn = 0;
    uint64_t dma_retry_tick_counter = 0;
    uint64_t dma_read_send_immediate_count = 0;
    uint64_t dma_read_send_queued_count = 0;
    uint64_t dma_read_send_flushed_count = 0;
    uint64_t dma_read_issue_count = 0;
    uint64_t dma_write_issue_count = 0;
    uint64_t dma_read_bytes_total = 0;
    uint64_t dma_write_bytes_total = 0;
    uint64_t dma_read_timeout_retry_count = 0;
    uint64_t dma_write_timeout_retry_count = 0;
    uint64_t dma_read_timeout_exhausted_count = 0;
    uint64_t dma_read_completion_count = 0;
    uint64_t dma_write_completion_count = 0;
    uint64_t dma_write_rtt_samples = 0;
    uint64_t dma_write_rtt_cycles_sum = 0;
    uint64_t dma_write_rtt_cycles_max = 0;
    uint64_t dma_write_first_issue_cycle = 0;
    uint64_t dma_write_last_issue_cycle = 0;
    uint64_t dma_write_last_complete_cycle = 0;
    uint64_t dma_read_completion_no_pending_count = 0;
    uint64_t dma_wait_count = 0;
    uint64_t dma_read_rtt_samples = 0;
    uint64_t dma_read_rtt_ticks_sum = 0;
    uint64_t dma_read_rtt_ticks_max = 0;
    uint64_t dma_read_e2e_rtt_samples = 0;
    uint64_t dma_read_e2e_rtt_ticks_sum = 0;
    uint64_t dma_read_e2e_rtt_ticks_max = 0;
    uint64_t dma_read_strict_rtt_samples = 0;
    uint64_t dma_read_strict_rtt_cycles_sum = 0;
    uint64_t dma_read_strict_rtt_cycles_max = 0;
    uint64_t dma_read_strict_e2e_rtt_samples = 0;
    uint64_t dma_read_strict_e2e_rtt_cycles_sum = 0;
    uint64_t dma_read_strict_e2e_rtt_cycles_max = 0;
    uint64_t request_read_submit_ready_samples = 0;
    uint64_t request_read_submit_ready_cycles_sum = 0;
    uint64_t request_read_submit_ready_cycles_max = 0;
    uint64_t control_send_immediate_count = 0;
    uint64_t control_send_queued_count = 0;
    uint64_t control_receive_count = 0;
    Statistic<uint64_t>* statControlSendImmediate_ = nullptr;
    Statistic<uint64_t>* statControlSendQueued_ = nullptr;
    Statistic<uint64_t>* statControlSendRejected_ = nullptr;
    Statistic<uint64_t>* statControlReceived_ = nullptr;
    Statistic<uint64_t>* statAttentionClusterLocalReadBusyUnion_ = nullptr;
    Statistic<uint64_t>* statAttentionClusterLocalReadBusySpan_ = nullptr;
    Statistic<uint64_t>* statAttentionClusterLocalReadIdleGap_ = nullptr;
    Statistic<uint64_t>* statAttentionClusterLocalReadMaxConcurrency_ = nullptr;
    Statistic<uint64_t>* statAttentionClusterLocalWriteBusyUnion_ = nullptr;
    Statistic<uint64_t>* statAttentionClusterLocalWriteBusySpan_ = nullptr;
    Statistic<uint64_t>* statAttentionClusterLocalWriteIdleGap_ = nullptr;
    Statistic<uint64_t>* statAttentionClusterLocalWriteMaxConcurrency_ = nullptr;
    ControlMessageHandler control_message_handler;
    size_t send_retry_queue_max_depth = 0;
    std::unordered_map<SST::Interfaces::SimpleNetwork::Request*, uint64_t> dma_read_req_to_key;

};

/**
 * A minimal local-only GlobalMemory implementation that does not require
 * network connectivity. Used as a safe fallback when tests don't wire a
 * networked GlobalMemory instance.
 */
class GlobalMemoryLocal : public GlobalMemoryAPI {
public:
    SST_ELI_REGISTER_SUBCOMPONENT(
        GlobalMemoryLocal,
        "golem", "GlobalMemoryLocal",
        SST_ELI_ELEMENT_VERSION(1,0,0),
        "Local-only global memory (no network)",
        SST::Golem::GlobalMemoryAPI
    )

    SST_ELI_DOCUMENT_PARAMS({
        {"baseAddr", "起始物理地址", "0"},
        {"size",     "内存字节容量", "65536"},
        {"src_id",   "Logical core index for this GlobalMemory instance.", "0"},
        
    })

    GlobalMemoryLocal(ComponentId_t id, Params& params);
    ~GlobalMemoryLocal() override {}

    void init(unsigned int) override {}
    void setup() override {}
    void complete(unsigned int) override {}
    void finish() override {}

    void wr_to_globalmem(uint64_t wr_addr, size_t length, const std::vector<uint8_t>& wr_data) override;
    void rd_from_globalmem(uint64_t rd_addr, size_t length, std::vector<uint8_t>& rd_data) override;
    bool localReadAsync(uint64_t addr, size_t length, LocalMemoryClient client,
                        uint64_t tag, LocalReadCallback cb) override;
    bool localWriteAsync(uint64_t addr, const std::vector<uint8_t>& data,
                         LocalMemoryClient client, uint64_t tag,
                         LocalWriteCallback cb) override;
    size_t localMaxRequestBytes() const override { return size; }
    void wr_to_network(uint64_t wr_addr, size_t length, std::vector<uint8_t>& wr_data) override;
    void rd_to_network(uint64_t rd_addr, size_t length, uint64_t returnAddr) override;
    void setBaseAddr(uint64_t addr) override { baseAddr = addr; }
    uint64_t getBaseAddr() const override { return baseAddr; }
    uint64_t getSize() const override { return size; }
    void dma_write_to_host(uint64_t, size_t, const std::vector<uint8_t>&, DmaCallback,
                           DmaRequestKind = DmaRequestKind::Unknown) override;
    void dma_write_from_globalmem_to_host(uint64_t, uint64_t, size_t, DmaCallback,
                                          DmaRequestKind = DmaRequestKind::Unknown) override;
    void dma_read_from_host_to_globalmem(uint64_t, size_t, uint64_t, DmaCallback,
                                         DmaRequestKind = DmaRequestKind::Unknown,
                                         const DmaConsumerMetadata& = DmaConsumerMetadata()) override;
    void dma_update_consumer_progress(uint64_t, uint64_t,
                                      const DmaConsumerMetadata&) override {}
    bool controlNetworkAvailable() const override { return false; }
    bool sendControlMessage(uint32_t, const ControlTransportMessage&) override { return false; }
    void setControlMessageHandler(ControlMessageHandler) override {}
    uint64_t dma_write_to_host_async(uint64_t dst_pa, size_t length,
                                     const std::vector<uint8_t>& data) override;
    uint64_t dma_read_from_host_to_globalmem_async(uint64_t src_pa, size_t length, uint64_t gm_dst_addr,
                                                  DmaRequestKind kind = DmaRequestKind::Unknown);
    bool dma_completion_done(uint64_t token) const override;
    void dma_completion_retire(uint64_t token) override;



private:
    uint64_t baseAddr = 0;
    uint64_t size = 0;
    std::vector<uint8_t> storage;
    std::unordered_map<uint64_t, bool> dma_completion_tokens;
    uint64_t next_dma_completion_token = 1;
};

} // namespace Golem
} // namespace SST

#endif /* _GLOBAL_MEMORY_H */
