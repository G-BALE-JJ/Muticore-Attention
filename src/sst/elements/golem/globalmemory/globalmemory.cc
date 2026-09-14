#include "sst_config.h"
#include "globalmemory.h"

#include <inttypes.h>
#include <typeinfo>
#include <memory>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <algorithm>
#include <sstream>
#include <limits>
#include <new>

#include "sst/elements/memHierarchy/memNICBase.h"
#include "sst/elements/memHierarchy/memLinkBase.h"

using namespace SST::Golem;

namespace {

bool envFlagDefault(const char* name, bool defaultValue) {
    const char* raw = std::getenv(name);
    if (raw == nullptr) {
        return defaultValue;
    }
    std::string value(raw);
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value == "1" || value == "true" || value == "yes" || value == "on";
}

uint64_t ceilDivLocalAccess(uint64_t value, uint64_t divisor) {
    return (value + divisor - 1) / divisor;
}

std::vector<uint8_t> parseMemoryRoutersParam(const std::string& raw, SST::Output* output, const std::string& compName) {
    std::vector<uint8_t> routers;
    if (raw.empty()) {
        return routers;
    }

    std::stringstream ss(raw);
    std::string token;
    while (std::getline(ss, token, ',')) {
        const size_t first = token.find_first_not_of(" \t\n\r");
        if (first == std::string::npos) {
            continue;
        }
        const size_t last = token.find_last_not_of(" \t\n\r");
        const std::string trimmed = token.substr(first, last - first + 1);

        try {
            const unsigned long value = std::stoul(trimmed);
            if (value > std::numeric_limits<uint8_t>::max()) {
                output->fatal(CALL_INFO, -1,
                              "GlobalMemory '%s' invalid router id '%s' in memoryRouters='%s' (must be 0..255)\n",
                              compName.c_str(), trimmed.c_str(), raw.c_str());
            }
            routers.push_back(static_cast<uint8_t>(value));
        } catch (const std::exception&) {
            output->fatal(CALL_INFO, -1,
                          "GlobalMemory '%s' failed to parse router id '%s' in memoryRouters='%s'\n",
                          compName.c_str(), trimmed.c_str(), raw.c_str());
        }
    }

    return routers;
}

}

GlobalMemoryImplement::GlobalMemoryImplement(ComponentId_t id, Params& params)
    : GlobalMemoryAPI(id, params)
{
    int verbose_level = params.find<int>("verbose", 1);
    output = new SST::Output("[GlobalMemory @t]: ", verbose_level, 0xFFFFFFFF, SST::Output::STDOUT);

    baseAddr = params.find<uint64_t>("baseAddr", 0);
    size     = params.find<uint64_t>("size", 0);
    assert(size > 0 && "GlobalMemory size must be > 0");
    storage.resize(size, 0x00);
    globalMemTransLatency = params.find<UnitAlgebra>("globalMemTransLatency", "30ns");
    const std::string localAccessClock =
        params.find<std::string>("local_access_clock", "1GHz");
    localAccessBaseLatencyCycles_ =
        params.find<uint32_t>("local_access_base_latency_cycles", 1);
    localAccessBytesPerCycle_ = params.find<uint32_t>("local_access_bytes_per_cycle", 64);
    localAccessReadPorts_ = params.find<uint32_t>("local_access_read_ports", 1);
    localAccessWritePorts_ = params.find<uint32_t>("local_access_write_ports", 1);
    localAccessQueueDepth_ = params.find<uint32_t>("local_access_queue_depth", 32);
    localAccessMaxRequestBytes_ =
        params.find<uint32_t>("local_access_max_request_bytes", 4096);
    attentionClusterEnable_ =
        params.find<bool>("attention_cluster_enable", false);
    if (localAccessBaseLatencyCycles_ == 0 || localAccessBytesPerCycle_ == 0 ||
        localAccessReadPorts_ == 0 || localAccessWritePorts_ == 0 ||
        localAccessQueueDepth_ == 0 || localAccessMaxRequestBytes_ == 0) {
        output->fatal(CALL_INFO, -1,
                      "GlobalMemory local-access parameters must be positive\n");
    }
    dma_read_retry_ticks = params.find<uint32_t>("dma_read_retry_ticks", 96);
    dma_read_max_retries = params.find<uint32_t>("dma_read_max_retries", 8);
    dma_read_max_inflight = params.find<uint32_t>("dma_read_max_inflight", 8);
    dma_burst_bytes = params.find<uint32_t>("dma_burst_bytes", 64);
    dma_retry_tick_cpu_cycles = params.find<uint64_t>("dma_retry_tick_cpu_cycles", 1);
    gm_dump_data = params.find<int>("dump_data", 0) != 0;
    dma_trace = params.find<int>("dma_trace", envFlagDefault("GOLEM_DMA_TRACE", false) ? 1 : 0) != 0;
    if (dma_read_retry_ticks == 0) dma_read_retry_ticks = 1;
    if (dma_read_max_inflight == 0) dma_read_max_inflight = 1;
    if (dma_burst_bytes == 0) dma_burst_bytes = 64;
    if (dma_retry_tick_cpu_cycles == 0) dma_retry_tick_cpu_cycles = 1;
    core_id = params.find<int>("src_id", -1);
    if (core_id < 0) {
        output->fatal(CALL_INFO, -1, "core_id (param 'src_id') must be set and non-negative.\n");
    }

    uint64_t expectedMinBase = static_cast<uint64_t>(core_id) * size;
    if (baseAddr < expectedMinBase) {
        output->fatal(CALL_INFO, -1,
                      "GlobalMemory '%s' configured with baseAddr=0x%" PRIx64
                      " which is smaller than core_id(%d) * size(0x%" PRIx64 ").\n",
                      getName().c_str(), baseAddr, core_id, size);
    }

    globalBase = baseAddr - expectedMinBase;
    coreEndpointMap.reserve(8);

    // 加载网络接口子组件
    num_vns = params.find<int>("num_vns", -1);
    if (num_vns == -1) {
        output->fatal(CALL_INFO, -1, "num_vns must be set!\n");
    }
    // Keep read requests, read responses, and writes in independent queues when
    // three virtual networks are available.
    request_vn = 0;
    response_vn = (num_vns >= 2) ? 1 : 0;
    reduction_vn = params.find<uint32_t>("reduction_vn", request_vn);
    if (reduction_vn >= static_cast<uint32_t>(num_vns)) {
        output->fatal(CALL_INFO, -1,
                      "GlobalMemory '%s' reduction_vn=%u is outside num_vns=%d\n",
                      getName().c_str(), reduction_vn, num_vns);
    }
    statReductionSendImmediate_ = registerStatistic<uint64_t>("gmem_reduction_send_immediate");
    statReductionSendQueued_ = registerStatistic<uint64_t>("gmem_reduction_send_queued");
    statReductionSendRejected_ = registerStatistic<uint64_t>("gmem_reduction_send_rejected");
    statReductionReceived_ = registerStatistic<uint64_t>("gmem_reduction_received");
    statAttentionClusterLocalReadBusyUnion_ = registerStatistic<uint64_t>(
        "attention_cluster_local_read_busy_union_ticks");
    statAttentionClusterLocalReadBusySpan_ = registerStatistic<uint64_t>(
        "attention_cluster_local_read_busy_span_ticks");
    statAttentionClusterLocalReadIdleGap_ = registerStatistic<uint64_t>(
        "attention_cluster_local_read_idle_gap_ticks");
    statAttentionClusterLocalReadMaxConcurrency_ = registerStatistic<uint64_t>(
        "attention_cluster_local_read_max_concurrency");
    statAttentionClusterLocalWriteBusyUnion_ = registerStatistic<uint64_t>(
        "attention_cluster_local_write_busy_union_ticks");
    statAttentionClusterLocalWriteBusySpan_ = registerStatistic<uint64_t>(
        "attention_cluster_local_write_busy_span_ticks");
    statAttentionClusterLocalWriteIdleGap_ = registerStatistic<uint64_t>(
        "attention_cluster_local_write_idle_gap_ticks");
    statAttentionClusterLocalWriteMaxConcurrency_ = registerStatistic<uint64_t>(
        "attention_cluster_local_write_max_concurrency");
    write_vn = params.find<uint32_t>("dma_write_vn", (num_vns >= 3) ? 2u : request_vn);
    if (write_vn >= static_cast<uint32_t>(num_vns)) {
        output->fatal(CALL_INFO, -1,
                      "GlobalMemory '%s' dma_write_vn=%u must be smaller than num_vns=%d.\n",
                      getName().c_str(), write_vn, num_vns);
    }

    // 读取 Identity Window 基础地址
    identityWindowBase = params.find<uint64_t>("identityWindowBase", 0x04000000ULL);

    // 读取 DMA 节点大小（优先 memNodeSize，其次 physMemSize/numMemNodes）
    bool found = false;
    std::string memNodeSizeStr = params.find<std::string>("memNodeSize", "", found);
    if (found && !memNodeSizeStr.empty()) {
        UnitAlgebra nodeSizeUA(memNodeSizeStr);
        if (!nodeSizeUA.hasUnits("B")) {
            output->fatal(CALL_INFO, -1,
                          "GlobalMemory '%s' invalid memNodeSize '%s' (must be bytes with units, e.g., 64MiB)\n",
                          getName().c_str(), memNodeSizeStr.c_str());
        }
        memNodeSize = nodeSizeUA.getRoundedValue();
    } else {
        std::string physMemStr = params.find<std::string>("physMemSize", "", found);
        if (found && !physMemStr.empty()) {
            UnitAlgebra physUA(physMemStr);
            if (!physUA.hasUnits("B")) {
                output->fatal(CALL_INFO, -1,
                              "GlobalMemory '%s' invalid physMemSize '%s' (must be bytes with units)\n",
                              getName().c_str(), physMemStr.c_str());
            }
            uint32_t numNodes = params.find<uint32_t>("numMemNodes", 0);
            if (numNodes == 0) {
                output->fatal(CALL_INFO, -1,
                              "GlobalMemory '%s' physMemSize provided but numMemNodes is 0\n",
                              getName().c_str());
            }
            memNodeSize = physUA.getRoundedValue() / numNodes;
        } else {
            memNodeSize = params.find<uint64_t>("memNodeSize", MEM_NODE_SIZE);
        }
    }
    if (memNodeSize == 0) {
        output->fatal(CALL_INFO, -1,
                      "GlobalMemory '%s' memNodeSize must be > 0\n",
                      getName().c_str());
    }

    memoryRouters = parseMemoryRoutersParam(
        params.find<std::string>("memoryRouters", ""), output, getName());
    if (memoryRouters.empty()) {
        memoryRouters.assign(std::begin(MEMORY_ROUTERS), std::end(MEMORY_ROUTERS));
        output->verbose(CALL_INFO, 1, 0,
                        "GlobalMemory '%s': memoryRouters not provided, using legacy fallback [%u,%u,%u,%u]\n",
                        getName().c_str(),
                        static_cast<unsigned>(MEMORY_ROUTERS[0]),
                        static_cast<unsigned>(MEMORY_ROUTERS[1]),
                        static_cast<unsigned>(MEMORY_ROUTERS[2]),
                        static_cast<unsigned>(MEMORY_ROUTERS[3]));
    }

    output->verbose(CALL_INFO, 2, 0,
                    "GlobalMemory init: core_id=%d, identityWindowBase=0x%" PRIx64 "\n",
                    core_id, identityWindowBase);
    std::ostringstream routers_desc;
    for (size_t i = 0; i < memoryRouters.size(); ++i) {
        if (i) routers_desc << ",";
        routers_desc << static_cast<unsigned>(memoryRouters[i]);
    }
    output->verbose(CALL_INFO, 2, 0,
                    "GlobalMemory DMA fallback routers: [%s]\n",
                    routers_desc.str().c_str());
    output->verbose(CALL_INFO, 2, 0,
                    "GlobalMemory VN mapping: read_request_vn=%u response_vn=%u write_vn=%u reduction_vn=%u (num_vns=%d)\n",
                    request_vn, response_vn, write_vn, reduction_vn, num_vns);
    output->verbose(CALL_INFO, 2, 0,
                    "GlobalMemory DMA window: max_inflight=%u retry_ticks=%u max_retries=%u\n",
                    dma_read_max_inflight, dma_read_retry_ticks, dma_read_max_retries);
    output->verbose(CALL_INFO, 2, 0,
                    "GlobalMemory DMA burst_bytes=%u\n",
                    dma_burst_bytes);

    link_control = loadUserSubComponent<SST::Interfaces::SimpleNetwork>(
        "networkIF", ComponentInfo::SHARE_NONE, num_vns);
    if (!link_control) {
        // 未显式提供 networkIF，使用 merlin.linkcontrol 作为匿名子组件
        Params if_params;
        if_params.insert("link_bw", params.find<std::string>("link_bw"));
        if_params.insert("input_buf_size", params.find<std::string>("buffer_length", "1kB"));
        if_params.insert("output_buf_size", params.find<std::string>("buffer_length", "1kB"));
        if_params.insert("port_name", "rtr");
        if_params.insert("vn_priority_order",
                         params.find<std::string>("network_vn_priority_order", ""));
        if_params.insert("vn_starvation_vn",
                         params.find<std::string>("network_vn_starvation_vn", "-1"));
        if_params.insert("max_starvation_cycles",
                         params.find<std::string>("network_vn_max_starvation_cycles", "0"));

        link_control = loadAnonymousSubComponent<SST::Interfaces::SimpleNetwork>(
            "merlin.linkcontrol", "networkIF", 0,
            ComponentInfo::SHARE_PORTS | ComponentInfo::INSERT_STATS,
            if_params, num_vns);
    }

    // Configure selfLink
    latencyTC = getTimeConverter(globalMemTransLatency);
    selfLink = configureSelfLink("Self", *latencyTC, new Event::Handler2<GlobalMemoryImplement,&GlobalMemoryImplement::handleSelfEvent>(this));
    selfLink->setDefaultTimeBase(*latencyTC);
    localAccessTC_ = getTimeConverter(localAccessClock);
    localAccessSelfLink_ = configureSelfLink(
        "LocalAccessSelf",
        *localAccessTC_,
        new Event::Handler2<GlobalMemoryImplement,
                            &GlobalMemoryImplement::handleLocalAccessEvent>(this));

    // 注册接收通知回调，当网络收到数据包时调用 handle_receives()
    link_control->setNotifyOnReceive(
        new SST::Interfaces::SimpleNetwork::Handler<GlobalMemoryImplement>(this, &GlobalMemoryImplement::handle_receives));
    link_control->setNotifyOnSend(
        new SST::Interfaces::SimpleNetwork::Handler2<GlobalMemoryImplement, &GlobalMemoryImplement::handle_send_available>(this));

    // DMA 功能复用 link_control，复用现有的接收回调
    output->verbose(CALL_INFO, 2, 0,
                    "DMA operations will use shared link_control (networkIF)\n");

}

GlobalMemoryImplement::~GlobalMemoryImplement() 
{ 
    while (!send_retry_queue.empty()) {
        delete send_retry_queue.front();
        send_retry_queue.pop_front();
    }
    if (dma_handlers) delete dma_handlers;
}

int GlobalMemoryImplement::ctrlLookupMemNicEndpointId(uint64_t phys_addr)
{
    return getMemNicEndpointId(phys_addr);
}

uint8_t GlobalMemoryImplement::ctrlLookupDmaTargetRouter(uint64_t phys_addr)
{
    return getDmaTargetRouter(phys_addr);
}

int GlobalMemoryImplement::ctrlResolveEndpointForAddress(uint64_t addr)
{
    return resolveEndpointForAddress(addr);
}

uint64_t GlobalMemoryImplement::ctrlGetReadFlagAddr(uint8_t readSlot) const
{
    uint64_t seq_addr = 0;
    uint64_t flag_addr = 0;
    get_dma_flag_addrs(false, readSlot, seq_addr, flag_addr);
    return flag_addr;
}

uint64_t GlobalMemoryImplement::ctrlReadLocalU64(uint64_t addr) const
{
    return read_u64_from_storage(addr);
}

void GlobalMemoryImplement::ctrlWriteLocalU64(uint64_t addr, uint64_t value)
{
    write_u64_to_storage(addr, value);
}

void GlobalMemoryImplement::ctrlRegisterPendingReadRequest(uint64_t requestId, uint64_t gmDstAddr,
                                                           uint64_t completionFlagAddr, uint64_t completionValue,
                                                           size_t totalLength)
{
    if (requestId == 0) {
        return;
    }

    PendingReadRequest pending;
    pending.gm_dst_addr = gmDstAddr;
    pending.completion_flag_addr = completionFlagAddr;
    pending.completion_value = completionValue;
    pending.total_len = totalLength;
    pending.received_len = 0;
    pending.submit_cycle = getCurrentSimCycle();
    request_pending[requestId] = pending;
    output->verbose(CALL_INFO, 1, 2,
                    "ctrlRegisterPendingReadRequest: core=%d req=%" PRIu64 " dst=0x%" PRIx64
                    " flag=0x%" PRIx64 " val=%" PRIu64 "\n",
                    core_id, requestId, gmDstAddr, completionFlagAddr, completionValue);
}

bool GlobalMemoryImplement::ctrlIsReadRequestPending(uint64_t requestId) const
{
    return request_pending.find(requestId) != request_pending.end();
}

bool GlobalMemoryImplement::reductionNetworkAvailable() const
{
    return link_control != nullptr && reduction_vn < static_cast<uint32_t>(num_vns);
}

void GlobalMemoryImplement::setReductionMessageHandler(ReductionMessageHandler handler)
{
    reduction_message_handler = std::move(handler);
}

bool GlobalMemoryImplement::sendReductionMessage(uint32_t destinationCore,
                                                  const ReductionTransportMessage& message)
{
    if (!reductionNetworkAvailable() ||
        destinationCore > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        statReductionSendRejected_->addData(1);
        return false;
    }

    if (network_id == -1) {
        network_id = link_control->getEndpointID();
        coreEndpointMap[core_id] = network_id;
    }

    const auto destination = coreEndpointMap.find(static_cast<int>(destinationCore));
    if (destination == coreEndpointMap.end() || destination->second < 0) {
        statReductionSendRejected_->addData(1);
        return false;
    }

    auto* req = new (std::nothrow) SST::Interfaces::SimpleNetwork::Request();
    ReductionTransportMessage stampedMessage = message;
    stampedMessage.sendCycle = getCurrentSimCycle();
    auto* payload = new ReductionTransportEvent(stampedMessage);
    if (req == nullptr || payload == nullptr) {
        delete req;
        delete payload;
        statReductionSendRejected_->addData(1);
        return false;
    }

    req->size_in_bits = sizeof(ReductionTransportMessage) * 8;
    req->src = network_id;
    req->dest = static_cast<SST::Interfaces::SimpleNetwork::nid_t>(destination->second);
    req->vn = reduction_vn;
    req->givePayload(payload);

    const bool immediateSend = try_send_or_queue(req, "sendReductionMessage");
    if (immediateSend) {
        reduction_send_immediate_count++;
        statReductionSendImmediate_->addData(1);
    } else {
        // try_send_or_queue retains ownership in send_retry_queue on a full NIC.
        reduction_send_queued_count++;
        statReductionSendQueued_->addData(1);
    }
    return true;
}

void GlobalMemoryImplement::ctrlSetReadCompletionCallback(std::function<void(uint64_t)> callback)
{
    ctrl_read_completion_callback = std::move(callback);
}

void GlobalMemoryImplement::notifyCtrlReadComplete(uint64_t requestId)
{
    if (ctrl_read_completion_callback) {
        ctrl_read_completion_callback(requestId);
    }
}

bool GlobalMemoryImplement::try_send_or_queue(SST::Interfaces::SimpleNetwork::Request* req, const char* context)
{
    if (!req) return false;

    if (!send_retry_queue.empty()) {
        flush_send_retry_queue();
    }

    if (send_retry_queue.empty() && link_control->send(req, req->vn)) {
        track_send_queue_stats(req, true);
        mark_dma_read_sent(req);
        return true;
    }

    send_retry_queue.push_back(req);
    if (send_retry_queue.size() > send_retry_queue_max_depth) {
        send_retry_queue_max_depth = send_retry_queue.size();
    }
    track_send_queue_stats(req, false);
    output->output("GlobalMemory: Warning - network buffer full, queued send (%s), retry_queue=%zu\n",
                   context ? context : "unknown", send_retry_queue.size());
    return false;
}

bool GlobalMemoryImplement::flush_send_retry_queue()
{
    size_t sent_count = 0;
    while (!send_retry_queue.empty()) {
        SST::Interfaces::SimpleNetwork::Request* req = send_retry_queue.front();
        bool is_dma_read_req = (dma_read_req_to_key.find(req) != dma_read_req_to_key.end());
        if (!link_control->send(req, req->vn)) {
            return false;
        }
        mark_dma_read_sent(req);
        if (is_dma_read_req) {
            dma_read_send_flushed_count++;
        }
        send_retry_queue.pop_front();
        sent_count++;
    }

    if (sent_count > 0) {
        output->verbose(CALL_INFO, 1, 0,
                        "GlobalMemory: flushed %zu queued sends\n", sent_count);
    }
    return true;
}

bool GlobalMemoryImplement::handle_send_available(int)
{
    return flush_send_retry_queue();
}

namespace {
constexpr uint64_t DMA_FLAG_REGION_SIZE = 64;   // bytes reserved at GM tail
constexpr uint64_t READ0_SEQ_OFFSET = 64;       // size - 64
constexpr uint64_t READ0_FLAG_OFFSET = 56;      // size - 56
constexpr uint64_t READ1_SEQ_OFFSET = 48;       // size - 48
constexpr uint64_t READ1_FLAG_OFFSET = 40;      // size - 40
constexpr uint64_t WRITE_SEQ_OFFSET = 32;       // size - 32
constexpr uint64_t WRITE_FLAG_OFFSET = 24;      // size - 24
constexpr uint64_t READ_SLOT_SELECT_OFFSET = 16; // size - 16
}

uint64_t GlobalMemoryImplement::read_u64_from_storage(uint64_t addr) const {
    if (addr < baseAddr || (addr + sizeof(uint64_t)) > (baseAddr + size)) {
        output->fatal(CALL_INFO, -1,
                      "GlobalMemory '%s' read_u64_from_storage out of range: addr=0x%" PRIx64 "\n",
                      getName().c_str(), addr);
    }
    uint64_t offset = addr - baseAddr;
    uint64_t value = 0;
    std::memcpy(&value, storage.data() + offset, sizeof(uint64_t));
    return value;
}

void GlobalMemoryImplement::write_u64_to_storage(uint64_t addr, uint64_t value) {
    if (addr < baseAddr || (addr + sizeof(uint64_t)) > (baseAddr + size)) {
        output->fatal(CALL_INFO, -1,
                      "GlobalMemory '%s' write_u64_to_storage out of range: addr=0x%" PRIx64 "\n",
                      getName().c_str(), addr);
    }
    uint64_t offset = addr - baseAddr;
    std::memcpy(storage.data() + offset, &value, sizeof(uint64_t));
}

void GlobalMemoryImplement::get_dma_flag_addrs(bool is_write, uint8_t read_slot, uint64_t& seq_addr, uint64_t& flag_addr) const {
    if (size < DMA_FLAG_REGION_SIZE) {
        output->fatal(CALL_INFO, -1,
                      "GlobalMemory '%s' size=0x%" PRIx64 " too small for DMA flag region\n",
                      getName().c_str(), size);
    }
    if (is_write) {
        seq_addr = baseAddr + size - WRITE_SEQ_OFFSET;
        flag_addr = baseAddr + size - WRITE_FLAG_OFFSET;
    } else {
        if (read_slot == 0) {
            seq_addr = baseAddr + size - READ0_SEQ_OFFSET;
            flag_addr = baseAddr + size - READ0_FLAG_OFFSET;
        } else if (read_slot == 1) {
            seq_addr = baseAddr + size - READ1_SEQ_OFFSET;
            flag_addr = baseAddr + size - READ1_FLAG_OFFSET;
        } else {
            output->fatal(CALL_INFO, -1,
                          "GlobalMemory '%s' invalid DMA read slot=%u\n",
                          getName().c_str(), static_cast<unsigned>(read_slot));
        }
    }
}

void GlobalMemoryImplement::wr_to_globalmem(uint64_t wr_addr, size_t length, const std::vector<uint8_t>& wr_data)
{
    // 确保写入地址和长度在本地存储范围内
    output->verbose(CALL_INFO, 3, 2,
                "wr_to_globalmem: core %d writing addr=0x%" PRIx64 " len=%zu\n"
                "baseAddr=0x%" PRIx64 " size=%zu\n", // 更新格式字符串
                core_id, wr_addr, length, baseAddr, size);

    if (gm_dump_data && length > 0) {
        const size_t dump_len = length < 64 ? length : 64;
        char hexbuf[3 * 64 + 1];
        for (size_t i = 0; i < dump_len; ++i) {
            std::snprintf(hexbuf + i * 3, 4, "%02X ", wr_data[i]);
        }
        hexbuf[3 * dump_len] = '\0';
        output->verbose(CALL_INFO, 10, 2,
                        "wr_to_globalmem: data[0..%zu]=%s\n",
                        dump_len - 1, hexbuf);
    }

    assert(wr_addr >= baseAddr);
    assert((wr_addr + length) <= (baseAddr + size));
    assert(wr_data.size() == length && "Data size must match the length");

    uint64_t offset = wr_addr - baseAddr;
    // 将数据写入本地 storage 向量的相对偏移位置
    std::copy(wr_data.begin(), wr_data.begin() + length, storage.begin() + offset);
}

void GlobalMemoryImplement::rd_from_globalmem(uint64_t rd_addr, size_t length, std::vector<uint8_t>& rd_data)
{
    // 确保读取地址和长度在本地存储范围内
    output->verbose(CALL_INFO, 3, 2,
                "rd_from_globalmem: core %d reading addr=0x%" PRIx64 " len=%zu, "
                "baseAddr=0x%" PRIx64 " size=%zu\n", // 更新格式字符串
                core_id, rd_addr, length, baseAddr, size); // 只传递 baseAddr 和 size

    assert(rd_addr >= baseAddr);
    assert((rd_addr + length) <= (baseAddr + size));

    uint64_t offset = rd_addr - baseAddr;
    // 从本地 storage 中读取指定范围的数据复制到 rd_data
    rd_data.assign(storage.begin() + offset, storage.begin() + offset + length);
}

bool GlobalMemoryImplement::localAccessValid(uint64_t addr, size_t length) const
{
    if (length == 0 || length > localAccessMaxRequestBytes_ || addr < baseAddr) {
        return false;
    }
    const uint64_t offset = addr - baseAddr;
    return offset <= size && length <= size - offset;
}

bool GlobalMemoryImplement::localReadAsync(uint64_t addr,
                                           size_t length,
                                           LocalMemoryClient client,
                                           uint64_t tag,
                                           LocalReadCallback cb)
{
    if (!cb || !localAccessValid(addr, length) ||
        localAccessPending_.size() >= localAccessQueueDepth_) {
        ++localQueueRejected_;
        return false;
    }

    PendingLocalAccess request;
    request.kind = PendingLocalAccess::Kind::Read;
    request.requestId = nextLocalAccessRequestId_++;
    request.addr = addr;
    request.length = length;
    request.client = client;
    request.tag = tag;
    request.submitCycle = getCurrentSimCycle();
    request.ownerGeneration = attentionClusterEnable_
        ? attentionGenerationFence_.active() : 0;
    request.readCallback = std::move(cb);
    const uint64_t requestId = request.requestId;
    localAccessPending_.emplace(requestId, std::move(request));
    localReadQueue_.push_back(requestId);
    ++localReadRequests_;

    const uint64_t depth = localAccessPending_.size();
    if (depth > localAccessQueueHighWater_) {
        localAccessQueueHighWater_ = depth;
    }
    tryIssueLocalAccesses();
    return true;
}

bool GlobalMemoryImplement::localWriteAsync(uint64_t addr,
                                            const std::vector<uint8_t>& data,
                                            LocalMemoryClient client,
                                            uint64_t tag,
                                            LocalWriteCallback cb)
{
    if (!cb || !localAccessValid(addr, data.size()) ||
        localAccessPending_.size() >= localAccessQueueDepth_) {
        ++localQueueRejected_;
        return false;
    }

    PendingLocalAccess request;
    request.kind = PendingLocalAccess::Kind::Write;
    request.requestId = nextLocalAccessRequestId_++;
    request.addr = addr;
    request.length = data.size();
    request.client = client;
    request.tag = tag;
    request.submitCycle = getCurrentSimCycle();
    request.ownerGeneration = attentionClusterEnable_
        ? attentionGenerationFence_.active() : 0;
    request.writeData = data;
    request.writeCallback = std::move(cb);
    const uint64_t requestId = request.requestId;
    localAccessPending_.emplace(requestId, std::move(request));
    localWriteQueue_.push_back(requestId);
    ++localWriteRequests_;

    const uint64_t depth = localAccessPending_.size();
    if (depth > localAccessQueueHighWater_) {
        localAccessQueueHighWater_ = depth;
    }
    tryIssueLocalAccesses();
    return true;
}

uint64_t GlobalMemoryImplement::localAccessDelayCycles(size_t length) const
{
    return static_cast<uint64_t>(localAccessBaseLatencyCycles_) +
           ceilDivLocalAccess(length, localAccessBytesPerCycle_);
}

void GlobalMemoryImplement::issueLocalAccess(uint64_t requestId)
{
    auto it = localAccessPending_.find(requestId);
    if (it == localAccessPending_.end()) {
        return;
    }
    PendingLocalAccess& request = it->second;
    request.issued = true;
    const uint64_t queueCycles = getCurrentSimCycle() - request.submitCycle;
    if (request.kind == PendingLocalAccess::Kind::Read) {
        localReadQueueCycles_ += queueCycles;
    } else {
        localWriteQueueCycles_ += queueCycles;
    }
    const uint64_t serviceCycles = localAccessDelayCycles(request.length);
    if (attentionClusterEnable_) {
        BusyIntervalUnion& intervals =
            request.kind == PendingLocalAccess::Kind::Read
                ? attentionClusterLocalReadIntervals_
                : attentionClusterLocalWriteIntervals_;
        const uint64_t start = getCurrentSimCycle();
        if (!intervals.add(
                start, start + localAccessTC_->convertToCoreTime(serviceCycles))) {
            output->fatal(CALL_INFO, -1,
                          "Attention cluster Local-GM interval order violation\n");
        }
    }
    localAccessSelfLink_->send(serviceCycles,
                               new LocalMemoryAccessEvent(requestId));
}

void GlobalMemoryImplement::tryIssueLocalAccesses()
{
    while (localReadsInFlight_ < localAccessReadPorts_ && !localReadQueue_.empty()) {
        const uint64_t requestId = localReadQueue_.front();
        localReadQueue_.pop_front();
        ++localReadsInFlight_;
        issueLocalAccess(requestId);
    }
    while (localWritesInFlight_ < localAccessWritePorts_ && !localWriteQueue_.empty()) {
        const uint64_t requestId = localWriteQueue_.front();
        localWriteQueue_.pop_front();
        ++localWritesInFlight_;
        issueLocalAccess(requestId);
    }
}

void GlobalMemoryImplement::handleLocalAccessEvent(Event* ev)
{
    auto* accessEvent = dynamic_cast<LocalMemoryAccessEvent*>(ev);
    if (accessEvent == nullptr) {
        delete ev;
        return;
    }
    const uint64_t requestId = accessEvent->requestId();
    delete accessEvent;

    auto it = localAccessPending_.find(requestId);
    if (it == localAccessPending_.end()) {
        return;
    }
    PendingLocalAccess request = std::move(it->second);
    localAccessPending_.erase(it);

    const bool generationValid = request.ownerGeneration == 0 ||
        attentionGenerationFence_.matches(request.ownerGeneration);

    if (request.kind == PendingLocalAccess::Kind::Read) {
        if (localReadsInFlight_ > 0) {
            --localReadsInFlight_;
        }
        if (request.cancelled || !generationValid) {
            tryIssueLocalAccesses();
            request.readCallback(false, request.tag, {});
            return;
        }
        std::vector<uint8_t> data;
        const uint64_t offset = request.addr - baseAddr;
        data.assign(storage.begin() + offset,
                    storage.begin() + offset + request.length);
        localReadBytes_ += request.length;
        tryIssueLocalAccesses();
        request.readCallback(true, request.tag, data);
        return;
    }

    if (localWritesInFlight_ > 0) {
        --localWritesInFlight_;
    }
    if (request.cancelled || !generationValid) {
        tryIssueLocalAccesses();
        request.writeCallback(false, request.tag);
        return;
    }
    const uint64_t offset = request.addr - baseAddr;
    std::copy(request.writeData.begin(), request.writeData.end(),
              storage.begin() + offset);
    localWriteBytes_ += request.length;
    tryIssueLocalAccesses();
    request.writeCallback(true, request.tag);
}

void GlobalMemoryImplement::completeDmaReadToLocalMemory(
    PendingDmaOp op, const std::vector<uint8_t>& data)
{
    if (op.ownerGeneration != 0 &&
        !attentionGenerationFence_.matches(op.ownerGeneration)) {
        return;
    }
    const uint64_t landingTag = nextLocalDmaTag_++;
    PendingDmaLanding landing;
    landing.op = std::move(op);
    landing.data = data;
    pendingDmaLandings_.emplace(landingTag, std::move(landing));
    issueNextDmaLandingChunk(landingTag);
}

void GlobalMemoryImplement::issueNextDmaLandingChunk(uint64_t landingTag)
{
    auto it = pendingDmaLandings_.find(landingTag);
    if (it == pendingDmaLandings_.end()) {
        return;
    }
    PendingDmaLanding& landing = it->second;
    if (landing.offset >= landing.data.size()) {
        PendingDmaOp completedOp = std::move(landing.op);
        pendingDmaLandings_.erase(it);
        finishDmaReadLanding(completedOp, true);
        return;
    }

    const size_t chunkBytes = std::min<size_t>(
        localAccessMaxRequestBytes_, landing.data.size() - landing.offset);
    std::vector<uint8_t> chunk(
        landing.data.begin() + landing.offset,
        landing.data.begin() + landing.offset + chunkBytes);
    const uint64_t chunkAddr = landing.op.gm_dst_addr + landing.offset;
    const bool accepted = localWriteAsync(
        chunkAddr, chunk, LocalMemoryClient::Dma, landingTag,
        [this, chunkBytes](bool ok, uint64_t tag) {
            auto landingIt = pendingDmaLandings_.find(tag);
            if (landingIt == pendingDmaLandings_.end()) {
                return;
            }
            if (!ok) {
                PendingDmaOp failedOp = std::move(landingIt->second.op);
                pendingDmaLandings_.erase(landingIt);
                finishDmaReadLanding(failedOp, false);
                return;
            }
            landingIt->second.offset += chunkBytes;
            issueNextDmaLandingChunk(tag);
        });
    if (!accepted) {
        PendingDmaOp failedOp = std::move(landing.op);
        pendingDmaLandings_.erase(it);
        finishDmaReadLanding(failedOp, false);
    }
}

void GlobalMemoryImplement::finishDmaReadLanding(PendingDmaOp& op, bool ok)
{
    ++dma_read_completion_count;
    if (op.ctx) {
        op.ctx->ok = op.ctx->ok && ok;
        if (op.ctx->remaining > 0) {
            --op.ctx->remaining;
        }
        if (op.ctx->remaining != 0) {
            return;
        }
        if (op.ctx->completion_enabled) {
            write_u64_to_storage(op.ctx->completion_flag_addr, op.ctx->completion_value);
        }
        if (op.ctx->completion_token != 0) {
            dma_completion_tokens[op.ctx->completion_token] = true;
        }
        if (op.ctx->cb) {
            op.ctx->cb(op.ctx->ok);
        }
        return;
    }

    if (op.completion_enabled) {
        write_u64_to_storage(op.completion_flag_addr, op.completion_value);
    }
    if (op.completion_token != 0) {
        dma_completion_tokens[op.completion_token] = true;
    }
    if (op.cb) {
        op.cb(ok);
    }
}

void GlobalMemoryImplement::wr_to_network(uint64_t wr_addr, size_t length, std::vector<uint8_t>& wr_data)
{
    
    // Schedule the send after latency (simulate NIC delay)
    //SimTime_t delay = getLatency(0);  // use configured latency (cycles)
    //selfLink->send(delay, new SST::Event());

    // 检查是否是 Identity Window 地址 (DMA 写主存)
    if (wr_addr >= identityWindowBase) {
        output->verbose(CALL_INFO, 2, 2,
                        "wr_to_network: core %d DMA write addr=0x%" PRIx64 " len=%zu\n",
                        core_id, wr_addr, length);
        dma_write_to_host(wr_addr, length, wr_data, nullptr);
        return;
    }

    int dest_endpoint = resolveEndpointForAddress(wr_addr);

    // 如果目标端点就是当前端点，直接执行本地写操作
    if (dest_endpoint == network_id) {
        wr_to_globalmem(wr_addr, length, wr_data);
        return;
    }

    // 创建网络请求包
    SST::Interfaces::SimpleNetwork::Request* req = new SST::Interfaces::SimpleNetwork::Request();

    // 计算请求数据总大小（字节 -> 比特）
    size_t total_size_in_bytes = sizeof(wr_addr) + sizeof(length) + wr_data.size();
    req->size_in_bits = total_size_in_bytes * 8;
    req->src = network_id;       // 源节点使用真实的网络端点 ID
    req->dest = dest_endpoint;   // 目的节点 ID
    req->vn = request_vn;

    // 创建 NetworkDataEvent 负载 (WRITE 类型，包含要写入的数据)
    NetworkDataEvent* payload = new NetworkDataEvent(NetworkDataEvent::WRITE, wr_addr, length, wr_data);
    req->givePayload(payload);

    // 通过网络接口发送请求
    try_send_or_queue(req, "wr_to_network");
}

void GlobalMemoryImplement::rd_to_network(uint64_t rd_addr, size_t length, uint64_t returnAddr)
{
    // 确保 network_id 已初始化
    if (network_id == -1) {
        if (!link_control) {
            output->fatal(CALL_INFO, -1,
                          "GlobalMemory '%s' rd_to_network: link_control is null!\n",
                          getName().c_str());
        }
        network_id = link_control->getEndpointID();
        coreEndpointMap[core_id] = network_id;
    }

    // 检查是否是 Identity Window 地址 (DMA 访问主存)
    if (rd_addr >= identityWindowBase) {
        output->verbose(CALL_INFO, 1, 2,
                        "rd_to_network: core %d DMA read addr=0x%" PRIx64 " len=%zu\n",
                        core_id, rd_addr, length);
        // 使用 DMA 路径访问主存
        dma_read_from_host_to_globalmem(rd_addr, length, returnAddr, nullptr);
        return;
    }

    int dest_endpoint = resolveEndpointForAddress(rd_addr);

    if (dest_endpoint == network_id) {
        // 目标在本地，无需经过网络
        return;
    }

    output->verbose(CALL_INFO, 1, 2,
                    "rd_to_network: core %d issuing READ addr=0x%" PRIx64 " len=%zu return=0x%" PRIx64 " dest_ep=%d src_ep=%d\n",
                    core_id, rd_addr, length, returnAddr, dest_endpoint, network_id);

    if (returnAddr >= baseAddr && (returnAddr + length) <= (baseAddr + size)) {
        uint64_t seq_addr = 0;
        uint64_t flag_addr = 0;
        get_dma_flag_addrs(false, 0, seq_addr, flag_addr);
        PendingReadReply pending;
        pending.completion_flag_addr = flag_addr;
        pending.completion_value = read_u64_from_storage(seq_addr);
        read_pending[returnAddr] = pending;
    }

    // 创建网络请求包
    SST::Interfaces::SimpleNetwork::Request* req = new SST::Interfaces::SimpleNetwork::Request();

    // 计算请求数据总大小（仅包含地址和长度，无数据）
    size_t total_size_in_bytes = sizeof(rd_addr) + sizeof(length);
    req->size_in_bits = total_size_in_bytes * 8;
    req->src = network_id;
    req->dest = dest_endpoint;
    req->vn = request_vn;

    // 创建 NetworkDataEvent 负载 (READ 类型，不携带数据)
    NetworkDataEvent* payload = new NetworkDataEvent(NetworkDataEvent::READ, rd_addr, length, std::vector<uint8_t>(), returnAddr);
    req->givePayload(payload);

    // 通过网络接口发送请求
    try_send_or_queue(req, "rd_to_network");
}

void GlobalMemoryImplement::schedule_dma_retry_event() {
    if (!dma_retry_event_scheduled && selfLink != nullptr) {
        selfLink->send(1, new Event());
        dma_retry_event_scheduled = true;
    }
}

bool GlobalMemoryImplement::mark_dma_read_sent(SST::Interfaces::SimpleNetwork::Request* req) {
    auto req_it = dma_read_req_to_key.find(req);
    if (req_it == dma_read_req_to_key.end()) {
        return false;
    }

    uint64_t pending_key = req_it->second;
    dma_read_req_to_key.erase(req_it);

    auto it = dma_pending.find(pending_key);
    if (it == dma_pending.end()) {
        return false;
    }

    PendingDmaOp& op = it->second;
    if (op.kind != PendingDmaOp::READ_TO_GM) {
        return false;
    }

    if (!op.first_send_seen) {
        op.first_send_tick = dma_retry_tick_counter;
        op.first_send_cycle = getCurrentSimCycle();
        op.first_send_seen = true;
    }
    op.request_sent = true;
    op.last_send_tick = dma_retry_tick_counter;
    op.last_send_cycle = getCurrentSimCycle();
    op.retry_ticks_left = dma_read_retry_ticks;
    return true;
}

void GlobalMemoryImplement::track_send_queue_stats(SST::Interfaces::SimpleNetwork::Request* req, bool immediate_send) {
    if (dma_read_req_to_key.find(req) == dma_read_req_to_key.end()) {
        return;
    }

    if (immediate_send) {
        dma_read_send_immediate_count++;
    } else {
        dma_read_send_queued_count++;
    }
}

void GlobalMemoryImplement::issue_dma_read_chunk(PendingDmaOp& op, const char* reason) {
    auto* req = new SST::Interfaces::SimpleNetwork::Request();
    req->src = network_id;

    int memnic_ep = getMemNicEndpointId(op.host_addr);
    if (memnic_ep != -1) {
        req->dest = memnic_ep;
    } else {
        req->dest = getDmaTargetRouter(op.host_addr);
    }

    req->vn = request_vn;
    req->size_in_bits = (sizeof(op.host_addr) + sizeof(op.length) + sizeof(op.gm_dst_addr)) * 8;

    auto* payload = new NetworkDataEvent(
        NetworkDataEvent::READ,
        op.host_addr,
        op.length,
        std::vector<uint8_t>(),
        op.gm_dst_addr,
        network_id,
        0,
        0,
        op.request_id,
        op.dmaRequestKind,
        op.dmaConsumer);
    payload->setOwnerGenerationToken(op.ownerGenerationToken);
    req->givePayload(payload);

    dma_read_req_to_key[req] = op.request_id;
    try_send_or_queue(req, reason ? reason : "issue_dma_read_chunk");
}

size_t GlobalMemoryImplement::count_issued_dma_reads() const {
    size_t count = 0;
    for (const auto& entry : dma_pending) {
        const PendingDmaOp& op = entry.second;
        if (op.kind == PendingDmaOp::READ_TO_GM && op.request_issued) {
            count++;
        }
    }
    return count;
}

void GlobalMemoryImplement::issue_pending_dma_read_window() {
    size_t in_flight = count_issued_dma_reads();
    if (in_flight >= dma_read_max_inflight) {
        return;
    }

    for (auto& entry : dma_pending) {
        PendingDmaOp& op = entry.second;
        if (op.kind != PendingDmaOp::READ_TO_GM || op.request_issued) {
            continue;
        }

        op.request_issued = true;
        op.request_sent = false;
        op.first_send_seen = false;
        op.first_send_tick = 0;
        op.last_send_tick = 0;
        op.first_send_cycle = 0;
        op.last_send_cycle = 0;
        op.retry_ticks_left = dma_read_retry_ticks;
        issue_dma_read_chunk(op, "dma_read_issue_window");
        in_flight++;
        if (in_flight >= dma_read_max_inflight) {
            break;
        }
    }
}

void GlobalMemoryImplement::process_dma_read_retries() {
    if (dma_pending.empty()) {
        return;
    }

    std::vector<uint64_t> to_erase;
    to_erase.reserve(8);

    for (auto& entry : dma_pending) {
        const uint64_t key = entry.first;
        PendingDmaOp& op = entry.second;
        if (op.kind != PendingDmaOp::READ_TO_GM) {
            continue;
        }
        if (!op.request_issued) {
            continue;
        }
        if (!op.request_sent) {
            continue;
        }

        if (op.retry_ticks_left > 0) {
            op.retry_ticks_left--;
            continue;
        }

        if (op.retry_attempts < dma_read_max_retries) {
            op.retry_attempts++;
            dma_read_timeout_retry_count++;
            output->output("GlobalMemory: DMA READ retry core=%d gm_dst=0x%" PRIx64
                           " host=0x%" PRIx64 " len=%zu attempt=%u/%u\n",
                           core_id, op.gm_dst_addr, op.host_addr, op.length,
                           op.retry_attempts, dma_read_max_retries);
            op.request_sent = false;
            issue_dma_read_chunk(op, "dma_read_retry");
            continue;
        }

        dma_read_timeout_exhausted_count++;

        output->output("GlobalMemory: DMA READ chunk timeout exhausted core=%d gm_dst=0x%" PRIx64
                       " host=0x%" PRIx64 " len=%zu attempts=%u; forcing completion to avoid deadlock\n",
                       core_id, op.gm_dst_addr, op.host_addr, op.length, op.retry_attempts);

        if (op.ctx) {
            op.ctx->ok = false;
            if (op.ctx->remaining > 0) {
                op.ctx->remaining--;
            }
            if (op.ctx->remaining == 0) {
                if (op.ctx->completion_enabled) {
                    write_u64_to_storage(op.ctx->completion_flag_addr, op.ctx->completion_value);
                }
                if (op.ctx->cb) {
                    op.ctx->cb(false);
                }
            }
        } else {
            if (op.completion_enabled) {
                write_u64_to_storage(op.completion_flag_addr, op.completion_value);
            }
            if (op.cb) {
                op.cb(false);
            }
        }

        to_erase.push_back(key);
    }

    for (uint64_t key : to_erase) {
        dma_pending.erase(key);
    }

    issue_pending_dma_read_window();
}

void GlobalMemoryImplement::handleSelfEvent(Event* ev) {
    dma_retry_tick_counter++;
    process_dma_read_retries();
    dma_retry_event_scheduled = false;
    if (!dma_pending.empty()) {
        schedule_dma_retry_event();
    }
    delete ev;
}

//这个函数是为了返回阵列的延迟时间，通常在模拟中用来表示阵列进行计算所需的时间。在这段代码中，延迟被硬编码为 1。
SST::SimTime_t GlobalMemoryImplement::getLatency(uint32_t arrayID)  {
    return 1;
}

void GlobalMemoryImplement::setBaseAddr(uint64_t addr) {
    baseAddr = addr;
}

uint64_t GlobalMemoryImplement::getBaseAddr() const {
    return baseAddr;
}

uint64_t GlobalMemoryImplement::getSize() const {
    return size;
}

void GlobalMemoryImplement::init(unsigned int phase) {
    output->verbose(CALL_INFO, 1, 2, "init phase=%u\n", phase);
    // 初始化网络接口的 init 阶段
    link_control->init(phase);
    if (dma_iface) {
        dma_iface->init(phase);
    }
    if (link_control->isNetworkInitialized()) {
        if (network_id == -1) {
            network_id = link_control->getEndpointID();
            coreEndpointMap[core_id] = network_id;
        }
        if ( !init_broadcast_done ) {
            // 广播自身 EndpointInfo，供其它组件解析映射
            SST::MemHierarchy::MemLinkBase::EndpointInfo info;
            info.name = getName();
            info.addr = static_cast<uint64_t>(network_id);
            info.id = static_cast<uint32_t>(core_id);
            info.region.setDefault();
            info.region.start = baseAddr;
            info.region.end = baseAddr + size - 1;
            info.region.interleaveSize = 0;
            info.region.interleaveStep = 0;

            auto* initEvent = new SST::MemHierarchy::MemNICBase::InitMemRtrEvent(info);
            auto* req = new SST::Interfaces::SimpleNetwork::Request();
            req->dest = SST::Interfaces::SimpleNetwork::INIT_BROADCAST_ADDR;
            req->src = network_id;
            req->givePayload(initEvent);
            link_control->sendUntimedData(req);
            init_broadcast_done = true;
        }

        // 持续接收其他端点的广播名称，填充 ID 映射表
        SST::Interfaces::SimpleNetwork::Request* recvReq = nullptr;
        while ((recvReq = link_control->recvUntimedData()) != nullptr) {
            SST::Event* payload = recvReq->takePayload();
            if (auto* nameEv = dynamic_cast<SST::Interfaces::StringEvent*>(payload)) {
                registerEndpoint(nameEv->getString(), recvReq->src);
            } else if (auto* initEv = dynamic_cast<SST::MemHierarchy::MemNICBase::InitMemRtrEvent*>(payload)) {
                registerEndpoint(initEv->info.name, static_cast<int>(initEv->info.addr));
            } else {
                const char* typeName = payload ? typeid(*payload).name() : "<null>";
                output->verbose(CALL_INFO, 1, 2,
                                "Ignoring untimed init event of type %s from endpoint %" PRI_NID "\n",
                                typeName, recvReq->src);
            }
            delete payload;
            delete recvReq;
        }
    }
}

void GlobalMemoryImplement::setup() {
    if (network_id == -1) {
        network_id = link_control->getEndpointID();
    }
    coreEndpointMap[core_id] = network_id;
    link_control->setup();
    if (dma_iface) {
        dma_iface->setup();
    }
    schedule_dma_retry_event();
}

void GlobalMemoryImplement::complete(unsigned int phase) {
    link_control->complete(phase);
    if (dma_iface) {
        dma_iface->complete(phase);
    }
}

void GlobalMemoryImplement::finish() {
    if (attentionClusterEnable_) {
        statAttentionClusterLocalReadBusyUnion_->addData(
            attentionClusterLocalReadIntervals_.unionCycles());
        statAttentionClusterLocalReadBusySpan_->addData(
            attentionClusterLocalReadIntervals_.spanCycles());
        statAttentionClusterLocalReadIdleGap_->addData(
            attentionClusterLocalReadIntervals_.idleCycles());
        statAttentionClusterLocalReadMaxConcurrency_->addData(
            attentionClusterLocalReadIntervals_.maxConcurrency());
        statAttentionClusterLocalWriteBusyUnion_->addData(
            attentionClusterLocalWriteIntervals_.unionCycles());
        statAttentionClusterLocalWriteBusySpan_->addData(
            attentionClusterLocalWriteIntervals_.spanCycles());
        statAttentionClusterLocalWriteIdleGap_->addData(
            attentionClusterLocalWriteIntervals_.idleCycles());
        statAttentionClusterLocalWriteMaxConcurrency_->addData(
            attentionClusterLocalWriteIntervals_.maxConcurrency());
    }
    output->output(
        "GOLEM_LOCAL_GM_STATS core=%d gmem_local_read_requests=%" PRIu64
        " gmem_local_write_requests=%" PRIu64
        " gmem_local_read_bytes=%" PRIu64
        " gmem_local_write_bytes=%" PRIu64
        " gmem_local_queue_rejected=%" PRIu64
        " gmem_local_queue_high_water=%" PRIu64
        " gmem_local_read_queue_cycles=%" PRIu64
        " gmem_local_write_queue_cycles=%" PRIu64 "\n",
        core_id,
        localReadRequests_,
        localWriteRequests_,
        localReadBytes_,
        localWriteBytes_,
        localQueueRejected_,
        localAccessQueueHighWater_,
        localReadQueueCycles_,
        localWriteQueueCycles_);
    uint64_t avg_rtt_ticks = (dma_read_rtt_samples > 0)
                                 ? (dma_read_rtt_ticks_sum / dma_read_rtt_samples)
                                 : 0;
    uint64_t avg_rtt_cpu_cycles = avg_rtt_ticks * dma_retry_tick_cpu_cycles;
    uint64_t max_rtt_cpu_cycles = dma_read_rtt_ticks_max * dma_retry_tick_cpu_cycles;
    uint64_t avg_e2e_rtt_ticks = (dma_read_e2e_rtt_samples > 0)
                                     ? (dma_read_e2e_rtt_ticks_sum / dma_read_e2e_rtt_samples)
                                     : 0;
    uint64_t avg_e2e_rtt_cpu_cycles = avg_e2e_rtt_ticks * dma_retry_tick_cpu_cycles;
    uint64_t max_e2e_rtt_cpu_cycles = dma_read_e2e_rtt_ticks_max * dma_retry_tick_cpu_cycles;
    uint64_t strict_avg_rtt_cycles = (dma_read_strict_rtt_samples > 0)
                                         ? (dma_read_strict_rtt_cycles_sum / dma_read_strict_rtt_samples)
                                         : 0;
    uint64_t strict_avg_e2e_rtt_cycles = (dma_read_strict_e2e_rtt_samples > 0)
                                             ? (dma_read_strict_e2e_rtt_cycles_sum / dma_read_strict_e2e_rtt_samples)
                                             : 0;
    uint64_t request_avg_submit_ready_cycles = (request_read_submit_ready_samples > 0)
                                                   ? (request_read_submit_ready_cycles_sum / request_read_submit_ready_samples)
                                                   : 0;
    uint64_t write_avg_rtt_cycles = (dma_write_rtt_samples > 0)
                                        ? (dma_write_rtt_cycles_sum / dma_write_rtt_samples)
                                        : 0;
    output->output("GlobalMemory core=%d DMA READ stats: immediate_send=%" PRIu64
                   " queued_send=%" PRIu64 " flushed_send=%" PRIu64
                   " read_issue_count=%" PRIu64 " write_issue_count=%" PRIu64
                   " read_bytes_total=%" PRIu64 " write_bytes_total=%" PRIu64
                   " timeout_retry=%" PRIu64 " timeout_exhausted=%" PRIu64
                   " write_timeout_retry=%" PRIu64
                   " completion=%" PRIu64 " write_completion=%" PRIu64
                   " completion_no_pending=%" PRIu64 " wait_count=%" PRIu64
                   " avg_rtt_ticks=%" PRIu64 " max_rtt_ticks=%" PRIu64
                   " avg_rtt_cycles=%" PRIu64 " max_rtt_cycles=%" PRIu64
                   " avg_e2e_rtt_ticks=%" PRIu64 " max_e2e_rtt_ticks=%" PRIu64
                   " avg_e2e_rtt_cycles=%" PRIu64 " max_e2e_rtt_cycles=%" PRIu64
                   " strict_avg_rtt_cycles=%" PRIu64 " strict_max_rtt_cycles=%" PRIu64
                   " strict_avg_e2e_rtt_cycles=%" PRIu64 " strict_max_e2e_rtt_cycles=%" PRIu64
                   " request_avg_submit_ready_cycles=%" PRIu64 " request_max_submit_ready_cycles=%" PRIu64
                   " write_rtt_samples=%" PRIu64 " write_rtt_cycles_sum=%" PRIu64
                   " write_avg_rtt_cycles=%" PRIu64 " write_max_rtt_cycles=%" PRIu64
                   " write_first_issue_cycle=%" PRIu64 " write_last_issue_cycle=%" PRIu64
                   " write_last_complete_cycle=%" PRIu64
                   " send_retry_q_max=%zu\n",
                   core_id,
                   dma_read_send_immediate_count,
                   dma_read_send_queued_count,
                   dma_read_send_flushed_count,
                   dma_read_issue_count,
                   dma_write_issue_count,
                   dma_read_bytes_total,
                   dma_write_bytes_total,
                   dma_read_timeout_retry_count,
                   dma_read_timeout_exhausted_count,
                   dma_write_timeout_retry_count,
                   dma_read_completion_count,
                   dma_write_completion_count,
                   dma_read_completion_no_pending_count,
                   dma_wait_count,
                   avg_rtt_ticks,
                   dma_read_rtt_ticks_max,
                   avg_rtt_cpu_cycles,
                   max_rtt_cpu_cycles,
                   avg_e2e_rtt_ticks,
                   dma_read_e2e_rtt_ticks_max,
                   avg_e2e_rtt_cpu_cycles,
                   max_e2e_rtt_cpu_cycles,
                   strict_avg_rtt_cycles,
                   dma_read_strict_rtt_cycles_max,
                   strict_avg_e2e_rtt_cycles,
                   dma_read_strict_e2e_rtt_cycles_max,
                   request_avg_submit_ready_cycles,
                   request_read_submit_ready_cycles_max,
                   dma_write_rtt_samples,
                   dma_write_rtt_cycles_sum,
                   write_avg_rtt_cycles,
                   dma_write_rtt_cycles_max,
                   dma_write_first_issue_cycle,
                   dma_write_last_issue_cycle,
                   dma_write_last_complete_cycle,
                   send_retry_queue_max_depth);
    if (reduction_send_immediate_count != 0 || reduction_send_queued_count != 0 ||
        reduction_receive_count != 0) {
        output->output("GlobalMemory core=%d reduction transport stats: immediate_send=%" PRIu64
                       " queued_send=%" PRIu64 " received=%" PRIu64 "\n",
                       core_id,
                       reduction_send_immediate_count,
                       reduction_send_queued_count,
                       reduction_receive_count);
    }
    link_control->finish();
}

void GlobalMemoryImplement::handleDmaMemEvent(SST::Interfaces::StandardMem::Request* req)
{
    // 旧版 StandardMem 接口已废弃，保留兼容但不做处理
    output->verbose(CALL_INFO, 1, 2,
                    "handleDmaMemEvent: Ignoring legacy StandardMem request\n");
    delete req;
}

void GlobalMemoryImplement::StdMemHandlers::handle(SST::Interfaces::StandardMem::ReadResp* ev)
{
    // 旧版 StandardMem 接口已废弃，保留兼容但不做处理
    delete ev;
}

void GlobalMemoryImplement::StdMemHandlers::handle(SST::Interfaces::StandardMem::WriteResp* ev)
{
    // 旧版 StandardMem 接口已废弃，保留兼容但不做处理
    delete ev;
}

void GlobalMemoryImplement::dma_write_to_host(uint64_t dst_pa, size_t length, const std::vector<uint8_t>& data, DmaCallback cb,
                                              DmaRequestKind kind)
{
    dma_write_to_host_impl(dst_pa, length, data, cb, 0, kind);
}

void GlobalMemoryImplement::dma_write_from_globalmem_to_host(uint64_t gm_src_addr,
                                                              uint64_t dst_pa,
                                                              size_t length,
                                                              DmaCallback cb,
                                                              DmaRequestKind kind)
{
    struct TransferState {
        uint64_t gmSrc = 0;
        uint64_t hostDst = 0;
        size_t length = 0;
        size_t offset = 0;
        DmaCallback callback;
    };

    if (!cb || length == 0 || gm_src_addr < baseAddr ||
        gm_src_addr - baseAddr > size || length > size - (gm_src_addr - baseAddr)) {
        if (cb) {
            cb(false);
        }
        return;
    }

    auto state = std::make_shared<TransferState>();
    state->gmSrc = gm_src_addr;
    state->hostDst = dst_pa;
    state->length = length;
    state->callback = std::move(cb);
    auto pump = std::make_shared<std::function<void()>>();
    *pump = [this, state, pump, kind]() {
        const size_t remaining = state->length - state->offset;
        const size_t chunkBytes = std::min<size_t>(remaining, localAccessMaxRequestBytes_);
        const uint64_t localTag = nextLocalDmaTag_++;
        const bool accepted = localReadAsync(
            state->gmSrc + state->offset,
            chunkBytes,
            LocalMemoryClient::Dma,
            localTag,
            [this, state, pump, chunkBytes, kind](bool ok,
                                            uint64_t,
                                            const std::vector<uint8_t>& data) {
                if (!ok || data.size() != chunkBytes) {
                    auto callback = std::move(state->callback);
                    *pump = {};
                    callback(false);
                    return;
                }
                dma_write_to_host(
                    state->hostDst + state->offset,
                    chunkBytes,
                    data,
                    [state, pump, chunkBytes](bool writeOk) {
                        if (!writeOk) {
                            auto callback = std::move(state->callback);
                            *pump = {};
                            callback(false);
                            return;
                        }
                        state->offset += chunkBytes;
                        if (state->offset == state->length) {
                            auto callback = std::move(state->callback);
                            *pump = {};
                            callback(true);
                            return;
                        }
                        (*pump)();
                    },
                    kind);
            });
        if (!accepted) {
            auto callback = std::move(state->callback);
            *pump = {};
            callback(false);
        }
    };
    (*pump)();
}

uint64_t GlobalMemoryImplement::allocate_dma_completion_token()
{
    const uint64_t token = next_dma_completion_token++;
    dma_completion_tokens[token] = false;
    return token;
}

uint64_t GlobalMemoryImplement::dma_write_to_host_async(uint64_t dst_pa, size_t length, const std::vector<uint8_t>& data)
{
    const uint64_t token = allocate_dma_completion_token();
    dma_write_to_host_impl(dst_pa, length, data, nullptr, token);
    if (length == 0) {
        dma_completion_tokens[token] = true;
    }
    return token;
}

uint64_t GlobalMemoryImplement::dma_read_from_host_to_globalmem_async(uint64_t src_pa, size_t length, uint64_t gm_dst_addr,
                                                                      DmaRequestKind kind)
{
    const uint64_t token = allocate_dma_completion_token();
    dma_read_from_host_to_globalmem_impl(src_pa, length, gm_dst_addr, nullptr, token, kind);
    if (length == 0) {
        dma_completion_tokens[token] = true;
    }
    return token;
}

bool GlobalMemoryImplement::dma_completion_done(uint64_t token) const
{
    if (token == 0) {
        return true;
    }
    auto it = dma_completion_tokens.find(token);
    return it != dma_completion_tokens.end() && it->second;
}

void GlobalMemoryImplement::dma_completion_retire(uint64_t token)
{
    if (token == 0) {
        return;
    }
    dma_completion_tokens.erase(token);
}

bool GlobalMemoryImplement::beginAttentionGeneration(uint64_t generation)
{
    return attentionClusterEnable_ && attentionGenerationFence_.begin(generation);
}

bool GlobalMemoryImplement::attentionGenerationDrained(uint64_t generation) const
{
    const auto dmaMatches = [generation](const auto& item) {
        return item.second.ownerGeneration == generation;
    };
    const auto localMatches = [generation](const auto& item) {
        return item.second.ownerGeneration == generation;
    };
    const auto landingMatches = [generation](const auto& item) {
        return item.second.op.ownerGeneration == generation;
    };
    return std::none_of(dma_pending.begin(), dma_pending.end(), dmaMatches) &&
        std::none_of(localAccessPending_.begin(), localAccessPending_.end(),
                     localMatches) &&
        std::none_of(pendingDmaLandings_.begin(), pendingDmaLandings_.end(),
                     landingMatches);
}

bool GlobalMemoryImplement::retireAttentionGeneration(uint64_t generation)
{
    return attentionGenerationDrained(generation) &&
        attentionGenerationFence_.retire(generation);
}

uint32_t GlobalMemoryImplement::cancelAttentionGeneration(uint64_t generation)
{
    uint32_t cancelled = 0;
    attentionGenerationFence_.cancel(generation);
    size_t pendingRemoteWrites = 0;
    for (const auto& item : dma_pending) {
        if (item.second.ownerGeneration == generation &&
            item.second.kind == PendingDmaOp::WRITE_TO_HOST) {
            ++pendingRemoteWrites;
        }
    }
    AttentionGenerationCancelRegistry::cancel(
        static_cast<uint64_t>(core_id), generation, pendingRemoteWrites);
    const uint64_t generationToken = AttentionGenerationCancelRegistry::token(
        static_cast<uint64_t>(core_id), generation);
    for (auto it = send_retry_queue.begin(); it != send_retry_queue.end();) {
        auto* event = dynamic_cast<NetworkDataEvent*>((*it)->inspectPayload());
        if (event != nullptr && event->getType() == NetworkDataEvent::DMA_WRITE &&
            event->getOwnerGenerationToken() == generationToken) {
            delete *it;
            it = send_retry_queue.erase(it);
            AttentionGenerationCancelRegistry::discardToken(generationToken);
        } else {
            ++it;
        }
    }
    for (auto it = dma_pending.begin(); it != dma_pending.end();) {
        if (it->second.ownerGeneration == generation) {
            it = dma_pending.erase(it);
            ++cancelled;
        } else {
            ++it;
        }
    }
    for (auto it = pendingDmaLandings_.begin();
         it != pendingDmaLandings_.end();) {
        if (it->second.op.ownerGeneration == generation) {
            it = pendingDmaLandings_.erase(it);
            ++cancelled;
        } else {
            ++it;
        }
    }
    for (auto& item : localAccessPending_) {
        PendingLocalAccess& request = item.second;
        if (request.ownerGeneration == generation && !request.cancelled) {
            request.cancelled = true;
            ++cancelled;
        }
    }
    issue_pending_dma_read_window();
    return cancelled;
}

void GlobalMemoryImplement::dma_write_to_host_impl(uint64_t dst_pa, size_t length, const std::vector<uint8_t>& data, DmaCallback cb,
                                                   uint64_t completion_token, DmaRequestKind kind)
{
    if (!link_control) {
        output->fatal(CALL_INFO, -1,
                      "GlobalMemory '%s' DMA operation called but 'link_control' is not wired.\n",
                      getName().c_str());
    }
    if (data.size() != length) {
        output->fatal(CALL_INFO, -1,
                      "GlobalMemory '%s' dma_write_to_host length mismatch: length=%zu data.size=%zu\n",
                      getName().c_str(), length, data.size());
    }

    // Implement burst/chunking (configurable bytes per chunk)
    const size_t kBurst = static_cast<size_t>(dma_burst_bytes);
    uint64_t seq_addr = 0;
    uint64_t flag_addr = 0;
    get_dma_flag_addrs(true, 0, seq_addr, flag_addr);
    uint64_t seq_value = read_u64_from_storage(seq_addr);
    size_t remaining = length;
    size_t offset = 0;
    std::shared_ptr<PendingDmaOp::DmaContext> ctx;
    if (remaining > kBurst) {
        ctx = std::make_shared<PendingDmaOp::DmaContext>();
        ctx->host_base = dst_pa;
        ctx->gm_base = 0;
        ctx->total_len = length;
        ctx->burst_size = kBurst;
        ctx->remaining = (length + kBurst - 1) / kBurst;
        ctx->ok = true;
        ctx->cb = cb;
        ctx->completion_enabled = true;
        ctx->completion_flag_addr = flag_addr;
        ctx->completion_value = seq_value;
        ctx->completion_token = completion_token;
    }

    while (remaining > 0) {
        size_t xfer = remaining < kBurst ? remaining : kBurst;
        dma_write_issue_count++;
        dma_write_bytes_total += xfer;
        dma_wait_count++;
        std::vector<uint8_t> chunk(data.begin() + offset, data.begin() + offset + xfer);

        output->verbose(CALL_INFO, 1, 2,
                        "dma_write_to_host: dst_pa=0x%" PRIx64 " len=%zu\n",
                        dst_pa + offset, xfer);
        if (gm_dump_data && !chunk.empty()) {
            const size_t dump_len = chunk.size() < 64 ? chunk.size() : 64;
            char hexbuf[3 * 64 + 1];
            for (size_t i = 0; i < dump_len; ++i) {
                std::snprintf(hexbuf + i * 3, 4, "%02X ", chunk[i]);
            }
            hexbuf[3 * dump_len] = '\0';
            output->verbose(CALL_INFO, 10, 2,
                            "dma_write_to_host: data[0..%zu]=%s\n",
                            dump_len - 1, hexbuf);
        }

        // 创建网络请求包
        auto* req = new SST::Interfaces::SimpleNetwork::Request();
        req->src = network_id;

        // 获取 MemNIC 的 endpoint ID
        int memnic_ep = getMemNicEndpointId(dst_pa + offset);
        if (memnic_ep != -1) {
            req->dest = memnic_ep;
        } else {
            req->dest = getDmaTargetRouter(dst_pa + offset);
        }

        // 记录 pending 操作（使用 host_addr 作为 key，用于 DMA_WRITE_COMPLETE 查找）
        PendingDmaOp op;
        op.kind = PendingDmaOp::WRITE_TO_HOST;
        op.host_addr = dst_pa + offset;
        op.gm_dst_addr = 0;
        op.cb = ctx ? DmaCallback() : cb;  // burst 时不设置回调，单次则设置
        op.length = xfer;
        op.ctx = ctx;
        op.completion_enabled = ctx ? false : true;
        op.completion_flag_addr = flag_addr;
        op.completion_value = seq_value;
        op.completion_token = ctx ? ctx->completion_token : completion_token;
        op.dmaRequestKind = kind;
        op.ownerGeneration = attentionClusterEnable_
            ? attentionGenerationFence_.active() : 0;
        op.ownerGenerationToken = op.ownerGeneration == 0 ? 0 :
            AttentionGenerationCancelRegistry::token(
                static_cast<uint64_t>(core_id), op.ownerGeneration);
        // The architecture clock is 1 GHz, so nanoseconds are architectural cycles.
        op.issue_cycle = getCurrentSimTimeNano();

        if (dma_write_first_issue_cycle == 0) {
            dma_write_first_issue_cycle = op.issue_cycle;
        }
        dma_write_last_issue_cycle = op.issue_cycle;

        req->vn = write_vn;
        req->size_in_bits = (sizeof(dst_pa) + sizeof(xfer) + xfer) * 8;

        // 创建 DMA_WRITE 负载
        auto* payload = new NetworkDataEvent(NetworkDataEvent::DMA_WRITE, dst_pa + offset, xfer, chunk,
                                             0, -1, 0, 0, op.request_id, op.dmaRequestKind);
        payload->setOwnerGenerationToken(op.ownerGenerationToken);
        req->givePayload(payload);

        const uint64_t pending_key = op.host_addr;
        auto inserted = dma_pending.emplace(pending_key, op);
        if (!inserted.second) {
            output->output("GlobalMemory: Warning - duplicate pending DMA_WRITE key addr=0x%" PRIx64
                           " (core=%d), replacing existing entry\n",
                           pending_key, core_id);
            inserted.first->second = op;
        }

        // 发送请求
        try_send_or_queue(req, "dma_write_to_host");

        remaining -= xfer;
        offset += xfer;
    }

    // 单次传输时也需要等待 DMA_WRITE_COMPLETE（对称实现）
    if (!ctx) {
        // 不立即调用回调，等待 DMA_WRITE_COMPLETE 通知
    }
}

void GlobalMemoryImplement::dma_read_from_host_to_globalmem(uint64_t src_pa, size_t length, uint64_t gm_dst_addr, DmaCallback cb,
                                                            DmaRequestKind kind,
                                                            const DmaConsumerMetadata& consumer)
{
    dma_read_from_host_to_globalmem_impl(src_pa, length, gm_dst_addr, cb, 0, kind, consumer);
}

void GlobalMemoryImplement::dma_read_from_host_to_globalmem_impl(uint64_t src_pa, size_t length, uint64_t gm_dst_addr, DmaCallback cb,
                                                                 uint64_t completion_token, DmaRequestKind kind,
                                                                 const DmaConsumerMetadata& consumer)
{
    if (!link_control) {
        output->fatal(CALL_INFO, -1,
                      "GlobalMemory '%s' dma_read_from_host_to_globalmem called but 'link_control' is not wired.\n",
                      getName().c_str());
    }
    // Validate GM destination range upfront.
    if (gm_dst_addr < baseAddr || (gm_dst_addr + length) > (baseAddr + size)) {
        output->fatal(CALL_INFO, -1,
                      "GlobalMemory '%s' dma_read_from_host_to_globalmem destination out of range: dst=0x%" PRIx64 " len=%zu base=0x%" PRIx64 " size=0x%" PRIx64 "\n",
                      getName().c_str(), gm_dst_addr, length, baseAddr, size);
    }

    const size_t kBurst = static_cast<size_t>(dma_burst_bytes); // bytes per chunk
    uint64_t seq_addr = 0;
    uint64_t flag_addr = 0;
    const uint64_t read_slot_addr = baseAddr + size - READ_SLOT_SELECT_OFFSET;
    const uint64_t read_slot_raw = read_u64_from_storage(read_slot_addr);
    const uint8_t read_slot = static_cast<uint8_t>(read_slot_raw & 0x1ULL);
    get_dma_flag_addrs(false, read_slot, seq_addr, flag_addr);
    uint64_t seq_value = read_u64_from_storage(seq_addr);
    size_t remaining = length;
    size_t offset = 0;
    std::shared_ptr<PendingDmaOp::DmaContext> ctx;
    if (remaining > kBurst) {
        ctx = std::make_shared<PendingDmaOp::DmaContext>();
        ctx->host_base = src_pa;
        ctx->gm_base = gm_dst_addr;
        ctx->total_len = length;
        ctx->burst_size = kBurst;
        ctx->remaining = (length + kBurst - 1) / kBurst;
        ctx->ok = true;
        ctx->cb = cb;
        ctx->completion_enabled = true;
        ctx->completion_flag_addr = flag_addr;
        ctx->completion_value = seq_value;
        ctx->completion_token = completion_token;
    }

    while (remaining > 0) {
        size_t xfer = remaining < kBurst ? remaining : kBurst;
        dma_read_issue_count++;
        dma_read_bytes_total += xfer;
        dma_wait_count++;

        output->verbose(CALL_INFO, 1, 2,
                        "dma_read_from_host_to_globalmem: src_pa=0x%" PRIx64 " len=%zu -> gm_dst=0x%" PRIx64 "\n",
                        src_pa + offset, xfer, gm_dst_addr + offset);

        // 记录 pending 操作
        PendingDmaOp op;
        op.kind = PendingDmaOp::READ_TO_GM;
        op.request_id = next_dma_request_id++;
        op.host_addr = src_pa + offset;
        op.gm_dst_addr = gm_dst_addr + offset;
        op.dmaRequestKind = kind;
        op.dmaConsumer = consumer;
        op.ownerGeneration = attentionClusterEnable_
            ? attentionGenerationFence_.active() : 0;
        op.ownerGenerationToken = op.ownerGeneration == 0 ? 0 :
            AttentionGenerationCancelRegistry::token(
                static_cast<uint64_t>(core_id), op.ownerGeneration);
        op.cb = ctx ? DmaCallback() : cb;
        op.length = xfer;
        op.ctx = ctx;
        op.completion_enabled = ctx ? false : true;
        op.completion_flag_addr = flag_addr;
        op.completion_value = seq_value;
        op.completion_token = ctx ? ctx->completion_token : completion_token;
        op.retry_ticks_left = dma_read_retry_ticks;
        op.retry_attempts = 0;
        op.request_issued = false;
        const uint64_t pending_key = op.request_id;
        auto inserted = dma_pending.emplace(pending_key, op);
        if (!inserted.second) {
            output->output("GlobalMemory: Warning - duplicate pending DMA_READ req=%" PRIu64
                           " (core=%d), replacing existing entry\n",
                           pending_key, core_id);
            inserted.first->second = op;
        }

        remaining -= xfer;
        offset += xfer;
    }

    issue_pending_dma_read_window();
    schedule_dma_retry_event();

    // 单次传输时设置回调（burst 时由 handleDmaReceives 处理）
    if (!ctx) {
        // 对于单次传输，我们等待 DMA_READ_COMPLETE 消息
        // 这里暂时不做处理，让 handleDmaReceives 处理完成通知
    }
}

void GlobalMemoryImplement::dma_update_consumer_progress(
    uint64_t firstHostAddr, uint64_t secondHostAddr,
    const DmaConsumerMetadata& consumer)
{
    if (!link_control || consumer.valid == 0) return;
    std::vector<int> destinations;
    for (uint64_t hostAddr : {firstHostAddr, secondHostAddr}) {
        int destination = getMemNicEndpointId(hostAddr);
        if (destination == -1) destination = getDmaTargetRouter(hostAddr);
        if (std::find(destinations.begin(), destinations.end(), destination) != destinations.end()) {
            continue;
        }
        destinations.push_back(destination);
        auto* req = new SST::Interfaces::SimpleNetwork::Request();
        req->src = network_id;
        req->dest = destination;
        req->vn = request_vn;
        req->size_in_bits = (sizeof(DmaConsumerMetadata) + sizeof(hostAddr)) * 8;
        req->givePayload(new NetworkDataEvent(
            NetworkDataEvent::DMA_CONSUMER_PROGRESS, hostAddr, 0,
            std::vector<uint8_t>(), 0, network_id, 0, 0, 0,
            DmaRequestKind::AttentionKvPrefetch, consumer));
        try_send_or_queue(req, "dma_consumer_progress");
    }
}

// 当网络收到数据包时由 SimpleNetwork 调用的回调函数
bool GlobalMemoryImplement::handle_receives(int vn) {
    if (!send_retry_queue.empty()) {
        flush_send_retry_queue();
    }

    bool handled_any = false;
    // 不断从网络接口提取已到达的请求进行处理
    SST::Interfaces::SimpleNetwork::Request* req = nullptr;
    while ((req = link_control->recv(vn)) != nullptr) {
        handled_any = true;
        // 获取并转换请求中的事件负载
        Event* ev_base = req->takePayload();
        if (auto* reductionEvent = dynamic_cast<ReductionTransportEvent*>(ev_base)) {
            reduction_receive_count++;
            statReductionReceived_->addData(1);
            if (reduction_message_handler) {
                reduction_message_handler(reductionEvent->getMessage());
            }
            delete reductionEvent;
            delete req;
            continue;
        }
        NetworkDataEvent* ev = dynamic_cast<NetworkDataEvent*>(ev_base);
        if (!ev) {
            // 收到非预期的事件类型，忽略处理
            output->output("GlobalMemory: Received an unknown event type, ignoring.\n");
            if (ev_base) delete ev_base;
            delete req;
            continue;
        }

        const uint64_t addr = ev->getAddr();
        const size_t length = ev->getLength();

        if (dma_trace) {
            output->output("GlobalMemory core=%d recv type=%d req=%" PRIu64 " addr=0x%" PRIx64
                           " len=%zu from=%" PRI_NID " vn=%d\n",
                           core_id, static_cast<int>(ev->getType()), ev->getRequestId(),
                           addr, length, req->src, vn);
        }

        auto check_range = [&](const char* opName) {
            const bool inRange = (addr >= baseAddr) && ((addr + length) <= (baseAddr + size));
            if (!inRange) {
                output->fatal(CALL_INFO, -1,
                              "GlobalMemory %s (core %d, network ID %d) received %s request out of range.\n"
                              "  addr=0x%" PRIx64 " length=%zu base=[0x%" PRIx64 ", 0x%" PRIx64 ") size=0x%" PRIx64 "\n"
                              "  requester network ID=%" PRI_NID " payload type=%d\n",
                              getName().c_str(), core_id, network_id, opName,
                              addr, length, baseAddr, baseAddr + size, size,
                              req->src, static_cast<int>(ev->getType()));
            }
        };

        if (ev->getType() == NetworkDataEvent::WRITE) {
            // 处理 WRITE 类型请求: 将数据写入本地内存
            auto req_it = request_pending.find(ev->getRequestId());
            if (req_it != request_pending.end()) {
                PendingReadRequest& pending = req_it->second;
                const uint64_t writeAddr = pending.total_len > 0 ? ev->getAddr() : pending.gm_dst_addr;
                const uint64_t flagAddr = pending.completion_flag_addr;
                const uint64_t flagValue = pending.completion_value;
                wr_to_globalmem(writeAddr, ev->getLength(), ev->getData());
                pending.received_len += ev->getLength();
                if (pending.total_len == 0 || pending.received_len >= pending.total_len) {
                    const uint64_t completeCycle = getCurrentSimCycle();
                    if (completeCycle >= pending.submit_cycle) {
                        const uint64_t submitReadyCycles = completeCycle - pending.submit_cycle;
                        request_read_submit_ready_samples++;
                        request_read_submit_ready_cycles_sum += submitReadyCycles;
                        if (submitReadyCycles > request_read_submit_ready_cycles_max) {
                            request_read_submit_ready_cycles_max = submitReadyCycles;
                        }
                    }
                    write_u64_to_storage(flagAddr, flagValue);
                    request_pending.erase(req_it);
                    notifyCtrlReadComplete(ev->getRequestId());
                }
                output->verbose(CALL_INFO, 1, 2,
                                "handle_receives: WRITE completed req=%" PRIu64 " dst=0x%" PRIx64
                                " flag=0x%" PRIx64 " val=%" PRIu64 " len=%zu\n",
                                ev->getRequestId(), writeAddr,
                                flagAddr, flagValue,
                                ev->getLength());
                delete ev;
                delete req;
                continue;
            }

            check_range("WRITE");
            wr_to_globalmem(ev->getAddr(), ev->getLength(), ev->getData());
            auto it = read_pending.find(ev->getAddr());
            if (it != read_pending.end()) {
                write_u64_to_storage(it->second.completion_flag_addr, it->second.completion_value);
                read_pending.erase(it);
            } else if (ev->getCompletionFlagAddr() != 0) {
                write_u64_to_storage(ev->getCompletionFlagAddr(), ev->getCompletionValue());
            }
            // 对于写请求，不需要发送回复
        }
        else if (ev->getType() == NetworkDataEvent::READ) {
            // 处理 READ 类型请求: 从本地内存读取数据并发送回复
            check_range("READ");
            std::vector<uint8_t> readData;
            rd_from_globalmem(ev->getAddr(), ev->getLength(), readData);
            output->verbose(CALL_INFO, 1, 2,
                            "handle_receives: READ addr=0x%" PRIx64 " len=%zu return=0x%" PRIx64 " -> sending reply to ep %" PRI_NID "\n",
                            ev->getAddr(), ev->getLength(), ev->getReturnAddr(), req->src);
            const size_t total_size_bytes = sizeof(ev->getAddr()) + sizeof(ev->getLength()) + readData.size();
            const uint64_t responseAddr = ev->getReturnAddr();
            const int responseEndpoint = ev->getReturnEndpoint();
            NetworkDataEvent* respEv = new NetworkDataEvent(
                NetworkDataEvent::WRITE,
                responseAddr,
                length,
                readData,
                responseAddr,
                responseEndpoint,
                ev->getCompletionFlagAddr(),
                ev->getCompletionValue(),
                ev->getRequestId());
            respEv->setOwnerGenerationToken(ev->getOwnerGenerationToken());
            SST::Interfaces::SimpleNetwork::Request* respReq = new SST::Interfaces::SimpleNetwork::Request();
            respReq->src = network_id;
            respReq->dest = (responseEndpoint >= 0) ? static_cast<SST::Interfaces::SimpleNetwork::nid_t>(responseEndpoint) : req->src;
            respReq->size_in_bits = total_size_bytes * 8;
            respReq->vn = response_vn;
            respReq->givePayload(respEv);
            try_send_or_queue(respReq, "handle_receives:READ_reply");
        }
        else if (ev->getType() == NetworkDataEvent::DMA_WRITE) {
            // DMA 写请求：直接写入本地内存
            check_range("DMA_WRITE");
            if (!AttentionGenerationCancelRegistry::cancelledToken(
                    ev->getOwnerGenerationToken())) {
                wr_to_globalmem(ev->getAddr(), ev->getLength(), ev->getData());
            }
            output->verbose(CALL_INFO, 1, 2,
                            "handle_receives: DMA_WRITE addr=0x%" PRIx64 " len=%zu from_ep=%" PRI_NID "\n",
                            ev->getAddr(), ev->getLength(), req->src);
        }
        else if (ev->getType() == NetworkDataEvent::DMA_READ_COMPLETE) {
            // DMA 读完成通知：写入数据到本地 GlobalMemory 并调用回调
            if (dma_trace) {
                output->verbose(CALL_INFO, 0, 2,
                                "handle_receives: DMA_READ_COMPLETE addr=0x%" PRIx64 " len=%zu\n",
                                ev->getAddr(), ev->getLength());
            }
            if (!ev->getData().empty()) {
                const size_t dump_len = ev->getData().size() < 64 ? ev->getData().size() : 64;
                char hexbuf[3 * 64 + 1];
                for (size_t i = 0; i < dump_len; ++i) {
                    std::snprintf(hexbuf + i * 3, 4, "%02X ", ev->getData()[i]);
                }
                hexbuf[3 * dump_len] = '\0';
            if (gm_dump_data) {
                output->verbose(CALL_INFO, 10, 2,
                                "DMA_READ_COMPLETE data[0..%zu]=%s\n",
                                dump_len - 1, hexbuf);
            }
            }

            auto it = dma_pending.find(ev->getRequestId());
            if (it != dma_pending.end()) {
                PendingDmaOp op = it->second;
                dma_pending.erase(it);

                if (op.kind == PendingDmaOp::READ_TO_GM) {
                    if (op.request_sent) {
                        const uint64_t completeCycle = getCurrentSimCycle();
                        const uint64_t last_try_rtt_ticks = dma_retry_tick_counter - op.last_send_tick;
                        dma_read_rtt_samples++;
                        dma_read_rtt_ticks_sum += last_try_rtt_ticks;
                        if (last_try_rtt_ticks > dma_read_rtt_ticks_max) {
                            dma_read_rtt_ticks_max = last_try_rtt_ticks;
                        }

                        const uint64_t e2e_rtt_ticks = op.first_send_seen
                                                           ? (dma_retry_tick_counter - op.first_send_tick)
                                                           : last_try_rtt_ticks;
                        dma_read_e2e_rtt_samples++;
                        dma_read_e2e_rtt_ticks_sum += e2e_rtt_ticks;
                        if (e2e_rtt_ticks > dma_read_e2e_rtt_ticks_max) {
                            dma_read_e2e_rtt_ticks_max = e2e_rtt_ticks;
                        }

                        if (completeCycle >= op.last_send_cycle) {
                            const uint64_t strictRttCycles = completeCycle - op.last_send_cycle;
                            dma_read_strict_rtt_samples++;
                            dma_read_strict_rtt_cycles_sum += strictRttCycles;
                            if (strictRttCycles > dma_read_strict_rtt_cycles_max) {
                                dma_read_strict_rtt_cycles_max = strictRttCycles;
                            }
                        }
                        if (op.first_send_seen && completeCycle >= op.first_send_cycle) {
                            const uint64_t strictE2eRttCycles = completeCycle - op.first_send_cycle;
                            dma_read_strict_e2e_rtt_samples++;
                            dma_read_strict_e2e_rtt_cycles_sum += strictE2eRttCycles;
                            if (strictE2eRttCycles > dma_read_strict_e2e_rtt_cycles_max) {
                                dma_read_strict_e2e_rtt_cycles_max = strictE2eRttCycles;
                            }
                        }
                    }
                    completeDmaReadToLocalMemory(std::move(op), ev->getData());
                }
            } else {
                auto req_it = request_pending.find(ev->getRequestId());
                if (req_it != request_pending.end()) {
                    PendingReadRequest& pending = req_it->second;
                    const uint64_t writeAddr = pending.total_len > 0 ? ev->getAddr() : pending.gm_dst_addr;
                    const uint64_t flagAddr = pending.completion_flag_addr;
                    const uint64_t flagValue = pending.completion_value;
                    wr_to_globalmem(writeAddr, ev->getLength(), ev->getData());
                    pending.received_len += ev->getLength();
                    if (dma_trace) {
                        fprintf(stderr, "[GlobalMemory] TRACE_REQ_WORKER_RECV_CHUNK cycle=%" PRIu64
                                        " req=%" PRIu64 " addr=0x%" PRIx64 " len=%zu received=%zu total=%zu\n",
                                getCurrentSimCycle(), ev->getRequestId(), writeAddr, ev->getLength(),
                                pending.received_len, pending.total_len);
                    }
                    if (pending.total_len == 0 || pending.received_len >= pending.total_len) {
                        const uint64_t completeCycle = getCurrentSimCycle();
                        if (completeCycle >= pending.submit_cycle) {
                            const uint64_t submitReadyCycles = completeCycle - pending.submit_cycle;
                            request_read_submit_ready_samples++;
                            request_read_submit_ready_cycles_sum += submitReadyCycles;
                            if (submitReadyCycles > request_read_submit_ready_cycles_max) {
                                request_read_submit_ready_cycles_max = submitReadyCycles;
                            }
                        }
                        write_u64_to_storage(flagAddr, flagValue);
                        if (dma_trace) {
                            fprintf(stderr, "[GlobalMemory] TRACE_REQ_PENDING_CLEAR cycle=%" PRIu64
                                            " req=%" PRIu64 " received=%zu total=%zu flag=0x%" PRIx64
                                            " val=%" PRIu64 "\n",
                                    getCurrentSimCycle(), ev->getRequestId(), pending.received_len,
                                    pending.total_len, flagAddr, flagValue);
                        }
                        dma_read_completion_count++;
                        request_pending.erase(req_it);
                        notifyCtrlReadComplete(ev->getRequestId());
                    }
                    output->verbose(CALL_INFO, 1, 2,
                                    "handle_receives: DMA_READ_COMPLETE req=%" PRIu64 " dst=0x%" PRIx64
                                    " flag=0x%" PRIx64 " val=%" PRIu64 " len=%zu\n",
                                    ev->getRequestId(), writeAddr,
                                    flagAddr, flagValue,
                                    ev->getLength());
                } else if (ev->getCompletionFlagAddr() != 0) {
                    check_range("DMA_READ_COMPLETE(direct)");
                    wr_to_globalmem(ev->getAddr(), ev->getLength(), ev->getData());
                    write_u64_to_storage(ev->getCompletionFlagAddr(), ev->getCompletionValue());
                    dma_read_completion_count++;
                } else {
                    dma_read_completion_no_pending_count++;
                    output->verbose(CALL_INFO, 1, 2,
                                    "handle_receives: No pending DMA operation found (requestId=%" PRIu64 ")\n",
                                    ev->getRequestId());
                }
            }
            issue_pending_dma_read_window();
        }
        else if (ev->getType() == NetworkDataEvent::DMA_WRITE_COMPLETE) {
            // DMA 写完成通知：调用回调（对称实现）
            AttentionGenerationCancelRegistry::discardToken(
                ev->getOwnerGenerationToken());
            output->verbose(CALL_INFO, 1, 2,
                            "handle_receives: DMA_WRITE_COMPLETE addr=0x%" PRIx64 "\n",
                            ev->getAddr());

            auto it = dma_pending.find(ev->getAddr());
            if (it != dma_pending.end()) {
                PendingDmaOp op = it->second;
                dma_pending.erase(it);

                if (op.kind == PendingDmaOp::WRITE_TO_HOST) {
                    dma_write_completion_count++;
                    const uint64_t complete_cycle = getCurrentSimTimeNano();
                    const uint64_t rtt_cycles = complete_cycle >= op.issue_cycle
                                                    ? complete_cycle - op.issue_cycle
                                                    : 0;
                    dma_write_rtt_samples++;
                    dma_write_rtt_cycles_sum += rtt_cycles;
                    dma_write_rtt_cycles_max = std::max(dma_write_rtt_cycles_max, rtt_cycles);
                    dma_write_last_complete_cycle = std::max(
                        dma_write_last_complete_cycle, complete_cycle);
                    // 处理 burst 上下文
                    if (op.ctx) {
                        if (op.ctx->remaining > 0) op.ctx->remaining--;
                        if (op.ctx->remaining == 0) {
                            if (op.ctx->completion_enabled) {
                                write_u64_to_storage(op.ctx->completion_flag_addr, op.ctx->completion_value);
                            }
                            if (op.ctx->completion_token != 0) {
                                dma_completion_tokens[op.ctx->completion_token] = true;
                            }
                            if (op.ctx->cb) {
                                op.ctx->cb(op.ctx->ok);
                            }
                        }
                    } else {
                        if (op.completion_enabled) {
                            write_u64_to_storage(op.completion_flag_addr, op.completion_value);
                        }
                        if (op.completion_token != 0) {
                            dma_completion_tokens[op.completion_token] = true;
                        }
                        if (op.cb) {
                            op.cb(true);
                        }
                    }
                }
            } else {
                output->verbose(CALL_INFO, 1, 2,
                                "handle_receives: No pending DMA WRITE operation found\n");
            }
        }

        if (ev->getType() == NetworkDataEvent::WRITE) {
            output->verbose(CALL_INFO, 1, 2,
                            "handle_receives: WRITE addr=0x%" PRIx64 " len=%zu from_ep=%" PRI_NID "\n",
                            ev->getAddr(), ev->getLength(), req->src);
        }

        // 释放当前请求和事件对象占用的内存
        delete ev;
        delete req;
    }
    issue_pending_dma_read_window();
    if (!dma_pending.empty()) {
        schedule_dma_retry_event();
    }
    return handled_any;
}

// 根据物理地址计算目标内存节点的索引 (0-3)
uint8_t GlobalMemoryImplement::getDmaTargetNode(uint64_t phys_addr) {
    return static_cast<uint8_t>(phys_addr / memNodeSize);
}

// 根据物理地址计算目标内存节点的 Router ID
uint8_t GlobalMemoryImplement::getDmaTargetRouter(uint64_t phys_addr) {
    uint64_t node_idx = static_cast<uint64_t>(getDmaTargetNode(phys_addr));
    if (node_idx >= memoryRouters.size()) {
        output->fatal(CALL_INFO, -1,
                      "GlobalMemory '%s' address 0x%" PRIx64
                      " maps to node index %" PRIu64 " but memoryRouters has %zu entries\n",
                      getName().c_str(), phys_addr, node_idx, memoryRouters.size());
    }
    return memoryRouters[node_idx];
}

// 根据物理地址获取 MemNIC 的 endpoint ID
int GlobalMemoryImplement::getMemNicEndpointId(uint64_t phys_addr) {
    uint8_t node_idx = getDmaTargetNode(phys_addr);

    // DirectoryController 的名称格式
    std::string dirctrl_name = "dirctrl_" + std::to_string(node_idx);

    // 直接查找 DirectoryController
    auto it = IDMap.find(dirctrl_name);
    if (it != IDMap.end()) {
        return it->second;
    }

    // 尝试查找 MemNIC 的名称（可能是 "dirctrl_N.highlink" 格式）
    std::string memnic_name = dirctrl_name + ".highlink";
    it = IDMap.find(memnic_name);
    if (it != IDMap.end()) {
        return it->second;
    }

    // 如果找不到，返回 -1 表示需要使用路由器编号
    output->verbose(CALL_INFO, 1, 2,
                    "getMemNicEndpointId: MemNIC not found for node %d, searching IDMap...\n",
                    node_idx);
    for (const auto& entry : IDMap) {
        output->verbose(CALL_INFO, 1, 2, "  IDMap entry: %s -> %d\n",
                        entry.first.c_str(), entry.second);
    }

    return -1;
}

// 处理来自 DMA 网络的请求（内存节点端）
bool GlobalMemoryImplement::handleDmaReceives(int vn) {
    bool handled_any = false;
    SST::Interfaces::SimpleNetwork::Request* req = nullptr;
    while ((req = link_control->recv(vn)) != nullptr) {
        handled_any = true;
        Event* ev_base = req->takePayload();
        NetworkDataEvent* ev = dynamic_cast<NetworkDataEvent*>(ev_base);
        if (!ev) {
            output->output("GlobalMemory: Received unknown DMA event type, ignoring.\n");
            if (ev_base) delete ev_base;
            delete req;
            continue;
        }

        const uint64_t addr = ev->getAddr();
        const size_t length = ev->getLength();

        // 验证地址范围
        if (addr < baseAddr || (addr + length) > (baseAddr + size)) {
            output->fatal(CALL_INFO, -1,
                          "GlobalMemory core %d received DMA request out of range.\n"
                          "  addr=0x%" PRIx64 " length=%zu base=[0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                          core_id, addr, length, baseAddr, baseAddr + size);
        }

        if (ev->getType() == NetworkDataEvent::DMA_WRITE) {
            // DMA 写请求：写入本地内存
            output->verbose(CALL_INFO, 1, 2,
                            "handleDmaReceives: DMA_WRITE addr=0x%" PRIx64 " len=%zu from_ep=%" PRI_NID "\n",
                            addr, length, req->src);
            if (!AttentionGenerationCancelRegistry::cancelledToken(
                    ev->getOwnerGenerationToken())) {
                wr_to_globalmem(addr, length, ev->getData());
            }

            // 发送 DMA_WRITE_COMPLETE 回源（对称实现）
            NetworkDataEvent* respEv = new NetworkDataEvent(
                NetworkDataEvent::DMA_WRITE_COMPLETE, ev->getAddr(), length, std::vector<uint8_t>(),
                ev->getAddr(), req->src, 0, 0, ev->getRequestId());
            respEv->setOwnerGenerationToken(ev->getOwnerGenerationToken());
            auto* respReq = new SST::Interfaces::SimpleNetwork::Request();
            respReq->src = network_id;
            respReq->dest = req->src;
            respReq->vn = 0;
            respReq->size_in_bits = (sizeof(addr) + sizeof(length)) * 8;
            respReq->givePayload(respEv);

            try_send_or_queue(respReq, "handleDmaReceives:DMA_WRITE_COMPLETE_reply");
        }
        else if (ev->getType() == NetworkDataEvent::READ) {
            // DMA 读请求：从本地内存读取数据并发送 DMA_READ_COMPLETE 回复
            output->verbose(CALL_INFO, 1, 2,
                            "handleDmaReceives: DMA_READ addr=0x%" PRIx64 " len=%zu return=0x%" PRIx64 " from_ep=%" PRI_NID "\n",
                            addr, length, ev->getReturnAddr(), req->src);

            std::vector<uint8_t> readData;
            rd_from_globalmem(addr, length, readData);

            NetworkDataEvent* respEv = new NetworkDataEvent(
                NetworkDataEvent::DMA_READ_COMPLETE,
                ev->getReturnAddr(),
                length,
                readData,
                ev->getReturnAddr(),
                ev->getReturnEndpoint(),
                ev->getCompletionFlagAddr(),
                ev->getCompletionValue(),
                ev->getRequestId());
            respEv->setOwnerGenerationToken(ev->getOwnerGenerationToken());
            auto* respReq = new SST::Interfaces::SimpleNetwork::Request();
            respReq->src = network_id;
            respReq->dest = (ev->getReturnEndpoint() >= 0)
                                ? static_cast<SST::Interfaces::SimpleNetwork::nid_t>(ev->getReturnEndpoint())
                                : req->src;
            respReq->vn = response_vn;
            respReq->size_in_bits = (sizeof(addr) + sizeof(length) + readData.size()) * 8;
            respReq->givePayload(respEv);

            try_send_or_queue(respReq, "handleDmaReceives:DMA_READ_COMPLETE_reply");
        }
        else if (ev->getType() == NetworkDataEvent::DMA_READ_COMPLETE) {
            // DMA 读完成通知：写入数据到本地 GlobalMemory 并调用回调
            if (dma_trace) {
                output->verbose(CALL_INFO, 0, 2,
                                "handleDmaReceives: DMA_READ_COMPLETE addr=0x%" PRIx64 " len=%zu\n",
                                addr, length);
            }
            if (!ev->getData().empty()) {
                const size_t dump_len = ev->getData().size() < 64 ? ev->getData().size() : 64;
                char hexbuf[3 * 64 + 1];
                for (size_t i = 0; i < dump_len; ++i) {
                    std::snprintf(hexbuf + i * 3, 4, "%02X ", ev->getData()[i]);
                }
                hexbuf[3 * dump_len] = '\0';
                if (gm_dump_data) {
                    output->verbose(CALL_INFO, 10, 2,
                                "DMA_READ_COMPLETE data[0..%zu]=%s\n",
                                dump_len - 1, hexbuf);
                }
            }

            // 查找对应的 pending 操作
            auto it = dma_pending.find(ev->getRequestId());
            if (it != dma_pending.end()) {
                PendingDmaOp op = it->second;
                dma_pending.erase(it);

                if (op.kind == PendingDmaOp::READ_TO_GM) {
                    dma_read_completion_count++;
                    if (op.request_sent) {
                        const uint64_t completeCycle = getCurrentSimCycle();
                        const uint64_t last_try_rtt_ticks = dma_retry_tick_counter - op.last_send_tick;
                        dma_read_rtt_samples++;
                        dma_read_rtt_ticks_sum += last_try_rtt_ticks;
                        if (last_try_rtt_ticks > dma_read_rtt_ticks_max) {
                            dma_read_rtt_ticks_max = last_try_rtt_ticks;
                        }

                        const uint64_t e2e_rtt_ticks = op.first_send_seen
                                                           ? (dma_retry_tick_counter - op.first_send_tick)
                                                           : last_try_rtt_ticks;
                        dma_read_e2e_rtt_samples++;
                        dma_read_e2e_rtt_ticks_sum += e2e_rtt_ticks;
                        if (e2e_rtt_ticks > dma_read_e2e_rtt_ticks_max) {
                            dma_read_e2e_rtt_ticks_max = e2e_rtt_ticks;
                        }

                        if (completeCycle >= op.last_send_cycle) {
                            const uint64_t strictRttCycles = completeCycle - op.last_send_cycle;
                            dma_read_strict_rtt_samples++;
                            dma_read_strict_rtt_cycles_sum += strictRttCycles;
                            if (strictRttCycles > dma_read_strict_rtt_cycles_max) {
                                dma_read_strict_rtt_cycles_max = strictRttCycles;
                            }
                        }
                        if (op.first_send_seen && completeCycle >= op.first_send_cycle) {
                            const uint64_t strictE2eRttCycles = completeCycle - op.first_send_cycle;
                            dma_read_strict_e2e_rtt_samples++;
                            dma_read_strict_e2e_rtt_cycles_sum += strictE2eRttCycles;
                            if (strictE2eRttCycles > dma_read_strict_e2e_rtt_cycles_max) {
                                dma_read_strict_e2e_rtt_cycles_max = strictE2eRttCycles;
                            }
                        }
                    }
                    // 写入数据到本地 GlobalMemory
                    wr_to_globalmem(op.gm_dst_addr, ev->getData().size(), ev->getData());

                    // 处理 burst 上下文
                    if (op.ctx) {
                        if (op.ctx->remaining > 0) op.ctx->remaining--;
                        if (op.ctx->remaining == 0) {
                            if (op.ctx->completion_enabled) {
                                write_u64_to_storage(op.ctx->completion_flag_addr, op.ctx->completion_value);
                            }
                            if (op.ctx->completion_token != 0) {
                                dma_completion_tokens[op.ctx->completion_token] = true;
                            }
                            if (op.ctx->cb) {
                                op.ctx->cb(op.ctx->ok);
                            }
                        }
                    } else {
                        if (op.completion_enabled) {
                            write_u64_to_storage(op.completion_flag_addr, op.completion_value);
                        }
                        if (op.completion_token != 0) {
                            dma_completion_tokens[op.completion_token] = true;
                        }
                        if (op.cb) {
                            op.cb(true);
                        }
                    }
                }
            } else {
                auto req_it = request_pending.find(ev->getRequestId());
                if (req_it != request_pending.end()) {
                    PendingReadRequest& pending = req_it->second;
                    const uint64_t writeAddr = pending.total_len > 0 ? ev->getAddr() : pending.gm_dst_addr;
                    const uint64_t flagAddr = pending.completion_flag_addr;
                    const uint64_t flagValue = pending.completion_value;
                    wr_to_globalmem(writeAddr, ev->getLength(), ev->getData());
                    pending.received_len += ev->getLength();
                    if (pending.total_len == 0 || pending.received_len >= pending.total_len) {
                        const uint64_t completeCycle = getCurrentSimCycle();
                        if (completeCycle >= pending.submit_cycle) {
                            const uint64_t submitReadyCycles = completeCycle - pending.submit_cycle;
                            request_read_submit_ready_samples++;
                            request_read_submit_ready_cycles_sum += submitReadyCycles;
                            if (submitReadyCycles > request_read_submit_ready_cycles_max) {
                                request_read_submit_ready_cycles_max = submitReadyCycles;
                            }
                        }
                        write_u64_to_storage(flagAddr, flagValue);
                        dma_read_completion_count++;
                        request_pending.erase(req_it);
                        notifyCtrlReadComplete(ev->getRequestId());
                    }
                    output->verbose(CALL_INFO, 1, 2,
                                    "handleDmaReceives: DMA_READ_COMPLETE req=%" PRIu64 " dst=0x%" PRIx64
                                    " flag=0x%" PRIx64 " val=%" PRIu64 " len=%zu\n",
                                    ev->getRequestId(), writeAddr,
                                    flagAddr, flagValue,
                                    ev->getLength());
                } else if (ev->getCompletionFlagAddr() != 0) {
                    wr_to_globalmem(ev->getAddr(), ev->getLength(), ev->getData());
                    write_u64_to_storage(ev->getCompletionFlagAddr(), ev->getCompletionValue());
                    dma_read_completion_count++;
                } else {
                    dma_read_completion_no_pending_count++;
                    output->verbose(CALL_INFO, 1, 2,
                                    "handleDmaReceives: No pending DMA operation found (requestId=%" PRIu64 ")\n",
                                    ev->getRequestId());
                }
            }
            issue_pending_dma_read_window();
        }

        delete ev;
        delete req;
    }
    return handled_any;
}

void GlobalMemoryImplement::registerEndpoint(const std::string& name, int endpointId) {
    IDMap[name] = endpointId;

    int discoveredCore = extractCoreId(name);
    if (discoveredCore >= 0) {
        coreEndpointMap[discoveredCore] = endpointId;
        output->verbose(CALL_INFO, 2, 2,
                        "Discovered GlobalMemory endpoint '%s' -> core %d, network ID %d\n",
                        name.c_str(), discoveredCore, endpointId);
    }
}

int GlobalMemoryImplement::extractCoreId(const std::string& name) const {
    if (name.find("global_memory") == std::string::npos) {
        return -1;
    }

    std::size_t corePos = name.find("core");
    if (corePos == std::string::npos) {
        return -1;
    }
    corePos += 4; // skip "core"

    std::size_t endPos = corePos;
    while (endPos < name.size() && std::isdigit(static_cast<unsigned char>(name[endPos]))) {
        ++endPos;
    }

    if (endPos == corePos) {
        return -1;
    }

    return std::stoi(name.substr(corePos, endPos - corePos));
}

int GlobalMemoryImplement::resolveEndpointForAddress(uint64_t addr) {
    if (network_id == -1) {
        network_id = link_control->getEndpointID();
        coreEndpointMap[core_id] = network_id;
    }

    if (addr < globalBase) {
        output->fatal(CALL_INFO, -1,
                      "Address 0x%" PRIx64 " is below global base 0x%" PRIx64 " for core %d.\n",
                      addr, globalBase, core_id);
    }

    uint64_t offset = addr - globalBase;
    uint64_t targetCore = offset / size;

    int coreIndex = static_cast<int>(targetCore);

    auto it = coreEndpointMap.find(coreIndex);
    if (it != coreEndpointMap.end() && it->second != -1) {
        return it->second;
    }

    // 如果映射尚未建立，但我们有字符串映射，可尝试解析
    for (const auto& entry : IDMap) {
        if (extractCoreId(entry.first) == coreIndex) {
            coreEndpointMap[coreIndex] = entry.second;
            return entry.second;
        }
    }

    if (coreIndex == core_id) {
        coreEndpointMap[coreIndex] = network_id;
        return network_id;
    }

    output->fatal(CALL_INFO, -1,
                  "GlobalMemory '%s' unable to resolve network endpoint for address 0x%" PRIx64 ". Known endpoints: %zu\n",
                  getName().c_str(), addr, coreEndpointMap.size());
    return -1; // Unreachable, but avoids compiler warning
}

// ---------------------- GlobalMemoryLocal (network-free) ----------------------
GlobalMemoryLocal::GlobalMemoryLocal(ComponentId_t id, Params& params)
    : GlobalMemoryAPI(id, params)
{
    baseAddr = params.find<uint64_t>("baseAddr", 0);
    size     = params.find<uint64_t>("size", 64 * 1024);
    if (size == 0) size = 64 * 1024; // ensure non-zero
    storage.resize(size, 0x00);
}

void GlobalMemoryLocal::wr_to_globalmem(uint64_t wr_addr, size_t length, const std::vector<uint8_t>& wr_data)
{
    assert(wr_addr >= baseAddr);
    assert((wr_addr + length) <= (baseAddr + size));
    assert(wr_data.size() == length);
    std::copy(wr_data.begin(), wr_data.begin() + length, storage.begin() + (wr_addr - baseAddr));
}

void GlobalMemoryLocal::rd_from_globalmem(uint64_t rd_addr, size_t length, std::vector<uint8_t>& rd_data)
{
    assert(rd_addr >= baseAddr);
    assert((rd_addr + length) <= (baseAddr + size));
    rd_data.assign(storage.begin() + (rd_addr - baseAddr), storage.begin() + (rd_addr - baseAddr) + length);
}

bool GlobalMemoryLocal::localReadAsync(uint64_t addr,
                                       size_t length,
                                       LocalMemoryClient,
                                       uint64_t tag,
                                       LocalReadCallback cb)
{
    if (!cb || length == 0 || addr < baseAddr || addr - baseAddr > size ||
        length > size - (addr - baseAddr)) {
        return false;
    }
    std::vector<uint8_t> data;
    rd_from_globalmem(addr, length, data);
    cb(true, tag, data);
    return true;
}

bool GlobalMemoryLocal::localWriteAsync(uint64_t addr,
                                        const std::vector<uint8_t>& data,
                                        LocalMemoryClient,
                                        uint64_t tag,
                                        LocalWriteCallback cb)
{
    if (!cb || data.empty() || addr < baseAddr || addr - baseAddr > size ||
        data.size() > size - (addr - baseAddr)) {
        return false;
    }
    wr_to_globalmem(addr, data.size(), data);
    cb(true, tag);
    return true;
}

void GlobalMemoryLocal::wr_to_network(uint64_t, size_t, std::vector<uint8_t>&)
{
    // No network in local mode; ignore.
}

void GlobalMemoryLocal::rd_to_network(uint64_t, size_t, uint64_t)
{
    // No network in local mode; ignore.
}

void GlobalMemoryLocal::dma_write_to_host(uint64_t, size_t, const std::vector<uint8_t>&, DmaCallback, DmaRequestKind)
{
    assert(false && "GlobalMemoryLocal does not support dma_write_to_host (no StandardMem interface wired)");
}

void GlobalMemoryLocal::dma_write_from_globalmem_to_host(uint64_t,
                                                          uint64_t,
                                                          size_t,
                                                          DmaCallback cb,
                                                          DmaRequestKind)
{
    if (cb) {
        cb(false);
    }
}

void GlobalMemoryLocal::dma_read_from_host_to_globalmem(uint64_t, size_t, uint64_t, DmaCallback, DmaRequestKind,
                                                       const DmaConsumerMetadata&)
{
    assert(false && "GlobalMemoryLocal does not support dma_read_from_host_to_globalmem (no StandardMem interface wired)");
}

uint64_t GlobalMemoryLocal::dma_write_to_host_async(uint64_t, size_t, const std::vector<uint8_t>&)
{
    const uint64_t token = next_dma_completion_token++;
    dma_completion_tokens[token] = true;
    return token;
}

uint64_t GlobalMemoryLocal::dma_read_from_host_to_globalmem_async(uint64_t, size_t, uint64_t, DmaRequestKind)
{
    const uint64_t token = next_dma_completion_token++;
    dma_completion_tokens[token] = true;
    return token;
}

bool GlobalMemoryLocal::dma_completion_done(uint64_t token) const
{
    if (token == 0) {
        return true;
    }
    auto it = dma_completion_tokens.find(token);
    return it != dma_completion_tokens.end() && it->second;
}

void GlobalMemoryLocal::dma_completion_retire(uint64_t token)
{
    if (token == 0) {
        return;
    }
    dma_completion_tokens.erase(token);
}
