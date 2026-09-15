#include <cinttypes>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <tuple>

#include <sst/elements/golem/sfu/sfu.h>

namespace SST {
namespace Golem {

namespace {

using SoftmaxReducerKey = std::pair<uint64_t, uint32_t>;
using DistributedSoftmaxReducerKey =
    std::tuple<uint64_t, uint64_t, uint32_t, uint32_t>;
using DistributedReductionResponseFanoutKey =
    std::tuple<uint64_t,
               uint64_t,
               uint32_t,
               uint32_t,
               ReductionTransportMessageKind>;
constexpr uint32_t GOLEM_DTYPE_FP32_VALUE = 1;
constexpr uint32_t GOLEM_SFU_PRIMITIVE_FLAG_REPEAT_CHUNK = 0x1;
constexpr uint32_t GOLEM_SFU_PRIMITIVE_BATCH_MAX_DESCS = 64;
constexpr uint32_t GOLEM_SFU_JOB_SOFTMAX_ROW_BAND_ROWS = 4;
uint64_t ceilDiv(uint64_t value, uint64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

uint64_t ceilMulDiv(uint64_t value, uint64_t multiplier, uint64_t divisor)
{
    const __uint128_t product = static_cast<__uint128_t>(value) * multiplier;
    return static_cast<uint64_t>((product + divisor - 1) / divisor);
}

struct SoftmaxReducerRowState {
    double m_acc = -std::numeric_limits<double>::infinity();
    double l_acc = 0.0;
    uint32_t partials_seen = 0;
    uint32_t n_tiles_expected = 0;
    uint32_t normalizes_done = 0;
    bool ready = false;
};

std::map<SoftmaxReducerKey, SoftmaxReducerRowState>& softmaxReducerRows()
{
    static std::map<SoftmaxReducerKey, SoftmaxReducerRowState> rows;
    return rows;
}

struct DistributedSoftmaxReducerRowState {
    uint32_t expectedWorkers = 0;
    uint32_t expectedRows = 0;
    uint32_t expectedCols = 0;
    std::vector<uint8_t> maxSeen;
    std::vector<uint8_t> sumSeen;
    std::vector<uint8_t> normalizeSeen;
    std::vector<uint8_t> abortSeen;
    double globalMax = -std::numeric_limits<double>::infinity();
    double globalSum = 0.0;
    bool aborted = false;
};

enum class DistributedReducerResult : uint8_t {
    Accepted,
    Pending,
    Ready,
    Aborted,
    Invalid,
};

std::map<DistributedSoftmaxReducerKey, DistributedSoftmaxReducerRowState>&
distributedSoftmaxReducerRows()
{
    static std::map<DistributedSoftmaxReducerKey, DistributedSoftmaxReducerRowState> rows;
    return rows;
}

std::set<DistributedSoftmaxReducerKey>& distributedSoftmaxAbortedRows()
{
    static std::set<DistributedSoftmaxReducerKey> rows;
    return rows;
}

DistributedSoftmaxReducerKey distributedSoftmaxReducerKey(uint64_t jobId,
                                                           uint64_t tag,
                                                           uint32_t ownerCore,
                                                           uint32_t row)
{
    return DistributedSoftmaxReducerKey(jobId, tag, ownerCore, row);
}

void markDistributedSoftmaxRowAborted(const DistributedSoftmaxReducerKey& key)
{
    distributedSoftmaxAbortedRows().insert(key);
}

bool distributedSoftmaxRowAborted(const DistributedSoftmaxReducerKey& key)
{
    return distributedSoftmaxAbortedRows().find(key) != distributedSoftmaxAbortedRows().end();
}

std::set<DistributedReductionResponseFanoutKey>& distributedReductionResponseFanoutRows()
{
    static std::set<DistributedReductionResponseFanoutKey> rows;
    return rows;
}

void clearDistributedReductionResponseFanout(const DistributedSoftmaxReducerKey& key)
{
    auto& rows = distributedReductionResponseFanoutRows();
    const uint64_t jobId = std::get<0>(key);
    const uint64_t tag = std::get<1>(key);
    const uint32_t ownerCore = std::get<2>(key);
    const uint32_t row = std::get<3>(key);
    rows.erase(DistributedReductionResponseFanoutKey(
        jobId, tag, ownerCore, row, ReductionTransportMessageKind::MaxResponse));
    rows.erase(DistributedReductionResponseFanoutKey(
        jobId, tag, ownerCore, row, ReductionTransportMessageKind::SumResponse));
}

bool distributedSoftmaxJobMatchesKey(const DistributedSoftmaxReducerKey& key,
                                     uint64_t jobId,
                                     uint64_t tag,
                                     uint32_t ownerCore)
{
    return std::get<0>(key) == jobId && std::get<1>(key) == tag &&
           std::get<2>(key) == ownerCore;
}

std::vector<DistributedSoftmaxReducerKey> collectDistributedSoftmaxJobKeys(
    uint64_t jobId,
    uint64_t tag,
    uint32_t ownerCore)
{
    std::vector<DistributedSoftmaxReducerKey> keys;
    for (const auto& entry : distributedSoftmaxReducerRows()) {
        if (distributedSoftmaxJobMatchesKey(entry.first, jobId, tag, ownerCore)) {
            keys.push_back(entry.first);
        }
    }
    return keys;
}

bool initializeDistributedSoftmaxRow(DistributedSoftmaxReducerRowState* row,
                                     uint32_t expectedWorkers,
                                     uint32_t expectedRows,
                                     uint32_t expectedCols)
{
    if (row == nullptr || expectedWorkers == 0 || expectedRows == 0 || expectedCols == 0) {
        return false;
    }
    if (row->expectedWorkers == 0) {
        row->expectedWorkers = expectedWorkers;
        row->expectedRows = expectedRows;
        row->expectedCols = expectedCols;
        row->maxSeen.assign(expectedWorkers, 0);
        row->sumSeen.assign(expectedWorkers, 0);
        row->normalizeSeen.assign(expectedWorkers, 0);
        row->abortSeen.assign(expectedWorkers, 0);
    }
    return row->expectedWorkers == expectedWorkers &&
           row->expectedRows == expectedRows && row->expectedCols == expectedCols;
}

bool allWorkerSlotsSeen(const std::vector<uint8_t>& seen)
{
    return !seen.empty() &&
           std::all_of(seen.begin(), seen.end(), [](uint8_t value) { return value != 0; });
}

DistributedReducerResult observeDistributedSoftmaxAbort(
    const DistributedSoftmaxReducerKey& key,
    DistributedSoftmaxReducerRowState* row,
    uint32_t workerSlot)
{
    if (row == nullptr || !row->aborted || workerSlot >= row->expectedWorkers) {
        return DistributedReducerResult::Invalid;
    }
    row->abortSeen[workerSlot] = 1;
    if (allWorkerSlotsSeen(row->abortSeen)) {
        clearDistributedReductionResponseFanout(key);
        distributedSoftmaxReducerRows().erase(key);
    }
    return DistributedReducerResult::Aborted;
}

void abortDistributedSoftmaxRow(uint64_t jobId,
                                uint64_t tag,
                                uint32_t ownerCore,
                                uint32_t row,
                                uint32_t workerSlot,
                                uint32_t expectedWorkers,
                                uint32_t expectedRows,
                                uint32_t expectedCols)
{
    auto& rows = distributedSoftmaxReducerRows();
    const auto key = distributedSoftmaxReducerKey(jobId, tag, ownerCore, row);
    markDistributedSoftmaxRowAborted(key);
    auto& rowState = rows[key];
    clearDistributedReductionResponseFanout(key);
    if (!initializeDistributedSoftmaxRow(
            &rowState, expectedWorkers, expectedRows, expectedCols)) {
        rowState.aborted = true;
        if (workerSlot < rowState.expectedWorkers) {
            (void)observeDistributedSoftmaxAbort(key, &rowState, workerSlot);
        }
        return;
    }
    rowState.aborted = true;
    (void)observeDistributedSoftmaxAbort(key, &rowState, workerSlot);
}

DistributedReducerResult submitDistributedSoftmaxMax(uint64_t jobId,
                                                      uint64_t tag,
                                                      uint32_t ownerCore,
                                                      uint32_t row,
                                                      uint32_t workerSlot,
                                                      uint32_t expectedWorkers,
                                                      uint32_t expectedRows,
                                                      uint32_t expectedCols,
                                                      double localMax)
{
    const auto key = distributedSoftmaxReducerKey(jobId, tag, ownerCore, row);
    if (distributedSoftmaxRowAborted(key)) {
        return DistributedReducerResult::Aborted;
    }
    auto& rows = distributedSoftmaxReducerRows();
    auto& rowState = rows[key];
    if (!initializeDistributedSoftmaxRow(
            &rowState, expectedWorkers, expectedRows, expectedCols) ||
        workerSlot >= expectedWorkers) {
        return DistributedReducerResult::Invalid;
    }
    if (rowState.aborted) {
        return observeDistributedSoftmaxAbort(key, &rowState, workerSlot);
    }
    if (rowState.maxSeen[workerSlot] != 0) {
        return DistributedReducerResult::Invalid;
    }
    rowState.maxSeen[workerSlot] = 1;
    rowState.globalMax = std::max(rowState.globalMax, localMax);
    return DistributedReducerResult::Accepted;
}

DistributedReducerResult distributedSoftmaxMaxReady(uint64_t jobId,
                                                     uint64_t tag,
                                                     uint32_t ownerCore,
                                                     uint32_t row,
                                                     uint32_t workerSlot,
                                                     double* globalMax)
{
    auto& rows = distributedSoftmaxReducerRows();
    const auto key = distributedSoftmaxReducerKey(jobId, tag, ownerCore, row);
    if (distributedSoftmaxRowAborted(key)) {
        return DistributedReducerResult::Aborted;
    }
    auto it = rows.find(key);
    if (it == rows.end()) {
        return DistributedReducerResult::Pending;
    }
    if (it->second.aborted) {
        return observeDistributedSoftmaxAbort(key, &it->second, workerSlot);
    }
    if (!allWorkerSlotsSeen(it->second.maxSeen)) {
        return DistributedReducerResult::Pending;
    }
    if (globalMax != nullptr) {
        *globalMax = it->second.globalMax;
    }
    return DistributedReducerResult::Ready;
}

DistributedReducerResult submitDistributedSoftmaxSum(uint64_t jobId,
                                                      uint64_t tag,
                                                      uint32_t ownerCore,
                                                      uint32_t row,
                                                      uint32_t workerSlot,
                                                      uint32_t expectedWorkers,
                                                      uint32_t expectedRows,
                                                      uint32_t expectedCols,
                                                      double localSum)
{
    auto& rows = distributedSoftmaxReducerRows();
    const auto key = distributedSoftmaxReducerKey(jobId, tag, ownerCore, row);
    if (distributedSoftmaxRowAborted(key)) {
        return DistributedReducerResult::Aborted;
    }
    auto it = rows.find(key);
    if (it == rows.end() ||
        !initializeDistributedSoftmaxRow(
            &it->second, expectedWorkers, expectedRows, expectedCols) ||
        workerSlot >= expectedWorkers) {
        return DistributedReducerResult::Invalid;
    }
    if (it->second.aborted) {
        return observeDistributedSoftmaxAbort(key, &it->second, workerSlot);
    }
    if (it->second.sumSeen[workerSlot] != 0 ||
        !allWorkerSlotsSeen(it->second.maxSeen)) {
        return DistributedReducerResult::Invalid;
    }
    it->second.sumSeen[workerSlot] = 1;
    it->second.globalSum += localSum;
    return DistributedReducerResult::Accepted;
}

DistributedReducerResult distributedSoftmaxSumReady(uint64_t jobId,
                                                     uint64_t tag,
                                                     uint32_t ownerCore,
                                                     uint32_t row,
                                                     uint32_t workerSlot,
                                                     double* globalSum)
{
    auto& rows = distributedSoftmaxReducerRows();
    const auto key = distributedSoftmaxReducerKey(jobId, tag, ownerCore, row);
    if (distributedSoftmaxRowAborted(key)) {
        return DistributedReducerResult::Aborted;
    }
    auto it = rows.find(key);
    if (it == rows.end()) {
        return DistributedReducerResult::Pending;
    }
    if (it->second.aborted) {
        return observeDistributedSoftmaxAbort(key, &it->second, workerSlot);
    }
    if (!allWorkerSlotsSeen(it->second.sumSeen)) {
        return DistributedReducerResult::Pending;
    }
    if (globalSum != nullptr) {
        *globalSum = it->second.globalSum;
    }
    return DistributedReducerResult::Ready;
}

DistributedReducerResult markDistributedSoftmaxNormalized(uint64_t jobId,
                                                           uint64_t tag,
                                                           uint32_t ownerCore,
                                                           uint32_t row,
                                                           uint32_t workerSlot,
                                                           uint32_t expectedWorkers,
                                                           uint32_t expectedRows,
                                                           uint32_t expectedCols)
{
    auto& rows = distributedSoftmaxReducerRows();
    const auto key = distributedSoftmaxReducerKey(jobId, tag, ownerCore, row);
    auto it = rows.find(key);
    if (it == rows.end() || it->second.expectedWorkers != expectedWorkers ||
        it->second.expectedRows != expectedRows || it->second.expectedCols != expectedCols ||
        workerSlot >= expectedWorkers) {
        return DistributedReducerResult::Invalid;
    }
    if (it->second.aborted) {
        return observeDistributedSoftmaxAbort(key, &it->second, workerSlot);
    }
    if (it->second.normalizeSeen[workerSlot] != 0 ||
        !allWorkerSlotsSeen(it->second.sumSeen)) {
        return DistributedReducerResult::Invalid;
    }
    it->second.normalizeSeen[workerSlot] = 1;
    if (allWorkerSlotsSeen(it->second.normalizeSeen)) {
        clearDistributedReductionResponseFanout(key);
        distributedSoftmaxReducerRows().erase(key);
    }
    return DistributedReducerResult::Accepted;
}

template <typename T>
void appendBytes(std::vector<uint8_t>& out, const T& value)
{
    const uint8_t* raw = reinterpret_cast<const uint8_t*>(&value);
    out.insert(out.end(), raw, raw + sizeof(T));
}

size_t tilePackedIndex(const SFUSoftmaxTileDesc& desc, uint32_t row, uint32_t col)
{
    return static_cast<size_t>(col) * desc.block_m + row;
}

uint32_t effectiveStride(uint32_t strideBytes)
{
    return strideBytes == 0 ? static_cast<uint32_t>(sizeof(float)) : strideBytes;
}

uint32_t workerColumnBegin(uint32_t cols, uint32_t worker, uint32_t workerCores)
{
    return static_cast<uint32_t>(static_cast<uint64_t>(cols) * worker / workerCores);
}

uint32_t workerColumnEnd(uint32_t cols, uint32_t worker, uint32_t workerCores)
{
    return static_cast<uint32_t>(static_cast<uint64_t>(cols) * (worker + 1) / workerCores);
}

struct SoftmaxJobRowBandState {
    uint32_t rowBegin = 0;
    uint32_t rowEnd = 0;
    uint32_t bandRows = 0;
    std::vector<double> localMax;
    std::vector<double> localSum;
    std::vector<double> globalMax;
    std::vector<double> globalSum;
};

bool readSoftmaxJobChunk(GlobalMemoryAPI* globalMem,
                         const SFUJobDesc& desc,
                         uint32_t row,
                         uint32_t colBegin,
                         uint32_t colEnd,
                         std::vector<float>* chunk)
{
    if (globalMem == nullptr || chunk == nullptr || colEnd < colBegin || colEnd > desc.cols) {
        return false;
    }

    const uint32_t elemCount = colEnd - colBegin;
    const uint64_t offsetElems = static_cast<uint64_t>(row) * desc.cols + colBegin;
    std::vector<uint8_t> raw;
    globalMem->rd_from_globalmem(desc.input0_addr + offsetElems * sizeof(float),
                                 static_cast<uint64_t>(elemCount) * sizeof(float),
                                 raw);
    if (raw.size() != static_cast<size_t>(elemCount) * sizeof(float)) {
        return false;
    }

    chunk->assign(elemCount, 0.0f);
    if (elemCount > 0) {
        std::memcpy(chunk->data(), raw.data(), raw.size());
    }
    return true;
}

bool readSoftmaxJobOutputChunk(GlobalMemoryAPI* globalMem,
                               const SFUJobDesc& desc,
                               uint32_t row,
                               uint32_t colBegin,
                               uint32_t colEnd,
                               std::vector<float>* chunk)
{
    if (globalMem == nullptr || chunk == nullptr || colEnd < colBegin || colEnd > desc.cols) {
        return false;
    }

    const uint32_t elemCount = colEnd - colBegin;
    const uint64_t offsetElems = static_cast<uint64_t>(row) * desc.cols + colBegin;
    std::vector<uint8_t> raw;
    globalMem->rd_from_globalmem(desc.output_addr + offsetElems * sizeof(float),
                                 static_cast<uint64_t>(elemCount) * sizeof(float),
                                 raw);
    if (raw.size() != static_cast<size_t>(elemCount) * sizeof(float)) {
        return false;
    }

    chunk->assign(elemCount, 0.0f);
    if (elemCount > 0) {
        std::memcpy(chunk->data(), raw.data(), raw.size());
    }
    return true;
}

bool writeSoftmaxJobChunk(GlobalMemoryAPI* globalMem,
                          const SFUJobDesc& desc,
                          uint32_t row,
                          uint32_t colBegin,
                          const std::vector<float>& chunk)
{
    if (globalMem == nullptr || colBegin > desc.cols || chunk.empty()) {
        return false;
    }
    if (static_cast<uint64_t>(colBegin) + chunk.size() > desc.cols) {
        return false;
    }

    std::vector<uint8_t> raw(chunk.size() * sizeof(float));
    std::memcpy(raw.data(), chunk.data(), raw.size());
    const uint64_t offsetElems = static_cast<uint64_t>(row) * desc.cols + colBegin;
    globalMem->wr_to_globalmem(desc.output_addr + offsetElems * sizeof(float),
                               raw.size(),
                               raw);
    return true;
}

bool readDistributedSoftmaxJobChunk(GlobalMemoryAPI* globalMem,
                                    uint64_t baseAddr,
                                    uint32_t rows,
                                    uint32_t localCols,
                                    uint32_t row,
                                    uint32_t localColBegin,
                                    uint32_t localColEnd,
                                    std::vector<float>* chunk)
{
    if (globalMem == nullptr || baseAddr == 0 || chunk == nullptr || row >= rows ||
        localColEnd <= localColBegin || localColEnd > localCols) {
        return false;
    }
    const uint32_t elemCount = localColEnd - localColBegin;
    const uint64_t offsetElems = static_cast<uint64_t>(row) * localCols + localColBegin;
    std::vector<uint8_t> raw;
    globalMem->rd_from_globalmem(baseAddr + offsetElems * sizeof(float),
                                 static_cast<uint64_t>(elemCount) * sizeof(float),
                                 raw);
    if (raw.size() != static_cast<size_t>(elemCount) * sizeof(float)) {
        return false;
    }
    chunk->resize(elemCount);
    std::memcpy(chunk->data(), raw.data(), raw.size());
    return true;
}

bool writeDistributedSoftmaxJobChunk(GlobalMemoryAPI* globalMem,
                                     uint64_t baseAddr,
                                     uint32_t rows,
                                     uint32_t localCols,
                                     uint32_t row,
                                     uint32_t localColBegin,
                                     const std::vector<float>& chunk)
{
    if (globalMem == nullptr || baseAddr == 0 || row >= rows || chunk.empty() ||
        static_cast<uint64_t>(localColBegin) + chunk.size() > localCols) {
        return false;
    }
    std::vector<uint8_t> raw(chunk.size() * sizeof(float));
    std::memcpy(raw.data(), chunk.data(), raw.size());
    const uint64_t offsetElems = static_cast<uint64_t>(row) * localCols + localColBegin;
    globalMem->wr_to_globalmem(baseAddr + offsetElems * sizeof(float), raw.size(), raw);
    return true;
}

} // namespace

SFU::SFU(ComponentId_t id, SST::Params& params)
    : SFUAPI(id, params),
      coreId_(params.find<uint32_t>("core_id", 0)),
      activeWorkerCores_(params.find<uint32_t>("active_worker_cores", 1)),
      maxInflight_(params.find<uint32_t>("max_inflight", 8)),
      statsLatency_(params.find<uint32_t>("stats_latency", 1)),
      mergeLatency_(params.find<uint32_t>("merge_latency", 1)),
      normalizeLatency_(params.find<uint32_t>("normalize_latency", 1)),
      rowEngineAcceleratorClockHz_(params.find<uint64_t>("accelerator_clock_hz", 2300000000ULL)),
      rowEngineVectorLanes_(params.find<uint32_t>("vector_lanes", 16)),
      rowEngineExpLanes_(params.find<uint32_t>("exp_lanes", 4)),
      rowEngineReductionTreeLatency_(params.find<uint32_t>("reduction_tree_latency", 4)),
      rowEngineExpLatency_(params.find<uint32_t>("exp_latency", 8)),
      rowEngineReciprocalLatency_(params.find<uint32_t>("reciprocal_latency", 1)),
      rowEngineRsqrtLatency_(params.find<uint32_t>("rsqrt_latency", 8)),
      rowEngineContexts_(params.find<uint32_t>("row_contexts", 4)),
      rowEngineScratchpadBytes_(params.find<uint64_t>("scratchpad_bytes", 65536)),
      rowEngineTimebaseTicksPerSecond_(0),
      rowEngineFreeTick_(0),
      rowEngineVectorFreeCycle_(0),
      rowEngineExpFreeCycle_(0),
      inflight_(0),
      verbose_(params.find<int>("verbose", 0)),
      distributedReductionTransport_(DistributedReductionTransport::Shared),
      globalMem_(nullptr),
      output_("Golem::SFU[@p:@l]: ", verbose_, 0, SST::Output::STDOUT),
      rowEngineSelfLink_(nullptr)
{
    attentionClusterEnable_ =
        params.find<bool>("attention_cluster_enable", false);
    attentionScoreFifoBytesPerCycle_ =
        params.find<uint64_t>("attention_score_fifo_bytes_per_cycle", 64);
    attentionScoreFifoLatencyCycles_ =
        params.find<uint64_t>("attention_score_fifo_latency_cycles", 1);
    attentionScoreFifoQueueDepth_ =
        params.find<uint32_t>("attention_score_fifo_queue_depth", 64);
    if (activeWorkerCores_ == 0) {
        activeWorkerCores_ = 1;
    }
    if (maxInflight_ == 0) {
        maxInflight_ = 1;
    }
    if (rowEngineAcceleratorClockHz_ == 0 || rowEngineVectorLanes_ == 0 ||
        rowEngineExpLanes_ == 0 || rowEngineContexts_ == 0 ||
        rowEngineScratchpadBytes_ == 0 ||
        (attentionClusterEnable_ &&
         (attentionScoreFifoBytesPerCycle_ == 0 ||
          attentionScoreFifoLatencyCycles_ == 0 ||
          attentionScoreFifoQueueDepth_ == 0))) {
        output_.fatal(CALL_INFO, -1, "Row Engine parameters must be positive\n");
    }
    const bool attentionKvPairReuse =
        params.find<bool>("attention_kv_pair_reuse", false);
    const uint32_t attentionKvQueryGroupSize =
        params.find<uint32_t>("attention_kv_query_group_size", 2);
    if (attentionKvPairReuse && attentionKvQueryGroupSize != 2 &&
        attentionKvQueryGroupSize != 4) {
        output_.fatal(
            CALL_INFO, -1,
            "Attention K/V query group size must be 2 or 4 when reuse is enabled\n");
    }
    attentionOnlineContexts_.resize(std::max<size_t>(64u,
        attentionKvPairReuse ? static_cast<size_t>(16u * attentionKvQueryGroupSize)
                              : static_cast<size_t>(rowEngineContexts_)));
    rowEngineTimebaseTicksPerSecond_ = getTimeConverter("1s")->getFactor();
    if (rowEngineTimebaseTicksPerSecond_ == 0) {
        output_.fatal(CALL_INFO, -1, "SST timebase conversion for 1s must be positive\n");
    }
    rowEngineSelfLink_ = configureSelfLink(
        "RowEngineSelf",
        std::to_string(rowEngineAcceleratorClockHz_) + "Hz",
        new SST::Event::Handler2<SFU, &SFU::handleTensorRowEngineEvent>(this));
    const std::string transport =
        params.find<std::string>("distributed_reduction_transport", "shared");
    if (transport == "shared") {
        distributedReductionTransport_ = DistributedReductionTransport::Shared;
    } else if (transport == "modeled_noc" || transport == "noc_model" ||
        transport == "noc") {
        distributedReductionTransport_ = DistributedReductionTransport::ModeledNoC;
    } else if (transport == "explicit_noc") {
        distributedReductionTransport_ = DistributedReductionTransport::ExplicitNoC;
    } else {
        output_.fatal(CALL_INFO,
                      -1,
                      "invalid distributed_reduction_transport='%s' "
                      "(expected shared, modeled_noc, or explicit_noc)\n",
                      transport.c_str());
    }

    statOpsIssued_ = registerStatistic<uint64_t>("sfu_ops_issued");
    statSoftmaxRows_ = registerStatistic<uint64_t>("sfu_softmax_rows");
    statSoftmaxTiles_ = registerStatistic<uint64_t>("sfu_softmax_tiles");
    statJobSoftmaxMaxChunks_ = registerStatistic<uint64_t>("sfu_job_softmax_max_chunks");
    statJobSoftmaxSumChunks_ = registerStatistic<uint64_t>("sfu_job_softmax_sum_chunks");
    statJobSoftmaxNormChunks_ = registerStatistic<uint64_t>("sfu_job_softmax_norm_chunks");
    statRowEngineJobs_ = registerStatistic<uint64_t>("sfu_row_engine_jobs");
    statRowEngineRows_ = registerStatistic<uint64_t>("sfu_row_engine_rows");
    statRowEngineMaxCycles_ = registerStatistic<uint64_t>("sfu_row_engine_max_cycles");
    statRowEngineExpSumCycles_ = registerStatistic<uint64_t>("sfu_row_engine_exp_sum_cycles");
    statRowEngineNormalizeCycles_ = registerStatistic<uint64_t>("sfu_row_engine_normalize_cycles");
    statRowEngineMaxStartCycles_ = registerStatistic<uint64_t>("sfu_row_engine_max_start_cycles");
    statRowEngineMaxEndCycles_ = registerStatistic<uint64_t>("sfu_row_engine_max_end_cycles");
    statRowEngineExpSumStartCycles_ = registerStatistic<uint64_t>("sfu_row_engine_exp_sum_start_cycles");
    statRowEngineExpSumEndCycles_ = registerStatistic<uint64_t>("sfu_row_engine_exp_sum_end_cycles");
    statRowEngineNormalizeStartCycles_ = registerStatistic<uint64_t>("sfu_row_engine_normalize_start_cycles");
    statRowEngineNormalizeEndCycles_ = registerStatistic<uint64_t>("sfu_row_engine_normalize_end_cycles");
    statRowEngineModeledCycles_ = registerStatistic<uint64_t>("sfu_row_engine_modeled_cycles");
    statRowEngineQueueWaitCycles_ = registerStatistic<uint64_t>("sfu_row_engine_queue_wait_cycles");
    statRowEngineWaitPolls_ = registerStatistic<uint64_t>("sfu_row_engine_wait_polls");
    statRowEngineCompletedJobs_ = registerStatistic<uint64_t>("sfu_row_engine_completed_jobs");
    statRowEngineIssueTick_ = registerStatistic<uint64_t>("sfu_row_engine_issue_tick");
    statRowEngineStartTick_ = registerStatistic<uint64_t>("sfu_row_engine_start_tick");
    statRowEngineReadyTick_ = registerStatistic<uint64_t>("sfu_row_engine_ready_tick");
    statRowEngineCompletionObservedTick_ = registerStatistic<uint64_t>("sfu_row_engine_completion_observed_tick");
    statTensorBandDispatchTick_ = registerStatistic<uint64_t>("sfu_tensor_band_dispatch_tick");
    statTensorWorkerDispatchTick_ = registerStatistic<uint64_t>("sfu_tensor_worker_dispatch_tick");
    statTensorInputDmaReadyTick_ = registerStatistic<uint64_t>("sfu_tensor_input_dma_ready_tick");
    statTensorMaxStartTick_ = registerStatistic<uint64_t>("sfu_tensor_max_start_tick");
    statTensorMaxDoneTick_ = registerStatistic<uint64_t>("sfu_tensor_max_done_tick");
    statTensorExpSumStartTick_ = registerStatistic<uint64_t>("sfu_tensor_exp_sum_start_tick");
    statTensorExpSumDoneTick_ = registerStatistic<uint64_t>("sfu_tensor_exp_sum_done_tick");
    statTensorNormalizeStartTick_ = registerStatistic<uint64_t>("sfu_tensor_normalize_start_tick");
    statTensorNormalizeDoneTick_ = registerStatistic<uint64_t>("sfu_tensor_normalize_done_tick");
    statTensorComputeDoneTick_ = registerStatistic<uint64_t>("sfu_tensor_compute_done_tick");
    statTensorOutputDmaAckTick_ = registerStatistic<uint64_t>("sfu_tensor_output_dma_ack_tick");
    statTensorCompletionReceivedTick_ = registerStatistic<uint64_t>("sfu_tensor_completion_received_tick");
    statTensorGuestWaitObservedTick_ = registerStatistic<uint64_t>("sfu_tensor_guest_wait_observed_tick");
    statAttentionJobs_ = registerStatistic<uint64_t>("sfu_attention_jobs");
    statAttentionRsqrtReadyTick_ = registerStatistic<uint64_t>("sfu_attention_rsqrt_ready_tick");
    statAttentionScaleMaskStartTick_ = registerStatistic<uint64_t>("sfu_attention_scale_mask_start_tick");
    statAttentionScaleMaskDoneTick_ = registerStatistic<uint64_t>("sfu_attention_scale_mask_done_tick");
    statAttentionScaledElements_ = registerStatistic<uint64_t>("sfu_attention_scaled_elements");
    statAttentionMaskedElements_ = registerStatistic<uint64_t>("sfu_attention_masked_elements");
    statAttentionClusterSfuBusyUnion_ = registerStatistic<uint64_t>(
        "attention_cluster_sfu_busy_union_ticks");
    statAttentionClusterSfuBusySpan_ = registerStatistic<uint64_t>(
        "attention_cluster_sfu_busy_span_ticks");
    statAttentionClusterSfuIdleGap_ = registerStatistic<uint64_t>(
        "attention_cluster_sfu_idle_gap_ticks");
    statAttentionClusterSfuMaxConcurrency_ = registerStatistic<uint64_t>(
        "attention_cluster_sfu_max_concurrency");
    statAttentionClusterSfuBackpressureStalls_ = registerStatistic<uint64_t>(
        "attention_cluster_sfu_backpressure_stalls");
    statAttentionClusterScoreFifoProducerWrites_ = registerStatistic<uint64_t>(
        "attention_cluster_score_fifo_producer_writes");
    statAttentionClusterScoreFifoProducerBytes_ = registerStatistic<uint64_t>(
        "attention_cluster_score_fifo_producer_bytes");
    statAttentionClusterScoreFifoReads_ = registerStatistic<uint64_t>(
        "attention_cluster_score_fifo_reads");
    statAttentionClusterScoreFifoReadBytes_ = registerStatistic<uint64_t>(
        "attention_cluster_score_fifo_read_bytes");
    statAttentionClusterScoreFifoInternalWrites_ = registerStatistic<uint64_t>(
        "attention_cluster_score_fifo_internal_writes");
    statAttentionClusterScoreFifoInternalWriteBytes_ = registerStatistic<uint64_t>(
        "attention_cluster_score_fifo_internal_write_bytes");
    statAttentionClusterScoreFifoPortWaitCycles_ = registerStatistic<uint64_t>(
        "attention_cluster_score_fifo_port_wait_cycles");
    statAttentionClusterScoreFifoBackpressureStalls_ = registerStatistic<uint64_t>(
        "attention_cluster_score_fifo_backpressure_stalls");
    statAttentionClusterPFifoReservations_ = registerStatistic<uint64_t>(
        "attention_cluster_p_fifo_reservations");
    statAttentionClusterPFifoReleases_ = registerStatistic<uint64_t>(
        "attention_cluster_p_fifo_releases");
    statAttentionClusterPFifoCancelled_ = registerStatistic<uint64_t>(
        "attention_cluster_p_fifo_cancelled");
    statAttentionClusterPFifoWrites_ = registerStatistic<uint64_t>(
        "attention_cluster_p_fifo_writes");
    statAttentionClusterPFifoWriteBytes_ = registerStatistic<uint64_t>(
        "attention_cluster_p_fifo_write_bytes");
    statAttentionClusterPFifoReads_ = registerStatistic<uint64_t>(
        "attention_cluster_p_fifo_reads");
    statAttentionClusterPFifoReadBytes_ = registerStatistic<uint64_t>(
        "attention_cluster_p_fifo_read_bytes");
    statAttentionClusterPFifoPortWaitCycles_ = registerStatistic<uint64_t>(
        "attention_cluster_p_fifo_port_wait_cycles");
    statAttentionClusterPFifoBackpressureStalls_ = registerStatistic<uint64_t>(
        "attention_cluster_p_fifo_backpressure_stalls");
    statPrimitiveElems_ = registerStatistic<uint64_t>("sfu_primitive_elems");
    statPartialSubmits_ = registerStatistic<uint64_t>("sfu_partial_submits");
    statPartialDone_ = registerStatistic<uint64_t>("sfu_partial_done");
    statReductionMaxRequests_ = registerStatistic<uint64_t>("sfu_reduction_max_requests");
    statReductionMaxResponses_ = registerStatistic<uint64_t>("sfu_reduction_max_responses");
    statReductionSumRequests_ = registerStatistic<uint64_t>("sfu_reduction_sum_requests");
    statReductionSumResponses_ = registerStatistic<uint64_t>("sfu_reduction_sum_responses");
    statReductionTransportReceived_ = registerStatistic<uint64_t>("sfu_reduction_transport_received");
    statReductionTransportStaleDropped_ = registerStatistic<uint64_t>("sfu_reduction_transport_stale_dropped");
    statReductionTransportInboxHighWater_ = registerStatistic<uint64_t>("sfu_reduction_transport_inbox_high_water");
    statReductionTransportLatencyCycles_ = registerStatistic<uint64_t>("sfu_reduction_transport_latency_cycles");
    statCreditStalls_ = registerStatistic<uint64_t>("sfu_credit_stalls");
    statCrossTileWaitCycles_ = registerStatistic<uint64_t>("sfu_cross_tile_wait_cycles");
    statRetryEvents_ = registerStatistic<uint64_t>("sfu_retry_events");
}

bool SFU::modeledDistributedReductionEnabled() const
{
    return distributedReductionTransport_ == DistributedReductionTransport::ModeledNoC;
}

bool SFU::explicitDistributedReductionEnabled() const
{
    return distributedReductionTransport_ == DistributedReductionTransport::ExplicitNoC;
}

void SFU::handleReductionTransportMessage(const ReductionTransportMessage& message)
{
    if (!explicitDistributedReductionEnabled() || globalMem_ == nullptr) {
        return;
    }

    statReductionTransportReceived_->addData(1);
    const uint64_t receiveCycle = getCurrentSimCycle();
    if (receiveCycle >= message.sendCycle) {
        statReductionTransportLatencyCycles_->addData(receiveCycle - message.sendCycle);
    }
    if (message.kind == ReductionTransportMessageKind::TensorRowDispatch) {
        handleTensorRowDispatch(message);
        return;
    }
    if (message.kind == ReductionTransportMessageKind::TensorRowComplete) {
        handleTensorRowComplete(message);
        return;
    }
    const auto staleDrop = [this]() {
        statReductionTransportStaleDropped_->addData(1);
        statRetryEvents_->addData(1);
    };

    const bool isMaxRequest = message.kind == ReductionTransportMessageKind::MaxRequest;
    const bool isSumRequest = message.kind == ReductionTransportMessageKind::SumRequest;
    const bool isMaxResponse = message.kind == ReductionTransportMessageKind::MaxResponse;
    const bool isSumResponse = message.kind == ReductionTransportMessageKind::SumResponse;

    if (isMaxResponse || isSumResponse) {
        auto job = pendingJobOps_.find(message.tag);
        if (job == pendingJobOps_.end()) {
            staleDrop();
            return;
        }
        const SoftmaxJobStage expectedStage = isMaxResponse
            ? SoftmaxJobStage::MaxSubmitted
            : SoftmaxJobStage::SumSubmitted;
        const std::vector<uint8_t>& responseSeen = isMaxResponse
            ? job->second.maxResponseSeen
            : job->second.sumResponseSeen;
        if ((job->second.desc.flags & SFU_JOB_FLAG_DISTRIBUTED_COLUMNS) == 0 ||
            job->second.distributedAbortObserved ||
            job->second.desc.job_id != message.jobId ||
            job->second.tag != message.tag ||
            job->second.desc.owner_core != message.ownerCore ||
            job->second.workerSlot != message.workerSlot ||
            job->second.desc.worker_cores != message.expectedWorkers ||
            job->second.desc.rows != message.expectedRows ||
            job->second.desc.cols != message.expectedCols ||
            message.row >= message.expectedRows ||
            job->second.stage != expectedStage ||
            message.row >= responseSeen.size() ||
            responseSeen[message.row] != 0 ||
            message.ownerCore > std::numeric_limits<uint32_t>::max() - message.workerSlot ||
            coreId_ != message.ownerCore + message.workerSlot) {
            staleDrop();
            return;
        }

        const DistributedReductionResponseInboxKey key(message.jobId,
                                                        message.tag,
                                                        message.ownerCore,
                                                        message.row,
                                                        message.workerSlot,
                                                        message.kind);
        if (!distributedReductionResponseInbox_.emplace(key, message).second) {
            staleDrop();
            return;
        }
        const uint64_t inboxSize = distributedReductionResponseInbox_.size();
        if (inboxSize > distributedReductionResponseInboxHighWater_) {
            statReductionTransportInboxHighWater_->addData(
                inboxSize - distributedReductionResponseInboxHighWater_);
            distributedReductionResponseInboxHighWater_ = inboxSize;
        }
        recordDistributedReductionResponse(isMaxResponse);
        return;
    }

    if ((!isMaxRequest && !isSumRequest) || coreId_ != message.ownerCore ||
        message.expectedWorkers == 0 || message.expectedRows == 0 ||
        message.expectedCols == 0 || message.row >= message.expectedRows ||
        message.workerSlot >= message.expectedWorkers) {
        staleDrop();
        return;
    }

    DistributedReducerResult result = DistributedReducerResult::Invalid;
    if (isMaxRequest) {
        result = submitDistributedSoftmaxMax(message.jobId,
                                             message.tag,
                                             message.ownerCore,
                                             message.row,
                                             message.workerSlot,
                                             message.expectedWorkers,
                                             message.expectedRows,
                                             message.expectedCols,
                                             message.value);
    } else {
        result = submitDistributedSoftmaxSum(message.jobId,
                                             message.tag,
                                             message.ownerCore,
                                             message.row,
                                             message.workerSlot,
                                             message.expectedWorkers,
                                             message.expectedRows,
                                             message.expectedCols,
                                             message.value);
    }
    if (result != DistributedReducerResult::Accepted) {
        staleDrop();
        return;
    }

    double reducedValue = 0.0;
    const DistributedReducerResult ready = isMaxRequest
        ? distributedSoftmaxMaxReady(message.jobId,
                                     message.tag,
                                     message.ownerCore,
                                     message.row,
                                     message.workerSlot,
                                     &reducedValue)
        : distributedSoftmaxSumReady(message.jobId,
                                     message.tag,
                                     message.ownerCore,
                                     message.row,
                                     message.workerSlot,
                                     &reducedValue);
    if (ready == DistributedReducerResult::Pending) {
        return;
    }
    if (ready != DistributedReducerResult::Ready) {
        staleDrop();
        return;
    }

    const ReductionTransportMessageKind responseKind = isMaxRequest
        ? ReductionTransportMessageKind::MaxResponse
        : ReductionTransportMessageKind::SumResponse;
    const DistributedReductionResponseFanoutKey fanoutKey(message.jobId,
                                                           message.tag,
                                                           message.ownerCore,
                                                           message.row,
                                                           responseKind);
    if (!distributedReductionResponseFanoutRows().insert(fanoutKey).second) {
        return;
    }

    for (uint32_t workerSlot = 0; workerSlot < message.expectedWorkers; ++workerSlot) {
        ReductionTransportMessage response = message;
        response.kind = responseKind;
        response.workerSlot = workerSlot;
        response.value = reducedValue;
        if (!globalMem_->sendReductionMessage(message.ownerCore + workerSlot, response)) {
            statRetryEvents_->addData(1);
            auto ownerJob = pendingJobOps_.find(message.tag);
            if (ownerJob != pendingJobOps_.end() &&
                ownerJob->second.desc.job_id == message.jobId &&
                ownerJob->second.desc.owner_core == message.ownerCore) {
                abortDistributedSoftmaxJob(&ownerJob->second);
            } else {
                abortDistributedSoftmaxRow(message.jobId,
                                           message.tag,
                                           message.ownerCore,
                                           message.row,
                                           workerSlot,
                                           message.expectedWorkers,
                                           message.expectedRows,
                                           message.expectedCols);
            }
            return;
        }
    }
}

void SFU::recordDistributedReductionRequest(bool maxStage)
{
    if (!modeledDistributedReductionEnabled() && !explicitDistributedReductionEnabled()) {
        return;
    }
    (maxStage ? statReductionMaxRequests_ : statReductionSumRequests_)->addData(1);
}

void SFU::recordDistributedReductionResponse(bool maxStage)
{
    if (!modeledDistributedReductionEnabled() && !explicitDistributedReductionEnabled()) {
        return;
    }
    (maxStage ? statReductionMaxResponses_ : statReductionSumResponses_)->addData(1);
}

void SFU::finish()
{
    if (attentionClusterEnable_) {
        const uint64_t now = getCurrentSimCycle();
        statAttentionClusterSfuBusyUnion_->addData(
            attentionClusterSfuActivity_.unionCycles(now));
        statAttentionClusterSfuBusySpan_->addData(
            attentionClusterSfuActivity_.spanCycles(now));
        statAttentionClusterSfuIdleGap_->addData(
            attentionClusterSfuActivity_.idleCycles(now));
        statAttentionClusterSfuMaxConcurrency_->addData(
            attentionClusterSfuActivity_.maxConcurrency());
    }
    output_.output(
        "GOLEM_TENSOR_LOCAL_STATS core=%" PRIu32
        " sfu_tensor_max_local_read_bytes=%" PRIu64
        " sfu_tensor_max_local_write_bytes=%" PRIu64
        " sfu_tensor_exp_sum_local_read_bytes=%" PRIu64
        " sfu_tensor_exp_sum_local_write_bytes=%" PRIu64
        " sfu_tensor_normalize_local_read_bytes=%" PRIu64
        " sfu_tensor_normalize_local_write_bytes=%" PRIu64
        " sfu_tensor_lane_buffer_high_water=%" PRIu64
        " sfu_tensor_local_retry_events=%" PRIu64 "\n",
        coreId_,
        tensorMaxLocalReadBytes_,
        tensorMaxLocalWriteBytes_,
        tensorExpSumLocalReadBytes_,
        tensorExpSumLocalWriteBytes_,
        tensorNormalizeLocalReadBytes_,
        tensorNormalizeLocalWriteBytes_,
        tensorLaneBufferHighWater_,
        tensorLocalRetryEvents_);
    if (verbose_ > 0) {
        output_.verbose(CALL_INFO, 1, 0,
            "SFU finish core=%" PRIu32 " inflight=%" PRIu32 "\n",
            coreId_, inflight_);
    }
}

bool SFU::issueSoftmaxTile(uint64_t descAddr, uint64_t tag)
{
    if (inflight_ >= maxInflight_) {
        statCreditStalls_->addData(1);
        return false;
    }

    SoftmaxOpState state = {};
    state.descAddr = descAddr;
    state.tag = tag;
    state.status = SFUStatus::Success;
    state.normalizeReady = false;

    if (!readSoftmaxDescriptor(descAddr, &state.desc)) {
        state.status = (globalMem_ == nullptr) ? SFUStatus::GlobalMemoryUnavailable
                                               : SFUStatus::InvalidDescriptor;
    } else {
        state.status = validateSoftmaxDescriptor(state.desc);
    }

    if (state.status == SFUStatus::Success && !readInputTile(state.desc, &state.inputTile)) {
        state.status = SFUStatus::InvalidDescriptor;
    }

    if (state.status == SFUStatus::Success) {
        computeTileStats(&state);
        state.normalizeReady = mergeTileStats(&state);
        state.status = SFUStatus::Pending;
    }

    pendingSoftmaxOps_[tag] = state;

    ++inflight_;
    statOpsIssued_->addData(1);
    statSoftmaxTiles_->addData(1);
    if (state.status == SFUStatus::Success || state.status == SFUStatus::Pending) {
        statPartialSubmits_->addData(state.rowStats.size());
    }
    return true;
}

bool SFU::issuePrimitive(uint64_t descAddr, uint64_t tag)
{
    if (inflight_ >= maxInflight_) {
        statCreditStalls_->addData(1);
        return false;
    }

    PrimitiveOpState state = {};
    state.descAddr = descAddr;
    state.tag = tag;
    state.status = SFUStatus::Success;

    if (!readPrimitiveDescriptor(descAddr, &state.desc)) {
        state.status = (globalMem_ == nullptr) ? SFUStatus::GlobalMemoryUnavailable
                                               : SFUStatus::InvalidDescriptor;
    } else {
        state.status = validatePrimitiveDescriptor(state.desc);
    }

    if (state.status == SFUStatus::Success && !readPrimitiveInput(state.desc, &state.input0)) {
        state.status = SFUStatus::InvalidDescriptor;
    }
    if (state.status == SFUStatus::Success && !executePrimitive(&state)) {
        state.status = SFUStatus::InvalidDescriptor;
    }
    if (state.status == SFUStatus::Success && !writePrimitiveOutput(state.desc, state.output)) {
        state.status = SFUStatus::InvalidDescriptor;
    }

    pendingPrimitiveOps_[tag] = state;

    ++inflight_;
    statOpsIssued_->addData(1);
    if (state.status == SFUStatus::Success) {
        statPrimitiveElems_->addData(primitiveProcessedElems(state.desc));
    } else {
        statPrimitiveElems_->addData(0);
    }
    return true;
}

bool SFU::issuePrimitiveBatch(uint64_t descAddr, uint64_t tag)
{
    if (inflight_ >= maxInflight_) {
        statCreditStalls_->addData(1);
        return false;
    }

    PrimitiveBatchOpState state = {};
    state.descAddr = descAddr;
    state.tag = tag;
    state.status = SFUStatus::Success;
    state.processedElems = 0;

    if (!readPrimitiveBatchDescriptor(descAddr, &state.desc)) {
        state.status = (globalMem_ == nullptr) ? SFUStatus::GlobalMemoryUnavailable
                                               : SFUStatus::InvalidDescriptor;
    } else {
        state.status = validatePrimitiveBatchDescriptor(state.desc);
    }

    if (state.status == SFUStatus::Success) {
        for (uint32_t i = 0; i < state.desc.desc_count; ++i) {
            const uint64_t childDescAddr =
                state.desc.desc_array_gm_addr +
                static_cast<uint64_t>(i) * sizeof(SFUPrimitiveDesc);
            uint64_t childElems = 0;
            if (!executePrimitiveDesc(childDescAddr, tag + i + 1, &childElems)) {
                state.status = SFUStatus::InvalidDescriptor;
                break;
            }
            state.processedElems += childElems;
        }
    }

    pendingPrimitiveBatchOps_[tag] = state;

    ++inflight_;
    statOpsIssued_->addData(1);
    statPrimitiveElems_->addData(state.processedElems);
    return true;
}

bool SFU::issueJob(uint64_t descAddr, uint64_t tag)
{
    auto existing = pendingJobOps_.find(tag);
    if (existing != pendingJobOps_.end()) {
        // Duplicate tags poison the shared tag identity; preserving either job would be ambiguous.
        abortDistributedSoftmaxJob(&existing->second);
        existing->second.status = SFUStatus::InvalidDescriptor;
        existing->second.stage = SoftmaxJobStage::Complete;
        statRetryEvents_->addData(1);
        return true;
    }
    if (inflight_ >= maxInflight_) {
        statCreditStalls_->addData(1);
        return false;
    }

    JobOpState state = {};
    state.descAddr = descAddr;
    state.tag = tag;
    state.status = SFUStatus::Success;
    state.processedElems = 0;

    if (!readJobDescriptor(descAddr, &state.desc)) {
        state.status = (globalMem_ == nullptr) ? SFUStatus::GlobalMemoryUnavailable
                                               : SFUStatus::InvalidDescriptor;
    } else {
        state.status = validateJobDescriptor(state.desc);
    }

    if (state.status == SFUStatus::Success &&
        (state.desc.flags & SFU_JOB_FLAG_DISTRIBUTED_COLUMNS) != 0) {
        for (uint32_t row = 0; row < state.desc.rows; ++row) {
            if (distributedSoftmaxRowAborted(distributedSoftmaxReducerKey(
                    state.desc.job_id, state.tag, state.desc.owner_core, row))) {
                // Abort tombstones reserve this transport identity until a generation exists.
                state.status = SFUStatus::InvalidDescriptor;
                statRetryEvents_->addData(1);
                break;
            }
        }
    }

    const bool tensorRowEngineJob = state.status == SFUStatus::Success &&
        (state.desc.flags & SFU_JOB_FLAG_TENSOR_ROW_ENGINE) != 0;
    if (tensorRowEngineJob &&
        !readTensorJobParams(state.desc.params_addr, &state.tensorParams)) {
        state.status = SFUStatus::InvalidDescriptor;
    }
    if (tensorRowEngineJob && state.status == SFUStatus::Success) {
        state.attentionMode =
            state.tensorParams.version == SFU_SOFTMAX_JOB_PARAMS_VERSION_ATTENTION;
        state.attentionCausal = state.attentionMode &&
            (state.tensorParams.flags & SFU_SOFTMAX_PARAMS_FLAG_CAUSAL) != 0;
        if (state.attentionMode) {
            const float headDim = static_cast<float>(state.tensorParams.reserved0);
            state.attentionScale = 1.0f / std::sqrt(headDim);
            statAttentionJobs_->addData(1);
        }
    }
    const bool tensorRowEngineBusy = tensorRowEngineJob &&
        std::any_of(pendingJobOps_.begin(), pendingJobOps_.end(),
                    [](const auto& entry) {
                        return entry.second.status == SFUStatus::Pending &&
                            (entry.second.desc.flags & SFU_JOB_FLAG_TENSOR_ROW_ENGINE) != 0;
                    });
    if (tensorRowEngineBusy) {
        statCreditStalls_->addData(1);
        return false;
    }
    const bool rowEngineJob = state.status == SFUStatus::Success &&
        (state.desc.flags & SFU_JOB_FLAG_ROW_ENGINE_MODEL) != 0;
    if (state.status == SFUStatus::Success && !rowEngineJob && !executeJob(&state)) {
        abortDistributedSoftmaxJob(&state);
        state.status = SFUStatus::InvalidDescriptor;
    }
    if (rowEngineJob) {
        uint64_t maxCycles = 0;
        uint64_t expSumCycles = 0;
        uint64_t normalizeCycles = 0;
        uint64_t maxStartCycles = 0;
        uint64_t maxEndCycles = 0;
        uint64_t expSumStartCycles = 0;
        uint64_t expSumEndCycles = 0;
        uint64_t normalizeStartCycles = 0;
        uint64_t normalizeEndCycles = 0;
        SFUJobDesc modeledDesc = state.desc;
        if (tensorRowEngineJob) {
            modeledDesc.rows = static_cast<uint32_t>(
                ceilDiv(state.desc.rows, state.desc.worker_cores));
        }
        state.rowEngineModeledCycles = rowEngineModeledCycles(
            modeledDesc, &maxCycles, &expSumCycles, &normalizeCycles,
            &maxStartCycles, &maxEndCycles, &expSumStartCycles, &expSumEndCycles,
            &normalizeStartCycles, &normalizeEndCycles);
        state.rowEngineIssueTick = getCurrentSimCycle();
        state.rowEngineStartTick = tensorRowEngineJob
            ? state.rowEngineIssueTick
            : std::max(state.rowEngineIssueTick, rowEngineFreeTick_);
        state.rowEngineReadyTick = 0;
        if (tensorRowEngineJob && state.attentionMode) {
            state.attentionRsqrtReadyTick = state.rowEngineIssueTick + ceilMulDiv(
                rowEngineRsqrtLatency_, rowEngineTimebaseTicksPerSecond_,
                rowEngineAcceleratorClockHz_);
        }
        if (!tensorRowEngineJob) {
            const uint64_t modeledTicks = ceilMulDiv(
                state.rowEngineModeledCycles,
                rowEngineTimebaseTicksPerSecond_,
                rowEngineAcceleratorClockHz_);
            state.rowEngineReadyTick = state.rowEngineStartTick + modeledTicks;
            rowEngineFreeTick_ = state.rowEngineReadyTick;
        }
        state.status = SFUStatus::Pending;
        statRowEngineJobs_->addData(1);
        statRowEngineRows_->addData(state.desc.rows);
        statRowEngineMaxCycles_->addData(maxCycles);
        statRowEngineExpSumCycles_->addData(expSumCycles);
        statRowEngineNormalizeCycles_->addData(normalizeCycles);
        statRowEngineMaxStartCycles_->addData(maxStartCycles);
        statRowEngineMaxEndCycles_->addData(maxEndCycles);
        statRowEngineExpSumStartCycles_->addData(expSumStartCycles);
        statRowEngineExpSumEndCycles_->addData(expSumEndCycles);
        statRowEngineNormalizeStartCycles_->addData(normalizeStartCycles);
        statRowEngineNormalizeEndCycles_->addData(normalizeEndCycles);
        statRowEngineModeledCycles_->addData(state.rowEngineModeledCycles);
        statRowEngineQueueWaitCycles_->addData(ceilMulDiv(
            state.rowEngineStartTick - state.rowEngineIssueTick,
            rowEngineAcceleratorClockHz_,
            rowEngineTimebaseTicksPerSecond_));
        statRowEngineIssueTick_->addData(state.rowEngineIssueTick);
        statRowEngineStartTick_->addData(state.rowEngineStartTick);
        if (!tensorRowEngineJob) {
            statRowEngineReadyTick_->addData(state.rowEngineReadyTick);
        }
    }
    if (state.status != SFUStatus::Success && state.status != SFUStatus::Pending && verbose_ > 0) {
        output_.verbose(
            CALL_INFO,
            1,
            0,
            "SFU job rejected core=%" PRIu32 " status=%" PRIu64
            " input=0x%" PRIx64 " output=0x%" PRIx64
            " op=%" PRIu32 " dtype=%" PRIu32 " rows=%" PRIu32 " cols=%" PRIu32
            " chunk=%" PRIu32 " workers=%" PRIu32 " slot=%" PRIu32
            " active_workers=%" PRIu32 " flags=0x%" PRIx32 "\n",
            coreId_,
            static_cast<uint64_t>(state.status),
            state.desc.input0_addr,
            state.desc.output_addr,
            state.desc.op_type,
            state.desc.dtype,
            state.desc.rows,
            state.desc.cols,
            state.desc.chunk_elems,
            state.desc.worker_cores,
            state.desc.reserved0,
            activeWorkerCores_,
            state.desc.flags);
    }

    pendingJobOps_[tag] = state;

    if (tensorRowEngineJob && state.status == SFUStatus::Pending &&
        !state.attentionMode &&
        !startTensorRowEngineJob(tag)) {
        pendingJobOps_[tag].status = SFUStatus::InvalidDescriptor;
    }

    ++inflight_;
    statOpsIssued_->addData(1);
    if (state.status == SFUStatus::Success || state.status == SFUStatus::Pending) {
        statPrimitiveElems_->addData(state.processedElems);
    }
    return true;
}

bool SFU::wait(uint64_t tag, uint64_t* status)
{
    SFUStatus opStatus = SFUStatus::Success;
    auto it = pendingSoftmaxOps_.find(tag);
    if (it != pendingSoftmaxOps_.end()) {
        SoftmaxOpState& state = it->second;
        if (state.status == SFUStatus::Pending && tileGlobalStatsReady(state)) {
            state.normalizeReady = true;
            state.status = normalizeTile(&state) ? SFUStatus::Success : SFUStatus::InvalidDescriptor;
            if (state.status == SFUStatus::Success) {
                statPartialDone_->addData(state.rowStats.size());
            }
        }

        opStatus = state.status;
        if (opStatus == SFUStatus::Pending) {
            statCrossTileWaitCycles_->addData(1);
            if (status != nullptr) {
                *status = static_cast<uint64_t>(SFUStatus::Pending);
            }
            return false;
        }

        pendingSoftmaxOps_.erase(it);
    } else {
        auto primitiveIt = pendingPrimitiveOps_.find(tag);
        if (primitiveIt != pendingPrimitiveOps_.end()) {
            opStatus = primitiveIt->second.status;
            pendingPrimitiveOps_.erase(primitiveIt);
        } else {
            auto batchIt = pendingPrimitiveBatchOps_.find(tag);
            if (batchIt != pendingPrimitiveBatchOps_.end()) {
                opStatus = batchIt->second.status;
                pendingPrimitiveBatchOps_.erase(batchIt);
            } else {
                auto jobIt = pendingJobOps_.find(tag);
                if (jobIt != pendingJobOps_.end()) {
                    JobOpState& state = jobIt->second;
                    if (state.status == SFUStatus::Pending &&
                        (state.desc.flags & SFU_JOB_FLAG_ROW_ENGINE_MODEL) != 0) {
                        if ((state.desc.flags & SFU_JOB_FLAG_TENSOR_ROW_ENGINE) != 0) {
                            if (state.attentionMode && !state.tensorStarted) {
                                if (getCurrentSimCycle() >= state.attentionRsqrtReadyTick) {
                                    statAttentionRsqrtReadyTick_->addData(getCurrentSimCycle());
                                    if (!startTensorRowEngineJob(state.tag)) {
                                        state.status = SFUStatus::InvalidDescriptor;
                                    }
                                } else {
                                    statRowEngineWaitPolls_->addData(1);
                                }
                            }
                            if (state.tensorStarted) {
                                finishTensorJobIfReady(&state);
                            }
                        } else if (getCurrentSimCycle() >= state.rowEngineReadyTick) {
                            if (executeJob(&state)) {
                                state.status = SFUStatus::Success;
                                statRowEngineCompletedJobs_->addData(1);
                                statRowEngineCompletionObservedTick_->addData(getCurrentSimCycle());
                            } else {
                                state.status = SFUStatus::InvalidDescriptor;
                            }
                        } else {
                            statRowEngineWaitPolls_->addData(1);
                        }
                    }
                    if (state.status == SFUStatus::Pending &&
                        (state.desc.flags & SFU_JOB_FLAG_DISTRIBUTED_COLUMNS) != 0 &&
                        !advanceDistributedSoftmaxJob(&state)) {
                        abortDistributedSoftmaxJob(&state);
                        state.status = SFUStatus::InvalidDescriptor;
                    }
                    opStatus = state.status;
                    if (opStatus == SFUStatus::Pending) {
                        if ((state.desc.flags & SFU_JOB_FLAG_ROW_ENGINE_MODEL) == 0) {
                            statCrossTileWaitCycles_->addData(1);
                        }
                        if (status != nullptr) {
                            *status = static_cast<uint64_t>(SFUStatus::Pending);
                        }
                        return false;
                    }
                    if ((state.desc.flags & SFU_JOB_FLAG_TENSOR_ROW_ENGINE) != 0) {
                        statTensorGuestWaitObservedTick_->addData(getCurrentSimCycle());
                    }
                    clearDistributedReductionResponseInbox(state);
                    pendingJobOps_.erase(jobIt);
                }
            }
        }
    }

    if (inflight_ > 0) {
        --inflight_;
    }
    if (status != nullptr) {
        *status = static_cast<uint64_t>(opStatus);
    }
    return true;
}

bool SFU::completionTick(uint64_t tag, uint64_t* tick) const
{
    const auto it = pendingJobOps_.find(tag);
    if (it == pendingJobOps_.end() ||
        (it->second.desc.flags & SFU_JOB_FLAG_ROW_ENGINE_MODEL) == 0 ||
        it->second.status != SFUStatus::Pending) {
        return false;
    }
    if (tick != nullptr) {
        const JobOpState& state = it->second;
        if ((state.desc.flags & SFU_JOB_FLAG_TENSOR_ROW_ENGINE) != 0) {
            return false;
        }
        *tick = state.rowEngineReadyTick;
    }
    return true;
}

bool SFU::readPrimitiveDescriptor(uint64_t descAddr, SFUPrimitiveDesc* desc)
{
    if (globalMem_ == nullptr || desc == nullptr || descAddr == 0) {
        return false;
    }

    std::vector<uint8_t> raw;
    globalMem_->rd_from_globalmem(descAddr, sizeof(SFUPrimitiveDesc), raw);
    if (raw.size() != sizeof(SFUPrimitiveDesc)) {
        return false;
    }

    std::memcpy(desc, raw.data(), sizeof(SFUPrimitiveDesc));
    return true;
}

bool SFU::readPrimitiveBatchDescriptor(uint64_t descAddr, SFUPrimitiveBatchDesc* desc)
{
    if (globalMem_ == nullptr || desc == nullptr || descAddr == 0) {
        return false;
    }

    std::vector<uint8_t> raw;
    globalMem_->rd_from_globalmem(descAddr, sizeof(SFUPrimitiveBatchDesc), raw);
    if (raw.size() != sizeof(SFUPrimitiveBatchDesc)) {
        return false;
    }

    std::memcpy(desc, raw.data(), sizeof(SFUPrimitiveBatchDesc));
    return true;
}

bool SFU::readJobDescriptor(uint64_t descAddr, SFUJobDesc* desc)
{
    if (globalMem_ == nullptr || desc == nullptr || descAddr == 0) {
        return false;
    }

    std::vector<uint8_t> raw;
    globalMem_->rd_from_globalmem(descAddr, sizeof(SFUJobDesc), raw);
    if (raw.size() != sizeof(SFUJobDesc)) {
        return false;
    }

    std::memcpy(desc, raw.data(), sizeof(SFUJobDesc));
    return true;
}

bool SFU::readTensorJobParams(uint64_t paramsAddr, SFUSoftmaxJobParamsV1* params)
{
    if (globalMem_ == nullptr || params == nullptr || paramsAddr == 0) {
        return false;
    }
    std::vector<uint8_t> raw;
    globalMem_->rd_from_globalmem(paramsAddr, sizeof(SFUSoftmaxJobParamsV1), raw);
    if (raw.size() != sizeof(SFUSoftmaxJobParamsV1)) {
        return false;
    }
    std::memcpy(params, raw.data(), sizeof(SFUSoftmaxJobParamsV1));
    const bool legacy = params->version == SFU_SOFTMAX_JOB_PARAMS_VERSION &&
        params->flags == 0;
    const bool attention =
        params->version == SFU_SOFTMAX_JOB_PARAMS_VERSION_ATTENTION &&
        (params->flags & SFU_SOFTMAX_PARAMS_FLAG_ATTENTION) != 0 &&
        (params->flags & ~(SFU_SOFTMAX_PARAMS_FLAG_ATTENTION |
                           SFU_SOFTMAX_PARAMS_FLAG_CAUSAL)) == 0 &&
        params->reserved0 != 0;
    return params->magic == SFU_SOFTMAX_JOB_PARAMS_MAGIC &&
        (legacy || attention) &&
        params->size_bytes == sizeof(SFUSoftmaxJobParamsV1) &&
        params->hbm_layout == SFU_SOFTMAX_HBM_LAYOUT_BAND_STRIPED &&
        params->data_node_mask != 0 && params->node_stride_bytes != 0 &&
        params->rows_per_band != 0 && params->coordinator_core == coreId_;
}

SFUStatus SFU::validatePrimitiveDescriptor(const SFUPrimitiveDesc& desc) const
{
    if (desc.dtype != GOLEM_DTYPE_FP32_VALUE) {
        return SFUStatus::UnsupportedElemBytes;
    }
    if (desc.input0_gm_addr == 0 || desc.output_gm_addr == 0 || desc.elem_count == 0) {
        return SFUStatus::InvalidShape;
    }
    if (effectiveStride(desc.input0_stride_bytes) < sizeof(float) ||
        effectiveStride(desc.output_stride_bytes) < sizeof(float)) {
        return SFUStatus::InvalidShape;
    }

    switch (desc.op) {
        case static_cast<uint32_t>(SFUPrimitiveOp::EXP):
        case static_cast<uint32_t>(SFUPrimitiveOp::LOG):
        case static_cast<uint32_t>(SFUPrimitiveOp::RECIPROCAL):
        case static_cast<uint32_t>(SFUPrimitiveOp::RSQRT):
        case static_cast<uint32_t>(SFUPrimitiveOp::TANH):
        case static_cast<uint32_t>(SFUPrimitiveOp::SIGMOID):
        case static_cast<uint32_t>(SFUPrimitiveOp::REDUCE_MAX):
        case static_cast<uint32_t>(SFUPrimitiveOp::REDUCE_SUM):
            return SFUStatus::Success;
        default:
            return SFUStatus::InvalidDescriptor;
    }
}

SFUStatus SFU::validatePrimitiveBatchDescriptor(const SFUPrimitiveBatchDesc& desc) const
{
    if (desc.desc_array_gm_addr == 0 || desc.desc_count == 0 ||
        desc.desc_count > GOLEM_SFU_PRIMITIVE_BATCH_MAX_DESCS) {
        return SFUStatus::InvalidShape;
    }
    return SFUStatus::Success;
}

SFUStatus SFU::validateJobDescriptor(const SFUJobDesc& desc) const
{
    constexpr uint32_t supportedFlags = SFU_JOB_FLAG_DISTRIBUTED_COLUMNS |
        SFU_JOB_FLAG_DISTRIBUTED_ABORT | SFU_JOB_FLAG_ROW_ENGINE_MODEL |
        SFU_JOB_FLAG_TENSOR_ROW_ENGINE;
    if ((desc.flags & ~supportedFlags) != 0 ||
        ((desc.flags & SFU_JOB_FLAG_DISTRIBUTED_ABORT) != 0 &&
         (desc.flags & SFU_JOB_FLAG_DISTRIBUTED_COLUMNS) == 0)) {
        return SFUStatus::InvalidDescriptor;
    }
    if (desc.dtype != GOLEM_DTYPE_FP32_VALUE) {
        return SFUStatus::UnsupportedElemBytes;
    }
    if (desc.input0_addr == 0 || desc.output_addr == 0) {
        return SFUStatus::InvalidShape;
    }
    if (desc.worker_cores == 0 || desc.chunk_elems == 0) {
        return SFUStatus::InvalidShape;
    }
    if ((desc.flags & SFU_JOB_FLAG_DISTRIBUTED_COLUMNS) != 0) {
        if (desc.reserved0 >= desc.worker_cores ||
            desc.worker_cores > activeWorkerCores_ || desc.owner_core > coreId_ ||
            coreId_ - desc.owner_core >= desc.worker_cores ||
            coreId_ - desc.owner_core != desc.reserved0) {
            return SFUStatus::InvalidShape;
        }
    }
    const bool tensorRowEngine =
        (desc.flags & SFU_JOB_FLAG_TENSOR_ROW_ENGINE) != 0;
    if (tensorRowEngine &&
        ((desc.flags & SFU_JOB_FLAG_ROW_ENGINE_MODEL) == 0 ||
         desc.params_addr == 0 || desc.scratch_addr == 0 ||
         desc.owner_core != coreId_ || desc.worker_cores > activeWorkerCores_)) {
        return SFUStatus::InvalidShape;
    }
    if ((desc.flags & SFU_JOB_FLAG_ROW_ENGINE_MODEL) != 0 && !tensorRowEngine) {
        const uint64_t rowBytes = static_cast<uint64_t>(desc.cols) * sizeof(float);
        const uint64_t contextBytes = rowEngineScratchpadBytes_ / rowEngineContexts_;
        if ((desc.flags & SFU_JOB_FLAG_DISTRIBUTED_COLUMNS) != 0 ||
            desc.worker_cores != 1 || rowBytes > contextBytes) {
            return SFUStatus::InvalidShape;
        }
    }

    switch (desc.op_type) {
        case static_cast<uint32_t>(SFUJobOp::SOFTMAX_ROW):
            if (desc.rows == 0 || desc.cols == 0) {
                return SFUStatus::InvalidShape;
            }
            return SFUStatus::Success;
        case static_cast<uint32_t>(SFUJobOp::ELEMENTWISE):
        case static_cast<uint32_t>(SFUJobOp::REDUCE):
            if (desc.elem_count == 0) {
                return SFUStatus::InvalidShape;
            }
            return SFUStatus::Success;
        default:
            return SFUStatus::InvalidDescriptor;
    }
}

bool SFU::executePrimitiveDesc(uint64_t descAddr, uint64_t tag, uint64_t* processedElems)
{
    PrimitiveOpState state = {};
    state.descAddr = descAddr;
    state.tag = tag;
    state.status = SFUStatus::Success;

    if (!readPrimitiveDescriptor(descAddr, &state.desc)) {
        return false;
    }
    state.status = validatePrimitiveDescriptor(state.desc);
    if (state.status != SFUStatus::Success) {
        return false;
    }
    if (!readPrimitiveInput(state.desc, &state.input0)) {
        return false;
    }
    if (!executePrimitive(&state)) {
        return false;
    }
    if (!writePrimitiveOutput(state.desc, state.output)) {
        return false;
    }
    if (processedElems != nullptr) {
        *processedElems = primitiveProcessedElems(state.desc);
    }
    return true;
}

bool SFU::readPrimitiveInput(const SFUPrimitiveDesc& desc, std::vector<float>* values)
{
    if (globalMem_ == nullptr || values == nullptr) {
        return false;
    }

    values->assign(desc.elem_count, 0.0f);
    const uint32_t stride = effectiveStride(desc.input0_stride_bytes);
    if (stride == sizeof(float)) {
        std::vector<uint8_t> raw;
        const size_t byteCount = static_cast<size_t>(desc.elem_count) * sizeof(float);
        globalMem_->rd_from_globalmem(desc.input0_gm_addr, byteCount, raw);
        if (raw.size() != byteCount) {
            return false;
        }
        std::memcpy(values->data(), raw.data(), byteCount);
        return true;
    }

    for (uint32_t i = 0; i < desc.elem_count; ++i) {
        std::vector<uint8_t> raw;
        globalMem_->rd_from_globalmem(desc.input0_gm_addr + static_cast<uint64_t>(i) * stride,
                                      sizeof(float), raw);
        if (raw.size() != sizeof(float)) {
            return false;
        }
        float value = 0.0f;
        std::memcpy(&value, raw.data(), sizeof(float));
        (*values)[i] = value;
    }
    return true;
}

bool SFU::executePrimitive(PrimitiveOpState* state)
{
    if (state == nullptr) {
        return false;
    }

    if (state->desc.op == static_cast<uint32_t>(SFUPrimitiveOp::REDUCE_MAX)) {
        auto maxIt = std::max_element(state->input0.begin(), state->input0.end());
        if (maxIt == state->input0.end()) {
            return false;
        }
        state->output.assign(1, 0.0f);
        state->output[0] = *maxIt;
        return true;
    }
    if (state->desc.op == static_cast<uint32_t>(SFUPrimitiveOp::REDUCE_SUM)) {
        double sum = 0.0;
        for (float value : state->input0) {
            sum += static_cast<double>(value);
        }
        state->output.assign(1, 0.0f);
        state->output[0] = static_cast<float>(sum);
        return true;
    }

    state->output.assign(state->input0.size(), 0.0f);
    for (size_t i = 0; i < state->input0.size(); ++i) {
        const float value = state->input0[i];
        switch (state->desc.op) {
            case static_cast<uint32_t>(SFUPrimitiveOp::EXP):
                state->output[i] = static_cast<float>(std::exp(static_cast<double>(value)));
                break;
            case static_cast<uint32_t>(SFUPrimitiveOp::LOG):
                state->output[i] = static_cast<float>(std::log(static_cast<double>(value)));
                break;
            case static_cast<uint32_t>(SFUPrimitiveOp::RECIPROCAL):
                state->output[i] = 1.0f / value;
                break;
            case static_cast<uint32_t>(SFUPrimitiveOp::RSQRT):
                state->output[i] = 1.0f / static_cast<float>(std::sqrt(static_cast<double>(value)));
                break;
            case static_cast<uint32_t>(SFUPrimitiveOp::TANH):
                state->output[i] = static_cast<float>(std::tanh(static_cast<double>(value)));
                break;
            case static_cast<uint32_t>(SFUPrimitiveOp::SIGMOID):
                state->output[i] = 1.0f / (1.0f + static_cast<float>(std::exp(-static_cast<double>(value))));
                break;
            default:
                return false;
        }
    }
    return true;
}

bool SFU::writePrimitiveOutput(const SFUPrimitiveDesc& desc, const std::vector<float>& values)
{
    const bool reductionOutput =
        (desc.op == static_cast<uint32_t>(SFUPrimitiveOp::REDUCE_MAX) ||
         desc.op == static_cast<uint32_t>(SFUPrimitiveOp::REDUCE_SUM)) &&
        values.size() == 1;
    if (globalMem_ == nullptr || (values.size() != desc.elem_count && !reductionOutput)) {
        return false;
    }

    const uint32_t stride = effectiveStride(desc.output_stride_bytes);
    if (stride == sizeof(float)) {
        std::vector<uint8_t> raw(values.size() * sizeof(float));
        std::memcpy(raw.data(), values.data(), raw.size());
        globalMem_->wr_to_globalmem(desc.output_gm_addr, raw.size(), raw);
        return true;
    }

    for (uint32_t i = 0; i < desc.elem_count; ++i) {
        std::vector<uint8_t> raw;
        appendBytes(raw, values[i]);
        globalMem_->wr_to_globalmem(desc.output_gm_addr + static_cast<uint64_t>(i) * stride,
                                    raw.size(), raw);
    }
    return true;
}

uint64_t SFU::primitiveProcessedElems(const SFUPrimitiveDesc& desc) const
{
    if ((desc.flags & GOLEM_SFU_PRIMITIVE_FLAG_REPEAT_CHUNK) == 0) {
        return desc.elem_count;
    }
    return desc.input1_gm_addr > desc.elem_count ? desc.input1_gm_addr : desc.elem_count;
}

bool SFU::executeJob(JobOpState* state)
{
    if (state == nullptr) {
        return false;
    }

    switch (state->desc.op_type) {
        case static_cast<uint32_t>(SFUJobOp::SOFTMAX_ROW):
            if ((state->desc.flags & SFU_JOB_FLAG_DISTRIBUTED_COLUMNS) != 0) {
                if ((state->desc.flags & SFU_JOB_FLAG_DISTRIBUTED_ABORT) != 0) {
                    abortDistributedSoftmaxJob(state);
                    state->stage = SoftmaxJobStage::Complete;
                    state->status = SFUStatus::InvalidDescriptor;
                    return true;
                }
                return executeDistributedSoftmaxRowJob(state);
            }
            return executeSoftmaxRowJob(state);
        default:
            return false;
    }
}

bool SFU::executeSoftmaxRowJob(JobOpState* state)
{
    if (state == nullptr || globalMem_ == nullptr) {
        return false;
    }

    const SFUJobDesc& desc = state->desc;
    const uint32_t workerCores = state->desc.worker_cores;
    const uint32_t chunkElems = state->desc.chunk_elems;

    for (uint32_t rowBandBegin = 0; rowBandBegin < desc.rows;
         rowBandBegin += GOLEM_SFU_JOB_SOFTMAX_ROW_BAND_ROWS) {
        SoftmaxJobRowBandState band = {};
        band.rowBegin = rowBandBegin;
        band.rowEnd = std::min(desc.rows, rowBandBegin + GOLEM_SFU_JOB_SOFTMAX_ROW_BAND_ROWS);
        band.bandRows = band.rowEnd - band.rowBegin;
        const uint32_t bandRows = band.bandRows;
        band.localMax.assign(static_cast<size_t>(workerCores) * bandRows,
                             -std::numeric_limits<double>::infinity());
        band.localSum.assign(static_cast<size_t>(workerCores) * bandRows, 0.0);
        band.globalMax.assign(bandRows, -std::numeric_limits<double>::infinity());
        band.globalSum.assign(bandRows, 0.0);

        for (uint32_t worker = 0; worker < workerCores; ++worker) {
            const uint32_t begin = workerColumnBegin(desc.cols, worker, workerCores);
            const uint32_t end = workerColumnEnd(desc.cols, worker, workerCores);
            for (uint32_t col = begin; col < end; col += chunkElems) {
                const uint32_t chunkEnd = std::min(end, col + chunkElems);
                for (uint32_t rowIdx = band.rowBegin; rowIdx < band.rowEnd; ++rowIdx) {
                    std::vector<float> chunk;
                    if (!readSoftmaxJobChunk(globalMem_, desc, rowIdx, col, chunkEnd, &chunk)) {
                        return false;
                    }
                    statJobSoftmaxMaxChunks_->addData(1);
                    const uint32_t bandRow = rowIdx - band.rowBegin;
                    double& localMax = band.localMax[worker * bandRows + bandRow];
                    for (float value : chunk) {
                        localMax = std::max(localMax, static_cast<double>(value));
                    }
                }
            }
        }

        for (uint32_t bandRow = 0; bandRow < bandRows; ++bandRow) {
            for (uint32_t worker = 0; worker < workerCores; ++worker) {
                band.globalMax[bandRow] =
                    std::max(band.globalMax[bandRow],
                             band.localMax[worker * bandRows + bandRow]);
            }
        }

        for (uint32_t worker = 0; worker < workerCores; ++worker) {
            const uint32_t begin = workerColumnBegin(desc.cols, worker, workerCores);
            const uint32_t end = workerColumnEnd(desc.cols, worker, workerCores);
            for (uint32_t col = begin; col < end; col += chunkElems) {
                const uint32_t chunkEnd = std::min(end, col + chunkElems);
                for (uint32_t rowIdx = band.rowBegin; rowIdx < band.rowEnd; ++rowIdx) {
                    std::vector<float> chunk;
                    if (!readSoftmaxJobChunk(globalMem_, desc, rowIdx, col, chunkEnd, &chunk)) {
                        return false;
                    }
                    const uint32_t bandRow = rowIdx - band.rowBegin;
                    double& localSum = band.localSum[worker * bandRows + bandRow];
                    for (float& value : chunk) {
                        const double expValue =
                            std::exp(static_cast<double>(value) - band.globalMax[bandRow]);
                        value = static_cast<float>(expValue);
                        localSum += expValue;
                    }
                    if (!writeSoftmaxJobChunk(globalMem_, desc, rowIdx, col, chunk)) {
                        return false;
                    }
                    statJobSoftmaxSumChunks_->addData(1);
                }
            }
        }

        for (uint32_t bandRow = 0; bandRow < bandRows; ++bandRow) {
            for (uint32_t worker = 0; worker < workerCores; ++worker) {
                band.globalSum[bandRow] += band.localSum[worker * bandRows + bandRow];
            }
            if (band.globalSum[bandRow] == 0.0) {
                return false;
            }
        }

        for (uint32_t worker = 0; worker < workerCores; ++worker) {
            const uint32_t begin = workerColumnBegin(desc.cols, worker, workerCores);
            const uint32_t end = workerColumnEnd(desc.cols, worker, workerCores);
            for (uint32_t col = begin; col < end; col += chunkElems) {
                const uint32_t chunkEnd = std::min(end, col + chunkElems);
                for (uint32_t rowIdx = band.rowBegin; rowIdx < band.rowEnd; ++rowIdx) {
                    std::vector<float> chunk;
                    if (!readSoftmaxJobOutputChunk(globalMem_, desc, rowIdx, col, chunkEnd, &chunk)) {
                        return false;
                    }
                    const uint32_t bandRow = rowIdx - band.rowBegin;
                    const float invSum = static_cast<float>(1.0 / band.globalSum[bandRow]);
                    for (float& value : chunk) {
                        value *= invSum;
                    }
                    if (!writeSoftmaxJobChunk(globalMem_, desc, rowIdx, col, chunk)) {
                        return false;
                    }
                    statJobSoftmaxNormChunks_->addData(1);
                }
            }
        }

        for (uint32_t rowIdx = band.rowBegin; rowIdx < band.rowEnd; ++rowIdx) {
            statSoftmaxRows_->addData(1);
        }
    }

    state->processedElems = static_cast<uint64_t>(desc.rows) * desc.cols;
    return true;
}

uint64_t SFU::rowEngineModeledCycles(const SFUJobDesc& desc,
                                     uint64_t* maxCycles,
                                     uint64_t* expSumCycles,
                                     uint64_t* normalizeCycles,
                                     uint64_t* maxStartCycles,
                                     uint64_t* maxEndCycles,
                                     uint64_t* expSumStartCycles,
                                     uint64_t* expSumEndCycles,
                                     uint64_t* normalizeStartCycles,
                                     uint64_t* normalizeEndCycles) const
{
    const uint64_t rows = desc.rows;
    const uint64_t maxPerRow = ceilDiv(desc.cols, rowEngineVectorLanes_);
    const uint64_t expSumPerRow = ceilDiv(desc.cols, rowEngineExpLanes_);
    const uint64_t normalizePerRow = ceilDiv(desc.cols, rowEngineVectorLanes_);
    if (maxCycles != nullptr) {
        *maxCycles = rows * maxPerRow;
    }
    if (expSumCycles != nullptr) {
        *expSumCycles = rows * expSumPerRow;
    }
    if (normalizeCycles != nullptr) {
        *normalizeCycles = rows * normalizePerRow;
    }
    const uint64_t pipelineDrain = rowEngineReductionTreeLatency_ +
        rowEngineExpLatency_ + rowEngineReciprocalLatency_;
    const uint64_t rowInterval = (rowEngineContexts_ > 1 && rows > 0)
        ? std::max(maxPerRow, std::max(expSumPerRow, normalizePerRow))
        : (maxPerRow + expSumPerRow + normalizePerRow);
    const uint64_t firstRow = maxPerRow + expSumPerRow + normalizePerRow;
    if (maxStartCycles != nullptr) {
        *maxStartCycles = 0;
    }
    if (maxEndCycles != nullptr) {
        *maxEndCycles = rows == 0 ? 0 : (rows - 1) * rowInterval + maxPerRow;
    }
    if (expSumStartCycles != nullptr) {
        *expSumStartCycles = maxPerRow;
    }
    if (expSumEndCycles != nullptr) {
        *expSumEndCycles = rows == 0 ? 0 : (rows - 1) * rowInterval + maxPerRow + expSumPerRow;
    }
    if (normalizeStartCycles != nullptr) {
        *normalizeStartCycles = maxPerRow + expSumPerRow;
    }
    if (normalizeEndCycles != nullptr) {
        *normalizeEndCycles = rows == 0 ? 0 : (rows - 1) * rowInterval + firstRow;
    }
    if (rowEngineContexts_ > 1 && rows > 0) {
        return firstRow + (rows - 1) * rowInterval + pipelineDrain;
    }
    return rows * (maxPerRow + expSumPerRow + normalizePerRow) + pipelineDrain;
}

bool SFU::finishTensorJobIfReady(JobOpState* state)
{
    if (state == nullptr || !state->tensorDmaComplete) {
        return false;
    }
    if (state->status == SFUStatus::Pending) {
        state->rowEngineReadyTick = getCurrentSimCycle();
        state->status = SFUStatus::Success;
        state->processedElems = static_cast<uint64_t>(state->desc.rows) * state->desc.cols;
        statRowEngineReadyTick_->addData(state->rowEngineReadyTick);
        statRowEngineCompletedJobs_->addData(1);
        statRowEngineCompletionObservedTick_->addData(getCurrentSimCycle());
    }
    return true;
}

uint64_t SFU::tensorWorkerHostAddress(const ReductionTransportMessage& message,
                                      uint32_t row,
                                      bool output) const
{
    const uint64_t base = output ? message.outputAddr : message.inputAddr;
    const uint32_t band = row / message.rowsPerBand;
    const uint32_t rowInBand = row % message.rowsPerBand;
    const uint32_t nodeCount = __builtin_popcount(message.dataNodeMask);
    uint32_t selectedNode = 0;
    uint32_t ordinal = 0;
    for (uint32_t bit = 0; bit < 32; ++bit) {
        if ((message.dataNodeMask & (1u << bit)) == 0) {
            continue;
        }
        if (ordinal == band % nodeCount) {
            selectedNode = bit;
            break;
        }
        ++ordinal;
    }
    const uint64_t localBand = band / nodeCount;
    const uint64_t rowBytes = static_cast<uint64_t>(message.expectedCols) * sizeof(float);
    return base + static_cast<uint64_t>(selectedNode) * message.nodeStrideBytes +
        (localBand * message.rowsPerBand + rowInBand) * rowBytes;
}

void SFU::handleTensorRowDispatch(const ReductionTransportMessage& message)
{
    if (message.workerCore != coreId_ || message.expectedRows == 0 ||
        message.expectedCols == 0 || message.rowsPerBand == 0 ||
        message.dataNodeMask == 0 || message.nodeStrideBytes == 0 ||
        globalMem_ == nullptr) {
        rejectTensorRowDispatch(message);
        return;
    }
    if (!tensorWorkerOps_.empty()) {
        rejectTensorRowDispatch(message);
        return;
    }
    const TensorWorkerKey key(0, message.tag, message.row);
    TensorWorkerState worker = {};
    worker.dispatch = message;
    worker.scratchAddr = globalMem_->getBaseAddr() + 0x2000;
    worker.nextRow = message.row;
    worker.rowsCompleted = 0;
    const uint64_t rowBytes = static_cast<uint64_t>(message.expectedCols) * sizeof(float);
    const uint32_t contextCount = std::min(message.expectedRows, rowEngineContexts_);
    if (rowBytes * contextCount > rowEngineScratchpadBytes_ ||
        worker.scratchAddr + rowBytes * contextCount >
            globalMem_->getBaseAddr() + globalMem_->getSize()) {
        rejectTensorRowDispatch(message);
        return;
    }
    worker.contexts.resize(contextCount);
    for (uint32_t contextIndex = 0; contextIndex < contextCount; ++contextIndex) {
        TensorWorkerState::Context& context = worker.contexts[contextIndex];
        context.scratchAddr = worker.scratchAddr + contextIndex * rowBytes;
    }
    if (!tensorWorkerOps_.emplace(key, std::move(worker)).second) {
        rejectTensorRowDispatch(message);
        return;
    }
    statTensorWorkerDispatchTick_->addData(getCurrentSimCycle());
    for (uint32_t contextIndex = 0; contextIndex < contextCount; ++contextIndex) {
        issueTensorInputDma(key, contextIndex);
    }
}

bool SFU::reserveAttentionScoreSlot(
    uint32_t slot, const AttentionClusterTag& tag, size_t elements)
{
    return attentionClusterEnable_ &&
        attentionScoreFifo_.reserve(slot, tag, elements);
}

uint64_t SFU::scheduleScoreFifoPort(uint64_t bytes, bool write)
{
    const uint64_t now = rowEngineCurrentCycle();
    uint64_t& nextCycle = write ? attentionScoreFifoNextWriteCycle_
                                : attentionScoreFifoNextReadCycle_;
    const uint64_t start = std::max(now, nextCycle);
    const uint64_t transfer = ceilDiv(bytes, attentionScoreFifoBytesPerCycle_);
    nextCycle = start + transfer;
    statAttentionClusterScoreFifoPortWaitCycles_->addData(start - now);
    return std::max<uint64_t>(
        1, start - now + attentionScoreFifoLatencyCycles_ + transfer);
}

bool SFU::writeAttentionScoreBeatAsync(
    uint32_t slot, const AttentionClusterTag& tag, size_t offset,
    const std::vector<float>& values, uint64_t transferTag,
    AttentionScoreWriteCallback callback)
{
    if (!attentionClusterEnable_ || !callback || values.empty() ||
        pendingAttentionScoreWrites_.size() >= attentionScoreFifoQueueDepth_) {
        statAttentionClusterScoreFifoBackpressureStalls_->addData(1);
        return false;
    }
    if (attentionScoreFifo_.ready(slot, tag) ||
        !attentionScoreFifo_.reserveWrite(slot, tag, offset, values.size())) {
        statAttentionClusterScoreFifoBackpressureStalls_->addData(1);
        return false;
    }
    const uint64_t pendingId = nextAttentionScoreWriteId_++;
    PendingAttentionScoreWrite pending;
    pending.generation = tag.generation;
    pending.slot = slot;
    pending.tag = tag;
    pending.offset = offset;
    pending.values = values;
    pending.transferTag = transferTag;
    pending.callback = std::move(callback);
    pendingAttentionScoreWrites_.emplace(pendingId, std::move(pending));
    const uint64_t delay = scheduleScoreFifoPort(values.size() * sizeof(float), true);
    rowEngineSelfLink_->send(
        delay, new TensorRowEngineEvent(
            0, 0, 0, TensorRowEngineStage::Max,
            TensorRowEngineEventKind::ScoreFifoWriteDone, pendingId,
            tag.generation));
    return true;
}

void SFU::completeAttentionScoreWrite(uint64_t pendingId)
{
    auto it = pendingAttentionScoreWrites_.find(pendingId);
    if (it == pendingAttentionScoreWrites_.end()) return;
    PendingAttentionScoreWrite pending = std::move(it->second);
    pendingAttentionScoreWrites_.erase(it);
    const bool ok = attentionScoreFifo_.commitWrite(
        pending.slot, pending.tag, pending.offset, pending.values);
    if (ok) {
        statAttentionClusterScoreFifoProducerWrites_->addData(1);
        statAttentionClusterScoreFifoProducerBytes_->addData(
            pending.values.size() * sizeof(float));
    }
    pending.callback(ok, pending.transferTag);
}

bool SFU::releaseAttentionScoreSlot(
    uint32_t slot, const AttentionClusterTag& tag)
{
    const bool pending = std::any_of(
        pendingAttentionScoreWrites_.begin(), pendingAttentionScoreWrites_.end(),
        [slot, &tag](const auto& item) {
            return item.second.slot == slot && item.second.tag == tag;
        });
    return !pending && attentionScoreFifo_.release(slot, tag);
}

bool SFU::reserveAttentionPSlot(
    uint32_t slot, const AttentionClusterTag& tag, size_t elements)
{
    const AttentionClusterAdmission admission =
        attentionPSlotAdmission(slot, tag, elements);
    if (admission != AttentionClusterAdmission::Ready ||
        !attentionPFifo_.reserve(slot, tag, elements)) {
        if (admission == AttentionClusterAdmission::Retry) {
            statAttentionClusterPFifoBackpressureStalls_->addData(1);
        }
        return false;
    }
    statAttentionClusterPFifoReservations_->addData(1);
    return true;
}

AttentionClusterAdmission SFU::attentionPSlotAdmission(
    uint32_t slot, const AttentionClusterTag& tag, size_t elements) const
{
    if (!attentionClusterEnable_ || slot >= AttentionScoreFifo::kSlots ||
        tag.generation == 0 || elements == 0 ||
        elements > AttentionScoreFifo::kElementsPerSlot) {
        return AttentionClusterAdmission::Invalid;
    }
    if (!attentionPFifo_.occupied(slot)) {
        return AttentionClusterAdmission::Ready;
    }
    return attentionPFifo_.reserved(slot, tag)
        ? AttentionClusterAdmission::Invalid
        : AttentionClusterAdmission::Retry;
}

uint64_t SFU::schedulePFifoPort(uint64_t bytes, bool write)
{
    const uint64_t now = rowEngineCurrentCycle();
    uint64_t& nextCycle = write ? attentionPFifoNextWriteCycle_
                                : attentionPFifoNextReadCycle_;
    const uint64_t start = std::max(now, nextCycle);
    const uint64_t transfer = ceilDiv(bytes, attentionScoreFifoBytesPerCycle_);
    nextCycle = start + transfer;
    statAttentionClusterPFifoPortWaitCycles_->addData(start - now);
    return std::max<uint64_t>(
        1, start - now + attentionScoreFifoLatencyCycles_ + transfer);
}

bool SFU::readAttentionPRowAsync(
    uint32_t slot, const AttentionClusterTag& tag, size_t offset,
    size_t count, uint64_t transferTag, AttentionPReadCallback callback)
{
    if (!attentionClusterEnable_ || !callback || count == 0 ||
        pendingAttentionPReads_.size() >= attentionScoreFifoQueueDepth_ ||
        !attentionPFifo_.canRead(slot, tag, offset, count)) {
        statAttentionClusterPFifoBackpressureStalls_->addData(1);
        return false;
    }
    const uint64_t pendingId = nextAttentionPReadId_++;
    PendingAttentionPRead pending;
    pending.generation = tag.generation;
    pending.slot = slot;
    pending.tag = tag;
    pending.offset = offset;
    pending.count = count;
    pending.transferTag = transferTag;
    pending.callback = std::move(callback);
    pendingAttentionPReads_.emplace(pendingId, std::move(pending));
    const uint64_t bytes = count * sizeof(float);
    const uint64_t delay = schedulePFifoPort(bytes, false);
    rowEngineSelfLink_->send(
        delay, new TensorRowEngineEvent(
            0, 0, 0, TensorRowEngineStage::Normalize,
            TensorRowEngineEventKind::PFifoReadDone, pendingId,
            tag.generation));
    return true;
}

void SFU::completeAttentionPRead(uint64_t pendingId)
{
    auto it = pendingAttentionPReads_.find(pendingId);
    if (it == pendingAttentionPReads_.end()) return;
    PendingAttentionPRead pending = std::move(it->second);
    pendingAttentionPReads_.erase(it);
    std::vector<float> values;
    const bool ok = attentionPFifo_.read(
        pending.slot, pending.tag, pending.offset, pending.count, values);
    if (ok) {
        statAttentionClusterPFifoReads_->addData(1);
        statAttentionClusterPFifoReadBytes_->addData(
            pending.count * sizeof(float));
    }
    pending.callback(ok, pending.transferTag, values);
}

bool SFU::releaseAttentionPSlot(
    uint32_t slot, const AttentionClusterTag& tag)
{
    const bool pending = std::any_of(
        pendingAttentionPReads_.begin(), pendingAttentionPReads_.end(),
        [slot, &tag](const auto& item) {
            return item.second.slot == slot && item.second.tag == tag;
        });
    if (pending || !attentionPFifo_.release(slot, tag)) return false;
    statAttentionClusterPFifoReleases_->addData(1);
    return true;
}

void SFU::cancelAttentionClusterGeneration(uint64_t generation)
{
    for (auto it = pendingAttentionScoreWrites_.begin();
         it != pendingAttentionScoreWrites_.end();) {
        if (it->second.generation == generation) {
            it = pendingAttentionScoreWrites_.erase(it);
        } else {
            ++it;
        }
    }
    attentionScoreFifo_.cancelGeneration(generation);
    for (auto it = pendingAttentionPReads_.begin();
         it != pendingAttentionPReads_.end();) {
        if (it->second.generation == generation) {
            it = pendingAttentionPReads_.erase(it);
        } else {
            ++it;
        }
    }
    statAttentionClusterPFifoCancelled_->addData(
        attentionPFifo_.cancelGeneration(generation));
    for (auto it = tensorWorkerOps_.begin(); it != tensorWorkerOps_.end();) {
        if (it->second.generation == generation) {
            const uint64_t jobId = it->second.attentionJobId;
            const uint32_t rowBegin = it->second.dispatch.row;
            const uint32_t rowEnd = rowBegin + it->second.dispatch.expectedRows;
            for (AttentionOnlineRowContext& online : attentionOnlineContexts_) {
                if (online.valid && online.jobId == jobId &&
                    online.globalRow >= rowBegin && online.globalRow < rowEnd) {
                    online.valid = false;
                }
            }
            if (attentionClusterEnable_ && it->second.localTileMode) {
                if (!attentionClusterSfuActivity_.leave(getCurrentSimCycle())) {
                    output_.fatal(
                        CALL_INFO, -1,
                        "Attention cluster SFU cancellation activity violation\n");
                }
            }
            it = tensorWorkerOps_.erase(it);
        } else {
            ++it;
        }
    }
}

AttentionClusterAdmission SFU::attentionTileAdmission(
    const AttentionTileRequest& request) const
{
    if (globalMem_ == nullptr || request.jobId == 0 || request.rows == 0 ||
        request.rows > 64 ||
        request.cols == 0 || request.headDim == 0 || request.keyTiles == 0 ||
        request.keyTile >= request.keyTiles ||
        (request.keyTiles > 1 && request.rows > attentionOnlineContexts_.size()) ||
        (!request.directScoreMode &&
         request.localScoreAddr < globalMem_->getBaseAddr()) ||
        (request.directScoreMode &&
         (request.generation == 0 || request.scoreSlot >= AttentionScoreFifo::kSlots ||
          request.scoreTag.generation != request.generation)) ||
        (request.directPMode &&
         (!request.directScoreMode || request.pSlot >= AttentionScoreFifo::kSlots ||
          request.pTag.generation != request.generation))) {
        return AttentionClusterAdmission::Invalid;
    }
    const uint64_t rowBytes = static_cast<uint64_t>(request.cols) * sizeof(float);
    const uint64_t tileBytes = rowBytes * request.rows;
    if (!request.directScoreMode &&
        (request.localScoreAddr + tileBytes < request.localScoreAddr ||
         request.localScoreAddr + tileBytes >
            globalMem_->getBaseAddr() + globalMem_->getSize() ||
         tileBytes > rowEngineScratchpadBytes_)) {
        return AttentionClusterAdmission::Invalid;
    }
    if (request.directScoreMode) {
        if (!attentionScoreFifo_.reserved(request.scoreSlot, request.scoreTag)) {
            return AttentionClusterAdmission::Invalid;
        }
        if (!attentionScoreFifo_.ready(request.scoreSlot, request.scoreTag)) {
            return AttentionClusterAdmission::Retry;
        }
    }
    if (request.directPMode &&
        !attentionPFifo_.reserved(request.pSlot, request.pTag)) {
        return AttentionClusterAdmission::Invalid;
    }

    const TensorWorkerKey key(
        request.generation, request.tag, request.globalRowBegin);
    if (tensorWorkerOps_.find(key) != tensorWorkerOps_.end()) {
        return AttentionClusterAdmission::Invalid;
    }
    const uint32_t contextCount = std::min(request.rows, rowEngineContexts_);
    // Validate the complete ownership set before mutating any online row. This
    // keeps a rejected concurrent request from partially advancing Softmax
    // state owned by another generation/query context.
    for (uint32_t index = 0; index < contextCount; ++index) {
        const uint32_t globalRow = request.globalRowBegin + index;
        const AttentionOnlineRowContext& online = attentionOnlineContexts_[
            globalRow % attentionOnlineContexts_.size()];
        if (request.keyTile != 0 &&
            (!online.valid || online.jobId != request.jobId ||
             online.globalRow != globalRow)) {
            return AttentionClusterAdmission::Invalid;
        }
    }
    return tensorWorkerOps_.size() >= rowEngineContexts_
        ? AttentionClusterAdmission::Retry
        : AttentionClusterAdmission::Ready;
}

bool SFU::issueAttentionTile(const AttentionTileRequest& request,
                             std::function<void(bool, const AttentionTileResult&)> callback)
{
    if (!callback) return false;
    const AttentionClusterAdmission admission = attentionTileAdmission(request);
    if (admission != AttentionClusterAdmission::Ready) {
        if (admission == AttentionClusterAdmission::Retry &&
            attentionClusterEnable_) {
            statAttentionClusterSfuBackpressureStalls_->addData(1);
        }
        return false;
    }

    const TensorWorkerKey key(
        request.generation, request.tag, request.globalRowBegin);
    TensorWorkerState worker = {};
    worker.dispatch.kind = ReductionTransportMessageKind::AttentionDispatch;
    worker.dispatch.tag = request.tag;
    worker.dispatch.ownerCore = coreId_;
    worker.dispatch.workerCore = coreId_;
    worker.dispatch.row = request.globalRowBegin;
    worker.dispatch.expectedRows = request.rows;
    worker.dispatch.expectedCols = request.cols;
    worker.dispatch.rowsPerBand = request.rows;
    worker.dispatch.headDim = request.headDim;
    const double scale = 1.0 / std::sqrt(static_cast<double>(request.headDim));
    worker.dispatch.value = request.causal ? -scale : scale;
    worker.scratchAddr = request.localScoreAddr;
    worker.nextRow = request.globalRowBegin;
    worker.rowsCompleted = 0;
    worker.localTileMode = true;
    worker.attentionJobId = request.jobId;
    worker.attentionKeyTile = request.keyTile;
    worker.attentionKeyTiles = request.keyTiles;
    worker.attentionKeyBegin = request.keyBegin;
    worker.attentionFirstTileForJob = request.firstTileForJob;
    worker.directScoreMode = request.directScoreMode;
    worker.generation = request.generation;
    worker.scoreSlot = request.scoreSlot;
    worker.scoreTag = request.scoreTag;
    worker.directPMode = request.directPMode;
    worker.pSlot = request.pSlot;
    worker.pTag = request.pTag;
    worker.attentionResult.rows = request.rows;
    worker.localTileCallback = std::move(callback);
    const uint32_t contextCount = std::min(request.rows, rowEngineContexts_);
    for (uint32_t index = 0; index < contextCount; ++index) {
        const uint32_t globalRow = request.globalRowBegin + index;
        AttentionOnlineRowContext& online = attentionOnlineContexts_[
            globalRow % attentionOnlineContexts_.size()];
        if (request.keyTile == 0) {
            online.valid = true;
            online.jobId = request.jobId;
            online.globalRow = globalRow;
            online.m = -std::numeric_limits<double>::infinity();
            online.l = 0.0;
        }
    }
    worker.contexts.resize(contextCount);
    if (!tensorWorkerOps_.emplace(key, std::move(worker)).second) {
        return false;
    }

    if (attentionClusterEnable_ &&
        !attentionClusterSfuActivity_.enter(getCurrentSimCycle())) {
        output_.fatal(CALL_INFO, -1,
                      "Attention cluster SFU activity order violation\n");
    }

    statAttentionJobs_->addData(1);
    rowEngineSelfLink_->send(
        request.firstTileForJob ? std::max<uint32_t>(1, rowEngineRsqrtLatency_) : 1,
        new TensorRowEngineEvent(request.tag, request.globalRowBegin, 0,
                                 TensorRowEngineStage::Max,
                                 TensorRowEngineEventKind::AttentionStart, 0,
                                 request.generation));
    return true;
}

void SFU::rejectTensorRowDispatch(const ReductionTransportMessage& message)
{
    statRetryEvents_->addData(1);
    if (globalMem_ == nullptr) {
        return;
    }
    ReductionTransportMessage completion = message;
    completion.kind = ReductionTransportMessageKind::TensorRowComplete;
    completion.sendCycle = getCurrentSimCycle();
    completion.value = 0.0;
    if (!globalMem_->sendReductionMessage(completion.ownerCore, completion)) {
        statRetryEvents_->addData(1);
    }
}

uint64_t SFU::rowEngineCurrentCycle() const
{
    return ceilMulDiv(
        getCurrentSimCycle(), rowEngineAcceleratorClockHz_, rowEngineTimebaseTicksPerSecond_);
}

void SFU::issueTensorInputDma(const TensorWorkerKey& key, uint32_t contextIndex)
{
    auto workerIt = tensorWorkerOps_.find(key);
    if (workerIt == tensorWorkerOps_.end() || globalMem_ == nullptr ||
        contextIndex >= workerIt->second.contexts.size()) {
        return;
    }
    TensorWorkerState& worker = workerIt->second;
    TensorWorkerState::Context& context = worker.contexts[contextIndex];
    const uint32_t bandEnd = worker.dispatch.row + worker.dispatch.expectedRows;
    if (context.busy || worker.nextRow >= bandEnd) {
        return;
    }

    context.busy = true;
    context.row = worker.nextRow++;
    context.laneValues.clear();
    context.laneValues.reserve(rowEngineVectorLanes_);
    const uint64_t rowBytes = static_cast<uint64_t>(worker.dispatch.expectedCols) * sizeof(float);
    if (worker.localTileMode) {
        AttentionOnlineRowContext& online = attentionOnlineContexts_[
            context.row % attentionOnlineContexts_.size()];
        if (worker.attentionKeyTile == 0) {
            online.valid = true;
            online.jobId = worker.attentionJobId;
            online.globalRow = context.row;
            online.m = -std::numeric_limits<double>::infinity();
            online.l = 0.0;
        } else if (!online.valid || online.jobId != worker.attentionJobId ||
                   online.globalRow != context.row) {
            finishTensorWorker(key, false);
            return;
        }
        context.scratchAddr = worker.scratchAddr +
            static_cast<uint64_t>(context.row - worker.dispatch.row) * rowBytes;
        beginTensorRowStage(key, contextIndex, TensorRowEngineStage::Max);
        return;
    }
    const uint64_t inputAddr = tensorWorkerHostAddress(worker.dispatch, context.row, false);
    globalMem_->dma_read_from_host_to_globalmem(
        inputAddr,
        rowBytes,
        context.scratchAddr,
        [this, key, contextIndex](bool ok) {
            auto it = tensorWorkerOps_.find(key);
            if (it == tensorWorkerOps_.end() || contextIndex >= it->second.contexts.size()) {
                return;
            }
            if (!ok) {
                finishTensorWorker(key, false);
                return;
            }
            statTensorInputDmaReadyTick_->addData(getCurrentSimCycle());
            beginTensorRowStage(key, contextIndex, TensorRowEngineStage::Max);
        });
}

void SFU::beginTensorRowStage(const TensorWorkerKey& key,
                              uint32_t contextIndex,
                              TensorRowEngineStage stage)
{
    auto workerIt = tensorWorkerOps_.find(key);
    if (workerIt == tensorWorkerOps_.end() || contextIndex >= workerIt->second.contexts.size()) {
        return;
    }
    TensorWorkerState::Context& context = workerIt->second.contexts[contextIndex];
    context.stage = stage;
    context.chunkBegin = 0;
    context.laneValues.clear();
    if (stage == TensorRowEngineStage::Max) {
        context.rowMax = -std::numeric_limits<float>::infinity();
        if (workerIt->second.localTileMode &&
            !workerIt->second.directScoreMode) {
            context.residentValues.assign(
                workerIt->second.dispatch.expectedCols, 0.0f);
        }
    } else if (stage == TensorRowEngineStage::ExpSum) {
        context.rowSum = 0.0;
    } else {
        if (context.rowSum == 0.0 && !workerIt->second.localTileMode) {
            finishTensorWorker(key, false);
            return;
        }
        context.invSum = workerIt->second.localTileMode
            ? context.tileWeightScale
            : static_cast<float>(1.0 / context.rowSum);
    }
    issueTensorLocalRead(key, contextIndex);
}

void SFU::scheduleTensorLocalRetry(const TensorWorkerKey& key,
                                   uint32_t contextIndex,
                                   TensorRowEngineEventKind kind)
{
    auto workerIt = tensorWorkerOps_.find(key);
    if (workerIt == tensorWorkerOps_.end() || rowEngineSelfLink_ == nullptr ||
        contextIndex >= workerIt->second.contexts.size()) {
        return;
    }
    const TensorWorkerState::Context& context = workerIt->second.contexts[contextIndex];
    ++tensorLocalRetryEvents_;
    rowEngineSelfLink_->send(
        1,
        new TensorRowEngineEvent(std::get<1>(key), std::get<2>(key),
                                 contextIndex, context.stage,
                                 kind, context.pendingLocalTag,
                                 workerIt->second.generation));
}

void SFU::issueTensorLocalRead(const TensorWorkerKey& key, uint32_t contextIndex)
{
    auto workerIt = tensorWorkerOps_.find(key);
    if (workerIt == tensorWorkerOps_.end() || globalMem_ == nullptr ||
        contextIndex >= workerIt->second.contexts.size()) {
        return;
    }
    TensorWorkerState& worker = workerIt->second;
    TensorWorkerState::Context& context = worker.contexts[contextIndex];
    const uint32_t remaining = worker.dispatch.expectedCols - context.chunkBegin;
    const uint32_t chunkElems = std::min(rowEngineVectorLanes_, remaining);
    const size_t chunkBytes = static_cast<size_t>(chunkElems) * sizeof(float);
    context.pendingLocalTag = nextTensorLocalTag_++;
    const uint64_t localTag = context.pendingLocalTag;
    const bool rowResident = worker.localTileMode && !worker.directScoreMode;
    if (rowResident && context.stage != TensorRowEngineStage::Max) {
        if (context.residentValues.size() != worker.dispatch.expectedCols) {
            finishTensorWorker(key, false);
            return;
        }
        context.laneValues.assign(
            context.residentValues.begin() + context.chunkBegin,
            context.residentValues.begin() + context.chunkBegin + chunkElems);
        tensorLaneBufferHighWater_ = std::max<uint64_t>(
            tensorLaneBufferHighWater_, context.laneValues.size());
        scheduleTensorRowStage(key, contextIndex, context.stage);
        return;
    }
    if (worker.directScoreMode) {
        const uint32_t rowIndex = context.row - worker.dispatch.row;
        const size_t begin = static_cast<size_t>(rowIndex) *
            worker.dispatch.expectedCols + context.chunkBegin;
        if (!attentionScoreFifo_.canRead(
                worker.scoreSlot, worker.scoreTag, begin, chunkElems)) {
            finishTensorWorker(key, false);
            return;
        }
        const uint64_t delay = scheduleScoreFifoPort(chunkBytes, false);
        statAttentionClusterScoreFifoReads_->addData(1);
        statAttentionClusterScoreFifoReadBytes_->addData(chunkBytes);
        rowEngineSelfLink_->send(
            delay, new TensorRowEngineEvent(
                std::get<1>(key), std::get<2>(key), contextIndex, context.stage,
                TensorRowEngineEventKind::DirectReadDone, localTag,
                worker.generation));
        return;
    }
    const bool accepted = globalMem_->localReadAsync(
        context.scratchAddr + static_cast<uint64_t>(context.chunkBegin) * sizeof(float),
        chunkBytes,
        LocalMemoryClient::SFU,
        localTag,
        [this, key, contextIndex, chunkElems, chunkBytes](
            bool ok, uint64_t tag, const std::vector<uint8_t>& raw) {
            auto it = tensorWorkerOps_.find(key);
            if (it == tensorWorkerOps_.end() || contextIndex >= it->second.contexts.size()) {
                return;
            }
            TensorWorkerState::Context& callbackContext = it->second.contexts[contextIndex];
            if (!ok || tag != callbackContext.pendingLocalTag || raw.size() != chunkBytes) {
                finishTensorWorker(key, false);
                return;
            }
            callbackContext.laneValues.resize(chunkElems);
            std::memcpy(callbackContext.laneValues.data(), raw.data(), chunkBytes);
            if (chunkElems > tensorLaneBufferHighWater_) {
                tensorLaneBufferHighWater_ = chunkElems;
            }
            if (callbackContext.stage == TensorRowEngineStage::Max) {
                tensorMaxLocalReadBytes_ += chunkBytes;
            } else if (callbackContext.stage == TensorRowEngineStage::ExpSum) {
                tensorExpSumLocalReadBytes_ += chunkBytes;
            } else {
                tensorNormalizeLocalReadBytes_ += chunkBytes;
            }
            scheduleTensorRowStage(key, contextIndex, callbackContext.stage);
        });
    if (!accepted) {
        scheduleTensorLocalRetry(key, contextIndex, TensorRowEngineEventKind::LocalReadRetry);
    }
}

void SFU::issueTensorLocalWrite(const TensorWorkerKey& key, uint32_t contextIndex)
{
    auto workerIt = tensorWorkerOps_.find(key);
    if (workerIt == tensorWorkerOps_.end() || globalMem_ == nullptr ||
        contextIndex >= workerIt->second.contexts.size()) {
        return;
    }
    TensorWorkerState& worker = workerIt->second;
    TensorWorkerState::Context& context = worker.contexts[contextIndex];
    std::vector<uint8_t> raw(context.laneValues.size() * sizeof(float));
    std::memcpy(raw.data(), context.laneValues.data(), raw.size());
    context.pendingLocalTag = nextTensorLocalTag_++;
    const uint64_t localTag = context.pendingLocalTag;
    const size_t chunkBytes = raw.size();
    if (worker.directScoreMode && context.stage != TensorRowEngineStage::Normalize) {
        const uint32_t rowIndex = context.row - worker.dispatch.row;
        const size_t begin = static_cast<size_t>(rowIndex) *
            worker.dispatch.expectedCols + context.chunkBegin;
        if (!attentionScoreFifo_.canRead(
                worker.scoreSlot, worker.scoreTag, begin,
                context.laneValues.size())) {
            finishTensorWorker(key, false);
            return;
        }
        const uint64_t delay = scheduleScoreFifoPort(chunkBytes, true);
        statAttentionClusterScoreFifoInternalWrites_->addData(1);
        statAttentionClusterScoreFifoInternalWriteBytes_->addData(chunkBytes);
        rowEngineSelfLink_->send(
            delay, new TensorRowEngineEvent(
                std::get<1>(key), std::get<2>(key), contextIndex, context.stage,
                TensorRowEngineEventKind::DirectWriteDone, localTag,
                worker.generation));
        return;
    }
    if (worker.directPMode && context.stage == TensorRowEngineStage::Normalize) {
        const uint32_t rowIndex = context.row - worker.dispatch.row;
        const size_t begin = static_cast<size_t>(rowIndex) *
            worker.dispatch.expectedCols + context.chunkBegin;
        if (!attentionPFifo_.reserveWrite(
                worker.pSlot, worker.pTag, begin, context.laneValues.size())) {
            finishTensorWorker(key, false);
            return;
        }
        const uint64_t delay = schedulePFifoPort(chunkBytes, true);
        statAttentionClusterPFifoWrites_->addData(1);
        statAttentionClusterPFifoWriteBytes_->addData(chunkBytes);
        rowEngineSelfLink_->send(
            delay, new TensorRowEngineEvent(
                std::get<1>(key), std::get<2>(key), contextIndex, context.stage,
                TensorRowEngineEventKind::DirectWriteDone, localTag,
                worker.generation));
        return;
    }
    const bool accepted = globalMem_->localWriteAsync(
        context.scratchAddr + static_cast<uint64_t>(context.chunkBegin) * sizeof(float),
        raw,
        LocalMemoryClient::SFU,
        localTag,
        [this, key, contextIndex, chunkBytes](bool ok, uint64_t tag) {
            auto it = tensorWorkerOps_.find(key);
            if (it == tensorWorkerOps_.end() || contextIndex >= it->second.contexts.size()) {
                return;
            }
            TensorWorkerState::Context& callbackContext = it->second.contexts[contextIndex];
            if (!ok || tag != callbackContext.pendingLocalTag) {
                finishTensorWorker(key, false);
                return;
            }
            if (callbackContext.stage == TensorRowEngineStage::Max) {
                tensorMaxLocalWriteBytes_ += chunkBytes;
            } else if (callbackContext.stage == TensorRowEngineStage::ExpSum) {
                tensorExpSumLocalWriteBytes_ += chunkBytes;
            } else {
                tensorNormalizeLocalWriteBytes_ += chunkBytes;
            }
            advanceTensorRowChunk(key, contextIndex);
        });
    if (!accepted) {
        scheduleTensorLocalRetry(key, contextIndex, TensorRowEngineEventKind::LocalWriteRetry);
    }
}

void SFU::scheduleTensorRowStage(const TensorWorkerKey& key,
                                 uint32_t contextIndex,
                                 TensorRowEngineStage stage)
{
    auto workerIt = tensorWorkerOps_.find(key);
    if (workerIt == tensorWorkerOps_.end() || rowEngineSelfLink_ == nullptr ||
        contextIndex >= workerIt->second.contexts.size() ||
        !workerIt->second.contexts[contextIndex].busy) {
        finishTensorWorker(key, false);
        return;
    }

    const uint64_t now = rowEngineCurrentCycle();
    TensorWorkerState::Context& context = workerIt->second.contexts[contextIndex];
    if (context.stage != stage || context.laneValues.empty()) {
        finishTensorWorker(key, false);
        return;
    }
    const uint64_t elems = context.laneValues.size();
    const bool finalChunk = context.chunkBegin + elems == workerIt->second.dispatch.expectedCols;
    uint64_t start = now;
    uint64_t duration = 0;
    if (stage == TensorRowEngineStage::Max) {
        const uint64_t active = ceilDiv(elems, rowEngineVectorLanes_);
        start = std::max(now, rowEngineVectorFreeCycle_);
        rowEngineVectorFreeCycle_ = start + active;
        duration = active + (finalChunk ? rowEngineReductionTreeLatency_ : 0);
    } else if (stage == TensorRowEngineStage::ExpSum) {
        const uint64_t active = ceilDiv(elems, rowEngineExpLanes_);
        start = std::max(now, rowEngineExpFreeCycle_);
        rowEngineExpFreeCycle_ = start + active;
        duration = active + (finalChunk ? rowEngineExpLatency_ + rowEngineReductionTreeLatency_ : 0);
    } else {
        const uint64_t active = ceilDiv(elems, rowEngineVectorLanes_);
        start = std::max(now, rowEngineVectorFreeCycle_);
        duration = active + (context.chunkBegin == 0 ? rowEngineReciprocalLatency_ : 0);
        rowEngineVectorFreeCycle_ = start + duration;
    }
    const uint64_t delay = std::max<uint64_t>(1, start - now + duration);
    const uint64_t startTick = ceilMulDiv(
        start, rowEngineTimebaseTicksPerSecond_, rowEngineAcceleratorClockHz_);
    if (context.chunkBegin == 0 && stage == TensorRowEngineStage::Max) {
        statTensorMaxStartTick_->addData(startTick);
    } else if (context.chunkBegin == 0 && stage == TensorRowEngineStage::ExpSum) {
        statTensorExpSumStartTick_->addData(startTick);
    } else if (context.chunkBegin == 0) {
        statTensorNormalizeStartTick_->addData(startTick);
    }
    rowEngineSelfLink_->send(
        delay,
        new TensorRowEngineEvent(std::get<1>(key), std::get<2>(key),
                                 contextIndex, stage,
                                 TensorRowEngineEventKind::ResourceDone,
                                 context.pendingLocalTag,
                                 workerIt->second.generation));
}

void SFU::advanceTensorRowChunk(const TensorWorkerKey& key, uint32_t contextIndex)
{
    auto workerIt = tensorWorkerOps_.find(key);
    if (workerIt == tensorWorkerOps_.end() || contextIndex >= workerIt->second.contexts.size()) {
        return;
    }
    TensorWorkerState::Context& context = workerIt->second.contexts[contextIndex];
    context.chunkBegin += context.laneValues.size();
    context.laneValues.clear();
    if (context.chunkBegin < workerIt->second.dispatch.expectedCols) {
        issueTensorLocalRead(key, contextIndex);
        return;
    }
    if (context.stage == TensorRowEngineStage::Max) {
        statTensorMaxDoneTick_->addData(getCurrentSimCycle());
        if (workerIt->second.dispatch.value != 0.0) {
            statAttentionScaleMaskDoneTick_->addData(getCurrentSimCycle());
        }
        beginTensorRowStage(key, contextIndex, TensorRowEngineStage::ExpSum);
    } else if (context.stage == TensorRowEngineStage::ExpSum) {
        statTensorExpSumDoneTick_->addData(getCurrentSimCycle());
        TensorWorkerState& worker = workerIt->second;
        if (worker.localTileMode) {
            if (contextIndex >= attentionOnlineContexts_.size()) {
                finishTensorWorker(key, false);
                return;
            }
            AttentionOnlineRowContext& online = attentionOnlineContexts_[
                context.row % attentionOnlineContexts_.size()];
            if (!online.valid || online.jobId != worker.attentionJobId ||
                online.globalRow != context.row) {
                finishTensorWorker(key, false);
                return;
            }
            const double mNew = std::max(online.m, static_cast<double>(context.rowMax));
            const double alpha = std::isinf(online.m) && online.m < 0.0
                ? 0.0 : std::exp(online.m - mNew);
            const double beta = std::exp(context.rowMax - mNew);
            const double lNew = online.l * alpha + context.rowSum * beta;
            if (!(lNew > 0.0) || !std::isfinite(lNew)) {
                finishTensorWorker(key, false);
                return;
            }
            const bool finalKeyTile = worker.attentionKeyTile + 1 == worker.attentionKeyTiles;
            const double normalization = finalKeyTile ? 1.0 / lNew : 1.0;
            context.oldOutputScale = static_cast<float>(alpha * normalization);
            context.tileWeightScale = static_cast<float>(beta * normalization);
            online.m = mNew;
            online.l = lNew;
        }
        beginTensorRowStage(key, contextIndex, TensorRowEngineStage::Normalize);
    } else {
        statTensorNormalizeDoneTick_->addData(getCurrentSimCycle());
        statTensorComputeDoneTick_->addData(getCurrentSimCycle());
        completeTensorRow(key, contextIndex);
    }
}

void SFU::completeTensorRow(const TensorWorkerKey& key, uint32_t contextIndex)
{
    auto workerIt = tensorWorkerOps_.find(key);
    if (workerIt == tensorWorkerOps_.end() || contextIndex >= workerIt->second.contexts.size()) {
        return;
    }
    TensorWorkerState& worker = workerIt->second;
    TensorWorkerState::Context& mutableContext = worker.contexts[contextIndex];
    if (worker.localTileMode) {
        const uint32_t resultIndex = mutableContext.row - worker.dispatch.row;
        if (resultIndex >= worker.attentionResult.oldOutputScale.size()) {
            finishTensorWorker(key, false);
            return;
        }
        worker.attentionResult.oldOutputScale[resultIndex] = mutableContext.oldOutputScale;
        mutableContext.busy = false;
        mutableContext.laneValues.clear();
        mutableContext.residentValues.clear();
        worker.rowsCompleted += 1;
        statSoftmaxRows_->addData(1);
        if (worker.rowsCompleted == worker.dispatch.expectedRows) {
            finishTensorWorker(key, true);
        } else {
            issueTensorInputDma(key, contextIndex);
        }
        return;
    }
    const TensorWorkerState::Context& context = mutableContext;
    const size_t rowBytes = static_cast<size_t>(workerIt->second.dispatch.expectedCols) * sizeof(float);
    const uint64_t outputAddr = tensorWorkerHostAddress(
        workerIt->second.dispatch, context.row, true);
    globalMem_->dma_write_from_globalmem_to_host(
        context.scratchAddr,
        outputAddr,
        rowBytes,
        [this, key, contextIndex](bool ok) {
            auto it = tensorWorkerOps_.find(key);
            if (it == tensorWorkerOps_.end() || contextIndex >= it->second.contexts.size()) {
                return;
            }
            if (!ok) {
                finishTensorWorker(key, false);
                return;
            }
            statTensorOutputDmaAckTick_->addData(getCurrentSimCycle());
            statSoftmaxRows_->addData(1);
            TensorWorkerState& worker = it->second;
            TensorWorkerState::Context& callbackContext = worker.contexts[contextIndex];
            callbackContext.busy = false;
            callbackContext.laneValues.clear();
            worker.rowsCompleted += 1;
            if (worker.rowsCompleted == worker.dispatch.expectedRows) {
                finishTensorWorker(key, true);
                return;
            }
            issueTensorInputDma(key, contextIndex);
        });
}

void SFU::handleTensorRowEngineEvent(SST::Event* event)
{
    auto* rowEvent = static_cast<TensorRowEngineEvent*>(event);
    const TensorWorkerKey key(
        rowEvent->generation(), rowEvent->tag(), rowEvent->bandRow());
    const uint32_t contextIndex = rowEvent->context();
    const TensorRowEngineStage stage = rowEvent->stage();
    const TensorRowEngineEventKind kind = rowEvent->kind();
    const uint64_t localTag = rowEvent->localTag();
    const uint64_t generation = rowEvent->generation();
    if (kind == TensorRowEngineEventKind::ScoreFifoWriteDone) {
        delete rowEvent;
        completeAttentionScoreWrite(localTag);
        return;
    }
    if (kind == TensorRowEngineEventKind::PFifoReadDone) {
        delete rowEvent;
        completeAttentionPRead(localTag);
        return;
    }
    delete rowEvent;

    auto workerIt = tensorWorkerOps_.find(key);
    if (workerIt == tensorWorkerOps_.end() ||
        contextIndex >= workerIt->second.contexts.size() ||
        workerIt->second.generation != generation) {
        return;
    }
    TensorWorkerState& worker = workerIt->second;
    if (kind == TensorRowEngineEventKind::AttentionStart) {
        if (worker.attentionFirstTileForJob) {
            statAttentionRsqrtReadyTick_->addData(getCurrentSimCycle());
        }
        for (uint32_t index = 0; index < worker.contexts.size(); ++index) {
            issueTensorInputDma(key, index);
        }
        return;
    }
    TensorWorkerState::Context& context = worker.contexts[contextIndex];
    if (!context.busy || context.stage != stage || context.pendingLocalTag != localTag) {
        return;
    }
    if (kind == TensorRowEngineEventKind::LocalReadRetry) {
        issueTensorLocalRead(key, contextIndex);
        return;
    }
    if (kind == TensorRowEngineEventKind::LocalWriteRetry) {
        issueTensorLocalWrite(key, contextIndex);
        return;
    }
    if (kind == TensorRowEngineEventKind::DirectReadDone) {
        const uint32_t rowIndex = context.row - worker.dispatch.row;
        const size_t begin = static_cast<size_t>(rowIndex) *
            worker.dispatch.expectedCols + context.chunkBegin;
        if (!attentionScoreFifo_.read(
                worker.scoreSlot, worker.scoreTag, begin,
                std::min(rowEngineVectorLanes_,
                    worker.dispatch.expectedCols - context.chunkBegin),
                context.laneValues)) {
            finishTensorWorker(key, false);
            return;
        }
        tensorLaneBufferHighWater_ = std::max<uint64_t>(
            tensorLaneBufferHighWater_, context.laneValues.size());
        scheduleTensorRowStage(key, contextIndex, stage);
        return;
    }
    if (kind == TensorRowEngineEventKind::DirectWriteDone) {
        const uint32_t rowIndex = context.row - worker.dispatch.row;
        const size_t begin = static_cast<size_t>(rowIndex) *
            worker.dispatch.expectedCols + context.chunkBegin;
        const bool ok = context.stage == TensorRowEngineStage::Normalize &&
            worker.directPMode
            ? attentionPFifo_.commitWrite(
                worker.pSlot, worker.pTag, begin, context.laneValues)
            : attentionScoreFifo_.overwrite(
                worker.scoreSlot, worker.scoreTag, begin, context.laneValues);
        if (!ok) {
            finishTensorWorker(key, false);
            return;
        }
        advanceTensorRowChunk(key, contextIndex);
        return;
    }
    if (context.laneValues.empty()) {
        finishTensorWorker(key, false);
        return;
    }

    if (stage == TensorRowEngineStage::Max) {
        if (worker.dispatch.value != 0.0) {
            if (context.chunkBegin == 0) {
                statAttentionScaleMaskStartTick_->addData(getCurrentSimCycle());
            }
            const bool causal = worker.dispatch.value < 0.0;
            const float scale = static_cast<float>(std::abs(worker.dispatch.value));
            uint64_t masked = 0;
            for (uint32_t lane = 0; lane < context.laneValues.size(); ++lane) {
                const uint32_t col = context.chunkBegin + lane;
                context.laneValues[lane] *= scale;
                if (causal && worker.attentionKeyBegin + col > context.row) {
                    context.laneValues[lane] = -std::numeric_limits<float>::infinity();
                    ++masked;
                }
            }
            statAttentionScaledElements_->addData(context.laneValues.size());
            statAttentionMaskedElements_->addData(masked);
        }
        context.rowMax = std::max(
            context.rowMax,
            *std::max_element(context.laneValues.begin(), context.laneValues.end()));
        const bool rowResident = worker.localTileMode && !worker.directScoreMode;
        if (rowResident) {
            std::copy(context.laneValues.begin(), context.laneValues.end(),
                      context.residentValues.begin() + context.chunkBegin);
        }
        if (worker.dispatch.value != 0.0 && !rowResident) {
            issueTensorLocalWrite(key, contextIndex);
        } else {
            advanceTensorRowChunk(key, contextIndex);
        }
        return;
    }
    if (stage == TensorRowEngineStage::ExpSum) {
        for (float& value : context.laneValues) {
            value = std::exp(value - context.rowMax);
            context.rowSum += value;
        }
        if (worker.localTileMode && !worker.directScoreMode) {
            std::copy(context.laneValues.begin(), context.laneValues.end(),
                      context.residentValues.begin() + context.chunkBegin);
            advanceTensorRowChunk(key, contextIndex);
        } else {
            issueTensorLocalWrite(key, contextIndex);
        }
        return;
    }
    for (float& value : context.laneValues) {
        value *= context.invSum;
    }
    issueTensorLocalWrite(key, contextIndex);
}

void SFU::finishTensorWorker(const TensorWorkerKey& key, bool ok)
{
    auto workerIt = tensorWorkerOps_.find(key);
    if (workerIt == tensorWorkerOps_.end()) {
        return;
    }
    ReductionTransportMessage completion = workerIt->second.dispatch;
    const bool localTileMode = workerIt->second.localTileMode;
    const bool finalAttentionTile = localTileMode &&
        workerIt->second.attentionKeyTile + 1 == workerIt->second.attentionKeyTiles;
    const uint64_t attentionJobId = workerIt->second.attentionJobId;
    const uint32_t attentionRowBegin = workerIt->second.dispatch.row;
    const uint32_t attentionRows = workerIt->second.dispatch.expectedRows;
    const AttentionTileResult attentionResult = workerIt->second.attentionResult;
    std::function<void(bool, const AttentionTileResult&)> callback =
        std::move(workerIt->second.localTileCallback);
    if (!ok && localTileMode) {
        uint32_t ownedContexts = 0;
        for (const AttentionOnlineRowContext& online : attentionOnlineContexts_) {
            if (online.valid && online.jobId == attentionJobId &&
                online.globalRow >= attentionRowBegin &&
                online.globalRow - attentionRowBegin < attentionRows) {
                ++ownedContexts;
            }
        }
        output_.output(
            "Attention SFU failure core=%" PRIu32 " key_tile=%" PRIu32
            " rows_completed=%" PRIu32 " owned_contexts=%" PRIu32 "\n",
            coreId_, workerIt->second.attentionKeyTile,
            workerIt->second.rowsCompleted, ownedContexts);
    }
    if (attentionClusterEnable_ && localTileMode &&
        !attentionClusterSfuActivity_.leave(getCurrentSimCycle())) {
        output_.fatal(CALL_INFO, -1,
                      "Attention cluster SFU activity underflow/order violation\n");
    }
    tensorWorkerOps_.erase(workerIt);
    if (localTileMode) {
        if (finalAttentionTile) {
            for (AttentionOnlineRowContext& online : attentionOnlineContexts_) {
                if (online.valid && online.jobId == attentionJobId &&
                    online.globalRow >= attentionRowBegin &&
                    online.globalRow - attentionRowBegin < attentionRows) {
                    online.valid = false;
                }
            }
        }
        callback(ok, attentionResult);
        if (!ok) {
            statRetryEvents_->addData(1);
        }
        return;
    }
    completion.kind = ReductionTransportMessageKind::TensorRowComplete;
    completion.sendCycle = getCurrentSimCycle();
    completion.value = ok ? 1.0 : 0.0;
    if (!globalMem_->sendReductionMessage(completion.ownerCore, completion)) {
        statRetryEvents_->addData(1);
    }
    if (!ok) {
        statRetryEvents_->addData(1);
    }
}

void SFU::handleTensorRowComplete(const ReductionTransportMessage& message)
{
    const auto staleDrop = [this]() {
        statReductionTransportStaleDropped_->addData(1);
        statRetryEvents_->addData(1);
    };
    auto it = pendingJobOps_.find(message.tag);
    if (it == pendingJobOps_.end() || message.ownerCore != coreId_ ||
        (message.value != 0.0 && message.value != 1.0) ||
        (it->second.desc.flags & SFU_JOB_FLAG_TENSOR_ROW_ENGINE) == 0) {
        staleDrop();
        return;
    }
    JobOpState& state = it->second;
    const uint32_t rowsPerBand = state.tensorParams.rows_per_band;
    if (state.status != SFUStatus::Pending || rowsPerBand == 0 ||
        message.jobId != state.desc.job_id || message.row >= state.desc.rows ||
        message.row % rowsPerBand != 0 || message.rowsPerBand != rowsPerBand ||
        message.expectedWorkers != state.desc.worker_cores ||
        message.expectedCols != state.desc.cols) {
        staleDrop();
        return;
    }
    const uint32_t band = message.row / rowsPerBand;
    const uint32_t expectedRows = std::min(rowsPerBand, state.desc.rows - message.row);
    if (band >= state.tensorCompletionSeen.size() ||
        message.workerSlot != band % state.desc.worker_cores ||
        message.expectedRows != expectedRows || state.tensorCompletionSeen[band] != 0) {
        staleDrop();
        return;
    }
    state.tensorCompletionSeen[band] = 1;
    statTensorCompletionReceivedTick_->addData(getCurrentSimCycle());
    if (message.value == 0.0) {
        state.status = SFUStatus::InvalidDescriptor;
        return;
    }
    state.tensorRowsCompleted += message.expectedRows;
    if (state.tensorRowsCompleted > state.desc.rows) {
        state.status = SFUStatus::InvalidDescriptor;
        return;
    }
    if (state.tensorRowsCompleted == state.desc.rows &&
        std::all_of(state.tensorCompletionSeen.begin(), state.tensorCompletionSeen.end(),
                    [](uint8_t seen) { return seen != 0; })) {
        state.tensorDmaComplete = true;
        finishTensorJobIfReady(&state);
    }
}

bool SFU::startTensorRowEngineJob(uint64_t tag)
{
    auto it = pendingJobOps_.find(tag);
    if (it == pendingJobOps_.end() || globalMem_ == nullptr) {
        return false;
    }
    JobOpState& state = it->second;
    const uint32_t rowsPerBand = state.tensorParams.rows_per_band;
    if (rowsPerBand == 0 || state.desc.worker_cores == 0 ||
        !explicitDistributedReductionEnabled() ||
        !globalMem_->reductionNetworkAvailable()) {
        return false;
    }
    const uint32_t bands = static_cast<uint32_t>(ceilDiv(state.desc.rows, rowsPerBand));
    const uint32_t contextCount = std::min(rowsPerBand, rowEngineContexts_);
    const uint64_t rowBytes = static_cast<uint64_t>(state.desc.cols) * sizeof(float);
    const uint64_t scratchAddr = globalMem_->getBaseAddr() + 0x2000;
    if (bands > state.desc.worker_cores ||
        rowBytes * contextCount > rowEngineScratchpadBytes_ ||
        scratchAddr + rowBytes * contextCount >
            globalMem_->getBaseAddr() + globalMem_->getSize()) {
        return false;
    }
    state.tensorRowsCompleted = 0;
    state.tensorDmaComplete = false;
    state.tensorStarted = true;
    state.tensorCompletionSeen.assign(bands, 0);
    for (uint32_t band = 0; band < bands; ++band) {
        ReductionTransportMessage dispatch = {};
        dispatch.kind = ReductionTransportMessageKind::TensorRowDispatch;
        dispatch.jobId = state.desc.job_id;
        dispatch.tag = tag;
        dispatch.ownerCore = coreId_;
        dispatch.workerSlot = band % state.desc.worker_cores;
        dispatch.workerCore = dispatch.workerSlot;
        dispatch.row = band * rowsPerBand;
        dispatch.expectedWorkers = state.desc.worker_cores;
        dispatch.expectedRows = std::min(rowsPerBand, state.desc.rows - dispatch.row);
        dispatch.expectedCols = state.desc.cols;
        dispatch.sendCycle = getCurrentSimCycle();
        dispatch.inputAddr = state.desc.input0_addr;
        dispatch.outputAddr = state.desc.output_addr;
        dispatch.nodeStrideBytes = state.tensorParams.node_stride_bytes;
        dispatch.dataNodeMask = state.tensorParams.data_node_mask;
        dispatch.rowsPerBand = rowsPerBand;
        dispatch.value = state.attentionMode
            ? (state.attentionCausal ? -state.attentionScale : state.attentionScale)
            : 0.0;
        statTensorBandDispatchTick_->addData(getCurrentSimCycle());
        if (!globalMem_->sendReductionMessage(dispatch.workerSlot, dispatch)) {
            return false;
        }
    }
    return true;
}

bool SFU::executeDistributedSoftmaxRowJob(JobOpState* state)
{
    if (state == nullptr || globalMem_ == nullptr) {
        return false;
    }
    if (explicitDistributedReductionEnabled() && !globalMem_->reductionNetworkAvailable()) {
        return false;
    }

    const SFUJobDesc& desc = state->desc;
    state->workerSlot = desc.reserved0;
    state->colBegin = workerColumnBegin(desc.cols, state->workerSlot, desc.worker_cores);
    state->colEnd = workerColumnEnd(desc.cols, state->workerSlot, desc.worker_cores);
    const uint32_t localCols = state->colEnd - state->colBegin;
    if (localCols == 0) {
        return false;
    }

    state->localMax.assign(desc.rows, -std::numeric_limits<double>::infinity());
    state->maxResponseSeen.assign(desc.rows, 0);
    state->sumResponseSeen.assign(desc.rows, 0);
    for (uint32_t col = state->colBegin; col < state->colEnd; col += desc.chunk_elems) {
        const uint32_t chunkEnd = std::min(state->colEnd, col + desc.chunk_elems);
        for (uint32_t row = 0; row < desc.rows; ++row) {
            std::vector<float> chunk;
            if (!readDistributedSoftmaxJobChunk(globalMem_,
                                                 desc.input0_addr,
                                                 desc.rows,
                                                 localCols,
                                                 row,
                                                 col - state->colBegin,
                                                 chunkEnd - state->colBegin,
                                                 &chunk)) {
                return false;
            }
            statJobSoftmaxMaxChunks_->addData(1);
            for (float value : chunk) {
                state->localMax[row] =
                    std::max(state->localMax[row], static_cast<double>(value));
            }
        }
    }

    for (uint32_t row = 0; row < desc.rows; ++row) {
        DistributedReducerResult result = DistributedReducerResult::Invalid;
        if (explicitDistributedReductionEnabled()) {
            ReductionTransportMessage message;
            message.kind = ReductionTransportMessageKind::MaxRequest;
            message.jobId = desc.job_id;
            message.tag = state->tag;
            message.ownerCore = desc.owner_core;
            message.workerSlot = state->workerSlot;
            message.row = row;
            message.expectedWorkers = desc.worker_cores;
            message.expectedRows = desc.rows;
            message.expectedCols = desc.cols;
            message.value = state->localMax[row];
            if (!globalMem_->sendReductionMessage(desc.owner_core, message)) {
                abortDistributedSoftmaxJob(state);
                return false;
            }
            recordDistributedReductionRequest(true);
            result = DistributedReducerResult::Accepted;
        } else {
            result = submitDistributedSoftmaxMax(desc.job_id,
                                                 state->tag,
                                                 desc.owner_core,
                                                 row,
                                                 state->workerSlot,
                                                 desc.worker_cores,
                                                 desc.rows,
                                                 desc.cols,
                                                 state->localMax[row]);
        }
        if (result == DistributedReducerResult::Aborted) {
            observeDistributedSoftmaxJobAbort(state);
            return false;
        }
        if (result != DistributedReducerResult::Accepted) {
            return false;
        }
        if (!explicitDistributedReductionEnabled()) {
            recordDistributedReductionRequest(true);
        }
    }
    statPartialSubmits_->addData(desc.rows);
    state->processedElems = static_cast<uint64_t>(desc.rows) * localCols;
    state->stage = SoftmaxJobStage::MaxSubmitted;
    state->status = SFUStatus::Pending;
    return true;
}

bool SFU::advanceDistributedSoftmaxJob(JobOpState* state)
{
    if (state == nullptr || globalMem_ == nullptr) {
        return false;
    }

    const SFUJobDesc& desc = state->desc;
    const uint32_t localCols = state->colEnd - state->colBegin;
    if (state->stage == SoftmaxJobStage::MaxSubmitted) {
        std::vector<double> globalMax(desc.rows, 0.0);
        if (explicitDistributedReductionEnabled()) {
            std::vector<DistributedReductionResponseInboxKey> responseKeys;
            responseKeys.reserve(desc.rows);
            for (uint32_t row = 0; row < desc.rows; ++row) {
                if (distributedSoftmaxRowAborted(distributedSoftmaxReducerKey(
                        desc.job_id, state->tag, desc.owner_core, row))) {
                    observeDistributedSoftmaxJobAbort(state);
                    return false;
                }
                const DistributedReductionResponseInboxKey key(
                    desc.job_id,
                    state->tag,
                    desc.owner_core,
                    row,
                    state->workerSlot,
                    ReductionTransportMessageKind::MaxResponse);
                if (distributedReductionResponseInbox_.find(key) ==
                    distributedReductionResponseInbox_.end()) {
                    return true;
                }
                responseKeys.push_back(key);
            }
            for (uint32_t row = 0; row < desc.rows; ++row) {
                auto response = distributedReductionResponseInbox_.find(responseKeys[row]);
                globalMax[row] = response->second.value;
                distributedReductionResponseInbox_.erase(response);
                state->maxResponseSeen[row] = 1;
            }
        } else {
            for (uint32_t row = 0; row < desc.rows; ++row) {
                double reducedMax = 0.0;
                const auto result = distributedSoftmaxMaxReady(desc.job_id,
                                                               state->tag,
                                                               desc.owner_core,
                                                               row,
                                                               state->workerSlot,
                                                               &reducedMax);
                if (result == DistributedReducerResult::Pending) {
                    return true;
                }
                if (result == DistributedReducerResult::Aborted) {
                    observeDistributedSoftmaxJobAbort(state);
                    return false;
                }
                if (result != DistributedReducerResult::Ready) {
                    return false;
                }
                globalMax[row] = reducedMax;
                if (row < state->maxResponseSeen.size() && state->maxResponseSeen[row] == 0) {
                    recordDistributedReductionResponse(true);
                    state->maxResponseSeen[row] = 1;
                }
            }
        }

        state->localSum.assign(desc.rows, 0.0);
        for (uint32_t col = state->colBegin; col < state->colEnd; col += desc.chunk_elems) {
            const uint32_t chunkEnd = std::min(state->colEnd, col + desc.chunk_elems);
            for (uint32_t row = 0; row < desc.rows; ++row) {
                std::vector<float> chunk;
                if (!readDistributedSoftmaxJobChunk(globalMem_,
                                                     desc.input0_addr,
                                                     desc.rows,
                                                     localCols,
                                                     row,
                                                     col - state->colBegin,
                                                     chunkEnd - state->colBegin,
                                                     &chunk)) {
                    return false;
                }
                for (float& value : chunk) {
                    const double expValue =
                        std::exp(static_cast<double>(value) - globalMax[row]);
                    value = static_cast<float>(expValue);
                    state->localSum[row] += expValue;
                }
                if (!writeDistributedSoftmaxJobChunk(globalMem_,
                                                      desc.output_addr,
                                                      desc.rows,
                                                      localCols,
                                                      row,
                                                      col - state->colBegin,
                                                      chunk)) {
                    return false;
                }
                statJobSoftmaxSumChunks_->addData(1);
            }
        }
        for (uint32_t row = 0; row < desc.rows; ++row) {
            DistributedReducerResult result = DistributedReducerResult::Invalid;
            if (explicitDistributedReductionEnabled()) {
                ReductionTransportMessage message;
                message.kind = ReductionTransportMessageKind::SumRequest;
                message.jobId = desc.job_id;
                message.tag = state->tag;
                message.ownerCore = desc.owner_core;
                message.workerSlot = state->workerSlot;
                message.row = row;
                message.expectedWorkers = desc.worker_cores;
                message.expectedRows = desc.rows;
                message.expectedCols = desc.cols;
                message.value = state->localSum[row];
                if (!globalMem_->sendReductionMessage(desc.owner_core, message)) {
                    abortDistributedSoftmaxJob(state);
                    return false;
                }
                recordDistributedReductionRequest(false);
                result = DistributedReducerResult::Accepted;
            } else {
                result = submitDistributedSoftmaxSum(desc.job_id,
                                                     state->tag,
                                                     desc.owner_core,
                                                     row,
                                                     state->workerSlot,
                                                     desc.worker_cores,
                                                     desc.rows,
                                                     desc.cols,
                                                     state->localSum[row]);
            }
            if (result == DistributedReducerResult::Aborted) {
                observeDistributedSoftmaxJobAbort(state);
                return false;
            }
            if (result != DistributedReducerResult::Accepted) {
                return false;
            }
            if (!explicitDistributedReductionEnabled()) {
                recordDistributedReductionRequest(false);
            }
        }
        statPartialSubmits_->addData(desc.rows);
        state->stage = SoftmaxJobStage::SumSubmitted;
        return true;
    }

    if (state->stage == SoftmaxJobStage::SumSubmitted) {
        std::vector<double> globalSum(desc.rows, 0.0);
        if (explicitDistributedReductionEnabled()) {
            std::vector<DistributedReductionResponseInboxKey> responseKeys;
            responseKeys.reserve(desc.rows);
            for (uint32_t row = 0; row < desc.rows; ++row) {
                if (distributedSoftmaxRowAborted(distributedSoftmaxReducerKey(
                        desc.job_id, state->tag, desc.owner_core, row))) {
                    observeDistributedSoftmaxJobAbort(state);
                    return false;
                }
                const DistributedReductionResponseInboxKey key(
                    desc.job_id,
                    state->tag,
                    desc.owner_core,
                    row,
                    state->workerSlot,
                    ReductionTransportMessageKind::SumResponse);
                if (distributedReductionResponseInbox_.find(key) ==
                    distributedReductionResponseInbox_.end()) {
                    return true;
                }
                responseKeys.push_back(key);
            }
            for (uint32_t row = 0; row < desc.rows; ++row) {
                auto response = distributedReductionResponseInbox_.find(responseKeys[row]);
                globalSum[row] = response->second.value;
                distributedReductionResponseInbox_.erase(response);
                state->sumResponseSeen[row] = 1;
                if (globalSum[row] == 0.0) {
                    return false;
                }
            }
        } else {
            for (uint32_t row = 0; row < desc.rows; ++row) {
                double reducedSum = 0.0;
                const auto result = distributedSoftmaxSumReady(desc.job_id,
                                                               state->tag,
                                                               desc.owner_core,
                                                               row,
                                                               state->workerSlot,
                                                               &reducedSum);
                if (result == DistributedReducerResult::Pending) {
                    return true;
                }
                if (result == DistributedReducerResult::Aborted) {
                    observeDistributedSoftmaxJobAbort(state);
                    return false;
                }
                if (result != DistributedReducerResult::Ready) {
                    return false;
                }
                globalSum[row] = reducedSum;
                if (row < state->sumResponseSeen.size() && state->sumResponseSeen[row] == 0) {
                    recordDistributedReductionResponse(false);
                    state->sumResponseSeen[row] = 1;
                }
                if (globalSum[row] == 0.0) {
                    return false;
                }
            }
        }

        for (uint32_t col = state->colBegin; col < state->colEnd; col += desc.chunk_elems) {
            const uint32_t chunkEnd = std::min(state->colEnd, col + desc.chunk_elems);
            for (uint32_t row = 0; row < desc.rows; ++row) {
                std::vector<float> chunk;
                if (!readDistributedSoftmaxJobChunk(globalMem_,
                                                     desc.output_addr,
                                                     desc.rows,
                                                     localCols,
                                                     row,
                                                     col - state->colBegin,
                                                     chunkEnd - state->colBegin,
                                                     &chunk)) {
                    return false;
                }
                const float invSum = static_cast<float>(1.0 / globalSum[row]);
                for (float& value : chunk) {
                    value *= invSum;
                }
                if (!writeDistributedSoftmaxJobChunk(globalMem_,
                                                      desc.output_addr,
                                                      desc.rows,
                                                      localCols,
                                                      row,
                                                      col - state->colBegin,
                                                      chunk)) {
                    return false;
                }
                statJobSoftmaxNormChunks_->addData(1);
            }
        }
        for (uint32_t row = 0; row < desc.rows; ++row) {
            const auto result = markDistributedSoftmaxNormalized(desc.job_id,
                                                                 state->tag,
                                                                 desc.owner_core,
                                                                 row,
                                                                 state->workerSlot,
                                                                 desc.worker_cores,
                                                                 desc.rows,
                                                                 desc.cols);
            if (result == DistributedReducerResult::Aborted) {
                observeDistributedSoftmaxJobAbort(state);
                return false;
            }
            if (result != DistributedReducerResult::Accepted) {
                return false;
            }
            statSoftmaxRows_->addData(1);
        }
        statPartialDone_->addData(desc.rows);
        state->stage = SoftmaxJobStage::Complete;
        state->status = SFUStatus::Success;
        return true;
    }

    return state->stage == SoftmaxJobStage::Complete;
}

void SFU::abortDistributedSoftmaxJob(JobOpState* state)
{
    if (state == nullptr ||
        (state->desc.flags & SFU_JOB_FLAG_DISTRIBUTED_COLUMNS) == 0) {
        return;
    }
    clearDistributedReductionResponseInbox(*state);
    if (state->distributedAbortObserved ||
        state->desc.worker_cores == 0 || state->desc.reserved0 >= state->desc.worker_cores) {
        return;
    }
    auto& rows = distributedSoftmaxReducerRows();
    const auto existingKeys = collectDistributedSoftmaxJobKeys(state->desc.job_id,
                                                               state->tag,
                                                               state->desc.owner_core);
    for (const auto& key : existingKeys) {
        auto it = rows.find(key);
        if (it == rows.end()) {
            continue;
        }
        it->second.aborted = true;
        markDistributedSoftmaxRowAborted(key);
        clearDistributedReductionResponseFanout(key);
        if (state->desc.reserved0 < it->second.expectedWorkers) {
            (void)observeDistributedSoftmaxAbort(key, &it->second, state->desc.reserved0);
        }
    }
    for (uint32_t row = 0; row < state->desc.rows; ++row) {
        const auto key = distributedSoftmaxReducerKey(state->desc.job_id,
                                                      state->tag,
                                                      state->desc.owner_core,
                                                      row);
        if (std::find(existingKeys.begin(), existingKeys.end(), key) != existingKeys.end()) {
            continue;
        }
        abortDistributedSoftmaxRow(state->desc.job_id,
                                   state->tag,
                                   state->desc.owner_core,
                                   row,
                                   state->desc.reserved0,
                                   state->desc.worker_cores,
                                   state->desc.rows,
                                   state->desc.cols);
    }
}

void SFU::clearDistributedReductionResponseInbox(const JobOpState& state)
{
    for (auto it = distributedReductionResponseInbox_.begin();
         it != distributedReductionResponseInbox_.end();) {
        const DistributedReductionResponseInboxKey& key = it->first;
        if (std::get<0>(key) == state.desc.job_id && std::get<1>(key) == state.tag &&
            std::get<2>(key) == state.desc.owner_core) {
            it = distributedReductionResponseInbox_.erase(it);
        } else {
            ++it;
        }
    }
}

void SFU::observeDistributedSoftmaxJobAbort(JobOpState* state)
{
    if (state == nullptr || state->distributedAbortObserved ||
        (state->desc.flags & SFU_JOB_FLAG_DISTRIBUTED_COLUMNS) == 0) {
        return;
    }
    auto& rows = distributedSoftmaxReducerRows();
    const auto keys = collectDistributedSoftmaxJobKeys(state->desc.job_id,
                                                       state->tag,
                                                       state->desc.owner_core);
    for (const auto& key : keys) {
        auto it = rows.find(key);
        if (it != rows.end() && it->second.aborted) {
            (void)observeDistributedSoftmaxAbort(key, &it->second, state->desc.reserved0);
        }
    }
    state->distributedAbortObserved = true;
}

bool SFU::readSoftmaxDescriptor(uint64_t descAddr, SFUSoftmaxTileDesc* desc)
{
    if (globalMem_ == nullptr || desc == nullptr || descAddr == 0) {
        return false;
    }

    std::vector<uint8_t> raw;
    globalMem_->rd_from_globalmem(descAddr, sizeof(SFUSoftmaxTileDesc), raw);
    if (raw.size() != sizeof(SFUSoftmaxTileDesc)) {
        return false;
    }

    std::memcpy(desc, raw.data(), sizeof(SFUSoftmaxTileDesc));
    return true;
}

SFUStatus SFU::validateSoftmaxDescriptor(const SFUSoftmaxTileDesc& desc) const
{
    if (desc.elem_bytes != sizeof(float)) {
        return SFUStatus::UnsupportedElemBytes;
    }
    if (desc.local_input_gm_addr == 0 || desc.local_output_gm_addr == 0 ||
        desc.global_m == 0 || desc.global_n == 0 ||
        desc.block_m == 0 || desc.block_n == 0 ||
        desc.valid_m == 0 || desc.valid_n == 0 ||
        desc.n_tiles_per_row == 0 ||
        desc.valid_m > desc.block_m || desc.valid_n > desc.block_n) {
        return SFUStatus::InvalidShape;
    }

    const uint64_t firstRow = static_cast<uint64_t>(desc.m_tile) * desc.block_m;
    const uint64_t firstCol = static_cast<uint64_t>(desc.n_tile) * desc.block_n;
    if (firstRow + desc.valid_m > desc.global_m || firstCol + desc.valid_n > desc.global_n) {
        return SFUStatus::InvalidShape;
    }
    return SFUStatus::Success;
}

bool SFU::readInputTile(const SFUSoftmaxTileDesc& desc, std::vector<float>* values)
{
    if (globalMem_ == nullptr || values == nullptr) {
        return false;
    }

    const size_t elemCount = static_cast<size_t>(desc.block_m) * desc.block_n;
    const size_t byteCount = elemCount * desc.elem_bytes;
    std::vector<uint8_t> raw;
    globalMem_->rd_from_globalmem(desc.local_input_gm_addr, byteCount, raw);
    if (raw.size() != byteCount) {
        return false;
    }

    values->assign(elemCount, 0.0f);
    for (size_t i = 0; i < elemCount; ++i) {
        float value = 0.0f;
        std::memcpy(&value, raw.data() + i * sizeof(float), sizeof(float));
        (*values)[i] = value;
    }
    return true;
}

void SFU::computeTileStats(SoftmaxOpState* state)
{
    if (state == nullptr) {
        return;
    }

    const SFUSoftmaxTileDesc& desc = state->desc;
    state->rowStats.clear();
    state->rowStats.reserve(desc.valid_m);
    for (uint32_t row = 0; row < desc.valid_m; ++row) {
        double tile_m = -std::numeric_limits<double>::infinity();
        for (uint32_t col = 0; col < desc.valid_n; ++col) {
            const size_t idx = tilePackedIndex(desc, row, col);
            tile_m = std::max(tile_m, static_cast<double>(state->inputTile[idx]));
        }

        double tile_l = 0.0;
        for (uint32_t col = 0; col < desc.valid_n; ++col) {
            const size_t idx = tilePackedIndex(desc, row, col);
            tile_l += std::exp(static_cast<double>(state->inputTile[idx]) - tile_m);
        }

        state->rowStats.push_back(SFUTileRowStats{
            static_cast<uint32_t>(desc.m_tile * desc.block_m + row),
            tile_m,
            tile_l,
        });
        statSoftmaxRows_->addData(1);
    }
}

bool SFU::mergeTileStats(SoftmaxOpState* state)
{
    if (state == nullptr) {
        return false;
    }

    bool allReady = true;
    auto& rows = softmaxReducerRows();
    for (const auto& row : state->rowStats) {
        const SoftmaxReducerKey key(state->desc.job_id, row.global_row);
        auto& rowState = rows[key];
        if (rowState.n_tiles_expected == 0) {
            rowState.n_tiles_expected = state->desc.n_tiles_per_row;
        }

        const double m_new = std::max(rowState.m_acc, row.tile_m);
        const double l_new = rowState.l_acc * std::exp(rowState.m_acc - m_new) +
                             row.tile_l * std::exp(row.tile_m - m_new);
        rowState.m_acc = m_new;
        rowState.l_acc = l_new;
        rowState.partials_seen += 1;
        rowState.ready = rowState.partials_seen >= rowState.n_tiles_expected;
        allReady = allReady && rowState.ready;
    }
    return allReady;
}

bool SFU::tileGlobalStatsReady(const SoftmaxOpState& state) const
{
    const auto& rows = softmaxReducerRows();
    for (const auto& row : state.rowStats) {
        const SoftmaxReducerKey key(state.desc.job_id, row.global_row);
        auto it = rows.find(key);
        if (it == rows.end() || !it->second.ready) {
            return false;
        }
    }
    return true;
}

bool SFU::normalizeTile(SoftmaxOpState* state)
{
    if (state == nullptr || globalMem_ == nullptr) {
        return false;
    }

    const SFUSoftmaxTileDesc& desc = state->desc;
    const size_t elemCount = static_cast<size_t>(desc.block_m) * desc.block_n;
    std::vector<float> outputTile(elemCount, 0.0f);

    auto& rows = softmaxReducerRows();
    for (uint32_t row = 0; row < desc.block_m; ++row) {
        const uint32_t globalRow = static_cast<uint32_t>(desc.m_tile * desc.block_m + row);
        const SoftmaxReducerKey key(desc.job_id, globalRow);
        auto rowIt = rows.find(key);
        if (row < desc.valid_m && (rowIt == rows.end() || !rowIt->second.ready || rowIt->second.l_acc == 0.0)) {
            return false;
        }

        for (uint32_t col = 0; col < desc.block_n; ++col) {
            if (row < desc.valid_m && col < desc.valid_n) {
                const auto& rowState = rowIt->second;
                const size_t idx = tilePackedIndex(desc, row, col);
                const double normalized =
                    std::exp(static_cast<double>(state->inputTile[idx]) - rowState.m_acc) / rowState.l_acc;
                outputTile[idx] = static_cast<float>(normalized);
            }
        }
    }

    std::vector<uint8_t> out;
    out.reserve(elemCount * sizeof(float));
    for (const float value : outputTile) {
        appendBytes(out, value);
    }
    globalMem_->wr_to_globalmem(desc.local_output_gm_addr, out.size(), out);

    for (const auto& row : state->rowStats) {
        const SoftmaxReducerKey key(desc.job_id, row.global_row);
        auto it = rows.find(key);
        if (it != rows.end()) {
            it->second.normalizes_done += 1;
            if (it->second.normalizes_done >= it->second.n_tiles_expected) {
                rows.erase(it);
            }
        }
    }
    return true;
}

void SFU::bindGlobalMemory(GlobalMemoryAPI* globalMem)
{
    globalMem_ = globalMem;
}

void SFU::setCoreInfo(uint32_t coreId, uint32_t activeWorkerCores)
{
    coreId_ = coreId;
    activeWorkerCores_ = activeWorkerCores == 0 ? 1 : activeWorkerCores;
}

void SFU::receiveReductionMessage(const ReductionTransportMessage& message)
{
    handleReductionTransportMessage(message);
}

} // namespace Golem
} // namespace SST
