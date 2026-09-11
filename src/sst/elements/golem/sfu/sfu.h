#ifndef _H_GOLEM_SFU
#define _H_GOLEM_SFU

#include <cstdint>
#include <array>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <sst/core/output.h>
#include <sst/core/event.h>
#include <sst/core/link.h>
#include <sst/core/params.h>
#include <sst/core/statapi/statbase.h>
#include <sst/core/subcomponent.h>

#include <sst/elements/golem/globalmemory/globalmemory.h>

namespace SST {
namespace Golem {

enum class TensorRowEngineStage : uint8_t {
    Max,
    ExpSum,
    Normalize,
};

enum class TensorRowEngineEventKind : uint8_t {
    ResourceDone,
    LocalReadRetry,
    LocalWriteRetry,
    AttentionStart,
};

class TensorRowEngineEvent : public SST::Event {
public:
    TensorRowEngineEvent() = default;
    TensorRowEngineEvent(uint64_t tag,
                         uint32_t bandRow,
                         uint32_t context,
                         TensorRowEngineStage stage,
                         TensorRowEngineEventKind kind,
                         uint64_t localTag)
        : tag_(tag), bandRow_(bandRow), context_(context), stage_(stage),
          kind_(kind), localTag_(localTag) {}

    uint64_t tag() const { return tag_; }
    uint32_t bandRow() const { return bandRow_; }
    uint32_t context() const { return context_; }
    TensorRowEngineStage stage() const { return stage_; }
    TensorRowEngineEventKind kind() const { return kind_; }
    uint64_t localTag() const { return localTag_; }

    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        ser & tag_;
        ser & bandRow_;
        ser & context_;
        ser & stage_;
        ser & kind_;
        ser & localTag_;
    }

    ImplementSerializable(SST::Golem::TensorRowEngineEvent);

private:
    uint64_t tag_ = 0;
    uint32_t bandRow_ = 0;
    uint32_t context_ = 0;
    TensorRowEngineStage stage_ = TensorRowEngineStage::Max;
    TensorRowEngineEventKind kind_ = TensorRowEngineEventKind::ResourceDone;
    uint64_t localTag_ = 0;
};

struct SFUSoftmaxTileDesc {
    uint64_t job_id;
    uint64_t local_input_gm_addr;
    uint64_t local_output_gm_addr;
    uint32_t global_m;
    uint32_t global_n;
    uint32_t block_m;
    uint32_t block_n;
    uint32_t m_tile;
    uint32_t n_tile;
    uint32_t valid_m;
    uint32_t valid_n;
    uint32_t n_tiles_per_row;
    uint32_t elem_bytes;
    uint32_t flags;
};

static_assert(sizeof(SFUSoftmaxTileDesc) == 72,
              "SFUSoftmaxTileDesc ABI must stay fixed for RISC-V workload descriptors");

enum class SFUPrimitiveOp : uint32_t {
    EXP = 0x01,
    LOG = 0x02,
    RECIPROCAL = 0x03,
    RSQRT = 0x04,
    SQRT = 0x05,
    TANH = 0x06,
    SIGMOID = 0x07,
    REDUCE_MAX = 0x20,
    REDUCE_SUM = 0x21,
    GELU = 0x40,
    LAYERNORM = 0x41,
    FUSED_SOFTMAX = 0x80,
};

struct SFUPrimitiveDesc {
    uint64_t job_id;
    uint64_t input0_gm_addr;
    uint64_t input1_gm_addr;
    uint64_t output_gm_addr;
    uint32_t op;
    uint32_t dtype;
    uint32_t elem_count;
    uint32_t input0_stride_bytes;
    uint32_t input1_stride_bytes;
    uint32_t output_stride_bytes;
    uint32_t flags;
    uint32_t approx_mode;
};

static_assert(sizeof(SFUPrimitiveDesc) == 64,
              "SFUPrimitiveDesc ABI must stay fixed for RISC-V workload descriptors");

struct SFUPrimitiveBatchDesc {
    uint64_t job_id;
    uint64_t desc_array_gm_addr;
    uint32_t desc_count;
    uint32_t flags;
    uint64_t reserved0;
};

static_assert(sizeof(SFUPrimitiveBatchDesc) == 32,
              "SFUPrimitiveBatchDesc ABI must stay fixed for RISC-V workload descriptors");

enum class SFUJobOp : uint32_t {
    ELEMENTWISE = 0x01,
    REDUCE = 0x02,
    SOFTMAX_ROW = 0x10,
    LAYERNORM = 0x11,
    GELU = 0x12,
};

enum class SFUJobSubOp : uint32_t {
    NONE = 0x00,
    EXP = 0x01,
    LOG = 0x02,
    RECIPROCAL = 0x03,
    RSQRT = 0x04,
    TANH = 0x05,
    SIGMOID = 0x06,
    REDUCE_MAX = 0x20,
    REDUCE_SUM = 0x21,
};

constexpr uint32_t SFU_JOB_FLAG_DISTRIBUTED_COLUMNS = 0x1u;
constexpr uint32_t SFU_JOB_FLAG_DISTRIBUTED_ABORT = 0x2u;
constexpr uint32_t SFU_JOB_FLAG_ROW_ENGINE_MODEL = 0x4u;
constexpr uint32_t SFU_JOB_FLAG_TENSOR_ROW_ENGINE = 0x8u;
constexpr uint32_t SFU_JOB_DTYPE_FP32 = 1u;

constexpr uint32_t SFU_SOFTMAX_JOB_PARAMS_MAGIC = 0x53465531u;
constexpr uint16_t SFU_SOFTMAX_JOB_PARAMS_VERSION = 1u;
constexpr uint16_t SFU_SOFTMAX_JOB_PARAMS_VERSION_ATTENTION = 2u;
constexpr uint16_t SFU_SOFTMAX_JOB_PARAMS_VERSION_MANAGER = 3u;
constexpr uint32_t SFU_SOFTMAX_PARAMS_FLAG_ATTENTION = 0x1u;
constexpr uint32_t SFU_SOFTMAX_PARAMS_FLAG_CAUSAL = 0x2u;
constexpr uint32_t SFU_SOFTMAX_HBM_LAYOUT_BAND_STRIPED = 1u;
constexpr uint32_t SFU_SOFTMAX_MAPPING_EXPLICIT_TOPOLOGY = 2u;
constexpr uint32_t SFU_WORKER_TOPOLOGY_MAP_MAGIC = 0x574d4150u;
constexpr uint16_t SFU_WORKER_TOPOLOGY_MAP_VERSION = 1u;
constexpr uint32_t SFU_WORKER_TOPOLOGY_MAX_WORKERS = 16u;

struct SFUSoftmaxJobParamsV1 {
    uint32_t magic;
    uint16_t version;
    uint16_t size_bytes;
    uint32_t mapping_policy;
    uint32_t tiles_per_row;
    uint32_t row_contexts_hint;
    uint32_t hbm_layout;
    uint32_t data_node_mask;
    uint32_t flags;
    uint64_t completion_addr;
    uint64_t node_stride_bytes;
    uint32_t rows_per_band;
    uint32_t coordinator_core;
    uint64_t reserved0;
};

static_assert(sizeof(SFUSoftmaxJobParamsV1) == 64,
              "Tensor softmax parameter ABI must stay fixed");

struct SFUWorkerTopologyMapV1 {
    uint32_t magic;
    uint16_t version;
    uint16_t size_bytes;
    uint32_t worker_count;
    uint32_t reserved0;
    uint32_t worker_core_ids[SFU_WORKER_TOPOLOGY_MAX_WORKERS];
};

static_assert(sizeof(SFUWorkerTopologyMapV1) == 80,
              "Worker topology map ABI must stay fixed");

struct SFUJobDesc {
    uint64_t job_id;
    uint64_t input0_addr;
    uint64_t input1_addr;
    uint64_t output_addr;
    uint64_t params_addr;
    uint64_t scratch_addr;
    uint32_t op_type;
    uint32_t sub_op;
    uint32_t dtype;
    uint32_t layout;
    uint32_t rows;
    uint32_t cols;
    uint32_t elem_count;
    uint32_t chunk_elems;
    uint32_t worker_cores;
    uint32_t owner_core;
    uint32_t flags;
    uint32_t reserved0;  // With DISTRIBUTED_COLUMNS, reserved0 stores the worker slot.
    uint64_t reserved1;
    uint64_t reserved2;
    uint64_t reserved3;
    uint64_t reserved4;
};

static_assert(sizeof(SFUJobDesc) == 128,
              "SFUJobDesc ABI must stay fixed for RISC-V workload descriptors");

enum class SFUStatus : uint64_t {
    Success = 0,
    Pending = 1,
    InvalidDescriptor = 2,
    UnsupportedElemBytes = 3,
    GlobalMemoryUnavailable = 4,
    InvalidShape = 5,
};

struct SFUTileRowStats {
    uint32_t global_row;
    double tile_m;
    double tile_l;
};

struct AttentionTileRequest {
    uint64_t tag = 0;
    uint64_t jobId = 0;
    uint64_t localScoreAddr = 0;
    uint32_t globalRowBegin = 0;
    uint32_t keyBegin = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t headDim = 0;
    uint32_t keyTile = 0;
    uint32_t keyTiles = 1;
    bool causal = false;
    bool firstTileForJob = false;
};

struct AttentionTileResult {
    uint32_t rows = 0;
    std::array<float, 16> oldOutputScale = {};
};

class SFUAPI : public SST::SubComponent {
public:
    SST_ELI_REGISTER_SUBCOMPONENT_API(SST::Golem::SFUAPI)

    SFUAPI(ComponentId_t id, SST::Params& params) : SST::SubComponent(id) {}
    ~SFUAPI() override = default;

    virtual bool issueSoftmaxTile(uint64_t descAddr, uint64_t tag) = 0;
    virtual bool issuePrimitive(uint64_t descAddr, uint64_t tag) = 0;
    virtual bool issuePrimitiveBatch(uint64_t descAddr, uint64_t tag) = 0;
    virtual bool issueJob(uint64_t descAddr, uint64_t tag) = 0;
    virtual bool issueAttentionTile(const AttentionTileRequest& request,
                                    std::function<void(bool, const AttentionTileResult&)> callback) = 0;
    virtual bool completionTick(uint64_t tag, uint64_t* tick) const = 0;
    virtual bool wait(uint64_t tag, uint64_t* status) = 0;
    virtual void bindGlobalMemory(GlobalMemoryAPI* globalMem) = 0;
    virtual void setCoreInfo(uint32_t coreId, uint32_t activeWorkerCores) = 0;
    virtual void receiveReductionMessage(const ReductionTransportMessage& message) = 0;
};

class SFU : public SFUAPI {
public:
    SST_ELI_REGISTER_SUBCOMPONENT(
        SFU,
        "golem",
        "SFU",
        SST_ELI_ELEMENT_VERSION(1, 0, 0),
        "Special Function Unit for Golem",
        SST::Golem::SFUAPI)

    SST_ELI_DOCUMENT_PARAMS(
        {"core_id", "Owning core id", "0"},
        {"active_worker_cores", "Number of active worker cores", "1"},
        {"max_inflight", "Maximum in-flight SFU operations", "8"},
        {"stats_latency", "Softmax tile statistics latency", "1"},
        {"merge_latency", "Softmax online merge latency", "1"},
        {"normalize_latency", "Softmax normalize latency", "1"},
        {"accelerator_clock_hz", "Row Engine accelerator clock frequency", "2300000000"},
        {"vector_lanes", "FP32 vector lanes per physical Row Engine", "16"},
        {"exp_lanes", "FP32 EXP issue lanes per physical Row Engine", "4"},
        {"reduction_tree_latency", "Row Engine reduction tree latency in accelerator cycles", "4"},
        {"exp_latency", "Row Engine EXP pipeline latency in accelerator cycles", "8"},
        {"reciprocal_latency", "Row Engine reciprocal latency in accelerator cycles", "1"},
        {"rsqrt_latency", "Attention scale RSQRT latency in accelerator cycles", "8"},
        {"row_contexts", "Row contexts per physical Row Engine", "4"},
        {"attention_kv_pair_reuse", "Retain online Softmax state for grouped Attention query blocks", "0"},
        {"attention_kv_query_group_size", "Interleaved Attention query blocks: 2 or 4", "2"},
        {"scratchpad_bytes", "Row Engine scratchpad capacity", "65536"},
        {"distributed_reduction_transport", "Distributed softmax reduction transport: shared, modeled_noc, or explicit_noc", "shared"},
        {"verbose", "Verbosity", "0"})

    SST_ELI_DOCUMENT_STATISTICS(
        {"sfu_ops_issued", "Issued SFU operations", "ops", 1},
        {"sfu_softmax_rows", "Softmax rows processed by SFU", "rows", 1},
        {"sfu_softmax_tiles", "Softmax tiles processed by SFU", "tiles", 1},
        {"sfu_job_softmax_max_chunks", "Unified softmax job max-pass chunks", "chunks", 1},
        {"sfu_job_softmax_sum_chunks", "Unified softmax job exp/sum-pass chunks", "chunks", 1},
        {"sfu_job_softmax_norm_chunks", "Unified softmax job normalize-pass chunks", "chunks", 1},
        {"sfu_row_engine_jobs", "Row Engine jobs issued", "jobs", 1},
        {"sfu_row_engine_rows", "Rows assigned to this physical Row Engine", "rows", 1},
        {"sfu_row_engine_max_cycles", "Modeled Row Engine max-pass active cycles", "cycles", 1},
        {"sfu_row_engine_exp_sum_cycles", "Modeled Row Engine EXP/sum-pass active cycles", "cycles", 1},
        {"sfu_row_engine_normalize_cycles", "Modeled Row Engine normalize-pass active cycles", "cycles", 1},
        {"sfu_row_engine_max_start_cycles", "Row Engine MAX stage start offset from job acceptance", "cycles", 1},
        {"sfu_row_engine_max_end_cycles", "Row Engine MAX stage end offset from job acceptance", "cycles", 1},
        {"sfu_row_engine_exp_sum_start_cycles", "Row Engine EXP/sum stage start offset from job acceptance", "cycles", 1},
        {"sfu_row_engine_exp_sum_end_cycles", "Row Engine EXP/sum stage end offset from job acceptance", "cycles", 1},
        {"sfu_row_engine_normalize_start_cycles", "Row Engine normalize stage start offset from job acceptance", "cycles", 1},
        {"sfu_row_engine_normalize_end_cycles", "Row Engine normalize stage end offset from job acceptance", "cycles", 1},
        {"sfu_row_engine_modeled_cycles", "Modeled Row Engine job latency", "cycles", 1},
        {"sfu_row_engine_queue_wait_cycles", "Modeled cycles waiting for this physical Row Engine", "cycles", 1},
        {"sfu_row_engine_wait_polls", "Compatibility wait polls while a Row Engine job is pending", "polls", 1},
        {"sfu_row_engine_completed_jobs", "Row Engine jobs observed complete", "jobs", 1},
        {"sfu_row_engine_issue_tick", "SST timebase tick when a Row Engine job was issued", "ticks", 1},
        {"sfu_row_engine_start_tick", "SST timebase tick when Row Engine resources were reserved", "ticks", 1},
        {"sfu_row_engine_ready_tick", "SST timebase tick when a Row Engine job became ready", "ticks", 1},
        {"sfu_row_engine_completion_observed_tick", "SST timebase tick when guest observed completion", "ticks", 1},
        {"sfu_tensor_band_dispatch_tick", "SST tick for each tensor band dispatch", "ticks", 1},
        {"sfu_tensor_worker_dispatch_tick", "SST tick when a worker accepts a tensor band", "ticks", 1},
        {"sfu_tensor_input_dma_ready_tick", "SST tick for each completed tensor input DMA", "ticks", 1},
        {"sfu_tensor_max_start_tick", "SST tick when a tensor row enters MAX", "ticks", 1},
        {"sfu_tensor_max_done_tick", "SST tick when a tensor row finishes MAX", "ticks", 1},
        {"sfu_tensor_exp_sum_start_tick", "SST tick when a tensor row enters EXP/SUM", "ticks", 1},
        {"sfu_tensor_exp_sum_done_tick", "SST tick when a tensor row finishes EXP/SUM", "ticks", 1},
        {"sfu_tensor_normalize_start_tick", "SST tick when a tensor row enters NORMALIZE", "ticks", 1},
        {"sfu_tensor_normalize_done_tick", "SST tick when a tensor row finishes NORMALIZE", "ticks", 1},
        {"sfu_tensor_compute_done_tick", "SST tick when NORMALIZE finishes and output DMA is issued", "ticks", 1},
        {"sfu_tensor_output_dma_ack_tick", "SST tick for each tensor output DMA ACK", "ticks", 1},
        {"sfu_tensor_completion_received_tick", "SST tick when the coordinator receives a band completion", "ticks", 1},
        {"sfu_tensor_guest_wait_observed_tick", "SST tick when the SFU wait returns tensor completion", "ticks", 1},
        {"sfu_attention_jobs", "Attention-aware tensor softmax jobs", "jobs", 1},
        {"sfu_attention_rsqrt_ready_tick", "SST tick when attention RSQRT(D) is ready", "ticks", 1},
        {"sfu_attention_scale_mask_start_tick", "SST tick when attention SCALE/MASK starts", "ticks", 1},
        {"sfu_attention_scale_mask_done_tick", "SST tick when attention SCALE/MASK completes", "ticks", 1},
        {"sfu_attention_scaled_elements", "Attention score elements scaled", "elements", 1},
        {"sfu_attention_masked_elements", "Future attention score elements masked", "elements", 1},
        {"sfu_primitive_elems", "Logical primitive elements processed by SFU", "elements", 1},
        {"sfu_partial_submits", "Softmax partial stats submitted", "partials", 1},
        {"sfu_partial_done", "Softmax partial stats completed", "partials", 1},
        {"sfu_reduction_max_requests", "Distributed softmax max reduction requests", "messages", 1},
        {"sfu_reduction_max_responses", "Distributed softmax max reduction responses", "messages", 1},
        {"sfu_reduction_sum_requests", "Distributed softmax sum reduction requests", "messages", 1},
        {"sfu_reduction_sum_responses", "Distributed softmax sum reduction responses", "messages", 1},
        {"sfu_reduction_transport_received", "Explicit-NoC reduction messages delivered to SFU", "messages", 1},
        {"sfu_reduction_transport_stale_dropped", "Explicit-NoC stale or duplicate reduction messages dropped", "messages", 1},
        {"sfu_reduction_transport_inbox_high_water", "Explicit-NoC reduction response inbox high-water mark", "messages", 1},
        {"sfu_reduction_transport_latency_cycles", "Explicit-NoC reduction message transport latency", "cycles", 1},
        {"sfu_credit_stalls", "SFU credit stalls", "stalls", 1},
        {"sfu_cross_tile_wait_cycles", "SFU cross-tile wait cycles", "cycles", 1},
        {"sfu_retry_events", "SFU retry events", "events", 1})

    SFU(ComponentId_t id, SST::Params& params);
    ~SFU() override = default;

    void finish() override;

    bool issueSoftmaxTile(uint64_t descAddr, uint64_t tag) override;
    bool issuePrimitive(uint64_t descAddr, uint64_t tag) override;
    bool issuePrimitiveBatch(uint64_t descAddr, uint64_t tag) override;
    bool issueJob(uint64_t descAddr, uint64_t tag) override;
    bool issueAttentionTile(const AttentionTileRequest& request,
                            std::function<void(bool, const AttentionTileResult&)> callback) override;
    bool completionTick(uint64_t tag, uint64_t* tick) const override;
    bool wait(uint64_t tag, uint64_t* status) override;
    void bindGlobalMemory(GlobalMemoryAPI* globalMem) override;
    void setCoreInfo(uint32_t coreId, uint32_t activeWorkerCores) override;
    void receiveReductionMessage(const ReductionTransportMessage& message) override;

private:
    enum class SoftmaxJobStage : uint8_t {
        None,
        MaxSubmitted,
        SumSubmitted,
        Complete,
    };

    enum class DistributedReductionTransport : uint8_t {
        Shared,
        ModeledNoC,
        ExplicitNoC,
    };

    using DistributedReductionResponseInboxKey =
        std::tuple<uint64_t,
                   uint64_t,
                   uint32_t,
                   uint32_t,
                   uint32_t,
                   ReductionTransportMessageKind>;

    struct SoftmaxOpState {
        SFUSoftmaxTileDesc desc;
        uint64_t descAddr;
        uint64_t tag;
        SFUStatus status;
        std::vector<SFUTileRowStats> rowStats;
        std::vector<float> inputTile;
        bool normalizeReady;
    };

    struct PrimitiveOpState {
        SFUPrimitiveDesc desc;
        uint64_t descAddr;
        uint64_t tag;
        SFUStatus status;
        std::vector<float> input0;
        std::vector<float> output;
    };

    struct PrimitiveBatchOpState {
        SFUPrimitiveBatchDesc desc;
        uint64_t descAddr;
        uint64_t tag;
        SFUStatus status;
        uint64_t processedElems;
    };

    struct JobOpState {
        SFUJobDesc desc;
        uint64_t descAddr;
        uint64_t tag;
        SFUStatus status;
        uint64_t processedElems;
        uint64_t rowEngineIssueTick;
        uint64_t rowEngineStartTick;
        uint64_t rowEngineReadyTick;
        uint64_t rowEngineModeledCycles;
        SFUSoftmaxJobParamsV1 tensorParams;
        uint32_t tensorRowsCompleted;
        bool tensorDmaComplete;
        bool tensorStarted;
        bool attentionMode;
        bool attentionCausal;
        float attentionScale;
        uint64_t attentionRsqrtReadyTick;
        std::vector<uint8_t> tensorCompletionSeen;
        SoftmaxJobStage stage;
        uint32_t workerSlot;
        uint32_t colBegin;
        uint32_t colEnd;
        std::vector<double> localMax;
        std::vector<double> localSum;
        std::vector<uint8_t> maxResponseSeen;
        std::vector<uint8_t> sumResponseSeen;
        bool distributedAbortObserved;
    };

    struct TensorWorkerState {
        struct Context {
            uint32_t row = 0;
            uint64_t scratchAddr = 0;
            bool busy = false;
            TensorRowEngineStage stage = TensorRowEngineStage::Max;
            uint32_t chunkBegin = 0;
            uint64_t pendingLocalTag = 0;
            float rowMax = 0.0f;
            double rowSum = 0.0;
            float invSum = 0.0f;
            float oldOutputScale = 0.0f;
            float tileWeightScale = 1.0f;
            std::vector<float> laneValues;
        };

        ReductionTransportMessage dispatch;
        uint64_t scratchAddr;
        uint32_t nextRow;
        uint32_t rowsCompleted;
        bool localTileMode = false;
        uint64_t attentionJobId = 0;
        uint32_t attentionKeyTile = 0;
        uint32_t attentionKeyTiles = 1;
        uint32_t attentionKeyBegin = 0;
        bool attentionFirstTileForJob = false;
        AttentionTileResult attentionResult;
        std::function<void(bool, const AttentionTileResult&)> localTileCallback;
        std::vector<Context> contexts;
    };

    struct AttentionOnlineRowContext {
        bool valid = false;
        uint64_t jobId = 0;
        uint32_t globalRow = 0;
        double m = -std::numeric_limits<double>::infinity();
        double l = 0.0;
    };

    using TensorWorkerKey = std::pair<uint64_t, uint32_t>;

    bool readSoftmaxDescriptor(uint64_t descAddr, SFUSoftmaxTileDesc* desc);
    SFUStatus validateSoftmaxDescriptor(const SFUSoftmaxTileDesc& desc) const;
    bool readInputTile(const SFUSoftmaxTileDesc& desc, std::vector<float>* values);
    void computeTileStats(SoftmaxOpState* state);
    bool mergeTileStats(SoftmaxOpState* state);
    bool tileGlobalStatsReady(const SoftmaxOpState& state) const;
    bool normalizeTile(SoftmaxOpState* state);
    bool readPrimitiveDescriptor(uint64_t descAddr, SFUPrimitiveDesc* desc);
    bool readPrimitiveBatchDescriptor(uint64_t descAddr, SFUPrimitiveBatchDesc* desc);
    SFUStatus validatePrimitiveDescriptor(const SFUPrimitiveDesc& desc) const;
    SFUStatus validatePrimitiveBatchDescriptor(const SFUPrimitiveBatchDesc& desc) const;
    bool readPrimitiveInput(const SFUPrimitiveDesc& desc, std::vector<float>* values);
    bool executePrimitive(PrimitiveOpState* state);
    bool writePrimitiveOutput(const SFUPrimitiveDesc& desc, const std::vector<float>& values);
    bool executePrimitiveDesc(uint64_t descAddr, uint64_t tag, uint64_t* processedElems);
    uint64_t primitiveProcessedElems(const SFUPrimitiveDesc& desc) const;
    bool readJobDescriptor(uint64_t descAddr, SFUJobDesc* desc);
    bool readTensorJobParams(uint64_t paramsAddr, SFUSoftmaxJobParamsV1* params);
    SFUStatus validateJobDescriptor(const SFUJobDesc& desc) const;
    bool executeJob(JobOpState* state);
    bool executeSoftmaxRowJob(JobOpState* state);
    uint64_t rowEngineModeledCycles(const SFUJobDesc& desc,
                                    uint64_t* maxCycles,
                                    uint64_t* expSumCycles,
                                    uint64_t* normalizeCycles,
                                    uint64_t* maxStartCycles,
                                    uint64_t* maxEndCycles,
                                    uint64_t* expSumStartCycles,
                                    uint64_t* expSumEndCycles,
                                    uint64_t* normalizeStartCycles,
                                    uint64_t* normalizeEndCycles) const;
    bool startTensorRowEngineJob(uint64_t tag);
    bool finishTensorJobIfReady(JobOpState* state);
    void handleTensorRowDispatch(const ReductionTransportMessage& message);
    void rejectTensorRowDispatch(const ReductionTransportMessage& message);
    void handleTensorRowComplete(const ReductionTransportMessage& message);
    void issueTensorInputDma(const TensorWorkerKey& key, uint32_t contextIndex);
    void beginTensorRowStage(const TensorWorkerKey& key,
                             uint32_t contextIndex,
                             TensorRowEngineStage stage);
    void issueTensorLocalRead(const TensorWorkerKey& key, uint32_t contextIndex);
    void issueTensorLocalWrite(const TensorWorkerKey& key, uint32_t contextIndex);
    void advanceTensorRowChunk(const TensorWorkerKey& key, uint32_t contextIndex);
    void completeTensorRow(const TensorWorkerKey& key, uint32_t contextIndex);
    void scheduleTensorLocalRetry(const TensorWorkerKey& key,
                                  uint32_t contextIndex,
                                  TensorRowEngineEventKind kind);
    void scheduleTensorRowStage(const TensorWorkerKey& key,
                                uint32_t contextIndex,
                                TensorRowEngineStage stage);
    void handleTensorRowEngineEvent(SST::Event* event);
    void finishTensorWorker(const TensorWorkerKey& key, bool ok);
    uint64_t rowEngineCurrentCycle() const;
    uint64_t tensorWorkerHostAddress(const ReductionTransportMessage& message,
                                     uint32_t row,
                                     bool output) const;
    bool executeDistributedSoftmaxRowJob(JobOpState* state);
    bool advanceDistributedSoftmaxJob(JobOpState* state);
    void abortDistributedSoftmaxJob(JobOpState* state);
    void observeDistributedSoftmaxJobAbort(JobOpState* state);
    bool modeledDistributedReductionEnabled() const;
    bool explicitDistributedReductionEnabled() const;
    void recordDistributedReductionRequest(bool maxStage);
    void recordDistributedReductionResponse(bool maxStage);
    void handleReductionTransportMessage(const ReductionTransportMessage& message);
    void clearDistributedReductionResponseInbox(const JobOpState& state);

    uint32_t coreId_;
    uint32_t activeWorkerCores_;
    uint32_t maxInflight_;
    uint32_t statsLatency_;
    uint32_t mergeLatency_;
    uint32_t normalizeLatency_;
    uint64_t rowEngineAcceleratorClockHz_;
    uint32_t rowEngineVectorLanes_;
    uint32_t rowEngineExpLanes_;
    uint32_t rowEngineReductionTreeLatency_;
    uint32_t rowEngineExpLatency_;
    uint32_t rowEngineReciprocalLatency_;
    uint32_t rowEngineRsqrtLatency_;
    uint32_t rowEngineContexts_;
    uint64_t rowEngineScratchpadBytes_;
    uint64_t rowEngineTimebaseTicksPerSecond_;
    uint64_t rowEngineFreeTick_;
    uint64_t rowEngineVectorFreeCycle_;
    uint64_t rowEngineExpFreeCycle_;
    uint64_t nextTensorLocalTag_ = 1;
    uint64_t tensorLaneBufferHighWater_ = 0;
    uint64_t tensorMaxLocalReadBytes_ = 0;
    uint64_t tensorMaxLocalWriteBytes_ = 0;
    uint64_t tensorExpSumLocalReadBytes_ = 0;
    uint64_t tensorExpSumLocalWriteBytes_ = 0;
    uint64_t tensorNormalizeLocalReadBytes_ = 0;
    uint64_t tensorNormalizeLocalWriteBytes_ = 0;
    uint64_t tensorLocalRetryEvents_ = 0;
    uint32_t inflight_;
    int verbose_;
    DistributedReductionTransport distributedReductionTransport_;
    std::map<DistributedReductionResponseInboxKey, ReductionTransportMessage>
        distributedReductionResponseInbox_;
    uint64_t distributedReductionResponseInboxHighWater_ = 0;

    GlobalMemoryAPI* globalMem_;
    SST::Output output_;
    std::unordered_map<uint64_t, SoftmaxOpState> pendingSoftmaxOps_;
    std::unordered_map<uint64_t, PrimitiveOpState> pendingPrimitiveOps_;
    std::unordered_map<uint64_t, PrimitiveBatchOpState> pendingPrimitiveBatchOps_;
    std::unordered_map<uint64_t, JobOpState> pendingJobOps_;
    std::map<TensorWorkerKey, TensorWorkerState> tensorWorkerOps_;
    std::vector<AttentionOnlineRowContext> attentionOnlineContexts_;
    SST::Link* rowEngineSelfLink_;

    Statistic<uint64_t>* statOpsIssued_;
    Statistic<uint64_t>* statSoftmaxRows_;
    Statistic<uint64_t>* statSoftmaxTiles_;
    Statistic<uint64_t>* statJobSoftmaxMaxChunks_;
    Statistic<uint64_t>* statJobSoftmaxSumChunks_;
    Statistic<uint64_t>* statJobSoftmaxNormChunks_;
    Statistic<uint64_t>* statRowEngineJobs_;
    Statistic<uint64_t>* statRowEngineRows_;
    Statistic<uint64_t>* statRowEngineMaxCycles_;
    Statistic<uint64_t>* statRowEngineExpSumCycles_;
    Statistic<uint64_t>* statRowEngineNormalizeCycles_;
    Statistic<uint64_t>* statRowEngineMaxStartCycles_;
    Statistic<uint64_t>* statRowEngineMaxEndCycles_;
    Statistic<uint64_t>* statRowEngineExpSumStartCycles_;
    Statistic<uint64_t>* statRowEngineExpSumEndCycles_;
    Statistic<uint64_t>* statRowEngineNormalizeStartCycles_;
    Statistic<uint64_t>* statRowEngineNormalizeEndCycles_;
    Statistic<uint64_t>* statRowEngineModeledCycles_;
    Statistic<uint64_t>* statRowEngineQueueWaitCycles_;
    Statistic<uint64_t>* statRowEngineWaitPolls_;
    Statistic<uint64_t>* statRowEngineCompletedJobs_;
    Statistic<uint64_t>* statRowEngineIssueTick_;
    Statistic<uint64_t>* statRowEngineStartTick_;
    Statistic<uint64_t>* statRowEngineReadyTick_;
    Statistic<uint64_t>* statRowEngineCompletionObservedTick_;
    Statistic<uint64_t>* statTensorBandDispatchTick_;
    Statistic<uint64_t>* statTensorWorkerDispatchTick_;
    Statistic<uint64_t>* statTensorInputDmaReadyTick_;
    Statistic<uint64_t>* statTensorMaxStartTick_;
    Statistic<uint64_t>* statTensorMaxDoneTick_;
    Statistic<uint64_t>* statTensorExpSumStartTick_;
    Statistic<uint64_t>* statTensorExpSumDoneTick_;
    Statistic<uint64_t>* statTensorNormalizeStartTick_;
    Statistic<uint64_t>* statTensorNormalizeDoneTick_;
    Statistic<uint64_t>* statTensorComputeDoneTick_;
    Statistic<uint64_t>* statTensorOutputDmaAckTick_;
    Statistic<uint64_t>* statTensorCompletionReceivedTick_;
    Statistic<uint64_t>* statTensorGuestWaitObservedTick_;
    Statistic<uint64_t>* statAttentionJobs_;
    Statistic<uint64_t>* statAttentionRsqrtReadyTick_;
    Statistic<uint64_t>* statAttentionScaleMaskStartTick_;
    Statistic<uint64_t>* statAttentionScaleMaskDoneTick_;
    Statistic<uint64_t>* statAttentionScaledElements_;
    Statistic<uint64_t>* statAttentionMaskedElements_;
    Statistic<uint64_t>* statPrimitiveElems_;
    Statistic<uint64_t>* statPartialSubmits_;
    Statistic<uint64_t>* statPartialDone_;
    Statistic<uint64_t>* statReductionMaxRequests_;
    Statistic<uint64_t>* statReductionMaxResponses_;
    Statistic<uint64_t>* statReductionSumRequests_;
    Statistic<uint64_t>* statReductionSumResponses_;
    Statistic<uint64_t>* statReductionTransportReceived_;
    Statistic<uint64_t>* statReductionTransportStaleDropped_;
    Statistic<uint64_t>* statReductionTransportInboxHighWater_;
    Statistic<uint64_t>* statReductionTransportLatencyCycles_;
    Statistic<uint64_t>* statCreditStalls_;
    Statistic<uint64_t>* statCrossTileWaitCycles_;
    Statistic<uint64_t>* statRetryEvents_;
};

} // namespace Golem
} // namespace SST

#endif
