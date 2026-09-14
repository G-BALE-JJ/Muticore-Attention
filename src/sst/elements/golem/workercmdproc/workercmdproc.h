#ifndef _H_GOLEM_WORKER_COMMAND_PROCESSOR
#define _H_GOLEM_WORKER_COMMAND_PROCESSOR

#include <cinttypes>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <sst/core/component.h>
#include <sst/core/output.h>
#include <sst/core/params.h>
#include <sst/core/statapi/statbase.h>
#include <sst/core/subcomponent.h>

#include <sst/elements/golem/array/computeArray.h>
#include <sst/elements/golem/fp16.h>
#include <sst/elements/golem/globalmemory/globalmemory.h>
#include <sst/elements/golem/requestscheduler/requestscheduler.h>

namespace SST {
namespace Golem {

struct WorkerTaskListHeader {
    uint32_t worker_slot = 0;
    uint32_t task_count = 0;
    uint32_t active_worker_cores = 0;
    uint32_t total_groups = 0;
    uint32_t data_memory_node_count = 0;
    uint64_t mem_node_size = 0;
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t hw_input_size = 0;
    uint32_t hw_output_size = 0;
    uint32_t block_m = 0;
    uint32_t block_n = 0;
    uint32_t block_k = 0;
    uint32_t elem_bytes = 0;
    uint64_t mat_stride_bytes = 0;
    uint64_t vec_stride_bytes = 0;
    uint64_t off_gemm_mat_base = 0;
    uint64_t off_gemm_vec_base = 0;
    uint64_t off_gemm_out_base = 0;
    uint64_t local_mat_ping_gm_addr = 0;
    uint64_t local_mat_pong_gm_addr = 0;
    uint64_t local_mat_slot2_gm_addr = 0;
    uint64_t local_mat_slot3_gm_addr = 0;
    uint64_t local_vec_ping_gm_addr = 0;
    uint64_t local_vec_pong_gm_addr = 0;
    uint64_t local_vec_slot2_gm_addr = 0;
    uint64_t local_vec_slot3_gm_addr = 0;
    uint64_t local_mat_slot_stride_bytes = 0;
    uint64_t local_vec_slot_stride_bytes = 0;
    uint32_t local_slot_count = 2;
    uint64_t local_accum_gm_addr = 0;
    uint64_t local_out_gm_addr = 0;
    uint64_t finished_mailbox_addr = 0;
    uint32_t a_reuse_n_tiles = 1;
    uint32_t n_group_count = 0;
    uint32_t b_reuse_m_tiles = 1;
    uint32_t m_group_count = 0;
    uint32_t data_node_map_mode = 0;
    uint64_t descriptor_start_cycle = 0;
};

struct WorkerWindowDescriptor {
    uint64_t task_id = 0;
    uint64_t task_flags = 0;
    uint64_t mat_base_addr = 0;
    uint64_t vec_base_addr = 0;
    uint64_t accum_base_addr = 0;
    uint64_t completion_flag_addr = 0;
    uint64_t completion_value = 0;
    uint32_t k_begin = 0;
    uint32_t k_count = 0;
    uint32_t block_n = 0;
    uint32_t hw_input_size = 0;
    uint32_t hw_output_size = 0;
    uint32_t array_input_size = 0;
    uint32_t array_output_size = 0;
    uint32_t elem_bytes = 0;
    uint64_t mat_stride_bytes = 0;
    uint64_t vec_stride_bytes = 0;
    uint64_t local_mat_gm_addr = 0;
    uint64_t local_vec_gm_addr = 0;
    uint64_t local_accum_gm_addr = 0;
    uint64_t local_out_gm_addr = 0;
    uint64_t c_base_addr = 0;
};

class WorkerCommandProcessorAPI : public SST::SubComponent {
public:
    SST_ELI_REGISTER_SUBCOMPONENT_API(SST::Golem::WorkerCommandProcessorAPI)

    WorkerCommandProcessorAPI(ComponentId_t id, SST::Params& params) : SST::SubComponent(id) {}
    ~WorkerCommandProcessorAPI() override = default;

    virtual void bindResources(
        uint32_t coreId,
        SST::Output* output,
        SST::Golem::GlobalMemoryAPI* globalMem,
        SST::Golem::ComputeArray* array,
        SST::Golem::RequestSchedulerAPI* requestScheduler) = 0;

    virtual bool startWindow(const WorkerTaskListHeader& header) = 0;
    using GemmBufferCallback = SST::Golem::ComputeArray::BufferCallback;
    using GemmReadCallback = SST::Golem::ComputeArray::BufferReadCallback;
    using GemmArrayDoneCallback = std::function<void(uint32_t, uint64_t)>;
    using AttentionTileReadCallback =
        std::function<void(bool, uint64_t, const std::vector<uint8_t>&)>;
    virtual bool programGemmMatrixAsync(
        uint32_t arrayId, const std::vector<double>& matrix, size_t elemBytes,
        uint64_t tag, uint64_t enqueueCycle, GemmBufferCallback callback) = 0;
    virtual bool programGemmMatrixGroupAsync(
        const std::vector<uint32_t>& arrayIds,
        const std::vector<double>& matrix, size_t elemBytes,
        uint64_t tag, uint64_t enqueueCycle, GemmBufferCallback callback) = 0;
    virtual bool programGemmInputAsync(
        uint32_t arrayId, const std::vector<double>& input, size_t elemBytes,
        uint64_t tag, uint64_t enqueueCycle, GemmBufferCallback callback) = 0;
    virtual bool programGemmMatrixActiveAsync(
        uint32_t arrayId, const std::vector<double>& matrix,
        uint32_t activeColumns, size_t elemBytes, uint64_t tag,
        uint64_t enqueueCycle, GemmBufferCallback callback) = 0;
    virtual bool programGemmMatrixGroupActiveAsync(
        const std::vector<uint32_t>& arrayIds,
        const std::vector<double>& matrix, uint32_t activeColumns,
        size_t elemBytes, uint64_t tag, uint64_t enqueueCycle,
        GemmBufferCallback callback) = 0;
    virtual bool programGemmInputActiveAsync(
        uint32_t arrayId, const std::vector<double>& input,
        uint32_t activeColumns, size_t elemBytes, uint64_t tag,
        uint64_t enqueueCycle, GemmBufferCallback callback) = 0;
    virtual bool programGemmMatrixBankAsync(
        uint32_t arrayId, uint32_t operandBank,
        const std::vector<double>& matrix, size_t elemBytes,
        uint64_t tag, uint64_t enqueueCycle, GemmBufferCallback callback) {
        return operandBank == 0 && programGemmMatrixAsync(
            arrayId, matrix, elemBytes, tag, enqueueCycle, std::move(callback));
    }
    virtual bool programGemmMatrixGroupBankAsync(
        const std::vector<uint32_t>& arrayIds, uint32_t operandBank,
        const std::vector<double>& matrix, size_t elemBytes,
        uint64_t tag, uint64_t enqueueCycle, GemmBufferCallback callback) {
        return operandBank == 0 && programGemmMatrixGroupAsync(
            arrayIds, matrix, elemBytes, tag, enqueueCycle, std::move(callback));
    }
    virtual bool programGemmInputBankAsync(
        uint32_t arrayId, uint32_t operandBank,
        const std::vector<double>& input, size_t elemBytes,
        uint64_t tag, uint64_t enqueueCycle, GemmBufferCallback callback) {
        return operandBank == 0 && programGemmInputAsync(
            arrayId, input, elemBytes, tag, enqueueCycle, std::move(callback));
    }
    virtual bool programGemmMatrixGroupClassBankAsync(
        const std::vector<uint32_t>& arrayIds, uint32_t operandBank,
        const std::vector<double>& matrix, size_t elemBytes,
        AttentionClusterTrafficClass trafficClass, uint64_t tag,
        uint64_t enqueueCycle, GemmBufferCallback callback) {
        return trafficClass == AttentionClusterTrafficClass::Legacy &&
            programGemmMatrixGroupBankAsync(
                arrayIds, operandBank, matrix, elemBytes, tag, enqueueCycle,
                std::move(callback));
    }
    virtual bool programGemmInputGroupBankAsync(
        const std::vector<uint32_t>& arrayIds, uint32_t operandBank,
        const std::vector<double>& input, size_t elemBytes,
        AttentionClusterTrafficClass trafficClass, uint64_t tag,
        uint64_t enqueueCycle, GemmBufferCallback callback) {
        (void)arrayIds;
        (void)operandBank;
        (void)input;
        (void)elemBytes;
        (void)trafficClass;
        (void)tag;
        (void)enqueueCycle;
        (void)callback;
        return false;
    }
    virtual bool programGemmMatrixActiveBankAsync(
        uint32_t arrayId, uint32_t operandBank,
        const std::vector<double>& matrix, uint32_t activeColumns,
        size_t elemBytes, uint64_t tag, uint64_t enqueueCycle,
        GemmBufferCallback callback) {
        return operandBank == 0 && programGemmMatrixActiveAsync(
            arrayId, matrix, activeColumns, elemBytes, tag, enqueueCycle,
            std::move(callback));
    }
    virtual bool programGemmMatrixGroupActiveBankAsync(
        const std::vector<uint32_t>& arrayIds, uint32_t operandBank,
        const std::vector<double>& matrix, uint32_t activeColumns,
        size_t elemBytes, uint64_t tag, uint64_t enqueueCycle,
        GemmBufferCallback callback) {
        return operandBank == 0 && programGemmMatrixGroupActiveAsync(
            arrayIds, matrix, activeColumns, elemBytes, tag, enqueueCycle,
            std::move(callback));
    }
    virtual bool programGemmInputActiveBankAsync(
        uint32_t arrayId, uint32_t operandBank,
        const std::vector<double>& input, uint32_t activeColumns,
        size_t elemBytes, uint64_t tag, uint64_t enqueueCycle,
        GemmBufferCallback callback) {
        return operandBank == 0 && programGemmInputActiveAsync(
            arrayId, input, activeColumns, elemBytes, tag, enqueueCycle,
            std::move(callback));
    }
    virtual bool writeGemmOutputAsync(
        uint32_t arrayId, const std::vector<double>& output, size_t elemBytes,
        uint64_t tag, uint64_t enqueueCycle, GemmBufferCallback callback) = 0;
    virtual bool readGemmOutputAsync(
        uint32_t arrayId, size_t elemBytes, uint64_t tag, uint64_t enqueueCycle,
        GemmReadCallback callback) = 0;
    virtual bool readGemmOutputClassAsync(
        uint32_t arrayId, size_t elemBytes,
        AttentionClusterTrafficClass trafficClass, uint64_t tag,
        uint64_t enqueueCycle, GemmReadCallback callback) {
        return trafficClass == AttentionClusterTrafficClass::Legacy &&
            readGemmOutputAsync(arrayId, elemBytes, tag, enqueueCycle,
                                std::move(callback));
    }
    virtual bool readGemmOutputGroupClassAsync(
        const std::vector<uint32_t>& arrayIds, size_t elemBytes,
        AttentionClusterTrafficClass trafficClass, uint64_t tag,
        uint64_t enqueueCycle, GemmReadCallback callback) {
        (void)arrayIds; (void)elemBytes; (void)trafficClass;
        (void)tag; (void)enqueueCycle; (void)callback;
        return false;
    }
    virtual bool launchGemmArray(
        uint32_t arrayId, uint64_t outputMode, uint64_t enqueueCycle,
        GemmArrayDoneCallback callback) = 0;
    virtual bool launchGemmArrayActive(
        uint32_t arrayId, uint64_t outputMode, uint32_t activeColumns,
        uint64_t enqueueCycle, GemmArrayDoneCallback callback) = 0;
    virtual bool launchGemmArrayBank(
        uint32_t arrayId, uint32_t operandBank, uint64_t outputMode,
        uint64_t enqueueCycle, GemmArrayDoneCallback callback) {
        return operandBank == 0 && launchGemmArray(
            arrayId, outputMode, enqueueCycle, std::move(callback));
    }
    virtual bool launchGemmArrayActiveBank(
        uint32_t arrayId, uint32_t operandBank, uint64_t outputMode,
        uint32_t activeColumns, uint64_t enqueueCycle,
        GemmArrayDoneCallback callback) {
        return operandBank == 0 && launchGemmArrayActive(
            arrayId, outputMode, activeColumns, enqueueCycle,
            std::move(callback));
    }
    virtual bool launchGemmArrayGroupActiveBank(
        const std::vector<uint32_t>& arrayIds, uint32_t operandBank,
        uint64_t outputMode, uint32_t activeColumns, uint64_t enqueueCycle,
        GemmArrayDoneCallback callback) {
        (void)arrayIds; (void)operandBank; (void)outputMode;
        (void)activeColumns; (void)enqueueCycle; (void)callback;
        return false;
    }
    virtual bool beginAttentionTileStorage(
        uint32_t rows, uint32_t columns, size_t elemBytes,
        uint64_t generation) = 0;
    virtual bool beginAttentionStorageSession(
        uint64_t qkScratchBytes, uint32_t accumulatorRows,
        size_t accumulatorRowBytes, uint64_t generation) = 0;
    virtual bool writeAttentionTileColumnAsync(
        uint32_t column, const std::vector<uint8_t>& values,
        uint64_t generation, uint64_t tag, uint64_t enqueueCycle,
        GemmBufferCallback callback) = 0;
    virtual bool readAttentionTileRowAsync(
        uint32_t row, uint64_t generation, uint64_t tag,
        uint64_t enqueueCycle, AttentionTileReadCallback callback) = 0;
    virtual bool writeAttentionAccumulatorRowAsync(
        uint32_t row, const std::vector<uint8_t>& values,
        uint64_t generation, uint64_t tag, uint64_t enqueueCycle,
        GemmBufferCallback callback) = 0;
    virtual bool readAttentionAccumulatorRowAsync(
        uint32_t row, uint64_t generation, uint64_t tag,
        uint64_t enqueueCycle, AttentionTileReadCallback callback) = 0;
    virtual bool endAttentionTileStorage(uint64_t generation) = 0;
    virtual bool endAttentionStorageSession(uint64_t generation) = 0;
    virtual bool cancelAttentionTileStorage(uint64_t generation) = 0;
    virtual bool isBusy() const = 0;
    virtual bool tick(uint64_t cycle) = 0;
    virtual bool handleArrayDone(uint32_t arrayId, uint64_t cycle) = 0;
};

class WorkerCommandProcessorLocal : public WorkerCommandProcessorAPI {
public:
    SST_ELI_REGISTER_SUBCOMPONENT(
        WorkerCommandProcessorLocal,
        "golem",
        "WorkerCommandProcessorLocal",
        SST_ELI_ELEMENT_VERSION(1, 0, 0),
        "Minimal local worker command processor prototype",
        SST::Golem::WorkerCommandProcessorAPI)

    SST_ELI_DOCUMENT_PARAMS(
        {"verbose", "Verbosity", "0"},
        {"dtype_is_float", "Output vector stores float elements", "0"},
        {"stage3_trace", "Enable Stage3 2D window trace", "0"},
        {"attention_cluster_qk_arrays", "First PV array id in the 64-array Attention partition", "16"},
        {"prefetch_windows", "Number of 2D K-windows to prefetch ahead of the active window", "1"},
        {"cross_macro_prefetch", "Prefetch the next macro task's first K-window while the current final window computes", "0"},
        {"window_k_tiles", "WCP K-tiles per scheduler transaction", "4"},
        {"c_buffer_bytes", "Private partial-C SRAM capacity in bytes", "0"},
        {"c_buffer_read_bytes_per_cycle", "Independent partial-C SRAM read bandwidth", "256"},
        {"c_buffer_write_bytes_per_cycle", "Independent partial-C SRAM write bandwidth", "256"},
        {"c_buffer_latency_cycles", "Fixed partial-C SRAM access latency", "1"},
        {"attention_tile_storage_banks", "Banks exposed in Attention tile-storage mode (1 through 16)", "16"},
        {"attention_tile_storage_bank_bytes_per_cycle", "Per-bank bandwidth in Attention tile-storage mode", "64"},
        {"gemm_proxy_queue_depth", "Maximum queued generic GEMM control commands", "32"},
        {"gemm_proxy_issue_width", "Generic GEMM control commands issued per WCP cycle", "1"},
        {"gemm_proxy_command_latency_cycles", "Minimum WCP control latency before a generic GEMM command can issue", "1"},
        {"gemm_proxy_completion_latency_cycles", "WCP latency from array completion to generic GEMM client callback", "1"},
        {"final_c_write_enable", "Legacy switch for final C DMA writes", "1"},
        {"output_mode", "Final C destination: hbm or fusion", "hbm"},
        {"fusion_dump_enable", "Dump fusion-consumed final C tiles for verification", "0"},
        {"fusion_dump_dir", "Directory for per-core fusion C records", ""})

    SST_ELI_DOCUMENT_STATISTICS(
        {"gemm_proxy_commands_issued", "Generic GEMM control commands issued by WCP", "commands", 1},
        {"gemm_proxy_queue_full_stalls", "Generic GEMM commands rejected because the WCP queue was full", "stalls", 1},
        {"gemm_proxy_queue_wait_cycles", "Aggregate command residence time in the generic GEMM WCP queue", "cycles", 1},
        {"gemm_proxy_launch_commands", "Generic GEMM array launch commands issued by WCP", "commands", 1},
        {"gemm_proxy_completion_callbacks", "Generic GEMM array completions delivered by WCP", "callbacks", 1},
        {"gemm_proxy_completion_delay_cycles", "Aggregate modeled array-completion callback delay", "cycles", 1},
        {"attention_tile_storage_acquires", "Successful Attention tile-storage mode acquisitions", "acquires", 1},
        {"attention_tile_storage_releases", "Successful Attention tile-storage mode releases", "releases", 1},
        {"attention_tile_storage_mode_conflicts", "Attention tile-storage acquisitions rejected by active C-buffer ownership", "conflicts", 1},
        {"attention_tile_storage_capacity_rejections", "Attention tile-storage acquisitions rejected by finite SRAM resources", "rejections", 1},
        {"attention_tile_storage_column_writes", "QK output columns transposed into Attention tile storage", "writes", 1},
        {"attention_tile_storage_row_reads", "Query-major rows read from Attention tile storage", "reads", 1},
        {"attention_tile_storage_write_bytes", "Bytes written into Attention tile storage", "bytes", 1},
        {"attention_tile_storage_read_bytes", "Bytes read from Attention tile storage", "bytes", 1},
        {"attention_tile_storage_write_wait_cycles", "Cycles waiting for the C-buffer write port or banks", "cycles", 1},
        {"attention_tile_storage_read_wait_cycles", "Cycles waiting for C-buffer RAW, read port, or bank readiness", "cycles", 1},
        {"attention_storage_session_acquires", "Persistent Attention C-buffer sessions acquired", "acquires", 1},
        {"attention_storage_session_releases", "Persistent Attention C-buffer sessions released", "releases", 1},
        {"attention_accumulator_row_writes", "FP32 Attention O rows written to C-buffer", "writes", 1},
        {"attention_accumulator_row_reads", "FP32 Attention O rows read from C-buffer", "reads", 1},
        {"attention_accumulator_write_bytes", "FP32 Attention O bytes written to C-buffer", "bytes", 1},
        {"attention_accumulator_read_bytes", "FP32 Attention O bytes read from C-buffer", "bytes", 1},
        {"attention_accumulator_write_wait_cycles", "Cycles waiting for O C-buffer write ports or banks", "cycles", 1},
        {"attention_accumulator_read_wait_cycles", "Cycles waiting for O C-buffer RAW, read ports, or banks", "cycles", 1})

    WorkerCommandProcessorLocal(ComponentId_t id, SST::Params& params)
        : WorkerCommandProcessorAPI(id, params),
          verbose_(params.find<int>("verbose", 0)),
          outputIsFloat_(params.find<int>("dtype_is_float", 0) != 0),
          stage3Trace_(params.find<int>("stage3_trace", 0) != 0),
          attentionClusterQkArrays_(params.find<uint32_t>(
              "attention_cluster_qk_arrays", 16)),
          prefetchWindowDepth_(static_cast<uint32_t>(std::max(1, params.find<int>("prefetch_windows", 1)))),
          crossMacroPrefetch_(params.find<int>("cross_macro_prefetch", 0) != 0),
          windowKtiles_(std::max(1, params.find<int>("window_k_tiles", 4))),
          cBufferBytes_(params.find<uint64_t>("c_buffer_bytes", 0)),
          cBufferReadBytesPerCycle_(params.find<uint64_t>("c_buffer_read_bytes_per_cycle", 256)),
          cBufferWriteBytesPerCycle_(params.find<uint64_t>("c_buffer_write_bytes_per_cycle", 256)),
          cBufferLatencyCycles_(params.find<uint64_t>("c_buffer_latency_cycles", 1)),
          attentionTileStorageBanks_(params.find<uint32_t>(
              "attention_tile_storage_banks", 16)),
          attentionTileStorageBankBytesPerCycle_(params.find<uint64_t>(
              "attention_tile_storage_bank_bytes_per_cycle", 64)),
          gemmProxyQueueDepth_(static_cast<uint32_t>(std::max(1, params.find<int>("gemm_proxy_queue_depth", 32)))),
          gemmProxyIssueWidth_(static_cast<uint32_t>(std::max(1, params.find<int>("gemm_proxy_issue_width", 1)))),
          gemmProxyCommandLatencyCycles_(params.find<uint64_t>("gemm_proxy_command_latency_cycles", 1)),
          gemmProxyCompletionLatencyCycles_(params.find<uint64_t>("gemm_proxy_completion_latency_cycles", 1)),
          finalCWriteEnable_(params.find<int>("final_c_write_enable", 1) != 0),
          outputMode_(params.find<std::string>("output_mode", finalCWriteEnable_ ? "hbm" : "fusion")),
          fusionDumpEnable_(params.find<int>("fusion_dump_enable", 0) != 0),
          fusionDumpDir_(params.find<std::string>("fusion_dump_dir", "")),
          cBufferStorage_(cBufferBytes_, 0),
          output_("WorkerCommandProcessor[@p:@l]: ", verbose_, 0, SST::Output::STDOUT) {
        if (attentionClusterQkArrays_ == 0 ||
            attentionClusterQkArrays_ >= 64 ||
            (attentionClusterQkArrays_ % 2) != 0) {
            output_.fatal(
                CALL_INFO, -1,
                "attention_cluster_qk_arrays must be even and in range [2, 62]\n");
        }
        if (gemmProxyCommandLatencyCycles_ == 0 ||
            gemmProxyCompletionLatencyCycles_ == 0) {
            output_.fatal(
                CALL_INFO, -1,
                "gemm_proxy command and completion latency must be positive\n");
        }
        if (attentionTileStorageBanks_ == 0 || attentionTileStorageBanks_ > 16) {
            output_.fatal(
                CALL_INFO, -1,
                "attention_tile_storage_banks must be in range [1, 16]\n");
        }
        finalCWriteEnable_ = outputMode_ == "hbm";
        statGemmProxyCommandsIssued_ = registerStatistic<uint64_t>("gemm_proxy_commands_issued");
        statGemmProxyQueueFullStalls_ = registerStatistic<uint64_t>("gemm_proxy_queue_full_stalls");
        statGemmProxyQueueWaitCycles_ = registerStatistic<uint64_t>("gemm_proxy_queue_wait_cycles");
        statGemmProxyLaunchCommands_ = registerStatistic<uint64_t>("gemm_proxy_launch_commands");
        statGemmProxyCompletionCallbacks_ = registerStatistic<uint64_t>("gemm_proxy_completion_callbacks");
        statGemmProxyCompletionDelayCycles_ = registerStatistic<uint64_t>("gemm_proxy_completion_delay_cycles");
        statAttentionTileStorageAcquires_ = registerStatistic<uint64_t>("attention_tile_storage_acquires");
        statAttentionTileStorageReleases_ = registerStatistic<uint64_t>("attention_tile_storage_releases");
        statAttentionTileStorageModeConflicts_ = registerStatistic<uint64_t>("attention_tile_storage_mode_conflicts");
        statAttentionTileStorageCapacityRejections_ = registerStatistic<uint64_t>("attention_tile_storage_capacity_rejections");
        statAttentionTileStorageColumnWrites_ = registerStatistic<uint64_t>("attention_tile_storage_column_writes");
        statAttentionTileStorageRowReads_ = registerStatistic<uint64_t>("attention_tile_storage_row_reads");
        statAttentionTileStorageWriteBytes_ = registerStatistic<uint64_t>("attention_tile_storage_write_bytes");
        statAttentionTileStorageReadBytes_ = registerStatistic<uint64_t>("attention_tile_storage_read_bytes");
        statAttentionTileStorageWriteWaitCycles_ = registerStatistic<uint64_t>("attention_tile_storage_write_wait_cycles");
        statAttentionTileStorageReadWaitCycles_ = registerStatistic<uint64_t>("attention_tile_storage_read_wait_cycles");
        statAttentionStorageSessionAcquires_ = registerStatistic<uint64_t>("attention_storage_session_acquires");
        statAttentionStorageSessionReleases_ = registerStatistic<uint64_t>("attention_storage_session_releases");
        statAttentionAccumulatorRowWrites_ = registerStatistic<uint64_t>("attention_accumulator_row_writes");
        statAttentionAccumulatorRowReads_ = registerStatistic<uint64_t>("attention_accumulator_row_reads");
        statAttentionAccumulatorWriteBytes_ = registerStatistic<uint64_t>("attention_accumulator_write_bytes");
        statAttentionAccumulatorReadBytes_ = registerStatistic<uint64_t>("attention_accumulator_read_bytes");
        statAttentionAccumulatorWriteWaitCycles_ = registerStatistic<uint64_t>("attention_accumulator_write_wait_cycles");
        statAttentionAccumulatorReadWaitCycles_ = registerStatistic<uint64_t>("attention_accumulator_read_wait_cycles");
    }

    void bindResources(
        uint32_t coreId,
        SST::Output* output,
        SST::Golem::GlobalMemoryAPI* globalMem,
        SST::Golem::ComputeArray* array,
        SST::Golem::RequestSchedulerAPI* requestScheduler) override {
        coreId_ = coreId;
        extOutput_ = output;
        globalMem_ = globalMem;
        array_ = array;
        requestScheduler_ = requestScheduler;
        if (outputMode_ == "fusion" && fusionDumpEnable_) {
            const std::string separator =
                (!fusionDumpDir_.empty() && fusionDumpDir_.back() == '/') ? "" : "/";
            const std::string path = fusionDumpDir_ + separator +
                "fusion_c_core" + std::to_string(coreId_) + ".bin";
            fusionDump_.open(path, std::ios::binary | std::ios::trunc);
            if (!fusionDump_.is_open() && extOutput_ != nullptr) {
                extOutput_->output(
                    "[Core %u] [wcp] ERROR: cannot open fusion C dump: %s\n",
                    coreId_, path.c_str());
            }
        }
    }

    bool programGemmMatrixAsync(
        uint32_t arrayId, const std::vector<double>& matrix, size_t elemBytes,
        uint64_t tag, uint64_t enqueueCycle, GemmBufferCallback callback) override {
        GemmProxyCommand command;
        command.kind = GemmProxyCommandKind::PROGRAM_MATRIX;
        command.arrayId = arrayId;
        command.payload = matrix;
        command.elemBytes = elemBytes;
        command.tag = tag;
        command.enqueueCycle = enqueueCycle;
        command.bufferCallback = std::move(callback);
        return enqueueGemmProxyCommand(std::move(command));
    }

    bool programGemmMatrixGroupAsync(
        const std::vector<uint32_t>& arrayIds,
        const std::vector<double>& matrix, size_t elemBytes,
        uint64_t tag, uint64_t enqueueCycle, GemmBufferCallback callback) override {
        // Permanent request errors must not enter the retry-on-backpressure queue.
        if (array_ == nullptr || !array_->validateMatrixBroadcastRequest(
                arrayIds, matrix.size(), elemBytes)) {
            return false;
        }
        GemmProxyCommand command;
        command.kind = GemmProxyCommandKind::PROGRAM_MATRIX_GROUP;
        command.arrayIds = arrayIds;
        command.payload = matrix;
        command.elemBytes = elemBytes;
        command.tag = tag;
        command.enqueueCycle = enqueueCycle;
        command.bufferCallback = std::move(callback);
        return enqueueGemmProxyCommand(std::move(command));
    }

    bool programGemmInputAsync(
        uint32_t arrayId, const std::vector<double>& input, size_t elemBytes,
        uint64_t tag, uint64_t enqueueCycle, GemmBufferCallback callback) override {
        GemmProxyCommand command;
        command.kind = GemmProxyCommandKind::PROGRAM_INPUT;
        command.arrayId = arrayId;
        command.payload = input;
        command.elemBytes = elemBytes;
        command.tag = tag;
        command.enqueueCycle = enqueueCycle;
        command.bufferCallback = std::move(callback);
        return enqueueGemmProxyCommand(std::move(command));
    }

    bool programGemmMatrixActiveAsync(
        uint32_t arrayId, const std::vector<double>& matrix,
        uint32_t activeColumns, size_t elemBytes, uint64_t tag,
        uint64_t enqueueCycle, GemmBufferCallback callback) override {
        if (array_ == nullptr || !array_->validateActiveMatrixRequest(
                arrayId, matrix.size(), activeColumns, elemBytes)) {
            return false;
        }
        GemmProxyCommand command;
        command.kind = GemmProxyCommandKind::PROGRAM_MATRIX;
        command.arrayId = arrayId;
        command.payload = matrix;
        command.activeColumns = activeColumns;
        command.elemBytes = elemBytes;
        command.tag = tag;
        command.enqueueCycle = enqueueCycle;
        command.bufferCallback = std::move(callback);
        return enqueueGemmProxyCommand(std::move(command));
    }

    bool programGemmMatrixGroupActiveAsync(
        const std::vector<uint32_t>& arrayIds,
        const std::vector<double>& matrix, uint32_t activeColumns,
        size_t elemBytes, uint64_t tag, uint64_t enqueueCycle,
        GemmBufferCallback callback) override {
        if (array_ == nullptr || !array_->validateMatrixBroadcastRequest(
                arrayIds, matrix.size(), elemBytes, activeColumns)) {
            return false;
        }
        GemmProxyCommand command;
        command.kind = GemmProxyCommandKind::PROGRAM_MATRIX_GROUP;
        command.arrayIds = arrayIds;
        command.payload = matrix;
        command.activeColumns = activeColumns;
        command.elemBytes = elemBytes;
        command.tag = tag;
        command.enqueueCycle = enqueueCycle;
        command.bufferCallback = std::move(callback);
        return enqueueGemmProxyCommand(std::move(command));
    }

    bool programGemmInputActiveAsync(
        uint32_t arrayId, const std::vector<double>& input,
        uint32_t activeColumns, size_t elemBytes, uint64_t tag,
        uint64_t enqueueCycle, GemmBufferCallback callback) override {
        if (array_ == nullptr || !array_->validateActiveInputRequest(
                arrayId, input.size(), activeColumns, elemBytes)) {
            return false;
        }
        GemmProxyCommand command;
        command.kind = GemmProxyCommandKind::PROGRAM_INPUT;
        command.arrayId = arrayId;
        command.payload = input;
        command.activeColumns = activeColumns;
        command.elemBytes = elemBytes;
        command.tag = tag;
        command.enqueueCycle = enqueueCycle;
        command.bufferCallback = std::move(callback);
        return enqueueGemmProxyCommand(std::move(command));
    }

    bool programGemmMatrixBankAsync(
        uint32_t arrayId, uint32_t operandBank,
        const std::vector<double>& matrix, size_t elemBytes,
        uint64_t tag, uint64_t enqueueCycle,
        GemmBufferCallback callback) override {
        GemmProxyCommand command;
        command.kind = GemmProxyCommandKind::PROGRAM_MATRIX;
        command.arrayId = arrayId;
        command.operandBank = operandBank;
        command.payload = matrix;
        command.elemBytes = elemBytes;
        command.tag = tag;
        command.enqueueCycle = enqueueCycle;
        command.bufferCallback = std::move(callback);
        return enqueueGemmProxyCommand(std::move(command));
    }

    bool programGemmMatrixGroupBankAsync(
        const std::vector<uint32_t>& arrayIds, uint32_t operandBank,
        const std::vector<double>& matrix, size_t elemBytes,
        uint64_t tag, uint64_t enqueueCycle,
        GemmBufferCallback callback) override {
        if (array_ == nullptr || arrayIds.empty() ||
            !array_->validateOperandContextRequest(arrayIds.front(), operandBank) ||
            !array_->validateMatrixBroadcastRequest(
                arrayIds, matrix.size(), elemBytes)) return false;
        GemmProxyCommand command;
        command.kind = GemmProxyCommandKind::PROGRAM_MATRIX_GROUP;
        command.arrayIds = arrayIds;
        command.operandBank = operandBank;
        command.payload = matrix;
        command.elemBytes = elemBytes;
        command.tag = tag;
        command.enqueueCycle = enqueueCycle;
        command.bufferCallback = std::move(callback);
        return enqueueGemmProxyCommand(std::move(command));
    }

    bool programGemmInputBankAsync(
        uint32_t arrayId, uint32_t operandBank,
        const std::vector<double>& input, size_t elemBytes,
        uint64_t tag, uint64_t enqueueCycle,
        GemmBufferCallback callback) override {
        GemmProxyCommand command;
        command.kind = GemmProxyCommandKind::PROGRAM_INPUT;
        command.arrayId = arrayId;
        command.operandBank = operandBank;
        command.payload = input;
        command.elemBytes = elemBytes;
        command.tag = tag;
        command.enqueueCycle = enqueueCycle;
        command.bufferCallback = std::move(callback);
        return enqueueGemmProxyCommand(std::move(command));
    }

    bool programGemmMatrixGroupClassBankAsync(
        const std::vector<uint32_t>& arrayIds, uint32_t operandBank,
        const std::vector<double>& matrix, size_t elemBytes,
        AttentionClusterTrafficClass trafficClass, uint64_t tag,
        uint64_t enqueueCycle, GemmBufferCallback callback) override {
        if (array_ == nullptr || arrayIds.empty() ||
            !array_->validateOperandContextRequest(arrayIds.front(), operandBank) ||
            trafficClass == AttentionClusterTrafficClass::Legacy ||
            !array_->validateMatrixBroadcastRequest(
                arrayIds, matrix.size(), elemBytes)) return false;
        GemmProxyCommand command;
        command.kind = GemmProxyCommandKind::PROGRAM_MATRIX_GROUP;
        command.arrayIds = arrayIds;
        command.operandBank = operandBank;
        command.payload = matrix;
        command.elemBytes = elemBytes;
        command.trafficClass = trafficClass;
        command.tag = tag;
        command.enqueueCycle = enqueueCycle;
        command.bufferCallback = std::move(callback);
        return enqueueGemmProxyCommand(std::move(command));
    }

    bool programGemmInputGroupBankAsync(
        const std::vector<uint32_t>& arrayIds, uint32_t operandBank,
        const std::vector<double>& input, size_t elemBytes,
        AttentionClusterTrafficClass trafficClass, uint64_t tag,
        uint64_t enqueueCycle, GemmBufferCallback callback) override {
        const bool validPvTopology =
            trafficClass != AttentionClusterTrafficClass::PvPInput ||
            (arrayIds.size() == 2 &&
             (input.size() == 32 || input.size() == 64) &&
             arrayIds[0] >= attentionClusterQkArrays_ &&
             arrayIds[1] == arrayIds[0] + 1 &&
             (arrayIds[0] % 2) == 0 && arrayIds[1] < 64);
        const bool validQkTopology =
            trafficClass != AttentionClusterTrafficClass::QkQPair ||
            (arrayIds.size() == 1 &&
             arrayIds.front() < attentionClusterQkArrays_ &&
             input.size() == 64);
        if (array_ == nullptr || arrayIds.empty() ||
            !array_->validateOperandContextRequest(arrayIds.front(), operandBank) ||
            (trafficClass != AttentionClusterTrafficClass::QkQPair &&
             trafficClass != AttentionClusterTrafficClass::PvPInput) ||
            !validQkTopology ||
            !validPvTopology ||
            !array_->validateInputMulticastRequest(
                arrayIds, input.size(), elemBytes)) return false;
        GemmProxyCommand command;
        command.kind = GemmProxyCommandKind::PROGRAM_INPUT;
        command.arrayIds = arrayIds;
        command.operandBank = operandBank;
        command.payload = input;
        command.elemBytes = elemBytes;
        command.trafficClass = trafficClass;
        command.tag = tag;
        command.enqueueCycle = enqueueCycle;
        command.bufferCallback = std::move(callback);
        return enqueueGemmProxyCommand(std::move(command));
    }

    bool programGemmMatrixActiveBankAsync(
        uint32_t arrayId, uint32_t operandBank,
        const std::vector<double>& matrix, uint32_t activeColumns,
        size_t elemBytes, uint64_t tag, uint64_t enqueueCycle,
        GemmBufferCallback callback) override {
        if (array_ == nullptr || !array_->validateActiveMatrixRequest(
                arrayId, matrix.size(), activeColumns, elemBytes)) return false;
        GemmProxyCommand command;
        command.kind = GemmProxyCommandKind::PROGRAM_MATRIX;
        command.arrayId = arrayId;
        command.operandBank = operandBank;
        command.payload = matrix;
        command.activeColumns = activeColumns;
        command.elemBytes = elemBytes;
        command.tag = tag;
        command.enqueueCycle = enqueueCycle;
        command.bufferCallback = std::move(callback);
        return enqueueGemmProxyCommand(std::move(command));
    }

    bool programGemmMatrixGroupActiveBankAsync(
        const std::vector<uint32_t>& arrayIds, uint32_t operandBank,
        const std::vector<double>& matrix, uint32_t activeColumns,
        size_t elemBytes, uint64_t tag, uint64_t enqueueCycle,
        GemmBufferCallback callback) override {
        if (array_ == nullptr || arrayIds.empty() ||
            !array_->validateOperandContextRequest(arrayIds.front(), operandBank) ||
            !array_->validateMatrixBroadcastRequest(
                arrayIds, matrix.size(), elemBytes, activeColumns)) return false;
        GemmProxyCommand command;
        command.kind = GemmProxyCommandKind::PROGRAM_MATRIX_GROUP;
        command.arrayIds = arrayIds;
        command.operandBank = operandBank;
        command.payload = matrix;
        command.activeColumns = activeColumns;
        command.elemBytes = elemBytes;
        command.tag = tag;
        command.enqueueCycle = enqueueCycle;
        command.bufferCallback = std::move(callback);
        return enqueueGemmProxyCommand(std::move(command));
    }

    bool programGemmInputActiveBankAsync(
        uint32_t arrayId, uint32_t operandBank,
        const std::vector<double>& input, uint32_t activeColumns,
        size_t elemBytes, uint64_t tag, uint64_t enqueueCycle,
        GemmBufferCallback callback) override {
        if (array_ == nullptr || !array_->validateActiveInputRequest(
                arrayId, input.size(), activeColumns, elemBytes)) return false;
        GemmProxyCommand command;
        command.kind = GemmProxyCommandKind::PROGRAM_INPUT;
        command.arrayId = arrayId;
        command.operandBank = operandBank;
        command.payload = input;
        command.activeColumns = activeColumns;
        command.elemBytes = elemBytes;
        command.tag = tag;
        command.enqueueCycle = enqueueCycle;
        command.bufferCallback = std::move(callback);
        return enqueueGemmProxyCommand(std::move(command));
    }

    bool writeGemmOutputAsync(
        uint32_t arrayId, const std::vector<double>& output, size_t elemBytes,
        uint64_t tag, uint64_t enqueueCycle, GemmBufferCallback callback) override {
        GemmProxyCommand command;
        command.kind = GemmProxyCommandKind::WRITE_OUTPUT;
        command.arrayId = arrayId;
        command.payload = output;
        command.elemBytes = elemBytes;
        command.tag = tag;
        command.enqueueCycle = enqueueCycle;
        command.bufferCallback = std::move(callback);
        return enqueueGemmProxyCommand(std::move(command));
    }

    bool readGemmOutputAsync(
        uint32_t arrayId, size_t elemBytes, uint64_t tag, uint64_t enqueueCycle,
        GemmReadCallback callback) override {
        GemmProxyCommand command;
        command.kind = GemmProxyCommandKind::READ_OUTPUT;
        command.arrayId = arrayId;
        command.elemBytes = elemBytes;
        command.tag = tag;
        command.enqueueCycle = enqueueCycle;
        command.readCallback = std::move(callback);
        return enqueueGemmProxyCommand(std::move(command));
    }

    bool readGemmOutputClassAsync(
        uint32_t arrayId, size_t elemBytes,
        AttentionClusterTrafficClass trafficClass, uint64_t tag,
        uint64_t enqueueCycle, GemmReadCallback callback) override {
        if (trafficClass != AttentionClusterTrafficClass::QkScoreOut) return false;
        GemmProxyCommand command;
        command.kind = GemmProxyCommandKind::READ_OUTPUT;
        command.arrayId = arrayId;
        command.elemBytes = elemBytes;
        command.trafficClass = trafficClass;
        command.tag = tag;
        command.enqueueCycle = enqueueCycle;
        command.readCallback = std::move(callback);
        return enqueueGemmProxyCommand(std::move(command));
    }

    bool readGemmOutputGroupClassAsync(
        const std::vector<uint32_t>& arrayIds, size_t elemBytes,
        AttentionClusterTrafficClass trafficClass, uint64_t tag,
        uint64_t enqueueCycle, GemmReadCallback callback) override {
        if (array_ == nullptr || !array_->validateOutputGroupRequest(
                arrayIds, elemBytes, trafficClass)) return false;
        GemmProxyCommand command;
        command.kind = GemmProxyCommandKind::READ_OUTPUT_GROUP;
        command.arrayIds = arrayIds;
        command.elemBytes = elemBytes;
        command.trafficClass = trafficClass;
        command.tag = tag;
        command.enqueueCycle = enqueueCycle;
        command.readCallback = std::move(callback);
        return enqueueGemmProxyCommand(std::move(command));
    }

    bool launchGemmArray(
        uint32_t arrayId, uint64_t outputMode, uint64_t enqueueCycle,
        GemmArrayDoneCallback callback) override {
        if (array_ == nullptr || busy_ || !callback ||
            gemmArrayDoneCallbacks_.find(arrayId) != gemmArrayDoneCallbacks_.end()) {
            return false;
        }
        if (gemmProxyCommands_.size() >= gemmProxyQueueDepth_) {
            statGemmProxyQueueFullStalls_->addData(1);
            return false;
        }
        gemmArrayDoneCallbacks_.emplace(arrayId, std::move(callback));
        GemmProxyCommand command;
        command.kind = GemmProxyCommandKind::LAUNCH;
        command.arrayId = arrayId;
        command.outputMode = outputMode;
        command.enqueueCycle = enqueueCycle;
        gemmProxyCommands_.push_back(std::move(command));
        return true;
    }

    bool launchGemmArrayActive(
        uint32_t arrayId, uint64_t outputMode, uint32_t activeColumns,
        uint64_t enqueueCycle, GemmArrayDoneCallback callback) override {
        if (array_ == nullptr ||
            !array_->validateActiveLaunchRequest(arrayId, activeColumns) ||
            busy_ || !callback ||
            gemmArrayDoneCallbacks_.find(arrayId) != gemmArrayDoneCallbacks_.end()) {
            return false;
        }
        if (gemmProxyCommands_.size() >= gemmProxyQueueDepth_) {
            statGemmProxyQueueFullStalls_->addData(1);
            return false;
        }
        gemmArrayDoneCallbacks_.emplace(arrayId, std::move(callback));
        GemmProxyCommand command;
        command.kind = GemmProxyCommandKind::LAUNCH;
        command.arrayId = arrayId;
        command.outputMode = outputMode;
        command.activeColumns = activeColumns;
        command.enqueueCycle = enqueueCycle;
        gemmProxyCommands_.push_back(std::move(command));
        return true;
    }

    bool launchGemmArrayBank(
        uint32_t arrayId, uint32_t operandBank, uint64_t outputMode,
        uint64_t enqueueCycle, GemmArrayDoneCallback callback) override {
        return launchGemmArrayBankImpl(
            arrayId, operandBank, outputMode, 0, enqueueCycle,
            std::move(callback));
    }

    bool launchGemmArrayActiveBank(
        uint32_t arrayId, uint32_t operandBank, uint64_t outputMode,
        uint32_t activeColumns, uint64_t enqueueCycle,
        GemmArrayDoneCallback callback) override {
        if (array_ == nullptr ||
            !array_->validateActiveLaunchRequest(arrayId, activeColumns)) {
            return false;
        }
        return launchGemmArrayBankImpl(
            arrayId, operandBank, outputMode, activeColumns, enqueueCycle,
            std::move(callback));
    }

    bool launchGemmArrayGroupActiveBank(
        const std::vector<uint32_t>& arrayIds, uint32_t operandBank,
        uint64_t outputMode, uint32_t activeColumns, uint64_t enqueueCycle,
        GemmArrayDoneCallback callback) override {
        if (array_ == nullptr || busy_ || !callback || arrayIds.empty() ||
            gemmProxyCommands_.size() >= gemmProxyQueueDepth_) {
            if (gemmProxyCommands_.size() >= gemmProxyQueueDepth_)
                statGemmProxyQueueFullStalls_->addData(1);
            return false;
        }
        std::unordered_set<uint32_t> unique;
        for (uint32_t arrayId : arrayIds) {
            if (!unique.insert(arrayId).second ||
                gemmArrayDoneCallbacks_.count(arrayId) != 0 ||
                !array_->validateOperandContextRequest(arrayId, operandBank) ||
                (activeColumns != 0 &&
                 !array_->validateActiveLaunchRequest(arrayId, activeColumns))) {
                return false;
            }
        }
        for (uint32_t arrayId : arrayIds)
            gemmArrayDoneCallbacks_.emplace(arrayId, callback);
        GemmProxyCommand command;
        command.kind = GemmProxyCommandKind::LAUNCH_GROUP;
        command.arrayIds = arrayIds;
        command.operandBank = operandBank;
        command.outputMode = outputMode;
        command.activeColumns = activeColumns;
        command.enqueueCycle = enqueueCycle;
        gemmProxyCommands_.push_back(std::move(command));
        return true;
    }

    bool beginAttentionTileStorage(
        uint32_t rows, uint32_t columns, size_t elemBytes,
        uint64_t generation) override {
        const bool ownsSession = attentionStorageSessionActive_ &&
            cBufferMode_ == CBufferMode::ATTENTION_TILE_STORAGE &&
            generation == attentionStorageSessionGeneration_;
        if ((!ownsSession && cBufferMode_ != CBufferMode::FREE) ||
            attentionTileStorageActive_ || busy_ ||
            !gemmProxyCommands_.empty() || !gemmArrayDoneCallbacks_.empty() ||
            !pendingGemmCompletions_.empty() ||
            !pendingAttentionStorageCompletions_.empty()) {
            statAttentionTileStorageModeConflicts_->addData(1);
            return false;
        }
        const bool sizeOverflow = rows != 0 && columns > UINT64_MAX / rows;
        const uint64_t elements = sizeOverflow ? UINT64_MAX :
            static_cast<uint64_t>(rows) * columns;
        const bool byteOverflow = elemBytes != 0 && elements > UINT64_MAX / elemBytes;
        const uint64_t requiredBytes = byteOverflow ? UINT64_MAX : elements * elemBytes;
        if (rows == 0 || columns == 0 || elemBytes == 0 || sizeOverflow ||
            byteOverflow || requiredBytes >
                (ownsSession ? attentionStorageQkScratchBytes_ : cBufferBytes_) ||
            cBufferReadBytesPerCycle_ == 0 || cBufferWriteBytesPerCycle_ == 0 ||
            attentionTileStorageBankBytesPerCycle_ == 0) {
            statAttentionTileStorageCapacityRejections_->addData(1);
            return false;
        }
        cBufferMode_ = CBufferMode::ATTENTION_TILE_STORAGE;
        attentionTileStorageActive_ = true;
        attentionTileStorageRows_ = rows;
        attentionTileStorageColumns_ = columns;
        attentionTileStorageElemBytes_ = elemBytes;
        attentionTileStorageGeneration_ = generation;
        attentionTileStorageColumnValid_.assign(columns, 0);
        attentionTileStorageRowRead_.assign(rows, 0);
        attentionTileStorageRowWriteReadyCycle_.assign(rows, 0);
        if (!ownsSession) {
            attentionTileStorageBankNextReadCycle_.assign(attentionTileStorageBanks_, 0);
            attentionTileStorageBankNextWriteCycle_.assign(attentionTileStorageBanks_, 0);
            cBufferWriteReadyCycles_.clear();
        }
        statAttentionTileStorageAcquires_->addData(1);
        return true;
    }

    bool beginAttentionStorageSession(
        uint64_t qkScratchBytes, uint32_t accumulatorRows,
        size_t accumulatorRowBytes, uint64_t generation) override {
        const bool sizeOverflow = accumulatorRows != 0 &&
            accumulatorRowBytes > UINT64_MAX / accumulatorRows;
        const uint64_t accumulatorBytes = sizeOverflow ? UINT64_MAX :
            static_cast<uint64_t>(accumulatorRows) * accumulatorRowBytes;
        const bool totalOverflow = qkScratchBytes > UINT64_MAX - accumulatorBytes;
        const uint64_t requiredBytes = totalOverflow ? UINT64_MAX :
            qkScratchBytes + accumulatorBytes;
        if (cBufferMode_ != CBufferMode::FREE || attentionTileStorageActive_ ||
            attentionStorageSessionActive_ || busy_ ||
            !gemmProxyCommands_.empty() || !gemmArrayDoneCallbacks_.empty() ||
            !pendingGemmCompletions_.empty() ||
            !pendingAttentionStorageCompletions_.empty()) {
            statAttentionTileStorageModeConflicts_->addData(1);
            return false;
        }
        if (qkScratchBytes == 0 || accumulatorRows == 0 ||
            accumulatorRowBytes == 0 || sizeOverflow || totalOverflow ||
            requiredBytes > cBufferBytes_ || cBufferReadBytesPerCycle_ == 0 ||
            cBufferWriteBytesPerCycle_ == 0 ||
            attentionTileStorageBankBytesPerCycle_ == 0) {
            statAttentionTileStorageCapacityRejections_->addData(1);
            return false;
        }
        cBufferMode_ = CBufferMode::ATTENTION_TILE_STORAGE;
        attentionStorageSessionActive_ = true;
        attentionStorageSessionGeneration_ = generation;
        attentionStorageQkScratchBytes_ = qkScratchBytes;
        attentionAccumulatorOffset_ = qkScratchBytes;
        attentionAccumulatorRows_ = accumulatorRows;
        attentionAccumulatorRowBytes_ = accumulatorRowBytes;
        attentionAccumulatorValid_.assign(accumulatorRows, 0);
        attentionAccumulatorWriteReadyCycle_.assign(accumulatorRows, 0);
        attentionTileStorageBankNextReadCycle_.assign(attentionTileStorageBanks_, 0);
        attentionTileStorageBankNextWriteCycle_.assign(attentionTileStorageBanks_, 0);
        cBufferWriteReadyCycles_.clear();
        statAttentionStorageSessionAcquires_->addData(1);
        return true;
    }

    bool writeAttentionTileColumnAsync(
        uint32_t column, const std::vector<uint8_t>& values,
        uint64_t generation, uint64_t tag, uint64_t enqueueCycle,
        GemmBufferCallback callback) override {
        if (cBufferMode_ != CBufferMode::ATTENTION_TILE_STORAGE ||
            generation != attentionTileStorageGeneration_ ||
            column >= attentionTileStorageColumns_ ||
            values.size() != static_cast<size_t>(attentionTileStorageRows_) *
                attentionTileStorageElemBytes_ || !callback) {
            return false;
        }
        GemmProxyCommand command;
        command.kind = GemmProxyCommandKind::ATTENTION_TILE_COLUMN_WRITE;
        command.index = column;
        command.bytePayload = values;
        command.storageGeneration = generation;
        command.tag = tag;
        command.enqueueCycle = enqueueCycle;
        command.bufferCallback = std::move(callback);
        return enqueueGemmProxyCommand(std::move(command));
    }

    bool readAttentionTileRowAsync(
        uint32_t row, uint64_t generation, uint64_t tag,
        uint64_t enqueueCycle, AttentionTileReadCallback callback) override {
        if (cBufferMode_ != CBufferMode::ATTENTION_TILE_STORAGE ||
            generation != attentionTileStorageGeneration_ ||
            row >= attentionTileStorageRows_ || !callback ||
            std::find(attentionTileStorageColumnValid_.begin(),
                      attentionTileStorageColumnValid_.end(), 0) !=
                attentionTileStorageColumnValid_.end()) {
            return false;
        }
        GemmProxyCommand command;
        command.kind = GemmProxyCommandKind::ATTENTION_TILE_ROW_READ;
        command.index = row;
        command.storageGeneration = generation;
        command.tag = tag;
        command.enqueueCycle = enqueueCycle;
        command.attentionReadCallback = std::move(callback);
        return enqueueGemmProxyCommand(std::move(command));
    }

    bool writeAttentionAccumulatorRowAsync(
        uint32_t row, const std::vector<uint8_t>& values,
        uint64_t generation, uint64_t tag, uint64_t enqueueCycle,
        GemmBufferCallback callback) override {
        if (!attentionStorageSessionActive_ ||
            cBufferMode_ != CBufferMode::ATTENTION_TILE_STORAGE ||
            generation != attentionStorageSessionGeneration_ ||
            row >= attentionAccumulatorRows_ ||
            values.size() != attentionAccumulatorRowBytes_ || !callback) {
            return false;
        }
        GemmProxyCommand command;
        command.kind = GemmProxyCommandKind::ATTENTION_ACCUMULATOR_ROW_WRITE;
        command.index = row;
        command.bytePayload = values;
        command.storageGeneration = generation;
        command.tag = tag;
        command.enqueueCycle = enqueueCycle;
        command.bufferCallback = std::move(callback);
        return enqueueGemmProxyCommand(std::move(command));
    }

    bool readAttentionAccumulatorRowAsync(
        uint32_t row, uint64_t generation, uint64_t tag,
        uint64_t enqueueCycle, AttentionTileReadCallback callback) override {
        if (!attentionStorageSessionActive_ ||
            cBufferMode_ != CBufferMode::ATTENTION_TILE_STORAGE ||
            generation != attentionStorageSessionGeneration_ ||
            row >= attentionAccumulatorRows_ ||
            attentionAccumulatorValid_[row] == 0 || !callback) {
            return false;
        }
        GemmProxyCommand command;
        command.kind = GemmProxyCommandKind::ATTENTION_ACCUMULATOR_ROW_READ;
        command.index = row;
        command.storageGeneration = generation;
        command.tag = tag;
        command.enqueueCycle = enqueueCycle;
        command.attentionReadCallback = std::move(callback);
        return enqueueGemmProxyCommand(std::move(command));
    }

    bool endAttentionTileStorage(uint64_t generation) override {
        if (!attentionTileStorageActive_ ||
            cBufferMode_ != CBufferMode::ATTENTION_TILE_STORAGE ||
            generation != attentionTileStorageGeneration_ ||
            !gemmProxyCommands_.empty() ||
            !pendingAttentionStorageCompletions_.empty() ||
            std::find(attentionTileStorageColumnValid_.begin(),
                      attentionTileStorageColumnValid_.end(), 0) !=
                attentionTileStorageColumnValid_.end() ||
            std::find(attentionTileStorageRowRead_.begin(),
                      attentionTileStorageRowRead_.end(), 0) !=
                attentionTileStorageRowRead_.end()) {
            return false;
        }
        clearAttentionTileMetadata();
        if (!attentionStorageSessionActive_) {
            clearAttentionTileStorage();
        }
        statAttentionTileStorageReleases_->addData(1);
        return true;
    }

    bool endAttentionStorageSession(uint64_t generation) override {
        if (!attentionStorageSessionActive_ || attentionTileStorageActive_ ||
            cBufferMode_ != CBufferMode::ATTENTION_TILE_STORAGE ||
            generation != attentionStorageSessionGeneration_ ||
            !gemmProxyCommands_.empty() || !gemmArrayDoneCallbacks_.empty() ||
            !pendingGemmCompletions_.empty() ||
            !pendingAttentionStorageCompletions_.empty()) {
            return false;
        }
        clearAttentionTileStorage();
        statAttentionStorageSessionReleases_->addData(1);
        return true;
    }

    bool cancelAttentionTileStorage(uint64_t generation) override {
        if (cBufferMode_ == CBufferMode::FREE) return true;
        const bool tileGenerationMatches = attentionTileStorageActive_ &&
            generation == attentionTileStorageGeneration_;
        const bool sessionGenerationMatches = attentionStorageSessionActive_ &&
            generation == attentionStorageSessionGeneration_;
        if (cBufferMode_ != CBufferMode::ATTENTION_TILE_STORAGE ||
            (!tileGenerationMatches && !sessionGenerationMatches)) {
            return false;
        }
        purgeAttentionTileStorageCommands(generation);
        clearAttentionTileStorage();
        return true;
    }

    bool startWindow(const WorkerTaskListHeader& header) override {
        if (isBusy() || cBufferMode_ != CBufferMode::FREE) {
            return false;
        }
        const bool blockKSupported =
            header.hw_input_size != 0 &&
            (header.block_k <= header.hw_input_size ||
             (header.block_k % header.hw_input_size) == 0);
        if (header.hw_input_size == 0 || header.hw_output_size == 0 ||
            header.block_k == 0 || header.block_m == 0 ||
            !blockKSupported || header.block_m != header.hw_output_size) {
            if (extOutput_ != nullptr) {
                extOutput_->output(
                    "[Core %u] [wcp] ERROR: minimal micro-tiling requires block_m==hw_out and block_k<=hw_in or block_k multiple of hw_in, got block_m=%u block_k=%u hw_out=%u hw_in=%u\n",
                    coreId_, header.block_m, header.block_k, header.hw_output_size, header.hw_input_size);
            }
            return false;
        }
        const uint32_t reuseN = std::max<uint32_t>(header.a_reuse_n_tiles, 1u);
        const uint32_t reuseM = std::max<uint32_t>(header.b_reuse_m_tiles, 1u);
        const uint32_t kTiles = header.block_k > 0 ? header.k / header.block_k : 0;
        const uint32_t slotCount = std::max<uint32_t>(header.local_slot_count, 1u);
        const uint32_t residentK = residentKTileCount(header, true);
        if (reuseM > 1 && reuseN > 1 && residentK == 0) {
            if (extOutput_ != nullptr) {
                extOutput_->output(
                    "[Core %u] [wcp] ERROR: 2D K-window reuse needs enough slots for active+prefetch window buffers, got k_tiles=%u local_slot_count=%u reuse_m=%u reuse_n=%u prefetch_windows=%u\n",
                    coreId_, kTiles, header.local_slot_count, reuseM, reuseN, prefetchWindowDepth_);
            }
            return false;
        }
        if (!(reuseM > 1 && reuseN > 1) && reuseN > 1 && kTiles > slotCount) {
            if (extOutput_ != nullptr) {
                extOutput_->output(
                    "[Core %u] [wcp] ERROR: A-reuse requires k_tiles<=local_slot_count, got k_tiles=%u local_slot_count=%u reuse_n=%u\n",
                    coreId_, kTiles, header.local_slot_count, reuseN);
            }
            return false;
        }
        if (!(reuseM > 1 && reuseN > 1) && reuseM > 1 && kTiles > slotCount) {
            if (extOutput_ != nullptr) {
                extOutput_->output(
                    "[Core %u] [wcp] ERROR: B-reuse requires k_tiles<=local_slot_count, got k_tiles=%u local_slot_count=%u reuse_m=%u\n",
                    coreId_, kTiles, header.local_slot_count, reuseM);
            }
            return false;
        }
        if (reuseM > 1 && reuseN > 1) {
            const uint64_t partialTileBytes = static_cast<uint64_t>(header.block_m) *
                                              static_cast<uint64_t>(header.block_n) *
                                              static_cast<uint64_t>(header.elem_bytes);
            const uint64_t requiredCBufferBytes = static_cast<uint64_t>(reuseM) *
                                                  static_cast<uint64_t>(reuseN) * partialTileBytes;
            if (cBufferBytes_ < requiredCBufferBytes ||
                cBufferReadBytesPerCycle_ == 0 || cBufferWriteBytesPerCycle_ == 0) {
                if (extOutput_ != nullptr) {
                    extOutput_->output(
                        "[Core %u] [wcp] ERROR: partial-C buffer too small: required=%" PRIu64
                        " configured=%" PRIu64 " reuse=%ux%u tile_bytes=%" PRIu64 "\n",
                        coreId_, requiredCBufferBytes,
                        cBufferBytes_,
                        reuseM, reuseN, partialTileBytes);
                }
                return false;
            }
        }
        header_ = header;
        busy_ = true;
        if (reuseM > 1 && reuseN > 1) {
            cBufferMode_ = CBufferMode::GEMM_PARTIAL_C;
        }
        taskIndex_ = 0;
        reuseNIndex_ = 0;
        reuseMIndex_ = 0;
        deriveTask(taskIndex_);
        if (extOutput_ != nullptr) {
            extOutput_->verbose(CALL_INFO, 1, 0,
                "[WCP] core=%u start worker_slot=%u tasks=%u block_n=%u\n",
                coreId_, header_.worker_slot, header_.task_count, header_.block_n);
        }
        lastAccountCycle_ = 0;
        workerStartCycle_ = 0;
        workerEndCycle_ = 0;
        totalWindowCycles_ = 0;
        computeCycles_ = 0;
        tileReadyWaitCycles_ = 0;
        txnWaitCycles_ = 0;
        writebackWaitCycles_ = 0;
        wait2DActivateCycles_ = 0;
        wait2DActiveNotReadyCycles_ = 0;
        waitNon2DTxnCycles_ = 0;
        waitNoActiveTxnCycles_ = 0;
        windowSubmitActiveCount_ = 0;
        windowSubmitPrefetchCount_ = 0;
        windowActivateCount_ = 0;
        windowAdvanceWaitPrefetchCount_ = 0;
        crossMacroPrefetchSubmitCount_ = 0;
        crossMacroPrefetchAdoptCount_ = 0;
        cBufferReadCount_ = 0;
        cBufferWriteCount_ = 0;
        cBufferReadWaitCycles_ = 0;
        fusionConsumedCount_ = 0;
        lastCBufferWaitCycle_ = UINT64_MAX;
        nextMacroPrefetchValid_ = false;
        nextMacroPrefetchTaskIndex_ = 0;
        nextMacroPrefetch_ = Prefetch2DWindow{};
        windowTimelines_.clear();
        previousWindowTxnId_ = 0;
        activeWindowTxnId_ = 0;
        pendingWritebackTokens_.clear();
        lastStage3TraceCycle_ = 0;
        tileComputeStartCycles_.clear();
        tileComputeDoneCycles_.clear();
        tileRetireCycles_.clear();
        tileComputeStartSchedCycles_.clear();
        tileComputeDoneSchedCycles_.clear();
        tileRetireSchedCycles_.clear();
        resetPipelineState();
        phase_ = Phase::RUN;
        return true;
    }

    bool isBusy() const override {
        return busy_ || !gemmProxyCommands_.empty() ||
            !gemmArrayDoneCallbacks_.empty() || !pendingGemmCompletions_.empty() ||
            !pendingAttentionStorageCompletions_.empty() ||
            (array_ != nullptr && array_->hasPendingBufferTransfers());
    }

    bool tick(uint64_t cycle) override {
        tickGemmProxy(cycle);
        if (!busy_) {
            return isBusy();
        }
        if (workerStartCycle_ == 0) {
            workerStartCycle_ = cycle;
        }

        accountCycles(cycle);

        switch (phase_) {
        case Phase::RUN:
            tryIssuePrefetches();
            if (!computeInFlight_) {
                if (tryLaunchNextReadyTile(cycle)) {
                    break;
                }
                if (phase_ != Phase::RUN) {
                    break;
                }
                if (allTilesScheduled_ && !computeInFlight_ && activeComputeTileIndex_ < 0 &&
                           activeTxnRetiredTileCount_ >= activeTxnTileCount_) {
                    traceStage3("RUN_TO_WRITEBACK", cycle);
                    phase_ = Phase::WRITEBACK;
                } else if (activeTxnId_ != 0 && activeTxnRetiredTileCount_ >= activeTxnTileCount_ &&
                           (!use2DWindowEngine() || allReuseDoneForWindow()) &&
                           requestScheduler_ != nullptr && activeTransactionsDone()) {
                    traceStage3("RETIRE_ACTIVE_TXN", cycle);
                    retireActiveTransactions();
                    activeTxnId_ = 0;
                    active2DTxnIds_.clear();
                    active2DSchedulerTileRetired_.clear();
                    activeTxnTileCount_ = 0;
                    nextTxnComputeTile_ = 0;
                } else {
                    traceStage3Periodic("RUN_WAIT", cycle);
                }
            }
            break;
        case Phase::WRITEBACK:
            {
                if (use2DWindowEngine() && !isFinal2DWindow()) {
                    traceStage3("SAVE_PARTIAL", cycle);
                    if (!savePartialCFromArray(cycle)) {
                        return false;
                    }
                    markCurrentReuseDone();
                    advanceAfterWriteback(cycle);
                    if (phase_ == Phase::RUN) {
                        tryIssuePrefetches();
                        tryLaunchNextReadyTile(cycle);
                    }
                    break;
                }
                traceStage3("ISSUE_WRITEBACK", cycle);
                writebackDone_ = false;
                if (outputMode_ == "hbm") {
                    std::vector<uint8_t> tile;
                    if (!captureArrayOutput(tile)) {
                        return false;
                    }
                    writebackToken_ = issueDmaWrite(
                        current_.c_base_addr,
                        tile.size(),
                        tile);
                    // Final-C writes overlap later tiles, but every response is
                    // drained before this worker reports completion.
                    if (writebackToken_ != 0) {
                        pendingWritebackTokens_.push_back(writebackToken_);
                    }
                } else if (outputMode_ == "fusion") {
                    if (fusionDumpEnable_) {
                        std::vector<uint8_t> tile;
                        if (!captureArrayOutput(tile) ||
                            !consumeFusionOutput(current_.c_base_addr, tile)) {
                            return false;
                        }
                    }
                    fusionConsumedCount_++;
                }
            }
            writebackDone_ = true;
            writebackToken_ = 0;
            if (use2DWindowEngine()) {
                markCurrentReuseDone();
            }
            advanceAfterWriteback(cycle);
            if (phase_ == Phase::RUN) {
                tryIssuePrefetches();
                tryLaunchNextReadyTile(cycle);
            }
            break;
        case Phase::WRITEBACK_WAIT:
            if (!drainPendingWritebacks()) {
                traceStage3Periodic("WRITEBACK_WAIT", cycle);
                break;
            }
            phase_ = Phase::DONE;
            break;
        case Phase::DONE:
            workerEndCycle_ = cycle;
            emitWindowTimelines();
            if (extOutput_ != nullptr) {
                const uint64_t dma_time = tileReadyWaitCycles_ + txnWaitCycles_ + writebackWaitCycles_;
                extOutput_->output(
                    "[Core %u] [wcp] LATENCY(cycles): dma_issue=0 dma_wait=%" PRIu64
                    " dma_total=%" PRIu64 " compute=%" PRIu64
                    " compute_submit=0 compute_wait=%" PRIu64
                    " sched_protocol=0 c_store=%" PRIu64
                    " tile_ready_wait=%" PRIu64 " txn_wait=%" PRIu64
                    " writeback_wait=%" PRIu64
                    " wait_2d_activate=%" PRIu64 " wait_2d_active_not_ready=%" PRIu64
                    " wait_non2d_txn=%" PRIu64 " wait_no_active_txn=%" PRIu64
                    " window_submit_active=%" PRIu64 " window_submit_prefetch=%" PRIu64
                    " window_activate=%" PRIu64 " window_advance_wait_prefetch=%" PRIu64
                    " group_wait=0 poll_iters=0 overlap_issue=0 overlap_wait=0"
                    " issue_block_q=0 issue_write=0 ov_issue_block_q=0 ov_issue_write=0"
                    " task_desc=0 nloop=0 submit_pack=0 finish_publish=0"
                    " c_buffer_reads=%" PRIu64 " c_buffer_writes=%" PRIu64
                    " c_buffer_read_wait=%" PRIu64 " total=%" PRIu64
                    " descriptor_start_cycle=%" PRIu64
                    " start_cycle=%" PRIu64 " end_cycle=%" PRIu64 "\n",
                    coreId_, dma_time, dma_time, computeCycles_, computeCycles_,
                    writebackWaitCycles_, tileReadyWaitCycles_, txnWaitCycles_,
                    writebackWaitCycles_, wait2DActivateCycles_, wait2DActiveNotReadyCycles_,
                    waitNon2DTxnCycles_, waitNoActiveTxnCycles_, windowSubmitActiveCount_,
                    windowSubmitPrefetchCount_, windowActivateCount_, windowAdvanceWaitPrefetchCount_,
                    cBufferReadCount_, cBufferWriteCount_, cBufferReadWaitCycles_,
                    totalWindowCycles_, header_.descriptor_start_cycle,
                    workerStartCycle_, workerEndCycle_);
                extOutput_->output(
                    "[Core %u] [wcp] MACRO_PREFETCH enabled=%u submitted=%" PRIu64
                    " adopted=%" PRIu64 "\n",
                    coreId_, crossMacroPrefetch_ ? 1u : 0u,
                    crossMacroPrefetchSubmitCount_, crossMacroPrefetchAdoptCount_);
                extOutput_->output(
                    "[Core %u] [wcp] OUTPUT mode=%s fusion_consumed=%" PRIu64 "\n",
                    coreId_, outputMode_.c_str(), fusionConsumedCount_);
            }
            if (header_.finished_mailbox_addr != 0) {
                std::vector<uint8_t> one(sizeof(uint64_t), 0);
                const uint64_t val = 1;
                std::memcpy(one.data(), &val, sizeof(uint64_t));
                globalMem_->wr_to_globalmem(header_.finished_mailbox_addr, one.size(), one);
            }
            busy_ = false;
            if (cBufferMode_ == CBufferMode::GEMM_PARTIAL_C) {
                cBufferMode_ = CBufferMode::FREE;
            }
            phase_ = Phase::IDLE;
            break;
        case Phase::IDLE:
        default:
            break;
        }
        return isBusy();
    }

    bool handleArrayDone(uint32_t arrayId, uint64_t cycle) override {
        const auto inlineIt = gemmArrayDoneCallbacks_.find(arrayId);
        if (inlineIt != gemmArrayDoneCallbacks_.end()) {
            auto callback = std::move(inlineIt->second);
            gemmArrayDoneCallbacks_.erase(inlineIt);
            PendingGemmCompletion completion;
            completion.arrayId = arrayId;
            completion.arrayDoneCycle = cycle;
            completion.readyCycle = cycle + gemmProxyCompletionLatencyCycles_;
            completion.callback = std::move(callback);
            pendingGemmCompletions_.push_back(std::move(completion));
            return true;
        }
        if (!busy_ || !computeInFlight_ || phase_ != Phase::RUN) {
            return false;
        }
        if (arrayId >= current_.block_n) {
            return false;
        }
        if (pendingArrays_ > 0) {
            pendingArrays_ -= 1;
        }
        if (pendingArrays_ == 0 && activeComputeTileIndex_ >= 0) {
            if (!completeActiveMicroTile()) {
                return false;
            }
            if (use2DWindowEngine() && activeWindowTxnId_ != 0) {
                markWindowComputeSegmentEnd(activeWindowTxnId_, cycle);
            }
            computeInFlight_ = false;
            // Chain the next ready K tile immediately. Reuse/window transitions
            // still run through the normal tick state machine.
            if (!tryLaunchNextReadyTile(cycle) &&
                allTilesScheduled_ && activeComputeTileIndex_ < 0 &&
                activeTxnRetiredTileCount_ >= activeTxnTileCount_) {
                phase_ = Phase::WRITEBACK;
            }
        }
        return true;
    }

private:
    enum class CBufferMode : uint8_t {
        FREE,
        GEMM_PARTIAL_C,
        ATTENTION_TILE_STORAGE,
    };

    enum class GemmProxyCommandKind : uint8_t {
        PROGRAM_MATRIX,
        PROGRAM_MATRIX_GROUP,
        PROGRAM_INPUT,
        WRITE_OUTPUT,
        READ_OUTPUT,
        READ_OUTPUT_GROUP,
        LAUNCH,
        LAUNCH_GROUP,
        ATTENTION_TILE_COLUMN_WRITE,
        ATTENTION_TILE_ROW_READ,
        ATTENTION_ACCUMULATOR_ROW_WRITE,
        ATTENTION_ACCUMULATOR_ROW_READ,
    };

    struct GemmProxyCommand {
        GemmProxyCommandKind kind = GemmProxyCommandKind::PROGRAM_MATRIX;
        uint32_t arrayId = 0;
        uint32_t operandBank = 0;
        std::vector<uint32_t> arrayIds;
        std::vector<double> payload;
        size_t elemBytes = 0;
        uint64_t tag = 0;
        uint64_t outputMode = 0;
        uint32_t activeColumns = 0;
        AttentionClusterTrafficClass trafficClass =
            AttentionClusterTrafficClass::Legacy;
        uint32_t index = 0;
        uint64_t storageGeneration = 0;
        uint64_t enqueueCycle = 0;
        std::vector<uint8_t> bytePayload;
        GemmBufferCallback bufferCallback;
        GemmReadCallback readCallback;
        AttentionTileReadCallback attentionReadCallback;
    };

    struct PendingGemmCompletion {
        uint32_t arrayId = 0;
        uint64_t arrayDoneCycle = 0;
        uint64_t readyCycle = 0;
        GemmArrayDoneCallback callback;
    };

    struct PendingAttentionStorageCompletion {
        uint64_t readyCycle = 0;
        uint64_t tag = 0;
        uint64_t storageGeneration = 0;
        std::vector<uint8_t> data;
        GemmBufferCallback bufferCallback;
        AttentionTileReadCallback readCallback;
    };

    bool enqueueGemmProxyCommand(GemmProxyCommand command) {
        if (array_ == nullptr || busy_) {
            return false;
        }
        if (gemmProxyCommands_.size() >= gemmProxyQueueDepth_) {
            statGemmProxyQueueFullStalls_->addData(1);
            return false;
        }
        gemmProxyCommands_.push_back(std::move(command));
        return true;
    }

    bool launchGemmArrayBankImpl(
        uint32_t arrayId, uint32_t operandBank, uint64_t outputMode,
        uint32_t activeColumns, uint64_t enqueueCycle,
        GemmArrayDoneCallback callback) {
        if (array_ == nullptr ||
            !array_->validateOperandContextRequest(arrayId, operandBank) ||
            busy_ || !callback ||
            gemmArrayDoneCallbacks_.find(arrayId) !=
                gemmArrayDoneCallbacks_.end()) {
            return false;
        }
        if (gemmProxyCommands_.size() >= gemmProxyQueueDepth_) {
            statGemmProxyQueueFullStalls_->addData(1);
            return false;
        }
        gemmArrayDoneCallbacks_.emplace(arrayId, std::move(callback));
        GemmProxyCommand command;
        command.kind = GemmProxyCommandKind::LAUNCH;
        command.arrayId = arrayId;
        command.operandBank = operandBank;
        command.outputMode = outputMode;
        command.activeColumns = activeColumns;
        command.enqueueCycle = enqueueCycle;
        gemmProxyCommands_.push_back(std::move(command));
        return true;
    }

    bool dispatchGemmProxyCommand(
        const GemmProxyCommand& command, uint64_t dispatchCycle) {
        switch (command.kind) {
        case GemmProxyCommandKind::PROGRAM_MATRIX: {
            auto callback = command.bufferCallback;
            if (command.activeColumns != 0) {
                return array_->programMatrixActiveBankAsync(
                    command.arrayId, command.operandBank, command.payload,
                    command.activeColumns,
                    command.elemBytes, command.tag, std::move(callback));
            }
            return array_->programMatrixBankAsync(
                command.arrayId, command.operandBank, command.payload,
                command.elemBytes,
                command.tag, std::move(callback));
        }
        case GemmProxyCommandKind::PROGRAM_MATRIX_GROUP: {
            auto callback = command.bufferCallback;
            if (command.trafficClass != AttentionClusterTrafficClass::Legacy) {
                return array_->programMatrixGroupClassBankAsync(
                    command.arrayIds, command.operandBank, command.payload,
                    command.elemBytes, command.trafficClass, command.tag,
                    std::move(callback));
            }
            if (command.activeColumns != 0) {
                return array_->programMatrixGroupActiveBankAsync(
                    command.arrayIds, command.operandBank, command.payload,
                    command.activeColumns,
                    command.elemBytes, command.tag, std::move(callback));
            }
            return array_->programMatrixGroupBankAsync(
                command.arrayIds, command.operandBank, command.payload,
                command.elemBytes,
                command.tag, std::move(callback));
        }
        case GemmProxyCommandKind::PROGRAM_INPUT: {
            auto callback = command.bufferCallback;
            if (!command.arrayIds.empty()) {
                return array_->programInputGroupBankAsync(
                    command.arrayIds, command.operandBank, command.payload,
                    command.elemBytes, command.trafficClass, command.tag,
                    std::move(callback));
            }
            if (command.activeColumns != 0) {
                return array_->programInputActiveBankAsync(
                    command.arrayId, command.operandBank, command.payload,
                    command.activeColumns,
                    command.elemBytes, command.tag, std::move(callback));
            }
            return array_->programInputBankAsync(
                command.arrayId, command.operandBank, command.payload,
                command.elemBytes,
                command.tag, std::move(callback));
        }
        case GemmProxyCommandKind::WRITE_OUTPUT: {
            auto callback = command.bufferCallback;
            return array_->writeOutputAsync(
                command.arrayId, command.payload, command.elemBytes,
                command.tag, std::move(callback));
        }
        case GemmProxyCommandKind::READ_OUTPUT: {
            auto callback = command.readCallback;
            if (command.trafficClass != AttentionClusterTrafficClass::Legacy) {
                return array_->readOutputClassAsync(
                    command.arrayId, command.elemBytes, command.trafficClass,
                    command.tag, std::move(callback));
            }
            return array_->readOutputAsync(
                command.arrayId, command.elemBytes, command.tag,
                std::move(callback));
        }
        case GemmProxyCommandKind::READ_OUTPUT_GROUP: {
            auto callback = command.readCallback;
            return array_->readOutputGroupClassAsync(
                command.arrayIds, command.elemBytes, command.trafficClass,
                command.tag, std::move(callback));
        }
        case GemmProxyCommandKind::LAUNCH:
            array_->configureOutputMode(command.arrayId, command.outputMode);
            if (command.activeColumns != 0) {
                array_->beginComputationActiveBank(
                    command.arrayId, command.operandBank,
                    command.activeColumns);
            } else {
                array_->beginComputationBank(
                    command.arrayId, command.operandBank);
            }
            return true;
        case GemmProxyCommandKind::LAUNCH_GROUP:
            for (uint32_t arrayId : command.arrayIds) {
                array_->configureOutputMode(arrayId, command.outputMode);
                if (command.activeColumns != 0) {
                    array_->beginComputationActiveBank(
                        arrayId, command.operandBank, command.activeColumns);
                } else {
                    array_->beginComputationBank(arrayId, command.operandBank);
                }
            }
            return true;
        case GemmProxyCommandKind::ATTENTION_TILE_COLUMN_WRITE: {
            uint64_t readyCycle = 0;
            if (!attentionTileColumnWrite(
                    command.index, command.bytePayload,
                    command.storageGeneration, dispatchCycle,
                    readyCycle)) {
                return false;
            }
            PendingAttentionStorageCompletion completion;
            completion.readyCycle = readyCycle;
            completion.tag = command.tag;
            completion.storageGeneration = command.storageGeneration;
            completion.bufferCallback = command.bufferCallback;
            pendingAttentionStorageCompletions_.push_back(std::move(completion));
            return true;
        }
        case GemmProxyCommandKind::ATTENTION_TILE_ROW_READ: {
            uint64_t readyCycle = 0;
            std::vector<uint8_t> data;
            if (!attentionTileRowRead(
                    command.index, command.storageGeneration,
                    dispatchCycle, data, readyCycle)) {
                return false;
            }
            PendingAttentionStorageCompletion completion;
            completion.readyCycle = readyCycle;
            completion.tag = command.tag;
            completion.storageGeneration = command.storageGeneration;
            completion.data = std::move(data);
            completion.readCallback = command.attentionReadCallback;
            pendingAttentionStorageCompletions_.push_back(std::move(completion));
            return true;
        }
        case GemmProxyCommandKind::ATTENTION_ACCUMULATOR_ROW_WRITE: {
            uint64_t readyCycle = 0;
            if (!attentionAccumulatorRowWrite(
                    command.index, command.bytePayload,
                    command.storageGeneration, dispatchCycle, readyCycle)) {
                return false;
            }
            PendingAttentionStorageCompletion completion;
            completion.readyCycle = readyCycle;
            completion.tag = command.tag;
            completion.storageGeneration = command.storageGeneration;
            completion.bufferCallback = command.bufferCallback;
            pendingAttentionStorageCompletions_.push_back(std::move(completion));
            return true;
        }
        case GemmProxyCommandKind::ATTENTION_ACCUMULATOR_ROW_READ: {
            uint64_t readyCycle = 0;
            std::vector<uint8_t> data;
            if (!attentionAccumulatorRowRead(
                    command.index, command.storageGeneration,
                    dispatchCycle, data, readyCycle)) {
                return false;
            }
            PendingAttentionStorageCompletion completion;
            completion.readyCycle = readyCycle;
            completion.tag = command.tag;
            completion.storageGeneration = command.storageGeneration;
            completion.data = std::move(data);
            completion.readCallback = command.attentionReadCallback;
            pendingAttentionStorageCompletions_.push_back(std::move(completion));
            return true;
        }
        }
        return false;
    }

    void tickGemmProxy(uint64_t cycle) {
        uint32_t issued = 0;
        while (issued < gemmProxyIssueWidth_ && !gemmProxyCommands_.empty()) {
            const GemmProxyCommand& front = gemmProxyCommands_.front();
            if (cycle < front.enqueueCycle + gemmProxyCommandLatencyCycles_) {
                break;
            }
            GemmProxyCommand command = std::move(gemmProxyCommands_.front());
            gemmProxyCommands_.pop_front();
            if (!dispatchGemmProxyCommand(command, cycle)) {
                gemmProxyCommands_.push_front(std::move(command));
                break;
            }
            statGemmProxyCommandsIssued_->addData(1);
            statGemmProxyQueueWaitCycles_->addData(cycle - command.enqueueCycle);
            if (command.kind == GemmProxyCommandKind::LAUNCH ||
                command.kind == GemmProxyCommandKind::LAUNCH_GROUP) {
                statGemmProxyLaunchCommands_->addData(1);
            }
            issued++;
        }

        // Deliver completions after issue so callbacks cannot inject another
        // command into the same cycle's issue budget.
        while (!pendingGemmCompletions_.empty() &&
               pendingGemmCompletions_.front().readyCycle <= cycle) {
            PendingGemmCompletion completion =
                std::move(pendingGemmCompletions_.front());
            pendingGemmCompletions_.pop_front();
            statGemmProxyCompletionCallbacks_->addData(1);
            statGemmProxyCompletionDelayCycles_->addData(
                cycle - completion.arrayDoneCycle);
            completion.callback(completion.arrayId, cycle);
        }
        while (!pendingAttentionStorageCompletions_.empty() &&
               pendingAttentionStorageCompletions_.front().readyCycle <= cycle) {
            PendingAttentionStorageCompletion completion =
                std::move(pendingAttentionStorageCompletions_.front());
            pendingAttentionStorageCompletions_.pop_front();
            if (completion.readCallback) {
                completion.readCallback(true, completion.tag, completion.data);
            } else if (completion.bufferCallback) {
                completion.bufferCallback(true, completion.tag);
            }
        }
    }

    struct BufferSlot {
        uint64_t mat_addr = 0;
        uint64_t vec_addr = 0;
        uint32_t k_tile = 0;
        bool mat_done = false;
        bool vec_done = false;
        bool ready = false;
        bool inflight = false;
        bool in_use = false;
        uint64_t mat_token = 0;
        uint64_t vec_token = 0;
    };

    struct MicroOp {
        uint64_t taskId = 0;
        uint32_t logicalTileIdx = 0;
        uint32_t mStep = 0;
        uint32_t nGroup = 0;
        uint32_t kStep = 0;
        uint32_t slotIdx = 0;
    };

    struct KStepScoreboard {
        std::vector<uint8_t> matReady;
        std::vector<uint8_t> vecReady;
        int32_t lastCompletedKStep = -1;
        bool valid = false;
    };

    struct Prefetch2DWindow {
        uint64_t txnId = 0;
        std::vector<uint64_t> txnIds;
        uint32_t kBegin = 0;
        uint32_t kCount = 0;
        uint32_t buffer = 0;
    };

    struct CBufferPrefetch {
        uint32_t targetKBegin = 0;
        size_t partialIndex = 0;
        uint64_t readyCycle = 0;
        std::vector<uint8_t> data;
    };

    struct WindowTimeline {
        uint32_t taskIndex = 0;
        uint64_t taskId = 0;
        uint32_t macroTaskId = 0;
        uint32_t windowId = 0;
        uint32_t kBegin = 0;
        uint32_t kCount = 0;
        uint32_t tileCount = 0;
        uint32_t buffer = 0;
        uint64_t txnId = 0;
        uint64_t submitCycle = 0;
        uint64_t firstDataCycle = 0;
        uint64_t firstTileReadyCycle = 0;
        uint64_t readyCycle = 0;
        uint64_t activateCycle = 0;
        uint64_t computeStartCycle = 0;
        uint64_t computeEndCycle = 0;
        uint64_t nextWindowReadyCycle = 0;
        uint64_t computeSegmentCount = 0;
        uint64_t computeActiveCycles = 0;
        uint64_t intraWindowTransitionCount = 0;
        uint64_t intraWindowBubbleCount = 0;
        uint64_t intraWindowBubbleCycles = 0;
        uint64_t currentSegmentStartCycle = 0;
        uint64_t lastSegmentEndCycle = 0;
        bool computeSegmentOpen = false;
    };

    void resetPipelineState() {
        const uint32_t slotCount = std::max<uint32_t>(header_.local_slot_count, 1u);
        buffers_.assign(slotCount, BufferSlot{});
        for (uint32_t slot = 0; slot < slotCount; ++slot) {
            buffers_[slot] = BufferSlot{
                header_.local_mat_ping_gm_addr + static_cast<uint64_t>(slot) * header_.local_mat_slot_stride_bytes,
                header_.local_vec_ping_gm_addr + static_cast<uint64_t>(slot) * header_.local_vec_slot_stride_bytes,
            };
        }
        nextPrefetchK_ = current_.k_begin;
        allTilesScheduled_ = false;
        computeInFlight_ = false;
        activeComputeTileIndex_ = -1;
        activeComputeSlotIndex_ = -1;
        pendingArrays_ = 0;
        activeTxnId_ = 0;
        activeTxnTileCount_ = 0;
        nextTxnComputeTile_ = 0;
        activeTxnRetiredTileCount_ = 0;
        activeTxnTileRetired_.clear();
        active2DSchedulerTileRetired_.clear();
        windowReuseDone_.clear();
        nextReadyScanCursor_ = 0;
        activeTxnKBegin_ = 0;
        activeMicroKStep_ = 0;
        taskAccumInitialized_ = false;
        activeMatPayload_.clear();
        activeVecPayload_.clear();
        activeTileMicroOps_.clear();
        activeTileMicroOpCursor_ = 0;
        activeTileScoreboard_ = KStepScoreboard{};
        activeComputeReadyQueue_.clear();
        activeMicroOpIssued_ = false;
        activeIssuedMicroOp_ = MicroOp{};
        activeTilePayloadLoaded_ = false;
        if (use2DWindowEngine()) {
            totalKTileCount_ = header_.block_k > 0 ? header_.k / header_.block_k : current_.k_count;
            residentKTileCount_ = std::min<uint32_t>(activeResidentKTileCount(), std::max<uint32_t>(totalKTileCount_, 1u));
            activeWindowKBegin_ = 0;
            activeWindowKCount_ = std::min<uint32_t>(residentKTileCount_, totalKTileCount_);
            activeWindowBuffer_ = 0;
            activeWindowValid_ = false;
            next2DPrefetchK_ = activeWindowKCount_;
            active2DTxnIds_.clear();
            activeWindowTxnId_ = 0;
            prefetch2DWindows_.clear();
            current_.k_begin = activeWindowKBegin_;
            current_.k_count = activeWindowKCount_;
            const size_t partialCount = static_cast<size_t>(std::max<uint32_t>(header_.b_reuse_m_tiles, 1u)) *
                                        static_cast<size_t>(std::max<uint32_t>(header_.a_reuse_n_tiles, 1u));
            partialValid_.assign(partialCount, 0);
            cBufferPrefetches_.clear();
            windowReuseDone_.assign(partialCount, 0);
        } else {
            totalKTileCount_ = current_.k_count;
            residentKTileCount_ = current_.k_count;
            activeWindowKBegin_ = 0;
            activeWindowKCount_ = current_.k_count;
            activeWindowBuffer_ = 0;
            activeWindowValid_ = false;
            next2DPrefetchK_ = 0;
            active2DTxnIds_.clear();
            activeWindowTxnId_ = 0;
            prefetch2DWindows_.clear();
            partialValid_.clear();
            cBufferPrefetches_.clear();
            windowReuseDone_.clear();
        }
    }

    bool noBuffersBusy() const {
        return !computeInFlight_ && activeTxnId_ == 0;
    }

    bool is2DReuse() const {
        return std::max<uint32_t>(header_.a_reuse_n_tiles, 1u) > 1 &&
               std::max<uint32_t>(header_.b_reuse_m_tiles, 1u) > 1;
    }

    uint32_t base2DWindowBufferCount() const {
        return std::max<uint32_t>(prefetchWindowDepth_ + 1u, 2u);
    }

    uint32_t twoDWindowBufferCount(const WorkerTaskListHeader& header) const {
        const uint32_t baseBuffers = base2DWindowBufferCount();
        if (!crossMacroPrefetch_) {
            return baseBuffers;
        }
        const uint32_t reuse = std::max<uint32_t>(
            std::max<uint32_t>(header.a_reuse_n_tiles, 1u),
            std::max<uint32_t>(header.b_reuse_m_tiles, 1u));
        const uint32_t requestedWindowK = std::max<uint32_t>(windowKtiles_, 1u);
        const uint64_t lookaheadSlots = static_cast<uint64_t>(baseBuffers + 1u) *
                                        static_cast<uint64_t>(reuse) * requestedWindowK;
        return static_cast<uint64_t>(std::max<uint32_t>(header.local_slot_count, 1u)) >= lookaheadSlots
                   ? baseBuffers + 1u
                   : baseBuffers;
    }

    uint32_t residentKTileCount(const WorkerTaskListHeader& header, bool pingPong) const {
        const uint32_t reuseN = std::max<uint32_t>(header.a_reuse_n_tiles, 1u);
        const uint32_t reuseM = std::max<uint32_t>(header.b_reuse_m_tiles, 1u);
        const uint32_t slotCount = std::max<uint32_t>(header.local_slot_count, 1u);
        const uint32_t buffers = pingPong ? twoDWindowBufferCount(header) : 1u;
        const uint32_t matLimit = slotCount / (buffers * reuseM);
        const uint32_t vecLimit = slotCount / (buffers * reuseN);
        const uint32_t requestedWindowK = std::max<uint32_t>(windowKtiles_, 1u);
        return std::min<uint32_t>(requestedWindowK, std::min<uint32_t>(matLimit, vecLimit));
    }

    uint32_t twoDWindowBufferCount() const {
        return twoDWindowBufferCount(header_);
    }

    bool allocatePrefetchWindowBuffer(uint32_t& buffer) const {
        const uint32_t bufferCount = twoDWindowBufferCount();
        for (uint32_t off = 1; off < bufferCount; ++off) {
            const uint32_t candidate = (activeWindowBuffer_ + off) % bufferCount;
            bool inUse = (candidate == activeWindowBuffer_);
            for (const auto& window : prefetch2DWindows_) {
                if (window.buffer == candidate) {
                    inUse = true;
                    break;
                }
            }
            if (!inUse) {
                buffer = candidate;
                return true;
            }
        }
        return false;
    }

    uint32_t activeResidentKTileCount() const {
        if (!is2DReuse()) {
            return current_.k_count;
        }
        uint32_t resident = residentKTileCount(header_, true);
        if (resident == 0) {
            resident = residentKTileCount(header_, false);
        }
        return std::max<uint32_t>(resident, 1u);
    }

    bool use2DWindowEngine() const {
        return is2DReuse();
    }

    void traceStage3(const char* event, uint64_t cycle) {
        if (!stage3Trace_ || extOutput_ == nullptr || !use2DWindowEngine()) {
            return;
        }
        const bool activeReady = !active2DTxnIds_.empty() &&
                                 are2DTransactionsReady(active2DTxnIds_, twoDWindowTransactionTileCount(activeWindowKCount_), false);
        const Prefetch2DWindow* frontPrefetch = prefetch2DWindows_.empty() ? nullptr : &prefetch2DWindows_.front();
        const bool prefetchReady = frontPrefetch != nullptr &&
                                  are2DTransactionsReady(frontPrefetch->txnIds, twoDWindowTransactionTileCount(frontPrefetch->kCount), false);
        extOutput_->output(
            "[Core %u] [wcp] STAGE3_TRACE event=%s cycle=%" PRIu64
            " task_idx=%u task=%" PRIu64 " macro=%u reuseM=%u/%u reuseN=%u/%u"
            " win_valid=%u win_k=%u+%u win_buf=%u totalK=%u residentK=%u"
            " active_txn=%" PRIu64 " active_ids=%zu active_ready=%u retired=%u/%u"
            " prefetch_txn=%" PRIu64 " prefetch_ids=%zu prefetch_ready=%u prefetch_k=%u+%u prefetch_buf=%u prefetch_queue=%zu/%u next_prefetch=%u"
            " all_sched=%u compute=%u active_tile=%d slot=%d buffers_busy=%u partial_idx=%zu final_win=%u\n",
            coreId_, event, cycle,
            taskIndex_, current_.task_id, currentMacroTaskId_,
            reuseMIndex_, currentReuseMCount_, reuseNIndex_, currentReuseNCount_,
            activeWindowValid_ ? 1u : 0u, activeWindowKBegin_, activeWindowKCount_, activeWindowBuffer_,
            totalKTileCount_, residentKTileCount_,
            activeTxnId_, active2DTxnIds_.size(), activeReady ? 1u : 0u,
            activeTxnRetiredTileCount_, activeTxnTileCount_,
            frontPrefetch != nullptr ? frontPrefetch->txnId : 0,
            frontPrefetch != nullptr ? frontPrefetch->txnIds.size() : 0,
            prefetchReady ? 1u : 0u,
            frontPrefetch != nullptr ? frontPrefetch->kBegin : 0,
            frontPrefetch != nullptr ? frontPrefetch->kCount : 0,
            frontPrefetch != nullptr ? frontPrefetch->buffer : 0,
            prefetch2DWindows_.size(), prefetchWindowDepth_, next2DPrefetchK_,
            allTilesScheduled_ ? 1u : 0u, computeInFlight_ ? 1u : 0u,
            activeComputeTileIndex_, activeComputeSlotIndex_, noBuffersBusy() ? 0u : 1u,
            currentPartialIndex(), isFinal2DWindow() ? 1u : 0u);
    }

    void traceStage3Periodic(const char* event, uint64_t cycle) {
        if (!use2DWindowEngine()) {
            return;
        }
        if (cycle < lastStage3TraceCycle_ + 100000) {
            return;
        }
        lastStage3TraceCycle_ = cycle;
        traceStage3(event, cycle);
    }

    uint32_t groupMatSlotFor(uint32_t kTile) const {
        const uint32_t kTiles = std::max<uint32_t>(current_.k_count, 1u);
        if (use2DWindowEngine()) {
            const uint32_t windowKCapacity = std::max<uint32_t>(residentKTileCount_, 1u);
            const uint32_t perBufferSlots = windowKCapacity * std::max<uint32_t>(header_.b_reuse_m_tiles, 1u);
            return activeWindowBuffer_ * perBufferSlots + reuseMIndex_ * std::max<uint32_t>(activeWindowKCount_, 1u) + kTile;
        }
        return is2DReuse() ? (reuseMIndex_ * kTiles + kTile) : kTile;
    }

    uint32_t groupVecSlotFor(uint32_t kTile) const {
        const uint32_t kTiles = std::max<uint32_t>(current_.k_count, 1u);
        if (use2DWindowEngine()) {
            const uint32_t windowKCapacity = std::max<uint32_t>(residentKTileCount_, 1u);
            const uint32_t perBufferSlots = windowKCapacity * std::max<uint32_t>(header_.a_reuse_n_tiles, 1u);
            return activeWindowBuffer_ * perBufferSlots + reuseNIndex_ * std::max<uint32_t>(activeWindowKCount_, 1u) + kTile;
        }
        return is2DReuse() ? (reuseNIndex_ * kTiles + kTile) : kTile;
    }

    size_t reuseIndex(uint32_t reuseM, uint32_t reuseN) const {
        return static_cast<size_t>(reuseM) * static_cast<size_t>(currentReuseNCount_) + static_cast<size_t>(reuseN);
    }

    void markCurrentReuseDone() {
        if (!use2DWindowEngine()) {
            return;
        }
        const size_t idx = reuseIndex(reuseMIndex_, reuseNIndex_);
        const size_t count = static_cast<size_t>(currentReuseMCount_) * static_cast<size_t>(currentReuseNCount_);
        if (windowReuseDone_.size() != count) {
            windowReuseDone_.assign(count, 0);
        }
        if (idx < windowReuseDone_.size()) {
            windowReuseDone_[idx] = 1;
        }
    }

    bool allReuseDoneForWindow() const {
        if (!use2DWindowEngine()) {
            return true;
        }
        const size_t count = static_cast<size_t>(currentReuseMCount_) * static_cast<size_t>(currentReuseNCount_);
        if (count == 0 || windowReuseDone_.size() != count) {
            return false;
        }
        for (uint8_t done : windowReuseDone_) {
            if (done == 0) {
                return false;
            }
        }
        return true;
    }

    bool buildActiveTileMicroOps(uint32_t localTileIdx, uint32_t slotIdx) {
        const uint32_t kSteps = microKStepCount();
        if (kSteps == 0) {
            return false;
        }

        activeTileMicroOps_.clear();
        activeTileMicroOps_.reserve(kSteps);
        for (uint32_t kStep = 0; kStep < kSteps; ++kStep) {
            MicroOp op{};
            op.taskId = current_.task_id;
            op.logicalTileIdx = localTileIdx;
            op.mStep = 0;
            op.nGroup = 0;
            op.kStep = kStep;
            op.slotIdx = slotIdx;
            activeTileMicroOps_.push_back(op);
        }

        activeTileMicroOpCursor_ = 0;
        activeTileScoreboard_.matReady.assign(kSteps, 0);
        activeTileScoreboard_.vecReady.assign(kSteps, 0);
        activeTileScoreboard_.lastCompletedKStep = -1;
        activeTileScoreboard_.valid = true;
        activeComputeReadyQueue_.clear();
        activeMicroOpIssued_ = false;
        activeIssuedMicroOp_ = MicroOp{};
        activeTilePayloadLoaded_ = false;
        refreshActiveComputeReadyQueue();
        return true;
    }

    bool isMicroOpReady(const MicroOp& op) const {
        if (!activeTileScoreboard_.valid) {
            return false;
        }
        if (op.kStep >= activeTileScoreboard_.matReady.size() ||
            op.kStep >= activeTileScoreboard_.vecReady.size()) {
            return false;
        }
        if (activeTileScoreboard_.matReady[op.kStep] == 0 ||
            activeTileScoreboard_.vecReady[op.kStep] == 0) {
            return false;
        }

        const int32_t expectedNextK = activeTileScoreboard_.lastCompletedKStep + 1;
        if (static_cast<int32_t>(op.kStep) != expectedNextK) {
            return false;
        }
        return true;
    }

    void refreshActiveComputeReadyQueue() {
        activeComputeReadyQueue_.clear();
        if (!activeTileScoreboard_.valid || activeMicroOpIssued_) {
            return;
        }
        for (size_t idx = activeTileMicroOpCursor_; idx < activeTileMicroOps_.size(); ++idx) {
            const MicroOp& op = activeTileMicroOps_[idx];
            if (isMicroOpReady(op)) {
                activeComputeReadyQueue_.push_back(op);
                break;
            }
        }
    }

    void updateActiveTileInputReadiness() {
        if (!activeTileScoreboard_.valid || activeComputeTileIndex_ < 0) {
            return;
        }
        bool changed = false;
        for (size_t i = 0; i < activeTileScoreboard_.matReady.size(); ++i) {
            bool matReady = true;
            bool vecReady = true;
            if (use2DWindowEngine() && activeWindowValid_) {
                const bool ready = is2DComputeTileReady(static_cast<uint32_t>(activeComputeTileIndex_));
                matReady = ready;
                vecReady = ready;
            } else if (is2DReuse() && groupResidentValid_ && residentMacroTaskId_ == currentMacroTaskId_) {
                matReady = true;
                vecReady = true;
            } else if (requestScheduler_ != nullptr && activeTxnId_ != 0) {
                matReady = false;
                vecReady = false;
                const uint32_t tileIdx = static_cast<uint32_t>(activeComputeTileIndex_);
                if (!requestScheduler_->getTileKStepReadiness(
                        activeTxnId_, tileIdx, static_cast<uint32_t>(i), matReady, vecReady)) {
                    if (requestScheduler_->isTileReady(activeTxnId_, tileIdx)) {
                        matReady = true;
                        vecReady = true;
                    } else {
                        WcpTileTimelineDebug dbg{};
                        if (requestScheduler_->getTileTimeline(activeTxnId_, tileIdx, dbg)) {
                            matReady = (dbg.matDoneCycle != 0);
                            vecReady = (dbg.vecDoneCycle != 0);
                        }
                    }
                }
            }
            const uint8_t matVal = matReady ? 1 : 0;
            const uint8_t vecVal = vecReady ? 1 : 0;
            if (activeTileScoreboard_.matReady[i] != matVal) {
                activeTileScoreboard_.matReady[i] = matVal;
                changed = true;
            }
            if (activeTileScoreboard_.vecReady[i] != vecVal) {
                activeTileScoreboard_.vecReady[i] = vecVal;
                changed = true;
            }
        }
        if (changed) {
            refreshActiveComputeReadyQueue();
        }
    }

    void accountCycles(uint64_t cycle) {
        if (!busy_) {
            lastAccountCycle_ = cycle;
            return;
        }
        if (lastAccountCycle_ == 0) {
            lastAccountCycle_ = cycle;
        }
        if (cycle < lastAccountCycle_) {
            return;
        }
        const uint64_t delta = cycle - lastAccountCycle_ + 1;
        totalWindowCycles_ += delta;
        if (phase_ == Phase::RUN) {
            if (computeInFlight_) {
                computeCycles_ += delta;
            } else if (activeTxnId_ != 0) {
                if (activeTxnRetiredTileCount_ < activeTxnTileCount_) {
                    tileReadyWaitCycles_ += delta;
                    if (use2DWindowEngine()) {
                        if (!activeWindowValid_) {
                            wait2DActivateCycles_ += delta;
                        } else {
                            wait2DActiveNotReadyCycles_ += delta;
                        }
                    } else {
                        waitNon2DTxnCycles_ += delta;
                    }
                } else {
                    txnWaitCycles_ += delta;
                }
            } else {
                waitNoActiveTxnCycles_ += delta;
            }
        } else if (phase_ == Phase::WRITEBACK || phase_ == Phase::WRITEBACK_WAIT) {
            writebackWaitCycles_ += delta;
        }
        lastAccountCycle_ = cycle + 1;
    }

    WindowTimeline* findWindowTimeline(uint64_t txnId) {
        for (auto& window : windowTimelines_) {
            if (window.txnId == txnId) {
                return &window;
            }
        }
        return nullptr;
    }

    void updateWindowTimeline(uint64_t txnId, const std::vector<uint64_t>& txnIds) {
        WindowTimeline* window = findWindowTimeline(txnId);
        if (window == nullptr || requestScheduler_ == nullptr) {
            return;
        }
        uint64_t firstData = 0;
        uint64_t firstReady = 0;
        uint64_t lastReady = 0;
        for (uint64_t id : txnIds) {
            for (uint32_t idx = 0; idx < window->tileCount; ++idx) {
                WcpTileTimelineDebug dbg{};
                if (!requestScheduler_->getTileTimeline(id, idx, dbg)) {
                    continue;
                }
                const uint64_t data = std::min(
                    dbg.matDoneCycle != 0 ? dbg.matDoneCycle : dbg.vecDoneCycle,
                    dbg.vecDoneCycle != 0 ? dbg.vecDoneCycle : dbg.matDoneCycle);
                if (data != 0 && (firstData == 0 || data < firstData)) {
                    firstData = data;
                }
                if (dbg.readyCycle != 0 && (firstReady == 0 || dbg.readyCycle < firstReady)) {
                    firstReady = dbg.readyCycle;
                }
                if (dbg.readyCycle > lastReady) {
                    lastReady = dbg.readyCycle;
                }
            }
        }
        if (firstData != 0 && (window->firstDataCycle == 0 || firstData < window->firstDataCycle)) {
            window->firstDataCycle = firstData;
        }
        if (firstReady != 0 && (window->firstTileReadyCycle == 0 || firstReady < window->firstTileReadyCycle)) {
            window->firstTileReadyCycle = firstReady;
        }
        if (lastReady != 0 && lastReady > window->readyCycle) {
            window->readyCycle = lastReady;
        }
    }

    void updateWindowReady(const std::vector<uint64_t>& txnIds) {
        if (!txnIds.empty()) {
            updateWindowTimeline(txnIds.front(), txnIds);
        }
    }

    void markWindowActivated(uint64_t txnId, uint64_t cycle) {
        WindowTimeline* window = findWindowTimeline(txnId);
        if (window == nullptr || window->activateCycle != 0) {
            return;
        }
        window->activateCycle = cycle;
        if (previousWindowTxnId_ != 0) {
            WindowTimeline* previous = findWindowTimeline(previousWindowTxnId_);
            if (previous != nullptr && previous->nextWindowReadyCycle == 0) {
                previous->nextWindowReadyCycle = window->readyCycle;
            }
            previousWindowTxnId_ = 0;
        }
    }

    void markWindowComputeStart(uint64_t txnId, uint64_t cycle) {
        WindowTimeline* window = findWindowTimeline(txnId);
        if (window != nullptr && window->computeStartCycle == 0) {
            window->computeStartCycle = cycle;
        }
    }

    void markWindowComputeEnd(uint64_t txnId, uint64_t cycle) {
        WindowTimeline* window = findWindowTimeline(txnId);
        if (window != nullptr && window->computeEndCycle == 0) {
            window->computeEndCycle = cycle;
        }
    }

    void markWindowComputeSegmentStart(uint64_t txnId, uint64_t cycle) {
        WindowTimeline* window = findWindowTimeline(txnId);
        if (window == nullptr || window->computeSegmentOpen) {
            return;
        }
        if (window->computeSegmentCount > 0) {
            const uint64_t gap = cycle > window->lastSegmentEndCycle
                                     ? cycle - window->lastSegmentEndCycle
                                     : 0;
            // A one-cycle WCP/array event handoff is present even when the
            // next tile is ready.  Report only the excess interval as an
            // intra-window bubble so this metric reflects exposed delay.
            const uint64_t exposedGap = gap > 1 ? gap - 1 : 0;
            window->intraWindowTransitionCount += 1;
            window->intraWindowBubbleCycles += exposedGap;
            window->intraWindowBubbleCount += static_cast<uint64_t>(exposedGap > 0);
        }
        window->computeSegmentCount += 1;
        window->currentSegmentStartCycle = cycle;
        window->computeSegmentOpen = true;
        markWindowComputeStart(txnId, cycle);
    }

    void markWindowComputeSegmentEnd(uint64_t txnId, uint64_t cycle) {
        WindowTimeline* window = findWindowTimeline(txnId);
        if (window == nullptr || !window->computeSegmentOpen) {
            return;
        }
        if (cycle > window->currentSegmentStartCycle) {
            window->computeActiveCycles += cycle - window->currentSegmentStartCycle;
        }
        window->lastSegmentEndCycle = cycle;
        window->computeSegmentOpen = false;
    }

    void emitWindowTimelines() const {
        if (extOutput_ == nullptr) {
            return;
        }
        for (const auto& window : windowTimelines_) {
            extOutput_->output(
                "[Core %u] [wcp] WINDOW_BREAKDOWN task_idx=%u task=%" PRIu64
                " macro=%u window=%u k_begin=%u k_count=%u tile_count=%u buffer=%u txn=%" PRIu64
                " submit=%" PRIu64 " first_data=%" PRIu64 " first_tile_ready=%" PRIu64
                " ready=%" PRIu64 " activate=%" PRIu64 " compute_start=%" PRIu64
                " compute_end=%" PRIu64 " next_ready=%" PRIu64
                " compute_segments=%" PRIu64 " compute_active=%" PRIu64
                " last_segment_end=%" PRIu64
                " intra_window_transitions=%" PRIu64
                " intra_window_bubbles=%" PRIu64
                " intra_window_bubble_cycles=%" PRIu64 "\n",
                coreId_, window.taskIndex, window.taskId, window.macroTaskId,
                window.windowId, window.kBegin, window.kCount, window.tileCount,
                window.buffer, window.txnId, window.submitCycle, window.firstDataCycle,
                window.firstTileReadyCycle, window.readyCycle, window.activateCycle,
                window.computeStartCycle, window.computeEndCycle,
                window.nextWindowReadyCycle, window.computeSegmentCount,
                window.computeActiveCycles, window.lastSegmentEndCycle,
                window.intraWindowTransitionCount,
                window.intraWindowBubbleCount, window.intraWindowBubbleCycles);
        }
    }

    uint64_t issueDmaRead(uint64_t src_pa, size_t length, uint64_t gm_dst_addr) {
        if (auto* gm = dynamic_cast<GlobalMemoryImplement*>(globalMem_)) {
            return gm->dma_read_from_host_to_globalmem_async(src_pa, length, gm_dst_addr);
        }
        if (auto* gm = dynamic_cast<GlobalMemoryLocal*>(globalMem_)) {
            return gm->dma_read_from_host_to_globalmem_async(src_pa, length, gm_dst_addr);
        }
        return 0;
    }

    uint64_t issueDmaWrite(uint64_t dst_pa, size_t length, const std::vector<uint8_t>& data) {
        if (auto* gm = dynamic_cast<GlobalMemoryImplement*>(globalMem_)) {
            return gm->dma_write_to_host_async(dst_pa, length, data);
        }
        if (auto* gm = dynamic_cast<GlobalMemoryLocal*>(globalMem_)) {
            return gm->dma_write_to_host_async(dst_pa, length, data);
        }
        return 0;
    }

    bool consumeFusionOutput(uint64_t cBaseAddr, const std::vector<uint8_t>& data) {
        if (!fusionDumpEnable_) {
            return true;
        }
        if (!fusionDump_.is_open()) {
            return false;
        }
        const uint64_t length = static_cast<uint64_t>(data.size());
        fusionDump_.write(reinterpret_cast<const char*>(&cBaseAddr), sizeof(cBaseAddr));
        fusionDump_.write(reinterpret_cast<const char*>(&length), sizeof(length));
        if (!data.empty()) {
            fusionDump_.write(reinterpret_cast<const char*>(data.data()),
                              static_cast<std::streamsize>(data.size()));
        }
        return fusionDump_.good();
    }

    bool dmaDone(uint64_t token) const {
        if (token == 0) {
            return true;
        }
        if (auto* gm = dynamic_cast<GlobalMemoryImplement*>(globalMem_)) {
            return gm->dma_completion_done(token);
        }
        if (auto* gm = dynamic_cast<GlobalMemoryLocal*>(globalMem_)) {
            return gm->dma_completion_done(token);
        }
        return false;
    }

    void retireDma(uint64_t token) {
        if (token == 0) {
            return;
        }
        if (auto* gm = dynamic_cast<GlobalMemoryImplement*>(globalMem_)) {
            gm->dma_completion_retire(token);
            return;
        }
        if (auto* gm = dynamic_cast<GlobalMemoryLocal*>(globalMem_)) {
            gm->dma_completion_retire(token);
        }
    }

    bool drainPendingWritebacks() {
        bool allDone = true;
        std::deque<uint64_t> remaining;
        while (!pendingWritebackTokens_.empty()) {
            const uint64_t token = pendingWritebackTokens_.front();
            pendingWritebackTokens_.pop_front();
            if (dmaDone(token)) {
                retireDma(token);
            } else {
                remaining.push_back(token);
                allDone = false;
            }
        }
        pendingWritebackTokens_.swap(remaining);
        return allDone;
    }

    void finishOrDrainWritebacks() {
        if (drainPendingWritebacks()) {
            phase_ = Phase::DONE;
        } else {
            phase_ = Phase::WRITEBACK_WAIT;
        }
    }

    void advanceAfterWriteback(uint64_t cycle) {
        if (use2DWindowEngine()) {
            traceStage3("ADVANCE_CALL", cycle);
            if (advance2DWindowEngine()) {
                traceStage3("ADVANCE_CONTINUE", cycle);
                phase_ = Phase::RUN;
                return;
            }
            traceStage3("ADVANCE_TASK_DONE", cycle);
            if ((taskIndex_ + 1) < header_.task_count) {
                if (nextMacroPrefetchValid_ &&
                    nextMacroPrefetchTaskIndex_ == (taskIndex_ + 1u)) {
                    previousWindowTxnId_ = activeWindowTxnId_;
                }
                taskIndex_ += 1;
                reuseNIndex_ = 0;
                reuseMIndex_ = 0;
                deriveTask(taskIndex_);
                adoptNextMacroPrefetch();
                phase_ = Phase::RUN;
                return;
            }
            finishOrDrainWritebacks();
            return;
        }
        if ((reuseNIndex_ + 1) < currentReuseNCount_) {
            reuseNIndex_ += 1;
            deriveTask(taskIndex_);
            phase_ = Phase::RUN;
            return;
        }
        if ((reuseMIndex_ + 1) < currentReuseMCount_) {
            reuseMIndex_ += 1;
            reuseNIndex_ = 0;
            deriveTask(taskIndex_);
            phase_ = Phase::RUN;
            return;
        }
        if ((taskIndex_ + 1) < header_.task_count) {
            taskIndex_ += 1;
            reuseNIndex_ = 0;
            reuseMIndex_ = 0;
            deriveTask(taskIndex_);
            phase_ = Phase::RUN;
            return;
        }
        finishOrDrainWritebacks();
    }

    void tryIssuePrefetches() {
        if (requestScheduler_ == nullptr) {
            return;
        }
        if (use2DWindowEngine()) {
            tryIssue2DWindowPrefetches();
            return;
        }
        if (activeTxnId_ != 0) {
            return;
        }
        const uint32_t k_end = current_.k_begin + current_.k_count;
        if (is2DReuse() && groupResidentValid_ && residentMacroTaskId_ == currentMacroTaskId_) {
            allTilesScheduled_ = true;
            if (activeTxnTileCount_ == 0) {
                activeTxnTileCount_ = current_.k_count;
                activeTxnTileRetired_.assign(current_.k_count, 0);
                tileComputeStartCycles_.assign(current_.k_count, 0);
                tileComputeDoneCycles_.assign(current_.k_count, 0);
                tileRetireCycles_.assign(current_.k_count, 0);
                tileComputeStartSchedCycles_.assign(current_.k_count, 0);
                tileComputeDoneSchedCycles_.assign(current_.k_count, 0);
                tileRetireSchedCycles_.assign(current_.k_count, 0);
            }
            return;
        }
        if (nextPrefetchK_ >= k_end) {
            allTilesScheduled_ = true;
            return;
        }
        const uint32_t tiles = is2DReuse()
                                   ? (std::max<uint32_t>(header_.b_reuse_m_tiles, 1u) * current_.k_count)
                                   : std::min<uint32_t>(windowKtiles_, k_end - nextPrefetchK_);
        WcpWindowTransaction txn{};
        txn.workerSlot = header_.worker_slot;
        txn.taskId = static_cast<uint32_t>(current_.task_id);
        txn.windowId = currentWindowId_++;
        txn.kBegin = nextPrefetchK_;
        txn.kTiles = tiles;
        txn.blockN = current_.block_n;
        txn.blockK = header_.block_k;
        txn.elemBytes = current_.elem_bytes;
        txn.hwInputSize = current_.array_input_size;
        txn.memNodeSize = header_.mem_node_size;
        txn.matBaseAddr = current_.mat_base_addr + static_cast<uint64_t>(nextPrefetchK_) * current_.mat_stride_bytes;
        txn.vecBaseAddr = current_.vec_base_addr + static_cast<uint64_t>(nextPrefetchK_) * current_.vec_stride_bytes;
        txn.localMatBaseAddr = header_.local_mat_ping_gm_addr;
        txn.localVecBaseAddr = header_.local_vec_ping_gm_addr;
        txn.localMatSlotStrideBytes = header_.local_mat_slot_stride_bytes;
        txn.localVecSlotStrideBytes = header_.local_vec_slot_stride_bytes;
        txn.slotCount = std::max<uint32_t>(header_.local_slot_count, 1u);
        txn.matStrideBytes = current_.mat_stride_bytes;
        txn.vecStrideBytes = current_.vec_stride_bytes;
        txn.skipMatRead = !is2DReuse() && (std::max<uint32_t>(header_.a_reuse_n_tiles, 1u) > 1 && reuseNIndex_ > 0);
        txn.skipVecRead = !is2DReuse() && (std::max<uint32_t>(header_.b_reuse_m_tiles, 1u) > 1 && reuseMIndex_ > 0);
        activeTxnId_ = requestScheduler_->submitWindowTransaction(txn);
        activeTxnTileCount_ = tiles;
        nextTxnComputeTile_ = 0;
        activeTxnRetiredTileCount_ = 0;
        activeTxnTileRetired_.assign(tiles, 0);
        nextReadyScanCursor_ = 0;
        activeTxnKBegin_ = nextPrefetchK_;
        tileComputeStartCycles_.assign(tiles, 0);
        tileComputeDoneCycles_.assign(tiles, 0);
        tileRetireCycles_.assign(tiles, 0);
        tileComputeStartSchedCycles_.assign(tiles, 0);
        tileComputeDoneSchedCycles_.assign(tiles, 0);
        tileRetireSchedCycles_.assign(tiles, 0);
        nextPrefetchK_ += is2DReuse() ? current_.k_count : tiles;
        allTilesScheduled_ = (nextPrefetchK_ >= k_end);
    }

    std::vector<uint64_t> submit2DWindowTransactions(uint32_t kBegin, uint32_t kCount, uint32_t buffer) {
        std::vector<uint64_t> txnIds;
        if (requestScheduler_ == nullptr || kCount == 0) {
            return txnIds;
        }
        const uint32_t reuseM = std::max<uint32_t>(header_.b_reuse_m_tiles, 1u);
        const uint32_t reuseN = std::max<uint32_t>(header_.a_reuse_n_tiles, 1u);
        const uint32_t windowKCapacity = std::max<uint32_t>(residentKTileCount_, kCount);
        const uint32_t perBufferMatSlots = reuseM * windowKCapacity;
        const uint32_t perBufferVecSlots = reuseN * windowKCapacity;
        const uint64_t matGroupBase = current_.mat_base_addr - static_cast<uint64_t>(reuseMIndex_) * static_cast<uint64_t>(totalKTileCount_) * current_.mat_stride_bytes;
        const uint64_t vecGroupBase = current_.vec_base_addr - static_cast<uint64_t>(reuseNIndex_) * static_cast<uint64_t>(totalKTileCount_) * current_.vec_stride_bytes;
        const uint64_t localMatBufferBase = header_.local_mat_ping_gm_addr + static_cast<uint64_t>(buffer) * perBufferMatSlots * header_.local_mat_slot_stride_bytes;
        const uint64_t localVecBufferBase = header_.local_vec_ping_gm_addr + static_cast<uint64_t>(buffer) * perBufferVecSlots * header_.local_vec_slot_stride_bytes;

        WcpWindowTransaction txn{};
        txn.workerSlot = header_.worker_slot;
        txn.taskId = static_cast<uint32_t>(current_.task_id);
        txn.windowId = currentWindowId_++;
        txn.kBegin = kBegin;
        txn.kTiles = twoDWindowTransactionTileCount(kCount);
        txn.blockN = current_.block_n;
        txn.blockK = header_.block_k;
        txn.elemBytes = current_.elem_bytes;
        txn.hwInputSize = current_.array_input_size;
        txn.memNodeSize = header_.mem_node_size;
        txn.matBaseAddr = matGroupBase + static_cast<uint64_t>(kBegin) * current_.mat_stride_bytes;
        txn.vecBaseAddr = vecGroupBase + static_cast<uint64_t>(kBegin) * current_.vec_stride_bytes;
        txn.localMatBaseAddr = localMatBufferBase;
        txn.localVecBaseAddr = localVecBufferBase;
        txn.localMatSlotStrideBytes = header_.local_mat_slot_stride_bytes;
        txn.localVecSlotStrideBytes = header_.local_vec_slot_stride_bytes;
        txn.slotCount = std::max<uint32_t>(txn.kTiles, 1u);
        txn.matStrideBytes = current_.mat_stride_bytes;
        txn.vecStrideBytes = current_.vec_stride_bytes;
        txn.useIndependentMatVecTiles = true;
        txn.matTileCount = currentReuseMCount_ * kCount;
        txn.vecTileCount = currentReuseNCount_ * kCount;
        txn.kWindowTiles = kCount;
        txn.totalKTileCount = totalKTileCount_;
        txnIds.push_back(requestScheduler_->submitWindowTransaction(txn));
        const uint64_t txnId = txnIds.back();
        windowTimelines_.push_back(WindowTimeline{
            taskIndex_, current_.task_id, currentMacroTaskId_, txn.windowId,
            kBegin, kCount, txn.kTiles, buffer, txnId, lastAccountCycle_});
        return txnIds;
    }

    uint32_t twoDWindowTransactionTileCount(uint32_t kCount) const {
        return std::max<uint32_t>(currentReuseMCount_, currentReuseNCount_) * kCount;
    }

    bool hasSplitKTransactions(const std::vector<uint64_t>& txnIds, uint32_t kCount) const {
        return kCount > 1 && txnIds.size() == static_cast<size_t>(kCount);
    }

    uint32_t twoDPerTransactionTileCount(const std::vector<uint64_t>& txnIds, uint32_t kCount) const {
        return hasSplitKTransactions(txnIds, kCount) ? twoDWindowTransactionTileCount(1) : twoDWindowTransactionTileCount(kCount);
    }

    bool are2DTransactionsReady(const std::vector<uint64_t>& txnIds, uint32_t tileCount, bool retireReady = true) {
        if (requestScheduler_ == nullptr || txnIds.empty()) {
            return false;
        }
        const uint32_t perTxnTileCount = txnIds.size() > 1 ? twoDWindowTransactionTileCount(1) : tileCount;
        bool allReady = true;
        for (uint64_t txnId : txnIds) {
            for (uint32_t idx = 0; idx < perTxnTileCount; ++idx) {
                if (!requestScheduler_->isTileReady(txnId, idx)) {
                    allReady = false;
                    continue;
                }
                if (retireReady) {
                    requestScheduler_->retireTileReady(txnId, idx);
                }
            }
        }
        return allReady;
    }

    void retire2DTransactions(const std::vector<uint64_t>& txnIds, uint32_t tileCount) {
        if (requestScheduler_ == nullptr) {
            return;
        }
        const uint32_t perTxnTileCount = txnIds.size() > 1 ? twoDWindowTransactionTileCount(1) : tileCount;
        for (uint64_t txnId : txnIds) {
            for (uint32_t idx = 0; idx < perTxnTileCount; ++idx) {
                if (requestScheduler_->isTileReady(txnId, idx)) {
                    requestScheduler_->retireTileReady(txnId, idx);
                }
            }
            if (requestScheduler_->isTransactionDone(txnId)) {
                requestScheduler_->retireTransaction(txnId);
            }
        }
    }

    bool activeTransactionsDone() const {
        if (requestScheduler_ == nullptr || activeTxnId_ == 0) {
            return false;
        }
        if (use2DWindowEngine() && !active2DTxnIds_.empty()) {
            for (uint64_t txnId : active2DTxnIds_) {
                if (!requestScheduler_->isTransactionDone(txnId)) {
                    return false;
                }
            }
            return true;
        }
        return requestScheduler_->isTransactionDone(activeTxnId_);
    }

    void retireActiveTransactions() {
        if (requestScheduler_ == nullptr || activeTxnId_ == 0) {
            return;
        }
        if (use2DWindowEngine() && !active2DTxnIds_.empty()) {
            for (uint64_t txnId : active2DTxnIds_) {
                if (requestScheduler_->isTransactionDone(txnId)) {
                    requestScheduler_->retireTransaction(txnId);
                }
            }
            return;
        }
        if (requestScheduler_->isTransactionDone(activeTxnId_)) {
            requestScheduler_->retireTransaction(activeTxnId_);
        }
    }

    void retireReady2DTransactions(
        const std::vector<uint64_t>& txnIds,
        uint32_t tileCount,
        std::vector<uint8_t>& retired) {
        if (requestScheduler_ == nullptr || txnIds.empty()) {
            return;
        }
        const uint32_t perTxnTileCount = txnIds.size() > 1 ? twoDWindowTransactionTileCount(1) : tileCount;
        const size_t retiredCount = txnIds.size() * static_cast<size_t>(perTxnTileCount);
        if (retired.size() != retiredCount) {
            retired.assign(retiredCount, 0);
        }
        for (size_t txnIdx = 0; txnIdx < txnIds.size(); ++txnIdx) {
            const uint64_t txnId = txnIds[txnIdx];
            for (uint32_t idx = 0; idx < perTxnTileCount; ++idx) {
                const size_t retiredIdx = txnIdx * static_cast<size_t>(perTxnTileCount) + idx;
                if (retired[retiredIdx] != 0) {
                    continue;
                }
                if (requestScheduler_->isTileReady(txnId, idx)) {
                    requestScheduler_->retireTileReady(txnId, idx);
                    retired[retiredIdx] = 1;
                }
            }
        }
    }

    bool is2DComputeTileReady(uint32_t localTileIdx) const {
        return is2DComputeTileReadyFor(localTileIdx, reuseMIndex_, reuseNIndex_);
    }

    bool is2DComputeTileReadyFor(uint32_t localTileIdx, uint32_t reuseM, uint32_t reuseN) const {
        if (!use2DWindowEngine()) {
            return false;
        }
        if (activeTxnId_ == 0 || active2DTxnIds_.empty()) {
            return true;
        }
        if (requestScheduler_ == nullptr || activeWindowKCount_ == 0) {
            return false;
        }
        const bool splitK = hasSplitKTransactions(active2DTxnIds_, activeWindowKCount_);
        const uint32_t matIdx = splitK ? reuseM : (reuseM * activeWindowKCount_ + localTileIdx);
        const uint32_t vecIdx = splitK ? reuseN : (reuseN * activeWindowKCount_ + localTileIdx);
        const size_t txnBegin = splitK ? static_cast<size_t>(localTileIdx) : 0;
        const size_t txnEnd = splitK ? std::min<size_t>(txnBegin + 1, active2DTxnIds_.size()) : active2DTxnIds_.size();
        for (size_t txnIdx = txnBegin; txnIdx < txnEnd; ++txnIdx) {
            const uint64_t txnId = active2DTxnIds_[txnIdx];
            bool matReady = false;
            bool ignoredVecReady = false;
            bool ignoredMatReady = false;
            bool vecReady = false;
            const bool haveMatState = requestScheduler_->getTileKStepReadiness(
                txnId, matIdx, 0, matReady, ignoredVecReady);
            const bool haveVecState = requestScheduler_->getTileKStepReadiness(
                txnId, vecIdx, 0, ignoredMatReady, vecReady);
            if (!haveMatState || !haveVecState) {
                matReady = requestScheduler_->isTileReady(txnId, matIdx);
                vecReady = requestScheduler_->isTileReady(txnId, vecIdx);
            }
            if (matReady && vecReady) {
                return true;
            }
        }
        return false;
    }

    bool selectNextReuseForActiveWindow(uint32_t& nextM, uint32_t& nextN, bool requireReady = false) const {
        if (!use2DWindowEngine()) {
            return false;
        }
        const size_t count = static_cast<size_t>(currentReuseMCount_) * static_cast<size_t>(currentReuseNCount_);
        if (count == 0 || windowReuseDone_.size() != count) {
            return false;
        }

        bool haveFallback = false;
        uint32_t fallbackM = 0;
        uint32_t fallbackN = 0;
        const uint32_t start = static_cast<uint32_t>(reuseIndex(reuseMIndex_, reuseNIndex_) + 1u);
        for (uint32_t off = 0; off < static_cast<uint32_t>(count); ++off) {
            const uint32_t flat = (start + off) % static_cast<uint32_t>(count);
            const uint32_t m = flat / currentReuseNCount_;
            const uint32_t n = flat % currentReuseNCount_;
            if (windowReuseDone_[flat] != 0) {
                continue;
            }
            if (!haveFallback) {
                fallbackM = m;
                fallbackN = n;
                haveFallback = true;
            }
            for (uint32_t k = 0; k < activeWindowKCount_; ++k) {
                if (is2DComputeTileReadyFor(k, m, n)) {
                    nextM = m;
                    nextN = n;
                    return true;
                }
            }
        }
        if (!requireReady && haveFallback) {
            nextM = fallbackM;
            nextN = fallbackN;
            return true;
        }
        return false;
    }

    void activate2DWindow(uint32_t kBegin, uint32_t kCount, uint32_t buffer) {
        windowActivateCount_++;
        activeWindowKBegin_ = kBegin;
        activeWindowKCount_ = kCount;
        activeWindowBuffer_ = buffer;
        activeWindowValid_ = true;
        current_.k_begin = kBegin;
        current_.k_count = kCount;
        activeTxnTileCount_ = kCount;
        activeTxnRetiredTileCount_ = 0;
        activeTxnTileRetired_.assign(kCount, 0);
        windowReuseDone_.assign(static_cast<size_t>(currentReuseMCount_) * static_cast<size_t>(currentReuseNCount_), 0);
        nextReadyScanCursor_ = 0;
        activeTxnKBegin_ = kBegin;
        tileComputeStartCycles_.assign(kCount, 0);
        tileComputeDoneCycles_.assign(kCount, 0);
        tileRetireCycles_.assign(kCount, 0);
        tileComputeStartSchedCycles_.assign(kCount, 0);
        tileComputeDoneSchedCycles_.assign(kCount, 0);
        tileRetireSchedCycles_.assign(kCount, 0);
        allTilesScheduled_ = true;
        if (!active2DTxnIds_.empty()) {
            activeWindowTxnId_ = active2DTxnIds_.front();
            updateWindowTimeline(activeWindowTxnId_, active2DTxnIds_);
            markWindowActivated(activeWindowTxnId_, lastAccountCycle_);
        }
        traceStage3("ACTIVATE_WINDOW", lastAccountCycle_);
    }

    void tryIssueNextMacroPrefetch() {
        const bool finalWindowSubmitted = next2DPrefetchK_ >= totalKTileCount_ &&
                                          !prefetch2DWindows_.empty() &&
                                          (prefetch2DWindows_.back().kBegin +
                                           prefetch2DWindows_.back().kCount) >= totalKTileCount_;
        if (!crossMacroPrefetch_ || nextMacroPrefetchValid_ ||
            !activeWindowValid_ || (!isFinal2DWindow() && !finalWindowSubmitted) ||
            (taskIndex_ + 1u) >= header_.task_count) {
            return;
        }
        uint32_t buffer = 0;
        if (!allocatePrefetchWindowBuffer(buffer)) {
            return;
        }

        const WorkerWindowDescriptor savedCurrent = current_;
        const uint32_t savedTaskIndex = taskIndex_;
        const uint32_t savedReuseMIndex = reuseMIndex_;
        const uint32_t savedReuseNIndex = reuseNIndex_;
        const uint32_t savedReuseMCount = currentReuseMCount_;
        const uint32_t savedReuseNCount = currentReuseNCount_;
        const uint32_t savedCurrentK = currentK_;
        const uint32_t savedMacroTaskId = currentMacroTaskId_;

        taskIndex_ = savedTaskIndex + 1u;
        reuseMIndex_ = 0;
        reuseNIndex_ = 0;
        deriveTask(taskIndex_, false);
        Prefetch2DWindow window{};
        window.kBegin = 0;
        window.kCount = std::min<uint32_t>(residentKTileCount_, totalKTileCount_);
        window.buffer = buffer;
        window.txnIds = submit2DWindowTransactions(window.kBegin, window.kCount, window.buffer);
        window.txnId = window.txnIds.empty() ? 0 : window.txnIds.front();

        current_ = savedCurrent;
        taskIndex_ = savedTaskIndex;
        reuseMIndex_ = savedReuseMIndex;
        reuseNIndex_ = savedReuseNIndex;
        currentReuseMCount_ = savedReuseMCount;
        currentReuseNCount_ = savedReuseNCount;
        currentK_ = savedCurrentK;
        currentMacroTaskId_ = savedMacroTaskId;

        if (window.txnIds.empty()) {
            return;
        }
        nextMacroPrefetchTaskIndex_ = savedTaskIndex + 1u;
        nextMacroPrefetch_ = std::move(window);
        nextMacroPrefetchValid_ = true;
        crossMacroPrefetchSubmitCount_++;
        traceStage3("SUBMIT_NEXT_MACRO_PREFETCH", lastAccountCycle_);
    }

    bool adoptNextMacroPrefetch() {
        if (!nextMacroPrefetchValid_ || nextMacroPrefetchTaskIndex_ != taskIndex_ ||
            nextMacroPrefetch_.txnIds.empty()) {
            return false;
        }
        Prefetch2DWindow window = std::move(nextMacroPrefetch_);
        nextMacroPrefetch_ = Prefetch2DWindow{};
        nextMacroPrefetchValid_ = false;
        nextMacroPrefetchTaskIndex_ = 0;

        activeWindowKBegin_ = window.kBegin;
        activeWindowKCount_ = window.kCount;
        activeWindowBuffer_ = window.buffer;
        activeWindowValid_ = false;
        current_.k_begin = window.kBegin;
        current_.k_count = window.kCount;
        next2DPrefetchK_ = window.kBegin + window.kCount;
        activeWindowTxnId_ = window.txnId;
        activeTxnId_ = window.txnId;
        active2DTxnIds_ = std::move(window.txnIds);
        activeTxnTileCount_ = window.kCount;
        activeTxnRetiredTileCount_ = 0;
        activeTxnTileRetired_.assign(window.kCount, 0);
        active2DSchedulerTileRetired_.assign(
            twoDWindowTransactionTileCount(window.kCount), 0);
        allTilesScheduled_ = false;
        crossMacroPrefetchAdoptCount_++;
        traceStage3("ADOPT_NEXT_MACRO_PREFETCH", lastAccountCycle_);
        return true;
    }

    void tryIssue2DWindowPrefetches() {
        const uint32_t totalK = std::max<uint32_t>(totalKTileCount_, 1u);
        if (!activeWindowValid_) {
            if (active2DTxnIds_.empty()) {
                active2DTxnIds_ = submit2DWindowTransactions(activeWindowKBegin_, activeWindowKCount_, activeWindowBuffer_);
                windowSubmitActiveCount_++;
                activeTxnId_ = active2DTxnIds_.empty() ? 0 : active2DTxnIds_.front();
                activeTxnTileCount_ = activeWindowKCount_;
                active2DSchedulerTileRetired_.assign(twoDWindowTransactionTileCount(activeWindowKCount_), 0);
                traceStage3("SUBMIT_ACTIVE_WINDOW", lastAccountCycle_);
            }
            updateWindowReady(active2DTxnIds_);
            retireReady2DTransactions(active2DTxnIds_, twoDWindowTransactionTileCount(activeWindowKCount_), active2DSchedulerTileRetired_);
            uint32_t readyM = reuseMIndex_;
            uint32_t readyN = reuseNIndex_;
            if (selectNextReuseForActiveWindow(readyM, readyN, true)) {
                reuseMIndex_ = readyM;
                reuseNIndex_ = readyN;
                deriveTask(taskIndex_, false);
                traceStage3("ACTIVE_WINDOW_READY_REUSE", lastAccountCycle_);
                activate2DWindow(activeWindowKBegin_, activeWindowKCount_, activeWindowBuffer_);
            } else if (is2DComputeTileReady(0)) {
                traceStage3("ACTIVE_WINDOW_PROGRESSIVE_READY", lastAccountCycle_);
                activate2DWindow(activeWindowKBegin_, activeWindowKCount_, activeWindowBuffer_);
            } else if (are2DTransactionsReady(active2DTxnIds_, twoDWindowTransactionTileCount(activeWindowKCount_))) {
                retire2DTransactions(active2DTxnIds_, twoDWindowTransactionTileCount(activeWindowKCount_));
                active2DTxnIds_.clear();
                activeTxnId_ = 0;
                traceStage3("ACTIVE_WINDOW_READY", lastAccountCycle_);
                activate2DWindow(activeWindowKBegin_, activeWindowKCount_, activeWindowBuffer_);
            }
            if (!active2DTxnIds_.empty() || activeWindowValid_) {
                fill2DPrefetchQueue(totalK);
            }
            tryIssueNextMacroPrefetch();
            return;
        }
        updateWindowReady(active2DTxnIds_);
        retireReady2DTransactions(active2DTxnIds_, twoDWindowTransactionTileCount(activeWindowKCount_), active2DSchedulerTileRetired_);
        fill2DPrefetchQueue(totalK);
        for (const auto& window : prefetch2DWindows_) {
            updateWindowReady(window.txnIds);
        }
        tryIssueNextMacroPrefetch();
    }

    void fill2DPrefetchQueue(uint32_t totalK) {
        while (prefetch2DWindows_.size() < prefetchWindowDepth_ && next2DPrefetchK_ < totalK) {
            uint32_t buffer = 0;
            if (!allocatePrefetchWindowBuffer(buffer)) {
                break;
            }
            Prefetch2DWindow window{};
            window.kBegin = next2DPrefetchK_;
            window.kCount = std::min<uint32_t>(residentKTileCount_, totalK - next2DPrefetchK_);
            window.buffer = buffer;
            window.txnIds = submit2DWindowTransactions(window.kBegin, window.kCount, window.buffer);
            if (window.txnIds.empty()) {
                break;
            }
            windowSubmitPrefetchCount_++;
            window.txnId = window.txnIds.front();
            next2DPrefetchK_ += window.kCount;
            prefetch2DWindows_.push_back(std::move(window));
            traceStage3("SUBMIT_PREFETCH_WINDOW", lastAccountCycle_);
        }
    }

    int selectNextTile() {
        if (activeComputeTileIndex_ >= 0) {
            return activeComputeTileIndex_;
        }
        if (use2DWindowEngine() && activeWindowValid_) {
            for (uint32_t idx = 0; idx < activeWindowKCount_; ++idx) {
                if (idx < activeTxnTileRetired_.size() && activeTxnTileRetired_[idx] == 0) {
                    if (is2DComputeTileReady(idx)) {
                        return static_cast<int>(idx);
                    }
                }
            }
            if (activeTxnRetiredTileCount_ == 0 &&
                !computeInFlight_ && activeComputeTileIndex_ < 0 &&
                !taskAccumInitialized_ && !activeTilePayloadLoaded_) {
                uint32_t readyM = reuseMIndex_;
                uint32_t readyN = reuseNIndex_;
                if (selectNextReuseForActiveWindow(readyM, readyN, true) &&
                    (readyM != reuseMIndex_ || readyN != reuseNIndex_)) {
                    reuseMIndex_ = readyM;
                    reuseNIndex_ = readyN;
                    deriveTask(taskIndex_, false);
                    resetComputeOnlyStateForNextTile();
                    allTilesScheduled_ = true;
                    traceStage3("SELECT_READY_REUSE", lastAccountCycle_);
                    for (uint32_t idx = 0; idx < activeWindowKCount_; ++idx) {
                        if (idx < activeTxnTileRetired_.size() && activeTxnTileRetired_[idx] == 0) {
                            if (is2DComputeTileReady(idx)) {
                                return static_cast<int>(idx);
                            }
                        }
                    }
                }
            }
            return -1;
        }
        if (use2DWindowEngine() && !activeWindowValid_) {
            return -1;
        }
        if (!use2DWindowEngine() && is2DReuse() && groupResidentValid_ && residentMacroTaskId_ == currentMacroTaskId_) {
            for (uint32_t idx = 0; idx < current_.k_count; ++idx) {
                if (idx < activeTxnTileRetired_.size() && activeTxnTileRetired_[idx] == 0) {
                    return static_cast<int>(idx);
                }
            }
            return -1;
        }
        if (requestScheduler_ == nullptr || activeTxnId_ == 0 || activeTxnTileCount_ == 0) {
            return -1;
        }

        if (!use2DWindowEngine() && is2DReuse() && activeTxnId_ != 0) {
            bool allReady = true;
            for (uint32_t idx = 0; idx < activeTxnTileCount_; ++idx) {
                if (requestScheduler_->isTileReady(activeTxnId_, idx)) {
                    requestScheduler_->retireTileReady(activeTxnId_, idx);
                } else {
                    allReady = false;
                }
            }
            if (!allReady) {
                return -1;
            }
            groupResidentValid_ = true;
            residentMacroTaskId_ = currentMacroTaskId_;
            requestScheduler_->retireTransaction(activeTxnId_);
            activeTxnId_ = 0;
            activeTxnTileCount_ = current_.k_count;
            activeTxnRetiredTileCount_ = 0;
            activeTxnTileRetired_.assign(current_.k_count, 0);
            nextReadyScanCursor_ = 0;
            for (uint32_t idx = 0; idx < current_.k_count; ++idx) {
                return static_cast<int>(idx);
            }
        }

        for (uint32_t off = 0; off < activeTxnTileCount_; ++off) {
            const uint32_t idx = (nextReadyScanCursor_ + off) % activeTxnTileCount_;
            if (idx < activeTxnTileRetired_.size() && activeTxnTileRetired_[idx] != 0) {
                continue;
            }
            if (requestScheduler_->isTileReady(activeTxnId_, idx)) {
                nextReadyScanCursor_ = (idx + 1) % activeTxnTileCount_;
                return static_cast<int>(idx);
            }
            bool matReady = false;
            bool vecReady = false;
            if (requestScheduler_->getTileKStepReadiness(activeTxnId_, idx, 0, matReady, vecReady) &&
                matReady && vecReady) {
                nextReadyScanCursor_ = (idx + 1) % activeTxnTileCount_;
                return static_cast<int>(idx);
            }
        }
        return -1;
    }

    bool loadTilePayload(uint32_t local_tile_idx) {
        const uint64_t vec_bytes = current_.vec_stride_bytes;
        const uint32_t slotCount = std::max<uint32_t>(header_.local_slot_count, 1u);
        const uint32_t matSlotIdx = is2DReuse() ? groupMatSlotFor(local_tile_idx) : (local_tile_idx % slotCount);
        const uint32_t vecSlotIdx = is2DReuse() ? groupVecSlotFor(local_tile_idx) : (local_tile_idx % slotCount);
        const uint64_t mat_addr = header_.local_mat_ping_gm_addr + static_cast<uint64_t>(matSlotIdx) * header_.local_mat_slot_stride_bytes;
        const uint64_t vec_addr = header_.local_vec_ping_gm_addr + static_cast<uint64_t>(vecSlotIdx) * header_.local_vec_slot_stride_bytes;
        globalMem_->rd_from_globalmem(mat_addr, static_cast<size_t>(current_.mat_stride_bytes), activeMatPayload_);
        globalMem_->rd_from_globalmem(vec_addr, static_cast<size_t>(vec_bytes), activeVecPayload_);
        if (activeMatPayload_.size() < current_.mat_stride_bytes || activeVecPayload_.size() < vec_bytes) {
            return false;
        }
        return true;
    }

    bool loadActiveMicroTileToArrays() {
        const uint32_t kBase = activeMicroKStep_ * current_.array_input_size;
        if (kBase >= header_.block_k) {
            return false;
        }
        const uint32_t activeK = std::min<uint32_t>(current_.array_input_size, header_.block_k - kBase);
        for (uint32_t array_id = 0; array_id < current_.block_n; ++array_id) {
            for (uint32_t idx = 0; idx < current_.array_input_size * current_.array_output_size; ++idx) {
                const uint32_t row = idx / current_.array_input_size;
                const uint32_t col = idx % current_.array_input_size;
                double value = 0.0;
                if (col < activeK) {
                    const size_t matIdx =
                        (static_cast<size_t>(row) * header_.block_k + static_cast<size_t>(kBase + col)) *
                        current_.elem_bytes;
                    value = decodeElement(&activeMatPayload_[matIdx]);
                }
                array_->setMatrixItem(static_cast<int32_t>(array_id), static_cast<int32_t>(idx), value);
            }
            for (uint32_t idx = 0; idx < current_.array_input_size; ++idx) {
                double value = 0.0;
                if (idx < activeK) {
                    const size_t off =
                        static_cast<size_t>(array_id) * header_.vec_stride_bytes +
                        static_cast<size_t>(kBase + idx) * current_.elem_bytes;
                    value = decodeElement(&activeVecPayload_[off]);
                }
                array_->setVectorItem(static_cast<int32_t>(array_id), static_cast<int32_t>(idx), value);
            }
        }
        return true;
    }

    bool tryLaunchNextReadyTile(uint64_t cycle) {
        if (computeInFlight_) {
            return false;
        }
        int tile = activeComputeTileIndex_;
        if (tile < 0) {
            tile = selectNextTile();
        }
        if (tile < 0) {
            return false;
        }
        if (activeComputeTileIndex_ < 0) {
            activeComputeTileIndex_ = tile;
            activeComputeSlotIndex_ = static_cast<int>(
                use2DWindowEngine()
                    ? groupMatSlotFor(static_cast<uint32_t>(tile))
                    : (static_cast<uint32_t>(tile) % std::max<uint32_t>(header_.local_slot_count, 1u)));
            activeMicroKStep_ = 0;
            if (activeComputeSlotIndex_ >= 0 && activeComputeSlotIndex_ < static_cast<int>(buffers_.size())) {
                buffers_[activeComputeSlotIndex_].in_use = true;
            }
            if (!buildActiveTileMicroOps(
                    static_cast<uint32_t>(activeComputeTileIndex_),
                    static_cast<uint32_t>(activeComputeSlotIndex_))) {
                phase_ = Phase::DONE;
                return false;
            }
        }
        return tryLaunchActiveMicroTile(cycle);
    }

    bool tryLaunchActiveMicroTile(uint64_t cycle) {
        if (activeComputeTileIndex_ < 0 || computeInFlight_) {
            return false;
        }
        updateActiveTileInputReadiness();
        if (activeComputeReadyQueue_.empty()) {
            return false;
        }
        if (!activeTilePayloadLoaded_) {
            if (!loadTilePayload(static_cast<uint32_t>(activeComputeTileIndex_))) {
                phase_ = Phase::DONE;
                return false;
            }
            activeTilePayloadLoaded_ = true;
        }
        if (use2DWindowEngine() && !taskAccumInitialized_ && activeWindowKBegin_ > 0) {
            if (!loadCurrentPartialCWhenReady(cycle)) {
                return false;
            }
            taskAccumInitialized_ = true;
        }
        if (!issueActiveMicroTile()) {
            phase_ = Phase::DONE;
            return false;
        }
        pendingArrays_ = current_.block_n;
        computeInFlight_ = true;
        prefetchNextPartialForCurrentWindow(cycle);
        prefetchFirstPartialForNextWindow(cycle);
        const size_t tile = static_cast<size_t>(activeComputeTileIndex_);
        if (tile < tileComputeStartCycles_.size()) {
            if (tileComputeStartCycles_[tile] == 0) {
                tileComputeStartCycles_[tile] = cycle;
                if (tile < tileComputeStartSchedCycles_.size()) {
                    tileComputeStartSchedCycles_[tile] = schedulerTimelineCycle();
                }
            }
            if (use2DWindowEngine() && activeWindowTxnId_ != 0) {
                markWindowComputeSegmentStart(activeWindowTxnId_, cycle);
            }
        }
        return true;
    }

    bool issueActiveMicroTile() {
        if (activeTileMicroOpCursor_ >= activeTileMicroOps_.size() || activeMicroOpIssued_) {
            return false;
        }
        if (activeComputeReadyQueue_.empty()) {
            refreshActiveComputeReadyQueue();
        }
        if (activeComputeReadyQueue_.empty()) {
            return false;
        }

        const MicroOp op = activeComputeReadyQueue_.front();
        activeComputeReadyQueue_.pop_front();
        if (activeComputeTileIndex_ < 0 || activeComputeSlotIndex_ < 0) {
            return false;
        }
        if (op.logicalTileIdx != static_cast<uint32_t>(activeComputeTileIndex_) ||
            op.slotIdx != static_cast<uint32_t>(activeComputeSlotIndex_)) {
            return false;
        }
        if (!isMicroOpReady(op)) {
            return false;
        }

        activeIssuedMicroOp_ = op;
        activeMicroOpIssued_ = true;
        activeMicroKStep_ = op.kStep;
        if (!loadActiveMicroTileToArrays()) {
            return false;
        }
        const uint64_t outputMode = taskAccumInitialized_ ? 1 : 0;
        taskAccumInitialized_ = true;
        for (uint32_t array_id = 0; array_id < current_.block_n; ++array_id) {
            array_->configureOutputMode(array_id, outputMode);
            array_->beginComputation(array_id);
        }
        return true;
    }

    size_t currentPartialIndex() const {
        const uint32_t reuseN = std::max<uint32_t>(header_.a_reuse_n_tiles, 1u);
        return static_cast<size_t>(reuseMIndex_) * reuseN + static_cast<size_t>(reuseNIndex_);
    }

    bool captureArrayOutput(std::vector<uint8_t>& tile) const {
        tile.assign(static_cast<size_t>(header_.block_m) * current_.block_n * current_.elem_bytes, 0);
        for (uint32_t n = 0; n < current_.block_n; ++n) {
            const size_t dst_off = static_cast<size_t>(n) * header_.block_m * current_.elem_bytes;
            if (outputIsFloat_) {
                auto* outVec = static_cast<std::vector<float>*>(array_->getOutputVector(n));
                if (outVec == nullptr || outVec->size() < header_.block_m) {
                    return false;
                }
                if (current_.elem_bytes == 2) {
                    for (uint32_t row = 0; row < header_.block_m; ++row) {
                        const uint16_t bits = golem_float_to_fp16((*outVec)[row]);
                        std::memcpy(&tile[dst_off + static_cast<size_t>(row) * 2], &bits, sizeof(bits));
                    }
                } else {
                    std::memcpy(&tile[dst_off], outVec->data(), static_cast<size_t>(header_.block_m) * current_.elem_bytes);
                }
            } else {
                auto* outVec = static_cast<std::vector<int32_t>*>(array_->getOutputVector(n));
                if (outVec == nullptr || outVec->size() < header_.block_m) {
                    return false;
                }
                std::memcpy(&tile[dst_off], outVec->data(), static_cast<size_t>(header_.block_m) * current_.elem_bytes);
            }
        }
        return true;
    }

    size_t partialTileBytes() const {
        return static_cast<size_t>(header_.block_m) * static_cast<size_t>(header_.block_n) *
               static_cast<size_t>(header_.elem_bytes);
    }

    uint64_t partialCBufferOffset(size_t idx) const {
        return static_cast<uint64_t>(idx) * static_cast<uint64_t>(partialTileBytes());
    }

    uint64_t cBufferTransferCycles(size_t length, uint64_t bytesPerCycle) const {
        return length == 0 ? 0 : (static_cast<uint64_t>(length) + bytesPerCycle - 1) / bytesPerCycle;
    }

    uint64_t attentionTileStorageOffset(uint32_t row, uint32_t column) const {
        return (static_cast<uint64_t>(row) * attentionTileStorageColumns_ + column) *
            attentionTileStorageElemBytes_;
    }

    uint64_t attentionAccumulatorRowOffset(uint32_t row) const {
        return attentionAccumulatorOffset_ +
            static_cast<uint64_t>(row) * attentionAccumulatorRowBytes_;
    }

    bool attentionTileColumnWrite(
        uint32_t column, const std::vector<uint8_t>& values,
        uint64_t generation, uint64_t issueCycle, uint64_t& readyCycle) {
        if (cBufferMode_ != CBufferMode::ATTENTION_TILE_STORAGE ||
            generation != attentionTileStorageGeneration_ ||
            column >= attentionTileStorageColumns_ ||
            values.size() != static_cast<size_t>(attentionTileStorageRows_) *
                attentionTileStorageElemBytes_) {
            return false;
        }
        uint64_t startCycle = std::max(issueCycle, cBufferNextWriteCycle_);
        for (uint32_t row = 0; row < attentionTileStorageRows_; ++row) {
            const uint32_t bank = row % attentionTileStorageBanks_;
            startCycle = std::max(
                startCycle, attentionTileStorageBankNextWriteCycle_[bank]);
        }
        const uint64_t globalCycles = cBufferTransferCycles(
            values.size(), cBufferWriteBytesPerCycle_);
        const uint64_t maxRowsPerBank =
            (static_cast<uint64_t>(attentionTileStorageRows_) +
             attentionTileStorageBanks_ - 1) / attentionTileStorageBanks_;
        const size_t maxBankBytes = static_cast<size_t>(maxRowsPerBank) *
            attentionTileStorageElemBytes_;
        const uint64_t bankCycles = cBufferTransferCycles(
            maxBankBytes,
            attentionTileStorageBankBytesPerCycle_);
        const uint64_t transferCycles = std::max(globalCycles, bankCycles);
        const uint64_t transferEndCycle = startCycle + transferCycles;
        cBufferNextWriteCycle_ = transferEndCycle;
        readyCycle = transferEndCycle + cBufferLatencyCycles_;
        for (uint32_t row = 0; row < attentionTileStorageRows_; ++row) {
            const uint32_t bank = row % attentionTileStorageBanks_;
            attentionTileStorageBankNextWriteCycle_[bank] = transferEndCycle;
            attentionTileStorageRowWriteReadyCycle_[row] = std::max(
                attentionTileStorageRowWriteReadyCycle_[row], readyCycle);
            const uint64_t offset = attentionTileStorageOffset(row, column);
            const size_t sourceOffset =
                static_cast<size_t>(row) * attentionTileStorageElemBytes_;
            std::copy(
                values.begin() + sourceOffset,
                values.begin() + sourceOffset + attentionTileStorageElemBytes_,
                cBufferStorage_.begin() + offset);
        }
        attentionTileStorageColumnValid_[column] = 1;
        statAttentionTileStorageColumnWrites_->addData(1);
        statAttentionTileStorageWriteBytes_->addData(values.size());
        statAttentionTileStorageWriteWaitCycles_->addData(startCycle - issueCycle);
        return true;
    }

    bool attentionTileRowRead(
        uint32_t row, uint64_t generation, uint64_t issueCycle,
        std::vector<uint8_t>& data, uint64_t& readyCycle) {
        if (cBufferMode_ != CBufferMode::ATTENTION_TILE_STORAGE ||
            generation != attentionTileStorageGeneration_ ||
            row >= attentionTileStorageRows_ ||
            std::find(attentionTileStorageColumnValid_.begin(),
                      attentionTileStorageColumnValid_.end(), 0) !=
                attentionTileStorageColumnValid_.end()) {
            return false;
        }
        const uint32_t bank = row % attentionTileStorageBanks_;
        uint64_t startCycle = std::max(issueCycle, cBufferNextReadCycle_);
        startCycle = std::max(
            startCycle, attentionTileStorageBankNextReadCycle_[bank]);
        startCycle = std::max(
            startCycle, attentionTileStorageRowWriteReadyCycle_[row]);
        const size_t rowBytes = static_cast<size_t>(attentionTileStorageColumns_) *
            attentionTileStorageElemBytes_;
        const uint64_t globalCycles = cBufferTransferCycles(
            rowBytes, cBufferReadBytesPerCycle_);
        const uint64_t bankCycles = cBufferTransferCycles(
            rowBytes, attentionTileStorageBankBytesPerCycle_);
        const uint64_t transferCycles = std::max(globalCycles, bankCycles);
        const uint64_t transferEndCycle = startCycle + transferCycles;
        cBufferNextReadCycle_ = transferEndCycle;
        attentionTileStorageBankNextReadCycle_[bank] = transferEndCycle;
        readyCycle = transferEndCycle + cBufferLatencyCycles_;
        const uint64_t offset = attentionTileStorageOffset(row, 0);
        data.assign(
            cBufferStorage_.begin() + offset,
            cBufferStorage_.begin() + offset + rowBytes);
        attentionTileStorageRowRead_[row] = 1;
        statAttentionTileStorageRowReads_->addData(1);
        statAttentionTileStorageReadBytes_->addData(rowBytes);
        statAttentionTileStorageReadWaitCycles_->addData(startCycle - issueCycle);
        return true;
    }

    bool attentionAccumulatorRowWrite(
        uint32_t row, const std::vector<uint8_t>& values,
        uint64_t generation, uint64_t issueCycle, uint64_t& readyCycle) {
        if (!attentionStorageSessionActive_ ||
            cBufferMode_ != CBufferMode::ATTENTION_TILE_STORAGE ||
            generation != attentionStorageSessionGeneration_ ||
            row >= attentionAccumulatorRows_ ||
            values.size() != attentionAccumulatorRowBytes_) {
            return false;
        }
        const uint32_t bank = row % attentionTileStorageBanks_;
        uint64_t startCycle = std::max(issueCycle, cBufferNextWriteCycle_);
        startCycle = std::max(
            startCycle, attentionTileStorageBankNextWriteCycle_[bank]);
        const uint64_t transferCycles = std::max(
            cBufferTransferCycles(values.size(), cBufferWriteBytesPerCycle_),
            cBufferTransferCycles(
                values.size(), attentionTileStorageBankBytesPerCycle_));
        const uint64_t transferEndCycle = startCycle + transferCycles;
        cBufferNextWriteCycle_ = transferEndCycle;
        attentionTileStorageBankNextWriteCycle_[bank] = transferEndCycle;
        readyCycle = transferEndCycle + cBufferLatencyCycles_;
        const uint64_t offset = attentionAccumulatorRowOffset(row);
        std::copy(values.begin(), values.end(), cBufferStorage_.begin() + offset);
        attentionAccumulatorValid_[row] = 1;
        attentionAccumulatorWriteReadyCycle_[row] = readyCycle;
        statAttentionAccumulatorRowWrites_->addData(1);
        statAttentionAccumulatorWriteBytes_->addData(values.size());
        statAttentionAccumulatorWriteWaitCycles_->addData(startCycle - issueCycle);
        return true;
    }

    bool attentionAccumulatorRowRead(
        uint32_t row, uint64_t generation, uint64_t issueCycle,
        std::vector<uint8_t>& data, uint64_t& readyCycle) {
        if (!attentionStorageSessionActive_ ||
            cBufferMode_ != CBufferMode::ATTENTION_TILE_STORAGE ||
            generation != attentionStorageSessionGeneration_ ||
            row >= attentionAccumulatorRows_ ||
            attentionAccumulatorValid_[row] == 0) {
            return false;
        }
        const uint32_t bank = row % attentionTileStorageBanks_;
        uint64_t startCycle = std::max(issueCycle, cBufferNextReadCycle_);
        startCycle = std::max(
            startCycle, attentionTileStorageBankNextReadCycle_[bank]);
        startCycle = std::max(
            startCycle, attentionAccumulatorWriteReadyCycle_[row]);
        const uint64_t transferCycles = std::max(
            cBufferTransferCycles(attentionAccumulatorRowBytes_, cBufferReadBytesPerCycle_),
            cBufferTransferCycles(
                attentionAccumulatorRowBytes_,
                attentionTileStorageBankBytesPerCycle_));
        const uint64_t transferEndCycle = startCycle + transferCycles;
        cBufferNextReadCycle_ = transferEndCycle;
        attentionTileStorageBankNextReadCycle_[bank] = transferEndCycle;
        readyCycle = transferEndCycle + cBufferLatencyCycles_;
        const uint64_t offset = attentionAccumulatorRowOffset(row);
        data.assign(
            cBufferStorage_.begin() + offset,
            cBufferStorage_.begin() + offset + attentionAccumulatorRowBytes_);
        statAttentionAccumulatorRowReads_->addData(1);
        statAttentionAccumulatorReadBytes_->addData(attentionAccumulatorRowBytes_);
        statAttentionAccumulatorReadWaitCycles_->addData(startCycle - issueCycle);
        return true;
    }

    void clearAttentionTileMetadata() {
        attentionTileStorageActive_ = false;
        attentionTileStorageRows_ = 0;
        attentionTileStorageColumns_ = 0;
        attentionTileStorageElemBytes_ = 0;
        attentionTileStorageGeneration_ = 0;
        attentionTileStorageColumnValid_.clear();
        attentionTileStorageRowRead_.clear();
        attentionTileStorageRowWriteReadyCycle_.clear();
    }

    void clearAttentionTileStorage() {
        clearAttentionTileMetadata();
        cBufferMode_ = CBufferMode::FREE;
        attentionStorageSessionActive_ = false;
        attentionStorageSessionGeneration_ = 0;
        attentionStorageQkScratchBytes_ = 0;
        attentionAccumulatorOffset_ = 0;
        attentionAccumulatorRows_ = 0;
        attentionAccumulatorRowBytes_ = 0;
        attentionAccumulatorValid_.clear();
        attentionAccumulatorWriteReadyCycle_.clear();
        attentionTileStorageBankNextReadCycle_.clear();
        attentionTileStorageBankNextWriteCycle_.clear();
    }

    void purgeAttentionTileStorageCommands(uint64_t generation) {
        gemmProxyCommands_.erase(
            std::remove_if(
                gemmProxyCommands_.begin(), gemmProxyCommands_.end(),
                [generation](const GemmProxyCommand& command) {
                    const bool storageCommand =
                        command.kind == GemmProxyCommandKind::ATTENTION_TILE_COLUMN_WRITE ||
                        command.kind == GemmProxyCommandKind::ATTENTION_TILE_ROW_READ ||
                        command.kind == GemmProxyCommandKind::ATTENTION_ACCUMULATOR_ROW_WRITE ||
                        command.kind == GemmProxyCommandKind::ATTENTION_ACCUMULATOR_ROW_READ;
                    return storageCommand && command.storageGeneration == generation;
                }),
            gemmProxyCommands_.end());
        pendingAttentionStorageCompletions_.erase(
            std::remove_if(
                pendingAttentionStorageCompletions_.begin(),
                pendingAttentionStorageCompletions_.end(),
                [generation](const PendingAttentionStorageCompletion& completion) {
                    return completion.storageGeneration == generation;
                }),
            pendingAttentionStorageCompletions_.end());
    }

    bool cBufferRead(uint64_t offset, size_t length, uint64_t issueCycle,
                     std::vector<uint8_t>& data, uint64_t& readyCycle) {
        if (offset > cBufferBytes_ || length > cBufferBytes_ - offset ||
            cBufferReadBytesPerCycle_ == 0) {
            return false;
        }
        uint64_t startCycle = std::max(issueCycle, cBufferNextReadCycle_);
        const auto writeIt = cBufferWriteReadyCycles_.find(offset);
        if (writeIt != cBufferWriteReadyCycles_.end()) {
            startCycle = std::max(startCycle, writeIt->second);
        }
        cBufferNextReadCycle_ = startCycle + cBufferTransferCycles(length, cBufferReadBytesPerCycle_);
        readyCycle = cBufferNextReadCycle_ + cBufferLatencyCycles_;
        data.assign(cBufferStorage_.begin() + offset, cBufferStorage_.begin() + offset + length);
        return true;
    }

    bool cBufferWrite(uint64_t offset, const std::vector<uint8_t>& data,
                      uint64_t issueCycle, uint64_t& readyCycle) {
        if (offset > cBufferBytes_ || data.size() > cBufferBytes_ - offset ||
            cBufferWriteBytesPerCycle_ == 0) {
            return false;
        }
        const uint64_t startCycle = std::max(issueCycle, cBufferNextWriteCycle_);
        cBufferNextWriteCycle_ = startCycle +
            cBufferTransferCycles(data.size(), cBufferWriteBytesPerCycle_);
        readyCycle = cBufferNextWriteCycle_ + cBufferLatencyCycles_;
        std::copy(data.begin(), data.end(), cBufferStorage_.begin() + offset);
        cBufferWriteReadyCycles_[offset] = readyCycle;
        return true;
    }

    std::vector<CBufferPrefetch>::iterator findCBufferPrefetch(
        uint32_t targetKBegin, size_t partialIndex) {
        return std::find_if(
            cBufferPrefetches_.begin(), cBufferPrefetches_.end(),
            [targetKBegin, partialIndex](const CBufferPrefetch& entry) {
                return entry.targetKBegin == targetKBegin && entry.partialIndex == partialIndex;
            });
    }

    bool issueCBufferPrefetch(uint32_t targetKBegin, size_t idx, uint64_t cycle) {
        if (idx >= partialValid_.size() || partialValid_[idx] == 0) {
            return false;
        }
        if (findCBufferPrefetch(targetKBegin, idx) != cBufferPrefetches_.end()) {
            return true;
        }
        cBufferPrefetches_.erase(
            std::remove_if(
                cBufferPrefetches_.begin(), cBufferPrefetches_.end(),
                [this](const CBufferPrefetch& entry) {
                    return entry.targetKBegin < activeWindowKBegin_;
                }),
            cBufferPrefetches_.end());
        if (cBufferPrefetches_.size() >= 2) {
            if (targetKBegin != activeWindowKBegin_) {
                return false;
            }
            auto speculative = std::find_if(
                cBufferPrefetches_.begin(), cBufferPrefetches_.end(),
                [this, idx](const CBufferPrefetch& entry) {
                    return entry.targetKBegin == activeWindowKBegin_ &&
                           entry.partialIndex != idx;
                });
            if (speculative == cBufferPrefetches_.end()) {
                return false;
            }
            cBufferPrefetches_.erase(speculative);
        }
        CBufferPrefetch entry{};
        entry.targetKBegin = targetKBegin;
        entry.partialIndex = idx;
        if (!cBufferRead(
                partialCBufferOffset(idx), partialTileBytes(), cycle,
                entry.data, entry.readyCycle)) {
            if (extOutput_ != nullptr) {
                extOutput_->output(
                    "[Core %u] [wcp] ERROR: C-buffer read out of range: idx=%zu bytes=%zu capacity=%" PRIu64 "\n",
                    coreId_, idx, partialTileBytes(), cBufferBytes_);
            }
            phase_ = Phase::DONE;
            return false;
        }
        cBufferReadCount_++;
        cBufferPrefetches_.push_back(std::move(entry));
        return true;
    }

    bool loadPartialCToArray(const std::vector<uint8_t>& tile) {
        if (tile.size() != partialTileBytes()) {
            return false;
        }
        for (uint32_t n = 0; n < current_.block_n; ++n) {
            const size_t src_off = static_cast<size_t>(n) * header_.block_m * current_.elem_bytes;
            if (outputIsFloat_) {
                auto* outVec = static_cast<std::vector<float>*>(array_->getOutputVector(n));
                if (outVec == nullptr || outVec->size() < header_.block_m) {
                    return false;
                }
                if (current_.elem_bytes == 2) {
                    for (uint32_t row = 0; row < header_.block_m; ++row) {
                        uint16_t bits = 0;
                        std::memcpy(&bits, &tile[src_off + static_cast<size_t>(row) * 2], sizeof(bits));
                        (*outVec)[row] = golem_fp16_to_float(bits);
                    }
                } else {
                    std::memcpy(outVec->data(), &tile[src_off], static_cast<size_t>(header_.block_m) * current_.elem_bytes);
                }
            } else {
                auto* outVec = static_cast<std::vector<int32_t>*>(array_->getOutputVector(n));
                if (outVec == nullptr || outVec->size() < header_.block_m) {
                    return false;
                }
                std::memcpy(outVec->data(), &tile[src_off], static_cast<size_t>(header_.block_m) * current_.elem_bytes);
            }
        }
        return true;
    }

    bool loadCurrentPartialCWhenReady(uint64_t cycle) {
        const size_t idx = currentPartialIndex();
        auto entry = findCBufferPrefetch(activeWindowKBegin_, idx);
        if (entry == cBufferPrefetches_.end()) {
            if (!issueCBufferPrefetch(activeWindowKBegin_, idx, cycle)) {
                return false;
            }
            entry = findCBufferPrefetch(activeWindowKBegin_, idx);
        }
        if (entry == cBufferPrefetches_.end() || cycle < entry->readyCycle) {
            if (lastCBufferWaitCycle_ != cycle) {
                cBufferReadWaitCycles_++;
                lastCBufferWaitCycle_ = cycle;
            }
            return false;
        }
        if (!loadPartialCToArray(entry->data)) {
            phase_ = Phase::DONE;
            return false;
        }
        cBufferPrefetches_.erase(entry);
        return true;
    }

    void prefetchNextPartialForCurrentWindow(uint64_t cycle) {
        if (!use2DWindowEngine() || activeWindowKBegin_ == 0 || activeMicroKStep_ != 0) {
            return;
        }
        uint32_t nextM = reuseMIndex_;
        uint32_t nextN = reuseNIndex_;
        if (selectNextReuseForActiveWindow(nextM, nextN) &&
            (nextM != reuseMIndex_ || nextN != reuseNIndex_)) {
            issueCBufferPrefetch(activeWindowKBegin_, reuseIndex(nextM, nextN), cycle);
        }
    }

    void prefetchFirstPartialForNextWindow(uint64_t cycle) {
        const uint32_t nextKBegin = activeWindowKBegin_ + activeWindowKCount_;
        if (!use2DWindowEngine() || nextKBegin >= totalKTileCount_) {
            return;
        }
        const size_t current = currentPartialIndex();
        for (size_t idx = 0; idx < windowReuseDone_.size(); ++idx) {
            if (idx != current && windowReuseDone_[idx] == 0) {
                return;
            }
        }
        issueCBufferPrefetch(nextKBegin, 0, cycle);
    }

    bool savePartialCFromArray(uint64_t cycle) {
        const size_t idx = currentPartialIndex();
        if (idx >= partialValid_.size()) {
            return false;
        }
        std::vector<uint8_t> tile;
        if (!captureArrayOutput(tile)) {
            return false;
        }
        uint64_t readyCycle = 0;
        if (!cBufferWrite(partialCBufferOffset(idx), tile, cycle, readyCycle)) {
            if (extOutput_ != nullptr) {
                extOutput_->output(
                    "[Core %u] [wcp] ERROR: C-buffer write out of range: idx=%zu bytes=%zu capacity=%" PRIu64 "\n",
                    coreId_, idx, tile.size(), cBufferBytes_);
            }
            phase_ = Phase::DONE;
            return false;
        }
        cBufferWriteCount_++;
        partialValid_[idx] = 1;
        return true;
    }

    bool isFinal2DWindow() const {
        return !use2DWindowEngine() || (activeWindowKBegin_ + activeWindowKCount_ >= totalKTileCount_);
    }

    void resetComputeOnlyStateForNextTile() {
        allTilesScheduled_ = false;
        computeInFlight_ = false;
        activeComputeTileIndex_ = -1;
        activeComputeSlotIndex_ = -1;
        pendingArrays_ = 0;
        activeTxnRetiredTileCount_ = 0;
        activeTxnTileRetired_.assign(activeWindowKCount_, 0);
        nextReadyScanCursor_ = 0;
        activeMicroKStep_ = 0;
        taskAccumInitialized_ = false;
        activeMatPayload_.clear();
        activeVecPayload_.clear();
        activeTileMicroOps_.clear();
        activeTileMicroOpCursor_ = 0;
        activeTileScoreboard_ = KStepScoreboard{};
        activeComputeReadyQueue_.clear();
        activeMicroOpIssued_ = false;
        activeIssuedMicroOp_ = MicroOp{};
        activeTilePayloadLoaded_ = false;
        tileComputeStartCycles_.assign(activeWindowKCount_, 0);
        tileComputeDoneCycles_.assign(activeWindowKCount_, 0);
        tileRetireCycles_.assign(activeWindowKCount_, 0);
        tileComputeStartSchedCycles_.assign(activeWindowKCount_, 0);
        tileComputeDoneSchedCycles_.assign(activeWindowKCount_, 0);
        tileRetireSchedCycles_.assign(activeWindowKCount_, 0);
    }

    bool advance2DWindowEngine() {
        traceStage3("ADVANCE_ENTER", lastAccountCycle_);
        if (activeWindowValid_ && !allReuseDoneForWindow()) {
            uint32_t nextM = reuseMIndex_;
            uint32_t nextN = reuseNIndex_;
            if (selectNextReuseForActiveWindow(nextM, nextN)) {
                reuseMIndex_ = nextM;
                reuseNIndex_ = nextN;
                deriveTask(taskIndex_, false);
                resetComputeOnlyStateForNextTile();
                allTilesScheduled_ = activeWindowValid_;
                traceStage3("ADVANCE_READY_REUSE", lastAccountCycle_);
                return true;
            }
        }
        if (activeWindowValid_ && allReuseDoneForWindow()) {
            if (activeWindowTxnId_ != 0) {
                markWindowComputeEnd(activeWindowTxnId_, lastAccountCycle_);
            }
            reuseNIndex_ = 0;
            reuseMIndex_ = 0;
        } else {
        if ((reuseNIndex_ + 1) < currentReuseNCount_) {
            reuseNIndex_ += 1;
            deriveTask(taskIndex_, false);
            resetComputeOnlyStateForNextTile();
            allTilesScheduled_ = activeWindowValid_;
            traceStage3("ADVANCE_REUSE_N", lastAccountCycle_);
            return true;
        }
        if ((reuseMIndex_ + 1) < currentReuseMCount_) {
            reuseMIndex_ += 1;
            reuseNIndex_ = 0;
            deriveTask(taskIndex_, false);
            resetComputeOnlyStateForNextTile();
            allTilesScheduled_ = activeWindowValid_;
            traceStage3("ADVANCE_REUSE_M", lastAccountCycle_);
            return true;
        }
        }
        reuseNIndex_ = 0;
        reuseMIndex_ = 0;
        if (isFinal2DWindow()) {
            if (activeTxnId_ != 0) {
                retireReady2DTransactions(active2DTxnIds_, twoDWindowTransactionTileCount(activeWindowKCount_), active2DSchedulerTileRetired_);
                if (activeTransactionsDone()) {
                    retireActiveTransactions();
                    activeTxnId_ = 0;
                    active2DTxnIds_.clear();
                    active2DSchedulerTileRetired_.clear();
                }
            }
            traceStage3("ADVANCE_FINAL_WINDOW_DONE", lastAccountCycle_);
            return false;
        }
        const uint64_t completedWindowTxnId = activeWindowTxnId_;
        previousWindowTxnId_ = completedWindowTxnId;
        if (!prefetch2DWindows_.empty()) {
            Prefetch2DWindow nextWindow = std::move(prefetch2DWindows_.front());
            prefetch2DWindows_.pop_front();
            if (!are2DTransactionsReady(nextWindow.txnIds, twoDWindowTransactionTileCount(nextWindow.kCount), false)) {
                windowAdvanceWaitPrefetchCount_++;
                traceStage3("ADVANCE_WAIT_PREFETCH", lastAccountCycle_);
                activeWindowValid_ = false;
                activeTxnId_ = nextWindow.txnId;
                active2DTxnIds_ = std::move(nextWindow.txnIds);
                activeTxnTileCount_ = nextWindow.kCount;
                activeWindowKBegin_ = nextWindow.kBegin;
                activeWindowKCount_ = nextWindow.kCount;
                activeWindowBuffer_ = nextWindow.buffer;
                active2DSchedulerTileRetired_.assign(twoDWindowTransactionTileCount(activeWindowKCount_), 0);
                windowReuseDone_.assign(static_cast<size_t>(currentReuseMCount_) * static_cast<size_t>(currentReuseNCount_), 0);
                deriveTask(taskIndex_, false);
                resetComputeOnlyStateForNextTile();
                return true;
            }
            retire2DTransactions(nextWindow.txnIds, twoDWindowTransactionTileCount(nextWindow.kCount));
            updateWindowTimeline(nextWindow.txnId, nextWindow.txnIds);
            markWindowActivated(nextWindow.txnId, lastAccountCycle_);
            activeWindowTxnId_ = nextWindow.txnId;
            activeTxnId_ = 0;
            active2DTxnIds_.clear();
            active2DSchedulerTileRetired_.clear();
            traceStage3("ADVANCE_PREFETCH_READY", lastAccountCycle_);
            activate2DWindow(nextWindow.kBegin, nextWindow.kCount, nextWindow.buffer);
        } else {
            traceStage3("ADVANCE_NO_PREFETCH", lastAccountCycle_);
            const uint32_t nextK = activeWindowKBegin_ + activeWindowKCount_;
            const uint32_t nextCount = std::min<uint32_t>(residentKTileCount_, totalKTileCount_ - nextK);
            activeWindowValid_ = false;
            activeWindowKBegin_ = nextK;
            activeWindowKCount_ = nextCount;
            uint32_t buffer = 0;
            if (allocatePrefetchWindowBuffer(buffer)) {
                activeWindowBuffer_ = buffer;
            } else {
                activeWindowBuffer_ = (activeWindowBuffer_ + 1u) % twoDWindowBufferCount();
            }
            windowReuseDone_.assign(static_cast<size_t>(currentReuseMCount_) * static_cast<size_t>(currentReuseNCount_), 0);
            next2DPrefetchK_ = nextK + nextCount;
        }
        deriveTask(taskIndex_, false);
        resetComputeOnlyStateForNextTile();
        allTilesScheduled_ = activeWindowValid_;
        return true;
    }

    bool completeActiveMicroTile() {
        if (!activeMicroOpIssued_ || activeTileMicroOpCursor_ >= activeTileMicroOps_.size()) {
            return false;
        }

        const MicroOp& expectedOp = activeTileMicroOps_[activeTileMicroOpCursor_];
        if (expectedOp.kStep != activeMicroKStep_ ||
            expectedOp.kStep != activeIssuedMicroOp_.kStep ||
            expectedOp.logicalTileIdx != activeIssuedMicroOp_.logicalTileIdx) {
            return false;
        }

        activeTileScoreboard_.lastCompletedKStep = static_cast<int32_t>(activeIssuedMicroOp_.kStep);
        activeMicroOpIssued_ = false;
        activeIssuedMicroOp_ = MicroOp{};
        activeTileMicroOpCursor_ += 1;
        refreshActiveComputeReadyQueue();
        if (activeTileMicroOpCursor_ < activeTileMicroOps_.size()) {
            activeTilePayloadLoaded_ = false;
            return true;
        }
        activeMicroKStep_ = 0;

        const int doneTile = activeComputeTileIndex_;
        const int doneSlot = activeComputeSlotIndex_;
        if (doneTile >= 0 && static_cast<size_t>(doneTile) < tileComputeDoneCycles_.size()) {
            tileComputeDoneCycles_[static_cast<size_t>(doneTile)] = lastAccountCycle_;
            if (static_cast<size_t>(doneTile) < tileComputeDoneSchedCycles_.size()) {
                tileComputeDoneSchedCycles_[static_cast<size_t>(doneTile)] = schedulerTimelineCycle();
            }
        }
        if (requestScheduler_ != nullptr && activeTxnId_ != 0) {
            if (use2DWindowEngine()) {
                retireReady2DTransactions(active2DTxnIds_, twoDWindowTransactionTileCount(activeWindowKCount_), active2DSchedulerTileRetired_);
            } else {
                WcpTileTimelineDebug dbg{};
                const bool haveDbg = requestScheduler_->getTileTimeline(activeTxnId_, static_cast<uint32_t>(doneTile), dbg);
                requestScheduler_->retireTileReady(activeTxnId_, static_cast<uint32_t>(doneTile));
                if (doneTile >= 0 && static_cast<size_t>(doneTile) < activeTxnTileRetired_.size() &&
                    activeTxnTileRetired_[static_cast<size_t>(doneTile)] == 0) {
                    activeTxnTileRetired_[static_cast<size_t>(doneTile)] = 1;
                    activeTxnRetiredTileCount_ += 1;
                }
                if (doneTile >= 0 && static_cast<size_t>(doneTile) < tileRetireCycles_.size()) {
                    tileRetireCycles_[static_cast<size_t>(doneTile)] = lastAccountCycle_;
                    if (static_cast<size_t>(doneTile) < tileRetireSchedCycles_.size()) {
                        tileRetireSchedCycles_[static_cast<size_t>(doneTile)] = schedulerTimelineCycle();
                    }
                }
                if (haveDbg && extOutput_ != nullptr && doneTile >= 0 && static_cast<size_t>(doneTile) < tileRetireCycles_.size()) {
                    extOutput_->output(
                        "[Core %u] [wcp] TILE TRACE: task=%" PRIu64 " tile=%d slot=%d submit=%" PRIu64
                        " mat_done=%" PRIu64 " vec_done=%" PRIu64 " ready=%" PRIu64
                        " compute_start=%" PRIu64 " compute_done=%" PRIu64 " retire=%" PRIu64
                        " compute_start_sched=%" PRIu64 " compute_done_sched=%" PRIu64 " retire_sched=%" PRIu64 "\n",
                        coreId_, current_.task_id, doneTile, doneSlot,
                        dbg.submitCycle, dbg.matDoneCycle, dbg.vecDoneCycle, dbg.readyCycle,
                        tileComputeStartCycles_[static_cast<size_t>(doneTile)],
                        tileComputeDoneCycles_[static_cast<size_t>(doneTile)],
                        tileRetireCycles_[static_cast<size_t>(doneTile)],
                        tileComputeStartSchedCycles_[static_cast<size_t>(doneTile)],
                        tileComputeDoneSchedCycles_[static_cast<size_t>(doneTile)],
                        tileRetireSchedCycles_[static_cast<size_t>(doneTile)]);
                }
            }
        }
        if (use2DWindowEngine() && activeWindowValid_) {
            if (doneTile >= 0 && static_cast<size_t>(doneTile) < activeTxnTileRetired_.size() &&
                activeTxnTileRetired_[static_cast<size_t>(doneTile)] == 0) {
                activeTxnTileRetired_[static_cast<size_t>(doneTile)] = 1;
                activeTxnRetiredTileCount_ += 1;
            }
        } else if (!use2DWindowEngine() && is2DReuse() && groupResidentValid_ && residentMacroTaskId_ == currentMacroTaskId_) {
            if (doneTile >= 0 && static_cast<size_t>(doneTile) < activeTxnTileRetired_.size() &&
                activeTxnTileRetired_[static_cast<size_t>(doneTile)] == 0) {
                activeTxnTileRetired_[static_cast<size_t>(doneTile)] = 1;
                activeTxnRetiredTileCount_ += 1;
            }
        }
        if (activeComputeSlotIndex_ >= 0 && activeComputeSlotIndex_ < static_cast<int>(std::max<uint32_t>(header_.local_slot_count, 1u))) {
            if (activeComputeSlotIndex_ < static_cast<int>(buffers_.size())) {
                buffers_[activeComputeSlotIndex_].in_use = false;
            }
        }
        activeComputeTileIndex_ = -1;
        activeComputeSlotIndex_ = -1;
        activeMicroKStep_ = 0;
        activeMatPayload_.clear();
        activeVecPayload_.clear();
        activeTileMicroOps_.clear();
        activeTileMicroOpCursor_ = 0;
        activeTileScoreboard_ = KStepScoreboard{};
        activeComputeReadyQueue_.clear();
        activeMicroOpIssued_ = false;
        activeIssuedMicroOp_ = MicroOp{};
        activeTilePayloadLoaded_ = false;
        return true;
    }

    uint32_t microKStepCount() const {
        const uint32_t inputSize = std::max<uint32_t>(current_.array_input_size, 1u);
        return std::max<uint32_t>((header_.block_k + inputSize - 1u) / inputSize, 1u);
    }

    uint64_t schedulerTimelineCycle() const {
        if (requestScheduler_ == nullptr) {
            return 0;
        }
        return requestScheduler_->getTimelineCycle();
    }

    double decodeElement(const uint8_t* raw) const {
        if (outputIsFloat_) {
            if (current_.elem_bytes == 2) {
                uint16_t bits = 0;
                std::memcpy(&bits, raw, sizeof(bits));
                return static_cast<double>(golem_fp16_to_float(bits));
            }
            float value = 0.0f;
            std::memcpy(&value, raw, sizeof(value));
            return static_cast<double>(value);
        }

        int32_t value = 0;
        std::memcpy(&value, raw, sizeof(value));
        return static_cast<double>(value);
    }

    void deriveTask(uint32_t taskIndex, bool resetState = true) {
        const uint32_t macro_task_id = header_.worker_slot + taskIndex * header_.active_worker_cores;
        const uint32_t m_tiles = header_.m / header_.block_m;
        const uint32_t n_tiles = header_.n / header_.block_n;
        const uint32_t k_tiles = header_.k / header_.block_k;
        const uint32_t reuseN = std::max<uint32_t>(header_.a_reuse_n_tiles, 1u);
        const uint32_t reuseM = std::max<uint32_t>(header_.b_reuse_m_tiles, 1u);
        uint32_t m_tile = 0;
        uint32_t n_tile = 0;
        const uint32_t m_groups = header_.m_group_count != 0
                                      ? header_.m_group_count
                                      : ((m_tiles + reuseM - 1) / reuseM);
        const uint32_t n_groups = header_.n_group_count != 0
                                      ? header_.n_group_count
                                      : ((n_tiles + reuseN - 1) / reuseN);
        const uint32_t m_group = macro_task_id % m_groups;
        const uint32_t n_group = ((macro_task_id / m_groups) + m_group) % n_groups;
        const uint32_t m_begin = m_group * reuseM;
        const uint32_t n_begin = n_group * reuseN;
        currentReuseMCount_ = std::min<uint32_t>(reuseM, m_tiles - m_begin);
        currentReuseNCount_ = std::min<uint32_t>(reuseN, n_tiles - n_begin);
        if (reuseM == 1) {
            reuseMIndex_ = 0;
        }
        if (reuseN == 1) {
            reuseNIndex_ = 0;
        }
        if (reuseMIndex_ >= currentReuseMCount_) {
            reuseMIndex_ = 0;
        }
        if (reuseNIndex_ >= currentReuseNCount_) {
            reuseNIndex_ = 0;
        }
        m_tile = m_begin + reuseMIndex_;
        n_tile = n_begin + reuseNIndex_;
        const uint64_t output_task_id = static_cast<uint64_t>(m_tile) * n_tiles + n_tile;
        const uint32_t activeWorkers = std::max<uint32_t>(header_.active_worker_cores, 1u);
        auto nodeForMacroTask = [&](uint32_t macroTaskId) -> uint32_t {
            const uint32_t owner_slot = macroTaskId % activeWorkers;
            const uint32_t group_id = header_.total_groups > 0 ? (owner_slot % header_.total_groups) : 0;
            const uint32_t dataMapKey = header_.data_node_map_mode != 0 ? owner_slot : group_id;
            return 1 + (header_.data_memory_node_count > 0 ? (dataMapKey % header_.data_memory_node_count) : 0);
        };
        auto macroSlotInNode = [&](uint32_t macroTaskId, uint32_t nodeIdx) -> uint32_t {
            uint32_t slot = 0;
            for (uint32_t i = 0; i < macroTaskId; ++i) {
                if (nodeForMacroTask(i) == nodeIdx) {
                    slot++;
                }
            }
            return slot;
        };
        const uint32_t c_node_idx = nodeForMacroTask(macro_task_id);
        const uint32_t macro_slot = macroSlotInNode(macro_task_id, c_node_idx);
        const uint32_t a_owner_macro = m_group;
        const uint32_t b_owner_macro = n_group;
        const uint32_t a_node_idx = nodeForMacroTask(a_owner_macro);
        const uint32_t b_node_idx = nodeForMacroTask(b_owner_macro);
        uint32_t a_slot = 0;
        for (uint32_t mt = 0; mt < m_tile; ++mt) {
            const uint32_t mt_group = mt / reuseM;
            if (nodeForMacroTask(mt_group) == a_node_idx) {
                a_slot++;
            }
        }
        uint32_t b_slot = 0;
        for (uint32_t nt = 0; nt < n_tile; ++nt) {
            const uint32_t nt_group = nt / reuseN;
            if (nodeForMacroTask(nt_group) == b_node_idx) {
                b_slot++;
            }
        }
        const uint64_t a_node_base = static_cast<uint64_t>(a_node_idx) * header_.mem_node_size;
        const uint64_t b_node_base = static_cast<uint64_t>(b_node_idx) * header_.mem_node_size;
        const uint64_t c_node_base = static_cast<uint64_t>(c_node_idx) * header_.mem_node_size;
        const uint64_t mat_group_slot = static_cast<uint64_t>(a_slot);
        const uint64_t vec_group_slot = static_cast<uint64_t>(b_slot);
        const uint64_t out_group_slot =
            static_cast<uint64_t>(macro_slot) * (reuseM > 1 ? reuseM : 1) * (reuseN > 1 ? reuseN : 1) +
            static_cast<uint64_t>(reuseMIndex_) * (reuseN > 1 ? reuseN : 1) + static_cast<uint64_t>(reuseNIndex_);
        current_.task_id = output_task_id;
        current_.task_flags = 0;
        current_.mat_base_addr = a_node_base + header_.off_gemm_mat_base + mat_group_slot * static_cast<uint64_t>(k_tiles) * header_.mat_stride_bytes;
        current_.vec_base_addr = b_node_base + header_.off_gemm_vec_base + vec_group_slot * static_cast<uint64_t>(k_tiles * header_.block_n) * header_.vec_stride_bytes;
        const uint64_t out_bytes = static_cast<uint64_t>(header_.block_m) * static_cast<uint64_t>(header_.block_n) * static_cast<uint64_t>(header_.elem_bytes);
        const uint64_t out_stride = ((out_bytes + 0xffULL) / 0x100ULL) * 0x100ULL;
        current_.c_base_addr = c_node_base + header_.off_gemm_out_base + out_group_slot * out_stride;
        current_.accum_base_addr = header_.local_accum_gm_addr;
        current_.completion_flag_addr = 0;
        current_.completion_value = 0;
        current_.k_begin = (!resetState && use2DWindowEngine()) ? activeWindowKBegin_ : 0;
        current_.k_count = (!resetState && use2DWindowEngine() && activeWindowKCount_ != 0) ? activeWindowKCount_ : k_tiles;
        current_.block_n = header_.block_n;
        current_.hw_input_size = header_.hw_input_size;
        current_.hw_output_size = header_.hw_output_size;
        current_.array_input_size = header_.hw_input_size;
        current_.array_output_size = header_.hw_output_size;
        current_.elem_bytes = header_.elem_bytes;
        current_.mat_stride_bytes = header_.mat_stride_bytes;
        current_.vec_stride_bytes = static_cast<uint64_t>(header_.block_n) * header_.vec_stride_bytes;
        current_.local_accum_gm_addr = header_.local_accum_gm_addr;
        current_.local_out_gm_addr = header_.local_out_gm_addr;
        currentK_ = current_.k_begin;
        currentMacroTaskId_ = macro_task_id;
        if (resetState) {
            resetPipelineState();
        }
    }

    enum class Phase : uint8_t {
        IDLE = 0,
        RUN,
        WRITEBACK,
        WRITEBACK_WAIT,
        DONE,
    };

    int verbose_ = 0;
    bool outputIsFloat_ = false;
    bool stage3Trace_ = false;
    uint32_t attentionClusterQkArrays_ = 16;
    uint32_t prefetchWindowDepth_ = 1;
    bool crossMacroPrefetch_ = false;
    uint32_t windowKtiles_ = 4;
    uint64_t cBufferBytes_ = 0;
    uint64_t cBufferReadBytesPerCycle_ = 256;
    uint64_t cBufferWriteBytesPerCycle_ = 256;
    uint64_t cBufferLatencyCycles_ = 1;
    uint32_t attentionTileStorageBanks_ = 16;
    uint64_t attentionTileStorageBankBytesPerCycle_ = 64;
    uint32_t gemmProxyQueueDepth_ = 32;
    uint32_t gemmProxyIssueWidth_ = 1;
    uint64_t gemmProxyCommandLatencyCycles_ = 1;
    uint64_t gemmProxyCompletionLatencyCycles_ = 1;
    bool finalCWriteEnable_ = true;
    std::string outputMode_ = "hbm";
    bool fusionDumpEnable_ = false;
    std::string fusionDumpDir_;
    std::ofstream fusionDump_;
    uint64_t cBufferNextReadCycle_ = 0;
    uint64_t cBufferNextWriteCycle_ = 0;
    std::vector<uint8_t> cBufferStorage_;
    std::unordered_map<uint64_t, uint64_t> cBufferWriteReadyCycles_;
    CBufferMode cBufferMode_ = CBufferMode::FREE;
    bool attentionStorageSessionActive_ = false;
    bool attentionTileStorageActive_ = false;
    uint64_t attentionStorageSessionGeneration_ = 0;
    uint64_t attentionStorageQkScratchBytes_ = 0;
    uint64_t attentionAccumulatorOffset_ = 0;
    uint32_t attentionAccumulatorRows_ = 0;
    size_t attentionAccumulatorRowBytes_ = 0;
    std::vector<uint8_t> attentionAccumulatorValid_;
    std::vector<uint64_t> attentionAccumulatorWriteReadyCycle_;
    uint32_t attentionTileStorageRows_ = 0;
    uint32_t attentionTileStorageColumns_ = 0;
    size_t attentionTileStorageElemBytes_ = 0;
    uint64_t attentionTileStorageGeneration_ = 0;
    std::vector<uint8_t> attentionTileStorageColumnValid_;
    std::vector<uint8_t> attentionTileStorageRowRead_;
    std::vector<uint64_t> attentionTileStorageRowWriteReadyCycle_;
    std::vector<uint64_t> attentionTileStorageBankNextReadCycle_;
    std::vector<uint64_t> attentionTileStorageBankNextWriteCycle_;
    SST::Output output_;
    uint32_t coreId_ = 0;
    SST::Output* extOutput_ = nullptr;
    SST::Golem::GlobalMemoryAPI* globalMem_ = nullptr;
    SST::Golem::ComputeArray* array_ = nullptr;
    SST::Golem::RequestSchedulerAPI* requestScheduler_ = nullptr;
    std::deque<GemmProxyCommand> gemmProxyCommands_;
    std::unordered_map<uint32_t, GemmArrayDoneCallback> gemmArrayDoneCallbacks_;
    std::deque<PendingGemmCompletion> pendingGemmCompletions_;
    std::deque<PendingAttentionStorageCompletion> pendingAttentionStorageCompletions_;
    Statistic<uint64_t>* statGemmProxyCommandsIssued_ = nullptr;
    Statistic<uint64_t>* statGemmProxyQueueFullStalls_ = nullptr;
    Statistic<uint64_t>* statGemmProxyQueueWaitCycles_ = nullptr;
    Statistic<uint64_t>* statGemmProxyLaunchCommands_ = nullptr;
    Statistic<uint64_t>* statGemmProxyCompletionCallbacks_ = nullptr;
    Statistic<uint64_t>* statGemmProxyCompletionDelayCycles_ = nullptr;
    Statistic<uint64_t>* statAttentionTileStorageAcquires_ = nullptr;
    Statistic<uint64_t>* statAttentionTileStorageReleases_ = nullptr;
    Statistic<uint64_t>* statAttentionTileStorageModeConflicts_ = nullptr;
    Statistic<uint64_t>* statAttentionTileStorageCapacityRejections_ = nullptr;
    Statistic<uint64_t>* statAttentionTileStorageColumnWrites_ = nullptr;
    Statistic<uint64_t>* statAttentionTileStorageRowReads_ = nullptr;
    Statistic<uint64_t>* statAttentionTileStorageWriteBytes_ = nullptr;
    Statistic<uint64_t>* statAttentionTileStorageReadBytes_ = nullptr;
    Statistic<uint64_t>* statAttentionTileStorageWriteWaitCycles_ = nullptr;
    Statistic<uint64_t>* statAttentionTileStorageReadWaitCycles_ = nullptr;
    Statistic<uint64_t>* statAttentionStorageSessionAcquires_ = nullptr;
    Statistic<uint64_t>* statAttentionStorageSessionReleases_ = nullptr;
    Statistic<uint64_t>* statAttentionAccumulatorRowWrites_ = nullptr;
    Statistic<uint64_t>* statAttentionAccumulatorRowReads_ = nullptr;
    Statistic<uint64_t>* statAttentionAccumulatorWriteBytes_ = nullptr;
    Statistic<uint64_t>* statAttentionAccumulatorReadBytes_ = nullptr;
    Statistic<uint64_t>* statAttentionAccumulatorWriteWaitCycles_ = nullptr;
    Statistic<uint64_t>* statAttentionAccumulatorReadWaitCycles_ = nullptr;
    WorkerTaskListHeader header_{};
    WorkerWindowDescriptor current_{};
    bool busy_ = false;
    Phase phase_ = Phase::IDLE;
    uint32_t currentK_ = 0;
    uint32_t nextPrefetchK_ = 0;
    uint32_t taskIndex_ = 0;
    uint32_t pendingArrays_ = 0;
    uint32_t currentWindowId_ = 0;
    uint32_t activeTxnTileCount_ = 0;
    uint32_t nextTxnComputeTile_ = 0;
    uint32_t activeTxnRetiredTileCount_ = 0;
    uint32_t nextReadyScanCursor_ = 0;
    uint32_t activeTxnKBegin_ = 0;
    uint32_t reuseNIndex_ = 0;
    uint32_t reuseMIndex_ = 0;
    uint32_t currentReuseNCount_ = 1;
    uint32_t currentReuseMCount_ = 1;
    uint32_t currentMacroTaskId_ = 0;
    uint32_t residentMacroTaskId_ = UINT32_MAX;
    bool groupResidentValid_ = false;
    uint32_t totalKTileCount_ = 0;
    uint32_t residentKTileCount_ = 0;
    uint32_t activeWindowKBegin_ = 0;
    uint32_t activeWindowKCount_ = 0;
    uint32_t activeWindowBuffer_ = 0;
    bool activeWindowValid_ = false;
    uint32_t next2DPrefetchK_ = 0;
    uint64_t previousWindowTxnId_ = 0;
    uint64_t activeWindowTxnId_ = 0;
    std::vector<uint64_t> active2DTxnIds_;
    std::deque<Prefetch2DWindow> prefetch2DWindows_;
    bool nextMacroPrefetchValid_ = false;
    uint32_t nextMacroPrefetchTaskIndex_ = 0;
    Prefetch2DWindow nextMacroPrefetch_{};
    std::vector<WindowTimeline> windowTimelines_;
    std::vector<uint8_t> active2DSchedulerTileRetired_;
    uint64_t lastAccountCycle_ = 0;
    uint64_t workerStartCycle_ = 0;
    uint64_t workerEndCycle_ = 0;
    uint64_t totalWindowCycles_ = 0;
    uint64_t computeCycles_ = 0;
    uint64_t tileReadyWaitCycles_ = 0;
    uint64_t txnWaitCycles_ = 0;
    uint64_t writebackWaitCycles_ = 0;
    uint64_t wait2DActivateCycles_ = 0;
    uint64_t wait2DActiveNotReadyCycles_ = 0;
    uint64_t waitNon2DTxnCycles_ = 0;
    uint64_t waitNoActiveTxnCycles_ = 0;
    uint64_t windowSubmitActiveCount_ = 0;
    uint64_t windowSubmitPrefetchCount_ = 0;
    uint64_t windowActivateCount_ = 0;
    uint64_t windowAdvanceWaitPrefetchCount_ = 0;
    uint64_t crossMacroPrefetchSubmitCount_ = 0;
    uint64_t crossMacroPrefetchAdoptCount_ = 0;
    uint64_t cBufferReadCount_ = 0;
    uint64_t cBufferWriteCount_ = 0;
    uint64_t cBufferReadWaitCycles_ = 0;
    uint64_t fusionConsumedCount_ = 0;
    uint64_t lastCBufferWaitCycle_ = UINT64_MAX;
    uint64_t lastStage3TraceCycle_ = 0;
    bool allTilesScheduled_ = false;
    bool computeInFlight_ = false;
    bool writebackDone_ = false;
    uint64_t activeTxnId_ = 0;
    uint64_t writebackToken_ = 0;
    std::deque<uint64_t> pendingWritebackTokens_;
    int activeComputeTileIndex_ = -1;
    int activeComputeSlotIndex_ = -1;
    uint32_t activeMicroKStep_ = 0;
    std::vector<BufferSlot> buffers_;
    std::vector<uint64_t> tileComputeStartCycles_;
    std::vector<uint64_t> tileComputeDoneCycles_;
    std::vector<uint64_t> tileRetireCycles_;
    std::vector<uint64_t> tileComputeStartSchedCycles_;
    std::vector<uint64_t> tileComputeDoneSchedCycles_;
    std::vector<uint64_t> tileRetireSchedCycles_;
    std::vector<uint8_t> activeTxnTileRetired_;
    std::vector<uint8_t> windowReuseDone_;
    std::vector<uint8_t> activeMatPayload_;
    std::vector<uint8_t> activeVecPayload_;
    std::vector<MicroOp> activeTileMicroOps_;
    size_t activeTileMicroOpCursor_ = 0;
    KStepScoreboard activeTileScoreboard_;
    std::deque<MicroOp> activeComputeReadyQueue_;
    bool activeMicroOpIssued_ = false;
    MicroOp activeIssuedMicroOp_{};
    bool activeTilePayloadLoaded_ = false;
    bool taskAccumInitialized_ = false;
    std::vector<CBufferPrefetch> cBufferPrefetches_;
    std::vector<uint8_t> partialValid_;
};

} // namespace Golem
} // namespace SST

#endif
