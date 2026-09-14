// Copyright 2009-2025 NTESS. Under the terms
// of Contract DE-NA0003525 with NTESS, the U.S.
// Government retains certain rights in this software.
//
// Copyright (c) 2009-2025, NTESS
// All rights reserved.
//
// Portions are copyright of other developers:
// See the file CONTRIBUTORS.TXT in the top level directory
// of the distribution for more information.
//
// This file is part of the SST software package. For license
// information, see the LICENSE file in the top level directory of the
// distribution.

#ifndef _COMPUTEARRAY_H
#define _COMPUTEARRAY_H

#include <sst/core/component.h>
#include <sst/core/subcomponent.h>
#include <sst/elements/golem/attention/attentionCluster.h>
#include <sst/core/event.h>
#include <sst/core/link.h>
#include <sst/core/output.h>
#include <sst/core/timeConverter.h>
#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace SST {
namespace Golem {

class ArrayEvent : public SST::Event {
public:
    ArrayEvent() {} // For serialization only
    ArrayEvent(uint32_t array, uint32_t operandBank = 0) :
        SST::Event(), arrayID(array), operandBank_(operandBank) {}

    uint32_t getArrayID() { return arrayID; };
    uint32_t getOperandBank() const { return operandBank_; }

protected:
    uint32_t arrayID;
    uint32_t operandBank_ = 0;

    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        SST_SER(arrayID);
        SST_SER(operandBank_);
    }
    ImplementSerializable(SST::Golem::ArrayEvent);
};

class ArrayBufferEvent : public SST::Event {
public:
    ArrayBufferEvent() = default;
    explicit ArrayBufferEvent(uint64_t requestId) : requestId_(requestId) {}

    uint64_t requestId() const { return requestId_; }

protected:
    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        SST_SER(requestId_);
    }
    ImplementSerializable(SST::Golem::ArrayBufferEvent);

private:
    uint64_t requestId_ = 0;
};

class ComputeArray : public SST::SubComponent {
public:
    SST_ELI_REGISTER_SUBCOMPONENT_API(
        SST::Golem::ComputeArray,
        TimeConverter*,
        Event::HandlerBase*
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"verbose", "Verbosity of outputs", "1"},
        {"core_id", "Owning core ID used in array-buffer statistics", "-1"},
        {"clock", "Array clock frequency", "1GHz"},
        {"arrayLatency", "Latency of array operation", "100ns"},
        {"modeledComputeCycles", "Modeled compute latency in array clock cycles", "1"},
        {"arrayMacPerCuPerCycle", "Active input columns processed per compute unit per array cycle", "1"},
        {"arrayPipelineDepth", "Fixed compute pipeline latency added to active-column execution", "0"},
        {"numArrays", "Number of arrays", "1"},
        {"arrayInputSize", "Input size of arrays", "1"},
        {"arrayOutputSize", "Output size of arrays", "1"},
        {"inputOperandSize", "Size of input operands", "1"},
        {"outputOperandSize", "Size of output operands", "1"},
        {"arrayBufferBaseLatencyCycles", "Base latency of an array-buffer transfer", "1"},
        {"arrayBufferBytesPerCycle", "Bytes transferred per array-buffer port per cycle", "64"},
        {"arrayBufferPorts", "Number of array-buffer transfer ports", "1"},
        {"arrayBufferQueueDepth", "Maximum queued plus active array-buffer transfers", "64"},
        {"arrayOutputReadCredits", "Maximum output reads in flight", "1"},
        {"arrayOutputReadBanks", "Number of independent output-read banks", "1"},
        {"attentionNearArrayOutputBytesPerCycle", "Bandwidth of the dedicated Attention near-array output fabric", "512"},
        {"attentionNearArrayOutputCredits", "Maximum grouped PV drains in flight", "2"},
        {"operandContextBanks", "Number of matrix/input operand contexts per physical array", "1"},
        {"matrixBroadcastMaxFanout", "Maximum number of array matrix banks reached by one broadcast", "16"},
        {"matrixBroadcastBytesPerCycle", "Ingress bytes delivered per cycle by the matrix broadcast tree", "64"},
        {"matrixBroadcastBaseLatencyCycles", "Base setup latency for a matrix broadcast", "1"},
        {"matrixBroadcastStageLatencyCycles", "Additional startup latency per binary fanout-tree stage", "1"},
        {"attention_cluster_enable", "Enable Attention cluster traffic classification and interval statistics", "0"},
    )

    SST_ELI_DOCUMENT_STATISTICS(
        {"matrix_broadcast_requests", "Accepted matrix broadcast transfers", "broadcasts", 1},
        {"matrix_broadcast_rejected", "Matrix broadcasts rejected by fanout or queue limits", "broadcasts", 1},
        {"matrix_broadcast_ingress_bytes", "Matrix payload bytes entering the broadcast fabric", "bytes", 1},
        {"matrix_broadcast_sink_bytes", "Aggregate matrix bytes written across destination array banks", "bytes", 1},
        {"matrix_broadcast_transfer_cycles", "Aggregate modeled matrix broadcast occupancy", "cycles", 1},
        {"matrix_broadcast_fanout", "Destination array count per accepted matrix broadcast", "arrays", 1},
        {"active_k_launches", "Array launches using fewer than the physical input columns", "launches", 1},
        {"active_k_columns", "Aggregate active input columns across active-K launches", "columns", 1},
        {"active_k_compute_cycles", "Aggregate modeled compute cycles across active-K launches", "cycles", 1},
        {"active_k_full_width_cycles_avoided", "Aggregate modeled compute cycles avoided relative to full-width launches", "cycles", 1},
        {"output_read_credit_stalls", "Output reads blocked by credit or bank ownership", "stalls", 1},
        {"output_read_bank_conflicts", "Output reads blocked by a busy output bank", "conflicts", 1},
        {"output_read_max_in_flight", "Maximum concurrent output reads", "reads", 1},
        {"attention_cluster_buffer_busy_union_ticks", "Union of cluster array-buffer service intervals", "ticks", 1},
        {"attention_cluster_buffer_busy_span_ticks", "Span of cluster array-buffer service intervals", "ticks", 1},
        {"attention_cluster_buffer_idle_gap_ticks", "Idle gaps inside the cluster array-buffer span", "ticks", 1},
        {"attention_cluster_buffer_max_concurrency", "Maximum simultaneous cluster array-buffer transfers", "requests", 1},
        {"attention_cluster_qk_k_matrix_requests", "Cluster K-panel ingress requests", "requests", 1},
        {"attention_cluster_qk_k_matrix_bytes", "Cluster K-panel ingress bytes", "bytes", 1},
        {"attention_cluster_qk_q_pair_requests", "Cluster Q pair multicast requests", "requests", 1},
        {"attention_cluster_qk_q_pair_bytes", "Cluster Q pair multicast ingress bytes", "bytes", 1},
        {"attention_cluster_qk_score_out_requests", "Cluster QK score output requests", "requests", 1},
        {"attention_cluster_qk_score_out_bytes", "Cluster QK score output bytes", "bytes", 1},
        {"attention_cluster_pv_group_drains", "Grouped PV output drains on the near-array fabric", "drains", 1},
        {"attention_cluster_pv_group_drain_bytes", "PV bytes moved by the near-array fabric", "bytes", 1},
        {"attention_cluster_pv_group_drain_cycles", "Modeled near-array fabric service cycles", "cycles", 1},
    )

    ComputeArray(ComponentId_t id, Params& params,
                 TimeConverter* tc,
                 Event::HandlerBase* handler)
        : SubComponent(id), out("", params.find<int>("verbose", 1), 0, Output::STDOUT),
          tileHandler(handler) {
        // Initialize parameters
        arrayClock = params.find<UnitAlgebra>("clock", "1GHz");
        arrayLatency = params.find<UnitAlgebra>("arrayLatency", "100ns");
        clockTC = getTimeConverter(arrayClock);
        latencyTC = getTimeConverter(arrayLatency);
        modeledComputeCycles = params.find<uint64_t>("modeledComputeCycles", 1);
        if (modeledComputeCycles == 0) {
            modeledComputeCycles = 1;
        }
        arrayMacPerCuPerCycle =
            std::max(params.find<double>("arrayMacPerCuPerCycle", 1.0), 0.000001);
        arrayPipelineDepth = params.find<uint64_t>("arrayPipelineDepth", 0);

        numArrays = params.find<uint64_t>("numArrays", 1);
        const uint32_t requestedOperandContextBanks =
            params.find<uint32_t>("operandContextBanks", 1);
        if (requestedOperandContextBanks == 0 || requestedOperandContextBanks > 2) {
            out.fatal(CALL_INFO, -1,
                "operandContextBanks must be one or two for the bounded array model\n");
        }
        operandContextBanks = requestedOperandContextBanks;
        inputArraySize = params.find<uint64_t>("arrayInputSize", 1);
        outputArraySize = params.find<uint64_t>("arrayOutputSize", 1);
        inputOperandSize = params.find<uint64_t>("inputOperandSize", 1);
        outputOperandSize = params.find<uint64_t>("outputOperandSize", 1);
        arrayCoreId_ = params.find<int>("core_id", -1);
        arrayBufferBaseLatencyCycles_ =
            std::max<uint64_t>(params.find<uint64_t>("arrayBufferBaseLatencyCycles", 1), 1);
        arrayBufferBytesPerCycle_ =
            std::max<uint64_t>(params.find<uint64_t>("arrayBufferBytesPerCycle", 64), 1);
        arrayBufferPorts_ =
            std::max<uint64_t>(params.find<uint64_t>("arrayBufferPorts", 1), 1);
        arrayBufferQueueDepth_ =
            std::max<uint64_t>(params.find<uint64_t>("arrayBufferQueueDepth", 64), 1);
        arrayOutputReadCredits_ =
            std::max<uint64_t>(params.find<uint64_t>("arrayOutputReadCredits", 1), 1);
        arrayOutputReadBanks_ =
            std::max<uint64_t>(params.find<uint64_t>("arrayOutputReadBanks", 1), 1);
        outputReadBankInFlight_.assign(arrayOutputReadBanks_, 0);
        attentionNearArrayOutputBytesPerCycle_ = std::max<uint64_t>(
            params.find<uint64_t>("attentionNearArrayOutputBytesPerCycle", 512), 1);
        attentionNearArrayOutputCredits_ = std::max<uint64_t>(
            params.find<uint64_t>("attentionNearArrayOutputCredits", 2), 1);
        matrixBroadcastMaxFanout_ =
            std::max<uint64_t>(params.find<uint64_t>("matrixBroadcastMaxFanout", 16), 1);
        matrixBroadcastBytesPerCycle_ =
            std::max<uint64_t>(params.find<uint64_t>("matrixBroadcastBytesPerCycle", 64), 1);
        matrixBroadcastBaseLatencyCycles_ =
            std::max<uint64_t>(params.find<uint64_t>("matrixBroadcastBaseLatencyCycles", 1), 1);
        matrixBroadcastStageLatencyCycles_ =
            params.find<uint64_t>("matrixBroadcastStageLatencyCycles", 1);
        attentionClusterEnable_ =
            params.find<bool>("attention_cluster_enable", false);
        bufferLink_ = configureSelfLink(
            "BufferSelf", *tc,
            new Event::Handler2<ComputeArray, &ComputeArray::handleBufferEvent>(this));
        bufferLink_->setDefaultTimeBase(*clockTC);
        statMatrixBroadcastRequests_ =
            registerStatistic<uint64_t>("matrix_broadcast_requests");
        statMatrixBroadcastRejected_ =
            registerStatistic<uint64_t>("matrix_broadcast_rejected");
        statMatrixBroadcastIngressBytes_ =
            registerStatistic<uint64_t>("matrix_broadcast_ingress_bytes");
        statMatrixBroadcastSinkBytes_ =
            registerStatistic<uint64_t>("matrix_broadcast_sink_bytes");
        statMatrixBroadcastTransferCycles_ =
            registerStatistic<uint64_t>("matrix_broadcast_transfer_cycles");
        statMatrixBroadcastFanout_ =
            registerStatistic<uint64_t>("matrix_broadcast_fanout");
        statActiveKLaunches_ = registerStatistic<uint64_t>("active_k_launches");
        statActiveKColumns_ = registerStatistic<uint64_t>("active_k_columns");
        statActiveKComputeCycles_ =
            registerStatistic<uint64_t>("active_k_compute_cycles");
        statActiveKFullWidthCyclesAvoided_ =
            registerStatistic<uint64_t>("active_k_full_width_cycles_avoided");
        statOutputReadCreditStalls_ =
            registerStatistic<uint64_t>("output_read_credit_stalls");
        statOutputReadBankConflicts_ =
            registerStatistic<uint64_t>("output_read_bank_conflicts");
        statOutputReadMaxInFlight_ =
            registerStatistic<uint64_t>("output_read_max_in_flight");
        statAttentionClusterBufferBusyUnion_ =
            registerStatistic<uint64_t>("attention_cluster_buffer_busy_union_ticks");
        statAttentionClusterBufferBusySpan_ =
            registerStatistic<uint64_t>("attention_cluster_buffer_busy_span_ticks");
        statAttentionClusterBufferIdleGap_ =
            registerStatistic<uint64_t>("attention_cluster_buffer_idle_gap_ticks");
        statAttentionClusterBufferMaxConcurrency_ =
            registerStatistic<uint64_t>("attention_cluster_buffer_max_concurrency");
        statAttentionClusterQkKMatrixRequests_ =
            registerStatistic<uint64_t>("attention_cluster_qk_k_matrix_requests");
        statAttentionClusterQkKMatrixBytes_ =
            registerStatistic<uint64_t>("attention_cluster_qk_k_matrix_bytes");
        statAttentionClusterQkQPairRequests_ =
            registerStatistic<uint64_t>("attention_cluster_qk_q_pair_requests");
        statAttentionClusterQkQPairBytes_ =
            registerStatistic<uint64_t>("attention_cluster_qk_q_pair_bytes");
        statAttentionClusterQkScoreOutRequests_ =
            registerStatistic<uint64_t>("attention_cluster_qk_score_out_requests");
        statAttentionClusterQkScoreOutBytes_ =
            registerStatistic<uint64_t>("attention_cluster_qk_score_out_bytes");
        statAttentionClusterPvGroupDrains_ =
            registerStatistic<uint64_t>("attention_cluster_pv_group_drains");
        statAttentionClusterPvGroupDrainBytes_ =
            registerStatistic<uint64_t>("attention_cluster_pv_group_drain_bytes");
        statAttentionClusterPvGroupDrainCycles_ =
            registerStatistic<uint64_t>("attention_cluster_pv_group_drain_cycles");
    }

    virtual ~ComputeArray() {}

    bool hasPendingBufferTransfers() const { return !bufferRequests_.empty(); }

    virtual void init(unsigned int phase) override {}
    virtual void setup() override {}
    virtual void finish() override {
        if (attentionClusterEnable_) {
            statAttentionClusterBufferBusyUnion_->addData(
                attentionClusterBufferIntervals_.unionCycles());
            statAttentionClusterBufferBusySpan_->addData(
                attentionClusterBufferIntervals_.spanCycles());
            statAttentionClusterBufferIdleGap_->addData(
                attentionClusterBufferIntervals_.idleCycles());
            statAttentionClusterBufferMaxConcurrency_->addData(
                attentionClusterBufferIntervals_.maxConcurrency());
        }
        out.output(
            "GOLEM_ARRAY_BUFFER_STATS core=%d requests=%" PRIu64
            " bytes=%" PRIu64 " rejected=%" PRIu64
            " high_water=%" PRIu64 " transfer_cycles=%" PRIu64
            " matrix_broadcast_requests=%" PRIu64
            " matrix_broadcast_rejected=%" PRIu64
            " matrix_broadcast_ingress_bytes=%" PRIu64
            " matrix_broadcast_sink_bytes=%" PRIu64
            " matrix_broadcast_transfer_cycles=%" PRIu64
            " matrix_broadcast_max_fanout=%" PRIu64
            " output_read_credits=%" PRIu64
            " output_read_banks=%" PRIu64
            " output_read_max_in_flight=%" PRIu64
            " output_read_credit_stalls=%" PRIu64
            " output_read_bank_conflicts=%" PRIu64
            " active_k_launches=%" PRIu64
            " active_k_columns=%" PRIu64
            " active_k_compute_cycles=%" PRIu64
            " active_k_full_width_cycles_avoided=%" PRIu64 "\n",
            arrayCoreId_, arrayBufferRequests_, arrayBufferBytes_, arrayBufferRejected_,
            arrayBufferHighWater_, arrayBufferTransferCycles_,
            matrixBroadcastRequests_, matrixBroadcastRejected_,
            matrixBroadcastIngressBytes_, matrixBroadcastSinkBytes_,
            matrixBroadcastTransferCycles_, matrixBroadcastObservedMaxFanout_,
            arrayOutputReadCredits_, arrayOutputReadBanks_, outputReadMaxInFlight_,
            outputReadCreditStalls_, outputReadBankConflicts_,
            activeKLaunches_, activeKColumns_, activeKComputeCycles_,
            activeKFullWidthCyclesAvoided_);
    }
    virtual void emergencyShutdown() override {}

    virtual void beginComputation(uint32_t arrayID) = 0;
    virtual void beginComputationBank(uint32_t arrayID, uint32_t operandBank) {
        if (operandBank == 0) beginComputation(arrayID);
    }
    virtual void beginComputationActive(uint32_t arrayID, uint32_t activeColumns) {
        if (activeColumns == inputArraySize) beginComputation(arrayID);
    }
    virtual void beginComputationActiveBank(
            uint32_t arrayID, uint32_t operandBank, uint32_t activeColumns) {
        if (operandBank == 0) beginComputationActive(arrayID, activeColumns);
    }
    virtual void handleSelfEvent(Event* ev) = 0;
    virtual SimTime_t getArrayLatency(uint32_t arrayID) = 0;
    virtual SimTime_t getArrayLatencyActive(
            uint32_t arrayID, uint32_t activeColumns) {
        return activeColumns == inputArraySize ? getArrayLatency(arrayID) : 0;
    }
    virtual void setMatrixItem(int32_t arrayID, int32_t index, double value) = 0;
    virtual void setVectorItem(int32_t arrayID, int32_t index, double value) = 0;
    virtual void compute(uint32_t arrayID) = 0;
    virtual void moveOutputToInput(uint32_t srcArrayID, uint32_t destArrayID) = 0;
    virtual void* getInputVector(uint32_t arrayID) = 0;
    virtual void* getOutputVector(uint32_t arrayID) = 0;
    using BufferCallback = std::function<void(bool, uint64_t)>;
    using BufferReadCallback =
        std::function<void(bool, uint64_t, const std::vector<double>&)>;
    using BufferByteReadCallback =
        std::function<void(bool, uint64_t, const std::vector<uint8_t>&)>;
    virtual bool programMatrixAsync(uint32_t arrayID,
                                    const std::vector<double>& matrix,
                                    size_t elemBytes,
                                    uint64_t tag,
                                    BufferCallback callback) = 0;
    virtual bool programMatrixBankAsync(
            uint32_t arrayID, uint32_t operandBank,
            const std::vector<double>& matrix, size_t elemBytes,
            uint64_t tag, BufferCallback callback) {
        return operandBank == 0 && programMatrixAsync(
            arrayID, matrix, elemBytes, tag, std::move(callback));
    }
    virtual bool programMatrixGroupAsync(const std::vector<uint32_t>& arrayIDs,
                                         const std::vector<double>& matrix,
                                         size_t elemBytes,
                                         uint64_t tag,
                                         BufferCallback callback) = 0;
    virtual bool programMatrixGroupBankAsync(
            const std::vector<uint32_t>& arrayIDs, uint32_t operandBank,
            const std::vector<double>& matrix, size_t elemBytes,
            uint64_t tag, BufferCallback callback) {
        return operandBank == 0 && programMatrixGroupAsync(
            arrayIDs, matrix, elemBytes, tag, std::move(callback));
    }
    virtual bool programMatrixGroupClassBankAsync(
            const std::vector<uint32_t>& arrayIDs, uint32_t operandBank,
            const std::vector<double>& matrix, size_t elemBytes,
            AttentionClusterTrafficClass trafficClass, uint64_t tag,
            BufferCallback callback) {
        if (trafficClass != AttentionClusterTrafficClass::Legacy) return false;
        return programMatrixGroupBankAsync(
            arrayIDs, operandBank, matrix, elemBytes, tag, std::move(callback));
    }
    virtual bool programInputAsync(uint32_t arrayID,
                                   const std::vector<double>& input,
                                   size_t elemBytes,
                                   uint64_t tag,
                                   BufferCallback callback) = 0;
    virtual bool programInputBankAsync(
            uint32_t arrayID, uint32_t operandBank,
            const std::vector<double>& input, size_t elemBytes,
            uint64_t tag, BufferCallback callback) {
        return operandBank == 0 && programInputAsync(
            arrayID, input, elemBytes, tag, std::move(callback));
    }
    virtual bool programInputGroupBankAsync(
            const std::vector<uint32_t>& arrayIDs, uint32_t operandBank,
            const std::vector<double>& input, size_t elemBytes,
            AttentionClusterTrafficClass trafficClass, uint64_t tag,
            BufferCallback callback) {
        (void)arrayIDs;
        (void)operandBank;
        (void)input;
        (void)elemBytes;
        (void)trafficClass;
        (void)tag;
        (void)callback;
        return false;
    }
    virtual bool programMatrixActiveAsync(
            uint32_t arrayID, const std::vector<double>& matrix,
            uint32_t activeColumns, size_t elemBytes, uint64_t tag,
            BufferCallback callback) {
        if (activeColumns != inputArraySize) return false;
        return programMatrixAsync(
            arrayID, matrix, elemBytes, tag, std::move(callback));
    }
    virtual bool programMatrixActiveBankAsync(
            uint32_t arrayID, uint32_t operandBank,
            const std::vector<double>& matrix, uint32_t activeColumns,
            size_t elemBytes, uint64_t tag, BufferCallback callback) {
        return operandBank == 0 && programMatrixActiveAsync(
            arrayID, matrix, activeColumns, elemBytes, tag, std::move(callback));
    }
    virtual bool programMatrixGroupActiveAsync(
            const std::vector<uint32_t>& arrayIDs,
            const std::vector<double>& matrix, uint32_t activeColumns,
            size_t elemBytes, uint64_t tag, BufferCallback callback) {
        if (activeColumns != inputArraySize) return false;
        return programMatrixGroupAsync(
            arrayIDs, matrix, elemBytes, tag, std::move(callback));
    }
    virtual bool programMatrixGroupActiveBankAsync(
            const std::vector<uint32_t>& arrayIDs, uint32_t operandBank,
            const std::vector<double>& matrix, uint32_t activeColumns,
            size_t elemBytes, uint64_t tag, BufferCallback callback) {
        return operandBank == 0 && programMatrixGroupActiveAsync(
            arrayIDs, matrix, activeColumns, elemBytes, tag, std::move(callback));
    }
    virtual bool programInputActiveAsync(
            uint32_t arrayID, const std::vector<double>& input,
            uint32_t activeColumns, size_t elemBytes, uint64_t tag,
            BufferCallback callback) {
        if (activeColumns != inputArraySize) return false;
        return programInputAsync(
            arrayID, input, elemBytes, tag, std::move(callback));
    }
    virtual bool programInputActiveBankAsync(
            uint32_t arrayID, uint32_t operandBank,
            const std::vector<double>& input, uint32_t activeColumns,
            size_t elemBytes, uint64_t tag, BufferCallback callback) {
        return operandBank == 0 && programInputActiveAsync(
            arrayID, input, activeColumns, elemBytes, tag, std::move(callback));
    }
    virtual bool programOperandsAsync(uint32_t arrayID,
                                      const std::vector<double>& matrix,
                                      const std::vector<double>& input,
                                      size_t elemBytes,
                                      uint64_t tag,
                                      BufferCallback callback) = 0;
    virtual bool readOutputAsync(uint32_t arrayID, size_t elemBytes,
                                 uint64_t tag, BufferReadCallback callback) = 0;
    virtual bool readOutputBytesAsync(uint32_t arrayID, size_t elemBytes,
                                      uint64_t tag,
                                      BufferByteReadCallback callback) = 0;
    virtual bool readOutputClassAsync(
            uint32_t arrayID, size_t elemBytes,
            AttentionClusterTrafficClass trafficClass, uint64_t tag,
            BufferReadCallback callback) {
        if (trafficClass != AttentionClusterTrafficClass::Legacy) return false;
        return readOutputAsync(
            arrayID, elemBytes, tag, std::move(callback));
    }
    virtual bool readOutputGroupClassAsync(
            const std::vector<uint32_t>& arrayIDs, size_t elemBytes,
            AttentionClusterTrafficClass trafficClass, uint64_t tag,
            BufferReadCallback callback) {
        (void)arrayIDs;
        (void)elemBytes;
        (void)trafficClass;
        (void)tag;
        (void)callback;
        return false;
    }
    virtual bool writeOutputAsync(uint32_t arrayID,
                                  const std::vector<double>& output,
                                  size_t elemBytes,
                                  uint64_t tag,
                                  BufferCallback callback) = 0;
    // Optional override: configure output buffer behavior (e.g., accumulate vs overwrite)
    virtual void configureOutputMode(uint32_t, uint64_t) {}

    virtual bool supportsActiveColumns() const { return false; }

    bool validateActiveColumnRequest(uint32_t activeColumns) const {
        return activeColumns > 0 && activeColumns <= inputArraySize &&
            (activeColumns == inputArraySize || supportsActiveColumns());
    }

    bool validateActiveMatrixRequest(
            uint32_t arrayID, size_t matrixElements,
            uint32_t activeColumns, size_t elemBytes) const {
        return arrayID < numArrays && validateActiveColumnRequest(activeColumns) &&
            matrixElements ==
                static_cast<size_t>(activeColumns) * outputArraySize &&
            elemBytes > 0;
    }

    bool validateActiveInputRequest(
            uint32_t arrayID, size_t inputElements,
            uint32_t activeColumns, size_t elemBytes) const {
        return arrayID < numArrays && validateActiveColumnRequest(activeColumns) &&
            inputElements == activeColumns && elemBytes > 0;
    }

    bool validateActiveLaunchRequest(
            uint32_t arrayID, uint32_t activeColumns) const {
        return arrayID < numArrays && validateActiveColumnRequest(activeColumns);
    }

    virtual bool validateOperandContextRequest(
            uint32_t arrayID, uint32_t operandBank) const {
        return arrayID < numArrays && operandBank == 0;
    }

    virtual bool validateOutputGroupRequest(
            const std::vector<uint32_t>& arrayIDs, size_t elemBytes,
            AttentionClusterTrafficClass trafficClass) const {
        (void)arrayIDs;
        (void)elemBytes;
        (void)trafficClass;
        return false;
    }

    bool validateMatrixBroadcastRequest(
            const std::vector<uint32_t>& arrayIDs, size_t matrixElements,
            size_t elemBytes, uint32_t activeColumns = 0) {
        const uint32_t columns = activeColumns == 0 ? inputArraySize : activeColumns;
        const std::unordered_set<uint32_t> uniqueIDs(
            arrayIDs.begin(), arrayIDs.end());
        const bool valid = !arrayIDs.empty() &&
            arrayIDs.size() <= matrixBroadcastMaxFanout_ &&
            uniqueIDs.size() == arrayIDs.size() &&
            validateActiveColumnRequest(columns) &&
            matrixElements == static_cast<size_t>(columns) * outputArraySize &&
            elemBytes > 0 &&
            std::all_of(
                arrayIDs.begin(), arrayIDs.end(),
                [this](uint32_t id) { return id < numArrays; });
        if (!valid) {
            matrixBroadcastRejected_ += 1;
            statMatrixBroadcastRejected_->addData(1);
        }
        return valid;
    }

    bool validateInputMulticastRequest(
            const std::vector<uint32_t>& arrayIDs, size_t inputElements,
            size_t elemBytes) const {
        const std::unordered_set<uint32_t> uniqueIDs(
            arrayIDs.begin(), arrayIDs.end());
        return !arrayIDs.empty() &&
            arrayIDs.size() <= matrixBroadcastMaxFanout_ &&
            uniqueIDs.size() == arrayIDs.size() &&
            inputElements > 0 && inputElements <= inputArraySize &&
            elemBytes > 0 &&
            std::all_of(arrayIDs.begin(), arrayIDs.end(),
                [this](uint32_t id) { return id < numArrays; });
    }

protected:
    bool enqueueBufferTransfer(
            size_t bytes, uint64_t tag, std::function<void()> completion,
            AttentionClusterTrafficClass trafficClass =
                AttentionClusterTrafficClass::Legacy) {
        const uint64_t transferCycles = arrayBufferBaseLatencyCycles_ +
            (bytes + arrayBufferBytesPerCycle_ - 1) / arrayBufferBytesPerCycle_;
        return enqueueModeledBufferTransfer(
            bytes, transferCycles, tag, std::move(completion), false, 0,
            trafficClass);
    }

    bool enqueueOutputReadTransfer(
            uint32_t arrayID, size_t bytes, uint64_t tag,
            std::function<void()> completion,
            AttentionClusterTrafficClass trafficClass =
                AttentionClusterTrafficClass::Legacy) {
        const uint32_t bank = arrayOutputReadBanks_ == 0 ? 0 :
            arrayID % arrayOutputReadBanks_;
        const uint64_t transferCycles = arrayBufferBaseLatencyCycles_ +
            (bytes + arrayBufferBytesPerCycle_ - 1) / arrayBufferBytesPerCycle_;
        return enqueueModeledBufferTransfer(
            bytes, transferCycles, tag, std::move(completion), true, bank,
            trafficClass);
    }

    bool enqueueNearArrayOutputTransfer(
            size_t bytes, uint64_t tag, std::function<void()> completion,
            AttentionClusterTrafficClass trafficClass =
                AttentionClusterTrafficClass::PvOFinalDrain) {
        if (!completion || !attentionClusterEnable_ ||
            nearArrayOutputInFlight_ >= attentionNearArrayOutputCredits_ ||
            bufferRequests_.size() >= arrayBufferQueueDepth_) return false;
        const uint64_t transferCycles = arrayBufferBaseLatencyCycles_ +
            (bytes + attentionNearArrayOutputBytesPerCycle_ - 1) /
                attentionNearArrayOutputBytesPerCycle_;
        const uint64_t requestId = nextBufferRequestId_++;
        bufferRequests_.emplace(
            requestId,
            BufferRequest{
                requestId, tag, bytes, transferCycles, std::move(completion),
                false, 0, trafficClass, true
            });
        ++nearArrayOutputInFlight_;
        if (trafficClass == AttentionClusterTrafficClass::PvOFinalDrain) {
            statAttentionClusterPvGroupDrains_->addData(1);
            statAttentionClusterPvGroupDrainBytes_->addData(bytes);
            statAttentionClusterPvGroupDrainCycles_->addData(transferCycles);
        } else if (trafficClass == AttentionClusterTrafficClass::QkScoreOut) {
            statAttentionClusterQkScoreOutRequests_->addData(1);
            statAttentionClusterQkScoreOutBytes_->addData(bytes);
        }
        bufferLink_->send(transferCycles, new ArrayBufferEvent(requestId));
        return true;
    }

    bool enqueueMatrixBroadcastTransfer(
            size_t bytes, size_t fanout, uint64_t tag,
            std::function<void()> completion,
            AttentionClusterTrafficClass trafficClass =
                AttentionClusterTrafficClass::Legacy) {
        const uint64_t treeStages = ceilLog2(fanout);
        const uint64_t transferCycles = matrixBroadcastBaseLatencyCycles_ +
            (bytes + matrixBroadcastBytesPerCycle_ - 1) /
                matrixBroadcastBytesPerCycle_ +
            treeStages * matrixBroadcastStageLatencyCycles_;
        if (!enqueueModeledBufferTransfer(
                bytes, transferCycles, tag, std::move(completion), false, 0,
                trafficClass)) {
            matrixBroadcastRejected_ += 1;
            statMatrixBroadcastRejected_->addData(1);
            return false;
        }
        const uint64_t sinkBytes = static_cast<uint64_t>(bytes) * fanout;
        matrixBroadcastRequests_ += 1;
        matrixBroadcastIngressBytes_ += bytes;
        matrixBroadcastSinkBytes_ += sinkBytes;
        matrixBroadcastTransferCycles_ += transferCycles;
        matrixBroadcastObservedMaxFanout_ =
            std::max<uint64_t>(matrixBroadcastObservedMaxFanout_, fanout);
        statMatrixBroadcastRequests_->addData(1);
        statMatrixBroadcastIngressBytes_->addData(bytes);
        statMatrixBroadcastSinkBytes_->addData(sinkBytes);
        statMatrixBroadcastTransferCycles_->addData(transferCycles);
        statMatrixBroadcastFanout_->addData(fanout);
        return true;
    }

    static uint64_t ceilLog2(uint64_t value) {
        uint64_t stages = 0;
        while (value > 1) {
            value = (value + 1) / 2;
            stages += 1;
        }
        return stages;
    }

private:
    bool enqueueModeledBufferTransfer(size_t bytes, uint64_t transferCycles,
                                      uint64_t tag,
                                      std::function<void()> completion,
                                      bool outputRead = false,
                                      uint32_t outputBank = 0,
                                      AttentionClusterTrafficClass trafficClass =
                                          AttentionClusterTrafficClass::Legacy) {
        if (!completion || bufferRequests_.size() >= arrayBufferQueueDepth_) {
            arrayBufferRejected_ += 1;
            return false;
        }
        const uint64_t requestId = nextBufferRequestId_++;
        bufferRequests_.emplace(
            requestId,
            BufferRequest{
                requestId, tag, bytes, transferCycles, std::move(completion),
                outputRead, outputBank, trafficClass, false
            });
        bufferQueue_.push_back(requestId);
        arrayBufferRequests_ += 1;
        arrayBufferBytes_ += bytes;
        arrayBufferHighWater_ =
            std::max<uint64_t>(arrayBufferHighWater_, bufferRequests_.size());
        tryIssueBufferTransfers();
        return true;
    }

protected:
    SST::Output out;
    SST::Link* selfLink = nullptr;
    SST::Event::HandlerBase* tileHandler = nullptr;
    UnitAlgebra arrayClock;
    UnitAlgebra arrayLatency;
    TimeConverter* clockTC = nullptr;
    TimeConverter* latencyTC = nullptr;
    uint64_t modeledComputeCycles = 1;
    double arrayMacPerCuPerCycle = 1.0;
    uint64_t arrayPipelineDepth = 0;

    uint64_t numArrays;
    uint32_t operandContextBanks = 1;
    uint64_t inputArraySize;
    uint64_t outputArraySize;
    uint64_t inputOperandSize;
    uint64_t outputOperandSize;

    SimTime_t modeledActiveComputeCycles(uint32_t activeColumns) const {
        return static_cast<SimTime_t>(
            std::ceil(static_cast<double>(activeColumns) / arrayMacPerCuPerCycle)) +
            arrayPipelineDepth;
    }

    void recordActiveKLaunch(uint32_t activeColumns, SimTime_t activeCycles) {
        if (activeColumns >= inputArraySize) return;
        activeKLaunches_ += 1;
        activeKColumns_ += activeColumns;
        activeKComputeCycles_ += activeCycles;
        const uint64_t avoided = modeledComputeCycles > activeCycles ?
            modeledComputeCycles - activeCycles : 0;
        activeKFullWidthCyclesAvoided_ += avoided;
        statActiveKLaunches_->addData(1);
        statActiveKColumns_->addData(activeColumns);
        statActiveKComputeCycles_->addData(activeCycles);
        statActiveKFullWidthCyclesAvoided_->addData(avoided);
    }

private:
    struct BufferRequest {
        uint64_t requestId = 0;
        uint64_t tag = 0;
        size_t bytes = 0;
        uint64_t transferCycles = 0;
        std::function<void()> completion;
        bool outputRead = false;
        uint32_t outputBank = 0;
        AttentionClusterTrafficClass trafficClass =
            AttentionClusterTrafficClass::Legacy;
        bool dedicatedOutput = false;
    };

    void tryIssueBufferTransfers() {
        while (arrayBufferInFlight_ < arrayBufferPorts_ && !bufferQueue_.empty()) {
            auto selected = bufferQueue_.end();
            for (auto it = bufferQueue_.begin(); it != bufferQueue_.end(); ++it) {
                const auto request = bufferRequests_.find(*it);
                if (request == bufferRequests_.end()) continue;
                if (request->second.outputRead) {
                    if (outputReadInFlight_ >= arrayOutputReadCredits_) {
                        outputReadCreditStalls_ += 1;
                        statOutputReadCreditStalls_->addData(1);
                        continue;
                    }
                    if (request->second.outputBank >= outputReadBankInFlight_.size() ||
                        outputReadBankInFlight_[request->second.outputBank] != 0) {
                        outputReadBankConflicts_ += 1;
                        statOutputReadBankConflicts_->addData(1);
                        continue;
                    }
                }
                selected = it;
                break;
            }
            if (selected == bufferQueue_.end()) break;
            const uint64_t requestId = *selected;
            bufferQueue_.erase(selected);
            const auto it = bufferRequests_.find(requestId);
            if (it == bufferRequests_.end()) {
                continue;
            }
            const uint64_t transferCycles = it->second.transferCycles;
            if (attentionClusterEnable_ &&
                it->second.trafficClass != AttentionClusterTrafficClass::Legacy) {
                const uint64_t start = getCurrentSimCycle();
                if (!attentionClusterBufferIntervals_.add(
                        start, start + clockTC->convertToCoreTime(transferCycles))) {
                    out.fatal(CALL_INFO, -1,
                              "Attention cluster buffer interval order violation\n");
                }
                const size_t classIndex =
                    static_cast<size_t>(it->second.trafficClass);
                attentionClusterTrafficRequests_[classIndex] += 1;
                attentionClusterTrafficBytes_[classIndex] += it->second.bytes;
                attentionClusterTrafficServiceCycles_[classIndex] += transferCycles;
                switch (it->second.trafficClass) {
                case AttentionClusterTrafficClass::QkKMatrix:
                    statAttentionClusterQkKMatrixRequests_->addData(1);
                    statAttentionClusterQkKMatrixBytes_->addData(it->second.bytes);
                    break;
                case AttentionClusterTrafficClass::QkQPair:
                    statAttentionClusterQkQPairRequests_->addData(1);
                    statAttentionClusterQkQPairBytes_->addData(it->second.bytes);
                    break;
                case AttentionClusterTrafficClass::QkScoreOut:
                    statAttentionClusterQkScoreOutRequests_->addData(1);
                    statAttentionClusterQkScoreOutBytes_->addData(it->second.bytes);
                    break;
                default:
                    break;
                }
            }
            arrayBufferTransferCycles_ += transferCycles;
            arrayBufferInFlight_ += 1;
            if (it->second.outputRead) {
                outputReadInFlight_ += 1;
                outputReadBankInFlight_[it->second.outputBank] += 1;
                outputReadMaxInFlight_ = std::max(outputReadMaxInFlight_, outputReadInFlight_);
                statOutputReadMaxInFlight_->addData(outputReadMaxInFlight_);
            }
            bufferLink_->send(transferCycles, new ArrayBufferEvent(requestId));
        }
    }

    void handleBufferEvent(Event* event) {
        auto* bufferEvent = dynamic_cast<ArrayBufferEvent*>(event);
        if (bufferEvent == nullptr) {
            delete event;
            return;
        }
        const auto it = bufferRequests_.find(bufferEvent->requestId());
        bool usedArrayBufferPort = false;
        if (it != bufferRequests_.end()) {
            usedArrayBufferPort = !it->second.dedicatedOutput;
            if (it->second.dedicatedOutput) {
                if (nearArrayOutputInFlight_ > 0) --nearArrayOutputInFlight_;
            } else if (it->second.outputRead) {
                if (outputReadInFlight_ > 0) outputReadInFlight_ -= 1;
                if (it->second.outputBank < outputReadBankInFlight_.size() &&
                    outputReadBankInFlight_[it->second.outputBank] > 0) {
                    outputReadBankInFlight_[it->second.outputBank] -= 1;
                }
            }
            auto completion = std::move(it->second.completion);
            bufferRequests_.erase(it);
            if (completion) {
                completion();
            }
        }
        if (usedArrayBufferPort && arrayBufferInFlight_ > 0) {
            arrayBufferInFlight_ -= 1;
        }
        delete bufferEvent;
        tryIssueBufferTransfers();
    }

    SST::Link* bufferLink_ = nullptr;
    uint64_t arrayBufferBaseLatencyCycles_ = 1;
    uint64_t arrayBufferBytesPerCycle_ = 64;
    uint64_t arrayBufferPorts_ = 1;
    uint64_t arrayBufferQueueDepth_ = 64;
    uint64_t matrixBroadcastMaxFanout_ = 16;
    uint64_t matrixBroadcastBytesPerCycle_ = 64;
    uint64_t matrixBroadcastBaseLatencyCycles_ = 1;
    uint64_t matrixBroadcastStageLatencyCycles_ = 1;
    bool attentionClusterEnable_ = false;
    BusyIntervalUnion attentionClusterBufferIntervals_;
    std::array<uint64_t,
        static_cast<size_t>(AttentionClusterTrafficClass::Count)>
        attentionClusterTrafficRequests_ = {};
    std::array<uint64_t,
        static_cast<size_t>(AttentionClusterTrafficClass::Count)>
        attentionClusterTrafficBytes_ = {};
    std::array<uint64_t,
        static_cast<size_t>(AttentionClusterTrafficClass::Count)>
        attentionClusterTrafficServiceCycles_ = {};
    uint64_t arrayBufferInFlight_ = 0;
    uint64_t nextBufferRequestId_ = 1;
    uint64_t arrayBufferRequests_ = 0;
    uint64_t arrayBufferBytes_ = 0;
    uint64_t arrayBufferRejected_ = 0;
    uint64_t arrayBufferHighWater_ = 0;
    uint64_t arrayBufferTransferCycles_ = 0;
    uint64_t arrayOutputReadCredits_ = 1;
    uint64_t arrayOutputReadBanks_ = 1;
    uint64_t attentionNearArrayOutputBytesPerCycle_ = 512;
    uint64_t attentionNearArrayOutputCredits_ = 2;
    uint64_t nearArrayOutputInFlight_ = 0;
    uint64_t outputReadInFlight_ = 0;
    uint64_t outputReadMaxInFlight_ = 0;
    uint64_t outputReadCreditStalls_ = 0;
    uint64_t outputReadBankConflicts_ = 0;
    std::vector<uint32_t> outputReadBankInFlight_;
    uint64_t matrixBroadcastRequests_ = 0;
    uint64_t matrixBroadcastRejected_ = 0;
    uint64_t matrixBroadcastIngressBytes_ = 0;
    uint64_t matrixBroadcastSinkBytes_ = 0;
    uint64_t matrixBroadcastTransferCycles_ = 0;
    uint64_t matrixBroadcastObservedMaxFanout_ = 0;
    uint64_t activeKLaunches_ = 0;
    uint64_t activeKColumns_ = 0;
    uint64_t activeKComputeCycles_ = 0;
    uint64_t activeKFullWidthCyclesAvoided_ = 0;
    int arrayCoreId_ = -1;
    Statistic<uint64_t>* statMatrixBroadcastRequests_ = nullptr;
    Statistic<uint64_t>* statMatrixBroadcastRejected_ = nullptr;
    Statistic<uint64_t>* statMatrixBroadcastIngressBytes_ = nullptr;
    Statistic<uint64_t>* statMatrixBroadcastSinkBytes_ = nullptr;
    Statistic<uint64_t>* statMatrixBroadcastTransferCycles_ = nullptr;
    Statistic<uint64_t>* statMatrixBroadcastFanout_ = nullptr;
    Statistic<uint64_t>* statActiveKLaunches_ = nullptr;
    Statistic<uint64_t>* statActiveKColumns_ = nullptr;
    Statistic<uint64_t>* statActiveKComputeCycles_ = nullptr;
    Statistic<uint64_t>* statActiveKFullWidthCyclesAvoided_ = nullptr;
    Statistic<uint64_t>* statOutputReadCreditStalls_ = nullptr;
    Statistic<uint64_t>* statOutputReadBankConflicts_ = nullptr;
    Statistic<uint64_t>* statOutputReadMaxInFlight_ = nullptr;
    Statistic<uint64_t>* statAttentionClusterBufferBusyUnion_ = nullptr;
    Statistic<uint64_t>* statAttentionClusterBufferBusySpan_ = nullptr;
    Statistic<uint64_t>* statAttentionClusterBufferIdleGap_ = nullptr;
    Statistic<uint64_t>* statAttentionClusterBufferMaxConcurrency_ = nullptr;
    Statistic<uint64_t>* statAttentionClusterQkKMatrixRequests_ = nullptr;
    Statistic<uint64_t>* statAttentionClusterQkKMatrixBytes_ = nullptr;
    Statistic<uint64_t>* statAttentionClusterQkQPairRequests_ = nullptr;
    Statistic<uint64_t>* statAttentionClusterQkQPairBytes_ = nullptr;
    Statistic<uint64_t>* statAttentionClusterQkScoreOutRequests_ = nullptr;
    Statistic<uint64_t>* statAttentionClusterQkScoreOutBytes_ = nullptr;
    Statistic<uint64_t>* statAttentionClusterPvGroupDrains_ = nullptr;
    Statistic<uint64_t>* statAttentionClusterPvGroupDrainBytes_ = nullptr;
    Statistic<uint64_t>* statAttentionClusterPvGroupDrainCycles_ = nullptr;
    std::deque<uint64_t> bufferQueue_;
    std::unordered_map<uint64_t, BufferRequest> bufferRequests_;
};

} // namespace Golem
} // namespace SST

#endif /* _COMPUTEARRAY_H */
