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

#ifndef GOLEM_TILE_UNIT_ARRAY_MODEL_H
#define GOLEM_TILE_UNIT_ARRAY_MODEL_H

#include <sst/elements/golem/tileunit/tileUnitAPI.h>
#include <sst/elements/golem/tileunit/tileUnit.h>
#include <sst/elements/golem/fp16.h>
#include <sst/elements/golem/computeunit/multiPrecisionComputeUnit.h>
#include <sst/elements/golem/computeunit/peTiming.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <type_traits>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace SST {
namespace Golem {

template<typename T>
class TileUnitArrayModel : public TileUnitAPI {
public:
    SST_ELI_REGISTER_SUBCOMPONENT_DERIVED_API(
            SST::Golem::TileUnitArrayModel<T>, 
            SST::Golem::TileUnitAPI,
            TimeConverter*,
            Event::HandlerBase*
    )

    TileUnitArrayModel(ComponentId_t id, Params& params,
                         TimeConverter* tc,
                         Event::HandlerBase* handler)
        : TileUnitAPI(id, params, tc, handler) {
        // Configure selfLink
        selfLink = configureSelfLink("Self", *tc, new Event::Handler2<TileUnitArrayModel,&TileUnitArrayModel::handleSelfEvent>(this));
        selfLink->setDefaultTimeBase(*clockTC);

        // Initialize vectors   numArrays表示有几个矩阵
        const size_t operandContexts = static_cast<size_t>(numArrays);
        inputVectors.resize(operandContexts);
        outputVectors.resize(numArrays);
        matrixData.resize(operandContexts);
        outputModes.resize(numArrays, OutputMode::Overwrite);
        for (size_t i=0;i<numArrays;++i) tileUnits.emplace_back(outputArraySize);
        activeInputColumns.resize(operandContexts, inputArraySize);
        projectionFp16Context.resize(operandContexts, false);
        for (size_t i = 0; i < operandContexts; i++) {
            inputVectors[i].resize(inputArraySize, T());
            matrixData[i].resize(inputArraySize * outputArraySize, T());
        }
        for (uint32_t i = 0; i < numArrays; i++) {
            outputVectors[i].resize(outputArraySize, T());
        }

        functionalCompute = params.find<int>("functionalCompute", 1) != 0;
        timingNumericalFastPath = params.find<int>("timingNumericalFastPath", 1) != 0;
        attentionClusterQkArrays = params.find<uint32_t>(
            "attention_cluster_qk_arrays", 16);
        dumpEnabled = params.find<int>("mvm_dump_enable", 0) != 0;
        coreId = params.find<int>("core_id", -1);
        std::printf("[ARRAY_NUMERICAL_DATA] core=%d functional=%d timing_fast_path=%d materialize=%d\n", coreId, int(functionalCompute), int(timingNumericalFastPath), int(requiresNumericalData()));
        std::printf("[ARRAY_OPERAND_STORAGE] core=%d arrays=%llu operand_banks=1 matrix_contexts=%zu input_contexts=%zu\n",
            coreId, (unsigned long long)numArrays, matrixData.size(), inputVectors.size());
        dumpRootDir = params.find<std::string>("mvm_dump_dir", "mvm_dumps");
        const std::string dumpMode = params.find<std::string>("mvm_dump_mode", "overwrite");
        dumpOverwrite = (dumpMode != "append");

        if (dumpEnabled) {
            std::ostringstream folder;
            folder << dumpRootDir << "/core_" << coreId;
            dumpCoreDir = folder.str();
            std::error_code ec;
            std::filesystem::create_directories(dumpCoreDir, ec);
            if (ec) {
                out.verbose(CALL_INFO, 1, 0,
                            "mvmComputeArray: failed to create dump dir %s, disable dump\n",
                            dumpCoreDir.c_str());
                dumpEnabled = false;
            }
        }
    }
    bool requiresNumericalData() const override { return functionalCompute || !timingNumericalFastPath; }
    bool configurePrecision(const XTileMC::PrecisionSpecV02& spec, uint32_t k) override {
        if (!XTileMC::validPrecision(spec) || k == 0 || k > inputArraySize) return false;
        const auto selectedMode = static_cast<XTileMC::ComputeMode>(spec.compute_mode);
        const uint32_t lanes = XTileMC::lanesPerBeat(selectedMode);
        const uint32_t packetCapacity = (inputArraySize + lanes - 1) / lanes;
        if (requiresNumericalData() && (rawMatrix.empty() || packetCapacity != rawPacketCapacity)) {
            rawPacketCapacity = packetCapacity;
            rawMatrix.assign(numArrays, std::vector<uint32_t>(rawPacketCapacity*outputArraySize));
            rawInput.assign(numArrays, std::vector<uint32_t>(rawPacketCapacity));
        }
        if (rawOutput.empty()) {
            rawOutput.resize(numArrays, std::vector<uint32_t>(outputArraySize));
            if (functionalCompute)
                rawCUs.resize(numArrays, std::vector<MultiPrecisionComputeUnit>(outputArraySize));
        }
        precisionEnabled = true;
        mode = selectedMode;
        activeK = k;
        return true;
    }
    void useLegacyPrecision() override { precisionEnabled = false; }
    void setMatrixBits(uint32_t tu, uint32_t idx, uint32_t bits) override {
        if (!requiresNumericalData()) return;
        const uint32_t row = idx/inputArraySize, col = idx%inputArraySize;
        setPacketLane(rawMatrix.at(tu), row*rawPacketCapacity, col, bits);
    }
    void setVectorBits(uint32_t tu, uint32_t idx, uint32_t bits) override {
        if (!requiresNumericalData()) return;
        setPacketLane(rawInput.at(tu), 0, idx, bits);
    }
    uint32_t outputBits(uint32_t tu, uint32_t row) const override { return rawOutput.at(tu).at(row); }
    void setOutputBits(uint32_t tu, uint32_t row, uint32_t bits) override { rawOutput.at(tu).at(row) = bits; }
    //启动计算 根据 arrayID 获取阵列的延迟，并通过 selfLink 发送一个计算事件
    virtual void beginComputation(uint32_t arrayID) override {
        beginComputationBank(arrayID, 0);
    }

    virtual void beginComputationBank(
            uint32_t arrayID, uint32_t operandBank) override {
        if (!validOperandContext(arrayID, operandBank)) return;
        activeInputColumns[operandIndex(arrayID, operandBank)] =
            precisionEnabled ? activeK : inputArraySize;
        SimTime_t latency = getArrayLatency(arrayID);   // 得到阵列延迟，比如返回1
        traceMacBegin(arrayID);
        ArrayEvent* ev = new ArrayEvent(arrayID, operandBank);
        selfLink->send(latency, ev);                    // 延迟latency后将事件ev发送给自己
        //把事件通过selfLink（SST框架的本地自环连接）发送，延迟latency后会回到当前组件。
        //这就是模拟“异步/延迟计算”，即MVM计算不是立刻完成，而是等一段时间（比如硬件计算延迟）。
    }

    virtual void beginComputationActive(
            uint32_t arrayID, uint32_t activeColumns) override {
        beginComputationActiveBank(arrayID, 0, activeColumns);
    }

    virtual void beginComputationActiveBank(
            uint32_t arrayID, uint32_t operandBank,
            uint32_t activeColumns) override {
        if (!validOperandContext(arrayID, operandBank) ||
            !validateActiveColumnRequest(activeColumns)) {
            return;
        }
        activeInputColumns[operandIndex(arrayID, operandBank)] = activeColumns;
        const SimTime_t latency = getArrayLatencyActive(arrayID, activeColumns);
        recordActiveKLaunch(activeColumns, latency);
        traceMacBegin(arrayID);
        selfLink->send(latency, new ArrayEvent(arrayID, operandBank));
    }


    /*
    1.beginComputation：“我要让阵列2开始算，硬件要1个cycle，先发事件出去。”
    2.SST内部延迟1个单位
    3.handleSelfEvent： “1个cycle到了！事件回来了，阵列2现在可以开始真正计算了！”*/

    virtual void handleSelfEvent(Event* ev) override {
        ArrayEvent* aev = static_cast<ArrayEvent*>(ev);
        uint32_t arrayID = aev->getArrayID();
        const uint32_t operandBank = aev->getOperandBank();

        if (functionalCompute) {
            computeBank(arrayID, operandBank);
        } else if (outputModes[arrayID] == OutputMode::Overwrite) {
            // Timing-only mode preserves the modeled completion event and data
            // movement while avoiding host-side numerical MAC execution.
            clearOutputVector(arrayID);
        }

        traceMacDone(arrayID);
        (*tileHandler)(ev);
    }
    //这个函数的作用是将 value 存入指定阵列 arrayID 中的矩阵 matrixData，位置是 index,
    //可以理解为arrayID是不同的二维矩阵的编号，index是二维矩阵里元素的索引
    virtual void setMatrixItem(int32_t arrayID, int32_t index, double value) override {
        if (!requiresNumericalData()) return;
        projectionFp16Context[arrayID] = false;
        matrixData[arrayID][index] = static_cast<T>(value);
    }
    //同理
    virtual void setVectorItem(int32_t arrayID, int32_t index, double value) override {
        if (!requiresNumericalData()) return;
        inputVectors[arrayID][index] = static_cast<T>(value);
    }

    virtual bool programMatrixAsync(
        uint32_t arrayID,
        const std::vector<double>& matrix,
        size_t elemBytes,
        uint64_t tag,
        typename ComputeArray::BufferCallback callback) override {
        return programMatrixBankAsync(
            arrayID, 0, matrix, elemBytes, tag, std::move(callback));
    }

    virtual bool programMatrixBankAsync(
        uint32_t arrayID, uint32_t operandBank,
        const std::vector<double>& matrix, size_t elemBytes, uint64_t tag,
        typename ComputeArray::BufferCallback callback) override {
        if (!validOperandContext(arrayID, operandBank) ||
            matrix.size() != inputArraySize * outputArraySize || elemBytes == 0) {
            return false;
        }
        const size_t context = operandIndex(arrayID, operandBank);
        return enqueueBufferTransfer(
            matrix.size() * elemBytes, tag,
            [this, context, matrix, tag, callback = std::move(callback)]() {
                projectionFp16Context[context] = false;
                if (requiresNumericalData()) std::transform(matrix.begin(), matrix.end(), matrixData[context].begin(),
                               [](double value) { return static_cast<T>(value); });
                if (callback) {
                    callback(true, tag);
                }
            });
    }

    virtual bool programMatrixGroupAsync(
        const std::vector<uint32_t>& arrayIDs,
        const std::vector<double>& matrix,
        size_t elemBytes,
        uint64_t tag,
        typename ComputeArray::BufferCallback callback) override {
        return programMatrixGroupBankAsync(
            arrayIDs, 0, matrix, elemBytes, tag, std::move(callback));
    }

    virtual bool programMatrixGroupBankAsync(
        const std::vector<uint32_t>& arrayIDs, uint32_t operandBank,
        const std::vector<double>& matrix, size_t elemBytes, uint64_t tag,
        typename ComputeArray::BufferCallback callback) override {
        if (!validateMatrixBroadcastRequest(
                arrayIDs, matrix.size(), elemBytes) ||
            operandBank >= operandContextBanks) {
            return false;
        }
        return enqueueMatrixBroadcastTransfer(
            matrix.size() * elemBytes, arrayIDs.size(), tag,
            [this, arrayIDs, operandBank, matrix, tag,
             callback = std::move(callback)]() {
                for (uint32_t arrayID : arrayIDs) {
                    projectionFp16Context[operandIndex(arrayID, operandBank)] = false;
                    auto& target = matrixData[operandIndex(arrayID, operandBank)];
                    if (requiresNumericalData()) std::transform(matrix.begin(), matrix.end(), target.begin(),
                                   [](double value) { return static_cast<T>(value); });
                }
                if (callback) callback(true, tag);
            });
    }

    virtual bool programMatrixGroupClassBankAsync(
        const std::vector<uint32_t>& arrayIDs, uint32_t operandBank,
        const std::vector<double>& matrix, size_t elemBytes,
        AttentionClusterTrafficClass trafficClass, uint64_t tag,
        typename ComputeArray::BufferCallback callback) override {
        if (!validateMatrixBroadcastRequest(
                arrayIDs, matrix.size(), elemBytes) ||
            operandBank >= operandContextBanks ||
            trafficClass == AttentionClusterTrafficClass::Legacy) {
            return false;
        }
        return enqueueMatrixBroadcastTransfer(
            matrix.size() * elemBytes, arrayIDs.size(), tag,
            [this, arrayIDs, operandBank, matrix, tag, trafficClass,
             callback = std::move(callback)]() {
                for (uint32_t arrayID : arrayIDs) {
                    // Dtype of the existing projection command, not a new
                    // operand bank or compute resource. Attention remains FP32.
                    projectionFp16Context[operandIndex(arrayID, operandBank)] =
                        trafficClass == AttentionClusterTrafficClass::ProjectionWeights;
                    auto& target = matrixData[operandIndex(arrayID, operandBank)];
                    if (requiresNumericalData()) std::transform(matrix.begin(), matrix.end(), target.begin(),
                                   [](double value) { return static_cast<T>(value); });
                }
                if (callback) callback(true, tag);
            }, trafficClass);
    }

    virtual bool programMatrixActiveAsync(
        uint32_t arrayID,
        const std::vector<double>& matrix,
        uint32_t activeColumns,
        size_t elemBytes,
        uint64_t tag,
        typename ComputeArray::BufferCallback callback) override {
        return programMatrixActiveBankAsync(
            arrayID, 0, matrix, activeColumns, elemBytes, tag,
            std::move(callback));
    }

    virtual bool programMatrixActiveBankAsync(
        uint32_t arrayID, uint32_t operandBank,
        const std::vector<double>& matrix, uint32_t activeColumns,
        size_t elemBytes, uint64_t tag,
        typename ComputeArray::BufferCallback callback) override {
        if (!validateActiveMatrixRequest(
                arrayID, matrix.size(), activeColumns, elemBytes) ||
            operandBank >= operandContextBanks) {
            return false;
        }
        return enqueueBufferTransfer(
            matrix.size() * elemBytes, tag,
            [this, arrayID, operandBank, matrix, activeColumns, tag,
             callback = std::move(callback)]() {
                writeCompactMatrix(arrayID, operandBank, matrix, activeColumns);
                if (callback) callback(true, tag);
            });
    }

    virtual bool programMatrixGroupActiveAsync(
        const std::vector<uint32_t>& arrayIDs,
        const std::vector<double>& matrix,
        uint32_t activeColumns,
        size_t elemBytes,
        uint64_t tag,
        typename ComputeArray::BufferCallback callback) override {
        return programMatrixGroupActiveBankAsync(
            arrayIDs, 0, matrix, activeColumns, elemBytes, tag,
            std::move(callback));
    }

    virtual bool programMatrixGroupActiveBankAsync(
        const std::vector<uint32_t>& arrayIDs, uint32_t operandBank,
        const std::vector<double>& matrix, uint32_t activeColumns,
        size_t elemBytes, uint64_t tag,
        typename ComputeArray::BufferCallback callback) override {
        if (!validateMatrixBroadcastRequest(
                arrayIDs, matrix.size(), elemBytes, activeColumns) ||
            operandBank >= operandContextBanks) {
            return false;
        }
        return enqueueMatrixBroadcastTransfer(
            matrix.size() * elemBytes, arrayIDs.size(), tag,
            [this, arrayIDs, operandBank, matrix, activeColumns, tag,
             callback = std::move(callback)]() {
                for (uint32_t arrayID : arrayIDs) {
                    writeCompactMatrix(
                        arrayID, operandBank, matrix, activeColumns);
                }
                if (callback) callback(true, tag);
            });
    }

    virtual bool programInputAsync(
        uint32_t arrayID,
        const std::vector<double>& input,
        size_t elemBytes,
        uint64_t tag,
        typename ComputeArray::BufferCallback callback) override {
        return programInputBankAsync(
            arrayID, 0, input, elemBytes, tag, std::move(callback));
    }

    virtual bool programInputBankAsync(
        uint32_t arrayID, uint32_t operandBank,
        const std::vector<double>& input, size_t elemBytes, uint64_t tag,
        typename ComputeArray::BufferCallback callback) override {
        if (!validOperandContext(arrayID, operandBank) || input.empty() ||
            input.size() > inputArraySize || elemBytes == 0) {
            return false;
        }
        const size_t context = operandIndex(arrayID, operandBank);
        return enqueueBufferTransfer(
            input.size() * elemBytes, tag,
            [this, context, input, tag, callback = std::move(callback)]() {
                if (requiresNumericalData()) std::transform(input.begin(), input.end(), inputVectors[context].begin(),
                               [](double value) { return static_cast<T>(value); });
                if (callback) {
                    callback(true, tag);
                }
            });
    }

    virtual bool programInputGroupBankAsync(
        const std::vector<uint32_t>& arrayIDs, uint32_t operandBank,
        const std::vector<double>& input, size_t elemBytes,
        AttentionClusterTrafficClass trafficClass, uint64_t tag,
        typename ComputeArray::BufferCallback callback) override {
        const bool validPvTopology =
            trafficClass != AttentionClusterTrafficClass::PvPInput ||
            (arrayIDs.size() == 2 &&
             (input.size() == 32 || input.size() == 64) &&
             arrayIDs[0] >= attentionClusterQkArrays &&
             arrayIDs[1] == arrayIDs[0] + 1 &&
             (arrayIDs[0] % 2) == 0 && arrayIDs[1] < 64);
        const bool validQkTopology =
            trafficClass != AttentionClusterTrafficClass::QkQPair ||
            (arrayIDs.size() == 1 &&
             arrayIDs.front() < attentionClusterQkArrays &&
             input.size() == 64);
        const bool validSequentialPvTopology =
            trafficClass != AttentionClusterTrafficClass::SequentialPvInput ||
            (arrayIDs.size() == 1 && input.size() == 64 && arrayIDs.front() < 64) ||
            (arrayIDs.size() == 64 && input.size() == 64);
        if (!validateInputMulticastRequest(
                arrayIDs, input.size(), elemBytes) ||
            operandBank >= operandContextBanks ||
            (trafficClass != AttentionClusterTrafficClass::QkQPair &&
             trafficClass != AttentionClusterTrafficClass::PvPInput &&
             trafficClass != AttentionClusterTrafficClass::SequentialPvInput) ||
            !validQkTopology || !validPvTopology ||
            !validSequentialPvTopology) {
            return false;
        }
        return enqueueMatrixBroadcastTransfer(
            input.size() * elemBytes, arrayIDs.size(), tag,
            [this, arrayIDs, operandBank, input, tag,
             callback = std::move(callback)]() {
                for (uint32_t arrayID : arrayIDs) {
                    auto& target = inputVectors[operandIndex(arrayID, operandBank)];
                    std::fill(target.begin(), target.end(), T());
                    if (requiresNumericalData()) std::transform(input.begin(), input.end(), target.begin(),
                                   [](double value) { return static_cast<T>(value); });
                }
                if (callback) callback(true, tag);
            }, trafficClass);
    }

    virtual bool programInputScatterBankAsync(
        const std::vector<uint32_t>& arrayIDs, uint32_t operandBank,
        const std::vector<double>& inputs, size_t elemBytes,
        AttentionClusterTrafficClass trafficClass, uint64_t tag,
        typename ComputeArray::BufferCallback callback) override {
        if (!validateInputScatterRequest(arrayIDs, inputs.size(), elemBytes) ||
            operandBank >= operandContextBanks ||
            (trafficClass != AttentionClusterTrafficClass::SequentialQkInputScatter &&
             trafficClass != AttentionClusterTrafficClass::SequentialPvInputScatter &&
             trafficClass != AttentionClusterTrafficClass::ProjectionInputScatter)) {
            return false;
        }
        return enqueueInputScatterTransfer(
            inputs.size() * elemBytes, arrayIDs.size(), tag,
            [this, arrayIDs, operandBank, inputs, tag,
             callback = std::move(callback)]() {
                for (size_t lane = 0; lane < arrayIDs.size(); ++lane) {
                    auto& target = inputVectors[
                        operandIndex(arrayIDs[lane], operandBank)];
                    const auto begin = inputs.begin() + lane * inputArraySize;
                    std::transform(
                        begin, begin + inputArraySize, target.begin(),
                        [](double value) { return static_cast<T>(value); });
                }
                if (callback) callback(true, tag);
            }, trafficClass);
    }

    virtual bool programInputActiveAsync(
        uint32_t arrayID,
        const std::vector<double>& input,
        uint32_t activeColumns,
        size_t elemBytes,
        uint64_t tag,
        typename ComputeArray::BufferCallback callback) override {
        return programInputActiveBankAsync(
            arrayID, 0, input, activeColumns, elemBytes, tag,
            std::move(callback));
    }

    virtual bool programInputActiveBankAsync(
        uint32_t arrayID, uint32_t operandBank,
        const std::vector<double>& input, uint32_t activeColumns,
        size_t elemBytes, uint64_t tag,
        typename ComputeArray::BufferCallback callback) override {
        if (!validateActiveInputRequest(
                arrayID, input.size(), activeColumns, elemBytes) ||
            operandBank >= operandContextBanks) {
            return false;
        }
        const size_t context = operandIndex(arrayID, operandBank);
        return enqueueBufferTransfer(
            input.size() * elemBytes, tag,
            [this, context, input, tag, callback = std::move(callback)]() {
                std::fill(inputVectors[context].begin(),
                          inputVectors[context].end(), T());
                if (requiresNumericalData()) std::transform(input.begin(), input.end(),
                               inputVectors[context].begin(),
                               [](double value) { return static_cast<T>(value); });
                if (callback) callback(true, tag);
            });
    }

    virtual bool programOperandsAsync(
        uint32_t arrayID,
        const std::vector<double>& matrix,
        const std::vector<double>& input,
        size_t elemBytes,
        uint64_t tag,
        typename ComputeArray::BufferCallback callback) override {
        if (arrayID >= numArrays ||
            matrix.size() != inputArraySize * outputArraySize ||
            input.size() != inputArraySize || elemBytes == 0) {
            return false;
        }
        const size_t bytes = (matrix.size() + input.size()) * elemBytes;
        return enqueueBufferTransfer(
            bytes, tag,
            [this, arrayID, matrix, input, tag, callback = std::move(callback)]() {
                projectionFp16Context[arrayID] = false;
                if (requiresNumericalData()) std::transform(matrix.begin(), matrix.end(), matrixData[arrayID].begin(),
                               [](double value) { return static_cast<T>(value); });
                if (requiresNumericalData()) std::transform(input.begin(), input.end(), inputVectors[arrayID].begin(),
                               [](double value) { return static_cast<T>(value); });
                if (callback) {
                    callback(true, tag);
                }
            });
    }

    virtual bool readOutputAsync(
        uint32_t arrayID,
        size_t elemBytes,
        uint64_t tag,
        typename ComputeArray::BufferReadCallback callback) override {
        if (arrayID >= outputVectors.size() || elemBytes == 0) {
            return false;
        }
        return enqueueOutputReadTransfer(arrayID,
            outputArraySize * elemBytes, tag,
            [this, arrayID, tag, callback = std::move(callback)]() {
                std::vector<double> values(outputVectors[arrayID].size(), 0.0);
                std::transform(outputVectors[arrayID].begin(), outputVectors[arrayID].end(),
                               values.begin(),
                               [](T value) { return static_cast<double>(value); });
                if (callback) {
                    callback(true, tag, values);
                }
            });
    }

    virtual bool readOutputClassAsync(
        uint32_t arrayID, size_t elemBytes,
        AttentionClusterTrafficClass trafficClass, uint64_t tag,
        typename ComputeArray::BufferReadCallback callback) override {
        if (arrayID >= outputVectors.size() || elemBytes == 0 ||
            trafficClass != AttentionClusterTrafficClass::QkScoreOut) {
            return false;
        }
        return enqueueOutputReadTransfer(
            arrayID, outputArraySize * elemBytes, tag,
            [this, arrayID, tag, callback = std::move(callback)]() {
                std::vector<double> values(outputVectors[arrayID].size(), 0.0);
                std::transform(outputVectors[arrayID].begin(),
                               outputVectors[arrayID].end(), values.begin(),
                               [](T value) { return static_cast<double>(value); });
                if (callback) callback(true, tag, values);
            }, trafficClass);
    }

    virtual bool readOutputGroupClassAsync(
        const std::vector<uint32_t>& arrayIDs, size_t elemBytes,
        AttentionClusterTrafficClass trafficClass, uint64_t tag,
        typename ComputeArray::BufferReadCallback callback) override {
        if (!validateOutputGroupRequest(arrayIDs, elemBytes, trafficClass)) {
            return false;
        }
        auto completion =
            [this, arrayIDs, tag, callback = std::move(callback)]() {
                std::vector<double> values;
                values.reserve(arrayIDs.size() * outputArraySize);
                for (uint32_t arrayID : arrayIDs) {
                    if (arrayID >= outputVectors.size()) {
                        if (callback) callback(false, tag, {});
                        return;
                    }
                    std::transform(
                        outputVectors[arrayID].begin(),
                        outputVectors[arrayID].end(),
                        std::back_inserter(values),
                        [](T value) { return static_cast<double>(value); });
                }
                if (callback) callback(true, tag, values);
            };
        const size_t bytes = arrayIDs.size() * outputArraySize * elemBytes;
        if (trafficClass == AttentionClusterTrafficClass::SequentialPvOOutput ||
            trafficClass == AttentionClusterTrafficClass::ProjectionOutput) {
            return enqueueOutputScatterGatherTransfer(
                bytes, arrayIDs.size(), false, tag, std::move(completion),
                trafficClass);
        }
        return enqueueNearArrayOutputTransfer(
            bytes, tag, std::move(completion), trafficClass);
    }

    virtual bool writeOutputGroupClassAsync(
        const std::vector<uint32_t>& arrayIDs,
        const std::vector<double>& outputs, size_t elemBytes,
        AttentionClusterTrafficClass trafficClass, uint64_t tag,
        typename ComputeArray::BufferCallback callback) override {
        if (!validateOutputGroupRequest(arrayIDs, elemBytes, trafficClass) ||
            (trafficClass != AttentionClusterTrafficClass::SequentialPvORestore &&
             trafficClass != AttentionClusterTrafficClass::ProjectionOutput) ||
            outputs.size() != arrayIDs.size() * outputArraySize) {
            return false;
        }
        return enqueueOutputScatterGatherTransfer(
            outputs.size() * elemBytes, arrayIDs.size(), true, tag,
            [this, arrayIDs, outputs, tag, callback = std::move(callback)]() {
                for (size_t lane = 0; lane < arrayIDs.size(); ++lane) {
                    auto& target = outputVectors[arrayIDs[lane]];
                    const auto begin = outputs.begin() + lane * outputArraySize;
                    std::transform(
                        begin, begin + outputArraySize, target.begin(),
                        [](double value) { return static_cast<T>(value); });
                }
                if (callback) callback(true, tag);
            }, trafficClass);
    }

    bool validateOperandContextRequest(
            uint32_t arrayID, uint32_t operandBank) const override {
        return validOperandContext(arrayID, operandBank);
    }

    bool validateOutputGroupRequest(
            const std::vector<uint32_t>& arrayIDs, size_t elemBytes,
            AttentionClusterTrafficClass trafficClass) const override {
        // Projection transports FP16 results even when the worker's functional
        // storage type is float. Other traffic classes retain their width check.
        const bool correctElementWidth = elemBytes == sizeof(T) ||
            (trafficClass == AttentionClusterTrafficClass::ProjectionOutput &&
             elemBytes == 2 && std::is_same<T, float>::value);
        if (numArrays != 64 || inputArraySize != 64 ||
            outputArraySize != 64 || !correctElementWidth) {
            return false;
        }
        if (trafficClass ==
            AttentionClusterTrafficClass::SequentialQkScoreOut) {
            if (arrayIDs.size() != 64) return false;
            for (uint32_t index = 0; index < 64; ++index) {
                if (arrayIDs[index] != index) return false;
            }
            return true;
        }
        if (trafficClass == AttentionClusterTrafficClass::ProjectionOutput ||
            trafficClass == AttentionClusterTrafficClass::SequentialPvORestore ||
            trafficClass == AttentionClusterTrafficClass::SequentialPvOOutput) {
            if (arrayIDs.empty() || arrayIDs.size() > 64) return false;
            for (uint32_t index = 0; index < arrayIDs.size(); ++index) {
                if (arrayIDs[index] != index) return false;
            }
            return true;
        }
        if (arrayIDs.size() != 2 ||
            arrayIDs[1] != arrayIDs[0] + 1 || (arrayIDs[0] % 2) != 0) {
            return false;
        }
        if (trafficClass == AttentionClusterTrafficClass::QkScoreOut)
            return arrayIDs[1] < attentionClusterQkArrays;
        if (trafficClass == AttentionClusterTrafficClass::PvOFinalDrain)
            return arrayIDs[0] >= attentionClusterQkArrays &&
                arrayIDs[1] < 64;
        return false;
    }

    virtual bool readOutputBytesAsync(
        uint32_t arrayID,
        size_t elemBytes,
        uint64_t tag,
        typename ComputeArray::BufferByteReadCallback callback) override {
        if (arrayID >= outputVectors.size() || elemBytes == 0) {
            return false;
        }
        return enqueueBufferTransfer(
            outputArraySize * elemBytes, tag,
            [this, arrayID, elemBytes, tag, callback = std::move(callback)]() {
                std::vector<uint8_t> bytes(outputVectors[arrayID].size() * elemBytes, 0);
                for (size_t index = 0; index < outputVectors[arrayID].size(); ++index) {
                    const T value = outputVectors[arrayID][index];
                    std::memcpy(bytes.data() + index * elemBytes, &value,
                                std::min<size_t>(sizeof(T), elemBytes));
                }
                if (callback) {
                    callback(true, tag, bytes);
                }
            });
    }

    virtual bool writeOutputAsync(
        uint32_t arrayID,
        const std::vector<double>& output,
        size_t elemBytes,
        uint64_t tag,
        typename ComputeArray::BufferCallback callback) override {
        if (arrayID >= outputVectors.size() || output.size() != outputArraySize ||
            elemBytes == 0) {
            return false;
        }
        return enqueueBufferTransfer(
            output.size() * elemBytes, tag,
            [this, arrayID, output, tag, callback = std::move(callback)]() {
                std::transform(output.begin(), output.end(), outputVectors[arrayID].begin(),
                               [](double value) { return static_cast<T>(value); });
                if (callback) {
                    callback(true, tag);
                }
            });
    }

    virtual void compute(uint32_t arrayID) override {
        computeBank(arrayID, 0);
    }

    void computeBank(uint32_t arrayID, uint32_t operandBank) {
        if (!validOperandContext(arrayID, operandBank)) return;
        const size_t context = operandIndex(arrayID, operandBank);
        if (precisionEnabled) {
            const bool accumulate = outputModes[arrayID] == OutputMode::Accumulate;
            const uint32_t precisionK = activeInputColumns[context];
            const uint32_t packets = (precisionK + XTileMC::lanesPerBeat(mode)-1) / XTileMC::lanesPerBeat(mode);
            for (uint32_t row=0;row<outputArraySize;++row)
                rawOutput[arrayID][row] = rawCUs[arrayID][row].dotPackets(mode,
                    rawMatrix[arrayID].data()+row*rawPacketCapacity,
                    rawInput[arrayID].data(),packets,precisionK,
                    accumulate ? rawOutput[arrayID][row] : 0);
            return;
        }
        auto& inputVector = inputVectors[context];
        auto& outputVector = outputVectors[arrayID];
        auto& matrix = matrixData[context];

        // Ensure output vector is correctly sized
        outputVector.resize(outputArraySize);

        bool accumulateMode = (outputModes[arrayID] == OutputMode::Accumulate);
        if (!accumulateMode) {
            clearOutputVector(arrayID);
        }

        // Print input vector
        out.verbose(CALL_INFO, 2, 0, "MVM for array %u:\n\n", arrayID);
        const uint32_t activeColumns = activeInputColumns[context];
        for (uint32_t col = 0; col < activeColumns; col++) {
            printValue(inputVector[col]);
        }
        out.verbose(CALL_INFO, 2, 0, "\n\n");

        // Perform matrix-vector multiplication
        for (uint32_t row = 0; row < outputArraySize; row++) {
            if (projectionFp16Context[context]) {
                GolemFp16 dot = tileUnits[arrayID].projectionDot(row,
                    matrix.data()+row*inputArraySize,inputVector.data(),activeColumns);
                if (accumulateMode)
                    dot += GolemFp16(static_cast<float>(outputVector[row]));
                outputVector[row] = static_cast<T>(static_cast<float>(dot));
                continue;
            }
            T dot = tileUnits[arrayID].dot(row,matrix.data()+row*inputArraySize,
                                           inputVector.data(),activeColumns);
            for (uint32_t col=0;col<activeColumns;++col)
                printValue(matrix[row*inputArraySize+col]);
            if (accumulateMode) {
                outputVector[row] += dot;
            } else {
                outputVector[row] = dot;
            }
            out.verbose(CALL_INFO, 2, 0, "  ");
            printValue(outputVector[row]);
            out.verbose(CALL_INFO, 2, 0, "\n");
        }
        out.verbose(CALL_INFO, 2, 0, "\n\n");

        if (dumpEnabled) {
            dumpMvmSnapshot(arrayID, inputVector, matrix, outputVector);
        }
    }
    //这个函数是为了返回阵列的延迟时间，通常在模拟中用来表示阵列进行计算所需的时间。在这段代码中，延迟被硬编码为 1。
    virtual SimTime_t getArrayLatency(uint32_t arrayID) override {
        (void)arrayID;
        if (precisionEnabled)
            return peTiming(mode).completionCycles((activeK+XTileMC::lanesPerBeat(mode)-1)/XTileMC::lanesPerBeat(mode));
        return modeledComputeCycles;
    }
    virtual SimTime_t getArrayLatencyActive(
            uint32_t arrayID, uint32_t activeColumns) override {
        (void)arrayID;
        if (!validateActiveColumnRequest(activeColumns)) return 0;
        if (precisionEnabled)
            return peTiming(mode).completionCycles((activeColumns+XTileMC::lanesPerBeat(mode)-1)/XTileMC::lanesPerBeat(mode));
        if (activeColumns == inputArraySize) return modeledComputeCycles;
        return modeledActiveComputeCycles(activeColumns);
    }

    virtual bool supportsActiveColumns() const override { return true; }
    //这个虚拟函数用于将某个阵列的输出向量移动到另一个阵列的输入向量
    virtual void moveOutputToInput(uint32_t srcArrayID, uint32_t destArrayID) override {
        std::copy(outputVectors[srcArrayID].begin(), outputVectors[srcArrayID].end(), inputVectors[destArrayID].begin());
    }
    //该函数返回指定阵列的输入向量的指针
    virtual void* getInputVector(uint32_t arrayID) override {
        return static_cast<void*>(&inputVectors[arrayID]);
    }
    //该函数返回指定阵列的输出向量的指针
    virtual void* getOutputVector(uint32_t arrayID) override {
        return static_cast<void*>(&outputVectors[arrayID]);
    }

    virtual void configureOutputMode(uint32_t arrayID, uint64_t command) override {
        if (arrayID >= outputModes.size()) {
            out.verbose(CALL_INFO, 1, 0,
                        "mvmComputeArray: invalid arrayID %u for ocfg\n", arrayID);
            return;
        }
        switch (command) {
            case 0:
                outputModes[arrayID] = OutputMode::Overwrite;
                out.verbose(CALL_INFO, 3, 0,
                            "mvmComputeArray: array %u set to overwrite mode\n", arrayID);
                break;
            case 1:
                outputModes[arrayID] = OutputMode::Accumulate;
                out.verbose(CALL_INFO, 3, 0,
                            "mvmComputeArray: array %u set to accumulate mode\n", arrayID);
                break;
            case 2:
                clearOutputVector(arrayID);
                out.verbose(CALL_INFO, 3, 0,
                            "mvmComputeArray: array %u output buffer cleared\n", arrayID);
                break;
            default:
                out.verbose(CALL_INFO, 1, 0,
                            "mvmComputeArray: unknown ocfg command %" PRIu64 "\n", command);
                break;
        }
    }

protected:
    struct TraceMacGroup {
        uint64_t id = 0, start = 0, count = 0, remaining = 0;
    };
    uint64_t traceMacSequence_ = 0;
    std::unordered_map<uint64_t, TraceMacGroup> traceMacGroups_;
    std::unordered_map<uint32_t, uint64_t> traceMacOwners_;
    void traceMacBegin(uint32_t arrayID) {
        if (!resourceTraceEnabled()) return;
        const uint64_t tick = getCurrentSimCycle();
        auto& group = traceMacGroups_[tick];
        if (group.id == 0) { group.id = ++traceMacSequence_; group.start = tick; }
        ++group.count;
        ++group.remaining;
        traceMacOwners_[arrayID] = tick;
    }
    void traceMacDone(uint32_t arrayID) {
        if (!resourceTraceEnabled()) return;
        const auto owner = traceMacOwners_.find(arrayID);
        if (owner == traceMacOwners_.end()) return;
        const uint64_t key = owner->second;
        traceMacOwners_.erase(owner);
        auto& group = traceMacGroups_.at(key);
        if (--group.remaining == 0) {
            traceResourceInterval("array", "compute", coreId, group.id, 0, 0,
                group.start, group.start, getCurrentSimCycle(), getCurrentSimCycle(),
                0, 0, 0, group.count);
            traceMacGroups_.erase(key);
        }
    }
    enum class OutputMode : uint8_t { Overwrite = 0, Accumulate = 1 };
    std::vector<TileUnit<T>> tileUnits;
    std::vector<std::vector<T>> inputVectors;
    std::vector<std::vector<T>> outputVectors;
    std::vector<std::vector<T>> matrixData;
    std::vector<OutputMode> outputModes;
    std::vector<uint32_t> activeInputColumns;
    std::vector<bool> projectionFp16Context;
    bool precisionEnabled = false;
    XTileMC::ComputeMode mode = XTileMC::ComputeMode::Fp32;
    uint32_t activeK = 0;
    uint32_t rawPacketCapacity = 0;
    std::vector<std::vector<uint32_t>> rawMatrix, rawInput, rawOutput;
    std::vector<std::vector<MultiPrecisionComputeUnit>> rawCUs;
    void setPacketLane(std::vector<uint32_t>& words, uint32_t base,
                       uint32_t element, uint32_t value) {
        const uint32_t bits = XTileMC::inputBits(mode), lanes = XTileMC::lanesPerBeat(mode);
        const uint32_t shift = (element % lanes)*bits;
        const uint32_t mask = bits == 32 ? 0xffffffffu : (1u << bits)-1;
        auto& word = words.at(base + element/lanes);
        word = (word & ~(mask << shift)) | ((value & mask) << shift);
    }
    bool timingNumericalFastPath = true;
    bool functionalCompute = true;
    uint32_t attentionClusterQkArrays = 24;
    bool dumpEnabled = false;
    int coreId = -1;
    std::string dumpRootDir;
    std::string dumpCoreDir;
    uint64_t dumpSeq = 0;
    bool dumpOverwrite = true;
    std::vector<bool> dumpFileInitialized;

    void writeCompactMatrix(
            uint32_t arrayID, uint32_t operandBank,
            const std::vector<double>& matrix,
            uint32_t activeColumns) {
        if (!requiresNumericalData()) return;
        projectionFp16Context[operandIndex(arrayID, operandBank)] = false;
        auto& target = matrixData[operandIndex(arrayID, operandBank)];
        std::fill(target.begin(), target.end(), T());
        for (uint32_t row = 0; row < outputArraySize; ++row) {
            for (uint32_t col = 0; col < activeColumns; ++col) {
                target[row * inputArraySize + col] =
                    static_cast<T>(matrix[row * activeColumns + col]);
            }
        }
    }

    void dumpMvmSnapshot(uint32_t arrayID,
                         const std::vector<T>& inputVector,
                         const std::vector<T>& matrix,
                         const std::vector<T>& outputVector) {
        if (dumpFileInitialized.size() < numArrays) {
            dumpFileInitialized.assign(numArrays, false);
        }

        std::ostringstream fpath;
        fpath << dumpCoreDir << "/mvm_array_" << arrayID << ".log";
        std::ios::openmode openMode = std::ios::out;
        if (!dumpOverwrite || dumpFileInitialized[arrayID]) {
            openMode |= std::ios::app;
        } else {
            openMode |= std::ios::trunc;
        }

        std::ofstream ofs(fpath.str(), openMode);
        if (!ofs) {
            out.verbose(CALL_INFO, 1, 0,
                        "mvmComputeArray: cannot open dump file %s\n", fpath.str().c_str());
            return;
        }
        dumpFileInitialized[arrayID] = true;

        ofs << "=== MVM Snapshot #" << dumpSeq++ << " core=" << coreId
            << " array=" << arrayID << " ===\n";
        ofs << "InputVector:";
        for (uint32_t col = 0; col < inputArraySize; col++) {
            if constexpr (std::is_same<T, int64_t>::value) {
                ofs << " " << static_cast<long long>(inputVector[col]);
            } else if constexpr (std::is_same<T, float>::value) {
                ofs << " " << inputVector[col];
            } else {
                ofs << " " << inputVector[col];
            }
        }
        ofs << "\n";

        ofs << "MatrixAndOutput:\n";
        for (uint32_t row = 0; row < outputArraySize; row++) {
            for (uint32_t col = 0; col < inputArraySize; col++) {
                const T value = matrix[row * inputArraySize + col];
                if constexpr (std::is_same<T, int64_t>::value) {
                    ofs << static_cast<long long>(value) << " ";
                } else if constexpr (std::is_same<T, float>::value) {
                    ofs << value << " ";
                } else {
                    ofs << value << " ";
                }
            }
            ofs << " | ";
            if constexpr (std::is_same<T, int64_t>::value) {
                ofs << static_cast<long long>(outputVector[row]);
            } else if constexpr (std::is_same<T, float>::value) {
                ofs << outputVector[row];
            } else {
                ofs << outputVector[row];
            }
            ofs << "\n";
        }
        ofs << "\n";
    }

    void clearOutputVector(uint32_t arrayID) {
        std::fill(outputVectors[arrayID].begin(), outputVectors[arrayID].end(), T());
        if (precisionEnabled)
            std::fill(rawOutput[arrayID].begin(),rawOutput[arrayID].end(),0u);
    }

    bool validOperandContext(uint32_t arrayID, uint32_t operandBank) const {
        return arrayID < numArrays && operandBank == 0;
    }

    size_t operandIndex(uint32_t arrayID, uint32_t operandBank) const {
        (void)operandBank;  // Nonzero bank IDs are rejected by request validation.
        return arrayID;
    }

    void printValue(const T& value) {
        if constexpr (std::is_same<T, int64_t>::value) {
            out.verbose(CALL_INFO, 2, 0, "%" PRId64 " ", value);
        } else if constexpr (std::is_same<T, float>::value) {
            out.verbose(CALL_INFO, 2, 0, "%f ", value);
        }
    }
};

} // namespace Golem
} // namespace SST

#endif 
