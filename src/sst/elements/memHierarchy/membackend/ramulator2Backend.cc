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


#include <sst_config.h>
#include "sst/elements/memHierarchy/util.h"
#include "membackend/ramulator2Backend.h"
#include "ramulator/base/config.h"
#include "ramulator/base/factory.h"
#include "ramulator/base/base.h"
#include "ramulator/controller/controller_base.h"
#include "ramulator/controller/plugin/i_controller_plugin.h"
#include "ramulator/dram/dram_spec.h"

#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <limits>
#include <sstream>
#include <unordered_map>

using namespace SST;
using namespace SST::MemHierarchy;

namespace Ramulator {

class GolemCommandStats : public IControllerPlugin, public Implementation {
    RAMULATOR_REGISTER_IMPLEMENTATION(
        IControllerPlugin, GolemCommandStats, "GolemCommandStats")

public:
    void init() override {
        RAMULATOR_PARSE_PARAM(backendId_, unsigned, "backend_id").required();
    }

    void setup(IFrontEnd*, IMemorySystem*) override {
        controller_ = cast_parent<ControllerBase>();
        const auto& spec = *controller_->m_device.m_spec;
        pseudoChannelLevel_ = spec.get_level_id("PseudoChannel");
        bankGroupLevel_ = spec.get_level_id("BankGroup");
        readCommand_ = spec.get_command_id("RD");
        readAutoPrechargeCommand_ = spec.get_command_id("RDA");
        writeCommand_ = spec.get_command_id("WR");
        writeAutoPrechargeCommand_ = spec.get_command_id("WRA");
        activateCommand_ = spec.get_command_id("ACT");
        prechargeCommand_ = spec.get_command_id("PREpb");
        refreshBankCommand_ = spec.get_command_id("REFpb");
        refreshAllCommand_ = spec.get_command_id("REFab");
        tccdShortCycles_ = spec.get_timing_value("nCCDS");
        tccdLongCycles_ = spec.get_timing_value("nCCDL");
    }

    void on_issue(const Request& req) override {
        commandCounts_[req.command]++;
        const bool isRead = req.command == readCommand_ ||
                            req.command == readAutoPrechargeCommand_;
        const bool isWrite = req.command == writeCommand_ ||
                             req.command == writeAutoPrechargeCommand_;

        if (isRead) {
            recordReadCommand(req);
        }

        if (isRead || isWrite) {
            if (haveLastColumn_) {
                const uint64_t gap = static_cast<uint64_t>(controller_->m_clk - lastColumnCycle_);
                columnGapSum_ += gap;
                columnPairCount_++;
                if (lastColumnWasRead_ != isRead) {
                    readWriteSwitches_++;
                    readWriteSwitchGapSum_ += gap;
                }
            }
            haveLastColumn_ = true;
            lastColumnWasRead_ = isRead;
            lastColumnCycle_ = controller_->m_clk;
        }
    }

    void finalize() override {
        auto count = [&](int command) -> uint64_t {
            const auto it = commandCounts_.find(command);
            return it == commandCounts_.end() ? 0 : it->second;
        };
        const uint64_t readWindow = readCommands_ == 0
            ? 0 : lastReadCycle_ - firstReadCycle_ + 1;
        std::ostringstream summary;
        summary << "RAMULATOR2_COMMAND_SUMMARY"
                << " node=" << backendId_
                << " channel=" << controller_->m_channel_id
                << " cycles=" << controller_->m_clk
                << " rd=" << readCommands_
                << " wr=" << count(writeCommand_) + count(writeAutoPrechargeCommand_)
                << " act=" << count(activateCommand_)
                << " pre=" << count(prechargeCommand_)
                << " refpb=" << count(refreshBankCommand_)
                << " refab=" << count(refreshAllCommand_)
                << " first_rd=" << (readCommands_ == 0 ? 0 : firstReadCycle_)
                << " last_rd=" << (readCommands_ == 0 ? 0 : lastReadCycle_)
                << " read_window=" << readWindow
                << " tccd_l_pairs=" << tccdLongPairs_
                << " tccd_l_gap_sum=" << tccdLongGapSum_
                << " tccd_l_min_gap=" << finiteMin(tccdLongMinGap_)
                << " tccd_l_violations=" << tccdLongViolations_
                << " tccd_s_pairs=" << tccdShortPairs_
                << " tccd_s_gap_sum=" << tccdShortGapSum_
                << " tccd_s_min_gap=" << finiteMin(tccdShortMinGap_)
                << " tccd_s_violations=" << tccdShortViolations_
                << " cross_pc_pairs=" << crossPseudoChannelPairs_
                << " cross_pc_gap_sum=" << crossPseudoChannelGapSum_
                << " cross_pc_min_gap=" << finiteMin(crossPseudoChannelMinGap_)
                << " rw_switches=" << readWriteSwitches_
                << " rw_switch_gap_sum=" << readWriteSwitchGapSum_
                << " column_pairs=" << columnPairCount_
                << " column_gap_sum=" << columnGapSum_;
        std::cout << summary.str() << std::endl;
    }

private:
    static uint64_t finiteMin(uint64_t value) {
        return value == std::numeric_limits<uint64_t>::max() ? 0 : value;
    }

    void recordReadCommand(const Request& req) {
        const uint64_t cycle = static_cast<uint64_t>(controller_->m_clk);
        const int pseudoChannel = req.addr_vec[pseudoChannelLevel_];
        const int bankGroup = req.addr_vec[bankGroupLevel_];
        if (readCommands_ == 0) {
            firstReadCycle_ = cycle;
        }
        lastReadCycle_ = cycle;
        readCommands_++;

        if (haveLastRead_ && lastColumnWasRead_) {
            const uint64_t gap = cycle - lastReadCycleForPair_;
            if (pseudoChannel != lastReadPseudoChannel_) {
                crossPseudoChannelPairs_++;
                crossPseudoChannelGapSum_ += gap;
                crossPseudoChannelMinGap_ = std::min(crossPseudoChannelMinGap_, gap);
            } else if (bankGroup == lastReadBankGroup_) {
                tccdLongPairs_++;
                tccdLongGapSum_ += gap;
                tccdLongMinGap_ = std::min(tccdLongMinGap_, gap);
                if (gap < tccdLongCycles_) tccdLongViolations_++;
            } else {
                tccdShortPairs_++;
                tccdShortGapSum_ += gap;
                tccdShortMinGap_ = std::min(tccdShortMinGap_, gap);
                if (gap < tccdShortCycles_) tccdShortViolations_++;
            }
        }
        haveLastRead_ = true;
        lastReadCycleForPair_ = cycle;
        lastReadPseudoChannel_ = pseudoChannel;
        lastReadBankGroup_ = bankGroup;
    }

    ControllerBase* controller_ = nullptr;
    unsigned backendId_ = 0;
    int pseudoChannelLevel_ = -1;
    int bankGroupLevel_ = -1;
    int readCommand_ = -1;
    int readAutoPrechargeCommand_ = -1;
    int writeCommand_ = -1;
    int writeAutoPrechargeCommand_ = -1;
    int activateCommand_ = -1;
    int prechargeCommand_ = -1;
    int refreshBankCommand_ = -1;
    int refreshAllCommand_ = -1;
    uint64_t tccdShortCycles_ = 0;
    uint64_t tccdLongCycles_ = 0;
    std::unordered_map<int, uint64_t> commandCounts_;
    bool haveLastRead_ = false;
    bool haveLastColumn_ = false;
    bool lastColumnWasRead_ = false;
    uint64_t firstReadCycle_ = 0;
    uint64_t lastReadCycle_ = 0;
    uint64_t lastReadCycleForPair_ = 0;
    uint64_t lastColumnCycle_ = 0;
    int lastReadPseudoChannel_ = -1;
    int lastReadBankGroup_ = -1;
    uint64_t readCommands_ = 0;
    uint64_t tccdLongPairs_ = 0;
    uint64_t tccdLongGapSum_ = 0;
    uint64_t tccdLongMinGap_ = std::numeric_limits<uint64_t>::max();
    uint64_t tccdLongViolations_ = 0;
    uint64_t tccdShortPairs_ = 0;
    uint64_t tccdShortGapSum_ = 0;
    uint64_t tccdShortMinGap_ = std::numeric_limits<uint64_t>::max();
    uint64_t tccdShortViolations_ = 0;
    uint64_t crossPseudoChannelPairs_ = 0;
    uint64_t crossPseudoChannelGapSum_ = 0;
    uint64_t crossPseudoChannelMinGap_ = std::numeric_limits<uint64_t>::max();
    uint64_t readWriteSwitches_ = 0;
    uint64_t readWriteSwitchGapSum_ = 0;
    uint64_t columnPairCount_ = 0;
    uint64_t columnGapSum_ = 0;
};

static Ramulator::ConfigNode addGolemCommandStats(
    Ramulator::ConfigNode config, unsigned backendId) {
    Ramulator::ConfigNode memorySystem = config["memory_system"];
    Ramulator::ConfigNode controllers = memorySystem["controllers"];
    Ramulator::ConfigNode updatedControllers(Ramulator::ConfigNode::Seq{});
    for (const auto& originalController : controllers.seq()) {
        Ramulator::ConfigNode controller = originalController;
        Ramulator::ConfigNode plugins = controller["controller_plugins"];
        if (!plugins || !plugins.is_sequence()) {
            plugins = Ramulator::ConfigNode(Ramulator::ConfigNode::Seq{});
        }
        plugins.push_back(Ramulator::ConfigNode(Ramulator::ConfigNode::Map{
            {"impl", Ramulator::ConfigNode("GolemCommandStats")},
            {"backend_id", Ramulator::ConfigNode(backendId)},
        }));
        controller.set("controller_plugins", std::move(plugins));
        updatedControllers.push_back(std::move(controller));
    }
    memorySystem.set("controllers", std::move(updatedControllers));
    config.set("memory_system", std::move(memorySystem));
    return config;
}

} // namespace Ramulator


ramulator2Memory::ramulator2Memory(ComponentId_t id, Params &params) :
    SimpleMemBackend(id, params),
    addressOffset_(params.find<Addr>("address_offset", 0)),
    transactionBytes_(0),
    backendId_(params.find<unsigned>("backend_id", 0)),
    currentCycle_(0),
    completedReads_(0),
    completedWrites_(0),
    completedReadBytes_(0),
    completedWriteBytes_(0),
    firstReadArrivalCycle_(std::numeric_limits<uint64_t>::max()),
    lastReadCompleteCycle_(0),
    dependencyReadBlocked_(0),
    dependencyWriteBlocked_(0),
    concurrentAliasReads_(0)
{
    config_path = params.find<std::string>("configFile",
                                            NO_STRING_DEFINED);
    if (config_path == NO_STRING_DEFINED) {
        output->fatal(CALL_INFO, -1, "Ramulator2 Backend must define a 'configFile' file parameter\n");
    }

    Ramulator::ConfigNode config = Ramulator::Config::parse_config_file(config_path);
    config = Ramulator::addGolemCommandStats(std::move(config), backendId_);
    ramulator2_frontend = Ramulator::Factory::create_frontend(config);
    ramulator2_memorysystem = Ramulator::Factory::create_memory_system(config);

    ramulator2_frontend->connect_memory_system(ramulator2_memorysystem);
    ramulator2_memorysystem->connect_frontend(ramulator2_frontend);

    const unsigned txBytes = static_cast<unsigned>(ramulator2_memorysystem->get_tx_bytes());
    transactionBytes_ = txBytes;
    if (m_reqWidth != txBytes) {
        output->fatal(CALL_INFO, -1,
            "Ramulator2 backend request_width must equal the DRAM transaction size: request_width=%u, tx_bytes=%u\n",
            m_reqWidth, txBytes);
    }

    output->output(CALL_INFO,
        "Instantiated Ramulator2 node=%u config=%s tx_bytes=%u address_offset=%" PRIu64
        "\n",
        backendId_, config_path.c_str(), txBytes, static_cast<uint64_t>(addressOffset_));
}

bool ramulator2Memory::issueRequest(ReqId reqId, Addr addr, bool isWrite, unsigned numBytes){
    if (addr < addressOffset_) {
        output->fatal(CALL_INFO, -1,
            "Ramulator2 node=%u received address 0x%" PRIx64 " below address_offset 0x%" PRIx64 "\n",
            backendId_, static_cast<uint64_t>(addr), static_cast<uint64_t>(addressOffset_));
    }
    const Ramulator::Addr_t localAddr = static_cast<Ramulator::Addr_t>(addr - addressOffset_);
    const Ramulator::Addr_t transactionAddr =
        localAddr & ~static_cast<Ramulator::Addr_t>(transactionBytes_ - 1);
    // SST applies writes to its backing store only when the backend callback
    // fires. Keep dependencies involving a write ordered, but allow duplicate
    // reads of the same transaction to use Ramulator's normal parallel path.
    auto dependency = outstandingTransactions_.find(transactionAddr);
    if (isWrite) {
        if (dependency != outstandingTransactions_.end()) {
            dependencyWriteBlocked_++;
            return false;
        }
    } else if (dependency != outstandingTransactions_.end()) {
        if (dependency->second.writer) {
            dependencyReadBlocked_++;
            return false;
        }
        concurrentAliasReads_++;
    }

    pendingRequests_[reqId] = {currentCycle_, isWrite, numBytes, transactionAddr};
    const bool enqueue_success = ramulator2_frontend->receive_external_requests(
        isWrite ? 1 : 0,
        transactionAddr,
        0,
        [this, reqId](Ramulator::Request&) { ramulatorDone(reqId); },
        static_cast<int>(numBytes));
    if (enqueue_success) {
        auto& state = outstandingTransactions_[transactionAddr];
        if (isWrite) {
            state.writer = true;
        } else {
            state.readers++;
        }
    } else {
        pendingRequests_.erase(reqId);
    }

    output->debug(_L10_, "Ramulator2Backend: enqueue %s\n", enqueue_success ? "successful" : "unsuccessful");
    return enqueue_success;
}

bool ramulator2Memory::clock(Cycle_t cycle){
    currentCycle_ = cycle;
#ifdef __SST_DEBUG_OUTPUT__
    output->debug(_L10_, "Ramulator2Backend: Ticking memory system.\n");
#endif
    ramulator2_memorysystem->tick();
    completeReadyRequests();
    return false;
}

void ramulator2Memory::finish(){
    completeReadyRequests();
    uint64_t latencySum = 0;
    uint64_t latencyP95 = 0;
    uint64_t latencyP99 = 0;
    uint64_t latencyMax = 0;
    if (!readLatencies_.empty()) {
        std::sort(readLatencies_.begin(), readLatencies_.end());
        for (const auto latency : readLatencies_) latencySum += latency;
        latencyP95 = readLatencies_[static_cast<size_t>(0.95 * (readLatencies_.size() - 1) + 0.5)];
        latencyP99 = readLatencies_[static_cast<size_t>(0.99 * (readLatencies_.size() - 1) + 0.5)];
        latencyMax = readLatencies_.back();
    }
    const uint64_t firstRead = readLatencies_.empty() ? 0 : firstReadArrivalCycle_;
    const uint64_t readWindow = readLatencies_.empty() ? 0
        : (lastReadCompleteCycle_ >= firstReadArrivalCycle_
            ? lastReadCompleteCycle_ - firstReadArrivalCycle_ + 1 : 0);
    std::ostringstream backendSummary;
    backendSummary << "RAMULATOR2_BACKEND_SUMMARY node=" << backendId_
                   << " read_count=" << completedReads_
                   << " write_count=" << completedWrites_
                   << " read_bytes=" << completedReadBytes_
                   << " write_bytes=" << completedWriteBytes_
                   << " first_read_arrival_cycle=" << firstRead
                   << " last_read_complete_cycle=" << lastReadCompleteCycle_
                   << " read_window_cycles=" << readWindow
                   << " avg_read_latency_cycles="
                   << (readLatencies_.empty() ? 0 : latencySum / readLatencies_.size())
                   << " p95_read_latency_cycles=" << latencyP95
                   << " p99_read_latency_cycles=" << latencyP99
                   << " max_read_latency_cycles=" << latencyMax
                   << " dependency_read_blocked=" << dependencyReadBlocked_
                   << " dependency_write_blocked=" << dependencyWriteBlocked_
                   << " concurrent_alias_reads=" << concurrentAliasReads_
                   << " pending=" << pendingRequests_.size();
    std::cout << backendSummary.str() << std::endl;
    ramulator2_frontend->finalize();
    ramulator2_memorysystem->finalize();
    const Ramulator::ConfigNode stats = ramulator2_memorysystem->collect_stats();
    const Ramulator::ConfigNode controllers = stats["controller"];
    if (controllers && controllers.is_sequence()) {
        unsigned channel = 0;
        for (const auto& controller : controllers.seq()) {
            auto value = [&](const char* key) {
                return controller[key].as<std::string>("0");
            };
            std::ostringstream controllerSummary;
            controllerSummary << "RAMULATOR2_CONTROLLER_SUMMARY"
                              << " node=" << backendId_
                              << " channel=" << channel++
                              << " cycles=" << value("cycles")
                              << " read_row_hits=" << value("read_row_hits")
                              << " read_row_misses=" << value("read_row_misses")
                              << " read_row_conflicts=" << value("read_row_conflicts")
                              << " write_row_hits=" << value("write_row_hits")
                              << " write_row_misses=" << value("write_row_misses")
                              << " write_row_conflicts=" << value("write_row_conflicts")
                              << " read_queue_len_avg=" << value("read_queue_len_avg")
                              << " write_queue_len_avg=" << value("write_queue_len_avg")
                              << " avg_read_latency=" << value("avg_read_latency");
            std::cout << controllerSummary.str() << std::endl;
        }
    }
}

void ramulator2Memory::ramulatorDone(ReqId reqId) {
    completedRequests_.push_back(reqId);
}

void ramulator2Memory::completeReadyRequests() {
    while (!completedRequests_.empty()) {
        const ReqId reqId = completedRequests_.front();
        completedRequests_.pop_front();
        const auto it = pendingRequests_.find(reqId);
        if (it == pendingRequests_.end()) {
            output->fatal(CALL_INFO, -1,
                "Ramulator2 node=%u completed unknown request\n", backendId_);
        }
        const PendingRequest request = it->second;
        const auto transactionIt = outstandingTransactions_.find(request.transactionAddr);
        if (transactionIt == outstandingTransactions_.end() ||
            (request.isWrite ? !transactionIt->second.writer
                             : transactionIt->second.readers == 0)) {
            output->fatal(CALL_INFO, -1,
                "Ramulator2 node=%u transaction dependency underflow\n", backendId_);
        }
        if (request.isWrite) {
            transactionIt->second.writer = false;
        } else {
            transactionIt->second.readers--;
        }
        if (!transactionIt->second.writer && transactionIt->second.readers == 0) {
            outstandingTransactions_.erase(transactionIt);
        }
        if (request.isWrite) {
            completedWrites_++;
            completedWriteBytes_ += request.numBytes;
        } else {
            completedReads_++;
            completedReadBytes_ += request.numBytes;
            firstReadArrivalCycle_ = std::min<uint64_t>(firstReadArrivalCycle_, request.issueCycle);
            lastReadCompleteCycle_ = std::max<uint64_t>(lastReadCompleteCycle_, currentCycle_);
            readLatencies_.push_back(currentCycle_ - request.issueCycle);
        }
        pendingRequests_.erase(it);
        handleMemResponse(reqId);
    }
}
