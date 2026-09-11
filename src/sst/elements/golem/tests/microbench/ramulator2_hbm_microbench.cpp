#include "ramulator/base/config.h"
#include "ramulator/base/factory.h"
#include "ramulator/base/request.h"
#include "ramulator/frontend/i_frontend.h"
#include "ramulator/memory_system/i_memory_system.h"

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr uint64_t kTransactionBytes = 32;
constexpr double kClockGHz = 1.25;

uint64_t address(unsigned column, unsigned pseudoChannel, unsigned bankGroup,
                 unsigned bank, unsigned row = 0) {
    // RoBaRaCoCh consumes: 5 column, 1 pseudo-channel, 2 bank-group,
    // 2 bank, then row bits from the 32-byte transaction index.
    const uint64_t transaction =
        (column & 0x1fU) |
        (static_cast<uint64_t>(pseudoChannel & 0x1U) << 5) |
        (static_cast<uint64_t>(bankGroup & 0x3U) << 6) |
        (static_cast<uint64_t>(bank & 0x3U) << 8) |
        (static_cast<uint64_t>(row) << 10);
    return transaction * kTransactionBytes;
}

std::vector<uint64_t> controlledAddresses(const std::string& mode) {
    std::vector<uint64_t> addresses;
    const unsigned pseudoChannels = mode == "pseudo-channel" ? 2 : 1;
    const unsigned bankGroups = mode == "tccd-l" ? 1 : 4;
    for (unsigned column = 0; column < 32; ++column) {
        for (unsigned pc = 0; pc < pseudoChannels; ++pc) {
            for (unsigned bg = 0; bg < bankGroups; ++bg) {
                for (unsigned bank = 0; bank < 4; ++bank) {
                    addresses.push_back(address(column, pc, bg, bank));
                }
            }
        }
    }
    return addresses;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: ramulator2_hbm_microbench CONFIG "
                     "tccd-l|tccd-s|pseudo-channel|stack-roofline|stack-refresh [request-count]\n";
        return 2;
    }
    const std::string configPath = argv[1];
    const std::string mode = argv[2];
    if (mode != "tccd-l" && mode != "tccd-s" &&
        mode != "pseudo-channel" && mode != "stack-roofline" &&
        mode != "stack-refresh") {
        throw std::runtime_error("unknown mode: " + mode);
    }

    auto config = Ramulator::Config::parse_config_file(configPath);
    auto* frontend = Ramulator::Factory::create_frontend(config);
    auto* memory = Ramulator::Factory::create_memory_system(config);
    frontend->connect_memory_system(memory);
    memory->connect_frontend(frontend);
    if (memory->get_tx_bytes() != static_cast<int>(kTransactionBytes)) {
        throw std::runtime_error("microbenchmark requires 32-byte HBM transactions");
    }

    std::vector<uint64_t> addresses;
    if (mode == "stack-roofline" || mode == "stack-refresh") {
        const uint64_t count = argc >= 4 ? std::strtoull(argv[3], nullptr, 0) : 262144;
        addresses.reserve(count);
        for (uint64_t i = 0; i < count; ++i) addresses.push_back(i * kTransactionBytes);
    } else {
        addresses = controlledAddresses(mode);
    }

    uint64_t cycle = 0;
    uint64_t issued = 0;
    uint64_t completed = 0;
    uint64_t firstCompletion = 0;
    uint64_t lastCompletion = 0;
    while (completed < addresses.size()) {
        while (issued < addresses.size()) {
            const bool accepted = frontend->receive_external_requests(
                0, addresses[issued], 0,
                [&cycle, &completed, &firstCompletion, &lastCompletion](Ramulator::Request&) {
                    if (completed == 0) firstCompletion = cycle;
                    lastCompletion = cycle;
                    ++completed;
                },
                kTransactionBytes);
            if (!accepted) break;
            ++issued;
        }
        memory->tick();
        ++cycle;
        if (cycle > 1000000000ULL) throw std::runtime_error("microbenchmark timeout");
    }

    const uint64_t completionSpan = lastCompletion - firstCompletion + 1;
    const double bytesPerCycle = static_cast<double>(addresses.size()) * kTransactionBytes /
                                 static_cast<double>(completionSpan);
    const double bandwidthGBs = bytesPerCycle * kClockGHz;
    const double commandInterval = addresses.size() > 1
        ? static_cast<double>(lastCompletion - firstCompletion) /
              static_cast<double>(addresses.size() - 1)
        : 0.0;
    std::cout << std::fixed << std::setprecision(4)
              << "mode=" << mode
              << " requests=" << addresses.size()
              << " first_complete=" << firstCompletion
              << " last_complete=" << lastCompletion
              << " completion_span_cycles=" << completionSpan
              << " avg_command_interval_cycles=" << commandInterval
              << " bytes_per_cycle=" << bytesPerCycle
              << " bandwidth_GBps=" << bandwidthGBs << '\n';

    frontend->finalize();
    memory->finalize();
    return 0;
}
