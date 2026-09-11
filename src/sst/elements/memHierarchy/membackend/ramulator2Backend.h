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


#ifndef _H_SST_MEMH_RAMULATOR2_BACKEND
#define _H_SST_MEMH_RAMULATOR2_BACKEND

#include "sst/elements/memHierarchy/membackend/memBackend.h"


#include "ramulator/base/request.h"
#include "ramulator/frontend/i_frontend.h"
#include "ramulator/memory_system/i_memory_system.h"

#include <deque>
#include <map>
#include <vector>


#ifdef OLD_DEBUG
#define DEBUG OLD_DEBUG
#undef OLD_DEBUG
#endif

namespace SST {
namespace MemHierarchy {

class ramulator2Memory : public SimpleMemBackend {
public:
/* Element Library Info */
    SST_ELI_REGISTER_SUBCOMPONENT(ramulator2Memory, "memHierarchy", "ramulator2", SST_ELI_ELEMENT_VERSION(1,0,0),
            "Ramulator2-driven memory timings", SST::MemHierarchy::SimpleMemBackend)

    SST_ELI_DOCUMENT_PARAMS( MEMBACKEND_ELI_PARAMS,
            /* Own parameters */
            {"configFile",  "Name of the Ramulator2 device config file", NULL},
            {"address_offset", "Address subtracted before sending a request to Ramulator2", "0"},
            {"backend_id", "Memory-node identifier used in backend summaries", "0"} )

/* Begin class definition */
    ramulator2Memory(ComponentId_t id, Params &params);
    bool issueRequest(ReqId, Addr, bool, unsigned );
    virtual bool clock(Cycle_t cycle);
    virtual void finish();

protected:
    struct PendingRequest {
        Cycle_t issueCycle;
        bool isWrite;
        unsigned numBytes;
        Ramulator::Addr_t transactionAddr;
    };

    struct TransactionDependency {
        unsigned readers = 0;
        bool writer = false;
    };

    void ramulatorDone(ReqId reqId);
    void completeReadyRequests();

    std::string config_path;
    Ramulator::IFrontEnd* ramulator2_frontend;
    Ramulator::IMemorySystem* ramulator2_memorysystem;

    Addr addressOffset_;
    unsigned transactionBytes_;
    unsigned backendId_;
    Cycle_t currentCycle_;
    std::map<ReqId, PendingRequest> pendingRequests_;
    std::map<Ramulator::Addr_t, TransactionDependency> outstandingTransactions_;
    std::deque<ReqId> completedRequests_;
    std::vector<uint64_t> readLatencies_;
    uint64_t completedReads_;
    uint64_t completedWrites_;
    uint64_t completedReadBytes_;
    uint64_t completedWriteBytes_;
    uint64_t firstReadArrivalCycle_;
    uint64_t lastReadCompleteCycle_;
    uint64_t dependencyReadBlocked_;
    uint64_t dependencyWriteBlocked_;
    uint64_t concurrentAliasReads_;

private:
};

}
}

#endif
