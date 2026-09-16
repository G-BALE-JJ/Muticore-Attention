// Copyright 2013-2025 NTESS. Under the terms
// of Contract DE-NA0003525 with NTESS, the U.S.
// Government retains certain rights in this software.
//
// Copyright (c) 2013-2025, NTESS
// All rights reserved.
//
// Portions are copyright of other developers:
// See the file CONTRIBUTORS.TXT in the top level directory
// of the distribution for more information.
//
// This file is part of the SST software package. For license
// information, see the LICENSE file in the top level directory of the
// distribution.

#ifndef _MEMHIERARCHY_MEMNICBASE_SUBCOMPONENT_H_
#define _MEMHIERARCHY_MEMNICBASE_SUBCOMPONENT_H_

#include <string>
#include <unordered_map>
#include <queue>
#include <deque>
#include <map>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <cctype>
#include <limits>
#include <tuple>
#include <vector>

#include <sst/core/event.h>
#include <sst/core/output.h>
#include <sst/core/subcomponent.h>
#include <sst/core/interfaces/simpleNetwork.h>

#include "sst/elements/memHierarchy/memEventBase.h"
#include "sst/elements/memHierarchy/memEvent.h"
#include "sst/elements/memHierarchy/util.h"
#include "sst/elements/memHierarchy/memLinkBase.h"
#include "sst/elements/golem/globalmemory/globalmemory.h"

namespace SST {
namespace MemHierarchy {

/* MemNIC Base class
 *  Base class to handle initialization and endpoint management for different NICs
 */
class MemNICBase : public MemLinkBase {
    public:

#define MEMNICBASE_ELI_PARAMS MEMLINKBASE_ELI_PARAMS, \
        { "group",                       "(int) Group ID. See params 'sources' and 'destinations'. If not specified, the parent component will guess.", "1"},\
        { "sources",                     "(comma-separated list of ints) List of group IDs that serve as sources for this component. If not specified, defaults to 'group - 1'.", "group-1"},\
        { "destinations",                "(comma-separated list of ints) List of group IDs that serve as destinations for this component. If not specified, defaults to 'group + 1'.", "group+1"},\
        { "range_check",                 "(int) Enable initial check for overlapping memory ranges. 0=Disabled 1=Enabled", "1"},\
        { "golem_dma_response_vn",       "(int) VN for Golem DMA completion responses. If unset, derives VN1 when num_vns >= 2, otherwise VN0.", ""},\
        { "golem_dma_response_drain_limit", "(int) Max queued Golem DMA responses drained per opportunity. 0 means unlimited.", "0"},\
        { "golem_dma_response_priority_enable", "(int) Prioritize queued Golem DMA responses by semantic kind.", "0"},\
        { "golem_dma_kv_coalesce_enable", "(int) Coalesce identical in-flight Golem Attention K/V DMA chunks.", "0"},\
        { "golem_dma_kv_multicast_bytes_per_cycle", "(int) Aggregate bytes per owner cycle for shared K/V response fanout.", "256"},\
        { "golem_dma_kv_expected_consumers", "(int) Worker consumers per shared K/V chunk before its completed copy can be released.", "16"},\
        { "golem_dma_credit_cap",         "(int) Shared Golem DMA read credits owned by this memory-node NIC. 0 disables admission control.", "0"},\
        { "golem_dma_credit_chunk_bytes", "(int) Bytes represented by one Golem DMA read credit.", "16384"},\
        { "golem_dma_admission_limit", "(int) Max DMA reads admitted from the ingress queue per NIC clock. 0 means unlimited.", "0"},\
        { "golem_dma_window_priority_enable", "(int) Prioritize older DMA windows and round-robin workers before HBM admission.", "0"},\
        { "golem_dma_window_reorder_cycles", "(int) Cycles to collect requests for window-aware admission.", "512"},\
        { "golem_dma_tile_chunk_quantum", "(int) Same-window/tile/worker chunks admitted before round-robin advances.", "1"},\
        { "golem_dma_response_tile_priority_enable", "(int) Reorder ready DMA responses to complete the next worker tile first.", "0"},\
        { "golem_dma_response_reorder_cycles", "(int) Minimum response collection interval before tile-aware dequeue.", "0"},\
        { "golem_dma_response_max_starvation_cycles", "(int) Maximum ready-response age before FIFO fallback. 0 disables fallback.", "65536"},\
        { "golem_dma_admission_max_starvation_cycles", "(int) Maximum ingress age before oldest-admissible fallback. 0 disables fallback.", "4096"},\
        { "golem_dma_write_response_vn", "(int) VN used for Golem DMA write completions. Defaults to VN2 when available.", "2"},\
        { "network_vn_priority_order", "Comma-separated VN injection priority order. Empty preserves round-robin.", ""},\
        { "network_vn_starvation_vn", "VN allowed to bypass injection priority after a bounded wait.", "-1"},\
        { "network_vn_max_starvation_cycles", "Maximum injection queue age before starvation bypass. 0 disables bypass.", "0"}

        SST_ELI_REGISTER_SUBCOMPONENT_DERIVED_API(SST::MemHierarchy::MemNICBase, SST::MemHierarchy::MemLinkBase)

        /* Constructor */
        MemNICBase(ComponentId_t id, Params &params, TimeConverter* tc) : MemLinkBase(id, params, tc) {
            golem_dma_clock_factor_ = tc ? std::max<SimTime_t>(1, tc->getFactor()) : 1;
            build(params);
        }

        /* Destructor */
        virtual ~MemNICBase() { }

        // Router events
        class MemRtrEvent : public SST::Event {
            protected:
                MemEventBase * event;
            public:
                MemRtrEvent() : Event(), event(nullptr) { }
                MemRtrEvent(MemEventBase * ev) : Event(), event(ev) { }
                ~MemRtrEvent() {
                    if (event) {
                        delete event;
                    }
                }

                virtual Event* clone(void) override {
                    MemRtrEvent *mre = new MemRtrEvent(*this);
                    if (this->event != nullptr)
                        mre->event = this->event->clone();
                    else
                        mre->event = nullptr;
                    return mre;
                }

                void putEvent(MemEventBase* ev) {
                    event = ev;
                }

                MemEventBase* takeEvent() {
                    MemEventBase* tmp = event;
                    event = nullptr;
                    return tmp;
                }

                MemEventBase* inspectEvent() {
                    return event;
                }

                virtual bool hasClientData() const { return true; }

                virtual std::string toString() const override {
                    return event->toString();
                }

                void serialize_order(SST::Core::Serialization::serializer &ser) override {
                    Event::serialize_order(ser);
                    SST_SER(event);
                }

                ImplementSerializable(SST::MemHierarchy::MemNICBase::MemRtrEvent);
        };

        class InitMemRtrEvent : public MemRtrEvent {
            public:
                EndpointInfo info;

                InitMemRtrEvent() {}
                InitMemRtrEvent(EndpointInfo info) : MemRtrEvent(), info(info) { }

                virtual Event* clone(void) override {
                    InitMemRtrEvent * imre = new InitMemRtrEvent(*this);
                    if (this->event != nullptr)
                        imre->event = this->event->clone();
                    else
                        imre->event = nullptr;
                    return imre;
                }

                virtual bool hasClientData() const override { return false; }
                
                virtual std::string toString() const override {
                    return info.toString();
                }

                void serialize_order(SST::Core::Serialization::serializer & ser) override {
                    MemRtrEvent::serialize_order(ser);
                    SST_SER(info);
                }

                ImplementSerializable(SST::MemHierarchy::MemNICBase::InitMemRtrEvent);
        };

        // Init/complete functions
        // Send untimed data immediately if possible
        void sendUntimedData(MemEventInit* ev, bool broadcast, bool lookup_dst, 
                SST::Interfaces::SimpleNetwork * linkcontrol) {
            if (!broadcast && lookup_dst) {
                std::string dst = findTargetDestination(ev->getRoutingAddress());
                if (dst == "") {
                    // Hold this request until we know the right address
                    initWaitForDst.insert(ev);
                    return;
                }
                ev->setDst(dst);
            }

            MemRtrEvent * mre = new MemRtrEvent(ev);
            SST::Interfaces::SimpleNetwork::Request* req = new SST::Interfaces::SimpleNetwork::Request();
            if (broadcast) {
                req->dest = SST::Interfaces::SimpleNetwork::INIT_BROADCAST_ADDR;
            } else {
                req->dest = lookupNetworkAddress(ev->getDst());
            }
            req->givePayload(mre);
            if (!linkcontrol->isNetworkInitialized()) {
                untimed_send_queue_.push(req);
            } else {
                linkcontrol->sendUntimedData(req);
            }
        }

        virtual MemEventInit* recvUntimedData() {
            if (untimed_receive_queue_.size()) {
                MemRtrEvent * mre = untimed_receive_queue_.front();
                untimed_receive_queue_.pop();
                MemEventInit * ev = static_cast<MemEventInit*>(mre->takeEvent());
                delete mre;
                return ev;
            }
            return nullptr;
        }

        virtual bool isSource(std::string str) { /* Note this is only used during init so doesn't need to be fast */
            for (std::set<EndpointInfo>::iterator it = sourceEndpointInfo.begin(); it != sourceEndpointInfo.end(); it++) {
                if (it->name == str) return true;
            }
            return false;
        }

        virtual bool isDest(std::string str) { /* Note this is only used during init so doesn't need to be fast */
            for (std::set<EndpointInfo>::iterator it = destEndpointInfo.begin(); it != destEndpointInfo.end(); it++) {
                if (it->name == str) return true;
            }
            return false;
        }

        virtual bool isPeer(std::string str) {
            for (std::set<EndpointInfo>::iterator it = peerEndpointInfo.begin(); it != peerEndpointInfo.end(); it++) {
                if (it->name == str) return true;
            }
            return false;
        }
        
        virtual bool isClocked() { return true; } // Tell parent to trigger our clock

        virtual std::set<EndpointInfo>* getSources() { return &sourceEndpointInfo; }
        virtual std::set<EndpointInfo>* getDests() { return &destEndpointInfo; }
        virtual std::set<EndpointInfo>* getPeers() { return &peerEndpointInfo; }

        virtual std::string findTargetDestination(Addr addr) {
            for (std::set<EndpointInfo>::const_iterator it = destEndpointInfo.begin(); it != destEndpointInfo.end(); it++) {
                if (it->region.contains(addr)) return it->name;
            }
            return "";
        }

        virtual std::string getTargetDestination(Addr addr) {
            std::string dst = findTargetDestination(addr);
            if (dst != "") {
                return dst;
            }

            stringstream error;
            error << getName() + " (MemNICBase) cannot find a destination for address " << std::hex << addr << endl;
            error << "Known destinations: " << endl;
            for (std::set<EndpointInfo>::const_iterator it = destEndpointInfo.begin(); it != destEndpointInfo.end(); it++) {
                error << it->name << " " << it->region.toString() << endl;
            }
            dbg.fatal(CALL_INFO, -1, "%s", error.str().c_str());
            return "";
        }

        virtual bool isReachable(std::string dst) {
            return reachableNames.find(dst) != reachableNames.end();
        }
        
        virtual std::string getAvailableDestinationsAsString() {
            stringstream str;
            for (std::set<EndpointInfo>::const_iterator it = destEndpointInfo.begin(); it != destEndpointInfo.end(); it++) {
                str << it->name << " " << it->region.toString() << endl;
            }
            return str.str();
        }


    protected:
        virtual void addSource(EndpointInfo info) { 
            sourceEndpointInfo.insert(info);
            reachableNames.insert(info.name);
        }
        virtual void addDest(EndpointInfo info) { 
            destEndpointInfo.insert(info); 
            reachableNames.insert(info.name);
        }

        virtual void addPeer(EndpointInfo info) {
            peerEndpointInfo.insert(info);
        }

        virtual void addEndpoint(EndpointInfo info) { endpointInfo.insert(info); }

        virtual InitMemRtrEvent* createInitMemRtrEvent() {
            return new InitMemRtrEvent(info);
        }

        virtual void processInitMemRtrEvent(InitMemRtrEvent* imre) {

            if (sourceIDs.find(imre->info.id) != sourceIDs.end()) {
                addSource(imre->info);
                dbg.debug(_L10_, "%s (memNICBase) received source imre. Name: %s, Addr: %" PRIu64 ", ID: %" PRIu32 ", start: %" PRIu64 ", end: %" PRIu64 ", size: %" PRIu64 ", step: %" PRIu64 "\n",
                        getName().c_str(), imre->info.name.c_str(), imre->info.addr, imre->info.id, imre->info.region.start, imre->info.region.end, imre->info.region.interleaveSize, imre->info.region.interleaveStep);
            }
            
            if (destIDs.find(imre->info.id) != destIDs.end()) {
                addDest(imre->info);
                dbg.debug(_L10_, "%s (memNICBase) received dest imre. Name: %s, Addr: %" PRIu64 ", ID: %" PRIu32 ", start: %" PRIu64 ", end: %" PRIu64 ", size: %" PRIu64 ", step: %" PRIu64 "\n",
                        getName().c_str(), imre->info.name.c_str(), imre->info.addr, imre->info.id, imre->info.region.start, imre->info.region.end, imre->info.region.interleaveSize, imre->info.region.interleaveStep);
            }

            if (imre->info.id == info.id) {
                addPeer(imre->info);
            }
        }

        /* NIC initialization so that subclasses don't have to do this. Subclasses should call this during init() */
        virtual void nicInit(SST::Interfaces::SimpleNetwork * linkcontrol, unsigned int phase) {
            bool networkReady = linkcontrol->isNetworkInitialized();

            // After we've set up network and exchanged params, drain the send queue
            if (networkReady && initMsgSent) {
                while (!untimed_send_queue_.empty()) {
                    linkcontrol->sendUntimedData(untimed_send_queue_.front());
                    untimed_send_queue_.pop();
                }

                for (auto it = initWaitForDst.begin(); it != initWaitForDst.end();) {
                    std::string dst = findTargetDestination((*it)->getRoutingAddress());
                    if (dst != "") {
                        (*it)->setDst(dst);
                        MemRtrEvent * mre = new MemRtrEvent(*it);
                        SST::Interfaces::SimpleNetwork::Request* req = new SST::Interfaces::SimpleNetwork::Request();
                        req->dest = SST::Interfaces::SimpleNetwork::INIT_BROADCAST_ADDR;
                        req->givePayload(mre);
                        linkcontrol->sendUntimedData(req);
                        it = initWaitForDst.erase(it);
                    } else {
                        it++;
                    }
                }
            }

            // On first init round, send our region out to all others
            if (networkReady && !initMsgSent) {
                info.addr = linkcontrol->getEndpointID();
                InitMemRtrEvent *ev = createInitMemRtrEvent();

                SST::Interfaces::SimpleNetwork::Request * req = new SST::Interfaces::SimpleNetwork::Request();
                req->dest = SST::Interfaces::SimpleNetwork::INIT_BROADCAST_ADDR;
                req->src = info.addr;
                req->givePayload(ev);
                linkcontrol->sendUntimedData(req);
                initMsgSent = true;
            }

            // Expect different kinds of init events
            // 1. MemNIC - record these as needed and do not inform parent
            // 2. MemEventBase - only notify parent if sender is a src or dst for us
            // We should know since network is in order and NIC does its init before the
            // parents do
            while (SST::Interfaces::SimpleNetwork::Request *req = linkcontrol->recvUntimedData()) {
                Event * payload = req->takePayload();
                InitMemRtrEvent * imre = dynamic_cast<InitMemRtrEvent*>(payload);
                if (imre) {
                    // Record name->address map for all other endpoints
                    networkAddressMap.insert(std::make_pair(imre->info.name, imre->info.addr));
                    processInitMemRtrEvent(imre);
                    delete imre;
                } else {
                    MemRtrEvent * mre = static_cast<MemRtrEvent*>(payload);
                    MemEventInit *ev = static_cast<MemEventInit*>(mre->takeEvent()); // mre no longer has a copy of its event
                    dbg.debug(_L10_, "%s (memNICBase) received mre during init. %s\n", getName().c_str(), ev->getVerboseString(dlevel).c_str());

                    /*
                     * Event is for us if:
                     *  1. We are the dst
                     *  2. Broadcast (dst = "") and:
                     *      src is a src/dst and a coherence init message
                     *      src is a src/dst?
                     */
                    if (ev->getInitCmd() == MemEventInit::InitCommand::Region) {
                        delete ev;
                        delete mre;
                    } else if (ev->getInitCmd() == MemEventInit::InitCommand::Endpoint) {
                        // Intercept and record so that we know how to find this endpoint if we need to
                        // for noncacheable accesses. We don't need to record whether a particular region
                        // is noncacheable because the StandardMem interfaces will enforce
                        MemEventInitEndpoint * mEvEndPt = static_cast<MemEventInitEndpoint*>(ev);
                        if (!isDest(mEvEndPt->getSrc())) {
                            delete ev;
                            delete mre;
                        } else {
                            dbg.debug(_L10_, "%s received init message: %s\n", getName().c_str(), mEvEndPt->getVerboseString(dlevel).c_str());
                            std::vector<std::pair<MemRegion,bool>> regions = mEvEndPt->getRegions();
                            for (auto it = regions.begin(); it != regions.end(); it++) {
                                EndpointInfo epInfo;
                                epInfo.name = mEvEndPt->getSrc();
                                epInfo.addr = 0; // Not on a network so don't need it
                                epInfo.id = 0; // Not on a network so don't need it
                                epInfo.region = it->first;
                                addEndpoint(epInfo);
                            }
                            mre->putEvent(ev); // If we did not delete the Event, give it back to the MemRtrEvent
                            untimed_receive_queue_.push(mre); // Our component will forward on all its other ports
                        }
                    } else if ((ev->getCmd() == Command::NULLCMD && (isSource(ev->getSrc()) || isDest(ev->getSrc()))) || ev->getDst() == info.name) {
                        mre->putEvent(ev); // If we did not delete the Event, give it back to the MemRtrEvent
                        untimed_receive_queue_.push(mre);
                    }
                }
                delete req;
            }
        }

        /* NIC complete so that subclasses don't have to do this. Subclasses should call this during complete() */
        virtual void nicComplete(SST::Interfaces::SimpleNetwork * linkcontrol, unsigned int phase) {
            /* Drain untimed messages into untimed_recv_queue_ */
            SST::Interfaces::SimpleNetwork::Request * req;
            while ((req = linkcontrol->recvUntimedData()) != nullptr) {
                Event * payload = req->takePayload();
                MemRtrEvent * mre = dynamic_cast<MemRtrEvent*>(payload);
                
                if (mre) {
                    MemEventInit *ev = static_cast<MemEventInit*>(mre->takeEvent()); // mre no longer has a copy of its event
                    dbg.debug(_L10_, "%s (memNICBase) received mre during complete. %s\n", getName().c_str(), ev->getVerboseString(dlevel).c_str());
                    
                    /*
                     * Expected events: Flush (from dst or src) or writeback/data (from src)
                     */
                    if (ev->getInitCmd() == MemEventInit::InitCommand::Flush && !isDest(ev->getSrc())) { // Broadcast Flush not intended for us
                        delete ev;
                        delete mre;
                        continue;
                    }
                    mre->putEvent(ev);
                    untimed_receive_queue_.push(mre); // deliver event
                }
                delete req;
            }
        }
        
        // Setup
        // Clean up state generated during init() and perform some sanity checks
        virtual void setup() {
            /* Limit destinations to the memory regions reported by endpoint messages that came through them */
            
            std::set<std::string> names;
            std::set<EndpointInfo> newDests;
            for (auto it = endpointInfo.begin(); it != endpointInfo.end(); it++) {
                names.insert(it->name);
            }
            
            dbg.debug(_L10_, "Routing information for %s\n", getName().c_str());
            for (auto it = destEndpointInfo.begin(); it != destEndpointInfo.end(); it++) {
                //dbg.debug(_L10_, "    Orig Dest: %s\n", it->toString().c_str());
                if (names.find(it->name) != names.end()) {
                    for (auto et = endpointInfo.begin(); et != endpointInfo.end(); et++) {
                        if (it->name == et->name) {
                            std::set<MemRegion> reg = (it->region).intersect(et->region);
                            for (auto mt = reg.begin(); mt != reg.end(); mt++) {
                                EndpointInfo epInfo;
                                epInfo.name = it->name;
                                epInfo.addr = it->addr;
                                epInfo.id = it->id;
                                epInfo.region = (*mt);
                                newDests.insert(epInfo);
                            }
                        }
                    }
                } else {
                    newDests.insert(*it); // Copy into the new set
                }
            }
            destEndpointInfo = newDests;

            // This algorithm can take an extremely long time for some memory configurations.
            if (range_check > 0) {
                int stopAfter = 20; // This is error checking, if it takes too long, stop
                for (auto et = destEndpointInfo.begin(); et != destEndpointInfo.end(); et++) {
                    for (auto it = std::next(et,1); it != destEndpointInfo.end(); it++) {
                        if (it->name == et->name) continue; // Not a problem
                        if ((it->region).doesIntersect(et->region)) {
                            dbg.fatal(CALL_INFO, -1, "%s, Error: Found destinations on the network with overlapping address regions. Cannot generate routing table."
                                    "\n  Destination 1: %s\n  Destination 2: %s\n", 
                                    getName().c_str(), it->toString().c_str(), et->toString().c_str());
                        }
                        stopAfter--;
                        if (stopAfter == 0) {
                            stopAfter = -1;
                            break;
                        }
                    }
                    if (stopAfter <= 0) {
                        stopAfter = -1;
                        break;
                    }
                }
                if (stopAfter == -1)
                    dbg.debug(_L2_, "%s, Notice: Too many regions to complete error check for overlapping destination regions. Checked first 20 pairs. To disable this check set range_check parameter to 0\n",
                            getName().c_str());
            }

            for (auto it = networkAddressMap.begin(); it != networkAddressMap.end(); it++) {
                dbg.debug(_L10_, "    Address: %s -> %" PRIu64 "\n", it->first.c_str(), it->second);
            }
            for (auto it = sourceEndpointInfo.begin(); it != sourceEndpointInfo.end(); it++) {
                dbg.debug(_L10_, "    Source: %s\n", it->toString().c_str()); 
            }
            if (sourceEndpointInfo.empty()) dbg.debug(_L10_, "    Source: NONE\n");
            for (std::set<EndpointInfo>::const_iterator it = destEndpointInfo.begin(); it != destEndpointInfo.end(); it++) {
                dbg.debug(_L10_, "    Dest: %s\n", it->toString().c_str()); 
            }
            if (destEndpointInfo.empty()) dbg.debug(_L10_, "    Dest: NONE\n");
            for (auto it = peerEndpointInfo.begin(); it != peerEndpointInfo.end(); it++) {
                dbg.debug(_L10_, "    Peer: %s\n", it->toString().c_str());
            }
            if (peerEndpointInfo.empty()) dbg.debug(_L10_, "    Peer: NONE\n");
            for (auto it = endpointInfo.begin(); it != endpointInfo.end(); it++) {
                dbg.debug(_L10_, "    Endpoint: %s\n", it->toString().c_str()); 
            }

            if (!initWaitForDst.empty()) {
                dbg.fatal(CALL_INFO, -1, "%s, Error: Unable to find destination for init event %s\n",
                        getName().c_str(), (*initWaitForDst.begin())->getVerboseString(dlevel).c_str());
            }
        }

        // Lookup the network address for a given endpoint
        virtual uint64_t lookupNetworkAddress(const std::string &dst) const {
            std::unordered_map<std::string,uint64_t>::const_iterator it = networkAddressMap.find(dst);
            if (it == networkAddressMap.end()) {
                dbg.fatal(CALL_INFO, -1, "%s (MemNICBase), Network address for destination '%s' not found in networkAddressMap.\n", getName().c_str(), dst.c_str());
            }
            return it->second;
        }

        /*
         * Some helper functions to avoid needing to repeat code everywhere
         */

        // Get a packet header size parameter & error check it
        size_t extractPacketHeaderSize(Params &params, std::string pname, std::string defsize = "8B") {
            UnitAlgebra size = UnitAlgebra(params.find<std::string>(pname, defsize));
            if (!size.hasUnits("B"))
                dbg.fatal(CALL_INFO, -1, "Invalid param(%s): %s - must have units of bytes (B). SI units OK. You specified '%s'\n.",
                        getName().c_str(), pname.c_str(), size.toString().c_str());
            return size.getRoundedValue();
        }

        static int makeGolemMerlinTraceId(uint64_t requestId) {
            const uint64_t core = (requestId >> 56) & 0xffULL;
            const uint64_t slot = (requestId >> 48) & 0xffULL;
            const uint64_t node = (requestId >> 32) & 0xffULL;
            const uint64_t seq = requestId & 0xfULL;
            return static_cast<int>((core << 20) | (slot << 12) | (node << 4) | seq);
        }

        struct GolemDmaKvSubscriber {
            uint64_t returnAddr = 0;
            int returnEndpoint = -1;
            uint64_t completionFlagAddr = 0;
            uint64_t completionValue = 0;
            uint64_t requestId = 0;
            uint32_t creditUnits = 0;
            uint64_t ingressCycle = 0;
            SST::Golem::DmaRequestKind dmaRequestKind = SST::Golem::DmaRequestKind::Unknown;
            SST::Golem::DmaConsumerMetadata dmaConsumer;
        };

        struct GolemDmaBridgeInfo : GolemDmaKvSubscriber {
            uint32_t size = 0;
            bool isWrite = false;
            uint64_t hostAddr = 0;
            std::vector<GolemDmaKvSubscriber> kvSubscribers;
        };

        struct GolemDmaKvCacheKey {
            uint64_t hostAddr = 0;
            uint32_t size = 0;
            uint64_t jobId = 0;
            uint32_t targetKvTileIndex = 0;
            uint8_t operand = 0;

            bool operator<(const GolemDmaKvCacheKey& other) const {
                return std::tie(hostAddr, size, jobId, targetKvTileIndex, operand) <
                    std::tie(other.hostAddr, other.size, other.jobId,
                             other.targetKvTileIndex, other.operand);
            }
        };

        struct GolemDmaKvCacheEntry {
            std::vector<uint8_t> data;
            std::vector<uint64_t> servedConsumers;
        };

        static uint64_t golemDmaKvConsumerId(
                const SST::Golem::DmaConsumerMetadata& consumer) {
            return (static_cast<uint64_t>(consumer.worker) << 32) |
                consumer.targetQueryTile;
        }

        struct GolemDmaIngressRequest {
            uint64_t arrivalCycle = 0;
            uint64_t sourceEndpoint = 0;
            std::string sourceName;
            uint64_t addr = 0;
            uint32_t size = 0;
            uint64_t returnAddr = 0;
            int returnEndpoint = -1;
            uint64_t completionFlagAddr = 0;
            uint64_t completionValue = 0;
            uint64_t requestId = 0;
            uint32_t creditUnits = 0;
            bool creditBlocked = false;
            SST::Golem::DmaRequestKind dmaRequestKind = SST::Golem::DmaRequestKind::Unknown;
            SST::Golem::DmaConsumerMetadata dmaConsumer;
        };

        struct GolemDmaResponseRequest {
            SST::Interfaces::SimpleNetwork::Request* request = nullptr;
            uint64_t readyCycle = 0;
            uint64_t notBeforeCycle = 0;
            uint64_t requestId = 0;
            uint64_t returnAddr = 0;
            size_t length = 0;
            uint64_t ingressCycle = 0;
            uint32_t creditUnits = 0;
            SST::Golem::DmaRequestKind dmaRequestKind = SST::Golem::DmaRequestKind::Unknown;
            SST::Golem::DmaConsumerMetadata dmaConsumer;
        };

        struct GolemDmaConsumerProgress {
            uint32_t queryTileIndex = 0;
            uint32_t tile = 0;
        };

        struct GolemDmaBundleKey {
            uint64_t jobId = 0;
            uint32_t worker = 0;
            uint32_t queryTileIndex = 0;
            uint32_t tile = 0;

            bool operator==(const GolemDmaBundleKey& other) const {
                return jobId == other.jobId && worker == other.worker &&
                    queryTileIndex == other.queryTileIndex && tile == other.tile;
            }
        };

        struct GolemDmaBundleKeyHash {
            size_t operator()(const GolemDmaBundleKey& key) const {
                size_t value = std::hash<uint64_t>{}(key.jobId);
                const auto combine = [&value](uint32_t part) {
                    value ^= std::hash<uint32_t>{}(part) +
                        static_cast<size_t>(0x9e3779b9U) + (value << 6) + (value >> 2);
                };
                combine(key.worker);
                combine(key.queryTileIndex);
                combine(key.tile);
                return value;
            }
        };

        static GolemDmaBundleKey golemDmaBundleKey(uint64_t requestId) {
            return GolemDmaBundleKey{
                0, golemDmaWorkerId(requestId), golemDmaWindowId(requestId),
                golemDmaTileIndex(requestId)};
        }

        static GolemDmaBundleKey golemDmaBundleKey(
                uint64_t requestId, const SST::Golem::DmaConsumerMetadata& consumer) {
            if (consumer.valid == 0) return golemDmaBundleKey(requestId);
            return GolemDmaBundleKey{
                consumer.jobId, consumer.worker, consumer.targetQueryTile,
                consumer.targetKvTileIndex};
        }

        void updateGolemDmaConsumerProgress(
                const SST::Golem::DmaConsumerMetadata& consumer) {
            if (consumer.valid == 0) return;
            auto& current = golem_dma_consumer_progress_[
                std::make_pair(consumer.jobId, consumer.worker)];
            if (std::tie(current.queryTileIndex, current.tile) <
                std::tie(consumer.consumerQueryTile, consumer.consumerKvTileIndex)) {
                current.queryTileIndex = consumer.consumerQueryTile;
                current.tile = consumer.consumerKvTileIndex;
            }
        }

        uint8_t golemDmaConsumerDistance(
                const SST::Golem::DmaConsumerMetadata& consumer) const {
            if (consumer.valid == 0) return 3;
            uint32_t queryTileIndex = consumer.consumerQueryTile;
            uint32_t tile = consumer.consumerKvTileIndex;
            auto progress = golem_dma_consumer_progress_.find(
                std::make_pair(consumer.jobId, consumer.worker));
            if (progress != golem_dma_consumer_progress_.end()) {
                queryTileIndex = progress->second.queryTileIndex;
                tile = progress->second.tile;
            }
            if (consumer.targetQueryTile < queryTileIndex) return 0;
            if (consumer.targetQueryTile > queryTileIndex) return 3;
            if (consumer.targetKvTileIndex <= tile) return 0;
            return static_cast<uint8_t>(
                std::min<uint32_t>(consumer.targetKvTileIndex - tile, 3));
        }

        void releaseGolemDmaCredits(uint32_t creditUnits, uint64_t requestId) {
            if (creditUnits == 0) return;
            if (creditUnits > golem_dma_credit_cap_ - golem_dma_credit_available_) {
                dbg.fatal(CALL_INFO, -1,
                          "%s Golem DMA credit release overflow req=%" PRIu64
                          " units=%u available=%u cap=%u.\n",
                          getName().c_str(), requestId, creditUnits,
                          golem_dma_credit_available_, golem_dma_credit_cap_);
            }
            golem_dma_credit_available_ += creditUnits;
            golem_dma_credit_released_requests_++;
        }

        uint64_t golemDmaResponseNowCycle() const {
            return getCurrentSimCycle() / golem_dma_clock_factor_;
        }

        static bool isGolemAttentionKvRead(
                SST::Golem::DmaRequestKind kind,
                const SST::Golem::DmaConsumerMetadata& consumer) {
            return consumer.valid != 0 &&
                (kind == SST::Golem::DmaRequestKind::AttentionKv ||
                 kind == SST::Golem::DmaRequestKind::AttentionKvPrefetch) &&
                (consumer.operand == SST::Golem::DmaOperand::AttentionK ||
                 consumer.operand == SST::Golem::DmaOperand::AttentionV);
        }

        GolemDmaKvSubscriber makeGolemDmaKvSubscriber(
                const GolemDmaIngressRequest& ingress) {
            GolemDmaKvSubscriber subscriber;
            subscriber.returnAddr = ingress.returnAddr;
            subscriber.returnEndpoint = ingress.returnEndpoint;
            subscriber.completionFlagAddr = ingress.completionFlagAddr;
            subscriber.completionValue = ingress.completionValue;
            subscriber.requestId = ingress.requestId;
            subscriber.creditUnits = ingress.creditUnits;
            subscriber.ingressCycle =
                ingress.arrivalCycle / golem_dma_clock_factor_;
            subscriber.dmaRequestKind = ingress.dmaRequestKind;
            subscriber.dmaConsumer = ingress.dmaConsumer;
            return subscriber;
        }

        static GolemDmaKvCacheKey golemDmaKvCacheKey(
                uint64_t hostAddr, uint32_t size,
                const SST::Golem::DmaConsumerMetadata& consumer) {
            return GolemDmaKvCacheKey{
                hostAddr, size, consumer.jobId, consumer.targetKvTileIndex,
                static_cast<uint8_t>(consumer.operand)};
        }

        bool tryServeGolemDmaKvCache(const GolemDmaIngressRequest& ingress) {
            const auto key = golemDmaKvCacheKey(
                ingress.addr, ingress.size, ingress.dmaConsumer);
            auto cached = golem_dma_kv_cache_.find(key);
            if (cached == golem_dma_kv_cache_.end()) return false;

            const auto subscriber = makeGolemDmaKvSubscriber(ingress);
            auto* req = new SST::Interfaces::SimpleNetwork::Request();
            req->src = this->info.addr;
            req->dest = subscriber.returnEndpoint >= 0
                ? static_cast<uint64_t>(subscriber.returnEndpoint)
                : lookupNetworkAddress(ingress.sourceName);
            req->vn = golem_dma_response_vn;
            auto* response = new SST::Golem::NetworkDataEvent(
                SST::Golem::NetworkDataEvent::DMA_READ_COMPLETE,
                subscriber.returnAddr, cached->second.data.size(),
                cached->second.data, subscriber.returnAddr,
                subscriber.returnEndpoint, subscriber.completionFlagAddr,
                subscriber.completionValue, subscriber.requestId,
                subscriber.dmaRequestKind, subscriber.dmaConsumer);
            // A completed cache hit joins the already materialized multicast
            // block, so only destination metadata is injected here.
            req->size_in_bits =
                (sizeof(subscriber.returnAddr) + sizeof(size_t)) * 8;
            req->givePayload(response);
            if (golem_dma_trace && subscriber.requestId != 0) {
                req->setTraceID(makeGolemMerlinTraceId(subscriber.requestId));
                req->setTraceType(
                    SST::Interfaces::SimpleNetwork::Request::FULL);
            }

            GolemDmaResponseRequest queued;
            queued.request = req;
            queued.readyCycle = golemDmaResponseNowCycle();
            queued.notBeforeCycle = queued.readyCycle;
            queued.requestId = subscriber.requestId;
            queued.returnAddr = subscriber.returnAddr;
            queued.length = cached->second.data.size();
            queued.ingressCycle = subscriber.ingressCycle;
            queued.creditUnits = subscriber.creditUnits;
            queued.dmaRequestKind = subscriber.dmaRequestKind;
            queued.dmaConsumer = subscriber.dmaConsumer;
            golem_dma_send_queue_.push_back(std::move(queued));
            golem_dma_read_response_attempted_++;
            golem_dma_read_response_enqueued_++;
            golem_dma_read_response_enqueue_ticks_[req] = getCurrentSimCycle();
            golem_dma_response_max_queue_ = std::max(
                golem_dma_response_max_queue_, golem_dma_send_queue_.size());
            golem_dma_read_response_queue_high_water_ = std::max(
                golem_dma_read_response_queue_high_water_,
                static_cast<uint64_t>(
                    golem_dma_read_response_enqueue_ticks_.size()));
            if (subscriber.dmaConsumer.valid != 0 &&
                golemDmaConsumerDistance(subscriber.dmaConsumer) == 0) {
                golem_dma_response_late_ready_++;
            }
            golem_dma_kv_cache_hits_++;
            golem_dma_kv_multicast_receivers_++;

            auto& consumers = cached->second.servedConsumers;
            const uint64_t consumerId =
                golemDmaKvConsumerId(subscriber.dmaConsumer);
            if (std::find(consumers.begin(), consumers.end(), consumerId) ==
                consumers.end()) {
                consumers.push_back(consumerId);
            }
            if (consumers.size() >= golem_dma_kv_expected_consumers_) {
                golem_dma_kv_cache_.erase(cached);
            }
            return true;
        }

        bool tryCoalesceGolemDmaRead(const GolemDmaIngressRequest& ingress) {
            if (golem_dma_kv_coalesce_enable_ == 0 ||
                !isGolemAttentionKvRead(
                    ingress.dmaRequestKind, ingress.dmaConsumer)) {
                return false;
            }
            if (tryServeGolemDmaKvCache(ingress)) return true;
            for (auto& pending : golem_dma_pending) {
                GolemDmaBridgeInfo& info = pending.second;
                if (info.isWrite || info.hostAddr != ingress.addr ||
                    info.size != ingress.size || info.kvSubscribers.empty() ||
                    info.dmaConsumer.jobId != ingress.dmaConsumer.jobId ||
                    info.dmaConsumer.targetQueryTile !=
                        ingress.dmaConsumer.targetQueryTile ||
                    info.dmaConsumer.targetKvTileIndex !=
                        ingress.dmaConsumer.targetKvTileIndex ||
                    info.dmaConsumer.operand != ingress.dmaConsumer.operand) {
                    continue;
                }
                const auto duplicate = std::find_if(
                    info.kvSubscribers.begin(), info.kvSubscribers.end(),
                    [&ingress](const GolemDmaKvSubscriber& subscriber) {
                        return subscriber.requestId == ingress.requestId &&
                            subscriber.returnEndpoint == ingress.returnEndpoint;
                    });
                if (duplicate != info.kvSubscribers.end()) {
                    releaseGolemDmaCredits(ingress.creditUnits, ingress.requestId);
                    return true;
                }
                info.kvSubscribers.push_back(makeGolemDmaKvSubscriber(ingress));
                golem_dma_response_pending_by_bundle_[
                    golemDmaBundleKey(
                        ingress.requestId, ingress.dmaConsumer)]++;
                golem_dma_kv_coalesced_requests_++;
                return true;
            }
            return false;
        }

        MemEvent* createGolemDmaRead(const GolemDmaIngressRequest& ingress) {
            if (tryCoalesceGolemDmaRead(ingress)) return nullptr;
            auto* me = new MemEvent(ingress.sourceName, ingress.addr, ingress.addr, Command::GetS, ingress.size);
            me->setFlag(MemEventBase::F_NONCACHEABLE);
            GolemDmaBridgeInfo info;
            info.returnAddr = ingress.returnAddr;
            info.returnEndpoint = ingress.returnEndpoint;
            info.completionFlagAddr = ingress.completionFlagAddr;
            info.completionValue = ingress.completionValue;
            info.requestId = ingress.requestId;
            info.size = ingress.size;
            info.creditUnits = ingress.creditUnits;
            info.isWrite = false;
            info.hostAddr = ingress.addr;
            info.ingressCycle = ingress.arrivalCycle / golem_dma_clock_factor_;
            info.dmaRequestKind = ingress.dmaRequestKind;
            info.dmaConsumer = ingress.dmaConsumer;
            if (golem_dma_kv_coalesce_enable_ != 0 &&
                isGolemAttentionKvRead(
                    ingress.dmaRequestKind, ingress.dmaConsumer)) {
                info.kvSubscribers.push_back(
                    makeGolemDmaKvSubscriber(ingress));
                golem_dma_kv_physical_reads_++;
            }
            golem_dma_pending.emplace(me->getID(), info);
            golem_dma_response_pending_by_bundle_[
                golemDmaBundleKey(info.requestId, info.dmaConsumer)]++;
            return me;
        }

        uint32_t golemDmaCreditUnits(uint32_t bytes) const {
            const uint32_t chunk = std::max<uint32_t>(golem_dma_credit_chunk_bytes_, 1u);
            return std::max<uint32_t>(1u, (bytes + chunk - 1u) / chunk);
        }

        static uint8_t golemDmaWorkerId(uint64_t requestId) {
            return static_cast<uint8_t>((requestId >> 56) & 0xffULL);
        }

        static uint32_t golemDmaWindowId(uint64_t requestId) {
            constexpr uint32_t tileBits = 8;
            return static_cast<uint32_t>((requestId & 0xffffffffULL) >> tileBits);
        }

        static uint8_t golemDmaRequestSlot(uint64_t requestId) {
            return static_cast<uint8_t>((requestId >> 48) & 0xffULL);
        }

        static uint8_t golemDmaTileIndex(uint64_t requestId) {
            return static_cast<uint8_t>(requestId & 0xffULL);
        }

        bool hasGolemDmaIngress() const {
            return !golem_dma_ingress_queue_.empty();
        }

        bool golemDmaIngressCanProgress(uint64_t currentCycle) const {
            if (golem_dma_window_priority_enable_ != 0 && !golem_dma_ingress_queue_.empty()) {
                const bool hasConsumerMetadata = std::any_of(
                    golem_dma_ingress_queue_.begin(), golem_dma_ingress_queue_.end(),
                    [](const GolemDmaIngressRequest& ingress) {
                        return ingress.dmaConsumer.valid != 0;
                    });
                if (hasConsumerMetadata) {
                    uint64_t firstArrival = std::numeric_limits<uint64_t>::max();
                    for (const auto& ingress : golem_dma_ingress_queue_) {
                        firstArrival = std::min(firstArrival, ingress.arrivalCycle);
                    }
                    if (currentCycle <= firstArrival + golem_dma_window_reorder_cycles_) {
                        return true;
                    }
                    return std::any_of(
                        golem_dma_ingress_queue_.begin(), golem_dma_ingress_queue_.end(),
                        [this](const GolemDmaIngressRequest& ingress) {
                            return ingress.creditUnits <= golem_dma_credit_available_;
                        });
                }
                uint32_t oldestWindow = std::numeric_limits<uint32_t>::max();
                uint64_t firstArrival = std::numeric_limits<uint64_t>::max();
                for (const auto& ingress : golem_dma_ingress_queue_) {
                    const uint32_t window = golemDmaWindowId(ingress.requestId);
                    if (window < oldestWindow) {
                        oldestWindow = window;
                        firstArrival = ingress.arrivalCycle;
                    } else if (window == oldestWindow) {
                        firstArrival = std::min(firstArrival, ingress.arrivalCycle);
                    }
                }
                if (currentCycle <= firstArrival + golem_dma_window_reorder_cycles_) {
                    return true;
                }
                for (const auto& ingress : golem_dma_ingress_queue_) {
                    if (golemDmaWindowId(ingress.requestId) == oldestWindow &&
                        ingress.creditUnits <= golem_dma_credit_available_) {
                        return true;
                    }
                }
                return false;
            }
            for (const auto& ingress : golem_dma_ingress_queue_) {
                if (ingress.arrivalCycle >= currentCycle || ingress.creditUnits <= golem_dma_credit_available_) {
                    return true;
                }
            }
            return false;
        }

        std::vector<MemEventBase*> admitGolemDmaIngress(uint64_t currentCycle) {
            std::vector<MemEventBase*> admitted;
            if (golem_dma_ingress_queue_.empty()) {
                return admitted;
            }
            const bool hasConsumerMetadata = std::any_of(
                golem_dma_ingress_queue_.begin(), golem_dma_ingress_queue_.end(),
                [](const GolemDmaIngressRequest& ingress) {
                    return ingress.dmaConsumer.valid != 0;
                });
            if (golem_dma_window_priority_enable_ != 0 && hasConsumerMetadata) {
                while (!golem_dma_ingress_queue_.empty() &&
                       (golem_dma_admission_limit_ == 0 ||
                        admitted.size() < golem_dma_admission_limit_)) {
                    size_t chosen = golem_dma_ingress_queue_.size();
                    bool starvationFallback = false;

                    if (golem_dma_admission_max_starvation_cycles_ != 0) {
                        for (size_t idx = 0; idx < golem_dma_ingress_queue_.size(); ++idx) {
                            const auto& ingress = golem_dma_ingress_queue_[idx];
                            if (ingress.creditUnits > golem_dma_credit_available_ ||
                                currentCycle < ingress.arrivalCycle +
                                    golem_dma_admission_max_starvation_cycles_) {
                                continue;
                            }
                            if (chosen == golem_dma_ingress_queue_.size() ||
                                std::tie(ingress.arrivalCycle, ingress.requestId) <
                                    std::tie(golem_dma_ingress_queue_[chosen].arrivalCycle,
                                             golem_dma_ingress_queue_[chosen].requestId)) {
                                chosen = idx;
                            }
                        }
                        starvationFallback = chosen != golem_dma_ingress_queue_.size();
                    }

                    if (!starvationFallback && golem_dma_bundle_active_) {
                        for (size_t idx = 0; idx < golem_dma_ingress_queue_.size(); ++idx) {
                            const auto& ingress = golem_dma_ingress_queue_[idx];
                            if (ingress.creditUnits <= golem_dma_credit_available_ &&
                                golemDmaBundleKey(ingress.requestId, ingress.dmaConsumer) ==
                                    golem_dma_bundle_key_) {
                                chosen = idx;
                                break;
                            }
                        }
                        if (chosen == golem_dma_ingress_queue_.size()) {
                            golem_dma_bundle_active_ = false;
                            golem_dma_bundle_admitted_ = 0;
                        }
                    }

                    if (chosen == golem_dma_ingress_queue_.size()) {
                        uint16_t bestWorkerDistance = std::numeric_limits<uint16_t>::max();
                        for (size_t idx = 0; idx < golem_dma_ingress_queue_.size(); ++idx) {
                            const auto& ingress = golem_dma_ingress_queue_[idx];
                            if (ingress.creditUnits > golem_dma_credit_available_) continue;
                            const uint8_t distance = golemDmaConsumerDistance(ingress.dmaConsumer);
                            const uint16_t workerDistance = ingress.dmaConsumer.valid != 0
                                ? static_cast<uint16_t>(static_cast<uint8_t>(
                                      ingress.dmaConsumer.worker - golem_dma_worker_cursor_))
                                : std::numeric_limits<uint8_t>::max();
                            if (chosen == golem_dma_ingress_queue_.size() ||
                                std::make_tuple(distance, workerDistance,
                                                ingress.dmaConsumer.targetQueryTile,
                                                ingress.dmaConsumer.targetKvTileIndex,
                                                ingress.arrivalCycle, ingress.requestId) <
                                    std::make_tuple(
                                        golemDmaConsumerDistance(
                                            golem_dma_ingress_queue_[chosen].dmaConsumer),
                                        bestWorkerDistance,
                                        golem_dma_ingress_queue_[chosen].dmaConsumer.targetQueryTile,
                                        golem_dma_ingress_queue_[chosen].dmaConsumer.targetKvTileIndex,
                                        golem_dma_ingress_queue_[chosen].arrivalCycle,
                                        golem_dma_ingress_queue_[chosen].requestId)) {
                                chosen = idx;
                                bestWorkerDistance = workerDistance;
                            }
                        }
                        if (chosen != golem_dma_ingress_queue_.size()) {
                            const auto& ingress = golem_dma_ingress_queue_[chosen];
                            golem_dma_bundle_active_ = true;
                            golem_dma_bundle_key_ = golemDmaBundleKey(
                                ingress.requestId, ingress.dmaConsumer);
                            golem_dma_bundle_admitted_ = 0;
                            golem_dma_bundle_turns_++;
                        }
                    }

                    if (chosen == golem_dma_ingress_queue_.size()) {
                        for (auto& ingress : golem_dma_ingress_queue_) {
                            if (ingress.creditUnits > golem_dma_credit_available_ &&
                                !ingress.creditBlocked) {
                                ingress.creditBlocked = true;
                                golem_dma_credit_blocked_requests_++;
                            }
                        }
                        break;
                    }

                    GolemDmaIngressRequest ingress = std::move(
                        golem_dma_ingress_queue_[chosen]);
                    golem_dma_ingress_queue_.erase(
                        golem_dma_ingress_queue_.begin() +
                        static_cast<std::ptrdiff_t>(chosen));
                    const uint8_t distance = golemDmaConsumerDistance(ingress.dmaConsumer);
                    const uint64_t waitCycles = currentCycle - ingress.arrivalCycle;
                    golem_dma_consumer_distance_wait_cycles_[distance] += waitCycles;
                    golem_dma_consumer_distance_admissions_[distance]++;
                    if (ingress.creditBlocked) {
                        golem_dma_credit_blocked_cycles_ += waitCycles;
                    }
                    if (starvationFallback) golem_dma_tile_starvation_++;
                    golem_dma_credit_available_ -= ingress.creditUnits;
                    golem_dma_credit_admitted_requests_++;
                    golem_dma_credit_max_used_ = std::max<uint32_t>(
                        golem_dma_credit_max_used_,
                        golem_dma_credit_cap_ - golem_dma_credit_available_);
                    golem_dma_bundle_admitted_++;
                    golem_dma_bundle_admissions_++;
                    if (golem_dma_bundle_admitted_ >= golem_dma_tile_chunk_quantum_ ||
                        starvationFallback) {
                        if (ingress.dmaConsumer.valid != 0) {
                            golem_dma_worker_cursor_ = static_cast<uint8_t>(
                                ingress.dmaConsumer.worker + 1u);
                        }
                        golem_dma_bundle_active_ = false;
                        golem_dma_bundle_admitted_ = 0;
                    }
                    if (auto* read = createGolemDmaRead(ingress)) {
                        admitted.push_back(read);
                    }
                }
                return admitted;
            }
            if (golem_dma_window_priority_enable_ != 0) {
                while (!golem_dma_ingress_queue_.empty() &&
                       (golem_dma_admission_limit_ == 0 ||
                        admitted.size() < golem_dma_admission_limit_)) {
                    uint32_t oldestWindow = std::numeric_limits<uint32_t>::max();
                    uint64_t firstArrival = std::numeric_limits<uint64_t>::max();
                    for (const auto& ingress : golem_dma_ingress_queue_) {
                        const uint32_t window = golemDmaWindowId(ingress.requestId);
                        if (window < oldestWindow) {
                            oldestWindow = window;
                            firstArrival = ingress.arrivalCycle;
                        } else if (window == oldestWindow) {
                            firstArrival = std::min(firstArrival, ingress.arrivalCycle);
                        }
                    }

                    // The bounded collection interval lets a late worker's window N
                    // overtake already queued window N+1 requests.
                    if (currentCycle <= firstArrival + golem_dma_window_reorder_cycles_) {
                        break;
                    }

                    size_t chosen = golem_dma_ingress_queue_.size();
                    if (golem_dma_bundle_active_ && golem_dma_bundle_window_ != oldestWindow) {
                        golem_dma_bundle_active_ = false;
                        golem_dma_bundle_admitted_ = 0;
                    }

                    if (golem_dma_bundle_active_) {
                        bool matchingQueued = false;
                        for (size_t idx = 0; idx < golem_dma_ingress_queue_.size(); ++idx) {
                            const auto& ingress = golem_dma_ingress_queue_[idx];
                            if (golemDmaWindowId(ingress.requestId) != golem_dma_bundle_window_ ||
                                golemDmaWorkerId(ingress.requestId) != golem_dma_bundle_worker_ ||
                                golemDmaTileIndex(ingress.requestId) != golem_dma_bundle_tile_) {
                                continue;
                            }
                            matchingQueued = true;
                            if (ingress.creditUnits > golem_dma_credit_available_) {
                                continue;
                            }
                            if (chosen == golem_dma_ingress_queue_.size()) {
                                chosen = idx;
                                continue;
                            }
                            const auto& incumbent = golem_dma_ingress_queue_[chosen];
                            if (std::make_tuple(golemDmaRequestSlot(ingress.requestId), ingress.addr,
                                                ingress.arrivalCycle, ingress.requestId)
                                < std::make_tuple(golemDmaRequestSlot(incumbent.requestId), incumbent.addr,
                                                  incumbent.arrivalCycle, incumbent.requestId)) {
                                chosen = idx;
                            }
                        }
                        if (chosen == golem_dma_ingress_queue_.size() && !matchingQueued) {
                            golem_dma_worker_cursor_ = static_cast<uint8_t>(golem_dma_bundle_worker_ + 1u);
                            golem_dma_bundle_active_ = false;
                            golem_dma_bundle_admitted_ = 0;
                            continue;
                        }
                    } else {
                        uint16_t bestDistance = std::numeric_limits<uint16_t>::max();
                        for (size_t idx = 0; idx < golem_dma_ingress_queue_.size(); ++idx) {
                            const auto& ingress = golem_dma_ingress_queue_[idx];
                            if (golemDmaWindowId(ingress.requestId) != oldestWindow ||
                                ingress.creditUnits > golem_dma_credit_available_) {
                                continue;
                            }
                            const uint8_t worker = golemDmaWorkerId(ingress.requestId);
                            const uint16_t distance = static_cast<uint8_t>(worker - golem_dma_worker_cursor_);
                            if (chosen == golem_dma_ingress_queue_.size() || distance < bestDistance) {
                                chosen = idx;
                                bestDistance = distance;
                                continue;
                            }
                            if (distance == bestDistance) {
                                const auto& incumbent = golem_dma_ingress_queue_[chosen];
                                if (std::make_tuple(golemDmaTileIndex(ingress.requestId),
                                                    golemDmaRequestSlot(ingress.requestId),
                                                    ingress.arrivalCycle, ingress.addr, ingress.requestId)
                                    < std::make_tuple(golemDmaTileIndex(incumbent.requestId),
                                                      golemDmaRequestSlot(incumbent.requestId),
                                                      incumbent.arrivalCycle, incumbent.addr,
                                                      incumbent.requestId)) {
                                    chosen = idx;
                                }
                            }
                        }
                        if (chosen != golem_dma_ingress_queue_.size()) {
                            const auto& ingress = golem_dma_ingress_queue_[chosen];
                            golem_dma_bundle_active_ = true;
                            golem_dma_bundle_window_ = golemDmaWindowId(ingress.requestId);
                            golem_dma_bundle_worker_ = golemDmaWorkerId(ingress.requestId);
                            golem_dma_bundle_tile_ = golemDmaTileIndex(ingress.requestId);
                            golem_dma_bundle_admitted_ = 0;
                            golem_dma_bundle_turns_++;
                        }
                    }

                    if (chosen == golem_dma_ingress_queue_.size()) {
                        for (auto& ingress : golem_dma_ingress_queue_) {
                            if (golemDmaWindowId(ingress.requestId) == oldestWindow &&
                                ingress.creditUnits > golem_dma_credit_available_ &&
                                !ingress.creditBlocked) {
                                ingress.creditBlocked = true;
                                golem_dma_credit_blocked_requests_++;
                            }
                        }
                        break;
                    }

                    GolemDmaIngressRequest ingress = std::move(golem_dma_ingress_queue_[chosen]);
                    if (chosen != 0) {
                        golem_dma_window_priority_reorders_++;
                    }
                    golem_dma_ingress_queue_.erase(
                        golem_dma_ingress_queue_.begin() + static_cast<std::ptrdiff_t>(chosen));
                    golem_dma_credit_available_ -= ingress.creditUnits;
                    golem_dma_credit_admitted_requests_++;
                    golem_dma_credit_max_used_ = std::max<uint32_t>(
                        golem_dma_credit_max_used_, golem_dma_credit_cap_ - golem_dma_credit_available_);
                    golem_dma_bundle_admitted_++;
                    golem_dma_bundle_admissions_++;
                    if (golem_dma_bundle_admitted_ >= golem_dma_tile_chunk_quantum_) {
                        golem_dma_worker_cursor_ = static_cast<uint8_t>(golem_dma_bundle_worker_ + 1u);
                        golem_dma_bundle_active_ = false;
                        golem_dma_bundle_admitted_ = 0;
                    }
                    if (auto* read = createGolemDmaRead(ingress)) {
                        admitted.push_back(read);
                    }
                }
                return admitted;
            }

            std::sort(golem_dma_ingress_queue_.begin(), golem_dma_ingress_queue_.end(),
                [](const GolemDmaIngressRequest& lhs, const GolemDmaIngressRequest& rhs) {
                    return std::tie(lhs.arrivalCycle, lhs.requestId, lhs.sourceEndpoint, lhs.addr,
                                    lhs.returnEndpoint, lhs.returnAddr, lhs.completionFlagAddr,
                                    lhs.completionValue)
                         < std::tie(rhs.arrivalCycle, rhs.requestId, rhs.sourceEndpoint, rhs.addr,
                                    rhs.returnEndpoint, rhs.returnAddr, rhs.completionFlagAddr,
                                    rhs.completionValue);
                });

            std::vector<GolemDmaIngressRequest> waiting;
            waiting.reserve(golem_dma_ingress_queue_.size());
            for (auto& ingress : golem_dma_ingress_queue_) {
                if (golem_dma_admission_limit_ != 0 && admitted.size() >= golem_dma_admission_limit_) {
                    waiting.push_back(std::move(ingress));
                    continue;
                }
                // Defer one owner clock so every request delivered in the same
                // cycle participates in the deterministic ordering above.
                if (ingress.arrivalCycle >= currentCycle) {
                    waiting.push_back(std::move(ingress));
                    continue;
                }
                if (ingress.creditUnits > golem_dma_credit_available_) {
                    if (!ingress.creditBlocked) {
                        ingress.creditBlocked = true;
                        golem_dma_credit_blocked_requests_++;
                    }
                    waiting.push_back(std::move(ingress));
                    continue;
                }
                golem_dma_credit_available_ -= ingress.creditUnits;
                golem_dma_credit_admitted_requests_++;
                golem_dma_credit_max_used_ = std::max<uint32_t>(
                    golem_dma_credit_max_used_, golem_dma_credit_cap_ - golem_dma_credit_available_);
                if (auto* read = createGolemDmaRead(ingress)) {
                    admitted.push_back(read);
                }
            }
            golem_dma_ingress_queue_.swap(waiting);
            return admitted;
        }

        void emitGolemDmaCreditSummary() const {
            if (golem_dma_credit_cap_ == 0) {
                return;
            }
            fprintf(stdout,
                    "GOLEM_MEMNIC_DMA_CREDIT_CONSERVATION name=%s cap=%u available=%u"
                    " admitted=%" PRIu64 " released=%" PRIu64 " pending=%zu\n",
                    getName().c_str(), golem_dma_credit_cap_, golem_dma_credit_available_,
                    golem_dma_credit_admitted_requests_,
                    golem_dma_credit_released_requests_,
                    golem_dma_ingress_queue_.size());
            fprintf(stdout,
                    "[memNICBase bridge] CREDIT_OWNER_SUMMARY name=%s group=%u cap=%u available=%u"
                    " chunk_bytes=%u admitted=%" PRIu64 " released=%" PRIu64
                    " blocked_requests=%" PRIu64 " max_used=%u max_queue=%zu pending=%zu"
                    " window_priority=%u reorder_cycles=%u priority_reorders=%" PRIu64
                    " tile_quantum=%u bundle_turns=%" PRIu64 " bundle_admissions=%" PRIu64
                    " response_tile_priority=%u response_reorder_cycles=%u response_quantum=%u"
                    " response_reorders=%" PRIu64 " response_bundle_turns=%" PRIu64
                    " response_sent=%" PRIu64 " response_max_queue=%zu"
                    " response_hold_mean=%" PRIu64 " response_hold_max=%" PRIu64 "\n",
                    getName().c_str(), info.id, golem_dma_credit_cap_, golem_dma_credit_available_,
                    golem_dma_credit_chunk_bytes_, golem_dma_credit_admitted_requests_,
                    golem_dma_credit_released_requests_, golem_dma_credit_blocked_requests_,
                    golem_dma_credit_max_used_, golem_dma_credit_max_queue_,
                    golem_dma_ingress_queue_.size(), golem_dma_window_priority_enable_,
                    golem_dma_window_reorder_cycles_, golem_dma_window_priority_reorders_,
                    golem_dma_tile_chunk_quantum_, golem_dma_bundle_turns_,
                    golem_dma_bundle_admissions_,
                    golem_dma_response_tile_priority_enable_,
                    golem_dma_response_reorder_cycles_, golem_dma_tile_chunk_quantum_,
                    golem_dma_response_reorders_, golem_dma_response_bundle_turns_,
                    golem_dma_response_sent_, golem_dma_response_max_queue_,
                    golem_dma_response_sent_ == 0 ? 0 :
                        golem_dma_response_hold_cycles_sum_ / golem_dma_response_sent_,
                    golem_dma_response_hold_cycles_max_);
        }

        // Drain a send queue
        static uint8_t golemDmaResponsePriority(SST::Interfaces::SimpleNetwork::Request* req) {
            auto* nd = dynamic_cast<SST::Golem::NetworkDataEvent*>(req->inspectPayload());
            if (nd == nullptr) return 4;
            switch (nd->getDmaRequestKind()) {
                case SST::Golem::DmaRequestKind::AttentionKv: return 0;
                case SST::Golem::DmaRequestKind::AttentionQuery: return 1;
                case SST::Golem::DmaRequestKind::AttentionOutput: return 2;
                case SST::Golem::DmaRequestKind::AttentionKvPrefetch: return 3;
                default: return 4;
            }
        }

        void drainQueue(std::queue<SST::Interfaces::SimpleNetwork::Request*>* queue, SST::Interfaces::SimpleNetwork* linkcontrol, size_t maxSends = 0) {
            size_t sends = 0;
            while (!(queue->empty()) && (maxSends == 0 || sends < maxSends)) {
                size_t selectedOffset = 0;
                const size_t selectedQueueSize = queue->size();
                SST::Interfaces::SimpleNetwork::Request* head = queue->front();
#ifdef __SST_DEBUG_OUTPUT__
                MemEventBase* ev = (static_cast<MemRtrEvent*>(head->inspectPayload()))->inspectEvent();
                std::string debugEvStr = ev ? ev->getBriefString() : "";
                uint64_t dst = head->dest;
                bool doDebug = ev ? mem_h_is_debug_event(ev) : false;
#endif
                uint64_t golemTraceReq = 0;
                uint64_t golemTraceAddr = 0;
                size_t golemTraceLen = 0;
                uint8_t golemTraceKind = 0;
                int golemTraceId = head->getTraceID();
                if (auto* nd = dynamic_cast<SST::Golem::NetworkDataEvent*>(head->inspectPayload())) {
                    if (nd->getType() == SST::Golem::NetworkDataEvent::DMA_READ_COMPLETE) {
                        golemTraceReq = nd->getRequestId();
                        golemTraceAddr = nd->getAddr();
                        golemTraceLen = nd->getLength();
                        golemTraceKind = static_cast<uint8_t>(nd->getDmaRequestKind());
                    }
                }
                const int vn = static_cast<int>(head->vn);
                auto golemDmaEnqueue = golem_dma_read_response_enqueue_ticks_.find(head);
                if (linkcontrol->spaceToSend(vn, head->size_in_bits) && linkcontrol->send(head, vn)) {
                    if (selectedOffset > 0) golem_dma_response_priority_reorders_++;
                    if (golemDmaEnqueue != golem_dma_read_response_enqueue_ticks_.end()) {
                        const uint64_t waitTicks =
                            getCurrentSimCycle() - golemDmaEnqueue->second;
                        golem_dma_read_response_queue_wait_ticks_ += waitTicks;
                        golem_dma_read_response_queue_wait_max_ticks_ = std::max(
                            golem_dma_read_response_queue_wait_max_ticks_, waitTicks);
                        golem_dma_read_response_drained_++;
                        golem_dma_read_response_enqueue_ticks_.erase(golemDmaEnqueue);
                    }
                    if (golem_dma_trace && golemTraceReq != 0) {
                        fprintf(stderr, "[memNICBase bridge] TRACE_REQ_RESP_CHUNK_SEND cycle=%" PRIu64
                                        " req=%" PRIu64 " trace_id=%d addr=0x%" PRIx64
                                        " len=%zu kind=%u queued=1 vn=%u\n",
                                getCurrentSimCycle(), golemTraceReq, golemTraceId, golemTraceAddr,
                                golemTraceLen, static_cast<unsigned>(golemTraceKind),
                                static_cast<unsigned>(head->vn));
                    }

#ifdef __SST_DEBUG_OUTPUT__
                    if (!debugEvStr.empty() && doDebug) {
                        dbg.debug(_L4_, "E: %-20" PRIu64 " %-20" PRIu64 " %-20s Event:Send    (%s), Dst: %" PRIu64 "\n",
                                getCurrentSimCycle(), uint64_t{0}, getName().c_str(), debugEvStr.c_str(), dst);
                    }
#endif
                    for (size_t offset = 0; offset < selectedOffset; ++offset) {
                        queue->push(queue->front());
                        queue->pop();
                    }
                    queue->pop();
                    for (size_t offset = 0; offset < selectedQueueSize - selectedOffset - 1; ++offset) {
                        queue->push(queue->front());
                        queue->pop();
                    }
                    sends++;
                } else {
                    break;
                }
            }
        }

        void drainGolemDmaResponseQueue(SST::Interfaces::SimpleNetwork* linkcontrol, size_t maxSends = 0) {
            size_t sends = 0;
            while (!golem_dma_send_queue_.empty() && (maxSends == 0 || sends < maxSends)) {
                size_t chosen = golem_dma_send_queue_.size();
                const uint64_t currentCycle = golemDmaResponseNowCycle();
                bool starvationFallback = false;

                auto eligible = [&](const GolemDmaResponseRequest& item) {
                    return currentCycle >= item.notBeforeCycle &&
                        (golem_dma_response_tile_priority_enable_ == 0 ||
                         currentCycle >= item.readyCycle +
                            golem_dma_response_reorder_cycles_);
                };

                if (golem_dma_response_tile_priority_enable_ != 0 &&
                    golem_dma_response_max_starvation_cycles_ != 0) {
                    for (size_t idx = 0; idx < golem_dma_send_queue_.size(); ++idx) {
                        const auto& item = golem_dma_send_queue_[idx];
                        if (currentCycle < item.readyCycle +
                                golem_dma_response_max_starvation_cycles_) {
                            continue;
                        }
                        if (chosen == golem_dma_send_queue_.size() ||
                            std::tie(item.readyCycle, item.requestId) <
                                std::tie(golem_dma_send_queue_[chosen].readyCycle,
                                         golem_dma_send_queue_[chosen].requestId)) {
                            chosen = idx;
                        }
                    }
                    starvationFallback = chosen != golem_dma_send_queue_.size();
                }

                if (!starvationFallback && golem_dma_response_tile_priority_enable_ != 0 &&
                    golem_dma_response_bundle_active_) {
                    for (size_t idx = 0; idx < golem_dma_send_queue_.size(); ++idx) {
                        const auto& item = golem_dma_send_queue_[idx];
                        if (eligible(item) &&
                            golemDmaBundleKey(item.requestId, item.dmaConsumer) ==
                                golem_dma_response_bundle_key_) {
                            chosen = idx;
                            break;
                        }
                    }
                    if (chosen == golem_dma_send_queue_.size()) {
                        golem_dma_response_bundle_active_ = false;
                        golem_dma_response_bundle_sent_ = 0;
                    }
                }

                if (chosen == golem_dma_send_queue_.size()) {
                    for (size_t idx = 0; idx < golem_dma_send_queue_.size(); ++idx) {
                        const auto& item = golem_dma_send_queue_[idx];
                        if (!eligible(item)) {
                            continue;
                        }
                        if (golem_dma_response_tile_priority_enable_ == 0) {
                            if (chosen == golem_dma_send_queue_.size() ||
                                (golem_dma_response_priority_enable &&
                                 golemDmaResponsePriority(item.request) <
                                     golemDmaResponsePriority(
                                         golem_dma_send_queue_[chosen].request))) {
                                chosen = idx;
                            }
                            if (!golem_dma_response_priority_enable) {
                                break;
                            }
                            continue;
                        }
                        if (chosen == golem_dma_send_queue_.size()) {
                            chosen = idx;
                            continue;
                        }
                        const auto& incumbent = golem_dma_send_queue_[chosen];
                        const auto itemKey = golemDmaBundleKey(
                            item.requestId, item.dmaConsumer);
                        const auto incumbentKey = golemDmaBundleKey(
                            incumbent.requestId, incumbent.dmaConsumer);
                        const uint32_t itemPending = golem_dma_response_pending_by_bundle_[itemKey];
                        const uint32_t incumbentPending = golem_dma_response_pending_by_bundle_[incumbentKey];
                        const uint16_t itemDistance = static_cast<uint8_t>(
                            (item.dmaConsumer.valid != 0 ? item.dmaConsumer.worker :
                             golemDmaWorkerId(item.requestId)) -
                            golem_dma_response_worker_cursor_);
                        const uint16_t incumbentDistance = static_cast<uint8_t>(
                            (incumbent.dmaConsumer.valid != 0 ? incumbent.dmaConsumer.worker :
                             golemDmaWorkerId(incumbent.requestId)) -
                            golem_dma_response_worker_cursor_);
                        if (std::make_tuple(golemDmaConsumerDistance(item.dmaConsumer),
                                            itemPending,
                                            itemDistance, golemDmaRequestSlot(item.requestId),
                                            item.returnAddr, item.readyCycle)
                            < std::make_tuple(golemDmaConsumerDistance(incumbent.dmaConsumer),
                                              incumbentPending,
                                              incumbentDistance, golemDmaRequestSlot(incumbent.requestId),
                                              incumbent.returnAddr, incumbent.readyCycle)) {
                            chosen = idx;
                        }
                    }
                    if (chosen == golem_dma_send_queue_.size()) {
                        break;
                    }
                    if (golem_dma_response_tile_priority_enable_ != 0) {
                        golem_dma_response_bundle_active_ = true;
                        golem_dma_response_bundle_key_ = golemDmaBundleKey(
                            golem_dma_send_queue_[chosen].requestId,
                            golem_dma_send_queue_[chosen].dmaConsumer);
                        golem_dma_response_bundle_sent_ = 0;
                        golem_dma_response_bundle_turns_++;
                    }
                }

                auto& item = golem_dma_send_queue_[chosen];
                auto* head = item.request;
                const uint64_t requestId = item.requestId;
                const uint64_t returnAddr = item.returnAddr;
                const size_t responseLength = item.length;
                const uint64_t readyCycle = item.readyCycle;
                const auto dmaRequestKind = item.dmaRequestKind;
                const int traceId = head->getTraceID();
                const unsigned responseVn = static_cast<unsigned>(head->vn);
                const uint8_t sentWorker = golemDmaWorkerId(requestId);
                const uint32_t metadataWorker = item.dmaConsumer.worker;
                const bool hasConsumerMetadata = item.dmaConsumer.valid != 0;
                const uint8_t consumerDistance = golemDmaConsumerDistance(item.dmaConsumer);
                const uint32_t responseCreditUnits = item.creditUnits;
                const int vn = static_cast<int>(head->vn);
                auto enqueueIt = golem_dma_read_response_enqueue_ticks_.find(head);
                if (!linkcontrol->spaceToSend(vn, head->size_in_bits) || !linkcontrol->send(head, vn)) {
                    break;
                }
                if (chosen != 0) {
                    golem_dma_response_reorders_++;
                    if (golem_dma_response_priority_enable &&
                        golem_dma_response_tile_priority_enable_ == 0) {
                        golem_dma_response_priority_reorders_++;
                    }
                }
                const uint64_t holdCycles = currentCycle - readyCycle;
                golem_dma_response_hold_cycles_sum_ += holdCycles;
                golem_dma_response_hold_cycles_max_ = std::max(
                    golem_dma_response_hold_cycles_max_, holdCycles);
                if (enqueueIt != golem_dma_read_response_enqueue_ticks_.end()) {
                    golem_dma_response_distance_wait_cycles_[consumerDistance] +=
                        readyCycle >= item.ingressCycle ? readyCycle - item.ingressCycle : 0;
                    golem_dma_response_distance_queue_wait_cycles_[consumerDistance] += holdCycles;
                    golem_dma_response_distance_responses_[consumerDistance]++;
                    const uint64_t waitTicks = getCurrentSimCycle() - enqueueIt->second;
                    golem_dma_read_response_queue_wait_ticks_ += waitTicks;
                    golem_dma_read_response_queue_wait_max_ticks_ = std::max(
                        golem_dma_read_response_queue_wait_max_ticks_, waitTicks);
                    golem_dma_read_response_drained_++;
                    golem_dma_read_response_enqueue_ticks_.erase(enqueueIt);
                }
                if (golem_dma_trace && requestId != 0) {
                    fprintf(stderr, "[memNICBase bridge] TRACE_REQ_RESP_CHUNK_SEND cycle=%" PRIu64
                                    " req=%" PRIu64 " trace_id=%d addr=0x%" PRIx64
                                    " len=%zu kind=%u queued=1 vn=%u\n",
                            currentCycle, requestId, traceId, returnAddr, responseLength,
                            static_cast<unsigned>(dmaRequestKind), responseVn);
                }
                golem_dma_send_queue_.erase(
                    golem_dma_send_queue_.begin() + static_cast<std::ptrdiff_t>(chosen));
                golem_dma_response_sent_++;
                releaseGolemDmaCredits(responseCreditUnits, requestId);
                if (starvationFallback) golem_dma_tile_starvation_++;
                sends++;

                if (golem_dma_response_tile_priority_enable_ != 0) {
                    golem_dma_response_bundle_sent_++;
                    if (golem_dma_response_bundle_sent_ >= golem_dma_tile_chunk_quantum_) {
                        golem_dma_response_worker_cursor_ = static_cast<uint8_t>(
                            (hasConsumerMetadata ? metadataWorker : sentWorker) + 1u);
                        golem_dma_response_bundle_active_ = false;
                        golem_dma_response_bundle_sent_ = 0;
                    }
                }
            }
        }

        MemRtrEvent* doRecv(SST::Interfaces::SimpleNetwork* linkcontrol, int preferredVn = -1) {
            drainGolemDmaResponseQueue(linkcontrol, golem_dma_response_drain_limit);
            SST::Interfaces::SimpleNetwork::Request* req = nullptr;
            const uint32_t startVn = preferredVn >= 0 &&
                                             static_cast<uint32_t>(preferredVn) < golem_network_num_vns_
                                         ? static_cast<uint32_t>(preferredVn)
                                         : golem_recv_vn_cursor_;
            for (uint32_t offset = 0; offset < golem_network_num_vns_; ++offset) {
                const uint32_t vn = (startVn + offset) % golem_network_num_vns_;
                req = linkcontrol->recv(static_cast<int>(vn));
                if (req != nullptr) {
                    golem_recv_vn_cursor_ = (vn + 1) % golem_network_num_vns_;
                    break;
                }
            }
            if (req != nullptr) {
                Event* payload = req->takePayload();

                // Check if this is a NetworkDataEvent (from Golem GlobalMemory)
                // These events use SimpleNetwork directly but can be bridged into MemHierarchy
                if (auto* nd = dynamic_cast<SST::Golem::NetworkDataEvent*>(payload)) {
                    const uint64_t addr = nd->getAddr();
                    const uint32_t size = static_cast<uint32_t>(nd->getLength());

                    std::string srcName = lookupNetworkName(req->src);
                    if (srcName.empty()) {
                        srcName = std::string("golem_ep_") + std::to_string(req->src);
                        networkAddressMap[srcName] = req->src;
                    }
                    reachableNames.insert(srcName);

                    MemEvent* me = nullptr;
                    if (nd->getType() == SST::Golem::NetworkDataEvent::READ) {
                        if (golem_dma_trace) {
                        fprintf(stderr, "[memNICBase bridge] recv READ cycle=%" PRIu64
                                        " src_ep=%" PRIu64 " srcName=%s addr=0x%" PRIx64
                                        " size=%u return_ep=%d return_addr=0x%" PRIx64 " req=%" PRIu64 "\n",
                                getCurrentSimCycle(), req->src, srcName.c_str(), addr, size,
                                nd->getReturnEndpoint(), nd->getReturnAddr(), nd->getRequestId());
                        }
                        GolemDmaIngressRequest ingress;
                        ingress.arrivalCycle = getCurrentSimCycle();
                        ingress.sourceEndpoint = req->src;
                        ingress.sourceName = srcName;
                        ingress.addr = addr;
                        ingress.size = size;
                        ingress.returnAddr = nd->getReturnAddr();
                        ingress.returnEndpoint = nd->getReturnEndpoint();
                        ingress.completionFlagAddr = nd->getCompletionFlagAddr();
                        ingress.completionValue = nd->getCompletionValue();
                        ingress.requestId = nd->getRequestId();
                        ingress.dmaRequestKind = nd->getDmaRequestKind();
                        ingress.dmaConsumer = nd->getDmaConsumerMetadata();
                        updateGolemDmaConsumerProgress(ingress.dmaConsumer);
                        if (golem_dma_credit_cap_ != 0 ||
                            golem_dma_window_priority_enable_ != 0) {
                            ingress.creditUnits = golem_dma_credit_cap_ != 0
                                ? golemDmaCreditUnits(size) : 0;
                            if (ingress.creditUnits > golem_dma_credit_cap_) {
                                dbg.fatal(CALL_INFO, -1,
                                          "%s Golem DMA request req=%" PRIu64 " needs %u credits, cap is %u.\n",
                                          getName().c_str(), ingress.requestId, ingress.creditUnits,
                                          golem_dma_credit_cap_);
                            }
                            golem_dma_ingress_queue_.push_back(std::move(ingress));
                            golem_dma_credit_max_queue_ = std::max(
                                golem_dma_credit_max_queue_, golem_dma_ingress_queue_.size());
                        } else {
                            me = createGolemDmaRead(ingress);
                        }
                    } else if (nd->getType() ==
                               SST::Golem::NetworkDataEvent::DMA_CONSUMER_PROGRESS) {
                        updateGolemDmaConsumerProgress(nd->getDmaConsumerMetadata());
                        golem_dma_consumer_progress_messages_++;
                    } else if (nd->getType() == SST::Golem::NetworkDataEvent::DMA_WRITE) {
                        std::vector<uint8_t> data = nd->getData();
                        me = new MemEvent(srcName, addr, addr, Command::Write, data);
                        me->setFlag(MemEventBase::F_NONCACHEABLE);
                        GolemDmaBridgeInfo info;
                        info.returnAddr = 0;
                        info.size = size;
                        info.isWrite = true;
                        info.hostAddr = addr;
                        info.requestId = nd->getRequestId();
                        info.dmaRequestKind = nd->getDmaRequestKind();
                        golem_dma_pending.emplace(me->getID(), info);
                    } else {
                        dbg.debug(_L10_, "%s received NetworkDataEvent type=%d not bridged, dropping.\n",
                                  getName().c_str(), static_cast<int>(nd->getType()));
                    }

                    delete nd;
                    delete req;

                    if (me) {
                        return new MemRtrEvent(me);
                    }
                    return nullptr;
                }

                // Continue with normal MemHierarchy processing
                MemRtrEvent * mre = static_cast<MemRtrEvent*>(payload);
                delete req;

                if (mre->hasClientData()) {
                    return mre;
                } else {
                    InitMemRtrEvent * imre = static_cast<InitMemRtrEvent*>(mre);
                    if (networkAddressMap.find(imre->info.name) == networkAddressMap.end()) {
                        dbg.fatal(CALL_INFO, -1, "%s received information about previously unknown endpoint. This case is not handled. Endpoint name: %s\n",
                                getName().c_str(), imre->info.name.c_str());
                    }
                    if (sourceIDs.find(imre->info.id) != sourceIDs.end()) {
                        addSource(imre->info);
                    }
                    if (destIDs.find(imre->info.id) != destIDs.end()) {
                        addDest(imre->info);
                    }
                    if (imre->info.id == info.id) {
                        addPeer(imre->info);
                    }
                    delete imre;
                }
            }
            return nullptr;
        }

        std::map<SST::Event::id_type, GolemDmaBridgeInfo> golem_dma_pending;
        std::map<GolemDmaKvCacheKey, GolemDmaKvCacheEntry> golem_dma_kv_cache_;
        std::deque<GolemDmaResponseRequest> golem_dma_send_queue_;
        std::unordered_map<SST::Interfaces::SimpleNetwork::Request*, uint64_t>
            golem_dma_read_response_enqueue_ticks_;
        uint64_t golem_dma_read_response_attempted_ = 0;
        uint64_t golem_dma_read_response_immediate_ = 0;
        uint64_t golem_dma_read_response_enqueued_ = 0;
        uint64_t golem_dma_read_response_drained_ = 0;
        uint64_t golem_dma_read_response_queue_high_water_ = 0;
        uint64_t golem_dma_read_response_queue_wait_ticks_ = 0;
        uint64_t golem_dma_read_response_queue_wait_max_ticks_ = 0;
        uint64_t golem_dma_response_priority_reorders_ = 0;
        uint64_t golem_dma_kv_multicast_next_cycle_ = 0;
        uint64_t golem_dma_kv_physical_reads_ = 0;
        uint64_t golem_dma_kv_coalesced_requests_ = 0;
        uint64_t golem_dma_kv_cache_hits_ = 0;
        uint64_t golem_dma_kv_multicast_receivers_ = 0;
        uint64_t golem_dma_kv_multicast_bytes_ = 0;

        std::string lookupNetworkName(uint64_t addr) const {
            for (const auto& entry : networkAddressMap) {
                if (entry.second == addr) {
                    return entry.first;
                }
            }
            return "";
        }

        bool trySendGolemDmaResponse(MemEventBase* ev, SST::Interfaces::SimpleNetwork* linkcontrol) {
            auto it = golem_dma_pending.find(ev->getResponseToID());
            if (it == golem_dma_pending.end()) return false;

            const GolemDmaBridgeInfo info = it->second;
            if (!info.isWrite) {
                const auto decrementBundle = [this](
                        const GolemDmaKvSubscriber& subscriber) {
                    if (subscriber.requestId == 0) return;
                    const auto bundleKey = golemDmaBundleKey(
                        subscriber.requestId, subscriber.dmaConsumer);
                    auto pendingIt =
                        golem_dma_response_pending_by_bundle_.find(bundleKey);
                    if (pendingIt != golem_dma_response_pending_by_bundle_.end() &&
                        pendingIt->second != 0) {
                        pendingIt->second--;
                    }
                };
                if (info.kvSubscribers.empty()) {
                    decrementBundle(info);
                } else {
                    for (const auto& subscriber : info.kvSubscribers) {
                        decrementBundle(subscriber);
                    }
                }
            }

            const uint64_t dest = info.returnEndpoint >= 0 ? static_cast<uint64_t>(info.returnEndpoint) : lookupNetworkAddress(ev->getDst());
            const uint64_t responseReadyCycle = golemDmaResponseNowCycle();
            uint64_t multicastReadyCycle = responseReadyCycle;
            if (!info.kvSubscribers.empty()) {
                const uint64_t startCycle = std::max(
                    responseReadyCycle, golem_dma_kv_multicast_next_cycle_);
                const uint64_t transferCycles =
                    (static_cast<uint64_t>(info.size) +
                     golem_dma_kv_multicast_bytes_per_cycle_ - 1) /
                    golem_dma_kv_multicast_bytes_per_cycle_;
                multicastReadyCycle = startCycle + transferCycles;
                golem_dma_kv_multicast_next_cycle_ = multicastReadyCycle;
            }

            auto sendOrQueue = [&](SST::Interfaces::SimpleNetwork::Request* req) -> bool {
                const int vn = static_cast<int>(req->vn);
                if (!info.isWrite) golem_dma_read_response_attempted_++;
                if (multicastReadyCycle <= responseReadyCycle &&
                    golem_dma_send_queue_.empty() &&
                    linkcontrol->spaceToSend(vn, req->size_in_bits) &&
                    linkcontrol->send(req, vn)) {
                    if (!info.isWrite) golem_dma_read_response_immediate_++;
                    if (!info.isWrite) {
                        const uint64_t readyCycle = golemDmaResponseNowCycle();
                        const uint8_t distance = golemDmaConsumerDistance(info.dmaConsumer);
                        golem_dma_response_distance_wait_cycles_[distance] +=
                            readyCycle >= info.ingressCycle ? readyCycle - info.ingressCycle : 0;
                        golem_dma_response_distance_responses_[distance]++;
                        if (info.dmaConsumer.valid != 0 && distance == 0) {
                            golem_dma_response_late_ready_++;
                        }
                    }
                    releaseGolemDmaCredits(info.creditUnits, info.requestId);
                    if (golem_dma_trace && info.requestId != 0) {
                        fprintf(stderr, "[memNICBase bridge] TRACE_REQ_RESP_CHUNK_SEND cycle=%" PRIu64
                                        " req=%" PRIu64 " trace_id=%d addr=0x%" PRIx64
                                        " len=%u queued=0 vn=%u\n",
                                getCurrentSimCycle(), info.requestId, req->getTraceID(),
                                info.returnAddr, info.size, static_cast<unsigned>(req->vn));
                    }
                    return false;
                } else {
                    dbg.debug(_L2_, "%s failed to send bridged DMA response (buffer full), queueing retry.\n", getName().c_str());
                    auto* response = dynamic_cast<SST::Golem::NetworkDataEvent*>(req->inspectPayload());
                    GolemDmaResponseRequest queued;
                    queued.request = req;
                    queued.readyCycle = responseReadyCycle;
                    queued.notBeforeCycle = multicastReadyCycle;
                    queued.requestId = response != nullptr ? response->getRequestId() : 0;
                    queued.returnAddr = response != nullptr ? response->getAddr() : 0;
                    queued.length = response != nullptr ? response->getLength() : 0;
                    queued.ingressCycle = info.ingressCycle;
                    queued.creditUnits = info.creditUnits;
                    queued.dmaRequestKind = response != nullptr
                        ? response->getDmaRequestKind()
                        : SST::Golem::DmaRequestKind::Unknown;
                    queued.dmaConsumer = response != nullptr
                        ? response->getDmaConsumerMetadata()
                        : SST::Golem::DmaConsumerMetadata();
                    if (queued.dmaConsumer.valid != 0 &&
                        golemDmaConsumerDistance(queued.dmaConsumer) == 0) {
                        golem_dma_response_late_ready_++;
                    }
                    golem_dma_send_queue_.push_back(queued);
                    golem_dma_response_max_queue_ = std::max(
                        golem_dma_response_max_queue_, golem_dma_send_queue_.size());
                    if (!info.isWrite) {
                        golem_dma_read_response_enqueued_++;
                        golem_dma_read_response_enqueue_ticks_[req] =
                            getCurrentSimCycle();
                        golem_dma_read_response_queue_high_water_ = std::max(
                            golem_dma_read_response_queue_high_water_,
                            static_cast<uint64_t>(
                                golem_dma_read_response_enqueue_ticks_.size()));
                    }
                    return true;
                }
            };

            if (info.isWrite) {
                if (golem_dma_trace) {
                    fprintf(stderr, "[memNICBase bridge] send WRITE_COMPLETE cycle=%" PRIu64
                                    " addr=0x%" PRIx64 " len=%zu req=%" PRIu64 " kind=%u\n",
                            getCurrentSimCycle(), info.hostAddr, static_cast<size_t>(info.size), info.requestId,
                            static_cast<unsigned>(info.dmaRequestKind));
                }
                auto* req = new SST::Interfaces::SimpleNetwork::Request();
                req->src = this->info.addr;
                req->dest = dest;
                req->vn = golem_dma_write_response_vn;
                auto* respEv = new SST::Golem::NetworkDataEvent(
                    SST::Golem::NetworkDataEvent::DMA_WRITE_COMPLETE,
                    info.hostAddr,
                    info.size,
                    std::vector<uint8_t>(),
                    0, -1, 0, 0, info.requestId, info.dmaRequestKind);
                req->size_in_bits = (sizeof(info.hostAddr) + sizeof(size_t)) * 8;
                req->givePayload(respEv);
                sendOrQueue(req);
            } else {
                auto* mev = dynamic_cast<MemEvent*>(ev);
                std::vector<uint8_t> data;
                if (mev) data = mev->getPayload();
                if (!info.kvSubscribers.empty()) {
                    const auto cacheKey = golemDmaKvCacheKey(
                        info.hostAddr, info.size, info.dmaConsumer);
                    GolemDmaKvCacheEntry cacheEntry;
                    cacheEntry.data = data;
                    for (const auto& subscriber : info.kvSubscribers) {
                        const uint64_t consumerId =
                            golemDmaKvConsumerId(subscriber.dmaConsumer);
                        if (std::find(cacheEntry.servedConsumers.begin(),
                                      cacheEntry.servedConsumers.end(), consumerId) ==
                            cacheEntry.servedConsumers.end()) {
                            cacheEntry.servedConsumers.push_back(consumerId);
                        }
                    }
                    if (cacheEntry.servedConsumers.size() <
                        golem_dma_kv_expected_consumers_) {
                        golem_dma_kv_cache_[cacheKey] = std::move(cacheEntry);
                    }
                }
                if (golem_dma_trace) {
                    fprintf(stderr, "[memNICBase bridge] send READ_RESP cycle=%" PRIu64
                                    " dst_ep=%" PRIu64 " return_addr=0x%" PRIx64
                                    " size=%u data=%zu req=%" PRIu64 " trace_id=%d kind=%u"
                                    " flag=0x%" PRIx64 " val=%" PRIu64 "\n",
                            getCurrentSimCycle(), dest, info.returnAddr, info.size, data.size(),
                            info.requestId, makeGolemMerlinTraceId(info.requestId),
                            static_cast<unsigned>(info.dmaRequestKind),
                            info.completionFlagAddr, info.completionValue);
                }
                auto* req = new SST::Interfaces::SimpleNetwork::Request();
                req->src = this->info.addr;
                req->dest = dest;
                req->vn = golem_dma_response_vn;
                auto* respEv = new SST::Golem::NetworkDataEvent(
                    SST::Golem::NetworkDataEvent::DMA_READ_COMPLETE,
                    info.returnAddr,
                    data.size(),
                    data,
                    info.returnAddr,
                    info.returnEndpoint,
                    info.completionFlagAddr,
                    info.completionValue,
                    info.requestId, info.dmaRequestKind, info.dmaConsumer);
                req->size_in_bits = (sizeof(info.returnAddr) + sizeof(size_t) + data.size()) * 8;
                req->givePayload(respEv);
                if (golem_dma_trace && info.requestId != 0) {
                    req->setTraceID(makeGolemMerlinTraceId(info.requestId));
                    req->setTraceType(SST::Interfaces::SimpleNetwork::Request::FULL);
                }
                const bool queued = sendOrQueue(req);
                if (golem_dma_trace && queued) {
                    fprintf(stderr, "[memNICBase bridge] TRACE_REQ_RESP_CHUNK_ENQUEUE cycle=%" PRIu64
                                    " req=%" PRIu64 " trace_id=%d addr=0x%" PRIx64
                                    " len=%zu kind=%u queue=%zu vn=%u\n",
                            getCurrentSimCycle(), info.requestId, req->getTraceID(),
                            info.returnAddr, data.size(), static_cast<unsigned>(info.dmaRequestKind),
                            golem_dma_send_queue_.size(),
                            static_cast<unsigned>(req->vn));
                }
                if (!info.kvSubscribers.empty()) {
                    golem_dma_kv_multicast_receivers_ +=
                        info.kvSubscribers.size();
                    golem_dma_kv_multicast_bytes_ += data.size();
                    for (size_t index = 1;
                         index < info.kvSubscribers.size(); ++index) {
                        const auto& subscriber = info.kvSubscribers[index];
                        auto* branchReq =
                            new SST::Interfaces::SimpleNetwork::Request();
                        branchReq->src = this->info.addr;
                        branchReq->dest = subscriber.returnEndpoint >= 0
                            ? static_cast<uint64_t>(subscriber.returnEndpoint)
                            : lookupNetworkAddress(ev->getDst());
                        branchReq->vn = golem_dma_response_vn;
                        auto* branchEv = new SST::Golem::NetworkDataEvent(
                            SST::Golem::NetworkDataEvent::DMA_READ_COMPLETE,
                            subscriber.returnAddr, data.size(), data,
                            subscriber.returnAddr, subscriber.returnEndpoint,
                            subscriber.completionFlagAddr,
                            subscriber.completionValue,
                            subscriber.requestId, subscriber.dmaRequestKind,
                            subscriber.dmaConsumer);
                        // The data traverses the shared tree once. Branches add
                        // destination metadata but do not consume payload width.
                        branchReq->size_in_bits =
                            (sizeof(subscriber.returnAddr) + sizeof(size_t)) * 8;
                        branchReq->givePayload(branchEv);
                        if (golem_dma_trace && subscriber.requestId != 0) {
                            branchReq->setTraceID(
                                makeGolemMerlinTraceId(subscriber.requestId));
                            branchReq->setTraceType(
                                SST::Interfaces::SimpleNetwork::Request::FULL);
                        }
                        GolemDmaResponseRequest branch;
                        branch.request = branchReq;
                        branch.readyCycle = responseReadyCycle;
                        branch.notBeforeCycle = multicastReadyCycle;
                        branch.requestId = subscriber.requestId;
                        branch.returnAddr = subscriber.returnAddr;
                        branch.length = data.size();
                        branch.ingressCycle = subscriber.ingressCycle;
                        branch.creditUnits = subscriber.creditUnits;
                        branch.dmaRequestKind = subscriber.dmaRequestKind;
                        branch.dmaConsumer = subscriber.dmaConsumer;
                        golem_dma_send_queue_.push_back(std::move(branch));
                        golem_dma_read_response_attempted_++;
                        golem_dma_read_response_enqueued_++;
                        golem_dma_read_response_enqueue_ticks_[branchReq] =
                            getCurrentSimCycle();
                    }
                    golem_dma_response_max_queue_ = std::max(
                        golem_dma_response_max_queue_,
                        golem_dma_send_queue_.size());
                    golem_dma_read_response_queue_high_water_ = std::max(
                        golem_dma_read_response_queue_high_water_,
                        static_cast<uint64_t>(
                            golem_dma_read_response_enqueue_ticks_.size()));
                }
            }

            golem_dma_pending.erase(it);
            return true;
        }

        void finishGolemDmaResponseStats() {
            if (golem_dma_read_response_attempted_ == 0) return;
            std::printf(
                "GOLEM_MEMNIC_DMA_RESPONSE_CONSERVATION component=%s"
                " attempted=%" PRIu64 " immediate=%" PRIu64
                " enqueued=%" PRIu64 " drained=%" PRIu64 " pending=%zu"
                " responses_d0=%" PRIu64 " responses_d1=%" PRIu64
                " responses_d2=%" PRIu64 " responses_far=%" PRIu64 "\n",
                getName().c_str(), golem_dma_read_response_attempted_,
                golem_dma_read_response_immediate_,
                golem_dma_read_response_enqueued_,
                golem_dma_read_response_drained_,
                golem_dma_read_response_enqueue_ticks_.size(),
                golem_dma_response_distance_responses_[0],
                golem_dma_response_distance_responses_[1],
                golem_dma_response_distance_responses_[2],
                golem_dma_response_distance_responses_[3]);
            std::printf(
                "GOLEM_MEMNIC_DMA_RESPONSE_STATS component=%s attempted=%" PRIu64
                " immediate=%" PRIu64 " enqueued=%" PRIu64
                " drained=%" PRIu64 " high_water=%" PRIu64
                " queue_wait_ticks=%" PRIu64 " queue_wait_max_ticks=%" PRIu64
                " priority_reorders=%" PRIu64 " pending=%zu"
                " response_wait_d0=%" PRIu64 " response_wait_d1=%" PRIu64
                " response_wait_d2=%" PRIu64 " response_wait_far=%" PRIu64
                " queue_wait_d0=%" PRIu64 " queue_wait_d1=%" PRIu64
                " queue_wait_d2=%" PRIu64 " queue_wait_far=%" PRIu64
                " responses_d0=%" PRIu64 " responses_d1=%" PRIu64
                " responses_d2=%" PRIu64 " responses_far=%" PRIu64
                " admission_wait_d0=%" PRIu64 " admission_wait_d1=%" PRIu64
                " admission_wait_d2=%" PRIu64 " admission_wait_far=%" PRIu64
                " credit_blocked=%" PRIu64 " credit_blocked_cycles=%" PRIu64
                " late_ready=%" PRIu64 " tile_starvation=%" PRIu64
                " consumer_progress=%" PRIu64 "\n",
                getName().c_str(), golem_dma_read_response_attempted_,
                golem_dma_read_response_immediate_,
                golem_dma_read_response_enqueued_,
                golem_dma_read_response_drained_,
                golem_dma_read_response_queue_high_water_,
                golem_dma_read_response_queue_wait_ticks_,
                golem_dma_read_response_queue_wait_max_ticks_,
                golem_dma_response_priority_reorders_,
                golem_dma_read_response_enqueue_ticks_.size(),
                golem_dma_response_distance_wait_cycles_[0],
                golem_dma_response_distance_wait_cycles_[1],
                golem_dma_response_distance_wait_cycles_[2],
                golem_dma_response_distance_wait_cycles_[3],
                golem_dma_response_distance_queue_wait_cycles_[0],
                golem_dma_response_distance_queue_wait_cycles_[1],
                golem_dma_response_distance_queue_wait_cycles_[2],
                golem_dma_response_distance_queue_wait_cycles_[3],
                golem_dma_response_distance_responses_[0],
                golem_dma_response_distance_responses_[1],
                golem_dma_response_distance_responses_[2],
                golem_dma_response_distance_responses_[3],
                golem_dma_consumer_distance_wait_cycles_[0],
                golem_dma_consumer_distance_wait_cycles_[1],
                golem_dma_consumer_distance_wait_cycles_[2],
                golem_dma_consumer_distance_wait_cycles_[3],
                golem_dma_credit_blocked_requests_,
                golem_dma_credit_blocked_cycles_,
                golem_dma_response_late_ready_, golem_dma_tile_starvation_,
                golem_dma_consumer_progress_messages_);
            if (golem_dma_kv_coalesce_enable_ != 0) {
                std::printf(
                    "GOLEM_MEMNIC_DMA_KV_MULTICAST_STATS component=%s"
                    " physical_reads=%" PRIu64
                    " coalesced_requests=%" PRIu64
                    " cache_hits=%" PRIu64
                    " receivers=%" PRIu64
                    " multicast_bytes=%" PRIu64
                    " resident_chunks=%zu\n",
                    getName().c_str(), golem_dma_kv_physical_reads_,
                    golem_dma_kv_coalesced_requests_,
                    golem_dma_kv_cache_hits_,
                    golem_dma_kv_multicast_receivers_,
                    golem_dma_kv_multicast_bytes_, golem_dma_kv_cache_.size());
            }
        }

        /*** Data Members ***/
        bool initMsgSent;

        // Data structures
        std::unordered_map<std::string,uint64_t> networkAddressMap; // Map of name -> address for each network endpoint
        std::set<EndpointInfo> sourceEndpointInfo;
        std::set<EndpointInfo> destEndpointInfo;
        std::set<EndpointInfo> peerEndpointInfo;
        std::set<EndpointInfo> endpointInfo;
        std::set<std::string> reachableNames;

        // Untimed and init event queues
        std::queue<MemRtrEvent*> untimed_receive_queue_; // Queue for received untimed events
        std::queue<SST::Interfaces::SimpleNetwork::Request*> untimed_send_queue_; // Queue of events waiting to be sent after network (linkcontrol) initializes
        std::set<MemEventInit*> initWaitForDst; // Set of events with unknown destinations - only possible during init() stage

        // Other parameters
        std::unordered_set<uint32_t> sourceIDs, destIDs; // IDs which this endpoint cares about
        uint32_t range_check = true; // Enable overlapping range check
        uint32_t golem_network_num_vns_ = 1;
        uint32_t golem_recv_vn_cursor_ = 0;
        uint32_t golem_dma_response_vn = 0;
        uint32_t golem_dma_write_response_vn = 0;
        uint32_t golem_dma_trace = 0;
        uint32_t golem_dma_response_priority_enable = 0;
        uint32_t golem_dma_kv_coalesce_enable_ = 0;
        uint32_t golem_dma_kv_multicast_bytes_per_cycle_ = 256;
        uint32_t golem_dma_kv_expected_consumers_ = 16;
        size_t golem_dma_response_drain_limit = 0;
        uint32_t golem_dma_credit_cap_ = 0;
        uint32_t golem_dma_credit_available_ = 0;
        uint32_t golem_dma_credit_chunk_bytes_ = 16384;
        size_t golem_dma_admission_limit_ = 0;
        uint32_t golem_dma_credit_max_used_ = 0;
        size_t golem_dma_credit_max_queue_ = 0;
        uint64_t golem_dma_credit_admitted_requests_ = 0;
        uint64_t golem_dma_credit_released_requests_ = 0;
        uint64_t golem_dma_credit_blocked_requests_ = 0;
        uint64_t golem_dma_credit_blocked_cycles_ = 0;
        uint32_t golem_dma_window_priority_enable_ = 0;
        uint32_t golem_dma_window_reorder_cycles_ = 512;
        uint32_t golem_dma_tile_chunk_quantum_ = 1;
        uint8_t golem_dma_worker_cursor_ = 0;
        uint64_t golem_dma_window_priority_reorders_ = 0;
        bool golem_dma_bundle_active_ = false;
        uint32_t golem_dma_bundle_window_ = 0;
        uint8_t golem_dma_bundle_worker_ = 0;
        uint8_t golem_dma_bundle_tile_ = 0;
        uint32_t golem_dma_bundle_admitted_ = 0;
        uint64_t golem_dma_bundle_turns_ = 0;
        uint64_t golem_dma_bundle_admissions_ = 0;
        GolemDmaBundleKey golem_dma_bundle_key_;
        uint32_t golem_dma_response_tile_priority_enable_ = 0;
        uint32_t golem_dma_response_reorder_cycles_ = 0;
        uint32_t golem_dma_response_max_starvation_cycles_ = 65536;
        uint32_t golem_dma_admission_max_starvation_cycles_ = 4096;
        SimTime_t golem_dma_clock_factor_ = 1;
        uint8_t golem_dma_response_worker_cursor_ = 0;
        bool golem_dma_response_bundle_active_ = false;
        GolemDmaBundleKey golem_dma_response_bundle_key_;
        uint32_t golem_dma_response_bundle_sent_ = 0;
        size_t golem_dma_response_max_queue_ = 0;
        uint64_t golem_dma_response_reorders_ = 0;
        uint64_t golem_dma_response_bundle_turns_ = 0;
        uint64_t golem_dma_response_sent_ = 0;
        uint64_t golem_dma_response_hold_cycles_sum_ = 0;
        uint64_t golem_dma_response_hold_cycles_max_ = 0;
        std::array<uint64_t, 4> golem_dma_response_distance_wait_cycles_{};
        std::array<uint64_t, 4> golem_dma_response_distance_queue_wait_cycles_{};
        std::array<uint64_t, 4> golem_dma_response_distance_responses_{};
        std::array<uint64_t, 4> golem_dma_consumer_distance_wait_cycles_{};
        std::array<uint64_t, 4> golem_dma_consumer_distance_admissions_{};
        uint64_t golem_dma_response_late_ready_ = 0;
        uint64_t golem_dma_tile_starvation_ = 0;
        uint64_t golem_dma_consumer_progress_messages_ = 0;
        std::map<std::pair<uint64_t, uint32_t>, GolemDmaConsumerProgress>
            golem_dma_consumer_progress_;
        std::unordered_map<GolemDmaBundleKey, uint32_t, GolemDmaBundleKeyHash>
            golem_dma_response_pending_by_bundle_;
        std::vector<GolemDmaIngressRequest> golem_dma_ingress_queue_;

    private:
        static uint32_t envFlagDefault(const char* name, uint32_t defaultValue) {
            const char* raw = std::getenv(name);
            if (raw == nullptr) {
                return defaultValue;
            }
            std::string value(raw);
            for (char& ch : value) {
                ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            }
            return (value == "1" || value == "true" || value == "yes" || value == "on") ? 1 : 0;
        }

        void build(Params& params) {
            // Get source/destination parameters
            // Each NIC has a group ID and talks to those with IDs in sources and destinations
            // If no source/destination provided, source = group ID - 1, destination = group ID + 1
            bool found;
            info.id = params.find<uint32_t>("group", 0, found);
            if (!found) {
                dbg.fatal(CALL_INFO, -1, "Param not specified(%s): group - group ID (or hierarchy level) for this NIC's component. Example: L2s in group 1, directories in group 2, memories (on network) in group 3.\n",
                        getName().c_str());
            }
            
            if (params.is_value_array("sources")) {
                std::vector<uint32_t> srcArr;
                params.find_array<uint32_t>("sources", srcArr);
                sourceIDs = std::unordered_set<uint32_t>(srcArr.begin(), srcArr.end());

            }
            if (params.is_value_array("destinations")) {
                std::vector<uint32_t> dstArr;
                params.find_array<uint32_t>("destinations", dstArr);
                destIDs = std::unordered_set<uint32_t>(dstArr.begin(), dstArr.end());
            }
            
            // range_check current is off(0) or on(1) but is using a uint32_t to 
            // allow for future selection of different algorithms.
            range_check=params.find<uint32_t>("range_check", 1);
            golem_network_num_vns_=std::max<uint32_t>(params.find<uint32_t>("num_vns", 1), 1u);
            bool golem_dma_response_vn_found = false;
            golem_dma_response_vn = params.find<uint32_t>(
                "golem_dma_response_vn", 0, golem_dma_response_vn_found);
            if (!golem_dma_response_vn_found) {
                golem_dma_response_vn = golem_network_num_vns_ >= 2 ? 1 : 0;
            }
            if (golem_dma_response_vn >= golem_network_num_vns_) {
                dbg.fatal(CALL_INFO, -1,
                        "%s invalid golem_dma_response_vn=%" PRIu32
                        "; value must be less than num_vns=%" PRIu32 ".\n",
                        getName().c_str(), golem_dma_response_vn, golem_network_num_vns_);
            }
            golem_dma_response_priority_enable=params.find<uint32_t>(
                "golem_dma_response_priority_enable",
                envFlagDefault("GOLEM_DMA_RESPONSE_PRIORITY_ENABLE", 0));
            golem_dma_kv_coalesce_enable_=params.find<uint32_t>(
                "golem_dma_kv_coalesce_enable", 0);
            golem_dma_kv_multicast_bytes_per_cycle_=params.find<uint32_t>(
                "golem_dma_kv_multicast_bytes_per_cycle", 256);
            golem_dma_kv_expected_consumers_=params.find<uint32_t>(
                "golem_dma_kv_expected_consumers", 16);
            if (golem_dma_kv_coalesce_enable_ != 0 &&
                (golem_dma_kv_multicast_bytes_per_cycle_ == 0 ||
                 golem_dma_kv_expected_consumers_ == 0)) {
                dbg.fatal(CALL_INFO, -1,
                          "%s, Error: shared K/V bandwidth and expected consumer count must be positive.\n",
                          getName().c_str());
            }
            golem_dma_write_response_vn=params.find<uint32_t>(
                "golem_dma_write_response_vn", golem_network_num_vns_ >= 3 ? 2u : 0u);
            if (golem_dma_write_response_vn >= golem_network_num_vns_) {
                dbg.fatal(CALL_INFO, -1,
                          "%s, Error: golem_dma_write_response_vn=%u must be smaller than num_vns=%u.\n",
                          getName().c_str(), golem_dma_write_response_vn, golem_network_num_vns_);
            }
            golem_dma_trace=params.find<uint32_t>("golem_dma_trace", envFlagDefault("GOLEM_DMA_TRACE", 0));
            golem_dma_response_drain_limit=params.find<size_t>("golem_dma_response_drain_limit", 0);
            golem_dma_credit_cap_=params.find<uint32_t>("golem_dma_credit_cap", 0);
            golem_dma_credit_available_=golem_dma_credit_cap_;
            golem_dma_credit_chunk_bytes_=params.find<uint32_t>("golem_dma_credit_chunk_bytes", 16384);
            golem_dma_admission_limit_=params.find<size_t>("golem_dma_admission_limit", 0);
            golem_dma_window_priority_enable_=params.find<uint32_t>("golem_dma_window_priority_enable", 0);
            golem_dma_window_reorder_cycles_=params.find<uint32_t>("golem_dma_window_reorder_cycles", 512);
            golem_dma_tile_chunk_quantum_=std::max<uint32_t>(
                params.find<uint32_t>("golem_dma_tile_chunk_quantum", 1), 1u);
            golem_dma_response_tile_priority_enable_=params.find<uint32_t>(
                "golem_dma_response_tile_priority_enable", 0);
            golem_dma_response_reorder_cycles_=params.find<uint32_t>(
                "golem_dma_response_reorder_cycles", 0);
            golem_dma_response_max_starvation_cycles_=params.find<uint32_t>(
                "golem_dma_response_max_starvation_cycles", 65536);
            golem_dma_admission_max_starvation_cycles_=params.find<uint32_t>(
                "golem_dma_admission_max_starvation_cycles", 4096);
            if (golem_dma_credit_cap_ != 0 && golem_dma_credit_chunk_bytes_ == 0) {
                dbg.fatal(CALL_INFO, -1,
                          "%s, Error: golem_dma_credit_chunk_bytes must be positive when credit ownership is enabled.\n",
                          getName().c_str());
            }
            if (golem_dma_trace) {
                fprintf(stderr,
                        "[memNICBase bridge] resolved golem_dma_response_vn=%" PRIu32
                        " num_vns=%" PRIu32 " explicit=%u\n",
                        golem_dma_response_vn, golem_network_num_vns_,
                        golem_dma_response_vn_found ? 1U : 0U);
            }

            std::stringstream sources, destinations;
            uint32_t id;

            if (sourceIDs.empty()) {
                sources.str(params.find<std::string>("sources", ""));
                while (sources >> id) {
                    sourceIDs.insert(id);
                    while (sources.peek() == ',' || sources.peek() == ' ')
                        sources.ignore();
                }
                if (sourceIDs.empty())
                    sourceIDs.insert(info.id - 1);
            }

            if (destIDs.empty()) {
                destinations.str(params.find<std::string>("destinations", ""));
                while (destinations >> id) {
                    destIDs.insert(id);
                    while (destinations.peek() == ',' || destinations.peek() == ' ')
                        destinations.ignore();
                }
                if (destIDs.empty())
                    destIDs.insert(info.id + 1);
            }
            initMsgSent = false;

            dbg.debug(_L10_, "%s memNICBase info is: Name: %s, group: %" PRIu32 "\n",
                    getName().c_str(), info.name.c_str(), info.id);
        }
};

} //namespace memHierarchy
} //namespace SST

#endif
