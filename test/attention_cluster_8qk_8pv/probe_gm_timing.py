#!/usr/bin/env python3
"""Execute the current GM scheduler functions in an isolated event harness.

Uses verbatim C++ method bodies, not a Python recreation of their timing.
The harness replaces SST's event delivery only; it is not an SST/RTL test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


HARNESS = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <deque>
#include <functional>
#include <memory>
#include <limits>
#include "localGmBankScheduler.h"
using SST::Golem::LocalGmBankScheduler;
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <vector>
enum class LocalMemoryClient { Dma, RoCC, SFU, WCP, Control, ProjectionOracleOperand };
uint64_t oracleServiceCycles(const char*,bool,uint64_t n) { return n; }
template<class... Args> void traceResourceInterval(Args...) {}
using LocalReadCallback = std::function<void(bool,uint64_t,const std::vector<uint8_t>&)>;
using LocalWriteCallback = std::function<void(bool,uint64_t)>;
struct Event { virtual ~Event() = default; };
struct LocalMemoryAccessEvent : Event {
    uint64_t id;
    explicit LocalMemoryAccessEvent(uint64_t n):id(n){}
    uint64_t requestId() const { return id; }
};
struct BusyIntervalUnion { bool add(uint64_t,uint64_t) { return true; } };
struct Converter { uint64_t factor=1; uint64_t convertToCoreTime(uint64_t n) { return n*factor; } uint64_t convertFromCoreTime(uint64_t n) { return n/factor; } };
struct Fence { uint64_t generation=1; uint64_t active() { return generation; } bool matches(uint64_t n) { return n==generation; } };
struct Output { template<class... Args> void fatal(Args...) { throw std::runtime_error("fatal"); } template<class... Args> void verbose(Args...) {} };
#define CALL_INFO 0
uint64_t ceilDivLocalAccess(uint64_t n,uint64_t d) { return (n+d-1)/d; }
using DmaCallback = std::function<void(bool)>;
enum class DmaRequestKind { Unknown };
struct DmaConsumerMetadata {};
struct GlobalMemoryImplement {
    /*PENDING_DMA_STRUCTS*/
    struct PendingLocalAccess {
        enum class Kind { Read, Write };
        Kind kind;
        uint64_t requestId=0,addr=0,tag=0,submitCycle=0,ownerGeneration=0,serviceStartTick=0,traceDepth=0;
        size_t length=0;
        LocalMemoryClient client=LocalMemoryClient::Control;
        bool issued=false,cancelled=false;
        LocalReadCallback readCallback;
        LocalWriteCallback writeCallback;
        std::vector<uint8_t> readData,writeData;
    };
    int core_id=0; uint64_t now=0,baseAddr=0,size=8192;
    std::vector<uint8_t> storage=std::vector<uint8_t>(size,7);
    uint32_t localAccessBaseLatencyCycles_=1,localAccessBytesPerCycle_=256;
    uint32_t localAccessReadPorts_=1,localAccessWritePorts_=1;
    uint32_t localAccessQueueDepth_=256,localAccessMaxRequestBytes_=4096;
    uint32_t localReadsInFlight_=0,localWritesInFlight_=0;
    uint64_t nextLocalAccessRequestId_=1,localQueueRejected_=0;
    uint64_t localReadRequests_=0,localWriteRequests_=0,localAccessQueueHighWater_=0;
    uint64_t localReadQueueCycles_=0,localWriteQueueCycles_=0,localReadBytes_=0,localWriteBytes_=0;
    bool attentionClusterEnable_=false,dmaLandingRetryScheduled_=false;
    Fence attentionGenerationFence_;
    BusyIntervalUnion attentionClusterLocalReadIntervals_,attentionClusterLocalWriteIntervals_;
    Converter converter,coreConverter; Converter* localAccessTC_=&converter; Converter* localCoreTC_=&coreConverter;
    std::unique_ptr<LocalGmBankScheduler> localBankScheduler_; bool localBankTickScheduled_=false;
    Output out; Output* output=&out;
    std::unordered_map<uint64_t,PendingLocalAccess> localAccessPending_;
    std::deque<uint64_t> localReadQueue_,localWriteQueue_;
    std::multimap<uint64_t,Event*> events;
    struct Link {
        GlobalMemoryImplement* gm;
        void send(uint64_t delay,Event* ev) {
            gm->events.emplace(gm->now+gm->localAccessTC_->convertToCoreTime(delay),ev);
        }
        void send(uint64_t delay,Converter tc,Event* ev) {
            gm->events.emplace(gm->now+tc.convertToCoreTime(delay),ev);
        }
    } link{this};
    Link* localAccessSelfLink_=&link;
    uint64_t getCurrentSimCycle() { return now; }
    uint64_t nextLocalDmaTag_=1,nextTraceDmaRequest_=1,controlCompletions=0;
    uint64_t dmaLandingBackpressureRetries_=0,dma_read_completion_count=0;
    uint64_t request_read_submit_ready_samples=0,request_read_submit_ready_cycles_sum=0,request_read_submit_ready_cycles_max=0;
    std::unordered_map<uint64_t,PendingReadRequest> request_pending;
    std::unordered_map<uint64_t,PendingDmaLanding> pendingDmaLandings_;
    std::unordered_map<uint64_t,bool> dma_completion_tokens;
    void write_u64_to_storage(uint64_t addr,uint64_t value) { std::memcpy(storage.data()+addr,&value,8); }
    uint64_t flag(uint64_t addr=0x1000) { uint64_t v; std::memcpy(&v,storage.data()+addr,8);return v; }
    void notifyCtrlReadComplete(uint64_t) { ++controlCompletions; }
    void ctrlRegisterPendingReadRequest(uint64_t,uint64_t,uint64_t,uint64_t,size_t);
    void landControlledReadChunk(uint64_t,uint64_t,const std::vector<uint8_t>&);
    void completeDmaReadToLocalMemory(PendingDmaOp,const std::vector<uint8_t>&);
    void issueNextDmaLandingChunk(uint64_t);
    void scheduleDmaLandingRetry();
    void retryDmaLandingChunks();
    void finishDmaReadLanding(PendingDmaOp&,bool);
    bool localAccessValid(uint64_t,size_t) const;
    bool localReadAsync(uint64_t,size_t,LocalMemoryClient,uint64_t,LocalReadCallback);
    bool localWriteAsync(uint64_t,const std::vector<uint8_t>&,LocalMemoryClient,uint64_t,LocalWriteCallback);
    uint64_t localAccessDelayCycles(size_t) const;
    void issueLocalAccess(uint64_t);
    void tryIssueLocalAccesses();
    void enqueueBankedLocalAccess(uint64_t);
    void stepBankedLocalAccess();
    void handleLocalAccessEvent(Event*);
    void enableBanked(uint32_t latency=1,uint32_t responseCapacity=32) {
        size=2*1024*1024; storage.resize(size,7);
        localBankScheduler_.reset(new LocalGmBankScheduler({8,128,latency,responseCapacity,256}));
    }
    void advance(uint64_t tick) {
        while (!events.empty() && events.begin()->first<=tick) {
            auto it=events.begin(); now=it->first;
            Event* ev=it->second; events.erase(it); handleLocalAccessEvent(ev);
        }
        now=tick;
    }
    void drain() {
        while (!events.empty()) {
            auto it=events.begin(); now=it->first;
            Event* ev=it->second; events.erase(it); handleLocalAccessEvent(ev);
        }
        assert(localAccessPending_.empty());
    }
};
'''

MAIN = r'''
int main() {
    for (size_t bytes : {size_t(256),size_t(4096)}) {
        GlobalMemoryImplement gm;
        std::printf("case,read_%zu\n",bytes);
        for (uint64_t tag=1;tag<=3;++tag) {
            bool accepted=gm.localReadAsync(0,bytes,LocalMemoryClient::Control,tag,[&gm,bytes](bool ok,uint64_t t,const auto& data){
                assert(ok && data.size()==bytes);
                assert(gm.now==t*(1+(bytes+255)/256));
                std::printf("return,%llu,%llu,%zu\n",(unsigned long long)t,(unsigned long long)gm.now,bytes);
            });
            assert(accepted);
            std::printf("accept,%llu,%llu,%zu\n",(unsigned long long)tag,(unsigned long long)gm.now,bytes);
        }
        gm.drain();
    }
    GlobalMemoryImplement gm;
    std::printf("case,independent_read_write\n");
    assert(gm.localReadAsync(0,256,LocalMemoryClient::Control,1,[&gm](bool ok,uint64_t t,const auto& data){
        assert(ok && gm.now==2 && data[0]==7);
        std::printf("return,%llu,%llu,256\n",(unsigned long long)t,(unsigned long long)gm.now);
    }));
    assert(gm.localWriteAsync(4096,std::vector<uint8_t>(256,9),LocalMemoryClient::Control,2,[&gm](bool ok,uint64_t t){
        assert(ok && gm.now==2 && gm.storage[4096]==9);
        std::printf("return,%llu,%llu,256\n",(unsigned long long)t,(unsigned long long)gm.now);
    }));
    gm.drain();
}
'''


BANK_MAIN = r"""
    for (bool reverse : {false,true}) {
        GlobalMemoryImplement bank; bank.enableBanked();
        unsigned callbacks=0;
        auto read=[&]{assert(bank.localReadAsync(0,128,LocalMemoryClient::WCP,1,
            [&](bool ok,uint64_t,const auto& data){assert(ok && bank.now==2 && data[0]==7);++callbacks;}));};
        auto write=[&]{assert(bank.localWriteAsync(0,std::vector<uint8_t>(128,9),LocalMemoryClient::Dma,2,
            [&](bool ok,uint64_t){assert(ok && bank.now==3 && bank.storage[0]==9);++callbacks;}));};
        if(reverse){write();read();}else{read();write();} bank.drain();
        assert(callbacks==2 && bank.localBankScheduler_->stats().bankConflictCycles==1);
    }
    { // Whole 4 KiB transfer occupies one ingress for 32 beats.
        GlobalMemoryImplement bank;bank.enableBanked();
        assert(bank.localReadAsync(0,4096,LocalMemoryClient::SFU,0,
            [&](bool ok,uint64_t,const auto& data){assert(ok&&data.size()==4096&&bank.now==33);}));
        bank.drain();assert(bank.localBankScheduler_->stats().laneBeats[0]==32);
    }
    { // Cross-line byte masks preserve neighbors.
        GlobalMemoryImplement bank;bank.enableBanked();
        assert(bank.localWriteAsync(127,{1,2,3},LocalMemoryClient::Control,0,
            [&](bool ok,uint64_t){assert(ok&&bank.now==3);}));bank.drain();
        assert(bank.storage[126]==7&&bank.storage[127]==1&&bank.storage[128]==2&&bank.storage[129]==3&&bank.storage[130]==7);
    }
    { // Different banks, but bounded return capacity serializes them.
        GlobalMemoryImplement bank;bank.enableBanked(4,1); unsigned callbacks=0;
        assert(bank.localReadAsync(0,128,LocalMemoryClient::Control,0,
            [&](bool ok,uint64_t,const auto&){assert(ok&&bank.now==5);++callbacks;}));
        assert(bank.localReadAsync(128,128,LocalMemoryClient::Dma,0,
            [&](bool ok,uint64_t,const auto&){assert(ok&&bank.now==9);++callbacks;}));
        bank.drain();assert(callbacks==2&&bank.localBankScheduler_->stats().responseHighWater==1);
        assert(bank.localBankScheduler_->stats().responseBlockedLaneCycles==4);
    }
    { // Queue backpressure and overflow-safe address validation.
        GlobalMemoryImplement bank;bank.enableBanked();bank.localAccessQueueDepth_=2;
        unsigned callbacks=0; auto cb=[&](bool ok,uint64_t,const auto&){assert(ok);++callbacks;};
        assert(bank.localReadAsync(0,1,LocalMemoryClient::Control,0,cb));
        assert(bank.localReadAsync(0,1,LocalMemoryClient::Control,0,cb));
        assert(!bank.localReadAsync(0,1,LocalMemoryClient::Control,0,cb));
        bank.drain();assert(callbacks==2);
        assert(!bank.localReadAsync(bank.size,1,LocalMemoryClient::Control,0,cb));
        assert(!bank.localReadAsync(UINT64_MAX,2,LocalMemoryClient::Control,0,cb));
    }
    { // Writes commit at grant; subsequent read sees the new data.
        GlobalMemoryImplement bank;bank.enableBanked();
        assert(bank.localWriteAsync(0,{42},LocalMemoryClient::Control,0,[&](bool ok,uint64_t){assert(ok&&bank.now==2);}));
        assert(bank.localReadAsync(0,1,LocalMemoryClient::Dma,0,[&](bool ok,uint64_t,const auto& data){assert(ok&&bank.now==3&&data[0]==42);}));
        bank.advance(1);assert(bank.storage[0]==42);bank.drain();
    }
    { // Cancel a partially granted old-generation write, then reuse its range.
        GlobalMemoryImplement bank;bank.enableBanked();bank.attentionClusterEnable_=true;
        unsigned cancelled=0,completed=0;
        assert(bank.localWriteAsync(0,std::vector<uint8_t>(4096,9),LocalMemoryClient::Control,0,[&](bool ok,uint64_t){assert(!ok);++cancelled;}));
        bank.advance(1);assert(bank.storage[0]==9&&bank.storage[128]==7);
        bank.localAccessPending_.begin()->second.cancelled=true;
        bank.attentionGenerationFence_.generation=2;
        assert(bank.localWriteAsync(0,std::vector<uint8_t>(4096,42),LocalMemoryClient::Control,1,[&](bool ok,uint64_t){assert(ok);++completed;}));
        bank.advance(32);assert(bank.storage[128]==7);bank.drain();
        assert(cancelled==1&&completed==1);
        assert(std::all_of(bank.storage.begin(),bank.storage.begin()+4096,[](uint8_t b){return b==42;}));
    }
    { // Non-edge arrival still grants only at the next global GM edge.
        GlobalMemoryImplement bank;bank.enableBanked();bank.converter.factor=1000;bank.now=625;
        assert(bank.localReadAsync(0,128,LocalMemoryClient::Control,0,[&](bool ok,uint64_t,const auto&){assert(ok&&bank.now==2000);}));
        bank.drain();
    }
    { // Control-initiated 8 KiB DMA must land before publishing ready.
        GlobalMemoryImplement bank;bank.enableBanked();bank.write_u64_to_storage(0x1000,0);
        bank.ctrlRegisterPendingReadRequest(7,0x10000,0x1000,42,8192);
        bank.landControlledReadChunk(7,0x10000,std::vector<uint8_t>(8192,9));
        bank.advance(65);assert(bank.flag()==0&&bank.controlCompletions==0);
        bank.drain();assert(bank.flag()==42&&bank.controlCompletions==1);
        assert(bank.storage[0x10000]==9&&bank.storage[0x11fff]==9&&bank.request_pending.empty());
        assert(bank.localWriteBytes_==8192&&bank.pendingDmaLandings_.empty());
    }
    { // Multiple network chunks arrive out of address order; flag waits for both.
        GlobalMemoryImplement bank;bank.enableBanked();bank.write_u64_to_storage(0x1000,0);
        bank.ctrlRegisterPendingReadRequest(7,0x10000,0x1000,42,8192);
        bank.landControlledReadChunk(7,0x11000,std::vector<uint8_t>(4096,8));
        bank.landControlledReadChunk(7,0x10000,std::vector<uint8_t>(4096,9));
        bank.advance(33);assert(bank.flag()==0&&bank.controlCompletions==0);
        bank.drain();assert(bank.flag()==42&&bank.controlCompletions==1);
        assert(bank.storage[0x10000]==9&&bank.storage[0x11000]==8);
    }
    { // Queue-full DMA retries eventually complete, preserving data/ready order.
        GlobalMemoryImplement bank;bank.enableBanked();bank.localAccessQueueDepth_=1;
        bank.write_u64_to_storage(0x1000,0);
        assert(bank.localReadAsync(0,128,LocalMemoryClient::Control,0,[](bool ok,uint64_t,const auto&){assert(ok);}));
        bank.ctrlRegisterPendingReadRequest(7,0x10000,0x1000,42,128);
        bank.landControlledReadChunk(7,0x10000,std::vector<uint8_t>(128,9));
        bank.drain();assert(bank.flag()==42&&bank.controlCompletions==1&&bank.dmaLandingBackpressureRetries_>0);
    }
    { // Old control-DMA callback cannot complete a reused request ID.
        GlobalMemoryImplement bank;bank.enableBanked();bank.attentionClusterEnable_=true;
        bank.write_u64_to_storage(0x1000,0);
        bank.ctrlRegisterPendingReadRequest(7,0x10000,0x1000,9,8192);
        bank.landControlledReadChunk(7,0x10000,std::vector<uint8_t>(8192,9));bank.advance(1);
        for(auto& item:bank.localAccessPending_)item.second.cancelled=true;
        bank.request_pending.clear();bank.attentionGenerationFence_.generation=2;
        bank.ctrlRegisterPendingReadRequest(7,0x10000,0x1000,42,4096);
        bank.landControlledReadChunk(7,0x10000,std::vector<uint8_t>(4096,42));bank.drain();
        assert(bank.flag()==42&&bank.controlCompletions==1&&bank.request_pending.empty());
        assert(std::all_of(bank.storage.begin()+0x10000,bank.storage.begin()+0x11000,[](uint8_t b){return b==42;}));
    }
    std::puts("PASS banked production functions: bank conflicts, RAW/WAR, split masks, bounded queues, cancellation and clock phase");
"""


def probe(output):
    root = Path(__file__).resolve().parents[2]
    source_path = root / 'src/sst/elements/golem/globalmemory/globalmemory.cc'
    source = source_path.read_text()
    # All local submission, issue and completion bodies are contiguous here.
    start = source.index('bool GlobalMemoryImplement::localAccessValid(')
    end = source.index('void GlobalMemoryImplement::wr_to_network(', start)
    register_start=source.index('void GlobalMemoryImplement::ctrlRegisterPendingReadRequest(')
    register_end=source.index('bool GlobalMemoryImplement::ctrlIsReadRequestPending(',register_start)
    bodies = source[register_start:register_end]+source[start:end]
    header=source_path.with_suffix('.h').read_text()
    dma=header[header.index('    struct PendingDmaOp {'):header.index('    struct PendingReadReply {')]
    pending=header[header.index('    struct PendingReadRequest {'):header.index('    struct PendingLocalAccess {')]
    landing=header[header.index('    struct PendingDmaLanding {'):header.index('    uint64_t baseAddr =')]
    harness=HARNESS.replace('/*PENDING_DMA_STRUCTS*/',dma+pending+landing)
    with tempfile.TemporaryDirectory(prefix='golem-gm-timing-') as tmp:
        cpp = Path(tmp) / 'probe.cpp'
        exe = Path(tmp) / 'probe'
        cpp.write_text(harness + bodies + MAIN.replace('\n}\n', BANK_MAIN+'\n}\n'))
        subprocess.run(['g++', '-std=c++17', '-Wall', '-Wextra', '-O0', '-I', str(source_path.parent), str(cpp), '-o', str(exe)], check=True)
        trace = subprocess.run([str(exe)], check=True, text=True, capture_output=True).stdout
    report = {'status': 'PASS', 'scope': 'verbatim GM C++ functions with isolated event delivery; not full SST or RTL',
              'source': str(source_path.relative_to(root)),
              'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
              'scheduler_bodies_sha256': hashlib.sha256(bodies.encode()).hexdigest(),
              'gm_header_sha256':hashlib.sha256(header.encode()).hexdigest(),
              'bank_scheduler_sha256':hashlib.sha256((source_path.parent/'localGmBankScheduler.h').read_bytes()).hexdigest(),
              'models_tested':['legacy','banked'],
              'banked_checks':['fixed-priority conflicts','same-tick order','4 KiB splitting','cross-line byte masks','return backpressure','API queue full','RAW/WAR visibility','cancel/reuse','clock phase','control DMA ready-after-landing','out-of-order chunks','DMA queue retry','stale control request callback'],
              'clock_ghz': 1, 'bytes_per_cycle_per_port': 256, 'base_latency_cycles': 1,
              'read_ports': 1, 'write_ports': 1, 'trace': trace.splitlines()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(trace, end='')
    print('PASS:', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/gm_timing_audit/probe.json'))
    probe(parser.parse_args().output)
