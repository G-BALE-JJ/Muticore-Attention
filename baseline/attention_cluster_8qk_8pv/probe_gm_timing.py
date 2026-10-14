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
#include <deque>
#include <functional>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <vector>
using LocalMemoryClient = int;
using LocalReadCallback = std::function<void(bool,uint64_t,const std::vector<uint8_t>&)>;
using LocalWriteCallback = std::function<void(bool,uint64_t)>;
struct Event { virtual ~Event() = default; };
struct LocalMemoryAccessEvent : Event {
    uint64_t id;
    explicit LocalMemoryAccessEvent(uint64_t n):id(n){}
    uint64_t requestId() const { return id; }
};
struct BusyIntervalUnion { bool add(uint64_t,uint64_t) { return true; } };
struct Converter { uint64_t convertToCoreTime(uint64_t n) { return n; } };
struct Fence { uint64_t active() { return 0; } bool matches(uint64_t) { return true; } };
struct Output { void fatal(int,int,const char*) { throw std::runtime_error("fatal"); } };
#define CALL_INFO 0
uint64_t ceilDivLocalAccess(uint64_t n,uint64_t d) { return (n+d-1)/d; }
struct GlobalMemoryImplement {
    struct PendingLocalAccess {
        enum class Kind { Read, Write };
        Kind kind;
        uint64_t requestId=0,addr=0,tag=0,submitCycle=0,ownerGeneration=0;
        size_t length=0;
        LocalMemoryClient client=0;
        bool issued=false,cancelled=false;
        LocalReadCallback readCallback;
        LocalWriteCallback writeCallback;
        std::vector<uint8_t> writeData;
    };
    uint64_t now=0,baseAddr=0,size=8192;
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
    Converter converter; Converter* localAccessTC_=&converter;
    Output out; Output* output=&out;
    std::unordered_map<uint64_t,PendingLocalAccess> localAccessPending_;
    std::deque<uint64_t> localReadQueue_,localWriteQueue_;
    std::multimap<uint64_t,Event*> events;
    struct Link {
        GlobalMemoryImplement* gm;
        void send(uint64_t delay,Event* ev) {
            auto* access=dynamic_cast<LocalMemoryAccessEvent*>(ev);
            auto& req=gm->localAccessPending_.at(access->requestId());
            std::printf("issue,%llu,%llu,%zu\n",(unsigned long long)req.tag,
                        (unsigned long long)gm->now,req.length);
            gm->events.emplace(gm->now+delay,ev);
        }
    } link{this};
    Link* localAccessSelfLink_=&link;
    uint64_t getCurrentSimCycle() { return now; }
    void retryDmaLandingChunks() { throw std::runtime_error("unexpected DMA retry"); }
    bool localAccessValid(uint64_t,size_t) const;
    bool localReadAsync(uint64_t,size_t,LocalMemoryClient,uint64_t,LocalReadCallback);
    bool localWriteAsync(uint64_t,const std::vector<uint8_t>&,LocalMemoryClient,uint64_t,LocalWriteCallback);
    uint64_t localAccessDelayCycles(size_t) const;
    void issueLocalAccess(uint64_t);
    void tryIssueLocalAccesses();
    void handleLocalAccessEvent(Event*);
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
            bool accepted=gm.localReadAsync(0,bytes,0,tag,[&gm,bytes](bool ok,uint64_t t,const auto& data){
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
    assert(gm.localReadAsync(0,256,0,1,[&gm](bool ok,uint64_t t,const auto& data){
        assert(ok && gm.now==2 && data[0]==7);
        std::printf("return,%llu,%llu,256\n",(unsigned long long)t,(unsigned long long)gm.now);
    }));
    assert(gm.localWriteAsync(4096,std::vector<uint8_t>(256,9),0,2,[&gm](bool ok,uint64_t t){
        assert(ok && gm.now==2 && gm.storage[4096]==9);
        std::printf("return,%llu,%llu,256\n",(unsigned long long)t,(unsigned long long)gm.now);
    }));
    gm.drain();
}
'''


def probe(output):
    root = Path(__file__).resolve().parents[2]
    source_path = root / 'src/sst/elements/golem/globalmemory/globalmemory.cc'
    source = source_path.read_text()
    # All local submission, issue and completion bodies are contiguous here.
    start = source.index('bool GlobalMemoryImplement::localAccessValid(')
    end = source.index('void GlobalMemoryImplement::completeDmaReadToLocalMemory(', start)
    bodies = source[start:end]
    with tempfile.TemporaryDirectory(prefix='golem-gm-timing-') as tmp:
        cpp = Path(tmp) / 'probe.cpp'
        exe = Path(tmp) / 'probe'
        cpp.write_text(HARNESS + bodies + MAIN)
        subprocess.run(['g++', '-std=c++17', '-Wall', '-Wextra', '-O0', str(cpp), '-o', str(exe)], check=True)
        trace = subprocess.run([str(exe)], check=True, text=True, capture_output=True).stdout
    report = {'status': 'PASS', 'scope': 'verbatim GM C++ functions with isolated event delivery; not full SST or RTL',
              'source': str(source_path.relative_to(root)),
              'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
              'scheduler_bodies_sha256': hashlib.sha256(bodies.encode()).hexdigest(),
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
