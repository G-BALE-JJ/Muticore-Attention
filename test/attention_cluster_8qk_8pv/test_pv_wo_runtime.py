"""Exercise production PV handoff with delayed GM, NoC retry and two epochs."""
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUB = r'''
#pragma once
#include <cstdint>
#include <vector>
#include <functional>
#include <map>
namespace SST { namespace Golem {
enum class ControlTransportMessageKind {AttentionPvWoData, AttentionPvWoAck};
enum class LocalMemoryClient {RoCC};
struct ControlTransportMessage {
 ControlTransportMessageKind kind{};
 uint64_t jobId=0,tag=0,outputAddr=0,inputAddr=0;
 uint32_t ownerCore=0,workerCore=0,workerSlot=0,row=0,groupQueryRowBegin=0,
 headDim=0,queryLength=0,kvLength=0,numQueryHeads=0,numKvHeads=0,kvHeadIndex=0;
 double value=0;std::vector<uint8_t> tensorPayload;
};
struct GlobalMemoryAPI {
 uint32_t core=0;bool blocked=false,rejectWrite=false;
 std::map<uint64_t,uint8_t> bytes;std::vector<std::function<void()>> events;
 std::function<void(uint32_t,const ControlTransportMessage&)> send;
 uint64_t getBaseAddr()const{return uint64_t(core)*0x200000;}
 uint64_t getSize()const{return 0x200000;}
 size_t localMaxRequestBytes()const{return 4096;}
 bool sendControlMessage(uint32_t target,const ControlTransportMessage& m){
  if(blocked)return false;send(target,m);return true;
 }
 bool localWriteAsync(uint64_t addr,const std::vector<uint8_t>& data,LocalMemoryClient,
 uint64_t tag,std::function<void(bool,uint64_t)> cb){
  if(rejectWrite){rejectWrite=false;return false;}
  events.push_back([=]{for(size_t i=0;i<data.size();++i)bytes[addr+i]=data[i];cb(true,tag);});return true;
 }
 void step(){auto pending=std::move(events);events.clear();for(auto& e:pending)e();}
};
}}
'''
SOURCE = r'''
#include <sst/elements/golem/attention/pvWoHandoff.h>
#include <cassert>
#include <memory>
using namespace SST::Golem;
int main(){
 GlobalMemoryAPI gm[20];std::unique_ptr<PvWoHandoff> service[20];
 for(uint32_t c=4;c<20;++c){gm[c].core=c;service[c].reset(new PvWoHandoff(&gm[c],c));
  gm[c].send=[&](uint32_t target,const ControlTransportMessage& m){service[target]->receive(m);};}
 for(uint32_t epoch=1;epoch<=2;++epoch){
  assert(!service[4]->consume(epoch,0,0));
  for(uint32_t head=0;head<32;++head)for(uint32_t manager=0;manager<4;++manager)
   for(uint32_t lane=0;lane<4;++lane){
    const uint32_t producer=12+(head+manager+lane)%8,target=4+manager+lane*4;
    ControlTransportMessage source{};source.jobId=0xD1000401ull+((epoch-1ull)<<16)+(uint64_t(head/4)<<32);
    source.tag=source.jobId+256;source.row=manager*1024;source.groupQueryRowBegin=(head%4)*256+lane*64;
    source.queryLength=256;source.kvLength=1024;source.numQueryHeads=32;source.numKvHeads=8;
    source.kvHeadIndex=head/4;source.headDim=64;
    std::vector<uint8_t> payload(8192);
    for(uint32_t b=0;b<8192;++b)payload[b]=uint8_t(head*19+epoch*7+b*11+(b>>8));
    bool done=false;
    gm[producer].blocked=true;gm[target].rejectWrite=true;
    assert(service[producer]->dispatch(source,payload,[&](bool ok){assert(ok);done=true;}));
    assert(!service[producer]->dispatch(source,payload,[](bool){}));
    service[producer]->tick();assert(!done);gm[producer].blocked=false;
    for(uint32_t tick=0;tick<12 && !done;++tick){
     for(uint32_t c=4;c<20;++c)service[c]->tick();
     // Producer completion must wait until both GM chunks have ACKed.
     if(tick==0)assert(!done);
     for(uint32_t c=4;c<20;++c)gm[c].step();
    }
    assert(done);
    for(uint32_t b=0;b<8192;++b)assert(gm[target].bytes[gm[target].getBaseAddr()+
     DecoderMemoryLayout::decoder_pv_wo_landing_gm+head*8192+b]==payload[b]);
    if(head==0){auto duplicate=source;duplicate.kind=ControlTransportMessageKind::AttentionPvWoData;
     duplicate.workerCore=target;duplicate.ownerCore=producer;duplicate.workerSlot=head;
     duplicate.outputAddr=manager;duplicate.inputAddr=epoch;duplicate.tensorPayload=payload;
     bool rejected=false;try{service[target]->receive(duplicate);}catch(const std::logic_error&){rejected=true;}
     assert(rejected);}
   }
  for(uint32_t manager=0;manager<4;++manager)for(uint32_t lane=0;lane<4;++lane){
   auto& s=service[4+manager+lane*4];assert(!s->consume(epoch+1,manager,lane));
   assert(s->consume(epoch,manager,lane));assert(!s->consume(epoch,manager,lane));
  }
 }
}
'''


class PvWoRuntimeTest(unittest.TestCase):
    def test_timed_landing_retries_bits_readiness_and_generations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            stub = path/'sst/elements/golem/globalmemory/globalmemory.h'
            stub.parent.mkdir(parents=True)
            stub.write_text(STUB)
            (path/'test.cc').write_text(SOURCE)
            subprocess.run(['g++','-std=c++17','-O2','-I',str(path),'-I',str(ROOT/'src'),
                            str(path/'test.cc'),'-o',str(path/'test')],check=True)
            result = subprocess.run([str(path/'test')],capture_output=True,timeout=45)
            self.assertEqual(result.returncode,0,result.stderr.decode())
            self.assertEqual(result.stdout.count(b'[PV_WO_HANDOFF]'),32)


if __name__ == '__main__':
    unittest.main()
