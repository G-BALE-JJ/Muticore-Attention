"""Production software cohort: timed transport/landing, eviction and epochs."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from test_decoder_tail import ROOT

STUB = r'''
#pragma once
#include <cassert>
#include <cstdint>
#include <map>
#include <vector>
#include <functional>
#include <algorithm>
namespace SST { namespace Golem {
enum class LocalMemoryClient {RoCC};
enum class ControlTransportMessageKind {GemmWeightJoin,GemmWeightRequest,GemmWeightDelivery,GemmWeightFinish,GemmWeightRelease};
struct ControlTransportMessage {ControlTransportMessageKind kind{};uint64_t jobId=0,tag=0,inputAddr=0,outputAddr=0;
 uint32_t ownerCore=0;std::vector<uint8_t> tensorPayload;};
inline uint64_t testCycle=0,version=0,networkTries=0,portTries=0,totalHbm=0;
inline std::vector<std::pair<uint64_t,std::function<void()>>> events;
inline std::map<uint32_t,std::function<void(const ControlTransportMessage&)>> receivers;
inline uint8_t value(uint64_t source,size_t i){return uint8_t(source/8192+i+17*version);}
struct GlobalMemoryAPI {
 uint32_t core;std::map<uint64_t,uint8_t> bytes;explicit GlobalMemoryAPI(uint32_t c):core(c){}
 uint64_t getBaseAddr()const{return uint64_t(core)*2*1024*1024;}uint64_t getSize()const{return 2*1024*1024;}
 size_t localMaxRequestBytes()const{return 4096;}
 bool sendControlMessage(uint32_t target,const ControlTransportMessage& message){
  if(++networkTries%7==0)return false;
  events.push_back({testCycle+3+networkTries%4,[=]{receivers.at(target)(message);}});return true;
 }
 void dma_read_from_host_to_globalmem(uint64_t source,size_t size,uint64_t destination,std::function<void(bool)> cb){
  assert(size==8192);++totalHbm;events.push_back({testCycle+19,[=]{for(size_t i=0;i<size;++i)bytes[destination+i]=value(source,i);cb(true);}});
 }
 bool localReadAsync(uint64_t a,size_t n,LocalMemoryClient,uint64_t tag,
  std::function<void(bool,uint64_t,const std::vector<uint8_t>&)> cb){
  if(++portTries%9==0)return false;
  assert(a>=getBaseAddr() && a+n<=getBaseAddr()+getSize());
  events.push_back({testCycle+5,[=]{std::vector<uint8_t> b(n);for(size_t i=0;i<n;++i)b[i]=bytes[a+i];cb(true,tag,b);}});return true;
 }
 bool localWriteAsync(uint64_t a,const std::vector<uint8_t>& b,LocalMemoryClient,uint64_t tag,std::function<void(bool,uint64_t)> cb){
  if(++portTries%9==0)return false;
  assert(a>=getBaseAddr() && a+b.size()<=getBaseAddr()+getSize());
  events.push_back({testCycle+7,[=]{for(size_t i=0;i<b.size();++i)bytes[a+i]=b[i];cb(true,tag);}});return true;
 }
};
}}
'''

SOURCE = r'''
#include <sst/elements/golem/attention/reuseWeightCohort.h>
#include <memory>
using namespace SST::Golem;
void run(unsigned width){
 std::vector<std::unique_ptr<GlobalMemoryAPI>> gm;
 std::vector<std::unique_ptr<ReuseWeightCohort>> services;
 for(unsigned i=0;i<width;++i){gm.emplace_back(new GlobalMemoryAPI(4+i*4));
  services.emplace_back(new ReuseWeightCohort(gm.back().get(),4+i*4,width));
  receivers[4+i*4]=[&,i](const ControlTransportMessage& m){services[i]->receive(m);};}
 for(unsigned epoch=1;epoch<=2;++epoch){
  version=epoch;const auto first=testCycle;
  for(unsigned i=0;i+1<width;++i)services[i]->begin(epoch);
  std::vector<unsigned> submitted(width,0),completed(width,0);std::vector<std::array<bool,4>> busy(width);bool released=false;
  for(unsigned step=0;step<30000 && !released;++step){
   ++testCycle;
   if(step==100)services.back()->begin(epoch);
   for(size_t i=events.size();i-->0;)if(events[i].first<=testCycle){auto cb=std::move(events[i].second);events.erase(events.begin()+i);cb();}
   for(unsigned i=0;i<width;++i){
    // <=4 requests per client, with skew, repeated addresses and many evictions.
    if((step>=100 || i+1<width) && (step%7)==i && submitted[i]<24 && !busy[i][submitted[i]%4]){
     const unsigned tile=submitted[i]++;const uint64_t source=0x10000000+uint64_t(tile)*8192;
     busy[i][tile%4]=true;
     const uint64_t destination=gm[i]->getBaseAddr()+0x180000+(tile%4)*8192;
     services[i]->fetch(epoch,source,destination,[&,i,tile,source,destination](bool ok){
      assert(ok);for(size_t j=0;j<8192;++j)assert(gm[i]->bytes[destination+j]==value(source,j));++completed[i];busy[i][tile%4]=false;});
    }
   }
   for(auto& service:services)service->tick();
   if(step<100)assert(totalHbm==(epoch==1?0:24));
   released=true;
   for(unsigned i=0;i<width;++i)released&=completed[i]==24 && services[i]->release(epoch);
  }
  assert(released && testCycle>first+100);
  for(unsigned t=0;t<100;++t){++testCycle;for(size_t i=events.size();i-->0;)if(events[i].first<=testCycle){auto cb=std::move(events[i].second);events.erase(events.begin()+i);cb();}for(auto& s:services)s->tick();}
  assert(events.empty());
  for(auto& s:services)s->trace(epoch);
 }
 assert(totalHbm==48); // exactly one fetch per unique tile in each epoch
 receivers.clear();totalHbm=0;
}
int main(){run(2);run(4);}
'''


class ReuseWeightCohortTest(unittest.TestCase):
    def test_bounded_sharing_landing_ACK_retry_eviction_and_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            header=root/'sst/elements/golem/globalmemory/globalmemory.h'
            header.parent.mkdir(parents=True);header.write_text(STUB)
            (root/'test.cc').write_text(SOURCE)
            subprocess.run(['g++','-std=c++17','-O2','-I',str(root),'-I',str(ROOT/'src'),
                            str(root/'test.cc'),'-o',str(root/'test')],check=True)
            r=subprocess.run([str(root/'test')],capture_output=True,timeout=30)
            self.assertEqual(r.returncode,0,r.stderr.decode())


if __name__=='__main__':unittest.main()
