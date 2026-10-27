"""Production scheduler: delayed ports, tails, and ownership of each array's C."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from test_decoder_panels import ASYNC_STUB
from test_decoder_tail import ROOT

WCP = r'''
struct WorkerCommandProcessorAPI {
 struct Array {std::vector<double> b,a,c;bool busy=false;};
 std::map<uint32_t,Array> arrays;uint64_t now=0,attempts=0,drains=0,restores=0;
 std::vector<std::pair<uint64_t,std::function<void()>>> pending;
 bool accept(){return ++attempts%7!=0;}
 void step(uint64_t t){now=t;for(size_t i=pending.size();i-->0;)if(pending[i].first<=t){auto cb=std::move(pending[i].second);pending.erase(pending.begin()+i);cb();}}
 bool programGemmMatrixGroupClassBankAsync(const std::vector<uint32_t>& ids,uint32_t,const std::vector<double>& b,size_t,
 AttentionClusterTrafficClass,uint64_t tag,uint64_t,std::function<void(bool,uint64_t)> cb){
  if(!accept())return false;assert(b.size()==4096);for(auto id:ids){assert(!arrays[id].busy);arrays[id].b=b;}
  pending.push_back({now+2,[=]{cb(true,tag);}});return true;
 }
 bool programGemmInputScatterBankAsync(const std::vector<uint32_t>& ids,uint32_t,const std::vector<double>& a,size_t,
 AttentionClusterTrafficClass,uint64_t tag,uint64_t,std::function<void(bool,uint64_t)> cb){
  if(!accept())return false;assert(a.size()==ids.size()*64);for(size_t i=0;i<ids.size();++i){assert(!arrays[ids[i]].busy);arrays[ids[i]].a.assign(a.begin()+i*64,a.begin()+(i+1)*64);}
  pending.push_back({now+2,[=]{cb(true,tag);}});return true;
 }
 bool launchGemmArrayGroupActiveBank(const std::vector<uint32_t>& ids,uint32_t,uint64_t mode,uint32_t,uint64_t,
 std::function<void(uint32_t,uint64_t)> cb){
  if(!accept())return false;for(auto id:ids){auto& a=arrays[id];assert(!a.busy);a.busy=true;if(!mode)a.c.assign(64,0);
   assert(a.c.size()==64);for(size_t n=0;n<64;++n){float v=a.c[n];for(size_t k=0;k<64;++k)v+=float(a.a[k])*float(a.b[n*64+k]);a.c[n]=v;}
   pending.push_back({now+11,[=]{arrays[id].busy=false;cb(id,now);}});
  }return true;
 }
 bool readGemmOutputGroupClassAsync(const std::vector<uint32_t>& ids,size_t bytes,AttentionClusterTrafficClass,
 uint64_t tag,uint64_t,std::function<void(bool,uint64_t,const std::vector<double>&)> cb){
  if(!accept())return false;++drains;std::vector<double> v;for(auto id:ids){assert(!arrays[id].busy);v.insert(v.end(),arrays[id].c.begin(),arrays[id].c.end());}
  if(bytes==2)for(auto& x:v)x=golem_fp16_to_float(golem_float_to_fp16(float(x)));
  pending.push_back({now+3,[=]{cb(true,tag,v);}});return true;
 }
 bool writeGemmOutputGroupClassAsync(const std::vector<uint32_t>& ids,const std::vector<double>& v,size_t,AttentionClusterTrafficClass,
 uint64_t tag,uint64_t,std::function<void(bool,uint64_t)> cb){
  if(!accept())return false;++restores;assert(v.size()==ids.size()*64);for(size_t i=0;i<ids.size();++i){assert(!arrays[ids[i]].busy);arrays[ids[i]].c.assign(v.begin()+i*64,v.begin()+(i+1)*64);}
  pending.push_back({now+3,[=]{cb(true,tag);}});return true;
 }
};
}}
'''
STUB = ASYNC_STUB[:ASYNC_STUB.index('struct WorkerCommandProcessorAPI {')] + WCP
SOURCE = r'''
#include <sst/elements/golem/attention/reuseWindowGemm.h>
#include <string>
using namespace SST::Golem;
void run(uint32_t rows,uint32_t nk,uint32_t nn,bool initial,bool resident) {
 GlobalMemoryAPI gm;WorkerCommandProcessorAPI wcp;ReuseWindowGemm job(&gm,&wcp);
 for(uint32_t n=0;n<nn;++n)for(uint32_t k=0;k<nk;++k)for(uint32_t j=0;j<64;++j)for(uint32_t i=0;i<64;++i)
  gm.put(0x10000000+(uint64_t(n)*nk+k)*8192+(j*64+i)*2,j==i?(n+1)*.0625:0);
 std::vector<uint32_t> tiles;for(uint32_t n=0;n<nn;++n)tiles.push_back(n);
 bool done=false;uint32_t outputs=0,inputs=0;
 uint64_t outputAckAt=0,clock=0,launchesAtOutput=0;bool computedWhileOutputPending=false;
 ReuseWindowGemm::Initial init;
 std::function<uint64_t(uint32_t)> source;
 if(resident){source=[rows](uint32_t k){return 0x10000+uint64_t(k)*rows*128;};
  for(uint32_t k=0;k<nk;++k)for(uint32_t r=0;r<rows;++r)for(uint32_t j=0;j<64;++j)gm.put(source(k)+(r*64+j)*2,(r+j+k+1)*.03125);}
 if(initial)init=[rows](uint32_t,ReuseWindowGemm::Loaded cb){cb(true,std::vector<double>(rows*64,.25));};
 job.start(rows,nk,tiles,0x10000,0x90000,0xa0000,
  [&](uint32_t k,ReuseWindowGemm::Loaded cb){++inputs;std::vector<double> a(rows*64);for(uint32_t r=0;r<rows;++r)for(uint32_t j=0;j<64;++j)a[r*64+j]=(r+j+k+1)*.03125;cb(true,a);},
  [nk](uint32_t n,uint32_t k){return 0x10000000+(uint64_t(n)*nk+k)*8192;},
  [&](uint32_t n,const std::vector<double>& v){assert(!outputAckAt);assert(v.size()==rows*64);for(uint32_t r=0;r<rows;++r)for(uint32_t j=0;j<64;++j){float e=initial?.25:0;for(uint32_t k=0;k<nk;++k)e+=float((r+j+k+1)*.03125)*float((n+1)*.0625);assert(v[r*64+j]==golem_fp16_to_float(golem_float_to_fp16(e)));}++outputs;outputAckAt=clock+(std::getenv("GOLEM_TEST_ACK_DELAY")?std::stoul(std::getenv("GOLEM_TEST_ACK_DELAY")):1000);launchesAtOutput=job.launches();},
  [&](bool ok){assert(ok);assert(!outputAckAt);done=true;},init,source,true);
 for(uint64_t t=1;t<100000 && !done;++t){clock=t;wcp.step(t);if(t%3==0)gm.drain();
  if(std::getenv("GOLEM_TEST_ACK_BEFORE_TICK") && outputAckAt && t>=outputAckAt){outputAckAt=0;job.resumeOutput();}
  job.tick(t);
  if(outputAckAt){assert(!done);if(job.launches()>launchesAtOutput)computedWhileOutputPending=true;
   if(t>=outputAckAt){outputAckAt=0;job.resumeOutput();}}
 }
 assert(done && outputs==nn && gm.delayedReads.empty() && wcp.pending.empty());
 unsigned chunk=std::min(nk,unsigned(std::stoul(std::getenv("GOLEM_REUSE_WINDOW_K_TILES"))));
 assert(wcp.drains==nn*((nk+chunk-1)/chunk));
 assert(wcp.restores==nn*((nk+chunk-1)/chunk-1+(initial?1:0)));
 assert(inputs==(resident?0:nk*((nn+3)/4)));assert(gm.reads.size()==nn*nk);
 assert(job.prefetchRequests()==nn*nk && job.prefetchHits()==nn*nk);
 assert(job.cReadBytes()==uint64_t(nn)*((nk+chunk-1)/chunk-1)*rows*256);
 assert(job.cWriteBytes()==job.cReadBytes());
 const bool overlap=std::string(std::getenv("GOLEM_REUSE_WINDOW_OUTPUT_OVERLAP"))=="1";
 if(!std::getenv("GOLEM_TEST_ACK_DELAY"))assert(computedWhileOutputPending==(overlap && !initial));
 job.trace(4,1,4,"test");
}
int main(){for(auto rows:{1u,33u,64u})for(bool resident:{false,true}){run(rows,9,5,false,resident);run(rows,5,3,true,resident);}}
'''

class ReuseWindowGemmTest(unittest.TestCase):
    def test_k_accumulation_tail_retries_and_array_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('globalmemory/globalmemory.h','workercmdproc/workercmdproc.h'):
                path=root/'sst/elements/golem'/name
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text('#include "stub.h"\n')
            (root/'stub.h').write_text(STUB)
            (root/'test.cc').write_text(SOURCE)
            subprocess.run(['g++','-std=c++17','-O2','-I',str(root),'-I',str(ROOT/'src'),str(root/'test.cc'),'-o',str(root/'test')],check=True)
            for chunk in (4,16):
                for groups in (1,2):
                    for early,overlap in ((0,0),(1,0),(0,1),(1,1)):
                        env=dict(os.environ,GOLEM_REUSE_WINDOW_K_TILES=str(chunk),
                                 GOLEM_REUSE_WINDOW_ARRAY_GROUPS=str(groups),
                                 GOLEM_REUSE_WINDOW_EARLY_A=str(early),GOLEM_REUSE_WINDOW_OUTPUT_OVERLAP=str(overlap))
                        result=subprocess.run([str(root/'test')],env=env,capture_output=True,timeout=30)
                        self.assertEqual(result.returncode,0,result.stderr.decode())
                        import re
                        for line in result.stdout.decode().splitlines():
                            if '[GEMM_WINDOW_FLOW]' not in line:continue
                            fields={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)}
                            self.assertEqual(sum(v for k,v in fields.items() if k.startswith('phase_')),
                                             fields['active_cycles'])
            # ACK after a GM read callback but before the next scheduler tick
            # must preserve the next N tile's restored partial C.
            for delay in (1,2,3,7,13,29):
                env=dict(os.environ,GOLEM_REUSE_WINDOW_K_TILES='4',
                         GOLEM_REUSE_WINDOW_ARRAY_GROUPS='1',GOLEM_REUSE_WINDOW_EARLY_A='1',
                         GOLEM_REUSE_WINDOW_OUTPUT_OVERLAP='1',GOLEM_TEST_ACK_DELAY=str(delay),
                         GOLEM_TEST_ACK_BEFORE_TICK='1')
                result=subprocess.run([str(root/'test')],env=env,capture_output=True,timeout=30)
                self.assertEqual(result.returncode,0,result.stderr.decode())

if __name__=='__main__':unittest.main()
