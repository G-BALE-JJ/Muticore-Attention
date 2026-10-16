"""Distinct N matrices, K order, tail owners and delayed ACKs on single-context arrays."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from test_decoder_groups import timed_stub
from test_decoder_panels import ASYNC_STUB
from test_decoder_tail import ROOT

SOURCE=r'''
#include <sst/elements/golem/attention/decoderOpJob.h>
#include <cassert>
using namespace SST::Golem;
std::vector<uint8_t> run(unsigned parallel,unsigned tiles,unsigned owner,bool prefetch) {
 setenv("GOLEM_DECODER_LM_N_PARALLEL",std::to_string(parallel).c_str(),1);
 setenv("GOLEM_DECODER_GEMM_PREFETCH",prefetch?"1":"0",1);
 GlobalMemoryAPI gm;WorkerCommandProcessorAPI wcp;SFUAPI sfu;DecoderOpJob job(&gm,&wcp,&sfu);
 DecoderOpDesc d{};d.magic=GOLEM_DECODER_OP_MAGIC;d.size_bytes=sizeof(d);d.op=1;d.rows=1;
 d.input_dim=192;d.output_dim=tiles*64;d.manager=owner%4;d.lane=owner/4;d.output_tile_begin=128;
 d.input_addr=0x40000000;d.output_addr=0x60000000;d.node_stride=0x08000000;d.weights_offset=0x01000000;
 d.completion_addr=0x70000000;d.epoch=9;
 for(unsigned k=0;k<192;++k)gm.put(d.input_addr+k*2,(int(k%17)-8)*.0625);
 std::map<uint64_t,size_t> wanted;
 for(unsigned tile=owner;tile<tiles;tile+=16)for(unsigned k=0;k<3;++k) {
  uint64_t addr=(1+(tile+128)%4)*d.node_stride+d.weights_offset+((tile+128)/4*3+k)*8192;wanted[addr]=8192;
  for(unsigned n=0;n<64;++n)for(unsigned j=0;j<64;++j)
   gm.put(addr+(n*64+j)*2,(n==j?1:-.015625)*(tile%7+1)*(k+1)*.03125);
 }
 assert(job.start(d));
 for(uint64_t t=1;t<1000000 && !job.complete();++t) {
  wcp.step(t);job.tick(t,t*1000);if(!gm.pending.empty())assert(!job.complete());if(t%13==0)gm.drain();
 }
 assert(job.complete() && job.status()==0 && gm.pending.empty() && gm.delayedReads.empty() && wcp.events.empty());
 const unsigned count=owner<tiles?(tiles-1-owner)/16+1:0;
 assert(gm.writes.size()==count+1 && gm.writes.back().first==d.completion_addr && gm.bytes[d.completion_addr]==9);
 std::map<uint64_t,size_t> actual;
 for(auto e:gm.reads)if(e.first!=d.input_addr)assert(actual.emplace(e).second);
 assert(actual==wanted && gm.reads.size()==wanted.size()+bool(count));
 std::vector<uint8_t> result;
 for(unsigned tile=owner;tile<tiles;tile+=16)for(unsigned n=0;n<64;++n) {
  double expected=0;
  for(unsigned k=0;k<3;++k) {
   double dot=0;const uint64_t addr=(1+(tile+128)%4)*d.node_stride+d.weights_offset+((tile+128)/4*3+k)*8192;
   for(unsigned j=0;j<64;++j)dot=vectorFp16Value(dot+vectorFp16Value(gm.get(d.input_addr+(k*64+j)*2)*gm.get(addr+(n*64+j)*2)));
   expected=vectorFp16Value(expected+dot);
  }
  const auto addr=d.output_addr+tile*128+n*2;assert(gm.get(addr)==expected);
  result.push_back(gm.bytes[addr]);result.push_back(gm.bytes[addr+1]);
 }
 return result;
}
int main() {
 for(const char* slots:{"2","4","8"}) {
  setenv("GOLEM_DECODER_PREFETCH_SLOTS",slots,1);
  for(auto shape:std::vector<std::pair<unsigned,unsigned>>{{128,0},{84,15},{37,4},{1,0},{1,15}})
   for(bool prefetch:{false,true}) {
    auto baseline=run(1,shape.first,shape.second,prefetch);
    for(unsigned p:{2,4,8})assert(run(p,shape.first,shape.second,prefetch)==baseline);
   }
 }
}
'''

class DecoderLmParallelTest(unittest.TestCase):
    def test_distinct_N_tiles_K_order_and_timed_single_context_ports(self):
        a=ASYNC_STUB.index('struct GlobalMemoryAPI {');b=ASYNC_STUB.index('struct WorkerCommandProcessorAPI {')
        stub=timed_stub();c=stub.index('struct GlobalMemoryAPI {');d=stub.index('struct WorkerCommandProcessorAPI {')
        stub=stub[:c]+ASYNC_STUB[a:b]+stub[d:]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('globalmemory/globalmemory.h','workercmdproc/workercmdproc.h'):
                p=root/'sst/elements/golem'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('#include "stub.h"\n')
            (root/'stub.h').write_text(stub);(root/'test.cc').write_text(SOURCE)
            subprocess.run(['g++','-std=c++17','-O2','-I',str(root),'-I',str(ROOT/'src'),str(root/'test.cc'),'-o',str(root/'test')],check=True)
            r=subprocess.run([str(root/'test')],capture_output=True,timeout=180)
            self.assertEqual(r.returncode,0,r.stderr.decode())

if __name__=='__main__':unittest.main()
