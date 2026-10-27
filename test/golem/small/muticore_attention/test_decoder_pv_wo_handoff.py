"""Run real Wo on head-major GM input; poison HBM to expose stale fallback."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from test_decoder_tail import ROOT
from test_decoder_groups import timed_stub

SOURCE = r'''
#include <sst/elements/golem/attention/decoderOpJob.h>
#include <cstdlib>
using namespace SST::Golem;
struct Input : PvWoInputInterface {
 bool ready=true;unsigned calls=0;
 bool consume(uint32_t epoch,uint32_t manager,uint32_t lane) override {
  ++calls;assert(epoch==7 && manager==2 && lane==3);return ready;
 }
};
std::vector<uint8_t> run(bool handoff,bool window) {
 setenv("GOLEM_PV_WO_GM_HANDOFF",handoff?"1":"0",1);
 setenv("GOLEM_ALL_GEMM_REUSE_WINDOW",window?"1":"0",1);
 GlobalMemoryAPI gm;WorkerCommandProcessorAPI wcp;SFUAPI sfu;Input input;
 DecoderOpJob job(&gm,&wcp,&sfu,nullptr,&input);
 DecoderOpDesc d{};d.magic=GOLEM_DECODER_OP_MAGIC;d.size_bytes=sizeof(d);
 d.op=1;d.stage=1;d.epoch=7;d.rows=256;d.lane=3;d.manager=2;
 d.input_dim=2048;d.output_dim=64;d.input_addr=0x30000000;d.output_addr=0x40000000;
 d.node_stride=0x08000000;d.weights_offset=0x01000000;
 std::vector<uint8_t> expectedPacked(64*2048*2);
 for(uint32_t r=0;r<64;++r)for(uint32_t k=0;k<2048;++k) {
  const double value=(int((r*23+k*7)%59)-29)*.015625;
  const uint64_t packed=(uint64_t(k/64)*64+r)*128+(k%64)*2;
  gm.put(DecoderMemoryLayout::decoder_pv_wo_landing_gm+packed,value);
  const uint16_t half=golem_float_to_fp16(value);
  expectedPacked[packed]=half&255;expectedPacked[packed+1]=half>>8;
  gm.put(d.input_addr+((d.lane*64+r)*2048+k)*2,handoff?-13:value);
 }
 for(uint32_t k=0;k<32;++k)for(uint32_t n=0;n<64;++n)for(uint32_t j=0;j<64;++j)
  gm.put(d.node_stride+d.weights_offset+k*8192+(n*64+j)*2,
   j==n?(k==0?.5:k==31?.25:0):0);
 if(handoff){
  auto invalid=d;invalid.rows=1024;assert(!job.start(invalid) && input.calls==0);
  input.ready=false;bool refused=false;
  try{job.start(d);}catch(const std::logic_error&){refused=true;}assert(refused);
  input.ready=true;
 }
 assert(job.start(d));
 for(uint64_t t=1;t<200000 && !job.complete();++t){
  wcp.step(t);job.tick(t,t*1000);if(t%19==0)gm.drain();
 }
 assert(job.complete() && !job.status());
 size_t inputBytes=0;
 for(auto e:gm.reads)if(d.input_addr<=e.first && e.first<d.input_addr+256*2048*2)inputBytes+=e.second;
 assert(inputBytes==(handoff?0:64*2048*2));
 // The Wo input remains intact after output and weight scratch reuse.
 for(size_t b=0;b<expectedPacked.size();++b)
  assert(gm.bytes[DecoderMemoryLayout::decoder_input_gm+b]==expectedPacked[b]);
 std::vector<uint8_t> result;
 for(uint32_t r=192;r<256;++r){
  for(uint32_t n=0;n<64;++n){
   const double a=(int(((r-192)*23+n*7)%59)-29)*.015625;
   const double b=(int(((r-192)*23+(1984+n)*7)%59)-29)*.015625;
   assert(gm.get(d.output_addr+r*128+n*2)==vectorFp16Value(a*.5+b*.25));
  }
  for(uint32_t b=0;b<128;++b)result.push_back(gm.bytes[d.output_addr+r*128+b]);
 }
 return result;
}
int main(){for(bool window:{false,true})assert(run(false,window)==run(true,window));}
'''


class DecoderPvWoHandoffTest(unittest.TestCase):
    def test_real_wo_layout_bits_and_zero_input_dma_with_poisoned_hbm(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            for name in ('globalmemory/globalmemory.h','workercmdproc/workercmdproc.h'):
                stub=path/'sst/elements/golem'/name
                stub.parent.mkdir(parents=True,exist_ok=True)
                stub.write_text('#include "stub.h"\n')
            (path/'stub.h').write_text(timed_stub())
            (path/'test.cc').write_text(SOURCE)
            subprocess.run(['g++','-std=c++17','-O2','-I',str(path),'-I',str(ROOT/'src'),
                            str(path/'test.cc'),'-o',str(path/'test')],check=True)
            env=dict(os.environ,GOLEM_REUSE_WINDOW_K_TILES='32',GOLEM_DECODER_OUTPUT_PANEL_TILES='16',
                     GOLEM_REUSE_WINDOW_ARRAY_GROUPS='2',
                     GOLEM_DECODER_GM_STAGE_CHAIN='0',GOLEM_DECODER_GEMM_PREFETCH='0')
            result=subprocess.run([str(path/'test')],capture_output=True,env=env,timeout=90)
            self.assertEqual(result.returncode,0,result.stderr.decode())


if __name__=='__main__':
    unittest.main()
