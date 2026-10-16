"""Actual fused job: K-order, timed partial restores, panel ownership and ACKs."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from test_decoder_ffn_panels import STUB
from test_decoder_tail import ROOT

SOURCE=r'''
#include <sst/elements/golem/attention/decoderOpJob.h>
using namespace SST::Golem;
std::vector<uint8_t> run(uint32_t rows,uint32_t lane,uint32_t tiles,uint32_t panel,bool prefetch,bool fused,bool residual=false) {
 setenv("GOLEM_DECODER_OUTPUT_PANEL_TILES","16",1);
 setenv("GOLEM_DECODER_OUTPUT_WRITE_WINDOW","8",1);
 setenv("GOLEM_DECODER_FFN_PANEL_TILES",std::to_string(panel).c_str(),1);
 setenv("GOLEM_DECODER_GEMM_PREFETCH",prefetch?"1":"0",1);
 GlobalMemoryAPI gm;WorkerCommandProcessorAPI wcp;SFUAPI sfu;DecoderOpJob job(&gm,&wcp,&sfu);
 constexpr uint64_t input=0x40000000,gated=0x60000000,down=0x70000000,flag=0x78000000;
 constexpr uint32_t residualOffset=0x06000000;
 constexpr uint64_t gateWeight=0x01000000,upWeight=0x01200000,downWeight=0x01400000,stride=0x08000000;
 for(uint32_t r=0;r<rows;++r)for(uint32_t k=0;k<128;++k)
  gm.put(input+(uint64_t(r)*128+k)*2,(int(r%11)-5)*.125+(int(k%17)-8)*.0625);
 for(uint32_t r=0;r<rows;++r)for(uint32_t n=0;n<128;++n)
  gm.put(stride+residualOffset+(uint64_t(r)*128+n)*2,(int(r%13)-6)*.03125+(int(n%19)-9)*.015625);
 for(uint32_t tile=0;tile<tiles;++tile)for(uint32_t k=0;k<2;++k)for(uint32_t n=0;n<64;++n)for(uint32_t j=0;j<64;++j) {
  const uint64_t address=(1+tile%4)*stride+(tile/4*2+k)*8192+(n*64+j)*2;
  gm.put(address+gateWeight,n==j?(int(tile%7)-3)*.125:0);
  gm.put(address+upWeight,n==j?(k+1)*.0625:0);
 }
 for(uint32_t n=0;n<2;++n)for(uint32_t k=0;k<tiles;++k)for(uint32_t o=0;o<64;++o)for(uint32_t j=0;j<64;++j)
  gm.put((1+n%4)*stride+downWeight+(n/4*tiles+k)*8192+(o*64+j)*2,
   (o==j || j==(o+7)%64)?(int(k%5)-2)*.03125*(n+1):0);
 auto execute=[&](DecoderOpDesc d) {
  assert(job.start(d));
  for(uint64_t t=1;t<1000000 && !job.complete();++t) {
   sfu.step(t*1000);job.tick(t,t*1000);if(t%19==0)gm.drain();
  }
  assert(job.complete() && !job.status() && gm.pending.empty() && gm.delayedReads.empty() && !sfu.pending);
  assert(!wcp.partialOwned);job.retire();sfu.now=0;
 };
 DecoderOpDesc d{};d.magic=GOLEM_DECODER_OP_MAGIC;d.size_bytes=sizeof(d);d.op=fused?9:8;d.stage=4;
 d.rows=rows;d.lane=lane;d.input_dim=128;d.output_dim=tiles*64;d.epoch=7;d.node_stride=stride;
 d.input_addr=input;d.input1_addr=upWeight;d.weights_offset=gateWeight;d.output_addr=fused?down:gated;
 if(fused){d.down_weights_offset=downWeight;d.completion_addr=flag;if(residual)d.residual_offset=residualOffset;}
 execute(d);
 if(!fused) {
  d.op=1;d.stage=8;d.head_dim=0;d.input_dim=tiles*64;d.output_dim=128;
  d.input_addr=gated;d.input1_addr=0;d.weights_offset=downWeight;d.output_addr=down;d.completion_addr=flag;
  for(uint32_t owner=0;owner<(rows==1?16u:1u);++owner) {
   if(rows==1){d.manager=owner%4;d.lane=owner/4;}execute(d);
  }
 }
 if(residual && !fused) {
  d.op=2;d.stage=9;d.input_dim=d.output_dim=128;d.lane=lane;d.manager=0;
  d.input_addr=down;d.input1_addr=stride+residualOffset;d.output_addr=down;d.completion_addr=flag;
  execute(d);
 }
 assert(gm.writes.back().first==flag && gm.bytes[flag]==7 && gm.highWater<=8);
 std::vector<uint8_t> result;uint32_t wanted=0;
 for(uint32_t group=lane*64;group<rows;group+=256)for(uint32_t r=group;r<std::min(group+64,rows);++r) {
  ++wanted;
  for(uint32_t n=0;n<128;++n)for(uint32_t b=0;b<2;++b)result.push_back(gm.bytes[down+(uint64_t(r)*128+n)*2+b]);
 }
 if(fused) {
  size_t inputBytes=0,outputBytes=0,weightBytes=0;
  for(auto e:gm.reads) {
   assert(e.first<gated || e.first>=down);
   if(input<=e.first && e.first<input+uint64_t(rows)*256)inputBytes+=e.second;
   else if(!(stride+residualOffset<=e.first && e.first<stride+residualOffset+uint64_t(rows)*256))weightBytes+=e.second;
  }
  for(auto e:gm.writes) {
   assert(e.first==flag || (down<=e.first && e.first<down+uint64_t(rows)*256));
   if(e.first!=flag)outputBytes+=e.second;
  }
  assert(inputBytes==uint64_t(wanted)*256 && outputBytes==uint64_t(wanted)*256);
  const size_t groups=(wanted+63)/64;
  assert(weightBytes==groups*(2*tiles*2+2*tiles)*8192);
 }
 return result;
}
int main() {
 for (const char* slots:{"2","4","8"}) {
 setenv("GOLEM_DECODER_PREFETCH_SLOTS",slots,1);
 for(auto shape:std::vector<std::pair<uint32_t,uint32_t>>{{65,1},{257,0},{1,0},{65,2}}) {
  const auto expected=run(shape.first,shape.second,17,8,true,false);
  assert(run(shape.first,shape.second,17,8,true,true)==expected);
  assert(run(shape.first,shape.second,17,3,true,true)==expected);
  assert(run(shape.first,shape.second,17,16,false,true)==expected);
  const auto added=run(shape.first,shape.second,17,8,true,false,true);
  assert(run(shape.first,shape.second,17,8,true,true,true)==added);
  assert(run(shape.first,shape.second,17,3,true,true,true)==added);
  assert(run(shape.first,shape.second,17,16,false,true,true)==added);
 }
 }
 // One panel and multiple down output DMA panels (32 tiles at H=2048)
 // are additionally covered by the real SST width tests.
}
'''


class DecoderDownFusionTest(unittest.TestCase):
    def test_production_fused_down_preserves_bits_K_order_tails_and_delayed_ACKs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('globalmemory/globalmemory.h','workercmdproc/workercmdproc.h'):
                path=root/'sst/elements/golem'/name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text('#include "stub.h"\n')
            (root/'stub.h').write_text(STUB);(root/'test.cc').write_text(SOURCE)
            subprocess.run(['g++','-std=c++17','-O2','-I',str(root),'-I',str(ROOT/'src'),str(root/'test.cc'),'-o',str(root/'test')],check=True)
            completed=subprocess.run([str(root/'test')],capture_output=True,timeout=120)
            self.assertEqual(completed.returncode,0,completed.stderr.decode())


if __name__=='__main__':unittest.main()
