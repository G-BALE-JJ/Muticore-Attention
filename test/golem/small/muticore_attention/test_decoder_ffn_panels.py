"""Real fused producer/consumer under DMA delays and independent GM retries."""
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_decoder_panels import ASYNC_STUB
from test_decoder_tail import ROOT

STUB = ASYNC_STUB.replace(
    'struct SFUAPI {bool issueVectorOp(const SFUVectorRequest&,SFUVectorResult*) {return false;}};',
    '''struct SFUAPI {
 uint64_t now=0,ready=0;size_t attempts=0;SFUVectorResult* pending=nullptr;
 std::vector<double> values;std::vector<size_t> sizes;
 bool issueVectorOp(const SFUVectorRequest& r,SFUVectorResult* result) {
  if(pending || ++attempts%7==0)return false;
  assert(r.input->size()<=8192);
  if(!evaluateSFUVectorOp(r,&values))return false;
  sizes.push_back(r.input->size());*result=SFUVectorResult{};
  ready=now+73000;pending=result;return true;
 }
 void step(uint64_t tick) {
  now=tick;if(pending && tick>=ready) {
   pending->values=std::move(values);pending->completed=true;pending->readyTick=tick;pending=nullptr;
  }
 }
};''').replace(
    'bool localWriteAsync(uint64_t addr,',
    'size_t localWriteAttempts=0;\n bool localWriteAsync(uint64_t addr,').replace(
    'for(size_t i=0;i<data.size();++i)bytes[addr+i]=data[i];cb(true,tag);return true;',
    'assert(addr+data.size()<=getSize());if(++localWriteAttempts%11==0)return false;\n  for(size_t i=0;i<data.size();++i)bytes[addr+i]=data[i];cb(true,tag);return true;')

SOURCE = r'''
#include <sst/elements/golem/attention/decoderOpJob.h>
using namespace SST::Golem;
std::vector<uint8_t> run(uint32_t rows,uint32_t lane,uint32_t tiles,uint32_t panel,bool prefetch,bool fused) {
 setenv("GOLEM_DECODER_OUTPUT_PANEL_TILES","16",1);
 setenv("GOLEM_DECODER_OUTPUT_WRITE_WINDOW","8",1);
 setenv("GOLEM_DECODER_FFN_PANEL_TILES",std::to_string(panel).c_str(),1);
 setenv("GOLEM_DECODER_GEMM_PREFETCH",prefetch?"1":"0",1);
 GlobalMemoryAPI gm;WorkerCommandProcessorAPI wcp;SFUAPI sfu;DecoderOpJob job(&gm,&wcp,&sfu);
 constexpr uint64_t input=0x40000000,gate=0x60000000,up=0x65000000,gated=0x70000000,flag=0x78000000;
 constexpr uint64_t gateWeight=0x01000000,upWeight=0x01200000,stride=0x08000000;
 for(uint32_t r=0;r<rows;++r)for(uint32_t k=0;k<128;++k)
  gm.put(input+(uint64_t(r)*128+k)*2,(int(r%11)-5)*.125+(int(k%17)-8)*.0625);
 for(uint32_t tile=0;tile<tiles;++tile)for(uint32_t k=0;k<2;++k)for(uint32_t n=0;n<64;++n)for(uint32_t j=0;j<64;++j) {
  const uint64_t address=(1+tile%4)*stride+(tile/4*2+k)*8192+(n*64+j)*2;
  gm.put(address+gateWeight,n==j?(int(tile%7)-3)*.125:0);
  gm.put(address+upWeight,n==j?(k+1)*.0625:0);
 }
 uint32_t ownerLane=lane,ownerManager=0;
 auto execute=[&](uint32_t op,uint32_t k,uint64_t in,uint64_t in1,uint64_t out,uint64_t weights,bool last) {
  DecoderOpDesc d{};d.magic=GOLEM_DECODER_OP_MAGIC;d.size_bytes=sizeof(d);d.op=op;d.stage=4;d.rows=rows;
  d.input_dim=k;d.output_dim=tiles*64;d.lane=op==1?ownerLane:lane;d.manager=op==1?ownerManager:0;d.epoch=5;
  d.input_addr=in;d.input1_addr=in1;d.output_addr=out;d.weights_offset=weights;d.node_stride=stride;
  d.completion_addr=last?flag:0;
  assert(job.start(d));
  for(uint64_t t=1;t<1000000 && !job.complete();++t) {
   sfu.step(t*1000);job.tick(t,t*1000);if(t%19==0)gm.drain();
  }
  assert(job.complete() && !job.status() && gm.pending.empty() && gm.delayedReads.empty() && !sfu.pending);
  job.retire();sfu.now=0;
 };
 if(fused)execute(8,128,input,upWeight,gated,gateWeight,true);
 else {
  // M=1 Linear distributes columns over 16 workers. Reconstruct the whole
  // baseline tensor, rather than comparing a single column owner to full FFN.
  for(uint32_t owner=0;owner<(rows==1?16u:1u);++owner) {
   if(rows==1) {ownerLane=owner/4;ownerManager=owner%4;}
   execute(1,128,input,0,gate,gateWeight,false);
   execute(1,128,input,0,up,upWeight,false);
  }
  execute(7,tiles*64,gate,up,gated,0,true);
 }
 assert(gm.writes.back().first==flag && gm.bytes[flag]==5);
 assert(gm.highWater<=8);std::vector<uint8_t> result;uint32_t wantedRows=0;
 for(uint32_t group=lane*64;group<rows;group+=256)for(uint32_t r=group;r<std::min(group+64,rows);++r) {
  ++wantedRows;
  for(uint32_t n=0;n<tiles*64;++n) {
   result.push_back(gm.bytes[gated+(uint64_t(r)*tiles*64+n)*2]);
   result.push_back(gm.bytes[gated+(uint64_t(r)*tiles*64+n)*2+1]);
  }
 }
 if(fused) {
  size_t outputBytes=0,inputBytes=0;
  for(auto e:gm.writes) {
   assert(e.first==flag || (e.first>=gated && e.first<gated+uint64_t(rows)*tiles*128));
   if(e.first!=flag)outputBytes+=e.second;
  }
  for(auto e:gm.reads) {
   assert(e.first<gate || e.first>=gated);
   if(e.first>=input && e.first<input+uint64_t(rows)*256)inputBytes+=e.second;
  }
  assert(outputBytes==uint64_t(wantedRows)*tiles*128 && inputBytes==uint64_t(wantedRows)*256);
  assert(gm.writes.size()==uint64_t(wantedRows)*((tiles+panel-1)/panel)+1);
 }
 return result;
}
int main() {
 for(auto shape:std::vector<std::pair<uint32_t,uint32_t>>{{65,1},{257,0},{1,0},{65,2}}) {
  auto expected=run(shape.first,shape.second,17,16,true,false);
  assert(run(shape.first,shape.second,17,16,true,true)==expected);
  assert(run(shape.first,shape.second,17,3,true,true)==expected);
  assert(run(shape.first,shape.second,17,3,false,true)==expected);
 }
}
'''


class DecoderFfnPanelsTest(unittest.TestCase):
    def test_production_paired_panels_tail_prefetch_transitions_and_ACK_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('globalmemory/globalmemory.h', 'workercmdproc/workercmdproc.h'):
                path = root / 'sst/elements/golem' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('#include "stub.h"\n')
            (root / 'stub.h').write_text(STUB)
            (root / 'test.cc').write_text(SOURCE)
            subprocess.run(['g++', '-std=c++17', '-O2', '-I', str(root), '-I', str(ROOT / 'src'),
                            str(root / 'test.cc'), '-o', str(root / 'test')], check=True)
            completed = subprocess.run([str(root / 'test')], capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr.decode())


if __name__ == '__main__':
    unittest.main()
