"""Production output panels under delayed, reordered DMA completion."""
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_decoder_tail import ROOT, STUB

ASYNC_STUB = STUB.replace(
    'void dma_write_from_globalmem_to_host(uint64_t src,uint64_t dst,size_t size,std::function<void(bool)> cb) {\n'
    '  writes.push_back({dst,size});for(size_t i=0;i<size;++i)bytes[dst+i]=bytes[src+i];cb(true);\n }',
    '''struct Pending {uint64_t src,dst;size_t size;std::function<void(bool)> cb;};
 std::vector<Pending> pending; size_t highWater=0;
 void dma_write_from_globalmem_to_host(uint64_t src,uint64_t dst,size_t size,std::function<void(bool)> cb) {
  writes.push_back({dst,size});pending.push_back({src,dst,size,cb});highWater=std::max(highWater,pending.size());
 }
 void drain() {
  if(pending.empty())return;
  // Read the GM source at completion, exposing premature panel reuse.
  auto p=pending.back();pending.pop_back();
  for(size_t i=0;i<p.size;++i)bytes[p.dst+i]=bytes[p.src+i];p.cb(true);
 }''')
ASYNC_STUB = ASYNC_STUB.replace(
    'void dma_read_from_host_to_globalmem(uint64_t src,size_t size,uint64_t dst,std::function<void(bool)> cb) {\n'
    '  reads.push_back({src,size});for(size_t i=0;i<size;++i)bytes[dst+i]=bytes[src+i];cb(true);\n }',
    '''std::vector<std::function<void()>> delayedReads;
 void dma_read_from_host_to_globalmem(uint64_t src,size_t size,uint64_t dst,std::function<void(bool)> cb) {
  reads.push_back({src,size});
  delayedReads.push_back([this,src,size,dst,cb] {
   for(size_t i=0;i<size;++i)bytes[dst+i]=bytes[src+i];cb(true);
  });
 }''').replace(
    'void drain() {\n  if(pending.empty())return;',
    '''void drain() {
  if(!delayedReads.empty()) {auto cb=delayedReads.back();delayedReads.pop_back();cb();}
  if(pending.empty())return;''').replace(
    'bool localReadAsync(uint64_t addr,size_t size,LocalMemoryClient,uint64_t tag,',
    'size_t localReadAttempts=0;\n bool localReadAsync(uint64_t addr,size_t size,LocalMemoryClient,uint64_t tag,').replace(
    'std::vector<uint8_t> data(size);for(size_t i=0;i<size;++i)data[i]=bytes[addr+i];',
    'assert(addr+size<=getSize());if(++localReadAttempts%7==0)return false;\n  std::vector<uint8_t> data(size);for(size_t i=0;i<size;++i)data[i]=bytes[addr+i];')

SOURCE = r'''
#include <sst/elements/golem/attention/decoderOpJob.h>
#include <cassert>
#include <cstdlib>
using namespace SST::Golem;
std::vector<uint8_t> run(uint32_t rows,uint32_t lane,uint32_t tiles,uint32_t panel,uint32_t window,bool prefetch=false) {
 setenv("GOLEM_DECODER_OUTPUT_PANEL_TILES",std::to_string(panel).c_str(),1);
 setenv("GOLEM_DECODER_OUTPUT_WRITE_WINDOW",std::to_string(window).c_str(),1);
 setenv("GOLEM_DECODER_GEMM_PREFETCH",prefetch?"1":"0",1);
 GlobalMemoryAPI gm;WorkerCommandProcessorAPI wcp;SFUAPI sfu;DecoderOpJob job(&gm,&wcp,&sfu);
 DecoderOpDesc d{};d.magic=GOLEM_DECODER_OP_MAGIC;d.size_bytes=sizeof(d);d.op=1;d.rows=rows;
 d.input_dim=128;d.output_dim=tiles*64;d.lane=lane;d.manager=0;d.epoch=7;
 d.input_addr=0x40000000;d.output_addr=0x60000000;d.node_stride=0x08000000;d.weights_offset=0x01000000;
 d.completion_addr=0x70000000;
 for(uint32_t r=0;r<rows;++r)for(uint32_t k=0;k<128;++k)gm.put(d.input_addr+(r*128+k)*2,(r%13)*.125+(k%7)*.0625);
 for(uint32_t tile=0;tile<tiles;++tile)for(uint32_t k=0;k<2;++k)for(uint32_t n=0;n<64;++n)for(uint32_t j=0;j<64;++j)
  gm.put((1+tile%4)*d.node_stride+d.weights_offset+(tile/4*2+k)*8192+(n*64+j)*2,n==j?(tile+1)*.0625:0);
 assert(job.start(d));
 for(uint64_t t=1;t<1000000 && !job.complete();++t) {
  job.tick(t,t*1000);
  if(!gm.pending.empty())assert(!job.complete());
  if(t%19==0)gm.drain();
 }
 assert(job.complete() && job.status()==0 && gm.pending.empty() && gm.delayedReads.empty());
 assert(gm.writes.back().first==d.completion_addr && gm.bytes[d.completion_addr]==7);
 assert(gm.highWater<=window);
 uint32_t wantedRows=0;
 for(uint32_t r=lane*64;r<rows;r+=256)wantedRows+=std::min(64u,rows-r);
 const uint32_t ownedTiles=rows==1?(lane*4>=tiles?0:(tiles-1-lane*4)/16+1):tiles;
 if(rows==1)wantedRows=ownedTiles?1:0;
 const uint32_t writesPerRow=panel && rows!=1?(tiles+panel-1)/panel:ownedTiles;
 assert(gm.writes.size()==size_t(wantedRows)*writesPerRow+1);
 std::vector<uint8_t> result;
 for(uint32_t group=lane*64;group<rows;group+=256)for(uint32_t r=group;r<std::min(group+64,rows);++r)
  for(uint32_t n=0;n<d.output_dim;++n) {
   if(rows==1 && n/64%16!=lane*4)continue;
   const double dot=vectorFp16Value(gm.get(d.input_addr+(r*128+n%64)*2)*((n/64+1)*.0625));
   const double dot1=vectorFp16Value(gm.get(d.input_addr+(r*128+64+n%64)*2)*((n/64+1)*.0625));
   assert(gm.get(d.output_addr+(uint64_t(r)*d.output_dim+n)*2)==vectorFp16Value(dot+dot1));
   result.push_back(gm.bytes[d.output_addr+(uint64_t(r)*d.output_dim+n)*2]);
   result.push_back(gm.bytes[d.output_addr+(uint64_t(r)*d.output_dim+n)*2+1]);
  }
 return result;
}
int main() {
 for(auto shape:std::vector<std::pair<uint32_t,uint32_t>>{{65,0},{65,1},{257,0},{1,0}}) {
  auto baseline=run(shape.first,shape.second,17,0,1);
  assert(run(shape.first,shape.second,17,16,8)==baseline);
  assert(run(shape.first,shape.second,17,8,1)==baseline);
  assert(run(shape.first,shape.second,17,16,8,true)==baseline);
 }
}
'''


class DecoderPanelsTest(unittest.TestCase):
    def test_real_job_panel_tail_backpressure_reordered_ACKs_and_M1(self):
        self.assertIn('struct Pending', ASYNC_STUB)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('globalmemory/globalmemory.h', 'workercmdproc/workercmdproc.h'):
                path = root / 'sst/elements/golem' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('#include "stub.h"\n')
            (root / 'stub.h').write_text(ASYNC_STUB)
            (root / 'test.cc').write_text(SOURCE)
            subprocess.run(['g++', '-std=c++17', '-O2', '-I', str(root), '-I', str(ROOT / 'src'),
                            str(root / 'test.cc'), '-o', str(root / 'test')], check=True)
            subprocess.run([str(root / 'test')], check=True, capture_output=True)


if __name__ == '__main__':
    unittest.main()
