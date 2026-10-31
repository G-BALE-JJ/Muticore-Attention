"""Bounded fused job preserves SiLU rounding, requests and output ACK order."""
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_decoder_tail import ROOT, STUB

FUSED_STUB = STUB.replace(
    'struct SFUAPI {bool issueVectorOp(const SFUVectorRequest&,SFUVectorResult*) {return false;}};',
    '''struct SFUAPI {
 std::vector<SFUVectorOp> ops;std::vector<size_t> sizes;
 bool rejectOnce=true;
 bool issueVectorOp(const SFUVectorRequest& r,SFUVectorResult* result) {
  if(rejectOnce) {rejectOnce=false;return false;}
  if(!evaluateSFUVectorOp(r,&result->values))return false;
  ops.push_back(r.op);sizes.push_back(r.input->size());
  result->completed=true;result->readyTick=0;return true;
 }
};''')

SOURCE = r'''
#include <sst/elements/golem/attention/decoderOpJob.h>
using namespace SST::Golem;
void run(uint32_t rows,uint32_t lane,uint32_t width) {
 GlobalMemoryAPI gm;WorkerCommandProcessorAPI wcp;SFUAPI sfu;DecoderOpJob job(&gm,&wcp,&sfu);
 DecoderOpDesc d{};d.magic=GOLEM_DECODER_OP_MAGIC;d.size_bytes=sizeof(d);d.op=7;d.rows=rows;
 d.input_dim=d.output_dim=width;d.lane=lane;d.manager=0;d.epoch=9;
 d.input_addr=0x40000000;d.input1_addr=0x50000000;d.output_addr=0x60000000;d.completion_addr=0x70000000;
 for(uint32_t r=0;r<rows;++r)for(uint32_t n=0;n<width;++n) {
  gm.put(d.input_addr+(uint64_t(r)*width+n)*2,(int(n%29)-14)*.125);
  gm.put(d.input1_addr+(uint64_t(r)*width+n)*2,(r%9+1)*.0625);
 }
 assert(job.start(d));for(uint64_t t=1;t<100000 && !job.complete();++t)job.tick(t,t*1000);
 assert(job.complete() && job.status()==0);
 uint32_t wantedRows=0;
 for(uint32_t group=lane*64;group<rows;group+=256)for(uint32_t r=group;r<std::min(group+64,rows);++r) {
  ++wantedRows;
  for(uint32_t n=0;n<width;++n) {
   const float x=gm.get(d.input_addr+(uint64_t(r)*width+n)*2);
   const float exponential=vectorFp16Value(std::exp(-std::abs(x)));
   const float denominator=vectorFp16Value(1.0f+exponential), reciprocal=vectorFp16Value(1.0f/denominator);
   const float numerator=vectorFp16Value(x<0 ? x*exponential : x);
   const float act=vectorFp16Value(numerator*reciprocal);
   assert(std::abs(gm.get(d.output_addr+(uint64_t(r)*width+n)*2)-vectorFp16Value(act*gm.get(d.input1_addr+(uint64_t(r)*width+n)*2))) < 1e-3f);
  }
 }
 assert(wcp.active.empty());size_t output=0;
 for(auto e:gm.writes) {assert(e.first==d.completion_addr || (d.output_addr<=e.first && e.first<d.output_addr+uint64_t(rows)*width*2));if(e.first!=d.completion_addr)output+=e.second;}
 assert(output==uint64_t(wantedRows)*width*2);
 assert(gm.writes.back().first==d.completion_addr && gm.bytes[d.completion_addr]==9);
 assert(!sfu.ops.empty());
 size_t elements=0;
 for(size_t i=0;i<sfu.ops.size();++i) {
  assert(sfu.ops[i]==SFUVectorOp::Silu || sfu.ops[i]==SFUVectorOp::Multiply || sfu.ops[i]==SFUVectorOp::SiluMultiply);
  assert(sfu.sizes[i]<=8192);
  if(sfu.ops[i]==SFUVectorOp::Silu || sfu.ops[i]==SFUVectorOp::SiluMultiply) elements+=sfu.sizes[i];
 }
 assert(elements==uint64_t(wantedRows)*width);
}
int main() {run(65,1,2048);run(2,0,8192);run(257,0,256);}
'''


class DecoderSiluFusionTest(unittest.TestCase):
    def test_real_job_rounding_bounded_chunks_SFU_retry_and_no_act_DMA(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('globalmemory/globalmemory.h', 'workercmdproc/workercmdproc.h'):
                path = root / 'sst/elements/golem' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('#include "stub.h"\n')
            (root / 'stub.h').write_text(FUSED_STUB)
            (root / 'test.cc').write_text(SOURCE)
            subprocess.run(['g++', '-std=c++17', '-O2', '-I', str(root), '-I', str(ROOT / 'src'),
                            str(root / 'test.cc'), '-o', str(root / 'test')], check=True)
            subprocess.run([str(root / 'test')], check=True, capture_output=True)


if __name__ == '__main__':
    unittest.main()
