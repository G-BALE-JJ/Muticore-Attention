"""Adjacent LM chunks reuse timed GM input; changed source must miss."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from test_decoder_groups import timed_stub
from test_decoder_tail import ROOT

SOURCE=r'''
#include <sst/elements/golem/attention/decoderOpJob.h>
using namespace SST::Golem;
std::vector<uint8_t> run(bool cache){
 setenv("GOLEM_DECODER_LM_INPUT_RESIDENCY",cache?"1":"0",1);
 GlobalMemoryAPI gm;WorkerCommandProcessorAPI wcp;SFUAPI sfu;DecoderOpJob job(&gm,&wcp,&sfu);
 constexpr uint64_t input=0x40000000,output=0x50000000,stride=0x08000000,weight=0x01000000;
 for(uint32_t i=0;i<128;++i){gm.put(input+i*2,.125);gm.put(input+4096+i*2,.25);}
 for(uint32_t chunk=0;chunk<4;++chunk)for(uint32_t k=0;k<2;++k)for(uint32_t i=0;i<64;++i)
  gm.put(stride+weight+uint64_t(chunk*4*2+k)*8192+(i*64+i)*2,.5);
 std::vector<uint8_t> result;
 for(uint32_t chunk=0;chunk<4;++chunk){
  DecoderOpDesc d{};d.magic=GOLEM_DECODER_OP_MAGIC;d.size_bytes=sizeof(d);d.op=1;d.stage=11;d.rows=1;
  d.input_dim=128;d.output_dim=1024;d.node_stride=stride;d.weights_offset=weight;
  d.input_addr=input+(chunk==3?4096:0);d.output_addr=output+chunk*2048;d.epoch=4+chunk;d.output_tile_begin=chunk*16;
  assert(job.start(d));
  for(uint64_t t=1;t<100000 && !job.complete();++t){sfu.step(t*1000);wcp.step(t);job.tick(t,t*1000);if(t%13==0)gm.drain();}
  assert(job.complete() && !job.status() && gm.pending.empty() && gm.delayedReads.empty() && wcp.events.empty());
  job.retire();sfu.now=0;
  for(uint32_t i=0;i<128;++i)result.push_back(gm.bytes[d.output_addr+i]);
 }
 uint32_t inputReads=0;
 for(auto e:gm.reads)if(e.first==input || e.first==input+4096){assert(e.second==256);++inputReads;}
 assert(inputReads==(cache?2u:4u));return result;
}
int main(){assert(run(false)==run(true));}
'''


class LmInputResidencyTest(unittest.TestCase):
    def test_contiguous_chunks_and_changed_address(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('globalmemory/globalmemory.h','workercmdproc/workercmdproc.h'):
                p=root/'sst/elements/golem'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('#include "stub.h"\n')
            (root/'stub.h').write_text(timed_stub());(root/'test.cc').write_text(SOURCE)
            subprocess.run(['g++','-std=c++17','-O2','-I',str(root),'-I',str(ROOT/'src'),str(root/'test.cc'),'-o',str(root/'test')],check=True)
            env=dict(os.environ,GOLEM_ALL_GEMM_REUSE_WINDOW='1',GOLEM_REUSE_WINDOW_ARRAY_GROUPS='1',
                     GOLEM_REUSE_WINDOW_A_PREFETCH='1',GOLEM_REUSE_WINDOW_OUTPUT_OVERLAP='1')
            p=subprocess.run([str(root/'test')],env=env,capture_output=True,timeout=60)
            self.assertEqual(p.returncode,0,p.stderr.decode())


if __name__=='__main__':unittest.main()
