"""RMSNorm gamma lifetime: one timed HBM load, same output and SFU work."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from test_decoder_panels import ASYNC_STUB
from test_decoder_tail import ROOT

STUB=ASYNC_STUB.replace(
    'struct SFUAPI {bool issueVectorOp(const SFUVectorRequest&,SFUVectorResult*) {return false;}};',
    '''struct SFUAPI {size_t requests=0;
 bool issueVectorOp(const SFUVectorRequest& r,SFUVectorResult* result){
  ++requests;if(!evaluateSFUVectorOp(r,&result->values))return false;
  result->completed=true;result->readyTick=0;return true;
 }};''')

SOURCE=r'''
#include <sst/elements/golem/attention/decoderOpJob.h>
using namespace SST::Golem;
std::vector<uint8_t> run(bool reuse){
 setenv("GOLEM_DECODER_GAMMA_REUSE",reuse?"1":"0",1);
 GlobalMemoryAPI gm;WorkerCommandProcessorAPI wcp;SFUAPI sfu;DecoderOpJob job(&gm,&wcp,&sfu);
 DecoderOpDesc d{};d.magic=GOLEM_DECODER_OP_MAGIC;d.size_bytes=sizeof(d);d.op=3;d.stage=3;d.epoch=1;
 d.rows=64;d.input_dim=d.output_dim=2048;d.input_addr=0x40000000;d.input1_addr=0x50000000;
 d.output_addr=0x60000000;d.completion_addr=0x70000000;
 for(size_t r=0;r<64;++r)for(size_t j=0;j<2048;++j)gm.put(d.input_addr+(r*2048+j)*2,(int((r+j)%17)-8)*.0625);
 for(size_t j=0;j<2048;++j)gm.put(d.input1_addr+j*2,.5+(j%3)*.125);
 assert(job.start(d));
 for(uint64_t t=1;t<100000 && !job.complete();++t){job.tick(t,t*1000);if(t%3==0)gm.drain();}
 assert(job.complete() && !job.status() && gm.pending.empty() && gm.delayedReads.empty());
 size_t gammaReads=0;for(auto entry:gm.reads)if(entry.first==d.input1_addr){assert(entry.second==4096);++gammaReads;}
 assert(gammaReads==(reuse?1:16));assert(sfu.requests==16);
 std::vector<uint8_t> output(64*2048*2);for(size_t i=0;i<output.size();++i)output[i]=gm.bytes[d.output_addr+i];return output;
}
int main(){assert(run(false)==run(true));}
'''


class DecoderGammaReuseTest(unittest.TestCase):
    def test_real_job_gamma_load_count_and_identical_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('globalmemory/globalmemory.h','workercmdproc/workercmdproc.h'):
                path=root/'sst/elements/golem'/name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text('#include "stub.h"\n')
            (root/'stub.h').write_text(STUB);(root/'test.cc').write_text(SOURCE)
            subprocess.run(['g++','-std=c++17','-O2','-I',str(root),'-I',str(ROOT/'src'),
                            str(root/'test.cc'),'-o',str(root/'test')],check=True)
            subprocess.run([str(root/'test')],check=True,capture_output=True,timeout=30)


if __name__=='__main__':unittest.main()
