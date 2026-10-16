"""Software layouts and K-major input reuse with real job ownership/retry logic."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from test_decoder_groups import timed_stub
from test_decoder_down_fusion import SOURCE
from test_decoder_tail import ROOT


class DecoderGmFlowTest(unittest.TestCase):
    def test_layout_reuse_tail_rows_panels_and_async_ownership(self):
        source = SOURCE.replace('bool residual=false) {', '''bool residual=false,unsigned layout=0,bool reuse=false,unsigned groups=2) {
 setenv("GOLEM_DECODER_FFN_LAYOUT",std::to_string(layout).c_str(),1);
 setenv("GOLEM_DECODER_DOWN_INPUT_REUSE",reuse?"1":"0",1);
 setenv("GOLEM_DECODER_ARRAY_GROUPS",std::to_string(groups).c_str(),1);''')
        source = source.replace('sfu.step(t*1000);job.tick(t,t*1000);',
                                'sfu.step(t*1000);wcp.step(t);job.tick(t,t*1000);')
        source = source.replace('assert(!wcp.partialOwned);job.retire();sfu.now=0;',
                                'assert(!wcp.partialOwned && wcp.events.empty());job.retire();sfu.now=0;wcp.bufferReady=0;')
        source = source[:source.index('int main()')] + r'''
int main() {
 setenv("GOLEM_DECODER_PREFETCH_SLOTS","4",1);
 for(auto shape:std::vector<std::pair<uint32_t,uint32_t>>{{64,0},{33,0},{65,1},{257,0},{1,0},{65,2}}) {
  for(bool prefetch:{false,true}) {
   auto expected=run(shape.first,shape.second,17,8,prefetch,true,true);
   for(unsigned layout:{0u,1u,2u})for(bool reuse:{false,true}) {
    assert(run(shape.first,shape.second,17,8,prefetch,true,true,layout,reuse)==expected);
    assert(run(shape.first,shape.second,17,3,prefetch,true,true,layout,reuse,1)==expected);
   }
  }
 }
}
'''
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('globalmemory/globalmemory.h','workercmdproc/workercmdproc.h'):
                path=root/'sst/elements/golem'/name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text('#include "stub.h"\n')
            # Delay and reverse local write completions, as well as rejecting
            # requests. Exercise the bounded strided writer with real callbacks.
            stub=timed_stub()
            old='for(size_t i=0;i<data.size();++i)bytes[addr+i]=data[i];cb(true,tag);return true;'
            new='delayedReads.push_back([this,addr,data,tag,cb]{for(size_t i=0;i<data.size();++i)bytes[addr+i]=data[i];cb(true,tag);});return true;'
            assert old in stub
            stub=stub.replace(old,new)
            (root/'stub.h').write_text(stub);(root/'test.cc').write_text(source)
            subprocess.run(['g++','-std=c++17','-O2','-I',str(root),'-I',str(ROOT/'src'),str(root/'test.cc'),'-o',str(root/'test')],check=True)
            result=subprocess.run([str(root/'test')],capture_output=True,timeout=240)
            self.assertEqual(result.returncode,0,result.stderr.decode())


if __name__=='__main__':unittest.main()
