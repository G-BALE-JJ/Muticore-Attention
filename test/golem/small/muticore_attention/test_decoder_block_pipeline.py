"""Concurrent production windows: C leases, bounded slots and delayed ACKs."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from test_decoder_groups import timed_stub
from test_decoder_down_fusion import SOURCE
from test_decoder_tail import ROOT


class DecoderBlockPipelineTest(unittest.TestCase):
    def test_pipeline_matches_serial_with_delayed_reordered_gm_and_tails(self):
        source = SOURCE.replace('bool residual=false) {', '''bool residual=false,bool pipeline=false) {
 setenv("GOLEM_ALL_GEMM_REUSE_WINDOW","1",1);
 setenv("GOLEM_DECODER_FFN_LAYOUT","2",1);
 setenv("GOLEM_DECODER_ARRAY_GROUPS","1",1);
 setenv("GOLEM_DECODER_FFN_BLOCK_PIPELINE",pipeline?"1":"0",1);''')
        source = source.replace('sfu.step(t*1000);job.tick(t,t*1000);',
                                'sfu.step(t*1000);wcp.step(t);job.tick(t,t*1000);')
        source = source.replace('assert(!wcp.partialOwned);job.retire();sfu.now=0;',
                                'assert(!wcp.partialOwned && wcp.events.empty());job.retire();sfu.now=0;wcp.bufferReady=0;')
        source = source[:source.index('int main()')] + r'''
int main(){
 for(auto chunk:{"4","16","32"}){
  setenv("GOLEM_REUSE_WINDOW_K_TILES",chunk,1);
  for(auto shape:std::vector<std::pair<uint32_t,uint32_t>>{{64,0},{33,0},{65,1},{257,0},{1,0}})
   for(auto panel:{3u,8u,16u}){
    const auto expected=run(shape.first,shape.second,17,panel,true,true,true,false);
    assert(run(shape.first,shape.second,17,panel,true,true,true,true)==expected);
   }
 }
}
'''
        stub = timed_stub()
        old = 'for(size_t i=0;i<data.size();++i)bytes[addr+i]=data[i];cb(true,tag);return true;'
        new = 'delayedReads.push_back([this,addr,data,tag,cb]{for(size_t i=0;i<data.size();++i)bytes[addr+i]=data[i];cb(true,tag);});return true;'
        self.assertIn(old, stub)
        stub = stub.replace(old, new)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('globalmemory/globalmemory.h', 'workercmdproc/workercmdproc.h'):
                path = root/'sst/elements/golem'/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('#include "stub.h"\n')
            (root/'stub.h').write_text(stub)
            (root/'test.cc').write_text(source)
            subprocess.run(['g++', '-std=c++17', '-O2', '-I', str(root), '-I', str(ROOT/'src'),
                            str(root/'test.cc'), '-o', str(root/'test')], check=True)
            env = dict(os.environ, GOLEM_REUSE_WINDOW_ARRAY_GROUPS='1', GOLEM_REUSE_WINDOW_PREFETCH_SLOTS='4',
                       GOLEM_REUSE_WINDOW_EARLY_A='1', GOLEM_REUSE_WINDOW_OUTPUT_OVERLAP='1')
            result = subprocess.run([str(root/'test')], capture_output=True, env=env, timeout=240)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            for line in result.stdout.decode().splitlines():
                if '[DECODER_BLOCK_PIPELINE]' in line:
                    self.assertIn('owner_at_finish=0 pending_at_finish=0', line)


if __name__ == '__main__':
    unittest.main()
