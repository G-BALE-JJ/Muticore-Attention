"""Timed GM handoff: correct packing, matching metadata, and safe fallback."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from test_decoder_groups import timed_stub
from test_decoder_down_fusion import SOURCE
from test_decoder_tail import ROOT


class DecoderNormHandoffTest(unittest.TestCase):
    def test_real_norm_then_ffn_preserves_bits_and_avoids_only_matching_dma(self):
        source = SOURCE.replace('bool residual=false) {', '''bool residual=false,uint32_t mode=0) {
 setenv("GOLEM_ALL_GEMM_REUSE_WINDOW","1",1);
 setenv("GOLEM_DECODER_FFN_LAYOUT","2",1);
 setenv("GOLEM_DECODER_NORM_FFN_HANDOFF",mode?"1":"0",1);''')
        source = source.replace('sfu.step(t*1000);job.tick(t,t*1000);',
                                'sfu.step(t*1000);wcp.step(t);job.tick(t,t*1000);')
        source = source.replace('assert(!wcp.partialOwned);job.retire();sfu.now=0;',
                                'assert(!wcp.partialOwned && wcp.events.empty());job.retire();sfu.now=0;wcp.bufferReady=0;')
        needle = ' DecoderOpDesc d{};d.magic=GOLEM_DECODER_OP_MAGIC;'
        source = source.replace(needle, r'''
 constexpr uint64_t rawInput=0x3e000000,gamma=0x3f000000;
 for(size_t i=0;i<size_t(rows)*256;++i)gm.bytes[rawInput+i]=gm.bytes[input+i];
 for(size_t i=0;i<128;++i)gm.put(gamma+i*2,.5+(i%3)*.125);
 DecoderOpDesc norm{};norm.magic=GOLEM_DECODER_OP_MAGIC;norm.size_bytes=sizeof(norm);
 norm.op=3;norm.stage=3;norm.rows=rows;norm.lane=lane;norm.input_dim=norm.output_dim=128;
 norm.epoch=mode==2?6:7;norm.input_addr=rawInput;norm.input1_addr=gamma;norm.output_addr=input;
 execute(norm);
''' + needle, 1)
        begin = source.index(' if(fused) {\n  size_t inputBytes=0,outputBytes=0,weightBytes=0;')
        end = source.index(' return result;', begin)
        source = source[:begin] + r'''
 size_t inputBytes=0;
 for(auto e:gm.reads)if(input<=e.first && e.first<input+uint64_t(rows)*256)inputBytes+=e.second;
 const bool hit=mode==1 && rows<=256;
 assert(inputBytes==(hit?0:uint64_t(wanted)*256));
 if(hit)for(uint32_t r=0;r<wanted;++r)for(uint32_t k=0;k<2;++k)
  for(uint32_t b=0;b<128;++b)
   assert(gm.bytes[DecoderMemoryLayout::decoder_norm_handoff_gm+(k*wanted+r)*128+b]==
          gm.bytes[input+((lane*64+r)*128+k*64)*2+b]);
''' + source[end:]
        source = source[:source.index('int main()')] + r'''
int main(){
 for(auto shape:std::vector<std::pair<uint32_t,uint32_t>>{{64,0},{33,0},{65,1},{257,0}}){
  auto expected=run(shape.first,shape.second,17,16,true,true,true,0);
  assert(run(shape.first,shape.second,17,16,true,true,true,1)==expected);
  assert(run(shape.first,shape.second,17,16,true,true,true,2)==expected);
 }
}
'''
        stub = timed_stub()
        old = 'for(size_t i=0;i<data.size();++i)bytes[addr+i]=data[i];cb(true,tag);return true;'
        stub = stub.replace(old, 'delayedReads.push_back([this,addr,data,tag,cb]{for(size_t i=0;i<data.size();++i)bytes[addr+i]=data[i];cb(true,tag);});return true;')
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
            result = subprocess.run([str(root/'test')], capture_output=True, env=env, timeout=180)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertEqual(result.stdout.count(b'consumer=1'), 3)


if __name__ == '__main__':
    unittest.main()
