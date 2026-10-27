"""Require direct, leased GM input and disjoint Wo weights on the real job."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_decoder_tail import ROOT
from test_decoder_groups import timed_stub
from test_decoder_pv_wo_handoff import SOURCE


def direct_source():
    source=SOURCE.replace('bool ready=true;unsigned calls=0;', '''
 bool ready=true,leased=false;unsigned calls=0,acquires=0,releases=0;
 bool acquire(uint32_t epoch,uint32_t manager,uint32_t lane) {
  assert(epoch==7 && manager==2 && lane==3);++acquires;
  if(!ready || leased)return false;leased=true;return true;
 }
 bool release(uint32_t epoch,uint32_t manager,uint32_t lane) {
  assert(epoch==7 && manager==2 && lane==3 && leased);++releases;leased=false;return true;
 }''')
    source=source.replace('run(bool handoff,bool window)', 'run(bool handoff,bool window,bool direct=false)')
    source=source.replace(' setenv("GOLEM_PV_WO_GM_HANDOFF",',
                          ' setenv("GOLEM_PV_WO_DIRECT_INPUT",direct?"1":"0",1);\n setenv("GOLEM_PV_WO_GM_HANDOFF",')
    source=source.replace('gm.bytes[DecoderMemoryLayout::decoder_input_gm+b]',
                          'gm.bytes[(direct?DecoderMemoryLayout::decoder_pv_wo_landing_gm:DecoderMemoryLayout::decoder_input_gm)+b]')
    source=source.replace(' std::vector<uint8_t> result;', '''
 if(direct){
  assert(input.calls==0 && input.acquires==2 && input.releases==1 && !input.leased);
  size_t relocated=0;
  for(auto e:gm.localWrites)
   if(DecoderMemoryLayout::decoder_input_gm<=e.first && e.first<DecoderMemoryLayout::decoder_input_gm+0x40000)
    relocated+=e.second;
  assert(relocated==0);
  // Every timed weight DMA must land below the immutable input tile.
  for(auto e:gm.dmaDestinations)assert(e.first+e.second<=DecoderMemoryLayout::decoder_pv_wo_landing_gm);
 }
 std::vector<uint8_t> result;''')
    source=source.replace(' return result;',r'''
 // The next real FFN job reuses the same tail for its Up panels. It must
 // start only after the direct Wo lease has been released, and must no
 // longer use the previous job's landing input address.
 job.retire();wcp.bufferReady=0;sfu.now=0;
 DecoderOpDesc f=d;f.op=9;f.stage=4;f.input_dim=128;f.output_dim=64;
 f.input_addr=0x60000000;f.output_addr=0x62000000;
 f.weights_offset=0x02000000;f.input1_addr=0x02400000;f.down_weights_offset=0x02800000;
 for(uint32_t r=192;r<256;++r)for(uint32_t c=0;c<128;++c)gm.put(f.input_addr+(r*128+c)*2,.25);
 for(uint32_t c=0;c<64;++c){
  gm.put(d.node_stride+f.weights_offset+(c*64+c)*2,.5);
  gm.put(d.node_stride+f.input1_addr+(c*64+c)*2,.25);
  for(uint32_t n=0;n<2;++n)gm.put((1+n%4)*d.node_stride+f.down_weights_offset+(c*64+c)*2,.5);
 }
 assert(!input.leased);
 assert(job.start(f));
 for(uint64_t t=1;t<500000 && !job.complete();++t){
  sfu.step(t*1000);wcp.step(t);job.tick(t,t*1000);if(t%19==0)gm.drain();
 }
 assert(job.complete() && !job.status() && !input.leased);
 assert(gm.pending.empty() && gm.delayedReads.empty() && wcp.events.empty());
 for(uint32_t r=192;r<256;++r)for(uint32_t b=0;b<256;++b)
  result.push_back(gm.bytes[f.output_addr+r*256+b]);
 if(direct){
  size_t reused=0;
  for(auto e:gm.localWrites)if(e.first<0x1e0000 && 0x1a0000<e.first+e.second)reused+=e.second;
  assert(reused>0);
 }
 return result;''')
    source=source[:source.index('int main()')]+'''
int main(){
 for(const char* groups:{"1","2"}){
  setenv("GOLEM_REUSE_WINDOW_ARRAY_GROUPS",groups,1);
  const auto expected=run(false,true);
  assert(expected==run(true,true));
  assert(expected==run(true,true,true));
 }
 // Legacy Wo loop remains on its explicitly relocated fallback.
 assert(run(false,false)==run(true,false));
}
'''
    return source


class DecoderPvWoDirectTest(unittest.TestCase):
    def test_direct_input_is_immutable_weights_disjoint_and_lease_released(self):
        self.exercise(direct_source())

    def test_failed_input_ack_keeps_the_landing_leased(self):
        source=direct_source()
        old=' assert(job.complete() && !job.status());'
        assert source.count(old)==1
        source=source.replace(old,"""
 assert(job.complete() && job.status());
 assert(input.leased && input.releases==0);
 job.retire();assert(!job.start(d));
 auto other=d;other.op=3;other.stage=3;other.output_dim=other.input_dim;other.input1_addr=0x50000000;
 assert(!job.start(other));
 return {};
""")
        source=source[:source.index('int main()')]+"int main(){run(true,true,true);}"
        self.exercise(source,fail_input=True)

    def exercise(self,source,fail_input=False):
        stub=timed_stub().replace(
            'std::vector<std::pair<uint64_t,size_t>> reads,writes;',
            'std::vector<std::pair<uint64_t,size_t>> reads,writes,localWrites,dmaDestinations;')
        stub=stub.replace('assert(addr+data.size()<=getSize());',
                          'localWrites.push_back({addr,data.size()});assert(addr+data.size()<=getSize());')
        stub=stub.replace('reads.push_back({src,size});',
                          'reads.push_back({src,size});dmaDestinations.push_back({dst,size});')
        # A local write ACK is delayed as well as GM reads, so publishing or
        # releasing the input early cannot be masked by synchronous callbacks.
        stub=stub.replace('for(size_t i=0;i<data.size();++i)bytes[addr+i]=data[i];cb(true,tag);return true;',
                          'delayedReads.push_back([this,addr,data,tag,cb]{for(size_t i=0;i<data.size();++i)bytes[addr+i]=data[i];cb(true,tag);});return true;')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            for name in ('globalmemory/globalmemory.h','workercmdproc/workercmdproc.h'):
                p=path/'sst/elements/golem'/name;p.parent.mkdir(parents=True,exist_ok=True)
                p.write_text('#include "stub.h"\n')
            if fail_input:
                old='std::vector<uint8_t> data(size);for(size_t i=0;i<size;++i)data[i]=bytes[addr+i];cb(true,tag,data);return true;'
                assert old in stub
                stub=stub.replace(old,'if(addr>=0x1a0000 && addr<0x1e0000){delayedReads.push_back([cb,tag]{cb(false,tag,{});});return true;}'+old)
            (path/'stub.h').write_text(stub);(path/'test.cc').write_text(source)
            includes=['-I',str(path)]
            overlay=os.environ.get('GOLEM_TEST_SOURCE_OVERLAY')
            if overlay:includes+=['-I',overlay]
            subprocess.run(['g++','-std=c++17','-O2',*includes,'-I',str(ROOT/'src'),str(path/'test.cc'),'-o',str(path/'test')],check=True)
            env=dict(os.environ,GOLEM_REUSE_WINDOW_K_TILES='32',GOLEM_DECODER_OUTPUT_PANEL_TILES='16',
                     GOLEM_REUSE_WINDOW_PREFETCH_SLOTS='4',GOLEM_REUSE_WINDOW_A_PREFETCH='1',
                     GOLEM_DECODER_GM_STAGE_CHAIN='0',GOLEM_DECODER_GEMM_PREFETCH='0',GOLEM_DECODER_FFN_LAYOUT='2')
            result=subprocess.run([str(path/'test')],capture_output=True,env=env,timeout=120)
            self.assertEqual(result.returncode,0,result.stderr.decode())


if __name__=='__main__':unittest.main()
