"""Disjoint single-context subsets with timed MACs, retries and late DMA ACKs."""
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_decoder_down_fusion import SOURCE
from test_decoder_ffn_panels import STUB
from test_decoder_tail import ROOT


def timed_stub():
    stub = STUB.replace('output=values;cb(true,tag);return true;', '''
  for(size_t r=0;r<a.size();++r) {
   assert(!busy[a[r]]);outputs[a[r]].assign(values.begin()+r*64,values.begin()+(r+1)*64);
  }
  cb(true,tag);return true;''')
    begin = stub.index(' std::vector<double> matrix,input,output;')
    end = stub.index('\n};', begin)
    return stub[:begin] + r'''
 uint64_t now=0,bufferReady=0;size_t attempts=0,overlaps=0;
 std::map<uint32_t,bool> busy;
 std::map<uint32_t,std::vector<double>> matrices,inputs,outputs;
 std::vector<std::pair<uint64_t,std::function<void()>>> events;
 void step(uint64_t cycle) {
  now=cycle;
  for(size_t i=0;i<events.size();) {
   if(events[i].first>now){++i;continue;}
   auto cb=events[i].second;events.erase(events.begin()+i);cb();
  }
 }
 bool retry() {return ++attempts%7==0;}
 void buffer(size_t cycles,std::function<void()> cb) {
  bufferReady=std::max(now,bufferReady)+cycles;events.push_back({bufferReady,cb});
 }
 bool isBusy() {return false;}
 bool programGemmMatrixGroupClassBankAsync(const std::vector<uint32_t>& a,uint32_t bank,
  const std::vector<double>& v,size_t,AttentionClusterTrafficClass,uint64_t tag,uint64_t,
  std::function<void(bool,uint64_t)> cb) {
  assert(bank==0 && v.size()==4096);
  if(retry())return false;
  for(auto id:a)assert(!busy[id]);
  for(auto b:busy)if(b.second)++overlaps;
  uint32_t fanout=0;for(size_t n=1;n<a.size();n*=2)++fanout;
  buffer(33+fanout,[this,a,v,tag,cb]{for(auto id:a)matrices[id]=v;cb(true,tag);});return true;
 }
 bool programGemmInputScatterBankAsync(const std::vector<uint32_t>& a,uint32_t bank,
  const std::vector<double>& v,size_t,AttentionClusterTrafficClass,uint64_t tag,uint64_t,
  std::function<void(bool,uint64_t)> cb) {
  assert(bank==0 && v.size()==a.size()*64);
  if(retry())return false;
  for(auto id:a)assert(!busy[id]);
  buffer(1+(v.size()*2+255)/256,[this,a,v,tag,cb] {
   for(size_t r=0;r<a.size();++r)inputs[a[r]].assign(v.begin()+r*64,v.begin()+(r+1)*64);
   cb(true,tag);
  });return true;
 }
 bool launchGemmArrayGroupActiveBank(const std::vector<uint32_t>& a,uint32_t bank,
  uint64_t mode,uint32_t columns,uint64_t cycle,std::function<void(uint32_t,uint64_t)> cb) {
  assert(bank==0 && columns==64);
  if(retry())return false;
  for(auto id:a) {
   assert(!busy[id]);busy[id]=true;
   // Deliver individual array callbacks on different cycles, in reverse order.
   // Using array state at completion exposes any premature reprogramming.
   events.push_back({cycle+66+(63-id)%3,[this,id,mode,cb] {
    auto& output=outputs[id];if(!mode)output.assign(64,0);
    assert(output.size()==64);
    for(size_t n=0;n<64;++n) {
     double dot=0;for(size_t k=0;k<64;++k)
      dot=vectorFp16Value(dot+vectorFp16Value(inputs[id][k]*matrices[id][n*64+k]));
     output[n]=vectorFp16Value(output[n]+dot);
    }
    busy[id]=false;cb(id,now);
   }});
  }
  return true;
 }
 bool readGemmOutputGroupClassAsync(const std::vector<uint32_t>& a,size_t,
  AttentionClusterTrafficClass,uint64_t tag,uint64_t,
  std::function<void(bool,uint64_t,const std::vector<double>&)> cb) {
  if(retry())return false;
  for(auto id:a)assert(!busy[id]);
  buffer(1+(a.size()*128+255)/256,[this,a,tag,cb] {
   std::vector<double> v;for(auto id:a)v.insert(v.end(),outputs[id].begin(),outputs[id].end());
   cb(true,tag,v);
  });return true;
 }
''' + stub[end:]


class DecoderGroupsTest(unittest.TestCase):
    def test_group_ownership_K_order_tails_retries_and_delayed_ACKs(self):
        source = SOURCE.replace(
            'bool residual=false) {', 'bool residual=false,unsigned groups=1) {\n'
            ' setenv("GOLEM_DECODER_ARRAY_GROUPS",std::to_string(groups).c_str(),1);')
        source = source.replace('sfu.step(t*1000);job.tick(t,t*1000);',
                                'sfu.step(t*1000);wcp.step(t);job.tick(t,t*1000);')
        source = source.replace('assert(!wcp.partialOwned);job.retire();sfu.now=0;',
                                'assert(!wcp.partialOwned && wcp.events.empty());job.retire();sfu.now=0;wcp.bufferReady=0;')
        source = source.replace('return result;', 'if(groups==2 && wanted>32)assert(wcp.overlaps>0);return result;')
        source = source[:source.index('int main()')] + r'''
int main() {
 for(const char* slots:{"2","4","8"}) {
  setenv("GOLEM_DECODER_PREFETCH_SLOTS",slots,1);
  for(auto shape:std::vector<std::pair<uint32_t,uint32_t>>{
       {64,0},{33,0},{65,1},{97,1},{257,0},{1,0},{65,2}}) {
   for(bool prefetch:{false,true}) {
    const auto expected=run(shape.first,shape.second,17,8,prefetch,true,true,1);
    assert(run(shape.first,shape.second,17,8,prefetch,true,true,2)==expected);
    assert(run(shape.first,shape.second,17,3,prefetch,true,true,2)==expected);
   }
  }
 }
}
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('globalmemory/globalmemory.h', 'workercmdproc/workercmdproc.h'):
                path = root / 'sst/elements/golem' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('#include "stub.h"\n')
            (root / 'stub.h').write_text(timed_stub())
            (root / 'test.cc').write_text(source)
            subprocess.run(['g++', '-std=c++17', '-O2', '-I', str(root), '-I', str(ROOT / 'src'),
                            str(root / 'test.cc'), '-o', str(root / 'test')], check=True)
            completed = subprocess.run([str(root / 'test')], capture_output=True, timeout=180)
            self.assertEqual(completed.returncode, 0, completed.stderr.decode())


if __name__ == '__main__':
    unittest.main()
