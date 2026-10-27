"""Require operator scope to preserve Wo scheduling while changing fused FFN."""
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from test_decoder_tail import ROOT
from test_decoder_pv_wo_direct import direct_source
from test_decoder_groups import timed_stub


class FfnDemandScopeTest(unittest.TestCase):
 def test_only_ffn_changes_and_all_output_bits_match(self):
  source=direct_source();source=source[:source.index('int main()')]+r'''
int main(){
 setenv("GOLEM_REUSE_WINDOW_GM_LOOKAHEAD","4",1);
 setenv("GOLEM_DECODER_FFN_DEMAND_ORDER","0",1);
 auto expected=run(false,true);
 std::printf("DEMAND_BEGIN\n");
 setenv("GOLEM_DECODER_FFN_DEMAND_ORDER","1",1);
 assert(expected==run(false,true));
 std::printf("DEMAND_END\n");
 setenv("GOLEM_DECODER_FFN_DEMAND_ORDER","0",1);
 assert(expected==run(false,true));
}
'''
  stub=timed_stub().replace('std::vector<std::pair<uint64_t,size_t>> reads,writes;',
    'std::vector<std::pair<uint64_t,size_t>> reads,writes,localWrites,dmaDestinations;')
  stub=stub.replace('assert(addr+data.size()<=getSize());',
    'localWrites.push_back({addr,data.size()});assert(addr+data.size()<=getSize());')
  stub=stub.replace('reads.push_back({src,size});','reads.push_back({src,size});dmaDestinations.push_back({dst,size});')
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)
   for name in ('globalmemory/globalmemory.h','workercmdproc/workercmdproc.h'):
    p=path/'sst/elements/golem'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('#include "stub.h"\n')
   (path/'stub.h').write_text(stub);(path/'test.cc').write_text(source)
   includes=['-I',str(path)]
   if os.environ.get('GOLEM_TEST_SOURCE_OVERLAY'):includes+=['-I',os.environ['GOLEM_TEST_SOURCE_OVERLAY']]
   subprocess.run(['g++','-std=c++17','-O2',*includes,'-I',str(ROOT/'src'),str(path/'test.cc'),'-o',str(path/'test')],check=True)
   env=dict(os.environ,GOLEM_REUSE_WINDOW_K_TILES='32',GOLEM_REUSE_WINDOW_PREFETCH_SLOTS='4',
      GOLEM_REUSE_WINDOW_ARRAY_GROUPS='2',GOLEM_REUSE_WINDOW_A_PREFETCH='1',GOLEM_DECODER_FFN_LAYOUT='2',
      GOLEM_DECODER_GM_STAGE_CHAIN='0',GOLEM_DECODER_GEMM_PREFETCH='0')
   result=subprocess.run([str(path/'test')],capture_output=True,env=env,timeout=120)
   self.assertEqual(result.returncode,0,result.stderr.decode())
   scoped=result.stdout.decode().split('DEMAND_BEGIN\n')[1].split('DEMAND_END\n')[0]
   records=[dict(re.findall(r'(\w+)=([^\s]+)',line)) for line in scoped.splitlines() if '[GEMM_WINDOW_FLOW]' in line]
   self.assertEqual({r['operator'] for r in records},{'linear','gate','up','down'})
   for r in records:
    self.assertEqual(int(r['demand_order']),0 if r['operator']=='linear' else 1,r)
    if r['operator']!='linear':self.assertEqual(int(r['b_head_bypass_admissions']),0,r)


if __name__=='__main__':unittest.main(verbosity=2)
