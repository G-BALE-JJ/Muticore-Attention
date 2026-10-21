"""Critical A must not queue behind speculative weight snapshots on one GM port."""
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from test_reuse_window_gemm import ROOT, STUB, SOURCE


TIMED_READ = r'''
 uint64_t now=0,portReady=0,readBytes=0;
 std::vector<std::pair<uint64_t,std::function<void()>>> localPending;
 void step(uint64_t t){now=t;for(size_t i=localPending.size();i-->0;)if(localPending[i].first<=t){
  auto cb=std::move(localPending[i].second);localPending.erase(localPending.begin()+i);cb();}}
 bool localReadAsync(uint64_t addr,size_t size,LocalMemoryClient,uint64_t tag,
  std::function<void(bool,uint64_t,const std::vector<uint8_t>&)> cb) {
  assert(addr+size<=getSize());if(++localReadAttempts%7==0)return false;
  readBytes+=size;portReady=std::max(now,portReady)+(size+255)/256;
  localPending.push_back({portReady,[=]{std::vector<uint8_t> data(size);
   for(size_t i=0;i<size;++i)data[i]=bytes[addr+i];cb(true,tag,data);}});return true;
 }
'''


class ReuseWindowGmScheduleTest(unittest.TestCase):
    def test_next_a_pipeline_on_banked_port(self):
        begin=STUB.index(' bool localReadAsync(');end=STUB.index(' bool localWriteAsync(',begin)
        stub=STUB[:begin]+TIMED_READ.replace('(size+255)/256','(size+127)/128')+STUB[end:]
        stub=stub.replace('now+2','now+39',1).replace('now+2','now+33').replace('now+11','now+66')
        source=SOURCE.replace('wcp.step(t);','wcp.step(t);gm.step(t);')
        source=source[:source.index('int main()')]+'int main(){run(64,9,5,false,true);}\n'
        source=source.replace('job.trace(4,1,4,"test");',
            'assert(gm.localPending.empty());job.trace(4,1,4,"test");'
            'std::printf("RESULT cycles=%llu reads=%llu\\n",(unsigned long long)clock,(unsigned long long)gm.readBytes);')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('globalmemory/globalmemory.h','workercmdproc/workercmdproc.h'):
                path=root/'sst/elements/golem'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('#include "stub.h"\n')
            (root/'stub.h').write_text(stub);(root/'test.cc').write_text(source)
            subprocess.run(['g++','-std=c++17','-O2','-I',str(root),'-I',str(ROOT/'src'),str(root/'test.cc'),'-o',str(root/'test')],check=True)
            results={}
            for enabled in (0,1):
                env=dict(os.environ,GOLEM_REUSE_WINDOW_K_TILES='16',GOLEM_REUSE_WINDOW_ARRAY_GROUPS='1',
                    GOLEM_REUSE_WINDOW_EARLY_A='1',GOLEM_REUSE_WINDOW_OUTPUT_OVERLAP='1',
                    GOLEM_REUSE_WINDOW_GM_LOOKAHEAD='4',GOLEM_REUSE_WINDOW_A_FIRST='0',
                    GOLEM_REUSE_WINDOW_A_PREFETCH=str(enabled))
                result=subprocess.run([str(root/'test')],env=env,capture_output=True,timeout=30)
                self.assertEqual(result.returncode,0,result.stderr.decode())
                results[enabled]={k:int(v) for k,v in re.findall(r'(cycles|reads)=(\d+)',result.stdout.decode())}
            self.assertLess(results[1]['cycles'],results[0]['cycles'],results)
            self.assertEqual(results[1]['reads'],results[0]['reads'])

    def test_critical_operand_latency_without_extra_bandwidth(self):
        begin=STUB.index(' bool localReadAsync(')
        end=STUB.index(' bool localWriteAsync(',begin)
        stub=STUB[:begin]+TIMED_READ+STUB[end:]
        stub=stub.replace('std::map<uint32_t,Array> arrays;uint64_t now=0,',
                          'std::map<uint32_t,Array> arrays;uint64_t firstLaunch=0,now=0,')
        stub=stub.replace('if(!accept())return false;for(auto id:ids){auto& a=arrays[id];',
                          'if(!accept())return false;if(!firstLaunch)firstLaunch=now;for(auto id:ids){auto& a=arrays[id];')
        source=SOURCE.replace('wcp.step(t);','wcp.step(t);gm.step(t);')
        source=source[:source.index('int main()')]+ 'int main(){run(64,9,5,false,true);}\n'
        source=source.replace('job.trace(4,1,4,"test");',
            'assert(gm.localPending.empty());job.trace(4,1,4,"test");'
            'std::printf("RESULT first=%llu cycles=%llu reads=%llu\\n",'
            '(unsigned long long)wcp.firstLaunch,(unsigned long long)clock,(unsigned long long)gm.readBytes);')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('globalmemory/globalmemory.h','workercmdproc/workercmdproc.h'):
                path=root/'sst/elements/golem'/name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text('#include "stub.h"\n')
            (root/'stub.h').write_text(stub);(root/'test.cc').write_text(source)
            subprocess.run(['g++','-std=c++17','-O2','-I',str(root),'-I',str(ROOT/'src'),str(root/'test.cc'),'-o',str(root/'test')],check=True)
            results={}
            for lookahead in (4,1,2):
                env=dict(os.environ,GOLEM_REUSE_WINDOW_K_TILES='16',GOLEM_REUSE_WINDOW_ARRAY_GROUPS='1',
                         GOLEM_REUSE_WINDOW_EARLY_A='1',GOLEM_REUSE_WINDOW_OUTPUT_OVERLAP='1',
                         GOLEM_REUSE_WINDOW_GM_LOOKAHEAD=str(lookahead),GOLEM_REUSE_WINDOW_A_FIRST='1')
                result=subprocess.run([str(root/'test')],env=env,capture_output=True,timeout=30)
                self.assertEqual(result.returncode,0,result.stderr.decode())
                results[lookahead]={k:int(v) for k,v in re.findall(r'(first|cycles|reads)=(\d+)',result.stdout.decode())}
            self.assertLess(results[1]['first'],results[4]['first'],results)
            self.assertEqual({v['reads'] for v in results.values()},{45*16384})
            # Exercise the unchanged K/C/output ownership protocol with M/N/K
            # tails, initial C, resident/nonresident A, two array subgroups,
            # refused submissions and delayed output ACKs on the same port.
            full_source=SOURCE.replace('wcp.step(t);','wcp.step(t);gm.step(t);')
            (root/'test.cc').write_text(full_source)
            subprocess.run(['g++','-std=c++17','-O2','-I',str(root),'-I',str(ROOT/'src'),str(root/'test.cc'),'-o',str(root/'test')],check=True)
            for lookahead in (1,2):
                for groups in (1,2):
                    for a_first in (0,1):
                        env=dict(os.environ,GOLEM_REUSE_WINDOW_K_TILES='4',GOLEM_REUSE_WINDOW_ARRAY_GROUPS=str(groups),
                                 GOLEM_REUSE_WINDOW_EARLY_A='1',GOLEM_REUSE_WINDOW_OUTPUT_OVERLAP='1',
                                 GOLEM_REUSE_WINDOW_GM_LOOKAHEAD=str(lookahead),GOLEM_REUSE_WINDOW_A_FIRST=str(a_first),
                                 GOLEM_REUSE_WINDOW_A_PREFETCH='1')
                        result=subprocess.run([str(root/'test')],env=env,capture_output=True,timeout=30)
                        self.assertEqual(result.returncode,0,result.stderr.decode())


if __name__=='__main__':unittest.main()
