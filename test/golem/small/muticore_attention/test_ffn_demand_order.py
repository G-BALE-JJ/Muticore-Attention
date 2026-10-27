"""Run the real window scheduler with delayed ports and modulo-slot wraparound."""
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from test_reuse_window_gemm import SOURCE, STUB
from test_decoder_tail import ROOT


class DemandOrderTest(unittest.TestCase):
    def test_demand_precedes_speculation_without_changing_work_or_outputs(self):
        source = SOURCE.replace('init,source,true);',
            'init,source,true,{},{},{},0,std::string(std::getenv("GOLEM_TEST_DEMAND_ORDER"))=="1");')
        # Batched delayed replies make all slot IDs contend in the same tick.
        # Keep FIFO callback order, independently of the reordered HBM replies.
        stub=STUB.replace('void drain() {',
            'std::vector<std::function<void()>> localReads;\n void drain() {\n'
            '  auto callbacks=std::move(localReads);localReads.clear();for(auto& cb:callbacks)cb();')
        source=source.replace('t%3==0','t%19==0')
        old='std::vector<uint8_t> data(size);for(size_t i=0;i<size;++i)data[i]=bytes[addr+i];cb(true,tag,data);return true;'
        self.assertIn(old,stub)
        stub=stub.replace(old,'localReads.push_back([this,addr,size,tag,cb]{std::vector<uint8_t> data(size);'
                          'for(size_t i=0;i<size;++i)data[i]=bytes[addr+i];cb(true,tag,data);});return true;')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('globalmemory/globalmemory.h','workercmdproc/workercmdproc.h'):
                path=root/'sst/elements/golem'/name
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text('#include "stub.h"\n')
            (root/'stub.h').write_text(stub)
            (root/'test.cc').write_text(source)
            subprocess.run(['g++','-std=c++17','-O2','-I',str(root),'-I',str(ROOT/'src'),
                            str(root/'test.cc'),'-o',str(root/'test')],check=True,capture_output=True)
            for chunk in (4,16):
                for mode in (0,1):
                    env=dict(os.environ,GOLEM_REUSE_WINDOW_K_TILES=str(chunk),
                             GOLEM_REUSE_WINDOW_ARRAY_GROUPS='2',GOLEM_REUSE_WINDOW_EARLY_A='1',
                             GOLEM_REUSE_WINDOW_OUTPUT_OVERLAP='1',GOLEM_REUSE_WINDOW_A_PREFETCH='1',
                             GOLEM_TEST_DEMAND_ORDER=str(mode))
                    result=subprocess.run([str(root/'test')],env=env,capture_output=True,text=True,timeout=30)
                    self.assertEqual(result.returncode,0,result.stderr)
                    records=[dict(re.findall(r'(\w+)=(\d+)',line)) for line in result.stdout.splitlines()
                             if '[GEMM_WINDOW_FLOW]' in line]
                    self.assertTrue(records)
                    self.assertEqual({int(r['demand_order']) for r in records},{mode})
                    bypass=sum(int(r['b_head_bypass_admissions']) for r in records)
                    if mode:self.assertEqual(bypass,0)
                    else:self.assertGreater(bypass,0)


if __name__=='__main__':unittest.main()
