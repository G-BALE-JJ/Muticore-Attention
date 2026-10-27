"""Existing four B slots prefetch across N windows; K spills fall back safely."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from test_reuse_window_gemm import ROOT, STUB, SOURCE


class WindowBridgeTest(unittest.TestCase):
    def test_seam_prefetch_bits_tails_and_ack_ownership(self):
        source=SOURCE.replace('++outputs;outputAckAt=',r'''
 if(n==3 && nn>4 && nk<=std::stoul(std::getenv("GOLEM_REUSE_WINDOW_K_TILES"))){
  bool nextWeight=false;
  for(auto e:gm.reads)if(e.first==0x10000000+uint64_t(4)*nk*8192)nextWeight=true;
  assert(nextWeight);
 }
 ++outputs;outputAckAt=''')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('globalmemory/globalmemory.h','workercmdproc/workercmdproc.h'):
                p=root/'sst/elements/golem'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('#include "stub.h"\n')
            (root/'stub.h').write_text(STUB);(root/'test.cc').write_text(source)
            subprocess.run(['g++','-std=c++17','-O2','-I',str(root),'-I',str(ROOT/'src'),str(root/'test.cc'),'-o',str(root/'test')],check=True)
            for chunk in (4,16):
                for groups in (1,2):
                    for prefetch in (0,1):
                        env=dict(os.environ,GOLEM_REUSE_WINDOW_K_TILES=str(chunk),GOLEM_REUSE_WINDOW_ARRAY_GROUPS=str(groups),
                                 GOLEM_REUSE_WINDOW_EARLY_A='1',GOLEM_REUSE_WINDOW_A_PREFETCH=str(prefetch),
                                 GOLEM_REUSE_WINDOW_OUTPUT_OVERLAP='1',GOLEM_REUSE_WINDOW_B_WINDOW_BRIDGE='1')
                        p=subprocess.run([str(root/'test')],env=env,capture_output=True,timeout=60)
                        self.assertEqual(p.returncode,0,p.stderr.decode())


if __name__=='__main__':unittest.main()
