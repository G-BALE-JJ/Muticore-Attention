"""One matrix broadcast, two scatter/MAC waves on existing single-context arrays."""
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from test_reuse_window_gemm import ROOT, STUB, SOURCE


class ScatterWaveTest(unittest.TestCase):
    def test_bits_broadcast_bytes_and_single_context_ownership(self):
        stub = STUB.replace('uint64_t now=0,attempts=0,',
                            'uint64_t matrixOps=0,matrixArrays=0,scatterArrays=0,now=0,attempts=0,')
        stub = stub.replace('assert(b.size()==4096);',
                            '++matrixOps;matrixArrays+=ids.size();assert(b.size()==4096);')
        stub = stub.replace('assert(a.size()==ids.size()*64);',
                            'scatterArrays+=ids.size();assert(a.size()==ids.size()*64);')
        source = SOURCE.replace('job.trace(4,1,4,"test");', r'''
 assert(wcp.matrixArrays==uint64_t(rows)*nn*nk);
 assert(wcp.scatterArrays==wcp.matrixArrays);
 assert(wcp.matrixOps==uint64_t(nn)*nk);
 job.trace(4,1,4,"test");
''')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('globalmemory/globalmemory.h', 'workercmdproc/workercmdproc.h'):
                p = root/'sst/elements/golem'/name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('#include "stub.h"\n')
            (root/'stub.h').write_text(stub)
            (root/'test.cc').write_text(source)
            subprocess.run(['g++', '-std=c++17', '-O2', '-I', str(root), '-I', str(ROOT/'src'),
                            str(root/'test.cc'), '-o', str(root/'test')], check=True)
            for chunk in (4, 16):
                for prefetch in (0, 1):
                    env = dict(os.environ, GOLEM_REUSE_WINDOW_K_TILES=str(chunk),
                               GOLEM_REUSE_WINDOW_ARRAY_GROUPS='1', GOLEM_REUSE_WINDOW_SCATTER_WAVE='1',
                               GOLEM_REUSE_WINDOW_A_PREFETCH=str(prefetch), GOLEM_REUSE_WINDOW_EARLY_A='1',
                               GOLEM_REUSE_WINDOW_OUTPUT_OVERLAP='1')
                    result = subprocess.run([str(root/'test')], env=env, capture_output=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stderr.decode())
                    for line in result.stdout.decode().splitlines():
                        if '[GEMM_WINDOW_FLOW]' in line:
                            f = {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', line)}
                            self.assertEqual(sum(v for k,v in f.items() if k.startswith('phase_')), f['active_cycles'])


if __name__ == '__main__': unittest.main()
