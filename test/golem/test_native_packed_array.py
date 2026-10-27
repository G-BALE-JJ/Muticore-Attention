"""Exercise installed SST TUs directly, without a compiler or guest ABI."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'configure.ac').is_file())
class NativePackedArrayTests(unittest.TestCase):
    def test_packed_arithmetic_and_timed_callbacks(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC','-I/local/sstcore/include',
                '-I'+str(ROOT/'src'),str(ROOT/'test/golem/unit/packed_array_driver.cpp'),
                '-o',str(p/'libllmalign.so')],check=True)
            config=p/'config.py'
            config.write_text("""import os,sst
sst.setProgramOption('timebase','1ps')
c=sst.Component('check','llmalign.PackedArrayDriver')
f=int(os.environ['CHECK_FUNCTIONAL'])
c.addParams({'functional':f})
a=c.setSubComponent('array','golem.MVMFloatArray')
a.addParams({'verbose':0,'clock':'1GHz','numArrays':2,'arrayInputSize':64,
 'arrayOutputSize':3,'inputOperandSize':4,'outputOperandSize':4,
 'modeledComputeCycles':19,'functionalCompute':f})
""")
            for f in (1,0):
                result=subprocess.run(['/local/sstcore/bin/sst','--no-env-config',
                    '--lib-path='+str(p)+':'+str(ROOT/'install/lib/sst-elements-library'),
                    str(config)],env=dict(os.environ,CHECK_FUNCTIONAL=str(f)),
                    capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertIn('[PASS] native packed TU',result.stdout)
    def test_shared_python_pe_timing_table(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('pe_timing',ROOT/'test/golem/architecture/pe_timing.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        expected={'int4':(5,0),'int8':(4,0),'fp16':(5,1),'bf16':(5,1),
                  'tf32':(5,0),'fp32':(6,0),'fp8e4m3':(5,0),'fp8e5m2':(5,0)}
        for mode,(latency,gap) in expected.items():
            for packets in (1,2,16,64,96):
                self.assertEqual(module.pe_timing(mode,packets),dict(packet_ii=1,
                    single_packet_latency=latency,completion_cycles=packets-1+latency,
                    start_interval=packets+gap))
        with self.assertRaises(ValueError):module.pe_timing('fp32',0)
if __name__=='__main__':unittest.main()
