from pathlib import Path
import os
import re
import subprocess
import tempfile
import unittest
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'configure.ac').is_file())
class NativeEngineTests(unittest.TestCase):
    def test_native_te_precision_and_explicit_panels(self):
        fast_path_available='timingNumericalFastPath' in (ROOT/'src/sst/elements/golem/array/mvmComputeArray.h').read_text()
        placement=(ROOT/'src/sst/elements/golem/tileengine/workloadPlacement.h').is_file()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            flags=['-DTEST_PLACEMENT'] if placement else []
            subprocess.run(['g++','-std=c++17','-O1','-shared','-fPIC','-Wno-deprecated-declarations',*flags,
                '-I/local/sstcore/include','-I'+str(ROOT/'src'),
                str(ROOT/'test/golem/unit/native_engine_driver.cpp'),str(ROOT/'install/lib/sst-elements-library/libgolem.so'),
                '-Wl,-rpath,'+str(ROOT/'install/lib/sst-elements-library'),'-o',str(p/'libllmnative.so')],check=True)
            cfg=p/'config.py';cfg.write_text("""import os,sst
sst.setProgramOption('timebase','1ps')
for instance in range(int(os.environ.get('CHECK_INSTANCES','1'))):
 c=sst.Component('native%d'%instance,'llmnative.Driver')
 f=int(os.environ['CHECK_FUNCTIONAL'])
 c.addParams({'placed':os.environ['CHECK_PLACED'],'replicate':os.environ['CHECK_REPLICATE'],'functional':f,'instance':instance})
 a=c.setSubComponent('array','golem.TileUnitArray')
 a.addParams({'verbose':0,'clock':'1GHz','numTileUnits':2,'tuKCapacity':17,'cusPerTileUnit':3,
  'inputOperandSize':4,'outputOperandSize':4,'modeledComputeCycles':19,'functionalCompute':f})
 if 'CHECK_TIMING_FAST_PATH' in os.environ: a.addParam('timingNumericalFastPath',int(os.environ['CHECK_TIMING_FAST_PATH']))
 t=c.setSubComponent('engine','golem.TileEngineLocal')
 t.addParams({'verbose':0,'dtype_is_float':1,'output_mode':'fusion','c_buffer_bytes':4096,
  'c_buffer_read_bytes_per_cycle':64,'c_buffer_write_bytes_per_cycle':64,'window_k_tiles':1})
 c.setSubComponent('memory','llmnative.Memory')
 c.setSubComponent('scheduler','llmnative.Scheduler')
""")
            cases=[(0,0,1,1),(0,0,0,1)]
            if placement:cases+=[(1,0,1,1),(1,1,1,1),(2,1,1,1)]
            if fast_path_available:cases.append((0,0,0,0))
            cycles={}
            for placed,replicate,functional,fast in cases:
                print('CASE',placed,replicate,functional,fast,flush=True)
                extra_env={'CHECK_TIMING_FAST_PATH':str(fast)} if fast_path_available else {}
                result=subprocess.run(['/local/sstcore/bin/sst','--no-env-config',
                    '--lib-path='+str(p)+':'+str(ROOT/'install/lib/sst-elements-library'),str(cfg)],
                    env=dict(os.environ,**extra_env,CHECK_PLACED=str(placed),CHECK_REPLICATE=str(replicate),CHECK_FUNCTIONAL=str(functional)),
                    capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertIn('[PASS] native TE',result.stdout)
                cycles[(placed,replicate,functional,fast)]=int(re.search(r'\[NATIVE_CYCLES\] cycles=(\d+)',result.stdout)[1])
            self.assertEqual(cycles[(0,0,1,1)],cycles[(0,0,0,1)],'Timing-only mode changes the native TE event schedule')
            if fast_path_available:
                self.assertEqual(cycles[(0,0,0,0)],cycles[(0,0,0,1)],'Same-build numerical fast path changes TE event cycles')
            if placement:
                result=subprocess.run(['/local/sstcore/bin/sst','--no-env-config',
                    '--lib-path='+str(p)+':'+str(ROOT/'install/lib/sst-elements-library'),str(cfg)],
                    env=dict(os.environ,CHECK_PLACED='1',CHECK_REPLICATE='1',CHECK_FUNCTIONAL='1',CHECK_INSTANCES='2'),
                    capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertEqual(result.stdout.count('[PASS] native TE'),2,result.stdout)
                self.assertIn('instance=0',result.stdout)
                self.assertIn('instance=1',result.stdout)

if __name__=='__main__':unittest.main()
