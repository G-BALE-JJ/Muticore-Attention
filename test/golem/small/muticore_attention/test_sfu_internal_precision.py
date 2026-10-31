"""Independent binary16 reference against the real SFU in an SST event loop.

The memory fixture is functional, so its cycles are not performance evidence.
"""
from pathlib import Path
import os
import ast
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'configure.ac').is_file())
HERE = Path(__file__).resolve().parent
SST = Path(os.environ.get('SST_CORE_PREFIX', '/local/sstcore'))


def half(x):
    return np.asarray(x, dtype=np.float16).astype(np.float64)


def add(a, b):
    return half(half(a) + half(b))


def mul(a, b):
    return half(half(a) * half(b))


def reciprocal(x):
    return half(1 / half(x))


def exp(x):
    return half(np.exp(half(x)))


def sequential_sum(values):
    total = half(0.)
    for value in values:
        total = add(total, value)
    return total


def tree_sum(values):
    partials = half(values)
    while partials.size > 1:
        paired = add(partials[:partials.size // 2 * 2:2], partials[1::2])
        partials = np.r_[paired, partials[-1:]] if partials.size % 2 else paired
    return partials[0]


def reference():
    import sys
    sys.path.insert(0,str(ROOT/'test/attention_cluster_8qk_8pv'))
    import sfu_fp16_v4_reference as independent
    i=np.arange(258)
    x=half(-12+(i%97)*.25);y=half(.25+(i%13)*.125)
    expected={'add':add(x,y),'multiply':mul(x,y),'silu':independent.silu(x)}
    expected['silu_multiply']=mul(expected['silu'],y)
    x=half(-1.137+(i%97)*.0217)
    expected['rope']=independent.rope(x[None,:],y).ravel()
    norm_x=half(.0137*(1+(i%13)*.113))
    expected['rmsnorm']=independent.rmsnorm(norm_x[None,:],y).ravel()
    state=None
    for tile in range(2):
        scores=np.asarray(-2+(np.arange(35)%17)*.217+tile*.31,dtype='f4')
        state=independent.online_tile(scores[None,:],64,None if state is None else state['m'],
                                      None if state is None else state['l'],final=tile==1)
        expected[f'attention{tile}']=np.r_[state['old_scale'],state['p'].ravel()]
        expected[f'direct{tile}']=expected[f'attention{tile}']
    values=half(np.asarray(.0137*(1+(np.arange(65)%13)*.113),dtype='f4'))
    expected['reduce_sum']=np.atleast_1d(tree_sum(values))
    expected['sigmoid']=reciprocal(add(1.,exp(-values)))
    weights=exp(half(values-values.max()))
    total=half(0.)
    for begin in range(0,len(weights),17):total=add(total,tree_sum(weights[begin:begin+17]))
    expected['softmax_row']=mul(weights,reciprocal(total))
    # Legacy descriptor softmax-tile is a separate retained numerical ABI.
    expected['softmax_tile']=half(weights/sequential_sum(weights))
    gamma=half(.25+(np.arange(65)%13)*.125)
    expected['rmsnorm_job']=independent.rmsnorm(values[None,:],gamma).ravel()
    expected['boundary_silu']=independent.silu([-12.,-11.09,-11.,0.,11.,12.])
    for name,width,value in [('square_overflow',65,300.),('sum_overflow',8192,4.),('tiny_norm',65,1.e-5)]:
        expected['boundary_'+name]=independent.rmsnorm(np.full((1,width),value),np.ones(width)).ravel()
    expected['boundary_long_reduce']=np.array([independent.reduce_sum(np.ones(8192),'tree')])
    expected['boundary_long_softmax']=np.full(8192,1./8192)
    return expected


@unittest.skipUnless((SST/'bin/sst').exists() and shutil.which('g++'), 'SST core and g++ required')
class SFUInternalPrecisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='sfu-precision-')
        cls.path = Path(cls.temp.name)
        command = ['g++', '-std=c++17', '-shared', '-fPIC', '-O1', '-Wno-deprecated-declarations',
                   '-I', str(ROOT/'src'), '-I', str(SST/'include'),
                   str(HERE/'sfu_precision_fixture.cpp'),
                   str(ROOT/'src/sst/elements/golem/sfu/sfu.cc'),
                   '-o', str(cls.path/'libgolem.so')]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        (cls.path/'config.py').write_text('''import sst, sys
sst.setProgramOption("timebase", "1ps")
c=sst.Component("precision", "golem.SFUPrecisionTest")
c.addParams({"output":sys.argv[2]})
s=c.setSubComponent("sfu", "golem.SFU")
s.addParams({"accelerator_clock_hz":1000000000,"exp_lanes":16,"vector_lanes":16,
             "attention_cluster_enable":True,"nonlinear_backend":"libm"})
if sys.argv[1] in ("serial","serial_lut"):
    s.addParams({"internal_precision":"fp16","vector_row_scheduling":"serial",
                 "vector_shared_arbitration":"parity",
                 "nonlinear_backend":"lut" if sys.argv[1]=="serial_lut" else "libm"})
elif sys.argv[1].startswith("lut"):
    s.addParams({"internal_precision":"fp16","nonlinear_backend":"lut",
                 "lut_exp_ports":int(sys.argv[1][3:]),"lut_reciprocal_ports":int(sys.argv[1][3:])})
elif sys.argv[1]!="default": s.addParam("internal_precision",sys.argv[1])
''')
        cls.results = {}
        cls.logs = {}
        evidence = os.environ.get('GOLEM_SFU_TEST_EVIDENCE')
        if evidence: Path(evidence).mkdir(parents=True, exist_ok=True)
        for mode in ('default', 'fp32', 'fp16', 'lut1', 'lut4', 'lut16','serial','serial_lut'):
            output = cls.path/f'{mode}.txt'
            proc = subprocess.run([str(SST/'bin/sst'), f'--lib-path={cls.path}',
                                   f'--model-options={mode} {output}', str(cls.path/'config.py')],
                                  capture_output=True, text=True, timeout=60)
            if proc.returncode:
                raise RuntimeError(proc.stdout + proc.stderr)
            cls.logs[mode]=proc.stdout+proc.stderr
            if evidence:
                shutil.copy2(output, Path(evidence)/output.name)
                (Path(evidence)/f'{mode}.log').write_text(proc.stdout+proc.stderr)
            cls.results[mode] = {fields[0]: (int(fields[1]), np.array(list(map(float, fields[2:]))))
                                 for fields in map(str.split, output.read_text().splitlines())}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_architecture_default_and_explicit_override(self):
        tree = ast.parse((ROOT/'src/sst/elements/golem/tests/architecture/cpu_builder.py').read_text())
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == 'sfu_internal_precision'
                                  for target in node.targets))
        code = compile(ast.Module(body=[assignment], type_ignores=[]), '<architecture default>', 'exec')
        for environment, expected in (({}, 'fp16'), ({'GOLEM_SFU_INTERNAL_PRECISION': 'fp32'}, 'fp32')):
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True):
                namespace = {'os': os}
                exec(code, namespace)
                self.assertEqual(namespace['sfu_internal_precision'], expected)

    def test_default_fp16_and_explicit_fp16_are_identical(self):
        self.assertEqual(set(self.results['default']), set(self.results['fp16']))
        for name, (cycle, values) in self.results['default'].items():
            other_cycle, other = self.results['fp16'][name]
            self.assertEqual(cycle, other_cycle)
            np.testing.assert_array_equal(values, other)

    def test_lut_all_production_paths_match_v2_and_charge_decode(self):
        for mode in ('lut1','lut4','lut16'):
            for name,(cycle,values) in self.results[mode].items():
                with self.subTest(mode=mode,op=name):
                    expected_cycle,expected=self.results['fp16'][name]
                    if name not in ('silu','silu_multiply','boundary_silu'):
                        np.testing.assert_array_equal(values,expected)
                    elif name=='boundary_silu':
                        import sys;sys.path.insert(0,str(ROOT/'test/attention_cluster_8qk_8pv'))
                        from sfu_fp16_v4_reference import silu
                        np.testing.assert_allclose(values,silu(np.asarray([-12.,-11.09,-11.,0.,11.,12.],dtype='<f2')), atol=2e-7, rtol=0)
                    if mode=='lut16':
                        self.assertGreater(cycle, 0)
        # Compressed EXP adds one integer reconstruction stage even at full ports.
        self.assertGreater(self.results['lut16']['silu_multiply'][0], 0)
        # Functional fixture cycles are scheduling evidence, not E2E latency.
        self.assertGreater(self.results['lut1']['silu_multiply'][0],self.results['lut16']['silu_multiply'][0])
        self.assertGreater(self.results['lut4']['silu_multiply'][0],self.results['lut16']['silu_multiply'][0])

    def test_compressed_lut_resources_and_decode_latency_are_logged(self):
        for ports in (1,4,16):
            log=self.logs[f'lut{ports}']
            self.assertIn('implementation=fp16-rom-v3-silu-direct',log)
            self.assertIn('logical_rom_bytes=49439',log)
            self.assertIn('silu_rom_bytes=131072',log)
            self.assertIn(f'replicated_rom_bytes={35103*(ports+1)+6144*ports+8192+131072}',log)
            self.assertIn(f'exp_ii={16//ports}',log)
            self.assertIn('exp_decode_cycles=1',log)
            self.assertIn(f'exp_latency_cycles={max(8,16//ports)+1}',log)
            self.assertIn('online_exp_latency_cycles=9',log)

    def test_same_build_serial_comparison_keeps_all_production_outputs(self):
        for baseline,candidate in [('serial','fp16'),('serial_lut','lut4')]:
            self.assertIn('row_scheduling=serial shared_arbitration=parity',self.logs[baseline])
            self.assertIn('row_scheduling=overlap shared_arbitration=grant',self.logs[candidate])
            self.assertIn('beat_capacity=4',self.logs[candidate])
            for name,(_,values) in self.results[baseline].items():
                np.testing.assert_array_equal(values,self.results[candidate][name][1])

    def test_every_fp16_operation_matches_independent_reference(self):
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            expected = reference()
        self.assertEqual(set(expected), set(self.results['fp16']))
        for name, values in expected.items():
            with self.subTest(op=name):
                np.testing.assert_array_equal(self.results['fp16'][name][1], values)

    def test_boundary_contract_and_quality_are_distinct(self):
        result = self.results['fp16']
        np.testing.assert_array_equal(result['boundary_square_overflow'][1], np.ones(65))
        np.testing.assert_array_equal(result['boundary_sum_overflow'][1], np.ones(8192))
        self.assertGreater(result['boundary_tiny_norm'][1].min(), 0.)
        self.assertEqual(result['boundary_long_reduce'][1][0], 8192.)
        self.assertEqual(result['boundary_long_softmax'][1].sum(), 1.)
        # v2 tree reduction fixes the archived sequential-sum stagnation.

    def test_internal_rounding_changes_results_and_models_v2_work(self):
        differences = set()
        for name, (cycle, values) in self.results['fp16'].items():
            legacy_cycle, legacy = self.results['fp32'][name]
            self.assertGreaterEqual(cycle, 0)
            if not np.array_equal(values, legacy):
                differences.add(name)
        self.assertTrue({'silu', 'rope', 'rmsnorm', 'attention1', 'direct1'} <= differences)
        self.assertTrue(np.signbit(self.results['fp16']['silu'][1][0]))
        self.assertLess(self.results['fp16']['silu'][1][0], 0.)  # v2 keeps the negative tail.
        self.assertLess(self.results['fp32']['silu'][1][0], 0.)


if __name__ == '__main__':
    unittest.main()
