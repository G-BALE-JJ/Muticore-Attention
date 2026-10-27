import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from run_lm_head_micro import lm_environment, compare, TESTS

class LmMicroTests(unittest.TestCase):
    def test_guest_config_is_forwarded_by_production_topology(self):
        args=SimpleNamespace(shape='small',workers=1,with_down=False,no_prefetch=False,mpi_ranks=1,prefetch_slots=4,weight_trace=True)
        env=lm_environment(Path('/tmp/lm-micro-test'),args,dict(hidden=256,vocab=13689),8)
        tree=ast.parse((TESTS/'architecture/ncores_selfcom_dma_ctrl.py').read_text())
        keys=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='process_env_keys' for t in n.targets))
        self.assertIn('GOLEM_DECODER_MICRO_CONFIG',keys)
        self.assertEqual(env['GOLEM_DECODER_MICRO_CONFIG'],'1,256,13689,256')
        self.assertEqual(env['GOLEM_DECODER_LM_N_PARALLEL'],'8')
        self.assertEqual(env['GOLEM_ARRAY_OPERAND_CONTEXT_BANKS'],'1')

    def test_comparison_rejects_hardware_build_and_dense_work_changes(self):
        base=dict(shape={},measurement=dict(cycles=100,dense_weight_tiles=12,hbm_read_bytes=128,hbm_write_bytes=64,verified_elements=64),sst_wall_seconds=1,
                  provenance=dict(input_sha256='i',source_sha256={},guest_sha256='g',sst_library_sha256='l',environment={'GOLEM_ARRAY_OPERAND_CONTEXT_BANKS':'1','GOLEM_DECODER_LM_N_PARALLEL':'1'}))
        candidate=copy.deepcopy(base);candidate['measurement']['cycles']=80;candidate['provenance']['environment']['GOLEM_DECODER_LM_N_PARALLEL']='8'
        self.assertAlmostEqual(compare({'1':base,'8':candidate})['cases']['8']['saved_percent'],20)
        for change in ('hardware','build','work','placement'):
            bad=copy.deepcopy(candidate)
            if change=='hardware':bad['provenance']['environment']['GOLEM_ARRAY_OPERAND_CONTEXT_BANKS']='2'
            elif change=='build':bad['provenance']['sst_library_sha256']='changed'
            elif change=='work':bad['measurement']['dense_weight_tiles']-=1
            else:bad['provenance']['command']=['mpirun','-np','16']
            with self.assertRaises(AssertionError):compare({'1':base,'8':bad})

if __name__=='__main__':unittest.main()
