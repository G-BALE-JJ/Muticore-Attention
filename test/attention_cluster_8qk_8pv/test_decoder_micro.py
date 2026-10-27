import copy
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path

from run_decoder_micro import compare, timing, sfu_work, configured_environment, down_partial_audit, group_audit


class LocalDecoderAuditTests(unittest.TestCase):
    def test_launch_uses_full_runner_dma_and_fp16_profile(self):
        args=SimpleNamespace(shape='small',workers=1,with_down=True,no_prefetch=False,mpi_ranks=1)
        env=configured_environment(Path('/tmp/decoder-micro-config-test'),args,False,16)
        for key,value in {'GOLEM_DMA_READ_RETRY_TICKS':'4096','GOLEM_DMA_READ_MAX_RETRIES':'32',
                          'GOLEM_ROCC_TYPE':'golem.RoCCAnalogFloat','GOLEM_ARRAY_TYPE':'golem.MVMFloatArray',
                          'GOLEM_GLOBAL_STRIDE_KB':'2048','GOLEM_ARRAY_OPERAND_CONTEXT_BANKS':'1',
                          'GOLEM_MATRIX_BROADCAST_MAX_FANOUT':'64','GOLEM_LOCAL_GM_BYTES_PER_CYCLE':'256'}.items():
            self.assertEqual(env[key],value,key)

    def fixture(self, root, ack=199, first_output=0, weight_read=1048576, completion=True):
        (root/'logs').mkdir();(root/'stdout').mkdir()
        m={'hidden':256,'layout':{'node_stride':134217728,'shared':{'norm2':4096,'gated':1048576}}}
        log=root/'logs/attention-sst.log'
        log.write_text(f'[DECODER_OP] event=done core=4 stage=4 op=8 epoch=1 start_tick=100 tick=200 status=0 hbm_read={weight_read+32768} hbm_write=131072\n')
        base=134217728+1048576
        log.with_suffix('.resources.jsonl').write_text(
            '[RESOURCE_INTERVAL] resource=dma_read_transport core=4 start_tick=101 done_tick=110 address=134221824 bytes=32768 ok=1\n'
            f'[RESOURCE_INTERVAL] resource=dma_read_transport core=4 start_tick=111 done_tick=130 address=200000000 bytes={weight_read} ok=1\n'
            f'[RESOURCE_INTERVAL] resource=dma_write_transport core=4 start_tick=131 done_tick={ack} address={base+first_output} bytes={131072-first_output} ok=1\n')
        if completion:(root/'stdout/stdout-100').write_text('DECODER_MICRO core=4 status=0 workers=1 rows=64 hidden=256 fused=1 down=0\n')
        return m

    def test_scope_ends_at_real_output_ack(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);m=self.fixture(root)
            result=timing(root,m,1,True,False)
            self.assertEqual(result['cycles'],.098)
            self.assertEqual(result['last_output_ack_tick'],199)

    def test_delayed_ack_cannot_finish_early(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);m=self.fixture(root,ack=201)
            with self.assertRaisesRegex(ValueError,'ACK outside'):timing(root,m,1,True,False)

    def test_output_hole_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);m=self.fixture(root,first_output=128)
            with self.assertRaisesRegex(ValueError,'incomplete'):timing(root,m,1,True,False)

    def test_skipped_dense_weight_traffic_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);m=self.fixture(root,weight_read=1040384)
            with self.assertRaisesRegex(ValueError,'dense weight'):timing(root,m,1,True,False)

    def test_missing_guest_completion_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);m=self.fixture(root,completion=False)
            with self.assertRaisesRegex(ValueError,'guest completion'):timing(root,m,1,True,False)

    def test_comparison_rejects_changed_inputs_hardware_and_work(self):
        base={'status':'PASS','shape':{'workers':1},'dma_timing':{'cycles':100,'hbm_read_bytes':32,'hbm_write_bytes':16},
              'sst_wall_seconds':1,'sfu_work':{'4.exp':64},'provenance':{'guest_sha256':'g','sst_library_sha256':'l',
              'source_sha256':{},'input_sha256':'i','environment':{'GOLEM_LOCAL_GM_READ_PORTS':'1'}}}
        candidate=copy.deepcopy(base);candidate['dma_timing']['cycles']=80
        self.assertAlmostEqual(compare({'baseline':base,'fused8':candidate})['cases']['fused8']['saved_percent'],20)
        for kind in ('inputs','hardware','sfu'):
            bad=copy.deepcopy(candidate)
            if kind=='inputs':bad['provenance']['input_sha256']='different'
            elif kind=='hardware':bad['provenance']['environment']['GOLEM_LOCAL_GM_READ_PORTS']='2'
            else:bad['sfu_work']['4.exp']=0
            with self.assertRaises(ValueError):compare({'baseline':base,'fused8':bad})

    def test_missing_sfu_statistics_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            with self.assertRaisesRegex(ValueError,'SFU EXP'):sfu_work(root)

    def test_down_partial_requires_all_panels_bytes_and_restores(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'logs').mkdir()
            log=root/'logs/attention-sst.log'
            fields={'core':4,'stage':4,'epoch':1,'panels_consumed':16,
                    'cbuffer_bytes':262144,'cbuffer_read_bytes':3932160,
                    'cbuffer_write_bytes':3932160,'restores':480}
            def write(values):
                log.write_text('[DECODER_DOWN_FLOW] '+' '.join(f'{k}={v}' for k,v in values.items())+'\n')
            write(fields)
            self.assertEqual(down_partial_audit(root,2048,1,8)['status'],'PASS')
            for key in ('panels_consumed','cbuffer_bytes','cbuffer_read_bytes','cbuffer_write_bytes','restores'):
                bad=dict(fields);bad[key]-=1;write(bad)
                with self.assertRaisesRegex(ValueError,'partial SRAM'):down_partial_audit(root,2048,1,8)
            write(fields)
            log.write_text(log.read_text()*2)
            with self.assertRaisesRegex(ValueError,'missing/duplicate'):down_partial_audit(root,2048,1,8)

    def test_groups_require_full_dense_MAC_and_broadcast_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'logs').mkdir()
            log=root/'logs/attention-sst.log'
            tiles=12288
            fields=dict(core=4,stage=4,epoch=1,groups=2,launches=tiles*2,
                        compute_array_cycles=tiles*64*67,program_cycles=tiles*2*39)
            def write(values):
                log.write_text('[DECODER_GROUP_FLOW] '+' '.join(f'{k}={v}' for k,v in values.items())+'\n')
            write(fields);self.assertEqual(group_audit(root,2048,1,2)['status'],'PASS')
            for key in ('launches','compute_array_cycles','program_cycles'):
                bad=dict(fields);bad[key]-=1;write(bad)
                with self.assertRaises(ValueError):group_audit(root,2048,1,2)
            write(fields);log.write_text(log.read_text()*2)
            with self.assertRaisesRegex(ValueError,'missing/duplicate'):group_audit(root,2048,1,2)


if __name__=='__main__':unittest.main()
