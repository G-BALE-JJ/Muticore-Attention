import tempfile
import json
import unittest
from pathlib import Path

from verify_decoder_dataflow_resources import verify, residual_audit


class DecoderDataflowResourceTests(unittest.TestCase):
    def test_residual_epilogue_lifecycle_rejects_missing_duplicate_and_wrong_traffic(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'logs').mkdir();log=root/'logs/attention-sst.log'
            good='[DECODER_RESIDUAL_FLOW] core=4 stage=4 epoch=1 chunks=32 residual_hbm_bytes=262144 gemm_overlap_cycles=100 scratch_bytes=8192\n'
            for content in (good,'',good+good,good.replace('262144','131072'),good.replace('8192','16384')):
                log.write_text(content)
                if content==good:self.assertEqual(residual_audit(root,{(4,1,4)},2048)['verified_jobs'],1)
                else:
                    with self.assertRaises(ValueError):residual_audit(root,{(4,1,4)},2048)
            # A 768-column panel fits only five complete FP16 rows per 8 KiB.
            log.write_text(good.replace('chunks=32','chunks=13').replace('262144','98304'))
            self.assertEqual(residual_audit(root,{(4,1,4)},768)['verified_jobs'],1)

    def test_residual_fusion_cannot_skip_h1_or_write_unfused_down(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'logs').mkdir()
            (root/'decoder_stack_inputs.json').write_text(json.dumps(
                {'ffn_panel_fusion':True,'ffn_down_fusion':True,'down_residual_fusion':True,'hidden':2048}))
            read=3*128*32*8192+2*64*2048*2;write=64*2048*2+8
            for extra_read,extra_write in ((0,0),(-64*2048*2,0),(0,64*2048*2)):
                jobs=[];resources=[]
                for epoch in (1,2):
                    start=epoch*1000;end=start+100
                    for core in range(4,20):
                        jobs.append(f'[DECODER_OP] event=done core={core} stage=4 op=9 epoch={epoch} status=0 start_tick={start} tick={end} hbm_read={read+extra_read} hbm_write={write+extra_write}\n')
                        jobs.append(f'[DECODER_RESIDUAL_FLOW] core={core} stage=4 epoch={epoch} chunks=32 residual_hbm_bytes=262144 gemm_overlap_cycles=100 scratch_bytes=8192\n')
                        resources.append(f'[RESOURCE_INTERVAL] resource=dma_read_transport core={core} start_tick={start+1} done_tick={end-1} bytes={read+extra_read}\n')
                        resources.append(f'[RESOURCE_INTERVAL] resource=dma_write_transport core={core} start_tick={start+2} done_tick={end-1} bytes={write+extra_write}\n')
                (root/'logs/attention-sst.log').write_text(''.join(jobs))
                (root/'logs/attention-sst.resources.jsonl').write_text(''.join(resources))
                if extra_read or extra_write:
                    with self.assertRaisesRegex(ValueError,'only final residual'):verify(root)
                else:self.assertTrue(verify(root)['down_residual_only_final_DMA'])

    def fixture(self, root, read_bytes=8192, write_end=190):
        (root/'logs').mkdir()
        log=root/'logs/attention-sst.log'
        log.write_text('[DECODER_OP] event=done core=4 stage=1 epoch=1 status=0 start_tick=100 tick=200 hbm_read=8192 hbm_write=2048\n')
        log.with_suffix('.resources.jsonl').write_text(
            f'[RESOURCE_INTERVAL] resource=dma_read_transport core=4 start_tick=110 done_tick=130 bytes={read_bytes}\n'
            f'[RESOURCE_INTERVAL] resource=dma_write_transport core=4 start_tick=160 done_tick={write_end} bytes=2048\n'
            '[RESOURCE_INTERVAL] resource=local_read core=4 start_tick=300 done_tick=400 bytes=8192\n')

    def test_projection_between_jobs_does_not_belong_to_finished_decoder(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            self.assertEqual(verify(root)['verified_decoder_jobs'],1)

    def test_missing_prefetch_bytes_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root,read_bytes=4096)
            with self.assertRaisesRegex(ValueError,'byte mismatch'):verify(root)

    def test_ACK_after_job_completion_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root,write_end=201)
            with self.assertRaisesRegex(ValueError,'crosses job completion'):verify(root)

    def test_fused_producer_cannot_hide_extra_gate_output_or_skipped_weights(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'logs').mkdir()
            (root/'decoder_stack_inputs.json').write_text(json.dumps({'ffn_panel_fusion':True,'hidden':256}))
            read=2*16*4*8192+64*256*2;write=64*1024*2
            for extra_read,extra_write in ((0,0),(-8192,0),(0,131072)):
                jobs=[];resources=[]
                for epoch in (1,2):
                    start=epoch*1000;end=start+100
                    for core in range(4,20):
                        jobs.append(f'[DECODER_OP] event=done core={core} stage=4 op=8 epoch={epoch} status=0 start_tick={start} tick={end} hbm_read={read+extra_read} hbm_write={write+extra_write}\n')
                        resources.append(f'[RESOURCE_INTERVAL] resource=dma_read_transport core={core} start_tick={start+1} done_tick={end-1} bytes={read+extra_read}\n')
                        resources.append(f'[RESOURCE_INTERVAL] resource=dma_write_transport core={core} start_tick={start+2} done_tick={end-1} bytes={write+extra_write}\n')
                (root/'logs/attention-sst.log').write_text(''.join(jobs))
                (root/'logs/attention-sst.resources.jsonl').write_text(''.join(resources))
                if extra_read or extra_write:
                    with self.assertRaisesRegex(ValueError,'only gated'):verify(root)
                else:self.assertTrue(verify(root)['ffn_panel_only_gated_DMA'])

    def test_fused_down_cannot_hide_gated_write_or_skip_dense_weights(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'logs').mkdir()
            (root/'decoder_stack_inputs.json').write_text(json.dumps(
                {'ffn_panel_fusion':True,'ffn_down_fusion':True,'hidden':2048}))
            read=3*128*32*8192+64*2048*2;write=64*2048*2
            for extra_read,extra_write in ((0,0),(-8192,0),(0,64*8192*2)):
                jobs=[];resources=[]
                for epoch in (1,2):
                    start=epoch*1000;end=start+100
                    for core in range(4,20):
                        jobs.append(f'[DECODER_OP] event=done core={core} stage=4 op=9 epoch={epoch} status=0 start_tick={start} tick={end} hbm_read={read+extra_read} hbm_write={write+extra_write}\n')
                        resources.append(f'[RESOURCE_INTERVAL] resource=dma_read_transport core={core} start_tick={start+1} done_tick={end-1} bytes={read+extra_read}\n')
                        resources.append(f'[RESOURCE_INTERVAL] resource=dma_write_transport core={core} start_tick={start+2} done_tick={end-1} bytes={write+extra_write}\n')
                (root/'logs/attention-sst.log').write_text(''.join(jobs))
                (root/'logs/attention-sst.resources.jsonl').write_text(''.join(resources))
                if extra_read or extra_write:
                    with self.assertRaisesRegex(ValueError,'only down'):verify(root)
                else:self.assertTrue(verify(root)['ffn_panel_only_down_DMA'])


if __name__=='__main__':unittest.main()
