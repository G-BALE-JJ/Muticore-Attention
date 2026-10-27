import tempfile
import json
import unittest
from pathlib import Path

from verify_decoder_dataflow_resources import verify, residual_audit, block_pipeline_audit, optimization_audit


class DecoderDataflowResourceTests(unittest.TestCase):
    def test_stream_and_handoff_require_real_capacity_drain_and_complete_pair(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'logs').mkdir();log=root/'logs/attention-sst.log'
            (root/'provenance.json').write_text(json.dumps({'environment':{
                'GOLEM_DECODER_FFN_SFU_STREAM':'1','GOLEM_DECODER_NORM_FFN_HANDOFF':'1'}}))
            jobs={4:[dict(core=4,epoch=1,stage=3,op=3),dict(core=4,epoch=1,stage=4,op=9)]}
            stream='[DECODER_SFU_STREAM] core=4 epoch=1 stage=4 enabled=1 contexts=2 chunk_elements=8192 high_water=2 io_compute_overlap=10 drained=1\n'
            producer='[DECODER_NORM_HANDOFF] core=4 epoch=1 stage=3 producer=1 consumer=0 cache_bytes=262144 saved_hbm_read=0 materialized=1 capacity_added=0\n'
            consumer='[DECODER_NORM_HANDOFF] core=4 epoch=1 stage=4 producer=0 consumer=1 cache_bytes=0 saved_hbm_read=262144 materialized=1 capacity_added=0\n'
            good=stream+producer+consumer;log.write_text(good)
            result,savings=optimization_audit(root,jobs,dict(hidden=2048,sequence=1024))
            self.assertEqual(result['saved_hbm_read_bytes'],262144)
            self.assertEqual(savings,{(4,1):262144})
            for bad in (stream+consumer,good+producer,good.replace('high_water=2','high_water=3'),
                        good.replace('drained=1','drained=0'),good.replace('saved_hbm_read=262144','saved_hbm_read=8192')):
                log.write_text(bad)
                with self.assertRaises(ValueError):optimization_audit(root,jobs,dict(hidden=2048,sequence=1024))

    def test_block_pipeline_requires_complete_lifecycle_and_released_array_owner(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'logs').mkdir();log=root/'logs/attention-sst.log'
            (root/'provenance.json').write_text(json.dumps({'environment':{
                'GOLEM_DECODER_FFN_BLOCK_PIPELINE':'1','GOLEM_DECODER_FFN_PANEL_TILES':'16'}}))
            jobs={4:[dict(core=4,epoch=1,stage=4,op=9)]}
            good=('[DECODER_BLOCK_PIPELINE] core=4 epoch=1 stage=4 panels=8 both_active=100 '
                  'producer_transfer_down_compute=20 down_transfer_producer_compute=30 '
                  'arrays=64 contexts=1 extra_gm_capacity=0 owner_at_finish=0 pending_at_finish=0\n')
            log.write_text(good)
            self.assertEqual(block_pipeline_audit(root,jobs)['both_active_cycles'],100)
            for bad in ('',good+good,good.replace('owner_at_finish=0','owner_at_finish=1'),
                        good.replace('contexts=1','contexts=2'),good.replace('pending_at_finish=0','pending_at_finish=1')):
                log.write_text(bad)
                with self.assertRaises(ValueError):block_pipeline_audit(root,jobs)
            (root/'provenance.json').write_text(json.dumps({'environment':{
                'GOLEM_DECODER_FFN_BLOCK_PIPELINE':'1','GOLEM_DECODER_FFN_PIPELINE_POLICY':'1'}}))
            log.write_text(good.strip()+' policy=1\n')
            self.assertEqual(block_pipeline_audit(root,jobs)['status'],'PASS')
            log.write_text(good.strip()+' policy=2\n')
            with self.assertRaisesRegex(ValueError,'policy differs'):block_pipeline_audit(root,jobs)

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
