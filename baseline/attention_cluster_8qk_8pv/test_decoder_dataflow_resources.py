import tempfile
import unittest
from pathlib import Path

from verify_decoder_dataflow_resources import verify


class DecoderDataflowResourceTests(unittest.TestCase):
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


if __name__=='__main__':unittest.main()
