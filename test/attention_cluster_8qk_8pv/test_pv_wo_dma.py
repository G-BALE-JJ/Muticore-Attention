"""Reject apparent PV-Wo success with missing GM coverage or residual HBM IO."""
import tempfile
import unittest
from pathlib import Path
from run_decoder_stack import verify_dma


def event(resource, address, size, start, end, core=4):
    return (f'[RESOURCE_INTERVAL] resource={resource} core={core} request=1 '
            f'address={address} bytes={size} start_tick={start} done_tick={end} ok=1\n')


class PvWoDmaTest(unittest.TestCase):
    def fixture(self):
        stride=0x8000000
        manifest=dict(layers=1,sequence=1024,hidden=2048,pv_wo_gm_handoff=True,
            layout=dict(node_stride=stride,shared=dict(o=0x100000),
                        layers=[dict(input=0x200000,output=0x300000)]),
            _lifecycle=dict(stage_spans={'1':{'1':dict(start_tick=4000)}},
                            attention_start_ticks={'1':1500}))
        lines=[event('dma_read_transport',stride+0x200000,4096,1000,1100)]
        lines.extend(event('local_write',core*0x200000+0x1a0000,0x40000,2000,3000,core)
                     for core in range(4,20))
        lines.extend(event('dma_write_transport',node*stride+0x300000,0x100000,5000,6000)
                     for node in range(1,5))
        return manifest,lines

    def verify(self,manifest,lines):
        with tempfile.TemporaryDirectory() as directory:
            log=Path(directory)/'test.log'
            log.with_suffix('.resources.jsonl').write_text(''.join(lines))
            return verify_dma(log,manifest,{1:0},{1:7000})

    def test_complete_timed_gm_coverage_and_zero_o_hbm(self):
        manifest,lines=self.fixture()
        result=self.verify(manifest,lines)['layers']['1']
        self.assertEqual(result['attention_o_hbm_read_bytes'],0)
        self.assertEqual(result['attention_o_hbm_write_bytes'],0)
        self.assertEqual(result['pv_wo_gm_landing_bytes'],4*1024*1024)
        self.assertEqual(result['last_attention_o_ack_tick'],3000)

    def test_excludes_projection_and_ffn_reuse_of_the_same_gm_addresses(self):
        manifest,lines=self.fixture()
        landing=4*0x200000+0x1a0000
        lines+=[event('local_write',landing,4096,1200,1300),
                event('local_write',landing,4096,4500,4600)]
        result=self.verify(manifest,lines)['layers']['1']
        self.assertEqual(result['pv_wo_gm_landing_bytes'],4*1024*1024)

    def test_rejects_missing_or_duplicate_landing(self):
        manifest,lines=self.fixture()
        for corrupted in (lines[:1]+lines[2:],lines+[lines[1]]):
            with self.assertRaisesRegex(ValueError,'PV-Wo'):
                self.verify(manifest,corrupted)

    def test_rejects_any_o_hbm_read_or_write(self):
        manifest,lines=self.fixture()
        for resource in ('dma_read_transport','dma_write_transport'):
            with self.assertRaisesRegex(ValueError,'PV-Wo'):
                self.verify(manifest,lines+[event(resource,0x8100000,128,2000,2100)])

    def test_rejects_gm_landing_after_wo_start(self):
        manifest,lines=self.fixture()
        lines[1]=event('local_write',4*0x200000+0x1a0000,0x40000,2000,4500)
        with self.assertRaisesRegex(ValueError,'Wo started'):
            self.verify(manifest,lines)


if __name__=='__main__':
    unittest.main()
