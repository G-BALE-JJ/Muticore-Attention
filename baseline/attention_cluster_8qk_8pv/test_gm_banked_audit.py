import tempfile
import unittest
from pathlib import Path
from verify_gm_banked import verify


class BankEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.case=Path(self.temp.name);(self.case/'logs').mkdir()
        self.log=self.case/'logs/attention-sst.log'
        text=[]
        for core in range(20):
            text += [f'GOLEM_LOCAL_GM_STATS core={core} gmem_local_read_bytes=128 gmem_local_write_bytes=0',
                f'GOLEM_LOCAL_GM_BANK_STATS core={core} capacity=2097152 banks=8 line_bytes=128 response_cycles=1 response_capacity=32 compute_beats=1 dma_beats=0 payload_bytes=128 bank_conflict_cycles=0 response_blocked_lane_cycles=0 response_high_water=1 pending=0 queued=0']
            text += [f'GOLEM_LOCAL_GM_BANK_GRANTS core={core} bank={bank} beats={int(bank==0)}' for bank in range(8)]
        self.log.write_text('\n'.join(text)+'\n')
        self.resources=self.log.with_suffix('.resources.jsonl')
        self.resources.write_text(''.join(f'[RESOURCE_INTERVAL] resource=local_read core={core} submit_tick=625 start_tick=1000 done_tick=2000 bytes=128 address={core*2**21} client=1 ok=1\n' for core in range(20)))

    def test_split_evidence_is_required(self):
        self.assertEqual(verify(self.case)['logical_intervals_checked'],20)
        self.resources.unlink()
        with self.assertRaises(ValueError):verify(self.case)

    def test_tensor_bypass_fails(self):
        with self.resources.open('a') as f:f.write('[RESOURCE_INTERVAL] resource=untimed_write core=4 address=8454144 bytes=8192\n')
        with self.assertRaises(ValueError):verify(self.case)

    def test_broken_clock_edge_fails(self):
        self.resources.write_text(self.resources.read_text().replace('start_tick=1000','start_tick=1625'))
        with self.assertRaises(AssertionError):verify(self.case)

    def test_cross_line_undercharge_fails(self):
        # 128 bytes starting at byte 1 require two physical beats, even though
        # payload bytes still fit the old aggregate byte/beat inequality.
        self.resources.write_text(self.resources.read_text().replace('address=0 ','address=1 ',1))
        with self.assertRaises(AssertionError):verify(self.case)

    def test_pending_and_traffic_mismatch_fail(self):
        original=self.log.read_text()
        for old,new in [('pending=0','pending=1'),('payload_bytes=128','payload_bytes=129'),('capacity=2097152','capacity=524288')]:
            self.log.write_text(original.replace(old,new,1))
            with self.assertRaises(AssertionError):verify(self.case)

if __name__=='__main__':unittest.main()
