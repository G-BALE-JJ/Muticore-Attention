import copy
import unittest
from verify_pv_wo_direct import audit_requests,LANDING,TILE_BYTES,WEIGHTS


class DirectAuditTests(unittest.TestCase):
    def fixture(self):
        states={(1,4):dict(core=4,start_tick=1000,end_tick=5000,release_tick=4500,n_tiles=1)}
        def event(offset,size):
            return dict(core=4,submit_tick=2000,start_tick=2100,done_tick=3000,ok=1,
                        address=4*0x200000+offset,bytes=size,client=1)
        reads=[('local_read',event(LANDING+i,4096)) for i in range(0,TILE_BYTES,4096)]
        weights=[('dma_read_landing',event(WEIGHTS+(i%4)*8192,8192)) for i in range(32)]
        return states,reads+weights,event

    def test_actual_reads_weights_and_release(self):
        states,events,_=self.fixture();r=audit_requests(states,events)
        self.assertEqual(r['direct_a_read_bytes'],TILE_BYTES)
        self.assertEqual(r['weight_landing_bytes'],32*8192)
        self.assertEqual(r['relocation_read_bytes'],0)

    def test_missing_duplicate_and_premature_release_rejected(self):
        states,events,_=self.fixture()
        for bad in [events[1:],events+[events[0]]]:
            with self.assertRaisesRegex(ValueError,'coverage'):audit_requests(copy.deepcopy(states),bad)
        states[(1,4)]['release_tick']=2500
        with self.assertRaisesRegex(ValueError,'released'):audit_requests(states,events)

    def test_overwrite_relocation_wrong_weight_and_failed_ack_rejected(self):
        states,events,event=self.fixture()
        for resource,offset in [('local_write',LANDING),('dma_read_landing',LANDING),('local_write',WEIGHTS)]:
            with self.assertRaises(ValueError):audit_requests(copy.deepcopy(states),events+[(resource,event(offset,4096))])
        bad=copy.deepcopy(events);bad[0][1]['ok']=0
        with self.assertRaisesRegex(ValueError,'failed'):audit_requests(states,bad)

    def test_other_phase_reuse_is_outside_the_lease(self):
        states,events,event=self.fixture();e=event(LANDING,4096)
        e.update(submit_tick=6000,start_tick=6100,done_tick=6200)
        audit_requests(states,events+[('local_write',e)])
