import copy
import unittest

from analyze_reuse_window_critical_path import additive_timeline
from audit_prefill_hbm import transport_ledger


def event(resource='dma_read_transport',request=1,address=1100,size=16,start=20,end=30,ok=1):
    return (f'[RESOURCE_INTERVAL] resource={resource} core=4 request={request} '
            f'address={address} bytes={size} start_tick={start} done_tick={end} visible_tick={end} ok={ok}\n')


class TransportAuditTests(unittest.TestCase):
    layout=dict(node_stride=1000,regions=[dict(name='o',layer=0,offset=100,bytes=64),
                                         dict(name='gate_weight',layer=1,offset=200,bytes=64)])

    def test_counts_transport_once_without_landing_or_window_double_count(self):
        lines=[event(),event(resource='dma_read_landing'),event(resource='local_read'),
               event(resource='dma_write_transport'),event(request=2,address=1200)]
        r=transport_ledger(lines,self.layout,10,100)
        self.assertEqual(r['totals'],dict(read_bytes=32,write_bytes=16,read_requests=2,write_requests=1))
        self.assertEqual(r['categories']['weights_and_coefficients']['read_bytes'],16)
        self.assertEqual(r['regions']['0.o']['read_workers'],[4])

    def test_rejects_duplicate_or_failed_completion(self):
        for lines in ([event(),event()],[event(ok=0)]):
            with self.assertRaises(ValueError):transport_ledger(lines,self.layout,10,100)

    def test_rejects_unmapped_or_crossing_region(self):
        for address,size in ((1090,16),(1150,16),(1100,0)):
            with self.assertRaises(ValueError):transport_ledger([event(address=address,size=size)],self.layout,10,100)

    def test_excludes_unrelated_time_and_rejects_straddling_transfer(self):
        r=transport_ledger([event(start=1,end=2),event(start=110,end=120)],self.layout,10,100)
        self.assertEqual(r['excluded_outside_timing'],2)
        self.assertEqual(r['totals']['read_bytes'],0)
        with self.assertRaisesRegex(ValueError,'boundary'):
            transport_ledger([event(start=1,end=20)],self.layout,10,100)

    def test_records_root_completion_poll_crossing_each_timing_edge(self):
        layout=copy.deepcopy(self.layout)
        layout['regions'].append(dict(name='lm_done',layer=0,offset=300,bytes=8))
        lines=[event(address=1300,size=8,start=1,end=20).replace('core=4','core=0'),
               event(address=1300,size=8,request=2,start=90,end=110).replace('core=4','core=0')]
        r=transport_ledger(lines,layout,10,100)
        self.assertEqual(len(r['excluded_boundary_control_polls']),2)
        self.assertEqual(r['totals']['read_bytes'],0)


class OverlappingTimelineTests(unittest.TestCase):
    result=dict(embedding=dict(first_token_ids_dma_tick=0,last_embedding_output_ack_tick=1000),
                dma_timing=dict(layers={'1':dict(first_input_dma_tick=2000,last_attention_o_ack_tick=3000,last_y_ack_tick=10000)}),
                final_norm=dict(first_input_dma_tick=7000,last_output_ack_tick=12000),
                lm_head=dict(first_input_dma_tick=13000,last_logits_ack_tick=15000))

    def test_eager_norm_is_charged_only_exposed_tail(self):
        phases,overlap=additive_timeline(self.result)
        self.assertEqual(sum(p['cycles'] for p in phases.values()),15)
        self.assertEqual(phases['before_final_norm']['cycles'],0)
        self.assertEqual(phases['final_norm']['cycles'],2)
        self.assertEqual(overlap['final_norm_overlap_with_decoder_cycles'],3)

    def test_serial_norm_keeps_idle_gap(self):
        r=copy.deepcopy(self.result);r['final_norm']['first_input_dma_tick']=11000
        phases,overlap=additive_timeline(r)
        self.assertEqual(phases['before_final_norm']['cycles'],1)
        self.assertEqual(phases['final_norm']['cycles'],1)
        self.assertEqual(overlap['final_norm_overlap_with_decoder_cycles'],0)

    def test_rejects_lm_before_all_norm_outputs(self):
        r=copy.deepcopy(self.result);r['lm_head']['first_input_dma_tick']=11000
        with self.assertRaisesRegex(ValueError,'partition'):additive_timeline(r)


if __name__=='__main__':unittest.main()
