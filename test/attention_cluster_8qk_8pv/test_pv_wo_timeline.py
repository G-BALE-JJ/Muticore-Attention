import unittest
from analyze_pv_wo_timeline import span


class TimelineTests(unittest.TestCase):
    def test_overlapping_requests_are_not_added_as_elapsed_time(self):
        events=[dict(bytes=4096,submit_tick=1000,start_tick=2000,done_tick=6000,ok=1),
                dict(bytes=4096,submit_tick=2000,start_tick=3000,done_tick=7000,ok=1)]
        r=span(events)
        self.assertEqual(r['span_cycles'],6)
        self.assertEqual(r['queue_request_sum_cycles'],2)
        self.assertEqual(r['service_request_sum_cycles'],8)
        self.assertEqual(r['bytes'],8192)

    def test_failed_and_reversed_requests_are_rejected(self):
        for event in [dict(bytes=4,submit_tick=1,start_tick=2,done_tick=3,ok=0),
                      dict(bytes=4,submit_tick=3,start_tick=2,done_tick=4,ok=1),
                      dict(bytes=4,submit_tick=1,start_tick=3,done_tick=2,ok=1)]:
            with self.assertRaises(ValueError):span([event])
