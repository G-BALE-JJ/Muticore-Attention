import unittest

from analyze_resource_timeline import intersection, summarize


def interval(resource='local_read', request=1, tag=7, generation=3,
             submit=10, start=20, done=40, visible=45, domain='gm'):
    return (f'[RESOURCE_INTERVAL] domain={domain} resource={resource} core=4 request={request} '
            f'tag={tag} generation={generation} submit_tick={submit} start_tick={start} '
            f'done_tick={done} visible_tick={visible} bytes=256 queue_depth=2 '
            f'address=1000 client=0 traffic=-1 ok=1')


class ResourceTimelineTests(unittest.TestCase):
    def summarize(self, lines):
        return summarize(['[RESOURCE_TIMEBASE] core=4 ticks_per_cycle=10'] + lines)

    def test_separates_wait_service_and_visibility(self):
        result = self.summarize([interval()])['resources']['gm.local_read']
        self.assertEqual(result['queue_cycles'], 1)
        self.assertEqual(result['service_cycles'], 2)
        self.assertEqual(result['visibility_delay_cycles'], 0.5)

    def test_union_overlap_does_not_double_count_parallel_arrays(self):
        self.assertEqual(intersection([(10, 30), (15, 35)], [(20, 40)]), 15)

    def test_rejects_duplicate_id(self):
        with self.assertRaisesRegex(ValueError, 'duplicate resource'):
            self.summarize([interval(), interval()])

    def test_rejects_negative_queue(self):
        with self.assertRaisesRegex(ValueError, 'timestamp order'):
            self.summarize([interval(submit=30)])

    def test_checks_arrival_before_cache_visibility(self):
        result = self.summarize([
            interval('dma_read_transport', done=40, visible=40),
            interval('dma_read_landing', submit=40, start=40, done=60, visible=60)])
        self.assertEqual(result['dma_arrival_landing_pairs'], 1)

    def test_rejects_missing_landing(self):
        with self.assertRaisesRegex(ValueError, 'missing DMA'):
            self.summarize([interval('dma_read_transport')])

    def test_rejects_cross_generation_landing(self):
        with self.assertRaisesRegex(ValueError, 'mismatched DMA generation'):
            self.summarize([
                interval('dma_read_transport', done=40, visible=40),
                interval('dma_read_landing', generation=4, submit=40, start=40, done=60, visible=60)])

    def test_rank_order_does_not_change_intervals(self):
        a, b = interval(request=1), interval(request=2, submit=50, start=60, done=80, visible=85)
        self.assertEqual(self.summarize([a, b]), self.summarize([b, a]))

    def test_rejects_missing_timebase(self):
        with self.assertRaisesRegex(ValueError, 'RESOURCE_TIMEBASE'):
            summarize([interval()])

    def test_control_trace_measures_release_and_guest_setup(self):
        controls = [f'[CONTROL_EVENT] stage={stage} core=4 target=4 tick={tick} job=0 address=1000'
                    for stage, tick in [('projection_release_issue', 100),
                                        ('projection_release_observed', 120),
                                        ('projection_accept', 620)]]
        result = self.summarize([interval()] + controls)['worker_control']['4']
        self.assertEqual(result['release_to_observed_cycles'], 2)
        self.assertEqual(result['observed_to_accept_cycles'], 50)

    def test_output_ack_attribution_with_unknown_traffic_kind(self):
        output = interval('dma_write_transport').replace('traffic=-1', 'traffic=0')
        panel = interval('dma_write_transport', request=2, done=80, visible=85).replace('address=1000', 'address=2000')
        root = '[ATTENTION_MILESTONE] stage=root_tensor_complete status=done sst_tick=50'
        result = summarize(['[RESOURCE_TIMEBASE] core=4 ticks_per_cycle=10', output, panel, root],
                           output_ranges=[(1000, 1256)])
        self.assertEqual(result['attention_output_ack_count'], 1)
        self.assertEqual(result['attention_output_ack_bytes'], 256)
        self.assertEqual(result['output_ack_to_root_cycles'], 0.5)

    def test_output_range_requires_full_request_to_fit(self):
        result = summarize(['[RESOURCE_TIMEBASE] core=4 ticks_per_cycle=10', interval('dma_write_transport')],
                           output_ranges=[(1000, 1100)])
        self.assertEqual(result['attention_output_ack_count'], 0)

    def test_projection_overlap_excludes_later_attention_work(self):
        lines = ['[PROJECTION_JOB] core=4 start=1 end=5',
                 interval('compute', domain='array', submit=20, start=20, done=40, visible=40),
                 interval(submit=30, start=30, done=60, visible=60),
                 interval('compute', domain='array', request=2, submit=70, start=70, done=90, visible=90)]
        result = self.summarize(lines)['projection_worker_overlap']['4']
        self.assertEqual(result['compute_busy_cycles'], 2)
        self.assertEqual(result['gm_read_busy_cycles'], 2)
        self.assertEqual(result['gm_read_compute_overlap_cycles'], 1)


if __name__ == '__main__':
    unittest.main()
