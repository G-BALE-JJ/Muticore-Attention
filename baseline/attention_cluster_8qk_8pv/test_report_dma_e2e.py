import unittest
from report_dma_e2e import summarize_events


def event(resource, address, size, submit, start, done, core=4):
    return dict(domain='gm', resource=resource, core=core, address=address,
                bytes=size, submit_tick=submit, start_tick=start,
                done_tick=done, visible_tick=done, ok=1)


class DmaBoundaryTest(unittest.TestCase):
    def report(self, events, factor=1000):
        return summarize_events(events, [(100, 200)], [(300, 400)], factor, 9000)

    def test_metadata_excluded_and_first_send_used(self):
        r = self.report([event('dma_read_transport', 10, 8, 1, 1, 2),
                         event('dma_read_transport', 100, 100, 1000, 2000, 3000),
                         event('dma_write_transport', 300, 100, 4000, 4000, 8000)])
        self.assertEqual(r['end_to_end_cycles'], 6)
        self.assertEqual(r['initial_dma_admission_delay_cycles'], 1)

    def test_all_output_nodes_and_fractional_ticks(self):
        r = self.report([event('dma_read_transport', 100, 100, 1000, 1000, 2000),
                         event('dma_write_transport', 300, 50, 4000, 4000, 7000),
                         event('dma_write_transport', 350, 50, 4000, 4000, 8125)])
        self.assertEqual(r['end_to_end_cycles'], 7.125)

    def test_missing_output_tail_rejected(self):
        with self.assertRaisesRegex(ValueError, 'coverage'):
            self.report([event('dma_read_transport', 100, 100, 1000, 1000, 2000),
                         event('dma_write_transport', 300, 50, 4000, 4000, 8000)])

    def test_wrong_clock_not_silently_accepted(self):
        with self.assertRaisesRegex(ValueError, 'timebase'):
            self.report([], factor=0)

    def test_output_ack_before_input_rejected(self):
        with self.assertRaisesRegex(ValueError, 'ordering'):
            self.report([event('dma_read_transport', 100, 100, 8000, 8000, 8500),
                         event('dma_write_transport', 300, 100, 4000, 4000, 7000)])


if __name__ == '__main__':
    unittest.main()
