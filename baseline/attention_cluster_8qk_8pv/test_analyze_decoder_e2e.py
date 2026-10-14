import unittest

from analyze_decoder_e2e import job_bounds, linear_bounds


class DecoderBoundTests(unittest.TestCase):
    def test_independent_ports_use_max(self):
        job = {'stage': 3}
        resources = {'local_read': {'cycles': 40000}, 'local_write': {'cycles': 50000}}
        self.assertEqual(job_bounds(job, resources)['resource_capacity'], 50000)

    def test_existing_order_is_stronger_than_capacity(self):
        for k, n, rows in ((2048, 32, 64), (2048, 128, 64), (8192, 32, 64), (2048, 8, 1)):
            b = linear_bounds(k, n, rows)
            self.assertGreater(b['serial_state_machine'], max(b['MAC'], b['array_buffer_port']))

    def test_one_row_keeps_same_K_compute_work(self):
        one, many = linear_bounds(2048, 8, 1), linear_bounds(2048, 8, 64)
        self.assertEqual(one['MAC'], many['MAC'])
        self.assertLess(one['array_buffer_port'], many['array_buffer_port'])

    def test_head_tail_uses_executed_columns(self):
        full = job_bounds({'stage': 11, 'hbm_read': 4096 + 8 * 32 * 8192}, {})
        tail = job_bounds({'stage': 11, 'hbm_read': 4096 + 6 * 32 * 8192}, {})
        self.assertEqual(full['components']['launches'], 8 * 32)
        self.assertEqual(tail['components']['launches'], 6 * 32)

    def test_SiLU_scalar_reciprocal_floor(self):
        bound = job_bounds({'stage': 6}, {})['resource_capacity']
        self.assertGreaterEqual(bound, 64 * 8192)
        self.assertLess(bound, 64 * 8192 + 2000)

    def test_work_scales_linearly_in_output_tiles(self):
        a, b = linear_bounds(2048, 32, 64), linear_bounds(2048, 128, 64)
        for key in ('launches', 'MAC', 'array_buffer_port', 'serial_state_machine'):
            self.assertEqual(b[key], a[key] * 4)


if __name__ == '__main__':
    unittest.main()
