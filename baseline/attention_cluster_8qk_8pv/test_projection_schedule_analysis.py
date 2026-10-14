import unittest
from analyze_projection_schedule import PHASES, partition


class CyclePartitionTest(unittest.TestCase):
    def example(self):
        fields = dict.fromkeys(PHASES, 0)
        fields.update(local_read=120, local_weight_read=40, local_input_read=60,
                      local_fusion_read=20, local_other_read=0,
                      norm_wait=40, launch=200, read_output=80, wrap_prefetch=1)
        return fields

    def test_role_counters_are_subdivisions_not_additional_cycles(self):
        groups = partition(self.example())
        self.assertEqual(sum(groups.values()), 440)
        self.assertEqual(groups['foreground_input_gm_read'], 60)
        self.assertEqual(groups['array_compute_and_launch'], 200)

    def test_rejects_unaccounted_gm_cycles(self):
        fields = self.example()
        fields['local_read'] += 1
        with self.assertRaisesRegex(ValueError, 'cycle partition'):
            partition(fields)


if __name__ == '__main__':
    unittest.main()
