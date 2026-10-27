import unittest
from analyze_dma_lower_bound import bounds


class CapacityBoundTest(unittest.TestCase):
    def test_fused_norm_and_compute_not_added(self):
        r=bounds({'norm':27952,'compute':101376,'buffer':112176},
                 {'EXP':139264,'MAC':42240},213552)
        self.assertEqual(r['barrier_capacity_cycles'],251440)
        self.assertEqual(r['group1_serialized_and_barrier_cycles'],352816)

    def test_other_resource_can_dominate_serialized_arrays(self):
        r=bounds({'GM_read':500,'arrays':100},{'EXP':70,'GM_read':90},200)
        self.assertEqual(r['group1_serialized_and_barrier_cycles'],590)

    def test_s2048_actual_exp_work_capacity(self):
        r=bounds({'buffer':224352,'compute':202752},
                 {'EXP':540672,'MAC':143616},427104)
        self.assertEqual(r['barrier_capacity_cycles'],765024)
        self.assertEqual(r['group1_serialized_and_barrier_cycles'],967776)


if __name__ == '__main__':unittest.main()
