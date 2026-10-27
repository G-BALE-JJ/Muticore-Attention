"""Budgets must follow the new last finisher instead of adding wait counters."""
import unittest
from analyze_ffn_critical_budget import fixed_start_budget


class BudgetTest(unittest.TestCase):
    def test_recomputes_last_finisher(self):
        rows=[dict(core=4,end_cycle=100,removable_cycles=40),
              dict(core=5,end_cycle=99,removable_cycles=0)]
        r=fixed_start_budget(rows)
        self.assertEqual(r['optimistic_saved_cycles'],1)
        self.assertEqual(r['optimistic_last_core'],5)
        self.assertEqual(r['current_last_core'],4)

    def test_not_sum_of_worker_waits(self):
        r=fixed_start_budget([dict(core=c,end_cycle=100,removable_cycles=20) for c in range(4,20)])
        self.assertEqual(r['optimistic_saved_cycles'],20)

    def test_negative_budget_rejected(self):
        with self.assertRaisesRegex(ValueError,'negative'):
            fixed_start_budget([dict(core=4,end_cycle=100,removable_cycles=-1)])


if __name__=='__main__':unittest.main()
