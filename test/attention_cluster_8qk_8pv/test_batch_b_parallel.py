import unittest
from run_batch_b_parallel import SHARDS, validate_host_sets
from run_batch_b import cases_for


class HostQueueTests(unittest.TestCase):
    def test_shards_cover_each_nonbaseline_case_once(self):
        cases=[c for shard in SHARDS for c in shard]
        self.assertEqual(len(cases),len(set(cases)))
        self.assertEqual(set(cases),{n for n,_,_ in cases_for('all')} - {'baseline'})

    def test_accepts_disjoint_physical_host_cores(self):
        validate_host_sets(['0-3','4-7','8-11','12-15'],set(range(16)))

    def test_rejects_smt_ids_used_as_physical_core_ids(self):
        with self.assertRaisesRegex(ValueError,'physical-core'):
            validate_host_sets(['4-7','8-11','16-19','20-23'],set(range(16)))

    def test_rejects_overlap_between_mpi_queues(self):
        with self.assertRaisesRegex(ValueError,'overlap'):
            validate_host_sets(['0-3','2-5','8-11','12-15'],set(range(16)))


if __name__=='__main__':unittest.main()
