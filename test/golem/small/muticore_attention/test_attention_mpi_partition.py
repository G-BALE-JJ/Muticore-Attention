import unittest
from pathlib import Path
import sys

from verify_attention_mpi_partition import expected_rank, expected_router_rank, expected_component_ranks

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'architecture'))
from partition_planner import attention_query_block_placements


class AttentionMPITest(unittest.TestCase):
    def test_rank8_uses_the_architecture_query_block_plan(self):
        placements = attention_query_block_placements(28, 20, 8)
        for core in range(20):
            self.assertEqual(expected_rank(core, 8), placements[core + 4][0])
        for router in range(28):
            self.assertEqual(expected_router_rank(router, 8), placements[router][0])
        components = expected_component_ranks(8)
        for node in range(1, 5):
            self.assertEqual(components[f'memory_{node}'], placements[node - 1][0])


if __name__ == '__main__':
    unittest.main()
