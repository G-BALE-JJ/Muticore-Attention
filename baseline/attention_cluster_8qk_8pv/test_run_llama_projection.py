import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_llama_projection import compare_reference


class ReferenceComparisonTest(unittest.TestCase):
    def test_exact_projection_bytes_and_measured_cycle_savings(self):
        with tempfile.TemporaryDirectory() as directory:
            reference, current = [Path(directory) / name for name in ('reference', 'current')]
            regions = [dict(name=name, begin=index * 2, end=index * 2 + 2)
                       for index, name in enumerate(('RMSNorm', 'Q', 'K', 'V',
                                                    'Q_panels', 'K_panels', 'V_panels', 'O'))]
            layout = {'nodes': [dict(node=node, regions=regions) for node in range(1, 5)]}
            for root in (reference, current):
                (root / 'hbm').mkdir(parents=True)
                (root / 'attention_hbm_layout.json').write_text(json.dumps(layout))
                for name in ('projection_x.bin', 'projection_gamma.bin', 'projection_weights.bin'):
                    (root / name).write_bytes(b'inputs')
                for node in range(1, 5):
                    (root / 'hbm' / f'hbm_out_node{node}.bin').write_bytes(bytes(range(16)))
            def summary(root):
                cycles = 100 if root == reference else 80
                return {'pipeline': {'shape': {'S': 1024}, 'end_to_end_cycles': cycles,
                                     'stages': {'projection': {'elapsed_cycles': cycles - 20}}}}
            with patch('run_llama_projection.summarize', side_effect=summary):
                result = compare_reference(current, reference)
                self.assertEqual(result['end_to_end']['saved_cycles'], 20)
                self.assertEqual(result['end_to_end']['reduction_percent'], 20)
                self.assertEqual(len(result['byte_exact_regions']), 28)
                output = current / 'hbm' / 'hbm_out_node4.bin'
                changed = bytearray(output.read_bytes())
                changed[2] ^= 1
                output.write_bytes(changed)
                with self.assertRaisesRegex(ValueError, 'node 4 Q'):
                    compare_reference(current, reference)
                (current / 'projection_weights.bin').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'reference input mismatch'):
                    compare_reference(current, reference)


if __name__ == '__main__':
    unittest.main()
