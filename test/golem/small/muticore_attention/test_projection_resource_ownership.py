"""Reject forbidden manager resources and broken projection row ownership."""
import unittest
from report_projection_e2e import summarize


def evidence():
    lines = []
    for c in range(20):
        present = int(c >= 4)
        lines.append(f'[CORE_RESOURCE_OWNERSHIP] core={c} array_present={present} sfu_present={present}')
    for c in range(4, 20):
        lane = (c - 4) // 4
        lines.append(f'[PROJECTION_WORKER_OWNER] core={c} manager={(c-4)%4} lane={lane} row_offset={lane*64} row_count=64')
    return '\n'.join(lines)


class ResourceOwnershipTest(unittest.TestCase):
    def test_invalid_projection_group_geometry_is_rejected(self):
        lines = [f'[PROJECTION_DATAFLOW] core={c} input_pair_prefetch=0 local_weight_prefetch=0 array_groups=3'
                 for c in range(4, 20)]
        with self.assertRaisesRegex(ValueError, 'dataflow configuration'):
            summarize(evidence() + '\n' + '\n'.join(lines), {})

    def test_overlap_without_output_ownership_is_rejected(self):
        lines = [f'[PROJECTION_DATAFLOW] core={c} input_pair_prefetch=1 local_weight_prefetch=0 array_groups=1'
                 for c in range(4, 20)]
        with self.assertRaisesRegex(ValueError, 'dataflow configuration'):
            summarize(evidence() + '\n' + '\n'.join(lines), {})

    def test_extra_operand_context_is_rejected(self):
        lines = [f'[ARRAY_OPERAND_STORAGE] core={c} arrays=64 operand_banks=2 matrix_contexts=128 input_contexts=128'
                 for c in range(4, 20)]
        with self.assertRaisesRegex(ValueError, 'operand storage'):
            summarize(evidence() + '\n' + '\n'.join(lines), {})

    def test_missing_single_operand_worker_is_rejected(self):
        lines = [f'[ARRAY_OPERAND_STORAGE] core={c} arrays=64 operand_banks=1 matrix_contexts=64 input_contexts=64'
                 for c in range(4, 19)]
        with self.assertRaisesRegex(ValueError, 'operand storage'):
            summarize(evidence() + '\n' + '\n'.join(lines), {})

    def test_inconsistent_accumulator_mode_is_rejected(self):
        lines = [f'[PROJECTION_ACCUMULATION] core={c} retain_output={int(c == 4)} retained_pairs=0'
                 for c in range(4, 20)]
        with self.assertRaisesRegex(ValueError, 'accumulator ownership'):
            summarize(evidence() + '\n' + '\n'.join(lines), {})

    def test_missing_accumulator_worker_is_rejected(self):
        lines = [f'[PROJECTION_ACCUMULATION] core={c} retain_output=1 retained_pairs=720'
                 for c in range(4, 19)]
        with self.assertRaisesRegex(ValueError, 'accumulator ownership'):
            summarize(evidence() + '\n' + '\n'.join(lines), {})

    def test_manager_array_is_rejected(self):
        log = evidence().replace('core=0 array_present=0', 'core=0 array_present=1')
        with self.assertRaisesRegex(ValueError, 'resource violation'):
            summarize(log, {})

    def test_manager_sfu_is_rejected(self):
        log = evidence().replace('core=1 array_present=0 sfu_present=0',
                                 'core=1 array_present=0 sfu_present=1')
        with self.assertRaisesRegex(ValueError, 'resource violation'):
            summarize(log, {})

    def test_duplicate_row_partition_is_rejected(self):
        log = evidence().replace('core=8 manager=0 lane=1 row_offset=64',
                                 'core=8 manager=0 lane=1 row_offset=0')
        with self.assertRaisesRegex(ValueError, 'worker ownership'):
            summarize(log, {})

    def test_missing_worker_is_rejected(self):
        log = '\n'.join(line for line in evidence().splitlines()
                        if not line.startswith('[PROJECTION_WORKER_OWNER] core=19 '))
        with self.assertRaisesRegex(ValueError, 'worker ownership'):
            summarize(log, {})


if __name__ == '__main__':
    unittest.main()
