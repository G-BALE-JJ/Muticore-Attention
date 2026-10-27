import unittest

from run_projection_overlap import verify_group_trace


class GroupOwnershipTraceTests(unittest.TestCase):
    @staticmethod
    def trace(early_program=False, duplicate=False, missing=False):
        rows = []
        schedules = [(0, 0, 0, 5, 15), (1, 0, 5, 10, 20),
                     (0, 1, 14 if early_program else 15, 20, 30),
                     (1, 1, 20, 25, 35)]
        for group, tile, program, launch, done in schedules:
            for event, cycle in [('program', program), ('launch', launch), ('done', done)]:
                if missing and group == 1 and tile == 1 and event == 'done':
                    continue
                rows.append(f'[PROJECTION_GROUP_TRACE] core=4 block=0 head=0 group={group} '
                            f'tile={tile} event={event} cycle={cycle}')
        if duplicate:
            rows.append(rows[0])
        return '\n'.join(rows)

    def test_disjoint_programming_overlaps_compute(self):
        result = verify_group_trace(self.trace(), 1, 1, 1, 2)
        self.assertEqual(result['same_operand_conflicts'], 0)
        self.assertGreater(result['same_tile_cross_group_overlap_cycles'], 0)

    def test_rejects_overwriting_computing_group(self):
        with self.assertRaisesRegex(ValueError, 'its own compute'):
            verify_group_trace(self.trace(early_program=True), 1, 1, 1, 2)

    def test_rejects_duplicate_event(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            verify_group_trace(self.trace(duplicate=True), 1, 1, 1, 2)

    def test_rejects_missing_completion(self):
        with self.assertRaisesRegex(ValueError, 'lifecycle timestamps'):
            verify_group_trace(self.trace(missing=True), 1, 1, 1, 2)


if __name__ == '__main__':
    unittest.main()
