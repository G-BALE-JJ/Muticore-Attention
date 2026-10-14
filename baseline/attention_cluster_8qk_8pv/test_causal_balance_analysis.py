import unittest

from analyze_causal_balance import audit_routes


def routes(heads=8, rows=512, group=4, balanced=False):
    return '\n'.join(
        f'[ATTENTION_TASK_ROUTE] manager={m} head={h} slot={s} '
        f'worker={4 + s * 4 + (3 - m if balanced and h % 2 else m)} '
        f'row={m * rows * group + s * rows * group // 2} '
        f'rows={rows * group // 2} query_length={rows}'
        for m in range(4) for h in range(heads) for s in range(2))


class RouteAuditTest(unittest.TestCase):
    def test_causal_work_is_conserved_and_balanced(self):
        shape = dict(Hkv=8, Hq=32, query_length=2048)
        before = audit_routes(routes(), shape, False)['by_core']
        after = audit_routes(routes(balanced=True), shape, True)['by_core']
        self.assertEqual({v['qk_tiles'] for v in after.values()}, {2112})
        self.assertEqual({v['pv_windows'] for v in after.values()}, {576})
        for key in ('qk_tiles', 'pv_windows', 'jobs'):
            self.assertEqual(sum(v[key] for v in before.values()),
                             sum(v[key] for v in after.values()))

    def test_odd_head_count_preserves_coverage(self):
        shape = dict(Hkv=3, Hq=6, query_length=1024)
        before = audit_routes(routes(3, 256, 2), shape, False)
        after = audit_routes(routes(3, 256, 2, True), shape, True)
        self.assertEqual(after['descriptors'], 24)
        self.assertEqual(sum(v['qk_tiles'] for v in before['by_core'].values()),
                         sum(v['qk_tiles'] for v in after['by_core'].values()))

    def test_wrong_query_owner_is_rejected(self):
        text = routes(balanced=True).replace('row=3072 ', 'row=0 ', 1)
        with self.assertRaisesRegex(ValueError, 'ownership'):
            audit_routes(text, dict(Hkv=8, Hq=32, query_length=2048), True)

    def test_duplicate_completion_identity_is_rejected(self):
        text = routes(balanced=True)
        with self.assertRaisesRegex(ValueError, 'duplicated'):
            audit_routes(text + '\n' + text.splitlines()[0],
                         dict(Hkv=8, Hq=32, query_length=2048), True)


if __name__ == '__main__':
    unittest.main()
