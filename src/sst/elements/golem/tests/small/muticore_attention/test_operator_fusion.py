"""Fusion controls must reach the simulated guest, not just host controllers."""
import ast
import hashlib
import json
import tempfile
from pathlib import Path
import unittest

from report_projection_e2e import summarize
from projection_case import reuse_verified_inputs
from verify_fused_attention_scale_output import rotate_rope


class FusionEnvironmentTest(unittest.TestCase):
    def test_guest_receives_all_fusion_controls(self):
        source = Path(__file__).resolve().parents[2] / 'architecture/ncores_selfcom_dma_ctrl.py'
        tree = ast.parse(source.read_text())
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'process_env_keys'
                                  for t in n.targets))
        forwarded = ast.literal_eval(assignment.value)
        for flag in ('GOLEM_FUSION_NORM_PROJECTION', 'GOLEM_FUSION_PROJECTION_ROPE',
                     'GOLEM_FUSION_SKIP_AUDIT_OUTPUT'):
            self.assertIn(flag, forwarded)


class RotatedPanelStorageTest(unittest.TestCase):
    def test_preserves_existing_sfu_underflow_contract(self):
        import numpy as np
        values = [-0.032867431640625, -0.00658416748046875]
        table = [-0.1964111328125, -0.98046875]
        ordinary = np.asarray(rotate_rope(values, 1, 2, table), dtype='<f2')
        storage = np.asarray(rotate_rope(values, 1, 2, table, simulator_storage=True), dtype='<f2')
        self.assertEqual(int(ordinary.view('u2')[0]), 0x8001)
        self.assertEqual(int(storage.view('u2')[0]), 0x8000)
        self.assertEqual(ordinary[1], storage[1])


class VerifiedInputReuseTest(unittest.TestCase):
    def test_rejects_shape_mismatch_and_corrupted_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, output = root / 'reference', root / 'output'
            reference.mkdir()
            shape = {'query_length': 1024, 'Hq': 4, 'Hkv': 2, 'head_dim': 64}
            (reference / 'projection_e2e_result.json').write_text(json.dumps({'status': 'PASS', 'shape': shape}))
            names = ['projection_x.bin', 'projection_gamma.bin', 'projection_norm.bin',
                     'projection_weights.bin', 'rope_64.bin', 'q_4x1024x64.bin',
                     'k_2x1024x64.bin', 'v_2x1024x64.bin']
            for name in names:
                (reference / name).write_bytes(name.encode())
            hashes = {name: hashlib.sha256((reference / name).read_bytes()).hexdigest() for name in names}
            (reference / 'baseline_provenance.json').write_text(json.dumps({'input_sha256': hashes}))
            reuse_verified_inputs(reference, 1024, 4, 2, 64, output)
            self.assertEqual((output / names[0]).read_bytes(), names[0].encode())
            with self.assertRaisesRegex(ValueError, 'shape/status mismatch'):
                reuse_verified_inputs(reference, 2048, 4, 2, 64, output)
            (reference / 'projection_weights.bin').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                reuse_verified_inputs(reference, 1024, 4, 2, 64, root / 'uncopied')
            self.assertFalse((root / 'uncopied').exists())


class FusionAccountingTest(unittest.TestCase):
    def test_fused_norm_and_rope_accounting_and_overlap(self):
        attention = {"status": "PASS", "shape": {"Hq": 4, "Hkv": 2,
            "query_length": 1024, "kv_length": 1024, "head_dim": 64, "causal": True},
            "start_cycle": 40000, "end_cycle": 50000}
        lines = []
        for core in range(4):
            for batch in range(8):
                start = (11000 + batch * 100) * 1000
                lines.append(f"[SFU_RMSNORM] core={core} issue_tick={start} "
                             f"complete_tick={start+50000} vector_cycles=40 status=0")
            lines.append(f"[PROJECTION_JOB] manager={core} start=10000 end=30000 "
                         "cycles=20000 weight_loads=32 weight_programs=32 weight_reuses=0 input_loads=1 status=0")
            lines.append(f"[PROJECTION_FUSION] manager={core} norm=1 rope=1 audit=0 "
                         "norm_jobs=8 rope_jobs=96 norm_vector_cycles=320 rope_vector_cycles=1000 "
                         "local_read_bytes=528384 local_write_bytes=327680 "
                         "norm_audit_bytes=0 raw_audit_bytes=0 panel_bytes=262144")
        for core in range(4, 12):
            lines.append(f"GOLEM_LOCAL_GM_STATS core={core} gmem_local_read_requests=1 "
                         "gmem_local_write_requests=0 gmem_local_read_bytes=1114112")
        log = "\n".join(lines)
        report = summarize(log, attention, 66)
        self.assertEqual(report['end_to_end_cycles'], 40000)
        self.assertTrue(report['stages']['projection']['includes_rmsnorm'])
        self.assertTrue(report['stages']['projection']['includes_rope'])
        for old, new in [('norm_jobs=8', 'norm_jobs=7'),
                         ('rope_jobs=96', 'rope_jobs=95'),
                         ('local_read_bytes=528384', 'local_read_bytes=0'),
                         ('raw_audit_bytes=0', 'raw_audit_bytes=1'),
                         ('norm_audit_bytes=0', 'norm_audit_bytes=1')]:
            with self.subTest(field=old), self.assertRaisesRegex(ValueError, 'fusion work/traffic'):
                summarize(log.replace(old, new, 1), attention, 66)
        with self.assertRaisesRegex(ValueError, 'QK operand timing'):
            summarize(log.replace('gmem_local_read_bytes=1114112', 'gmem_local_read_bytes=0', 1), attention, 66)
        with self.assertRaisesRegex(ValueError, 'fusion evidence'):
            summarize(log.replace('manager=3 norm=1', 'manager=3 norm=0'), attention, 66)


class PartialMovementAccountingTest(unittest.TestCase):
    def fixture(self):
        attention = {'status': 'PASS', 'shape': {'Hq': 4, 'Hkv': 2,
            'query_length': 1024, 'kv_length': 1024, 'head_dim': 64, 'causal': True},
            'start_cycle': 40000, 'end_cycle': 50000}
        lines = []
        for core in range(4):
            for batch in range(8):
                start = (11000 + batch * 100) * 1000
                lines.append(f'[SFU_RMSNORM] core={core} issue_tick={start} complete_tick={start+50000} vector_cycles=40 status=0')
            lines.extend([
                f'[PROJECTION_JOB] manager={core} start=10000 end=30000 cycles=20000 weight_loads=32 weight_programs=32 weight_reuses=96 input_loads=1 status=0',
                f'[PROJECTION_FUSION] manager={core} norm=1 rope=0 audit=0 norm_jobs=8 rope_jobs=0 norm_vector_cycles=320 rope_vector_cycles=0 local_read_bytes=135168 local_write_bytes=131072 norm_audit_bytes=0 raw_audit_bytes=0 panel_bytes=262144',
                f'[PROJECTION_LOCAL_GM] manager={core} read_bytes=1310720 write_bytes=0 read_cycles=100 write_cycles=0 timed=1 reuse_block=1 paired_weights=1',
                f'[PROJECTION_PIPELINE] manager={core} enabled=1 shared_input=1 input_prefetches=48 partial_prefetches=24 scatter_prefetches=64 background_read_cycles=200 background_write_cycles=0 staging_bytes=32768 array_count=64',
                f'[PROJECTION_SCHEDULE] manager={core} row_first=0 cbuffer_partial=1 cbuffer_bytes=32768 cbuffer_read_bytes=262144 cbuffer_write_bytes=262144',
                f'[PROJECTION_CYCLE_BREAKDOWN] manager={core} wrap_prefetch=0 read_output=320 restore_output_ready=200 local_read=100 local_input_read=100 local_weight_read=0 local_fusion_read=0 local_other_read=0',
                f'[PROJECTION_PARTIAL_MOVEMENT] manager={core} restore_ahead=1 partial_reads=32 final_reads=32 partial_read_cycles=160 final_read_cycles=160 restores_foreground=8 restores_ahead=24 restore_cbuffer_wait=8 restore_submit=8 restore_transfer=96 restore_ahead_wait=64 restore_ahead_consume=24 ahead_retries=0 cbuffer_prefetch_reads=24 cbuffer_demand_reads=8 cbuffer_stores=32 cbuffer_prefetch_residence=792 cbuffer_demand_residence=264 cbuffer_store_residence=1056',
            ])
        return '\n'.join(lines), attention

    def test_mixed_early_and_foreground_restores_preserve_work(self):
        log, attention = self.fixture()
        report = summarize(log, attention, 66)
        movement = report['stages']['projection']['partial_movement_by_manager']['0']
        self.assertEqual(movement['restores_foreground'] + movement['restores_ahead'], 32)
        self.assertEqual(movement['partial_read_cycles'] + movement['final_read_cycles'], 320)

    def test_rejects_lost_partial_or_unaccounted_waits(self):
        log, attention = self.fixture()
        for old, new in [('partial_reads=32', 'partial_reads=31'),
                         ('final_reads=32', 'final_reads=31'),
                         ('restores_ahead=24', 'restores_ahead=23'),
                         ('partial_read_cycles=160', 'partial_read_cycles=159'),
                         ('restore_ahead_wait=64', 'restore_ahead_wait=63'),
                         ('cbuffer_prefetch_reads=24', 'cbuffer_prefetch_reads=23'),
                         ('cbuffer_stores=32', 'cbuffer_stores=31')]:
            with self.subTest(field=old), self.assertRaisesRegex(ValueError, 'partial movement work/cycle'):
                summarize(log.replace(old, new, 1), attention, 66)


class InputFlowAccountingTest(unittest.TestCase):
    def fixture(self, packed):
        log, attention = PartialMovementAccountingTest().fixture()
        foreground, background = (64, 192) if packed else (1024, 3072)
        log += '\n' + '\n'.join(
            f'[PROJECTION_INPUT_FLOW] manager={core} packed_input={packed} input_prefetch_budget=4 max_request_bytes=4096 input_read_requests={foreground} background_input_read_requests={background} weight_read_requests=64 norm_write_requests=32 budget_stalls=3'
            for core in range(4))
        return log, attention

    def test_row_major_and_packed_request_counts(self):
        for packed in (0, 1):
            with self.subTest(packed=packed):
                log, attention = self.fixture(packed)
                report = summarize(log, attention, 66)
                flow = report['stages']['projection']['input_flow_by_manager']['0']
                self.assertEqual(flow['input_read_requests'] + flow['background_input_read_requests'],
                                 256 if packed else 4096)

    def test_rejects_missing_requests_and_invalid_limits(self):
        log, attention = self.fixture(1)
        for old, new in [('input_read_requests=64', 'input_read_requests=63'),
                         ('background_input_read_requests=192', 'background_input_read_requests=191'),
                         ('weight_read_requests=64', 'weight_read_requests=63'),
                         ('norm_write_requests=32', 'norm_write_requests=31'),
                         ('max_request_bytes=4096', 'max_request_bytes=0'),
                         ('input_prefetch_budget=4', 'input_prefetch_budget=65')]:
            with self.subTest(field=old), self.assertRaisesRegex(ValueError, 'input flow request accounting'):
                summarize(log.replace(old, new, 1), attention, 66)

    def test_rejects_missing_manager(self):
        log, attention = self.fixture(1)
        with self.assertRaisesRegex(ValueError, 'incomplete projection input flow'):
            summarize(log[:log.index('[PROJECTION_INPUT_FLOW] manager=3')], attention, 66)


if __name__ == '__main__':
    unittest.main()
