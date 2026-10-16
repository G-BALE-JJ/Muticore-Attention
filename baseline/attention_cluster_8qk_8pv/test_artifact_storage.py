import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from artifact_storage import preflight, finalize
from merge_mpi_rank_logs import merge


class ArtifactStorageTest(unittest.TestCase):
    def case(self, root):
        (root/'logs').mkdir()
        (root/'result.json').write_text('{"status":"PASS"}')
        for i in range(2):
            rank=root/'mpi_rank_output'/'1'/f'rank.{i}'
            rank.mkdir(parents=True)
            (rank/'stdout').write_text(f'core={i}\n[RESOURCE_INTERVAL] request={i}\n')
            (rank/'stderr').write_text(f'diagnostic {i}\n')
        log=root/'logs'/'attention-sst.log'
        log.write_text('interleaved console\n')
        merge(root/'mpi_rank_output',log,2)
        return log

    def test_verified_duplicates_removed_canonical_and_tensor_bytes_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);log=self.case(root)
            (root/'hbm').mkdir();p=root/'hbm'/'hbm_out_node1.bin'
            p.write_bytes(b'ABCD'+bytes(1024*1024)+b'END')
            original=hashlib.sha256(p.read_bytes()).hexdigest()
            report=finalize(root)
            self.assertEqual(report['status'],'PASS')
            self.assertFalse(log.with_suffix('.mpi-console-raw').exists())
            self.assertFalse(list((root/'mpi_rank_output').rglob('stdout')))
            self.assertIn('diagnostic 1',log.read_text())
            self.assertIn('request=1',log.with_suffix('.resources.jsonl').read_text())
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),original)
            self.assertEqual(p.stat().st_size,1024*1024+7)
            self.assertEqual(finalize(root)['status'],'PASS')

    def test_corrupt_canonical_preserves_all_raw_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);log=self.case(root)
            log.with_suffix('.resources.jsonl').write_text('truncated\n')
            report=finalize(root)
            self.assertEqual(report['logs']['status'],'SKIP')
            self.assertTrue(log.with_suffix('.mpi-console-raw').exists())
            self.assertEqual(len(list((root/'mpi_rank_output').rglob('stdout'))),2)

    def test_named_stack_log_is_compacted_and_causal_records_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);log=self.case(root)
            rank=root/'mpi_rank_output'/'1'/'rank.0'/'stdout'
            with rank.open('a') as stream:stream.write('[PROJECTION_CAUSAL] id=9\n')
            with log.with_suffix('.causal.jsonl').open('a') as stream:stream.write('[PROJECTION_CAUSAL] id=9\n')
            named=log.with_name('attention-sst_fused_attention_q1024.log')
            for extension in ('.log','.resources.jsonl','.causal.jsonl','.mpi-console-raw'):
                log.with_suffix(extension).rename(named.with_suffix(extension))
            finalize(root,log=named)
            self.assertFalse(list((root/'mpi_rank_output').rglob('stdout')))
            self.assertIn('id=9',named.with_suffix('.causal.jsonl').read_text())

    def test_missing_rank_preserves_raw_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);log=self.case(root)
            (root/'mpi_rank_output'/'1'/'rank.0'/'stdout').unlink()
            self.assertEqual(finalize(root)['logs']['status'],'SKIP')
            self.assertTrue(log.with_suffix('.mpi-console-raw').exists())

    def test_external_rank_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'case';root.mkdir();self.case(root)
            with self.assertRaises(ValueError):finalize(root,rank_root=Path(tmp)/'other')

    def test_failure_and_full_policy_keep_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);log=self.case(root)
            (root/'result.json').write_text('{"status":"FAIL"}')
            with self.assertRaisesRegex(ValueError,'PASS'):finalize(root)
            (root/'result.json').write_text('{"status":"PASS"}')
            finalize(root,policy='full')
            self.assertTrue(log.with_suffix('.mpi-console-raw').exists())

    def test_single_rank_raw_and_symlink_backing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'case';root.mkdir();(root/'logs').mkdir()
            (root/'result.json').write_text('{"status":"PASS"}')
            log=root/'logs'/'attention-sst.log';log.write_text('done\n')
            log.with_suffix('.resources.jsonl').write_text('[RESOURCE_INTERVAL] id=1\n')
            log.with_suffix('.raw').write_text('done\n[RESOURCE_INTERVAL] id=1\n')
            backing=Path(tmp)/'backing';backing.mkdir();(root/'hbm').symlink_to(backing,target_is_directory=True)
            (backing/'data.bin').write_bytes(bytes(8192))
            finalize(root)
            self.assertFalse(log.with_suffix('.raw').exists())
            self.assertEqual((backing/'data.bin').read_bytes(),bytes(8192))

    def test_low_space_stops_before_directory_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'not-created'
            with patch('artifact_storage.shutil.disk_usage') as usage:
                usage.return_value.free=2**30
                with self.assertRaisesRegex(OSError,'free'):preflight(path,min_free_gib=2)
            self.assertFalse(path.exists())

    def test_summary_deletes_only_case_tensors_and_saves_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'case';root.mkdir();self.case(root)
            shared=Path(tmp)/'inputs';shared.mkdir();(shared/'weights.bin').write_bytes(b'weights')
            (root/'hbm').mkdir();p=root/'hbm'/'hbm_out_node1.bin';p.write_bytes(b'output')
            (root/'logits.npy').write_bytes(b'numpy output')
            finalize(root,discard_tensors=True)
            self.assertFalse(p.exists());self.assertTrue((root/'logits.npy').exists())
            self.assertTrue((shared/'weights.bin').exists())
            manifest=json.loads((root/'artifact_cleanup.json').read_text())
            self.assertEqual(manifest['discarded_tensors'][0]['sha256'],hashlib.sha256(b'output').hexdigest())


if __name__=='__main__':unittest.main()
