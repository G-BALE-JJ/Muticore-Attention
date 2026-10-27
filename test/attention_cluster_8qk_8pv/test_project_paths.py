import unittest
import tempfile
from pathlib import Path
from project_paths import resolve_project_path


class ProjectPathMigrationTest(unittest.TestCase):
    def test_feature_workspace_resolves_public_historical_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            public = Path(directory)/'LLM'
            feature = public/'exp-example'
            (public/'test/history').mkdir(parents=True)
            (feature/'test').mkdir(parents=True)
            (feature/'test/history').symlink_to(public/'test/history')
            old = public/'Muticore-Attention/results/case/x.bin'
            self.assertEqual(resolve_project_path(old, feature),
                             feature/'test/history/legacy-primary/results/case/x.bin')

    def test_legacy_input_location_maps_to_preserved_archive(self):
        root = Path('/example/LLM')
        self.assertEqual(resolve_project_path(root/'Muticore-Attention/results/case/x.bin', root),
                         root/'test/history/legacy-primary/results/case/x.bin')

    def test_latest_experimental_workspace_keeps_its_own_path(self):
        path = Path('/example/LLM/Muticore-Attention-all-gemm/results/case/x.bin')
        self.assertEqual(resolve_project_path(path, Path('/example/LLM')), path)

    def test_old_public_test_paths_resolve_without_a_symlink(self):
        root = Path('/example/LLM')
        self.assertEqual(resolve_project_path('baseline/attention_cluster_8qk_8pv/run_sst.sh', root),
                         root/'test/attention_cluster_8qk_8pv/run_sst.sh')
        self.assertEqual(resolve_project_path('src/sst/elements/golem/tests/configs/default.env', root),
                         root/'test/golem/configs/default.env')

    def test_external_and_current_artifact_paths_are_preserved(self):
        root = Path('/example/LLM')
        for path in (Path('/other/input.bin'), root/'test/prefill/results/case/input.bin'):
            self.assertEqual(resolve_project_path(path, root), path)
