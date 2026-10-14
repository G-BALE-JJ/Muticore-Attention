import json
from pathlib import Path
import tempfile
import unittest

from summarize_batch_b import combine


class BatchBProvenanceTests(unittest.TestCase):
    def fixture(self, root, changed_input=False, changed_library=False, origin=100, duplicate=False):
        baseline=root/'reference'/'s1024'
        baseline.mkdir(parents=True)
        base={'case':'baseline','status':'PASS','shape':{'query_length':1024},
              'fixed_origin_cycle':100,'artifact_root':str(baseline)}
        provenance={'input_sha256':{'x':'original'},'sha256':{'libgolem.so':'original'},'environment':{}}
        (baseline/'experiment_result.json').write_text(json.dumps(base))
        (baseline/'baseline_provenance.json').write_text(json.dumps(provenance))
        shard=root/'shard';shard.mkdir()
        case=shard/'oracle_gm_read'/'s1024';case.mkdir(parents=True)
        row={**base,'case':'baseline' if duplicate else 'oracle_gm_read','artifact_root':str(case),
             'fixed_origin_cycle':origin,'byte_exact':{'status':'PASS'}}
        current=json.loads(json.dumps(provenance))
        if changed_input:current['input_sha256']['x']='different'
        if changed_library:current['sha256']['libgolem.so']='different'
        (case/'baseline_provenance.json').write_text(json.dumps(current))
        (shard/'summary.json').write_text(json.dumps({'status':'PASS','cases':[row]}))
        return baseline.parent,[shard]

    def test_rejects_input_mismatch_across_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots=self.fixture(Path(tmp),changed_input=True)
            with self.assertRaisesRegex(ValueError,'inputs changed'):combine(*roots,sequences=(1024,))

    def test_rejects_model_binary_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots=self.fixture(Path(tmp),changed_library=True)
            with self.assertRaisesRegex(ValueError,'model/guest changed'):combine(*roots,sequences=(1024,))

    def test_rejects_moving_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots=self.fixture(Path(tmp),origin=101)
            with self.assertRaisesRegex(ValueError,'origin moved'):combine(*roots,sequences=(1024,))

    def test_rejects_duplicate_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots=self.fixture(Path(tmp),duplicate=True)
            with self.assertRaisesRegex(ValueError,'duplicate case'):combine(*roots,sequences=(1024,))

    def test_rejects_partial_matrix_even_if_shard_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots=self.fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError,'missing/unexpected'):combine(*roots,sequences=(1024,))


if __name__=='__main__':unittest.main()
