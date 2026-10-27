"""Only one FFN scheduling flag may differ in a same-build comparison."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from summarize_ffn_demand_order import check_pair, FLAG
from compare_final_norm_lm_stream import BINARY_KEYS


class DemandComparisonTest(unittest.TestCase):
    def test_rejects_changed_hardware_build_or_inputs(self):
        cases=[Path('/control'),Path('/candidate')]
        manifests=[dict(hidden=2048,oracle_sha256='same',ffn_demand_order=bool(i)) for i in range(2)]
        provenance=[dict(**{k:'same' for k in BINARY_KEYS},source_sha256={'runtime':'same'},
                         environment={FLAG:str(i),'GOLEM_GM_BANKS':'8'}) for i in range(2)]
        def check():
            with patch('summarize_ffn_demand_order.comparable_inputs',side_effect=copy.deepcopy(manifests)):
                return check_pair(cases,provenance)
        self.assertEqual(check(),{FLAG:dict(control='0',candidate='1')})
        provenance[1]['environment']['GOLEM_GM_BANKS']='16'
        with self.assertRaisesRegex(ValueError,'environment'):check()
        provenance[1]['environment']['GOLEM_GM_BANKS']='8'
        provenance[1]['guest_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'binary'):check()
        provenance[1]['guest_sha256']='same'
        provenance[1]['source_sha256']['runtime']='changed'
        with self.assertRaisesRegex(ValueError,'source'):check()
        provenance[1]['source_sha256']['runtime']='same'
        manifests[1]['oracle_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'input'):check()


if __name__=='__main__':unittest.main()
