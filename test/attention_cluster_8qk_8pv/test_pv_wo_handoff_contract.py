import unittest

from pv_wo_handoff_contract import build_contract


class PvWoHandoffContractTest(unittest.TestCase):
    def test_route_covers_each_worker_tile_without_gaps(self):
        contract = build_contract()
        self.assertEqual(contract["status"], "PASS")
        self.assertEqual(contract["payloads_per_layer"], 512)
        self.assertEqual(contract["payload_bytes_per_layer"], 4 * 1024 * 1024)
        self.assertTrue(contract["worker_gm_capacity_ok"])
        self.assertEqual(contract["worker_gm_bytes"], 256 * 1024)
        for transfer in contract["transfers"]:
            expected_worker = 4 + transfer["manager"] + 4 * transfer["lane"]
            self.assertEqual(transfer["worker_core"], expected_worker)
            self.assertEqual(transfer["group_query_row_begin"] // 256, transfer["head"] % 4)
            self.assertEqual(transfer["gm_offset"] // 8192, transfer["head"])

    def test_hbm_opportunity_is_write_plus_read(self):
        contract = build_contract()
        self.assertEqual(contract["current_pv_o_hbm_write_bytes_per_layer"], 4 * 1024 * 1024)
        self.assertEqual(contract["current_wo_o_hbm_read_bytes_per_layer"], 4 * 1024 * 1024)
        self.assertEqual(contract["removable_hbm_bytes_per_layer"], 8 * 1024 * 1024)
        self.assertEqual(contract["local_gm_landing_write_bytes_per_layer"], 4 * 1024 * 1024)

    def test_contract_rejects_unverified_shapes(self):
        with self.assertRaises(ValueError):
            build_contract(sequence=2048)


if __name__ == "__main__":
    unittest.main()
