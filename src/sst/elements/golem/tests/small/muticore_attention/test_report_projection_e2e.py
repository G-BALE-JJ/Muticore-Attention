import unittest

from report_projection_e2e import summarize


class ProjectionReportTest(unittest.TestCase):
    def test_overlapping_manager_stages(self):
        attention = {
            "status": "PASS",
            "shape": {"Hq": 1, "Hkv": 1, "query_length": 1024,
                      "kv_length": 1024, "head_dim": 128, "causal": True},
            "start_cycle": 40000,
            "end_cycle": 50000,
        }
        lines = []
        for core in range(4):
            for batch in range(4):
                start = (10000 + core * 200 + batch * 1000) * 1000
                end = start + 500000
                lines.append(f"[SFU_RMSNORM] core={core} issue_tick={start} "
                             f"complete_tick={end} vector_cycles=400 status=0")
            projection_start = 13800 + core * 200
            lines.append(f"[PROJECTION_JOB] manager={core} start={projection_start} "
                         f"end=30000 cycles={30000-projection_start} "
                         "weight_loads=12 weight_programs=192 weight_reuses=0 "
                         "input_loads=3 status=0")
        report = summarize("\n".join(lines), attention)
        self.assertEqual(report["stages"]["rmsnorm"]["jobs"], 16)
        self.assertEqual(report["stages"]["projection"]["weight_loads"], 48)
        self.assertEqual(report["stages"]["projection"]["input_loads"], 12)
        self.assertLess(report["stage_handoff_cycles"]["rmsnorm_to_projection"], 0)
        self.assertEqual(report["end_to_end_cycles"], 40000)

    def test_rejects_missing_projection_manager(self):
        attention = {
            "status": "PASS",
            "shape": {"Hq": 1, "Hkv": 1, "query_length": 1024,
                      "kv_length": 1024, "head_dim": 64, "causal": True},
            "start_cycle": 40000,
            "end_cycle": 50000,
        }
        with self.assertRaisesRegex(ValueError, "incomplete"):
            summarize("", attention)


if __name__ == "__main__":
    unittest.main()
