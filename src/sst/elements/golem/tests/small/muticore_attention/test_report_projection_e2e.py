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
        lines = ["[GOLEM] MVM compute latency cycles=64"]
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
            lines.append(f"[PROJECTION_PHASE] manager={core} input_dma=10 "
                         "weight_dma=20 matrix_program=30 input_scatter=40 "
                         "output_restore=50 compute=60 output_read=70 write_drain=80")
            lines.append(f"[PROJECTION_LOCAL_GM] manager={core} "
                         f"read_bytes={3 * 16 * 128 * 2 + 192 * 8192} "
                         "write_bytes=0 read_cycles=100 write_cycles=0 timed=1 reuse_block=0")
            lines.append(f"[PROJECTION_SYNC] core={core} stage=local_wait cycle=30000 status=0")
            last_flag = 30100 + core * 100
            for slot in range(4):
                lines.append(f"[PROJECTION_SYNC] core={core} stage=flag_wait "
                             f"cycle={last_flag - 3 + slot} flag={slot} status=0")
            lines.append("[ATTENTION_MILESTONE] stage=manager_descriptor_accept "
                         f"status=done sst_tick=0 rocc_cycle={last_flag + 51} "
                         f"core={core} role=manager job=0 tag=0 query_tile=-1 kv_tile=-1")
        report = summarize("\n".join(lines), attention)
        self.assertEqual(report["stages"]["rmsnorm"]["jobs"], 16)
        self.assertEqual(report["stages"]["projection"]["weight_loads"], 48)
        self.assertEqual(report["stages"]["projection"]["input_loads"], 12)
        self.assertEqual(report["stages"]["projection"]["phase_cycles_by_manager"]["0"]["matrix_program"], 30)
        self.assertEqual(report["stage_handoff_cycles"]["sync_to_descriptor"], 51)
        self.assertEqual(report["stage_handoff_cycles"]["sync_to_descriptor_by_manager"]["0"], 51)
        self.assertLess(report["stage_handoff_cycles"]["rmsnorm_to_projection"], 0)
        self.assertEqual(report["end_to_end_cycles"], 40000)
        self.assertEqual(report["floor_components_cycles"]["projection_array"], 12288)
        self.assertEqual(report["stages"]["projection"]["local_gm_by_manager"]["0"]["timed"], 1)
        with self.assertRaisesRegex(ValueError, "byte accounting"):
            summarize("\n".join(lines).replace("read_bytes=1585152", "read_bytes=1"), attention)
        with self.assertRaisesRegex(ValueError, "Local-GM timing"):
            summarize("\n".join(lines).replace("timed=1", "timed=0"), attention)

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

    def test_paired_weight_work_and_sram_bytes(self):
        attention = {
            "status": "PASS",
            "shape": {"Hq": 4, "Hkv": 2, "query_length": 1024,
                      "kv_length": 1024, "head_dim": 64, "causal": True},
            "start_cycle": 40000, "end_cycle": 50000,
        }
        lines = []
        for core in range(4):
            for batch in range(8):
                start = (10000 + batch * 100) * 1000
                lines.append(f"[SFU_RMSNORM] core={core} issue_tick={start} "
                             f"complete_tick={start + 50000} vector_cycles=40 status=0")
            lines.append(f"[PROJECTION_JOB] manager={core} start=13000 "
                         "end=30000 cycles=17000 weight_loads=32 weight_programs=32 "
                         "weight_reuses=480 input_loads=8 status=0")
            lines.append(f"[PROJECTION_LOCAL_GM] manager={core} read_bytes=1572864 "
                         "write_bytes=262144 read_cycles=100 write_cycles=50 "
                         "timed=1 reuse_block=1 paired_weights=1")
        log = "\n".join(lines)
        report = summarize(log, attention, 66)
        self.assertEqual(report["stages"]["projection"]["weight_programs"], 128)
        self.assertEqual(report["stages"]["projection"]["local_gm_by_manager"]["0"]["paired_weights"], 1)
        overlap = log.replace("input_loads=8", "input_loads=1") + "\n" + "\n".join(
            f"[PROJECTION_PIPELINE] manager={core} enabled=1 shared_input=1 "
            "input_prefetches=240 partial_prefetches=120 scatter_prefetches=400 "
            "background_read_cycles=100 background_write_cycles=100 staging_bytes=8192"
            for core in range(4))
        shared_report = summarize(overlap, attention, 66)
        self.assertEqual(shared_report["stages"]["projection"]["input_loads"], 4)
        self.assertEqual(shared_report["stages"]["projection"]["pipeline_by_manager"]["0"]["enabled"], 1)
        for enabled in (0, 1):
            count = 24 if enabled else 0
            evidence = "\n".join(
                f"[PROJECTION_WEIGHT_PREFETCH] manager={core} enabled={enabled} "
                f"requests={count} hits={count} wait_cycles=0 inflight=0"
                for core in range(4))
            with_prefetch = overlap + "\n" + evidence
            report = summarize(with_prefetch, attention, 66)
            self.assertEqual(report["stages"]["projection"]["weight_prefetch_by_manager"]["0"]["requests"], count)
            for invalid in (with_prefetch.replace("inflight=0", "inflight=1", 1),
                            with_prefetch.replace(f"hits={count}", f"hits={count + 1}", 1),
                            with_prefetch.replace("requests=24 hits=24", "requests=33 hits=33", 1)):
                if invalid == with_prefetch:
                    continue
                with self.assertRaisesRegex(ValueError, "prefetch accounting"):
                    summarize(invalid, attention, 66)
            with self.assertRaisesRegex(ValueError, "incomplete.*prefetch"):
                summarize(with_prefetch.rsplit("\n", 1)[0], attention, 66)
        for width in (32, 64):
            with self.subTest(array_count=width):
                groups = (256 // width) * 8
                prefetch = 8 * 2 * (256 // width - 1)
                partial_prefetch = 8 * (256 // width - 1)
                wide = (overlap.replace("weight_reuses=480", f"weight_reuses={groups * 4 - 32}")
                        .replace("input_prefetches=240", f"input_prefetches={prefetch}")
                        .replace("partial_prefetches=120", f"partial_prefetches={partial_prefetch}")
                        .replace("staging_bytes=8192", f"staging_bytes={width * 512} array_count={width}"))
                report = summarize(wide, attention, 66)
                self.assertEqual(report["stages"]["projection"]["array_count"], width)
                self.assertEqual(report["floor_components_cycles"]["projection_array"], 540672 // width)
                for invalid in (wide.replace(f"array_count={width}", "array_count=48"),
                                wide.replace(f"array_count={width}", "array_count=16", 1)):
                    with self.assertRaisesRegex(ValueError, "array count"):
                        summarize(invalid, attention, 66)
                # Total bytes remain constant when rows are grouped differently.
                self.assertEqual(report["stages"]["projection"]["local_gm_by_manager"]["0"]["read_bytes"], 1572864)
                with self.assertRaisesRegex(ValueError, "overlap work"):
                    summarize(wide.replace(f"staging_bytes={width * 512}", "staging_bytes=8192"), attention, 66)
        for wrong in (overlap.replace("input_prefetches=240", "input_prefetches=241"),
                      overlap.replace("partial_prefetches=120", "partial_prefetches=119"),
                      overlap.replace("staging_bytes=8192", "staging_bytes=0")):
            with self.assertRaisesRegex(ValueError, "overlap work"):
                summarize(wrong, attention, 66)
        with self.assertRaisesRegex(ValueError, "paired projection"):
            summarize(overlap.replace("shared_input=1", "shared_input=0"), attention, 66)
        for corruption in (log.replace("read_bytes=1572864", "read_bytes=1"),
                           log.replace("weight_programs=32", "weight_programs=33"),
                           log.replace("weight_reuses=480", "weight_reuses=481")):
            with self.assertRaisesRegex(ValueError, "paired projection"):
                summarize(corruption, attention, 66)


if __name__ == "__main__":
    unittest.main()
