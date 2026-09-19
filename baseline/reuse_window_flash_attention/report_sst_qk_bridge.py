#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


LINE = re.compile(
    r"\[ATTENTION_REUSE_WINDOW_QK\] core=(?P<core>\d+) "
    r"cycles=(?P<cycles>\d+) start=(?P<start>\d+) end=(?P<end>\d+) "
    r"fusion_tiles=(?P<tiles>\d+) expected_tiles=(?P<expected>\d+)"
)
E2E_LINE = re.compile(
    r"\[ATTENTION_REUSE_WINDOW_E2E\] core=(?P<core>\d+) "
    r"cycles=(?P<cycles>\d+) start=(?P<start>\d+) end=(?P<end>\d+) "
    r"qk_end=(?P<qk_end>\d+) pv_cycles=(?P<pv_cycles>\d+)"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--num-kv-heads", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = [
        {key: int(value) for key, value in match.groupdict().items()}
        for match in LINE.finditer(args.log.read_text(errors="replace"))
    ]
    e2e_records = [
        {key: int(value) for key, value in match.groupdict().items()}
        for match in E2E_LINE.finditer(args.log.read_text(errors="replace"))
    ]
    expected_records = 16 * args.num_kv_heads
    valid = (
        len(records) == expected_records
        and len(e2e_records) == expected_records
        and all(item["tiles"] == item["expected"] for item in records)
        and all(item["end"] >= item["start"] for item in e2e_records)
    )
    qk_start = min((item["start"] for item in records), default=0)
    qk_end = max((item["end"] for item in records), default=0)
    start = min((item["start"] for item in e2e_records), default=0)
    end = max((item["end"] for item in e2e_records), default=0)
    result = {
        "status": "PASS" if valid else "FAIL",
        "measurement": "sst_reuse_window_flash_attention_gqa_critical_path",
        "scope": "QK WCP + online SFU softmax + PV WCP + output DMA",
        "worker_records": len(records),
        "e2e_worker_records": len(e2e_records),
        "expected_worker_records": expected_records,
        "start_cycle": start,
        "end_cycle": end,
        "end_to_end_cycles": end - start if end >= start else 0,
        "qk_start_cycle": qk_start,
        "qk_end_cycle": qk_end,
        "qk_gqa_cycles": qk_end - qk_start if qk_end >= qk_start else 0,
        "slowest_worker_window_cycles": max(
            (item["cycles"] for item in records), default=0
        ),
        "slowest_worker_e2e_cycles": max(
            (item["cycles"] for item in e2e_records), default=0
        ),
        "pv_component_cycles_sum": sum(
            item["pv_cycles"] for item in e2e_records
        ),
        "fusion_tiles": sum(item["tiles"] for item in records),
        "workers": records,
        "e2e_workers": e2e_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
