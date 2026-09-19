#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

QK = re.compile(r"\[ATTENTION_REUSE_WINDOW_QK\] core=(\d+) cycles=(\d+) start=(\d+) end=(\d+).*fusion_tiles=(\d+) expected_tiles=(\d+)")
PV = re.compile(r"\[ATTENTION_WORKER_CLUSTER_PV\] core=(\d+) qk_core=(\d+) row=(\d+) window=(\d+) cycles=(\d+) start=(\d+) end=(\d+)")
E2E = re.compile(r"\[ATTENTION_WORKER_CLUSTER_E2E\] core=(\d+) start=(\d+) end=(\d+) cycles=(\d+)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--num-kv-heads", type=int, required=True)
    ap.add_argument("--qk-workers-per-manager", type=int, choices=(1, 2), default=1)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    text = args.log.read_text(errors="replace")
    qk, pv, e2e = QK.findall(text), PV.findall(text), E2E.findall(text)
    jobs = 4 * args.num_kv_heads * args.qk_workers_per_manager
    row_blocks_per_job = 8 // args.qk_workers_per_manager
    expected_pv_windows = jobs * row_blocks_per_job * 4
    qk_columns = range(args.qk_workers_per_manager)
    pv_columns = range(args.qk_workers_per_manager, 4)
    expected_qk_cores = sorted(
        4 + column * 4 + manager
        for column in qk_columns for manager in range(4)
    )
    expected_pv_cores = sorted(
        4 + column * 4 + manager
        for column in pv_columns for manager in range(4)
    )
    observed_qk_cores = sorted({int(x[0]) for x in qk})
    observed_pv_cores = sorted({int(x[0]) for x in pv})
    qk_fusion_complete = all(int(item[4]) == int(item[5]) for item in qk)
    valid = (
        len(qk) == jobs and len(e2e) == jobs and
        len(pv) == expected_pv_windows and
        qk_fusion_complete and
        observed_qk_cores == expected_qk_cores and
        observed_pv_cores == expected_pv_cores
    )
    starts = [int(x[1]) for x in e2e]
    ends = [int(x[2]) for x in e2e]
    result = {
        "status": "PASS" if valid else "FAIL",
        "measurement": "sst_attention_worker_cluster_critical_path",
        "scope": "QK WCP + online SFU + NoC P dispatch + PV WCP + final PV ACK",
        "qk_workers_per_manager": args.qk_workers_per_manager,
        "qk_worker_count": 4 * args.qk_workers_per_manager,
        "pv_worker_count": 4 * (4 - args.qk_workers_per_manager),
        "qk_jobs": len(qk), "pv_windows": len(pv), "completed_jobs": len(e2e),
        "qk_fusion_complete": qk_fusion_complete,
        "start_cycle": min(starts, default=0), "end_cycle": max(ends, default=0),
        "end_to_end_cycles": max(ends, default=0) - min(starts, default=0),
        "slowest_job_cycles": max((int(x[3]) for x in e2e), default=0),
        "qk_cores": observed_qk_cores,
        "pv_cores": observed_pv_cores,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if valid else 1

if __name__ == "__main__":
    raise SystemExit(main())
