#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

QK = re.compile(r"\[ATTENTION_REUSE_WINDOW_QK\] core=(\d+) cycles=(\d+) start=(\d+) end=(\d+).*fusion_tiles=(\d+) expected_tiles=(\d+)")
SFU = re.compile(r"\[ATTENTION_WORKER_CLUSTER_SFU\] core=(\d+) start=(\d+) end=(\d+) cycles=(\d+)")
PV = re.compile(r"\[ATTENTION_WORKER_CLUSTER_PV\] core=(\d+) qk_core=(\d+) row=(\d+) window=(\d+) cycles=(\d+) start=(\d+) end=(\d+)")
PV_TIMING = re.compile(r"\[ATTENTION_WORKER_CLUSTER_PV\].*p_ready=(\d+) pv_send=(\d+) pv_receive=(\d+) pv_start=(\d+)")
E2E = re.compile(r"\[ATTENTION_WORKER_CLUSTER_E2E\] core=(\d+) start=(\d+) end=(\d+) cycles=(\d+)")
V_RESIDENCY = re.compile(r"\[Core (\d+)\] \[wcp\] PV_V_RESIDENCY hit=([01])")
WCP_LATENCY = re.compile(r"\[Core (\d+)\] \[wcp\] LATENCY\(cycles\): ([^\n]+)")
SFU_EXP = re.compile(r"GOLEM_SFU_HW_PIPELINE core=(\d+) unit=exp lanes=(\d+).*?accepted_tokens=(\d+)")


def stage_span(events, start_index, end_index):
    starts = [int(item[start_index]) for item in events]
    ends = [int(item[end_index]) for item in events]
    start = min(starts, default=0)
    end = max(ends, default=0)
    return {"start_cycle": start, "end_cycle": end,
            "elapsed_cycles": end - start, "count": len(events)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--num-kv-heads", type=int, required=True)
    ap.add_argument("--num-query-heads", type=int, default=4)
    ap.add_argument("--query-length", type=int, default=1024)
    ap.add_argument("--kv-length", type=int, default=1024)
    ap.add_argument("--head-dim", type=int, choices=(64, 128), default=128)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--qk-workers-per-manager", type=int, choices=(1, 2), default=1)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    text = args.log.read_text(errors="replace")
    qk, sfu, pv, e2e = (pattern.findall(text) for pattern in (QK, SFU, PV, E2E))
    pv_timing = [tuple(map(int, item)) for item in PV_TIMING.findall(text)]
    residency = V_RESIDENCY.findall(text)
    jobs = 4 * args.num_kv_heads * args.qk_workers_per_manager
    group_size = args.num_query_heads // args.num_kv_heads
    row_blocks_per_job = group_size * args.query_length // (
        4 * args.qk_workers_per_manager * 64)
    if args.causal and args.query_length != args.kv_length:
        ap.error("causal prefill requires equal query and KV lengths")
    dense_pv_windows = args.num_query_heads * (args.query_length // 64) * (
        args.kv_length // 256)
    expected_pv_windows = (args.num_query_heads * sum(
        (block * 64 + 255) // 256
        for block in range(1, args.query_length // 64 + 1))
        if args.causal else dense_pv_windows)
    dense_qk_tiles = args.num_query_heads * (args.query_length // 64) * (
        args.kv_length // 64)
    expected_qk_tiles = (args.num_query_heads *
        (args.query_length // 64) * (args.query_length // 64 + 1) // 2
        if args.causal else dense_qk_tiles)
    qk_columns = range(args.qk_workers_per_manager)
    pv_lanes_per_manager = 4 - args.qk_workers_per_manager
    pv_columns = range(args.qk_workers_per_manager,
                       args.qk_workers_per_manager +
                       min(row_blocks_per_job, pv_lanes_per_manager))
    expected_qk_cores = sorted(
        4 + column * 4 + manager
        for column in qk_columns for manager in range(4)
    )
    expected_pv_cores = sorted(
        4 + column * 4 + manager
        for column in pv_columns for manager in range(4)
    )
    observed_qk_cores = sorted({int(x[0]) for x in qk})
    observed_sfu_cores = sorted({int(x[0]) for x in sfu})
    observed_pv_cores = sorted({int(x[0]) for x in pv})
    exp_events = [tuple(map(int, event)) for event in SFU_EXP.findall(text)
                  if int(event[0]) in observed_qk_cores]
    exp_elements = sum(lanes * tokens for _, lanes, tokens in exp_events)
    expected_exp_elements = (expected_qk_tiles * 64 * 64 if args.causal else
                             args.num_query_heads * args.query_length * args.kv_length)
    valid_exp_elements = (args.num_query_heads * args.query_length *
                          (args.query_length + 1) // 2 if args.causal else
                          expected_exp_elements)
    qk_latency = {str(core): {} for core in observed_qk_cores}
    latency_fields = ("compute", "dma_wait", "total", "wait_2d_activate",
                      "wait_2d_active_not_ready", "c_buffer_read_wait")
    for core, fields in WCP_LATENCY.findall(text):
        if core not in qk_latency:
            continue
        values = dict(re.findall(r"(\w+)=(\d+)", fields))
        for name in latency_fields:
            qk_latency[core][name] = qk_latency[core].get(name, 0) + int(values.get(name, 0))
    qk_fusion_complete = all(int(item[4]) == int(item[5]) for item in qk)
    valid = (
        len(qk) == jobs and len(sfu) == jobs and len(e2e) == jobs and
        len(pv) == expected_pv_windows and
        qk_fusion_complete and
        (not args.causal or
         sum(int(item[4]) for item in qk) == expected_qk_tiles) and
        observed_qk_cores == expected_qk_cores and
        observed_sfu_cores == expected_qk_cores and
        observed_pv_cores == expected_pv_cores
    )
    if exp_events:
        valid = valid and exp_elements == expected_exp_elements
    starts = [int(x[1]) for x in e2e]
    ends = [int(x[2]) for x in e2e]
    transport = {}
    if pv_timing:
        waits = {
            "p_ready_to_dispatch": [item[1] - item[0] for item in pv_timing],
            "dispatch_to_receive": [item[2] - item[1] for item in pv_timing],
            "receive_to_start": [item[3] - item[2] for item in pv_timing],
        }
        transport = {
            name: {"average_cycles": sum(values) / len(values),
                   "max_cycles": max(values), "count": len(values)}
            for name, values in waits.items()
        }
    result = {
        "status": "PASS" if valid else "FAIL",
        "measurement": "sst_attention_worker_cluster_critical_path",
        "scope": "QK WCP + online SFU + NoC P dispatch + PV WCP + final PV ACK",
        "qk_workers_per_manager": args.qk_workers_per_manager,
        "shape": {"Hq": args.num_query_heads, "Hkv": args.num_kv_heads,
                  "query_length": args.query_length, "kv_length": args.kv_length,
                  "head_dim": args.head_dim, "causal": args.causal},
        "qk_worker_count": 4 * args.qk_workers_per_manager,
        "pv_worker_count": 4 * (4 - args.qk_workers_per_manager),
        "qk_jobs": len(qk), "sfu_jobs": len(sfu),
        "pv_windows": len(pv), "completed_jobs": len(e2e),
        "qk_fusion_complete": qk_fusion_complete,
        "qk_tiles": {"expected": expected_qk_tiles,
                     "observed": sum(int(item[4]) for item in qk),
                     "skipped": dense_qk_tiles - expected_qk_tiles},
        "pv_windows_skipped": dense_pv_windows - expected_pv_windows,
        "stage_spans": {
            "qk": stage_span(qk, 2, 3),
            "sfu": stage_span(sfu, 1, 2),
            "pv": stage_span(pv, 5, 6),
        },
        "pv_transport_waits": transport,
        "pv_service": {
            "average_window_cycles": sum(int(item[4]) for item in pv) / len(pv) if pv else 0,
            "max_window_cycles": max((int(item[4]) for item in pv), default=0),
            "core_busy_cycles": {
                str(core): sum(int(item[4]) for item in pv if int(item[0]) == core)
                for core in observed_pv_cores
            },
        },
        "v_cache": {"hits": sum(hit == "1" for _, hit in residency),
                    "misses": sum(hit == "0" for _, hit in residency)},
        "qk_wcp_cycles_by_core": qk_latency,
        "sfu_exp_work": {"expected_elements": expected_exp_elements,
                         "logically_valid_elements": valid_exp_elements,
                         "masked_elements": expected_exp_elements - valid_exp_elements,
                         "observed_elements": exp_elements if exp_events else None,
                         "complete": exp_elements == expected_exp_elements if exp_events else None},
        "pv_tail_after_sfu_cycles": max(0, max(ends, default=0) -
            max((int(item[2]) for item in sfu), default=0)),
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
