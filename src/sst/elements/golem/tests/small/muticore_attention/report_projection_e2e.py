#!/usr/bin/env python3
"""Summarize the measured RMSNorm -> projection -> causal attention path."""

import argparse
import json
import math
import re
from pathlib import Path


RMS = re.compile(r"\[SFU_RMSNORM\] core=(\d+).*?issue_tick=(\d+).*?complete_tick=(\d+).*?vector_cycles=(\d+) status=(\d+)")
PROJECTION = re.compile(r"\[PROJECTION_JOB\] manager=(\d+) start=(\d+) end=(\d+) cycles=(\d+) weight_loads=(\d+) weight_programs=(\d+) weight_reuses=(\d+)(?: input_loads=(\d+))? status=(\d+)")
PHASE = re.compile(r"\[PROJECTION_PHASE\] manager=(\d+) input_dma=(\d+) weight_dma=(\d+) matrix_program=(\d+) input_scatter=(\d+) output_restore=(\d+) compute=(\d+) output_read=(\d+) write_drain=(\d+)")
SYNC = re.compile(r"\[PROJECTION_SYNC\] core=(\d+) stage=flag_wait cycle=(\d+) flag=(\d+) status=(\d+)")
LOCAL_WAIT = re.compile(r"\[PROJECTION_SYNC\] core=(\d+) stage=local_wait cycle=(\d+) status=(\d+)")
DESCRIPTOR = re.compile(r"\[ATTENTION_MILESTONE\] stage=(?:root_|manager_)descriptor_accept status=done .*?rocc_cycle=(\d+) core=(\d+)")


def summarize(log_text, attention):
    rms = [tuple(map(int, match)) for match in RMS.findall(log_text)]
    projection = [tuple(int(value) if value else None for value in match)
                  for match in PROJECTION.findall(log_text)]
    phases = [tuple(map(int, match)) for match in PHASE.findall(log_text)]
    sync = [tuple(map(int, match)) for match in SYNC.findall(log_text)]
    local_wait = [tuple(map(int, match)) for match in LOCAL_WAIT.findall(log_text)]
    descriptors = [tuple(map(int, event)) for event in DESCRIPTOR.findall(log_text)]
    shape = attention["shape"]
    hq, hkv = shape["Hq"], shape["Hkv"]
    sequence, dim = shape["query_length"], shape["head_dim"]
    hidden = hq * dim
    rms_batch_rows = 8192 // hidden
    expected_rms_jobs = 4 * math.ceil((sequence // 4) / rms_batch_rows)
    if (attention["status"] != "PASS" or not shape["causal"] or
            shape["kv_length"] != sequence or len(rms) != expected_rms_jobs or
            len(projection) != 4 or
            any(sum(event[0] == core for event in rms) != expected_rms_jobs // 4
                for core in range(4)) or
            sorted(event[0] for event in projection) != list(range(4)) or
            any(event[-1] for event in rms + projection)):
        raise ValueError("incomplete or failed projection pipeline")
    rms_start = min(event[1] for event in rms) // 1000
    rms_end = math.ceil(max(event[2] for event in rms) / 1000)
    projection_start = min(event[1] for event in projection)
    projection_end = (max(event[1] for event in local_wait) if local_wait else
                      max(event[2] for event in projection))
    attention_start = attention["start_cycle"]
    attention_end = attention["end_cycle"]
    rms_end_by_core = {core: max(event[2] for event in rms if event[0] == core)
                       for core in range(4)}
    if (not (rms_start < rms_end and projection_start < projection_end <=
             attention_start < attention_end) or
            any(event[1] * 1000 < rms_end_by_core[event[0]] for event in projection)):
        raise ValueError("stage timestamps are not ordered")
    rms_core_work = {}
    for core, _, _, vector_cycles, _ in rms:
        rms_core_work[core] = rms_core_work.get(core, 0) + vector_cycles
    projection_macs = sequence * (hq + 2 * hkv) * dim * hidden
    projection_floor = math.ceil(projection_macs / (4 * 16 * 64 * 64))
    causal_tiles = hq * (sequence // 64) * (sequence // 64 + 1) // 2
    exp_elements = causal_tiles * 64 * 64
    attention_exp_floor = math.ceil(exp_elements / (8 * 16))
    floor = max(rms_core_work.values()) + projection_floor + attention_exp_floor
    actual = attention_end - rms_start
    result = {
        "status": "PASS",
        "shape": shape,
        "stages": {
            "rmsnorm": {"start_cycle": rms_start, "end_cycle": rms_end,
                        "elapsed_cycles": rms_end - rms_start, "jobs": len(rms)},
            "projection": {"start_cycle": projection_start, "end_cycle": projection_end,
                           "elapsed_cycles": projection_end - projection_start,
                           "jobs": len(projection),
                           "weight_loads": sum(event[4] for event in projection),
                           "weight_programs": sum(event[5] for event in projection),
                           "weight_reuses": sum(event[6] for event in projection)},
            "attention": {"start_cycle": attention_start, "end_cycle": attention_end,
                          "elapsed_cycles": attention_end - attention_start},
        },
        "stage_handoff_cycles": {
            "rmsnorm_to_projection": projection_start - rms_end,
            "projection_to_attention": attention_start - projection_end,
        },
        "end_to_end_cycles": actual,
        "optimistic_resource_floor_cycles": floor,
        "floor_components_cycles": {
            "rmsnorm_vector": max(rms_core_work.values()),
            "projection_array": projection_floor,
            "attention_sfu_exp": attention_exp_floor,
        },
        "actual_to_floor_ratio": round(actual / floor, 3),
        "floor_scope": "RMSNorm vector work + projection array MACs + causal attention exp lanes; excludes DMA, NoC, control and handoff",
    }
    if all(event[7] is not None for event in projection):
        result["stages"]["projection"]["input_loads"] = sum(event[7] for event in projection)
    if sorted(event[0] for event in phases) == list(range(4)):
        names = ("input_dma", "weight_dma", "matrix_program", "input_scatter",
                 "output_restore", "compute", "output_read", "write_drain")
        result["stages"]["projection"]["phase_cycles_by_manager"] = {
            str(event[0]): dict(zip(names, event[1:])) for event in phases
        }
    if bool(sync) != bool(descriptors):
        raise ValueError("incomplete projection synchronization")
    if sync:
        if (len(sync) != 16 or any(event[3] for event in sync) or
                len(local_wait) != 4 or
                sorted(event[0] for event in local_wait) != list(range(4)) or
                any(event[2] for event in local_wait)):
            raise ValueError("incomplete projection synchronization")
        sync_by_core = {core: max(event[1] for event in sync if event[0] == core)
                        for core in range(4)}
        descriptor_by_core = {
            core: min(event[0] for event in descriptors if event[1] == core)
            for core in range(4) if any(event[1] == core for event in descriptors)
        }
        sync_end = max(sync_by_core.values())
        descriptor_end = max(descriptor_by_core.values(), default=0)
        if (len(descriptor_by_core) != 4 or
                not projection_end <= sync_end <= descriptor_end <= attention_start or
                any(sync_by_core[core] > descriptor_by_core[core] for core in range(4))):
            raise ValueError("synchronization timestamps are not ordered")
        result["stage_handoff_cycles"].update({
            "projection_to_sync": sync_end - projection_end,
            "sync_to_descriptor": descriptor_end - sync_end,
            "descriptor_to_attention": attention_start - descriptor_end,
            "sync_to_descriptor_by_manager": {
                str(core): descriptor_by_core[core] - sync_by_core[core]
                for core in range(4)
            },
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--attention-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.log.read_text(errors="replace"),
                       json.loads(args.attention_result.read_text()))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
