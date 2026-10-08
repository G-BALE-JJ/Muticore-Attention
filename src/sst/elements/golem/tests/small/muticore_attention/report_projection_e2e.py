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
LOCAL_GM = re.compile(r"\[PROJECTION_LOCAL_GM\] manager=(\d+) read_bytes=(\d+) write_bytes=(\d+) read_cycles=(\d+) write_cycles=(\d+) timed=(\d+) reuse_block=(\d+)(?: paired_weights=(\d+))?")
PIPELINE = re.compile(r"\[PROJECTION_PIPELINE\] manager=(\d+) enabled=(\d+) shared_input=(\d+) input_prefetches=(\d+) partial_prefetches=(\d+) scatter_prefetches=(\d+) background_read_cycles=(\d+) background_write_cycles=(\d+) staging_bytes=(\d+)(?: array_count=(\d+))?")
ARRAY_LATENCY = re.compile(r"MVM compute latency cycles=(\d+)")
WEIGHT_PREFETCH = re.compile(r"\[PROJECTION_WEIGHT_PREFETCH\] manager=(\d+) enabled=(\d+) requests=(\d+) hits=(\d+) wait_cycles=(\d+) inflight=(\d+)")
ROPE_TABLE = re.compile(r"\[ATTENTION_ROPE_TABLE\] core=(\d+) bytes=(\d+) cycles=(\d+) dma_cycles=(\d+) local_read_cycles=(\d+) start=(\d+) end=(\d+)")
SYNC = re.compile(r"\[PROJECTION_SYNC\] core=(\d+) stage=flag_wait cycle=(\d+) flag=(\d+) status=(\d+)")
LOCAL_WAIT = re.compile(r"\[PROJECTION_SYNC\] core=(\d+) stage=local_wait cycle=(\d+) status=(\d+)")
DESCRIPTOR = re.compile(r"\[ATTENTION_MILESTONE\] stage=(?:root_|manager_)descriptor_accept status=done .*?rocc_cycle=(\d+) core=(\d+)")


def summarize(log_text, attention, array_compute_cycles=None):
    rms = [tuple(map(int, match)) for match in RMS.findall(log_text)]
    projection = [tuple(int(value) if value else None for value in match)
                  for match in PROJECTION.findall(log_text)]
    phases = [tuple(map(int, match)) for match in PHASE.findall(log_text)]
    local_gm = [tuple(int(value) if value else 0 for value in match)
                for match in LOCAL_GM.findall(log_text)]
    pipelines = [tuple(int(value) if value else 16 for value in match)
                 for match in PIPELINE.findall(log_text)]
    if pipelines and sorted(event[0] for event in pipelines) != list(range(4)):
        raise ValueError("incomplete projection overlap evidence")
    pipeline_by_core = {event[0]: event for event in pipelines}
    widths = {event[9] for event in pipelines} or {16}
    if len(widths) != 1 or not widths <= {16, 32, 64}:
        raise ValueError("invalid projection array count")
    array_count = next(iter(widths))
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
    projection_floor = math.ceil(projection_macs / (4 * array_count * 64 * 64))
    array_latencies = {int(value) for value in ARRAY_LATENCY.findall(log_text)}
    if array_compute_cycles is not None:
        array_latencies.add(array_compute_cycles)
    if len(array_latencies) > 1:
        raise ValueError("inconsistent array compute latency")
    array_latency = next(iter(array_latencies), 1)
    projection_floor *= array_latency
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
                           "jobs": len(projection), "array_count": array_count,
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
        "projection_full_width_compute_cycles": array_latency,
        "floor_scope": "RMSNorm vector work + projection launches at configured full-width array latency + causal attention exp lanes; excludes DMA, NoC, control and handoff",
    }
    if all(event[7] is not None for event in projection):
        result["stages"]["projection"]["input_loads"] = sum(event[7] for event in projection)
    if local_gm:
        if not array_latencies:
            raise ValueError("missing projection array compute latency")
        if (sorted(event[0] for event in local_gm) != list(range(4)) or
                any(event[1] <= 0 or event[3] <= 0 or event[5] != 1
                    for event in local_gm)):
            raise ValueError("incomplete projection Local-GM timing")
        for core, read_bytes, write_bytes, _, _, _, reuse, paired in local_gm:
            job = next(event for event in projection if event[0] == core)
            if paired:
                groups = (sequence // 4 // array_count) * (hq + 2 * hkv)
                blocks = (sequence // 4 // 256) * (hq + 2 * hkv)
                programs = blocks * (hidden // 64)
                partials = groups * (hidden // 128 - 1) * array_count * 128
                expected_reads = groups * array_count * hidden * 2 + programs * 8192 + partials
                overlap = pipeline_by_core.get(core)
                shared = overlap[2] if overlap else 0
                expected_inputs = sequence // 4 // 256 if shared else blocks
                expected_weights = programs if shared else (hq + 2 * hkv) * (hidden // 64)
                if overlap and overlap[1]:
                    expected_prefetches = blocks * (hidden // 128) * (256 // array_count - 1)
                    expected_partial_prefetches = blocks * (hidden // 128 - 1) * (256 // array_count - 1)
                    if (overlap[3] != expected_prefetches or
                            overlap[4] != expected_partial_prefetches or
                            overlap[5] <= 0 or overlap[6] <= 0 or
                            overlap[7] <= 0 or overlap[8] != array_count * 512):
                        raise ValueError("projection overlap work mismatch")
                if (dim != 64 or not reuse or hidden % 128 or
                        read_bytes != expected_reads or write_bytes != partials or
                        job[5] != programs or job[6] != groups * (hidden // 64) - programs or
                        job[7] != expected_inputs or job[4] != expected_weights):
                    raise ValueError("paired projection work or byte accounting mismatch")
            if not reuse and job[7] is not None:
                expected = job[7] * array_count * hidden * 2 + job[5] * 8192
                if read_bytes != expected or write_bytes != 0:
                    raise ValueError("projection Local-GM byte accounting mismatch")
        names = ("read_bytes", "write_bytes", "read_cycles", "write_cycles",
                 "timed", "reuse_block", "paired_weights")
        result["stages"]["projection"]["local_gm_by_manager"] = {
            str(event[0]): dict(zip(names, event[1:])) for event in local_gm
        }
    if pipelines:
        names = ("enabled", "shared_input", "input_prefetches", "partial_prefetches",
                 "scatter_prefetches", "background_read_cycles", "background_write_cycles",
                 "staging_bytes", "array_count")
        result["stages"]["projection"]["pipeline_by_manager"] = {
            str(event[0]): dict(zip(names, event[1:])) for event in pipelines}
    weight_prefetch = [tuple(map(int, match)) for match in WEIGHT_PREFETCH.findall(log_text)]
    if weight_prefetch:
        if sorted(event[0] for event in weight_prefetch) != list(range(4)):
            raise ValueError("incomplete projection weight prefetch evidence")
        for core, enabled, requests, hits, waits, inflight in weight_prefetch:
            loads = next(event[4] for event in projection if event[0] == core)
            if (enabled not in (0, 1) or inflight or hits != requests or requests > loads or
                    (not enabled and (requests or hits or waits)) or (enabled and not requests)):
                raise ValueError("projection weight prefetch accounting mismatch")
        names = ("enabled", "requests", "hits", "wait_cycles", "inflight")
        result["stages"]["projection"]["weight_prefetch_by_manager"] = {
            str(event[0]): dict(zip(names, event[1:])) for event in weight_prefetch}
    rope_tables = [tuple(map(int, event)) for event in ROPE_TABLE.findall(log_text)]
    if rope_tables:
        names = ("bytes", "elapsed_cycles", "dma_cycles", "local_read_cycles",
                 "start_cycle", "end_cycle")
        result["attention_rope_table_first_load_by_worker"] = {
            str(event[0]): dict(zip(names, event[1:])) for event in rope_tables
        }
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
    parser.add_argument("--run-config", type=Path)
    args = parser.parse_args()
    array_cycles = None
    if args.run_config:
        config = args.run_config.read_text()
        mac = re.search(r"GOLEM_ARRAY_MAC_PER_CU_PER_CYCLE=([0-9.]+)", config)
        depth = re.search(r"GOLEM_ARRAY_PIPELINE_DEPTH=(\d+)", config)
        if not mac or not depth:
            raise ValueError("missing array timing configuration")
        array_cycles = math.ceil(64 / float(mac[1])) + int(depth[1])
    report = summarize(args.log.read_text(errors="replace"),
                       json.loads(args.attention_result.read_text()), array_cycles)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
