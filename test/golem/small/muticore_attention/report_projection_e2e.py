#!/usr/bin/env python3
"""Summarize the measured RMSNorm -> projection -> causal attention path."""

import argparse
import json
import math
import re
from pathlib import Path


RMS = re.compile(r"\[SFU_RMSNORM\] core=(\d+).*?issue_tick=(\d+).*?complete_tick=(\d+).*?vector_cycles=(\d+) status=(\d+)")
PROJECTION = re.compile(r"\[PROJECTION_JOB\] (?:manager|core)=(\d+) start=(\d+) end=(\d+) cycles=(\d+) weight_loads=(\d+) weight_programs=(\d+) weight_reuses=(\d+)(?: input_loads=(\d+))? status=(\d+)")
PHASE = re.compile(r"\[PROJECTION_PHASE\] (?:manager|core)=(\d+) input_dma=(\d+) weight_dma=(\d+) matrix_program=(\d+) input_scatter=(\d+) output_restore=(\d+) compute=(\d+) output_read=(\d+) write_drain=(\d+)")
LOCAL_GM = re.compile(r"\[PROJECTION_LOCAL_GM\] (?:manager|core)=(\d+) read_bytes=(\d+) write_bytes=(\d+) read_cycles=(\d+) write_cycles=(\d+) timed=(\d+) reuse_block=(\d+)(?: paired_weights=(\d+))?")
PIPELINE = re.compile(r"\[PROJECTION_PIPELINE\] (?:manager|core)=(\d+) enabled=(\d+) shared_input=(\d+) input_prefetches=(\d+) partial_prefetches=(\d+) scatter_prefetches=(\d+) background_read_cycles=(\d+) background_write_cycles=(\d+) staging_bytes=(\d+)(?: array_count=(\d+))?")
ARRAY_LATENCY = re.compile(r"MVM compute latency cycles=(\d+)")
WEIGHT_PREFETCH = re.compile(r"\[PROJECTION_WEIGHT_PREFETCH\] (?:manager|core)=(\d+) enabled=(\d+) requests=(\d+) hits=(\d+) wait_cycles=(\d+) inflight=(\d+)")
SCHEDULE = re.compile(r"\[PROJECTION_SCHEDULE\] (?:manager|core)=(\d+) row_first=(\d+) cbuffer_partial=(\d+) cbuffer_bytes=(\d+) cbuffer_read_bytes=(\d+) cbuffer_write_bytes=(\d+)")
ROPE_TABLE = re.compile(r"\[ATTENTION_ROPE_TABLE\] core=(\d+) bytes=(\d+) cycles=(\d+) dma_cycles=(\d+) local_read_cycles=(\d+) start=(\d+) end=(\d+)")
FUSION = re.compile(r"\[PROJECTION_FUSION\] (?:manager|core)=(\d+) norm=(\d+) rope=(\d+) audit=(\d+) norm_jobs=(\d+) rope_jobs=(\d+) norm_vector_cycles=(\d+) rope_vector_cycles=(\d+) local_read_bytes=(\d+) local_write_bytes=(\d+) norm_audit_bytes=(\d+) raw_audit_bytes=(\d+) panel_bytes=(\d+)")
SYNC = re.compile(r"\[PROJECTION_SYNC\] core=(\d+) stage=flag_wait cycle=(\d+) flag=(\d+) status=(\d+)")
LOCAL_WAIT = re.compile(r"\[PROJECTION_SYNC\] core=(\d+) stage=local_wait cycle=(\d+) status=(\d+)")
DESCRIPTOR = re.compile(r"\[ATTENTION_MILESTONE\] stage=(?:root_|manager_)descriptor_accept status=done .*?rocc_cycle=(\d+) core=(\d+)")


def summarize(log_text, attention, array_compute_cycles=None):
    ownership = [tuple(map(int, e)) for e in re.findall(
        r"\[PROJECTION_WORKER_OWNER\] core=(\d+) manager=(\d+) lane=(\d+) row_offset=(\d+) row_count=(\d+)", log_text)]
    worker_projection = bool(ownership)
    execution_cores = list(range(4, 20)) if worker_projection else list(range(4))
    execution_count = len(execution_cores)
    row_span = 64 if worker_projection else 256
    accumulation = {int(c): {'retain_output': int(enabled), 'retained_pairs': int(pairs)}
                    for c, enabled, pairs in re.findall(
                        r'\[PROJECTION_ACCUMULATION\] core=(\d+) retain_output=(\d+) retained_pairs=(\d+)', log_text)}
    if accumulation and (sorted(accumulation) != execution_cores or
            len({e['retain_output'] for e in accumulation.values()}) != 1 or
            any(e['retain_output'] not in (0, 1) for e in accumulation.values())):
        raise ValueError('invalid projection accumulator ownership')
    dataflow = {}
    for record in re.findall(r'\[PROJECTION_DATAFLOW\] ([^\n]+)', log_text):
        fields = {key: int(value) for key, value in re.findall(r'(\w+)=(\d+)', record)}
        core = fields.pop('core')
        if core in dataflow:
            raise ValueError('duplicate projection dataflow evidence')
        dataflow[core] = fields
    if dataflow and (sorted(dataflow) != execution_cores or
            len({(f['input_pair_prefetch'], f['local_weight_prefetch'], f['array_groups'])
                 for f in dataflow.values()}) != 1 or
            any(f['input_pair_prefetch'] not in (0, 1) or f['local_weight_prefetch'] not in (0, 1) or
                f['array_groups'] not in (1, 2) or
                ((f['input_pair_prefetch'] or f['local_weight_prefetch'] or f['array_groups'] == 2) and
                 (not worker_projection or not accumulation.get(c, {}).get('retain_output')))
                for c, f in dataflow.items())):
        raise ValueError('invalid projection dataflow configuration')
    if worker_projection:
        if sorted(e[0] for e in ownership) != execution_cores or any(
                e != (c, (c-4)%4, (c-4)//4, ((c-4)//4)*64, 64) for e in ownership for c in [e[0]]):
            raise ValueError("invalid projection worker ownership")
        resources = {int(c): (int(a), int(f)) for c, a, f in re.findall(
            r"\[CORE_RESOURCE_OWNERSHIP\] core=(\d+) array_present=(\d+) sfu_present=(\d+)", log_text)}
        if sorted(resources) != list(range(20)) or any(resources[c] != ((0, 0) if c < 4 else (1, 1)) for c in resources):
            raise ValueError("manager/worker compute resource violation")
    operand_storage = [tuple(map(int, e)) for e in re.findall(
        r'\[ARRAY_OPERAND_STORAGE\] core=(\d+) arrays=(\d+) operand_banks=(\d+) matrix_contexts=(\d+) input_contexts=(\d+)', log_text)]
    if operand_storage and (not worker_projection or sorted(e[0] for e in operand_storage) != execution_cores or
            any(e[1:] != (64, 1, 64, 64) for e in operand_storage)):
        raise ValueError('extra or missing array operand storage')
    fusion = [tuple(map(int, event)) for event in FUSION.findall(log_text)]
    if fusion and (sorted(event[0] for event in fusion) != execution_cores or
                   len({event[1:4] for event in fusion}) != 1):
        raise ValueError("incomplete or inconsistent fusion evidence")
    fused_norm = bool(fusion and fusion[0][1])
    fused_rope = bool(fusion and fusion[0][2])
    rms = [tuple(map(int, match)) for match in RMS.findall(log_text)]
    projection = [tuple(int(value) if value else None for value in match)
                  for match in PROJECTION.findall(log_text)]
    phases = [tuple(map(int, match)) for match in PHASE.findall(log_text)]
    cycle_breakdown = {}
    for record in re.findall(r"\[PROJECTION_CYCLE_BREAKDOWN\] ([^\n]+)", log_text):
        fields = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", record)}
        core = fields.pop('core') if 'core' in fields else fields.pop('manager')
        if core in cycle_breakdown:
            raise ValueError("duplicate projection cycle breakdown")
        if sum(fields[k] for k in ('local_weight_read', 'local_input_read', 'local_fusion_read', 'local_other_read')) != fields['local_read']:
            raise ValueError("projection local-read cycle partition mismatch")
        cycle_breakdown[core] = fields
    if cycle_breakdown and sorted(cycle_breakdown) != execution_cores:
        raise ValueError("incomplete projection cycle breakdown")
    partial_movement = {}
    for record in re.findall(r"\[PROJECTION_PARTIAL_MOVEMENT\] ([^\n]+)", log_text):
        fields = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", record)}
        core = fields.pop('core') if 'core' in fields else fields.pop('manager')
        if core in partial_movement:
            raise ValueError("duplicate projection partial movement")
        partial_movement[core] = fields
    if partial_movement and (sorted(partial_movement) != execution_cores or not cycle_breakdown):
        raise ValueError("incomplete projection partial movement")
    input_flow = {}
    for record in re.findall(r"\[PROJECTION_INPUT_FLOW\] ([^\n]+)", log_text):
        fields = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", record)}
        core = fields.pop('core') if 'core' in fields else fields.pop('manager')
        if core in input_flow:
            raise ValueError("duplicate projection input flow")
        input_flow[core] = fields
    if input_flow and sorted(input_flow) != execution_cores:
        raise ValueError("incomplete projection input flow")
    local_gm = [tuple(int(value) if value else 0 for value in match)
                for match in LOCAL_GM.findall(log_text)]
    pipelines = [tuple(int(value) if value else 16 for value in match)
                 for match in PIPELINE.findall(log_text)]
    if pipelines and sorted(event[0] for event in pipelines) != execution_cores:
        raise ValueError("incomplete projection overlap evidence")
    pipeline_by_core = {event[0]: event for event in pipelines}
    schedules = [tuple(map(int, match)) for match in SCHEDULE.findall(log_text)]
    if schedules and sorted(event[0] for event in schedules) != execution_cores:
        raise ValueError("incomplete projection schedule evidence")
    schedule_by_core = {event[0]: event for event in schedules}
    if schedules and len({event[1:] for event in schedules}) != 1:
        raise ValueError("inconsistent projection schedule")
    widths = {event[9] for event in pipelines} or {16}
    if len(widths) != 1 or not widths <= {16, 32, 64}:
        raise ValueError("invalid projection array count")
    array_count = next(iter(widths))
    # The shared HBM-flag instruction also waits for optional decoder jobs.
    # Exclude only their explicitly reserved flags, retaining strict checks on
    # the original Projection barrier and all other unexpected records.
    decoder_flags = {(node+1)*0x08000000+0x01ff0100+lane*8
                     for node in range(4) for lane in range(4)}
    sync = [event for match in SYNC.findall(log_text)
            if (event := tuple(map(int, match)))[2] not in decoder_flags]
    local_wait = [tuple(map(int, match)) for match in LOCAL_WAIT.findall(log_text)]
    descriptors = [tuple(map(int, event)) for event in DESCRIPTOR.findall(log_text)]
    shape = attention["shape"]
    hq, hkv = shape["Hq"], shape["Hkv"]
    sequence, dim = shape["query_length"], shape["head_dim"]
    hidden = hq * dim
    rms_batch_rows = 8192 // hidden
    expected_rms_jobs = (4 * (sequence // 1024) * math.ceil(256 / rms_batch_rows)
                         if fused_norm else 4 * math.ceil((sequence // 4) / rms_batch_rows))
    if (attention["status"] != "PASS" or not shape["causal"] or
            shape["kv_length"] != sequence or len(rms) != expected_rms_jobs or
            len(projection) != execution_count or
            any(sum(event[0] == core for event in rms) != expected_rms_jobs // execution_count
                for core in execution_cores) or
            sorted(event[0] for event in projection) != execution_cores or
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
                       for core in execution_cores}
    if (not (rms_start < rms_end and projection_start < projection_end <=
             attention_start < attention_end) or
            (not fused_norm and any(event[1] * 1000 < rms_end_by_core[event[0]] for event in projection))):
        raise ValueError("stage timestamps are not ordered")
    rms_core_work = {}
    for core, _, _, vector_cycles, _ in rms:
        rms_core_work[core] = rms_core_work.get(core, 0) + vector_cycles
    projection_macs = sequence * (hq + 2 * hkv) * dim * hidden
    projection_floor = math.ceil(projection_macs / (execution_count * array_count * 64 * 64))
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
    chain_start = min(rms_start, projection_start) if fused_norm else rms_start
    actual = attention_end - chain_start
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
    if fusion:
        names = ("norm", "rope", "audit", "norm_jobs", "rope_jobs", "norm_vector_cycles",
                 "rope_vector_cycles", "local_read_bytes", "local_write_bytes", "norm_audit_bytes",
                 "raw_audit_bytes", "panel_bytes")
        rows = sequence // execution_count
        norm_bytes = rows * hidden * 2
        output_bytes = rows * (hq + 2 * hkv) * dim * 2
        for event in fusion:
            core, norm, rope, audit, nj, rj, nc, rc, reads, writes, na, ra, panels = event
            expected_nj = math.ceil(rows / rms_batch_rows) if norm else 0
            expected_rj = rows // array_count * (hq + hkv) * (dim // 64) if rope else 0
            rope_bytes = rows * (hq + hkv) * dim * 2 if rope else 0
            expected_reads = (expected_nj * hidden * 2 + norm_bytes if norm else 0) + 2 * rope_bytes
            expected_writes = (norm_bytes if norm else 0) + rope_bytes
            if (nj != expected_nj or rj != expected_rj or
                reads != expected_reads or writes != expected_writes or
                na != (norm_bytes if norm and audit else 0) or
                ra != (output_bytes if audit else 0) or panels != output_bytes or
                bool(nc) != bool(norm) or bool(rc) != bool(rope)):
                raise ValueError("fusion work/traffic accounting mismatch")
        result["fusion_by_manager"] = {str(e[0]): dict(zip(names, e[1:])) for e in fusion}
        if fused_rope:
            reads = {int(c): int(b) for c, b in re.findall(
                r"GOLEM_LOCAL_GM_STATS core=(\d+).*?gmem_local_read_bytes=(\d+)", log_text)
                if 4 <= int(c) <= 11}
            # Projection and QK now reuse the same worker GM. Its lifetime
            # counters include both stages; remove the independently checked
            # projection operand and fusion payload before checking QK reads.
            if worker_projection:
                projection_reads = {e[0]: e[1] for e in local_gm}
                fusion_reads = {e[0]: e[8] for e in fusion}
                reads = {c: b - projection_reads[c] - fusion_reads[c]
                         for c, b in reads.items()}
            expected = causal_tiles * 2 * 64 * 64 * 2 * (dim // 64)
            if (len(reads) != 8 or any(b == 0 for b in reads.values()) or
                    sum(reads.values()) != expected):
                raise ValueError("missing or inconsistent fused QK operand timing")
            result["fusion_qk_local_read_bytes"] = {
                "expected": expected, "observed": sum(reads.values()),
                "by_worker": {str(c): b for c, b in reads.items()}}

        result["stages"]["projection"]["includes_rmsnorm"] = fused_norm
        result["stages"]["projection"]["includes_rope"] = fused_rope
        result["stages"]["rmsnorm"]["active_cycles_by_manager"] = {
            str(core): sum(math.ceil((e[2] - e[1]) / 1000) for e in rms if e[0] == core)
            for core in execution_cores}
        result["chain_start_cycle"] = chain_start
    if all(event[7] is not None for event in projection):
        result["stages"]["projection"]["input_loads"] = sum(event[7] for event in projection)
    if local_gm:
        if not array_latencies:
            raise ValueError("missing projection array compute latency")
        if (sorted(event[0] for event in local_gm) != execution_cores or
                any(event[1] <= 0 or event[3] <= 0 or event[5] != 1
                    for event in local_gm)):
            raise ValueError("incomplete projection Local-GM timing")
        for core, read_bytes, write_bytes, _, _, _, reuse, paired in local_gm:
            job = next(event for event in projection if event[0] == core)
            row_first = schedule_by_core.get(core, (0, 0, 0, 0))[1]
            cbuffer = schedule_by_core.get(core, (0, 0, 0, 0))[2]
            retention = accumulation.get(core, {}).get('retain_output', 0)
            if retention and (not worker_projection or array_count != 64 or row_first or cbuffer or not paired):
                raise ValueError('invalid projection accumulator schedule')
            if paired:
                groups = (sequence // execution_count // array_count) * (hq + 2 * hkv)
                blocks = (sequence // 1024) * (hq + 2 * hkv)
                plan = dataflow.get(core, {})
                array_groups = plan.get('array_groups', 1)
                weight_tiles = (groups if row_first or operand_storage else blocks) * (hidden // 64)
                programs = weight_tiles * array_groups
                partials = (0 if row_first or retention else
                            groups * (hidden // 128 - 1) * array_count * 128)
                expected_retained = groups * (hidden // 128 - 1) if retention else 0
                if accumulation.get(core, {}).get('retained_pairs', 0) != expected_retained:
                    raise ValueError('projection accumulator work accounting mismatch')
                input_bytes = groups * array_count * hidden * 2
                expected_reads = input_bytes + weight_tiles * 8192 + (0 if cbuffer else partials)
                overlap = pipeline_by_core.get(core)
                shared = overlap[2] if overlap else 0
                expected_inputs = blocks if row_first else sequence // 1024 if shared else blocks
                expected_weights = ((hq + 2 * hkv) * (hidden // 64) if row_first else
                                    blocks * (hidden // 64) if shared else (hq + 2 * hkv) * (hidden // 64))
                if overlap and overlap[1]:
                    expected_prefetches = blocks * (hidden // 128) * (row_span // array_count - 1)
                    expected_partial_prefetches = blocks * (hidden // 128 - 1) * (row_span // array_count - 1)
                    if cycle_breakdown.get(core, {}).get('wrap_prefetch') and array_count < row_span:
                        expected_prefetches += blocks * (hidden // 128 - 1)
                        expected_partial_prefetches += blocks * (hidden // 128 - 1)
                    if plan.get('input_pair_prefetch'):
                        expected_prefetches += blocks * (hidden // 128 - 1)
                    if (overlap[3] != expected_prefetches or
                            overlap[4] not in ((0, expected_partial_prefetches) if cbuffer else
                                               (expected_partial_prefetches,)) or
                            (expected_prefetches > 0 and ((overlap[5] != 0 if operand_storage else overlap[5] <= 0) or overlap[6] <= 0)) or
                            (not cbuffer and expected_partial_prefetches > 0 and overlap[7] <= 0) or
                            overlap[8] != array_count * 512):
                        raise ValueError("projection overlap work mismatch")
                if (dim != 64 or not reuse or hidden % 128 or
                        read_bytes != expected_reads or write_bytes != (0 if cbuffer else partials) or
                        job[5] != programs or job[6] != groups * (hidden // 64) * array_groups - programs or
                        job[7] != expected_inputs or job[4] != expected_weights):
                    raise ValueError("paired projection work or byte accounting mismatch")
                if cbuffer:
                    schedule = schedule_by_core[core]
                    if schedule[3] != 256 * 128 or schedule[4] != partials or schedule[5] != partials:
                        raise ValueError("projection C-buffer byte accounting mismatch")
                if core in input_flow:
                    flow = input_flow[core]
                    limit = flow['max_request_bytes']
                    if limit <= 0:
                        raise ValueError("projection input flow request accounting mismatch")
                    packed = flow['packed_input']
                    reads_per_pair = (math.ceil(array_count * 256 / limit) if packed else
                                      array_count * math.ceil(256 / limit)) if limit else 0
                    expected_norm_writes = 0
                    if fused_norm:
                        if packed:
                            expected_norm_writes = (sequence // execution_count // rms_batch_rows) * (hidden // 128) * math.ceil(rms_batch_rows * 256 / limit)
                        else:
                            expected_norm_writes = sum(math.ceil(min(rms_batch_rows, row_span - r) * hidden * 2 / limit)
                                                      for r in range(0, row_span, rms_batch_rows)) * (sequence // 1024)
                    if (not limit or packed not in (0, 1) or flow['input_prefetch_budget'] > 64 or
                            flow['input_read_requests'] + flow['background_input_read_requests'] != groups * (hidden // 128) * reads_per_pair or
                            flow['weight_read_requests'] != weight_tiles * math.ceil(8192 / limit) or
                            flow['norm_write_requests'] != expected_norm_writes or
                            (packed and (not fused_norm or audit or row_first or array_count % rms_batch_rows))):
                        raise ValueError("projection input flow request accounting mismatch")
                if plan:
                    if (plan['array_launches'] != groups * (hidden // 64) * array_groups or
                            plan['local_weight_prefetches'] != plan['local_weight_hits'] or
                            plan['local_weight_hits'] > weight_tiles - blocks or
                            (not plan['local_weight_prefetch'] and plan['local_weight_prefetches']) or
                            plan['buffer_compute_overlap_cycles'] > plan['compute_busy_cycles'] or
                            plan['gm_compute_overlap_cycles'] > plan['compute_busy_cycles']):
                        raise ValueError('projection dataflow work accounting mismatch')
                if core in partial_movement:
                    movement, cycles = partial_movement[core], cycle_breakdown[core]
                    partial_jobs = (groups * (hidden // 128 - 1)
                                    if not row_first and not retention else 0)
                    ready_parts = sum(movement[k] for k in ('restore_cbuffer_wait', 'restore_submit', 'restore_transfer', 'restore_ahead_wait', 'restore_ahead_consume'))
                    if (movement['partial_reads'] != partial_jobs or
                            movement['final_reads'] != groups or
                            movement['restores_foreground'] + movement['restores_ahead'] != partial_jobs or
                            movement['partial_read_cycles'] + movement['final_read_cycles'] != cycles['read_output'] or
                            ready_parts != cycles['restore_output_ready'] or
                            (not movement['restore_ahead'] and movement['restores_ahead']) or
                            (cbuffer and (movement['cbuffer_prefetch_reads'] + movement['cbuffer_demand_reads'] != partial_jobs or movement['cbuffer_stores'] != partial_jobs))):
                        raise ValueError("projection partial movement work/cycle accounting mismatch")
            if not reuse and job[7] is not None:
                expected = job[7] * array_count * hidden * 2 + job[5] * 8192
                if read_bytes != expected or write_bytes != 0:
                    raise ValueError("projection Local-GM byte accounting mismatch")
        names = ("read_bytes", "write_bytes", "read_cycles", "write_cycles",
                 "timed", "reuse_block", "paired_weights")
        result["stages"]["projection"]["local_gm_by_manager"] = {
            str(event[0]): dict(zip(names, event[1:])) for event in local_gm
        }
    if input_flow:
        result["stages"]["projection"]["input_flow_by_manager"] = {
            str(core): fields for core, fields in input_flow.items()}
    if dataflow:
        result['stages']['projection']['dataflow_by_manager'] = {
            str(core): fields for core, fields in dataflow.items()}
    if pipelines:
        names = ("enabled", "shared_input", "input_prefetches", "partial_prefetches",
                 "scatter_prefetches", "background_read_cycles", "background_write_cycles",
                 "staging_bytes", "array_count")
        result["stages"]["projection"]["pipeline_by_manager"] = {
            str(event[0]): dict(zip(names, event[1:])) for event in pipelines}
    if schedules:
        names = ("row_first", "cbuffer_partial", "cbuffer_bytes",
                 "cbuffer_read_bytes", "cbuffer_write_bytes")
        result["stages"]["projection"]["schedule_by_manager"] = {
            str(event[0]): dict(zip(names, event[1:])) for event in schedules}
    weight_prefetch = [tuple(map(int, match)) for match in WEIGHT_PREFETCH.findall(log_text)]
    if weight_prefetch:
        if sorted(event[0] for event in weight_prefetch) != execution_cores:
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
    if sorted(event[0] for event in phases) == execution_cores:
        names = ("input_dma", "weight_dma", "matrix_program", "input_scatter",
                 "output_restore", "compute", "output_read", "write_drain")
        result["stages"]["projection"]["phase_cycles_by_manager"] = {
            str(event[0]): dict(zip(names, event[1:])) for event in phases
        }
    if bool(sync) != bool(descriptors):
        raise ValueError("incomplete projection synchronization")
    if sync:
        if (len(sync) != (64 if worker_projection else 16) or any(event[3] for event in sync) or
                len(local_wait) != execution_count or
                sorted(event[0] for event in local_wait) != execution_cores or
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
                not projection_end <= max(sync_by_core.values()) or
                not min(descriptor_by_core.values(), default=0) <= attention_start or
                ('[ORACLE_EVENT] oracle=projection_barrier' not in log_text and
                 (sync_end > descriptor_end or
                  any(sync_by_core[core] > descriptor_by_core[core] for core in range(4))))):
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
    if cycle_breakdown:
        result['stages']['projection']['cycle_breakdown_by_manager'] = {
            str(core): fields for core, fields in cycle_breakdown.items()}
    if partial_movement:
        result['stages']['projection']['partial_movement_by_manager'] = {
            str(core): fields for core, fields in partial_movement.items()}
    result['resource_topology_valid'] = worker_projection
    prefetch_config = {int(c): (int(a), int(b), int(d), int(w or 0)) for c, a, b, d, w in re.findall(
        r'\[PROJECTION_PREFETCH_CONFIG\] core=(\d+) weight_lookahead=(\d+) input_single_slot=(\d+) partial_single_slot=(\d+)(?: weight_single_slot=(\d+))?', log_text)}
    if prefetch_config and (sorted(prefetch_config) != execution_cores or
                            any(v[0] < 1 or v[0] > 8 or v[1:] not in ((0, 0, 0), (1, 1, 0), (1, 0, 1)) for v in prefetch_config.values())):
        raise ValueError('invalid projection prefetch configuration')
    result['prefetch_config_by_worker'] = {str(c): {'weight_lookahead': v[0],
        'input_single_slot': v[1], 'partial_single_slot': v[2], 'weight_single_slot': v[3]}
        for c, v in prefetch_config.items()}
    result['array_operand_banks'] = 1 if operand_storage else None
    result['single_operand_context_valid'] = bool(operand_storage)
    if operand_storage:
        result['array_operand_storage_by_worker'] = {str(e[0]): {
            'arrays': e[1], 'operand_banks': e[2], 'matrix_contexts': e[3], 'input_contexts': e[4]}
            for e in operand_storage}
    if accumulation:
        result['stages']['projection']['accumulation_by_manager'] = {
            str(core): fields for core, fields in accumulation.items()}
    if worker_projection:
        result['projection_worker_ownership'] = ownership
        result['compute_resource_cores'] = execution_cores
        result['fusion_by_worker'] = result.pop('fusion_by_manager', {})
        for stage in ('projection', 'rmsnorm'):
            data = result['stages'][stage]
            for key in list(data):
                if key.endswith('_by_manager'):
                    data[key[:-len('_by_manager')] + '_by_worker'] = data.pop(key)
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
    if args.run_config:
        expected = {
            "norm": "GOLEM_FUSION_NORM_PROJECTION",
            "rope": "GOLEM_FUSION_PROJECTION_ROPE",
            "audit": "GOLEM_FUSION_SKIP_AUDIT_OUTPUT",
        }
        for field, key in expected.items():
            match = re.search(rf"^{key}=([01])$", config, re.MULTILINE)
            if match:
                value = int(match[1]) if field != "audit" else 1 - int(match[1])
                managers = report.get("fusion_by_worker", report.get("fusion_by_manager", {}))
                if len(managers) != (16 if report.get("resource_topology_valid") else 4) or any(e[field] != value for e in managers.values()):
                    raise ValueError(f"requested {key} differs from measured fusion path")
        retention = re.search(r'^GOLEM_PROJECTION_RETAIN_OUTPUT=([01])$', config, re.MULTILINE)
        if retention and report.get('resource_topology_valid'):
            projection = report['stages']['projection']
            modes = projection.get('accumulation_by_worker', {})
            if len(modes) != 16:
                raise ValueError('missing projection accumulator mode evidence')
            for core, mode in modes.items():
                gm = projection['local_gm_by_worker'][core]
                schedule = projection['schedule_by_worker'][core]
                expected_mode = int(bool(int(retention[1]) and projection['array_count'] == 64 and
                                         gm['paired_weights'] and not schedule['row_first']))
                if mode['retain_output'] != expected_mode:
                    raise ValueError('requested projection accumulator mode differs from execution')
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
