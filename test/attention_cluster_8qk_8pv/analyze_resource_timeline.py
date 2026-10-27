#!/usr/bin/env python3
"""Validate resource-level timestamps and summarize wait, service and overlap.

Input timestamps are SST core ticks. RESOURCE_TIMEBASE records provide the
conversion to the architecture's 1 GHz cycles; no picosecond assumption is used.
Transport intervals include NoC and memory time and must not be summed with
the nested HBM / local-GM intervals as independent work.
"""
import argparse
from collections import defaultdict
from itertools import chain
import json
from pathlib import Path
import re

FIELDS = re.compile(r"(\w+)=([^\s]+)")


def merged(spans):
    result = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def intersection(left, right):
    left, right = merged(left), merged(right)
    i = j = total = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def summarize(lines, output_ranges=()):
    seen = set()
    resources = {}
    spans = defaultdict(list)
    periods = set()
    controls = defaultdict(dict)
    dma_reads = defaultdict(dict)
    root_done_tick = 0
    last_output_ack_tick = 0
    output_ack_count = 0
    output_ack_bytes = 0
    projection_jobs = {}
    for line in lines:
        if '[PROJECTION_JOB]' in line:
            job = dict(FIELDS.findall(line))
            projection_jobs[int(job['core'])] = (int(job['start']), int(job['end']))
            continue
        if '[ATTENTION_MILESTONE]' in line and 'stage=root_tensor_complete ' in line and 'status=done ' in line:
            root_done_tick = max(root_done_tick, int(dict(FIELDS.findall(line))['sst_tick']))
            continue
        if "[RESOURCE_TIMEBASE]" in line:
            periods.add(int(dict(FIELDS.findall(line))["ticks_per_cycle"]))
            continue
        if "[CONTROL_EVENT]" in line:
            record = dict(FIELDS.findall(line))
            target = int(record["target"])
            stage = record["stage"]
            if stage in controls[target]:
                raise ValueError(f"duplicate control event: {target} {stage}")
            controls[target][stage] = int(record["tick"])
            continue
        if "[RESOURCE_INTERVAL]" not in line:
            continue
        raw = dict(FIELDS.findall(line))
        required = {"domain", "resource", "core", "request", "tag", "generation",
                    "submit_tick", "start_tick", "done_tick", "visible_tick", "bytes", "ok", "queue_depth"}
        if not required <= raw.keys():
            raise ValueError(f"incomplete resource interval: {raw}")
        record = {key: value if key in ("domain", "resource") else int(value)
                  for key, value in raw.items()}
        key = (record["domain"], record["resource"], record["core"], record["request"])
        if key in seen:
            raise ValueError(f"duplicate resource interval: {key}")
        seen.add(key)
        submit, start, done, visible = (record[name] for name in (
            "submit_tick", "start_tick", "done_tick", "visible_tick"))
        if not 0 <= submit <= start <= done <= visible:
            raise ValueError(f"resource timestamp order violation: {key}")
        if record["ok"] != 1:
            raise ValueError(f"failed or cancelled resource interval: {key}")
        resource = record["domain"] + "." + record["resource"]
        summary = resources.setdefault(resource, {"count": 0, "bytes": 0,
            "queue_ticks": 0, "service_ticks": 0, "visibility_delay_ticks": 0,
            "max_queue_ticks": 0, "max_service_ticks": 0, "max_queue_depth": 0,
            "by_core": {}})
        per_core = summary["by_core"].setdefault(str(record["core"]), {
            "count": 0, "bytes": 0, "queue_ticks": 0, "service_ticks": 0,
            "visibility_delay_ticks": 0})
        for target in (summary, per_core):
            target["count"] += 1
            target["bytes"] += record["bytes"]
            target["queue_ticks"] += start - submit
            target["service_ticks"] += done - start
            target["visibility_delay_ticks"] += visible - done
        summary["max_queue_ticks"] = max(summary["max_queue_ticks"], start - submit)
        summary["max_service_ticks"] = max(summary["max_service_ticks"], done - start)
        summary["max_queue_depth"] = max(summary["max_queue_depth"], record["queue_depth"])
        spans[(record["core"], resource)].append((start, done))
        if record['domain'] == 'gm' and record['resource'] == 'dma_write_transport':
            is_output = record.get('traffic') == 4 or any(
                lower <= record.get('address', -1) and
                record.get('address', -1) + record['bytes'] <= upper
                for lower, upper in output_ranges)
            if is_output:
                last_output_ack_tick = max(last_output_ack_tick, visible)
                output_ack_count += 1
                output_ack_bytes += record['bytes']
        if record["domain"] == "gm" and record["resource"] in ("dma_read_transport", "dma_read_landing"):
            dma_reads[(record["core"], record["request"])][record["resource"]] = record
    if not resources:
        raise ValueError("resource timeline is absent")
    if len(periods) != 1 or next(iter(periods)) <= 0:
        raise ValueError("missing or inconsistent RESOURCE_TIMEBASE")
    factor = next(iter(periods))
    for key, pair in dma_reads.items():
        if len(pair) != 2:
            raise ValueError(f"missing DMA arrival/landing pair: {key}")
        transport, landing = pair["dma_read_transport"], pair["dma_read_landing"]
        if (transport["visible_tick"] != landing["submit_tick"] or
                transport["tag"] != landing["tag"] or
                transport["generation"] != landing["generation"] or
                transport["bytes"] != landing["bytes"]):
            raise ValueError(f"mismatched DMA generation/landing pair: {key}")
    for resource, summary in resources.items():
        for core, totals in summary["by_core"].items():
            totals["busy_union_cycles"] = sum(end - start for start, end in merged(spans[(int(core), resource)])) / factor
        for source, dest in (("queue_ticks", "queue_cycles"), ("service_ticks", "service_cycles"),
                             ("visibility_delay_ticks", "visibility_delay_cycles"),
                             ("max_queue_ticks", "max_queue_cycles"), ("max_service_ticks", "max_service_cycles")):
            summary[dest] = summary[source] / factor
    overlap = {}
    for core in range(4, 20):
        compute = spans[(core, "array.compute")]
        read = spans[(core, "gm.local_read")]
        buffer = [span for (owner, resource), values in list(spans.items())
                  if owner == core and resource.startswith("array.") and resource != "array.compute" for span in values]
        overlap[str(core)] = {"gm_read_compute_cycles": intersection(read, compute) / factor,
                              "array_buffer_compute_cycles": intersection(buffer, compute) / factor}
    control_summary = {}
    for target, stages in controls.items():
        names = ("projection_release_issue", "projection_release_observed", "projection_accept")
        if not all(name in stages for name in names):
            raise ValueError(f"missing release/observed/accept event: {target}")
        issue, observed, accept = [stages[name] for name in names]
        if not issue <= observed <= accept:
            raise ValueError(f"control timestamp order violation: {target}")
        control_summary[str(target)] = {
            "release_issue_cycle": issue / factor, "release_observed_cycle": observed / factor,
            "job_accept_cycle": accept / factor,
            "release_to_observed_cycles": (observed - issue) / factor,
            "observed_to_accept_cycles": (accept - observed) / factor}
    projection_overlap = {}
    for core, (start_cycle, end_cycle) in projection_jobs.items():
        lower, upper = start_cycle * factor, end_cycle * factor
        def clipped(resource):
            return [(max(lower, start), min(upper, end)) for start, end in spans[(core, resource)]
                    if start < upper and end > lower]
        compute = clipped('array.compute')
        gm = clipped('gm.local_read')
        matrix = clipped('array.matrix_broadcast')
        inputs = clipped('array.input_scatter')
        projection_overlap[str(core)] = {
            'worker_elapsed_cycles': end_cycle - start_cycle,
            'compute_busy_cycles': sum(b - a for a, b in merged(compute)) / factor,
            'gm_read_busy_cycles': sum(b - a for a, b in merged(gm)) / factor,
            'matrix_program_busy_cycles': sum(b - a for a, b in merged(matrix)) / factor,
            'input_scatter_busy_cycles': sum(b - a for a, b in merged(inputs)) / factor,
            'gm_read_compute_overlap_cycles': intersection(gm, compute) / factor,
            'matrix_program_compute_overlap_cycles': intersection(matrix, compute) / factor,
            'input_scatter_compute_overlap_cycles': intersection(inputs, compute) / factor,
        }
    return {"status": "PASS", "ticks_per_cycle": factor,
            "interval_count": len(seen), "resources": resources,
            "worker_control": control_summary, "worker_overlap": overlap,
            "projection_worker_overlap": projection_overlap,
            "dma_arrival_landing_pairs": len(dma_reads),
            "attention_output_ack_count": output_ack_count,
            "attention_output_ack_bytes": output_ack_bytes,
            "last_attention_output_ack_cycle": last_output_ack_tick / factor,
            "last_root_tensor_complete_cycle": root_done_tick / factor,
            "output_ack_to_root_cycles": (root_done_tick - last_output_ack_tick) / factor if output_ack_count and root_done_tick else None,
            "scope": "Resource intervals overlap and may nest. Transport RTT is not pure HBM service. cbuffer_model intervals are scheduled reservations. Snapshot visibility delay is ready-to-consume residence."}


def analyze(case, allow_filtered=False):
    case = Path(case)
    logs = list((case / "logs").glob("attention-sst*.log"))
    if len(logs) != 1:
        raise ValueError("expected one merged SST log")
    sidecar = logs[0].with_suffix('.resources.jsonl')
    # Worker output DMA currently uses traffic=Unknown. Classify it offline
    # from the verified layout rather than changing a scheduler-facing kind.
    layout = json.loads((case / 'attention_hbm_layout.json').read_text())
    if layout.get('status') != 'PASS':
        raise ValueError('output address attribution requires a verified HBM layout')
    output_ranges = [(node['node'] * node['capacity_bytes'] + region['begin'],
                      node['node'] * node['capacity_bytes'] + region['end'])
                     for node in layout['nodes'] for region in node['regions']
                     if region['name'] == 'O']
    if not output_ranges:
        raise ValueError('verified layout has no O output regions')
    with logs[0].open(errors="replace") as stream:
        if sidecar.exists():
            with sidecar.open(errors="replace") as resources:
                result = summarize(chain(stream, resources), output_ranges)
        else:
            result = summarize(stream, output_ranges)
    if set(result["worker_control"]) != {str(core) for core in range(4, 20)}:
        raise ValueError("release timeline does not cover all 16 workers")
    required_resources = ("gm.local_read", "gm.local_write", "gm.dma_read_transport",
                     "gm.dma_read_landing", "gm.dma_write_transport",
                     "array.matrix_broadcast", "array.input_scatter", "wcp.command_queue",
                     "wcp.array_compute", "array.compute")
    if not allow_filtered:
        required_resources += ("hbm.dram_read", "hbm.dram_write", "hbm.backend_admission")
    for resource in required_resources:
        if resource not in result["resources"]:
            raise ValueError(f"missing expected resource: {resource}")
    result["artifact_root"] = str(case)
    result['filtered_resource_scope'] = allow_filtered
    result['output_address_ranges'] = output_ranges
    if not result['attention_output_ack_count'] or result['output_ack_to_root_cycles'] is None:
        raise ValueError('missing output ACK or root completion boundary')
    if result['output_ack_to_root_cycles'] < 0:
        raise ValueError('root completed before the last output ACK')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequences", type=int, nargs="+", default=[1024, 2048])
    parser.add_argument('--allow-filtered', action='store_true', help='GM/array/WCP/projection trace without HBM/SFU intervals')
    args = parser.parse_args()
    cases = [analyze(args.artifact_root / f"s{sequence}", args.allow_filtered) for sequence in args.sequences]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"status": "PASS", "cases": cases}, indent=2) + "\n")
    for case in cases:
        print(f"{Path(case['artifact_root']).name}: PASS, {case['interval_count']} resource intervals, "
              f"{case['dma_arrival_landing_pairs']} DMA arrival/landing pairs")


if __name__ == "__main__":
    main()
