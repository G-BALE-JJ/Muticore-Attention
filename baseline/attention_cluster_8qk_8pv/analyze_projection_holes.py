#!/usr/bin/env python3
"""Classify gaps between Projection MAC service intervals from existing SST traces."""
import argparse
from collections import Counter, defaultdict
from bisect import bisect_left, bisect_right
import json
from pathlib import Path
import re


FIELDS = re.compile(r"(\w+)=([^\s]+)")


def merge(spans):
    result = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def clipped_union(spans, lower, upper):
    return merge((max(a, lower), min(b, upper)) for a, b in spans if a < upper and b > lower)


def intersecting(intervals, starts, ends, lower, upper):
    first = bisect_right(ends, lower)
    last = bisect_left(starts, upper)
    return intervals[first:last]


def analyze(case):
    case = Path(case)
    log, = (case / "logs").glob("attention-sst*.log")
    sidecar = log.with_suffix(".resources.jsonl")
    resource_report = json.loads((case / "resource_analysis.json").read_text())
    projection = json.loads((case / "projection_e2e_result.json").read_text())["stages"]["projection"]
    factor = resource_report["ticks_per_cycle"]

    jobs = {}
    phase_events = defaultdict(list)
    with log.open(errors="replace") as stream:
        for line in stream:
            if "[PROJECTION_JOB]" in line:
                row = dict(FIELDS.findall(line))
                jobs[int(row["core"])] = (int(row["start"]), int(row["end"]))
            elif "[CRITICAL_EVENT]" in line:
                row = dict(FIELDS.findall(line))
                phase_events[int(row["core"])].append({
                    "cycle": int(row["cycle"]), "event_id": int(row["event_id"]),
                    "kind": row["kind"], "phase": row["phase"],
                })
    if set(jobs) != set(range(4, 20)):
        raise ValueError("Projection job windows do not cover workers 4..19")

    spans = defaultdict(list)
    queues = defaultdict(list)
    array_macs = defaultdict(float)
    with sidecar.open(errors="replace") as stream:
        for line in stream:
            if "[RESOURCE_INTERVAL]" not in line:
                continue
            row = dict(FIELDS.findall(line))
            core = int(row["core"])
            if core not in jobs:
                continue
            domain, resource = row["domain"], row["resource"]
            lo, hi = int(row["start_tick"]), int(row["done_tick"])
            if hi > lo:
                spans[(core, domain, resource)].append((lo, hi))
            submit = int(row["submit_tick"])
            if lo > submit and domain == "gm":
                queues[(core, domain, resource)].append((submit, lo))
            if domain == "array" and resource == "compute":
                job_lo, job_hi = (value * factor for value in jobs[core])
                array_macs[core] += max(0, min(hi, job_hi) - max(lo, job_lo)) * int(row["client"])

    by_worker = {}
    for core, (job_start, job_end) in jobs.items():
        lower, upper = job_start * factor, job_end * factor
        compute = clipped_union(spans[(core, "array", "compute")], lower, upper)
        if len(compute) < 2:
            raise ValueError(f"worker {core}: fewer than two MAC intervals")
        marks = sorted(phase_events[core], key=lambda event: event["event_id"])
        phase_spans = []
        opened = None
        for event in marks:
            tick = event["cycle"] * factor
            if event["kind"] == "phase_start":
                opened = (event["phase"], tick)
            elif event["kind"] == "phase_done":
                if opened is None or opened[0] != event["phase"]:
                    raise ValueError(f"worker {core}: unmatched phase events")
                phase_spans.append((opened[0], opened[1], tick))
                opened = None
        phase_starts = [item[1] for item in phase_spans]
        phase_ends = [item[2] for item in phase_spans]
        gm = merge(span for (owner, domain, _), values in spans.items()
                   if owner == core and domain == "gm" for span in values)
        gm_queue = merge(queues[(core, "gm", "local_read")])
        local_gm = merge(spans[(core, "gm", "local_read")])
        matrix = merge(spans[(core, "array", "matrix_broadcast")])
        scatter = merge(spans[(core, "array", "input_scatter")])
        gm_starts, gm_ends = [x[0] for x in gm], [x[1] for x in gm]
        queue_starts, queue_ends = [x[0] for x in gm_queue], [x[1] for x in gm_queue]
        matrix_starts, matrix_ends = [x[0] for x in matrix], [x[1] for x in matrix]
        scatter_starts, scatter_ends = [x[0] for x in scatter], [x[1] for x in scatter]

        gaps = []
        exclusive_phase_cycles = Counter()
        overlap_cycles = Counter()
        for (_, previous_end), (next_start, _) in zip(compute, compute[1:]):
            if next_start <= previous_end:
                continue
            gap = (previous_end, next_start)
            cuts = {gap[0], gap[1]}
            phase_candidates = intersecting(phase_spans, phase_starts, phase_ends, *gap)
            for _, a, b in phase_candidates:
                cuts.update((max(a, gap[0]), min(b, gap[1])))
            for intervals, starts, ends in ((gm, gm_starts, gm_ends),
                                             (gm_queue, queue_starts, queue_ends),
                                             (matrix, matrix_starts, matrix_ends),
                                             (scatter, scatter_starts, scatter_ends)):
                for a, b in intersecting(intervals, starts, ends, *gap):
                    cuts.update((max(a, gap[0]), min(b, gap[1])))
            cuts = sorted(cuts)
            for a, b in zip(cuts, cuts[1:]):
                mid = (a + b) // 2
                pi = bisect_right(phase_starts, mid) - 1
                phase = phase_spans[pi][0] if pi >= 0 and mid < phase_spans[pi][2] else "unclassified"
                duration = (b - a) / factor
                exclusive_phase_cycles[phase] += duration
                for name, intervals, starts, ends in (("gm_active", gm, gm_starts, gm_ends),
                    ("matrix_service", matrix, matrix_starts, matrix_ends),
                    ("scatter_service", scatter, scatter_starts, scatter_ends)):
                    idx = bisect_right(starts, mid) - 1
                    if idx >= 0 and mid < ends[idx]:
                        overlap_cycles[name] += duration
                gaps.append((a, b))
        gap_spans = merge(gaps)
        durations = sorted((b - a) / factor for a, b in gap_spans)
        elapsed = (upper - lower) / factor
        mac = array_macs[core] / factor
        compute_union_cycles = sum(b-a for a,b in compute)/factor
        before_first_mac = (compute[0][0]-lower)/factor
        after_last_mac = (upper-compute[-1][1])/factor
        if abs(elapsed-compute_union_cycles-sum(durations)-before_first_mac-after_last_mac) > 1e-9:
            raise ValueError('Projection MAC/window accounting does not close')
        if abs(sum(exclusive_phase_cycles.values())-sum(durations)) > 1e-9 or exclusive_phase_cycles['unclassified']:
            raise ValueError('MAC holes have missing phase coverage')
        largest_gap = max(gap_spans, key=lambda span: span[1] - span[0])
        largest_lo, largest_hi = largest_gap
        largest_mid = (largest_lo + largest_hi) // 2
        largest_phase_idx = bisect_right(phase_starts, largest_mid) - 1
        largest_phase = (phase_spans[largest_phase_idx][0] if largest_phase_idx >= 0 and
                         largest_mid < phase_spans[largest_phase_idx][2] else "unclassified")
        by_worker[str(core)] = {
            "job_start_cycle": job_start,
            "job_end_cycle": job_end,
            "job_elapsed_cycles": elapsed,
            "mac_service_cycles": mac / 64,
            "compute_union_cycles": compute_union_cycles,
            "before_first_mac_cycles": before_first_mac,
            "after_last_mac_cycles": after_last_mac,
            "mac_utilization": mac / (64 * elapsed),
            "mac_interval_count": len(compute),
            "between_mac_gap_count": len(durations),
            "between_mac_gap_cycles": sum(durations),
            "gap_p50_cycles": durations[len(durations) // 2] if durations else 0,
            "gap_p95_cycles": durations[min(len(durations) - 1, int(.95 * len(durations)))] if durations else 0,
            "gap_max_cycles": max(durations, default=0),
            "gap_cycles_by_controller_phase": dict(exclusive_phase_cycles),
            "gap_cycles_with_resource_active": dict(overlap_cycles),
            "largest_gap": {"start_cycle": largest_lo / factor, "end_cycle": largest_hi / factor,
                            "duration_cycles": (largest_hi - largest_lo) / factor,
                            "controller_phase_at_midpoint": largest_phase,
                            "gm_local_read_queue_overlap_cycles": sum(
                                max(0, min(largest_hi, b) - max(largest_lo, a))
                                for a, b in intersecting(gm_queue, queue_starts, queue_ends,
                                                         largest_lo, largest_hi)) / factor,
                            "gm_local_read_service_overlap_cycles": sum(
                                max(0, min(largest_hi, b) - max(largest_lo, a))
                                for a, b in local_gm) / factor},
        }
    slow = max(by_worker, key=lambda core: by_worker[core]["job_end_cycle"])
    fast = min(by_worker, key=lambda core: by_worker[core]["job_end_cycle"])
    return {
        "status": "PASS", "artifact_root": str(case), "ticks_per_cycle": factor,
        "projection_stage_cycles": projection["elapsed_cycles"],
        "critical_worker": int(slow), "fastest_worker": int(fast),
        "worker_finish_span_cycles": max(v["job_end_cycle"] for v in by_worker.values()) -
                                     min(v["job_end_cycle"] for v in by_worker.values()),
        "worker_elapsed_span_cycles": max(v["job_elapsed_cycles"] for v in by_worker.values()) -
                                     min(v["job_elapsed_cycles"] for v in by_worker.values()),
        "by_worker": by_worker,
        "scope": "Each gap is between adjacent physical array.compute intervals on one worker. MAC utilization weights each interval by its array count. Phase totals are exclusive; resource-active totals overlap and must not be added to phase totals. GM queue and service overlaps are calculated from actual request submit/start/done timestamps.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path, nargs="+", help="case directories containing existing SST traces")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [analyze(path) for path in args.cases]
    result = {"status": "PASS", "cases": cases}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for case in cases:
        print(f"{Path(case['artifact_root']).name}: projection={case['projection_stage_cycles']}, "
              f"critical core={case['critical_worker']}, worker finish span={case['worker_finish_span_cycles']}")


if __name__ == "__main__":
    main()
