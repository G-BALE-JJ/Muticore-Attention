#!/usr/bin/env python3
"""Validate and summarize the opt-in end-to-end critical-path trace."""
import argparse
import json
from pathlib import Path
import re


EVENT = re.compile(r"\[CRITICAL_EVENT\] (?P<body>[^\n]+)")
FIELD = re.compile(r"(\w+)=([^\s]+)")
JOB = re.compile(r"\[PROJECTION_JOB\] core=(\d+) start=(\d+) end=(\d+) cycles=(\d+)")
SYNC = re.compile(r"\[PROJECTION_SYNC\] core=(\d+) stage=local_wait cycle=(\d+) status=0")
MILESTONE = re.compile(
    r"\[ATTENTION_MILESTONE\] stage=(\w+) status=(\w+) sst_tick=(\d+) "
    r"rocc_cycle=(\d+) core=(\d+) role=(\w+) job=(\d+) tag=(\d+) "
    r"query_tile=(-?\d+) kv_tile=(-?\d+)"
)


def parse_records(text, pattern):
    records = []
    for match in pattern.finditer(text):
        records.append({k: v for k, v in FIELD.findall(match.group("body"))} if pattern is EVENT else match.groups())
    return records


def analyze(case):
    case = Path(case)
    report = json.loads((case / "projection_e2e_result.json").read_text())
    logs = list((case / "logs").glob("attention-sst*.log"))
    if len(logs) != 1:
        raise ValueError("expected exactly one merged SST log")
    text = logs[0].read_text(errors="replace")
    raw_events = parse_records(text, EVENT)
    if not raw_events:
        raise ValueError("critical trace is absent; set GOLEM_CRITICAL_PATH_TRACE=1")
    events = []
    for raw in raw_events:
        required = {"event_id", "predecessor", "kind", "phase", "resource", "cycle", "core", "job", "generation"}
        if not required <= raw.keys():
            raise ValueError(f"incomplete critical event: {raw}")
        event = {k: int(raw[k]) if k not in ("kind", "phase", "resource") else raw[k] for k in raw}
        events.append(event)

    by_core = {}
    for event in events:
        by_core.setdefault(event["core"], []).append(event)
    workers = list(range(4, 20))
    if sorted(by_core) != workers:
        raise ValueError(f"critical trace does not cover workers: {sorted(by_core)}")
    phase_pairs = 0
    phase_records = []
    for core, stream in by_core.items():
        previous_id = 0
        open_phase = None
        for event in sorted(stream, key=lambda item: item["event_id"]):
            if event["event_id"] <= previous_id or event["predecessor"] != previous_id:
                raise ValueError(f"broken event predecessor chain on core {core}")
            if event["kind"] == "phase_start":
                if open_phase is not None:
                    raise ValueError(f"nested phase on core {core}")
                open_phase = event
            elif event["kind"] == "phase_done":
                if open_phase is None or open_phase["phase"] != event["phase"]:
                    raise ValueError(f"unmatched phase completion on core {core}")
                if event["cycle"] < open_phase["cycle"]:
                    raise ValueError(f"negative phase duration on core {core}")
                phase_pairs += 1
                phase_records.append({
                    "core": core,
                    "phase": open_phase["phase"],
                    "resource": open_phase["resource"],
                    "start_cycle": open_phase["cycle"],
                    "end_cycle": event["cycle"],
                    "duration_cycles": event["cycle"] - open_phase["cycle"],
                    "generation": open_phase["generation"],
                })
                open_phase = None
            else:
                raise ValueError(f"unknown critical event kind: {event['kind']}")
            previous_id = event["event_id"]
        # retire() changes Complete -> Idle after the final completion.  The
        # trace observes that bookkeeping transition on the next tick, so a
        # trailing idle start is a valid terminal marker without a matching
        # done event.
        if open_phase is not None and open_phase["phase"] not in ("complete", "idle"):
            raise ValueError(f"incomplete final phase on core {core}")

    jobs = [tuple(map(int, row)) for row in JOB.findall(text)]
    sync = [tuple(map(int, row)) for row in SYNC.findall(text)]
    if sorted(row[0] for row in jobs) != workers or sorted(row[0] for row in sync) != workers:
        raise ValueError("projection worker timing is incomplete")
    projection = report["stages"]["projection"]
    critical_core, critical_sync = max(sync, key=lambda row: row[1])
    critical_job = next(row for row in jobs if row[0] == critical_core)
    if critical_sync != projection["end_cycle"]:
        raise ValueError("local_wait timing does not match projection end")
    resource_spans = {}
    for record in phase_records:
        summary = resource_spans.setdefault(record["resource"], {
            "count": 0, "total_cycles": 0, "max_cycles": 0,
        })
        summary["count"] += 1
        summary["total_cycles"] += record["duration_cycles"]
        summary["max_cycles"] = max(summary["max_cycles"], record["duration_cycles"])
    worker_starts = {row[0]: row[1] for row in jobs}
    worker_ends = {row[0]: row[2] for row in jobs}
    critical_phase_top = sorted(
        (record for record in phase_records if record["core"] == critical_core),
        key=lambda record: (-record["duration_cycles"], record["start_cycle"]),
    )[:20]

    milestones = []
    for row in MILESTONE.finditer(text):
        stage, status, sst, cycle, core, role, job, tag, query, kv = row.groups()
        milestones.append({"stage": stage, "status": status, "sst_tick": int(sst),
                           "cycle": int(cycle), "core": int(core), "role": role,
                           "job": int(job), "tag": int(tag), "query_tile": int(query), "kv_tile": int(kv)})
    done = [event for event in milestones if event["status"] == "done"]
    if not done:
        raise ValueError("attention milestone trace is absent")
    worker_complete = [event for event in done if event["stage"] == "worker_complete"]
    if not worker_complete:
        raise ValueError("attention worker completion trace is absent")
    attention_end = max(event["cycle"] for event in worker_complete)
    chain_start = report.get("chain_start_cycle")
    if chain_start is None:
        raise ValueError("projection report has no chain_start_cycle")
    reconstructed_e2e = attention_end - chain_start
    if reconstructed_e2e != report["end_to_end_cycles"]:
        raise ValueError(
            f"critical trace does not reconstruct E2E: {reconstructed_e2e} != "
            f"{report['end_to_end_cycles']}"
        )
    first_descriptor = min((event["cycle"] for event in done if "descriptor_accept" in event["stage"]), default=None)
    final_tensor = max((event["cycle"] for event in done if event["stage"] == "root_tensor_complete"), default=None)
    if first_descriptor is None or final_tensor is None or final_tensor <= first_descriptor:
        raise ValueError("attention milestone chain is incomplete")
    stage_spans = {
        stage: {"first_done_cycle": min(event["cycle"] for event in done if event["stage"] == stage),
                "last_done_cycle": max(event["cycle"] for event in done if event["stage"] == stage),
                "count": sum(event["stage"] == stage for event in done)}
        for stage in sorted({event["stage"] for event in done})
    }
    result = {
        "status": "PASS", "artifact_root": str(case),
        "e2e_cycles": report["end_to_end_cycles"],
        "reconstructed_e2e_cycles": reconstructed_e2e,
        "chain_start_cycle": chain_start,
        "attention_end_cycle": attention_end,
        "reported_projection_cycles": projection["elapsed_cycles"],
        "critical_worker": critical_core,
        "critical_worker_job": {"start": critical_job[1], "end": critical_job[2], "cycles": critical_job[3]},
        "critical_worker_local_wait": critical_sync,
        "worker_start_skew": critical_job[1] - projection["start_cycle"],
        "worker_start_span": max(worker_starts.values()) - min(worker_starts.values()),
        "worker_finish_span": max(worker_ends.values()) - min(worker_ends.values()),
        "resource_phase_summary": resource_spans,
        "critical_worker_longest_phases": critical_phase_top,
        "projection_to_attention_handoff": report["stage_handoff_cycles"]["projection_to_attention"],
        "critical_event_count": len(events), "phase_pairs": phase_pairs,
        "attention_first_descriptor_cycle": first_descriptor,
        "attention_final_tensor_cycle": final_tensor,
        "attention_span_cycles": final_tensor - first_descriptor,
        "attention_stage_spans": stage_spans,
        "trace_scope": "phase-level Projection events plus existing Attention milestones; exact E2E remains the verified report boundary",
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequences", type=int, nargs="+", default=[1024, 2048])
    args = parser.parse_args()
    cases = [analyze(args.artifact_root / f"s{sequence}") for sequence in args.sequences]
    args.output.write_text(json.dumps({"status": "PASS", "cases": cases}, indent=2) + "\n")
    for case in cases:
        print(f"{Path(case['artifact_root']).name}: E2E={case['e2e_cycles']}, "
              f"critical core={case['critical_worker']}, events={case['critical_event_count']}, "
              f"Attention span={case['attention_span_cycles']}")


if __name__ == "__main__":
    main()
