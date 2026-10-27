#!/usr/bin/env python3
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


REQ_ISSUE_RE = re.compile(
    r"TRACE_REQ_ISSUE\s+cycle=(?P<cycle>\d+)\s+req=(?P<req>\d+)(?:\s+trace_id=(?P<trace_id>\d+))?"
)
REQ_DONE_RE = re.compile(
    r"TRACE_REQ_DONE\s+cycle=(?P<done>\d+)\s+req=(?P<req>\d+).*?"
    r"issue_cycle=(?P<issue>\d+)\s+pending_clear_cycle=(?P<pending>\d+)"
)
MEMNIC_RECV_RE = re.compile(
    r"\[memNICBase bridge\]\s+recv\s+READ\s+cycle=(?P<cycle>\d+).*?\sreq=(?P<req>\d+)"
)
MEMNIC_RESP_RE = re.compile(
    r"\[memNICBase bridge\]\s+send\s+READ_RESP\s+cycle=(?P<cycle>\d+).*?"
    r"\sreq=(?P<req>\d+)(?:\s+trace_id=(?P<trace_id>\d+))?"
)
MEMNIC_RESP_ACCEPT_RE = re.compile(
    r"TRACE_REQ_RESP_CHUNK_SEND\s+cycle=(?P<cycle>\d+)\s+req=(?P<req>\d+)"
    r"(?:\s+trace_id=(?P<trace_id>\d+))?"
)
WORKER_RECV_RE = re.compile(
    r"TRACE_REQ_WORKER_RECV_CHUNK\s+cycle=(?P<cycle>\d+)\s+req=(?P<req>\d+).*"
)
GM_CLEAR_RE = re.compile(
    r"TRACE_REQ_PENDING_CLEAR\s+cycle=(?P<cycle>\d+)\s+req=(?P<req>\d+)"
)
MERLIN_RE = re.compile(
    r"TRACE\((?P<trace_id>\d+)\):\s+(?P<ns>\d+)\s+ns:\s+(?P<msg>.*)$"
)


def decode_slot(req: int) -> int:
    return (req >> 48) & 0xFF


def trace_id_str(trace_id):
    return "" if trace_id is None else str(trace_id)


def add_event(events, req, trace_id, name, raw_time, time_cycles, detail):
    events.append(
        {
            "req": str(req),
            "slot": str(decode_slot(req)) if req else "",
            "trace_id": trace_id_str(trace_id),
            "event": name,
            "raw_time": str(raw_time),
            "time_cycles": f"{time_cycles:.6f}",
            "detail": detail,
        }
    )


def nearest_scale(ratios):
    ratios = [r for r in ratios if r > 0.0]
    if not ratios:
        return 1.0
    ratios.sort()
    med = ratios[len(ratios) // 2]
    if 0.25 <= med <= 4.0:
        return 1.0
    candidates = [10.0, 100.0, 1000.0, 10000.0, 1000000.0]
    best = min(candidates, key=lambda c: abs(med - c))
    if abs(med - best) / best <= 0.30:
        return best
    return med


def compact_msg(msg: str) -> str:
    msg = " ".join(msg.strip().split())
    return msg[:180]


def main():
    parser = argparse.ArgumentParser(description="Extract request-id and Merlin FULL trace timelines.")
    parser.add_argument("--log", required=True)
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--segments", required=True)
    args = parser.parse_args()

    reqs = defaultdict(dict)
    trace_to_req = {}
    merlin = defaultdict(list)

    for line in Path(args.log).read_text(errors="ignore").splitlines():
        if m := REQ_ISSUE_RE.search(line):
            req = int(m.group("req"))
            reqs[req]["issue"] = int(m.group("cycle"))
            if m.group("trace_id"):
                tid = int(m.group("trace_id"))
                reqs[req]["trace_id"] = tid
                trace_to_req[tid] = req
            continue
        if m := REQ_DONE_RE.search(line):
            req = int(m.group("req"))
            reqs[req]["issue"] = int(m.group("issue"))
            reqs[req]["pending_sched"] = int(m.group("pending"))
            reqs[req]["done"] = int(m.group("done"))
            continue
        if m := MEMNIC_RECV_RE.search(line):
            reqs[int(m.group("req"))]["memnic_recv"] = int(m.group("cycle"))
            continue
        if m := MEMNIC_RESP_RE.search(line):
            req = int(m.group("req"))
            reqs[req]["memnic_resp"] = int(m.group("cycle"))
            if m.group("trace_id"):
                tid = int(m.group("trace_id"))
                reqs[req]["trace_id"] = tid
                trace_to_req[tid] = req
            continue
        if m := MEMNIC_RESP_ACCEPT_RE.search(line):
            req = int(m.group("req"))
            reqs[req]["memnic_resp_accept"] = int(m.group("cycle"))
            if m.group("trace_id"):
                tid = int(m.group("trace_id"))
                reqs[req]["trace_id"] = tid
                trace_to_req[tid] = req
            continue
        if m := WORKER_RECV_RE.search(line):
            reqs[int(m.group("req"))]["worker_recv"] = int(m.group("cycle"))
            continue
        if m := GM_CLEAR_RE.search(line):
            reqs[int(m.group("req"))]["gm_clear"] = int(m.group("cycle"))
            continue
        if m := MERLIN_RE.search(line):
            tid = int(m.group("trace_id"))
            merlin[tid].append((float(m.group("ns")), compact_msg(m.group("msg"))))

    ratios = []
    for data in reqs.values():
        issue = data.get("issue")
        if not issue:
            continue
        for key in ("memnic_recv", "memnic_resp", "memnic_resp_accept", "worker_recv", "gm_clear"):
            raw = data.get(key)
            if raw:
                ratios.append(float(raw) / float(issue))
    scale = nearest_scale(ratios)

    timeline_rows = []
    segment_rows = []
    fixed_events = [
        ("issue", "scheduler_issue"),
        ("memnic_recv", "memnic_recv_read"),
        ("memnic_resp", "memnic_read_resp_ready"),
        ("memnic_resp_accept", "memnic_resp_link_accept"),
        ("worker_recv", "worker_recv_chunk"),
        ("gm_clear", "gm_pending_clear"),
        ("pending_sched", "scheduler_observe_pending_clear"),
        ("done", "scheduler_done_ack"),
    ]

    for req, data in sorted(reqs.items()):
        tid = data.get("trace_id")
        for key, name in fixed_events:
            raw = data.get(key)
            if raw is None:
                continue
            norm = float(raw) if key in ("issue", "pending_sched", "done") else float(raw) / scale
            add_event(timeline_rows, req, tid, name, raw, norm, "")

        for ns, msg in sorted(merlin.get(tid, [])):
            add_event(timeline_rows, req, tid, "merlin_trace", int(ns), ns, msg)

        events = []
        for key, name in fixed_events:
            raw = data.get(key)
            if raw is None:
                continue
            norm = float(raw) if key in ("issue", "pending_sched", "done") else float(raw) / scale
            events.append((norm, name, ""))
        for ns, msg in merlin.get(tid, []):
            events.append((ns, "merlin_trace", msg))
        events.sort(key=lambda x: (x[0], x[1], x[2]))

        for (t0, e0, d0), (t1, e1, d1) in zip(events, events[1:]):
            segment_rows.append(
                {
                    "req": str(req),
                    "slot": str(decode_slot(req)),
                    "trace_id": trace_id_str(tid),
                    "from_event": e0,
                    "to_event": e1,
                    "from_detail": d0,
                    "to_detail": d1,
                    "from_time_cycles": f"{t0:.6f}",
                    "to_time_cycles": f"{t1:.6f}",
                    "delta_cycles": f"{max(t1 - t0, 0.0):.6f}",
                    "cycle_scale": f"{scale:.6f}",
                }
            )

    Path(args.timeline).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.timeline).open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["req", "slot", "trace_id", "event", "raw_time", "time_cycles", "detail"],
        )
        writer.writeheader()
        writer.writerows(sorted(timeline_rows, key=lambda r: (int(r["req"]), float(r["time_cycles"]), r["event"])))

    with Path(args.segments).open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "req",
                "slot",
                "trace_id",
                "from_event",
                "to_event",
                "from_detail",
                "to_detail",
                "from_time_cycles",
                "to_time_cycles",
                "delta_cycles",
                "cycle_scale",
            ],
        )
        writer.writeheader()
        writer.writerows(segment_rows)


if __name__ == "__main__":
    main()
