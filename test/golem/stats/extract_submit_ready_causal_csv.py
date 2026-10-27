#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


TRACE_LINE_RE = re.compile(r"TRACE_SUBMIT_READY_BREAKDOWN\(cycles\):\s+(?P<body>.+)$")
TRACE_REQ_ISSUE_RE = re.compile(
    r"TRACE_REQ_ISSUE\s+cycle=(?P<cycle>\d+)\s+req=(?P<req>\d+)"
)
TRACE_REQ_DONE_RE = re.compile(
    r"TRACE_REQ_DONE\s+cycle=(?P<done_cycle>\d+)\s+req=(?P<req>\d+)\s+slot=(?P<slot>\d+)"
    r"\s+issue_cycle=(?P<issue_cycle>\d+)\s+pending_clear_cycle=(?P<pending_clear_cycle>\d+)"
)
MEMNIC_RECV_RE = re.compile(
    r"\[memNICBase bridge\]\s+recv\s+READ(?:\s+cycle=(?P<cycle>\d+))?.*?\sreq=(?P<req>\d+)"
)
MEMNIC_SEND_RE = re.compile(
    r"\[memNICBase bridge\]\s+send\s+READ_RESP(?:\s+cycle=(?P<cycle>\d+))?.*?\sreq=(?P<req>\d+)"
)
MEMNIC_RESP_CHUNK_SEND_RE = re.compile(
    r"TRACE_REQ_RESP_CHUNK_SEND\s+cycle=(?P<cycle>\d+)\s+req=(?P<req>\d+)"
)
WORKER_RECV_CHUNK_RE = re.compile(
    r"TRACE_REQ_WORKER_RECV_CHUNK\s+cycle=(?P<cycle>\d+)\s+req=(?P<req>\d+)"
)
GM_PENDING_CLEAR_RE = re.compile(
    r"TRACE_REQ_PENDING_CLEAR\s+cycle=(?P<cycle>\d+)\s+req=(?P<req>\d+)"
)


def build_metric_re(metric_name: str):
    return re.compile(
        rf"{re.escape(metric_name)}\(n=(\d+)\s+mean=([0-9.]+)\s+p50=(\d+)\s+p95=(\d+)\s+min=(\d+)\s+max=(\d+)\)"
    )


METRIC_NAMES = [
    "submit_to_issue_mat",
    "submit_to_issue_vec",
    "issue_to_pending_clear_mat",
    "issue_to_pending_clear_vec",
]
METRIC_RES = {name: build_metric_re(name) for name in METRIC_NAMES}


def decode_request_slot(request_id: int) -> int:
    return (request_id >> 48) & 0xFF


def percentile(values, q: float):
    if not values:
        return 0.0
    vals = sorted(values)
    idx = int(round(q * (len(vals) - 1)))
    return float(vals[idx])


def stats_from_values(values):
    if not values:
        return {
            "n": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    arr = [float(v) for v in values]
    return {
        "n": float(len(arr)),
        "mean": sum(arr) / float(len(arr)),
        "p50": percentile(arr, 0.50),
        "p95": percentile(arr, 0.95),
        "min": min(arr),
        "max": max(arr),
    }


def parse_clock_ghz(text: str, default_ghz: float):
    if not text:
        return default_ghz
    raw = text.strip().lower()
    try:
        return float(raw)
    except ValueError:
        pass

    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*(ghz|mhz|khz|hz)$", raw)
    if not m:
        return default_ghz
    val = float(m.group(1))
    unit = m.group(2)
    if unit == "ghz":
        return val
    if unit == "mhz":
        return val / 1_000.0
    if unit == "khz":
        return val / 1_000_000.0
    return val / 1_000_000_000.0


def read_metric_value_csv(path: Path):
    if not path.exists():
        return {}
    out = {}
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    if not rows or rows[0] != ["metric", "value"]:
        return out
    for row in rows[1:]:
        if len(row) < 2:
            continue
        out[row[0]] = row[1]
    return out


def parse_float(d, key: str, default: float = 0.0):
    try:
        return float(d.get(key, default))
    except (TypeError, ValueError):
        return default


def resolve_inputs(log_path: Path | None, log_dir: Path, stdout_glob: str):
    inputs = []
    if log_dir.exists() and log_dir.is_dir():
        inputs.extend(sorted(p for p in log_dir.glob(stdout_glob) if p.is_file()))
    if log_path is not None and log_path.exists() and log_path.is_file():
        inputs.append(log_path)
    return inputs


def parse_trace_samples(log_paths):
    samples = {name: [] for name in METRIC_NAMES}
    for log_path in log_paths:
        for line in log_path.read_text(errors="ignore").splitlines():
            lm = TRACE_LINE_RE.search(line)
            if not lm:
                continue
            body = lm.group("body")
            for name, metric_re in METRIC_RES.items():
                m = metric_re.search(body)
                if not m:
                    continue
                samples[name].append(
                    {
                        "n": int(m.group(1)),
                        "mean": float(m.group(2)),
                        "p50": float(m.group(3)),
                        "p95": float(m.group(4)),
                        "min": float(m.group(5)),
                        "max": float(m.group(6)),
                    }
                )
    return samples


def aggregate_samples(samples):
    if not samples:
        return {
            "n": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    total_n = sum(s["n"] for s in samples)
    if total_n <= 0:
        total_n = float(len(samples))

    def wavg(key: str):
        return sum(s[key] * s["n"] for s in samples) / total_n

    return {
        "n": float(total_n),
        "mean": wavg("mean"),
        "p50": wavg("p50"),
        "p95": wavg("p95"),
        "min": min(s["min"] for s in samples),
        "max": max(s["max"] for s in samples),
    }


def parse_event_maps(log_paths):
    issue_cycle = {}
    pending_clear_cycle = {}
    memnic_recv_cycle = {}
    memnic_send_cycle = {}
    memnic_chunk_first_send_cycle = {}
    memnic_chunk_last_send_cycle = {}
    worker_first_recv_cycle = {}
    worker_last_recv_cycle = {}
    gm_pending_clear_cycle = {}
    done_cycle = {}

    for log_path in log_paths:
        for line in log_path.read_text(errors="ignore").splitlines():
            m = TRACE_REQ_ISSUE_RE.search(line)
            if m:
                req = int(m.group("req"))
                cyc = int(m.group("cycle"))
                prev = issue_cycle.get(req)
                issue_cycle[req] = cyc if prev is None else min(prev, cyc)
                continue

            m = TRACE_REQ_DONE_RE.search(line)
            if m:
                req = int(m.group("req"))
                issue = int(m.group("issue_cycle"))
                pending = int(m.group("pending_clear_cycle"))
                done = int(m.group("done_cycle"))
                if issue > 0:
                    prev_issue = issue_cycle.get(req)
                    issue_cycle[req] = issue if prev_issue is None else min(prev_issue, issue)
                if pending > 0:
                    prev_pending = pending_clear_cycle.get(req)
                    pending_clear_cycle[req] = (
                        pending if prev_pending is None else max(prev_pending, pending)
                    )
                prev_done = done_cycle.get(req)
                done_cycle[req] = done if prev_done is None else max(prev_done, done)
                continue

            m = MEMNIC_RECV_RE.search(line)
            if m:
                if m.group("cycle") is None:
                    continue
                req = int(m.group("req"))
                cyc = int(m.group("cycle"))
                prev = memnic_recv_cycle.get(req)
                memnic_recv_cycle[req] = cyc if prev is None else min(prev, cyc)
                continue

            m = MEMNIC_SEND_RE.search(line)
            if m:
                if m.group("cycle") is None:
                    continue
                req = int(m.group("req"))
                cyc = int(m.group("cycle"))
                prev = memnic_send_cycle.get(req)
                memnic_send_cycle[req] = cyc if prev is None else min(prev, cyc)
                continue

            m = MEMNIC_RESP_CHUNK_SEND_RE.search(line)
            if m:
                req = int(m.group("req"))
                cyc = int(m.group("cycle"))
                prev_first = memnic_chunk_first_send_cycle.get(req)
                prev_last = memnic_chunk_last_send_cycle.get(req)
                memnic_chunk_first_send_cycle[req] = cyc if prev_first is None else min(prev_first, cyc)
                memnic_chunk_last_send_cycle[req] = cyc if prev_last is None else max(prev_last, cyc)
                continue

            m = WORKER_RECV_CHUNK_RE.search(line)
            if m:
                req = int(m.group("req"))
                cyc = int(m.group("cycle"))
                prev_first = worker_first_recv_cycle.get(req)
                prev_last = worker_last_recv_cycle.get(req)
                worker_first_recv_cycle[req] = cyc if prev_first is None else min(prev_first, cyc)
                worker_last_recv_cycle[req] = cyc if prev_last is None else max(prev_last, cyc)
                continue

            m = GM_PENDING_CLEAR_RE.search(line)
            if m:
                req = int(m.group("req"))
                cyc = int(m.group("cycle"))
                prev = gm_pending_clear_cycle.get(req)
                gm_pending_clear_cycle[req] = cyc if prev is None else max(prev, cyc)
                continue

    return {
        "issue": issue_cycle,
        "pending": pending_clear_cycle,
        "done": done_cycle,
        "memnic_recv": memnic_recv_cycle,
        "memnic_send": memnic_send_cycle,
        "memnic_chunk_first_send": memnic_chunk_first_send_cycle,
        "memnic_chunk_last_send": memnic_chunk_last_send_cycle,
        "worker_first_recv": worker_first_recv_cycle,
        "worker_last_recv": worker_last_recv_cycle,
        "gm_pending_clear": gm_pending_clear_cycle,
    }


def build_event_causal_stats(event_maps, sched_ghz: float, mem_ghz: float):
    mem_to_sched = sched_ghz / mem_ghz if mem_ghz > 0.0 else 1.0

    mat = {
        "issue_to_pending": [],
        "forward": [],
        "memory": [],
        "return": [],
        "return_ready_to_first_send": [],
        "return_first_send_to_first_recv": [],
        "return_chunk_span": [],
        "return_last_recv_to_gm_clear": [],
        "return_gm_clear_to_sched_observe": [],
    }
    vec = {
        "issue_to_pending": [],
        "forward": [],
        "memory": [],
        "return": [],
        "return_ready_to_first_send": [],
        "return_first_send_to_first_recv": [],
        "return_chunk_span": [],
        "return_last_recv_to_gm_clear": [],
        "return_gm_clear_to_sched_observe": [],
    }

    raw_req_count = 0
    full_match_count = 0
    invalid_order_count = 0

    issue = event_maps["issue"]
    pending = event_maps["pending"]
    recv = event_maps["memnic_recv"]
    send = event_maps["memnic_send"]
    chunk_first_send = event_maps["memnic_chunk_first_send"]
    worker_first_recv = event_maps["worker_first_recv"]
    worker_last_recv = event_maps["worker_last_recv"]
    gm_pending_clear = event_maps["gm_pending_clear"]

    all_reqs = set(issue.keys()) | set(pending.keys()) | set(recv.keys()) | set(send.keys())
    for req in all_reqs:
        raw_req_count += 1
        if req not in issue or req not in pending or req not in recv or req not in send:
            continue

        issue_cycle = float(issue[req])
        pending_cycle = float(pending[req])
        recv_cycle = float(recv[req]) * mem_to_sched
        send_cycle = float(send[req]) * mem_to_sched

        total = pending_cycle - issue_cycle
        forward = recv_cycle - issue_cycle
        memory = send_cycle - recv_cycle
        back = pending_cycle - send_cycle

        # small negative tolerance for clock conversion/rounding
        if total < -2.0 or forward < -2.0 or memory < -2.0 or back < -2.0:
            invalid_order_count += 1
            continue

        total = max(total, 0.0)
        forward = max(forward, 0.0)
        memory = max(memory, 0.0)
        back = max(back, 0.0)

        slot = decode_request_slot(req)
        target = mat if slot == 0 else (vec if slot == 1 else None)
        if target is None:
            continue

        target["issue_to_pending"].append(total)
        target["forward"].append(forward)
        target["memory"].append(memory)
        target["return"].append(back)
        if req in chunk_first_send and req in worker_first_recv and req in worker_last_recv and req in gm_pending_clear:
            first_send_cycle = float(chunk_first_send[req]) * mem_to_sched
            first_recv_cycle = float(worker_first_recv[req]) * mem_to_sched
            last_recv_cycle = float(worker_last_recv[req]) * mem_to_sched
            gm_clear_cycle = float(gm_pending_clear[req]) * mem_to_sched
            ready_to_first_send = first_send_cycle - send_cycle
            first_send_to_first_recv = first_recv_cycle - first_send_cycle
            chunk_span = last_recv_cycle - first_recv_cycle
            last_recv_to_gm_clear = gm_clear_cycle - last_recv_cycle
            gm_clear_to_sched_observe = pending_cycle - gm_clear_cycle
            if min(ready_to_first_send, first_send_to_first_recv, chunk_span, last_recv_to_gm_clear, gm_clear_to_sched_observe) >= -2.0:
                target["return_ready_to_first_send"].append(max(ready_to_first_send, 0.0))
                target["return_first_send_to_first_recv"].append(max(first_send_to_first_recv, 0.0))
                target["return_chunk_span"].append(max(chunk_span, 0.0))
                target["return_last_recv_to_gm_clear"].append(max(last_recv_to_gm_clear, 0.0))
                target["return_gm_clear_to_sched_observe"].append(max(gm_clear_to_sched_observe, 0.0))
        full_match_count += 1

    return {
        "mat": {k: stats_from_values(v) for k, v in mat.items()},
        "vec": {k: stats_from_values(v) for k, v in vec.items()},
        "raw_req_count": raw_req_count,
        "full_match_count": full_match_count,
        "invalid_order_count": invalid_order_count,
    }


def pct(v, total):
    if total <= 0.0:
        return 0.0
    return 100.0 * v / total


def combine_mean(a_stats, b_stats):
    an = a_stats["n"]
    bn = b_stats["n"]
    total = an + bn
    if total <= 0.0:
        return 0.0
    return (a_stats["mean"] * an + b_stats["mean"] * bn) / total


def infer_memnic_cycle_scale(issue_cycle_map, recv_cycle_map):
    reqs = set(issue_cycle_map.keys()) & set(recv_cycle_map.keys())
    if not reqs:
        return 1.0

    ratios = []
    for req in reqs:
        issue = float(issue_cycle_map[req])
        recv = float(recv_cycle_map[req])
        if issue <= 0.0 or recv <= 0.0:
            continue
        ratios.append(recv / issue)

    ratios.sort()
    med = ratios[len(ratios) // 2]
    if 0.25 <= med <= 4.0:
        return 1.0

    candidates = [1.0, 10.0, 100.0, 1000.0, 10000.0, 1000000.0]
    best = min(candidates, key=lambda c: abs(med - c))
    rel_err = abs(med - best) / max(best, 1.0)
    if rel_err <= 0.25:
        return best
    return med


def main():
    p = argparse.ArgumentParser(
        description=(
            "Build submit->ready causal split CSV. Prefer request-level event traces "
            "(RequestScheduler TRACE_REQ_* + memNICBase bridge cycle trace); fallback to residual model."
        )
    )
    p.add_argument("--log", default="", help="Main simulation log path")
    p.add_argument("--log-dir", default=".", help="Directory for sharded logs")
    p.add_argument("--stdout-glob", default="stdout-*", help="Glob for sharded logs")
    p.add_argument(
        "--noc-latency-summary",
        required=True,
        help="Path to noc_latency_summary.csv (metric,value)",
    )
    p.add_argument(
        "--memory-queue-summary",
        required=True,
        help="Path to memory_queue_summary.csv (metric,value)",
    )
    p.add_argument(
        "--sched-clock",
        default="1GHz",
        help="Scheduler clock frequency (e.g., 1GHz)",
    )
    p.add_argument(
        "--memory-clock",
        default="1GHz",
        help="Memory/backend/memNIC clock frequency used for cycle-domain conversion",
    )
    p.add_argument("--summary", required=True, help="Output summary CSV (metric,value)")
    p.add_argument("--table", required=True, help="Output causal table CSV")
    args = p.parse_args()

    log_path = Path(args.log) if args.log else None
    log_dir = Path(args.log_dir)
    input_logs = resolve_inputs(log_path, log_dir, args.stdout_glob)

    noc = read_metric_value_csv(Path(args.noc_latency_summary))
    memq = read_metric_value_csv(Path(args.memory_queue_summary))

    sched_ghz = parse_clock_ghz(args.sched_clock, 1.0)
    mem_ghz = parse_clock_ghz(args.memory_clock, 1.0)
    if mem_ghz <= 0.0:
        mem_ghz = 1.0

    # fallback inputs (legacy residual model)
    fallback_samples = parse_trace_samples(input_logs)
    submit_to_issue_mat = aggregate_samples(fallback_samples["submit_to_issue_mat"])
    submit_to_issue_vec = aggregate_samples(fallback_samples["submit_to_issue_vec"])
    fb_issue_mat = aggregate_samples(fallback_samples["issue_to_pending_clear_mat"])
    fb_issue_vec = aggregate_samples(fallback_samples["issue_to_pending_clear_vec"])

    noc_avg_ns = parse_float(noc, "noc_avg_packet_latency_ns", 0.0)
    noc_p95_ns = parse_float(noc, "noc_p95_packet_latency_ns", noc_avg_ns)
    mem_avg_cycles = parse_float(memq, "memory_backend_read_latency_avg_cycles", 0.0)
    mem_p95_cycles = parse_float(memq, "memory_backend_read_latency_p95_cycles", mem_avg_cycles)

    fb_forward_mean = noc_avg_ns * sched_ghz
    fb_forward_p95 = noc_p95_ns * sched_ghz
    fb_mem_mean = mem_avg_cycles * (sched_ghz / mem_ghz)
    fb_mem_p95 = mem_p95_cycles * (sched_ghz / mem_ghz)

    fb_ret_mat_mean = max(fb_issue_mat["mean"] - fb_forward_mean - fb_mem_mean, 0.0)
    fb_ret_vec_mean = max(fb_issue_vec["mean"] - fb_forward_mean - fb_mem_mean, 0.0)
    fb_ret_mat_p95 = max(fb_issue_mat["p95"] - fb_forward_p95 - fb_mem_p95, 0.0)
    fb_ret_vec_p95 = max(fb_issue_vec["p95"] - fb_forward_p95 - fb_mem_p95, 0.0)

    # event-level model (new)
    event_maps = parse_event_maps(input_logs)
    memnic_cycle_scale = infer_memnic_cycle_scale(
        event_maps["issue"],
        event_maps["memnic_recv"],
    )
    event_stats = build_event_causal_stats(
        event_maps,
        sched_ghz,
        mem_ghz * memnic_cycle_scale,
    )

    event_ready = (
        event_stats["mat"]["issue_to_pending"]["n"] > 0
        and event_stats["vec"]["issue_to_pending"]["n"] > 0
        and event_stats["mat"]["forward"]["n"] > 0
        and event_stats["vec"]["forward"]["n"] > 0
        and event_stats["mat"]["memory"]["n"] > 0
        and event_stats["vec"]["memory"]["n"] > 0
        and event_stats["mat"]["return"]["n"] > 0
        and event_stats["vec"]["return"]["n"] > 0
    )

    if event_ready:
        issue_mat = event_stats["mat"]["issue_to_pending"]
        issue_vec = event_stats["vec"]["issue_to_pending"]
        forward_mat = event_stats["mat"]["forward"]
        forward_vec = event_stats["vec"]["forward"]
        memory_mat = event_stats["mat"]["memory"]
        memory_vec = event_stats["vec"]["memory"]
        ret_mat = event_stats["mat"]["return"]
        ret_vec = event_stats["vec"]["return"]
        ret_ready_to_first_send_mat = event_stats["mat"]["return_ready_to_first_send"]
        ret_ready_to_first_send_vec = event_stats["vec"]["return_ready_to_first_send"]
        ret_first_send_to_first_recv_mat = event_stats["mat"]["return_first_send_to_first_recv"]
        ret_first_send_to_first_recv_vec = event_stats["vec"]["return_first_send_to_first_recv"]
        ret_chunk_span_mat = event_stats["mat"]["return_chunk_span"]
        ret_chunk_span_vec = event_stats["vec"]["return_chunk_span"]
        ret_last_recv_to_gm_clear_mat = event_stats["mat"]["return_last_recv_to_gm_clear"]
        ret_last_recv_to_gm_clear_vec = event_stats["vec"]["return_last_recv_to_gm_clear"]
        ret_gm_clear_to_sched_observe_mat = event_stats["mat"]["return_gm_clear_to_sched_observe"]
        ret_gm_clear_to_sched_observe_vec = event_stats["vec"]["return_gm_clear_to_sched_observe"]

        forward_mean = combine_mean(forward_mat, forward_vec)
        memory_mean = combine_mean(memory_mat, memory_vec)
        forward_p95 = max(forward_mat["p95"], forward_vec["p95"])
        memory_p95 = max(memory_mat["p95"], memory_vec["p95"])
        model_source = "event"
        trace_found = "1"
    else:
        issue_mat = fb_issue_mat
        issue_vec = fb_issue_vec
        forward_mat = {
            "mean": fb_forward_mean,
            "p95": fb_forward_p95,
        }
        forward_vec = {
            "mean": fb_forward_mean,
            "p95": fb_forward_p95,
        }
        memory_mat = {
            "mean": fb_mem_mean,
            "p95": fb_mem_p95,
        }
        memory_vec = {
            "mean": fb_mem_mean,
            "p95": fb_mem_p95,
        }
        ret_mat = {
            "mean": fb_ret_mat_mean,
            "p95": fb_ret_mat_p95,
        }
        ret_vec = {
            "mean": fb_ret_vec_mean,
            "p95": fb_ret_vec_p95,
        }
        zero_stats = stats_from_values([])
        ret_ready_to_first_send_mat = zero_stats
        ret_ready_to_first_send_vec = zero_stats
        ret_first_send_to_first_recv_mat = zero_stats
        ret_first_send_to_first_recv_vec = zero_stats
        ret_chunk_span_mat = zero_stats
        ret_chunk_span_vec = zero_stats
        ret_last_recv_to_gm_clear_mat = zero_stats
        ret_last_recv_to_gm_clear_vec = zero_stats
        ret_gm_clear_to_sched_observe_mat = zero_stats
        ret_gm_clear_to_sched_observe_vec = zero_stats
        forward_mean = fb_forward_mean
        memory_mean = fb_mem_mean
        forward_p95 = fb_forward_p95
        memory_p95 = fb_mem_p95
        model_source = "residual"
        trace_found = "1" if (fb_issue_mat["n"] > 0 and fb_issue_vec["n"] > 0) else "0"

    summary_rows = [
        ("causal_model_source", model_source),
        ("trace_found", trace_found),
        ("trace_samples_mat", f"{issue_mat['n']:.0f}"),
        ("trace_samples_vec", f"{issue_vec['n']:.0f}"),
        ("sched_clock_ghz", f"{sched_ghz:.6f}"),
        ("memory_clock_ghz", f"{mem_ghz:.6f}"),
        ("memnic_cycle_scale", f"{memnic_cycle_scale:.6f}"),
        ("event_raw_req_count", str(event_stats["raw_req_count"])),
        ("event_full_match_count", str(event_stats["full_match_count"])),
        ("event_invalid_order_count", str(event_stats["invalid_order_count"])),
        ("causal_submit_to_issue_mat_mean_cycles", f"{submit_to_issue_mat['mean']:.6f}"),
        ("causal_submit_to_issue_vec_mean_cycles", f"{submit_to_issue_vec['mean']:.6f}"),
        ("causal_submit_to_issue_mat_p95_cycles", f"{submit_to_issue_mat['p95']:.6f}"),
        ("causal_submit_to_issue_vec_p95_cycles", f"{submit_to_issue_vec['p95']:.6f}"),
        ("causal_issue_to_pending_mat_mean_cycles", f"{issue_mat['mean']:.6f}"),
        ("causal_issue_to_pending_vec_mean_cycles", f"{issue_vec['mean']:.6f}"),
        ("causal_issue_to_pending_mat_p95_cycles", f"{issue_mat['p95']:.6f}"),
        ("causal_issue_to_pending_vec_p95_cycles", f"{issue_vec['p95']:.6f}"),
        ("causal_forward_to_memnic_mean_cycles", f"{forward_mean:.6f}"),
        ("causal_forward_to_memnic_p95_cycles", f"{forward_p95:.6f}"),
        ("causal_memory_service_mean_cycles", f"{memory_mean:.6f}"),
        ("causal_memory_service_p95_cycles", f"{memory_p95:.6f}"),
        ("causal_return_path_mat_mean_cycles", f"{ret_mat['mean']:.6f}"),
        ("causal_return_path_vec_mean_cycles", f"{ret_vec['mean']:.6f}"),
        ("causal_return_path_mat_p95_cycles", f"{ret_mat['p95']:.6f}"),
        ("causal_return_path_vec_p95_cycles", f"{ret_vec['p95']:.6f}"),
        (
            "causal_return_path_mat_mean_share_pct",
            f"{pct(ret_mat['mean'], issue_mat['mean']):.6f}",
        ),
        (
            "causal_return_path_vec_mean_share_pct",
            f"{pct(ret_vec['mean'], issue_vec['mean']):.6f}",
        ),
        ("causal_return_ready_to_first_send_mat_mean_cycles", f"{ret_ready_to_first_send_mat['mean']:.6f}"),
        ("causal_return_ready_to_first_send_vec_mean_cycles", f"{ret_ready_to_first_send_vec['mean']:.6f}"),
        ("causal_return_first_send_to_first_recv_mat_mean_cycles", f"{ret_first_send_to_first_recv_mat['mean']:.6f}"),
        ("causal_return_first_send_to_first_recv_vec_mean_cycles", f"{ret_first_send_to_first_recv_vec['mean']:.6f}"),
        ("causal_return_chunk_span_mat_mean_cycles", f"{ret_chunk_span_mat['mean']:.6f}"),
        ("causal_return_chunk_span_vec_mean_cycles", f"{ret_chunk_span_vec['mean']:.6f}"),
        ("causal_return_last_recv_to_gm_clear_mat_mean_cycles", f"{ret_last_recv_to_gm_clear_mat['mean']:.6f}"),
        ("causal_return_last_recv_to_gm_clear_vec_mean_cycles", f"{ret_last_recv_to_gm_clear_vec['mean']:.6f}"),
        ("causal_return_gm_clear_to_sched_observe_mat_mean_cycles", f"{ret_gm_clear_to_sched_observe_mat['mean']:.6f}"),
        ("causal_return_gm_clear_to_sched_observe_vec_mean_cycles", f"{ret_gm_clear_to_sched_observe_vec['mean']:.6f}"),
        ("causal_return_ready_to_first_send_mat_p95_cycles", f"{ret_ready_to_first_send_mat['p95']:.6f}"),
        ("causal_return_ready_to_first_send_vec_p95_cycles", f"{ret_ready_to_first_send_vec['p95']:.6f}"),
        ("causal_return_first_send_to_first_recv_mat_p95_cycles", f"{ret_first_send_to_first_recv_mat['p95']:.6f}"),
        ("causal_return_first_send_to_first_recv_vec_p95_cycles", f"{ret_first_send_to_first_recv_vec['p95']:.6f}"),
        ("causal_return_chunk_span_mat_p95_cycles", f"{ret_chunk_span_mat['p95']:.6f}"),
        ("causal_return_chunk_span_vec_p95_cycles", f"{ret_chunk_span_vec['p95']:.6f}"),
        ("causal_return_last_recv_to_gm_clear_mat_p95_cycles", f"{ret_last_recv_to_gm_clear_mat['p95']:.6f}"),
        ("causal_return_last_recv_to_gm_clear_vec_p95_cycles", f"{ret_last_recv_to_gm_clear_vec['p95']:.6f}"),
        ("causal_return_gm_clear_to_sched_observe_mat_p95_cycles", f"{ret_gm_clear_to_sched_observe_mat['p95']:.6f}"),
        ("causal_return_gm_clear_to_sched_observe_vec_p95_cycles", f"{ret_gm_clear_to_sched_observe_vec['p95']:.6f}"),
    ]

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for row in summary_rows:
            w.writerow(row)

    table_rows = [
        {
            "segment": "submit_to_issue_credit_wait",
            "mat_mean_cycles": f"{submit_to_issue_mat['mean']:.6f}",
            "vec_mean_cycles": f"{submit_to_issue_vec['mean']:.6f}",
            "mat_p95_cycles": f"{submit_to_issue_mat['p95']:.6f}",
            "vec_p95_cycles": f"{submit_to_issue_vec['p95']:.6f}",
            "source": "manager_trace",
        },
        {
            "segment": "issue_to_pending_clear_total",
            "mat_mean_cycles": f"{issue_mat['mean']:.6f}",
            "vec_mean_cycles": f"{issue_vec['mean']:.6f}",
            "mat_p95_cycles": f"{issue_mat['p95']:.6f}",
            "vec_p95_cycles": f"{issue_vec['p95']:.6f}",
            "source": "request_scheduler_event" if event_ready else "manager_trace",
        },
        {
            "segment": "forward_to_memnic",
            "mat_mean_cycles": f"{forward_mat['mean']:.6f}",
            "vec_mean_cycles": f"{forward_vec['mean']:.6f}",
            "mat_p95_cycles": f"{forward_mat['p95']:.6f}",
            "vec_p95_cycles": f"{forward_vec['p95']:.6f}",
            "source": "memnic_recv_event" if event_ready else "noc_latency_summary",
        },
        {
            "segment": "memory_service",
            "mat_mean_cycles": f"{memory_mat['mean']:.6f}",
            "vec_mean_cycles": f"{memory_vec['mean']:.6f}",
            "mat_p95_cycles": f"{memory_mat['p95']:.6f}",
            "vec_p95_cycles": f"{memory_vec['p95']:.6f}",
            "source": "memnic_service_event" if event_ready else "memory_queue_summary",
        },
        {
            "segment": "return_path",
            "mat_mean_cycles": f"{ret_mat['mean']:.6f}",
            "vec_mean_cycles": f"{ret_vec['mean']:.6f}",
            "mat_p95_cycles": f"{ret_mat['p95']:.6f}",
            "vec_p95_cycles": f"{ret_vec['p95']:.6f}",
            "source": (
                "pending_clear_minus_memnic_send_event"
                if event_ready
                else "residual(issue_to_pending_clear-forward-memory)"
            ),
        },
        {
            "segment": "return_ready_to_first_chunk_send",
            "mat_mean_cycles": f"{ret_ready_to_first_send_mat['mean']:.6f}",
            "vec_mean_cycles": f"{ret_ready_to_first_send_vec['mean']:.6f}",
            "mat_p95_cycles": f"{ret_ready_to_first_send_mat['p95']:.6f}",
            "vec_p95_cycles": f"{ret_ready_to_first_send_vec['p95']:.6f}",
            "source": "event" if event_ready else "unavailable",
        },
        {
            "segment": "return_first_chunk_send_to_first_recv",
            "mat_mean_cycles": f"{ret_first_send_to_first_recv_mat['mean']:.6f}",
            "vec_mean_cycles": f"{ret_first_send_to_first_recv_vec['mean']:.6f}",
            "mat_p95_cycles": f"{ret_first_send_to_first_recv_mat['p95']:.6f}",
            "vec_p95_cycles": f"{ret_first_send_to_first_recv_vec['p95']:.6f}",
            "source": "event" if event_ready else "unavailable",
        },
        {
            "segment": "return_worker_chunk_span",
            "mat_mean_cycles": f"{ret_chunk_span_mat['mean']:.6f}",
            "vec_mean_cycles": f"{ret_chunk_span_vec['mean']:.6f}",
            "mat_p95_cycles": f"{ret_chunk_span_mat['p95']:.6f}",
            "vec_p95_cycles": f"{ret_chunk_span_vec['p95']:.6f}",
            "source": "event" if event_ready else "unavailable",
        },
        {
            "segment": "return_last_recv_to_gm_clear",
            "mat_mean_cycles": f"{ret_last_recv_to_gm_clear_mat['mean']:.6f}",
            "vec_mean_cycles": f"{ret_last_recv_to_gm_clear_vec['mean']:.6f}",
            "mat_p95_cycles": f"{ret_last_recv_to_gm_clear_mat['p95']:.6f}",
            "vec_p95_cycles": f"{ret_last_recv_to_gm_clear_vec['p95']:.6f}",
            "source": "event" if event_ready else "unavailable",
        },
        {
            "segment": "return_gm_clear_to_sched_observe",
            "mat_mean_cycles": f"{ret_gm_clear_to_sched_observe_mat['mean']:.6f}",
            "vec_mean_cycles": f"{ret_gm_clear_to_sched_observe_vec['mean']:.6f}",
            "mat_p95_cycles": f"{ret_gm_clear_to_sched_observe_mat['p95']:.6f}",
            "vec_p95_cycles": f"{ret_gm_clear_to_sched_observe_vec['p95']:.6f}",
            "source": "event" if event_ready else "unavailable",
        },
    ]

    table_path = Path(args.table)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", newline="") as f:
        fieldnames = [
            "segment",
            "mat_mean_cycles",
            "vec_mean_cycles",
            "mat_p95_cycles",
            "vec_p95_cycles",
            "source",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in table_rows:
            w.writerow(row)

    print("[OK] submit-ready causal split exported")
    print(f"[OK] model_source: {model_source}")
    print(f"[OK] summary: {summary_path}")
    print(f"[OK] table: {table_path}")
    if input_logs:
        print("[OK] parsed logs:")
        for pth in input_logs:
            print(f"  - {pth}")


if __name__ == "__main__":
    main()
