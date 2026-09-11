#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path


CHANNEL_RE = re.compile(r"## Statistics of Channel (?P<channel>\d+)")
KV_RE = re.compile(r"^(?P<key>[A-Za-z0-9_\.\[\]\-]+)\s*=\s*(?P<value>[-+0-9.eE]+)")


def parse_channel_blocks(text: str):
    channels = {}
    current = None
    for line in text.splitlines():
        m = CHANNEL_RE.search(line)
        if m:
            current = int(m.group("channel"))
            channels[current] = {}
            continue
        if current is None:
            continue
        kv = KV_RE.match(line.strip())
        if not kv:
            continue
        key = kv.group("key")
        raw = kv.group("value")
        try:
            value = float(raw)
        except ValueError:
            continue
        channels[current][key] = value
    return channels


def estimate_p95_from_histogram(counts):
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    threshold = total * 0.95
    running = 0
    for upper, count in sorted(counts.items(), key=lambda item: item[0]):
        running += count
        if running >= threshold:
            return float(upper)
    return float(max(counts.keys()))


def stddev(values):
    if not values:
        return 0.0
    avg = sum(values) / len(values)
    return math.sqrt(sum((v - avg) ** 2 for v in values) / len(values))


def main():
    parser = argparse.ArgumentParser(
        description="Extract compact HBM/DRAMSim3 summary metrics"
    )
    parser.add_argument(
        "--json",
        action="append",
        required=True,
        help="dramsim3.json path (repeat for MPI memory nodes)",
    )
    parser.add_argument(
        "--txt",
        action="append",
        required=True,
        help="dramsim3.txt path (repeat for MPI memory nodes)",
    )
    parser.add_argument(
        "--exclude-node",
        action="append",
        type=int,
        default=[],
        help="Exclude nodeN parent directories from aggregate statistics",
    )
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    if len(args.json) != len(args.txt):
        parser.error("--json and --txt must have the same number of paths")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    excluded_nodes = set(args.exclude_node)

    def included(path_name):
        match = re.fullmatch(r"node(\d+)", Path(path_name).parent.name)
        return match is None or int(match.group(1)) not in excluded_nodes

    json_names = [name for name in args.json if included(name)]
    txt_names = [name for name in args.txt if included(name)]
    if not json_names:
        parser.error("all DRAMSim3 node inputs were excluded")

    channels = []
    for json_name in json_names:
        data = json.loads(Path(json_name).read_text())
        channels.extend(data[key] for key in sorted(data.keys(), key=lambda x: int(x)))

    txt_channels = {}
    for txt_name in txt_names:
        for channel, channel_stats in parse_channel_blocks(
            Path(txt_name).read_text(errors="ignore")
        ).items():
            merged_stats = txt_channels.setdefault(channel, {})
            for key, value in channel_stats.items():
                merged_stats[key] = merged_stats.get(key, 0.0) + value

    avg_latencies = [float(ch.get("average_read_latency", 0.0)) for ch in channels]
    avg_bandwidths = [float(ch.get("average_bandwidth", 0.0)) for ch in channels]
    reads_done = [int(ch.get("num_reads_done", 0)) for ch in channels]
    writes_done = [int(ch.get("num_writes_done", 0)) for ch in channels]
    total_reads = sum(reads_done)
    total_writes = sum(writes_done)

    weighted_avg_latency = 0.0
    if total_reads > 0:
        weighted_avg_latency = (
            sum(lat * cnt for lat, cnt in zip(avg_latencies, reads_done)) / total_reads
        )

    write_hist = {}
    for channel in channels:
        for latency, count in (channel.get("write_latency") or {}).items():
            cycle = int(latency)
            write_hist[cycle] = write_hist.get(cycle, 0) + int(count)
    write_hist_total = sum(write_hist.values())
    weighted_avg_write_latency = (
        sum(cycle * count for cycle, count in write_hist.items()) / write_hist_total
        if write_hist_total > 0
        else 0.0
    )

    global_hist = {}
    tail_ge_100 = 0
    for channel_stats in txt_channels.values():
        for key, value in channel_stats.items():
            if not key.startswith("read_latency["):
                continue
            count = int(value)
            bucket = key[len("read_latency[") : -1]
            if bucket == "-0":
                upper = 0
            elif bucket.endswith("-"):
                upper = int(bucket[:-1])
            else:
                _, upper_text = bucket.split("-")
                upper = int(upper_text)
            global_hist[upper] = global_hist.get(upper, 0) + count
            if upper >= 100:
                tail_ge_100 += count

    hist_total = sum(global_hist.values())
    tail_ge_100_pct = (100.0 * tail_ge_100 / hist_total) if hist_total > 0 else 0.0
    bandwidth_mean = (
        (sum(avg_bandwidths) / len(avg_bandwidths)) if avg_bandwidths else 0.0
    )
    bandwidth_imbalance = (
        (stddev(avg_bandwidths) / bandwidth_mean) if bandwidth_mean > 0 else 0.0
    )

    rows = [
        ["metric", "value"],
        ["data_node_count", len(json_names)],
        ["channel_count", len(channels)],
        ["total_reads_done", total_reads],
        ["total_writes_done", total_writes],
        ["mem_avg_read_latency_cycles", f"{weighted_avg_latency:.6f}"],
        [
            "mem_max_channel_avg_read_latency_cycles",
            f"{max(avg_latencies) if avg_latencies else 0.0:.6f}",
        ],
        [
            "mem_p95_read_latency_bucket_cycles",
            f"{estimate_p95_from_histogram(global_hist):.6f}",
        ],
        ["mem_read_tail_ge_100_pct", f"{tail_ge_100_pct:.6f}"],
        ["mem_avg_write_latency_cycles", f"{weighted_avg_write_latency:.6f}"],
        [
            "mem_p95_write_latency_cycles",
            f"{estimate_p95_from_histogram(write_hist):.6f}",
        ],
        ["hbm_channel_avg_bandwidth_gbps", f"{bandwidth_mean:.6f}"],
        ["hbm_aggregate_bandwidth_gbps", f"{sum(avg_bandwidths):.6f}"],
        ["hbm_peak_channel_bandwidth_gbps", f"{max(avg_bandwidths) if avg_bandwidths else 0.0:.6f}"],
        ["hbm_channel_bandwidth_imbalance", f"{bandwidth_imbalance:.6f}"],
    ]

    with output_path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"[OK] wrote {output_path}")


if __name__ == "__main__":
    main()
