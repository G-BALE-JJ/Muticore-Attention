#!/usr/bin/env python3
import argparse
import csv
import math
import re
from pathlib import Path


LINE_RE = re.compile(
    r"^(rtr_(?P<router>\d+)),(?P<metric>[a-zA-Z0-9_]+),(?P<port>port\d+),Accumulator,(?P<sim_ps>\d+),\d+,(?P<sum>\d+),"
)


def parse_bw_bits_per_sec(text: str) -> float:
    raw = text.strip()
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTP]?)(B|b)/s", raw)
    if not m:
        raise ValueError(f"unsupported bandwidth format: {text}")
    value = float(m.group(1))
    scale = {"": 1.0, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12, "P": 1e15}[m.group(2)]
    unit = m.group(3)
    bits_per_sec = value * scale
    if unit == "B":
        bits_per_sec *= 8.0
    return bits_per_sec


def p95(values):
    if not values:
        return 0.0
    vals = sorted(values)
    idx = int(round(0.95 * (len(vals) - 1)))
    return vals[idx]


def main():
    parser = argparse.ArgumentParser(
        description="Extract compact NoC summary metrics from SST router stats"
    )
    parser.add_argument("--input-file", required=True, help="stats_selfcom.txt path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--link-bw", default="100GB/s", help="NoC link bandwidth, e.g. 100GB/s"
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    router_stalls = {}
    router_packets = {}
    port_bits = {}
    sim_ps = None

    for line in input_path.read_text(errors="ignore").splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        router = int(m.group("router"))
        metric = m.group("metric")
        port = m.group("port")
        value = int(m.group("sum"))
        sim_ps = int(m.group("sim_ps"))

        if metric == "xbar_stalls":
            router_stalls[router] = router_stalls.get(router, 0) + value
        elif metric == "send_packet_count":
            router_packets[router] = router_packets.get(router, 0) + value
        elif metric == "send_bit_count":
            port_bits[(router, port)] = value

    if sim_ps is None:
        raise SystemExit(f"[ERROR] no router stats found in {input_path}")

    bits_per_sec = parse_bw_bits_per_sec(args.link_bw)
    sim_seconds = sim_ps * 1e-12
    max_bits_per_port = bits_per_sec * sim_seconds if sim_seconds > 0 else 0.0
    port_utils = []
    for value in port_bits.values():
        util = (100.0 * value / max_bits_per_port) if max_bits_per_port > 0 else 0.0
        port_utils.append(util)

    router_ids = sorted(
        set(router_stalls) | set(router_packets) | {r for r, _ in port_bits.keys()}
    )
    stall_values = [router_stalls.get(r, 0) for r in router_ids]
    packet_values = [router_packets.get(r, 0) for r in router_ids]
    top_k = max(1, math.ceil(len(port_utils) * 0.05)) if port_utils else 1
    top_utils = sorted(port_utils, reverse=True)[:top_k]

    rows = [
        ["metric", "value"],
        ["router_count", len(router_ids)],
        ["simulated_time_ps", sim_ps],
        ["total_send_packets", sum(packet_values)],
        ["total_send_bits", sum(port_bits.values())],
        ["total_xbar_stalls", sum(stall_values)],
        [
            "avg_router_xbar_stalls",
            f"{(sum(stall_values) / len(stall_values)) if stall_values else 0.0:.6f}",
        ],
        ["p95_router_xbar_stalls", f"{p95(stall_values):.6f}"],
        ["max_router_xbar_stalls", max(stall_values) if stall_values else 0],
        ["max_router_send_packets", max(packet_values) if packet_values else 0],
        [
            "avg_port_util_pct",
            f"{(sum(port_utils) / len(port_utils)) if port_utils else 0.0:.6f}",
        ],
        [
            "hotspot_top5pct_port_util_pct",
            f"{(sum(top_utils) / len(top_utils)) if top_utils else 0.0:.6f}",
        ],
        ["max_port_util_pct", f"{max(port_utils) if port_utils else 0.0:.6f}"],
    ]

    with output_path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"[OK] wrote {output_path}")


if __name__ == "__main__":
    main()
