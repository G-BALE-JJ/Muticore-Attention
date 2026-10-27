#!/usr/bin/env python3
import argparse
import csv
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


def pct(value: float, total: float) -> float:
    if total <= 0.0:
        return 0.0
    return 100.0 * value / total


def main():
    parser = argparse.ArgumentParser(
        description="Extract router/port hotspot tables from SST NoC stats"
    )
    parser.add_argument("--input-file", required=True, help="stats_selfcom.txt path")
    parser.add_argument("--summary", required=True, help="Output summary CSV (metric,value)")
    parser.add_argument("--router-table", required=True, help="Output router hotspot table CSV")
    parser.add_argument("--port-table", required=True, help="Output port hotspot table CSV")
    parser.add_argument(
        "--link-bw",
        default="100GB/s",
        help="NoC link bandwidth used for port utilization, e.g. 25GB/s",
    )
    parser.add_argument("--topk-router", type=int, default=10, help="Rows in router table")
    parser.add_argument("--topk-port", type=int, default=20, help="Rows in port table")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    summary_path = Path(args.summary)
    router_table_path = Path(args.router_table)
    port_table_path = Path(args.port_table)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    router_table_path.parent.mkdir(parents=True, exist_ok=True)
    port_table_path.parent.mkdir(parents=True, exist_ok=True)

    bits_per_sec = parse_bw_bits_per_sec(args.link_bw)

    router_xbar = {}
    router_out_stalls = {}
    router_send_packets = {}
    router_send_bits = {}
    port_xbar = {}
    port_out_stalls = {}
    port_send_packets = {}
    port_send_bits = {}
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
        key = (router, port)

        if metric == "xbar_stalls":
            router_xbar[router] = router_xbar.get(router, 0) + value
            port_xbar[key] = value
        elif metric == "output_port_stalls":
            router_out_stalls[router] = router_out_stalls.get(router, 0) + value
            port_out_stalls[key] = value
        elif metric == "send_packet_count":
            router_send_packets[router] = router_send_packets.get(router, 0) + value
            port_send_packets[key] = value
        elif metric == "send_bit_count":
            router_send_bits[router] = router_send_bits.get(router, 0) + value
            port_send_bits[key] = value

    if sim_ps is None:
        raise SystemExit(f"[ERROR] no router stats found in {input_path}")

    router_ids = sorted(
        set(router_xbar)
        | set(router_out_stalls)
        | set(router_send_packets)
        | set(router_send_bits)
        | {r for r, _ in port_xbar.keys()}
        | {r for r, _ in port_send_bits.keys()}
    )
    port_keys = sorted(
        set(port_xbar)
        | set(port_out_stalls)
        | set(port_send_packets)
        | set(port_send_bits),
        key=lambda x: (x[0], int(x[1].replace("port", ""))),
    )

    sim_seconds = sim_ps * 1e-12
    max_bits_per_port = bits_per_sec * sim_seconds if sim_seconds > 0.0 else 0.0

    port_util = {}
    for key in port_keys:
        bits = float(port_send_bits.get(key, 0))
        util = (100.0 * bits / max_bits_per_port) if max_bits_per_port > 0.0 else 0.0
        port_util[key] = util

    total_xbar = float(sum(router_xbar.get(r, 0) for r in router_ids))
    total_out_stalls = sum(router_out_stalls.get(r, 0) for r in router_ids)
    total_packets = sum(router_send_packets.get(r, 0) for r in router_ids)
    total_bits = sum(router_send_bits.get(r, 0) for r in router_ids)

    router_rank = sorted(
        router_ids,
        key=lambda r: (
            router_xbar.get(r, 0),
            router_send_packets.get(r, 0),
            -r,
        ),
        reverse=True,
    )

    router_rows = []
    for rank, router in enumerate(router_rank[: max(0, args.topk_router)], start=1):
        router_ports = [k for k in port_keys if k[0] == router]
        util_vals = [port_util.get(k, 0.0) for k in router_ports]
        router_rows.append(
            {
                "rank": rank,
                "router_id": f"rtr_{router}",
                "xbar_stalls": router_xbar.get(router, 0),
                "xbar_share_pct": f"{pct(router_xbar.get(router, 0), total_xbar):.6f}",
                "output_port_stalls": router_out_stalls.get(router, 0),
                "send_packets": router_send_packets.get(router, 0),
                "send_bits": router_send_bits.get(router, 0),
                "avg_port_util_pct": f"{(sum(util_vals) / len(util_vals)) if util_vals else 0.0:.6f}",
                "max_port_util_pct": f"{max(util_vals) if util_vals else 0.0:.6f}",
            }
        )

    port_rank = sorted(
        port_keys,
        key=lambda k: (
            port_xbar.get(k, 0),
            port_send_packets.get(k, 0),
            -k[0],
            -int(k[1].replace("port", "")),
        ),
        reverse=True,
    )

    port_rows = []
    for rank, key in enumerate(port_rank[: max(0, args.topk_port)], start=1):
        router, port = key
        xbar_value = port_xbar.get(key, 0)
        port_rows.append(
            {
                "rank": rank,
                "router_id": f"rtr_{router}",
                "port": port,
                "xbar_stalls": xbar_value,
                "xbar_share_pct_global": f"{pct(xbar_value, total_xbar):.6f}",
                "output_port_stalls": port_out_stalls.get(key, 0),
                "send_packets": port_send_packets.get(key, 0),
                "send_bits": port_send_bits.get(key, 0),
                "port_util_pct": f"{port_util.get(key, 0.0):.6f}",
            }
        )

    def top_router(idx: int):
        if idx < 0 or idx >= len(router_rank):
            return ("", 0, 0.0)
        r = router_rank[idx]
        v = router_xbar.get(r, 0)
        return (f"rtr_{r}", v, pct(v, total_xbar))

    def top_port():
        if not port_rank:
            return ("", "", 0, 0.0, 0.0)
        key = port_rank[0]
        router, port = key
        xbar_value = port_xbar.get(key, 0)
        return (
            f"rtr_{router}",
            port,
            xbar_value,
            pct(xbar_value, total_xbar),
            port_util.get(key, 0.0),
        )

    top1 = top_router(0)
    top2 = top_router(1)
    top3 = top_router(2)
    top_port_info = top_port()

    top3_cover = top1[1] + top2[1] + top3[1]
    nonzero_router_count = sum(1 for r in router_ids if router_xbar.get(r, 0) > 0)
    nonzero_port_count = sum(1 for k in port_keys if port_xbar.get(k, 0) > 0)

    summary_rows = [
        ("router_count", str(len(router_ids))),
        ("port_count", str(len(port_keys))),
        ("simulated_time_ps", str(sim_ps)),
        ("total_xbar_stalls", str(int(total_xbar))),
        ("total_output_port_stalls", str(total_out_stalls)),
        ("total_send_packets", str(total_packets)),
        ("total_send_bits", str(total_bits)),
        ("xbar_nonzero_router_count", str(nonzero_router_count)),
        ("xbar_nonzero_port_count", str(nonzero_port_count)),
        ("top1_router", top1[0]),
        ("top1_router_xbar_stalls", str(top1[1])),
        ("top1_router_xbar_share_pct", f"{top1[2]:.6f}"),
        ("top2_router", top2[0]),
        ("top2_router_xbar_stalls", str(top2[1])),
        ("top2_router_xbar_share_pct", f"{top2[2]:.6f}"),
        ("top3_router", top3[0]),
        ("top3_router_xbar_stalls", str(top3[1])),
        ("top3_router_xbar_share_pct", f"{top3[2]:.6f}"),
        ("top3_router_xbar_coverage_pct", f"{pct(top3_cover, total_xbar):.6f}"),
        ("top1_port_router", top_port_info[0]),
        ("top1_port", top_port_info[1]),
        ("top1_port_xbar_stalls", str(top_port_info[2])),
        ("top1_port_xbar_share_pct", f"{top_port_info[3]:.6f}"),
        ("top1_port_util_pct", f"{top_port_info[4]:.6f}"),
    ]

    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(summary_rows)

    with router_table_path.open("w", newline="") as f:
        fieldnames = [
            "rank",
            "router_id",
            "xbar_stalls",
            "xbar_share_pct",
            "output_port_stalls",
            "send_packets",
            "send_bits",
            "avg_port_util_pct",
            "max_port_util_pct",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in router_rows:
            w.writerow(row)

    with port_table_path.open("w", newline="") as f:
        fieldnames = [
            "rank",
            "router_id",
            "port",
            "xbar_stalls",
            "xbar_share_pct_global",
            "output_port_stalls",
            "send_packets",
            "send_bits",
            "port_util_pct",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in port_rows:
            w.writerow(row)

    print("[OK] NoC hotspot exports completed")
    print(f"[OK] summary: {summary_path}")
    print(f"[OK] router table: {router_table_path}")
    print(f"[OK] port table: {port_table_path}")


if __name__ == "__main__":
    main()
