#!/usr/bin/env python3
import argparse
import configparser
import csv
import json
import re
from pathlib import Path


def node_id(path: Path) -> int:
    match = re.fullmatch(r"node(\d+)", path.parent.name)
    return int(match.group(1)) if match else 0


def load_channels(path: Path):
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        channels = [value for value in raw.values() if isinstance(value, dict)]
        if channels and "read_issue_span_cycles" not in channels[0]:
            raise RuntimeError(
                f"{path} was produced by DRAMSim3 without READ issue instrumentation"
            )
        return channels
    return []


def read_burst_cycles(path: Path) -> int:
    config = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    config.read(path)
    protocol = config.get("dram_structure", "protocol", fallback="").upper()
    burst_length = config.getint("dram_structure", "BL", fallback=0)
    if burst_length == 0:
        return 0
    divisor = {"GDDR5": 4, "GDDR5X": 8, "GDDR6": 16}.get(protocol, 2)
    return burst_length // divisor


def main():
    parser = argparse.ArgumentParser(
        description="Compute actual HBM READ-command issue-slot utilization."
    )
    parser.add_argument("--json", action="append", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--exclude-node", action="append", type=int, default=[])
    parser.add_argument("--summary", required=True)
    parser.add_argument("--nodes", required=True)
    args = parser.parse_args()

    excluded = set(args.exclude_node)
    burst_cycles = read_burst_cycles(args.config)
    if burst_cycles <= 0:
        raise RuntimeError(f"Cannot derive a positive burst_cycle from {args.config}")
    rows = []
    for name in sorted(args.json, key=lambda item: node_id(Path(item))):
        path = Path(name)
        nid = node_id(path)
        if nid in excluded:
            continue
        channels = load_channels(path)
        active = [ch for ch in channels if int(ch.get("num_read_cmds", 0)) > 0]
        read_cmds = sum(int(ch.get("num_read_cmds", 0)) for ch in channels)
        first = min((int(ch.get("read_issue_first_cycle", 0)) for ch in active), default=0)
        last = max(
            (
                int(ch.get("read_issue_first_cycle", 0))
                + int(ch.get("read_issue_span_cycles", 0))
                for ch in active
            ),
            default=0,
        )
        window = last - first + 1 if active else 0
        capacity_slots = len(channels) * window
        occupied_slots = read_cmds * burst_cycles
        utilization = 100.0 * occupied_slots / capacity_slots if capacity_slots else 0.0
        s_pairs = sum(int(ch.get("read_issue_tccd_s_pairs", 0)) for ch in channels)
        l_pairs = sum(int(ch.get("read_issue_tccd_l_pairs", 0)) for ch in channels)
        cross_rank_pairs = sum(
            int(ch.get("read_issue_cross_rank_pairs", 0)) for ch in channels
        )
        pair_count = s_pairs + l_pairs + cross_rank_pairs
        rows.append(
            {
                "node": nid,
                "channels": len(channels),
                "read_commands": read_cmds,
                "first_read_issue_cycle": first,
                "last_read_issue_cycle": last,
                "read_issue_window_cycles": window,
                "capacity_command_slots": capacity_slots,
                "occupied_data_bus_cycles": occupied_slots,
                "read_command_utilization_pct": f"{utilization:.6f}",
                "tccd_s_pairs": s_pairs,
                "tccd_l_pairs": l_pairs,
                "cross_rank_pairs": cross_rank_pairs,
                "tccd_s_pair_pct": f"{100.0 * s_pairs / pair_count:.6f}" if pair_count else "",
                "tccd_l_pair_pct": f"{100.0 * l_pairs / pair_count:.6f}" if pair_count else "",
            }
        )

    total_commands = sum(row["read_commands"] for row in rows)
    total_slots = sum(row["capacity_command_slots"] for row in rows)
    total_occupied_slots = sum(row["occupied_data_bus_cycles"] for row in rows)
    aggregate = 100.0 * total_occupied_slots / total_slots if total_slots else 0.0
    all_pairs = sum(
        row["tccd_s_pairs"] + row["tccd_l_pairs"] + row["cross_rank_pairs"]
        for row in rows
    )
    total_s = sum(row["tccd_s_pairs"] for row in rows)
    total_l = sum(row["tccd_l_pairs"] for row in rows)
    node_array = "[" + ",".join(
        f"{float(row['read_command_utilization_pct']):.2f}%" for row in rows
    ) + "]"

    summary_rows = [
        ("hbm_read_command_utilization_pct", f"{aggregate:.6f}"),
        ("hbm_read_command_node_utilization_pct_array", node_array),
        ("hbm_read_command_count", str(total_commands)),
        ("hbm_read_command_burst_cycles", str(burst_cycles)),
        ("hbm_read_command_occupied_data_bus_cycles", str(total_occupied_slots)),
        ("hbm_read_command_capacity_slots", str(total_slots)),
        ("hbm_read_command_tccd_s_pair_pct", f"{100.0 * total_s / all_pairs:.6f}" if all_pairs else ""),
        ("hbm_read_command_tccd_l_pair_pct", f"{100.0 * total_l / all_pairs:.6f}" if all_pairs else ""),
    ]

    node_output = Path(args.nodes)
    node_output.parent.mkdir(parents=True, exist_ok=True)
    with node_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["node"])
        writer.writeheader()
        writer.writerows(rows)

    summary_output = Path(args.summary)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    with summary_output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
