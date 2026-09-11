#!/usr/bin/env python3
import argparse
import csv
import math
import re
from pathlib import Path


KEY_VALUE_RE = re.compile(r"([a-zA-Z0-9_]+)=([^\s]+)")


def parse_key_values(line):
    return {key: value for key, value in KEY_VALUE_RE.findall(line)}


def parse_records(paths):
    records = {}
    commands = {}
    controllers = {}
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        for line in path.read_text(errors="ignore").splitlines():
            if "RAMULATOR2_BACKEND_SUMMARY" in line:
                raw = parse_key_values(line)
                record = {key: int(value) for key, value in raw.items()}
                records[record["node"]] = record
            elif "RAMULATOR2_COMMAND_SUMMARY" in line:
                raw = parse_key_values(line)
                record = {key: int(value) for key, value in raw.items()}
                commands[(record["node"], record["channel"])] = record
            elif "RAMULATOR2_CONTROLLER_SUMMARY" in line:
                raw = parse_key_values(line)
                record = {
                    key: (float(value) if "." in value else int(value))
                    for key, value in raw.items()
                }
                controllers[(record["node"], record["channel"])] = record
    return records, commands, controllers


def stddev(values):
    if not values:
        return 0.0
    avg = sum(values) / len(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def write_metric_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Extract Ramulator2 HBM backend summaries")
    parser.add_argument("--log", action="append", required=True)
    parser.add_argument("--exclude-node", action="append", type=int, default=[])
    parser.add_argument("--memory-clock-ghz", type=float, default=1.25)
    parser.add_argument("--channels-per-stack", type=int, default=8)
    parser.add_argument("--transaction-bytes", type=int, default=32)
    parser.add_argument("--memory-summary", required=True)
    parser.add_argument("--hbm-summary", required=True)
    parser.add_argument("--hbm-nodes", required=True)
    args = parser.parse_args()

    records, commands, controllers = parse_records([Path(name) for name in args.log])
    excluded = set(args.exclude_node)
    rows = [records[node] for node in sorted(records) if node not in excluded]
    if not rows:
        raise RuntimeError("no Ramulator2 backend summary records found after exclusions")

    peak_bytes_per_cycle = args.channels_per_stack * args.transaction_bytes
    node_rows = []
    for row in rows:
        capacity = row["read_window_cycles"] * peak_bytes_per_cycle
        utilization = 100.0 * row["read_bytes"] / capacity if capacity else 0.0
        bandwidth = row["read_bytes"] / row["read_window_cycles"] * args.memory_clock_ghz if row["read_window_cycles"] else 0.0
        node_commands = [
            command for (node, _), command in sorted(commands.items())
            if node == row["node"]
        ]
        active_commands = [command for command in node_commands if command["rd"] > 0]
        first_issue = min((command["first_rd"] for command in active_commands), default=0)
        last_issue = max((command["last_rd"] for command in active_commands), default=0)
        issue_window = last_issue - first_issue + 1 if active_commands else 0
        read_commands = sum(command["rd"] for command in node_commands)
        write_commands = sum(command["wr"] for command in node_commands)
        if len(node_commands) != args.channels_per_stack:
            raise RuntimeError(
                f"node {row['node']}: expected {args.channels_per_stack} command summaries, "
                f"found {len(node_commands)}"
            )
        if read_commands != row["read_count"] or write_commands != row["write_count"]:
            raise RuntimeError(
                f"node {row['node']}: command/completion conservation failed: "
                f"RD {read_commands}/{row['read_count']}, "
                f"WR {write_commands}/{row['write_count']}"
            )
        if row["read_bytes"] != row["read_count"] * args.transaction_bytes:
            raise RuntimeError(f"node {row['node']}: read byte/count conservation failed")
        if row["write_bytes"] != row["write_count"] * args.transaction_bytes:
            raise RuntimeError(f"node {row['node']}: write byte/count conservation failed")
        if row["pending"] != 0:
            raise RuntimeError(f"node {row['node']}: {row['pending']} requests remain pending")
        violations = sum(
            command.get("tccd_s_violations", 0)
            + command.get("tccd_l_violations", 0)
            for command in node_commands
        )
        if violations:
            raise RuntimeError(
                f"node {row['node']}: observed {violations} illegal tCCD intervals"
            )
        issue_slots = issue_window * args.channels_per_stack
        issue_utilization = 100.0 * read_commands / issue_slots if issue_slots else 0.0
        tccd_s = sum(command["tccd_s_pairs"] for command in node_commands)
        tccd_l = sum(command["tccd_l_pairs"] for command in node_commands)
        cross_pc = sum(command["cross_pc_pairs"] for command in node_commands)
        same_pc_pairs = tccd_s + tccd_l
        all_read_pairs = same_pc_pairs + cross_pc
        controller_rows = [
            controller for (node, _), controller in sorted(controllers.items())
            if node == row["node"]
        ]
        row_hits = sum(item["read_row_hits"] for item in controller_rows)
        row_misses = sum(item["read_row_misses"] for item in controller_rows)
        row_conflicts = sum(item["read_row_conflicts"] for item in controller_rows)
        if len(controller_rows) != args.channels_per_stack:
            raise RuntimeError(
                f"node {row['node']}: expected {args.channels_per_stack} controller summaries, "
                f"found {len(controller_rows)}"
            )
        if row_hits + row_misses + row_conflicts != read_commands:
            raise RuntimeError(
                f"node {row['node']}: row-state conservation failed: "
                f"{row_hits}+{row_misses}+{row_conflicts}!={read_commands}"
            )
        tccd_s_gap_sum = sum(command["tccd_s_gap_sum"] for command in node_commands)
        tccd_l_gap_sum = sum(command["tccd_l_gap_sum"] for command in node_commands)
        cross_pc_gap_sum = sum(command["cross_pc_gap_sum"] for command in node_commands)
        rw_switches = sum(command["rw_switches"] for command in node_commands)
        rw_switch_gap_sum = sum(command["rw_switch_gap_sum"] for command in node_commands)
        node_rows.append(
            {
                "node": row["node"],
                "channels": args.channels_per_stack,
                "read_commands": read_commands or row["read_count"],
                "write_commands": write_commands or row["write_count"],
                "read_bytes": row["read_bytes"],
                "write_bytes": row["write_bytes"],
                "first_read_arrival_cycle": row["first_read_arrival_cycle"],
                "last_read_complete_cycle": row["last_read_complete_cycle"],
                "read_service_window_cycles": row["read_window_cycles"],
                "read_service_window_utilization_pct": f"{utilization:.6f}",
                "first_read_issue_cycle": first_issue,
                "last_read_issue_cycle": last_issue,
                "read_issue_window_cycles": issue_window,
                "read_command_utilization_pct": f"{issue_utilization:.6f}",
                "tccd_s_pairs": tccd_s,
                "tccd_l_pairs": tccd_l,
                "cross_pseudo_channel_pairs": cross_pc,
                "tccd_s_pair_pct": f"{100.0 * tccd_s / same_pc_pairs:.6f}" if same_pc_pairs else "",
                "tccd_l_pair_pct": f"{100.0 * tccd_l / same_pc_pairs:.6f}" if same_pc_pairs else "",
                "cross_pseudo_channel_pair_pct": f"{100.0 * cross_pc / all_read_pairs:.6f}" if all_read_pairs else "",
                "tccd_s_avg_gap_cycles": f"{tccd_s_gap_sum / tccd_s:.6f}" if tccd_s else "",
                "tccd_l_avg_gap_cycles": f"{tccd_l_gap_sum / tccd_l:.6f}" if tccd_l else "",
                "cross_pseudo_channel_avg_gap_cycles": f"{cross_pc_gap_sum / cross_pc:.6f}" if cross_pc else "",
                "tccd_s_min_gap_cycles": min(
                    (item.get("tccd_s_min_gap", 0) for item in node_commands if item.get("tccd_s_pairs", 0)),
                    default="",
                ),
                "tccd_l_min_gap_cycles": min(
                    (item.get("tccd_l_min_gap", 0) for item in node_commands if item.get("tccd_l_pairs", 0)),
                    default="",
                ),
                "read_write_switches": rw_switches,
                "read_write_switch_avg_gap_cycles": f"{rw_switch_gap_sum / rw_switches:.6f}" if rw_switches else "",
                "read_row_hit_pct": (
                    f"{100.0 * row_hits / read_commands:.6f}"
                    if controller_rows and read_commands else ""
                ),
                "read_row_miss_pct": (
                    f"{100.0 * row_misses / read_commands:.6f}"
                    if controller_rows and read_commands else ""
                ),
                "read_row_conflict_pct": (
                    f"{100.0 * row_conflicts / read_commands:.6f}"
                    if controller_rows and read_commands else ""
                ),
                "read_bandwidth_gbps": f"{bandwidth:.6f}",
                "avg_read_latency_cycles": row["avg_read_latency_cycles"],
                "p95_read_latency_cycles": row["p95_read_latency_cycles"],
                "p99_read_latency_cycles": row["p99_read_latency_cycles"],
                "max_read_latency_cycles": row["max_read_latency_cycles"],
                "dependency_read_blocked": row.get("dependency_read_blocked", 0),
                "dependency_write_blocked": row.get("dependency_write_blocked", 0),
                "concurrent_alias_reads": row.get("concurrent_alias_reads", 0),
                "pending": row["pending"],
            }
        )

    node_path = Path(args.hbm_nodes)
    node_path.parent.mkdir(parents=True, exist_ok=True)
    with node_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(node_rows[0]))
        writer.writeheader()
        writer.writerows(node_rows)

    active_rows = [row for row in rows if row["read_count"] > 0]
    total_capacity = sum(row["read_window_cycles"] * peak_bytes_per_cycle for row in active_rows)
    total_read_bytes = sum(row["read_bytes"] for row in rows)
    total_reads = sum(row["read_count"] for row in rows)
    total_writes = sum(row["write_count"] for row in rows)
    aggregate_util = 100.0 * total_read_bytes / total_capacity if total_capacity else 0.0
    command_node_util_array = "[" + ",".join(
        f"{float(row['read_command_utilization_pct']):.2f}%" for row in node_rows
    ) + "]"
    service_node_util_array = "[" + ",".join(
        f"{float(row['read_service_window_utilization_pct']):.2f}%" for row in node_rows
    ) + "]"
    total_issue_slots = sum(row["read_issue_window_cycles"] * args.channels_per_stack for row in node_rows)
    total_read_commands = sum(row["read_commands"] for row in node_rows)
    command_util = 100.0 * total_read_commands / total_issue_slots if total_issue_slots else 0.0
    total_s = sum(row["tccd_s_pairs"] for row in node_rows)
    total_l = sum(row["tccd_l_pairs"] for row in node_rows)
    total_cross_pc = sum(row["cross_pseudo_channel_pairs"] for row in node_rows)
    total_same_pc = total_s + total_l
    total_read_pairs = total_same_pc + total_cross_pc
    write_metric_csv(
        Path(args.hbm_summary),
        [
            ("hbm_read_command_utilization_pct", f"{command_util:.6f}"),
            ("hbm_read_command_node_utilization_pct_array", command_node_util_array),
            ("hbm_read_command_count", total_read_commands),
            ("hbm_read_command_burst_cycles", 1),
            ("hbm_read_command_occupied_data_bus_cycles", total_reads),
            ("hbm_read_command_capacity_slots", total_issue_slots),
            ("hbm_read_command_tccd_s_pair_pct", f"{100.0 * total_s / total_same_pc:.6f}" if total_same_pc else ""),
            ("hbm_read_command_tccd_l_pair_pct", f"{100.0 * total_l / total_same_pc:.6f}" if total_same_pc else ""),
            ("hbm_read_command_cross_pseudo_channel_pair_pct", f"{100.0 * total_cross_pc / total_read_pairs:.6f}" if total_read_pairs else ""),
            ("hbm_read_command_tccd_s_avg_gap_cycles", f"{sum(float(row['tccd_s_avg_gap_cycles']) * row['tccd_s_pairs'] for row in node_rows if row['tccd_s_pairs']) / total_s:.6f}" if total_s else ""),
            ("hbm_read_command_tccd_l_avg_gap_cycles", f"{sum(float(row['tccd_l_avg_gap_cycles']) * row['tccd_l_pairs'] for row in node_rows if row['tccd_l_pairs']) / total_l:.6f}" if total_l else ""),
            ("hbm_read_row_hit_pct", f"{sum(float(row['read_row_hit_pct']) * row['read_commands'] for row in node_rows if row['read_row_hit_pct']) / total_read_commands:.6f}" if total_read_commands and all(row['read_row_hit_pct'] != '' for row in node_rows if row['read_commands']) else ""),
            ("hbm_read_write_switch_count", sum(row["read_write_switches"] for row in node_rows)),
            ("ramulator2_dependency_read_blocked", sum(row["dependency_read_blocked"] for row in node_rows)),
            ("ramulator2_dependency_write_blocked", sum(row["dependency_write_blocked"] for row in node_rows)),
            ("ramulator2_concurrent_alias_reads", sum(row["concurrent_alias_reads"] for row in node_rows)),
        ],
    )

    weighted_avg = (
        sum(row["avg_read_latency_cycles"] * row["read_count"] for row in rows) / total_reads if total_reads else 0.0
    )
    bandwidths = [float(row["read_bandwidth_gbps"]) for row in node_rows]
    bandwidth_mean = sum(bandwidths) / len(bandwidths)
    imbalance = stddev(bandwidths) / bandwidth_mean if bandwidth_mean else 0.0
    global_read_window = (
        max(row["last_read_complete_cycle"] for row in active_rows)
        - min(row["first_read_arrival_cycle"] for row in active_rows)
        + 1
        if active_rows else 0
    )
    write_metric_csv(
        Path(args.memory_summary),
        [
            ("data_node_count", len(rows)),
            ("channel_count", len(rows) * args.channels_per_stack),
            ("total_reads_done", total_reads),
            ("total_writes_done", total_writes),
            ("mem_avg_read_latency_cycles", f"{weighted_avg:.6f}"),
            ("mem_max_channel_avg_read_latency_cycles", max(row["avg_read_latency_cycles"] for row in rows)),
            ("mem_p95_read_latency_bucket_cycles", max(row["p95_read_latency_cycles"] for row in rows)),
            ("mem_read_tail_ge_100_pct", ""),
            ("mem_avg_write_latency_cycles", ""),
            ("mem_p95_write_latency_cycles", ""),
            ("hbm_channel_avg_bandwidth_gbps", f"{bandwidth_mean / args.channels_per_stack:.6f}"),
            ("hbm_aggregate_bandwidth_gbps", f"{sum(bandwidths):.6f}"),
            ("hbm_peak_channel_bandwidth_gbps", f"{max(bandwidths) / args.channels_per_stack:.6f}"),
            ("hbm_channel_bandwidth_imbalance", f"{imbalance:.6f}"),
            ("backend_read_window_cycles", global_read_window),
            ("hbm_read_utilization_pct", f"{aggregate_util:.6f}"),
            ("hbm_read_service_window_node_utilization_pct_array", service_node_util_array),
            ("memory_backend_read_latency_avg_cycles", f"{weighted_avg:.6f}"),
            ("memory_backend_read_latency_p99_cycles", max(row["p99_read_latency_cycles"] for row in rows)),
        ],
    )


if __name__ == "__main__":
    main()
