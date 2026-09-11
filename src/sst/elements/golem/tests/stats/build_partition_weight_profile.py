#!/usr/bin/env python3

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


ROUTER_RE = re.compile(r"^rtr_(\d+)$")
CORE_RE = re.compile(r"^core(\d+)$")
NODE_RE = re.compile(r"node(\d+)$")


def _scaled(values, total_weight):
    total = float(sum(values.values()))
    if total <= 0.0 or total_weight <= 0.0:
        return {key: 0.0 for key in values}
    return {key: total_weight * value / total for key, value in values.items()}


def _read_plan(path):
    router_roles = {}
    cpu_router = {}
    memory_router = {}
    manager_routers = set()
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            router_id = int(row["router_id"])
            roles = set(row["roles"].split(";"))
            router_roles[router_id] = row["roles"]
            for role in roles:
                if role.startswith("cpu") and role[3:].isdigit():
                    cpu_router[int(role[3:])] = router_id
                elif role.startswith("memory") and role[6:].isdigit():
                    memory_router[int(role[6:])] = router_id
            if "manager" in roles:
                manager_routers.add(router_id)
    if not router_roles:
        raise ValueError(f"empty partition plan: {path}")
    return router_roles, cpu_router, memory_router, manager_routers


def _read_sst_events(path):
    router_events = defaultdict(int)
    cpu_cycles = defaultdict(int)
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"ComponentName", "StatisticName", "Sum.u64"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"SST stats missing columns {sorted(required)}: {path}")
        for row in reader:
            component = row["ComponentName"]
            statistic = row["StatisticName"]
            value = int(row["Sum.u64"] or 0)
            router_match = ROUTER_RE.fullmatch(component)
            if router_match and statistic in {"send_packet_count", "xbar_stalls"}:
                router_events[int(router_match.group(1))] += value
                continue
            core_match = CORE_RE.fullmatch(component)
            if core_match and statistic == "cycles":
                cpu_cycles[int(core_match.group(1))] += value
    return dict(router_events), dict(cpu_cycles)


def _read_memory_events(directory):
    memory_events = {}
    for path in sorted(Path(directory).glob("node*/dramsim3.json")):
        match = NODE_RE.fullmatch(path.parent.name)
        if not match:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        transactions = 0
        for channel in data.values():
            if not isinstance(channel, dict):
                continue
            transactions += int(channel.get("num_reads_done", 0))
            transactions += int(channel.get("num_writes_done", 0))
        memory_events[int(match.group(1))] = transactions
    return memory_events


def main():
    parser = argparse.ArgumentParser(
        description="Build event-weighted router costs for weighted_topology partitioning"
    )
    parser.add_argument("--stats", required=True, help="Merged stats_selfcom.txt")
    parser.add_argument("--partition-plan", required=True, help="Prior partition_plan.csv")
    parser.add_argument("--dramsim-dir", required=True, help="Directory containing node*/dramsim3.json")
    parser.add_argument("--output", required=True, help="Output router_id,weight CSV")
    parser.add_argument("--router-total-weight", type=float, default=2.8)
    parser.add_argument("--cpu-total-weight", type=float, default=20.0)
    parser.add_argument("--memory-total-weight", type=float, default=20.0)
    parser.add_argument("--manager-total-weight", type=float, default=2.0)
    parser.add_argument("--os-weight", type=float, default=0.5)
    parser.add_argument("--base-weight", type=float, default=0.01)
    args = parser.parse_args()

    for name in (
        "router_total_weight",
        "cpu_total_weight",
        "memory_total_weight",
        "manager_total_weight",
        "os_weight",
        "base_weight",
    ):
        if getattr(args, name) < 0.0:
            raise SystemExit(f"[ERROR] --{name.replace('_', '-')} must be non-negative")
    if args.base_weight <= 0.0:
        raise SystemExit("[ERROR] --base-weight must be positive")

    router_roles, cpu_router, memory_router, manager_routers = _read_plan(
        args.partition_plan
    )
    router_events, cpu_cycles = _read_sst_events(args.stats)
    memory_events = _read_memory_events(args.dramsim_dir)

    router_cost = _scaled(
        {router_id: router_events.get(router_id, 0) for router_id in router_roles},
        args.router_total_weight,
    )
    cpu_cost_by_core = _scaled(
        {core_id: cpu_cycles.get(core_id, 0) for core_id in cpu_router},
        args.cpu_total_weight,
    )
    data_nodes = {
        node_id: memory_events.get(node_id, 0)
        for node_id in memory_router
        if node_id != 0
    }
    memory_cost_by_node = _scaled(data_nodes, args.memory_total_weight)
    manager_cost = (
        args.manager_total_weight / len(manager_routers) if manager_routers else 0.0
    )

    weights = {router_id: args.base_weight for router_id in router_roles}
    for router_id, value in router_cost.items():
        weights[router_id] += value
    for core_id, router_id in cpu_router.items():
        weights[router_id] += cpu_cost_by_core.get(core_id, 0.0)
    for node_id, router_id in memory_router.items():
        if node_id == 0:
            weights[router_id] += args.os_weight
        else:
            weights[router_id] += memory_cost_by_node.get(node_id, 0.0)
    for router_id in manager_routers:
        weights[router_id] += manager_cost

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "router_id",
                "weight",
                "router_events",
                "cpu_cycles",
                "memory_transactions",
                "roles",
            )
        )
        router_to_cpu = {router_id: core_id for core_id, router_id in cpu_router.items()}
        router_to_memory = {
            router_id: node_id for node_id, router_id in memory_router.items()
        }
        for router_id in sorted(router_roles):
            core_id = router_to_cpu.get(router_id)
            node_id = router_to_memory.get(router_id)
            writer.writerow(
                (
                    router_id,
                    f"{weights[router_id]:.9f}",
                    router_events.get(router_id, 0),
                    cpu_cycles.get(core_id, 0) if core_id is not None else 0,
                    memory_events.get(node_id, 0) if node_id is not None else 0,
                    router_roles[router_id],
                )
            )

    print(f"[OK] wrote event-weighted partition profile: {output}")
    print(
        f"[OK] observed routers={len(router_events)} cpus={len(cpu_cycles)} "
        f"memory_nodes={len(memory_events)} total_weight={sum(weights.values()):.6f}"
    )


if __name__ == "__main__":
    main()
