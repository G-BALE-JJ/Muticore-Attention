#!/usr/bin/env python3
"""Verify that each FlashAttention manager group is colocated on one MPI rank."""

import argparse
import csv
import json
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'architecture'))
from partition_planner import attention_query_block_placements


CORE_COMPONENT = re.compile(r"^core([0-9]+)(?::|$)")
ROUTER_COMPONENT = re.compile(r"^rtr_([0-9]+)(?::|$)")


def expected_rank(core_id, mpi_ranks):
    if mpi_ranks == 8:
        return attention_query_block_placements(28, 20, mpi_ranks)[core_id + 4][0]
    manager_id = core_id if core_id < 4 else core_id % 4
    return manager_id % mpi_ranks


def expected_router_rank(router_id, mpi_ranks):
    if mpi_ranks == 8:
        return attention_query_block_placements(28, 20, mpi_ranks)[router_id][0]
    return router_id % mpi_ranks if router_id < 24 else 0


def expected_component_ranks(mpi_ranks):
    expected = {}
    for core_id in range(20):
        rank = expected_rank(core_id, mpi_ranks)
        for suffix in (
            "",
            ".processorBus",
            ".l1dcache",
            ".l1icache",
            ".dtlb",
            ".itlb",
            ".bus",
            ".l2cache",
        ):
            expected[f"core{core_id}{suffix}"] = rank
    for router_id in range(28):
        expected[f"rtr_{router_id}"] = expected_router_rank(router_id, mpi_ranks)
    expected["os"] = 0
    expected["node.os_l1cache"] = 0
    for node_index in range(5):
        rank = 0 if node_index == 0 else expected_router_rank(node_index - 1, mpi_ranks)
        expected[f"dirctrl_{node_index}"] = rank
        expected[f"memory_{node_index}"] = rank
    return expected


def verify(stats_file, placement_file, mpi_ranks):
    stats_file = Path(stats_file)
    expected_files = [
        stats_file.with_name(f"{stats_file.stem}_{rank}{stats_file.suffix}")
        for rank in range(mpi_ranks)
    ]
    discovered_files = sorted(
        stats_file.parent.glob(f"{stats_file.stem}_*{stats_file.suffix}")
    )
    ranked_files = [path for path in expected_files if path.is_file()]
    missing_rank_files = [str(path) for path in expected_files if not path.is_file()]
    unexpected_rank_files = [
        str(path) for path in discovered_files if path not in expected_files
    ]
    observed = {}
    observed_routers = {}
    conflicts = {}
    router_conflicts = {}
    file_rank_mismatches = []
    for file_rank, path in enumerate(expected_files):
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                rank = int(row["Rank"])
                if rank != file_rank:
                    file_rank_mismatches.append(
                        {"file": str(path), "reported_rank": rank}
                    )
                component_name = row.get("ComponentName", "")
                core_match = CORE_COMPONENT.match(component_name)
                if core_match:
                    core_id = int(core_match.group(1))
                    previous = observed.setdefault(core_id, rank)
                    if previous != rank:
                        conflicts[core_id] = sorted({previous, rank})
                router_match = ROUTER_COMPONENT.match(component_name)
                if router_match:
                    router_id = int(router_match.group(1))
                    previous = observed_routers.setdefault(router_id, rank)
                    if previous != rank:
                        router_conflicts[router_id] = sorted({previous, rank})

    expected = {core_id: expected_rank(core_id, mpi_ranks) for core_id in range(20)}
    missing = sorted(set(expected) - set(observed))
    misplaced = {
        core_id: {"expected": rank, "actual": observed[core_id]}
        for core_id, rank in expected.items()
        if core_id in observed and observed[core_id] != rank
    }
    expected_routers = {
        router_id: expected_router_rank(router_id, mpi_ranks)
        for router_id in range(28)
    }
    missing_routers = sorted(set(expected_routers) - set(observed_routers))
    misplaced_routers = {
        router_id: {"expected": rank, "actual": observed_routers[router_id]}
        for router_id, rank in expected_routers.items()
        if router_id in observed_routers and observed_routers[router_id] != rank
    }
    placement_file = Path(placement_file)
    placement_error = None
    configured_components = {}
    placement_mpi_ranks = None
    if not placement_file.is_file():
        placement_error = f"missing placement manifest: {placement_file}"
    else:
        try:
            placement = json.loads(placement_file.read_text(encoding="utf-8"))
            placement_mpi_ranks = placement.get("mpi_ranks")
            configured_components = placement.get("component_ranks", {})
            if not isinstance(configured_components, dict):
                raise ValueError("component_ranks must be an object")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            placement_error = str(error)
            configured_components = {}
    expected_components = expected_component_ranks(mpi_ranks)
    missing_components = sorted(set(expected_components) - set(configured_components))
    unexpected_components = sorted(set(configured_components) - set(expected_components))
    misplaced_components = {
        name: {"expected": rank, "actual": configured_components[name]}
        for name, rank in expected_components.items()
        if name in configured_components and configured_components[name] != rank
    }
    failures = (
        missing_rank_files,
        unexpected_rank_files,
        file_rank_mismatches,
        missing,
        conflicts,
        misplaced,
        missing_routers,
        router_conflicts,
        misplaced_routers,
        [placement_error] if placement_error else [],
        [placement_mpi_ranks] if placement_mpi_ranks != mpi_ranks else [],
        missing_components,
        unexpected_components,
        misplaced_components,
    )
    status = "PASS" if not any(failures) else "FAIL"
    return {
        "status": status,
        "mpi_ranks": mpi_ranks,
        "ranked_stats_files": [str(path) for path in ranked_files],
        "missing_rank_files": missing_rank_files,
        "unexpected_rank_files": unexpected_rank_files,
        "file_rank_mismatches": file_rank_mismatches,
        "expected_core_ranks": expected,
        "observed_core_ranks": observed,
        "missing_cores": missing,
        "conflicts": conflicts,
        "misplaced_cores": misplaced,
        "expected_router_ranks": expected_routers,
        "observed_router_ranks": observed_routers,
        "missing_routers": missing_routers,
        "router_conflicts": router_conflicts,
        "misplaced_routers": misplaced_routers,
        "placement_manifest": str(placement_file),
        "placement_manifest_error": placement_error,
        "placement_manifest_mpi_ranks": placement_mpi_ranks,
        "expected_component_ranks": expected_components,
        "configured_component_ranks": configured_components,
        "missing_components": missing_components,
        "unexpected_components": unexpected_components,
        "misplaced_components": misplaced_components,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-file", required=True)
    parser.add_argument("--placement-file", required=True)
    parser.add_argument("--mpi-ranks", type=int, required=True)
    parser.add_argument("--result-json")
    args = parser.parse_args()
    if args.mpi_ranks not in (2, 4, 8):
        parser.error("--mpi-ranks must be 2, 4 or 8")

    result = verify(args.stats_file, args.placement_file, args.mpi_ranks)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.result_json:
        Path(args.result_json).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
