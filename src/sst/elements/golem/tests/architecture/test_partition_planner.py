#!/usr/bin/env python3

import csv
import tempfile
import unittest
from collections import deque
from pathlib import Path

from partition_planner import build_weighted_topology_plan, write_partition_audit


DIM_X = 4
DIM_Y = 7
CPU_BY_ROUTER = {router_id: router_id - 4 for router_id in range(4, 24)}
DATA_MEMORY_BY_ROUTER = {router_id: router_id + 1 for router_id in range(4)}


def _plan(ranks, threads=1, profile_path=None):
    return build_weighted_topology_plan(
        dim_x=DIM_X,
        dim_y=DIM_Y,
        num_ranks=ranks,
        num_threads=threads,
        cpu_by_router=CPU_BY_ROUTER,
        data_memory_by_router=DATA_MEMORY_BY_ROUTER,
        os_router=24,
        manager_cores=range(4),
        profile_path=profile_path,
    )


def _rank_is_connected(plan, rank):
    nodes = {
        router_id
        for router_id, placement in enumerate(plan.placements)
        if placement[0] == rank
    }
    if not nodes:
        return False
    seen = {next(iter(nodes))}
    queue = deque(seen)
    while queue:
        router_id = queue.popleft()
        row, col = divmod(router_id, DIM_X)
        neighbors = []
        if col:
            neighbors.append(router_id - 1)
        if col + 1 < DIM_X:
            neighbors.append(router_id + 1)
        if row:
            neighbors.append(router_id - DIM_X)
        if row + 1 < DIM_Y:
            neighbors.append(router_id + DIM_X)
        for neighbor in neighbors:
            if neighbor in nodes and neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen == nodes


class PartitionPlannerTests(unittest.TestCase):
    def test_four_ranks_keep_each_manager_column_local(self):
        plan = _plan(4)
        self.assertEqual(plan.total_mesh_edges, 45)
        self.assertEqual(plan.cross_rank_edges, 21)
        self.assertEqual(plan.cross_rank_affinity_edges, 0)
        for core_id in range(20):
            manager_router = 4 + (core_id % 4)
            core_router = 4 + core_id
            self.assertEqual(
                plan.placements[manager_router][0], plan.placements[core_router][0]
            )
        for rank in range(4):
            self.assertTrue(_rank_is_connected(plan, rank))

    def test_eight_ranks_reduce_round_robin_mesh_cuts(self):
        plan = _plan(8)
        self.assertEqual(plan.cross_rank_edges, 25)
        self.assertLess(plan.cross_rank_edges, 37)
        self.assertEqual(plan.cross_rank_affinity_edges, 0)
        self.assertEqual(len({placement[0] for placement in plan.placements}), 8)
        for rank in range(8):
            self.assertTrue(_rank_is_connected(plan, rank))

    def test_threads_stay_inside_rank_regions(self):
        plan = _plan(4, threads=2)
        self.assertEqual(plan.cross_rank_edges, 21)
        self.assertEqual(plan.cross_thread_edges, 4)
        self.assertEqual(plan.cross_rank_affinity_edges, 0)
        self.assertEqual(plan.cross_thread_affinity_edges, 0)
        self.assertEqual(
            len({placement for placement in plan.placements}),
            8,
        )

    def test_profile_override_is_deterministic_and_auditable(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "profile.csv"
            with profile_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("router_id", "weight"))
                writer.writerow((0, 20.0))
                writer.writerow((24, 2.0))
            first = _plan(8, profile_path=str(profile_path))
            second = _plan(8, profile_path=str(profile_path))
            self.assertEqual(first, second)
            self.assertEqual(first.router_weights[0], 20.0)
            self.assertIn("profile_weight", first.router_roles[0])

            audit_path = Path(directory) / "partition_plan.csv"
            write_partition_audit(str(audit_path), first)
            with audit_path.open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), DIM_X * DIM_Y)
            self.assertEqual(rows[0]["weight"], "20.000000")

    def test_profile_balance_does_not_split_manager_affinity(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "profile.csv"
            with profile_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("router_id", "weight"))
                writer.writerow((3, 4.0))
            plan = _plan(8, profile_path=str(profile_path))
            manager_rank = plan.placements[7][0]
            for worker_router in (11, 15, 19, 23):
                self.assertEqual(plan.placements[worker_router][0], manager_rank)

    def test_rejects_more_ranks_than_routers(self):
        with self.assertRaisesRegex(ValueError, "one router per rank"):
            _plan(29)


if __name__ == "__main__":
    unittest.main()
