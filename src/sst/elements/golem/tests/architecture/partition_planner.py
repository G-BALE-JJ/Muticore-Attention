#!/usr/bin/env python3

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


Placement = Tuple[int, int]


@dataclass(frozen=True)
class _Region:
    x0: int
    x1: int
    y0: int
    y1: int
    rank: int


@dataclass(frozen=True)
class PartitionPlan:
    dim_x: int
    dim_y: int
    placements: Tuple[Placement, ...]
    router_weights: Tuple[float, ...]
    router_roles: Tuple[str, ...]
    cross_rank_edges: int
    cross_thread_edges: int
    total_mesh_edges: int
    cross_rank_affinity_edges: int
    cross_thread_affinity_edges: int

    def rank_weights(self, num_ranks: int) -> Tuple[float, ...]:
        values = [0.0] * num_ranks
        for router_id, placement in enumerate(self.placements):
            values[placement[0]] += self.router_weights[router_id]
        return tuple(values)

    def lane_weights(self, num_ranks: int, num_threads: int) -> Tuple[float, ...]:
        values = [0.0] * (num_ranks * num_threads)
        for router_id, (rank, thread) in enumerate(self.placements):
            values[rank * num_threads + thread] += self.router_weights[router_id]
        return tuple(values)


def _linear_segments(
    weights: Sequence[float],
    part_count: int,
    cut_penalties: Optional[Mapping[int, float]] = None,
) -> List[Tuple[int, int]]:
    """Return an optimal contiguous linear partition using squared load error."""
    item_count = len(weights)
    if part_count <= 0 or part_count > item_count:
        raise ValueError(
            f"cannot split {item_count} items into {part_count} non-empty parts"
        )

    prefix = [0.0]
    for weight in weights:
        prefix.append(prefix[-1] + weight)
    target = prefix[-1] / part_count

    # dp[(parts, end)] = (cost, cuts), where cuts includes the final end index.
    dp: Dict[Tuple[int, int], Tuple[float, Tuple[int, ...]]] = {(0, 0): (0.0, ())}
    for parts in range(1, part_count + 1):
        for end in range(parts, item_count + 1):
            best: Optional[Tuple[float, Tuple[int, ...]]] = None
            for start in range(parts - 1, end):
                previous = dp.get((parts - 1, start))
                if previous is None:
                    continue
                segment_weight = prefix[end] - prefix[start]
                cut_penalty = (
                    cut_penalties.get(start, 0.0)
                    if cut_penalties is not None and start > 0
                    else 0.0
                )
                candidate = (
                    previous[0] + (segment_weight - target) ** 2 + cut_penalty,
                    previous[1] + (end,),
                )
                if best is None or candidate < best:
                    best = candidate
            if best is not None:
                dp[(parts, end)] = best

    cuts = dp[(part_count, item_count)][1]
    segments = []
    start = 0
    for end in cuts:
        segments.append((start, end))
        start = end
    return segments


def _region_weight(
    region: _Region, router_weights: Sequence[float], dim_x: int
) -> float:
    return sum(
        router_weights[row * dim_x + col]
        for row in range(region.y0, region.y1)
        for col in range(region.x0, region.x1)
    )


def _rank_regions(
    dim_x: int,
    dim_y: int,
    num_ranks: int,
    router_weights: Sequence[float],
    vertical_cut_penalties: Optional[Mapping[Tuple[int, int], float]] = None,
) -> List[_Region]:
    if num_ranks > dim_x * dim_y:
        raise ValueError(
            f"weighted_topology needs at least one router per rank: "
            f"ranks={num_ranks}, routers={dim_x * dim_y}"
        )

    column_weights = [
        sum(router_weights[row * dim_x + col] for row in range(dim_y))
        for col in range(dim_x)
    ]
    regions: List[_Region] = []

    if num_ranks <= dim_x:
        for rank, (x0, x1) in enumerate(_linear_segments(column_weights, num_ranks)):
            regions.append(_Region(x0, x1, 0, dim_y, rank))
        return regions

    ranks_per_column = [1] * dim_x
    while sum(ranks_per_column) < num_ranks:
        candidates = [
            (column_weights[col] / ranks_per_column[col], -col, col)
            for col in range(dim_x)
            if ranks_per_column[col] < dim_y
        ]
        if not candidates:
            raise ValueError("unable to allocate all ranks to non-empty topology regions")
        ranks_per_column[max(candidates)[2]] += 1

    next_rank = 0
    for col, part_count in enumerate(ranks_per_column):
        row_weights = [
            router_weights[row * dim_x + col] for row in range(dim_y)
        ]
        column_cut_penalties = (
            {
                cut: vertical_cut_penalties.get((col, cut), 0.0)
                for cut in range(1, dim_y)
            }
            if vertical_cut_penalties is not None
            else None
        )
        for y0, y1 in _linear_segments(
            row_weights, part_count, column_cut_penalties
        ):
            regions.append(_Region(col, col + 1, y0, y1, next_rank))
            next_rank += 1
    return regions


def _thread_regions(
    region: _Region,
    num_threads: int,
    router_weights: Sequence[float],
    dim_x: int,
    vertical_cut_penalties: Optional[Mapping[Tuple[int, int], float]] = None,
) -> List[Tuple[_Region, int]]:
    width = region.x1 - region.x0
    height = region.y1 - region.y0
    active_threads = min(num_threads, width * height)
    if active_threads == 1:
        return [(region, 0)]

    if height >= active_threads:
        row_weights = [
            sum(
                router_weights[row * dim_x + col]
                for col in range(region.x0, region.x1)
            )
            for row in range(region.y0, region.y1)
        ]
        row_cut_penalties = (
            {
                cut: sum(
                    vertical_cut_penalties.get(
                        (col, region.y0 + cut), 0.0
                    )
                    for col in range(region.x0, region.x1)
                )
                for cut in range(1, height)
            }
            if vertical_cut_penalties is not None
            else None
        )
        return [
            (_Region(region.x0, region.x1, region.y0 + y0, region.y0 + y1, region.rank), thread)
            for thread, (y0, y1) in enumerate(
                _linear_segments(row_weights, active_threads, row_cut_penalties)
            )
        ]

    if width >= active_threads:
        column_weights = [
            sum(
                router_weights[row * dim_x + col]
                for row in range(region.y0, region.y1)
            )
            for col in range(region.x0, region.x1)
        ]
        return [
            (_Region(region.x0 + x0, region.x0 + x1, region.y0, region.y1, region.rank), thread)
            for thread, (x0, x1) in enumerate(
                _linear_segments(column_weights, active_threads)
            )
        ]

    # This only occurs for unusually high thread counts on a multi-column rank.
    # Assign by rows while preserving rank locality; some thread IDs may be idle.
    result = []
    thread = 0
    for row in range(region.y0, region.y1):
        for col in range(region.x0, region.x1):
            result.append((_Region(col, col + 1, row, row + 1, region.rank), thread))
            thread = (thread + 1) % active_threads
    return result


def _load_profile(path: str, router_count: int) -> Dict[int, float]:
    overrides: Dict[int, float] = {}
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"router_id", "weight"}.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                "partition weight profile must contain router_id and weight columns"
            )
        for row in reader:
            router_id = int(row["router_id"])
            weight = float(row["weight"])
            if router_id < 0 or router_id >= router_count:
                raise ValueError(f"profile router_id out of range: {router_id}")
            if weight <= 0:
                raise ValueError(f"profile weight must be positive: router {router_id}")
            if router_id in overrides:
                raise ValueError(f"duplicate profile router_id: {router_id}")
            overrides[router_id] = weight
    return overrides


def build_weighted_topology_plan(
    *,
    dim_x: int,
    dim_y: int,
    num_ranks: int,
    num_threads: int,
    cpu_by_router: Mapping[int, int],
    data_memory_by_router: Mapping[int, int],
    os_router: int,
    manager_cores: Iterable[int] = (),
    router_weight: float = 0.1,
    cpu_weight: float = 1.0,
    manager_weight: float = 0.5,
    data_memory_weight: float = 5.0,
    os_weight: float = 0.5,
    affinity_edge_weight: float = 2.0,
    profile_path: Optional[str] = None,
) -> PartitionPlan:
    if dim_x <= 0 or dim_y <= 0 or num_ranks <= 0 or num_threads <= 0:
        raise ValueError("mesh dimensions, ranks, and threads must be positive")
    scalar_weights = {
        "router_weight": router_weight,
        "cpu_weight": cpu_weight,
        "manager_weight": manager_weight,
        "data_memory_weight": data_memory_weight,
        "os_weight": os_weight,
        "affinity_edge_weight": affinity_edge_weight,
    }
    for name, value in scalar_weights.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    router_count = dim_x * dim_y
    manager_set = set(manager_cores)
    weights = [router_weight] * router_count
    roles: List[List[str]] = [["router"] for _ in range(router_count)]

    for router_id, core_id in cpu_by_router.items():
        if router_id < 0 or router_id >= router_count:
            raise ValueError(f"CPU router out of range: {router_id}")
        weights[router_id] += cpu_weight
        roles[router_id].append(f"cpu{core_id}")
        if core_id in manager_set:
            weights[router_id] += manager_weight
            roles[router_id].append("manager")

    for router_id, node_id in data_memory_by_router.items():
        if router_id < 0 or router_id >= router_count:
            raise ValueError(f"memory router out of range: {router_id}")
        weights[router_id] += data_memory_weight
        roles[router_id].append(f"memory{node_id}")

    if os_router < 0 or os_router >= router_count:
        raise ValueError(f"OS router out of range: {os_router}")
    weights[os_router] += os_weight
    roles[os_router].extend(("os", "memory0"))

    if profile_path:
        for router_id, weight in _load_profile(profile_path, router_count).items():
            weights[router_id] = weight
            roles[router_id].append("profile_weight")

    vertical_cut_penalties: Dict[Tuple[int, int], float] = defaultdict(float)
    affinity_pairs = []
    manager_routers = {
        router_id
        for router_id, core_id in cpu_by_router.items()
        if core_id in manager_set
    }
    for manager_router in manager_routers:
        manager_row, manager_col = divmod(manager_router, dim_x)
        affinity_routers = [
            router_id
            for router_id in cpu_by_router
            if router_id % dim_x == manager_col and router_id != manager_router
        ]
        affinity_pairs.extend(
            (manager_router, affinity_router)
            for affinity_router in affinity_routers
        )
        for cut in range(1, dim_y):
            manager_above = manager_row < cut
            crossing = sum(
                (router_id // dim_x < cut) != manager_above
                for router_id in affinity_routers
            )
            vertical_cut_penalties[(manager_col, cut)] += (
                crossing * affinity_edge_weight
            )

    rank_regions = _rank_regions(
        dim_x,
        dim_y,
        num_ranks,
        weights,
        vertical_cut_penalties,
    )
    placements: List[Optional[Placement]] = [None] * router_count
    for rank_region in rank_regions:
        for thread_region, thread in _thread_regions(
            rank_region,
            num_threads,
            weights,
            dim_x,
            vertical_cut_penalties,
        ):
            for row in range(thread_region.y0, thread_region.y1):
                for col in range(thread_region.x0, thread_region.x1):
                    router_id = row * dim_x + col
                    if placements[router_id] is not None:
                        raise RuntimeError(f"router {router_id} assigned more than once")
                    placements[router_id] = (rank_region.rank, thread)
    if any(placement is None for placement in placements):
        raise RuntimeError("partition plan left one or more routers unassigned")

    typed_placements = tuple(placement for placement in placements if placement is not None)
    total_edges = 0
    cross_rank_edges = 0
    cross_thread_edges = 0
    for row in range(dim_y):
        for col in range(dim_x):
            router_id = row * dim_x + col
            for neighbor in (
                router_id + 1 if col + 1 < dim_x else None,
                router_id + dim_x if row + 1 < dim_y else None,
            ):
                if neighbor is None:
                    continue
                total_edges += 1
                if typed_placements[router_id][0] != typed_placements[neighbor][0]:
                    cross_rank_edges += 1
                elif typed_placements[router_id][1] != typed_placements[neighbor][1]:
                    cross_thread_edges += 1

    cross_rank_affinity_edges = sum(
        typed_placements[left][0] != typed_placements[right][0]
        for left, right in affinity_pairs
    )
    cross_thread_affinity_edges = sum(
        typed_placements[left][0] == typed_placements[right][0]
        and typed_placements[left][1] != typed_placements[right][1]
        for left, right in affinity_pairs
    )

    return PartitionPlan(
        dim_x=dim_x,
        dim_y=dim_y,
        placements=typed_placements,
        router_weights=tuple(weights),
        router_roles=tuple(";".join(role) for role in roles),
        cross_rank_edges=cross_rank_edges,
        cross_thread_edges=cross_thread_edges,
        total_mesh_edges=total_edges,
        cross_rank_affinity_edges=cross_rank_affinity_edges,
        cross_thread_affinity_edges=cross_thread_affinity_edges,
    )


def write_partition_audit(path: str, plan: PartitionPlan) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("router_id", "row", "column", "rank", "thread", "weight", "roles"))
        for router_id, ((rank, thread), weight, roles) in enumerate(
            zip(plan.placements, plan.router_weights, plan.router_roles)
        ):
            writer.writerow(
                (
                    router_id,
                    router_id // plan.dim_x,
                    router_id % plan.dim_x,
                    rank,
                    thread,
                    f"{weight:.6f}",
                    roles,
                )
            )
