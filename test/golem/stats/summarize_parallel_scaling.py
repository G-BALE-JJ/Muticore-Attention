#!/usr/bin/env python3

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Golem host-parallel wall time")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with args.summary.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"no rows in {args.summary}")

    simulated_times = {row["simulated_time"] for row in rows}
    if len(simulated_times) != 1:
        raise SystemExit(f"simulated time mismatch: {sorted(simulated_times)}")

    groups = defaultdict(list)
    for row in rows:
        key = (
            row.get("cpu_binding_mode", "default"),
            row.get("mpi_ranks", "1"),
            row.get("sst_threads", "1"),
            row.get("host_cpuset", ""),
        )
        groups[key].append(float(row["wall_time_sec"]))

    physical_means = {
        (ranks, threads): statistics.mean(values)
        for (mode, ranks, threads, _), values in groups.items()
        if mode == "physical"
    }
    records = []
    for (mode, ranks, threads, cpuset), values in sorted(groups.items()):
        mean = statistics.mean(values)
        physical_mean = physical_means.get((ranks, threads), mean)
        records.append(
            {
                "cpu_binding_mode": mode,
                "mpi_ranks": ranks,
                "sst_threads": threads,
                "host_cpuset": cpuset,
                "repeats": len(values),
                "wall_mean_sec": f"{mean:.3f}",
                "wall_median_sec": f"{statistics.median(values):.3f}",
                "wall_min_sec": f"{min(values):.3f}",
                "wall_max_sec": f"{max(values):.3f}",
                "wall_stdev_sec": f"{statistics.stdev(values) if len(values) > 1 else 0.0:.3f}",
                "runs_per_hour": f"{3600.0 / mean:.3f}",
                "throughput_vs_physical": f"{physical_mean / mean:.4f}",
                "simulated_time": next(iter(simulated_times)),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    for record in records:
        print(
            f"{record['cpu_binding_mode']}: mean={record['wall_mean_sec']}s "
            f"range={record['wall_min_sec']}..{record['wall_max_sec']}s "
            f"runs/hour={record['runs_per_hour']} "
            f"vs_physical={record['throughput_vs_physical']}"
        )
    print(f"simulated_time={next(iter(simulated_times))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
