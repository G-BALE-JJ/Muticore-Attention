#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


PATTERN = re.compile(
    r"MEM_QUEUE_DELAY_GLOBAL count=(?P<count>\d+) avg_cycles=(?P<avg>\d+) p95_cycles=(?P<p95>\d+) p99_cycles=(?P<p99>\d+) max_cycles=(?P<max>\d+)"
)

BACKEND_PATTERN = re.compile(
    r"DRAMSIM3_BACKEND_READ_LATENCY_GLOBAL count=(?P<count>\d+) avg_cycles=(?P<avg>\d+) p95_cycles=(?P<p95>\d+) p99_cycles=(?P<p99>\d+) max_cycles=(?P<max>\d+)"
)


def resolve_inputs(log_path: Path | None, log_dir: Path, stdout_glob: str):
    inputs = []
    if log_path is not None and log_path.exists() and log_path.is_file():
        inputs.append(log_path)
    if log_dir.exists() and log_dir.is_dir():
        inputs.extend(sorted(p for p in log_dir.glob(stdout_glob) if p.is_file()))
    return inputs


def main():
    parser = argparse.ArgumentParser(
        description="Extract memory queue delay summary from logs"
    )
    parser.add_argument("--log", default="")
    parser.add_argument("--log-dir", default=".")
    parser.add_argument("--stdout-glob", default="stdout-*")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    log_path = Path(args.log) if args.log else None
    inputs = resolve_inputs(log_path, Path(args.log_dir), args.stdout_glob)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    record = None
    backend_record = None
    for p in inputs:
        for line in p.read_text(errors="ignore").splitlines():
            m = PATTERN.search(line)
            if m:
                record = m.groupdict()
            m2 = BACKEND_PATTERN.search(line)
            if m2:
                backend_record = m2.groupdict()

    with output.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        if record is None:
            w.writerow(["queue_sample_count", 0])
            w.writerow(["memory_queue_delay_avg_cycles", 0])
            w.writerow(["memory_queue_delay_p95_cycles", 0])
            w.writerow(["memory_queue_delay_p99_cycles", 0])
            w.writerow(["memory_queue_delay_max_cycles", 0])
        else:
            w.writerow(["queue_sample_count", record["count"]])
            w.writerow(["memory_queue_delay_avg_cycles", record["avg"]])
            w.writerow(["memory_queue_delay_p95_cycles", record["p95"]])
            w.writerow(["memory_queue_delay_p99_cycles", record["p99"]])
            w.writerow(["memory_queue_delay_max_cycles", record["max"]])
        if backend_record is None:
            w.writerow(["backend_read_latency_sample_count", 0])
            w.writerow(["memory_backend_read_latency_avg_cycles", 0])
            w.writerow(["memory_backend_read_latency_p95_cycles", 0])
            w.writerow(["memory_backend_read_latency_p99_cycles", 0])
            w.writerow(["memory_backend_read_latency_max_cycles", 0])
        else:
            w.writerow(["backend_read_latency_sample_count", backend_record["count"]])
            w.writerow(
                ["memory_backend_read_latency_avg_cycles", backend_record["avg"]]
            )
            w.writerow(
                ["memory_backend_read_latency_p95_cycles", backend_record["p95"]]
            )
            w.writerow(
                ["memory_backend_read_latency_p99_cycles", backend_record["p99"]]
            )
            w.writerow(
                ["memory_backend_read_latency_max_cycles", backend_record["max"]]
            )
    print(f"[OK] wrote {output}")


if __name__ == "__main__":
    main()
