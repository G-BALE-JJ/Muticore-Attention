#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


PATTERN = re.compile(
    r"MERLIN_PACKET_LATENCY_GLOBAL count=(?P<count>\d+) avg_ns=(?P<avg>\d+) p95_ns=(?P<p95>\d+) p99_ns=(?P<p99>\d+) max_ns=(?P<max>\d+)"
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
        description="Extract exact NoC packet latency summary from logs"
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
    for p in inputs:
        for line in p.read_text(errors="ignore").splitlines():
            m = PATTERN.search(line)
            if m:
                record = m.groupdict()

    with output.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        if record is None:
            w.writerow(["packet_count", 0])
            w.writerow(["noc_avg_packet_latency_ns", 0])
            w.writerow(["noc_p95_packet_latency_ns", 0])
            w.writerow(["noc_p99_packet_latency_ns", 0])
            w.writerow(["noc_max_packet_latency_ns", 0])
        else:
            w.writerow(["packet_count", record["count"]])
            w.writerow(["noc_avg_packet_latency_ns", record["avg"]])
            w.writerow(["noc_p95_packet_latency_ns", record["p95"]])
            w.writerow(["noc_p99_packet_latency_ns", record["p99"]])
            w.writerow(["noc_max_packet_latency_ns", record["max"]])
    print(f"[OK] wrote {output}")


if __name__ == "__main__":
    main()
