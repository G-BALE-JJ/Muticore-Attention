#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


OWNER_RE = re.compile(
    r"\[memNICBase bridge\] CREDIT_OWNER_SUMMARY "
    r"name=(?P<name>\S+) group=(?P<group>\d+) cap=(?P<cap>\d+) "
    r"available=(?P<available>\d+) chunk_bytes=(?P<chunk_bytes>\d+) "
    r"admitted=(?P<admitted>\d+) released=(?P<released>\d+) "
    r"blocked_requests=(?P<blocked_requests>\d+) max_used=(?P<max_used>\d+) "
    r"max_queue=(?P<max_queue>\d+) pending=(?P<pending>\d+)"
    r"(?: window_priority=(?P<window_priority>\d+)"
    r" reorder_cycles=(?P<reorder_cycles>\d+)"
    r" priority_reorders=(?P<priority_reorders>\d+)"
    r"(?: tile_quantum=(?P<tile_quantum>\d+)"
    r" bundle_turns=(?P<bundle_turns>\d+)"
    r" bundle_admissions=(?P<bundle_admissions>\d+)"
    r"(?: response_tile_priority=(?P<response_tile_priority>\d+)"
    r" response_reorder_cycles=(?P<response_reorder_cycles>\d+)"
    r" response_quantum=(?P<response_quantum>\d+)"
    r" response_reorders=(?P<response_reorders>\d+)"
    r" response_bundle_turns=(?P<response_bundle_turns>\d+)"
    r" response_sent=(?P<response_sent>\d+)"
    r" response_max_queue=(?P<response_max_queue>\d+)"
    r" response_hold_mean=(?P<response_hold_mean>\d+)"
    r" response_hold_max=(?P<response_hold_max>\d+))?)?)?"
)


def parse_records(paths):
    latest = {}
    for path in paths:
        for line in path.read_text(errors="ignore").splitlines():
            match = OWNER_RE.search(line)
            if match:
                record = match.groupdict()
                record["window_priority"] = record["window_priority"] or "0"
                record["reorder_cycles"] = record["reorder_cycles"] or "0"
                record["priority_reorders"] = record["priority_reorders"] or "0"
                record["tile_quantum"] = record["tile_quantum"] or "1"
                record["bundle_turns"] = record["bundle_turns"] or "0"
                record["bundle_admissions"] = record["bundle_admissions"] or "0"
                for key in (
                    "response_tile_priority", "response_reorder_cycles", "response_reorders",
                    "response_bundle_turns", "response_sent", "response_max_queue",
                    "response_hold_mean", "response_hold_max",
                ):
                    record[key] = record[key] or "0"
                record["response_quantum"] = record["response_quantum"] or "1"
                latest[int(record["group"])] = record
    return [latest[group] for group in sorted(latest)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="accept configurations with node admission credit disabled",
    )
    args = parser.parse_args()

    paths = []
    if args.log_dir.is_dir():
        paths.extend(sorted(path for path in args.log_dir.glob("stdout-*") if path.is_file()))
    if args.log.is_file():
        paths.append(args.log)
    records = parse_records(paths)

    args.table.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0]) if records else ["name", "group"]
    with args.table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    total_admitted = sum(int(record["admitted"]) for record in records)
    total_released = sum(int(record["released"]) for record in records)
    conserved = all(
        int(record["admitted"]) == int(record["released"])
        and int(record["available"]) == int(record["cap"])
        and int(record["pending"]) == 0
        for record in records
    )
    metrics = {
        "credit_owner_count": len(records),
        "credit_owner_total_admitted": total_admitted,
        "credit_owner_total_released": total_released,
        "credit_owner_total_blocked_requests": sum(
            int(record["blocked_requests"]) for record in records
        ),
        "credit_owner_max_used": max((int(record["max_used"]) for record in records), default=0),
        "credit_owner_max_queue": max((int(record["max_queue"]) for record in records), default=0),
        "credit_owner_window_priority_enabled": int(
            any(int(record["window_priority"]) != 0 for record in records)
        ),
        "credit_owner_reorder_cycles": max(
            (int(record["reorder_cycles"]) for record in records), default=0
        ),
        "credit_owner_priority_reorders": sum(
            int(record["priority_reorders"]) for record in records
        ),
        "credit_owner_tile_quantum": max(
            (int(record["tile_quantum"]) for record in records), default=1
        ),
        "credit_owner_bundle_turns": sum(
            int(record["bundle_turns"]) for record in records
        ),
        "credit_owner_bundle_admissions": sum(
            int(record["bundle_admissions"]) for record in records
        ),
        "credit_owner_response_tile_priority_enabled": int(
            any(int(record["response_tile_priority"]) != 0 for record in records)
        ),
        "credit_owner_response_reorder_cycles": max(
            (int(record["response_reorder_cycles"]) for record in records), default=0
        ),
        "credit_owner_response_quantum": max(
            (int(record["response_quantum"]) for record in records), default=1
        ),
        "credit_owner_response_reorders": sum(
            int(record["response_reorders"]) for record in records
        ),
        "credit_owner_response_bundle_turns": sum(
            int(record["response_bundle_turns"]) for record in records
        ),
        "credit_owner_response_sent": sum(
            int(record["response_sent"]) for record in records
        ),
        "credit_owner_response_max_queue": max(
            (int(record["response_max_queue"]) for record in records), default=0
        ),
        "credit_owner_response_hold_mean_max": max(
            (int(record["response_hold_mean"]) for record in records), default=0
        ),
        "credit_owner_response_hold_max": max(
            (int(record["response_hold_max"]) for record in records), default=0
        ),
        "credit_owner_conserved": int((not records and args.allow_empty) or (bool(records) and conserved)),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "value"])
        writer.writerows(metrics.items())

    if not records and not args.allow_empty:
        raise SystemExit("no memory-node credit owner summaries found")
    if not conserved:
        raise SystemExit("memory-node credit owner conservation check failed")
    print(f"[OK] credit owner conservation passed for {len(records)} nodes")


if __name__ == "__main__":
    main()
