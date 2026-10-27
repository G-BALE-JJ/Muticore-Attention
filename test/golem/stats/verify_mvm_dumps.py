#!/usr/bin/env python3

import argparse
import csv
import glob
import math
import os
import re
from dataclasses import dataclass


@dataclass
class SnapshotResult:
    core_id: int
    array_id: int
    snapshot_id: int
    rows: int
    cols: int
    mismatch_count: int


HEADER_RE = re.compile(r"^=== MVM Snapshot #(\d+) core=(\d+) array=(\d+) ===$")


def _parse_numbers(text: str):
    if not text.strip():
        return []
    return [float(token) for token in text.strip().split()]


def _close_enough(a: float, b: float, atol: float = 1e-4, rtol: float = 1e-4) -> bool:
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def parse_dump_file(path: str):
    results = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    idx = 0
    while idx < len(lines):
        header = HEADER_RE.match(lines[idx].strip())
        if not header:
            idx += 1
            continue

        snapshot_id = int(header.group(1))
        core_id = int(header.group(2))
        array_id = int(header.group(3))

        idx += 1
        if idx >= len(lines) or not lines[idx].startswith("InputVector:"):
            raise ValueError(
                f"{path}: malformed snapshot #{snapshot_id}, missing InputVector"
            )
        input_vec = _parse_numbers(lines[idx].split(":", 1)[1])

        idx += 1
        if idx >= len(lines) or lines[idx].strip() != "MatrixAndOutput:":
            raise ValueError(
                f"{path}: malformed snapshot #{snapshot_id}, missing MatrixAndOutput"
            )

        idx += 1
        mismatch_count = 0
        row_count = 0
        col_count = len(input_vec)
        while idx < len(lines) and lines[idx].strip():
            row_line = lines[idx]
            if "|" not in row_line:
                raise ValueError(
                    f"{path}: malformed matrix row in snapshot #{snapshot_id}: {row_line}"
                )
            left, right = row_line.split("|", 1)
            row_values = _parse_numbers(left)
            out_values = _parse_numbers(right)
            if len(out_values) != 1:
                raise ValueError(
                    f"{path}: malformed output value in snapshot #{snapshot_id}: {row_line}"
                )
            if len(row_values) != len(input_vec):
                raise ValueError(
                    f"{path}: row width mismatch in snapshot #{snapshot_id}, row={len(row_values)} input={len(input_vec)}"
                )

            expected = sum(row_values[i] * input_vec[i] for i in range(len(input_vec)))
            actual = out_values[0]
            if not _close_enough(actual, expected):
                mismatch_count += 1

            row_count += 1
            idx += 1

        results.append(
            SnapshotResult(
                core_id=core_id,
                array_id=array_id,
                snapshot_id=snapshot_id,
                rows=row_count,
                cols=col_count,
                mismatch_count=mismatch_count,
            )
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Verify MVM dump files off-RISC-V")
    parser.add_argument(
        "--dump-dir", default="artifacts/mvm_dumps", help="MVM dump root directory"
    )
    parser.add_argument(
        "--summary",
        default="artifacts/stats/mvm_verify_summary.csv",
        help="Output CSV summary path",
    )
    args = parser.parse_args()

    pattern = os.path.join(args.dump_dir, "core_*", "mvm_array_*.log")
    dump_files = sorted(glob.glob(pattern))
    if not dump_files:
        print(f"[VERIFY] No dump files found under {args.dump_dir}")
        return 2

    all_results = []
    for path in dump_files:
        all_results.extend(parse_dump_file(path))

    if not all_results:
        print(f"[VERIFY] No snapshots parsed from dumps under {args.dump_dir}")
        return 2

    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    with open(args.summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "core_id",
                "array_id",
                "snapshot_id",
                "rows",
                "cols",
                "mismatch_count",
                "status",
            ]
        )
        for item in all_results:
            writer.writerow(
                [
                    item.core_id,
                    item.array_id,
                    item.snapshot_id,
                    item.rows,
                    item.cols,
                    item.mismatch_count,
                    "PASS" if item.mismatch_count == 0 else "FAIL",
                ]
            )

    fail_count = sum(1 for item in all_results if item.mismatch_count != 0)
    print(
        f"[VERIFY] snapshots={len(all_results)} failed={fail_count} summary={args.summary}"
    )
    if fail_count == 0:
        print("[VERIFY] PASS")
        return 0

    print("[VERIFY] FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
