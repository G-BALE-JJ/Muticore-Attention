#!/usr/bin/env python3
"""Run an attention worker-cluster SST sweep and aggregate cycle counts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "baseline/attention_cluster_8qk_8pv/run_sst.sh"


def terminate_case(process: subprocess.Popen) -> None:
    """Stop SST's detached MPI process group as well as the runner."""
    table = subprocess.check_output(
        ["ps", "-eo", "pid=,ppid=,pgid="], text=True
    ).splitlines()
    children: dict[int, list[tuple[int, int]]] = {}
    for line in table:
        pid, parent, group = (int(value) for value in line.split())
        children.setdefault(parent, []).append((pid, group))
    pending = [process.pid]
    groups = {process.pid}
    while pending:
        for child, group in children.get(pending.pop(), []):
            pending.append(child)
            groups.add(group)
    groups.discard(os.getpgrp())
    for group in groups:
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if all(not Path(f"/proc/{group}").exists() for group in groups):
            break
        time.sleep(0.1)
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def parse_int_list(value: str, name: str) -> list[int]:
    values = []
    for token in value.split(","):
        try:
            parsed = int(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name} must contain integers") from exc
        if parsed <= 0:
            raise argparse.ArgumentTypeError(f"{name} values must be positive")
        values.append(parsed)
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must not be empty")
    return values


def parse_pairs(value: str) -> list[tuple[int, int]]:
    pairs = []
    for token in value.split(","):
        try:
            hq_text, hkv_text = token.split(":", 1)
            hq, hkv = int(hq_text), int(hkv_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "--pairs must use Hq:Hkv entries, for example 1:1,2:1,4:2"
            ) from exc
        if hq <= 0 or hkv <= 0 or hq % hkv or hq // hkv not in (1, 2, 4):
            raise argparse.ArgumentTypeError(
                f"invalid GQA pair {hq}:{hkv}; Hq/Hkv must be 1, 2, or 4"
            )
        pairs.append((hq, hkv))
    if not pairs:
        raise argparse.ArgumentTypeError("--pairs must not be empty")
    return pairs


def theoretical_floor(hq: int, query_length: int, kv_length: int, head_dim: int,
                      causal: bool = False) -> int:
    """Perfect-overlap resource floor for the fixed 8+8 worker split."""
    if causal:
        query_tiles = query_length // 64
        qk_tiles = hq * query_tiles * (query_tiles + 1) // 2
        pv_windows = hq * sum((block * 64 + 255) // 256
                              for block in range(1, query_tiles + 1))
        qk_or_pv = max(math.ceil(qk_tiles * 64 * 64 * head_dim /
                                 (8 * 4096)),
                       math.ceil(pv_windows * 64 * 256 * head_dim /
                                 (8 * 4096)))
        softmax_exp = math.ceil(qk_tiles * 64 * 64 / (8 * 16))
        return max(qk_or_pv, softmax_exp)
    macs = hq * query_length * kv_length * head_dim
    array_macs_per_cycle = 64 * 64
    qk_or_pv = math.ceil(macs / (8 * array_macs_per_cycle))
    softmax_exp = math.ceil((hq * query_length * kv_length) / (8 * 16))
    return max(qk_or_pv, softmax_exp)


def case_id(hq: int, hkv: int, query_length: int, kv_length: int,
            head_dim: int = 128, dtype: str = "fp16", causal: bool = False) -> str:
    mode = "causal_" if causal else ""
    return f"{dtype}_{mode}hq{hq}_hkv{hkv}_q{query_length}_k{kv_length}_d{head_dim}"


def load_result(path: Path) -> dict:
    if not path.exists():
        return {"status": "MISSING", "error": f"missing {path}"}
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "INVALID", "error": repr(exc)}
    return result


def run_case(args: argparse.Namespace, output_root: Path, hq: int, hkv: int,
             query_length: int, kv_length: int,
             previous: dict | None = None) -> dict:
    identifier = case_id(hq, hkv, query_length, kv_length, args.head_dim, args.dtype, args.causal)
    artifact = output_root / identifier
    result_path = artifact / "sst_qk_bridge_result.json"
    floor = theoretical_floor(hq, query_length, kv_length, args.head_dim, args.causal)
    base = {
        "case_id": identifier,
        "Hq": hq,
        "Hkv": hkv,
        "query_length": query_length,
        "kv_length": kv_length,
        "head_dim": args.head_dim,
        "dtype": args.dtype,
        "causal": args.causal,
        "mpi_ranks": args.mpi_ranks,
        "local_gm_queue_depth": args.local_gm_queue_depth,
        "element_library_sha256": args.element_library_sha256,
        "row_priority": int(os.environ.get("GOLEM_ATTENTION_WORKER_CLUSTER_ROW_PRIORITY", "1")),
        "v_broadcast": int(os.environ.get("GOLEM_ATTENTION_WORKER_CLUSTER_V_BROADCAST", "1")),
        "window_major": int(os.environ.get("GOLEM_ATTENTION_WORKER_CLUSTER_WINDOW_MAJOR", "1")),
        "cross_macro_prefetch": int(os.environ.get("GOLEM_WCP_CROSS_MACRO_PREFETCH_ENABLE", "0")),
        "theoretical_resource_floor_cycles": floor,
        "artifact_root": str(artifact),
    }
    same_config = previous is not None and all(
        previous.get(key) == base[key]
        for key in ("case_id", "Hq", "Hkv", "query_length", "kv_length",
                    "head_dim", "dtype", "causal", "mpi_ranks", "local_gm_queue_depth",
                    "element_library_sha256", "row_priority", "v_broadcast", "window_major",
                    "cross_macro_prefetch")
    )
    numerical_path = artifact / "fused_attention_result.json"
    mpi_path = artifact / "attention_mpi_partition.json"
    layout_path = artifact / "attention_hbm_layout.json"
    if (args.resume and same_config and previous.get("status") == "PASS" and
            load_result(numerical_path).get("status") == "PASS" and
            load_result(mpi_path).get("status") == "PASS" and
            load_result(layout_path).get("status") == "PASS" and
            load_result(result_path).get("status") == "PASS"):
        result = load_result(result_path)
        numerical = load_result(numerical_path)
        layout = load_result(layout_path)
        return {**base, **summarize_result(result, floor),
                "numerical_status": "PASS", "mpi_status": "PASS",
                "layout_status": "PASS", "runner_returncode": 0,
                "max_abs_error": numerical.get("max_abs_error"),
                "checked_elements": numerical.get("checked"),
                "layout_checked_bytes": layout.get("checked_bytes")}
    if args.dry_run:
        return {**base, "status": "DRY_RUN", "command": " ".join(str(x) for x in command(args, artifact, hq, hkv, query_length, kv_length))}

    artifact.mkdir(parents=True, exist_ok=True)
    command_line = command(args, artifact, hq, hkv, query_length, kv_length)
    log_path = artifact / "sweep_runner.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(str(x) for x in command_line) + "\n")
        process = subprocess.Popen(
            command_line, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ,
                 "GOLEM_MPI_RANKS": str(args.mpi_ranks),
                 "GOLEM_LOCAL_GM_QUEUE_DEPTH": str(args.local_gm_queue_depth)},
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=args.timeout + 60)
        except KeyboardInterrupt:
            terminate_case(process)
            raise
        except subprocess.TimeoutExpired:
            terminate_case(process)
            return {**base, "status": "TIMEOUT", "runner_returncode": process.returncode,
                    "error": f"SST exceeded {args.timeout + 60} seconds"}
    result = load_result(result_path)
    summary = summarize_result(result, floor)
    numerical = load_result(numerical_path)
    mpi = load_result(mpi_path)
    layout = load_result(layout_path)
    summary.update(numerical_status=numerical.get("status"),
                   max_abs_error=numerical.get("max_abs_error"),
                   checked_elements=numerical.get("checked"),
                   mpi_status=mpi.get("status"),
                   layout_status=layout.get("status"),
                   layout_checked_bytes=layout.get("checked_bytes"))
    if (numerical.get("status") != "PASS" or mpi.get("status") != "PASS" or
            layout.get("status") != "PASS"):
        summary["status"] = "VERIFICATION_FAILED"
    summary["runner_returncode"] = returncode
    if returncode != 0 and summary["status"] == "PASS":
        summary["status"] = "RUNNER_FAILED"
    return {**base, **summary}


def command(args: argparse.Namespace, artifact: Path, hq: int, hkv: int,
            query_length: int, kv_length: int) -> list[str]:
    command_line = [
        str(RUNNER), "--artifact-root", str(artifact),
        "--query-length", str(query_length), "--kv-length", str(kv_length),
        "--num-query-heads", str(hq), "--num-kv-heads", str(hkv),
        "--head-dim", str(args.head_dim), "--timeout", str(args.timeout),
        "--dtype", args.dtype,
    ]
    if args.causal:
        command_line.append("--causal")
    return command_line


def summarize_result(result: dict, floor: int) -> dict:
    status = result.get("status", "INVALID")
    summary = {
        "status": status,
        "end_to_end_cycles": result.get("end_to_end_cycles"),
        "slowest_job_cycles": result.get("slowest_job_cycles"),
        "qk_span_cycles": result.get("stage_spans", {}).get("qk", {}).get("elapsed_cycles"),
        "sfu_span_cycles": result.get("stage_spans", {}).get("sfu", {}).get("elapsed_cycles"),
        "pv_span_cycles": result.get("stage_spans", {}).get("pv", {}).get("elapsed_cycles"),
        "pv_receive_to_start_avg_cycles": result.get("pv_transport_waits", {}).get("receive_to_start", {}).get("average_cycles"),
        "pv_receive_to_start_max_cycles": result.get("pv_transport_waits", {}).get("receive_to_start", {}).get("max_cycles"),
        "qk_jobs": result.get("qk_jobs"),
        "pv_windows": result.get("pv_windows"),
        "v_cache_hits": result.get("v_cache", {}).get("hits"),
        "v_cache_misses": result.get("v_cache", {}).get("misses"),
        "pv_average_window_cycles": result.get("pv_service", {}).get("average_window_cycles"),
        "pv_tail_after_sfu_cycles": result.get("pv_tail_after_sfu_cycles"),
        "error": result.get("error"),
    }
    actual = summary["end_to_end_cycles"]
    if isinstance(actual, (int, float)) and status == "PASS":
        summary["excess_over_floor_cycles"] = actual - floor
        summary["actual_over_floor_ratio"] = actual / floor
    else:
        summary["excess_over_floor_cycles"] = None
        summary["actual_over_floor_ratio"] = None
    return summary


FIELDS = [
    "case_id", "Hq", "Hkv", "query_length", "kv_length", "head_dim",
    "mpi_ranks", "local_gm_queue_depth", "dtype", "causal", "status",
    "numerical_status", "max_abs_error", "checked_elements", "mpi_status",
    "layout_status", "layout_checked_bytes",
    "element_library_sha256", "row_priority", "v_broadcast", "window_major", "cross_macro_prefetch",
    "theoretical_resource_floor_cycles", "end_to_end_cycles", "excess_over_floor_cycles",
    "actual_over_floor_ratio", "slowest_job_cycles", "qk_span_cycles", "sfu_span_cycles",
    "pv_span_cycles", "pv_receive_to_start_avg_cycles", "pv_receive_to_start_max_cycles",
    "qk_jobs", "pv_windows", "v_cache_hits", "v_cache_misses",
    "pv_average_window_cycles", "pv_tail_after_sfu_cycles",
    "artifact_root", "runner_returncode", "error",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/sweeps")
    parser.add_argument("--lengths", type=lambda v: parse_int_list(v, "--lengths"), default=[512, 1024, 2048, 4096])
    parser.add_argument("--kv-lengths", type=lambda v: parse_int_list(v, "--kv-lengths"))
    parser.add_argument("--pairs", type=parse_pairs, default=parse_pairs("1:1,2:1,4:2,4:1"))
    parser.add_argument("--mpi-ranks", type=int, default=4)
    parser.add_argument("--local-gm-queue-depth", type=int, default=256)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    library = ROOT / "install/lib/sst-elements-library/libgolem.so"
    args.element_library_sha256 = hashlib.sha256(library.read_bytes()).hexdigest() if library.exists() else None
    if args.head_dim not in (64, 128):
        parser.error("--head-dim must be 64 or 128")
    if args.mpi_ranks not in (1, 4):
        parser.error("--mpi-ranks must be 1 or 4 for this topology")
    if args.local_gm_queue_depth <= 0:
        parser.error("--local-gm-queue-depth must be positive")
    if args.causal and args.kv_lengths is not None:
        parser.error("--causal currently requires equal query and KV lengths; omit --kv-lengths")
    length_pairs = (
        [(length, length) for length in args.lengths]
        if args.kv_lengths is None
        else [(query_length, kv_length)
              for query_length in args.lengths
              for kv_length in args.kv_lengths]
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    previous_rows = {}
    latest_path = output_root / "summary-latest.json"
    if args.resume and latest_path.exists():
        try:
            previous_rows = {
                row["case_id"]: row for row in json.loads(latest_path.read_text())
                if row.get("status") == "PASS"
            }
        except (OSError, ValueError, KeyError, TypeError):
            previous_rows = {}
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_root / f"summary-{timestamp}.json"
    csv_path = output_root / f"summary-{timestamp}.csv"
    rows = []
    total = len(args.pairs) * len(length_pairs)
    print(f"sweep: {total} cases", flush=True)
    for index, (hq, hkv) in enumerate(args.pairs):
        for query_length, kv_length in length_pairs:
            if query_length % 256 or kv_length % 128:
                raise SystemExit("lengths must satisfy query % 256 == 0 and kv % 128 == 0")
            print(f"[{len(rows)+1}/{total}] Hq={hq} Hkv={hkv} Q={query_length} K={kv_length}", flush=True)
            row = run_case(args, output_root, hq, hkv, query_length, kv_length,
                           previous_rows.get(case_id(hq, hkv, query_length, kv_length, args.head_dim, args.dtype, args.causal)))
            rows.append(row)
            print(json.dumps({k: row.get(k) for k in ("case_id", "status", "end_to_end_cycles", "actual_over_floor_ratio")}, sort_keys=True), flush=True)
            write_summary(rows, output_root, csv_path, json_path)
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    return 0 if all(row.get("status") in {"PASS", "DRY_RUN"} for row in rows) else 1


def write_summary(rows: list[dict], output_root: Path, csv_path: Path,
                  json_path: Path) -> None:
    rendered = json.dumps(rows, indent=2) + "\n"
    (output_root / "summary-latest.json").write_text(rendered, encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "summary-latest.csv").write_bytes(csv_path.read_bytes())
    json_path.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
