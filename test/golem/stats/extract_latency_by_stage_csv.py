#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


MILESTONE_RE = re.compile(
    r"\[MILESTONE\]\s+stage=(?P<stage>[A-Za-z0-9_]+)\s+status=(?P<status>start|done|fail)\s+cycle=(?P<cycle>\d+)"
)


def resolve_inputs(log_path: Path | None, log_dir: Path, stdout_glob: str):
    inputs = []
    if log_path is not None and log_path.exists() and log_path.is_file():
        inputs.append(log_path)
    if log_dir.exists() and log_dir.is_dir():
        inputs.extend(sorted(p for p in log_dir.glob(stdout_glob) if p.is_file()))
    return inputs


def parse_milestones(log_paths):
    items = []
    for p in log_paths:
        for line in p.read_text(errors="ignore").splitlines():
            m = MILESTONE_RE.search(line)
            if not m:
                continue
            items.append(
                {
                    "stage": m.group("stage"),
                    "status": m.group("status"),
                    "cycle": int(m.group("cycle")),
                }
            )
    return items


def build_stage_records(items):
    starts = {}
    records = []
    for it in items:
        stage = it["stage"]
        status = it["status"]
        cyc = it["cycle"]
        if status == "start":
            starts[stage] = cyc
            continue
        if status in {"done", "fail"} and stage in starts:
            st = starts.pop(stage)
            records.append(
                {
                    "stage": stage,
                    "status": status,
                    "start_cycle": st,
                    "end_cycle": cyc,
                    "duration_cycles": max(0, cyc - st),
                }
            )
    return records


def inject_fixed_conv1_im2col(records, conv1_im2col_ms: float, clock_ghz: float):
    if conv1_im2col_ms <= 0:
        return records
    cyc = int(round(conv1_im2col_ms * 1_000_000.0 * clock_ghz))
    return [
        {
            "stage": "conv1_im2col",
            "status": "fixed",
            "start_cycle": 0,
            "end_cycle": cyc,
            "duration_cycles": cyc,
        }
    ] + records


def expected_stages_from_plan(plan_file: Path):
    if not plan_file.exists():
        return []
    try:
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
    except Exception:
        return []
    flow = plan.get("stage_flow", [])
    out = []
    for st in flow:
        if isinstance(st, dict) and st.get("latency_track", False):
            name = st.get("name")
            if isinstance(name, str) and name:
                out.append(name)
    return out


def write_breakdown(records, out_path: Path, clock_ghz: float):
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "stage",
                "status",
                "start_cycle",
                "end_cycle",
                "duration_cycles",
                "duration_ms",
            ],
        )
        w.writeheader()
        for r in records:
            rr = dict(r)
            rr["duration_ms"] = (
                f"{(float(r['duration_cycles']) / (clock_ghz * 1_000_000.0)):.6f}"
            )
            w.writerow(rr)


def write_summary(records, out_path: Path, clock_ghz: float):
    total = sum(int(r["duration_cycles"]) for r in records)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["stage_count", len(records)])
        w.writerow(["total_stage_cycles", total])
        w.writerow(
            ["total_stage_ms", f"{(float(total) / (clock_ghz * 1_000_000.0)):.6f}"]
        )
        for r in records:
            w.writerow([f"stage_{r['stage']}_cycles", r["duration_cycles"]])
            w.writerow(
                [
                    f"stage_{r['stage']}_ms",
                    f"{(float(r['duration_cycles']) / (clock_ghz * 1_000_000.0)):.6f}",
                ]
            )


def main():
    p = argparse.ArgumentParser(description="Extract stage latency from milestone logs")
    p.add_argument("--log", required=True, help="Main simulation log path")
    p.add_argument(
        "--log-dir",
        default=".",
        help="Directory to scan for sharded stdout logs (default: current dir)",
    )
    p.add_argument(
        "--stdout-glob",
        default="stdout-*",
        help="Glob for sharded stdout logs (default: stdout-*)",
    )
    p.add_argument(
        "--out",
        default="latency_by_stage_breakdown.csv",
        help="Output stage breakdown CSV",
    )
    p.add_argument(
        "--summary",
        default="latency_by_stage_summary.csv",
        help="Output stage summary CSV",
    )
    p.add_argument(
        "--conv1-im2col-ms",
        type=float,
        default=0.023,
        help="Fixed conv1_im2col duration in ms (default: 0.023)",
    )
    p.add_argument(
        "--clock-ghz",
        type=float,
        default=1.0,
        help="Clock in GHz for cycles<->ms conversion (default: 1.0)",
    )
    p.add_argument(
        "--plan-file",
        default="",
        help="Optional plan file to validate stage completeness",
    )
    p.add_argument(
        "--strict-stage-check",
        type=int,
        default=0,
        help="If 1, exit non-zero when plan-tracked stages are missing",
    )
    args = p.parse_args()

    log_path = Path(args.log)
    out_path = Path(args.out)
    sum_path = Path(args.summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sum_path.parent.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir)
    input_logs = resolve_inputs(log_path, log_dir, args.stdout_glob)
    if not input_logs:
        raise SystemExit(
            f"[ERROR] no input logs found (log={log_path}, dir={log_dir}, glob={args.stdout_glob})"
        )

    items = parse_milestones(input_logs)
    recs = build_stage_records(items)
    allowed = {
        "conv1_gemm",
        "conv1_relu",
        "pool1",
        "conv2_im2col",
        "conv2_gemm",
        "conv2_relu",
        "pool2",
        "fc1",
        "fc2",
        "fc3",
    }
    recs = [r for r in recs if r["stage"] in allowed]
    recs = inject_fixed_conv1_im2col(recs, args.conv1_im2col_ms, args.clock_ghz)
    if not recs:
        write_breakdown([], out_path, args.clock_ghz)
        write_summary([], sum_path, args.clock_ghz)
        print(
            "[WARN] no milestone stage records found; wrote empty stage CSV templates"
        )
        print("[INFO] searched logs:")
        for p in input_logs:
            print(f"  - {p}")
        print(f"[OK] wrote {out_path}")
        print(f"[OK] wrote {sum_path}")
        return

    write_breakdown(recs, out_path, args.clock_ghz)
    write_summary(recs, sum_path, args.clock_ghz)

    expected = []
    if args.plan_file:
        expected = expected_stages_from_plan(Path(args.plan_file))
    actual = {r["stage"] for r in recs}
    missing = [s for s in expected if s not in actual]

    print("[OK] parsed logs:")
    for p in input_logs:
        print(f"  - {p}")
    print(f"[OK] wrote {out_path}")
    print(f"[OK] wrote {sum_path}")
    print(f"[OK] stages: {len(recs)}")
    if expected:
        if missing:
            print(f"[WARN] stage completeness check missing={missing}")
            if int(args.strict_stage_check) != 0:
                raise SystemExit(2)
        else:
            print("[OK] stage completeness check passed")


if __name__ == "__main__":
    main()
