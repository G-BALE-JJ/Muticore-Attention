#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import math
import statistics
import sys
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parents[1]
DEFAULT_COMPLETE = (
    TESTS_DIR
    / "artifacts/stats/sweeps/low_sysuti_iterative/tile_op_complete_workload_results_20260625.csv"
)
DEFAULT_OUT_DIR = TESTS_DIR / "artifacts/stats/analysis/selected_operator_load_balance"
EXPECTED_WORKER_CORES = list(range(4, 20))


SELECTED = [
    ("LLaMA QKV", "LLaMA-7B", "QKV", "llama_qkv_attn_out"),
    ("score", "LLaMA-7B", "score", "score"),
    ("context", "LLaMA-7B", "context", "context"),
    ("qwen_attn_out", "Qwen2-7B", "attn_out", "qwen_q_attn_out"),
    ("deepseek_router", "DeepSeek-MoE-16B", "router", "deepseek_router"),
    ("deepseek_qkv", "DeepSeek-MoE-16B", "QKV", "deepseek_qkv_attn_out"),
    ("deepseek_shared_ff_l1", "DeepSeek-MoE-16B", "shared_ff_l1", "deepseek_shared_ff_l1"),
    ("deepseek_shared_ff_l2", "DeepSeek-MoE-16B", "shared_ff_l2", "deepseek_shared_ff_l2"),
    ("deepseek_routed_ff_l1", "DeepSeek-MoE-16B", "routed_ff_l1", "deepseek_routed_ff_l1"),
    ("deepseek_routed_ff_l2", "DeepSeek-MoE-16B", "routed_ff_l2", "deepseek_routed_ff_l2"),
]


CORE_FIELDS = [
    "select_name",
    "model",
    "operator",
    "unique_label",
    "run_id",
    "tile_op_best_run_id",
    "tile_op_best_kind",
    "tile_op_best_status",
    "original_shape_m_k_n_b",
    "padded_unique_shape_m_k_n_b",
    "gemm_m",
    "gemm_n",
    "gemm_k",
    "original_b",
    "config",
    "source_note",
    "source_log",
    "worker_index",
    "core",
    "dtype",
    "dma_issue",
    "dma_wait",
    "dma_total",
    "compute",
    "compute_submit",
    "compute_wait",
    "sched_protocol",
    "c_store",
    "tile_ready_wait",
    "txn_wait",
    "writeback_wait",
    "wait_2d_activate",
    "wait_2d_active_not_ready",
    "wait_non2d_txn",
    "wait_no_active_txn",
    "window_submit_active",
    "window_submit_prefetch",
    "window_activate",
    "window_advance_wait_prefetch",
    "group_wait",
    "poll_iters",
    "overlap_issue",
    "overlap_wait",
    "issue_block_q",
    "issue_write",
    "ov_issue_block_q",
    "ov_issue_write",
    "task_desc",
    "nloop",
    "submit_pack",
    "finish_publish",
    "total",
    "start_cycle",
    "end_cycle",
    "total_cycle_B",
    "compute_cycle_B",
    "dma_wait_cycle_B",
    "tile_ready_wait_cycle_B",
    "load_balance_core_scope",
]


SUMMARY_FIELDS = [
    "select_name",
    "model",
    "operator",
    "unique_label",
    "shape_mxnxk",
    "original_shape_m_k_n_b",
    "original_b",
    "config",
    "tile_op_best_kind",
    "tile_op_best_status",
    "num_worker_records",
    "task_core_count",
    "all_expected_cores_have_task",
    "missing_worker_cores",
    "mean_total_cycle",
    "min_total_cycle",
    "p95_total_cycle",
    "max_total_cycle",
    "max_over_mean",
    "cv_total_cycle",
    "mean_total_cycle_B",
    "max_total_cycle_B",
    "source_log",
    "source_note",
]


def read_csv_rows(path: Path):
    with path.open(newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        rows = []
        for raw in reader:
            row = {}
            for k, v in raw.items():
                if k is None:
                    continue
                key = str(k).strip()
                row[key] = v.strip() if isinstance(v, str) else v
            rows.append(row)
        return rows


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_latency_module():
    path = TESTS_DIR / "stats/extract_latency_csv.py"
    spec = importlib.util.spec_from_file_location("extract_latency_csv", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def p95(values):
    if not values:
        return 0
    vals = sorted(values)
    idx = int(round(0.95 * (len(vals) - 1)))
    return vals[idx]


def to_int(value, default=0):
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def find_complete_row(rows, model, operator, unique_label):
    for row in rows:
        if (
            row.get("model") == model
            and row.get("operator") == operator
            and row.get("unique_label") == unique_label
        ):
            return row
    raise KeyError(f"missing complete row: {model}/{operator}/{unique_label}")


def find_row_by_run_id(csv_path: Path, run_id: str):
    rows = read_csv_rows(csv_path)
    for row in rows:
        if row.get("run_id", "").strip() == run_id:
            return row
        if row.get("unique_run_id", "").strip() == run_id:
            return row
    raise KeyError(f"missing run_id={run_id} in {csv_path}")


def infer_stdout_dir(log_path: Path, run_id: str):
    if not log_path:
        return None
    artifacts = None
    for parent in log_path.parents:
        if parent.name == "artifacts":
            artifacts = parent
            break
    if artifacts is None:
        return None
    stdout_root = artifacts / "stdout/overlap0"
    if not stdout_root.exists():
        return None
    direct = stdout_root / run_id
    if direct.exists():
        return direct
    stem = log_path.stem
    children = [p for p in stdout_root.iterdir() if p.is_dir()]
    for child in children:
        if child.name in stem or stem.endswith(child.name):
            return child
    if "dim_4096" in stem:
        for child in children:
            if child.name.startswith("dim_4096_"):
                return child
    return None


def resolve_from_run_summary(source_csv: Path, run_id: str):
    summary = source_csv.parent / "run_summary.csv"
    row = find_row_by_run_id(summary, run_id)
    log_path = Path(row.get("log_file", ""))
    stdout_dir = infer_stdout_dir(log_path, run_id)
    return log_path, stdout_dir, run_id


def resolve_actual_source(best_row):
    best_kind = best_row.get("tile_op_best_kind", "")
    best_status = best_row.get("tile_op_best_status", "")
    best_run_id = best_row.get("tile_op_best_run_id", "")
    best_source_path = best_row.get("tile_op_best_source_path", "")
    best_source = Path(best_source_path) if best_source_path else None

    if best_kind in ("optimized_low_sysuti_sweep", "global_tileop_measured", "baseline_covered"):
        log_path, stdout_dir, actual_run_id = resolve_from_run_summary(best_source, best_run_id)
        note = f"actual {best_kind} per-core records from best run"
        return log_path, stdout_dir, actual_run_id, note

    if best_kind == "global_tileop_estimated" or best_status.startswith("ESTIMATED"):
        baseline_rows = read_csv_rows(Path(best_row["tile_unique_source_path"]))
        match = None
        for row in baseline_rows:
            if (
                row.get("model") == best_row.get("model")
                and row.get("operator") == best_row.get("operator")
                and row.get("unique_label") == best_row.get("unique_label")
            ):
                match = row
                break
        if match is None:
            raise KeyError(
                "missing tile-unique anchor for "
                f"{best_row.get('model')}/{best_row.get('operator')}/{best_row.get('unique_label')}"
            )
        source_csv = Path(match["source_path"])
        anchor_run_id = match["source_run_id"]
        log_path, stdout_dir, actual_run_id = resolve_from_run_summary(source_csv, anchor_run_id)
        note = "tile-op best is estimated; per-core records use actual tile-unique anchor"
        return log_path, stdout_dir, actual_run_id, note

    raise ValueError(
        f"unsupported best source kind={best_kind!r} status={best_status!r} "
        f"run_id={best_run_id!r}"
    )


def parse_core_records(latency_module, log_path: Path, stdout_dir: Path | None):
    log_dir = stdout_dir if stdout_dir is not None else Path("__missing_stdout_dir__")
    inputs = latency_module.resolve_inputs(log_path, log_dir, "stdout-*")
    return latency_module.parse_logs(inputs)


def has_task(record):
    return (
        to_int(record.get("total")) > 0
        and (
            to_int(record.get("compute")) > 0
            or to_int(record.get("compute_wait")) > 0
            or to_int(record.get("task_desc")) > 0
            or to_int(record.get("nloop")) > 0
        )
    )


def build_rows(complete_path: Path):
    complete_rows = read_csv_rows(complete_path)
    latency_module = load_latency_module()
    core_rows = []
    summary_rows = []
    failures = []

    for select_name, model, operator, unique_label in SELECTED:
        best_row = find_complete_row(complete_rows, model, operator, unique_label)
        log_path, stdout_dir, actual_run_id, source_note = resolve_actual_source(best_row)
        records = parse_core_records(latency_module, log_path, stdout_dir)
        records = sorted(records, key=lambda r: to_int(r.get("core")))

        task_cores = [to_int(r.get("core")) for r in records if has_task(r)]
        missing = [c for c in EXPECTED_WORKER_CORES if c not in task_cores]
        all_have_task = len(missing) == 0 and len(task_cores) == len(EXPECTED_WORKER_CORES)
        if not all_have_task:
            failures.append((select_name, task_cores, missing, str(log_path)))
        scope = (
            "all_16_worker_cores_with_nonzero_task"
            if all_have_task
            else "partial_worker_cores_with_latency_records"
        )

        b = to_int(best_row.get("original_b"), 1)
        totals = [to_int(r.get("total")) for r in records if has_task(r)]
        shape = f"{best_row.get('runtime_gemm_m')}x{best_row.get('runtime_gemm_n')}x{best_row.get('runtime_gemm_k')}"
        mean_total = statistics.mean(totals) if totals else 0.0
        min_total = min(totals) if totals else 0
        max_total = max(totals) if totals else 0
        p95_total = p95(totals)
        cv = statistics.pstdev(totals) / mean_total if len(totals) > 1 and mean_total else 0.0
        note = f"{source_note}; best={best_row.get('tile_op_best_params', '')}"

        for worker_index, rec in enumerate(records):
            out = {
                "select_name": select_name,
                "model": model,
                "operator": operator,
                "unique_label": unique_label,
                "run_id": actual_run_id,
                "tile_op_best_run_id": best_row.get("tile_op_best_run_id", ""),
                "tile_op_best_kind": best_row.get("tile_op_best_kind", ""),
                "tile_op_best_status": best_row.get("tile_op_best_status", ""),
                "original_shape_m_k_n_b": best_row.get("original_shape_m_k_n_b", ""),
                "padded_unique_shape_m_k_n_b": best_row.get("padded_unique_shape_m_k_n_b", ""),
                "gemm_m": best_row.get("runtime_gemm_m", ""),
                "gemm_n": best_row.get("runtime_gemm_n", ""),
                "gemm_k": best_row.get("runtime_gemm_k", ""),
                "original_b": best_row.get("original_b", ""),
                "config": best_row.get("tile_op_best_params", ""),
                "source_note": note,
                "source_log": str(log_path),
                "worker_index": worker_index,
                "load_balance_core_scope": scope,
            }
            for field in CORE_FIELDS:
                if field in out:
                    continue
                value = rec.get(field, "")
                out[field] = value
            out["total_cycle_B"] = to_int(rec.get("total")) * b
            out["compute_cycle_B"] = to_int(rec.get("compute")) * b
            out["dma_wait_cycle_B"] = to_int(rec.get("dma_wait")) * b
            out["tile_ready_wait_cycle_B"] = to_int(rec.get("tile_ready_wait")) * b
            core_rows.append(out)

        summary_rows.append(
            {
                "select_name": select_name,
                "model": model,
                "operator": operator,
                "unique_label": unique_label,
                "shape_mxnxk": shape,
                "original_shape_m_k_n_b": best_row.get("original_shape_m_k_n_b", ""),
                "original_b": best_row.get("original_b", ""),
                "config": best_row.get("tile_op_best_params", ""),
                "tile_op_best_kind": best_row.get("tile_op_best_kind", ""),
                "tile_op_best_status": best_row.get("tile_op_best_status", ""),
                "num_worker_records": len(records),
                "task_core_count": len(task_cores),
                "all_expected_cores_have_task": "yes" if all_have_task else "no",
                "missing_worker_cores": "none" if not missing else " ".join(map(str, missing)),
                "mean_total_cycle": mean_total,
                "min_total_cycle": min_total,
                "p95_total_cycle": p95_total,
                "max_total_cycle": max_total,
                "max_over_mean": (max_total / mean_total) if mean_total else math.nan,
                "cv_total_cycle": cv,
                "mean_total_cycle_B": mean_total * b,
                "max_total_cycle_B": max_total * b,
                "source_log": str(log_path),
                "source_note": note,
            }
        )

    return core_rows, summary_rows, failures


def write_markdown(path: Path, summary_rows):
    lines = [
        "# Selected Operator Per-Core Load Balance Summary",
        "",
        "Source: latest tile-op complete table. `tile_op_best_kind=baseline_covered` means the inclusive tile-op best is the tile-unique baseline. Estimated rows use an actual tile-unique anchor for per-core records.",
        "",
        "| op | shape | B | config | workers | task cores | all cores have task | mean total | min | p95 | max | max/mean | CV |",
        "|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {op} | {shape} | {b} | {config} | {workers} | {task_cores} | {ok} | "
            "{mean:.1f} | {minv} | {p95v} | {maxv} | {ratio:.4f} | {cv:.4f} |".format(
                op=row["select_name"],
                shape=row["shape_mxnxk"],
                b=row["original_b"],
                config=row["config"],
                workers=row["num_worker_records"],
                task_cores=row["task_core_count"],
                ok=row["all_expected_cores_have_task"],
                mean=float(row["mean_total_cycle"]),
                minv=row["min_total_cycle"],
                p95v=row["p95_total_cycle"],
                maxv=row["max_total_cycle"],
                ratio=float(row["max_over_mean"]),
                cv=float(row["cv_total_cycle"]),
            )
        )
    lines.append("")
    lines.append("## Source Notes")
    lines.append("")
    for row in summary_rows:
        lines.append(f"- `{row['select_name']}`: {row['source_note']}")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Refresh selected-operator per-core load-balance CSV/MD from latest tile-op best data."
    )
    parser.add_argument("--complete", default=str(DEFAULT_COMPLETE))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    complete_path = Path(args.complete)
    out_dir = Path(args.out_dir)
    core_rows, summary_rows, failures = build_rows(complete_path)

    if failures:
        for name, task_cores, missing, log_path in failures:
            print(
                f"[ERROR] {name}: task_cores={task_cores} missing={missing} log={log_path}",
                file=sys.stderr,
            )
        raise SystemExit(1)

    cycles_csv = out_dir / "selected_operator_per_core_cycles.csv"
    summary_csv = out_dir / "selected_operator_per_core_load_balance_summary.csv"
    summary_md = out_dir / "selected_operator_per_core_load_balance_summary.md"
    write_csv(cycles_csv, core_rows, CORE_FIELDS)
    write_csv(summary_csv, summary_rows, SUMMARY_FIELDS)
    write_markdown(summary_md, summary_rows)

    print(f"[OK] wrote {cycles_csv}")
    print(f"[OK] wrote {summary_csv}")
    print(f"[OK] wrote {summary_md}")
    print(f"[OK] selected_ops={len(summary_rows)} per_core_rows={len(core_rows)}")


if __name__ == "__main__":
    main()
