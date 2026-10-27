#!/usr/bin/env python3

import argparse
import csv
import dataclasses
import math
import pathlib
import re
import subprocess
import sys


ROWS = 16
DIM = 512
EXPECTED_WORKERS = (4, 8, 16)
ROOT_SPECS = (
    ("sfu_phase4e_modeled_control_matrix_20260715", "modeled-NoC", None),
    ("sfu_phase4e_explicit_vn0_matrix_20260715", "explicit-NoC", 0),
    ("sfu_phase4e_explicit_vn1_matrix_20260715", "explicit-NoC", 1),
    ("sfu_phase4e_explicit_vn2_matrix_20260715", "explicit-NoC", 2),
)
SOURCE_CSV_NAME = "sfu_phase4e_group_figure_source_data.csv"


@dataclasses.dataclass(frozen=True)
class PointRecord:
    transport: str
    reduction_vn: int | None
    workers: int
    band_cores: int
    simulated_time_us: float
    transport_events: int
    latency_average_cycles: float
    latency_max_cycles: int
    inbox_high_water: int
    queued: int
    rejected: int
    stale: int
    total_send_packets: int
    total_send_bits: int
    total_xbar_stalls: int
    dma_timeout_retry: int
    dma_timeout_exhausted: int
    dma_write_timeout_retry: int
    golden_checked: int
    golden_mismatches: int
    source_root: str
    run_id: str


@dataclasses.dataclass(frozen=True)
class DerivedContracts:
    explicit_runtime_spread: float
    max_explicit_modeled_pct: float
    latency_growth_pct: float
    transport_events: tuple[int, int, int]
    inbox_high_water: int
    golden_checked: int
    golden_mismatches: int
    queued: int
    rejected: int
    stale: int
    dma_timeout_retry: int
    dma_timeout_exhausted: int
    dma_write_timeout_retry: int


def _load_plotting_backend():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "svg.fonttype": "none",
        "svg.hashsalt": "sfu-phase4e-group-figure-20260715",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    import matplotlib.pyplot as plt

    return plt


def _error(root: pathlib.Path, run_id: str, field: str, detail: str) -> ValueError:
    return ValueError(f"root={root} run_id={run_id or '<manifest>'} field={field}: {detail}")


def _required_int(root: pathlib.Path, row: dict[str, str], field: str) -> int:
    run_id = row.get("run_id", "")
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise _error(root, run_id, field, f"expected integer, got {row.get(field)!r}") from exc


def select_manifest_rows(
    root: pathlib.Path, expected_transport: str, expected_vn: int | None
) -> list[dict[str, str]]:
    expected_names = {
        (transport, vn): name for name, transport, vn in ROOT_SPECS
    }
    expected_name = expected_names.get((expected_transport, expected_vn))
    if expected_name is None:
        raise _error(root, "", "transport", f"unsupported transport/VN {expected_transport}/{expected_vn}")
    if root.name != expected_name:
        raise _error(
            root,
            "",
            "root",
            f"VN/root mismatch: expected {expected_name} for {expected_transport} VN={expected_vn}",
        )

    manifest = root / "sweep_manifest.csv"
    try:
        with manifest.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, strict=True)
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise _error(root, "", "sweep_manifest.csv", str(exc)) from exc
    if not rows:
        raise _error(root, "", "sweep_manifest.csv", "manifest is empty")
    if reader.fieldnames is None or any(
        None in row or any(value is None for value in row.values()) for row in rows
    ):
        raise _error(root, "", "sweep_manifest.csv", "malformed CSV structure")

    observed_workers: set[int] = set()
    by_run_id: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        run_id = row.get("run_id", "")
        if not run_id:
            raise _error(root, "", "run_id", "missing value")
        workers = _required_int(root, row, "worker_cores")
        observed_workers.add(workers)
        if workers not in EXPECTED_WORKERS:
            raise _error(root, run_id, "worker_cores", f"unexpected worker count {workers}")
        if _required_int(root, row, "rows") != ROWS:
            raise _error(root, run_id, "rows", f"expected {ROWS}")
        if _required_int(root, row, "dim") != DIM:
            raise _error(root, run_id, "dim", f"expected {DIM}")
        if _required_int(root, row, "band_cores") != workers:
            raise _error(root, run_id, "band_cores", f"expected {workers}")
        if expected_transport == "explicit-NoC":
            row_vn = _required_int(root, row, "reduction_vn")
            if row_vn != expected_vn or not run_id.endswith(f"_vn{expected_vn}"):
                raise _error(
                    root,
                    run_id,
                    "reduction_vn",
                    f"VN/root mismatch: expected VN {expected_vn}, got {row_vn}",
                )
        by_run_id.setdefault(run_id, []).append(row)

    if observed_workers != set(EXPECTED_WORKERS):
        raise _error(
            root,
            "",
            "worker_cores",
            f"expected worker points {list(EXPECTED_WORKERS)}, got {sorted(observed_workers)}",
        )

    selected: list[dict[str, str]] = []
    for run_id, run_rows in by_run_id.items():
        canonical = []
        for row in run_rows:
            status = row.get("status")
            artifact = row.get("artifact_validation")
            if status != "PASS":
                raise _error(root, run_id, "status", f"expected PASS, got {status!r}")
            if artifact == "PASS":
                canonical.append(row)
            elif artifact != "CACHED":
                raise _error(
                    root,
                    run_id,
                    "artifact_validation",
                    f"expected PASS or ignorable CACHED, got {artifact!r}",
                )
        if len(canonical) != 1:
            detail = "duplicate PASS/PASS rows" if len(canonical) > 1 else "missing PASS/PASS row"
            raise _error(root, run_id, "artifact_validation", detail)
        row = canonical[0]
        if row.get("exit_code") != "0":
            raise _error(root, run_id, "exit_code", f"expected 0, got {row.get('exit_code')!r}")
        selected.append(row)

    selected.sort(key=lambda row: _required_int(root, row, "worker_cores"))
    if len(selected) != len(EXPECTED_WORKERS):
        raise _error(root, "", "run_id", f"expected 3 canonical rows, got {len(selected)}")
    return selected


def _read_stats(
    root: pathlib.Path,
    run_id: str,
    path: pathlib.Path,
    required_fields: tuple[str, ...],
) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, strict=True)
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise _error(root, run_id, path.name, str(exc)) from exc
    if not rows:
        raise _error(root, run_id, path.name, "CSV evidence is empty")
    fieldnames = reader.fieldnames or []
    missing = [field for field in required_fields if field not in fieldnames]
    if missing:
        raise _error(root, run_id, path.name, f"missing columns {missing}")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise _error(root, run_id, path.name, "malformed CSV structure")
    return rows


def _stat_values(
    root: pathlib.Path, run_id: str, rows: list[dict[str, str]], statistic: str, field: str
) -> list[int]:
    matching = [row for row in rows if row.get("StatisticName") == statistic]
    if not matching:
        raise _error(root, run_id, statistic, "statistic is missing")
    values = []
    for row in matching:
        try:
            values.append(int(row[field]))
        except (KeyError, TypeError, ValueError) as exc:
            raise _error(root, run_id, f"{statistic}.{field}", f"invalid value {row.get(field)!r}") from exc
    return values


def _metric_value(
    root: pathlib.Path,
    run_id: str,
    path: pathlib.Path,
    metric: str,
    value_field: str,
) -> int:
    rows = _read_stats(root, run_id, path, ("metric", value_field))
    matches = [row for row in rows if row.get("metric") == metric]
    if len(matches) != 1:
        raise _error(root, run_id, metric, f"expected one row in {path.name}, got {len(matches)}")
    try:
        return int(matches[0][value_field])
    except (KeyError, TypeError, ValueError) as exc:
        raise _error(
            root, run_id, f"{metric}.{value_field}", f"invalid value {matches[0].get(value_field)!r}"
        ) from exc


def _simulated_time_us(root: pathlib.Path, run_id: str, log: pathlib.Path) -> float:
    try:
        text = log.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise _error(root, run_id, "simulated_time", str(exc)) from exc
    matches = list(re.finditer(
        r"Simulation is complete, simulated time:\s*([0-9]+(?:\.[0-9]+)?)\s*(us|ms|s)\b",
        text,
    ))
    if len(matches) != 1:
        raise _error(
            root,
            run_id,
            "simulated_time",
            f"expected exactly one valid completion line, got {len(matches)}",
        )
    match = matches[0]
    multipliers = {"us": 1.0, "ms": 1_000.0, "s": 1_000_000.0}
    return float(match.group(1)) * multipliers[match.group(2)]


def _golden_evidence(
    root: pathlib.Path, run_id: str, output_file: pathlib.Path, verifier: pathlib.Path
) -> tuple[int, int]:
    required = [
        verifier,
        root / "inputs" / "a.bin",
        root / "inputs" / "b.bin",
        output_file,
        root / "inputs" / "softmax_logits_16x512.bin",
    ]
    for path in required:
        if not path.is_file():
            raise _error(root, run_id, "golden_input", f"missing {path}")
    command = [
        sys.executable,
        str(verifier),
        "--a-file",
        str(root / "inputs" / "a.bin"),
        "--b-file",
        str(root / "inputs" / "b.bin"),
        "--c-file",
        str(output_file),
        "--m",
        "16",
        "--n",
        "512",
        "--k",
        "512",
        "--block-m",
        "4",
        "--block-n",
        "64",
        "--dtype",
        "fp32",
        "--reference",
        "logits",
        "--logits-file",
        str(root / "inputs" / "softmax_logits_16x512.bin"),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise _error(root, run_id, "verifier", f"exit {result.returncode}: {result.stdout.strip()}")
    evidence = re.search(
        r"^\[VERIFY-SFU-SOFTMAX\]\s+(PASS|FAIL).*\bchecked=(\d+)\b.*\bmismatches=(\d+)\b",
        result.stdout,
        flags=re.MULTILINE,
    )
    if evidence is None:
        raise _error(root, run_id, "verifier_output", "successful PASS evidence line is missing")
    checked = int(evidence.group(2))
    mismatches = int(evidence.group(3))
    if checked != ROWS * DIM:
        raise _error(root, run_id, "checked", f"expected {ROWS * DIM}, got {checked}")
    if mismatches != 0:
        raise _error(root, run_id, "mismatches", f"expected 0, got {mismatches}")
    if evidence.group(1) != "PASS":
        raise _error(root, run_id, "verifier_output", "verifier did not report PASS")
    return checked, mismatches


def parse_point(
    root: pathlib.Path,
    manifest_row: dict[str, str],
    transport: str,
    reduction_vn: int | None,
    verifier: pathlib.Path,
) -> PointRecord:
    run_id = manifest_row.get("run_id", "")
    if not run_id:
        raise _error(root, "", "run_id", "missing value")
    workers = _required_int(root, manifest_row, "worker_cores")
    band_cores = _required_int(root, manifest_row, "band_cores")
    stats_dir = root / "stats" / "overlap0" / run_id
    stats_file = stats_dir / "stats_selfcom.txt"
    matching_logs = sorted((root / "logs").glob(f"*{run_id}*.log"))
    if len(matching_logs) != 1:
        raise _error(
            root,
            run_id,
            "sst_log",
            f"expected exactly one matching log, got {len(matching_logs)}",
        )
    sst_log = matching_logs[0]
    output_file = root / "outputs" / f"{run_id}.bin"

    stats_rows = _read_stats(
        root,
        run_id,
        stats_file,
        ("StatisticName", "Sum.u64", "Count.u64", "Max.u64"),
    )
    for stats_row in stats_rows:
        for field in ("Sum.u64", "Count.u64", "Max.u64"):
            try:
                int(stats_row[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise _error(
                    root,
                    run_id,
                    f"{stats_file.name}.{field}",
                    f"invalid value {stats_row.get(field)!r}",
                ) from exc
    noc_file = stats_dir / "noc_summary.csv"
    dma_file = stats_dir / "dma_summary.csv"
    total_send_packets = _metric_value(root, run_id, noc_file, "total_send_packets", "value")
    total_send_bits = _metric_value(root, run_id, noc_file, "total_send_bits", "value")
    total_xbar_stalls = _metric_value(root, run_id, noc_file, "total_xbar_stalls", "value")
    dma_timeout_retry = _metric_value(root, run_id, dma_file, "timeout_retry", "sum")
    dma_timeout_exhausted = _metric_value(root, run_id, dma_file, "timeout_exhausted", "sum")
    dma_write_timeout_retry = _metric_value(root, run_id, dma_file, "write_timeout_retry", "sum")

    if transport == "explicit-NoC":
        transport_events = sum(
            _stat_values(root, run_id, stats_rows, "sfu_reduction_transport_received", "Sum.u64")
        )
        latency_sum = sum(
            _stat_values(root, run_id, stats_rows, "sfu_reduction_transport_latency_cycles", "Sum.u64")
        )
        latency_count = sum(
            _stat_values(root, run_id, stats_rows, "sfu_reduction_transport_latency_cycles", "Count.u64")
        )
        if latency_count == 0:
            raise _error(root, run_id, "latency_average_cycles", "zero sample count")
        latency_average_cycles = latency_sum / latency_count
        latency_max_cycles = max(
            _stat_values(root, run_id, stats_rows, "sfu_reduction_transport_latency_cycles", "Max.u64")
        )
        inbox_high_water = max(
            _stat_values(root, run_id, stats_rows, "sfu_reduction_transport_inbox_high_water", "Sum.u64")
        )
        stale = sum(
            _stat_values(root, run_id, stats_rows, "sfu_reduction_transport_stale_dropped", "Sum.u64")
        )
        queued = sum(
            _stat_values(root, run_id, stats_rows, "gmem_reduction_send_queued", "Sum.u64")
        )
        rejected = sum(
            _stat_values(root, run_id, stats_rows, "gmem_reduction_send_rejected", "Sum.u64")
        )
    elif transport == "modeled-NoC":
        transport_events = 0
        latency_average_cycles = 0.0
        latency_max_cycles = 0
        inbox_high_water = 0
        stale = 0
        queued = 0
        rejected = 0
    else:
        raise _error(root, run_id, "transport", f"unexpected value {transport!r}")

    golden_checked, golden_mismatches = _golden_evidence(
        root, run_id, output_file, verifier
    )
    return PointRecord(
        transport=transport,
        reduction_vn=reduction_vn,
        workers=workers,
        band_cores=band_cores,
        simulated_time_us=_simulated_time_us(root, run_id, sst_log),
        transport_events=transport_events,
        latency_average_cycles=latency_average_cycles,
        latency_max_cycles=latency_max_cycles,
        inbox_high_water=inbox_high_water,
        queued=queued,
        rejected=rejected,
        stale=stale,
        total_send_packets=total_send_packets,
        total_send_bits=total_send_bits,
        total_xbar_stalls=total_xbar_stalls,
        dma_timeout_retry=dma_timeout_retry,
        dma_timeout_exhausted=dma_timeout_exhausted,
        dma_write_timeout_retry=dma_write_timeout_retry,
        golden_checked=golden_checked,
        golden_mismatches=golden_mismatches,
        source_root=str(root),
        run_id=run_id,
    )


def load_and_validate_matrix(
    sweeps_root: pathlib.Path, verifier: pathlib.Path
) -> list[PointRecord]:
    records = []
    for root_name, transport, vn in ROOT_SPECS:
        root = sweeps_root / root_name
        selected = select_manifest_rows(root, transport, vn)
        records.extend(parse_point(root, row, transport, vn, verifier) for row in selected)
    if len(records) != 12:
        raise _error(sweeps_root, "", "record_count", f"expected 12, got {len(records)}")

    combinations = {(record.transport, record.reduction_vn, record.workers) for record in records}
    expected = {
        (transport, vn, workers)
        for _, transport, vn in ROOT_SPECS
        for workers in EXPECTED_WORKERS
    }
    if combinations != expected:
        raise _error(sweeps_root, "", "matrix", f"unexpected combinations {sorted(combinations, key=str)}")
    for record in records:
        if record.transport == "explicit-NoC":
            expected_events = 4 * ROWS * record.workers
            if record.transport_events != expected_events:
                raise _error(
                    pathlib.Path(record.source_root),
                    record.run_id,
                    "transport_events",
                    f"expected {expected_events}, got {record.transport_events}",
                )
        zero_fields = {
            "dma_timeout_retry": record.dma_timeout_retry,
            "dma_timeout_exhausted": record.dma_timeout_exhausted,
            "dma_write_timeout_retry": record.dma_write_timeout_retry,
            "queued": record.queued,
            "rejected": record.rejected,
            "stale": record.stale,
        }
        for field, value in zero_fields.items():
            if value != 0:
                raise _error(pathlib.Path(record.source_root), record.run_id, field, f"expected 0, got {value}")
    return records


def _ordered_records(records: list[PointRecord]) -> list[PointRecord]:
    def key(record: PointRecord) -> tuple[int, int]:
        series = 0 if record.transport == "modeled-NoC" else 1 + int(record.reduction_vn)
        return series, record.workers

    return sorted(records, key=key)


def write_source_csv(records: list[PointRecord], output_csv: pathlib.Path) -> None:
    point_fields = [field.name for field in dataclasses.fields(PointRecord)]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=point_fields + ["single_simulation_outcome"],
            lineterminator="\n",
        )
        writer.writeheader()
        for record in _ordered_records(records):
            row = {}
            for field in point_fields:
                value = getattr(record, field)
                if value is None:
                    row[field] = ""
                elif isinstance(value, float):
                    row[field] = format(value, ".17g")
                else:
                    row[field] = str(value)
            row["single_simulation_outcome"] = "1"
            writer.writerow(row)


def load_source_csv(source_csv: pathlib.Path) -> list[PointRecord]:
    point_fields = [field.name for field in dataclasses.fields(PointRecord)]
    expected_fields = point_fields + ["single_simulation_outcome"]
    try:
        with source_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, strict=True)
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"source_csv={source_csv}: {exc}") from exc
    if reader.fieldnames != expected_fields:
        raise ValueError(
            f"source_csv={source_csv}: expected columns {expected_fields}, got {reader.fieldnames}"
        )

    float_fields = {"simulated_time_us", "latency_average_cycles"}
    string_fields = {"transport", "source_root", "run_id"}
    restored = []
    for row_number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(
                f"source_csv={source_csv} row={row_number}: malformed CSV structure"
            )
        if row.get("single_simulation_outcome") != "1":
            raise ValueError(
                f"source_csv={source_csv} row={row_number}: single_simulation_outcome must be 1"
            )
        values = {}
        try:
            for field in point_fields:
                value = row[field]
                if field in float_fields:
                    values[field] = float(value)
                    if not math.isfinite(values[field]):
                        raise ValueError(f"non-finite {field}")
                elif field in string_fields:
                    values[field] = value
                elif field == "reduction_vn":
                    values[field] = None if value == "" else int(value)
                else:
                    values[field] = int(value)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"source_csv={source_csv} row={row_number}: invalid PointRecord value: {exc}"
            ) from exc
        restored.append(PointRecord(**values))
    return restored


def validate_derived_contracts(records: list[PointRecord]) -> DerivedContracts:
    modeled = {
        record.workers: record.simulated_time_us
        for record in records
        if record.transport == "modeled-NoC"
    }
    explicit = [record for record in records if record.transport == "explicit-NoC"]
    spreads = []
    explicit_modeled_pct = []
    latency_by_workers = {}
    transport_events = []
    for workers in EXPECTED_WORKERS:
        worker_records = [record for record in explicit if record.workers == workers]
        if len(worker_records) != 3 or workers not in modeled:
            raise ValueError(f"derived contracts: incomplete worker point {workers}")
        vn_identities = [record.reduction_vn for record in worker_records]
        if set(vn_identities) != {0, 1, 2}:
            raise ValueError(
                "derived contracts: VN identities "
                f"at workers={workers}: expected [0, 1, 2], got {vn_identities}"
            )
        expected_events = 4 * ROWS * workers
        for record in worker_records:
            if record.inbox_high_water != 4:
                raise ValueError(
                    "derived contracts: inbox high-water "
                    f"at workers={workers} VN={record.reduction_vn}: "
                    f"expected 4, got {record.inbox_high_water}"
                )
            if record.transport_events != expected_events:
                raise ValueError(
                    "derived contracts: transport events "
                    f"at workers={workers} VN={record.reduction_vn}: "
                    f"expected {expected_events}, got {record.transport_events}"
                )
        equality_fields = (
            ("simulated_time_us", "runtime"),
            ("latency_average_cycles", "average latency"),
            ("latency_max_cycles", "maximum latency"),
            ("inbox_high_water", "inbox high-water"),
            ("total_xbar_stalls", "xbar stalls"),
            ("transport_events", "transport events"),
        )
        for field, label in equality_fields:
            values = [getattr(record, field) for record in worker_records]
            if any(value != values[0] for value in values[1:]):
                raise ValueError(
                    f"derived contracts: VN {label} mismatch at workers={workers}"
                )
        runtimes = [record.simulated_time_us for record in worker_records]
        spread = max(runtimes) - min(runtimes)
        spreads.append(spread)
        latencies = [record.latency_average_cycles for record in worker_records]
        if modeled[workers] <= 0.0:
            raise ValueError(f"derived contracts: nonpositive modeled runtime at workers={workers}")
        explicit_modeled_pct.extend(
            abs(runtime - modeled[workers]) / modeled[workers] * 100.0
            for runtime in runtimes
        )
        latency_by_workers[workers] = latencies[0]
        transport_events.append(worker_records[0].transport_events)

    evidence_fields = (
        ("golden_checked", ROWS * DIM),
        ("golden_mismatches", 0),
        ("queued", 0),
        ("rejected", 0),
        ("stale", 0),
        ("dma_timeout_retry", 0),
        ("dma_timeout_exhausted", 0),
        ("dma_write_timeout_retry", 0),
    )
    for field, expected in evidence_fields:
        for record in records:
            value = getattr(record, field)
            if value != expected:
                raise ValueError(
                    f"derived contracts: {field} expected {expected}, got {value} "
                    f"for run_id={record.run_id}"
                )

    max_pct = max(explicit_modeled_pct)
    if max_pct >= 0.061:
        raise ValueError(
            f"derived contracts: max explicit/modeled difference {max_pct}% is not < 0.061%"
        )
    if latency_by_workers[4] <= 0.0:
        raise ValueError("derived contracts: nonpositive four-worker latency")
    latency_growth_pct = (
        latency_by_workers[16] / latency_by_workers[4] - 1.0
    ) * 100.0
    if round(latency_growth_pct) != 65:
        raise ValueError(
            f"derived contracts: latency growth rounds to {round(latency_growth_pct)}%, expected 65%"
        )
    return DerivedContracts(
        explicit_runtime_spread=max(spreads),
        max_explicit_modeled_pct=max_pct,
        latency_growth_pct=latency_growth_pct,
        transport_events=tuple(transport_events),
        inbox_high_water=explicit[0].inbox_high_water,
        golden_checked=records[0].golden_checked,
        golden_mismatches=records[0].golden_mismatches,
        queued=records[0].queued,
        rejected=records[0].rejected,
        stale=records[0].stale,
        dma_timeout_retry=records[0].dma_timeout_retry,
        dma_timeout_exhausted=records[0].dma_timeout_exhausted,
        dma_write_timeout_retry=records[0].dma_write_timeout_retry,
    )


def _validation_annotations(derived: DerivedContracts) -> tuple[str, str]:
    transport_text = "Transport events: " + " / ".join(
        str(value) for value in derived.transport_events
    )
    validation_text = (
        f"Inbox high-water = {derived.inbox_high_water}\n"
        "Queued / rejected / stale = "
        f"{derived.queued} / {derived.rejected} / {derived.stale}\n"
        f"Golden = {derived.golden_checked} checked, "
        f"{derived.golden_mismatches} mismatches (all points)\n"
        "DMA retry / exhaustion / write retry = "
        f"{derived.dma_timeout_retry} / {derived.dma_timeout_exhausted} / "
        f"{derived.dma_write_timeout_retry}"
    )
    return transport_text, validation_text


def _records_by_worker(
    records: list[PointRecord], transport: str, reduction_vn: int | None
) -> dict[int, PointRecord]:
    selected = {
        record.workers: record
        for record in records
        if record.transport == transport and record.reduction_vn == reduction_vn
    }
    if tuple(sorted(selected)) != EXPECTED_WORKERS:
        raise ValueError(
            f"figure rendering: incomplete {transport} VN={reduction_vn} records"
        )
    return selected


def _panel_label(axis, label: str) -> None:
    axis.text(
        -0.04,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=15,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def build_figure(records: list[PointRecord]):
    derived = validate_derived_contracts(records)
    plt = _load_plotting_backend()
    workers = list(EXPECTED_WORKERS)
    modeled = _records_by_worker(records, "modeled-NoC", None)
    explicit = {
        vn: _records_by_worker(records, "explicit-NoC", vn) for vn in (0, 1, 2)
    }

    signal_blue = "#3977A8"
    modeled_gray = "#8A9199"
    orange = "#D9772B"
    vn_styles = {
        0: ("#55A6B8", "o"),
        1: ("#3977A8", "s"),
        2: ("#264A70", "^"),
    }

    figure = plt.figure(figsize=(13.333, 7.5), constrained_layout=True)
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=[0.82, 1.35],
        height_ratios=[1.25, 0.82],
    )
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])
    axis_c = figure.add_subplot(grid[1, :])
    figure.suptitle(
        "Phase 4E: VN equivalence and reduction-network pressure",
        fontsize=21,
        fontweight="bold",
    )

    modeled_runtime = [modeled[worker].simulated_time_us for worker in workers]
    axis_a.plot(
        workers,
        modeled_runtime,
        color=modeled_gray,
        marker="D",
        markersize=6,
        linewidth=1.8,
        label="modeled-NoC",
        zorder=2,
    )
    offsets = {0: -0.16, 1: 0.0, 2: 0.16}
    for vn in (0, 1, 2):
        color, marker = vn_styles[vn]
        axis_a.plot(
            [worker + offsets[vn] for worker in workers],
            [explicit[vn][worker].simulated_time_us for worker in workers],
            color=color,
            marker=marker,
            markersize=6,
            linewidth=1.2,
            alpha=0.9,
            label=f"explicit VN{vn}",
            zorder=3,
        )
    all_runtimes = modeled_runtime + [
        explicit[vn][worker].simulated_time_us
        for vn in (0, 1, 2)
        for worker in workers
    ]
    axis_a.set_ylim(min(all_runtimes) - 0.65, max(all_runtimes) + 1.05)
    axis_a.set_xticks(workers)
    axis_a.set_xlabel("Worker cores")
    axis_a.set_ylabel("Simulated time (us)")
    axis_a.set_title("End-to-end runtime", loc="left", fontweight="bold")
    axis_a.grid(axis="y", color="#D8DCE0", linewidth=0.7, alpha=0.7)
    axis_a.legend(loc="upper left", frameon=False, ncols=2, handlelength=1.6)
    axis_a.text(
        0.98,
        0.12,
        "VN0/VN1/VN2 overlap exactly",
        transform=axis_a.transAxes,
        color=vn_styles[2][0],
        ha="right",
        fontweight="bold",
    )
    axis_a.text(
        0.98,
        0.04,
        "max |explicit - modeled| < 0.061%",
        transform=axis_a.transAxes,
        color="#4E555B",
        ha="right",
        fontsize=10,
    )
    _panel_label(axis_a, "a")

    explicit_reference = explicit[0]
    average_latency = [
        explicit_reference[worker].latency_average_cycles for worker in workers
    ]
    maximum_latency = [explicit_reference[worker].latency_max_cycles for worker in workers]
    axis_b.plot(
        workers,
        average_latency,
        color=signal_blue,
        marker="o",
        markersize=7,
        linewidth=2.6,
        label="Average latency",
    )
    axis_b.plot(
        workers,
        maximum_latency,
        color=orange,
        marker="o",
        markersize=6,
        linewidth=2.0,
        linestyle="--",
        label="Maximum latency",
    )
    axis_b.set_xticks(workers)
    axis_b.set_xlabel("Worker cores")
    axis_b.set_ylabel("Latency (cycles)")
    axis_b.set_title("Reduction transport latency", loc="left", fontweight="bold")
    axis_b.grid(axis="y", color="#D8DCE0", linewidth=0.7, alpha=0.7)
    axis_b.annotate(
        f"{round(average_latency[0]):,}",
        (workers[0], average_latency[0]),
        xytext=(9, 10),
        textcoords="offset points",
        color=signal_blue,
        fontweight="bold",
    )
    axis_b.annotate(
        f"Average: {round(average_latency[-1]):,} cycles",
        (workers[-1], average_latency[-1]),
        xytext=(-8, -35),
        textcoords="offset points",
        ha="right",
        color=signal_blue,
        fontweight="bold",
    )
    axis_b.annotate(
        f"Maximum: {round(maximum_latency[-1]):,} cycles",
        (workers[-1], maximum_latency[-1]),
        xytext=(-8, 10),
        textcoords="offset points",
        ha="right",
        color=orange,
        fontweight="bold",
    )
    axis_b.text(
        0.98,
        0.47,
        f"+{round(derived.latency_growth_pct)}% average latency from 4 to 16 workers",
        transform=axis_b.transAxes,
        color=orange,
        ha="right",
        fontsize=12,
        fontweight="bold",
    )
    axis_b.text(
        0.98,
        0.08,
        "identical for VN0/VN1/VN2",
        transform=axis_b.transAxes,
        color=signal_blue,
        ha="right",
        fontweight="bold",
    )
    _panel_label(axis_b, "b")

    positions = list(range(len(workers)))
    bar_width = 0.32
    modeled_stalls = [modeled[worker].total_xbar_stalls for worker in workers]
    explicit_stalls = [explicit_reference[worker].total_xbar_stalls for worker in workers]
    modeled_bars = axis_c.bar(
        [position - bar_width / 2 for position in positions],
        modeled_stalls,
        width=bar_width,
        color=modeled_gray,
        label="modeled-NoC",
    )
    explicit_bars = axis_c.bar(
        [position + bar_width / 2 for position in positions],
        explicit_stalls,
        width=bar_width,
        color=signal_blue,
        label="explicit-NoC",
    )
    axis_c.bar_label(modeled_bars, padding=3, fontsize=9, color="#4E555B")
    axis_c.bar_label(explicit_bars, padding=3, fontsize=9, color=signal_blue)
    axis_c.set_xlim(-0.6, 5.15)
    axis_c.set_ylim(0, max(modeled_stalls + explicit_stalls) * 1.24)
    axis_c.set_xticks(positions, workers)
    axis_c.set_xlabel("Worker cores")
    axis_c.set_ylabel("Total xbar stalls")
    axis_c.set_title("NoC pressure and validation", loc="left", fontweight="bold")
    axis_c.grid(axis="y", color="#D8DCE0", linewidth=0.7, alpha=0.7)
    axis_c.set_axisbelow(True)
    axis_c.text(
        2.35,
        modeled_stalls[-1],
        "modeled-NoC",
        color="#616970",
        va="center",
        fontweight="bold",
        fontsize=10,
    )
    axis_c.text(
        2.35,
        explicit_stalls[-1],
        "explicit-NoC",
        color=signal_blue,
        va="center",
        fontweight="bold",
        fontsize=10,
    )
    transport_text, validation_text = _validation_annotations(derived)
    axis_c.text(
        0.63,
        0.14,
        transport_text,
        transform=axis_c.transAxes,
        color=signal_blue,
        fontweight="bold",
    )
    axis_c.text(
        0.63,
        0.54,
        validation_text,
        transform=axis_c.transAxes,
        va="center",
        linespacing=1.5,
        color="#30363B",
        fontsize=11,
    )
    _panel_label(axis_c, "c")

    figure.text(
        0.995,
        0.012,
        "Single deterministic simulation per configuration",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#626A70",
    )
    return figure


def render_figure(records: list[PointRecord], output_prefix: pathlib.Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure = build_figure(records)
    try:
        figure.savefig(
            output_prefix.with_suffix(".svg"),
            facecolor="white",
            metadata={"Date": None},
        )
        _normalize_svg_text(output_prefix.with_suffix(".svg"))
        figure.savefig(
            output_prefix.with_suffix(".pdf"),
            facecolor="white",
            metadata={"CreationDate": None, "ModDate": None},
        )
        figure.savefig(output_prefix.with_suffix(".png"), dpi=300, facecolor="white")
    finally:
        plt = _load_plotting_backend()
        plt.close(figure)


def _normalize_svg_text(svg_path: pathlib.Path) -> None:
    text = svg_path.read_text(encoding="utf-8")
    normalized = re.sub(r"[ \t]+(?=\r?$)", "", text, flags=re.MULTILINE)
    svg_path.write_text(normalized, encoding="utf-8", newline="\n")


def main(
    argv: list[str] | None = None, verifier: pathlib.Path | None = None
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweeps-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--parse-only", action="store_true")
    args = parser.parse_args(argv)

    if verifier is None:
        verifier = (
            pathlib.Path(__file__).resolve().parents[4]
            / "small/mvm_noc_softmax_sfu/verify_softmax_sfu_against_golden.py"
        )
    records = load_and_validate_matrix(args.sweeps_root, verifier)
    derived = validate_derived_contracts(records)
    source_csv = args.output_dir / SOURCE_CSV_NAME
    write_source_csv(records, source_csv)
    for record in _ordered_records(records):
        print(
            f"validated transport={record.transport} vn={record.reduction_vn} "
            f"workers={record.workers} events={record.transport_events} "
            f"golden_checked={record.golden_checked} "
            f"golden_mismatches={record.golden_mismatches}"
        )
    print(
        f"validated_records={len(records)} "
        f"explicit_runtime_spread={derived.explicit_runtime_spread:.17g} "
        f"max_explicit_modeled_pct={derived.max_explicit_modeled_pct:.17g} "
        f"latency_growth_pct={derived.latency_growth_pct:.17g}"
    )
    if args.parse_only:
        return 0
    render_figure(records, args.output_dir / "sfu_phase4e_group_figure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
