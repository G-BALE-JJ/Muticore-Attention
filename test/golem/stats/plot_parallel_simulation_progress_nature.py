#!/usr/bin/env python3
"""Generate a compact, publication-style TileMC optimization figure."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
from pathlib import Path

from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Group, Line, PolyLine, Rect, String
from reportlab.lib.colors import HexColor, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "parallel_simulation_walltime_milestones.csv"
DEFAULT_OUTPUT = HERE.parent / "artifacts" / "stats" / "figures"
REGULAR = "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"
BOLD = "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"

W, H = 520, 300  # 183 x 106 mm, suitable for a double-column figure.
INK = HexColor("#252525")
MUTED = HexColor("#666666")
GRID = HexColor("#E5E5E5")
AXIS = HexColor("#A6A6A6")
LEGACY = HexColor("#D55E00")
OPTIMIZED = HexColor("#0072B2")
FINAL = HexColor("#009E73")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prefix", default="parallel_simulation_walltime_nature")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 5:
        raise SystemExit(f"expected five milestones in {path}, found {len(rows)}")
    for row in rows:
        row["wall_time_seconds"] = float(row["wall_time_seconds"])
        row["outputs"] = int(row["outputs"])
    return rows


def add_text(group: Group, x: float, y: float, value: str, size: float = 7,
             *, bold: bool = False, color=INK, anchor: str = "start") -> None:
    group.add(
        String(
            x,
            y,
            value,
            fontName="LiberationSans-Bold" if bold else "LiberationSans",
            fontSize=size,
            fillColor=color,
            textAnchor=anchor,
        )
    )


def log_y(value: float, bottom: float, top: float) -> float:
    low, high = math.log10(100), math.log10(10000)
    return bottom + (math.log10(value) - low) / (high - low) * (top - bottom)


def draw_wall_time(drawing: Drawing, rows: list[dict[str, object]]) -> None:
    group = Group()
    x0, x1, y0, y1 = 43.0, 329.0, 67.0, 244.0
    xs = [59.0 + i * 66.0 for i in range(5)]
    wall = [float(row["wall_time_seconds"]) for row in rows]
    colors = [LEGACY, OPTIMIZED, OPTIMIZED, OPTIMIZED, FINAL]

    add_text(group, 8, 278, "a", 11, bold=True)
    add_text(group, 27, 279, "Wall time", 9, bold=True)
    add_text(group, 27, 265, "End-to-end pipeline (log scale)", 7, color=MUTED)

    ticks = [(100, "100"), (300, "300"), (1000, "1,000"), (3000, "3,000"), (10000, "10,000")]
    for value, label in ticks:
        y = log_y(value, y0, y1)
        group.add(Line(x0, y, x1, y, strokeColor=GRID, strokeWidth=0.55))
        add_text(group, x0 - 5, y - 2.3, label, 6.5, color=MUTED, anchor="end")
    group.add(Line(x0, y0, x0, y1, strokeColor=AXIS, strokeWidth=0.7))
    group.add(Line(x0, y0, x1, y0, strokeColor=AXIS, strokeWidth=0.7))
    add_text(group, x0 - 29, y1 + 5, "Wall time (s)", 6.5, color=MUTED)

    four_min_y = log_y(240, y0, y1)
    four_min = Line(x0, four_min_y, x1, four_min_y, strokeColor=FINAL, strokeWidth=0.7)
    four_min.strokeDashArray = [2, 2]
    group.add(four_min)
    add_text(group, x0 + 3, four_min_y + 3.5, "4 min", 6.2, color=FINAL)

    points = []
    for x, seconds in zip(xs, wall):
        points.extend([x, log_y(seconds, y0, y1)])
    group.add(PolyLine(points, strokeColor=MUTED, strokeWidth=1.15, fillColor=None))

    value_labels = ["5,021", "510", "209", "160", "236 (4 cases)"]
    stage_labels = [
        ("Legacy", "baseline"),
        ("MPI +", "weighted"),
        ("Full-", "timing"),
        ("Event-", "driven"),
        ("4-case", "sweep"),
    ]
    for i, (x, seconds, color) in enumerate(zip(xs, wall, colors)):
        y = log_y(seconds, y0, y1)
        group.add(Circle(x, y, 3.4, fillColor=color, strokeColor=white, strokeWidth=0.8))
        label_y = y + 7 if i not in {2, 3} else y - 11
        add_text(group, x, label_y, value_labels[i], 6.8, bold=True, color=color, anchor="middle")
        add_text(group, x, 51, stage_labels[i][0], 6.7, bold=True, color=INK, anchor="middle")
        add_text(group, x, 42, stage_labels[i][1], 6.7, color=MUTED, anchor="middle")
        add_text(group, x, 31, str(i + 1), 6.2, color=MUTED, anchor="middle")
    add_text(group, (x0 + x1) / 2, 18, "Optimization stage", 6.8, color=MUTED, anchor="middle")
    drawing.add(group)


def draw_throughput(drawing: Drawing, rows: list[dict[str, object]]) -> None:
    group = Group()
    x0, x1, y0, y1 = 377.0, 507.0, 67.0, 244.0
    xs = [386.0 + i * 28.0 for i in range(5)]
    wall = [float(row["wall_time_seconds"]) for row in rows]
    outputs = [int(row["outputs"]) for row in rows]
    rates = [3600.0 * count / seconds for count, seconds in zip(outputs, wall)]
    colors = [LEGACY, OPTIMIZED, OPTIMIZED, OPTIMIZED, FINAL]

    add_text(group, 350, 278, "b", 11, bold=True)
    add_text(group, 369, 279, "Experiment throughput", 9, bold=True)
    add_text(group, 369, 265, "Completed results per host hour", 7, color=MUTED)

    for value in [0, 20, 40, 60]:
        y = y0 + value / 70.0 * (y1 - y0)
        group.add(Line(x0, y, x1, y, strokeColor=GRID, strokeWidth=0.55))
        add_text(group, x0 - 5, y - 2.3, str(value), 6.5, color=MUTED, anchor="end")
    group.add(Line(x0, y0, x0, y1, strokeColor=AXIS, strokeWidth=0.7))
    group.add(Line(x0, y0, x1, y0, strokeColor=AXIS, strokeWidth=0.7))
    add_text(group, x0 - 24, y1 + 5, "Results/hour", 6.5, color=MUTED)

    for i, (x, value, color) in enumerate(zip(xs, rates, colors)):
        top = y0 + value / 70.0 * (y1 - y0)
        group.add(Rect(x - 7.0, y0, 14.0, top - y0, fillColor=color, strokeColor=None))
        label = f"{value:.1f}"
        if i == 4:
            label += "*"
        add_text(group, x, top + 5, label, 6.8, bold=True, color=color, anchor="middle")
        add_text(group, x, 49, str(i + 1), 6.7, bold=True, color=INK, anchor="middle")
    add_text(group, (x0 + x1) / 2, 35, "Optimization stage", 6.8, color=MUTED, anchor="middle")
    add_text(group, x1, 22, "* 85.1-fold vs stage 1", 6.5, bold=True, color=FINAL, anchor="end")
    drawing.add(group)


def build_figure(rows: list[dict[str, object]]) -> Drawing:
    drawing = Drawing(W, H)
    drawing.add(Rect(0, 0, W, H, fillColor=white, strokeColor=None))
    draw_wall_time(drawing, rows)
    draw_throughput(drawing, rows)
    return drawing


def write_caption(path: Path) -> None:
    path.write_text(
        "Figure | Cumulative optimization of TileMC parallel simulation for a "
        "4096 x 4096 x 4096 GEMM workload. a, End-to-end wall time across five "
        "milestones; the dashed line marks four minutes. Stages 1-4 report one "
        "simulation, whereas stage 5 reports the makespan of four concurrently "
        "executed tile-shape cases. b, Corresponding experiment throughput in "
        "completed results per host hour. The final batch amortizes to 59 s per "
        "result and provides 85.1-fold higher throughput than the legacy baseline. "
        "The 5,021-s baseline was measured on the previous host; therefore, the "
        "end-to-end difference combines platform and software effects. Stages 2-5 "
        "use full-system SST simulation; stages 3-5 use full-timing mode.\n"
    )


def main() -> None:
    args = arguments()
    pdfmetrics.registerFont(TTFont("LiberationSans", REGULAR))
    pdfmetrics.registerFont(TTFont("LiberationSans-Bold", BOLD))
    rows = load_rows(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure = build_figure(rows)
    pdf = args.output_dir / f"{args.prefix}.pdf"
    svg = args.output_dir / f"{args.prefix}.svg"
    png = args.output_dir / f"{args.prefix}.png"
    caption = args.output_dir / f"{args.prefix}_caption.txt"
    renderPDF.drawToFile(figure, str(pdf))
    renderSVG.drawToFile(figure, str(svg))
    subprocess.run(
        ["pdftocairo", "-png", "-singlefile", "-r", "300", str(pdf), str(png.with_suffix(""))],
        check=True,
    )
    write_caption(caption)
    for path in (pdf, svg, png, caption):
        print(path)


if __name__ == "__main__":
    main()
