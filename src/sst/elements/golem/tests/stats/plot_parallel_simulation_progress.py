#!/usr/bin/env python3
"""Generate the TileMC wall-time milestone figure using Pillow and SVG."""

from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "parallel_simulation_walltime_milestones.csv"
DEFAULT_OUTPUT = HERE.parent / "artifacts" / "stats" / "figures"
FONT = "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"
BOLD = "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
W, H, SCALE = 1800, 1050, 2
COLORS = ["#D86459", "#4C78A8", "#2A9D8F", "#6C8E3D", "#18A06A"]
TEXT, MUTED, GRID, AXIS = "#20262C", "#626C74", "#E3E7EA", "#AEB5BB"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prefix", default="parallel_simulation_walltime_progress")
    return parser.parse_args()


def load(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"no milestones in {path}")
    for row in rows:
        row["wall_time_seconds"] = float(row["wall_time_seconds"])
        row["outputs"] = int(row["outputs"])
    return rows


def duration(seconds: float) -> str:
    total = round(seconds)
    minutes, secs = divmod(total, 60)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def wall_y(seconds: float) -> float:
    low, high = math.log10(110), math.log10(7600)
    return 620 - (math.log10(seconds) - low) / (high - low) * 395


def x_positions(count: int) -> list[float]:
    return [220 + i * 1360 / (count - 1) for i in range(count)]


def svg_text(x: float, y: float, value: str, size: int, anchor: str = "start",
             color: str = TEXT, bold: bool = False) -> str:
    weight = 700 if bold else 400
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Liberation Sans,Arial,sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}">{html.escape(value)}</text>'
    )


def metrics(rows: list[dict[str, object]]) -> tuple[list[float], list[int], list[float], list[float]]:
    wall = [float(row["wall_time_seconds"]) for row in rows]
    outputs = [int(row["outputs"]) for row in rows]
    rates = [3600 * count / seconds for count, seconds in zip(outputs, wall)]
    return wall, outputs, rates, x_positions(len(rows))


def render_svg(rows: list[dict[str, object]], output: Path) -> None:
    wall, outputs, rates, xs = metrics(rows)
    gain, turnaround = rates[-1] / rates[0], wall[0] / wall[-1]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(110, 82, "TileMC Parallel Simulation", 31, bold=True),
        svg_text(110, 125, "From one 84-minute run to four runs in under 4 minutes", 24, bold=True),
        svg_text(110, 162, f"4096 x 4096 x 4096 GEMM   |   {gain:.1f}x higher experiment throughput", 17, color=MUTED),
        svg_text(110, 215, "END-TO-END WALL TIME", 14, color=MUTED, bold=True),
    ]
    for value, label in [(120, "2 min"), (240, "4 min"), (480, "8 min"),
                         (960, "16 min"), (1920, "32 min"), (3840, "64 min")]:
        y = wall_y(value)
        parts += [
            f'<line x1="175" y1="{y:.1f}" x2="1630" y2="{y:.1f}" stroke="{GRID}" stroke-width="1.5"/>',
            svg_text(158, y + 5, label, 14, "end", MUTED),
        ]
    parts.append(f'<line x1="175" y1="620" x2="1630" y2="620" stroke="{AXIS}" stroke-width="1.5"/>')
    points = " ".join(f"{x:.1f},{wall_y(value):.1f}" for x, value in zip(xs, wall))
    parts.append(f'<polyline points="{points}" fill="none" stroke="#7A838B" stroke-width="3"/>')
    for i, (x, seconds, count) in enumerate(zip(xs, wall, outputs)):
        y = wall_y(seconds)
        label = duration(seconds) + (f" / {count} cases" if count > 1 else "")
        parts += [
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="{COLORS[i]}" stroke="white" stroke-width="3"/>',
            svg_text(x, y - 21, label, 17, "middle", COLORS[i], True),
        ]
    end_x, end_y = xs[-1] - 12, wall_y(wall[-1]) - 8
    parts += [
        svg_text(1300, 292, f"{turnaround:.1f}x shorter batch turnaround", 16, "middle", COLORS[-1], True),
        f'<line x1="1370" y1="315" x2="{end_x:.1f}" y2="{end_y:.1f}" stroke="{COLORS[-1]}" stroke-width="2"/>',
        f'<circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="3" fill="{COLORS[-1]}"/>',
        svg_text(110, 690, "EXPERIMENT THROUGHPUT", 14, color=MUTED, bold=True),
        f'<line x1="175" y1="895" x2="1630" y2="895" stroke="{AXIS}" stroke-width="1.5"/>',
    ]
    max_rate = max(rates) * 1.15
    for i, (x, rate) in enumerate(zip(xs, rates)):
        height = rate / max_rate * 175
        y = 895 - height
        parts += [
            f'<rect x="{x - 48:.1f}" y="{y:.1f}" width="96" height="{height:.1f}" rx="2" fill="{COLORS[i]}"/>',
            svg_text(x, y - 12, f"{rate:.1f}", 16, "middle", TEXT, True),
        ]
        words = str(rows[i]["short_label"]).split()
        parts.append(svg_text(x, 930, words[0], 15, "middle", COLORS[i], True))
        if len(words) > 1:
            parts.append(svg_text(x, 950, " ".join(words[1:]), 15, "middle", COLORS[i], True))
    parts += [
        svg_text(155, 805, "results / hour", 14, "end", MUTED),
        svg_text(110, 1008, "Complete-pipeline wall time. Final point: four-case concurrent makespan (59 s/result amortized); earlier points: single-case runs. Legacy baseline: previous host.", 13, color=MUTED),
        "</svg>",
    ]
    output.write_text("\n".join(parts) + "\n")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(BOLD if bold else FONT, size * SCALE)


def render_png(rows: list[dict[str, object]], output: Path) -> None:
    image = Image.new("RGB", (W * SCALE, H * SCALE), "white")
    draw = ImageDraw.Draw(image)
    wall, outputs, rates, xs = metrics(rows)
    gain, turnaround = rates[-1] / rates[0], wall[0] / wall[-1]

    def p(x: float, y: float) -> tuple[int, int]:
        return round(x * SCALE), round(y * SCALE)

    def txt(x: float, y: float, value: str, size: int, anchor: str = "la",
            color: str = TEXT, bold: bool = False) -> None:
        draw.text(p(x, y), value, font=font(size, bold), fill=color, anchor=anchor)

    txt(110, 58, "TileMC Parallel Simulation", 31, bold=True)
    txt(110, 102, "From one 84-minute run to four runs in under 4 minutes", 24, bold=True)
    txt(110, 143, f"4096 x 4096 x 4096 GEMM   |   {gain:.1f}x higher experiment throughput", 17, color=MUTED)
    txt(110, 195, "END-TO-END WALL TIME", 14, color=MUTED, bold=True)
    for value, label in [(120, "2 min"), (240, "4 min"), (480, "8 min"),
                         (960, "16 min"), (1920, "32 min"), (3840, "64 min")]:
        y = wall_y(value)
        draw.line([p(175, y), p(1630, y)], fill=GRID, width=3)
        txt(158, y, label, 14, "rm", MUTED)
    draw.line([p(175, 620), p(1630, 620)], fill=AXIS, width=3)
    draw.line([p(x, wall_y(value)) for x, value in zip(xs, wall)], fill="#7A838B", width=6)
    for i, (x, seconds, count) in enumerate(zip(xs, wall, outputs)):
        y = wall_y(seconds)
        cx, cy, radius = *p(x, y), 11 * SCALE
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=COLORS[i], outline="white", width=5)
        label = duration(seconds) + (f" / {count} cases" if count > 1 else "")
        txt(x, y - 24, label, 17, "ms", COLORS[i], True)
    txt(1300, 288, f"{turnaround:.1f}x shorter batch turnaround", 16, "mm", COLORS[-1], True)
    draw.line([p(1370, 315), p(xs[-1] - 12, wall_y(wall[-1]) - 8)], fill=COLORS[-1], width=4)
    txt(110, 670, "EXPERIMENT THROUGHPUT", 14, color=MUTED, bold=True)
    draw.line([p(175, 895), p(1630, 895)], fill=AXIS, width=3)
    max_rate = max(rates) * 1.15
    for i, (x, rate) in enumerate(zip(xs, rates)):
        height = rate / max_rate * 175
        y = 895 - height
        draw.rounded_rectangle((*p(x - 48, y), *p(x + 48, 895)), radius=4, fill=COLORS[i])
        txt(x, y - 13, f"{rate:.1f}", 16, "ms", bold=True)
        words = str(rows[i]["short_label"]).split()
        txt(x, 925, words[0], 15, "ms", COLORS[i], True)
        if len(words) > 1:
            txt(x, 947, " ".join(words[1:]), 15, "ms", COLORS[i], True)
    txt(155, 805, "results / hour", 14, "rm", MUTED)
    txt(110, 1000, "Complete-pipeline wall time. Final point: four-case concurrent makespan (59 s/result amortized); earlier points: single-case runs. Legacy baseline: previous host.", 13, color=MUTED)
    resampling = getattr(Image, "Resampling", Image)
    image.resize((W, H), resampling.LANCZOS).save(output, dpi=(240, 240))


def main() -> None:
    args = arguments()
    rows = load(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    svg = args.output_dir / f"{args.prefix}.svg"
    png = args.output_dir / f"{args.prefix}.png"
    render_svg(rows, svg)
    render_png(rows, png)
    print(svg)
    print(png)


if __name__ == "__main__":
    main()
