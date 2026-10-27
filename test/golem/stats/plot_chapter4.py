#!/usr/bin/env python3
import argparse
import csv
import html
import math
from pathlib import Path


W, H = 1100, 620
COLORS = ["#2f6f9f", "#e38b35", "#2f8f83", "#b85c73"]


def val(row, key, default=0.0):
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def esc(value):
    return html.escape(str(value))


def start(title, subtitle):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.small{font-size:14px;fill:#596579}.axis{stroke:#8994a5;stroke-width:1}.grid{stroke:#e4e8ef}.title{font-size:25px;font-weight:700}.label{font-size:16px;font-weight:600}.legend{font-size:14px}</style>',
        f'<text x="50" y="42" class="title">{esc(title)}</text>',
        f'<text x="50" y="67" class="small">{esc(subtitle)}</text>',
    ]


def finish(lines, path):
    lines.append("</svg>")
    path.write_text("\n".join(lines))


def panel(lines, x, y, width, height, labels, series, ylabel, colors, ymax=None):
    left, right = x + 70, x + width - 20
    top, bottom = y + 8, y + height - 55
    plot_w, plot_h = right - left, bottom - top
    finite = [v for values in series for v in values if math.isfinite(v)]
    ymax = max(1.0, max(finite, default=1.0) * 1.18) if ymax is None else max(1.0, ymax)
    for tick in range(5):
        yy = bottom - plot_h * tick / 4
        lines.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}" class="grid"/>')
        lines.append(f'<text x="{left-12}" y="{yy+5:.1f}" text-anchor="end" class="small">{ymax*tick/4:.1f}</text>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>')
    lines.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>')
    groups = max(1, len(labels))
    group_w = plot_w / groups
    bar_w = min(34, group_w / (len(series) + 1.5))
    for i, label in enumerate(labels):
        gx = left + group_w * (i + 0.5)
        lines.append(f'<text x="{gx:.1f}" y="{bottom+27}" text-anchor="middle" class="small">{esc(label)}</text>')
        for j, values in enumerate(series):
            value = values[i] if i < len(values) else 0.0
            bar_h = plot_h * value / ymax if math.isfinite(value) else 0.0
            xx = gx + (j - (len(series)-1)/2) * bar_w
            lines.append(f'<rect x="{xx-bar_w/2:.1f}" y="{bottom-bar_h:.1f}" width="{max(2, bar_w-3):.1f}" height="{max(0, bar_h):.1f}" fill="{colors[j]}"/>')
    lines.append(f'<text x="{x+18}" y="{y+height/2:.1f}" transform="rotate(-90 {x+18} {y+height/2:.1f})" text-anchor="middle" class="label">{esc(ylabel)}</text>')


def legend(lines, x, y, names, colors):
    for i, (name, color) in enumerate(zip(names, colors)):
        xx = x + i * 190
        lines.append(f'<rect x="{xx}" y="{y-12}" width="13" height="13" fill="{color}"/>')
        lines.append(f'<text x="{xx+19}" y="{y}" class="legend">{esc(name)}</text>')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(args.input.open(newline="")))
    valid = [row for row in rows if row.get("status") == "PASS"]
    if not valid:
        raise SystemExit("no PASS rows in chapter4_results.csv")

    reuse = [row for row in valid if row.get("phase") == "reuse_ablation"]
    if reuse:
        labels = [row["case"].replace("reuse_", "") for row in reuse]
        lines = start("2D Reuse Window: Cross-Tile Traffic", "A/B panel reads after hierarchical multicast")
        panel(lines, 0, 88, W, 430, labels,
              [[val(row, "a_read_bytes") / 2**30 for row in reuse],
               [val(row, "b_read_bytes") / 2**30 for row in reuse]],
              "Read traffic (GiB)", COLORS[:2])
        legend(lines, 120, 570, ["A panel reads", "B panel reads"], COLORS[:2])
        finish(lines, args.output_dir / "reuse_ablation.svg")

    credit = [row for row in valid if row.get("phase") == "credit_ablation"]
    if credit:
        labels = ["unbounded" if row["case"] == "credit_unbounded" else row["node_credit"] for row in credit]
        lines = start("Credit Admission: Throughput and Queueing", "4x4 reuse; node credit is the only changed parameter")
        panel(lines, 0, 88, W, 220, labels,
              [[val(row, "exec_system_array_utilization_pct") for row in credit]],
              "Array utilization (%)", [COLORS[2]], ymax=100)
        panel(lines, 0, 330, W, 220, labels,
              [[val(row, "memory_queue_delay_p99_cycles") for row in credit]],
              "Queue delay p99 (cycles)", [COLORS[3]])
        legend(lines, 120, 590, ["Utilization", "Queue delay p99"], [COLORS[2], COLORS[3]])
        finish(lines, args.output_dir / "credit_ablation.svg")

    prefetch = [row for row in valid if row.get("phase") == "prefetch_ablation"]
    if prefetch:
        labels = [row["prefetch"] for row in prefetch]
        lines = start("Prefetch Window: Burst Sensitivity", "4x4 reuse and credit=84; larger windows increase backend queue tail")
        panel(lines, 0, 88, W, 220, labels,
              [[val(row, "exec_system_array_utilization_pct") for row in prefetch]],
              "Array utilization (%)", [COLORS[2]], ymax=100)
        panel(lines, 0, 330, W, 220, labels,
              [[val(row, "memory_queue_delay_p99_cycles") for row in prefetch]],
              "Queue delay p99 (cycles)", [COLORS[3]])
        legend(lines, 120, 590, ["Utilization", "Queue delay p99"], [COLORS[2], COLORS[3]])
        finish(lines, args.output_dir / "prefetch_ablation.svg")

    multicast = next((row for row in valid if row.get("phase") == "profile_multicast"), None)
    if multicast:
        lines = start("Motivation: Residual Cross-Tile Traffic", "Hierarchical multicast removes intra-tile duplication; 1x1 leaves inter-tile reuse on the table")
        panel(lines, 0, 88, W, 430, ["1x1"],
              [[val(multicast, "a_read_bytes") / 2**30], [val(multicast, "b_read_bytes") / 2**30]],
              "Read traffic (GiB)", COLORS[:2])
        legend(lines, 120, 570, ["A panel reads", "B panel reads"], COLORS[:2])
        finish(lines, args.output_dir / "method_multicast_residual.svg")

    congestion = next((row for row in valid if row.get("phase") == "profile_congestion"), None)
    if congestion:
        lines = start("Motivation: Burst-Induced Congestion", "4x4 reuse lowers total traffic but concentrates prefetch bursts at the memory backend")
        panel(lines, 0, 88, W, 220, ["average", "p99"],
              [[val(congestion, "memory_queue_delay_avg_cycles"), val(congestion, "memory_queue_delay_p99_cycles")],
               [val(congestion, "memory_backend_read_latency_avg_cycles"), val(congestion, "memory_backend_read_latency_p99_cycles")]],
              "Latency (cycles)", [COLORS[3], COLORS[0]])
        legend(lines, 120, 370, ["Queue delay", "Backend read latency"], [COLORS[3], COLORS[0]])
        finish(lines, args.output_dir / "method_congestion.svg")

    print(f"[OK] wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
