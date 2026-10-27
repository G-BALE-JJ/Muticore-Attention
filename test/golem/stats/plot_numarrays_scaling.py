#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {
    "actual": "#4e79a7",
    "ideal": "#e15759",
    "util": "#16a34a",
    "guide": "#9ca3af",
    "text": "#334155",
}


def setup_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#d1d5db",
            "axes.linewidth": 0.8,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "grid.color": "#e5e7eb",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "font.family": "DejaVu Sans",
            "savefig.bbox": "tight",
        }
    )


def load_rows(path: Path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["num_arrays"]))
    return rows


def main():
    p = argparse.ArgumentParser(description="Plot num_arrays scaling speedup")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--title", default="Density Scaling")
    p.add_argument("--baseline", type=int, default=8)
    args = p.parse_args()

    rows = load_rows(Path(args.input))
    if not rows:
        raise SystemExit("empty input csv")

    base_row = None
    for row in rows:
        if int(row["num_arrays"]) == args.baseline:
            base_row = row
            break
    if base_row is None:
        raise SystemExit(f"missing num_arrays={args.baseline} baseline")

    base_cycles = float(base_row["exec_total_mean"])
    base_arrays = int(base_row["num_arrays"])

    xs = [int(r["num_arrays"]) for r in rows]
    actual = [base_cycles / float(r["exec_total_mean"]) for r in rows]
    ideal = [x / base_arrays for x in xs]
    util = []
    for r in rows:
        try:
            util.append(float(r.get("exec_array_utilization_pct", "")))
        except ValueError:
            util.append(float("nan"))

    setup_style()
    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    ax.plot(
        xs,
        actual,
        color=COLORS["actual"],
        marker="o",
        linewidth=2.6,
        markersize=7,
        label="Measured speedup",
    )
    ax.plot(
        xs,
        ideal,
        color=COLORS["ideal"],
        marker="s",
        linewidth=2.2,
        markersize=6,
        linestyle="--",
        label="Ideal linear scaling",
    )

    for x, y in zip(xs, actual):
        ax.annotate(
            f"{y:.2f}x",
            xy=(x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=COLORS["text"],
        )

    ax2 = ax.twinx()
    ax2.plot(
        xs,
        util,
        color=COLORS["util"],
        marker="^",
        linewidth=2.2,
        markersize=7,
        linestyle=":",
        label="Array utilization",
    )
    for x, y in zip(xs, util):
        if y == y:
            ax2.annotate(
                f"{y:.1f}%",
                xy=(x, y),
                xytext=(0, -16),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=COLORS["util"],
            )

    ax.set_title(args.title, fontsize=20, pad=10)
    ax.set_xlabel("GOLEM_NUM_ARRAYS (= block_n)")
    ax.set_ylabel(f"Speedup vs num_arrays={base_arrays} baseline")
    ax2.set_ylabel("Array utilization (%)", color=COLORS["util"])
    ax2.tick_params(axis="y", colors=COLORS["util"])
    ax.set_xticks(xs)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(COLORS["util"])

    ymax = max(max(actual), max(ideal))
    ax.set_ylim(0, ymax * 1.18)
    finite_util = [v for v in util if v == v]
    if finite_util:
        ax2.set_ylim(0, max(100.0, max(finite_util) * 1.15))

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")

    fig.savefig(args.output, dpi=240)
    if args.output.lower().endswith(".png"):
        fig.savefig(args.output[:-4] + ".svg")
    plt.close(fig)
    print(f"[OK] wrote scaling plot: {args.output}")


if __name__ == "__main__":
    main()
