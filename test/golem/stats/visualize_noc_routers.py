#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NoC Router Visualization Script
Generates heatmaps/line chart for router-level metrics.
"""

import re
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def infer_mesh_dims(router_ids):
    """Infer mesh dimensions from router ids and optional env overrides."""
    if not router_ids:
        return 0, 0

    router_count = len(router_ids)

    env_x = os.getenv("GOLEM_MESH_DIM_X", "").strip()
    if env_x:
        try:
            mesh_cols = int(env_x)
            if mesh_cols > 0:
                mesh_rows = int(np.ceil(router_count / mesh_cols))
                return mesh_cols, mesh_rows
        except ValueError:
            pass

    sqrt_n = int(np.sqrt(router_count))
    best_pair = None
    best_score = None
    for cols in range(1, sqrt_n + 1):
        if router_count % cols != 0:
            continue
        rows = router_count // cols
        score = abs(rows - cols)
        if best_score is None or score < best_score:
            best_score = score
            best_pair = (cols, rows)

    if best_pair is not None:
        return best_pair

    mesh_cols = int(np.ceil(np.sqrt(router_count)))
    mesh_rows = int(np.ceil(router_count / mesh_cols))
    return mesh_cols, mesh_rows


def parse_router_stats(filename, metric="xbar_stalls", max_routers=None):
    """Parse router statistics data for a given metric (optionally limited to first N routers)."""
    router_data = {}

    with open(filename, "r") as f:
        for line in f:
            # Match: rtr_X,<metric>,portY,Accumulator,...
            match = re.match(
                rf"rtr_(\d+),{re.escape(metric)},port\d+,Accumulator,\d+,\d+,(\d+),",
                line,
            )
            if match:
                router_num = int(match.group(1))
                stalls = int(match.group(2))

                # Optionally skip routers beyond the requested limit
                if max_routers is not None and router_num >= max_routers:
                    continue

                if router_num not in router_data:
                    router_data[router_num] = 0
                router_data[router_num] += stalls

    return router_data


def create_heatmap(
    router_data,
    output_file="noc_router_heatmap.png",
    mesh_cols=4,
    mesh_rows=5,
    title_prefix="NoC Router Heatmap",
    value_label="Crossbar Stalls",
):
    """Create router heatmap matching actual Mesh topology

    Args:
        router_data: Dictionary of router_id -> stalls
        output_file: Output filename
        mesh_cols: Number of columns in actual Mesh (default 4)
        mesh_rows: Number of rows in actual Mesh (default 5)
    """
    num_routers = len(router_data)

    # Use actual Mesh dimensions
    cols = mesh_cols
    rows = mesh_rows

    # Create grid data
    grid = np.zeros((rows, cols))
    grid[:] = np.nan  # Fill with NaN for empty cells

    for router_num, stalls in router_data.items():
        row = router_num // cols
        col = router_num % cols
        if row < rows and col < cols:
            grid[row, col] = stalls

    # Create figure
    fig, ax = plt.subplots(figsize=(max(10, cols * 2), max(9, rows * 2)))

    # Draw heatmap (use masked array to handle NaN)
    masked_grid = np.ma.masked_invalid(grid)
    im = ax.imshow(masked_grid, cmap="YlOrRd", aspect="auto")

    # Set axes
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels([f"Col {i}" for i in range(cols)], fontsize=14)
    ax.set_yticklabels([f"Row {i}" for i in range(rows)], fontsize=14)

    # Display values and router numbers in each cell
    max_stalls = max(router_data.values()) if router_data else 1
    for i in range(rows):
        for j in range(cols):
            router_num = i * cols + j
            if router_num in router_data:
                stalls = int(grid[i, j])

                # Adjust text color based on value
                text_color = "white" if stalls > max_stalls * 0.5 else "black"

                # Show router number and stall count
                text = ax.text(
                    j,
                    i,
                    f"rtr_{router_num}\n{stalls}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=12,
                    weight="bold",
                )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(value_label, rotation=270, labelpad=20, fontsize=16)

    # Set title
    ax.set_title(
        f"{title_prefix} ({cols}×{rows} Mesh)\n{value_label} Distribution - {num_routers} Routers",
        fontsize=18,
        weight="bold",
        pad=20,
    )

    # Add grid lines
    ax.set_xticks(np.arange(cols) - 0.5, minor=True)
    ax.set_yticks(np.arange(rows) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=2)

    # Add info text
    total_stalls = sum(router_data.values())
    max_router = max(router_data.items(), key=lambda x: x[1])
    info_text = f"Total Routers: {num_routers}\nTotal Stalls: {total_stalls}\nMax: rtr_{max_router[0]} ({max_router[1]})"
    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Heatmap saved: {output_file}")
    plt.close()


def create_line_chart(router_data, output_file="noc_router_stalls_line.png"):
    """Create router stalls line chart"""
    # Prepare data
    routers = sorted(router_data.keys())
    stalls = [router_data[r] for r in routers]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    # Draw line chart
    line = ax.plot(
        routers,
        stalls,
        marker="o",
        linewidth=2,
        markersize=8,
        color="#2E86AB",
        markerfacecolor="#A23B72",
        markeredgewidth=2,
        markeredgecolor="#2E86AB",
    )

    # Annotate maximum point
    max_idx = stalls.index(max(stalls))
    max_router = routers[max_idx]
    max_stalls = stalls[max_idx]
    ax.annotate(
        f"Max Bottleneck\nrtr_{max_router}\n{max_stalls} stalls",
        xy=(max_router, max_stalls),
        xytext=(max_router, max_stalls + 80),
        ha="center",
        fontsize=11,
        weight="bold",
        color="red",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
        arrowprops=dict(
            arrowstyle="->", connectionstyle="arc3,rad=0", color="red", lw=2
        ),
    )

    # Annotate second maximum point
    stalls_copy = stalls.copy()
    stalls_copy[max_idx] = -1
    second_max_idx = stalls_copy.index(max(stalls_copy))
    second_router = routers[second_max_idx]
    second_stalls = stalls[second_max_idx]
    ax.annotate(
        f"2nd Bottleneck\nrtr_{second_router}\n{second_stalls} stalls",
        xy=(second_router, second_stalls),
        xytext=(second_router, second_stalls + 50),
        ha="center",
        fontsize=10,
        color="orange",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7),
        arrowprops=dict(
            arrowstyle="->", connectionstyle="arc3,rad=0", color="orange", lw=1.5
        ),
    )

    # Add horizontal reference line (average)
    avg_stalls = np.mean(stalls)
    ax.axhline(
        y=avg_stalls,
        color="green",
        linestyle="--",
        linewidth=1.5,
        label=f"Average: {avg_stalls:.1f} stalls",
        alpha=0.7,
    )

    # Add threshold line
    threshold = 50
    ax.axhline(
        y=threshold,
        color="red",
        linestyle=":",
        linewidth=1.5,
        label=f"Severe Threshold: {threshold} stalls",
        alpha=0.7,
    )

    # Set axes
    ax.set_xlabel("Router Number (rtr_X)", fontsize=12, weight="bold")
    ax.set_ylabel("Crossbar Stalls", fontsize=12, weight="bold")
    ax.set_title(
        f"NoC Router Crossbar Stalls Distribution\nAll {len(routers)} Routers (rtr_0 ~ rtr_{max(routers)})",
        fontsize=14,
        weight="bold",
        pad=20,
    )

    # Set x-axis ticks
    ax.set_xticks(routers)
    ax.set_xticklabels(
        [f"rtr_{r}" for r in routers], rotation=45, ha="right", fontsize=12
    )

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Add legend
    ax.legend(loc="upper left", fontsize=12, framealpha=0.9)

    # Add statistics info box
    total_stalls = sum(stalls)
    severe_count = sum(1 for s in stalls if s > 50)
    moderate_count = sum(1 for s in stalls if 10 < s <= 50)
    stats_text = (
        f"Statistics:\n"
        f"Total Stalls: {total_stalls}\n"
        f"Average: {avg_stalls:.1f}\n"
        f"Maximum: {max_stalls}\n"
        f"Severe (>50): {severe_count} routers\n"
        f"Moderate (10-50): {moderate_count} routers"
    )

    ax.text(
        0.98,
        0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )

    # Add value labels for each point (only show >10)
    for i, (r, s) in enumerate(zip(routers, stalls)):
        if s > 10:
            ax.text(
                r,
                s + 5,
                str(s),
                ha="center",
                va="bottom",
                fontsize=10,
                color="darkblue",
            )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Line chart saved: {output_file}")
    plt.close()


def create_combined_view(
    router_data, output_file="noc_router_combined.png", mesh_cols=4, mesh_rows=5
):
    """Create combined view: heatmap + line chart

    Args:
        router_data: Dictionary of router_id -> stalls
        output_file: Output filename
        mesh_cols: Number of columns in actual Mesh (default 4)
        mesh_rows: Number of rows in actual Mesh (default 5)
    """
    num_routers = len(router_data)

    # Use actual Mesh dimensions
    cols = mesh_cols
    rows = mesh_rows

    fig = plt.figure(figsize=(18, 8))

    # Left: Heatmap
    ax1 = plt.subplot(1, 2, 1)

    # Create grid data
    grid = np.zeros((rows, cols))
    grid[:] = np.nan

    for router_num, stalls in router_data.items():
        row = router_num // cols
        col = router_num % cols
        if row < rows and col < cols:
            grid[row, col] = stalls

    # Draw heatmap
    masked_grid = np.ma.masked_invalid(grid)
    im = ax1.imshow(masked_grid, cmap="YlOrRd", aspect="auto")

    # Set axes
    ax1.set_xticks(np.arange(cols))
    ax1.set_yticks(np.arange(rows))
    ax1.set_xticklabels([f"{i}" for i in range(cols)], fontsize=13)
    ax1.set_yticklabels([f"{i}" for i in range(rows)], fontsize=13)

    # Display values in each cell
    max_stalls = max(router_data.values()) if router_data else 1
    for i in range(rows):
        for j in range(cols):
            router_num = i * cols + j
            if router_num in router_data:
                stalls = int(grid[i, j])
                text_color = "white" if stalls > max_stalls * 0.5 else "black"
                ax1.text(
                    j,
                    i,
                    f"rtr_{router_num}\n{stalls}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=16,
                    weight="bold",
                )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Stalls", rotation=270, labelpad=15, fontsize=15)

    ax1.set_title(f"Heatmap ({cols}×{rows} Mesh)", fontsize=16, weight="bold", pad=15)

    # Add grid lines
    ax1.set_xticks(np.arange(cols) - 0.5, minor=True)
    ax1.set_yticks(np.arange(rows) - 0.5, minor=True)
    ax1.grid(which="minor", color="gray", linestyle="-", linewidth=2)

    # Right: Line chart
    ax2 = plt.subplot(1, 2, 2)

    routers = sorted(router_data.keys())
    stalls = [router_data[r] for r in routers]

    # Draw line chart
    ax2.plot(
        routers,
        stalls,
        marker="o",
        linewidth=2,
        markersize=7,
        color="#2E86AB",
        markerfacecolor="#A23B72",
        markeredgewidth=2,
        markeredgecolor="#2E86AB",
    )

    # Annotate maximum point
    max_idx = stalls.index(max(stalls))
    max_router = routers[max_idx]
    max_stalls = stalls[max_idx]
    ax2.annotate(
        f"rtr_{max_router}\n{max_stalls}",
        xy=(max_router, max_stalls),
        xytext=(max_router, max_stalls + 60),
        ha="center",
        fontsize=9,
        weight="bold",
        color="red",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
    )

    # Average line
    avg_stalls = np.mean(stalls)
    ax2.axhline(
        y=avg_stalls,
        color="green",
        linestyle="--",
        linewidth=1.5,
        label=f"Avg: {avg_stalls:.1f}",
        alpha=0.7,
    )

    # Threshold line
    ax2.axhline(
        y=50,
        color="red",
        linestyle=":",
        linewidth=1.5,
        label="Threshold: 50",
        alpha=0.7,
    )

    ax2.set_xlabel("Router Number", fontsize=13, weight="bold")
    ax2.set_ylabel("Crossbar Stalls", fontsize=13, weight="bold")
    ax2.set_title("Stalls Distribution Line Chart", fontsize=14, weight="bold", pad=15)

    ax2.set_xticks(routers)
    ax2.set_xticklabels([f"{r}" for r in routers], fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.legend(loc="upper left", fontsize=11)

    # Overall title
    fig.suptitle(
        f"NoC Router Bottleneck Analysis - {cols}×{rows} Mesh ({len(routers)} Routers)",
        fontsize=14,
        weight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Combined view saved: {output_file}")
    plt.close()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Visualize NoC router metrics from SST CSV stats"
    )
    parser.add_argument(
        "--input-file",
        default="stats_selfcom.txt",
        help="Input stats CSV file path (default: stats_selfcom.txt)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for generated images (default: current dir)",
    )
    parser.add_argument(
        "--output-prefix",
        default="noc_router",
        help="Output filename prefix (default: noc_router)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("NoC Router Visualization Tool")
    print("=" * 60)

    # Parse data
    input_file = args.input_file
    output_dir = args.output_dir
    output_prefix = args.output_prefix
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nParsing file: {input_file}")

    router_data = parse_router_stats(input_file, metric="xbar_stalls", max_routers=None)

    if not router_data:
        print("Error: No router data found")
        return

    print(f"✓ Successfully parsed {len(router_data)} routers")

    # Display statistics
    total_stalls = sum(router_data.values())
    avg_stalls = total_stalls / len(router_data)
    max_router = max(router_data.items(), key=lambda x: x[1])

    print(f"\nStatistics Summary:")
    print(f"  Total Stalls: {total_stalls}")
    print(f"  Average Stalls: {avg_stalls:.1f}")
    print(f"  Max Bottleneck: rtr_{max_router[0]} ({max_router[1]} stalls)")

    # Detect Mesh dimensions from parsed router ids
    router_ids = sorted(router_data.keys())
    mesh_cols, mesh_rows = infer_mesh_dims(router_ids)
    if os.getenv("GOLEM_MESH_DIM_X", "").strip():
        print(
            f"  Detected Mesh topology (from GOLEM_MESH_DIM_X): {mesh_cols}×{mesh_rows}"
        )
    else:
        print(f"  Detected Mesh topology (auto): {mesh_cols}×{mesh_rows}")

    # Generate visualizations
    print(f"\nGenerating visualizations...")
    create_heatmap(
        router_data,
        output_file=os.path.join(output_dir, f"{output_prefix}_heatmap.png"),
        mesh_cols=mesh_cols,
        mesh_rows=mesh_rows,
        title_prefix="NoC Router Heatmap",
        value_label="Crossbar Stalls",
    )

    send_packet_data = parse_router_stats(
        input_file, metric="send_packet_count", max_routers=None
    )
    if send_packet_data:
        create_heatmap(
            send_packet_data,
            output_file=os.path.join(
                output_dir, f"{output_prefix}_send_packets_heatmap.png"
            ),
            mesh_cols=mesh_cols,
            mesh_rows=mesh_rows,
            title_prefix="NoC Router Heatmap",
            value_label="Send Packet Count",
        )
    create_line_chart(
        router_data,
        output_file=os.path.join(output_dir, f"{output_prefix}_stalls_line.png"),
    )
    create_combined_view(
        router_data,
        output_file=os.path.join(output_dir, f"{output_prefix}_combined.png"),
        mesh_cols=mesh_cols,
        mesh_rows=mesh_rows,
    )

    print(f"\n" + "=" * 60)
    print("✓ All visualizations completed!")
    print("=" * 60)
    print("\nGenerated files:")
    print(
        f"  1. {os.path.join(output_dir, f'{output_prefix}_heatmap.png')} - {mesh_cols}×{mesh_rows} Mesh heatmap"
    )
    print(
        f"  2. {os.path.join(output_dir, f'{output_prefix}_send_packets_heatmap.png')} - {mesh_cols}×{mesh_rows} Send Packet Count heatmap"
    )
    print(
        f"  3. {os.path.join(output_dir, f'{output_prefix}_stalls_line.png')} - Line chart (all {len(router_data)} routers)"
    )
    print(
        f"  4. {os.path.join(output_dir, f'{output_prefix}_combined.png')} - Combined view ({mesh_cols}×{mesh_rows} Mesh)"
    )
    print(
        f"\nMesh Topology: {mesh_cols} columns × {mesh_rows} rows = {len(router_data)} routers"
    )
    print(f"Router range: rtr_0 ~ rtr_{max(router_data.keys())}")
    print()


if __name__ == "__main__":
    main()
