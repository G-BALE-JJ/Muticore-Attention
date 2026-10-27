#!/usr/bin/env python3
import argparse
import csv
import html
import math
from pathlib import Path


WIDTH, HEIGHT = 1800, 1020
LEFT, RIGHT = 150, 1700
TOP, BOTTOM = 90, 830
def value(row, key):
    try:
        parsed = float(row[key])
        return parsed if math.isfinite(parsed) else 0.0
    except (KeyError, TypeError, ValueError):
        return 0.0


def esc(text):
    return html.escape(str(text))


def nice_tick_step(span, max_ticks=10):
    rough = max(1.0, span / max_ticks)
    magnitude = 10 ** math.floor(math.log10(rough))
    for multiplier in (1, 2, 5, 10):
        candidate = multiplier * magnitude
        if candidate >= rough:
            return int(candidate)
    return int(10 * magnitude)


def main():
    parser = argparse.ArgumentParser(description="Plot the legacy first-tile/full-window timeline")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel-chunk", default="2 KiB")
    parser.add_argument("--title", default="GEMM window timeline")
    parser.add_argument("--total-groups", type=int, default=4)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.input.open(newline="")))
    if not rows:
        raise SystemExit("empty window breakdown")
    component_cores = {int(value(row, "core")) for row in rows}
    if args.total_groups <= 0 or len(component_cores) % args.total_groups != 0:
        raise SystemExit("worker core count must be divisible by --total-groups")
    workers_per_group = len(component_cores) // args.total_groups

    # SST component IDs remain row-major. Only the plotted IDs and row order
    # become group-major, so topology, routing, and task ownership are unchanged.
    def display_core_id(component_core):
        group_id = component_core % args.total_groups
        local_worker_id = component_core // args.total_groups - 1
        if not 0 <= local_worker_id < workers_per_group:
            raise SystemExit(
                f"component core {component_core} is outside the expected worker rows"
            )
        return group_id * workers_per_group + local_worker_id

    bound_core_ids = {core: display_core_id(core) for core in component_cores}
    cores = sorted(component_cores, key=lambda core: bound_core_ids[core])
    rows.sort(
        key=lambda row: (
            bound_core_ids[int(value(row, "core"))],
            int(value(row, "window")),
        )
    )

    barrier_cycles = [value(row, "barrier_cycle") for row in rows if value(row, "barrier_cycle") > 0]
    if not barrier_cycles:
        raise SystemExit("window breakdown has no barrier_cycle")
    barrier_origin = min(barrier_cycles)
    raw_max = max(max(value(row, "ready_cycle"), value(row, "compute_end_cycle")) for row in rows)
    relative_max = max(0.0, raw_max - barrier_origin)
    tick_step = nice_tick_step(relative_max)
    t_max = math.ceil(relative_max / tick_step) * tick_step
    span = max(1.0, t_max)

    def x(cycle):
        return LEFT + (cycle - barrier_origin) / span * (RIGHT - LEFT)

    row_step = (BOTTOM - TOP) / max(1, len(cores) - 1)
    by_core = {core: [] for core in cores}
    for row in rows:
        by_core[int(value(row, "core"))].append(row)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="timeline-title timeline-desc">',
        f'<title id="timeline-title">{esc(args.title)}</title>',
        f'<desc id="timeline-desc">Per-core GEMM window timeline with a panel chunk size of {esc(args.panel_chunk)}.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#202124;letter-spacing:0}.tick{font-size:24px}.legend{font-size:24px}.axis-label{font-size:28px;font-weight:600}.title{font-size:32px;font-weight:600}.grid{stroke:#e1e4e8;stroke-width:1}.group-separator{stroke:#b7bec7;stroke-width:1.5}.axis-line{stroke:#33383f;stroke-width:1.5}.cold{stroke:#0072b2;stroke-width:7;stroke-linecap:butt}.compute{stroke:#8c959f;stroke-width:9;stroke-linecap:butt;opacity:.72}.first{stroke:#9f3a00;stroke-width:1.2}.full{stroke:#111827;stroke-width:1.2}</style>',
        f'<text x="{WIDTH/2:.1f}" y="46" text-anchor="middle" class="title">{esc(args.title)}</text>',
    ]

    for i, core in enumerate(cores):
        yy = TOP + i * row_step
        bound_core = bound_core_ids[core]
        lines.append(f'<line x1="{LEFT}" y1="{yy:.1f}" x2="{RIGHT}" y2="{yy:.1f}" class="grid"/>')
        lines.append(f'<text x="140" y="{yy+6:.1f}" text-anchor="end" class="tick">{bound_core}</text>')

    for group in range(1, args.total_groups):
        separator_row = group * workers_per_group
        yy = TOP + (separator_row - 0.5) * row_step
        lines.append(
            f'<line x1="{LEFT}" y1="{yy:.1f}" x2="{RIGHT}" y2="{yy:.1f}" class="group-separator"/>'
        )

    for relative_cycle in range(0, int(t_max) + 1, tick_step):
        xx = x(barrier_origin + relative_cycle)
        lines.append(f'<line x1="{xx:.1f}" y1="{TOP}" x2="{xx:.1f}" y2="{BOTTOM}" class="grid"/>')
        lines.append(f'<text x="{xx:.1f}" y="866" text-anchor="middle" class="tick">{relative_cycle:,}</text>')
    lines.extend([
        f'<line x1="{LEFT}" y1="{BOTTOM}" x2="{RIGHT}" y2="{BOTTOM}" class="axis-line"/>',
        f'<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{BOTTOM}" class="axis-line"/>',
        f'<text x="{WIDTH/2:.1f}" y="910" text-anchor="middle" class="axis-label">Cycles since barrier</text>',
        f'<text x="32" y="{(TOP + BOTTOM) / 2:.1f}" text-anchor="middle" transform="rotate(-90 32 {(TOP + BOTTOM) / 2:.1f})" class="axis-label">Core ID</text>',
    ])

    for i, core in enumerate(cores):
        yy = TOP + i * row_step
        bound_core = bound_core_ids[core]
        first_row = by_core[core][0]
        barrier = value(first_row, "barrier_cycle")
        first_tick = value(first_row, "wcp_first_tick")
        if barrier > 0 and first_tick >= barrier:
            lines.append(
                f'<line x1="{x(barrier):.1f}" y1="{yy:.1f}" x2="{x(first_tick):.1f}" y2="{yy:.1f}" class="cold">'
                f'<title>core {bound_core}, descriptor cold start {int(first_tick - barrier)} cycles</title></line>'
            )
        for row in by_core[core]:
            window = int(value(row, "window"))
            first_tile = value(row, "first_tile_ready_cycle")
            full_ready = value(row, "ready_cycle")
            compute_start = value(row, "compute_start_cycle")
            compute_end = value(row, "compute_end_cycle")
            lines.append(f'<line x1="{x(compute_start):.1f}" y1="{yy:.1f}" x2="{x(compute_end):.1f}" y2="{yy:.1f}" class="compute"/>')
            tx = x(first_tile)
            lines.append(
                f'<polygon points="{tx:.1f},{yy-6:.1f} {tx-5:.1f},{yy+4:.1f} {tx+5:.1f},{yy+4:.1f}" fill="#e69f00" class="first">'
                f'<title>core {bound_core}, window {window}, first_tile_ready_cycle {int(first_tile)}</title></polygon>'
            )
            cx = x(full_ready)
            lines.append(
                f'<circle cx="{cx:.1f}" cy="{yy:.1f}" r="4.5" fill="#111827" class="full">'
                f'<title>core {bound_core}, window {window}, ready_cycle {int(full_ready)}</title></circle>'
            )

    lines.extend([
        '<line x1="190" y1="974" x2="234" y2="974" class="cold"/>',
        '<text x="250" y="981" class="legend">Descriptor cold start</text>',
        '<polygon points="650,967 644,979 656,979" fill="#e69f00" class="first"/>',
        '<text x="672" y="981" class="legend">First tile ready</text>',
        '<circle cx="1030" cy="974" r="5.5" fill="#111827" class="full"/>',
        '<text x="1050" y="981" class="legend">Full window ready</text>',
        '<line x1="1390" y1="974" x2="1434" y2="974" class="compute"/>',
        '<text x="1450" y="981" class="legend">Compute interval</text>',
    ])
    lines.append('</svg>')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
