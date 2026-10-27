#!/usr/bin/env python3
"""Compare same-shape 4:12 and 8:8 SST worker-cluster measurements."""

import argparse
import json
from pathlib import Path


FLOORS = {
    "4qk_12pv": {
        "qk_wcp": 33792,
        "sfu": 65536,
        "pv_wcp": 12672,
        "pv_local_sram": 12480,
    },
    "8qk_8pv": {
        "qk_wcp": 16896,
        "sfu": 32768,
        "pv_wcp": 16896,
        "pv_local_sram": 16640,
    },
}


def load(path: Path) -> dict:
    result = json.loads(path.read_text())
    if result.get("status") != "PASS":
        raise ValueError(f"{path} is not a passing SST result")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-4qk-12pv", type=Path, required=True)
    parser.add_argument("--experiment-8qk-8pv", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = {
        "4qk_12pv": load(args.baseline_4qk_12pv),
        "8qk_8pv": load(args.experiment_8qk_8pv),
    }
    variants = {}
    for name, measured in results.items():
        floor = max(FLOORS[name].values())
        actual = measured["end_to_end_cycles"]
        variants[name] = {
            "actual_end_to_end_cycles": actual,
            "resource_floors": FLOORS[name],
            "theoretical_end_to_end_floor": floor,
            "actual_over_floor_cycles": actual - floor,
            "actual_over_floor_ratio": actual / floor,
            "qk_cores": measured["qk_cores"],
            "pv_cores": measured["pv_cores"],
        }
    old = variants["4qk_12pv"]["actual_end_to_end_cycles"]
    new = variants["8qk_8pv"]["actual_end_to_end_cycles"]
    report = {
        "shape": {"Hq": 4, "Hkv": 2, "Sq": 1024, "Skv": 1024, "Dh": 128},
        "variants": variants,
        "actual_cycle_delta_8qk_minus_4qk": new - old,
        "actual_speedup_8qk_over_4qk": old / new,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
