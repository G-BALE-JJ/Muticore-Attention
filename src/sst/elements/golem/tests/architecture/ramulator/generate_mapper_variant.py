#!/usr/bin/env python3
import argparse
from pathlib import Path

import yaml


SUPPORTED_MAPPERS = {"MOP4CLXOR", "RoBaRaCoCh", "ChRaBaRoCo"}


def main():
    parser = argparse.ArgumentParser(
        description="Generate an HBM2E config using an official Ramulator2 address mapper."
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--mapper", choices=sorted(SUPPORTED_MAPPERS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.base.read_text())
    controllers = config["memory_system"]["controllers"]
    if len(controllers) != 8:
        raise RuntimeError(f"expected 8 HBM channels, found {len(controllers)}")
    for controller in controllers:
        controller["addr_mapper"] = {"impl": args.mapper}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False))
    print(args.output)


if __name__ == "__main__":
    main()
