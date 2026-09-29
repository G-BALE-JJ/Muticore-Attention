#!/usr/bin/env python3
"""Check every striped tensor and panel against its source bytes."""

import argparse
import json
from pathlib import Path

import numpy as np


def verify_layout(q_file, k_file, v_file, hbm_dir, query_length, kv_length,
                  hq, hkv, head_dim, dtype, offsets):
    if query_length % 256 or kv_length % 256 or head_dim % 64:
        raise ValueError("panel layout requires lengths divisible by 256 and D by 64")
    element = np.dtype("<f2" if dtype == "fp16" else "<f4")
    elem_bytes = element.itemsize
    tensors = {name: np.fromfile(path, dtype=element).reshape(heads, rows, head_dim)
               for name, path, heads, rows in (("Q", q_file, hq, query_length),
                   ("K", k_file, hkv, kv_length), ("V", v_file, hkv, kv_length))}
    windows = kv_length // 256
    windows_per_node = (windows + 3) // 4
    panel_bytes = 64 * 64 * elem_bytes
    checked = 0
    nodes = []
    for band in range(4):
        path = Path(hbm_dir) / f"hbm_init_node{band + 1}.bin"
        capacity = path.stat().st_size
        regions = []
        with path.open("rb") as handle:
            def check(name, offset, expected):
                nonlocal checked
                handle.seek(offset)
                if handle.read(len(expected)) != expected:
                    raise ValueError(f"{path}: {name} bytes differ at {offset:#x}")
                checked += len(expected)
                regions.append((name, offset, offset + len(expected)))

            for name in ("Q", "K", "V"):
                tensor = tensors[name]
                rows = tensor.shape[1] // 4
                shard = tensor[:, band * rows:(band + 1) * rows]
                check(name, offsets[name], shard.tobytes())
                if name != "V":
                    panels = shard.reshape(tensor.shape[0], rows // 64, 64,
                        head_dim // 64, 64).transpose(0, 1, 3, 2, 4)
                    check(name + "_panels", 0x04000000 if name == "Q" else 0x05000000,
                          panels.tobytes())
            output_bytes = hq * (query_length // 4) * head_dim * elem_bytes
            regions.append(("O", offsets["O"], offsets["O"] + output_bytes))
            begin = band * windows_per_node
            count = max(0, min(windows_per_node, windows - begin))
            v_panels = bytearray(hkv * windows_per_node * 4 * (head_dim // 64) * panel_bytes)
            if count:
                selected = tensors["V"][:, begin * 256:(begin + count) * 256]
                panels = selected.reshape(hkv, count, 4, 64, head_dim // 64, 64)
                panels = panels.transpose(0, 1, 4, 2, 5, 3)
                head_bytes = windows_per_node * 4 * (head_dim // 64) * panel_bytes
                for head in range(hkv):
                    data = panels[head].tobytes()
                    v_panels[head * head_bytes:head * head_bytes + len(data)] = data
            check("V_panels", 0x06000000, bytes(v_panels))
        regions.sort(key=lambda entry: entry[1])
        for name, begin, end in regions:
            if begin < 0 or end > capacity:
                raise ValueError(f"{path}: {name} exceeds capacity")
        for left, right in zip(regions, regions[1:]):
            if left[2] > right[1]:
                raise ValueError(f"{path}: {left[0]} overlaps {right[0]}")
        nodes.append({"node": band + 1, "capacity_bytes": capacity,
                      "regions": [{"name": n, "begin": b, "end": e} for n, b, e in regions]})
    return {"status": "PASS", "dtype": dtype, "element_bytes": elem_bytes,
            "checked_bytes": checked, "nodes": nodes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("q-file", "k-file", "v-file", "hbm-dir", "result-json"):
        parser.add_argument("--" + name, required=True)
    for name in ("query-length", "kv-length", "num-query-heads", "num-kv-heads", "head-dim"):
        parser.add_argument("--" + name, type=int, required=True)
    for name in "qkvo":
        parser.add_argument(f"--{name}-offset", type=lambda value: int(value, 0), required=True)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    args = parser.parse_args()
    try:
        result = verify_layout(args.q_file, args.k_file, args.v_file, args.hbm_dir,
            args.query_length, args.kv_length, args.num_query_heads, args.num_kv_heads,
            args.head_dim, args.dtype, {name.upper(): getattr(args, f"{name}_offset") for name in "qkvo"})
    except (OSError, ValueError) as error:
        result = {"status": "FAIL", "error": str(error)}
    Path(args.result_json).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
