#!/usr/bin/env python3
"""Check every striped tensor and panel against its source bytes."""

import argparse
import json
from pathlib import Path

import numpy as np


def verify_layout(q_file, k_file, v_file, hbm_dir, query_length, kv_length,
                  hq, hkv, head_dim, dtype, offsets, projection=False):
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
        source = "hbm_out" if projection else "hbm_init"
        path = Path(hbm_dir) / f"{source}_node{band + 1}.bin"
        capacity = path.stat().st_size
        regions = []
        with path.open("rb") as handle:
            def check(name, offset, expected, numeric=False):
                nonlocal checked
                handle.seek(offset)
                actual = handle.read(len(expected))
                if numeric:
                    found = np.frombuffer(actual, dtype=element).astype(np.float32)
                    wanted = np.frombuffer(expected, dtype=element).astype(np.float32)
                    # Projection is an FP16 tiled reduction. Its NumPy golden
                    # uses the same storage format but a different vectorized
                    # reduction order, so compare with a bounded FP16 error.
                    tolerance = 8e-2 if projection else 5e-4
                    matches = found.size == wanted.size and np.allclose(
                        found, wanted, atol=tolerance, rtol=5e-3)
                else:
                    matches = actual == expected
                if not matches:
                    raise ValueError(f"{path}: {name} bytes differ at {offset:#x}")
                checked += len(expected)
                regions.append((name, offset, offset + len(expected)))
                return actual

            if projection:
                norm_file = Path(hbm_dir).parent / "projection_norm.bin"
                norm = np.fromfile(norm_file, dtype=element).reshape(query_length, hq * head_dim)
                norm_shard = norm[band * (query_length // 4):(band + 1) * (query_length // 4)]
                check("RMSNorm", 0x01100000, norm_shard.tobytes(), numeric=True)

            for name in ("Q", "K", "V"):
                tensor = tensors[name]
                rows = tensor.shape[1] // 4
                shard = tensor[:, band * rows:(band + 1) * rows]
                actual = check(name, offsets[name], shard.tobytes(), numeric=projection)
                if projection:
                    shard = np.frombuffer(actual, dtype=element).reshape(shard.shape)
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
                if projection:
                    handle.seek(offsets["V"])
                    actual_v = handle.read(hkv * (kv_length // 4) * head_dim * elem_bytes)
                    selected = np.frombuffer(actual_v, dtype=element).reshape(
                        hkv, kv_length // 4, head_dim)[:, :count * 256]
                else:
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
    parser.add_argument("--projection", action="store_true")
    args = parser.parse_args()
    try:
        result = verify_layout(args.q_file, args.k_file, args.v_file, args.hbm_dir,
            args.query_length, args.kv_length, args.num_query_heads, args.num_kv_heads,
            args.head_dim, args.dtype, {name.upper(): getattr(args, f"{name}_offset") for name in "qkvo"},
            projection=args.projection)
    except (OSError, ValueError) as error:
        result = {"status": "FAIL", "error": str(error)}
    Path(args.result_json).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
