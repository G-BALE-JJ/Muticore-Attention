#!/usr/bin/env python3

import json
import os
import re
import struct


FC3_OUT = 10

# Keep consistent with small/lenet5/lenet5_layout.h and gen_lenet5_hbm_init.py
LAYER_BASE = 0x01000000
INPUT_OFF = LAYER_BASE
CONV1_OFF = INPUT_OFF + 0x00002000
POOL1_OFF = CONV1_OFF + 0x00004000
CONV2_OFF = POOL1_OFF + 0x00002000
POOL2_OFF = CONV2_OFF + 0x00002000
FC1_OFF = POOL2_OFF + 0x00001000
FC2_OFF = FC1_OFF + 0x00001000
FC3_OFF = FC2_OFF + 0x00001000


def read_logits_from_hbm_file(path: str, offset: int):
    need = FC3_OUT * 4
    with open(path, "rb") as f:
        f.seek(offset)
        raw = f.read(need)
    if len(raw) != need:
        raise ValueError(
            f"file too short when reading logits: need={need}, got={len(raw)}, file={path}, off=0x{offset:x}"
        )
    return list(struct.unpack("<10f", raw))


def infer_expected_label_from_input_path(input_path: str):
    if not input_path:
        return None
    base = os.path.basename(input_path)
    m = re.search(r"image(\d+)\.bin$", base)
    if not m:
        return None
    value = int(m.group(1))
    if value < 0 or value > 9:
        return None
    return value


def main():
    tests_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifact_root = os.getenv(
        "GOLEM_ARTIFACT_ROOT", os.path.join(tests_dir, "artifacts")
    )
    hbm_dir = os.getenv("GOLEM_HBM_DIR", os.path.join(artifact_root, "hbm"))

    node_idx = int(os.getenv("LENET5_OUTPUT_NODE", "1"))
    expected_label = infer_expected_label_from_input_path(os.getenv("LENET5_INPUT", ""))
    dump_json = os.getenv("LENET5_VERIFY_JSON", "")

    hbm_out_file = os.path.join(hbm_dir, f"hbm_out_node{node_idx}.bin")

    logits = read_logits_from_hbm_file(hbm_out_file, FC3_OFF)
    top1 = max(range(FC3_OUT), key=lambda i: logits[i])

    print(f"[LENET-VERIFY] file={hbm_out_file}")
    print(f"[LENET-VERIFY] fc3_off=0x{FC3_OFF:08x}")
    print("[LENET-VERIFY] logits=" + " ".join(f"{v:.6f}" for v in logits))
    print(f"[LENET-VERIFY] top1={top1}")

    if dump_json:
        os.makedirs(os.path.dirname(os.path.abspath(dump_json)), exist_ok=True)
        with open(dump_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "hbm_out_file": hbm_out_file,
                    "fc3_off": FC3_OFF,
                    "logits": logits,
                    "top1": top1,
                    "expected_label": expected_label,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    if expected_label is not None:
        if top1 != expected_label:
            print(
                f"[LENET-VERIFY] FAIL expected_label={expected_label}, got_top1={top1}"
            )
            return 1
        print("[LENET-VERIFY] PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
