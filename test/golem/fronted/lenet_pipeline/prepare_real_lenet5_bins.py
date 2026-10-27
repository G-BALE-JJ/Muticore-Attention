#!/usr/bin/env python3
import argparse
import os
import struct
from pathlib import Path
from typing import List

import onnx
from onnx import numpy_helper

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'configure.ac').is_file())
DEFAULT_TEST_ROOT = REPO_ROOT / "test/golem"


def read_f32_file(path: str) -> List[float]:
    raw = open(path, "rb").read()
    if len(raw) % 4 != 0:
        raise ValueError(f"file size not multiple of 4: {path}")
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def write_f32_file(path: str, values: List[float]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}f", *values))


def conv1_weight_to_kn(weight_6x25: List[List[float]]) -> List[float]:
    out = []
    for k in range(64):
        for oc in range(6):
            out.append(weight_6x25[oc][k] if k < 25 else 0.0)
    return out


def conv2_bpack(weight_16x150: List[List[float]]) -> List[float]:
    # KN padded to 192x16, packed by (k_tile, n_col, kk)
    w_kn = [[0.0 for _ in range(16)] for _ in range(192)]
    for oc in range(16):
        for k in range(150):
            w_kn[k][oc] = weight_16x150[oc][k]
    out = []
    for k_tile in range(3):
        k_base = k_tile * 64
        for n_col in range(16):
            for kk in range(64):
                out.append(w_kn[k_base + kk][n_col])
    return out


def fc1_wslice(weight_256x120: List[List[float]]) -> List[float]:
    out = []
    for task_id in range(4):
        for chunk in range(2):
            out_base = chunk * 64
            for out_r in range(64):
                out_idx = out_base + out_r
                for local_k in range(64):
                    oc = local_k // 4
                    pw = local_k % 4
                    global_k = oc * 16 + task_id * 4 + pw
                    v = weight_256x120[global_k][out_idx] if out_idx < 120 else 0.0
                    out.append(v)
    return out


def fc_singlecore_wpack(
    weight_in_out: List[List[float]], in_len: int, out_len: int
) -> List[float]:
    out = []
    for out_chunk in range(2):
        out_base = out_chunk * 64
        for in_chunk in range(2):
            in_base = in_chunk * 64
            for out_r in range(64):
                out_idx = out_base + out_r
                for col in range(64):
                    in_idx = in_base + col
                    v = (
                        weight_in_out[in_idx][out_idx]
                        if (in_idx < in_len and out_idx < out_len)
                        else 0.0
                    )
                    out.append(v)
    return out


def conv1_a_banded_from_image(image_784: List[float]) -> List[float]:
    if len(image_784) != 28 * 28:
        raise ValueError("image must be 28x28 fp32")
    img = [image_784[r * 28 : (r + 1) * 28] for r in range(28)]
    rows: List[List[float]] = []
    for band in range(12):
        oh0 = band * 2
        band_rows: List[List[float]] = []
        for oh in range(oh0, oh0 + 2):
            for ow in range(24):
                row = []
                for kh in range(5):
                    for kw in range(5):
                        row.append(img[oh + kh][ow + kw])
                band_rows.append(row)
        for _ in range(16):
            band_rows.append([0.0 for _ in range(25)])
        rows.extend(band_rows)

    # pad K 25->64
    flat = []
    for r in rows:
        rr = r + [0.0 for _ in range(64 - 25)]
        flat.extend(rr)
    return flat


def load_weights_from_onnx(dataset_dir: str):
    model = onnx.load(os.path.join(dataset_dir, "lenet5.onnx"))
    tensors = {
        t.name: numpy_helper.to_array(t).astype("float32")
        for t in model.graph.initializer
    }

    conv1_w = tensors["conv1.weight"]  # [6,1,5,5]
    conv1_b = tensors["conv1.bias"].reshape(-1)
    conv2_w = tensors["conv2.weight"]  # [16,6,5,5]
    conv2_b = tensors["conv2.bias"].reshape(-1)
    fc1_w = tensors["fc1.weight"]  # [120,256]
    fc1_b = tensors["fc1.bias"].reshape(-1)
    fc2_w = tensors["fc2.weight"]  # [84,120]
    fc2_b = tensors["fc2.bias"].reshape(-1)
    fc3_w = tensors["fc3.weight"]  # [10,84]
    fc3_b = tensors["fc3.bias"].reshape(-1)

    weight_6x25 = [conv1_w[oc, 0].reshape(-1).tolist() for oc in range(6)]
    weight_16x150 = [conv2_w[oc].reshape(-1).tolist() for oc in range(16)]

    # FC reference path uses [in][out]
    weight_256x120 = [[float(fc1_w[o, i]) for o in range(120)] for i in range(256)]
    weight_120x84 = [[float(fc2_w[o, i]) for o in range(84)] for i in range(120)]
    weight_84x10 = [[float(fc3_w[o, i]) for o in range(10)] for i in range(84)]

    return (
        weight_6x25,
        conv1_b.tolist(),
        weight_16x150,
        conv2_b.tolist(),
        weight_256x120,
        fc1_b.tolist(),
        weight_120x84,
        fc2_b.tolist(),
        weight_84x10,
        fc3_b.tolist(),
    )


def load_weights_from_flat_file(dataset_dir: str):
    w_all = read_f32_file(os.path.join(dataset_dir, "lenet5_fp32.weights"))
    w = w_all[5:]
    off = 0
    conv1_b = w[off : off + 6]
    off += 6
    conv1_w = w[off : off + 150]
    off += 150
    conv2_b = w[off : off + 16]
    off += 16
    conv2_w = w[off : off + 2400]
    off += 2400
    fc1_b = w[off : off + 120]
    off += 120
    fc1_w = w[off : off + 30720]
    off += 30720
    fc2_b = w[off : off + 84]
    off += 84
    fc2_w = w[off : off + 10080]
    off += 10080
    fc3_b = w[off : off + 10]
    off += 10
    fc3_w = w[off : off + 840]

    weight_6x25 = [conv1_w[i * 25 : (i + 1) * 25] for i in range(6)]
    weight_16x150 = [conv2_w[i * 150 : (i + 1) * 150] for i in range(16)]
    weight_256x120 = [fc1_w[i * 120 : (i + 1) * 120] for i in range(256)]
    weight_120x84 = [fc2_w[i * 84 : (i + 1) * 84] for i in range(120)]
    weight_84x10 = [fc3_w[i * 10 : (i + 1) * 10] for i in range(84)]

    return (
        weight_6x25,
        conv1_b,
        weight_16x150,
        conv2_b,
        weight_256x120,
        fc1_b,
        weight_120x84,
        fc2_b,
        weight_84x10,
        fc3_b,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare real LeNet5 bins for golem pipeline"
    )
    ap.add_argument(
        "--dataset-dir",
        default=str(DEFAULT_TEST_ROOT / "task/task lenet5"),
        help="Directory containing lenet5.onnx(.data) and input/imageX.bin",
    )
    ap.add_argument("--image-index", type=int, default=0, help="Input image index 0..9")
    ap.add_argument(
        "--out-dir",
        default=str(DEFAULT_TEST_ROOT / "data/real_lenet5"),
        help="Output directory for generated bins",
    )
    ap.add_argument(
        "--weights-source",
        choices=["onnx", "flat"],
        default="onnx",
        help="Weight source: onnx (default/canonical) or flat compatibility file",
    )
    args = ap.parse_args()

    image_path = os.path.join(args.dataset_dir, "input", f"image{args.image_index}.bin")

    if args.weights_source == "onnx":
        (
            conv1_w_6x25,
            conv1_b,
            conv2_w_16x150,
            conv2_b,
            fc1_w_256x120,
            fc1_b,
            fc2_w_120x84,
            fc2_b,
            fc3_w_84x10,
            fc3_b,
        ) = load_weights_from_onnx(args.dataset_dir)
    else:
        print("[WARN] using flat weight compatibility mode instead of ONNX truth")
        (
            conv1_w_6x25,
            conv1_b,
            conv2_w_16x150,
            conv2_b,
            fc1_w_256x120,
            fc1_b,
            fc2_w_120x84,
            fc2_b,
            fc3_w_84x10,
            fc3_b,
        ) = load_weights_from_flat_file(args.dataset_dir)

    image = read_f32_file(image_path)
    a_banded = conv1_a_banded_from_image(image)

    write_f32_file(os.path.join(args.out_dir, "a_conv1_banded_768x64.bin"), a_banded)
    write_f32_file(
        os.path.join(args.out_dir, "b_conv1_kn_64x6.bin"),
        conv1_weight_to_kn(conv1_w_6x25),
    )
    write_f32_file(os.path.join(args.out_dir, "bias_conv1_6.bin"), conv1_b)
    write_f32_file(
        os.path.join(args.out_dir, "conv2_bpack_3x16x64.bin"),
        conv2_bpack(conv2_w_16x150),
    )
    write_f32_file(os.path.join(args.out_dir, "bias_conv2_16.bin"), conv2_b)
    write_f32_file(
        os.path.join(args.out_dir, "fc1_wslice_4x2x64x64.bin"),
        fc1_wslice(fc1_w_256x120),
    )
    write_f32_file(os.path.join(args.out_dir, "bias_fc1_120.bin"), fc1_b)
    write_f32_file(
        os.path.join(args.out_dir, "fc2_wpack_2x2x64x64.bin"),
        fc_singlecore_wpack(fc2_w_120x84, 120, 84),
    )
    write_f32_file(os.path.join(args.out_dir, "bias_fc2_84.bin"), fc2_b)
    write_f32_file(
        os.path.join(args.out_dir, "fc3_wpack_2x2x64x64.bin"),
        fc_singlecore_wpack(fc3_w_84x10, 84, 10),
    )
    write_f32_file(os.path.join(args.out_dir, "bias_fc3_10.bin"), fc3_b)

    print("[OK] generated bins in", args.out_dir)


if __name__ == "__main__":
    main()
