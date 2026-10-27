import os
import struct
from typing import List, Tuple

import onnx
from onnx import numpy_helper

from .plan import REAL_DATASET_DIR


def _read_f32_bin(path: str) -> List[float]:
    raw = open(path, "rb").read()
    if len(raw) % 4 != 0:
        raise ValueError(f"invalid fp32 binary size: {path}")
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def load_real_lenet5_case(
    dataset_dir: str = REAL_DATASET_DIR,
    image_index: int = 0,
    weights_source: str = "onnx",
) -> Tuple[
    List[List[float]],
    List[List[float]],
    List[float],
    List[List[float]],
    List[float],
    List[List[float]],
    List[float],
    List[List[float]],
    List[float],
    List[List[float]],
    List[float],
]:
    image = _read_f32_bin(os.path.join(dataset_dir, "input", f"image{image_index}.bin"))
    input_28x28 = [image[r * 28 : (r + 1) * 28] for r in range(28)]

    if weights_source == "onnx":
        model = onnx.load(os.path.join(dataset_dir, "lenet5.onnx"))
        tensors = {
            t.name: numpy_helper.to_array(t).astype("float32")
            for t in model.graph.initializer
        }

        conv1_w = tensors["conv1.weight"]  # [6,1,5,5]
        conv1_b = tensors["conv1.bias"].reshape(-1).tolist()
        conv2_w = tensors["conv2.weight"]  # [16,6,5,5]
        conv2_b = tensors["conv2.bias"].reshape(-1).tolist()
        fc1_w = tensors["fc1.weight"]  # [120,256]
        fc1_b = tensors["fc1.bias"].reshape(-1).tolist()
        fc2_w = tensors["fc2.weight"]  # [84,120]
        fc2_b = tensors["fc2.bias"].reshape(-1).tolist()
        fc3_w = tensors["fc3.weight"]  # [10,84]
        fc3_b = tensors["fc3.bias"].reshape(-1).tolist()

        weight_6x25 = [conv1_w[oc, 0].reshape(-1).tolist() for oc in range(6)]
        weight_16x150 = [conv2_w[oc].reshape(-1).tolist() for oc in range(16)]
        weight_256x120 = [[float(fc1_w[o, i]) for o in range(120)] for i in range(256)]
        weight_120x84 = [[float(fc2_w[o, i]) for o in range(84)] for i in range(120)]
        weight_84x10 = [[float(fc3_w[o, i]) for o in range(10)] for i in range(84)]
    elif weights_source == "flat":
        w_all = _read_f32_bin(os.path.join(dataset_dir, "lenet5_fp32.weights"))
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
    else:
        raise ValueError(
            f"unsupported weights_source={weights_source}, expected onnx|flat"
        )

    return (
        input_28x28,
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


def conv1_ref_direct(input_28x28, weight_6x25, bias_6):
    conv_relu = [[[0.0 for _ in range(24)] for _ in range(24)] for _ in range(6)]
    for oc in range(6):
        for oh in range(24):
            for ow in range(24):
                acc = float(bias_6[oc])
                for kh in range(5):
                    for kw in range(5):
                        acc += float(input_28x28[oh + kh][ow + kw]) * float(
                            weight_6x25[oc][kh * 5 + kw]
                        )
                conv_relu[oc][oh][ow] = acc if acc > 0.0 else 0.0
    pool = [[[0.0 for _ in range(12)] for _ in range(12)] for _ in range(6)]
    for oc in range(6):
        for ph in range(12):
            for pw in range(12):
                bh, bw = ph * 2, pw * 2
                m = conv_relu[oc][bh][bw]
                m = max(
                    m,
                    conv_relu[oc][bh][bw + 1],
                    conv_relu[oc][bh + 1][bw],
                    conv_relu[oc][bh + 1][bw + 1],
                )
                pool[oc][ph][pw] = m
    return pool


def conv2_ref_direct(pool1_6x12x12, weight_16x150, bias_16):
    conv_relu = [[[0.0 for _ in range(8)] for _ in range(8)] for _ in range(16)]
    for oc in range(16):
        for oh in range(8):
            for ow in range(8):
                acc = float(bias_16[oc])
                for ic in range(6):
                    for kh in range(5):
                        for kw in range(5):
                            k = (ic * 5 + kh) * 5 + kw
                            acc += float(pool1_6x12x12[ic][oh + kh][ow + kw]) * float(
                                weight_16x150[oc][k]
                            )
                conv_relu[oc][oh][ow] = acc if acc > 0.0 else 0.0
    pool2 = [[[0.0 for _ in range(4)] for _ in range(4)] for _ in range(16)]
    for oc in range(16):
        for ph in range(4):
            for pw in range(4):
                bh, bw = ph * 2, pw * 2
                m = conv_relu[oc][bh][bw]
                m = max(
                    m,
                    conv_relu[oc][bh][bw + 1],
                    conv_relu[oc][bh + 1][bw],
                    conv_relu[oc][bh + 1][bw + 1],
                )
                pool2[oc][ph][pw] = m
    return pool2


def fc1_ref_direct(pool2_16x4x4, weight_256x120, bias_120):
    x = []
    for oc in range(16):
        for ph in range(4):
            for pw in range(4):
                x.append(float(pool2_16x4x4[oc][ph][pw]))
    out = [0.0 for _ in range(120)]
    for n in range(120):
        acc = float(bias_120[n])
        for k in range(256):
            acc += x[k] * float(weight_256x120[k][n])
        out[n] = acc if acc > 0.0 else 0.0
    return out


def fc2_ref_direct(fc1_120, weight_120x84, bias_84):
    out = [0.0 for _ in range(84)]
    for n in range(84):
        acc = float(bias_84[n])
        for k in range(120):
            acc += float(fc1_120[k]) * float(weight_120x84[k][n])
        out[n] = acc if acc > 0.0 else 0.0
    return out


def fc3_ref_direct(fc2_84, weight_84x10, bias_10):
    out = [0.0 for _ in range(10)]
    for n in range(10):
        acc = float(bias_10[n])
        for k in range(84):
            acc += float(fc2_84[k]) * float(weight_84x10[k][n])
        out[n] = acc
    return out


def conv_max_abs_diff(a, b) -> float:
    m = 0.0
    for c in range(len(a)):
        for h in range(len(a[c])):
            for w in range(len(a[c][h])):
                d = abs(float(a[c][h][w]) - float(b[c][h][w]))
                if d > m:
                    m = d
    return m


def vec_max_abs_diff(a, b) -> float:
    return max(abs(float(x) - float(y)) for x, y in zip(a, b)) if a else 0.0
