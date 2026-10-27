#!/usr/bin/env python3

import os
import struct


TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LENET_DIR = os.path.join(TESTS_DIR, "task", "task lenet5")
ARTIFACT_ROOT = os.getenv("GOLEM_ARTIFACT_ROOT", os.path.join(TESTS_DIR, "artifacts"))
HBM_DIR = os.getenv("GOLEM_HBM_DIR", os.path.join(ARTIFACT_ROOT, "hbm"))
MEM_NODE_SIZE = int(os.getenv("GOLEM_MEM_NODE_SIZE_BYTES", str(128 * 1024 * 1024)))
NUM_MEMORY_NODES = int(os.getenv("GOLEM_NUM_MEMORY_NODES", "4"))
HBM_NODE_IDX = 1
DATA_NODE_IDXS = list(range(1, NUM_MEMORY_NODES))
HBM_ALIGN = 0x1000
GOLEM_DIM = int(os.getenv("GOLEM_DIM", "16"))

WEIGHTS_PATH = os.getenv(
    "LENET5_WEIGHTS",
    os.path.join(DEFAULT_LENET_DIR, "lenet5_fp32.weights"),
)
IMAGE_PATH = os.getenv(
    "LENET5_INPUT",
    os.path.join(DEFAULT_LENET_DIR, "input", "image8.bin"),
)

IMAGE_SIZE = 28 * 28
CONV1_OUT = 6
CONV2_OUT = 16
KERNEL = 5
FC1_OUT = 120
FC2_OUT = 84
FC3_OUT = 10

LAYER_BASE = 0x01000000
INPUT_OFF = LAYER_BASE
CONV1_OFF = INPUT_OFF + 0x00002000
POOL1_OFF = CONV1_OFF + 0x00004000
CONV2_OFF = POOL1_OFF + 0x00002000
POOL2_OFF = CONV2_OFF + 0x00002000
FC1_OFF = POOL2_OFF + 0x00001000
FC2_OFF = FC1_OFF + 0x00001000
FC3_OFF = FC2_OFF + 0x00001000
CONV1_IM2COL_OFF = FC3_OFF + 0x00001000
CONV2_IM2COL_OFF = CONV1_IM2COL_OFF + 0x00010000
PRELOAD_BASE = 0x01200000


def align_up(value: int, align: int) -> int:
    return ((value + align - 1) // align) * align


def read_f32_file(path: str, count: int) -> bytes:
    with open(path, "rb") as f:
        data = f.read()
    expect = count * 4
    if len(data) != expect:
        raise ValueError(f"{path} size mismatch: expected {expect}, got {len(data)}")
    return data


def read_weights(path: str):
    with open(path, "rb") as f:
        header = f.read(5 * 4)
        if len(header) != 20:
            raise ValueError("weights header too short")

        def take(count: int) -> bytes:
            data = f.read(count * 4)
            if len(data) != count * 4:
                raise ValueError(f"weights payload too short for count={count}")
            return data

        weights = {
            "b_conv1": take(CONV1_OUT),
            "w_conv1": take(CONV1_OUT * KERNEL * KERNEL),
            "b_conv2": take(CONV2_OUT),
            "w_conv2": take(CONV2_OUT * CONV1_OUT * KERNEL * KERNEL),
            "b_fc1": take(FC1_OUT),
            "w_fc1": take(FC1_OUT * 256),
            "b_fc2": take(FC2_OUT),
            "w_fc2": take(FC2_OUT * FC1_OUT),
            "b_fc3": take(FC3_OUT),
            "w_fc3": take(FC3_OUT * FC2_OUT),
        }
        extra = f.read()
        if extra:
            raise ValueError(
                f"weights file has unexpected trailing bytes: {len(extra)}"
            )

        # Pre-transpose + K-padding all weight matrices from [OUT, IN] to [K_PAD, N].
        # Runtime can consume weights directly without heap transpose/copy.
        weights["w_conv1"] = transpose_out_in_blob(
            weights["w_conv1"], CONV1_OUT, KERNEL * KERNEL
        )
        weights["w_conv1"] = pad_k_n_blob(
            weights["w_conv1"], KERNEL * KERNEL, CONV1_OUT, GOLEM_DIM
        )
        weights["w_conv2"] = transpose_out_in_blob(
            weights["w_conv2"], CONV2_OUT, CONV1_OUT * KERNEL * KERNEL
        )
        weights["w_conv2"] = pad_k_n_blob(
            weights["w_conv2"], CONV1_OUT * KERNEL * KERNEL, CONV2_OUT, GOLEM_DIM
        )
        weights["w_fc1"] = transpose_out_in_blob(weights["w_fc1"], FC1_OUT, 256)
        weights["w_fc1"] = pad_k_n_blob(weights["w_fc1"], 256, FC1_OUT, GOLEM_DIM)
        weights["w_fc2"] = transpose_out_in_blob(weights["w_fc2"], FC2_OUT, FC1_OUT)
        weights["w_fc2"] = pad_k_n_blob(weights["w_fc2"], FC1_OUT, FC2_OUT, GOLEM_DIM)
        weights["w_fc3"] = transpose_out_in_blob(weights["w_fc3"], FC3_OUT, FC2_OUT)
        weights["w_fc3"] = pad_k_n_blob(weights["w_fc3"], FC2_OUT, FC3_OUT, GOLEM_DIM)

        return weights


def transpose_out_in_blob(raw: bytes, out_dim: int, in_dim: int) -> bytes:
    count = out_dim * in_dim
    vals = struct.unpack(f"<{count}f", raw)
    transposed = [0.0] * count
    for o in range(out_dim):
        row_off = o * in_dim
        for i in range(in_dim):
            transposed[i * out_dim + o] = vals[row_off + i]
    return struct.pack(f"<{count}f", *transposed)


def pad_k_n_blob(raw_kn: bytes, k: int, n: int, align_k: int) -> bytes:
    k_pad = align_up(k, align_k)
    if k_pad == k:
        return raw_kn
    vals = struct.unpack(f"<{k * n}f", raw_kn)
    padded = [0.0] * (k_pad * n)
    for kk in range(k):
        src_off = kk * n
        dst_off = kk * n
        padded[dst_off : dst_off + n] = vals[src_off : src_off + n]
    return struct.pack(f"<{k_pad * n}f", *padded)


def write_block(buf: bytearray, offset: int, data: bytes, label: str):
    end = offset + len(data)
    if end > len(buf):
        raise ValueError(f"{label} out of range: off=0x{offset:x} size=0x{len(data):x}")
    buf[offset:end] = data


def main():
    os.makedirs(HBM_DIR, exist_ok=True)

    image = read_f32_file(IMAGE_PATH, IMAGE_SIZE)
    weights = read_weights(WEIGHTS_PATH)

    b_conv1_off = PRELOAD_BASE
    w_conv1_off = align_up(b_conv1_off + len(weights["b_conv1"]), HBM_ALIGN)
    b_conv2_off = align_up(w_conv1_off + len(weights["w_conv1"]), HBM_ALIGN)
    w_conv2_off = align_up(b_conv2_off + len(weights["b_conv2"]), HBM_ALIGN)
    b_fc1_off = align_up(w_conv2_off + len(weights["w_conv2"]), HBM_ALIGN)
    w_fc1_off = align_up(b_fc1_off + len(weights["b_fc1"]), HBM_ALIGN)
    b_fc2_off = align_up(w_fc1_off + len(weights["w_fc1"]), HBM_ALIGN)
    w_fc2_off = align_up(b_fc2_off + len(weights["b_fc2"]), HBM_ALIGN)
    b_fc3_off = align_up(w_fc2_off + len(weights["w_fc2"]), HBM_ALIGN)
    w_fc3_off = align_up(b_fc3_off + len(weights["b_fc3"]), HBM_ALIGN)

    node_buffers = {node_idx: bytearray(MEM_NODE_SIZE) for node_idx in DATA_NODE_IDXS}
    primary_node = DATA_NODE_IDXS[0]

    write_block(node_buffers[primary_node], INPUT_OFF, image, "image")
    write_block(node_buffers[primary_node], b_conv1_off, weights["b_conv1"], "b_conv1")
    write_block(node_buffers[primary_node], w_conv1_off, weights["w_conv1"], "w_conv1")
    write_block(node_buffers[primary_node], b_conv2_off, weights["b_conv2"], "b_conv2")
    write_block(node_buffers[primary_node], w_conv2_off, weights["w_conv2"], "w_conv2")
    write_block(node_buffers[primary_node], b_fc1_off, weights["b_fc1"], "b_fc1")
    write_block(node_buffers[primary_node], w_fc1_off, weights["w_fc1"], "w_fc1")
    write_block(node_buffers[primary_node], b_fc2_off, weights["b_fc2"], "b_fc2")
    write_block(node_buffers[primary_node], w_fc2_off, weights["w_fc2"], "w_fc2")
    write_block(node_buffers[primary_node], b_fc3_off, weights["b_fc3"], "b_fc3")
    write_block(node_buffers[primary_node], w_fc3_off, weights["w_fc3"], "w_fc3")

    for node_idx in DATA_NODE_IDXS:
        init_path = os.path.join(HBM_DIR, f"hbm_init_node{node_idx}.bin")
        out_path = os.path.join(HBM_DIR, f"hbm_out_node{node_idx}.bin")
        with open(init_path, "wb") as f:
            f.write(node_buffers[node_idx])
        with open(out_path, "wb") as f:
            f.seek(MEM_NODE_SIZE - 1)
            f.write(b"\0")

    print("[lenet5_hbm_init] generated backing files")
    print(
        f"  data_nodes={DATA_NODE_IDXS} size=0x{MEM_NODE_SIZE:x} golem_dim={GOLEM_DIM}"
    )
    print(
        f"  input      off=0x{INPUT_OFF:08x} abs=0x{HBM_NODE_IDX * MEM_NODE_SIZE + INPUT_OFF:08x} bytes=0x{len(image):x}"
    )
    print(
        f"  conv1_b    off=0x{b_conv1_off:08x} abs=0x{HBM_NODE_IDX * MEM_NODE_SIZE + b_conv1_off:08x} bytes=0x{len(weights['b_conv1']):x}"
    )
    print(
        f"  conv1_w    off=0x{w_conv1_off:08x} abs=0x{HBM_NODE_IDX * MEM_NODE_SIZE + w_conv1_off:08x} bytes=0x{len(weights['w_conv1']):x}"
    )
    print(
        f"  conv2_b    off=0x{b_conv2_off:08x} abs=0x{HBM_NODE_IDX * MEM_NODE_SIZE + b_conv2_off:08x} bytes=0x{len(weights['b_conv2']):x}"
    )
    print(
        f"  conv2_w    off=0x{w_conv2_off:08x} abs=0x{HBM_NODE_IDX * MEM_NODE_SIZE + w_conv2_off:08x} bytes=0x{len(weights['w_conv2']):x}"
    )
    print(
        f"  fc1_b      off=0x{b_fc1_off:08x} abs=0x{HBM_NODE_IDX * MEM_NODE_SIZE + b_fc1_off:08x} bytes=0x{len(weights['b_fc1']):x}"
    )
    print(
        f"  fc1_w      off=0x{w_fc1_off:08x} abs=0x{HBM_NODE_IDX * MEM_NODE_SIZE + w_fc1_off:08x} bytes=0x{len(weights['w_fc1']):x}"
    )
    print(
        f"  fc2_b      off=0x{b_fc2_off:08x} abs=0x{HBM_NODE_IDX * MEM_NODE_SIZE + b_fc2_off:08x} bytes=0x{len(weights['b_fc2']):x}"
    )
    print(
        f"  fc2_w      off=0x{w_fc2_off:08x} abs=0x{HBM_NODE_IDX * MEM_NODE_SIZE + w_fc2_off:08x} bytes=0x{len(weights['w_fc2']):x}"
    )
    print(
        f"  fc3_b      off=0x{b_fc3_off:08x} abs=0x{HBM_NODE_IDX * MEM_NODE_SIZE + b_fc3_off:08x} bytes=0x{len(weights['b_fc3']):x}"
    )
    print(
        f"  fc3_w      off=0x{w_fc3_off:08x} abs=0x{HBM_NODE_IDX * MEM_NODE_SIZE + w_fc3_off:08x} bytes=0x{len(weights['w_fc3']):x}"
    )
    print(
        f"  layer_out  conv1=0x{CONV1_OFF:08x} pool1=0x{POOL1_OFF:08x} conv2=0x{CONV2_OFF:08x} pool2=0x{POOL2_OFF:08x} fc1=0x{FC1_OFF:08x} fc2=0x{FC2_OFF:08x} fc3=0x{FC3_OFF:08x}"
    )
    print(
        f"  scratch    conv1_im2col=0x{CONV1_IM2COL_OFF:08x} conv2_im2col=0x{CONV2_IM2COL_OFF:08x}"
    )


if __name__ == "__main__":
    main()
