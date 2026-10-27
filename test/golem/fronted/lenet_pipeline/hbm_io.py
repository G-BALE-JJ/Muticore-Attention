import os
import struct
from typing import List

from .plan import LENET_ARTIFACT_ROOT, POOL1_OFF, load_lenet_plan


def read_pool1_from_hbm(
    artifact_root: str = LENET_ARTIFACT_ROOT,
) -> List[List[List[float]]]:
    plan = load_lenet_plan(artifact_root)
    data_nodes = plan["memory"]["data_nodes"]
    conv1_stage = plan["stages"]["conv1_gemm"]
    task_map = {int(t["task_id"]): t for t in conv1_stage["tasks"]}
    hbm_dir = os.path.join(artifact_root, "hbm")
    node_buffers = {}
    for node_idx in data_nodes:
        with open(os.path.join(hbm_dir, f"hbm_out_node{node_idx}.bin"), "rb") as f:
            node_buffers[node_idx] = f.read()
    out = [[[0.0 for _ in range(12)] for _ in range(12)] for _ in range(6)]
    for band in range(12):
        node_idx = int(task_map[band]["node_idx"])
        node_data = node_buffers[node_idx]
        for oc in range(6):
            for pw in range(12):
                off = POOL1_OFF + 4 * (oc * 144 + band * 12 + pw)
                out[oc][band][pw] = struct.unpack("<f", node_data[off : off + 4])[0]
    return out


def read_pool2_from_hbm(
    artifact_root: str = LENET_ARTIFACT_ROOT,
) -> List[List[List[float]]]:
    plan = load_lenet_plan(artifact_root)
    pool2_off = int(plan["layouts"]["pool2_off"])
    conv2_stage = plan["stages"]["conv2_gemm"]
    data_nodes = plan["memory"]["data_nodes"]
    task_map = {int(t["task_id"]): t for t in conv2_stage["tasks"]}
    hbm_dir = os.path.join(artifact_root, "hbm")
    node_buffers = {}
    for node_idx in data_nodes:
        with open(os.path.join(hbm_dir, f"hbm_out_node{node_idx}.bin"), "rb") as f:
            node_buffers[node_idx] = f.read()
    out = [[[0.0 for _ in range(4)] for _ in range(4)] for _ in range(16)]
    for band in range(4):
        node_idx = int(task_map[band]["node_idx"])
        node_data = node_buffers[node_idx]
        for oc in range(16):
            for pw in range(4):
                off = pool2_off + 4 * (oc * 16 + band * 4 + pw)
                out[oc][band][pw] = struct.unpack("<f", node_data[off : off + 4])[0]
    return out


def _read_vec(artifact_root: str, key: str, length: int) -> List[float]:
    plan = load_lenet_plan(artifact_root)
    off = int(plan["layouts"][key])
    node_idx = int(plan["stages"]["conv2_gemm"]["tasks"][0]["node_idx"])
    with open(
        os.path.join(artifact_root, "hbm", f"hbm_out_node{node_idx}.bin"), "rb"
    ) as f:
        f.seek(off)
        raw = f.read(length * 4)
    return list(struct.unpack(f"<{length}f", raw))


def read_fc1_from_hbm(artifact_root: str = LENET_ARTIFACT_ROOT) -> List[float]:
    return _read_vec(artifact_root, "fc1_out_off", 120)


def read_fc2_from_hbm(artifact_root: str = LENET_ARTIFACT_ROOT) -> List[float]:
    return _read_vec(artifact_root, "fc2_out_off", 84)


def read_fc3_from_hbm(artifact_root: str = LENET_ARTIFACT_ROOT) -> List[float]:
    return _read_vec(artifact_root, "fc3_out_off", 10)
