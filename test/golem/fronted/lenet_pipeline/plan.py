import json
import os
from typing import List


TESTS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(TESTS_DIR, "data")
LENET_ARTIFACT_ROOT = os.path.join(TESTS_DIR, "artifacts_lenet")
CONTRACT_DIR = os.path.join(LENET_ARTIFACT_ROOT, "contracts")
LENET_PLAN_FILE = os.path.join(CONTRACT_DIR, "lenet_plan_v2.json")

REAL_DATASET_DIR = os.path.join(TESTS_DIR, "task", "task lenet5")
REAL_BIN_DIR = os.path.join(DATA_DIR, "real_lenet5")

POOL1_OFF = 0x01006000
POOL2_OFF = 0x0100A000
FC1_OUT_OFF = 0x01273000
FC2_OUT_OFF = 0x01286000
FC3_OUT_OFF = 0x01299000


def _build_task_map(total_tasks: int, num_memory_nodes: int) -> List[dict]:
    data_nodes = [idx for idx in range(num_memory_nodes) if idx != 0]
    if not data_nodes:
        raise ValueError("GOLEM_NUM_MEMORY_NODES must provide at least one data node")
    tasks: List[dict] = []
    for task_id in range(total_tasks):
        node_idx = data_nodes[(task_id * len(data_nodes)) // total_tasks]
        slot = 0
        for prev in range(task_id):
            prev_node = data_nodes[(prev * len(data_nodes)) // total_tasks]
            if prev_node == node_idx:
                slot += 1
        tasks.append(
            {"task_id": task_id, "node_idx": node_idx, "task_slot_in_node": slot}
        )
    return tasks


def write_lenet_plan(artifact_root: str) -> str:
    num_memory_nodes = int(os.environ.get("GOLEM_NUM_MEMORY_NODES", "4"))
    conv1_m, conv1_n, conv1_k = 768, 6, 64
    conv1_bm, conv1_bn, conv1_bk = 64, 6, 64
    conv1_m_tiles = conv1_m // conv1_bm
    conv1_n_tiles = conv1_n // conv1_bn
    conv1_total_tasks = conv1_m_tiles * conv1_n_tiles
    conv1_tasks = _build_task_map(conv1_total_tasks, num_memory_nodes)
    for t in conv1_tasks:
        task_id = int(t["task_id"])
        t["m_tile"] = task_id // conv1_n_tiles
        t["n_tile"] = task_id % conv1_n_tiles

    conv2 = {"m": 256, "n": 16, "k": 192, "block_m": 64, "block_n": 16, "block_k": 64}
    conv2_m_tiles = conv2["m"] // conv2["block_m"]
    conv2_n_tiles = conv2["n"] // conv2["block_n"]
    conv2_total_tasks = conv2_m_tiles * conv2_n_tiles
    conv2_tasks = _build_task_map(conv2_total_tasks, num_memory_nodes)
    for t in conv2_tasks:
        task_id = int(t["task_id"])
        t["m_tile"] = task_id // conv2_n_tiles
        t["n_tile"] = task_id % conv2_n_tiles

    stage_flow = [
        {
            "name": "conv1_im2col",
            "type": "fixed",
            "latency_track": True,
            "fixed_ms": 1.423,
            "task_group": "host",
        },
        {
            "name": "conv1_gemm",
            "type": "gemm",
            "latency_track": True,
            "task_group": "conv1_gemm_tasks",
        },
        {
            "name": "conv1_relu",
            "type": "relu",
            "latency_track": True,
            "task_group": "conv1_gemm_tasks",
        },
        {
            "name": "pool1",
            "type": "pool",
            "latency_track": True,
            "task_group": "conv1_gemm_tasks",
        },
        {
            "name": "conv2_im2col",
            "type": "repack",
            "latency_track": True,
            "task_group": "conv2_gemm_tasks",
        },
        {
            "name": "conv2_gemm",
            "type": "gemm",
            "latency_track": True,
            "task_group": "conv2_gemm_tasks",
        },
        {
            "name": "conv2_relu",
            "type": "relu",
            "latency_track": True,
            "task_group": "conv2_gemm_tasks",
        },
        {
            "name": "pool2",
            "type": "pool",
            "latency_track": True,
            "task_group": "conv2_gemm_tasks",
        },
        {
            "name": "fc1",
            "type": "splitk_reduce",
            "latency_track": True,
            "task_group": "conv2_gemm_tasks",
        },
        {"name": "fc2", "type": "mvm", "latency_track": True, "task_group": "core0"},
        {"name": "fc3", "type": "mvm", "latency_track": True, "task_group": "core0"},
    ]

    plan = {
        "version": "lenet_plan_v2",
        "memory": {
            "num_memory_nodes": num_memory_nodes,
            "os_memory_node": 0,
            "data_nodes": [idx for idx in range(num_memory_nodes) if idx != 0],
        },
        "layouts": {
            "pool1_off": POOL1_OFF,
            "pool1_shape_chw": [6, 12, 12],
            "pool2_off": POOL2_OFF,
            "pool2_shape_chw": [16, 4, 4],
            "fc1_out_off": FC1_OUT_OFF,
            "fc1_out_len": 120,
            "fc2_out_off": FC2_OUT_OFF,
            "fc2_out_len": 84,
            "fc3_out_off": FC3_OUT_OFF,
            "fc3_out_len": 10,
        },
        "stage_flow": stage_flow,
        "stages": {
            "conv1_gemm": {
                "m": conv1_m,
                "n": conv1_n,
                "k": conv1_k,
                "block_m": conv1_bm,
                "block_n": conv1_bn,
                "block_k": conv1_bk,
                "m_tiles": conv1_m_tiles,
                "n_tiles": conv1_n_tiles,
                "total_tasks": conv1_total_tasks,
                "tasks": conv1_tasks,
            },
            "conv2_repack": {
                "input": "pool1_chw_6x12x12",
                "output": "conv2_a_banded_256x192",
            },
            "conv2_gemm": {
                "m": conv2["m"],
                "n": conv2["n"],
                "k": conv2["k"],
                "block_m": conv2["block_m"],
                "block_n": conv2["block_n"],
                "block_k": conv2["block_k"],
                "m_tiles": conv2_m_tiles,
                "n_tiles": conv2_n_tiles,
                "total_tasks": conv2_total_tasks,
                "tasks": conv2_tasks,
            },
            "fc1_splitk": {"tasks": conv2_tasks, "k_per_task": 64, "out": 120},
        },
        "task_groups": {
            "conv1_gemm_tasks": conv1_tasks,
            "conv2_gemm_tasks": conv2_tasks,
            "core0": [{"core_id": 0}],
            "host": [{"executor": "python_host"}],
        },
    }

    plan_v2_path = os.path.join(artifact_root, "contracts", "lenet_plan_v2.json")
    plan_v1_compat = os.path.join(artifact_root, "contracts", "lenet_plan_v1.json")
    os.makedirs(os.path.dirname(plan_v2_path), exist_ok=True)
    with open(plan_v2_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, sort_keys=True)
    with open(plan_v1_compat, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, sort_keys=True)
    return plan_v2_path


def load_lenet_plan(artifact_root: str) -> dict:
    plan_path = os.path.join(artifact_root, "contracts", "lenet_plan_v2.json")
    if not os.path.exists(plan_path):
        plan_path = os.path.join(artifact_root, "contracts", "lenet_plan_v1.json")
    with open(plan_path, "r", encoding="utf-8") as f:
        return json.load(f)
