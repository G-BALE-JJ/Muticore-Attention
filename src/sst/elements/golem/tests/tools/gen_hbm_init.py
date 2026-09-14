#!/usr/bin/env python3
"""
HBM 初始化文件生成器（GEMM 模式）

地址布局与 pipeline_config.h 保持一致：
- 首内存节点（Node0）挂 OS，不放 GEMM 数据
- 后续数据节点按核心均匀分配
- 每个数据节点放一份共享矩阵 + 节点内各核心向量
"""

import os
import sys
import csv
import json
import argparse

if __package__ in {None, ""}:
    _tests_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _tests_dir not in sys.path:
        sys.path.insert(0, _tests_dir)

from golem_dtype import (
    cast_scalar,
    elem_nbytes,
    normalize_dtype,
    numpy_dtype_name,
    pack_values,
    parse_scalar_text,
    unpack_values,
)
from softmax_hbm_layout import softmax_row_location

TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_ROOT = os.getenv("GOLEM_ARTIFACT_ROOT", os.path.join(TESTS_DIR, "artifacts"))
HBM_DIR = os.getenv("GOLEM_HBM_DIR", os.path.join(ARTIFACT_ROOT, "hbm"))
SFU_STANDALONE_SOFTMAX = int(os.getenv("GOLEM_SFU_STANDALONE_SOFTMAX", "0")) != 0
SFU_JOB_SOFTMAX_DIRECT_ROWMAJOR_HBM = (
    int(os.getenv("GOLEM_SFU_JOB_SOFTMAX_DIRECT_ROWMAJOR_HBM", "0")) != 0
)
SFU_PRIMITIVE_HBM_STREAM = int(os.getenv("GOLEM_SFU_PRIMITIVE_HBM_STREAM", "0")) != 0
SFU_PRIMITIVE_HBM_OPS = os.getenv("GOLEM_SFU_PRIMITIVE_HBM_OPS", "EXP")
SFU_PRIMITIVE_HBM_ELEMS = max(1, int(os.getenv("GOLEM_SFU_PRIMITIVE_HBM_ELEMS", "64")))
ATTENTION_FUSED = int(os.getenv("GOLEM_ATTENTION_FUSED", "0")) != 0
ATTENTION_Q_FILE = os.getenv("GOLEM_ATTENTION_Q_FILE", "")
ATTENTION_K_FILE = os.getenv("GOLEM_ATTENTION_K_FILE", "")
ATTENTION_V_FILE = os.getenv("GOLEM_ATTENTION_V_FILE", "")
ATTENTION_Q_OFFSET = int(os.getenv("GOLEM_ATTENTION_Q_OFFSET", "0x02000000"), 0)
ATTENTION_K_OFFSET = int(os.getenv("GOLEM_ATTENTION_K_OFFSET", "0x02010000"), 0)
ATTENTION_V_OFFSET = int(os.getenv("GOLEM_ATTENTION_V_OFFSET", "0x02020000"), 0)
ATTENTION_QUERIES = int(os.getenv("GOLEM_ATTENTION_QUERIES", "32"))
ATTENTION_KEYS = int(os.getenv("GOLEM_ATTENTION_KEYS", "32"))
ATTENTION_HEAD_DIM = int(os.getenv("GOLEM_ATTENTION_HEAD_DIM", "64"))
ATTENTION_HBM_STRIPED = int(os.getenv("GOLEM_ATTENTION_HBM_STRIPED", "0")) != 0
SOFTMAX_LOGITS_FILE = os.getenv("GOLEM_SOFTMAX_LOGITS_FILE", "")
SFU_SOFTMAX_HBM_LAYOUT = os.getenv("GOLEM_SFU_SOFTMAX_HBM_LAYOUT", "single_node")
SFU_SOFTMAX_STAGING_ROWS = max(
    1, int(os.getenv("GOLEM_SFU_JOB_SOFTMAX_STAGING_ROWS", "1"))
)
HBM_DUMP_OUTPUT = os.getenv("GOLEM_HBM_DUMP_OUTPUT", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
PRINT_CORE_MAP = int(os.getenv("GOLEM_PRINT_CORE_MAP", "0"))
CORE_MAP_FILE = os.getenv(
    "GOLEM_CORE_MAP_FILE", os.path.join(ARTIFACT_ROOT, "stats", "core_memory_map.csv")
)

_MEM_NODE_SIZE_ENV = os.getenv("GOLEM_MEM_NODE_SIZE_BYTES", str(64 * 1024**2))
MEM_NODE_SIZE_AUTO = _MEM_NODE_SIZE_ENV.strip().lower() == "auto"
MEM_NODE_SIZE = 0 if MEM_NODE_SIZE_AUTO else int(_MEM_NODE_SIZE_ENV, 0)
TOTAL_GROUPS = int(os.getenv("GOLEM_TOTAL_GROUPS", "4"))
# 优先使用运行脚本导出的 GOLEM_TOTAL_GEMM_CORES（对应 --gemm-cores）
# 兼容旧变量 GOLEM_TOTAL_CORES
TOTAL_GEMM_CORES = int(
    os.getenv("GOLEM_TOTAL_GEMM_CORES", os.getenv("GOLEM_TOTAL_CORES", "16"))
)
GROUP_MANAGER_ENABLED = int(os.getenv("GOLEM_GROUP_MANAGER_ENABLE", "0")) != 0
NUM_MEMORY_NODES = int(os.getenv("GOLEM_NUM_MEMORY_NODES", "5"))
OS_MEMORY_NODE_INDEX = 0
DATA_NODE_IDS = [idx for idx in range(NUM_MEMORY_NODES) if idx != OS_MEMORY_NODE_INDEX]

MM_ALIGN = 0x100

ARRAY_INPUT_SIZE = int(os.getenv("GOLEM_ARRAY_INPUT_SIZE", "4"))
ARRAY_OUTPUT_SIZE = int(os.getenv("GOLEM_ARRAY_OUTPUT_SIZE", "4"))
NUM_ARRAYS = int(os.getenv("GOLEM_NUM_ARRAYS", "1"))
GEMM_M_ENV = int(os.getenv("GOLEM_GEMM_M", str(ARRAY_OUTPUT_SIZE)))
GEMM_N_ENV = int(os.getenv("GOLEM_GEMM_N", str(NUM_ARRAYS)))
GEMM_K_ENV = int(os.getenv("GOLEM_GEMM_K", str(ARRAY_INPUT_SIZE)))

MATMUL_ENV_MAP = {
    "m": "GOLEM_MATMUL_M",
    "n": "GOLEM_MATMUL_N",
    "k": "GOLEM_MATMUL_K",
    "block_m": "GOLEM_MATMUL_BLOCK_M",
    "block_n": "GOLEM_MATMUL_BLOCK_N",
    "block_k": "GOLEM_MATMUL_BLOCK_K",
    "dtype": "GOLEM_MATMUL_DTYPE",
    "layout": "GOLEM_MATMUL_LAYOUT",
    "transpose_a": "GOLEM_MATMUL_TRANSPOSE_A",
    "transpose_b": "GOLEM_MATMUL_TRANSPOSE_B",
}


def _int_from_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


MATMUL_OP_DESC = {
    "m": _int_from_env(MATMUL_ENV_MAP["m"], GEMM_M_ENV),
    "n": _int_from_env(MATMUL_ENV_MAP["n"], GEMM_N_ENV),
    "k": _int_from_env(MATMUL_ENV_MAP["k"], GEMM_K_ENV),
    "block_m": _int_from_env(MATMUL_ENV_MAP["block_m"], ARRAY_OUTPUT_SIZE),
    "block_n": _int_from_env(MATMUL_ENV_MAP["block_n"], NUM_ARRAYS),
    "block_k": _int_from_env(MATMUL_ENV_MAP["block_k"], ARRAY_INPUT_SIZE),
    "dtype": os.getenv(MATMUL_ENV_MAP["dtype"], "int32"),
    "layout": os.getenv(MATMUL_ENV_MAP["layout"], "row_major"),
    "transpose_a": _int_from_env(MATMUL_ENV_MAP["transpose_a"], 0),
    "transpose_b": _int_from_env(MATMUL_ENV_MAP["transpose_b"], 0),
}
MATMUL_DTYPE = normalize_dtype(str(MATMUL_OP_DESC["dtype"]))
MATMUL_OP_DESC["dtype"] = MATMUL_DTYPE

GEMM_M = int(MATMUL_OP_DESC["m"])
GEMM_N = int(MATMUL_OP_DESC["n"])
GEMM_K = int(MATMUL_OP_DESC["k"])

BLOCK_M = int(MATMUL_OP_DESC["block_m"])
BLOCK_N = int(MATMUL_OP_DESC["block_n"])
BLOCK_K = int(MATMUL_OP_DESC["block_k"])

CONTRACT_DIR = os.path.join(ARTIFACT_ROOT, "contracts")
CONTRACT_MAPPING_FILE = os.path.join(CONTRACT_DIR, "matmul_env_mapping_v1.json")
CONTRACT_RESOLVED_FILE = os.path.join(CONTRACT_DIR, "matmul_op_desc_resolved.json")

GEMM_M_TILES = GEMM_M // BLOCK_M
GEMM_N_TILES = GEMM_N // BLOCK_N
GEMM_K_TILES = GEMM_K // BLOCK_K
TOTAL_GEMM_TASKS = GEMM_M_TILES * GEMM_N_TILES
A_REUSE_N_TILES = max(1, int(os.getenv("GOLEM_A_REUSE_N_TILES", "1")))
B_REUSE_M_TILES = max(1, int(os.getenv("GOLEM_B_REUSE_M_TILES", "1")))
GEMM_M_GROUPS = (GEMM_M_TILES + B_REUSE_M_TILES - 1) // B_REUSE_M_TILES
GEMM_N_GROUPS = (GEMM_N_TILES + A_REUSE_N_TILES - 1) // A_REUSE_N_TILES
TOTAL_GEMM_MACRO_TASKS = GEMM_M_GROUPS * GEMM_N_GROUPS
DEDICATED_MANAGER_CORES = TOTAL_GROUPS if GROUP_MANAGER_ENABLED else 0
ACTIVE_GEMM_CORES = TOTAL_GEMM_CORES - DEDICATED_MANAGER_CORES
FIRST_WORKER_CORE = DEDICATED_MANAGER_CORES
MAT_ELEMS = ARRAY_OUTPUT_SIZE * ARRAY_INPUT_SIZE
VEC_ELEMS = ARRAY_INPUT_SIZE
MAT_BYTES = MAT_ELEMS * elem_nbytes(MATMUL_DTYPE)
VEC_BYTES = VEC_ELEMS * elem_nbytes(MATMUL_DTYPE)
BLOCK_MAT_BYTES = BLOCK_M * BLOCK_K * elem_nbytes(MATMUL_DTYPE)
BLOCK_VEC_BYTES = BLOCK_K * elem_nbytes(MATMUL_DTYPE)


def _align_up(value: int, align: int) -> int:
    return ((value + align - 1) // align) * align


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


MM_MAT_STRIDE = _align_up(BLOCK_MAT_BYTES, MM_ALIGN)
OFF_GEMM_MAT = 0x0
# Pack B vectors back-to-back.  Padding each FP16 vector to a 256B line would
# waste half of the transferred bytes and is not part of the data layout.
MM_VEC_STRIDE = BLOCK_VEC_BYTES
MAT_REUSE_SLOTS = B_REUSE_M_TILES if B_REUSE_M_TILES > 1 else 1
VEC_REUSE_SLOTS = A_REUSE_N_TILES if A_REUSE_N_TILES > 1 else 1
OUT_REUSE_SLOTS = MAT_REUSE_SLOTS * VEC_REUSE_SLOTS


def _layout_owner_core_for_task(task_id: int) -> int:
    if ACTIVE_GEMM_CORES <= 0:
        return FIRST_WORKER_CORE
    return FIRST_WORKER_CORE + (task_id % ACTIVE_GEMM_CORES)


def _layout_group_id_for_core(core_id: int) -> int:
    if TOTAL_GROUPS <= 0:
        return 0
    return core_id % TOTAL_GROUPS


def _layout_data_node_for_task(task_id: int) -> int:
    if not DATA_NODE_IDS:
        return 1
    group_id = _layout_group_id_for_core(_layout_owner_core_for_task(task_id))
    return DATA_NODE_IDS[group_id % len(DATA_NODE_IDS)]


def _max_macro_tasks_per_data_node() -> int:
    max_count = 0
    for node_idx in DATA_NODE_IDS:
        count = sum(
            1
            for task_id in range(TOTAL_GEMM_MACRO_TASKS)
            if _layout_data_node_for_task(task_id) == node_idx
        )
        max_count = max(max_count, count)
    return max_count or 1


MAX_GEMM_MACRO_TASKS_PER_DATA_NODE = _max_macro_tasks_per_data_node()


def _layout_m_group_for_m_tile(m_tile: int) -> int:
    return m_tile // B_REUSE_M_TILES


def _layout_n_group_for_n_tile(n_tile: int) -> int:
    return n_tile // A_REUSE_N_TILES


def _layout_a_data_node_for_m_tile(m_tile: int) -> int:
    return _layout_data_node_for_task(_layout_m_group_for_m_tile(m_tile))


def _layout_b_data_node_for_n_tile(n_tile: int) -> int:
    return _layout_data_node_for_task(_layout_n_group_for_n_tile(n_tile))


def _max_a_m_tiles_per_data_node() -> int:
    max_count = 0
    for node_idx in DATA_NODE_IDS:
        count = sum(1 for m_tile in range(GEMM_M_TILES) if _layout_a_data_node_for_m_tile(m_tile) == node_idx)
        max_count = max(max_count, count)
    return max_count or 1


def _max_b_n_tiles_per_data_node() -> int:
    max_count = 0
    for node_idx in DATA_NODE_IDS:
        count = sum(1 for n_tile in range(GEMM_N_TILES) if _layout_b_data_node_for_n_tile(n_tile) == node_idx)
        max_count = max(max_count, count)
    return max_count or 1


MAX_GEMM_A_M_TILES_PER_DATA_NODE = _max_a_m_tiles_per_data_node()
MAX_GEMM_B_N_TILES_PER_DATA_NODE = _max_b_n_tiles_per_data_node()
OFF_GEMM_VEC_BASE = OFF_GEMM_MAT + MAX_GEMM_A_M_TILES_PER_DATA_NODE * GEMM_K_TILES * MM_MAT_STRIDE
GEMM_OUT_STRIDE_MM = _align_up(BLOCK_M * BLOCK_N * elem_nbytes(MATMUL_DTYPE), MM_ALIGN)
OFF_GEMM_OUT_BASE = (
    OFF_GEMM_VEC_BASE
    + MAX_GEMM_B_N_TILES_PER_DATA_NODE * GEMM_K_TILES * BLOCK_N * MM_VEC_STRIDE
)
GEMM_BIAS_STRIDE_MM = _align_up(GEMM_N * elem_nbytes(MATMUL_DTYPE), MM_ALIGN)
GEMM_DATA_REGION_END = OFF_GEMM_OUT_BASE + MAX_GEMM_MACRO_TASKS_PER_DATA_NODE * OUT_REUSE_SLOTS * GEMM_OUT_STRIDE_MM
SFU_SOFTMAX_ROWMAJOR_MATRIX_BYTES = GEMM_M * GEMM_N * elem_nbytes(MATMUL_DTYPE)
SFU_SOFTMAX_ROWMAJOR_STRIDE_BYTES = _align_up(SFU_SOFTMAX_ROWMAJOR_MATRIX_BYTES, MM_ALIGN)
OFF_SFU_SOFTMAX_ROWMAJOR_BASE = _align_up(GEMM_DATA_REGION_END, MM_ALIGN)
OFF_SFU_SOFTMAX_ROWMAJOR_OUT_BASE = (
    OFF_SFU_SOFTMAX_ROWMAJOR_BASE + SFU_SOFTMAX_ROWMAJOR_STRIDE_BYTES
)
SFU_SOFTMAX_ROWMAJOR_REGION_END = (
    OFF_SFU_SOFTMAX_ROWMAJOR_OUT_BASE + SFU_SOFTMAX_ROWMAJOR_STRIDE_BYTES
)
FIXED_AUX_REGION_END = 0x01300000

if MEM_NODE_SIZE_AUTO:
    required_size = max(
        GEMM_DATA_REGION_END + GEMM_BIAS_STRIDE_MM,
        SFU_SOFTMAX_ROWMAJOR_REGION_END + GEMM_BIAS_STRIDE_MM,
        FIXED_AUX_REGION_END,
        64 * 1024**2,
    )
    MEM_NODE_SIZE = _next_power_of_two(required_size)

IDENTITY_BASE = MEM_NODE_SIZE
OFF_GEMM_BIAS_BASE = MEM_NODE_SIZE - GEMM_BIAS_STRIDE_MM
POOL1_OFF = 0x01006000
POOL1_CH = 6
POOL1_H = 12
POOL1_W = 12
CONV2_BPACK_OFF = 0x01240000
CONV2_BIAS_OFF = 0x01248000
POOL1_READY_OFF = POOL1_OFF + 0x1000
FC1_WSLICE_OFF = 0x01250000
FC1_BIAS_OFF = 0x01270000
FC1_PARTIAL_OFF = 0x01271000
FC1_READY_OFF = 0x01272000
FC1_OUT_OFF = 0x01273000
FC2_WPACK_OFF = 0x01274000
FC2_BIAS_OFF = 0x01285000
FC2_OUT_OFF = 0x01286000
FC3_WPACK_OFF = 0x01287000
FC3_BIAS_OFF = 0x01298000
FC3_OUT_OFF = 0x01299000


def _format_bytes(data: bytes, bytes_per_line: int = 16) -> str:
    lines = []
    for i in range(0, len(data), bytes_per_line):
        chunk = data[i : i + bytes_per_line]
        line = " ".join(f"{b:02X}" for b in chunk)
        lines.append(line)
    return "\n".join(lines)


def _pack_tensor_values(values):
    return pack_values(MATMUL_DTYPE, values)


class SparseNodeBuffer:
    def __init__(self, path: str, size: int):
        self.path = path
        self.size = size
        self._file = open(path, "wb+")
        self._file.truncate(size)

    def __len__(self):
        return self.size

    def write_block(self, offset: int, data: bytes):
        self._file.seek(offset)
        self._file.write(data)

    def close(self):
        self._file.flush()
        self._file.close()


def _write_block(buf, offset: int, data: bytes, tag: str):
    end = offset + len(data)
    if offset < 0 or end > len(buf):
        raise ValueError(f"{tag} 写入越界: offset=0x{offset:x}, size={len(data)}")
    if hasattr(buf, "write_block"):
        buf.write_block(offset, data)
    else:
        buf[offset:end] = data


def _preload_fused_attention(node_buffers):
    if not ATTENTION_FUSED:
        return
    if 1 not in node_buffers:
        raise ValueError("fused Attention requires HBM data node 1")
    tensors = (
        ("Q", ATTENTION_Q_FILE, ATTENTION_Q_OFFSET,
         ATTENTION_QUERIES * ATTENTION_HEAD_DIM * 4),
        ("K", ATTENTION_K_FILE, ATTENTION_K_OFFSET,
         ATTENTION_KEYS * ATTENTION_HEAD_DIM * 4),
        ("V", ATTENTION_V_FILE, ATTENTION_V_OFFSET,
         ATTENTION_KEYS * ATTENTION_HEAD_DIM * 4),
    )
    loaded = {}
    for name, path, offset, expected_bytes in tensors:
        if not path:
            raise ValueError(f"GOLEM_ATTENTION_{name}_FILE is required")
        with open(path, "rb") as tensor_file:
            data = tensor_file.read()
        if len(data) != expected_bytes:
            raise ValueError(
                f"fused Attention {name} expected {expected_bytes} bytes, got {len(data)}"
            )
        loaded[name] = (offset, data)
    if ATTENTION_HBM_STRIPED:
        if len(DATA_NODE_IDS) != 4 or ATTENTION_QUERIES % 4 != 0 or ATTENTION_KEYS % 4 != 0:
            raise ValueError("striped fused Attention requires four data nodes and /4 shapes")
        q_band_bytes = (ATTENTION_QUERIES // 4) * ATTENTION_HEAD_DIM * 4
        kv_band_bytes = (ATTENTION_KEYS // 4) * ATTENTION_HEAD_DIM * 4
        for band, node_idx in enumerate(DATA_NODE_IDS):
            q_offset, q_data = loaded["Q"]
            _write_block(
                node_buffers[node_idx], q_offset,
                q_data[band * q_band_bytes:(band + 1) * q_band_bytes],
                f"attention_q_band{band}",
            )
            for name in ("K", "V"):
                offset, data = loaded[name]
                _write_block(
                    node_buffers[node_idx], offset,
                    data[band * kv_band_bytes:(band + 1) * kv_band_bytes],
                    f"attention_kv_band{band}_{name}",
                )
        print("Preloaded fused Attention Q/K/V bands across four HBM data nodes")
    else:
        for name, (offset, data) in loaded.items():
            _write_block(node_buffers[1], offset, data, f"fused_attention_{name}")
        print("Preloaded fused Attention Q/K/V in HBM node1; S/P have no HBM region")


def _node_base(node_idx: int) -> int:
    return node_idx * MEM_NODE_SIZE


def _is_worker_core(core_id: int) -> bool:
    if GROUP_MANAGER_ENABLED:
        return FIRST_WORKER_CORE <= core_id < TOTAL_GEMM_CORES
    return 0 <= core_id < ACTIVE_GEMM_CORES


def _worker_slot_for_core(core_id: int) -> int:
    if not _is_worker_core(core_id):
        return -1
    return core_id - FIRST_WORKER_CORE if GROUP_MANAGER_ENABLED else core_id


def _task_ids_for_core(core_id: int):
    worker_slot = _worker_slot_for_core(core_id)
    if worker_slot < 0:
        return
    task_id = worker_slot
    while task_id < TOTAL_GEMM_MACRO_TASKS:
        yield task_id
        task_id += ACTIVE_GEMM_CORES


def _m_group_of_macro_task(task_id: int) -> int:
    return task_id % GEMM_M_GROUPS


def _n_group_of_macro_task(task_id: int) -> int:
    m_group = _m_group_of_macro_task(task_id)
    return ((task_id // GEMM_M_GROUPS) + m_group) % GEMM_N_GROUPS


def _macro_task_for_group(m_group: int, n_group: int) -> int:
    n_band = (n_group - m_group) % GEMM_N_GROUPS
    return n_band * GEMM_M_GROUPS + m_group


def _m_tile_of_task(task_id: int) -> int:
    return _m_group_of_macro_task(task_id) * B_REUSE_M_TILES


def _n_tile_of_task(task_id: int) -> int:
    return _n_group_of_macro_task(task_id) * A_REUSE_N_TILES


def _m_count_for_group(m_group: int) -> int:
    m_begin = m_group * B_REUSE_M_TILES
    return min(B_REUSE_M_TILES, GEMM_M_TILES - m_begin)


def _n_count_for_group(n_group: int) -> int:
    n_begin = n_group * A_REUSE_N_TILES
    return min(A_REUSE_N_TILES, GEMM_N_TILES - n_begin)


def _owner_core_for_task(task_id: int) -> int:
    if ACTIVE_GEMM_CORES <= 0:
        return FIRST_WORKER_CORE
    worker_slot = task_id % ACTIVE_GEMM_CORES
    return worker_slot + FIRST_WORKER_CORE


def _group_id_for_core(core_id: int) -> int:
    if TOTAL_GROUPS <= 0:
        return 0
    return core_id % TOTAL_GROUPS


def _primary_data_node_for_group(group_id: int) -> int:
    if not DATA_NODE_IDS:
        return 1
    return DATA_NODE_IDS[group_id % len(DATA_NODE_IDS)]


def _data_node_for_task(task_id: int) -> int:
    owner_core = _owner_core_for_task(task_id)
    group_id = _group_id_for_core(owner_core)
    return _primary_data_node_for_group(group_id)


def _task_slot_in_node(task_id: int) -> int:
    node_idx = _data_node_for_task(task_id)
    slot = 0
    for prev in range(task_id):
        if _data_node_for_task(prev) == node_idx:
            slot += 1
    return slot


def _a_data_node_for_m_tile(m_tile: int) -> int:
    return _data_node_for_task(m_tile // B_REUSE_M_TILES)


def _b_data_node_for_n_tile(n_tile: int) -> int:
    return _data_node_for_task(n_tile // A_REUSE_N_TILES)


def _a_slot_for_m_tile(m_tile: int) -> int:
    node_idx = _a_data_node_for_m_tile(m_tile)
    slot = 0
    for prev in range(m_tile):
        if _a_data_node_for_m_tile(prev) == node_idx:
            slot += 1
    return slot


def _b_slot_for_n_tile(n_tile: int) -> int:
    node_idx = _b_data_node_for_n_tile(n_tile)
    slot = 0
    for prev in range(n_tile):
        if _b_data_node_for_n_tile(prev) == node_idx:
            slot += 1
    return slot


def _build_matrix_tile(rows: int, cols: int, m_tile: int, k_tile: int):
    # 所有元素 < 10，且不同 tile 可区分
    seed = (m_tile * 3 + k_tile * 5) % 9
    return [((seed + r + c) % 9) + 1 for r in range(rows) for c in range(cols)]


def _build_vector_tile(n_tile: int, k_tile: int, n_col: int, length: int):
    # 每个 (n_tile, k_tile, n_col) 唯一向量，匹配 packed-once B 布局。
    seed = (n_tile * 2 + k_tile * 7 + n_col * 3) % 9
    return [((seed + i) % 9) + 1 for i in range(length)]


def _build_softmax_logits_matrix(rows: int, cols: int):
    matrix = []
    for r in range(rows):
        row = []
        for c in range(cols):
            raw = ((r * 3 + c * 5 + (r // 7) * 11) % 31) - 15
            row.append(cast_scalar(MATMUL_DTYPE, raw / 8.0))
        matrix.append(row)
    return matrix


def _write_matrix_to_file(path: str, matrix):
    flat = [value for row in matrix for value in row]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.endswith(".csv"):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(matrix)
    elif path.endswith(".npy"):
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                f"standalone softmax logits uses .npy but numpy is not available: {path}"
            ) from exc
        np.save(path, np.array(matrix, dtype=numpy_dtype_name(MATMUL_DTYPE)))
    else:
        with open(path, "wb") as f:
            f.write(_pack_tensor_values(flat))


def _load_matrix_from_file(path: str, rows: int, cols: int, tensor_name: str):
    if not path:
        return None

    if path.endswith(".csv"):
        values = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                for item in row:
                    item = item.strip()
                    if item:
                        values.append(parse_scalar_text(MATMUL_DTYPE, item))
    elif path.endswith(".npy"):
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                f"{tensor_name} uses .npy but numpy is not available: {path}"
            ) from exc
        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(
                f"{tensor_name} npy must be 2D, got ndim={arr.ndim}: {path}"
            )
        if arr.shape != (rows, cols):
            raise ValueError(
                f"{tensor_name} shape mismatch, expected ({rows},{cols}), got {arr.shape}: {path}"
            )
        return arr.astype(numpy_dtype_name(MATMUL_DTYPE), copy=False)
    else:
        with open(path, "rb") as f:
            data = f.read()
        expected_bytes = rows * cols * elem_nbytes(MATMUL_DTYPE)
        if len(data) != expected_bytes:
            raise ValueError(
                f"{tensor_name} binary size mismatch, expected {expected_bytes} bytes, got {len(data)}: {path}"
            )
        values = unpack_values(MATMUL_DTYPE, data)

    if len(values) != rows * cols:
        raise ValueError(
            f"{tensor_name} element count mismatch, expected {rows * cols}, got {len(values)}: {path}"
        )

    return [values[r * cols : (r + 1) * cols] for r in range(rows)]


def _matrix_tile_from_input(
    a_matrix, m_tile: int, k_tile: int, block_m: int, block_k: int
):
    m_base = m_tile * block_m
    k_base = k_tile * block_k
    out = []
    for r in range(block_m):
        for c in range(block_k):
            out.append(cast_scalar(MATMUL_DTYPE, a_matrix[m_base + r][k_base + c]))
    return out


def _vector_from_input(
    b_matrix,
    k_tile: int,
    n_tile: int,
    n_col: int,
    block_k: int,
    block_n: int,
    *,
    transpose_b: bool = False,
):
    k_base = k_tile * block_k
    n_base = n_tile * block_n + n_col
    out = []
    for i in range(block_k):
        value = (
            b_matrix[n_base][k_base + i]
            if transpose_b
            else b_matrix[k_base + i][n_base]
        )
        out.append(cast_scalar(MATMUL_DTYPE, value))
    return out


def _load_bias_vector_from_file(path: str, length: int):
    if not path:
        return None

    if path.endswith(".csv"):
        values = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                for item in row:
                    item = item.strip()
                    if item:
                        values.append(parse_scalar_text(MATMUL_DTYPE, item))
    elif path.endswith(".npy"):
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                f"bias uses .npy but numpy is not available: {path}"
            ) from exc
        arr = np.load(path)
        if arr.ndim == 2 and 1 in arr.shape:
            arr = arr.reshape(-1)
        if arr.ndim != 1:
            raise ValueError(
                f"bias npy must be 1D (or Nx1/1xN), got shape={arr.shape}: {path}"
            )
        if arr.shape[0] != length:
            raise ValueError(
                f"bias shape mismatch, expected ({length},), got {arr.shape}: {path}"
            )
        return arr.astype(numpy_dtype_name(MATMUL_DTYPE), copy=False).tolist()
    else:
        with open(path, "rb") as f:
            data = f.read()
        expected_bytes = length * elem_nbytes(MATMUL_DTYPE)
        if len(data) != expected_bytes:
            raise ValueError(
                f"bias binary size mismatch, expected {expected_bytes} bytes, got {len(data)}: {path}"
            )
        values = unpack_values(MATMUL_DTYPE, data)

    if len(values) != length:
        raise ValueError(
            f"bias element count mismatch, expected {length}, got {len(values)}: {path}"
        )
    return [cast_scalar(MATMUL_DTYPE, v) for v in values]


def _write_standalone_softmax_logits(node_buffers, logits_matrix):
    out_tile_bytes = BLOCK_M * BLOCK_N * elem_nbytes(MATMUL_DTYPE)
    for task_id in range(TOTAL_GEMM_TASKS):
        m_tile = task_id // GEMM_N_TILES
        n_tile = task_id % GEMM_N_TILES
        m_group = m_tile // B_REUSE_M_TILES
        n_group = n_tile // A_REUSE_N_TILES
        m_offset = m_tile % B_REUSE_M_TILES
        n_offset = n_tile % A_REUSE_N_TILES
        macro_task_id = _macro_task_for_group(m_group, n_group)
        node_idx = _data_node_for_task(macro_task_id)
        task_slot = _task_slot_in_node(macro_task_id)
        reuse_offset = m_offset * VEC_REUSE_SLOTS + n_offset
        tile_off = OFF_GEMM_OUT_BASE + (task_slot * OUT_REUSE_SLOTS + reuse_offset) * GEMM_OUT_STRIDE_MM

        tile = [cast_scalar(MATMUL_DTYPE, 0) for _ in range(BLOCK_M * BLOCK_N)]
        for r in range(BLOCK_M):
            for cc in range(BLOCK_N):
                global_m = m_tile * BLOCK_M + r
                global_n = n_tile * BLOCK_N + cc
                tile[cc * BLOCK_M + r] = cast_scalar(
                    MATMUL_DTYPE, logits_matrix[global_m][global_n]
                )
        tile_data = _pack_tensor_values(tile)
        if len(tile_data) != out_tile_bytes:
            raise ValueError(
                f"standalone softmax logits tile size mismatch: expected {out_tile_bytes}, got {len(tile_data)}"
            )
        _write_block(
            node_buffers[node_idx],
            tile_off,
            tile_data,
            f"softmax_logits_m{m_tile}_n{n_tile}_node{node_idx}",
        )


def _write_standalone_softmax_rowmajor_logits(node_buffers, logits_matrix):
    if not DATA_NODE_IDS:
        raise ValueError("direct row-major standalone softmax requires at least one data node")
    if SFU_SOFTMAX_ROWMAJOR_REGION_END > OFF_GEMM_BIAS_BASE:
        raise ValueError(
            "SFU softmax row-major HBM region exceeds bias area: "
            f"input=0x{OFF_SFU_SOFTMAX_ROWMAJOR_BASE:X}, "
            f"output=0x{OFF_SFU_SOFTMAX_ROWMAJOR_OUT_BASE:X}, "
            f"end=0x{SFU_SOFTMAX_ROWMAJOR_REGION_END:X}, "
            f"bias=0x{OFF_GEMM_BIAS_BASE:X}"
        )
    row_bytes = GEMM_N * elem_nbytes(MATMUL_DTYPE)
    for r, row in enumerate(logits_matrix):
        node_idx, local_row = softmax_row_location(
            r, SFU_SOFTMAX_STAGING_ROWS, DATA_NODE_IDS, SFU_SOFTMAX_HBM_LAYOUT
        )
        row_data = _pack_tensor_values(row)
        if len(row_data) != row_bytes:
            raise ValueError(
                f"standalone softmax row-major row size mismatch: expected {row_bytes}, got {len(row_data)}"
            )
        _write_block(
            node_buffers[node_idx],
            OFF_SFU_SOFTMAX_ROWMAJOR_BASE + local_row * row_bytes,
            row_data,
            f"softmax_rowmajor_logits_row{r}_node{node_idx}",
        )


def _sfu_primitive_hbm_op_count() -> int:
    tokens = [
        token.strip()
        for token in SFU_PRIMITIVE_HBM_OPS.replace(";", ",").split(",")
        if token.strip()
    ]
    if not tokens:
        return 1
    return 6 if any(token.upper() == "ALL" for token in tokens) else len(tokens)


def _sfu_primitive_hbm_input_value(elem_idx: int) -> float:
    return 0.5 + float(elem_idx % 32) * 0.125


def _write_sfu_primitive_hbm_input(node_buffers):
    if MATMUL_DTYPE != "fp32":
        raise ValueError("SFU primitive HBM stream input preload currently requires fp32")
    task_id = 0
    m_tile = task_id // GEMM_N_TILES
    n_tile = task_id % GEMM_N_TILES
    m_group = m_tile // B_REUSE_M_TILES
    n_group = n_tile // A_REUSE_N_TILES
    m_offset = m_tile % B_REUSE_M_TILES
    n_offset = n_tile % A_REUSE_N_TILES
    macro_task_id = _macro_task_for_group(m_group, n_group)
    node_idx = _data_node_for_task(macro_task_id)
    task_slot = _task_slot_in_node(macro_task_id)
    reuse_offset = m_offset * VEC_REUSE_SLOTS + n_offset
    base_off = OFF_GEMM_OUT_BASE + (task_slot * OUT_REUSE_SLOTS + reuse_offset) * GEMM_OUT_STRIDE_MM
    op_count = _sfu_primitive_hbm_op_count()
    elem_bytes = elem_nbytes(MATMUL_DTYPE)
    total_bytes = SFU_PRIMITIVE_HBM_ELEMS * elem_bytes
    if base_off + total_bytes * op_count > OFF_GEMM_BIAS_BASE:
        raise ValueError(
            "SFU primitive HBM stream input exceeds GEMM C region before bias area: "
            f"bytes={total_bytes * op_count}, base=0x{base_off:X}, bias=0x{OFF_GEMM_BIAS_BASE:X}"
        )
    values = [
        cast_scalar(MATMUL_DTYPE, _sfu_primitive_hbm_input_value(i))
        for i in range(SFU_PRIMITIVE_HBM_ELEMS)
    ]
    data = _pack_tensor_values(values)
    for op_idx in range(op_count):
        _write_block(
            node_buffers[node_idx],
            base_off + op_idx * total_bytes,
            data,
            f"sfu_primitive_hbm_input_op{op_idx}_node{node_idx}",
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate HBM init files for GOLEM matmul"
    )
    parser.add_argument(
        "--a-file",
        default=os.getenv("GOLEM_TENSOR_A_FILE", ""),
        help="Input A tensor file (.bin/.csv/.npy; dtype follows GOLEM_MATMUL_DTYPE)",
    )
    parser.add_argument(
        "--b-file",
        default=os.getenv("GOLEM_TENSOR_B_FILE", ""),
        help="Input B tensor file (.bin/.csv/.npy; dtype follows GOLEM_MATMUL_DTYPE)",
    )
    parser.add_argument(
        "--bias-file",
        default=os.getenv("GOLEM_BIAS_FILE", ""),
        help="Optional bias vector file (.bin/.csv/.npy), length should be GEMM_N",
    )
    parser.add_argument(
        "--softmax-logits-file",
        default=SOFTMAX_LOGITS_FILE,
        help="Optional standalone softmax logits file (.bin/.csv/.npy), shape=GEMM_MxGEMM_N",
    )
    parser.add_argument(
        "--pool1-file",
        default=os.getenv("GOLEM_POOL1_FILE", ""),
        help="Optional pool1 CHW tensor file (.bin), shape=6x12x12 fp32",
    )
    parser.add_argument(
        "--conv2-bpack-file",
        default=os.getenv("GOLEM_CONV2_BPACK_FILE", ""),
        help="Optional conv2 packed B file (.bin), shape=3x16x64 fp32",
    )
    parser.add_argument(
        "--conv2-bias-file",
        default=os.getenv("GOLEM_CONV2_BIAS_FILE", ""),
        help="Optional conv2 bias file (.bin), len=16 fp32",
    )
    parser.add_argument(
        "--fc1-weight-file",
        default=os.getenv("GOLEM_FC1_WEIGHT_FILE", ""),
        help="Optional fc1 sliced weight file (.bin), shape=4x2x64x64 fp32",
    )
    parser.add_argument(
        "--fc1-bias-file",
        default=os.getenv("GOLEM_FC1_BIAS_FILE", ""),
        help="Optional fc1 bias file (.bin), len=120 fp32",
    )
    parser.add_argument(
        "--fc2-weight-file",
        default=os.getenv("GOLEM_FC2_WEIGHT_FILE", ""),
        help="Optional fc2 weight file (.bin), shape=2x2x64x64 fp32",
    )
    parser.add_argument(
        "--fc2-bias-file",
        default=os.getenv("GOLEM_FC2_BIAS_FILE", ""),
        help="Optional fc2 bias file (.bin), len=84 fp32",
    )
    parser.add_argument(
        "--fc3-weight-file",
        default=os.getenv("GOLEM_FC3_WEIGHT_FILE", ""),
        help="Optional fc3 weight file (.bin), shape=2x2x64x64 fp32",
    )
    parser.add_argument(
        "--fc3-bias-file",
        default=os.getenv("GOLEM_FC3_BIAS_FILE", ""),
        help="Optional fc3 bias file (.bin), len=10 fp32",
    )
    args = parser.parse_args(argv)

    if ARRAY_INPUT_SIZE <= 0 or ARRAY_OUTPUT_SIZE <= 0:
        raise ValueError(
            f"GOLEM_ARRAY_INPUT_SIZE/GOLEM_ARRAY_OUTPUT_SIZE must be positive, got {ARRAY_INPUT_SIZE}/{ARRAY_OUTPUT_SIZE}"
        )
    if TOTAL_GROUPS <= 0:
        raise ValueError(f"GOLEM_TOTAL_GROUPS must be positive, got {TOTAL_GROUPS}")
    if TOTAL_GEMM_CORES <= 0:
        raise ValueError(
            f"GOLEM_TOTAL_GEMM_CORES must be positive, got {TOTAL_GEMM_CORES}"
        )
    if GEMM_M <= 0 or GEMM_N <= 0 or GEMM_K <= 0:
        raise ValueError(
            f"GOLEM_GEMM_M/N/K must be positive, got {GEMM_M}/{GEMM_N}/{GEMM_K}"
        )
    block_k = MATMUL_OP_DESC["block_k"]
    if not ATTENTION_FUSED and (
        MATMUL_OP_DESC["block_m"] % ARRAY_OUTPUT_SIZE != 0
        or (block_k > ARRAY_INPUT_SIZE and block_k % ARRAY_INPUT_SIZE != 0)
    ):
        raise ValueError(
            f"Phase-1 requires block_M to be a multiple of ARRAY_OUTPUT({ARRAY_OUTPUT_SIZE}); "
            f"block_K must be <= ARRAY_INPUT({ARRAY_INPUT_SIZE}) or an integer multiple of it, "
            f"got {MATMUL_OP_DESC['block_m']}/{block_k}"
        )
    if MATMUL_OP_DESC["block_n"] <= 0 or MATMUL_OP_DESC["block_n"] > NUM_ARRAYS:
        raise ValueError(
            f"Phase-1 requires 0 < block_N <= GOLEM_NUM_ARRAYS({NUM_ARRAYS}), got {MATMUL_OP_DESC['block_n']}"
        )
    if MATMUL_OP_DESC["layout"] != "row_major":
        raise ValueError(
            f"Phase-1 requires GOLEM_MATMUL_LAYOUT=row_major, got {MATMUL_OP_DESC['layout']}"
        )
    if MATMUL_OP_DESC["transpose_a"] != 0 or MATMUL_OP_DESC["transpose_b"] not in (0, 1):
        raise ValueError("Phase-1 requires transpose_a=0 and transpose_b in {0,1}")
    if GEMM_M % BLOCK_M != 0 or GEMM_N % BLOCK_N != 0 or GEMM_K % BLOCK_K != 0:
        raise ValueError(
            f"GOLEM_GEMM_M/N/K must be divisible by block_M/N/K ({BLOCK_M}/{BLOCK_N}/{BLOCK_K}), got {GEMM_M}/{GEMM_N}/{GEMM_K}"
        )
    if GEMM_K_TILES <= 0:
        raise ValueError(f"Derived GEMM_K_TILES must be positive, got {GEMM_K_TILES}")
    if TOTAL_GEMM_TASKS <= 0:
        raise ValueError("TOTAL_GEMM_TASKS must be positive")
    if NUM_MEMORY_NODES < 2:
        raise ValueError(f"GOLEM_NUM_MEMORY_NODES must be >= 2, got {NUM_MEMORY_NODES}")
    if MEM_NODE_SIZE <= 0:
        raise ValueError(
            f"GOLEM_MEM_NODE_SIZE_BYTES must be positive, got {MEM_NODE_SIZE}"
        )
    if OFF_GEMM_BIAS_BASE <= 0:
        raise ValueError(
            f"Bias region is invalid, OFF_GEMM_BIAS_BASE={OFF_GEMM_BIAS_BASE}"
        )
    if GEMM_DATA_REGION_END > OFF_GEMM_BIAS_BASE:
        raise ValueError(
            f"GEMM HBM layout exceeds per-node memory: data_end=0x{GEMM_DATA_REGION_END:x}, "
            f"bias_base=0x{OFF_GEMM_BIAS_BASE:x}, mem_node_size=0x{MEM_NODE_SIZE:x}"
        )

    plan_file = os.getenv("GOLEM_PLAN_FILE", "").strip()
    plan_stage_name = os.getenv("GOLEM_PLAN_STAGE", "conv1_gemm").strip()

    print("=" * 60)
    print("HBM Backing File Generator (Per-Node Files)")
    print("=" * 60)
    print(f"Memory nodes: {NUM_MEMORY_NODES} (OS node: {OS_MEMORY_NODE_INDEX})")
    print(f"Data nodes: {DATA_NODE_IDS}")
    if MEM_NODE_SIZE_AUTO:
        print(
            f"Auto memory node size: {MEM_NODE_SIZE} bytes "
            f"({MEM_NODE_SIZE // (1024 * 1024)} MiB)"
        )
    print(f"HBM output dump: {'enabled' if HBM_DUMP_OUTPUT else 'disabled'}")
    if plan_file:
        print(f"LeNet plan file: {plan_file} (stage={plan_stage_name})")
    print(f"Matmul env mapping file: {CONTRACT_MAPPING_FILE}")
    print(f"Resolved matmul op desc: {CONTRACT_RESOLVED_FILE}")
    print("=" * 60)

    os.makedirs(HBM_DIR, exist_ok=True)
    os.makedirs(CONTRACT_DIR, exist_ok=True)

    plan_task_map = None
    plan = None
    if plan_file:
        if A_REUSE_N_TILES > 1:
            raise ValueError("GOLEM_A_REUSE_N_TILES>1 is not supported with plan-file layout yet")
        if B_REUSE_M_TILES > 1:
            raise ValueError("GOLEM_B_REUSE_M_TILES>1 is not supported with plan-file layout yet")
        with open(plan_file, "r", encoding="utf-8") as f:
            plan = json.load(f)
        stages = plan.get("stages", {})
        stage = stages.get(plan_stage_name)
        if stage is None:
            raise ValueError(
                f"GOLEM_PLAN_STAGE={plan_stage_name} not found in plan {plan_file}"
            )
        plan_tasks = stage.get("tasks", [])
        plan_task_map = {int(t["task_id"]): t for t in plan_tasks}
        if len(plan_task_map) != TOTAL_GEMM_TASKS:
            raise ValueError(
                f"plan task count mismatch: expect {TOTAL_GEMM_TASKS}, got {len(plan_task_map)}"
            )

    def task_info(task_id: int):
        if plan_task_map is not None:
            t = plan_task_map[task_id]
            return (
                int(t["node_idx"]),
                int(t["task_slot_in_node"]),
                int(t["m_tile"]),
                int(t["n_tile"]),
            )
        return (
            _data_node_for_task(task_id),
            _task_slot_in_node(task_id),
            _m_tile_of_task(task_id),
            _n_tile_of_task(task_id),
        )

    with open(CONTRACT_MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(MATMUL_ENV_MAP, f, indent=2, sort_keys=True)
    with open(CONTRACT_RESOLVED_FILE, "w", encoding="utf-8") as f:
        json.dump(MATMUL_OP_DESC, f, indent=2, sort_keys=True)

    node_buffers = {}
    for node_idx in DATA_NODE_IDS:
        init_file = os.path.join(HBM_DIR, f"hbm_init_node{node_idx}.bin")
        node_buffers[node_idx] = SparseNodeBuffer(init_file, MEM_NODE_SIZE)

    _preload_fused_attention(node_buffers)

    a_matrix = _load_matrix_from_file(args.a_file, GEMM_M, GEMM_K, "A")
    transpose_b = MATMUL_OP_DESC["transpose_b"] == 1
    b_rows = GEMM_N if transpose_b else GEMM_K
    b_cols = GEMM_K if transpose_b else GEMM_N
    b_matrix = _load_matrix_from_file(args.b_file, b_rows, b_cols, "B")
    bias_vec = _load_bias_vector_from_file(args.bias_file, GEMM_N)
    softmax_logits = None
    bias_enabled = int(os.getenv("GOLEM_BIAS_ENABLE", "0")) != 0
    bias_value = parse_scalar_text(MATMUL_DTYPE, os.getenv("GOLEM_BIAS_VALUE", "0"))

    if (a_matrix is None) != (b_matrix is None):
        raise ValueError("--a-file and --b-file must be provided together")

    if a_matrix is not None:
        print(f"Using external tensor files: A={args.a_file}, B={args.b_file}")
    else:
        print("Using synthetic matrix/vector seed data (no --a-file/--b-file)")

    if SFU_STANDALONE_SOFTMAX:
        if args.softmax_logits_file and os.path.exists(args.softmax_logits_file):
            softmax_logits = _load_matrix_from_file(
                args.softmax_logits_file, GEMM_M, GEMM_N, "standalone softmax logits"
            )
            print(f"Using standalone softmax logits file: {args.softmax_logits_file}")
        else:
            softmax_logits = _build_softmax_logits_matrix(GEMM_M, GEMM_N)
            if args.softmax_logits_file:
                _write_matrix_to_file(args.softmax_logits_file, softmax_logits)
                print(f"Generated standalone softmax logits file: {args.softmax_logits_file}")
            else:
                print("Using synthetic standalone softmax logits without file export")

    if bias_enabled and bias_vec is None:
        print(
            f"Bias enabled without --bias-file, using scalar fallback value={bias_value}"
        )
        bias_vec = [cast_scalar(MATMUL_DTYPE, bias_value) for _ in range(GEMM_N)]
    elif bias_vec is not None:
        print(f"Using external bias vector file: {args.bias_file}")

    if args.pool1_file:
        expect_pool1_bytes = POOL1_CH * POOL1_H * POOL1_W * 4
        with open(args.pool1_file, "rb") as f:
            pool1_raw = f.read()
        if len(pool1_raw) != expect_pool1_bytes:
            raise ValueError(
                f"pool1 size mismatch, expected {expect_pool1_bytes}, got {len(pool1_raw)}: {args.pool1_file}"
            )

        conv1_tasks = None
        if plan is not None:
            conv1_stage = plan.get("stages", {}).get("conv1_gemm")
            if conv1_stage is not None:
                conv1_tasks = {
                    int(t["task_id"]): t for t in conv1_stage.get("tasks", [])
                }

        for ph in range(POOL1_H):
            if conv1_tasks is not None and ph in conv1_tasks:
                node_idx = int(conv1_tasks[ph]["node_idx"])
            else:
                node_idx = DATA_NODE_IDS[(ph * len(DATA_NODE_IDS)) // POOL1_H]

            for oc in range(POOL1_CH):
                src_off = (oc * POOL1_H * POOL1_W + ph * POOL1_W) * 4
                row_bytes = pool1_raw[src_off : src_off + POOL1_W * 4]
                dst_off = POOL1_OFF + src_off
                _write_block(
                    node_buffers[node_idx],
                    dst_off,
                    row_bytes,
                    f"pool1_oc{oc}_ph{ph}_node{node_idx}",
                )
        print(f"Using external pool1 tensor file: {args.pool1_file}")

    if args.conv2_bpack_file:
        expect_bpack_bytes = 3 * 16 * 64 * 4
        with open(args.conv2_bpack_file, "rb") as f:
            conv2_bpack_raw = f.read()
        if len(conv2_bpack_raw) != expect_bpack_bytes:
            raise ValueError(
                f"conv2 bpack size mismatch, expected {expect_bpack_bytes}, got {len(conv2_bpack_raw)}: {args.conv2_bpack_file}"
            )
        for node_idx in DATA_NODE_IDS:
            _write_block(
                node_buffers[node_idx],
                CONV2_BPACK_OFF,
                conv2_bpack_raw,
                f"conv2_bpack_node{node_idx}",
            )
        print(f"Using external conv2 bpack file: {args.conv2_bpack_file}")

    if args.conv2_bias_file:
        expect_bias_bytes = 16 * 4
        with open(args.conv2_bias_file, "rb") as f:
            conv2_bias_raw = f.read()
        if len(conv2_bias_raw) != expect_bias_bytes:
            raise ValueError(
                f"conv2 bias size mismatch, expected {expect_bias_bytes}, got {len(conv2_bias_raw)}: {args.conv2_bias_file}"
            )
        for node_idx in DATA_NODE_IDS:
            _write_block(
                node_buffers[node_idx],
                CONV2_BIAS_OFF,
                conv2_bias_raw,
                f"conv2_bias_node{node_idx}",
            )
        print(f"Using external conv2 bias file: {args.conv2_bias_file}")

    if args.fc1_weight_file:
        expect_fc1_w_bytes = 4 * 2 * 64 * 64 * 4
        with open(args.fc1_weight_file, "rb") as f:
            fc1_w_raw = f.read()
        if len(fc1_w_raw) != expect_fc1_w_bytes:
            raise ValueError(
                f"fc1 weight size mismatch, expected {expect_fc1_w_bytes}, got {len(fc1_w_raw)}: {args.fc1_weight_file}"
            )
        for node_idx in DATA_NODE_IDS:
            _write_block(
                node_buffers[node_idx],
                FC1_WSLICE_OFF,
                fc1_w_raw,
                f"fc1_wslice_node{node_idx}",
            )
        print(f"Using external fc1 weight file: {args.fc1_weight_file}")

    if args.fc1_bias_file:
        expect_fc1_bias_bytes = 120 * 4
        with open(args.fc1_bias_file, "rb") as f:
            fc1_bias_raw = f.read()
        if len(fc1_bias_raw) != expect_fc1_bias_bytes:
            raise ValueError(
                f"fc1 bias size mismatch, expected {expect_fc1_bias_bytes}, got {len(fc1_bias_raw)}: {args.fc1_bias_file}"
            )
        for node_idx in DATA_NODE_IDS:
            _write_block(
                node_buffers[node_idx],
                FC1_BIAS_OFF,
                fc1_bias_raw,
                f"fc1_bias_node{node_idx}",
            )
        print(f"Using external fc1 bias file: {args.fc1_bias_file}")

    if args.fc2_weight_file:
        expect_fc2_w_bytes = 2 * 2 * 64 * 64 * 4
        with open(args.fc2_weight_file, "rb") as f:
            fc2_w_raw = f.read()
        if len(fc2_w_raw) != expect_fc2_w_bytes:
            raise ValueError(
                f"fc2 weight size mismatch, expected {expect_fc2_w_bytes}, got {len(fc2_w_raw)}: {args.fc2_weight_file}"
            )
        for node_idx in DATA_NODE_IDS:
            _write_block(
                node_buffers[node_idx],
                FC2_WPACK_OFF,
                fc2_w_raw,
                f"fc2_w_node{node_idx}",
            )
        print(f"Using external fc2 weight file: {args.fc2_weight_file}")

    if args.fc2_bias_file:
        expect_fc2_bias_bytes = 84 * 4
        with open(args.fc2_bias_file, "rb") as f:
            fc2_bias_raw = f.read()
        if len(fc2_bias_raw) != expect_fc2_bias_bytes:
            raise ValueError(
                f"fc2 bias size mismatch, expected {expect_fc2_bias_bytes}, got {len(fc2_bias_raw)}: {args.fc2_bias_file}"
            )
        for node_idx in DATA_NODE_IDS:
            _write_block(
                node_buffers[node_idx],
                FC2_BIAS_OFF,
                fc2_bias_raw,
                f"fc2_bias_node{node_idx}",
            )
        print(f"Using external fc2 bias file: {args.fc2_bias_file}")

    if args.fc3_weight_file:
        expect_fc3_w_bytes = 2 * 2 * 64 * 64 * 4
        with open(args.fc3_weight_file, "rb") as f:
            fc3_w_raw = f.read()
        if len(fc3_w_raw) != expect_fc3_w_bytes:
            raise ValueError(
                f"fc3 weight size mismatch, expected {expect_fc3_w_bytes}, got {len(fc3_w_raw)}: {args.fc3_weight_file}"
            )
        for node_idx in DATA_NODE_IDS:
            _write_block(
                node_buffers[node_idx],
                FC3_WPACK_OFF,
                fc3_w_raw,
                f"fc3_w_node{node_idx}",
            )
        print(f"Using external fc3 weight file: {args.fc3_weight_file}")

    if args.fc3_bias_file:
        expect_fc3_bias_bytes = 10 * 4
        with open(args.fc3_bias_file, "rb") as f:
            fc3_bias_raw = f.read()
        if len(fc3_bias_raw) != expect_fc3_bias_bytes:
            raise ValueError(
                f"fc3 bias size mismatch, expected {expect_fc3_bias_bytes}, got {len(fc3_bias_raw)}: {args.fc3_bias_file}"
            )
        for node_idx in DATA_NODE_IDS:
            _write_block(
                node_buffers[node_idx],
                FC3_BIAS_OFF,
                fc3_bias_raw,
                f"fc3_bias_node{node_idx}",
            )
        print(f"Using external fc3 bias file: {args.fc3_bias_file}")

    for node_idx in DATA_NODE_IDS:
        _write_block(
            node_buffers[node_idx],
            POOL1_READY_OFF,
            b"\x00" * (POOL1_H * 8),
            f"pool1_ready_node{node_idx}",
        )
        _write_block(
            node_buffers[node_idx],
            FC1_READY_OFF,
            b"\x00" * (8 * 8),
            f"fc1_ready_node{node_idx}",
        )
        _write_block(
            node_buffers[node_idx],
            FC1_PARTIAL_OFF,
            b"\x00" * (4 * 128 * 4),
            f"fc1_partial_node{node_idx}",
        )

    for m_tile in range(GEMM_M_TILES):
        node_idx = _a_data_node_for_m_tile(m_tile)
        a_slot = _a_slot_for_m_tile(m_tile)
        for k_tile in range(GEMM_K_TILES):
            if a_matrix is not None:
                mat = _matrix_tile_from_input(
                    a_matrix, m_tile, k_tile, BLOCK_M, BLOCK_K
                )
            else:
                mat = _build_matrix_tile(BLOCK_M, BLOCK_K, m_tile, k_tile)
            mat_data = _pack_tensor_values(mat)
            mat_off = OFF_GEMM_MAT + (a_slot * GEMM_K_TILES + k_tile) * MM_MAT_STRIDE
            _write_block(
                node_buffers[node_idx],
                mat_off,
                mat_data,
                f"a_m{m_tile}_k{k_tile}_node{node_idx}",
            )

    for n_tile in range(GEMM_N_TILES):
        node_idx = _b_data_node_for_n_tile(n_tile)
        b_slot = _b_slot_for_n_tile(n_tile)
        for k_tile in range(GEMM_K_TILES):
            for n_col in range(BLOCK_N):
                if b_matrix is not None:
                    vec = _vector_from_input(
                        b_matrix,
                        k_tile,
                        n_tile,
                        n_col,
                        BLOCK_K,
                        BLOCK_N,
                        transpose_b=transpose_b,
                    )
                else:
                    vec = _build_vector_tile(n_tile, k_tile, n_col, BLOCK_K)
                vec_data = _pack_tensor_values(vec)
                vec_slot = (b_slot * GEMM_K_TILES + k_tile) * BLOCK_N + n_col
                vec_off = OFF_GEMM_VEC_BASE + vec_slot * MM_VEC_STRIDE
                _write_block(
                    node_buffers[node_idx],
                    vec_off,
                    vec_data,
                    f"b_n{n_tile}_k{k_tile}_n{n_col}_node{node_idx}",
                )

    if bias_vec is not None:
        bias_data = _pack_tensor_values(bias_vec)
        for node_idx in DATA_NODE_IDS:
            _write_block(
                node_buffers[node_idx],
                OFF_GEMM_BIAS_BASE,
                bias_data,
                f"bias_node{node_idx}",
            )

    if softmax_logits is not None and SFU_JOB_SOFTMAX_DIRECT_ROWMAJOR_HBM:
        _write_standalone_softmax_rowmajor_logits(node_buffers, softmax_logits)
        print("Preloaded standalone softmax logits into direct row-major HBM region")
    elif softmax_logits is not None:
        _write_standalone_softmax_logits(node_buffers, softmax_logits)
        print("Preloaded standalone softmax logits into GEMM C tile region")

    if SFU_PRIMITIVE_HBM_STREAM:
        _write_sfu_primitive_hbm_input(node_buffers)
        print("Preloaded SFU primitive HBM stream input into GEMM C tile region")

    for node_idx in DATA_NODE_IDS:
        init_file = os.path.join(HBM_DIR, f"hbm_init_node{node_idx}.bin")
        out_file = os.path.join(HBM_DIR, f"hbm_out_node{node_idx}.bin")

        node_buffers[node_idx].close()

        if HBM_DUMP_OUTPUT:
            with open(out_file, "wb") as f:
                f.truncate(MEM_NODE_SIZE)
            print(f"生成 {out_file}: {MEM_NODE_SIZE // (1024 * 1024)}MB")
        else:
            if os.path.exists(out_file):
                os.remove(out_file)

        print(f"生成 {init_file}: {MEM_NODE_SIZE // (1024 * 1024)}MB")
    if not HBM_DUMP_OUTPUT:
        print("HBM 输出落盘已关闭：未生成 hbm_out_node*.bin")

    print("\n[布局信息]")
    print(
        f"  ARRAY_OUTPUT/INPUT={ARRAY_OUTPUT_SIZE}/{ARRAY_INPUT_SIZE}, DTYPE={MATMUL_DTYPE}, CORES={TOTAL_GEMM_CORES}, MAT_BYTES={MAT_BYTES}, VEC_BYTES={VEC_BYTES}, "
        f"MM_MAT_STRIDE=0x{MM_MAT_STRIDE:X}, MM_VEC_STRIDE=0x{MM_VEC_STRIDE:X}"
    )
    print(
        f"  MAX_A_M_TILES_PER_DATA_NODE={MAX_GEMM_A_M_TILES_PER_DATA_NODE}, MAX_B_N_TILES_PER_DATA_NODE={MAX_GEMM_B_N_TILES_PER_DATA_NODE}, "
        f"MAX_MACRO_TASKS_PER_DATA_NODE={MAX_GEMM_MACRO_TASKS_PER_DATA_NODE}, OFF_GEMM_OUT_BASE=0x{OFF_GEMM_OUT_BASE:X}, "
        f"DATA_REGION_END=0x{GEMM_DATA_REGION_END:X}, OFF_GEMM_BIAS_BASE=0x{OFF_GEMM_BIAS_BASE:X}, BIAS_STRIDE=0x{GEMM_BIAS_STRIDE_MM:X}"
    )
    print(
        f"  GEMM_M/N/K={GEMM_M}/{GEMM_N}/{GEMM_K}, block(M,N,K)={BLOCK_M}/{BLOCK_N}/{BLOCK_K}, tiles(M,N,K)={GEMM_M_TILES}/{GEMM_N_TILES}/{GEMM_K_TILES}, tasks={TOTAL_GEMM_TASKS}, macro_tasks={TOTAL_GEMM_MACRO_TASKS}, a_reuse_n={A_REUSE_N_TILES}, b_reuse_m={B_REUSE_M_TILES}, active_cores={ACTIVE_GEMM_CORES}"
    )
    print(f"  packed_once A 基地址（首数据节点）: 0x{IDENTITY_BASE + OFF_GEMM_MAT:08X}")
    if bias_vec is not None:
        print(
            f"  bias 基地址（每个数据节点）: 0x{IDENTITY_BASE + OFF_GEMM_BIAS_BASE:08X}, len={GEMM_N}"
        )

    print("\n[预期读取值] (Identity Window 地址映射)")
    print(
        "matrix/vector: A 按 (m_tile,k_tile) packed once；B 按 (n_tile,k_tile,n_col) packed once"
    )

    for core_id in range(FIRST_WORKER_CORE, TOTAL_GEMM_CORES):
        task_ids = list(_task_ids_for_core(core_id))
        if not task_ids:
            print(f"  core{core_id:02d}: no task")
            continue
        for task_id in task_ids:
            c_node_idx, task_slot, m_tile, n_tile = task_info(task_id)
            a_node_idx = _a_data_node_for_m_tile(m_tile)
            b_node_idx = _b_data_node_for_n_tile(n_tile)
            a_slot = _a_slot_for_m_tile(m_tile)
            b_slot = _b_slot_for_n_tile(n_tile)
            mat_addr = (
                _node_base(a_node_idx)
                + OFF_GEMM_MAT
                + a_slot * GEMM_K_TILES * MM_MAT_STRIDE
            )
            vec_addr = (
                _node_base(b_node_idx)
                + OFF_GEMM_VEC_BASE
                + b_slot * GEMM_K_TILES * BLOCK_N * MM_VEC_STRIDE
            )
            print(
                f"  core{core_id:02d}: task={task_id} (m_tile={m_tile},n_tile={n_tile}) mat_k0@0x{mat_addr:08X} (node{a_node_idx}, slot={a_slot}), vec_k0@0x{vec_addr:08X} (node{b_node_idx}, slot={b_slot}), c_node={c_node_idx}, c_slot={task_slot}"
            )

    if PRINT_CORE_MAP:
        os.makedirs(os.path.dirname(CORE_MAP_FILE), exist_ok=True)
        with open(CORE_MAP_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "core_id",
                    "task_seq_on_core",
                    "memory_node",
                    "mat_addr_hex",
                    "vec_addr_hex",
                    "vec_slot",
                    "task_id",
                    "m_tile",
                    "n_tile",
                ]
            )
            for core_id in range(FIRST_WORKER_CORE, TOTAL_GEMM_CORES):
                for seq_idx, task_id in enumerate(_task_ids_for_core(core_id)):
                    c_node_idx, node_slot, m_tile, n_tile = task_info(task_id)
                    a_node_idx = _a_data_node_for_m_tile(m_tile)
                    b_node_idx = _b_data_node_for_n_tile(n_tile)
                    a_slot = _a_slot_for_m_tile(m_tile)
                    b_slot = _b_slot_for_n_tile(n_tile)
                    mat_addr = (
                        _node_base(a_node_idx)
                        + OFF_GEMM_MAT
                        + a_slot * GEMM_K_TILES * MM_MAT_STRIDE
                    )
                    vec_addr = (
                        _node_base(b_node_idx)
                        + OFF_GEMM_VEC_BASE
                        + b_slot * GEMM_K_TILES * BLOCK_N * MM_VEC_STRIDE
                    )
                    writer.writerow(
                        [
                            core_id,
                            seq_idx,
                            c_node_idx,
                            f"0x{mat_addr:08X}",
                            f"0x{vec_addr:08X}",
                            node_slot,
                            task_id,
                            m_tile,
                            n_tile,
                        ]
                    )
        print(f"\n[CoreMap] 已生成: {CORE_MAP_FILE}")

    return 0


if __name__ == "__main__":
    exit(main())
