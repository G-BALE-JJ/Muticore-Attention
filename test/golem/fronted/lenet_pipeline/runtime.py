import os
import subprocess
from typing import List, Union

from golem_dtype import cast_scalar, elem_nbytes, normalize_dtype, pack_values

from .hbm_io import (
    read_fc1_from_hbm,
    read_fc2_from_hbm,
    read_fc3_from_hbm,
    read_pool1_from_hbm,
    read_pool2_from_hbm,
)
from .plan import LENET_ARTIFACT_ROOT, TESTS_DIR, write_lenet_plan

Scalar = Union[int, float]


def _write_tensor_bin(path: str, matrix: List[List[Scalar]], dtype: str) -> None:
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    flat = []
    for r in matrix:
        if len(r) != cols:
            raise ValueError("matrix is not rectangular")
        flat.extend(cast_scalar(dtype, v) for v in r)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(pack_values(dtype, flat))


def _write_vector_bin(path: str, vec: List[Scalar], dtype: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(pack_values(dtype, [cast_scalar(dtype, v) for v in vec]))


def _align_up(v: int, a: int) -> int:
    return ((v + a - 1) // a) * a


def _required_global_stride_kb(tile_dim: int) -> int:
    mat_bytes = tile_dim * tile_dim * 4
    vec_bytes = tile_dim * 4
    mat_aligned = _align_up(mat_bytes, 0x100)
    vec_aligned = _align_up(vec_bytes, 0x100)
    required_bytes = 0x2000 + 3 * mat_aligned + vec_aligned + 0x20
    return (required_bytes + 1023) // 1024


def _assert_tensor_bin_size(path: str, rows: int, cols: int, dtype: str) -> None:
    expected = rows * cols * elem_nbytes(dtype)
    actual = os.path.getsize(path)
    if actual != expected:
        raise ValueError(
            f"tensor size mismatch for {path}: expected {expected} bytes ({rows}x{cols}), got {actual}"
        )


def _validate_matrix(name: str, m: List[List[Scalar]], rows: int, cols: int) -> None:
    if len(m) != rows:
        raise ValueError(f"{name} row mismatch: expected {rows}, got {len(m)}")
    for i, r in enumerate(m):
        if len(r) != cols:
            raise ValueError(
                f"{name} col mismatch at row {i}: expected {cols}, got {len(r)}"
            )


class MatmulKernel:
    _built_variants = set()

    def __init__(
        self,
        block_m: int,
        block_n: int,
        block_k: int,
        tile_dim: int = 64,
        dtype: str = "int32",
        verify_c: bool = True,
    ):
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.tile_dim = tile_dim
        self.dtype = normalize_dtype(dtype)
        self.verify_c = verify_c

    def __call__(
        self,
        a,
        b,
        *,
        bias_file="",
        conv2_bpack_file="",
        conv2_bias_file="",
        fc1_weight_file="",
        fc1_bias_file="",
        fc2_weight_file="",
        fc2_bias_file="",
        fc3_weight_file="",
        fc3_bias_file="",
        verify_c=True,
        force_bias_enable=None,
        vanadis_exe="lenet_conv1",
        plan_stage="conv1_gemm",
        dma_overlap=None,
    ):
        m = len(a)
        k = len(a[0]) if m else 0
        if m == 0 or k == 0:
            raise ValueError("A must be non-empty")
        n = len(b[0]) if b else 0
        if n == 0:
            raise ValueError("B must be non-empty")
        if len(b) != k:
            raise ValueError("A/B shape mismatch")

        m_pad = _align_up(m, self.block_m)
        n_pad = _align_up(n, self.block_n)
        k_pad = _align_up(k, self.block_k)

        a_pad = [
            [cast_scalar(self.dtype, 0) for _ in range(k_pad)] for _ in range(m_pad)
        ]
        for i in range(m):
            for j in range(k):
                a_pad[i][j] = cast_scalar(self.dtype, a[i][j])
        b_pad = [
            [cast_scalar(self.dtype, 0) for _ in range(n_pad)] for _ in range(k_pad)
        ]
        for i in range(k):
            for j in range(n):
                b_pad[i][j] = cast_scalar(self.dtype, b[i][j])

        a_path = os.path.join(TESTS_DIR, "data", "py_a.bin")
        b_path = os.path.join(TESTS_DIR, "data", "py_b.bin")
        c_csv = os.path.join(TESTS_DIR, "data", "py_c_out.csv")
        _write_tensor_bin(a_path, a_pad, self.dtype)
        _write_tensor_bin(b_path, b_pad, self.dtype)
        _assert_tensor_bin_size(a_path, m_pad, k_pad, self.dtype)
        _assert_tensor_bin_size(b_path, k_pad, n_pad, self.dtype)

        total_cores = int(os.environ.get("GOLEM_TOTAL_CORES", "16"))
        total_groups = int(os.environ.get("GOLEM_TOTAL_GROUPS", "4"))
        gemm_cores = int(os.environ.get("GOLEM_TOTAL_GEMM_CORES", str(total_cores)))
        num_mem_nodes = int(os.environ.get("GOLEM_NUM_MEMORY_NODES", "4"))
        mem_node_size = int(
            os.environ.get("GOLEM_MEM_NODE_SIZE_BYTES", str(128 * 1024 * 1024))
        )
        global_stride_kb = int(os.environ.get("GOLEM_GLOBAL_STRIDE_KB", "64"))
        required_stride_kb = _required_global_stride_kb(self.tile_dim)
        if global_stride_kb < required_stride_kb:
            global_stride_kb = required_stride_kb
        global_stride_bytes = int(
            os.environ.get("GOLEM_GLOBAL_STRIDE_BYTES", str(global_stride_kb * 1024))
        )
        if global_stride_bytes < global_stride_kb * 1024:
            global_stride_bytes = global_stride_kb * 1024
        dma_stagger = int(os.environ.get("GOLEM_DMA_STAGGER_CYCLES", "0"))
        dma_overlap = (
            int(os.environ.get("GOLEM_DMA_OVERLAP", "0"))
            if dma_overlap is None
            else int(dma_overlap)
        )
        bias_enable = (
            int(os.environ.get("GOLEM_BIAS_ENABLE", "0"))
            if force_bias_enable is None
            else int(force_bias_enable)
        )
        bias_value = int(os.environ.get("GOLEM_BIAS_VALUE", "0"))

        build_key = (
            self.tile_dim,
            total_cores,
            total_groups,
            gemm_cores,
            num_mem_nodes,
            mem_node_size,
            global_stride_bytes,
            dma_stagger,
            dma_overlap,
            bias_enable,
            str(bias_value),
            vanadis_exe,
        )
        tmp_dir = os.path.join(TESTS_DIR, "artifacts_lenet", "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        build_env = os.environ.copy()
        build_env.setdefault("TMPDIR", tmp_dir)
        if build_key not in MatmulKernel._built_variants:
            subprocess.run(
                [
                    "make",
                    "-C",
                    "small/lenet5",
                    "ARCH=riscv64",
                    f"GOLEM_DIM={self.tile_dim}",
                    f"GOLEM_TOTAL_CORES={total_cores}",
                    f"GOLEM_TOTAL_GROUPS={total_groups}",
                    f"GOLEM_TOTAL_GEMM_CORES={gemm_cores}",
                    f"GOLEM_NUM_MEMORY_NODES={num_mem_nodes}",
                    f"GOLEM_MEM_NODE_SIZE_BYTES={mem_node_size}",
                    f"GOLEM_GLOBAL_STRIDE_BYTES={global_stride_bytes}",
                    f"GOLEM_DMA_STAGGER_CYCLES={dma_stagger}",
                    f"GOLEM_DMA_OVERLAP={dma_overlap}",
                    f"GOLEM_BIAS_ENABLE={bias_enable}",
                    f"GOLEM_BIAS_VALUE={bias_value}",
                    vanadis_exe,
                ],
                cwd=TESTS_DIR,
                env=build_env,
                check=True,
            )
            MatmulKernel._built_variants.add(build_key)

        cmd = [
            "./run_noc_dma_pipeline.sh",
            "--dim",
            str(self.tile_dim),
            "--groups",
            str(total_groups),
            "--num-cores",
            str(total_cores),
            "--gemm-cores",
            str(gemm_cores),
            "--num-mem-nodes",
            str(num_mem_nodes),
            "--global-stride-kb",
            str(global_stride_kb),
            "--gemm-m",
            str(m_pad),
            "--gemm-n",
            str(n_pad),
            "--gemm-k",
            str(k_pad),
            "--gemm-block-m",
            str(self.block_m),
            "--gemm-block-n",
            str(self.block_n),
            "--gemm-block-k",
            str(self.block_k),
            "--dtype",
            self.dtype,
            "--tensor-source",
            "file",
            "--tensor-a",
            os.path.relpath(a_path, TESTS_DIR),
            "--tensor-b",
            os.path.relpath(b_path, TESTS_DIR),
            "--dump-c",
            os.path.relpath(c_csv, TESTS_DIR),
        ]
        if verify_c:
            cmd.append("--verify-c")
        if bias_file:
            cmd.extend(["--bias-file", os.path.relpath(bias_file, TESTS_DIR)])
        for flag, val in [
            ("--conv2-bpack-file", conv2_bpack_file),
            ("--conv2-bias-file", conv2_bias_file),
            ("--fc1-weight-file", fc1_weight_file),
            ("--fc1-bias-file", fc1_bias_file),
            ("--fc2-weight-file", fc2_weight_file),
            ("--fc2-bias-file", fc2_bias_file),
            ("--fc3-weight-file", fc3_weight_file),
            ("--fc3-bias-file", fc3_bias_file),
        ]:
            if val:
                cmd.extend([flag, os.path.relpath(val, TESTS_DIR)])

        env = os.environ.copy()
        env.setdefault("TMPDIR", tmp_dir)
        env["GOLEM_MEM_NODE_SIZE_BYTES"] = str(mem_node_size)
        env["GOLEM_GLOBAL_STRIDE_BYTES"] = str(global_stride_bytes)
        env["GOLEM_VERIFY_C"] = "1" if verify_c else "0"
        artifact_root = env.get("GOLEM_ARTIFACT_ROOT", LENET_ARTIFACT_ROOT)
        plan_path = write_lenet_plan(artifact_root)
        env["VANADIS_EXE"] = os.path.join(
            TESTS_DIR, "small", "lenet5", "riscv64", vanadis_exe
        )
        env.setdefault("GOLEM_ARTIFACT_ROOT", LENET_ARTIFACT_ROOT)
        env["GOLEM_PLAN_FILE"] = plan_path
        env["GOLEM_PLAN_STAGE"] = plan_stage
        if force_bias_enable is not None:
            env["GOLEM_BIAS_ENABLE"] = str(int(force_bias_enable))

        subprocess.run(cmd, cwd=TESTS_DIR, env=env, check=True)
        return []


def run_gemm_via_pipeline(*args, **kwargs):
    a, b, tile_dim, block_n, block_k = args[0], args[1], args[2], args[3], args[4]
    block_m = tile_dim
    kernel = MatmulKernel(
        block_m,
        block_n,
        block_k,
        tile_dim,
        kwargs.get("dtype", "int32"),
        kwargs.get("verify_c", True),
    )
    return kernel(a, b, **{k: v for k, v in kwargs.items() if k not in {"dtype"}})


def _read_f32_bin(path: str) -> List[float]:
    raw = open(path, "rb").read()
    return list(__import__("struct").unpack(f"<{len(raw) // 4}f", raw))


def _read_matrix_bin(path: str, rows: int, cols: int) -> List[List[float]]:
    vals = _read_f32_bin(path)
    return [vals[r * cols : (r + 1) * cols] for r in range(rows)]


def run_conv12_single_sim_from_prebuilt_bins(real_bin_dir: str):
    a_banded = _read_matrix_bin(
        os.path.join(real_bin_dir, "a_conv1_banded_768x64.bin"), 768, 64
    )
    b_conv1_kn = _read_matrix_bin(
        os.path.join(real_bin_dir, "b_conv1_kn_64x6.bin"), 64, 6
    )

    run_gemm_via_pipeline(
        a_banded,
        b_conv1_kn,
        64,
        6,
        64,
        dma_overlap=0,
        dtype="fp32",
        bias_file=os.path.join(real_bin_dir, "bias_conv1_6.bin"),
        conv2_bpack_file=os.path.join(real_bin_dir, "conv2_bpack_3x16x64.bin"),
        conv2_bias_file=os.path.join(real_bin_dir, "bias_conv2_16.bin"),
        fc1_weight_file=os.path.join(real_bin_dir, "fc1_wslice_4x2x64x64.bin"),
        fc1_bias_file=os.path.join(real_bin_dir, "bias_fc1_120.bin"),
        fc2_weight_file=os.path.join(real_bin_dir, "fc2_wpack_2x2x64x64.bin"),
        fc2_bias_file=os.path.join(real_bin_dir, "bias_fc2_84.bin"),
        fc3_weight_file=os.path.join(real_bin_dir, "fc3_wpack_2x2x64x64.bin"),
        fc3_bias_file=os.path.join(real_bin_dir, "bias_fc3_10.bin"),
        verify_c=False,
        force_bias_enable=1,
        vanadis_exe="lenet_conv12",
        plan_stage="conv1_gemm",
    )

    artifact_root = os.environ.get("GOLEM_ARTIFACT_ROOT", LENET_ARTIFACT_ROOT)
    return (
        read_pool1_from_hbm(artifact_root),
        read_pool2_from_hbm(artifact_root),
        read_fc1_from_hbm(artifact_root),
        read_fc2_from_hbm(artifact_root),
        read_fc3_from_hbm(artifact_root),
    )
