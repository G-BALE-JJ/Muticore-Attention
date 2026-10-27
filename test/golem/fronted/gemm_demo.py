#!/usr/bin/env python3

import csv
import os
import random
import subprocess
import sys
from typing import List, Tuple, Union

if __package__ in {None, ""}:
    _fronted_dir = os.path.dirname(os.path.abspath(__file__))
    _tests_dir = os.path.dirname(_fronted_dir)
    if _tests_dir not in sys.path:
        sys.path.insert(0, _tests_dir)

from golem_dtype import cast_scalar, elem_nbytes, normalize_dtype, pack_values


TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(TESTS_DIR, "data")

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


def _read_csv_tensor(path: str, dtype: str) -> List[List[Scalar]]:
    out: List[List[Scalar]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            out.append([cast_scalar(dtype, x) for x in row])
    return out


def _align_up(v: int, a: int) -> int:
    return ((v + a - 1) // a) * a


def _required_global_stride_kb(tile_dim: int) -> int:
    mat_bytes = tile_dim * tile_dim * 4
    vec_bytes = tile_dim * 4
    mat_aligned = _align_up(mat_bytes, 0x100)
    vec_aligned = _align_up(vec_bytes, 0x100)
    required_bytes = 0x2000 + 3 * mat_aligned + vec_aligned + 0x20
    return (required_bytes + 1023) // 1024


def ceil_div(x: int, y: int) -> int:
    if y <= 0:
        raise ValueError(f"divisor must be positive, got {y}")
    return (x + y - 1) // y


def pad_shape(
    m: int, n: int, k: int, bm: int, bn: int, bk: int
) -> Tuple[int, int, int]:
    return (ceil_div(m, bm) * bm, ceil_div(n, bn) * bn, ceil_div(k, bk) * bk)


def pad_a(a: List[List[Scalar]], m_pad: int, k_pad: int) -> List[List[Scalar]]:
    m = len(a)
    k = len(a[0]) if m > 0 else 0
    out = [[0 for _ in range(k_pad)] for _ in range(m_pad)]
    for i in range(m):
        for kk in range(k):
            out[i][kk] = a[i][kk]
    return out


def pad_b(b: List[List[Scalar]], k_pad: int, n_pad: int) -> List[List[Scalar]]:
    k = len(b)
    n = len(b[0]) if k > 0 else 0
    out = [[0 for _ in range(n_pad)] for _ in range(k_pad)]
    for kk in range(k):
        for j in range(n):
            out[kk][j] = b[kk][j]
    return out


def crop_c(c_pad: List[List[Scalar]], m: int, n: int) -> List[List[Scalar]]:
    if len(c_pad) < m:
        raise ValueError(f"C padded rows too small: need {m}, got {len(c_pad)}")
    if m > 0 and len(c_pad[0]) < n:
        raise ValueError(f"C padded cols too small: need {n}, got {len(c_pad[0])}")
    return [row[:n] for row in c_pad[:m]]


def _assert_tensor_bin_size(
    path: str, rows: int, cols: int, name: str, dtype: str
) -> None:
    expect = rows * cols * elem_nbytes(dtype)
    got = os.path.getsize(path)
    if got != expect:
        raise ValueError(
            f"{name} binary size mismatch, expect {expect} bytes ({rows}x{cols}x{elem_nbytes(dtype)}), got {got}"
        )


def _validate_matrix(
    name: str, m: List[List[Scalar]], expect_rows: int = -1, expect_cols: int = -1
) -> None:
    rows = len(m)
    cols = len(m[0]) if rows > 0 else 0
    if rows == 0 or cols == 0:
        raise ValueError(f"{name} is empty")
    for row in m:
        if len(row) != cols:
            raise ValueError(f"{name} is not rectangular")
    if expect_rows >= 0 and rows != expect_rows:
        raise ValueError(f"{name} row mismatch: expect {expect_rows}, got {rows}")
    if expect_cols >= 0 and cols != expect_cols:
        raise ValueError(f"{name} col mismatch: expect {expect_cols}, got {cols}")


class MatmulKernel:
    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        block_m: int,
        block_n: int,
        block_k: int,
        dma_overlap: int = 0,
        dtype: str = "int32",
    ):
        self.m = m
        self.n = n
        self.k = k
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.tile_dim = block_m
        self.dma_overlap = dma_overlap
        self.dtype = normalize_dtype(dtype)

    def __call__(
        self, a: List[List[Scalar]], b: List[List[Scalar]]
    ) -> List[List[Scalar]]:
        _validate_matrix("A", a, self.m, self.k)
        _validate_matrix("B", b, self.k, self.n)

        m_pad, n_pad, k_pad = pad_shape(
            self.m, self.n, self.k, self.block_m, self.block_n, self.block_k
        )
        a_exec = a if (m_pad == self.m and k_pad == self.k) else pad_a(a, m_pad, k_pad)
        b_exec = b if (k_pad == self.k and n_pad == self.n) else pad_b(b, k_pad, n_pad)

        a_path = os.path.join(DATA_DIR, "py_a.bin")
        b_path = os.path.join(DATA_DIR, "py_b.bin")
        c_path = os.path.join(DATA_DIR, "py_c_out.csv")
        log_name = "python_gemm_demo.log"

        _write_tensor_bin(a_path, a_exec, self.dtype)
        _write_tensor_bin(b_path, b_exec, self.dtype)
        _assert_tensor_bin_size(a_path, m_pad, k_pad, "A", self.dtype)
        _assert_tensor_bin_size(b_path, k_pad, n_pad, "B", self.dtype)

        cmd = [
            "./run_noc_dma_pipeline.sh",
            "--dim",
            str(self.tile_dim),
            "--global-stride-kb",
            str(max(64, _required_global_stride_kb(self.tile_dim))),
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
            "--dma-overlap",
            str(self.dma_overlap),
            "--tensor-source",
            "file",
            "--tensor-a",
            os.path.relpath(a_path, TESTS_DIR),
            "--tensor-b",
            os.path.relpath(b_path, TESTS_DIR),
            "--dump-c",
            os.path.relpath(c_path, TESTS_DIR),
            "--verify-c",
            "--orig-m",
            str(self.m),
            "--orig-n",
            str(self.n),
            "--orig-k",
            str(self.k),
            "--log",
            log_name,
        ]

        subprocess.run(cmd, cwd=TESTS_DIR, check=True)
        c_pad = _read_csv_tensor(c_path, self.dtype)
        _validate_matrix("C_padded", c_pad, m_pad, n_pad)
        return crop_c(c_pad, self.m, self.n)


def matmul(
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    dma_overlap: int = 0,
    dtype: str = "int32",
) -> MatmulKernel:
    if min(m, n, k, block_m, block_n, block_k) <= 0:
        raise ValueError("M/N/K and block_M/N/K must be positive")
    if block_m != block_k:
        raise ValueError("current phase requires block_M == block_K")
    return MatmulKernel(
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        dma_overlap=dma_overlap,
        dtype=dtype,
    )


def run_gemm_via_pipeline(
    a: List[List[Scalar]],
    b: List[List[Scalar]],
    block_m: int,
    block_n: int,
    block_k: int,
    dma_overlap: int = 0,
    dtype: str = "int32",
) -> List[List[Scalar]]:
    m = len(a)
    k = len(a[0]) if m > 0 else 0
    n = len(b[0]) if len(b) > 0 else 0
    kernel = matmul(
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        dma_overlap=dma_overlap,
        dtype=dtype,
    )
    return kernel(a, b)


def samp_ref(a: List[List[int]], b: List[List[int]]) -> List[List[int]]:
    m = len(a)
    k = len(a[0])
    n = len(b[0])
    c = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for kk in range(k):
            av = a[i][kk]
            for j in range(n):
                c[i][j] += av * b[kk][j]
    return c


if __name__ == "__main__":
    random.seed(123)

    M, N, K = 1024, 16, 1024
    BM, BN, BK = 64, 4, 64

    a = [[random.randint(-3, 3) for _ in range(K)] for _ in range(M)]
    b = [[random.randint(-3, 3) for _ in range(N)] for _ in range(K)]

    kernel = matmul(M, N, K, BM, BN, BK, dtype="fp32")
    c = kernel(a, b)
    ref = samp_ref(a, b)

    ok = c == ref
    print(f"python_gemm_demo: {'PASS' if ok else 'FAIL'}")
