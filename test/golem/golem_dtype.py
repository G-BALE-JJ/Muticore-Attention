#!/usr/bin/env python3

import struct
from typing import Iterable, Optional, Tuple


SUPPORTED_DTYPES = {"int32", "fp16", "fp32"}


def normalize_dtype(dtype: str) -> str:
    value = (dtype or "int32").strip().lower()
    aliases = {
        "float": "fp32",
        "float32": "fp32",
        "fp32": "fp32",
        "half": "fp16",
        "float16": "fp16",
        "fp16": "fp16",
        "int": "int32",
        "int32": "int32",
        "i32": "int32",
    }
    value = aliases.get(value, value)
    if value not in SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype: {dtype}")
    return value


def elem_nbytes(dtype: str) -> int:
    normalize_dtype(dtype)
    return 2 if normalize_dtype(dtype) == "fp16" else 4


def struct_scalar_fmt(dtype: str) -> str:
    value = normalize_dtype(dtype)
    if value == "int32":
        return "i"
    if value == "fp16":
        return "e"
    return "f"


def struct_vector_fmt(dtype: str, count: int) -> str:
    return f"<{count}{struct_scalar_fmt(dtype)}"


def cast_scalar(dtype: str, value):
    kind = normalize_dtype(dtype)
    if kind == "int32":
        return int(value)
    return float(value)


def zero_value(dtype: str):
    return cast_scalar(dtype, 0)


def parse_scalar_text(dtype: str, text: str):
    return cast_scalar(dtype, text)


def pack_values(dtype: str, values: Iterable) -> bytes:
    items = [cast_scalar(dtype, v) for v in values]
    return struct.pack(struct_vector_fmt(dtype, len(items)), *items)


def unpack_values(dtype: str, blob: bytes):
    count, rem = divmod(len(blob), elem_nbytes(dtype))
    if rem != 0:
        raise ValueError(
            f"blob size {len(blob)} is not divisible by element size {elem_nbytes(dtype)}"
        )
    return list(struct.unpack(struct_vector_fmt(dtype, count), blob))


def numpy_dtype_name(dtype: str) -> str:
    value = normalize_dtype(dtype)
    if value == "int32":
        return "int32"
    if value == "fp16":
        return "float16"
    return "float32"


def default_tolerance(dtype: str) -> Tuple[float, float]:
    value = normalize_dtype(dtype)
    if value == "int32":
        return (0.0, 0.0)
    if value == "fp16":
        return (1e-2, 1e-2)
    return (1e-5, 1e-4)


def values_close(
    dtype: str, got, ref, atol: Optional[float] = None, rtol: Optional[float] = None
):
    value = normalize_dtype(dtype)
    if value == "int32":
        return int(got) == int(ref)
    if atol is None or rtol is None:
        atol, rtol = default_tolerance(value)
    diff = abs(float(got) - float(ref))
    limit = float(atol) + float(rtol) * abs(float(ref))
    return diff <= limit
