from __future__ import annotations

from ._enum import _StringEnum

__all__ = ["DataType"]


class DataType(_StringEnum):
    String = "string"
    Int64 = "int64"
    Float64 = "float64"
    Bool = "bool"
    VectorFp32 = "vector_fp32"
