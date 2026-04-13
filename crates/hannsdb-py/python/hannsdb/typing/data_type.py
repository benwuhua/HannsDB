from __future__ import annotations

from ._enum import _StringEnum

__all__ = ["DataType"]


class DataType(_StringEnum):
    String = "string"
    Int32 = "int32"
    Int64 = "int64"
    UInt32 = "uint32"
    UInt64 = "uint64"
    Float = "float"
    Float64 = "float64"
    Bool = "bool"
    VectorFp32 = "vector_fp32"
