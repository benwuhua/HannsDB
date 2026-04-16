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
    VectorInt8 = "vector_int8"
    SparseVectorFp32 = "sparse_vector_fp32"
    # SCREAMING_SNAKE_CASE aliases for zvec compatibility
    STRING = "string"
    INT32 = "int32"
    INT64 = "int64"
    UINT32 = "uint32"
    UINT64 = "uint64"
    FLOAT = "float"
    FLOAT64 = "float64"
    BOOL = "bool"
    VECTOR_FP32 = "vector_fp32"
    VECTOR_INT8 = "vector_int8"
    SPARSE_VECTOR_FP32 = "sparse_vector_fp32"
