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
    VectorFp16 = "vector_fp16"
    VectorFp64 = "vector_fp64"
    SparseVectorFp32 = "sparse_vector_fp32"
    SparseVectorFp16 = "sparse_vector_fp16"
    ArrayString = "array_string"
    ArrayInt32 = "array_int32"
    ArrayInt64 = "array_int64"
    ArrayFloat = "array_float"
    ArrayDouble = "array_double"
    ArrayBool = "array_bool"
    ArrayUInt32 = "array_uint32"
    ArrayUInt64 = "array_uint64"
    # SCREAMING_SNAKE_CASE aliases
    STRING = "string"
    INT32 = "int32"
    INT64 = "int64"
    UINT32 = "uint32"
    UINT64 = "uint64"
    FLOAT = "float"
    FLOAT64 = "float64"
    DOUBLE = "float64"  # alias for float64
    BOOL = "bool"
    VECTOR_FP32 = "vector_fp32"
    VECTOR_INT8 = "vector_int8"
    VECTOR_FP16 = "vector_fp16"
    VECTOR_FP64 = "vector_fp64"
    SPARSE_VECTOR_FP32 = "sparse_vector_fp32"
    SPARSE_VECTOR_FP16 = "sparse_vector_fp16"
    ARRAY_STRING = "array_string"
    ARRAY_INT32 = "array_int32"
    ARRAY_INT64 = "array_int64"
    ARRAY_FLOAT = "array_float"
    ARRAY_DOUBLE = "array_double"
    ARRAY_BOOL = "array_bool"
    ARRAY_UINT32 = "array_uint32"
    ARRAY_UINT64 = "array_uint64"
