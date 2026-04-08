from __future__ import annotations

from ._enum import _StringEnum

__all__ = ["QuantizeType"]


class QuantizeType(_StringEnum):
    Undefined = "undefined"
    Fp16 = "fp16"
    Int8 = "int8"
    Int4 = "int4"
