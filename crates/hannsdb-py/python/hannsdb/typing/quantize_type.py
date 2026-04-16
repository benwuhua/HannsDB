from __future__ import annotations

from ._enum import _StringEnum

__all__ = ["QuantizeType"]


class QuantizeType(_StringEnum):
    Undefined = "undefined"
    Fp16 = "fp16"
    Int8 = "int8"
    Int4 = "int4"
    # SCREAMING_SNAKE_CASE aliases
    UNDEFINED = "undefined"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
