from __future__ import annotations

from ._enum import _StringEnum

__all__ = ["MetricType"]


class MetricType(_StringEnum):
    L2 = "l2"
    Cosine = "cosine"
    Ip = "ip"
