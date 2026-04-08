from __future__ import annotations

from dataclasses import dataclass

from ... import _native as _native_module

__all__ = ["HnswIndexParam", "IVFIndexParam", "HnswQueryParam"]


def _normalize_text(value):
    if value is None:
        return None
    return str(value).strip().lower()


@dataclass(frozen=True, slots=True)
class HnswIndexParam:
    metric_type: str | None = None
    m: int = 16
    ef_construction: int = 64
    quantize_type: str = "undefined"

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric_type", _normalize_text(self.metric_type))
        object.__setattr__(self, "m", int(self.m))
        object.__setattr__(self, "ef_construction", int(self.ef_construction))
        object.__setattr__(self, "quantize_type", _normalize_text(self.quantize_type))

    def _get_native(self):
        return _native_module.HnswIndexParam(
            metric_type=self.metric_type,
            m=self.m,
            ef_construction=self.ef_construction,
            quantize_type=self.quantize_type,
        )


@dataclass(frozen=True, slots=True)
class IVFIndexParam:
    metric_type: str | None = None
    nlist: int = 1024

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric_type", _normalize_text(self.metric_type))
        object.__setattr__(self, "nlist", int(self.nlist))

    def _get_native(self):
        return _native_module.IVFIndexParam(
            metric_type=self.metric_type,
            nlist=self.nlist,
        )


@dataclass(frozen=True, slots=True)
class HnswQueryParam:
    ef: int = 32
    is_using_refiner: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "ef", int(self.ef))
        object.__setattr__(self, "is_using_refiner", bool(self.is_using_refiner))

    def _get_native(self):
        return _native_module.HnswQueryParam(
            ef=self.ef,
            is_using_refiner=self.is_using_refiner,
        )
