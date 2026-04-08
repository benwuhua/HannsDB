from __future__ import annotations

from dataclasses import dataclass

from ... import _native as _native_module

__all__ = ["HnswIndexParam", "IVFIndexParam", "HnswQueryParam"]


def _normalize_text(name, value):
    if value is None:
        return None
    if type(value) is not str:
        raise TypeError(f"{name} must be a string or None")
    return value.strip().lower()


def _require_int(name, value):
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    return value


def _validate_metric_type(name, value):
    normalized = _normalize_text(name, value)
    if normalized is None:
        return None
    if normalized not in {"l2", "cosine", "ip"}:
        raise ValueError(f"unsupported {name} value")
    return normalized


def _validate_quantize_type(name, value):
    normalized = _normalize_text(name, value)
    if normalized not in {"undefined", "fp16", "int8", "int4"}:
        raise ValueError(f"unsupported {name} value")
    return normalized


@dataclass(frozen=True)
class HnswIndexParam:
    metric_type: str | None = None
    m: int = 16
    ef_construction: int = 64
    quantize_type: str = "undefined"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metric_type",
            _validate_metric_type("metric_type", self.metric_type),
        )
        object.__setattr__(self, "m", _require_int("m", self.m))
        object.__setattr__(
            self,
            "ef_construction",
            _require_int("ef_construction", self.ef_construction),
        )
        object.__setattr__(
            self,
            "quantize_type",
            _validate_quantize_type("quantize_type", self.quantize_type),
        )

    def _get_native(self):
        return _native_module.HnswIndexParam(
            metric_type=self.metric_type,
            m=self.m,
            ef_construction=self.ef_construction,
            quantize_type=self.quantize_type,
        )


@dataclass(frozen=True)
class IVFIndexParam:
    metric_type: str | None = None
    nlist: int = 1024

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metric_type",
            _validate_metric_type("metric_type", self.metric_type),
        )
        object.__setattr__(self, "nlist", _require_int("nlist", self.nlist))

    def _get_native(self):
        return _native_module.IVFIndexParam(
            metric_type=self.metric_type,
            nlist=self.nlist,
        )


@dataclass(frozen=True)
class HnswQueryParam:
    ef: int = 32
    is_using_refiner: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "ef", _require_int("ef", self.ef))
        if type(self.is_using_refiner) is not bool:
            raise TypeError("is_using_refiner must be a bool")

    def _get_native(self):
        return _native_module.HnswQueryParam(
            ef=self.ef,
            is_using_refiner=self.is_using_refiner,
        )
