from __future__ import annotations

from dataclasses import dataclass

from ... import _native as _native_module

__all__ = [
    "FlatIndexParam",
    "HnswIndexParam",
    "HnswHvqIndexParam",
    "IvfUsqIndexParam",
    "IvfUsqQueryParam",
    "IVFIndexParam",
    "HnswQueryParam",
    "IVFQueryParam",
]


def _normalize_text(name, value):
    if value is None:
        return None
    if not isinstance(value, str):
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
class FlatIndexParam:
    metric_type: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metric_type",
            _validate_metric_type("metric_type", self.metric_type),
        )

    def _get_native(self):
        return _native_module.FlatIndexParam(
            metric_type=self.metric_type,
        )


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
class HnswHvqIndexParam:
    metric_type: str | None = "ip"
    m: int = 16
    m_max0: int = 32
    ef_construction: int = 100
    ef_search: int = 64
    nbits: int = 4

    def __post_init__(self) -> None:
        normalized_metric = _validate_metric_type("metric_type", self.metric_type)
        if normalized_metric is None:
            normalized_metric = "ip"
        object.__setattr__(self, "metric_type", normalized_metric)
        if self.metric_type != "ip":
            raise ValueError("hnsw_hvq currently supports only metric_type='ip'")
        object.__setattr__(self, "m", _require_int("m", self.m))
        object.__setattr__(self, "m_max0", _require_int("m_max0", self.m_max0))
        object.__setattr__(
            self,
            "ef_construction",
            _require_int("ef_construction", self.ef_construction),
        )
        object.__setattr__(self, "ef_search", _require_int("ef_search", self.ef_search))
        object.__setattr__(self, "nbits", _require_int("nbits", self.nbits))

    def _get_native(self):
        return _native_module.HnswHvqIndexParam(
            metric_type=self.metric_type,
            m=self.m,
            m_max0=self.m_max0,
            ef_construction=self.ef_construction,
            ef_search=self.ef_search,
            nbits=self.nbits,
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
class IvfUsqIndexParam:
    metric_type: str | None = None
    nlist: int = 1024
    bits_per_dim: int = 4
    rotation_seed: int = 42
    rerank_k: int = 64
    use_high_accuracy_scan: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metric_type",
            _validate_metric_type("metric_type", self.metric_type),
        )
        object.__setattr__(self, "nlist", _require_int("nlist", self.nlist))
        object.__setattr__(
            self,
            "bits_per_dim",
            _require_int("bits_per_dim", self.bits_per_dim),
        )
        object.__setattr__(
            self,
            "rotation_seed",
            _require_int("rotation_seed", self.rotation_seed),
        )
        object.__setattr__(self, "rerank_k", _require_int("rerank_k", self.rerank_k))
        if type(self.use_high_accuracy_scan) is not bool:
            raise TypeError("use_high_accuracy_scan must be a bool")

    def _get_native(self):
        return _native_module.IvfUsqIndexParam(
            metric_type=self.metric_type,
            nlist=self.nlist,
            bits_per_dim=self.bits_per_dim,
            rotation_seed=self.rotation_seed,
            rerank_k=self.rerank_k,
            use_high_accuracy_scan=self.use_high_accuracy_scan,
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


@dataclass(frozen=True)
class IVFQueryParam:
    nprobe: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "nprobe", _require_int("nprobe", self.nprobe))

    def _get_native(self):
        return _native_module.IVFQueryParam(
            nprobe=self.nprobe,
        )


@dataclass(frozen=True)
class IvfUsqQueryParam:
    nprobe: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "nprobe", _require_int("nprobe", self.nprobe))

    def _get_native(self):
        return _native_module.IvfUsqQueryParam(
            nprobe=self.nprobe,
        )
