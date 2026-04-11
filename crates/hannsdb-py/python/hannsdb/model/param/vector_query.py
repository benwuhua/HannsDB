from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ...extension.rerank_function import ReRanker


def _flatten_vector(value: Any):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_vector(item)
        return

    yield value


def _normalize_vector(vector: Any) -> list[float]:
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is optional at runtime
        np = None

    if np is not None and isinstance(vector, np.ndarray):
        return [float(item) for item in np.asarray(vector).reshape(-1).tolist()]

    if isinstance(vector, (str, bytes, bytearray, dict, set, frozenset)):
        raise TypeError("vector must be a list, tuple, or numpy.ndarray of numbers")
    if isinstance(vector, list) or isinstance(vector, tuple):
        return [float(item) for item in _flatten_vector(vector)]
    raise TypeError("vector must be a list, tuple, or numpy.ndarray of numbers")


class SparseVector:
    """Sparse vector with separate indices and values arrays."""

    __slots__ = ("_indices", "_values")

    def __init__(self, indices: list[int], values: list[float]):
        if len(indices) != len(values):
            raise ValueError("indices and values must have the same length")
        self._indices = list(indices)
        self._values = [float(v) for v in values]

    @property
    def indices(self) -> list[int]:
        return list(self._indices)

    @property
    def values(self) -> list[float]:
        return list(self._values)

    def __repr__(self) -> str:
        return f"SparseVector(indices={self._indices}, values={self._values})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SparseVector):
            return NotImplemented
        return self._indices == other._indices and self._values == other._values


@dataclass
class VectorQuery:
    field_name: str
    vector: Any  # Can be list[float], numpy.ndarray, or SparseVector
    param: Optional[Any] = None

    def __post_init__(self) -> None:
        # Only normalize if it's a dense vector (not SparseVector)
        if not isinstance(self.vector, SparseVector):
            self.vector = _normalize_vector(self.vector)


@dataclass
class QueryGroupBy:
    field_name: str


@dataclass
class QueryContext:
    top_k: int = 10
    filter: Optional[str] = None
    output_fields: Optional[list[str]] = None
    include_vector: bool = False
    queries: list[VectorQuery] = field(default_factory=list)
    query_by_id: Optional[list[int | str] | int | str] = None
    query_by_id_field_name: Optional[str] = None
    group_by: Optional[QueryGroupBy] = None
    reranker: Optional[ReRanker] = None

    def __post_init__(self) -> None:
        if not isinstance(self.queries, list):
            self.queries = list(self.queries)
        if self.query_by_id is not None and not isinstance(self.query_by_id, list):
            if isinstance(self.query_by_id, str):
                self.query_by_id = [self.query_by_id]
            else:
                try:
                    self.query_by_id = list(self.query_by_id)
                except TypeError:
                    self.query_by_id = [self.query_by_id]
        if self.output_fields is not None and isinstance(self.output_fields, str):
            self.output_fields = [self.output_fields]
        elif self.output_fields is not None and not isinstance(self.output_fields, list):
            self.output_fields = list(self.output_fields)


__all__ = ["QueryContext", "QueryGroupBy", "SparseVector", "VectorQuery"]
