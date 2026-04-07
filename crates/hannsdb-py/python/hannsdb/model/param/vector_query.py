from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from ..._native import VectorQuery


@dataclass(slots=True)
class QueryGroupBy:
    field_name: str


@dataclass(slots=True)
class QueryReranker:
    model: str


@dataclass(slots=True)
class QueryContext:
    top_k: int = 10
    filter: Optional[str] = None
    output_fields: Optional[list[str]] = None
    include_vector: bool = False
    queries: list[VectorQuery] = field(default_factory=list)
    query_by_id: Optional[list[str]] = None
    group_by: Optional[QueryGroupBy] = None
    reranker: Optional[QueryReranker] = None

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


__all__ = ["QueryContext", "QueryGroupBy", "QueryReranker", "VectorQuery"]
