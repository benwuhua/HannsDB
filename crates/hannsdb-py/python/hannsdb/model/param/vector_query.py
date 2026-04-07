from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from ..._native import VectorQuery


@dataclass(slots=True)
class QueryContext:
    top_k: int = 10
    filter: Optional[str] = None
    output_fields: Optional[list[str]] = None
    include_vector: bool = False
    queries: list[VectorQuery] = field(default_factory=list)
    query_by_id: Optional[str] = None
    group_by: Optional[object] = None
    reranker: Optional[object] = None

    def __post_init__(self) -> None:
        if not isinstance(self.queries, list):
            self.queries = list(self.queries)


__all__ = ["QueryContext", "VectorQuery"]
