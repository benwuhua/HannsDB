from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ..extension.rerank_function import ReRanker
from ..model.param.vector_query import QueryContext
from ..model.schema.collection_schema import CollectionSchema


@dataclass(slots=True)
class QueryExecutor:
    schema: CollectionSchema

    def execute(self, collection: Any, context: QueryContext):
        if context.include_vector:
            raise NotImplementedError("include_vector is not supported by the Python facade yet")
        if context.reranker is None:
            return collection.query_context(context)
        if not isinstance(context.reranker, ReRanker):
            raise NotImplementedError("reranker is not supported by the Python facade yet")
        if len(context.queries) == 0:
            raise ValueError("reranker requires at least one vector query")
        if context.query_by_id is not None:
            raise NotImplementedError("query_by_id is not supported by the Python facade yet")
        if context.group_by is not None:
            raise NotImplementedError("group_by is not supported by the Python facade yet")

        query_results = {}
        query_name_counts: dict[str, int] = {}
        for query in context.queries:
            count = query_name_counts.get(query.field_name, 0) + 1
            query_name_counts[query.field_name] = count
            result_key = query.field_name if count == 1 else f"{query.field_name}#{count}"
            # `top_k` still controls per-query candidate depth on the fan-out path;
            # the reranker decides the final cutoff via its own `topn`.
            query_results[result_key] = collection.query_context(
                replace(
                    context,
                    queries=[query],
                    query_by_id=None,
                    group_by=None,
                    reranker=None,
                )
            )

        return context.reranker.rerank(query_results)


@dataclass(slots=True)
class QueryExecutorFactory:
    schema: CollectionSchema

    @classmethod
    def create(cls, schema: CollectionSchema) -> "QueryExecutorFactory":
        if not isinstance(schema, CollectionSchema):
            raise TypeError("schema must be a CollectionSchema")
        return cls(schema=schema)

    def build(self) -> QueryExecutor:
        return QueryExecutor(schema=self.schema)


__all__ = ["QueryExecutor", "QueryExecutorFactory"]
