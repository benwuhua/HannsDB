from __future__ import annotations

import os
from dataclasses import dataclass, replace
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..extension.rerank_function import ReRanker
from ..model.param.vector_query import QueryContext
from ..model.schema.collection_schema import CollectionSchema


@dataclass(slots=True)
class QueryExecutor:
    schema: CollectionSchema

    @staticmethod
    def _query_concurrency() -> int:
        for env_name in ("ZVEC_QUERY_CONCURRENCY", "HANNSDB_QUERY_CONCURRENCY"):
            raw_value = os.getenv(env_name)
            if raw_value is None:
                continue
            concurrency = int(raw_value)
            return max(1, concurrency)
        return 1

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
        fan_out_contexts = []
        for query in context.queries:
            count = query_name_counts.get(query.field_name, 0) + 1
            query_name_counts[query.field_name] = count
            result_key = query.field_name if count == 1 else f"{query.field_name}#{count}"
            fan_out_contexts.append(
                (
                    result_key,
                    replace(
                        context,
                        queries=[query],
                        query_by_id=None,
                        group_by=None,
                        reranker=None,
                    ),
                )
            )

        concurrency = self._query_concurrency()
        if concurrency > 1 and len(fan_out_contexts) > 1:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    (result_key, executor.submit(collection.query_context, query_context))
                    for result_key, query_context in fan_out_contexts
                ]
                for result_key, future in futures:
                    query_results[result_key] = future.result()
        else:
            for result_key, query_context in fan_out_contexts:
                # `top_k` still controls per-query candidate depth on the fan-out path;
                # the reranker decides the final cutoff via its own `topn`.
                query_results[result_key] = collection.query_context(query_context)

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
