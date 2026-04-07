from __future__ import annotations

import os
from dataclasses import dataclass, replace
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from ..extension.rerank_function import ReRanker
from ..model.param.vector_query import QueryContext
from ..model.schema.collection_schema import CollectionSchema


@dataclass
class QueryExecutor:
    schema: CollectionSchema

    @staticmethod
    def _query_concurrency() -> int:
        for env_name in ("ZVEC_QUERY_CONCURRENCY", "HANNSDB_QUERY_CONCURRENCY"):
            raw_value = os.getenv(env_name)
            if raw_value is None:
                continue
            try:
                concurrency = int(raw_value)
            except ValueError as error:
                raise ValueError(f"{env_name} must be an integer, got {raw_value!r}") from error
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
            executor = ThreadPoolExecutor(max_workers=concurrency)
            futures = {}
            ordered_results: list[tuple[str, Any] | None] = [None] * len(fan_out_contexts)
            try:
                futures = {
                    executor.submit(collection.query_context, query_context): (
                        index,
                        result_key,
                    )
                    for index, (result_key, query_context) in enumerate(fan_out_contexts)
                }
                for future in as_completed(futures):
                    index, result_key = futures[future]
                    ordered_results[index] = (result_key, future.result())
            except Exception:
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=True, cancel_futures=False)
                for item in ordered_results:
                    if item is None:
                        raise RuntimeError("query fan-out completed without a result")
                    result_key, docs = item
                    query_results[result_key] = docs
        else:
            for result_key, query_context in fan_out_contexts:
                # `top_k` still controls per-query candidate depth on the fan-out path;
                # the reranker decides the final cutoff via its own `topn`.
                query_results[result_key] = collection.query_context(query_context)

        return context.reranker.rerank(query_results)


@dataclass
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
