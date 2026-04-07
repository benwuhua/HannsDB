from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..model.param.vector_query import QueryContext
from ..model.schema.collection_schema import CollectionSchema


@dataclass(slots=True)
class QueryExecutor:
    schema: CollectionSchema

    def execute(self, collection: Any, context: QueryContext):
        if context.include_vector:
            raise NotImplementedError("include_vector is not supported by the Python facade yet")
        if context.query_by_id is not None:
            raise NotImplementedError("query_by_id is not supported by the Python facade yet")
        if context.reranker is not None:
            raise NotImplementedError("reranker is not supported by the Python facade yet")
        if context.group_by is not None:
            raise NotImplementedError("group_by is not supported by the Python facade yet")
        if len(context.queries) != 1:
            raise NotImplementedError("Python facade currently supports exactly one vector query")

        query = context.queries[0]
        return collection.query(
            vectors=query,
            output_fields=context.output_fields,
            topk=context.top_k,
            filter=context.filter,
        )


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
