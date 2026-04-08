from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .. import _native as _native_module
from ..model.doc import Doc
from ..model.param import CollectionOption, HnswIndexParam, IVFIndexParam
from ..model.schema.collection_schema import CollectionSchema, _coerce_collection_schema
from ..model.schema.field_schema import FieldSchema, VectorSchema

__all__ = ["Collection", "create_and_open", "open"]


def _build_query_executor(schema: CollectionSchema):
    from ..executor import QueryExecutorFactory

    return QueryExecutorFactory.create(schema).build()


def _schema_to_native(schema):
    native_getter = getattr(schema, "_get_native", None)
    if native_getter is not None:
        return native_getter()
    return schema


def _resolve_index_target(schema: CollectionSchema, field_name: str):
    vector = None
    scalar = None
    try:
        vector = schema.vector(field_name)
    except KeyError:
        pass
    try:
        scalar = schema.field(field_name)
    except KeyError:
        pass

    if vector is not None and scalar is not None:
        raise ValueError(
            f"ambiguous field name {field_name!r}: matches both vector and scalar schemas"
        )
    if vector is not None:
        return "vector", vector
    if scalar is not None:
        return "scalar", scalar
    raise KeyError(field_name)


def _native_value(value):
    native_getter = getattr(value, "_get_native", None)
    if native_getter is not None:
        return native_getter()
    return value


def _coerce_doc_to_native(doc):
    native_getter = getattr(doc, "_get_native", None)
    if native_getter is not None:
        return native_getter()
    return doc


def _coerce_docs_to_native(docs):
    return [_coerce_doc_to_native(doc) for doc in list(docs)]


def _is_native_collection(core_collection) -> bool:
    native_collection_type = getattr(_native_module, "Collection", None)
    return native_collection_type is not None and isinstance(core_collection, native_collection_type)


def _doc_vector_count(doc) -> int:
    vectors = getattr(doc, "vectors", None)
    if vectors is None:
        return 0
    try:
        return len(vectors)
    except TypeError:
        return len(list(vectors))


def _raise_if_multi_vector_write_unsupported(core_collection, docs):
    if not _is_native_collection(core_collection):
        return
    for doc in docs:
        if _doc_vector_count(doc) > 1:
            raise NotImplementedError(
                "multi-vector document writes are not supported yet; only the primary vector can be written"
            )


def _wrap_doc_result(result):
    if isinstance(result, (list, tuple)):
        return [_wrap_doc(item) for item in result]
    return result


def _wrap_doc(item):
    if isinstance(item, Doc):
        return item
    native_doc_type = getattr(_native_module, "Doc", None)
    if native_doc_type is not None and isinstance(item, native_doc_type):
        return Doc._from_native(item)
    if hasattr(item, "id") and hasattr(item, "fields"):
        vectors = getattr(item, "vectors", None)
        if vectors:
            return Doc(
                id=getattr(item, "id"),
                score=getattr(item, "score", None),
                fields=getattr(item, "fields", None),
                vectors=vectors,
                field_name=getattr(item, "field_name", "dense"),
            )
        vector = getattr(item, "vector", None)
        if vector is not None and hasattr(item, "field_name"):
            return Doc(
                id=getattr(item, "id"),
                score=getattr(item, "score", None),
                fields=getattr(item, "fields", None),
                vector=vector,
                field_name=getattr(item, "field_name"),
            )
        return Doc(
            id=getattr(item, "id"),
            score=getattr(item, "score", None),
            fields=getattr(item, "fields", None),
        )
    return item


def _build_query_context(
    vectors=None,
    output_fields=None,
    topk: int = 100,
    filter: str | None = None,
    include_vector: bool = False,
    reranker=None,
    query_by_id=None,
    group_by=None,
):
    from ..model.param.vector_query import QueryContext

    if vectors is None:
        query_list = []
    elif isinstance(vectors, (list, tuple)):
        query_list = list(vectors)
    else:
        query_list = [vectors]

    return QueryContext(
        top_k=topk,
        filter=filter,
        output_fields=output_fields,
        include_vector=include_vector,
        queries=query_list,
        query_by_id=query_by_id,
        group_by=group_by,
        reranker=reranker,
    )


def _field_data_type_name(data_type: str) -> str:
    mapping = {
        "Bool": "bool",
        "Float64": "float64",
        "Int64": "int64",
        "String": "string",
        "VectorFp32": "vector_fp32",
    }
    return mapping.get(data_type, data_type.lower())


def _index_param_from_metadata(metadata: dict[str, Any]):
    kind = str(metadata.get("kind", "")).lower()
    metric = metadata.get("metric")
    if kind == "hnsw":
        return HnswIndexParam(
            metric_type=metric,
            m=int(metadata.get("m", 16)),
            ef_construction=int(metadata.get("ef_construction", 128)),
            quantize_type=metadata.get("quantize_type", "undefined"),
        )
    if kind == "ivf":
        return IVFIndexParam(
            metric_type=metric,
            nlist=int(metadata.get("nlist", 1024)),
        )
    return None


def _schema_from_current_metadata(metadata: dict[str, Any]) -> CollectionSchema:
    vectors = []
    for vector in metadata.get("vectors", []):
        vectors.append(
            VectorSchema(
                name=vector["name"],
                data_type=_field_data_type_name(vector["data_type"]),
                dimension=int(vector.get("dimension", 0)),
                index_param=_index_param_from_metadata(vector["index_param"])
                if vector.get("index_param") is not None
                else None,
            )
        )

    fields = []
    for field in metadata.get("fields", []):
        fields.append(
            FieldSchema(
                name=field["name"],
                data_type=_field_data_type_name(field["data_type"]),
                nullable=bool(field.get("nullable", False)),
                array=bool(field.get("array", False)),
            )
        )

    primary_vector = metadata.get("primary_vector") or (
        vectors[0].name if vectors else "vector"
    )
    return CollectionSchema(
        name=metadata["name"],
        fields=fields,
        vectors=vectors,
        primary_vector=primary_vector,
    )


def _schema_from_legacy_metadata(metadata: dict[str, Any]) -> CollectionSchema:
    fields = []
    for field in metadata.get("fields", []):
        fields.append(
            FieldSchema(
                name=field["name"],
                data_type=_field_data_type_name(field["data_type"]),
                nullable=bool(field.get("nullable", False)),
                array=bool(field.get("array", False)),
            )
        )

    primary_vector = metadata.get("primary_vector", "vector")
    vectors = []
    dimension = int(metadata.get("dimension", 0))
    if dimension > 0:
        vectors.append(
            VectorSchema(
                name=primary_vector,
                data_type="vector_fp32",
                dimension=dimension,
                index_param=HnswIndexParam(
                    metric_type=metadata.get("metric"),
                    m=int(metadata.get("hnsw_m", 16)),
                    ef_construction=int(metadata.get("hnsw_ef_construction", 128)),
                ),
            )
        )
    return CollectionSchema(
        name=metadata["name"],
        fields=fields,
        vectors=vectors,
        primary_vector=primary_vector,
    )


def _schema_from_collection_metadata(path: Path) -> CollectionSchema:
    metadata = json.loads(path.read_text())
    if "vectors" in metadata:
        return _schema_from_current_metadata(metadata)
    if "dimension" in metadata:
        return _schema_from_legacy_metadata(metadata)
    raise ValueError(f"unsupported collection metadata format: {path}")


class Collection:
    def __init__(self, core_collection, schema: CollectionSchema | None = None):
        self._core = core_collection
        self._schema = _coerce_collection_schema(schema) if schema is not None else None
        self._querier = _build_query_executor(self._schema) if self._schema is not None else None

    @classmethod
    def _from_core(
        cls, core_collection, schema: CollectionSchema | None = None
    ) -> "Collection":
        if not core_collection:
            raise ValueError("Collection is None")
        inst = cls.__new__(cls)
        inst._core = core_collection
        if schema is None:
            metadata_path = (
                Path(core_collection.path)
                / "collections"
                / core_collection.collection_name
                / "collection.json"
            )
            schema = _schema_from_collection_metadata(metadata_path)
        schema = _coerce_collection_schema(schema)
        inst._schema = schema
        inst._querier = _build_query_executor(schema)
        return inst

    @property
    def path(self) -> str:
        return self._core.path

    @property
    def collection_name(self) -> str:
        return self._core.collection_name

    @property
    def schema(self) -> CollectionSchema:
        return self._schema

    @property
    def stats(self):
        return self._core.stats

    def query(
        self,
        vectors=None,
        output_fields=None,
        topk: int = 100,
        filter: str | None = None,
        include_vector: bool = False,
        reranker=None,
        query_by_id=None,
        group_by=None,
        query_context=None,
        context=None,
    ):
        from ..model.param.vector_query import QueryContext

        if query_context is not None and context is not None:
            raise TypeError("query() received both query_context and context")
        if query_context is None:
            query_context = context
        if query_context is None and isinstance(vectors, QueryContext):
            query_context = vectors
        if query_context is None:
            query_context = _build_query_context(
                vectors=vectors,
                output_fields=output_fields,
                topk=topk,
                filter=filter,
                include_vector=include_vector,
                reranker=reranker,
                query_by_id=query_by_id,
                group_by=group_by,
            )
        return _wrap_doc_result(self._querier.execute(self, query_context))

    def query_context(self, context):
        return _wrap_doc_result(self._core.query_context(context))

    def insert(self, docs):
        docs = list(docs)
        _raise_if_multi_vector_write_unsupported(self._core, docs)
        return self._core.insert(_coerce_docs_to_native(docs))

    def upsert(self, docs):
        docs = list(docs)
        _raise_if_multi_vector_write_unsupported(self._core, docs)
        return self._core.upsert(_coerce_docs_to_native(docs))

    def fetch(self, ids):
        return _wrap_doc_result(self._core.fetch(ids))

    def delete(self, ids):
        return self._core.delete(ids)

    def optimize(self, option=None):
        if option is None:
            return self._core.optimize()
        return self._core.optimize(_native_value(option))

    def flush(self):
        return self._core.flush()

    def destroy(self):
        return self._core.destroy()

    def create_index(self, field_name, index_param=None):
        if self._schema is None:
            raise RuntimeError("collection schema is required for index operations")
        kind, _ = _resolve_index_target(self._schema, field_name)
        if kind == "vector":
            return self.create_vector_index(field_name, index_param)
        if index_param is not None:
            raise NotImplementedError("scalar index params are not supported")
        return self.create_scalar_index(field_name)

    def drop_index(self, field_name):
        if self._schema is None:
            raise RuntimeError("collection schema is required for index operations")
        kind, _ = _resolve_index_target(self._schema, field_name)
        if kind == "vector":
            return self.drop_vector_index(field_name)
        return self.drop_scalar_index(field_name)

    def create_vector_index(self, field_name, index_param=None):
        return self._core.create_vector_index(field_name, _native_value(index_param))

    def drop_vector_index(self, field_name):
        return self._core.drop_vector_index(field_name)

    def create_scalar_index(self, field_name):
        return self._core.create_scalar_index(field_name)

    def drop_scalar_index(self, field_name):
        return self._core.drop_scalar_index(field_name)

    def list_vector_indexes(self):
        return self._core.list_vector_indexes()

    def list_scalar_indexes(self):
        return self._core.list_scalar_indexes()

    def __getattr__(self, name: str):
        return getattr(self._core, name)


def create_and_open(path, schema, option: CollectionOption | None = None):
    core_collection = _native_module.create_and_open(
        path,
        _schema_to_native(schema),
        _native_value(option),
    )
    return Collection._from_core(core_collection, schema=schema)


def open(path, option: CollectionOption | None = None):
    core_collection = _native_module.open(path, _native_value(option))
    return Collection._from_core(core_collection)
