from __future__ import annotations

import json
import threading
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


def _wrap_collection_option(option):
    if option is None:
        return CollectionOption()
    if isinstance(option, CollectionOption):
        return option
    return CollectionOption(read_only=option.read_only, enable_mmap=option.enable_mmap)


def _coerce_doc_to_native(doc):
    native_getter = getattr(doc, "_get_native", None)
    if native_getter is not None:
        return native_getter()
    return doc


def _coerce_docs_to_native(docs):
    return [_coerce_doc_to_native(doc) for doc in list(docs)]


def _coerce_docs_input(docs):
    native_doc_type = getattr(_native_module, "Doc", None)
    if isinstance(docs, Doc) or (
        native_doc_type is not None and isinstance(docs, native_doc_type)
    ):
        return [docs]
    return list(docs)


def _coerce_id_input(ids):
    if isinstance(ids, str):
        return True, [ids]
    return False, ids


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
        if vectors is not None:
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


def _merged_update_doc(current, patch):
    merged_fields = current.fields
    merged_fields.update(patch.fields)

    merged_vectors = current.vectors
    merged_vectors.update(patch.vectors)

    field_name = current.field_name
    if merged_vectors and field_name not in merged_vectors:
        if patch.field_name in merged_vectors:
            field_name = patch.field_name
        else:
            field_name = next(iter(merged_vectors))

    return Doc(
        id=current.id,
        fields=merged_fields,
        vectors=merged_vectors,
        field_name=field_name,
    )


def _build_query_context(
    vectors=None,
    output_fields=None,
    topk: int = 100,
    filter: str | None = None,
    include_vector: bool = False,
    reranker=None,
    query_by_id=None,
    query_by_id_field_name=None,
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
        query_by_id_field_name=query_by_id_field_name,
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
        self._core_lock = threading.RLock()
        self._schema = _coerce_collection_schema(schema) if schema is not None else None
        self._querier = _build_query_executor(self._schema) if self._schema is not None else None
        self._option = _wrap_collection_option(getattr(core_collection, "option", None))

    @classmethod
    def _from_core(
        cls, core_collection, schema: CollectionSchema | None = None
    ) -> "Collection":
        if not core_collection:
            raise ValueError("Collection is None")
        inst = cls.__new__(cls)
        inst._core = core_collection
        inst._core_lock = threading.RLock()
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
        inst._option = _wrap_collection_option(getattr(core_collection, "option", None))
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
    def option(self) -> CollectionOption:
        return self._option

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
        query_by_id_field_name=None,
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
                query_by_id_field_name=query_by_id_field_name,
                group_by=group_by,
            )
        return _wrap_doc_result(self._querier.execute(self, query_context))

    def query_context(self, context):
        with self._core_lock:
            return _wrap_doc_result(self._core.query_context(context))

    def insert(self, docs):
        docs = _coerce_docs_input(docs)
        with self._core_lock:
            return self._core.insert(_coerce_docs_to_native(docs))

    def upsert(self, docs):
        docs = _coerce_docs_input(docs)
        with self._core_lock:
            return self._core.upsert(_coerce_docs_to_native(docs))

    def update(self, docs):
        patches = [_wrap_doc(doc) for doc in _coerce_docs_input(docs)]
        ids = [doc.id for doc in patches]
        unique_ids = list(dict.fromkeys(ids))
        with self._core_lock:
            fetched_docs = _wrap_doc_result(self._core.fetch(unique_ids))
            current_docs = {doc.id: doc for doc in fetched_docs}

            for doc_id in unique_ids:
                if doc_id not in current_docs:
                    raise KeyError(doc_id)

            merged_docs = {}
            for patch in patches:
                merged = _merged_update_doc(current_docs[patch.id], patch)
                current_docs[patch.id] = merged
                merged_docs[patch.id] = merged

            return self._core.upsert(
                _coerce_docs_to_native([merged_docs[doc_id] for doc_id in unique_ids if doc_id in merged_docs])
            )

    def fetch(self, ids):
        single_id, ids = _coerce_id_input(ids)
        with self._core_lock:
            result = _wrap_doc_result(self._core.fetch(ids))
        if single_id:
            return result[0] if result else None
        return result

    def delete(self, ids):
        _, ids = _coerce_id_input(ids)
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
