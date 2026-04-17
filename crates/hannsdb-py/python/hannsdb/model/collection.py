from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

from .. import _native as _native_module
from ..model.doc import Doc
from ..model.param import (
    CollectionOption,
    HnswHvqIndexParam,
    HnswIndexParam,
    InvertIndexParam,
    IvfUsqIndexParam,
    IVFIndexParam,
)
from ..model.schema.collection_schema import CollectionSchema, _coerce_collection_schema
from ..model.schema.field_schema import (
    FieldSchema,
    VectorSchema,
    _coerce_field_schema,
    _coerce_vector_schema,
)

__all__ = [
    "Collection",
    "LanceCollection",
    "create_and_open",
    "open",
    "create_lance_collection",
    "open_lance_collection",
]

_INT_LITERAL_RE = re.compile(r"-?(0|[1-9][0-9]*)$")
_FLOAT_LITERAL_RE = re.compile(r"-?(0|[1-9][0-9]*)\.[0-9]+$")


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


class MutationResult(int):
    """An int subclass returned by insert/upsert/update/delete/delete_by_filter.

    Preserves all int semantics (comparison, arithmetic) while adding an .ok()
    method that returns True when the value is non-negative.
    """

    def ok(self) -> bool:
        return self >= 0


def _native_value(value):
    native_getter = getattr(value, "_get_native", None)
    if native_getter is not None:
        return native_getter()
    return value


def _normalize_scalar_index_param(index_param):
    if index_param is None:
        return None
    if type(index_param) is not InvertIndexParam:
        raise NotImplementedError(
            f"unsupported scalar index param type: {index_param.__class__.__name__}"
        )
    return _native_value(index_param)


def _wrap_collection_option(option):
    if option is None:
        return CollectionOption()
    if isinstance(option, CollectionOption):
        return option
    return CollectionOption(read_only=option.read_only, enable_mmap=option.enable_mmap)


def _replace_collection_schema(
    schema: CollectionSchema,
    *,
    fields=None,
    vectors=None,
) -> CollectionSchema:
    return CollectionSchema(
        name=schema.name,
        fields=schema.fields if fields is None else fields,
        vectors=schema.vectors if vectors is None else vectors,
        primary_vector=schema.primary_vector,
    )


def _normalize_add_column_input(
    field_or_name,
    data_type="string",
    nullable=False,
    array=False,
    expression="",
    option=None,
) -> tuple[FieldSchema, str, Any]:
    if isinstance(field_or_name, VectorSchema):
        raise NotImplementedError("vector add_column is not supported yet")
    if hasattr(field_or_name, "dimension") and hasattr(field_or_name, "index_param"):
        _coerce_vector_schema(field_or_name)
        raise NotImplementedError("vector add_column is not supported yet")

    if isinstance(field_or_name, FieldSchema) or hasattr(field_or_name, "data_type"):
        field_schema = _coerce_field_schema(field_or_name)
    else:
        field_schema = FieldSchema(
            name=field_or_name,
            data_type=data_type,
            nullable=nullable,
            array=array,
        )

    expression = _normalize_add_column_expression(expression, field_schema)

    return field_schema, expression, option


def _normalize_add_column_expression(expression: str, field_schema: FieldSchema) -> str:
    if expression == "":
        return ""

    expr = expression.strip()
    if expr == "":
        return ""
    if field_schema.array:
        raise NotImplementedError("add_column expression does not support array fields yet")

    data_type = str(field_schema.data_type)

    if expr == "null":
        if not field_schema.nullable:
            raise ValueError("null expression requires a nullable field")
        return expr

    if expr in {"true", "false"}:
        if data_type != "bool":
            raise ValueError("boolean constant requires a bool destination field")
        return expr

    if expr.startswith('"'):
        if not expr.endswith('"') or len(expr) < 2:
            raise ValueError("invalid string literal")
        inner = expr[1:-1]
        if "\\" in inner or '"' in inner:
            raise NotImplementedError(
                "add_column expression does not support string escapes or embedded quotes yet"
            )
        if data_type != "string":
            raise ValueError("string constant requires a string destination field")
        return expr

    if expr.startswith("+"):
        raise NotImplementedError("add_column expression supports only constant literals")

    if "e" in expr.lower():
        raise NotImplementedError("add_column expression does not support scientific notation")

    if _FLOAT_LITERAL_RE.fullmatch(expr):
        if data_type not in {"float", "float64"}:
            raise ValueError("float constant requires a float destination field")
        return expr

    if _INT_LITERAL_RE.fullmatch(expr):
        value = int(expr)
        if data_type == "int32" and not (-(2**31) <= value < 2**31):
            raise ValueError("int32 constant is out of range")
        if data_type == "uint32":
            if value < 0:
                raise ValueError("uint32 constant must be >= 0")
            if value >= 2**32:
                raise ValueError("uint32 constant is out of range")
        if data_type == "uint64":
            if value < 0:
                raise ValueError("uint64 constant must be >= 0")
            if value >= 2**64:
                raise ValueError("uint64 constant is out of range")
        if data_type not in {"int64", "int32", "uint32", "uint64", "float", "float64"}:
            raise ValueError("numeric constant requires a numeric destination field")
        return expr

    raise NotImplementedError("add_column expression supports only constant literals")


def _normalize_alter_column_input(
    schema: CollectionSchema | None,
    field_name,
    new_name=None,
    field_schema=None,
    option=None,
) -> tuple[str, str | None, FieldSchema | None, Any]:
    old_name = str(field_name)
    if field_schema is not None:
        if schema is None:
            raise RuntimeError("collection schema is required for alter_column migration")
        target_field = _coerce_field_schema(field_schema)
        current_field = schema.field(old_name)

        if new_name not in (None, "") and str(new_name) != target_field.name:
                raise ValueError("alter_column new_name must match field_schema.name")
        if target_field.nullable != current_field.nullable:
            raise NotImplementedError("alter_column field_schema migration does not support nullable changes yet")
        if target_field.array != current_field.array:
            raise NotImplementedError("alter_column field_schema migration does not support array changes yet")

        supported_widening = {
            ("int32", "int64"),
            ("uint32", "uint64"),
            ("float", "float64"),
        }
        if (current_field.data_type, target_field.data_type) not in supported_widening:
            raise NotImplementedError("alter_column field_schema migration supports only widening scalar conversions")
        return old_name, target_field.name, target_field, option

    if new_name in (None, ""):
        raise ValueError("alter_column rename requires new_name")
    return old_name, str(new_name), None, option


def _coerce_doc_to_native(doc):
    native_getter = getattr(doc, "_get_native", None)
    if native_getter is not None:
        return native_getter()
    return doc


def _coerce_docs_to_native(docs):
    return [_coerce_doc_to_native(doc) for doc in list(docs)]


def _coerce_scalar_value_for_field_schema(value, field_schema):
    data_type = str(field_schema.data_type)
    if field_schema.array and isinstance(value, (list, tuple)):
        return [_coerce_scalar_value_for_field_schema(item, field_schema._replace(array=False) if hasattr(field_schema, "_replace") else FieldSchema(name=field_schema.name, data_type=field_schema.data_type, nullable=field_schema.nullable, array=False)) for item in value]

    if data_type == "int64":
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return int(value)
        return value

    if data_type == "int32":
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and -(2**31) <= int(value) < 2**31:
            return int(value)
        return value

    if data_type == "uint32":
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and 0 <= int(value) < 2**32:
            return int(value)
        return value

    if data_type == "uint64":
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and 0 <= int(value) < 2**64:
            return int(value)
        return value

    if data_type in {"float", "float64"}:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return float(value)
        return value

    return value


def _validate_doc_nullable_fields(doc, schema: CollectionSchema | None, *, is_insert: bool) -> None:
    """Raise ValueError if the doc violates nullable=False constraints.

    For inserts/upserts (is_insert=True) every nullable=False field must be
    present and non-None.  For partial updates (is_insert=False) only
    explicitly-provided nullable=False fields are checked — missing ones are
    allowed because they will be filled from the existing stored document.
    """
    if schema is None:
        return
    fields = getattr(doc, "fields", {}) or {}
    for field_schema in schema.fields:
        if field_schema.nullable:
            continue
        name = field_schema.name
        if name in fields:
            if fields[name] is None:
                raise ValueError(
                    f"doc validate failed: field[{name}] is configured not nullable, "
                    f"but doc does not contain this field"
                )
        elif is_insert:
            raise ValueError(
                f"doc validate failed: field[{name}] is configured not nullable, "
                f"but doc does not contain this field"
            )


def _coerce_doc_to_collection_schema(doc, schema: CollectionSchema | None):
    if schema is None:
        return _wrap_doc(doc)

    doc = _wrap_doc(doc)
    normalized_fields = {}
    for name, value in doc.fields.items():
        try:
            field_schema = schema.field(name)
        except KeyError:
            normalized_fields[name] = value
            continue
        normalized_fields[name] = _coerce_scalar_value_for_field_schema(value, field_schema)
    return doc._replace(fields=normalized_fields)


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
    order_by=None,
):
    from ..model.param.vector_query import QueryContext

    if vectors is None:
        query_list = []
    elif isinstance(vectors, (list, tuple)):
        query_list = list(vectors)
    else:
        query_list = [vectors]

    # Extract VectorQuery objects that carry .id instead of .vector and promote
    # them to query_by_id/query_by_id_field_name on the context (only when the
    # caller hasn't already provided an explicit query_by_id).
    if query_by_id is None:
        id_queries = [q for q in query_list if getattr(q, "id", None) is not None]
        if id_queries:
            if len(id_queries) > 1:
                raise ValueError(
                    "At most one id-based VectorQuery may be provided per query call"
                )
            id_vq = id_queries[0]
            query_list = [q for q in query_list if getattr(q, "id", None) is None]
            query_by_id = id_vq.id
            query_by_id_field_name = id_vq.field_name

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
        order_by=order_by,
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
    if kind == "hnsw_hvq":
        return HnswHvqIndexParam(
            metric_type=metric,
            m=int(metadata.get("m", 16)),
            m_max0=int(metadata.get("m_max0", 32)),
            ef_construction=int(metadata.get("ef_construction", 100)),
            ef_search=int(metadata.get("ef_search", 64)),
            nbits=int(metadata.get("nbits", 4)),
        )
    if kind == "ivf":
        return IVFIndexParam(
            metric_type=metric,
            nlist=int(metadata.get("nlist", 1024)),
        )
    if kind == "ivf_usq":
        return IvfUsqIndexParam(
            metric_type=metric,
            nlist=int(metadata.get("nlist", 1024)),
            bits_per_dim=int(metadata.get("bits_per_dim", 4)),
            rotation_seed=int(metadata.get("rotation_seed", 42)),
            rerank_k=int(metadata.get("rerank_k", 64)),
            use_high_accuracy_scan=bool(metadata.get("use_high_accuracy_scan", False)),
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
        order_by=None,
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
                order_by=order_by,
            )
        return _wrap_doc_result(self._querier.execute(self, query_context))

    def query_context(self, context):
        with self._core_lock:
            return _wrap_doc_result(self._core.query_context(context))

    def insert(self, docs):
        coerced = []
        for doc in _coerce_docs_input(docs):
            _validate_doc_nullable_fields(doc, self._schema, is_insert=True)
            coerced.append(_coerce_doc_to_collection_schema(doc, self._schema))
        with self._core_lock:
            return MutationResult(self._core.insert(_coerce_docs_to_native(coerced)))

    def upsert(self, docs):
        coerced = []
        for doc in _coerce_docs_input(docs):
            _validate_doc_nullable_fields(doc, self._schema, is_insert=True)
            coerced.append(_coerce_doc_to_collection_schema(doc, self._schema))
        with self._core_lock:
            return MutationResult(self._core.upsert(_coerce_docs_to_native(coerced)))

    def update(self, docs):
        patches = []
        for doc in _coerce_docs_input(docs):
            _validate_doc_nullable_fields(doc, self._schema, is_insert=False)
            patches.append(_coerce_doc_to_collection_schema(doc, self._schema))
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

            return MutationResult(self._core.upsert(
                _coerce_docs_to_native([merged_docs[doc_id] for doc_id in unique_ids if doc_id in merged_docs])
            ))

    def fetch(self, ids):
        single_id, ids = _coerce_id_input(ids)
        with self._core_lock:
            result = _wrap_doc_result(self._core.fetch(ids))
        if single_id:
            return result[0] if result else None
        return result

    def delete(self, ids):
        _, ids = _coerce_id_input(ids)
        return MutationResult(self._core.delete(ids))

    def delete_by_filter(self, filter: str):
        return MutationResult(self._core.delete_by_filter(filter))

    def _set_schema(self, schema: CollectionSchema | None):
        self._schema = _coerce_collection_schema(schema) if schema is not None else None
        self._querier = _build_query_executor(self._schema) if self._schema is not None else None

    def add_column(
        self,
        field_name,
        data_type="string",
        nullable=False,
        array=False,
        *,
        expression="",
        option=None,
    ):
        field_schema, expression, option = _normalize_add_column_input(
            field_name,
            data_type=data_type,
            nullable=nullable,
            array=array,
            expression=expression,
            option=option,
        )
        result = self._core.add_column(
            field_schema.name,
            field_schema.data_type,
            field_schema.nullable,
            field_schema.array,
            expression,
            _native_value(option),
        )
        if self._schema is not None:
            fields = [*self._schema.fields, field_schema]
            self._set_schema(_replace_collection_schema(self._schema, fields=fields))
        return result

    def drop_column(self, field_name):
        result = self._core.drop_column(field_name)
        if self._schema is not None:
            fields = [field for field in self._schema.fields if field.name != field_name]
            self._set_schema(_replace_collection_schema(self._schema, fields=fields))
        return result

    def alter_column(self, field_name, new_name=None, *, field_schema=None, option=None):
        old_name, normalized_new_name, normalized_field_schema, option = _normalize_alter_column_input(
            self._schema,
            field_name,
            new_name=new_name,
            field_schema=field_schema,
            option=option,
        )
        list_scalar_indexes = getattr(self._core, "list_scalar_indexes", None)
        if (
            normalized_field_schema is not None
            and callable(list_scalar_indexes)
            and old_name in list_scalar_indexes()
        ):
            raise NotImplementedError(
                "alter_column field_schema migration is not supported for fields with scalar index descriptors"
            )
        result = self._core.alter_column(
            old_name,
            normalized_new_name or "",
            _native_value(normalized_field_schema),
            _native_value(option),
        )
        if self._schema is not None:
            fields = []
            for field in self._schema.fields:
                if field.name != old_name:
                    fields.append(field)
                elif normalized_field_schema is not None:
                    fields.append(normalized_field_schema)
                else:
                    fields.append(
                        FieldSchema(
                            name=normalized_new_name,
                            data_type=field.data_type,
                            nullable=field.nullable,
                            array=field.array,
                        )
                    )
            self._set_schema(_replace_collection_schema(self._schema, fields=fields))
        return result

    def optimize(self, option=None):
        if option is None:
            return self._core.optimize()
        return self._core.optimize(_native_value(option))

    def flush(self):
        return self._core.flush()

    def destroy(self):
        return self._core.destroy()

    def create_index(self, field_name, index_param=None, option=None):
        if self._schema is None:
            raise RuntimeError("collection schema is required for index operations")
        kind, _ = _resolve_index_target(self._schema, field_name)
        if kind == "vector":
            return self.create_vector_index(field_name, index_param, option)
        return self.create_scalar_index(field_name, index_param, option)

    def drop_index(self, field_name):
        if self._schema is None:
            raise RuntimeError("collection schema is required for index operations")
        kind, _ = _resolve_index_target(self._schema, field_name)
        if kind == "vector":
            return self.drop_vector_index(field_name)
        return self.drop_scalar_index(field_name)

    def create_vector_index(self, field_name, index_param=None, option=None):
        native_option = _native_value(option)
        if native_option is None:
            return self._core.create_vector_index(field_name, _native_value(index_param))
        return self._core.create_vector_index(field_name, _native_value(index_param), native_option)

    def drop_vector_index(self, field_name):
        return self._core.drop_vector_index(field_name)

    def create_scalar_index(self, field_name, index_param=None, option=None):
        facade_index_param = index_param  # keep facade for schema update
        native_index_param = _normalize_scalar_index_param(index_param)
        native_option = _native_value(option)
        if native_index_param is None and native_option is None:
            result = self._core.create_scalar_index(field_name)
        elif native_index_param is None:
            result = self._core.create_scalar_index(field_name, option=native_option)
        elif native_option is None:
            result = self._core.create_scalar_index(field_name, native_index_param)
        else:
            result = self._core.create_scalar_index(field_name, native_index_param, native_option)
        if self._schema is not None and facade_index_param is not None:
            fields = [
                FieldSchema(
                    name=f.name,
                    data_type=f.data_type,
                    nullable=f.nullable,
                    array=f.array,
                    index_param=facade_index_param if f.name == field_name else f.index_param,
                )
                for f in self._schema.fields
            ]
            self._set_schema(_replace_collection_schema(self._schema, fields=fields))
        return result

    def drop_scalar_index(self, field_name):
        result = self._core.drop_scalar_index(field_name)
        if self._schema is not None:
            fields = [
                FieldSchema(
                    name=f.name,
                    data_type=f.data_type,
                    nullable=f.nullable,
                    array=f.array,
                    index_param=None if f.name == field_name else f.index_param,
                )
                for f in self._schema.fields
            ]
            self._set_schema(_replace_collection_schema(self._schema, fields=fields))
        return result

    def list_vector_indexes(self):
        return self._core.list_vector_indexes()

    def list_scalar_indexes(self):
        return self._core.list_scalar_indexes()

    def __getattr__(self, name: str):
        return getattr(self._core, name)


def _normalize_storage(storage):
    normalized = str(storage).strip().lower()
    if normalized in {"hannsdb", "default", ""}:
        return "hannsdb"
    if normalized == "lance":
        return "lance"
    raise ValueError(f"unsupported storage backend: {storage}")


def create_and_open(path, schema, option: CollectionOption | None = None, *, storage="hannsdb"):
    storage = _normalize_storage(storage)
    if storage == "lance":
        return create_lance_collection(path, schema, [])
    core_collection = _native_module.create_and_open(
        path,
        _schema_to_native(schema),
        _native_value(option),
    )
    return Collection._from_core(core_collection, schema=schema)


def open(path, option: CollectionOption | None = None, *, storage="hannsdb", schema=None):
    storage = _normalize_storage(storage)
    if storage == "lance":
        return open_lance_collection(path, schema)
    core_collection = _native_module.open(path, _native_value(option))
    return Collection._from_core(core_collection)


class LanceCollection:
    def __init__(self, core_collection, schema: CollectionSchema):
        self._core = core_collection
        self._schema = _coerce_collection_schema(schema)
        self._core_lock = threading.RLock()

    @property
    def name(self) -> str:
        return self._core.name

    @property
    def uri(self) -> str:
        return self._core.uri

    @property
    def schema(self) -> CollectionSchema:
        return self._schema

    def insert(self, docs):
        coerced = []
        for doc in _coerce_docs_input(docs):
            _validate_doc_nullable_fields(doc, self._schema, is_insert=True)
            coerced.append(_coerce_doc_to_collection_schema(doc, self._schema))
        with self._core_lock:
            return MutationResult(self._core.insert(_coerce_docs_to_native(coerced)))

    def upsert(self, docs):
        coerced = []
        for doc in _coerce_docs_input(docs):
            _validate_doc_nullable_fields(doc, self._schema, is_insert=True)
            coerced.append(_coerce_doc_to_collection_schema(doc, self._schema))
        with self._core_lock:
            return MutationResult(self._core.upsert(_coerce_docs_to_native(coerced)))

    def fetch(self, ids):
        single_id, ids = _coerce_id_input(ids)
        with self._core_lock:
            result = _wrap_doc_result(self._core.fetch(ids))
        if single_id:
            return result[0] if result else None
        return result

    def delete(self, ids):
        _, ids = _coerce_id_input(ids)
        with self._core_lock:
            return MutationResult(self._core.delete(ids))

    def hanns_index_path(self, field_name):
        return Path(self._core.hanns_index_path(field_name))

    def optimize_hanns(self, field_name, metric="l2"):
        with self._core_lock:
            return self._core.optimize_hanns(field_name, metric)

    def search(self, vector, topk=10, metric="l2"):
        with self._core_lock:
            return _wrap_doc_result(self._core.search(vector, topk, metric))

    def destroy(self):
        with self._core_lock:
            return self._core.destroy()


def create_lance_collection(path, schema, docs):
    coerced_schema = _coerce_collection_schema(schema)
    coerced_docs = []
    for doc in _coerce_docs_input(docs):
        _validate_doc_nullable_fields(doc, coerced_schema, is_insert=True)
        coerced_docs.append(_coerce_doc_to_collection_schema(doc, coerced_schema))
    core_collection = _native_module.create_lance_collection(
        path,
        _schema_to_native(coerced_schema),
        _coerce_docs_to_native(coerced_docs),
    )
    return LanceCollection(core_collection, coerced_schema)


def _infer_single_lance_collection_name(path) -> str:
    collections_dir = Path(path) / "collections"
    names = sorted(
        child.name[: -len(".lance")]
        for child in collections_dir.glob("*.lance")
        if child.is_dir()
    )
    if len(names) == 1:
        return names[0]
    if not names:
        raise ValueError(
            f"cannot infer Lance collection schema: no .lance datasets found under {collections_dir}"
        )
    raise ValueError(
        "cannot infer Lance collection schema: multiple .lance datasets found; pass schema explicitly"
    )


def open_lance_collection(path, schema=None):
    if schema is None:
        name = _infer_single_lance_collection_name(path)
        core_collection = _native_module.open_lance_collection_infer_schema(path, name)
        return LanceCollection(core_collection, core_collection.schema)
    coerced_schema = _coerce_collection_schema(schema)
    core_collection = _native_module.open_lance_collection(
        path,
        _schema_to_native(coerced_schema),
    )
    return LanceCollection(core_collection, coerced_schema)
