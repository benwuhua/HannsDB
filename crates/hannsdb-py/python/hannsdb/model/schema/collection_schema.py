from __future__ import annotations

from ... import _native as _native_module
from .field_schema import (
    FieldSchema,
    VectorSchema,
    _coerce_field_schema,
    _coerce_vector_schema,
)

__all__ = ["CollectionSchema", "_coerce_collection_schema"]


def _coerce_collection_schema(value) -> "CollectionSchema":
    if isinstance(value, CollectionSchema):
        return value
    fields = [_coerce_field_schema(field) for field in getattr(value, "fields", [])]
    vectors = [_coerce_vector_schema(vector) for vector in getattr(value, "vectors", [])]
    primary_vector = getattr(value, "primary_vector", None)
    if primary_vector is None and vectors:
        primary_vector = vectors[0].name
    return CollectionSchema(
        name=value.name,
        fields=fields,
        vectors=vectors,
        primary_vector=primary_vector,
    )


class CollectionSchema:
    __slots__ = ("_name", "_primary_vector", "_fields", "_vectors")

    def __init__(
        self,
        name,
        vector_schema=None,
        fields=None,
        vectors=None,
        primary_vector=None,
    ):
        self._name = str(name)

        normalized_vectors = []
        if vector_schema is not None:
            normalized_vectors.append(_coerce_vector_schema(vector_schema))
        if vectors is not None:
            normalized_vectors.extend(_coerce_vector_schema(vector) for vector in vectors)
        if not normalized_vectors:
            raise ValueError("CollectionSchema requires at least one vector schema")

        self._vectors = tuple(normalized_vectors)
        self._primary_vector = (
            str(primary_vector)
            if primary_vector is not None
            else self._vectors[0].name
        )
        self._fields = tuple(_coerce_field_schema(field) for field in (fields or []))

    @property
    def name(self) -> str:
        return self._name

    @property
    def primary_vector(self) -> str:
        return self._primary_vector

    @property
    def fields(self):
        return list(self._fields)

    @property
    def vectors(self):
        return list(self._vectors)

    def field(self, name: str) -> FieldSchema:
        for field in self._fields:
            if field.name == name:
                return field
        raise KeyError(name)

    def vector(self, name: str) -> VectorSchema:
        for vector in self._vectors:
            if vector.name == name:
                return vector
        raise KeyError(name)

    def _get_native(self):
        return _native_module.CollectionSchema(
            self.name,
            fields=[field._get_native() for field in self._fields],
            vectors=[vector._get_native() for vector in self._vectors],
            primary_vector=self.primary_vector,
        )

    def _as_tuple(self):
        return self.name, self.primary_vector, self._fields, self._vectors

    def __repr__(self) -> str:
        return (
            "CollectionSchema("
            f"name={self.name!r}, primary_vector={self.primary_vector!r}, "
            f"fields={list(self._fields)!r}, vectors={list(self._vectors)!r})"
        )

    def __eq__(self, other) -> bool:
        return isinstance(other, CollectionSchema) and self._as_tuple() == other._as_tuple()

    def __hash__(self) -> int:
        return hash(self._as_tuple())
