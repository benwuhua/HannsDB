from __future__ import annotations

from ... import _native as _native_module

__all__ = ["FieldSchema", "VectorSchema", "_coerce_field_schema", "_coerce_vector_schema"]


def _normalize_data_type(data_type: str) -> str:
    value = str(data_type).strip().lower()
    aliases = {
        "bool": "bool",
        "float64": "float64",
        "int64": "int64",
        "string": "string",
        "vector_fp32": "vector_fp32",
    }
    return aliases.get(value, value)


def _coerce_field_schema(value) -> "FieldSchema":
    if isinstance(value, FieldSchema):
        return value
    return FieldSchema(
        name=value.name,
        data_type=value.data_type,
        nullable=bool(getattr(value, "nullable", False)),
        array=bool(getattr(value, "array", False)),
    )


def _coerce_vector_schema(value) -> "VectorSchema":
    if isinstance(value, VectorSchema):
        return value
    return VectorSchema(
        name=value.name,
        data_type=value.data_type,
        dimension=int(value.dimension),
        index_param=getattr(value, "index_param", None),
    )


class FieldSchema:
    __slots__ = ("_name", "_data_type", "_nullable", "_array")

    def __init__(self, name, data_type, nullable: bool = False, array: bool = False):
        self._name = str(name)
        self._data_type = _normalize_data_type(data_type)
        self._nullable = bool(nullable)
        self._array = bool(array)

    @property
    def name(self) -> str:
        return self._name

    @property
    def data_type(self) -> str:
        return self._data_type

    @property
    def nullable(self) -> bool:
        return self._nullable

    @property
    def array(self) -> bool:
        return self._array

    def _get_native(self):
        return _native_module.FieldSchema(
            self.name,
            self.data_type,
            self.nullable,
            self.array,
        )

    def _as_tuple(self):
        return self.name, self.data_type, self.nullable, self.array

    def __repr__(self) -> str:
        return (
            "FieldSchema("
            f"name={self.name!r}, data_type={self.data_type!r}, "
            f"nullable={self.nullable!r}, array={self.array!r})"
        )

    def __eq__(self, other) -> bool:
        return isinstance(other, FieldSchema) and self._as_tuple() == other._as_tuple()

    def __hash__(self) -> int:
        return hash(self._as_tuple())


class VectorSchema:
    __slots__ = ("_name", "_data_type", "_dimension", "_index_param")

    def __init__(self, name, data_type, dimension, index_param=None):
        self._name = str(name)
        self._data_type = _normalize_data_type(data_type)
        self._dimension = int(dimension)
        self._index_param = index_param

    @property
    def name(self) -> str:
        return self._name

    @property
    def data_type(self) -> str:
        return self._data_type

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def index_param(self):
        return self._index_param

    def _get_native(self):
        index_param = self.index_param
        native_getter = getattr(index_param, "_get_native", None)
        if native_getter is not None:
            index_param = native_getter()
        return _native_module.VectorSchema(
            self.name,
            self.data_type,
            self.dimension,
            index_param,
        )

    def __repr__(self) -> str:
        return (
            "VectorSchema("
            f"name={self.name!r}, data_type={self.data_type!r}, "
            f"dimension={self.dimension!r}, index_param={self.index_param!r})"
        )
