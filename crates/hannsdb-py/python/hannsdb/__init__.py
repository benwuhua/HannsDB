from . import _native as _native_module
from ._native import *  # noqa: F401,F403
from . import typing as typing
from .typing import DataType, IndexType, LogLevel, MetricType, QuantizeType
from .executor import QueryExecutorFactory
from .extension import ReRanker, RrfReRanker, WeightedReRanker
from .model import (
    AddColumnOption,
    AlterColumnOption,
    Collection,
    CollectionSchema,
    CollectionOption,
    Doc,
    FieldSchema,
    FlatIndexParam,
    HnswIndexParam,
    HnswHvqIndexParam,
    HnswSqIndexParam,
    HnswSqQueryParam,
    HnswHvqQueryParam,
    HnswQueryParam,
    InvertIndexParam,
    IvfUsqIndexParam,
    IvfUsqQueryParam,
    IVFIndexParam,
    IVFQueryParam,
    OptimizeOption,
    QueryContext,
    QueryGroupBy,
    QueryOrderBy,
    VectorQuery,
    VectorSchema,
)
from .model.collection import create_and_open, open

_native_exports = [name for name in dir(_native_module) if not name.startswith("_")]
_facade_exports = [
    "AddColumnOption",
    "AlterColumnOption",
    "Collection",
    "CollectionSchema",
    "CollectionOption",
    "DataType",
    "IndexType",
    "Doc",
    "FieldSchema",
    "FlatIndexParam",
    "HnswIndexParam",
    "HnswHvqIndexParam",
    "HnswSqIndexParam",
    "HnswSqQueryParam",
    "HnswHvqQueryParam",
    "HnswQueryParam",
    "IndexOption",
    "InvertIndexParam",
    "IvfUsqIndexParam",
    "IvfUsqQueryParam",
    "LogLevel",
    "QueryContext",
    "QueryGroupBy",
    "QueryOrderBy",
    "QueryExecutorFactory",
    "OptimizeOption",
    "MetricType",
    "ReRanker",
    "RrfReRanker",
    "WeightedReRanker",
    "IVFIndexParam",
    "IVFQueryParam",
    "QuantizeType",
    "VectorQuery",
    "VectorSchema",
]


def _normalize_log_level(log_level):
    if not isinstance(log_level, str):
        raise TypeError("log_level must be a string or a hannsdb.typing.LogLevel")
    return log_level.strip().lower()


def init(log_level="warn"):
    return _native_module.init(_normalize_log_level(log_level))

__all__ = sorted(set(_native_exports + _facade_exports))
