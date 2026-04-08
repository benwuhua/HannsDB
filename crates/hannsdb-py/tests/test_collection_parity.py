import dataclasses

import hannsdb


def test_python_facade_reexports_pure_param_wrappers():
    assert hannsdb.CollectionOption is hannsdb.model.param.CollectionOption
    assert hannsdb.OptimizeOption is hannsdb.model.param.OptimizeOption
    assert hannsdb.HnswIndexParam is hannsdb.model.param.HnswIndexParam
    assert hannsdb.IVFIndexParam is hannsdb.model.param.IVFIndexParam
    assert hannsdb.HnswQueryParam is hannsdb.model.param.HnswQueryParam
    assert hannsdb.CollectionOption.__module__ == "hannsdb.model.param.collection_option"
    assert hannsdb.HnswIndexParam.__module__ == "hannsdb.model.param.index_params"
    assert hannsdb.IVFIndexParam.__module__ == "hannsdb.model.param.index_params"
    assert hannsdb.HnswQueryParam.__module__ == "hannsdb.model.param.index_params"
    assert dataclasses.is_dataclass(hannsdb.CollectionOption)
    assert dataclasses.is_dataclass(hannsdb.HnswIndexParam)
    assert dataclasses.is_dataclass(hannsdb.IVFIndexParam)
    assert dataclasses.is_dataclass(hannsdb.HnswQueryParam)


def test_python_facade_reexports_core_schema_and_executor_types():
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        vectors=[
            hannsdb.VectorSchema(
                name="dense",
                data_type="vector_fp32",
                dimension=2,
            )
        ],
    )

    factory = hannsdb.QueryExecutorFactory.create(schema)

    assert hannsdb.CollectionSchema is not None
    assert hannsdb.VectorSchema is not None
    assert hannsdb.QueryContext is not None
    assert factory.schema is schema
    assert factory.build().schema is schema


def test_star_import_exposes_native_and_facade_symbols():
    namespace = {}
    exec("from hannsdb import *", namespace)

    for name in [
        "MetricType",
        "QuantizeType",
        "DataType",
        "LogLevel",
        "CollectionOption",
        "OptimizeOption",
        "HnswIndexParam",
        "IVFIndexParam",
        "HnswQueryParam",
        "create_and_open",
        "ReRanker",
        "QueryExecutorFactory",
        "QueryGroupBy",
        "RrfReRanker",
    ]:
        assert name in namespace
