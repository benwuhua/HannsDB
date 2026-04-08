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


def test_native_collection_exposes_option_property(tmp_path):
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
    option = hannsdb._native.CollectionOption(True, False)

    collection = hannsdb._native.create_and_open(
        str(tmp_path),
        schema._get_native(),
        option,
    )

    assert collection.option.__class__ is hannsdb._native.CollectionOption
    assert collection.option.read_only is True
    assert collection.option.enable_mmap is False

    collection.destroy()


def test_native_collection_defaults_option_when_omitted(tmp_path):
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

    collection = hannsdb._native.create_and_open(
        str(tmp_path),
        schema._get_native(),
    )

    assert collection.option.__class__ is hannsdb._native.CollectionOption
    assert collection.option.read_only is False
    assert collection.option.enable_mmap is True

    collection.destroy()


def test_python_facade_exports_weighted_reranker_from_extension_and_top_level():
    assert hannsdb.extension.WeightedReRanker is hannsdb.WeightedReRanker
    from hannsdb.extension import WeightedReRanker as ExtensionWeightedReRanker

    assert ExtensionWeightedReRanker is hannsdb.WeightedReRanker


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
        "WeightedReRanker",
    ]:
        assert name in namespace
