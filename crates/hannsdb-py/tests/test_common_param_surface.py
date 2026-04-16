import dataclasses

import hannsdb
import pytest


def test_common_param_surface_reexports_flat_and_ivf_query_wrappers():
    for public_cls, model_cls in (
        (hannsdb.FlatIndexParam, hannsdb.model.param.FlatIndexParam),
        (hannsdb.InvertIndexParam, hannsdb.model.param.InvertIndexParam),
        (hannsdb.IVFQueryParam, hannsdb.model.param.IVFQueryParam),
    ):
        assert public_cls is model_cls
        assert public_cls.__module__ == "hannsdb.model.param.index_params"
        assert dataclasses.is_dataclass(public_cls)


def test_flat_index_param_validates_metric_type_and_bridges_to_native():
    param = hannsdb.FlatIndexParam(metric_type=hannsdb.MetricType.L2)

    assert param.metric_type == "l2"
    assert param._get_native().__class__ is hannsdb._native.FlatIndexParam

    with pytest.raises(ValueError, match="unsupported metric_type"):
        hannsdb.FlatIndexParam(metric_type="bogus")


def test_ivf_query_param_validates_nprobe_and_bridges_to_native():
    param = hannsdb.IVFQueryParam(nprobe=8)

    assert param.nprobe == 8
    assert param._get_native().__class__ is hannsdb._native.IVFQueryParam

    with pytest.raises(TypeError, match="nprobe"):
        hannsdb.IVFQueryParam(nprobe="8")


def test_vector_schema_accepts_flat_index_param():
    schema = hannsdb.VectorSchema(
        name="dense",
        data_type=hannsdb.DataType.VectorFp32,
        dimension=2,
        index_param=hannsdb.FlatIndexParam(metric_type=hannsdb.MetricType.Cosine),
    )

    assert schema.index_param.metric_type == "cosine"
    assert schema._get_native().__class__ is hannsdb._native.VectorSchema


def test_vector_query_accepts_ivf_query_param():
    query = hannsdb.VectorQuery(
        field_name="dense",
        vector=[0.1, 0.2],
        param=hannsdb.IVFQueryParam(nprobe=4),
    )

    assert query.param.nprobe == 4


def test_vector_schema_rejects_unsupported_index_param_type():
    with pytest.raises(TypeError, match="index_param"):
        hannsdb.VectorSchema(
            name="dense",
            data_type=hannsdb.DataType.VectorFp32,
            dimension=2,
            index_param=object(),
        )


def test_vector_query_rejects_unsupported_query_param_type():
    with pytest.raises(TypeError, match="param"):
        hannsdb.VectorQuery(
            field_name="dense",
            vector=[0.1, 0.2],
            param=object(),
        )


def test_invert_index_param_validates_default_only_shape_and_bridges_to_native():
    param = hannsdb.InvertIndexParam()

    assert param.enable_range_optimization is False
    assert param.enable_extended_wildcard is False
    assert param._get_native().__class__ is hannsdb._native.InvertIndexParam


def test_invert_index_param_accepts_functional_flags():
    # enable_range_optimization=True should be accepted (no ValueError)
    param_ro = hannsdb.InvertIndexParam(enable_range_optimization=True)
    assert param_ro.enable_range_optimization is True
    assert param_ro.enable_extended_wildcard is False
    assert param_ro._get_native().__class__ is hannsdb._native.InvertIndexParam

    # enable_extended_wildcard=True should be accepted (no ValueError)
    param_ew = hannsdb.InvertIndexParam(enable_extended_wildcard=True)
    assert param_ew.enable_range_optimization is False
    assert param_ew.enable_extended_wildcard is True
    assert param_ew._get_native().__class__ is hannsdb._native.InvertIndexParam

    # Both flags True should be accepted
    param_both = hannsdb.InvertIndexParam(
        enable_range_optimization=True, enable_extended_wildcard=True
    )
    assert param_both.enable_range_optimization is True
    assert param_both.enable_extended_wildcard is True
    assert param_both._get_native().__class__ is hannsdb._native.InvertIndexParam

    # Native layer also accepts both flags
    native_ro = hannsdb._native.InvertIndexParam(enable_range_optimization=True)
    assert native_ro.enable_range_optimization is True

    native_ew = hannsdb._native.InvertIndexParam(enable_extended_wildcard=True)
    assert native_ew.enable_extended_wildcard is True


def test_advanced_param_families_remain_absent_from_public_surface():
    assert hasattr(hannsdb, "HnswRabitqIndexParam") is False
    assert hasattr(hannsdb, "HnswRabitqQueryParam") is False
    assert hasattr(hannsdb.DataType, "VectorFp16") is False


def test_hnsw_sq_query_param_defaults_and_validation():
    param = hannsdb.HnswSqQueryParam()
    assert param.ef_search == 50
    assert param._get_native().__class__ is hannsdb._native.HnswSqQueryParam


def test_hnsw_sq_query_param_custom_ef_search():
    param = hannsdb.HnswSqQueryParam(ef_search=100)
    assert param.ef_search == 100


def test_hnsw_sq_query_param_rejects_invalid():
    import pytest
    with pytest.raises((ValueError, TypeError)):
        hannsdb.HnswSqQueryParam(ef_search="bad")


def test_index_option_defaults_and_validation():
    opt = hannsdb.IndexOption()
    assert opt.concurrency == 0


def test_index_option_custom_concurrency():
    opt = hannsdb.IndexOption(concurrency=4)
    assert opt.concurrency == 4


def test_hnsw_sq_query_param_and_index_option_are_public():
    assert hannsdb.HnswSqQueryParam is hannsdb.model.param.HnswSqQueryParam
    assert hannsdb.IndexOption is not None


def test_hnsw_hvq_query_param_defaults_and_validation():
    param = hannsdb.HnswHvqQueryParam()
    assert param.ef_search == 50
    assert param._get_native().__class__ is hannsdb._native.HnswHvqQueryParam


def test_hnsw_hvq_query_param_custom_ef_search():
    param = hannsdb.HnswHvqQueryParam(ef_search=200)
    assert param.ef_search == 200


def test_hnsw_hvq_query_param_rejects_invalid():
    import pytest
    with pytest.raises((ValueError, TypeError)):
        hannsdb.HnswHvqQueryParam(ef_search="bad")


def test_hnsw_hvq_query_param_is_public():
    assert hannsdb.HnswHvqQueryParam is hannsdb.model.param.HnswHvqQueryParam


# --- IndexType enum ---

def test_index_type_is_exported():
    assert hasattr(hannsdb, "IndexType")
    assert hannsdb.IndexType.HNSW == 1
    assert hannsdb.IndexType.IVF == 3
    assert hannsdb.IndexType.FLAT == 4
    assert hannsdb.IndexType.INVERT == 10


def test_index_param_type_properties():
    assert hannsdb.FlatIndexParam().type == hannsdb.IndexType.FLAT
    assert hannsdb.HnswIndexParam().type == hannsdb.IndexType.HNSW
    assert hannsdb.InvertIndexParam().type == hannsdb.IndexType.INVERT
    assert hannsdb.IVFIndexParam().type == hannsdb.IndexType.IVF
    assert hannsdb.IvfUsqIndexParam().type == hannsdb.IndexType.IVF_USQ
    assert hannsdb.HnswSqIndexParam().type == hannsdb.IndexType.HNSW_SQ
    assert hannsdb.HnswHvqIndexParam().type == hannsdb.IndexType.HNSW_HVQ


# --- DataType uppercase aliases ---

def test_data_type_uppercase_aliases():
    assert hannsdb.DataType.FLOAT is hannsdb.DataType.Float
    assert hannsdb.DataType.INT64 is hannsdb.DataType.Int64
    assert hannsdb.DataType.INT32 is hannsdb.DataType.Int32
    assert hannsdb.DataType.STRING is hannsdb.DataType.String
    assert hannsdb.DataType.BOOL is hannsdb.DataType.Bool
    assert hannsdb.DataType.VECTOR_FP32 is hannsdb.DataType.VectorFp32


def test_data_type_new_variants():
    assert hannsdb.DataType.VECTOR_INT8 is not None
    assert str(hannsdb.DataType.VECTOR_INT8) == "vector_int8"
    assert hannsdb.DataType.SPARSE_VECTOR_FP32 is not None
    assert str(hannsdb.DataType.SPARSE_VECTOR_FP32) == "sparse_vector_fp32"


# --- MetricType uppercase aliases ---

def test_metric_type_uppercase_aliases():
    assert hannsdb.MetricType.COSINE is hannsdb.MetricType.Cosine
    assert hannsdb.MetricType.IP is hannsdb.MetricType.Ip
    assert hannsdb.MetricType.L2 is hannsdb.MetricType.L2  # L2 is already uppercase


# --- FieldSchema.index_param ---

def test_field_schema_accepts_index_param():
    f = hannsdb.FieldSchema("id", data_type="int64", index_param=hannsdb.InvertIndexParam())
    assert f.index_param is not None
    assert f.index_param.type == hannsdb.IndexType.INVERT
    assert f.index_param.enable_range_optimization is False


def test_field_schema_index_param_defaults_to_none():
    f = hannsdb.FieldSchema("name", data_type="string")
    assert f.index_param is None


def test_field_schema_index_param_is_readonly():
    f = hannsdb.FieldSchema("id", data_type="int64", index_param=hannsdb.InvertIndexParam())
    with pytest.raises(AttributeError):
        f.index_param = hannsdb.InvertIndexParam(enable_range_optimization=True)
