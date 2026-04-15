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
