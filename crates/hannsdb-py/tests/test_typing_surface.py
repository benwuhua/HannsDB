import hannsdb


def test_typing_surface_is_available_from_package_and_top_level():
    assert hannsdb.typing.MetricType is hannsdb.MetricType
    assert hannsdb.typing.QuantizeType is hannsdb.QuantizeType
    assert hannsdb.typing.DataType is hannsdb.DataType
    assert hannsdb.typing.LogLevel is hannsdb.LogLevel

    assert hannsdb.MetricType.__module__ == "hannsdb.typing.metric_type"
    assert hannsdb.QuantizeType.__module__ == "hannsdb.typing.quantize_type"
    assert hannsdb.DataType.__module__ == "hannsdb.typing.data_type"
    assert hannsdb.LogLevel.__module__ == "hannsdb.typing.log_level"

    assert str(hannsdb.MetricType.L2) == "l2"
    assert str(hannsdb.QuantizeType.Fp16) == "fp16"
    assert str(hannsdb.DataType.VectorFp32) == "vector_fp32"
    assert str(hannsdb.LogLevel.Warn) == "warn"


def test_typing_enums_are_accepted_by_schema_and_index_param_wrappers():
    schema = hannsdb.CollectionSchema(
        name="docs",
        fields=[
            hannsdb.FieldSchema(name="session_id", data_type=hannsdb.DataType.String),
        ],
        vectors=[
            hannsdb.VectorSchema(
                name="dense",
                data_type=hannsdb.DataType.VectorFp32,
                dimension=2,
                index_param=hannsdb.IVFIndexParam(
                    metric_type=hannsdb.MetricType.L2,
                    nlist=8,
                ),
            )
        ],
    )

    hnsw = hannsdb.HnswIndexParam(
        metric_type=hannsdb.MetricType.Cosine,
        quantize_type=hannsdb.QuantizeType.Fp16,
    )

    assert schema.field("session_id").data_type == "string"
    assert schema.vector("dense").data_type == "vector_fp32"
    assert schema.vector("dense").index_param.metric_type == "l2"
    assert hnsw.metric_type == "cosine"
    assert hnsw.quantize_type == "fp16"


def test_init_accepts_pure_python_log_level():
    hannsdb.init(log_level=hannsdb.LogLevel.Warn)
    hannsdb.init(log_level="warn")
