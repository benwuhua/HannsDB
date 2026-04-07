import tempfile

import hannsdb


def test_collection_schema_multi_vector_path_is_rejected_by_create_and_open():
    dense = hannsdb.VectorSchema(
        name="dense",
        data_type="vector_fp32",
        dimension=384,
    )
    sparse = hannsdb.VectorSchema(
        name="sparse",
        data_type="vector_fp32",
        dimension=384,
    )
    schema = hannsdb.CollectionSchema(
        name="docs",
        fields=[hannsdb.FieldSchema(name="session_id", data_type="string")],
        vectors=[dense, sparse],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        hannsdb.create_and_open(tmpdir, schema, None)


def test_vector_schema_missing_ivf_index_param_surface():
    ivf_cls = getattr(hannsdb, "IVFIndexParam")

    hannsdb.VectorSchema(
        name="dense",
        data_type="vector_fp32",
        dimension=384,
        index_param=ivf_cls(metric_type="l2", nlist=1024),
    )
