import hannsdb


def test_collection_schema_vectors_property_surface_is_missing():
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

    vectors = getattr(schema, "vectors")
    assert len(vectors) == 2


def test_vector_schema_missing_ivf_index_param_surface():
    ivf_cls = getattr(hannsdb, "IVFIndexParam")

    hannsdb.VectorSchema(
        name="dense",
        data_type="vector_fp32",
        dimension=384,
        index_param=ivf_cls(metric_type="l2", nlist=1024),
    )
