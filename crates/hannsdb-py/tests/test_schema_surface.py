import hannsdb


def test_collection_schema_accepts_multi_vector_ivf_metadata():
    ivf_cls = getattr(hannsdb, "IVFIndexParam")

    dense = hannsdb.VectorSchema(
        name="dense",
        data_type="vector_fp32",
        dimension=384,
        index_param=ivf_cls(metric_type="l2", nlist=1024),
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

    assert [vector.name for vector in schema.vectors] == ["dense", "sparse"]
