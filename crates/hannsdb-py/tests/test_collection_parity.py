import hannsdb


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
