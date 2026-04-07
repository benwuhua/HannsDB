import hannsdb


def test_query_context_accepts_queries_shape():
    query = hannsdb.VectorQuery(
        field_name="dense",
        vector=[0.0, 0.1],
        param=None,
    )

    context = hannsdb.QueryContext(queries=[query])

    assert len(context.queries) == 1
    assert context.queries[0].field_name == "dense"


def test_query_executor_factory_exposes_create_method():
    schema = hannsdb.CollectionSchema(
        name="docs",
        fields=[],
        vectors=[
            hannsdb.VectorSchema(
                name="dense",
                data_type="vector_fp32",
                dimension=2,
            )
        ],
    )

    factory_cls = getattr(hannsdb, "QueryExecutorFactory")
    factory = factory_cls.create(schema)

    assert factory is not None
