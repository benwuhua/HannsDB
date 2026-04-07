import hannsdb


def test_query_context_accepts_queries_shape():
    query = hannsdb.VectorQuery(
        field_name="dense",
        vector=[0.0, 0.1],
        param=None,
    )

    hannsdb.QueryContext(queries=[query])


def test_query_executor_factory_exposes_create_method():
    factory_cls = getattr(hannsdb, "QueryExecutorFactory")
    create = getattr(factory_cls, "create")

    try:
        create(None)
    except TypeError:
        pass
