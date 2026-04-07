import hannsdb


def test_query_context_and_executor_factory_surface():
    query_context_cls = getattr(hannsdb, "QueryContext")

    query = hannsdb.VectorQuery(
        field_name="dense",
        vector=[0.0, 0.1],
        param=None,
    )
    context = query_context_cls(queries=[query])

    factory_cls = getattr(hannsdb, "QueryExecutorFactory")
    executor = factory_cls.create(context)

    assert executor is not None
