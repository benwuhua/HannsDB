import hannsdb


def test_query_context_surface_is_missing():
    getattr(hannsdb, "QueryContext")


def test_query_executor_factory_surface_is_missing():
    getattr(hannsdb, "QueryExecutorFactory")
