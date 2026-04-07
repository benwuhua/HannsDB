import hannsdb


def build_schema():
    return hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[
            hannsdb.FieldSchema(name="session_id", data_type="string"),
            hannsdb.FieldSchema(
                name="tags",
                data_type="string",
                nullable=True,
                array=True,
            ),
        ],
        vectors=[
            hannsdb.VectorSchema(
                name="title",
                data_type="vector_fp32",
                dimension=384,
                index_param=hannsdb.IVFIndexParam(metric_type="l2", nlist=1024),
            ),
            hannsdb.VectorSchema(
                name="dense",
                data_type="vector_fp32",
                dimension=384,
                index_param=hannsdb.HnswIndexParam(
                    metric_type="cosine",
                    m=32,
                    ef_construction=128,
                ),
            ),
        ],
    )


def test_create_and_open_returns_python_facade_and_keeps_schema(tmp_path):
    schema = build_schema()

    collection = hannsdb.create_and_open(str(tmp_path), schema)

    assert isinstance(collection, hannsdb.Collection)
    assert collection.schema is schema
    assert collection.collection_name == "docs"
    assert collection.path == str(tmp_path)
    assert [vector.name for vector in collection.schema.vectors] == ["title", "dense"]

    collection.destroy()


def test_open_recovers_schema_from_collection_metadata(tmp_path):
    created = hannsdb.create_and_open(str(tmp_path), build_schema())
    reopened = hannsdb.open(str(tmp_path))

    assert isinstance(reopened, hannsdb.Collection)
    assert reopened.schema.name == "docs"
    assert reopened.schema.primary_vector == "dense"
    assert [field.name for field in reopened.schema.fields] == ["session_id", "tags"]
    assert reopened.schema.fields[1].nullable is True
    assert reopened.schema.fields[1].array is True
    assert [vector.name for vector in reopened.schema.vectors] == ["title", "dense"]
    assert [vector.dimension for vector in reopened.schema.vectors] == [384, 384]

    created.destroy()


def test_collection_query_routes_through_executor_and_query_context_delegate(monkeypatch):
    schema = build_schema()
    calls = []

    class FakeCore:
        path = "/tmp/hannsdb"
        collection_name = "docs"

        def query_context(self, context):
            calls.append(("query_context", context))
            return ["native-result"]

    class FakeExecutor:
        def execute(self, collection, context):
            calls.append(("execute", collection, context))
            return collection.query_context(context)

    class FakeFactory:
        def build(self):
            calls.append(("build", None, None))
            return FakeExecutor()

    monkeypatch.setattr(hannsdb.QueryExecutorFactory, "create", lambda schema: FakeFactory())

    collection = hannsdb.Collection._from_core(FakeCore(), schema=schema)
    context = hannsdb.QueryContext(
        top_k=1,
        queries=[
            hannsdb.VectorQuery(field_name="dense", vector=[0.0, 0.0], param=None)
        ],
    )

    result = collection.query(context)

    assert result == ["native-result"]
    assert [call[0] for call in calls] == ["build", "execute", "query_context"]


def test_collection_surface_methods_delegate_to_core_handle(monkeypatch):
    schema = build_schema()
    calls = []

    class FakeCore:
        path = "/tmp/hannsdb"
        collection_name = "docs"
        stats = {"live_count": 3}

        def query_context(self, context):
            calls.append(("query_context", context))
            return ["native-result"]

        def insert(self, docs):
            calls.append(("insert", docs))
            return 1

        def upsert(self, docs):
            calls.append(("upsert", docs))
            return 2

        def fetch(self, ids):
            calls.append(("fetch", ids))
            return ["fetch-result"]

        def delete(self, ids):
            calls.append(("delete", ids))
            return 3

        def flush(self):
            calls.append(("flush", None))

        def destroy(self):
            calls.append(("destroy", None))

        def create_vector_index(self, field_name, index_param=None):
            calls.append(("create_vector_index", field_name, index_param))

        def drop_vector_index(self, field_name):
            calls.append(("drop_vector_index", field_name))

        def list_vector_indexes(self):
            calls.append(("list_vector_indexes", None))
            return ["dense"]

        def create_scalar_index(self, field_name):
            calls.append(("create_scalar_index", field_name))

        def drop_scalar_index(self, field_name):
            calls.append(("drop_scalar_index", field_name))

        def list_scalar_indexes(self):
            calls.append(("list_scalar_indexes", None))
            return ["session_id"]

    class FakeFactory:
        def build(self):
            return object()

    monkeypatch.setattr(hannsdb.QueryExecutorFactory, "create", lambda schema: FakeFactory())
    monkeypatch.setattr(
        hannsdb.Collection,
        "_refresh_schema",
        lambda self: calls.append(("refresh_schema", None)),
    )

    collection = hannsdb.Collection._from_core(FakeCore(), schema=schema)

    assert collection.path == "/tmp/hannsdb"
    assert collection.collection_name == "docs"
    assert collection.stats == {"live_count": 3}
    assert collection.insert([]) == 1
    assert collection.upsert([]) == 2
    assert collection.fetch(["1"]) == ["fetch-result"]
    assert collection.delete(["1"]) == 3
    assert collection.list_vector_indexes() == ["dense"]
    assert collection.list_scalar_indexes() == ["session_id"]
    collection.flush()
    collection.create_vector_index("dense", hannsdb.IVFIndexParam(metric_type="l2", nlist=8))
    collection.drop_vector_index("dense")
    collection.create_scalar_index("session_id")
    collection.drop_scalar_index("session_id")
    collection.destroy()

    assert [call[0] for call in calls] == [
        "insert",
        "upsert",
        "fetch",
        "delete",
        "list_vector_indexes",
        "list_scalar_indexes",
        "flush",
        "create_vector_index",
        "refresh_schema",
        "drop_vector_index",
        "refresh_schema",
        "create_scalar_index",
        "refresh_schema",
        "drop_scalar_index",
        "refresh_schema",
        "destroy",
    ]
