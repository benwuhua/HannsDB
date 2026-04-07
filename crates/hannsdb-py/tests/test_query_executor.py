import hannsdb
import pytest


def build_collection(tmp_path):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[
            hannsdb.FieldSchema(
                name="group",
                data_type="int64",
            )
        ],
        vectors=[
            hannsdb.VectorSchema(
                name="dense",
                data_type="vector_fp32",
                dimension=2,
            )
        ],
    )
    collection = hannsdb.create_and_open(str(tmp_path), schema)
    collection.insert(
        [
            hannsdb.Doc(
                id="1",
                vector=[0.0, 0.0],
                fields={"group": 1},
            ),
            hannsdb.Doc(
                id="2",
                vector=[0.1, 0.0],
                fields={"group": 1},
            ),
            hannsdb.Doc(
                id="3",
                vector=[0.2, 0.0],
                fields={"group": 2},
            ),
        ]
    )
    return collection, schema


def test_query_context_accepts_queries_shape():
    query = hannsdb.VectorQuery(
        field_name="dense",
        vector=[0.0, 0.1],
        param=None,
    )

    context = hannsdb.QueryContext(queries=[query])

    assert context.queries


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


def test_query_executor_supports_multi_query_and_query_by_id(tmp_path):
    collection, schema = build_collection(tmp_path)
    executor = hannsdb.QueryExecutorFactory.create(schema).build()

    context = hannsdb.QueryContext(
        top_k=3,
        queries=[
            hannsdb.VectorQuery(field_name="dense", vector=[0.0, 0.0], param=None),
            hannsdb.VectorQuery(field_name="dense", vector=[0.2, 0.0], param=None),
        ],
        query_by_id=["2"],
        output_fields=["group"],
    )

    hits = executor.execute(collection, context)

    assert {hit.id for hit in hits} == {"1", "2", "3"}
    assert [hit.fields["group"] for hit in hits] == [1, 1, 2]


def test_query_executor_supports_group_by(tmp_path):
    collection, schema = build_collection(tmp_path)
    executor = hannsdb.QueryExecutorFactory.create(schema).build()

    context = hannsdb.QueryContext(
        top_k=2,
        queries=[
            hannsdb.VectorQuery(field_name="dense", vector=[0.0, 0.0], param=None),
        ],
        group_by=hannsdb.QueryGroupBy(field_name="group"),
        output_fields=["group"],
    )

    hits = executor.execute(collection, context)

    assert [hit.fields["group"] for hit in hits] == [1, 2]


def test_query_executor_still_rejects_reranker(tmp_path):
    collection, schema = build_collection(tmp_path)
    executor = hannsdb.QueryExecutorFactory.create(schema).build()

    context = hannsdb.QueryContext(
        top_k=2,
        queries=[
            hannsdb.VectorQuery(field_name="dense", vector=[0.0, 0.0], param=None),
        ],
        reranker=hannsdb.QueryReranker(model="dummy"),
    )

    with pytest.raises(NotImplementedError, match="reranker"):
        executor.execute(collection, context)
