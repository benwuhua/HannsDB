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


class RecordingReranker(hannsdb.ReRanker):
    def __init__(self):
        super().__init__(topn=2)
        self.seen_keys = []

    def rerank(self, query_results):
        self.seen_keys.append(tuple(query_results.keys()))
        first_key = next(iter(query_results))
        return list(query_results[first_key])[: self.topn]


def test_query_context_accepts_queries_shape():
    query = hannsdb.VectorQuery(
        field_name="dense",
        vector=[0.0, 0.1],
        param=None,
    )

    context = hannsdb.QueryContext(queries=[query])

    assert context.queries


def test_query_context_normalizes_scalar_output_fields_and_query_by_id():
    context = hannsdb.QueryContext(
        output_fields="group",
        query_by_id=2,
    )

    assert context.output_fields == ["group"]
    assert context.query_by_id == [2]


def test_reranker_constructors_validate_arguments():
    with pytest.raises(ValueError, match="topn"):
        hannsdb.RrfReRanker(topn=-1)
    with pytest.raises(ValueError, match="rank_constant"):
        hannsdb.RrfReRanker(rank_constant=-1)
    with pytest.raises(TypeError, match="topn"):
        hannsdb.RrfReRanker(topn="2")
    with pytest.raises(TypeError, match="rank_constant"):
        hannsdb.RrfReRanker(rank_constant="60")


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


def test_query_executor_supports_builtin_rrf_reranker(tmp_path):
    collection, schema = build_collection(tmp_path)
    executor = hannsdb.QueryExecutorFactory.create(schema).build()

    context = hannsdb.QueryContext(
        top_k=2,
        queries=[
            hannsdb.VectorQuery(field_name="dense", vector=[0.0, 0.0], param=None),
            hannsdb.VectorQuery(field_name="dense", vector=[0.2, 0.0], param=None),
        ],
        reranker=hannsdb.RrfReRanker(topn=3),
        output_fields=["group"],
    )

    hits = executor.execute(collection, context)

    # `top_k` is the per-query fan-out depth on the reranker path; `topn`
    # controls the final fused result size.
    assert [hit.id for hit in hits] == ["2", "1", "3"]
    assert [hit.fields["group"] for hit in hits] == [1, 1, 2]


def test_query_executor_invokes_custom_reranker_with_stable_duplicate_field_labels(tmp_path):
    collection, schema = build_collection(tmp_path)
    executor = hannsdb.QueryExecutorFactory.create(schema).build()
    reranker = RecordingReranker()

    context = hannsdb.QueryContext(
        top_k=2,
        queries=[
            hannsdb.VectorQuery(field_name="dense", vector=[0.0, 0.0], param=None),
            hannsdb.VectorQuery(field_name="dense", vector=[0.2, 0.0], param=None),
        ],
        reranker=reranker,
    )

    hits = executor.execute(collection, context)

    assert reranker.seen_keys == [("dense", "dense#2")]
    assert [hit.id for hit in hits] == ["1", "2"]


def test_query_executor_rejects_core_unsupported_query_shape_as_not_implemented(tmp_path):
    collection, schema = build_collection(tmp_path)
    executor = hannsdb.QueryExecutorFactory.create(schema).build()

    context = hannsdb.QueryContext(
        top_k=2,
        queries=[
            hannsdb.VectorQuery(
                field_name="dense",
                vector=[0.0, 0.0],
                param=hannsdb.HnswQueryParam(ef=64, is_using_refiner=False),
            ),
        ],
        query_by_id=2,
    )

    with pytest.raises(NotImplementedError, match="unsupported"):
        executor.execute(collection, context)


def test_query_executor_rejects_include_vector_as_not_implemented(tmp_path):
    collection, schema = build_collection(tmp_path)
    executor = hannsdb.QueryExecutorFactory.create(schema).build()

    context = hannsdb.QueryContext(
        top_k=1,
        queries=[
            hannsdb.VectorQuery(field_name="dense", vector=[0.0, 0.0], param=None),
        ],
        include_vector=True,
    )

    with pytest.raises(NotImplementedError, match="include_vector"):
        executor.execute(collection, context)


def test_query_executor_rejects_query_by_id_with_reranker_as_not_implemented(tmp_path):
    collection, schema = build_collection(tmp_path)
    executor = hannsdb.QueryExecutorFactory.create(schema).build()

    context = hannsdb.QueryContext(
        top_k=2,
        queries=[
            hannsdb.VectorQuery(field_name="dense", vector=[0.0, 0.0], param=None),
        ],
        query_by_id=[1],
        reranker=hannsdb.RrfReRanker(topn=2),
    )

    with pytest.raises(NotImplementedError, match="query_by_id"):
        executor.execute(collection, context)
