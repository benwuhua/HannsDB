import threading
import time
import math

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


def build_secondary_vector_collection(tmp_path):
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
            ),
            hannsdb.VectorSchema(
                name="title",
                data_type="vector_fp32",
                dimension=2,
            ),
        ],
    )
    collection = hannsdb.create_and_open(str(tmp_path), schema)
    collection.insert(
        [
            hannsdb.Doc(
                id="1",
                vectors={"dense": [5.0, 5.0], "title": [0.0, 0.0]},
                fields={"group": 1},
            ),
            hannsdb.Doc(
                id="2",
                vectors={"dense": [0.0, 0.0], "title": [0.2, 0.0]},
                fields={"group": 1},
            ),
            hannsdb.Doc(
                id="3",
                vectors={"dense": [1.0, 1.0], "title": [1.0, 0.0]},
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


class InstrumentedCollection:
    def __init__(self, collection, delays=None, barrier=None, fail_key=None):
        self.collection = collection
        self.delays = delays or {}
        self.barrier = barrier
        self.fail_key = fail_key
        self.active = 0
        self.max_active = 0
        self.seen_keys = []
        self._lock = threading.Lock()

    def query_context(self, context):
        query = context.queries[0]
        key = tuple(query.vector)

        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.seen_keys.append(key)

        try:
            if self.barrier is not None:
                self.barrier.wait(timeout=1)

            delay = self.delays.get(key, 0.0)
            if delay:
                time.sleep(delay)

            if self.fail_key is not None and self._matches_key(key, self.fail_key):
                raise RuntimeError(f"forced failure for {key}")

            return self.collection.query_context(context)
        finally:
            with self._lock:
                self.active -= 1

    @staticmethod
    def _matches_key(left, right):
        if len(left) != len(right):
            return False
        return all(math.isclose(lhs, rhs, rel_tol=0.0, abs_tol=1e-8) for lhs, rhs in zip(left, right))


def test_query_context_accepts_queries_shape():
    query = hannsdb.VectorQuery(
        field_name="dense",
        vector=[0.0, 0.1],
        param=None,
    )

    context = hannsdb.QueryContext(queries=[query])

    assert context.queries


def test_query_context_accepts_legacy_native_vector_query(tmp_path):
    collection, schema = build_collection(tmp_path)
    executor = hannsdb.QueryExecutorFactory.create(schema).build()

    legacy_query = hannsdb._native.VectorQuery(
        field_name="dense",
        vector=[0.0, 0.0],
        param=None,
    )
    context = hannsdb.QueryContext(queries=[legacy_query])

    hits = executor.execute(collection, context)

    assert [hit.id for hit in hits] == ["1", "2", "3"]


def test_query_context_accepts_pure_hnsw_query_param(tmp_path):
    collection, schema = build_collection(tmp_path)
    executor = hannsdb.QueryExecutorFactory.create(schema).build()

    context = hannsdb.QueryContext(
        queries=[
            hannsdb.VectorQuery(
                field_name="dense",
                vector=[0.0, 0.0],
                param=hannsdb.HnswQueryParam(ef=64, is_using_refiner=False),
            )
        ]
    )

    hits = executor.execute(collection, context)

    assert [hit.id for hit in hits] == ["1", "2", "3"]


def test_query_context_accepts_query_like_object_without_param_attribute(tmp_path):
    collection, schema = build_collection(tmp_path)
    executor = hannsdb.QueryExecutorFactory.create(schema).build()

    context = hannsdb.QueryContext(
        queries=[
            type(
                "QueryLike",
                (),
                {
                    "field_name": "dense",
                    "vector": [0.0, 0.0],
                },
            )()
        ]
    )

    hits = executor.execute(collection, context)

    assert [hit.id for hit in hits] == ["1", "2", "3"]


def test_query_context_normalizes_scalar_output_fields_and_query_by_id():
    context = hannsdb.QueryContext(
        output_fields="group",
        query_by_id=2,
    )

    assert context.output_fields == ["group"]
    assert context.query_by_id == [2]


def test_query_context_accepts_query_by_id_field_name():
    context = hannsdb.QueryContext(
        query_by_id=["2"],
        query_by_id_field_name="title",
    )

    assert context.query_by_id == ["2"]
    assert context.query_by_id_field_name == "title"


def test_reranker_constructors_validate_arguments():
    with pytest.raises(ValueError, match="topn"):
        hannsdb.RrfReRanker(topn=-1)
    with pytest.raises(ValueError, match="rank_constant"):
        hannsdb.RrfReRanker(rank_constant=-1)
    with pytest.raises(TypeError, match="topn"):
        hannsdb.RrfReRanker(topn="2")
    with pytest.raises(TypeError, match="rank_constant"):
        hannsdb.RrfReRanker(rank_constant="60")
    with pytest.raises(ValueError, match="topn"):
        hannsdb.WeightedReRanker(topn=-1)
    with pytest.raises(ValueError, match="metric"):
        hannsdb.WeightedReRanker(metric="euclidean")
    with pytest.raises(TypeError, match="weights"):
        hannsdb.WeightedReRanker(weights=["dense"])
    with pytest.raises(TypeError, match="weights"):
        hannsdb.WeightedReRanker(weights={"dense": "1.0"})
    with pytest.raises(TypeError, match="weights"):
        hannsdb.WeightedReRanker(weights={1: 1.0})


def test_weighted_reranker_combines_normalized_scores_by_field_weight():
    reranker = hannsdb.WeightedReRanker(
        topn=3,
        metric=hannsdb.MetricType.L2,
        weights={"dense": 2.0},
    )

    query_results = {
        "dense": [
            hannsdb.Doc(id="a", score=0.0, fields={"source": "dense"}),
            hannsdb.Doc(id="b", score=1.0, fields={"source": "dense"}),
        ],
        "title": [
            hannsdb.Doc(id="a", score=0.5, fields={"source": "title"}),
            hannsdb.Doc(id="c", score=0.1, fields={"source": "title"}),
        ],
    }

    hits = reranker.rerank(query_results)

    def normalize_l2(score):
        return 1.0 - 2.0 * math.atan(score) / math.pi

    assert [hit.id for hit in hits] == ["a", "b", "c"]
    assert hits[0] is not query_results["dense"][0]
    assert hits[0].score == pytest.approx(2.0 * normalize_l2(0.0) + normalize_l2(0.5))
    assert hits[1].score == pytest.approx(2.0 * normalize_l2(1.0))
    assert hits[2].score == pytest.approx(normalize_l2(0.1))


def test_weighted_reranker_orders_ip_scores_with_distance_convention():
    reranker = hannsdb.WeightedReRanker(metric=hannsdb.MetricType.Ip)

    query_results = {
        "dense": [
            hannsdb.Doc(id="good", score=-10.0),
            hannsdb.Doc(id="bad", score=0.0),
        ]
    }

    hits = reranker.rerank(query_results)

    assert [hit.id for hit in hits] == ["good", "bad"]
    assert hits[0].score > hits[1].score


def test_weighted_reranker_accepts_string_metric_name():
    reranker = hannsdb.WeightedReRanker(metric="cosine")

    assert reranker.metric is hannsdb.MetricType.Cosine


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


def test_query_executor_supports_secondary_query_by_id_field_name(tmp_path):
    collection, schema = build_secondary_vector_collection(tmp_path)
    executor = hannsdb.QueryExecutorFactory.create(schema).build()

    context = hannsdb.QueryContext(
        top_k=3,
        queries=[
            hannsdb.VectorQuery(field_name="dense", vector=[0.0, 0.0], param=None),
        ],
        query_by_id=["1"],
        query_by_id_field_name="title",
        output_fields=["group"],
    )

    hits = executor.execute(collection, context)

    assert [hit.id for hit in hits] == ["1", "2", "3"]
    assert [hit.fields["group"] for hit in hits] == [1, 1, 2]
    assert hits[0].score == pytest.approx(0.0, abs=1e-6)


def test_query_executor_supports_secondary_vector_field_in_query_context(tmp_path):
    collection, schema = build_secondary_vector_collection(tmp_path)
    executor = hannsdb.QueryExecutorFactory.create(schema).build()

    context = hannsdb.QueryContext(
        top_k=3,
        queries=[
            hannsdb.VectorQuery(field_name="title", vector=[0.0, 0.0], param=None),
        ],
        output_fields=["group"],
    )

    hits = executor.execute(collection, context)

    assert [hit.id for hit in hits] == ["1", "2", "3"]
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


def test_query_executor_supports_builtin_rrf_reranker(tmp_path, monkeypatch):
    collection, schema = build_collection(tmp_path)
    proxy = InstrumentedCollection(
        collection,
        delays={
            (0.0, 0.0): 0.05,
            (0.2, 0.0): 0.0,
        },
    )
    executor = hannsdb.QueryExecutorFactory.create(schema).build()

    monkeypatch.delenv("ZVEC_QUERY_CONCURRENCY", raising=False)
    monkeypatch.delenv("HANNSDB_QUERY_CONCURRENCY", raising=False)

    context = hannsdb.QueryContext(
        top_k=2,
        queries=[
            hannsdb.VectorQuery(field_name="dense", vector=[0.0, 0.0], param=None),
            hannsdb.VectorQuery(field_name="dense", vector=[0.2, 0.0], param=None),
        ],
        reranker=hannsdb.RrfReRanker(topn=3),
        output_fields=["group"],
    )

    hits = executor.execute(proxy, context)

    # `top_k` is the per-query fan-out depth on the reranker path; `topn`
    # controls the final fused result size.
    assert [hit.id for hit in hits] == ["2", "1", "3"]
    assert [hit.fields["group"] for hit in hits] == [1, 1, 2]
    assert proxy.max_active == 1


def test_query_executor_supports_weighted_reranker(tmp_path):
    collection, schema = build_secondary_vector_collection(tmp_path)
    executor = hannsdb.QueryExecutorFactory.create(schema).build()

    context = hannsdb.QueryContext(
        top_k=3,
        queries=[
            hannsdb.VectorQuery(field_name="dense", vector=[0.0, 0.0], param=None),
            hannsdb.VectorQuery(field_name="title", vector=[0.0, 0.0], param=None),
        ],
        reranker=hannsdb.WeightedReRanker(
            topn=3,
            metric=hannsdb.MetricType.L2,
            weights={"dense": 2.0},
        ),
        output_fields=["group"],
    )

    hits = executor.execute(collection, context)

    assert [hit.id for hit in hits] == ["2", "3", "1"]
    assert [hit.fields["group"] for hit in hits] == [1, 2, 1]


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


def test_query_executor_reranker_fanout_can_run_concurrently_when_enabled(
    tmp_path, monkeypatch
):
    collection, schema = build_collection(tmp_path)
    proxy = InstrumentedCollection(
        collection,
        delays={
            (0.0, 0.0): 0.05,
            (0.2, 0.0): 0.0,
        },
    )
    executor = hannsdb.QueryExecutorFactory.create(schema).build()
    reranker = RecordingReranker()

    monkeypatch.setenv("ZVEC_QUERY_CONCURRENCY", "2")

    context = hannsdb.QueryContext(
        top_k=2,
        queries=[
            hannsdb.VectorQuery(field_name="dense", vector=[0.0, 0.0], param=None),
            hannsdb.VectorQuery(field_name="dense", vector=[0.2, 0.0], param=None),
        ],
        reranker=reranker,
    )

    hits = executor.execute(proxy, context)

    assert proxy.max_active >= 2
    assert reranker.seen_keys == [("dense", "dense#2")]
    assert [hit.id for hit in hits] == ["1", "2"]


def test_query_executor_propagates_errors_from_concurrent_fanout(
    tmp_path, monkeypatch
):
    collection, schema = build_collection(tmp_path)
    proxy = InstrumentedCollection(
        collection,
        fail_key=(0.2, 0.0),
    )
    executor = hannsdb.QueryExecutorFactory.create(schema).build()

    monkeypatch.setenv("ZVEC_QUERY_CONCURRENCY", "2")

    context = hannsdb.QueryContext(
        top_k=2,
        queries=[
            hannsdb.VectorQuery(field_name="dense", vector=[0.0, 0.0], param=None),
            hannsdb.VectorQuery(field_name="dense", vector=[0.2, 0.0], param=None),
        ],
        reranker=RecordingReranker(),
    )

    with pytest.raises(RuntimeError, match="forced failure"):
        executor.execute(proxy, context)


def test_query_executor_surfaces_fast_failure_before_slow_neighbor(
    tmp_path, monkeypatch
):
    collection, schema = build_collection(tmp_path)
    proxy = InstrumentedCollection(
        collection,
        delays={
            (0.0, 0.0): 0.25,
        },
        fail_key=(0.2, 0.0),
    )
    executor = hannsdb.QueryExecutorFactory.create(schema).build()

    monkeypatch.setenv("ZVEC_QUERY_CONCURRENCY", "2")

    context = hannsdb.QueryContext(
        top_k=2,
        queries=[
            hannsdb.VectorQuery(field_name="dense", vector=[0.0, 0.0], param=None),
            hannsdb.VectorQuery(field_name="dense", vector=[0.2, 0.0], param=None),
        ],
        reranker=RecordingReranker(),
    )

    start = time.monotonic()
    with pytest.raises(RuntimeError, match="forced failure"):
        executor.execute(proxy, context)
    elapsed = time.monotonic() - start

    assert elapsed < 0.15


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


def test_query_executor_supports_include_vector(tmp_path):
    collection, schema = build_collection(tmp_path)
    executor = hannsdb.QueryExecutorFactory.create(schema).build()

    context = hannsdb.QueryContext(
        top_k=1,
        queries=[
            hannsdb.VectorQuery(field_name="dense", vector=[0.0, 0.0], param=None),
        ],
        include_vector=True,
    )

    hits = executor.execute(collection, context)

    assert [hit.id for hit in hits] == ["1"]
    assert hits[0].vectors["dense"] == [0.0, 0.0]


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
