import dataclasses

import hannsdb
import pytest


def build_ivf_usq_index_param(**overrides):
    params = {
        "metric_type": hannsdb.MetricType.L2,
        "nlist": 64,
        "bits_per_dim": 4,
        "rotation_seed": 42,
        "rerank_k": 64,
        "use_high_accuracy_scan": False,
    }
    params.update(overrides)
    return hannsdb.IvfUsqIndexParam(**params)


def test_ivf_usq_surface_reexports_public_wrappers():
    assert hannsdb.IvfUsqIndexParam is hannsdb.model.param.IvfUsqIndexParam
    assert hannsdb.IvfUsqQueryParam is hannsdb.model.param.IvfUsqQueryParam
    assert hannsdb.IvfUsqIndexParam.__module__ == "hannsdb.model.param.index_params"
    assert hannsdb.IvfUsqQueryParam.__module__ == "hannsdb.model.param.index_params"
    assert dataclasses.is_dataclass(hannsdb.IvfUsqIndexParam)
    assert dataclasses.is_dataclass(hannsdb.IvfUsqQueryParam)


def test_ivf_usq_index_param_validates_and_bridges_to_native():
    param = build_ivf_usq_index_param()

    assert param.metric_type == "l2"
    assert param.nlist == 64
    assert param.bits_per_dim == 4
    assert param.rotation_seed == 42
    assert param.rerank_k == 64
    assert param.use_high_accuracy_scan is False
    assert param._get_native().__class__ is hannsdb._native.IvfUsqIndexParam

    with pytest.raises(ValueError, match="unsupported metric_type"):
        build_ivf_usq_index_param(metric_type="bogus")
    with pytest.raises(TypeError, match="nlist"):
        build_ivf_usq_index_param(nlist="64")
    with pytest.raises(TypeError, match="bits_per_dim"):
        build_ivf_usq_index_param(bits_per_dim="4")
    with pytest.raises(TypeError, match="rotation_seed"):
        build_ivf_usq_index_param(rotation_seed="42")
    with pytest.raises(TypeError, match="rerank_k"):
        build_ivf_usq_index_param(rerank_k="64")
    with pytest.raises(TypeError, match="use_high_accuracy_scan"):
        build_ivf_usq_index_param(use_high_accuracy_scan="false")


def test_ivf_usq_query_param_validates_and_bridges_to_native():
    param = hannsdb.IvfUsqQueryParam(nprobe=8)

    assert param.nprobe == 8
    assert param._get_native().__class__ is hannsdb._native.IvfUsqQueryParam

    with pytest.raises(TypeError, match="nprobe"):
        hannsdb.IvfUsqQueryParam(nprobe="8")


def test_vector_schema_accepts_ivf_usq_index_param():
    schema = hannsdb.VectorSchema(
        name="dense",
        data_type=hannsdb.DataType.VectorFp32,
        dimension=2,
        index_param=build_ivf_usq_index_param(),
    )

    assert schema.index_param.nlist == 64
    assert schema.index_param.bits_per_dim == 4
    assert schema._get_native().__class__ is hannsdb._native.VectorSchema


def test_vector_query_accepts_ivf_usq_query_param():
    query = hannsdb.VectorQuery(
        field_name="dense",
        vector=[0.1, 0.2],
        param=hannsdb.IvfUsqQueryParam(nprobe=4),
    )

    assert query.param.nprobe == 4


def test_ivf_usq_create_and_open_optimize_query_and_reopen(tmp_path):
    schema = hannsdb.CollectionSchema(
        name="docs",
        fields=[
            hannsdb.FieldSchema(name="title", data_type=hannsdb.DataType.String),
            hannsdb.FieldSchema(name="rank", data_type=hannsdb.DataType.Int32),
        ],
        vectors=[
            hannsdb.VectorSchema(
                name="dense",
                data_type=hannsdb.DataType.VectorFp32,
                dimension=2,
                index_param=build_ivf_usq_index_param(nlist=1),
            )
        ],
        primary_vector="dense",
    )

    collection = hannsdb.create_and_open(str(tmp_path), schema)
    collection.insert(
        [
            hannsdb.Doc(
                id="doc-1",
                fields={"title": "a", "rank": 1},
                vectors={"dense": [0.0, 0.0]},
            ),
            hannsdb.Doc(
                id="doc-2",
                fields={"title": "b", "rank": 2},
                vectors={"dense": [0.1, 0.0]},
            ),
            hannsdb.Doc(
                id="doc-3",
                fields={"title": "c", "rank": 3},
                vectors={"dense": [5.0, 5.0]},
            ),
        ]
    )

    collection.optimize()
    hits = collection.query(
        query_context=hannsdb.QueryContext(
            top_k=2,
            output_fields=["title", "rank"],
            queries=[
                hannsdb.VectorQuery(
                    field_name="dense",
                    vector=[0.0, 0.0],
                    param=hannsdb.IvfUsqQueryParam(nprobe=4),
                )
            ],
        )
    )

    assert [doc.id for doc in hits] == ["doc-1", "doc-2"]
    assert hits[0].fields["rank"] == 1

    reopened = hannsdb.open(str(tmp_path))
    assert reopened.schema.vector("dense").index_param.__class__ is hannsdb.IvfUsqIndexParam
    assert reopened.schema.vector("dense").index_param.nlist == 1

    reopened_hits = reopened.query(
        query_context=hannsdb.QueryContext(
            top_k=2,
            queries=[
                hannsdb.VectorQuery(
                    field_name="dense",
                    vector=[0.0, 0.0],
                    param=hannsdb.IvfUsqQueryParam(nprobe=4),
                )
            ],
        )
    )
    assert [doc.id for doc in reopened_hits] == ["doc-1", "doc-2"]
