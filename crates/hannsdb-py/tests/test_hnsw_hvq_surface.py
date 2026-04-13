import dataclasses

import hannsdb
import pytest


def build_hnsw_hvq_index_param(**overrides):
    params = {
        "metric_type": hannsdb.MetricType.Ip,
        "m": 8,
        "m_max0": 16,
        "ef_construction": 32,
        "ef_search": 32,
        "nbits": 4,
    }
    params.update(overrides)
    return hannsdb.HnswHvqIndexParam(**params)


def test_hnsw_hvq_surface_reexports_public_wrapper():
    assert hannsdb.HnswHvqIndexParam is hannsdb.model.param.HnswHvqIndexParam
    assert hannsdb.HnswHvqIndexParam.__module__ == "hannsdb.model.param.index_params"
    assert dataclasses.is_dataclass(hannsdb.HnswHvqIndexParam)


def test_hnsw_hvq_index_param_validates_and_bridges_to_native():
    param = build_hnsw_hvq_index_param()

    assert param.metric_type == "ip"
    assert param.m == 8
    assert param.m_max0 == 16
    assert param.ef_construction == 32
    assert param.ef_search == 32
    assert param.nbits == 4
    assert param._get_native().__class__ is hannsdb._native.HnswHvqIndexParam

    default_metric = hannsdb.HnswHvqIndexParam(metric_type=None)
    assert default_metric.metric_type == "ip"
    assert default_metric._get_native().__class__ is hannsdb._native.HnswHvqIndexParam

    with pytest.raises(ValueError, match="hnsw_hvq"):
        build_hnsw_hvq_index_param(metric_type=hannsdb.MetricType.Cosine)


def test_hnsw_hvq_create_and_open_optimize_query_and_reopen(tmp_path):
    schema = hannsdb.CollectionSchema(
        name="docs",
        fields=[hannsdb.FieldSchema(name="rank", data_type=hannsdb.DataType.Int32)],
        vectors=[
            hannsdb.VectorSchema(
                name="dense",
                data_type=hannsdb.DataType.VectorFp32,
                dimension=2,
                index_param=build_hnsw_hvq_index_param(),
            )
        ],
        primary_vector="dense",
    )

    collection = hannsdb.create_and_open(str(tmp_path), schema)
    collection.insert(
        [
            hannsdb.Doc(id="doc-1", fields={"rank": 1}, vectors={"dense": [1.0, 0.0]}),
            hannsdb.Doc(id="doc-2", fields={"rank": 2}, vectors={"dense": [0.9, 0.0]}),
            hannsdb.Doc(id="doc-3", fields={"rank": 3}, vectors={"dense": [0.0, 1.0]}),
        ]
    )

    collection.optimize()
    hits = collection.query(
        query_context=hannsdb.QueryContext(
            top_k=2,
            output_fields=["rank"],
            queries=[hannsdb.VectorQuery(field_name="dense", vector=[1.0, 0.0])],
        )
    )

    assert [doc.id for doc in hits] == ["doc-1", "doc-2"]

    reopened = hannsdb.open(str(tmp_path))
    assert reopened.schema.vector("dense").index_param.__class__ is hannsdb.HnswHvqIndexParam
    reopened_hits = reopened.query(
        query_context=hannsdb.QueryContext(
            top_k=2,
            queries=[hannsdb.VectorQuery(field_name="dense", vector=[1.0, 0.0])],
        )
    )
    assert [doc.id for doc in reopened_hits] == ["doc-1", "doc-2"]
