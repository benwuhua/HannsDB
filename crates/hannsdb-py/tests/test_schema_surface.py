import json
from pathlib import Path

import hannsdb
import pytest


def build_schema():
    title = hannsdb.VectorSchema(
        name="title",
        data_type="vector_fp32",
        dimension=384,
        index_param=hannsdb.IVFIndexParam(metric_type="l2", nlist=1024),
    )
    dense = hannsdb.VectorSchema(
        name="dense",
        data_type="vector_fp32",
        dimension=384,
        index_param=hannsdb.HnswIndexParam(
            metric_type="cosine",
            m=32,
            ef_construction=128,
        ),
    )
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
        vectors=[title, dense],
    )


def test_collection_schema_accepts_old_positional_vector_schema():
    title = hannsdb.VectorSchema(
        name="title",
        data_type="vector_fp32",
        dimension=384,
    )

    schema = hannsdb.CollectionSchema("docs", title)

    assert schema.primary_vector == "title"
    assert [vector.name for vector in schema.vectors] == ["title"]


def test_create_and_open_accepts_primary_ivf_index(tmp_path):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="title",
        fields=[],
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

    collection = hannsdb.create_and_open(str(tmp_path), schema)
    metadata = json.loads((tmp_path / "collections" / "docs" / "collection.json").read_text())

    assert collection.collection_name == "docs"
    assert metadata["primary_vector"] == "title"
    assert metadata["vectors"][0]["index_param"] == {
        "kind": "ivf",
        "metric": "l2",
        "nlist": 1024,
    }

    collection.destroy()


def test_collection_schema_vectors_property_surface():
    schema = build_schema()

    assert schema.primary_vector == "dense"
    vectors = schema.vectors
    assert [vector.name for vector in vectors] == ["title", "dense"]
    assert vectors[0].dimension == 384
    assert vectors[1].dimension == 384
    assert schema.fields[1].nullable is True
    assert schema.fields[1].array is True


def test_create_and_open_persists_richer_schema(tmp_path):
    collection = hannsdb.create_and_open(str(tmp_path), build_schema())
    metadata = json.loads(
        (
            tmp_path
            / "collections"
            / "docs"
            / "collection.json"
        ).read_text()
    )

    assert metadata["primary_vector"] == "dense"
    assert [vector["name"] for vector in metadata["vectors"]] == ["title", "dense"]
    assert metadata["fields"][1]["nullable"] is True
    assert metadata["fields"][1]["array"] is True
    assert metadata["vectors"][0]["index_param"] == {
        "kind": "ivf",
        "metric": "l2",
        "nlist": 1024,
    }
    assert metadata["vectors"][1]["index_param"] == {
        "kind": "hnsw",
        "metric": "cosine",
        "m": 32,
        "ef_construction": 128,
    }

    collection.destroy()


def test_collection_index_ddl_surface_persists_and_lists_indexes(tmp_path):
    collection = hannsdb.create_and_open(str(tmp_path), build_schema())

    hnsw_blob = (
        Path(collection.path)
        / "collections"
        / collection.collection_name
        / "hnsw_index.bin"
    )
    hnsw_blob.write_bytes(b"stale graph")

    collection.create_vector_index(
        "title",
        hannsdb.IVFIndexParam(metric_type="l2", nlist=8),
    )
    collection.create_scalar_index("session_id")

    assert collection.list_vector_indexes() == ["title"]
    assert collection.list_scalar_indexes() == ["session_id"]

    indexes_path = (
        Path(collection.path)
        / "collections"
        / collection.collection_name
        / "indexes.json"
    )
    catalog = json.loads(indexes_path.read_text())
    assert catalog["vector_indexes"][0]["field_name"] == "title"
    assert catalog["vector_indexes"][0]["kind"] == "ivf"
    assert catalog["vector_indexes"][0]["metric"] == "l2"
    assert catalog["vector_indexes"][0]["params"] == {"nlist": 8}
    assert catalog["scalar_indexes"][0]["field_name"] == "session_id"
    assert catalog["scalar_indexes"][0]["kind"] == "inverted"
    assert not hnsw_blob.exists()

    collection.drop_vector_index("title")
    collection.drop_scalar_index("session_id")

    assert collection.list_vector_indexes() == []
    assert collection.list_scalar_indexes() == []
    assert not hnsw_blob.exists()

    collection.destroy()


def test_collection_index_ddl_surface_runs_core_validation(tmp_path):
    collection = hannsdb.create_and_open(str(tmp_path), build_schema())

    with pytest.raises(ValueError):
        collection.create_vector_index(
            "missing_vector",
            hannsdb.IVFIndexParam(metric_type="l2", nlist=8),
        )

    collection.destroy()
