import json

import hannsdb


def build_schema():
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
    title = hannsdb.VectorSchema(
        name="title",
        data_type="vector_fp32",
        dimension=384,
        index_param=hannsdb.IVFIndexParam(metric_type="l2", nlist=1024),
    )
    return hannsdb.CollectionSchema(
        name="docs",
        fields=[
            hannsdb.FieldSchema(name="session_id", data_type="string"),
            hannsdb.FieldSchema(name="tags", data_type="string"),
        ],
        vectors=[dense, title],
    )


def test_collection_schema_vectors_property_surface():
    schema = build_schema()

    vectors = schema.vectors
    assert [vector.name for vector in vectors] == ["dense", "title"]
    assert vectors[0].dimension == 384
    assert vectors[1].dimension == 384


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

    assert [vector["name"] for vector in metadata["vectors"]] == ["dense", "title"]
    assert metadata["vectors"][0]["index_param"] == {
        "kind": "hnsw",
        "metric": "cosine",
        "m": 32,
        "ef_construction": 128,
    }
    assert metadata["vectors"][1]["index_param"] == {
        "kind": "ivf",
        "metric": "l2",
        "nlist": 1024,
    }

    collection.destroy()
