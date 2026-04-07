import json

import hannsdb


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
