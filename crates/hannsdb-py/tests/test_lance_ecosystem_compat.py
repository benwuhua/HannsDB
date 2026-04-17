import pytest

import hannsdb


def _schema():
    return hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[hannsdb.FieldSchema(name="title", data_type="string")],
        vectors=[
            hannsdb.VectorSchema(
                name="dense",
                data_type="vector_fp32",
                dimension=2,
            )
        ],
    )


def test_hannsdb_lance_collection_is_readable_by_external_lance_python(tmp_path):
    lance = pytest.importorskip("lance")

    collection = hannsdb.create_lance_collection(
        str(tmp_path),
        _schema(),
        [
            hannsdb.Doc(id="10", fields={"title": "alpha"}, vectors={"dense": [1.0, 0.0]}),
            hannsdb.Doc(id="20", fields={"title": "beta"}, vectors={"dense": [0.0, 1.0]}),
        ],
    )

    dataset = lance.dataset(collection.uri)
    assert dataset.count_rows() == 2

    table = dataset.to_table()
    assert table.num_rows == 2
    assert set(table.column_names) >= {"id", "title", "dense"}
    assert table.column("title").to_pylist() == ["alpha", "beta"]


def test_hannsdb_storage_lance_selector_is_readable_by_external_lance_python(tmp_path):
    lance = pytest.importorskip("lance")

    collection = hannsdb.create_and_open(str(tmp_path), _schema(), storage="lance")
    collection.insert(
        [
            hannsdb.Doc(id="10", fields={"title": "alpha"}, vectors={"dense": [1.0, 0.0]}),
            hannsdb.Doc(id="20", fields={"title": "beta"}, vectors={"dense": [0.0, 1.0]}),
        ]
    )

    dataset = lance.dataset(collection.uri)
    assert dataset.count_rows() == 2

    table = dataset.to_table()
    assert table.column("title").to_pylist() == ["alpha", "beta"]


def test_hannsdb_native_lance_selector_named_dataset_is_readable_by_external_lance_python(
    tmp_path,
):
    lance = pytest.importorskip("lance")

    collection = hannsdb._native.create_and_open(
        str(tmp_path),
        _schema()._get_native(),
        storage="lance",
        name="docs",
    )
    collection.insert(
        [
            hannsdb._native.Doc(
                id="10",
                field_name="dense",
                fields={"title": "alpha"},
                vectors={"dense": [1.0, 0.0]},
            ),
            hannsdb._native.Doc(
                id="20",
                field_name="dense",
                fields={"title": "beta"},
                vectors={"dense": [0.0, 1.0]},
            ),
        ]
    )

    reopened = hannsdb._native.open(str(tmp_path), storage="lance")
    assert reopened.name == "docs"

    dataset = lance.dataset(reopened.uri)
    assert dataset.count_rows() == 2

    table = dataset.to_table()
    assert set(table.column_names) >= {"id", "title", "dense"}
    assert table.column("title").to_pylist() == ["alpha", "beta"]


def test_external_lance_dataset_is_readable_by_hannsdb_lance_selector(tmp_path):
    lance = pytest.importorskip("lance")
    pa = pytest.importorskip("pyarrow")

    values = pa.array([1.0, 0.0, 0.0, 1.0], type=pa.float32())
    table = pa.table(
        {
            "id": pa.array([10, 20], type=pa.int64()),
            "title": pa.array(["alpha", "beta"], type=pa.string()),
            "dense": pa.FixedSizeListArray.from_arrays(values, 2),
        }
    )
    uri = tmp_path / "collections" / "docs.lance"
    uri.parent.mkdir()
    lance.write_dataset(table, uri)

    collection = hannsdb.open(str(tmp_path), storage="lance")

    assert collection.name == "docs"
    assert collection.schema.name == "docs"
    assert [(field.name, field.data_type) for field in collection.schema.fields] == [
        ("title", "string")
    ]
    assert [
        (vector.name, vector.data_type, vector.dimension)
        for vector in collection.schema.vectors
    ] == [
        ("dense", "vector_fp32", 2)
    ]
    assert [doc.id for doc in collection.fetch(["10", "20"])] == ["10", "20"]
    assert [doc.field("title") for doc in collection.fetch(["10", "20"])] == [
        "alpha",
        "beta",
    ]

    native = hannsdb._native.open(str(tmp_path), storage="lance")
    assert native.name == "docs"
    assert [doc.id for doc in native.fetch(["20"])] == ["20"]
