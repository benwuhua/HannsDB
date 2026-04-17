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
