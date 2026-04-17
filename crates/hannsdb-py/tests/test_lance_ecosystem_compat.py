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


def _array_schema():
    return hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[
            hannsdb.FieldSchema(name="tags", data_type="string", array=True),
            hannsdb.FieldSchema(name="scores", data_type="int64", array=True),
            hannsdb.FieldSchema(name="i32s", data_type="int32", array=True),
            hannsdb.FieldSchema(name="u32s", data_type="uint32", array=True),
            hannsdb.FieldSchema(name="u64s", data_type="uint64", array=True),
            hannsdb.FieldSchema(name="f32s", data_type="float", array=True),
            hannsdb.FieldSchema(name="f64s", data_type="float64", array=True),
            hannsdb.FieldSchema(name="flags", data_type="bool", array=True),
        ],
        vectors=[
            hannsdb.VectorSchema(
                name="dense",
                data_type="vector_fp32",
                dimension=2,
            )
        ],
    )


def _nullable_schema():
    return hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[
            hannsdb.FieldSchema(name="title", data_type="string", nullable=True),
            hannsdb.FieldSchema(
                name="tags", data_type="string", nullable=True, array=True
            ),
        ],
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


def test_hannsdb_lance_array_scalars_are_readable_by_external_lance_python(tmp_path):
    lance = pytest.importorskip("lance")

    collection = hannsdb.create_lance_collection(
        str(tmp_path),
        _array_schema(),
        [
            hannsdb.Doc(
                id="10",
                fields={
                    "tags": ["red", "blue"],
                    "scores": [1, 2],
                    "i32s": [3, 4],
                    "u32s": [5, 6],
                    "u64s": [7, 8],
                    "f32s": [1.5, 2.5],
                    "f64s": [3.5, 4.5],
                    "flags": [True, False],
                },
                vectors={"dense": [1.0, 0.0]},
            ),
            hannsdb.Doc(
                id="20",
                fields={
                    "tags": ["green"],
                    "scores": [9],
                    "i32s": [10],
                    "u32s": [11],
                    "u64s": [12],
                    "f32s": [5.5],
                    "f64s": [6.5],
                    "flags": [False],
                },
                vectors={"dense": [0.0, 1.0]},
            ),
        ],
    )

    dataset = lance.dataset(collection.uri)
    table = dataset.to_table()

    assert table.column("tags").to_pylist() == [["red", "blue"], ["green"]]
    assert table.column("scores").to_pylist() == [[1, 2], [9]]
    assert table.column("i32s").to_pylist() == [[3, 4], [10]]
    assert table.column("u32s").to_pylist() == [[5, 6], [11]]
    assert table.column("u64s").to_pylist() == [[7, 8], [12]]
    assert table.column("f32s").to_pylist() == [[1.5, 2.5], [5.5]]
    assert table.column("f64s").to_pylist() == [[3.5, 4.5], [6.5]]
    assert table.column("flags").to_pylist() == [[True, False], [False]]


def test_hannsdb_lance_nullable_scalars_are_written_as_lance_nulls(tmp_path):
    lance = pytest.importorskip("lance")

    collection = hannsdb.create_lance_collection(
        str(tmp_path),
        _nullable_schema(),
        [
            hannsdb.Doc(
                id="10",
                fields={"title": "alpha", "tags": ["red", "blue"]},
                vectors={"dense": [1.0, 0.0]},
            ),
            hannsdb.Doc(id="20", fields={}, vectors={"dense": [0.0, 1.0]}),
        ],
    )

    table = lance.dataset(collection.uri).to_table()
    assert table.column("title").to_pylist() == ["alpha", None]
    assert table.column("tags").to_pylist() == [["red", "blue"], None]

    reopened = hannsdb.open(str(tmp_path), storage="lance")
    first, second = reopened.fetch(["10", "20"])
    assert first.field("title") == "alpha"
    assert first.field("tags") == ["red", "blue"]
    assert second.has_field("title") is False
    assert second.has_field("tags") is False


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


def test_external_lance_dataset_supported_scalars_roundtrip_through_hannsdb(tmp_path):
    lance = pytest.importorskip("lance")
    pa = pytest.importorskip("pyarrow")

    values = pa.array([1.0, 0.0, 0.0, 1.0], type=pa.float32())
    table = pa.table(
        {
            "id": pa.array([10, 20], type=pa.int64()),
            "title": pa.array(["alpha", "beta"], type=pa.string()),
            "i32": pa.array([1, 2], type=pa.int32()),
            "u32": pa.array([3, 4], type=pa.uint32()),
            "u64": pa.array([5, 6], type=pa.uint64()),
            "f32": pa.array([1.5, 2.5], type=pa.float32()),
            "f64": pa.array([3.5, 4.5], type=pa.float64()),
            "flag": pa.array([True, False], type=pa.bool_()),
            "dense": pa.FixedSizeListArray.from_arrays(values, 2),
        }
    )
    uri = tmp_path / "collections" / "docs.lance"
    uri.parent.mkdir()
    lance.write_dataset(table, uri)

    collection = hannsdb.open(str(tmp_path), storage="lance")

    assert [(field.name, field.data_type) for field in collection.schema.fields] == [
        ("title", "string"),
        ("i32", "int32"),
        ("u32", "uint32"),
        ("u64", "uint64"),
        ("f32", "float"),
        ("f64", "float64"),
        ("flag", "bool"),
    ]
    doc = collection.fetch(["10"])[0]
    assert doc.field("title") == "alpha"
    assert doc.field("i32") == 1
    assert doc.field("u32") == 3
    assert doc.field("u64") == 5
    assert doc.field("f32") == 1.5
    assert doc.field("f64") == 3.5
    assert doc.field("flag") is True


def test_external_lance_dataset_nullable_scalars_are_omitted_from_docs(tmp_path):
    lance = pytest.importorskip("lance")
    pa = pytest.importorskip("pyarrow")

    values = pa.array([1.0, 0.0, 0.0, 1.0], type=pa.float32())
    table = pa.table(
        {
            "id": pa.array([10, 20], type=pa.int64()),
            "title": pa.array(["alpha", None], type=pa.string()),
            "dense": pa.FixedSizeListArray.from_arrays(values, 2),
        }
    )
    uri = tmp_path / "collections" / "docs.lance"
    uri.parent.mkdir()
    lance.write_dataset(table, uri)

    collection = hannsdb.open(str(tmp_path), storage="lance")

    assert collection.schema.field("title").nullable is True
    alpha, missing = collection.fetch(["10", "20"])
    assert alpha.field("title") == "alpha"
    assert missing.has_field("title") is False
    with pytest.raises(KeyError, match="title"):
        missing.field("title")


def test_external_lance_dataset_null_vectors_are_rejected_by_hannsdb(tmp_path):
    lance = pytest.importorskip("lance")
    pa = pytest.importorskip("pyarrow")

    table = pa.table(
        {
            "id": pa.array([10, 20], type=pa.int64()),
            "title": pa.array(["alpha", "beta"], type=pa.string()),
            "dense": pa.array([[1.0, 0.0], None], type=pa.list_(pa.float32(), 2)),
        }
    )
    uri = tmp_path / "collections" / "docs.lance"
    uri.parent.mkdir()
    lance.write_dataset(table, uri)

    collection = hannsdb.open(str(tmp_path), storage="lance")

    with pytest.raises(RuntimeError, match="null Lance vector values"):
        collection.fetch(["20"])


def test_external_lance_dataset_list_scalars_roundtrip_through_hannsdb(tmp_path):
    lance = pytest.importorskip("lance")
    pa = pytest.importorskip("pyarrow")

    values = pa.array([1.0, 0.0, 0.0, 1.0], type=pa.float32())
    table = pa.table(
        {
            "id": pa.array([10, 20], type=pa.int64()),
            "tags": pa.array([["red", "blue"], ["green"]], type=pa.list_(pa.string())),
            "scores": pa.array([[1, 2], [3]], type=pa.list_(pa.int64())),
            "dense": pa.FixedSizeListArray.from_arrays(values, 2),
        }
    )
    uri = tmp_path / "collections" / "docs.lance"
    uri.parent.mkdir()
    lance.write_dataset(table, uri)

    collection = hannsdb.open(str(tmp_path), storage="lance")

    assert [(field.name, field.data_type, field.array) for field in collection.schema.fields] == [
        ("tags", "string", True),
        ("scores", "int64", True),
    ]
    doc = collection.fetch(["10"])[0]
    assert doc.field("tags") == ["red", "blue"]
    assert doc.field("scores") == [1, 2]
