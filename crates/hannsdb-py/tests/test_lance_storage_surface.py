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


def _docs():
    return [
        hannsdb.Doc(id="10", fields={"title": "alpha"}, vectors={"dense": [1.0, 0.0]}),
        hannsdb.Doc(id="20", fields={"title": "beta"}, vectors={"dense": [0.0, 1.0]}),
    ]


def test_lance_collection_surface_create_insert_fetch_search_delete_upsert(tmp_path):
    collection = hannsdb.create_lance_collection(str(tmp_path), _schema(), _docs()[:1])

    assert collection.name == "docs"
    assert collection.uri.endswith("collections/docs.lance")
    assert collection.insert(_docs()[1:]) == 1

    fetched = collection.fetch(["20", "10"])
    assert [doc.id for doc in fetched] == ["20", "10"]
    assert fetched[0].field("title") == "beta"

    hits = collection.search([1.0, 0.0], topk=1, metric="l2")
    assert [doc.id for doc in hits] == ["10"]
    assert hits[0].score == 0.0

    assert collection.delete(["10"]) == 1
    assert collection.fetch(["10"]) == []

    assert collection.upsert(
        [
            hannsdb.Doc(id="20", fields={"title": "beta-v2"}, vectors={"dense": [2.0, 0.0]}),
            hannsdb.Doc(id="30", fields={"title": "gamma"}, vectors={"dense": [3.0, 0.0]}),
        ]
    ) == 2
    assert [doc.id for doc in collection.fetch(["20", "30"])] == ["20", "30"]

    reopened = hannsdb.open_lance_collection(str(tmp_path), _schema())
    assert [doc.id for doc in reopened.fetch(["20", "30"])] == ["20", "30"]


def test_lance_collection_surface_exposes_hanns_sidecar_optimize(tmp_path):
    collection = hannsdb.create_lance_collection(str(tmp_path), _schema(), _docs())

    path = collection.hanns_index_path("dense")
    assert str(path).endswith("_hannsdb/ann/dense.hanns")

    collection.optimize_hanns("dense", metric="l2")
    assert path.exists()

    hits = collection.search([1.0, 0.0], topk=1, metric="l2")
    assert [doc.id for doc in hits] == ["10"]

    collection.insert(
        [hannsdb.Doc(id="30", fields={"title": "gamma"}, vectors={"dense": [2.0, 0.0]})]
    )
    assert not path.exists()


def test_create_and_open_storage_lance_routes_to_lance_collection(tmp_path):
    collection = hannsdb.create_and_open(str(tmp_path), _schema(), storage="lance")

    assert isinstance(collection, hannsdb.LanceCollection)
    assert collection.name == "docs"
    assert collection.insert(_docs()) == 2
    assert [doc.id for doc in collection.fetch(["10", "20"])] == ["10", "20"]


def test_open_storage_lance_requires_schema_and_reopens_collection(tmp_path):
    created = hannsdb.create_and_open(str(tmp_path), _schema(), storage="lance")
    created.insert(_docs())

    reopened = hannsdb.open(str(tmp_path), storage="lance", schema=_schema())

    assert isinstance(reopened, hannsdb.LanceCollection)
    assert [doc.id for doc in reopened.fetch(["20"])] == ["20"]


def test_open_storage_lance_infers_schema_and_reopens_collection(tmp_path):
    created = hannsdb.create_and_open(str(tmp_path), _schema(), storage="lance")
    created.insert(_docs())

    reopened = hannsdb.open(str(tmp_path), storage="lance")

    assert isinstance(reopened, hannsdb.LanceCollection)
    assert reopened.schema.name == "docs"
    assert reopened.schema.primary_vector == "dense"
    assert [(field.name, field.data_type) for field in reopened.schema.fields] == [
        ("title", "string")
    ]
    assert [
        (vector.name, vector.data_type, vector.dimension)
        for vector in reopened.schema.vectors
    ] == [
        ("dense", "vector_fp32", 2)
    ]
    assert [doc.id for doc in reopened.fetch(["10", "20"])] == ["10", "20"]
