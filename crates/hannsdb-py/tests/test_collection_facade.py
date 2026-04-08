import hannsdb
import pytest


def test_doc_is_pure_python_facade_type():
    assert hannsdb.Doc is not hannsdb._native.Doc
    assert hannsdb.Doc.__module__ == "hannsdb.model.doc"


def test_doc_constructor_supports_legacy_and_zvec_shapes():
    legacy = hannsdb.Doc(
        id="1",
        vector=[0.1, 0.2],
        field_name="dense",
        fields={"session_id": "abc"},
        score=0.5,
    )
    zvec = hannsdb.Doc(
        id="2",
        score=0.25,
        vectors={"dense": [0.3, 0.4]},
        fields={"session_id": "def"},
    )

    assert legacy.vector("dense") == [0.1, 0.2]
    assert legacy.field("session_id") == "abc"
    assert legacy.has_vector("dense") is True
    assert legacy.has_field("session_id") is True
    assert zvec.vector("dense") == [0.3, 0.4]
    assert zvec.field("session_id") == "def"
    assert zvec.has_vector("dense") is True
    assert zvec.has_field("session_id") is True
    assert legacy._get_native().__class__ is hannsdb._native.Doc
    assert zvec._get_native().__class__ is hannsdb._native.Doc


def test_doc_normalizes_numpy_vectors_and_replace():
    np = pytest.importorskip("numpy")

    doc = hannsdb.Doc(
        id="1",
        vectors={"dense": np.array([1.0, 2.0], dtype=np.float32)},
        fields={"session_id": "abc"},
        score=0.1,
    )
    replaced = doc._replace(score=0.2)

    assert doc.vector("dense") == [1.0, 2.0]
    assert isinstance(doc.vector("dense"), list)
    assert replaced.score == 0.2
    assert replaced.vector("dense") == [1.0, 2.0]


@pytest.mark.parametrize(
    "bad_vector",
    [
        "abc",
        b"abc",
        bytearray(b"abc"),
        {"x": 1},
        {1, 2},
        frozenset({1, 2}),
        1,
    ],
)
def test_doc_rejects_malformed_vectors_early(bad_vector):
    with pytest.raises(TypeError, match="vector must be"):
        hannsdb.Doc(id="1", vector=bad_vector, field_name="dense")

    with pytest.raises(TypeError, match="vector must be"):
        hannsdb.Doc(id="1", vectors={"dense": bad_vector})


def test_doc_rejects_none_inside_vector():
    with pytest.raises(TypeError, match="vector must be"):
        hannsdb.Doc(id="1", vector=[1.0, None], field_name="dense")

    with pytest.raises(TypeError, match="vector must be"):
        hannsdb.Doc(id="1", vectors={"dense": [1.0, None]})

    with pytest.raises(TypeError, match="vector must be"):
        hannsdb.Doc(id="1", vectors={"dense": None})


def test_wrap_doc_preserves_legacy_doc_like_vector_and_field_name():
    wrapped = hannsdb.model.collection._wrap_doc(
        type(
            "LegacyDocLike",
            (),
            {
                "id": "1",
                "score": 0.1,
                "fields": {"session_id": "abc"},
                "vector": [0.1, 0.2],
                "field_name": "dense",
            },
        )()
    )

    assert wrapped.field_name == "dense"
    assert wrapped.vector("dense") == [0.1, 0.2]
    assert wrapped.field("session_id") == "abc"


def test_wrap_doc_preserves_field_name_for_empty_vectors_mapping():
    wrapped = hannsdb.model.collection._wrap_doc(
        type(
            "LegacyDocLike",
            (),
            {
                "id": "1",
                "score": 0.1,
                "fields": {"session_id": "abc"},
                "vectors": {},
                "field_name": "z_primary",
            },
        )()
    )

    assert wrapped.field_name == "z_primary"
    assert wrapped.vectors == {}
    assert wrapped.field("session_id") == "abc"


def test_real_collection_supports_multi_vector_doc_writes_and_round_trip(tmp_path):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[hannsdb.FieldSchema(name="session_id", data_type="string")],
        vectors=[
            hannsdb.VectorSchema(
                name="dense",
                data_type="vector_fp32",
                dimension=2,
            ),
            hannsdb.VectorSchema(
                name="sparse",
                data_type="vector_fp32",
                dimension=2,
            ),
        ],
    )
    collection = hannsdb.create_and_open(str(tmp_path), schema)
    doc = hannsdb.Doc(
        id="1",
        score=0.1,
        vectors={"dense": [1.0, 2.0], "sparse": [3.0, 4.0]},
        fields={"session_id": "abc"},
    )

    assert doc._get_native().vectors == {
        "dense": [1.0, 2.0],
        "sparse": [3.0, 4.0],
    }

    assert collection.insert([doc]) == 1
    upserted = doc._replace(
        score=0.2,
        vectors={
            "dense": [2.0, 3.0],
            "sparse": [4.0, 5.0],
        },
    )
    assert collection.upsert([upserted]) == 1

    fetched = collection.fetch(["1"])
    assert fetched[0].vector("dense") == [2.0, 3.0]
    assert fetched[0].vector("sparse") == [4.0, 5.0]
    assert fetched[0].field("session_id") == "abc"

    reopened = hannsdb.open(str(tmp_path))
    replayed = reopened.fetch(["1"])
    assert replayed[0].vector("dense") == [2.0, 3.0]
    assert replayed[0].vector("sparse") == [4.0, 5.0]
    reopened.destroy()


def test_real_collection_insert_and_upsert_accept_single_vector_docs(tmp_path):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[hannsdb.FieldSchema(name="session_id", data_type="string")],
        vectors=[
            hannsdb.VectorSchema(
                name="dense",
                data_type="vector_fp32",
                dimension=2,
            )
        ],
    )
    collection = hannsdb.create_and_open(str(tmp_path), schema)
    doc = hannsdb.Doc(
        id="1",
        vector=[1.0, 2.0],
        field_name="dense",
        fields={"session_id": "abc"},
        score=0.1,
    )

    assert collection.insert([doc]) == 1
    assert collection.upsert([doc]) == 1
    fetched = collection.fetch(["1"])

    assert fetched[0].vector("dense") == [1.0, 2.0]
    assert fetched[0].field("session_id") == "abc"

    collection.destroy()


def test_real_collection_insert_and_upsert_accept_single_doc_input(tmp_path):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[hannsdb.FieldSchema(name="session_id", data_type="string")],
        vectors=[
            hannsdb.VectorSchema(
                name="dense",
                data_type="vector_fp32",
                dimension=2,
            )
        ],
    )
    collection = hannsdb.create_and_open(str(tmp_path), schema)
    doc = hannsdb.Doc(
        id="1",
        vector=[1.0, 2.0],
        field_name="dense",
        fields={"session_id": "abc"},
        score=0.1,
    )
    updated = doc._replace(
        score=0.2,
        fields={"session_id": "xyz"},
        vector=[3.0, 4.0],
    )

    assert collection.insert(doc) == 1
    assert collection.upsert(updated) == 1

    fetched = collection.fetch(["1"])
    assert fetched[0].vector("dense") == [3.0, 4.0]
    assert fetched[0].field("session_id") == "xyz"

    collection.destroy()


def test_real_collection_fetch_preserves_primary_field_name_for_legacy_replace(tmp_path):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="z_primary",
        fields=[hannsdb.FieldSchema(name="session_id", data_type="string")],
        vectors=[
            hannsdb.VectorSchema(
                name="a_secondary",
                data_type="vector_fp32",
                dimension=2,
            ),
            hannsdb.VectorSchema(
                name="z_primary",
                data_type="vector_fp32",
                dimension=2,
            ),
        ],
    )
    collection = hannsdb.create_and_open(str(tmp_path), schema)
    original = hannsdb.Doc(
        id="1",
        fields={"session_id": "abc"},
        vectors={
            "z_primary": [1.0, 2.0],
            "a_secondary": [3.0, 4.0],
        },
    )
    assert collection.insert([original]) == 1

    fetched = collection.fetch(["1"])[0]
    assert fetched.field_name == "z_primary"

    updated = fetched._replace(vector=[9.0, 8.0])
    assert collection.upsert([updated]) == 1
    replayed = collection.fetch(["1"])[0]
    assert replayed.vector("z_primary") == [9.0, 8.0]
    assert replayed.vector("a_secondary") == [3.0, 4.0]

    collection.destroy()


def test_real_collection_single_id_fetch_and_delete_parity(tmp_path):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[hannsdb.FieldSchema(name="session_id", data_type="string")],
        vectors=[
            hannsdb.VectorSchema(
                name="dense",
                data_type="vector_fp32",
                dimension=2,
            )
        ],
    )
    collection = hannsdb.create_and_open(str(tmp_path), schema)
    first = hannsdb.Doc(
        id="1",
        vectors={"dense": [1.0, 2.0]},
        fields={"session_id": "abc"},
    )
    second = hannsdb.Doc(
        id="2",
        vectors={"dense": [3.0, 4.0]},
        fields={"session_id": "def"},
    )

    assert collection.insert([first, second]) == 2

    fetched = collection.fetch("1")
    assert fetched is not None
    assert fetched.id == "1"
    assert fetched.vector("dense") == [1.0, 2.0]
    assert fetched.field("session_id") == "abc"

    assert collection.fetch("3") is None

    assert collection.fetch(["1", "2"]) == [first, second]
    assert collection.delete("1") == 1
    assert collection.fetch("1") is None
    assert collection.fetch(["2"]) == [second]
    assert collection.delete(["2"]) == 1
    assert collection.fetch(["2"]) == []

    collection.destroy()


def build_update_parity_collection(tmp_path):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[
            hannsdb.FieldSchema(name="session_id", data_type="string"),
            hannsdb.FieldSchema(name="tag", data_type="string"),
        ],
        vectors=[
            hannsdb.VectorSchema(
                name="dense",
                data_type="vector_fp32",
                dimension=2,
            ),
            hannsdb.VectorSchema(
                name="sparse",
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
                fields={"session_id": "abc", "tag": "draft"},
                vectors={
                    "dense": [1.0, 2.0],
                    "sparse": [3.0, 4.0],
                },
            )
        ]
    )
    return collection


def test_real_collection_update_partial_scalar_preserves_vectors_and_other_fields(
    tmp_path,
):
    collection = build_update_parity_collection(tmp_path)

    assert collection.update([hannsdb.Doc(id="1", fields={"tag": "published"})]) == 1

    fetched = collection.fetch(["1"])[0]
    assert fetched.field("session_id") == "abc"
    assert fetched.field("tag") == "published"
    assert fetched.vector("dense") == [1.0, 2.0]
    assert fetched.vector("sparse") == [3.0, 4.0]

    collection.destroy()


def test_real_collection_update_partial_vector_preserves_other_vectors_and_fields(
    tmp_path,
):
    collection = build_update_parity_collection(tmp_path)

    assert collection.update(
        [hannsdb.Doc(id="1", vectors={"dense": [9.0, 8.0]})]
    ) == 1

    fetched = collection.fetch(["1"])[0]
    assert fetched.field("session_id") == "abc"
    assert fetched.field("tag") == "draft"
    assert fetched.vector("dense") == [9.0, 8.0]
    assert fetched.vector("sparse") == [3.0, 4.0]

    collection.destroy()


def test_real_collection_update_accepts_single_doc_input(tmp_path):
    collection = build_update_parity_collection(tmp_path)

    assert collection.update(hannsdb.Doc(id="1", fields={"tag": "published"})) == 1

    fetched = collection.fetch(["1"])[0]
    assert fetched.field("tag") == "published"
    assert fetched.field("session_id") == "abc"
    assert fetched.vector("dense") == [1.0, 2.0]
    assert fetched.vector("sparse") == [3.0, 4.0]

    collection.destroy()


def test_real_collection_update_raises_key_error_for_missing_id(tmp_path):
    collection = build_update_parity_collection(tmp_path)

    with pytest.raises(KeyError, match="999"):
        collection.update([hannsdb.Doc(id="999", fields={"tag": "published"})])

    collection.destroy()


def test_real_collection_update_coalesces_duplicate_ids_in_patch_order(tmp_path):
    collection = build_update_parity_collection(tmp_path)

    assert collection.update(
        [
            hannsdb.Doc(id="1", fields={"tag": "reviewed"}),
            hannsdb.Doc(id="1", vectors={"dense": [9.0, 8.0]}),
            hannsdb.Doc(id="1", fields={"session_id": "xyz"}),
        ]
    ) == 1

    fetched = collection.fetch(["1"])[0]
    assert fetched.field("session_id") == "xyz"
    assert fetched.field("tag") == "reviewed"
    assert fetched.vector("dense") == [9.0, 8.0]
    assert fetched.vector("sparse") == [3.0, 4.0]

    collection.destroy()


def test_real_collection_update_mixed_batch_raises_before_partial_write(tmp_path):
    collection = build_update_parity_collection(tmp_path)

    with pytest.raises(KeyError, match="999"):
        collection.update(
            [
                hannsdb.Doc(id="1", fields={"tag": "published"}),
                hannsdb.Doc(id="999", fields={"tag": "missing"}),
            ]
        )

    fetched = collection.fetch(["1"])[0]
    assert fetched.field("session_id") == "abc"
    assert fetched.field("tag") == "draft"
    assert fetched.vector("dense") == [1.0, 2.0]
    assert fetched.vector("sparse") == [3.0, 4.0]

    collection.destroy()


def test_real_collection_update_preserves_primary_field_name_for_secondary_only_patch(
    tmp_path,
):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="z_primary",
        fields=[hannsdb.FieldSchema(name="session_id", data_type="string")],
        vectors=[
            hannsdb.VectorSchema(
                name="a_secondary",
                data_type="vector_fp32",
                dimension=2,
            ),
            hannsdb.VectorSchema(
                name="z_primary",
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
                fields={"session_id": "abc"},
                vectors={
                    "z_primary": [1.0, 2.0],
                    "a_secondary": [3.0, 4.0],
                },
            )
        ]
    )

    assert collection.update([hannsdb.Doc(id="1", vectors={"a_secondary": [7.0, 6.0]})]) == 1

    fetched = collection.fetch(["1"])[0]
    assert fetched.field_name == "z_primary"
    assert fetched.vector("z_primary") == [1.0, 2.0]
    assert fetched.vector("a_secondary") == [7.0, 6.0]

    collection.destroy()


def build_multi_vector_collection(tmp_path):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[hannsdb.FieldSchema(name="group", data_type="int64")],
        vectors=[
            hannsdb.VectorSchema(
                name="dense",
                data_type="vector_fp32",
                dimension=2,
            ),
            hannsdb.VectorSchema(
                name="sparse",
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
                vectors={
                    "dense": [0.0, 0.0],
                    "sparse": [1.0, 1.0],
                },
                fields={"group": 1},
            ),
            hannsdb.Doc(
                id="2",
                vectors={
                    "dense": [0.1, 0.0],
                    "sparse": [2.0, 2.0],
                },
                fields={"group": 2},
            ),
        ]
    )
    return collection


def build_secondary_vector_collection(tmp_path):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[hannsdb.FieldSchema(name="group", data_type="int64")],
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
                vectors={
                    "dense": [5.0, 5.0],
                    "title": [0.0, 0.0],
                },
                fields={"group": 1},
            ),
            hannsdb.Doc(
                id="2",
                vectors={
                    "dense": [0.0, 0.0],
                    "title": [0.2, 0.0],
                },
                fields={"group": 1},
            ),
            hannsdb.Doc(
                id="3",
                vectors={
                    "dense": [1.0, 1.0],
                    "title": [1.0, 0.0],
                },
                fields={"group": 2},
            ),
        ]
    )
    return collection


def test_real_collection_query_matches_manual_ground_truth_for_filtered_typed_surface(
    tmp_path,
):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[
            hannsdb.FieldSchema(name="group", data_type="int64"),
            hannsdb.FieldSchema(name="color", data_type="string"),
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
    docs = [
        hannsdb.Doc(
            id="11",
            vector=[0.0, 0.0],
            field_name="dense",
            fields={"group": 1},
            score=0.0,
        ),
        hannsdb.Doc(
            id="12",
            vector=[1.0, 0.0],
            field_name="dense",
            fields={"group": 1},
            score=0.0,
        ),
        hannsdb.Doc(
            id="13",
            vector=[0.0, 0.5],
            field_name="dense",
            fields={"group": 2},
            score=0.0,
        ),
        hannsdb.Doc(
            id="14",
            vector=[1.0, 1.0],
            field_name="dense",
            fields={"group": 1},
            score=0.0,
        ),
        hannsdb.Doc(
            id="15",
            vector=[2.0, 0.0],
            field_name="dense",
            fields={"group": 1},
            score=0.0,
        ),
        hannsdb.Doc(
            id="16",
            vector=[10.0, 10.0],
            field_name="dense",
            fields={"group": 2},
            score=0.0,
        ),
    ]
    query_vector = [0.0, 0.0]
    assert collection.insert(docs) == len(docs)

    result = collection.query(
        vectors=hannsdb.VectorQuery(field_name="dense", vector=query_vector, param=None),
        output_fields=["group"],
        topk=3,
        filter="group == 1",
    )

    expected = [
        ("11", _l2_distance(query_vector, [0.0, 0.0])),
        ("12", _l2_distance(query_vector, [1.0, 0.0])),
        ("14", _l2_distance(query_vector, [1.0, 1.0])),
    ]
    expected.sort(key=lambda item: (item[1], item[0]))

    assert [doc.id for doc in result] == [doc_id for doc_id, _ in expected]
    assert all(doc.field("group") == 1 for doc in result)
    assert [doc.score for doc in result] == pytest.approx(
        [distance for _, distance in expected],
        rel=1e-6,
        abs=1e-6,
    )

    collection.destroy()


def test_real_collection_query_context_routes_query_by_id_and_output_fields(
    tmp_path,
):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[hannsdb.FieldSchema(name="group", data_type="int64")],
        vectors=[
            hannsdb.VectorSchema(
                name="dense",
                data_type="vector_fp32",
                dimension=2,
            )
        ],
    )
    collection = hannsdb.create_and_open(str(tmp_path), schema)
    docs = [
        hannsdb.Doc(
            id="11",
            vector=[0.0, 0.0],
            field_name="dense",
            fields={"group": 1, "color": "red"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="12",
            vector=[0.2, 0.0],
            field_name="dense",
            fields={"group": 1, "color": "blue"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="13",
            vector=[1.0, 0.0],
            field_name="dense",
            fields={"group": 2, "color": "green"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="14",
            vector=[2.0, 0.0],
            field_name="dense",
            fields={"group": 1, "color": "yellow"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="15",
            vector=[10.0, 10.0],
            field_name="dense",
            fields={"group": 2, "color": "purple"},
            score=0.0,
        ),
    ]
    query_vector = [0.1, 0.0]
    assert collection.insert(docs) == len(docs)

    context = hannsdb.QueryContext(
        top_k=3,
        queries=[hannsdb.VectorQuery(field_name="dense", vector=query_vector, param=None)],
        query_by_id=["12"],
        output_fields=["group"],
    )
    result = collection.query(context)

    assert len(result) == 3
    assert result[0].id == "12"
    assert result[0].score == pytest.approx(0.0, abs=1e-6)
    assert "12" in [doc.id for doc in result]
    assert [doc.field("group") for doc in result] == [1, 1, 2]
    assert [doc.fields for doc in result] == [{"group": 1}, {"group": 1}, {"group": 2}]
    assert all(not doc.has_field("color") for doc in result)
    assert all(not doc.has_field("tag") for doc in result)
    assert [doc.score for doc in result] == sorted(doc.score for doc in result)

    collection.destroy()


def test_real_collection_query_context_routes_query_by_id_field_name_to_secondary_vector(
    tmp_path,
):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[hannsdb.FieldSchema(name="group", data_type="int64")],
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
    docs = [
        hannsdb.Doc(
            id="11",
            vectors={"dense": [9.0, 9.0], "title": [0.0, 0.0]},
            field_name="dense",
            fields={"group": 1},
            score=0.0,
        ),
        hannsdb.Doc(
            id="12",
            vectors={"dense": [0.0, 0.0], "title": [0.2, 0.0]},
            field_name="dense",
            fields={"group": 1},
            score=0.0,
        ),
        hannsdb.Doc(
            id="13",
            vectors={"dense": [1.0, 0.0], "title": [1.0, 0.0]},
            field_name="dense",
            fields={"group": 2},
            score=0.0,
        ),
    ]
    assert collection.insert(docs) == len(docs)

    context = hannsdb.QueryContext(
        top_k=3,
        queries=[hannsdb.VectorQuery(field_name="dense", vector=[0.0, 0.0], param=None)],
        query_by_id=["11"],
        query_by_id_field_name="title",
        output_fields=["group"],
    )
    result = collection.query(context)

    assert [doc.id for doc in result] == ["11", "12", "13"]
    assert [doc.field("group") for doc in result] == [1, 1, 2]
    assert result[0].score == pytest.approx(0.0, abs=1e-6)

    collection.destroy()


def test_real_collection_query_context_defaults_query_by_id_to_primary_vector(
    tmp_path,
):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[hannsdb.FieldSchema(name="group", data_type="int64")],
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
    docs = [
        hannsdb.Doc(
            id="11",
            vectors={"dense": [5.0, 5.0], "title": [0.0, 0.0]},
            field_name="dense",
            fields={"group": 1},
            score=0.0,
        ),
        hannsdb.Doc(
            id="12",
            vectors={"dense": [0.0, 0.0], "title": [0.2, 0.0]},
            field_name="dense",
            fields={"group": 1},
            score=0.0,
        ),
        hannsdb.Doc(
            id="13",
            vectors={"dense": [1.0, 1.0], "title": [1.0, 0.0]},
            field_name="dense",
            fields={"group": 2},
            score=0.0,
        ),
    ]
    assert collection.insert(docs) == len(docs)

    context = hannsdb.QueryContext(
        top_k=3,
        query_by_id=["11"],
        output_fields=["group"],
    )
    result = collection.query(context)

    assert [doc.id for doc in result] == ["11", "13", "12"]
    assert [doc.field("group") for doc in result] == [1, 2, 1]
    assert result[0].score == pytest.approx(0.0, abs=1e-6)

    collection.destroy()


def test_real_collection_query_accepts_query_by_id_field_name_legacy_kwargs(
    tmp_path,
):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[hannsdb.FieldSchema(name="group", data_type="int64")],
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
    docs = [
        hannsdb.Doc(
            id="11",
            vectors={"dense": [9.0, 9.0], "title": [0.0, 0.0]},
            field_name="dense",
            fields={"group": 1},
            score=0.0,
        ),
        hannsdb.Doc(
            id="12",
            vectors={"dense": [0.0, 0.0], "title": [0.2, 0.0]},
            field_name="dense",
            fields={"group": 1},
            score=0.0,
        ),
        hannsdb.Doc(
            id="13",
            vectors={"dense": [1.0, 0.0], "title": [1.0, 0.0]},
            field_name="dense",
            fields={"group": 2},
            score=0.0,
        ),
    ]
    assert collection.insert(docs) == len(docs)

    result = collection.query(
        vectors=hannsdb.VectorQuery(field_name="dense", vector=[0.0, 0.0], param=None),
        query_by_id=["11"],
        query_by_id_field_name="title",
        topk=3,
    )

    assert [doc.id for doc in result] == ["11", "12", "13"]
    assert [doc.field("group") for doc in result] == [1, 1, 2]

    collection.destroy()


def test_real_collection_query_accepts_legacy_kwargs_with_query_by_id_and_output_fields(
    tmp_path,
):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[
            hannsdb.FieldSchema(name="group", data_type="int64"),
            hannsdb.FieldSchema(name="color", data_type="string"),
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
    docs = [
        hannsdb.Doc(
            id="11",
            vector=[0.0, 0.0],
            field_name="dense",
            fields={"group": 1, "color": "red"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="12",
            vector=[0.2, 0.0],
            field_name="dense",
            fields={"group": 1, "color": "blue"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="13",
            vector=[1.0, 0.0],
            field_name="dense",
            fields={"group": 2, "color": "green"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="14",
            vector=[2.0, 0.0],
            field_name="dense",
            fields={"group": 1, "color": "yellow"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="15",
            vector=[10.0, 10.0],
            field_name="dense",
            fields={"group": 2, "color": "purple"},
            score=0.0,
        ),
    ]
    query_vector = [0.1, 0.0]
    assert collection.insert(docs) == len(docs)

    result = collection.query(
        vectors=hannsdb.VectorQuery(field_name="dense", vector=query_vector, param=None),
        output_fields=["group"],
        topk=3,
        filter="",
        query_by_id=["12"],
    )

    assert len(result) == 3
    assert result[0].id == "12"
    assert result[0].score == pytest.approx(0.0, abs=1e-6)
    assert "12" in [doc.id for doc in result]
    assert [doc.field("group") for doc in result] == [1, 1, 2]
    assert [doc.fields for doc in result] == [{"group": 1}, {"group": 1}, {"group": 2}]
    assert all(not doc.has_field("color") for doc in result)
    assert all(not doc.has_field("tag") for doc in result)
    assert [doc.score for doc in result] == sorted(doc.score for doc in result)

    collection.destroy()


def test_real_collection_query_accepts_scalar_query_by_id_and_output_fields_legacy_kwargs(
    tmp_path,
):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[
            hannsdb.FieldSchema(name="group", data_type="int64"),
            hannsdb.FieldSchema(name="color", data_type="string"),
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
    docs = [
        hannsdb.Doc(
            id="11",
            vector=[0.0, 0.0],
            field_name="dense",
            fields={"group": 1, "color": "red"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="12",
            vector=[0.2, 0.0],
            field_name="dense",
            fields={"group": 1, "color": "blue"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="13",
            vector=[1.0, 0.0],
            field_name="dense",
            fields={"group": 2, "color": "green"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="14",
            vector=[2.0, 0.0],
            field_name="dense",
            fields={"group": 1, "color": "yellow"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="15",
            vector=[10.0, 10.0],
            field_name="dense",
            fields={"group": 2, "color": "purple"},
            score=0.0,
        ),
    ]
    query_vector = [0.1, 0.0]
    assert collection.insert(docs) == len(docs)

    result = collection.query(
        vectors=hannsdb.VectorQuery(field_name="dense", vector=query_vector, param=None),
        output_fields="group",
        topk=3,
        filter="",
        query_by_id="12",
    )

    assert len(result) == 3
    assert result[0].id == "12"
    assert result[0].score == pytest.approx(0.0, abs=1e-6)
    assert "12" in [doc.id for doc in result]
    assert [doc.field("group") for doc in result] == [1, 1, 2]
    assert [doc.fields for doc in result] == [{"group": 1}, {"group": 1}, {"group": 2}]
    assert all(not doc.has_field("color") for doc in result)
    assert all(not doc.has_field("tag") for doc in result)
    assert [doc.score for doc in result] == sorted(doc.score for doc in result)

    collection.destroy()


def test_real_collection_query_accepts_filter_only_legacy_kwargs_and_projects_output_fields(
    tmp_path,
):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[
            hannsdb.FieldSchema(name="group", data_type="int64"),
            hannsdb.FieldSchema(name="color", data_type="string"),
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
    docs = [
        hannsdb.Doc(
            id="11",
            vector=[0.0, 0.0],
            field_name="dense",
            fields={"group": 1, "color": "red"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="12",
            vector=[0.1, 0.0],
            field_name="dense",
            fields={"group": 1, "color": "blue"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="13",
            vector=[0.2, 0.0],
            field_name="dense",
            fields={"group": 1, "color": "green"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="21",
            vector=[1.0, 0.0],
            field_name="dense",
            fields={"group": 2, "color": "yellow"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="22",
            vector=[2.0, 0.0],
            field_name="dense",
            fields={"group": 2, "color": "purple"},
            score=0.0,
        ),
    ]
    assert collection.insert(docs) == len(docs)

    result = collection.query(
        vectors=None,
        output_fields=["group"],
        topk=2,
        filter="group == 1",
    )

    assert len(result) == 2
    assert [doc.id for doc in result] == ["11", "12"]
    assert [doc.field("group") for doc in result] == [1, 1]
    assert [doc.fields for doc in result] == [{"group": 1}, {"group": 1}]
    assert all(not doc.has_field("color") for doc in result)
    assert all(not doc.has_field("tag") for doc in result)

    collection.destroy()


def test_real_collection_query_rejects_filter_only_group_by_legacy_kwargs(
    tmp_path,
):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[
            hannsdb.FieldSchema(name="group", data_type="int64"),
            hannsdb.FieldSchema(name="color", data_type="string"),
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
    docs = [
        hannsdb.Doc(
            id="11",
            vector=[0.0, 0.0],
            field_name="dense",
            fields={"group": 1, "color": "red"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="12",
            vector=[0.1, 0.0],
            field_name="dense",
            fields={"group": 1, "color": "blue"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="13",
            vector=[0.2, 0.0],
            field_name="dense",
            fields={"group": 2, "color": "green"},
            score=0.0,
        ),
    ]
    assert collection.insert(docs) == len(docs)

    with pytest.raises(NotImplementedError) as exc_info:
        collection.query(
            vectors=None,
            output_fields=["group"],
            topk=2,
            filter="group == 1",
            group_by=hannsdb.QueryGroupBy(field_name="group"),
        )

    message = str(exc_info.value)
    assert "group_by requires at least one recall source" in message
    assert "unsupported:" in message

    collection.destroy()


def test_real_collection_query_accepts_query_context_keyword_alias(tmp_path):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[
            hannsdb.FieldSchema(name="group", data_type="int64"),
            hannsdb.FieldSchema(name="color", data_type="string"),
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
    docs = [
        hannsdb.Doc(
            id="11",
            vector=[0.0, 0.0],
            field_name="dense",
            fields={"group": 1, "color": "red"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="12",
            vector=[0.2, 0.0],
            field_name="dense",
            fields={"group": 1, "color": "blue"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="13",
            vector=[1.0, 0.0],
            field_name="dense",
            fields={"group": 2, "color": "green"},
            score=0.0,
        ),
    ]
    assert collection.insert(docs) == len(docs)

    context = hannsdb.QueryContext(
        top_k=1,
        queries=[hannsdb.VectorQuery(field_name="dense", vector=[0.0, 0.0], param=None)],
        output_fields=["group"],
    )
    result = collection.query(query_context=context)

    assert len(result) == 1
    assert result[0].id == "11"
    assert [doc.field("group") for doc in result] == [1]
    assert [doc.fields for doc in result] == [{"group": 1}]
    assert all(not doc.has_field("color") for doc in result)

    collection.destroy()


def test_real_collection_query_accepts_context_keyword_alias(tmp_path):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[
            hannsdb.FieldSchema(name="group", data_type="int64"),
            hannsdb.FieldSchema(name="color", data_type="string"),
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
    docs = [
        hannsdb.Doc(
            id="11",
            vector=[0.0, 0.0],
            field_name="dense",
            fields={"group": 1, "color": "red"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="12",
            vector=[0.2, 0.0],
            field_name="dense",
            fields={"group": 1, "color": "blue"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="13",
            vector=[1.0, 0.0],
            field_name="dense",
            fields={"group": 2, "color": "green"},
            score=0.0,
        ),
    ]
    assert collection.insert(docs) == len(docs)

    context = hannsdb.QueryContext(
        top_k=1,
        queries=[hannsdb.VectorQuery(field_name="dense", vector=[0.0, 0.0], param=None)],
        output_fields=["group"],
    )
    result = collection.query(context=context)

    assert len(result) == 1
    assert result[0].id == "11"
    assert [doc.field("group") for doc in result] == [1]
    assert [doc.fields for doc in result] == [{"group": 1}]
    assert all(not doc.has_field("color") for doc in result)

    collection.destroy()


def test_real_collection_query_accepts_query_context_via_vectors_argument(tmp_path):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[
            hannsdb.FieldSchema(name="group", data_type="int64"),
            hannsdb.FieldSchema(name="color", data_type="string"),
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
    docs = [
        hannsdb.Doc(
            id="11",
            vector=[0.0, 0.0],
            field_name="dense",
            fields={"group": 1, "color": "red"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="12",
            vector=[0.2, 0.0],
            field_name="dense",
            fields={"group": 1, "color": "blue"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="13",
            vector=[1.0, 0.0],
            field_name="dense",
            fields={"group": 2, "color": "green"},
            score=0.0,
        ),
    ]
    assert collection.insert(docs) == len(docs)

    context = hannsdb.QueryContext(
        top_k=1,
        queries=[hannsdb.VectorQuery(field_name="dense", vector=[0.0, 0.0], param=None)],
        output_fields=["group"],
    )
    result = collection.query(vectors=context)

    assert len(result) == 1
    assert result[0].id == "11"
    assert [doc.field("group") for doc in result] == [1]
    assert [doc.fields for doc in result] == [{"group": 1}]
    assert all(not doc.has_field("color") for doc in result)

    collection.destroy()


def test_real_collection_query_accepts_secondary_vector_field_in_query_context(tmp_path):
    collection = build_secondary_vector_collection(tmp_path)

    context = hannsdb.QueryContext(
        top_k=3,
        queries=[hannsdb.VectorQuery(field_name="title", vector=[0.0, 0.0], param=None)],
        output_fields=["group"],
    )
    result = collection.query(context)

    assert [doc.id for doc in result] == ["1", "2", "3"]
    assert [doc.field("group") for doc in result] == [1, 1, 2]
    assert [doc.fields for doc in result] == [{"group": 1}, {"group": 1}, {"group": 2}]

    collection.destroy()


def test_real_collection_query_accepts_legacy_native_vector_query_object(tmp_path):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[
            hannsdb.FieldSchema(name="group", data_type="int64"),
            hannsdb.FieldSchema(name="color", data_type="string"),
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
    docs = [
        hannsdb.Doc(
            id="1",
            vector=[0.0, 0.0],
            field_name="dense",
            fields={"group": 1, "color": "red"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="2",
            vector=[0.1, 0.0],
            field_name="dense",
            fields={"group": 1, "color": "blue"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="3",
            vector=[0.2, 0.0],
            field_name="dense",
            fields={"group": 2, "color": "green"},
            score=0.0,
        ),
    ]
    assert collection.insert(docs) == len(docs)

    legacy_query = hannsdb._native.VectorQuery(
        field_name="dense",
        vector=[0.0, 0.0],
        param=None,
    )
    result = collection.query(vectors=legacy_query, output_fields=["group"], topk=2, filter="")

    assert [doc.id for doc in result] == ["1", "2"]
    assert [doc.field("group") for doc in result] == [1, 1]
    assert [doc.fields for doc in result] == [{"group": 1}, {"group": 1}]
    assert all(not doc.has_field("color") for doc in result)
    assert all(not doc.has_field("tag") for doc in result)

    collection.destroy()


def test_real_collection_query_accepts_secondary_vector_field_via_legacy_kwargs(tmp_path):
    collection = build_secondary_vector_collection(tmp_path)

    result = collection.query(
        vectors=hannsdb.VectorQuery(field_name="title", vector=[0.0, 0.0], param=None),
        output_fields=["group"],
        topk=3,
        filter="",
    )

    assert [doc.id for doc in result] == ["1", "2", "3"]
    assert [doc.field("group") for doc in result] == [1, 1, 2]
    assert [doc.fields for doc in result] == [{"group": 1}, {"group": 1}, {"group": 2}]

    collection.destroy()


def test_real_collection_query_accepts_pure_hnsw_query_param(tmp_path):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[
            hannsdb.FieldSchema(name="group", data_type="int64"),
            hannsdb.FieldSchema(name="color", data_type="string"),
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
    docs = [
        hannsdb.Doc(
            id="1",
            vector=[0.0, 0.0],
            field_name="dense",
            fields={"group": 1, "color": "red"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="2",
            vector=[0.1, 0.0],
            field_name="dense",
            fields={"group": 1, "color": "blue"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="3",
            vector=[0.2, 0.0],
            field_name="dense",
            fields={"group": 2, "color": "green"},
            score=0.0,
        ),
    ]
    assert collection.insert(docs) == len(docs)

    query = hannsdb.VectorQuery(
        field_name="dense",
        vector=[0.0, 0.0],
        param=hannsdb.HnswQueryParam(ef=64, is_using_refiner=False),
    )
    result = collection.query(vectors=query, output_fields=["group"], topk=2, filter="")

    assert [doc.id for doc in result] == ["1", "2"]
    assert [doc.field("group") for doc in result] == [1, 1]
    assert [doc.fields for doc in result] == [{"group": 1}, {"group": 1}]
    assert all(not doc.has_field("color") for doc in result)
    assert all(not doc.has_field("tag") for doc in result)

    with pytest.raises(NotImplementedError) as exc_info:
        collection.query(
            vectors=query,
            output_fields=["group"],
            topk=2,
            filter="group == 1",
        )
    assert "ef_search" in str(exc_info.value)
    assert "typed single-vector fast path" in str(exc_info.value)
    assert "unsupported:" in str(exc_info.value)

    collection.destroy()


def test_real_collection_query_rejects_both_query_context_and_context(tmp_path):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[hannsdb.FieldSchema(name="group", data_type="int64")],
        vectors=[
            hannsdb.VectorSchema(
                name="dense",
                data_type="vector_fp32",
                dimension=2,
            )
        ],
    )
    collection = hannsdb.create_and_open(str(tmp_path), schema)
    assert collection.insert(
        [
            hannsdb.Doc(
                id="1",
                vector=[0.0, 0.0],
                field_name="dense",
                fields={"group": 1},
                score=0.0,
            ),
            hannsdb.Doc(
                id="2",
                vector=[0.1, 0.0],
                field_name="dense",
                fields={"group": 1},
                score=0.0,
            ),
        ]
    ) == 2

    context = hannsdb.QueryContext(
        top_k=1,
        queries=[hannsdb.VectorQuery(field_name="dense", vector=[0.0, 0.0], param=None)],
        output_fields=["group"],
    )

    try:
        with pytest.raises(TypeError, match="both query_context and context"):
            collection.query(query_context=context, context=context)
    finally:
        collection.destroy()


def test_real_collection_query_accepts_legacy_kwargs_with_multiple_queries_and_output_fields(
    tmp_path,
):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[
            hannsdb.FieldSchema(name="group", data_type="int64"),
            hannsdb.FieldSchema(name="tag", data_type="string"),
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
    docs = [
        hannsdb.Doc(
            id="11",
            vector=[0.0, 0.0],
            field_name="dense",
            fields={"group": 1, "tag": "shared"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="12",
            vector=[-0.05, 0.06],
            field_name="dense",
            fields={"group": 1, "tag": "q1-only"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="13",
            vector=[0.05, 0.06],
            field_name="dense",
            fields={"group": 2, "tag": "q2-only"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="14",
            vector=[-0.05, 0.07],
            field_name="dense",
            fields={"group": 1, "tag": "q1-distractor"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="15",
            vector=[0.05, 0.07],
            field_name="dense",
            fields={"group": 2, "tag": "q2-distractor"},
            score=0.0,
        ),
    ]
    assert collection.insert(docs) == len(docs)

    result = collection.query(
        vectors=[
            hannsdb.VectorQuery(field_name="dense", vector=[-0.05, 0.0], param=None),
            hannsdb.VectorQuery(field_name="dense", vector=[0.05, 0.0], param=None),
        ],
        output_fields=["group"],
        topk=3,
        filter="",
    )

    assert len(result) == 3
    assert result[0].id == "11"
    assert {doc.id for doc in result} == {"11", "12", "13"}
    assert len({doc.id for doc in result}) == len(result)
    assert "14" not in {doc.id for doc in result}
    assert "15" not in {doc.id for doc in result}
    assert sorted(doc.field("group") for doc in result) == [1, 1, 2]
    assert all(doc.fields == {"group": doc.field("group")} for doc in result)
    assert all(not doc.has_field("tag") for doc in result)
    assert result[0].score < result[1].score
    assert result[0].score < result[2].score

    collection.destroy()


def test_real_collection_query_context_merges_multiple_queries_and_projects_output_fields(
    tmp_path,
):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[
            hannsdb.FieldSchema(name="group", data_type="int64"),
            hannsdb.FieldSchema(name="tag", data_type="string"),
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
    docs = [
        hannsdb.Doc(
            id="11",
            vector=[0.0, 0.0],
            field_name="dense",
            fields={"group": 1, "tag": "shared"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="12",
            vector=[-0.05, 0.06],
            field_name="dense",
            fields={"group": 1, "tag": "q1-only"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="13",
            vector=[0.05, 0.06],
            field_name="dense",
            fields={"group": 2, "tag": "q2-only"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="14",
            vector=[-0.05, 0.07],
            field_name="dense",
            fields={"group": 1, "tag": "q1-distractor"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="15",
            vector=[0.05, 0.07],
            field_name="dense",
            fields={"group": 2, "tag": "q2-distractor"},
            score=0.0,
        ),
    ]
    assert collection.insert(docs) == len(docs)

    context = hannsdb.QueryContext(
        top_k=3,
        queries=[
            hannsdb.VectorQuery(field_name="dense", vector=[-0.05, 0.0], param=None),
            hannsdb.VectorQuery(field_name="dense", vector=[0.05, 0.0], param=None),
        ],
        output_fields=["group"],
    )
    result = collection.query(context)

    assert len(result) == 3
    assert result[0].id == "11"
    assert set(doc.id for doc in result) == {"11", "12", "13"}
    assert len({doc.id for doc in result}) == len(result)
    assert "14" not in {doc.id for doc in result}
    assert "15" not in {doc.id for doc in result}
    assert sorted(doc.field("group") for doc in result) == [1, 1, 2]
    assert all(doc.fields == {"group": doc.field("group")} for doc in result)
    assert all(not doc.has_field("tag") for doc in result)
    assert result[0].score < result[1].score
    assert result[0].score < result[2].score

    collection.destroy()


def test_real_collection_query_context_applies_builtin_rrf_reranker_and_output_fields(
    tmp_path,
):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[
            hannsdb.FieldSchema(name="group", data_type="int64"),
            hannsdb.FieldSchema(name="tag", data_type="string"),
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
    docs = [
        hannsdb.Doc(
            id="1",
            vector=[0.0, 0.0],
            field_name="dense",
            fields={"group": 1, "tag": "a"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="2",
            vector=[0.1, 0.0],
            field_name="dense",
            fields={"group": 1, "tag": "b"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="3",
            vector=[0.2, 0.0],
            field_name="dense",
            fields={"group": 2, "tag": "c"},
            score=0.0,
        ),
    ]
    assert collection.insert(docs) == len(docs)

    context = hannsdb.QueryContext(
        top_k=2,
        queries=[
            hannsdb.VectorQuery(field_name="dense", vector=[0.0, 0.0], param=None),
            hannsdb.VectorQuery(field_name="dense", vector=[0.2, 0.0], param=None),
        ],
        reranker=hannsdb.RrfReRanker(topn=3, rank_constant=60),
        output_fields=["group"],
    )
    result = collection.query(context)

    assert result[0].id == "2"
    assert {doc.id for doc in result[1:]} == {"1", "3"}
    assert [doc.field("group") for doc in result] == [1, 1, 2]
    assert all(doc.fields == {"group": doc.field("group")} for doc in result)
    assert all(not doc.has_field("tag") for doc in result)
    assert result[0].score == pytest.approx(2.0 / 62.0, rel=1e-6)
    assert result[1].score == pytest.approx(1.0 / 61.0, rel=1e-6)
    assert result[2].score == pytest.approx(1.0 / 61.0, rel=1e-6)

    collection.destroy()


def test_real_collection_query_accepts_legacy_kwargs_with_builtin_rrf_reranker(
    tmp_path,
):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[
            hannsdb.FieldSchema(name="group", data_type="int64"),
            hannsdb.FieldSchema(name="tag", data_type="string"),
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
    docs = [
        hannsdb.Doc(
            id="1",
            vector=[0.0, 0.0],
            field_name="dense",
            fields={"group": 1, "tag": "a"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="2",
            vector=[0.1, 0.0],
            field_name="dense",
            fields={"group": 1, "tag": "b"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="3",
            vector=[0.2, 0.0],
            field_name="dense",
            fields={"group": 2, "tag": "c"},
            score=0.0,
        ),
    ]
    assert collection.insert(docs) == len(docs)

    result = collection.query(
        vectors=[
            hannsdb.VectorQuery(field_name="dense", vector=[0.0, 0.0], param=None),
            hannsdb.VectorQuery(field_name="dense", vector=[0.2, 0.0], param=None),
        ],
        output_fields=["group"],
        topk=2,
        reranker=hannsdb.RrfReRanker(topn=3, rank_constant=60),
        filter="",
    )

    assert result[0].id == "2"
    assert {doc.id for doc in result[1:]} == {"1", "3"}
    assert [doc.field("group") for doc in result] == [1, 1, 2]
    assert all(doc.fields == {"group": doc.field("group")} for doc in result)
    assert all(not doc.has_field("tag") for doc in result)
    assert result[0].score == pytest.approx(2.0 / 62.0, rel=1e-6)
    assert result[1].score == pytest.approx(1.0 / 61.0, rel=1e-6)
    assert result[2].score == pytest.approx(1.0 / 61.0, rel=1e-6)

    collection.destroy()


def test_real_collection_query_context_rejects_group_by_with_reranker(tmp_path):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[hannsdb.FieldSchema(name="group", data_type="int64")],
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
                field_name="dense",
                fields={"group": 1},
                score=0.0,
            ),
            hannsdb.Doc(
                id="2",
                vector=[0.1, 0.0],
                field_name="dense",
                fields={"group": 1},
                score=0.0,
            ),
            hannsdb.Doc(
                id="3",
                vector=[0.2, 0.0],
                field_name="dense",
                fields={"group": 2},
                score=0.0,
            ),
        ]
    )

    context = hannsdb.QueryContext(
        top_k=2,
        queries=[
            hannsdb.VectorQuery(field_name="dense", vector=[0.0, 0.0], param=None)
        ],
        group_by=hannsdb.QueryGroupBy(field_name="group"),
        reranker=hannsdb.RrfReRanker(topn=2),
    )

    try:
        with pytest.raises(NotImplementedError, match="group_by"):
            collection.query(context)
    finally:
        collection.destroy()


def test_real_collection_query_rejects_group_by_with_reranker_legacy_kwargs(tmp_path):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[hannsdb.FieldSchema(name="group", data_type="int64")],
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
                field_name="dense",
                fields={"group": 1},
                score=0.0,
            ),
            hannsdb.Doc(
                id="2",
                vector=[0.1, 0.0],
                field_name="dense",
                fields={"group": 1},
                score=0.0,
            ),
            hannsdb.Doc(
                id="3",
                vector=[0.2, 0.0],
                field_name="dense",
                fields={"group": 2},
                score=0.0,
            ),
        ]
    )

    try:
        with pytest.raises(NotImplementedError) as exc_info:
            collection.query(
                vectors=hannsdb.VectorQuery(
                    field_name="dense",
                    vector=[0.0, 0.0],
                    param=None,
                ),
                output_fields=[],
                topk=2,
                filter="",
                group_by=hannsdb.QueryGroupBy(field_name="group"),
                reranker=hannsdb.RrfReRanker(topn=2),
            )
        assert str(exc_info.value) == (
            "group_by is not supported by the Python facade yet"
        )
        assert "unsupported:" not in str(exc_info.value)
    finally:
        collection.destroy()


def test_real_collection_query_context_rejects_query_by_id_with_reranker(tmp_path):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[hannsdb.FieldSchema(name="group", data_type="int64")],
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
                field_name="dense",
                fields={"group": 1},
                score=0.0,
            ),
            hannsdb.Doc(
                id="2",
                vector=[0.1, 0.0],
                field_name="dense",
                fields={"group": 1},
                score=0.0,
            ),
            hannsdb.Doc(
                id="3",
                vector=[0.2, 0.0],
                field_name="dense",
                fields={"group": 2},
                score=0.0,
            ),
        ]
    )

    context = hannsdb.QueryContext(
        top_k=2,
        queries=[
            hannsdb.VectorQuery(field_name="dense", vector=[0.0, 0.0], param=None)
        ],
        query_by_id=["1"],
        reranker=hannsdb.RrfReRanker(topn=2),
    )

    try:
        with pytest.raises(NotImplementedError, match="query_by_id"):
            collection.query(context)
    finally:
        collection.destroy()


def test_real_collection_query_rejects_query_by_id_with_reranker_legacy_kwargs(tmp_path):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[hannsdb.FieldSchema(name="group", data_type="int64")],
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
                field_name="dense",
                fields={"group": 1},
                score=0.0,
            ),
            hannsdb.Doc(
                id="2",
                vector=[0.1, 0.0],
                field_name="dense",
                fields={"group": 1},
                score=0.0,
            ),
            hannsdb.Doc(
                id="3",
                vector=[0.2, 0.0],
                field_name="dense",
                fields={"group": 2},
                score=0.0,
            ),
        ]
    )

    try:
        with pytest.raises(NotImplementedError) as exc_info:
            collection.query(
                vectors=hannsdb.VectorQuery(
                    field_name="dense",
                    vector=[0.0, 0.0],
                    param=None,
                ),
                output_fields=[],
                topk=2,
                filter="",
                query_by_id=["1"],
                reranker=hannsdb.RrfReRanker(topn=2),
            )
        assert str(exc_info.value) == (
            "query_by_id is not supported by the Python facade yet"
        )
        assert "unsupported:" not in str(exc_info.value)
    finally:
        collection.destroy()


def test_real_collection_query_context_returns_vectors_when_requested(tmp_path):
    collection = build_multi_vector_collection(tmp_path)
    context = hannsdb.QueryContext(
        top_k=1,
        queries=[
            hannsdb.VectorQuery(field_name="dense", vector=[0.0, 0.0], param=None)
        ],
        output_fields=["group"],
        include_vector=True,
    )

    result = collection.query(context)

    assert [doc.id for doc in result] == ["1"]
    assert result[0].field("group") == 1
    assert result[0].vectors["dense"] == [0.0, 0.0]
    assert result[0].vectors["sparse"] == [1.0, 1.0]
    collection.destroy()


def test_real_collection_query_accepts_include_vector_legacy_kwargs(tmp_path):
    collection = build_multi_vector_collection(tmp_path)

    result = collection.query(
        vectors=hannsdb.VectorQuery(
            field_name="dense",
            vector=[0.0, 0.0],
            param=None,
        ),
        output_fields=["group"],
        topk=1,
        filter="",
        include_vector=True,
    )

    assert [doc.id for doc in result] == ["1"]
    assert result[0].field("group") == 1
    assert result[0].vectors["dense"] == [0.0, 0.0]
    assert result[0].vectors["sparse"] == [1.0, 1.0]


def test_real_collection_query_context_applies_group_by_and_output_fields(
    tmp_path,
):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[
            hannsdb.FieldSchema(name="group", data_type="int64"),
            hannsdb.FieldSchema(name="tag", data_type="string"),
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
    docs = [
        hannsdb.Doc(
            id="12",
            vector=[0.5, 0.0],
            field_name="dense",
            fields={"group": 1, "tag": "far"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="11",
            vector=[0.0, 0.0],
            field_name="dense",
            fields={"group": 1, "tag": "near"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="21",
            vector=[1.0, 0.0],
            field_name="dense",
            fields={"group": 2, "tag": "near"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="22",
            vector=[2.0, 0.0],
            field_name="dense",
            fields={"group": 2, "tag": "far"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="31",
            vector=[10.0, 10.0],
            field_name="dense",
            fields={"group": 3, "tag": "far"},
            score=0.0,
        ),
    ]
    query_vector = [0.0, 0.0]
    assert collection.insert(docs) == len(docs)

    context = hannsdb.QueryContext(
        top_k=2,
        queries=[hannsdb.VectorQuery(field_name="dense", vector=query_vector, param=None)],
        group_by=hannsdb.QueryGroupBy(field_name="group"),
        output_fields=["group"],
    )
    result = collection.query(context)

    assert [doc.id for doc in result] == ["11", "21"]
    assert [doc.field("group") for doc in result] == [1, 2]
    assert all(not doc.has_field("tag") for doc in result)
    assert [doc.fields for doc in result] == [{"group": 1}, {"group": 2}]
    assert [doc.score for doc in result] == pytest.approx([0.0, 1.0], abs=1e-6)

    collection.destroy()


def test_real_collection_query_accepts_legacy_kwargs_with_group_by_and_output_fields(
    tmp_path,
):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[
            hannsdb.FieldSchema(name="group", data_type="int64"),
            hannsdb.FieldSchema(name="tag", data_type="string"),
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
    docs = [
        hannsdb.Doc(
            id="12",
            vector=[0.5, 0.0],
            field_name="dense",
            fields={"group": 1, "tag": "far"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="11",
            vector=[0.0, 0.0],
            field_name="dense",
            fields={"group": 1, "tag": "near"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="21",
            vector=[1.0, 0.0],
            field_name="dense",
            fields={"group": 2, "tag": "near"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="22",
            vector=[2.0, 0.0],
            field_name="dense",
            fields={"group": 2, "tag": "far"},
            score=0.0,
        ),
        hannsdb.Doc(
            id="31",
            vector=[10.0, 10.0],
            field_name="dense",
            fields={"group": 3, "tag": "far"},
            score=0.0,
        ),
    ]
    assert collection.insert(docs) == len(docs)

    result = collection.query(
        vectors=hannsdb.VectorQuery(field_name="dense", vector=[0.0, 0.0], param=None),
        output_fields=["group"],
        topk=2,
        filter="",
        group_by=hannsdb.QueryGroupBy(field_name="group"),
    )

    assert [doc.id for doc in result] == ["11", "21"]
    assert [doc.field("group") for doc in result] == [1, 2]
    assert all(not doc.has_field("tag") for doc in result)
    assert [doc.fields for doc in result] == [{"group": 1}, {"group": 2}]
    assert [doc.score for doc in result] == pytest.approx([0.0, 1.0], abs=1e-6)

    collection.destroy()


def _l2_distance(left, right):
    return sum((x - y) ** 2 for x, y in zip(left, right)) ** 0.5


def test_collection_insert_and_upsert_accept_pure_and_native_docs(monkeypatch):
    schema = build_schema()
    calls = []
    np = pytest.importorskip("numpy")

    class FakeCore:
        path = "/tmp/hannsdb"
        collection_name = "docs"

        def insert(self, docs):
            calls.append(("insert", docs))
            return len(docs)

        def upsert(self, docs):
            calls.append(("upsert", docs))
            return len(docs)

    class FakeFactory:
        def build(self):
            return object()

    monkeypatch.setattr(hannsdb.QueryExecutorFactory, "create", lambda schema: FakeFactory())

    collection = hannsdb.Collection._from_core(FakeCore(), schema=schema)
    pure_doc = hannsdb.Doc(
        id="1",
        vectors={"dense": np.array([0.1, 0.2], dtype=np.float32)},
        fields={"session_id": "abc"},
    )
    native_doc = hannsdb._native.Doc(
        id="2",
        vector=[0.3, 0.4],
        field_name="dense",
        fields={"session_id": "def"},
    )

    assert collection.insert([pure_doc, native_doc]) == 2
    assert collection.upsert([pure_doc, native_doc]) == 2
    assert [call[0] for call in calls] == ["insert", "upsert"]
    assert all(doc.__class__ is hannsdb._native.Doc for doc in calls[0][1])
    assert calls[0][1][0].id == "1"
    assert calls[0][1][0].vectors["dense"] == pytest.approx([0.1, 0.2], rel=1e-6)
    assert calls[0][1][1].fields == {"session_id": "def"}


def test_collection_fetch_query_and_query_context_wrap_native_docs(monkeypatch):
    schema = build_schema()
    calls = []

    class FakeCore:
        path = "/tmp/hannsdb"
        collection_name = "docs"

        def query_context(self, context):
            calls.append(("query_context", context))
            return [
                hannsdb._native.Doc(
                    id="1",
                    vector=[0.1, 0.2],
                    field_name="dense",
                    fields={"session_id": "abc"},
                    score=0.25,
                )
            ]

        def fetch(self, ids):
            calls.append(("fetch", ids))
            return [
                hannsdb._native.Doc(
                    id="2",
                    vector=[0.3, 0.4],
                    field_name="dense",
                    fields={"session_id": "def"},
                    score=0.5,
                )
            ]

    class FakeExecutor:
        def execute(self, collection, context):
            calls.append(("execute", collection, context))
            return collection.query_context(context)

    class FakeFactory:
        def build(self):
            calls.append(("build", None, None))
            return FakeExecutor()

    monkeypatch.setattr(hannsdb.QueryExecutorFactory, "create", lambda schema: FakeFactory())

    collection = hannsdb.Collection._from_core(FakeCore(), schema=schema)

    fetched = collection.fetch(["2"])
    queried = collection.query(
        vectors=hannsdb.VectorQuery(field_name="dense", vector=[0.0, 0.0], param=None),
        output_fields=["session_id"],
        topk=1,
        filter="",
    )
    direct = collection.query_context(
        hannsdb.QueryContext(
            top_k=1,
            queries=[hannsdb.VectorQuery(field_name="dense", vector=[0.0, 0.0], param=None)],
        )
    )

    assert fetched[0].__class__ is hannsdb.Doc
    assert fetched[0].field("session_id") == "def"
    assert queried[0].__class__ is hannsdb.Doc
    assert queried[0].field("session_id") == "abc"
    assert direct[0].__class__ is hannsdb.Doc
    assert direct[0].field("session_id") == "abc"
    assert [call[0] for call in calls] == [
        "build",
        "fetch",
        "execute",
        "query_context",
        "query_context",
    ]


def build_schema():
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


def test_create_and_open_returns_python_facade_and_keeps_schema(tmp_path):
    schema = build_schema()

    collection = hannsdb.create_and_open(str(tmp_path), schema)

    assert isinstance(collection, hannsdb.Collection)
    assert collection.schema is schema
    assert collection.collection_name == "docs"
    assert collection.path == str(tmp_path)
    assert [vector.name for vector in collection.schema.vectors] == ["title", "dense"]
    assert collection.option == hannsdb.CollectionOption()
    assert collection.option.__class__ is hannsdb.CollectionOption

    collection.destroy()


def test_create_and_open_round_trips_collection_option_on_python_facade(tmp_path):
    schema = build_schema()
    option = hannsdb.CollectionOption(read_only=True, enable_mmap=False)

    collection = hannsdb.create_and_open(str(tmp_path), schema, option=option)

    assert collection.option == option
    assert collection.option.__class__ is hannsdb.CollectionOption

    collection.destroy()


def test_open_round_trips_collection_option_on_python_facade(tmp_path):
    schema = build_schema()
    option = hannsdb.CollectionOption(read_only=True, enable_mmap=False)

    created = hannsdb.create_and_open(str(tmp_path), schema, option=option)
    reopened = hannsdb.open(str(tmp_path), option=option)

    assert reopened.option == option
    assert reopened.option.__class__ is hannsdb.CollectionOption

    reopened.destroy()


def test_collection_from_core_wraps_native_option_as_python_wrapper(monkeypatch):
    schema = build_schema()

    class FakeCore:
        path = "/tmp/hannsdb"
        collection_name = "docs"
        option = hannsdb._native.CollectionOption(True, False)

    class FakeFactory:
        def build(self):
            return object()

    monkeypatch.setattr(hannsdb.QueryExecutorFactory, "create", lambda schema: FakeFactory())

    collection = hannsdb.Collection._from_core(FakeCore(), schema=schema)

    assert collection.option == hannsdb.CollectionOption(read_only=True, enable_mmap=False)
    assert collection.option.__class__ is hannsdb.CollectionOption


def test_create_and_open_accepts_pure_collection_option(monkeypatch, tmp_path):
    schema = build_schema()
    calls = []

    class FakeCore:
        path = str(tmp_path)
        collection_name = "docs"

    def fake_create_and_open(path, native_schema, option):
        calls.append((path, native_schema, option))
        return FakeCore()

    monkeypatch.setattr(hannsdb.model.collection._native_module, "create_and_open", fake_create_and_open)

    collection = hannsdb.create_and_open(
        str(tmp_path),
        schema,
        option=hannsdb.CollectionOption(read_only=True, enable_mmap=False),
    )

    assert calls[0][0] == str(tmp_path)
    assert calls[0][1].__class__ is hannsdb._native.CollectionSchema
    assert calls[0][2].__class__ is hannsdb._native.CollectionOption
    assert calls[0][2].read_only is True
    assert calls[0][2].enable_mmap is False
    assert collection.collection_name == "docs"


def test_collection_create_vector_index_accepts_pure_and_native_params(monkeypatch):
    calls = []

    class FakeCore:
        path = "/tmp/hannsdb"
        collection_name = "docs"

        def create_vector_index(self, field_name, index_param=None):
            calls.append((field_name, index_param))

    class FakeFactory:
        def build(self):
            return object()

    monkeypatch.setattr(hannsdb.QueryExecutorFactory, "create", lambda schema: FakeFactory())

    collection = hannsdb.Collection._from_core(FakeCore(), schema=build_schema())
    pure_param = hannsdb.HnswIndexParam(metric_type="cosine", m=32, ef_construction=128)
    native_param = hannsdb._native.IVFIndexParam(metric_type="l2", nlist=8)

    collection.create_vector_index("dense", pure_param)
    collection.create_vector_index("title", native_param)

    assert calls[0][0] == "dense"
    assert calls[0][1].__class__ is hannsdb._native.HnswIndexParam
    assert calls[1][0] == "title"
    assert calls[1][1].__class__ is hannsdb._native.IVFIndexParam


def test_open_recovers_schema_from_collection_metadata(tmp_path):
    created = hannsdb.create_and_open(str(tmp_path), build_schema())
    reopened = hannsdb.open(str(tmp_path))

    assert isinstance(reopened, hannsdb.Collection)
    assert reopened.schema.name == "docs"
    assert reopened.schema.primary_vector == "dense"
    assert [field.name for field in reopened.schema.fields] == ["session_id", "tags"]
    assert reopened.schema.fields[1].nullable is True
    assert reopened.schema.fields[1].array is True
    assert [vector.name for vector in reopened.schema.vectors] == ["title", "dense"]
    assert [vector.dimension for vector in reopened.schema.vectors] == [384, 384]

    created.destroy()


def test_collection_query_routes_through_executor_and_query_context_delegate(monkeypatch):
    schema = build_schema()
    calls = []

    class FakeCore:
        path = "/tmp/hannsdb"
        collection_name = "docs"

        def query_context(self, context):
            calls.append(("query_context", context))
            return ["native-result"]

    class FakeExecutor:
        def execute(self, collection, context):
            calls.append(("execute", collection, context))
            return collection.query_context(context)

    class FakeFactory:
        def build(self):
            calls.append(("build", None, None))
            return FakeExecutor()

    monkeypatch.setattr(hannsdb.QueryExecutorFactory, "create", lambda schema: FakeFactory())

    collection = hannsdb.Collection._from_core(FakeCore(), schema=schema)
    context = hannsdb.QueryContext(
        top_k=1,
        queries=[
            hannsdb.VectorQuery(field_name="dense", vector=[0.0, 0.0], param=None)
        ],
    )

    result = collection.query(context)

    assert result == ["native-result"]
    assert [call[0] for call in calls] == ["build", "execute", "query_context"]


def test_collection_query_accepts_legacy_vector_arguments(monkeypatch):
    schema = build_schema()
    calls = []

    class FakeCore:
        path = "/tmp/hannsdb"
        collection_name = "docs"

        def query_context(self, context):
            calls.append(("query_context", context))
            return ["native-result"]

    class FakeExecutor:
        def execute(self, collection, context):
            calls.append(("execute", collection, context))
            return collection.query_context(context)

    class FakeFactory:
        def build(self):
            calls.append(("build", None, None))
            return FakeExecutor()

    monkeypatch.setattr(hannsdb.QueryExecutorFactory, "create", lambda schema: FakeFactory())

    collection = hannsdb.Collection._from_core(FakeCore(), schema=schema)
    query = hannsdb.VectorQuery(field_name="dense", vector=[0.0, 0.0], param=None)

    result = collection.query(vectors=query, output_fields=[], topk=1, filter="")

    assert result == ["native-result"]
    assert [call[0] for call in calls] == ["build", "execute", "query_context"]
    _, _, context = calls[1]
    assert context.top_k == 1
    assert context.output_fields == []
    assert context.filter == ""
    assert len(context.queries) == 1
    assert context.queries[0].field_name == "dense"


def test_collection_query_accepts_pure_python_vector_query_with_numpy_input(tmp_path):
    np = pytest.importorskip("numpy")

    schema = build_schema()
    collection = hannsdb.Collection._from_core(
        type(
            "FakeCore",
            (),
            {
                "path": "/tmp/hannsdb",
                "collection_name": "docs",
                "query_context": lambda self, context: ["native-result"],
            },
        )(),
        schema=schema,
    )
    collection._querier = type(
        "FakeExecutor",
        (),
        {
            "execute": lambda self, core_collection, context: core_collection.query_context(
                context
            )
        },
    )()

    query = hannsdb.VectorQuery(
        field_name="dense",
        vector=np.array([[0.0, 0.0]], dtype=np.float32),
        param=None,
    )

    result = collection.query(vectors=query, output_fields=[], topk=1, filter="")

    assert result == ["native-result"]


def test_collection_surface_methods_delegate_to_core_handle(monkeypatch):
    schema = build_schema()
    calls = []

    class FakeCore:
        path = "/tmp/hannsdb"
        collection_name = "docs"
        stats = {"live_count": 3}

        def query_context(self, context):
            calls.append(("query_context", context))
            return ["native-result"]

        def insert(self, docs):
            calls.append(("insert", docs))
            return 1

        def upsert(self, docs):
            calls.append(("upsert", docs))
            return 2

        def fetch(self, ids):
            calls.append(("fetch", ids))
            return ["fetch-result"]

        def delete(self, ids):
            calls.append(("delete", ids))
            return 3

        def flush(self):
            calls.append(("flush", None))

        def destroy(self):
            calls.append(("destroy", None))

        def create_vector_index(self, field_name, index_param=None):
            calls.append(("create_vector_index", field_name, index_param))

        def drop_vector_index(self, field_name):
            calls.append(("drop_vector_index", field_name))

        def list_vector_indexes(self):
            calls.append(("list_vector_indexes", None))
            return ["dense"]

        def create_scalar_index(self, field_name):
            calls.append(("create_scalar_index", field_name))

        def drop_scalar_index(self, field_name):
            calls.append(("drop_scalar_index", field_name))

        def list_scalar_indexes(self):
            calls.append(("list_scalar_indexes", None))
            return ["session_id"]

    class FakeFactory:
        def build(self):
            return object()

    monkeypatch.setattr(hannsdb.QueryExecutorFactory, "create", lambda schema: FakeFactory())

    collection = hannsdb.Collection._from_core(FakeCore(), schema=schema)

    assert collection.path == "/tmp/hannsdb"
    assert collection.collection_name == "docs"
    assert collection.stats == {"live_count": 3}
    assert collection.insert([]) == 1
    assert collection.upsert([]) == 2
    assert collection.fetch(["1"]) == ["fetch-result"]
    assert collection.delete(["1"]) == 3
    assert collection.list_vector_indexes() == ["dense"]
    assert collection.list_scalar_indexes() == ["session_id"]
    collection.flush()
    collection.create_vector_index("dense", hannsdb.IVFIndexParam(metric_type="l2", nlist=8))
    collection.drop_vector_index("dense")
    collection.create_scalar_index("session_id")
    collection.drop_scalar_index("session_id")
    collection.destroy()

    assert [call[0] for call in calls] == [
        "insert",
        "upsert",
        "fetch",
        "delete",
        "list_vector_indexes",
        "list_scalar_indexes",
        "flush",
        "create_vector_index",
        "drop_vector_index",
        "create_scalar_index",
        "drop_scalar_index",
        "destroy",
    ]


def test_collection_surface_single_doc_writes_delegate_via_one_element_list(monkeypatch):
    schema = build_schema()
    calls = []

    class FakeCore:
        path = "/tmp/hannsdb"
        collection_name = "docs"

        def insert(self, docs):
            calls.append(("insert", docs))
            return 1

        def upsert(self, docs):
            calls.append(("upsert", docs))
            return 2

    class FakeFactory:
        def build(self):
            return object()

    monkeypatch.setattr(hannsdb.QueryExecutorFactory, "create", lambda schema: FakeFactory())

    collection = hannsdb.Collection._from_core(FakeCore(), schema=schema)
    doc = hannsdb.Doc(
        id="1",
        vectors={"dense": [1.0, 2.0]},
        fields={"session_id": "abc"},
    )

    assert collection.insert(doc) == 1
    assert collection.upsert(doc) == 2

    assert len(calls) == 2
    for name, docs in calls:
        assert isinstance(docs, list)
        assert len(docs) == 1
        assert docs[0].__class__ is hannsdb._native.Doc
        assert docs[0].id == "1"
        assert docs[0].fields == {"session_id": "abc"}
        assert docs[0].vectors == {"dense": [1.0, 2.0]}


def test_collection_surface_single_id_helpers_delegate_via_list(monkeypatch):
    schema = build_schema()
    calls = []

    class FakeCore:
        path = "/tmp/hannsdb"
        collection_name = "docs"

        def fetch(self, ids):
            calls.append(("fetch", ids))
            if ids == ["1"]:
                return [
                    hannsdb._native.Doc(
                        id="1",
                        fields={"session_id": "abc"},
                        vectors={"dense": [1.0, 2.0]},
                        field_name="dense",
                    )
                ]
            return []

        def delete(self, ids):
            calls.append(("delete", ids))
            return len(ids)

    class FakeFactory:
        def build(self):
            return object()

    monkeypatch.setattr(hannsdb.QueryExecutorFactory, "create", lambda schema: FakeFactory())

    collection = hannsdb.Collection._from_core(FakeCore(), schema=schema)

    fetched = collection.fetch("1")
    assert fetched is not None
    assert fetched.id == "1"
    assert fetched.vector("dense") == [1.0, 2.0]
    assert collection.fetch("missing") is None
    assert collection.delete("1") == 1
    assert calls == [
        ("fetch", ["1"]),
        ("fetch", ["missing"]),
        ("delete", ["1"]),
    ]


def test_collection_create_index_routes_by_schema_and_rejects_scalar_params(monkeypatch):
    schema = build_schema()
    calls = []

    class FakeCore:
        path = "/tmp/hannsdb"
        collection_name = "docs"

        def create_vector_index(self, field_name, index_param=None):
            calls.append(("create_vector_index", field_name, index_param))

        def create_scalar_index(self, field_name):
            calls.append(("create_scalar_index", field_name))

    class FakeFactory:
        def build(self):
            return object()

    monkeypatch.setattr(hannsdb.QueryExecutorFactory, "create", lambda schema: FakeFactory())

    collection = hannsdb.Collection._from_core(FakeCore(), schema=schema)

    collection.create_index("dense", hannsdb.IVFIndexParam(metric_type="l2", nlist=8))
    collection.create_index("session_id")

    assert calls[0][0] == "create_vector_index"
    assert calls[0][1] == "dense"
    assert calls[0][2].__class__ is hannsdb._native.IVFIndexParam
    assert calls[1] == ("create_scalar_index", "session_id")

    with pytest.raises(NotImplementedError, match="scalar index params are not supported"):
        collection.create_index("session_id", hannsdb.OptimizeOption())


def test_collection_create_and_drop_index_reject_unknown_fields(monkeypatch):
    schema = build_schema()

    class FakeCore:
        path = "/tmp/hannsdb"
        collection_name = "docs"

        def create_vector_index(self, field_name, index_param=None):
            raise AssertionError("should not be called")

        def create_scalar_index(self, field_name):
            raise AssertionError("should not be called")

        def drop_vector_index(self, field_name):
            raise AssertionError("should not be called")

        def drop_scalar_index(self, field_name):
            raise AssertionError("should not be called")

    class FakeFactory:
        def build(self):
            return object()

    monkeypatch.setattr(hannsdb.QueryExecutorFactory, "create", lambda schema: FakeFactory())

    collection = hannsdb.Collection._from_core(FakeCore(), schema=schema)

    with pytest.raises(KeyError, match="missing"):
        collection.create_index("missing")

    with pytest.raises(KeyError, match="missing"):
        collection.drop_index("missing")


def test_collection_drop_index_routes_by_schema(monkeypatch):
    schema = build_schema()
    calls = []

    class FakeCore:
        path = "/tmp/hannsdb"
        collection_name = "docs"

        def drop_vector_index(self, field_name):
            calls.append(("drop_vector_index", field_name))

        def drop_scalar_index(self, field_name):
            calls.append(("drop_scalar_index", field_name))

    class FakeFactory:
        def build(self):
            return object()

    monkeypatch.setattr(hannsdb.QueryExecutorFactory, "create", lambda schema: FakeFactory())

    collection = hannsdb.Collection._from_core(FakeCore(), schema=schema)
    collection.drop_index("dense")
    collection.drop_index("session_id")

    assert calls == [
        ("drop_vector_index", "dense"),
        ("drop_scalar_index", "session_id"),
    ]


def test_collection_optimize_explicitly_delegates(monkeypatch):
    schema = build_schema()
    calls = []

    class FakeCore:
        path = "/tmp/hannsdb"
        collection_name = "docs"

        def optimize(self, option=None):
            calls.append(option)

    class FakeFactory:
        def build(self):
            return object()

    monkeypatch.setattr(hannsdb.QueryExecutorFactory, "create", lambda schema: FakeFactory())

    collection = hannsdb.Collection._from_core(FakeCore(), schema=schema)
    collection.optimize()
    collection.optimize(hannsdb.OptimizeOption())

    assert calls[0] is None
    assert calls[1].__class__ is hannsdb._native.OptimizeOption


def test_ddl_surface_keeps_base_schema_but_indexes_are_visible_and_survive_reopen(tmp_path):
    collection = hannsdb.create_and_open(str(tmp_path), build_schema())
    schema_before = collection.schema

    collection.create_vector_index(
        "title",
        hannsdb.IVFIndexParam(metric_type="l2", nlist=8),
    )
    collection.create_scalar_index("session_id")

    assert collection.schema is schema_before
    assert [vector.name for vector in collection.schema.vectors] == ["title", "dense"]
    assert collection.list_vector_indexes() == ["title"]
    assert collection.list_scalar_indexes() == ["session_id"]

    reopened = hannsdb.open(str(tmp_path))

    assert [vector.name for vector in reopened.schema.vectors] == ["title", "dense"]
    assert reopened.list_vector_indexes() == ["title"]
    assert reopened.list_scalar_indexes() == ["session_id"]

    collection.destroy()


def test_collection_convenience_create_and_drop_index_persist_through_reopen(tmp_path):
    schema = build_schema()
    collection = hannsdb.create_and_open(str(tmp_path), schema)

    collection.create_index("dense", hannsdb.HnswIndexParam(metric_type="cosine", m=32))
    collection.create_index("session_id")

    reopened = hannsdb.open(str(tmp_path))
    assert reopened.list_vector_indexes() == ["dense"]
    assert reopened.list_scalar_indexes() == ["session_id"]

    reopened.drop_index("dense")
    reopened.drop_index("session_id")

    reopened_again = hannsdb.open(str(tmp_path))
    assert reopened_again.list_vector_indexes() == []
    assert reopened_again.list_scalar_indexes() == []

    collection.destroy()


def test_collection_convenience_index_operations_reject_ambiguous_field_names(tmp_path):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="shared",
        fields=[hannsdb.FieldSchema(name="shared", data_type="string")],
        vectors=[
            hannsdb.VectorSchema(
                name="shared",
                data_type="vector_fp32",
                dimension=2,
            )
        ],
    )
    collection = hannsdb.create_and_open(str(tmp_path), schema)

    with pytest.raises(ValueError, match="ambiguous field name.*shared"):
        collection.create_index("shared")

    with pytest.raises(ValueError, match="ambiguous field name.*shared"):
        collection.drop_index("shared")

    collection.destroy()
