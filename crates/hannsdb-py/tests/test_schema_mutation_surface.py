import hannsdb
import pytest


def add_option(concurrency=1):
    return hannsdb.AddColumnOption(concurrency=concurrency)


def alter_option(concurrency=1):
    return hannsdb.AlterColumnOption(concurrency=concurrency)


def build_schema():
    return hannsdb.CollectionSchema(
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


def create_collection(tmp_path):
    return hannsdb.create_and_open(str(tmp_path), build_schema())


def insert_seed_doc(collection):
    collection.insert(
        [
            hannsdb.Doc(
                id="doc-1",
                vectors={"dense": [0.0, 0.0]},
                fields={"session_id": "s-1"},
            )
        ]
    )


def create_populated_collection(tmp_path):
    collection = create_collection(tmp_path)
    insert_seed_doc(collection)
    return collection


def create_collection_with_scalar_field(tmp_path, field_name, data_type, value):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[hannsdb.FieldSchema(name=field_name, data_type=data_type)],
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
                id="doc-1",
                vectors={"dense": [0.0, 0.0]},
                fields={field_name: value},
            )
        ]
    )
    return collection


def fetch_only_doc(collection):
    return collection.fetch(["doc-1"])[0]


def test_schema_mutation_options_are_public_from_top_level_and_param_module():
    add_option = hannsdb.AddColumnOption(concurrency=2)
    alter_option = hannsdb.AlterColumnOption(concurrency=3)

    assert hannsdb.AddColumnOption is hannsdb.model.param.AddColumnOption
    assert hannsdb.AlterColumnOption is hannsdb.model.param.AlterColumnOption
    assert add_option.concurrency == 2
    assert alter_option.concurrency == 3


def test_schema_mutation_options_reject_invalid_concurrency():
    with pytest.raises(TypeError, match="concurrency"):
        hannsdb.AddColumnOption(concurrency="2")

    with pytest.raises(ValueError, match="concurrency"):
        hannsdb.AlterColumnOption(concurrency=-1)


def test_real_collection_add_column_accepts_field_schema_contract(tmp_path):
    collection = create_populated_collection(tmp_path)

    collection.add_column(
        hannsdb.FieldSchema(name="group", data_type="int64", nullable=True),
        expression="",
        option=add_option(),
    )

    assert collection.schema.field("group").name == "group"
    assert fetch_only_doc(collection).has_field("group") is False


def test_real_collection_alter_column_accepts_richer_rename_contract(tmp_path):
    collection = hannsdb.create_and_open(str(tmp_path), build_schema())

    collection.add_column(
        hannsdb.FieldSchema(name="group", data_type="int64"),
        option=add_option(),
    )
    collection.alter_column(
        "group",
        new_name="bucket",
        option=alter_option(),
    )

    assert collection.schema.field("bucket").name == "bucket"
    with pytest.raises(KeyError):
        collection.schema.field("group")


def test_real_collection_add_column_rejects_unsupported_expression(tmp_path):
    collection = hannsdb.create_and_open(str(tmp_path), build_schema())

    with pytest.raises(NotImplementedError, match="expression"):
        collection.add_column(
            hannsdb.FieldSchema(name="group", data_type="int64"),
            expression="1e6",
            option=add_option(),
        )


def test_real_collection_add_column_backfills_string_constant(tmp_path):
    collection = create_populated_collection(tmp_path)

    collection.add_column(
        hannsdb.FieldSchema(name="tag", data_type="string", nullable=False),
        expression='"hello"',
        option=add_option(),
    )

    assert fetch_only_doc(collection).field("tag") == "hello"


def test_real_collection_add_column_backfills_empty_string_constant(tmp_path):
    collection = create_populated_collection(tmp_path)

    collection.add_column(
        hannsdb.FieldSchema(name="tag", data_type="string", nullable=False),
        expression='""',
        option=add_option(),
    )

    assert fetch_only_doc(collection).field("tag") == ""


def test_real_collection_add_column_backfills_int_and_bool_constants(tmp_path):
    collection = create_populated_collection(tmp_path)

    collection.add_column(
        hannsdb.FieldSchema(name="turn", data_type="int64", nullable=False),
        expression="7",
        option=add_option(),
    )
    collection.add_column(
        hannsdb.FieldSchema(name="active", data_type="bool", nullable=False),
        expression="true",
        option=add_option(),
    )

    fetched = fetch_only_doc(collection)
    assert fetched.field("turn") == 7
    assert fetched.field("active") is True


def test_real_collection_add_column_allows_null_constant_only_for_nullable_field(tmp_path):
    collection = create_populated_collection(tmp_path)

    collection.add_column(
        hannsdb.FieldSchema(name="maybe_tag", data_type="string", nullable=True),
        expression="null",
        option=add_option(),
    )
    assert fetch_only_doc(collection).has_field("maybe_tag") is False

    with pytest.raises(ValueError, match="nullable"):
        collection.add_column(
            hannsdb.FieldSchema(name="tag", data_type="string", nullable=False),
            expression="null",
            option=add_option(),
        )


@pytest.mark.parametrize(
    ("expression", "error_type", "message"),
    [
        ("group", NotImplementedError, "expression"),
        ("a + b", NotImplementedError, "expression"),
        ("foo()", NotImplementedError, "expression"),
        ("+1", NotImplementedError, "expression"),
        ("1e6", NotImplementedError, "expression"),
        ('"unterminated', ValueError, "string literal"),
    ],
)
def test_real_collection_add_column_rejects_unsupported_constant_expression_shapes(
    tmp_path, expression, error_type, message
):
    collection = create_collection(tmp_path)

    with pytest.raises(error_type, match=message):
        collection.add_column(
            hannsdb.FieldSchema(name="tag", data_type="string", nullable=True),
            expression=expression,
            option=add_option(),
        )


@pytest.mark.parametrize(
    ("field_name", "data_type", "expression"),
    [
        ("bad_int", "int64", "01"),
        ("bad_float", "float64", "00.1"),
        ("bad_float_short", "float64", "-.5"),
    ],
)
def test_native_add_column_rejects_non_canonical_numeric_literals(
    tmp_path, field_name, data_type, expression
):
    collection = create_collection(tmp_path)

    with pytest.raises(NotImplementedError, match="constant"):
        collection._core.add_column(
            field_name,
            data_type,
            False,
            False,
            expression,
            hannsdb._native.AddColumnOption(0),
        )


def test_real_collection_add_column_rejects_vector_field_schema(tmp_path):
    collection = create_collection(tmp_path)

    with pytest.raises(NotImplementedError, match="vector"):
        collection.add_column(
            hannsdb.VectorSchema(name="aux", data_type="vector_fp32", dimension=2),
            expression="",
            option=add_option(),
        )


def test_real_collection_alter_column_rejects_field_schema_migration_request(tmp_path):
    collection = create_collection(tmp_path)

    with pytest.raises(NotImplementedError, match="field_schema"):
        collection.alter_column(
            "session_id",
            field_schema=hannsdb.FieldSchema(
                name="session_id",
                data_type="string",
                nullable=True,
            ),
            option=alter_option(),
        )

@pytest.mark.parametrize(
    ("field_name", "source_type", "source_value", "target_type"),
    [
        ("score", "int32", 7, "int64"),
        ("count", "uint32", 7, "uint64"),
        ("score", "float", 1.5, "float64"),
    ],
)
def test_real_collection_alter_column_widens_supported_scalar_types(
    tmp_path, field_name, source_type, source_value, target_type
):
    collection = create_collection_with_scalar_field(
        tmp_path, field_name, source_type, source_value
    )

    collection.alter_column(
        field_name,
        field_schema=hannsdb.FieldSchema(name=field_name, data_type=target_type),
        option=alter_option(),
    )

    assert collection.schema.field(field_name).data_type == target_type
    assert fetch_only_doc(collection).field(field_name) == source_value


@pytest.mark.parametrize(
    ("field_name", "source_type", "source_value", "target_field", "message"),
    [
        (
            "score",
            "int64",
            7,
            hannsdb.FieldSchema(name="score", data_type="int32"),
            "widening",
        ),
        (
            "score",
            "string",
            "7",
            hannsdb.FieldSchema(name="score", data_type="int64"),
            "widening",
        ),
        (
            "score",
            "float64",
            1.5,
            hannsdb.FieldSchema(name="score", data_type="int64"),
            "widening",
        ),
        (
            "score",
            "int32",
            7,
            hannsdb.FieldSchema(name="renamed", data_type="int32"),
            "widening",
        ),
        (
            "score",
            "int32",
            7,
            hannsdb.FieldSchema(name="score", data_type="int64", nullable=True),
            "nullable",
        ),
        (
            "score",
            "int32",
            7,
            hannsdb.FieldSchema(name="score", data_type="int64", array=True),
            "array",
        ),
    ],
)
def test_real_collection_alter_column_rejects_unsupported_migration_shapes(
    tmp_path, field_name, source_type, source_value, target_field, message
):
    collection = create_collection_with_scalar_field(tmp_path, field_name, source_type, source_value)

    with pytest.raises(NotImplementedError, match=message):
        collection.alter_column(
            field_name,
            field_schema=target_field,
            option=alter_option(),
        )


def test_real_collection_alter_column_rejects_indexed_field_migration(tmp_path):
    collection = create_collection_with_scalar_field(tmp_path, "count", "uint32", 7)
    collection.create_scalar_index("count")

    with pytest.raises(NotImplementedError, match="scalar index"):
        collection.alter_column(
            "count",
            field_schema=hannsdb.FieldSchema(name="count", data_type="uint64"),
            option=alter_option(),
        )


@pytest.mark.parametrize(
    ("old_name", "new_name", "source_type", "source_value", "target_type"),
    [
        ("score", "total_score", "int32", 7, "int64"),
        ("count", "doc_count", "uint32", 7, "uint64"),
        ("ratio", "ratio64", "float", 1.5, "float64"),
    ],
)
def test_real_collection_alter_column_renames_and_widens_supported_scalar_types(
    tmp_path, old_name, new_name, source_type, source_value, target_type
):
    collection = create_collection_with_scalar_field(
        tmp_path, old_name, source_type, source_value
    )

    collection.alter_column(
        old_name,
        new_name=new_name,
        field_schema=hannsdb.FieldSchema(name=new_name, data_type=target_type),
        option=alter_option(),
    )

    assert collection.schema.field(new_name).data_type == target_type
    with pytest.raises(KeyError):
        collection.schema.field(old_name)
    fetched = fetch_only_doc(collection)
    with pytest.raises(KeyError):
        fetched.field(old_name)
    assert fetched.field(new_name) == source_value


def test_real_collection_alter_column_rejects_rename_migration_name_mismatch(tmp_path):
    collection = create_collection_with_scalar_field(tmp_path, "score", "int32", 7)

    with pytest.raises(ValueError, match="new_name"):
        collection.alter_column(
            "score",
            new_name="total_score",
            field_schema=hannsdb.FieldSchema(name="final_score", data_type="int64"),
            option=alter_option(),
        )


def test_real_collection_alter_column_rejects_rename_migration_target_name_conflict(tmp_path):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[
            hannsdb.FieldSchema(name="score", data_type="int32"),
            hannsdb.FieldSchema(name="total_score", data_type="int64"),
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

    with pytest.raises(ValueError, match="already exists"):
        collection.alter_column(
            "score",
            new_name="total_score",
            field_schema=hannsdb.FieldSchema(name="total_score", data_type="int64"),
            option=alter_option(),
        )


def test_real_collection_alter_column_rejects_indexed_field_rename_migration(tmp_path):
    collection = create_collection_with_scalar_field(tmp_path, "count", "uint32", 7)
    collection.create_scalar_index("count")

    with pytest.raises(NotImplementedError, match="scalar index"):
        collection.alter_column(
            "count",
            new_name="doc_count",
            field_schema=hannsdb.FieldSchema(name="doc_count", data_type="uint64"),
            option=alter_option(),
        )


def test_collection_schema_accepts_single_field_and_vector():
    """CollectionSchema can take a single FieldSchema/VectorSchema (not just lists)."""
    schema = hannsdb.CollectionSchema(
        name="test",
        fields=hannsdb.FieldSchema("id", "int64", index_param=hannsdb.InvertIndexParam(), nullable=False),
        vectors=hannsdb.VectorSchema("vec", "vector_fp32", dimension=4),
    )
    assert len(schema.fields) == 1
    assert len(schema.vectors) == 1
    f = schema.field("id")
    assert f.index_param is not None
    assert f.index_param.type == hannsdb.IndexType.INVERT


def test_create_scalar_index_updates_schema_index_param(tmp_path):
    col = create_collection(tmp_path)
    col.add_column(hannsdb.FieldSchema("weight", "float64", nullable=True), add_option())
    assert col.schema.field("weight").index_param is None
    col.create_scalar_index("weight", hannsdb.InvertIndexParam())
    f = col.schema.field("weight")
    assert f.index_param is not None
    assert f.index_param.type == hannsdb.IndexType.INVERT
    assert f.index_param.enable_range_optimization is False


def test_drop_scalar_index_clears_schema_index_param(tmp_path):
    col = create_collection(tmp_path)
    col.add_column(hannsdb.FieldSchema("weight", "float64", nullable=True), add_option())
    col.create_scalar_index("weight", hannsdb.InvertIndexParam())
    assert col.schema.field("weight").index_param is not None
    col.drop_scalar_index("weight")
    assert col.schema.field("weight").index_param is None
