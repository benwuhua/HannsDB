import hannsdb
import pytest


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
    collection = hannsdb.create_and_open(str(tmp_path), build_schema())

    collection.insert(
        [
            hannsdb.Doc(
                id="doc-1",
                vectors={"dense": [0.0, 0.0]},
                fields={"session_id": "s-1"},
            )
        ]
    )

    collection.add_column(
        hannsdb.FieldSchema(name="group", data_type="int64", nullable=True),
        expression="",
        option=hannsdb.AddColumnOption(concurrency=1),
    )

    assert collection.schema.field("group").name == "group"
    assert collection.fetch(["doc-1"])[0].has_field("group") is False


def test_real_collection_alter_column_accepts_richer_rename_contract(tmp_path):
    collection = hannsdb.create_and_open(str(tmp_path), build_schema())

    collection.add_column(
        hannsdb.FieldSchema(name="group", data_type="int64"),
        option=hannsdb.AddColumnOption(concurrency=1),
    )
    collection.alter_column(
        "group",
        new_name="bucket",
        option=hannsdb.AlterColumnOption(concurrency=1),
    )

    assert collection.schema.field("bucket").name == "bucket"
    with pytest.raises(KeyError):
        collection.schema.field("group")


def test_real_collection_add_column_rejects_non_empty_expression(tmp_path):
    collection = hannsdb.create_and_open(str(tmp_path), build_schema())

    with pytest.raises(NotImplementedError, match="expression"):
        collection.add_column(
            hannsdb.FieldSchema(name="group", data_type="int64"),
            expression="1",
            option=hannsdb.AddColumnOption(concurrency=1),
        )


def test_real_collection_add_column_rejects_vector_field_schema(tmp_path):
    collection = hannsdb.create_and_open(str(tmp_path), build_schema())

    with pytest.raises(NotImplementedError, match="vector"):
        collection.add_column(
            hannsdb.VectorSchema(name="aux", data_type="vector_fp32", dimension=2),
            expression="",
            option=hannsdb.AddColumnOption(concurrency=1),
        )


def test_real_collection_alter_column_rejects_field_schema_migration_request(tmp_path):
    collection = hannsdb.create_and_open(str(tmp_path), build_schema())

    with pytest.raises(NotImplementedError, match="field_schema"):
        collection.alter_column(
            "session_id",
            field_schema=hannsdb.FieldSchema(
                name="session_id",
                data_type="string",
                nullable=True,
            ),
            option=hannsdb.AlterColumnOption(concurrency=1),
        )
