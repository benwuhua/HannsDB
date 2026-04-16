"""Exception handling tests — aligned with zvec test_collection_exception.py.

Verifies that missing required parameters raise TypeError, and that
positive-path lifecycle operations succeed without exceptions.
"""

import tempfile

import hannsdb
from hannsdb import (
    CollectionSchema,
    DataType,
    Doc,
    FieldSchema,
    VectorSchema,
)
import pytest


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def col(tmp_path):
    schema = CollectionSchema(
        name="exc_test",
        fields=[FieldSchema("name", DataType.STRING)],
        vectors=[VectorSchema("dense", DataType.VECTOR_FP32, dimension=4)],
    )
    return hannsdb.create_and_open(str(tmp_path / "db"), schema)


# ---------------------------------------------------------------------------
# Missing required parameter → TypeError
# ---------------------------------------------------------------------------


class TestMissingParameters:
    def test_create_and_open_missing_path(self):
        with pytest.raises(TypeError):
            hannsdb.create_and_open()

    def test_create_and_open_missing_schema(self, tmp_path):
        with pytest.raises(TypeError):
            hannsdb.create_and_open(str(tmp_path / "db"))

    def test_open_missing_path(self):
        with pytest.raises(TypeError):
            hannsdb.open()

    def test_insert_missing_docs(self, col):
        with pytest.raises(TypeError):
            col.insert()

    def test_update_missing_docs(self, col):
        with pytest.raises(TypeError):
            col.update()

    def test_upsert_missing_docs(self, col):
        with pytest.raises(TypeError):
            col.upsert()

    def test_delete_missing_ids(self, col):
        with pytest.raises(TypeError):
            col.delete()

    def test_fetch_missing_ids(self, col):
        with pytest.raises(TypeError):
            col.fetch()

    def test_add_column_missing_field_schema(self, col):
        with pytest.raises(TypeError):
            col.add_column()

    def test_alter_column_missing_old_name(self, col):
        with pytest.raises(TypeError):
            col.alter_column(new_name="x")

    def test_alter_column_missing_new_name(self, col):
        with pytest.raises(TypeError):
            col.alter_column(old_name="x")

    def test_drop_column_missing_field_name(self, col):
        with pytest.raises(TypeError):
            col.drop_column()


# ---------------------------------------------------------------------------
# Empty collection operations — no crash
# ---------------------------------------------------------------------------


class TestEmptyCollectionOperations:
    def test_fetch_on_empty_collection(self, col):
        result = col.fetch(["999"])
        assert isinstance(result, list)

    def test_query_on_empty_collection(self, col):
        from hannsdb import VectorQuery
        vq = VectorQuery(vector=[1.0, 2.0, 3.0, 4.0], field_name="dense")
        result = col.query(vectors=[vq], topk=10)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_delete_on_empty_collection(self, col):
        result = col.delete("999")
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# Resource management — full CRUD lifecycle
# ---------------------------------------------------------------------------


class TestResourceManagement:
    def test_full_crud_lifecycle(self, col):
        doc = Doc(
            id="0",
            fields={"name": "test"},
            vectors={"dense": [1.0, 2.0, 3.0, 4.0]},
        )

        # Insert
        result = col.insert(doc)
        assert result.ok()
        assert result == 1

        # Fetch
        fetched = col.fetch("0")
        assert fetched.id == "0"
        assert fetched.field("name") == "test"

        # Update
        updated = Doc(
            id="0",
            fields={"name": "updated"},
            vectors={"dense": [2.0, 3.0, 4.0, 5.0]},
        )
        result = col.update(updated)
        assert result.ok()

        # Verify update
        fetched = col.fetch("0")
        assert fetched.field("name") == "updated"

        # Delete
        result = col.delete("0")
        assert result.ok()
        assert result == 1

        # Verify deletion
        fetched = col.fetch("0")
        assert fetched is None

    def test_upsert_lifecycle(self, col):
        doc = Doc(
            id="1",
            fields={"name": "first"},
            vectors={"dense": [1.0, 2.0, 3.0, 4.0]},
        )
        # First upsert → insert
        result = col.upsert(doc)
        assert result.ok()

        # Second upsert → overwrite
        doc2 = Doc(
            id="1",
            fields={"name": "second"},
            vectors={"dense": [5.0, 6.0, 7.0, 8.0]},
        )
        result = col.upsert(doc2)
        assert result.ok()

        fetched = col.fetch("1")
        assert fetched.field("name") == "second"
