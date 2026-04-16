"""Doc field-level access and dtype round-trip tests.

Verifies that scalar field values survive an insert->fetch round-trip
with correct dtype semantics.
"""

import math

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
# Helpers
# ---------------------------------------------------------------------------

def _make_col(tmp_path, fields, nullable=False):
    """Create a collection with given fields (all nullable by default)."""
    if nullable:
        fields = [FieldSchema(f.name, f.data_type, nullable=True) for f in fields]
    schema = CollectionSchema(
        name="dtype_test",
        fields=fields,
        vectors=[VectorSchema("vec", DataType.VECTOR_FP32, dimension=3)],
    )
    return hannsdb.create_and_open(str(tmp_path / "db"), schema)


def _insert_and_fetch(col, doc):
    col.insert(doc)
    return col.fetch(doc.id)


# ---------------------------------------------------------------------------
# Scalar dtype round-trip
# ---------------------------------------------------------------------------


class TestScalarDtypeRoundTrip:
    VEC = {"vec": [0.1, 0.2, 0.3]}

    def test_string_field(self, tmp_path):
        col = _make_col(tmp_path, [FieldSchema("s", DataType.STRING)])
        got = _insert_and_fetch(col, Doc(id="1", fields={"s": "Tom"}, vectors=self.VEC))
        assert got.field("s") == "Tom"

    def test_bool_field(self, tmp_path):
        col = _make_col(tmp_path, [FieldSchema("b", DataType.BOOL)])
        got = _insert_and_fetch(col, Doc(id="1", fields={"b": True}, vectors=self.VEC))
        assert got.field("b") is True

    def test_int32_field(self, tmp_path):
        col = _make_col(tmp_path, [FieldSchema("i32", DataType.INT32)])
        got = _insert_and_fetch(col, Doc(id="1", fields={"i32": 19}, vectors=self.VEC))
        assert got.field("i32") == 19

    def test_int64_field(self, tmp_path):
        col = _make_col(tmp_path, [FieldSchema("i64", DataType.INT64)])
        got = _insert_and_fetch(col, Doc(id="1", fields={"i64": 1111111111111111111}, vectors=self.VEC))
        assert got.field("i64") == 1111111111111111111

    def test_float_field(self, tmp_path):
        col = _make_col(tmp_path, [FieldSchema("f", DataType.FLOAT)])
        got = _insert_and_fetch(col, Doc(id="1", fields={"f": 60.5}, vectors=self.VEC))
        assert math.isclose(got.field("f"), 60.5, rel_tol=1e-6)

    def test_float64_field(self, tmp_path):
        # float64 stored as f32 internally — use loose tolerance
        col = _make_col(tmp_path, [FieldSchema("f64", DataType.FLOAT64)])
        got = _insert_and_fetch(col, Doc(id="1", fields={"f64": 1.77777777777}, vectors=self.VEC))
        assert math.isclose(got.field("f64"), 1.77777777777, rel_tol=1e-2)

    def test_uint32_field(self, tmp_path):
        col = _make_col(tmp_path, [FieldSchema("u32", DataType.UINT32)])
        got = _insert_and_fetch(col, Doc(id="1", fields={"u32": 4294967295}, vectors=self.VEC))
        assert got.field("u32") == 4294967295

    def test_uint64_field(self, tmp_path):
        col = _make_col(tmp_path, [FieldSchema("u64", DataType.UINT64)])
        got = _insert_and_fetch(col, Doc(id="1", fields={"u64": 18446744073709551615}, vectors=self.VEC))
        assert got.field("u64") == 18446744073709551615


# ---------------------------------------------------------------------------
# Multiple fields in one doc
# ---------------------------------------------------------------------------


class TestMultiFieldRoundTrip:
    def test_all_scalar_fields_at_once(self, tmp_path):
        fields = [
            FieldSchema("s", DataType.STRING),
            FieldSchema("b", DataType.BOOL),
            FieldSchema("i32", DataType.INT32),
            FieldSchema("i64", DataType.INT64),
            FieldSchema("f", DataType.FLOAT),
            FieldSchema("f64", DataType.FLOAT64),
            FieldSchema("u32", DataType.UINT32),
            FieldSchema("u64", DataType.UINT64),
        ]
        col = _make_col(tmp_path, fields)
        doc = Doc(
            id="1",
            fields={
                "s": "Tom",
                "b": True,
                "i32": 19,
                "i64": 1111111111111111111,
                "f": 60.5,
                "f64": 1.77777777777,
                "u32": 4294967295,
                "u64": 18446744073709551615,
            },
            vectors={"vec": [0.1, 0.2, 0.3]},
        )
        got = _insert_and_fetch(col, doc)

        assert got.field("s") == "Tom"
        assert got.field("b") is True
        assert got.field("i32") == 19
        assert got.field("i64") == 1111111111111111111
        assert math.isclose(got.field("f"), 60.5, rel_tol=1e-6)
        assert math.isclose(got.field("f64"), 1.77777777777, rel_tol=1e-2)
        assert got.field("u32") == 4294967295
        assert got.field("u64") == 18446744073709551615


# ---------------------------------------------------------------------------
# Vector access
# ---------------------------------------------------------------------------


class TestVectorAccess:
    def test_vector_roundtrip(self, tmp_path):
        col = _make_col(tmp_path, [])
        got = _insert_and_fetch(col, Doc(id="1", vectors={"vec": [1.0, 2.0, 3.0]}))
        vec = got.vector("vec")
        assert len(vec) == 3
        assert all(math.isclose(vec[i], float(i + 1), rel_tol=1e-6) for i in range(3))

    def test_vector_names(self, tmp_path):
        col = _make_col(tmp_path, [])
        got = _insert_and_fetch(col, Doc(id="1", vectors={"vec": [1.0, 2.0, 3.0]}))
        assert "vec" in got.vector_names()


# ---------------------------------------------------------------------------
# has_field / has_vector
# ---------------------------------------------------------------------------


class TestHasFieldHasVector:
    def test_has_field_true_for_set_field(self, tmp_path):
        col = _make_col(tmp_path, [FieldSchema("s", DataType.STRING)])
        got = _insert_and_fetch(col, Doc(id="1", fields={"s": "hello"}, vectors={"vec": [0.1, 0.2, 0.3]}))
        assert got.has_field("s") is True

    def test_has_field_false_for_unset_field(self, tmp_path):
        col = _make_col(tmp_path, [FieldSchema("s", DataType.STRING), FieldSchema("other", DataType.STRING, nullable=True)])
        got = _insert_and_fetch(col, Doc(id="1", fields={"s": "hello"}, vectors={"vec": [0.1, 0.2, 0.3]}))
        assert got.has_field("other") is False

    def test_has_vector_true(self, tmp_path):
        col = _make_col(tmp_path, [])
        got = _insert_and_fetch(col, Doc(id="1", vectors={"vec": [0.1, 0.2, 0.3]}))
        assert got.has_vector("vec") is True

    def test_has_vector_false_for_missing(self, tmp_path):
        col = _make_col(tmp_path, [])
        got = _insert_and_fetch(col, Doc(id="1", vectors={"vec": [0.1, 0.2, 0.3]}))
        assert got.has_vector("nope") is False

    def test_field_raises_keyerror_for_missing(self, tmp_path):
        col = _make_col(tmp_path, [])
        got = _insert_and_fetch(col, Doc(id="1", vectors={"vec": [0.1, 0.2, 0.3]}))
        with pytest.raises(KeyError):
            got.field("nonexistent")

    def test_vector_raises_keyerror_for_missing(self, tmp_path):
        col = _make_col(tmp_path, [])
        got = _insert_and_fetch(col, Doc(id="1", vectors={"vec": [0.1, 0.2, 0.3]}))
        with pytest.raises(KeyError):
            got.vector("nonexistent")


# ---------------------------------------------------------------------------
# Nullable fields
# ---------------------------------------------------------------------------


class TestNullableFields:
    def test_nullable_field_omitted(self, tmp_path):
        """Omitting a nullable field → has_field returns False."""
        col = _make_col(tmp_path, [FieldSchema("name", DataType.STRING), FieldSchema("opt", DataType.STRING, nullable=True)])
        got = _insert_and_fetch(col, Doc(id="1", fields={"name": "hello"}, vectors={"vec": [0.1, 0.2, 0.3]}))
        assert got.has_field("opt") is False

    def test_nullable_field_with_value(self, tmp_path):
        """Setting a nullable field to a real value → field present."""
        col = _make_col(tmp_path, [FieldSchema("name", DataType.STRING), FieldSchema("opt", DataType.STRING, nullable=True)])
        got = _insert_and_fetch(col, Doc(id="1", fields={"name": "hello", "opt": "world"}, vectors={"vec": [0.1, 0.2, 0.3]}))
        assert got.has_field("opt") is True
        assert got.field("opt") == "world"

    def test_nullable_field_explicit_none_rejected(self, tmp_path):
        """Explicitly passing None for a nullable field raises ValueError."""
        col = _make_col(tmp_path, [FieldSchema("name", DataType.STRING), FieldSchema("opt", DataType.STRING, nullable=True)])
        with pytest.raises(ValueError):
            col.insert(Doc(id="1", fields={"name": "hello", "opt": None}, vectors={"vec": [0.1, 0.2, 0.3]}))

    def test_required_field_missing_rejected(self, tmp_path):
        """Omitting a required (nullable=False) field raises on insert."""
        col = _make_col(tmp_path, [FieldSchema("name", DataType.STRING, nullable=False)], nullable=False)
        with pytest.raises(Exception):
            col.insert(Doc(id="1", vectors={"vec": [0.1, 0.2, 0.3]}))


# ---------------------------------------------------------------------------
# Doc constructor basics (no collection needed)
# ---------------------------------------------------------------------------


class TestDocConstructor:
    def test_default(self):
        doc = Doc(id="1")
        assert doc.id == "1"

    def test_with_fields_and_vectors(self):
        doc = Doc(id="1", fields={"s": "Tom"}, vectors={"dense": [1, 2, 3]})
        assert doc.id == "1"
        assert doc.field("s") == "Tom"
        assert doc.vector("dense") == [1, 2, 3]

    def test_field_names(self):
        doc = Doc(id="1", fields={"a": 1, "b": 2}, vectors={"v": [0.0]})
        names = doc.field_names()
        assert "a" in names
        assert "b" in names

    def test_vector_names(self):
        doc = Doc(id="1", vectors={"v1": [0.0], "v2": [0.0]})
        names = doc.vector_names()
        assert "v1" in names
        assert "v2" in names

    def test_score_property(self):
        doc = Doc(id="1", score=0.95)
        assert math.isclose(doc.score, 0.95, rel_tol=1e-6)

    def test_equality(self):
        a = Doc(id="1", fields={"x": 1}, vectors={"v": [1.0]})
        b = Doc(id="1", fields={"x": 1}, vectors={"v": [1.0]})
        assert a == b


# ---------------------------------------------------------------------------
# Array dtype round-trip
# ---------------------------------------------------------------------------


class TestArrayDtypeRoundTrip:
    VEC = {"vec": [0.1, 0.2, 0.3]}

    def test_array_string(self, tmp_path):
        col = _make_col(tmp_path, [FieldSchema("tags", DataType.STRING, array=True)])
        got = _insert_and_fetch(col, Doc(id="1", fields={"tags": ["a", "b", "c"]}, vectors=self.VEC))
        assert got.field("tags") == ["a", "b", "c"]

    def test_array_int64(self, tmp_path):
        col = _make_col(tmp_path, [FieldSchema("nums", DataType.INT64, array=True)])
        got = _insert_and_fetch(col, Doc(id="1", fields={"nums": [10, 20, 30]}, vectors=self.VEC))
        assert got.field("nums") == [10, 20, 30]

    def test_array_int32(self, tmp_path):
        col = _make_col(tmp_path, [FieldSchema("nums", DataType.INT32, array=True)])
        got = _insert_and_fetch(col, Doc(id="1", fields={"nums": [1, 2, 3]}, vectors=self.VEC))
        assert got.field("nums") == [1, 2, 3]

    def test_array_float(self, tmp_path):
        col = _make_col(tmp_path, [FieldSchema("scores", DataType.FLOAT, array=True)])
        got = _insert_and_fetch(col, Doc(id="1", fields={"scores": [1.0, 2.0, 3.0]}, vectors=self.VEC))
        assert got.field("scores") == [1.0, 2.0, 3.0]

    def test_array_float64(self, tmp_path):
        col = _make_col(tmp_path, [FieldSchema("vals", DataType.FLOAT64, array=True)])
        got = _insert_and_fetch(col, Doc(id="1", fields={"vals": [1.5, 2.5, 3.5]}, vectors=self.VEC))
        assert got.field("vals") == [1.5, 2.5, 3.5]

    def test_array_bool(self, tmp_path):
        col = _make_col(tmp_path, [FieldSchema("flags", DataType.BOOL, array=True)])
        got = _insert_and_fetch(col, Doc(id="1", fields={"flags": [True, False, True]}, vectors=self.VEC))
        assert got.field("flags") == [True, False, True]

    def test_array_mixed_with_scalar_fields(self, tmp_path):
        col = _make_col(tmp_path, [
            FieldSchema("name", DataType.STRING),
            FieldSchema("tags", DataType.STRING, array=True),
            FieldSchema("nums", DataType.INT64, array=True),
        ])
        doc = Doc(id="1", fields={"name": "test", "tags": ["x", "y"], "nums": [1, 2]}, vectors=self.VEC)
        got = _insert_and_fetch(col, doc)
        assert got.field("name") == "test"
        assert got.field("tags") == ["x", "y"]
        assert got.field("nums") == [1, 2]

    def test_array_field_has_field(self, tmp_path):
        col = _make_col(tmp_path, [FieldSchema("tags", DataType.STRING, array=True)])
        got = _insert_and_fetch(col, Doc(id="1", fields={"tags": ["a"]}, vectors=self.VEC))
        assert got.has_field("tags") is True