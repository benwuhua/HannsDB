import json
import dataclasses
from types import SimpleNamespace
from pathlib import Path

import hannsdb
import pytest


def test_schema_types_are_pure_python_wrappers():
    assert hannsdb.CollectionSchema.__module__ == "hannsdb.model.schema.collection_schema"
    assert hannsdb.FieldSchema.__module__ == "hannsdb.model.schema.field_schema"
    assert hannsdb.VectorSchema.__module__ == "hannsdb.model.schema.field_schema"
    assert hannsdb.CollectionOption.__module__ == "hannsdb.model.param.collection_option"
    assert hannsdb.OptimizeOption.__module__ == "hannsdb.model.param.optimize_option"
    assert hannsdb.HnswIndexParam.__module__ == "hannsdb.model.param.index_params"
    assert hannsdb.IVFIndexParam.__module__ == "hannsdb.model.param.index_params"
    assert hannsdb.HnswQueryParam.__module__ == "hannsdb.model.param.index_params"


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
            quantize_type=hannsdb.QuantizeType.Fp16,
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


class ConstantReprIndexParam:
    def __repr__(self):
        return "<constant-repr-index-param>"


def test_collection_schema_accepts_old_positional_vector_schema():
    title = hannsdb.VectorSchema(
        name="title",
        data_type="vector_fp32",
        dimension=384,
    )

    schema = hannsdb.CollectionSchema("docs", title)

    assert schema.primary_vector == "title"
    assert [vector.name for vector in schema.vectors] == ["title"]


def test_schema_helpers_can_find_fields_and_vectors():
    schema = build_schema()

    assert schema.field("session_id").name == "session_id"
    assert schema.vector("dense").dimension == 384


def test_vector_query_is_a_pure_python_dataclass_and_flattens_numpy_arrays():
    np = pytest.importorskip("numpy")

    assert dataclasses.is_dataclass(hannsdb.VectorQuery)

    query = hannsdb.VectorQuery(
        field_name="dense",
        vector=np.array([[1, 2], [3, 4]], dtype=np.float32),
        param=None,
    )

    assert query.field_name == "dense"
    assert query.vector == [1.0, 2.0, 3.0, 4.0]


def test_param_wrappers_bridge_to_native_classes():
    option = hannsdb.CollectionOption(read_only=True, enable_mmap=False)
    optimize = hannsdb.OptimizeOption()
    hnsw = hannsdb.HnswIndexParam(
        metric_type="cosine",
        m=32,
        ef_construction=128,
        quantize_type="fp16",
    )
    ivf = hannsdb.IVFIndexParam(metric_type="l2", nlist=512)
    query = hannsdb.HnswQueryParam(ef=64, is_using_refiner=True)

    assert option._get_native().__class__ is hannsdb._native.CollectionOption
    assert optimize._get_native().__class__ is hannsdb._native.OptimizeOption
    assert hnsw._get_native().__class__ is hannsdb._native.HnswIndexParam
    assert ivf._get_native().__class__ is hannsdb._native.IVFIndexParam
    assert query._get_native().__class__ is hannsdb._native.HnswQueryParam
    assert option._get_native().read_only is True
    assert option._get_native().enable_mmap is False
    assert hnsw.m == 32
    assert hnsw.ef_construction == 128
    assert ivf.nlist == 512
    assert query.ef == 64
    assert query.is_using_refiner is True


def test_optimize_option_is_exported_from_model_param_and_top_level():
    assert hannsdb.OptimizeOption is hannsdb.model.param.OptimizeOption
    assert hannsdb.OptimizeOption.__module__ == "hannsdb.model.param.optimize_option"


def test_param_wrappers_reject_invalid_boolean_and_integer_inputs():
    with pytest.raises(TypeError, match="read_only"):
        hannsdb.CollectionOption(read_only="False")

    with pytest.raises(TypeError, match="m"):
        hannsdb.HnswIndexParam(metric_type="cosine", m="8")

    with pytest.raises(TypeError, match="ef_construction"):
        hannsdb.HnswIndexParam(metric_type="cosine", ef_construction="8")

    with pytest.raises(TypeError, match="nlist"):
        hannsdb.IVFIndexParam(metric_type="l2", nlist="8")

    with pytest.raises(TypeError, match="ef"):
        hannsdb.HnswQueryParam(ef="8")


@pytest.mark.parametrize("bad_metric_type", ["bogus", 1, object()])
def test_index_param_wrappers_reject_invalid_metric_types(bad_metric_type):
    expected_error = ValueError if isinstance(bad_metric_type, str) else TypeError

    with pytest.raises(expected_error):
        hannsdb.HnswIndexParam(metric_type=bad_metric_type)

    with pytest.raises(expected_error):
        hannsdb.IVFIndexParam(metric_type=bad_metric_type)


@pytest.mark.parametrize("bad_quantize_type", ["bogus", 1, object()])
def test_hnsw_index_param_rejects_invalid_quantize_types(bad_quantize_type):
    expected_error = ValueError if isinstance(bad_quantize_type, str) else TypeError

    with pytest.raises(expected_error):
        hannsdb.HnswIndexParam(quantize_type=bad_quantize_type)


@pytest.mark.parametrize("value", [1.23, {1, 2}, frozenset({1, 2})])
def test_vector_query_rejects_scalar_and_set_like_inputs(value):
    with pytest.raises(TypeError, match="vector must be"):
        hannsdb.VectorQuery(field_name="dense", vector=value, param=None)


def test_create_and_open_accepts_primary_ivf_index(tmp_path):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="title",
        fields=[],
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

    collection = hannsdb.create_and_open(str(tmp_path), schema)
    metadata = json.loads((tmp_path / "collections" / "docs" / "collection.json").read_text())

    assert collection.collection_name == "docs"
    assert metadata["primary_vector"] == "title"
    assert metadata["vectors"][0]["index_param"] == {
        "kind": "ivf",
        "metric": "l2",
        "nlist": 1024,
    }

    collection.destroy()


def test_create_and_open_accepts_pure_python_schema_and_reopens_as_wrappers(tmp_path):
    schema = build_schema()
    collection = hannsdb.create_and_open(str(tmp_path), schema)

    assert collection.schema is schema
    assert isinstance(collection.schema, hannsdb.CollectionSchema)
    assert isinstance(collection.schema.field("session_id"), hannsdb.FieldSchema)
    assert isinstance(collection.schema.vector("dense"), hannsdb.VectorSchema)
    assert collection.schema._get_native().__class__ == hannsdb._native.CollectionSchema

    reopened = hannsdb.open(str(tmp_path))
    assert isinstance(reopened.schema, hannsdb.CollectionSchema)
    assert isinstance(reopened.schema.field("tags"), hannsdb.FieldSchema)
    assert isinstance(reopened.schema.vector("title"), hannsdb.VectorSchema)

    collection.destroy()


def test_vector_and_collection_schema_do_not_use_index_param_repr_for_equality_or_hash():
    left = hannsdb.VectorSchema(
        name="dense",
        data_type="vector_fp32",
        dimension=384,
        index_param=ConstantReprIndexParam(),
    )
    right = hannsdb.VectorSchema(
        name="dense",
        data_type="vector_fp32",
        dimension=384,
        index_param=ConstantReprIndexParam(),
    )

    assert left != right
    assert len({left, right}) == 2

    left_collection = hannsdb.CollectionSchema("docs", vectors=[left])
    right_collection = hannsdb.CollectionSchema("docs", vectors=[right])

    assert left_collection != right_collection
    assert len({left_collection, right_collection}) == 2


def test_create_and_open_accepts_legacy_native_schema_input(tmp_path):
    schema = hannsdb._native.CollectionSchema(
        name="docs",
        fields=[],
        vectors=[
            hannsdb._native.VectorSchema(
                name="dense",
                data_type="vector_fp32",
                dimension=384,
            )
        ],
    )

    collection = hannsdb.create_and_open(str(tmp_path), schema)

    assert isinstance(collection.schema, hannsdb.CollectionSchema)
    assert collection.schema.name == "docs"
    assert collection.schema.vector("dense").dimension == 384

    collection.destroy()


def test_collection_schema_coercion_accepts_legacy_vector_schema_only_shape():
    legacy = SimpleNamespace(
        name="docs",
        primary_vector=None,
        vector_schema=hannsdb.VectorSchema(
            name="dense",
            data_type="vector_fp32",
            dimension=384,
        ),
        fields=[],
    )

    schema = hannsdb.model.schema.collection_schema._coerce_collection_schema(legacy)

    assert schema.name == "docs"
    assert [vector.name for vector in schema.vectors] == ["dense"]
    assert schema.primary_vector == "dense"


def test_open_recovers_legacy_dimension_style_collection_json(tmp_path):
    collection = hannsdb.create_and_open(str(tmp_path), build_schema())

    legacy_metadata = {
        "format_version": 1,
        "name": "docs",
        "primary_vector": "dense",
        "fields": [
            {
                "name": "session_id",
                "data_type": "String",
                "nullable": False,
                "array": False,
            }
        ],
        "dimension": 384,
        "metric": "cosine",
        "hnsw_m": 32,
        "hnsw_ef_construction": 128,
    }
    (
        tmp_path
        / "collections"
        / "docs"
        / "collection.json"
    ).write_text(json.dumps(legacy_metadata))

    reopened = hannsdb.open(str(tmp_path))

    assert reopened.schema.name == "docs"
    assert reopened.schema.primary_vector == "dense"
    assert [field.name for field in reopened.schema.fields] == ["session_id"]
    assert reopened.schema.vector("dense").dimension == 384
    assert reopened.schema.vector("dense").index_param.quantize_type == "undefined"

    collection.destroy()


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
        "quantize_type": "fp16",
    }

    reopened = hannsdb.open(str(tmp_path))
    assert reopened.schema.vector("dense").index_param.quantize_type == "fp16"

    collection.destroy()


def test_collection_index_ddl_surface_persists_and_lists_indexes(tmp_path):
    collection = hannsdb.create_and_open(str(tmp_path), build_schema())

    hnsw_blob = (
        Path(collection.path)
        / "collections"
        / collection.collection_name
        / "hnsw_index.bin"
    )
    hnsw_blob.write_bytes(b"stale graph")

    collection.create_vector_index(
        "title",
        hannsdb.IVFIndexParam(metric_type="l2", nlist=8),
    )
    collection.create_scalar_index("session_id")

    assert collection.list_vector_indexes() == ["title"]
    assert collection.list_scalar_indexes() == ["session_id"]

    indexes_path = (
        Path(collection.path)
        / "collections"
        / collection.collection_name
        / "indexes.json"
    )
    catalog = json.loads(indexes_path.read_text())
    assert catalog["vector_indexes"][0]["field_name"] == "title"
    assert catalog["vector_indexes"][0]["kind"] == "ivf"
    assert catalog["vector_indexes"][0]["metric"] == "l2"
    assert catalog["vector_indexes"][0]["params"] == {"nlist": 8}
    assert catalog["scalar_indexes"][0]["field_name"] == "session_id"
    assert catalog["scalar_indexes"][0]["kind"] == "inverted"
    assert not hnsw_blob.exists()

    collection.drop_vector_index("title")
    collection.drop_scalar_index("session_id")

    assert collection.list_vector_indexes() == []
    assert collection.list_scalar_indexes() == []
    assert not hnsw_blob.exists()

    collection.destroy()


def test_collection_index_ddl_surface_runs_core_validation(tmp_path):
    collection = hannsdb.create_and_open(str(tmp_path), build_schema())

    with pytest.raises(ValueError):
        collection.create_vector_index(
            "missing_vector",
            hannsdb.IVFIndexParam(metric_type="l2", nlist=8),
        )

    collection.destroy()


# ---------------------------------------------------------------------------
# Create / Open input validation
# ---------------------------------------------------------------------------


def test_create_rejects_zero_dimension_vector(tmp_path):
    """VectorSchema with dimension=0 (non-sparse) raises ValueError."""
    with pytest.raises(ValueError):
        hannsdb.create_and_open(
            str(tmp_path / "zero_dim"),
            hannsdb.CollectionSchema(
                name="zero_dim",
                fields=[],
                vectors=[hannsdb.VectorSchema("v", hannsdb.DataType.VECTOR_FP32, dimension=0)],
            ),
        )


def test_create_rejects_no_vectors(tmp_path):
    """CollectionSchema with empty vectors list raises ValueError."""
    with pytest.raises(ValueError):
        hannsdb.create_and_open(
            str(tmp_path / "no_vec"),
            hannsdb.CollectionSchema(
                name="no_vec",
                fields=[hannsdb.FieldSchema("x", hannsdb.DataType.STRING)],
                vectors=[],
            ),
        )


def test_create_accepts_empty_name(tmp_path):
    """Empty collection name is currently accepted (documented behavior)."""
    col = hannsdb.create_and_open(
        str(tmp_path / "empty_name"),
        hannsdb.CollectionSchema(
            name="",
            fields=[hannsdb.FieldSchema("x", hannsdb.DataType.STRING)],
            vectors=[hannsdb.VectorSchema("v", hannsdb.DataType.VECTOR_FP32, dimension=2)],
        ),
    )
    assert col is not None
    col.destroy()


def test_create_accepts_duplicate_field_names(tmp_path):
    """Duplicate field names are currently accepted (documented behavior)."""
    col = hannsdb.create_and_open(
        str(tmp_path / "dup_fields"),
        hannsdb.CollectionSchema(
            name="dup",
            fields=[
                hannsdb.FieldSchema("x", hannsdb.DataType.STRING),
                hannsdb.FieldSchema("x", hannsdb.DataType.INT64),
            ],
            vectors=[hannsdb.VectorSchema("v", hannsdb.DataType.VECTOR_FP32, dimension=2)],
        ),
    )
    assert col is not None
    col.destroy()


def test_create_accepts_special_chars_in_field_name(tmp_path):
    """Field names with special chars are currently accepted (documented behavior)."""
    col = hannsdb.create_and_open(
        str(tmp_path / "special_field"),
        hannsdb.CollectionSchema(
            name="special",
            fields=[hannsdb.FieldSchema("a/b", hannsdb.DataType.STRING)],
            vectors=[hannsdb.VectorSchema("v", hannsdb.DataType.VECTOR_FP32, dimension=2)],
        ),
    )
    assert col is not None
    col.destroy()


def test_create_accepts_special_chars_in_collection_name(tmp_path):
    """Collection name with special chars is currently accepted (documented behavior)."""
    col = hannsdb.create_and_open(
        str(tmp_path / "special_coll"),
        hannsdb.CollectionSchema(
            name="a/b",
            fields=[hannsdb.FieldSchema("x", hannsdb.DataType.STRING)],
            vectors=[hannsdb.VectorSchema("v", hannsdb.DataType.VECTOR_FP32, dimension=2)],
        ),
    )
    assert col is not None
    col.destroy()
