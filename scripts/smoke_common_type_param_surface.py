from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    python_root = repo_root / "crates" / "hannsdb-py" / "python"
    sys.path.insert(0, str(python_root))

    import hannsdb

    assert hasattr(hannsdb, "InvertIndexParam") is False
    assert hasattr(hannsdb, "HnswRabitqIndexParam") is False
    assert hasattr(hannsdb, "HnswRabitqQueryParam") is False
    assert hasattr(hannsdb.DataType, "VectorFp16") is False

    temp_dir = Path(tempfile.mkdtemp(prefix="hannsdb_common_param_smoke_"))
    try:
        schema = hannsdb.CollectionSchema(
            name="docs",
            primary_vector="dense_ivf",
            fields=[
                hannsdb.FieldSchema(name="rank", data_type=hannsdb.DataType.Int32),
                hannsdb.FieldSchema(name="bucket", data_type=hannsdb.DataType.UInt32),
                hannsdb.FieldSchema(name="weight", data_type=hannsdb.DataType.Float),
            ],
            vectors=[
                hannsdb.VectorSchema(
                    name="dense_ivf",
                    data_type=hannsdb.DataType.VectorFp32,
                    dimension=2,
                    index_param=hannsdb.IVFIndexParam(
                        metric_type=hannsdb.MetricType.L2,
                        nlist=8,
                    ),
                ),
                hannsdb.VectorSchema(
                    name="aux_flat",
                    data_type=hannsdb.DataType.VectorFp32,
                    dimension=2,
                    index_param=hannsdb.FlatIndexParam(
                        metric_type=hannsdb.MetricType.Cosine,
                    ),
                ),
            ],
        )

        collection = hannsdb.create_and_open(str(temp_dir), schema)
        collection_path = collection.path

        assert collection.schema.field("rank").data_type == "int32"
        assert collection.schema.field("bucket").data_type == "uint32"
        assert collection.schema.field("weight").data_type == "float"
        assert collection.schema.vector("aux_flat").index_param.metric_type == "cosine"

        collection.insert(
            [
                hannsdb.Doc(
                    id="doc-1",
                    vectors={"dense_ivf": [0.0, 0.0], "aux_flat": [0.0, 1.0]},
                    fields={"rank": 1, "bucket": 7, "weight": 1.5},
                ),
                hannsdb.Doc(
                    id="doc-2",
                    vectors={"dense_ivf": [0.1, 0.0], "aux_flat": [0.1, 1.0]},
                    fields={"rank": 2, "bucket": 8, "weight": 2.5},
                ),
            ]
        )

        hits = collection.query(
            query_context=hannsdb.QueryContext(
                top_k=2,
                output_fields=["rank", "bucket", "weight"],
                queries=[
                    hannsdb.VectorQuery(
                        field_name="dense_ivf",
                        vector=[0.0, 0.0],
                        param=hannsdb.IVFQueryParam(nprobe=2),
                    )
                ],
            )
        )
        assert len(hits) == 2
        assert {doc.id for doc in hits} == {"doc-1", "doc-2"}

        reopened = hannsdb.open(collection_path)
        assert reopened.schema.field("rank").data_type == "int32"
        assert reopened.schema.field("bucket").data_type == "uint32"
        assert reopened.schema.field("weight").data_type == "float"
        assert {vector.name for vector in reopened.schema.vectors} == {"dense_ivf", "aux_flat"}
        return 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
