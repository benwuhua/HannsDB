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

    assert hannsdb.HnswHvqIndexParam is hannsdb.model.param.HnswHvqIndexParam

    temp_dir = Path(tempfile.mkdtemp(prefix="hannsdb_hnsw_hvq_smoke_"))
    try:
        schema = hannsdb.CollectionSchema(
            name="docs_hvq",
            primary_vector="dense",
            fields=[hannsdb.FieldSchema(name="rank", data_type=hannsdb.DataType.Int32)],
            vectors=[
                hannsdb.VectorSchema(
                    name="dense",
                    data_type=hannsdb.DataType.VectorFp32,
                    dimension=2,
                    index_param=hannsdb.HnswHvqIndexParam(
                        metric_type=hannsdb.MetricType.Ip,
                        m=8,
                        m_max0=16,
                        ef_construction=32,
                        ef_search=32,
                        nbits=4,
                    ),
                )
            ],
        )

        collection = hannsdb.create_and_open(str(temp_dir), schema)
        collection.insert(
            [
                hannsdb.Doc(id="doc-1", fields={"rank": 1}, vectors={"dense": [1.0, 0.0]}),
                hannsdb.Doc(id="doc-2", fields={"rank": 2}, vectors={"dense": [0.9, 0.0]}),
                hannsdb.Doc(id="doc-3", fields={"rank": 3}, vectors={"dense": [0.0, 1.0]}),
            ]
        )
        collection.optimize()

        hits = collection.query(
            query_context=hannsdb.QueryContext(
                top_k=2,
                output_fields=["rank"],
                queries=[hannsdb.VectorQuery(field_name="dense", vector=[1.0, 0.0])],
            )
        )
        assert [doc.id for doc in hits] == ["doc-1", "doc-2"]

        reopened = hannsdb.open(str(temp_dir))
        assert reopened.schema.vector("dense").index_param.__class__ is hannsdb.HnswHvqIndexParam
        reopened_hits = reopened.query(
            query_context=hannsdb.QueryContext(
                top_k=2,
                queries=[hannsdb.VectorQuery(field_name="dense", vector=[1.0, 0.0])],
            )
        )
        assert [doc.id for doc in reopened_hits] == ["doc-1", "doc-2"]
        return 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
