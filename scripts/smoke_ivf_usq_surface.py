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

    assert hannsdb.IvfUsqIndexParam is hannsdb.model.param.IvfUsqIndexParam
    assert hannsdb.IvfUsqQueryParam is hannsdb.model.param.IvfUsqQueryParam

    ivf_usq_dir = Path(tempfile.mkdtemp(prefix="hannsdb_ivf_usq_smoke_"))
    try:
        ivf_usq_schema = hannsdb.CollectionSchema(
            name="docs_ivf_usq",
            primary_vector="dense",
            fields=[
                hannsdb.FieldSchema(name="title", data_type=hannsdb.DataType.String),
                hannsdb.FieldSchema(name="rank", data_type=hannsdb.DataType.Int32),
            ],
            vectors=[
                hannsdb.VectorSchema(
                    name="dense",
                    data_type=hannsdb.DataType.VectorFp32,
                    dimension=2,
                    index_param=hannsdb.IvfUsqIndexParam(
                        metric_type=hannsdb.MetricType.L2,
                        nlist=1,
                        bits_per_dim=4,
                        rotation_seed=42,
                        rerank_k=64,
                        use_high_accuracy_scan=False,
                    ),
                )
            ],
        )

        collection = hannsdb.create_and_open(str(ivf_usq_dir), ivf_usq_schema)
        collection.insert(
            [
                hannsdb.Doc(
                    id="doc-1",
                    fields={"title": "a", "rank": 1},
                    vectors={"dense": [0.0, 0.0]},
                ),
                hannsdb.Doc(
                    id="doc-2",
                    fields={"title": "b", "rank": 2},
                    vectors={"dense": [0.1, 0.0]},
                ),
                hannsdb.Doc(
                    id="doc-3",
                    fields={"title": "c", "rank": 3},
                    vectors={"dense": [5.0, 5.0]},
                ),
            ]
        )
        collection.optimize()

        hits = collection.query(
            query_context=hannsdb.QueryContext(
                top_k=2,
                output_fields=["title", "rank"],
                queries=[
                    hannsdb.VectorQuery(
                        field_name="dense",
                        vector=[0.0, 0.0],
                        param=hannsdb.IvfUsqQueryParam(nprobe=2),
                    )
                ],
            )
        )
        assert len(hits) == 2
        assert [doc.id for doc in hits] == ["doc-1", "doc-2"]

        reopened = hannsdb.open(str(ivf_usq_dir))
        assert reopened.schema.vector("dense").index_param.__class__ is hannsdb.IvfUsqIndexParam
        assert reopened.schema.vector("dense").index_param.nlist == 1

        reopened_hits = reopened.query(
            query_context=hannsdb.QueryContext(
                top_k=2,
                queries=[
                    hannsdb.VectorQuery(
                        field_name="dense",
                        vector=[0.0, 0.0],
                        param=hannsdb.IvfUsqQueryParam(nprobe=2),
                    )
                ],
            )
        )
        assert [doc.id for doc in reopened_hits] == ["doc-1", "doc-2"]
        return 0
    finally:
        shutil.rmtree(ivf_usq_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
