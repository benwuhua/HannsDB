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

    temp_dir = Path(tempfile.mkdtemp(prefix="hannsdb_string_pk_smoke_"))
    try:
        schema = hannsdb.CollectionSchema(
            name="docs",
            primary_vector="dense",
            fields=[hannsdb.FieldSchema(name="rank", data_type="int64")],
            vectors=[
                hannsdb.VectorSchema(
                    name="dense",
                    data_type="vector_fp32",
                    dimension=2,
                )
            ],
        )

        collection = hannsdb.create_and_open(str(temp_dir), schema)
        collection_path = collection.path
        collection.insert(
            [
                hannsdb.Doc(id="user-a", vector=[0.0, 0.0], fields={"rank": 2}),
                hannsdb.Doc(id="user-b", vector=[0.1, 0.0], fields={"rank": 1}),
                hannsdb.Doc(id="user-c", vector=[0.2, 0.0], fields={"rank": 3}),
            ]
        )

        by_id = collection.query(
            query_context=hannsdb.QueryContext(
                top_k=3,
                query_by_id=["user-b"],
                output_fields=["rank"],
            )
        )
        assert [doc.id for doc in by_id] == ["user-b", "user-a", "user-c"]
        assert [doc.field("rank") for doc in by_id] == [1, 2, 3]

        ordered = collection.query(
            query_by_id=["user-b"],
            output_fields=["rank"],
            order_by=hannsdb.QueryOrderBy(field_name="rank"),
        )
        assert [doc.id for doc in ordered] == ["user-b", "user-a", "user-c"]
        assert [doc.field("rank") for doc in ordered] == [1, 2, 3]

        reranked = collection.query(
            query_context=hannsdb.QueryContext(
                top_k=2,
                queries=[
                    hannsdb.VectorQuery(
                        field_name="dense",
                        vector=[0.0, 0.0],
                        param=None,
                    )
                ],
                query_by_id=["user-b"],
                reranker=hannsdb.RrfReRanker(topn=2),
                output_fields=["rank"],
            )
        )
        assert [doc.id for doc in reranked] == ["user-a", "user-b"]
        assert [doc.field("rank") for doc in reranked] == [2, 1]

        reopened = hannsdb.open(collection_path)
        reopened_hits = reopened.query(
            query_context=hannsdb.QueryContext(
                top_k=3,
                query_by_id=["user-b"],
                output_fields=["rank"],
            )
        )
        assert [doc.id for doc in reopened_hits] == ["user-b", "user-a", "user-c"]
        assert [doc.field("rank") for doc in reopened_hits] == [1, 2, 3]
        return 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
