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

    temp_dir = Path(tempfile.mkdtemp(prefix="hannsdb_schema_mutation_smoke_"))
    try:
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

        collection = hannsdb.create_and_open(str(temp_dir), schema)
        collection_path = collection.path

        collection.insert(
            [
                hannsdb.Doc(
                    id="doc-1",
                    vector=[0.0, 0.0],
                    field_name="dense",
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

        collection.alter_column(
            "group",
            new_name="bucket",
            option=hannsdb.AlterColumnOption(concurrency=1),
        )
        assert collection.schema.field("bucket").name == "bucket"
        try:
            collection.schema.field("group")
        except KeyError:
            pass
        else:
            raise AssertionError("expected old field name to be removed after rename")

        try:
            collection.add_column(
                hannsdb.FieldSchema(name="bad_expr", data_type="int64"),
                expression="1",
                option=hannsdb.AddColumnOption(concurrency=1),
            )
        except NotImplementedError as exc:
            assert "expression" in str(exc)
        else:
            raise AssertionError("expected add_column expression path to be rejected")

        reopened = hannsdb.open(collection_path)
        assert reopened.schema.field("bucket").name == "bucket"
        try:
            reopened.schema.field("group")
        except KeyError:
            pass
        else:
            raise AssertionError("expected rename to survive reopen")

        return 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
