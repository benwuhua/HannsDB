from __future__ import annotations

import tempfile

import hannsdb


def test_hannsdb_smoke() -> None:
    hannsdb.init(hannsdb.LogLevel.Warn)

    with tempfile.TemporaryDirectory() as tmp:
        index_param = hannsdb.HnswIndexParam(
            metric_type=hannsdb.MetricType.L2,
            m=16,
            ef_construction=64,
            quantize_type=hannsdb.QuantizeType.Undefined,
        )
        vector_schema = hannsdb.VectorSchema(
            name="dense",
            data_type=hannsdb.DataType.VectorFp32,
            dimension=2,
            index_param=index_param,
        )
        schema = hannsdb.CollectionSchema(name="bench_col", vector_schema=vector_schema)
        option = hannsdb.CollectionOption(read_only=False, enable_mmap=True)

        collection = hannsdb.create_and_open(tmp, schema, option)
        docs = [
            hannsdb.Doc(id="11", vector=[0.0, 0.0], field_name="dense"),
            hannsdb.Doc(id="22", vector=[10.0, 10.0], field_name="dense"),
        ]
        inserted = collection.insert(docs)
        assert inserted == 2

        query = hannsdb.VectorQuery(
            field_name="dense",
            vector=[0.1, -0.1],
            param=hannsdb.HnswQueryParam(ef=32, is_using_refiner=False),
        )
        hits = collection.query(output_fields=[], topk=1, filter="", vectors=query)
        assert hits[0].id == "11"

        collection.optimize()
        reopened = hannsdb.open(tmp, option)
        assert reopened.collection_name == "bench_col"
        reopened.destroy()


if __name__ == "__main__":
    test_hannsdb_smoke()
