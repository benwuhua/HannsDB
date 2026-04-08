import threading

import hannsdb


def build_collection(tmp_path):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[
            hannsdb.FieldSchema(name="group", data_type="int64"),
        ],
        vectors=[
            hannsdb.VectorSchema(
                name="dense",
                data_type="vector_fp32",
                dimension=2,
            )
        ],
    )
    return hannsdb.create_and_open(str(tmp_path), schema)


def _seed_docs(collection, count=12):
    docs = [
        hannsdb.Doc(
            id=str(i),
            vector=[float(i), float(i + 1)],
            fields={"group": i % 3},
        )
        for i in range(count)
    ]
    assert collection.insert(docs) == count


def test_concurrent_query(tmp_path):
    collection = build_collection(tmp_path)
    _seed_docs(collection)

    start = threading.Event()
    errors = []
    lock = threading.Lock()

    def worker():
        try:
            start.wait()
            result = collection.query(
                vectors=hannsdb.VectorQuery(
                    field_name="dense",
                    vector=[0.0, 0.0],
                    param=None,
                ),
                topk=3,
            )
            assert len(result) >= 1
        except Exception as exc:  # pragma: no cover - exercised on failure
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for thread in threads:
        thread.start()

    start.set()

    for thread in threads:
        thread.join()

    assert errors == []

    collection.destroy()


def test_mixed_read_write_keeps_collection_usable(tmp_path):
    collection = build_collection(tmp_path)
    _seed_docs(collection)

    start = threading.Event()
    errors = []
    lock = threading.Lock()

    def record_error(exc):
        with lock:
            errors.append(exc)

    def reader(thread_id):
        try:
            start.wait()
            fetched = collection.fetch(["1", "2"])
            queried = collection.query(
                vectors=hannsdb.VectorQuery(
                    field_name="dense",
                    vector=[1.0, 2.0],
                    param=None,
                ),
                topk=3,
            )
            assert fetched
            assert queried
        except Exception as exc:  # pragma: no cover - exercised on failure
            record_error(exc)

    def writer(thread_id):
        try:
            start.wait()
            doc = hannsdb.Doc(
                id=str(100 + thread_id),
                vector=[float(thread_id), float(thread_id + 1)],
                fields={"group": thread_id % 3},
            )
            collection.upsert([doc])
        except Exception as exc:  # pragma: no cover - exercised on failure
            record_error(exc)

    threads = []
    for i in range(2):
        threads.append(threading.Thread(target=reader, args=(i,)))
    for i in range(3):
        threads.append(threading.Thread(target=writer, args=(i,)))

    for thread in threads:
        thread.start()

    start.set()

    for thread in threads:
        thread.join()

    assert errors == []

    final_query = collection.query(
        vectors=hannsdb.VectorQuery(
            field_name="dense",
            vector=[0.0, 1.0],
            param=None,
        ),
        topk=3,
    )
    final_fetch = collection.fetch(["100", "101", "102"])

    assert final_query
    assert len(final_fetch) == 3

    collection.destroy()
