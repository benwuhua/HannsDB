import threading

import hannsdb


def _join_threads(threads, timeout=5.0):
    for thread in threads:
        thread.join(timeout=timeout)
    alive = [thread.name for thread in threads if thread.is_alive()]
    assert not alive, f"threads did not finish within {timeout}s: {alive}"


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
    results = []
    lock = threading.Lock()

    def worker(thread_id):
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
            with lock:
                results.append((thread_id, len(result), tuple(doc.id for doc in result)))
        except Exception as exc:  # pragma: no cover - exercised on failure
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for thread in threads:
        thread.start()

    start.set()

    _join_threads(threads)

    assert errors == []
    assert [count for _, count, _ in sorted(results)] == [3, 3, 3, 3, 3]
    assert len({ids for _, _, ids in results}) == 1

    collection.destroy()


def test_mixed_read_write_keeps_collection_usable(tmp_path):
    collection = build_collection(tmp_path)
    _seed_docs(collection)

    start = threading.Event()
    errors = []
    results = []
    lock = threading.Lock()

    def record_error(exc):
        with lock:
            errors.append(exc)

    def record_result(kind, value):
        with lock:
            results.append((kind, value))

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
            record_result("fetch", tuple(doc.id for doc in fetched))
            record_result("query", len(queried))
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
            if thread_id < 2:
                count = collection.insert([doc, doc._replace(id=str(200 + thread_id))])
                record_result("insert", count)
            else:
                count = collection.upsert([doc, doc._replace(id=str(300 + thread_id))])
                record_result("upsert", count)
        except Exception as exc:  # pragma: no cover - exercised on failure
            record_error(exc)

    threads = []
    for i in range(2):
        threads.append(threading.Thread(target=reader, args=(i,)))
    for i in range(4):
        threads.append(threading.Thread(target=writer, args=(i,)))

    for thread in threads:
        thread.start()

    start.set()

    _join_threads(threads)

    assert errors == []
    assert sorted(value for kind, value in results if kind == "insert") == [2, 2]
    assert sorted(value for kind, value in results if kind == "upsert") == [2, 2]
    assert sorted(value for kind, value in results if kind == "fetch") == [("1", "2"), ("1", "2")]
    assert sorted(value for kind, value in results if kind == "query") == [3, 3]

    final_query = collection.query(
        vectors=hannsdb.VectorQuery(
            field_name="dense",
            vector=[0.0, 1.0],
            param=None,
        ),
        topk=20,
    )
    final_fetch = collection.fetch(
        ["100", "101", "102", "103", "200", "201", "302", "303"]
    )

    assert len(final_query) == 20
    assert {doc.id for doc in final_fetch} == {
        "100",
        "101",
        "102",
        "103",
        "200",
        "201",
        "302",
        "303",
    }

    collection.destroy()
