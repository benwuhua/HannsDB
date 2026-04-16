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


def test_concurrent_insert_and_query(tmp_path):
    """Multiple threads insert while others query — no crashes, consistent results."""
    collection = build_collection(tmp_path)

    start = threading.Event()
    errors = []
    lock = threading.Lock()

    def inserter(thread_id):
        try:
            start.wait()
            for i in range(5):
                doc = hannsdb.Doc(
                    id=f"{thread_id}_{i}",
                    vector=[float(thread_id), float(i)],
                    fields={"group": thread_id % 3},
                )
                collection.insert(doc)
        except Exception as exc:
            with lock:
                errors.append(exc)

    def querier(_):
        try:
            start.wait()
            for _ in range(5):
                collection.query(
                    vectors=hannsdb.VectorQuery(
                        field_name="dense", vector=[0.0, 0.0], param=None
                    ),
                    topk=3,
                )
        except Exception as exc:
            with lock:
                errors.append(exc)

    threads = [
        *[threading.Thread(target=inserter, args=(i,)) for i in range(3)],
        *[threading.Thread(target=querier, args=(i,)) for i in range(3)],
    ]
    for t in threads:
        t.start()
    start.set()
    _join_threads(threads)

    assert errors == []
    stats = collection.stats
    assert stats.doc_count == 15  # 3 threads * 5 docs

    collection.destroy()


def test_concurrent_insert_and_delete(tmp_path):
    """Interleaved insert/delete — final doc_count matches expected."""
    collection = build_collection(tmp_path)

    errors = []
    lock = threading.Lock()

    # Phase 1: insert all docs first
    def inserter(thread_id):
        try:
            for i in range(10):
                doc = hannsdb.Doc(
                    id=f"{thread_id}_{i}",
                    vector=[float(thread_id), float(i)],
                    fields={"group": thread_id},
                )
                collection.insert(doc)
        except Exception as exc:
            with lock:
                errors.append(exc)

    insert_threads = [threading.Thread(target=inserter, args=(i,)) for i in range(2)]
    for t in insert_threads:
        t.start()
    _join_threads(insert_threads)

    assert errors == []

    # Phase 2: concurrent deletes
    start = threading.Event()

    def deleter(thread_id):
        try:
            start.wait()
            for i in range(5):
                collection.delete(f"{thread_id}_{i}")
        except Exception as exc:
            with lock:
                errors.append(exc)

    delete_threads = [threading.Thread(target=deleter, args=(i,)) for i in range(2)]
    for t in delete_threads:
        t.start()
    start.set()
    _join_threads(delete_threads)

    assert errors == []
    stats = collection.stats
    assert stats.doc_count == 10  # 20 inserted - 10 deleted

    collection.destroy()


def test_read_write_locking(tmp_path):
    """Write holds lock; reads queue but don't deadlock."""
    collection = build_collection(tmp_path)
    _seed_docs(collection, 5)

    start = threading.Event()
    errors = []
    completed = []
    lock = threading.Lock()

    def writer(_):
        try:
            start.wait()
            for i in range(10, 20):
                doc = hannsdb.Doc(
                    id=str(i),
                    vector=[float(i), float(i)],
                    fields={"group": i % 3},
                )
                collection.insert(doc)
            with lock:
                completed.append("write")
        except Exception as exc:
            with lock:
                errors.append(exc)

    def reader(_):
        try:
            start.wait()
            collection.fetch(["0", "1", "2"])
            collection.query(
                vectors=hannsdb.VectorQuery(
                    field_name="dense", vector=[0.0, 0.0], param=None
                ),
                topk=3,
            )
            with lock:
                completed.append("read")
        except Exception as exc:
            with lock:
                errors.append(exc)

    threads = [
        threading.Thread(target=writer, args=(0,)),
        *[threading.Thread(target=reader, args=(i,)) for i in range(3)],
    ]
    for t in threads:
        t.start()
    start.set()
    _join_threads(threads, timeout=10.0)

    assert errors == []
    assert "write" in completed
    assert completed.count("read") == 3

    collection.destroy()


def test_race_condition_detection(tmp_path):
    """Rapid fire insert/fetch/query cycle — no crashes."""
    collection = build_collection(tmp_path)

    errors = []
    lock = threading.Lock()

    def worker(thread_id):
        try:
            for i in range(20):
                doc_id = f"{thread_id}_{i}"
                doc = hannsdb.Doc(
                    id=doc_id,
                    vector=[float(thread_id), float(i)],
                    fields={"group": thread_id % 3},
                )
                collection.insert(doc)
                collection.fetch([doc_id])
                collection.query(
                    vectors=hannsdb.VectorQuery(
                        field_name="dense",
                        vector=[float(thread_id), float(i)],
                        param=None,
                    ),
                    topk=1,
                )
        except Exception as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    _join_threads(threads, timeout=15.0)

    assert errors == []
    stats = collection.stats
    assert stats.doc_count == 80  # 4 threads * 20 docs

    collection.destroy()
