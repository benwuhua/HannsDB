//! DML edge-case and concurrent-operation tests ported from zvec's
//! test_collection_dml.py and test_collection_concurrency.py.
//!
//! Covers:
//!   batch insert, duplicate-id rejection, upsert, partial update,
//!   delete by id / by filter, fetch edge cases, sparse-vector DML,
//!   empty-collection query, insert-after-delete, and basic concurrency.

use std::collections::BTreeMap;
use std::io::ErrorKind;
use std::sync::{Arc, Mutex};
use std::thread;

use hannsdb_core::db::HannsDb;
use hannsdb_core::document::{
    CollectionSchema, Document, DocumentUpdate, FieldType, FieldValue, ScalarFieldSchema,
    SparseVector, VectorFieldSchema,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Primary vector name used across all tests.  `Document::new()` hard-codes
/// "vector" as the primary vector key, so the schema must match.
const PV: &str = "vector";

fn make_schema() -> CollectionSchema {
    CollectionSchema::new(
        PV,
        4,
        "l2",
        vec![
            ScalarFieldSchema::new("name", FieldType::String),
            ScalarFieldSchema::new("age", FieldType::Int64),
        ],
    )
}

fn make_schema_with_sparse() -> CollectionSchema {
    let mut schema = CollectionSchema::new(
        PV,
        4,
        "l2",
        vec![
            ScalarFieldSchema::new("name", FieldType::String),
            ScalarFieldSchema::new("age", FieldType::Int64),
        ],
    );
    schema.vectors.push(VectorFieldSchema {
        name: "sparse".to_string(),
        data_type: FieldType::VectorSparse,
        dimension: 0,
        index_param: None,
        bm25_params: None,
    });
    schema
}

fn doc(id: i64, name: &str, age: i64, vec: Vec<f32>) -> Document {
    Document::new(
        id,
        [
            ("name".to_string(), FieldValue::String(name.to_string())),
            ("age".to_string(), FieldValue::Int64(age)),
        ],
        vec,
    )
}

fn doc_sparse(id: i64, name: &str, age: i64, vec: Vec<f32>, sv: SparseVector) -> Document {
    Document::with_sparse_vectors(
        id,
        [
            ("name".to_string(), FieldValue::String(name.to_string())),
            ("age".to_string(), FieldValue::Int64(age)),
        ],
        PV,
        vec,
        [("sparse".to_string(), sv)],
    )
}

// ---------------------------------------------------------------------------
// 1. Batch insert
// ---------------------------------------------------------------------------

#[test]
fn batch_insert_100_docs_all_fetched() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(tmp.path()).expect("open");
    db.create_collection_with_schema("c", &make_schema()).expect("create");

    let docs: Vec<Document> = (0..100)
        .map(|i| doc(i, &format!("user_{i}"), 20 + i, vec![i as f32; 4]))
        .collect();
    let n = db.insert_documents("c", &docs).expect("insert");
    assert_eq!(n, 100);

    let ids: Vec<i64> = (0..100).collect();
    let fetched = db.fetch_documents("c", &ids).expect("fetch");
    assert_eq!(fetched.len(), 100);

    let info = db.get_collection_info("c").expect("info");
    assert_eq!(info.live_count, 100);
}

// ---------------------------------------------------------------------------
// 2. Insert duplicate ID
// ---------------------------------------------------------------------------

#[test]
fn insert_duplicate_id_fails() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(tmp.path()).expect("open");
    db.create_collection_with_schema("c", &make_schema()).expect("create");

    db.insert_documents("c", &[doc(1, "alice", 30, vec![0.1; 4])])
        .expect("first insert");

    let err = db
        .insert_documents("c", &[doc(1, "bob", 25, vec![0.2; 4])])
        .expect_err("duplicate insert should fail");
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.to_string().contains("already exists"));

    // Collection count should still be 1.
    let info = db.get_collection_info("c").expect("info");
    assert_eq!(info.live_count, 1);
}

// ---------------------------------------------------------------------------
// 3. Upsert existing doc: update fields and vector
// ---------------------------------------------------------------------------

#[test]
fn upsert_existing_doc_updates_fields_and_vector() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(tmp.path()).expect("open");
    db.create_collection_with_schema("c", &make_schema()).expect("create");

    db.insert_documents("c", &[doc(1, "alice", 30, vec![0.1; 4])])
        .expect("insert");

    db.upsert_documents("c", &[doc(1, "bob", 25, vec![0.9; 4])])
        .expect("upsert");

    let fetched = db.fetch_documents("c", &[1]).expect("fetch");
    assert_eq!(fetched.len(), 1);
    assert_eq!(
        fetched[0].fields.get("name"),
        Some(&FieldValue::String("bob".into()))
    );
    assert_eq!(fetched[0].fields.get("age"), Some(&FieldValue::Int64(25)));
    assert_eq!(
        fetched[0].vectors.get(PV),
        Some(&vec![0.9; 4])
    );

    // Only one live doc.
    let info = db.get_collection_info("c").expect("info");
    assert_eq!(info.live_count, 1);
}

// ---------------------------------------------------------------------------
// 4. Upsert new doc: acts as insert
// ---------------------------------------------------------------------------

#[test]
fn upsert_new_doc_acts_as_insert() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(tmp.path()).expect("open");
    db.create_collection_with_schema("c", &make_schema()).expect("create");

    db.upsert_documents("c", &[doc(1, "alice", 30, vec![0.1; 4])])
        .expect("upsert as insert");

    let fetched = db.fetch_documents("c", &[1]).expect("fetch");
    assert_eq!(fetched.len(), 1);
    assert_eq!(
        fetched[0].fields.get("name"),
        Some(&FieldValue::String("alice".into()))
    );
}

// ---------------------------------------------------------------------------
// 5. Partial field update: update only some fields, others unchanged
// ---------------------------------------------------------------------------

#[test]
fn partial_field_update_preserves_untouched_fields() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(tmp.path()).expect("open");
    db.create_collection_with_schema("c", &make_schema()).expect("create");

    db.insert_documents("c", &[doc(1, "alice", 30, vec![0.1; 4])])
        .expect("insert");

    let mut fields = BTreeMap::new();
    fields.insert("age".to_string(), Some(FieldValue::Int64(31)));
    // "name" is NOT included => should remain "alice".

    db.update_documents(
        "c",
        &[DocumentUpdate {
            id: 1,
            fields,
            vectors: BTreeMap::new(),
            sparse_vectors: BTreeMap::new(),
        }],
    )
    .expect("update");

    let fetched = db.fetch_documents("c", &[1]).expect("fetch");
    assert_eq!(fetched.len(), 1);
    assert_eq!(
        fetched[0].fields.get("name"),
        Some(&FieldValue::String("alice".into()))
    );
    assert_eq!(fetched[0].fields.get("age"), Some(&FieldValue::Int64(31)));
    // Vector unchanged.
    assert_eq!(fetched[0].vectors.get(PV), Some(&vec![0.1; 4]));
}

// ---------------------------------------------------------------------------
// 6. Update vector only: update dense vector, fields unchanged
// ---------------------------------------------------------------------------

#[test]
fn update_vector_only_preserves_fields() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(tmp.path()).expect("open");
    db.create_collection_with_schema("c", &make_schema()).expect("create");

    db.insert_documents("c", &[doc(1, "alice", 30, vec![0.1; 4])])
        .expect("insert");

    let mut vectors = BTreeMap::new();
    vectors.insert(PV.to_string(), Some(vec![0.5; 4]));

    db.update_documents(
        "c",
        &[DocumentUpdate {
            id: 1,
            fields: BTreeMap::new(),
            vectors,
            sparse_vectors: BTreeMap::new(),
        }],
    )
    .expect("update vector only");

    let fetched = db.fetch_documents("c", &[1]).expect("fetch");
    assert_eq!(fetched.len(), 1);
    assert_eq!(
        fetched[0].fields.get("name"),
        Some(&FieldValue::String("alice".into()))
    );
    assert_eq!(fetched[0].vectors.get(PV), Some(&vec![0.5; 4]));
}

// ---------------------------------------------------------------------------
// 7. Delete by filter: delete docs matching filter, verify deleted
// ---------------------------------------------------------------------------

#[test]
fn delete_by_filter_removes_matching_docs() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(tmp.path()).expect("open");
    db.create_collection_with_schema("c", &make_schema()).expect("create");

    db.insert_documents(
        "c",
        &[
            doc(1, "alice", 30, vec![0.1; 4]),
            doc(2, "bob", 25, vec![0.2; 4]),
            doc(3, "carol", 35, vec![0.3; 4]),
            doc(4, "dave", 40, vec![0.4; 4]),
        ],
    )
    .expect("insert");

    let deleted = db.delete_by_filter("c", "age > 30").expect("delete by filter");
    assert_eq!(deleted, 2); // carol(35), dave(40)

    let fetched = db.fetch_documents("c", &[1, 2, 3, 4]).expect("fetch");
    let fetched_ids: Vec<i64> = fetched.iter().map(|d| d.id).collect();
    assert_eq!(fetched_ids, vec![1, 2]);
}

// ---------------------------------------------------------------------------
// 8. Delete by IDs: delete specific docs, verify gone
// ---------------------------------------------------------------------------

#[test]
fn delete_by_ids_removes_target_docs() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(tmp.path()).expect("open");
    db.create_collection_with_schema("c", &make_schema()).expect("create");

    db.insert_documents(
        "c",
        &[
            doc(1, "alice", 30, vec![0.1; 4]),
            doc(2, "bob", 25, vec![0.2; 4]),
            doc(3, "carol", 35, vec![0.3; 4]),
        ],
    )
    .expect("insert");

    let deleted = db.delete("c", &[1, 3]).expect("delete");
    assert_eq!(deleted, 2);

    let fetched = db.fetch_documents("c", &[1, 2, 3]).expect("fetch");
    assert_eq!(fetched.len(), 1);
    assert_eq!(fetched[0].id, 2);
}

// ---------------------------------------------------------------------------
// 9. Delete non-existent ID: should succeed (no-op or return 0)
// ---------------------------------------------------------------------------

#[test]
fn delete_non_existent_id_is_noop() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(tmp.path()).expect("open");
    db.create_collection_with_schema("c", &make_schema()).expect("create");

    // Empty collection — deleting anything is a no-op.
    let deleted = db.delete("c", &[999]).expect("delete non-existent");
    assert_eq!(deleted, 0);
}

// ---------------------------------------------------------------------------
// 10. Fetch non-existent ID: should return empty
// ---------------------------------------------------------------------------

#[test]
fn fetch_non_existent_id_returns_empty() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(tmp.path()).expect("open");
    db.create_collection_with_schema("c", &make_schema()).expect("create");

    db.insert_documents("c", &[doc(1, "alice", 30, vec![0.1; 4])])
        .expect("insert");

    let fetched = db.fetch_documents("c", &[999]).expect("fetch");
    assert!(fetched.is_empty());
}

// ---------------------------------------------------------------------------
// 11. Fetch partial non-existent: some exist, some don't
// ---------------------------------------------------------------------------

#[test]
fn fetch_partial_non_existent_returns_only_existing() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(tmp.path()).expect("open");
    db.create_collection_with_schema("c", &make_schema()).expect("create");

    db.insert_documents(
        "c",
        &[
            doc(1, "alice", 30, vec![0.1; 4]),
            doc(3, "carol", 35, vec![0.3; 4]),
        ],
    )
    .expect("insert");

    let fetched = db.fetch_documents("c", &[1, 2, 3]).expect("fetch");
    let fetched_ids: Vec<i64> = fetched.iter().map(|d| d.id).collect();
    assert_eq!(fetched_ids, vec![1, 3]);
}

// ---------------------------------------------------------------------------
// 12. Insert with sparse vectors: verify sparse vector stored and fetchable
// ---------------------------------------------------------------------------

#[test]
fn insert_with_sparse_vectors_stored_and_fetchable() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(tmp.path()).expect("open");
    db.create_collection_with_schema("c", &make_schema_with_sparse())
        .expect("create");

    let sv = SparseVector::new(vec![1, 5, 10], vec![0.5, 1.0, 2.0]);
    db.insert_documents("c", &[doc_sparse(1, "alice", 30, vec![0.1; 4], sv)])
        .expect("insert");

    let fetched = db.fetch_documents("c", &[1]).expect("fetch");
    assert_eq!(fetched.len(), 1);
    let got = fetched[0].sparse_vectors.get("sparse").expect("sparse field");
    assert_eq!(got.indices, vec![1, 5, 10]);
    assert_eq!(got.values, vec![0.5, 1.0, 2.0]);
}

// ---------------------------------------------------------------------------
// 13. Update sparse vector: update sparse vector field
// ---------------------------------------------------------------------------

#[test]
fn update_sparse_vector_replaces_value() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(tmp.path()).expect("open");
    db.create_collection_with_schema("c", &make_schema_with_sparse())
        .expect("create");

    let sv_old = SparseVector::new(vec![1], vec![0.1]);
    db.insert_documents("c", &[doc_sparse(1, "alice", 30, vec![0.1; 4], sv_old)])
        .expect("insert");

    let mut sparse_updates = BTreeMap::new();
    let sv_new = SparseVector::new(vec![2, 7], vec![3.0, 4.0]);
    sparse_updates.insert("sparse".to_string(), Some(sv_new));

    db.update_documents(
        "c",
        &[DocumentUpdate {
            id: 1,
            fields: BTreeMap::new(),
            vectors: BTreeMap::new(),
            sparse_vectors: sparse_updates,
        }],
    )
    .expect("update sparse");

    let fetched = db.fetch_documents("c", &[1]).expect("fetch");
    let got = fetched[0].sparse_vectors.get("sparse").expect("sparse field");
    assert_eq!(got.indices, vec![2, 7]);
    assert_eq!(got.values, vec![3.0, 4.0]);
    // Other fields unchanged.
    assert_eq!(
        fetched[0].fields.get("name"),
        Some(&FieldValue::String("alice".into()))
    );
}

// ---------------------------------------------------------------------------
// 14. Empty collection query: query returns empty
// ---------------------------------------------------------------------------

#[test]
fn empty_collection_search_returns_no_hits() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(tmp.path()).expect("open");
    db.create_collection_with_schema("c", &make_schema()).expect("create");

    let hits = db.search("c", &[0.0; 4], 10).expect("search");
    assert!(hits.is_empty());

    let hits = db
        .query_documents("c", &[0.0; 4], 10, None)
        .expect("query_documents");
    assert!(hits.is_empty());
}

// ---------------------------------------------------------------------------
// 15. Insert after delete: re-insert deleted ID
// ---------------------------------------------------------------------------

#[test]
fn insert_after_delete_succeeds() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(tmp.path()).expect("open");
    db.create_collection_with_schema("c", &make_schema()).expect("create");

    db.insert_documents("c", &[doc(1, "alice", 30, vec![0.1; 4])])
        .expect("insert");
    db.delete("c", &[1]).expect("delete");

    // Re-insert same ID with new data.
    db.insert_documents("c", &[doc(1, "bob", 25, vec![0.9; 4])])
        .expect("re-insert");

    let fetched = db.fetch_documents("c", &[1]).expect("fetch");
    assert_eq!(fetched.len(), 1);
    assert_eq!(
        fetched[0].fields.get("name"),
        Some(&FieldValue::String("bob".into()))
    );
    assert_eq!(fetched[0].vectors.get(PV), Some(&vec![0.9; 4]));
}

// ---------------------------------------------------------------------------
// 16. Upsert batch: batch upsert mix of new and existing docs
// ---------------------------------------------------------------------------

#[test]
fn batch_upsert_mix_new_and_existing() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(tmp.path()).expect("open");
    db.create_collection_with_schema("c", &make_schema()).expect("create");

    // Pre-insert id=1.
    db.insert_documents("c", &[doc(1, "alice", 30, vec![0.1; 4])])
        .expect("insert");

    // Batch upsert: update id=1, insert id=2 and id=3.
    db.upsert_documents(
        "c",
        &[
            doc(1, "alice_v2", 31, vec![0.2; 4]),
            doc(2, "bob", 25, vec![0.3; 4]),
            doc(3, "carol", 35, vec![0.4; 4]),
        ],
    )
    .expect("batch upsert");

    let fetched = db.fetch_documents("c", &[1, 2, 3]).expect("fetch");
    assert_eq!(fetched.len(), 3);

    let f1 = fetched.iter().find(|d| d.id == 1).expect("id 1");
    assert_eq!(
        f1.fields.get("name"),
        Some(&FieldValue::String("alice_v2".into()))
    );

    let info = db.get_collection_info("c").expect("info");
    assert_eq!(info.live_count, 3);
}

// ---------------------------------------------------------------------------
// 17. Delete all + re-query: verify empty results
// ---------------------------------------------------------------------------

#[test]
fn delete_all_then_query_returns_empty() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(tmp.path()).expect("open");
    db.create_collection_with_schema("c", &make_schema()).expect("create");

    db.insert_documents(
        "c",
        &[
            doc(1, "alice", 30, vec![0.1; 4]),
            doc(2, "bob", 25, vec![0.2; 4]),
        ],
    )
    .expect("insert");

    db.delete("c", &[1, 2]).expect("delete all");

    let fetched = db.fetch_documents("c", &[1, 2]).expect("fetch");
    assert!(fetched.is_empty());

    let hits = db.search("c", &[0.0; 4], 10).expect("search");
    assert!(hits.is_empty());
}

// ---------------------------------------------------------------------------
// 18. Concurrent insert and query: serialized via Mutex, no deadlock
// ---------------------------------------------------------------------------

#[test]
fn concurrent_insert_and_query_no_deadlock() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(tmp.path()).expect("open");
    db.create_collection_with_schema("c", &make_schema()).expect("create");

    // Seed one document so search has something to find.
    db.insert_documents("c", &[doc(0, "seed", 0, vec![0.0; 4])])
        .expect("seed insert");

    let db = Arc::new(Mutex::new(db));
    let mut handles = Vec::new();

    // 4 threads: 2 inserters, 2 searchers.
    for i in 0..4 {
        let db_clone = db.clone();
        handles.push(thread::spawn(move || {
            let mut db = db_clone.lock().expect("lock");
            if i % 2 == 0 {
                // Inserter.
                let docs: Vec<Document> = (1..=5)
                    .map(|j| {
                        doc(i * 100 + j, &format!("t{i}_d{j}"), 20 + j, vec![j as f32; 4])
                    })
                    .collect();
                db.insert_documents("c", &docs).expect("concurrent insert");
            } else {
                // Searcher.
                let _hits = db.search("c", &[0.0; 4], 10).expect("concurrent search");
            }
        }));
    }

    for h in handles {
        h.join().expect("thread panicked");
    }

    // Verify final state is consistent.
    let db = Arc::try_unwrap(db)
        .ok()
        .expect("unwrap arc")
        .into_inner()
        .expect("into_inner");
    let info = db.get_collection_info("c").expect("info");
    // Seed (1) + 2 inserters * 5 docs = 11.
    assert_eq!(info.live_count, 11);
}

// ---------------------------------------------------------------------------
// Concurrent mixed operations: insert, update, delete, fetch
// ---------------------------------------------------------------------------

#[test]
fn concurrent_mixed_operations_no_deadlock() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(tmp.path()).expect("open");
    db.create_collection_with_schema("c", &make_schema()).expect("create");

    // Seed 10 documents.
    let seed: Vec<Document> = (0..10)
        .map(|i| doc(i, &format!("seed_{i}"), 20 + i, vec![i as f32; 4]))
        .collect();
    db.insert_documents("c", &seed).expect("seed insert");

    let db = Arc::new(Mutex::new(db));
    let mut handles = Vec::new();

    // 6 threads performing different operations.
    for i in 0..6 {
        let db_clone = db.clone();
        handles.push(thread::spawn(move || {
            let mut db = db_clone.lock().expect("lock");
            match i % 3 {
                0 => {
                    // Insert new docs.
                    let docs = vec![doc(100 + i, &format!("new_{i}"), 30, vec![1.0; 4])];
                    let _ = db.insert_documents("c", &docs);
                }
                1 => {
                    // Update existing docs.
                    let mut fields = BTreeMap::new();
                    fields.insert(
                        "name".to_string(),
                        Some(FieldValue::String(format!("updated_{i}"))),
                    );
                    let _ = db.update_documents(
                        "c",
                        &[DocumentUpdate {
                            id: i as i64,
                            fields,
                            vectors: BTreeMap::new(),
                            sparse_vectors: BTreeMap::new(),
                        }],
                    );
                }
                2 => {
                    // Fetch + search.
                    let _ = db.fetch_documents("c", &[0, 1, 2]);
                    let _ = db.search("c", &[0.0; 4], 5);
                }
                _ => unreachable!(),
            }
        }));
    }

    for h in handles {
        h.join().expect("thread panicked");
    }
}

// ---------------------------------------------------------------------------
// Concurrent rapid fire: many threads performing quick operations
// ---------------------------------------------------------------------------

#[test]
fn concurrent_rapid_fire_operations_no_panic() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(tmp.path()).expect("open");
    db.create_collection_with_schema("c", &make_schema()).expect("create");

    let db = Arc::new(Mutex::new(db));
    let mut handles = Vec::new();

    for i in 0..20 {
        let db_clone = db.clone();
        handles.push(thread::spawn(move || {
            let mut db = db_clone.lock().expect("lock");
            match i % 4 {
                0 => {
                    // Insert.
                    let _ = db.insert_documents(
                        "c",
                        &[doc(
                            1000 + i,
                            &format!("rapid_{i}"),
                            i,
                            vec![i as f32; 4],
                        )],
                    );
                }
                1 => {
                    // Upsert.
                    let _ = db.upsert_documents(
                        "c",
                        &[doc(
                            2000 + i,
                            &format!("upsert_{i}"),
                            i,
                            vec![i as f32; 4],
                        )],
                    );
                }
                2 => {
                    // Search.
                    let _ = db.search("c", &[0.0; 4], 3);
                }
                3 => {
                    // Fetch.
                    let _ = db.fetch_documents("c", &[1000, 1001, 2000]);
                }
                _ => unreachable!(),
            }
        }));
    }

    for h in handles {
        h.join().expect("thread panicked");
    }

    // Verify collection is still functional after all operations.
    let db = Arc::try_unwrap(db)
        .ok()
        .expect("unwrap")
        .into_inner()
        .expect("inner");
    let _info = db.get_collection_info("c").expect("info should succeed");
}
