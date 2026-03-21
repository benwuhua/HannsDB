use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use hannsdb_core::db::HannsDb;
use hannsdb_core::document::{
    CollectionSchema, Document, FieldType, FieldValue, ScalarFieldSchema,
};
use hannsdb_core::wal::{append_wal_record, load_wal_records, WalRecord};

fn sample_schema() -> CollectionSchema {
    CollectionSchema::new(
        "dense",
        2,
        "cosine",
        vec![
            ScalarFieldSchema::new("session_id", FieldType::String),
            ScalarFieldSchema::new("turn", FieldType::Int64),
            ScalarFieldSchema::new("active", FieldType::Bool),
        ],
    )
}

fn sample_document(id: i64) -> Document {
    Document::new(
        id,
        BTreeMap::from([
            (
                "session_id".to_string(),
                FieldValue::String("s1".to_string()),
            ),
            ("turn".to_string(), FieldValue::Int64(2)),
            ("active".to_string(), FieldValue::Bool(true)),
        ]),
        vec![0.1, 0.2],
    )
}

fn custom_document(
    id: i64,
    session_id: &str,
    turn: i64,
    active: bool,
    vector: Vec<f32>,
) -> Document {
    Document::new(
        id,
        BTreeMap::from([
            (
                "session_id".to_string(),
                FieldValue::String(session_id.to_string()),
            ),
            ("turn".to_string(), FieldValue::Int64(turn)),
            ("active".to_string(), FieldValue::Bool(active)),
        ]),
        vector,
    )
}

fn collection_dir(root: &Path, name: &str) -> std::path::PathBuf {
    root.join("collections").join(name)
}

#[test]
fn wal_recovery_record_roundtrip_preserves_all_operation_payloads() {
    let temp = tempfile::tempdir().expect("tempdir");
    let wal_path = temp.path().join("wal.jsonl");

    let records = vec![
        WalRecord::CreateCollection {
            collection: "docs".to_string(),
            schema: sample_schema(),
        },
        WalRecord::Insert {
            collection: "docs".to_string(),
            ids: vec![11, 22],
            vectors: vec![0.0, 0.1, 1.0, 1.1],
        },
        WalRecord::InsertDocuments {
            collection: "docs".to_string(),
            documents: vec![sample_document(33)],
        },
        WalRecord::UpsertDocuments {
            collection: "docs".to_string(),
            documents: vec![sample_document(22)],
        },
        WalRecord::Delete {
            collection: "docs".to_string(),
            ids: vec![11, 22],
        },
        WalRecord::DropCollection {
            collection: "docs".to_string(),
        },
    ];

    for record in &records {
        append_wal_record(&wal_path, record).expect("append wal record");
    }

    let loaded = load_wal_records(&wal_path).expect("load wal records");
    assert_eq!(loaded, records);
}

#[test]
fn wal_recovery_create_collection_record_keeps_full_schema() {
    let record = WalRecord::CreateCollection {
        collection: "docs".to_string(),
        schema: sample_schema(),
    };

    match record {
        WalRecord::CreateCollection { schema, .. } => {
            assert_eq!(schema.primary_vector, "dense");
            assert_eq!(schema.dimension, 2);
            assert_eq!(schema.metric, "cosine");
            assert_eq!(
                schema.fields,
                vec![
                    ScalarFieldSchema::new("session_id", FieldType::String),
                    ScalarFieldSchema::new("turn", FieldType::Int64),
                    ScalarFieldSchema::new("active", FieldType::Bool),
                ]
            );
        }
        other => panic!("unexpected record variant: {other:?}"),
    }
}

#[test]
fn wal_recovery_document_records_keep_typed_fields_and_vectors() {
    let record = WalRecord::UpsertDocuments {
        collection: "docs".to_string(),
        documents: vec![sample_document(42)],
    };

    match record {
        WalRecord::UpsertDocuments { documents, .. } => {
            assert_eq!(documents.len(), 1);
            assert_eq!(documents[0].id, 42);
            assert_eq!(documents[0].vector, vec![0.1, 0.2]);
            assert_eq!(
                documents[0].fields.get("session_id"),
                Some(&FieldValue::String("s1".to_string()))
            );
            assert_eq!(documents[0].fields.get("turn"), Some(&FieldValue::Int64(2)));
            assert_eq!(
                documents[0].fields.get("active"),
                Some(&FieldValue::Bool(true))
            );
        }
        other => panic!("unexpected record variant: {other:?}"),
    }
}

#[test]
fn wal_recovery_mutating_core_operations_append_records() {
    let temp = tempfile::tempdir().expect("tempdir");
    let wal_path = temp.path().join("wal.jsonl");
    let mut db = HannsDb::open(temp.path()).expect("open db");

    let schema = sample_schema();
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");
    db.insert_documents("docs", &[sample_document(11)])
        .expect("insert document");
    db.upsert_documents("docs", &[sample_document(11)])
        .expect("upsert document");
    db.delete("docs", &[11]).expect("delete document");

    let records = load_wal_records(&wal_path).expect("load wal records");
    assert_eq!(records.len(), 4);
    assert!(matches!(
        &records[0],
        WalRecord::CreateCollection { collection, .. } if collection == "docs"
    ));
    assert!(matches!(
        &records[1],
        WalRecord::InsertDocuments { collection, documents }
            if collection == "docs" && documents.len() == 1 && documents[0].id == 11
    ));
    assert!(matches!(
        &records[2],
        WalRecord::UpsertDocuments { collection, documents }
            if collection == "docs" && documents.len() == 1 && documents[0].id == 11
    ));
    assert!(matches!(
        &records[3],
        WalRecord::Delete { collection, ids } if collection == "docs" && ids == &vec![11]
    ));
}

#[test]
fn wal_recovery_open_replays_logged_operations_into_missing_storage() {
    let temp = tempfile::tempdir().expect("tempdir");
    let wal_path = temp.path().join("wal.jsonl");

    append_wal_record(
        &wal_path,
        &WalRecord::CreateCollection {
            collection: "docs".to_string(),
            schema: sample_schema(),
        },
    )
    .expect("append create");
    append_wal_record(
        &wal_path,
        &WalRecord::InsertDocuments {
            collection: "docs".to_string(),
            documents: vec![sample_document(11)],
        },
    )
    .expect("append insert");

    let db = HannsDb::open(temp.path()).expect("open db should replay wal");
    let info = db
        .get_collection_info("docs")
        .expect("collection should exist");
    assert_eq!(info.record_count, 1);
    assert_eq!(info.deleted_count, 0);
    assert_eq!(info.live_count, 1);

    let fetched = db
        .fetch_documents("docs", &[11])
        .expect("fetch replayed document");
    assert_eq!(fetched.len(), 1);
    assert_eq!(fetched[0].id, 11);
    assert_eq!(
        fetched[0].fields.get("session_id"),
        Some(&FieldValue::String("s1".to_string()))
    );
    assert_eq!(fetched[0].vector, vec![0.1, 0.2]);
}

#[test]
fn wal_recovery_open_replays_wal_owned_collection_when_payloads_are_missing() {
    let temp = tempfile::tempdir().expect("tempdir");
    let wal_path = temp.path().join("wal.jsonl");

    append_wal_record(
        &wal_path,
        &WalRecord::CreateCollection {
            collection: "docs".to_string(),
            schema: sample_schema(),
        },
    )
    .expect("append create");
    append_wal_record(
        &wal_path,
        &WalRecord::InsertDocuments {
            collection: "docs".to_string(),
            documents: vec![sample_document(11)],
        },
    )
    .expect("append insert");
    append_wal_record(
        &wal_path,
        &WalRecord::UpsertDocuments {
            collection: "docs".to_string(),
            documents: vec![custom_document(11, "s2", 7, false, vec![0.9, 0.8])],
        },
    )
    .expect("append upsert");

    let db = HannsDb::open(temp.path()).expect("initial replay");
    let live = db
        .fetch_documents("docs", &[11])
        .expect("fetch replayed document");
    assert_eq!(live.len(), 1);
    assert_eq!(live[0].id, 11);
    assert_eq!(live[0].vector, vec![0.9, 0.8]);
    assert_eq!(
        live[0].fields.get("session_id"),
        Some(&FieldValue::String("s2".to_string()))
    );
    assert_eq!(live[0].fields.get("turn"), Some(&FieldValue::Int64(7)));
    assert_eq!(
        live[0].fields.get("active"),
        Some(&FieldValue::Bool(false))
    );
    drop(db);

    let collection_dir = collection_dir(temp.path(), "docs");
    fs::remove_file(collection_dir.join("payloads.jsonl")).expect("remove payloads");

    let reopened = HannsDb::open(temp.path()).expect("reopen should replay wal");
    let info = reopened
        .get_collection_info("docs")
        .expect("collection should still exist");
    assert_eq!(info.record_count, 2);
    assert_eq!(info.deleted_count, 1);
    assert_eq!(info.live_count, 1);

    let replayed = reopened
        .fetch_documents("docs", &[11])
        .expect("fetch replayed document after recovery");
    assert_eq!(replayed.len(), 1);
    assert_eq!(replayed[0].id, 11);
    assert_eq!(replayed[0].vector, vec![0.9, 0.8]);
    assert_eq!(
        replayed[0].fields.get("session_id"),
        Some(&FieldValue::String("s2".to_string()))
    );
    assert_eq!(replayed[0].fields.get("turn"), Some(&FieldValue::Int64(7)));
    assert_eq!(
        replayed[0].fields.get("active"),
        Some(&FieldValue::Bool(false))
    );
    assert!(
        collection_dir.join("payloads.jsonl").exists(),
        "payloads file should be recreated by replay"
    );
}

#[test]
fn wal_recovery_open_replays_stale_partial_files_and_restores_latest_live_view() {
    let temp = tempfile::tempdir().expect("tempdir");
    let wal_path = temp.path().join("wal.jsonl");

    append_wal_record(
        &wal_path,
        &WalRecord::CreateCollection {
            collection: "docs".to_string(),
            schema: sample_schema(),
        },
    )
    .expect("append create");
    append_wal_record(
        &wal_path,
        &WalRecord::InsertDocuments {
            collection: "docs".to_string(),
            documents: vec![sample_document(11)],
        },
    )
    .expect("append insert");
    append_wal_record(
        &wal_path,
        &WalRecord::UpsertDocuments {
            collection: "docs".to_string(),
            documents: vec![custom_document(11, "s3", 9, true, vec![0.4, 0.6])],
        },
    )
    .expect("append upsert");

    let db = HannsDb::open(temp.path()).expect("initial replay");
    assert_eq!(
        db.fetch_documents("docs", &[11])
            .expect("fetch initial live document")
            .len(),
        1
    );
    drop(db);

    let collection_dir = collection_dir(temp.path(), "docs");
    fs::remove_file(collection_dir.join("records.bin")).expect("remove records");
    fs::write(collection_dir.join("stale.tmp"), b"stale partial storage")
        .expect("write stale file");

    let reopened = HannsDb::open(temp.path()).expect("reopen should replay wal");
    let info = reopened
        .get_collection_info("docs")
        .expect("collection should still exist");
    assert_eq!(info.record_count, 2);
    assert_eq!(info.deleted_count, 1);
    assert_eq!(info.live_count, 1);

    let replayed = reopened
        .fetch_documents("docs", &[11])
        .expect("fetch replayed document after cleanup");
    assert_eq!(replayed.len(), 1);
    assert_eq!(replayed[0].id, 11);
    assert_eq!(replayed[0].vector, vec![0.4, 0.6]);
    assert_eq!(
        replayed[0].fields.get("session_id"),
        Some(&FieldValue::String("s3".to_string()))
    );
    assert_eq!(replayed[0].fields.get("turn"), Some(&FieldValue::Int64(9)));
    assert_eq!(replayed[0].fields.get("active"), Some(&FieldValue::Bool(true)));
    assert!(
        !collection_dir.join("stale.tmp").exists(),
        "stale files should be removed when the collection is replayed"
    );
    assert!(
        collection_dir.join("records.bin").exists(),
        "records file should be recreated by replay"
    );
}

#[test]
fn wal_recovery_open_replays_wal_only_upsert_and_delete_into_latest_live_view() {
    let temp = tempfile::tempdir().expect("tempdir");
    let wal_path = temp.path().join("wal.jsonl");

    append_wal_record(
        &wal_path,
        &WalRecord::CreateCollection {
            collection: "docs".to_string(),
            schema: sample_schema(),
        },
    )
    .expect("append create");
    append_wal_record(
        &wal_path,
        &WalRecord::UpsertDocuments {
            collection: "docs".to_string(),
            documents: vec![custom_document(11, "s1", 7, false, vec![0.9, 0.8])],
        },
    )
    .expect("append upsert");
    append_wal_record(
        &wal_path,
        &WalRecord::Delete {
            collection: "docs".to_string(),
            ids: vec![11],
        },
    )
    .expect("append delete");

    let db = HannsDb::open(temp.path()).expect("open db should replay wal");
    let info = db
        .get_collection_info("docs")
        .expect("collection should exist");
    assert_eq!(info.record_count, 1);
    assert_eq!(info.deleted_count, 1);
    assert_eq!(info.live_count, 0);
    assert!(
        db.fetch_documents("docs", &[11])
            .expect("fetch replayed document")
            .is_empty()
    );
}

#[test]
fn wal_recovery_open_replays_drop_collection_without_leaving_manifest_or_dir() {
    let temp = tempfile::tempdir().expect("tempdir");
    let wal_path = temp.path().join("wal.jsonl");

    let mut db = HannsDb::open(temp.path()).expect("open db");
    db.create_collection_with_schema("docs", &sample_schema())
        .expect("create collection");
    drop(db);

    append_wal_record(
        &wal_path,
        &WalRecord::DropCollection {
            collection: "docs".to_string(),
        },
    )
    .expect("append drop");

    let reopened = HannsDb::open(temp.path()).expect("reopen should replay drop");
    assert!(
        reopened
            .list_collections()
            .expect("list collections after reopen")
            .is_empty()
    );
    assert!(
        !collection_dir(temp.path(), "docs").exists(),
        "collection dir should be removed after replay"
    );
}

#[test]
fn wal_recovery_open_does_not_duplicate_rows_after_replay() {
    let temp = tempfile::tempdir().expect("tempdir");
    let wal_path = temp.path().join("wal.jsonl");

    append_wal_record(
        &wal_path,
        &WalRecord::CreateCollection {
            collection: "docs".to_string(),
            schema: sample_schema(),
        },
    )
    .expect("append create");
    append_wal_record(
        &wal_path,
        &WalRecord::InsertDocuments {
            collection: "docs".to_string(),
            documents: vec![sample_document(11)],
        },
    )
    .expect("append insert");

    let db = HannsDb::open(temp.path()).expect("first open should replay wal");
    let first_info = db
        .get_collection_info("docs")
        .expect("collection should exist after first replay");
    assert_eq!(first_info.record_count, 1);
    drop(db);

    let reopened = HannsDb::open(temp.path()).expect("second open should not duplicate rows");
    let second_info = reopened
        .get_collection_info("docs")
        .expect("collection should still exist after reopen");
    assert_eq!(second_info.record_count, 1);
    assert_eq!(second_info.deleted_count, 0);
    assert_eq!(second_info.live_count, 1);
}

#[test]
fn wal_recovery_crash_missing_records_bin_replays_and_restores_search() {
    let temp = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(temp.path()).expect("open db");
    db.create_collection_with_schema("docs", &sample_schema())
        .expect("create collection");
    db.insert_documents(
        "docs",
        &[
            custom_document(11, "s1", 1, true, vec![0.1, 0.2]),
            custom_document(22, "s1", 2, true, vec![10.0, 10.0]),
        ],
    )
    .expect("insert documents");
    drop(db);

    let dir = collection_dir(temp.path(), "docs");
    fs::remove_file(dir.join("records.bin")).expect("remove records.bin");

    let reopened = HannsDb::open(temp.path()).expect("reopen should replay wal");
    let hits = reopened
        .search("docs", &[0.1, 0.2], 1)
        .expect("search after replay");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, 11);
    assert!(
        dir.join("records.bin").exists(),
        "records.bin should be recreated by replay"
    );
}

#[test]
fn wal_recovery_crash_missing_segment_meta_replays_and_restores_collection() {
    let temp = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(temp.path()).expect("open db");
    db.create_collection_with_schema("docs", &sample_schema())
        .expect("create collection");
    db.insert_documents("docs", &[sample_document(11)])
        .expect("insert document");
    drop(db);

    let dir = collection_dir(temp.path(), "docs");
    fs::remove_file(dir.join("segment.json")).expect("remove segment.json");

    let reopened = HannsDb::open(temp.path()).expect("reopen should replay wal");
    let info = reopened
        .get_collection_info("docs")
        .expect("collection info after replay");
    assert_eq!(info.record_count, 1);
    assert_eq!(info.deleted_count, 0);
    assert_eq!(info.live_count, 1);
    assert!(
        dir.join("segment.json").exists(),
        "segment.json should be recreated by replay"
    );
}

#[test]
fn wal_recovery_crash_truncated_wal_tail_is_ignored_and_open_succeeds() {
    use std::io::Write;

    let temp = tempfile::tempdir().expect("tempdir");
    let wal_path = temp.path().join("wal.jsonl");
    append_wal_record(
        &wal_path,
        &WalRecord::CreateCollection {
            collection: "docs".to_string(),
            schema: sample_schema(),
        },
    )
    .expect("append create");
    append_wal_record(
        &wal_path,
        &WalRecord::InsertDocuments {
            collection: "docs".to_string(),
            documents: vec![sample_document(11)],
        },
    )
    .expect("append insert");

    let mut wal = std::fs::OpenOptions::new()
        .append(true)
        .open(&wal_path)
        .expect("open wal for append");
    wal.write_all(br#"{"InsertDocuments":{"collection":"docs","documents":["#)
        .expect("append truncated json");

    let reopened = HannsDb::open(temp.path()).expect("open should ignore truncated tail wal line");
    let info = reopened
        .get_collection_info("docs")
        .expect("collection should be available");
    assert_eq!(info.record_count, 1);
    assert_eq!(info.deleted_count, 0);
    assert_eq!(info.live_count, 1);
    let fetched = reopened
        .fetch_documents("docs", &[11])
        .expect("fetch replayed document");
    assert_eq!(fetched.len(), 1);
    assert_eq!(fetched[0].id, 11);
}

#[test]
fn wal_recovery_crash_missing_tombstones_replays_delete_state() {
    let temp = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(temp.path()).expect("open db");
    db.create_collection_with_schema("docs", &sample_schema())
        .expect("create collection");
    db.insert_documents(
        "docs",
        &[
            custom_document(11, "s1", 1, true, vec![0.1, 0.2]),
            custom_document(22, "s1", 2, true, vec![5.0, 5.0]),
        ],
    )
    .expect("insert documents");
    db.delete("docs", &[11]).expect("delete id 11");
    drop(db);

    let dir = collection_dir(temp.path(), "docs");
    fs::remove_file(dir.join("tombstones.json")).expect("remove tombstones");

    let reopened = HannsDb::open(temp.path()).expect("reopen should replay wal");
    let info = reopened
        .get_collection_info("docs")
        .expect("collection info after replay");
    assert_eq!(info.record_count, 2);
    assert_eq!(info.deleted_count, 1);
    assert_eq!(info.live_count, 1);

    let deleted = reopened
        .fetch_documents("docs", &[11])
        .expect("fetch deleted id after replay");
    assert!(deleted.is_empty());

    let hits = reopened
        .search("docs", &[0.1, 0.2], 2)
        .expect("search after tombstone replay");
    assert!(hits.iter().all(|hit| hit.id != 11));
    assert!(
        dir.join("tombstones.json").exists(),
        "tombstones should be recreated by replay"
    );
}
