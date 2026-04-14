use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use hannsdb_core::db::HannsDb;
use hannsdb_core::document::{
    CollectionSchema, Document, FieldType, FieldValue, ScalarFieldSchema,
};
use hannsdb_core::segment::{
    append_payloads, append_record_ids, append_records, SegmentMetadata, SegmentSet, TombstoneMask,
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
    Document::with_primary_vector_name(
        id,
        BTreeMap::from([
            (
                "session_id".to_string(),
                FieldValue::String("s1".to_string()),
            ),
            ("turn".to_string(), FieldValue::Int64(2)),
            ("active".to_string(), FieldValue::Bool(true)),
        ]),
        "dense",
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
    Document::with_primary_vector_name(
        id,
        BTreeMap::from([
            (
                "session_id".to_string(),
                FieldValue::String(session_id.to_string()),
            ),
            ("turn".to_string(), FieldValue::Int64(turn)),
            ("active".to_string(), FieldValue::Bool(active)),
        ]),
        "dense",
        vector,
    )
}

fn collection_dir(root: &Path, name: &str) -> std::path::PathBuf {
    root.join("collections").join(name)
}

fn storage_dir_with_file(root: &Path, name: &str, file_name: &str) -> std::path::PathBuf {
    let collection_dir = collection_dir(root, name);
    let root_candidate = collection_dir.join(file_name);
    if root_candidate.exists() {
        return collection_dir;
    }

    let segment_set_path = collection_dir.join("segment_set.json");
    if segment_set_path.exists() {
        let segment_set = SegmentSet::load_from_path(&segment_set_path)
            .expect("load segment_set for file lookup");
        let candidates = segment_set
            .immutable_segment_ids
            .iter()
            .rev()
            .cloned()
            .chain(std::iter::once(segment_set.active_segment_id));
        for segment_id in candidates {
            let dir = collection_dir.join("segments").join(&segment_id);
            if dir.join(file_name).exists() {
                return dir;
            }
        }
    }

    collection_dir
}

fn storage_file_path(root: &Path, name: &str, file_names: &[&str]) -> std::path::PathBuf {
    for file_name in file_names {
        let dir = storage_dir_with_file(root, name, file_name);
        let path = dir.join(file_name);
        if path.exists() {
            return path;
        }
    }
    storage_dir_with_file(root, name, file_names[0]).join(file_names[0])
}

fn rewrite_collection_to_two_segment_layout(
    root: &Path,
    collection: &str,
    dimension: usize,
    second_segment_documents: &[Document],
    deleted_second_segment_rows: &[usize],
) {
    let collection_dir = collection_dir(root, collection);
    let segments_dir = collection_dir.join("segments");
    let seg1_dir = segments_dir.join("seg-0001");
    let seg2_dir = segments_dir.join("seg-0002");
    fs::create_dir_all(&seg1_dir).expect("create seg-0001 dir");
    fs::create_dir_all(&seg2_dir).expect("create seg-0002 dir");

    for file in [
        "segment.json",
        "records.bin",
        "ids.bin",
        "payloads.jsonl",
        "vectors.jsonl",
        "tombstones.json",
    ] {
        let source = collection_dir.join(file);
        if source.exists() {
            fs::rename(source, seg1_dir.join(file)).expect("move seg-0001 file");
        }
    }

    let mut second_ids = Vec::with_capacity(second_segment_documents.len());
    let mut second_vectors = Vec::with_capacity(second_segment_documents.len() * dimension);
    let mut second_payloads = Vec::with_capacity(second_segment_documents.len());
    for document in second_segment_documents {
        second_ids.push(document.id);
        second_vectors.extend_from_slice(document.primary_vector_for("dense").unwrap());
        second_payloads.push(document.fields.clone());
    }

    let inserted = append_records(&seg2_dir.join("records.bin"), dimension, &second_vectors)
        .expect("append seg-0002 records");
    assert_eq!(inserted, second_segment_documents.len());
    let _ = append_record_ids(&seg2_dir.join("ids.bin"), &second_ids).expect("append seg-0002 ids");
    let _ = append_payloads(&seg2_dir.join("payloads.jsonl"), &second_payloads)
        .expect("append seg-0002 payloads");
    let _ = hannsdb_core::segment::append_vectors(
        &seg2_dir.join("vectors.jsonl"),
        &vec![BTreeMap::new(); second_segment_documents.len()],
    )
    .expect("append seg-0002 vectors");

    let mut seg2_tombstone = TombstoneMask::new(second_segment_documents.len());
    for row_idx in deleted_second_segment_rows {
        let marked = seg2_tombstone.mark_deleted(*row_idx);
        assert!(marked, "row index must be valid in seg-0002 tombstone");
    }
    seg2_tombstone
        .save_to_path(&seg2_dir.join("tombstones.json"))
        .expect("save seg-0002 tombstones");

    SegmentMetadata::new(
        "seg-0002",
        dimension,
        second_segment_documents.len(),
        seg2_tombstone.deleted_count(),
    )
    .save_to_path(&seg2_dir.join("segment.json"))
    .expect("save seg-0002 metadata");

    SegmentSet {
        active_segment_id: "seg-0002".to_string(),
        immutable_segment_ids: vec!["seg-0001".to_string()],
    }
    .save_to_path(&collection_dir.join("segment_set.json"))
    .expect("save segment_set metadata");
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
            assert_eq!(schema.primary_vector_name(), "dense");
            assert_eq!(schema.dimension(), 2);
            assert_eq!(schema.metric(), "cosine");
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
            assert_eq!(
                documents[0].primary_vector_for("dense").unwrap(),
                &[0.1, 0.2]
            );
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
fn wal_recovery_delete_by_filter_appends_one_delete_record() {
    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path();
    let mut db = HannsDb::open(root).expect("open db");

    let schema = sample_schema();
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");

    let docs = vec![sample_document(11), sample_document(22)];
    db.insert_documents("docs", &docs)
        .expect("insert documents");

    let deleted = db
        .delete_by_filter("docs", "session_id == \"s1\"")
        .expect("delete by filter");
    assert_eq!(deleted, 2);

    let records = load_wal_records(&root.join("wal.jsonl")).expect("load wal records");
    assert_eq!(records.len(), 3);
    assert_eq!(
        records[2],
        WalRecord::Delete {
            collection: "docs".to_string(),
            ids: vec![11, 22],
        }
    );
}

#[test]
fn wal_recovery_replays_delete_by_filter_outcome() {
    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path();
    let mut db = HannsDb::open(root).expect("open db");

    let schema = sample_schema();
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");

    let docs = vec![sample_document(11), sample_document(22)];
    db.insert_documents("docs", &docs)
        .expect("insert documents");

    let deleted = db
        .delete_by_filter("docs", "session_id == \"s1\"")
        .expect("delete by filter");
    assert_eq!(deleted, 2);
    drop(db);

    fs::remove_file(collection_dir(root, "docs").join("tombstones.json"))
        .expect("remove tombstones");

    let reopened = HannsDb::open(root).expect("reopen should replay wal");
    let info = reopened
        .get_collection_info("docs")
        .expect("collection info after replay");
    assert_eq!(info.record_count, 2);
    assert_eq!(info.deleted_count, 2);
    assert_eq!(info.live_count, 0);

    let replayed = reopened
        .fetch_documents("docs", &[11, 22])
        .expect("fetch after replay");
    assert!(
        replayed.is_empty(),
        "delete_by_filter outcome must survive WAL replay"
    );
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
    assert_eq!(fetched[0].primary_vector_for("dense").unwrap(), &[0.1, 0.2]);
}

#[test]
fn wal_recovery_replays_segment_aware_delete_outcome() {
    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path();

    let mut db = HannsDb::open(root).expect("open db");
    db.create_collection_with_schema("docs", &sample_schema())
        .expect("create collection");

    let inserted_docs = vec![
        custom_document(10, "s1", 1, true, vec![0.1, 0.2]),
        custom_document(20, "s1", 2, true, vec![0.2, 0.3]),
        custom_document(30, "s1", 3, true, vec![0.3, 0.4]),
        custom_document(40, "s1", 4, true, vec![0.4, 0.5]),
    ];
    db.insert_documents("docs", &inserted_docs)
        .expect("insert docs");

    let second_segment_docs = vec![
        custom_document(10, "s2", 5, false, vec![0.9, 0.8]),
        custom_document(20, "s2", 6, false, vec![0.8, 0.7]),
    ];
    rewrite_collection_to_two_segment_layout(root, "docs", 2, &second_segment_docs, &[1]);

    let deleted = db.delete("docs", &[10, 20]).expect("delete ids");
    assert_eq!(deleted, 1);
    drop(db);

    let active_segment_dir = collection_dir(root, "docs")
        .join("segments")
        .join("seg-0002");
    fs::remove_file(active_segment_dir.join("tombstones.json")).expect("remove active tombstones");

    let reopened = HannsDb::open(root).expect("reopen should replay wal");
    let info = reopened
        .get_collection_info("docs")
        .expect("collection info after replay");
    assert_eq!(info.record_count, 4);
    assert_eq!(info.deleted_count, 2);
    assert_eq!(info.live_count, 2);

    let replayed = reopened
        .fetch_documents("docs", &[10, 20, 30, 40])
        .expect("fetch after replay");
    let replayed_ids = replayed
        .into_iter()
        .map(|document| document.id)
        .collect::<Vec<_>>();
    assert_eq!(replayed_ids, vec![30, 40]);
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
    assert_eq!(live[0].primary_vector_for("dense").unwrap(), &[0.9, 0.8]);
    assert_eq!(
        live[0].fields.get("session_id"),
        Some(&FieldValue::String("s2".to_string()))
    );
    assert_eq!(live[0].fields.get("turn"), Some(&FieldValue::Int64(7)));
    assert_eq!(live[0].fields.get("active"), Some(&FieldValue::Bool(false)));
    drop(db);

    let payload_path =
        storage_file_path(temp.path(), "docs", &["payloads.jsonl", "payloads.arrow"]);
    fs::remove_file(&payload_path).expect("remove payloads");

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
    assert_eq!(
        replayed[0].primary_vector_for("dense").unwrap(),
        &[0.9, 0.8]
    );
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
        storage_file_path(temp.path(), "docs", &["payloads.jsonl", "payloads.arrow"]).exists(),
        "payload file should be recreated by replay"
    );
}

#[test]
fn wal_recovery_open_skips_replay_when_arrow_snapshot_can_authoritatively_reopen_collection() {
    let temp = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(temp.path()).expect("open db");
    db.create_collection_with_schema("docs", &sample_schema())
        .expect("create collection");
    db.insert_documents("docs", &[sample_document(11)])
        .expect("insert document");
    db.flush_collection("docs")
        .expect("flush should materialize active-segment arrows");
    drop(db);

    let collection_dir = collection_dir(temp.path(), "docs");
    let payloads_jsonl = collection_dir.join("payloads.jsonl");
    let vectors_jsonl = collection_dir.join("vectors.jsonl");
    assert!(
        collection_dir.join("payloads.arrow").exists(),
        "flush should create payloads.arrow before removing jsonl sidecars"
    );
    assert!(
        collection_dir.join("vectors.arrow").exists(),
        "flush should create vectors.arrow before removing jsonl sidecars"
    );
    fs::remove_file(&payloads_jsonl).expect("remove payloads jsonl");
    fs::remove_file(&vectors_jsonl).expect("remove vectors jsonl");

    let reopened = HannsDb::open(temp.path()).expect("reopen from authoritative arrows");
    let fetched = reopened
        .fetch_documents("docs", &[11])
        .expect("fetch from arrow-authoritative reopen");
    assert_eq!(fetched.len(), 1);
    assert_eq!(fetched[0].id, 11);
    assert!(
        !payloads_jsonl.exists(),
        "reopen should not replay WAL and recreate payloads.jsonl when Arrow snapshots are authoritative"
    );
    assert!(
        !vectors_jsonl.exists(),
        "reopen should not replay WAL and recreate vectors.jsonl when Arrow snapshots are authoritative"
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

    let records_dir = storage_dir_with_file(temp.path(), "docs", "records.bin");
    fs::remove_file(records_dir.join("records.bin")).expect("remove records");
    fs::write(records_dir.join("stale.tmp"), b"stale partial storage").expect("write stale file");

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
    assert_eq!(
        replayed[0].primary_vector_for("dense").unwrap(),
        &[0.4, 0.6]
    );
    assert_eq!(
        replayed[0].fields.get("session_id"),
        Some(&FieldValue::String("s3".to_string()))
    );
    assert_eq!(replayed[0].fields.get("turn"), Some(&FieldValue::Int64(9)));
    assert_eq!(
        replayed[0].fields.get("active"),
        Some(&FieldValue::Bool(true))
    );
    assert!(
        !storage_dir_with_file(temp.path(), "docs", "records.bin")
            .join("stale.tmp")
            .exists(),
        "stale files should be removed when the collection is replayed"
    );
    assert!(
        storage_dir_with_file(temp.path(), "docs", "records.bin")
            .join("records.bin")
            .exists(),
        "records file should be recreated by replay"
    );
}

#[test]
fn wal_recovery_replays_segmented_post_rollover_latest_live_view() {
    let temp = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(temp.path()).expect("open db");
    db.create_collection_with_schema("docs", &sample_schema())
        .expect("create collection");

    let ids: Vec<i64> = (0..100).collect();
    let documents = ids
        .iter()
        .map(|id| custom_document(*id, "s1", *id, true, vec![*id as f32, 0.0]))
        .collect::<Vec<_>>();
    db.insert_documents("docs", &documents)
        .expect("insert docs");
    db.delete("docs", &(0..21).collect::<Vec<_>>())
        .expect("delete docs");
    db.insert_documents(
        "docs",
        &[custom_document(200, "s2", 200, true, vec![0.0, 0.0])],
    )
    .expect("trigger rollover");
    db.insert_documents(
        "docs",
        &[custom_document(201, "s3", 201, false, vec![0.1, 0.1])],
    )
    .expect("insert after rollover");
    drop(db);

    let active_records =
        storage_dir_with_file(temp.path(), "docs", "records.bin").join("records.bin");
    fs::remove_file(&active_records).expect("remove active records to force replay");

    let reopened = HannsDb::open(temp.path()).expect("reopen should replay segmented rollover");
    let fetched = reopened
        .fetch_documents("docs", &[200, 201])
        .expect("fetch replayed post-rollover docs");
    let fetched_ids = fetched
        .iter()
        .map(|document| document.id)
        .collect::<Vec<_>>();
    assert_eq!(fetched_ids, vec![200, 201]);
    assert_eq!(
        fetched[0].fields.get("session_id"),
        Some(&FieldValue::String("s2".into()))
    );
    assert_eq!(
        fetched[1].fields.get("session_id"),
        Some(&FieldValue::String("s3".into()))
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
    assert!(db
        .fetch_documents("docs", &[11])
        .expect("fetch replayed document")
        .is_empty());
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
    assert!(reopened
        .list_collections()
        .expect("list collections after reopen")
        .is_empty());
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

// ── WAL truncation / checkpointing tests ──────────────────────────────────

#[test]
fn wal_truncate_removes_all_wal_entries_after_flush() {
    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path();
    let wal_path = root.join("wal.jsonl");

    let mut db = HannsDb::open(root).expect("open db");
    db.create_collection_with_schema("docs", &sample_schema())
        .expect("create collection");
    db.insert_documents("docs", &[sample_document(11)])
        .expect("insert document");

    // WAL should have entries before flush.
    let records_before = load_wal_records(&wal_path).expect("load wal before flush");
    assert!(
        !records_before.is_empty(),
        "WAL should have entries before flush"
    );

    db.flush_collection("docs").expect("flush collection");

    // WAL file should still exist but be empty after flush.
    assert!(wal_path.exists(), "WAL file should still exist after flush");
    let records_after = load_wal_records(&wal_path).expect("load wal after flush");
    assert!(
        records_after.is_empty(),
        "WAL should be empty after flush, got {} entries",
        records_after.len()
    );
}

#[test]
fn wal_truncate_data_survives_reopen_after_flush() {
    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path();

    {
        let mut db = HannsDb::open(root).expect("open db");
        db.create_collection_with_schema("docs", &sample_schema())
            .expect("create collection");
        db.insert_documents("docs", &[sample_document(11), sample_document(22)])
            .expect("insert documents");
        db.delete("docs", &[11]).expect("delete document");
        db.flush_collection("docs").expect("flush collection");
    }

    // Reopen should work without WAL replay since all data is on disk.
    let reopened = HannsDb::open(root).expect("reopen db after flush");
    let info = reopened
        .get_collection_info("docs")
        .expect("collection info");
    assert_eq!(info.record_count, 2);
    assert_eq!(info.deleted_count, 1);
    assert_eq!(info.live_count, 1);

    let fetched = reopened
        .fetch_documents("docs", &[22])
        .expect("fetch surviving document");
    assert_eq!(fetched.len(), 1);
    assert_eq!(fetched[0].id, 22);
}

#[test]
fn wal_truncate_subsequent_writes_work_after_flush() {
    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path();
    let wal_path = root.join("wal.jsonl");

    let mut db = HannsDb::open(root).expect("open db");
    db.create_collection_with_schema("docs", &sample_schema())
        .expect("create collection");
    db.insert_documents("docs", &[sample_document(11)])
        .expect("insert first document");
    db.flush_collection("docs").expect("flush collection");

    // WAL should be empty after flush.
    assert!(load_wal_records(&wal_path)
        .expect("load wal after flush")
        .is_empty());

    // New writes should append to the WAL normally.
    db.insert_documents("docs", &[sample_document(22)])
        .expect("insert second document");

    let records = load_wal_records(&wal_path).expect("load wal after new write");
    assert_eq!(
        records.len(),
        1,
        "WAL should have exactly 1 entry for the new insert"
    );
    assert!(matches!(
        &records[0],
        WalRecord::InsertDocuments { collection, .. } if collection == "docs"
    ));
}

#[test]
fn wal_truncate_optimize_truncates_wal() {
    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path();
    let wal_path = root.join("wal.jsonl");

    let mut db = HannsDb::open(root).expect("open db");
    db.create_collection_with_schema("docs", &sample_schema())
        .expect("create collection");
    db.insert_documents("docs", &[sample_document(11)])
        .expect("insert document");

    // WAL should have entries before optimize.
    assert!(!load_wal_records(&wal_path)
        .expect("load wal before optimize")
        .is_empty());

    db.optimize_collection("docs").expect("optimize collection");

    // WAL should be empty after optimize.
    assert!(
        load_wal_records(&wal_path)
            .expect("load wal after optimize")
            .is_empty(),
        "WAL should be empty after optimize"
    );
}

#[cfg(feature = "hanns-backend")]
#[test]
fn wal_truncate_optimize_preserves_ann_completeness_after_reopen() {
    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path();

    {
        let mut db = HannsDb::open(root).expect("open db");
        db.create_collection_with_schema("docs", &sample_schema())
            .expect("create collection");
        db.insert_documents(
            "docs",
            &[
                custom_document(11, "s1", 1, true, vec![0.1, 0.2]),
                custom_document(22, "s2", 2, true, vec![0.9, 0.8]),
            ],
        )
        .expect("insert documents");
        db.optimize_collection("docs").expect("optimize collection");
    }

    let wal_records = load_wal_records(&root.join("wal.jsonl")).expect("load wal");
    assert!(
        wal_records.is_empty(),
        "optimize should truncate wal before reopen"
    );

    let reopened = HannsDb::open(root).expect("reopen after optimize");
    let info = reopened
        .get_collection_info("docs")
        .expect("collection info after reopen");
    assert_eq!(info.record_count, 2);
    assert_eq!(info.deleted_count, 0);
    assert_eq!(info.live_count, 2);
    assert_eq!(info.index_completeness.get("dense"), Some(&1.0));

    let hits = reopened
        .search("docs", &[0.1_f32, 0.2], 1)
        .expect("search should use persisted ann after reopen");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, 11);
}

#[test]
fn wal_truncate_flush_then_reopen_then_insert_search_works() {
    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path();

    // Phase 1: create, insert, flush, close.
    {
        let mut db = HannsDb::open(root).expect("open db phase 1");
        db.create_collection_with_schema("docs", &sample_schema())
            .expect("create");
        db.insert_documents(
            "docs",
            &[
                custom_document(1, "a", 1, true, vec![1.0, 0.0]),
                custom_document(2, "b", 2, false, vec![0.0, 1.0]),
            ],
        )
        .expect("insert");
        db.flush_collection("docs").expect("flush");
    }

    // Phase 2: reopen, verify data, insert more, close.
    {
        let mut db = HannsDb::open(root).expect("open db phase 2");
        let info = db.get_collection_info("docs").expect("info");
        assert_eq!(info.live_count, 2);

        db.insert_documents("docs", &[custom_document(3, "c", 3, true, vec![0.5, 0.5])])
            .expect("insert more");

        let hits = db.search("docs", &[0.5, 0.5], 1).expect("search");
        assert_eq!(hits[0].id, 3);
        db.flush_collection("docs").expect("flush");
    }

    // Phase 3: reopen again, verify all data.
    let db = HannsDb::open(root).expect("open db phase 3");
    let info = db.get_collection_info("docs").expect("info");
    assert_eq!(info.live_count, 3);

    let fetched = db.fetch_documents("docs", &[1, 2, 3]).expect("fetch all");
    assert_eq!(fetched.len(), 3);
}
