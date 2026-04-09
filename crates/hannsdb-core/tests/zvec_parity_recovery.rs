use std::fs;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use hannsdb_core::catalog::COLLECTION_RUNTIME_FORMAT_VERSION;
use hannsdb_core::db::HannsDb;
use hannsdb_core::document::{Document, FieldValue};
use hannsdb_core::segment::{
    append_payloads, append_record_ids, append_records, SegmentMetadata, SegmentSet, TombstoneMask,
    VersionSet,
};

fn unique_temp_dir(name: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("{}_{}", name, nanos))
}

fn doc(id: i64, vector: Vec<f32>) -> Document {
    Document::new(id, Vec::<(String, FieldValue)>::new(), vector)
}

fn write_segment(
    segment_dir: &std::path::Path,
    segment_id: &str,
    dimension: usize,
    documents: &[Document],
    deleted_rows: &[usize],
) {
    fs::create_dir_all(segment_dir).expect("create segment dir");

    let mut ids = Vec::with_capacity(documents.len());
    let mut vectors = Vec::with_capacity(documents.len() * dimension);
    let mut payloads = Vec::with_capacity(documents.len());
    for document in documents {
        ids.push(document.id);
        vectors.extend_from_slice(document.primary_vector());
        payloads.push(document.fields.clone());
    }

    let inserted =
        append_records(&segment_dir.join("records.bin"), dimension, &vectors).expect("write records");
    assert_eq!(inserted, documents.len(), "record count mismatch");
    let _ = append_record_ids(&segment_dir.join("ids.bin"), &ids).expect("write ids");
    let _ = append_payloads(&segment_dir.join("payloads.jsonl"), &payloads).expect("write payloads");

    let mut tombstone = TombstoneMask::new(documents.len());
    for &row in deleted_rows {
        assert!(tombstone.mark_deleted(row), "row index out of range");
    }
    tombstone
        .save_to_path(&segment_dir.join("tombstones.json"))
        .expect("write tombstones");

    SegmentMetadata::new(segment_id, dimension, documents.len(), tombstone.deleted_count())
        .save_to_path(&segment_dir.join("segment.json"))
        .expect("write segment metadata");
}

fn rewrite_collection_to_two_segment_layout(
    root: &std::path::Path,
    collection: &str,
    dimension: usize,
    second_segment_documents: &[Document],
    deleted_second_segment_rows: &[usize],
) {
    let collection_dir = root.join("collections").join(collection);
    let segments_dir = collection_dir.join("segments");
    let seg1_dir = segments_dir.join("seg-0001");
    let seg2_dir = segments_dir.join("seg-0002");
    fs::create_dir_all(&seg1_dir).expect("create seg-0001 dir");

    for file in [
        "segment.json",
        "records.bin",
        "ids.bin",
        "payloads.jsonl",
        "tombstones.json",
    ] {
        fs::rename(collection_dir.join(file), seg1_dir.join(file)).expect("move seg-0001 file");
    }

    write_segment(
        &seg2_dir,
        "seg-0002",
        dimension,
        second_segment_documents,
        deleted_second_segment_rows,
    );

    SegmentSet {
        active_segment_id: "seg-0002".to_string(),
        immutable_segment_ids: vec!["seg-0001".to_string()],
    }
    .save_to_path(&collection_dir.join("segment_set.json"))
    .expect("save segment_set metadata");
}

#[test]
fn zvec_parity_recovery_opening_same_collection_twice_reuses_same_handle() {
    let root = unique_temp_dir("hannsdb_zvec_handle_reuse");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");

    let handle_a = db.open_collection_handle("docs").expect("first handle");
    let handle_b = db.open_collection_handle("docs").expect("second handle");

    assert!(
        Arc::ptr_eq(&handle_a, &handle_b),
        "db should reuse the same collection handle"
    );
}

#[test]
fn zvec_parity_recovery_version_set_persists_format_metadata_on_disk() {
    let root = unique_temp_dir("hannsdb_zvec_version_set_persisted");
    fs::create_dir_all(&root).expect("create temp dir");
    let path = root.join("segment_set.json");

    let version_set = VersionSet::new("seg-0002", vec!["seg-0001".to_string()]);
    version_set.save_to_path(&path).expect("save version set");

    let persisted: serde_json::Value =
        serde_json::from_slice(&fs::read(&path).expect("read version set")).expect("parse json");
    assert_eq!(
        persisted.get("format_version").and_then(|value| value.as_u64()),
        Some(COLLECTION_RUNTIME_FORMAT_VERSION as u64)
    );

    let loaded = VersionSet::load_from_path(&path).expect("load version set");
    assert_eq!(loaded, version_set);
}

#[test]
fn zvec_parity_recovery_version_set_loads_legacy_segment_set_json() {
    let root = unique_temp_dir("hannsdb_zvec_version_set_legacy");
    fs::create_dir_all(&root).expect("create temp dir");
    let path = root.join("segment_set.json");

    SegmentSet {
        active_segment_id: "seg-0002".to_string(),
        immutable_segment_ids: vec!["seg-0001".to_string()],
    }
    .save_to_path(&path)
    .expect("save legacy segment_set");

    let loaded = VersionSet::load_from_path(&path).expect("load legacy version set");
    assert_eq!(
        loaded,
        VersionSet::new("seg-0002", vec!["seg-0001".to_string()])
    );
}

#[test]
fn zvec_parity_recovery_reopen_reads_multi_segment_version_metadata_through_handle() {
    let root = unique_temp_dir("hannsdb_zvec_reopen_version_set");
    {
        let mut db = HannsDb::open(&root).expect("open db");
        db.create_collection("docs", 2, "l2")
            .expect("create collection");
        db.insert("docs", &[1, 2], &[0.0_f32, 0.0, 1.0, 1.0])
            .expect("insert docs");
    }
    fs::remove_file(root.join("wal.jsonl")).expect("remove wal");

    let second_segment_docs = vec![doc(10, vec![0.1, 0.0]), doc(20, vec![0.2, 0.0])];
    rewrite_collection_to_two_segment_layout(&root, "docs", 2, &second_segment_docs, &[1]);

    let db = HannsDb::open(&root).expect("reopen db");
    let handle = db
        .open_collection_handle("docs")
        .expect("collection handle");
    let version_set = handle.version_set().expect("load version set");

    assert_eq!(
        version_set,
        VersionSet::from_segment_set(SegmentSet {
            active_segment_id: "seg-0002".to_string(),
            immutable_segment_ids: vec!["seg-0001".to_string()],
        })
    );

    assert_eq!(
        version_set.all_segment_ids(),
        vec!["seg-0002".to_string(), "seg-0001".to_string()]
    );

    fs::remove_file(root.join("collections").join("docs").join("segment_set.json"))
        .expect("remove segment_set after handle open");
    let cached_version_set = handle.version_set().expect("load cached version set");
    assert_eq!(cached_version_set, version_set);
}

#[test]
fn zvec_parity_recovery_collection_handle_supports_concurrent_search_and_optimize() {
    let root = unique_temp_dir("hannsdb_zvec_handle_concurrency");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");
    db.insert("docs", &[1, 2, 3], &[0.0_f32, 0.0, 0.1, 0.0, 1.0, 1.0])
        .expect("insert docs");

    let handle = db
        .open_collection_handle("docs")
        .expect("collection handle");
    let search_handle = Arc::clone(&handle);
    let optimize_handle = Arc::clone(&handle);
    let barrier = Arc::new(Barrier::new(2));

    let search_barrier = Arc::clone(&barrier);
    let search_thread = thread::spawn(move || {
        search_barrier.wait();
        let hits = search_handle
            .search_with_ef(&[0.0_f32, 0.0], 2, 32)
            .expect("search succeeds");
        hits.into_iter().map(|hit| hit.id).collect::<Vec<_>>()
    });

    let optimize_barrier = Arc::clone(&barrier);
    let optimize_thread = thread::spawn(move || {
        optimize_barrier.wait();
        optimize_handle.optimize().expect("optimize succeeds");
    });

    let hit_ids = search_thread.join().expect("join search thread");
    optimize_thread.join().expect("join optimize thread");

    assert_eq!(hit_ids, vec![1, 2]);
}
