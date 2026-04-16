use std::fs;
use std::path::Path;

use hannsdb_core::db::HannsDb;
use hannsdb_core::document::{
    CollectionSchema, Document, FieldType, FieldValue, ScalarFieldSchema,
};
use hannsdb_core::forward_store::{ForwardFileFormat, ForwardStoreDescriptor, ForwardStoreReader};
use hannsdb_core::segment::{
    append_payloads, append_record_ids, append_records, write_payloads_arrow, SegmentMetadata,
    SegmentSet, TombstoneMask,
};
use hannsdb_core::wal::{append_wal_record, truncate_wal, WalRecord};

fn doc(id: i64, vector: Vec<f32>) -> Document {
    Document::new(id, Vec::<(String, FieldValue)>::new(), vector)
}

fn doc_with_field(
    id: i64,
    vector: Vec<f32>,
    field_name: &str,
    field_value: FieldValue,
) -> Document {
    let mut fields = Vec::new();
    fields.push((field_name.to_string(), field_value));
    Document::new(id, fields, vector)
}

fn write_segment(
    segment_dir: &Path,
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

    let inserted = append_records(&segment_dir.join("records.bin"), dimension, &vectors)
        .expect("write records");
    assert_eq!(inserted, documents.len(), "record count must match docs");
    let _ = append_record_ids(&segment_dir.join("ids.bin"), &ids).expect("write ids");
    let _ =
        append_payloads(&segment_dir.join("payloads.jsonl"), &payloads).expect("write payloads");
    let _ = hannsdb_core::segment::append_vectors(
        &segment_dir.join("vectors.jsonl"),
        &vec![std::collections::BTreeMap::new(); documents.len()],
    )
    .expect("write vectors");

    let mut tombstone = TombstoneMask::new(documents.len());
    for row_idx in deleted_rows {
        assert!(
            tombstone.mark_deleted(*row_idx),
            "deleted row index must be in range"
        );
    }
    tombstone
        .save_to_path(&segment_dir.join("tombstones.json"))
        .expect("write tombstones");

    SegmentMetadata::new(
        segment_id,
        dimension,
        documents.len(),
        tombstone.deleted_count(),
    )
    .save_to_path(&segment_dir.join("segment.json"))
    .expect("write segment metadata");
}

fn collection_segments_dir(root: &Path, collection: &str) -> std::path::PathBuf {
    root.join("collections").join(collection).join("segments")
}

fn load_forward_store_descriptor(path: &Path) -> ForwardStoreDescriptor {
    serde_json::from_slice(&fs::read(path).expect("read persisted forward_store descriptor json"))
        .expect("parse persisted forward_store descriptor")
}

#[test]
fn compaction_merges_multiple_immutable_segments_and_removes_old_dirs() {
    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path();
    let mut db = HannsDb::open(root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");

    let segments_dir = collection_segments_dir(root, "docs");
    write_segment(
        &segments_dir.join("seg-000001"),
        "seg-000001",
        2,
        &[doc(1, vec![0.0, 0.0]), doc(2, vec![1.0, 0.0])],
        &[],
    );
    write_segment(
        &segments_dir.join("seg-000002"),
        "seg-000002",
        2,
        &[doc(3, vec![0.0, 1.0]), doc(4, vec![2.0, 0.0])],
        &[],
    );
    write_segment(
        &segments_dir.join("seg-000003"),
        "seg-000003",
        2,
        &[doc(10, vec![10.0, 10.0])],
        &[],
    );

    SegmentSet {
        active_segment_id: "seg-000003".to_string(),
        immutable_segment_ids: vec!["seg-000001".to_string(), "seg-000002".to_string()],
    }
    .save_to_path(
        &root
            .join("collections")
            .join("docs")
            .join("segment_set.json"),
    )
    .expect("write segment_set");

    db.compact_collection("docs").expect("compact collection");

    assert!(
        !segments_dir.join("seg-000001").exists(),
        "old immutable seg-000001 must be removed"
    );
    assert!(
        !segments_dir.join("seg-000002").exists(),
        "old immutable seg-000002 must be removed"
    );

    let segment_set = SegmentSet::load_from_path(
        &root
            .join("collections")
            .join("docs")
            .join("segment_set.json"),
    )
    .expect("load segment_set");
    assert_eq!(segment_set.active_segment_id, "seg-000003");
    assert_eq!(segment_set.immutable_segment_ids.len(), 1);

    let compacted_id = &segment_set.immutable_segment_ids[0];
    let compacted_meta =
        SegmentMetadata::load_from_path(&segments_dir.join(compacted_id).join("segment.json"))
            .expect("load compacted metadata");
    assert_eq!(compacted_meta.record_count, 4);
    assert_eq!(compacted_meta.deleted_count, 0);
    assert_eq!(compacted_meta.storage_format, "forward_store");

    let compacted_dir = segments_dir.join(compacted_id);
    let descriptor = load_forward_store_descriptor(&compacted_dir.join("forward_store.json"));
    assert_eq!(descriptor.row_count, 4);
    assert!(
        descriptor.artifact(ForwardFileFormat::ArrowIpc).is_some(),
        "compacted segment should persist Arrow IPC forward_store artifact"
    );
    assert!(
        descriptor.artifact(ForwardFileFormat::Parquet).is_some(),
        "compacted segment should persist Parquet forward_store artifact"
    );
    let reader = ForwardStoreReader::open(&descriptor, ForwardFileFormat::Parquet)
        .expect("open compacted forward_store parquet snapshot");
    let latest_live_ids = reader
        .latest_live_rows()
        .into_iter()
        .map(|row| row.internal_id as i64)
        .collect::<Vec<_>>();
    assert_eq!(
        latest_live_ids,
        vec![1, 2, 3, 4],
        "compacted forward_store snapshot should preserve the immutable latest-live set"
    );

    let hits = db
        .search("docs", &[0.0, 0.0], 10)
        .expect("search after compact");
    let hit_ids = hits.iter().map(|hit| hit.id).collect::<Vec<_>>();
    assert_eq!(hit_ids, vec![1, 2, 3, 4, 10]);

    let _ = fs::remove_file(compacted_dir.join("payloads.arrow"));
    let _ = fs::remove_file(compacted_dir.join("vectors.arrow"));
    let _ = fs::remove_file(compacted_dir.join("payloads.jsonl"));
    let _ = fs::remove_file(compacted_dir.join("vectors.jsonl"));

    truncate_wal(&root.join("wal.jsonl")).expect("truncate wal before reopening manual fixture");
    let reopened = HannsDb::open(root).expect("reopen db after compact");
    let reopened_hits = reopened
        .search("docs", &[0.0, 0.0], 10)
        .expect("search after compacted reopen");
    let reopened_hit_ids = reopened_hits.iter().map(|hit| hit.id).collect::<Vec<_>>();
    assert_eq!(reopened_hit_ids, vec![1, 2, 3, 4, 10]);
    let fetched = reopened
        .fetch_documents("docs", &[1, 2, 3, 4, 10])
        .expect("fetch after compacted Arrow-only reopen");
    let fetched_ids = fetched.iter().map(|doc| doc.id).collect::<Vec<_>>();
    assert_eq!(fetched_ids, vec![1, 2, 3, 4, 10]);
}

#[test]
fn compaction_filters_tombstoned_rows_from_merged_segment() {
    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path();
    let mut db = HannsDb::open(root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");

    let segments_dir = collection_segments_dir(root, "docs");
    write_segment(
        &segments_dir.join("seg-000001"),
        "seg-000001",
        2,
        &[doc(11, vec![0.0, 0.0]), doc(12, vec![0.1, 0.0])],
        &[1],
    );
    write_segment(
        &segments_dir.join("seg-000002"),
        "seg-000002",
        2,
        &[doc(21, vec![0.2, 0.0]), doc(22, vec![0.3, 0.0])],
        &[0],
    );
    write_segment(
        &segments_dir.join("seg-000003"),
        "seg-000003",
        2,
        &[doc(30, vec![5.0, 5.0])],
        &[],
    );

    SegmentSet {
        active_segment_id: "seg-000003".to_string(),
        immutable_segment_ids: vec!["seg-000001".to_string(), "seg-000002".to_string()],
    }
    .save_to_path(
        &root
            .join("collections")
            .join("docs")
            .join("segment_set.json"),
    )
    .expect("write segment_set");

    db.compact_collection("docs").expect("compact collection");

    let hits = db
        .search("docs", &[0.0, 0.0], 10)
        .expect("search after compact");
    let hit_ids = hits.iter().map(|hit| hit.id).collect::<Vec<_>>();
    assert_eq!(hit_ids, vec![11, 22, 30]);
}

#[test]
fn compaction_reopen_preserves_flushed_active_segment_after_post_compaction_write() {
    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path();
    let mut db = HannsDb::open(root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");

    let ids: Vec<i64> = (0..100).collect();
    let vectors: Vec<f32> = ids.iter().flat_map(|&i| vec![i as f32, 0.0_f32]).collect();
    db.insert("docs", &ids, &vectors).expect("insert 100 docs");
    let to_delete: Vec<i64> = (0..21).collect();
    db.delete("docs", &to_delete).expect("delete 21 docs");
    db.insert("docs", &[200], &[99.0_f32, 99.0])
        .expect("trigger rollover");
    db.compact_collection("docs")
        .expect("compact rolled-over collection");
    db.insert("docs", &[300], &[100.0_f32, 100.0])
        .expect("insert into post-compaction active segment");
    db.flush_collection("docs")
        .expect("flush refreshed post-compaction active segment");

    let collection_dir = root.join("collections").join("docs");
    let segment_set_before =
        SegmentSet::load_from_path(&collection_dir.join("segment_set.json")).expect("segment_set");
    let active_segment_dir =
        collection_segments_dir(root, "docs").join(&segment_set_before.active_segment_id);
    let payloads_jsonl = active_segment_dir.join("payloads.jsonl");
    let vectors_jsonl = active_segment_dir.join("vectors.jsonl");
    let payloads_arrow = active_segment_dir.join("payloads.arrow");
    let vectors_arrow = active_segment_dir.join("vectors.arrow");
    let _ = fs::remove_file(&payloads_jsonl);
    let _ = fs::remove_file(&vectors_jsonl);

    let reopened = HannsDb::open(root).expect("reopen after compact + active flush");
    let fetched = reopened
        .fetch_documents("docs", &[200, 300])
        .expect("fetch docs after compact + reopen");
    assert_eq!(
        fetched.iter().map(|doc| doc.id).collect::<Vec<_>>(),
        vec![200, 300]
    );

    let segment_set_after = SegmentSet::load_from_path(&collection_dir.join("segment_set.json"))
        .expect("segment_set after reopen");
    assert_eq!(
        segment_set_after.active_segment_id, segment_set_before.active_segment_id,
        "reopen should preserve the post-compaction active segment"
    );
    assert_eq!(
        segment_set_after.immutable_segment_ids, segment_set_before.immutable_segment_ids,
        "reopen should preserve compacted immutable segments"
    );
    assert!(
        !payloads_jsonl.exists() && !vectors_jsonl.exists(),
        "reopen should preserve arrow-only active storage after post-compaction writes"
    );
    assert!(
        payloads_arrow.exists() && vectors_arrow.exists(),
        "reopen should keep flushed active arrow snapshots after compaction"
    );
}

#[test]
fn compaction_reopen_prefers_jsonl_when_storage_format_is_jsonl() {
    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path();
    let mut db = HannsDb::open(root).expect("open db");
    let schema = CollectionSchema::new(
        "vector",
        2,
        "l2",
        vec![ScalarFieldSchema::new("tag", FieldType::String)],
    );
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");

    let segments_dir = collection_segments_dir(root, "docs");
    write_segment(
        &segments_dir.join("seg-000001"),
        "seg-000001",
        2,
        &[Document::new(
            1,
            vec![("tag".to_string(), FieldValue::String("jsonl".to_string()))],
            vec![0.0, 0.0],
        )],
        &[],
    );
    write_segment(
        &segments_dir.join("seg-000002"),
        "seg-000002",
        2,
        &[doc(10, vec![10.0, 10.0])],
        &[],
    );

    write_payloads_arrow(
        &segments_dir.join("seg-000001").join("payloads.arrow"),
        &[std::collections::BTreeMap::from([(
            "tag".to_string(),
            FieldValue::String("arrow".to_string()),
        )])],
        &[ScalarFieldSchema::new("tag", FieldType::String)],
    )
    .expect("write conflicting arrow payloads");

    SegmentSet {
        active_segment_id: "seg-000002".to_string(),
        immutable_segment_ids: vec!["seg-000001".to_string()],
    }
    .save_to_path(
        &root
            .join("collections")
            .join("docs")
            .join("segment_set.json"),
    )
    .expect("write segment_set");

    truncate_wal(&root.join("wal.jsonl")).expect("truncate wal before reopen");
    let reopened = HannsDb::open(root).expect("reopen db");
    let fetched = reopened
        .fetch_documents("docs", &[1])
        .expect("fetch immutable jsonl-authoritative doc");
    assert_eq!(fetched.len(), 1);
    assert_eq!(
        fetched[0].fields.get("tag"),
        Some(&FieldValue::String("jsonl".to_string())),
        "storage_format=jsonl should prefer payloads.jsonl over conflicting payloads.arrow"
    );
}

// ---------------------------------------------------------------------------
// Boundary test: single immutable segment compaction
// ---------------------------------------------------------------------------

#[test]
fn compaction_single_immutable_segment() {
    let root = tempfile::tempdir().expect("tempdir");
    let root = root.path();
    let dim = 4;
    let mut db = HannsDb::open(root).expect("open db");

    let schema = CollectionSchema::new("vector", dim, "l2", vec![]);
    db.create_collection_with_schema("test", &schema)
        .expect("create collection");

    let seg_dir = root.join("collections").join("test");
    let segments_dir = seg_dir.join("segments");

    // Insert 3 batches, manually create 1 immutable segment via file manipulation
    let docs: Vec<Document> = (0..5)
        .map(|i| doc(i as i64, vec![1.0, 2.0, 3.0, 4.0]))
        .collect();
    db.insert_documents("test", &docs)
        .expect("insert first batch");
    db.flush_collection("test").expect("flush");

    // Manually move the single segment into an immutable segment dir
    drop(db);
    fs::create_dir_all(&segments_dir).expect("create segments dir");
    let immut_dir = segments_dir.join("seg-0001");
    fs::create_dir_all(&immut_dir).expect("create immutable dir");
    for name in &[
        "records.bin",
        "ids.bin",
        "payloads.jsonl",
        "payloads.arrow",
        "tombstones.json",
        "segment.json",
        "vectors.jsonl",
        "vectors.arrow",
    ] {
        let src = seg_dir.join(name);
        if src.exists() {
            fs::rename(&src, immut_dir.join(name)).expect(&format!("move {name}"));
        }
    }
    // Copy forward_store artifacts if they exist
    for name in &["forward_store.arrow", "forward_store.parquet"] {
        let src = seg_dir.join(name);
        if src.exists() {
            let _ = fs::rename(&src, immut_dir.join(name));
        }
    }

    // Create new active segment
    let active_dir = segments_dir.join("seg-0002");
    fs::create_dir_all(&active_dir).expect("create active dir");
    SegmentMetadata::new("seg-0002", dim, 0, 0)
        .save_to_path(&active_dir.join("segment.json"))
        .expect("write active segment meta");
    TombstoneMask::new(0)
        .save_to_path(&active_dir.join("tombstones.json"))
        .expect("write active tombstones");

    // Write segment_set.json
    let set = SegmentSet {
        active_segment_id: "seg-0002".to_string(),
        immutable_segment_ids: vec!["seg-0001".to_string()],
    };
    set.save_to_path(&seg_dir.join("segment_set.json"))
        .expect("write segment_set");

    let mut db = HannsDb::open(root).expect("reopen db");

    // Compact — should merge the single immutable segment
    db.compact_collection("test")
        .expect("compact single immutable segment");

    // Verify: 1 immutable segment exists in the set
    let set_after = SegmentSet::load_from_path(&seg_dir.join("segment_set.json"))
        .expect("load segment set after compact");
    assert_eq!(
        set_after.immutable_segment_ids.len(),
        1,
        "should have 1 immutable after compact"
    );

    // Verify old immutable dir is gone
    assert!(!immut_dir.exists(), "old immutable dir should be removed");

    // Verify search works
    let hits = db
        .search("test", &[1.0, 2.0, 3.0, 4.0], 10)
        .expect("search after compact");
    assert_eq!(hits.len(), 5, "all 5 docs should be searchable after compact");

    // Verify fetch works
    let fetched = db
        .fetch_documents("test", &[0, 1, 2, 3, 4])
        .expect("fetch after compact");
    assert_eq!(fetched.len(), 5);
}

// ---------------------------------------------------------------------------
// Boundary test: multiple compaction rounds
// ---------------------------------------------------------------------------

#[test]
fn compaction_multiple_rounds() {
    let root = tempfile::tempdir().expect("tempdir");
    let root = root.path();
    let dim = 4;
    let mut db = HannsDb::open(root).expect("open db");

    let schema = CollectionSchema::new("vector", dim, "l2", vec![]);
    db.create_collection_with_schema("test", &schema)
        .expect("create collection");

    let seg_dir = root.join("collections").join("test");
    let segments_dir = seg_dir.join("segments");

    // Round 1: use write_segment to create 2 immutable segments directly
    let docs1: Vec<Document> = (0..3).map(|i| doc(i as i64, vec![1.0, 2.0, 3.0, 4.0])).collect();
    let docs2: Vec<Document> = (3..5).map(|i| doc(i as i64, vec![5.0, 6.0, 7.0, 8.0])).collect();
    write_segment(&segments_dir.join("seg-a1"), "seg-a1", dim, &docs1, &[]);
    write_segment(&segments_dir.join("seg-a2"), "seg-a2", dim, &docs2, &[]);
    // Active segment with 0 rows
    let active_dir = segments_dir.join("seg-a3");
    fs::create_dir_all(&active_dir).expect("create active");
    SegmentMetadata::new("seg-a3", dim, 0, 0)
        .save_to_path(&active_dir.join("segment.json"))
        .expect("meta");
    TombstoneMask::new(0)
        .save_to_path(&active_dir.join("tombstones.json"))
        .expect("tomb");
    SegmentSet {
        active_segment_id: "seg-a3".to_string(),
        immutable_segment_ids: vec!["seg-a1".to_string(), "seg-a2".to_string()],
    }
    .save_to_path(&seg_dir.join("segment_set.json"))
    .expect("segment_set");

    db.compact_collection("test").expect("first compact");

    // Verify round 1: 1 compacted immutable, old dirs gone
    let set1 =
        SegmentSet::load_from_path(&seg_dir.join("segment_set.json")).expect("set1 after compact");
    assert_eq!(set1.immutable_segment_ids.len(), 1);
    assert!(!segments_dir.join("seg-a1").exists());
    assert!(!segments_dir.join("seg-a2").exists());
    let compacted_id_1 = set1.immutable_segment_ids[0].clone();

    // Verify round 1: search returns all 5 docs
    let hits1 = db
        .search("test", &[1.0, 2.0, 3.0, 4.0], 10)
        .expect("search after first compact");
    assert_eq!(hits1.len(), 5, "all 5 docs should survive first compact");

    // Round 2: add another immutable segment, compact again
    drop(db);
    let docs3: Vec<Document> = (5..8).map(|i| doc(i as i64, vec![9.0, 10.0, 11.0, 12.0])).collect();
    write_segment(&segments_dir.join("seg-b1"), "seg-b1", dim, &docs3, &[]);
    // New active
    let active_dir2 = segments_dir.join("seg-b2");
    fs::create_dir_all(&active_dir2).expect("create active2");
    SegmentMetadata::new("seg-b2", dim, 0, 0)
        .save_to_path(&active_dir2.join("segment.json"))
        .expect("meta");
    TombstoneMask::new(0)
        .save_to_path(&active_dir2.join("tombstones.json"))
        .expect("tomb");
    SegmentSet {
        active_segment_id: "seg-b2".to_string(),
        immutable_segment_ids: vec![compacted_id_1, "seg-b1".to_string()],
    }
    .save_to_path(&seg_dir.join("segment_set.json"))
    .expect("segment_set round 2");

    truncate_wal(&root.join("wal.jsonl")).expect("truncate wal");
    let mut db = HannsDb::open(root).expect("reopen db for compact 2");
    db.compact_collection("test").expect("second compact");

    // Verify round 2: all 8 docs searchable
    let hits = db
        .search("test", &[1.0, 2.0, 3.0, 4.0], 10)
        .expect("search after second compact");
    assert_eq!(
        hits.len(),
        8,
        "all 8 docs should survive two compaction rounds"
    );

    let fetched = db
        .fetch_documents("test", &[0, 1, 2, 3, 4, 5, 6, 7])
        .expect("fetch after second compact");
    assert_eq!(fetched.len(), 8);
}

// ---------------------------------------------------------------------------
// Boundary test: upsert-triggered auto-compact
// ---------------------------------------------------------------------------

#[test]
fn compaction_triggered_by_upsert_auto_compact() {
    let root = tempfile::tempdir().expect("tempdir");
    let root = root.path();
    let dim = 4;
    let mut db = HannsDb::open(root).expect("open db");

    let schema = CollectionSchema::new("vector", dim, "l2", vec![]);
    db.create_collection_with_schema("test", &schema)
        .expect("create collection");

    let seg_dir = root.join("collections").join("test");
    let segments_dir = seg_dir.join("segments");

    // Manually create 4 immutable segments + 1 active to reach COMPACTION_THRESHOLD
    for seg_idx in 1..=4 {
        let seg_id = format!("seg-{seg_idx:04}");
        let dir = segments_dir.join(&seg_id);
        fs::create_dir_all(&dir).expect("create seg dir");

        let docs: Vec<Document> = (0..2)
            .map(|i| doc((seg_idx * 10 + i) as i64, vec![seg_idx as f32, 0.0, 0.0, 0.0]))
            .collect();
        write_segment(&dir, &seg_id, dim, &docs, &[]);
    }

    // Create active segment
    let active_dir = segments_dir.join("seg-0005");
    fs::create_dir_all(&active_dir).expect("create active");
    SegmentMetadata::new("seg-0005", dim, 0, 0)
        .save_to_path(&active_dir.join("segment.json"))
        .expect("active meta");
    TombstoneMask::new(0)
        .save_to_path(&active_dir.join("tombstones.json"))
        .expect("active tombstones");

    SegmentSet {
        active_segment_id: "seg-0005".to_string(),
        immutable_segment_ids: vec![
            "seg-0001".to_string(),
            "seg-0002".to_string(),
            "seg-0003".to_string(),
            "seg-0004".to_string(),
        ],
    }
    .save_to_path(&seg_dir.join("segment_set.json"))
    .expect("segment_set");

    drop(db);
    truncate_wal(&root.join("wal.jsonl")).expect("truncate wal");
    let mut db = HannsDb::open(root).expect("reopen db");

    // Upsert should trigger auto-compact (4 immutable >= COMPACTION_THRESHOLD)
    let upsert_doc = doc(11, vec![99.0, 0.0, 0.0, 0.0]);
    db.upsert_documents("test", &[upsert_doc])
        .expect("upsert should trigger auto-compact");

    // Verify: compaction reduced immutable segments to 1
    let set =
        SegmentSet::load_from_path(&seg_dir.join("segment_set.json")).expect("load segment_set");
    assert!(
        set.immutable_segment_ids.len() <= 1,
        "auto-compact should reduce immutable segments, got {}",
        set.immutable_segment_ids.len()
    );

    // Verify all original docs are still searchable
    let hits = db
        .search("test", &[0.0, 0.0, 0.0, 0.0], 20)
        .expect("search after auto-compact");
    assert!(
        hits.len() >= 8,
        "all original docs should survive, got {}",
        hits.len()
    );

    // Verify the upserted doc is also there
    let fetched = db
        .fetch_documents("test", &[11])
        .expect("fetch upserted doc");
    assert_eq!(fetched.len(), 1, "upserted doc should be fetchable");
}

// ---------------------------------------------------------------------------
// Crash-during-compaction recovery via WAL replay
// ---------------------------------------------------------------------------

#[test]
fn compaction_crash_during_compaction_recovers_via_wal() {
    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path();
    let mut db = HannsDb::open(root).expect("open db");

    let schema = CollectionSchema::new("vector", 2, "l2", vec![]);
    db.create_collection_with_schema("test", &schema)
        .expect("create collection");

    let segments_dir = collection_segments_dir(root, "test");

    // Write 2 immutable segments with real data
    write_segment(
        &segments_dir.join("seg-000001"),
        "seg-000001",
        2,
        &[doc(1, vec![0.0, 0.0]), doc(2, vec![1.0, 0.0])],
        &[],
    );
    write_segment(
        &segments_dir.join("seg-000002"),
        "seg-000002",
        2,
        &[doc(3, vec![0.0, 1.0]), doc(4, vec![2.0, 0.0])],
        &[],
    );

    // Active segment (empty, just to hold the active slot)
    let active_dir = segments_dir.join("seg-000003");
    std::fs::create_dir_all(&active_dir).expect("create active dir");
    SegmentMetadata::new("seg-000003", 2, 0, 0)
        .save_to_path(&active_dir.join("segment.json"))
        .expect("write active segment meta");
    TombstoneMask::new(0)
        .save_to_path(&active_dir.join("tombstones.json"))
        .expect("write active tombstones");

    SegmentSet {
        active_segment_id: "seg-000003".to_string(),
        immutable_segment_ids: vec!["seg-000001".to_string(), "seg-000002".to_string()],
    }
    .save_to_path(
        &root
            .join("collections")
            .join("test")
            .join("segment_set.json"),
    )
    .expect("write segment_set");

    // Truncate any WAL records from collection creation
    truncate_wal(&root.join("wal.jsonl")).expect("truncate wal before crash simulation");

    // Simulate crash-during-compaction: write a CompactCollection WAL record
    // but do NOT actually compact (as if the process died before compacting).
    append_wal_record(
        &root.join("wal.jsonl"),
        &WalRecord::CompactCollection {
            collection_name: "test".to_string(),
            compacted_segment_id: "compact-crash".to_string(),
        },
    )
    .expect("append wal");

    // Drop the db (simulating crash) and reopen
    drop(db);
    let mut db = HannsDb::open(root).expect("reopen db after simulated crash");

    // WAL replay should have handled the incomplete compaction gracefully.
    // The original immutable segments should still be searchable (either
    // WAL replay re-compacted them, or they survived as-is).
    let hits = db
        .search("test", &[0.0, 0.0], 10)
        .expect("search after crash recovery");
    let hit_ids: Vec<i64> = hits.iter().map(|hit| hit.id).collect();
    assert_eq!(
        hit_ids, vec![1, 2, 3, 4],
        "all 4 docs should be searchable after crash-during-compaction recovery"
    );

    // Verify fetch_documents also works
    let fetched = db
        .fetch_documents("test", &[1, 2, 3, 4])
        .expect("fetch after crash recovery");
    let fetched_ids: Vec<i64> = fetched.iter().map(|doc| doc.id).collect();
    assert_eq!(
        fetched_ids, vec![1, 2, 3, 4],
        "all 4 docs should be fetchable after crash-during-compaction recovery"
    );
}
