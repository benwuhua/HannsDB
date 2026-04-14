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
use hannsdb_core::wal::truncate_wal;

fn doc(id: i64, vector: Vec<f32>) -> Document {
    Document::new(id, Vec::<(String, FieldValue)>::new(), vector)
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
