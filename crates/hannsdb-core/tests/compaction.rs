use std::fs;
use std::path::Path;

use hannsdb_core::db::HannsDb;
use hannsdb_core::document::{
    CollectionSchema, Document, FieldType, FieldValue, ScalarFieldSchema,
};
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

    let hits = db
        .search("docs", &[0.0, 0.0], 10)
        .expect("search after compact");
    let hit_ids = hits.iter().map(|hit| hit.id).collect::<Vec<_>>();
    assert_eq!(hit_ids, vec![1, 2, 3, 4, 10]);

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
