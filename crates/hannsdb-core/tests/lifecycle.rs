//! Lifecycle integration tests.
//!
//! These tests cover end-to-end flows that cross multiple sub-systems:
//! compaction, rollover trigger, WAL persistence after compaction, and
//! segment listing.  Each test starts from a clean tempdir so there is no
//! shared state between cases.

use std::fs;
use std::path::Path;

use hannsdb_core::db::HannsDb;
use hannsdb_core::document::{Document, FieldValue};
use hannsdb_core::segment::{
    append_payloads, append_record_ids, append_records, SegmentMetadata, SegmentSet, TombstoneMask,
};
use hannsdb_core::wal::{load_wal_records, WalRecord};

// ── helpers ──────────────────────────────────────────────────────────────────

fn doc(id: i64, vector: Vec<f32>) -> Document {
    Document::new(id, Vec::<(String, FieldValue)>::new(), vector)
}

/// Write a complete segment directory: records, ids, payloads, tombstones,
/// segment.json.  `deleted_rows` is a slice of zero-based row indices to mark
/// deleted in the tombstone.
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
    for d in documents {
        ids.push(d.id);
        vectors.extend_from_slice(d.primary_vector());
        payloads.push(d.fields.clone());
    }

    let inserted = append_records(&segment_dir.join("records.bin"), dimension, &vectors)
        .expect("write records");
    assert_eq!(inserted, documents.len(), "record count mismatch");
    let _ = append_record_ids(&segment_dir.join("ids.bin"), &ids).expect("write ids");
    let _ =
        append_payloads(&segment_dir.join("payloads.jsonl"), &payloads).expect("write payloads");

    let mut tombstone = TombstoneMask::new(documents.len());
    for &row in deleted_rows {
        assert!(tombstone.mark_deleted(row), "row index out of range");
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

fn segments_dir(root: &Path, collection: &str) -> std::path::PathBuf {
    root.join("collections").join(collection).join("segments")
}

fn segment_set_path(root: &Path, collection: &str) -> std::path::PathBuf {
    root.join("collections")
        .join(collection)
        .join("segment_set.json")
}

// ── tests ─────────────────────────────────────────────────────────────────────

/// Compact multiple immutable segments, drop the DB handle, reopen, and
/// verify that search still returns all live documents.
///
/// The WAL on disk contains only a `CompactCollection` record (no
/// `CreateCollection`), so WAL replay treats the collection as externally
/// owned and does not wipe it.  The on-disk compacted state is therefore
/// preserved across the reopen.
#[test]
fn lifecycle_compact_reopen_search_returns_correct_results() {
    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path();

    // Create the collection via the DB API so its directory and collection.json
    // exist, then remove the WAL so that future reopens see no owned collections.
    {
        let mut db = HannsDb::open(root).expect("open db");
        db.create_collection("docs", 2, "l2")
            .expect("create collection");
    }
    fs::remove_file(root.join("wal.jsonl")).expect("remove wal");

    // Write three segments directly — two immutable, one active.
    let segs = segments_dir(root, "docs");
    write_segment(
        &segs.join("seg-000001"),
        "seg-000001",
        2,
        &[doc(1, vec![0.0, 0.0]), doc(2, vec![1.0, 0.0])],
        &[],
    );
    write_segment(
        &segs.join("seg-000002"),
        "seg-000002",
        2,
        &[doc(3, vec![0.0, 1.0]), doc(4, vec![2.0, 0.0])],
        &[],
    );
    write_segment(
        &segs.join("seg-000003"),
        "seg-000003",
        2,
        &[doc(10, vec![10.0, 10.0])],
        &[],
    );
    SegmentSet {
        active_segment_id: "seg-000003".to_string(),
        immutable_segment_ids: vec!["seg-000001".to_string(), "seg-000002".to_string()],
    }
    .save_to_path(&segment_set_path(root, "docs"))
    .expect("write segment_set");

    // Compact and drop — this writes a CompactCollection WAL record.
    {
        let mut db = HannsDb::open(root).expect("reopen before compact");
        db.compact_collection("docs").expect("compact collection");
    }

    // Reopen.  WAL has only CompactCollection (no CreateCollection) →
    // plan has no owned collections → no wipe → on-disk compacted state kept.
    let db = HannsDb::open(root).expect("reopen after compact");
    let hits = db
        .search("docs", &[0.0_f32, 0.0], 10)
        .expect("search after reopen");
    let hit_ids: Vec<i64> = hits.iter().map(|h| h.id).collect();
    assert_eq!(
        hit_ids,
        vec![1, 2, 3, 4, 10],
        "all live docs must be found after compact + reopen"
    );
}

/// Inserting past the tombstone-ratio threshold (20 %) triggers the rollover
/// mechanism and keeps the collection readable after reopen.
#[test]
fn lifecycle_tombstone_ratio_rollover_creates_segment_set_structure() {
    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path();
    let mut db = HannsDb::open(root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");

    // Insert 100 docs.
    let ids: Vec<i64> = (0..100).collect();
    let vectors: Vec<f32> = ids.iter().flat_map(|&i| vec![i as f32, 0.0_f32]).collect();
    db.insert("docs", &ids, &vectors).expect("insert 100 docs");

    // Delete 21 — tombstone ratio becomes 21 % but no insert triggers rollover yet.
    let to_delete: Vec<i64> = (0..21).collect();
    let deleted = db.delete("docs", &to_delete).expect("delete 21 docs");
    assert_eq!(deleted, 21);
    assert!(
        !segment_set_path(root, "docs").exists(),
        "segment_set must not be created by delete alone"
    );

    // One more insert triggers maybe_trigger_segment_rollover:
    //   record_count=101, deleted_count=21, ratio≈20.8 % > 20 % threshold.
    db.insert("docs", &[200], &[99.0_f32, 99.0])
        .expect("trigger insert");

    assert!(
        segment_set_path(root, "docs").exists(),
        "segment_set must be created when tombstone ratio exceeds threshold"
    );

    let set =
        SegmentSet::load_from_path(&segment_set_path(root, "docs")).expect("load segment_set");
    assert!(
        segments_dir(root, "docs")
            .join(&set.active_segment_id)
            .exists(),
        "active segment directory must exist: {}",
        set.active_segment_id
    );
    for id in &set.immutable_segment_ids {
        assert!(
            segments_dir(root, "docs").join(id).exists(),
            "immutable segment directory must exist: {id}"
        );
    }

    let reopened = HannsDb::open(root).expect("reopen after rollover");
    let hits = reopened
        .search("docs", &[0.0_f32, 0.0], 3)
        .expect("search after rollover reopen");
    let hit_ids: Vec<i64> = hits.iter().map(|hit| hit.id).collect();
    assert_eq!(hit_ids, vec![21, 22, 23]);
}

#[test]
fn lifecycle_reopen_preserves_flushed_active_segment_after_rollover_write() {
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
    db.insert("docs", &[300], &[100.0_f32, 100.0])
        .expect("insert into new active segment");
    db.flush_collection("docs")
        .expect("flush refreshed active segment");

    let segment_set_before =
        SegmentSet::load_from_path(&segment_set_path(root, "docs")).expect("load segment_set");
    let active_segment_dir = segments_dir(root, "docs").join(&segment_set_before.active_segment_id);
    let payloads_jsonl = active_segment_dir.join("payloads.jsonl");
    let vectors_jsonl = active_segment_dir.join("vectors.jsonl");
    let payloads_arrow = active_segment_dir.join("payloads.arrow");
    let vectors_arrow = active_segment_dir.join("vectors.arrow");
    let _ = fs::remove_file(&payloads_jsonl);
    let _ = fs::remove_file(&vectors_jsonl);

    let reopened = HannsDb::open(root).expect("reopen after rollover active flush");
    let fetched = reopened
        .fetch_documents("docs", &[200, 300])
        .expect("fetch rollover active docs after reopen");
    assert_eq!(
        fetched.iter().map(|doc| doc.id).collect::<Vec<_>>(),
        vec![200, 300]
    );

    let segment_set_after =
        SegmentSet::load_from_path(&segment_set_path(root, "docs")).expect("reload segment_set");
    assert_eq!(
        segment_set_after.active_segment_id, segment_set_before.active_segment_id,
        "reopen should preserve the active segment instead of rebuilding from WAL"
    );
    assert_eq!(
        segment_set_after.immutable_segment_ids, segment_set_before.immutable_segment_ids,
        "reopen should preserve rollover-produced immutable segments"
    );
    assert!(
        !payloads_jsonl.exists() && !vectors_jsonl.exists(),
        "reopen should preserve arrow-only active storage after later writes"
    );
    assert!(
        payloads_arrow.exists() && vectors_arrow.exists(),
        "reopen should keep flushed active arrow snapshots in the rolled-over layout"
    );
}

/// Calling compact on a collection that has no `segment_set.json` (flat
/// single-segment layout) is a no-op: it succeeds, does not create any
/// segment-set files, and leaves search results intact.
#[test]
fn lifecycle_compact_noop_for_collection_without_segment_set() {
    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path();
    let mut db = HannsDb::open(root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");
    db.insert("docs", &[1, 2], &[0.0_f32, 0.0, 1.0, 1.0])
        .expect("insert docs");

    db.compact_collection("docs")
        .expect("compact must succeed as a no-op when segment_set is absent");

    assert!(
        !segment_set_path(root, "docs").exists(),
        "compact no-op must not create segment_set.json"
    );

    let hits = db
        .search("docs", &[0.0_f32, 0.0], 2)
        .expect("search must still work after no-op compact");
    assert_eq!(hits.len(), 2, "both docs must remain searchable");
}

/// `list_collection_segments` returns one entry per segment with correct
/// live/dead counts when the collection has a multi-segment layout.
#[test]
fn lifecycle_list_segments_returns_correct_counts_for_multi_segment_layout() {
    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path();

    // Bootstrap collection metadata, then drop and remove WAL.
    {
        let mut db = HannsDb::open(root).expect("open db");
        db.create_collection("docs", 2, "l2")
            .expect("create collection");
    }
    fs::remove_file(root.join("wal.jsonl")).expect("remove wal");

    let segs = segments_dir(root, "docs");
    // seg-000001: doc1 live, doc2 deleted (row 1)
    write_segment(
        &segs.join("seg-000001"),
        "seg-000001",
        2,
        &[doc(1, vec![0.0, 0.0]), doc(2, vec![1.0, 0.0])],
        &[1],
    );
    // seg-000002 (active): doc3 live, no deletes
    write_segment(
        &segs.join("seg-000002"),
        "seg-000002",
        2,
        &[doc(3, vec![0.0, 1.0])],
        &[],
    );
    SegmentSet {
        active_segment_id: "seg-000002".to_string(),
        immutable_segment_ids: vec!["seg-000001".to_string()],
    }
    .save_to_path(&segment_set_path(root, "docs"))
    .expect("write segment_set");

    let db = HannsDb::open(root).expect("reopen db");
    let segments = db.list_collection_segments("docs").expect("list segments");

    assert_eq!(segments.len(), 2, "must report 2 segments");

    let active = segments
        .iter()
        .find(|s| s.id == "seg-000002")
        .expect("active segment must be present");
    assert_eq!(active.live_count, 1, "active: 1 live");
    assert_eq!(active.dead_count, 0, "active: 0 dead");

    let immut = segments
        .iter()
        .find(|s| s.id == "seg-000001")
        .expect("immutable segment must be present");
    assert_eq!(immut.live_count, 1, "immutable: 1 live");
    assert_eq!(immut.dead_count, 1, "immutable: 1 dead");
}

/// After compaction the segment set has exactly one immutable segment (the
/// newly merged one), and the active segment is untouched.
#[test]
fn lifecycle_compact_reduces_immutable_count_to_one() {
    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path();

    {
        let mut db = HannsDb::open(root).expect("open db");
        db.create_collection("docs", 2, "l2")
            .expect("create collection");
    }
    fs::remove_file(root.join("wal.jsonl")).expect("remove wal");

    let segs = segments_dir(root, "docs");
    write_segment(
        &segs.join("seg-000001"),
        "seg-000001",
        2,
        &[doc(1, vec![0.0, 0.0])],
        &[],
    );
    write_segment(
        &segs.join("seg-000002"),
        "seg-000002",
        2,
        &[doc(2, vec![1.0, 0.0])],
        &[],
    );
    write_segment(
        &segs.join("seg-000003"),
        "seg-000003",
        2,
        &[doc(3, vec![0.0, 1.0])],
        &[],
    );
    SegmentSet {
        active_segment_id: "seg-000003".to_string(),
        immutable_segment_ids: vec!["seg-000001".to_string(), "seg-000002".to_string()],
    }
    .save_to_path(&segment_set_path(root, "docs"))
    .expect("write segment_set");

    let mut db = HannsDb::open(root).expect("reopen db");
    db.compact_collection("docs").expect("compact");

    let segments = db
        .list_collection_segments("docs")
        .expect("list segments after compact");
    let immutable_segments: Vec<_> = segments.iter().filter(|s| s.id != "seg-000003").collect();

    assert_eq!(
        immutable_segments.len(),
        1,
        "compact must collapse multiple immutable segments into one"
    );
    assert_eq!(
        immutable_segments[0].live_count, 2,
        "compacted segment must contain 2 live rows from seg-1 + seg-2"
    );
    assert_eq!(immutable_segments[0].dead_count, 0);
    assert!(
        segments.iter().any(|s| s.id == "seg-000003"),
        "active segment must remain unchanged after compact"
    );
}

/// `compact_collection` writes a `CompactCollection` WAL record when an
/// actual merge takes place.  The record's `compacted_segment_id` must
/// match the directory that exists on disk after compaction.
#[test]
fn lifecycle_wal_compact_record_written_with_correct_segment_id() {
    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path();

    {
        let mut db = HannsDb::open(root).expect("open db");
        db.create_collection("docs", 2, "l2")
            .expect("create collection");
    }
    fs::remove_file(root.join("wal.jsonl")).expect("remove wal");

    let segs = segments_dir(root, "docs");
    write_segment(
        &segs.join("seg-000001"),
        "seg-000001",
        2,
        &[doc(1, vec![0.0, 0.0])],
        &[],
    );
    write_segment(
        &segs.join("seg-000002"),
        "seg-000002",
        2,
        &[doc(2, vec![1.0, 0.0])],
        &[],
    );
    write_segment(
        &segs.join("seg-000003"),
        "seg-000003",
        2,
        &[doc(3, vec![0.0, 1.0])],
        &[],
    );
    SegmentSet {
        active_segment_id: "seg-000003".to_string(),
        immutable_segment_ids: vec!["seg-000001".to_string(), "seg-000002".to_string()],
    }
    .save_to_path(&segment_set_path(root, "docs"))
    .expect("write segment_set");

    {
        let mut db = HannsDb::open(root).expect("reopen db");
        db.compact_collection("docs").expect("compact");
    }

    let records = load_wal_records(&root.join("wal.jsonl")).expect("load wal");
    assert_eq!(
        records.len(),
        1,
        "WAL must contain exactly one record after compact"
    );
    match &records[0] {
        WalRecord::CompactCollection {
            collection_name,
            compacted_segment_id,
        } => {
            assert_eq!(collection_name, "docs");
            assert!(
                segs.join(compacted_segment_id).exists(),
                "compacted_segment_id must point to an existing directory: {compacted_segment_id}"
            );
        }
        other => panic!("expected CompactCollection WAL record, got: {other:?}"),
    }
}

/// When `compact_collection` is a no-op (no `segment_set.json`), no WAL
/// record is appended.  The WAL line count must be unchanged.
#[test]
fn lifecycle_compact_noop_does_not_append_wal_record() {
    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path();
    let mut db = HannsDb::open(root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");
    db.insert("docs", &[1], &[0.0_f32, 0.0]).expect("insert");

    let wal_before = fs::read_to_string(root.join("wal.jsonl")).expect("read wal before compact");
    db.compact_collection("docs").expect("no-op compact");
    let wal_after = fs::read_to_string(root.join("wal.jsonl")).expect("read wal after compact");

    assert_eq!(
        wal_before.lines().count(),
        wal_after.lines().count(),
        "WAL must not grow when compact is a no-op"
    );
}
