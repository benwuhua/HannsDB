use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use std::time::{SystemTime, UNIX_EPOCH};

use hannsdb_core::catalog::ManifestMetadata;
use hannsdb_core::db::HannsDb;
use hannsdb_core::document::{
    CollectionSchema, Document, FieldType, FieldValue, ScalarFieldSchema,
};
use hannsdb_core::segment::{
    append_payloads, append_record_ids, append_records, SegmentMetadata, SegmentSet, TombstoneMask,
};
use hannsdb_core::wal::{append_wal_record, WalRecord};
use hannsdb_index::descriptor::{
    ScalarIndexDescriptor, ScalarIndexKind, VectorIndexDescriptor, VectorIndexKind,
};
use serde_json::json;

fn unique_temp_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("{}_{}", name, nanos))
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
    fs::create_dir_all(&seg2_dir).expect("create seg-0002 dir");

    for file in [
        "segment.json",
        "records.bin",
        "ids.bin",
        "payloads.jsonl",
        "tombstones.json",
    ] {
        fs::rename(collection_dir.join(file), seg1_dir.join(file)).expect("move seg-0001 file");
    }

    let mut second_ids = Vec::with_capacity(second_segment_documents.len());
    let mut second_vectors = Vec::with_capacity(second_segment_documents.len() * dimension);
    let mut second_payloads = Vec::with_capacity(second_segment_documents.len());
    for document in second_segment_documents {
        second_ids.push(document.id);
        second_vectors.extend_from_slice(&document.vector);
        second_payloads.push(document.fields.clone());
    }

    let inserted = append_records(&seg2_dir.join("records.bin"), dimension, &second_vectors)
        .expect("append seg-0002 records");
    assert_eq!(inserted, second_segment_documents.len());
    let _ = append_record_ids(&seg2_dir.join("ids.bin"), &second_ids).expect("append seg-0002 ids");
    let _ = append_payloads(&seg2_dir.join("payloads.jsonl"), &second_payloads)
        .expect("append seg-0002 payloads");

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
fn collection_api_create_insert_search() {
    let root = unique_temp_dir("hannsdb_collection_api_create");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");

    let inserted = db
        .insert("docs", &[42, 84], &[0.0_f32, 0.0, 1.0, 1.0])
        .expect("insert vectors");
    assert_eq!(inserted, 2);

    let hits = db
        .search("docs", &[0.0_f32, 0.0], 1)
        .expect("search vectors");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, 42);
}

#[test]
fn collection_api_search_merges_topk_across_two_segments() {
    let root = unique_temp_dir("hannsdb_collection_api_two_segment_search");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");

    let mut ids = Vec::with_capacity(100);
    let mut vectors = Vec::with_capacity(200);
    for i in 0_i64..100_i64 {
        ids.push(i);
        if i == 0 {
            vectors.extend_from_slice(&[0.0_f32, 0.0]);
        } else {
            vectors.extend_from_slice(&[100.0_f32 + i as f32, 100.0 + i as f32]);
        }
    }
    db.insert("docs", &ids, &vectors)
        .expect("insert seg-0001 rows");

    let mut second_docs = Vec::with_capacity(100);
    for i in 0_i64..100_i64 {
        let id = 100_i64 + i;
        let vector = if i < 5 {
            vec![(i as f32 + 1.0) * 0.1, 0.0]
        } else {
            vec![200.0 + i as f32, 200.0 + i as f32]
        };
        second_docs.push(Document::new(
            id,
            Vec::<(String, FieldValue)>::new(),
            vector,
        ));
    }

    rewrite_collection_to_two_segment_layout(&root, "docs", 2, &second_docs, &[]);

    let hits = db
        .search("docs", &[0.0_f32, 0.0], 5)
        .expect("cross-segment search");
    let hit_ids = hits.iter().map(|hit| hit.id).collect::<Vec<_>>();
    assert_eq!(hit_ids, vec![0, 100, 101, 102, 103]);
}

#[test]
fn collection_api_filter_query_respects_cross_segment_tombstones() {
    let root = unique_temp_dir("hannsdb_collection_api_two_segment_filter_tombstone");
    let mut db = HannsDb::open(&root).expect("open db");
    let schema = CollectionSchema::new(
        "vector",
        2,
        "l2",
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");

    let mut first_docs = Vec::with_capacity(100);
    for i in 0_i64..100_i64 {
        let group = if i == 10 || i == 20 { 1 } else { 2 };
        let vector = if i == 10 {
            vec![0.0_f32, 0.0]
        } else if i == 20 {
            vec![0.2_f32, 0.0]
        } else {
            vec![100.0 + i as f32, 100.0 + i as f32]
        };
        first_docs.push(Document::new(
            i,
            vec![("group".to_string(), FieldValue::Int64(group))],
            vector,
        ));
    }
    db.insert_documents("docs", &first_docs)
        .expect("insert seg-0001 docs");

    let mut second_docs = Vec::with_capacity(100);
    for i in 0_i64..100_i64 {
        let id = 100_i64 + i;
        let (group, vector) = if i == 10 {
            (1_i64, vec![0.1_f32, 0.0])
        } else if i == 20 {
            (1_i64, vec![0.3_f32, 0.0])
        } else {
            (2_i64, vec![200.0 + i as f32, 200.0 + i as f32])
        };
        second_docs.push(Document::new(
            id,
            vec![("group".to_string(), FieldValue::Int64(group))],
            vector,
        ));
    }

    rewrite_collection_to_two_segment_layout(&root, "docs", 2, &second_docs, &[10]);

    let hits = db
        .query_documents("docs", &[0.0_f32, 0.0], 10, Some("group == 1"))
        .expect("cross-segment filtered query");
    let hit_ids = hits.iter().map(|hit| hit.id).collect::<Vec<_>>();
    assert_eq!(hit_ids, vec![10, 20, 120]);
}

#[test]
fn collection_api_multi_segment_read_paths_use_segment_aware_loading() {
    let root = unique_temp_dir("hannsdb_collection_api_multi_segment_reads");
    let mut db = HannsDb::open(&root).expect("open db");
    let schema = CollectionSchema::new(
        "vector",
        2,
        "l2",
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");

    let first_docs = vec![
        Document::new(
            10,
            vec![("group".to_string(), FieldValue::Int64(1))],
            vec![0.0_f32, 0.0],
        ),
        Document::new(
            20,
            vec![("group".to_string(), FieldValue::Int64(2))],
            vec![1.0_f32, 1.0],
        ),
    ];
    db.insert_documents("docs", &first_docs)
        .expect("insert seg-0001 docs");

    let second_docs = vec![
        Document::new(
            30,
            vec![("group".to_string(), FieldValue::Int64(3))],
            vec![0.1_f32, 0.0],
        ),
        Document::new(
            40,
            vec![("group".to_string(), FieldValue::Int64(4))],
            vec![0.2_f32, 0.0],
        ),
    ];
    rewrite_collection_to_two_segment_layout(&root, "docs", 2, &second_docs, &[1]);

    let info = db.get_collection_info("docs").expect("collection info");
    assert_eq!(info.record_count, 4);
    assert_eq!(info.deleted_count, 1);
    assert_eq!(info.live_count, 3);

    let fetched = db
        .fetch_documents("docs", &[10, 30, 40])
        .expect("fetch across segments");
    let fetched_ids = fetched
        .iter()
        .map(|document| document.id)
        .collect::<Vec<_>>();
    assert_eq!(fetched_ids, vec![10, 30]);

    let hits = db
        .query_documents("docs", &[0.0_f32, 0.0], 3, None)
        .expect("unfiltered query across segments");
    let hit_ids = hits.iter().map(|hit| hit.id).collect::<Vec<_>>();
    assert_eq!(hit_ids, vec![10, 30, 20]);
}

#[test]
fn collection_api_rollover_eligible_sequence_keeps_follow_up_reads_on_flat_layout() {
    let root = unique_temp_dir("hannsdb_collection_api_rollover_guard");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");

    let ids: Vec<i64> = (0..100).collect();
    let vectors: Vec<f32> = ids.iter().flat_map(|&i| vec![i as f32, 0.0_f32]).collect();
    db.insert("docs", &ids, &vectors).expect("insert docs");

    let to_delete: Vec<i64> = (0..21).collect();
    let deleted = db.delete("docs", &to_delete).expect("delete docs");
    assert_eq!(deleted, 21);

    db.insert("docs", &[200], &[0.0_f32, 0.0])
        .expect("follow-up insert after rollover threshold");

    assert!(
        !root
            .join("collections")
            .join("docs")
            .join("segment_set.json")
            .exists(),
        "incomplete auto-rollover must stay disabled"
    );

    let info = db.get_collection_info("docs").expect("collection info");
    assert_eq!(info.record_count, 101);
    assert_eq!(info.deleted_count, 21);
    assert_eq!(info.live_count, 80);

    let hits = db
        .search("docs", &[0.0_f32, 0.0], 3)
        .expect("search after rollover-eligible sequence");
    let hit_ids = hits.iter().map(|hit| hit.id).collect::<Vec<_>>();
    assert_eq!(hit_ids, vec![200, 21, 22]);
}

#[test]
fn collection_api_delete_masks_results() {
    let root = unique_temp_dir("hannsdb_collection_api_delete");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");
    db.insert("docs", &[42, 84], &[0.0_f32, 0.0, 1.0, 1.0])
        .expect("insert vectors");

    let deleted = db.delete("docs", &[42]).expect("delete one id");
    assert_eq!(deleted, 1);

    let hits = db
        .search("docs", &[0.0_f32, 0.0], 2)
        .expect("search vectors");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, 84);
}

#[test]
fn collection_api_reopen_recovery() {
    let root = unique_temp_dir("hannsdb_collection_api_reopen");
    {
        let mut db = HannsDb::open(&root).expect("open db");
        db.create_collection("docs", 2, "l2")
            .expect("create collection");
        db.insert("docs", &[42, 84], &[0.0_f32, 0.0, 1.0, 1.0])
            .expect("insert vectors");
        db.delete("docs", &[42]).expect("delete one id");
    }

    let db = HannsDb::open(&root).expect("reopen db");
    let hits = db
        .search("docs", &[0.0_f32, 0.0], 2)
        .expect("search vectors");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, 84);
}

#[test]
fn collection_api_create_collection_rejects_zero_dimension() {
    let root = unique_temp_dir("hannsdb_collection_api_zero_dim");
    let mut db = HannsDb::open(&root).expect("open db");

    let result = db.create_collection("bad", 0, "l2");
    assert!(result.is_err(), "dimension 0 must be rejected");
}

#[test]
fn collection_api_insert_rejects_mismatched_id_vector_counts_without_panic() {
    let root = unique_temp_dir("hannsdb_collection_api_mismatch");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        db.insert("docs", &[1, 2, 3], &[0.0_f32, 0.0, 1.0, 1.0])
    }));

    assert!(result.is_ok(), "insert should not panic");
    assert!(
        result.expect("insert call should return").is_err(),
        "mismatched id/vector counts must return Err"
    );
}

#[test]
fn collection_api_insert_rejects_duplicate_external_ids() {
    let root = unique_temp_dir("hannsdb_collection_api_duplicate_ids");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");

    let result = db.insert("docs", &[7, 7], &[0.0_f32, 0.0, 1.0, 1.0]);
    assert!(result.is_err(), "duplicate external ids must be rejected");
}

#[test]
fn collection_api_drop_collection_removes_directory_and_manifest_entry() {
    let root = unique_temp_dir("hannsdb_collection_api_drop");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");

    db.drop_collection("docs").expect("drop collection");

    assert!(
        !root.join("collections").join("docs").exists(),
        "collection directory must be removed"
    );
    let manifest = ManifestMetadata::load_from_path(&root.join("manifest.json"))
        .expect("load manifest metadata");
    assert!(
        !manifest.collections.iter().any(|name| name == "docs"),
        "manifest entry must be removed"
    );
}

#[test]
fn collection_api_recreate_collection_after_drop_succeeds() {
    let root = unique_temp_dir("hannsdb_collection_api_recreate");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");
    db.drop_collection("docs").expect("drop collection");

    db.create_collection("docs", 2, "l2")
        .expect("recreate collection");
    let inserted = db
        .insert("docs", &[101], &[0.5_f32, 0.5])
        .expect("insert after recreate");
    assert_eq!(inserted, 1);
}

#[test]
fn collection_api_drop_missing_collection_returns_not_found() {
    let root = unique_temp_dir("hannsdb_collection_api_drop_missing");
    let mut db = HannsDb::open(&root).expect("open db");

    let err = db
        .drop_collection("missing")
        .expect_err("dropping missing collection should fail");
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
}

#[test]
fn collection_api_search_honors_ip_metric() {
    let root = unique_temp_dir("hannsdb_collection_api_metric_ip");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs_ip", 2, "ip")
        .expect("create ip collection");
    db.insert("docs_ip", &[1001, 1002], &[100.0_f32, 100.0, 1.0, 1.0])
        .expect("insert vectors");

    let hits = db
        .search("docs_ip", &[1.0_f32, 1.0], 1)
        .expect("search vectors");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, 1001, "IP should favor larger dot product");
}

#[test]
fn collection_api_search_honors_cosine_metric() {
    let root = unique_temp_dir("hannsdb_collection_api_metric_cosine");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs_cos", 2, "cosine")
        .expect("create cosine collection");
    db.insert("docs_cos", &[2001, 2002], &[10.0_f32, 0.0, 1.0, 1.0])
        .expect("insert vectors");

    let hits = db
        .search("docs_cos", &[1.0_f32, 0.0], 1)
        .expect("search vectors");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, 2001, "cosine should favor aligned direction");
}

#[test]
fn collection_api_get_collection_info_reports_stats() {
    let root = unique_temp_dir("hannsdb_collection_api_info_stats");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");
    db.insert("docs", &[11, 22], &[0.0_f32, 0.0, 1.0, 1.0])
        .expect("insert vectors");
    db.delete("docs", &[11]).expect("delete one id");

    let info = db.get_collection_info("docs").expect("get collection info");
    assert_eq!(info.name, "docs");
    assert_eq!(info.dimension, 2);
    assert_eq!(info.metric, "l2");
    assert_eq!(info.record_count, 2);
    assert_eq!(info.deleted_count, 1);
    assert_eq!(info.live_count, 1);
}

#[test]
fn collection_api_get_collection_info_missing_returns_not_found() {
    let root = unique_temp_dir("hannsdb_collection_api_info_missing");
    let db = HannsDb::open(&root).expect("open db");
    let err = db
        .get_collection_info("missing")
        .expect_err("missing collection should return not found");
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
}

#[test]
fn collection_api_flush_collection_succeeds_for_existing_collection() {
    let root = unique_temp_dir("hannsdb_collection_api_flush_ok");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");

    db.flush_collection("docs")
        .expect("flush should succeed for existing collection");
}

#[test]
fn collection_api_open_replay_does_not_touch_pre_wal_collection_under_partial_storage() {
    let root = unique_temp_dir("hannsdb_collection_api_replay_pre_wal_guard");

    {
        let mut db = HannsDb::open(&root).expect("open db");
        db.create_collection("legacy", 2, "l2")
            .expect("create legacy collection");
        db.insert("legacy", &[7], &[0.2_f32, 0.2])
            .expect("insert into legacy");
        db.flush_collection("legacy").expect("flush legacy");
    }

    fs::remove_file(root.join("wal.jsonl")).expect("remove historical wal");
    let replay_schema = CollectionSchema::new("vector", 2, "l2", Vec::new());
    append_wal_record(
        &root.join("wal.jsonl"),
        &WalRecord::CreateCollection {
            collection: "replay_only".to_string(),
            schema: replay_schema,
        },
    )
    .expect("append replay create");
    append_wal_record(
        &root.join("wal.jsonl"),
        &WalRecord::Insert {
            collection: "replay_only".to_string(),
            ids: vec![99],
            vectors: vec![0.0_f32, 0.0],
        },
    )
    .expect("append replay insert");

    let db = HannsDb::open(&root).expect("reopen db and replay wal");
    let legacy_hits = db
        .search("legacy", &[0.2_f32, 0.2], 1)
        .expect("legacy search should remain available");
    assert_eq!(legacy_hits.len(), 1);
    assert_eq!(
        legacy_hits[0].id, 7,
        "legacy collection must not be touched"
    );

    let replay_hits = db
        .search("replay_only", &[0.0_f32, 0.0], 1)
        .expect("replayed collection should be restored");
    assert_eq!(replay_hits.len(), 1);
    assert_eq!(replay_hits[0].id, 99);
}

#[test]
fn collection_api_flush_collection_succeeds_after_create_and_insert() {
    let root = unique_temp_dir("hannsdb_collection_api_flush_after_insert");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");
    db.insert("docs", &[10, 20], &[0.0_f32, 0.0, 1.0, 1.0])
        .expect("insert vectors");

    db.flush_collection("docs")
        .expect("flush should succeed after create+insert");
}

#[test]
fn collection_api_flush_collection_succeeds_after_create_insert_and_delete() {
    let root = unique_temp_dir("hannsdb_collection_api_flush_after_delete");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");
    db.insert("docs", &[10, 20], &[0.0_f32, 0.0, 1.0, 1.0])
        .expect("insert vectors");
    db.delete("docs", &[10]).expect("delete one id");

    db.flush_collection("docs")
        .expect("flush should succeed after create+delete");
}

#[test]
fn collection_api_flush_collection_missing_returns_not_found() {
    let root = unique_temp_dir("hannsdb_collection_api_flush_missing");
    let db = HannsDb::open(&root).expect("open db");

    let err = db
        .flush_collection("missing")
        .expect_err("missing collection flush should fail");
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
}

#[test]
fn collection_api_flush_collection_fails_when_wal_is_missing() {
    let root = unique_temp_dir("hannsdb_collection_api_flush_missing_wal");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");
    fs::remove_file(root.join("wal.jsonl")).expect("remove wal");

    let err = db
        .flush_collection("docs")
        .expect_err("flush should fail when wal is missing");
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
}

#[test]
fn collection_api_flush_collection_fails_when_segment_is_missing() {
    let root = unique_temp_dir("hannsdb_collection_api_flush_missing_segment");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");
    fs::remove_file(root.join("collections").join("docs").join("segment.json"))
        .expect("remove segment");

    let err = db
        .flush_collection("docs")
        .expect_err("flush should fail when segment metadata is missing");
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
}

#[test]
fn collection_api_flush_collection_fails_when_tombstones_are_missing() {
    let root = unique_temp_dir("hannsdb_collection_api_flush_missing_tombstones");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");
    fs::remove_file(
        root.join("collections")
            .join("docs")
            .join("tombstones.json"),
    )
    .expect("remove tombstones");

    let err = db
        .flush_collection("docs")
        .expect_err("flush should fail when tombstones metadata is missing");
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
}

#[test]
fn collection_api_optimize_warms_search_state_for_repeated_queries() {
    let root = unique_temp_dir("hannsdb_collection_api_optimize_cache");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");
    db.insert("docs", &[11, 22], &[0.0_f32, 0.0, 10.0, 10.0])
        .expect("insert vectors");
    db.optimize_collection("docs")
        .expect("optimize should warm search state");

    let collection_dir = root.join("collections").join("docs");
    fs::remove_file(collection_dir.join("records.bin")).expect("remove records.bin");
    fs::remove_file(collection_dir.join("ids.bin")).expect("remove ids.bin");

    let hits = db
        .search("docs", &[0.1_f32, -0.1], 1)
        .expect("cached search should still work");
    assert_eq!(hits[0].id, 11);
    assert!(
        (hits[0].distance - 0.14142136).abs() < 1e-6,
        "l2 distance should preserve brute-force semantics after optimize: got {}",
        hits[0].distance
    );
}

#[cfg(feature = "knowhere-backend")]
#[test]
fn collection_api_optimize_ann_maps_external_ids_and_normalizes_l2_distance() {
    let root = unique_temp_dir("hannsdb_collection_api_optimize_ann_mapping");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");
    db.insert("docs", &[-7, 42], &[1.0_f32, 1.0, 4.0, 5.0])
        .expect("insert vectors");

    let brute_hits = db
        .search("docs", &[2.0_f32, 1.0], 2)
        .expect("brute force search should work");
    assert_eq!(brute_hits[0].id, -7);

    db.optimize_collection("docs")
        .expect("optimize should build ann state");

    let collection_dir = root.join("collections").join("docs");
    fs::remove_file(collection_dir.join("records.bin")).expect("remove records.bin");
    fs::remove_file(collection_dir.join("ids.bin")).expect("remove ids.bin");

    let ann_hits = db
        .search("docs", &[2.0_f32, 1.0], 2)
        .expect("ann-backed search should work from optimized state");
    assert_eq!(ann_hits[0].id, -7, "ann hit id must map to external id");

    let expected = brute_hits[0].distance;
    let actual = ann_hits[0].distance;
    assert!(
        (expected - actual).abs() < 1e-6,
        "l2 distance must match brute-force semantics: expected {expected}, got {actual}"
    );
}

#[cfg(feature = "knowhere-backend")]
#[test]
fn collection_api_optimize_ann_preserves_ip_distance_semantics() {
    let root = unique_temp_dir("hannsdb_collection_api_optimize_ann_ip_distance");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs_ip", 2, "ip")
        .expect("create collection");
    db.insert("docs_ip", &[1001, 1002], &[2.0_f32, 2.0, 1.0, 0.0])
        .expect("insert vectors");

    let brute_hits = db
        .search("docs_ip", &[1.0_f32, 1.0], 1)
        .expect("brute-force search should work");
    assert_eq!(brute_hits[0].id, 1001);

    db.optimize_collection("docs_ip")
        .expect("optimize should build ann state");

    let collection_dir = root.join("collections").join("docs_ip");
    fs::remove_file(collection_dir.join("records.bin")).expect("remove records.bin");
    fs::remove_file(collection_dir.join("ids.bin")).expect("remove ids.bin");

    let ann_hits = db
        .search("docs_ip", &[1.0_f32, 1.0], 1)
        .expect("ann-backed ip search should work from optimized state");
    assert_eq!(ann_hits[0].id, 1001);
    assert!(
        (ann_hits[0].distance - brute_hits[0].distance).abs() < 1e-6,
        "ip distance must match brute-force semantics: expected {}, got {}",
        brute_hits[0].distance,
        ann_hits[0].distance
    );
}

#[cfg(feature = "knowhere-backend")]
#[test]
fn collection_api_reopen_loads_persisted_hnsw_index() {
    let root = unique_temp_dir("hannsdb_collection_api_reopen_loads_persisted_hnsw");
    {
        let mut db = HannsDb::open(&root).expect("open db");
        db.create_collection("docs", 2, "l2")
            .expect("create collection");
        db.insert("docs", &[11, 22], &[0.0_f32, 0.0, 10.0, 10.0])
            .expect("insert vectors");
        db.optimize_collection("docs")
            .expect("optimize should persist hnsw index");
    }

    let collection_dir = root.join("collections").join("docs");
    assert!(
        collection_dir.join("hnsw_index.bin").exists(),
        "persisted hnsw index should exist after optimize"
    );
    fs::remove_file(collection_dir.join("records.bin")).expect("remove records.bin");
    fs::remove_file(collection_dir.join("ids.bin")).expect("remove ids.bin");

    let db = HannsDb::open(&root).expect("reopen db");
    let hits = db
        .search_with_ef("docs", &[0.1_f32, -0.1], 1, 32)
        .expect("reopen search should use persisted hnsw index");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, 11);
}

#[test]
fn collection_api_optimize_benchmark_entry() {
    fn read_env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(default)
    }

    fn read_env_metric(name: &str, default: &str) -> String {
        std::env::var(name)
            .unwrap_or_else(|_| default.to_string())
            .to_ascii_lowercase()
    }

    fn synthetic_value(i: usize, j: usize) -> f32 {
        let x = (i as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add((j as u64).wrapping_mul(1442695040888963407))
            .wrapping_add(1);
        let y = ((x >> 16) as u32) as f32 / (u32::MAX as f32);
        y * 2.0 - 1.0
    }

    fn build_synthetic_vectors(n: usize, dim: usize, metric: &str) -> Vec<f32> {
        let mut vectors = vec![0.0_f32; n * dim];
        for i in 0..n {
            let mut norm_sq = 0.0_f32;
            let row = &mut vectors[i * dim..(i + 1) * dim];
            for (j, v) in row.iter_mut().enumerate() {
                let val = synthetic_value(i, j);
                *v = val;
                norm_sq += val * val;
            }
            if metric == "cosine" {
                let norm = norm_sq.sqrt();
                if norm > 0.0 {
                    for v in row.iter_mut() {
                        *v /= norm;
                    }
                }
            }
        }
        vectors
    }

    let n = read_env_usize("HANNSSDB_OPT_BENCH_N", 2000);
    let dim = read_env_usize("HANNSSDB_OPT_BENCH_DIM", 256);
    let top_k = read_env_usize("HANNSSDB_OPT_BENCH_TOPK", 10);
    let metric = read_env_metric("HANNSSDB_OPT_BENCH_METRIC", "cosine");
    assert!(
        matches!(metric.as_str(), "l2" | "cosine" | "ip"),
        "unsupported metric, expected one of l2/cosine/ip, got: {metric}"
    );

    let root = unique_temp_dir("hannsdb_collection_api_optimize_bench");
    let collection = "optimize_bench";
    let ids = (0..n as i64).collect::<Vec<_>>();
    let vectors = build_synthetic_vectors(n, dim, &metric);
    let query = vectors[..dim].to_vec();

    let bench_start = Instant::now();
    let mut db = HannsDb::open(&root).expect("open db");

    let create_start = Instant::now();
    db.create_collection(collection, dim, &metric)
        .expect("create collection");
    let create_ms = create_start.elapsed().as_millis();

    let insert_start = Instant::now();
    let inserted = db
        .insert(collection, &ids, &vectors)
        .expect("insert synthetic vectors");
    let insert_ms = insert_start.elapsed().as_millis();
    assert_eq!(inserted, n);

    let optimize_start = Instant::now();
    db.optimize_collection(collection)
        .expect("optimize collection");
    let optimize_ms = optimize_start.elapsed().as_millis();

    let search_start = Instant::now();
    let hits = db
        .search(collection, &query, top_k)
        .expect("search optimized collection");
    let search_ms = search_start.elapsed().as_millis();
    let total_ms = bench_start.elapsed().as_millis();

    assert!(
        !hits.is_empty(),
        "optimized search should return at least one hit"
    );

    println!(
        "OPT_BENCH_CONFIG n={} dim={} metric={} top_k={}",
        n, dim, metric, top_k
    );
    println!(
        "OPT_BENCH_TIMING_MS create={} insert={} optimize={} search={} total={}",
        create_ms, insert_ms, optimize_ms, search_ms, total_ms
    );
}

#[test]
fn collection_api_persists_index_descriptors_across_reopen() {
    let root = unique_temp_dir("hannsdb_collection_api_index_descriptor_persistence");
    let mut db = HannsDb::open(&root).expect("open db");
    let schema = CollectionSchema::new(
        "vector",
        2,
        "l2",
        vec![ScalarFieldSchema::new("category", FieldType::String)],
    );
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");

    let vector_descriptor = VectorIndexDescriptor {
        field_name: "vector".to_string(),
        kind: VectorIndexKind::Ivf,
        metric: Some("l2".to_string()),
        params: json!({
            "nlist": 8
        }),
    };
    let scalar_descriptor = ScalarIndexDescriptor {
        field_name: "category".to_string(),
        kind: ScalarIndexKind::Inverted,
        params: json!({
            "tokenizer": "keyword"
        }),
    };

    db.create_vector_index("docs", vector_descriptor.clone())
        .expect("register vector descriptor");
    db.create_scalar_index("docs", scalar_descriptor.clone())
        .expect("register scalar descriptor");

    assert_eq!(
        db.list_vector_indexes("docs")
            .expect("list vector descriptors"),
        vec![vector_descriptor.clone()]
    );
    assert_eq!(
        db.list_scalar_indexes("docs")
            .expect("list scalar descriptors"),
        vec![scalar_descriptor.clone()]
    );
    assert!(
        root.join("collections")
            .join("docs")
            .join("indexes.json")
            .exists(),
        "index metadata file should be written"
    );

    let reopened = HannsDb::open(&root).expect("reopen db");
    assert_eq!(
        reopened
            .list_vector_indexes("docs")
            .expect("list vector descriptors after reopen"),
        vec![vector_descriptor]
    );
    assert_eq!(
        reopened
            .list_scalar_indexes("docs")
            .expect("list scalar descriptors after reopen"),
        vec![scalar_descriptor]
    );
}

#[test]
fn collection_api_drop_vector_index_keeps_other_descriptors() {
    let root = unique_temp_dir("hannsdb_collection_api_drop_vector_index_descriptor");
    let mut db = HannsDb::open(&root).expect("open db");
    let schema = CollectionSchema::new(
        "vector",
        2,
        "l2",
        vec![ScalarFieldSchema::new("category", FieldType::String)],
    );
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");

    db.create_vector_index(
        "docs",
        VectorIndexDescriptor {
            field_name: "vector".to_string(),
            kind: VectorIndexKind::Flat,
            metric: Some("l2".to_string()),
            params: json!({}),
        },
    )
    .expect("register vector descriptor");
    db.create_scalar_index(
        "docs",
        ScalarIndexDescriptor {
            field_name: "category".to_string(),
            kind: ScalarIndexKind::Inverted,
            params: json!({
                "tokenizer": "keyword"
            }),
        },
    )
    .expect("register scalar descriptor");

    db.drop_vector_index("docs", "vector")
        .expect("drop vector descriptor");

    assert!(
        db.list_vector_indexes("docs")
            .expect("list vector descriptors")
            .is_empty(),
        "vector descriptor should be removed"
    );
    assert_eq!(
        db.list_scalar_indexes("docs")
            .expect("list scalar descriptors")
            .len(),
        1
    );
}
