use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use hannsdb_core::db::HannsDb;
use hannsdb_core::document::{
    CollectionSchema, Document, FieldType, FieldValue, ScalarFieldSchema,
};
use hannsdb_core::query::{
    QueryContext, QueryGroupBy, QueryReranker, VectorQuery, VectorQueryParam,
};
use hannsdb_core::segment::{
    append_payloads, append_record_ids, append_records, SegmentMetadata, SegmentSet, TombstoneMask,
};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("repo root")
        .to_path_buf()
}

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

    let mut ids = Vec::with_capacity(second_segment_documents.len());
    let mut vectors = Vec::with_capacity(second_segment_documents.len() * dimension);
    let mut payloads = Vec::with_capacity(second_segment_documents.len());
    for document in second_segment_documents {
        ids.push(document.id);
        vectors.extend_from_slice(&document.vector);
        payloads.push(document.fields.clone());
    }

    let inserted =
        append_records(&seg2_dir.join("records.bin"), dimension, &vectors).expect("append records");
    assert_eq!(inserted, second_segment_documents.len());
    let _ = append_record_ids(&seg2_dir.join("ids.bin"), &ids).expect("append ids");
    let _ = append_payloads(&seg2_dir.join("payloads.jsonl"), &payloads).expect("append payloads");

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
    .expect("save segment set");
}

#[test]
fn zvec_parity_schema_query_surface_compiles_against_typed_batch_request() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let crate_dir = tempdir.path().join("query-surface-check");
    let src_dir = crate_dir.join("src");
    fs::create_dir_all(&src_dir).expect("create src dir");

    let core_path = repo_root().join("crates/hannsdb-core");
    fs::write(
        crate_dir.join("Cargo.toml"),
        format!(
            r#"[package]
name = "query-surface-check"
version = "0.1.0"
edition = "2021"

[dependencies]
hannsdb_core = {{ package = "hannsdb-core", path = "{}" }}
"#,
            core_path.display()
        ),
    )
    .expect("write Cargo.toml");

    fs::write(
        src_dir.join("main.rs"),
        r#"use hannsdb_core::query::{QueryContext, VectorQuery};

fn main() {
    let query = VectorQuery {
        field_name: "dense".to_string(),
        vector: vec![0.0_f32, 0.1],
        param: None,
    };
    let _request = QueryContext {
        top_k: 8,
        queries: vec![query],
        query_by_id: Some(vec![11, 22]),
        filter: Some("group == 1".to_string()),
        output_fields: Some(vec!["group".to_string()]),
        include_vector: false,
        group_by: None,
        reranker: None,
    };
}
"#,
    )
    .expect("write main.rs");

    let output = Command::new("cargo")
        .arg("check")
        .arg("--quiet")
        .current_dir(&crate_dir)
        .output()
        .expect("run cargo check");

    assert!(
        output.status.success(),
        "cargo check failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn zvec_parity_query_context_merges_vector_and_query_by_id_sources_with_filter() {
    let root = unique_temp_dir("hannsdb_typed_query_context_merge");
    let mut db = HannsDb::open(&root).expect("open db");
    let schema = CollectionSchema::new(
        "vector",
        2,
        "l2",
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");
    db.insert_documents(
        "docs",
        &[
            Document::new(
                1,
                [("group".to_string(), FieldValue::Int64(1))],
                vec![0.0_f32, 0.0],
            ),
            Document::new(
                2,
                [("group".to_string(), FieldValue::Int64(1))],
                vec![0.2_f32, 0.0],
            ),
            Document::new(
                3,
                [("group".to_string(), FieldValue::Int64(1))],
                vec![0.1_f32, 0.0],
            ),
            Document::new(
                4,
                [("group".to_string(), FieldValue::Int64(2))],
                vec![0.05_f32, 0.0],
            ),
            Document::new(
                5,
                [("group".to_string(), FieldValue::Int64(1))],
                vec![10.0_f32, 10.0],
            ),
        ],
    )
    .expect("insert documents");

    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 3,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: vec![0.0_f32, 0.0],
                    param: None,
                }],
                query_by_id: Some(vec![2]),
                filter: Some("group == 1".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
            },
        )
        .expect("query with merged recall sources");

    let hit_ids = hits.iter().map(|hit| hit.id).collect::<Vec<_>>();
    assert_eq!(hit_ids, vec![1, 2, 3]);
    assert_eq!(hits.len(), 3, "merged recall sources should dedupe ids");
    assert_eq!(hits[0].distance, 0.0);
    assert_eq!(hits[1].distance, 0.0);
    assert_eq!(
        hits.iter()
            .map(|hit| hit.fields.get("group"))
            .collect::<Vec<_>>(),
        vec![
            Some(&FieldValue::Int64(1)),
            Some(&FieldValue::Int64(1)),
            Some(&FieldValue::Int64(1)),
        ]
    );
}

#[test]
fn zvec_parity_query_context_group_by_returns_best_hit_per_group() {
    let root = unique_temp_dir("hannsdb_typed_query_group_by_recall");
    let mut db = HannsDb::open(&root).expect("open db");
    let schema = CollectionSchema::new(
        "vector",
        2,
        "l2",
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");
    db.insert_documents(
        "docs",
        &[
            Document::new(
                10,
                [("group".to_string(), FieldValue::Int64(1))],
                vec![0.0_f32, 0.0],
            ),
            Document::new(
                11,
                [("group".to_string(), FieldValue::Int64(1))],
                vec![0.1_f32, 0.0],
            ),
            Document::new(
                20,
                [("group".to_string(), FieldValue::Int64(2))],
                vec![0.05_f32, 0.0],
            ),
            Document::new(
                21,
                [("group".to_string(), FieldValue::Int64(2))],
                vec![0.2_f32, 0.0],
            ),
            Document::new(
                30,
                [("group".to_string(), FieldValue::Int64(3))],
                vec![0.15_f32, 0.0],
            ),
        ],
    )
    .expect("insert documents");

    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 2,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: vec![0.0_f32, 0.0],
                    param: None,
                }],
                query_by_id: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: Some(QueryGroupBy {
                    field_name: "group".to_string(),
                }),
                reranker: None,
            },
        )
        .expect("group_by should keep the best hit from each group");

    let hit_ids = hits.iter().map(|hit| hit.id).collect::<Vec<_>>();
    assert_eq!(hit_ids, vec![10, 20]);
    assert_eq!(hits.len(), 2, "top_k should cap the number of groups");
    assert_eq!(hits[0].fields.get("group"), Some(&FieldValue::Int64(1)));
    assert_eq!(hits[1].fields.get("group"), Some(&FieldValue::Int64(2)));
    assert!(
        hits[0].distance < hits[1].distance,
        "winning groups should preserve the original ranking order"
    );
}

#[test]
fn zvec_parity_query_context_rejects_group_by_on_invalid_or_vector_field() {
    let root = unique_temp_dir("hannsdb_typed_query_group_by_invalid_field");
    let mut db = HannsDb::open(&root).expect("open db");
    let schema = CollectionSchema::new(
        "vector",
        2,
        "l2",
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");

    let missing_field_err = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 3,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: vec![0.0_f32, 0.0],
                    param: None,
                }],
                query_by_id: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: Some(QueryGroupBy {
                    field_name: "missing".to_string(),
                }),
                reranker: None,
            },
        )
        .expect_err("group_by should reject an undefined field");

    assert_eq!(missing_field_err.kind(), std::io::ErrorKind::InvalidInput);
    assert!(missing_field_err.to_string().contains("group_by"));
    assert!(missing_field_err.to_string().contains("missing"));

    let vector_field_err = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 3,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: vec![0.0_f32, 0.0],
                    param: None,
                }],
                query_by_id: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: Some(QueryGroupBy {
                    field_name: "vector".to_string(),
                }),
                reranker: None,
            },
        )
        .expect_err("group_by should reject vector fields");

    assert_eq!(vector_field_err.kind(), std::io::ErrorKind::InvalidInput);
    assert!(vector_field_err.to_string().contains("group_by"));
    assert!(vector_field_err.to_string().contains("vector"));
}

#[test]
fn zvec_parity_query_context_rejects_filter_only_group_by_in_this_slice() {
    let root = unique_temp_dir("hannsdb_typed_query_group_by_filter_only");
    let mut db = HannsDb::open(&root).expect("open db");
    let schema = CollectionSchema::new(
        "vector",
        2,
        "l2",
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");

    let err = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 3,
                queries: Vec::new(),
                query_by_id: None,
                filter: Some("group == 1".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: Some(QueryGroupBy {
                    field_name: "group".to_string(),
                }),
                reranker: None,
            },
        )
        .expect_err("filter-only group_by should remain unsupported in this slice");

    assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
    assert!(err.to_string().contains("group_by"));
}

#[test]
fn zvec_parity_query_context_rejects_reranker_until_supported() {
    let root = unique_temp_dir("hannsdb_typed_query_reranker");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");

    let err = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 3,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: vec![0.0_f32, 0.0],
                    param: None,
                }],
                query_by_id: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: Some(QueryReranker {
                    model: "cross-encoder".to_string(),
                }),
            },
        )
        .expect_err("reranker should be rejected in the first slice");

    assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
    assert!(err.to_string().contains("reranker"));
}

#[test]
fn zvec_parity_query_context_prefers_newer_segment_version_over_better_old_match() {
    let root = unique_temp_dir("hannsdb_typed_query_segment_shadowing");
    let mut db = HannsDb::open(&root).expect("open db");
    let schema = CollectionSchema::new(
        "vector",
        2,
        "l2",
        vec![ScalarFieldSchema::new("version", FieldType::String)],
    );
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");
    db.insert_documents(
        "docs",
        &[
            Document::new(
                7,
                [("version".to_string(), FieldValue::String("old".to_string()))],
                vec![0.0_f32, 0.0],
            ),
            Document::new(
                8,
                [(
                    "version".to_string(),
                    FieldValue::String("stable".to_string()),
                )],
                vec![0.1_f32, 0.0],
            ),
        ],
    )
    .expect("insert seg-0001 docs");

    rewrite_collection_to_two_segment_layout(
        &root,
        "docs",
        2,
        &[Document::new(
            7,
            [("version".to_string(), FieldValue::String("new".to_string()))],
            vec![5.0_f32, 5.0],
        )],
        &[],
    );

    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 2,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: vec![0.0_f32, 0.0],
                    param: None,
                }],
                query_by_id: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
            },
        )
        .expect("query with shadowed duplicate ids");

    let hit_ids = hits.iter().map(|hit| hit.id).collect::<Vec<_>>();
    assert_eq!(hit_ids, vec![8, 7]);
    assert_eq!(
        hits[1].fields.get("version"),
        Some(&FieldValue::String("new".to_string()))
    );
    assert!(
        hits[1].distance > hits[0].distance,
        "newer row should shadow the old one even if it is farther away"
    );
}

#[test]
fn zvec_parity_query_context_tombstoned_newer_duplicate_still_shadows_older_segment_row() {
    let root = unique_temp_dir("hannsdb_typed_query_tombstoned_shadowing");
    let mut db = HannsDb::open(&root).expect("open db");
    let schema = CollectionSchema::new(
        "vector",
        2,
        "l2",
        vec![ScalarFieldSchema::new("version", FieldType::String)],
    );
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");
    db.insert_documents(
        "docs",
        &[
            Document::new(
                7,
                [("version".to_string(), FieldValue::String("old".to_string()))],
                vec![0.0_f32, 0.0],
            ),
            Document::new(
                8,
                [(
                    "version".to_string(),
                    FieldValue::String("stable".to_string()),
                )],
                vec![0.1_f32, 0.0],
            ),
        ],
    )
    .expect("insert seg-0001 docs");

    rewrite_collection_to_two_segment_layout(
        &root,
        "docs",
        2,
        &[Document::new(
            7,
            [(
                "version".to_string(),
                FieldValue::String("new-deleted".to_string()),
            )],
            vec![0.05_f32, 0.0],
        )],
        &[0],
    );

    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 2,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: vec![0.0_f32, 0.0],
                    param: None,
                }],
                query_by_id: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
            },
        )
        .expect("query with tombstoned shadowed duplicate ids");

    let hit_ids = hits.iter().map(|hit| hit.id).collect::<Vec<_>>();
    assert_eq!(hit_ids, vec![8]);
}

#[test]
fn zvec_parity_query_by_id_rejects_older_segment_row_when_newer_state_is_tombstoned() {
    let root = unique_temp_dir("hannsdb_typed_query_by_id_tombstoned_shadowing");
    let mut db = HannsDb::open(&root).expect("open db");
    let schema = CollectionSchema::new(
        "vector",
        2,
        "l2",
        vec![ScalarFieldSchema::new("version", FieldType::String)],
    );
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");
    db.insert_documents(
        "docs",
        &[
            Document::new(
                7,
                [("version".to_string(), FieldValue::String("old".to_string()))],
                vec![0.0_f32, 0.0],
            ),
            Document::new(
                8,
                [(
                    "version".to_string(),
                    FieldValue::String("stable".to_string()),
                )],
                vec![0.1_f32, 0.0],
            ),
        ],
    )
    .expect("insert seg-0001 docs");

    rewrite_collection_to_two_segment_layout(
        &root,
        "docs",
        2,
        &[Document::new(
            7,
            [(
                "version".to_string(),
                FieldValue::String("new-deleted".to_string()),
            )],
            vec![0.05_f32, 0.0],
        )],
        &[0],
    );

    let err = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 1,
                queries: Vec::new(),
                query_by_id: Some(vec![7]),
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
            },
        )
        .expect_err("query_by_id should not resolve a tombstoned newer duplicate");

    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    assert!(err.to_string().contains("query_by_id"));
}

#[test]
fn zvec_parity_query_context_single_vector_ef_search_matches_legacy_search_path() {
    let root = unique_temp_dir("hannsdb_typed_query_single_vector_ef_search");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");
    db.insert_documents(
        "docs",
        &[
            Document::new(1, [], vec![0.0_f32, 0.0]),
            Document::new(2, [], vec![0.2_f32, 0.0]),
            Document::new(3, [], vec![0.1_f32, 0.0]),
            Document::new(4, [], vec![5.0_f32, 5.0]),
        ],
    )
    .expect("insert documents");

    let legacy_hits = db
        .search_with_ef("docs", &[0.0_f32, 0.0], 3, 64)
        .expect("legacy search_with_ef");

    let typed_hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 3,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: vec![0.0_f32, 0.0],
                    param: Some(VectorQueryParam {
                        ef_search: Some(64),
                    }),
                }],
                query_by_id: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
            },
        )
        .expect("typed query should reuse the legacy single-vector path");

    assert_eq!(
        typed_hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        legacy_hits.iter().map(|hit| hit.id).collect::<Vec<_>>()
    );
    assert_eq!(
        typed_hits
            .iter()
            .map(|hit| hit.distance)
            .collect::<Vec<_>>(),
        legacy_hits
            .iter()
            .map(|hit| hit.distance)
            .collect::<Vec<_>>()
    );
}

#[test]
fn zvec_parity_query_context_rejects_ef_search_on_query_by_id_merge_shape() {
    let root = unique_temp_dir("hannsdb_typed_query_params");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");
    db.insert_documents("docs", &[Document::new(7, [], vec![0.0_f32, 0.0])])
        .expect("insert documents");

    let err = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 2,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: vec![0.0_f32, 0.0],
                    param: Some(VectorQueryParam {
                        ef_search: Some(64),
                    }),
                }],
                query_by_id: Some(vec![7]),
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
            },
        )
        .expect_err("mixed query_by_id + ef_search should be rejected");

    assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
    assert!(err.to_string().contains("ef_search"));
}

#[test]
fn zvec_parity_filter_only_query_returns_live_docs_in_id_order_and_respects_top_k() {
    let root = unique_temp_dir("hannsdb_typed_query_filter_only");
    let mut db = HannsDb::open(&root).expect("open db");
    let schema = CollectionSchema::new(
        "vector",
        2,
        "l2",
        vec![
            ScalarFieldSchema::new("group", FieldType::Int64),
            ScalarFieldSchema::new("version", FieldType::String),
        ],
    );
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");
    db.insert_documents(
        "docs",
        &[
            Document::new(
                1,
                [
                    ("group".to_string(), FieldValue::Int64(1)),
                    (
                        "version".to_string(),
                        FieldValue::String("seg1".to_string()),
                    ),
                ],
                vec![1.0_f32, 1.0],
            ),
            Document::new(
                2,
                [
                    ("group".to_string(), FieldValue::Int64(1)),
                    ("version".to_string(), FieldValue::String("old".to_string())),
                ],
                vec![2.0_f32, 2.0],
            ),
            Document::new(
                4,
                [
                    ("group".to_string(), FieldValue::Int64(2)),
                    (
                        "version".to_string(),
                        FieldValue::String("other-group".to_string()),
                    ),
                ],
                vec![4.0_f32, 4.0],
            ),
        ],
    )
    .expect("insert seg-0001 docs");

    rewrite_collection_to_two_segment_layout(
        &root,
        "docs",
        2,
        &[
            Document::new(
                2,
                [
                    ("group".to_string(), FieldValue::Int64(1)),
                    ("version".to_string(), FieldValue::String("new".to_string())),
                ],
                vec![20.0_f32, 20.0],
            ),
            Document::new(
                3,
                [
                    ("group".to_string(), FieldValue::Int64(1)),
                    (
                        "version".to_string(),
                        FieldValue::String("deleted".to_string()),
                    ),
                ],
                vec![30.0_f32, 30.0],
            ),
            Document::new(
                5,
                [
                    ("group".to_string(), FieldValue::Int64(1)),
                    (
                        "version".to_string(),
                        FieldValue::String("fresh".to_string()),
                    ),
                ],
                vec![50.0_f32, 50.0],
            ),
        ],
        &[1],
    );

    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 2,
                queries: Vec::new(),
                query_by_id: None,
                filter: Some("group == 1".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
            },
        )
        .expect("filter-only query should scan live documents");

    assert_eq!(
        hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        vec![1, 2]
    );
    assert_eq!(
        hits.iter().map(|hit| hit.distance).collect::<Vec<_>>(),
        vec![0.0, 0.0]
    );
    assert_eq!(
        hits[1].fields.get("version"),
        Some(&FieldValue::String("new".to_string()))
    );
}

#[test]
fn zvec_parity_query_context_projects_output_fields_on_typed_hits() {
    let root = unique_temp_dir("hannsdb_typed_query_output_fields");
    let mut db = HannsDb::open(&root).expect("open db");
    let schema = CollectionSchema::new(
        "vector",
        2,
        "l2",
        vec![
            ScalarFieldSchema::new("group", FieldType::Int64),
            ScalarFieldSchema::new("color", FieldType::String),
        ],
    );
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");
    db.insert_documents(
        "docs",
        &[
            Document::new(
                1,
                [
                    ("group".to_string(), FieldValue::Int64(1)),
                    ("color".to_string(), FieldValue::String("red".to_string())),
                ],
                vec![0.0_f32, 0.0],
            ),
            Document::new(
                2,
                [
                    ("group".to_string(), FieldValue::Int64(2)),
                    ("color".to_string(), FieldValue::String("blue".to_string())),
                ],
                vec![1.0_f32, 1.0],
            ),
        ],
    )
    .expect("insert documents");

    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 2,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: vec![0.0_f32, 0.0],
                    param: None,
                }],
                query_by_id: None,
                filter: None,
                output_fields: Some(vec!["color".to_string()]),
                include_vector: false,
                group_by: None,
                reranker: None,
            },
        )
        .expect("typed query with output field projection");

    assert_eq!(
        hits[0].fields,
        [("color".to_string(), FieldValue::String("red".to_string()),)]
            .into_iter()
            .collect()
    );
    assert_eq!(
        hits[1].fields,
        [("color".to_string(), FieldValue::String("blue".to_string()),)]
            .into_iter()
            .collect()
    );
}

#[test]
fn zvec_parity_query_context_rejects_include_vector_until_supported() {
    let root = unique_temp_dir("hannsdb_typed_query_include_vector");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");

    let err = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 1,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: vec![0.0_f32, 0.0],
                    param: None,
                }],
                query_by_id: None,
                filter: None,
                output_fields: None,
                include_vector: true,
                group_by: None,
                reranker: None,
            },
        )
        .expect_err("include_vector should be rejected until the result type carries vectors");

    assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
    assert!(err.to_string().contains("include_vector"));
}

#[test]
fn zvec_parity_query_context_rejects_include_vector_before_query_by_id_resolution() {
    let root = unique_temp_dir("hannsdb_typed_query_include_vector_query_by_id_guard");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");
    db.insert_documents("docs", &[Document::new(7, [], vec![0.0_f32, 0.0])])
        .expect("insert documents");

    fs::remove_file(
        root.join("collections")
            .join("docs")
            .join("tombstones.json"),
    )
    .expect("remove tombstones to prove query_by_id resolution is not reached");

    let err = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 1,
                queries: Vec::new(),
                query_by_id: Some(vec![7]),
                filter: Some("1 == 1".to_string()),
                output_fields: None,
                include_vector: true,
                group_by: None,
                reranker: None,
            },
        )
        .expect_err("include_vector should fail before query_by_id resolution touches storage");

    assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
    assert!(err.to_string().contains("include_vector"));
}

#[test]
fn zvec_parity_filter_only_query_projects_output_fields() {
    let root = unique_temp_dir("hannsdb_typed_query_filter_only_output_fields");
    let mut db = HannsDb::open(&root).expect("open db");
    let schema = CollectionSchema::new(
        "vector",
        2,
        "l2",
        vec![
            ScalarFieldSchema::new("group", FieldType::Int64),
            ScalarFieldSchema::new("color", FieldType::String),
        ],
    );
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");
    db.insert_documents(
        "docs",
        &[
            Document::new(
                1,
                [
                    ("group".to_string(), FieldValue::Int64(1)),
                    ("color".to_string(), FieldValue::String("red".to_string())),
                ],
                vec![0.0_f32, 0.0],
            ),
            Document::new(
                2,
                [
                    ("group".to_string(), FieldValue::Int64(1)),
                    ("color".to_string(), FieldValue::String("blue".to_string())),
                ],
                vec![1.0_f32, 1.0],
            ),
        ],
    )
    .expect("insert documents");

    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 2,
                queries: Vec::new(),
                query_by_id: None,
                filter: Some("group == 1".to_string()),
                output_fields: Some(vec!["color".to_string()]),
                include_vector: false,
                group_by: None,
                reranker: None,
            },
        )
        .expect("filter-only query with output field projection");

    assert_eq!(
        hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        vec![1, 2]
    );
    assert_eq!(
        hits[0].fields,
        [("color".to_string(), FieldValue::String("red".to_string())),]
            .into_iter()
            .collect()
    );
    assert_eq!(
        hits[1].fields,
        [("color".to_string(), FieldValue::String("blue".to_string())),]
            .into_iter()
            .collect()
    );
}
