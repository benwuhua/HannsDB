use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use hannsdb_core::db::HannsDb;
use hannsdb_core::document::{
    CollectionSchema, Document, FieldType, FieldValue, ScalarFieldSchema, VectorFieldSchema,
    VectorIndexSchema,
};
use hannsdb_core::query::{
    QueryContext, QueryGroupBy, QueryReranker, QueryVector, VectorQuery, VectorQueryParam,
};
use hannsdb_core::segment::{
    append_payloads, append_record_ids, append_records, append_vectors, SegmentMetadata,
    SegmentSet, TombstoneMask,
};
use hannsdb_index::descriptor::{VectorIndexDescriptor, VectorIndexKind};

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
        vectors.extend_from_slice(document.primary_vector());
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

fn rewrite_collection_to_two_segment_layout_with_secondary_vectors(
    root: &std::path::Path,
    collection: &str,
    dimension: usize,
    primary_vector_name: &str,
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
        "vectors.jsonl",
        "tombstones.json",
    ] {
        fs::rename(collection_dir.join(file), seg1_dir.join(file)).expect("move seg-0001 file");
    }

    let mut ids = Vec::with_capacity(second_segment_documents.len());
    let mut vectors = Vec::with_capacity(second_segment_documents.len() * dimension);
    let mut payloads = Vec::with_capacity(second_segment_documents.len());
    let mut vector_sidecars = Vec::with_capacity(second_segment_documents.len());
    for document in second_segment_documents {
        ids.push(document.id);
        vectors.extend_from_slice(
            document
                .primary_vector_for(primary_vector_name)
                .expect("document must have primary vector"),
        );
        payloads.push(document.fields.clone());
        vector_sidecars.push(document.vectors.clone());
    }

    let inserted =
        append_records(&seg2_dir.join("records.bin"), dimension, &vectors).expect("append records");
    assert_eq!(inserted, second_segment_documents.len());
    let _ = append_record_ids(&seg2_dir.join("ids.bin"), &ids).expect("append ids");
    let _ = append_payloads(&seg2_dir.join("payloads.jsonl"), &payloads).expect("append payloads");
    let _ =
        append_vectors(&seg2_dir.join("vectors.jsonl"), &vector_sidecars).expect("append vectors");

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
        r#"use hannsdb_core::query::{QueryContext, QueryVector, VectorQuery};

fn main() {
    let query = VectorQuery {
        field_name: "dense".to_string(),
        vector: QueryVector::Dense(vec![0.0_f32, 0.1]),
        param: None,
    };
    let _request = QueryContext {
        top_k: 8,
        queries: vec![query],
        query_by_id: Some(vec![11, 22]),
        query_by_id_field_name: Some("secondary".to_string()),
        filter: Some("group == 1".to_string()),
        output_fields: Some(vec!["group".to_string()]),
        include_vector: false,
        group_by: None,
        reranker: None,
                order_by: None,
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
                    vector: QueryVector::Dense(vec![0.0_f32, 0.0]),
                    param: None,
                }],
                query_by_id: Some(vec![2]),
                query_by_id_field_name: None,
                filter: Some("group == 1".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
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
fn zvec_parity_query_context_supports_secondary_vector_field_on_typed_bruteforce_path() {
    let root = unique_temp_dir("hannsdb_typed_query_secondary_vector");
    let mut db = HannsDb::open(&root).expect("open db");
    let mut schema = CollectionSchema::new(
        "dense",
        2,
        "l2",
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    schema.vectors.push(VectorFieldSchema::new("title", 2));
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");
    db.insert_documents(
        "docs",
        &[
            Document::with_named_vectors(
                1,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![5.0_f32, 5.0],
                [("title".to_string(), vec![0.0_f32, 0.0])],
            ),
            Document::with_named_vectors(
                2,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![0.0_f32, 0.0],
                [("title".to_string(), vec![0.2_f32, 0.0])],
            ),
            Document::with_named_vectors(
                3,
                [("group".to_string(), FieldValue::Int64(2))],
                "dense",
                vec![1.0_f32, 1.0],
                [("title".to_string(), vec![1.0_f32, 0.0])],
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
                    field_name: "title".to_string(),
                    vector: QueryVector::Dense(vec![0.0_f32, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("secondary vector typed query");

    assert_eq!(
        hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    assert_eq!(
        hits.iter().map(|hit| hit.distance).collect::<Vec<_>>(),
        vec![0.0, 0.2, 1.0]
    );
}

#[test]
fn zvec_parity_query_context_supports_schema_indexed_secondary_vector_field_on_typed_fast_path_with_ef_search(
) {
    let root = unique_temp_dir("hannsdb_typed_query_secondary_schema_indexed_fast_path");
    let mut db = HannsDb::open(&root).expect("open db");
    let mut schema = CollectionSchema::new(
        "dense",
        2,
        "l2",
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    schema
        .vectors
        .push(VectorFieldSchema::new("title", 2).with_index_param(
            hannsdb_core::document::VectorIndexSchema::hnsw(Some("l2"), 16, 128),
        ));
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");
    db.insert_documents(
        "docs",
        &[
            Document::with_named_vectors(
                1,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![5.0_f32, 5.0],
                [("title".to_string(), vec![0.0_f32, 0.0])],
            ),
            Document::with_named_vectors(
                2,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![0.0_f32, 0.0],
                [("title".to_string(), vec![2.0_f32, 0.0])],
            ),
            Document::with_named_vectors(
                3,
                [("group".to_string(), FieldValue::Int64(2))],
                "dense",
                vec![1.0_f32, 1.0],
                [("title".to_string(), vec![1.0_f32, 0.0])],
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
                    field_name: "title".to_string(),
                    vector: QueryVector::Dense(vec![0.0_f32, 0.0]),
                    param: Some(VectorQueryParam {
                        ef_search: Some(64),
                    }),
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("schema-indexed secondary vector fast path");

    assert_eq!(
        hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        vec![1, 3, 2]
    );
    assert_eq!(
        hits.iter().map(|hit| hit.distance).collect::<Vec<_>>(),
        vec![0.0, 1.0, 2.0]
    );
}

#[test]
fn zvec_parity_query_context_supports_schema_indexed_secondary_vector_field_with_dimension_mismatch_between_primary_and_secondary(
) {
    let root = unique_temp_dir("hannsdb_typed_query_secondary_schema_indexed_dimension_mismatch");
    let mut db = HannsDb::open(&root).expect("open db");
    let mut schema = CollectionSchema::new(
        "dense",
        3,
        "l2",
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    schema
        .vectors
        .push(VectorFieldSchema::new("title", 2).with_index_param(
            hannsdb_core::document::VectorIndexSchema::hnsw(Some("l2"), 16, 128),
        ));
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");
    db.insert_documents(
        "docs",
        &[
            Document::with_named_vectors(
                1,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![5.0_f32, 5.0, 5.0],
                [("title".to_string(), vec![0.0_f32, 0.0])],
            ),
            Document::with_named_vectors(
                2,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![0.0_f32, 0.0, 0.0],
                [("title".to_string(), vec![0.2_f32, 0.0])],
            ),
            Document::with_named_vectors(
                3,
                [("group".to_string(), FieldValue::Int64(2))],
                "dense",
                vec![1.0_f32, 1.0, 1.0],
                [("title".to_string(), vec![1.0_f32, 0.0])],
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
                    field_name: "title".to_string(),
                    vector: QueryVector::Dense(vec![0.0_f32, 0.0]),
                    param: Some(VectorQueryParam {
                        ef_search: Some(64),
                    }),
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("schema-indexed secondary vector fast path across dimension mismatch");

    assert_eq!(
        hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    assert_eq!(
        hits.iter().map(|hit| hit.distance).collect::<Vec<_>>(),
        vec![0.0, 0.2, 1.0]
    );
}

#[test]
fn zvec_parity_query_context_supports_descriptor_backed_secondary_vector_field_on_typed_fast_path_with_ef_search(
) {
    let root = unique_temp_dir("hannsdb_typed_query_secondary_descriptor_indexed_fast_path");
    let mut db = HannsDb::open(&root).expect("open db");
    let mut schema = CollectionSchema::new(
        "dense",
        2,
        "l2",
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    schema.vectors.push(VectorFieldSchema::new("title", 2));
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");
    db.create_vector_index(
        "docs",
        VectorIndexDescriptor {
            field_name: "title".to_string(),
            kind: VectorIndexKind::Hnsw,
            metric: Some("l2".to_string()),
            params: serde_json::json!({
                "m": 16,
                "ef_construction": 128,
            }),
        },
    )
    .expect("register descriptor-backed secondary vector index");
    db.insert_documents(
        "docs",
        &[
            Document::with_named_vectors(
                1,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![5.0_f32, 5.0],
                [("title".to_string(), vec![0.0_f32, 0.0])],
            ),
            Document::with_named_vectors(
                2,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![0.0_f32, 0.0],
                [("title".to_string(), vec![2.0_f32, 0.0])],
            ),
            Document::with_named_vectors(
                3,
                [("group".to_string(), FieldValue::Int64(2))],
                "dense",
                vec![1.0_f32, 1.0],
                [("title".to_string(), vec![1.0_f32, 0.0])],
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
                    field_name: "title".to_string(),
                    vector: QueryVector::Dense(vec![0.0_f32, 0.0]),
                    param: Some(VectorQueryParam {
                        ef_search: Some(64),
                    }),
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("descriptor-backed secondary vector fast path");

    assert_eq!(
        hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        vec![1, 3, 2]
    );
    assert_eq!(
        hits.iter().map(|hit| hit.distance).collect::<Vec<_>>(),
        vec![0.0, 1.0, 2.0]
    );
}

#[test]
fn zvec_parity_query_context_rejects_ef_search_for_unindexed_secondary_vector_field() {
    let root = unique_temp_dir("hannsdb_typed_query_secondary_unindexed_ef_search");
    let mut db = HannsDb::open(&root).expect("open db");
    let mut schema = CollectionSchema::new(
        "dense",
        2,
        "l2",
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    schema.vectors.push(VectorFieldSchema::new("title", 2));
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");
    db.insert_documents(
        "docs",
        &[
            Document::with_named_vectors(
                1,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![5.0_f32, 5.0],
                [("title".to_string(), vec![0.0_f32, 0.0])],
            ),
            Document::with_named_vectors(
                2,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![0.0_f32, 0.0],
                [("title".to_string(), vec![2.0_f32, 0.0])],
            ),
        ],
    )
    .expect("insert documents");

    let err = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 2,
                queries: vec![VectorQuery {
                    field_name: "title".to_string(),
                    vector: QueryVector::Dense(vec![0.0_f32, 0.0]),
                    param: Some(VectorQueryParam {
                        ef_search: Some(64),
                    }),
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect_err("unindexed secondary ef_search should remain unsupported");

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert!(err.to_string().contains("ef_search"));
}

#[test]
fn zvec_parity_query_context_uses_secondary_vector_field_metric_on_typed_bruteforce_path() {
    let root = unique_temp_dir("hannsdb_typed_query_secondary_vector_metric");
    let mut db = HannsDb::open(&root).expect("open db");
    let mut schema = CollectionSchema::new(
        "dense",
        2,
        "l2",
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    schema.vectors.push(VectorFieldSchema::new("title", 2));
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");
    db.create_vector_index(
        "docs",
        VectorIndexDescriptor {
            field_name: "title".to_string(),
            kind: VectorIndexKind::Hnsw,
            metric: Some("cosine".to_string()),
            params: serde_json::json!({
                "m": 16,
                "ef_construction": 128,
            }),
        },
    )
    .expect("register descriptor-backed secondary vector index");
    db.insert_documents(
        "docs",
        &[
            Document::with_named_vectors(
                1,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![5.0_f32, 5.0],
                [("title".to_string(), vec![0.9_f32, 0.1])],
            ),
            Document::with_named_vectors(
                2,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![0.0_f32, 0.0],
                [("title".to_string(), vec![10.0_f32, 0.0])],
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
                    field_name: "title".to_string(),
                    vector: QueryVector::Dense(vec![1.0_f32, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("secondary vector typed query");

    assert_eq!(
        hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        vec![2, 1]
    );
    assert!((hits[0].distance - 0.0).abs() < 1e-6);
    assert!((hits[1].distance - 0.006116271).abs() < 1e-6);
}

#[test]
fn zvec_parity_query_context_uses_query_by_id_field_name_metric_on_typed_bruteforce_path() {
    let root = unique_temp_dir("hannsdb_typed_query_query_by_id_field_metric");
    let mut db = HannsDb::open(&root).expect("open db");
    let mut schema = CollectionSchema::new(
        "dense",
        2,
        "l2",
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    schema.vectors.push(VectorFieldSchema::new("title", 2));
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");
    db.create_vector_index(
        "docs",
        VectorIndexDescriptor {
            field_name: "title".to_string(),
            kind: VectorIndexKind::Hnsw,
            metric: Some("cosine".to_string()),
            params: serde_json::json!({
                "m": 16,
                "ef_construction": 128,
            }),
        },
    )
    .expect("register descriptor-backed secondary vector index");
    db.insert_documents(
        "docs",
        &[
            Document::with_named_vectors(
                1,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![5.0_f32, 5.0],
                [("title".to_string(), vec![0.9_f32, 0.1])],
            ),
            Document::with_named_vectors(
                2,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![0.0_f32, 0.0],
                [("title".to_string(), vec![10.0_f32, 0.0])],
            ),
            Document::with_named_vectors(
                3,
                [("group".to_string(), FieldValue::Int64(2))],
                "dense",
                vec![1.0_f32, 1.0],
                [("title".to_string(), vec![1.0_f32, 0.0])],
            ),
        ],
    )
    .expect("insert documents");

    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 2,
                queries: vec![],
                query_by_id: Some(vec![3]),
                query_by_id_field_name: Some("title".to_string()),
                filter: Some("group == 1".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("query_by_id typed query");

    assert_eq!(
        hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        vec![2, 1]
    );
    assert!((hits[0].distance - 0.0).abs() < 1e-6);
    assert!((hits[1].distance - 0.006116271).abs() < 1e-6);
}

#[test]
fn zvec_parity_query_context_accepts_mixed_metric_typed_recall_sources() {
    let root = unique_temp_dir("hannsdb_typed_query_mixed_metric_recall");
    let mut db = HannsDb::open(&root).expect("open db");
    let mut schema = CollectionSchema::new(
        "dense",
        2,
        "l2",
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    schema.vectors.push(VectorFieldSchema::new("title", 2));
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");
    db.create_vector_index(
        "docs",
        VectorIndexDescriptor {
            field_name: "title".to_string(),
            kind: VectorIndexKind::Hnsw,
            metric: Some("cosine".to_string()),
            params: serde_json::json!({
                "m": 16,
                "ef_construction": 128,
            }),
        },
    )
    .expect("register descriptor-backed secondary vector index");

    let result = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 2,
                queries: vec![
                    VectorQuery {
                        field_name: "dense".to_string(),
                        vector: QueryVector::Dense(vec![1.0_f32, 0.0]),
                        param: None,
                    },
                    VectorQuery {
                        field_name: "title".to_string(),
                        vector: QueryVector::Dense(vec![1.0_f32, 0.0]),
                        param: None,
                    },
                ],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("mixed metric typed recall should now be accepted");

    assert!(result.is_empty(), "empty collection should return no hits");
}

#[test]
fn zvec_parity_query_context_merges_primary_and_secondary_vector_recall_sources() {
    let root = unique_temp_dir("hannsdb_typed_query_mixed_recall");
    let mut db = HannsDb::open(&root).expect("open db");
    let mut schema = CollectionSchema::new(
        "dense",
        2,
        "l2",
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    schema.vectors.push(VectorFieldSchema::new("title", 2));
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");
    db.insert_documents(
        "docs",
        &[
            Document::with_named_vectors(
                1,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![0.0_f32, 0.0],
                [("title".to_string(), vec![0.9_f32, 0.0])],
            ),
            Document::with_named_vectors(
                2,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![0.2_f32, 0.0],
                [("title".to_string(), vec![0.0_f32, 0.0])],
            ),
            Document::with_named_vectors(
                3,
                [("group".to_string(), FieldValue::Int64(2))],
                "dense",
                vec![1.0_f32, 0.0],
                [("title".to_string(), vec![0.1_f32, 0.0])],
            ),
        ],
    )
    .expect("insert documents");

    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 3,
                queries: vec![
                    VectorQuery {
                        field_name: "dense".to_string(),
                        vector: QueryVector::Dense(vec![0.0_f32, 0.0]),
                        param: None,
                    },
                    VectorQuery {
                        field_name: "title".to_string(),
                        vector: QueryVector::Dense(vec![0.0_f32, 0.0]),
                        param: None,
                    },
                ],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("mixed typed recall sources");

    assert_eq!(
        hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    assert_eq!(
        hits.iter().map(|hit| hit.distance).collect::<Vec<_>>(),
        vec![0.0, 0.0, 0.1]
    );
}

#[test]
fn zvec_parity_query_context_single_vector_ef_search_matches_legacy_search_path_with_secondary_schema(
) {
    let root = unique_temp_dir("hannsdb_typed_query_single_vector_ef_search_secondary_schema");
    let mut db = HannsDb::open(&root).expect("open db");
    let mut schema = CollectionSchema::new(
        "dense",
        2,
        "l2",
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    schema.vectors.push(VectorFieldSchema::new("title", 2));
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");
    db.insert_documents(
        "docs",
        &[
            Document::with_named_vectors(
                1,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![0.0_f32, 0.0],
                [("title".to_string(), vec![10.0_f32, 10.0])],
            ),
            Document::with_named_vectors(
                2,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![0.2_f32, 0.0],
                [("title".to_string(), vec![11.0_f32, 11.0])],
            ),
            Document::with_named_vectors(
                3,
                [("group".to_string(), FieldValue::Int64(2))],
                "dense",
                vec![0.1_f32, 0.0],
                [("title".to_string(), vec![12.0_f32, 12.0])],
            ),
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
                    field_name: "dense".to_string(),
                    vector: QueryVector::Dense(vec![0.0_f32, 0.0]),
                    param: Some(VectorQueryParam {
                        ef_search: Some(64),
                    }),
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("typed query should reuse the legacy single-vector path");

    assert_eq!(
        typed_hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        legacy_hits.iter().map(|hit| hit.id).collect::<Vec<_>>()
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
                    vector: QueryVector::Dense(vec![0.0_f32, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: Some(QueryGroupBy {
                    field_name: "group".to_string(),
                    group_topk: 0,
                    group_count: 0,
                }),
                reranker: None,
                order_by: None,
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
fn zvec_parity_query_context_group_by_keeps_top_ranked_missing_field_hit() {
    let root = unique_temp_dir("hannsdb_typed_query_group_by_missing_field");
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
            Document::new(1, [], vec![0.0_f32, 0.0]),
            Document::new(
                2,
                [("group".to_string(), FieldValue::Int64(7))],
                vec![0.05_f32, 0.0],
            ),
            Document::new(
                3,
                [("group".to_string(), FieldValue::Int64(7))],
                vec![0.10_f32, 0.0],
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
                    vector: QueryVector::Dense(vec![0.0_f32, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: Some(QueryGroupBy {
                    field_name: "group".to_string(),
                    group_topk: 0,
                    group_count: 0,
                }),
                reranker: None,
                order_by: None,
            },
        )
        .expect("group_by should retain the top-ranked hit with a missing group field");

    let hit_ids = hits.iter().map(|hit| hit.id).collect::<Vec<_>>();
    assert_eq!(hit_ids, vec![1, 2]);
    assert_eq!(
        hits[0].fields.get("group"),
        None,
        "missing grouped field should survive as its own group"
    );
    assert_eq!(hits[1].fields.get("group"), Some(&FieldValue::Int64(7)));
}

#[test]
fn zvec_parity_query_context_group_by_canonicalizes_float_groups() {
    let root = unique_temp_dir("hannsdb_typed_query_group_by_float_canonical");
    let mut db = HannsDb::open(&root).expect("open db");
    let schema = CollectionSchema::new(
        "vector",
        2,
        "l2",
        vec![ScalarFieldSchema::new("score", FieldType::Float64)],
    );
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");
    db.insert_documents(
        "docs",
        &[
            Document::new(
                10,
                [("score".to_string(), FieldValue::Float64(-0.0))],
                vec![0.0_f32, 0.0],
            ),
            Document::new(
                11,
                [("score".to_string(), FieldValue::Float64(0.0))],
                vec![0.05_f32, 0.0],
            ),
            Document::new(
                30,
                [("score".to_string(), FieldValue::Float64(1.0))],
                vec![0.10_f32, 0.0],
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
                    vector: QueryVector::Dense(vec![0.0_f32, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: Some(QueryGroupBy {
                    field_name: "score".to_string(),
                    group_topk: 0,
                    group_count: 0,
                }),
                reranker: None,
                order_by: None,
            },
        )
        .expect("group_by should canonicalize float group keys");

    let hit_ids = hits.iter().map(|hit| hit.id).collect::<Vec<_>>();
    assert_eq!(hit_ids, vec![10, 30]);
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
                    vector: QueryVector::Dense(vec![0.0_f32, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: Some(QueryGroupBy {
                    field_name: "missing".to_string(),
                    group_topk: 0,
                    group_count: 0,
                }),
                reranker: None,
                order_by: None,
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
                    vector: QueryVector::Dense(vec![0.0_f32, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: Some(QueryGroupBy {
                    field_name: "vector".to_string(),
                    group_topk: 0,
                    group_count: 0,
                }),
                reranker: None,
                order_by: None,
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
                query_by_id_field_name: None,
                filter: Some("group == 1".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: Some(QueryGroupBy {
                    field_name: "group".to_string(),
                    group_topk: 0,
                    group_count: 0,
                }),
                reranker: None,
                order_by: None,
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
    db.insert("docs", &[1, 2], &[0.0_f32, 0.0, 1.0, 1.0])
        .expect("insert vectors");

    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 3,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: QueryVector::Dense(vec![0.0_f32, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: Some(QueryReranker::Rrf { rank_constant: 60 }),
                order_by: None,
            },
        )
        .expect("reranker with single vector should succeed");

    // RRF with a single field just returns the same results as unranked search
    assert!(!hits.is_empty());
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
                    vector: QueryVector::Dense(vec![0.0_f32, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
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
fn zvec_parity_query_context_secondary_fast_path_shadowing_across_segments_respects_tombstones_and_missing_vectors(
) {
    let root = unique_temp_dir("hannsdb_typed_query_secondary_shadowing_across_segments");
    let mut db = HannsDb::open(&root).expect("open db");
    let mut schema = CollectionSchema::new(
        "dense",
        3,
        "l2",
        vec![ScalarFieldSchema::new("version", FieldType::String)],
    );
    schema.vectors.push(
        VectorFieldSchema::new("title", 2).with_index_param(VectorIndexSchema::hnsw(
            Some("l2"),
            16,
            128,
        )),
    );
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");
    db.insert_documents(
        "docs",
        &[
            Document::with_named_vectors(
                1,
                [(
                    "version".to_string(),
                    FieldValue::String("old-1".to_string()),
                )],
                "dense",
                vec![9.0_f32, 9.0, 9.0],
                [("title".to_string(), vec![0.0_f32, 0.0])],
            ),
            Document::with_named_vectors(
                2,
                [(
                    "version".to_string(),
                    FieldValue::String("old-2".to_string()),
                )],
                "dense",
                vec![9.0_f32, 9.0, 9.0],
                [("title".to_string(), vec![0.1_f32, 0.0])],
            ),
            Document::with_named_vectors(
                3,
                [(
                    "version".to_string(),
                    FieldValue::String("old-3".to_string()),
                )],
                "dense",
                vec![9.0_f32, 9.0, 9.0],
                [("title".to_string(), vec![0.05_f32, 0.0])],
            ),
            Document::with_named_vectors(
                4,
                [(
                    "version".to_string(),
                    FieldValue::String("old-4".to_string()),
                )],
                "dense",
                vec![9.0_f32, 9.0, 9.0],
                [("title".to_string(), vec![0.2_f32, 0.0])],
            ),
        ],
    )
    .expect("insert documents");

    rewrite_collection_to_two_segment_layout_with_secondary_vectors(
        &root,
        "docs",
        3,
        "dense",
        &[
            Document::with_named_vectors(
                1,
                [(
                    "version".to_string(),
                    FieldValue::String("new-1".to_string()),
                )],
                "dense",
                vec![1.0_f32, 1.0, 1.0],
                [("title".to_string(), vec![10.0_f32, 10.0])],
            ),
            Document::with_named_vectors(
                2,
                [(
                    "version".to_string(),
                    FieldValue::String("new-2".to_string()),
                )],
                "dense",
                vec![1.0_f32, 1.0, 1.0],
                [("title".to_string(), vec![0.0_f32, 0.0])],
            ),
            Document::with_primary_vector_name(
                3,
                [(
                    "version".to_string(),
                    FieldValue::String("new-3".to_string()),
                )],
                "dense",
                vec![1.0_f32, 1.0, 1.0],
            ),
            Document::with_named_vectors(
                5,
                [(
                    "version".to_string(),
                    FieldValue::String("new-5".to_string()),
                )],
                "dense",
                vec![1.0_f32, 1.0, 1.0],
                [("title".to_string(), vec![0.3_f32, 0.0])],
            ),
        ],
        &[1],
    );

    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 4,
                queries: vec![VectorQuery {
                    field_name: "title".to_string(),
                    vector: QueryVector::Dense(vec![0.0_f32, 0.0]),
                    param: Some(VectorQueryParam {
                        ef_search: Some(64),
                    }),
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("secondary typed fast path across segments");

    let hit_ids = hits.iter().map(|hit| hit.id).collect::<Vec<_>>();
    assert_eq!(hit_ids, vec![4, 5, 1]);
    assert_eq!(
        hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        vec![4, 5, 1]
    );
    assert!(hits.iter().all(|hit| hit.id != 2));
    assert!(hits.iter().all(|hit| hit.id != 3));
    let hit1 = hits.iter().find(|hit| hit.id == 1).expect("id 1 result");
    assert!(
        hit1.distance > 1.0,
        "newer live row should shadow the older zero-distance secondary vector"
    );
}

#[test]
fn zvec_parity_query_context_invalidates_secondary_fast_path_cache_after_secondary_insert() {
    let root = unique_temp_dir("hannsdb_typed_query_secondary_cache_invalidation");
    let mut db = HannsDb::open(&root).expect("open db");
    let mut schema = CollectionSchema::new(
        "dense",
        3,
        "l2",
        vec![ScalarFieldSchema::new("version", FieldType::String)],
    );
    schema.vectors.push(VectorFieldSchema::new("title", 2));
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");
    db.create_vector_index(
        "docs",
        VectorIndexDescriptor {
            field_name: "title".to_string(),
            kind: VectorIndexKind::Hnsw,
            metric: Some("l2".to_string()),
            params: serde_json::json!({
                "m": 16,
                "ef_construction": 128,
            }),
        },
    )
    .expect("register descriptor-backed secondary vector index");
    db.insert_documents(
        "docs",
        &[
            Document::with_named_vectors(
                1,
                [(
                    "version".to_string(),
                    FieldValue::String("first".to_string()),
                )],
                "dense",
                vec![9.0_f32, 9.0, 9.0],
                [("title".to_string(), vec![0.9_f32, 0.0])],
            ),
            Document::with_named_vectors(
                2,
                [(
                    "version".to_string(),
                    FieldValue::String("second".to_string()),
                )],
                "dense",
                vec![9.0_f32, 9.0, 9.0],
                [("title".to_string(), vec![0.1_f32, 0.0])],
            ),
        ],
    )
    .expect("insert documents");

    let warm_hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 2,
                queries: vec![VectorQuery {
                    field_name: "title".to_string(),
                    vector: QueryVector::Dense(vec![0.0_f32, 0.0]),
                    param: Some(VectorQueryParam {
                        ef_search: Some(64),
                    }),
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("warm secondary cache");
    assert_eq!(
        warm_hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        vec![2, 1]
    );

    db.insert_documents(
        "docs",
        &[Document::with_named_vectors(
            3,
            [(
                "version".to_string(),
                FieldValue::String("fresh".to_string()),
            )],
            "dense",
            vec![9.0_f32, 9.0, 9.0],
            [("title".to_string(), vec![0.0_f32, 0.0])],
        )],
    )
    .expect("insert closer document");

    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 3,
                queries: vec![VectorQuery {
                    field_name: "title".to_string(),
                    vector: QueryVector::Dense(vec![0.0_f32, 0.0]),
                    param: Some(VectorQueryParam {
                        ef_search: Some(64),
                    }),
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("second secondary fast-path query should reflect inserted document");

    assert_eq!(
        hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        vec![3, 2, 1]
    );
    assert_eq!(hits[0].distance, 0.0);
    assert_eq!(
        hits[0].fields.get("version"),
        Some(&FieldValue::String("fresh".to_string()))
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
                    vector: QueryVector::Dense(vec![0.0_f32, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
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
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
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
                    vector: QueryVector::Dense(vec![0.0_f32, 0.0]),
                    param: Some(VectorQueryParam {
                        ef_search: Some(64),
                    }),
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
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
                    vector: QueryVector::Dense(vec![0.0_f32, 0.0]),
                    param: Some(VectorQueryParam {
                        ef_search: Some(64),
                    }),
                }],
                query_by_id: Some(vec![7]),
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
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
                query_by_id_field_name: None,
                filter: Some("group == 1".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
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
fn zvec_parity_query_context_matches_manual_ground_truth_for_typed_filter_queries() {
    let root = unique_temp_dir("hannsdb_typed_query_ground_truth");
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
                11,
                [("group".to_string(), FieldValue::Int64(1))],
                vec![0.0_f32, 0.0],
            ),
            Document::new(
                12,
                [("group".to_string(), FieldValue::Int64(1))],
                vec![1.0_f32, 0.0],
            ),
            Document::new(
                13,
                [("group".to_string(), FieldValue::Int64(2))],
                vec![0.0_f32, 1.0],
            ),
            Document::new(
                14,
                [("group".to_string(), FieldValue::Int64(1))],
                vec![1.0_f32, 1.0],
            ),
            Document::new(
                15,
                [("group".to_string(), FieldValue::Int64(1))],
                vec![2.0_f32, 0.0],
            ),
            Document::new(
                16,
                [("group".to_string(), FieldValue::Int64(2))],
                vec![10.0_f32, 10.0],
            ),
        ],
    )
    .expect("insert documents");

    let query = vec![0.0_f32, 0.0];
    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 3,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: QueryVector::Dense(query.clone()),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: Some("group == 1".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("typed filtered query");

    let mut expected = vec![
        (11_i64, l2_distance(&query, &[0.0_f32, 0.0])),
        (12_i64, l2_distance(&query, &[1.0_f32, 0.0])),
        (14_i64, l2_distance(&query, &[1.0_f32, 1.0])),
    ];
    expected.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });

    assert_eq!(hits.len(), expected.len());
    assert_eq!(
        hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        expected.iter().map(|(id, _)| *id).collect::<Vec<_>>()
    );
    for (hit, (_, expected_distance)) in hits.iter().zip(expected.iter()) {
        assert!(
            (hit.distance - expected_distance).abs() < 1e-6,
            "distance mismatch for id {}: expected {}, got {}",
            hit.id,
            expected_distance,
            hit.distance
        );
        assert_eq!(
            hit.fields.get("group"),
            Some(&FieldValue::Int64(1)),
            "filtered typed query should only return group == 1 rows"
        );
    }
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let delta = x - y;
            delta * delta
        })
        .sum::<f32>()
        .sqrt()
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
                    vector: QueryVector::Dense(vec![0.0_f32, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: Some(vec!["color".to_string()]),
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
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
fn zvec_parity_query_context_includes_vectors_on_single_vector_fast_path() {
    let root = unique_temp_dir("hannsdb_typed_query_include_vector");
    let mut db = HannsDb::open(&root).expect("open db");
    let mut schema = CollectionSchema::new(
        "dense",
        2,
        "l2",
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    schema.vectors.push(VectorFieldSchema::new("sparse", 2));
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");
    db.insert_documents(
        "docs",
        &[
            Document::with_named_vectors(
                7,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![0.0_f32, 0.0],
                [("sparse".to_string(), vec![7.0_f32, 7.0])],
            ),
            Document::with_named_vectors(
                8,
                [("group".to_string(), FieldValue::Int64(2))],
                "dense",
                vec![0.1_f32, 0.0],
                [("sparse".to_string(), vec![8.0_f32, 8.0])],
            ),
        ],
    )
    .expect("insert documents");

    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 1,
                queries: vec![VectorQuery {
                    field_name: "dense".to_string(),
                    vector: QueryVector::Dense(vec![0.0_f32, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: Some(vec!["group".to_string()]),
                include_vector: true,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("typed query should include vectors on the fast path");

    assert_eq!(hits.len(), 1);
    assert_eq!(
        hits[0].fields,
        [("group".to_string(), FieldValue::Int64(1))]
            .into_iter()
            .collect()
    );
    assert_eq!(hits[0].vectors.get("dense"), Some(&vec![0.0_f32, 0.0]));
    assert_eq!(hits[0].vectors.get("sparse"), Some(&vec![7.0_f32, 7.0]));
}

#[test]
fn zvec_parity_query_context_includes_vectors_with_query_by_id_recall_source() {
    let root = unique_temp_dir("hannsdb_typed_query_include_vector_query_by_id");
    let mut db = HannsDb::open(&root).expect("open db");
    let mut schema = CollectionSchema::new(
        "dense",
        2,
        "l2",
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    schema.vectors.push(VectorFieldSchema::new("sparse", 2));
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");
    db.insert_documents(
        "docs",
        &[
            Document::with_named_vectors(
                7,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![0.0_f32, 0.0],
                [("sparse".to_string(), vec![7.0_f32, 7.0])],
            ),
            Document::with_named_vectors(
                8,
                [("group".to_string(), FieldValue::Int64(2))],
                "dense",
                vec![0.1_f32, 0.0],
                [("sparse".to_string(), vec![8.0_f32, 8.0])],
            ),
        ],
    )
    .expect("insert documents");

    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 1,
                queries: Vec::new(),
                query_by_id: Some(vec![7]),
                query_by_id_field_name: None,
                filter: None,
                output_fields: Some(vec!["group".to_string()]),
                include_vector: true,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("typed query should include vectors when recall comes from query_by_id");

    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, 7);
    assert_eq!(
        hits[0].fields,
        [("group".to_string(), FieldValue::Int64(1))]
            .into_iter()
            .collect()
    );
    assert_eq!(hits[0].vectors.get("dense"), Some(&vec![0.0_f32, 0.0]));
    assert_eq!(hits[0].vectors.get("sparse"), Some(&vec![7.0_f32, 7.0]));
}

#[test]
fn zvec_parity_query_context_defaults_query_by_id_to_primary_vector_when_field_name_is_unset() {
    let root = unique_temp_dir("hannsdb_typed_query_query_by_id_primary_default");
    let mut db = HannsDb::open(&root).expect("open db");
    let mut schema = CollectionSchema::new(
        "dense",
        2,
        "l2",
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    schema.vectors.push(VectorFieldSchema::new("sparse", 2));
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");
    db.insert_documents(
        "docs",
        &[
            Document::with_named_vectors(
                7,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![5.0_f32, 5.0],
                [("sparse".to_string(), vec![0.0_f32, 0.0])],
            ),
            Document::with_named_vectors(
                8,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![0.0_f32, 0.0],
                [("sparse".to_string(), vec![0.2_f32, 0.0])],
            ),
            Document::with_named_vectors(
                9,
                [("group".to_string(), FieldValue::Int64(2))],
                "dense",
                vec![1.0_f32, 1.0],
                [("sparse".to_string(), vec![1.0_f32, 0.0])],
            ),
        ],
    )
    .expect("insert documents");

    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 3,
                queries: Vec::new(),
                query_by_id: Some(vec![7]),
                query_by_id_field_name: None,
                filter: None,
                output_fields: Some(vec!["group".to_string()]),
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("query should default query_by_id to primary vector recall");

    assert_eq!(
        hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        vec![7, 9, 8]
    );
    assert_eq!(hits[0].distance, 0.0);
    assert_eq!(hits[0].fields.get("group"), Some(&FieldValue::Int64(1)));
}

#[test]
fn zvec_parity_query_context_uses_secondary_vector_for_query_by_id_when_configured() {
    let root = unique_temp_dir("hannsdb_typed_query_query_by_id_secondary_recall");
    let mut db = HannsDb::open(&root).expect("open db");
    let mut schema = CollectionSchema::new(
        "dense",
        2,
        "l2",
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    schema.vectors.push(VectorFieldSchema::new("sparse", 2));
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");
    db.insert_documents(
        "docs",
        &[
            Document::with_named_vectors(
                7,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![9.0_f32, 9.0],
                [("sparse".to_string(), vec![0.0_f32, 0.0])],
            ),
            Document::with_named_vectors(
                8,
                [("group".to_string(), FieldValue::Int64(2))],
                "dense",
                vec![0.1_f32, 0.0],
                [("sparse".to_string(), vec![0.2_f32, 0.0])],
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
                    field_name: "dense".to_string(),
                    vector: QueryVector::Dense(vec![0.0_f32, 0.0]),
                    param: None,
                }],
                query_by_id: Some(vec![7]),
                query_by_id_field_name: Some("sparse".to_string()),
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("query should recall from secondary vector");

    assert_eq!(
        hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        vec![7, 8]
    );
    assert_eq!(hits[0].distance, 0.0);
    assert!(hits[1].distance > hits[0].distance);
}

#[test]
fn zvec_parity_query_context_rejects_invalid_query_by_id_field_name() {
    let root = unique_temp_dir("hannsdb_typed_query_query_by_id_invalid_field");
    let mut db = HannsDb::open(&root).expect("open db");
    let schema = CollectionSchema::new("dense", 2, "l2", vec![]);
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");

    let err = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 1,
                queries: vec![VectorQuery {
                    field_name: "dense".to_string(),
                    vector: QueryVector::Dense(vec![0.0_f32, 0.0]),
                    param: None,
                }],
                query_by_id: Some(vec![1]),
                query_by_id_field_name: Some("missing".to_string()),
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect_err("invalid query_by_id field should error");

    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    assert!(err.to_string().contains("missing"));
}

#[test]
fn zvec_parity_query_context_errors_when_secondary_query_by_id_vector_is_missing() {
    let root = unique_temp_dir("hannsdb_typed_query_query_by_id_missing_secondary");
    let mut db = HannsDb::open(&root).expect("open db");
    let mut schema = CollectionSchema::new(
        "dense",
        2,
        "l2",
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    schema.vectors.push(VectorFieldSchema::new("sparse", 2));
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");
    db.insert_documents(
        "docs",
        &[Document::with_primary_vector_name(
            7,
            [("group".to_string(), FieldValue::Int64(1))],
            "dense",
            vec![0.0_f32, 0.0],
        )],
    )
    .expect("insert document");

    let err = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 1,
                queries: vec![VectorQuery {
                    field_name: "dense".to_string(),
                    vector: QueryVector::Dense(vec![0.0_f32, 0.0]),
                    param: None,
                }],
                query_by_id: Some(vec![7]),
                query_by_id_field_name: Some("sparse".to_string()),
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect_err("missing secondary vector should error");

    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("sparse"));
}

#[test]
fn zvec_parity_query_context_ignores_query_by_id_field_name_when_query_by_id_is_absent() {
    let root = unique_temp_dir("hannsdb_typed_query_ignore_query_by_id_field_name");
    let mut db = HannsDb::open(&root).expect("open db");
    let schema = CollectionSchema::new(
        "dense",
        2,
        "l2",
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");
    db.insert_documents(
        "docs",
        &[
            Document::with_primary_vector_name(
                1,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![0.0_f32, 0.0],
            ),
            Document::with_primary_vector_name(
                2,
                [("group".to_string(), FieldValue::Int64(2))],
                "dense",
                vec![0.2_f32, 0.0],
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
                    field_name: "dense".to_string(),
                    vector: QueryVector::Dense(vec![0.0_f32, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: Some("missing".to_string()),
                filter: None,
                output_fields: Some(vec!["group".to_string()]),
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("query_by_id_field_name should be ignored when query_by_id is absent");

    assert_eq!(
        hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        vec![1, 2]
    );
    assert_eq!(
        hits.iter()
            .map(|hit| hit.fields.get("group"))
            .collect::<Vec<_>>(),
        vec![Some(&FieldValue::Int64(1)), Some(&FieldValue::Int64(2))]
    );
}

#[test]
fn zvec_parity_query_context_include_vector_errors_when_fetched_hit_disappears_from_disk() {
    let root = unique_temp_dir("hannsdb_typed_query_include_vector_fetch_mismatch");
    let mut db = HannsDb::open(&root).expect("open db");
    let schema = CollectionSchema::new(
        "dense",
        2,
        "l2",
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");
    db.insert_documents(
        "docs",
        &[Document::with_primary_vector_name(
            7,
            [("group".to_string(), FieldValue::Int64(1))],
            "dense",
            vec![0.0_f32, 0.0],
        )],
    )
    .expect("insert document");

    let warm_hits = db
        .search("docs", &[0.0_f32, 0.0], 1)
        .expect("warm search cache");
    assert_eq!(warm_hits.len(), 1);
    assert_eq!(warm_hits[0].id, 7);

    let tombstones_path = root
        .join("collections")
        .join("docs")
        .join("tombstones.json");
    let mut tombstones =
        TombstoneMask::load_from_path(&tombstones_path).expect("load tombstones from disk");
    assert!(tombstones.mark_deleted(0));
    tombstones
        .save_to_path(&tombstones_path)
        .expect("persist tombstones");

    let segment_meta_path = root.join("collections").join("docs").join("segment.json");
    let mut segment_meta =
        SegmentMetadata::load_from_path(&segment_meta_path).expect("load segment metadata");
    segment_meta.deleted_count = tombstones.deleted_count();
    segment_meta
        .save_to_path(&segment_meta_path)
        .expect("persist segment metadata");

    let err = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 1,
                queries: vec![VectorQuery {
                    field_name: "dense".to_string(),
                    vector: QueryVector::Dense(vec![0.0_f32, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: true,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect_err("include_vector should error when fetch cannot materialize a cached hit");

    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    assert!(err.to_string().contains("failed to resolve"));
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
                query_by_id_field_name: None,
                filter: Some("group == 1".to_string()),
                output_fields: Some(vec!["color".to_string()]),
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
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
