use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use hannsdb_core::db::HannsDb;
use hannsdb_core::document::{
    CollectionSchema, Document, FieldType, FieldValue, ScalarFieldSchema,
};
use hannsdb_core::query::{QueryContext, QueryGroupBy, QueryReranker, VectorQuery};

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
fn zvec_parity_query_context_rejects_group_by_until_supported() {
    let root = unique_temp_dir("hannsdb_typed_query_group_by");
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
                group_by: Some(QueryGroupBy {
                    field_name: "group".to_string(),
                }),
                reranker: None,
            },
        )
        .expect_err("group_by should be rejected in the first slice");

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
