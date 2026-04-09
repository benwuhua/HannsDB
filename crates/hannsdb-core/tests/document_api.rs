use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use hannsdb_core::catalog::CollectionMetadata;
use hannsdb_core::db::HannsDb;
use hannsdb_core::document::{
    CollectionSchema, Document, FieldType, FieldValue, ScalarFieldSchema,
};
use hannsdb_core::segment::{append_payloads, load_payloads};

fn unique_temp_file(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("{}_{}.json", name, nanos))
}

fn unique_temp_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("{}_{}", name, nanos))
}

#[test]
fn document_api_document_model_preserves_typed_scalar_fields() {
    let doc = Document::new(
        7,
        [
            ("active".to_string(), FieldValue::Bool(true)),
            ("name".to_string(), FieldValue::String("memo".to_string())),
            ("rank".to_string(), FieldValue::Int64(3)),
            ("score".to_string(), FieldValue::Float64(0.875)),
        ],
        vec![0.25_f32, 0.75],
    );

    assert_eq!(doc.id, 7);
    assert_eq!(doc.primary_vector(), &[0.25_f32, 0.75]);
    assert_eq!(doc.fields.get("active"), Some(&FieldValue::Bool(true)));
    assert_eq!(
        doc.fields.get("name"),
        Some(&FieldValue::String("memo".to_string()))
    );
    assert_eq!(doc.fields.get("rank"), Some(&FieldValue::Int64(3)));
    assert_eq!(doc.fields.get("score"), Some(&FieldValue::Float64(0.875)));
}

#[test]
fn document_api_collection_metadata_roundtrip_keeps_schema_shape() {
    let path = unique_temp_file("hannsdb_document_schema_meta");
    let schema = CollectionSchema::new(
        "embedding",
        768,
        "cosine",
        vec![
            ScalarFieldSchema::new("active", FieldType::Bool),
            ScalarFieldSchema::new("session_id", FieldType::String),
            ScalarFieldSchema::new("turn", FieldType::Int64),
        ],
    );
    let metadata = CollectionMetadata::new_with_schema("docs", schema.clone());

    metadata
        .save_to_path(&path)
        .expect("save collection metadata");
    let loaded = CollectionMetadata::load_from_path(&path).expect("load collection metadata");

    assert_eq!(loaded.name, "docs");
    assert_eq!(loaded.dimension, 768);
    assert_eq!(loaded.metric, "cosine");
    assert_eq!(loaded.primary_vector, "embedding");
    assert_eq!(loaded.fields, schema.fields);
    assert_eq!(loaded.schema(), schema);
}

#[test]
fn document_api_payload_storage_roundtrip_preserves_row_order() {
    let path = unique_temp_file("hannsdb_document_payloads");
    let payloads = vec![
        [
            ("active".to_string(), FieldValue::Bool(true)),
            ("turn".to_string(), FieldValue::Int64(1)),
        ]
        .into_iter()
        .collect(),
        [
            (
                "session_id".to_string(),
                FieldValue::String("sess-1".to_string()),
            ),
            ("score".to_string(), FieldValue::Float64(0.5)),
        ]
        .into_iter()
        .collect(),
    ];

    let appended = append_payloads(&path, &payloads).expect("append payloads");
    assert_eq!(appended, 2);

    let loaded = load_payloads(&path).expect("load payloads");
    assert_eq!(loaded, payloads);
}

#[test]
fn document_api_insert_and_fetch_roundtrip_preserves_fields_and_vector() {
    let root = unique_temp_dir("hannsdb_document_insert_fetch");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");

    let inserted = db
        .insert_documents(
            "docs",
            &[
                Document::new(
                    7,
                    [
                        (
                            "session_id".to_string(),
                            FieldValue::String("sess-a".to_string()),
                        ),
                        ("turn".to_string(), FieldValue::Int64(1)),
                    ],
                    vec![0.0_f32, 0.0],
                ),
                Document::new(
                    9,
                    [
                        ("active".to_string(), FieldValue::Bool(true)),
                        ("score".to_string(), FieldValue::Float64(0.5)),
                    ],
                    vec![1.0_f32, 1.0],
                ),
            ],
        )
        .expect("insert documents");
    assert_eq!(inserted, 2);

    let fetched = db
        .fetch_documents("docs", &[9, 7])
        .expect("fetch documents");
    assert_eq!(fetched.len(), 2);
    assert_eq!(fetched[0].id, 9);
    assert_eq!(fetched[0].primary_vector(), &[1.0_f32, 1.0]);
    assert_eq!(
        fetched[0].fields.get("active"),
        Some(&FieldValue::Bool(true))
    );
    assert_eq!(fetched[1].id, 7);
    assert_eq!(
        fetched[1].fields.get("session_id"),
        Some(&FieldValue::String("sess-a".to_string()))
    );
    assert_eq!(fetched[1].fields.get("turn"), Some(&FieldValue::Int64(1)));
}

#[test]
fn document_api_insert_documents_rejects_existing_live_id() {
    let root = unique_temp_dir("hannsdb_document_insert_duplicate");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");

    db.insert_documents(
        "docs",
        &[Document::new(
            7,
            [("turn".to_string(), FieldValue::Int64(1))],
            vec![0.0_f32, 0.0],
        )],
    )
    .expect("insert first document");

    let err = db
        .insert_documents(
            "docs",
            &[Document::new(
                7,
                [("turn".to_string(), FieldValue::Int64(2))],
                vec![1.0_f32, 1.0],
            )],
        )
        .expect_err("duplicate live id must be rejected");
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
}

#[test]
fn document_api_upsert_replaces_existing_row_for_fetch_and_search() {
    let root = unique_temp_dir("hannsdb_document_upsert");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");

    db.insert_documents(
        "docs",
        &[Document::new(
            7,
            [(
                "session_id".to_string(),
                FieldValue::String("old".to_string()),
            )],
            vec![100.0_f32, 100.0],
        )],
    )
    .expect("insert old document");

    let upserted = db
        .upsert_documents(
            "docs",
            &[Document::new(
                7,
                [(
                    "session_id".to_string(),
                    FieldValue::String("new".to_string()),
                )],
                vec![0.0_f32, 0.0],
            )],
        )
        .expect("upsert document");
    assert_eq!(upserted, 1);

    let fetched = db
        .fetch_documents("docs", &[7])
        .expect("fetch documents after upsert");
    assert_eq!(fetched.len(), 1);
    assert_eq!(
        fetched[0].fields.get("session_id"),
        Some(&FieldValue::String("new".to_string()))
    );
    assert_eq!(fetched[0].primary_vector(), &[0.0_f32, 0.0]);

    let hits = db.search("docs", &[0.0_f32, 0.0], 1).expect("search docs");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, 7);
}

#[test]
fn document_api_delete_hides_rows_from_fetch_and_search() {
    let root = unique_temp_dir("hannsdb_document_delete");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");

    db.insert_documents(
        "docs",
        &[
            Document::new(
                7,
                [("turn".to_string(), FieldValue::Int64(1))],
                vec![0.0_f32, 0.0],
            ),
            Document::new(
                9,
                [("turn".to_string(), FieldValue::Int64(2))],
                vec![1.0_f32, 1.0],
            ),
        ],
    )
    .expect("insert documents");

    let deleted = db.delete("docs", &[7]).expect("delete one document");
    assert_eq!(deleted, 1);

    let fetched = db
        .fetch_documents("docs", &[7, 9])
        .expect("fetch documents after delete");
    assert_eq!(fetched.len(), 1);
    assert_eq!(fetched[0].id, 9);

    let hits = db.search("docs", &[0.0_f32, 0.0], 2).expect("search docs");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, 9);
}

#[test]
fn document_api_legacy_vector_insert_keeps_payload_alignment_for_new_documents() {
    let root = unique_temp_dir("hannsdb_document_legacy_payload_alignment");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");

    db.insert("docs", &[7], &[0.0_f32, 0.0])
        .expect("insert legacy vector row");
    db.insert_documents(
        "docs",
        &[Document::new(
            9,
            [(
                "session_id".to_string(),
                FieldValue::String("sess-b".to_string()),
            )],
            vec![1.0_f32, 1.0],
        )],
    )
    .expect("insert document row");

    let fetched = db
        .fetch_documents("docs", &[7, 9])
        .expect("fetch mixed rows");
    assert_eq!(fetched.len(), 2);
    assert!(fetched[0].fields.is_empty());
    assert_eq!(
        fetched[1].fields.get("session_id"),
        Some(&FieldValue::String("sess-b".to_string()))
    );
}

#[test]
fn document_api_query_documents_applies_filter_before_topk() {
    let root = unique_temp_dir("hannsdb_document_query_filter");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");

    db.insert_documents(
        "docs",
        &[
            Document::new(
                7,
                [
                    (
                        "session_id".to_string(),
                        FieldValue::String("s1".to_string()),
                    ),
                    ("turn".to_string(), FieldValue::Int64(1)),
                ],
                vec![0.0_f32, 0.0],
            ),
            Document::new(
                9,
                [
                    (
                        "session_id".to_string(),
                        FieldValue::String("s1".to_string()),
                    ),
                    ("turn".to_string(), FieldValue::Int64(2)),
                ],
                vec![0.2_f32, 0.2],
            ),
            Document::new(
                11,
                [
                    (
                        "session_id".to_string(),
                        FieldValue::String("s2".to_string()),
                    ),
                    ("turn".to_string(), FieldValue::Int64(5)),
                ],
                vec![0.05_f32, 0.05],
            ),
        ],
    )
    .expect("insert documents");

    let hits = db
        .query_documents(
            "docs",
            &[0.0_f32, 0.0],
            1,
            Some("session_id == \"s1\" and turn >= 2"),
        )
        .expect("query filtered documents");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, 9);
    assert_eq!(
        hits[0].fields.get("session_id"),
        Some(&FieldValue::String("s1".to_string()))
    );
    assert_eq!(hits[0].fields.get("turn"), Some(&FieldValue::Int64(2)));
}

#[test]
fn document_api_query_documents_rejects_invalid_filter_syntax() {
    let root = unique_temp_dir("hannsdb_document_query_invalid_filter");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection("docs", 2, "l2")
        .expect("create collection");
    db.insert_documents(
        "docs",
        &[Document::new(
            7,
            [("turn".to_string(), FieldValue::Int64(1))],
            vec![0.0_f32, 0.0],
        )],
    )
    .expect("insert document");

    let err = db
        .query_documents("docs", &[0.0_f32, 0.0], 1, Some("turn ~= 2"))
        .expect_err("invalid filter syntax must fail");
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
}
