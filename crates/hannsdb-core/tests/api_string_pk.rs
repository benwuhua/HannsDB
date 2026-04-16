use std::time::{SystemTime, UNIX_EPOCH};

use hannsdb_core::db::HannsDb;
use hannsdb_core::document::{
    CollectionSchema, Document, FieldType, FieldValue, ScalarFieldSchema,
};
use hannsdb_core::query::QueryContext;

fn unique_temp_dir(name: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("{}_{}", name, nanos))
}

#[test]
fn api_string_pk_reopens_and_query_by_id_resolves_alphanumeric_key() {
    let root = unique_temp_dir("hannsdb_string_pk_query_by_id");
    let mut db = HannsDb::open(&root).expect("open db");
    let schema = CollectionSchema::new(
        "dense",
        2,
        "l2",
        vec![ScalarFieldSchema::new("rank", FieldType::Int64)],
    );
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");

    db.insert_documents_with_primary_keys(
        "docs",
        &[
            (
                "user-a".to_string(),
                Document::with_primary_vector_name(
                    101,
                    [("rank".to_string(), FieldValue::Int64(2))],
                    "dense",
                    vec![0.0, 0.0],
                ),
            ),
            (
                "user-b".to_string(),
                Document::with_primary_vector_name(
                    102,
                    [("rank".to_string(), FieldValue::Int64(1))],
                    "dense",
                    vec![0.1, 0.0],
                ),
            ),
            (
                "user-c".to_string(),
                Document::with_primary_vector_name(
                    103,
                    [("rank".to_string(), FieldValue::Int64(3))],
                    "dense",
                    vec![2.0, 0.0],
                ),
            ),
        ],
    )
    .expect("insert string-pk documents");

    drop(db);

    let db = HannsDb::open(&root).expect("reopen db");
    let query_ids = db
        .resolve_query_ids_by_primary_keys("docs", &["user-b".to_string()])
        .expect("resolve string primary key");
    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 3,
                queries: Vec::new(),
                query_by_id: Some(query_ids),
                query_by_id_field_name: None,
                filter: None,
                output_fields: Some(vec!["rank".to_string()]),
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("query_by_id using resolved string key");

    assert_eq!(hits.len(), 3);
    assert_eq!(hits[0].id, 102);
    assert_eq!(hits[0].fields.get("rank"), Some(&FieldValue::Int64(1)));
}
