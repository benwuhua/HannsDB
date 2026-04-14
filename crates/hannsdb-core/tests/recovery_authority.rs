use std::collections::BTreeMap;
use std::fs;

use hannsdb_core::db::HannsDb;
use hannsdb_core::document::{
    CollectionSchema, Document, FieldType, FieldValue, ScalarFieldSchema,
};
use hannsdb_core::segment::write_payloads_arrow;

fn sample_schema() -> CollectionSchema {
    CollectionSchema::new(
        "dense",
        2,
        "cosine",
        vec![ScalarFieldSchema::new("session_id", FieldType::String)],
    )
}

#[test]
fn recovery_authority_reopen_prefers_newer_compatibility_payloads_over_stale_forward_store() {
    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path();
    let mut db = HannsDb::open(root).expect("open db");
    let schema = sample_schema();

    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");
    db.insert_documents(
        "docs",
        &[Document::with_primary_vector_name(
            11,
            BTreeMap::new(),
            "dense",
            vec![0.1, 0.2],
        )],
    )
    .expect("insert document");
    db.flush_collection("docs")
        .expect("flush should materialize forward_store artifacts");
    drop(db);

    let collection_dir = root.join("collections").join("docs");
    let payloads_jsonl = collection_dir.join("payloads.jsonl");
    let payloads_arrow = collection_dir.join("payloads.arrow");
    assert!(
        collection_dir.join("forward_store.json").exists(),
        "flush should persist forward_store descriptor before reopen"
    );

    write_payloads_arrow(
        &payloads_arrow,
        &[BTreeMap::from([(
            "session_id".to_string(),
            FieldValue::String("fresh-session".to_string()),
        )])],
        &schema.fields,
    )
    .expect("rewrite compatibility payloads.arrow with newer field data");
    let _ = fs::remove_file(&payloads_jsonl);

    let reopened = HannsDb::open(root).expect("reopen db");
    let fetched = reopened
        .fetch_documents("docs", &[11])
        .expect("fetch doc after reopen");

    assert_eq!(fetched.len(), 1);
    assert_eq!(
        fetched[0].fields.get("session_id"),
        Some(&FieldValue::String("fresh-session".to_string())),
        "reopen should prefer newer compatibility payloads over stale forward_store field data"
    );
}
