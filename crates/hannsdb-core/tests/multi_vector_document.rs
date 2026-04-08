use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use hannsdb_core::db::HannsDb;
use hannsdb_core::document::{
    CollectionSchema, Document, FieldType, FieldValue, ScalarFieldSchema, VectorFieldSchema,
};

fn unique_temp_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("{}_{}", name, nanos))
}

fn sample_schema() -> CollectionSchema {
    CollectionSchema {
        primary_vector: "dense".to_string(),
        fields: vec![ScalarFieldSchema::new("session_id", FieldType::String)],
        vectors: vec![
            VectorFieldSchema::new("dense", 2),
            VectorFieldSchema::new("title", 2),
        ],
    }
}

fn sample_document(id: i64, dense: [f32; 2], title: [f32; 2], session_id: &str) -> Document {
    Document::with_vectors(
        id,
        BTreeMap::from([(
            "session_id".to_string(),
            FieldValue::String(session_id.to_string()),
        )]),
        dense.to_vec(),
        [("title".to_string(), title.to_vec())],
    )
}

#[test]
fn multi_vector_document_insert_upsert_fetch_and_reopen_round_trip() {
    let root = unique_temp_dir("hannsdb_multi_vector_document");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection_with_schema("docs", &sample_schema())
        .expect("create collection");

    let inserted = db
        .insert_documents(
            "docs",
            &[sample_document(7, [0.1, 0.2], [0.3, 0.4], "inserted")],
        )
        .expect("insert document");
    assert_eq!(inserted, 1);

    let fetched = db.fetch_documents("docs", &[7]).expect("fetch inserted doc");
    assert_eq!(fetched.len(), 1);
    assert_eq!(fetched[0].vector, vec![0.1, 0.2]);
    assert_eq!(
        fetched[0].vectors.get("title"),
        Some(&vec![0.3, 0.4])
    );
    assert_eq!(
        fetched[0].fields.get("session_id"),
        Some(&FieldValue::String("inserted".to_string()))
    );

    let upserted = db
        .upsert_documents(
            "docs",
            &[sample_document(7, [1.1, 1.2], [1.3, 1.4], "upserted")],
        )
        .expect("upsert document");
    assert_eq!(upserted, 1);

    let fetched = db.fetch_documents("docs", &[7]).expect("fetch upserted doc");
    assert_eq!(fetched.len(), 1);
    assert_eq!(fetched[0].vector, vec![1.1, 1.2]);
    assert_eq!(
        fetched[0].vectors.get("title"),
        Some(&vec![1.3, 1.4])
    );
    assert_eq!(
        fetched[0].fields.get("session_id"),
        Some(&FieldValue::String("upserted".to_string()))
    );
    drop(db);

    let collection_dir = root.join("collections").join("docs");
    let vectors_path = collection_dir.join("vectors.jsonl");
    if vectors_path.exists() {
        fs::remove_file(&vectors_path).expect("remove vectors sidecar");
    }

    let reopened = HannsDb::open(&root).expect("reopen db");
    let replayed = reopened
        .fetch_documents("docs", &[7])
        .expect("fetch replayed doc");
    assert_eq!(replayed.len(), 1);
    assert_eq!(replayed[0].vector, vec![1.1, 1.2]);
    assert_eq!(
        replayed[0].vectors.get("title"),
        Some(&vec![1.3, 1.4])
    );
    assert_eq!(
        replayed[0].fields.get("session_id"),
        Some(&FieldValue::String("upserted".to_string()))
    );
}
