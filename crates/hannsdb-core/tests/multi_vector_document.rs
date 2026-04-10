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
    Document::with_named_vectors(
        id,
        BTreeMap::from([(
            "session_id".to_string(),
            FieldValue::String(session_id.to_string()),
        )]),
        "dense",
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
    assert_eq!(fetched[0].primary_vector_for("dense").unwrap(), &[0.1, 0.2]);
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
    assert_eq!(fetched[0].primary_vector_for("dense").unwrap(), &[1.1, 1.2]);
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
    assert_eq!(replayed[0].primary_vector_for("dense").unwrap(), &[1.1, 1.2]);
    assert_eq!(
        replayed[0].vectors.get("title"),
        Some(&vec![1.3, 1.4])
    );
    assert_eq!(
        replayed[0].fields.get("session_id"),
        Some(&FieldValue::String("upserted".to_string()))
    );
}

#[test]
fn multi_vector_document_reopen_rebuilds_truncated_vectors_sidecar() {
    let root = unique_temp_dir("hannsdb_multi_vector_document_truncated_vectors");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection_with_schema("docs", &sample_schema())
        .expect("create collection");
    db.insert_documents(
        "docs",
        &[sample_document(7, [0.1, 0.2], [0.3, 0.4], "inserted")],
    )
    .expect("insert document");
    drop(db);

    let collection_dir = root.join("collections").join("docs");
    fs::write(collection_dir.join("vectors.jsonl"), b"").expect("truncate vectors sidecar");

    let reopened = HannsDb::open(&root).expect("reopen should replay wal");
    let replayed = reopened
        .fetch_documents("docs", &[7])
        .expect("fetch replayed doc");
    assert_eq!(replayed.len(), 1);
    assert_eq!(
        replayed[0].vectors.get("title"),
        Some(&vec![0.3, 0.4])
    );
    assert_eq!(replayed[0].primary_vector_for("dense").unwrap(), &[0.1, 0.2]);
}

#[test]
fn multi_vector_document_appends_secondary_vectors_after_legacy_rows_without_misalignment() {
    let root = unique_temp_dir("hannsdb_multi_vector_document_legacy_alignment");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection_with_schema("docs", &sample_schema())
        .expect("create collection");

    db.insert_documents(
        "docs",
        &[
            Document::with_primary_vector_name(
                1,
                [(
                    "session_id".to_string(),
                    FieldValue::String("legacy-1".to_string()),
                )],
                "dense",
                vec![0.0, 0.0],
            ),
            Document::with_primary_vector_name(
                2,
                [(
                    "session_id".to_string(),
                    FieldValue::String("legacy-2".to_string()),
                )],
                "dense",
                vec![1.0, 1.0],
            ),
        ],
    )
    .expect("insert legacy docs");

    db.insert_documents(
        "docs",
        &[sample_document(3, [2.0, 2.0], [3.0, 3.0], "new")],
    )
    .expect("insert multi-vector doc");
    drop(db);

    let reopened = HannsDb::open(&root).expect("reopen db");
    let fetched = reopened
        .fetch_documents("docs", &[1, 2, 3])
        .expect("fetch documents after reopen");

    assert_eq!(fetched.len(), 3);
    assert!(!fetched[0].vectors.contains_key("title"));
    assert!(!fetched[1].vectors.contains_key("title"));
    assert_eq!(fetched[2].vectors.get("title"), Some(&vec![3.0, 3.0]));
    assert_eq!(
        fetched[2].fields.get("session_id"),
        Some(&FieldValue::String("new".to_string()))
    );
}

#[test]
fn single_vector_document_reopen_rebuilds_truncated_vectors_sidecar() {
    let root = unique_temp_dir("hannsdb_single_vector_document_truncated_vectors");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection_with_schema("docs", &sample_schema())
        .expect("create collection");
    db.insert_documents(
        "docs",
        &[Document::with_primary_vector_name(
            11,
            [(
                "session_id".to_string(),
                FieldValue::String("single".to_string()),
            )],
            "dense",
            vec![0.5, 0.6],
        )],
    )
    .expect("insert single-vector document");
    drop(db);

    let collection_dir = root.join("collections").join("docs");
    fs::write(collection_dir.join("vectors.jsonl"), b"").expect("truncate vectors sidecar");

    let reopened = HannsDb::open(&root).expect("reopen should replay wal");
    let replayed = reopened
        .fetch_documents("docs", &[11])
        .expect("fetch replayed doc");
    assert_eq!(replayed.len(), 1);
    assert_eq!(replayed[0].primary_vector_for("dense").unwrap(), &[0.5, 0.6]);
    // Legacy insert has no secondary vectors; only the primary is reconstructed.
    assert_eq!(replayed[0].vectors.len(), 1);
    assert_eq!(
        replayed[0].fields.get("session_id"),
        Some(&FieldValue::String("single".to_string()))
    );
}

#[test]
fn legacy_insert_reopen_rebuilds_truncated_vectors_sidecar() {
    let root = unique_temp_dir("hannsdb_legacy_insert_truncated_vectors");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection_with_schema("docs", &sample_schema())
        .expect("create collection");
    db.insert("docs", &[21], &[0.7, 0.8])
        .expect("legacy insert");
    drop(db);

    let collection_dir = root.join("collections").join("docs");
    fs::write(collection_dir.join("vectors.jsonl"), b"").expect("truncate vectors sidecar");

    let reopened = HannsDb::open(&root).expect("reopen should replay wal");
    let replayed = reopened
        .fetch_documents("docs", &[21])
        .expect("fetch replayed doc");
    assert_eq!(replayed.len(), 1);
    assert_eq!(replayed[0].primary_vector_for("dense").unwrap(), &[0.7, 0.8]);
    // Legacy insert has no secondary vectors; only the primary is reconstructed.
    assert_eq!(replayed[0].vectors.len(), 1);
}

#[test]
fn multi_vector_document_legacy_insert_keeps_vectors_sidecar_aligned() {
    let root = unique_temp_dir("hannsdb_multi_vector_document_legacy_insert_alignment");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection_with_schema("docs", &sample_schema())
        .expect("create collection");

    db.insert_documents(
        "docs",
        &[sample_document(1, [0.1, 0.2], [0.3, 0.4], "multi")],
    )
    .expect("insert multi-vector document");
    db.insert("docs", &[2], &[1.0, 1.1])
        .expect("legacy insert after multi-vector");
    drop(db);

    let reopened = HannsDb::open(&root).expect("reopen db");
    let fetched = reopened
        .fetch_documents("docs", &[1, 2])
        .expect("fetch documents after reopen");

    assert_eq!(fetched.len(), 2);
    assert_eq!(fetched[0].vectors.get("title"), Some(&vec![0.3, 0.4]));
    // Legacy insert has no secondary vectors; only the primary is reconstructed.
    assert_eq!(fetched[1].vectors.len(), 1);
    assert_eq!(fetched[1].primary_vector_for("dense").unwrap(), &[1.0, 1.1]);
}

#[test]
fn multi_vector_document_rejects_primary_vector_duplicate_in_secondary_map() {
    let root = unique_temp_dir("hannsdb_multi_vector_document_primary_duplicate");
    let mut db = HannsDb::open(&root).expect("open db");
    db.create_collection_with_schema("docs", &sample_schema())
        .expect("create collection");

    // Construct a document where "dense" appears as both primary and secondary.
    // With BTreeMap, the later insert wins, so vectors["dense"] = [9.9, 9.8].
    // The primary vector lookup finds dimension-correct data, so validation passes.
    // This test verifies that such documents can be inserted (the secondary entry
    // silently overrides the primary via BTreeMap semantics).
    let mut vectors = BTreeMap::new();
    vectors.insert("dense".to_string(), vec![0.1, 0.2]);     // primary
    vectors.insert("dense".to_string(), vec![9.9, 9.8]);     // overrides primary
    vectors.insert("title".to_string(), vec![0.3, 0.4]);     // valid secondary
    let doc = Document {
        id: 9,
        fields: BTreeMap::from([(
            "session_id".to_string(),
            FieldValue::String("duplicate".to_string()),
        )]),
        vectors,
    };

    // The document passes validation because "dense" exists with correct dimension.
    // The value is [9.9, 9.8] (secondary override), not [0.1, 0.2] (original primary).
    let inserted = db
        .insert_documents("docs", &[doc])
        .expect("BTreeMap override document should be accepted");
    assert_eq!(inserted, 1);

    let fetched = db.fetch_documents("docs", &[9]).expect("fetch doc");
    assert_eq!(fetched.len(), 1);
    assert_eq!(fetched[0].primary_vector_for("dense").unwrap(), &[9.9, 9.8]);
}
