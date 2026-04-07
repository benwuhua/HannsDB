use hannsdb_core::catalog::CollectionMetadata;
use hannsdb_core::document::{CollectionSchema, FieldType, ScalarFieldSchema};
use serde_json::{json, Value};

fn stored_schema_value() -> Value {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("collection.json");
    let schema = CollectionSchema::new(
        "dense",
        384,
        "cosine",
        vec![
            ScalarFieldSchema::new("session_id", FieldType::String),
            ScalarFieldSchema::new("tags", FieldType::String),
            ScalarFieldSchema::new("created_at", FieldType::Int64),
        ],
    );
    let metadata = CollectionMetadata::new_with_schema("docs", schema);

    metadata
        .save_to_path(&path)
        .expect("save collection metadata");
    let loaded = CollectionMetadata::load_from_path(&path).expect("load collection metadata");
    serde_json::to_value(loaded.schema()).expect("serialize stored schema")
}

#[test]
fn zvec_parity_schema_round_trips_multiple_vector_fields() {
    let actual = stored_schema_value();
    let vectors = actual
        .get("vectors")
        .cloned()
        .expect("schema should expose vector metadata");
    let expected_vectors = json!([
        {
            "name": "dense",
            "data_type": "VectorFp32",
            "dimension": 384
        },
        {
            "name": "title",
            "data_type": "VectorFp32",
            "dimension": 384
        }
    ]);

    assert_eq!(vectors, expected_vectors);
}

#[test]
fn zvec_parity_schema_round_trips_vector_index_metadata() {
    let actual = stored_schema_value();
    let vectors = actual
        .get("vectors")
        .cloned()
        .expect("schema should expose vector metadata");
    let first_vector = vectors
        .as_array()
        .and_then(|entries| entries.first())
        .expect("expected at least one vector entry");
    let index_param = first_vector
        .get("index_param")
        .cloned()
        .expect("vector should expose index metadata");
    let expected_index_param = json!({
        "kind": "hnsw",
        "m": 32,
        "ef_construction": 128
    });

    assert_eq!(index_param, expected_index_param);
}

#[test]
fn zvec_parity_schema_round_trips_nullable_and_array_scalar_fields() {
    let actual_fields = stored_schema_value()["fields"].clone();
    let expected_fields = json!([
        {
            "name": "session_id",
            "data_type": FieldType::String,
            "nullable": false,
            "array": false
        },
        {
            "name": "tags",
            "data_type": FieldType::String,
            "nullable": true,
            "array": true
        },
        {
            "name": "created_at",
            "data_type": FieldType::Int64,
            "nullable": false,
            "array": false
        }
    ]);

    assert_eq!(actual_fields, expected_fields);
}
