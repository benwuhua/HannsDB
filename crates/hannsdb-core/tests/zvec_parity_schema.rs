use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use hannsdb_core::catalog::CollectionMetadata;
use hannsdb_core::document::{CollectionSchema, FieldType, ScalarFieldSchema};
use serde_json::{json, Value};

fn schema_roundtrip_path() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("hannsdb_zvec_schema_{nanos}"));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir.join("collection.json")
}

fn stored_schema_value() -> Value {
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
    let path = schema_roundtrip_path();

    metadata
        .save_to_path(&path)
        .expect("save collection metadata");
    let loaded = CollectionMetadata::load_from_path(&path).expect("load collection metadata");
    serde_json::to_value(loaded.schema()).expect("serialize stored schema")
}

#[test]
fn zvec_parity_schema_round_trips_multiple_vector_fields() {
    let actual = stored_schema_value();
    let expected = json!({
        "primary_vector": "dense",
        "dimension": 384,
        "metric": "cosine",
        "fields": [
            {
                "name": "session_id",
                "data_type": FieldType::String,
                "nullable": false,
                "array": false,
                "index_param": {
                    "kind": "ivf",
                    "nlist": 1024
                }
            },
            {
                "name": "tags",
                "data_type": FieldType::String,
                "nullable": true,
                "array": true,
                "index_param": null
            },
            {
                "name": "created_at",
                "data_type": FieldType::Int64,
                "nullable": false,
                "array": false,
                "index_param": null
            }
        ],
        "vectors": [
            {
                "name": "dense",
                "data_type": "VectorFp32",
                "dimension": 384,
                "index_param": {
                    "kind": "ivf",
                    "nlist": 1024
                }
            },
            {
                "name": "title",
                "data_type": "VectorFp32",
                "dimension": 384,
                "index_param": {
                    "kind": "hnsw",
                    "m": 32,
                    "ef_construction": 128
                }
            }
        ],
        "hnsw_m": 16,
        "hnsw_ef_construction": 128
    });

    assert_eq!(actual, expected);
}

#[test]
fn zvec_parity_schema_round_trips_nullable_and_array_scalar_fields() {
    let actual_fields = stored_schema_value()["fields"].clone();
    let expected_fields = json!([
        {
            "name": "session_id",
            "data_type": FieldType::String,
            "nullable": false,
            "array": false,
            "index_param": {
                "kind": "ivf",
                "nlist": 1024
            }
        },
        {
            "name": "tags",
            "data_type": FieldType::String,
            "nullable": true,
            "array": true,
            "index_param": null
        },
        {
            "name": "created_at",
            "data_type": FieldType::Int64,
            "nullable": false,
            "array": false,
            "index_param": null
        }
    ]);

    assert_eq!(actual_fields, expected_fields);
}
