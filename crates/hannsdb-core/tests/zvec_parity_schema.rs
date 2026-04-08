use hannsdb_core::catalog::{CollectionMetadata, CATALOG_FORMAT_VERSION};
use hannsdb_core::document::{
    CollectionSchema, FieldType, ScalarFieldSchema, VectorFieldSchema, VectorIndexSchema,
};
use serde_json::{json, Value};

fn stored_schema_value() -> Value {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("collection.json");
    let schema = CollectionSchema {
        primary_vector: "dense".to_string(),
        fields: vec![
            ScalarFieldSchema::new("session_id", FieldType::String),
            ScalarFieldSchema::new("tags", FieldType::String).with_flags(true, true),
            ScalarFieldSchema::new("created_at", FieldType::Int64),
        ],
        vectors: vec![
            VectorFieldSchema::new("title", 384),
            VectorFieldSchema::new("dense", 384).with_index_param(
                VectorIndexSchema::hnsw(Some("cosine"), 32, 128)
                    .with_quantize_type(Some("fp16")),
            ),
        ],
    };
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
    let vectors = actual["vectors"]
        .as_array()
        .expect("schema should expose vector metadata");

    assert_eq!(vectors.len(), 2);
    assert_eq!(actual["primary_vector"], "dense");
    assert_eq!(vectors[0]["name"], "title");
    assert_eq!(vectors[0]["data_type"], "VectorFp32");
    assert_eq!(vectors[0]["dimension"], 384);
    assert_eq!(vectors[1]["name"], "dense");
    assert_eq!(vectors[1]["data_type"], "VectorFp32");
    assert_eq!(vectors[1]["dimension"], 384);
}

#[test]
fn zvec_parity_schema_round_trips_vector_index_metadata() {
    let actual = stored_schema_value();
    let vectors = actual
        .get("vectors")
        .cloned()
        .expect("schema should expose vector metadata");
    let dense_vector = vectors
        .as_array()
        .and_then(|entries| {
            entries
                .iter()
                .find(|entry| entry.get("name") == Some(&Value::String("dense".to_string())))
        })
        .expect("expected dense vector entry");
    let index_param = dense_vector
        .get("index_param")
        .cloned()
        .expect("vector should expose index metadata");
    let expected_index_param = json!({
        "kind": "hnsw",
        "metric": "cosine",
        "m": 32,
        "ef_construction": 128,
        "quantize_type": "fp16"
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

#[test]
fn zvec_parity_schema_migrates_legacy_single_vector_metadata_into_field_registries() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("collection.json");
    let legacy_metadata = json!({
        "format_version": CATALOG_FORMAT_VERSION,
        "name": "docs",
        "dimension": 384,
        "metric": "cosine",
        "primary_vector": "dense",
        "fields": [
            {
                "name": "session_id",
                "data_type": "String"
            }
        ],
        "hnsw_m": 16,
        "hnsw_ef_construction": 128
    });
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&legacy_metadata).expect("serialize legacy metadata"),
    )
    .expect("write legacy metadata");

    let loaded = CollectionMetadata::load_from_path(&path).expect("load collection metadata");
    let schema = loaded.schema();

    assert_eq!(schema.primary_vector, "dense");
    assert_eq!(
        schema.fields,
        vec![ScalarFieldSchema::new("session_id", FieldType::String)]
    );
    assert_eq!(
        schema.vectors,
        vec![
            VectorFieldSchema::new("dense", 384).with_index_param(VectorIndexSchema::hnsw(
                Some("cosine"),
                16,
                128
            ))
        ]
    );
}

#[test]
fn zvec_parity_schema_legacy_constructor_maps_to_single_primary_vector_registry() {
    let schema = CollectionSchema::new(
        "dense",
        384,
        "cosine",
        vec![ScalarFieldSchema::new("session_id", FieldType::String)],
    );

    assert_eq!(
        schema.vectors,
        vec![
            VectorFieldSchema::new("dense", 384).with_index_param(VectorIndexSchema::hnsw(
                Some("cosine"),
                16,
                128
            ))
        ]
    );
    assert_eq!(schema.primary_vector, "dense");
}
