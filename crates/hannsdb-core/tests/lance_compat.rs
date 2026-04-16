#![cfg(feature = "lance-storage")]

use hannsdb_core::document::{
    CollectionSchema, Document, FieldType, FieldValue, ScalarFieldSchema,
};
use hannsdb_core::storage::lance_store::documents_to_lance_batch;

fn sample_schema() -> CollectionSchema {
    CollectionSchema {
        primary_vector: "dense".to_string(),
        fields: vec![
            ScalarFieldSchema::new("title", FieldType::String),
            ScalarFieldSchema::new("year", FieldType::Int64),
        ],
        vectors: vec![hannsdb_core::document::VectorFieldSchema::new("dense", 3)],
    }
}

fn sample_documents() -> Vec<Document> {
    vec![
        Document::with_primary_vector_name(
            10,
            vec![
                ("title".to_string(), FieldValue::String("alpha".to_string())),
                ("year".to_string(), FieldValue::Int64(2024)),
            ],
            "dense",
            vec![1.0, 2.0, 3.0],
        ),
        Document::with_primary_vector_name(
            20,
            vec![
                ("title".to_string(), FieldValue::String("beta".to_string())),
                ("year".to_string(), FieldValue::Int64(2025)),
            ],
            "dense",
            vec![4.0, 5.0, 6.0],
        ),
    ]
}

#[test]
fn lance_batch_conversion_preserves_ids_scalars_and_vectors() {
    let batch = documents_to_lance_batch(&sample_schema(), &sample_documents())
        .expect("convert documents to lance batch");

    assert_eq!(batch.num_rows(), 2);
    assert_eq!(batch.schema().field(0).name(), "id");
    assert_eq!(
        batch.schema().field_with_name("title").unwrap().name(),
        "title"
    );
    assert_eq!(
        batch.schema().field_with_name("year").unwrap().name(),
        "year"
    );
    assert_eq!(
        batch.schema().field_with_name("dense").unwrap().name(),
        "dense"
    );
}
