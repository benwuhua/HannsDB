use std::collections::BTreeMap;

use hannsdb_core::document::{
    CollectionSchema, FieldType, FieldValue, ScalarFieldSchema, VectorFieldSchema,
};
use hannsdb_core::forward_store::{
    ChunkedFileWriter, ForwardFileFormat, ForwardRow, ForwardStoreReader, MemForwardStore,
};
use tempfile::tempdir;

fn test_schema() -> CollectionSchema {
    CollectionSchema {
        primary_vector: "vector".to_string(),
        fields: vec![
            ScalarFieldSchema::new("title", FieldType::String),
            ScalarFieldSchema::new("score", FieldType::Float64).with_flags(true, false),
            ScalarFieldSchema::new("active", FieldType::Bool).with_flags(true, false),
        ],
        vectors: vec![
            VectorFieldSchema::new("vector", 3),
            VectorFieldSchema::new("aux", 2),
        ],
    }
}

fn row(
    internal_id: u64,
    op_seq: u64,
    is_deleted: bool,
    title: Option<&str>,
    score: Option<f64>,
    active: Option<bool>,
    vector: Option<Vec<f32>>,
    aux: Option<Vec<f32>>,
) -> ForwardRow {
    let mut fields = BTreeMap::new();
    if let Some(title) = title {
        fields.insert("title".to_string(), FieldValue::String(title.to_string()));
    }
    if let Some(score) = score {
        fields.insert("score".to_string(), FieldValue::Float64(score));
    }
    if let Some(active) = active {
        fields.insert("active".to_string(), FieldValue::Bool(active));
    }

    let mut vectors = BTreeMap::new();
    if let Some(vector) = vector {
        vectors.insert("vector".to_string(), vector);
    }
    if let Some(aux) = aux {
        vectors.insert("aux".to_string(), aux);
    }

    ForwardRow {
        internal_id,
        op_seq,
        is_deleted,
        fields,
        vectors,
    }
}

#[test]
fn forward_store_roundtrips_arrow_and_parquet_equivalently() {
    let schema = test_schema();
    let mut store = MemForwardStore::new(schema.clone());
    let rows = vec![
        row(
            10,
            1,
            false,
            Some("alpha"),
            Some(1.5),
            Some(true),
            Some(vec![1.0, 2.0, 3.0]),
            Some(vec![0.1, 0.2]),
        ),
        row(
            20,
            2,
            false,
            Some("beta"),
            None,
            Some(false),
            Some(vec![4.0, 5.0, 6.0]),
            None,
        ),
        row(
            30,
            3,
            false,
            Some("gamma"),
            Some(8.25),
            None,
            Some(vec![7.0, 8.0, 9.0]),
            Some(vec![0.3, 0.4]),
        ),
    ];
    for row in rows.clone() {
        store.append(row).expect("append test row");
    }

    let dir = tempdir().expect("tempdir");
    let writer = ChunkedFileWriter::new(dir.path());
    let descriptor = writer
        .write(
            "roundtrip",
            &store,
            &[ForwardFileFormat::ArrowIpc, ForwardFileFormat::Parquet],
        )
        .expect("write both forward-store formats");

    let arrow_reader = ForwardStoreReader::open(&descriptor, ForwardFileFormat::ArrowIpc)
        .expect("open arrow reader");
    let parquet_reader = ForwardStoreReader::open(&descriptor, ForwardFileFormat::Parquet)
        .expect("open parquet reader");

    assert_eq!(arrow_reader.row_count(), rows.len());
    assert_eq!(parquet_reader.row_count(), rows.len());
    assert_eq!(
        arrow_reader.scan_columns(None).expect("scan arrow rows"),
        rows,
        "arrow reopen should preserve every logical row",
    );
    assert_eq!(
        parquet_reader.scan_columns(None).expect("scan parquet rows"),
        rows,
        "parquet reopen should preserve every logical row",
    );

    let projection = ["internal_id", "is_deleted", "title", "aux"];
    let expected_projection = vec![
        row(30, 3, false, Some("gamma"), None, None, None, Some(vec![0.3, 0.4])),
        row(10, 1, false, Some("alpha"), None, None, None, Some(vec![0.1, 0.2])),
    ];
    assert_eq!(
        arrow_reader
            .fetch_rows(&[2, 0], Some(&projection))
            .expect("projected arrow fetch"),
        expected_projection,
    );
    assert_eq!(
        parquet_reader
            .fetch_rows(&[2, 0], Some(&projection))
            .expect("projected parquet fetch"),
        expected_projection,
    );
}

#[test]
fn forward_store_preserves_latest_live_semantics_across_formats() {
    let schema = test_schema();
    let mut store = MemForwardStore::new(schema);
    let rows = vec![
        row(
            10,
            1,
            false,
            Some("stale"),
            Some(0.5),
            Some(true),
            Some(vec![1.0, 1.0, 1.0]),
            None,
        ),
        row(
            10,
            2,
            false,
            Some("fresh"),
            Some(2.5),
            Some(false),
            Some(vec![2.0, 2.0, 2.0]),
            Some(vec![0.5, 0.6]),
        ),
        row(
            20,
            3,
            false,
            Some("delete-me"),
            Some(5.0),
            Some(true),
            Some(vec![3.0, 3.0, 3.0]),
            None,
        ),
        row(20, 4, true, None, None, None, None, None),
    ];
    for row in rows {
        store.append(row).expect("append versioned row");
    }

    let dir = tempdir().expect("tempdir");
    let writer = ChunkedFileWriter::new(dir.path());
    let descriptor = writer
        .write(
            "latest-live",
            &store,
            &[ForwardFileFormat::ArrowIpc, ForwardFileFormat::Parquet],
        )
        .expect("write latest-live fixture");

    let arrow_latest = ForwardStoreReader::open(&descriptor, ForwardFileFormat::ArrowIpc)
        .expect("open arrow latest-live reader")
        .latest_live_rows();
    let parquet_latest = ForwardStoreReader::open(&descriptor, ForwardFileFormat::Parquet)
        .expect("open parquet latest-live reader")
        .latest_live_rows();

    let expected = vec![row(
        10,
        2,
        false,
        Some("fresh"),
        Some(2.5),
        Some(false),
        Some(vec![2.0, 2.0, 2.0]),
        Some(vec![0.5, 0.6]),
    )];
    assert_eq!(arrow_latest, expected);
    assert_eq!(parquet_latest, expected);
}
