use std::collections::BTreeMap;

use hannsdb_core::document::{
    CollectionSchema, FieldType, FieldValue, ScalarFieldSchema, VectorFieldSchema,
};
use hannsdb_core::forward_store::{
    ChunkedFileWriter, ForwardFileFormat, ForwardRow, ForwardStoreReader, MemForwardStore,
};

fn sample_schema() -> CollectionSchema {
    let mut schema = CollectionSchema::new(
        "dense",
        3,
        "l2",
        vec![
            ScalarFieldSchema::new("session_id", FieldType::String).with_flags(true, false),
            ScalarFieldSchema::new("turn", FieldType::Int64),
            ScalarFieldSchema::new("score", FieldType::Float).with_flags(true, false),
        ],
    );
    schema.vectors.push(VectorFieldSchema::new("secondary", 2));
    schema
}

fn base_rows() -> Vec<ForwardRow> {
    vec![
        ForwardRow {
            internal_id: 10,
            op_seq: 1,
            is_deleted: false,
            fields: BTreeMap::from([
                (
                    "session_id".to_string(),
                    FieldValue::String("alpha".to_string()),
                ),
                ("turn".to_string(), FieldValue::Int64(1)),
                ("score".to_string(), FieldValue::Float(1.5)),
            ]),
            vectors: BTreeMap::from([
                ("dense".to_string(), vec![0.1, 0.2, 0.3]),
                ("secondary".to_string(), vec![9.0, 9.5]),
            ]),
        },
        ForwardRow {
            internal_id: 10,
            op_seq: 2,
            is_deleted: false,
            fields: BTreeMap::from([
                (
                    "session_id".to_string(),
                    FieldValue::String("alpha".to_string()),
                ),
                ("turn".to_string(), FieldValue::Int64(2)),
                ("score".to_string(), FieldValue::Float(2.5)),
            ]),
            vectors: BTreeMap::from([("dense".to_string(), vec![1.0, 1.1, 1.2])]),
        },
        ForwardRow {
            internal_id: 20,
            op_seq: 1,
            is_deleted: false,
            fields: BTreeMap::from([
                (
                    "session_id".to_string(),
                    FieldValue::String("beta".to_string()),
                ),
                ("turn".to_string(), FieldValue::Int64(1)),
            ]),
            vectors: BTreeMap::from([
                ("dense".to_string(), vec![2.0, 2.1, 2.2]),
                ("secondary".to_string(), vec![8.0, 8.5]),
            ]),
        },
        ForwardRow {
            internal_id: 20,
            op_seq: 3,
            is_deleted: true,
            fields: BTreeMap::from([
                (
                    "session_id".to_string(),
                    FieldValue::String("beta".to_string()),
                ),
                ("turn".to_string(), FieldValue::Int64(3)),
            ]),
            vectors: BTreeMap::new(),
        },
        ForwardRow {
            internal_id: 30,
            op_seq: 4,
            is_deleted: false,
            fields: BTreeMap::from([("turn".to_string(), FieldValue::Int64(9))]),
            vectors: BTreeMap::from([
                ("dense".to_string(), vec![3.0, 3.1, 3.2]),
                ("secondary".to_string(), vec![7.0, 7.5]),
            ]),
        },
    ]
}

fn build_store(rows: &[ForwardRow]) -> MemForwardStore {
    let mut store = MemForwardStore::new(sample_schema());
    for row in rows {
        store.append(row.clone()).expect("append forward-store row");
    }
    store
}

fn write_descriptor(rows: &[ForwardRow]) -> hannsdb_core::forward_store::ForwardStoreDescriptor {
    let temp = tempfile::tempdir().expect("tempdir");
    let base_dir = temp.keep();
    let store = build_store(rows);
    let writer = ChunkedFileWriter::new(&base_dir);
    let descriptor = writer
        .write(
            "forward_store_roundtrip",
            &store,
            &[ForwardFileFormat::ArrowIpc, ForwardFileFormat::Parquet],
        )
        .expect("write forward-store artifacts");

    assert_eq!(descriptor.row_count, rows.len());
    assert!(
        descriptor.artifact(ForwardFileFormat::ArrowIpc).is_some(),
        "descriptor should include Arrow IPC artifact"
    );
    assert!(
        descriptor.artifact(ForwardFileFormat::Parquet).is_some(),
        "descriptor should include Parquet artifact"
    );

    descriptor
}

fn open_pair(
    descriptor: &hannsdb_core::forward_store::ForwardStoreDescriptor,
) -> (ForwardStoreReader, ForwardStoreReader) {
    (
        ForwardStoreReader::open(descriptor, ForwardFileFormat::ArrowIpc)
            .expect("open arrow reader"),
        ForwardStoreReader::open(descriptor, ForwardFileFormat::Parquet)
            .expect("open parquet reader"),
    )
}

#[test]
fn forward_store_roundtrip_arrow_and_parquet_are_equivalent() {
    let rows = base_rows();
    let descriptor = write_descriptor(&rows);
    let (arrow_reader, parquet_reader) = open_pair(&descriptor);

    assert_eq!(arrow_reader.row_count(), rows.len());
    assert_eq!(parquet_reader.row_count(), rows.len());
    assert_eq!(
        arrow_reader.scan_columns(None).expect("scan arrow rows"),
        parquet_reader
            .scan_columns(None)
            .expect("scan parquet rows"),
    );
}

#[test]
fn forward_store_fetch_rows_projects_requested_columns_for_both_formats() {
    let descriptor = write_descriptor(&base_rows());
    let (arrow_reader, parquet_reader) = open_pair(&descriptor);
    let requested_columns = ["internal_id", "turn", "secondary"];

    let expected = vec![
        ForwardRow {
            internal_id: 10,
            op_seq: 0,
            is_deleted: false,
            fields: BTreeMap::from([("turn".to_string(), FieldValue::Int64(2))]),
            vectors: BTreeMap::new(),
        },
        ForwardRow {
            internal_id: 30,
            op_seq: 0,
            is_deleted: false,
            fields: BTreeMap::from([("turn".to_string(), FieldValue::Int64(9))]),
            vectors: BTreeMap::from([("secondary".to_string(), vec![7.0, 7.5])]),
        },
    ];

    let arrow_rows = arrow_reader
        .fetch_rows(&[1, 4], Some(&requested_columns))
        .expect("fetch projected arrow rows");
    let parquet_rows = parquet_reader
        .fetch_rows(&[1, 4], Some(&requested_columns))
        .expect("fetch projected parquet rows");

    assert_eq!(arrow_rows, expected);
    assert_eq!(parquet_rows, expected);
}

#[test]
fn forward_store_latest_live_rows_respect_versions_and_terminal_tombstones() {
    let descriptor = write_descriptor(&base_rows());
    let (arrow_reader, parquet_reader) = open_pair(&descriptor);

    let expected = vec![
        ForwardRow {
            internal_id: 10,
            op_seq: 2,
            is_deleted: false,
            fields: BTreeMap::from([
                (
                    "session_id".to_string(),
                    FieldValue::String("alpha".to_string()),
                ),
                ("turn".to_string(), FieldValue::Int64(2)),
                ("score".to_string(), FieldValue::Float(2.5)),
            ]),
            vectors: BTreeMap::from([("dense".to_string(), vec![1.0, 1.1, 1.2])]),
        },
        ForwardRow {
            internal_id: 30,
            op_seq: 4,
            is_deleted: false,
            fields: BTreeMap::from([("turn".to_string(), FieldValue::Int64(9))]),
            vectors: BTreeMap::from([
                ("dense".to_string(), vec![3.0, 3.1, 3.2]),
                ("secondary".to_string(), vec![7.0, 7.5]),
            ]),
        },
    ];

    assert_eq!(arrow_reader.latest_live_rows(), expected);
    assert_eq!(parquet_reader.latest_live_rows(), expected);
}

#[test]
fn forward_store_rejects_undeclared_fields_inside_the_new_core_only() {
    let mut store = MemForwardStore::new(sample_schema());
    let err = store
        .append(ForwardRow {
            internal_id: 40,
            op_seq: 1,
            is_deleted: false,
            fields: BTreeMap::from([
                (
                    "session_id".to_string(),
                    FieldValue::String("gamma".to_string()),
                ),
                (
                    "unexpected".to_string(),
                    FieldValue::String("out-of-scope".to_string()),
                ),
            ]),
            vectors: BTreeMap::from([("dense".to_string(), vec![4.0, 4.1, 4.2])]),
        })
        .expect_err("undeclared field should be rejected by forward-store core");

    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    assert!(
        err.to_string()
            .contains("undeclared forward-store field: unexpected"),
        "unexpected error: {err}"
    );
}
