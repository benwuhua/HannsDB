use std::collections::BTreeMap;
use std::io;
use std::sync::Arc;

use arrow::array::builder::*;
use arrow::array::{
    ArrayRef, BooleanArray, FixedSizeListArray, Float32Array, Int64Array, StringArray, UInt64Array,
};
use arrow::buffer::{BooleanBuffer, NullBuffer};
use arrow::datatypes::{DataType, Field};
use arrow::record_batch::RecordBatch;

use crate::document::{Document, FieldType, FieldValue, SparseVector};

use super::schema::{
    field_value_matches_schema, ForwardColumnKind, ForwardColumnSchema, ForwardStoreSchema,
    ForwardSystemColumnKind,
};

#[derive(Debug, Clone, PartialEq)]
pub struct ForwardStoreRow {
    pub internal_id: i64,
    pub op_seq: u64,
    pub is_deleted: bool,
    pub fields: BTreeMap<String, FieldValue>,
    pub vectors: BTreeMap<String, Vec<f32>>,
    pub sparse_vectors: BTreeMap<String, SparseVector>,
}

impl ForwardStoreRow {
    pub fn new(
        internal_id: i64,
        op_seq: u64,
        is_deleted: bool,
        fields: BTreeMap<String, FieldValue>,
        vectors: BTreeMap<String, Vec<f32>>,
    ) -> Self {
        Self {
            internal_id,
            op_seq,
            is_deleted,
            fields,
            vectors,
            sparse_vectors: BTreeMap::new(),
        }
    }

    pub fn from_document(document: Document, op_seq: u64, is_deleted: bool) -> Self {
        Self {
            internal_id: document.id,
            op_seq,
            is_deleted,
            fields: document.fields,
            vectors: document.vectors,
            sparse_vectors: document.sparse_vectors,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemForwardStore {
    schema: ForwardStoreSchema,
    rows: Vec<ForwardStoreRow>,
    estimated_bytes: usize,
}

impl MemForwardStore {
    pub fn new(schema: ForwardStoreSchema) -> Self {
        Self {
            schema,
            rows: Vec::new(),
            estimated_bytes: 0,
        }
    }

    pub fn schema(&self) -> &ForwardStoreSchema {
        &self.schema
    }

    pub fn rows(&self) -> &[ForwardStoreRow] {
        &self.rows
    }

    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn estimated_bytes(&self) -> usize {
        self.estimated_bytes
    }

    pub fn is_full(&self, target_bytes: usize) -> bool {
        self.estimated_bytes >= target_bytes
    }

    pub fn append(&mut self, row: ForwardStoreRow) -> io::Result<()> {
        validate_row_against_schema(&self.schema, &row)?;
        self.estimated_bytes += estimate_row_size(&row);
        self.rows.push(row);
        Ok(())
    }

    pub fn append_rows<I>(&mut self, rows: I) -> io::Result<()>
    where
        I: IntoIterator<Item = ForwardStoreRow>,
    {
        for row in rows {
            self.append(row)?;
        }
        Ok(())
    }

    pub fn record_batch(&self) -> io::Result<RecordBatch> {
        let schema = Arc::new(self.schema.arrow_schema()?);
        let arrays = self
            .schema
            .columns()
            .into_iter()
            .map(|column| build_column_array(&self.rows, column))
            .collect::<io::Result<Vec<_>>>()?;
        RecordBatch::try_new(schema, arrays).map_err(arrow_to_io)
    }
}

fn validate_row_against_schema(schema: &ForwardStoreSchema, row: &ForwardStoreRow) -> io::Result<()> {
    if !row.sparse_vectors.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "forward-store core does not yet support sparse vector fields",
        ));
    }

    for (name, value) in &row.fields {
        schema.validate_scalar_value(name, value)?;
    }

    for (name, vector) in &row.vectors {
        let column = schema.vector_field(name).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "forward-store row vector '{}' is not declared in the collection schema",
                    name
                ),
            )
        })?;
        let expected = column.dimension.expect("vector column dimension");
        if vector.len() != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "forward-store row vector '{}' dimension mismatch: expected {}, got {}",
                    name,
                    expected,
                    vector.len()
                ),
            ));
        }
    }

    if !row.vectors.contains_key(schema.primary_vector.as_str()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "forward-store row is missing primary vector '{}'",
                schema.primary_vector
            ),
        ));
    }

    Ok(())
}

fn estimate_row_size(row: &ForwardStoreRow) -> usize {
    let mut bytes = std::mem::size_of::<i64>() + std::mem::size_of::<u64>() + 1;
    for (name, value) in &row.fields {
        bytes += name.len() + estimate_field_value_size(value);
    }
    for (name, vector) in &row.vectors {
        bytes += name.len() + std::mem::size_of_val(vector.as_slice());
    }
    bytes
}

fn estimate_field_value_size(value: &FieldValue) -> usize {
    match value {
        FieldValue::String(v) => v.len(),
        FieldValue::Int64(_) => std::mem::size_of::<i64>(),
        FieldValue::Int32(_) => std::mem::size_of::<i32>(),
        FieldValue::UInt32(_) => std::mem::size_of::<u32>(),
        FieldValue::UInt64(_) => std::mem::size_of::<u64>(),
        FieldValue::Float(_) => std::mem::size_of::<f32>(),
        FieldValue::Float64(_) => std::mem::size_of::<f64>(),
        FieldValue::Bool(_) => std::mem::size_of::<bool>(),
        FieldValue::Array(items) => items.iter().map(estimate_field_value_size).sum(),
    }
}

fn build_column_array(rows: &[ForwardStoreRow], column: &ForwardColumnSchema) -> io::Result<ArrayRef> {
    match column.kind {
        ForwardColumnKind::System {
            system: ForwardSystemColumnKind::InternalId,
        } => Ok(Arc::new(Int64Array::from(
            rows.iter().map(|row| row.internal_id).collect::<Vec<_>>(),
        ))),
        ForwardColumnKind::System {
            system: ForwardSystemColumnKind::OpSeq,
        } => Ok(Arc::new(UInt64Array::from(
            rows.iter().map(|row| row.op_seq).collect::<Vec<_>>(),
        ))),
        ForwardColumnKind::System {
            system: ForwardSystemColumnKind::IsDeleted,
        } => Ok(Arc::new(BooleanArray::from(
            rows.iter().map(|row| row.is_deleted).collect::<Vec<_>>(),
        ))),
        ForwardColumnKind::Scalar => build_scalar_column(rows, column),
        ForwardColumnKind::PrimaryVector | ForwardColumnKind::SecondaryVector => {
            build_vector_column(rows, column)
        }
    }
}

fn build_scalar_column(rows: &[ForwardStoreRow], column: &ForwardColumnSchema) -> io::Result<ArrayRef> {
    if column.array {
        return build_scalar_array_column(rows, column);
    }

    let name = column.name.as_str();
    match column.data_type {
        FieldType::String => {
            let mut builder = StringBuilder::new();
            for row in rows {
                match row.fields.get(name) {
                    Some(FieldValue::String(value)) => builder.append_value(value),
                    None => builder.append_null(),
                    Some(other) => return scalar_mismatch_error(column, other),
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        FieldType::Int64 => {
            let mut builder = Int64Builder::new();
            for row in rows {
                match row.fields.get(name) {
                    Some(FieldValue::Int64(value)) => builder.append_value(*value),
                    None => builder.append_null(),
                    Some(other) => return scalar_mismatch_error(column, other),
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        FieldType::Int32 => {
            let mut builder = Int32Builder::new();
            for row in rows {
                match row.fields.get(name) {
                    Some(FieldValue::Int32(value)) => builder.append_value(*value),
                    None => builder.append_null(),
                    Some(other) => return scalar_mismatch_error(column, other),
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        FieldType::UInt32 => {
            let mut builder = UInt32Builder::new();
            for row in rows {
                match row.fields.get(name) {
                    Some(FieldValue::UInt32(value)) => builder.append_value(*value),
                    None => builder.append_null(),
                    Some(other) => return scalar_mismatch_error(column, other),
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        FieldType::UInt64 => {
            let mut builder = UInt64Builder::new();
            for row in rows {
                match row.fields.get(name) {
                    Some(FieldValue::UInt64(value)) => builder.append_value(*value),
                    None => builder.append_null(),
                    Some(other) => return scalar_mismatch_error(column, other),
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        FieldType::Float => {
            let mut builder = Float32Builder::new();
            for row in rows {
                match row.fields.get(name) {
                    Some(FieldValue::Float(value)) => builder.append_value(*value),
                    None => builder.append_null(),
                    Some(other) => return scalar_mismatch_error(column, other),
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        FieldType::Float64 => {
            let mut builder = Float64Builder::new();
            for row in rows {
                match row.fields.get(name) {
                    Some(FieldValue::Float64(value)) => builder.append_value(*value),
                    None => builder.append_null(),
                    Some(other) => return scalar_mismatch_error(column, other),
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        FieldType::Bool => {
            let mut builder = BooleanBuilder::new();
            for row in rows {
                match row.fields.get(name) {
                    Some(FieldValue::Bool(value)) => builder.append_value(*value),
                    None => builder.append_null(),
                    Some(other) => return scalar_mismatch_error(column, other),
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        FieldType::VectorFp32 | FieldType::VectorFp16 | FieldType::VectorSparse => Err(
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "forward-store scalar column '{}' cannot use vector data type {:?}",
                    column.name, column.data_type
                ),
            ),
        ),
    }
}

fn build_scalar_array_column(
    rows: &[ForwardStoreRow],
    column: &ForwardColumnSchema,
) -> io::Result<ArrayRef> {
    let name = column.name.as_str();
    match column.data_type {
        FieldType::String => {
            let mut builder = ListBuilder::new(StringBuilder::new());
            for row in rows {
                append_array_row(
                    &mut builder,
                    row.fields.get(name),
                    |values, item| match item {
                        FieldValue::String(value) => {
                            values.append_value(value);
                            Ok(())
                        }
                        other => array_item_mismatch_error(column, other),
                    },
                )?;
            }
            Ok(Arc::new(builder.finish()))
        }
        FieldType::Int64 => {
            let mut builder = ListBuilder::new(Int64Builder::new());
            for row in rows {
                append_array_row(
                    &mut builder,
                    row.fields.get(name),
                    |values, item| match item {
                        FieldValue::Int64(value) => {
                            values.append_value(*value);
                            Ok(())
                        }
                        other => array_item_mismatch_error(column, other),
                    },
                )?;
            }
            Ok(Arc::new(builder.finish()))
        }
        FieldType::Int32 => {
            let mut builder = ListBuilder::new(Int32Builder::new());
            for row in rows {
                append_array_row(
                    &mut builder,
                    row.fields.get(name),
                    |values, item| match item {
                        FieldValue::Int32(value) => {
                            values.append_value(*value);
                            Ok(())
                        }
                        other => array_item_mismatch_error(column, other),
                    },
                )?;
            }
            Ok(Arc::new(builder.finish()))
        }
        FieldType::UInt32 => {
            let mut builder = ListBuilder::new(UInt32Builder::new());
            for row in rows {
                append_array_row(
                    &mut builder,
                    row.fields.get(name),
                    |values, item| match item {
                        FieldValue::UInt32(value) => {
                            values.append_value(*value);
                            Ok(())
                        }
                        other => array_item_mismatch_error(column, other),
                    },
                )?;
            }
            Ok(Arc::new(builder.finish()))
        }
        FieldType::UInt64 => {
            let mut builder = ListBuilder::new(UInt64Builder::new());
            for row in rows {
                append_array_row(
                    &mut builder,
                    row.fields.get(name),
                    |values, item| match item {
                        FieldValue::UInt64(value) => {
                            values.append_value(*value);
                            Ok(())
                        }
                        other => array_item_mismatch_error(column, other),
                    },
                )?;
            }
            Ok(Arc::new(builder.finish()))
        }
        FieldType::Float => {
            let mut builder = ListBuilder::new(Float32Builder::new());
            for row in rows {
                append_array_row(
                    &mut builder,
                    row.fields.get(name),
                    |values, item| match item {
                        FieldValue::Float(value) => {
                            values.append_value(*value);
                            Ok(())
                        }
                        other => array_item_mismatch_error(column, other),
                    },
                )?;
            }
            Ok(Arc::new(builder.finish()))
        }
        FieldType::Float64 => {
            let mut builder = ListBuilder::new(Float64Builder::new());
            for row in rows {
                append_array_row(
                    &mut builder,
                    row.fields.get(name),
                    |values, item| match item {
                        FieldValue::Float64(value) => {
                            values.append_value(*value);
                            Ok(())
                        }
                        other => array_item_mismatch_error(column, other),
                    },
                )?;
            }
            Ok(Arc::new(builder.finish()))
        }
        FieldType::Bool => {
            let mut builder = ListBuilder::new(BooleanBuilder::new());
            for row in rows {
                append_array_row(
                    &mut builder,
                    row.fields.get(name),
                    |values, item| match item {
                        FieldValue::Bool(value) => {
                            values.append_value(*value);
                            Ok(())
                        }
                        other => array_item_mismatch_error(column, other),
                    },
                )?;
            }
            Ok(Arc::new(builder.finish()))
        }
        FieldType::VectorFp32 | FieldType::VectorFp16 | FieldType::VectorSparse => Err(
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "forward-store array column '{}' cannot use vector data type {:?}",
                    column.name, column.data_type
                ),
            ),
        ),
    }
}

fn append_array_row<T, F>(
    builder: &mut ListBuilder<T>,
    value: Option<&FieldValue>,
    mut append_item: F,
) -> io::Result<()>
where
    T: ArrayBuilder,
    F: FnMut(&mut T, &FieldValue) -> io::Result<()>,
{
    match value {
        None => {
            builder.append(false);
            Ok(())
        }
        Some(FieldValue::Array(items)) => {
            let values = builder.values();
            for item in items {
                append_item(values, item)?;
            }
            builder.append(true);
            Ok(())
        }
        Some(other) => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("expected array value, got {:?}", other),
        )),
    }
}

fn build_vector_column(rows: &[ForwardStoreRow], column: &ForwardColumnSchema) -> io::Result<ArrayRef> {
    let dimension = column.dimension.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("forward-store vector column '{}' is missing dimension", column.name),
        )
    })?;

    let mut values = Float32Builder::new();
    let mut validity = BooleanBufferBuilder::new(rows.len());
    let mut saw_null = false;

    for row in rows {
        match row.vectors.get(column.name.as_str()) {
            Some(vector) => {
                if vector.len() != dimension {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!(
                            "forward-store row vector '{}' dimension mismatch: expected {}, got {}",
                            column.name,
                            dimension,
                            vector.len()
                        ),
                    ));
                }
                values.append_slice(vector);
                validity.append(true);
            }
            None if column.nullable => {
                for _ in 0..dimension {
                    values.append_value(0.0);
                }
                validity.append(false);
                saw_null = true;
            }
            None => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "forward-store row is missing required vector '{}'",
                        column.name
                    ),
                ))
            }
        }
    }

    let values = Arc::new(values.finish());
    let nulls = if saw_null {
        Some(NullBuffer::new(BooleanBuffer::from(validity.finish())))
    } else {
        None
    };

    let array = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, false)),
        dimension as i32,
        values,
        nulls,
    )
    .map_err(arrow_to_io)?;

    Ok(Arc::new(array))
}

fn scalar_mismatch_error(column: &ForwardColumnSchema, value: &FieldValue) -> io::Result<ArrayRef> {
    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!(
            "forward-store column '{}' expected {:?}{} but got {:?}",
            column.name,
            column.data_type,
            if column.array { "[]" } else { "" },
            value
        ),
    ))
}

fn array_item_mismatch_error<T>(column: &ForwardColumnSchema, value: &FieldValue) -> io::Result<T> {
    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!(
            "forward-store column '{}' expected {:?}[] item but got {:?}",
            column.name, column.data_type, value
        ),
    ))
}

fn arrow_to_io(err: arrow::error::ArrowError) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{CollectionSchema, FieldType, ScalarFieldSchema, VectorFieldSchema};

    #[test]
    fn forward_store_mem_store_preserves_row_order_and_vectors() {
        let mut collection = CollectionSchema::new(
            "embedding",
            3,
            "l2",
            vec![
                ScalarFieldSchema::new("title", FieldType::String),
                ScalarFieldSchema::new("tags", FieldType::String).with_flags(true, true),
            ],
        );
        collection.vectors.push(VectorFieldSchema::new("image", 2));

        let schema = ForwardStoreSchema::from_collection_schema(&collection).expect("schema");
        let mut store = MemForwardStore::new(schema);

        let mut first_fields = BTreeMap::new();
        first_fields.insert("title".to_string(), FieldValue::String("alpha".to_string()));
        first_fields.insert(
            "tags".to_string(),
            FieldValue::Array(vec![
                FieldValue::String("x".to_string()),
                FieldValue::String("y".to_string()),
            ]),
        );
        let mut first_vectors = BTreeMap::new();
        first_vectors.insert("embedding".to_string(), vec![1.0, 2.0, 3.0]);
        first_vectors.insert("image".to_string(), vec![9.0, 8.0]);
        store
            .append(ForwardStoreRow::new(7, 11, false, first_fields, first_vectors))
            .expect("append first row");

        let mut second_fields = BTreeMap::new();
        second_fields.insert("title".to_string(), FieldValue::String("beta".to_string()));
        let mut second_vectors = BTreeMap::new();
        second_vectors.insert("embedding".to_string(), vec![4.0, 5.0, 6.0]);
        store
            .append(ForwardStoreRow::new(8, 12, true, second_fields, second_vectors))
            .expect("append second row");

        assert_eq!(store.row_count(), 2);
        assert!(store.estimated_bytes() > 0);
        assert!(store.is_full(1));

        let batch = store.record_batch().expect("record batch");
        let internal_ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("internal ids");
        let titles = batch
            .column(3)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("titles");
        let primary_vectors = batch
            .column(5)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .expect("primary vectors");
        let secondary_vectors = batch
            .column(6)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .expect("secondary vectors");

        assert_eq!(internal_ids.values(), &[7, 8]);
        assert_eq!(titles.value(0), "alpha");
        assert_eq!(titles.value(1), "beta");

        let first_primary = primary_vectors.value(0);
        let first_primary = first_primary
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("first primary");
        assert_eq!(first_primary.values(), &[1.0, 2.0, 3.0]);
        assert!(secondary_vectors.is_null(1));
    }

    #[test]
    fn forward_store_mem_store_rejects_undeclared_fields_only_inside_core() {
        let schema = ForwardStoreSchema::from_collection_schema(&CollectionSchema::new(
            "embedding",
            3,
            "l2",
            vec![ScalarFieldSchema::new("title", FieldType::String)],
        ))
        .expect("schema");
        let mut store = MemForwardStore::new(schema);

        let mut fields = BTreeMap::new();
        fields.insert("rogue".to_string(), FieldValue::String("value".to_string()));
        let mut vectors = BTreeMap::new();
        vectors.insert("embedding".to_string(), vec![1.0, 2.0, 3.0]);

        let err = store
            .append(ForwardStoreRow::new(1, 1, false, fields, vectors))
            .expect_err("undeclared field should fail");
        assert!(
            err.to_string().contains("is not declared in the collection schema"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn forward_store_mem_store_rejects_sparse_vector_rows() {
        let schema = ForwardStoreSchema::from_collection_schema(&CollectionSchema::new(
            "embedding",
            3,
            "l2",
            vec![],
        ))
        .expect("schema");
        let mut store = MemForwardStore::new(schema);

        let mut row = ForwardStoreRow::new(1, 1, false, BTreeMap::new(), {
            let mut vectors = BTreeMap::new();
            vectors.insert("embedding".to_string(), vec![1.0, 2.0, 3.0]);
            vectors
        });
        row.sparse_vectors
            .insert("terms".to_string(), SparseVector::new(vec![1], vec![0.5]));

        let err = store.append(row).expect_err("sparse vectors should fail");
        assert!(
            err.to_string().contains("does not yet support sparse vector fields"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn forward_store_mem_store_from_document_carries_system_columns() {
        let schema = ForwardStoreSchema::from_collection_schema(&CollectionSchema::new(
            "embedding",
            2,
            "l2",
            vec![ScalarFieldSchema::new("title", FieldType::String)],
        ))
        .expect("schema");
        let mut store = MemForwardStore::new(schema);

        let row = ForwardStoreRow::from_document(
            Document::with_primary_vector_name(
                42,
                [("title".to_string(), FieldValue::String("doc".to_string()))],
                "embedding",
                vec![3.0, 4.0],
            ),
            9,
            true,
        );

        store.append(row).expect("append document row");
        let batch = store.record_batch().expect("record batch");
        let op_seq = batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("op seq");
        let is_deleted = batch
            .column(2)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("deleted");

        assert_eq!(op_seq.value(0), 9);
        assert!(is_deleted.value(0));
    }
}
