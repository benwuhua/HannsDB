use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::io;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, BooleanBufferBuilder, BooleanBuilder, FixedSizeListArray,
    Float32Array, Float32Builder, Float64Builder, Int32Builder, Int64Builder, StringArray,
    StringBuilder, UInt32Array, UInt32Builder, UInt64Array, UInt64Builder,
};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use serde::{Deserialize, Serialize};

use crate::document::{
    CollectionSchema, FieldType, FieldValue, ScalarFieldSchema, VectorFieldSchema,
};

pub const INTERNAL_ID_COLUMN: &str = "internal_id";
pub const OP_SEQ_COLUMN: &str = "op_seq";
pub const IS_DELETED_COLUMN: &str = "is_deleted";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForwardRow {
    pub internal_id: u64,
    pub op_seq: u64,
    pub is_deleted: bool,
    pub fields: BTreeMap<String, FieldValue>,
    pub vectors: BTreeMap<String, Vec<f32>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForwardSchema {
    collection: CollectionSchema,
}

impl ForwardSchema {
    pub fn new(collection: CollectionSchema) -> Self {
        Self { collection }
    }

    pub fn collection(&self) -> &CollectionSchema {
        &self.collection
    }

    pub fn scalar_fields(&self) -> &[ScalarFieldSchema] {
        &self.collection.fields
    }

    pub fn vector_fields(&self) -> &[VectorFieldSchema] {
        &self.collection.vectors
    }

    pub fn primary_vector_name(&self) -> &str {
        self.collection.primary_vector_name()
    }

    pub fn validate_row(&self, row: &ForwardRow) -> io::Result<()> {
        let declared_fields: HashSet<&str> =
            self.scalar_fields().iter().map(|field| field.name.as_str()).collect();
        let declared_vectors: BTreeMap<&str, &VectorFieldSchema> = self
            .vector_fields()
            .iter()
            .map(|field| (field.name.as_str(), field))
            .collect();

        for field_name in row.fields.keys() {
            if !declared_fields.contains(field_name.as_str()) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("undeclared forward-store field: {}", field_name),
                ));
            }
        }

        for vector_name in row.vectors.keys() {
            let vector_schema = declared_vectors.get(vector_name.as_str()).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("undeclared forward-store vector: {}", vector_name),
                )
            })?;
            if row.vectors[vector_name].len() != vector_schema.dimension {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "vector {} dimension mismatch: expected {}, got {}",
                        vector_name,
                        vector_schema.dimension,
                        row.vectors[vector_name].len()
                    ),
                ));
            }
        }

        if !row.is_deleted && !row.vectors.contains_key(self.primary_vector_name()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "forward-store row missing primary vector {}",
                    self.primary_vector_name()
                ),
            ));
        }

        Ok(())
    }

    pub fn to_record_batch(
        &self,
        rows: &[ForwardRow],
        columns: Option<&[&str]>,
    ) -> io::Result<RecordBatch> {
        let selected = selected_columns(columns);
        let mut fields = Vec::new();
        let mut arrays = Vec::<ArrayRef>::new();

        if includes(&selected, INTERNAL_ID_COLUMN) {
            fields.push(Field::new(INTERNAL_ID_COLUMN, DataType::UInt64, false));
            let mut builder = UInt64Builder::new();
            for row in rows {
                builder.append_value(row.internal_id);
            }
            arrays.push(Arc::new(builder.finish()));
        }

        if includes(&selected, OP_SEQ_COLUMN) {
            fields.push(Field::new(OP_SEQ_COLUMN, DataType::UInt64, false));
            let mut builder = UInt64Builder::new();
            for row in rows {
                builder.append_value(row.op_seq);
            }
            arrays.push(Arc::new(builder.finish()));
        }

        if includes(&selected, IS_DELETED_COLUMN) {
            fields.push(Field::new(IS_DELETED_COLUMN, DataType::Boolean, false));
            let mut builder = BooleanBuilder::new();
            for row in rows {
                builder.append_value(row.is_deleted);
            }
            arrays.push(Arc::new(builder.finish()));
        }

        for scalar in self.scalar_fields() {
            if !includes(&selected, scalar.name.as_str()) {
                continue;
            }
            fields.push(Field::new(
                &scalar.name,
                scalar_arrow_type(scalar),
                true,
            ));
            arrays.push(build_scalar_array(rows, scalar)?);
        }

        for vector in self.vector_fields() {
            if !includes(&selected, vector.name.as_str()) {
                continue;
            }
            fields.push(Field::new(
                &vector.name,
                vector_arrow_type(vector),
                true,
            ));
            arrays.push(build_vector_array(rows, vector)?);
        }

        RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).map_err(arrow_to_io)
    }

    pub fn rows_from_batch(&self, batch: &RecordBatch) -> io::Result<Vec<ForwardRow>> {
        let mut rows = Vec::with_capacity(batch.num_rows());

        for row_index in 0..batch.num_rows() {
            let mut row = ForwardRow {
                internal_id: 0,
                op_seq: 0,
                is_deleted: false,
                fields: BTreeMap::new(),
                vectors: BTreeMap::new(),
            };

            for (column_index, field) in batch.schema().fields().iter().enumerate() {
                let column = batch.column(column_index);
                match field.name().as_str() {
                    INTERNAL_ID_COLUMN => {
                        let values = column
                            .as_any()
                            .downcast_ref::<UInt64Array>()
                            .expect("internal_id should be UInt64");
                        row.internal_id = values.value(row_index);
                    }
                    OP_SEQ_COLUMN => {
                        let values = column
                            .as_any()
                            .downcast_ref::<UInt64Array>()
                            .expect("op_seq should be UInt64");
                        row.op_seq = values.value(row_index);
                    }
                    IS_DELETED_COLUMN => {
                        let values = column
                            .as_any()
                            .downcast_ref::<BooleanArray>()
                            .expect("is_deleted should be Boolean");
                        row.is_deleted = values.value(row_index);
                    }
                    name => {
                        if let Some(value) = arrow_value_to_field_value(column, row_index, field.data_type()) {
                            row.fields.insert(name.to_string(), value);
                            continue;
                        }

                        if let Some(vector) = arrow_value_to_vector(column, row_index) {
                            row.vectors.insert(name.to_string(), vector);
                        }
                    }
                }
            }

            self.validate_row(&row)?;
            rows.push(row);
        }

        Ok(rows)
    }
}

pub fn project_row(row: &ForwardRow, columns: Option<&[&str]>) -> ForwardRow {
    let selected = selected_columns(columns);
    let include_all = selected.is_none();
    let selected = selected.unwrap_or_default();

    let include_system = |name: &str| include_all || selected.contains(name);
    let include_dynamic = |name: &str| include_all || selected.contains(name);

    let mut projected = ForwardRow {
        internal_id: if include_system(INTERNAL_ID_COLUMN) {
            row.internal_id
        } else {
            0
        },
        op_seq: if include_system(OP_SEQ_COLUMN) { row.op_seq } else { 0 },
        is_deleted: if include_system(IS_DELETED_COLUMN) {
            row.is_deleted
        } else {
            false
        },
        fields: BTreeMap::new(),
        vectors: BTreeMap::new(),
    };

    for (name, value) in &row.fields {
        if include_dynamic(name) {
            projected.fields.insert(name.clone(), value.clone());
        }
    }
    for (name, vector) in &row.vectors {
        if include_dynamic(name) {
            projected.vectors.insert(name.clone(), vector.clone());
        }
    }
    projected
}

pub fn estimate_row_bytes(row: &ForwardRow) -> usize {
    let field_bytes = row
        .fields
        .iter()
        .map(|(name, value)| name.len() + estimate_field_value_bytes(value))
        .sum::<usize>();
    let vector_bytes = row
        .vectors
        .iter()
        .map(|(name, vector)| name.len() + vector.len() * std::mem::size_of::<f32>())
        .sum::<usize>();
    (3 * std::mem::size_of::<u64>()) + field_bytes + vector_bytes
}

fn estimate_field_value_bytes(value: &FieldValue) -> usize {
    match value {
        FieldValue::String(value) => value.len(),
        FieldValue::Int64(_) => std::mem::size_of::<i64>(),
        FieldValue::Int32(_) => std::mem::size_of::<i32>(),
        FieldValue::UInt32(_) => std::mem::size_of::<u32>(),
        FieldValue::UInt64(_) => std::mem::size_of::<u64>(),
        FieldValue::Float(_) => std::mem::size_of::<f32>(),
        FieldValue::Float64(_) => std::mem::size_of::<f64>(),
        FieldValue::Bool(_) => std::mem::size_of::<bool>(),
        FieldValue::Array(values) => values.iter().map(estimate_field_value_bytes).sum(),
    }
}

fn includes(selected: &Option<BTreeSet<String>>, column: &str) -> bool {
    selected.as_ref().map_or(true, |set| set.contains(column))
}

fn selected_columns(columns: Option<&[&str]>) -> Option<BTreeSet<String>> {
    columns.map(|columns| columns.iter().map(|column| (*column).to_string()).collect())
}

fn scalar_arrow_type(field: &ScalarFieldSchema) -> DataType {
    if field.array {
        DataType::List(Arc::new(Field::new(
            "item",
            base_scalar_arrow_type(&field.data_type),
            true,
        )))
    } else {
        base_scalar_arrow_type(&field.data_type)
    }
}

fn base_scalar_arrow_type(field_type: &FieldType) -> DataType {
    match field_type {
        FieldType::String => DataType::Utf8,
        FieldType::Int64 => DataType::Int64,
        FieldType::Int32 => DataType::Int32,
        FieldType::UInt32 => DataType::UInt32,
        FieldType::UInt64 => DataType::UInt64,
        FieldType::Float => DataType::Float32,
        FieldType::Float64 => DataType::Float64,
        FieldType::Bool => DataType::Boolean,
        unsupported => panic!("unsupported forward-store scalar type: {:?}", unsupported),
    }
}

fn vector_arrow_type(field: &VectorFieldSchema) -> DataType {
    DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float32, false)),
        field.dimension as i32,
    )
}

fn build_scalar_array(rows: &[ForwardRow], field: &ScalarFieldSchema) -> io::Result<ArrayRef> {
    if field.array {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!("array forward-store scalar fields not yet supported: {}", field.name),
        ));
    }

    let array: ArrayRef = match &field.data_type {
        FieldType::String => {
            let mut builder = StringBuilder::new();
            for row in rows {
                match row.fields.get(&field.name) {
                    Some(FieldValue::String(value)) => builder.append_value(value),
                    Some(_) => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("field {} had non-string value", field.name),
                        ))
                    }
                    None => builder.append_null(),
                }
            }
            Arc::new(builder.finish())
        }
        FieldType::Int64 => {
            let mut builder = Int64Builder::new();
            for row in rows {
                match row.fields.get(&field.name) {
                    Some(FieldValue::Int64(value)) => builder.append_value(*value),
                    Some(_) => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("field {} had non-int64 value", field.name),
                        ))
                    }
                    None => builder.append_null(),
                }
            }
            Arc::new(builder.finish())
        }
        FieldType::Int32 => {
            let mut builder = Int32Builder::new();
            for row in rows {
                match row.fields.get(&field.name) {
                    Some(FieldValue::Int32(value)) => builder.append_value(*value),
                    Some(_) => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("field {} had non-int32 value", field.name),
                        ))
                    }
                    None => builder.append_null(),
                }
            }
            Arc::new(builder.finish())
        }
        FieldType::UInt32 => {
            let mut builder = UInt32Builder::new();
            for row in rows {
                match row.fields.get(&field.name) {
                    Some(FieldValue::UInt32(value)) => builder.append_value(*value),
                    Some(_) => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("field {} had non-uint32 value", field.name),
                        ))
                    }
                    None => builder.append_null(),
                }
            }
            Arc::new(builder.finish())
        }
        FieldType::UInt64 => {
            let mut builder = UInt64Builder::new();
            for row in rows {
                match row.fields.get(&field.name) {
                    Some(FieldValue::UInt64(value)) => builder.append_value(*value),
                    Some(_) => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("field {} had non-uint64 value", field.name),
                        ))
                    }
                    None => builder.append_null(),
                }
            }
            Arc::new(builder.finish())
        }
        FieldType::Float => {
            let mut builder = Float32Builder::new();
            for row in rows {
                match row.fields.get(&field.name) {
                    Some(FieldValue::Float(value)) => builder.append_value(*value),
                    Some(_) => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("field {} had non-float value", field.name),
                        ))
                    }
                    None => builder.append_null(),
                }
            }
            Arc::new(builder.finish())
        }
        FieldType::Float64 => {
            let mut builder = Float64Builder::new();
            for row in rows {
                match row.fields.get(&field.name) {
                    Some(FieldValue::Float64(value)) => builder.append_value(*value),
                    Some(_) => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("field {} had non-float64 value", field.name),
                        ))
                    }
                    None => builder.append_null(),
                }
            }
            Arc::new(builder.finish())
        }
        FieldType::Bool => {
            let mut builder = BooleanBuilder::new();
            for row in rows {
                match row.fields.get(&field.name) {
                    Some(FieldValue::Bool(value)) => builder.append_value(*value),
                    Some(_) => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("field {} had non-bool value", field.name),
                        ))
                    }
                    None => builder.append_null(),
                }
            }
            Arc::new(builder.finish())
        }
        unsupported => {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("unsupported forward-store scalar type: {:?}", unsupported),
            ))
        }
    };

    Ok(array)
}

fn build_vector_array(rows: &[ForwardRow], field: &VectorFieldSchema) -> io::Result<ArrayRef> {
    let mut values = Float32Builder::new();
    let mut validity = BooleanBufferBuilder::new(rows.len());

    for row in rows {
        match row.vectors.get(&field.name) {
            Some(vector) => {
                if vector.len() != field.dimension {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!(
                            "vector {} dimension mismatch: expected {}, got {}",
                            field.name,
                            field.dimension,
                            vector.len()
                        ),
                    ));
                }
                values.append_slice(vector);
                validity.append(true);
            }
            None => {
                values.append_slice(&vec![0.0; field.dimension]);
                validity.append(false);
            }
        }
    }

    let values = values.finish();
    let nulls = NullBuffer::new(validity.finish());
    let array = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, false)),
        field.dimension as i32,
        Arc::new(values),
        Some(nulls),
    )
    .map_err(arrow_to_io)?;
    Ok(Arc::new(array))
}

fn arrow_value_to_field_value(
    column: &ArrayRef,
    row_index: usize,
    data_type: &DataType,
) -> Option<FieldValue> {
    if column.is_null(row_index) {
        return None;
    }

    match data_type {
        DataType::Utf8 => Some(FieldValue::String(
            column
                .as_any()
                .downcast_ref::<StringArray>()?
                .value(row_index)
                .to_string(),
        )),
        DataType::Int64 => Some(FieldValue::Int64(
            column
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()?
                .value(row_index),
        )),
        DataType::Int32 => Some(FieldValue::Int32(
            column
                .as_any()
                .downcast_ref::<arrow::array::Int32Array>()?
                .value(row_index),
        )),
        DataType::UInt32 => Some(FieldValue::UInt32(
            column
                .as_any()
                .downcast_ref::<UInt32Array>()?
                .value(row_index),
        )),
        DataType::UInt64 => Some(FieldValue::UInt64(
            column
                .as_any()
                .downcast_ref::<UInt64Array>()?
                .value(row_index),
        )),
        DataType::Float32 => Some(FieldValue::Float(
            column
                .as_any()
                .downcast_ref::<arrow::array::Float32Array>()?
                .value(row_index),
        )),
        DataType::Float64 => Some(FieldValue::Float64(
            column
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()?
                .value(row_index),
        )),
        DataType::Boolean => Some(FieldValue::Bool(
            column
                .as_any()
                .downcast_ref::<arrow::array::BooleanArray>()?
                .value(row_index),
        )),
        _ => None,
    }
}

fn arrow_value_to_vector(column: &ArrayRef, row_index: usize) -> Option<Vec<f32>> {
    if column.is_null(row_index) {
        return None;
    }

    let list = column.as_any().downcast_ref::<FixedSizeListArray>()?;
    let values = list.value(row_index);
    let values = values.as_any().downcast_ref::<Float32Array>()?;
    Some(values.values().to_vec())
}

fn arrow_to_io(err: arrow::error::ArrowError) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, err)
}
