use std::collections::BTreeMap;
use std::fs::File;
use std::io;
use std::path::Path;
use std::sync::Arc;

use arrow::array::builder::*;
use arrow::array::*;
use arrow::buffer::NullBuffer;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::reader::FileReader;
use arrow::ipc::writer::FileWriter;
use arrow::record_batch::RecordBatch;

use crate::document::{FieldType, FieldValue, ScalarFieldSchema, VectorFieldSchema};

// ---------------------------------------------------------------------------
// Payloads: write
// ---------------------------------------------------------------------------

pub fn write_payloads_arrow(
    path: &Path,
    payloads: &[BTreeMap<String, FieldValue>],
    field_schemas: &[ScalarFieldSchema],
) -> io::Result<()> {
    if payloads.is_empty() {
        // Write an empty file so the .arrow sentinel exists.
        let batch = safe_empty_batch(&Schema::empty());
        return write_batch(path, &batch);
    }

    // Discover ad-hoc field names not in the declared schema.
    let declared_names: std::collections::HashSet<&str> =
        field_schemas.iter().map(|f| f.name.as_str()).collect();
    let mut ad_hoc_names: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for row in payloads {
        for key in row.keys() {
            if !declared_names.contains(key.as_str()) && !seen.contains(key.as_str()) {
                ad_hoc_names.push(key.clone());
                seen.insert(key.as_str());
            }
        }
    }

    // If no declared fields and no ad-hoc fields, write a sentinel file.
    if field_schemas.is_empty() && ad_hoc_names.is_empty() {
        let batch = safe_empty_batch(&Schema::empty());
        return write_batch(path, &batch);
    }

    let schema = build_payload_schema(field_schemas, &ad_hoc_names);
    let batch = build_payload_batch(payloads, &schema, field_schemas, &ad_hoc_names)?;
    write_batch(path, &batch)
}

// ---------------------------------------------------------------------------
// Payloads: read
// ---------------------------------------------------------------------------

pub fn load_payloads_arrow(path: &Path) -> io::Result<Vec<BTreeMap<String, FieldValue>>> {
    load_payloads_arrow_with_projection(path, None)
}

/// Load payloads from Arrow IPC, optionally projecting only the specified fields.
/// When `fields` is `Some`, only those columns are read from the file.
/// When `fields` is `None`, all columns are read.
pub fn load_payloads_arrow_with_projection(
    path: &Path,
    fields: Option<&[String]>,
) -> io::Result<Vec<BTreeMap<String, FieldValue>>> {
    if is_zero_byte_arrow_sentinel(path)? {
        return Ok(vec![
            BTreeMap::new();
            infer_segment_row_count_from_ids(path)?
        ]);
    }

    let file = File::open(path)?;
    // Arrow FileReader projection uses column indices. We need to read the
    // schema first to map field names to indices.
    let metadata_reader = FileReader::try_new(file, None).map_err(arrow_to_io)?;
    let schema = metadata_reader.schema();
    if is_empty_arrow_schema(schema.as_ref()) {
        return Ok(vec![
            BTreeMap::new();
            infer_segment_row_count_from_ids(path)?
        ]);
    }
    let projection: Option<Vec<usize>> = fields.map(|names| {
        let name_set: std::collections::HashSet<&str> = names.iter().map(String::as_str).collect();
        schema
            .fields()
            .iter()
            .enumerate()
            .filter(|(_, f)| name_set.contains(f.name().as_str()))
            .map(|(i, _)| i)
            .collect()
    });

    // Re-open the file to create a fresh reader with projection.
    let file = File::open(path)?;
    let reader = FileReader::try_new(file, projection).map_err(arrow_to_io)?;
    let mut payloads = Vec::new();
    for batch_result in reader {
        let batch = batch_result.map_err(arrow_to_io)?;
        let num_rows = batch.num_rows();
        let schema = batch.schema();
        for row in 0..num_rows {
            let mut map = BTreeMap::new();
            for (col_idx, field) in schema.fields().iter().enumerate() {
                let column = batch.column(col_idx);
                if column.is_null(row) {
                    continue;
                }
                if let Some(value) = arrow_value_to_field_value(column, row, field.data_type()) {
                    map.insert(field.name().clone(), value);
                }
            }
            payloads.push(map);
        }
    }
    Ok(payloads)
}

// ---------------------------------------------------------------------------
// Vectors: write
// ---------------------------------------------------------------------------

pub fn write_vectors_arrow(
    path: &Path,
    vectors: &[BTreeMap<String, Vec<f32>>],
    vector_fields: &[VectorFieldSchema],
    primary_vector_name: &str,
) -> io::Result<()> {
    let secondary: Vec<&VectorFieldSchema> = vector_fields
        .iter()
        .filter(|v| v.name != primary_vector_name)
        .collect();

    if secondary.is_empty() || vectors.is_empty() {
        let batch = safe_empty_batch(&Schema::empty());
        return write_batch(path, &batch);
    }

    let num_rows = vectors.len();
    let mut fields: Vec<Field> = Vec::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();

    for vf in &secondary {
        let dim = vf.dimension as i32;
        let dt =
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), dim);
        fields.push(Field::new(&vf.name, dt, true));

        let mut values = Float32Builder::new();
        let mut validity = BooleanBufferBuilder::new(num_rows);

        for row in vectors {
            match row.get(&vf.name) {
                Some(vec) => {
                    values.append_slice(vec);
                    validity.append(true);
                }
                None => {
                    values.append_slice(&vec![0.0f32; vf.dimension]);
                    validity.append(false);
                }
            }
        }

        let values_array = values.finish();
        let null_buf = NullBuffer::new(validity.finish());
        let list = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            dim,
            Arc::new(values_array),
            Some(null_buf),
        )
        .map_err(arrow_to_io)?;

        arrays.push(Arc::new(list));
    }

    let schema = Schema::new(fields);
    let batch = RecordBatch::try_new(Arc::new(schema), arrays).map_err(arrow_to_io)?;
    write_batch(path, &batch)
}

// ---------------------------------------------------------------------------
// Vectors: read
// ---------------------------------------------------------------------------

pub fn load_vectors_arrow(path: &Path) -> io::Result<Vec<BTreeMap<String, Vec<f32>>>> {
    if is_zero_byte_arrow_sentinel(path)? {
        return Ok(vec![
            BTreeMap::new();
            infer_segment_row_count_from_ids(path)?
        ]);
    }

    let file = File::open(path)?;
    let reader = FileReader::try_new(file, None).map_err(arrow_to_io)?;
    if is_empty_arrow_schema(reader.schema().as_ref()) {
        return Ok(vec![
            BTreeMap::new();
            infer_segment_row_count_from_ids(path)?
        ]);
    }
    let mut vectors = Vec::new();
    for batch_result in reader {
        let batch = batch_result.map_err(arrow_to_io)?;
        let num_rows = batch.num_rows();
        let schema = batch.schema();
        if schema.fields().is_empty() {
            // Empty schema → rows have no secondary vectors.
            for _ in 0..num_rows {
                vectors.push(BTreeMap::new());
            }
            continue;
        }
        for row in 0..num_rows {
            let mut map = BTreeMap::new();
            for (col_idx, field) in schema.fields().iter().enumerate() {
                let column = batch.column(col_idx);
                if column.is_null(row) {
                    continue;
                }
                if let DataType::FixedSizeList(_, _) = field.data_type() {
                    let list = column
                        .as_any()
                        .downcast_ref::<FixedSizeListArray>()
                        .expect("expected FixedSizeListArray");
                    let values = list.value(row);
                    let float_arr = values
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .expect("expected Float32Array");
                    let vec: Vec<f32> = float_arr.values().to_vec();
                    map.insert(field.name().clone(), vec);
                }
            }
            vectors.push(map);
        }
    }
    Ok(vectors)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn write_batch(path: &Path, batch: &RecordBatch) -> io::Result<()> {
    let file = File::create(path)?;
    let schema = batch.schema();
    // Arrow IPC requires at least one column for FileWriter. If the schema
    // is empty, write a sentinel file so the .arrow path exists.
    if is_empty_arrow_schema(schema.as_ref()) {
        // Just write an empty file as a marker.
        return Ok(());
    }
    let mut writer = FileWriter::try_new(file, schema.as_ref()).map_err(arrow_to_io)?;
    writer.write(batch).map_err(arrow_to_io)?;
    writer.finish().map_err(arrow_to_io)?;
    Ok(())
}

fn is_zero_byte_arrow_sentinel(path: &Path) -> io::Result<bool> {
    Ok(std::fs::metadata(path)?.len() == 0)
}

fn is_empty_arrow_schema(schema: &Schema) -> bool {
    schema.fields().len() == 1
        && schema.fields()[0].name() == "_dummy"
        && matches!(schema.fields()[0].data_type(), DataType::Null)
}

fn infer_segment_row_count_from_ids(arrow_path: &Path) -> io::Result<usize> {
    let ids_path = arrow_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("ids.bin");
    let bytes = std::fs::read(ids_path)?;
    if bytes.len() % 8 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "ids.bin byte length is not aligned to i64 row ids",
        ));
    }
    Ok(bytes.len() / 8)
}

fn build_payload_schema(field_schemas: &[ScalarFieldSchema], ad_hoc_names: &[String]) -> Schema {
    let mut fields: Vec<Field> = field_schemas
        .iter()
        .map(|fs| {
            if fs.array {
                Field::new(&fs.name, field_type_to_arrow_list(&fs.data_type), true)
            } else {
                Field::new(&fs.name, field_type_to_arrow(&fs.data_type), true)
            }
        })
        .collect();
    for name in ad_hoc_names {
        fields.push(Field::new(name, DataType::Utf8, true));
    }
    Schema::new(fields)
}

fn build_payload_batch(
    payloads: &[BTreeMap<String, FieldValue>],
    _schema: &Schema,
    field_schemas: &[ScalarFieldSchema],
    ad_hoc_names: &[String],
) -> io::Result<RecordBatch> {
    let schema = build_payload_schema(field_schemas, ad_hoc_names);
    let num_rows = payloads.len();
    let mut arrays: Vec<ArrayRef> = Vec::new();

    for fs in field_schemas {
        let arr = if fs.array {
            build_array_column(payloads, &fs.name, &fs.data_type, num_rows)
        } else {
            build_typed_column(payloads, &fs.name, &fs.data_type, num_rows)
        };
        arrays.push(arr);
    }
    for name in ad_hoc_names {
        let mut builder = StringBuilder::new();
        for row in payloads {
            match row.get(name) {
                Some(value) => builder.append_value(serde_json::to_string(value).unwrap()),
                None => builder.append_null(),
            }
        }
        arrays.push(Arc::new(builder.finish()));
    }

    RecordBatch::try_new(Arc::new(schema), arrays).map_err(arrow_to_io)
}

/// Create an empty RecordBatch with at least a dummy column if the schema is empty.
/// This is needed because Arrow's RecordBatch::new_empty panics with zero columns.
fn safe_empty_batch(schema: &Schema) -> RecordBatch {
    if schema.fields().is_empty() {
        let dummy_schema = Schema::new(vec![Field::new("_dummy", DataType::Null, true)]);
        RecordBatch::new_empty(Arc::new(dummy_schema))
    } else {
        RecordBatch::new_empty(Arc::new(schema.clone()))
    }
}

fn build_typed_column(
    payloads: &[BTreeMap<String, FieldValue>],
    name: &str,
    dt: &FieldType,
    _num_rows: usize,
) -> ArrayRef {
    match dt {
        FieldType::String => {
            let mut b = StringBuilder::new();
            for row in payloads {
                match row.get(name) {
                    Some(FieldValue::String(s)) => b.append_value(s),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        FieldType::Int64 => {
            let mut b = Int64Builder::new();
            for row in payloads {
                match row.get(name) {
                    Some(FieldValue::Int64(v)) => b.append_value(*v),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        FieldType::Int32 => {
            let mut b = Int32Builder::new();
            for row in payloads {
                match row.get(name) {
                    Some(FieldValue::Int32(v)) => b.append_value(*v),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        FieldType::UInt32 => {
            let mut b = UInt32Builder::new();
            for row in payloads {
                match row.get(name) {
                    Some(FieldValue::UInt32(v)) => b.append_value(*v),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        FieldType::UInt64 => {
            let mut b = UInt64Builder::new();
            for row in payloads {
                match row.get(name) {
                    Some(FieldValue::UInt64(v)) => b.append_value(*v),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        FieldType::Float => {
            let mut b = Float32Builder::new();
            for row in payloads {
                match row.get(name) {
                    Some(FieldValue::Float(v)) => b.append_value(*v),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        FieldType::Float64 => {
            let mut b = Float64Builder::new();
            for row in payloads {
                match row.get(name) {
                    Some(FieldValue::Float64(v)) => b.append_value(*v),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        FieldType::Bool => {
            let mut b = BooleanBuilder::new();
            for row in payloads {
                match row.get(name) {
                    Some(FieldValue::Bool(v)) => b.append_value(*v),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        FieldType::VectorFp32 | FieldType::VectorFp16 | FieldType::VectorSparse => {
            // Scalar column should never have vector type, write nulls.
            let mut b = NullBuilder::new();
            for _ in payloads {
                b.append_null();
            }
            Arc::new(b.finish())
        }
    }
}

/// Build a ListArray column for an array-typed field.
fn build_array_column(
    payloads: &[BTreeMap<String, FieldValue>],
    name: &str,
    element_type: &FieldType,
    _num_rows: usize,
) -> ArrayRef {
    match element_type {
        FieldType::String => build_string_array_column(payloads, name),
        FieldType::Int64 => build_int64_array_column(payloads, name),
        FieldType::Int32 => build_int32_array_column(payloads, name),
        FieldType::UInt32 => build_uint32_array_column(payloads, name),
        FieldType::UInt64 => build_uint64_array_column(payloads, name),
        FieldType::Float => build_float32_array_column(payloads, name),
        FieldType::Float64 => build_float64_array_column(payloads, name),
        FieldType::Bool => build_bool_array_column(payloads, name),
        FieldType::VectorFp32 | FieldType::VectorFp16 | FieldType::VectorSparse => {
            let mut b = NullBuilder::new();
            for _ in payloads {
                b.append_null();
            }
            Arc::new(b.finish())
        }
    }
}

fn build_string_array_column(payloads: &[BTreeMap<String, FieldValue>], name: &str) -> ArrayRef {
    let mut builder = ListBuilder::new(StringBuilder::new());
    for row in payloads {
        match row.get(name) {
            Some(FieldValue::Array(items)) => {
                let inner = builder.values();
                for item in items {
                    match item {
                        FieldValue::String(s) => inner.append_value(s),
                        _ => inner.append_null(),
                    }
                }
                builder.append(true);
            }
            _ => builder.append(false),
        }
    }
    Arc::new(builder.finish())
}

fn build_int64_array_column(payloads: &[BTreeMap<String, FieldValue>], name: &str) -> ArrayRef {
    let mut builder = ListBuilder::new(Int64Builder::new());
    for row in payloads {
        match row.get(name) {
            Some(FieldValue::Array(items)) => {
                let inner = builder.values();
                for item in items {
                    match item {
                        FieldValue::Int64(v) => inner.append_value(*v),
                        FieldValue::Int32(v) => inner.append_value(*v as i64),
                        FieldValue::UInt32(v) => inner.append_value(*v as i64),
                        FieldValue::UInt64(v) => inner.append_value(*v as i64),
                        _ => inner.append_null(),
                    }
                }
                builder.append(true);
            }
            _ => builder.append(false),
        }
    }
    Arc::new(builder.finish())
}

fn build_int32_array_column(payloads: &[BTreeMap<String, FieldValue>], name: &str) -> ArrayRef {
    let mut builder = ListBuilder::new(Int32Builder::new());
    for row in payloads {
        match row.get(name) {
            Some(FieldValue::Array(items)) => {
                let inner = builder.values();
                for item in items {
                    match item {
                        FieldValue::Int32(v) => inner.append_value(*v),
                        _ => inner.append_null(),
                    }
                }
                builder.append(true);
            }
            _ => builder.append(false),
        }
    }
    Arc::new(builder.finish())
}

fn build_uint32_array_column(payloads: &[BTreeMap<String, FieldValue>], name: &str) -> ArrayRef {
    let mut builder = ListBuilder::new(UInt32Builder::new());
    for row in payloads {
        match row.get(name) {
            Some(FieldValue::Array(items)) => {
                let inner = builder.values();
                for item in items {
                    match item {
                        FieldValue::UInt32(v) => inner.append_value(*v),
                        _ => inner.append_null(),
                    }
                }
                builder.append(true);
            }
            _ => builder.append(false),
        }
    }
    Arc::new(builder.finish())
}

fn build_uint64_array_column(payloads: &[BTreeMap<String, FieldValue>], name: &str) -> ArrayRef {
    let mut builder = ListBuilder::new(UInt64Builder::new());
    for row in payloads {
        match row.get(name) {
            Some(FieldValue::Array(items)) => {
                let inner = builder.values();
                for item in items {
                    match item {
                        FieldValue::UInt64(v) => inner.append_value(*v),
                        _ => inner.append_null(),
                    }
                }
                builder.append(true);
            }
            _ => builder.append(false),
        }
    }
    Arc::new(builder.finish())
}

fn build_float32_array_column(payloads: &[BTreeMap<String, FieldValue>], name: &str) -> ArrayRef {
    let mut builder = ListBuilder::new(Float32Builder::new());
    for row in payloads {
        match row.get(name) {
            Some(FieldValue::Array(items)) => {
                let inner = builder.values();
                for item in items {
                    match item {
                        FieldValue::Float(v) => inner.append_value(*v),
                        _ => inner.append_null(),
                    }
                }
                builder.append(true);
            }
            _ => builder.append(false),
        }
    }
    Arc::new(builder.finish())
}

fn build_float64_array_column(payloads: &[BTreeMap<String, FieldValue>], name: &str) -> ArrayRef {
    let mut builder = ListBuilder::new(Float64Builder::new());
    for row in payloads {
        match row.get(name) {
            Some(FieldValue::Array(items)) => {
                let inner = builder.values();
                for item in items {
                    match item {
                        FieldValue::Float64(v) => inner.append_value(*v),
                        _ => inner.append_null(),
                    }
                }
                builder.append(true);
            }
            _ => builder.append(false),
        }
    }
    Arc::new(builder.finish())
}

fn build_bool_array_column(payloads: &[BTreeMap<String, FieldValue>], name: &str) -> ArrayRef {
    let mut builder = ListBuilder::new(BooleanBuilder::new());
    for row in payloads {
        match row.get(name) {
            Some(FieldValue::Array(items)) => {
                let inner = builder.values();
                for item in items {
                    match item {
                        FieldValue::Bool(v) => inner.append_value(*v),
                        _ => inner.append_null(),
                    }
                }
                builder.append(true);
            }
            _ => builder.append(false),
        }
    }
    Arc::new(builder.finish())
}

fn field_type_to_arrow(ft: &FieldType) -> DataType {
    match ft {
        FieldType::String => DataType::Utf8,
        FieldType::Int64 => DataType::Int64,
        FieldType::Int32 => DataType::Int32,
        FieldType::UInt32 => DataType::UInt32,
        FieldType::UInt64 => DataType::UInt64,
        FieldType::Float => DataType::Float32,
        FieldType::Float64 => DataType::Float64,
        FieldType::Bool => DataType::Boolean,
        FieldType::VectorFp32 | FieldType::VectorFp16 | FieldType::VectorSparse => DataType::Utf8, // fallback
    }
}

fn field_type_to_arrow_list(ft: &FieldType) -> DataType {
    DataType::List(Arc::new(Field::new("item", field_type_to_arrow(ft), true)))
}

fn arrow_value_to_field_value(column: &ArrayRef, row: usize, dt: &DataType) -> Option<FieldValue> {
    match dt {
        DataType::Utf8 => {
            let arr = column.as_any().downcast_ref::<StringArray>()?;
            Some(FieldValue::String(arr.value(row).to_string()))
        }
        DataType::Int64 => {
            let arr = column.as_any().downcast_ref::<Int64Array>()?;
            Some(FieldValue::Int64(arr.value(row)))
        }
        DataType::Int32 => {
            let arr = column.as_any().downcast_ref::<Int32Array>()?;
            Some(FieldValue::Int32(arr.value(row)))
        }
        DataType::UInt32 => {
            let arr = column.as_any().downcast_ref::<UInt32Array>()?;
            Some(FieldValue::UInt32(arr.value(row)))
        }
        DataType::UInt64 => {
            let arr = column.as_any().downcast_ref::<UInt64Array>()?;
            Some(FieldValue::UInt64(arr.value(row)))
        }
        DataType::Float32 => {
            let arr = column.as_any().downcast_ref::<Float32Array>()?;
            Some(FieldValue::Float(arr.value(row)))
        }
        DataType::Float64 => {
            let arr = column.as_any().downcast_ref::<Float64Array>()?;
            Some(FieldValue::Float64(arr.value(row)))
        }
        DataType::Boolean => {
            let arr = column.as_any().downcast_ref::<BooleanArray>()?;
            Some(FieldValue::Bool(arr.value(row)))
        }
        DataType::List(ref field) => {
            let list = column.as_any().downcast_ref::<ListArray>()?;
            let inner = list.value(row);
            let items = arrow_array_to_field_values(&inner, field.data_type())?;
            Some(FieldValue::Array(items))
        }
        _ => None,
    }
}

fn arrow_array_to_field_values(array: &ArrayRef, dt: &DataType) -> Option<Vec<FieldValue>> {
    let mut values = Vec::new();
    for i in 0..array.len() {
        if array.is_null(i) {
            continue;
        }
        values.push(arrow_value_to_field_value(&array.clone(), i, dt)?);
    }
    Some(values)
}

fn arrow_to_io(err: arrow::error::ArrowError) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, err)
}
