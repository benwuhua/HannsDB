use std::io;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, BooleanArray, FixedSizeListArray, Float32Array, Float64Array, Int32Array, Int64Array,
    RecordBatch, StringArray, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field};

use crate::document::{CollectionSchema, Document, FieldType, FieldValue};
use crate::storage::lance_schema::arrow_schema_for_lance;

pub fn documents_to_lance_batch(
    schema: &CollectionSchema,
    documents: &[Document],
) -> io::Result<RecordBatch> {
    let arrow_schema = arrow_schema_for_lance(schema)?;
    let mut arrays = Vec::<ArrayRef>::with_capacity(arrow_schema.fields().len());

    arrays.push(Arc::new(Int64Array::from(
        documents
            .iter()
            .map(|document| document.id)
            .collect::<Vec<_>>(),
    )));

    for scalar in &schema.fields {
        arrays.push(scalar_array_for_documents(
            scalar.name.as_str(),
            &scalar.data_type,
            documents,
        )?);
    }

    for vector in &schema.vectors {
        arrays.push(vector_array_for_documents(
            vector.name.as_str(),
            vector.dimension,
            documents,
        )?);
    }

    RecordBatch::try_new(arrow_schema, arrays).map_err(arrow_to_io)
}

fn scalar_array_for_documents(
    field_name: &str,
    data_type: &FieldType,
    documents: &[Document],
) -> io::Result<ArrayRef> {
    match data_type {
        FieldType::String => Ok(Arc::new(StringArray::from(
            documents
                .iter()
                .map(|document| match required_field(document, field_name)? {
                    FieldValue::String(value) => Ok(value.clone()),
                    value => type_mismatch(field_name, "String", value),
                })
                .collect::<io::Result<Vec<_>>>()?,
        ))),
        FieldType::Int64 => Ok(Arc::new(Int64Array::from(
            documents
                .iter()
                .map(|document| match required_field(document, field_name)? {
                    FieldValue::Int64(value) => Ok(*value),
                    value => type_mismatch(field_name, "Int64", value),
                })
                .collect::<io::Result<Vec<_>>>()?,
        ))),
        FieldType::Int32 => Ok(Arc::new(Int32Array::from(
            documents
                .iter()
                .map(|document| match required_field(document, field_name)? {
                    FieldValue::Int32(value) => Ok(*value),
                    value => type_mismatch(field_name, "Int32", value),
                })
                .collect::<io::Result<Vec<_>>>()?,
        ))),
        FieldType::UInt32 => Ok(Arc::new(UInt32Array::from(
            documents
                .iter()
                .map(|document| match required_field(document, field_name)? {
                    FieldValue::UInt32(value) => Ok(*value),
                    value => type_mismatch(field_name, "UInt32", value),
                })
                .collect::<io::Result<Vec<_>>>()?,
        ))),
        FieldType::UInt64 => Ok(Arc::new(UInt64Array::from(
            documents
                .iter()
                .map(|document| match required_field(document, field_name)? {
                    FieldValue::UInt64(value) => Ok(*value),
                    value => type_mismatch(field_name, "UInt64", value),
                })
                .collect::<io::Result<Vec<_>>>()?,
        ))),
        FieldType::Float => Ok(Arc::new(Float32Array::from(
            documents
                .iter()
                .map(|document| match required_field(document, field_name)? {
                    FieldValue::Float(value) => Ok(*value),
                    value => type_mismatch(field_name, "Float", value),
                })
                .collect::<io::Result<Vec<_>>>()?,
        ))),
        FieldType::Float64 => Ok(Arc::new(Float64Array::from(
            documents
                .iter()
                .map(|document| match required_field(document, field_name)? {
                    FieldValue::Float64(value) => Ok(*value),
                    value => type_mismatch(field_name, "Float64", value),
                })
                .collect::<io::Result<Vec<_>>>()?,
        ))),
        FieldType::Bool => Ok(Arc::new(BooleanArray::from(
            documents
                .iter()
                .map(|document| match required_field(document, field_name)? {
                    FieldValue::Bool(value) => Ok(*value),
                    value => type_mismatch(field_name, "Bool", value),
                })
                .collect::<io::Result<Vec<_>>>()?,
        ))),
        FieldType::VectorFp32 | FieldType::VectorFp16 | FieldType::VectorSparse => {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("vector field type cannot be encoded as scalar column: {field_name}"),
            ))
        }
    }
}

fn vector_array_for_documents(
    vector_name: &str,
    dimension: usize,
    documents: &[Document],
) -> io::Result<ArrayRef> {
    let mut values = Vec::with_capacity(documents.len().saturating_mul(dimension));
    for document in documents {
        let vector = document.vectors.get(vector_name).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "document {} is missing vector field {vector_name}",
                    document.id
                ),
            )
        })?;
        if vector.len() != dimension {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "document {} vector {vector_name} dimension mismatch: expected {dimension}, got {}",
                    document.id,
                    vector.len()
                ),
            ));
        }
        values.extend_from_slice(vector);
    }

    let value_array: ArrayRef = Arc::new(Float32Array::from(values));
    FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, false)),
        i32::try_from(dimension).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("vector dimension exceeds i32 range: {vector_name}"),
            )
        })?,
        value_array,
        None,
    )
    .map(|array| Arc::new(array) as ArrayRef)
    .map_err(arrow_to_io)
}

fn required_field<'a>(document: &'a Document, field_name: &str) -> io::Result<&'a FieldValue> {
    document.fields.get(field_name).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "document {} is missing scalar field {field_name}",
                document.id
            ),
        )
    })
}

fn type_mismatch<T>(field_name: &str, expected: &str, actual: &FieldValue) -> io::Result<T> {
    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("field {field_name} expected {expected}, got {actual:?}"),
    ))
}

fn arrow_to_io(err: arrow_schema::ArrowError) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, err)
}
