use std::io;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BooleanArray, FixedSizeListArray, Float32Array, Float64Array, Int32Array,
    Int64Array, RecordBatch, RecordBatchIterator, StringArray, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field};
use lance::dataset::{WriteMode, WriteParams};
use lance::Dataset;

use crate::document::{CollectionSchema, Document, FieldType, FieldValue};
use crate::query::{distance_by_metric, SearchHit};
use crate::storage::lance_schema::arrow_schema_for_lance;

pub struct LanceDatasetStore {
    uri: String,
    schema: CollectionSchema,
}

impl LanceDatasetStore {
    pub fn new(uri: impl Into<String>, schema: CollectionSchema) -> Self {
        Self {
            uri: uri.into(),
            schema,
        }
    }

    pub async fn create(&self, documents: &[Document]) -> io::Result<()> {
        self.write(documents, WriteMode::Create).await
    }

    pub async fn append(&self, documents: &[Document]) -> io::Result<()> {
        self.write(documents, WriteMode::Append).await
    }

    pub async fn open_lance(&self) -> io::Result<Dataset> {
        Dataset::open(self.uri.as_str()).await.map_err(lance_to_io)
    }

    pub async fn read_all_documents(&self) -> io::Result<Vec<Document>> {
        let dataset = self.open_lance().await?;
        let batch = dataset.scan().try_into_batch().await.map_err(lance_to_io)?;
        documents_from_lance_batch(&self.schema, &batch)
    }

    pub async fn fetch(&self, ids: &[i64]) -> io::Result<Vec<Document>> {
        let documents = self.read_all_documents().await?;
        Ok(ids
            .iter()
            .filter_map(|id| {
                documents
                    .iter()
                    .find(|document| document.id == *id)
                    .cloned()
            })
            .collect())
    }

    pub async fn search(
        &self,
        query: &[f32],
        top_k: usize,
        metric: &str,
    ) -> io::Result<Vec<SearchHit>> {
        let primary_vector = self.schema.primary_vector_name();
        let mut hits = Vec::new();
        for document in self.read_all_documents().await? {
            let vector = document.primary_vector_for(primary_vector).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "document {} is missing primary vector {primary_vector}",
                        document.id
                    ),
                )
            })?;
            hits.push(SearchHit {
                id: document.id,
                distance: distance_by_metric(query, vector, metric)?,
            });
        }
        hits.sort_by(|left, right| {
            left.distance
                .total_cmp(&right.distance)
                .then_with(|| left.id.cmp(&right.id))
        });
        if hits.len() > top_k {
            hits.truncate(top_k);
        }
        Ok(hits)
    }

    async fn write(&self, documents: &[Document], mode: WriteMode) -> io::Result<()> {
        let batch = documents_to_lance_batch(&self.schema, documents)?;
        let schema = batch.schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let params = WriteParams {
            mode,
            ..WriteParams::default()
        };
        Dataset::write(reader, self.uri.as_str(), Some(params))
            .await
            .map(|_| ())
            .map_err(lance_to_io)
    }
}

pub fn documents_from_lance_batch(
    schema: &CollectionSchema,
    batch: &RecordBatch,
) -> io::Result<Vec<Document>> {
    let id_column = batch.column_by_name("id").ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "Lance batch missing id column")
    })?;
    let ids = id_column
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "Lance id column is not Int64")
        })?;

    let mut documents = Vec::with_capacity(batch.num_rows());
    for row_idx in 0..batch.num_rows() {
        let mut fields = Vec::with_capacity(schema.fields.len());
        for scalar in &schema.fields {
            let column = batch.column_by_name(&scalar.name).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Lance batch missing scalar column {}", scalar.name),
                )
            })?;
            fields.push((
                scalar.name.clone(),
                field_value_from_column(column, row_idx, &scalar.data_type)?,
            ));
        }

        let mut vectors = Vec::with_capacity(schema.vectors.len());
        for vector_schema in &schema.vectors {
            let column = batch.column_by_name(&vector_schema.name).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Lance batch missing vector column {}", vector_schema.name),
                )
            })?;
            vectors.push((
                vector_schema.name.clone(),
                vector_from_column(column, row_idx, vector_schema.dimension)?,
            ));
        }

        documents.push(Document {
            id: ids.value(row_idx),
            fields: fields.into_iter().collect(),
            vectors: vectors.into_iter().collect(),
            sparse_vectors: Default::default(),
        });
    }
    Ok(documents)
}

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

fn field_value_from_column(
    column: &ArrayRef,
    row_idx: usize,
    data_type: &FieldType,
) -> io::Result<FieldValue> {
    if column.is_null(row_idx) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "null Lance values are not supported by Lance storage P1",
        ));
    }
    match data_type {
        FieldType::String => column
            .as_any()
            .downcast_ref::<StringArray>()
            .map(|array| FieldValue::String(array.value(row_idx).to_string()))
            .ok_or_else(|| column_type_error("String")),
        FieldType::Int64 => column
            .as_any()
            .downcast_ref::<Int64Array>()
            .map(|array| FieldValue::Int64(array.value(row_idx)))
            .ok_or_else(|| column_type_error("Int64")),
        FieldType::Int32 => column
            .as_any()
            .downcast_ref::<Int32Array>()
            .map(|array| FieldValue::Int32(array.value(row_idx)))
            .ok_or_else(|| column_type_error("Int32")),
        FieldType::UInt32 => column
            .as_any()
            .downcast_ref::<UInt32Array>()
            .map(|array| FieldValue::UInt32(array.value(row_idx)))
            .ok_or_else(|| column_type_error("UInt32")),
        FieldType::UInt64 => column
            .as_any()
            .downcast_ref::<UInt64Array>()
            .map(|array| FieldValue::UInt64(array.value(row_idx)))
            .ok_or_else(|| column_type_error("UInt64")),
        FieldType::Float => column
            .as_any()
            .downcast_ref::<Float32Array>()
            .map(|array| FieldValue::Float(array.value(row_idx)))
            .ok_or_else(|| column_type_error("Float")),
        FieldType::Float64 => column
            .as_any()
            .downcast_ref::<Float64Array>()
            .map(|array| FieldValue::Float64(array.value(row_idx)))
            .ok_or_else(|| column_type_error("Float64")),
        FieldType::Bool => column
            .as_any()
            .downcast_ref::<BooleanArray>()
            .map(|array| FieldValue::Bool(array.value(row_idx)))
            .ok_or_else(|| column_type_error("Bool")),
        FieldType::VectorFp32 | FieldType::VectorFp16 | FieldType::VectorSparse => {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "vector field type cannot be decoded as scalar column",
            ))
        }
    }
}

fn vector_from_column(column: &ArrayRef, row_idx: usize, dimension: usize) -> io::Result<Vec<f32>> {
    if column.is_null(row_idx) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "null Lance vector values are not supported by Lance storage P1",
        ));
    }
    let list = column
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| column_type_error("FixedSizeList<Float32>"))?;
    let values = list.value(row_idx);
    let values = values
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| column_type_error("Float32 vector values"))?;
    let vector = values.values().to_vec();
    if vector.len() != dimension {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Lance vector dimension mismatch: expected {dimension}, got {}",
                vector.len()
            ),
        ));
    }
    Ok(vector)
}

fn column_type_error(expected: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("Lance column type mismatch: expected {expected}"),
    )
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

fn lance_to_io(err: lance::Error) -> io::Error {
    io::Error::new(io::ErrorKind::Other, err.to_string())
}
