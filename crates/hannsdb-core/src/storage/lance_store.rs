use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{
    builder::{
        BooleanBuilder, Float32Builder, Float64Builder, Int32Builder, Int64Builder, ListBuilder,
        StringBuilder, UInt32Builder, UInt64Builder,
    },
    Array, ArrayRef, BooleanArray, FixedSizeListArray, Float32Array, Float64Array, Int32Array,
    Int64Array, ListArray, RecordBatch, RecordBatchIterator, RecordBatchReader, StringArray,
    UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field};
#[cfg(feature = "hanns-backend")]
use hannsdb_index::descriptor::{VectorIndexDescriptor, VectorIndexKind};
#[cfg(feature = "hanns-backend")]
use hannsdb_index::factory::DefaultIndexFactory;
use lance::dataset::{MergeInsertBuilder, WhenMatched, WhenNotMatched, WriteMode, WriteParams};
use lance::Dataset;
#[cfg(feature = "hanns-backend")]
use serde::{Deserialize, Serialize};

use crate::document::{CollectionSchema, Document, FieldType, FieldValue};
use crate::query::{distance_by_metric, FilterExpr, SearchHit};
use crate::storage::lance_schema::{arrow_schema_for_lance, collection_schema_from_lance_arrow};

pub struct LanceCollection {
    name: String,
    store: LanceDatasetStore,
}

impl LanceCollection {
    pub async fn create(
        root: impl AsRef<Path>,
        name: impl Into<String>,
        schema: CollectionSchema,
        documents: &[Document],
    ) -> io::Result<Self> {
        let name = name.into();
        let uri = lance_collection_uri(root.as_ref(), &name);
        if let Some(parent) = uri.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let store = LanceDatasetStore::new(path_to_uri(&uri)?, schema);
        store.create(documents).await?;
        Ok(Self { name, store })
    }

    pub async fn open(
        root: impl AsRef<Path>,
        name: impl Into<String>,
        schema: CollectionSchema,
    ) -> io::Result<Self> {
        let name = name.into();
        let uri = lance_collection_uri(root.as_ref(), &name);
        let store = LanceDatasetStore::new(path_to_uri(&uri)?, schema);
        let _ = store.open_lance().await?;
        Ok(Self { name, store })
    }

    pub async fn open_inferred(
        root: impl AsRef<Path>,
        name: impl Into<String>,
    ) -> io::Result<Self> {
        let name = name.into();
        let uri = lance_collection_uri(root.as_ref(), &name);
        let store = LanceDatasetStore::open_inferred(path_to_uri(&uri)?).await?;
        Ok(Self { name, store })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn uri(&self) -> &str {
        self.store.uri()
    }

    pub fn schema(&self) -> &CollectionSchema {
        self.store.schema()
    }

    #[cfg(feature = "hanns-backend")]
    pub fn hanns_index_path(&self, field_name: &str) -> PathBuf {
        self.store.hanns_index_path(field_name)
    }

    #[cfg(feature = "hanns-backend")]
    pub async fn optimize_hanns(&self, field_name: &str, metric: &str) -> io::Result<()> {
        self.store.optimize_hanns(field_name, metric).await
    }

    pub async fn insert_documents(&self, documents: &[Document]) -> io::Result<()> {
        self.store.append(documents).await
    }

    pub async fn delete_documents(&self, ids: &[i64]) -> io::Result<usize> {
        self.store.delete(ids).await
    }

    pub async fn delete_by_filter(&self, filter: &FilterExpr) -> io::Result<usize> {
        self.store.delete_by_filter(filter).await
    }

    pub async fn upsert_documents(&self, documents: &[Document]) -> io::Result<usize> {
        self.store.upsert(documents).await
    }

    pub async fn fetch_documents(&self, ids: &[i64]) -> io::Result<Vec<Document>> {
        self.store.fetch(ids).await
    }

    pub async fn count_rows(&self) -> io::Result<usize> {
        self.store.count_rows().await
    }

    pub async fn search(
        &self,
        query: &[f32],
        top_k: usize,
        metric: &str,
    ) -> io::Result<Vec<SearchHit>> {
        self.store.search(query, top_k, metric).await
    }

    pub async fn search_vector_field(
        &self,
        field_name: &str,
        query: &[f32],
        top_k: usize,
        metric: &str,
    ) -> io::Result<Vec<SearchHit>> {
        self.store
            .search_vector_field(field_name, query, top_k, metric)
            .await
    }

    pub async fn search_vector_field_filtered(
        &self,
        field_name: &str,
        query: &[f32],
        top_k: usize,
        metric: &str,
        filter: Option<&FilterExpr>,
    ) -> io::Result<Vec<SearchHit>> {
        self.store
            .search_vector_field_filtered(field_name, query, top_k, metric, filter)
            .await
    }

    pub async fn search_filtered(
        &self,
        query: &[f32],
        top_k: usize,
        metric: &str,
        filter: Option<&FilterExpr>,
    ) -> io::Result<Vec<SearchHit>> {
        self.store
            .search_filtered(query, top_k, metric, filter)
            .await
    }
}

fn lance_collection_uri(root: &Path, name: &str) -> PathBuf {
    root.join("collections").join(format!("{name}.lance"))
}

fn path_to_uri(path: &Path) -> io::Result<String> {
    path.to_str()
        .map(str::to_string)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Lance path is not valid UTF-8"))
}

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

    pub fn uri(&self) -> &str {
        self.uri.as_str()
    }

    pub fn schema(&self) -> &CollectionSchema {
        &self.schema
    }

    pub async fn open_inferred(uri: impl Into<String>) -> io::Result<Self> {
        let uri = uri.into();
        let dataset = Dataset::open(uri.as_str()).await.map_err(lance_to_io)?;
        let arrow_schema: arrow_schema::Schema = dataset.schema().into();
        let schema = collection_schema_from_lance_arrow(&arrow_schema)?;
        Ok(Self { uri, schema })
    }

    #[cfg(feature = "hanns-backend")]
    pub fn hanns_index_path(&self, field_name: &str) -> PathBuf {
        self.hanns_index_dir().join(format!("{field_name}.hanns"))
    }

    #[cfg(feature = "hanns-backend")]
    pub async fn optimize_hanns(&self, field_name: &str, metric: &str) -> io::Result<()> {
        let vector_schema = self
            .schema
            .vectors
            .iter()
            .find(|vector| vector.name == field_name)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("vector field not found: {field_name}"),
                )
            })?;
        if !matches!(vector_schema.data_type, FieldType::VectorFp32) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Hanns sidecar supports only dense f32 vectors: {field_name}"),
            ));
        }

        let documents = self.read_all_documents().await?;
        let mut ids = Vec::with_capacity(documents.len());
        let mut flat_vectors =
            Vec::with_capacity(documents.len().saturating_mul(vector_schema.dimension));
        for document in documents {
            let id = u64::try_from(document.id).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("Hanns sidecar cannot index negative id {}", document.id),
                )
            })?;
            let vector = document.vectors.get(field_name).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "document {} is missing vector field {field_name}",
                        document.id
                    ),
                )
            })?;
            if vector.len() != vector_schema.dimension {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "document {} vector {field_name} dimension mismatch: expected {}, got {}",
                        document.id,
                        vector_schema.dimension,
                        vector.len()
                    ),
                ));
            }
            ids.push(id);
            flat_vectors.extend_from_slice(vector);
        }

        let descriptor = hanns_hnsw_descriptor(field_name, metric);
        let mut backend = DefaultIndexFactory::default()
            .create_vector_index(vector_schema.dimension, &descriptor, None)
            .map_err(adapter_error_to_io)?;
        if !flat_vectors.is_empty() {
            backend
                .insert_flat(&ids, &flat_vectors, vector_schema.dimension)
                .map_err(adapter_error_to_io)?;
        }
        let bytes = backend.serialize_to_bytes().map_err(adapter_error_to_io)?;
        let bytes = bytes.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "selected Hanns backend cannot serialize sidecar index",
            )
        })?;

        std::fs::create_dir_all(self.hanns_index_dir())?;
        std::fs::write(self.hanns_index_path(field_name), bytes)?;
        let meta = HannsSidecarMetadata {
            field_name: field_name.to_string(),
            metric: metric.to_ascii_lowercase(),
            dimension: vector_schema.dimension,
            row_count: ids.len(),
        };
        std::fs::write(
            self.hanns_index_metadata_path(field_name),
            serde_json::to_vec_pretty(&meta)
                .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?,
        )?;
        Ok(())
    }

    pub async fn create(&self, documents: &[Document]) -> io::Result<()> {
        self.write(documents, WriteMode::Create).await
    }

    pub async fn append(&self, documents: &[Document]) -> io::Result<()> {
        self.write(documents, WriteMode::Append).await
    }

    pub async fn delete(&self, ids: &[i64]) -> io::Result<usize> {
        if ids.is_empty() {
            return Ok(0);
        }
        let mut dataset = self.open_lance().await?;
        let result = dataset
            .delete(id_predicate(ids).as_str())
            .await
            .map_err(lance_to_io)?;
        #[cfg(feature = "hanns-backend")]
        self.invalidate_hanns_sidecars()?;
        usize::try_from(result.num_deleted_rows).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("delete count exceeds usize: {}", result.num_deleted_rows),
            )
        })
    }

    pub async fn delete_by_filter(&self, filter: &FilterExpr) -> io::Result<usize> {
        let ids = self
            .read_all_documents()
            .await?
            .into_iter()
            .filter(|document| filter.matches(&document.fields))
            .map(|document| document.id)
            .collect::<Vec<_>>();
        self.delete(&ids).await
    }

    pub async fn upsert(&self, documents: &[Document]) -> io::Result<usize> {
        if documents.is_empty() {
            return Ok(0);
        }
        let dataset = Arc::new(self.open_lance().await?);
        let batch = documents_to_lance_batch(&self.schema, documents)?;
        let schema = batch.schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut builder =
            MergeInsertBuilder::try_new(dataset, vec!["id".to_string()]).map_err(lance_to_io)?;
        let job = builder
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .use_index(false)
            .try_build()
            .map_err(lance_to_io)?;
        let (_dataset, stats) = job
            .execute_reader(Box::new(reader) as Box<dyn RecordBatchReader + Send>)
            .await
            .map_err(lance_to_io)?;
        #[cfg(feature = "hanns-backend")]
        self.invalidate_hanns_sidecars()?;
        usize::try_from(stats.num_inserted_rows + stats.num_updated_rows)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "upsert count exceeds usize"))
    }

    pub async fn open_lance(&self) -> io::Result<Dataset> {
        Dataset::open(self.uri.as_str()).await.map_err(lance_to_io)
    }

    pub async fn count_rows(&self) -> io::Result<usize> {
        let dataset = self.open_lance().await?;
        let rows = dataset.count_rows(None).await.map_err(lance_to_io)?;
        usize::try_from(rows).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Lance row count exceeds usize: {rows}"),
            )
        })
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
        self.search_filtered(query, top_k, metric, None).await
    }

    pub async fn search_filtered(
        &self,
        query: &[f32],
        top_k: usize,
        metric: &str,
        filter: Option<&FilterExpr>,
    ) -> io::Result<Vec<SearchHit>> {
        self.search_vector_field_filtered(
            self.schema.primary_vector_name(),
            query,
            top_k,
            metric,
            filter,
        )
        .await
    }

    pub async fn search_vector_field(
        &self,
        field_name: &str,
        query: &[f32],
        top_k: usize,
        metric: &str,
    ) -> io::Result<Vec<SearchHit>> {
        self.search_vector_field_filtered(field_name, query, top_k, metric, None)
            .await
    }

    pub async fn search_vector_field_filtered(
        &self,
        field_name: &str,
        query: &[f32],
        top_k: usize,
        metric: &str,
        filter: Option<&FilterExpr>,
    ) -> io::Result<Vec<SearchHit>> {
        let vector_schema = self
            .schema
            .vectors
            .iter()
            .find(|vector| vector.name == field_name)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("vector field not found: {field_name}"),
                )
            })?;
        if query.len() != vector_schema.dimension {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "query vector dimension mismatch for field '{field_name}': expected {}, got {}",
                    vector_schema.dimension,
                    query.len()
                ),
            ));
        }

        #[cfg(feature = "hanns-backend")]
        if field_name == self.schema.primary_vector_name() && filter.is_none() {
            if let Some(hits) = self.search_hanns_sidecar(query, top_k, metric).await? {
                return Ok(hits);
            }
        }

        let mut hits = Vec::new();
        for document in self.read_all_documents().await? {
            if filter.is_some_and(|filter| !filter.matches(&document.fields)) {
                continue;
            }
            let vector = document.vectors.get(field_name).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "document {} is missing vector field {field_name}",
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

    #[cfg(feature = "hanns-backend")]
    async fn search_hanns_sidecar(
        &self,
        query: &[f32],
        top_k: usize,
        metric: &str,
    ) -> io::Result<Option<Vec<SearchHit>>> {
        let field_name = self.schema.primary_vector_name();
        let index_path = self.hanns_index_path(field_name);
        let metadata_path = self.hanns_index_metadata_path(field_name);
        if !index_path.exists() || !metadata_path.exists() {
            return Ok(None);
        }
        let metadata: HannsSidecarMetadata = serde_json::from_slice(&std::fs::read(metadata_path)?)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
        if metadata.metric != metric.to_ascii_lowercase() || metadata.field_name != field_name {
            return Ok(None);
        }
        let descriptor = hanns_hnsw_descriptor(field_name, metric);
        let bytes = std::fs::read(index_path)?;
        let backend = DefaultIndexFactory::default()
            .create_vector_index(metadata.dimension, &descriptor, Some(&bytes))
            .map_err(adapter_error_to_io)?;
        let hits = backend
            .search(query, top_k, 32)
            .map_err(adapter_error_to_io)?;
        Ok(Some(
            hits.into_iter()
                .filter_map(|hit| {
                    i64::try_from(hit.id).ok().map(|id| SearchHit {
                        id,
                        distance: public_distance(metadata.metric.as_str(), hit.distance),
                    })
                })
                .collect(),
        ))
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
            .map_err(lance_to_io)?;
        #[cfg(feature = "hanns-backend")]
        self.invalidate_hanns_sidecars()?;
        Ok(())
    }

    #[cfg(feature = "hanns-backend")]
    fn hanns_index_dir(&self) -> PathBuf {
        PathBuf::from(self.uri.as_str())
            .join("_hannsdb")
            .join("ann")
    }

    #[cfg(feature = "hanns-backend")]
    fn hanns_index_metadata_path(&self, field_name: &str) -> PathBuf {
        self.hanns_index_dir().join(format!("{field_name}.json"))
    }

    #[cfg(feature = "hanns-backend")]
    fn invalidate_hanns_sidecars(&self) -> io::Result<()> {
        let dir = self.hanns_index_dir();
        match std::fs::remove_dir_all(dir) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(err),
        }
    }
}

#[cfg(feature = "hanns-backend")]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct HannsSidecarMetadata {
    field_name: String,
    metric: String,
    dimension: usize,
    row_count: usize,
}

#[cfg(feature = "hanns-backend")]
fn hanns_hnsw_descriptor(field_name: &str, metric: &str) -> VectorIndexDescriptor {
    VectorIndexDescriptor {
        field_name: field_name.to_string(),
        kind: VectorIndexKind::Hnsw,
        metric: Some(metric.to_ascii_lowercase()),
        params: serde_json::json!({
            "m": 16,
            "ef_construction": 128,
        }),
    }
}

#[cfg(feature = "hanns-backend")]
fn adapter_error_to_io(err: hannsdb_index::adapter::AdapterError) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("{err:?}"))
}

#[cfg(feature = "hanns-backend")]
fn public_distance(metric: &str, backend_distance: f32) -> f32 {
    match metric {
        "ip" => -backend_distance,
        _ => backend_distance,
    }
}

fn id_predicate(ids: &[i64]) -> String {
    if ids.len() == 1 {
        format!("id = {}", ids[0])
    } else {
        let ids = ids
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        format!("id in ({ids})")
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
            if let Some(value) =
                field_value_from_column(column, row_idx, &scalar.data_type, scalar.array)?
            {
                fields.push((scalar.name.clone(), value));
            }
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
            scalar.array,
            scalar.nullable,
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
    is_array: bool,
    nullable: bool,
    documents: &[Document],
) -> io::Result<ArrayRef> {
    if is_array {
        return scalar_list_array_for_documents(field_name, data_type, nullable, documents);
    }
    match data_type {
        FieldType::String => Ok(Arc::new(StringArray::from(
            documents
                .iter()
                .map(
                    |document| match optional_field(document, field_name, nullable)? {
                        Some(FieldValue::String(value)) => Ok(Some(value.clone())),
                        Some(value) => type_mismatch(field_name, "String", value),
                        None => Ok(None),
                    },
                )
                .collect::<io::Result<Vec<_>>>()?,
        ))),
        FieldType::Int64 => Ok(Arc::new(Int64Array::from(
            documents
                .iter()
                .map(
                    |document| match optional_field(document, field_name, nullable)? {
                        Some(FieldValue::Int64(value)) => Ok(Some(*value)),
                        Some(value) => type_mismatch(field_name, "Int64", value),
                        None => Ok(None),
                    },
                )
                .collect::<io::Result<Vec<_>>>()?,
        ))),
        FieldType::Int32 => Ok(Arc::new(Int32Array::from(
            documents
                .iter()
                .map(
                    |document| match optional_field(document, field_name, nullable)? {
                        Some(FieldValue::Int32(value)) => Ok(Some(*value)),
                        Some(value) => type_mismatch(field_name, "Int32", value),
                        None => Ok(None),
                    },
                )
                .collect::<io::Result<Vec<_>>>()?,
        ))),
        FieldType::UInt32 => Ok(Arc::new(UInt32Array::from(
            documents
                .iter()
                .map(
                    |document| match optional_field(document, field_name, nullable)? {
                        Some(FieldValue::UInt32(value)) => Ok(Some(*value)),
                        Some(value) => type_mismatch(field_name, "UInt32", value),
                        None => Ok(None),
                    },
                )
                .collect::<io::Result<Vec<_>>>()?,
        ))),
        FieldType::UInt64 => Ok(Arc::new(UInt64Array::from(
            documents
                .iter()
                .map(
                    |document| match optional_field(document, field_name, nullable)? {
                        Some(FieldValue::UInt64(value)) => Ok(Some(*value)),
                        Some(value) => type_mismatch(field_name, "UInt64", value),
                        None => Ok(None),
                    },
                )
                .collect::<io::Result<Vec<_>>>()?,
        ))),
        FieldType::Float => Ok(Arc::new(Float32Array::from(
            documents
                .iter()
                .map(
                    |document| match optional_field(document, field_name, nullable)? {
                        Some(FieldValue::Float(value)) => Ok(Some(*value)),
                        Some(value) => type_mismatch(field_name, "Float", value),
                        None => Ok(None),
                    },
                )
                .collect::<io::Result<Vec<_>>>()?,
        ))),
        FieldType::Float64 => Ok(Arc::new(Float64Array::from(
            documents
                .iter()
                .map(
                    |document| match optional_field(document, field_name, nullable)? {
                        Some(FieldValue::Float64(value)) => Ok(Some(*value)),
                        Some(value) => type_mismatch(field_name, "Float64", value),
                        None => Ok(None),
                    },
                )
                .collect::<io::Result<Vec<_>>>()?,
        ))),
        FieldType::Bool => Ok(Arc::new(BooleanArray::from(
            documents
                .iter()
                .map(
                    |document| match optional_field(document, field_name, nullable)? {
                        Some(FieldValue::Bool(value)) => Ok(Some(*value)),
                        Some(value) => type_mismatch(field_name, "Bool", value),
                        None => Ok(None),
                    },
                )
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

fn scalar_list_array_for_documents(
    field_name: &str,
    data_type: &FieldType,
    nullable: bool,
    documents: &[Document],
) -> io::Result<ArrayRef> {
    match data_type {
        FieldType::String => string_list_array_for_documents(field_name, nullable, documents),
        FieldType::Int64 => int64_list_array_for_documents(field_name, nullable, documents),
        FieldType::Int32 => int32_list_array_for_documents(field_name, nullable, documents),
        FieldType::UInt32 => uint32_list_array_for_documents(field_name, nullable, documents),
        FieldType::UInt64 => uint64_list_array_for_documents(field_name, nullable, documents),
        FieldType::Float => float32_list_array_for_documents(field_name, nullable, documents),
        FieldType::Float64 => float64_list_array_for_documents(field_name, nullable, documents),
        FieldType::Bool => bool_list_array_for_documents(field_name, nullable, documents),
        unsupported => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("array scalar type is not supported by Lance storage yet: {unsupported:?}"),
        )),
    }
}

fn string_list_array_for_documents(
    field_name: &str,
    nullable: bool,
    documents: &[Document],
) -> io::Result<ArrayRef> {
    let mut builder = ListBuilder::new(StringBuilder::new());
    for document in documents {
        let Some(value) = optional_field(document, field_name, nullable)? else {
            builder.append(false);
            continue;
        };
        let FieldValue::Array(items) = value else {
            return type_mismatch(field_name, "Array<String>", value);
        };
        for item in items {
            match item {
                FieldValue::Null => builder.values().append_null(),
                FieldValue::String(value) => builder.values().append_value(value),
                value => return type_mismatch(field_name, "String array item", value),
            }
        }
        builder.append(true);
    }
    Ok(Arc::new(builder.finish()))
}

macro_rules! scalar_list_array_for_documents {
    ($fn_name:ident, $builder:ident, $variant:ident, $expected:literal, $item_expected:literal) => {
        fn $fn_name(
            field_name: &str,
            nullable: bool,
            documents: &[Document],
        ) -> io::Result<ArrayRef> {
            let mut builder = ListBuilder::new($builder::new());
            for document in documents {
                let Some(value) = optional_field(document, field_name, nullable)? else {
                    builder.append(false);
                    continue;
                };
                let FieldValue::Array(items) = value else {
                    return type_mismatch(field_name, $expected, value);
                };
                for item in items {
                    match item {
                        FieldValue::Null => builder.values().append_null(),
                        FieldValue::$variant(value) => builder.values().append_value(*value),
                        value => return type_mismatch(field_name, $item_expected, value),
                    }
                }
                builder.append(true);
            }
            Ok(Arc::new(builder.finish()))
        }
    };
}

scalar_list_array_for_documents!(
    int32_list_array_for_documents,
    Int32Builder,
    Int32,
    "Array<Int32>",
    "Int32 array item"
);

fn int64_list_array_for_documents(
    field_name: &str,
    nullable: bool,
    documents: &[Document],
) -> io::Result<ArrayRef> {
    let mut builder = ListBuilder::new(Int64Builder::new());
    for document in documents {
        let Some(value) = optional_field(document, field_name, nullable)? else {
            builder.append(false);
            continue;
        };
        let FieldValue::Array(items) = value else {
            return type_mismatch(field_name, "Array<Int64>", value);
        };
        for item in items {
            match item {
                FieldValue::Null => builder.values().append_null(),
                FieldValue::Int64(value) => builder.values().append_value(*value),
                value => return type_mismatch(field_name, "Int64 array item", value),
            }
        }
        builder.append(true);
    }
    Ok(Arc::new(builder.finish()))
}

scalar_list_array_for_documents!(
    uint32_list_array_for_documents,
    UInt32Builder,
    UInt32,
    "Array<UInt32>",
    "UInt32 array item"
);

scalar_list_array_for_documents!(
    uint64_list_array_for_documents,
    UInt64Builder,
    UInt64,
    "Array<UInt64>",
    "UInt64 array item"
);

scalar_list_array_for_documents!(
    float32_list_array_for_documents,
    Float32Builder,
    Float,
    "Array<Float>",
    "Float array item"
);

scalar_list_array_for_documents!(
    float64_list_array_for_documents,
    Float64Builder,
    Float64,
    "Array<Float64>",
    "Float64 array item"
);

scalar_list_array_for_documents!(
    bool_list_array_for_documents,
    BooleanBuilder,
    Bool,
    "Array<Bool>",
    "Bool array item"
);

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

fn optional_field<'a>(
    document: &'a Document,
    field_name: &str,
    nullable: bool,
) -> io::Result<Option<&'a FieldValue>> {
    match document.fields.get(field_name) {
        Some(value) => Ok(Some(value)),
        None if nullable => Ok(None),
        None => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "document {} is missing scalar field {field_name}",
                document.id
            ),
        )),
    }
}

fn field_value_from_column(
    column: &ArrayRef,
    row_idx: usize,
    data_type: &FieldType,
    is_array: bool,
) -> io::Result<Option<FieldValue>> {
    if column.is_null(row_idx) {
        return Ok(None);
    }
    if is_array {
        return array_field_value_from_column(column, row_idx, data_type).map(Some);
    }
    scalar_field_value_from_column(column, row_idx, data_type).map(Some)
}

fn scalar_field_value_from_column(
    column: &ArrayRef,
    row_idx: usize,
    data_type: &FieldType,
) -> io::Result<FieldValue> {
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

fn array_field_value_from_column(
    column: &ArrayRef,
    row_idx: usize,
    data_type: &FieldType,
) -> io::Result<FieldValue> {
    let list = column
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| column_type_error("List"))?;
    let values = list.value(row_idx);
    let mut items = Vec::new();
    for item_idx in 0..values.len() {
        if values.is_null(item_idx) {
            items.push(FieldValue::Null);
            continue;
        }
        items.push(scalar_field_value_from_column(
            &values, item_idx, data_type,
        )?);
    }
    Ok(FieldValue::Array(items))
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
