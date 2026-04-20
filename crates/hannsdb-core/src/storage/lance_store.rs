use std::collections::BTreeSet;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{
    builder::{
        BooleanBuilder, Float32Builder, Float64Builder, Int32Builder, Int64Builder, ListBuilder,
        StringBuilder, UInt32Builder, UInt64Builder,
    },
    types::{Float32Type, UInt32Type},
    Array, ArrayRef, BooleanArray, FixedSizeListArray, Float32Array, Float64Array, Int32Array,
    Int64Array, ListArray, RecordBatch, RecordBatchIterator, RecordBatchReader, StringArray,
    StructArray, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
#[cfg(feature = "hanns-backend")]
use hannsdb_index::descriptor::{
    SparseIndexDescriptor, SparseIndexKind, VectorIndexDescriptor, VectorIndexKind,
};
#[cfg(feature = "hanns-backend")]
use hannsdb_index::factory::DefaultIndexFactory;
#[cfg(feature = "hanns-backend")]
use hannsdb_index::sparse::{SparseIndexBackend, SparseVectorData};
use lance::dataset::{
    ColumnAlteration, MergeInsertBuilder, NewColumnTransform, WhenMatched, WhenNotMatched,
    WriteMode, WriteParams,
};
use lance::Dataset;
use serde::{Deserialize, Serialize};

use crate::document::{CollectionSchema, Document, FieldType, FieldValue, SparseVector};
use crate::query::{distance_by_metric, ComparisonOp, FilterExpr, SearchHit};
use crate::storage::lance_schema::{
    arrow_schema_for_lance, collection_schema_from_lance_arrow, scalar_data_type_for_lance,
    sparse_vector_data_type_for_lance, LANCE_SPARSE_INDICES_FIELD, LANCE_SPARSE_VALUES_FIELD,
};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LanceScanObservation {
    pub predicate: Option<String>,
    pub projected_columns: Vec<String>,
    pub fallback_reason: Option<String>,
    pub sparse_index: Option<LanceSparseIndexObservation>,
}

impl LanceScanObservation {
    pub fn predicate_pushdown_used(&self) -> bool {
        self.predicate.is_some()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LanceSparseIndexObservation {
    pub field_name: String,
    pub metric: String,
    pub path: LanceSparseIndexPath,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LanceSparseIndexPath {
    Loaded,
    RebuiltMissing,
    RebuiltCorrupt,
}

#[derive(Debug, Clone, Default)]
pub struct LanceSearchProjection {
    pub output_fields: BTreeSet<String>,
    pub required_fields: BTreeSet<String>,
    pub required_vectors: BTreeSet<String>,
}

impl LanceSearchProjection {
    pub fn with_output_fields(fields: impl IntoIterator<Item = String>) -> Self {
        Self {
            output_fields: fields.into_iter().collect(),
            required_fields: BTreeSet::new(),
            required_vectors: BTreeSet::new(),
        }
    }

    pub fn required_field(mut self, field: impl Into<String>) -> Self {
        self.required_fields.insert(field.into());
        self
    }

    pub fn required_vector(mut self, field: impl Into<String>) -> Self {
        self.required_vectors.insert(field.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LanceSearchDocument {
    pub document: Document,
    pub distance: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LanceSearchResult {
    pub hits: Vec<SearchHit>,
    pub documents: Vec<LanceSearchDocument>,
    pub observation: LanceScanObservation,
}

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

    #[cfg(feature = "hanns-backend")]
    pub fn sparse_index_path(&self, field_name: &str) -> PathBuf {
        self.store.sparse_index_path(field_name)
    }

    #[cfg(feature = "hanns-backend")]
    pub async fn optimize_sparse(&self, field_name: &str, metric: &str) -> io::Result<()> {
        self.store.optimize_sparse(field_name, metric).await
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

    pub async fn search_vector_field_filtered_projected(
        &self,
        field_name: &str,
        query: &[f32],
        top_k: usize,
        metric: &str,
        filter: Option<&FilterExpr>,
        projection: LanceSearchProjection,
    ) -> io::Result<LanceSearchResult> {
        self.store
            .search_vector_field_filtered_projected(
                field_name, projection, query, top_k, metric, filter,
            )
            .await
    }

    #[cfg(feature = "hanns-backend")]
    pub async fn search_sparse_vector_field_projected(
        &self,
        field_name: &str,
        query: &SparseVector,
        top_k: usize,
        metric: &str,
        projection: LanceSearchProjection,
    ) -> io::Result<LanceSearchResult> {
        self.store
            .search_sparse_vector_field_projected(field_name, query, top_k, metric, projection)
            .await
    }

    #[cfg(feature = "hanns-backend")]
    pub async fn search_sparse_vector_field(
        &self,
        field_name: &str,
        query: &SparseVector,
        top_k: usize,
        metric: &str,
    ) -> io::Result<Vec<SearchHit>> {
        self.store
            .search_sparse_vector_field_projected(
                field_name,
                query,
                top_k,
                metric,
                LanceSearchProjection::default(),
            )
            .await
            .map(|result| result.hits)
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

    pub async fn add_scalar_column(
        &self,
        field: crate::document::ScalarFieldSchema,
    ) -> io::Result<()> {
        self.store.add_scalar_column(field).await
    }

    pub async fn drop_scalar_column(&self, field_name: &str) -> io::Result<()> {
        self.store.drop_scalar_column(field_name).await
    }

    pub async fn rename_scalar_column(&self, field_name: &str, new_name: &str) -> io::Result<()> {
        self.store.rename_scalar_column(field_name, new_name).await
    }

    pub async fn drop_vector_field(&self, field_name: &str) -> io::Result<()> {
        self.store.drop_vector_field(field_name).await
    }

    pub fn create_scalar_index_descriptor(
        &self,
        field_name: &str,
        kind: &str,
        params: serde_json::Value,
    ) -> io::Result<()> {
        self.store
            .create_scalar_index_descriptor(field_name, kind, params)
    }

    pub fn list_scalar_index_descriptors(&self) -> io::Result<Vec<LanceScalarIndexDescriptor>> {
        self.store.list_scalar_index_descriptors()
    }

    pub fn drop_scalar_index_descriptor(&self, field_name: &str) -> io::Result<()> {
        self.store.drop_scalar_index_descriptor(field_name)
    }

    #[cfg(feature = "hanns-backend")]
    pub fn list_hanns_indexes(&self) -> io::Result<Vec<LanceHannsIndexDescriptor>> {
        self.store.list_hanns_indexes()
    }

    #[cfg(feature = "hanns-backend")]
    pub fn drop_hanns_index(&self, field_name: &str) -> io::Result<()> {
        self.store.drop_hanns_index(field_name)
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
    pub fn sparse_index_path(&self, field_name: &str) -> PathBuf {
        self.sparse_index_dir().join(format!("{field_name}.sparse"))
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

    #[cfg(feature = "hanns-backend")]
    pub async fn optimize_sparse(&self, field_name: &str, metric: &str) -> io::Result<()> {
        let _ = self
            .build_sparse_sidecar(field_name, &metric.trim().to_ascii_lowercase())
            .await?;
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
        self.invalidate_hannsdb_vector_sidecars()?;
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
        self.invalidate_hannsdb_vector_sidecars()?;
        usize::try_from(stats.num_inserted_rows + stats.num_updated_rows)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "upsert count exceeds usize"))
    }

    pub async fn add_scalar_column(
        &self,
        field: crate::document::ScalarFieldSchema,
    ) -> io::Result<()> {
        if self
            .schema
            .fields
            .iter()
            .any(|existing| existing.name == field.name)
            || self
                .schema
                .vectors
                .iter()
                .any(|existing| existing.name == field.name)
            || field.name == crate::storage::lance_schema::LANCE_ID_COLUMN
        {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("field already exists: {}", field.name),
            ));
        }
        if !field.nullable {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Lance add column supports only nullable scalar columns for existing datasets",
            ));
        }
        let data_type = scalar_data_type_for_lance(&field.data_type, field.array)?;
        let mut dataset = self.open_lance().await?;
        dataset
            .add_columns(
                NewColumnTransform::AllNulls(Arc::new(ArrowSchema::new(vec![Field::new(
                    field.name.clone(),
                    data_type,
                    true,
                )]))),
                None,
                None,
            )
            .await
            .map_err(lance_to_io)?;
        #[cfg(feature = "hanns-backend")]
        self.invalidate_hannsdb_vector_sidecars()?;
        Ok(())
    }

    pub async fn drop_scalar_column(&self, field_name: &str) -> io::Result<()> {
        self.schema
            .fields
            .iter()
            .find(|field| field.name == field_name)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("scalar field not found: {field_name}"),
                )
            })?;
        self.remove_scalar_index_descriptor_if_exists(field_name)?;
        let mut dataset = self.open_lance().await?;
        dataset
            .drop_columns(&[field_name])
            .await
            .map_err(lance_to_io)?;
        #[cfg(feature = "hanns-backend")]
        self.invalidate_hannsdb_vector_sidecars()?;
        Ok(())
    }

    pub async fn rename_scalar_column(&self, field_name: &str, new_name: &str) -> io::Result<()> {
        if new_name.trim().is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "new column name must not be empty",
            ));
        }
        self.schema
            .fields
            .iter()
            .find(|field| field.name == field_name)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("scalar field not found: {field_name}"),
                )
            })?;
        if self
            .schema
            .fields
            .iter()
            .any(|field| field.name == new_name)
            || self
                .schema
                .vectors
                .iter()
                .any(|field| field.name == new_name)
            || new_name == crate::storage::lance_schema::LANCE_ID_COLUMN
        {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("field already exists: {new_name}"),
            ));
        }
        let mut dataset = self.open_lance().await?;
        dataset
            .alter_columns(&[
                ColumnAlteration::new(field_name.to_string()).rename(new_name.to_string())
            ])
            .await
            .map_err(lance_to_io)?;
        self.rename_scalar_index_descriptor_if_exists(field_name, new_name)?;
        #[cfg(feature = "hanns-backend")]
        self.invalidate_hannsdb_vector_sidecars()?;
        Ok(())
    }

    pub async fn drop_vector_field(&self, field_name: &str) -> io::Result<()> {
        self.schema
            .vectors
            .iter()
            .find(|field| field.name == field_name)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("vector field not found: {field_name}"),
                )
            })?;
        if field_name == self.schema.primary_vector_name() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Lance storage cannot drop the primary vector field",
            ));
        }
        let mut dataset = self.open_lance().await?;
        dataset
            .drop_columns(&[field_name])
            .await
            .map_err(lance_to_io)?;
        #[cfg(feature = "hanns-backend")]
        self.invalidate_hannsdb_vector_sidecars()?;
        Ok(())
    }

    pub fn create_scalar_index_descriptor(
        &self,
        field_name: &str,
        kind: &str,
        params: serde_json::Value,
    ) -> io::Result<()> {
        self.schema
            .fields
            .iter()
            .find(|field| field.name == field_name)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("scalar field not found: {field_name}"),
                )
            })?;
        let kind = kind.trim().to_ascii_lowercase();
        if kind != "inverted" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Lance scalar index DDL supports only inverted kind, got: {kind}"),
            ));
        }
        let mut descriptors = self.list_scalar_index_descriptors()?;
        if descriptors
            .iter()
            .any(|descriptor| descriptor.field_name == field_name)
        {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("scalar index already exists: {field_name}"),
            ));
        }
        descriptors.push(LanceScalarIndexDescriptor {
            field_name: field_name.to_string(),
            kind,
            params,
        });
        self.write_scalar_index_descriptors(&descriptors)
    }

    pub fn list_scalar_index_descriptors(&self) -> io::Result<Vec<LanceScalarIndexDescriptor>> {
        let path = self.scalar_index_descriptors_path();
        match std::fs::read(path) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err)),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(err) => Err(err),
        }
    }

    pub fn drop_scalar_index_descriptor(&self, field_name: &str) -> io::Result<()> {
        let mut descriptors = self.list_scalar_index_descriptors()?;
        let before = descriptors.len();
        descriptors.retain(|descriptor| descriptor.field_name != field_name);
        if descriptors.len() == before {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("scalar index not found: {field_name}"),
            ));
        }
        self.write_scalar_index_descriptors(&descriptors)
    }

    fn remove_scalar_index_descriptor_if_exists(&self, field_name: &str) -> io::Result<()> {
        let mut descriptors = self.list_scalar_index_descriptors()?;
        let before = descriptors.len();
        descriptors.retain(|descriptor| descriptor.field_name != field_name);
        if descriptors.len() != before {
            self.write_scalar_index_descriptors(&descriptors)?;
        }
        Ok(())
    }

    fn rename_scalar_index_descriptor_if_exists(
        &self,
        field_name: &str,
        new_name: &str,
    ) -> io::Result<()> {
        let mut descriptors = self.list_scalar_index_descriptors()?;
        let mut changed = false;
        for descriptor in &mut descriptors {
            if descriptor.field_name == field_name {
                descriptor.field_name = new_name.to_string();
                changed = true;
            }
        }
        if changed {
            self.write_scalar_index_descriptors(&descriptors)?;
        }
        Ok(())
    }

    fn write_scalar_index_descriptors(
        &self,
        descriptors: &[LanceScalarIndexDescriptor],
    ) -> io::Result<()> {
        let path = self.scalar_index_descriptors_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(
            path,
            serde_json::to_vec_pretty(descriptors)
                .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?,
        )
    }

    fn scalar_index_descriptors_path(&self) -> PathBuf {
        PathBuf::from(self.uri.as_str())
            .join("_hannsdb")
            .join("scalar_indexes.json")
    }

    #[cfg(feature = "hanns-backend")]
    pub fn list_hanns_indexes(&self) -> io::Result<Vec<LanceHannsIndexDescriptor>> {
        let mut descriptors = Vec::new();
        let dir = self.hanns_index_dir();
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(descriptors),
            Err(err) => return Err(err),
        };
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                continue;
            }
            let metadata: HannsSidecarMetadata = serde_json::from_slice(&std::fs::read(path)?)
                .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
            descriptors.push(LanceHannsIndexDescriptor {
                field_name: metadata.field_name,
                kind: "hnsw".to_string(),
                metric: metadata.metric,
                params: serde_json::json!({"m": 16, "ef_construction": 128}),
            });
        }
        descriptors.sort_by(|left, right| left.field_name.cmp(&right.field_name));
        Ok(descriptors)
    }

    #[cfg(feature = "hanns-backend")]
    pub fn drop_hanns_index(&self, field_name: &str) -> io::Result<()> {
        let index_path = self.hanns_index_path(field_name);
        let metadata_path = self.hanns_index_metadata_path(field_name);
        let existed = index_path.exists() || metadata_path.exists();
        match std::fs::remove_file(index_path) {
            Ok(()) => {}
            Err(err) if err.kind() == io::ErrorKind::NotFound => {}
            Err(err) => return Err(err),
        }
        match std::fs::remove_file(metadata_path) {
            Ok(()) => {}
            Err(err) if err.kind() == io::ErrorKind::NotFound => {}
            Err(err) => return Err(err),
        }
        if existed {
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("vector index not found: {field_name}"),
            ))
        }
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
        self.search_vector_field_filtered_projected(
            field_name,
            LanceSearchProjection::default(),
            query,
            top_k,
            metric,
            filter,
        )
        .await
        .map(|result| result.hits)
    }

    pub async fn search_vector_field_filtered_projected(
        &self,
        field_name: &str,
        projection: LanceSearchProjection,
        query: &[f32],
        top_k: usize,
        metric: &str,
        filter: Option<&FilterExpr>,
    ) -> io::Result<LanceSearchResult> {
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

        let translated_filter =
            filter.map(|filter| lance_predicate_for_filter(&self.schema, filter));
        let mut projected_columns = BTreeSet::new();
        projected_columns.insert(crate::storage::lance_schema::LANCE_ID_COLUMN.to_string());
        projected_columns.insert(field_name.to_string());
        projected_columns.extend(projection.output_fields.iter().cloned());
        projected_columns.extend(projection.required_fields.iter().cloned());
        projected_columns.extend(projection.required_vectors.iter().cloned());
        if translated_filter.as_ref().is_some_and(Result::is_err) {
            if let Some(filter) = filter {
                projected_columns.extend(filter.referenced_fields());
            }
        }

        let mut projected_columns = projected_columns.into_iter().collect::<Vec<_>>();
        projected_columns.sort();

        #[cfg(feature = "hanns-backend")]
        if field_name == self.schema.primary_vector_name() && filter.is_none() {
            if let Some(hits) = self.search_hanns_sidecar(query, top_k, metric).await? {
                let ids = hits.iter().map(|hit| hit.id).collect::<Vec<_>>();
                let scores = hits
                    .iter()
                    .map(|hit| (hit.id, hit.distance))
                    .collect::<std::collections::BTreeMap<_, _>>();
                let documents = self
                    .fetch(&ids)
                    .await?
                    .into_iter()
                    .map(|document| LanceSearchDocument {
                        distance: scores.get(&document.id).copied().unwrap_or_default(),
                        document,
                    })
                    .collect();
                return Ok(LanceSearchResult {
                    hits,
                    documents,
                    observation: LanceScanObservation {
                        predicate: None,
                        projected_columns,
                        fallback_reason: None,
                        sparse_index: None,
                    },
                });
            }
        }

        let dataset = self.open_lance().await?;
        let mut scanner = dataset.scan();
        scanner.project(&projected_columns).map_err(lance_to_io)?;
        let (predicate, fallback_reason) = match translated_filter {
            Some(Ok(predicate)) => {
                scanner.filter(predicate.as_str()).map_err(lance_to_io)?;
                (Some(predicate), None)
            }
            Some(Err(reason)) => (None, Some(reason)),
            None => (None, None),
        };
        let batch = scanner.try_into_batch().await.map_err(lance_to_io)?;
        let mut documents = documents_from_projected_lance_batch(&self.schema, &batch)?;

        if predicate.is_none() {
            if let Some(filter) = filter {
                documents.retain(|document| filter.matches(&document.fields));
            }
        }

        let mut scored = Vec::new();
        for document in documents {
            let vector = document.vectors.get(field_name).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "document {} is missing vector field {field_name}",
                        document.id
                    ),
                )
            })?;
            scored.push(LanceSearchDocument {
                distance: distance_by_metric(query, vector, metric)?,
                document,
            });
        }
        scored.sort_by(|left, right| {
            left.distance
                .total_cmp(&right.distance)
                .then_with(|| left.document.id.cmp(&right.document.id))
        });
        if scored.len() > top_k {
            scored.truncate(top_k);
        }
        let hits = scored
            .iter()
            .map(|document| SearchHit {
                id: document.document.id,
                distance: document.distance,
            })
            .collect();
        Ok(LanceSearchResult {
            hits,
            documents: scored,
            observation: LanceScanObservation {
                predicate,
                projected_columns,
                fallback_reason,
                sparse_index: None,
            },
        })
    }

    #[cfg(feature = "hanns-backend")]
    pub async fn search_sparse_vector_field_projected(
        &self,
        field_name: &str,
        query: &SparseVector,
        top_k: usize,
        metric: &str,
        projection: LanceSearchProjection,
    ) -> io::Result<LanceSearchResult> {
        let metric = metric.trim().to_ascii_lowercase();
        self.sparse_vector_schema(field_name)?;

        let mut projected_columns = BTreeSet::new();
        projected_columns.insert(crate::storage::lance_schema::LANCE_ID_COLUMN.to_string());
        projected_columns.insert(field_name.to_string());
        projected_columns.extend(projection.output_fields.iter().cloned());
        projected_columns.extend(projection.required_fields.iter().cloned());
        projected_columns.extend(projection.required_vectors.iter().cloned());
        let mut projected_columns = projected_columns.into_iter().collect::<Vec<_>>();
        projected_columns.sort();

        let (index, path) = self
            .load_or_rebuild_sparse_sidecar(field_name, &metric)
            .await?;
        let query_data = SparseVectorData::new(query.indices.clone(), query.values.clone());
        let hits = index
            .search(&query_data, top_k, None)
            .map_err(adapter_error_to_io)?;
        let hits = hits
            .into_iter()
            .map(|hit| SearchHit {
                id: hit.id,
                distance: -hit.score,
            })
            .collect::<Vec<_>>();
        let scores = hits
            .iter()
            .map(|hit| (hit.id, hit.distance))
            .collect::<std::collections::BTreeMap<_, _>>();
        let ids = hits.iter().map(|hit| hit.id).collect::<Vec<_>>();
        let documents = self
            .fetch(&ids)
            .await?
            .into_iter()
            .map(|document| LanceSearchDocument {
                distance: scores.get(&document.id).copied().unwrap_or_default(),
                document,
            })
            .collect();

        Ok(LanceSearchResult {
            hits,
            documents,
            observation: LanceScanObservation {
                predicate: None,
                projected_columns,
                fallback_reason: None,
                sparse_index: Some(LanceSparseIndexObservation {
                    field_name: field_name.to_string(),
                    metric,
                    path,
                }),
            },
        })
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

    #[cfg(feature = "hanns-backend")]
    async fn load_or_rebuild_sparse_sidecar(
        &self,
        field_name: &str,
        metric: &str,
    ) -> io::Result<(Box<dyn SparseIndexBackend>, LanceSparseIndexPath)> {
        match self.load_sparse_sidecar(field_name, metric) {
            Ok(Some(index)) => Ok((index, LanceSparseIndexPath::Loaded)),
            Ok(None) => self
                .build_sparse_sidecar(field_name, metric)
                .await
                .map(|index| (index, LanceSparseIndexPath::RebuiltMissing)),
            Err(err) if err.kind() == io::ErrorKind::InvalidData => {
                self.remove_sparse_sidecar(field_name)?;
                self.build_sparse_sidecar(field_name, metric)
                    .await
                    .map(|index| (index, LanceSparseIndexPath::RebuiltCorrupt))
            }
            Err(err) => Err(err),
        }
    }

    #[cfg(feature = "hanns-backend")]
    fn load_sparse_sidecar(
        &self,
        field_name: &str,
        metric: &str,
    ) -> io::Result<Option<Box<dyn SparseIndexBackend>>> {
        let index_path = self.sparse_index_path(field_name);
        let metadata_path = self.sparse_index_metadata_path(field_name);
        if !index_path.exists() || !metadata_path.exists() {
            return Ok(None);
        }
        let metadata: SparseSidecarMetadata =
            serde_json::from_slice(&std::fs::read(metadata_path)?)
                .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
        if metadata.field_name != field_name || metadata.metric != metric {
            return Ok(None);
        }
        let descriptor = sparse_index_descriptor(field_name, metric);
        let bytes = std::fs::read(index_path)?;
        DefaultIndexFactory::default()
            .create_sparse_index(&descriptor, Some(&bytes))
            .map(Some)
            .map_err(adapter_error_to_io)
    }

    #[cfg(feature = "hanns-backend")]
    async fn build_sparse_sidecar(
        &self,
        field_name: &str,
        metric: &str,
    ) -> io::Result<Box<dyn SparseIndexBackend>> {
        let vector_schema = self.sparse_vector_schema(field_name)?.clone();
        let documents = self.read_all_documents().await?;
        let sparse_rows = documents
            .iter()
            .filter_map(|document| {
                document.sparse_vectors.get(field_name).map(|sparse| {
                    (
                        document.id,
                        SparseVectorData::new(sparse.indices.clone(), sparse.values.clone()),
                    )
                })
            })
            .collect::<Vec<_>>();

        let descriptor = sparse_index_descriptor(field_name, metric);
        let mut index = DefaultIndexFactory::default()
            .create_sparse_index(&descriptor, None)
            .map_err(adapter_error_to_io)?;
        if !sparse_rows.is_empty() {
            index.add(&sparse_rows).map_err(adapter_error_to_io)?;
        }
        if let Some(bm25) = &vector_schema.bm25_params {
            index.set_bm25_params(bm25.k1, bm25.b, bm25.avgdl);
        }

        let bytes = index.serialize_to_bytes().map_err(adapter_error_to_io)?;
        let bytes = bytes.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "selected sparse backend cannot serialize sidecar index",
            )
        })?;
        std::fs::create_dir_all(self.sparse_index_dir())?;
        std::fs::write(self.sparse_index_path(field_name), bytes)?;
        let metadata = SparseSidecarMetadata {
            field_name: field_name.to_string(),
            metric: metric.to_string(),
            row_count: sparse_rows.len(),
        };
        std::fs::write(
            self.sparse_index_metadata_path(field_name),
            serde_json::to_vec_pretty(&metadata)
                .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?,
        )?;
        Ok(index)
    }

    #[cfg(feature = "hanns-backend")]
    fn sparse_vector_schema(
        &self,
        field_name: &str,
    ) -> io::Result<&crate::document::VectorFieldSchema> {
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
        if !matches!(vector_schema.data_type, FieldType::VectorSparse) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("sparse sidecar supports only sparse vector fields: {field_name}"),
            ));
        }
        Ok(vector_schema)
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
        self.invalidate_hannsdb_vector_sidecars()?;
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
    fn sparse_index_dir(&self) -> PathBuf {
        PathBuf::from(self.uri.as_str())
            .join("_hannsdb")
            .join("sparse")
    }

    #[cfg(feature = "hanns-backend")]
    fn sparse_index_metadata_path(&self, field_name: &str) -> PathBuf {
        self.sparse_index_dir().join(format!("{field_name}.json"))
    }

    #[cfg(feature = "hanns-backend")]
    fn remove_sparse_sidecar(&self, field_name: &str) -> io::Result<()> {
        for path in [
            self.sparse_index_path(field_name),
            self.sparse_index_metadata_path(field_name),
        ] {
            match std::fs::remove_file(path) {
                Ok(()) => {}
                Err(err) if err.kind() == io::ErrorKind::NotFound => {}
                Err(err) => return Err(err),
            }
        }
        Ok(())
    }

    #[cfg(feature = "hanns-backend")]
    fn invalidate_hannsdb_vector_sidecars(&self) -> io::Result<()> {
        self.remove_sidecar_dir(self.hanns_index_dir())?;
        self.remove_sidecar_dir(self.sparse_index_dir())
    }

    #[cfg(feature = "hanns-backend")]
    fn remove_sidecar_dir(&self, dir: PathBuf) -> io::Result<()> {
        match std::fs::remove_dir_all(dir) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(err),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LanceScalarIndexDescriptor {
    pub field_name: String,
    pub kind: String,
    #[serde(default)]
    pub params: serde_json::Value,
}

#[cfg(feature = "hanns-backend")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LanceHannsIndexDescriptor {
    pub field_name: String,
    pub kind: String,
    pub metric: String,
    #[serde(default)]
    pub params: serde_json::Value,
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct SparseSidecarMetadata {
    field_name: String,
    metric: String,
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
fn sparse_index_descriptor(field_name: &str, metric: &str) -> SparseIndexDescriptor {
    SparseIndexDescriptor {
        field_name: field_name.to_string(),
        kind: SparseIndexKind::SparseInverted,
        metric: Some(metric.to_ascii_lowercase()),
        params: serde_json::Value::Object(serde_json::Map::new()),
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

fn lance_predicate_for_filter(
    schema: &CollectionSchema,
    filter: &FilterExpr,
) -> Result<String, String> {
    match filter {
        FilterExpr::And(exprs) => {
            if exprs.is_empty() {
                return Err("empty and expression cannot be pushed down".to_string());
            }
            let parts = exprs
                .iter()
                .map(|expr| {
                    lance_predicate_for_filter(schema, expr).map(|part| format!("({part})"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(parts.join(" AND "))
        }
        FilterExpr::Clause { field, op, value } => {
            if matches!(op, ComparisonOp::Ne) {
                return Err("!= is not in the initial Lance pushdown subset".to_string());
            }
            let scalar = schema
                .fields
                .iter()
                .find(|scalar| scalar.name == *field)
                .ok_or_else(|| format!("field '{field}' is not a Lance scalar field"))?;
            if scalar.array {
                return Err(format!("array field '{field}' is not pushdown-safe"));
            }
            if !is_safe_lance_identifier(field) {
                return Err(format!(
                    "field '{field}' is not a safe Lance predicate identifier"
                ));
            }
            let literal = lance_literal_for_field_value(&scalar.data_type, value)?;
            let op = match op {
                ComparisonOp::Eq => "=",
                ComparisonOp::Gt => ">",
                ComparisonOp::Gte => ">=",
                ComparisonOp::Lt => "<",
                ComparisonOp::Lte => "<=",
                ComparisonOp::Ne => unreachable!("handled above"),
            };
            Ok(format!("{field} {op} {literal}"))
        }
        FilterExpr::Or(_) => Err("or expressions require fallback in this slice".to_string()),
        FilterExpr::Not(_) => Err("not expressions require fallback in this slice".to_string()),
        FilterExpr::InList { .. } => {
            Err("in-list filters require fallback in this slice".to_string())
        }
        FilterExpr::NullCheck { .. } => {
            Err("null checks require fallback in this slice".to_string())
        }
        FilterExpr::Like { .. } => Err("like filters require fallback in this slice".to_string()),
        FilterExpr::HasPrefix { .. } => {
            Err("has_prefix filters require fallback in this slice".to_string())
        }
        FilterExpr::HasSuffix { .. } => {
            Err("has_suffix filters require fallback in this slice".to_string())
        }
        FilterExpr::ArrayContains { .. }
        | FilterExpr::ArrayContainsAny { .. }
        | FilterExpr::ArrayContainsAll { .. } => {
            Err("array filters require fallback in this slice".to_string())
        }
    }
}

fn is_safe_lance_identifier(field: &str) -> bool {
    let mut chars = field.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first == '_' || first.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

fn lance_literal_for_field_value(
    data_type: &FieldType,
    value: &FieldValue,
) -> Result<String, String> {
    match (data_type, value) {
        (FieldType::String, FieldValue::String(value)) => {
            Ok(format!("'{}'", value.replace('\'', "''")))
        }
        (FieldType::Bool, FieldValue::Bool(value)) => Ok(value.to_string()),
        (FieldType::Int64, FieldValue::Int64(value)) => Ok(value.to_string()),
        (FieldType::Int64, FieldValue::Int32(value)) => Ok(value.to_string()),
        (FieldType::Int32, FieldValue::Int32(value)) => Ok(value.to_string()),
        (FieldType::UInt32, FieldValue::UInt32(value)) => Ok(value.to_string()),
        (FieldType::UInt64, FieldValue::UInt64(value)) => Ok(value.to_string()),
        (FieldType::UInt64, FieldValue::UInt32(value)) => Ok(value.to_string()),
        (FieldType::Float, FieldValue::Float(value)) => Ok(value.to_string()),
        (FieldType::Float, FieldValue::Float64(value)) if value.is_finite() => {
            Ok(value.to_string())
        }
        (FieldType::Float, FieldValue::Int32(value)) => Ok(value.to_string()),
        (FieldType::Float, FieldValue::Int64(value)) => Ok(value.to_string()),
        (FieldType::Float64, FieldValue::Float64(value)) if value.is_finite() => {
            Ok(value.to_string())
        }
        (FieldType::Float64, FieldValue::Float(value)) if value.is_finite() => {
            Ok(value.to_string())
        }
        (FieldType::Float64, FieldValue::Int32(value)) => Ok(value.to_string()),
        (FieldType::Float64, FieldValue::Int64(value)) => Ok(value.to_string()),
        _ => Err(format!(
            "literal {value:?} is not compatible with Lance field type {data_type:?}"
        )),
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

        let mut vectors = Vec::new();
        let mut sparse_vectors = std::collections::BTreeMap::new();
        for vector_schema in &schema.vectors {
            let column = batch.column_by_name(&vector_schema.name).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Lance batch missing vector column {}", vector_schema.name),
                )
            })?;
            match vector_schema.data_type {
                FieldType::VectorFp32 => {
                    vectors.push((
                        vector_schema.name.clone(),
                        vector_from_column(column, row_idx, vector_schema.dimension)?,
                    ));
                }
                FieldType::VectorSparse => {
                    if let Some(sparse) = sparse_vector_from_column(column, row_idx)? {
                        sparse_vectors.insert(vector_schema.name.clone(), sparse);
                    }
                }
                FieldType::VectorFp16 => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "fp16 vector field {} is not supported by Lance storage",
                            vector_schema.name
                        ),
                    ));
                }
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "non-vector field registered as vector: {}",
                            vector_schema.name
                        ),
                    ));
                }
            }
        }

        documents.push(Document {
            id: ids.value(row_idx),
            fields: fields.into_iter().collect(),
            vectors: vectors.into_iter().collect(),
            sparse_vectors,
        });
    }
    Ok(documents)
}

fn documents_from_projected_lance_batch(
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
        let mut fields = Vec::new();
        for scalar in &schema.fields {
            let Some(column) = batch.column_by_name(&scalar.name) else {
                continue;
            };
            if let Some(value) =
                field_value_from_column(column, row_idx, &scalar.data_type, scalar.array)?
            {
                fields.push((scalar.name.clone(), value));
            }
        }

        let mut vectors = Vec::new();
        let mut sparse_vectors = std::collections::BTreeMap::new();
        for vector_schema in &schema.vectors {
            let Some(column) = batch.column_by_name(&vector_schema.name) else {
                continue;
            };
            match vector_schema.data_type {
                FieldType::VectorFp32 => {
                    vectors.push((
                        vector_schema.name.clone(),
                        vector_from_column(column, row_idx, vector_schema.dimension)?,
                    ));
                }
                FieldType::VectorSparse => {
                    if let Some(sparse) = sparse_vector_from_column(column, row_idx)? {
                        sparse_vectors.insert(vector_schema.name.clone(), sparse);
                    }
                }
                FieldType::VectorFp16 => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "fp16 vector field {} is not supported by Lance storage",
                            vector_schema.name
                        ),
                    ));
                }
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "non-vector field registered as vector: {}",
                            vector_schema.name
                        ),
                    ));
                }
            }
        }

        documents.push(Document {
            id: ids.value(row_idx),
            fields: fields.into_iter().collect(),
            vectors: vectors.into_iter().collect(),
            sparse_vectors,
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
        match vector.data_type {
            FieldType::VectorFp32 => arrays.push(vector_array_for_documents(
                vector.name.as_str(),
                vector.dimension,
                documents,
            )?),
            FieldType::VectorSparse => arrays.push(sparse_vector_array_for_documents(
                vector.name.as_str(),
                documents,
            )?),
            FieldType::VectorFp16 => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "fp16 vectors are not supported by Lance storage: {}",
                        vector.name
                    ),
                ));
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("non-vector field registered as vector: {}", vector.name),
                ));
            }
        }
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

fn sparse_vector_array_for_documents(
    vector_name: &str,
    documents: &[Document],
) -> io::Result<ArrayRef> {
    let sparse_rows = documents
        .iter()
        .map(|document| {
            document
                .sparse_vectors
                .get(vector_name)
                .map(|sparse| validate_sparse_vector(document.id, vector_name, sparse))
                .transpose()
        })
        .collect::<io::Result<Vec<_>>>()?;

    let indices_rows = sparse_rows.iter().map(|sparse| {
        sparse.as_ref().map(|sparse| {
            sparse
                .indices
                .iter()
                .copied()
                .map(Some)
                .collect::<Vec<Option<u32>>>()
        })
    });
    let values_rows = sparse_rows.iter().map(|sparse| {
        sparse.as_ref().map(|sparse| {
            sparse
                .values
                .iter()
                .copied()
                .map(Some)
                .collect::<Vec<Option<f32>>>()
        })
    });

    let indices_array: ArrayRef = Arc::new(ListArray::from_iter_primitive::<UInt32Type, _, _>(
        indices_rows,
    ));
    let values_array: ArrayRef = Arc::new(ListArray::from_iter_primitive::<Float32Type, _, _>(
        values_rows,
    ));
    let DataType::Struct(fields) = sparse_vector_data_type_for_lance() else {
        unreachable!("sparse vector data type must be a struct");
    };
    StructArray::try_new(fields, vec![indices_array, values_array], None)
        .map(|array| Arc::new(array) as ArrayRef)
        .map_err(arrow_to_io)
}

fn validate_sparse_vector(
    document_id: i64,
    vector_name: &str,
    sparse: &SparseVector,
) -> io::Result<SparseVector> {
    if sparse.indices.len() != sparse.values.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "document {document_id} sparse vector {vector_name} has mismatched indices/values lengths: {} != {}",
                sparse.indices.len(),
                sparse.values.len()
            ),
        ));
    }
    if !sparse.is_sorted() && !sparse.indices.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "document {document_id} sparse vector {vector_name} indices must be strictly sorted"
            ),
        ));
    }
    Ok(sparse.clone())
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

fn sparse_vector_from_column(
    column: &ArrayRef,
    row_idx: usize,
) -> io::Result<Option<SparseVector>> {
    let sparse = column
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| column_type_error("Struct<indices: List<UInt32>, values: List<Float32>>"))?;
    let indices_column = sparse
        .column_by_name(LANCE_SPARSE_INDICES_FIELD)
        .ok_or_else(|| column_type_error("sparse vector indices field"))?;
    let values_column = sparse
        .column_by_name(LANCE_SPARSE_VALUES_FIELD)
        .ok_or_else(|| column_type_error("sparse vector values field"))?;
    let indices_list = indices_column
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| column_type_error("List<UInt32> sparse indices"))?;
    let values_list = values_column
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| column_type_error("List<Float32> sparse values"))?;

    match (indices_list.is_null(row_idx), values_list.is_null(row_idx)) {
        (true, true) => return Ok(None),
        (true, false) | (false, true) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Lance sparse vector row has only one null child list",
            ));
        }
        (false, false) => {}
    }

    let indices_values = indices_list.value(row_idx);
    let indices = indices_values
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| column_type_error("UInt32 sparse indices values"))?
        .values()
        .to_vec();
    let values_values = values_list.value(row_idx);
    let values = values_values
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| column_type_error("Float32 sparse values"))?
        .values()
        .to_vec();

    if indices.len() != values.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Lance sparse vector row has mismatched indices/values lengths: {} != {}",
                indices.len(),
                values.len()
            ),
        ));
    }
    Ok(Some(SparseVector::new(indices, values)))
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
