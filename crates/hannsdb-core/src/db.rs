use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use hannsdb_index::adapter::VectorIndexBackend;
use hannsdb_index::descriptor::{ScalarIndexDescriptor, VectorIndexDescriptor, VectorIndexKind};
use hannsdb_index::factory::DefaultIndexFactory;
use hannsdb_index::scalar::{InvertedScalarIndex, RangeOp, ScalarValue};
use hannsdb_index::sparse::{SparseIndexBackend, SparseVectorData};

use crate::catalog::{CollectionMetadata, IndexCatalog, ManifestMetadata};
use crate::document::{
    CollectionSchema, Document, DocumentUpdate, FieldValue, ScalarFieldSchema, SparseVector,
    VectorFieldSchema, VectorIndexSchema,
};
use crate::query::{
    distance_by_metric, parse_filter, resolve_vector_descriptor_for_field, search_by_metric,
    search_sparse_bruteforce, ComparisonOp, FilterExpr, OrderBy, QueryContext, QueryExecutor,
    QueryPlan, QueryPlanner, SearchHit,
};
#[cfg(feature = "hanns-backend")]
use crate::segment::index_runtime::ann_search_with_bitset;
#[cfg(feature = "hanns-backend")]
use crate::segment::index_runtime::HNSW_INDEX_FILE;
use crate::segment::index_runtime::{
    ann_blob_path, ann_ids_path, ann_search, build_optimized_ann_state, invalidate_ann_blobs,
    persist_ann_blob, CachedSearchState, OptimizedAnnState,
};
use crate::segment::{
    append_payloads, append_record_ids, append_records, append_records_f16, append_sparse_vectors,
    append_vectors, ensure_payload_rows, ensure_vector_rows, load_payloads, load_payloads_jsonl,
    load_payloads_with_fields, load_record_ids, load_records, load_records_f16,
    load_sparse_vectors, load_vectors, load_vectors_jsonl, write_payloads_arrow,
    write_vectors_arrow, SegmentManager, SegmentMetadata, SegmentPaths, SegmentSet, TombstoneMask,
    VersionSet,
};
use crate::wal::{append_wal_record, load_wal_records, truncate_wal, WalRecord};

const DEFAULT_EF_SEARCH: usize = 32;
const DEFAULT_NPROBE: usize = 32;
const INDEX_CATALOG_FILE: &str = "indexes.json";

pub struct HannsDb {
    root: PathBuf,
    read_only: bool,
    collection_handles: RwLock<HashMap<String, Arc<CollectionHandle>>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CollectionInfo {
    pub name: String,
    pub dimension: usize,
    pub metric: String,
    pub record_count: usize,
    pub deleted_count: usize,
    pub live_count: usize,
    /// For each vector field, the fraction of live data covered by an ANN index (0.0..=1.0).
    /// A value of 1.0 means the field is fully indexed; 0.0 means no index exists.
    pub index_completeness: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionSegmentInfo {
    pub id: String,
    pub live_count: usize,
    pub dead_count: usize,
    pub ann_ready: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DocumentHit {
    pub id: i64,
    pub distance: f32,
    pub fields: BTreeMap<String, FieldValue>,
    pub vectors: BTreeMap<String, Vec<f32>>,
    pub sparse_vectors: BTreeMap<String, crate::document::SparseVector>,
    /// Set when group_by is active: the value of the group_by field for this hit.
    pub group_key: Option<FieldValue>,
}

#[derive(Clone)]
struct CachedDocumentState {
    documents: Arc<HashMap<i64, Document>>,
    collection_meta: Arc<CollectionMetadata>,
    index_catalog: Arc<IndexCatalog>,
}

#[derive(Debug, Default)]
struct IndexRegistry;

pub struct CollectionHandle {
    name: String,
    root: PathBuf,
    segment_manager: SegmentManager,
    version_set: RwLock<VersionSet>,
    index_registry: Arc<IndexRegistry>,
    search_cache: Mutex<HashMap<String, CachedSearchState>>,
    scalar_cache: Mutex<HashMap<String, InvertedScalarIndex>>,
    sparse_index_cache: Mutex<HashMap<String, Box<dyn SparseIndexBackend>>>,
    document_cache: Mutex<Option<CachedDocumentState>>,
}

impl HannsDb {
    pub fn open(root: &Path) -> io::Result<Self> {
        Self::open_internal(root, false)
    }

    pub fn open_read_only(root: &Path) -> io::Result<Self> {
        Self::open_internal(root, true)
    }

    fn open_internal(root: &Path, read_only: bool) -> io::Result<Self> {
        fs::create_dir_all(root)?;
        fs::create_dir_all(root.join("collections"))?;

        let manifest_path = manifest_path(root);
        if !manifest_path.exists() {
            ManifestMetadata::new("hannsdb-local", Vec::new()).save_to_path(&manifest_path)?;
        } else {
            let _ = ManifestMetadata::load_from_path(&manifest_path)?;
        }

        let mut db = Self {
            root: root.to_path_buf(),
            read_only,
            collection_handles: RwLock::new(HashMap::new()),
        };
        db.replay_wal_if_needed()?;
        Ok(db)
    }

    fn require_write(&self) -> io::Result<()> {
        if self.read_only {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "database is opened in read-only mode",
            ));
        }
        Ok(())
    }

    pub fn open_collection_handle(&self, name: &str) -> io::Result<Arc<CollectionHandle>> {
        if let Some(handle) = self
            .collection_handles
            .read()
            .expect("collection handles rwlock poisoned")
            .get(name)
            .cloned()
        {
            return Ok(handle);
        }

        let mut handles = self
            .collection_handles
            .write()
            .expect("collection handles rwlock poisoned");
        if let Some(handle) = handles.get(name) {
            return Ok(Arc::clone(handle));
        }

        let paths = self.collection_paths(name);
        let _ = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let segment_manager = SegmentManager::new(paths.dir.clone());
        let version_set = segment_manager.version_set()?;
        let handle = Arc::new(CollectionHandle::new(
            name.to_string(),
            self.root.clone(),
            segment_manager,
            version_set,
            Arc::new(IndexRegistry),
        ));
        handles.insert(name.to_string(), Arc::clone(&handle));
        Ok(handle)
    }

    pub fn create_collection(
        &mut self,
        name: &str,
        dimension: usize,
        metric: &str,
    ) -> io::Result<()> {
        self.create_collection_with_schema(
            name,
            &CollectionSchema::new("vector", dimension, metric, Vec::new()),
        )
    }

    pub fn create_collection_with_schema(
        &mut self,
        name: &str,
        schema: &CollectionSchema,
    ) -> io::Result<()> {
        self.require_write()?;
        self.create_collection_with_schema_internal(name, schema, true)
    }

    fn create_collection_with_schema_internal(
        &mut self,
        name: &str,
        schema: &CollectionSchema,
        log_wal: bool,
    ) -> io::Result<()> {
        let _primary_vector = schema.primary_vector().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "collection primary vector '{}' is not defined in schema vectors",
                    schema.primary_vector_name()
                ),
            )
        })?;
        validate_schema_primary_vector_descriptor(schema)?;
        validate_schema_secondary_vector_descriptors(schema)?;
        let dimension = schema.dimension();
        if dimension == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "collection dimension must be > 0",
            ));
        }

        let paths = self.collection_paths(name);
        if paths.collection_meta.exists() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("collection already exists: {name}"),
            ));
        }

        if log_wal {
            append_wal_record(
                &wal_path(&self.root),
                &WalRecord::CreateCollection {
                    collection: name.to_string(),
                    schema: schema.clone(),
                },
            )?;
        }

        fs::create_dir_all(&paths.dir)?;

        let collection = CollectionMetadata::new_with_schema(name, schema.clone());
        collection.save_to_path(&paths.collection_meta)?;

        let segment = SegmentMetadata::new("seg-0001", dimension, 0, 0);
        segment.save_to_path(&paths.segment_meta)?;

        let tombstone = TombstoneMask::new(0);
        tombstone.save_to_path(&paths.tombstones)?;

        let mut manifest = ManifestMetadata::load_from_path(&manifest_path(&self.root))?;
        if !manifest.collections.iter().any(|entry| entry == name) {
            manifest.collections.push(name.to_string());
            manifest.save_to_path(&manifest_path(&self.root))?;
        }

        self.invalidate_search_cache(name);
        Ok(())
    }

    pub fn add_column(&mut self, collection: &str, field: ScalarFieldSchema) -> io::Result<()> {
        self.require_write()?;
        self.add_column_internal(collection, field, true)
    }

    fn add_column_internal(
        &mut self,
        collection: &str,
        field: ScalarFieldSchema,
        log_wal: bool,
    ) -> io::Result<()> {
        let paths = self.collection_paths(collection);
        let mut meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;

        crate::catalog::schema_mutation::add_field_to_schema(&mut meta.fields, field.clone())?;

        if log_wal {
            append_wal_record(
                &wal_path(&self.root),
                &WalRecord::AddColumn {
                    collection: collection.to_string(),
                    field,
                },
            )?;
        }

        meta.save_to_path(&paths.collection_meta)?;
        self.invalidate_search_cache(collection);
        Ok(())
    }

    pub fn drop_column(&mut self, collection: &str, field_name: &str) -> io::Result<()> {
        self.require_write()?;
        self.drop_column_internal(collection, field_name, true)
    }

    fn drop_column_internal(
        &mut self,
        collection: &str,
        field_name: &str,
        log_wal: bool,
    ) -> io::Result<()> {
        let paths = self.collection_paths(collection);
        let mut meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;

        if log_wal {
            append_wal_record(
                &wal_path(&self.root),
                &WalRecord::DropColumn {
                    collection: collection.to_string(),
                    field_name: field_name.to_string(),
                },
            )?;
        }

        crate::catalog::schema_mutation::remove_field_from_schema(&mut meta.fields, field_name)?;
        meta.save_to_path(&paths.collection_meta)?;
        self.invalidate_search_cache(collection);
        Ok(())
    }

    pub fn alter_column(
        &mut self,
        collection: &str,
        old_name: &str,
        new_name: &str,
    ) -> io::Result<()> {
        self.require_write()?;
        self.alter_column_internal(collection, old_name, new_name, true)
    }

    fn alter_column_internal(
        &mut self,
        collection: &str,
        old_name: &str,
        new_name: &str,
        log_wal: bool,
    ) -> io::Result<()> {
        let paths = self.collection_paths(collection);
        let mut meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;

        if log_wal {
            append_wal_record(
                &wal_path(&self.root),
                &WalRecord::AlterColumn {
                    collection: collection.to_string(),
                    old_name: old_name.to_string(),
                    new_name: new_name.to_string(),
                },
            )?;
        }

        crate::catalog::schema_mutation::rename_field_in_schema(
            &mut meta.fields,
            old_name,
            new_name,
        )?;
        meta.save_to_path(&paths.collection_meta)?;
        self.invalidate_search_cache(collection);
        Ok(())
    }

    pub fn add_vector_field(
        &mut self,
        collection: &str,
        field: VectorFieldSchema,
    ) -> io::Result<()> {
        self.require_write()?;
        self.add_vector_field_internal(collection, field, true)
    }

    fn add_vector_field_internal(
        &mut self,
        collection: &str,
        field: VectorFieldSchema,
        log_wal: bool,
    ) -> io::Result<()> {
        let paths = self.collection_paths(collection);
        let mut meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;

        // Also check scalar fields for name collisions.
        if meta.fields.iter().any(|f| f.name == field.name) {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("a scalar field with name '{}' already exists", field.name),
            ));
        }

        crate::catalog::schema_mutation::add_vector_field_to_schema(
            &mut meta.vectors,
            field.clone(),
        )?;

        if log_wal {
            append_wal_record(
                &wal_path(&self.root),
                &WalRecord::AddVectorField {
                    collection: collection.to_string(),
                    field,
                },
            )?;
        }

        meta.save_to_path(&paths.collection_meta)?;
        self.invalidate_search_cache(collection);
        Ok(())
    }

    pub fn drop_vector_field(&mut self, collection: &str, field_name: &str) -> io::Result<()> {
        self.require_write()?;
        self.drop_vector_field_internal(collection, field_name, true)
    }

    fn drop_vector_field_internal(
        &mut self,
        collection: &str,
        field_name: &str,
        log_wal: bool,
    ) -> io::Result<()> {
        let paths = self.collection_paths(collection);
        let mut meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;

        crate::catalog::schema_mutation::remove_vector_field_from_schema(
            &mut meta.vectors,
            field_name,
            &meta.primary_vector,
        )?;

        if log_wal {
            append_wal_record(
                &wal_path(&self.root),
                &WalRecord::DropVectorField {
                    collection: collection.to_string(),
                    field_name: field_name.to_string(),
                },
            )?;
        }

        // Drop any associated vector index descriptor.
        if paths.index_catalog.exists() {
            if let Ok(mut catalog) = IndexCatalog::load_from_path(&paths.index_catalog) {
                catalog.drop_vector_index(field_name);
                catalog.save_to_path(&paths.index_catalog)?;
            }
        }

        // Remove persisted ANN blobs for this field.
        let ann_blob = ann_blob_path(&paths.dir, field_name);
        if ann_blob.exists() {
            let _ = fs::remove_file(&ann_blob);
        }
        let ann_ids = ann_ids_path(&paths.dir, field_name);
        if ann_ids.exists() {
            let _ = fs::remove_file(&ann_ids);
        }

        meta.save_to_path(&paths.collection_meta)?;
        self.invalidate_search_cache(collection);
        Ok(())
    }

    pub fn drop_collection(&mut self, name: &str) -> io::Result<()> {
        self.require_write()?;
        self.drop_collection_internal(name, true)
    }

    fn drop_collection_internal(&mut self, name: &str, log_wal: bool) -> io::Result<()> {
        let paths = self.collection_paths(name);
        if !paths.dir.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("collection not found: {name}"),
            ));
        }

        if log_wal {
            append_wal_record(
                &wal_path(&self.root),
                &WalRecord::DropCollection {
                    collection: name.to_string(),
                },
            )?;
        }

        fs::remove_dir_all(&paths.dir)?;

        let mut manifest = ManifestMetadata::load_from_path(&manifest_path(&self.root))?;
        manifest.collections.retain(|entry| entry != name);
        manifest.save_to_path(&manifest_path(&self.root))?;
        self.collection_handles
            .write()
            .expect("collection handles rwlock poisoned")
            .remove(name);
        self.invalidate_search_cache(name);
        Ok(())
    }

    pub fn list_collections(&self) -> io::Result<Vec<String>> {
        let manifest = ManifestMetadata::load_from_path(&manifest_path(&self.root))?;
        Ok(manifest.collections)
    }

    pub fn create_vector_index(
        &self,
        collection: &str,
        descriptor: VectorIndexDescriptor,
    ) -> io::Result<()> {
        let paths = self.collection_paths(collection);
        let metadata = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let vector = if let Some(vector) = metadata
            .vectors
            .iter()
            .find(|vector| vector.name == descriptor.field_name)
        {
            vector
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "vector field '{}' is not defined in collection '{}'",
                    descriptor.field_name, collection
                ),
            ));
        };
        validate_vector_index_descriptor(vector.dimension, &descriptor)?;

        let mut catalog = IndexCatalog::load_from_path(&paths.index_catalog)?;
        catalog.upsert_vector_index(descriptor);
        catalog.save_to_path(&paths.index_catalog)?;
        invalidate_ann_blobs(&paths.dir)?;
        self.invalidate_search_cache(collection);
        Ok(())
    }

    pub fn drop_vector_index(&self, collection: &str, field_name: &str) -> io::Result<()> {
        let paths = self.collection_paths(collection);
        let _ = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let mut catalog = IndexCatalog::load_from_path(&paths.index_catalog)?;
        if !catalog.drop_vector_index(field_name) {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "vector index descriptor not found for field '{}' in '{}'",
                    field_name, collection
                ),
            ));
        }
        catalog.save_to_path(&paths.index_catalog)?;
        invalidate_ann_blobs(&paths.dir)?;
        self.invalidate_search_cache(collection);
        Ok(())
    }

    pub fn list_vector_indexes(&self, collection: &str) -> io::Result<Vec<VectorIndexDescriptor>> {
        let paths = self.collection_paths(collection);
        let _ = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        Ok(IndexCatalog::load_from_path(&paths.index_catalog)?.vector_indexes)
    }

    pub fn create_scalar_index(
        &self,
        collection: &str,
        descriptor: ScalarIndexDescriptor,
    ) -> io::Result<()> {
        let paths = self.collection_paths(collection);
        let metadata = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        if !metadata
            .fields
            .iter()
            .any(|field| field.name == descriptor.field_name)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "scalar field '{}' is not defined in collection '{}'",
                    descriptor.field_name, collection
                ),
            ));
        }

        let mut catalog = IndexCatalog::load_from_path(&paths.index_catalog)?;
        catalog.upsert_scalar_index(descriptor);
        catalog.save_to_path(&paths.index_catalog)?;
        Ok(())
    }

    pub fn drop_scalar_index(&self, collection: &str, field_name: &str) -> io::Result<()> {
        let paths = self.collection_paths(collection);
        let _ = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let mut catalog = IndexCatalog::load_from_path(&paths.index_catalog)?;
        if !catalog.drop_scalar_index(field_name) {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "scalar index descriptor not found for field '{}' in '{}'",
                    field_name, collection
                ),
            ));
        }
        catalog.save_to_path(&paths.index_catalog)?;
        Ok(())
    }

    pub fn list_scalar_indexes(&self, collection: &str) -> io::Result<Vec<ScalarIndexDescriptor>> {
        let paths = self.collection_paths(collection);
        let _ = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        Ok(IndexCatalog::load_from_path(&paths.index_catalog)?.scalar_indexes)
    }

    pub fn flush_collection(&self, name: &str) -> io::Result<()> {
        self.open_collection_handle(name)?.flush()?;
        // After successful flush, all WAL entries are reflected in the segment
        // files on disk. Truncate the WAL to reclaim space and avoid redundant
        // replay on the next open.
        let wal = wal_path(&self.root);
        if wal.exists() {
            truncate_wal(&wal)?;
        }
        Ok(())
    }

    pub fn optimize_collection(&self, name: &str) -> io::Result<()> {
        self.open_collection_handle(name)?.optimize()?;
        // After successful optimize, all data is persisted in segment files and
        // ANN index blobs. Truncate the WAL since replay is no longer needed.
        let wal = wal_path(&self.root);
        if wal.exists() {
            truncate_wal(&wal)?;
        }
        Ok(())
    }

    pub fn compact_collection(&mut self, name: &str) -> io::Result<()> {
        self.require_write()?;
        self.compact_collection_internal(name, true)
    }

    fn compact_collection_internal(&mut self, name: &str, log_wal: bool) -> io::Result<()> {
        let paths = self.collection_paths(name);
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        if !paths.segment_set.exists() {
            return Ok(());
        }

        let mut version_set = VersionSet::load_from_path(&paths.segment_set)?;
        if version_set.immutable_segment_ids().is_empty() {
            return Ok(());
        }

        fs::create_dir_all(&paths.segments_dir)?;
        let immutable_segment_ids = version_set.immutable_segment_ids().to_vec();
        let active_segment_id = version_set.active_segment_id().to_string();
        let compacted_segment_id = next_compacted_segment_id(
            immutable_segment_ids
                .iter()
                .chain(std::iter::once(&active_segment_id)),
        );
        let compacted_dir = paths.segments_dir.join(&compacted_segment_id);
        fs::create_dir_all(&compacted_dir)?;
        let compacted_paths =
            SegmentPaths::from_segment_dir(compacted_dir.clone(), compacted_segment_id.clone());

        let mut compacted_ids = Vec::new();
        let mut compacted_records = Vec::new();
        let mut compacted_payloads = Vec::new();
        let mut compacted_vectors = Vec::new();

        for segment_id in &immutable_segment_ids {
            let segment_dir = paths.segments_dir.join(segment_id);
            let segment_meta = SegmentMetadata::load_from_path(&segment_dir.join("segment.json"))?;
            if segment_meta.dimension != collection_meta.dimension {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "segment dimension mismatch: expected {}, got {}",
                        collection_meta.dimension, segment_meta.dimension
                    ),
                ));
            }

            let segment_paths = SegmentPaths::from_segment_dir(segment_dir, segment_id.clone());
            let fp16 = collection_meta.primary_is_fp16();
            let segment_records = if fp16 {
                load_records_f16(&segment_paths.records, collection_meta.dimension)?
            } else {
                load_records(&segment_paths.records, collection_meta.dimension)?
            };
            let segment_external_ids = load_record_ids(&segment_paths.external_ids)?;
            let segment_payloads =
                load_payloads_or_empty(&segment_paths.payloads, segment_external_ids.len())?;
            let segment_vectors =
                load_vectors_or_empty(&segment_paths.vectors, segment_external_ids.len())?;
            let segment_tombstone = TombstoneMask::load_from_path(&segment_paths.tombstones)?;

            if segment_external_ids
                .len()
                .saturating_mul(collection_meta.dimension)
                != segment_records.len()
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "records and ids are not aligned",
                ));
            }

            for (row_idx, vector) in segment_records
                .chunks_exact(collection_meta.dimension)
                .enumerate()
            {
                if segment_tombstone.is_deleted(row_idx) {
                    continue;
                }
                compacted_records.extend_from_slice(vector);
                compacted_ids.push(segment_external_ids[row_idx]);
                compacted_payloads.push(segment_payloads[row_idx].clone());
                compacted_vectors.push(segment_vectors[row_idx].clone());
            }
        }

        let fp16 = collection_meta.primary_is_fp16();
        let inserted = if fp16 {
            append_records_f16(
                &compacted_paths.records,
                collection_meta.dimension,
                &compacted_records,
            )?
        } else {
            append_records(
                &compacted_paths.records,
                collection_meta.dimension,
                &compacted_records,
            )?
        };
        if inserted != compacted_ids.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "compacted records and ids are not aligned",
            ));
        }
        let _ = append_record_ids(&compacted_paths.external_ids, &compacted_ids)?;
        write_payloads_arrow(
            &compacted_paths.payloads_arrow,
            &compacted_payloads,
            &collection_meta.fields,
        )?;
        write_vectors_arrow(
            &compacted_paths.vectors_arrow,
            &compacted_vectors,
            &collection_meta.vectors,
            &collection_meta.primary_vector,
        )?;
        TombstoneMask::new(compacted_ids.len()).save_to_path(&compacted_paths.tombstones)?;
        let mut compacted_meta = SegmentMetadata::new(
            compacted_segment_id.clone(),
            collection_meta.dimension,
            compacted_ids.len(),
            0,
        );
        compacted_meta.storage_format = "arrow".to_string();
        compacted_meta.save_to_path(&compacted_dir.join("segment.json"))?;

        version_set = VersionSet::new(
            version_set.active_segment_id().to_string(),
            vec![compacted_segment_id.clone()],
        );
        version_set.save_to_path(&paths.segment_set)?;
        for segment_id in &immutable_segment_ids {
            fs::remove_dir_all(paths.segments_dir.join(segment_id))?;
        }

        if log_wal {
            append_wal_record(
                &wal_path(&self.root),
                &WalRecord::CompactCollection {
                    collection_name: name.to_string(),
                    compacted_segment_id,
                },
            )?;
        }

        self.invalidate_search_cache(name);
        Ok(())
    }

    pub fn get_collection_info(&self, name: &str) -> io::Result<CollectionInfo> {
        self.open_collection_handle(name)?.collection_info()
    }

    pub fn list_collection_segments(&self, name: &str) -> io::Result<Vec<CollectionSegmentInfo>> {
        self.open_collection_handle(name)?.list_segments()
    }

    pub fn insert(
        &mut self,
        collection: &str,
        external_ids: &[i64],
        vectors: &[f32],
    ) -> io::Result<usize> {
        self.require_write()?;
        self.insert_internal(collection, external_ids, vectors, true)
    }

    fn insert_internal(
        &mut self,
        collection: &str,
        external_ids: &[i64],
        vectors: &[f32],
        log_wal: bool,
    ) -> io::Result<usize> {
        let paths = self.collection_paths(collection);
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let mut segment_meta = SegmentMetadata::load_from_path(&paths.segment_meta)?;
        let mut tombstone = TombstoneMask::load_from_path(&paths.tombstones)?;

        if collection_meta.dimension == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "collection dimension must be > 0",
            ));
        }

        if vectors.len() % collection_meta.dimension != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "vector count must be aligned to dimension",
            ));
        }
        let expected = vectors.len() / collection_meta.dimension;
        if external_ids.len() != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "external id count must match vector count",
            ));
        }

        let mut new_ids = HashSet::with_capacity(external_ids.len());
        for external_id in external_ids {
            if !new_ids.insert(*external_id) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("duplicate external id in batch: {external_id}"),
                ));
            }
        }

        let existing_ids = load_record_ids_or_empty(&paths.external_ids)?;
        let existing_set: HashSet<i64> = existing_ids.into_iter().collect();
        for external_id in external_ids {
            if existing_set.contains(external_id) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("external id already exists: {external_id}"),
                ));
            }
        }

        if log_wal {
            append_wal_record(
                &wal_path(&self.root),
                &WalRecord::Insert {
                    collection: collection.to_string(),
                    ids: external_ids.to_vec(),
                    vectors: vectors.to_vec(),
                },
            )?;
        }

        ensure_payload_rows(&paths.payloads, segment_meta.record_count)?;
        ensure_vector_rows(&paths.vectors, segment_meta.record_count)?;
        let fp16 = collection_meta.primary_is_fp16();
        let inserted = if fp16 {
            append_records_f16(&paths.records, collection_meta.dimension, vectors)?
        } else {
            append_records(&paths.records, collection_meta.dimension, vectors)?
        };
        let _ = append_record_ids(&paths.external_ids, external_ids)?;
        let empty_payloads = vec![BTreeMap::new(); inserted];
        let empty_vectors = vec![BTreeMap::new(); inserted];
        let empty_sparse = vec![BTreeMap::new(); inserted];
        let _ = append_payloads(&paths.payloads, &empty_payloads)?;
        let _ = append_vectors(&paths.vectors, &empty_vectors)?;
        let _ = append_sparse_vectors(&paths.sparse_vectors, &empty_sparse)?;
        segment_meta.record_count += inserted;
        segment_meta.deleted_count = tombstone.deleted_count();

        let needed = segment_meta.record_count.saturating_sub(tombstone.len());
        if needed > 0 {
            tombstone.extend(needed);
        }

        segment_meta.save_to_path(&paths.segment_meta)?;
        tombstone.save_to_path(&paths.tombstones)?;
        self.maybe_trigger_segment_rollover(&paths, &segment_meta)?;
        if self.should_auto_compact(collection)? {
            self.compact_collection_internal(collection, true)?;
        }
        self.invalidate_search_cache(collection);
        Ok(inserted)
    }

    pub fn insert_documents(
        &mut self,
        collection: &str,
        documents: &[Document],
    ) -> io::Result<usize> {
        self.require_write()?;
        self.insert_documents_internal(collection, documents, true)
    }

    fn insert_documents_internal(
        &mut self,
        collection: &str,
        documents: &[Document],
        log_wal: bool,
    ) -> io::Result<usize> {
        let paths = self.collection_paths(collection);
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let mut segment_meta = SegmentMetadata::load_from_path(&paths.segment_meta)?;
        let mut tombstone = TombstoneMask::load_from_path(&paths.tombstones)?;

        validate_documents(documents, &collection_meta)?;

        let existing_ids = load_record_ids_or_empty(&paths.external_ids)?;
        for document in documents {
            if has_live_id(&existing_ids, &tombstone, document.id) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("external id already exists: {}", document.id),
                ));
            }
        }

        if log_wal {
            append_wal_record(
                &wal_path(&self.root),
                &WalRecord::InsertDocuments {
                    collection: collection.to_string(),
                    documents: documents.to_vec(),
                },
            )?;
        }

        ensure_payload_rows(&paths.payloads, segment_meta.record_count)?;
        let inserted = append_documents(
            &paths,
            collection_meta.dimension,
            segment_meta.record_count,
            &collection_meta.primary_vector,
            documents,
            collection_meta.primary_is_fp16(),
        )?;
        segment_meta.record_count += inserted;
        segment_meta.deleted_count = tombstone.deleted_count();

        let needed = segment_meta.record_count.saturating_sub(tombstone.len());
        if needed > 0 {
            tombstone.extend(needed);
        }

        segment_meta.save_to_path(&paths.segment_meta)?;
        tombstone.save_to_path(&paths.tombstones)?;
        if self.should_auto_compact(collection)? {
            self.compact_collection_internal(collection, true)?;
        }
        self.invalidate_search_cache(collection);
        Ok(inserted)
    }

    pub fn upsert_documents(
        &mut self,
        collection: &str,
        documents: &[Document],
    ) -> io::Result<usize> {
        self.require_write()?;
        self.upsert_documents_internal(collection, documents, true)
    }

    fn upsert_documents_internal(
        &mut self,
        collection: &str,
        documents: &[Document],
        log_wal: bool,
    ) -> io::Result<usize> {
        let paths = self.collection_paths(collection);
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let mut segment_meta = SegmentMetadata::load_from_path(&paths.segment_meta)?;
        let mut tombstone = TombstoneMask::load_from_path(&paths.tombstones)?;
        let existing_ids = load_record_ids_or_empty(&paths.external_ids)?;

        validate_documents(documents, &collection_meta)?;

        if log_wal {
            append_wal_record(
                &wal_path(&self.root),
                &WalRecord::UpsertDocuments {
                    collection: collection.to_string(),
                    documents: documents.to_vec(),
                },
            )?;
        }

        for document in documents {
            mark_live_id_deleted(
                &existing_ids,
                &mut tombstone,
                document.id,
                segment_meta.record_count,
            );
        }

        ensure_payload_rows(&paths.payloads, segment_meta.record_count)?;
        let inserted = append_documents(
            &paths,
            collection_meta.dimension,
            segment_meta.record_count,
            &collection_meta.primary_vector,
            documents,
            collection_meta.primary_is_fp16(),
        )?;
        segment_meta.record_count += inserted;
        segment_meta.deleted_count = tombstone.deleted_count();

        let needed = segment_meta.record_count.saturating_sub(tombstone.len());
        if needed > 0 {
            tombstone.extend(needed);
        }

        segment_meta.save_to_path(&paths.segment_meta)?;
        tombstone.save_to_path(&paths.tombstones)?;
        if self.should_auto_compact(collection)? {
            self.compact_collection_internal(collection, true)?;
        }
        self.invalidate_search_cache(collection);
        Ok(inserted)
    }

    /// Partially update existing documents.
    ///
    /// For each `DocumentUpdate`, fields/vectors set to `Some(..)` replace the
    /// current value; `None` means "keep current". The old row is tombstoned
    /// and a merged row is appended.
    pub fn update_documents(
        &mut self,
        collection: &str,
        updates: &[DocumentUpdate],
    ) -> io::Result<usize> {
        self.require_write()?;
        self.update_documents_internal(collection, updates, true)
    }

    fn update_documents_internal(
        &mut self,
        collection: &str,
        updates: &[DocumentUpdate],
        log_wal: bool,
    ) -> io::Result<usize> {
        if updates.is_empty() {
            return Ok(0);
        }

        // Fetch existing documents.
        let ids: Vec<i64> = updates.iter().map(|u| u.id).collect();
        let existing = self.fetch_documents(collection, &ids)?;
        if existing.is_empty() {
            return Ok(0);
        }

        let existing_map: HashMap<i64, Document> =
            existing.into_iter().map(|doc| (doc.id, doc)).collect();

        // Merge updates into full documents.
        let merged: Vec<Document> = updates
            .iter()
            .filter_map(|update| {
                let existing = existing_map.get(&update.id)?;
                let mut fields = existing.fields.clone();
                for (key, value) in &update.fields {
                    match value {
                        Some(v) => {
                            fields.insert(key.clone(), v.clone());
                        }
                        None => {
                            fields.remove(key);
                        }
                    }
                }
                let mut vectors = existing.vectors.clone();
                for (key, value) in &update.vectors {
                    match value {
                        Some(v) => {
                            vectors.insert(key.clone(), v.clone());
                        }
                        None => {
                            vectors.remove(key);
                        }
                    }
                }
                Some(Document {
                    id: update.id,
                    fields,
                    vectors,
                    sparse_vectors: existing.sparse_vectors.clone(),
                })
            })
            .collect();

        if merged.is_empty() {
            return Ok(0);
        }

        if log_wal {
            append_wal_record(
                &wal_path(&self.root),
                &WalRecord::UpdateDocuments {
                    collection: collection.to_string(),
                    updates: updates.to_vec(),
                },
            )?;
        }

        // Delegate to upsert (which does tombstone+append), but skip its own
        // WAL write since we already logged the UpdateDocuments record.
        self.upsert_documents_internal(collection, &merged, false)
    }

    pub fn fetch_documents(
        &self,
        collection: &str,
        external_ids: &[i64],
    ) -> io::Result<Vec<Document>> {
        self.open_collection_handle(collection)?
            .fetch_documents(external_ids)
    }

    pub fn delete(&mut self, collection: &str, external_ids: &[i64]) -> io::Result<usize> {
        self.require_write()?;
        self.delete_internal(collection, external_ids, true)
    }

    pub fn delete_by_filter(&mut self, collection: &str, filter: &str) -> io::Result<usize> {
        self.require_write()?;
        let filter_expr = parse_filter(filter)?;
        let matching_ids = self.collect_latest_live_filtered_ids(collection, &filter_expr)?;
        if matching_ids.is_empty() {
            return Ok(0);
        }

        let matching_ids = matching_ids.into_iter().collect::<Vec<_>>();
        self.delete_internal(collection, &matching_ids, true)
    }

    fn delete_internal(
        &mut self,
        collection: &str,
        external_ids: &[i64],
        log_wal: bool,
    ) -> io::Result<usize> {
        let paths = self.collection_paths(collection);
        let segment_paths = SegmentManager::new(paths.dir.clone()).segment_paths()?;

        if log_wal {
            append_wal_record(
                &wal_path(&self.root),
                &WalRecord::Delete {
                    collection: collection.to_string(),
                    ids: external_ids.to_vec(),
                },
            )?;
        }

        let mut newly_deleted = 0usize;
        let mut remaining_ids = external_ids.iter().copied().collect::<HashSet<_>>();

        for segment in segment_paths {
            if remaining_ids.is_empty() {
                break;
            }

            let stored_ids = load_record_ids_or_empty(&segment.external_ids)?;
            if stored_ids.is_empty() {
                continue;
            }

            let mut segment_meta = SegmentMetadata::load_from_path(&segment.metadata)?;
            let mut tombstone = TombstoneMask::load_from_path(&segment.tombstones)?;
            let mut segment_changed = false;
            let row_limit = segment_meta.record_count.min(stored_ids.len());
            let stored_ids = &stored_ids[..row_limit];
            let ids_to_check = remaining_ids.iter().copied().collect::<Vec<_>>();

            for external_id in ids_to_check {
                let Some(row_idx) = latest_row_index_for_id(stored_ids, external_id) else {
                    continue;
                };
                remaining_ids.remove(&external_id);
                if tombstone.is_deleted(row_idx) {
                    continue;
                }
                if tombstone.mark_deleted(row_idx) {
                    newly_deleted += 1;
                    segment_changed = true;
                }
            }

            if segment_changed {
                segment_meta.deleted_count = tombstone.deleted_count();
                segment_meta.save_to_path(&segment.metadata)?;
                tombstone.save_to_path(&segment.tombstones)?;
            }
        }

        if self.should_auto_compact(collection)? {
            self.compact_collection_internal(collection, true)?;
        }
        self.invalidate_search_cache(collection);
        Ok(newly_deleted)
    }

    fn collect_latest_live_filtered_ids(
        &self,
        collection: &str,
        filter_expr: &FilterExpr,
    ) -> io::Result<BTreeSet<i64>> {
        let paths = self.collection_paths(collection);
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let segment_paths = SegmentManager::new(paths.dir.clone()).segment_paths()?;

        let mut seen_ids = HashSet::new();
        let mut matching_ids = BTreeSet::new();

        for segment in segment_paths {
            let segment_meta = SegmentMetadata::load_from_path(&segment.metadata)?;
            let records = load_records_or_empty(
                &segment.records,
                collection_meta.dimension,
                collection_meta.primary_is_fp16(),
            )?;
            let stored_ids = load_record_ids_or_empty(&segment.external_ids)?;
            let payloads = load_payloads_or_empty(&segment.payloads, stored_ids.len())?;
            let tombstone = TombstoneMask::load_from_path(&segment.tombstones)?;

            let row_limit = segment_meta
                .record_count
                .min(stored_ids.len())
                .min(records.len() / collection_meta.dimension);
            if row_limit == 0 {
                continue;
            }

            for row_idx in (0..row_limit).rev() {
                let external_id = stored_ids[row_idx];
                if !seen_ids.insert(external_id) {
                    continue;
                }
                if tombstone.is_deleted(row_idx) {
                    continue;
                }
                if filter_expr.matches(&payloads[row_idx]) {
                    matching_ids.insert(external_id);
                }
            }
        }

        Ok(matching_ids)
    }

    pub fn search(
        &self,
        collection: &str,
        query: &[f32],
        top_k: usize,
    ) -> io::Result<Vec<SearchHit>> {
        self.search_with_ef(collection, query, top_k, DEFAULT_EF_SEARCH)
    }

    pub fn search_with_ef(
        &self,
        collection: &str,
        query: &[f32],
        top_k: usize,
        _ef_search: usize,
    ) -> io::Result<Vec<SearchHit>> {
        self.open_collection_handle(collection)?
            .search_with_ef(query, top_k, _ef_search)
    }

    pub fn query_documents(
        &self,
        collection: &str,
        query: &[f32],
        top_k: usize,
        filter: Option<&str>,
    ) -> io::Result<Vec<DocumentHit>> {
        self.open_collection_handle(collection)?
            .query_documents(query, top_k, filter)
    }

    pub fn query_with_context(
        &self,
        collection: &str,
        context: &QueryContext,
    ) -> io::Result<Vec<DocumentHit>> {
        self.open_collection_handle(collection)?
            .query_with_context(context)
    }

    /// Search a sparse vector field by brute force.
    ///
    /// Returns top-k hits ranked by sparse inner product (descending similarity).
    pub fn search_sparse(
        &self,
        collection: &str,
        field_name: &str,
        query: &SparseVector,
        top_k: usize,
    ) -> io::Result<Vec<SearchHit>> {
        self.open_collection_handle(collection)?
            .search_sparse(field_name, query, top_k)
    }

    fn collection_paths(&self, name: &str) -> CollectionPaths {
        collection_paths_for_root(&self.root, name)
    }

    fn should_auto_compact(&self, collection: &str) -> io::Result<bool> {
        let paths = self.collection_paths(collection);
        if !paths.segment_set.exists() {
            return Ok(false);
        }
        let version_set = VersionSet::load_from_path(&paths.segment_set)?;
        Ok(SegmentSet::should_compact(
            version_set.immutable_segment_ids().len(),
        ))
    }

    fn invalidate_search_cache(&self, collection: &str) {
        let handle = {
            let handles = self
                .collection_handles
                .read()
                .expect("collection handles rwlock poisoned");
            handles.get(collection).cloned()
        };
        if let Some(handle) = handle {
            let _ = handle.refresh_version_set();
            handle.invalidate_search_cache();
        }
    }

    fn maybe_trigger_segment_rollover(
        &self,
        paths: &CollectionPaths,
        segment_meta: &SegmentMetadata,
    ) -> io::Result<()> {
        if !SegmentSet::should_rollover(
            segment_meta.record_count as u64,
            segment_meta.deleted_count as u64,
        ) {
            return Ok(());
        }

        let mut version_set = if paths.segment_set.exists() {
            VersionSet::load_from_path(&paths.segment_set)?
        } else {
            VersionSet::single(&segment_meta.segment_id)
        };

        let new_id = version_set.rollover();

        // If no segment_set.json exists, migrate root-level files into segments/
        let old_seg_dir = if !paths.segment_set.exists() {
            let segments_dir = paths.segments_dir.clone();
            let old_dir = segments_dir.join(&segment_meta.segment_id);
            fs::create_dir_all(&segments_dir)?;
            fs::create_dir_all(&old_dir)?;

            let files_to_move = [
                "segment.json",
                "records.bin",
                "ids.bin",
                "payloads.jsonl",
                "vectors.jsonl",
                "tombstones.json",
            ];
            for file_name in &files_to_move {
                let src = paths.dir.join(file_name);
                if src.exists() {
                    fs::rename(&src, old_dir.join(file_name))?;
                }
            }
            old_dir
        } else {
            paths.segments_dir.join(&segment_meta.segment_id)
        };

        // Convert JSONL → Arrow IPC for the now-immutable segment.
        if let Err(e) = Self::convert_segment_jsonl_to_arrow(&old_seg_dir, &paths.dir) {
            log::warn!(
                "JSONL→Arrow conversion failed for {}, keeping JSONL: {}",
                segment_meta.segment_id,
                e
            );
        }

        // Create the new active segment directory
        let seg_manager = SegmentManager::new(paths.dir.clone());
        seg_manager.create_segment_dir(&new_id, segment_meta.dimension)?;

        // Persist the updated version set
        version_set.save_to_path(&paths.segment_set)?;

        log::info!("Segment rollover: {} → {}", segment_meta.segment_id, new_id);

        Ok(())
    }

    /// Convert JSONL files to Arrow IPC for an immutable segment directory.
    fn convert_segment_jsonl_to_arrow(
        seg_dir: &std::path::Path,
        collection_dir: &std::path::Path,
    ) -> io::Result<()> {
        let collection_meta_path = collection_dir.join("collection.json");
        if !collection_meta_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "collection.json not found",
            ));
        }
        let collection_meta = CollectionMetadata::load_from_path(&collection_meta_path)?;

        let jsonl_payloads = seg_dir.join("payloads.jsonl");
        let jsonl_vectors = seg_dir.join("vectors.jsonl");
        let arrow_payloads = seg_dir.join("payloads.arrow");
        let arrow_vectors = seg_dir.join("vectors.arrow");

        if jsonl_payloads.exists() {
            let payloads = load_payloads_jsonl(&jsonl_payloads)?;
            write_payloads_arrow(&arrow_payloads, &payloads, &collection_meta.fields)?;
            let _ = fs::remove_file(&jsonl_payloads);
        }

        if jsonl_vectors.exists() {
            let vectors = load_vectors_jsonl(&jsonl_vectors)?;
            write_vectors_arrow(
                &arrow_vectors,
                &vectors,
                &collection_meta.vectors,
                &collection_meta.primary_vector,
            )?;
            let _ = fs::remove_file(&jsonl_vectors);
        }

        let meta_path = seg_dir.join("segment.json");
        if meta_path.exists() {
            let mut meta = SegmentMetadata::load_from_path(&meta_path)?;
            meta.storage_format = "arrow".to_string();
            meta.save_to_path(&meta_path)?;
        }

        Ok(())
    }

    fn replay_wal_if_needed(&mut self) -> io::Result<()> {
        let records = load_wal_records_or_empty(&wal_path(&self.root))?;
        if records.is_empty() {
            return Ok(());
        }

        let plan = WalReplayPlan::build(&records);
        if !plan.has_owned_collections() || !plan.requires_replay(self)? {
            return Ok(());
        }

        let manifest_path = manifest_path(&self.root);
        let mut manifest = ManifestMetadata::load_from_path(&manifest_path)?;
        for collection in plan.owned_collections() {
            let paths = self.collection_paths(collection);
            if paths.dir.exists() {
                fs::remove_dir_all(&paths.dir)?;
            }
            manifest.collections.retain(|entry| entry != collection);
            self.invalidate_search_cache(collection);
        }
        manifest.save_to_path(&manifest_path)?;
        fs::create_dir_all(self.root.join("collections"))?;

        for record in &records {
            if plan.owns(collection_name_for_wal_record(record)) {
                self.apply_wal_record(record)?;
            }
        }
        Ok(())
    }

    fn apply_wal_record(&mut self, record: &WalRecord) -> io::Result<()> {
        match record {
            WalRecord::CreateCollection { collection, schema } => {
                self.create_collection_with_schema_internal(collection, schema, false)
            }
            WalRecord::DropCollection { collection } => {
                let paths = self.collection_paths(collection);
                if !paths.dir.exists() {
                    return Ok(());
                }
                self.drop_collection_internal(collection, false)
            }
            WalRecord::Insert {
                collection,
                ids,
                vectors,
            } => self
                .insert_internal(collection, ids, vectors, false)
                .map(|_| ()),
            WalRecord::InsertDocuments {
                collection,
                documents,
            } => self
                .insert_documents_internal(collection, documents, false)
                .map(|_| ()),
            WalRecord::UpsertDocuments {
                collection,
                documents,
            } => self
                .upsert_documents_internal(collection, documents, false)
                .map(|_| ()),
            WalRecord::Delete { collection, ids } => {
                self.delete_internal(collection, ids, false).map(|_| ())
            }
            WalRecord::CompactCollection {
                collection_name, ..
            } => self.compact_collection_internal(collection_name, false),
            WalRecord::UpdateDocuments {
                collection,
                updates,
            } => self
                .update_documents_internal(collection, updates, false)
                .map(|_| ()),
            WalRecord::AddColumn { collection, field } => {
                self.add_column_internal(collection, field.clone(), false)
            }
            WalRecord::DropColumn {
                collection,
                field_name,
            } => self.drop_column_internal(collection, field_name, false),
            WalRecord::AlterColumn {
                collection,
                old_name,
                new_name,
            } => self.alter_column_internal(collection, old_name, new_name, false),
            WalRecord::AddVectorField { collection, field } => {
                self.add_vector_field_internal(collection, field.clone(), false)
            }
            WalRecord::DropVectorField {
                collection,
                field_name,
            } => self.drop_vector_field_internal(collection, field_name, false),
        }
    }
}

impl CollectionHandle {
    fn new(
        name: String,
        root: PathBuf,
        segment_manager: SegmentManager,
        version_set: VersionSet,
        index_registry: Arc<IndexRegistry>,
    ) -> Self {
        Self {
            name,
            root,
            segment_manager,
            version_set: RwLock::new(version_set),
            index_registry,
            search_cache: Mutex::new(HashMap::new()),
            scalar_cache: Mutex::new(HashMap::new()),
            sparse_index_cache: Mutex::new(HashMap::new()),
            document_cache: Mutex::new(None),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn search(&self, query: &[f32], top_k: usize) -> io::Result<Vec<SearchHit>> {
        self.search_with_ef(query, top_k, DEFAULT_EF_SEARCH)
    }

    pub fn search_with_ef(
        &self,
        query: &[f32],
        top_k: usize,
        _ef_search: usize,
    ) -> io::Result<Vec<SearchHit>> {
        let field_name = self.collection_primary_vector_name()?;
        self.search_field_with_ef_internal(&field_name, query, top_k, _ef_search)
    }

    /// Brute-force search over a sparse vector field.
    ///
    /// Returns top-k hits ranked by sparse inner product (descending similarity).
    pub fn search_sparse(
        &self,
        field_name: &str,
        query: &SparseVector,
        top_k: usize,
    ) -> io::Result<Vec<SearchHit>> {
        let paths = self.collection_paths();
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;

        // Verify the field is a sparse vector field in the schema.
        collection_meta
            .vectors
            .iter()
            .find(|v| {
                v.name == field_name && v.data_type == crate::document::FieldType::VectorSparse
            })
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "sparse vector field '{}' is not defined in collection '{}'",
                        field_name, collection_meta.name
                    ),
                )
            })?;

        // Try cached sparse index first.
        {
            let sparse_cache = self
                .sparse_index_cache
                .lock()
                .expect("sparse index cache mutex poisoned");
            if let Some(index) = sparse_cache.get(field_name) {
                let query_data = SparseVectorData::new(query.indices.clone(), query.values.clone());
                let hits = index.search(&query_data, top_k, None).map_err(|e| {
                    io::Error::new(io::ErrorKind::Other, format!("sparse index search: {e:?}"))
                })?;
                return Ok(hits
                    .into_iter()
                    .map(|hit| SearchHit {
                        id: hit.id,
                        distance: -hit.score, // score → distance (negate for consistency)
                    })
                    .collect());
            }
        }

        // Fallback: brute-force scan.
        let segment_paths = self.segment_manager.segment_paths()?;

        // Collect all live sparse vectors for the target field across segments.
        let mut all_sparse: Vec<SparseVector> = Vec::new();
        let mut all_ids: Vec<i64> = Vec::new();
        let mut shadowed: HashSet<i64> = HashSet::new();

        for segment in &segment_paths {
            let stored_ids = load_record_ids_or_empty(&segment.external_ids)?;
            let sparse_rows =
                load_sparse_vectors_or_empty(&segment.sparse_vectors, stored_ids.len())?;
            let tombstone = TombstoneMask::load_from_path(&segment.tombstones)?;

            for row_idx in (0..stored_ids.len()).rev() {
                let ext_id = stored_ids[row_idx];
                if !shadowed.insert(ext_id) {
                    continue;
                }
                if tombstone.is_deleted(row_idx) {
                    continue;
                }
                if let Some(sv) = sparse_rows[row_idx].get(field_name) {
                    all_sparse.push(sv.clone());
                    all_ids.push(ext_id);
                }
            }
        }

        let tombstone = TombstoneMask::new(all_ids.len());
        search_sparse_bruteforce(&all_sparse, &all_ids, &tombstone, query, top_k)
    }

    pub fn query_with_context(&self, context: &QueryContext) -> io::Result<Vec<DocumentHit>> {
        let paths = self.collection_paths();
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let index_catalog = IndexCatalog::load_from_path(&paths.index_catalog)?;
        let query_by_id_documents = match context.query_by_id.as_ref() {
            Some(query_by_id) => self.fetch_documents(query_by_id)?,
            None => Vec::new(),
        };
        let plan = QueryPlanner::build(
            &collection_meta,
            &index_catalog,
            context,
            &query_by_id_documents,
        )?;
        match plan {
            QueryPlan::LegacySingleVector(plan) => {
                let ef_search = plan.ef_search.unwrap_or(DEFAULT_EF_SEARCH);
                let nprobe = plan.nprobe.unwrap_or(DEFAULT_NPROBE);

                // Resolve the effective search parameter: when nprobe is explicitly
                // provided, it overrides ef_search for IVF indexes (the IVF backend
                // maps the ef_search argument to nprobe internally).
                let state = self.cached_search_state_for_field(&plan.field_name)?;
                let is_ivf =
                    state.descriptor.kind == hannsdb_index::descriptor::VectorIndexKind::Ivf;
                let effective_search_param = if is_ivf && plan.nprobe.is_some() {
                    nprobe
                } else {
                    ef_search
                };

                #[cfg(feature = "hanns-backend")]
                if let Some(filter) = &plan.filter {
                    // Pre-filter path: construct BitsetView from filter evaluation
                    // and pass it to the ANN backend for efficient pre-filtered search.
                    if let Some(optimized) = state.optimized_ann.as_ref() {
                        // Fetch payloads for all indexed vectors to evaluate filter
                        let documents = self.fetch_documents(&optimized.ann_external_ids)?;
                        let bitset =
                            hannsdb_index::bitset::filter_to_bitset(documents.len(), |i| {
                                filter.matches(&documents[i].fields)
                            });
                        let ann_hits = match &plan.vector {
                            crate::query::QueryVector::Dense(v) => ann_search_with_bitset(
                                optimized.backend.as_ref(),
                                &optimized.ann_external_ids,
                                &optimized.metric,
                                v,
                                plan.top_k,
                                effective_search_param,
                                &bitset,
                            )?,
                            crate::query::QueryVector::Sparse(_) => {
                                return Err(io::Error::new(
                                    io::ErrorKind::InvalidInput,
                                    "sparse queries cannot use ANN pre-filter path",
                                ));
                            }
                        };
                        // Hits already have correct external IDs; fetch for output fields
                        let fetched = self
                            .fetch_documents(&ann_hits.iter().map(|h| h.id).collect::<Vec<_>>())?;
                        let mut doc_hits: Vec<DocumentHit> = ann_hits
                            .into_iter()
                            .zip(fetched)
                            .map(|(hit, document)| DocumentHit {
                                id: hit.id,
                                distance: hit.distance,
                                fields: document.fields,
                                vectors: BTreeMap::new(),
                                sparse_vectors: BTreeMap::new(),
                                group_key: None,
                            })
                            .collect();
                        project_document_hits(&mut doc_hits, plan.output_fields.as_deref());
                        return Ok(doc_hits);
                    }
                }

                // Fallback: overfetch + post-filter (non-hanns-backend or no ANN)
                const FILTER_OVERFETCH_FACTOR: usize = 4;
                let search_k = if plan.filter.is_some() {
                    plan.top_k.saturating_mul(FILTER_OVERFETCH_FACTOR)
                } else {
                    plan.top_k
                };
                let query_vec = match &plan.vector {
                    crate::query::QueryVector::Dense(v) => v.clone(),
                    crate::query::QueryVector::Sparse(_) => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "sparse queries cannot use the legacy single-vector fast path",
                        ));
                    }
                };
                let mut hits = self.query_documents_for_field_with_ef_internal(
                    &collection_meta,
                    &plan.field_name,
                    &query_vec,
                    search_k,
                    effective_search_param,
                    context.include_vector,
                )?;

                if let Some(filter) = &plan.filter {
                    hits.retain(|hit| filter.matches(&hit.fields));
                    hits.truncate(plan.top_k);
                }

                if let Some(order_by) = &plan.order_by {
                    sort_document_hits_by_field(
                        &mut hits,
                        &order_by.field_name,
                        order_by.descending,
                    );
                }

                project_document_hits(&mut hits, plan.output_fields.as_deref());
                Ok(hits)
            }
            QueryPlan::BruteForce(plan) => {
                let mut hits =
                    QueryExecutor::execute(&self.segment_manager, &collection_meta, &plan)?;
                if context.include_vector {
                    self.materialize_document_hit_vectors(&collection_meta, &mut hits)?;
                }
                project_document_hits(&mut hits, plan.output_fields.as_deref());
                Ok(hits)
            }
        }
    }

    pub fn optimize(&self) -> io::Result<()> {
        let _index_registry = Arc::clone(&self.index_registry);
        let paths = self.collection_paths();
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let index_catalog = IndexCatalog::load_from_path(&paths.index_catalog)?;

        let mut cache = self
            .search_cache
            .lock()
            .expect("search cache mutex poisoned");

        for vector_schema in &collection_meta.vectors {
            let field_name = &vector_schema.name;
            let Some(descriptor) =
                resolve_vector_descriptor_for_field(&collection_meta, &index_catalog, field_name)?
            else {
                continue;
            };

            let mut state = self.load_search_state_for_field(field_name)?;
            let mut blob_bytes = None;
            state.optimized_ann = Some(build_optimized_ann_state(&state, Some(&mut blob_bytes))?);

            #[cfg(feature = "hanns-backend")]
            if let Some(ref bytes) = blob_bytes {
                let blob_path = ann_blob_path(&paths.dir, field_name);
                if let Some(parent) = blob_path.parent() {
                    if let Err(e) = fs::create_dir_all(parent) {
                        log::warn!("Failed to create ANN dir for field '{}': {e}", field_name);
                        cache.insert(state.field_name.clone(), state);
                        continue;
                    }
                }
                if let Err(e) = fs::write(&blob_path, &bytes) {
                    log::warn!("Failed to persist ANN index for '{}': {e}", field_name);
                } else {
                    log::info!(
                        "Persisted ANN index ({} bytes) for field '{}' in collection '{}'",
                        bytes.len(),
                        field_name,
                        self.name
                    );
                    // Persist external IDs alongside the blob so the ANN can be
                    // loaded without the original records.bin / ids.bin files.
                    if let Some(ref ann) = state.optimized_ann {
                        let ids_path = ann_ids_path(&paths.dir, field_name);
                        let ids_bytes: Vec<u8> = ann
                            .ann_external_ids
                            .iter()
                            .flat_map(|id| id.to_le_bytes())
                            .collect();
                        if let Err(e) = fs::write(&ids_path, &ids_bytes) {
                            log::warn!("Failed to persist ANN ids for '{}': {e}", field_name);
                        }
                    }
                }
            }

            cache.insert(state.field_name.clone(), state);

            // Suppress unused-variable warnings when hanns-backend is off
            let _ = (&descriptor, &blob_bytes);
        }

        // Build scalar indexes for fields that have descriptors in the catalog.
        if !index_catalog.scalar_indexes.is_empty() {
            let segment_paths = self.segment_manager.segment_paths()?;
            let mut all_payloads: Vec<BTreeMap<String, FieldValue>> = Vec::new();
            let mut all_ids: Vec<i64> = Vec::new();
            for segment in &segment_paths {
                let stored_ids = load_record_ids_or_empty(&segment.external_ids)?;
                let payloads = load_payloads_or_empty(&segment.payloads, stored_ids.len())?;
                let tombstone = TombstoneMask::load_from_path(&segment.tombstones)?;
                for (row_idx, ext_id) in stored_ids.iter().enumerate() {
                    if tombstone.is_deleted(row_idx) {
                        continue;
                    }
                    all_payloads.push(payloads[row_idx].clone());
                    all_ids.push(*ext_id);
                }
            }

            // Convert FieldValue payloads to ScalarValue payloads once.
            let scalar_payloads: Vec<BTreeMap<String, ScalarValue>> = all_payloads
                .iter()
                .map(|map| {
                    map.iter()
                        .map(|(k, v)| (k.clone(), field_value_to_scalar(v)))
                        .collect()
                })
                .collect();

            let mut scalar_cache = self
                .scalar_cache
                .lock()
                .expect("scalar cache mutex poisoned");
            for scalar_descriptor in &index_catalog.scalar_indexes {
                let field_name = &scalar_descriptor.field_name;
                let index = InvertedScalarIndex::build_from_payloads(
                    scalar_descriptor.clone(),
                    field_name,
                    &scalar_payloads,
                    &all_ids,
                );
                log::info!(
                    "Built scalar index for field '{}' in collection '{}' ({} indexed IDs)",
                    field_name,
                    self.name,
                    index.all_indexed_ids().len(),
                );
                scalar_cache.insert(field_name.clone(), index);
            }
        }

        // Build sparse indexes for sparse vector fields.
        use crate::document::FieldType;
        use hannsdb_index::descriptor::{SparseIndexDescriptor, SparseIndexKind};
        use hannsdb_index::factory::DefaultIndexFactory;

        let sparse_fields: Vec<_> = collection_meta
            .vectors
            .iter()
            .filter(|v| v.data_type == FieldType::VectorSparse)
            .collect();

        if !sparse_fields.is_empty() {
            let segment_paths = self.segment_manager.segment_paths()?;

            for vector_schema in &sparse_fields {
                let field_name = &vector_schema.name;
                let mut all_sparse: Vec<(i64, SparseVectorData)> = Vec::new();
                let mut shadowed: HashSet<i64> = HashSet::new();

                for segment in &segment_paths {
                    let stored_ids = load_record_ids_or_empty(&segment.external_ids)?;
                    let sparse_rows =
                        load_sparse_vectors_or_empty(&segment.sparse_vectors, stored_ids.len())?;
                    let tombstone = TombstoneMask::load_from_path(&segment.tombstones)?;

                    for row_idx in (0..stored_ids.len()).rev() {
                        let ext_id = stored_ids[row_idx];
                        if !shadowed.insert(ext_id) {
                            continue;
                        }
                        if tombstone.is_deleted(row_idx) {
                            continue;
                        }
                        if let Some(sv) = sparse_rows[row_idx].get(field_name) {
                            all_sparse.push((
                                ext_id,
                                SparseVectorData::new(sv.indices.clone(), sv.values.clone()),
                            ));
                        }
                    }
                }

                if all_sparse.is_empty() {
                    continue;
                }

                let descriptor = SparseIndexDescriptor {
                    field_name: field_name.clone(),
                    kind: SparseIndexKind::SparseInverted,
                    metric: Some("ip".to_string()),
                    params: serde_json::Value::Object(serde_json::Map::new()),
                };

                let factory = DefaultIndexFactory::default();
                let mut index = factory
                    .create_sparse_index(&descriptor, None)
                    .map_err(|e| {
                        io::Error::new(io::ErrorKind::Other, format!("sparse index create: {e:?}"))
                    })?;

                index.add(&all_sparse).map_err(|e| {
                    io::Error::new(io::ErrorKind::Other, format!("sparse index build: {e:?}"))
                })?;

                // Pass BM25 params from the vector field schema to the sparse index.
                if let Some(bm25_params) = &vector_schema.bm25_params {
                    index.set_bm25_params(bm25_params.k1, bm25_params.b, bm25_params.avgdl);
                    log::info!(
                        "Set BM25 params (k1={}, b={}, avgdl={}) on sparse index for field '{}' in collection '{}'",
                        bm25_params.k1, bm25_params.b, bm25_params.avgdl,
                        field_name,
                        self.name,
                    );
                }

                log::info!(
                    "Built sparse index for field '{}' in collection '{}' ({} vectors)",
                    field_name,
                    self.name,
                    index.len(),
                );

                // Persist to disk.
                if let Ok(Some(bytes)) = index.serialize_to_bytes() {
                    let blob_path = paths
                        .dir
                        .join("ann")
                        .join(format!("{field_name}.sparse.bin"));
                    if let Some(parent) = blob_path.parent() {
                        if let Err(e) = fs::create_dir_all(parent) {
                            log::warn!(
                                "Failed to create ANN dir for sparse field '{}': {e}",
                                field_name
                            );
                        }
                    }
                    if let Err(e) = fs::write(&blob_path, &bytes) {
                        log::warn!("Failed to persist sparse index for '{}': {e}", field_name);
                    } else {
                        log::info!(
                            "Persisted sparse index ({} bytes) for field '{}' in collection '{}'",
                            bytes.len(),
                            field_name,
                            self.name
                        );
                    }
                }

                let mut sparse_cache = self
                    .sparse_index_cache
                    .lock()
                    .expect("sparse index cache mutex poisoned");
                sparse_cache.insert(field_name.clone(), index);
            }
        }

        Ok(())
    }

    pub fn version_set(&self) -> io::Result<VersionSet> {
        Ok(self
            .version_set
            .read()
            .expect("version_set rwlock poisoned")
            .clone())
    }

    fn collection_info(&self) -> io::Result<CollectionInfo> {
        let paths = self.collection_paths();
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;

        let mut record_count = 0usize;
        let mut deleted_count = 0usize;
        for segment in self.segment_manager.segment_paths()? {
            let metadata = SegmentMetadata::load_from_path(&segment.metadata)?;
            record_count += metadata.record_count;
            deleted_count += metadata.deleted_count;
        }
        let live_count = record_count.saturating_sub(deleted_count);

        // Build index_completeness: check the search cache for each vector field.
        // If the cache has an entry with optimized_ann populated, that field is fully indexed.
        let mut index_completeness = BTreeMap::new();
        let cache = self
            .search_cache
            .lock()
            .expect("search cache mutex poisoned");
        for vector_schema in &collection_meta.vectors {
            let completeness = if live_count == 0 {
                // No data: vacuously fully indexed.
                1.0
            } else if let Some(state) = cache.get(&vector_schema.name) {
                if state.optimized_ann.is_some() {
                    1.0
                } else {
                    0.0
                }
            } else {
                0.0
            };
            index_completeness.insert(vector_schema.name.clone(), completeness);
        }

        Ok(CollectionInfo {
            name: collection_meta.name,
            dimension: collection_meta.dimension,
            metric: collection_meta.metric,
            record_count,
            deleted_count,
            live_count,
            index_completeness,
        })
    }

    fn flush(&self) -> io::Result<()> {
        let paths = self.collection_paths();
        let _ = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let _ = SegmentMetadata::load_from_path(&paths.segment_meta)?;
        let _ = TombstoneMask::load_from_path(&paths.tombstones)?;
        let _ = load_wal_records(&wal_path(&self.root))?;
        Ok(())
    }

    fn list_segments(&self) -> io::Result<Vec<CollectionSegmentInfo>> {
        let paths = self.collection_paths();
        let _ = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        self.segment_manager
            .segment_paths()?
            .into_iter()
            .map(|segment| {
                let metadata = SegmentMetadata::load_from_path(&segment.metadata)?;
                Ok(CollectionSegmentInfo {
                    id: segment.segment_id.clone(),
                    live_count: metadata.record_count.saturating_sub(metadata.deleted_count),
                    dead_count: metadata.deleted_count,
                    ann_ready: segment.ann_dir().exists(),
                })
            })
            .collect()
    }

    fn query_documents(
        &self,
        query: &[f32],
        top_k: usize,
        filter: Option<&str>,
    ) -> io::Result<Vec<DocumentHit>> {
        self.query_documents_with_ef(query, top_k, filter, DEFAULT_EF_SEARCH)
    }

    fn query_documents_with_ef(
        &self,
        query: &[f32],
        top_k: usize,
        filter: Option<&str>,
        ef_search: usize,
    ) -> io::Result<Vec<DocumentHit>> {
        self.query_documents_with_ef_internal(None, query, top_k, filter, ef_search, false)
    }

    fn query_documents_with_ef_internal(
        &self,
        collection_meta: Option<&CollectionMetadata>,
        query: &[f32],
        top_k: usize,
        filter: Option<&str>,
        ef_search: usize,
        include_vector: bool,
    ) -> io::Result<Vec<DocumentHit>> {
        match filter.map(str::trim) {
            None | Some("") => {
                let field_name = match collection_meta {
                    Some(collection_meta) => collection_meta.primary_vector.clone(),
                    None => self.collection_primary_vector_name()?,
                };
                let hits =
                    self.search_field_with_ef_internal(&field_name, query, top_k, ef_search)?;
                let documents =
                    self.fetch_documents(&hits.iter().map(|hit| hit.id).collect::<Vec<_>>())?;
                if include_vector && documents.len() != hits.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::NotFound,
                        format!(
                            "failed to resolve {} typed query hit(s) for vector materialization, got {}",
                            hits.len(),
                            documents.len()
                        ),
                    ));
                }
                Ok(hits
                    .into_iter()
                    .zip(documents)
                    .map(|(hit, document)| {
                        let vectors =
                            if include_vector {
                                document.vectors_with_primary(
                                collection_meta
                                    .expect("collection metadata must exist when include_vector")
                                    .primary_vector
                                    .as_str(),
                            ).clone()
                            } else {
                                BTreeMap::new()
                            };
                        DocumentHit {
                            id: hit.id,
                            distance: hit.distance,
                            fields: document.fields,
                            vectors,
                            sparse_vectors: BTreeMap::new(),
                            group_key: None,
                        }
                    })
                    .collect())
            }
            Some(filter) => {
                let mut hits = self.query_documents_with_filter(query, top_k, filter)?;
                if include_vector {
                    let collection_meta = collection_meta
                        .expect("collection metadata must exist when include_vector");
                    self.materialize_document_hit_vectors(collection_meta, &mut hits)?;
                }
                Ok(hits)
            }
        }
    }

    fn query_documents_for_field_with_ef_internal(
        &self,
        collection_meta: &CollectionMetadata,
        field_name: &str,
        query: &[f32],
        top_k: usize,
        ef_search: usize,
        include_vector: bool,
    ) -> io::Result<Vec<DocumentHit>> {
        let hits = self.search_field_with_ef_internal(field_name, query, top_k, ef_search)?;
        let documents = self.fetch_documents(&hits.iter().map(|hit| hit.id).collect::<Vec<_>>())?;
        if include_vector && documents.len() != hits.len() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "failed to resolve {} typed query hit(s) for vector materialization, got {}",
                    hits.len(),
                    documents.len()
                ),
            ));
        }
        Ok(hits
            .into_iter()
            .zip(documents)
            .map(|(hit, document)| {
                let vectors = if include_vector {
                    document
                        .vectors_with_primary(collection_meta.primary_vector.as_str())
                        .clone()
                } else {
                    BTreeMap::new()
                };
                DocumentHit {
                    id: hit.id,
                    distance: hit.distance,
                    fields: document.fields,
                    vectors,
                    sparse_vectors: BTreeMap::new(),
                    group_key: None,
                }
            })
            .collect())
    }

    fn search_field_with_ef_internal(
        &self,
        field_name: &str,
        query: &[f32],
        top_k: usize,
        ef_search: usize,
    ) -> io::Result<Vec<SearchHit>> {
        let state = self.cached_search_state_for_field(field_name)?;
        let ef_search = ef_search.max(1);
        if let Some(optimized_ann) = state.optimized_ann.as_ref() {
            let optimized_snapshot = (
                Arc::clone(&optimized_ann.backend),
                Arc::clone(&optimized_ann.ann_external_ids),
                optimized_ann.metric.clone(),
            );
            return ann_search(
                optimized_snapshot.0.as_ref(),
                optimized_snapshot.1.as_slice(),
                &optimized_snapshot.2,
                query,
                top_k,
                ef_search,
            );
        }

        search_by_metric(
            &state.records,
            &state.external_ids,
            state.dimension,
            state.tombstone.as_ref(),
            query,
            top_k,
            &state.metric,
        )
    }

    fn fetch_documents(&self, external_ids: &[i64]) -> io::Result<Vec<Document>> {
        let state = self.cached_document_state()?;
        let docs = &*state.documents;
        Ok(external_ids
            .iter()
            .filter_map(|id| docs.get(id).cloned())
            .collect())
    }

    fn materialize_document_hit_vectors(
        &self,
        collection_meta: &CollectionMetadata,
        hits: &mut [DocumentHit],
    ) -> io::Result<()> {
        if hits.is_empty() {
            return Ok(());
        }

        let documents = self.fetch_documents(&hits.iter().map(|hit| hit.id).collect::<Vec<_>>())?;
        if documents.len() != hits.len() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "failed to resolve one or more typed query hits for vector materialization",
            ));
        }

        for (hit, document) in hits.iter_mut().zip(documents) {
            hit.vectors = document
                .vectors_with_primary(collection_meta.primary_vector.as_str())
                .clone();
            hit.sparse_vectors = document.sparse_vectors.clone();
        }
        Ok(())
    }

    fn query_documents_with_filter(
        &self,
        query: &[f32],
        top_k: usize,
        filter: &str,
    ) -> io::Result<Vec<DocumentHit>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let filter_expr = parse_filter(filter)?;
        let paths = self.collection_paths();
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let segment_paths = self.segment_manager.segment_paths()?;

        // Compute the minimal set of payload fields needed for filter evaluation.
        let filter_fields: Vec<String> = filter_expr.referenced_fields().into_iter().collect();
        let projection = if filter_fields.is_empty() {
            None
        } else {
            Some(filter_fields.as_slice())
        };

        // Try to extract candidate IDs from scalar indexes.
        // For now we handle the simplest case: a single Eq or InList clause
        // on a field that has a scalar index.
        let indexed_candidates = self.try_scalar_index_candidates(&filter_expr);

        let mut heap: BinaryHeap<RankedDocumentHit> = BinaryHeap::new();
        for segment in segment_paths {
            let records = load_records_or_empty(
                &segment.records,
                collection_meta.dimension,
                collection_meta.primary_is_fp16(),
            )?;
            let stored_ids = load_record_ids_or_empty(&segment.external_ids)?;
            let payloads = if projection.is_some() {
                load_payloads_with_fields_or_empty(&segment.payloads, stored_ids.len(), projection)?
            } else {
                load_payloads_or_empty(&segment.payloads, stored_ids.len())?
            };
            let tombstone = TombstoneMask::load_from_path(&segment.tombstones)?;

            if stored_ids.len().saturating_mul(collection_meta.dimension) != records.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "records and ids are not aligned",
                ));
            }

            for (row_idx, vector) in records.chunks_exact(collection_meta.dimension).enumerate() {
                if tombstone.is_deleted(row_idx) {
                    continue;
                }
                let ext_id = stored_ids[row_idx];

                // If the scalar index gave us a candidate set, skip rows
                // whose external ID is not in the set.
                if let Some(ref candidates) = indexed_candidates {
                    if !candidates.contains(&ext_id) {
                        continue;
                    }
                }

                let fields = &payloads[row_idx];
                if !filter_expr.matches(fields) {
                    continue;
                }
                let candidate = RankedDocumentHit {
                    hit: DocumentHit {
                        id: ext_id,
                        distance: distance_by_metric(query, vector, &collection_meta.metric)?,
                        fields: fields.clone(),
                        vectors: BTreeMap::new(),
                        sparse_vectors: BTreeMap::new(),
                        group_key: None,
                    },
                };

                if heap.len() < top_k {
                    heap.push(candidate);
                    continue;
                }

                if let Some(worst) = heap.peek() {
                    if candidate.cmp(worst) == Ordering::Less {
                        let _ = heap.pop();
                        heap.push(candidate);
                    }
                }
            }
        }

        let mut hits = heap.into_iter().map(|entry| entry.hit).collect::<Vec<_>>();
        hits.sort_by(|a, b| {
            a.distance
                .total_cmp(&b.distance)
                .then_with(|| a.id.cmp(&b.id))
        });
        Ok(hits)
    }

    /// Attempt to use scalar indexes to narrow down candidate IDs.
    ///
    /// Returns `Some(set)` if an index-accelerated candidate set was produced,
    /// or `None` if no scalar index could be applied (fall back to brute-force).
    fn try_scalar_index_candidates(&self, filter_expr: &FilterExpr) -> Option<BTreeSet<i64>> {
        let scalar_cache = self.scalar_cache.lock().ok()?;
        if scalar_cache.is_empty() {
            return None;
        }

        match filter_expr {
            FilterExpr::Clause { field, op, value } => {
                let index = scalar_cache.get(field)?;
                let scalar_value = field_value_to_scalar(value);
                match op {
                    ComparisonOp::Eq => Some(index.lookup_eq(&scalar_value)),
                    ComparisonOp::Ne => Some(index.lookup_range(RangeOp::Ne, &scalar_value)),
                    ComparisonOp::Gt => Some(index.lookup_range(RangeOp::Gt, &scalar_value)),
                    ComparisonOp::Gte => Some(index.lookup_range(RangeOp::Gte, &scalar_value)),
                    ComparisonOp::Lt => Some(index.lookup_range(RangeOp::Lt, &scalar_value)),
                    ComparisonOp::Lte => Some(index.lookup_range(RangeOp::Lte, &scalar_value)),
                }
            }
            FilterExpr::InList {
                field,
                negated,
                values,
            } => {
                let index = scalar_cache.get(field)?;
                let scalar_values: Vec<ScalarValue> =
                    values.iter().map(field_value_to_scalar).collect();
                Some(index.lookup_in(&scalar_values, *negated))
            }
            // AND: intersect candidates from each sub-expression.
            FilterExpr::And(exprs) => {
                let mut result: Option<BTreeSet<i64>> = None;
                for expr in exprs {
                    let sub = self.try_scalar_index_candidates(expr);
                    match (result.take(), sub) {
                        (None, Some(s)) => result = Some(s),
                        (Some(r), Some(s)) => {
                            let intersection: BTreeSet<i64> = r.intersection(&s).cloned().collect();
                            result = Some(intersection);
                        }
                        (prior, None) => result = prior,
                    }
                }
                result
            }
            // OR: union candidates from each sub-expression.
            FilterExpr::Or(exprs) => {
                let mut result: Option<BTreeSet<i64>> = None;
                for expr in exprs {
                    let sub = self.try_scalar_index_candidates(expr);
                    match (result.take(), sub) {
                        (None, Some(s)) => result = Some(s),
                        (Some(r), Some(s)) => {
                            let union: BTreeSet<i64> = r.union(&s).cloned().collect();
                            result = Some(union);
                        }
                        (prior, None) => result = prior,
                    }
                }
                result
            }
            // NOT and NullCheck are harder to accelerate; fall back.
            _ => None,
        }
    }

    fn collection_paths(&self) -> CollectionPaths {
        collection_paths_for_root(&self.root, &self.name)
    }

    fn collection_primary_vector_name(&self) -> io::Result<String> {
        let paths = self.collection_paths();
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        Ok(collection_meta.primary_vector)
    }

    fn load_search_state_for_field(&self, field_name: &str) -> io::Result<CachedSearchState> {
        let paths = self.collection_paths();
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let index_catalog = IndexCatalog::load_from_path(&paths.index_catalog)?;
        let descriptor = resolve_vector_descriptor_for_field(&collection_meta, &index_catalog, field_name)?
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!(
                        "vector field '{}' does not have a resolvable index descriptor in collection '{}'",
                        field_name, collection_meta.name
                    ),
                )
            })?;
        let vector_schema = collection_meta
            .vectors
            .iter()
            .find(|vector| vector.name == field_name)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "vector field '{}' is not defined in collection '{}'",
                        field_name, collection_meta.name
                    ),
                )
            })?;

        let (records, external_ids) = if field_name == collection_meta.primary_vector {
            load_shadowed_live_records(
                &self.segment_manager,
                vector_schema.dimension,
                collection_meta.primary_is_fp16(),
            )?
        } else {
            load_shadowed_live_vector_records(
                &self.segment_manager,
                vector_schema.dimension,
                field_name,
            )?
        };

        Ok(CachedSearchState {
            records: Arc::new(records),
            external_ids: Arc::new(external_ids),
            tombstone: Arc::new(TombstoneMask::new(0)),
            dimension: vector_schema.dimension,
            metric: descriptor
                .metric
                .clone()
                .unwrap_or_else(|| collection_meta.metric.clone())
                .to_ascii_lowercase(),
            field_name: field_name.to_string(),
            descriptor,
            optimized_ann: None,
        })
    }

    fn cached_search_state_for_field(&self, field_name: &str) -> io::Result<CachedSearchState> {
        if let Some(state) = self
            .search_cache
            .lock()
            .expect("search cache mutex poisoned")
            .get(field_name)
            .cloned()
        {
            return Ok(state);
        }

        let state = self
            .try_load_persisted_ann_state(field_name)?
            .unwrap_or(self.load_search_state_for_field(field_name)?);

        let mut cache = self
            .search_cache
            .lock()
            .expect("search cache mutex poisoned");
        if let Some(state) = cache.get(field_name) {
            return Ok(state.clone());
        }
        cache.insert(field_name.to_string(), state.clone());
        Ok(state)
    }

    fn cached_document_state(&self) -> io::Result<CachedDocumentState> {
        {
            let cache = self
                .document_cache
                .lock()
                .expect("document cache mutex poisoned");
            if let Some(ref state) = *cache {
                return Ok(state.clone());
            }
        }

        let state = self.build_document_cache()?;

        let mut cache = self
            .document_cache
            .lock()
            .expect("document cache mutex poisoned");
        if let Some(ref existing) = *cache {
            return Ok(existing.clone());
        }
        *cache = Some(state.clone());
        Ok(state)
    }

    fn build_document_cache(&self) -> io::Result<CachedDocumentState> {
        let paths = self.collection_paths();
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let index_catalog = IndexCatalog::load_from_path(&paths.index_catalog)?;
        let segment_paths = self.segment_manager.segment_paths()?;

        let mut documents: HashMap<i64, Document> = HashMap::new();
        let mut shadowed_ids: HashSet<i64> = HashSet::new();

        for segment in &segment_paths {
            let stored_ids = load_record_ids_or_empty(&segment.external_ids)?;
            let records = load_records_or_empty(
                &segment.records,
                collection_meta.dimension,
                collection_meta.primary_is_fp16(),
            )?;
            let payloads = load_payloads_or_empty(&segment.payloads, stored_ids.len())?;
            let vectors = load_vectors_or_empty(&segment.vectors, stored_ids.len())?;
            let sparse = load_sparse_vectors_or_empty(&segment.sparse_vectors, stored_ids.len())?;
            let tombstone = TombstoneMask::load_from_path(&segment.tombstones)?;

            if stored_ids.len().saturating_mul(collection_meta.dimension) != records.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "records and ids are not aligned",
                ));
            }

            for row_idx in (0..stored_ids.len()).rev() {
                let ext_id = stored_ids[row_idx];
                if !shadowed_ids.insert(ext_id) {
                    continue;
                }
                if tombstone.is_deleted(row_idx) {
                    continue;
                }
                let start = row_idx * collection_meta.dimension;
                let end = start + collection_meta.dimension;
                let mut doc_vectors = vectors[row_idx].clone();
                doc_vectors.insert(
                    collection_meta.primary_vector.clone(),
                    records[start..end].to_vec(),
                );
                documents.insert(
                    ext_id,
                    Document {
                        id: ext_id,
                        fields: payloads[row_idx].clone(),
                        vectors: doc_vectors,
                        sparse_vectors: sparse.get(row_idx).cloned().unwrap_or_default(),
                    },
                );
            }
        }

        Ok(CachedDocumentState {
            documents: Arc::new(documents),
            collection_meta: Arc::new(collection_meta),
            index_catalog: Arc::new(index_catalog),
        })
    }

    fn try_load_persisted_ann_state(
        &self,
        field_name: &str,
    ) -> io::Result<Option<CachedSearchState>> {
        #[cfg(not(feature = "hanns-backend"))]
        {
            let _ = field_name;
            return Ok(None);
        }

        #[cfg(feature = "hanns-backend")]
        {
            let paths = self.collection_paths();
            let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
            let index_catalog = IndexCatalog::load_from_path(&paths.index_catalog)?;
            let descriptor =
                resolve_vector_descriptor_for_field(&collection_meta, &index_catalog, field_name)?
                    .ok_or_else(|| {
                        io::Error::new(io::ErrorKind::NotFound, "no descriptor for field")
                    })?;

            // Try new per-field path first
            let new_path = ann_blob_path(&paths.dir, field_name);
            // Fall back to legacy single-file path for primary field
            let primary_name = &collection_meta.primary_vector;
            let legacy_path = paths.dir.join(HNSW_INDEX_FILE);
            let is_primary = field_name == primary_name;

            let blob_path = if new_path.exists() {
                new_path
            } else if is_primary && legacy_path.exists() {
                legacy_path
            } else {
                return Ok(None);
            };

            let vector_schema = collection_meta
                .vectors
                .iter()
                .find(|v| v.name == field_name)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("vector field '{}' not found", field_name),
                    )
                })?;

            let metric_lc = descriptor
                .metric
                .clone()
                .unwrap_or_else(|| collection_meta.metric.clone())
                .to_ascii_lowercase();

            match fs::read(&blob_path).and_then(|bytes| {
                DefaultIndexFactory::default()
                    .create_vector_index(vector_schema.dimension, &descriptor, Some(&bytes))
                    .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, format!("{err:?}")))
            }) {
                Ok(backend) => {
                    // Try to load persisted external IDs (avoids needing records.bin).
                    let ids_file = ann_ids_path(&paths.dir, field_name);
                    let ann_external_ids = if ids_file.exists() {
                        let raw = fs::read(&ids_file)?;
                        let ids: Vec<i64> = raw
                            .chunks_exact(8)
                            .map(|chunk| {
                                let buf: [u8; 8] = chunk.try_into().expect("chunk size");
                                i64::from_le_bytes(buf)
                            })
                            .collect();
                        Arc::new(ids)
                    } else {
                        // Legacy: fall back to loading from segment data.
                        let state = self.load_search_state_for_field(field_name)?;
                        Arc::clone(&state.external_ids)
                    };

                    let state = CachedSearchState {
                        records: Arc::new(Vec::new()),
                        external_ids: Arc::clone(&ann_external_ids),
                        tombstone: Arc::new(TombstoneMask::new(0)),
                        dimension: vector_schema.dimension,
                        metric: metric_lc.clone(),
                        field_name: field_name.to_string(),
                        descriptor,
                        optimized_ann: Some(OptimizedAnnState {
                            backend: Arc::from(backend),
                            ann_external_ids,
                            metric: metric_lc,
                        }),
                    };

                    log::info!(
                        "Loaded persisted ANN index for field '{}' in '{}' from '{}'",
                        field_name,
                        self.name,
                        blob_path.display()
                    );
                    Ok(Some(state))
                }
                Err(e) => {
                    log::warn!(
                        "Failed to load persisted ANN for field '{}' in '{}' from '{}': {e}",
                        field_name,
                        self.name,
                        blob_path.display()
                    );
                    Ok(None)
                }
            }
        }
    }

    fn invalidate_search_cache(&self) {
        let mut cache = self
            .search_cache
            .lock()
            .expect("search cache mutex poisoned");
        cache.clear();
        let mut doc_cache = self
            .document_cache
            .lock()
            .expect("document cache mutex poisoned");
        *doc_cache = None;
        let mut sparse_cache = self
            .sparse_index_cache
            .lock()
            .expect("sparse index cache mutex poisoned");
        sparse_cache.clear();
    }

    fn refresh_version_set(&self) -> io::Result<()> {
        let version_set = self.segment_manager.version_set()?;
        let mut snapshot = self
            .version_set
            .write()
            .expect("version_set rwlock poisoned");
        *snapshot = version_set;
        Ok(())
    }
}

fn project_document_hits(hits: &mut [DocumentHit], output_fields: Option<&[String]>) {
    let Some(output_fields) = output_fields else {
        return;
    };
    let requested = output_fields.iter().cloned().collect::<BTreeSet<_>>();
    for hit in hits {
        hit.fields
            .retain(|field_name, _| requested.contains(field_name));
    }
}

fn sort_document_hits_by_field(hits: &mut [DocumentHit], field_name: &str, descending: bool) {
    hits.sort_by(|a, b| {
        let av = a.fields.get(field_name);
        let bv = b.fields.get(field_name);
        let ord = match (av, bv) {
            (None, None) => std::cmp::Ordering::Equal,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (Some(_), None) => std::cmp::Ordering::Less,
            (Some(av), Some(bv)) => compare_field_values_for_sort(av, bv),
        };
        if descending { ord.reverse() } else { ord }.then_with(|| {
            a.distance
                .total_cmp(&b.distance)
                .then_with(|| a.id.cmp(&b.id))
        })
    });
}

fn compare_field_values_for_sort(a: &FieldValue, b: &FieldValue) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a, b) {
        (FieldValue::String(sa), FieldValue::String(sb)) => sa.cmp(sb),
        (FieldValue::Int64(va), FieldValue::Int64(vb)) => va.cmp(vb),
        (FieldValue::Int32(va), FieldValue::Int32(vb)) => va.cmp(vb),
        (FieldValue::UInt32(va), FieldValue::UInt32(vb)) => va.cmp(vb),
        (FieldValue::UInt64(va), FieldValue::UInt64(vb)) => va.cmp(vb),
        (FieldValue::Float(va), FieldValue::Float(vb)) => va.total_cmp(vb),
        (FieldValue::Float64(va), FieldValue::Float64(vb)) => va.total_cmp(vb),
        (FieldValue::Bool(va), FieldValue::Bool(vb)) => va.cmp(vb),
        _ => format!("{a:?}").cmp(&format!("{b:?}")),
    }
}

fn load_shadowed_live_records(
    segment_manager: &SegmentManager,
    dimension: usize,
    fp16: bool,
) -> io::Result<(Vec<f32>, Vec<i64>)> {
    let mut records = Vec::new();
    let mut external_ids = Vec::new();
    let mut shadowed_ids = HashSet::new();

    for segment in segment_manager.segment_paths()? {
        let segment_records = load_records_or_empty(&segment.records, dimension, fp16)?;
        let segment_external_ids = load_record_ids_or_empty(&segment.external_ids)?;
        let tombstone = TombstoneMask::load_from_path(&segment.tombstones)?;

        if segment_external_ids.len().saturating_mul(dimension) != segment_records.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "records and ids are not aligned",
            ));
        }

        for row_idx in (0..segment_external_ids.len()).rev() {
            let external_id = segment_external_ids[row_idx];
            if !shadowed_ids.insert(external_id) {
                continue;
            }
            if tombstone.is_deleted(row_idx) {
                continue;
            }
            let start = row_idx * dimension;
            let end = start + dimension;
            records.extend_from_slice(&segment_records[start..end]);
            external_ids.push(external_id);
        }
    }

    Ok((records, external_ids))
}

fn load_shadowed_live_vector_records(
    segment_manager: &SegmentManager,
    dimension: usize,
    field_name: &str,
) -> io::Result<(Vec<f32>, Vec<i64>)> {
    let mut records = Vec::new();
    let mut external_ids = Vec::new();
    let mut shadowed_ids = HashSet::new();

    for segment in segment_manager.segment_paths()? {
        let segment_external_ids = load_record_ids_or_empty(&segment.external_ids)?;
        let segment_vectors = load_vectors_or_empty(&segment.vectors, segment_external_ids.len())?;
        let tombstone = TombstoneMask::load_from_path(&segment.tombstones)?;
        if segment_vectors.len() != segment_external_ids.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "vector rows and ids are not aligned",
            ));
        }

        for row_idx in (0..segment_external_ids.len()).rev() {
            let external_id = segment_external_ids[row_idx];
            if !shadowed_ids.insert(external_id) {
                continue;
            }
            if tombstone.is_deleted(row_idx) {
                continue;
            }
            let Some(vector) = segment_vectors[row_idx].get(field_name) else {
                continue;
            };
            if vector.len() != dimension {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "vector field '{}' dimension mismatch: expected {}, got {}",
                        field_name,
                        dimension,
                        vector.len()
                    ),
                ));
            }
            records.extend_from_slice(vector);
            external_ids.push(external_id);
        }
    }

    Ok((records, external_ids))
}

fn manifest_path(root: &Path) -> PathBuf {
    root.join("manifest.json")
}

fn wal_path(root: &Path) -> PathBuf {
    root.join("wal.jsonl")
}

fn collection_paths_for_root(root: &Path, name: &str) -> CollectionPaths {
    let dir = root.join("collections").join(name);
    CollectionPaths {
        dir: dir.clone(),
        collection_meta: dir.join("collection.json"),
        index_catalog: dir.join(INDEX_CATALOG_FILE),
        segment_set: dir.join("segment_set.json"),
        segments_dir: dir.join("segments"),
        segment_meta: dir.join("segment.json"),
        records: dir.join("records.bin"),
        external_ids: dir.join("ids.bin"),
        payloads: dir.join("payloads.jsonl"),
        vectors: dir.join("vectors.jsonl"),
        sparse_vectors: dir.join("sparse_vectors.jsonl"),
        tombstones: dir.join("tombstones.json"),
    }
}

struct CollectionPaths {
    dir: PathBuf,
    collection_meta: PathBuf,
    index_catalog: PathBuf,
    segment_set: PathBuf,
    segments_dir: PathBuf,
    segment_meta: PathBuf,
    records: PathBuf,
    external_ids: PathBuf,
    payloads: PathBuf,
    vectors: PathBuf,
    sparse_vectors: PathBuf,
    tombstones: PathBuf,
}

#[derive(Debug, Clone)]
struct RankedDocumentHit {
    hit: DocumentHit,
}

impl PartialEq for RankedDocumentHit {
    fn eq(&self, other: &Self) -> bool {
        self.hit.id == other.hit.id && self.hit.distance.to_bits() == other.hit.distance.to_bits()
    }
}

impl Eq for RankedDocumentHit {}

impl PartialOrd for RankedDocumentHit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RankedDocumentHit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.hit
            .distance
            .total_cmp(&other.hit.distance)
            .then_with(|| self.hit.id.cmp(&other.hit.id))
    }
}

#[derive(Debug, Default)]
struct WalCollectionPlan {
    requires_data_files: bool,
    requires_vector_sidecar: bool,
}

#[derive(Debug, Default)]
struct WalReplayPlan {
    collections: HashMap<String, WalCollectionPlan>,
    dropped_collections: HashSet<String>,
}

impl WalReplayPlan {
    fn build(records: &[WalRecord]) -> Self {
        let mut collections = HashMap::new();
        let mut dropped_collections = HashSet::new();
        for record in records {
            match record {
                WalRecord::CreateCollection { collection, .. } => {
                    collections.insert(collection.clone(), WalCollectionPlan::default());
                    dropped_collections.remove(collection);
                }
                WalRecord::DropCollection { collection } => {
                    if collections.remove(collection).is_some() {
                        dropped_collections.insert(collection.clone());
                    }
                }
                WalRecord::Insert {
                    collection, ids, ..
                } if !ids.is_empty() => {
                    if let Some(plan) = collections.get_mut(collection) {
                        plan.requires_data_files = true;
                        plan.requires_vector_sidecar = true;
                    }
                }
                WalRecord::InsertDocuments {
                    collection,
                    documents,
                } if !documents.is_empty() => {
                    if let Some(plan) = collections.get_mut(collection) {
                        plan.requires_data_files = true;
                        plan.requires_vector_sidecar = true;
                    }
                }
                WalRecord::UpsertDocuments {
                    collection,
                    documents,
                } if !documents.is_empty() => {
                    if let Some(plan) = collections.get_mut(collection) {
                        plan.requires_data_files = true;
                        plan.requires_vector_sidecar = true;
                    }
                }
                WalRecord::Delete { collection, ids } if !ids.is_empty() => {
                    if let Some(plan) = collections.get_mut(collection) {
                        plan.requires_data_files = true;
                    }
                }
                WalRecord::CompactCollection {
                    collection_name, ..
                } => {
                    if let Some(plan) = collections.get_mut(collection_name) {
                        plan.requires_data_files = true;
                    }
                }
                _ => {}
            }
        }
        Self {
            collections,
            dropped_collections,
        }
    }

    fn requires_replay(&self, db: &HannsDb) -> io::Result<bool> {
        let manifest = ManifestMetadata::load_from_path(&manifest_path(&db.root))?;
        for (collection, plan) in &self.collections {
            let paths = db.collection_paths(collection);
            let segment_meta = if paths.segment_meta.exists() {
                Some(SegmentMetadata::load_from_path(&paths.segment_meta)?)
            } else {
                None
            };
            if !manifest.collections.iter().any(|entry| entry == collection) {
                return Ok(true);
            }
            if !paths.collection_meta.exists()
                || segment_meta.is_none()
                || !paths.tombstones.exists()
            {
                return Ok(true);
            }
            if plan.requires_data_files
                && (!paths.records.exists()
                    || !paths.external_ids.exists()
                    || !paths.payloads.exists())
            {
                return Ok(true);
            }
            if plan.requires_vector_sidecar {
                match load_vectors(&paths.vectors) {
                    Ok(vectors) => {
                        if Some(vectors.len())
                            != segment_meta.as_ref().map(|meta| meta.record_count)
                        {
                            return Ok(true);
                        }
                    }
                    Err(_) => return Ok(true),
                }
            }
        }
        for collection in &self.dropped_collections {
            let paths = db.collection_paths(collection);
            if manifest.collections.iter().any(|entry| entry == collection) || paths.dir.exists() {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn owned_collections(&self) -> impl Iterator<Item = &str> {
        self.collections
            .keys()
            .chain(self.dropped_collections.iter())
            .map(String::as_str)
    }

    fn has_owned_collections(&self) -> bool {
        !(self.collections.is_empty() && self.dropped_collections.is_empty())
    }

    fn owns(&self, collection: &str) -> bool {
        self.collections.contains_key(collection) || self.dropped_collections.contains(collection)
    }
}

fn collection_name_for_wal_record(record: &WalRecord) -> &str {
    match record {
        WalRecord::CreateCollection { collection, .. }
        | WalRecord::DropCollection { collection }
        | WalRecord::Insert { collection, .. }
        | WalRecord::InsertDocuments { collection, .. }
        | WalRecord::UpsertDocuments { collection, .. }
        | WalRecord::Delete { collection, .. }
        | WalRecord::UpdateDocuments { collection, .. }
        | WalRecord::AddColumn { collection, .. }
        | WalRecord::DropColumn { collection, .. }
        | WalRecord::AlterColumn { collection, .. }
        | WalRecord::AddVectorField { collection, .. }
        | WalRecord::DropVectorField { collection, .. } => collection,
        WalRecord::CompactCollection {
            collection_name, ..
        } => collection_name,
    }
}

/// Convert a core `FieldValue` into the index crate's `ScalarValue`.
fn field_value_to_scalar(value: &FieldValue) -> ScalarValue {
    match value {
        FieldValue::String(s) => ScalarValue::String(s.clone()),
        FieldValue::Int64(v) => ScalarValue::Int64(*v),
        FieldValue::Int32(v) => ScalarValue::Int64(*v as i64),
        FieldValue::UInt32(v) => ScalarValue::Int64(*v as i64),
        FieldValue::UInt64(v) => ScalarValue::Int64(*v as i64),
        FieldValue::Float(v) => ScalarValue::Float64(*v as f64),
        FieldValue::Float64(v) => ScalarValue::Float64(*v),
        FieldValue::Bool(b) => ScalarValue::Bool(*b),
        FieldValue::Array(items) => {
            // Flatten: use first element's scalar value if available.
            // Array fields are not directly indexable as scalar indexes.
            match items.first() {
                Some(first) => field_value_to_scalar(first),
                None => ScalarValue::String("[]".to_string()),
            }
        }
    }
}

fn next_compacted_segment_id<'a>(segment_ids: impl Iterator<Item = &'a String>) -> String {
    let max_id = segment_ids
        .filter_map(|segment_id| {
            segment_id
                .strip_prefix("seg-")
                .and_then(|suffix| suffix.parse::<u64>().ok())
        })
        .max()
        .unwrap_or(0);
    format!("seg-{:06}", max_id.saturating_add(1))
}

fn load_records_or_empty(path: &Path, dimension: usize, fp16: bool) -> io::Result<Vec<f32>> {
    let result = if fp16 {
        load_records_f16(path, dimension)
    } else {
        load_records(path, dimension)
    };
    match result {
        Ok(records) => Ok(records),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(err) => Err(err),
    }
}

fn load_wal_records_or_empty(path: &Path) -> io::Result<Vec<WalRecord>> {
    match load_wal_records(path) {
        Ok(records) => Ok(records),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(err) => Err(err),
    }
}

fn load_record_ids_or_empty(path: &Path) -> io::Result<Vec<i64>> {
    match load_record_ids(path) {
        Ok(ids) => Ok(ids),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(err) => Err(err),
    }
}

fn load_payloads_or_empty(
    path: &Path,
    expected_rows: usize,
) -> io::Result<Vec<BTreeMap<String, FieldValue>>> {
    match load_payloads(path) {
        Ok(mut payloads) => {
            if payloads.len() > expected_rows {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "payload row count exceeds record row count",
                ));
            }
            if payloads.len() < expected_rows {
                payloads.resize_with(expected_rows, BTreeMap::new);
            }
            Ok(payloads)
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            Ok(vec![BTreeMap::new(); expected_rows])
        }
        Err(err) => Err(err),
    }
}

fn load_vectors_or_empty(
    path: &Path,
    expected_rows: usize,
) -> io::Result<Vec<BTreeMap<String, Vec<f32>>>> {
    match load_vectors(path) {
        Ok(vectors) => {
            if vectors.len() > expected_rows {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "vector row count exceeds record row count",
                ));
            }
            if vectors.len() < expected_rows {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "vector row count is shorter than record row count",
                ));
            }
            Ok(vectors)
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            Ok(vec![BTreeMap::new(); expected_rows])
        }
        Err(err) => Err(err),
    }
}

fn load_payloads_with_fields_or_empty(
    path: &Path,
    expected_rows: usize,
    fields: Option<&[String]>,
) -> io::Result<Vec<BTreeMap<String, FieldValue>>> {
    match load_payloads_with_fields(path, fields) {
        Ok(mut payloads) => {
            if payloads.len() > expected_rows {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "payload row count exceeds record row count",
                ));
            }
            if payloads.len() < expected_rows {
                payloads.resize_with(expected_rows, BTreeMap::new);
            }
            Ok(payloads)
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            Ok(vec![BTreeMap::new(); expected_rows])
        }
        Err(err) => Err(err),
    }
}

fn load_sparse_vectors_or_empty(
    path: &Path,
    expected_rows: usize,
) -> io::Result<Vec<BTreeMap<String, crate::document::SparseVector>>> {
    match load_sparse_vectors(path) {
        Ok(vectors) => {
            if vectors.len() > expected_rows {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "sparse vector row count exceeds record row count",
                ));
            }
            if vectors.len() < expected_rows {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "sparse vector row count is shorter than record row count",
                ));
            }
            Ok(vectors)
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            Ok(vec![BTreeMap::new(); expected_rows])
        }
        Err(err) => Err(err),
    }
}

fn validate_documents(
    documents: &[Document],
    collection_meta: &CollectionMetadata,
) -> io::Result<()> {
    if collection_meta.dimension == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "collection dimension must be > 0",
        ));
    }

    let vector_schemas = collection_meta
        .vectors
        .iter()
        .map(|vector| (vector.name.as_str(), vector))
        .collect::<HashMap<_, _>>();
    let mut ids = HashSet::with_capacity(documents.len());
    for document in documents {
        if !ids.insert(document.id) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("duplicate external id in batch: {}", document.id),
            ));
        }
        let primary_vector = document
            .vectors
            .get(collection_meta.primary_vector.as_str())
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "document is missing primary vector '{}'",
                        collection_meta.primary_vector
                    ),
                )
            })?;
        if primary_vector.len() != collection_meta.dimension {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "document vector dimension mismatch: expected {}, got {}",
                    collection_meta.dimension,
                    primary_vector.len()
                ),
            ));
        }
        for (name, vector) in &document.vectors {
            let schema = vector_schemas.get(name.as_str()).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "document vector '{}' is not defined in collection schema",
                        name
                    ),
                )
            })?;
            if vector.len() != schema.dimension {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "document vector '{}' dimension mismatch: expected {}, got {}",
                        name,
                        schema.dimension,
                        vector.len()
                    ),
                ));
            }
        }
    }

    Ok(())
}

fn append_documents(
    paths: &CollectionPaths,
    dimension: usize,
    existing_rows: usize,
    primary_vector_name: &str,
    documents: &[Document],
    fp16: bool,
) -> io::Result<usize> {
    ensure_vector_rows(&paths.vectors, existing_rows)?;
    let mut ids = Vec::with_capacity(documents.len());
    let mut records = Vec::with_capacity(documents.len().saturating_mul(dimension));
    let mut payloads = Vec::with_capacity(documents.len());
    let mut vectors = Vec::with_capacity(documents.len());
    let mut sparse_vecs = Vec::with_capacity(documents.len());
    for document in documents {
        ids.push(document.id);
        let primary = document.vectors.get(primary_vector_name).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "document {} is missing primary vector '{}'",
                    document.id, primary_vector_name
                ),
            )
        })?;
        records.extend_from_slice(primary);
        payloads.push(document.fields.clone());
        let mut secondary = document.vectors.clone();
        secondary.remove(primary_vector_name);
        vectors.push(secondary);
        sparse_vecs.push(document.sparse_vectors.clone());
    }

    let inserted = if fp16 {
        append_records_f16(&paths.records, dimension, &records)?
    } else {
        append_records(&paths.records, dimension, &records)?
    };
    let _ = append_record_ids(&paths.external_ids, &ids)?;
    let _ = append_payloads(&paths.payloads, &payloads)?;
    let _ = append_vectors(&paths.vectors, &vectors)?;
    let _ = append_sparse_vectors(&paths.sparse_vectors, &sparse_vecs)?;
    Ok(inserted)
}

fn has_live_id(stored_ids: &[i64], tombstone: &TombstoneMask, external_id: i64) -> bool {
    latest_live_row_index(stored_ids, tombstone, external_id).is_some()
}

fn latest_live_row_index(
    stored_ids: &[i64],
    tombstone: &TombstoneMask,
    external_id: i64,
) -> Option<usize> {
    latest_row_index_for_id(stored_ids, external_id)
        .filter(|row_idx| !tombstone.is_deleted(*row_idx))
}

fn latest_row_index_for_id(stored_ids: &[i64], external_id: i64) -> Option<usize> {
    stored_ids
        .iter()
        .enumerate()
        .rev()
        .find_map(|(row_idx, stored_id)| (*stored_id == external_id).then_some(row_idx))
}

fn mark_live_id_deleted(
    stored_ids: &[i64],
    tombstone: &mut TombstoneMask,
    external_id: i64,
    row_limit: usize,
) {
    for (row_idx, stored_id) in stored_ids.iter().enumerate().take(row_limit) {
        if *stored_id == external_id {
            let _ = tombstone.mark_deleted(row_idx);
        }
    }
}

fn validate_vector_index_descriptor(
    dimension: usize,
    descriptor: &VectorIndexDescriptor,
) -> io::Result<()> {
    if dimension == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "vector dimension must be > 0",
        ));
    }

    let params = descriptor.params.as_object().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "vector index params must be a JSON object",
        )
    })?;

    match descriptor.kind {
        VectorIndexKind::Flat => {
            if !params.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "flat index does not accept params",
                ));
            }
        }
        VectorIndexKind::Ivf => {
            for key in params.keys() {
                if key != "nlist" {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("unsupported ivf param: {key}"),
                    ));
                }
            }
            if let Some(nlist) = params.get("nlist") {
                let nlist = nlist.as_u64().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "ivf nlist must be an unsigned integer",
                    )
                })?;
                if nlist == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "ivf nlist must be > 0",
                    ));
                }
            }
        }
        VectorIndexKind::Hnsw => {
            for key in params.keys() {
                if key != "m" && key != "ef_construction" {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("unsupported hnsw param: {key}"),
                    ));
                }
            }
            for key in ["m", "ef_construction"] {
                if let Some(value) = params.get(key) {
                    let value = value.as_u64().ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("hnsw {key} must be an unsigned integer"),
                        )
                    })?;
                    if value == 0 {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("hnsw {key} must be > 0"),
                        ));
                    }
                }
            }
        }
    }

    DefaultIndexFactory::default()
        .create_vector_index(dimension, descriptor, None)
        .map(|_| ())
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, format!("{err:?}")))
}

fn validate_schema_primary_vector_descriptor(schema: &CollectionSchema) -> io::Result<()> {
    let primary_vector = schema.primary_vector().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "collection primary vector '{}' is not defined in schema vectors",
                schema.primary_vector_name()
            ),
        )
    })?;
    let descriptor = match primary_vector.index_param.as_ref() {
        Some(VectorIndexSchema::Ivf { metric, nlist }) => VectorIndexDescriptor {
            field_name: primary_vector.name.clone(),
            kind: VectorIndexKind::Ivf,
            metric: metric.clone(),
            params: serde_json::json!({ "nlist": nlist }),
        },
        Some(VectorIndexSchema::Hnsw {
            metric,
            m,
            ef_construction,
            ..
        }) => VectorIndexDescriptor {
            field_name: primary_vector.name.clone(),
            kind: VectorIndexKind::Hnsw,
            metric: metric.clone(),
            params: serde_json::json!({
                "m": m,
                "ef_construction": ef_construction
            }),
        },
        None => VectorIndexDescriptor {
            field_name: primary_vector.name.clone(),
            kind: VectorIndexKind::Hnsw,
            metric: Some(schema.metric().to_string()),
            params: serde_json::json!({
                "m": schema.hnsw_m(),
                "ef_construction": schema.hnsw_ef_construction()
            }),
        },
    };
    validate_vector_index_descriptor(primary_vector.dimension, &descriptor)
}

fn validate_schema_secondary_vector_descriptors(schema: &CollectionSchema) -> io::Result<()> {
    let primary_vector_name = schema.primary_vector_name();
    for vector in schema
        .vectors
        .iter()
        .filter(|vector| vector.name != primary_vector_name)
    {
        let Some(index_param) = vector.index_param.as_ref() else {
            continue;
        };
        let descriptor = match index_param {
            VectorIndexSchema::Ivf { metric, nlist } => VectorIndexDescriptor {
                field_name: vector.name.clone(),
                kind: VectorIndexKind::Ivf,
                metric: metric.clone(),
                params: serde_json::json!({ "nlist": nlist }),
            },
            VectorIndexSchema::Hnsw {
                metric,
                m,
                ef_construction,
                ..
            } => VectorIndexDescriptor {
                field_name: vector.name.clone(),
                kind: VectorIndexKind::Hnsw,
                metric: metric.clone(),
                params: serde_json::json!({
                    "m": m,
                    "ef_construction": ef_construction
                }),
            },
        };
        validate_vector_index_descriptor(vector.dimension, &descriptor)?;
    }
    Ok(())
}

#[cfg(feature = "hanns-backend")]
fn resolve_primary_vector_descriptor(
    collection_meta: &CollectionMetadata,
    index_catalog: &IndexCatalog,
) -> io::Result<VectorIndexDescriptor> {
    if let Some(descriptor) = index_catalog
        .vector_indexes
        .iter()
        .find(|descriptor| descriptor.field_name == collection_meta.primary_vector)
    {
        return Ok(descriptor.clone());
    }

    let primary_vector = collection_meta
        .vectors
        .iter()
        .find(|vector| vector.name == collection_meta.primary_vector)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "primary vector '{}' is not defined in collection metadata",
                    collection_meta.primary_vector
                ),
            )
        })?;

    let (kind, metric, params) = match primary_vector.index_param.as_ref() {
        Some(VectorIndexSchema::Ivf { metric, nlist }) => (
            VectorIndexKind::Ivf,
            metric.clone(),
            serde_json::json!({ "nlist": nlist }),
        ),
        Some(VectorIndexSchema::Hnsw {
            metric,
            m,
            ef_construction,
            ..
        }) => (
            VectorIndexKind::Hnsw,
            metric.clone(),
            serde_json::json!({
                "m": m,
                "ef_construction": ef_construction
            }),
        ),
        None => (
            VectorIndexKind::Hnsw,
            Some(collection_meta.metric.clone()),
            serde_json::json!({
                "m": collection_meta.hnsw_m,
                "ef_construction": collection_meta.hnsw_ef_construction
            }),
        ),
    };

    Ok(VectorIndexDescriptor {
        field_name: collection_meta.primary_vector.clone(),
        kind,
        metric,
        params,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{CollectionSchema, Document, VectorFieldSchema, VectorIndexSchema};
    use std::path::Path;
    use tempfile::tempdir;

    #[test]
    fn secondary_indexed_field_search_returns_correct_results_without_optimize() {
        let tempdir = tempdir().expect("tempdir");
        let db = seeded_db_with_secondary_hnsw_indexed_collection(tempdir.path());
        let handle = db
            .open_collection_handle("docs")
            .expect("open collection handle");

        let hits = handle
            .search_field_with_ef_internal("title", &[0.0_f32, 0.0], 3, 64)
            .expect("search secondary vector field");

        assert_eq!(
            hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
            vec![1, 3, 2]
        );

        let cache = handle
            .search_cache
            .lock()
            .expect("search cache mutex poisoned");
        let state = cache.get("title").expect("cached secondary search state");
        assert!(
            state.optimized_ann.is_none(),
            "secondary field should not have ANN runtime until optimize() is called or persisted blob is loaded"
        );
    }

    #[test]
    fn primary_plain_search_keeps_cached_state_unoptimized_without_optimize() {
        let tempdir = tempdir().expect("tempdir");
        let db = seeded_db_with_secondary_hnsw_indexed_collection(tempdir.path());
        let handle = db
            .open_collection_handle("docs")
            .expect("open collection handle");

        let hits = db
            .search("docs", &[0.0_f32, 0.0], 1)
            .expect("plain primary search");

        assert_eq!(hits.iter().map(|hit| hit.id).collect::<Vec<_>>(), vec![2]);

        let cache = handle
            .search_cache
            .lock()
            .expect("search cache mutex poisoned");
        let state = cache.get("dense").expect("cached primary search state");
        assert!(
            state.optimized_ann.is_none(),
            "plain primary search should stay unoptimized until optimize() or persisted HNSW load"
        );
    }

    #[test]
    fn secondary_ivf_indexed_field_search_returns_correct_results_without_optimize() {
        let tempdir = tempdir().expect("tempdir");
        let db = seeded_db_with_secondary_ivf_indexed_collection(tempdir.path());
        let handle = db
            .open_collection_handle("docs")
            .expect("open collection handle");

        let hits = handle
            .search_field_with_ef_internal("title", &[0.0_f32, 0.0], 3, 64)
            .expect("search secondary ivf vector field");

        assert_eq!(
            hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
            vec![1, 3, 2]
        );

        let cache = handle
            .search_cache
            .lock()
            .expect("search cache mutex poisoned");
        let state = cache.get("title").expect("cached secondary search state");
        assert!(
            state.optimized_ann.is_none(),
            "secondary IVF field should not have ANN runtime until optimize() is called or persisted blob is loaded"
        );
    }

    #[test]
    fn build_optimized_ann_state_skips_serialization_when_bytes_not_requested() {
        let tempdir = tempdir().expect("tempdir");
        let db = seeded_db_with_secondary_hnsw_indexed_collection(tempdir.path());
        let handle = db
            .open_collection_handle("docs")
            .expect("open collection handle");
        let state = handle
            .load_search_state_for_field("title")
            .expect("load secondary state");

        let optimized = build_optimized_ann_state(&state, None)
            .expect("build optimized ann state without requesting serialized bytes");

        let hits = ann_search(
            optimized.backend.as_ref(),
            optimized.ann_external_ids.as_slice(),
            &optimized.metric,
            &[0.0_f32, 0.0],
            3,
            64,
        )
        .expect("ann search");
        assert_eq!(
            hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
            vec![1, 3, 2]
        );
    }

    fn seeded_db_with_secondary_hnsw_indexed_collection(root: &Path) -> HannsDb {
        seeded_db_with_secondary_indexed_collection(
            root,
            VectorIndexSchema::hnsw(Some("l2"), 16, 128),
        )
    }

    fn seeded_db_with_secondary_ivf_indexed_collection(root: &Path) -> HannsDb {
        seeded_db_with_secondary_indexed_collection(root, VectorIndexSchema::ivf(Some("l2"), 8))
    }

    fn seeded_db_with_secondary_indexed_collection(
        root: &Path,
        secondary_index: VectorIndexSchema,
    ) -> HannsDb {
        let mut db = HannsDb::open(root).expect("open db");
        let mut schema = CollectionSchema::new("dense", 2, "l2", Vec::new());
        schema
            .vectors
            .push(VectorFieldSchema::new("title", 2).with_index_param(secondary_index));
        db.create_collection_with_schema("docs", &schema)
            .expect("create collection");
        db.insert_documents(
            "docs",
            &[
                Document::with_named_vectors(
                    1,
                    [],
                    "dense",
                    vec![5.0_f32, 5.0],
                    [("title".to_string(), vec![0.0_f32, 0.0])],
                ),
                Document::with_named_vectors(
                    2,
                    [],
                    "dense",
                    vec![0.0_f32, 0.0],
                    [("title".to_string(), vec![2.0_f32, 0.0])],
                ),
                Document::with_named_vectors(
                    3,
                    [],
                    "dense",
                    vec![1.0_f32, 1.0],
                    [("title".to_string(), vec![1.0_f32, 0.0])],
                ),
            ],
        )
        .expect("insert documents");
        db
    }

    #[test]
    fn auto_compact_triggers_after_threshold() {
        use crate::segment::segment_set::COMPACTION_THRESHOLD;
        let tempdir = tempdir().expect("tempdir");
        let mut db = HannsDb::open(tempdir.path()).expect("open db");
        db.create_collection("docs", 2, "l2").expect("create");

        let dim = 2usize;

        // Manually create COMPACTION_THRESHOLD immutable segments + an active one.
        let paths = collection_paths_for_root(tempdir.path(), "docs");
        let seg_manager = SegmentManager::new(paths.dir.clone());

        // Create immutable segments
        let mut immutable_ids = Vec::new();
        for i in 0..COMPACTION_THRESHOLD {
            let seg_id = format!("seg-{:06}", i + 1);
            seg_manager
                .create_segment_dir(&seg_id, dim)
                .expect("create seg");
            let seg_paths =
                SegmentPaths::from_segment_dir(paths.segments_dir.join(&seg_id), seg_id.clone());
            let ids = vec![i as i64 * 10 + 1];
            let vectors = vec![1.0_f32, 0.0];
            append_records(&seg_paths.records, dim, &vectors).expect("append");
            append_record_ids(&seg_paths.external_ids, &ids).expect("append ids");
            let mut meta = SegmentMetadata::load_from_path(&seg_paths.metadata).expect("meta");
            meta.record_count = 1;
            meta.save_to_path(&seg_paths.metadata).expect("save meta");
            TombstoneMask::new(1)
                .save_to_path(&seg_paths.tombstones)
                .expect("save tombstones");
            immutable_ids.push(seg_id);
        }

        // Create active segment
        let active_id = format!("seg-{:06}", COMPACTION_THRESHOLD + 1);
        seg_manager
            .create_segment_dir(&active_id, dim)
            .expect("create active seg");
        let active_paths =
            SegmentPaths::from_segment_dir(paths.segments_dir.join(&active_id), active_id.clone());
        let active_ids = vec![999];
        let active_vectors = vec![1.0_f32, 0.0];
        append_records(&active_paths.records, dim, &active_vectors).expect("append");
        append_record_ids(&active_paths.external_ids, &active_ids).expect("append ids");
        let mut meta = SegmentMetadata::load_from_path(&active_paths.metadata).expect("meta");
        meta.record_count = 1;
        meta.save_to_path(&active_paths.metadata)
            .expect("save meta");
        TombstoneMask::new(1)
            .save_to_path(&active_paths.tombstones)
            .expect("save tombstones");

        let vs = VersionSet::new(active_id, immutable_ids);
        vs.save_to_path(&seg_manager.version_set_path())
            .expect("save vs");

        // Call compact_collection directly — same code path as auto-trigger.
        db.compact_collection("docs").expect("compact");

        let segments = db.list_collection_segments("docs").expect("segments");
        assert!(
            segments.len() <= 2,
            "expected compaction to reduce segments to <= 2 (active + compacted), got {}",
            segments.len()
        );

        // Verify data is intact.
        let info = db.get_collection_info("docs").expect("info");
        assert_eq!(
            info.live_count,
            COMPACTION_THRESHOLD + 1,
            "all live documents should survive compaction"
        );

        let hits = db.search("docs", &[1.0, 0.0], 5).expect("search");
        assert_eq!(hits.len(), COMPACTION_THRESHOLD + 1);
    }

    #[test]
    fn compact_output_is_arrow_format() {
        let tempdir = tempdir().expect("tempdir");
        let mut db = HannsDb::open(tempdir.path()).expect("open db");
        db.create_collection("docs", 2, "l2").expect("create");

        let dim = 2usize;
        let paths = collection_paths_for_root(tempdir.path(), "docs");
        let seg_manager = SegmentManager::new(paths.dir.clone());

        // Create 3 immutable segments with data.
        let mut immutable_ids = Vec::new();
        for i in 0..3 {
            let seg_id = format!("seg-{:06}", i + 1);
            seg_manager
                .create_segment_dir(&seg_id, dim)
                .expect("create seg");
            let seg_paths =
                SegmentPaths::from_segment_dir(paths.segments_dir.join(&seg_id), seg_id.clone());
            let ids = vec![i as i64 * 10 + 1];
            let vectors = vec![1.0_f32, 0.0];
            append_records(&seg_paths.records, dim, &vectors).expect("append");
            append_record_ids(&seg_paths.external_ids, &ids).expect("append ids");
            let mut meta = SegmentMetadata::load_from_path(&seg_paths.metadata).expect("meta");
            meta.record_count = 1;
            meta.save_to_path(&seg_paths.metadata).expect("save meta");
            TombstoneMask::new(1)
                .save_to_path(&seg_paths.tombstones)
                .expect("save tombstones");
            immutable_ids.push(seg_id);
        }

        // Create active segment.
        let active_id = "seg-0004".to_string();
        seg_manager
            .create_segment_dir(&active_id, dim)
            .expect("create active seg");
        let vs = VersionSet::new(active_id, immutable_ids);
        vs.save_to_path(&seg_manager.version_set_path())
            .expect("save vs");

        db.compact_collection("docs").expect("compact");

        // Verify compacted segment has arrow files.
        let version_set = VersionSet::load_from_path(&paths.segment_set).expect("vs");
        let immutables = version_set.immutable_segment_ids();
        assert_eq!(immutables.len(), 1, "should have 1 compacted segment");

        let compacted_dir = paths.segments_dir.join(&immutables[0]);
        assert!(
            compacted_dir.join("payloads.arrow").exists(),
            "compacted segment should have payloads.arrow"
        );
        assert!(
            compacted_dir.join("vectors.arrow").exists(),
            "compacted segment should have vectors.arrow"
        );
    }

    #[test]
    fn atomic_save_no_tmp_residual() {
        let tempdir = tempdir().expect("tempdir");
        let path = tempdir.path().join("test.json");
        let vs = VersionSet::single("seg-0001");
        vs.save_to_path(&path).expect("save");
        assert!(path.exists(), "file should exist");
        assert!(
            !path.with_extension("tmp").exists(),
            "tmp file should not remain after atomic save"
        );
        let loaded = VersionSet::load_from_path(&path).expect("load");
        assert_eq!(loaded, vs);
    }

    #[test]
    fn order_by_scalar_field_ascending() {
        use crate::query::{OrderBy, QueryContext, QueryVector, VectorQuery};
        let tempdir = tempdir().expect("tempdir");
        let mut db = HannsDb::open(tempdir.path()).expect("open db");
        db.create_collection("docs", 2, "l2").expect("create");

        db.insert_documents(
            "docs",
            &[
                Document::new(1, [("score".into(), FieldValue::Int64(10))], vec![1.0, 0.0]),
                Document::new(2, [("score".into(), FieldValue::Int64(30))], vec![0.5, 0.0]),
                Document::new(3, [("score".into(), FieldValue::Int64(20))], vec![0.8, 0.0]),
            ],
        )
        .expect("insert");

        let ctx = QueryContext {
            top_k: 10,
            queries: vec![VectorQuery {
                field_name: "vector".to_string(),
                vector: QueryVector::Dense(vec![0.0, 0.0]),
                param: None,
            }],
            order_by: Some(OrderBy {
                field_name: "score".to_string(),
                descending: false,
            }),
            ..Default::default()
        };

        let hits = db.query_with_context("docs", &ctx).expect("query");
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].id, 1);
        assert_eq!(hits[1].id, 3);
        assert_eq!(hits[2].id, 2);
    }

    #[test]
    fn order_by_scalar_field_descending() {
        use crate::query::{OrderBy, QueryContext, QueryVector, VectorQuery};
        let tempdir = tempdir().expect("tempdir");
        let mut db = HannsDb::open(tempdir.path()).expect("open db");
        db.create_collection("docs", 2, "l2").expect("create");

        db.insert_documents(
            "docs",
            &[
                Document::new(
                    1,
                    [("name".into(), FieldValue::String("alice".into()))],
                    vec![1.0, 0.0],
                ),
                Document::new(
                    2,
                    [("name".into(), FieldValue::String("charlie".into()))],
                    vec![0.5, 0.0],
                ),
                Document::new(
                    3,
                    [("name".into(), FieldValue::String("bob".into()))],
                    vec![0.8, 0.0],
                ),
            ],
        )
        .expect("insert");

        let ctx = QueryContext {
            top_k: 10,
            queries: vec![VectorQuery {
                field_name: "vector".to_string(),
                vector: QueryVector::Dense(vec![0.0, 0.0]),
                param: None,
            }],
            order_by: Some(OrderBy {
                field_name: "name".to_string(),
                descending: true,
            }),
            ..Default::default()
        };

        let hits = db.query_with_context("docs", &ctx).expect("query");
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].id, 2);
        assert_eq!(hits[1].id, 3);
        assert_eq!(hits[2].id, 1);
    }

    #[test]
    fn order_by_rejects_vector_field() {
        use crate::query::{OrderBy, QueryContext, QueryVector, VectorQuery};
        let tempdir = tempdir().expect("tempdir");
        let mut db = HannsDb::open(tempdir.path()).expect("open db");
        db.create_collection("docs", 2, "l2").expect("create");

        let ctx = QueryContext {
            top_k: 10,
            queries: vec![VectorQuery {
                field_name: "vector".to_string(),
                vector: QueryVector::Dense(vec![0.0, 0.0]),
                param: None,
            }],
            order_by: Some(OrderBy {
                field_name: "vector".to_string(),
                descending: false,
            }),
            ..Default::default()
        };

        let result = db.query_with_context("docs", &ctx);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn order_by_rejects_empty_field_name() {
        use crate::query::{OrderBy, QueryContext, QueryVector, VectorQuery};
        let tempdir = tempdir().expect("tempdir");
        let mut db = HannsDb::open(tempdir.path()).expect("open db");
        db.create_collection("docs", 2, "l2").expect("create");

        let ctx = QueryContext {
            top_k: 10,
            queries: vec![VectorQuery {
                field_name: "vector".to_string(),
                vector: QueryVector::Dense(vec![0.0, 0.0]),
                param: None,
            }],
            order_by: Some(OrderBy {
                field_name: "".to_string(),
                descending: false,
            }),
            ..Default::default()
        };

        let result = db.query_with_context("docs", &ctx);
        assert!(result.is_err());
    }

    #[test]
    fn sparse_vector_crud_and_search() {
        use crate::document::{FieldType, SparseVector, VectorFieldSchema};
        let tempdir = tempdir().expect("tempdir");
        let mut db = HannsDb::open(tempdir.path()).expect("open db");

        // Create collection with a primary dense vector and a sparse vector field.
        let mut schema = CollectionSchema::new("dense", 2, "l2", Vec::new());
        schema.vectors.push(VectorFieldSchema {
            name: "sparse_title".to_string(),
            data_type: FieldType::VectorSparse,
            dimension: 0, // sparse vectors have no fixed dimension
            index_param: None,
            bm25_params: None,
        });
        db.create_collection_with_schema("docs", &schema)
            .expect("create collection");

        let docs = &[
            Document::with_sparse_vectors(
                1,
                [("label".into(), FieldValue::String("a".into()))],
                "dense",
                vec![1.0_f32, 0.0],
                [(
                    "sparse_title".to_string(),
                    SparseVector::new(vec![0, 3, 7], vec![1.0, 2.0, 0.5]),
                )],
            ),
            Document::with_sparse_vectors(
                2,
                [("label".into(), FieldValue::String("b".into()))],
                "dense",
                vec![0.0_f32, 1.0],
                [(
                    "sparse_title".to_string(),
                    SparseVector::new(vec![1, 3, 5], vec![0.5, 3.0, 1.0]),
                )],
            ),
            Document::with_sparse_vectors(
                3,
                [("label".into(), FieldValue::String("c".into()))],
                "dense",
                vec![0.5_f32, 0.5],
                [(
                    "sparse_title".to_string(),
                    SparseVector::new(vec![0, 5], vec![2.0, 1.0]),
                )],
            ),
        ];
        db.insert_documents("docs", docs).expect("insert documents");

        // Fetch and verify sparse vectors round-trip.
        let fetched = db.fetch_documents("docs", &[1, 2, 3]).expect("fetch");
        assert_eq!(fetched.len(), 3);
        let doc1 = fetched.iter().find(|d| d.id == 1).expect("doc 1");
        let sv1 = doc1
            .sparse_vectors
            .get("sparse_title")
            .expect("sparse field");
        assert_eq!(sv1.indices, vec![0, 3, 7]);
        assert_eq!(sv1.values, vec![1.0, 2.0, 0.5]);

        // Search sparse with a query that has high overlap with doc 2.
        // Query: indices [1, 3, 5], values [1.0, 1.0, 1.0]
        // IP(doc1): index 3 → 1.0*2.0 = 2.0
        // IP(doc2): index 1→0.5*1.0, 3→3.0*1.0, 5→1.0*1.0 = 4.5
        // IP(doc3): index 5→1.0*1.0 = 1.0
        // Distance = -IP, so smallest distance wins: doc2 (−4.5) < doc1 (−2.0) < doc3 (−1.0)
        let query = SparseVector::new(vec![1, 3, 5], vec![1.0, 1.0, 1.0]);
        let hits = db
            .search_sparse("docs", "sparse_title", &query, 3)
            .expect("sparse search");
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].id, 2);
        assert!(hits[0].distance < hits[1].distance);
        assert_eq!(hits[1].id, 1);
        assert_eq!(hits[2].id, 3);
    }

    #[test]
    fn sparse_inner_product_correctness() {
        use crate::document::SparseVector;
        use crate::query::sparse_inner_product;

        let a = SparseVector::new(vec![0, 3, 7], vec![1.0, 2.0, 0.5]);
        let b = SparseVector::new(vec![1, 3, 5], vec![0.5, 3.0, 1.0]);
        // Only index 3 matches: 2.0 * 3.0 = 6.0
        assert!((sparse_inner_product(&a, &b) - 6.0).abs() < 1e-6);

        // Self product: 1*1 + 2*2 + 0.5*0.5 = 5.25
        assert!((sparse_inner_product(&a, &a) - 5.25).abs() < 1e-6);

        // No overlap
        let c = SparseVector::new(vec![10, 20], vec![1.0, 1.0]);
        assert!((sparse_inner_product(&a, &c)).abs() < 1e-6);
    }
}
