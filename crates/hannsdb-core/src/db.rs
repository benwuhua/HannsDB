use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use hannsdb_index::adapter::{AdapterError, VectorIndexBackend};
use hannsdb_index::descriptor::{
    ScalarIndexDescriptor, VectorIndexDescriptor, VectorIndexKind,
};
use hannsdb_index::factory::DefaultIndexFactory;

use crate::catalog::{CollectionMetadata, IndexCatalog, ManifestMetadata};
use crate::document::{CollectionSchema, Document, FieldValue, VectorIndexSchema};
use crate::query::{distance_by_metric, parse_filter, search_by_metric, SearchHit};
use crate::segment::{
    append_payloads, append_record_ids, append_records, load_payloads, load_record_ids,
    load_records, SegmentManager, SegmentMetadata, SegmentPaths, SegmentSet, TombstoneMask,
    VersionSet,
};
use crate::wal::{append_wal_record, load_wal_records, WalRecord};

const DEFAULT_EF_SEARCH: usize = 32;
const HNSW_INDEX_FILE: &str = "hnsw_index.bin";
const INDEX_CATALOG_FILE: &str = "indexes.json";

pub struct HannsDb {
    root: PathBuf,
    collection_handles: RwLock<HashMap<String, Arc<CollectionHandle>>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionInfo {
    pub name: String,
    pub dimension: usize,
    pub metric: String,
    pub record_count: usize,
    pub deleted_count: usize,
    pub live_count: usize,
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
}

struct CachedSearchState {
    records: Arc<Vec<f32>>,
    external_ids: Arc<Vec<i64>>,
    tombstone: Arc<TombstoneMask>,
    dimension: usize,
    metric: String,
    primary_index: VectorIndexDescriptor,
    optimized_ann: Option<OptimizedAnnState>,
}

struct OptimizedAnnState {
    backend: Arc<dyn VectorIndexBackend>,
    ann_external_ids: Arc<Vec<i64>>,
    metric: String,
}

#[derive(Debug, Default)]
struct IndexRegistry;

pub struct CollectionHandle {
    name: String,
    root: PathBuf,
    segment_manager: SegmentManager,
    version_set: RwLock<VersionSet>,
    index_registry: Arc<IndexRegistry>,
    search_cache: Mutex<Option<CachedSearchState>>,
}

impl HannsDb {
    pub fn open(root: &Path) -> io::Result<Self> {
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
            collection_handles: RwLock::new(HashMap::new()),
        };
        db.replay_wal_if_needed()?;
        Ok(db)
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

    pub fn drop_collection(&mut self, name: &str) -> io::Result<()> {
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
        invalidate_persisted_hnsw_blob(&paths.dir)?;
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
        invalidate_persisted_hnsw_blob(&paths.dir)?;
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
        self.open_collection_handle(name)?.flush()
    }

    pub fn optimize_collection(&self, name: &str) -> io::Result<()> {
        self.open_collection_handle(name)?.optimize()
    }

    pub fn compact_collection(&mut self, name: &str) -> io::Result<()> {
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
            let segment_records = load_records(&segment_paths.records, collection_meta.dimension)?;
            let segment_external_ids = load_record_ids(&segment_paths.external_ids)?;
            let segment_payloads =
                load_payloads_or_empty(&segment_paths.payloads, segment_external_ids.len())?;
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
            }
        }

        let inserted = append_records(
            &compacted_paths.records,
            collection_meta.dimension,
            &compacted_records,
        )?;
        if inserted != compacted_ids.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "compacted records and ids are not aligned",
            ));
        }
        let _ = append_record_ids(&compacted_paths.external_ids, &compacted_ids)?;
        let _ = append_payloads(&compacted_paths.payloads, &compacted_payloads)?;
        TombstoneMask::new(compacted_ids.len()).save_to_path(&compacted_paths.tombstones)?;
        SegmentMetadata::new(
            compacted_segment_id.clone(),
            collection_meta.dimension,
            compacted_ids.len(),
            0,
        )
        .save_to_path(&compacted_dir.join("segment.json"))?;

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
        let inserted = append_records(&paths.records, collection_meta.dimension, vectors)?;
        let _ = append_record_ids(&paths.external_ids, external_ids)?;
        let empty_payloads = vec![BTreeMap::new(); inserted];
        let _ = append_payloads(&paths.payloads, &empty_payloads)?;
        segment_meta.record_count += inserted;
        segment_meta.deleted_count = tombstone.deleted_count();

        let needed = segment_meta.record_count.saturating_sub(tombstone.len());
        if needed > 0 {
            tombstone.extend(needed);
        }

        segment_meta.save_to_path(&paths.segment_meta)?;
        tombstone.save_to_path(&paths.tombstones)?;
        self.maybe_trigger_segment_rollover(&paths, &segment_meta)?;
        self.invalidate_search_cache(collection);
        Ok(inserted)
    }

    pub fn insert_documents(
        &mut self,
        collection: &str,
        documents: &[Document],
    ) -> io::Result<usize> {
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

        validate_documents(documents, collection_meta.dimension)?;

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
        let inserted = append_documents(&paths, collection_meta.dimension, documents)?;
        segment_meta.record_count += inserted;
        segment_meta.deleted_count = tombstone.deleted_count();

        let needed = segment_meta.record_count.saturating_sub(tombstone.len());
        if needed > 0 {
            tombstone.extend(needed);
        }

        segment_meta.save_to_path(&paths.segment_meta)?;
        tombstone.save_to_path(&paths.tombstones)?;
        self.invalidate_search_cache(collection);
        Ok(inserted)
    }

    pub fn upsert_documents(
        &mut self,
        collection: &str,
        documents: &[Document],
    ) -> io::Result<usize> {
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

        validate_documents(documents, collection_meta.dimension)?;

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
        let inserted = append_documents(&paths, collection_meta.dimension, documents)?;
        segment_meta.record_count += inserted;
        segment_meta.deleted_count = tombstone.deleted_count();

        let needed = segment_meta.record_count.saturating_sub(tombstone.len());
        if needed > 0 {
            tombstone.extend(needed);
        }

        segment_meta.save_to_path(&paths.segment_meta)?;
        tombstone.save_to_path(&paths.tombstones)?;
        self.invalidate_search_cache(collection);
        Ok(inserted)
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
        self.delete_internal(collection, external_ids, true)
    }

    fn delete_internal(
        &mut self,
        collection: &str,
        external_ids: &[i64],
        log_wal: bool,
    ) -> io::Result<usize> {
        let paths = self.collection_paths(collection);
        let mut segment_meta = SegmentMetadata::load_from_path(&paths.segment_meta)?;
        let mut tombstone = TombstoneMask::load_from_path(&paths.tombstones)?;
        let stored_ids = load_record_ids_or_empty(&paths.external_ids)?;

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
        for (row_idx, stored_id) in stored_ids.iter().enumerate() {
            if external_ids.contains(stored_id)
                && row_idx < segment_meta.record_count
                && tombstone.mark_deleted(row_idx)
            {
                newly_deleted += 1;
            }
        }

        segment_meta.deleted_count = tombstone.deleted_count();
        segment_meta.save_to_path(&paths.segment_meta)?;
        tombstone.save_to_path(&paths.tombstones)?;
        self.invalidate_search_cache(collection);
        Ok(newly_deleted)
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

    fn collection_paths(&self, name: &str) -> CollectionPaths {
        collection_paths_for_root(&self.root, name)
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

        let _ = (paths, segment_meta);
        // Auto-rollover is intentionally disabled until the mutable segment
        // writer can initialize a fully readable multi-segment layout.
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
            search_cache: Mutex::new(None),
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
        let ef_search = _ef_search.max(1);
        let mut cache = self
            .search_cache
            .lock()
            .expect("search cache mutex poisoned");
        if cache.is_none() {
            let state = if let Some(ann_state) = self.try_load_persisted_ann_state()? {
                ann_state
            } else {
                self.load_search_state()?
            };
            *cache = Some(state);
        }

        let state = cache
            .as_ref()
            .expect("search cache must contain requested collection");
        if let Some(optimized_ann) = state.optimized_ann.as_ref() {
            let optimized_snapshot = (
                Arc::clone(&optimized_ann.backend),
                Arc::clone(&optimized_ann.ann_external_ids),
                optimized_ann.metric.clone(),
            );
            drop(cache);
            return ann_search(
                optimized_snapshot.0.as_ref(),
                optimized_snapshot.1.as_slice(),
                &optimized_snapshot.2,
                query,
                top_k,
                ef_search,
            );
        }
        let brute_force_snapshot = (
            Arc::clone(&state.records),
            Arc::clone(&state.external_ids),
            Arc::clone(&state.tombstone),
            state.dimension,
            state.metric.clone(),
        );
        drop(cache);

        let (records, external_ids, tombstone, dimension, metric) = brute_force_snapshot;

        search_by_metric(
            &records,
            &external_ids,
            dimension,
            tombstone.as_ref(),
            query,
            top_k,
            &metric,
        )
    }

    pub fn optimize(&self) -> io::Result<()> {
        let _index_registry = Arc::clone(&self.index_registry);
        let mut state = self.load_search_state()?;
        let mut hnsw_bytes = None;
        state.optimized_ann = Some(build_optimized_ann_state(&state, &mut hnsw_bytes)?);
        #[cfg(feature = "knowhere-backend")]
        {
            let paths = self.collection_paths();
            if let Some(bytes) = hnsw_bytes {
                if let Err(e) = fs::write(paths.dir.join(HNSW_INDEX_FILE), &bytes) {
                    log::warn!("Failed to persist HNSW index: {e}");
                } else {
                    log::info!(
                        "Persisted HNSW index ({} bytes) for collection '{}'",
                        bytes.len(),
                        self.name
                    );
                }
            } else {
                let hnsw_path = paths.dir.join(HNSW_INDEX_FILE);
                if hnsw_path.exists() {
                    let _ = fs::remove_file(hnsw_path);
                }
            }
        }
        let mut cache = self
            .search_cache
            .lock()
            .expect("search cache mutex poisoned");
        *cache = Some(state);
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

        Ok(CollectionInfo {
            name: collection_meta.name,
            dimension: collection_meta.dimension,
            metric: collection_meta.metric,
            record_count,
            deleted_count,
            live_count: record_count.saturating_sub(deleted_count),
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
        match filter.map(str::trim) {
            None | Some("") => {
                let hits = self.search(query, top_k)?;
                let documents =
                    self.fetch_documents(&hits.iter().map(|hit| hit.id).collect::<Vec<_>>())?;
                Ok(hits
                    .into_iter()
                    .zip(documents)
                    .map(|(hit, document)| DocumentHit {
                        id: hit.id,
                        distance: hit.distance,
                        fields: document.fields,
                    })
                    .collect())
            }
            Some(filter) => self.query_documents_with_filter(query, top_k, filter),
        }
    }

    fn fetch_documents(&self, external_ids: &[i64]) -> io::Result<Vec<Document>> {
        let paths = self.collection_paths();
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let mut documents = Vec::with_capacity(external_ids.len());

        for external_id in external_ids {
            let mut found = None;
            for segment in self.segment_manager.segment_paths()? {
                let records = load_records_or_empty(&segment.records, collection_meta.dimension)?;
                let stored_ids = load_record_ids_or_empty(&segment.external_ids)?;
                let payloads = load_payloads_or_empty(&segment.payloads, stored_ids.len())?;
                let tombstone = TombstoneMask::load_from_path(&segment.tombstones)?;

                if stored_ids.len().saturating_mul(collection_meta.dimension) != records.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "records and ids are not aligned",
                    ));
                }

                if let Some(row_idx) = latest_live_row_index(&stored_ids, &tombstone, *external_id)
                {
                    let start = row_idx * collection_meta.dimension;
                    let end = start + collection_meta.dimension;
                    found = Some(Document {
                        id: *external_id,
                        fields: payloads[row_idx].clone(),
                        vector: records[start..end].to_vec(),
                    });
                    break;
                }
            }

            if let Some(document) = found {
                documents.push(document);
            }
        }

        Ok(documents)
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

        let mut heap: BinaryHeap<RankedDocumentHit> = BinaryHeap::new();
        for segment in segment_paths {
            let records = load_records_or_empty(&segment.records, collection_meta.dimension)?;
            let stored_ids = load_record_ids_or_empty(&segment.external_ids)?;
            let payloads = load_payloads_or_empty(&segment.payloads, stored_ids.len())?;
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
                let fields = &payloads[row_idx];
                if !filter_expr.matches(fields) {
                    continue;
                }
                let candidate = RankedDocumentHit {
                    hit: DocumentHit {
                        id: stored_ids[row_idx],
                        distance: distance_by_metric(query, vector, &collection_meta.metric)?,
                        fields: fields.clone(),
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

    fn collection_paths(&self) -> CollectionPaths {
        collection_paths_for_root(&self.root, &self.name)
    }

    fn load_search_state(&self) -> io::Result<CachedSearchState> {
        let paths = self.collection_paths();
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let index_catalog = IndexCatalog::load_from_path(&paths.index_catalog)?;
        let primary_index = resolve_primary_vector_descriptor(&collection_meta, &index_catalog)?;
        let metric = collection_meta.metric.clone();
        let segment_paths = self.segment_manager.segment_paths()?;

        let mut records = Vec::new();
        let mut external_ids = Vec::new();

        for segment in segment_paths {
            let segment_records =
                load_records_or_empty(&segment.records, collection_meta.dimension)?;
            let segment_external_ids = load_record_ids_or_empty(&segment.external_ids)?;
            let tombstone = TombstoneMask::load_from_path(&segment.tombstones)?;

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
                if tombstone.is_deleted(row_idx) {
                    continue;
                }
                records.extend_from_slice(vector);
                external_ids.push(segment_external_ids[row_idx]);
            }
        }

        Ok(CachedSearchState {
            records: Arc::new(records),
            external_ids: Arc::new(external_ids),
            tombstone: Arc::new(TombstoneMask::new(0)),
            dimension: collection_meta.dimension,
            metric,
            primary_index,
            optimized_ann: None,
        })
    }

    fn try_load_persisted_ann_state(&self) -> io::Result<Option<CachedSearchState>> {
        #[cfg(not(feature = "knowhere-backend"))]
        {
            return Ok(None);
        }

        #[cfg(feature = "knowhere-backend")]
        {
        let paths = self.collection_paths();
        let hnsw_path = paths.dir.join(HNSW_INDEX_FILE);
        if !hnsw_path.exists() {
            return Ok(None);
        }

        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let index_catalog = IndexCatalog::load_from_path(&paths.index_catalog)?;
        let primary_index = resolve_primary_vector_descriptor(&collection_meta, &index_catalog)?;
        if primary_index.kind != VectorIndexKind::Hnsw {
            return Ok(None);
        }
        let metric = collection_meta.metric.clone();
        let metric_lc = primary_index
            .metric
            .clone()
            .unwrap_or_else(|| metric.clone())
            .to_ascii_lowercase();
        let segment_paths = self.segment_manager.segment_paths()?;

        let mut live_external_ids = Vec::new();
        for segment in segment_paths {
            let segment_external_ids = load_record_ids_or_empty(&segment.external_ids)?;
            let tombstone = TombstoneMask::load_from_path(&segment.tombstones)?;
            for (row_idx, external_id) in segment_external_ids.into_iter().enumerate() {
                if !tombstone.is_deleted(row_idx) {
                    live_external_ids.push(external_id);
                }
            }
        }

        match fs::read(&hnsw_path).and_then(|bytes| {
            DefaultIndexFactory::default()
                .create_vector_index(collection_meta.dimension, &primary_index, Some(&bytes))
                .map_err(adapter_error_to_io)
        }) {
            Ok(backend) => {
                log::info!(
                    "Loaded persisted HNSW index for '{}' from '{}'",
                    self.name,
                    hnsw_path.display()
                );
                let ann_external_ids = Arc::new(live_external_ids);
                Ok(Some(CachedSearchState {
                    records: Arc::new(Vec::new()),
                    external_ids: Arc::clone(&ann_external_ids),
                    tombstone: Arc::new(TombstoneMask::new(0)),
                    dimension: collection_meta.dimension,
                    metric,
                    primary_index,
                    optimized_ann: Some(OptimizedAnnState {
                        backend: Arc::from(backend),
                        ann_external_ids,
                        metric: metric_lc,
                    }),
                }))
            }
            Err(e) => {
                log::warn!(
                    "Fast persisted HNSW load failed for '{}' from '{}': {e}",
                    self.name,
                    hnsw_path.display()
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
        *cache = None;
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
                    }
                }
                WalRecord::InsertDocuments {
                    collection,
                    documents,
                } if !documents.is_empty() => {
                    if let Some(plan) = collections.get_mut(collection) {
                        plan.requires_data_files = true;
                    }
                }
                WalRecord::UpsertDocuments {
                    collection,
                    documents,
                } if !documents.is_empty() => {
                    if let Some(plan) = collections.get_mut(collection) {
                        plan.requires_data_files = true;
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
            if !manifest.collections.iter().any(|entry| entry == collection) {
                return Ok(true);
            }
            if !paths.collection_meta.exists()
                || !paths.segment_meta.exists()
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
        | WalRecord::Delete { collection, .. } => collection,
        WalRecord::CompactCollection {
            collection_name, ..
        } => collection_name,
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

fn load_records_or_empty(path: &Path, dimension: usize) -> io::Result<Vec<f32>> {
    match load_records(path, dimension) {
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

fn validate_documents(documents: &[Document], dimension: usize) -> io::Result<()> {
    if dimension == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "collection dimension must be > 0",
        ));
    }

    let mut ids = HashSet::with_capacity(documents.len());
    for document in documents {
        if !ids.insert(document.id) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("duplicate external id in batch: {}", document.id),
            ));
        }
        if document.vector.len() != dimension {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "document vector dimension mismatch: expected {dimension}, got {}",
                    document.vector.len()
                ),
            ));
        }
    }

    Ok(())
}

fn append_documents(
    paths: &CollectionPaths,
    dimension: usize,
    documents: &[Document],
) -> io::Result<usize> {
    let mut ids = Vec::with_capacity(documents.len());
    let mut records = Vec::with_capacity(documents.len().saturating_mul(dimension));
    let mut payloads = Vec::with_capacity(documents.len());
    for document in documents {
        ids.push(document.id);
        records.extend_from_slice(&document.vector);
        payloads.push(document.fields.clone());
    }

    let inserted = append_records(&paths.records, dimension, &records)?;
    let _ = append_record_ids(&paths.external_ids, &ids)?;
    let _ = append_payloads(&paths.payloads, &payloads)?;
    Ok(inserted)
}

fn ensure_payload_rows(path: &Path, expected_rows: usize) -> io::Result<()> {
    match load_payloads(path) {
        Ok(payloads) => {
            if payloads.len() > expected_rows {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "payload row count exceeds record row count",
                ));
            }
            if payloads.len() < expected_rows {
                let missing = vec![BTreeMap::new(); expected_rows - payloads.len()];
                let _ = append_payloads(path, &missing)?;
            }
            Ok(())
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            if expected_rows > 0 {
                let _ = append_payloads(path, &vec![BTreeMap::new(); expected_rows])?;
            }
            Ok(())
        }
        Err(err) => Err(err),
    }
}

fn has_live_id(stored_ids: &[i64], tombstone: &TombstoneMask, external_id: i64) -> bool {
    latest_live_row_index(stored_ids, tombstone, external_id).is_some()
}

fn latest_live_row_index(
    stored_ids: &[i64],
    tombstone: &TombstoneMask,
    external_id: i64,
) -> Option<usize> {
    stored_ids
        .iter()
        .enumerate()
        .rev()
        .find_map(|(row_idx, stored_id)| {
            if *stored_id == external_id && !tombstone.is_deleted(row_idx) {
                Some(row_idx)
            } else {
                None
            }
        })
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

fn invalidate_persisted_hnsw_blob(collection_dir: &Path) -> io::Result<()> {
    let hnsw_path = collection_dir.join(HNSW_INDEX_FILE);
    match fs::remove_file(&hnsw_path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err),
    }
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

fn build_optimized_ann_state(
    state: &CachedSearchState,
    index_bytes_out: &mut Option<Vec<u8>>,
) -> io::Result<OptimizedAnnState> {
    let metric = state
        .primary_index
        .metric
        .clone()
        .unwrap_or_else(|| state.metric.clone())
        .to_ascii_lowercase();
    let (ann_external_ids, flat_vectors) = if state.tombstone.deleted_count() == 0 {
        (
            Arc::clone(&state.external_ids),
            state.records.as_ref().clone(),
        )
    } else {
        let live_count = state
            .external_ids
            .len()
            .saturating_sub(state.tombstone.deleted_count());
        let mut live_external_ids = Vec::with_capacity(live_count);
        let mut flat_vectors = Vec::with_capacity(live_count.saturating_mul(state.dimension));
        for (row_idx, vector) in state.records.chunks_exact(state.dimension).enumerate() {
            if state.tombstone.is_deleted(row_idx) {
                continue;
            }
            flat_vectors.extend_from_slice(vector);
            live_external_ids.push(state.external_ids[row_idx]);
        }
        (Arc::new(live_external_ids), flat_vectors)
    };

    let mut backend = DefaultIndexFactory::default()
        .create_vector_index(state.dimension, &state.primary_index, None)
        .map_err(adapter_error_to_io)?;
    if !flat_vectors.is_empty() {
        backend
            .insert_flat_identity(&flat_vectors, state.dimension)
            .map_err(adapter_error_to_io)?;
    }
    *index_bytes_out = backend.serialize_to_bytes().map_err(adapter_error_to_io)?;

    Ok(OptimizedAnnState {
        backend: Arc::from(backend),
        ann_external_ids,
        metric,
    })
}

fn ann_search(
    backend: &dyn VectorIndexBackend,
    ann_external_ids: &[i64],
    metric: &str,
    query: &[f32],
    top_k: usize,
    ef_search: usize,
) -> io::Result<Vec<SearchHit>> {
    if top_k == 0 {
        return Ok(Vec::new());
    }

    let mut ids_buf = vec![0_i64; top_k];
    let mut dists_buf = vec![0.0_f32; top_k];
    let n = backend
        .search_into(query, top_k, ef_search, &mut ids_buf, &mut dists_buf)
        .map_err(adapter_error_to_io)?;
    let mut mapped = Vec::with_capacity(n);
    for i in 0..n {
        let ann_idx = usize::try_from(ids_buf[i]).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "optimized ANN hit id cannot be converted to usize",
            )
        })?;
        let external_id = ann_external_ids.get(ann_idx).copied().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("optimized ANN hit id out of range: {}", ids_buf[i]),
            )
        })?;
        let distance = match metric {
            "l2" => dists_buf[i].max(0.0).sqrt(),
            "ip" => -dists_buf[i],
            _ => dists_buf[i],
        };
        mapped.push(SearchHit {
            id: external_id,
            distance,
        });
    }
    Ok(mapped)
}

fn adapter_error_to_io(err: AdapterError) -> io::Error {
    match err {
        AdapterError::InvalidDimension { expected, got } => io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("dimension mismatch: expected {expected}, got {got}"),
        ),
        AdapterError::EmptyInsert => io::Error::new(io::ErrorKind::InvalidInput, "empty insert"),
        AdapterError::Backend(msg) => io::Error::other(msg),
    }
}
