use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[cfg(feature = "knowhere-backend")]
use hannsdb_index::adapter::{AdapterError, HnswBackend};
#[cfg(feature = "knowhere-backend")]
use hannsdb_index::hnsw::{InMemoryHnswIndex, KnowhereHnswIndex};

use crate::catalog::{CollectionMetadata, ManifestMetadata};
use crate::document::{CollectionSchema, Document, FieldValue};
use crate::query::{distance_by_metric, parse_filter, search_by_metric, SearchHit};
use crate::segment::{
    append_payloads, append_record_ids, append_records, load_payloads, load_record_ids,
    load_records, SegmentMetadata, SegmentSet, TombstoneMask,
};
use crate::wal::{append_wal_record, load_wal_records, WalRecord};

const DEFAULT_EF_SEARCH: usize = 32;
const HNSW_INDEX_FILE: &str = "hnsw_index.bin";

pub struct HannsDb {
    root: PathBuf,
    search_cache: Mutex<HashMap<String, CachedSearchState>>,
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
    hnsw_m: usize,
    hnsw_ef_construction: usize,
    #[cfg(feature = "knowhere-backend")]
    optimized_ann: Option<OptimizedAnnState>,
}

#[cfg(feature = "knowhere-backend")]
struct OptimizedAnnState {
    backend: Arc<dyn HnswBackend + Send + Sync>,
    ann_external_ids: Arc<Vec<i64>>,
    metric: String,
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
            search_cache: Mutex::new(HashMap::new()),
        };
        db.replay_wal_if_needed()?;
        Ok(db)
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
        if schema.dimension == 0 {
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

        let segment = SegmentMetadata::new("seg-0001", schema.dimension, 0, 0);
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
        self.invalidate_search_cache(name);
        Ok(())
    }

    pub fn list_collections(&self) -> io::Result<Vec<String>> {
        let manifest = ManifestMetadata::load_from_path(&manifest_path(&self.root))?;
        Ok(manifest.collections)
    }

    pub fn flush_collection(&self, name: &str) -> io::Result<()> {
        let paths = self.collection_paths(name);
        let _ = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let _ = SegmentMetadata::load_from_path(&paths.segment_meta)?;
        let _ = TombstoneMask::load_from_path(&paths.tombstones)?;
        let _ = load_wal_records(&wal_path(&self.root))?;
        Ok(())
    }

    pub fn optimize_collection(&self, name: &str) -> io::Result<()> {
        let state = self.load_search_state(name)?;
        #[cfg(feature = "knowhere-backend")]
        let mut state = state;
        #[cfg(feature = "knowhere-backend")]
        {
            let mut hnsw_bytes = None;
            state.optimized_ann = Some(build_optimized_ann_state(&state, &mut hnsw_bytes)?);
            if let Some(bytes) = hnsw_bytes {
                let paths = self.collection_paths(name);
                if let Err(e) = fs::write(paths.dir.join(HNSW_INDEX_FILE), &bytes) {
                    log::warn!("Failed to persist HNSW index: {e}");
                } else {
                    log::info!(
                        "Persisted HNSW index ({} bytes) for collection '{name}'",
                        bytes.len()
                    );
                }
            }
        }
        let mut cache = self
            .search_cache
            .lock()
            .expect("search cache mutex poisoned");
        cache.insert(name.to_string(), state);
        Ok(())
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

        let mut segment_set = SegmentSet::load_from_path(&paths.segment_set)?;
        if segment_set.immutable_segment_ids.is_empty() {
            return Ok(());
        }

        fs::create_dir_all(&paths.segments_dir)?;
        let immutable_segment_ids = segment_set.immutable_segment_ids.clone();
        let compacted_segment_id = next_compacted_segment_id(
            immutable_segment_ids
                .iter()
                .chain(std::iter::once(&segment_set.active_segment_id)),
        );
        let compacted_dir = paths.segments_dir.join(&compacted_segment_id);
        fs::create_dir_all(&compacted_dir)?;
        let compacted_paths = SegmentPaths::from_segment_dir(compacted_dir.clone());

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

            let segment_paths = SegmentPaths::from_segment_dir(segment_dir);
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

        segment_set.immutable_segment_ids = vec![compacted_segment_id.clone()];
        segment_set.save_to_path(&paths.segment_set)?;
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
        let paths = self.collection_paths(name);
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let segment_meta = SegmentMetadata::load_from_path(&paths.segment_meta)?;
        let live_count = segment_meta
            .record_count
            .saturating_sub(segment_meta.deleted_count);

        Ok(CollectionInfo {
            name: collection_meta.name,
            dimension: collection_meta.dimension,
            metric: collection_meta.metric,
            record_count: segment_meta.record_count,
            deleted_count: segment_meta.deleted_count,
            live_count,
        })
    }

    pub fn list_collection_segments(&self, name: &str) -> io::Result<Vec<CollectionSegmentInfo>> {
        let paths = self.collection_paths(name);
        let _ = CollectionMetadata::load_from_path(&paths.collection_meta)?;

        if !paths.segment_set.exists() {
            let segment_meta = SegmentMetadata::load_from_path(&paths.segment_meta)?;
            return Ok(vec![CollectionSegmentInfo {
                id: segment_meta.segment_id,
                live_count: segment_meta
                    .record_count
                    .saturating_sub(segment_meta.deleted_count),
                dead_count: segment_meta.deleted_count,
                ann_ready: false,
            }]);
        }

        let segment_set = SegmentSet::load_from_path(&paths.segment_set)?;
        let mut segment_ids = Vec::with_capacity(1 + segment_set.immutable_segment_ids.len());
        segment_ids.push(segment_set.active_segment_id);
        segment_ids.extend(segment_set.immutable_segment_ids);

        segment_ids
            .into_iter()
            .map(|segment_id| {
                let segment_dir = paths.segments_dir.join(&segment_id);
                let metadata = SegmentMetadata::load_from_path(&segment_dir.join("segment.json"))?;
                Ok(CollectionSegmentInfo {
                    id: segment_id,
                    live_count: metadata.record_count.saturating_sub(metadata.deleted_count),
                    dead_count: metadata.deleted_count,
                    ann_ready: segment_dir.join("ann").exists(),
                })
            })
            .collect()
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
        let paths = self.collection_paths(collection);
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let records = load_records_or_empty(&paths.records, collection_meta.dimension)?;
        let stored_ids = load_record_ids_or_empty(&paths.external_ids)?;
        let payloads = load_payloads_or_empty(&paths.payloads, stored_ids.len())?;
        let tombstone = TombstoneMask::load_from_path(&paths.tombstones)?;

        if stored_ids.len().saturating_mul(collection_meta.dimension) != records.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "records and ids are not aligned",
            ));
        }

        let mut documents = Vec::with_capacity(external_ids.len());
        for external_id in external_ids {
            if let Some(row_idx) = latest_live_row_index(&stored_ids, &tombstone, *external_id) {
                let start = row_idx * collection_meta.dimension;
                let end = start + collection_meta.dimension;
                documents.push(Document {
                    id: *external_id,
                    fields: payloads[row_idx].clone(),
                    vector: records[start..end].to_vec(),
                });
            }
        }

        Ok(documents)
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
        #[cfg(feature = "knowhere-backend")]
        let ef_search = _ef_search.max(1);
        let mut cache = self
            .search_cache
            .lock()
            .expect("search cache mutex poisoned");
        if !cache.contains_key(collection) {
            let state = self.load_search_state(collection)?;
            cache.insert(collection.to_string(), state);
        }

        let state = cache
            .get(collection)
            .expect("search cache must contain requested collection");
        #[cfg(feature = "knowhere-backend")]
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

    pub fn query_documents(
        &self,
        collection: &str,
        query: &[f32],
        top_k: usize,
        filter: Option<&str>,
    ) -> io::Result<Vec<DocumentHit>> {
        match filter.map(str::trim) {
            None | Some("") => {
                let hits = self.search(collection, query, top_k)?;
                let documents = self.fetch_documents(
                    collection,
                    &hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
                )?;
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
            Some(filter) => self.query_documents_with_filter(collection, query, top_k, filter),
        }
    }

    fn collection_paths(&self, name: &str) -> CollectionPaths {
        let dir = self.root.join("collections").join(name);
        CollectionPaths {
            dir: dir.clone(),
            collection_meta: dir.join("collection.json"),
            segment_set: dir.join("segment_set.json"),
            segments_dir: dir.join("segments"),
            segment_meta: dir.join("segment.json"),
            records: dir.join("records.bin"),
            external_ids: dir.join("ids.bin"),
            payloads: dir.join("payloads.jsonl"),
            tombstones: dir.join("tombstones.json"),
        }
    }

    fn search_segment_paths(&self, collection: &str) -> io::Result<Vec<SegmentPaths>> {
        let paths = self.collection_paths(collection);
        if !paths.segment_set.exists() {
            return Ok(vec![SegmentPaths::from_collection(&paths)]);
        }

        let segment_set = SegmentSet::load_from_path(&paths.segment_set)?;
        let mut segment_ids = Vec::with_capacity(1 + segment_set.immutable_segment_ids.len());
        segment_ids.push(segment_set.active_segment_id);
        segment_ids.extend(segment_set.immutable_segment_ids);

        Ok(segment_ids
            .into_iter()
            .map(|segment_id| SegmentPaths::from_segment_dir(paths.segments_dir.join(segment_id)))
            .collect())
    }

    fn load_search_state(&self, collection: &str) -> io::Result<CachedSearchState> {
        let paths = self.collection_paths(collection);
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let metric = collection_meta.metric.clone();
        let segment_paths = self.search_segment_paths(collection)?;

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

        #[cfg(feature = "knowhere-backend")]
        let optimized_ann = {
            let hnsw_path = paths.dir.join(HNSW_INDEX_FILE);
            if hnsw_path.exists() {
                match fs::read(&hnsw_path).and_then(|bytes| {
                    KnowhereHnswIndex::from_bytes(collection_meta.dimension, &bytes)
                        .map_err(|e| io::Error::other(format!("{e:?}")))
                }) {
                    Ok(backend) => Some(OptimizedAnnState {
                        backend: Arc::new(backend),
                        ann_external_ids: Arc::new(external_ids.clone()),
                        metric: metric.to_ascii_lowercase(),
                    }),
                    Err(e) => {
                        log::warn!("Failed to load persisted HNSW index: {e}");
                        None
                    }
                }
            } else {
                None
            }
        };

        Ok(CachedSearchState {
            records: Arc::new(records),
            external_ids: Arc::new(external_ids),
            tombstone: Arc::new(TombstoneMask::new(0)),
            dimension: collection_meta.dimension,
            metric,
            hnsw_m: collection_meta.hnsw_m,
            hnsw_ef_construction: collection_meta.hnsw_ef_construction,
            #[cfg(feature = "knowhere-backend")]
            optimized_ann,
        })
    }

    fn invalidate_search_cache(&self, collection: &str) {
        let mut cache = self
            .search_cache
            .lock()
            .expect("search cache mutex poisoned");
        cache.remove(collection);
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

        // Backward compatibility: existing collections keep flat-file read/write.
        // This trigger materializes segment-set metadata and directories once.
        if paths.segment_set.exists() {
            return Ok(());
        }

        fs::create_dir_all(&paths.segments_dir)?;
        let mut set = SegmentSet::new_single("seg-000001");
        set.rollover();
        for segment_id in set
            .immutable_segment_ids
            .iter()
            .chain(std::iter::once(&set.active_segment_id))
        {
            fs::create_dir_all(paths.segments_dir.join(segment_id))?;
        }
        set.save_to_path(&paths.segment_set)
    }

    fn query_documents_with_filter(
        &self,
        collection: &str,
        query: &[f32],
        top_k: usize,
        filter: &str,
    ) -> io::Result<Vec<DocumentHit>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let filter_expr = parse_filter(filter)?;
        let paths = self.collection_paths(collection);
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let segment_paths = self.search_segment_paths(collection)?;

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

fn manifest_path(root: &Path) -> PathBuf {
    root.join("manifest.json")
}

fn wal_path(root: &Path) -> PathBuf {
    root.join("wal.jsonl")
}

struct CollectionPaths {
    dir: PathBuf,
    collection_meta: PathBuf,
    segment_set: PathBuf,
    segments_dir: PathBuf,
    segment_meta: PathBuf,
    records: PathBuf,
    external_ids: PathBuf,
    payloads: PathBuf,
    tombstones: PathBuf,
}

#[derive(Debug, Clone)]
struct SegmentPaths {
    records: PathBuf,
    external_ids: PathBuf,
    payloads: PathBuf,
    tombstones: PathBuf,
}

impl SegmentPaths {
    fn from_collection(paths: &CollectionPaths) -> Self {
        Self {
            records: paths.records.clone(),
            external_ids: paths.external_ids.clone(),
            payloads: paths.payloads.clone(),
            tombstones: paths.tombstones.clone(),
        }
    }

    fn from_segment_dir(segment_dir: PathBuf) -> Self {
        Self {
            records: segment_dir.join("records.bin"),
            external_ids: segment_dir.join("ids.bin"),
            payloads: segment_dir.join("payloads.jsonl"),
            tombstones: segment_dir.join("tombstones.json"),
        }
    }
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

#[cfg(feature = "knowhere-backend")]
fn build_optimized_ann_state(
    state: &CachedSearchState,
    index_bytes_out: &mut Option<Vec<u8>>,
) -> io::Result<OptimizedAnnState> {
    let metric = state.metric.to_ascii_lowercase();
    let (ann_external_ids, flat_vectors) = if state.tombstone.deleted_count() == 0 {
        (Arc::clone(&state.external_ids), state.records.as_ref().clone())
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

    if metric == "l2" || metric == "cosine" || metric == "ip" {
        let mut backend = KnowhereHnswIndex::new(
            state.dimension,
            &metric,
            state.hnsw_m,
            state.hnsw_ef_construction,
        )
        .map_err(adapter_error_to_io)?;
        if !flat_vectors.is_empty() {
            backend
                .insert_flat_identity(&flat_vectors, state.dimension)
                .map_err(adapter_error_to_io)?;
        }
        *index_bytes_out = backend.serialize_to_bytes().ok();
        return Ok(OptimizedAnnState {
            backend: Arc::new(backend),
            ann_external_ids,
            metric,
        });
    }

    let mut backend = InMemoryHnswIndex::new(state.dimension, &metric).map_err(adapter_error_to_io)?;
    if !flat_vectors.is_empty() {
        backend
            .insert_flat_identity(&flat_vectors, state.dimension)
            .map_err(adapter_error_to_io)?;
    }
    *index_bytes_out = None;

    Ok(OptimizedAnnState {
        backend: Arc::new(backend),
        ann_external_ids,
        metric,
    })
}

#[cfg(feature = "knowhere-backend")]
fn ann_search(
    backend: &(dyn HnswBackend + Send + Sync),
    ann_external_ids: &[i64],
    metric: &str,
    query: &[f32],
    top_k: usize,
    ef_search: usize,
) -> io::Result<Vec<SearchHit>> {
    let hits = backend
        .search(query, top_k, ef_search)
        .map_err(adapter_error_to_io)?;
    let mut mapped = Vec::with_capacity(hits.len());
    for hit in hits {
        let ann_idx = usize::try_from(hit.id).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "optimized ANN hit id cannot be converted to usize",
            )
        })?;
        let external_id = ann_external_ids.get(ann_idx).copied().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("optimized ANN hit id out of range: {}", hit.id),
            )
        })?;
        let distance = match metric {
            "ip" => -hit.distance,
            _ => hit.distance,
        };
        mapped.push(SearchHit {
            id: external_id,
            distance,
        });
    }
    Ok(mapped)
}

#[cfg(feature = "knowhere-backend")]
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
