use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[cfg(feature = "knowhere-backend")]
use hannsdb_index::adapter::{AdapterError, HnswBackend};
#[cfg(feature = "knowhere-backend")]
use hannsdb_index::hnsw::{InMemoryHnswIndex, KnowhereHnswIndex};

use crate::catalog::{CollectionMetadata, ManifestMetadata};
use crate::document::{CollectionSchema, Document, FieldValue};
use crate::query::{distance_by_metric, parse_filter, search_by_metric, SearchHit};
use crate::segment::{
    append_payloads, append_record_ids, append_records, load_payloads, load_record_ids,
    load_records, SegmentMetadata, TombstoneMask,
};

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

#[derive(Debug, Clone, PartialEq)]
pub struct DocumentHit {
    pub id: i64,
    pub distance: f32,
    pub fields: BTreeMap<String, FieldValue>,
}

struct CachedSearchState {
    records: Vec<f32>,
    external_ids: Vec<i64>,
    tombstone: TombstoneMask,
    dimension: usize,
    metric: String,
    #[cfg(feature = "knowhere-backend")]
    optimized_ann: Option<OptimizedAnnState>,
}

#[cfg(feature = "knowhere-backend")]
struct OptimizedAnnState {
    backend: Box<dyn HnswBackend + Send + Sync>,
    ann_external_ids: Vec<i64>,
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

        Ok(Self {
            root: root.to_path_buf(),
            search_cache: Mutex::new(HashMap::new()),
        })
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
        let paths = self.collection_paths(name);
        if !paths.dir.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("collection not found: {name}"),
            ));
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
        Ok(())
    }

    pub fn optimize_collection(&self, name: &str) -> io::Result<()> {
        let state = self.load_search_state(name)?;
        #[cfg(feature = "knowhere-backend")]
        let mut state = state;
        #[cfg(feature = "knowhere-backend")]
        {
            state.optimized_ann = Some(build_optimized_ann_state(&state)?);
        }
        let mut cache = self
            .search_cache
            .lock()
            .expect("search cache mutex poisoned");
        cache.insert(name.to_string(), state);
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

    pub fn insert(
        &mut self,
        collection: &str,
        external_ids: &[i64],
        vectors: &[f32],
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
        self.invalidate_search_cache(collection);
        Ok(inserted)
    }

    pub fn insert_documents(
        &mut self,
        collection: &str,
        documents: &[Document],
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
        let paths = self.collection_paths(collection);
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let mut segment_meta = SegmentMetadata::load_from_path(&paths.segment_meta)?;
        let mut tombstone = TombstoneMask::load_from_path(&paths.tombstones)?;
        let existing_ids = load_record_ids_or_empty(&paths.external_ids)?;

        validate_documents(documents, collection_meta.dimension)?;

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
        let paths = self.collection_paths(collection);
        let mut segment_meta = SegmentMetadata::load_from_path(&paths.segment_meta)?;
        let mut tombstone = TombstoneMask::load_from_path(&paths.tombstones)?;
        let stored_ids = load_record_ids_or_empty(&paths.external_ids)?;

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
            return ann_search(optimized_ann, query, top_k);
        }

        search_by_metric(
            &state.records,
            &state.external_ids,
            state.dimension,
            &state.tombstone,
            query,
            top_k,
            &state.metric,
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
            segment_meta: dir.join("segment.json"),
            records: dir.join("records.bin"),
            external_ids: dir.join("ids.bin"),
            payloads: dir.join("payloads.jsonl"),
            tombstones: dir.join("tombstones.json"),
        }
    }

    fn load_search_state(&self, collection: &str) -> io::Result<CachedSearchState> {
        let paths = self.collection_paths(collection);
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let records = load_records_or_empty(&paths.records, collection_meta.dimension)?;
        let external_ids = load_record_ids_or_empty(&paths.external_ids)?;
        let tombstone = TombstoneMask::load_from_path(&paths.tombstones)?;

        Ok(CachedSearchState {
            records,
            external_ids,
            tombstone,
            dimension: collection_meta.dimension,
            metric: collection_meta.metric,
            #[cfg(feature = "knowhere-backend")]
            optimized_ann: None,
        })
    }

    fn invalidate_search_cache(&self, collection: &str) {
        let mut cache = self
            .search_cache
            .lock()
            .expect("search cache mutex poisoned");
        cache.remove(collection);
    }

    fn query_documents_with_filter(
        &self,
        collection: &str,
        query: &[f32],
        top_k: usize,
        filter: &str,
    ) -> io::Result<Vec<DocumentHit>> {
        let filter_expr = parse_filter(filter)?;
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

        let mut hits = Vec::new();
        for (row_idx, vector) in records.chunks_exact(collection_meta.dimension).enumerate() {
            if tombstone.is_deleted(row_idx) {
                continue;
            }
            let fields = &payloads[row_idx];
            if !filter_expr.matches(fields) {
                continue;
            }
            hits.push(DocumentHit {
                id: stored_ids[row_idx],
                distance: distance_by_metric(query, vector, &collection_meta.metric)?,
                fields: fields.clone(),
            });
        }

        hits.sort_by(|a, b| {
            a.distance
                .total_cmp(&b.distance)
                .then_with(|| a.id.cmp(&b.id))
        });
        if hits.len() > top_k {
            hits.truncate(top_k);
        }
        Ok(hits)
    }
}

fn manifest_path(root: &Path) -> PathBuf {
    root.join("manifest.json")
}

struct CollectionPaths {
    dir: PathBuf,
    collection_meta: PathBuf,
    segment_meta: PathBuf,
    records: PathBuf,
    external_ids: PathBuf,
    payloads: PathBuf,
    tombstones: PathBuf,
}

fn load_records_or_empty(path: &Path, dimension: usize) -> io::Result<Vec<f32>> {
    match load_records(path, dimension) {
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
fn build_optimized_ann_state(state: &CachedSearchState) -> io::Result<OptimizedAnnState> {
    let metric = state.metric.to_ascii_lowercase();
    let mut backend = make_ann_backend(state.dimension, &metric)?;
    let mut ann_external_ids;
    if state.tombstone.deleted_count() == 0 {
        ann_external_ids = state.external_ids.clone();
        if !state.records.is_empty() {
            backend
                .insert_flat_identity(&state.records, state.dimension)
                .map_err(adapter_error_to_io)?;
        }
    } else {
        let live_count = state
            .external_ids
            .len()
            .saturating_sub(state.tombstone.deleted_count());
        ann_external_ids = Vec::with_capacity(live_count);
        let mut ann_ids = Vec::with_capacity(live_count);
        let mut flat_vectors = Vec::with_capacity(live_count.saturating_mul(state.dimension));
        for (row_idx, vector) in state.records.chunks_exact(state.dimension).enumerate() {
            if state.tombstone.is_deleted(row_idx) {
                continue;
            }
            let ann_id = ann_external_ids.len() as u64;
            ann_ids.push(ann_id);
            flat_vectors.extend_from_slice(vector);
            ann_external_ids.push(state.external_ids[row_idx]);
        }

        if !ann_ids.is_empty() {
            backend
                .insert_flat(&ann_ids, &flat_vectors, state.dimension)
                .map_err(adapter_error_to_io)?;
        }
    }

    Ok(OptimizedAnnState {
        backend,
        ann_external_ids,
        metric,
    })
}

#[cfg(feature = "knowhere-backend")]
fn make_ann_backend(
    dimension: usize,
    metric: &str,
) -> io::Result<Box<dyn HnswBackend + Send + Sync>> {
    if metric == "l2" || metric == "cosine" || metric == "ip" {
        return KnowhereHnswIndex::new(dimension, metric)
            .map(|index| Box::new(index) as Box<dyn HnswBackend + Send + Sync>)
            .map_err(adapter_error_to_io);
    }

    InMemoryHnswIndex::new(dimension, metric)
        .map(|index| Box::new(index) as Box<dyn HnswBackend + Send + Sync>)
        .map_err(adapter_error_to_io)
}

#[cfg(feature = "knowhere-backend")]
fn ann_search(
    optimized_ann: &OptimizedAnnState,
    query: &[f32],
    top_k: usize,
) -> io::Result<Vec<SearchHit>> {
    let hits = optimized_ann
        .backend
        .search(query, top_k)
        .map_err(adapter_error_to_io)?;
    let mut mapped = Vec::with_capacity(hits.len());
    for hit in hits {
        let ann_idx = usize::try_from(hit.id).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "optimized ANN hit id cannot be converted to usize",
            )
        })?;
        let external_id = optimized_ann
            .ann_external_ids
            .get(ann_idx)
            .copied()
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("optimized ANN hit id out of range: {}", hit.id),
                )
            })?;
        let distance = match optimized_ann.metric.as_str() {
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
