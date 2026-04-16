//! Index persistence: build and persist ANN, scalar, and sparse indexes.
//!
//! Pure storage-layer operations for serializing index structures to disk
//! and deserializing them back.  Cache mutation and DB-level coordination
//! remain in `db.rs`.

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::io;
use std::path::Path;
use std::sync::Arc;

use crate::catalog::{CollectionMetadata, IndexCatalog};
use crate::document::{field_value_to_scalar, FieldType, FieldValue};
use crate::query::resolve_vector_descriptor_for_field;
use crate::segment::index_runtime::{
    ann_blob_path, ann_ids_path, CachedSearchState, OptimizedAnnState,
};
use crate::segment::{SegmentMetadata, SegmentPaths, TombstoneMask};
use crate::storage::paths::CollectionPaths;
use crate::storage::segment_io::{
    load_external_ids_for_segment_or_empty, load_payloads_or_empty, load_sparse_vectors_or_empty,
};

use hannsdb_index::descriptor::{ScalarIndexDescriptor, SparseIndexDescriptor, SparseIndexKind};
use hannsdb_index::factory::DefaultIndexFactory;
use hannsdb_index::scalar::{InvertedScalarIndex, ScalarValue};
use hannsdb_index::sparse::{SparseIndexBackend, SparseVectorData};

// ---------------------------------------------------------------------------
// ANN blob persistence
// ---------------------------------------------------------------------------

/// Persist an ANN index blob and its external IDs to disk.
pub fn persist_ann_blob(
    collection_dir: &Path,
    field_name: &str,
    blob_bytes: &[u8],
    ann_external_ids: &[i64],
) -> io::Result<()> {
    let blob_path = ann_blob_path(collection_dir, field_name);
    if let Some(parent) = blob_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&blob_path, blob_bytes)?;

    let ids_path = ann_ids_path(collection_dir, field_name);
    let ids_bytes: Vec<u8> = ann_external_ids
        .iter()
        .flat_map(|id| id.to_le_bytes())
        .collect();
    fs::write(&ids_path, &ids_bytes)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Scalar index building
// ---------------------------------------------------------------------------

/// Build scalar indexes for all fields listed in the catalog, scanning
/// live rows across all segments.
///
/// Returns a map of `(field_name -> InvertedScalarIndex)`.
pub fn build_scalar_indexes_from_segments(
    segment_paths: &[SegmentPaths],
    scalar_descriptors: &[ScalarIndexDescriptor],
) -> io::Result<Vec<(String, InvertedScalarIndex)>> {
    if scalar_descriptors.is_empty() {
        return Ok(Vec::new());
    }

    let mut all_payloads: Vec<BTreeMap<String, FieldValue>> = Vec::new();
    let mut all_ids: Vec<i64> = Vec::new();

    for segment in segment_paths {
        let segment_meta = SegmentMetadata::load_from_path(&segment.metadata)?;
        let stored_ids = load_external_ids_for_segment_or_empty(segment, &segment_meta)?;
        let payloads = load_payloads_or_empty(&segment, &segment_meta, stored_ids.len())?;
        let tombstone = TombstoneMask::load_from_path(&segment.tombstones)?;
        for (row_idx, ext_id) in stored_ids.iter().enumerate() {
            if tombstone.is_deleted(row_idx) {
                continue;
            }
            all_payloads.push(payloads[row_idx].clone());
            all_ids.push(*ext_id);
        }
    }

    let scalar_payloads: Vec<BTreeMap<String, ScalarValue>> = all_payloads
        .iter()
        .map(|map| {
            map.iter()
                .map(|(k, v)| (k.clone(), field_value_to_scalar(v)))
                .collect()
        })
        .collect();

    let mut indexes = Vec::with_capacity(scalar_descriptors.len());
    for scalar_descriptor in scalar_descriptors {
        let field_name = &scalar_descriptor.field_name;
        let index = InvertedScalarIndex::build_from_payloads(
            scalar_descriptor.clone(),
            field_name,
            &scalar_payloads,
            &all_ids,
        );
        indexes.push((field_name.clone(), index));
    }
    Ok(indexes)
}

// ---------------------------------------------------------------------------
// Sparse index building
// ---------------------------------------------------------------------------

/// Build sparse indexes for all sparse vector fields, scanning live rows
/// across all segments.
///
/// Returns a list of `(field_name, SparseIndexBackend)`.
pub fn build_sparse_indexes_from_segments(
    paths: &CollectionPaths,
    segment_paths: &[SegmentPaths],
    collection_meta: &CollectionMetadata,
) -> io::Result<Vec<(String, Box<dyn SparseIndexBackend>)>> {
    let sparse_fields: Vec<_> = collection_meta
        .vectors
        .iter()
        .filter(|v| v.data_type == FieldType::VectorSparse)
        .collect();

    if sparse_fields.is_empty() {
        return Ok(Vec::new());
    }

    let mut results = Vec::with_capacity(sparse_fields.len());

    for vector_schema in &sparse_fields {
        let field_name = &vector_schema.name;
        let mut all_sparse: Vec<(i64, SparseVectorData)> = Vec::new();
        let mut shadowed: HashSet<i64> = HashSet::new();

        for segment in segment_paths {
            let segment_meta = SegmentMetadata::load_from_path(&segment.metadata)?;
            let stored_ids = load_external_ids_for_segment_or_empty(segment, &segment_meta)?;
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
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("sparse index create: {e:?}")))?;

        index
            .add(&all_sparse)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("sparse index build: {e:?}")))?;

        if let Some(bm25_params) = &vector_schema.bm25_params {
            index.set_bm25_params(bm25_params.k1, bm25_params.b, bm25_params.avgdl);
        }

        // Persist to disk.
        if let Ok(Some(bytes)) = index.serialize_to_bytes() {
            let blob_path = paths.dir.join("ann").join(format!("{field_name}.sparse.bin"));
            if let Some(parent) = blob_path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            let _ = fs::write(&blob_path, &bytes);
        }

        results.push((field_name.clone(), index));
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// ANN index loading from disk
// ---------------------------------------------------------------------------

/// Load a persisted ANN index from disk.
///
/// Returns `Ok(Some(CachedSearchState))` if a valid blob was found,
/// `Ok(None)` if no persisted blob exists.  Falls back to loading external
/// IDs from segment data when the persisted ID file is absent.
#[cfg(feature = "hanns-backend")]
pub fn load_persisted_ann_from_disk(
    paths: &CollectionPaths,
    collection_meta: &CollectionMetadata,
    index_catalog: &IndexCatalog,
    field_name: &str,
    fallback_external_ids: Arc<Vec<i64>>,
) -> io::Result<Option<CachedSearchState>> {
    use hannsdb_index::descriptor::VectorIndexDescriptor;
    use crate::segment::index_runtime::HNSW_INDEX_FILE;

    let descriptor =
        resolve_vector_descriptor_for_field(collection_meta, index_catalog, field_name)?
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no descriptor for field"))?;

    let new_path = ann_blob_path(&paths.dir, field_name);
    let legacy_path = paths.dir.join(HNSW_INDEX_FILE);
    let is_primary = field_name == collection_meta.primary_vector;

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
                fallback_external_ids
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
            Ok(Some(state))
        }
        Err(_) => Ok(None),
    }
}
