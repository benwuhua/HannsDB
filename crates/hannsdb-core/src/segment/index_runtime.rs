//! Per-field ANN index lifecycle management.
//!
//! Centralises the build / search / persist / invalidate logic for optimized
//! ANN runtimes. Loading of raw vector data from segments remains in `db.rs`
//! (which owns the `SegmentManager` and field-loading helpers).

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use hannsdb_index::adapter::{AdapterError, VectorIndexBackend};
use hannsdb_index::descriptor::VectorIndexDescriptor;
use hannsdb_index::factory::DefaultIndexFactory;

use crate::query::SearchHit;
use crate::segment::TombstoneMask;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Cached search state for a single vector field, including optional
/// pre-built ANN index.
#[derive(Clone)]
pub(crate) struct CachedSearchState {
    pub records: Arc<Vec<f32>>,
    pub external_ids: Arc<Vec<i64>>,
    pub tombstone: Arc<TombstoneMask>,
    pub dimension: usize,
    pub metric: String,
    pub field_name: String,
    pub descriptor: VectorIndexDescriptor,
    pub optimized_ann: Option<OptimizedAnnState>,
}

/// An optimized ANN runtime for a single vector field.
#[derive(Clone)]
pub(crate) struct OptimizedAnnState {
    pub backend: Arc<dyn VectorIndexBackend>,
    pub ann_external_ids: Arc<Vec<i64>>,
    pub metric: String,
}

// ---------------------------------------------------------------------------
// ANN index build
// ---------------------------------------------------------------------------

/// Build an optimized ANN runtime from raw search state.
///
/// If `index_bytes_out` is `Some`, the serialized index bytes are written
/// into the provided slot (used for persisting to disk).
pub(crate) fn build_optimized_ann_state(
    state: &CachedSearchState,
    index_bytes_out: Option<&mut Option<Vec<u8>>>,
) -> io::Result<OptimizedAnnState> {
    let metric = state
        .descriptor
        .metric
        .clone()
        .unwrap_or_else(|| state.metric.clone())
        .to_ascii_lowercase();
    let (ann_external_ids, flat_vectors) = if state.tombstone.deleted_count() == 0 {
        (Arc::clone(&state.external_ids), Arc::clone(&state.records))
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
        (Arc::new(live_external_ids), Arc::new(flat_vectors))
    };

    let mut backend = DefaultIndexFactory::default()
        .create_vector_index(state.dimension, &state.descriptor, None)
        .map_err(adapter_error_to_io)?;
    if !flat_vectors.is_empty() {
        backend
            .insert_flat_identity(&flat_vectors, state.dimension)
            .map_err(adapter_error_to_io)?;
    }
    if let Some(index_bytes_out) = index_bytes_out {
        *index_bytes_out = backend.serialize_to_bytes().map_err(adapter_error_to_io)?;
    }

    Ok(OptimizedAnnState {
        backend: Arc::from(backend),
        ann_external_ids,
        metric,
    })
}

// ---------------------------------------------------------------------------
// ANN search
// ---------------------------------------------------------------------------

/// Execute an ANN search using an optimized backend.
pub(crate) fn ann_search(
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
            "l2" => dists_buf[i].max(0.0),
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

/// Execute an ANN search with a pre-filter bitset.
#[cfg(feature = "hanns-backend")]
pub(crate) fn ann_search_with_bitset(
    backend: &dyn VectorIndexBackend,
    ann_external_ids: &[i64],
    metric: &str,
    query: &[f32],
    top_k: usize,
    ef_search: usize,
    bitset: &hannsdb_index::BitsetView,
) -> io::Result<Vec<SearchHit>> {
    if top_k == 0 {
        return Ok(Vec::new());
    }
    let hits = backend
        .search_with_bitset(query, top_k, ef_search, bitset)
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
            "l2" => hit.distance.max(0.0),
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

// ---------------------------------------------------------------------------
// ANN blob persistence
// ---------------------------------------------------------------------------

/// Delete all persisted ANN blobs for a collection.
pub(crate) fn invalidate_ann_blobs(collection_dir: &Path) -> io::Result<()> {
    let ann_dir = collection_dir.join("ann");
    if ann_dir.exists() {
        fs::remove_dir_all(&ann_dir)?;
    }
    let old_path = collection_dir.join(HNSW_INDEX_FILE);
    if old_path.exists() {
        fs::remove_file(&old_path)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

pub(crate) const HNSW_INDEX_FILE: &str = "hnsw_index.bin";

pub(crate) fn ann_blob_path(collection_dir: &Path, field_name: &str) -> PathBuf {
    collection_dir.join("ann").join(format!("{field_name}.bin"))
}

pub(crate) fn ann_ids_path(collection_dir: &Path, field_name: &str) -> PathBuf {
    collection_dir
        .join("ann")
        .join(format!("{field_name}.ids.bin"))
}

// ---------------------------------------------------------------------------
// Error conversion
// ---------------------------------------------------------------------------

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
