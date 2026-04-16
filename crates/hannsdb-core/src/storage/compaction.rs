//! Compaction: merge immutable segments into a single compacted segment.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;

use crate::catalog::CollectionMetadata;
use crate::document::FieldType;
use crate::segment::{
    append_record_ids, append_records, append_records_f16, append_sparse_vectors, load_record_ids,
    load_records, load_records_f16, write_payloads_arrow, write_vectors_arrow, SegmentMetadata,
    SegmentPaths, TombstoneMask, VersionSet,
};
use crate::storage::paths::{CollectionPaths, CollectionPaths as Paths};
use crate::storage::segment_io::{
    load_payloads_or_empty, load_sparse_vectors_or_empty, load_vectors_or_empty,
    materialize_forward_store_snapshot, next_compacted_segment_id,
};

/// Result of a successful compaction.
pub struct CompactionResult {
    /// The new compacted segment ID.
    pub compacted_segment_id: String,
    /// Number of live rows in the compacted segment.
    pub live_row_count: usize,
    /// The immutable segment IDs that were merged (and should be removed).
    pub merged_segment_ids: Vec<String>,
}

/// Compact all immutable segments of a collection into a single segment.
///
/// This is a pure storage-layer operation: it reads immutable segments, filters
/// tombstoned rows, writes a new compacted segment, and updates the version set.
/// The caller is responsible for invalidating search caches and logging WAL records.
pub fn compact_immutable_segments(
    paths: &CollectionPaths,
    collection_meta: &CollectionMetadata,
) -> io::Result<CompactionResult> {
    if !paths.segment_set.exists() {
        return Ok(CompactionResult {
            compacted_segment_id: String::new(),
            live_row_count: 0,
            merged_segment_ids: vec![],
        });
    }

    let mut version_set = VersionSet::load_from_path(&paths.segment_set)?;
    if version_set.immutable_segment_ids().is_empty() {
        return Ok(CompactionResult {
            compacted_segment_id: String::new(),
            live_row_count: 0,
            merged_segment_ids: vec![],
        });
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
    let mut compacted_sparse = Vec::new();

    let has_sparse = collection_meta
        .vectors
        .iter()
        .any(|v| matches!(v.data_type, FieldType::VectorSparse));

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
            load_payloads_or_empty(&segment_paths, &segment_meta, segment_external_ids.len())?;
        let segment_vectors = load_vectors_or_empty(
            &segment_paths,
            &segment_meta,
            &collection_meta.primary_vector,
            segment_external_ids.len(),
        )?;
        let segment_tombstone = TombstoneMask::load_from_path(&segment_paths.tombstones)?;

        let segment_sparse = if has_sparse {
            load_sparse_vectors_or_empty(&segment_paths.sparse_vectors, segment_external_ids.len())?
        } else {
            vec![BTreeMap::new(); segment_external_ids.len()]
        };

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
            if has_sparse {
                compacted_sparse.push(segment_sparse[row_idx].clone());
            }
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
    if has_sparse && !compacted_sparse.is_empty() {
        append_sparse_vectors(&compacted_paths.sparse_vectors, &compacted_sparse)?;
    }
    TombstoneMask::new(compacted_ids.len()).save_to_path(&compacted_paths.tombstones)?;
    let mut compacted_meta = SegmentMetadata::new(
        compacted_segment_id.clone(),
        collection_meta.dimension,
        compacted_ids.len(),
        0,
    );
    compacted_meta.storage_format = "forward_store".to_string();
    compacted_meta.save_to_path(&compacted_dir.join("segment.json"))?;
    materialize_forward_store_snapshot(&compacted_paths, collection_meta)?;

    let live_row_count = compacted_ids.len();

    version_set = VersionSet::new(
        version_set.active_segment_id().to_string(),
        vec![compacted_segment_id.clone()],
    );
    version_set.save_to_path(&paths.segment_set)?;
    for segment_id in &immutable_segment_ids {
        fs::remove_dir_all(paths.segments_dir.join(segment_id))?;
    }

    Ok(CompactionResult {
        compacted_segment_id,
        live_row_count,
        merged_segment_ids: immutable_segment_ids,
    })
}
