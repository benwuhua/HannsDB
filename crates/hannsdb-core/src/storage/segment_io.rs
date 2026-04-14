use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::convert::TryFrom;
use std::io;
use std::path::Path;

use crate::catalog::CollectionMetadata;
use crate::document::{FieldValue, SparseVector};
use crate::forward_store::{ChunkedFileWriter, ForwardFileFormat, ForwardRow, MemForwardStore};
use crate::segment::index_runtime::{ann_blob_path, HNSW_INDEX_FILE};
use crate::segment::{
    load_payloads_jsonl, load_payloads_with_fields, load_record_ids, load_records,
    load_records_f16, load_sparse_vectors, load_vectors, load_vectors_jsonl, write_payloads_arrow,
    write_vectors_arrow, NormalizedStorageFormat, SegmentManager, SegmentMetadata, SegmentPaths,
    TombstoneMask,
};
use crate::storage::paths::CollectionPaths;

fn load_payloads_for_segment(
    segment: &SegmentPaths,
    segment_meta: &SegmentMetadata,
    fields: Option<&[String]>,
) -> io::Result<Vec<BTreeMap<String, FieldValue>>> {
    match segment_meta.normalized_storage_format() {
        NormalizedStorageFormat::Arrow => {
            if segment.payloads_arrow.exists() {
                match load_payloads_with_fields(&segment.payloads, fields) {
                    Ok(payloads) => return Ok(payloads),
                    Err(err) if segment.payloads.exists() => {
                        let _ = err;
                    }
                    Err(err) => return Err(err),
                }
            }
            load_payload_rows_jsonl(&segment.payloads, fields)
        }
        NormalizedStorageFormat::Jsonl => {
            if segment.payloads.exists() {
                return load_payload_rows_jsonl(&segment.payloads, fields);
            }
            load_payloads_with_fields(&segment.payloads, fields)
        }
    }
}

fn load_payload_rows_jsonl(
    path: &Path,
    fields: Option<&[String]>,
) -> io::Result<Vec<BTreeMap<String, FieldValue>>> {
    let payloads = load_payloads_jsonl(path)?;
    if let Some(field_names) = fields {
        let keep: std::collections::BTreeSet<&str> =
            field_names.iter().map(String::as_str).collect();
        return Ok(payloads
            .into_iter()
            .map(|mut map| {
                map.retain(|k, _| keep.contains(k.as_str()));
                map
            })
            .collect());
    }
    Ok(payloads)
}

pub(crate) fn load_payloads_or_empty(
    segment: &SegmentPaths,
    segment_meta: &SegmentMetadata,
    expected_rows: usize,
) -> io::Result<Vec<BTreeMap<String, FieldValue>>> {
    match load_payloads_for_segment(segment, segment_meta, None) {
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

pub(crate) fn load_records_or_empty(
    path: &Path,
    dimension: usize,
    fp16: bool,
) -> io::Result<Vec<f32>> {
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

pub(crate) fn load_record_ids_or_empty(path: &Path) -> io::Result<Vec<i64>> {
    match load_record_ids(path) {
        Ok(ids) => Ok(ids),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(err) => Err(err),
    }
}

pub(crate) fn materialize_active_segment_arrow_snapshots(
    segment_paths: &SegmentPaths,
    collection_meta: &CollectionMetadata,
) -> io::Result<()> {
    if segment_paths.payloads.exists() {
        let payloads = load_payloads_jsonl(&segment_paths.payloads)?;
        write_payloads_arrow(
            &segment_paths.payloads_arrow,
            &payloads,
            &collection_meta.fields,
        )?;
    }

    if segment_paths.vectors.exists() {
        let vectors = load_vectors_jsonl(&segment_paths.vectors)?;
        write_vectors_arrow(
            &segment_paths.vectors_arrow,
            &vectors,
            &collection_meta.vectors,
            &collection_meta.primary_vector,
        )?;
    }

    Ok(())
}

pub(crate) fn materialize_forward_store_snapshot(
    segment_paths: &SegmentPaths,
    collection_meta: &CollectionMetadata,
) -> io::Result<()> {
    let segment_meta = SegmentMetadata::load_from_path(&segment_paths.metadata)?;
    let stored_ids = load_record_ids_or_empty(&segment_paths.external_ids)?;
    let records = load_records_or_empty(
        &segment_paths.records,
        collection_meta.dimension,
        collection_meta.primary_is_fp16(),
    )?;
    let row_capacity = records.len() / collection_meta.dimension.max(1);
    let row_limit = segment_meta
        .record_count
        .min(stored_ids.len())
        .min(row_capacity);

    if row_limit.saturating_mul(collection_meta.dimension) > records.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "forward_store snapshot record rows are misaligned",
        ));
    }

    let payloads = load_payloads_or_empty(segment_paths, &segment_meta, row_limit)?;
    let vectors = load_vectors_or_empty(segment_paths, &segment_meta, row_limit)?;
    let tombstone = TombstoneMask::load_from_path(&segment_paths.tombstones)?;
    let mut store = MemForwardStore::new(collection_meta.schema());
    let declared_fields = collection_meta
        .fields
        .iter()
        .map(|field| field.name.clone())
        .collect::<BTreeSet<_>>();

    for row_idx in 0..row_limit {
        let internal_id = u64::try_from(stored_ids[row_idx]).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "forward_store snapshot cannot encode negative internal id {}",
                    stored_ids[row_idx]
                ),
            )
        })?;
        let mut row_vectors = vectors[row_idx].clone();
        let start = row_idx * collection_meta.dimension;
        let end = start + collection_meta.dimension;
        row_vectors.insert(
            collection_meta.primary_vector.clone(),
            records[start..end].to_vec(),
        );
        let forward_fields = payloads[row_idx]
            .iter()
            .filter(|(field_name, _)| declared_fields.contains(field_name.as_str()))
            .map(|(field_name, value)| (field_name.clone(), value.clone()))
            .collect::<BTreeMap<_, _>>();
        store.append(ForwardRow {
            internal_id,
            op_seq: row_idx as u64 + 1,
            is_deleted: tombstone.is_deleted(row_idx),
            fields: forward_fields,
            vectors: row_vectors,
        })?;
    }

    let descriptor = ChunkedFileWriter::new(&segment_paths.dir).write(
        "forward_store",
        &store,
        &[ForwardFileFormat::ArrowIpc, ForwardFileFormat::Parquet],
    )?;
    crate::segment::atomic_write(
        &segment_paths.forward_store_descriptor(),
        &serde_json::to_vec_pretty(&descriptor).map_err(json_to_io_error)?,
    )?;
    Ok(())
}

pub(crate) fn materialize_latest_live_forward_store_snapshot(
    segment_manager: &SegmentManager,
    collection_meta: &CollectionMetadata,
) -> io::Result<()> {
    let active_segment = segment_manager.active_segment_path()?;
    let mut store = MemForwardStore::new(collection_meta.schema());
    let mut seen_ids = HashSet::new();
    let mut op_seq = 1_u64;

    for segment in segment_manager.segment_paths()? {
        let segment_meta = SegmentMetadata::load_from_path(&segment.metadata)?;
        let records = load_records_or_empty(
            &segment.records,
            collection_meta.dimension,
            collection_meta.primary_is_fp16(),
        )?;
        let stored_ids = load_record_ids_or_empty(&segment.external_ids)?;
        let payloads = load_payloads_or_empty(&segment, &segment_meta, stored_ids.len())?;
        let vectors = load_vectors_or_empty(&segment, &segment_meta, stored_ids.len())?;
        let tombstone = TombstoneMask::load_from_path(&segment.tombstones)?;

        let row_limit = segment_meta
            .record_count
            .min(stored_ids.len())
            .min(records.len() / collection_meta.dimension.max(1))
            .min(payloads.len())
            .min(vectors.len());

        for row_idx in (0..row_limit).rev() {
            let internal_id = stored_ids[row_idx];
            if !seen_ids.insert(internal_id) || tombstone.is_deleted(row_idx) {
                continue;
            }
            let internal_id = u64::try_from(internal_id).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "forward_store latest-live snapshot cannot encode negative internal id {}",
                        stored_ids[row_idx]
                    ),
                )
            })?;
            let start = row_idx * collection_meta.dimension;
            let end = start + collection_meta.dimension;
            let mut row_vectors = vectors[row_idx].clone();
            row_vectors.insert(
                collection_meta.primary_vector.clone(),
                records[start..end].to_vec(),
            );
            store.append(ForwardRow {
                internal_id,
                op_seq,
                is_deleted: false,
                fields: payloads[row_idx].clone(),
                vectors: row_vectors,
            })?;
            op_seq += 1;
        }
    }

    let descriptor = ChunkedFileWriter::new(&active_segment.dir).write(
        "forward_store",
        &store,
        &[ForwardFileFormat::ArrowIpc, ForwardFileFormat::Parquet],
    )?;
    crate::segment::atomic_write(
        &active_segment.forward_store_descriptor(),
        &serde_json::to_vec_pretty(&descriptor).map_err(json_to_io_error)?,
    )?;
    Ok(())
}

pub(crate) fn invalidate_forward_store_snapshot(segment_paths: &SegmentPaths) -> io::Result<()> {
    for path in [
        segment_paths.forward_store_descriptor(),
        segment_paths.forward_store_artifact(ForwardFileFormat::ArrowIpc),
        segment_paths.forward_store_artifact(ForwardFileFormat::Parquet),
    ] {
        match std::fs::remove_file(path) {
            Ok(()) => {}
            Err(err) if err.kind() == io::ErrorKind::NotFound => {}
            Err(err) => return Err(err),
        }
    }
    Ok(())
}

pub(crate) fn load_vectors_or_empty(
    segment: &SegmentPaths,
    segment_meta: &SegmentMetadata,
    expected_rows: usize,
) -> io::Result<Vec<BTreeMap<String, Vec<f32>>>> {
    match load_vectors_for_segment(segment, segment_meta) {
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

fn load_vectors_for_segment(
    segment: &SegmentPaths,
    segment_meta: &SegmentMetadata,
) -> io::Result<Vec<BTreeMap<String, Vec<f32>>>> {
    match segment_meta.normalized_storage_format() {
        NormalizedStorageFormat::Arrow => {
            if segment.vectors_arrow.exists() {
                match load_vectors(&segment.vectors) {
                    Ok(vectors) => return Ok(vectors),
                    Err(err) if segment.vectors.exists() => {
                        let _ = err;
                    }
                    Err(err) => return Err(err),
                }
            }
            load_vectors_jsonl(&segment.vectors)
        }
        NormalizedStorageFormat::Jsonl => {
            if segment.vectors.exists() {
                return load_vectors_jsonl(&segment.vectors);
            }
            load_vectors(&segment.vectors)
        }
    }
}

pub(crate) fn load_payloads_with_fields_or_empty(
    segment: &SegmentPaths,
    segment_meta: &SegmentMetadata,
    expected_rows: usize,
    fields: Option<&[String]>,
) -> io::Result<Vec<BTreeMap<String, FieldValue>>> {
    match load_payloads_for_segment(segment, segment_meta, fields) {
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

pub(crate) fn load_sparse_vectors_or_empty(
    path: &Path,
    expected_rows: usize,
) -> io::Result<Vec<BTreeMap<String, SparseVector>>> {
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

pub(crate) fn load_shadowed_live_records(
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

pub(crate) fn load_shadowed_live_vector_records(
    segment_manager: &SegmentManager,
    dimension: usize,
    field_name: &str,
) -> io::Result<(Vec<f32>, Vec<i64>)> {
    let mut records = Vec::new();
    let mut external_ids = Vec::new();
    let mut shadowed_ids = HashSet::new();

    for segment in segment_manager.segment_paths()? {
        let segment_meta = SegmentMetadata::load_from_path(&segment.metadata)?;
        let segment_external_ids = load_record_ids_or_empty(&segment.external_ids)?;
        let segment_vectors =
            load_vectors_or_empty(&segment, &segment_meta, segment_external_ids.len())?;
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

pub(crate) fn persisted_ann_exists(
    collection_dir: &Path,
    field_name: &str,
    is_primary: bool,
) -> bool {
    ann_blob_path(collection_dir, field_name).exists()
        || (is_primary && collection_dir.join(HNSW_INDEX_FILE).exists())
}

pub(crate) fn load_all_collection_ids(paths: &CollectionPaths) -> io::Result<Vec<i64>> {
    let segment_paths = SegmentManager::new(paths.dir.clone()).segment_paths()?;
    let mut all_ids = Vec::new();
    for segment in segment_paths {
        all_ids.extend(load_record_ids_or_empty(&segment.external_ids)?);
    }
    Ok(all_ids)
}

pub(crate) fn latest_live_row_index(
    stored_ids: &[i64],
    tombstone: &TombstoneMask,
    external_id: i64,
) -> Option<usize> {
    latest_row_index_for_id(stored_ids, external_id)
        .filter(|row_idx| !tombstone.is_deleted(*row_idx))
}

pub(crate) fn latest_row_index_for_id(stored_ids: &[i64], external_id: i64) -> Option<usize> {
    stored_ids
        .iter()
        .enumerate()
        .rev()
        .find_map(|(row_idx, stored_id)| (*stored_id == external_id).then_some(row_idx))
}

pub(crate) fn next_compacted_segment_id<'a>(
    segment_ids: impl Iterator<Item = &'a String>,
) -> String {
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

fn json_to_io_error(err: serde_json::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, err)
}
