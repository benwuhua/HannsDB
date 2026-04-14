use std::collections::BTreeMap;
use std::io;
use std::path::Path;
use std::path::PathBuf;

use super::{
    append_record_ids, append_records, append_records_f16, append_sparse_vectors, load_payloads,
    load_payloads_jsonl, load_record_ids, load_records, load_records_f16, load_vectors,
    load_vectors_jsonl, write_payloads_arrow, write_vectors_arrow, SegmentManager, SegmentMetadata,
    SegmentPaths, TombstoneMask, VersionSet,
};
use crate::catalog::CollectionMetadata;
use crate::document::{Document, FieldValue};
use crate::forward_store::{ChunkedFileWriter, ForwardFileFormat, ForwardRow, MemForwardStore};
use crate::storage::segment_io::{
    invalidate_forward_store_snapshot, materialize_forward_store_snapshot,
};

/// Manages writes to the active segment of a collection, including
/// auto-rollover when the segment exceeds size or tombstone thresholds.
pub struct SegmentWriter {
    collection_dir: PathBuf,
    segment_manager: SegmentManager,
}

impl SegmentWriter {
    pub fn new(collection_dir: PathBuf, segment_manager: SegmentManager) -> Self {
        Self {
            collection_dir,
            segment_manager,
        }
    }

    /// Return the active segment paths, falling back to the legacy root-level
    /// layout when no `segment_set.json` exists.
    fn active_paths(&self) -> io::Result<ActiveSegment> {
        let version_set_path = self.segment_manager.version_set_path();
        if version_set_path.exists() {
            let version_set = self.segment_manager.version_set()?;
            let paths = self.segment_manager.active_segment_path()?;
            let meta = SegmentMetadata::load_from_path(&paths.metadata)?;
            Ok(ActiveSegment {
                paths,
                meta,
                version_set,
                multi_segment: true,
            })
        } else {
            let paths =
                SegmentPaths::from_collection_dir(&self.collection_dir, "seg-0001".to_string());
            let meta = SegmentMetadata::load_from_path(&paths.metadata)?;
            Ok(ActiveSegment {
                paths,
                meta,
                version_set: VersionSet::single("seg-0001"),
                multi_segment: false,
            })
        }
    }

    fn append_batch(
        &self,
        collection_meta: &CollectionMetadata,
        external_ids: &[i64],
        records: &[f32],
        payloads: &[BTreeMap<String, FieldValue>],
        vector_rows: &[BTreeMap<String, Vec<f32>>],
        sparse_rows: &[BTreeMap<String, crate::document::SparseVector>],
        fp16: bool,
    ) -> io::Result<AppendResult> {
        let mut active = self.active_paths()?;
        invalidate_forward_store_snapshot(&active.paths)?;
        let mut tombstone = TombstoneMask::load_from_path(&active.paths.tombstones)?;
        if self.should_rollover(&active.meta, &tombstone) {
            self.do_rollover(&mut active)?;
            active = self.active_paths()?;
            tombstone = TombstoneMask::load_from_path(&active.paths.tombstones)?;
        }

        let existing_count = active.meta.record_count;
        let existing_ids =
            load_rows_or_default(&active.paths.external_ids, existing_count, load_record_ids)?;
        let existing_records = if fp16 {
            load_fp16_records_or_default(
                &active.paths.records,
                collection_meta.dimension,
                existing_count,
            )?
        } else {
            load_records_or_default(
                &active.paths.records,
                collection_meta.dimension,
                existing_count,
            )?
        };
        let existing_payloads =
            load_rows_or_default(&active.paths.payloads, existing_count, load_payloads)?;
        let existing_vectors =
            load_rows_or_default(&active.paths.vectors, existing_count, load_vectors)?;

        let inserted = if fp16 {
            append_records_f16(&active.paths.records, collection_meta.dimension, records)?
        } else {
            append_records(&active.paths.records, collection_meta.dimension, records)?
        };
        let _ = append_record_ids(&active.paths.external_ids, external_ids)?;
        let _ = append_sparse_vectors(&active.paths.sparse_vectors, sparse_rows)?;

        let mut all_ids = existing_ids;
        all_ids.extend_from_slice(external_ids);

        let mut all_records = existing_records;
        all_records.extend_from_slice(records);

        let mut all_payloads = existing_payloads;
        all_payloads.extend(payloads.iter().cloned());

        let mut all_vectors = existing_vectors;
        all_vectors.extend(vector_rows.iter().cloned());

        active.meta.record_count += inserted;
        active.meta.deleted_count = tombstone.deleted_count();
        active.meta.storage_format = "forward_store".to_string();

        let needed = active.meta.record_count.saturating_sub(tombstone.len());
        if needed > 0 {
            tombstone.extend(needed);
        }

        persist_forward_store_artifacts(
            &active.paths.dir,
            collection_meta,
            &all_ids,
            &all_records,
            &all_payloads,
            &all_vectors,
            &tombstone,
        )?;

        active.meta.save_to_path(&active.paths.metadata)?;
        tombstone.save_to_path(&active.paths.tombstones)?;

        let rolled_over = if self.should_rollover(&active.meta, &tombstone) {
            self.do_rollover(&mut active)?;
            true
        } else {
            false
        };

        Ok(AppendResult {
            inserted,
            rolled_over,
        })
    }

    /// Append raw vectors and IDs to the active segment.
    pub fn append_records(
        &self,
        collection_meta: &CollectionMetadata,
        external_ids: &[i64],
        vectors: &[f32],
        payloads: &[BTreeMap<String, FieldValue>],
        vector_rows: &[BTreeMap<String, Vec<f32>>],
        sparse_rows: &[BTreeMap<String, crate::document::SparseVector>],
        fp16: bool,
    ) -> io::Result<AppendResult> {
        self.append_batch(
            collection_meta,
            external_ids,
            vectors,
            payloads,
            vector_rows,
            sparse_rows,
            fp16,
        )
    }

    pub fn append_documents(
        &self,
        collection_meta: &CollectionMetadata,
        documents: &[Document],
    ) -> io::Result<AppendResult> {
        let mut ids = Vec::with_capacity(documents.len());
        let mut records =
            Vec::with_capacity(documents.len().saturating_mul(collection_meta.dimension));
        let mut payloads = Vec::with_capacity(documents.len());
        let mut vectors = Vec::with_capacity(documents.len());
        let mut sparse_rows = Vec::with_capacity(documents.len());

        for document in documents {
            ids.push(document.id);
            let primary = document
                .vectors
                .get(collection_meta.primary_vector.as_str())
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!(
                            "document {} is missing primary vector '{}'",
                            document.id, collection_meta.primary_vector
                        ),
                    )
                })?;
            records.extend_from_slice(primary);
            payloads.push(document.fields.clone());
            let mut secondary = document.vectors.clone();
            secondary.remove(collection_meta.primary_vector.as_str());
            vectors.push(secondary);
            sparse_rows.push(document.sparse_vectors.clone());
        }

        self.append_batch(
            collection_meta,
            &ids,
            &records,
            &payloads,
            &vectors,
            &sparse_rows,
            collection_meta.primary_is_fp16(),
        )
    }

    /// Update tombstone count and save. Used by upsert (which tombstones
    /// old rows) before appending new rows.
    pub fn save_tombstone(&self, tombstone: &TombstoneMask, record_count: usize) -> io::Result<()> {
        let active = self.active_paths()?;
        let mut meta = active.meta;
        meta.record_count = record_count;
        meta.deleted_count = tombstone.deleted_count();
        meta.save_to_path(&active.paths.metadata)?;
        tombstone.save_to_path(&active.paths.tombstones)?;
        invalidate_forward_store_snapshot(&active.paths)?;
        Ok(())
    }

    fn should_rollover(&self, meta: &SegmentMetadata, tombstone: &TombstoneMask) -> bool {
        super::segment_set::SegmentSet::should_rollover(
            meta.record_count as u64,
            tombstone.deleted_count() as u64,
        )
    }

    fn do_rollover(&self, active: &mut ActiveSegment) -> io::Result<()> {
        let mut version_set = active.version_set.clone();
        let new_id = version_set.rollover();

        // If this was a legacy single-segment layout, we need to migrate
        // the root-level files into segments/seg-000001/.
        let old_dir = if !active.multi_segment {
            let segments_dir = self.segment_manager.segments_dir();
            let old_seg_dir = segments_dir.join(&active.paths.segment_id);
            std::fs::create_dir_all(&segments_dir)?;
            std::fs::create_dir_all(&old_seg_dir)?;

            // Move root-level segment files into segments/<old_id>/
            let files_to_move = [
                "segment.json",
                "records.bin",
                "ids.bin",
                "payloads.jsonl",
                "payloads.arrow",
                "vectors.jsonl",
                "vectors.arrow",
                "sparse_vectors.jsonl",
                "forward_store.arrow",
                "forward_store.parquet",
                "tombstones.json",
            ];
            for file_name in &files_to_move {
                let src = self.collection_dir.join(file_name);
                if src.exists() {
                    let dst = old_seg_dir.join(file_name);
                    std::fs::rename(&src, &dst)?;
                }
            }
            old_seg_dir
        } else {
            active.paths.dir.clone()
        };

        let collection_meta_path = self.collection_dir.join("collection.json");
        if !collection_meta_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "collection.json not found",
            ));
        }
        let collection_meta = CollectionMetadata::load_from_path(&collection_meta_path)?;
        let sealed_paths =
            SegmentPaths::from_segment_dir(old_dir.clone(), active.paths.segment_id.clone());
        materialize_forward_store_snapshot(&sealed_paths, &collection_meta)?;

        // Convert JSONL → Arrow IPC for the now-immutable segment.
        if let Err(e) = self.convert_jsonl_to_arrow(&old_dir, &collection_meta) {
            log::warn!(
                "JSONL→Arrow conversion failed for {}, keeping JSONL: {}",
                active.paths.segment_id,
                e
            );
        }

        // Create the new active segment directory
        self.segment_manager
            .create_segment_dir(&new_id, active.meta.dimension)?;

        // Persist the updated version set
        version_set.save_to_path(&self.segment_manager.version_set_path())?;

        log::info!(
            "Segment rollover: {} → {} (immutable: {:?})",
            active.paths.segment_id,
            new_id,
            version_set.immutable_segment_ids(),
        );

        Ok(())
    }

    /// Convert the JSONL payload and vector files in a segment directory to
    /// Arrow IPC. On success, delete the JSONL files and update the segment
    /// metadata. On failure, leave the JSONL files intact.
    fn convert_jsonl_to_arrow(
        &self,
        seg_dir: &std::path::Path,
        collection_meta: &CollectionMetadata,
    ) -> io::Result<()> {
        let jsonl_payloads = seg_dir.join("payloads.jsonl");
        let jsonl_vectors = seg_dir.join("vectors.jsonl");
        let arrow_payloads = seg_dir.join("payloads.arrow");
        let arrow_vectors = seg_dir.join("vectors.arrow");

        // Convert payloads.
        if jsonl_payloads.exists() {
            let payloads = load_payloads_jsonl(&jsonl_payloads)?;
            write_payloads_arrow(&arrow_payloads, &payloads, &collection_meta.fields)?;
            let _ = std::fs::remove_file(&jsonl_payloads);
        }

        // Convert vectors sidecar.
        if jsonl_vectors.exists() {
            let vectors = load_vectors_jsonl(&jsonl_vectors)?;
            write_vectors_arrow(
                &arrow_vectors,
                &vectors,
                &collection_meta.vectors,
                &collection_meta.primary_vector,
            )?;
            let _ = std::fs::remove_file(&jsonl_vectors);
        }

        // Update segment metadata to record arrow format.
        let meta_path = seg_dir.join("segment.json");
        if meta_path.exists() {
            let mut meta = SegmentMetadata::load_from_path(&meta_path)?;
            meta.storage_format = "forward_store".to_string();
            meta.save_to_path(&meta_path)?;
        }

        Ok(())
    }
}

fn load_rows_or_default<T, F>(path: &Path, expected_rows: usize, loader: F) -> io::Result<Vec<T>>
where
    T: Clone + Default,
    F: Fn(&Path) -> io::Result<Vec<T>>,
{
    match loader(path) {
        Ok(mut rows) => {
            if rows.len() > expected_rows {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "row count exceeds expected active-segment row count",
                ));
            }
            rows.resize(expected_rows, T::default());
            Ok(rows)
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(vec![T::default(); expected_rows]),
        Err(err) => Err(err),
    }
}

fn load_records_or_default(
    path: &Path,
    dimension: usize,
    expected_rows: usize,
) -> io::Result<Vec<f32>> {
    match load_records(path, dimension) {
        Ok(records) => {
            let expected_len = expected_rows.saturating_mul(dimension);
            if records.len() > expected_len {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "record vector count exceeds expected active-segment row count",
                ));
            }
            if records.len() < expected_len {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "record vector count is shorter than expected active-segment row count",
                ));
            }
            Ok(records)
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(err) => Err(err),
    }
}

fn load_fp16_records_or_default(
    path: &Path,
    dimension: usize,
    expected_rows: usize,
) -> io::Result<Vec<f32>> {
    match load_records_f16(path, dimension) {
        Ok(records) => {
            let expected_len = expected_rows.saturating_mul(dimension);
            if records.len() > expected_len {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "record vector count exceeds expected active-segment row count",
                ));
            }
            if records.len() < expected_len {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "record vector count is shorter than expected active-segment row count",
                ));
            }
            Ok(records)
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(err) => Err(err),
    }
}

fn persist_forward_store_artifacts(
    segment_dir: &Path,
    collection_meta: &CollectionMetadata,
    external_ids: &[i64],
    records: &[f32],
    payloads: &[BTreeMap<String, FieldValue>],
    vector_rows: &[BTreeMap<String, Vec<f32>>],
    tombstone: &TombstoneMask,
) -> io::Result<()> {
    if external_ids.len() != payloads.len() || external_ids.len() != vector_rows.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "forward-store sidecar rows are misaligned with external ids",
        ));
    }
    if records.len() != external_ids.len().saturating_mul(collection_meta.dimension) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "primary records are misaligned with external ids",
        ));
    }

    let mut store = MemForwardStore::new(collection_meta.schema());
    let declared_fields = collection_meta
        .fields
        .iter()
        .map(|field| field.name.clone())
        .collect::<std::collections::BTreeSet<_>>();
    let mut compatibility_payloads = Vec::with_capacity(external_ids.len());
    let mut compatibility_vectors = Vec::with_capacity(external_ids.len());

    for (row_idx, external_id) in external_ids.iter().enumerate() {
        let start = row_idx * collection_meta.dimension;
        let end = start + collection_meta.dimension;
        let mut row_vectors = vector_rows[row_idx].clone();
        row_vectors.insert(
            collection_meta.primary_vector.clone(),
            records[start..end].to_vec(),
        );
        let forward_fields = payloads[row_idx]
            .iter()
            .filter(|(field_name, _)| declared_fields.contains(field_name.as_str()))
            .map(|(field_name, value)| (field_name.clone(), value.clone()))
            .collect::<BTreeMap<_, _>>();
        compatibility_payloads.push(payloads[row_idx].clone());
        compatibility_vectors.push(vector_rows[row_idx].clone());
        store.append(ForwardRow {
            internal_id: *external_id as u64,
            op_seq: row_idx as u64 + 1,
            is_deleted: tombstone.is_deleted(row_idx),
            fields: forward_fields,
            vectors: row_vectors,
        })?;
    }

    let writer = ChunkedFileWriter::new(segment_dir);
    let _ = writer.write(
        "forward_store",
        &store,
        &[ForwardFileFormat::ArrowIpc, ForwardFileFormat::Parquet],
    )?;
    write_payloads_arrow(
        &segment_dir.join("payloads.arrow"),
        &compatibility_payloads,
        &collection_meta.fields,
    )?;
    write_vectors_arrow(
        &segment_dir.join("vectors.arrow"),
        &compatibility_vectors,
        &collection_meta.vectors,
        &collection_meta.primary_vector,
    )?;

    remove_if_exists(&segment_dir.join("payloads.jsonl"))?;
    remove_if_exists(&segment_dir.join("vectors.jsonl"))?;
    Ok(())
}

fn remove_if_exists(path: &Path) -> io::Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err),
    }
}

pub struct ActiveSegment {
    pub paths: SegmentPaths,
    pub meta: SegmentMetadata,
    pub version_set: VersionSet,
    pub multi_segment: bool,
}

pub struct AppendResult {
    pub inserted: usize,
    pub rolled_over: bool,
}
