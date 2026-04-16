use std::collections::BTreeMap;
use std::io;
use std::path::PathBuf;

use super::{
    append_payloads, append_record_ids, append_records, append_records_f16, append_sparse_vectors,
    append_vectors, load_payloads_jsonl, load_vectors_jsonl, write_payloads_arrow,
    write_vectors_arrow, SegmentManager, SegmentMetadata, SegmentPaths, TombstoneMask, VersionSet,
};
use crate::catalog::CollectionMetadata;
use crate::document::{Document, FieldValue};
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

        // O(m) appends — do NOT load existing rows; Arrow snapshot is deferred
        // to rollover/compaction so the hot path stays O(batch size).
        let inserted = if fp16 {
            append_records_f16(&active.paths.records, collection_meta.dimension, records)?
        } else {
            append_records(&active.paths.records, collection_meta.dimension, records)?
        };
        let _ = append_record_ids(&active.paths.external_ids, external_ids)?;
        let _ = append_payloads(&active.paths.payloads, payloads)?;
        let _ = append_vectors(&active.paths.vectors, vector_rows)?;
        let _ = append_sparse_vectors(&active.paths.sparse_vectors, sparse_rows)?;

        active.meta.record_count += inserted;
        active.meta.deleted_count = tombstone.deleted_count();
        active.meta.storage_format = "jsonl".to_string();

        let needed = active.meta.record_count.saturating_sub(tombstone.len());
        if needed > 0 {
            tombstone.extend(needed);
        }

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
