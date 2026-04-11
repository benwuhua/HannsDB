use std::collections::BTreeMap;
use std::io;
use std::path::PathBuf;

use super::{
    append_payloads, append_record_ids, append_records, append_vectors, ensure_payload_rows,
    ensure_vector_rows, load_payloads_jsonl, load_vectors_jsonl, write_payloads_arrow,
    write_vectors_arrow, SegmentManager, SegmentMetadata, SegmentPaths, TombstoneMask, VersionSet,
};
use crate::catalog::CollectionMetadata;

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

    /// Append raw vectors and IDs to the active segment.
    pub fn append_records(
        &self,
        dimension: usize,
        external_ids: &[i64],
        vectors: &[f32],
        payloads: &[BTreeMap<String, crate::document::FieldValue>],
        vector_rows: &[BTreeMap<String, Vec<f32>>],
    ) -> io::Result<AppendResult> {
        let mut active = self.active_paths()?;

        ensure_payload_rows(&active.paths.payloads, active.meta.record_count)?;
        ensure_vector_rows(&active.paths.vectors, active.meta.record_count)?;

        let mut tombstone = TombstoneMask::load_from_path(&active.paths.tombstones)?;

        let inserted = append_records(&active.paths.records, dimension, vectors)?;
        let _ = append_record_ids(&active.paths.external_ids, external_ids)?;
        let _ = append_payloads(&active.paths.payloads, payloads)?;
        let _ = append_vectors(&active.paths.vectors, vector_rows)?;

        active.meta.record_count += inserted;
        active.meta.deleted_count = tombstone.deleted_count();

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

    /// Update tombstone count and save. Used by upsert (which tombstones
    /// old rows) before appending new rows.
    pub fn save_tombstone(&self, tombstone: &TombstoneMask, record_count: usize) -> io::Result<()> {
        let active = self.active_paths()?;
        let mut meta = active.meta;
        meta.record_count = record_count;
        meta.deleted_count = tombstone.deleted_count();
        meta.save_to_path(&active.paths.metadata)?;
        tombstone.save_to_path(&active.paths.tombstones)?;
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
        if !active.multi_segment {
            let segments_dir = self.segment_manager.segments_dir();
            let old_seg_dir = segments_dir.join(&active.paths.segment_id);
            std::fs::create_dir_all(&segments_dir)?;

            // Move root-level segment files into segments/<old_id>/
            let files_to_move = [
                "segment.json",
                "records.bin",
                "ids.bin",
                "payloads.jsonl",
                "vectors.jsonl",
                "tombstones.json",
            ];
            for file_name in &files_to_move {
                let src = self.collection_dir.join(file_name);
                if src.exists() {
                    let dst = old_seg_dir.join(file_name);
                    std::fs::create_dir_all(old_seg_dir.parent().unwrap())?;
                    std::fs::rename(&src, &dst)?;
                }
            }
        }

        // Convert JSONL → Arrow IPC for the now-immutable segment.
        let old_dir = active.paths.dir.clone();
        if let Err(e) = self.convert_jsonl_to_arrow(&old_dir) {
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
    fn convert_jsonl_to_arrow(&self, seg_dir: &std::path::Path) -> io::Result<()> {
        let jsonl_payloads = seg_dir.join("payloads.jsonl");
        let jsonl_vectors = seg_dir.join("vectors.jsonl");
        let arrow_payloads = seg_dir.join("payloads.arrow");
        let arrow_vectors = seg_dir.join("vectors.arrow");

        // Load collection metadata for schema info.
        let collection_meta_path = self.collection_dir.join("collection.json");
        if !collection_meta_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "collection.json not found",
            ));
        }
        let collection_meta = CollectionMetadata::load_from_path(&collection_meta_path)?;

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
            meta.storage_format = "arrow".to_string();
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
