//! Tombstone management: mark rows as deleted across segments.
//!
//! Pure storage-layer operations that load external IDs per segment,
//! locate matching rows, and update the `TombstoneMask` and `SegmentMetadata`.
//! Callers are responsible for WAL logging, cache invalidation, and
//! auto-compaction decisions.

use std::collections::HashSet;
use std::io;

use crate::segment::{SegmentManager, SegmentMetadata, SegmentPaths, TombstoneMask};
use crate::storage::paths::CollectionPaths;
use crate::storage::segment_io::{
    invalidate_forward_store_snapshot, latest_live_row_index, latest_row_index_for_id,
    load_external_ids_for_segment_or_empty,
};

/// Result of a tombstone-marking operation.
pub struct TombstoneResult {
    /// Number of rows that were newly marked as deleted.
    pub deleted_count: usize,
}

/// Mark the given external IDs as deleted across all segments.
///
/// Iterates through segments, finds the latest row index for each ID
/// (whether already deleted or not), and marks it in the tombstone mask.
/// Returns the count of rows that were newly deleted.
pub fn mark_ids_deleted(
    paths: &CollectionPaths,
    external_ids: &[i64],
) -> io::Result<TombstoneResult> {
    if external_ids.is_empty() {
        return Ok(TombstoneResult { deleted_count: 0 });
    }

    let segment_paths = SegmentManager::new(paths.dir.clone()).segment_paths()?;

    let mut newly_deleted = 0usize;
    let mut remaining_ids = external_ids.iter().copied().collect::<HashSet<_>>();

    for segment in segment_paths {
        if remaining_ids.is_empty() {
            break;
        }

        let mut segment_meta = SegmentMetadata::load_from_path(&segment.metadata)?;
        let stored_ids = load_external_ids_for_segment_or_empty(&segment, &segment_meta)?;
        if stored_ids.is_empty() {
            continue;
        }
        let mut tombstone = TombstoneMask::load_from_path(&segment.tombstones)?;
        let mut segment_changed = false;
        let row_limit = segment_meta.record_count.min(stored_ids.len());
        let stored_ids = &stored_ids[..row_limit];
        let ids_to_check = remaining_ids.iter().copied().collect::<Vec<_>>();

        for external_id in ids_to_check {
            let Some(row_idx) = latest_row_index_for_id(stored_ids, external_id) else {
                continue;
            };
            remaining_ids.remove(&external_id);
            if tombstone.is_deleted(row_idx) {
                continue;
            }
            if tombstone.mark_deleted(row_idx) {
                newly_deleted += 1;
                segment_changed = true;
            }
        }

        if segment_changed {
            segment_meta.deleted_count = tombstone.deleted_count();
            segment_meta.save_to_path(&segment.metadata)?;
            tombstone.save_to_path(&segment.tombstones)?;
            invalidate_forward_store_snapshot(&segment)?;
        }
    }

    Ok(TombstoneResult {
        deleted_count: newly_deleted,
    })
}

/// Mark the given external IDs as deleted across all segments, but only
/// if the corresponding row is still live (not already tombstoned).
///
/// Used by the upsert path to soft-delete old versions of a document
/// before appending the new version.  Returns the count of newly deleted rows.
pub fn mark_live_ids_deleted_across_segments(
    paths: &CollectionPaths,
    external_ids: &[i64],
) -> io::Result<TombstoneResult> {
    if external_ids.is_empty() {
        return Ok(TombstoneResult { deleted_count: 0 });
    }

    let segment_paths = SegmentManager::new(paths.dir.clone()).segment_paths()?;
    let mut remaining_ids = external_ids.iter().copied().collect::<HashSet<_>>();

    for segment in segment_paths {
        if remaining_ids.is_empty() {
            break;
        }

        let mut segment_meta = SegmentMetadata::load_from_path(&segment.metadata)?;
        let stored_ids = load_external_ids_for_segment_or_empty(&segment, &segment_meta)?;
        if stored_ids.is_empty() {
            continue;
        }
        let mut tombstone = TombstoneMask::load_from_path(&segment.tombstones)?;
        let mut segment_changed = false;
        let row_limit = segment_meta.record_count.min(stored_ids.len());
        let stored_ids = &stored_ids[..row_limit];
        let ids_to_check = remaining_ids.iter().copied().collect::<Vec<_>>();

        for external_id in ids_to_check {
            let Some(row_idx) = latest_live_row_index(stored_ids, &tombstone, external_id) else {
                continue;
            };
            remaining_ids.remove(&external_id);
            if tombstone.mark_deleted(row_idx) {
                segment_changed = true;
            }
        }

        if segment_changed {
            segment_meta.deleted_count = tombstone.deleted_count();
            segment_meta.save_to_path(&segment.metadata)?;
            tombstone.save_to_path(&segment.tombstones)?;
        }
    }

    Ok(TombstoneResult {
        deleted_count: 0,
    })
}
