use std::io;
use std::path::Path;

use std::collections::{HashMap, HashSet};

use crate::catalog::ManifestMetadata;
use crate::segment::{load_vectors, SegmentManager, SegmentMetadata};
use crate::storage::paths::CollectionPaths;
use crate::storage::wal::WalRecord;

#[derive(Debug, Default)]
pub(crate) struct WalCollectionPlan {
    pub(crate) requires_data_files: bool,
    pub(crate) requires_vector_sidecar: bool,
}

#[derive(Debug, Default)]
pub(crate) struct WalReplayPlan {
    collections: HashMap<String, WalCollectionPlan>,
    dropped_collections: HashSet<String>,
}

impl WalReplayPlan {
    pub(crate) fn build(records: &[WalRecord]) -> Self {
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
                        plan.requires_vector_sidecar = true;
                    }
                }
                WalRecord::InsertDocuments {
                    collection,
                    documents,
                } if !documents.is_empty() => {
                    if let Some(plan) = collections.get_mut(collection) {
                        plan.requires_data_files = true;
                        plan.requires_vector_sidecar = true;
                    }
                }
                WalRecord::UpsertDocuments {
                    collection,
                    documents,
                } if !documents.is_empty() => {
                    if let Some(plan) = collections.get_mut(collection) {
                        plan.requires_data_files = true;
                        plan.requires_vector_sidecar = true;
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

    pub(crate) fn requires_replay<F>(
        &self,
        manifest_path: &Path,
        mut collection_paths: F,
    ) -> io::Result<bool>
    where
        F: FnMut(&str) -> CollectionPaths,
    {
        let manifest = ManifestMetadata::load_from_path(manifest_path)?;
        for (collection, plan) in &self.collections {
            let paths = collection_paths(collection);
            if !manifest.collections.iter().any(|entry| entry == collection) {
                return Ok(true);
            }
            if !paths.collection_meta.exists() {
                return Ok(true);
            }
            if paths.segment_set.exists() {
                if requires_replay_for_segment_set(&paths, plan)? {
                    return Ok(true);
                }
                continue;
            }

            let segment_meta = if paths.segment_meta.exists() {
                Some(SegmentMetadata::load_from_path(&paths.segment_meta)?)
            } else {
                None
            };
            if !segment_is_authoritative(
                &crate::segment::SegmentPaths::from_collection_dir(
                    &paths.dir,
                    segment_meta
                        .as_ref()
                        .map(|meta| meta.segment_id.clone())
                        .unwrap_or_else(|| "seg-0001".to_string()),
                ),
                segment_meta.as_ref(),
                plan,
            )? {
                return Ok(true);
            }
        }
        for collection in &self.dropped_collections {
            let paths = collection_paths(collection);
            if manifest.collections.iter().any(|entry| entry == collection) || paths.dir.exists() {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(crate) fn owned_collections(&self) -> impl Iterator<Item = &str> {
        self.collections
            .keys()
            .chain(self.dropped_collections.iter())
            .map(String::as_str)
    }

    pub(crate) fn has_owned_collections(&self) -> bool {
        !(self.collections.is_empty() && self.dropped_collections.is_empty())
    }

    pub(crate) fn owns(&self, collection: &str) -> bool {
        self.collections.contains_key(collection) || self.dropped_collections.contains(collection)
    }
}

fn requires_replay_for_segment_set(
    paths: &CollectionPaths,
    plan: &WalCollectionPlan,
) -> io::Result<bool> {
    let segment_manager = SegmentManager::new(paths.dir.clone());
    for segment in segment_manager.segment_paths()? {
        let segment_meta = match SegmentMetadata::load_from_path(&segment.metadata) {
            Ok(meta) => meta,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(true),
            Err(err) => return Err(err),
        };
        if !segment_is_authoritative(&segment, Some(&segment_meta), plan)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn segment_is_authoritative(
    segment: &crate::segment::SegmentPaths,
    segment_meta: Option<&SegmentMetadata>,
    plan: &WalCollectionPlan,
) -> io::Result<bool> {
    let Some(segment_meta) = segment_meta else {
        return Ok(false);
    };

    if !segment.tombstones.exists() {
        return Ok(false);
    }

    if segment_meta.record_count > 0
        && (!segment.records.exists() || !segment.external_ids.exists())
    {
        return Ok(false);
    }

    if plan.requires_data_files && !segment.payloads.exists() && !segment.payloads_arrow.exists() {
        return Ok(false);
    }

    if plan.requires_vector_sidecar {
        match load_vectors(&segment.vectors) {
            Ok(vectors) => {
                if vectors.len() != segment_meta.record_count {
                    return Ok(false);
                }
            }
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(false),
            Err(err) => return Err(err),
        }
    }

    Ok(true)
}

pub(crate) fn collection_name_for_wal_record(record: &WalRecord) -> &str {
    match record {
        WalRecord::CreateCollection { collection, .. }
        | WalRecord::DropCollection { collection }
        | WalRecord::Insert { collection, .. }
        | WalRecord::InsertDocuments { collection, .. }
        | WalRecord::UpsertDocuments { collection, .. }
        | WalRecord::Delete { collection, .. }
        | WalRecord::UpdateDocuments { collection, .. }
        | WalRecord::AddColumn { collection, .. }
        | WalRecord::DropColumn { collection, .. }
        | WalRecord::AlterColumn { collection, .. }
        | WalRecord::AddVectorField { collection, .. }
        | WalRecord::DropVectorField { collection, .. } => collection,
        WalRecord::CompactCollection {
            collection_name, ..
        } => collection_name,
    }
}
