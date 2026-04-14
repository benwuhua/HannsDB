use std::io;
use std::path::Path;

use std::collections::{HashMap, HashSet};

use crate::catalog::ManifestMetadata;
use crate::segment::{load_vectors, SegmentMetadata};
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
            let segment_meta = if paths.segment_meta.exists() {
                Some(SegmentMetadata::load_from_path(&paths.segment_meta)?)
            } else {
                None
            };
            if !manifest.collections.iter().any(|entry| entry == collection) {
                return Ok(true);
            }
            if !paths.collection_meta.exists()
                || segment_meta.is_none()
                || !paths.tombstones.exists()
            {
                return Ok(true);
            }
            if plan.requires_data_files
                && (!paths.records.exists()
                    || !paths.external_ids.exists()
                    || !paths.payloads.exists())
            {
                return Ok(true);
            }
            if plan.requires_vector_sidecar {
                match load_vectors(&paths.vectors) {
                    Ok(vectors) => {
                        if Some(vectors.len())
                            != segment_meta.as_ref().map(|meta| meta.record_count)
                        {
                            return Ok(true);
                        }
                    }
                    Err(_) => return Ok(true),
                }
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
