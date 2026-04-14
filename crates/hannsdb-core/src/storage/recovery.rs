use std::io;
use std::path::Path;

use std::collections::{HashMap, HashSet};

use crate::catalog::{CollectionMetadata, ManifestMetadata};
use crate::document::Document;
use crate::forward_store::{ForwardFileFormat, ForwardStoreDescriptor, ForwardStoreReader};
use crate::segment::TombstoneMask;
use crate::segment::{load_vectors, NormalizedStorageFormat, SegmentManager, SegmentMetadata};
use crate::storage::segment_io::{
    load_payloads_or_empty, load_primary_dense_rows_for_segment_or_empty, load_vectors_or_empty,
};
use crate::storage::paths::CollectionPaths;
use crate::storage::wal::WalRecord;

#[derive(Debug, Default)]
pub(crate) struct WalCollectionPlan {
    pub(crate) created_in_wal: bool,
    pub(crate) requires_delta_replay: bool,
    pub(crate) requires_data_files: bool,
    pub(crate) requires_vector_sidecar: bool,
    delta_expectations: HashMap<i64, ReplayExpectedRow>,
}

#[derive(Debug, Default)]
pub(crate) struct WalReplayPlan {
    collections: HashMap<String, WalCollectionPlan>,
    dropped_collections: HashSet<String>,
}

#[derive(Debug, Clone)]
enum ReplayExpectedRow {
    PresentDocument(Document),
    PresentPrimaryVector(Vec<f32>),
    Deleted,
}

impl WalReplayPlan {
    pub(crate) fn build(records: &[WalRecord]) -> Self {
        let mut collections = HashMap::new();
        let mut dropped_collections = HashSet::new();
        for record in records {
            match record {
                WalRecord::CreateCollection { collection, .. } => {
                    collections.insert(
                        collection.clone(),
                        WalCollectionPlan {
                            created_in_wal: true,
                            ..WalCollectionPlan::default()
                        },
                    );
                    dropped_collections.remove(collection);
                }
                WalRecord::DropCollection { collection } => {
                    collections.remove(collection);
                    dropped_collections.insert(collection.clone());
                }
                WalRecord::Insert {
                    collection, ids, vectors
                } if !ids.is_empty() => {
                    let plan = ensure_collection_plan(&mut collections, collection);
                    plan.requires_data_files = true;
                    plan.requires_vector_sidecar = true;
                    if !plan.created_in_wal {
                        if let Some(dim) = vectors.len().checked_div(ids.len()) {
                            if dim.saturating_mul(ids.len()) == vectors.len() {
                                for (id, vector) in ids.iter().copied().zip(vectors.chunks_exact(dim)) {
                                    plan.delta_expectations.insert(
                                        id,
                                        ReplayExpectedRow::PresentPrimaryVector(vector.to_vec()),
                                    );
                                }
                            } else {
                                plan.requires_delta_replay = true;
                            }
                        } else {
                            plan.requires_delta_replay = true;
                        }
                    }
                }
                WalRecord::InsertDocuments {
                    collection,
                    documents,
                } if !documents.is_empty() => {
                    let plan = ensure_collection_plan(&mut collections, collection);
                    plan.requires_data_files = true;
                    plan.requires_vector_sidecar = true;
                    if !plan.created_in_wal {
                        for document in documents {
                            plan.delta_expectations.insert(
                                document.id,
                                ReplayExpectedRow::PresentDocument(document.clone()),
                            );
                        }
                    }
                }
                WalRecord::UpsertDocuments {
                    collection,
                    documents,
                } if !documents.is_empty() => {
                    let plan = ensure_collection_plan(&mut collections, collection);
                    plan.requires_data_files = true;
                    plan.requires_vector_sidecar = true;
                    if !plan.created_in_wal {
                        for document in documents {
                            plan.delta_expectations.insert(
                                document.id,
                                ReplayExpectedRow::PresentDocument(document.clone()),
                            );
                        }
                    }
                }
                WalRecord::Delete { collection, ids } if !ids.is_empty() => {
                    let plan = ensure_collection_plan(&mut collections, collection);
                    plan.requires_data_files = true;
                    if !plan.created_in_wal {
                        for id in ids {
                            plan.delta_expectations
                                .insert(*id, ReplayExpectedRow::Deleted);
                        }
                    }
                }
                WalRecord::CompactCollection {
                    collection_name, ..
                } => {
                    let plan = ensure_collection_plan(&mut collections, collection_name);
                    plan.requires_data_files = true;
                    if !plan.created_in_wal {
                        plan.requires_delta_replay = true;
                    }
                }
                WalRecord::UpdateDocuments {
                    collection,
                    updates,
                } if !updates.is_empty() => {
                    let plan = ensure_collection_plan(&mut collections, collection);
                    plan.requires_data_files = true;
                    plan.requires_vector_sidecar = true;
                    if !plan.created_in_wal {
                        plan.requires_delta_replay = true;
                    }
                }
                WalRecord::AddColumn { collection, .. }
                | WalRecord::DropColumn { collection, .. }
                | WalRecord::AlterColumn { collection, .. }
                | WalRecord::AddVectorField { collection, .. }
                | WalRecord::DropVectorField { collection, .. } => {
                    let plan = ensure_collection_plan(&mut collections, collection);
                    plan.requires_data_files = true;
                    if !plan.created_in_wal {
                        plan.requires_delta_replay = true;
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
            if !plan.created_in_wal {
                if plan.requires_delta_replay {
                    return Ok(true);
                }
                if !persisted_collection_matches_delta(&paths, plan)? {
                    return Ok(true);
                }
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

    pub(crate) fn collections_to_reset(&self) -> impl Iterator<Item = &str> {
        self.collections
            .iter()
            .filter(|(_, plan)| plan.created_in_wal)
            .map(|(collection, _)| collection.as_str())
            .chain(self.dropped_collections.iter().map(String::as_str))
    }

    pub(crate) fn has_owned_collections(&self) -> bool {
        !(self.collections.is_empty() && self.dropped_collections.is_empty())
    }

    pub(crate) fn owns(&self, collection: &str) -> bool {
        self.collections.contains_key(collection) || self.dropped_collections.contains(collection)
    }
}

fn ensure_collection_plan<'a>(
    collections: &'a mut HashMap<String, WalCollectionPlan>,
    collection: &str,
) -> &'a mut WalCollectionPlan {
    collections.entry(collection.to_string()).or_default()
}

fn persisted_collection_matches_delta(
    paths: &CollectionPaths,
    plan: &WalCollectionPlan,
) -> io::Result<bool> {
    if plan.delta_expectations.is_empty() {
        return Ok(true);
    }

    let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
    let segment_paths = SegmentManager::new(paths.dir.clone()).segment_paths()?;
    let mut actual_rows: HashMap<i64, Option<Document>> = HashMap::new();
    let mut shadowed_ids = HashSet::new();

    for segment in &segment_paths {
        let segment_meta = SegmentMetadata::load_from_path(&segment.metadata)?;
        let dense_rows = load_primary_dense_rows_for_segment_or_empty(
            segment,
            &segment_meta,
            &collection_meta.primary_vector,
            collection_meta.dimension,
            collection_meta.primary_is_fp16(),
        )?;
        let stored_ids = dense_rows.external_ids;
        let records = dense_rows.primary_vectors;
        let payloads = load_payloads_or_empty(segment, &segment_meta, stored_ids.len())?;
        let vectors = load_vectors_or_empty(
            segment,
            &segment_meta,
            &collection_meta.primary_vector,
            stored_ids.len(),
        )?;
        let tombstone = TombstoneMask::load_from_path(&segment.tombstones)?;

        if stored_ids.len().saturating_mul(collection_meta.dimension) != records.len() {
            return Ok(false);
        }

        for row_idx in (0..stored_ids.len()).rev() {
            let external_id = stored_ids[row_idx];
            if !plan.delta_expectations.contains_key(&external_id) || !shadowed_ids.insert(external_id) {
                continue;
            }
            if tombstone.is_deleted(row_idx) {
                actual_rows.insert(external_id, None);
                continue;
            }

            let start = row_idx * collection_meta.dimension;
            let end = start + collection_meta.dimension;
            let mut doc_vectors = vectors[row_idx].clone();
            doc_vectors.insert(
                collection_meta.primary_vector.clone(),
                records[start..end].to_vec(),
            );
            actual_rows.insert(
                external_id,
                Some(Document {
                    id: external_id,
                    fields: payloads[row_idx].clone(),
                    vectors: doc_vectors,
                    sparse_vectors: Default::default(),
                }),
            );
        }
    }

    for (external_id, expected) in &plan.delta_expectations {
        let actual = actual_rows.get(external_id);
        let matches = match (expected, actual) {
            (ReplayExpectedRow::Deleted, None | Some(None)) => true,
            (ReplayExpectedRow::PresentDocument(expected), Some(Some(actual))) => {
                actual.fields == expected.fields
                    && actual.vectors == *expected.vectors_with_primary(&collection_meta.primary_vector)
            }
            (ReplayExpectedRow::PresentPrimaryVector(expected), Some(Some(actual))) => actual
                .primary_vector_for(&collection_meta.primary_vector)
                .map(|vector| vector == expected.as_slice())
                .unwrap_or(false),
            _ => false,
        };
        if !matches {
            return Ok(false);
        }
    }

    Ok(true)
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

    if segment_uses_forward_store_authority(segment_meta)
        && segment.forward_store_descriptor().exists()
    {
        let descriptor = load_forward_store_descriptor(&segment.forward_store_descriptor())?;
        let format = preferred_forward_store_format(&descriptor)?;
        let reader = ForwardStoreReader::open(&descriptor, format)?;
        if reader.row_count() == segment_meta.record_count {
            return Ok(true);
        }
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

fn segment_uses_forward_store_authority(segment_meta: &SegmentMetadata) -> bool {
    matches!(
        segment_meta.normalized_storage_format(),
        NormalizedStorageFormat::ForwardStore | NormalizedStorageFormat::Arrow
    )
}

fn load_forward_store_descriptor(path: &Path) -> io::Result<ForwardStoreDescriptor> {
    let bytes = std::fs::read(path)?;
    serde_json::from_slice(&bytes)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))
}

fn preferred_forward_store_format(
    descriptor: &ForwardStoreDescriptor,
) -> io::Result<ForwardFileFormat> {
    descriptor
        .artifact(ForwardFileFormat::ArrowIpc)
        .map(|_| ForwardFileFormat::ArrowIpc)
        .or_else(|| {
            descriptor
                .artifact(ForwardFileFormat::Parquet)
                .map(|_| ForwardFileFormat::Parquet)
        })
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "forward_store descriptor has no readable artifacts",
            )
        })
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
