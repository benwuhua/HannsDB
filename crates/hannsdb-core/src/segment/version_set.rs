use std::io;
use std::path::Path;

use crate::catalog::COLLECTION_RUNTIME_FORMAT_VERSION;

use super::SegmentSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VersionSet {
    active_segment_id: String,
    immutable_segment_ids: Vec<String>,
}

impl VersionSet {
    pub fn new(active_segment_id: impl Into<String>, immutable_segment_ids: Vec<String>) -> Self {
        Self {
            active_segment_id: active_segment_id.into(),
            immutable_segment_ids,
        }
    }

    pub fn single(segment_id: impl Into<String>) -> Self {
        Self::new(segment_id, Vec::new())
    }

    pub fn from_segment_set(segment_set: SegmentSet) -> Self {
        Self {
            active_segment_id: segment_set.active_segment_id,
            immutable_segment_ids: segment_set.immutable_segment_ids,
        }
    }

    pub fn load_from_path(path: &Path) -> io::Result<Self> {
        SegmentSet::load_from_path(path).map(Self::from_segment_set)
    }

    pub fn active_segment_id(&self) -> &str {
        &self.active_segment_id
    }

    pub fn immutable_segment_ids(&self) -> &[String] {
        &self.immutable_segment_ids
    }

    pub fn all_segment_ids(&self) -> Vec<String> {
        let mut ids = Vec::with_capacity(1 + self.immutable_segment_ids.len());
        ids.push(self.active_segment_id.clone());
        ids.extend(self.immutable_segment_ids.iter().cloned());
        ids
    }

    pub fn format_version(&self) -> u32 {
        COLLECTION_RUNTIME_FORMAT_VERSION
    }

    pub fn into_segment_set(self) -> SegmentSet {
        SegmentSet {
            active_segment_id: self.active_segment_id,
            immutable_segment_ids: self.immutable_segment_ids,
        }
    }
}
