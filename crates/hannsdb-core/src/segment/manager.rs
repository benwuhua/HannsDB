use std::io;
use std::path::{Path, PathBuf};

use super::{SegmentMetadata, VersionSet};

#[derive(Debug, Clone)]
pub struct SegmentPaths {
    pub segment_id: String,
    pub dir: PathBuf,
    pub metadata: PathBuf,
    pub records: PathBuf,
    pub external_ids: PathBuf,
    pub payloads: PathBuf,
    pub vectors: PathBuf,
    pub tombstones: PathBuf,
}

impl SegmentPaths {
    pub(crate) fn from_collection_dir(collection_dir: &Path, segment_id: String) -> Self {
        Self {
            segment_id,
            dir: collection_dir.to_path_buf(),
            metadata: collection_dir.join("segment.json"),
            records: collection_dir.join("records.bin"),
            external_ids: collection_dir.join("ids.bin"),
            payloads: collection_dir.join("payloads.jsonl"),
            vectors: collection_dir.join("vectors.jsonl"),
            tombstones: collection_dir.join("tombstones.json"),
        }
    }

    pub(crate) fn from_segment_dir(segment_dir: PathBuf, segment_id: String) -> Self {
        Self {
            segment_id,
            metadata: segment_dir.join("segment.json"),
            records: segment_dir.join("records.bin"),
            external_ids: segment_dir.join("ids.bin"),
            payloads: segment_dir.join("payloads.jsonl"),
            vectors: segment_dir.join("vectors.jsonl"),
            tombstones: segment_dir.join("tombstones.json"),
            dir: segment_dir,
        }
    }

    pub fn ann_dir(&self) -> PathBuf {
        self.dir.join("ann")
    }
}

#[derive(Debug, Clone)]
pub struct SegmentManager {
    collection_dir: PathBuf,
}

impl SegmentManager {
    pub fn new(collection_dir: PathBuf) -> Self {
        Self { collection_dir }
    }

    pub fn collection_dir(&self) -> &Path {
        &self.collection_dir
    }

    pub fn segments_dir(&self) -> PathBuf {
        self.collection_dir.join("segments")
    }

    pub fn version_set_path(&self) -> PathBuf {
        self.collection_dir.join("segment_set.json")
    }

    pub fn version_set(&self) -> io::Result<VersionSet> {
        let version_path = self.version_set_path();
        if version_path.exists() {
            return VersionSet::load_from_path(&version_path);
        }

        let metadata = SegmentMetadata::load_from_path(&self.collection_dir.join("segment.json"))?;
        Ok(VersionSet::single(metadata.segment_id))
    }

    pub fn active_segment_path(&self) -> io::Result<SegmentPaths> {
        let version_set = self.version_set()?;
        if self.version_set_path().exists() {
            Ok(SegmentPaths::from_segment_dir(
                self.segments_dir().join(version_set.active_segment_id()),
                version_set.active_segment_id().to_string(),
            ))
        } else {
            Ok(SegmentPaths::from_collection_dir(
                &self.collection_dir,
                version_set.active_segment_id().to_string(),
            ))
        }
    }

    pub fn immutable_segment_paths(&self) -> io::Result<Vec<SegmentPaths>> {
        let version_set = self.version_set()?;
        if !self.version_set_path().exists() {
            return Ok(Vec::new());
        }

        Ok(version_set
            .immutable_segment_ids()
            .iter()
            .cloned()
            .map(|segment_id| {
                SegmentPaths::from_segment_dir(self.segments_dir().join(&segment_id), segment_id)
            })
            .collect())
    }

    pub fn segment_paths(&self) -> io::Result<Vec<SegmentPaths>> {
        let mut paths = Vec::new();
        paths.push(self.active_segment_path()?);
        paths.extend(self.immutable_segment_paths()?);
        Ok(paths)
    }
}
