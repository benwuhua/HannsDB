use std::io;
use std::path::Path;
use std::{fs, io::ErrorKind};

use serde::{Deserialize, Serialize};

use crate::catalog::COLLECTION_RUNTIME_FORMAT_VERSION;

use super::SegmentSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VersionSet {
    format_version: u32,
    active_segment_id: String,
    immutable_segment_ids: Vec<String>,
}

impl VersionSet {
    pub fn new(active_segment_id: impl Into<String>, immutable_segment_ids: Vec<String>) -> Self {
        Self {
            format_version: COLLECTION_RUNTIME_FORMAT_VERSION,
            active_segment_id: active_segment_id.into(),
            immutable_segment_ids,
        }
    }

    pub fn single(segment_id: impl Into<String>) -> Self {
        Self::new(segment_id, Vec::new())
    }

    pub fn from_segment_set(segment_set: SegmentSet) -> Self {
        Self {
            format_version: COLLECTION_RUNTIME_FORMAT_VERSION,
            active_segment_id: segment_set.active_segment_id,
            immutable_segment_ids: segment_set.immutable_segment_ids,
        }
    }

    pub fn load_from_path(path: &Path) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        if let Ok(versioned) = serde_json::from_slice::<VersionSetFile>(&bytes) {
            if versioned.format_version != COLLECTION_RUNTIME_FORMAT_VERSION {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "unsupported version_set format_version: expected {}, got {}",
                        COLLECTION_RUNTIME_FORMAT_VERSION, versioned.format_version
                    ),
                ));
            }
            return Ok(Self {
                format_version: versioned.format_version,
                active_segment_id: versioned.active_segment_id,
                immutable_segment_ids: versioned.immutable_segment_ids,
            });
        }

        serde_json::from_slice::<SegmentSet>(&bytes)
            .map(Self::from_segment_set)
            .map_err(|err| io::Error::new(ErrorKind::InvalidData, err))
    }

    pub fn save_to_path(&self, path: &Path) -> io::Result<()> {
        let bytes = serde_json::to_vec_pretty(&VersionSetFile {
            format_version: self.format_version,
            active_segment_id: self.active_segment_id.clone(),
            immutable_segment_ids: self.immutable_segment_ids.clone(),
        })
        .map_err(|err| io::Error::new(ErrorKind::InvalidData, err))?;
        atomic_write(path, &bytes)
    }

    pub fn active_segment_id(&self) -> &str {
        &self.active_segment_id
    }

    pub fn immutable_segment_ids(&self) -> &[String] {
        &self.immutable_segment_ids
    }

    pub fn immutable_segment_ids_mut(&mut self) -> &mut Vec<String> {
        &mut self.immutable_segment_ids
    }

    pub fn all_segment_ids(&self) -> Vec<String> {
        let mut ids = Vec::with_capacity(1 + self.immutable_segment_ids.len());
        ids.push(self.active_segment_id.clone());
        ids.extend(self.immutable_segment_ids.iter().cloned());
        ids
    }

    /// Perform a rollover: push the current active segment to immutable,
    /// set a new active segment ID. Returns the new active segment ID.
    pub fn rollover(&mut self) -> String {
        let next = next_segment_id(&self.active_segment_id);
        self.immutable_segment_ids
            .push(self.active_segment_id.clone());
        self.active_segment_id = next.clone();
        next
    }

    pub fn format_version(&self) -> u32 {
        self.format_version
    }

    pub fn into_segment_set(self) -> SegmentSet {
        SegmentSet {
            active_segment_id: self.active_segment_id,
            immutable_segment_ids: self.immutable_segment_ids,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct VersionSetFile {
    format_version: u32,
    active_segment_id: String,
    immutable_segment_ids: Vec<String>,
}

fn next_segment_id(current: &str) -> String {
    let value = current
        .strip_prefix("seg-")
        .and_then(|suffix| suffix.parse::<u64>().ok())
        .unwrap_or(0)
        .saturating_add(1);
    format!("seg-{value:06}")
}

/// Write bytes to a file atomically using write-to-tmp + rename.
/// On POSIX systems, `rename` is atomic, so a crash mid-write leaves
/// the old file intact.
pub fn atomic_write(path: &Path, data: &[u8]) -> io::Result<()> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, data)?;
    fs::rename(&tmp, path)
}
