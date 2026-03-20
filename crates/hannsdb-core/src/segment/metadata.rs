use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::SEGMENT_FORMAT_VERSION;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentMetadata {
    pub format_version: u32,
    pub segment_id: String,
    pub dimension: usize,
    pub record_count: usize,
    pub deleted_count: usize,
}

impl SegmentMetadata {
    pub fn new(
        segment_id: impl Into<String>,
        dimension: usize,
        record_count: usize,
        deleted_count: usize,
    ) -> Self {
        Self {
            format_version: SEGMENT_FORMAT_VERSION,
            segment_id: segment_id.into(),
            dimension,
            record_count,
            deleted_count,
        }
    }

    pub fn save_to_path(&self, path: &Path) -> io::Result<()> {
        let bytes = serde_json::to_vec_pretty(self).map_err(json_to_io_error)?;
        fs::write(path, bytes)
    }

    pub fn load_from_path(path: &Path) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        let metadata: Self = serde_json::from_slice(&bytes).map_err(json_to_io_error)?;
        if metadata.format_version != SEGMENT_FORMAT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported segment metadata version: {}",
                    metadata.format_version
                ),
            ));
        }
        Ok(metadata)
    }
}

fn json_to_io_error(err: serde_json::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, err)
}
