use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::CATALOG_FORMAT_VERSION;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestMetadata {
    pub format_version: u32,
    pub database_id: String,
    pub collections: Vec<String>,
}

impl ManifestMetadata {
    pub fn new(database_id: impl Into<String>, collections: Vec<String>) -> Self {
        Self {
            format_version: CATALOG_FORMAT_VERSION,
            database_id: database_id.into(),
            collections,
        }
    }

    pub fn save_to_path(&self, path: &Path) -> io::Result<()> {
        let bytes = serde_json::to_vec_pretty(self).map_err(json_to_io_error)?;
        fs::write(path, bytes)
    }

    pub fn load_from_path(path: &Path) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        let metadata: Self = serde_json::from_slice(&bytes).map_err(json_to_io_error)?;
        if metadata.format_version != CATALOG_FORMAT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported manifest metadata version: {}",
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
