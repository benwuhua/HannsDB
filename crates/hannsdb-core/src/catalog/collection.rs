use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::document::{CollectionSchema, ScalarFieldSchema};

use super::CATALOG_FORMAT_VERSION;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectionMetadata {
    pub format_version: u32,
    pub name: String,
    pub dimension: usize,
    pub metric: String,
    #[serde(default = "default_primary_vector_name")]
    pub primary_vector: String,
    #[serde(default)]
    pub fields: Vec<ScalarFieldSchema>,
}

impl CollectionMetadata {
    pub fn new(name: impl Into<String>, dimension: usize, metric: impl Into<String>) -> Self {
        Self {
            format_version: CATALOG_FORMAT_VERSION,
            name: name.into(),
            dimension,
            metric: metric.into(),
            primary_vector: default_primary_vector_name(),
            fields: Vec::new(),
        }
    }

    pub fn new_with_schema(name: impl Into<String>, schema: CollectionSchema) -> Self {
        Self {
            format_version: CATALOG_FORMAT_VERSION,
            name: name.into(),
            dimension: schema.dimension,
            metric: schema.metric,
            primary_vector: schema.primary_vector,
            fields: schema.fields,
        }
    }

    pub fn schema(&self) -> CollectionSchema {
        CollectionSchema {
            primary_vector: self.primary_vector.clone(),
            dimension: self.dimension,
            metric: self.metric.clone(),
            fields: self.fields.clone(),
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
                    "unsupported collection metadata version: {}",
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

fn default_primary_vector_name() -> String {
    "vector".to_string()
}
