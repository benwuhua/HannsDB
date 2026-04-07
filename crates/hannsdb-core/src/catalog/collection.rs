use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::document::{
    default_hnsw_ef_construction, default_hnsw_m, CollectionSchema, ScalarFieldSchema,
    VectorFieldSchema, VectorIndexSchema,
};

use super::CATALOG_FORMAT_VERSION;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionMetadata {
    pub format_version: u32,
    pub name: String,
    pub dimension: usize,
    pub metric: String,
    pub primary_vector: String,
    pub fields: Vec<ScalarFieldSchema>,
    pub vectors: Vec<VectorFieldSchema>,
    pub hnsw_m: usize,
    pub hnsw_ef_construction: usize,
}

impl CollectionMetadata {
    pub fn new(name: impl Into<String>, dimension: usize, metric: impl Into<String>) -> Self {
        Self::new_with_schema(
            name,
            CollectionSchema::new(default_primary_vector_name(), dimension, metric, Vec::new()),
        )
    }

    pub fn new_with_schema(name: impl Into<String>, schema: CollectionSchema) -> Self {
        Self::from_schema_parts(
            CATALOG_FORMAT_VERSION,
            name.into(),
            schema.fields,
            schema.vectors,
        )
    }

    pub fn schema(&self) -> CollectionSchema {
        CollectionSchema {
            fields: self.fields.clone(),
            vectors: self.vectors.clone(),
        }
    }

    pub fn save_to_path(&self, path: &Path) -> io::Result<()> {
        let persisted = PersistedCollectionMetadata {
            format_version: self.format_version,
            name: self.name.clone(),
            fields: self.fields.clone(),
            vectors: self.vectors.clone(),
        };
        let bytes = serde_json::to_vec_pretty(&persisted).map_err(json_to_io_error)?;
        fs::write(path, bytes)
    }

    pub fn load_from_path(path: &Path) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        let persisted: PersistedCollectionMetadataCompat =
            serde_json::from_slice(&bytes).map_err(json_to_io_error)?;
        let metadata = match persisted {
            PersistedCollectionMetadataCompat::Current(current) => Self::from_schema_parts(
                current.format_version,
                current.name,
                current.fields,
                current.vectors,
            ),
            PersistedCollectionMetadataCompat::Legacy(legacy) => {
                let mut vectors = Vec::new();
                if legacy.dimension > 0 {
                    vectors.push(
                        VectorFieldSchema::new(legacy.primary_vector.clone(), legacy.dimension)
                            .with_index_param(VectorIndexSchema::hnsw(
                                Some(legacy.metric.as_str()),
                                legacy.hnsw_m,
                                legacy.hnsw_ef_construction,
                            )),
                    );
                }
                Self::from_schema_parts(legacy.format_version, legacy.name, legacy.fields, vectors)
            }
        };
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

    fn from_schema_parts(
        format_version: u32,
        name: String,
        fields: Vec<ScalarFieldSchema>,
        vectors: Vec<VectorFieldSchema>,
    ) -> Self {
        let primary_vector = vectors
            .first()
            .map(|vector| vector.name.clone())
            .unwrap_or_else(default_primary_vector_name);
        let dimension = vectors.first().map_or(0, |vector| vector.dimension);
        let metric = vectors
            .first()
            .and_then(VectorFieldSchema::metric)
            .unwrap_or("l2")
            .to_string();
        let (hnsw_m, hnsw_ef_construction) = vectors
            .first()
            .and_then(VectorFieldSchema::hnsw_settings)
            .unwrap_or((default_hnsw_m(), default_hnsw_ef_construction()));
        Self {
            format_version,
            name,
            dimension,
            metric,
            primary_vector,
            fields,
            vectors,
            hnsw_m,
            hnsw_ef_construction,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct PersistedCollectionMetadata {
    pub format_version: u32,
    pub name: String,
    #[serde(default)]
    pub fields: Vec<ScalarFieldSchema>,
    pub vectors: Vec<VectorFieldSchema>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
enum PersistedCollectionMetadataCompat {
    Current(PersistedCollectionMetadata),
    Legacy(LegacyCollectionMetadata),
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct LegacyCollectionMetadata {
    pub format_version: u32,
    pub name: String,
    pub dimension: usize,
    pub metric: String,
    #[serde(default = "default_primary_vector_name")]
    pub primary_vector: String,
    #[serde(default)]
    pub fields: Vec<ScalarFieldSchema>,
    #[serde(default = "default_hnsw_m")]
    pub hnsw_m: usize,
    #[serde(default = "default_hnsw_ef_construction")]
    pub hnsw_ef_construction: usize,
}

fn json_to_io_error(err: serde_json::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, err)
}

fn default_primary_vector_name() -> String {
    "vector".to_string()
}
