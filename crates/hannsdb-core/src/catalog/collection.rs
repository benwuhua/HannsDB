use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::document::{
    default_hnsw_ef_construction, default_hnsw_m, default_primary_vector_name, CollectionSchema,
    FieldType, ScalarFieldSchema, VectorFieldSchema, VectorIndexSchema,
};
use crate::pk::{default_primary_key_mode, PrimaryKeyMode};
use crate::segment::atomic_write;

use super::CATALOG_FORMAT_VERSION;

#[derive(Debug, Clone, PartialEq)]
pub struct CollectionMetadata {
    pub format_version: u32,
    pub name: String,
    pub primary_key_mode: PrimaryKeyMode,
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
            default_primary_key_mode(),
            schema.primary_vector,
            schema.fields,
            schema.vectors,
        )
    }

    pub fn schema(&self) -> CollectionSchema {
        CollectionSchema {
            primary_vector: self.primary_vector.clone(),
            fields: self.fields.clone(),
            vectors: self.vectors.clone(),
        }
    }

    /// Return true if the primary vector field uses FP16 storage.
    pub fn primary_is_fp16(&self) -> bool {
        self.vectors
            .iter()
            .find(|v| v.name == self.primary_vector)
            .map_or(false, |v| v.data_type == FieldType::VectorFp16)
    }

    pub fn save_to_path(&self, path: &Path) -> io::Result<()> {
        let persisted = PersistedCollectionMetadata {
            format_version: self.format_version,
            name: self.name.clone(),
            primary_key_mode: self.primary_key_mode.clone(),
            primary_vector: Some(self.primary_vector.clone()),
            fields: self.fields.clone(),
            vectors: self.vectors.clone(),
        };
        let bytes = serde_json::to_vec_pretty(&persisted).map_err(json_to_io_error)?;
        atomic_write(path, &bytes)
    }

    pub fn load_from_path(path: &Path) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        let persisted: PersistedCollectionMetadataCompat =
            serde_json::from_slice(&bytes).map_err(json_to_io_error)?;
        let metadata = match persisted {
            PersistedCollectionMetadataCompat::Current(current) => Self::from_schema_parts(
                current.format_version,
                current.name,
                current.primary_key_mode,
                current.primary_vector.unwrap_or_else(|| {
                    current
                        .vectors
                        .first()
                        .map(|vector| vector.name.clone())
                        .unwrap_or_else(default_primary_vector_name)
                }),
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
                Self::from_schema_parts(
                    legacy.format_version,
                    legacy.name,
                    default_primary_key_mode(),
                    legacy.primary_vector,
                    legacy.fields,
                    vectors,
                )
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
        primary_key_mode: PrimaryKeyMode,
        primary_vector: String,
        fields: Vec<ScalarFieldSchema>,
        vectors: Vec<VectorFieldSchema>,
    ) -> Self {
        let primary_vector_schema = vectors.iter().find(|vector| vector.name == primary_vector);
        let dimension = primary_vector_schema.map_or(0, |vector| vector.dimension);
        let metric = primary_vector_schema
            .and_then(VectorFieldSchema::metric)
            .unwrap_or("l2")
            .to_string();
        let (hnsw_m, hnsw_ef_construction) = primary_vector_schema
            .and_then(VectorFieldSchema::hnsw_settings)
            .unwrap_or((default_hnsw_m(), default_hnsw_ef_construction()));
        Self {
            format_version,
            name,
            primary_key_mode,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct PersistedCollectionMetadata {
    pub format_version: u32,
    pub name: String,
    #[serde(default = "default_primary_key_mode")]
    pub primary_key_mode: PrimaryKeyMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primary_vector: Option<String>,
    #[serde(default)]
    pub fields: Vec<ScalarFieldSchema>,
    pub vectors: Vec<VectorFieldSchema>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
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
