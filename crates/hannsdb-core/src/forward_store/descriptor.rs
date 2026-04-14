use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::schema::ForwardStoreSchema;

pub const FORWARD_STORE_DESCRIPTOR_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForwardFileFormat {
    ArrowIpc,
    Parquet,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForwardFileArtifact {
    pub format: ForwardFileFormat,
    pub data_path: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub row_count: Option<usize>,
}

impl ForwardFileArtifact {
    pub fn new(
        format: ForwardFileFormat,
        data_path: impl Into<PathBuf>,
        row_count: Option<usize>,
    ) -> Self {
        Self {
            format,
            data_path: data_path.into(),
            row_count,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForwardStoreDescriptor {
    #[serde(default = "default_descriptor_version")]
    pub version: u32,
    pub schema: ForwardStoreSchema,
    pub row_count: usize,
    pub artifacts: Vec<ForwardFileArtifact>,
}

impl ForwardStoreDescriptor {
    pub fn new(
        schema: ForwardStoreSchema,
        row_count: usize,
        artifacts: Vec<ForwardFileArtifact>,
    ) -> Self {
        Self {
            version: FORWARD_STORE_DESCRIPTOR_VERSION,
            schema,
            row_count,
            artifacts,
        }
    }

    pub fn artifact(&self, format: ForwardFileFormat) -> Option<&ForwardFileArtifact> {
        self.artifacts.iter().find(|artifact| artifact.format == format)
    }
}

fn default_descriptor_version() -> u32 {
    FORWARD_STORE_DESCRIPTOR_VERSION
}

#[cfg(test)]
mod tests {
    use crate::document::{CollectionSchema, FieldType, ScalarFieldSchema};

    use super::*;

    #[test]
    fn forward_store_descriptor_tracks_explicit_artifacts_by_format() {
        let schema = ForwardStoreSchema::from_collection_schema(&CollectionSchema::new(
            "embedding",
            3,
            "l2",
            vec![ScalarFieldSchema::new("title", FieldType::String)],
        ))
        .expect("schema");

        let descriptor = ForwardStoreDescriptor::new(
            schema,
            4,
            vec![
                ForwardFileArtifact::new(
                    ForwardFileFormat::ArrowIpc,
                    "segment/forward.arrow",
                    Some(4),
                ),
                ForwardFileArtifact::new(
                    ForwardFileFormat::Parquet,
                    "segment/forward.parquet",
                    Some(4),
                ),
            ],
        );

        assert_eq!(
            descriptor
                .artifact(ForwardFileFormat::ArrowIpc)
                .expect("arrow artifact")
                .data_path,
            PathBuf::from("segment/forward.arrow")
        );
        assert_eq!(
            descriptor
                .artifact(ForwardFileFormat::Parquet)
                .expect("parquet artifact")
                .row_count,
            Some(4)
        );
    }
}
