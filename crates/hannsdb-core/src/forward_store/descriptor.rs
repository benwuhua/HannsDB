use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::schema::ForwardSchema;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ForwardFileFormat {
    ArrowIpc,
    Parquet,
}

impl ForwardFileFormat {
    pub fn extension(self) -> &'static str {
        match self {
            Self::ArrowIpc => "arrow",
            Self::Parquet => "parquet",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::ArrowIpc => "arrow_ipc",
            Self::Parquet => "parquet",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForwardStoreArtifact {
    pub format: ForwardFileFormat,
    pub path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForwardStoreDescriptor {
    pub schema: ForwardSchema,
    pub row_count: usize,
    pub artifacts: Vec<ForwardStoreArtifact>,
}

impl ForwardStoreDescriptor {
    pub fn artifact(&self, format: ForwardFileFormat) -> Option<&ForwardStoreArtifact> {
        self.artifacts.iter().find(|artifact| artifact.format == format)
    }
}
