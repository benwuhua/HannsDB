use std::fs;
use std::io;
use std::path::Path;

use hannsdb_index::descriptor::{ScalarIndexDescriptor, VectorIndexDescriptor};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct IndexCatalog {
    #[serde(default)]
    pub vector_indexes: Vec<VectorIndexDescriptor>,
    #[serde(default)]
    pub scalar_indexes: Vec<ScalarIndexDescriptor>,
}

impl IndexCatalog {
    pub fn load_from_path(path: &Path) -> io::Result<Self> {
        match fs::read(path) {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(json_to_io_error),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(Self::default()),
            Err(err) => Err(err),
        }
    }

    pub fn save_to_path(&self, path: &Path) -> io::Result<()> {
        let bytes = serde_json::to_vec_pretty(self).map_err(json_to_io_error)?;
        fs::write(path, bytes)
    }

    pub fn upsert_vector_index(&mut self, descriptor: VectorIndexDescriptor) {
        self.vector_indexes
            .retain(|entry| entry.field_name != descriptor.field_name);
        self.vector_indexes.push(descriptor);
    }

    pub fn upsert_scalar_index(&mut self, descriptor: ScalarIndexDescriptor) {
        self.scalar_indexes
            .retain(|entry| entry.field_name != descriptor.field_name);
        self.scalar_indexes.push(descriptor);
    }

    pub fn drop_vector_index(&mut self, field_name: &str) -> bool {
        let before = self.vector_indexes.len();
        self.vector_indexes
            .retain(|descriptor| descriptor.field_name != field_name);
        before != self.vector_indexes.len()
    }

    pub fn drop_scalar_index(&mut self, field_name: &str) -> bool {
        let before = self.scalar_indexes.len();
        self.scalar_indexes
            .retain(|descriptor| descriptor.field_name != field_name);
        before != self.scalar_indexes.len()
    }
}

fn json_to_io_error(err: serde_json::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, err)
}
