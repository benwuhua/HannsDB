use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::segment::atomic_write;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PrimaryKeyMode {
    #[default]
    Numeric,
    String,
}

pub fn default_primary_key_mode() -> PrimaryKeyMode {
    PrimaryKeyMode::Numeric
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrimaryKeyRegistry {
    #[serde(default = "default_primary_key_mode")]
    pub mode: PrimaryKeyMode,
    #[serde(default = "default_next_internal_id")]
    pub next_internal_id: i64,
    #[serde(default)]
    pub key_to_id: BTreeMap<String, i64>,
    #[serde(default)]
    pub id_to_key: BTreeMap<i64, String>,
}

impl PrimaryKeyRegistry {
    pub fn new(mode: PrimaryKeyMode, next_internal_id: i64) -> Self {
        Self {
            mode,
            next_internal_id: next_internal_id.max(1),
            key_to_id: BTreeMap::new(),
            id_to_key: BTreeMap::new(),
        }
    }

    pub fn load_from_path(path: &Path) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        let registry = serde_json::from_slice(&bytes).map_err(json_to_io_error)?;
        Ok(registry)
    }

    pub fn save_to_path(&self, path: &Path) -> io::Result<()> {
        let bytes = serde_json::to_vec_pretty(self).map_err(json_to_io_error)?;
        atomic_write(path, &bytes)
    }
}

fn default_next_internal_id() -> i64 {
    1
}

fn json_to_io_error(err: serde_json::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, err)
}
