use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::SEGMENT_FORMAT_VERSION;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TombstoneMask {
    pub format_version: u32,
    bits: Vec<bool>,
}

impl TombstoneMask {
    pub fn new(length: usize) -> Self {
        Self {
            format_version: SEGMENT_FORMAT_VERSION,
            bits: vec![false; length],
        }
    }

    pub fn mark_deleted(&mut self, index: usize) -> bool {
        if let Some(bit) = self.bits.get_mut(index) {
            if *bit {
                return false;
            }
            *bit = true;
            return true;
        }
        false
    }

    pub fn is_deleted(&self, index: usize) -> bool {
        self.bits.get(index).copied().unwrap_or(false)
    }

    pub fn len(&self) -> usize {
        self.bits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    pub fn extend(&mut self, additional: usize) {
        self.bits.extend(std::iter::repeat(false).take(additional));
    }

    pub fn deleted_count(&self) -> usize {
        self.bits.iter().filter(|deleted| **deleted).count()
    }

    pub fn visible_indices(&self) -> Vec<usize> {
        self.bits
            .iter()
            .enumerate()
            .filter_map(|(idx, deleted)| if !deleted { Some(idx) } else { None })
            .collect()
    }

    pub fn save_to_path(&self, path: &Path) -> io::Result<()> {
        let bytes = serde_json::to_vec_pretty(self).map_err(json_to_io_error)?;
        fs::write(path, bytes)
    }

    pub fn load_from_path(path: &Path) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        let mask: Self = serde_json::from_slice(&bytes).map_err(json_to_io_error)?;
        if mask.format_version != SEGMENT_FORMAT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported tombstone mask version: {}",
                    mask.format_version
                ),
            ));
        }
        Ok(mask)
    }
}

fn json_to_io_error(err: serde_json::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, err)
}
