use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

pub const ROLLOVER_MAX_ROWS: u64 = 200_000;
pub const ROLLOVER_MAX_TOMBSTONE_RATIO: f64 = 0.20;
pub const COMPACTION_THRESHOLD: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentSet {
    pub active_segment_id: String,
    pub immutable_segment_ids: Vec<String>,
}

impl SegmentSet {
    pub fn new_single(id: &str) -> Self {
        Self {
            active_segment_id: id.to_string(),
            immutable_segment_ids: Vec::new(),
        }
    }

    pub fn should_rollover(record_count: u64, tombstone_count: u64) -> bool {
        record_count >= ROLLOVER_MAX_ROWS
            || tombstone_ratio(record_count, tombstone_count) >= ROLLOVER_MAX_TOMBSTONE_RATIO
    }

    pub fn should_compact(immutable_count: usize) -> bool {
        immutable_count >= COMPACTION_THRESHOLD
    }

    pub fn rollover(&mut self) {
        let next = next_segment_id(&self.active_segment_id);
        self.immutable_segment_ids
            .push(self.active_segment_id.clone());
        self.active_segment_id = next;
    }

    pub fn save_to_path(&self, path: &Path) -> io::Result<()> {
        let bytes = serde_json::to_vec_pretty(self).map_err(json_to_io_error)?;
        fs::write(path, bytes)
    }

    pub fn load_from_path(path: &Path) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        serde_json::from_slice(&bytes).map_err(json_to_io_error)
    }
}

fn json_to_io_error(err: serde_json::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, err)
}

fn tombstone_ratio(record_count: u64, tombstone_count: u64) -> f64 {
    if record_count == 0 {
        return 0.0;
    }
    tombstone_count as f64 / record_count as f64
}

fn next_segment_id(current: &str) -> String {
    let value = current
        .strip_prefix("seg-")
        .and_then(|suffix| suffix.parse::<u64>().ok())
        .unwrap_or(0)
        .saturating_add(1);
    format!("seg-{value:06}")
}

#[cfg(test)]
mod tests {
    use super::SegmentSet;

    #[test]
    fn segment_set_should_rollover_false_below_threshold() {
        assert!(!SegmentSet::should_rollover(199_999, 10_000));
    }

    #[test]
    fn segment_set_should_rollover_true_on_record_threshold() {
        assert!(SegmentSet::should_rollover(200_000, 0));
    }

    #[test]
    fn segment_set_should_rollover_true_on_tombstone_ratio_threshold() {
        assert!(SegmentSet::should_rollover(100, 20));
    }
}
