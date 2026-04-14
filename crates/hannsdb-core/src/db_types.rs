use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub struct CollectionInfo {
    pub name: String,
    pub dimension: usize,
    pub metric: String,
    pub record_count: usize,
    pub deleted_count: usize,
    pub live_count: usize,
    /// For each vector field, the fraction of live data covered by an ANN index (0.0..=1.0).
    /// A value of 1.0 means the field is fully indexed; 0.0 means no index exists.
    pub index_completeness: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionSegmentInfo {
    pub id: String,
    pub live_count: usize,
    pub dead_count: usize,
    pub ann_ready: bool,
}
