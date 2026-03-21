#[derive(Debug, Clone, PartialEq, Eq)]
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
}
