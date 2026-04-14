use std::collections::BTreeMap;

use crate::document::{FieldValue, SparseVector};

#[derive(Debug, Clone, PartialEq)]
pub struct DocumentHit {
    pub id: i64,
    pub distance: f32,
    pub fields: BTreeMap<String, FieldValue>,
    pub vectors: BTreeMap<String, Vec<f32>>,
    pub sparse_vectors: BTreeMap<String, SparseVector>,
    /// Set when group_by is active: the value of the group_by field for this hit.
    pub group_key: Option<FieldValue>,
}

pub(crate) fn compare_hits(left: &DocumentHit, right: &DocumentHit) -> std::cmp::Ordering {
    left.distance
        .total_cmp(&right.distance)
        .then_with(|| left.id.cmp(&right.id))
}

#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit {
    pub id: i64,
    pub distance: f32,
}

pub(crate) fn compare_search_hits(left: &SearchHit, right: &SearchHit) -> std::cmp::Ordering {
    left.distance
        .total_cmp(&right.distance)
        .then_with(|| left.id.cmp(&right.id))
}
