use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::Bound;

use crate::descriptor::ScalarIndexDescriptor;

// ---------------------------------------------------------------------------
// Value type (local mirror of hannsdb_core::document::FieldValue)
// ---------------------------------------------------------------------------

/// Scalar field value used by the inverted index.
///
/// This is a local duplicate of `hannsdb_core::document::FieldValue` because
/// `hannsdb-index` cannot depend on `hannsdb-core`. The two types are kept
/// structurally identical so that conversion at the integration boundary is
/// trivial.
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarValue {
    String(String),
    Int64(i64),
    Int32(i32),
    UInt32(u32),
    UInt64(u64),
    Float(f32),
    Float64(f64),
    Bool(bool),
}

// ---------------------------------------------------------------------------
// Range operator (local mirror of hannsdb_core::query::ComparisonOp)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeOp {
    Eq,
    Ne,
    Gt,
    Gte,
    Lt,
    Lte,
}

// ---------------------------------------------------------------------------
// OrderedF64 — total-order wrapper for f64 so it can be used in BTreeMap
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrderedF64(pub f64);

impl Eq for OrderedF64 {}

impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// InvertedScalarIndex
// ---------------------------------------------------------------------------

/// OrderedF32 -- total-order wrapper for f32 so it can be used in BTreeMap
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrderedF32(pub f32);

impl Eq for OrderedF32 {}

impl PartialOrd for OrderedF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum InvertedScalarIndex {
    String {
        descriptor: ScalarIndexDescriptor,
        map: HashMap<String, BTreeSet<i64>>,
    },
    Int64 {
        descriptor: ScalarIndexDescriptor,
        entries: BTreeMap<i64, BTreeSet<i64>>,
    },
    Int32 {
        descriptor: ScalarIndexDescriptor,
        entries: BTreeMap<i32, BTreeSet<i64>>,
    },
    UInt32 {
        descriptor: ScalarIndexDescriptor,
        entries: BTreeMap<u32, BTreeSet<i64>>,
    },
    UInt64 {
        descriptor: ScalarIndexDescriptor,
        entries: BTreeMap<u64, BTreeSet<i64>>,
    },
    Float {
        descriptor: ScalarIndexDescriptor,
        entries: BTreeMap<OrderedF32, BTreeSet<i64>>,
    },
    Float64 {
        descriptor: ScalarIndexDescriptor,
        entries: BTreeMap<OrderedF64, BTreeSet<i64>>,
    },
    Bool {
        descriptor: ScalarIndexDescriptor,
        true_ids: BTreeSet<i64>,
        false_ids: BTreeSet<i64>,
    },
    Empty {
        descriptor: ScalarIndexDescriptor,
    },
}

impl InvertedScalarIndex {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    pub fn new(descriptor: ScalarIndexDescriptor) -> Self {
        Self::Empty { descriptor }
    }

    /// Build an index from payload data.
    ///
    /// * `descriptor` — index descriptor (field name + kind).
    /// * `field_name` — the field to index.
    /// * `payloads`   — one `BTreeMap<String, ScalarValue>` per row.
    /// * `external_ids` — external ID for each row (same length as `payloads`).
    pub fn build_from_payloads(
        descriptor: ScalarIndexDescriptor,
        field_name: &str,
        payloads: &[BTreeMap<String, ScalarValue>],
        external_ids: &[i64],
    ) -> Self {
        assert_eq!(
            payloads.len(),
            external_ids.len(),
            "payloads and external_ids must have the same length"
        );

        // Determine the type from the first non-empty value found.
        let first_value = payloads.iter().find_map(|payload| payload.get(field_name));

        let field_type = match first_value {
            Some(ScalarValue::String(_)) => "string",
            Some(ScalarValue::Int64(_)) => "int64",
            Some(ScalarValue::Int32(_)) => "int32",
            Some(ScalarValue::UInt32(_)) => "uint32",
            Some(ScalarValue::UInt64(_)) => "uint64",
            Some(ScalarValue::Float(_)) => "float",
            Some(ScalarValue::Float64(_)) => "float64",
            Some(ScalarValue::Bool(_)) => "bool",
            None => return Self::Empty { descriptor },
        };

        match field_type {
            "string" => {
                let mut map: HashMap<String, BTreeSet<i64>> = HashMap::new();
                for (payload, &ext_id) in payloads.iter().zip(external_ids.iter()) {
                    if let Some(ScalarValue::String(s)) = payload.get(field_name) {
                        map.entry(s.clone()).or_default().insert(ext_id);
                    }
                }
                Self::String { descriptor, map }
            }
            "int64" => {
                let mut entries: BTreeMap<i64, BTreeSet<i64>> = BTreeMap::new();
                for (payload, &ext_id) in payloads.iter().zip(external_ids.iter()) {
                    if let Some(ScalarValue::Int64(v)) = payload.get(field_name) {
                        entries.entry(*v).or_default().insert(ext_id);
                    }
                }
                Self::Int64 {
                    descriptor,
                    entries,
                }
            }
            "int32" => {
                let mut entries: BTreeMap<i32, BTreeSet<i64>> = BTreeMap::new();
                for (payload, &ext_id) in payloads.iter().zip(external_ids.iter()) {
                    if let Some(ScalarValue::Int32(v)) = payload.get(field_name) {
                        entries.entry(*v).or_default().insert(ext_id);
                    }
                }
                Self::Int32 {
                    descriptor,
                    entries,
                }
            }
            "uint32" => {
                let mut entries: BTreeMap<u32, BTreeSet<i64>> = BTreeMap::new();
                for (payload, &ext_id) in payloads.iter().zip(external_ids.iter()) {
                    if let Some(ScalarValue::UInt32(v)) = payload.get(field_name) {
                        entries.entry(*v).or_default().insert(ext_id);
                    }
                }
                Self::UInt32 {
                    descriptor,
                    entries,
                }
            }
            "uint64" => {
                let mut entries: BTreeMap<u64, BTreeSet<i64>> = BTreeMap::new();
                for (payload, &ext_id) in payloads.iter().zip(external_ids.iter()) {
                    if let Some(ScalarValue::UInt64(v)) = payload.get(field_name) {
                        entries.entry(*v).or_default().insert(ext_id);
                    }
                }
                Self::UInt64 {
                    descriptor,
                    entries,
                }
            }
            "float" => {
                let mut entries: BTreeMap<OrderedF32, BTreeSet<i64>> = BTreeMap::new();
                for (payload, &ext_id) in payloads.iter().zip(external_ids.iter()) {
                    if let Some(ScalarValue::Float(v)) = payload.get(field_name) {
                        entries.entry(OrderedF32(*v)).or_default().insert(ext_id);
                    }
                }
                Self::Float {
                    descriptor,
                    entries,
                }
            }
            "float64" => {
                let mut entries: BTreeMap<OrderedF64, BTreeSet<i64>> = BTreeMap::new();
                for (payload, &ext_id) in payloads.iter().zip(external_ids.iter()) {
                    if let Some(ScalarValue::Float64(v)) = payload.get(field_name) {
                        entries.entry(OrderedF64(*v)).or_default().insert(ext_id);
                    }
                }
                Self::Float64 {
                    descriptor,
                    entries,
                }
            }
            "bool" => {
                let mut true_ids = BTreeSet::new();
                let mut false_ids = BTreeSet::new();
                for (payload, &ext_id) in payloads.iter().zip(external_ids.iter()) {
                    if let Some(ScalarValue::Bool(b)) = payload.get(field_name) {
                        if *b {
                            true_ids.insert(ext_id);
                        } else {
                            false_ids.insert(ext_id);
                        }
                    }
                }
                Self::Bool {
                    descriptor,
                    true_ids,
                    false_ids,
                }
            }
            _ => Self::Empty { descriptor },
        }
    }

    // -----------------------------------------------------------------------
    // Query API
    // -----------------------------------------------------------------------

    /// Lookup all external IDs where the indexed field equals the given value.
    pub fn lookup_eq(&self, value: &ScalarValue) -> BTreeSet<i64> {
        match self {
            InvertedScalarIndex::String { map, .. } => {
                if let ScalarValue::String(s) = value {
                    map.get(s).cloned().unwrap_or_default()
                } else {
                    BTreeSet::new()
                }
            }
            InvertedScalarIndex::Int64 { entries, .. } => {
                if let ScalarValue::Int64(v) = value {
                    entries.get(v).cloned().unwrap_or_default()
                } else {
                    BTreeSet::new()
                }
            }
            InvertedScalarIndex::Int32 { entries, .. } => {
                if let ScalarValue::Int32(v) = value {
                    entries.get(v).cloned().unwrap_or_default()
                } else {
                    BTreeSet::new()
                }
            }
            InvertedScalarIndex::UInt32 { entries, .. } => {
                if let ScalarValue::UInt32(v) = value {
                    entries.get(v).cloned().unwrap_or_default()
                } else {
                    BTreeSet::new()
                }
            }
            InvertedScalarIndex::UInt64 { entries, .. } => {
                if let ScalarValue::UInt64(v) = value {
                    entries.get(v).cloned().unwrap_or_default()
                } else {
                    BTreeSet::new()
                }
            }
            InvertedScalarIndex::Float { entries, .. } => {
                if let ScalarValue::Float(v) = value {
                    entries.get(&OrderedF32(*v)).cloned().unwrap_or_default()
                } else {
                    BTreeSet::new()
                }
            }
            InvertedScalarIndex::Float64 { entries, .. } => {
                if let ScalarValue::Float64(v) = value {
                    entries.get(&OrderedF64(*v)).cloned().unwrap_or_default()
                } else {
                    BTreeSet::new()
                }
            }
            InvertedScalarIndex::Bool {
                true_ids,
                false_ids,
                ..
            } => match value {
                ScalarValue::Bool(true) => true_ids.clone(),
                ScalarValue::Bool(false) => false_ids.clone(),
                _ => BTreeSet::new(),
            },
            InvertedScalarIndex::Empty { .. } => BTreeSet::new(),
        }
    }

    /// Lookup all external IDs satisfying a range comparison.
    ///
    /// For `Ne`, returns `all_indexed_ids() - lookup_eq(value)`.
    pub fn lookup_range(&self, op: RangeOp, value: &ScalarValue) -> BTreeSet<i64> {
        if op == RangeOp::Eq {
            return self.lookup_eq(value);
        }
        if op == RangeOp::Ne {
            let eq = self.lookup_eq(value);
            return self.all_indexed_ids().difference(&eq).cloned().collect();
        }

        match self {
            InvertedScalarIndex::Int64 { entries, .. } => {
                if let ScalarValue::Int64(v) = value {
                    let range = match op {
                        RangeOp::Gt => entries.range((Bound::Excluded(*v), Bound::Unbounded)),
                        RangeOp::Gte => entries.range((Bound::Included(*v), Bound::Unbounded)),
                        RangeOp::Lt => entries.range((Bound::Unbounded, Bound::Excluded(*v))),
                        RangeOp::Lte => entries.range((Bound::Unbounded, Bound::Included(*v))),
                        _ => unreachable!("Eq and Ne handled above"),
                    };
                    let mut result = BTreeSet::new();
                    for (_, ids) in range {
                        result.extend(ids.iter().cloned());
                    }
                    result
                } else {
                    BTreeSet::new()
                }
            }
            InvertedScalarIndex::Int32 { entries, .. } => {
                if let ScalarValue::Int32(v) = value {
                    let range = match op {
                        RangeOp::Gt => entries.range((Bound::Excluded(*v), Bound::Unbounded)),
                        RangeOp::Gte => entries.range((Bound::Included(*v), Bound::Unbounded)),
                        RangeOp::Lt => entries.range((Bound::Unbounded, Bound::Excluded(*v))),
                        RangeOp::Lte => entries.range((Bound::Unbounded, Bound::Included(*v))),
                        _ => unreachable!("Eq and Ne handled above"),
                    };
                    let mut result = BTreeSet::new();
                    for (_, ids) in range {
                        result.extend(ids.iter().cloned());
                    }
                    result
                } else {
                    BTreeSet::new()
                }
            }
            InvertedScalarIndex::UInt32 { entries, .. } => {
                if let ScalarValue::UInt32(v) = value {
                    let range = match op {
                        RangeOp::Gt => entries.range((Bound::Excluded(*v), Bound::Unbounded)),
                        RangeOp::Gte => entries.range((Bound::Included(*v), Bound::Unbounded)),
                        RangeOp::Lt => entries.range((Bound::Unbounded, Bound::Excluded(*v))),
                        RangeOp::Lte => entries.range((Bound::Unbounded, Bound::Included(*v))),
                        _ => unreachable!("Eq and Ne handled above"),
                    };
                    let mut result = BTreeSet::new();
                    for (_, ids) in range {
                        result.extend(ids.iter().cloned());
                    }
                    result
                } else {
                    BTreeSet::new()
                }
            }
            InvertedScalarIndex::UInt64 { entries, .. } => {
                if let ScalarValue::UInt64(v) = value {
                    let range = match op {
                        RangeOp::Gt => entries.range((Bound::Excluded(*v), Bound::Unbounded)),
                        RangeOp::Gte => entries.range((Bound::Included(*v), Bound::Unbounded)),
                        RangeOp::Lt => entries.range((Bound::Unbounded, Bound::Excluded(*v))),
                        RangeOp::Lte => entries.range((Bound::Unbounded, Bound::Included(*v))),
                        _ => unreachable!("Eq and Ne handled above"),
                    };
                    let mut result = BTreeSet::new();
                    for (_, ids) in range {
                        result.extend(ids.iter().cloned());
                    }
                    result
                } else {
                    BTreeSet::new()
                }
            }
            InvertedScalarIndex::Float { entries, .. } => {
                if let ScalarValue::Float(v) = value {
                    let key = OrderedF32(*v);
                    let range = match op {
                        RangeOp::Gt => entries.range((Bound::Excluded(key), Bound::Unbounded)),
                        RangeOp::Gte => entries.range((Bound::Included(key), Bound::Unbounded)),
                        RangeOp::Lt => entries.range((Bound::Unbounded, Bound::Excluded(key))),
                        RangeOp::Lte => entries.range((Bound::Unbounded, Bound::Included(key))),
                        _ => unreachable!("Eq and Ne handled above"),
                    };
                    let mut result = BTreeSet::new();
                    for (_, ids) in range {
                        result.extend(ids.iter().cloned());
                    }
                    result
                } else {
                    BTreeSet::new()
                }
            }
            InvertedScalarIndex::Float64 { entries, .. } => {
                if let ScalarValue::Float64(v) = value {
                    let key = OrderedF64(*v);
                    let range = match op {
                        RangeOp::Gt => entries.range((Bound::Excluded(key), Bound::Unbounded)),
                        RangeOp::Gte => entries.range((Bound::Included(key), Bound::Unbounded)),
                        RangeOp::Lt => entries.range((Bound::Unbounded, Bound::Excluded(key))),
                        RangeOp::Lte => entries.range((Bound::Unbounded, Bound::Included(key))),
                        _ => unreachable!("Eq and Ne handled above"),
                    };
                    let mut result = BTreeSet::new();
                    for (_, ids) in range {
                        result.extend(ids.iter().cloned());
                    }
                    result
                } else {
                    BTreeSet::new()
                }
            }
            // String and Bool do not support range queries beyond Eq/Ne.
            _ => BTreeSet::new(),
        }
    }

    /// All external IDs present in the index.
    pub fn all_indexed_ids(&self) -> BTreeSet<i64> {
        match self {
            InvertedScalarIndex::String { map, .. } => {
                let mut ids = BTreeSet::new();
                for set in map.values() {
                    ids.extend(set.iter().cloned());
                }
                ids
            }
            InvertedScalarIndex::Int64 { entries, .. } => {
                let mut ids = BTreeSet::new();
                for set in entries.values() {
                    ids.extend(set.iter().cloned());
                }
                ids
            }
            InvertedScalarIndex::Int32 { entries, .. } => {
                let mut ids = BTreeSet::new();
                for set in entries.values() {
                    ids.extend(set.iter().cloned());
                }
                ids
            }
            InvertedScalarIndex::UInt32 { entries, .. } => {
                let mut ids = BTreeSet::new();
                for set in entries.values() {
                    ids.extend(set.iter().cloned());
                }
                ids
            }
            InvertedScalarIndex::UInt64 { entries, .. } => {
                let mut ids = BTreeSet::new();
                for set in entries.values() {
                    ids.extend(set.iter().cloned());
                }
                ids
            }
            InvertedScalarIndex::Float { entries, .. } => {
                let mut ids = BTreeSet::new();
                for set in entries.values() {
                    ids.extend(set.iter().cloned());
                }
                ids
            }
            InvertedScalarIndex::Float64 { entries, .. } => {
                let mut ids = BTreeSet::new();
                for set in entries.values() {
                    ids.extend(set.iter().cloned());
                }
                ids
            }
            InvertedScalarIndex::Bool {
                true_ids,
                false_ids,
                ..
            } => {
                let mut ids = true_ids.clone();
                ids.extend(false_ids.iter().cloned());
                ids
            }
            InvertedScalarIndex::Empty { .. } => BTreeSet::new(),
        }
    }

    /// Lookup IDs matching an in-list.
    ///
    /// When `negated` is true, returns `all_indexed_ids() - union(matches)`.
    pub fn lookup_in(&self, values: &[ScalarValue], negated: bool) -> BTreeSet<i64> {
        let mut result = BTreeSet::new();
        for value in values {
            result.extend(self.lookup_eq(value));
        }
        if negated {
            let all = self.all_indexed_ids();
            all.difference(&result).cloned().collect()
        } else {
            result
        }
    }

    /// Return the field name stored in the descriptor.
    pub fn field_name(&self) -> &str {
        match self {
            InvertedScalarIndex::String { descriptor, .. } => &descriptor.field_name,
            InvertedScalarIndex::Int64 { descriptor, .. } => &descriptor.field_name,
            InvertedScalarIndex::Int32 { descriptor, .. } => &descriptor.field_name,
            InvertedScalarIndex::UInt32 { descriptor, .. } => &descriptor.field_name,
            InvertedScalarIndex::UInt64 { descriptor, .. } => &descriptor.field_name,
            InvertedScalarIndex::Float { descriptor, .. } => &descriptor.field_name,
            InvertedScalarIndex::Float64 { descriptor, .. } => &descriptor.field_name,
            InvertedScalarIndex::Bool { descriptor, .. } => &descriptor.field_name,
            InvertedScalarIndex::Empty { descriptor } => &descriptor.field_name,
        }
    }

    /// Return the descriptor.
    pub fn descriptor(&self) -> &ScalarIndexDescriptor {
        match self {
            InvertedScalarIndex::String { descriptor, .. } => descriptor,
            InvertedScalarIndex::Int64 { descriptor, .. } => descriptor,
            InvertedScalarIndex::Int32 { descriptor, .. } => descriptor,
            InvertedScalarIndex::UInt32 { descriptor, .. } => descriptor,
            InvertedScalarIndex::UInt64 { descriptor, .. } => descriptor,
            InvertedScalarIndex::Float { descriptor, .. } => descriptor,
            InvertedScalarIndex::Float64 { descriptor, .. } => descriptor,
            InvertedScalarIndex::Bool { descriptor, .. } => descriptor,
            InvertedScalarIndex::Empty { descriptor } => descriptor,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_descriptor(field: &str) -> ScalarIndexDescriptor {
        ScalarIndexDescriptor {
            field_name: field.to_string(),
            kind: crate::descriptor::ScalarIndexKind::Inverted,
            params: serde_json::Value::Null,
        }
    }

    // -- String index tests --

    #[test]
    fn string_index_lookup_eq() {
        let descriptor = test_descriptor("name");
        let payloads = vec![
            BTreeMap::from([("name".into(), ScalarValue::String("alice".into()))]),
            BTreeMap::from([("name".into(), ScalarValue::String("bob".into()))]),
            BTreeMap::from([("name".into(), ScalarValue::String("alice".into()))]),
        ];
        let ids = vec![1, 2, 3];

        let index = InvertedScalarIndex::build_from_payloads(descriptor, "name", &payloads, &ids);

        let result = index.lookup_eq(&ScalarValue::String("alice".into()));
        assert_eq!(result, BTreeSet::from([1, 3]));

        let result = index.lookup_eq(&ScalarValue::String("bob".into()));
        assert_eq!(result, BTreeSet::from([2]));

        let result = index.lookup_eq(&ScalarValue::String("charlie".into()));
        assert!(result.is_empty());
    }

    #[test]
    fn string_index_all_ids() {
        let descriptor = test_descriptor("name");
        let payloads = vec![
            BTreeMap::from([("name".into(), ScalarValue::String("alice".into()))]),
            BTreeMap::from([("name".into(), ScalarValue::String("bob".into()))]),
        ];
        let ids = vec![10, 20];

        let index = InvertedScalarIndex::build_from_payloads(descriptor, "name", &payloads, &ids);
        assert_eq!(index.all_indexed_ids(), BTreeSet::from([10, 20]));
    }

    #[test]
    fn string_index_lookup_in() {
        let descriptor = test_descriptor("name");
        let payloads = vec![
            BTreeMap::from([("name".into(), ScalarValue::String("alice".into()))]),
            BTreeMap::from([("name".into(), ScalarValue::String("bob".into()))]),
            BTreeMap::from([("name".into(), ScalarValue::String("carol".into()))]),
        ];
        let ids = vec![1, 2, 3];

        let index = InvertedScalarIndex::build_from_payloads(descriptor, "name", &payloads, &ids);

        let result = index.lookup_in(
            &[
                ScalarValue::String("alice".into()),
                ScalarValue::String("bob".into()),
            ],
            false,
        );
        assert_eq!(result, BTreeSet::from([1, 2]));

        // Negated: everything except alice and bob
        let result = index.lookup_in(
            &[
                ScalarValue::String("alice".into()),
                ScalarValue::String("bob".into()),
            ],
            true,
        );
        assert_eq!(result, BTreeSet::from([3]));
    }

    // -- Int64 index tests --

    #[test]
    fn int64_index_lookup_eq() {
        let descriptor = test_descriptor("age");
        let payloads = vec![
            BTreeMap::from([("age".into(), ScalarValue::Int64(25))]),
            BTreeMap::from([("age".into(), ScalarValue::Int64(30))]),
            BTreeMap::from([("age".into(), ScalarValue::Int64(25))]),
        ];
        let ids = vec![1, 2, 3];

        let index = InvertedScalarIndex::build_from_payloads(descriptor, "age", &payloads, &ids);

        let result = index.lookup_eq(&ScalarValue::Int64(25));
        assert_eq!(result, BTreeSet::from([1, 3]));

        let result = index.lookup_eq(&ScalarValue::Int64(30));
        assert_eq!(result, BTreeSet::from([2]));
    }

    #[test]
    fn int64_index_range_gt() {
        let descriptor = test_descriptor("age");
        let payloads = vec![
            BTreeMap::from([("age".into(), ScalarValue::Int64(10))]),
            BTreeMap::from([("age".into(), ScalarValue::Int64(20))]),
            BTreeMap::from([("age".into(), ScalarValue::Int64(30))]),
            BTreeMap::from([("age".into(), ScalarValue::Int64(40))]),
        ];
        let ids = vec![1, 2, 3, 4];

        let index = InvertedScalarIndex::build_from_payloads(descriptor, "age", &payloads, &ids);

        let result = index.lookup_range(RangeOp::Gt, &ScalarValue::Int64(20));
        assert_eq!(result, BTreeSet::from([3, 4]));

        let result = index.lookup_range(RangeOp::Gte, &ScalarValue::Int64(20));
        assert_eq!(result, BTreeSet::from([2, 3, 4]));

        let result = index.lookup_range(RangeOp::Lt, &ScalarValue::Int64(30));
        assert_eq!(result, BTreeSet::from([1, 2]));

        let result = index.lookup_range(RangeOp::Lte, &ScalarValue::Int64(30));
        assert_eq!(result, BTreeSet::from([1, 2, 3]));
    }

    #[test]
    fn int64_index_range_ne() {
        let descriptor = test_descriptor("age");
        let payloads = vec![
            BTreeMap::from([("age".into(), ScalarValue::Int64(10))]),
            BTreeMap::from([("age".into(), ScalarValue::Int64(20))]),
            BTreeMap::from([("age".into(), ScalarValue::Int64(20))]),
        ];
        let ids = vec![1, 2, 3];

        let index = InvertedScalarIndex::build_from_payloads(descriptor, "age", &payloads, &ids);

        let result = index.lookup_range(RangeOp::Ne, &ScalarValue::Int64(20));
        assert_eq!(result, BTreeSet::from([1]));
    }

    #[test]
    fn int64_index_lookup_in() {
        let descriptor = test_descriptor("group");
        let payloads = vec![
            BTreeMap::from([("group".into(), ScalarValue::Int64(1))]),
            BTreeMap::from([("group".into(), ScalarValue::Int64(2))]),
            BTreeMap::from([("group".into(), ScalarValue::Int64(3))]),
            BTreeMap::from([("group".into(), ScalarValue::Int64(4))]),
        ];
        let ids = vec![10, 20, 30, 40];

        let index = InvertedScalarIndex::build_from_payloads(descriptor, "group", &payloads, &ids);

        let result = index.lookup_in(&[ScalarValue::Int64(1), ScalarValue::Int64(3)], false);
        assert_eq!(result, BTreeSet::from([10, 30]));

        let result = index.lookup_in(&[ScalarValue::Int64(1), ScalarValue::Int64(3)], true);
        assert_eq!(result, BTreeSet::from([20, 40]));
    }

    // -- Float64 index tests --

    #[test]
    fn float64_index_lookup_eq() {
        let descriptor = test_descriptor("score");
        let payloads = vec![
            BTreeMap::from([("score".into(), ScalarValue::Float64(1.5))]),
            BTreeMap::from([("score".into(), ScalarValue::Float64(2.5))]),
            BTreeMap::from([("score".into(), ScalarValue::Float64(1.5))]),
        ];
        let ids = vec![1, 2, 3];

        let index = InvertedScalarIndex::build_from_payloads(descriptor, "score", &payloads, &ids);

        let result = index.lookup_eq(&ScalarValue::Float64(1.5));
        assert_eq!(result, BTreeSet::from([1, 3]));
    }

    #[test]
    fn float64_index_range() {
        let descriptor = test_descriptor("score");
        let payloads = vec![
            BTreeMap::from([("score".into(), ScalarValue::Float64(1.0))]),
            BTreeMap::from([("score".into(), ScalarValue::Float64(2.0))]),
            BTreeMap::from([("score".into(), ScalarValue::Float64(3.0))]),
            BTreeMap::from([("score".into(), ScalarValue::Float64(4.0))]),
        ];
        let ids = vec![1, 2, 3, 4];

        let index = InvertedScalarIndex::build_from_payloads(descriptor, "score", &payloads, &ids);

        let result = index.lookup_range(RangeOp::Gt, &ScalarValue::Float64(2.0));
        assert_eq!(result, BTreeSet::from([3, 4]));

        let result = index.lookup_range(RangeOp::Lt, &ScalarValue::Float64(3.0));
        assert_eq!(result, BTreeSet::from([1, 2]));

        let result = index.lookup_range(RangeOp::Ne, &ScalarValue::Float64(2.0));
        assert_eq!(result, BTreeSet::from([1, 3, 4]));
    }

    // -- Bool index tests --

    #[test]
    fn bool_index_lookup() {
        let descriptor = test_descriptor("active");
        let payloads = vec![
            BTreeMap::from([("active".into(), ScalarValue::Bool(true))]),
            BTreeMap::from([("active".into(), ScalarValue::Bool(false))]),
            BTreeMap::from([("active".into(), ScalarValue::Bool(true))]),
        ];
        let ids = vec![1, 2, 3];

        let index = InvertedScalarIndex::build_from_payloads(descriptor, "active", &payloads, &ids);

        let result = index.lookup_eq(&ScalarValue::Bool(true));
        assert_eq!(result, BTreeSet::from([1, 3]));

        let result = index.lookup_eq(&ScalarValue::Bool(false));
        assert_eq!(result, BTreeSet::from([2]));
    }

    #[test]
    fn bool_index_all_ids() {
        let descriptor = test_descriptor("active");
        let payloads = vec![
            BTreeMap::from([("active".into(), ScalarValue::Bool(true))]),
            BTreeMap::from([("active".into(), ScalarValue::Bool(false))]),
        ];
        let ids = vec![1, 2];

        let index = InvertedScalarIndex::build_from_payloads(descriptor, "active", &payloads, &ids);
        assert_eq!(index.all_indexed_ids(), BTreeSet::from([1, 2]));
    }

    // -- Empty index tests --

    #[test]
    fn empty_index_when_no_values() {
        let descriptor = test_descriptor("missing");
        let payloads: Vec<BTreeMap<String, ScalarValue>> = vec![BTreeMap::new(), BTreeMap::new()];
        let ids = vec![1, 2];

        let index =
            InvertedScalarIndex::build_from_payloads(descriptor, "missing", &payloads, &ids);

        assert!(matches!(index, InvertedScalarIndex::Empty { .. }));
        assert!(index.all_indexed_ids().is_empty());
        assert!(index.lookup_eq(&ScalarValue::Int64(1)).is_empty());
    }

    #[test]
    fn new_is_empty() {
        let descriptor = test_descriptor("field");
        let index = InvertedScalarIndex::new(descriptor);
        assert!(matches!(index, InvertedScalarIndex::Empty { .. }));
    }

    // -- Field name accessor --

    #[test]
    fn field_name_accessor() {
        let descriptor = test_descriptor("my_field");
        let index = InvertedScalarIndex::new(descriptor);
        assert_eq!(index.field_name(), "my_field");
    }

    // -- Rows without the field are skipped --

    #[test]
    fn rows_without_field_are_skipped() {
        let descriptor = test_descriptor("name");
        let payloads = vec![
            BTreeMap::from([("name".into(), ScalarValue::String("alice".into()))]),
            BTreeMap::new(), // row 2 has no "name"
            BTreeMap::from([("name".into(), ScalarValue::String("bob".into()))]),
        ];
        let ids = vec![1, 2, 3];

        let index = InvertedScalarIndex::build_from_payloads(descriptor, "name", &payloads, &ids);

        assert_eq!(index.all_indexed_ids(), BTreeSet::from([1, 3]));
    }
}
