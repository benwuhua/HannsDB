use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::io;
use std::path::Path;

use crate::catalog::CollectionMetadata;
use crate::document::{compare_field_value_for_sort, FieldValue};
use crate::segment::{load_sparse_vectors, SegmentManager, SegmentMetadata, TombstoneMask};
use crate::storage::segment_io::{
    load_payloads_or_empty as load_segment_payloads_or_empty,
    load_primary_dense_rows_for_segment_or_empty,
    load_vectors_or_empty as load_segment_vectors_or_empty,
};

use super::hits::{compare_hits, DocumentHit};
use super::planner::{BruteForceExecutionMode, BruteForceQueryPlan};
use super::rerank::{fuse, PerFieldResults};
use super::search::{distance_by_metric, sparse_inner_product};
use super::QueryVector;

pub(crate) struct QueryExecutor;

pub(crate) fn project_hits_output_fields(
    hits: &mut [DocumentHit],
    output_fields: Option<&[String]>,
) {
    let Some(output_fields) = output_fields else {
        return;
    };
    let requested = output_fields.iter().cloned().collect::<BTreeSet<_>>();
    for hit in hits {
        hit.fields
            .retain(|field_name, _| requested.contains(field_name));
    }
}

/// Per-document per-field distances collected for reranking.
struct CollectedDoc {
    id: i64,
    fields: BTreeMap<String, FieldValue>,
    per_field_distance: HashMap<String, f32>,
    best_distance: Option<f32>,
}

impl QueryExecutor {
    pub(crate) fn execute(
        segment_manager: &SegmentManager,
        collection: &CollectionMetadata,
        plan: &BruteForceQueryPlan,
    ) -> io::Result<Vec<DocumentHit>> {
        if plan.top_k == 0 {
            return Ok(Vec::new());
        }

        let use_reranker = plan.reranker.is_some();
        let mut collected: HashMap<i64, CollectedDoc> = HashMap::new();
        let mut shadowed_ids = HashSet::new();
        let needs_secondary_vectors = plan.recall_sources.iter().any(|source| match &source.kind {
            super::planner::RecallSourceKind::ExplicitVector { field_name } => {
                field_name != &collection.primary_vector
            }
            super::planner::RecallSourceKind::QueryById { field_name, .. } => {
                field_name != &collection.primary_vector
            }
        });
        let needs_sparse_vectors = plan
            .recall_sources
            .iter()
            .any(|source| matches!(&source.vector, QueryVector::Sparse(_)));
        for segment in segment_manager.segment_paths()? {
            let segment_meta = SegmentMetadata::load_from_path(&segment.metadata)?;
            let dense_rows = load_primary_dense_rows_for_segment_or_empty(
                &segment,
                &segment_meta,
                &collection.primary_vector,
                collection.dimension,
                collection.primary_is_fp16(),
            )?;
            let records = dense_rows.primary_vectors;
            let external_ids = dense_rows.external_ids;
            let payloads =
                load_segment_payloads_or_empty(&segment, &segment_meta, external_ids.len())?;
            let vectors = if needs_secondary_vectors {
                Some(load_segment_vectors_or_empty(
                    &segment,
                    &segment_meta,
                    &collection.primary_vector,
                    external_ids.len(),
                )?)
            } else {
                None
            };
            let sparse_vectors = if needs_sparse_vectors {
                Some(load_sparse_vectors_or_empty(
                    &segment.sparse_vectors,
                    external_ids.len(),
                )?)
            } else {
                None
            };
            let tombstone = TombstoneMask::load_from_path(&segment.tombstones)?;

            if external_ids.len().saturating_mul(collection.dimension) != records.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "records and ids are not aligned",
                ));
            }

            for row_idx in (0..external_ids.len()).rev() {
                let id = external_ids[row_idx];
                if !shadowed_ids.insert(id) {
                    continue;
                }
                if tombstone.is_deleted(row_idx) {
                    continue;
                }

                let start = row_idx * collection.dimension;
                let end = start + collection.dimension;
                let vector = &records[start..end];

                let fields = &payloads[row_idx];
                if let Some(filter) = plan.filter.as_ref() {
                    if !filter.matches(fields) {
                        continue;
                    }
                }

                let per_field_distance = compute_per_field_distances(
                    plan,
                    vector,
                    vectors.as_deref(),
                    sparse_vectors.as_deref(),
                    row_idx,
                    &collection.primary_vector,
                )?;

                if per_field_distance.is_empty() && plan.mode == BruteForceExecutionMode::Recall {
                    continue;
                }

                let best_distance = per_field_distance
                    .values()
                    .copied()
                    .into_iter()
                    .reduce(f32::min);

                collected.insert(
                    id,
                    CollectedDoc {
                        id,
                        fields: fields.clone(),
                        per_field_distance,
                        best_distance,
                    },
                );
            }
        }

        let mut hits = if use_reranker {
            apply_reranker(collected, plan)
        } else {
            // Default: best-distance merge (existing behavior)
            collected
                .into_values()
                .filter_map(|doc| {
                    let distance = match doc.best_distance {
                        Some(d) => d,
                        None => 0.0, // FilterOnlyScan
                    };
                    Some(DocumentHit {
                        id: doc.id,
                        distance,
                        fields: doc.fields,
                        vectors: BTreeMap::new(),
                        sparse_vectors: BTreeMap::new(),
                        group_key: None,
                    })
                })
                .collect::<Vec<_>>()
        };
        if let Some(order_by) = plan.order_by.as_ref() {
            sort_hits_by_field(&mut hits, &order_by.field_name, order_by.descending);
        } else {
            hits.sort_by(compare_hits);
        }
        if let Some(group_by) = plan.group_by.as_ref() {
            return Ok(collapse_hits_by_group(
                hits,
                &group_by.field_name,
                group_by.group_topk,
                group_by.group_count,
                plan.top_k,
            ));
        }

        if hits.len() > plan.top_k {
            hits.truncate(plan.top_k);
        }
        Ok(hits)
    }
}

fn compute_per_field_distances(
    plan: &BruteForceQueryPlan,
    primary_vector: &[f32],
    secondary_vectors: Option<&[BTreeMap<String, Vec<f32>>]>,
    sparse_vectors: Option<&[BTreeMap<String, crate::document::SparseVector>]>,
    row_idx: usize,
    primary_vector_name: &str,
) -> io::Result<HashMap<String, f32>> {
    let mut distances = HashMap::new();
    for source in &plan.recall_sources {
        let candidate_vector = match &source.kind {
            super::planner::RecallSourceKind::QueryById { field_name, .. }
            | super::planner::RecallSourceKind::ExplicitVector { field_name } => {
                // First check sparse vectors, then dense.
                if let Some(sparse) = sparse_vectors {
                    if let Some(sv) = sparse.get(row_idx).and_then(|row| row.get(field_name)) {
                        Some(CandidateVector::Sparse(sv))
                    } else {
                        candidate_vector_for_field(
                            field_name,
                            primary_vector,
                            secondary_vectors,
                            row_idx,
                            primary_vector_name,
                        )
                    }
                } else {
                    candidate_vector_for_field(
                        field_name,
                        primary_vector,
                        secondary_vectors,
                        row_idx,
                        primary_vector_name,
                    )
                }
            }
        };
        let Some(candidate_vector) = candidate_vector else {
            continue;
        };
        let distance = match (&source.vector, candidate_vector) {
            (QueryVector::Dense(q), CandidateVector::Dense(v)) => {
                distance_by_metric(q, v, &source.metric)?
            }
            (QueryVector::Sparse(q), CandidateVector::Sparse(v)) => -sparse_inner_product(q, v),
            _ => continue, // type mismatch, skip
        };
        distances.insert(source.field_key(), distance);
    }
    Ok(distances)
}

fn apply_reranker(
    collected: HashMap<i64, CollectedDoc>,
    plan: &BruteForceQueryPlan,
) -> Vec<DocumentHit> {
    let reranker = plan.reranker.as_ref().expect("reranker must be present");

    // Build per-field ranked lists
    let mut per_field: PerFieldResults = HashMap::new();
    let mut metrics: BTreeMap<String, String> = BTreeMap::new();

    for source in &plan.recall_sources {
        let field_key = source.field_key();
        metrics.insert(field_key.clone(), source.metric.clone());
    }

    // Collect all (id, distance) per field, then sort
    for doc in collected.values() {
        for (field_key, distance) in &doc.per_field_distance {
            per_field
                .entry(field_key.clone())
                .or_default()
                .push((doc.id, *distance));
        }
    }

    // Sort each field's list by distance ascending
    for list in per_field.values_mut() {
        list.sort_by(|a, b| a.1.total_cmp(&b.1));
    }

    let fused = fuse(&per_field, reranker, &metrics, plan.top_k);

    // Build hits from fused results
    fused
        .into_iter()
        .filter_map(|(id, _fused_score)| {
            let doc = collected.get(&id)?;
            // Use the best_distance for display; fused_score is the rerank score
            let distance = doc.best_distance.unwrap_or(0.0);
            Some(DocumentHit {
                id,
                distance,
                fields: doc.fields.clone(),
                vectors: BTreeMap::new(),
                sparse_vectors: BTreeMap::new(),
                group_key: None,
            })
        })
        .collect()
}

enum CandidateVector<'a> {
    Dense(&'a [f32]),
    Sparse(&'a crate::document::SparseVector),
}

fn candidate_vector_for_field<'a>(
    field_name: &str,
    primary_vector: &'a [f32],
    secondary_vectors: Option<&'a [BTreeMap<String, Vec<f32>>]>,
    row_idx: usize,
    primary_vector_name: &str,
) -> Option<CandidateVector<'a>> {
    if field_name == primary_vector_name {
        Some(CandidateVector::Dense(primary_vector))
    } else if let Some(vectors) = secondary_vectors {
        vectors[row_idx]
            .get(field_name)
            .map(|v| CandidateVector::Dense(v.as_slice()))
    } else {
        None
    }
}

fn collapse_hits_by_group(
    hits: Vec<DocumentHit>,
    field_name: &str,
    group_topk: usize,
    group_count: usize,
    top_k: usize,
) -> Vec<DocumentHit> {
    // group_topk == 0 means unlimited per group; treat as a very large number.
    // Default behavior (group_topk == 0) keeps exactly 1 per group (legacy).
    let per_group_limit = if group_topk == 0 { 1 } else { group_topk };

    let mut per_group_count: HashMap<GroupByValueKey, usize> = HashMap::new();
    let mut groups_seen: usize = 0;
    let mut result = Vec::new();

    for mut hit in hits {
        let group_key_value = hit.fields.get(field_name).cloned();
        let group_key = group_key_value
            .as_ref()
            .map(GroupByValueKey::from_field_value)
            .unwrap_or(GroupByValueKey::Missing);

        let count = per_group_count.entry(group_key.clone()).or_insert(0);
        if *count == 0 {
            // First hit for this group: check group_count limit.
            if group_count > 0 && groups_seen >= group_count {
                continue;
            }
            groups_seen += 1;
        }
        if *count >= per_group_limit {
            continue;
        }
        *count += 1;
        hit.group_key = group_key_value;
        result.push(hit);
        if result.len() >= top_k {
            break;
        }
    }
    result
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum GroupByValueKey {
    Missing,
    String(String),
    Int64(i64),
    Float64(FloatGroupKey),
    Bool(bool),
}

impl GroupByValueKey {
    fn from_field_value(value: &FieldValue) -> Self {
        match value {
            FieldValue::String(value) => Self::String(value.clone()),
            FieldValue::Int64(value) => Self::Int64(*value),
            FieldValue::Int32(value) => Self::Int64(*value as i64),
            FieldValue::UInt32(value) => Self::Int64(*value as i64),
            FieldValue::UInt64(value) => Self::Int64(*value as i64),
            FieldValue::Float(value) => Self::Float64(FloatGroupKey::new(*value as f64)),
            FieldValue::Float64(value) => Self::Float64(FloatGroupKey::new(*value)),
            FieldValue::Bool(value) => Self::Bool(*value),
            FieldValue::Array(_) => Self::Missing, // arrays not groupable
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum FloatGroupKey {
    Nan,
    Zero,
    Exact(u64),
}

impl FloatGroupKey {
    fn new(value: f64) -> Self {
        if value.is_nan() {
            Self::Nan
        } else if value == 0.0 {
            Self::Zero
        } else {
            Self::Exact(value.to_bits())
        }
    }
}

fn load_sparse_vectors_or_empty(
    path: &Path,
    expected_rows: usize,
) -> io::Result<Vec<BTreeMap<String, crate::document::SparseVector>>> {
    match load_sparse_vectors(path) {
        Ok(vectors) => {
            if vectors.len() > expected_rows {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "sparse vector row count exceeds record row count",
                ));
            }
            if vectors.len() < expected_rows {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "sparse vector row count is shorter than record row count",
                ));
            }
            Ok(vectors)
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            Ok(vec![BTreeMap::new(); expected_rows])
        }
        Err(err) => Err(err),
    }
}

/// Sort hits by a scalar field value. Missing values sort last (ascending) or first (descending).
pub(crate) fn sort_hits_by_field(hits: &mut [DocumentHit], field_name: &str, descending: bool) {
    hits.sort_by(|a, b| {
        let av = a.fields.get(field_name);
        let bv = b.fields.get(field_name);
        let ord = match (av, bv) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) => Ordering::Greater,
            (Some(_), None) => Ordering::Less,
            (Some(av), Some(bv)) => compare_field_value_for_sort(av, bv),
        };
        if descending { ord.reverse() } else { ord }.then_with(|| compare_hits(a, b))
    });
}
