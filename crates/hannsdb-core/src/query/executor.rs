use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::io;
use std::path::Path;

use crate::catalog::CollectionMetadata;
use crate::db::DocumentHit;
use crate::document::FieldValue;
use crate::segment::{load_payloads, load_record_ids, load_records, SegmentManager, TombstoneMask};

use super::planner::QueryPlan;
use super::search::distance_by_metric;

pub(crate) struct QueryExecutor;

impl QueryExecutor {
    pub(crate) fn execute(
        segment_manager: &SegmentManager,
        collection: &CollectionMetadata,
        plan: &QueryPlan,
    ) -> io::Result<Vec<DocumentHit>> {
        if plan.top_k == 0 {
            return Ok(Vec::new());
        }

        let mut best_hits = HashMap::new();
        for segment in segment_manager.segment_paths()? {
            let records = load_records_or_empty(&segment.records, collection.dimension)?;
            let external_ids = load_record_ids_or_empty(&segment.external_ids)?;
            let payloads = load_payloads_or_empty(&segment.payloads, external_ids.len())?;
            let tombstone = TombstoneMask::load_from_path(&segment.tombstones)?;

            if external_ids.len().saturating_mul(collection.dimension) != records.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "records and ids are not aligned",
                ));
            }

            for (row_idx, vector) in records.chunks_exact(collection.dimension).enumerate() {
                if tombstone.is_deleted(row_idx) {
                    continue;
                }

                let fields = &payloads[row_idx];
                if let Some(filter) = plan.filter.as_ref() {
                    if !filter.matches(fields) {
                        continue;
                    }
                }

                let candidate = DocumentHit {
                    id: external_ids[row_idx],
                    distance: best_distance(plan, vector, &collection.metric)?,
                    fields: fields.clone(),
                };
                insert_best_hit(&mut best_hits, candidate);
            }
        }

        let mut hits = best_hits.into_values().collect::<Vec<_>>();
        hits.sort_by(compare_hits);
        if hits.len() > plan.top_k {
            hits.truncate(plan.top_k);
        }
        Ok(hits)
    }
}

fn best_distance(plan: &QueryPlan, vector: &[f32], metric: &str) -> io::Result<f32> {
    let mut best = None;
    for source in &plan.recall_sources {
        let _ = &source.kind;
        let distance = distance_by_metric(&source.vector, vector, metric)?;
        match best {
            Some(current) if distance >= current => {}
            _ => best = Some(distance),
        }
    }

    best.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "query plan must contain at least one recall source",
        )
    })
}

fn insert_best_hit(best_hits: &mut HashMap<i64, DocumentHit>, candidate: DocumentHit) {
    match best_hits.get_mut(&candidate.id) {
        Some(existing) if compare_hits(&candidate, existing) == Ordering::Less => {
            *existing = candidate;
        }
        None => {
            best_hits.insert(candidate.id, candidate);
        }
        _ => {}
    }
}

fn compare_hits(left: &DocumentHit, right: &DocumentHit) -> Ordering {
    left.distance
        .total_cmp(&right.distance)
        .then_with(|| left.id.cmp(&right.id))
}

fn load_records_or_empty(path: &Path, dimension: usize) -> io::Result<Vec<f32>> {
    match load_records(path, dimension) {
        Ok(records) => Ok(records),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(err) => Err(err),
    }
}

fn load_record_ids_or_empty(path: &Path) -> io::Result<Vec<i64>> {
    match load_record_ids(path) {
        Ok(ids) => Ok(ids),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(err) => Err(err),
    }
}

fn load_payloads_or_empty(
    path: &Path,
    expected_rows: usize,
) -> io::Result<Vec<BTreeMap<String, FieldValue>>> {
    match load_payloads(path) {
        Ok(mut payloads) => {
            if payloads.len() > expected_rows {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "payload row count exceeds record row count",
                ));
            }
            if payloads.len() < expected_rows {
                payloads.resize_with(expected_rows, BTreeMap::new);
            }
            Ok(payloads)
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            Ok(vec![BTreeMap::new(); expected_rows])
        }
        Err(err) => Err(err),
    }
}
