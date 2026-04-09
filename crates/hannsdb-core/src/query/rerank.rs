use std::collections::{BTreeMap, HashMap};

use super::ast::QueryReranker;

/// Result of fusion: (external_id, fused_score).
pub type FusedResult = Vec<(i64, f64)>;

/// Per-field ranked results: field_name → sorted list of (id, distance).
pub type PerFieldResults = HashMap<String, Vec<(i64, f32)>>;

/// Apply the selected reranker to per-field results.
pub fn fuse(
    per_field: &PerFieldResults,
    reranker: &QueryReranker,
    metrics: &BTreeMap<String, String>,
    top_k: usize,
) -> FusedResult {
    match reranker {
        QueryReranker::Rrf { rank_constant } => rrf_fusion(per_field, *rank_constant, top_k),
        QueryReranker::Weighted { weights } => {
            weighted_fusion(per_field, weights, metrics, top_k)
        }
    }
}

/// RRF fusion: score(id) = Σ 1/(rank_constant + rank + 1) across fields.
///
/// Ranks are 0-based. Higher fused score = better.
fn rrf_fusion(per_field: &PerFieldResults, rank_constant: u64, top_k: usize) -> FusedResult {
    let rc = rank_constant as f64;
    let mut scores: HashMap<i64, f64> = HashMap::new();

    for ranked_list in per_field.values() {
        for (rank, (id, _distance)) in ranked_list.iter().enumerate() {
            let contribution = 1.0 / (rc + rank as f64 + 1.0);
            *scores.entry(*id).or_default() += contribution;
        }
    }

    let mut results: Vec<(i64, f64)> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.total_cmp(&a.1));
    results.truncate(top_k);
    results
}

/// Weighted fusion: normalize per-metric, then weighted sum.
///
/// Normalization:
/// - L2:     1 - 2*atan(score)/π
/// - IP:     0.5 + atan(score)/π
/// - Cosine: 1 - score/2
fn weighted_fusion(
    per_field: &PerFieldResults,
    weights: &BTreeMap<String, f64>,
    metrics: &BTreeMap<String, String>,
    top_k: usize,
) -> FusedResult {
    let mut scores: HashMap<i64, f64> = HashMap::new();

    for (field_name, ranked_list) in per_field {
        let weight = weights.get(field_name).copied().unwrap_or(1.0);
        let metric = metrics
            .get(field_name)
            .map(String::as_str)
            .unwrap_or("l2");

        for (id, distance) in ranked_list {
            let normalized = normalize_distance(*distance, metric);
            *scores.entry(*id).or_default() += weight * normalized;
        }
    }

    let mut results: Vec<(i64, f64)> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.total_cmp(&a.1));
    results.truncate(top_k);
    results
}

fn normalize_distance(distance: f32, metric: &str) -> f64 {
    let d = distance as f64;
    match metric {
        "l2" => 1.0 - 2.0 * d.atan() / std::f64::consts::PI,
        "ip" => 0.5 + d.atan() / std::f64::consts::PI,
        "cosine" => 1.0 - d / 2.0,
        _ => 1.0 - d, // fallback: treat as distance (lower = better → higher score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_basic() {
        let mut per_field = PerFieldResults::new();
        per_field.insert(
            "dense".to_string(),
            vec![(1, 0.1), (2, 0.2), (3, 0.3)],
        );
        per_field.insert(
            "title".to_string(),
            vec![(2, 0.1), (1, 0.2), (4, 0.3)],
        );

        let result = rrf_fusion(&per_field, 60, 10);
        // id=1: 1/(60+0+1) + 1/(60+1+1) = 1/61 + 1/62
        // id=2: 1/(60+1+1) + 1/(60+0+1) = 1/62 + 1/61
        // So id=1 and id=2 should have same score, then id=3, then id=4
        assert_eq!(result.len(), 4);
        // id=1 and id=2 tied at top
        assert!((result[0].1 - result[1].1).abs() < 1e-10);
    }

    #[test]
    fn weighted_basic() {
        let mut per_field = PerFieldResults::new();
        per_field.insert("dense".to_string(), vec![(1, 0.1), (2, 0.5)]);
        per_field.insert("title".to_string(), vec![(2, 0.1), (1, 0.5)]);

        let mut weights = BTreeMap::new();
        weights.insert("dense".to_string(), 0.7);
        weights.insert("title".to_string(), 0.3);

        let mut metrics = BTreeMap::new();
        metrics.insert("dense".to_string(), "l2".to_string());
        metrics.insert("title".to_string(), "l2".to_string());

        let result = weighted_fusion(&per_field, &weights, &metrics, 10);
        assert_eq!(result.len(), 2);
        // Both appear in both fields; dense has higher weight
        // id=1: 0.7*normalize(0.1,l2) + 0.3*normalize(0.5,l2)
        // id=2: 0.7*normalize(0.5,l2) + 0.3*normalize(0.1,l2)
        // Since 0.1 < 0.5 (closer), normalize(0.1,l2) > normalize(0.5,l2)
        // id=1 has 0.7*higher + 0.3*lower, id=2 has 0.7*lower + 0.3*higher
        // So id=1 should win
        assert_eq!(result[0].0, 1);
    }
}
