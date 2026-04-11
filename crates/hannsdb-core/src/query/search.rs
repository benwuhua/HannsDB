use std::io;

use crate::document::SparseVector;
use crate::segment::TombstoneMask;

#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit {
    pub id: i64,
    pub distance: f32,
}

pub fn search_by_metric(
    records: &[f32],
    external_ids: &[i64],
    dimension: usize,
    tombstones: &TombstoneMask,
    query: &[f32],
    top_k: usize,
    metric: &str,
) -> io::Result<Vec<SearchHit>> {
    if dimension == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "dimension must be > 0",
        ));
    }
    if query.len() != dimension {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "query dimension mismatch",
        ));
    }
    if records.len() % dimension != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "records are not aligned to dimension",
        ));
    }
    if external_ids.len() != records.len() / dimension {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "external ids count does not match record count",
        ));
    }

    let mut hits = Vec::new();
    for (idx, vector) in records.chunks_exact(dimension).enumerate() {
        if tombstones.is_deleted(idx) {
            continue;
        }
        let distance = distance_by_metric(query, vector, metric)?;
        hits.push(SearchHit {
            id: external_ids[idx],
            distance,
        });
    }

    hits.sort_by(|a, b| {
        a.distance
            .total_cmp(&b.distance)
            .then_with(|| a.id.cmp(&b.id))
    });
    if hits.len() > top_k {
        hits.truncate(top_k);
    }
    Ok(hits)
}

/// Brute-force search over sparse vectors.
///
/// Each entry in `sparse_vectors` corresponds to an external ID at the same index.
/// The distance is the negated sparse inner product (so smaller = more similar).
pub fn search_sparse_bruteforce(
    sparse_vectors: &[SparseVector],
    external_ids: &[i64],
    tombstones: &TombstoneMask,
    query: &SparseVector,
    top_k: usize,
) -> io::Result<Vec<SearchHit>> {
    if sparse_vectors.len() != external_ids.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "sparse vector count does not match external id count",
        ));
    }

    let mut hits = Vec::new();
    for (idx, vector) in sparse_vectors.iter().enumerate() {
        if tombstones.is_deleted(idx) {
            continue;
        }
        let distance = -sparse_inner_product(query, vector);
        hits.push(SearchHit {
            id: external_ids[idx],
            distance,
        });
    }

    hits.sort_by(|a, b| {
        a.distance
            .total_cmp(&b.distance)
            .then_with(|| a.id.cmp(&b.id))
    });
    if hits.len() > top_k {
        hits.truncate(top_k);
    }
    Ok(hits)
}

/// Sparse inner product: sum of products where both vectors have matching indices.
///
/// Both vectors must have sorted indices. Uses a merge-style two-pointer algorithm.
pub fn sparse_inner_product(a: &SparseVector, b: &SparseVector) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0usize;
    let mut j = 0usize;
    while i < a.indices.len() && j < b.indices.len() {
        match a.indices[i].cmp(&b.indices[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                sum += a.values[i] * b.values[j];
                i += 1;
                j += 1;
            }
        }
    }
    sum
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

pub fn distance_by_metric(query: &[f32], vector: &[f32], metric: &str) -> io::Result<f32> {
    match metric.to_ascii_lowercase().as_str() {
        "l2" => Ok(l2_distance(query, vector)),
        "ip" => Ok(-inner_product(query, vector)),
        "cosine" => Ok(1.0 - cosine_similarity(query, vector)),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("unsupported metric: {other}"),
        )),
    }
}

fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = inner_product(a, b);
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return -1.0;
    }
    dot / (norm_a * norm_b)
}
