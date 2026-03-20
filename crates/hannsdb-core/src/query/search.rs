use std::io;

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
