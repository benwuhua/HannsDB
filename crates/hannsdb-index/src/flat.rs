use crate::adapter::{AdapterError, HnswSearchHit, MetricKind, VectorIndexBackend};

pub struct FlatIndex {
    dim: usize,
    metric: MetricKind,
    points: Vec<(u64, Vec<f32>)>,
}

impl FlatIndex {
    pub fn new(dim: usize, metric: &str) -> Result<Self, AdapterError> {
        Ok(Self {
            dim,
            metric: MetricKind::parse(metric)?,
            points: Vec::new(),
        })
    }
}

impl VectorIndexBackend for FlatIndex {
    fn insert(&mut self, vectors: &[(u64, Vec<f32>)]) -> Result<(), AdapterError> {
        for (_, vector) in vectors {
            if vector.len() != self.dim {
                return Err(AdapterError::InvalidDimension {
                    expected: self.dim,
                    got: vector.len(),
                });
            }
        }
        self.points.extend(vectors.iter().cloned());
        Ok(())
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        _ef_search: usize,
    ) -> Result<Vec<HnswSearchHit>, AdapterError> {
        if query.len() != self.dim {
            return Err(AdapterError::InvalidDimension {
                expected: self.dim,
                got: query.len(),
            });
        }
        if k == 0 || self.points.is_empty() {
            return Ok(Vec::new());
        }

        let mut hits = self
            .points
            .iter()
            .map(|(id, vector)| HnswSearchHit {
                id: *id,
                distance: self.metric.distance(query, vector),
            })
            .collect::<Vec<_>>();
        hits.sort_by(|lhs, rhs| lhs.distance.total_cmp(&rhs.distance));
        Ok(hits.into_iter().take(k).collect())
    }
}
