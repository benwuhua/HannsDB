use crate::adapter::{AdapterError, HnswBackend, HnswSearchHit};

#[derive(Debug, Clone, Copy)]
enum MetricKind {
    L2,
    Cosine,
    Ip,
}

impl MetricKind {
    fn parse(metric: &str) -> Result<Self, AdapterError> {
        match metric.to_ascii_lowercase().as_str() {
            "l2" => Ok(Self::L2),
            "cosine" => Ok(Self::Cosine),
            "ip" => Ok(Self::Ip),
            other => Err(AdapterError::Backend(format!(
                "unsupported metric: {other}"
            ))),
        }
    }

    fn distance(self, query: &[f32], vector: &[f32]) -> f32 {
        match self {
            Self::L2 => l2_sq(query, vector),
            Self::Cosine => cosine_distance(query, vector),
            Self::Ip => -dot(query, vector),
        }
    }
}

pub struct InMemoryHnswIndex {
    dim: usize,
    metric: MetricKind,
    points: Vec<(u64, Vec<f32>)>,
}

impl InMemoryHnswIndex {
    pub fn new(dim: usize, metric: &str) -> Result<Self, AdapterError> {
        let metric = MetricKind::parse(metric)?;
        Ok(Self {
            dim,
            metric,
            points: Vec::new(),
        })
    }
}

impl HnswBackend for InMemoryHnswIndex {
    fn insert(&mut self, vectors: &[(u64, Vec<f32>)]) -> Result<(), AdapterError> {
        for (_, v) in vectors {
            if v.len() != self.dim {
                return Err(AdapterError::InvalidDimension {
                    expected: self.dim,
                    got: v.len(),
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

        let mut scored: Vec<HnswSearchHit> = self
            .points
            .iter()
            .map(|(id, vec)| HnswSearchHit {
                id: *id,
                distance: self.metric.distance(query, vec),
            })
            .collect();
        scored.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        Ok(scored.into_iter().take(k).collect())
    }
}

fn l2_sq(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum()
}

fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum()
}

fn cosine_distance(lhs: &[f32], rhs: &[f32]) -> f32 {
    let num = dot(lhs, rhs);
    let lhs_norm = lhs.iter().map(|x| x * x).sum::<f32>().sqrt();
    let rhs_norm = rhs.iter().map(|x| x * x).sum::<f32>().sqrt();
    if lhs_norm == 0.0 || rhs_norm == 0.0 {
        return 1.0;
    }
    1.0 - (num / (lhs_norm * rhs_norm))
}

#[cfg(feature = "knowhere-backend")]
pub struct KnowhereHnswIndex {
    dim: usize,
    inner: knowhere_rs::HnswIndex,
}

#[cfg(feature = "knowhere-backend")]
impl KnowhereHnswIndex {
    pub fn new(dim: usize, metric: &str) -> Result<Self, AdapterError> {
        let metric_type = match MetricKind::parse(metric)? {
            MetricKind::L2 => knowhere_rs::MetricType::L2,
            MetricKind::Cosine => knowhere_rs::MetricType::Cosine,
            MetricKind::Ip => knowhere_rs::MetricType::Ip,
        };
        let mut cfg = knowhere_rs::IndexConfig::new(knowhere_rs::IndexType::Hnsw, metric_type, dim);
        // Keep index-level ef floor minimal; query-time ef_search is provided per request.
        cfg.params.ef_search = Some(1);
        cfg.params.ef_construction = Some(128);
        cfg.params.m = Some(16);
        cfg.params.random_seed = Some(42);

        let inner = knowhere_rs::HnswIndex::new(&cfg)
            .map_err(|e| AdapterError::Backend(format!("knowhere create failed: {e}")))?;

        Ok(Self { dim, inner })
    }

    pub fn serialize_to_bytes(&self) -> Result<Vec<u8>, AdapterError> {
        self.inner
            .serialize_to_bytes()
            .map_err(|e| AdapterError::Backend(format!("hnsw serialize failed: {e}")))
    }

    pub fn from_bytes(dim: usize, bytes: &[u8]) -> Result<Self, AdapterError> {
        let inner = knowhere_rs::HnswIndex::deserialize_from_bytes(bytes)
            .map_err(|e| AdapterError::Backend(format!("hnsw deserialize failed: {e}")))?;
        Ok(Self { dim, inner })
    }
}

#[cfg(feature = "knowhere-backend")]
impl HnswBackend for KnowhereHnswIndex {
    fn insert(&mut self, vectors: &[(u64, Vec<f32>)]) -> Result<(), AdapterError> {
        for (_, v) in vectors {
            if v.len() != self.dim {
                return Err(AdapterError::InvalidDimension {
                    expected: self.dim,
                    got: v.len(),
                });
            }
        }
        let mut flat = Vec::with_capacity(vectors.len() * self.dim);
        let mut ids = Vec::with_capacity(vectors.len());
        for (id, v) in vectors {
            flat.extend_from_slice(v);
            ids.push(*id as i64);
        }

        self.inner
            .train(&flat)
            .map_err(|e| AdapterError::Backend(format!("knowhere train failed: {e}")))?;
        self.inner
            .add(&flat, Some(&ids))
            .map_err(|e| AdapterError::Backend(format!("knowhere add failed: {e}")))?;
        Ok(())
    }

    fn insert_flat(
        &mut self,
        ids: &[u64],
        vectors: &[f32],
        dim: usize,
    ) -> Result<(), AdapterError> {
        if dim != self.dim {
            return Err(AdapterError::InvalidDimension {
                expected: self.dim,
                got: dim,
            });
        }
        if vectors.len() % self.dim != 0 {
            return Err(AdapterError::InvalidDimension {
                expected: self.dim,
                got: vectors.len() % self.dim,
            });
        }
        let count = vectors.len() / self.dim;
        if ids.len() != count {
            return Err(AdapterError::Backend(format!(
                "id count mismatch for flat insert: ids={}, vectors={count}",
                ids.len()
            )));
        }
        let ids = ids.iter().map(|id| *id as i64).collect::<Vec<_>>();

        self.inner
            .train(vectors)
            .map_err(|e| AdapterError::Backend(format!("knowhere train failed: {e}")))?;
        self.inner
            .add(vectors, Some(&ids))
            .map_err(|e| AdapterError::Backend(format!("knowhere add failed: {e}")))?;
        Ok(())
    }

    fn insert_flat_identity(&mut self, vectors: &[f32], dim: usize) -> Result<(), AdapterError> {
        if dim != self.dim {
            return Err(AdapterError::InvalidDimension {
                expected: self.dim,
                got: dim,
            });
        }
        if vectors.len() % self.dim != 0 {
            return Err(AdapterError::InvalidDimension {
                expected: self.dim,
                got: vectors.len() % self.dim,
            });
        }
        self.inner
            .train(vectors)
            .map_err(|e| AdapterError::Backend(format!("knowhere train failed: {e}")))?;
        self.inner
            .add(vectors, None)
            .map_err(|e| AdapterError::Backend(format!("knowhere add failed: {e}")))?;
        Ok(())
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<HnswSearchHit>, AdapterError> {
        if query.len() != self.dim {
            return Err(AdapterError::InvalidDimension {
                expected: self.dim,
                got: query.len(),
            });
        }
        let req = knowhere_rs::SearchRequest {
            top_k: k,
            nprobe: ef_search,
            filter: None,
            params: None,
            radius: None,
        };
        let result = self
            .inner
            .search(query, &req)
            .map_err(|e| AdapterError::Backend(format!("knowhere search failed: {e}")))?;

        Ok(result
            .ids
            .into_iter()
            .zip(result.distances)
            .filter_map(|(id, distance)| {
                if id < 0 {
                    None
                } else {
                    Some(HnswSearchHit {
                        id: id as u64,
                        distance,
                    })
                }
            })
            .collect())
    }
}
