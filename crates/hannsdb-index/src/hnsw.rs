use crate::adapter::{AdapterError, HnswSearchHit, MetricKind, VectorIndexBackend};

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

impl VectorIndexBackend for InMemoryHnswIndex {
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

#[cfg(feature = "hanns-backend")]
pub struct KnowhereHnswIndex {
    dim: usize,
    inner: hanns::HnswIndex,
}

#[cfg(feature = "hanns-backend")]
impl KnowhereHnswIndex {
    pub fn new(
        dim: usize,
        metric: &str,
        m: usize,
        ef_construction: usize,
    ) -> Result<Self, AdapterError> {
        let metric_type = match MetricKind::parse(metric)? {
            MetricKind::L2 => hanns::MetricType::L2,
            MetricKind::Cosine => hanns::MetricType::Cosine,
            MetricKind::Ip => hanns::MetricType::Ip,
        };
        let mut cfg = hanns::IndexConfig::new(hanns::IndexType::Hnsw, metric_type, dim);
        // Keep index-level ef floor minimal; query-time ef_search is provided per request.
        cfg.params.ef_search = Some(1);
        cfg.params.ef_construction = Some(ef_construction);
        cfg.params.m = Some(m);
        cfg.params.random_seed = Some(42);

        let inner = hanns::HnswIndex::new(&cfg)
            .map_err(|e| AdapterError::Backend(format!("knowhere create failed: {e}")))?;

        Ok(Self { dim, inner })
    }

    pub fn serialize_to_bytes(&self) -> Result<Vec<u8>, AdapterError> {
        self.inner.serialize_to_bytes().map_err(|e| {
            AdapterError::Backend(format!("hnsw serialize failed (dim={}): {e}", self.dim))
        })
    }

    pub fn from_bytes(dim: usize, bytes: &[u8]) -> Result<Self, AdapterError> {
        let inner = std::panic::catch_unwind(|| hanns::HnswIndex::deserialize_from_bytes(bytes))
            .map_err(|_| {
                AdapterError::Backend(format!(
                    "hnsw deserialize panicked (dim={dim}, bytes={})",
                    bytes.len()
                ))
            })?
            .map_err(|e| {
                AdapterError::Backend(format!(
                    "hnsw deserialize failed (dim={dim}, bytes={}): {e}",
                    bytes.len()
                ))
            })?;
        Ok(Self { dim, inner })
    }
}

#[cfg(feature = "hanns-backend")]
impl VectorIndexBackend for KnowhereHnswIndex {
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
        let req = hanns::SearchRequest {
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

    fn search_into(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        ids_out: &mut [i64],
        dists_out: &mut [f32],
    ) -> Result<usize, AdapterError> {
        if query.len() != self.dim {
            return Err(AdapterError::InvalidDimension {
                expected: self.dim,
                got: query.len(),
            });
        }
        let k = k.min(ids_out.len()).min(dists_out.len());
        let req = hanns::SearchRequest {
            top_k: k,
            nprobe: ef_search,
            filter: None,
            params: None,
            radius: None,
        };
        self.inner
            .search_into(query, &req, ids_out, dists_out)
            .map_err(|e| AdapterError::Backend(format!("knowhere search_into failed: {e}")))
    }

    fn serialize_to_bytes(&self) -> Result<Option<Vec<u8>>, AdapterError> {
        KnowhereHnswIndex::serialize_to_bytes(self).map(Some)
    }

    #[cfg(feature = "hanns-backend")]
    fn search_with_bitset(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        bitset: &hanns::BitsetView,
    ) -> Result<Vec<HnswSearchHit>, AdapterError> {
        if query.len() != self.dim {
            return Err(AdapterError::InvalidDimension {
                expected: self.dim,
                got: query.len(),
            });
        }
        let req = hanns::SearchRequest {
            top_k: k,
            nprobe: ef_search,
            filter: None,
            params: None,
            radius: None,
        };
        let result = self
            .inner
            .search_with_bitset(query, &req, bitset)
            .map_err(|e| {
                AdapterError::Backend(format!("knowhere search_with_bitset failed: {e}"))
            })?;

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
