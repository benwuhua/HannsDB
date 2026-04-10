use crate::adapter::{AdapterError, HnswSearchHit, MetricKind, VectorIndexBackend};

#[cfg(feature = "knowhere-backend")]
use std::sync::Arc;

// ---------------------------------------------------------------------------
// knowhere-backend: real IVF backed by hanns::faiss::IvfFlatIndex
// ---------------------------------------------------------------------------

#[cfg(feature = "knowhere-backend")]
pub struct IvfIndex {
    inner: knowhere_rs::faiss::IvfFlatIndex,
    dim: usize,
    #[allow(dead_code)]
    metric: MetricKind,
    #[allow(dead_code)]
    nlist: usize,
}

#[cfg(feature = "knowhere-backend")]
impl IvfIndex {
    pub fn new(dim: usize, metric: &str, nlist: usize) -> Result<Self, AdapterError> {
        let metric_kind = MetricKind::parse(metric)?;
        let metric_type = match metric_kind {
            MetricKind::L2 => knowhere_rs::MetricType::L2,
            MetricKind::Cosine => knowhere_rs::MetricType::Cosine,
            MetricKind::Ip => knowhere_rs::MetricType::Ip,
        };
        let nlist = nlist.max(1);
        let config = knowhere_rs::IndexConfig {
            index_type: knowhere_rs::IndexType::IvfFlat,
            metric_type,
            dim,
            data_type: knowhere_rs::api::data_type::DataType::Float,
            params: knowhere_rs::api::index::IndexParams::ivf(nlist, nlist.min(8)),
        };
        let inner = knowhere_rs::faiss::IvfFlatIndex::new(&config)
            .map_err(|e| AdapterError::Backend(format!("ivf create failed: {e}")))?;

        Ok(Self {
            inner,
            dim,
            metric: metric_kind,
            nlist: nlist.max(1),
        })
    }

    pub fn from_bytes(
        dim: usize,
        metric: &str,
        nlist: usize,
        bytes: &[u8],
    ) -> Result<Self, AdapterError> {
        let metric_kind = MetricKind::parse(metric)?;
        let inner = knowhere_rs::faiss::IvfFlatIndex::deserialize_from_bytes(bytes, dim)
            .map_err(|e| AdapterError::Backend(format!("ivf deserialize failed: {e}")))?;
        Ok(Self {
            inner,
            dim,
            metric: metric_kind,
            nlist: nlist.max(1),
        })
    }

    pub fn search_with_bitset(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
        bitset: &knowhere_rs::BitsetView,
    ) -> Result<Vec<HnswSearchHit>, AdapterError> {
        if query.len() != self.dim {
            return Err(AdapterError::InvalidDimension {
                expected: self.dim,
                got: query.len(),
            });
        }
        let predicate = knowhere_rs::api::search::BitsetPredicate::new(bitset.clone());
        let req = knowhere_rs::SearchRequest {
            top_k: k,
            nprobe,
            filter: Some(Arc::new(predicate)),
            params: None,
            radius: None,
        };
        let result = self
            .inner
            .search(query, &req)
            .map_err(|e| AdapterError::Backend(format!("ivf search_with_bitset failed: {e}")))?;

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

#[cfg(feature = "knowhere-backend")]
impl VectorIndexBackend for IvfIndex {
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
            .map_err(|e| AdapterError::Backend(format!("ivf train failed: {e}")))?;
        self.inner
            .add(&flat, Some(&ids))
            .map_err(|e| AdapterError::Backend(format!("ivf add failed: {e}")))?;
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
        if !vectors.is_empty() {
            self.inner
                .train(vectors)
                .map_err(|e| AdapterError::Backend(format!("ivf train failed: {e}")))?;
            self.inner
                .add(vectors, None)
                .map_err(|e| AdapterError::Backend(format!("ivf add failed: {e}")))?;
        }
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
            .map_err(|e| AdapterError::Backend(format!("ivf search failed: {e}")))?;

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

    fn serialize_to_bytes(&self) -> Result<Option<Vec<u8>>, AdapterError> {
        let bytes = self
            .inner
            .serialize_to_bytes()
            .map_err(|e| AdapterError::Backend(format!("ivf serialize failed: {e}")))?;
        Ok(Some(bytes))
    }

    fn search_with_bitset(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        bitset: &knowhere_rs::BitsetView,
    ) -> Result<Vec<HnswSearchHit>, AdapterError> {
        // Delegate to the inherent method which uses BitsetPredicate for pre-filter.
        self.search_with_bitset(query, k, ef_search, bitset)
    }
}

// ---------------------------------------------------------------------------
// Fallback without knowhere-backend: FlatIndex wrapper (existing behavior)
// ---------------------------------------------------------------------------

#[cfg(not(feature = "knowhere-backend"))]
use crate::flat::FlatIndex;

#[cfg(not(feature = "knowhere-backend"))]
pub struct IvfIndex {
    inner: FlatIndex,
    #[allow(dead_code)]
    nlist: usize,
}

#[cfg(not(feature = "knowhere-backend"))]
impl IvfIndex {
    pub fn new(dim: usize, metric: &str, nlist: usize) -> Result<Self, AdapterError> {
        Ok(Self {
            inner: FlatIndex::new(dim, metric)?,
            nlist: nlist.max(1),
        })
    }
}

#[cfg(not(feature = "knowhere-backend"))]
impl VectorIndexBackend for IvfIndex {
    fn insert(&mut self, vectors: &[(u64, Vec<f32>)]) -> Result<(), AdapterError> {
        self.inner.insert(vectors)
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<HnswSearchHit>, AdapterError> {
        self.inner.search(query, k, ef_search)
    }
}
