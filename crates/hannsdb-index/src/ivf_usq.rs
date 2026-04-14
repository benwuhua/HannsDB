use crate::adapter::AdapterError;

#[cfg(feature = "hanns-backend")]
use crate::adapter::{HnswSearchHit, MetricKind, VectorIndexBackend};

#[cfg(feature = "hanns-backend")]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "hanns-backend")]
static IVF_USQ_TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

// ---------------------------------------------------------------------------
// hanns-backend: real IVF-USQ backed by hanns::faiss::IvfUsqIndex
// ---------------------------------------------------------------------------

#[cfg(feature = "hanns-backend")]
pub struct IvfUsqIndex {
    inner: hanns::faiss::IvfUsqIndex,
    dim: usize,
    #[allow(dead_code)]
    metric: MetricKind,
    #[allow(dead_code)]
    nlist: usize,
    #[allow(dead_code)]
    bits_per_dim: usize,
    #[allow(dead_code)]
    rotation_seed: usize,
    #[allow(dead_code)]
    rerank_k: usize,
    #[allow(dead_code)]
    use_high_accuracy_scan: bool,
}

#[cfg(feature = "hanns-backend")]
impl IvfUsqIndex {
    pub fn new(
        dim: usize,
        metric: &str,
        nlist: usize,
        bits_per_dim: usize,
        rotation_seed: usize,
        rerank_k: usize,
        use_high_accuracy_scan: bool,
    ) -> Result<Self, AdapterError> {
        let metric_kind = MetricKind::parse(metric)?;
        let metric_type = match metric_kind {
            MetricKind::L2 => hanns::MetricType::L2,
            MetricKind::Cosine => hanns::MetricType::Cosine,
            MetricKind::Ip => hanns::MetricType::Ip,
        };
        let nlist = nlist.max(1);
        let config = hanns::IndexConfig {
            index_type: hanns::IndexType::IvfUsq,
            metric_type,
            dim,
            data_type: hanns::api::data_type::DataType::Float,
            params: hanns::api::index::IndexParams {
                nlist: Some(nlist),
                nprobe: Some(nlist.min(8)),
                exrabitq_bits_per_dim: Some(bits_per_dim),
                exrabitq_rerank_k: Some(rerank_k),
                exrabitq_rotation_seed: Some(rotation_seed as u64),
                exrabitq_use_high_accuracy_scan: Some(use_high_accuracy_scan),
                ..Default::default()
            },
        };
        let inner = hanns::faiss::IvfUsqIndex::from_index_config(&config)
            .map_err(|e| AdapterError::Backend(format!("ivf_usq create failed: {e}")))?;

        Ok(Self {
            inner,
            dim,
            metric: metric_kind,
            nlist,
            bits_per_dim,
            rotation_seed,
            rerank_k,
            use_high_accuracy_scan,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_bytes(
        dim: usize,
        metric: &str,
        nlist: usize,
        bits_per_dim: usize,
        rotation_seed: usize,
        rerank_k: usize,
        use_high_accuracy_scan: bool,
        bytes: &[u8],
    ) -> Result<Self, AdapterError> {
        let path = temp_index_path("ivf_usq");
        std::fs::write(&path, bytes)
            .map_err(|e| AdapterError::Backend(format!("ivf_usq bytes write failed: {e}")))?;
        let inner = hanns::faiss::IvfUsqIndex::load(&path)
            .map_err(|e| AdapterError::Backend(format!("ivf_usq deserialize failed: {e}")));
        let _ = std::fs::remove_file(&path);
        let inner = inner?;
        Ok(Self {
            inner,
            dim,
            metric: MetricKind::parse(metric)?,
            nlist: nlist.max(1),
            bits_per_dim,
            rotation_seed,
            rerank_k,
            use_high_accuracy_scan,
        })
    }

    fn temp_index_path(prefix: &str) -> std::path::PathBuf {
        temp_index_path(prefix)
    }
}

#[cfg(feature = "hanns-backend")]
fn temp_index_path(prefix: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "hannsdb_index_{prefix}_{}_{}.bin",
        std::process::id(),
        IVF_USQ_TMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ))
}

#[cfg(feature = "hanns-backend")]
impl VectorIndexBackend for IvfUsqIndex {
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
            .map_err(|e| AdapterError::Backend(format!("ivf_usq train failed: {e}")))?;
        self.inner
            .add(&flat, Some(&ids))
            .map_err(|e| AdapterError::Backend(format!("ivf_usq add failed: {e}")))?;
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
                .map_err(|e| AdapterError::Backend(format!("ivf_usq train failed: {e}")))?;
            self.inner
                .add(vectors, None)
                .map_err(|e| AdapterError::Backend(format!("ivf_usq add failed: {e}")))?;
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
            .map_err(|e| AdapterError::Backend(format!("ivf_usq search failed: {e}")))?;

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
        let overfetch_k = self.inner.ntotal().max(k);
        let req = hanns::SearchRequest {
            top_k: overfetch_k,
            nprobe: ef_search,
            filter: None,
            params: None,
            radius: None,
        };
        let result = self.inner.search(query, &req).map_err(|e| {
            AdapterError::Backend(format!("ivf_usq search_with_bitset failed: {e}"))
        })?;

        Ok(result
            .ids
            .into_iter()
            .zip(result.distances)
            .filter_map(|(id, distance)| {
                if id < 0 || bitset.get(id as usize) {
                    None
                } else {
                    Some(HnswSearchHit {
                        id: id as u64,
                        distance,
                    })
                }
            })
            .take(k)
            .collect())
    }

    fn serialize_to_bytes(&self) -> Result<Option<Vec<u8>>, AdapterError> {
        let path = Self::temp_index_path("ivf_usq");
        let result = self
            .inner
            .save(&path)
            .map_err(|e| AdapterError::Backend(format!("ivf_usq save failed: {e}")))
            .and_then(|_| {
                std::fs::read(&path)
                    .map_err(|e| AdapterError::Backend(format!("ivf_usq read bytes failed: {e}")))
            });
        let _ = std::fs::remove_file(&path);
        let bytes = result?;
        Ok(Some(bytes))
    }
}

// ---------------------------------------------------------------------------
// Fallback without hanns-backend: explicit unsupported error
// ---------------------------------------------------------------------------

#[cfg(not(feature = "hanns-backend"))]
pub struct IvfUsqIndex;

#[cfg(not(feature = "hanns-backend"))]
impl IvfUsqIndex {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        _dim: usize,
        _metric: &str,
        _nlist: usize,
        _bits_per_dim: usize,
        _rotation_seed: usize,
        _rerank_k: usize,
        _use_high_accuracy_scan: bool,
    ) -> Result<Self, AdapterError> {
        Err(AdapterError::Backend(
            "ivf_usq requires hanns-backend".to_string(),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_bytes(
        _dim: usize,
        _metric: &str,
        _nlist: usize,
        _bits_per_dim: usize,
        _rotation_seed: usize,
        _rerank_k: usize,
        _use_high_accuracy_scan: bool,
        _bytes: &[u8],
    ) -> Result<Self, AdapterError> {
        Err(AdapterError::Backend(
            "ivf_usq requires hanns-backend".to_string(),
        ))
    }
}
