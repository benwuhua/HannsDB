use crate::adapter::AdapterError;

#[cfg(feature = "hanns-backend")]
use crate::adapter::{HnswSearchHit, MetricKind, VectorIndexBackend};

#[cfg(feature = "hanns-backend")]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "hanns-backend")]
static HNSW_HVQ_TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "hanns-backend")]
const HNSW_HVQ_ADAPTER_MAGIC: &[u8; 8] = b"HDBHHVQ0";

// ---------------------------------------------------------------------------
// hanns-backend: real HNSW-HVQ backed by hanns::faiss::HnswHvqIndex
// ---------------------------------------------------------------------------

#[cfg(feature = "hanns-backend")]
pub struct HnswHvqIndex {
    inner: hanns::faiss::HnswHvqIndex,
    dim: usize,
    external_ids: Vec<u64>,
}

#[cfg(feature = "hanns-backend")]
impl HnswHvqIndex {
    pub fn new(
        dim: usize,
        metric: &str,
        m: usize,
        m_max0: usize,
        ef_construction: usize,
        ef_search: usize,
        nbits: usize,
    ) -> Result<Self, AdapterError> {
        let metric_kind = MetricKind::parse(metric)?;
        let metric_type =
            match metric_kind {
                MetricKind::Ip => hanns::MetricType::Ip,
                MetricKind::Cosine => hanns::MetricType::Cosine,
                MetricKind::L2 => return Err(AdapterError::Backend(
                    "hnsw_hvq currently supports only ip/cosine in Hanns; HannsDB exposes ip only"
                        .to_string(),
                )),
            };
        let config = hanns::faiss::HnswHvqConfig {
            dim,
            m,
            m_max0,
            ef_construction,
            ef_search,
            nbits: nbits as u8,
            metric_type,
        };
        Ok(Self {
            inner: hanns::faiss::HnswHvqIndex::new(config),
            dim,
            external_ids: Vec::new(),
        })
    }

    pub fn from_bytes(dim: usize, bytes: &[u8]) -> Result<Self, AdapterError> {
        if bytes.len() < 8 || &bytes[..8] != HNSW_HVQ_ADAPTER_MAGIC {
            return Err(AdapterError::Backend(
                "invalid hnsw_hvq adapter blob magic".to_string(),
            ));
        }
        let mut cursor = 8usize;
        let read_u64 = |bytes: &[u8], cursor: &mut usize| -> Result<u64, AdapterError> {
            let end = cursor.saturating_add(8);
            let slice = bytes.get(*cursor..end).ok_or_else(|| {
                AdapterError::Backend("truncated hnsw_hvq adapter blob".to_string())
            })?;
            *cursor = end;
            let mut buf = [0u8; 8];
            buf.copy_from_slice(slice);
            Ok(u64::from_le_bytes(buf))
        };

        let ext_len = read_u64(bytes, &mut cursor)? as usize;
        let mut external_ids = Vec::with_capacity(ext_len);
        for _ in 0..ext_len {
            external_ids.push(read_u64(bytes, &mut cursor)?);
        }
        let inner_len = read_u64(bytes, &mut cursor)? as usize;
        let inner_bytes = bytes
            .get(cursor..cursor.saturating_add(inner_len))
            .ok_or_else(|| AdapterError::Backend("truncated hnsw_hvq inner blob".to_string()))?;

        let path = temp_index_path("hnsw_hvq");
        std::fs::write(&path, inner_bytes)
            .map_err(|e| AdapterError::Backend(format!("hnsw_hvq bytes write failed: {e}")))?;
        let inner = hanns::faiss::HnswHvqIndex::load(&path)
            .map_err(|e| AdapterError::Backend(format!("hnsw_hvq deserialize failed: {e}")));
        let _ = std::fs::remove_file(&path);
        let inner = inner?;

        Ok(Self {
            inner,
            dim,
            external_ids,
        })
    }
}

#[cfg(feature = "hanns-backend")]
fn temp_index_path(prefix: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "hannsdb_index_{prefix}_{}_{}.bin",
        std::process::id(),
        HNSW_HVQ_TMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ))
}

#[cfg(feature = "hanns-backend")]
impl VectorIndexBackend for HnswHvqIndex {
    fn insert(&mut self, vectors: &[(u64, Vec<f32>)]) -> Result<(), AdapterError> {
        for (_, v) in vectors {
            if v.len() != self.dim {
                return Err(AdapterError::InvalidDimension {
                    expected: self.dim,
                    got: v.len(),
                });
            }
        }
        let n = vectors.len();
        if n == 0 {
            return Ok(());
        }
        let mut flat = Vec::with_capacity(n * self.dim);
        for (_, v) in vectors {
            flat.extend_from_slice(v);
        }
        self.inner.train(&flat, n);
        self.inner.add(&flat, n);
        self.external_ids.extend(vectors.iter().map(|(id, _)| *id));
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
        let hits = self.inner.search(query, k);
        Ok(hits
            .into_iter()
            .filter_map(|(internal_id, score)| {
                let idx = usize::try_from(internal_id).ok()?;
                let external_id = *self.external_ids.get(idx)?;
                Some(HnswSearchHit {
                    id: external_id,
                    distance: -score,
                })
            })
            .collect())
    }

    #[cfg(feature = "hanns-backend")]
    fn search_with_bitset(
        &self,
        query: &[f32],
        k: usize,
        _ef_search: usize,
        bitset: &hanns::BitsetView,
    ) -> Result<Vec<HnswSearchHit>, AdapterError> {
        if query.len() != self.dim {
            return Err(AdapterError::InvalidDimension {
                expected: self.dim,
                got: query.len(),
            });
        }
        let overfetch_k = self.external_ids.len().max(k);
        let hits = self.inner.search(query, overfetch_k);
        Ok(hits
            .into_iter()
            .filter_map(|(internal_id, score)| {
                let idx = usize::try_from(internal_id).ok()?;
                if bitset.get(idx) {
                    return None;
                }
                let external_id = *self.external_ids.get(idx)?;
                Some(HnswSearchHit {
                    id: external_id,
                    distance: -score,
                })
            })
            .take(k)
            .collect())
    }

    fn serialize_to_bytes(&self) -> Result<Option<Vec<u8>>, AdapterError> {
        let path = temp_index_path("hnsw_hvq");
        let result = self
            .inner
            .save(&path)
            .map_err(|e| AdapterError::Backend(format!("hnsw_hvq save failed: {e}")))
            .and_then(|_| {
                std::fs::read(&path)
                    .map_err(|e| AdapterError::Backend(format!("hnsw_hvq read bytes failed: {e}")))
            });
        let _ = std::fs::remove_file(&path);
        let inner_bytes = result?;

        let mut bytes = Vec::new();
        bytes.extend_from_slice(HNSW_HVQ_ADAPTER_MAGIC);
        bytes.extend_from_slice(&(self.external_ids.len() as u64).to_le_bytes());
        for id in &self.external_ids {
            bytes.extend_from_slice(&id.to_le_bytes());
        }
        bytes.extend_from_slice(&(inner_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&inner_bytes);
        Ok(Some(bytes))
    }
}

#[cfg(not(feature = "hanns-backend"))]
pub struct HnswHvqIndex;

#[cfg(not(feature = "hanns-backend"))]
impl HnswHvqIndex {
    pub fn new(
        _dim: usize,
        _metric: &str,
        _m: usize,
        _m_max0: usize,
        _ef_construction: usize,
        _ef_search: usize,
        _nbits: usize,
    ) -> Result<Self, AdapterError> {
        Err(AdapterError::Backend(
            "hnsw_hvq requires hanns-backend".to_string(),
        ))
    }

    pub fn from_bytes(_dim: usize, _bytes: &[u8]) -> Result<Self, AdapterError> {
        Err(AdapterError::Backend(
            "hnsw_hvq requires hanns-backend".to_string(),
        ))
    }
}
