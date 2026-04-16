use crate::adapter::AdapterError;

#[cfg(feature = "hanns-backend")]
use crate::adapter::{HnswSearchHit, MetricKind, VectorIndexBackend};

#[cfg(feature = "hanns-backend")]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "hanns-backend")]
static HNSW_SQ_TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "hanns-backend")]
const HNSW_SQ_ADAPTER_MAGIC: &[u8; 8] = b"HDBHSQ00";

// ---------------------------------------------------------------------------
// hanns-backend: real HNSW-SQ backed by hanns::faiss::HnswSqIndex
// ---------------------------------------------------------------------------

#[cfg(feature = "hanns-backend")]
pub struct HnswSqIndex {
    inner: hanns::faiss::HnswSqIndex,
    dim: usize,
    /// External IDs in insertion order (index == positional row in the inner index).
    /// Used for bitset-filtered search where we need to skip rows by position.
    external_ids: Vec<i64>,
}

#[cfg(feature = "hanns-backend")]
impl HnswSqIndex {
    pub fn new(
        dim: usize,
        metric: &str,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
    ) -> Result<Self, AdapterError> {
        // Validate metric — HnswSq supports all metrics (l2, ip, cosine).
        let _metric_kind = MetricKind::parse(metric)?;
        let inner = hanns::faiss::HnswSqIndex::new_with_config(
            dim,
            hanns::faiss::hnsw_quantized::HnswQuantizeConfig {
                use_pq: false,
                pq_m: 8,
                pq_k: 256,
                sq_bit: 8,
                ef_search,
                ef_construction,
                max_neighbors: m,
            },
        );
        Ok(Self {
            inner,
            dim,
            external_ids: Vec::new(),
        })
    }

    pub fn from_bytes(dim: usize, bytes: &[u8]) -> Result<Self, AdapterError> {
        if bytes.len() < 8 || &bytes[..8] != HNSW_SQ_ADAPTER_MAGIC {
            return Err(AdapterError::Backend(
                "invalid hnsw_sq adapter blob magic".to_string(),
            ));
        }
        let mut cursor = 8usize;
        let read_u64 = |bytes: &[u8], cursor: &mut usize| -> Result<u64, AdapterError> {
            let end = cursor.saturating_add(8);
            let slice = bytes.get(*cursor..end).ok_or_else(|| {
                AdapterError::Backend("truncated hnsw_sq adapter blob".to_string())
            })?;
            *cursor = end;
            let mut buf = [0u8; 8];
            buf.copy_from_slice(slice);
            Ok(u64::from_le_bytes(buf))
        };
        let read_i64 = |bytes: &[u8], cursor: &mut usize| -> Result<i64, AdapterError> {
            read_u64(bytes, cursor).map(|v| v as i64)
        };

        let ext_len = read_u64(bytes, &mut cursor)? as usize;
        let mut external_ids = Vec::with_capacity(ext_len);
        for _ in 0..ext_len {
            external_ids.push(read_i64(bytes, &mut cursor)?);
        }
        let inner_len = read_u64(bytes, &mut cursor)? as usize;
        let inner_bytes = bytes
            .get(cursor..cursor.saturating_add(inner_len))
            .ok_or_else(|| AdapterError::Backend("truncated hnsw_sq inner blob".to_string()))?;

        let path = temp_index_path("hnsw_sq");
        std::fs::write(&path, inner_bytes)
            .map_err(|e| AdapterError::Backend(format!("hnsw_sq bytes write failed: {e}")))?;
        let inner = hanns::faiss::HnswSqIndex::load(&path, dim)
            .map_err(|e| AdapterError::Backend(format!("hnsw_sq deserialize failed: {e}")));
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
        HNSW_SQ_TMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ))
}

#[cfg(feature = "hanns-backend")]
impl VectorIndexBackend for HnswSqIndex {
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
        // HnswSqIndex uses i64 IDs; cast u64 external IDs to i64.
        let ids: Vec<i64> = vectors.iter().map(|(id, _)| *id as i64).collect();
        self.inner
            .train(&flat)
            .map_err(|e| AdapterError::Backend(format!("hnsw_sq train failed: {e}")))?;
        self.inner
            .add(&flat, Some(&ids))
            .map_err(|e| AdapterError::Backend(format!("hnsw_sq add failed: {e}")))?;
        self.external_ids.extend(ids.iter().copied());
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
        let req = hanns::api::SearchRequest {
            top_k: k,
            nprobe: if ef_search > 0 { ef_search } else { self.inner.ef_search() },
            filter: None,
            params: None,
            radius: None,
        };
        let result = self
            .inner
            .search(query, &req)
            .map_err(|e| AdapterError::Backend(format!("hnsw_sq search failed: {e}")))?;
        Ok(result
            .ids
            .into_iter()
            .zip(result.distances.into_iter())
            .filter(|&(id, _)| id >= 0)
            .map(|(id, dist)| HnswSearchHit {
                id: id as u64,
                distance: dist,
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
        // Over-fetch then filter by bitset using positional index.
        let overfetch_k = self.external_ids.len().max(k);
        let req = hanns::api::SearchRequest {
            top_k: overfetch_k,
            nprobe: if ef_search > 0 { ef_search } else { self.inner.ef_search() },
            filter: None,
            params: None,
            radius: None,
        };
        let result = self.inner.search(query, &req).map_err(|e| {
            AdapterError::Backend(format!("hnsw_sq search_with_bitset failed: {e}"))
        })?;

        // Map returned IDs back to positional indices via external_ids lookup.
        Ok(result
            .ids
            .into_iter()
            .zip(result.distances.into_iter())
            .filter(|&(id, _)| id >= 0)
            .filter_map(|(id, dist)| {
                // Find the position of this id in external_ids to check the bitset.
                let pos = self.external_ids.iter().position(|&eid| eid == id)?;
                if bitset.get(pos) {
                    return None;
                }
                Some(HnswSearchHit {
                    id: id as u64,
                    distance: dist,
                })
            })
            .take(k)
            .collect())
    }

    fn serialize_to_bytes(&self) -> Result<Option<Vec<u8>>, AdapterError> {
        let path = temp_index_path("hnsw_sq");
        let result = self
            .inner
            .save(&path)
            .map_err(|e| AdapterError::Backend(format!("hnsw_sq save failed: {e}")))
            .and_then(|_| {
                std::fs::read(&path)
                    .map_err(|e| AdapterError::Backend(format!("hnsw_sq read bytes failed: {e}")))
            });
        let _ = std::fs::remove_file(&path);
        let inner_bytes = result?;

        let mut bytes = Vec::new();
        bytes.extend_from_slice(HNSW_SQ_ADAPTER_MAGIC);
        bytes.extend_from_slice(&(self.external_ids.len() as u64).to_le_bytes());
        for id in &self.external_ids {
            bytes.extend_from_slice(&(*id as u64).to_le_bytes());
        }
        bytes.extend_from_slice(&(inner_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&inner_bytes);
        Ok(Some(bytes))
    }
}

#[cfg(not(feature = "hanns-backend"))]
pub struct HnswSqIndex;

#[cfg(not(feature = "hanns-backend"))]
impl HnswSqIndex {
    pub fn new(
        _dim: usize,
        _metric: &str,
        _m: usize,
        _ef_construction: usize,
        _ef_search: usize,
    ) -> Result<Self, AdapterError> {
        Err(AdapterError::Backend(
            "hnsw_sq requires hanns-backend".to_string(),
        ))
    }

    pub fn from_bytes(_dim: usize, _bytes: &[u8]) -> Result<Self, AdapterError> {
        Err(AdapterError::Backend(
            "hnsw_sq requires hanns-backend".to_string(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_sq_stub_returns_error() {
        let result = HnswSqIndex::new(4, "l2", 16, 200, 50);
        // Without hanns-backend, must return an error.
        #[cfg(not(feature = "hanns-backend"))]
        assert!(result.is_err());
        // With hanns-backend, construction should succeed.
        #[cfg(feature = "hanns-backend")]
        assert!(result.is_ok());
    }

    #[cfg(feature = "hanns-backend")]
    #[test]
    fn test_hnsw_sq_basic() {
        use crate::adapter::VectorIndexBackend;
        use rand::{rngs::StdRng, Rng, SeedableRng};

        let dim = 16usize;
        let n = 500usize;
        let mut rng = StdRng::seed_from_u64(42);
        let raw: Vec<f32> = (0..n * dim).map(|_| rng.gen::<f32>()).collect();

        let mut index = HnswSqIndex::new(dim, "l2", 16, 200, 50).unwrap();
        let batch: Vec<(u64, Vec<f32>)> = (0..n)
            .map(|i| (i as u64, raw[i * dim..(i + 1) * dim].to_vec()))
            .collect();
        index.insert(&batch).unwrap();

        let query = raw[0..dim].to_vec();
        let hits = index.search(&query, 10, 50).unwrap();
        assert!(!hits.is_empty(), "search must return at least one hit");
        assert!(hits.len() <= 10, "search must return at most k hits");
    }

    #[cfg(feature = "hanns-backend")]
    #[test]
    fn test_hnsw_sq_recall() {
        use crate::adapter::VectorIndexBackend;
        use crate::flat::FlatIndex;
        use rand::{rngs::StdRng, Rng, SeedableRng};

        let dim = 16usize;
        let n = 500usize;
        let k = 10usize;
        let mut rng = StdRng::seed_from_u64(7);
        let raw: Vec<f32> = (0..n * dim).map(|_| rng.gen::<f32>()).collect();

        let mut hnsw_sq = HnswSqIndex::new(dim, "l2", 16, 200, 50).unwrap();
        let mut flat = FlatIndex::new(dim, "l2").unwrap();
        let batch: Vec<(u64, Vec<f32>)> = (0..n)
            .map(|i| (i as u64, raw[i * dim..(i + 1) * dim].to_vec()))
            .collect();
        hnsw_sq.insert(&batch).unwrap();
        flat.insert(&batch).unwrap();

        let query = raw[0..dim].to_vec();
        let sq_hits = hnsw_sq.search(&query, k, 50).unwrap();
        let flat_hits = flat.search(&query, k, 50).unwrap();

        let flat_ids: std::collections::HashSet<u64> = flat_hits.iter().map(|h| h.id).collect();
        let overlap = sq_hits.iter().filter(|h| flat_ids.contains(&h.id)).count();
        let recall = overlap as f32 / k as f32;
        assert!(
            recall >= 0.5,
            "hnsw_sq recall {recall:.2} below threshold 0.5"
        );
    }

    #[cfg(feature = "hanns-backend")]
    #[test]
    fn test_hnsw_sq_save_load_roundtrip() {
        use crate::adapter::VectorIndexBackend;
        use rand::{rngs::StdRng, Rng, SeedableRng};

        let dim = 16usize;
        let n = 300usize;
        let mut rng = StdRng::seed_from_u64(13);
        let raw: Vec<f32> = (0..n * dim).map(|_| rng.gen::<f32>()).collect();

        let mut index = HnswSqIndex::new(dim, "l2", 16, 200, 50).unwrap();
        let batch: Vec<(u64, Vec<f32>)> = (0..n)
            .map(|i| (i as u64, raw[i * dim..(i + 1) * dim].to_vec()))
            .collect();
        index.insert(&batch).unwrap();

        let bytes = index.serialize_to_bytes().unwrap().unwrap();
        let loaded = HnswSqIndex::from_bytes(dim, &bytes).unwrap();

        let query = raw[0..dim].to_vec();
        let orig_hits = index.search(&query, 5, 50).unwrap();
        let loaded_hits = loaded.search(&query, 5, 50).unwrap();

        assert_eq!(
            orig_hits.len(),
            loaded_hits.len(),
            "hit count must match after roundtrip"
        );
        let orig_ids: std::collections::HashSet<u64> = orig_hits.iter().map(|h| h.id).collect();
        let loaded_ids: std::collections::HashSet<u64> = loaded_hits.iter().map(|h| h.id).collect();
        assert_eq!(orig_ids, loaded_ids, "hit IDs must match after roundtrip");
    }

    #[cfg(feature = "hanns-backend")]
    #[test]
    fn test_hnsw_sq_metric_variants() {
        let dim = 8usize;
        for metric in &["l2", "ip", "cosine"] {
            let result = HnswSqIndex::new(dim, metric, 16, 200, 50);
            assert!(result.is_ok(), "HnswSqIndex must accept metric '{metric}'");
        }
    }

    #[test]
    fn test_hnsw_sq_ef_search_override_compiles() {
        // Without hanns-backend, ef_search override is not tested functionally,
        // but the stub must still compile with the renamed parameter.
        let result = HnswSqIndex::new(4, "l2", 16, 200, 50);
        #[cfg(not(feature = "hanns-backend"))]
        assert!(result.is_err());
        #[cfg(feature = "hanns-backend")]
        {
            use crate::adapter::VectorIndexBackend;
            let mut index = result.unwrap();
            use rand::{rngs::StdRng, Rng, SeedableRng};
            let dim = 4usize;
            let n = 100usize;
            let mut rng = StdRng::seed_from_u64(99);
            let raw: Vec<f32> = (0..n * dim).map(|_| rng.gen::<f32>()).collect();
            let batch: Vec<(u64, Vec<f32>)> = (0..n)
                .map(|i| (i as u64, raw[i * dim..(i + 1) * dim].to_vec()))
                .collect();
            index.insert(&batch).unwrap();
            let query = raw[0..dim].to_vec();
            // ef_search=1 (low quality but valid)
            let hits_low = index.search(&query, 5, 1).unwrap();
            // ef_search=200 (high quality)
            let hits_high = index.search(&query, 5, 200).unwrap();
            assert!(!hits_low.is_empty());
            assert!(!hits_high.is_empty());
        }
    }
}
