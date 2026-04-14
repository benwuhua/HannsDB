//! Sparse index backend trait and implementations.
//!
//! Parallel to `VectorIndexBackend` but for sparse vectors.
//! Uses `SparseVectorData` (parallel arrays of indices/values) instead of `&[f32]`.

use crate::adapter::AdapterError;

/// Sparse vector in parallel-array form (same layout as HannsDB core's SparseVector).
#[derive(Debug, Clone)]
pub struct SparseVectorData {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

impl SparseVectorData {
    pub fn new(indices: Vec<u32>, values: Vec<f32>) -> Self {
        Self { indices, values }
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// A hit from sparse index search.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseSearchHit {
    pub id: i64,
    pub score: f32, // positive score (higher = more relevant), caller negates for distance
}

/// Trait for sparse index backends.
pub trait SparseIndexBackend: Send + Sync {
    /// Add sparse vectors with their external IDs.
    fn add(&mut self, vectors: &[(i64, SparseVectorData)]) -> Result<(), AdapterError>;

    /// Search for top-k results. Returns hits sorted by score descending.
    /// `bitset` semantics: bit=1 means excluded, bit=0 means kept.
    fn search(
        &self,
        query: &SparseVectorData,
        k: usize,
        bitset: Option<&[bool]>,
    ) -> Result<Vec<SparseSearchHit>, AdapterError>;

    /// Serialize the index to bytes for persistence.
    fn serialize_to_bytes(&self) -> Result<Option<Vec<u8>>, AdapterError> {
        Ok(None)
    }

    /// Number of indexed vectors.
    fn len(&self) -> usize;

    /// Whether the backend currently holds zero indexed vectors.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Set BM25 scoring parameters on the underlying index.
    ///
    /// Default implementation is a no-op so that brute-force backends
    /// (which don't use BM25) don't need to override this.
    fn set_bm25_params(&mut self, _k1: f32, _b: f32, _avgdl: f32) {}
}

// ---------------------------------------------------------------------------
// Brute-force fallback (no hanns-backend feature)
// ---------------------------------------------------------------------------

pub struct BruteForceSparseIndex {
    vectors: Vec<(i64, SparseVectorData)>,
}

impl BruteForceSparseIndex {
    pub fn new() -> Self {
        Self {
            vectors: Vec::new(),
        }
    }
}

impl Default for BruteForceSparseIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseIndexBackend for BruteForceSparseIndex {
    fn add(&mut self, vectors: &[(i64, SparseVectorData)]) -> Result<(), AdapterError> {
        self.vectors
            .extend(vectors.iter().map(|(id, v)| (*id, v.clone())));
        Ok(())
    }

    fn search(
        &self,
        query: &SparseVectorData,
        k: usize,
        bitset: Option<&[bool]>,
    ) -> Result<Vec<SparseSearchHit>, AdapterError> {
        let mut hits: Vec<SparseSearchHit> = Vec::new();
        for (idx, (id, vector)) in self.vectors.iter().enumerate() {
            if let Some(bs) = bitset {
                if idx < bs.len() && bs[idx] {
                    continue;
                }
            }
            let score = sparse_ip(query, vector);
            if score > 0.0 {
                hits.push(SparseSearchHit { id: *id, score });
            }
        }
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        hits.truncate(k);
        Ok(hits)
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }
}

/// Sparse inner product using two-pointer merge.
fn sparse_ip(a: &SparseVectorData, b: &SparseVectorData) -> f32 {
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

// ---------------------------------------------------------------------------
// hanns-backend: wrappers around hanns SparseInvertedIndex / SparseWandIndex
// ---------------------------------------------------------------------------

#[cfg(feature = "hanns-backend")]
pub struct HannsSparseInvertedIndex {
    inner: hanns::faiss::SparseInvertedIndex,
}

#[cfg(feature = "hanns-backend")]
impl HannsSparseInvertedIndex {
    pub fn new(metric: &str) -> Result<Self, AdapterError> {
        let metric_type = parse_sparse_metric(metric);
        Ok(Self {
            inner: hanns::faiss::SparseInvertedIndex::new(metric_type),
        })
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, AdapterError> {
        let inner = hanns::faiss::SparseInvertedIndex::deserialize_from_bytes(bytes)
            .map_err(|e| AdapterError::Backend(format!("sparse index deserialize: {e:?}")))?;
        Ok(Self { inner })
    }

    pub fn set_bm25_params(&mut self, k1: f32, b: f32, avgdl: f32) {
        self.inner.set_bm25_params(k1, b, avgdl);
    }
}

#[cfg(feature = "hanns-backend")]
impl SparseIndexBackend for HannsSparseInvertedIndex {
    fn add(&mut self, vectors: &[(i64, SparseVectorData)]) -> Result<(), AdapterError> {
        for (id, v) in vectors {
            let hanns_vec = convert_to_hanns_sparse(v);
            self.inner
                .add(&hanns_vec, *id)
                .map_err(|e| AdapterError::Backend(format!("sparse index add: {e}")))?;
        }
        Ok(())
    }

    fn search(
        &self,
        query: &SparseVectorData,
        k: usize,
        bitset: Option<&[bool]>,
    ) -> Result<Vec<SparseSearchHit>, AdapterError> {
        let hanns_query = convert_to_hanns_sparse(query);
        let results = self.inner.search(&hanns_query, k, bitset);
        Ok(results
            .into_iter()
            .map(|(id, score)| SparseSearchHit { id, score })
            .collect())
    }

    fn serialize_to_bytes(&self) -> Result<Option<Vec<u8>>, AdapterError> {
        let bytes = self
            .inner
            .serialize_to_bytes()
            .map_err(|e| AdapterError::Backend(format!("sparse index serialize: {e:?}")))?;
        Ok(Some(bytes))
    }

    fn len(&self) -> usize {
        self.inner.n_rows()
    }

    fn set_bm25_params(&mut self, k1: f32, b: f32, avgdl: f32) {
        self.inner.set_bm25_params(k1, b, avgdl);
    }
}

#[cfg(feature = "hanns-backend")]
pub struct HannsSparseWandIndex {
    inner: hanns::faiss::SparseWandIndex,
}

#[cfg(feature = "hanns-backend")]
impl HannsSparseWandIndex {
    pub fn new(metric: &str) -> Result<Self, AdapterError> {
        let metric_type = parse_sparse_metric(metric);
        Ok(Self {
            inner: hanns::faiss::SparseWandIndex::new(metric_type),
        })
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, AdapterError> {
        let inner = hanns::faiss::SparseWandIndex::deserialize_from_bytes(bytes)
            .map_err(|e| AdapterError::Backend(format!("sparse wand deserialize: {e:?}")))?;
        Ok(Self { inner })
    }

    pub fn set_bm25_params(&mut self, k1: f32, b: f32, avgdl: f32) {
        self.inner.set_bm25_params(k1, b, avgdl);
    }
}

#[cfg(feature = "hanns-backend")]
impl SparseIndexBackend for HannsSparseWandIndex {
    fn add(&mut self, vectors: &[(i64, SparseVectorData)]) -> Result<(), AdapterError> {
        for (id, v) in vectors {
            let hanns_vec = convert_to_hanns_sparse(v);
            self.inner
                .add(&hanns_vec, *id)
                .map_err(|e| AdapterError::Backend(format!("sparse wand add: {e}")))?;
        }
        Ok(())
    }

    fn search(
        &self,
        query: &SparseVectorData,
        k: usize,
        bitset: Option<&[bool]>,
    ) -> Result<Vec<SparseSearchHit>, AdapterError> {
        let hanns_query = convert_to_hanns_sparse(query);
        let results = self.inner.search(&hanns_query, k, bitset);
        Ok(results
            .into_iter()
            .map(|(id, score)| SparseSearchHit { id, score })
            .collect())
    }

    fn serialize_to_bytes(&self) -> Result<Option<Vec<u8>>, AdapterError> {
        let bytes = self
            .inner
            .serialize_to_bytes()
            .map_err(|e| AdapterError::Backend(format!("sparse wand serialize: {e:?}")))?;
        Ok(Some(bytes))
    }

    fn len(&self) -> usize {
        self.inner.n_rows()
    }

    fn set_bm25_params(&mut self, k1: f32, b: f32, avgdl: f32) {
        self.inner.set_bm25_params(k1, b, avgdl);
    }
}

#[cfg(feature = "hanns-backend")]
fn convert_to_hanns_sparse(v: &SparseVectorData) -> hanns::faiss::sparse_inverted::SparseVector {
    hanns::faiss::sparse_inverted::SparseVector::from_pairs(
        &v.indices
            .iter()
            .zip(v.values.iter())
            .map(|(&dim, &val)| (dim, val))
            .collect::<Vec<_>>(),
    )
}

#[cfg(feature = "hanns-backend")]
fn parse_sparse_metric(metric: &str) -> hanns::faiss::sparse_inverted::SparseMetricType {
    match metric.to_ascii_lowercase().as_str() {
        "bm25" => hanns::faiss::sparse_inverted::SparseMetricType::Bm25,
        _ => hanns::faiss::sparse_inverted::SparseMetricType::Ip,
    }
}
