#[derive(Debug, Clone, Copy)]
pub(crate) enum MetricKind {
    L2,
    Cosine,
    Ip,
}

impl MetricKind {
    pub(crate) fn parse(metric: &str) -> Result<Self, AdapterError> {
        match metric.to_ascii_lowercase().as_str() {
            "l2" => Ok(Self::L2),
            "cosine" => Ok(Self::Cosine),
            "ip" => Ok(Self::Ip),
            other => Err(AdapterError::Backend(format!(
                "unsupported metric: {other}"
            ))),
        }
    }

    pub(crate) fn distance(self, query: &[f32], vector: &[f32]) -> f32 {
        match self {
            Self::L2 => l2_sq(query, vector).sqrt(),
            Self::Cosine => cosine_distance(query, vector),
            Self::Ip => -dot(query, vector),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdapterError {
    InvalidDimension { expected: usize, got: usize },
    EmptyInsert,
    Backend(String),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HnswSearchHit {
    pub id: u64,
    pub distance: f32,
}

pub trait VectorIndexBackend: Send + Sync {
    fn insert(&mut self, vectors: &[(u64, Vec<f32>)]) -> Result<(), AdapterError>;
    fn insert_flat(
        &mut self,
        ids: &[u64],
        vectors: &[f32],
        dim: usize,
    ) -> Result<(), AdapterError> {
        if dim == 0 {
            return Err(AdapterError::Backend("dimension must be > 0".to_string()));
        }
        if vectors.len() % dim != 0 {
            return Err(AdapterError::InvalidDimension {
                expected: dim,
                got: vectors.len() % dim,
            });
        }
        let count = vectors.len() / dim;
        if ids.len() != count {
            return Err(AdapterError::Backend(format!(
                "id count mismatch for flat insert: ids={}, vectors={count}",
                ids.len()
            )));
        }
        let tuples = ids
            .iter()
            .enumerate()
            .map(|(i, id)| {
                let start = i * dim;
                let end = start + dim;
                (*id, vectors[start..end].to_vec())
            })
            .collect::<Vec<_>>();
        self.insert(&tuples)
    }

    fn insert_flat_identity(&mut self, vectors: &[f32], dim: usize) -> Result<(), AdapterError> {
        if dim == 0 {
            return Err(AdapterError::Backend("dimension must be > 0".to_string()));
        }
        if vectors.len() % dim != 0 {
            return Err(AdapterError::InvalidDimension {
                expected: dim,
                got: vectors.len() % dim,
            });
        }
        let count = vectors.len() / dim;
        let ids = (0..count as u64).collect::<Vec<_>>();
        self.insert_flat(&ids, vectors, dim)
    }
    fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<HnswSearchHit>, AdapterError>;

    fn search_into(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        ids_out: &mut [i64],
        dists_out: &mut [f32],
    ) -> Result<usize, AdapterError> {
        let hits = self.search(query, k, ef_search)?;
        let n = hits.len().min(k).min(ids_out.len()).min(dists_out.len());
        for (i, h) in hits.iter().take(n).enumerate() {
            ids_out[i] = h.id as i64;
            dists_out[i] = h.distance;
        }
        Ok(n)
    }

    fn serialize_to_bytes(&self) -> Result<Option<Vec<u8>>, AdapterError> {
        Ok(None)
    }

    /// Search with a bitset filter applied during search (pre-filter).
    ///
    /// `bitset` semantics: bit=1 means **filtered OUT** (excluded), bit=0 means **kept**.
    ///
    /// The default implementation falls back to post-filter: it runs a normal
    /// `search` and then removes any hit whose `id` index has the bitset bit set.
    /// Backends that support native pre-filtered search should override this
    /// for better performance.
    #[cfg(feature = "hanns-backend")]
    fn search_with_bitset(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        bitset: &hanns::BitsetView,
    ) -> Result<Vec<HnswSearchHit>, AdapterError> {
        // Default: search then post-filter
        let hits = self.search(query, k, ef_search)?;
        Ok(hits
            .into_iter()
            .filter(|h| !bitset.get(h.id as usize))
            .collect())
    }
}

pub trait HnswBackend: VectorIndexBackend {}

impl<T> HnswBackend for T where T: VectorIndexBackend + ?Sized {}

pub struct HnswAdapter<B> {
    backend: B,
}

impl<B> HnswAdapter<B>
where
    B: HnswBackend,
{
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    pub fn insert(&mut self, vectors: Vec<(u64, Vec<f32>)>) -> Result<(), AdapterError> {
        if vectors.is_empty() {
            return Err(AdapterError::EmptyInsert);
        }
        self.backend.insert(&vectors)
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<HnswSearchHit>, AdapterError> {
        self.search_with_ef(query, k, 32)
    }

    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<HnswSearchHit>, AdapterError> {
        self.backend.search(query, k, ef_search)
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
