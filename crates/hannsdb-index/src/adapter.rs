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

pub trait HnswBackend {
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
}

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
