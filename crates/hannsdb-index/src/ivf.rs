use crate::adapter::{AdapterError, HnswSearchHit, VectorIndexBackend};
use crate::flat::FlatIndex;

pub struct IvfIndex {
    inner: FlatIndex,
    #[allow(dead_code)]
    nlist: usize,
}

impl IvfIndex {
    pub fn new(dim: usize, metric: &str, nlist: usize) -> Result<Self, AdapterError> {
        Ok(Self {
            inner: FlatIndex::new(dim, metric)?,
            nlist: nlist.max(1),
        })
    }
}

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
