use crate::adapter::{AdapterError, VectorIndexBackend};
use crate::descriptor::{
    SparseIndexDescriptor, SparseIndexKind, VectorIndexDescriptor, VectorIndexKind,
};
use crate::flat::FlatIndex;
#[cfg(not(feature = "hanns-backend"))]
use crate::hnsw::InMemoryHnswIndex;
#[cfg(feature = "hanns-backend")]
use crate::hnsw::KnowhereHnswIndex;
#[cfg(feature = "hanns-backend")]
use crate::hnsw_hvq::HnswHvqIndex;
#[cfg(feature = "hanns-backend")]
use crate::hnsw_sq::HnswSqIndex;
use crate::ivf::IvfIndex;
#[cfg(feature = "hanns-backend")]
use crate::ivf_usq::IvfUsqIndex;
#[cfg(not(feature = "hanns-backend"))]
use crate::sparse::BruteForceSparseIndex;
use crate::sparse::SparseIndexBackend;

pub trait IndexFactory {
    fn create_vector_index(
        &self,
        dim: usize,
        descriptor: &VectorIndexDescriptor,
        serialized: Option<&[u8]>,
    ) -> Result<Box<dyn VectorIndexBackend>, AdapterError>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultIndexFactory;

impl DefaultIndexFactory {
    pub fn create_vector_index(
        &self,
        dim: usize,
        descriptor: &VectorIndexDescriptor,
        serialized: Option<&[u8]>,
    ) -> Result<Box<dyn VectorIndexBackend>, AdapterError> {
        <Self as IndexFactory>::create_vector_index(self, dim, descriptor, serialized)
    }
}

impl IndexFactory for DefaultIndexFactory {
    fn create_vector_index(
        &self,
        dim: usize,
        descriptor: &VectorIndexDescriptor,
        serialized: Option<&[u8]>,
    ) -> Result<Box<dyn VectorIndexBackend>, AdapterError> {
        let metric = descriptor.metric.as_deref().unwrap_or("l2");
        match descriptor.kind {
            VectorIndexKind::Flat => Ok(Box::new(FlatIndex::new(dim, metric)?)),
            VectorIndexKind::HnswHvq => {
                #[cfg(feature = "hanns-backend")]
                {
                    let m = read_usize_param(&descriptor.params, "m").unwrap_or(16);
                    let m_max0 = read_usize_param(&descriptor.params, "m_max0").unwrap_or(m * 2);
                    let ef_construction =
                        read_usize_param(&descriptor.params, "ef_construction").unwrap_or(100);
                    let ef_search = read_usize_param(&descriptor.params, "ef_search").unwrap_or(64);
                    let nbits = read_usize_param(&descriptor.params, "nbits").unwrap_or(4);
                    if let Some(bytes) = serialized {
                        return Ok(Box::new(HnswHvqIndex::from_bytes(dim, bytes)?));
                    }
                    Ok(Box::new(HnswHvqIndex::new(
                        dim,
                        metric,
                        m,
                        m_max0,
                        ef_construction,
                        ef_search,
                        nbits,
                    )?))
                }
                #[cfg(not(feature = "hanns-backend"))]
                {
                    let _ = serialized;
                    Err(AdapterError::Backend(
                        "hnsw_hvq requires hanns-backend".to_string(),
                    ))
                }
            }
            VectorIndexKind::HnswSq => {
                #[cfg(feature = "hanns-backend")]
                {
                    let m = read_usize_param(&descriptor.params, "m").unwrap_or(16);
                    let ef_construction =
                        read_usize_param(&descriptor.params, "ef_construction").unwrap_or(200);
                    let ef_search = read_usize_param(&descriptor.params, "ef_search").unwrap_or(50);
                    if let Some(bytes) = serialized {
                        return Ok(Box::new(HnswSqIndex::from_bytes(dim, bytes)?));
                    }
                    Ok(Box::new(HnswSqIndex::new(
                        dim,
                        metric,
                        m,
                        ef_construction,
                        ef_search,
                    )?))
                }
                #[cfg(not(feature = "hanns-backend"))]
                {
                    let _ = serialized;
                    Err(AdapterError::Backend(
                        "hnsw_sq requires hanns-backend".to_string(),
                    ))
                }
            }
            VectorIndexKind::Ivf => {
                let nlist = read_usize_param(&descriptor.params, "nlist").unwrap_or(1);
                #[cfg(feature = "hanns-backend")]
                {
                    if let Some(bytes) = serialized {
                        return Ok(Box::new(IvfIndex::from_bytes(dim, metric, nlist, bytes)?));
                    }
                    Ok(Box::new(IvfIndex::new(dim, metric, nlist)?))
                }
                #[cfg(not(feature = "hanns-backend"))]
                {
                    let _ = serialized;
                    Ok(Box::new(IvfIndex::new(dim, metric, nlist)?))
                }
            }
            VectorIndexKind::IvfUsq => {
                #[cfg(feature = "hanns-backend")]
                {
                    let nlist = read_usize_param(&descriptor.params, "nlist").unwrap_or(1);
                    let bits_per_dim =
                        read_usize_param(&descriptor.params, "bits_per_dim").unwrap_or(4);
                    let rotation_seed =
                        read_usize_param(&descriptor.params, "rotation_seed").unwrap_or(42);
                    let rerank_k = read_usize_param(&descriptor.params, "rerank_k").unwrap_or(64);
                    let use_high_accuracy_scan = descriptor
                        .params
                        .get("use_high_accuracy_scan")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false);
                    if let Some(bytes) = serialized {
                        return Ok(Box::new(IvfUsqIndex::from_bytes(
                            dim,
                            metric,
                            nlist,
                            bits_per_dim,
                            rotation_seed,
                            rerank_k,
                            use_high_accuracy_scan,
                            bytes,
                        )?));
                    }
                    Ok(Box::new(IvfUsqIndex::new(
                        dim,
                        metric,
                        nlist,
                        bits_per_dim,
                        rotation_seed,
                        rerank_k,
                        use_high_accuracy_scan,
                    )?))
                }
                #[cfg(not(feature = "hanns-backend"))]
                {
                    let _ = serialized;
                    Err(AdapterError::Backend(
                        "ivf_usq requires hanns-backend".to_string(),
                    ))
                }
            }
            VectorIndexKind::Hnsw => {
                #[cfg(feature = "hanns-backend")]
                {
                    if let Some(bytes) = serialized {
                        return Ok(Box::new(KnowhereHnswIndex::from_bytes(dim, bytes)?));
                    }
                    Ok(Box::new(KnowhereHnswIndex::new(
                        dim,
                        metric,
                        read_usize_param(&descriptor.params, "m").unwrap_or(16),
                        read_usize_param(&descriptor.params, "ef_construction").unwrap_or(128),
                    )?))
                }

                #[cfg(not(feature = "hanns-backend"))]
                {
                    let _ = serialized;
                    Ok(Box::new(InMemoryHnswIndex::new(dim, metric)?))
                }
            }
        }
    }
}

fn read_usize_param(value: &serde_json::Value, key: &str) -> Option<usize> {
    value
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

impl DefaultIndexFactory {
    pub fn create_sparse_index(
        &self,
        descriptor: &SparseIndexDescriptor,
        serialized: Option<&[u8]>,
    ) -> Result<Box<dyn SparseIndexBackend>, AdapterError> {
        let metric = descriptor.metric.as_deref().unwrap_or("ip");
        match descriptor.kind {
            SparseIndexKind::SparseInverted => {
                #[cfg(feature = "hanns-backend")]
                {
                    if let Some(bytes) = serialized {
                        return Ok(Box::new(
                            crate::sparse::HannsSparseInvertedIndex::from_bytes(bytes)?,
                        ));
                    }
                    Ok(Box::new(crate::sparse::HannsSparseInvertedIndex::new(
                        metric,
                    )?))
                }
                #[cfg(not(feature = "hanns-backend"))]
                {
                    let _ = (serialized, metric);
                    Ok(Box::new(BruteForceSparseIndex::new()))
                }
            }
            SparseIndexKind::SparseWand => {
                #[cfg(feature = "hanns-backend")]
                {
                    if let Some(bytes) = serialized {
                        return Ok(Box::new(crate::sparse::HannsSparseWandIndex::from_bytes(
                            bytes,
                        )?));
                    }
                    Ok(Box::new(crate::sparse::HannsSparseWandIndex::new(metric)?))
                }
                #[cfg(not(feature = "hanns-backend"))]
                {
                    let _ = (serialized, metric);
                    Ok(Box::new(BruteForceSparseIndex::new()))
                }
            }
        }
    }
}
