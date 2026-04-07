use crate::adapter::{AdapterError, VectorIndexBackend};
use crate::descriptor::{VectorIndexDescriptor, VectorIndexKind};
use crate::flat::FlatIndex;
use crate::hnsw::InMemoryHnswIndex;
#[cfg(feature = "knowhere-backend")]
use crate::hnsw::KnowhereHnswIndex;
use crate::ivf::IvfIndex;

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
            VectorIndexKind::Ivf => Ok(Box::new(IvfIndex::new(
                dim,
                metric,
                read_usize_param(&descriptor.params, "nlist").unwrap_or(1),
            )?)),
            VectorIndexKind::Hnsw => {
                #[cfg(feature = "knowhere-backend")]
                {
                    if let Some(bytes) = serialized {
                        return Ok(Box::new(KnowhereHnswIndex::from_bytes(dim, bytes)?));
                    }
                    return Ok(Box::new(KnowhereHnswIndex::new(
                        dim,
                        metric,
                        read_usize_param(&descriptor.params, "m").unwrap_or(16),
                        read_usize_param(&descriptor.params, "ef_construction").unwrap_or(128),
                    )?));
                }

                #[cfg(not(feature = "knowhere-backend"))]
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
