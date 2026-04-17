use std::io;
use std::path::Path;
#[cfg(feature = "hanns-backend")]
use std::path::PathBuf;

use crate::document::{CollectionSchema, Document};
use crate::query::SearchHit;
use crate::storage::lance_store::LanceCollection;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageBackend {
    Lance,
}

pub enum StorageCollection {
    Lance(LanceCollection),
}

impl StorageBackend {
    pub async fn create_collection(
        self,
        root: impl AsRef<Path>,
        name: impl Into<String>,
        schema: CollectionSchema,
        documents: &[Document],
    ) -> io::Result<StorageCollection> {
        match self {
            StorageBackend::Lance => LanceCollection::create(root, name, schema, documents)
                .await
                .map(StorageCollection::Lance),
        }
    }

    pub async fn open_collection(
        self,
        root: impl AsRef<Path>,
        name: impl Into<String>,
        schema: CollectionSchema,
    ) -> io::Result<StorageCollection> {
        match self {
            StorageBackend::Lance => LanceCollection::open(root, name, schema)
                .await
                .map(StorageCollection::Lance),
        }
    }

    pub async fn open_collection_inferred(
        self,
        root: impl AsRef<Path>,
        name: impl Into<String>,
    ) -> io::Result<StorageCollection> {
        match self {
            StorageBackend::Lance => LanceCollection::open_inferred(root, name)
                .await
                .map(StorageCollection::Lance),
        }
    }
}

impl StorageCollection {
    pub fn name(&self) -> &str {
        match self {
            StorageCollection::Lance(collection) => collection.name(),
        }
    }

    pub fn uri(&self) -> &str {
        match self {
            StorageCollection::Lance(collection) => collection.uri(),
        }
    }

    pub fn schema(&self) -> &CollectionSchema {
        match self {
            StorageCollection::Lance(collection) => collection.schema(),
        }
    }

    #[cfg(feature = "hanns-backend")]
    pub fn hanns_index_path(&self, field_name: &str) -> PathBuf {
        match self {
            StorageCollection::Lance(collection) => collection.hanns_index_path(field_name),
        }
    }

    #[cfg(feature = "hanns-backend")]
    pub async fn optimize_hanns(&self, field_name: &str, metric: &str) -> io::Result<()> {
        match self {
            StorageCollection::Lance(collection) => {
                collection.optimize_hanns(field_name, metric).await
            }
        }
    }

    pub async fn insert_documents(&self, documents: &[Document]) -> io::Result<()> {
        match self {
            StorageCollection::Lance(collection) => collection.insert_documents(documents).await,
        }
    }

    pub async fn delete_documents(&self, ids: &[i64]) -> io::Result<usize> {
        match self {
            StorageCollection::Lance(collection) => collection.delete_documents(ids).await,
        }
    }

    pub async fn upsert_documents(&self, documents: &[Document]) -> io::Result<usize> {
        match self {
            StorageCollection::Lance(collection) => collection.upsert_documents(documents).await,
        }
    }

    pub async fn fetch_documents(&self, ids: &[i64]) -> io::Result<Vec<Document>> {
        match self {
            StorageCollection::Lance(collection) => collection.fetch_documents(ids).await,
        }
    }

    pub async fn search(
        &self,
        query: &[f32],
        top_k: usize,
        metric: &str,
    ) -> io::Result<Vec<SearchHit>> {
        match self {
            StorageCollection::Lance(collection) => collection.search(query, top_k, metric).await,
        }
    }
}
