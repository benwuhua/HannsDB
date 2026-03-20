use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Deserialize)]
pub struct CreateCollectionRequest {
    pub name: String,
    pub dimension: usize,
    pub metric: String,
}

#[derive(Debug, Serialize)]
pub struct CreateCollectionResponse {
    pub name: String,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
}

impl HealthResponse {
    pub fn ok() -> Self {
        Self { status: "ok" }
    }
}

#[derive(Debug, Deserialize)]
pub struct InsertRecordsRequest {
    pub ids: Vec<String>,
    pub vectors: Vec<Vec<f32>>,
    #[serde(default)]
    pub fields: Vec<BTreeMap<String, Value>>,
}

#[derive(Debug, Serialize)]
pub struct InsertRecordsResponse {
    pub inserted: u64,
}

#[derive(Debug, Serialize)]
pub struct UpsertRecordsResponse {
    pub upserted: u64,
}

#[derive(Debug, Deserialize)]
pub struct DeleteRecordsRequest {
    pub ids: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct DeleteRecordsResponse {
    pub deleted: u64,
}

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub vector: Vec<f32>,
    pub top_k: usize,
    #[serde(default)]
    pub output_fields: Option<Vec<String>>,
    pub filter: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SearchHitResponse {
    pub id: String,
    pub distance: f32,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub fields: BTreeMap<String, Value>,
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub hits: Vec<SearchHitResponse>,
}

#[derive(Debug, Deserialize)]
pub struct FetchRecordsRequest {
    pub ids: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct FetchRecordResponse {
    pub id: String,
    pub fields: BTreeMap<String, Value>,
    pub vector: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct FetchRecordsResponse {
    pub documents: Vec<FetchRecordResponse>,
}

#[derive(Debug, Serialize)]
pub struct ListCollectionsResponse {
    pub collections: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct DropCollectionResponse {
    pub dropped: String,
}

#[derive(Debug, Serialize)]
pub struct FlushCollectionResponse {
    pub flushed: String,
}

#[derive(Debug, Serialize)]
pub struct CollectionInfoResponse {
    pub name: String,
    pub dimension: usize,
    pub metric: String,
    pub record_count: usize,
    pub deleted_count: usize,
    pub live_count: usize,
}
