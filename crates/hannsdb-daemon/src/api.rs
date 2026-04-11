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
#[serde(deny_unknown_fields)]
pub struct SparseVectorRequest {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

#[derive(Debug, Deserialize)]
pub struct InsertRecordsRequest {
    pub ids: Vec<String>,
    #[serde(default)]
    pub vectors: Vec<Vec<f32>>,
    #[serde(default)]
    pub fields: Vec<BTreeMap<String, Value>>,
    #[serde(default)]
    pub named_vectors: Option<Vec<BTreeMap<String, Vec<f32>>>>,
    #[serde(default)]
    pub sparse_vectors: Option<Vec<BTreeMap<String, SparseVectorRequest>>>,
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

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeleteByFilterRequest {
    pub filter: String,
}

#[derive(Debug, Serialize)]
pub struct DeleteRecordsResponse {
    pub deleted: u64,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum SearchRequest {
    Legacy(LegacySearchRequest),
    Typed(TypedSearchRequest),
}

#[derive(Debug, Deserialize)]
pub struct LegacySearchRequest {
    pub vector: Vec<f32>,
    pub top_k: usize,
    #[serde(default)]
    pub output_fields: Option<Vec<String>>,
    #[serde(default)]
    pub filter: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedSearchRequest {
    pub top_k: usize,
    #[serde(default)]
    pub queries: Vec<TypedVectorQueryRequest>,
    #[serde(default)]
    pub query_by_id: Option<Vec<String>>,
    #[serde(default)]
    pub query_by_id_field_name: Option<String>,
    #[serde(default)]
    pub filter: Option<String>,
    #[serde(default)]
    pub output_fields: Option<Vec<String>>,
    #[serde(default)]
    pub include_vector: bool,
    #[serde(default)]
    pub group_by: Option<TypedQueryGroupByRequest>,
    #[serde(default)]
    pub reranker: Option<TypedQueryRerankerRequest>,
    #[serde(default)]
    pub order_by: Option<TypedQueryOrderByRequest>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedVectorQueryRequest {
    pub field_name: String,
    #[serde(default)]
    pub vector: Option<Vec<f32>>,
    #[serde(default)]
    pub sparse_vector: Option<SparseVectorRequest>,
    #[serde(default)]
    pub param: Option<TypedVectorQueryParamRequest>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedVectorQueryParamRequest {
    #[serde(default)]
    pub ef_search: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedQueryGroupByRequest {
    pub field_name: String,
    #[serde(default)]
    pub group_topk: Option<usize>,
    #[serde(default)]
    pub group_count: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedQueryRerankerRequest {
    #[serde(default)]
    pub rank_constant: Option<u64>,
    #[serde(default)]
    pub weights: BTreeMap<String, f64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedQueryOrderByRequest {
    pub field_name: String,
    #[serde(default)]
    pub descending: bool,
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
    #[serde(default)]
    pub output_fields: Option<Vec<String>>,
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
pub struct CompactCollectionResponse {
    pub compacted: bool,
}

#[derive(Debug, Serialize)]
pub struct OptimizeCollectionResponse {
    pub optimized: String,
}

#[derive(Debug, Serialize)]
pub struct SegmentInfoResponse {
    pub id: String,
    pub live: usize,
    pub dead: usize,
    pub ann_ready: bool,
}

#[derive(Debug, Serialize)]
pub struct SegmentsResponse {
    pub segments: Vec<SegmentInfoResponse>,
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

#[derive(Debug, Deserialize)]
pub struct VectorIndexRequest {
    pub field_name: String,
    pub kind: String,
    #[serde(default)]
    pub metric: Option<String>,
    #[serde(default)]
    pub params: Value,
}

#[derive(Debug, Deserialize)]
pub struct ScalarIndexRequest {
    pub field_name: String,
    pub kind: String,
    #[serde(default)]
    pub params: Value,
}

#[derive(Debug, Serialize)]
pub struct CreateIndexResponse {
    pub field_name: String,
}

#[derive(Debug, Serialize)]
pub struct DropIndexResponse {
    pub dropped: String,
}

#[derive(Debug, Serialize)]
pub struct VectorIndexesResponse {
    pub vector_indexes: Vec<Value>,
}

#[derive(Debug, Serialize)]
pub struct ScalarIndexesResponse {
    pub scalar_indexes: Vec<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpdateRecordsRequest {
    pub ids: Vec<String>,
    #[serde(default)]
    pub fields: Vec<BTreeMap<String, Option<Value>>>,
    #[serde(default)]
    pub vectors: BTreeMap<String, Option<Vec<f32>>>,
}

#[derive(Debug, Serialize)]
pub struct UpdateRecordsResponse {
    pub updated: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AddColumnRequest {
    pub name: String,
    pub data_type: String,
    #[serde(default)]
    pub nullable: bool,
    #[serde(default)]
    pub array: bool,
}

#[derive(Debug, Serialize)]
pub struct AddColumnResponse {
    pub added: String,
}

#[derive(Debug, Serialize)]
pub struct DropColumnResponse {
    pub dropped: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AlterColumnRequest {
    pub new_name: String,
}

#[derive(Debug, Serialize)]
pub struct AlterColumnResponse {
    pub old_name: String,
    pub new_name: String,
}
