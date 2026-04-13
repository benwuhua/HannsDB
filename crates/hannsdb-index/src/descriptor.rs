use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorIndexKind {
    Flat,
    Hnsw,
    HnswHvq,
    Ivf,
    IvfUsq,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseIndexKind {
    SparseInverted,
    SparseWand,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarIndexKind {
    Inverted,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorIndexDescriptor {
    pub field_name: String,
    pub kind: VectorIndexKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metric: Option<String>,
    #[serde(default)]
    pub params: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseIndexDescriptor {
    pub field_name: String,
    pub kind: SparseIndexKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metric: Option<String>, // "ip" | "bm25"
    #[serde(default)]
    pub params: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalarIndexDescriptor {
    pub field_name: String,
    pub kind: ScalarIndexKind,
    #[serde(default)]
    pub params: Value,
}
