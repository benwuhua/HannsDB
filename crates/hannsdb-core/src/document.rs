use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldType {
    String,
    Int64,
    Float64,
    Bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FieldValue {
    String(String),
    Int64(i64),
    Float64(f64),
    Bool(bool),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScalarFieldSchema {
    pub name: String,
    pub data_type: FieldType,
}

impl ScalarFieldSchema {
    pub fn new(name: impl Into<String>, data_type: FieldType) -> Self {
        Self {
            name: name.into(),
            data_type,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectionSchema {
    pub primary_vector: String,
    pub dimension: usize,
    pub metric: String,
    pub fields: Vec<ScalarFieldSchema>,
    #[serde(default = "default_hnsw_m")]
    pub hnsw_m: usize,
    #[serde(default = "default_hnsw_ef_construction")]
    pub hnsw_ef_construction: usize,
}

impl CollectionSchema {
    pub fn new(
        primary_vector: impl Into<String>,
        dimension: usize,
        metric: impl Into<String>,
        fields: Vec<ScalarFieldSchema>,
    ) -> Self {
        Self {
            primary_vector: primary_vector.into(),
            dimension,
            metric: metric.into(),
            fields,
            hnsw_m: default_hnsw_m(),
            hnsw_ef_construction: default_hnsw_ef_construction(),
        }
    }
}

fn default_hnsw_m() -> usize {
    16
}

fn default_hnsw_ef_construction() -> usize {
    128
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Document {
    pub id: i64,
    pub fields: BTreeMap<String, FieldValue>,
    pub vector: Vec<f32>,
}

impl Document {
    pub fn new(
        id: i64,
        fields: impl IntoIterator<Item = (String, FieldValue)>,
        vector: Vec<f32>,
    ) -> Self {
        Self {
            id,
            fields: fields.into_iter().collect(),
            vector,
        }
    }
}
