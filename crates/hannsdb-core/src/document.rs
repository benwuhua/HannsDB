use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldType {
    String,
    Int64,
    Float64,
    Bool,
    VectorFp32,
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
    #[serde(default)]
    pub nullable: bool,
    #[serde(default)]
    pub array: bool,
}

impl ScalarFieldSchema {
    pub fn new(name: impl Into<String>, data_type: FieldType) -> Self {
        Self {
            name: name.into(),
            data_type,
            nullable: false,
            array: false,
        }
    }

    pub fn with_flags(mut self, nullable: bool, array: bool) -> Self {
        self.nullable = nullable;
        self.array = array;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VectorIndexSchema {
    Hnsw {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metric: Option<String>,
        m: usize,
        ef_construction: usize,
    },
    Ivf {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metric: Option<String>,
        nlist: usize,
    },
}

impl VectorIndexSchema {
    pub fn hnsw(metric: Option<&str>, m: usize, ef_construction: usize) -> Self {
        Self::Hnsw {
            metric: metric.map(str::to_string),
            m,
            ef_construction,
        }
    }

    pub fn ivf(metric: Option<&str>, nlist: usize) -> Self {
        Self::Ivf {
            metric: metric.map(str::to_string),
            nlist,
        }
    }

    pub fn metric(&self) -> Option<&str> {
        match self {
            Self::Hnsw { metric, .. } | Self::Ivf { metric, .. } => metric.as_deref(),
        }
    }

    pub fn hnsw_settings(&self) -> Option<(usize, usize)> {
        match self {
            Self::Hnsw {
                m, ef_construction, ..
            } => Some((*m, *ef_construction)),
            Self::Ivf { .. } => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VectorFieldSchema {
    pub name: String,
    pub data_type: FieldType,
    pub dimension: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_param: Option<VectorIndexSchema>,
}

impl VectorFieldSchema {
    pub fn new(name: impl Into<String>, dimension: usize) -> Self {
        Self {
            name: name.into(),
            data_type: FieldType::VectorFp32,
            dimension,
            index_param: None,
        }
    }

    pub fn with_index_param(mut self, index_param: VectorIndexSchema) -> Self {
        self.index_param = Some(index_param);
        self
    }

    pub fn metric(&self) -> Option<&str> {
        self.index_param
            .as_ref()
            .and_then(VectorIndexSchema::metric)
    }

    pub fn hnsw_settings(&self) -> Option<(usize, usize)> {
        self.index_param
            .as_ref()
            .and_then(VectorIndexSchema::hnsw_settings)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectionSchema {
    #[serde(default)]
    pub fields: Vec<ScalarFieldSchema>,
    pub vectors: Vec<VectorFieldSchema>,
}

impl CollectionSchema {
    pub fn new(
        primary_vector: impl Into<String>,
        dimension: usize,
        metric: impl Into<String>,
        fields: Vec<ScalarFieldSchema>,
    ) -> Self {
        let metric = metric.into();
        Self {
            fields,
            vectors: vec![
                VectorFieldSchema::new(primary_vector, dimension).with_index_param(
                    VectorIndexSchema::hnsw(
                        Some(metric.as_str()),
                        default_hnsw_m(),
                        default_hnsw_ef_construction(),
                    ),
                ),
            ],
        }
    }

    pub fn primary_vector(&self) -> Option<&VectorFieldSchema> {
        self.vectors.first()
    }

    pub fn primary_vector_name(&self) -> &str {
        self.primary_vector()
            .map(|vector| vector.name.as_str())
            .unwrap_or("vector")
    }

    pub fn dimension(&self) -> usize {
        self.primary_vector().map_or(0, |vector| vector.dimension)
    }

    pub fn metric(&self) -> &str {
        self.primary_vector()
            .and_then(VectorFieldSchema::metric)
            .unwrap_or("l2")
    }

    pub fn hnsw_m(&self) -> usize {
        self.primary_vector()
            .and_then(VectorFieldSchema::hnsw_settings)
            .map_or(default_hnsw_m(), |(m, _)| m)
    }

    pub fn hnsw_ef_construction(&self) -> usize {
        self.primary_vector()
            .and_then(VectorFieldSchema::hnsw_settings)
            .map_or(default_hnsw_ef_construction(), |(_, ef)| ef)
    }
}

pub fn default_hnsw_m() -> usize {
    16
}

pub fn default_hnsw_ef_construction() -> usize {
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
