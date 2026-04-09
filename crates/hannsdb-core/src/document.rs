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
        #[serde(default, skip_serializing_if = "Option::is_none")]
        quantize_type: Option<String>,
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
            quantize_type: None,
        }
    }

    pub fn with_quantize_type(self, quantize_type: Option<&str>) -> Self {
        match self {
            Self::Hnsw {
                metric,
                m,
                ef_construction,
                ..
            } => Self::Hnsw {
                metric,
                m,
                ef_construction,
                quantize_type: quantize_type.map(str::to_string),
            },
            other => other,
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

    pub fn quantize_type(&self) -> Option<&str> {
        match self {
            Self::Hnsw { quantize_type, .. } => quantize_type.as_deref(),
            Self::Ivf { .. } => None,
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
    #[serde(default = "default_primary_vector_name")]
    pub primary_vector: String,
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
        let primary_vector = primary_vector.into();
        let metric = metric.into();
        Self {
            primary_vector: primary_vector.clone(),
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

    pub fn with_primary_vector(mut self, primary_vector: impl Into<String>) -> Self {
        self.primary_vector = primary_vector.into();
        self
    }

    pub fn primary_vector(&self) -> Option<&VectorFieldSchema> {
        self.vectors
            .iter()
            .find(|vector| vector.name == self.primary_vector)
    }

    pub fn primary_vector_name(&self) -> &str {
        self.primary_vector.as_str()
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

pub fn default_primary_vector_name() -> String {
    "vector".to_string()
}

/// Partial update descriptor for an existing document.
///
/// `Some` values replace the existing field/vector; `None` means "keep current".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocumentUpdate {
    pub id: i64,
    pub fields: BTreeMap<String, Option<FieldValue>>,
    pub vectors: BTreeMap<String, Option<Vec<f32>>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Document {
    pub id: i64,
    pub fields: BTreeMap<String, FieldValue>,
    #[serde(default)]
    pub vectors: BTreeMap<String, Vec<f32>>,
}

impl Document {
    /// Creates a document with a primary vector stored under the default key "vector".
    /// For collections where the primary vector has a different name, use `with_primary_vector_name`.
    pub fn new(
        id: i64,
        fields: impl IntoIterator<Item = (String, FieldValue)>,
        vector: Vec<f32>,
    ) -> Self {
        let mut vectors = BTreeMap::new();
        vectors.insert("vector".to_string(), vector);
        Self {
            id,
            fields: fields.into_iter().collect(),
            vectors,
        }
    }

    pub fn with_primary_vector_name(
        id: i64,
        fields: impl IntoIterator<Item = (String, FieldValue)>,
        primary_vector_name: &str,
        vector: Vec<f32>,
    ) -> Self {
        let mut vectors = BTreeMap::new();
        vectors.insert(primary_vector_name.to_string(), vector);
        Self {
            id,
            fields: fields.into_iter().collect(),
            vectors,
        }
    }

    pub fn with_vectors(
        id: i64,
        fields: impl IntoIterator<Item = (String, FieldValue)>,
        vector: Vec<f32>,
        secondary_vectors: impl IntoIterator<Item = (String, Vec<f32>)>,
    ) -> Self {
        let mut vectors = BTreeMap::new();
        vectors.insert("vector".to_string(), vector);
        for (name, vec) in secondary_vectors {
            vectors.insert(name, vec);
        }
        Self {
            id,
            fields: fields.into_iter().collect(),
            vectors,
        }
    }

    pub fn with_named_vectors(
        id: i64,
        fields: impl IntoIterator<Item = (String, FieldValue)>,
        primary_vector_name: &str,
        vector: Vec<f32>,
        secondary_vectors: impl IntoIterator<Item = (String, Vec<f32>)>,
    ) -> Self {
        let mut vectors = BTreeMap::new();
        vectors.insert(primary_vector_name.to_string(), vector);
        for (name, vec) in secondary_vectors {
            vectors.insert(name, vec);
        }
        Self {
            id,
            fields: fields.into_iter().collect(),
            vectors,
        }
    }

    pub fn primary_vector(&self) -> &[f32] {
        self.vectors
            .get("vector")
            .expect("document must have primary vector")
    }

    pub fn primary_vector_for(&self, primary_vector_name: &str) -> Option<&[f32]> {
        self.vectors.get(primary_vector_name).map(Vec::as_slice)
    }

    pub fn vectors_with_primary(&self, _primary_vector_name: &str) -> &BTreeMap<String, Vec<f32>> {
        &self.vectors
    }
}
