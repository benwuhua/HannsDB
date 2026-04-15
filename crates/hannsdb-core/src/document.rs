use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io;

use hannsdb_index::descriptor::{VectorIndexDescriptor, VectorIndexKind};
use hannsdb_index::scalar::ScalarValue;
use serde::{Deserialize, Serialize};

use crate::catalog::CollectionMetadata;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldType {
    String,
    Int64,
    Int32,
    UInt32,
    UInt64,
    Float,
    Float64,
    Bool,
    VectorFp32,
    VectorFp16,
    VectorSparse,
}

/// A sparse vector represented as parallel index/value arrays.
///
/// Indices must be sorted in ascending order with no duplicates.
/// This is the standard CSR-like representation used by most vector DB systems.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseVector {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

impl SparseVector {
    pub fn new(indices: Vec<u32>, values: Vec<f32>) -> Self {
        assert_eq!(
            indices.len(),
            values.len(),
            "sparse vector indices and values must have equal length"
        );
        Self { indices, values }
    }

    pub fn is_sorted(&self) -> bool {
        self.indices.windows(2).all(|w| w[0] < w[1])
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FieldValue {
    String(String),
    Int64(i64),
    Int32(i32),
    UInt32(u32),
    UInt64(u64),
    Float(f32),
    Float64(f64),
    Bool(bool),
    Array(Vec<FieldValue>),
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
    HnswHvq {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metric: Option<String>,
        m: usize,
        m_max0: usize,
        ef_construction: usize,
        ef_search: usize,
        nbits: usize,
    },
    HnswSq {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metric: Option<String>,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
    },
    Ivf {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metric: Option<String>,
        nlist: usize,
    },
    IvfUsq {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metric: Option<String>,
        nlist: usize,
        bits_per_dim: usize,
        rotation_seed: usize,
        rerank_k: usize,
        use_high_accuracy_scan: bool,
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

    pub fn hnsw_hvq(
        metric: Option<&str>,
        m: usize,
        m_max0: usize,
        ef_construction: usize,
        ef_search: usize,
        nbits: usize,
    ) -> Self {
        Self::HnswHvq {
            metric: metric.map(str::to_string),
            m,
            m_max0,
            ef_construction,
            ef_search,
            nbits,
        }
    }

    pub fn hnsw_sq(
        metric: Option<&str>,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
    ) -> Self {
        Self::HnswSq {
            metric: metric.map(str::to_string),
            m,
            ef_construction,
            ef_search,
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

    pub fn ivf_usq(
        metric: Option<&str>,
        nlist: usize,
        bits_per_dim: usize,
        rotation_seed: usize,
        rerank_k: usize,
        use_high_accuracy_scan: bool,
    ) -> Self {
        Self::IvfUsq {
            metric: metric.map(str::to_string),
            nlist,
            bits_per_dim,
            rotation_seed,
            rerank_k,
            use_high_accuracy_scan,
        }
    }

    pub fn metric(&self) -> Option<&str> {
        match self {
            Self::Hnsw { metric, .. }
            | Self::HnswHvq { metric, .. }
            | Self::HnswSq { metric, .. }
            | Self::Ivf { metric, .. }
            | Self::IvfUsq { metric, .. } => metric.as_deref(),
        }
    }

    pub fn quantize_type(&self) -> Option<&str> {
        match self {
            Self::Hnsw { quantize_type, .. } => quantize_type.as_deref(),
            Self::HnswHvq { .. } | Self::HnswSq { .. } | Self::Ivf { .. } | Self::IvfUsq { .. } => {
                None
            }
        }
    }

    pub fn hnsw_settings(&self) -> Option<(usize, usize)> {
        match self {
            Self::Hnsw {
                m, ef_construction, ..
            } => Some((*m, *ef_construction)),
            Self::HnswHvq {
                m, ef_construction, ..
            } => Some((*m, *ef_construction)),
            Self::HnswSq {
                m, ef_construction, ..
            } => Some((*m, *ef_construction)),
            Self::Ivf { .. } | Self::IvfUsq { .. } => None,
        }
    }
}

/// BM25 scoring parameters for sparse vector fields.
///
/// When present on a sparse vector field, these parameters are passed to the
/// sparse index backend so that BM25-style scoring can be applied during search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bm25Params {
    pub k1: f32,
    pub b: f32,
    pub avgdl: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorFieldSchema {
    pub name: String,
    pub data_type: FieldType,
    pub dimension: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_param: Option<VectorIndexSchema>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bm25_params: Option<Bm25Params>,
}

impl VectorFieldSchema {
    pub fn new(name: impl Into<String>, dimension: usize) -> Self {
        Self {
            name: name.into(),
            data_type: FieldType::VectorFp32,
            dimension,
            index_param: None,
            bm25_params: None,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    #[serde(default)]
    pub sparse_vectors: BTreeMap<String, Option<SparseVector>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Document {
    pub id: i64,
    pub fields: BTreeMap<String, FieldValue>,
    #[serde(default)]
    pub vectors: BTreeMap<String, Vec<f32>>,
    #[serde(default)]
    pub sparse_vectors: BTreeMap<String, SparseVector>,
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
            sparse_vectors: BTreeMap::new(),
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
            sparse_vectors: BTreeMap::new(),
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
            sparse_vectors: BTreeMap::new(),
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
            sparse_vectors: BTreeMap::new(),
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

    /// Create a document with sparse vector fields in addition to the primary dense vector.
    pub fn with_sparse_vectors(
        id: i64,
        fields: impl IntoIterator<Item = (String, FieldValue)>,
        primary_vector_name: &str,
        dense_vector: Vec<f32>,
        sparse_vectors: impl IntoIterator<Item = (String, SparseVector)>,
    ) -> Self {
        let mut vectors = BTreeMap::new();
        vectors.insert(primary_vector_name.to_string(), dense_vector);
        Self {
            id,
            fields: fields.into_iter().collect(),
            vectors,
            sparse_vectors: sparse_vectors.into_iter().collect(),
        }
    }
}

pub(crate) fn validate_documents(
    documents: &[Document],
    collection_meta: &CollectionMetadata,
) -> io::Result<()> {
    if collection_meta.dimension == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "collection dimension must be > 0",
        ));
    }

    let vector_schemas = collection_meta
        .vectors
        .iter()
        .map(|vector| (vector.name.as_str(), vector))
        .collect::<HashMap<_, _>>();
    let mut ids = HashSet::with_capacity(documents.len());
    for document in documents {
        if !ids.insert(document.id) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("duplicate external id in batch: {}", document.id),
            ));
        }
        let primary_vector = document
            .vectors
            .get(collection_meta.primary_vector.as_str())
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "document is missing primary vector '{}'",
                        collection_meta.primary_vector
                    ),
                )
            })?;
        if primary_vector.len() != collection_meta.dimension {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "document vector dimension mismatch: expected {}, got {}",
                    collection_meta.dimension,
                    primary_vector.len()
                ),
            ));
        }
        for (name, vector) in &document.vectors {
            let schema = vector_schemas.get(name.as_str()).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "document vector '{}' is not defined in collection schema",
                        name
                    ),
                )
            })?;
            if vector.len() != schema.dimension {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "document vector '{}' dimension mismatch: expected {}, got {}",
                        name,
                        schema.dimension,
                        vector.len()
                    ),
                ));
            }
        }
    }

    Ok(())
}

pub(crate) fn validate_vector_index_descriptor(
    dimension: usize,
    descriptor: &VectorIndexDescriptor,
) -> io::Result<()> {
    if dimension == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "vector dimension must be > 0",
        ));
    }

    let params = descriptor.params.as_object().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "vector index params must be a JSON object",
        )
    })?;

    match descriptor.kind {
        VectorIndexKind::Flat => {
            if !params.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "flat index does not accept params",
                ));
            }
        }
        VectorIndexKind::Ivf => {
            for key in params.keys() {
                if key != "nlist" {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("unsupported ivf param: {key}"),
                    ));
                }
            }
            if let Some(nlist) = params.get("nlist") {
                let nlist = nlist.as_u64().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "ivf nlist must be an unsigned integer",
                    )
                })?;
                if nlist == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "ivf nlist must be > 0",
                    ));
                }
            }
        }
        VectorIndexKind::IvfUsq => {
            for key in params.keys() {
                if key != "nlist"
                    && key != "bits_per_dim"
                    && key != "rotation_seed"
                    && key != "rerank_k"
                    && key != "use_high_accuracy_scan"
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("unsupported ivf_usq param: {key}"),
                    ));
                }
            }
            for key in ["nlist", "bits_per_dim", "rotation_seed", "rerank_k"] {
                if let Some(value) = params.get(key) {
                    let value = value.as_u64().ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("ivf_usq {key} must be an unsigned integer"),
                        )
                    })?;
                    if value == 0 {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("ivf_usq {key} must be > 0"),
                        ));
                    }
                }
            }
            if let Some(value) = params.get("use_high_accuracy_scan") {
                value.as_bool().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "ivf_usq use_high_accuracy_scan must be a boolean",
                    )
                })?;
            }
        }
        VectorIndexKind::HnswHvq => {
            for key in params.keys() {
                if key != "m"
                    && key != "m_max0"
                    && key != "ef_construction"
                    && key != "ef_search"
                    && key != "nbits"
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("unsupported hnsw_hvq param: {key}"),
                    ));
                }
            }
            for key in ["m", "m_max0", "ef_construction", "ef_search", "nbits"] {
                if let Some(value) = params.get(key) {
                    let value = value.as_u64().ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("hnsw_hvq {key} must be an unsigned integer"),
                        )
                    })?;
                    if value == 0 {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("hnsw_hvq {key} must be > 0"),
                        ));
                    }
                }
            }
            if descriptor.metric.as_deref() != Some("ip") {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "hnsw_hvq currently supports only ip in HannsDB",
                ));
            }
        }
        VectorIndexKind::HnswSq => {
            for key in params.keys() {
                if key != "m" && key != "ef_construction" && key != "ef_search" {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("unsupported hnsw_sq param: {key}"),
                    ));
                }
            }
            for key in ["m", "ef_construction", "ef_search"] {
                if let Some(value) = params.get(key) {
                    let value = value.as_u64().ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("hnsw_sq {key} must be an unsigned integer"),
                        )
                    })?;
                    if value == 0 {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("hnsw_sq {key} must be > 0"),
                        ));
                    }
                }
            }
            // HnswSq supports all metrics (l2, ip, cosine); no metric restriction.
        }
        VectorIndexKind::Hnsw => {
            for key in params.keys() {
                if key != "m" && key != "ef_construction" {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("unsupported hnsw param: {key}"),
                    ));
                }
            }
            for key in ["m", "ef_construction"] {
                if let Some(value) = params.get(key) {
                    let value = value.as_u64().ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("hnsw {key} must be an unsigned integer"),
                        )
                    })?;
                    if value == 0 {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("hnsw {key} must be > 0"),
                        ));
                    }
                }
            }
        }
    }

    // Validate metric string if present (HnswHvq is already validated above).
    if descriptor.kind != VectorIndexKind::HnswHvq {
        if let Some(metric) = descriptor.metric.as_deref() {
            match metric.to_ascii_lowercase().as_str() {
                "l2" | "ip" | "cosine" => {}
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("unsupported metric: {other}"),
                    ));
                }
            }
        }
    }

    Ok(())
}

pub(crate) fn validate_schema_primary_vector_descriptor(
    schema: &CollectionSchema,
) -> io::Result<()> {
    let primary_vector = schema.primary_vector().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "collection primary vector '{}' is not defined in schema vectors",
                schema.primary_vector_name()
            ),
        )
    })?;
    let descriptor = match primary_vector.index_param.as_ref() {
        Some(VectorIndexSchema::Ivf { metric, nlist }) => VectorIndexDescriptor {
            field_name: primary_vector.name.clone(),
            kind: VectorIndexKind::Ivf,
            metric: metric.clone(),
            params: serde_json::json!({ "nlist": nlist }),
        },
        Some(VectorIndexSchema::IvfUsq {
            metric,
            nlist,
            bits_per_dim,
            rotation_seed,
            rerank_k,
            use_high_accuracy_scan,
        }) => VectorIndexDescriptor {
            field_name: primary_vector.name.clone(),
            kind: VectorIndexKind::IvfUsq,
            metric: metric.clone(),
            params: serde_json::json!({
                "nlist": nlist,
                "bits_per_dim": bits_per_dim,
                "rotation_seed": rotation_seed,
                "rerank_k": rerank_k,
                "use_high_accuracy_scan": use_high_accuracy_scan
            }),
        },
        Some(VectorIndexSchema::HnswHvq {
            metric,
            m,
            m_max0,
            ef_construction,
            ef_search,
            nbits,
        }) => VectorIndexDescriptor {
            field_name: primary_vector.name.clone(),
            kind: VectorIndexKind::HnswHvq,
            metric: metric.clone(),
            params: serde_json::json!({
                "m": m,
                "m_max0": m_max0,
                "ef_construction": ef_construction,
                "ef_search": ef_search,
                "nbits": nbits
            }),
        },
        Some(VectorIndexSchema::HnswSq {
            metric,
            m,
            ef_construction,
            ef_search,
        }) => VectorIndexDescriptor {
            field_name: primary_vector.name.clone(),
            kind: VectorIndexKind::HnswSq,
            metric: metric.clone(),
            params: serde_json::json!({
                "m": m,
                "ef_construction": ef_construction,
                "ef_search": ef_search
            }),
        },
        Some(VectorIndexSchema::Hnsw {
            metric,
            m,
            ef_construction,
            ..
        }) => VectorIndexDescriptor {
            field_name: primary_vector.name.clone(),
            kind: VectorIndexKind::Hnsw,
            metric: metric.clone(),
            params: serde_json::json!({
                "m": m,
                "ef_construction": ef_construction
            }),
        },
        None => VectorIndexDescriptor {
            field_name: primary_vector.name.clone(),
            kind: VectorIndexKind::Hnsw,
            metric: Some(schema.metric().to_string()),
            params: serde_json::json!({
                "m": schema.hnsw_m(),
                "ef_construction": schema.hnsw_ef_construction()
            }),
        },
    };
    validate_vector_index_descriptor(primary_vector.dimension, &descriptor)
}

pub(crate) fn validate_schema_secondary_vector_descriptors(
    schema: &CollectionSchema,
) -> io::Result<()> {
    let primary_vector_name = schema.primary_vector_name();
    for vector in schema
        .vectors
        .iter()
        .filter(|vector| vector.name != primary_vector_name)
    {
        let Some(index_param) = vector.index_param.as_ref() else {
            continue;
        };
        let descriptor = match index_param {
            VectorIndexSchema::Ivf { metric, nlist } => VectorIndexDescriptor {
                field_name: vector.name.clone(),
                kind: VectorIndexKind::Ivf,
                metric: metric.clone(),
                params: serde_json::json!({ "nlist": nlist }),
            },
            VectorIndexSchema::IvfUsq {
                metric,
                nlist,
                bits_per_dim,
                rotation_seed,
                rerank_k,
                use_high_accuracy_scan,
            } => VectorIndexDescriptor {
                field_name: vector.name.clone(),
                kind: VectorIndexKind::IvfUsq,
                metric: metric.clone(),
                params: serde_json::json!({
                    "nlist": nlist,
                    "bits_per_dim": bits_per_dim,
                    "rotation_seed": rotation_seed,
                    "rerank_k": rerank_k,
                    "use_high_accuracy_scan": use_high_accuracy_scan
                }),
            },
            VectorIndexSchema::HnswHvq {
                metric,
                m,
                m_max0,
                ef_construction,
                ef_search,
                nbits,
            } => VectorIndexDescriptor {
                field_name: vector.name.clone(),
                kind: VectorIndexKind::HnswHvq,
                metric: metric.clone(),
                params: serde_json::json!({
                    "m": m,
                    "m_max0": m_max0,
                    "ef_construction": ef_construction,
                    "ef_search": ef_search,
                    "nbits": nbits
                }),
            },
            VectorIndexSchema::HnswSq {
                metric,
                m,
                ef_construction,
                ef_search,
            } => VectorIndexDescriptor {
                field_name: vector.name.clone(),
                kind: VectorIndexKind::HnswSq,
                metric: metric.clone(),
                params: serde_json::json!({
                    "m": m,
                    "ef_construction": ef_construction,
                    "ef_search": ef_search
                }),
            },
            VectorIndexSchema::Hnsw {
                metric,
                m,
                ef_construction,
                ..
            } => VectorIndexDescriptor {
                field_name: vector.name.clone(),
                kind: VectorIndexKind::Hnsw,
                metric: metric.clone(),
                params: serde_json::json!({
                    "m": m,
                    "ef_construction": ef_construction
                }),
            },
        };
        validate_vector_index_descriptor(vector.dimension, &descriptor)?;
    }
    Ok(())
}

/// Convert a core `FieldValue` into the index crate's `ScalarValue`.
pub(crate) fn field_value_to_scalar(value: &FieldValue) -> ScalarValue {
    match value {
        FieldValue::String(s) => ScalarValue::String(s.clone()),
        FieldValue::Int64(v) => ScalarValue::Int64(*v),
        FieldValue::Int32(v) => ScalarValue::Int64(*v as i64),
        FieldValue::UInt32(v) => ScalarValue::Int64(*v as i64),
        FieldValue::UInt64(v) => ScalarValue::Int64(*v as i64),
        FieldValue::Float(v) => ScalarValue::Float64(*v as f64),
        FieldValue::Float64(v) => ScalarValue::Float64(*v),
        FieldValue::Bool(b) => ScalarValue::Bool(*b),
        FieldValue::Array(items) => match items.first() {
            Some(first) => field_value_to_scalar(first),
            None => ScalarValue::String("[]".to_string()),
        },
    }
}

pub(crate) fn compare_field_value_for_sort(a: &FieldValue, b: &FieldValue) -> Ordering {
    match (a, b) {
        (FieldValue::String(sa), FieldValue::String(sb)) => sa.cmp(sb),
        (FieldValue::Int64(va), FieldValue::Int64(vb)) => va.cmp(vb),
        (FieldValue::Int32(va), FieldValue::Int32(vb)) => va.cmp(vb),
        (FieldValue::UInt32(va), FieldValue::UInt32(vb)) => va.cmp(vb),
        (FieldValue::UInt64(va), FieldValue::UInt64(vb)) => va.cmp(vb),
        (FieldValue::Float(va), FieldValue::Float(vb)) => va.total_cmp(vb),
        (FieldValue::Float64(va), FieldValue::Float64(vb)) => va.total_cmp(vb),
        (FieldValue::Bool(va), FieldValue::Bool(vb)) => va.cmp(vb),
        (FieldValue::Int64(va), FieldValue::Float64(vb)) => (*va as f64).total_cmp(vb),
        (FieldValue::Float64(va), FieldValue::Int64(vb)) => va.total_cmp(&(*vb as f64)),
        (FieldValue::Int32(va), FieldValue::Float64(vb)) => (*va as f64).total_cmp(vb),
        (FieldValue::Float64(va), FieldValue::Int32(vb)) => va.total_cmp(&(*vb as f64)),
        (FieldValue::UInt64(va), FieldValue::Float64(vb)) => (*va as f64).total_cmp(vb),
        (FieldValue::Float64(va), FieldValue::UInt64(vb)) => va.total_cmp(&(*vb as f64)),
        (FieldValue::Int64(va), FieldValue::Int32(vb)) => va.cmp(&(*vb as i64)),
        (FieldValue::Int32(va), FieldValue::Int64(vb)) => (*va as i64).cmp(vb),
        (FieldValue::Int64(va), FieldValue::UInt32(vb)) => va.cmp(&(*vb as i64)),
        (FieldValue::UInt32(va), FieldValue::Int64(vb)) => (*va as i64).cmp(vb),
        (FieldValue::Int64(va), FieldValue::UInt64(vb)) => {
            if *va < 0 {
                Ordering::Less
            } else {
                (*va as u64).cmp(vb)
            }
        }
        (FieldValue::UInt64(va), FieldValue::Int64(vb)) => {
            if *vb < 0 {
                Ordering::Greater
            } else {
                va.cmp(&(*vb as u64))
            }
        }
        _ => format!("{a:?}").cmp(&format!("{b:?}")),
    }
}
