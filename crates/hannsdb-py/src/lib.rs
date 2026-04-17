use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "python-binding")]
use numpy::PyReadonlyArray1;
#[cfg(feature = "python-binding")]
use pyo3::exceptions::{PyFileNotFoundError, PyNotImplementedError, PyRuntimeError, PyValueError};
#[cfg(feature = "python-binding")]
use pyo3::prelude::*;
#[cfg(feature = "python-binding")]
use pyo3::types::{PyAny, PyBool, PyDict, PyList};

use hannsdb_core::catalog::{CollectionMetadata, IndexCatalog};
use hannsdb_core::document::{
    CollectionSchema as CoreCollectionSchema, Document as CoreDocument, FieldType as CoreFieldType,
    FieldValue, ScalarFieldSchema as CoreScalarFieldSchema,
    VectorFieldSchema as CoreVectorFieldSchema, VectorIndexSchema as CoreVectorIndexSchema,
};
#[cfg(feature = "python-binding")]
use hannsdb_core::query::{
    OrderBy as CoreOrderBy, QueryContext as CoreQueryContext, QueryGroupBy as CoreQueryGroupBy,
    QueryReranker as CoreQueryReranker, VectorQuery as CoreVectorQuery,
    VectorQueryParam as CoreVectorQueryParam,
};
#[cfg(all(feature = "python-binding", feature = "lance-storage"))]
use hannsdb_core::storage::lance_store::LanceCollection as CoreLanceCollection;
#[cfg(feature = "python-binding")]
use hannsdb_core::wal::{AddColumnBackfill, AlterColumnMigration};

pub fn bootstrap_symbol() -> &'static str {
    "hannsdb_py_bootstrap"
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricType {
    L2,
    Cosine,
    Ip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizeType {
    Undefined,
    Fp16,
    Int8,
    Int4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OptimizeOption {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddColumnOption {
    pub concurrency: usize,
}

impl Default for AddColumnOption {
    fn default() -> Self {
        Self { concurrency: 0 }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlterColumnOption {
    pub concurrency: usize,
}

impl Default for AlterColumnOption {
    fn default() -> Self {
        Self { concurrency: 0 }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataType {
    String,
    Int64,
    Int32,
    UInt32,
    UInt64,
    Float,
    Float64,
    Bool,
    VectorFp32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionOption {
    pub read_only: bool,
    pub enable_mmap: bool,
}

impl Default for CollectionOption {
    fn default() -> Self {
        Self {
            read_only: false,
            enable_mmap: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HnswIndexParam {
    pub metric_type: Option<MetricType>,
    pub m: usize,
    pub ef_construction: usize,
    pub quantize_type: QuantizeType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HnswHvqIndexParam {
    pub metric_type: Option<MetricType>,
    pub m: usize,
    pub m_max0: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub nbits: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HnswSqIndexParam {
    pub metric_type: Option<MetricType>,
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IvfIndexParam {
    pub metric_type: Option<MetricType>,
    pub nlist: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IvfUsqIndexParam {
    pub metric_type: Option<MetricType>,
    pub nlist: usize,
    pub bits_per_dim: usize,
    pub rotation_seed: usize,
    pub rerank_k: usize,
    pub use_high_accuracy_scan: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatIndexParam {
    pub metric_type: Option<MetricType>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvertIndexParam {
    pub enable_range_optimization: bool,
    pub enable_extended_wildcard: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexParam {
    Flat(FlatIndexParam),
    Hnsw(HnswIndexParam),
    HnswHvq(HnswHvqIndexParam),
    HnswSq(HnswSqIndexParam),
    Ivf(IvfIndexParam),
    IvfUsq(IvfUsqIndexParam),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HnswQueryParam {
    pub ef: usize,
    pub nprobe: usize,
    pub is_using_refiner: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IvfQueryParam {
    pub nprobe: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IvfUsqQueryParam {
    pub nprobe: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HnswSqQueryParam {
    pub ef_search: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HnswHvqQueryParam {
    pub ef_search: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexOption {
    pub concurrency: usize,
}

impl Default for IndexOption {
    fn default() -> Self {
        Self { concurrency: 0 }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldSchema {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub array: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorSchema {
    pub name: String,
    pub data_type: DataType,
    pub dimension: usize,
    pub index_param: Option<IndexParam>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionSchema {
    pub name: String,
    pub primary_vector: String,
    pub fields: Vec<FieldSchema>,
    pub vectors: Vec<VectorSchema>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Doc {
    pub id: String,
    pub score: Option<f32>,
    pub fields: BTreeMap<String, FieldValue>,
    pub vectors: BTreeMap<String, Vec<f32>>,
    pub field_name: String,
    /// Set when group_by is active: the value of the group_by field.
    pub group_key: Option<FieldValue>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseVectorData {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum QueryVector {
    Dense(Vec<f32>),
    Sparse(SparseVectorData),
}

#[derive(Debug, Clone, PartialEq)]
pub struct VectorQuery {
    pub field_name: String,
    pub vector: QueryVector,
    pub param: Option<HnswQueryParam>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CollectionStats {
    pub name: String,
    pub dimension: usize,
    pub metric: String,
    pub record_count: usize,
    pub deleted_count: usize,
    pub live_count: usize,
    /// For each vector field, fraction of data covered by an ANN index (0.0..=1.0).
    pub index_completeness: BTreeMap<String, f64>,
}

pub struct Collection {
    pub path: String,
    pub collection_name: String,
    pub primary_vector_name: String,
    pub option: CollectionOption,
    pub(crate) db: hannsdb_core::db::HannsDb,
}

pub fn init(_log_level: LogLevel) {}

fn metric_type_name(metric: MetricType) -> &'static str {
    match metric {
        MetricType::L2 => "l2",
        MetricType::Cosine => "cosine",
        MetricType::Ip => "ip",
    }
}

fn quantize_type_name(quantize: QuantizeType) -> &'static str {
    match quantize {
        QuantizeType::Undefined => "undefined",
        QuantizeType::Fp16 => "fp16",
        QuantizeType::Int8 => "int8",
        QuantizeType::Int4 => "int4",
    }
}

fn quantize_type_value(quantize: QuantizeType) -> Option<&'static str> {
    match quantize {
        QuantizeType::Undefined => None,
        _ => Some(quantize_type_name(quantize)),
    }
}

fn core_schema_from_schema(schema: &CollectionSchema) -> std::io::Result<CoreCollectionSchema> {
    if schema.vectors.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "CollectionSchema requires at least one vector schema",
        ));
    }
    if !schema
        .vectors
        .iter()
        .any(|vector| vector.name == schema.primary_vector)
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "primary vector '{}' is not defined in vectors",
                schema.primary_vector
            ),
        ));
    }
    let fields = schema
        .fields
        .iter()
        .map(|field| {
            let data_type = match field.data_type {
                DataType::String => CoreFieldType::String,
                DataType::Int64 => CoreFieldType::Int64,
                DataType::Int32 => CoreFieldType::Int32,
                DataType::UInt32 => CoreFieldType::UInt32,
                DataType::UInt64 => CoreFieldType::UInt64,
                DataType::Float => CoreFieldType::Float,
                DataType::Float64 => CoreFieldType::Float64,
                DataType::Bool => CoreFieldType::Bool,
                DataType::VectorFp32 => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("field '{}' cannot use vector_fp32", field.name),
                    ))
                }
            };
            Ok(CoreScalarFieldSchema::new(field.name.clone(), data_type)
                .with_flags(field.nullable, field.array))
        })
        .collect::<std::io::Result<Vec<_>>>()?;

    let vectors = schema
        .vectors
        .iter()
        .map(|vector| {
            if vector.data_type != DataType::VectorFp32 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "vector schema must use vector_fp32",
                ));
            }

            let index_param = match vector.index_param.as_ref() {
                Some(IndexParam::Flat(_params)) => None,
                Some(IndexParam::Hnsw(params)) => Some(
                    CoreVectorIndexSchema::hnsw(
                        params.metric_type.map(metric_type_name),
                        params.m,
                        params.ef_construction,
                    )
                    .with_quantize_type(quantize_type_value(params.quantize_type)),
                ),
                Some(IndexParam::HnswHvq(params)) => Some(CoreVectorIndexSchema::hnsw_hvq(
                    params.metric_type.map(metric_type_name),
                    params.m,
                    params.m_max0,
                    params.ef_construction,
                    params.ef_search,
                    params.nbits,
                )),
                Some(IndexParam::HnswSq(params)) => Some(CoreVectorIndexSchema::hnsw_sq(
                    params.metric_type.map(metric_type_name),
                    params.m,
                    params.ef_construction,
                    params.ef_search,
                )),
                Some(IndexParam::Ivf(params)) => Some(CoreVectorIndexSchema::ivf(
                    params.metric_type.map(metric_type_name),
                    params.nlist,
                )),
                Some(IndexParam::IvfUsq(params)) => Some(CoreVectorIndexSchema::ivf_usq(
                    params.metric_type.map(metric_type_name),
                    params.nlist,
                    params.bits_per_dim,
                    params.rotation_seed,
                    params.rerank_k,
                    params.use_high_accuracy_scan,
                )),
                None => None,
            };

            Ok(CoreVectorFieldSchema {
                name: vector.name.clone(),
                data_type: CoreFieldType::VectorFp32,
                dimension: vector.dimension,
                index_param,
                bm25_params: None,
            })
        })
        .collect::<std::io::Result<Vec<_>>>()?;

    Ok(CoreCollectionSchema {
        primary_vector: schema.primary_vector.clone(),
        fields,
        vectors,
    })
}

#[cfg(all(feature = "python-binding", feature = "lance-storage"))]
fn schema_from_core_schema(
    name: String,
    schema: &CoreCollectionSchema,
) -> std::io::Result<CollectionSchema> {
    let fields = schema
        .fields
        .iter()
        .map(|field| {
            let data_type = match field.data_type {
                CoreFieldType::String => DataType::String,
                CoreFieldType::Int64 => DataType::Int64,
                CoreFieldType::Int32 => DataType::Int32,
                CoreFieldType::UInt32 => DataType::UInt32,
                CoreFieldType::UInt64 => DataType::UInt64,
                CoreFieldType::Float => DataType::Float,
                CoreFieldType::Float64 => DataType::Float64,
                CoreFieldType::Bool => DataType::Bool,
                CoreFieldType::VectorFp32
                | CoreFieldType::VectorFp16
                | CoreFieldType::VectorSparse => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("field '{}' cannot use vector data type", field.name),
                    ))
                }
            };
            Ok(FieldSchema {
                name: field.name.clone(),
                data_type,
                nullable: field.nullable,
                array: field.array,
            })
        })
        .collect::<std::io::Result<Vec<_>>>()?;

    let vectors = schema
        .vectors
        .iter()
        .map(|vector| {
            if !matches!(vector.data_type, CoreFieldType::VectorFp32) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("vector '{}' is not vector_fp32", vector.name),
                ));
            }
            Ok(VectorSchema {
                name: vector.name.clone(),
                data_type: DataType::VectorFp32,
                dimension: vector.dimension,
                index_param: None,
            })
        })
        .collect::<std::io::Result<Vec<_>>>()?;

    Ok(CollectionSchema {
        name,
        primary_vector: schema.primary_vector.clone(),
        fields,
        vectors,
    })
}

#[cfg(feature = "python-binding")]
fn parse_query_ids(
    py: Python<'_>,
    collection: &Collection,
    value: &Bound<'_, PyAny>,
) -> PyResult<Option<Vec<i64>>> {
    if value.is_none() {
        return Ok(None);
    }

    let items: Vec<Py<PyAny>> = value.extract()?;
    let mut public_ids = Vec::with_capacity(items.len());
    for item in items {
        let item = item.bind(py);
        if let Ok(id) = item.extract::<i64>() {
            public_ids.push(id.to_string());
            continue;
        }

        let id = item.extract::<String>()?;
        public_ids.push(id);
    }
    if public_ids.is_empty() {
        Ok(None)
    } else {
        let resolved = collection
            .db
            .resolve_query_ids_by_primary_keys(&collection.collection_name, &public_ids)
            .map_err(io_to_py_err)?;
        if resolved.is_empty() {
            Ok(None)
        } else {
            Ok(Some(resolved))
        }
    }
}

#[cfg(feature = "python-binding")]
fn py_vector_query_from_pyany(query: &Bound<'_, PyAny>) -> PyResult<VectorQuery> {
    let field_name = query.getattr("field_name")?.extract::<String>()?;

    // Check if the vector is a sparse vector (has indices and values attributes)
    let vector_attr = query.getattr("vector")?;
    let vector = if vector_attr.hasattr("indices")? && vector_attr.hasattr("values")? {
        // Sparse vector
        let indices = vector_attr.getattr("indices")?.extract::<Vec<u32>>()?;
        let values = vector_attr.getattr("values")?.extract::<Vec<f32>>()?;
        QueryVector::Sparse(SparseVectorData { indices, values })
    } else {
        // Dense vector
        let vec = vector_attr.extract::<Vec<f32>>()?;
        QueryVector::Dense(vec)
    };

    let param = match query.getattr("param") {
        Ok(param) if !param.is_none() => {
            // Check if it's a native PyHnswSqQueryParam first (via ef_search attr, no ef/nprobe).
            let ef_search_val = param
                .getattr("ef_search")
                .ok()
                .and_then(|v| v.extract::<usize>().ok());
            let has_ef = param
                .getattr("ef")
                .ok()
                .and_then(|v| v.extract::<usize>().ok())
                .is_some();
            if let (Some(ef_search), false) = (ef_search_val, has_ef) {
                // HnswSqQueryParam path: use ef_search as the ef parameter
                Some(HnswQueryParam {
                    ef: ef_search,
                    nprobe: 0,
                    is_using_refiner: false,
                })
            } else {
                Some(HnswQueryParam {
                    ef: param
                        .getattr("ef")
                        .ok()
                        .and_then(|value| value.extract::<usize>().ok())
                        .unwrap_or(0),
                    nprobe: param
                        .getattr("nprobe")
                        .ok()
                        .and_then(|value| value.extract::<usize>().ok())
                        .unwrap_or(0),
                    is_using_refiner: param
                        .getattr("is_using_refiner")
                        .ok()
                        .and_then(|value| value.extract::<bool>().ok())
                        .unwrap_or(false),
                })
            }
        }
        _ => None,
    };

    Ok(VectorQuery {
        field_name,
        vector,
        param,
    })
}

#[cfg(feature = "python-binding")]
fn py_query_context_to_core(
    py: Python<'_>,
    collection: &Collection,
    context: &Bound<'_, PyAny>,
) -> PyResult<CoreQueryContext> {
    let top_k = context.getattr("top_k")?.extract::<usize>()?;
    let queries = context.getattr("queries")?;
    let query_objects: Vec<Py<PyAny>> = queries.extract()?;
    let mut core_queries = Vec::with_capacity(query_objects.len());
    for query_object in query_objects {
        let query = py_vector_query_from_pyany(&query_object.bind(py))?;
        let vector = match query.vector {
            QueryVector::Dense(v) => hannsdb_core::query::QueryVector::Dense(v),
            QueryVector::Sparse(sv) => hannsdb_core::query::QueryVector::Sparse(
                hannsdb_core::document::SparseVector::new(sv.indices, sv.values),
            ),
        };
        core_queries.push(CoreVectorQuery {
            field_name: query.field_name,
            vector,
            param: query.param.map(|param| CoreVectorQueryParam {
                ef_search: Some(param.ef),
                nprobe: if param.nprobe > 0 {
                    Some(param.nprobe)
                } else {
                    None
                },
            }),
        });
    }

    let query_by_id_attr = context.getattr("query_by_id")?;
    let query_by_id = parse_query_ids(py, collection, &query_by_id_attr)?;
    let query_by_id_field_name = context
        .getattr("query_by_id_field_name")?
        .extract::<Option<String>>()?;
    let filter = context.getattr("filter")?.extract::<Option<String>>()?;
    let output_fields = context
        .getattr("output_fields")?
        .extract::<Option<Vec<String>>>()?;
    let include_vector = context.getattr("include_vector")?.extract::<bool>()?;

    let reranker_attr = context.getattr("reranker")?;
    let reranker = if reranker_attr.is_none() {
        None
    } else {
        let reranker_type = reranker_attr.getattr("__class__")?;
        let type_name = reranker_type.getattr("__name__")?.extract::<String>()?;
        match type_name.as_str() {
            "RrfReRanker" => {
                let rank_constant = reranker_attr.getattr("rank_constant")?.extract::<u64>()?;
                Some(CoreQueryReranker::Rrf { rank_constant })
            }
            "WeightedReRanker" => {
                let weights = reranker_attr
                    .getattr("weights")?
                    .extract::<std::collections::HashMap<String, f64>>()?;
                let metric = reranker_attr
                    .getattr("metric")?
                    .extract::<Option<String>>()?;
                Some(CoreQueryReranker::Weighted {
                    weights: weights.into_iter().collect(),
                    metric,
                })
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unsupported reranker type: {type_name}"
                )));
            }
        }
    };

    let group_by_attr = context.getattr("group_by")?;
    let group_by = match group_by_attr {
        value if value.is_none() => None,
        value => Some(CoreQueryGroupBy {
            field_name: value.getattr("field_name")?.extract::<String>()?,
            group_topk: value
                .getattr("group_topk")
                .ok()
                .and_then(|v| v.extract::<usize>().ok())
                .unwrap_or(0),
            group_count: value
                .getattr("group_count")
                .ok()
                .and_then(|v| v.extract::<usize>().ok())
                .unwrap_or(0),
        }),
    };

    let order_by_attr = context.getattr("order_by")?;
    let order_by = match order_by_attr {
        value if value.is_none() => None,
        value => Some(CoreOrderBy {
            field_name: value.getattr("field_name")?.extract::<String>()?,
            descending: value
                .getattr("descending")
                .ok()
                .and_then(|v| v.extract::<bool>().ok())
                .unwrap_or(false),
        }),
    };

    Ok(CoreQueryContext {
        top_k,
        queries: core_queries,
        query_by_id,
        query_by_id_field_name,
        filter,
        output_fields,
        include_vector,
        group_by,
        reranker,
        order_by,
    })
}

fn core_document_from_doc(
    doc: &Doc,
    primary_vector_name: &str,
    internal_id: i64,
) -> std::io::Result<CoreDocument> {
    let _primary_vector = doc.vectors.get(primary_vector_name).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("doc must contain a vector named '{primary_vector_name}'"),
        )
    })?;

    Ok(CoreDocument {
        id: internal_id,
        fields: doc.fields.clone(),
        vectors: doc.vectors.clone(),
        sparse_vectors: Default::default(),
    })
}

fn doc_from_core_document(
    document: CoreDocument,
    primary_vector_name: &str,
    public_id: String,
) -> Doc {
    let fields = document.fields.clone();
    let vectors = document.vectors_with_primary(primary_vector_name).clone();
    Doc {
        id: public_id,
        score: None,
        fields,
        vectors,
        field_name: primary_vector_name.to_string(),
        group_key: None,
    }
}

fn select_output_fields(
    fields: &BTreeMap<String, FieldValue>,
    output_fields: &Option<Vec<String>>,
) -> BTreeMap<String, FieldValue> {
    match output_fields {
        None => fields.clone(),
        Some(names) => names
            .iter()
            .filter_map(|name| fields.get(name).cloned().map(|value| (name.clone(), value)))
            .collect(),
    }
}

fn public_ids_for_internal_ids(
    collection: &Collection,
    internal_ids: &[i64],
) -> std::io::Result<Vec<String>> {
    collection
        .db
        .display_primary_keys_for_document_ids(&collection.collection_name, internal_ids)
}

fn core_documents_from_docs(
    collection: &mut Collection,
    docs: &[Doc],
) -> std::io::Result<Vec<(String, CoreDocument)>> {
    let collection_metadata = collection_metadata_from_root(
        std::path::Path::new(&collection.path),
        &collection.collection_name,
    )?;
    let docs = docs
        .iter()
        .map(|doc| normalize_doc_fields_for_collection(doc, &collection_metadata))
        .collect::<std::io::Result<Vec<_>>>()?;
    let public_ids = docs.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>();
    let internal_ids = collection
        .db
        .assign_internal_ids_for_primary_keys(&collection.collection_name, &public_ids)?;

    docs.iter()
        .zip(public_ids)
        .zip(internal_ids)
        .map(|((doc, public_id), internal_id)| {
            core_document_from_doc(doc, &collection.primary_vector_name, internal_id)
                .map(|document| (public_id, document))
        })
        .collect()
}

fn normalize_doc_fields_for_collection(
    doc: &Doc,
    metadata: &CollectionMetadata,
) -> std::io::Result<Doc> {
    Ok(Doc {
        id: doc.id.clone(),
        score: doc.score,
        fields: normalize_fields_for_collection(&doc.fields, metadata)?,
        vectors: doc.vectors.clone(),
        field_name: doc.field_name.clone(),
        group_key: doc.group_key.clone(),
    })
}

fn normalize_fields_for_collection(
    fields: &BTreeMap<String, FieldValue>,
    metadata: &CollectionMetadata,
) -> std::io::Result<BTreeMap<String, FieldValue>> {
    fields
        .iter()
        .map(|(name, value)| {
            let normalized = match metadata.fields.iter().find(|field| field.name == *name) {
                Some(field_schema) => coerce_field_value_for_schema(name, value, field_schema)?,
                None => value.clone(),
            };
            Ok((name.clone(), normalized))
        })
        .collect()
}

fn coerce_field_value_for_schema(
    field_name: &str,
    value: &FieldValue,
    schema: &CoreScalarFieldSchema,
) -> std::io::Result<FieldValue> {
    if schema.array {
        let FieldValue::Array(items) = value else {
            return Err(schema_mismatch_error(field_name, &schema.data_type));
        };
        let element_schema =
            CoreScalarFieldSchema::new(field_name.to_string(), schema.data_type.clone())
                .with_flags(schema.nullable, false);
        return items
            .iter()
            .map(|item| coerce_field_value_for_schema(field_name, item, &element_schema))
            .collect::<std::io::Result<Vec<_>>>()
            .map(FieldValue::Array);
    }

    match schema.data_type {
        CoreFieldType::String => match value {
            FieldValue::String(v) => Ok(FieldValue::String(v.clone())),
            _ => Err(schema_mismatch_error(field_name, &schema.data_type)),
        },
        CoreFieldType::Int64 => match value {
            FieldValue::Int64(v) => Ok(FieldValue::Int64(*v)),
            FieldValue::Int32(v) => Ok(FieldValue::Int64((*v).into())),
            FieldValue::UInt32(v) => Ok(FieldValue::Int64((*v).into())),
            FieldValue::UInt64(v) => i64::try_from(*v)
                .map(FieldValue::Int64)
                .map_err(|_| schema_mismatch_error(field_name, &schema.data_type)),
            _ => Err(schema_mismatch_error(field_name, &schema.data_type)),
        },
        CoreFieldType::Int32 => match value {
            FieldValue::Int32(v) => Ok(FieldValue::Int32(*v)),
            FieldValue::Int64(v) => i32::try_from(*v)
                .map(FieldValue::Int32)
                .map_err(|_| schema_mismatch_error(field_name, &schema.data_type)),
            FieldValue::UInt32(v) => i32::try_from(*v)
                .map(FieldValue::Int32)
                .map_err(|_| schema_mismatch_error(field_name, &schema.data_type)),
            FieldValue::UInt64(v) => i32::try_from(*v)
                .map(FieldValue::Int32)
                .map_err(|_| schema_mismatch_error(field_name, &schema.data_type)),
            _ => Err(schema_mismatch_error(field_name, &schema.data_type)),
        },
        CoreFieldType::UInt32 => match value {
            FieldValue::UInt32(v) => Ok(FieldValue::UInt32(*v)),
            FieldValue::Int32(v) => u32::try_from(*v)
                .map(FieldValue::UInt32)
                .map_err(|_| schema_mismatch_error(field_name, &schema.data_type)),
            FieldValue::Int64(v) => u32::try_from(*v)
                .map(FieldValue::UInt32)
                .map_err(|_| schema_mismatch_error(field_name, &schema.data_type)),
            FieldValue::UInt64(v) => u32::try_from(*v)
                .map(FieldValue::UInt32)
                .map_err(|_| schema_mismatch_error(field_name, &schema.data_type)),
            _ => Err(schema_mismatch_error(field_name, &schema.data_type)),
        },
        CoreFieldType::UInt64 => match value {
            FieldValue::UInt64(v) => Ok(FieldValue::UInt64(*v)),
            FieldValue::UInt32(v) => Ok(FieldValue::UInt64((*v).into())),
            FieldValue::Int32(v) => u64::try_from(*v)
                .map(FieldValue::UInt64)
                .map_err(|_| schema_mismatch_error(field_name, &schema.data_type)),
            FieldValue::Int64(v) => u64::try_from(*v)
                .map(FieldValue::UInt64)
                .map_err(|_| schema_mismatch_error(field_name, &schema.data_type)),
            _ => Err(schema_mismatch_error(field_name, &schema.data_type)),
        },
        CoreFieldType::Float => match value {
            FieldValue::Float(v) => Ok(FieldValue::Float(*v)),
            FieldValue::Float64(v) => Ok(FieldValue::Float(*v as f32)),
            FieldValue::Int32(v) => Ok(FieldValue::Float(*v as f32)),
            FieldValue::Int64(v) => Ok(FieldValue::Float(*v as f32)),
            FieldValue::UInt32(v) => Ok(FieldValue::Float(*v as f32)),
            FieldValue::UInt64(v) => Ok(FieldValue::Float(*v as f32)),
            _ => Err(schema_mismatch_error(field_name, &schema.data_type)),
        },
        CoreFieldType::Float64 => match value {
            FieldValue::Float64(v) => Ok(FieldValue::Float64(*v)),
            FieldValue::Float(v) => Ok(FieldValue::Float64((*v).into())),
            FieldValue::Int32(v) => Ok(FieldValue::Float64((*v).into())),
            FieldValue::Int64(v) => Ok(FieldValue::Float64(*v as f64)),
            FieldValue::UInt32(v) => Ok(FieldValue::Float64((*v).into())),
            FieldValue::UInt64(v) => Ok(FieldValue::Float64(*v as f64)),
            _ => Err(schema_mismatch_error(field_name, &schema.data_type)),
        },
        CoreFieldType::Bool => match value {
            FieldValue::Bool(v) => Ok(FieldValue::Bool(*v)),
            _ => Err(schema_mismatch_error(field_name, &schema.data_type)),
        },
        CoreFieldType::VectorFp32 | CoreFieldType::VectorFp16 | CoreFieldType::VectorSparse => {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("field {field_name} cannot use vector type in scalar schema"),
            ))
        }
    }
}

fn schema_mismatch_error(field_name: &str, data_type: &CoreFieldType) -> std::io::Error {
    let expected = match data_type {
        CoreFieldType::String => "string",
        CoreFieldType::Int64 => "int64",
        CoreFieldType::Int32 => "int32",
        CoreFieldType::UInt32 => "uint32",
        CoreFieldType::UInt64 => "uint64",
        CoreFieldType::Float => "float",
        CoreFieldType::Float64 => "float64",
        CoreFieldType::Bool => "bool",
        CoreFieldType::VectorFp32 => "vector_fp32",
        CoreFieldType::VectorFp16 => "vector_fp16",
        CoreFieldType::VectorSparse => "vector_sparse",
    };
    std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        format!("field {field_name} had non-{expected} value"),
    )
}

fn docs_from_core_documents(
    collection: &Collection,
    documents: Vec<CoreDocument>,
) -> std::io::Result<Vec<Doc>> {
    let internal_ids = documents
        .iter()
        .map(|document| document.id)
        .collect::<Vec<_>>();
    let public_ids = public_ids_for_internal_ids(collection, &internal_ids)?;
    Ok(documents
        .into_iter()
        .zip(public_ids)
        .map(|(document, public_id)| {
            doc_from_core_document(document, &collection.primary_vector_name, public_id)
        })
        .collect())
}

fn collection_metadata_from_root(
    root: &std::path::Path,
    collection_name: &str,
) -> std::io::Result<CollectionMetadata> {
    CollectionMetadata::load_from_path(
        &root
            .join("collections")
            .join(collection_name)
            .join("collection.json"),
    )
}

static INDEX_CATALOG_TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn json_string(value: &str) -> String {
    format!("{value:?}")
}

fn metric_json(value: Option<MetricType>) -> String {
    match value {
        Some(metric) => json_string(metric_type_name(metric)),
        None => "null".to_string(),
    }
}

fn load_catalog_from_json(json: String) -> std::io::Result<IndexCatalog> {
    let temp_path = std::env::temp_dir().join(format!(
        "hannsdb_py_index_catalog_{}_{}.json",
        std::process::id(),
        INDEX_CATALOG_TMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&temp_path, json)?;
    let catalog = IndexCatalog::load_from_path(&temp_path);
    let _ = std::fs::remove_file(&temp_path);
    catalog
}

fn vector_index_catalog_json(field_name: &str, index_param: Option<&IndexParam>) -> String {
    let (kind, metric, params) = match index_param {
        Some(IndexParam::Flat(params)) => {
            ("flat", metric_json(params.metric_type), "{}".to_string())
        }
        Some(IndexParam::Hnsw(params)) => (
            "hnsw",
            metric_json(params.metric_type),
            match quantize_type_value(params.quantize_type) {
                Some(quantize_type) => format!(
                    r#"{{"m":{},"ef_construction":{},"quantize_type":{}}}"#,
                    params.m,
                    params.ef_construction,
                    json_string(quantize_type)
                ),
                None => format!(
                    r#"{{"m":{},"ef_construction":{}}}"#,
                    params.m, params.ef_construction
                ),
            },
        ),
        Some(IndexParam::HnswHvq(params)) => (
            "hnsw_hvq",
            metric_json(params.metric_type),
            format!(
                r#"{{"m":{},"m_max0":{},"ef_construction":{},"ef_search":{},"nbits":{}}}"#,
                params.m, params.m_max0, params.ef_construction, params.ef_search, params.nbits
            ),
        ),
        Some(IndexParam::HnswSq(params)) => (
            "hnsw_sq",
            metric_json(params.metric_type),
            format!(
                r#"{{"m":{},"ef_construction":{},"ef_search":{}}}"#,
                params.m, params.ef_construction, params.ef_search
            ),
        ),
        Some(IndexParam::Ivf(params)) => (
            "ivf",
            metric_json(params.metric_type),
            format!(r#"{{"nlist":{}}}"#, params.nlist),
        ),
        Some(IndexParam::IvfUsq(params)) => (
            "ivf_usq",
            metric_json(params.metric_type),
            format!(
                r#"{{"nlist":{},"bits_per_dim":{},"rotation_seed":{},"rerank_k":{},"use_high_accuracy_scan":{}}}"#,
                params.nlist,
                params.bits_per_dim,
                params.rotation_seed,
                params.rerank_k,
                params.use_high_accuracy_scan
            ),
        ),
        None => ("flat", "null".to_string(), "{}".to_string()),
    };

    format!(
        r#"{{"vector_indexes":[{{"field_name":{},"kind":{},"metric":{},"params":{}}}],"scalar_indexes":[]}}"#,
        json_string(field_name),
        json_string(kind),
        metric,
        params
    )
}

fn scalar_index_catalog_json(field_name: &str, index_param: Option<&InvertIndexParam>) -> String {
    let params = match index_param {
        Some(p) => format!(
            r#"{{"enable_range_optimization":{},"enable_extended_wildcard":{}}}"#,
            p.enable_range_optimization, p.enable_extended_wildcard
        ),
        None => "{}".to_string(),
    };
    format!(
        r#"{{"vector_indexes":[],"scalar_indexes":[{{"field_name":{},"kind":"inverted","params":{}}}]}}"#,
        json_string(field_name),
        params,
    )
}

pub fn create_and_open(
    path: impl Into<String>,
    schema: CollectionSchema,
    option: Option<CollectionOption>,
) -> std::io::Result<Collection> {
    let path = path.into();
    let root = std::path::Path::new(&path);
    let option = option.unwrap_or_default();
    let core_schema = core_schema_from_schema(&schema)?;
    if option.read_only {
        let mut write_db = hannsdb_core::db::HannsDb::open(root)?;
        write_db.create_collection_with_schema(&schema.name, &core_schema)?;
        let db = hannsdb_core::db::HannsDb::open_read_only(root)?;
        return Ok(Collection {
            path,
            collection_name: schema.name,
            primary_vector_name: core_schema.primary_vector_name().to_string(),
            option,
            db,
        });
    }
    let mut db = hannsdb_core::db::HannsDb::open(root)?;
    db.create_collection_with_schema(&schema.name, &core_schema)?;
    Ok(Collection {
        path,
        collection_name: schema.name,
        primary_vector_name: core_schema.primary_vector_name().to_string(),
        option,
        db,
    })
}

pub fn open(
    path: impl Into<String>,
    option: Option<CollectionOption>,
) -> std::io::Result<Collection> {
    let path = path.into();
    let root = std::path::Path::new(&path);
    let option = option.unwrap_or_default();
    let db = if option.read_only {
        hannsdb_core::db::HannsDb::open_read_only(root)?
    } else {
        hannsdb_core::db::HannsDb::open(root)?
    };
    let manifest =
        hannsdb_core::catalog::ManifestMetadata::load_from_path(&root.join("manifest.json"))?;
    let collection_name = manifest.collections.first().cloned().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "no collection registered in manifest",
        )
    })?;
    let metadata = collection_metadata_from_root(root, &collection_name)?;
    // Warm search cache on open so persisted HNSW deserialize cost is paid before
    // benchmark search windows start. Ignore errors for empty collections.
    if metadata.dimension > 0 {
        let warm_query = vec![0.0_f32; metadata.dimension];
        let _ = db.search_with_ef(&collection_name, &warm_query, 1, 1);
    }
    Ok(Collection {
        path,
        collection_name,
        primary_vector_name: metadata.primary_vector,
        option,
        db,
    })
}

impl Collection {
    #[cfg(feature = "python-binding")]
    fn query_ids_scores(
        &self,
        topk: usize,
        vectors: &VectorQuery,
    ) -> std::io::Result<Vec<(String, f32)>> {
        if vectors.field_name != self.primary_vector_name {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "query field '{}' does not match primary vector '{}'",
                    vectors.field_name, self.primary_vector_name
                ),
            ));
        }

        let dense = match &vectors.vector {
            QueryVector::Dense(v) => v,
            QueryVector::Sparse(_) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "query_ids_scores does not support sparse vectors, use query_context instead",
                ));
            }
        };

        let ef_search = vectors.param.as_ref().map_or(32, |param| param.ef).max(1);
        let hits = self
            .db
            .search_with_ef(&self.collection_name, dense, topk, ef_search)?;
        let public_ids =
            public_ids_for_internal_ids(self, &hits.iter().map(|hit| hit.id).collect::<Vec<_>>())?;
        Ok(hits
            .into_iter()
            .zip(public_ids)
            .map(|(hit, public_id)| (public_id, hit.distance))
            .collect())
    }

    pub fn insert(&mut self, docs: &[Doc]) -> std::io::Result<usize> {
        let keyed_documents = core_documents_from_docs(self, docs)?;
        self.db
            .insert_documents_with_primary_keys(&self.collection_name, &keyed_documents)
    }

    pub fn upsert(&mut self, docs: &[Doc]) -> std::io::Result<usize> {
        let keyed_documents = core_documents_from_docs(self, docs)?;
        self.db
            .upsert_documents_with_primary_keys(&self.collection_name, &keyed_documents)
    }

    pub fn fetch(&self, ids: &[String]) -> std::io::Result<Vec<Doc>> {
        let documents = self
            .db
            .fetch_documents_by_primary_keys(&self.collection_name, ids)?;
        docs_from_core_documents(self, documents)
    }

    pub fn delete(&mut self, ids: &[String]) -> std::io::Result<usize> {
        self.db.delete_by_primary_keys(&self.collection_name, ids)
    }

    pub fn flush(&self) -> std::io::Result<()> {
        self.db.flush_collection(&self.collection_name)
    }

    pub fn stats(&self) -> std::io::Result<CollectionStats> {
        let info = self.db.get_collection_info(&self.collection_name)?;
        Ok(CollectionStats {
            name: info.name,
            dimension: info.dimension,
            metric: info.metric,
            record_count: info.record_count,
            deleted_count: info.deleted_count,
            live_count: info.live_count,
            index_completeness: info.index_completeness,
        })
    }

    pub fn create_vector_index(
        &self,
        field_name: &str,
        index_param: Option<&IndexParam>,
    ) -> std::io::Result<()> {
        let mut catalog =
            load_catalog_from_json(vector_index_catalog_json(field_name, index_param))?;
        let descriptor = catalog.vector_indexes.pop().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "vector index descriptor is missing",
            )
        })?;
        self.db
            .create_vector_index(&self.collection_name, descriptor)
    }

    pub fn drop_vector_index(&self, field_name: &str) -> std::io::Result<()> {
        self.db.drop_vector_index(&self.collection_name, field_name)
    }

    pub fn list_vector_indexes(&self) -> std::io::Result<Vec<String>> {
        self.db
            .list_vector_indexes(&self.collection_name)
            .map(|descriptors| {
                descriptors
                    .into_iter()
                    .map(|descriptor| descriptor.field_name)
                    .collect()
            })
    }

    pub fn create_scalar_index(&self, field_name: &str) -> std::io::Result<()> {
        self.create_scalar_index_with_param(field_name, None)
    }

    pub fn create_scalar_index_with_param(
        &self,
        field_name: &str,
        index_param: Option<&InvertIndexParam>,
    ) -> std::io::Result<()> {
        let catalog_json = scalar_index_catalog_json(field_name, index_param);
        let mut catalog = load_catalog_from_json(catalog_json)?;
        let descriptor = catalog.scalar_indexes.pop().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "scalar index descriptor is missing",
            )
        })?;
        self.db
            .create_scalar_index(&self.collection_name, descriptor)
    }

    pub fn drop_scalar_index(&self, field_name: &str) -> std::io::Result<()> {
        self.db.drop_scalar_index(&self.collection_name, field_name)
    }

    pub fn list_scalar_indexes(&self) -> std::io::Result<Vec<String>> {
        self.db
            .list_scalar_indexes(&self.collection_name)
            .map(|descriptors| {
                descriptors
                    .into_iter()
                    .map(|descriptor| descriptor.field_name)
                    .collect()
            })
    }

    pub fn add_vector_field(
        &mut self,
        name: &str,
        data_type: &str,
        dimension: usize,
        index_param: Option<&IndexParam>,
    ) -> std::io::Result<()> {
        let core_data_type = match data_type.to_ascii_lowercase().as_str() {
            "vector_fp32" | "vectorfp32" => hannsdb_core::document::FieldType::VectorFp32,
            "vector_fp16" | "vectorfp16" => hannsdb_core::document::FieldType::VectorFp16,
            "vector_sparse" | "vectorsparse" => hannsdb_core::document::FieldType::VectorSparse,
            other => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("unsupported vector data type: {other}"),
                ))
            }
        };
        let core_index_param = match index_param {
            Some(IndexParam::Flat(_params)) => None,
            Some(IndexParam::Hnsw(params)) => Some(
                hannsdb_core::document::VectorIndexSchema::hnsw(
                    params.metric_type.map(metric_type_name),
                    params.m,
                    params.ef_construction,
                )
                .with_quantize_type(quantize_type_value(params.quantize_type)),
            ),
            Some(IndexParam::HnswHvq(params)) => {
                Some(hannsdb_core::document::VectorIndexSchema::hnsw_hvq(
                    params.metric_type.map(metric_type_name),
                    params.m,
                    params.m_max0,
                    params.ef_construction,
                    params.ef_search,
                    params.nbits,
                ))
            }
            Some(IndexParam::HnswSq(params)) => {
                Some(hannsdb_core::document::VectorIndexSchema::hnsw_sq(
                    params.metric_type.map(metric_type_name),
                    params.m,
                    params.ef_construction,
                    params.ef_search,
                ))
            }
            Some(IndexParam::Ivf(params)) => Some(hannsdb_core::document::VectorIndexSchema::ivf(
                params.metric_type.map(metric_type_name),
                params.nlist,
            )),
            Some(IndexParam::IvfUsq(params)) => {
                Some(hannsdb_core::document::VectorIndexSchema::ivf_usq(
                    params.metric_type.map(metric_type_name),
                    params.nlist,
                    params.bits_per_dim,
                    params.rotation_seed,
                    params.rerank_k,
                    params.use_high_accuracy_scan,
                ))
            }
            None => None,
        };
        let field = hannsdb_core::document::VectorFieldSchema {
            name: name.to_string(),
            data_type: core_data_type,
            dimension,
            index_param: core_index_param,
            bm25_params: None,
        };
        self.db.add_vector_field(&self.collection_name, field)
    }

    pub fn drop_vector_field(&mut self, field_name: &str) -> std::io::Result<()> {
        self.db.drop_vector_field(&self.collection_name, field_name)
    }

    pub fn query(
        &self,
        output_fields: Option<Vec<String>>,
        topk: usize,
        filter: Option<&str>,
        vectors: VectorQuery,
    ) -> std::io::Result<Vec<Doc>> {
        if vectors.field_name != self.primary_vector_name {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "query field '{}' does not match primary vector '{}'",
                    vectors.field_name, self.primary_vector_name
                ),
            ));
        }

        let dense = match &vectors.vector {
            QueryVector::Dense(v) => v.clone(),
            QueryVector::Sparse(_) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "query does not support sparse vectors, use query_context instead",
                ));
            }
        };

        if let Some(filter) = filter.map(str::trim).filter(|filter| !filter.is_empty()) {
            let hits =
                self.db
                    .query_documents(&self.collection_name, &dense, topk, Some(filter))?;
            let public_ids = public_ids_for_internal_ids(
                self,
                &hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
            )?;
            return Ok(hits
                .into_iter()
                .zip(public_ids)
                .map(|(hit, public_id)| Doc {
                    id: public_id,
                    score: Some(hit.distance),
                    fields: select_output_fields(&hit.fields, &output_fields),
                    vectors: BTreeMap::new(),
                    field_name: self.primary_vector_name.clone(),
                    group_key: None,
                })
                .collect());
        }

        let ef_search = vectors.param.as_ref().map_or(32, |param| param.ef).max(1);
        let hits = self
            .db
            .search_with_ef(&self.collection_name, &dense, topk, ef_search)?;
        let should_fetch_fields = output_fields
            .as_ref()
            .map_or(true, |fields| !fields.is_empty());
        if !should_fetch_fields {
            let public_ids = public_ids_for_internal_ids(
                self,
                &hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
            )?;
            return Ok(hits
                .into_iter()
                .zip(public_ids)
                .map(|(hit, public_id)| Doc {
                    id: public_id,
                    score: Some(hit.distance),
                    fields: BTreeMap::new(),
                    vectors: BTreeMap::new(),
                    field_name: self.primary_vector_name.clone(),
                    group_key: None,
                })
                .collect());
        }

        let fetched = self.db.fetch_documents(
            &self.collection_name,
            &hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        )?;
        let public_ids =
            public_ids_for_internal_ids(self, &hits.iter().map(|hit| hit.id).collect::<Vec<_>>())?;
        Ok(hits
            .into_iter()
            .zip(fetched)
            .zip(public_ids)
            .map(|((hit, document), public_id)| Doc {
                id: public_id,
                score: Some(hit.distance),
                fields: select_output_fields(&document.fields, &output_fields),
                vectors: BTreeMap::new(),
                field_name: self.primary_vector_name.clone(),
                group_key: None,
            })
            .collect())
    }

    #[cfg(feature = "python-binding")]
    pub fn query_context(&self, py: Python<'_>, context: &Bound<'_, PyAny>) -> PyResult<Vec<Doc>> {
        let core_context = py_query_context_to_core(py, self, context)?;
        let hits = self
            .db
            .query_with_context(&self.collection_name, &core_context)
            .map_err(io_to_py_err)?;
        let public_ids =
            public_ids_for_internal_ids(self, &hits.iter().map(|hit| hit.id).collect::<Vec<_>>())
                .map_err(io_to_py_err)?;
        Ok(hits
            .into_iter()
            .zip(public_ids)
            .map(|(hit, public_id)| Doc {
                id: public_id,
                score: Some(hit.distance),
                fields: hit.fields,
                vectors: hit.vectors,
                field_name: self.primary_vector_name.clone(),
                group_key: hit.group_key,
            })
            .collect())
    }

    pub fn optimize(&mut self) -> std::io::Result<()> {
        self.db.optimize_collection(&self.collection_name)
    }

    pub fn destroy(mut self) -> std::io::Result<()> {
        if self.option.read_only {
            return Ok(());
        }
        self.db.drop_collection(&self.collection_name)
    }
}

#[cfg(feature = "python-binding")]
fn io_to_py_err(error: std::io::Error) -> PyErr {
    match error.kind() {
        std::io::ErrorKind::InvalidInput | std::io::ErrorKind::AlreadyExists => {
            PyValueError::new_err(error.to_string())
        }
        std::io::ErrorKind::NotFound => PyFileNotFoundError::new_err(error.to_string()),
        std::io::ErrorKind::Unsupported => {
            pyo3::exceptions::PyNotImplementedError::new_err(format!("unsupported: {}", error))
        }
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}

#[cfg(feature = "python-binding")]
fn py_dict_to_fields(fields: &Bound<'_, PyDict>) -> PyResult<BTreeMap<String, FieldValue>> {
    let mut out = BTreeMap::new();
    for (key, value) in fields.iter() {
        let key = key.extract::<String>()?;
        let value = if value.is_instance_of::<PyBool>() {
            FieldValue::Bool(value.extract::<bool>()?)
        } else if let Ok(value) = value.extract::<String>() {
            FieldValue::String(value)
        } else if let Ok(value) = value.extract::<i32>() {
            FieldValue::Int32(value)
        } else if let Ok(value) = value.extract::<i64>() {
            FieldValue::Int64(value)
        } else if let Ok(value) = value.extract::<u32>() {
            FieldValue::UInt32(value)
        } else if let Ok(value) = value.extract::<u64>() {
            FieldValue::UInt64(value)
        } else if let Ok(value) = value.extract::<f32>() {
            FieldValue::Float(value)
        } else if let Ok(value) = value.extract::<f64>() {
            FieldValue::Float64(value)
        } else if let Ok(list) = value.downcast::<PyList>() {
            let items: Vec<FieldValue> = list
                .iter()
                .map(|item| {
                    if item.is_instance_of::<PyBool>() {
                        Ok(FieldValue::Bool(item.extract::<bool>()?))
                    } else if let Ok(v) = item.extract::<String>() {
                        Ok(FieldValue::String(v))
                    } else if let Ok(v) = item.extract::<i32>() {
                        Ok(FieldValue::Int32(v))
                    } else if let Ok(v) = item.extract::<i64>() {
                        Ok(FieldValue::Int64(v))
                    } else if let Ok(v) = item.extract::<u32>() {
                        Ok(FieldValue::UInt32(v))
                    } else if let Ok(v) = item.extract::<u64>() {
                        Ok(FieldValue::UInt64(v))
                    } else if let Ok(v) = item.extract::<f32>() {
                        Ok(FieldValue::Float(v))
                    } else if let Ok(v) = item.extract::<f64>() {
                        Ok(FieldValue::Float64(v))
                    } else {
                        Err(PyValueError::new_err(format!(
                            "unsupported array element for '{key}'"
                        )))
                    }
                })
                .collect::<PyResult<Vec<_>>>()?;
            FieldValue::Array(items)
        } else {
            return Err(PyValueError::new_err(format!(
                "unsupported field value for '{key}'"
            )));
        };
        out.insert(key, value);
    }
    Ok(out)
}

#[cfg(feature = "python-binding")]
fn py_dict_to_vectors(fields: &Bound<'_, PyDict>) -> PyResult<BTreeMap<String, Vec<f32>>> {
    let mut out = BTreeMap::new();
    for (key, value) in fields.iter() {
        let key = key.extract::<String>()?;
        let value = value.extract::<Vec<f32>>().map_err(|error| {
            PyValueError::new_err(format!("unsupported vector value for '{key}': {error}"))
        })?;
        out.insert(key, value);
    }
    Ok(out)
}

#[cfg(feature = "python-binding")]
fn field_value_to_py<'py>(py: Python<'py>, value: &FieldValue) -> PyResult<Bound<'py, PyAny>> {
    match value {
        FieldValue::String(v) => Ok(v.clone().into_pyobject(py)?.into_any()),
        FieldValue::Int64(v) => Ok(v.into_pyobject(py)?.into_any()),
        FieldValue::Int32(v) => Ok(v.into_pyobject(py)?.into_any()),
        FieldValue::UInt32(v) => Ok((*v as u64).into_pyobject(py)?.into_any()),
        FieldValue::UInt64(v) => Ok(v.into_pyobject(py)?.into_any()),
        FieldValue::Float(v) => Ok(v.into_pyobject(py)?.into_any()),
        FieldValue::Float64(v) => Ok(v.into_pyobject(py)?.into_any()),
        FieldValue::Bool(v) => Ok(PyBool::new(py, *v).to_owned().into_any()),
        FieldValue::Array(items) => {
            let list = PyList::new(
                py,
                items
                    .iter()
                    .map(|item| field_value_to_py(py, item))
                    .collect::<PyResult<Vec<_>>>()?,
            )?;
            Ok(list.into_any())
        }
    }
}

#[cfg(feature = "python-binding")]
fn fields_to_py_dict<'py>(
    py: Python<'py>,
    fields: &BTreeMap<String, FieldValue>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (name, value) in fields {
        match value {
            FieldValue::String(value) => dict.set_item(name, value)?,
            FieldValue::Int64(value) => dict.set_item(name, *value)?,
            FieldValue::Int32(value) => dict.set_item(name, *value)?,
            FieldValue::UInt32(value) => dict.set_item(name, *value)?,
            FieldValue::UInt64(value) => dict.set_item(name, *value)?,
            FieldValue::Float(value) => dict.set_item(name, *value)?,
            FieldValue::Float64(value) => dict.set_item(name, *value)?,
            FieldValue::Bool(value) => dict.set_item(name, *value)?,
            FieldValue::Array(_) => dict.set_item(name, field_value_to_py(py, value)?)?,
        }
    }
    Ok(dict)
}

#[cfg(feature = "python-binding")]
fn vectors_to_py_dict<'py>(
    py: Python<'py>,
    vectors: &BTreeMap<String, Vec<f32>>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (name, vector) in vectors {
        dict.set_item(name, vector.clone())?;
    }
    Ok(dict)
}

#[cfg(feature = "python-binding")]
fn parse_metric_type(value: &str) -> PyResult<MetricType> {
    match value.to_ascii_lowercase().as_str() {
        "l2" => Ok(MetricType::L2),
        "cosine" => Ok(MetricType::Cosine),
        "ip" => Ok(MetricType::Ip),
        other => Err(PyValueError::new_err(format!(
            "unsupported MetricType value: {other}"
        ))),
    }
}

#[cfg(feature = "python-binding")]
fn parse_quantize_type(value: &str) -> PyResult<QuantizeType> {
    match value.to_ascii_lowercase().as_str() {
        "undefined" => Ok(QuantizeType::Undefined),
        "fp16" => Ok(QuantizeType::Fp16),
        "int8" => Ok(QuantizeType::Int8),
        "int4" => Ok(QuantizeType::Int4),
        other => Err(PyValueError::new_err(format!(
            "unsupported QuantizeType value: {other}"
        ))),
    }
}

#[cfg(feature = "python-binding")]
fn parse_log_level(value: &str) -> PyResult<LogLevel> {
    match value.to_ascii_lowercase().as_str() {
        "debug" => Ok(LogLevel::Debug),
        "info" => Ok(LogLevel::Info),
        "warn" => Ok(LogLevel::Warn),
        "error" => Ok(LogLevel::Error),
        other => Err(PyValueError::new_err(format!(
            "unsupported LogLevel value: {other}"
        ))),
    }
}

#[cfg(feature = "python-binding")]
fn parse_data_type(value: &str) -> PyResult<DataType> {
    match value.to_ascii_lowercase().as_str() {
        "string" => Ok(DataType::String),
        "int64" => Ok(DataType::Int64),
        "int32" => Ok(DataType::Int32),
        "uint32" => Ok(DataType::UInt32),
        "uint64" => Ok(DataType::UInt64),
        "float" => Ok(DataType::Float),
        "float64" => Ok(DataType::Float64),
        "bool" => Ok(DataType::Bool),
        "vectorfp32" | "vector_fp32" => Ok(DataType::VectorFp32),
        other => Err(PyValueError::new_err(format!(
            "unsupported DataType value: {other}"
        ))),
    }
}

#[cfg(feature = "python-binding")]
fn normalize_storage_backend(value: &str) -> PyResult<&'static str> {
    match value.trim().to_ascii_lowercase().as_str() {
        "hannsdb" | "default" | "" => Ok("hannsdb"),
        "lance" => Ok("lance"),
        other => Err(PyValueError::new_err(format!(
            "unsupported storage backend: {other}"
        ))),
    }
}

#[cfg(feature = "python-binding")]
fn unsupported_add_column_constant_literal() -> PyErr {
    PyNotImplementedError::new_err("add_column expression supports only constant literals")
}

#[cfg(feature = "python-binding")]
fn constant_add_column_backfill(value: FieldValue) -> Option<AddColumnBackfill> {
    Some(AddColumnBackfill::Constant { value: Some(value) })
}

#[cfg(feature = "python-binding")]
fn is_canonical_digits(value: &str) -> bool {
    value == "0"
        || (!value.is_empty()
            && !value.starts_with('0')
            && value.chars().all(|c| c.is_ascii_digit()))
}

#[cfg(feature = "python-binding")]
fn is_supported_int_literal(expr: &str) -> bool {
    let digits = expr.strip_prefix('-').unwrap_or(expr);
    is_canonical_digits(digits)
}

#[cfg(feature = "python-binding")]
fn is_supported_float_literal(expr: &str) -> bool {
    let Some((whole, frac)) = expr.split_once('.') else {
        return false;
    };
    let whole_digits = whole.strip_prefix('-').unwrap_or(whole);
    is_canonical_digits(whole_digits)
        && !frac.is_empty()
        && frac.chars().all(|c| c.is_ascii_digit())
}

#[cfg(feature = "python-binding")]
fn parse_add_column_backfill(
    expression: &str,
    field: &CoreScalarFieldSchema,
) -> PyResult<Option<AddColumnBackfill>> {
    let expr = expression.trim();
    if expr.is_empty() {
        return Ok(None);
    }
    if field.array {
        return Err(PyNotImplementedError::new_err(
            "add_column expression does not support array fields yet",
        ));
    }

    if expr == "null" {
        if !field.nullable {
            return Err(PyValueError::new_err(
                "null expression requires a nullable field",
            ));
        }
        return Ok(Some(AddColumnBackfill::Constant { value: None }));
    }

    if expr == "true" || expr == "false" {
        if field.data_type != CoreFieldType::Bool {
            return Err(PyValueError::new_err(
                "boolean constant requires a bool destination field",
            ));
        }
        return Ok(Some(AddColumnBackfill::Constant {
            value: Some(FieldValue::Bool(expr == "true")),
        }));
    }

    if expr.starts_with('"') {
        if !expr.ends_with('"') || expr.len() < 2 {
            return Err(PyValueError::new_err("invalid string literal"));
        }
        let inner = &expr[1..expr.len() - 1];
        if inner.contains('\\') || inner.contains('"') {
            return Err(PyNotImplementedError::new_err(
                "add_column expression does not support string escapes or embedded quotes yet",
            ));
        }
        if field.data_type != CoreFieldType::String {
            return Err(PyValueError::new_err(
                "string constant requires a string destination field",
            ));
        }
        return Ok(constant_add_column_backfill(FieldValue::String(
            inner.to_string(),
        )));
    }

    if expr.starts_with('+') {
        return Err(unsupported_add_column_constant_literal());
    }
    if expr.contains('e') || expr.contains('E') {
        return Err(PyNotImplementedError::new_err(
            "add_column expression does not support scientific notation",
        ));
    }

    if expr.contains('.') {
        if !is_supported_float_literal(expr) {
            return Err(unsupported_add_column_constant_literal());
        }
        let value = expr
            .parse::<f64>()
            .map_err(|_| PyValueError::new_err("invalid float literal"))?;
        let field_value = match field.data_type {
            CoreFieldType::Float => FieldValue::Float(value as f32),
            CoreFieldType::Float64 => FieldValue::Float64(value),
            _ => {
                return Err(PyValueError::new_err(
                    "float constant requires a float destination field",
                ))
            }
        };
        return Ok(constant_add_column_backfill(field_value));
    }

    if expr.starts_with('-') || expr.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        if !is_supported_int_literal(expr) {
            return Err(unsupported_add_column_constant_literal());
        }
        let value = expr
            .parse::<i128>()
            .map_err(|_| PyValueError::new_err("invalid integer literal"))?;
        let field_value = match field.data_type {
            CoreFieldType::Int64 => FieldValue::Int64(
                i64::try_from(value)
                    .map_err(|_| PyValueError::new_err("int64 constant is out of range"))?,
            ),
            CoreFieldType::Int32 => FieldValue::Int32(
                i32::try_from(value)
                    .map_err(|_| PyValueError::new_err("int32 constant is out of range"))?,
            ),
            CoreFieldType::UInt32 => FieldValue::UInt32(
                u32::try_from(value)
                    .map_err(|_| PyValueError::new_err("uint32 constant is out of range"))?,
            ),
            CoreFieldType::UInt64 => FieldValue::UInt64(
                u64::try_from(value)
                    .map_err(|_| PyValueError::new_err("uint64 constant is out of range"))?,
            ),
            CoreFieldType::Float => FieldValue::Float(value as f32),
            CoreFieldType::Float64 => FieldValue::Float64(value as f64),
            _ => {
                return Err(PyValueError::new_err(
                    "numeric constant requires a numeric destination field",
                ))
            }
        };
        return Ok(constant_add_column_backfill(field_value));
    }

    Err(unsupported_add_column_constant_literal())
}

#[cfg(feature = "python-binding")]
fn py_field_schema_to_core(field_schema: &FieldSchema) -> PyResult<CoreScalarFieldSchema> {
    let core_type = match field_schema.data_type {
        DataType::String => hannsdb_core::document::FieldType::String,
        DataType::Int64 => hannsdb_core::document::FieldType::Int64,
        DataType::Int32 => hannsdb_core::document::FieldType::Int32,
        DataType::UInt32 => hannsdb_core::document::FieldType::UInt32,
        DataType::UInt64 => hannsdb_core::document::FieldType::UInt64,
        DataType::Float => hannsdb_core::document::FieldType::Float,
        DataType::Float64 => hannsdb_core::document::FieldType::Float64,
        DataType::Bool => hannsdb_core::document::FieldType::Bool,
        DataType::VectorFp32 => {
            return Err(PyNotImplementedError::new_err(
                "alter_column field_schema migration does not support vector fields yet",
            ))
        }
    };
    Ok(
        hannsdb_core::document::ScalarFieldSchema::new(field_schema.name.clone(), core_type)
            .with_flags(field_schema.nullable, field_schema.array),
    )
}

#[cfg(feature = "python-binding")]
fn classify_alter_column_migration(
    root: &std::path::Path,
    collection_name: &str,
    old_name: &str,
    new_name: &str,
    target_field: &FieldSchema,
) -> PyResult<AlterColumnMigration> {
    let metadata = collection_metadata_from_root(root, collection_name).map_err(io_to_py_err)?;
    let current_field = metadata
        .fields
        .iter()
        .find(|field| field.name == old_name)
        .ok_or_else(|| PyValueError::new_err(format!("field not found: {old_name}")))?;

    if !new_name.is_empty() && new_name != target_field.name {
        return Err(PyValueError::new_err(
            "alter_column new_name must match field_schema.name",
        ));
    }
    let is_rename = target_field.name != old_name;
    if target_field.name.is_empty() {
        return Err(PyNotImplementedError::new_err(
            "alter_column field_schema migration requires a non-empty target field name",
        ));
    }
    if target_field.nullable != current_field.nullable {
        return Err(PyNotImplementedError::new_err(
            "alter_column field_schema migration does not support nullable changes yet",
        ));
    }
    if target_field.array != current_field.array {
        return Err(PyNotImplementedError::new_err(
            "alter_column field_schema migration does not support array changes yet",
        ));
    }

    match (&current_field.data_type, &target_field.data_type) {
        (CoreFieldType::Int32, DataType::Int64) => Ok(if is_rename {
            AlterColumnMigration::RenameAndInt32ToInt64
        } else {
            AlterColumnMigration::Int32ToInt64
        }),
        (CoreFieldType::UInt32, DataType::UInt64) => Ok(if is_rename {
            AlterColumnMigration::RenameAndUInt32ToUInt64
        } else {
            AlterColumnMigration::UInt32ToUInt64
        }),
        (CoreFieldType::Float, DataType::Float64) => Ok(if is_rename {
            AlterColumnMigration::RenameAndFloatToFloat64
        } else {
            AlterColumnMigration::FloatToFloat64
        }),
        _ => Err(PyNotImplementedError::new_err(
            "alter_column field_schema migration supports only widening scalar conversions",
        )),
    }
}

#[cfg(feature = "python-binding")]
#[pyclass(name = "MetricType", module = "hannsdb")]
struct PyMetricType;

#[cfg(feature = "python-binding")]
#[pymethods]
#[allow(non_upper_case_globals)]
impl PyMetricType {
    #[classattr]
    const L2: &'static str = "l2";
    #[classattr]
    const Cosine: &'static str = "cosine";
    #[classattr]
    const Ip: &'static str = "ip";
}

#[cfg(feature = "python-binding")]
#[pyclass(name = "QuantizeType", module = "hannsdb")]
struct PyQuantizeType;

#[cfg(feature = "python-binding")]
#[pymethods]
#[allow(non_upper_case_globals)]
impl PyQuantizeType {
    #[classattr]
    const Undefined: &'static str = "undefined";
    #[classattr]
    const Fp16: &'static str = "fp16";
    #[classattr]
    const Int8: &'static str = "int8";
    #[classattr]
    const Int4: &'static str = "int4";
}

#[cfg(feature = "python-binding")]
#[pyclass(name = "LogLevel", module = "hannsdb")]
struct PyLogLevel;

#[cfg(feature = "python-binding")]
#[pymethods]
#[allow(non_upper_case_globals)]
impl PyLogLevel {
    #[classattr]
    const Debug: &'static str = "debug";
    #[classattr]
    const Info: &'static str = "info";
    #[classattr]
    const Warn: &'static str = "warn";
    #[classattr]
    const Error: &'static str = "error";
}

#[cfg(feature = "python-binding")]
#[pyclass(name = "DataType", module = "hannsdb")]
struct PyDataType;

#[cfg(feature = "python-binding")]
#[pymethods]
#[allow(non_upper_case_globals)]
impl PyDataType {
    #[classattr]
    const String: &'static str = "string";
    #[classattr]
    const Int64: &'static str = "int64";
    #[classattr]
    const Int32: &'static str = "int32";
    #[classattr]
    const UInt32: &'static str = "uint32";
    #[classattr]
    const UInt64: &'static str = "uint64";
    #[classattr]
    const Float: &'static str = "float";
    #[classattr]
    const Float64: &'static str = "float64";
    #[classattr]
    const Bool: &'static str = "bool";
    #[classattr]
    const VectorFp32: &'static str = "vector_fp32";
}

#[cfg(feature = "python-binding")]
#[pyclass(name = "OptimizeOption", module = "hannsdb")]
#[derive(Clone)]
#[allow(dead_code)]
struct PyOptimizeOption {
    inner: OptimizeOption,
}

#[cfg(feature = "python-binding")]
#[pymethods]
impl PyOptimizeOption {
    #[new]
    fn new() -> Self {
        Self {
            inner: OptimizeOption::default(),
        }
    }
}

#[cfg(feature = "python-binding")]
#[pyclass(name = "CollectionOption", module = "hannsdb")]
#[derive(Clone)]
struct PyCollectionOption {
    inner: CollectionOption,
}

#[cfg(feature = "python-binding")]
#[pymethods]
impl PyCollectionOption {
    #[new]
    #[pyo3(signature = (read_only=false, enable_mmap=true))]
    fn new(read_only: bool, enable_mmap: bool) -> Self {
        Self {
            inner: CollectionOption {
                read_only,
                enable_mmap,
            },
        }
    }

    #[getter]
    fn read_only(&self) -> bool {
        self.inner.read_only
    }

    #[getter]
    fn enable_mmap(&self) -> bool {
        self.inner.enable_mmap
    }
}

#[cfg(feature = "python-binding")]
#[pyclass(name = "AddColumnOption", module = "hannsdb")]
#[derive(Clone)]
struct PyAddColumnOption {
    inner: AddColumnOption,
}

#[cfg(feature = "python-binding")]
#[pymethods]
impl PyAddColumnOption {
    #[new]
    #[pyo3(signature = (concurrency=0))]
    fn new(concurrency: usize) -> Self {
        Self {
            inner: AddColumnOption { concurrency },
        }
    }

    #[getter]
    fn concurrency(&self) -> usize {
        self.inner.concurrency
    }
}

#[cfg(feature = "python-binding")]
#[pyclass(name = "AlterColumnOption", module = "hannsdb")]
#[derive(Clone)]
struct PyAlterColumnOption {
    inner: AlterColumnOption,
}

#[cfg(feature = "python-binding")]
#[pymethods]
impl PyAlterColumnOption {
    #[new]
    #[pyo3(signature = (concurrency=0))]
    fn new(concurrency: usize) -> Self {
        Self {
            inner: AlterColumnOption { concurrency },
        }
    }

    #[getter]
    fn concurrency(&self) -> usize {
        self.inner.concurrency
    }
}

#[cfg(feature = "python-binding")]
#[pyclass(name = "HnswIndexParam", module = "hannsdb")]
#[derive(Clone)]
struct PyHnswIndexParam {
    inner: HnswIndexParam,
}

#[cfg(feature = "python-binding")]
#[pymethods]
impl PyHnswIndexParam {
    #[new]
    #[pyo3(signature = (metric_type=None, m=16, ef_construction=64, quantize_type="undefined"))]
    fn new(
        metric_type: Option<String>,
        m: usize,
        ef_construction: usize,
        quantize_type: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: HnswIndexParam {
                metric_type: metric_type.as_deref().map(parse_metric_type).transpose()?,
                m,
                ef_construction,
                quantize_type: parse_quantize_type(quantize_type)?,
            },
        })
    }
}

#[cfg(feature = "python-binding")]
#[pyclass(name = "HnswHvqIndexParam", module = "hannsdb")]
#[derive(Clone)]
struct PyHnswHvqIndexParam {
    inner: HnswHvqIndexParam,
}

#[cfg(feature = "python-binding")]
#[pymethods]
impl PyHnswHvqIndexParam {
    #[new]
    #[pyo3(signature = (metric_type=Some("ip".to_string()), m=16, m_max0=32, ef_construction=100, ef_search=64, nbits=4))]
    fn new(
        metric_type: Option<String>,
        m: usize,
        m_max0: usize,
        ef_construction: usize,
        ef_search: usize,
        nbits: usize,
    ) -> PyResult<Self> {
        let parsed_metric = match metric_type.as_deref() {
            Some(value) => Some(parse_metric_type(value)?),
            None => Some(MetricType::Ip),
        };
        if parsed_metric != Some(MetricType::Ip) {
            return Err(PyValueError::new_err(
                "hnsw_hvq currently supports only metric_type='ip'",
            ));
        }
        Ok(Self {
            inner: HnswHvqIndexParam {
                metric_type: parsed_metric,
                m,
                m_max0,
                ef_construction,
                ef_search,
                nbits,
            },
        })
    }
}

#[cfg(feature = "python-binding")]
#[pyclass(name = "HnswSqIndexParam", module = "hannsdb")]
#[derive(Clone)]
struct PyHnswSqIndexParam {
    inner: HnswSqIndexParam,
}

#[cfg(feature = "python-binding")]
#[pymethods]
impl PyHnswSqIndexParam {
    #[new]
    #[pyo3(signature = (metric_type=None, m=16, ef_construction=200, ef_search=50))]
    fn new(
        metric_type: Option<String>,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
    ) -> PyResult<Self> {
        let parsed_metric = metric_type.as_deref().map(parse_metric_type).transpose()?;
        Ok(Self {
            inner: HnswSqIndexParam {
                metric_type: parsed_metric,
                m,
                ef_construction,
                ef_search,
            },
        })
    }
}

#[cfg(feature = "python-binding")]
#[pyclass(name = "FlatIndexParam", module = "hannsdb")]
#[derive(Clone)]
struct PyFlatIndexParam {
    inner: FlatIndexParam,
}

#[cfg(feature = "python-binding")]
#[pymethods]
impl PyFlatIndexParam {
    #[new]
    #[pyo3(signature = (metric_type=None))]
    fn new(metric_type: Option<String>) -> PyResult<Self> {
        Ok(Self {
            inner: FlatIndexParam {
                metric_type: metric_type.as_deref().map(parse_metric_type).transpose()?,
            },
        })
    }

    #[getter]
    fn metric_type(&self) -> Option<&'static str> {
        self.inner.metric_type.map(metric_type_name)
    }
}

#[cfg(feature = "python-binding")]
#[pyclass(name = "InvertIndexParam", module = "hannsdb")]
#[derive(Clone)]
struct PyInvertIndexParam {
    inner: InvertIndexParam,
}

#[cfg(feature = "python-binding")]
#[pymethods]
impl PyInvertIndexParam {
    #[new]
    #[pyo3(signature = (enable_range_optimization=false, enable_extended_wildcard=false))]
    fn new(enable_range_optimization: bool, enable_extended_wildcard: bool) -> PyResult<Self> {
        Ok(Self {
            inner: InvertIndexParam {
                enable_range_optimization,
                enable_extended_wildcard,
            },
        })
    }

    #[getter]
    fn enable_range_optimization(&self) -> bool {
        self.inner.enable_range_optimization
    }

    #[getter]
    fn enable_extended_wildcard(&self) -> bool {
        self.inner.enable_extended_wildcard
    }
}

#[cfg(feature = "python-binding")]
#[pyclass(name = "IVFIndexParam", module = "hannsdb")]
#[derive(Clone)]
struct PyIVFIndexParam {
    inner: IvfIndexParam,
}

#[cfg(feature = "python-binding")]
#[pymethods]
impl PyIVFIndexParam {
    #[new]
    #[pyo3(signature = (metric_type=None, nlist=1024))]
    fn new(metric_type: Option<String>, nlist: usize) -> PyResult<Self> {
        Ok(Self {
            inner: IvfIndexParam {
                metric_type: metric_type.as_deref().map(parse_metric_type).transpose()?,
                nlist,
            },
        })
    }
}

#[cfg(feature = "python-binding")]
#[pyclass(name = "IvfUsqIndexParam", module = "hannsdb")]
#[derive(Clone)]
struct PyIvfUsqIndexParam {
    inner: IvfUsqIndexParam,
}

#[cfg(feature = "python-binding")]
#[pymethods]
impl PyIvfUsqIndexParam {
    #[new]
    #[pyo3(signature = (metric_type=None, nlist=1024, bits_per_dim=4, rotation_seed=42, rerank_k=64, use_high_accuracy_scan=false))]
    fn new(
        metric_type: Option<String>,
        nlist: usize,
        bits_per_dim: usize,
        rotation_seed: usize,
        rerank_k: usize,
        use_high_accuracy_scan: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: IvfUsqIndexParam {
                metric_type: metric_type.as_deref().map(parse_metric_type).transpose()?,
                nlist,
                bits_per_dim,
                rotation_seed,
                rerank_k,
                use_high_accuracy_scan,
            },
        })
    }

    #[getter]
    fn metric_type(&self) -> Option<&'static str> {
        self.inner.metric_type.map(metric_type_name)
    }

    #[getter]
    fn nlist(&self) -> usize {
        self.inner.nlist
    }

    #[getter]
    fn bits_per_dim(&self) -> usize {
        self.inner.bits_per_dim
    }

    #[getter]
    fn rotation_seed(&self) -> usize {
        self.inner.rotation_seed
    }

    #[getter]
    fn rerank_k(&self) -> usize {
        self.inner.rerank_k
    }

    #[getter]
    fn use_high_accuracy_scan(&self) -> bool {
        self.inner.use_high_accuracy_scan
    }
}

#[cfg(feature = "python-binding")]
#[pyclass(name = "HnswQueryParam", module = "hannsdb")]
#[derive(Clone)]
struct PyHnswQueryParam {
    inner: HnswQueryParam,
}

#[cfg(feature = "python-binding")]
#[pymethods]
impl PyHnswQueryParam {
    #[new]
    #[pyo3(signature = (ef=32, nprobe=0, is_using_refiner=false))]
    fn new(ef: usize, nprobe: usize, is_using_refiner: bool) -> Self {
        Self {
            inner: HnswQueryParam {
                ef,
                nprobe,
                is_using_refiner,
            },
        }
    }

    #[getter]
    fn ef(&self) -> usize {
        self.inner.ef
    }

    #[getter]
    fn nprobe(&self) -> usize {
        self.inner.nprobe
    }

    #[getter]
    fn is_using_refiner(&self) -> bool {
        self.inner.is_using_refiner
    }
}

#[cfg(feature = "python-binding")]
#[pyclass(name = "IVFQueryParam", module = "hannsdb")]
#[derive(Clone)]
struct PyIVFQueryParam {
    inner: IvfQueryParam,
}

#[cfg(feature = "python-binding")]
#[pymethods]
impl PyIVFQueryParam {
    #[new]
    #[pyo3(signature = (nprobe=1))]
    fn new(nprobe: usize) -> Self {
        Self {
            inner: IvfQueryParam { nprobe },
        }
    }

    #[getter]
    fn nprobe(&self) -> usize {
        self.inner.nprobe
    }
}

#[cfg(feature = "python-binding")]
#[pyclass(name = "IvfUsqQueryParam", module = "hannsdb")]
#[derive(Clone)]
struct PyIvfUsqQueryParam {
    inner: IvfUsqQueryParam,
}

#[cfg(feature = "python-binding")]
#[pymethods]
impl PyIvfUsqQueryParam {
    #[new]
    #[pyo3(signature = (nprobe=1))]
    fn new(nprobe: usize) -> Self {
        Self {
            inner: IvfUsqQueryParam { nprobe },
        }
    }

    #[getter]
    fn nprobe(&self) -> usize {
        self.inner.nprobe
    }
}

#[cfg(feature = "python-binding")]
#[pyclass(name = "HnswSqQueryParam", module = "hannsdb")]
#[derive(Clone)]
struct PyHnswSqQueryParam {
    inner: HnswSqQueryParam,
}

#[cfg(feature = "python-binding")]
#[pymethods]
impl PyHnswSqQueryParam {
    #[new]
    #[pyo3(signature = (ef_search=50))]
    fn new(ef_search: usize) -> Self {
        Self {
            inner: HnswSqQueryParam { ef_search },
        }
    }

    #[getter]
    fn ef_search(&self) -> usize {
        self.inner.ef_search
    }
}

#[cfg(feature = "python-binding")]
#[pyclass(name = "HnswHvqQueryParam", module = "hannsdb")]
#[derive(Clone)]
struct PyHnswHvqQueryParam {
    inner: HnswHvqQueryParam,
}

#[cfg(feature = "python-binding")]
#[pymethods]
impl PyHnswHvqQueryParam {
    #[new]
    #[pyo3(signature = (ef_search=50))]
    fn new(ef_search: usize) -> Self {
        Self {
            inner: HnswHvqQueryParam { ef_search },
        }
    }

    #[getter]
    fn ef_search(&self) -> usize {
        self.inner.ef_search
    }
}

#[cfg(feature = "python-binding")]
#[pyclass(name = "IndexOption", module = "hannsdb")]
#[derive(Clone)]
struct PyIndexOption {
    inner: IndexOption,
}

#[cfg(feature = "python-binding")]
#[pymethods]
impl PyIndexOption {
    #[new]
    #[pyo3(signature = (concurrency=0))]
    fn new(concurrency: usize) -> Self {
        Self {
            inner: IndexOption { concurrency },
        }
    }

    #[getter]
    fn concurrency(&self) -> usize {
        self.inner.concurrency
    }
}

#[cfg(feature = "python-binding")]
#[pyclass(name = "FieldSchema", module = "hannsdb")]
#[derive(Clone)]
struct PyFieldSchema {
    inner: FieldSchema,
}

#[cfg(feature = "python-binding")]
#[pymethods]
impl PyFieldSchema {
    #[new]
    #[pyo3(signature = (name, data_type, nullable=false, array=false))]
    fn new(name: String, data_type: String, nullable: bool, array: bool) -> PyResult<Self> {
        Ok(Self {
            inner: FieldSchema {
                name,
                data_type: parse_data_type(&data_type)?,
                nullable,
                array,
            },
        })
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn data_type(&self) -> &'static str {
        match self.inner.data_type {
            DataType::String => "string",
            DataType::Int64 => "int64",
            DataType::Int32 => "int32",
            DataType::UInt32 => "uint32",
            DataType::UInt64 => "uint64",
            DataType::Float => "float",
            DataType::Float64 => "float64",
            DataType::Bool => "bool",
            DataType::VectorFp32 => "vector_fp32",
        }
    }

    #[getter]
    fn nullable(&self) -> bool {
        self.inner.nullable
    }

    #[getter]
    fn array(&self) -> bool {
        self.inner.array
    }
}

#[cfg(feature = "python-binding")]
#[pyclass(name = "VectorSchema", module = "hannsdb")]
#[derive(Clone)]
struct PyVectorSchema {
    inner: VectorSchema,
}

#[cfg(feature = "python-binding")]
#[pymethods]
impl PyVectorSchema {
    #[new]
    #[pyo3(signature = (name, data_type, dimension, index_param=None))]
    fn new(
        py: Python<'_>,
        name: String,
        data_type: String,
        dimension: usize,
        index_param: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let index_param = match index_param {
            Some(param) => {
                let bound = param.bind(py);
                if bound.is_instance_of::<PyFlatIndexParam>() {
                    Some(IndexParam::Flat(
                        bound
                            .extract::<PyRef<'_, PyFlatIndexParam>>()?
                            .inner
                            .clone(),
                    ))
                } else if bound.is_instance_of::<PyHnswIndexParam>() {
                    Some(IndexParam::Hnsw(
                        bound
                            .extract::<PyRef<'_, PyHnswIndexParam>>()?
                            .inner
                            .clone(),
                    ))
                } else if bound.is_instance_of::<PyHnswHvqIndexParam>() {
                    Some(IndexParam::HnswHvq(
                        bound
                            .extract::<PyRef<'_, PyHnswHvqIndexParam>>()?
                            .inner
                            .clone(),
                    ))
                } else if bound.is_instance_of::<PyHnswSqIndexParam>() {
                    Some(IndexParam::HnswSq(
                        bound
                            .extract::<PyRef<'_, PyHnswSqIndexParam>>()?
                            .inner
                            .clone(),
                    ))
                } else if bound.is_instance_of::<PyIVFIndexParam>() {
                    Some(IndexParam::Ivf(
                        bound.extract::<PyRef<'_, PyIVFIndexParam>>()?.inner.clone(),
                    ))
                } else if bound.is_instance_of::<PyIvfUsqIndexParam>() {
                    Some(IndexParam::IvfUsq(
                        bound
                            .extract::<PyRef<'_, PyIvfUsqIndexParam>>()?
                            .inner
                            .clone(),
                    ))
                } else {
                    return Err(PyValueError::new_err(
                        "index_param must be FlatIndexParam, HnswIndexParam, HnswHvqIndexParam, HnswSqIndexParam, IVFIndexParam, or IvfUsqIndexParam",
                    ));
                }
            }
            None => None,
        };
        Ok(Self {
            inner: VectorSchema {
                name,
                data_type: parse_data_type(&data_type)?,
                dimension,
                index_param,
            },
        })
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn data_type(&self) -> &'static str {
        match self.inner.data_type {
            DataType::String => "string",
            DataType::Int64 => "int64",
            DataType::Int32 => "int32",
            DataType::UInt32 => "uint32",
            DataType::UInt64 => "uint64",
            DataType::Float => "float",
            DataType::Float64 => "float64",
            DataType::Bool => "bool",
            DataType::VectorFp32 => "vector_fp32",
        }
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension
    }
}

#[cfg(feature = "python-binding")]
#[pyclass(name = "CollectionSchema", module = "hannsdb")]
#[derive(Clone)]
struct PyCollectionSchema {
    inner: CollectionSchema,
}

#[cfg(feature = "python-binding")]
#[pymethods]
impl PyCollectionSchema {
    #[new]
    #[pyo3(signature = (name, vector_schema=None, fields=None, vectors=None, primary_vector=None))]
    fn new(
        py: Python<'_>,
        name: String,
        vector_schema: Option<Py<PyVectorSchema>>,
        fields: Option<Vec<Py<PyFieldSchema>>>,
        vectors: Option<Vec<Py<PyVectorSchema>>>,
        primary_vector: Option<String>,
    ) -> PyResult<Self> {
        let mut inner_vectors = Vec::new();
        if let Some(vector_schema) = vector_schema {
            inner_vectors.push(vector_schema.borrow(py).inner.clone());
        }
        if let Some(vectors) = vectors {
            inner_vectors.extend(
                vectors
                    .into_iter()
                    .map(|vector| vector.borrow(py).inner.clone()),
            );
        }
        if inner_vectors.is_empty() {
            return Err(PyValueError::new_err(
                "CollectionSchema requires at least one vector schema",
            ));
        }
        let primary_vector = primary_vector.unwrap_or_else(|| {
            inner_vectors
                .first()
                .map(|vector| vector.name.clone())
                .unwrap_or_else(|| "vector".to_string())
        });
        let inner_fields = fields
            .unwrap_or_default()
            .into_iter()
            .map(|field| field.borrow(py).inner.clone())
            .collect();
        Ok(Self {
            inner: CollectionSchema {
                name,
                primary_vector,
                fields: inner_fields,
                vectors: inner_vectors,
            },
        })
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn primary_vector(&self) -> String {
        self.inner.primary_vector.clone()
    }

    #[getter]
    fn fields(&self, py: Python<'_>) -> PyResult<Vec<Py<PyFieldSchema>>> {
        self.inner
            .fields
            .iter()
            .cloned()
            .map(|field| Py::new(py, PyFieldSchema { inner: field }))
            .collect()
    }

    #[getter]
    fn vectors(&self, py: Python<'_>) -> PyResult<Vec<Py<PyVectorSchema>>> {
        self.inner
            .vectors
            .iter()
            .cloned()
            .map(|vector| Py::new(py, PyVectorSchema { inner: vector }))
            .collect()
    }
}

#[cfg(feature = "python-binding")]
#[pyclass(name = "Doc", module = "hannsdb")]
#[derive(Clone)]
struct PyDoc {
    inner: Doc,
}

#[cfg(feature = "python-binding")]
#[pymethods]
impl PyDoc {
    #[new]
    #[pyo3(signature = (id, vector=None, field_name="dense", fields=None, score=None, vectors=None))]
    fn new(
        id: String,
        vector: Option<Vec<f32>>,
        field_name: &str,
        fields: Option<Bound<'_, PyDict>>,
        score: Option<f32>,
        vectors: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let mut inner_vectors = vectors
            .as_ref()
            .map(py_dict_to_vectors)
            .transpose()?
            .unwrap_or_default();
        if let Some(vector) = vector {
            inner_vectors.insert(field_name.to_string(), vector);
        }
        Ok(Self {
            inner: {
                let field_name = if inner_vectors.is_empty() {
                    field_name.to_string()
                } else if inner_vectors.contains_key(field_name) {
                    field_name.to_string()
                } else {
                    inner_vectors
                        .keys()
                        .next()
                        .cloned()
                        .unwrap_or_else(|| field_name.to_string())
                };
                Doc {
                    id,
                    score,
                    fields: fields
                        .as_ref()
                        .map(py_dict_to_fields)
                        .transpose()?
                        .unwrap_or_default(),
                    vectors: inner_vectors,
                    field_name,
                    group_key: None,
                }
            },
        })
    }

    #[getter]
    fn id(&self) -> String {
        self.inner.id.clone()
    }

    #[getter]
    fn score(&self) -> Option<f32> {
        self.inner.score
    }

    #[getter]
    fn fields<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        fields_to_py_dict(py, &self.inner.fields)
    }

    #[getter]
    fn vectors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        vectors_to_py_dict(py, &self.inner.vectors)
    }

    #[getter]
    fn field_name(&self) -> String {
        self.inner.field_name.clone()
    }

    #[getter]
    fn group_key<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match &self.inner.group_key {
            Some(value) => Ok(Some(field_value_to_py(py, value)?)),
            None => Ok(None),
        }
    }
}
#[cfg(feature = "python-binding")]
#[pyclass(name = "SparseVector", module = "hannsdb")]
#[derive(Clone)]
struct PySparseVector {
    inner: SparseVectorData,
}

#[cfg(feature = "python-binding")]
#[pymethods]
impl PySparseVector {
    #[new]
    fn new(indices: Vec<u32>, values: Vec<f32>) -> PyResult<Self> {
        if indices.len() != values.len() {
            return Err(PyValueError::new_err(
                "indices and values must have the same length",
            ));
        }
        Ok(Self {
            inner: SparseVectorData { indices, values },
        })
    }

    #[getter]
    fn indices(&self) -> Vec<u32> {
        self.inner.indices.clone()
    }

    #[getter]
    fn values(&self) -> Vec<f32> {
        self.inner.values.clone()
    }
}

#[cfg(feature = "python-binding")]
#[pyclass(name = "VectorQuery", module = "hannsdb")]
#[derive(Clone)]
struct PyVectorQuery {
    inner: VectorQuery,
}

#[cfg(feature = "python-binding")]
#[pymethods]
impl PyVectorQuery {
    #[new]
    #[pyo3(signature = (field_name, vector, param=None))]
    fn new(
        py: Python<'_>,
        field_name: String,
        vector: Bound<'_, PyAny>,
        param: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let query_vector = if vector.hasattr("indices")? && vector.hasattr("values")? {
            let indices = vector.getattr("indices")?.extract::<Vec<u32>>()?;
            let values = vector.getattr("values")?.extract::<Vec<f32>>()?;
            QueryVector::Sparse(crate::SparseVectorData { indices, values })
        } else {
            let dense = vector.extract::<Vec<f32>>()?;
            QueryVector::Dense(dense)
        };
        let param = match param {
            Some(param) => {
                let bound = param.bind(py);
                if bound.is_instance_of::<PyHnswQueryParam>() {
                    Some(
                        bound
                            .extract::<PyRef<'_, PyHnswQueryParam>>()?
                            .inner
                            .clone(),
                    )
                } else if bound.is_instance_of::<PyIVFQueryParam>() {
                    let ivf = bound.extract::<PyRef<'_, PyIVFQueryParam>>()?.inner.clone();
                    Some(HnswQueryParam {
                        ef: 0,
                        nprobe: ivf.nprobe,
                        is_using_refiner: false,
                    })
                } else if bound.is_instance_of::<PyIvfUsqQueryParam>() {
                    let ivf_usq = bound
                        .extract::<PyRef<'_, PyIvfUsqQueryParam>>()?
                        .inner
                        .clone();
                    Some(HnswQueryParam {
                        ef: 0,
                        nprobe: ivf_usq.nprobe,
                        is_using_refiner: false,
                    })
                } else if bound.is_instance_of::<PyHnswSqQueryParam>() {
                    let sq = bound
                        .extract::<PyRef<'_, PyHnswSqQueryParam>>()?
                        .inner
                        .clone();
                    Some(HnswQueryParam {
                        ef: sq.ef_search,
                        nprobe: 0,
                        is_using_refiner: false,
                    })
                } else if bound.is_instance_of::<PyHnswHvqQueryParam>() {
                    let hvq = bound
                        .extract::<PyRef<'_, PyHnswHvqQueryParam>>()?
                        .inner
                        .clone();
                    Some(HnswQueryParam {
                        ef: hvq.ef_search,
                        nprobe: 0,
                        is_using_refiner: false,
                    })
                } else {
                    return Err(PyValueError::new_err(
                        "query param must be HnswQueryParam, IVFQueryParam, IvfUsqQueryParam, HnswSqQueryParam, or HnswHvqQueryParam",
                    ));
                }
            }
            None => None,
        };
        Ok(Self {
            inner: VectorQuery {
                field_name,
                vector: query_vector,
                param,
            },
        })
    }

    #[getter]
    fn field_name(&self) -> String {
        self.inner.field_name.clone()
    }

    #[getter]
    fn vector(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner.vector {
            QueryVector::Dense(v) => Ok(v.clone().into_pyobject(py)?.into_any().unbind()),
            QueryVector::Sparse(sv) => {
                let py_sparse = Py::new(py, PySparseVector { inner: sv.clone() })?;
                Ok(py_sparse.into_any())
            }
        }
    }

    #[getter]
    fn param(&self, py: Python<'_>) -> PyResult<Option<Py<PyHnswQueryParam>>> {
        match self.inner.param.as_ref() {
            Some(param) => Py::new(
                py,
                PyHnswQueryParam {
                    inner: param.clone(),
                },
            )
            .map(Some),
            None => Ok(None),
        }
    }
}

// ---------------------------------------------------------------------------
// Reranker Python classes
// ---------------------------------------------------------------------------

#[cfg(feature = "python-binding")]
#[pyclass(name = "RrfReRanker", module = "hannsdb")]
#[derive(Clone)]
struct PyRrfReRanker {
    rank_constant: u64,
}

#[cfg(feature = "python-binding")]
#[pymethods]
impl PyRrfReRanker {
    #[new]
    #[pyo3(signature = (rank_constant=60))]
    fn new(rank_constant: u64) -> Self {
        Self { rank_constant }
    }

    #[getter]
    fn rank_constant(&self) -> u64 {
        self.rank_constant
    }
}

#[cfg(feature = "python-binding")]
#[pyclass(name = "WeightedReRanker", module = "hannsdb")]
#[derive(Clone)]
struct PyWeightedReRanker {
    weights: std::collections::HashMap<String, f64>,
    metric: Option<String>,
}

#[cfg(feature = "python-binding")]
#[pymethods]
impl PyWeightedReRanker {
    #[new]
    #[pyo3(signature = (weights, metric=None))]
    fn new(weights: std::collections::HashMap<String, f64>, metric: Option<String>) -> Self {
        Self { weights, metric }
    }

    #[getter]
    fn weights(&self) -> std::collections::HashMap<String, f64> {
        self.weights.clone()
    }

    #[getter]
    fn metric(&self) -> Option<String> {
        self.metric.clone()
    }
}

#[cfg(feature = "python-binding")]
#[pyclass(name = "Collection", module = "hannsdb")]
struct PyCollection {
    inner: Option<Collection>,
}

#[cfg(all(feature = "python-binding", feature = "lance-storage"))]
#[pyclass(name = "LanceCollection", module = "hannsdb")]
struct PyLanceCollection {
    inner: Option<CoreLanceCollection>,
    schema: CollectionSchema,
}

#[cfg(feature = "python-binding")]
#[pyclass(name = "CollectionStats", module = "hannsdb")]
#[derive(Clone)]
struct PyCollectionStats {
    inner: CollectionStats,
}

#[cfg(all(feature = "python-binding", feature = "lance-storage"))]
fn block_on_lance<T>(f: impl std::future::Future<Output = std::io::Result<T>>) -> PyResult<T> {
    tokio::runtime::Runtime::new()
        .map_err(|err| PyRuntimeError::new_err(format!("failed to create tokio runtime: {err}")))?
        .block_on(f)
        .map_err(io_to_py_err)
}

#[cfg(all(feature = "python-binding", feature = "lance-storage"))]
fn core_documents_from_lance_docs(
    docs: &[Doc],
    schema: &CollectionSchema,
) -> std::io::Result<Vec<CoreDocument>> {
    let core_schema = core_schema_from_schema(schema)?;
    let metadata = CollectionMetadata::new_with_schema(schema.name.clone(), core_schema);
    docs.iter()
        .map(|doc| {
            let doc = normalize_doc_fields_for_collection(doc, &metadata)?;
            let id = doc.id.parse::<i64>().map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "Lance collection P4 requires numeric document ids, got '{}'",
                        doc.id
                    ),
                )
            })?;
            core_document_from_doc(&doc, &schema.primary_vector, id)
        })
        .collect()
}

#[cfg(all(feature = "python-binding", feature = "lance-storage"))]
fn py_docs_from_lance_core_documents(
    documents: Vec<CoreDocument>,
    primary_vector_name: &str,
) -> Vec<Doc> {
    documents
        .into_iter()
        .map(|document| {
            let public_id = document.id.to_string();
            doc_from_core_document(document, primary_vector_name, public_id)
        })
        .collect()
}

#[cfg(all(feature = "python-binding", feature = "lance-storage"))]
impl PyLanceCollection {
    fn inner_ref(&self) -> PyResult<&CoreLanceCollection> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Lance collection already destroyed"))
    }
}

#[cfg(all(feature = "python-binding", feature = "lance-storage"))]
#[pymethods]
impl PyLanceCollection {
    #[getter]
    fn name(&self) -> PyResult<String> {
        Ok(self.inner_ref()?.name().to_string())
    }

    #[getter]
    fn uri(&self) -> PyResult<String> {
        Ok(self.inner_ref()?.uri().to_string())
    }

    #[getter]
    fn schema(&self) -> PyCollectionSchema {
        PyCollectionSchema {
            inner: self.schema.clone(),
        }
    }

    fn insert(&self, py: Python<'_>, docs: Vec<Py<PyDoc>>) -> PyResult<usize> {
        let docs = docs
            .into_iter()
            .map(|doc| doc.borrow(py).inner.clone())
            .collect::<Vec<_>>();
        let docs = core_documents_from_lance_docs(&docs, &self.schema).map_err(io_to_py_err)?;
        block_on_lance(self.inner_ref()?.insert_documents(&docs)).map(|_| docs.len())
    }

    fn upsert(&self, py: Python<'_>, docs: Vec<Py<PyDoc>>) -> PyResult<usize> {
        let docs = docs
            .into_iter()
            .map(|doc| doc.borrow(py).inner.clone())
            .collect::<Vec<_>>();
        let docs = core_documents_from_lance_docs(&docs, &self.schema).map_err(io_to_py_err)?;
        block_on_lance(self.inner_ref()?.upsert_documents(&docs))
    }

    fn fetch(&self, py: Python<'_>, ids: Vec<String>) -> PyResult<Vec<Py<PyDoc>>> {
        let ids = ids
            .iter()
            .map(|id| {
                id.parse::<i64>().map_err(|_| {
                    PyValueError::new_err(format!(
                        "Lance collection P4 requires numeric document ids, got '{id}'"
                    ))
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let docs = block_on_lance(self.inner_ref()?.fetch_documents(&ids))?;
        py_docs_from_lance_core_documents(docs, &self.schema.primary_vector)
            .into_iter()
            .map(|inner| Py::new(py, PyDoc { inner }))
            .collect()
    }

    fn delete(&self, ids: Vec<String>) -> PyResult<usize> {
        let ids = ids
            .iter()
            .map(|id| {
                id.parse::<i64>().map_err(|_| {
                    PyValueError::new_err(format!(
                        "Lance collection P4 requires numeric document ids, got '{id}'"
                    ))
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        block_on_lance(self.inner_ref()?.delete_documents(&ids))
    }

    #[cfg(feature = "hanns-backend")]
    fn hanns_index_path(&self, field_name: &str) -> PyResult<String> {
        Ok(self
            .inner_ref()?
            .hanns_index_path(field_name)
            .to_string_lossy()
            .into_owned())
    }

    #[cfg(feature = "hanns-backend")]
    #[pyo3(signature = (field_name, metric="l2"))]
    fn optimize_hanns(&self, field_name: &str, metric: &str) -> PyResult<()> {
        block_on_lance(self.inner_ref()?.optimize_hanns(field_name, metric))
    }

    #[pyo3(signature = (vector, topk=10, metric="l2"))]
    fn search(
        &self,
        py: Python<'_>,
        vector: Vec<f32>,
        topk: usize,
        metric: &str,
    ) -> PyResult<Vec<Py<PyDoc>>> {
        let hits = block_on_lance(self.inner_ref()?.search(&vector, topk, metric))?;
        let ids = hits.iter().map(|hit| hit.id).collect::<Vec<_>>();
        let scores = hits
            .iter()
            .map(|hit| (hit.id, hit.distance))
            .collect::<BTreeMap<_, _>>();
        let docs = block_on_lance(self.inner_ref()?.fetch_documents(&ids))?;
        py_docs_from_lance_core_documents(docs, &self.schema.primary_vector)
            .into_iter()
            .map(|mut inner| {
                if let Ok(id) = inner.id.parse::<i64>() {
                    inner.score = scores.get(&id).copied();
                }
                Py::new(py, PyDoc { inner })
            })
            .collect()
    }

    fn destroy(&mut self) {
        self.inner = None;
    }
}

#[cfg(feature = "python-binding")]
#[pymethods]
impl PyCollectionStats {
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension
    }

    #[getter]
    fn metric(&self) -> String {
        self.inner.metric.clone()
    }

    #[getter]
    fn record_count(&self) -> usize {
        self.inner.record_count
    }

    #[getter]
    fn deleted_count(&self) -> usize {
        self.inner.deleted_count
    }

    #[getter]
    fn live_count(&self) -> usize {
        self.inner.live_count
    }

    /// Alias for live_count — mirrors CollectionStats.doc_count.
    #[getter]
    fn doc_count(&self) -> usize {
        self.inner.live_count
    }

    #[getter]
    fn index_completeness(&self) -> std::collections::HashMap<String, f64> {
        self.inner
            .index_completeness
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect()
    }
}

#[cfg(feature = "python-binding")]
impl PyCollection {
    fn inner_ref(&self) -> PyResult<&Collection> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("collection already destroyed"))
    }

    fn inner_mut(&mut self) -> PyResult<&mut Collection> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("collection already destroyed"))
    }
}

#[cfg(feature = "python-binding")]
#[pymethods]
impl PyCollection {
    #[getter]
    fn path(&self) -> PyResult<String> {
        Ok(self.inner_ref()?.path.clone())
    }

    #[getter]
    fn collection_name(&self) -> PyResult<String> {
        Ok(self.inner_ref()?.collection_name.clone())
    }

    #[getter]
    fn option(&self) -> PyResult<PyCollectionOption> {
        Ok(PyCollectionOption {
            inner: self.inner_ref()?.option.clone(),
        })
    }

    fn insert(&mut self, py: Python<'_>, docs: Vec<Py<PyDoc>>) -> PyResult<usize> {
        let docs = docs
            .into_iter()
            .map(|doc| doc.borrow(py).inner.clone())
            .collect::<Vec<_>>();
        self.inner_mut()?.insert(&docs).map_err(io_to_py_err)
    }

    fn upsert(&mut self, py: Python<'_>, docs: Vec<Py<PyDoc>>) -> PyResult<usize> {
        let docs = docs
            .into_iter()
            .map(|doc| doc.borrow(py).inner.clone())
            .collect::<Vec<_>>();
        self.inner_mut()?.upsert(&docs).map_err(io_to_py_err)
    }

    fn fetch(&self, py: Python<'_>, ids: Vec<String>) -> PyResult<Vec<Py<PyDoc>>> {
        let docs = self.inner_ref()?.fetch(&ids).map_err(io_to_py_err)?;
        docs.into_iter()
            .map(|doc| Py::new(py, PyDoc { inner: doc }))
            .collect()
    }

    fn delete(&mut self, ids: Vec<String>) -> PyResult<usize> {
        self.inner_mut()?.delete(&ids).map_err(io_to_py_err)
    }

    fn update(&mut self, py: Python<'_>, docs: Vec<Py<PyDoc>>) -> PyResult<usize> {
        let inner = self.inner_ref()?;
        let collection_name = inner.collection_name.clone();
        let collection_metadata =
            collection_metadata_from_root(std::path::Path::new(&inner.path), &collection_name)
                .map_err(io_to_py_err)?;
        let mut updates = Vec::with_capacity(docs.len());
        for doc in docs {
            let doc = doc.borrow(py);
            let normalized_fields =
                normalize_fields_for_collection(&doc.inner.fields, &collection_metadata)
                    .map_err(io_to_py_err)?;
            let mut fields = BTreeMap::new();
            for (key, value) in &normalized_fields {
                fields.insert(key.clone(), Some(value.clone()));
            }
            let resolved_ids = inner
                .db
                .resolve_query_ids_by_primary_keys(&collection_name, &[doc.inner.id.clone()])
                .map_err(io_to_py_err)?;
            let internal_id = resolved_ids.first().copied().ok_or_else(|| {
                PyFileNotFoundError::new_err(format!("document not found: {}", doc.inner.id))
            })?;
            updates.push(hannsdb_core::document::DocumentUpdate {
                id: internal_id,
                fields,
                vectors: BTreeMap::new(),
                sparse_vectors: BTreeMap::new(),
            });
        }
        self.inner_mut()?
            .db
            .update_documents(&collection_name, &updates)
            .map_err(io_to_py_err)
    }

    fn delete_by_filter(&mut self, filter: String) -> PyResult<usize> {
        let collection_name = self.inner_ref()?.collection_name.clone();
        self.inner_mut()?
            .db
            .delete_by_filter(&collection_name, &filter)
            .map_err(|error| match error.kind() {
                std::io::ErrorKind::InvalidInput => {
                    PyValueError::new_err(format!("invalid filter: {}", error))
                }
                _ => io_to_py_err(error),
            })
    }

    #[pyo3(signature = (field_name, data_type, nullable=false, array=false, expression="", _option=None))]
    fn add_column(
        &mut self,
        field_name: String,
        data_type: String,
        nullable: bool,
        array: bool,
        expression: &str,
        _option: Option<Py<PyAddColumnOption>>,
    ) -> PyResult<()> {
        let collection_name = self.inner_ref()?.collection_name.clone();
        let dt = parse_data_type(&data_type)?;
        let core_type = match dt {
            DataType::String => hannsdb_core::document::FieldType::String,
            DataType::Int64 => hannsdb_core::document::FieldType::Int64,
            DataType::Int32 => hannsdb_core::document::FieldType::Int32,
            DataType::UInt32 => hannsdb_core::document::FieldType::UInt32,
            DataType::UInt64 => hannsdb_core::document::FieldType::UInt64,
            DataType::Float => hannsdb_core::document::FieldType::Float,
            DataType::Float64 => hannsdb_core::document::FieldType::Float64,
            DataType::Bool => hannsdb_core::document::FieldType::Bool,
            DataType::VectorFp32 => {
                return Err(PyValueError::new_err(
                    "add_column does not support vector_fp32 type",
                ))
            }
        };
        let field = hannsdb_core::document::ScalarFieldSchema::new(field_name, core_type)
            .with_flags(nullable, array);
        let backfill = parse_add_column_backfill(expression, &field)?;
        self.inner_mut()?
            .db
            .add_column_with_backfill(&collection_name, field, backfill)
            .map_err(io_to_py_err)
    }

    fn drop_column(&mut self, field_name: String) -> PyResult<()> {
        let collection_name = self.inner_ref()?.collection_name.clone();
        self.inner_mut()?
            .db
            .drop_column(&collection_name, &field_name)
            .map_err(io_to_py_err)
    }

    #[pyo3(signature = (field_name, new_name="", field_schema=None, _option=None))]
    fn alter_column(
        &mut self,
        py: Python<'_>,
        field_name: String,
        new_name: &str,
        field_schema: Option<Py<PyFieldSchema>>,
        _option: Option<Py<PyAlterColumnOption>>,
    ) -> PyResult<()> {
        let collection_name = self.inner_ref()?.collection_name.clone();
        if let Some(field_schema) = field_schema {
            let target_field = field_schema.borrow(py).inner.clone();
            let migration = classify_alter_column_migration(
                std::path::Path::new(&self.inner_ref()?.path),
                &collection_name,
                &field_name,
                new_name,
                &target_field,
            )?;
            let target_field = py_field_schema_to_core(&target_field)?;
            self.inner_mut()?
                .db
                .alter_column_with_field_schema(
                    &collection_name,
                    &field_name,
                    new_name,
                    target_field,
                    migration,
                )
                .map_err(io_to_py_err)
        } else {
            self.inner_mut()?
                .db
                .alter_column(&collection_name, &field_name, new_name)
                .map_err(io_to_py_err)
        }
    }

    #[pyo3(signature = (vectors, output_fields=None, topk=100, filter=None))]
    fn query(
        &self,
        py: Python<'_>,
        vectors: Bound<'_, PyAny>,
        output_fields: Option<Vec<String>>,
        topk: usize,
        filter: Option<String>,
    ) -> PyResult<Vec<Py<PyDoc>>> {
        let vectors = py_vector_query_from_pyany(&vectors)?;
        let inner = self.inner_ref()?;
        let empty_output_fields = output_fields
            .as_ref()
            .is_some_and(|fields| fields.is_empty());
        let has_filter = filter
            .as_deref()
            .map(str::trim)
            .is_some_and(|value| !value.is_empty());
        if empty_output_fields && !has_filter {
            let hits = py
                .allow_threads(|| inner.query_ids_scores(topk, &vectors))
                .map_err(io_to_py_err)?;
            return hits
                .into_iter()
                .map(|(id, score)| {
                    Py::new(
                        py,
                        PyDoc {
                            inner: Doc {
                                id,
                                score: Some(score),
                                fields: BTreeMap::new(),
                                vectors: BTreeMap::new(),
                                field_name: inner.primary_vector_name.clone(),
                                group_key: None,
                            },
                        },
                    )
                })
                .collect();
        }

        let docs = py
            .allow_threads(|| inner.query(output_fields, topk, filter.as_deref(), vectors))
            .map_err(io_to_py_err)?;
        docs.into_iter()
            .map(|doc| Py::new(py, PyDoc { inner: doc }))
            .collect()
    }

    #[pyo3(signature = (context))]
    fn query_context(&self, py: Python<'_>, context: Bound<'_, PyAny>) -> PyResult<Vec<Py<PyDoc>>> {
        let inner = self.inner_ref()?;
        let docs = inner.query_context(py, &context)?;
        docs.into_iter()
            .map(|doc| Py::new(py, PyDoc { inner: doc }))
            .collect()
    }

    #[pyo3(signature = (field_name, vector, topk, ef=32))]
    fn search_ids_raw(
        &self,
        field_name: String,
        vector: PyReadonlyArray1<f32>,
        topk: usize,
        ef: usize,
    ) -> PyResult<Vec<i64>> {
        let inner = self.inner_ref()?;
        if field_name != inner.primary_vector_name {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "query field '{}' does not match primary vector '{}'",
                field_name, inner.primary_vector_name
            )));
        }
        let slice = vector.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("array slice error: {e}"))
        })?;
        let hits = inner
            .db
            .search_with_ef(&inner.collection_name, slice, topk, ef.max(1))
            .map_err(io_to_py_err)?;
        Ok(hits.into_iter().map(|hit| hit.id).collect())
    }

    #[pyo3(signature = (_option=None))]
    fn optimize(&mut self, _option: Option<Py<PyOptimizeOption>>) -> PyResult<()> {
        self.inner_mut()?.optimize().map_err(io_to_py_err)
    }

    fn flush(&self) -> PyResult<()> {
        self.inner_ref()?.flush().map_err(io_to_py_err)
    }

    #[getter]
    fn stats(&self) -> PyResult<PyCollectionStats> {
        Ok(PyCollectionStats {
            inner: self.inner_ref()?.stats().map_err(io_to_py_err)?,
        })
    }

    fn destroy(&mut self) -> PyResult<()> {
        self.inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("collection already destroyed"))?
            .destroy()
            .map_err(io_to_py_err)
    }

    #[pyo3(signature = (field_name, index_param=None, option=None))]
    fn create_vector_index(
        &self,
        py: Python<'_>,
        field_name: String,
        index_param: Option<Py<PyAny>>,
        option: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let _ = option; // accepted, ignored (concurrency not yet wired to core)
        let index_param = match index_param {
            Some(param) => {
                let bound = param.bind(py);
                if bound.is_instance_of::<PyFlatIndexParam>() {
                    Some(IndexParam::Flat(
                        bound
                            .extract::<PyRef<'_, PyFlatIndexParam>>()?
                            .inner
                            .clone(),
                    ))
                } else if bound.is_instance_of::<PyHnswIndexParam>() {
                    Some(IndexParam::Hnsw(
                        bound
                            .extract::<PyRef<'_, PyHnswIndexParam>>()?
                            .inner
                            .clone(),
                    ))
                } else if bound.is_instance_of::<PyHnswHvqIndexParam>() {
                    Some(IndexParam::HnswHvq(
                        bound
                            .extract::<PyRef<'_, PyHnswHvqIndexParam>>()?
                            .inner
                            .clone(),
                    ))
                } else if bound.is_instance_of::<PyIVFIndexParam>() {
                    Some(IndexParam::Ivf(
                        bound.extract::<PyRef<'_, PyIVFIndexParam>>()?.inner.clone(),
                    ))
                } else if bound.is_instance_of::<PyIvfUsqIndexParam>() {
                    Some(IndexParam::IvfUsq(
                        bound
                            .extract::<PyRef<'_, PyIvfUsqIndexParam>>()?
                            .inner
                            .clone(),
                    ))
                } else {
                    return Err(PyValueError::new_err(
                        "index_param must be FlatIndexParam, HnswIndexParam, HnswHvqIndexParam, IVFIndexParam, or IvfUsqIndexParam",
                    ));
                }
            }
            None => None,
        };

        self.inner_ref()?
            .create_vector_index(&field_name, index_param.as_ref())
            .map_err(io_to_py_err)
    }

    fn drop_vector_index(&self, field_name: String) -> PyResult<()> {
        self.inner_ref()?
            .drop_vector_index(&field_name)
            .map_err(io_to_py_err)
    }

    fn list_vector_indexes(&self) -> PyResult<Vec<String>> {
        self.inner_ref()?
            .list_vector_indexes()
            .map_err(io_to_py_err)
    }

    #[pyo3(signature = (field_name, index_param=None, option=None))]
    fn create_scalar_index(
        &self,
        py: Python<'_>,
        field_name: String,
        index_param: Option<Py<PyAny>>,
        option: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let _ = option; // accepted, ignored (concurrency not yet wired to core)
        let index_param = match index_param {
            Some(param) => {
                let bound = param.bind(py);
                if bound.is_instance_of::<PyInvertIndexParam>() {
                    Some(
                        bound
                            .extract::<PyRef<'_, PyInvertIndexParam>>()?
                            .inner
                            .clone(),
                    )
                } else {
                    return Err(PyValueError::new_err(
                        "scalar index param must be InvertIndexParam",
                    ));
                }
            }
            None => None,
        };
        self.inner_ref()?
            .create_scalar_index_with_param(&field_name, index_param.as_ref())
            .map_err(io_to_py_err)
    }

    fn drop_scalar_index(&self, field_name: String) -> PyResult<()> {
        self.inner_ref()?
            .drop_scalar_index(&field_name)
            .map_err(io_to_py_err)
    }

    fn list_scalar_indexes(&self) -> PyResult<Vec<String>> {
        self.inner_ref()?
            .list_scalar_indexes()
            .map_err(io_to_py_err)
    }

    #[pyo3(signature = (name, data_type, dimension, index_param=None))]
    fn add_vector_field(
        &mut self,
        name: String,
        data_type: String,
        dimension: usize,
        index_param: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let index_param = match index_param {
            Some(param) => {
                let py = unsafe { pyo3::Python::assume_gil_acquired() };
                let bound = param.bind(py);
                if bound.is_instance_of::<PyFlatIndexParam>() {
                    Some(IndexParam::Flat(
                        bound
                            .extract::<pyo3::PyRef<'_, PyFlatIndexParam>>()?
                            .inner
                            .clone(),
                    ))
                } else if bound.is_instance_of::<PyHnswIndexParam>() {
                    Some(IndexParam::Hnsw(
                        bound
                            .extract::<pyo3::PyRef<'_, PyHnswIndexParam>>()?
                            .inner
                            .clone(),
                    ))
                } else if bound.is_instance_of::<PyHnswHvqIndexParam>() {
                    Some(IndexParam::HnswHvq(
                        bound
                            .extract::<pyo3::PyRef<'_, PyHnswHvqIndexParam>>()?
                            .inner
                            .clone(),
                    ))
                } else if bound.is_instance_of::<PyIVFIndexParam>() {
                    Some(IndexParam::Ivf(
                        bound
                            .extract::<pyo3::PyRef<'_, PyIVFIndexParam>>()?
                            .inner
                            .clone(),
                    ))
                } else if bound.is_instance_of::<PyIvfUsqIndexParam>() {
                    Some(IndexParam::IvfUsq(
                        bound
                            .extract::<pyo3::PyRef<'_, PyIvfUsqIndexParam>>()?
                            .inner
                            .clone(),
                    ))
                } else {
                    None
                }
            }
            None => None,
        };
        self.inner_mut()?
            .add_vector_field(&name, &data_type, dimension, index_param.as_ref())
            .map_err(io_to_py_err)
    }

    fn drop_vector_field(&mut self, field_name: String) -> PyResult<()> {
        self.inner_mut()?
            .drop_vector_field(&field_name)
            .map_err(io_to_py_err)
    }
}

#[cfg(feature = "python-binding")]
#[pyfunction(name = "init")]
#[pyo3(signature = (log_level="warn"))]
fn py_init(log_level: &str) -> PyResult<()> {
    init(parse_log_level(log_level)?);
    Ok(())
}

#[cfg(feature = "python-binding")]
#[pyfunction(name = "create_and_open")]
#[pyo3(signature = (path, schema, option=None, *, storage="hannsdb"))]
fn py_create_and_open(
    py: Python<'_>,
    path: String,
    schema: Py<PyCollectionSchema>,
    option: Option<Py<PyCollectionOption>>,
    storage: &str,
) -> PyResult<Py<PyAny>> {
    let schema = schema.borrow(py).inner.clone();
    let option = option.map(|value| value.borrow(py).inner.clone());
    if normalize_storage_backend(storage)? == "lance" {
        #[cfg(feature = "lance-storage")]
        {
            let core_schema = core_schema_from_schema(&schema).map_err(io_to_py_err)?;
            let empty_docs: Vec<CoreDocument> = Vec::new();
            let collection = block_on_lance(CoreLanceCollection::create(
                path,
                schema.name.clone(),
                core_schema,
                &empty_docs,
            ))?;
            return Ok(Py::new(
                py,
                PyLanceCollection {
                    inner: Some(collection),
                    schema,
                },
            )?
            .into_any());
        }
        #[cfg(not(feature = "lance-storage"))]
        {
            return Err(PyNotImplementedError::new_err(
                "Lance storage requires the lance-storage feature",
            ));
        }
    }
    let collection = create_and_open(path, schema, option).map_err(io_to_py_err)?;
    Ok(Py::new(
        py,
        PyCollection {
            inner: Some(collection),
        },
    )?
    .into_any())
}

#[cfg(feature = "python-binding")]
#[pyfunction(name = "open")]
#[pyo3(signature = (path, option=None, *, storage="hannsdb", schema=None, name=None))]
fn py_open(
    py: Python<'_>,
    path: String,
    option: Option<Py<PyCollectionOption>>,
    storage: &str,
    schema: Option<Py<PyCollectionSchema>>,
    name: Option<String>,
) -> PyResult<Py<PyAny>> {
    let option = option.map(|value| value.borrow(py).inner.clone());
    if normalize_storage_backend(storage)? == "lance" {
        #[cfg(feature = "lance-storage")]
        {
            let (collection, schema) = if let Some(schema) = schema {
                let schema = schema.borrow(py).inner.clone();
                if let Some(requested_name) = name.as_deref() {
                    if requested_name != schema.name {
                        return Err(PyValueError::new_err(
                            "name must match schema.name when schema is provided",
                        ));
                    }
                }
                let core_schema = core_schema_from_schema(&schema).map_err(io_to_py_err)?;
                let collection = block_on_lance(CoreLanceCollection::open(
                    path,
                    schema.name.clone(),
                    core_schema,
                ))?;
                (collection, schema)
            } else {
                let name = name.ok_or_else(|| {
                    PyValueError::new_err(
                        "name is required when opening Lance storage without schema",
                    )
                })?;
                let collection =
                    block_on_lance(CoreLanceCollection::open_inferred(path, name.clone()))?;
                let schema =
                    schema_from_core_schema(name, collection.schema()).map_err(io_to_py_err)?;
                (collection, schema)
            };
            return Ok(Py::new(
                py,
                PyLanceCollection {
                    inner: Some(collection),
                    schema,
                },
            )?
            .into_any());
        }
        #[cfg(not(feature = "lance-storage"))]
        {
            return Err(PyNotImplementedError::new_err(
                "Lance storage requires the lance-storage feature",
            ));
        }
    }
    if schema.is_some() || name.is_some() {
        return Err(PyValueError::new_err(
            "schema and name are only supported when opening Lance storage",
        ));
    }
    let collection = open(path, option).map_err(io_to_py_err)?;
    Ok(Py::new(
        py,
        PyCollection {
            inner: Some(collection),
        },
    )?
    .into_any())
}

#[cfg(all(feature = "python-binding", feature = "lance-storage"))]
#[pyfunction(name = "create_lance_collection")]
#[pyo3(signature = (path, schema, docs))]
fn py_create_lance_collection(
    py: Python<'_>,
    path: String,
    schema: Py<PyCollectionSchema>,
    docs: Vec<Py<PyDoc>>,
) -> PyResult<PyLanceCollection> {
    let schema = schema.borrow(py).inner.clone();
    let docs = docs
        .into_iter()
        .map(|doc| doc.borrow(py).inner.clone())
        .collect::<Vec<_>>();
    let core_documents = core_documents_from_lance_docs(&docs, &schema).map_err(io_to_py_err)?;
    let core_schema = core_schema_from_schema(&schema).map_err(io_to_py_err)?;
    let collection = block_on_lance(CoreLanceCollection::create(
        path,
        schema.name.clone(),
        core_schema,
        &core_documents,
    ))?;
    Ok(PyLanceCollection {
        inner: Some(collection),
        schema,
    })
}

#[cfg(all(feature = "python-binding", feature = "lance-storage"))]
#[pyfunction(name = "open_lance_collection")]
#[pyo3(signature = (path, schema))]
fn py_open_lance_collection(
    py: Python<'_>,
    path: String,
    schema: Py<PyCollectionSchema>,
) -> PyResult<PyLanceCollection> {
    let schema = schema.borrow(py).inner.clone();
    let core_schema = core_schema_from_schema(&schema).map_err(io_to_py_err)?;
    let collection = block_on_lance(CoreLanceCollection::open(
        path,
        schema.name.clone(),
        core_schema,
    ))?;
    Ok(PyLanceCollection {
        inner: Some(collection),
        schema,
    })
}

#[cfg(all(feature = "python-binding", feature = "lance-storage"))]
#[pyfunction(name = "open_lance_collection_infer_schema")]
#[pyo3(signature = (path, name))]
fn py_open_lance_collection_infer_schema(
    path: String,
    name: String,
) -> PyResult<PyLanceCollection> {
    let collection = block_on_lance(CoreLanceCollection::open_inferred(path, name.clone()))?;
    let schema = schema_from_core_schema(name, collection.schema()).map_err(io_to_py_err)?;
    Ok(PyLanceCollection {
        inner: Some(collection),
        schema,
    })
}

#[cfg(feature = "python-binding")]
#[pymodule(name = "_native")]
fn python_module(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMetricType>()?;
    module.add_class::<PyQuantizeType>()?;
    module.add_class::<PyLogLevel>()?;
    module.add_class::<PyDataType>()?;
    module.add_class::<PyOptimizeOption>()?;
    module.add_class::<PyCollectionOption>()?;
    module.add_class::<PyAddColumnOption>()?;
    module.add_class::<PyAlterColumnOption>()?;
    module.add_class::<PyInvertIndexParam>()?;
    module.add_class::<PyFlatIndexParam>()?;
    module.add_class::<PyHnswIndexParam>()?;
    module.add_class::<PyHnswHvqIndexParam>()?;
    module.add_class::<PyIVFIndexParam>()?;
    module.add_class::<PyIvfUsqIndexParam>()?;
    module.add_class::<PyHnswQueryParam>()?;
    module.add_class::<PyIVFQueryParam>()?;
    module.add_class::<PyIvfUsqQueryParam>()?;
    module.add_class::<PyHnswSqQueryParam>()?;
    module.add_class::<PyHnswHvqQueryParam>()?;
    module.add_class::<PyIndexOption>()?;
    module.add_class::<PyFieldSchema>()?;
    module.add_class::<PyVectorSchema>()?;
    module.add_class::<PyCollectionSchema>()?;
    module.add_class::<PyDoc>()?;
    module.add_class::<PySparseVector>()?;
    module.add_class::<PyVectorQuery>()?;
    module.add_class::<PyRrfReRanker>()?;
    module.add_class::<PyWeightedReRanker>()?;
    module.add_class::<PyCollectionStats>()?;
    module.add_class::<PyCollection>()?;
    #[cfg(feature = "lance-storage")]
    module.add_class::<PyLanceCollection>()?;
    module.add_function(wrap_pyfunction!(py_init, module)?)?;
    module.add_function(wrap_pyfunction!(py_create_and_open, module)?)?;
    module.add_function(wrap_pyfunction!(py_open, module)?)?;
    #[cfg(feature = "lance-storage")]
    module.add_function(wrap_pyfunction!(py_create_lance_collection, module)?)?;
    #[cfg(feature = "lance-storage")]
    module.add_function(wrap_pyfunction!(py_open_lance_collection, module)?)?;
    #[cfg(feature = "lance-storage")]
    module.add_function(wrap_pyfunction!(
        py_open_lance_collection_infer_schema,
        module
    )?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::io::ErrorKind;

    use crate::{
        create_and_open, init, open, vector_index_catalog_json, CollectionOption, CollectionSchema,
        DataType, Doc, FieldSchema, HnswIndexParam, HnswQueryParam, IndexParam, IvfUsqIndexParam,
        LogLevel, MetricType, OptimizeOption, QuantizeType, VectorQuery, VectorSchema,
    };
    use hannsdb_core::document::FieldValue;

    fn hnsw_index_param(metric_type: MetricType) -> IndexParam {
        IndexParam::Hnsw(HnswIndexParam {
            metric_type: Some(metric_type),
            m: 16,
            ef_construction: 64,
            quantize_type: QuantizeType::Undefined,
        })
    }

    fn doc(
        id: &str,
        vector: Vec<f32>,
        fields: impl IntoIterator<Item = (&'static str, FieldValue)>,
    ) -> Doc {
        Doc {
            id: id.to_string(),
            score: None,
            fields: fields
                .into_iter()
                .map(|(name, value)| (name.to_string(), value))
                .collect(),
            vectors: [("dense".to_string(), vector)].into_iter().collect(),
            field_name: "dense".to_string(),
            group_key: None,
        }
    }

    #[test]
    fn exports_bootstrap_symbol() {
        assert_eq!(crate::bootstrap_symbol(), "hannsdb_py_bootstrap");
    }

    #[test]
    fn exposes_benchmark_facing_types() {
        let option = CollectionOption {
            read_only: false,
            enable_mmap: true,
        };
        let vec_schema = VectorSchema {
            name: "dense".to_string(),
            data_type: DataType::VectorFp32,
            dimension: 4,
            index_param: Some(hnsw_index_param(MetricType::L2)),
        };
        let schema = CollectionSchema {
            name: "bench".to_string(),
            primary_vector: "dense".to_string(),
            fields: vec![FieldSchema {
                name: "id".to_string(),
                data_type: DataType::Int64,
                nullable: false,
                array: false,
            }],
            vectors: vec![vec_schema],
        };
        let _doc = Doc {
            id: "1".to_string(),
            score: None,
            fields: [("id".to_string(), FieldValue::Int64(1))]
                .into_iter()
                .collect(),
            vectors: [("dense".to_string(), vec![0.0, 0.1, 0.2, 0.3])]
                .into_iter()
                .collect(),
            field_name: "dense".to_string(),
            group_key: None,
        };
        let _query = VectorQuery {
            field_name: "dense".to_string(),
            vector: crate::QueryVector::Dense(vec![0.0, 0.1, 0.2, 0.3]),
            param: Some(HnswQueryParam {
                ef: 32,
                nprobe: 0,
                is_using_refiner: false,
            }),
        };
        let _opt = OptimizeOption {};
        let _metric = MetricType::L2;
        let _quant = QuantizeType::Undefined;

        let _ = (option, schema, _opt, _metric, _quant);
    }

    #[test]
    fn lifecycle_shells_create_and_open_collection() {
        init(LogLevel::Warn);
        let temp = tempfile::tempdir().expect("tempdir");
        let db_path = temp.path().to_string_lossy().to_string();
        let option = CollectionOption {
            read_only: false,
            enable_mmap: true,
        };
        let schema = CollectionSchema {
            name: "bench".to_string(),
            primary_vector: "dense".to_string(),
            fields: vec![],
            vectors: vec![VectorSchema {
                name: "dense".to_string(),
                data_type: DataType::VectorFp32,
                dimension: 2,
                index_param: None,
            }],
        };

        let created = create_and_open(db_path.clone(), schema, Some(option.clone()))
            .expect("create should work");
        assert_eq!(created.path, db_path);
        assert_eq!(created.option, option.clone());

        let reopened = open(created.path.clone(), Some(option.clone())).expect("open should work");
        assert_eq!(reopened.path, created.path);
        assert_eq!(reopened.option, option);
    }

    #[test]
    fn benchmark_flow_over_core_wrapper() {
        use std::fs;

        let temp = tempfile::tempdir().expect("tempdir");
        let db_path = temp.path().to_string_lossy().to_string();

        let schema = CollectionSchema {
            name: "bench_col".to_string(),
            primary_vector: "dense".to_string(),
            fields: vec![FieldSchema {
                name: "id".to_string(),
                data_type: DataType::Int64,
                nullable: false,
                array: false,
            }],
            vectors: vec![VectorSchema {
                name: "dense".to_string(),
                data_type: DataType::VectorFp32,
                dimension: 2,
                index_param: Some(hnsw_index_param(MetricType::L2)),
            }],
        };
        let option = CollectionOption::default();

        let mut collection = create_and_open(db_path.clone(), schema, Some(option))
            .expect("create_and_open should succeed");
        assert!(temp
            .path()
            .join("collections")
            .join("bench_col")
            .join("collection.json")
            .exists());

        collection
            .insert(&[
                doc("11", vec![0.0, 0.0], [("id", FieldValue::Int64(11))]),
                doc("22", vec![10.0, 10.0], [("id", FieldValue::Int64(22))]),
            ])
            .expect("insert should succeed");

        let hits = collection
            .query(
                Some(Vec::new()),
                1,
                Some(""),
                VectorQuery {
                    field_name: "dense".to_string(),
                    vector: crate::QueryVector::Dense(vec![0.1, -0.1]),
                    param: Some(HnswQueryParam {
                        ef: 32,
                        nprobe: 0,
                        is_using_refiner: false,
                    }),
                },
            )
            .expect("query should succeed");
        assert_eq!(hits[0].id, "11".to_string());

        collection.optimize().expect("optimize should warm cache");

        let collection_dir = temp.path().join("collections").join("bench_col");
        fs::remove_file(collection_dir.join("records.bin")).expect("remove records.bin");
        fs::remove_file(collection_dir.join("ids.bin")).expect("remove ids.bin");

        let hits_after_optimize = collection
            .query(
                Some(Vec::new()),
                1,
                Some(""),
                VectorQuery {
                    field_name: "dense".to_string(),
                    vector: crate::QueryVector::Dense(vec![0.1, -0.1]),
                    param: Some(HnswQueryParam {
                        ef: 32,
                        nprobe: 0,
                        is_using_refiner: false,
                    }),
                },
            )
            .expect("query after optimize warm should succeed without on-disk payload files");
        assert_eq!(hits_after_optimize[0].id, "11".to_string());

        let reopened =
            open(db_path.clone(), Some(CollectionOption::default())).expect("open should succeed");
        assert_eq!(reopened.collection_name, "bench_col".to_string());
        reopened
            .destroy()
            .expect("destroy should remove collection");

        assert!(!temp.path().join("collections").join("bench_col").exists());
        let err = open(db_path, Some(CollectionOption::default()))
            .err()
            .expect("destroy should remove collection from manifest too");
        assert_eq!(err.kind(), ErrorKind::NotFound);
    }

    #[test]
    fn create_and_open_uses_metric_type_from_schema() {
        let temp = tempfile::tempdir().expect("tempdir");
        let db_path = temp.path().to_string_lossy().to_string();
        let schema = CollectionSchema {
            name: "metric_col".to_string(),
            primary_vector: "dense".to_string(),
            fields: vec![],
            vectors: vec![VectorSchema {
                name: "dense".to_string(),
                data_type: DataType::VectorFp32,
                dimension: 2,
                index_param: Some(hnsw_index_param(MetricType::Ip)),
            }],
        };

        let _collection =
            create_and_open(db_path, schema, Some(CollectionOption::default())).expect("create");

        let metadata = hannsdb_core::catalog::CollectionMetadata::load_from_path(
            &temp
                .path()
                .join("collections")
                .join("metric_col")
                .join("collection.json"),
        )
        .expect("load collection metadata");
        assert_eq!(metadata.metric, "ip");
    }

    #[test]
    fn wrapper_supports_upsert_fetch_filter_query_flush_and_stats() {
        let temp = tempfile::tempdir().expect("tempdir");
        let db_path = temp.path().to_string_lossy().to_string();
        let schema = CollectionSchema {
            name: "agent_docs".to_string(),
            primary_vector: "dense".to_string(),
            fields: vec![
                FieldSchema {
                    name: "session_id".to_string(),
                    data_type: DataType::String,
                    nullable: false,
                    array: false,
                },
                FieldSchema {
                    name: "turn".to_string(),
                    data_type: DataType::Int64,
                    nullable: false,
                    array: false,
                },
                FieldSchema {
                    name: "active".to_string(),
                    data_type: DataType::Bool,
                    nullable: false,
                    array: false,
                },
            ],
            vectors: vec![VectorSchema {
                name: "dense".to_string(),
                data_type: DataType::VectorFp32,
                dimension: 2,
                index_param: Some(hnsw_index_param(MetricType::L2)),
            }],
        };

        let mut collection = create_and_open(db_path, schema, Some(CollectionOption::default()))
            .expect("create_and_open");

        collection
            .insert(&[
                doc(
                    "11",
                    vec![100.0, 100.0],
                    [
                        ("session_id", FieldValue::String("s1".to_string())),
                        ("turn", FieldValue::Int64(1)),
                        ("active", FieldValue::Bool(true)),
                    ],
                ),
                doc(
                    "22",
                    vec![0.1, 0.1],
                    [
                        ("session_id", FieldValue::String("s1".to_string())),
                        ("turn", FieldValue::Int64(2)),
                        ("active", FieldValue::Bool(true)),
                    ],
                ),
                doc(
                    "33",
                    vec![0.05, 0.05],
                    [
                        ("session_id", FieldValue::String("s2".to_string())),
                        ("turn", FieldValue::Int64(3)),
                        ("active", FieldValue::Bool(false)),
                    ],
                ),
            ])
            .expect("insert docs");

        let stats = collection.stats().expect("stats");
        assert_eq!(stats.record_count, 3);
        assert_eq!(stats.live_count, 3);
        collection.flush().expect("flush");

        let fetched = collection
            .fetch(&["22".to_string()])
            .expect("fetch by id should work");
        assert_eq!(fetched.len(), 1);
        assert_eq!(fetched[0].id, "22");
        assert_eq!(
            fetched[0].fields.get("session_id"),
            Some(&FieldValue::String("s1".to_string()))
        );
        assert_eq!(fetched[0].vectors.get("dense"), Some(&vec![0.1_f32, 0.1]));

        let filtered = collection
            .query(
                Some(vec!["session_id".to_string(), "turn".to_string()]),
                1,
                Some("session_id == \"s1\" and turn >= 2"),
                VectorQuery {
                    field_name: "dense".to_string(),
                    vector: crate::QueryVector::Dense(vec![0.0, 0.0]),
                    param: Some(HnswQueryParam {
                        ef: 32,
                        nprobe: 0,
                        is_using_refiner: false,
                    }),
                },
            )
            .expect("filtered query");
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].id, "22");
        assert!(filtered[0].score.is_some());
        assert_eq!(
            filtered[0].fields.get("session_id"),
            Some(&FieldValue::String("s1".to_string()))
        );
        assert_eq!(filtered[0].fields.get("turn"), Some(&FieldValue::Int64(2)));
        assert_eq!(filtered[0].vectors, BTreeMap::new());

        collection
            .upsert(&[doc(
                "22",
                vec![0.0, 0.0],
                [
                    ("session_id", FieldValue::String("s1".to_string())),
                    ("turn", FieldValue::Int64(4)),
                    ("active", FieldValue::Bool(true)),
                ],
            )])
            .expect("upsert");
        let fetched_after_upsert = collection
            .fetch(&["22".to_string()])
            .expect("fetch upserted");
        assert_eq!(
            fetched_after_upsert[0].fields.get("turn"),
            Some(&FieldValue::Int64(4))
        );
        assert_eq!(
            fetched_after_upsert[0].vectors.get("dense"),
            Some(&vec![0.0_f32, 0.0])
        );
    }

    #[test]
    fn wrapper_supports_delete_and_updates_stats() {
        let temp = tempfile::tempdir().expect("tempdir");
        let db_path = temp.path().to_string_lossy().to_string();
        let schema = CollectionSchema {
            name: "agent_docs".to_string(),
            primary_vector: "dense".to_string(),
            fields: vec![FieldSchema {
                name: "session_id".to_string(),
                data_type: DataType::String,
                nullable: false,
                array: false,
            }],
            vectors: vec![VectorSchema {
                name: "dense".to_string(),
                data_type: DataType::VectorFp32,
                dimension: 2,
                index_param: Some(hnsw_index_param(MetricType::L2)),
            }],
        };

        let mut collection = create_and_open(db_path, schema, Some(CollectionOption::default()))
            .expect("create_and_open");

        collection
            .insert(&[
                doc(
                    "11",
                    vec![0.0, 0.0],
                    [("session_id", FieldValue::String("s1".to_string()))],
                ),
                doc(
                    "22",
                    vec![1.0, 1.0],
                    [("session_id", FieldValue::String("s2".to_string()))],
                ),
            ])
            .expect("insert docs");

        let deleted = collection
            .delete(&["11".to_string()])
            .expect("delete existing id should succeed");
        assert_eq!(deleted, 1);

        let fetched = collection
            .fetch(&["11".to_string()])
            .expect("fetch deleted doc should not error");
        assert!(fetched.is_empty());

        let stats = collection.stats().expect("stats");
        assert_eq!(stats.record_count, 2);
        assert_eq!(stats.deleted_count, 1);
        assert_eq!(stats.live_count, 1);
    }

    #[test]
    fn create_and_open_accepts_metadata_only_secondary_vectors() {
        let temp = tempfile::tempdir().expect("tempdir");
        let db_path = temp.path().to_string_lossy().to_string();
        let schema = CollectionSchema {
            name: "multi_vec".to_string(),
            primary_vector: "dense".to_string(),
            fields: vec![],
            vectors: vec![
                VectorSchema {
                    name: "dense".to_string(),
                    data_type: DataType::VectorFp32,
                    dimension: 2,
                    index_param: None,
                },
                VectorSchema {
                    name: "sparse".to_string(),
                    data_type: DataType::VectorFp32,
                    dimension: 2,
                    index_param: None,
                },
            ],
        };

        let collection =
            create_and_open(db_path, schema, Some(CollectionOption::default())).expect("create");
        assert_eq!(collection.primary_vector_name, "dense");

        let metadata = hannsdb_core::catalog::CollectionMetadata::load_from_path(
            &temp
                .path()
                .join("collections")
                .join("multi_vec")
                .join("collection.json"),
        )
        .expect("load collection metadata");
        assert_eq!(metadata.primary_vector, "dense");
        assert_eq!(metadata.vectors.len(), 2);
    }

    #[test]
    fn core_schema_from_schema_keeps_explicit_primary_vector_when_not_first() {
        let schema = CollectionSchema {
            name: "primary_by_name".to_string(),
            primary_vector: "dense".to_string(),
            fields: vec![FieldSchema {
                name: "tags".to_string(),
                data_type: DataType::String,
                nullable: true,
                array: true,
            }],
            vectors: vec![
                VectorSchema {
                    name: "title".to_string(),
                    data_type: DataType::VectorFp32,
                    dimension: 2,
                    index_param: None,
                },
                VectorSchema {
                    name: "dense".to_string(),
                    data_type: DataType::VectorFp32,
                    dimension: 2,
                    index_param: Some(hnsw_index_param(MetricType::L2)),
                },
            ],
        };

        let core_schema = super::core_schema_from_schema(&schema).expect("schema conversion");
        assert_eq!(core_schema.primary_vector, "dense");
        assert!(core_schema.fields[0].nullable);
        assert!(core_schema.fields[0].array);
    }

    #[test]
    fn create_and_open_accepts_primary_ivf_index() {
        let temp = tempfile::tempdir().expect("tempdir");
        let db_path = temp.path().to_string_lossy().to_string();
        let schema = CollectionSchema {
            name: "primary_ivf".to_string(),
            primary_vector: "dense".to_string(),
            fields: vec![],
            vectors: vec![
                VectorSchema {
                    name: "dense".to_string(),
                    data_type: DataType::VectorFp32,
                    dimension: 2,
                    index_param: Some(IndexParam::Ivf(crate::IvfIndexParam {
                        metric_type: Some(MetricType::L2),
                        nlist: 1024,
                    })),
                },
                VectorSchema {
                    name: "title".to_string(),
                    data_type: DataType::VectorFp32,
                    dimension: 2,
                    index_param: None,
                },
            ],
        };

        let created = create_and_open(db_path.clone(), schema, Some(CollectionOption::default()))
            .expect("primary IVF should now be accepted");
        assert_eq!(created.path, db_path);
        assert_eq!(created.collection_name, "primary_ivf");
    }

    #[test]
    fn vector_index_catalog_json_preserves_ivf_usq_kind() {
        let json = vector_index_catalog_json(
            "dense",
            Some(&IndexParam::IvfUsq(IvfUsqIndexParam {
                metric_type: Some(MetricType::L2),
                nlist: 128,
                bits_per_dim: 4,
                rotation_seed: 42,
                rerank_k: 64,
                use_high_accuracy_scan: true,
            })),
        );

        assert!(json.contains("\"kind\":\"ivf_usq\""));
        assert!(json.contains("\"nlist\":128"));
        assert!(json.contains("\"bits_per_dim\":4"));
        assert!(json.contains("\"rotation_seed\":42"));
        assert!(json.contains("\"rerank_k\":64"));
        assert!(json.contains("\"use_high_accuracy_scan\":true"));
    }

    #[test]
    fn create_and_open_accepts_ivf_usq_index() {
        let temp = tempfile::tempdir().expect("tempdir");
        let db_path = temp.path().to_string_lossy().to_string();
        let schema = CollectionSchema {
            name: "ivf_usq_docs".to_string(),
            primary_vector: "dense".to_string(),
            fields: vec![],
            vectors: vec![VectorSchema {
                name: "dense".to_string(),
                data_type: DataType::VectorFp32,
                dimension: 2,
                index_param: Some(IndexParam::IvfUsq(IvfUsqIndexParam {
                    metric_type: Some(MetricType::L2),
                    nlist: 1,
                    bits_per_dim: 4,
                    rotation_seed: 42,
                    rerank_k: 64,
                    use_high_accuracy_scan: false,
                })),
            }],
        };

        let created = create_and_open(db_path.clone(), schema, Some(CollectionOption::default()))
            .expect("ivf_usq should now be accepted");
        assert_eq!(created.path, db_path);
        assert_eq!(created.collection_name, "ivf_usq_docs");
    }
}
