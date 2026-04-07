use std::collections::BTreeMap;

#[cfg(feature = "python-binding")]
use numpy::PyReadonlyArray1;
#[cfg(feature = "python-binding")]
use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError, PyValueError};
#[cfg(feature = "python-binding")]
use pyo3::prelude::*;
#[cfg(feature = "python-binding")]
use pyo3::types::{PyAny, PyBool, PyDict};

use hannsdb_core::catalog::CollectionMetadata;
use hannsdb_core::document::{
    CollectionSchema as CoreCollectionSchema, Document as CoreDocument, FieldType as CoreFieldType,
    FieldValue, ScalarFieldSchema as CoreScalarFieldSchema,
    VectorFieldSchema as CoreVectorFieldSchema, VectorIndexSchema as CoreVectorIndexSchema,
};

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
pub enum DataType {
    String,
    Int64,
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
pub struct IvfIndexParam {
    pub metric_type: Option<MetricType>,
    pub nlist: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexParam {
    Hnsw(HnswIndexParam),
    Ivf(IvfIndexParam),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HnswQueryParam {
    pub ef: usize,
    pub is_using_refiner: bool,
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
}

#[derive(Debug, Clone, PartialEq)]
pub struct VectorQuery {
    pub field_name: String,
    pub vector: Vec<f32>,
    pub param: Option<HnswQueryParam>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionStats {
    pub name: String,
    pub dimension: usize,
    pub metric: String,
    pub record_count: usize,
    pub deleted_count: usize,
    pub live_count: usize,
}

pub struct Collection {
    pub path: String,
    pub collection_name: String,
    pub primary_vector_name: String,
    pub option: CollectionOption,
    db: hannsdb_core::db::HannsDb,
}

pub fn init(_log_level: LogLevel) {}

fn metric_type_name(metric: MetricType) -> &'static str {
    match metric {
        MetricType::L2 => "l2",
        MetricType::Cosine => "cosine",
        MetricType::Ip => "ip",
    }
}

fn core_schema_from_schema(schema: &CollectionSchema) -> std::io::Result<CoreCollectionSchema> {
    if schema.vectors.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "CollectionSchema requires at least one vector schema",
        ));
    }
    let primary_vector = schema
        .vectors
        .iter()
        .find(|vector| vector.name == schema.primary_vector)
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "primary vector '{}' is not defined in vectors",
                    schema.primary_vector
                ),
            )
        })?;
    if matches!(
        primary_vector.index_param.as_ref(),
        Some(IndexParam::Ivf(_))
    ) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "primary vector '{}' must use hnsw or no index_param for the current runtime",
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
                Some(IndexParam::Hnsw(params)) => Some(CoreVectorIndexSchema::hnsw(
                    params.metric_type.map(metric_type_name),
                    params.m,
                    params.ef_construction,
                )),
                Some(IndexParam::Ivf(params)) => Some(CoreVectorIndexSchema::ivf(
                    params.metric_type.map(metric_type_name),
                    params.nlist,
                )),
                None => None,
            };

            Ok(CoreVectorFieldSchema {
                name: vector.name.clone(),
                data_type: CoreFieldType::VectorFp32,
                dimension: vector.dimension,
                index_param,
            })
        })
        .collect::<std::io::Result<Vec<_>>>()?;

    Ok(CoreCollectionSchema {
        primary_vector: schema.primary_vector.clone(),
        fields,
        vectors,
    })
}

fn parse_doc_id(id: &str) -> std::io::Result<i64> {
    id.parse::<i64>().map_err(|_| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "doc id must parse to i64")
    })
}

fn core_document_from_doc(doc: &Doc, primary_vector_name: &str) -> std::io::Result<CoreDocument> {
    if doc.vectors.len() != 1 || !doc.vectors.contains_key(primary_vector_name) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("doc must contain exactly one vector named '{primary_vector_name}'"),
        ));
    }

    Ok(CoreDocument::new(
        parse_doc_id(&doc.id)?,
        doc.fields.clone(),
        doc.vectors
            .get(primary_vector_name)
            .cloned()
            .expect("checked primary vector presence"),
    ))
}

fn doc_from_core_document(document: CoreDocument, primary_vector_name: &str) -> Doc {
    Doc {
        id: document.id.to_string(),
        score: None,
        fields: document.fields,
        vectors: [(primary_vector_name.to_string(), document.vector)]
            .into_iter()
            .collect(),
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

pub fn create_and_open(
    path: impl Into<String>,
    schema: CollectionSchema,
    option: Option<CollectionOption>,
) -> std::io::Result<Collection> {
    let path = path.into();
    let root = std::path::Path::new(&path);
    let mut db = hannsdb_core::db::HannsDb::open(root)?;
    let core_schema = core_schema_from_schema(&schema)?;
    db.create_collection_with_schema(&schema.name, &core_schema)?;
    Ok(Collection {
        path,
        collection_name: schema.name,
        primary_vector_name: core_schema.primary_vector_name().to_string(),
        option: option.unwrap_or_default(),
        db,
    })
}

pub fn open(
    path: impl Into<String>,
    option: Option<CollectionOption>,
) -> std::io::Result<Collection> {
    let path = path.into();
    let root = std::path::Path::new(&path);
    let db = hannsdb_core::db::HannsDb::open(root)?;
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
        option: option.unwrap_or_default(),
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

        let ef_search = vectors.param.as_ref().map_or(32, |param| param.ef).max(1);
        let hits =
            self.db
                .search_with_ef(&self.collection_name, &vectors.vector, topk, ef_search)?;
        Ok(hits
            .into_iter()
            .map(|hit| (hit.id.to_string(), hit.distance))
            .collect())
    }

    pub fn insert(&mut self, docs: &[Doc]) -> std::io::Result<usize> {
        let documents = docs
            .iter()
            .map(|doc| core_document_from_doc(doc, &self.primary_vector_name))
            .collect::<std::io::Result<Vec<_>>>()?;
        self.db.insert_documents(&self.collection_name, &documents)
    }

    pub fn upsert(&mut self, docs: &[Doc]) -> std::io::Result<usize> {
        let documents = docs
            .iter()
            .map(|doc| core_document_from_doc(doc, &self.primary_vector_name))
            .collect::<std::io::Result<Vec<_>>>()?;
        self.db.upsert_documents(&self.collection_name, &documents)
    }

    pub fn fetch(&self, ids: &[String]) -> std::io::Result<Vec<Doc>> {
        let ids = ids
            .iter()
            .map(|id| parse_doc_id(id))
            .collect::<std::io::Result<Vec<_>>>()?;
        let documents = self.db.fetch_documents(&self.collection_name, &ids)?;
        Ok(documents
            .into_iter()
            .map(|document| doc_from_core_document(document, &self.primary_vector_name))
            .collect())
    }

    pub fn delete(&mut self, ids: &[String]) -> std::io::Result<usize> {
        let ids = ids
            .iter()
            .map(|id| parse_doc_id(id))
            .collect::<std::io::Result<Vec<_>>>()?;
        self.db.delete(&self.collection_name, &ids)
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
        })
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

        if let Some(filter) = filter.map(str::trim).filter(|filter| !filter.is_empty()) {
            return self
                .db
                .query_documents(&self.collection_name, &vectors.vector, topk, Some(filter))
                .map(|hits| {
                    hits.into_iter()
                        .map(|hit| Doc {
                            id: hit.id.to_string(),
                            score: Some(hit.distance),
                            fields: select_output_fields(&hit.fields, &output_fields),
                            vectors: BTreeMap::new(),
                        })
                        .collect()
                });
        }

        let ef_search = vectors.param.as_ref().map_or(32, |param| param.ef).max(1);
        let hits =
            self.db
                .search_with_ef(&self.collection_name, &vectors.vector, topk, ef_search)?;
        let should_fetch_fields = output_fields
            .as_ref()
            .map_or(true, |fields| !fields.is_empty());
        if !should_fetch_fields {
            return Ok(hits
                .into_iter()
                .map(|hit| Doc {
                    id: hit.id.to_string(),
                    score: Some(hit.distance),
                    fields: BTreeMap::new(),
                    vectors: BTreeMap::new(),
                })
                .collect());
        }

        let fetched = self.db.fetch_documents(
            &self.collection_name,
            &hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        )?;
        Ok(hits
            .into_iter()
            .zip(fetched)
            .map(|(hit, document)| Doc {
                id: hit.id.to_string(),
                score: Some(hit.distance),
                fields: select_output_fields(&document.fields, &output_fields),
                vectors: BTreeMap::new(),
            })
            .collect())
    }

    pub fn optimize(&mut self) -> std::io::Result<()> {
        self.db.optimize_collection(&self.collection_name)
    }

    pub fn destroy(mut self) -> std::io::Result<()> {
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
        } else if let Ok(value) = value.extract::<i64>() {
            FieldValue::Int64(value)
        } else if let Ok(value) = value.extract::<f64>() {
            FieldValue::Float64(value)
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
fn fields_to_py_dict<'py>(
    py: Python<'py>,
    fields: &BTreeMap<String, FieldValue>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (name, value) in fields {
        match value {
            FieldValue::String(value) => dict.set_item(name, value)?,
            FieldValue::Int64(value) => dict.set_item(name, *value)?,
            FieldValue::Float64(value) => dict.set_item(name, *value)?,
            FieldValue::Bool(value) => dict.set_item(name, *value)?,
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
        "float64" => Ok(DataType::Float64),
        "bool" => Ok(DataType::Bool),
        "vectorfp32" | "vector_fp32" => Ok(DataType::VectorFp32),
        other => Err(PyValueError::new_err(format!(
            "unsupported DataType value: {other}"
        ))),
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
#[pyclass(name = "HnswQueryParam", module = "hannsdb")]
#[derive(Clone)]
struct PyHnswQueryParam {
    inner: HnswQueryParam,
}

#[cfg(feature = "python-binding")]
#[pymethods]
impl PyHnswQueryParam {
    #[new]
    #[pyo3(signature = (ef=32, is_using_refiner=false))]
    fn new(ef: usize, is_using_refiner: bool) -> Self {
        Self {
            inner: HnswQueryParam {
                ef,
                is_using_refiner,
            },
        }
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
                if bound.is_instance_of::<PyHnswIndexParam>() {
                    Some(IndexParam::Hnsw(
                        bound
                            .extract::<PyRef<'_, PyHnswIndexParam>>()?
                            .inner
                            .clone(),
                    ))
                } else if bound.is_instance_of::<PyIVFIndexParam>() {
                    Some(IndexParam::Ivf(
                        bound.extract::<PyRef<'_, PyIVFIndexParam>>()?.inner.clone(),
                    ))
                } else {
                    return Err(PyValueError::new_err(
                        "index_param must be HnswIndexParam or IVFIndexParam",
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
    #[pyo3(signature = (id, vector=None, field_name="dense", fields=None, score=None))]
    fn new(
        id: String,
        vector: Option<Vec<f32>>,
        field_name: &str,
        fields: Option<Bound<'_, PyDict>>,
        score: Option<f32>,
    ) -> PyResult<Self> {
        let mut vectors = BTreeMap::new();
        if let Some(vector) = vector {
            vectors.insert(field_name.to_string(), vector);
        }
        Ok(Self {
            inner: Doc {
                id,
                score,
                fields: fields
                    .as_ref()
                    .map(py_dict_to_fields)
                    .transpose()?
                    .unwrap_or_default(),
                vectors,
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
        vector: Vec<f32>,
        param: Option<Py<PyHnswQueryParam>>,
    ) -> Self {
        Self {
            inner: VectorQuery {
                field_name,
                vector,
                param: param.map(|value| value.borrow(py).inner.clone()),
            },
        }
    }
}

#[cfg(feature = "python-binding")]
#[pyclass(name = "Collection", module = "hannsdb")]
struct PyCollection {
    inner: Option<Collection>,
}

#[cfg(feature = "python-binding")]
#[pyclass(name = "CollectionStats", module = "hannsdb")]
#[derive(Clone)]
struct PyCollectionStats {
    inner: CollectionStats,
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

    #[pyo3(signature = (vectors, output_fields=None, topk=100, filter=None))]
    fn query(
        &self,
        py: Python<'_>,
        vectors: Py<PyVectorQuery>,
        output_fields: Option<Vec<String>>,
        topk: usize,
        filter: Option<String>,
    ) -> PyResult<Vec<Py<PyDoc>>> {
        let borrowed = vectors.borrow(py);
        let empty_output_fields = output_fields
            .as_ref()
            .is_some_and(|fields| fields.is_empty());
        let has_filter = filter
            .as_deref()
            .map(str::trim)
            .is_some_and(|value| !value.is_empty());
        if empty_output_fields && !has_filter {
            let hits = self
                .inner_ref()?
                .query_ids_scores(topk, &borrowed.inner)
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
                            },
                        },
                    )
                })
                .collect();
        }

        let vectors = borrowed.inner.clone();
        let docs = self
            .inner_ref()?
            .query(output_fields, topk, filter.as_deref(), vectors)
            .map_err(io_to_py_err)?;
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
#[pyo3(signature = (path, schema, option=None))]
fn py_create_and_open(
    py: Python<'_>,
    path: String,
    schema: Py<PyCollectionSchema>,
    option: Option<Py<PyCollectionOption>>,
) -> PyResult<PyCollection> {
    let schema = schema.borrow(py).inner.clone();
    let option = option.map(|value| value.borrow(py).inner.clone());
    let collection = create_and_open(path, schema, option).map_err(io_to_py_err)?;
    Ok(PyCollection {
        inner: Some(collection),
    })
}

#[cfg(feature = "python-binding")]
#[pyfunction(name = "open")]
#[pyo3(signature = (path, option=None))]
fn py_open(
    py: Python<'_>,
    path: String,
    option: Option<Py<PyCollectionOption>>,
) -> PyResult<PyCollection> {
    let option = option.map(|value| value.borrow(py).inner.clone());
    let collection = open(path, option).map_err(io_to_py_err)?;
    Ok(PyCollection {
        inner: Some(collection),
    })
}

#[cfg(feature = "python-binding")]
#[pymodule(name = "hannsdb")]
fn python_module(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMetricType>()?;
    module.add_class::<PyQuantizeType>()?;
    module.add_class::<PyLogLevel>()?;
    module.add_class::<PyDataType>()?;
    module.add_class::<PyOptimizeOption>()?;
    module.add_class::<PyCollectionOption>()?;
    module.add_class::<PyHnswIndexParam>()?;
    module.add_class::<PyIVFIndexParam>()?;
    module.add_class::<PyHnswQueryParam>()?;
    module.add_class::<PyFieldSchema>()?;
    module.add_class::<PyVectorSchema>()?;
    module.add_class::<PyCollectionSchema>()?;
    module.add_class::<PyDoc>()?;
    module.add_class::<PyVectorQuery>()?;
    module.add_class::<PyCollectionStats>()?;
    module.add_class::<PyCollection>()?;
    module.add_function(wrap_pyfunction!(py_init, module)?)?;
    module.add_function(wrap_pyfunction!(py_create_and_open, module)?)?;
    module.add_function(wrap_pyfunction!(py_open, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::io::ErrorKind;

    use crate::{
        create_and_open, init, open, CollectionOption, CollectionSchema, DataType, Doc,
        FieldSchema, HnswIndexParam, HnswQueryParam, IndexParam, LogLevel, MetricType,
        OptimizeOption, QuantizeType, VectorQuery, VectorSchema,
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
        };
        let _query = VectorQuery {
            field_name: "dense".to_string(),
            vector: vec![0.0, 0.1, 0.2, 0.3],
            param: Some(HnswQueryParam {
                ef: 32,
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
                    vector: vec![0.1, -0.1],
                    param: Some(HnswQueryParam {
                        ef: 32,
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
                    vector: vec![0.1, -0.1],
                    param: Some(HnswQueryParam {
                        ef: 32,
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
                    vector: vec![0.0, 0.0],
                    param: Some(HnswQueryParam {
                        ef: 32,
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
    fn create_and_open_rejects_primary_ivf_index() {
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

        let result = create_and_open(db_path, schema, Some(CollectionOption::default()));
        assert!(result.is_err(), "primary IVF must be rejected");
        assert_eq!(
            result.err().expect("result is err").kind(),
            ErrorKind::InvalidInput
        );
    }
}
